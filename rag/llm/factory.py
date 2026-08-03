"""统一 LLM Provider、轻重任务路由、重试与自动降级。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Lock, local
from typing import Any, Dict, Optional

from rag.reliability import (
    CircuitBreaker,
    CircuitOpenError,
    bounded_timeout,
    budget_attributes,
    remaining_budget_seconds,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str
    base_url: Optional[str]
    model: str


def _provider_config(provider: str, settings) -> ProviderConfig:
    name = (provider or "deepseek").lower()
    attributes = {
        "deepseek": (
            "deepseek_api_key",
            "deepseek_base_url",
            "deepseek_model",
        ),
        "dashscope": (
            "dashscope_api_key",
            "dashscope_base_url",
            "dashscope_model",
        ),
    }
    if name not in attributes:
        raise ValueError(f"不支持的 LLM Provider: {name}，可选：{', '.join(sorted(attributes))}")
    api_key_attr, base_url_attr, model_attr = attributes[name]
    api_key = getattr(settings, api_key_attr, None)
    base_url = getattr(settings, base_url_attr, None)
    model = getattr(settings, model_attr, None)
    if not api_key:
        env_names = {
            "deepseek": "DEEPSEEK_API_KEY",
            "dashscope": "DASHSCOPE_API_KEY",
        }
        raise ValueError(
            f"Provider '{name}' 缺少 API Key，请配置 {env_names[name]}"
        )
    return ProviderConfig(name, api_key, base_url, model)


def _build_client(client_type: str, config: ProviderConfig, settings):
    if (client_type or "openai").lower() == "langchain":
        from rag.chains.rag_chain import LangChainLLMClient

        return LangChainLLMClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )

    from rag.chains.rag_chain import LLMClient

    return LLMClient(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        timeout=settings.llm_timeout_seconds,
    )


class ResilientLLMClient:
    VERIFIER_STAGES = {"citation_repair", "citation_verification"}
    FAST_STAGES = {
        "agent_intent",
        "citation_repair",
        "citation_verification",
        "query_rewrite",
        "subquestion_evidence_judgment",
        "subquestion_planning",
    }
    """在不侵入 RAGChain 的前提下提供任务路由、重试和主备降级。"""

    def __init__(
        self,
        primary,
        primary_info: ProviderConfig,
        fast=None,
        fast_info: Optional[ProviderConfig] = None,
        verifier=None,
        verifier_info: Optional[ProviderConfig] = None,
        fallback=None,
        fallback_info: Optional[ProviderConfig] = None,
        max_retries: int = 2,
        retry_base_seconds: float = 0.5,
        default_timeout_seconds: float = 20.0,
        circuit_failure_threshold: int = 3,
        circuit_recovery_seconds: float = 30.0,
        fast_stage_max_retries: int = 0,
        generation_max_retries: int = 1,
    ):
        self.primary = primary
        self.primary_info = primary_info
        self.fast = fast or primary
        self.fast_info = fast_info or primary_info
        self.verifier = verifier or self.fast
        self.verifier_info = verifier_info or self.fast_info
        self.fallback = fallback
        self.fallback_info = fallback_info
        self.max_retries = max(0, max_retries)
        self.retry_base_seconds = max(0.0, retry_base_seconds)
        self.default_timeout_seconds = max(0.001, float(default_timeout_seconds))
        self.fast_stage_max_retries = max(0, int(fast_stage_max_retries))
        self.generation_max_retries = max(0, int(generation_max_retries))
        self._lock = Lock()
        self._usage_local = local()
        self._metrics = {"requests": 0, "retries": 0, "fallbacks": 0, "failures": 0}
        providers = {
            primary_info.name,
            self.fast_info.name,
            self.verifier_info.name,
        }
        if self.fallback_info:
            providers.add(self.fallback_info.name)
        self._breakers = {
            name: CircuitBreaker(circuit_failure_threshold, circuit_recovery_seconds)
            for name in providers
        }

    def _record(self, key: str) -> None:
        with self._lock:
            self._metrics[key] += 1

    def reset_request_usage(self) -> None:
        self._usage_local.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._usage_local.events = []

    def _capture_usage(self, client) -> None:
        if not hasattr(client, "get_last_usage"):
            return
        current = getattr(self._usage_local, "usage", None)
        if current is None:
            self.reset_request_usage()
            current = self._usage_local.usage
        usage = client.get_last_usage()
        for key in current:
            current[key] += int(usage.get(key, 0) or 0)

    def request_usage(self) -> Dict[str, int]:
        return dict(getattr(self._usage_local, "usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}))

    def request_events(self) -> list[dict]:
        return [dict(item) for item in getattr(self._usage_local, "events", [])]

    def _append_event(self, event: dict) -> None:
        events = getattr(self._usage_local, "events", None)
        if events is None:
            self._usage_local.events = []
            events = self._usage_local.events
        events.append(event)

    def metrics(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._metrics)

    def provider_summary(self) -> Dict[str, Any]:
        return {
            "primary": {"provider": self.primary_info.name, "model": self.primary_info.model},
            "fast": {"provider": self.fast_info.name, "model": self.fast_info.model},
            "verifier": {
                "provider": self.verifier_info.name,
                "model": self.verifier_info.model,
            },
            "fallback": (
                {"provider": self.fallback_info.name, "model": self.fallback_info.model}
                if self.fallback_info else None
            ),
            "metrics": self.metrics(),
            "circuits": {
                provider: {
                    "state": snapshot.state,
                    "failure_count": snapshot.failure_count,
                    "opened_for_ms": snapshot.opened_for_ms,
                }
                for provider, breaker in self._breakers.items()
                for snapshot in [breaker.snapshot()]
            },
        }

    def _route(self, max_tokens: Optional[int], stage: str = "generation"):
        # 查询改写、意图识别等短输出使用 fast provider；正式回答使用 primary。
        if stage in self.VERIFIER_STAGES:
            return self.verifier, self.verifier_info
        if stage in self.FAST_STAGES or (
            max_tokens is not None and max_tokens <= 512
        ):
            return self.fast, self.fast_info
        return self.primary, self.primary_info

    def _max_retries_for_stage(self, stage: str) -> int:
        if stage in self.FAST_STAGES:
            return min(self.max_retries, self.fast_stage_max_retries)
        if stage == "generation":
            return min(self.max_retries, self.generation_max_retries)
        return self.max_retries

    def _call_with_retry(
        self, client, info, method: str, *, stage: str, timeout: float, **kwargs
    ):
        started = time.perf_counter()
        breaker = self._breakers[info.name]
        try:
            circuit_before = breaker.before_call()
        except CircuitOpenError as exc:
            self._append_event({
                "stage": stage,
                "provider": info.name,
                "model": info.model,
                "retry_count": 0,
                "timeout_ms": int(timeout * 1000),
                "queue_time_ms": 0,
                "upstream_request_id": None,
                "circuit_state": "open",
                "error_type": type(exc).__name__,
                "duration_ms": 0,
                **budget_attributes(),
            })
            raise
        max_retries = self._max_retries_for_stage(stage)
        for attempt in range(max_retries + 1):
            try:
                stage_remaining = timeout - (time.perf_counter() - started)
                if stage_remaining <= 0.001:
                    raise TimeoutError(f"{stage} stage budget exhausted")
                effective_timeout = bounded_timeout(stage_remaining)
                result = getattr(client, method)(
                    **kwargs, timeout=effective_timeout
                )
                self._capture_usage(client)
                breaker.record_success()
                metadata = (
                    client.get_last_metadata()
                    if hasattr(client, "get_last_metadata")
                    else {}
                ) or {}
                self._append_event({
                    "stage": stage,
                    "provider": info.name,
                    "model": info.model,
                    "retry_count": attempt,
                    "timeout_ms": int(timeout * 1000),
                    "effective_timeout_ms": int(effective_timeout * 1000),
                    "queue_time_ms": 0,
                    "upstream_request_id": metadata.get("upstream_request_id"),
                    "circuit_state": circuit_before.state,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    **budget_attributes(),
                })
                return result
            except Exception as exc:
                if attempt >= max_retries:
                    circuit_after = breaker.record_failure()
                    self._append_event({
                        "stage": stage,
                        "provider": info.name,
                        "model": info.model,
                        "retry_count": attempt,
                        "timeout_ms": int(timeout * 1000),
                        "queue_time_ms": 0,
                        "upstream_request_id": self._request_id_from_error(exc),
                        "circuit_state": circuit_after.state,
                        "error_type": type(exc).__name__,
                        "duration_ms": int((time.perf_counter() - started) * 1000),
                        **budget_attributes(),
                    })
                    raise
                self._record("retries")
                delay = self.retry_base_seconds * (2 ** attempt)
                logger.warning(
                    "llm_retry provider=%s model=%s attempt=%s error=%s",
                    info.name, info.model, attempt + 1, type(exc).__name__,
                )
                remaining = remaining_budget_seconds()
                stage_remaining = timeout - (time.perf_counter() - started)
                if (
                    delay
                    and stage_remaining > delay + 0.001
                    and (remaining is None or remaining > delay + 0.001)
                ):
                    time.sleep(delay)

    @staticmethod
    def _request_id_from_error(exc: Exception) -> Optional[str]:
        request_id = getattr(exc, "request_id", None)
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if request_id is None and headers:
            request_id = headers.get("x-request-id") or headers.get("request-id")
        return str(request_id) if request_id else None

    def generate(
        self, system_prompt: str, user_prompt: str, temperature=0.3,
        max_tokens=None, response_format=None, thinking=None,
        stage: str = "generation", timeout: Optional[float] = None,
    ) -> str:
        self._record("requests")
        client, info = self._route(max_tokens, stage)
        kwargs = dict(system_prompt=system_prompt, user_prompt=user_prompt,
                      temperature=temperature, max_tokens=max_tokens,
                      response_format=response_format, thinking=thinking)
        try:
            return self._call_with_retry(
                client, info, "generate", stage=stage,
                timeout=timeout or self.default_timeout_seconds, **kwargs
            )
        except Exception as primary_error:
            if (
                stage in self.FAST_STAGES
                or not self.fallback
                or client is self.fallback
            ):
                self._record("failures")
                raise
            self._record("fallbacks")
            logger.error(
                "llm_fallback from_provider=%s to_provider=%s error=%s",
                info.name, self.fallback_info.name, type(primary_error).__name__,
            )
            try:
                return self._call_with_retry(
                    self.fallback, self.fallback_info, "generate", stage=stage,
                    timeout=timeout or self.default_timeout_seconds, **kwargs
                )
            except Exception:
                self._record("failures")
                raise

    def generate_stream(
        self, system_prompt: str, user_prompt: str, temperature=0.3,
        max_tokens=None, thinking=None, stage: str = "generation",
        timeout: Optional[float] = None,
    ):
        self._record("requests")
        client, info = self._route(max_tokens, stage)
        kwargs = dict(system_prompt=system_prompt, user_prompt=user_prompt,
                      temperature=temperature, max_tokens=max_tokens,
                      thinking=thinking)
        emitted = False
        breaker = self._breakers[info.name]
        circuit_before = breaker.before_call()
        started = time.perf_counter()
        stage_timeout = timeout or self.default_timeout_seconds
        try:
            max_retries = self._max_retries_for_stage(stage)
            for attempt in range(max_retries + 1):
                try:
                    stage_remaining = stage_timeout - (
                        time.perf_counter() - started
                    )
                    if stage_remaining <= 0.001:
                        raise TimeoutError(
                            f"{stage} stage budget exhausted"
                        )
                    effective_timeout = bounded_timeout(stage_remaining)
                    for chunk in client.generate_stream(
                        **kwargs, timeout=effective_timeout
                    ):
                        emitted = True
                        yield chunk
                    self._capture_usage(client)
                    breaker.record_success()
                    metadata = (
                        client.get_last_metadata()
                        if hasattr(client, "get_last_metadata")
                        else {}
                    ) or {}
                    self._append_event({
                        "stage": stage, "provider": info.name, "model": info.model,
                        "retry_count": attempt,
                        "timeout_ms": int(stage_timeout * 1000),
                        "effective_timeout_ms": int(effective_timeout * 1000),
                        "queue_time_ms": 0,
                        "upstream_request_id": metadata.get("upstream_request_id"),
                        "circuit_state": circuit_before.state,
                        "duration_ms": int((time.perf_counter() - started) * 1000),
                        **budget_attributes(),
                    })
                    break
                except Exception as exc:
                    if emitted or attempt >= max_retries:
                        breaker.record_failure()
                        raise
                    self._record("retries")
                    delay = self.retry_base_seconds * (2 ** attempt)
                    logger.warning(
                        "llm_stream_retry provider=%s model=%s attempt=%s error=%s",
                        info.name, info.model, attempt + 1, type(exc).__name__,
                    )
                    if delay:
                        remaining = remaining_budget_seconds()
                        stage_remaining = stage_timeout - (
                            time.perf_counter() - started
                        )
                        if (
                            stage_remaining > delay + 0.001
                            and (remaining is None or remaining > delay + 0.001)
                        ):
                            time.sleep(delay)
        except Exception as primary_error:
            # 已经向客户端发送 token 后切换模型会造成重复文本，因此只允许首 token 前降级。
            if (
                emitted
                or stage in self.FAST_STAGES
                or not self.fallback
                or client is self.fallback
            ):
                self._record("failures")
                raise
            self._record("fallbacks")
            logger.error(
                "llm_stream_fallback from_provider=%s to_provider=%s error=%s",
                info.name, self.fallback_info.name, type(primary_error).__name__,
            )
            try:
                fallback_emitted = False
                for attempt in range(max_retries + 1):
                    try:
                        fallback_timeout = bounded_timeout(timeout or self.default_timeout_seconds)
                        for chunk in self.fallback.generate_stream(
                            **kwargs, timeout=fallback_timeout
                        ):
                            fallback_emitted = True
                            yield chunk
                        self._capture_usage(self.fallback)
                        break
                    except Exception as exc:
                        if fallback_emitted or attempt >= max_retries:
                            raise
                        self._record("retries")
                        delay = self.retry_base_seconds * (2 ** attempt)
                        logger.warning(
                            "llm_fallback_stream_retry provider=%s model=%s attempt=%s error=%s",
                            self.fallback_info.name, self.fallback_info.model,
                            attempt + 1, type(exc).__name__,
                        )
                        if delay:
                            time.sleep(delay)
            except Exception:
                self._record("failures")
                raise


def create_llm_client(client_type: str, provider: str, settings):
    primary_name = settings.llm_primary_provider or provider or settings.llm_default_provider
    fast_name = settings.llm_fast_provider or primary_name
    verifier_name = settings.llm_verifier_provider or fast_name
    fallback_name = settings.llm_fallback_provider

    def with_model(info: ProviderConfig, override: Optional[str]) -> ProviderConfig:
        if not override or override == info.model:
            return info
        return ProviderConfig(
            info.name, info.api_key, info.base_url, override
        )

    primary_info = _provider_config(primary_name, settings)
    primary = _build_client(client_type, primary_info, settings)

    fast_info = with_model(
        _provider_config(fast_name, settings), settings.llm_fast_model
    )
    if fast_info == primary_info:
        fast, fast_info = primary, primary_info
    else:
        fast = _build_client(client_type, fast_info, settings)

    verifier_info = with_model(
        _provider_config(verifier_name, settings),
        settings.llm_verifier_model,
    )
    if verifier_info == primary_info:
        verifier = primary
    elif verifier_info == fast_info:
        verifier = fast
    else:
        verifier = _build_client(client_type, verifier_info, settings)

    fallback = fallback_info = None
    if fallback_name:
        candidate_info = _provider_config(fallback_name, settings)
        if candidate_info == primary_info:
            fallback, fallback_info = primary, primary_info
        elif candidate_info == fast_info:
            fallback, fallback_info = fast, fast_info
        else:
            fallback_info = candidate_info
            fallback = _build_client(client_type, fallback_info, settings)

    return ResilientLLMClient(
        primary=primary,
        primary_info=primary_info,
        fast=fast,
        fast_info=fast_info,
        verifier=verifier,
        verifier_info=verifier_info,
        fallback=fallback,
        fallback_info=fallback_info,
        max_retries=settings.llm_max_retries,
        retry_base_seconds=settings.llm_retry_base_seconds,
        default_timeout_seconds=settings.llm_timeout_seconds,
        circuit_failure_threshold=settings.circuit_breaker_failure_threshold,
        circuit_recovery_seconds=settings.circuit_breaker_recovery_seconds,
        fast_stage_max_retries=settings.llm_fast_stage_max_retries,
        generation_max_retries=settings.llm_generation_max_retries,
    )


def get_provider_info(settings) -> dict:
    primary = settings.llm_primary_provider or settings.llm_default_provider or "openai"
    config = _provider_config(primary, settings)
    return {"provider": config.name, "model": config.model, "base_url": config.base_url}
