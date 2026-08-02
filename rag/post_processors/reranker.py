"""检索结果重排器。"""

from __future__ import annotations

import copy
import math
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional

from models.document import RetrievedChunk
from rag.reliability import (
    CircuitBreaker,
    CircuitOpenError,
    bounded_timeout,
    budget_attributes,
    remaining_budget_seconds,
)


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_n: int = 5,
        timeout: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        pass


class NoOpReranker(BaseReranker):
    """保留混合检索分数，仅截取前 N 条。"""

    def rerank(self, query, candidates, top_n=5, timeout=None):
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)[:top_n]
        for index, item in enumerate(ranked, 1):
            item.rank = index
        return ranked


class ScoreThresholdFilter(BaseReranker):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def rerank(self, query, candidates, top_n=5, timeout=None):
        filtered = [item for item in candidates if item.score >= self.threshold]
        return NoOpReranker().rerank(query, filtered, top_n, timeout=timeout)


class Qwen3Reranker(BaseReranker):
    """通过阿里云百炼 ``qwen3-rerank`` 对召回候选进行语义重排。

    qwen3-rerank 使用独立于 Chat/Embedding 的 OpenAI-compatible
    ``/reranks`` 接口。服务端只需返回候选文档在请求数组中的索引，
    本类始终使用该索引映射本地 Chunk，避免依赖服务端回传原文。

    调用失败时保留可用性，自动降级到 :class:`NoOpReranker`。最近一次
    调用元数据存在线程本地存储中，供 RAG Trace 读取且不会在并发请求间串扰。
    """

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-api/v1"
    DEFAULT_INSTRUCT = (
        "Given a web search query, retrieve relevant passages that answer the query."
    )
    MAX_DOCUMENTS = 500

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "qwen3-rerank",
        timeout: float = 20.0,
        max_retries: int = 2,
        retry_base_seconds: float = 0.5,
        instruct: Optional[str] = DEFAULT_INSTRUCT,
        max_candidates: int = MAX_DOCUMENTS,
        client: Any = None,
        circuit_failure_threshold: int = 3,
        circuit_recovery_seconds: float = 30.0,
    ):
        if max_candidates < 1 or max_candidates > self.MAX_DOCUMENTS:
            raise ValueError(
                f"qwen3-rerank max_candidates 必须在 1~{self.MAX_DOCUMENTS} 之间"
            )

        self.model = model_name
        self.base_url = base_url or os.getenv(
            "DASHSCOPE_RERANK_BASE_URL", self.DEFAULT_BASE_URL
        )
        self.max_retries = max(0, int(max_retries))
        self.retry_base_seconds = max(0.0, float(retry_base_seconds))
        self.instruct = instruct
        self.max_candidates = int(max_candidates)
        self.timeout = max(0.001, float(timeout))
        self._breaker = CircuitBreaker(
            circuit_failure_threshold, circuit_recovery_seconds
        )
        self._thread_local = threading.local()

        if client is not None:
            self.client = client
            return

        resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not resolved_api_key:
            raise ValueError("请配置 DASHSCOPE_API_KEY")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("请安装 openai: pip install openai") from exc

        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=self.base_url,
            timeout=timeout,
            # 此处自行重试并记录降级信息，关闭 SDK 隐式重试以免重复放大等待时间。
            max_retries=0,
        )

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_n: int = 5,
        timeout: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        candidate_count = min(len(candidates), self.max_candidates)
        submitted = list(candidates[:candidate_count])
        requested_top_n = min(max(0, int(top_n)), candidate_count)

        if not submitted or requested_top_n == 0:
            self._set_last_metadata(
                fallback=False,
                error=None,
                request_id=None,
                usage={},
                candidate_count=candidate_count,
                output_count=0,
                retry_count=0,
                timeout_ms=int((timeout or self.timeout) * 1000),
                circuit_state=self._breaker.snapshot().state,
            )
            return []

        body: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": [candidate.content for candidate in submitted],
            "top_n": requested_top_n,
        }
        if self.instruct:
            body["instruct"] = self.instruct

        last_error: Optional[Exception] = None
        request_started = time.perf_counter()
        retry_count = 0
        try:
            circuit_before = self._breaker.before_call()
        except CircuitOpenError as exc:
            fallback_result = NoOpReranker().rerank(
                query, candidates, top_n, timeout=timeout
            )
            self._set_last_metadata(
                fallback=True, error=self._error_message(exc), request_id=None,
                usage={}, candidate_count=candidate_count,
                output_count=len(fallback_result), retry_count=0,
                timeout_ms=int((timeout or self.timeout) * 1000),
                circuit_state="open",
            )
            return fallback_result
        for attempt in range(self.max_retries + 1):
            try:
                stage_timeout = timeout or self.timeout
                stage_remaining = stage_timeout - (
                    time.perf_counter() - request_started
                )
                if stage_remaining <= 0.001:
                    raise TimeoutError("reranker stage budget exhausted")
                effective_timeout = bounded_timeout(stage_remaining)
                client = (
                    self.client.with_options(timeout=effective_timeout)
                    if hasattr(self.client, "with_options")
                    else self.client
                )
                response = client.post(
                    "/reranks",
                    body=body,
                    cast_to=object,
                )
                payload = self._as_mapping(response)
                ranked = self._map_results(payload, submitted, requested_top_n)
                self._breaker.record_success()
                self._set_last_metadata(
                    fallback=False,
                    error=None,
                    request_id=payload.get("id") or payload.get("request_id"),
                    usage=self._usage_from_payload(payload),
                    candidate_count=candidate_count,
                    output_count=len(ranked),
                    retry_count=retry_count,
                    timeout_ms=int((timeout or self.timeout) * 1000),
                    circuit_state=circuit_before.state,
                    duration_ms=int((time.perf_counter() - request_started) * 1000),
                )
                return ranked
            except Exception as exc:  # 网络、限流和异常响应统一进入显式重试/降级
                last_error = exc
                if attempt < self.max_retries:
                    retry_count += 1
                    delay = self.retry_base_seconds * (2 ** attempt)
                    remaining = remaining_budget_seconds()
                    stage_remaining = (timeout or self.timeout) - (
                        time.perf_counter() - request_started
                    )
                    if (
                        stage_remaining > delay + 0.001
                        and (remaining is None or remaining > delay + 0.001)
                    ):
                        time.sleep(delay)

        circuit_after = self._breaker.record_failure()
        fallback_result = NoOpReranker().rerank(
            query, candidates, top_n, timeout=timeout
        )
        self._set_last_metadata(
            fallback=True,
            error=self._error_message(last_error),
            request_id=self._request_id_from_error(last_error),
            usage={},
            candidate_count=candidate_count,
            output_count=len(fallback_result),
            retry_count=retry_count,
            timeout_ms=int((timeout or self.timeout) * 1000),
            circuit_state=circuit_after.state,
            duration_ms=int((time.perf_counter() - request_started) * 1000),
        )
        return fallback_result

    def get_last_metadata(self) -> Dict[str, Any]:
        """返回当前线程最近一次重排调用的观测信息。"""
        value = getattr(self._thread_local, "last_metadata", None)
        return copy.deepcopy(value) if value else {}

    @property
    def last_metadata(self) -> Dict[str, Any]:
        """``get_last_metadata`` 的属性形式，便于追踪组件读取。"""
        return self.get_last_metadata()

    def _map_results(
        self,
        payload: Mapping[str, Any],
        submitted: List[RetrievedChunk],
        top_n: int,
    ) -> List[RetrievedChunk]:
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("qwen3-rerank 响应缺少 results 数组")

        parsed = []
        seen_indices = set()
        for raw_item in raw_results:
            try:
                item = self._as_mapping(raw_item)
            except (TypeError, ValueError):
                continue

            index = item.get("index")
            # bool 是 int 的子类，但不能作为文档索引接受。
            if isinstance(index, bool) or not isinstance(index, int):
                continue
            if index < 0 or index >= len(submitted) or index in seen_indices:
                continue

            try:
                score = float(item.get("relevance_score"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                continue

            seen_indices.add(index)
            parsed.append((index, score))

        if not parsed:
            raise ValueError("qwen3-rerank 响应中没有可映射的有效结果")

        # 官方结果应已降序；本地再排序可防止代理层或测试桩返回乱序数据。
        parsed.sort(key=lambda pair: pair[1], reverse=True)
        ranked = []
        for rank, (index, score) in enumerate(parsed[:top_n], 1):
            source = submitted[index]
            chunk = source.model_copy(deep=True)
            chunk.metadata = dict(chunk.metadata or {})
            chunk.metadata.setdefault("rrf_score", float(source.score))
            chunk.metadata["rerank_provider"] = "dashscope"
            chunk.metadata["rerank_model"] = self.model
            chunk.score = score
            chunk.rank = rank
            ranked.append(chunk)
        return ranked

    def _set_last_metadata(
        self,
        *,
        fallback: bool,
        error: Optional[str],
        request_id: Any,
        usage: Mapping[str, Any],
        candidate_count: int,
        output_count: int,
        retry_count: int = 0,
        timeout_ms: Optional[int] = None,
        circuit_state: str = "closed",
        duration_ms: int = 0,
    ) -> None:
        self._thread_local.last_metadata = {
            "provider": "dashscope",
            "model": self.model,
            "fallback": fallback,
            "error": error,
            "request_id": str(request_id) if request_id is not None else None,
            "usage": dict(usage),
            "candidate_count": int(candidate_count),
            "output_count": int(output_count),
            "retry_count": int(retry_count),
            "timeout_ms": timeout_ms,
            "queue_time_ms": 0,
            "upstream_request_id": (
                str(request_id) if request_id is not None else None
            ),
            "circuit_state": circuit_state,
            "duration_ms": max(0, int(duration_ms)),
            **budget_attributes(),
        }

    @staticmethod
    def _usage_from_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
        usage = payload.get("usage")
        if isinstance(usage, Mapping):
            return dict(usage)
        if hasattr(usage, "model_dump"):
            dumped = usage.model_dump()
            return dict(dumped) if isinstance(dumped, Mapping) else {}
        return {}

    @staticmethod
    def _as_mapping(value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        if hasattr(value, "model_dump"):
            dumped = value.model_dump()
            if isinstance(dumped, Mapping):
                return dumped
        raise TypeError("qwen3-rerank 响应不是可解析的对象")

    @staticmethod
    def _error_message(error: Optional[Exception]) -> Optional[str]:
        if error is None:
            return None
        return f"{type(error).__name__}: {error}"

    @staticmethod
    def _request_id_from_error(error: Optional[Exception]) -> Optional[str]:
        if error is None:
            return None
        request_id = getattr(error, "request_id", None)
        if request_id is None:
            response = getattr(error, "response", None)
            headers = getattr(response, "headers", None)
            if headers:
                request_id = headers.get("x-request-id") or headers.get("request-id")
        return str(request_id) if request_id is not None else None
