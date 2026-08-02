"""阿里云百炼文本向量客户端。"""

from __future__ import annotations

import math
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from rag.reliability import (
    CircuitBreaker,
    bounded_timeout,
    budget_attributes,
    remaining_budget_seconds,
)


class BaseEmbeddingClient(ABC):
    @abstractmethod
    def embed(
        self, texts: List[str], timeout: Optional[float] = None
    ) -> List[List[float]]:
        """把文本批量编码为向量。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度。"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称。"""

    @staticmethod
    def _normalize(vectors: List[List[float]]) -> List[List[float]]:
        normalized = []
        for vector in vectors:
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            normalized.append([value / norm for value in vector])
        return normalized


class DashScopeEmbeddingClient(BaseEmbeddingClient):
    """通过 OpenAI-compatible API 调用百炼 text-embedding-v4。"""

    SUPPORTED_DIMENSIONS = {64, 128, 256, 512, 768, 1024, 1536, 2048}

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "text-embedding-v4",
        dimension: int = 1024,
        batch_size: int = 10,
        max_retries: int = 2,
        timeout: float = 60.0,
        circuit_failure_threshold: int = 3,
        circuit_recovery_seconds: float = 30.0,
    ):
        if batch_size < 1 or batch_size > 10:
            raise ValueError("text-embedding-v4 单批次最多支持 10 条文本")
        if dimension not in self.SUPPORTED_DIMENSIONS:
            raise ValueError(
                f"text-embedding-v4 不支持 {dimension} 维，可选：{sorted(self.SUPPORTED_DIMENSIONS)}"
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("请安装 openai: pip install openai") from exc

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请配置 DASHSCOPE_API_KEY")

        self.base_url = base_url or os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model_name
        self._dimension = dimension
        self.batch_size = batch_size
        self.max_retries = max(0, max_retries)
        self.timeout = max(0.001, float(timeout))
        self._breaker = CircuitBreaker(
            circuit_failure_threshold, circuit_recovery_seconds
        )
        self._usage_local = threading.local()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self.model

    def reset_request_usage(self) -> None:
        """Reset query-embedding usage for the current request thread."""
        self._usage_local.usage = {"prompt_tokens": 0, "total_tokens": 0}
        self._usage_local.events = []

    def request_usage(self) -> dict:
        return dict(
            getattr(
                self._usage_local,
                "usage",
                {"prompt_tokens": 0, "total_tokens": 0},
            )
        )

    def request_events(self) -> list[dict]:
        return [dict(item) for item in getattr(self._usage_local, "events", [])]

    def get_last_metadata(self) -> dict:
        return dict(getattr(self._usage_local, "last_metadata", {}))

    def _capture_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        current = getattr(self._usage_local, "usage", None)
        if current is None:
            self.reset_request_usage()
            current = self._usage_local.usage
        prompt_tokens = int(
            getattr(usage, "prompt_tokens", None)
            or getattr(usage, "input_tokens", None)
            or getattr(usage, "total_tokens", 0)
            or 0
        )
        total_tokens = int(getattr(usage, "total_tokens", 0) or prompt_tokens)
        current["prompt_tokens"] += prompt_tokens
        current["total_tokens"] += total_tokens

    def embed(
        self, texts: List[str], timeout: Optional[float] = None
    ) -> List[List[float]]:
        if not texts:
            return []

        safe_texts = [text if text.strip() else " " for text in texts]
        vectors: List[List[float]] = []
        request_started = time.perf_counter()
        retry_count = 0
        upstream_ids: List[str] = []
        breaker = getattr(self, "_breaker", None)
        if breaker is None:
            breaker = CircuitBreaker()
            self._breaker = breaker
        configured_timeout = getattr(self, "timeout", 60.0)
        if not hasattr(self, "_usage_local"):
            self._usage_local = threading.local()
        circuit_before = breaker.before_call()
        for offset in range(0, len(safe_texts), self.batch_size):
            batch = safe_texts[offset:offset + self.batch_size]
            for attempt in range(self.max_retries + 1):
                try:
                    stage_remaining = (
                        timeout or configured_timeout
                    ) - (time.perf_counter() - request_started)
                    if stage_remaining <= 0.001:
                        raise TimeoutError("embedding stage budget exhausted")
                    effective_timeout = bounded_timeout(
                        stage_remaining
                    )
                    client = (
                        self.client.with_options(timeout=effective_timeout)
                        if hasattr(self.client, "with_options")
                        else self.client
                    )
                    response = client.embeddings.create(
                        model=self.model,
                        input=batch,
                        dimensions=self._dimension,
                    )
                    self._capture_usage(response)
                    response_id = getattr(response, "id", None)
                    if response_id:
                        upstream_ids.append(str(response_id))
                    # OpenAI-compatible 响应带有原输入 index；显式排序避免批量回填缓存时错配。
                    items = sorted(
                        response.data,
                        key=lambda item: int(getattr(item, "index", 0)),
                    )
                    if len(items) != len(batch):
                        raise RuntimeError(
                            f"Embedding 返回数量不一致：请求 {len(batch)}，返回 {len(items)}"
                        )
                    batch_vectors = [item.embedding for item in items]
                    if any(len(vector) != self._dimension for vector in batch_vectors):
                        raise RuntimeError("Embedding 返回维度与配置不一致")
                    vectors.extend(batch_vectors)
                    break
                except Exception:
                    if attempt >= self.max_retries:
                        circuit_after = breaker.record_failure()
                        metadata = {
                            "provider": "dashscope",
                            "model": self.model,
                            "retry_count": retry_count,
                            "timeout_ms": int(
                                (timeout or configured_timeout) * 1000
                            ),
                            "queue_time_ms": 0,
                            "upstream_request_ids": upstream_ids,
                            "circuit_state": circuit_after.state,
                            "duration_ms": int(
                                (time.perf_counter() - request_started) * 1000
                            ),
                            **budget_attributes(),
                        }
                        self._usage_local.last_metadata = metadata
                        self._usage_local.events = [metadata]
                        raise
                    retry_count += 1
                    delay = 0.5 * (2 ** attempt)
                    remaining = remaining_budget_seconds()
                    stage_remaining = (
                        timeout or configured_timeout
                    ) - (time.perf_counter() - request_started)
                    if (
                        stage_remaining > delay + 0.001
                        and (remaining is None or remaining > delay + 0.001)
                    ):
                        time.sleep(delay)

        breaker.record_success()
        metadata = {
            "provider": "dashscope",
            "model": self.model,
            "retry_count": retry_count,
            "timeout_ms": int((timeout or configured_timeout) * 1000),
            "queue_time_ms": 0,
            "upstream_request_ids": upstream_ids,
            "circuit_state": circuit_before.state,
            "duration_ms": int((time.perf_counter() - request_started) * 1000),
            **budget_attributes(),
        }
        self._usage_local.last_metadata = metadata
        events = getattr(self._usage_local, "events", None)
        if events is None:
            self._usage_local.events = []
            events = self._usage_local.events
        events.append(metadata)
        return self._normalize(vectors)


class EmbeddingFactory:
    @staticmethod
    def create(provider: str, **kwargs) -> BaseEmbeddingClient:
        if provider.lower() not in {"dashscope", "aliyun", "qwen"}:
            raise ValueError("本项目仅支持阿里云百炼 Embedding")
        return DashScopeEmbeddingClient(**kwargs)
