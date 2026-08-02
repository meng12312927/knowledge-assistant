"""外部依赖调用的请求预算、阶段超时和进程内熔断器。"""

from __future__ import annotations

import contextvars
import threading
import time
from dataclasses import dataclass
from typing import Optional


class RequestBudgetExceeded(TimeoutError):
    """整条问答请求的剩余时间已经耗尽。"""


class CircuitOpenError(RuntimeError):
    """上游熔断器处于打开状态。"""


_deadline: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar(
    "rag_request_deadline", default=None
)


def start_request_budget(total_seconds: float):
    deadline = time.monotonic() + max(0.001, float(total_seconds))
    return _deadline.set(deadline)


def clear_request_budget(token) -> None:
    # StreamingResponse 的同步生成器可能在不同 worker context 中推进，
    # 因此不能依赖 token 必须在创建它的 Context 中 reset。
    _deadline.set(None)


def remaining_budget_seconds() -> Optional[float]:
    deadline = _deadline.get()
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def bounded_timeout(stage_timeout_seconds: float) -> float:
    """取阶段预算和请求剩余预算的较小值，并在耗尽时快速失败。"""
    stage_timeout = max(0.001, float(stage_timeout_seconds))
    remaining = remaining_budget_seconds()
    if remaining is None:
        return stage_timeout
    if remaining <= 0.001:
        raise RequestBudgetExceeded("total request budget exhausted")
    return max(0.001, min(stage_timeout, remaining))


def budget_attributes() -> dict:
    remaining = remaining_budget_seconds()
    return {
        "request_budget_remaining_ms": (
            None if remaining is None else max(0, int(remaining * 1000))
        )
    }


@dataclass
class CircuitSnapshot:
    state: str
    failure_count: int
    opened_for_ms: int = 0


class CircuitBreaker:
    """线程安全的 closed/open/half-open 熔断器。"""

    def __init__(self, failure_threshold: int = 3, recovery_seconds: float = 30.0):
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_seconds = max(0.001, float(recovery_seconds))
        self._lock = threading.Lock()
        self._failure_count = 0
        self._opened_at: Optional[float] = None
        self._half_open_inflight = False

    def before_call(self) -> CircuitSnapshot:
        now = time.monotonic()
        with self._lock:
            if self._opened_at is None:
                return CircuitSnapshot("closed", self._failure_count)
            elapsed = now - self._opened_at
            if elapsed < self.recovery_seconds:
                raise CircuitOpenError(
                    f"circuit open; retry after {self.recovery_seconds - elapsed:.2f}s"
                )
            if self._half_open_inflight:
                raise CircuitOpenError("circuit half-open probe already in progress")
            self._half_open_inflight = True
            return CircuitSnapshot(
                "half_open", self._failure_count, int(elapsed * 1000)
            )

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._opened_at = None
            self._half_open_inflight = False

    def record_failure(self) -> CircuitSnapshot:
        now = time.monotonic()
        with self._lock:
            self._half_open_inflight = False
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._opened_at = now
                return CircuitSnapshot("open", self._failure_count)
            return CircuitSnapshot("closed", self._failure_count)

    def snapshot(self) -> CircuitSnapshot:
        with self._lock:
            if self._opened_at is None:
                return CircuitSnapshot("closed", self._failure_count)
            return CircuitSnapshot(
                "open",
                self._failure_count,
                int((time.monotonic() - self._opened_at) * 1000),
            )
