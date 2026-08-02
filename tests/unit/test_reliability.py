import time

import pytest

from rag.reliability import (
    CircuitBreaker,
    CircuitOpenError,
    RequestBudgetExceeded,
    bounded_timeout,
    clear_request_budget,
    start_request_budget,
)


def test_stage_timeout_is_capped_by_total_request_budget():
    token = start_request_budget(0.05)
    try:
        timeout = bounded_timeout(5)
        assert 0 < timeout <= 0.05
        time.sleep(0.06)
        with pytest.raises(RequestBudgetExceeded):
            bounded_timeout(5)
    finally:
        clear_request_budget(token)


def test_circuit_breaker_opens_and_half_open_probe_recovers():
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=0.01)

    assert breaker.before_call().state == "closed"
    assert breaker.record_failure().state == "closed"
    assert breaker.record_failure().state == "open"
    with pytest.raises(CircuitOpenError):
        breaker.before_call()

    time.sleep(0.02)
    assert breaker.before_call().state == "half_open"
    breaker.record_success()
    assert breaker.snapshot().state == "closed"
