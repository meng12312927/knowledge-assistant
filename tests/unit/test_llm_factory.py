from types import SimpleNamespace

import pytest

from rag.llm.factory import ProviderConfig, ResilientLLMClient, _provider_config


class FakeClient:
    def __init__(self, responses=None, failures=0, usage=None):
        self.responses = list(responses or ["ok"])
        self.failures = failures
        self.calls = 0
        self.usage = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def generate(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary failure")
        return self.responses[min(self.calls - self.failures - 1, len(self.responses) - 1)]

    def generate_stream(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary failure")
        yield from self.responses

    def get_last_usage(self):
        return self.usage

    def get_last_metadata(self):
        return {"upstream_request_id": "upstream-test-1"}


def info(name):
    return ProviderConfig(name=name, api_key="secret", base_url="https://example.test", model=f"{name}-model")


def test_short_task_routes_to_fast_provider():
    primary, fast = FakeClient(["primary"]), FakeClient(["fast"])
    client = ResilientLLMClient(primary, info("deepseek"), fast, info("dashscope"), max_retries=0)

    result = client.generate("system", "query", max_tokens=300)

    assert result == "fast"
    assert fast.calls == 1
    assert primary.calls == 0


def test_citation_verification_routes_to_fast_provider_with_large_budget():
    primary, fast = FakeClient(["primary"]), FakeClient(["verified"])
    client = ResilientLLMClient(
        primary,
        info("deepseek"),
        fast,
        info("dashscope"),
        max_retries=2,
        fast_stage_max_retries=0,
    )

    result = client.generate(
        "system",
        "claims",
        max_tokens=800,
        stage="citation_verification",
    )

    assert result == "verified"
    assert fast.calls == 1
    assert primary.calls == 0


def test_fast_stage_does_not_amplify_tail_with_retries():
    primary, fast = FakeClient(["primary"]), FakeClient(failures=5)
    client = ResilientLLMClient(
        primary,
        info("deepseek"),
        fast,
        info("dashscope"),
        max_retries=2,
        fast_stage_max_retries=0,
        retry_base_seconds=0,
    )

    with pytest.raises(RuntimeError):
        client.generate(
            "system", "claims", max_tokens=800, stage="citation_verification"
        )

    assert fast.calls == 1
    assert client.metrics()["retries"] == 0


def test_primary_retries_before_succeeding():
    primary = FakeClient(["recovered"], failures=1)
    client = ResilientLLMClient(primary, info("deepseek"), max_retries=1, retry_base_seconds=0)

    assert client.generate("system", "question") == "recovered"
    assert client.metrics()["retries"] == 1
    event = client.request_events()[-1]
    assert event["stage"] == "generation"
    assert event["retry_count"] == 1
    assert event["upstream_request_id"] == "upstream-test-1"
    assert event["circuit_state"] == "closed"


def test_llm_circuit_opens_and_is_visible_in_request_events():
    primary = FakeClient(failures=10)
    client = ResilientLLMClient(
        primary,
        info("deepseek"),
        max_retries=0,
        retry_base_seconds=0,
        circuit_failure_threshold=1,
        circuit_recovery_seconds=60,
    )

    with pytest.raises(RuntimeError):
        client.generate("system", "question", stage="query_rewrite")
    with pytest.raises(Exception):
        client.generate("system", "question", stage="query_rewrite")

    event = client.request_events()[-1]
    assert event["circuit_state"] == "open"
    assert event["error_type"] == "CircuitOpenError"


def test_fallback_is_used_after_primary_exhausts_retries():
    primary, fallback = FakeClient(failures=3), FakeClient(["fallback answer"])
    client = ResilientLLMClient(
        primary, info("deepseek"), fallback=fallback, fallback_info=info("dashscope"),
        max_retries=1, retry_base_seconds=0,
    )

    assert client.generate("system", "question") == "fallback answer"
    assert client.metrics()["fallbacks"] == 1


def test_stream_falls_back_before_first_token():
    primary, fallback = FakeClient(failures=2), FakeClient(["fallback", " stream"])
    client = ResilientLLMClient(
        primary, info("deepseek"), fallback=fallback, fallback_info=info("dashscope"),
        max_retries=1, retry_base_seconds=0,
    )

    assert "".join(client.generate_stream("system", "question")) == "fallback stream"
    assert client.metrics()["fallbacks"] == 1


def test_provider_config_requires_key():
    settings = SimpleNamespace(
        deepseek_api_key=None, deepseek_base_url="https://api.deepseek.com", deepseek_model="model",
    )
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        _provider_config("deepseek", settings)


def test_request_usage_aggregates_all_llm_stages():
    primary = FakeClient(usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13})
    client = ResilientLLMClient(primary, info("deepseek"), max_retries=0)

    client.reset_request_usage()
    client.generate("system", "rewrite", max_tokens=160)
    client.generate("system", "answer")

    assert client.request_usage() == {
        "prompt_tokens": 20,
        "completion_tokens": 6,
        "total_tokens": 26,
    }
