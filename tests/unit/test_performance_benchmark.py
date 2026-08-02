import json

from tests.performance_benchmark import EndpointRunner, generation_was_skipped


def test_not_found_is_treated_as_intentionally_skipped_generation():
    assert generation_was_skipped({}, "not_found") is True


def test_generation_span_can_explicitly_mark_skip():
    trace = {
        "spans": [{
            "name": "generation",
            "attributes": {"stream": True, "skipped": True},
        }],
    }

    assert generation_was_skipped(trace, "answerable") is True


def test_answerable_request_without_skip_still_requires_generation_ttft():
    trace = {
        "spans": [{
            "name": "generation",
            "attributes": {"stream": True, "skipped": False},
        }],
    }

    assert generation_was_skipped(trace, "answerable") is False


def test_not_found_sse_without_generation_ttft_remains_observability_complete(monkeypatch):
    events = [
        {"type": "token", "content": "根据现有知识库，无法找到相关信息。"},
        {"type": "sources", "sources": [], "answer_status": "not_found"},
        {
            "type": "rag_trace",
            "trace": {
                "spans": [{
                    "name": "generation",
                    "start_offset_ms": 5,
                    "duration_ms": 0,
                    "attributes": {"stream": True, "skipped": True},
                }],
                "verified_ttft_ms": 8,
                "user_visible_ttft_ms": 8,
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reranker_tokens": 12,
                    "total_tokens": 12,
                },
            },
        },
        {"type": "done", "metrics": {"server_done_emit_ms": 9}},
    ]

    class FakeResponse:
        ok = True
        status_code = 200
        headers = {"X-Request-ID": "response-id"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_lines(self, **kwargs):
            for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False)}"

    class FakeSession:
        def post(self, *args, **kwargs):
            return FakeResponse()

    runner = EndpointRunner("test", "http://127.0.0.1:8000", timeout=1, top_k=6)
    monkeypatch.setattr(runner, "_session", lambda: FakeSession())

    record = runner.run_one({"query": "食堂早餐几点供应？"}, 1)

    assert record["success"] is True
    assert record["generation_ttft_ms"] is None
    assert record["generation_skipped"] is True
    assert record["observability_errors"] == []
    assert record["observability_complete"] is True
