import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.api.main as api_main
from models.document import RAGTrace


class _FakeLLM:
    def reset_request_usage(self):
        return None

    def request_usage(self):
        return {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}


class _FakeRAGChain:
    llm = _FakeLLM()

    def stream(self, request, state):
        state.update({
            "sources": [],
            "citations": [],
            "citation_verification": None,
            "trace": RAGTrace(
                generation_ttft_ms=7,
                generation_first_token_at_ms=11,
                verified_ttft_ms=19,
            ),
            "answer_status": "answerable",
        })
        yield "测试回答"


def test_stream_persists_the_same_completed_trace_that_it_emits(monkeypatch):
    saved_messages = []
    monkeypatch.setattr(api_main, "rag_chain", _FakeRAGChain())
    monkeypatch.setattr(
        api_main,
        "get_settings",
        lambda: SimpleNamespace(enable_agent=False),
    )
    monkeypatch.setattr(
        api_main,
        "save_message",
        lambda *args, **kwargs: saved_messages.append((args, kwargs)),
    )

    client = TestClient(api_main.app)
    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"query": "测试", "session_id": "trace-session"},
    ) as response:
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert response.status_code == 200
    assert [event["type"] for event in events] == [
        "token", "sources", "citations", "rag_trace", "metrics", "done",
    ]

    assistant_save = next(
        args for args, _ in saved_messages if args[1] == "assistant"
    )
    persisted_trace = assistant_save[5]
    emitted_trace = next(
        event["trace"] for event in events if event["type"] == "rag_trace"
    )
    done_metrics = events[-1]["metrics"]

    assert persisted_trace["sse_total_latency_ms"] is not None
    assert persisted_trace["sse_total_latency_ms"] == emitted_trace["sse_total_latency_ms"]
    assert persisted_trace["sse_total_latency_ms"] == done_metrics["sse_total_latency_ms"]
    assert persisted_trace["generation_ttft_ms"] == 7
    assert persisted_trace["generation_first_token_at_ms"] == 11
    assert persisted_trace["verified_ttft_ms"] == 19
    assert persisted_trace["ttft_ms"] == persisted_trace["user_visible_ttft_ms"]
