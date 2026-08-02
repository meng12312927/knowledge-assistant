import threading

import pytest

from models.document import RetrievedChunk
from rag.post_processors.reranker import Qwen3Reranker


def chunk(chunk_id, score):
    return RetrievedChunk(
        content=f"content-{chunk_id}",
        metadata={"chunk_id": chunk_id, "source_file": "制度.txt"},
        score=score,
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, path, **kwargs):
        self.requests.append((path, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_qwen3_reranker_calls_official_endpoint_and_maps_local_chunks():
    client = FakeClient([{
        "id": "rerank-request-1",
        "model": "qwen3-rerank",
        "results": [
            {"index": 2, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.71},
        ],
        "usage": {"total_tokens": 42},
    }])
    reranker = Qwen3Reranker(client=client, max_retries=0)
    candidates = [chunk("c1", 0.031), chunk("c2", 0.028), chunk("c3", 0.026)]

    ranked = reranker.rerank("差旅费如何审批", candidates, top_n=2)

    assert [item.metadata["chunk_id"] for item in ranked] == ["c3", "c1"]
    assert [item.score for item in ranked] == [0.95, 0.71]
    assert [item.rank for item in ranked] == [1, 2]
    assert ranked[0].metadata["rrf_score"] == pytest.approx(0.026)
    assert ranked[0].metadata["rerank_model"] == "qwen3-rerank"
    assert candidates[2].score == pytest.approx(0.026)
    assert "rrf_score" not in candidates[2].metadata

    path, kwargs = client.requests[0]
    assert path == "/reranks"
    assert kwargs["cast_to"] is object
    assert kwargs["body"]["model"] == "qwen3-rerank"
    assert kwargs["body"]["documents"] == [item.content for item in candidates]
    assert kwargs["body"]["top_n"] == 2
    assert kwargs["body"]["instruct"].startswith("Given a web search query")

    metadata = reranker.get_last_metadata()
    assert {
        key: metadata[key]
        for key in (
            "provider", "model", "fallback", "error", "request_id",
            "usage", "candidate_count", "output_count",
        )
    } == {
        "provider": "dashscope",
        "model": "qwen3-rerank",
        "fallback": False,
        "error": None,
        "request_id": "rerank-request-1",
        "usage": {"total_tokens": 42},
        "candidate_count": 3,
        "output_count": 2,
    }
    assert metadata["retry_count"] == 0
    assert metadata["timeout_ms"] == 20000
    assert metadata["queue_time_ms"] == 0
    assert metadata["upstream_request_id"] == "rerank-request-1"
    assert metadata["circuit_state"] == "closed"


def test_qwen3_reranker_safely_ignores_invalid_duplicate_indices_and_scores():
    client = FakeClient([{
        "results": [
            {"index": 1, "relevance_score": 0.75},
            {"index": 1, "relevance_score": 0.99},  # duplicate
            {"index": -1, "relevance_score": 0.80},
            {"index": 99, "relevance_score": 0.80},
            {"index": True, "relevance_score": 0.80},
            {"index": "0", "relevance_score": 0.80},
            {"index": 0, "relevance_score": float("nan")},
            {"index": 2, "relevance_score": 1.01},
        ]
    }])
    reranker = Qwen3Reranker(client=client, max_retries=0)

    ranked = reranker.rerank(
        "审批",
        [chunk("c1", 0.03), chunk("c2", 0.02), chunk("c3", 0.01)],
        top_n=3,
    )

    assert [item.metadata["chunk_id"] for item in ranked] == ["c2"]
    assert reranker.last_metadata["output_count"] == 1
    assert reranker.last_metadata["fallback"] is False


def test_qwen3_reranker_retries_then_succeeds():
    client = FakeClient([
        RuntimeError("temporary-1"),
        RuntimeError("temporary-2"),
        {"id": "ok", "results": [{"index": 0, "relevance_score": 0.8}]},
    ])
    reranker = Qwen3Reranker(
        client=client,
        max_retries=2,
        retry_base_seconds=0,
    )

    ranked = reranker.rerank("报销", [chunk("c1", 0.03)], top_n=1)

    assert len(client.requests) == 3
    assert ranked[0].score == pytest.approx(0.8)
    assert reranker.last_metadata["fallback"] is False
    assert reranker.last_metadata["retry_count"] == 2


def test_qwen3_reranker_falls_back_to_noop_and_records_error():
    error = RuntimeError("service unavailable")
    error.request_id = "failed-request"
    client = FakeClient([error, error])
    reranker = Qwen3Reranker(
        client=client,
        max_retries=1,
        retry_base_seconds=0,
    )
    candidates = [chunk("low", 0.01), chunk("high", 0.04)]

    ranked = reranker.rerank("制度", candidates, top_n=1)

    assert [item.metadata["chunk_id"] for item in ranked] == ["high"]
    metadata = reranker.get_last_metadata()
    assert metadata["fallback"] is True
    assert metadata["request_id"] == "failed-request"
    assert metadata["error"] == "RuntimeError: service unavailable"
    assert metadata["candidate_count"] == 2
    assert metadata["output_count"] == 1


def test_qwen3_reranker_empty_input_does_not_call_api():
    client = FakeClient([])
    reranker = Qwen3Reranker(client=client)

    assert reranker.rerank("制度", [], top_n=5) == []
    assert client.requests == []
    assert reranker.last_metadata["candidate_count"] == 0
    assert reranker.last_metadata["output_count"] == 0


def test_qwen3_reranker_metadata_is_thread_local():
    class ThreadAwareClient:
        def post(self, path, **kwargs):
            query = kwargs["body"]["query"]
            return {
                "id": query,
                "results": [{"index": 0, "relevance_score": 0.8}],
            }

    reranker = Qwen3Reranker(client=ThreadAwareClient(), max_retries=0)
    observed = {}

    def run(query):
        reranker.rerank(query, [chunk(query, 0.03)], top_n=1)
        observed[query] = reranker.get_last_metadata()["request_id"]

    threads = [threading.Thread(target=run, args=(query,)) for query in ("q1", "q2")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert observed == {"q1": "q1", "q2": "q2"}
    assert reranker.get_last_metadata() == {}
