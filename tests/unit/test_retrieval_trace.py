from models.document import RetrievedChunk
from rag.retrievers.hybrid import HybridRetriever


def chunk(chunk_id, score, source="制度.txt"):
    return RetrievedChunk(
        content=f"content-{chunk_id}",
        metadata={"chunk_id": chunk_id, "source_file": source},
        score=score,
    )


class FakeVectorStore:
    def similarity_search(self, **kwargs):
        return [chunk("c1", 0.91), chunk("c2", 0.82)]


class FakeBM25:
    def retrieve(self, query, top_k):
        return [chunk("c2", 1.0), chunk("c3", 0.6)]


def test_hybrid_retriever_exposes_channel_ranks_and_rrf_scores():
    retriever = HybridRetriever(FakeVectorStore(), FakeBM25())
    results, trace = retriever.retrieve(
        query="审批流程",
        query_embedding=[1.0, 0.0],
        top_k=3,
        return_trace=True,
    )

    assert [item["chunk_id"] for item in trace["dense"]] == ["c1", "c2"]
    assert [item["rank"] for item in trace["bm25"]] == [1, 2]
    assert trace["rrf"][0]["chunk_id"] == "c2"
    assert results[0].metadata["chunk_id"] == "c2"
