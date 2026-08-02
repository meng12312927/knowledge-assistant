import pytest
from pydantic import ValidationError

from models.document import Citation, QueryRequest, RetrievedChunk
from embeddings.factory import DashScopeEmbeddingClient
from rag.chains.rag_chain import RAGChain


def test_query_request_rejects_blank_query():
    with pytest.raises(ValidationError):
        QueryRequest(query="")


def test_query_request_rejects_excessive_top_k():
    with pytest.raises(ValidationError):
        QueryRequest(query="hello", top_k=21)


def test_dashscope_embedding_rejects_unsupported_dimension():
    with pytest.raises(ValueError, match="不支持"):
        DashScopeEmbeddingClient(api_key="test", dimension=999)


def test_dashscope_embedding_rejects_batch_larger_than_ten():
    with pytest.raises(ValueError, match="最多支持 10"):
        DashScopeEmbeddingClient(api_key="test", batch_size=11)


def test_dashscope_embedding_sends_model_dimension_and_batches():
    class EmbeddingItem:
        def __init__(self, value):
            self.embedding = value

    class EmbeddingsAPI:
        def __init__(self):
            self.requests = []

        def create(self, **kwargs):
            self.requests.append(kwargs)
            return type("Response", (), {
                "data": [EmbeddingItem([1.0, 0.0]) for _ in kwargs["input"]]
            })()

    client = DashScopeEmbeddingClient.__new__(DashScopeEmbeddingClient)
    client.model = "text-embedding-v4"
    client._dimension = 2
    client.batch_size = 2
    client.max_retries = 0
    api = EmbeddingsAPI()
    client.client = type("Client", (), {"embeddings": api})()

    vectors = client.embed(["a", "b", "c"])

    assert len(vectors) == 3
    assert len(api.requests) == 2
    assert api.requests[0]["model"] == "text-embedding-v4"
    assert api.requests[0]["dimensions"] == 2


def test_citation_requires_structured_id():
    with pytest.raises(ValidationError):
        Citation(
            citation_id="1",
            chunk_id="chunk-1",
            source_file="制度.txt",
            content="原文",
            score=0.9,
        )


def test_context_labels_match_structured_citations():
    chain = RAGChain.__new__(RAGChain)
    chain.max_context_tokens = 8000
    chunks = [RetrievedChunk(
        content="费用超过5000元且不超过20000元，需要财务负责人审批。",
        metadata={
            "chunk_id": "chunk-travel-1",
            "doc_id": "doc-travel",
            "source_file": "差旅与费用报销制度.txt",
            "page_number": 1,
        },
        score=0.032,
    )]

    context, _, included = chain._build_context(chunks)
    citations = chain._build_citations(included)

    assert "[S1]" in context
    assert citations[0].citation_id == "S1"
    assert citations[0].chunk_id == "chunk-travel-1"
    assert citations[0].content == chunks[0].content


def test_only_citations_used_in_answer_are_returned():
    citations = [
        Citation(citation_id="S1", chunk_id="c1", source_file="a.txt", content="A", score=0.1),
        Citation(citation_id="S2", chunk_id="c2", source_file="b.txt", content="B", score=0.2),
    ]
    used = RAGChain._filter_used_citations("结论一[S2]，再次引用[S2]。", citations)
    assert [citation.citation_id for citation in used] == ["S2"]
