import threading
import time
from concurrent.futures import ThreadPoolExecutor

from models.document import QueryRequest, RAGTrace, RetrievedChunk
from rag.chains.rag_chain import RAGChain
from rag.post_processors.reranker import NoOpReranker


def chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        content=f"content-{chunk_id}",
        metadata={"chunk_id": chunk_id, "source_file": "制度.txt"},
        score=score,
    )


class FakeEmbedder:
    model_name = "fake-embedding"
    dimension = 2

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(index + 1), 0.0] for index, _ in enumerate(texts)]


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        return "差旅费用审批权限\n15000元报销审批人\n费用报销分级审批"

    def generate_stream(self, **kwargs):
        self.calls += 1
        yield "测试回答 [S1]"


class FakeRetriever:
    def __init__(self, first_score=0.032):
        self.first_score = first_score
        self.calls = []

    def retrieve_many(self, queries, query_embeddings, top_k, filter_dict, return_trace):
        self.calls.append(list(queries))
        outputs = []
        for index, query in enumerate(queries):
            score = self.first_score
            item = chunk(f"c-{len(self.calls)}-{index}", score)
            trace_item = {
                "chunk_id": item.metadata["chunk_id"],
                "rank": 1,
                "score": score,
                "source_file": "制度.txt",
            }
            outputs.append(([item], {
                "dense": [trace_item],
                "bm25": [trace_item],
                "rrf": [trace_item],
                "timing": {
                    "dense_start_ms": 0,
                    "dense_duration_ms": 1,
                    "bm25_start_ms": 0,
                    "bm25_duration_ms": 1,
                },
            }))
        return outputs


def make_chain(first_score=0.032):
    embedder = FakeEmbedder()
    retriever = FakeRetriever(first_score)
    llm = FakeLLM()
    chain = RAGChain(
        embedder=embedder,
        retriever=retriever,
        reranker=NoOpReranker(),
        llm=llm,
        answer_status_threshold_high=0.025,
        answer_status_threshold_low=0.012,
        simple_query_min_rrf_score=0.025,
        retrieval_candidate_k=40,
    )
    return chain, embedder, retriever, llm


def run_retrieval(chain, query):
    trace = RAGTrace()
    candidates = chain._adaptive_retrieve(
        QueryRequest(query=query), trace, time.perf_counter()
    )
    return candidates, trace


def test_exact_amount_skips_multiquery_when_first_recall_is_sufficient():
    chain, embedder, retriever, llm = make_chain(first_score=0.032)

    candidates, trace = run_retrieval(chain, "预计差旅费用15000元需要谁审批？")

    assert candidates[0].score == 0.032
    assert trace.query_strategy == "direct"
    assert trace.multiquery_triggered is False
    assert llm.calls == 0
    assert retriever.calls == [["预计差旅费用15000元需要谁审批？"]]
    assert next(span for span in trace.spans if span.name == "query_rewrite").attributes["skipped"] is True
    assert trace.cache_stats == {
        "embedding_hits": 0,
        "embedding_misses": 1,
        "embedding_singleflight_waits": 0,
        "rewrite_singleflight_waits": 0,
    }

    _, cached_trace = run_retrieval(chain, "预计差旅费用15000元需要谁审批？")
    assert len(embedder.calls) == 1
    assert cached_trace.cache_hits["query_embedding"] is True


def test_exact_query_falls_back_to_multiquery_only_after_weak_recall():
    chain, _, retriever, llm = make_chain(first_score=0.01)

    _, trace = run_retrieval(chain, "差旅费用15000元由谁审批？")

    assert trace.query_strategy == "adaptive_fallback"
    assert trace.multiquery_triggered is True
    assert trace.multiquery_reason == "original_retrieval_insufficient"
    assert llm.calls == 1
    assert len(retriever.calls) == 2
    assert retriever.calls[0] == ["差旅费用15000元由谁审批？"]
    assert "差旅费用15000元由谁审批？" not in retriever.calls[1]


def test_fuzzy_query_stays_direct_when_original_retrieval_is_sufficient():
    chain, embedder, retriever, llm = make_chain()

    _, first = run_retrieval(chain, "公司的报销要求是什么？")
    _, second = run_retrieval(chain, "公司的报销要求是什么？")

    assert first.query_strategy == "direct"
    assert first.multiquery_triggered is False
    assert retriever.calls[0] == ["公司的报销要求是什么？"]
    assert llm.calls == 0
    assert len(embedder.calls) == 1
    assert second.cache_hits == {
        "query_rewrite": False,
        "query_embedding": True,
    }


def test_classifier_keeps_cross_policy_question_out_of_direct_route():
    assert RAGChain._classify_simple_query("《差旅制度》和《福利制度》有什么区别？") == (
        False,
        "complex_or_multi_intent",
    )
    assert RAGChain._classify_simple_query("《差旅制度》第五条是什么？")[0] is True


def test_prepared_retrieval_is_reused_by_generation_without_second_search():
    chain, _, retriever, _ = make_chain(first_score=0.032)
    request = QueryRequest(query="预计差旅费用15000元需要谁审批？")

    prepared = chain.prepare(request)
    retrieval_calls = len(retriever.calls)
    response = chain.invoke(request, prepared=prepared)

    assert retrieval_calls == 1
    assert len(retriever.calls) == retrieval_calls
    assert response.trace.initial_retrieval_top_score == 0.032
    assert response.trace.retrieval_quality == "sufficient"


def test_known_agent_intent_keeps_original_retrieval_but_skips_multiquery():
    chain, _, retriever, llm = make_chain(first_score=0.01)
    request = QueryRequest(query="计算一下 1250*0.8")

    prepared = chain.prepare(request, allow_multiquery=False)

    assert retriever.calls == [["计算一下 1250*0.8"]]
    assert llm.calls == 0
    assert prepared.trace.query_strategy == "direct"
    assert (
        prepared.trace.multiquery_reason
        == "agent_intent_skips_multiquery"
    )


def test_qwen_low_score_marks_obvious_ood_but_fallback_does_not():
    chain, _, _, _ = make_chain()
    low_ranked = [chunk("ood", 0.28)]

    assert chain._refine_status_with_reranker(
        "answerable", low_ranked, {"provider": "dashscope", "fallback": False}
    ) == "not_found"
    assert chain._refine_status_with_reranker(
        "answerable", low_ranked, {"provider": "dashscope", "fallback": True}
    ) == "answerable"


def test_embedding_singleflight_coalesces_concurrent_identical_queries():
    chain, embedder, _, _ = make_chain()
    original_embed = embedder.embed
    barrier = threading.Barrier(2)

    def slow_embed(texts):
        time.sleep(0.05)
        return original_embed(texts)

    embedder.embed = slow_embed

    def run():
        barrier.wait()
        return chain._embed_queries(["同一个并发问题"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: run(), range(2)))

    assert len(embedder.calls) == 1
    assert first[0] == second[0]
    assert (
        first[1]["singleflight_waits"] + second[1]["singleflight_waits"]
    ) == 1


def test_rewrite_singleflight_coalesces_concurrent_identical_queries():
    chain, _, _, llm = make_chain(first_score=0.01)
    original_generate = llm.generate
    barrier = threading.Barrier(2)

    def slow_generate(**kwargs):
        time.sleep(0.05)
        return original_generate(**kwargs)

    llm.generate = slow_generate

    def run():
        barrier.wait()
        return chain._generate_query_variants("模糊报销问题", n=3)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: run(), range(2)))

    assert llm.calls == 1
    assert first[0] == second[0]
    assert int(first[2]) + int(second[2]) == 1
