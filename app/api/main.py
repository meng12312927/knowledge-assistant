"""
FastAPI RESTful API 服务（v0.2）

功能：
- 文件上传（multipart/form-data）
- 对话历史持久化（SQLite）
- 文档去重/覆盖更新（MD5 哈希）
- 删除文档
"""

from dotenv import load_dotenv
load_dotenv()

import hashlib
import logging
import re
import tempfile
import time
import uuid
from pathlib import Path

import json

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from app.core.config import get_settings
from agent.lightweight_router import AgentRouteDecision, decide_agent_route
from agent.router import Intent
from models.document import QueryRequest, ChatResponse, RAGTrace, TokenUsage, TraceSpan
from rag.reliability import clear_request_budget, start_request_budget
from models.database import (
    init_db, save_message, get_conversation_history, list_documents,
    delete_document_meta, clear_all_data, get_knowledge_base_version,
    bump_knowledge_base_version,
)

logging.basicConfig(
    level=getattr(logging, get_settings().log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# 全局依赖（lifespan 中初始化）
rag_chain = None
agent_router = None
vector_store = None
bm25_retriever = None
retriever = None


def _resolve_agent_route(
    settings,
    request: QueryRequest,
    retrieval_quality: str = "unknown",
) -> AgentRouteDecision:
    """Combine retrieval quality, deterministic intent rules and API override."""
    if not settings.enable_agent:
        return AgentRouteDecision(
            False, Intent.FACTUAL_QA, "agent_globally_disabled"
        )
    return decide_agent_route(
        request.query,
        retrieval_quality=retrieval_quality,
        user_requested_agent=request.enable_agent,
    )


def _refresh_trace_token_usage(trace) -> None:
    """合并本次所有 LLM Token 与独立 Reranker Token，避免 API 层覆盖后者。"""
    if not trace or not rag_chain:
        return
    llm_usage = (
        rag_chain.llm.request_usage()
        if hasattr(rag_chain.llm, "request_usage") else {}
    )
    reranker_tokens = int(getattr(trace.token_usage, "reranker_tokens", 0) or 0)
    llm_total = int(llm_usage.get("total_tokens", 0) or 0)
    trace.token_usage = TokenUsage(
        prompt_tokens=int(llm_usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(llm_usage.get("completion_tokens", 0) or 0),
        reranker_tokens=reranker_tokens,
        total_tokens=llm_total + reranker_tokens,
    )


def _rebuild_bm25():
    """重建 BM25 语料库（文档增删后调用）"""
    global bm25_retriever, retriever, vector_store
    if not vector_store:
        return False
    try:
        from rag.retrievers.hybrid import BM25Retriever
        all_docs = vector_store.get_all()
        if not all_docs:
            bm25_retriever = None
            if retriever:
                retriever.bm25_retriever = None
            print("[OK] BM25 cleared (no docs)")
            return True
        new_bm25 = BM25Retriever(
            texts=[d.content for d in all_docs],
            metadatas=[d.metadata for d in all_docs]
        )
        bm25_retriever = new_bm25
        if retriever:
            retriever.bm25_retriever = new_bm25
        print(f"[OK] BM25 rebuilt | docs: {len(all_docs)}")
        return True
    except Exception as e:
        print(f"[WARN] BM25 rebuild failed: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, agent_router, vector_store, bm25_retriever, retriever

    # 初始化数据库
    init_db()

    settings = get_settings()
    print(f"[START] {settings.app_name} v{settings.app_version}")
    print(f"[EMBED] {settings.embedding_provider} / {settings.embedding_model}")

    # 初始化 Embedding
    from embeddings.factory import EmbeddingFactory
    embedder = EmbeddingFactory.create(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        **({
            "api_key": settings.dashscope_api_key,
            "base_url": settings.dashscope_base_url,
            "dimension": settings.embedding_dimension,
            "batch_size": settings.embedding_batch_size,
            "max_retries": settings.embedding_max_retries,
            "timeout": settings.embedding_timeout_seconds,
            "circuit_failure_threshold": settings.circuit_breaker_failure_threshold,
            "circuit_recovery_seconds": settings.circuit_breaker_recovery_seconds,
        } if settings.embedding_provider.lower() in {"dashscope", "aliyun", "qwen"} else {})
    )

    # 初始化向量库
    from vectorstore.factory import VectorStoreFactory
    vs_provider = settings.vectorstore_provider.lower()

    vs_kwargs = {}
    if vs_provider == "chroma":
        chroma_path = Path(settings.chroma_persist_dir)
        if chroma_path.exists() and any(chroma_path.iterdir()):
            print(f"[INFO] Vector store exists: {settings.chroma_persist_dir}")
        vs_kwargs["persist_directory"] = settings.chroma_persist_dir
    else:
        raise ValueError("本项目仅支持 Chroma 向量数据库")

    print(f"[INFO] Embedding dim: {embedder.dimension}D")

    vector_store = VectorStoreFactory.create(
        provider=vs_provider,
        collection_name=settings.vectorstore_collection,
        dimension=embedder.dimension,
        **vs_kwargs
    )

    # 初始化检索链
    from rag.retrievers.hybrid import HybridRetriever, BM25Retriever
    from rag.post_processors.reranker import NoOpReranker, Qwen3Reranker
    from rag.post_processors.citation_verifier import CitationVerifier
    from rag.chains.rag_chain import RAGChain
    from rag.conversation import ConversationManager
    from rag.llm.factory import create_llm_client, get_provider_info
    from agent.router import AgentRouter
    from agent.langgraph_router import LangGraphAgentRouter

    # 构建 BM25 语料库（从向量库读取已有文档）
    doc_count = vector_store.count()
    if doc_count > 0:
        try:
            all_docs = vector_store.get_all()
            bm25_retriever = BM25Retriever(
                texts=[d.content for d in all_docs],
                metadatas=[d.metadata for d in all_docs]
            )
            print(f"[OK] BM25 corpus built | docs: {len(all_docs)}")
        except Exception as e:
            print(f"[WARN] BM25 build failed: {e}, fallback to dense only")

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        parallel_workers=settings.retrieval_parallel_workers,
    )

    if settings.reranker_provider.lower() == "qwen3":
        reranker = Qwen3Reranker(
            api_key=settings.dashscope_api_key,
            base_url=settings.reranker_base_url,
            model_name=settings.reranker_model,
            timeout=settings.reranker_timeout_seconds,
            max_retries=settings.reranker_max_retries,
            retry_base_seconds=settings.llm_retry_base_seconds,
            instruct=settings.reranker_instruct,
            max_candidates=settings.reranker_candidate_k,
            circuit_failure_threshold=settings.circuit_breaker_failure_threshold,
            circuit_recovery_seconds=settings.circuit_breaker_recovery_seconds,
        )
        print(f"[OK] Reranker enabled | dashscope / {settings.reranker_model}")
    else:
        reranker = NoOpReranker()
        print("[WARN] Reranker disabled; using RRF order")

    llm = create_llm_client(
        client_type=settings.llm_client_type,
        provider=settings.llm_default_provider,
        settings=settings
    )
    provider_info = get_provider_info(settings)
    print(f"[OK] {settings.llm_client_type.upper()} LLM client enabled | provider={provider_info['provider']} model={provider_info['model']}")
    citation_verifier = CitationVerifier(
        llm=llm,
        enabled=settings.citation_verification_enabled,
        strict=settings.citation_verification_strict,
    )
    rag_chain = RAGChain(
        embedder=embedder,
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        citation_verifier=citation_verifier,
        knowledge_base_version_getter=get_knowledge_base_version,
        max_context_tokens=settings.max_context_tokens,
        answer_status_threshold_high=settings.answer_status_threshold_high,
        answer_status_threshold_low=settings.answer_status_threshold_low,
        adaptive_multiquery_enabled=settings.adaptive_multiquery_enabled,
        query_rewrite_cache_size=settings.query_rewrite_cache_size,
        query_embedding_cache_size=settings.query_embedding_cache_size,
        simple_query_min_rrf_score=settings.simple_query_min_rrf_score,
        retrieval_candidate_k=settings.reranker_candidate_k,
        rerank_top_n=settings.reranker_top_n,
        reranker_not_found_threshold=settings.reranker_not_found_threshold,
        subquestion_planning_enabled=settings.subquestion_planning_enabled,
        subquestion_max_count=settings.subquestion_max_count,
        subquestion_rerank_top_n=settings.subquestion_rerank_top_n,
        subquestion_rerank_candidate_k=(
            settings.subquestion_rerank_candidate_k
        ),
        evidence_per_source_limit=settings.evidence_per_source_limit,
        query_rewrite_timeout_seconds=settings.query_rewrite_timeout_seconds,
        embedding_timeout_seconds=settings.embedding_timeout_seconds,
        reranker_timeout_seconds=settings.reranker_timeout_seconds,
        generation_timeout_seconds=settings.generation_timeout_seconds,
        citation_verification_timeout_seconds=(
            settings.citation_verification_timeout_seconds
        ),
        conversation_manager=ConversationManager(llm=llm),
    )

    if settings.agent_router_type == "langgraph":
        from agent.tools.base import DEFAULT_TOOLS
        agent_router = LangGraphAgentRouter(
            llm=llm,
            rag_chain=rag_chain,
            tools=DEFAULT_TOOLS
        )
        print("[OK] LangGraph AgentRouter enabled")
    else:
        agent_router = AgentRouter(llm=llm, rag_chain=rag_chain)
        print("[OK] Legacy AgentRouter enabled")

    print(f"[OK] Init done | docs: {doc_count} | hybrid: {'Y' if bm25_retriever else 'N'}")

    yield

    print("[SHUTDOWN] App stopped")


app = FastAPI(
    title="企业制度与员工手册助手 API",
    description="支持制度检索、引用溯源与多模型降级的员工自助问答系统",
    version="1.0.0",
    lifespan=lifespan
)

# CORS：从配置读取允许的来源，生产环境不应开放 *
_settings = get_settings()
_cors_origins = [o.strip() for o in (_settings.cors_origins or "http://localhost:8501,http://127.0.0.1:8501").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed request_id=%s method=%s path=%s", request_id, request.method, request.url.path)
        raise
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id, request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


# ═══════════════════════════════════════════════════════════
# 健康检查
# ═══════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    settings = get_settings()
    ready = rag_chain is not None and vector_store is not None
    return {
        "status": "ok" if ready else "starting",
        "service": settings.app_name,
        "version": settings.app_version,
        "ready": ready,
    }


@app.get("/api/v1/models/status")
def model_status():
    """返回已配置的模型路由和进程内调用指标，不执行收费的探测请求。"""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="服务初始化中，请稍后重试")
    llm = rag_chain.llm
    summary = llm.provider_summary() if hasattr(llm, "provider_summary") else {
        "primary": {"model": getattr(llm, "model", "unknown")}
    }
    settings = get_settings()
    reranker_metadata = (
        rag_chain.reranker.get_last_metadata()
        if hasattr(rag_chain.reranker, "get_last_metadata") else {}
    )
    return {
        "status": "configured",
        **summary,
        "embedding": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "dimension": settings.embedding_dimension,
        },
        "reranker": {
            "provider": settings.reranker_provider,
            "model": settings.reranker_model,
            "candidate_k": settings.reranker_candidate_k,
            "top_n": settings.reranker_top_n,
            "not_found_threshold": settings.reranker_not_found_threshold,
            "last_call": reranker_metadata,
        },
        "retrieval": {
            "simple_query_min_rrf_score": settings.simple_query_min_rrf_score,
            "answer_status_threshold_low": settings.answer_status_threshold_low,
            "answer_status_threshold_high": settings.answer_status_threshold_high,
            "subquestion_planning_enabled": settings.subquestion_planning_enabled,
            "subquestion_max_count": settings.subquestion_max_count,
            "subquestion_rerank_top_n": settings.subquestion_rerank_top_n,
            "subquestion_rerank_candidate_k": (
                settings.subquestion_rerank_candidate_k
            ),
            "evidence_per_source_limit": settings.evidence_per_source_limit,
        },
        "reliability": {
            "request_timeout_seconds": settings.request_timeout_seconds,
            "stage_timeouts_seconds": {
                "query_rewrite": settings.query_rewrite_timeout_seconds,
                "embedding": settings.embedding_timeout_seconds,
                "reranker": settings.reranker_timeout_seconds,
                "generation": settings.generation_timeout_seconds,
                "citation_verification": (
                    settings.citation_verification_timeout_seconds
                ),
            },
            "circuit_breaker": {
                "failure_threshold": (
                    settings.circuit_breaker_failure_threshold
                ),
                "recovery_seconds": settings.circuit_breaker_recovery_seconds,
            },
        },
    }


# ═══════════════════════════════════════════════════════════
# 问答接口
# ═══════════════════════════════════════════════════════════

@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: QueryRequest):
    """问答接口（主入口），返回后自动保存到对话历史"""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="服务初始化中，请稍后重试")

    settings = get_settings()
    request_timeout = getattr(settings, "request_timeout_seconds", 30.0)
    budget_token = start_request_budget(request_timeout)
    endpoint_started = time.perf_counter()
    if hasattr(rag_chain.llm, "reset_request_usage"):
        rag_chain.llm.reset_request_usage()
    request_embedder = getattr(rag_chain, "embedder", None)
    if hasattr(request_embedder, "reset_request_usage"):
        request_embedder.reset_request_usage()

    # 保存用户提问
    if request.session_id:
        save_message(request.session_id, "user", request.query)

    try:
        preliminary_route = _resolve_agent_route(
            settings, request, "unknown"
        )
        prepared = (
            rag_chain.prepare(
                request,
                allow_multiquery=not preliminary_route.use_agent,
            )
            if hasattr(rag_chain, "prepare")
            else None
        )
        agent_route = _resolve_agent_route(
            settings,
            request,
            prepared.trace.retrieval_quality if prepared else "unknown",
        )
        if agent_route.use_agent:
            response = agent_router.execute(
                agent_route.plan,
                request.query,
                prepared=prepared,
            )
        else:
            response = (
                rag_chain.invoke(request, prepared=prepared)
                if prepared
                else rag_chain.invoke(request)
            )

        if agent_route.use_agent and response.trace is None:
            response.trace = prepared.trace if prepared else RAGTrace()
        if response.trace:
            response.trace.agent_decision = agent_route.use_agent
            response.trace.agent_reason = agent_route.reason
            if prepared and response.trace is not prepared.trace:
                response.trace.initial_retrieval_top_score = (
                    prepared.trace.initial_retrieval_top_score
                )
                response.trace.final_retrieval_top_score = (
                    prepared.trace.final_retrieval_top_score
                )
                response.trace.retrieval_quality = (
                    prepared.trace.retrieval_quality
                )
                response.trace.routing_probe_strategy = (
                    prepared.trace.routing_probe_strategy
                )
                response.trace.routing_probe_multiquery_triggered = (
                    prepared.trace.routing_probe_multiquery_triggered
                )
            total_ms = int((time.perf_counter() - endpoint_started) * 1000)
            response.trace.total_latency_ms = total_ms
            response.trace.verified_ttft_ms = response.trace.verified_ttft_ms or total_ms
            response.trace.user_visible_ttft_ms = total_ms
            response.trace.ttft_ms = total_ms
            response.trace.sse_total_latency_ms = total_ms
            _refresh_trace_token_usage(response.trace)
            response.trace.spans.insert(0, TraceSpan(
                name="api_total", start_offset_ms=0, duration_ms=total_ms,
                attributes={
                    "endpoint": "/api/v1/chat",
                    "agent_requested": request.enable_agent,
                    "agent": agent_route.use_agent,
                    "agent_intent": agent_route.intent.value,
                    "agent_route_reason": agent_route.reason,
                    "request_budget_ms": int(
                        request_timeout * 1000
                    ),
                },
            ))

        # 保存助手回答
        if request.session_id:
            citation_records = [citation.model_dump() for citation in response.citations]
            verification_record = response.citation_verification.model_dump() if response.citation_verification else None
            trace_record = response.trace.model_dump() if response.trace else None
            save_message(request.session_id, "assistant", response.answer, citation_records, verification_record, trace_record)

        return response
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部服务器错误，请稍后重试")
    finally:
        clear_request_budget(budget_token)


@app.post("/api/v1/chat/stream")
def chat_stream(request: QueryRequest):
    """流式问答接口，返回 SSE 事件流"""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="服务初始化中，请稍后重试")

    settings = get_settings()

    def event_generator():
        endpoint_started = time.perf_counter()
        request_timeout = getattr(settings, "request_timeout_seconds", 30.0)
        budget_token = start_request_budget(request_timeout)
        api_ttft_ms = None
        if hasattr(rag_chain.llm, "reset_request_usage"):
            rag_chain.llm.reset_request_usage()
        request_embedder = getattr(rag_chain, "embedder", None)
        if hasattr(request_embedder, "reset_request_usage"):
            request_embedder.reset_request_usage()
        # 保存用户提问
        if request.session_id:
            save_message(request.session_id, "user", request.query)

        full_answer = ""
        try:
            preliminary_route = _resolve_agent_route(
                settings, request, "unknown"
            )
            prepared = (
                rag_chain.prepare(
                    request,
                    allow_multiquery=not preliminary_route.use_agent,
                )
                if hasattr(rag_chain, "prepare")
                else None
            )
            agent_route = _resolve_agent_route(
                settings,
                request,
                prepared.trace.retrieval_quality if prepared else "unknown",
            )
            if agent_route.use_agent:
                # Agent 模式暂不支持流式，fallback 到非流式
                # 轻量规则已提供确定性计划，跳过额外的 LLM 意图分类。
                response = agent_router.execute(
                    agent_route.plan,
                    request.query,
                    prepared=prepared,
                )
                if response.trace is None:
                    response.trace = prepared.trace if prepared else RAGTrace()
                full_answer = response.answer
                if not full_answer:
                    raise RuntimeError("模型返回了空回答，请稍后重试")
                api_ttft_ms = int((time.perf_counter() - endpoint_started) * 1000)
                yield f"data: {json.dumps({'type': 'token', 'content': full_answer})}\n\n"
                sources = response.sources
                citations = response.citations
                verification = response.citation_verification
                trace = response.trace
                answer_status = response.answer_status
            else:
                rag_state = {}
                stream = (
                    rag_chain.stream(
                        request, state=rag_state, prepared=prepared
                    )
                    if prepared
                    else rag_chain.stream(request, state=rag_state)
                )
                for chunk in stream:
                    full_answer += chunk
                    if api_ttft_ms is None:
                        api_ttft_ms = int((time.perf_counter() - endpoint_started) * 1000)
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                sources = rag_state.get("sources", [])
                citations = rag_state.get("citations", [])
                verification = rag_state.get("citation_verification")
                trace = rag_state.get("trace")
                answer_status = rag_state.get("answer_status", "answerable")

            # 发送 sources（RAG 模式附带 answer_status）
            sources_json = [
                {"content": s.content, "metadata": s.metadata, "score": s.score}
                for s in sources
            ]
            citations_json = [citation.model_dump() for citation in citations]
            if trace:
                trace.agent_decision = agent_route.use_agent
                trace.agent_reason = agent_route.reason
                if prepared and trace is not prepared.trace:
                    trace.initial_retrieval_top_score = (
                        prepared.trace.initial_retrieval_top_score
                    )
                    trace.final_retrieval_top_score = (
                        prepared.trace.final_retrieval_top_score
                    )
                    trace.retrieval_quality = (
                        prepared.trace.retrieval_quality
                    )
                    trace.routing_probe_strategy = (
                        prepared.trace.routing_probe_strategy
                    )
                    trace.routing_probe_multiquery_triggered = (
                        prepared.trace.routing_probe_multiquery_triggered
                    )
                total_ms = int((time.perf_counter() - endpoint_started) * 1000)
                trace.user_visible_ttft_ms = api_ttft_ms if api_ttft_ms is not None else total_ms
                trace.ttft_ms = trace.user_visible_ttft_ms
                trace.total_latency_ms = total_ms
                _refresh_trace_token_usage(trace)
                trace.spans.insert(0, TraceSpan(
                    name="api_total", start_offset_ms=0, duration_ms=total_ms,
                    attributes={
                        "endpoint": "/api/v1/chat/stream",
                        "agent_requested": request.enable_agent,
                        "agent": agent_route.use_agent,
                        "agent_intent": agent_route.intent.value,
                        "agent_route_reason": agent_route.reason,
                        "request_budget_ms": int(
                            request_timeout * 1000
                        ),
                    },
                ))
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_json, 'answer_status': answer_status})}\n\n"
            yield f"data: {json.dumps({'type': 'citations', 'citations': citations_json})}\n\n"
            if verification:
                yield f"data: {json.dumps({'type': 'citation_verification', 'verification': verification.model_dump()})}\n\n"

            # 服务端只能观测到“终止事件已准备好”的时刻；真正包含网络 flush 的
            # done 到达时间由客户端性能脚本记录。先写入 Trace，再持久化，确保
            # 会话历史与本次 SSE 返回的 sse_total_latency_ms 保持一致。
            server_done_emit_ms = int((time.perf_counter() - endpoint_started) * 1000)
            metrics = {"server_done_emit_ms": server_done_emit_ms}
            if trace:
                trace.sse_total_latency_ms = server_done_emit_ms
                metrics.update({
                    "generation_ttft_ms": trace.generation_ttft_ms,
                    "generation_first_token_at_ms": trace.generation_first_token_at_ms,
                    "verified_ttft_ms": trace.verified_ttft_ms,
                    "user_visible_ttft_ms": trace.user_visible_ttft_ms,
                    "sse_total_latency_ms": trace.sse_total_latency_ms,
                })

            # 保存助手回答
            if request.session_id:
                verification_record = verification.model_dump() if verification else None
                trace_record = trace.model_dump() if trace else None
                save_message(request.session_id, "assistant", full_answer, citations_json, verification_record, trace_record)

            if trace:
                yield f"data: {json.dumps({'type': 'rag_trace', 'trace': trace.model_dump()})}\n\n"
            yield f"data: {json.dumps({'type': 'metrics', 'metrics': metrics})}\n\n"
            # 客户端收到 done 的时刻由性能脚本测量，才包含真正的网络 flush。
            yield f"data: {json.dumps({'type': 'done', 'metrics': metrics})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            clear_request_budget(budget_token)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/api/v1/chat/history")
def get_history(session_id: str = Query(..., description="对话 session ID")):
    """获取某个 session 的对话历史"""
    return {"session_id": session_id, "messages": get_conversation_history(session_id)}


@app.delete("/api/v1/chat/history")
def clear_history(session_id: str = Query(..., description="对话 session ID")):
    """清空某个 session 的对话历史"""
    from models.database import clear_conversation
    clear_conversation(session_id)
    return {"status": "success", "message": f"已清空 session {session_id} 的对话历史"}


# ═══════════════════════════════════════════════════════════
# 文档摄取接口（真正的文件上传）
# ═══════════════════════════════════════════════════════════

@app.post("/api/v1/ingest")
def ingest_document(file: UploadFile = File(...)):
    """
    文档摄取接口（v0.2）

    - 接收上传的文件（multipart/form-data）
    - 计算 MD5 作为 doc_id，实现去重和覆盖
    - 如果同名/同内容文件已存在，先删除旧数据再入库
    """
    from ingestion.pipeline import run_ingestion_pipeline
    from models.database import (
        save_document_meta, get_document_meta, get_document_meta_by_filename,
        delete_document_meta,
    )

    settings = get_settings()

    # 文件类型白名单在解析前检查，避免把任意内容交给复杂文档解析器。
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()
    allowed = {f".{ext.strip().lower().lstrip('.')}" for ext in settings.allowed_upload_extensions.split(",") if ext.strip()}
    if suffix not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件类型 {suffix or '(无扩展名)'}，允许：{', '.join(sorted(allowed))}",
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    bytes_written = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        while True:
            data = file.file.read(1024 * 1024)
            if not data:
                break
            bytes_written += len(data)
            if bytes_written > max_bytes:
                Path(tmp_path).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"文件超过 {settings.max_upload_size_mb}MB 限制",
                )
            tmp.write(data)
    if bytes_written == 0:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="不能上传空文件")

    try:
        # 计算文件 MD5 作为 doc_id
        hasher = hashlib.md5()
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        doc_id = hasher.hexdigest()

        # 检查是否已存在相同文件
        existing = get_document_meta(doc_id)
        existing_filename = get_document_meta_by_filename(filename)
        declared_version_match = re.search(
            r"_v(?P<version>\d+)(?=\.[^.]+$)", filename, flags=re.IGNORECASE
        )
        declared_version = (
            int(declared_version_match.group("version"))
            if declared_version_match else 1
        )
        document_version = (
            max(existing["version"], declared_version) if existing else
            max(
                existing_filename["version"] + 1 if existing_filename else 1,
                declared_version,
            )
        )
        next_kb_version = get_knowledge_base_version() + 1
        if existing:
            print(f"[DEDUP] Same file detected: {filename} (doc_id={doc_id[:8]}...)")
            print(f"[去重] 先删除旧数据...")
            # 按 doc_id 删除向量库中的旧 chunk
            if vector_store:
                vector_store.delete(filter_dict={"doc_id": doc_id})
            delete_document_meta(doc_id)
        elif existing_filename:
            if vector_store:
                vector_store.delete(filter_dict={"doc_id": existing_filename["doc_id"]})
            delete_document_meta(existing_filename["doc_id"])

        # 执行摄取（把 doc_id 注入到 metadata 中）
        chunks = run_ingestion_pipeline(
            tmp_path,
            doc_id=doc_id,
            source_filename=filename,
            document_version=document_version,
            knowledge_base_version=f"kb-v{next_kb_version}",
        )
        if not chunks:
            raise ValueError("文档未生成任何可入库文本块")

        # 记录文档元数据
        save_document_meta(
            doc_id=doc_id, filename=file.filename,
            chunk_count=len(chunks), version=document_version,
        )
        knowledge_base_version = bump_knowledge_base_version()

        # 重建 BM25 语料库（新增文档后热更新）
        _rebuild_bm25()

        return {
            "status": "success",
            "filename": file.filename,
            "doc_id": doc_id,
            "chunks_ingested": len(chunks),
            "is_update": existing is not None
            or existing_filename is not None,
            "document_version": document_version,
            "knowledge_base_version": f"kb-v{knowledge_base_version}",
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        Path(tmp_path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════
# 文档管理接口
# ═══════════════════════════════════════════════════════════

@app.get("/api/v1/documents")
def list_docs():
    """列出所有已上传的文档"""
    docs = list_documents()
    return {"documents": docs, "total": len(docs)}


@app.delete("/api/v1/documents/{doc_id}")
def delete_document(doc_id: str):
    """删除指定文档（从向量库和元数据中）"""
    try:
        if vector_store:
            vector_store.delete(filter_dict={"doc_id": doc_id})
        delete_document_meta(doc_id)
        knowledge_base_version = bump_knowledge_base_version()

        # 重建 BM25 语料库（删除文档后热更新）
        _rebuild_bm25()

        return {"status": "success", "doc_id": doc_id, "knowledge_base_version": f"kb-v{knowledge_base_version}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# 统计接口
# ═══════════════════════════════════════════════════════════

@app.post("/api/v1/database/clear")
def clear_database():
    """
    一键清空数据库（向量库 + 文档元数据 + 对话历史）

    ⚠️ 危险操作，不可恢复！
    """
    global bm25_retriever, retriever
    try:
        # 1. 清空向量库
        if vector_store:
            vector_store.clear()
            print("[OK] Vector store cleared")

        # 2. 清空 SQLite（documents + conversations）
        clear_all_data()
        knowledge_base_version = bump_knowledge_base_version()
        print("[OK] SQLite cleared")

        # 3. 重置 BM25
        bm25_retriever = None
        if retriever:
            retriever.bm25_retriever = None
        print("[OK] BM25 reset")

        return {"status": "success", "message": "数据库已清空", "knowledge_base_version": f"kb-v{knowledge_base_version}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
def get_stats():
    """获取知识库统计信息"""
    settings = get_settings()
    docs = list_documents()

    return {
        "total_documents_in_store": vector_store.count() if vector_store else 0,
        "total_files_uploaded": len(docs),
        "collection_name": settings.vectorstore_collection,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "vectorstore_provider": settings.vectorstore_provider,
        "knowledge_base_version": f"kb-v{get_knowledge_base_version()}"
    }
