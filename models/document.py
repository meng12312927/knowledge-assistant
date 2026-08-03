"""
共享数据模型定义。

所有模块共享的数据结构，使用 Pydantic 进行类型校验。
Pydantic 的优势：自动类型转换、数据验证、JSON 序列化。
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """支持的文档类型"""
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    MD = "markdown"
    HTML = "html"
    TXT = "txt"
    CSV = "csv"


class DocumentChunk(BaseModel):
    """
    文本分块后的单元

    每个 chunk 包含：
    - content: 实际文本内容
    - metadata: 来源信息（文件名、页码、章节等）
    - embedding: 向量化后的结果（可选，离线阶段生成）
    """
    content: str = Field(description="文本块内容")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据：文件名、页码、章节、 chunk 索引等"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="向量表示（由 Embedding 模型生成）"
    )
    model_config = ConfigDict(from_attributes=True)


class SourceDocument(BaseModel):
    """
    原始文档的元信息

    记录文档的基本信息，用于溯源和展示。
    """
    doc_id: str = Field(description="文档唯一标识（通常用 UUID 或文件哈希）")
    filename: str = Field(description="原始文件名")
    doc_type: DocumentType = Field(description="文档类型")
    file_path: str = Field(description="文件存储路径")
    total_pages: Optional[int] = Field(default=None, description="总页数（如适用）")
    created_at: datetime = Field(default_factory=datetime.now, description="入库时间")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="额外元数据：作者、创建日期、标签等"
    )


class RetrievedChunk(DocumentChunk):
    """
    检索返回的文本块（继承自 DocumentChunk）

    相比普通 chunk，多了相似度分数，用于排序和阈值过滤。
    """
    score: float = Field(description="相似度分数（由向量检索或重排序模型给出）")
    rank: Optional[int] = Field(default=None, description="重排序后的位次")


class QueryRequest(BaseModel):
    """
    用户查询请求

    这是进入系统的入口数据结构。
    """
    query: str = Field(min_length=1, max_length=4000, description="用户原始问题")
    session_id: Optional[str] = Field(default=None, description="对话 session ID，用于多轮对话")
    top_k: int = Field(default=5, description="召回文档数量", ge=1, le=20)
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="元数据过滤条件，如 {'doc_type': 'pdf', 'author': '张三'}"
    )
    enable_agent: bool = Field(
        default=False,
        description="API 显式强制 Agent；前端不暴露此开关，默认由服务端自动路由",
    )


class Citation(BaseModel):
    """可验证的结构化引用：引用编号映射到唯一 chunk 和原文。"""
    citation_id: str = Field(pattern=r"^S[1-9]\d*$", description="回答中使用的引用编号，如 S1")
    chunk_id: str = Field(description="向量库中的唯一文本块 ID")
    doc_id: Optional[str] = Field(default=None, description="原始文档内容 ID")
    source_file: str = Field(description="原始文件名")
    page_number: Optional[int | str] = Field(default=None, description="页码或段落序号")
    content: str = Field(description="该引用对应的原始文本块")
    score: float = Field(description="最终检索或重排分数")
    subquestion_ids: List[str] = Field(
        default_factory=list,
        description="该证据覆盖的原子子问题编号",
    )


class CitationVerificationItem(BaseModel):
    """单条答案结论与引用证据之间的核验结果。"""
    claim_id: Optional[str] = Field(
        default=None, description="程序预切分的结论编号，如 C1"
    )
    claim: str = Field(description="从答案中识别出的事实结论")
    subquestion_id: Optional[str] = Field(
        default=None,
        description="该结论对应的子问题编号，如 SQ1",
    )
    citation_ids: List[str] = Field(default_factory=list, description="该结论引用的证据编号")
    verdict: str = Field(description="supported / partial / unsupported / uncited")
    reason: str = Field(default="", description="核验理由")


class CitationVerification(BaseModel):
    """答案级 Citation Verification 结果。"""
    status: str = Field(
        description="verified / partially_verified / failed / unverified / skipped"
    )
    items: List[CitationVerificationItem] = Field(default_factory=list)
    invalid_citation_ids: List[str] = Field(default_factory=list)
    uncited_claims: List[str] = Field(default_factory=list)
    total_claims: int = Field(default=0, ge=0)
    supported_claims: int = Field(default=0, ge=0)
    claim_coverage_rate: Optional[float] = Field(default=None, ge=0, le=1)
    message: str = Field(default="")


class RetrievalRankItem(BaseModel):
    """单个召回或融合通道中的 Chunk 排名。"""
    query: str
    chunk_id: str
    stable_chunk_id: Optional[str] = None
    rank: int = Field(ge=1)
    score: float
    source_file: Optional[str] = None


class RerankTraceItem(BaseModel):
    """最终后处理阶段的排名和分数。"""
    chunk_id: str
    stable_chunk_id: Optional[str] = None
    rank: int = Field(ge=1)
    score: float
    source_file: Optional[str] = None


class TraceSpan(BaseModel):
    """相对于请求开始时间的阶段级追踪记录。"""
    name: str
    start_offset_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    status: str = "ok"
    attributes: Dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    """一次完整问答中所有 LLM 调用的 Token 汇总。"""
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    reranker_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class SubquestionTrace(BaseModel):
    """复合问题中一个原子子问题的检索、覆盖和回答状态。"""
    subquestion_id: str = Field(pattern=r"^SQ[1-9]\d*$")
    query: str
    status: str = Field(
        description="answerable / low_confidence / not_found / conflict"
    )
    selected_chunk_ids: List[str] = Field(default_factory=list)
    selected_stable_chunk_ids: List[str] = Field(default_factory=list)
    source_files: List[str] = Field(default_factory=list)
    top_score: Optional[float] = None
    covered: bool = False
    status_reason: str = ""


class RAGTrace(BaseModel):
    """一次 Query 从改写到引用核验的完整可观测轨迹。"""
    query_variants: List[str] = Field(default_factory=list)
    dense_rankings: List[RetrievalRankItem] = Field(default_factory=list)
    bm25_rankings: List[RetrievalRankItem] = Field(default_factory=list)
    rrf_rankings: List[RetrievalRankItem] = Field(default_factory=list)
    candidate_rankings: List[RerankTraceItem] = Field(
        default_factory=list,
        description="MultiQuery 融合后的最终候选排名，位于远程 Rerank 之前",
    )
    rerank_rankings: List[RerankTraceItem] = Field(default_factory=list)
    selected_chunk_ids: List[str] = Field(default_factory=list)
    citation_map: Dict[str, str] = Field(default_factory=dict)
    citation_verification: Optional[CitationVerification] = None
    subquestion_planning_triggered: bool = False
    subquestions: List[SubquestionTrace] = Field(default_factory=list)
    evidence_coverage: Optional[float] = Field(default=None, ge=0, le=1)
    spans: List[TraceSpan] = Field(default_factory=list)
    query_strategy: str = "adaptive"
    multiquery_triggered: bool = False
    multiquery_reason: str = ""
    initial_retrieval_top_score: Optional[float] = None
    final_retrieval_top_score: Optional[float] = None
    retrieval_quality: str = "unknown"
    routing_probe_strategy: str = ""
    routing_probe_multiquery_triggered: bool = False
    agent_decision: bool = False
    agent_reason: str = ""
    cache_hits: Dict[str, bool] = Field(default_factory=dict)
    cache_stats: Dict[str, int] = Field(default_factory=dict)
    knowledge_base_version: str = "kb-v1"
    document_versions: Dict[str, int] = Field(default_factory=dict)
    # 兼容旧客户端：ttft_ms 始终等于用户可见首 Token 时间。
    ttft_ms: Optional[int] = Field(default=None, ge=0)
    generation_ttft_ms: Optional[int] = Field(default=None, ge=0)
    generation_first_token_at_ms: Optional[int] = Field(default=None, ge=0)
    verified_ttft_ms: Optional[int] = Field(default=None, ge=0)
    user_visible_ttft_ms: Optional[int] = Field(default=None, ge=0)
    sse_total_latency_ms: Optional[int] = Field(default=None, ge=0)
    stage_token_usage: Dict[str, TokenUsage] = Field(default_factory=dict)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    total_latency_ms: Optional[int] = Field(default=None, ge=0)


class ChatResponse(BaseModel):
    """
    系统回答

    包含：生成的答案、引用的来源、处理耗时等。
    """
    answer: str = Field(description="LLM 生成的回答")
    sources: List[RetrievedChunk] = Field(
        default_factory=list,
        description="回答所引用的文档片段（用于溯源）"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="回答中的 [S1] 等编号到 chunk_id 和原文的结构化映射"
    )
    citation_verification: Optional[CitationVerification] = Field(
        default=None,
        description="答案结论与引用原文的一致性核验结果"
    )
    trace: Optional[RAGTrace] = Field(default=None, description="六阶段 RAG 全流程追踪")
    query_time_ms: Optional[int] = Field(default=None, description="查询耗时（毫秒）")
    session_id: Optional[str] = Field(default=None, description="对话 session ID")
    tool_results: Optional[List[dict]] = Field(default=None, description="工具调用结果（Agent 模式）")
    answer_status: str = Field(
        default="answerable",
        description="回答状态: answerable / partially_answerable / low_confidence / not_found / conflict"
    )
