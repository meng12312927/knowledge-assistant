"""
应用核心配置。

统一管理项目的所有配置项，支持从环境变量和 .env 文件加载。
使用 Pydantic Settings 进行类型校验和默认值管理。
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    应用配置类

    优先级（从高到低）：
    1. 环境变量
    2. .env 文件
    3. 默认值
    """

    # === LLM 配置 ===
    llm_temperature: float = 0.3

    # === 多 LLM Provider 路由配置 ===
    # 默认 Provider；支持 deepseek / dashscope
    llm_default_provider: str = "deepseek"

    # DeepSeek（OpenAI-compatible）
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    # 阿里云百炼 / DashScope（OpenAI-compatible）
    dashscope_api_key: Optional[str] = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-plus"
    # 多模型路由：primary 负责正式回答，fast 负责查询改写等轻任务，
    # fallback 在 primary 调用失败时自动接管。留空则分别继承 primary。
    llm_primary_provider: Optional[str] = None
    llm_fast_provider: Optional[str] = None
    llm_fallback_provider: Optional[str] = None
    llm_max_retries: int = 2
    llm_retry_base_seconds: float = 0.5
    llm_timeout_seconds: float = 20.0
    query_rewrite_timeout_seconds: float = 2.0
    generation_timeout_seconds: float = 20.0
    citation_verification_timeout_seconds: float = 5.0

    # === Embedding 配置 ===
    embedding_provider: str = "dashscope"
    embedding_model: str = "text-embedding-v4"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 10
    embedding_max_retries: int = 2
    embedding_timeout_seconds: float = 3.0

    # === Query 优化 ===
    adaptive_multiquery_enabled: bool = True
    query_rewrite_cache_size: int = 512
    query_embedding_cache_size: int = 512
    simple_query_min_rrf_score: float = 0.025
    retrieval_parallel_workers: int = 8

    # === Reranker 配置 ===
    reranker_provider: str = "qwen3"
    reranker_model: str = "qwen3-rerank"
    reranker_base_url: str = "https://dashscope.aliyuncs.com/compatible-api/v1"
    reranker_candidate_k: int = 40
    reranker_top_n: int = 6
    reranker_not_found_threshold: float = 0.30
    reranker_timeout_seconds: float = 4.0
    reranker_max_retries: int = 2
    reranker_instruct: str = "Given a web search query, retrieve relevant passages that answer the query."

    # === 向量数据库配置 ===
    vectorstore_provider: str = "chroma"
    vectorstore_collection: str = "documents_v4_1024"
    chroma_persist_dir: str = "./chroma_db"

    # === RAG 配置 ===
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 6  # Qwen3 Rerank 后进入上下文的最终数量
    score_threshold: float = 0.5
    max_context_tokens: int = 5000  # Top 5-8 高质量证据即可，避免无效上下文 Token

    # === 回答置信度阈值（基于 RRF 融合分数） ===
    # high:  最高分 >= 此值 → answerable（正常回答）
    # low:   最高分 < 此值 → not_found（拒答）；[low, high) → low_confidence（谨慎回答）
    answer_status_threshold_high: float = 0.025
    answer_status_threshold_low: float = 0.012

    # === 引用核验 ===
    citation_verification_enabled: bool = True
    citation_verification_strict: bool = True

    # === 外部依赖可靠性 ===
    request_timeout_seconds: float = 30.0
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_recovery_seconds: float = 30.0

    # === Agent 配置 ===
    enable_agent: bool = True
    agent_router_type: str = "langgraph"

    # === LangChain 配置 ===
    llm_client_type: str = "openai"  # openai / langchain

    # === CORS 配置 ===
    cors_origins: Optional[str] = "http://localhost:8501,http://127.0.0.1:8501"  # 逗号分隔，生产环境应限制具体域名

    # === 应用配置 ===
    app_name: str = "Employee Policy Assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    max_upload_size_mb: int = 20
    allowed_upload_extensions: str = "pdf,docx,pptx,txt,md,html"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置实例（单例模式，带缓存）

    使用 lru_cache 确保配置只加载一次，提升性能。
    """
    return Settings()
