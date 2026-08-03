"""
RAG 主链组装。

把前面所有组件串联成完整的问答流程：
    用户问题 → 查询分析 → 混合检索 → 重排序 → 上下文组装 → LLM 生成 → 带溯源的回答

核心接口：
    chain = RAGChain(
        embedder=embedding_client,
        retriever=hybrid_retriever,
        reranker=cross_encoder_reranker,
        llm=llm_client
    )
    response = chain.invoke("如何申请退款？")
"""

import inspect
import json
import threading
import re
import time
import unicodedata
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

from models.document import (
    Citation,
    QueryRequest,
    ChatResponse,
    RAGTrace,
    RetrievalRankItem,
    RerankTraceItem,
    TokenUsage,
    TraceSpan,
    RetrievedChunk,
    SubquestionTrace,
)
from embeddings.factory import BaseEmbeddingClient
from rag.chunk_identity import stable_chunk_id
from rag.retrievers.hybrid import HybridRetriever
from rag.post_processors.reranker import BaseReranker
from rag.reliability import bounded_timeout


@dataclass
class _SingleFlight:
    """A shared in-flight result for concurrent identical requests."""

    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


@dataclass
class PreparedRAGRequest:
    """Reusable retrieval and rerank result produced before route selection."""

    request_started: float
    trace: RAGTrace
    candidates: List[RetrievedChunk]
    ranked: List[RetrievedChunk]
    status: str
    rerank_metadata: Dict[str, Any]
    subquestion_results: List["_SubquestionResult"] = field(default_factory=list)


@dataclass
class _SubquestionResult:
    """Internal evidence bundle for one atomic subquestion."""

    subquestion_id: str
    query: str
    candidates: List[RetrievedChunk]
    ranked: List[RetrievedChunk]
    status: str
    rerank_metadata: Dict[str, Any] = field(default_factory=dict)
    status_reason: str = ""


class LLMClient:
    """
    LLM 客户端（简化版，实际项目中可扩展为工厂模式）

    封装 OpenAI / Azure / 本地模型等不同的 LLM 调用方式。
    这里以 OpenAI 为例。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model
        self._usage_local = threading.local()

        # LLM 客户端初始化完成

        client_kwargs = {"api_key": self.api_key, "timeout": timeout}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        # Reliability wrapper owns retry/backoff and Trace metadata. Disable
        # SDK retries so one slow upstream call cannot be multiplied invisibly.
        self.client = openai.OpenAI(**client_kwargs, max_retries=0)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = None,
        response_format: Optional[Dict[str, str]] = None,
        thinking: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """调用 LLM 生成文本（非流式）"""
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        if thinking is not None:
            kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if thinking else "disabled"}
            }
        client = self.client.with_options(timeout=timeout) if timeout else self.client
        response = client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        self._usage_local.last_usage = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        self._usage_local.last_metadata = {
            "upstream_request_id": str(getattr(response, "id", "") or "") or None,
        }
        return response.choices[0].message.content

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = None,
        thinking: Optional[bool] = None,
        timeout: Optional[float] = None,
    ):
        """调用 LLM 生成文本（流式），yield 文本片段"""
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if thinking is not None:
            kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if thinking else "disabled"}
            }
        client = self.client.with_options(timeout=timeout) if timeout else self.client
        response = client.chat.completions.create(**kwargs)
        usage_result = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for chunk in response:
            usage = getattr(chunk, "usage", None)
            if usage:
                usage_result = {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                }
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        self._usage_local.last_usage = usage_result
        self._usage_local.last_metadata = {
            "upstream_request_id": str(getattr(response, "_request_id", "") or "") or None,
        }

    def get_last_usage(self) -> dict:
        return getattr(self._usage_local, "last_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    def get_last_metadata(self) -> dict:
        return dict(getattr(self._usage_local, "last_metadata", {}))


class LangChainLLMClient:
    """
    LangChain 版 LLM 客户端

    基于 langchain_openai.ChatOpenAI + LCEL，与 LLMClient 保持相同接口。
    提供原生流式、回调钩子、结构化输出等能力。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        timeout: float = 60.0,
    ):
        from langchain_openai import ChatOpenAI

        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model
        self.temperature = temperature
        self._usage_local = threading.local()

        kwargs = {
            "api_key": self.api_key,
            "model": self.model,
            "temperature": self.temperature,
            "timeout": timeout,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self.llm = ChatOpenAI(**kwargs)

    @staticmethod
    def _escape_braces(text: str) -> str:
        """转义花括号，避免被 ChatPromptTemplate 解析为变量占位符"""
        return text.replace("{", "{{").replace("}", "}}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = None,
        response_format: Optional[Dict[str, str]] = None,
        thinking: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """调用 LLM 生成文本（非流式）"""
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", self._escape_braces(system_prompt)),
            ("user", self._escape_braces(user_prompt)),
        ])
        bind_kwargs = {"temperature": temperature}
        if max_tokens is not None:
            bind_kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            bind_kwargs["response_format"] = response_format
        if thinking is not None:
            bind_kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if thinking else "disabled"}
            }
        if timeout is not None:
            bind_kwargs["timeout"] = timeout
        chain = prompt | self.llm.bind(**bind_kwargs)
        response = chain.invoke({})
        usage = getattr(response, "usage_metadata", None) or {}
        self._usage_local.last_usage = {
            "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
            "completion_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        response_metadata = getattr(response, "response_metadata", None) or {}
        self._usage_local.last_metadata = {
            "upstream_request_id": (
                response_metadata.get("request_id")
                or response_metadata.get("id")
            ),
        }
        return response.content

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = None,
        thinking: Optional[bool] = None,
        timeout: Optional[float] = None,
    ):
        """调用 LLM 生成文本（流式），yield 文本片段"""
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", self._escape_braces(system_prompt)),
            ("user", self._escape_braces(user_prompt)),
        ])
        bind_kwargs = {"temperature": temperature}
        if max_tokens is not None:
            bind_kwargs["max_tokens"] = max_tokens
        if thinking is not None:
            bind_kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if thinking else "disabled"}
            }
        if timeout is not None:
            bind_kwargs["timeout"] = timeout
        chain = prompt | self.llm.bind(**bind_kwargs)
        usage_result = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for chunk in chain.stream({}):
            usage = getattr(chunk, "usage_metadata", None) or {}
            if usage:
                usage_result = {
                    "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("output_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                }
            if chunk.content:
                yield chunk.content
        self._usage_local.last_usage = usage_result

    def get_last_usage(self) -> dict:
        return getattr(self._usage_local, "last_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    def get_last_metadata(self) -> dict:
        return dict(getattr(self._usage_local, "last_metadata", {}))


class RAGChain:
    """
    RAG 问答链

    这是整个 RAG 层的核心编排类，协调 Embedding、检索、重排、生成四个阶段。
    """

    # 系统提示词：约束 LLM 的行为
    SYSTEM_PROMPT = """你是一个严谨的企业制度与员工手册问答助手，擅长跨制度检索、比较和归纳。

回答规则：
1. 【忠实检索】你只能根据提供的「参考资料」回答问题，禁止编造知识库中没有的信息。
2. 【跨文档边界】命中多个文件不代表必须跨文档推理。只有用户明确要求比较、
   原因或综合分析时才建立文件间关系；普通制度问答只提取与问题直接相关的条款。
3. 【原子化结论】每个要点只表达一个原文直接支持的事实，并在该句末尾紧邻引用。
   不得添加原文没有写明的建议、目的、原因、效果、优先级或推导结论。
4. 【术语保真】涉及数据等级、适用范围、金额、比例、期限、数量和例外条件时，
   必须沿用原文完整措辞，不得增删“以上、以下、以内、超过、至少、仅限”等
   限定词，也不得用看似等价的术语改写。
5. 【制度边界】区分公司制度、审批例外和法律事项；资料要求咨询 HR 或个案处理时，不得擅自给出确定结论。
6. 【引用规范】每个事实结论后必须引用参考资料编号，格式仅允许 `[S1]`、`[S2]`；禁止编造不存在的编号。
7. 【边界判定】只有当参考资料完全与问题无关时，才回答："根据现有知识库，无法找到相关信息。"
8. 【禁止缺失推断】不得因为参考资料没有提到某事项，就声称“没有例外”“不存在其他规定”或作出类似负面结论；只陈述原文直接支持的内容。
9. 【条件代入】原文明示必要资格、金额区间或适用范围时，可以把用户给出的明确条件直接代入判断，但必须同时引用包含该条件的原文。
10. 【禁止擅自合并制度】不得声称某种员工状态“不影响”、某专项制度“统一适用”或两套制度可以叠加，除非原文明示这种关系；关系不明确时只分别陈述可确认的条款。
11. 【禁止无依据的包装】不要添加“可以回答”“要求一致”“建议咨询/确认”“请以实际版本为准”等没有引用的开场、总结或建议；标题只写主题名，不写事实判断。
"""

    # 低置信度时的系统提示词：要求 LLM 更谨慎
    SYSTEM_PROMPT_LOW_CONFIDENCE = """你是一个基于知识库的问答助手。请注意：本次查询检索到的参考资料相关性较低，可能无法完整回答问题。

回答规则：
1. 【谨慎回答】你只能根据提供的「参考资料」回答，如果资料不足以支撑结论，必须明确说明"根据现有资料，该问题无法完全确认"。
2. 【部分回答】如果资料只能回答问题的某一部分，请说明"以下回答仅基于有限资料，可能不全面"。
3. 【不编造】禁止编造知识库中没有的信息，禁止为了给出完整答案而进行合理推测。
4. 【引用规范】每个有资料支撑的结论后必须引用参考资料编号，如 `[S1]`；禁止编造不存在的编号。
5. 【边界判定】如果参考资料完全与问题无关，直接回答："根据现有知识库，无法找到相关信息。"
6. 【禁止缺失推断】不得把“资料未提及”解释为“不存在”；只陈述原文直接支持的内容。
"""

    SYSTEM_PROMPT_PARTIAL = """你是一个严谨的企业制度问答助手。本次复合问题只有部分子问题找到了可靠证据。

回答规则：
1. 只回答标记为 answerable 或 low_confidence 的子问题，每条事实句末紧邻 `[Sx]` 引用。
2. 对标记为 not_found 的子问题，逐项说明“根据现有知识库无法确认”，禁止推测或补全。
3. 使用“可以确认”和“暂无法确认”两个小节，不能把未找到信息表述为制度明确不存在。
4. 每个要点只表达一个事实；金额、期限、范围、否定和例外条件必须沿用原文。
5. 禁止使用未提供的引用编号，禁止让一个引用支持超出原文范围的合并结论。
6. 不得用“未提及”推断两套制度互不影响或自动叠加；只把该关系列入“暂无法确认”。
"""

    def __init__(
        self,
        embedder: BaseEmbeddingClient,
        retriever: HybridRetriever,
        reranker: BaseReranker,
        llm: LLMClient,
        citation_verifier=None,
        knowledge_base_version_getter=None,
        max_context_tokens: int = 4000,
        answer_status_threshold_high: float = 0.6,
        answer_status_threshold_low: float = 0.3,
        adaptive_multiquery_enabled: bool = True,
        query_rewrite_cache_size: int = 512,
        query_embedding_cache_size: int = 512,
        simple_query_min_rrf_score: float = 0.031,
        retrieval_candidate_k: int = 40,
        rerank_top_n: int = 6,
        reranker_not_found_threshold: float = 0.50,
        subquestion_planning_enabled: bool = True,
        subquestion_max_count: int = 4,
        subquestion_rerank_top_n: int = 3,
        subquestion_rerank_candidate_k: int = 24,
        evidence_per_source_limit: int = 2,
        query_rewrite_timeout_seconds: float = 2.0,
        embedding_timeout_seconds: float = 3.0,
        reranker_timeout_seconds: float = 4.0,
        generation_timeout_seconds: float = 20.0,
        citation_verification_timeout_seconds: float = 5.0,
    ):
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.citation_verifier = citation_verifier
        self.knowledge_base_version_getter = knowledge_base_version_getter
        self.max_context_tokens = max_context_tokens
        self.answer_status_threshold_high = answer_status_threshold_high
        self.answer_status_threshold_low = answer_status_threshold_low
        self.adaptive_multiquery_enabled = adaptive_multiquery_enabled
        self.query_rewrite_cache_size = max(0, query_rewrite_cache_size)
        self.query_embedding_cache_size = max(0, query_embedding_cache_size)
        self.simple_query_min_rrf_score = simple_query_min_rrf_score
        self.retrieval_candidate_k = max(1, retrieval_candidate_k)
        self.rerank_top_n = min(8, max(1, rerank_top_n))
        self.reranker_not_found_threshold = max(0.0, reranker_not_found_threshold)
        self.subquestion_planning_enabled = bool(subquestion_planning_enabled)
        self.subquestion_max_count = min(6, max(2, int(subquestion_max_count)))
        self.subquestion_rerank_top_n = min(
            6, max(2, int(subquestion_rerank_top_n))
        )
        self.subquestion_rerank_candidate_k = min(
            self.retrieval_candidate_k,
            max(self.subquestion_rerank_top_n, int(subquestion_rerank_candidate_k)),
        )
        self.evidence_per_source_limit = max(
            1, int(evidence_per_source_limit)
        )
        self.query_rewrite_timeout_seconds = max(
            0.001, float(query_rewrite_timeout_seconds)
        )
        self.embedding_timeout_seconds = max(
            0.001, float(embedding_timeout_seconds)
        )
        self.reranker_timeout_seconds = max(
            0.001, float(reranker_timeout_seconds)
        )
        self.generation_timeout_seconds = max(
            0.001, float(generation_timeout_seconds)
        )
        self.citation_verification_timeout_seconds = max(
            0.001, float(citation_verification_timeout_seconds)
        )
        self._rewrite_cache: OrderedDict[str, Tuple[str, ...]] = OrderedDict()
        self._embedding_cache: OrderedDict[Tuple[str, int, str], Tuple[float, ...]] = OrderedDict()
        self._rewrite_inflight: Dict[str, _SingleFlight] = {}
        self._embedding_inflight: Dict[Tuple[str, int, str], _SingleFlight] = {}
        self._cache_lock = threading.RLock()
        self._thread_local = threading.local()

    def _generate_query_variants(
        self, query: str, n: int = 3
    ) -> Tuple[List[str], bool, bool]:
        """
        MultiQuery：返回 (variants, cache_hit, singleflight_wait)。

        并发相同 Query 只允许一个 leader 调用改写模型，其余请求等待并复用结果。
        仅成功结果进入显式有界缓存，避免暂时性模型错误被永久缓存。

        用 LLM 生成同一问题的不同表述，覆盖更多关键词和语义角度。
        """
        normalized = self._normalize_query(query)
        cache_key = f"v2:{n}:{normalized}"
        with self._cache_lock:
            cached = self._rewrite_cache.get(cache_key)
            if cached is not None:
                self._rewrite_cache.move_to_end(cache_key)
                return list(cached), True, False
            flight = self._rewrite_inflight.get(cache_key)
            leader = flight is None
            if leader:
                flight = _SingleFlight()
                self._rewrite_inflight[cache_key] = flight

        if not leader:
            wait_started = time.perf_counter()
            completed = flight.event.wait(
                timeout=bounded_timeout(self.query_rewrite_timeout_seconds)
            )
            self._thread_local.rewrite_queue_time_ms = int(
                (time.perf_counter() - wait_started) * 1000
            )
            if not completed:
                return [query], False, True
            if flight.error:
                return [query], False, True
            return list(flight.result or [query]), True, True

        prompt = f"""基于以下用户问题，生成 {n} 个不同表述的查询变体。
要求：
1. 保持原意不变
2. 使用不同的关键词和句式
3. 每个变体单独一行，不要编号
4. 不要添加任何解释

原始问题：{query}

变体："""
        result = [query]
        error = None
        try:
            response = self.llm.generate(
                system_prompt="你是一个查询改写专家。",
                user_prompt=prompt,
                temperature=0.5,
                # 查询改写只需要三行短文本，限制输出可减少推理 token 和等待时间。
                max_tokens=160,
                thinking=False,
                stage="query_rewrite",
                timeout=self.query_rewrite_timeout_seconds,
            )
            variants = [line.strip() for line in response.split('\n') if line.strip()]
            # 加入原始查询，去重
            all_queries = [query] + variants[:n]
            result = list(dict.fromkeys(all_queries))
            if len(result) > 1 and self.query_rewrite_cache_size:
                with self._cache_lock:
                    self._rewrite_cache[cache_key] = tuple(result)
                    self._rewrite_cache.move_to_end(cache_key)
                    while len(self._rewrite_cache) > self.query_rewrite_cache_size:
                        self._rewrite_cache.popitem(last=False)
        except Exception as e:
            print(f"[MultiQuery] variant generation failed: {e}, using original query")
            error = e
        finally:
            with self._cache_lock:
                flight.result = tuple(result)
                flight.error = error
                self._rewrite_inflight.pop(cache_key, None)
                flight.event.set()
        return result, False, False

    @staticmethod
    def _normalize_query(query: str) -> str:
        normalized = unicodedata.normalize("NFKC", query or "")
        return re.sub(r"\s+", " ", normalized).strip()

    @classmethod
    def _classify_simple_query(cls, query: str) -> Tuple[bool, str]:
        """规则路由：金额、比例、时长、制度名和条款编号等精确查询优先直检。"""
        text = cls._normalize_query(query)
        complex_pattern = r"比较|区别|差异|分别|综合|为什么|原因|影响|冲突|以及|同时|并且|如何结合"
        if re.search(complex_pattern, text):
            return False, "complex_or_multi_intent"
        exact_patterns = {
            "amount_or_number": r"\d+(?:\.\d+)?\s*(?:元|万元|%|％|天|小时|分钟|个月|年)",
            "quantitative_question": (
                r"(?:多少|几)\s*(?:元|万元|%|％|天|小时|分钟|个月|年|次|个)?"
                r"|多久|多长时间|何时|什么时候"
            ),
            "policy_name": r"《[^》]{2,}》|[\u4e00-\u9fff]{2,}(?:制度|办法|规定|手册|流程)",
            "article_or_id": r"第[一二三四五六七八九十百千万\d]+条|(?:编号|制度号|条款|文号)\s*[:：]?\s*[A-Za-z0-9_-]+",
        }
        for reason, pattern in exact_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return True, reason
        return False, "fuzzy_query"

    @classmethod
    def _needs_subquestion_planning(cls, query: str) -> bool:
        """Only decompose explicit compound questions; simple queries stay fast."""
        text = cls._normalize_query(query)
        explicit_multi = re.search(
            r"分别|以及|同时|并且|各自|两者|三者|比较|对比|有什么不同|"
            r"(?:还|又).*(?:吗|如何|什么)|[？?].+[？?]",
            text,
        )
        enumerated = len(re.findall(r"[、；;]", text)) >= 1
        repeated_question = len(
            re.findall(r"(?:如何|哪些|什么|多少|多久|是否|吗)[？?]?", text)
        ) >= 2
        if explicit_multi or enumerated or repeated_question:
            return True
        simple, _ = cls._classify_simple_query(text)
        if not simple and re.search(
            r"(?:和|与|且).*(?:标准|流程|审批|规定|风险|评估|要求|处理|怎么算|是否)",
            text,
        ):
            return True
        domains = re.findall(
            r"远程办公|差旅|住宿|餐补|采购|合同|预算|审计|供应商|"
            r"信息安全|数据|绩效|培训|认证|利益冲突|项目",
            text,
        )
        return len(set(domains)) >= 2 and len(text) >= 24

    def _rule_decompose_query(self, query: str) -> List[str]:
        """Conservative fallback when the planner times out or returns bad JSON."""
        text = self._normalize_query(query).strip("？?。 ")
        # Sentence boundaries are safe; conjunction splitting is only enabled for
        # explicit `分别`, avoiding accidental splits such as “公司和员工”.
        parts = [
            part.strip("，,、；;。？? ")
            for part in re.split(r"[？?；;]", text)
            if part.strip("，,、；;。？? ")
        ]
        if len(parts) < 2 and "、" in text:
            tail_match = re.search(
                r"(是什么|有什么规定|有哪些规定|需要哪些审批|如何处理)$",
                text,
            )
            tail = tail_match.group(1) if tail_match else "是什么"
            body = text[: tail_match.start()] if tail_match else text
            first_boundary = body.find("、")
            left = body[:first_boundary]
            prefix_boundary = left.rfind("的")
            if prefix_boundary >= 0:
                prefix = left[: prefix_boundary + 1]
                first_item = left[prefix_boundary + 1:]
                item_text = first_item + "、" + body[first_boundary + 1:]
                items = [
                    item.strip()
                    for item in re.split(r"、|和|与", item_text)
                    if item.strip()
                ]
                parts = [f"{prefix}{item}{tail}" for item in items]
        if len(parts) < 2 and "分别" in text:
            head = text.replace("分别", "")
            parts = [
                part.strip("，,、；;。？? ")
                for part in re.split(r"、|以及|并且|和|与", head)
                if part.strip("，,、；;。？? ")
            ]
        if len(parts) < 2:
            domains = list(dict.fromkeys(re.findall(
                r"远程办公|差旅|出差|住宿|餐补|采购|合同|预算|审计|供应商|"
                r"信息安全|数据|绩效|培训|认证|利益冲突|项目",
                text,
            )))
            if len(domains) >= 2:
                domains = ["差旅" if value == "出差" else value for value in domains]
                domains = list(dict.fromkeys(domains))
                if (
                    "远程办公" in domains
                    and "绩效" in domains
                    and re.search(r"绩效.{0,8}符合预期", text)
                ):
                    domains.remove("绩效")
                parts = [
                    f"{domain}相关制度中，与“{text}”直接相关的规定是什么"
                    for domain in domains[: self.subquestion_max_count - 1]
                ]
                if re.search(r"同时|叠加|所有制度|既.+又|但.+(?:出差|差旅)", text):
                    parts.append(
                        f"{domains[0]}与{domains[1]}在该场景下如何叠加或冲突"
                    )
        unique = list(dict.fromkeys(part for part in parts if len(part) >= 3))
        return unique[: self.subquestion_max_count]

    @classmethod
    def _enrich_policy_coverage_queries(
        cls, original_query: str, questions: List[str]
    ) -> List[str]:
        """Add policy-native facets to broad "all requirements/process" branches.

        The planner still decides *which* atomic branches exist.  This method
        only expands an already-requested broad policy branch with the known
        chapter vocabulary, so reranking does not collapse onto one chapter.
        """
        original = cls._normalize_query(original_query)
        broad = bool(re.search(r"所有|全部|哪些|流程|要求|怎样|如何", original))
        if not broad:
            return questions
        enriched: List[str] = []
        for question in questions:
            value = question
            if (
                re.search(r"出差|差旅", value)
                and not re.search(r"交通|住宿|餐补|票据", value)
            ):
                value += "，包括交通、住宿、餐补和票据要求"
            if (
                "采购" in value
                and re.search(r"流程|审批|规定|要求", value)
                and not re.search(r"询价|比选|供应商准入", value)
            ):
                value += "，包括申请审批、询价比选和供应商准入"
            enriched.append(value)
        return list(dict.fromkeys(enriched))

    def _plan_subquestions(
        self,
        query: str,
        trace: RAGTrace,
        request_started: float,
    ) -> List[str]:
        """Use the fast LLM once to turn a compound query into atomic queries."""
        started = time.perf_counter()
        usage_before = self._usage_snapshot(self.llm)
        questions: List[str] = []
        error = None
        try:
            raw = self.llm.generate(
                system_prompt=(
                    "你是企业制度检索的查询拆分器，只拆分用户明确提出的独立问题。"
                ),
                user_prompt=f"""把下面的复合问题拆成 2 到 {self.subquestion_max_count} 个可独立检索的原子子问题。

要求：
1. 每个子问题补全共享主语、场景和条件，单独阅读也完整。
2. 不新增用户未提出的条件，不回答问题。
3. 金额、期限、否定词和例外条件必须原样保留。
4. 按用户原问题中的出现顺序输出。
5. 当问题叠加两个制度并询问“同时、如何执行、所有要求”时，先分别拆出每个制度可独立查证的事实，再把两者的叠加关系作为最后一个子问题。
6. 例如“远程办公日出差，住宿怎么算、能否同时领补贴”应拆成“出差住宿标准”“差旅餐补标准”“两种状态同日时是否有额外或重复补贴”。
7. 只输出紧凑 JSON：{{"subquestions":["...","..."]}}

用户问题：{query}
""",
                temperature=0,
                max_tokens=320,
                response_format={"type": "json_object"},
                thinking=False,
                stage="subquestion_planning",
                timeout=self.query_rewrite_timeout_seconds,
            )
            payload = json.loads(raw)
            values = payload.get("subquestions") or []
            questions = [
                self._normalize_query(str(value))
                for value in values
                if self._normalize_query(str(value))
            ]
            questions = list(dict.fromkeys(questions))[: self.subquestion_max_count]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            questions = self._rule_decompose_query(query)

        questions = self._enrich_policy_coverage_queries(query, questions)
        if len(questions) < 2:
            questions = []
        self._record_stage_usage(
            trace,
            "subquestion_planning",
            usage_before,
            self._usage_snapshot(self.llm),
        )
        self._record_span(
            trace,
            "subquestion_planning",
            request_started,
            started,
            {
                "triggered": True,
                "subquestion_count": len(questions),
                "fallback": bool(error),
                "error": error,
                **self._latest_component_event(
                    self.llm, "subquestion_planning"
                ),
            },
        )
        trace.subquestion_planning_triggered = bool(questions)
        return questions

    @staticmethod
    def _record_span(
        trace: RAGTrace,
        name: str,
        request_started: float,
        span_started: float,
        attributes: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[int] = None,
    ) -> TraceSpan:
        span = TraceSpan(
            name=name,
            start_offset_ms=max(0, int((span_started - request_started) * 1000)),
            duration_ms=(
                max(0, int((time.perf_counter() - span_started) * 1000))
                if duration_ms is None else max(0, int(duration_ms))
            ),
            attributes=attributes or {},
        )
        trace.spans.append(span)
        return span

    @staticmethod
    def _usage_snapshot(component: Any) -> Dict[str, int]:
        if not hasattr(component, "request_usage"):
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        usage = component.request_usage() or {}
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }

    @staticmethod
    def _latest_component_event(component: Any, stage: str) -> Dict[str, Any]:
        if not hasattr(component, "request_events"):
            return {}
        events = component.request_events() or []
        for event in reversed(events):
            if event.get("stage") == stage:
                return dict(event)
        return {}

    @staticmethod
    def _record_stage_usage(
        trace: RAGTrace,
        name: str,
        before: Dict[str, int],
        after: Dict[str, int],
        *,
        reranker_tokens: int = 0,
        accumulate: bool = False,
    ) -> None:
        prompt_tokens = max(
            0, int(after.get("prompt_tokens", 0)) - int(before.get("prompt_tokens", 0))
        )
        completion_tokens = max(
            0,
            int(after.get("completion_tokens", 0))
            - int(before.get("completion_tokens", 0)),
        )
        total_tokens = max(
            0, int(after.get("total_tokens", 0)) - int(before.get("total_tokens", 0))
        )
        reranker_value = max(0, int(reranker_tokens))
        if accumulate and name in trace.stage_token_usage:
            existing = trace.stage_token_usage[name]
            prompt_tokens += existing.prompt_tokens
            completion_tokens += existing.completion_tokens
            reranker_value += existing.reranker_tokens
            total_tokens += existing.total_tokens - existing.reranker_tokens
        trace.stage_token_usage[name] = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reranker_tokens=reranker_value,
            total_tokens=total_tokens + reranker_value,
        )

    def _embed_queries(self, queries: List[str]) -> Tuple[List[List[float]], Dict[str, int]]:
        """只缓存在线 Query 向量；文档摄取仍直接调用 Embedder，不污染缓存。"""
        if not queries:
            return [], {
                "cache_hits": 0,
                "cache_misses": 0,
                "singleflight_waits": 0,
                "queue_time_ms": 0,
            }
        model_name = str(getattr(self.embedder, "model_name", "unknown"))
        dimension = int(getattr(self.embedder, "dimension", 0))
        keys = [(model_name, dimension, self._normalize_query(query)) for query in queries]
        results: List[Optional[List[float]]] = [None] * len(queries)
        leader_indices: Dict[Tuple[str, int, str], List[int]] = defaultdict(list)
        waiting: Dict[Tuple[str, int, str], Tuple[_SingleFlight, List[int]]] = {}
        hit_count = 0
        miss_count = 0
        with self._cache_lock:
            for index, key in enumerate(keys):
                cached = self._embedding_cache.get(key)
                if cached is None:
                    miss_count += 1
                    flight = self._embedding_inflight.get(key)
                    if flight is None:
                        flight = _SingleFlight()
                        self._embedding_inflight[key] = flight
                        leader_indices[key].append(index)
                    elif key in leader_indices:
                        leader_indices[key].append(index)
                    else:
                        existing = waiting.get(key)
                        if existing:
                            existing[1].append(index)
                        else:
                            waiting[key] = (flight, [index])
                else:
                    self._embedding_cache.move_to_end(key)
                    results[index] = list(cached)
                    hit_count += 1

        leader_keys = list(leader_indices)
        try:
            if leader_keys:
                embed_kwargs = {}
                if "timeout" in inspect.signature(self.embedder.embed).parameters:
                    embed_kwargs["timeout"] = self.embedding_timeout_seconds
                vectors = self.embedder.embed(
                    [key[2] for key in leader_keys],
                    **embed_kwargs,
                )
                if len(vectors) != len(leader_keys):
                    raise RuntimeError("Query Embedding 返回数量与请求不一致")
                for key, vector in zip(leader_keys, vectors):
                    immutable = tuple(float(value) for value in vector)
                    for index in leader_indices[key]:
                        results[index] = list(immutable)
                    with self._cache_lock:
                        flight = self._embedding_inflight.pop(key)
                        flight.result = immutable
                        self._embedding_cache[key] = immutable
                        self._embedding_cache.move_to_end(key)
                        while len(self._embedding_cache) > self.query_embedding_cache_size:
                            self._embedding_cache.popitem(last=False)
                        flight.event.set()
        except BaseException as exc:
            with self._cache_lock:
                for key in leader_keys:
                    flight = self._embedding_inflight.pop(key, None)
                    if flight:
                        flight.error = exc
                        flight.event.set()
            raise

        for flight, indices in waiting.values():
            wait_started = time.perf_counter()
            completed = flight.event.wait(
                timeout=bounded_timeout(self.embedding_timeout_seconds)
            )
            queue_ms = int((time.perf_counter() - wait_started) * 1000)
            if not completed:
                raise TimeoutError("Query Embedding single-flight wait timed out")
            if flight.error:
                raise RuntimeError("并发 Query Embedding 失败") from flight.error
            vector = list(flight.result or ())
            for index in indices:
                results[index] = vector

        return [vector or [] for vector in results], {
            "cache_hits": hit_count,
            "cache_misses": miss_count,
            "singleflight_waits": sum(len(indices) for _, indices in waiting.values()),
            "queue_time_ms": locals().get("queue_ms", 0),
        }

    def _retrieve_query_batch(
        self,
        queries: List[str],
        embeddings: List[List[float]],
        filters: Optional[Dict[str, Any]],
        trace: RAGTrace,
        request_started: float,
        span_name: str,
    ) -> List[List[RetrievedChunk]]:
        stage_started = time.perf_counter()
        outputs = self.retriever.retrieve_many(
            queries=queries,
            query_embeddings=embeddings,
            top_k=self.retrieval_candidate_k,
            filter_dict=filters,
            return_trace=True,
        )
        parent = self._record_span(
            trace, span_name, request_started, stage_started,
            attributes={"query_count": len(queries), "parallel": True},
        )
        batches: List[List[RetrievedChunk]] = []
        for query_index, (query, output) in enumerate(zip(queries, outputs)):
            batch, route_trace = output
            batches.append(batch)
            self._append_route_trace(trace, query, route_trace)
            timing = route_trace.get("timing", {})
            for route in ("dense", "bm25"):
                start_delta = int(timing.get(f"{route}_start_ms", 0))
                duration = int(timing.get(f"{route}_duration_ms", 0))
                trace.spans.append(TraceSpan(
                    name=f"{route}_retrieval",
                    start_offset_ms=parent.start_offset_ms + max(0, start_delta),
                    duration_ms=max(0, duration),
                    attributes={
                        "query_index": query_index,
                        "query": query,
                        "parallel_group": span_name,
                    },
                ))
        return batches

    @classmethod
    def _merge_query_batches(
        cls,
        batches: List[List[RetrievedChunk]],
        limit: int,
    ) -> List[RetrievedChunk]:
        """对多个 Query 的 RRF 结果再做一次稳定 Query-RRF，避免先到者偏置。"""
        if not batches:
            return []
        if len(batches) == 1:
            return list(batches[0][:limit])
        scores: Dict[str, float] = defaultdict(float)
        chunks: Dict[str, RetrievedChunk] = {}
        for batch in batches:
            for rank, chunk in enumerate(batch, 1):
                chunk_id = cls._chunk_id(chunk)
                scores[chunk_id] += 1.0 / (60 + rank)
                chunks.setdefault(chunk_id, chunk)
        ranked_ids = sorted(scores, key=lambda key: (-scores[key], key))[:limit]
        merged = []
        for chunk_id in ranked_ids:
            source = chunks[chunk_id]
            metadata = dict(source.metadata)
            metadata["route_rrf_score"] = float(source.score)
            merged.append(RetrievedChunk(
                content=source.content,
                metadata=metadata,
                score=float(scores[chunk_id]),
            ))
        return merged

    def _adaptive_retrieve(
        self,
        query_request: QueryRequest,
        trace: RAGTrace,
        request_started: float,
        allow_multiquery: bool = True,
    ) -> List[RetrievedChunk]:
        classify_started = time.perf_counter()
        embedding_usage_before = self._usage_snapshot(self.embedder)
        simple, classification_reason = self._classify_simple_query(query_request.query)
        self._record_span(
            trace, "query_classification", request_started, classify_started,
            {"simple": simple, "reason": classification_reason},
        )

        total_embedding_hits = 0
        total_embedding_misses = 0
        total_embedding_singleflight_waits = 0
        rewrite_cache_hit = False
        rewrite_singleflight_wait = False
        rewrite_attempted = False

        def embed_and_trace(queries: List[str], name: str):
            nonlocal total_embedding_hits, total_embedding_misses
            nonlocal total_embedding_singleflight_waits
            started = time.perf_counter()
            vectors, stats = self._embed_queries(queries)
            total_embedding_hits += stats["cache_hits"]
            total_embedding_misses += stats["cache_misses"]
            total_embedding_singleflight_waits += stats["singleflight_waits"]
            self._record_span(
                trace, name, request_started, started,
                {
                    **stats,
                    "cache_hit_all": bool(queries) and stats["cache_misses"] == 0,
                    "mode": "cache+batch",
                    "query_count": len(queries),
                    **(
                        self.embedder.get_last_metadata()
                        if stats["cache_misses"]
                        and hasattr(self.embedder, "get_last_metadata")
                        else {}
                    ),
                },
            )
            return vectors

        # Retrieval-first：所有 Query 都先执行一次原始检索，复杂意图不再直接
        # 触发 MultiQuery。只有原始召回不足时才升级。
        original_embeddings = embed_and_trace(
            [query_request.query], "embedding.original"
        )
        original_batches = self._retrieve_query_batch(
            [query_request.query],
            original_embeddings,
            query_request.filters,
            trace,
            request_started,
            "retrieval.original",
        )
        original_candidates = self._merge_query_batches(
            original_batches, self.retrieval_candidate_k
        )
        trace.initial_retrieval_top_score = (
            float(original_candidates[0].score) if original_candidates else None
        )
        dense_top_source = (
            trace.dense_rankings[0].source_file
            if trace.dense_rankings
            else None
        )
        bm25_top_source = (
            trace.bm25_rankings[0].source_file
            if trace.bm25_rankings
            else None
        )
        channel_top_source_agreement = bool(
            dense_top_source
            and bm25_top_source
            and dense_top_source == bm25_top_source
        )
        sufficient = bool(
            original_candidates
            and original_candidates[0].score >= self.simple_query_min_rrf_score
            and (simple or channel_top_source_agreement)
        )
        classification_span = next(
            (
                span
                for span in trace.spans
                if span.name == "query_classification"
            ),
            None,
        )
        if classification_span:
            classification_span.attributes.update(
                {
                    "dense_top_source": dense_top_source,
                    "bm25_top_source": bm25_top_source,
                    "channel_top_source_agreement": (
                        channel_top_source_agreement
                    ),
                }
            )

        if (
            sufficient
            or not self.adaptive_multiquery_enabled
            or not allow_multiquery
        ):
            trace.query_strategy = "direct"
            trace.multiquery_reason = (
                "original_retrieval_sufficient"
                if sufficient
                else (
                    "multiquery_disabled"
                    if not self.adaptive_multiquery_enabled
                    else "agent_intent_skips_multiquery"
                )
            )
            trace.query_variants = [query_request.query]
            now = time.perf_counter()
            self._record_span(
                trace,
                "query_rewrite",
                request_started,
                now,
                {
                    "skipped": True,
                    "reason": trace.multiquery_reason,
                    "cache_hit": False,
                    "singleflight_wait": False,
                    "queue_time_ms": 0,
                    "retry_count": 0,
                    "timeout_ms": int(
                        self.query_rewrite_timeout_seconds * 1000
                    ),
                    "circuit_state": "skipped",
                },
                duration_ms=0,
            )
            candidates = original_candidates
        else:
            rewrite_attempted = True
            trace.query_strategy = "adaptive_fallback"
            trace.multiquery_triggered = True
            trace.multiquery_reason = "original_retrieval_insufficient"
            rewrite_started = time.perf_counter()
            rewrite_usage_before = self._usage_snapshot(self.llm)
            (
                variants,
                rewrite_cache_hit,
                rewrite_singleflight_wait,
            ) = self._generate_query_variants(query_request.query, n=3)
            self._record_stage_usage(
                trace,
                "query_rewrite",
                rewrite_usage_before,
                self._usage_snapshot(self.llm),
            )
            self._record_span(
                trace, "query_rewrite", request_started, rewrite_started,
                {
                    "skipped": False,
                    "reason": trace.multiquery_reason,
                    "cache_hit": rewrite_cache_hit,
                    "singleflight_wait": rewrite_singleflight_wait,
                    "queue_time_ms": (
                        getattr(self._thread_local, "rewrite_queue_time_ms", 0)
                        if rewrite_singleflight_wait else 0
                    ),
                    **self._latest_component_event(
                        self.llm, "query_rewrite"
                    ),
                },
            )
            normalized_original = self._normalize_query(query_request.query)
            extra_queries = [
                query
                for query in variants
                if self._normalize_query(query) != normalized_original
            ]
            extra_batches = []
            if extra_queries:
                extra_embeddings = embed_and_trace(
                    extra_queries, "embedding.multiquery"
                )
                extra_batches = self._retrieve_query_batch(
                    extra_queries,
                    extra_embeddings,
                    query_request.filters,
                    trace,
                    request_started,
                    "retrieval.multiquery",
                )
            trace.query_variants = [query_request.query] + extra_queries
            candidates = self._merge_query_batches(
                original_batches + extra_batches, self.retrieval_candidate_k
            )

        trace.cache_hits = {
            "query_rewrite": bool(rewrite_attempted and rewrite_cache_hit),
            "query_embedding": total_embedding_misses == 0 and total_embedding_hits > 0,
        }
        trace.cache_stats = {
            "embedding_hits": total_embedding_hits,
            "embedding_misses": total_embedding_misses,
            "embedding_singleflight_waits": total_embedding_singleflight_waits,
            "rewrite_singleflight_waits": int(rewrite_singleflight_wait),
        }
        trace.final_retrieval_top_score = (
            float(candidates[0].score) if candidates else None
        )
        trace.routing_probe_strategy = trace.query_strategy
        trace.routing_probe_multiquery_triggered = (
            trace.multiquery_triggered
        )
        trace.stage_token_usage.setdefault("query_rewrite", TokenUsage())
        self._record_stage_usage(
            trace,
            "embedding",
            embedding_usage_before,
            self._usage_snapshot(self.embedder),
        )
        trace.candidate_rankings = [
            RerankTraceItem(
                chunk_id=self._chunk_id(chunk),
                stable_chunk_id=stable_chunk_id(chunk.content, chunk.metadata),
                rank=index,
                score=float(chunk.score),
                source_file=chunk.metadata.get("source_file"),
            )
            for index, chunk in enumerate(candidates, 1)
        ]
        return candidates

    @staticmethod
    def _canonical_source(source_file: str) -> str:
        return re.sub(r"_v\d+(?=\.[^.]+$|$)", "", source_file or "")

    def _prefer_latest_document_versions(
        self, chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """Do not mix obsolete and current versions of the same policy."""
        latest: Dict[str, int] = {}
        for chunk in chunks:
            source = self._canonical_source(
                str(chunk.metadata.get("source_file") or "")
            )
            version = int(chunk.metadata.get("document_version") or 1)
            latest[source] = max(latest.get(source, 0), version)
        return [
            chunk
            for chunk in chunks
            if int(chunk.metadata.get("document_version") or 1)
            >= latest.get(
                self._canonical_source(
                    str(chunk.metadata.get("source_file") or "")
                ),
                1,
            )
        ]

    def _retrieve_and_rerank_subquestions(
        self,
        questions: List[str],
        query_request: QueryRequest,
        trace: RAGTrace,
        request_started: float,
    ) -> List[_SubquestionResult]:
        """Batch-embed and parallel-retrieve, then parallel-rerank each branch."""
        embedding_started = time.perf_counter()
        embedding_before = self._usage_snapshot(self.embedder)
        embeddings, cache_stats = self._embed_queries(questions)
        self._record_stage_usage(
            trace,
            "embedding",
            embedding_before,
            self._usage_snapshot(self.embedder),
            accumulate=True,
        )
        self._record_span(
            trace,
            "embedding.subquestions",
            request_started,
            embedding_started,
            {
                **cache_stats,
                "query_count": len(questions),
                "mode": "cache+batch",
            },
        )
        batches = self._retrieve_query_batch(
            questions,
            embeddings,
            query_request.filters,
            trace,
            request_started,
            "retrieval.subquestions",
        )
        candidate_batches = [
            self._prefer_latest_document_versions(
                self._merge_query_batches([batch], self.retrieval_candidate_k)
            )
            for batch in batches
        ]

        def rerank_one(index: int) -> _SubquestionResult:
            query = questions[index]
            candidates = candidate_batches[index][
                : self.subquestion_rerank_candidate_k
            ]
            status = self._judge_answer_status(candidates)
            ranked = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_n=self.subquestion_rerank_top_n,
                timeout=self.reranker_timeout_seconds,
            )
            metadata = (
                self.reranker.get_last_metadata()
                if hasattr(self.reranker, "get_last_metadata")
                else {}
            )
            ranked = self._fuse_subquestion_anchors(
                query, ranked, candidates, self.subquestion_rerank_top_n
            )
            status = self._refine_status_with_reranker(
                status, ranked, metadata
            )
            return _SubquestionResult(
                subquestion_id=f"SQ{index + 1}",
                query=query,
                candidates=candidates,
                ranked=ranked,
                status=status,
                rerank_metadata=metadata,
            )

        rerank_started = time.perf_counter()
        results: List[Optional[_SubquestionResult]] = [None] * len(questions)
        workers = min(4, len(questions))
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="subquestion-rerank",
        ) as pool:
            futures = {
                pool.submit(rerank_one, index): index
                for index in range(len(questions))
            }
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
        resolved = [result for result in results if result is not None]
        reranker_tokens = sum(
            int((result.rerank_metadata.get("usage") or {}).get("total_tokens", 0) or 0)
            for result in resolved
        )
        self._record_stage_usage(
            trace,
            "rerank",
            {},
            {},
            reranker_tokens=reranker_tokens,
        )
        self._record_span(
            trace,
            "rerank.subquestions",
            request_started,
            rerank_started,
            {
                "parallel": True,
                "branch_count": len(resolved),
                "input_count": sum(len(item.candidates) for item in resolved),
                "output_count": sum(len(item.ranked) for item in resolved),
                "fallback_count": sum(
                    bool(item.rerank_metadata.get("fallback"))
                    for item in resolved
                ),
                "upstream_request_ids": [
                    item.rerank_metadata.get("request_id")
                    for item in resolved
                    if item.rerank_metadata.get("request_id")
                ],
            },
        )
        return resolved

    @staticmethod
    def _fuse_subquestion_anchors(
        query: str,
        reranked: List[RetrievedChunk],
        candidates: List[RetrievedChunk],
        top_n: int,
    ) -> List[RetrievedChunk]:
        """Keep remote leaders and restore source/facet anchors from RRF."""
        if not reranked:
            return candidates[:top_n]
        fused: List[RetrievedChunk] = []
        seen_ids: Set[str] = set()
        seen_sources: Set[str] = set()

        def add(chunk: RetrievedChunk) -> None:
            chunk_id = str(chunk.metadata.get("chunk_id") or stable_chunk_id(
                chunk.content, chunk.metadata
            ))
            if chunk_id in seen_ids or len(fused) >= top_n:
                return
            fused.append(chunk)
            seen_ids.add(chunk_id)
            seen_sources.add(str(chunk.metadata.get("source_file") or ""))

        for chunk in reranked[:2]:
            add(chunk)

        # A broad branch often names several policy chapters. Remote rerankers
        # may concentrate all slots on one facet, so explicitly restore one RRF
        # anchor for each requested facet before generic source diversification.
        facet_groups = [
            (("申请审批",), ("采购申请",)),
            (("询价比选",), ("询价", "比选")),
            (("供应商准入",), ("供应商准入",)),
            (("交通", "住宿"), ("交通", "住宿")),
            (("餐补", "票据"), ("餐补", "票据")),
        ]
        for query_terms, evidence_terms in facet_groups:
            if not any(term in query for term in query_terms):
                continue
            if any(
                any(term in chunk.content for term in evidence_terms)
                for chunk in fused
            ):
                continue
            anchor = next(
                (
                    chunk for chunk in candidates[:16]
                    if any(term in chunk.content for term in evidence_terms)
                ),
                None,
            )
            if anchor is not None:
                add(anchor)
        for chunk in candidates[:8]:
            source = str(chunk.metadata.get("source_file") or "")
            if source not in seen_sources:
                add(chunk)
            if len(fused) >= min(top_n, 4):
                break
        for chunk in reranked[2:]:
            add(chunk)
        for chunk in candidates:
            add(chunk)
        for index, chunk in enumerate(fused, 1):
            chunk.rank = index
        return fused

    @staticmethod
    def _aggregate_subquestion_status(
        results: List[_SubquestionResult],
    ) -> str:
        if not results or all(item.status == "not_found" for item in results):
            return "not_found"
        if any(item.status == "conflict" for item in results):
            return "conflict"
        covered = [item for item in results if item.status != "not_found"]
        if len(covered) < len(results):
            return "partially_answerable"
        if any(item.status == "low_confidence" for item in results):
            return "low_confidence"
        return "answerable"

    def _judge_subquestion_evidence(
        self,
        results: List[_SubquestionResult],
        trace: RAGTrace,
        request_started: float,
    ) -> List[_SubquestionResult]:
        """Distinguish topical relevance from evidence that directly answers an SQ."""
        if not results:
            return results
        started = time.perf_counter()
        usage_before = self._usage_snapshot(self.llm)
        blocks = []
        for result in results:
            evidence = "\n".join(
                f"E{index}: {chunk.content}"
                for index, chunk in enumerate(result.ranked[:3], 1)
            ) or "（无证据）"
            blocks.append(
                f"[{result.subquestion_id}] {result.query}\n{evidence}"
            )
        error = None
        try:
            raw = self.llm.generate(
                system_prompt=(
                    "你是严格的企业制度证据覆盖分类器，只判断证据是否直接回答问题。"
                ),
                user_prompt=f"""逐个判断下面原子子问题是否被其候选证据直接回答。

{chr(10).join(blocks)}

规则：
1. answerable：证据直接给出了问题所问的制度事实、条件、数值或处理方式；也包括把证据中的明确必要条件、区间阈值或适用范围直接代入用户已给条件即可得到结论的情况。
2. not_found：只有主题相关内容、相邻规定、示例，或没有给出问题所问的具体信息。
3. 不得使用外部知识；“制度未提及”不能推断为否定答案。
4. 金额边界按证据中的数学区间判断，例如“不超过8000元”覆盖5001元；不得要求原文逐字出现该具体金额。
5. 资格判断中，证据明确要求“转正”等必要条件时，可以直接判断试用期员工不满足该条件。
6. 若问题询问时间/比例/评分标准，证据仍必须明确包含对应信息；仅列评估维度不等于给出评分标准。
7. 通用制度直接规定某类采购、差旅或员工的流程时，可覆盖该类别下的具体对象；例如“采购申请”的通用审批规则可直接回答“采购新系统”的审批，不要求原文逐字出现“新系统”。
8. 只输出 JSON：{{"results":[{{"id":"SQ1","status":"answerable","reason":"不超过15字"}}]}}
""",
                temperature=0,
                max_tokens=400,
                response_format={"type": "json_object"},
                thinking=False,
                stage="subquestion_evidence_judgment",
                timeout=min(
                    4.0, self.citation_verification_timeout_seconds
                ),
            )
            payload = json.loads(raw)
            decisions = {
                str(item.get("id")): item
                for item in payload.get("results") or []
                if item.get("id")
            }
            for result in results:
                decision = decisions.get(result.subquestion_id) or {}
                decided_status = str(decision.get("status") or "")
                if decided_status in {"answerable", "not_found", "low_confidence", "conflict"}:
                    result.status = decided_status
                    result.status_reason = str(decision.get("reason") or "")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            # Preserve calibrated Reranker/RRF status on classifier failure.
        self._record_stage_usage(
            trace,
            "subquestion_evidence_judgment",
            usage_before,
            self._usage_snapshot(self.llm),
        )
        self._record_span(
            trace,
            "subquestion_evidence_judgment",
            request_started,
            started,
            {
                "branch_count": len(results),
                "answerable_count": sum(
                    item.status == "answerable" for item in results
                ),
                "not_found_count": sum(
                    item.status == "not_found" for item in results
                ),
                "fallback": bool(error),
                "error": error,
                **self._latest_component_event(
                    self.llm, "subquestion_evidence_judgment"
                ),
            },
        )
        return results

    def _select_evidence_coverage(
        self,
        results: List[_SubquestionResult],
    ) -> List[RetrievedChunk]:
        """Coverage-first selection with branch, source and content diversity."""
        selected: List[RetrievedChunk] = []
        selected_by_id: Dict[str, RetrievedChunk] = {}
        source_counts: Dict[str, int] = defaultdict(int)

        normalized_contents: List[Set[str]] = []

        def content_terms(content: str) -> Set[str]:
            return set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]+", content.lower()))

        def add(chunk: RetrievedChunk, subquestion_id: str, enforce_cap: bool) -> bool:
            chunk_id = self._chunk_id(chunk) or stable_chunk_id(
                chunk.content, chunk.metadata
            )
            existing = selected_by_id.get(chunk_id)
            if existing is not None:
                ids = list(existing.metadata.get("subquestion_ids") or [])
                if subquestion_id not in ids:
                    ids.append(subquestion_id)
                existing.metadata["subquestion_ids"] = ids
                return False
            source = str(chunk.metadata.get("source_file") or "未知文件")
            if enforce_cap and source_counts[source] >= self.evidence_per_source_limit:
                return False
            terms = content_terms(chunk.content)
            if enforce_cap and terms and any(
                len(terms & existing) / max(1, min(len(terms), len(existing))) >= 0.85
                for existing in normalized_contents
            ):
                return False
            copied = chunk.model_copy(deep=True)
            copied.metadata["subquestion_ids"] = [subquestion_id]
            selected.append(copied)
            selected_by_id[chunk_id] = copied
            source_counts[source] += 1
            normalized_contents.append(terms)
            return True

        # Coverage pass ignores the per-source cap so no answerable branch is lost.
        for result in results:
            if result.status != "not_found" and result.ranked:
                add(result.ranked[0], result.subquestion_id, enforce_cap=False)

        # Fill remaining slots round-robin. This keeps second-hop evidence from
        # a lower-scoring branch instead of letting one easy branch occupy all
        # remaining context slots.
        max_rank = max((len(item.ranked) for item in results), default=0)
        for branch_rank in range(1, max_rank):
            if len(selected) >= self.rerank_top_n:
                break
            ordered = sorted(
                (
                    result for result in results
                    if result.status != "not_found"
                    and branch_rank < len(result.ranked)
                ),
                key=lambda item: float(item.ranked[branch_rank].score),
                reverse=True,
            )
            for result in ordered:
                if len(selected) >= self.rerank_top_n:
                    break
                add(
                    result.ranked[branch_rank],
                    result.subquestion_id,
                    enforce_cap=True,
                )
        return selected[: self.rerank_top_n]

    def _populate_subquestion_trace(
        self,
        trace: RAGTrace,
        results: List[_SubquestionResult],
        selected: List[RetrievedChunk],
    ) -> None:
        selected_ids_by_sq: Dict[str, List[str]] = defaultdict(list)
        selected_stable_by_sq: Dict[str, List[str]] = defaultdict(list)
        selected_sources_by_sq: Dict[str, Set[str]] = defaultdict(set)
        for chunk in selected:
            for subquestion_id in chunk.metadata.get("subquestion_ids") or []:
                selected_ids_by_sq[subquestion_id].append(self._chunk_id(chunk))
                selected_stable_by_sq[subquestion_id].append(
                    stable_chunk_id(chunk.content, chunk.metadata)
                )
                selected_sources_by_sq[subquestion_id].add(
                    str(chunk.metadata.get("source_file") or "未知文件")
                )
        trace.subquestions = [
            SubquestionTrace(
                subquestion_id=result.subquestion_id,
                query=result.query,
                status=result.status,
                selected_chunk_ids=selected_ids_by_sq[result.subquestion_id],
                selected_stable_chunk_ids=selected_stable_by_sq[result.subquestion_id],
                source_files=sorted(selected_sources_by_sq[result.subquestion_id]),
                top_score=(
                    float(result.ranked[0].score) if result.ranked else None
                ),
                covered=bool(selected_ids_by_sq[result.subquestion_id]),
                status_reason=result.status_reason,
            )
            for result in results
        ]
        trace.evidence_coverage = (
            sum(item.covered for item in trace.subquestions)
            / len(trace.subquestions)
            if trace.subquestions
            else None
        )

    def prepare(
        self,
        query_request: QueryRequest,
        *,
        allow_multiquery: bool = True,
    ) -> PreparedRAGRequest:
        """Run retrieval-first and rerank once so routing and RAG can reuse it."""
        request_started = time.perf_counter()
        trace = RAGTrace()
        should_plan = bool(
            self.subquestion_planning_enabled
            and self._needs_subquestion_planning(query_request.query)
        )
        candidates = self._adaptive_retrieve(
            query_request,
            trace,
            request_started,
            # Atomic subqueries replace paraphrase MultiQuery for explicit
            # compound questions, avoiding two expansion LLM calls.
            allow_multiquery=allow_multiquery and not should_plan,
        )
        subquestion_results: List[_SubquestionResult] = []
        rerank_metadata: Dict[str, Any] = {}
        if should_plan:
            trace.multiquery_reason = "subquestion_planning_replaces_multiquery"
            questions = self._plan_subquestions(
                query_request.query, trace, request_started
            )
            if questions:
                subquestion_results = self._retrieve_and_rerank_subquestions(
                    questions, query_request, trace, request_started
                )
                subquestion_results = self._judge_subquestion_evidence(
                    subquestion_results, trace, request_started
                )

        if subquestion_results:
            ranked = self._select_evidence_coverage(subquestion_results)
            status = self._aggregate_subquestion_status(subquestion_results)
            self._populate_subquestion_trace(
                trace, subquestion_results, ranked
            )
            trace.query_strategy = "subquestion_coverage"
            trace.multiquery_triggered = False
            rerank_metadata = {
                "provider": "multi-branch",
                "branch_count": len(subquestion_results),
                "fallback": any(
                    item.rerank_metadata.get("fallback")
                    for item in subquestion_results
                ),
            }
            # Candidate ranking becomes coverage-aware so Recall@K measures the
            # actual decomposed route rather than only the original-query probe.
            coverage_candidates = list(ranked)
            seen = {
                self._chunk_id(chunk) or stable_chunk_id(chunk.content, chunk.metadata)
                for chunk in coverage_candidates
            }
            max_branch_candidates = max(
                (len(result.candidates) for result in subquestion_results),
                default=0,
            )
            # Round-robin keeps the first K diagnostic candidates balanced
            # across SQ branches; appending one whole branch first hides the
            # evidence of later branches and depresses Recall@K artificially.
            for branch_rank in range(max_branch_candidates):
                for result in subquestion_results:
                    if branch_rank >= len(result.candidates):
                        continue
                    chunk = result.candidates[branch_rank]
                    key = self._chunk_id(chunk) or stable_chunk_id(
                        chunk.content, chunk.metadata
                    )
                    if key not in seen:
                        coverage_candidates.append(chunk)
                        seen.add(key)
            trace.candidate_rankings = [
                RerankTraceItem(
                    chunk_id=self._chunk_id(chunk),
                    stable_chunk_id=stable_chunk_id(
                        chunk.content, chunk.metadata
                    ),
                    rank=index,
                    score=float(chunk.score),
                    source_file=chunk.metadata.get("source_file"),
                )
                for index, chunk in enumerate(coverage_candidates, 1)
            ]
        else:
            if should_plan:
                now = time.perf_counter()
                trace.subquestion_planning_triggered = False
                trace.query_strategy = "direct"
                trace.multiquery_reason = "subquestion_planning_failed_direct_fallback"
            else:
                trace.stage_token_usage.setdefault(
                    "subquestion_planning", TokenUsage()
                )
            status = self._judge_answer_status(candidates)
            stage_started = time.perf_counter()
            ranked = self.reranker.rerank(
                query=query_request.query,
                candidates=self._cascade_rerank_candidates(
                    query_request.query, candidates, trace, request_started
                ),
                top_n=min(query_request.top_k, self.rerank_top_n),
                timeout=self.reranker_timeout_seconds,
            )
            rerank_metadata = (
                self.reranker.get_last_metadata()
                if hasattr(self.reranker, "get_last_metadata")
                else {}
            )
            reranker_tokens = int(
                (rerank_metadata.get("usage") or {}).get("total_tokens", 0) or 0
            )
            self._record_stage_usage(
                trace,
                "rerank",
                {},
                {},
                reranker_tokens=reranker_tokens,
            )
            status = self._refine_status_with_reranker(
                status, ranked, rerank_metadata
            )
            self._record_span(
                trace,
                "rerank",
                request_started,
                stage_started,
                {
                    **rerank_metadata,
                    "input_count": len(candidates),
                    "output_count": len(ranked),
                },
            )
        trace.rerank_rankings = [
            RerankTraceItem(
                chunk_id=self._chunk_id(chunk),
                stable_chunk_id=stable_chunk_id(
                    chunk.content, chunk.metadata
                ),
                rank=index,
                score=float(chunk.score),
                source_file=chunk.metadata.get("source_file"),
            )
            for index, chunk in enumerate(ranked, 1)
        ]
        if status == "not_found":
            trace.retrieval_quality = "not_found"
        elif status == "partially_answerable":
            trace.retrieval_quality = "partial_coverage"
        elif trace.multiquery_triggered:
            trace.retrieval_quality = "recoverable_low"
        elif status == "low_confidence":
            trace.retrieval_quality = "low_confidence"
        else:
            trace.retrieval_quality = "sufficient"

        return PreparedRAGRequest(
            request_started=request_started,
            trace=trace,
            candidates=candidates,
            ranked=ranked,
            status=status,
            rerank_metadata=rerank_metadata,
            subquestion_results=subquestion_results,
        )

    def invoke(
        self,
        query_request: QueryRequest,
        prepared: Optional[PreparedRAGRequest] = None,
    ) -> ChatResponse:
        """
        执行完整的 RAG 问答流程

        Args:
            query_request: 包含用户查询、top_k、过滤条件等

        Returns:
            ChatResponse: 包含回答、引用来源、耗时等
        """
        prepared = prepared or self.prepare(query_request)
        request_started = prepared.request_started
        trace = prepared.trace
        candidates = prepared.candidates
        ranked = prepared.ranked
        status = prepared.status
        self._thread_local.last_ranked = ranked
        self._thread_local.last_answer_status = status

        # === 阶段 4：上下文压缩与组装 ===
        stage_started = time.perf_counter()
        context, files_in_context, context_chunks = self._build_context(ranked)
        if status == "not_found":
            context, files_in_context, context_chunks = "", set(), []
        if prepared.subquestion_results:
            self._populate_subquestion_trace(
                trace, prepared.subquestion_results, context_chunks
            )
        trace.selected_chunk_ids = [self._chunk_id(chunk) for chunk in context_chunks]
        self._enrich_trace_versions(trace, context_chunks)
        citations = self._build_citations(context_chunks)
        self._record_span(trace, "context", request_started, stage_started, {
            "selected_chunks": len(context_chunks), "context_chars": len(context),
        })

        # === 阶段 5：判断回答置信度状态 + LLM 生成 ===
        stage_started = time.perf_counter()

        generation_usage_before = self._usage_snapshot(self.llm)
        if status == "not_found":
            answer = "根据现有知识库，无法找到与您的提问相关的信息。"
        else:
            user_prompt = self._build_prompt(
                query_request.query,
                context,
                ranked,
                files_in_context,
                trace.subquestions,
            )
            system_prompt = (
                self.SYSTEM_PROMPT_PARTIAL
                if status == "partially_answerable"
                else (
                    self.SYSTEM_PROMPT_LOW_CONFIDENCE
                    if status in {"low_confidence", "conflict"}
                    else self.SYSTEM_PROMPT
                )
            )
            answer = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                thinking=False,
                stage="generation",
                timeout=self.generation_timeout_seconds,
            )
        self._record_stage_usage(
            trace,
            "generation",
            generation_usage_before,
            self._usage_snapshot(self.llm),
        )
        self._record_span(trace, "generation", request_started, stage_started, {
            "stream": False,
            **self._latest_component_event(self.llm, "generation"),
        })
        answer, citations, verification = (
            self._verify_with_optional_citation_repair(
                answer,
                citations,
                status,
                trace,
                request_started,
            )
        )
        self._thread_local.last_citations = citations
        self._thread_local.last_citation_verification = verification
        trace.citation_map = {citation.citation_id: citation.chunk_id for citation in citations}
        trace.citation_verification = verification
        self._thread_local.last_trace = trace

        elapsed = int((time.perf_counter() - request_started) * 1000)
        trace.verified_ttft_ms = elapsed
        trace.user_visible_ttft_ms = elapsed
        trace.ttft_ms = elapsed
        self._apply_trace_metrics(trace, {}, elapsed)
        print(
            f"[RAG] total:{elapsed}ms strategy:{trace.query_strategy} "
            f"rewrite_cache:{trace.cache_hits.get('query_rewrite')} "
            f"embedding_cache:{trace.cache_hits.get('query_embedding')} "
            f"status:{status} variants:{len(trace.query_variants)} "
            f"candidates:{len(candidates)} chunks:{len(ranked)} context_len:{len(context)}"
        )

        return ChatResponse(
            answer=answer,
            sources=[] if status == "not_found" else ranked,
            citations=citations,
            citation_verification=verification,
            trace=trace,
            query_time_ms=elapsed,
            session_id=query_request.session_id,
            answer_status=status
        )

    def stream(
        self,
        query_request: QueryRequest,
        state: Optional[Dict[str, Any]] = None,
        prepared: Optional[PreparedRAGRequest] = None,
    ):
        """流式 RAG；state 显式传回请求结果，避免依赖跨 worker 的 thread-local。"""
        prepared = prepared or self.prepare(query_request)
        stream_started = prepared.request_started
        state = state if state is not None else {}
        trace = prepared.trace
        candidates = prepared.candidates
        ranked = prepared.ranked
        status = prepared.status
        self._thread_local.last_ranked = ranked
        self._thread_local.last_answer_status = status

        stage_started = time.perf_counter()
        context, files_in_context, context_chunks = self._build_context(ranked)
        if prepared.subquestion_results:
            self._populate_subquestion_trace(
                trace, prepared.subquestion_results, context_chunks
            )
        trace.selected_chunk_ids = [self._chunk_id(chunk) for chunk in context_chunks]
        self._enrich_trace_versions(trace, context_chunks)
        citations = self._build_citations(context_chunks)
        self._record_span(trace, "context", stream_started, stage_started, {
            "selected_chunks": len(context_chunks), "context_chars": len(context),
        })

        def publish(verification_value, citation_values, source_values=None):
            self._thread_local.last_citations = citation_values
            self._thread_local.last_citation_verification = verification_value
            self._thread_local.last_trace = trace
            state.update({
                "sources": ranked if source_values is None else source_values,
                "citations": citation_values,
                "citation_verification": verification_value,
                "trace": trace,
                "answer_status": status,
            })

        if status == "not_found":
            answer = "根据现有知识库，无法找到与您的提问相关的信息。"
            now = time.perf_counter()
            self._record_span(
                trace, "generation", stream_started, now,
                {"stream": True, "skipped": True}, duration_ms=0,
            )
            verification = (
                self.citation_verifier.verify(answer, [], status)
                if self.citation_verifier else None
            )
            self._record_span(
                trace, "citation_verification", stream_started, now,
                {"status": "skipped"}, duration_ms=0,
            )
            trace.citation_verification = verification
            trace.stage_token_usage.setdefault("generation", TokenUsage())
            trace.stage_token_usage.setdefault("citation_verification", TokenUsage())
            elapsed = int((time.perf_counter() - stream_started) * 1000)
            trace.verified_ttft_ms = elapsed
            trace.user_visible_ttft_ms = elapsed
            trace.ttft_ms = elapsed
            self._apply_trace_metrics(trace, {}, elapsed)
            publish(verification, [], source_values=[])
            yield answer
            return

        user_prompt = self._build_prompt(
            query_request.query,
            context,
            ranked,
            files_in_context,
            trace.subquestions,
        )
        system_prompt = (
            self.SYSTEM_PROMPT_PARTIAL
            if status == "partially_answerable"
            else (
                self.SYSTEM_PROMPT_LOW_CONFIDENCE
                if status in {"low_confidence", "conflict"}
                else self.SYSTEM_PROMPT
            )
        )
        answer_parts: List[str] = []
        strict_verification = bool(
            self.citation_verifier
            and self.citation_verifier.enabled
            and self.citation_verifier.strict
        )
        generation_started = time.perf_counter()
        generation_usage_before = self._usage_snapshot(self.llm)
        for part in self.llm.generate_stream(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            thinking=False,
            stage="generation",
            timeout=self.generation_timeout_seconds,
        ):
            if trace.generation_ttft_ms is None:
                first_token_at = time.perf_counter()
                trace.generation_ttft_ms = int((first_token_at - generation_started) * 1000)
                trace.generation_first_token_at_ms = int((first_token_at - stream_started) * 1000)
            answer_parts.append(part)
            if not strict_verification:
                if trace.user_visible_ttft_ms is None:
                    trace.user_visible_ttft_ms = int((time.perf_counter() - stream_started) * 1000)
                    trace.ttft_ms = trace.user_visible_ttft_ms
                yield part
        self._record_span(
            trace, "generation", stream_started, generation_started,
            {
                "stream": True,
                "strict_buffered": strict_verification,
                **self._latest_component_event(self.llm, "generation"),
            },
        )
        generated_answer = "".join(answer_parts)
        self._record_stage_usage(
            trace,
            "generation",
            generation_usage_before,
            self._usage_snapshot(self.llm),
        )
        if not generated_answer:
            raise RuntimeError("生成模型返回了空回答")

        final_answer, citations, verification = (
            self._verify_with_optional_citation_repair(
                generated_answer,
                citations,
                status,
                trace,
                stream_started,
            )
        )
        trace.citation_map = {citation.citation_id: citation.chunk_id for citation in citations}
        trace.citation_verification = verification
        trace.verified_ttft_ms = int((time.perf_counter() - stream_started) * 1000)
        if strict_verification:
            trace.user_visible_ttft_ms = trace.verified_ttft_ms
            trace.ttft_ms = trace.user_visible_ttft_ms
        total_ms = int((time.perf_counter() - stream_started) * 1000)
        self._apply_trace_metrics(trace, {}, total_ms)
        publish(verification, citations)
        if strict_verification:
            yield final_answer

    def _judge_answer_status(self, ranked: List[RetrievedChunk]) -> str:
        """
        根据 RRF 融合分数判断回答置信度状态。

        规则（基于 top chunk 的分数）：
        - answerable:    top_score >= high_threshold  → 高置信度，正常回答
        - low_confidence: top_score ∈ [low_threshold, high_threshold) → 谨慎回答
        - not_found:      ranked 为空 或 top_score < low_threshold → 拒答

        Returns:
            "answerable" | "low_confidence" | "not_found"
        """
        if not ranked:
            return "not_found"

        top_score = ranked[0].score
        if top_score >= self.answer_status_threshold_high:
            return "answerable"
        elif top_score >= self.answer_status_threshold_low:
            return "low_confidence"
        else:
            return "not_found"

    def _refine_status_with_reranker(
        self,
        current_status: str,
        ranked: List[RetrievedChunk],
        metadata: Dict[str, Any],
    ) -> str:
        """只用 Qwen 分数识别明显 OOD；降级时仍沿用原 RRF 置信度。"""
        if (
            ranked
            and metadata.get("provider") == "dashscope"
            and not metadata.get("fallback")
            and float(ranked[0].score) < self.reranker_not_found_threshold
        ):
            return "not_found"
        return current_status

    def _cascade_rerank_candidates(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        trace: RAGTrace,
        request_started: float,
    ) -> List[RetrievedChunk]:
        """Use a smaller remote window for strong atomic retrieval results."""
        if len(candidates) <= 24:
            return candidates
        simple, reason = self._classify_simple_query(query)
        top_score = float(candidates[0].score) if candidates else 0.0
        if simple and top_score >= self.simple_query_min_rrf_score:
            limit = 24 if reason == "quantitative_question" else 20
        else:
            limit = min(32, len(candidates))
        self._record_span(
            trace,
            "rerank.cascade",
            request_started,
            trace_start := time.perf_counter(),
            {
                "input_count": len(candidates),
                "submitted_count": limit,
                "simple": simple,
                "reason": reason,
                "skipped_candidates": len(candidates) - limit,
            },
            duration_ms=0,
        )
        return candidates[:limit]

    def get_last_sources(self):
        """获取当前线程上一次 stream/invoke 的检索结果"""
        return getattr(getattr(self, '_thread_local', None), 'last_ranked', [])

    def get_last_answer_status(self) -> str:
        """获取当前线程上一次 stream/invoke 的回答置信度状态"""
        return getattr(getattr(self, '_thread_local', None), 'last_answer_status', 'answerable')

    def get_last_citations(self) -> List[Citation]:
        """获取当前线程上一次请求实际进入 Prompt 的结构化引用。"""
        return getattr(getattr(self, '_thread_local', None), 'last_citations', [])

    def get_last_citation_verification(self):
        """获取当前线程上一次回答的引用一致性核验结果。"""
        return getattr(getattr(self, '_thread_local', None), 'last_citation_verification', None)

    def get_last_trace(self):
        """获取当前线程上一次请求的六阶段 RAG 轨迹。"""
        return getattr(getattr(self, '_thread_local', None), 'last_trace', None)

    @staticmethod
    def _chunk_id(chunk: RetrievedChunk) -> str:
        return str(chunk.metadata.get("chunk_id") or "")

    @staticmethod
    def _append_route_trace(trace: RAGTrace, query: str, route_trace: dict) -> None:
        target_map = {
            "dense": trace.dense_rankings,
            "bm25": trace.bm25_rankings,
            "rrf": trace.rrf_rankings,
        }
        for route, target in target_map.items():
            for item in route_trace.get(route, []):
                target.append(RetrievalRankItem(
                    query=query,
                    chunk_id=item["chunk_id"],
                    stable_chunk_id=item.get("stable_chunk_id"),
                    rank=item["rank"],
                    score=item["score"],
                    source_file=item.get("source_file"),
                ))

    def _enrich_trace_versions(self, trace: RAGTrace, chunks: List[RetrievedChunk]) -> None:
        version = self.knowledge_base_version_getter() if self.knowledge_base_version_getter else 1
        trace.knowledge_base_version = f"kb-v{version}"
        trace.document_versions = {
            self._chunk_id(chunk): int(chunk.metadata.get("document_version") or 1)
            for chunk in chunks
        }

    def _apply_trace_metrics(self, trace: RAGTrace, stage_times: dict, total_ms: int) -> None:
        # Span 在真实执行点记录；这里只排序，不再伪造成串行累计时间。
        trace.spans.sort(key=lambda span: (span.start_offset_ms, span.name))
        trace.total_latency_ms = total_ms
        llm_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(self.llm, "request_usage"):
            llm_usage.update(self.llm.request_usage())
        reranker_tokens = int(
            getattr(trace.stage_token_usage.get("rerank"), "reranker_tokens", 0)
            or 0
        )
        if not reranker_tokens and hasattr(self.reranker, "get_last_metadata"):
            metadata = self.reranker.get_last_metadata()
            reranker_tokens = int((metadata.get("usage") or {}).get("total_tokens", 0) or 0)
        trace.token_usage = TokenUsage(
            prompt_tokens=int(llm_usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(llm_usage.get("completion_tokens", 0) or 0),
            reranker_tokens=reranker_tokens,
            total_tokens=int(llm_usage.get("total_tokens", 0) or 0) + reranker_tokens,
        )

    def _build_context(self, chunks: List[RetrievedChunk]) -> Tuple[str, Set[str], List[RetrievedChunk]]:
        """
        将检索到的 chunk 组装成上下文字符串

        策略：
        1. 按文件来源分组，确保每个文件至少有一个 chunk 进入上下文
        2. 全局按分数排序，确保截断时保留最高分的 chunk
        3. 累计长度接近上限时截断

        Returns:
            上下文、实际文件集合、实际进入 Prompt 的 chunk 列表
        """
        if not chunks:
            return "", set(), []

        chars_limit = int((self.max_context_tokens - 400) / 1.5)  # 预留 system_prompt + query 的 token 预算

        coverage_aware = any(
            chunk.metadata.get("subquestion_ids") for chunk in chunks
        )
        if coverage_aware:
            # Evidence selector already ordered one representative per SQ first.
            # Re-sorting here would reintroduce the top-score crowding problem.
            selected = list(chunks)
        else:
            file_groups = defaultdict(list)
            for chunk in chunks:
                filename = chunk.metadata.get('source_file', '未知')
                file_groups[filename].append(chunk)
            selected = []
            for group in file_groups.values():
                group.sort(key=lambda x: x.score, reverse=True)
                selected.append(group[0])
            seen = {id(c) for c in selected}
            for chunk in chunks:
                if id(chunk) not in seen:
                    selected.append(chunk)
                    seen.add(id(chunk))
            selected.sort(key=lambda x: x.score, reverse=True)

        # 组装上下文
        context_parts = []
        current_length = 0
        files_in_context = set()

        included_chunks = []
        for chunk in selected:
            filename = chunk.metadata.get('source_file', '未知文件')
            page_raw = chunk.metadata.get('page_number', chunk.metadata.get('page_index', 'N/A'))
            # 处理 0-based page_index
            if isinstance(page_raw, int) and 'page_index' in chunk.metadata and 'page_number' not in chunk.metadata:
                page = page_raw + 1
            else:
                page = page_raw
            citation_id = f"S{len(included_chunks) + 1}"
            source_label = f"[{citation_id}]【{filename} 第{page}页】"

            part = f"\n--- {source_label} ---\n{chunk.content}\n"
            part_length = len(part)

            if current_length + part_length > chars_limit and context_parts:
                break

            context_parts.append(part)
            included_chunks.append(chunk)
            current_length += part_length
            files_in_context.add(filename)

        return "".join(context_parts), files_in_context, included_chunks

    @staticmethod
    def _build_citations(chunks: List[RetrievedChunk]) -> List[Citation]:
        citations = []
        for index, chunk in enumerate(chunks, 1):
            metadata = chunk.metadata or {}
            page = metadata.get("page_number", metadata.get("page_index"))
            if isinstance(page, int) and "page_index" in metadata and "page_number" not in metadata:
                page += 1
            citations.append(Citation(
                citation_id=f"S{index}",
                chunk_id=str(metadata.get("chunk_id") or f"legacy-{index}"),
                doc_id=metadata.get("doc_id"),
                source_file=str(metadata.get("source_file") or "未知文件"),
                page_number=page,
                content=chunk.content,
                score=chunk.score,
                subquestion_ids=list(
                    dict.fromkeys(metadata.get("subquestion_ids") or [])
                ),
            ))
        return citations

    @staticmethod
    def _filter_used_citations(answer: str, citations: List[Citation]) -> List[Citation]:
        """只返回答案实际引用的编号；模型漏引时保留候选证据便于排查。"""
        used_ids = set(re.findall(r"\[(S[1-9]\d*)\]", answer or ""))
        if not used_ids:
            return citations
        return [citation for citation in citations if citation.citation_id in used_ids]

    @staticmethod
    def _needs_citation_repair(verification) -> bool:
        """只修复确定性的引用标记问题，不重写证据不支持的事实。"""
        if not verification or verification.status != "failed":
            return False
        return bool(
            verification.invalid_citation_ids
            or verification.message in {
                "答案没有引用任何结构化证据",
                "答案包含不存在的引用编号",
            }
        )

    @staticmethod
    def _apply_verified_citation_bindings(answer: str, verification) -> str:
        """Insert verifier-approved IDs for claims that generation left uncited."""
        repaired = answer
        for item in verification.items:
            if item.verdict != "supported" or not item.citation_ids or not item.claim:
                continue
            claim_pattern = re.escape(item.claim)
            match = re.search(claim_pattern + r"([。！？；!?;]?)", repaired)
            if not match:
                continue
            tail = repaired[match.end(): match.end() + 32]
            if re.match(r"\s*(?:\[S[1-9]\d*\]\s*)+", tail):
                continue
            marker = "".join(f"[{value}]" for value in item.citation_ids)
            punctuation = match.group(1)
            replacement = item.claim + punctuation + marker
            repaired = repaired[:match.start()] + replacement + repaired[match.end():]
        return repaired

    @staticmethod
    def _normalize_partial_unanswered(answer: str) -> str:
        """Keep unavailable branches as uncited coverage statements, not facts."""
        lines = []
        in_unanswered = False
        for line in (answer or "").splitlines():
            if "暂无法确认" in line and not line.lstrip().startswith(("-", "*")):
                in_unanswered = True
                lines.append(re.sub(r"\[S[1-9]\d*\]", "", line))
                continue
            if in_unanswered and line.lstrip().startswith(("-", "*")):
                cleaned = re.sub(r"\[S[1-9]\d*\]", "", line).strip()
                match = re.search(r"(?:暂)?无法确认", cleaned)
                if match:
                    cleaned = cleaned[: match.end()].rstrip("，,；;。 ") + "。"
                lines.append(cleaned)
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def _repair_citation_markup(
        self,
        answer: str,
        citations: List[Citation],
    ) -> str:
        """最多调用一次快速 LLM，只修复答案中的 ``[Sx]`` 引用标记。"""
        evidence = "\n\n".join(
            f"[{citation.citation_id}] {citation.source_file}\n{citation.content}"
            for citation in citations
        )
        prompt = f"""请只修复下面答案的引用标记。

【原答案】
{answer}

【可用证据】
{evidence}

要求：
1. 保留原答案中有证据支持的事实，不添加新事实、解释、建议或版本推断。
2. 删除证据无法直接支持的事实。
3. 每个事实句末添加一个或多个真实存在的 `[Sx]`。
4. 只能使用上面列出的编号。
5. 只输出修复后的完整答案，不输出说明或 Markdown 代码块。
"""
        return self.llm.generate(
            system_prompt="你是引用标记修复器，只能依据给定证据修复引用。",
            user_prompt=prompt,
            temperature=0,
            max_tokens=1500,
            thinking=False,
            stage="citation_repair",
            timeout=self.citation_verification_timeout_seconds,
        ).strip()

    def _verify_with_optional_citation_repair(
        self,
        answer: str,
        all_citations: List[Citation],
        answer_status: str,
        trace: RAGTrace,
        request_started: float,
    ):
        """核验答案；引用编号缺失或非法时进行一次受限修复后重新核验。"""
        verification_started = time.perf_counter()
        if answer_status == "partially_answerable":
            answer = self._normalize_partial_unanswered(answer)
        citations = self._filter_used_citations(answer, all_citations)
        if not self.citation_verifier:
            self._record_span(
                trace,
                "citation_verification",
                request_started,
                verification_started,
                {"status": "disabled", "repair_attempted": False},
            )
            return answer, citations, None

        usage_before = self._usage_snapshot(self.llm)
        verification = self.citation_verifier.verify(
            answer,
            all_citations,
            answer_status,
            timeout=self.citation_verification_timeout_seconds,
            subquestions=trace.subquestions,
        )
        self._record_stage_usage(
            trace,
            "citation_verification",
            usage_before,
            self._usage_snapshot(self.llm),
        )
        initial_status = verification.status
        answer_before_binding = answer
        answer = self._apply_verified_citation_bindings(answer, verification)
        binding_repaired = answer != answer_before_binding
        citations = self._filter_used_citations(answer, all_citations)
        if binding_repaired:
            now = time.perf_counter()
            self._record_span(
                trace,
                "citation_binding_repair",
                request_started,
                now,
                {
                    "mode": "deterministic",
                    "repaired_claims": sum(
                        item.verdict == "supported" and bool(item.citation_ids)
                        for item in verification.items
                    ),
                },
                duration_ms=0,
            )
        repair_attempted = bool(
            self.citation_verifier.strict
            and self._needs_citation_repair(verification)
        )
        repair_succeeded = False

        if repair_attempted and all_citations:
            repair_started = time.perf_counter()
            repair_usage_before = self._usage_snapshot(self.llm)
            repair_error = None
            repair_usage_recorded = False
            try:
                repaired_answer = self._repair_citation_markup(
                    answer, all_citations
                )
                self._record_stage_usage(
                    trace,
                    "citation_repair",
                    repair_usage_before,
                    self._usage_snapshot(self.llm),
                )
                repair_usage_recorded = True
                if repaired_answer:
                    answer = repaired_answer
                    citations = self._filter_used_citations(
                        answer, all_citations
                    )
                    retry_usage_before = self._usage_snapshot(self.llm)
                    verification = self.citation_verifier.verify(
                        answer,
                        citations,
                        answer_status,
                        timeout=self.citation_verification_timeout_seconds,
                        subquestions=trace.subquestions,
                    )
                    self._record_stage_usage(
                        trace,
                        "citation_verification",
                        retry_usage_before,
                        self._usage_snapshot(self.llm),
                        accumulate=True,
                    )
                    repair_succeeded = verification.status == "verified"
            except Exception as exc:
                repair_error = f"{type(exc).__name__}: {exc}"
            if not repair_usage_recorded:
                self._record_stage_usage(
                    trace,
                    "citation_repair",
                    repair_usage_before,
                    self._usage_snapshot(self.llm),
                )
            self._record_span(
                trace,
                "citation_repair",
                request_started,
                repair_started,
                {
                    "attempted": True,
                    "succeeded": repair_succeeded,
                    "error": repair_error,
                    **self._latest_component_event(
                        self.llm, "citation_repair"
                    ),
                },
            )
        else:
            trace.stage_token_usage.setdefault("citation_repair", TokenUsage())

        self._record_span(
            trace,
            "citation_verification",
            request_started,
            verification_started,
            {
                "status": verification.status,
                "initial_status": initial_status,
                "repair_attempted": repair_attempted,
                "repair_succeeded": repair_succeeded,
                "binding_repaired": binding_repaired,
                **self._latest_component_event(
                    self.llm, "citation_verification"
                ),
            },
        )
        final_answer = self.citation_verifier.apply_policy(
            answer, verification, answer_status
        )
        return final_answer, citations, verification

    def _build_prompt(
        self,
        query: str,
        context: str,
        chunks: List[RetrievedChunk],
        files_in_context: Set[str] = None,
        subquestions: Optional[List[SubquestionTrace]] = None,
    ) -> str:
        """
        构建发送给 LLM 的用户提示词

        引用格式要求：只能使用参考资料前的 `[S1]` 等编号。
        """
        # 检测是否多文件来源（基于实际进入 context 的文件，而非全部 ranked chunks）
        unique_files = files_in_context if files_in_context is not None else set()

        analysis_instruction = ""
        if len(unique_files) >= 2:
            analysis_instruction = """
【多文件引用要求】
- 只选择与用户问题直接相关的条款，不要因为检索到多个文件就强行建立因果或冲突关系。
- 需要列举多项要求时，每个要点只写一个事实，并在句末紧邻其证据编号。
- 某条引用只能支持其中一部分时，拆成多条分别引用，不要合并成更宽泛的结论。
- 等级、范围、金额、期限和例外条件直接沿用原文措辞，不做同义改写。
"""

        subquestion_instruction = ""
        if subquestions:
            rows = "\n".join(
                f"- {item.subquestion_id} [{item.status}]：{item.query}"
                for item in subquestions
            )
            subquestion_instruction = f"""
【子问题覆盖状态】
{rows}

- 每条事实必须对应一个子问题；只回答有证据覆盖的子问题。
- not_found 子问题只能说明知识库暂无法确认，不能推断答案。
"""

        prompt = f"""用户问题：{query}

=== 参考资料 ===
{context}
{analysis_instruction}
{subquestion_instruction}

请根据以上参考资料回答用户问题。每个事实结论后使用对应编号引用，如 `[S1]`；只能使用上面真实存在的编号，不要自行生成文件名或页码。如果资料中没有相关信息，请明确说明无法找到答案。
"""
        return prompt


class ContextCompressor:
    """
    上下文压缩器（进阶功能）

    当检索结果总长度远超 LLM 上下文窗口时，使用 Map-Reduce 策略：
    1. Map：让每个 chunk 独立生成一个"要点摘要"
    2. Reduce：把所有要点汇总，作为最终上下文

    适用场景：用户问题需要浏览大量文档（如"总结这份报告的所有风险点"）
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def compress(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """
        Map-Reduce 压缩
        """
        # Map 阶段：每个 chunk 生成一个相关要点
        map_prompt = """给定用户问题和一段参考资料，提取与问题相关的关键要点。
如果参考资料与问题无关，回答"无关"。

用户问题：{query}

参考资料：
{content}

相关要点（最多3条）："""

        all_points = []
        for chunk in chunks:
            prompt = map_prompt.format(query=query, content=chunk.content)
            response = self.llm.generate(
                system_prompt="你是一个信息提取助手。",
                user_prompt=prompt,
                temperature=0.1
            )
            if "无关" not in response:
                all_points.append(response.strip())

        # Reduce 阶段：汇总要点
        if not all_points:
            return "无相关资料"

        combined = "\n".join([f"- {p}" for p in all_points])
        reduce_prompt = f"""将以下要点整理成连贯的上下文摘要：

{combined}

整理后的摘要："""

        summary = self.llm.generate(
            system_prompt="你是一个文本摘要助手。",
            user_prompt=reduce_prompt,
            temperature=0.2
        )

        return summary
