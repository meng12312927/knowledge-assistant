"""
LangGraph Agent Router

基于 LangGraph StateGraph 重构的 Agent 路由器。
保留原有意图识别和规划能力，新增：
- ReAct 工具调用循环
- 状态机驱动的执行流程
- 对话记忆注入
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import asdict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool

from models.document import Citation, CitationVerification, QueryRequest, ChatResponse, RAGTrace
from rag.chains.rag_chain import RAGChain
from agent.router import AgentRouter, Intent, TaskPlan, ToolResult
from agent.tools.base import BaseTool, ToolCallGuard, ToolRegistry, CalculatorTool, DatabaseQueryTool


# ═══════════════════════════════════════════════════════════
# 1. 状态定义
# ═══════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """LangGraph Agent 状态"""
    messages: Annotated[list, add_messages]  # 对话历史（LangChain message 格式）
    query: str                               # 当前用户查询
    intent: Optional[str]                    # 识别出的意图
    plan: Optional[Dict]                     # 任务计划（序列化为 dict）
    rag_result: Optional[Dict]               # RAG 结果（序列化为 dict）
    tool_results: List[Dict]                 # 工具执行结果
    final_answer: str                        # 最终答案
    sources: List[Dict]                      # 引用来源
    citations: List[Dict]                    # [S1] 到 chunk_id/原文的映射
    citation_verification: Optional[Dict]    # 引用一致性核验
    trace: Optional[Dict]                    # 六阶段 RAG 追踪
    answer_status: str                       # RAG 回答置信度状态
    prepared_rag: Any                        # API retrieval-first 阶段的可复用结果


# ═══════════════════════════════════════════════════════════
# 2. 工具包装（将现有 BaseTool 包装为 LangChain 工具）
# ═══════════════════════════════════════════════════════════

class LangChainToolWrapper:
    """将自定义 BaseTool 包装为 LangChain @tool 函数"""

    def __init__(self, base_tool: BaseTool):
        self.base_tool = base_tool
        self._lc_tool = None

    @property
    def lc_tool(self):
        if self._lc_tool is None:
            self._lc_tool = self._build_tool()
        return self._lc_tool

    def _build_tool(self):
        bt = self.base_tool
        # 使用闭包捕获 base_tool 实例
        def _run(expression: str = "", sql: str = "") -> str:
            # 根据工具类型选择参数
            if bt.name == "calculator":
                result = bt.execute(expression=expression)
            elif bt.name == "database_query":
                result = bt.execute(sql=sql)
            else:
                # 通用：尝试传入所有参数
                kwargs = {k: v for k, v in {"expression": expression, "sql": sql}.items() if v}
                result = bt.execute(**kwargs)

            if result.success:
                return str(result.output)
            return f"Tool error: {result.error_message}"

        # 动态生成 docstring
        _run.__doc__ = bt.description
        _run.__name__ = bt.name

        return tool(_run)


def build_langchain_tools(tool_registry: ToolRegistry) -> List[Any]:
    """将 ToolRegistry 中的所有工具转换为 LangChain 工具列表"""
    tools = []
    for info in tool_registry.list_tools():
        name = info["name"]
        try:
            base_tool = tool_registry.get(name)
            wrapper = LangChainToolWrapper(base_tool)
            tools.append(wrapper.lc_tool)
        except Exception:
            pass
    return tools


# ═══════════════════════════════════════════════════════════
# 3. LangGraph Agent Router
# ═══════════════════════════════════════════════════════════

class LangGraphAgentRouter:
    """
    LangGraph 版 Agent 路由器

    使用 StateGraph 管理执行流程，支持：
    - 意图识别 → 规划 → 路由 → 执行 → 汇总
    - ReAct 工具调用循环
    - 对话记忆自动注入
    """

    def __init__(
        self,
        llm: Any,
        rag_chain: RAGChain,
        tools: Optional[ToolRegistry] = None
    ):
        self.llm = llm
        self.rag_chain = rag_chain
        self.tools = tools or ToolRegistry()

        # 构建 LangChain 工具列表
        self.lc_tools = build_langchain_tools(self.tools)
        self.tool_node = ToolNode(self.lc_tools) if self.lc_tools else None

        # 缓存 legacy AgentRouter 实例，避免节点中重复创建
        self._legacy_router = AgentRouter(llm=self.llm, rag_chain=self.rag_chain)

        # 构建图
        self.graph = self._build_graph()

    # ── 节点方法 ──

    def _analyze_intent_node(self, state: AgentState) -> AgentState:
        """意图识别节点（若 state 中已提供 intent 则保留）"""
        if state.get("intent"):
            return state
        query = state["query"]
        intent = self._analyze_intent(query)
        return {**state, "intent": intent}

    def _plan_node(self, state: AgentState) -> AgentState:
        """任务规划节点（若 state 中已提供 plan 则保留）"""
        if state.get("plan"):
            return state
        query = state["query"]
        intent_str = state.get("intent") or "factual_qa"
        intent = Intent(intent_str)
        plan = self._plan(intent, query)
        return {**state, "plan": asdict(plan)}

    def _route_node(self, state: AgentState) -> str:
        """路由条件：根据意图决定下一个节点"""
        intent = state.get("intent")
        plan = state.get("plan", {})
        requires_tools = plan.get("requires_tools", [])
        query = state.get("query", "")

        # 关键词触发：包含工具相关关键词 → 优先走工具
        tool_keywords = ["计算", "调用工具", "计算器", "查询数据库", "sql",
                         "calculate", "calculator", "compute", "query", "database", "look up"]
        if any(kw in query.lower() for kw in tool_keywords):
            if self.tool_node:
                return "tool_call"

        # TOOL_CALL 意图或计划需要工具 → 走工具分支
        if intent == "tool_call" or "dynamic_tool" in requires_tools:
            if self.tool_node:
                return "tool_call"
            # 没有可用工具时 fallback 到 RAG
            return "factual_qa"

        route_map = {
            "factual_qa": "factual_qa",
            "comparison": "comparison",
            "summarization": "summarization",
            "multi_step": "multi_step",
            "chitchat": "chitchat",
        }
        return route_map.get(intent, "factual_qa")

    def _factual_qa_node(self, state: AgentState) -> AgentState:
        """事实问答节点：直接走 RAG"""
        query = state["query"]
        result = self.rag_chain.invoke(
            QueryRequest(query=query),
            prepared=state.get("prepared_rag"),
        )
        return {
            **state,
            "final_answer": result.answer,
            "sources": [self._chunk_to_dict(s) for s in result.sources],
            "citations": [c.model_dump() for c in result.citations],
            "citation_verification": result.citation_verification.model_dump() if result.citation_verification else None,
            "trace": result.trace.model_dump() if result.trace else None,
            "answer_status": result.answer_status,
        }

    def _comparison_node(self, state: AgentState) -> AgentState:
        """对比分析节点：复用子问题覆盖检索和逐结论引用核验。"""
        query = state["query"]
        result = self.rag_chain.invoke(
            QueryRequest(query=query),
            prepared=state.get("prepared_rag"),
        )
        return {
            **state,
            "final_answer": result.answer,
            "sources": [self._chunk_to_dict(s) for s in result.sources],
            "citations": [c.model_dump() for c in result.citations],
            "citation_verification": result.citation_verification.model_dump() if result.citation_verification else None,
            "trace": result.trace.model_dump() if result.trace else None,
            "answer_status": result.answer_status,
        }

    def _summarization_node(self, state: AgentState) -> AgentState:
        """摘要节点"""
        query = state["query"]
        result = self._legacy_router._execute_summarization(query)
        return {
            **state,
            "final_answer": result.answer,
            "sources": [self._chunk_to_dict(s) for s in result.sources],
            "citations": [c.model_dump() for c in result.citations],
            "citation_verification": result.citation_verification.model_dump() if result.citation_verification else None,
            "trace": result.trace.model_dump() if result.trace else None,
        }

    def _multi_step_node(self, state: AgentState) -> AgentState:
        """多步推理节点"""
        query = state["query"]
        plan_dict = state.get("plan", {})
        plan = TaskPlan(
            intent=Intent(plan_dict.get("intent", "multi_step")),
            steps=plan_dict.get("steps", []),
            requires_tools=plan_dict.get("requires_tools", []),
            estimated_complexity=plan_dict.get("estimated_complexity", 1),
        )
        prepared = state.get("prepared_rag")
        if prepared and prepared.trace.subquestion_planning_triggered:
            result = self.rag_chain.invoke(
                QueryRequest(query=query), prepared=prepared
            )
        else:
            result = self._legacy_router._execute_multi_step(plan, query)
        return {
            **state,
            "final_answer": result.answer,
            "sources": [self._chunk_to_dict(s) for s in result.sources],
            "citations": [c.model_dump() for c in result.citations],
            "citation_verification": result.citation_verification.model_dump() if result.citation_verification else None,
            "trace": result.trace.model_dump() if result.trace else None,
            "answer_status": result.answer_status,
        }

    def _chitchat_node(self, state: AgentState) -> AgentState:
        """闲聊节点"""
        query = state["query"]
        result = self._legacy_router._execute_chitchat(query)
        return {
            **state,
            "final_answer": result.answer,
            "sources": [],
            "citations": [],
            "citation_verification": None,
            "trace": None,
            "answer_status": "answerable",
        }

    def _tool_call_node(self, state: AgentState) -> AgentState:
        """
        工具调用节点（带断路器 + 模糊匹配 + 失败反馈重试）。

        四层防护：
        1. 断路器防止死循环 / 重复调用 / 超时
        2. 工具名模糊匹配防 LLM 拼写错误
        3. 参数 schema 校验在 execute 前拦截
        4. 失败后将错误信息反馈给 LLM，最多重试 2 次
        """
        import json, re, ast

        query = state["query"]
        guard = ToolCallGuard(max_rounds=5, max_total_seconds=10.0)

        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description}" for t in self.lc_tools
        ])

        def _build_prompt(last_error: str = None) -> str:
            error_section = ""
            if last_error:
                error_section = f"""
上一次调用失败，错误信息：{last_error}
请修正后重新调用。"""

            return f"""用户问题：{query}

可用工具：
{tool_descriptions}
{error_section}
规则：
1. 禁止直接回答。
2. 禁止进行任何计算或推理。
3. 必须调用工具。
4. 只输出以下格式，其他任何内容都不允许：
TOOL:calculator
ARGS:{{\"expression\":\"123*456\"}}

或者：
TOOL:database_query
ARGS:{{\"sql\":\"SELECT * FROM sales\"}}"""

        answer = ""
        tool_results: list = []
        last_error: str | None = None
        max_retries = 2

        for retry_attempt in range(max_retries + 1):
            response = self.llm.generate(
                system_prompt="你是一个只能调用工具的助手。禁止直接计算或推理。只能输出指定格式。",
                user_prompt=_build_prompt(last_error),
                temperature=0,
            )
            answer = response

            # 解析工具调用格式
            match = re.search(
                r"TOOL:(\w+)\s*\nARGS:(\{.*?\})", response, re.DOTALL
            )
            if not match:
                match = re.search(
                    r"调用工具\s+(\w+)\s+参数[:：]\s*(\{.*?\})",
                    response, re.DOTALL,
                )

            if not match:
                # LLM 完全没输出工具格式 → 直接用原始回答兜底
                break

            raw_tool_name = match.group(1)
            raw_params = match.group(2)

            try:
                # ① 解析参数
                try:
                    params = json.loads(raw_params)
                except json.JSONDecodeError:
                    params = ast.literal_eval(raw_params)  # 兼容单引号

                # ② 工具名模糊匹配
                tool_name = self.tools.fuzzy_find(raw_tool_name)

                # ③ 断路器检查
                guard.check(tool_name, params)

                # ④ 执行工具（含参数校验）
                result = self.tools.execute(tool_name, **params)
                tool_results.append(asdict(result))

                if result.success:
                    # 成功 → 让 LLM 整合结果
                    integration_prompt = f"""用户问题：{query}

工具调用结果：
{result.output}

请基于工具结果回答用户问题。"""
                    answer = self.llm.generate(
                        system_prompt="你是一个智能助手，请基于工具结果回答用户问题。",
                        user_prompt=integration_prompt,
                        temperature=0.3,
                    )
                    break
                else:
                    # 工具执行失败（如参数非法）
                    last_error = (
                        f"工具 {tool_name} 执行失败：{result.error_message}"
                    )
                    logger.warning(
                        "tool_call_retry attempt=%s tool=%s error=%s",
                        retry_attempt + 1, tool_name, result.error_message,
                    )
                    if retry_attempt >= max_retries:
                        answer = f"工具调用失败：{result.error_message}"
                        break

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "tool_call_error attempt=%s tool=%s error=%s",
                    retry_attempt + 1, raw_tool_name, last_error,
                )
                if retry_attempt >= max_retries:
                    answer = (
                        f"工具调用失败：{last_error}。"
                        "已重试 {max_retries} 次仍无法完成，请检查问题描述。"
                    )
                    break

        return {
            **state,
            "final_answer": answer,
            "tool_results": tool_results,
            "sources": [],
            "citations": [],
            "citation_verification": None,
            "trace": None,
        }

    def _finalize_node(self, state: AgentState) -> AgentState:
        """最终汇总节点（占位，可用于后处理）"""
        return state

    # ── 图构建 ──

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态图"""
        builder = StateGraph(AgentState)

        # 注册节点
        builder.add_node("analyze_intent", self._analyze_intent_node)
        builder.add_node("plan_node", self._plan_node)
        builder.add_node("factual_qa", self._factual_qa_node)
        builder.add_node("comparison", self._comparison_node)
        builder.add_node("summarization", self._summarization_node)
        builder.add_node("multi_step", self._multi_step_node)
        builder.add_node("chitchat", self._chitchat_node)
        builder.add_node("tool_call", self._tool_call_node)
        builder.add_node("finalize", self._finalize_node)

        # 设置边
        builder.set_entry_point("analyze_intent")
        builder.add_edge("analyze_intent", "plan_node")
        builder.add_conditional_edges(
            "plan_node",
            self._route_node,
            {
                "factual_qa": "factual_qa",
                "comparison": "comparison",
                "summarization": "summarization",
                "multi_step": "multi_step",
                "chitchat": "chitchat",
                "tool_call": "tool_call",
            }
        )
        builder.add_edge("factual_qa", "finalize")
        builder.add_edge("comparison", "finalize")
        builder.add_edge("summarization", "finalize")
        builder.add_edge("multi_step", "finalize")
        builder.add_edge("chitchat", "finalize")
        builder.add_edge("tool_call", "finalize")
        builder.add_edge("finalize", END)

        return builder.compile()

    # ── 公共接口 ──

    def execute(
        self,
        plan: TaskPlan,
        query: str,
        prepared=None,
    ) -> ChatResponse:
        """
        执行入口（兼容原有 AgentRouter 接口）

        Args:
            plan: 任务计划（若提供，优先使用其意图和工具需求）
            query: 用户查询
        """
        # 标准化意图为小写字符串
        raw_intent = None
        if plan and plan.intent:
            if isinstance(plan.intent, str):
                raw_intent = plan.intent.lower()
            elif hasattr(plan.intent, 'value'):
                raw_intent = plan.intent.value
            else:
                raw_intent = str(plan.intent).lower()
        initial_state: AgentState = {
            "messages": [],
            "query": query,
            "intent": raw_intent,
            "plan": asdict(plan) if plan else None,
            "rag_result": None,
            "tool_results": [],
            "final_answer": "",
            "sources": [],
            "citations": [],
            "citation_verification": None,
            "trace": None,
            "answer_status": "answerable",
            "prepared_rag": prepared,
        }

        final_state = self.graph.invoke(initial_state)

        # 转换 sources 回 RetrievedChunk（简化：仅保留必要字段）
        from models.document import RetrievedChunk
        sources = []
        for s in final_state.get("sources", []):
            if isinstance(s, dict):
                sources.append(RetrievedChunk(
                    content=s.get("content", ""),
                    metadata=s.get("metadata", {}),
                    score=s.get("score", 0.0)
                ))
            else:
                sources.append(s)

        citations = [Citation.model_validate(item) for item in final_state.get("citations", [])]
        verification_data = final_state.get("citation_verification")
        verification = CitationVerification.model_validate(verification_data) if verification_data else None
        trace_data = final_state.get("trace")
        trace = RAGTrace.model_validate(trace_data) if trace_data else None

        return ChatResponse(
            answer=final_state.get("final_answer", ""),
            sources=sources,
            citations=citations,
            citation_verification=verification,
            trace=trace,
            query_time_ms=None,
            tool_results=final_state.get("tool_results") or None,
            answer_status=final_state.get("answer_status", "answerable"),
        )

    # ── 流式接口（实验性） ──

    def stream_execute(self, plan: TaskPlan, query: str):
        """
        流式执行入口

        Yields:
            dict: 包含 node 名称和输出的事件
        """
        initial_state: AgentState = {
            "messages": [],
            "query": query,
            "intent": None,
            "plan": None,
            "rag_result": None,
            "tool_results": [],
            "final_answer": "",
            "sources": [],
            "citations": [],
            "citation_verification": None,
            "trace": None,
            "answer_status": "answerable",
            "prepared_rag": None,
        }

        for event in self.graph.stream(initial_state):
            yield event

    # ── 内部辅助 ──

    def _analyze_intent(self, query: str) -> str:
        """复用原 AgentRouter 的意图识别逻辑"""
        legacy = AgentRouter(llm=self.llm, rag_chain=self.rag_chain)
        intent = legacy.analyze_intent(query)
        return intent.value

    def _plan(self, intent: Intent, query: str) -> TaskPlan:
        """复用原 AgentRouter 的规划逻辑"""
        legacy = AgentRouter(llm=self.llm, rag_chain=self.rag_chain)
        return legacy.plan(intent, query)

    @staticmethod
    def _chunk_to_dict(chunk: Any) -> Dict:
        """将 RetrievedChunk 序列化为 dict"""
        if hasattr(chunk, "content"):
            return {
                "content": chunk.content,
                "metadata": getattr(chunk, "metadata", {}),
                "score": getattr(chunk, "score", 0.0),
            }
        if isinstance(chunk, dict):
            return chunk
        return {"content": str(chunk), "metadata": {}, "score": 0.0}
