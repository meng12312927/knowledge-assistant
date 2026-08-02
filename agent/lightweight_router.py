"""零 LLM 成本的自动 Agent 路由。

普通制度问答保留在可流式的 RAG 链路；只有明确工具、比较、顺序型
多步骤意图或可恢复低召回才升级到 Agent。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.router import Intent, TaskPlan


@dataclass(frozen=True)
class AgentRouteDecision:
    use_agent: bool
    intent: Intent
    reason: str
    plan: TaskPlan | None = None


_DATABASE_PATTERNS = (
    re.compile(r"\bselect\b.+\bfrom\b", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"\b(insert|update|delete)\b.+\b(table|set|from)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"(查询|查一下|访问|连接|执行).{0,8}(数据库|SQL)", re.IGNORECASE),
)
_CALCULATOR_PATTERNS = (
    re.compile(r"(用计算器|调用计算器|帮我计算|计算一下|算一下)", re.IGNORECASE),
    re.compile(r"\d+(?:\.\d+)?\s*[\+\-\*/×÷]\s*\d+(?:\.\d+)?"),
)
_COMPARISON_PATTERNS = (
    re.compile(
        r"(对比|比较|区别|差异|异同|优缺点).{0,30}(和|与|及|两者|二者)",
        re.IGNORECASE,
    ),
    re.compile(r"(和|与).{0,30}(相比|哪个好|有何不同)", re.IGNORECASE),
    re.compile(r"(和|与).{0,30}(区别|差异|异同)", re.IGNORECASE),
)
_SEQUENCE_PATTERNS = (
    re.compile(r"先.+(?:再|然后|接着|最后).+", re.DOTALL),
    re.compile(
        r"(分步骤|逐步|依次|分别).{0,20}(分析|查询|检索|计算|处理|回答)",
        re.IGNORECASE,
    ),
    re.compile(r"(制定|生成).{0,20}(计划|方案).{0,20}(并|然后|再).+", re.DOTALL),
)


def _matches_any(query: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(query) for pattern in patterns)


def decide_agent_route(
    query: str,
    *,
    retrieval_quality: str = "unknown",
    user_requested_agent: bool = False,
) -> AgentRouteDecision:
    """Combine retrieval quality, intent rules, tools and explicit API requests."""
    normalized = " ".join((query or "").split())

    if _matches_any(normalized, _DATABASE_PATTERNS + _CALCULATOR_PATTERNS):
        plan = TaskPlan(
            intent=Intent.TOOL_CALL,
            steps=["识别并调用匹配工具", "校验工具结果", "组织最终回答"],
            requires_tools=["dynamic_tool"],
            estimated_complexity=3,
        )
        return AgentRouteDecision(
            True, Intent.TOOL_CALL, "explicit_tool_request", plan
        )

    if _matches_any(normalized, _COMPARISON_PATTERNS):
        plan = TaskPlan(
            intent=Intent.COMPARISON,
            steps=["识别比较对象", "分别检索证据", "按共同维度对比"],
            requires_tools=["rag_retrieval"],
            estimated_complexity=4,
        )
        return AgentRouteDecision(
            True, Intent.COMPARISON, "explicit_comparison", plan
        )

    if _matches_any(normalized, _SEQUENCE_PATTERNS):
        plan = TaskPlan(
            intent=Intent.MULTI_STEP,
            steps=["拆分子问题", "分别检索或处理", "汇总并核对最终结论"],
            requires_tools=["rag_retrieval"],
            estimated_complexity=3,
        )
        return AgentRouteDecision(
            True, Intent.MULTI_STEP, "explicit_multi_step", plan
        )

    if user_requested_agent:
        plan = TaskPlan(
            intent=Intent.FACTUAL_QA,
            steps=["使用 Agent 执行用户显式请求"],
            requires_tools=["rag_retrieval"],
            estimated_complexity=1,
        )
        return AgentRouteDecision(
            True, Intent.FACTUAL_QA, "user_requested_agent", plan
        )

    if retrieval_quality in {"recoverable_low", "low_confidence"}:
        plan = TaskPlan(
            intent=Intent.FACTUAL_QA,
            steps=["复用升级检索结果生成可验证回答"],
            requires_tools=["rag_retrieval"],
            estimated_complexity=2,
        )
        return AgentRouteDecision(
            True, Intent.FACTUAL_QA, "recoverable_low_retrieval", plan
        )

    if retrieval_quality == "not_found":
        return AgentRouteDecision(
            False, Intent.FACTUAL_QA, "terminal_not_found"
        )

    return AgentRouteDecision(
        False, Intent.FACTUAL_QA, "ordinary_rag_query"
    )
