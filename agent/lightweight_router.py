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


# ═══════════════════════════════════════════════════════════
# 正则模式定义
# ═══════════════════════════════════════════════════════════

_CHITCHAT_PATTERNS = (
    re.compile(r"^(你好|您好|hi|hello|嗨|嘿)[\s!！。.]*$", re.IGNORECASE),
    re.compile(r"^(谢谢|多谢|感谢|thank|3q)", re.IGNORECASE),
    re.compile(r"^(再见|拜拜|bye|see you)[\s!！。.]*$", re.IGNORECASE),
    re.compile(
        r"^(你是谁|你能做什么|你会什么|你的.*(?:功能|能力|作用)|介绍一下自己)",
        re.IGNORECASE,
    ),
    re.compile(r"^(帮助|help|怎么用|使用说明|menu|菜单)[\s!！。.]*$", re.IGNORECASE),
)

_DATABASE_PATTERNS = (
    re.compile(r"\bselect\b.+\bfrom\b", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"\b(insert|update|delete)\b.+\b(table|set|from)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(查|查询|查一下|访问|连接|执行|查查).{0,12}(数据库|SQL|数据|销售|用户|订单)",
        re.IGNORECASE,
    ),
)

_CALCULATOR_PATTERNS = (
    re.compile(
        r"(用计算器|调用计算器|帮我计算|计算一下|算一下|帮我算)",
        re.IGNORECASE,
    ),
    re.compile(r"\d+(?:\.\d+)?\s*[\+\-\*/×÷]\s*\d+(?:\.\d+)?"),
)

_COMPARISON_PATTERNS = (
    re.compile(
        r"(对比|比较|区别|差异|异同|优缺点).{0,30}(和|与|及|两者|二者)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(和|与).{0,30}(有什么不同|有何不同|哪个好|哪个更|哪一个更)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(和|与).{0,30}(区别|差异|异同|的不同)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(两者|两个|二者).{0,20}(区别|差异|不同|有什么|哪个)",
        re.IGNORECASE,
    ),
    re.compile(r"分别有多少", re.IGNORECASE),
)

_SEQUENCE_PATTERNS = (
    re.compile(r"先.+(?:再|然后|接着|最后).+", re.DOTALL),
    re.compile(
        r"(分步骤|逐步|依次|分别).{0,20}(分析|查询|检索|计算|处理|回答)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(制定|生成).{0,20}(计划|方案).{0,20}(并|然后|再).+", re.DOTALL,
    ),
)

_SUMMARIZATION_PATTERNS = (
    re.compile(
        r"(总结|概括|归纳|摘要|提炼|概述|汇总|整理).{0,10}"
        r"(一下|全文|内容|要点|中心|这篇|这个|.*规定|.*制度)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(讲了什么|主要内容|核心要点|有哪些.*(?:规定|制度|政策|条款))",
        re.IGNORECASE,
    ),
)


def _matches_any(query: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(query) for pattern in patterns)


# ═══════════════════════════════════════════════════════════
# 路由决策
# ═══════════════════════════════════════════════════════════

def decide_agent_route(
    query: str,
    *,
    retrieval_quality: str = "unknown",
    user_requested_agent: bool = False,
) -> AgentRouteDecision:
    """Combine retrieval quality, intent rules, tools and explicit API requests."""
    normalized = " ".join((query or "").split())

    # ① 闲聊 — 零检索，直接 LLM 友好回复
    if _matches_any(normalized, _CHITCHAT_PATTERNS):
        return AgentRouteDecision(
            True,
            Intent.CHITCHAT,
            "explicit_chitchat",
            TaskPlan(
                intent=Intent.CHITCHAT,
                steps=["直接友好回复"],
                requires_tools=[],
                estimated_complexity=1,
            ),
        )

    # ② 工具调用 — 计算器 / 数据库查询
    if _matches_any(normalized, _DATABASE_PATTERNS + _CALCULATOR_PATTERNS):
        return AgentRouteDecision(
            True,
            Intent.TOOL_CALL,
            "explicit_tool_request",
            TaskPlan(
                intent=Intent.TOOL_CALL,
                steps=["识别并调用匹配工具", "校验工具结果", "组织最终回答"],
                requires_tools=["dynamic_tool"],
                estimated_complexity=3,
            ),
        )

    # ③ 多步推理 — 先...再...然后...（优先级高于对比，避免被关键词抢走）
    if _matches_any(normalized, _SEQUENCE_PATTERNS):
        return AgentRouteDecision(
            True,
            Intent.MULTI_STEP,
            "explicit_multi_step",
            TaskPlan(
                intent=Intent.MULTI_STEP,
                steps=["拆分子问题", "分别检索或处理", "汇总并核对最终结论"],
                requires_tools=["rag_retrieval"],
                estimated_complexity=3,
            ),
        )

    # ④ 对比分析 — A 和 B 的区别
    if _matches_any(normalized, _COMPARISON_PATTERNS):
        return AgentRouteDecision(
            True,
            Intent.COMPARISON,
            "explicit_comparison",
            TaskPlan(
                intent=Intent.COMPARISON,
                steps=["识别比较对象", "分别检索证据", "按共同维度对比"],
                requires_tools=["rag_retrieval"],
                estimated_complexity=4,
            ),
        )

    # ⑤ 摘要总结 — 总结/概括/整理
    if _matches_any(normalized, _SUMMARIZATION_PATTERNS):
        return AgentRouteDecision(
            True,
            Intent.SUMMARIZATION,
            "explicit_summarization",
            TaskPlan(
                intent=Intent.SUMMARIZATION,
                steps=["广泛检索相关内容", "提取关键要点", "组织成连贯摘要"],
                requires_tools=["rag_retrieval", "context_compressor"],
                estimated_complexity=3,
            ),
        )

    # ⑥ 用户显式开 Agent
    if user_requested_agent:
        return AgentRouteDecision(
            True,
            Intent.FACTUAL_QA,
            "user_requested_agent",
            TaskPlan(
                intent=Intent.FACTUAL_QA,
                steps=["使用 Agent 执行用户显式请求"],
                requires_tools=["rag_retrieval"],
                estimated_complexity=1,
            ),
        )

    # ⑦ 检索不够好 → 升级到 Agent 双路交叉验证
    if retrieval_quality in {"recoverable_low", "low_confidence"}:
        return AgentRouteDecision(
            True,
            Intent.FACTUAL_QA,
            "recoverable_low_retrieval",
            TaskPlan(
                intent=Intent.FACTUAL_QA,
                steps=["复用升级检索结果生成可验证回答"],
                requires_tools=["rag_retrieval"],
                estimated_complexity=2,
            ),
        )

    # ⑧ 检索完全不行 → 拒答
    if retrieval_quality == "not_found":
        return AgentRouteDecision(
            False, Intent.FACTUAL_QA, "terminal_not_found"
        )

    # ⑨ 兜底 — 普通 RAG
    return AgentRouteDecision(
        False, Intent.FACTUAL_QA, "ordinary_rag_query"
    )
