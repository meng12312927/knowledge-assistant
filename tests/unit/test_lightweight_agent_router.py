from types import SimpleNamespace

from agent.lightweight_router import decide_agent_route
from agent.router import Intent
from app.api.main import _resolve_agent_route
from models.document import QueryRequest


def test_ordinary_policy_qa_stays_on_rag():
    for query in [
        "年假最多可以累计多少天？",
        "查询年假制度的申请条件",
    ]:
        decision = decide_agent_route(query)
        assert decision.use_agent is False
        assert decision.intent == Intent.FACTUAL_QA
        assert decision.plan is None


def test_explicit_calculator_and_database_requests_use_tools():
    for query in [
        "用计算器算一下 1250*0.8",
        "执行 SQL：SELECT name FROM employees",
        "帮我查询数据库中的报销记录",
    ]:
        decision = decide_agent_route(query)
        assert decision.use_agent is True
        assert decision.intent == Intent.TOOL_CALL
        assert decision.plan.requires_tools == ["dynamic_tool"]


def test_comparison_and_explicit_sequence_use_agent_without_llm_classification():
    comparison = decide_agent_route("对比年假和病假的申请条件")
    sequence = decide_agent_route("先查询差旅标准，然后计算三天的补贴，最后汇总")

    assert comparison.use_agent is True
    assert comparison.intent == Intent.COMPARISON
    assert comparison.reason == "explicit_comparison"
    assert sequence.use_agent is True
    assert sequence.intent == Intent.MULTI_STEP
    assert sequence.reason == "explicit_multi_step"


def test_compound_factual_question_is_not_mistaken_for_multi_step():
    decision = decide_agent_route(
        "临时借用设备的最长期限是多少天？到期前应如何处理？"
    )

    assert decision.use_agent is False
    assert decision.intent == Intent.FACTUAL_QA
    assert decision.reason == "ordinary_rag_query"


def test_tools_are_automatic_and_api_flag_forces_agent():
    settings = SimpleNamespace(enable_agent=True)

    disabled = _resolve_agent_route(
        settings,
        QueryRequest(query="计算一下 2+2", enable_agent=False),
    )
    ordinary = _resolve_agent_route(
        settings,
        QueryRequest(query="试用期多久？", enable_agent=True),
    )
    tool = _resolve_agent_route(
        settings,
        QueryRequest(query="计算一下 2+2", enable_agent=True),
    )
    automatic_rag = _resolve_agent_route(
        settings,
        QueryRequest(query="试用期多久？", enable_agent=False),
        retrieval_quality="sufficient",
    )

    assert disabled.use_agent is True
    assert disabled.reason == "explicit_tool_request"
    assert ordinary.use_agent is True
    assert ordinary.reason == "user_requested_agent"
    assert tool.use_agent is True
    assert automatic_rag.use_agent is False
    assert automatic_rag.reason == "ordinary_rag_query"


def test_recoverable_low_retrieval_escalates_but_terminal_ood_does_not():
    recoverable = decide_agent_route(
        "公司的相关规定是什么？",
        retrieval_quality="recoverable_low",
    )
    terminal = decide_agent_route(
        "公司宿舍允许养宠物吗？",
        retrieval_quality="not_found",
    )

    assert recoverable.use_agent is True
    assert recoverable.reason == "recoverable_low_retrieval"
    assert recoverable.intent == Intent.FACTUAL_QA
    assert terminal.use_agent is False
    assert terminal.reason == "terminal_not_found"
