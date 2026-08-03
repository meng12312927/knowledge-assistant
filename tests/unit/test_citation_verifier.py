import time

from models.document import Citation, RAGTrace, SubquestionTrace
from rag.chains.rag_chain import RAGChain
from rag.post_processors.citation_verifier import CitationVerifier


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def generate(self, **kwargs):
        return self.response


def citation(citation_id="S1"):
    return Citation(
        citation_id=citation_id,
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_file="制度.txt",
        page_number=1,
        content="超过5000元且不超过20000元，需要部门负责人和财务负责人审批。",
        score=0.03,
    )


def test_verifier_accepts_fully_supported_claim():
    llm = FakeLLM(
        '{"claims":[{"claim":"15000元需要部门负责人和财务负责人审批",'
        '"citation_ids":["S1"],"verdict":"supported","reason":"金额区间和审批人一致"}],'
        '"uncited_claims":[]}'
    )
    verifier = CitationVerifier(llm, strict=True)
    result = verifier.verify("15000元需要部门负责人和财务负责人审批。[S1]", [citation()])
    assert result.status == "verified"
    assert verifier.apply_policy("原答案", result) == "原答案"


def test_verifier_fails_closed_for_unsupported_claim():
    llm = FakeLLM(
        '{"claims":[{"claim":"15000元只需部门负责人审批",'
        '"citation_ids":["S1"],"verdict":"unsupported","reason":"遗漏财务负责人"}],'
        '"uncited_claims":[]}'
    )
    verifier = CitationVerifier(llm, strict=True)
    result = verifier.verify("15000元只需部门负责人审批。[S1]", [citation()])
    assert result.status == "failed"
    assert verifier.apply_policy("错误答案", result) == CitationVerifier.SAFE_FAILURE_ANSWER


def test_claim_level_policy_salvages_supported_part_of_regular_answer():
    llm = FakeLLM(
        '{"claims":['
        '{"claim_id":"C1","citation_ids":["S1"],'
        '"verdict":"supported","reason":"原文支持"},'
        '{"claim_id":"C2","citation_ids":[],'
        '"verdict":"unsupported","reason":"原文未说明"}]}'
    )
    verifier = CitationVerifier(llm, strict=True)
    answer = "- 15000元需要两级审批。[S1]\n- 远程状态不影响审批。"

    result = verifier.verify(answer, [citation()], answer_status="answerable")
    sanitized = verifier.apply_policy(answer, result, answer_status="answerable")

    assert result.status == "partially_verified"
    assert "15000元需要两级审批。[S1]" in sanitized
    assert "远程状态不影响审批" not in sanitized


def test_verifier_rejects_unknown_citation_without_llm_call():
    verifier = CitationVerifier(FakeLLM("should not be used"), strict=True)
    result = verifier.verify("结论。[S2]", [citation("S1")])
    assert result.status == "failed"
    assert result.invalid_citation_ids == ["S2"]


class SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_missing_citation_is_repaired_once_then_verified():
    llm = SequenceLLM([
        '{"claims":[{"claim_id":"C1",'
        '"citation_ids":["S1"],"verdict":"supported","reason":"原文支持"}]}'
    ])
    chain = RAGChain.__new__(RAGChain)
    chain.llm = llm
    chain.citation_verifier = CitationVerifier(llm, strict=True)
    chain.citation_verification_timeout_seconds = 5.0
    trace = RAGTrace()

    answer, citations, verification = chain._verify_with_optional_citation_repair(
        "15000元需要部门负责人和财务负责人审批。",
        [citation()],
        "answerable",
        trace,
        time.perf_counter(),
    )

    assert answer.endswith("[S1]")
    assert [item.citation_id for item in citations] == ["S1"]
    assert verification.status == "verified"
    repair_span = next(
        span for span in trace.spans if span.name == "citation_binding_repair"
    )
    assert repair_span.attributes["mode"] == "deterministic"
    assert [call["stage"] for call in llm.calls] == ["citation_verification"]


def test_semantically_unsupported_claim_is_not_repaired():
    llm = SequenceLLM([
        '{"claims":[{"claim":"15000元只需部门负责人审批",'
        '"citation_ids":["S1"],"verdict":"unsupported","reason":"遗漏审批人"}],'
        '"uncited_claims":[]}',
    ])
    chain = RAGChain.__new__(RAGChain)
    chain.llm = llm
    chain.citation_verifier = CitationVerifier(llm, strict=True)
    chain.citation_verification_timeout_seconds = 5.0
    trace = RAGTrace()

    answer, _, verification = chain._verify_with_optional_citation_repair(
        "15000元只需部门负责人审批。[S1]",
        [citation()],
        "answerable",
        trace,
        time.perf_counter(),
    )

    assert answer == CitationVerifier.SAFE_FAILURE_ANSWER
    assert verification.status == "failed"
    assert not any(span.name == "citation_repair" for span in trace.spans)


def test_partial_answer_keeps_supported_claim_and_removes_rejected_claim():
    llm = FakeLLM(
        '{"claims":['
        '{"claim":"远程办公必须使用公司设备","subquestion_id":"SQ1",'
        '"citation_ids":["S1"],"verdict":"supported","reason":"原文支持"},'
        '{"claim":"停车位每月收费100元","subquestion_id":"SQ2",'
        '"citation_ids":["S2"],"verdict":"supported","reason":"看似支持"}'
        '],"uncited_claims":[]}'
    )
    verifier = CitationVerifier(llm, strict=True)
    subquestions = [
        SubquestionTrace(
            subquestion_id="SQ1",
            query="远程办公要求",
            status="answerable",
            covered=True,
        ),
        SubquestionTrace(
            subquestion_id="SQ2",
            query="停车位规定",
            status="not_found",
            covered=False,
        ),
    ]
    answer = (
        "- 远程办公必须使用公司设备。[S1]\n"
        "- 停车位每月收费100元。[S2]\n"
        "- 根据现有知识库无法确认停车位规定。"
    )
    result = verifier.verify(
        answer,
        [citation("S1"), citation("S2")],
        answer_status="partially_answerable",
        subquestions=subquestions,
    )
    sanitized = verifier.apply_policy(
        answer, result, answer_status="partially_answerable"
    )

    assert result.status == "partially_verified"
    assert result.items[0].subquestion_id == "SQ1"
    assert "远程办公必须使用公司设备。[S1]" in sanitized
    assert "停车位每月收费100元" not in sanitized
    assert "无法确认停车位规定" in sanitized


def test_partial_unanswered_section_drops_citations_and_negative_explanation():
    answer = (
        "可以确认：\n- 每半年评估一次。[S1]\n\n"
        "暂无法确认：\n- 评分标准根据现有知识库无法确认。"
        "资料中没有任何评分维度。[S1]"
    )
    normalized = RAGChain._normalize_partial_unanswered(answer)

    assert "每半年评估一次。[S1]" in normalized
    assert "- 评分标准根据现有知识库无法确认。" in normalized
    assert "没有任何评分维度" not in normalized
    assert normalized.count("[S1]") == 1


def test_claim_level_verifier_detects_uncited_sentence_in_cited_paragraph():
    llm = FakeLLM(
        '{"claims":[{"claim_id":"C1","citation_ids":["S1"],'
        '"verdict":"supported","reason":"原文支持"}]}'
    )
    verifier = CitationVerifier(llm, strict=True)

    result = verifier.verify(
        "年假为五天。[S1] 未休年假可以无限累计。",
        [citation("S1")],
    )

    assert result.status == "partially_verified"
    assert result.total_claims == 2
    assert result.supported_claims == 1
    assert result.claim_coverage_rate == 0.5
    assert result.items[1].verdict == "unsupported"


def test_claim_level_verifier_rejects_changed_citation_binding():
    llm = FakeLLM(
        '{"claims":[{"claim_id":"C1","citation_ids":["S2"],'
        '"verdict":"supported","reason":"原文支持"}]}'
    )
    verifier = CitationVerifier(llm, strict=True)

    result = verifier.verify(
        "年假为五天。[S1]", [citation("S1"), citation("S2")]
    )

    assert result.status == "failed"
    assert result.items[0].verdict == "unsupported"
    assert "绑定不一致" in result.items[0].reason


def test_claim_extractor_ignores_structural_headings_and_epistemic_notes():
    claims = CitationVerifier._extract_claims(
        "# 回答\n一、禁止输入的数据类型\n远程办公安全要求\n"
        "- 不得输入客户密码。[S1]\n"
        "- 现有知识库未提供其他类型。\n"
        "- 建议咨询管理员确认适用版本。",
        {"S1": citation("S1")},
    )

    assert [claim.text for claim in claims] == ["不得输入客户密码"]
