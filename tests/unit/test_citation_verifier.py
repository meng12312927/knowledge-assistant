from models.document import Citation
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


def test_verifier_rejects_unknown_citation_without_llm_call():
    verifier = CitationVerifier(FakeLLM("should not be used"), strict=True)
    result = verifier.verify("结论。[S2]", [citation("S1")])
    assert result.status == "failed"
    assert result.invalid_citation_ids == ["S2"]
