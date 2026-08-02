"""生成后引用核验：检查答案结论是否被引用原文真实支持。"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from models.document import (
    Citation,
    CitationVerification,
    CitationVerificationItem,
)


class CitationVerifier:
    """先做确定性编号校验，再使用 LLM 做逐结论证据蕴含判断。"""

    SAFE_FAILURE_ANSWER = "引用核验未通过，本次回答未展示。请重试或查看核验详情。"
    SAFE_UNVERIFIED_ANSWER = "引用核验服务暂时不可用，本次回答未展示。请稍后重试。"

    def __init__(self, llm, enabled: bool = True, strict: bool = True):
        self.llm = llm
        self.enabled = enabled
        self.strict = strict

    def verify(
        self,
        answer: str,
        citations: List[Citation],
        answer_status: str = "answerable",
        timeout: Optional[float] = None,
    ) -> CitationVerification:
        if not self.enabled or answer_status == "not_found":
            return CitationVerification(status="skipped", message="无需执行引用核验")

        available = {citation.citation_id: citation for citation in citations}
        referenced = set(re.findall(r"\[(S[1-9]\d*)\]", answer or ""))
        invalid = sorted(referenced - set(available))
        if invalid:
            return CitationVerification(
                status="failed",
                invalid_citation_ids=invalid,
                message="答案包含不存在的引用编号",
            )
        if not referenced:
            return CitationVerification(
                status="failed",
                message="答案没有引用任何结构化证据",
            )

        evidence = "\n\n".join(
            f"[{citation_id}] {available[citation_id].source_file}\n"
            f"{available[citation_id].content}"
            for citation_id in sorted(referenced)
        )
        prompt = f"""请核验答案中的每一条事实结论是否被对应引用原文直接支持。

【答案】
{answer}

【引用原文】
{evidence}

严格规则：
1. 将答案拆成独立事实结论；纯格式、标题和礼貌用语不算事实结论。
2. 每条事实结论必须列出答案中紧邻或明确关联的引用编号。
3. supported：原文直接支持完整结论；partial：只支持部分；unsupported：不支持或矛盾；uncited：事实结论没有引用。
4. 不得使用外部知识，不得因为结论看起来合理就判为 supported。
5. claim 保留答案中的最短完整结论，reason 不超过 20 个汉字，避免复述原文。
6. 只输出紧凑 JSON，不要输出 Markdown：
{{"claims":[{{"claim":"...","citation_ids":["S1"],"verdict":"supported","reason":"..."}}],"uncited_claims":[]}}
"""
        try:
            raw = self.llm.generate(
                system_prompt="你是严格的引用一致性审计器，只依据给定原文判断证据支持关系。",
                user_prompt=prompt,
                temperature=0,
                # DeepSeek 思考 Token 也计入输出预算；并发下 900 容易截断 JSON。
                # 保留 1500 的安全预算，Prompt 已要求短 claim/reason 控制正常开销。
                max_tokens=1500,
                response_format={"type": "json_object"},
                # 引用核验是受限证据分类，不需要开放式推理；关闭思考可显著降低
                # 延迟并避免 reasoning token 挤占 JSON 输出预算。
                thinking=False,
                stage="citation_verification",
                timeout=timeout,
            )
            payload = self._parse_json(raw)
            items = [
                CitationVerificationItem(
                    claim=str(item.get("claim", "")),
                    citation_ids=[str(value) for value in item.get("citation_ids", [])],
                    verdict=str(item.get("verdict", "unsupported")).lower(),
                    reason=str(item.get("reason", "")),
                )
                for item in payload.get("claims", [])
            ]
            uncited = [str(value) for value in payload.get("uncited_claims", [])]
            if not items:
                return CitationVerification(
                    status="unverified",
                    message="核验模型没有返回可解析的事实结论",
                )
            failed = bool(uncited) or any(
                item.verdict not in {"supported"} for item in items
            )
            return CitationVerification(
                status="failed" if failed else "verified",
                items=items,
                uncited_claims=uncited,
                message="存在未被原文完整支持的结论" if failed else "所有事实结论均有原文支持",
            )
        except Exception as exc:
            return CitationVerification(
                status="unverified",
                message=f"引用核验执行失败：{type(exc).__name__}",
            )

    def apply_policy(self, answer: str, result: CitationVerification) -> str:
        """严格模式失败关闭；宽松模式保留答案并仅返回核验状态。"""
        if not self.strict or result.status in {"verified", "skipped"}:
            return answer
        if result.status == "unverified":
            return self.SAFE_UNVERIFIED_ANSWER
        return self.SAFE_FAILURE_ANSWER

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = (raw or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError("核验模型未返回 JSON")
            return json.loads(match.group(0))
