"""生成后引用核验：检查答案结论是否被引用原文真实支持。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from models.document import (
    Citation,
    CitationVerification,
    CitationVerificationItem,
    SubquestionTrace,
)


class CitationVerifier:
    """先做确定性编号校验，再使用 LLM 做逐结论证据蕴含判断。"""

    SAFE_FAILURE_ANSWER = "引用核验未通过，本次回答未展示。请重试或查看核验详情。"
    SAFE_UNVERIFIED_ANSWER = "引用核验服务暂时不可用，本次回答未展示。请稍后重试。"

    def __init__(self, llm, enabled: bool = True, strict: bool = True):
        self.llm = llm
        self.enabled = enabled
        self.strict = strict

    @dataclass
    class _Claim:
        claim_id: str
        text: str
        citation_ids: List[str]
        subquestion_id: Optional[str] = None

    @classmethod
    def _extract_claims(
        cls, answer: str, citations: dict[str, Citation]
    ) -> List["CitationVerifier._Claim"]:
        """Deterministically split factual sentences and bind adjacent citations."""
        claims: List[CitationVerifier._Claim] = []
        for raw_line in (answer or "").splitlines():
            raw = raw_line.strip()
            has_citation = bool(re.search(r"\[S[1-9]\d*\]", raw))
            if not has_citation and (
                re.match(r"^#{1,6}\s+", raw)
                or re.fullmatch(r"\*\*[^*]+\*\*", raw)
                or (
                    len(raw) <= 45
                    and re.match(r"^[一二三四五六七八九十]+[、.．]", raw)
                    and not re.search(r"[。！？；!?;]", raw)
                )
            ):
                continue
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s*", "", raw_line).strip()
            line = re.sub(r"[*_#]", "", line).strip()
            if not line or line in {"可以确认", "暂无法确认", "回答", "结论"}:
                continue
            if re.fullmatch(r"[【\[（(]?[^：:]{1,20}[】\]）)]?[：:]", line):
                continue
            if not re.search(r"\[S[1-9]\d*\]", line) and line.endswith(("：", ":")):
                continue
            if (
                not re.search(r"\[S[1-9]\d*\]", line)
                and len(line) <= 40
                and (
                    re.search(r"SQ\d+|(?:餐补|住宿|审批|预算|合同|资格).*(?:标准|流程|要求|问题)", line)
                    or (
                        re.search(r"(?:要求|规定|流程|标准|结论|回答)$", line)
                        and not re.search(r"应|不得|必须|可以|需要|为|是", line)
                    )
                )
            ):
                continue
            # Coverage/refusal statements describe system state, not policy facts.
            if any(
                phrase in line
                for phrase in (
                    "根据现有知识库，可以回答",
                    "根据现有知识库，无法",
                    "根据现有知识库无法",
                    "根据现有资料，该问题无法",
                    "现有参考资料中未找到",
                    "现有参考资料中未出现",
                    "现有资料中未找到",
                    "现有资料中未出现",
                    "现有知识库未提供",
                    "现有资料未提供",
                    "资料未说明",
                    "资料中未说明",
                    "用户问题中未说明",
                    "无法确定唯一",
                    "引用核验未通过",
                    "引用核验服务暂时不可用",
                )
            ):
                continue
            if "未提及" in line or (
                re.search(r"知识库|现有资料|参考资料", line)
                and re.search(r"无直接规定|无法确认|无法给出", line)
            ):
                continue
            if re.search(r"现有(?:知识库|资料|参考资料).*(?:未明确说明|无法确认)", line):
                continue
            if (
                not has_citation
                and (
                    re.search(r"(?:建议|请).*(?:咨询|确认|为准)", line)
                    or re.search(r"未(?:明确)?说明.*是否", line)
                    or (len(line) <= 35 and line.startswith("关于"))
                )
            ):
                continue
            parts = [
                value.strip()
                for value in re.split(
                    r"(?<=[。！？；!?;])(?!\s*\[S[1-9]\d*\])\s*"
                    r"|(?<=\])\s*(?=[^\s\[])",
                    line,
                )
                if value.strip()
            ]
            for part in parts:
                citation_ids = list(dict.fromkeys(re.findall(r"\[(S[1-9]\d*)\]", part)))
                text = re.sub(r"\[(S[1-9]\d*)\]", "", part).strip(" -—\t")
                text = text.rstrip("。！？；!?; ")
                if not text:
                    continue
                sq_ids = {
                    sq_id
                    for citation_id in citation_ids
                    for sq_id in (
                        citations[citation_id].subquestion_ids
                        if citation_id in citations else []
                    )
                }
                claims.append(
                    cls._Claim(
                        claim_id=f"C{len(claims) + 1}",
                        text=text,
                        citation_ids=citation_ids,
                        subquestion_id=next(iter(sq_ids)) if len(sq_ids) == 1 else None,
                    )
                )
        return claims

    def verify(
        self,
        answer: str,
        citations: List[Citation],
        answer_status: str = "answerable",
        timeout: Optional[float] = None,
        subquestions: Optional[List[SubquestionTrace]] = None,
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
        claims = self._extract_claims(answer, available)
        if not claims:
            return CitationVerification(
                status="unverified",
                message="答案中没有可核验的事实结论",
            )
        evidence = "\n\n".join(
            f"[{citation_id}] {available[citation_id].source_file}\n"
            f"{available[citation_id].content}"
            for citation_id in sorted(available)
        )
        subquestion_context = ""
        if subquestions:
            subquestion_context = "\n【子问题】\n" + "\n".join(
                f"{item.subquestion_id} [{item.status}]：{item.query}"
                for item in subquestions
            )
        claim_lines = "\n".join(
            f"{claim.claim_id} | subquestion={claim.subquestion_id or 'unknown'} "
            f"| citations={','.join(claim.citation_ids) or 'NONE'} | {claim.text}"
            for claim in claims
        )
        prompt = f"""请逐条核验程序已经切分并绑定引用的事实结论。

【待核验结论】
{claim_lines}

【引用原文】
{evidence}
{subquestion_context}

严格规则：
1. 不得合并、拆分、遗漏或改写结论；必须为每个 claim_id 返回且只返回一项。
2. 已绑定 citation_ids 的结论必须原样保留编号，不得增加、删除或替换。
   citations=NONE 的结论若被某段原文直接完整支持，应返回真实证据编号并判 supported；
   找不到直接证据则 citation_ids=[] 且判 uncited/unsupported。
3. supported：原文直接支持完整结论；partial：只支持部分；unsupported：不支持或矛盾；uncited：事实结论没有引用。
4. 不得使用外部知识，不得因为结论看起来合理就判为 supported。
5. 如果提供了子问题，每条事实结论必须填写对应的 subquestion_id；不得把 not_found 子问题判为 supported。
6. “根据现有知识库无法确认某子问题”属于覆盖状态说明，不算需要引用的事实结论。
7. reason 不超过 20 个汉字，避免复述原文。
8. 只输出紧凑 JSON，不要输出 Markdown：
{{"claims":[{{"claim_id":"C1","subquestion_id":"SQ1","citation_ids":["S1"],"verdict":"supported","reason":"..."}}]}}
"""
        try:
            raw = self.llm.generate(
                system_prompt="你是严格的引用一致性审计器，只依据给定原文判断证据支持关系。",
                user_prompt=prompt,
                temperature=0,
                # Claim 已由程序切分，输出只做分类；小预算可稳定路由到 fast model。
                max_tokens=min(900, max(320, len(claims) * 110)),
                response_format={"type": "json_object"},
                # 引用核验是受限证据分类，不需要开放式推理；关闭思考可显著降低
                # 延迟并避免 reasoning token 挤占 JSON 输出预算。
                thinking=False,
                stage="citation_verification",
                timeout=timeout,
            )
            payload = self._parse_json(raw)
            raw_items = payload.get("claims") or []
            raw_by_id = {
                str(item.get("claim_id")): item
                for item in raw_items
                if item.get("claim_id")
            }
            # Backward-compatible positional fallback supports older local model
            # prompts, while missing/duplicate production IDs still become failures.
            if not raw_by_id and len(raw_items) == len(claims):
                raw_by_id = {
                    claim.claim_id: raw_items[index]
                    for index, claim in enumerate(claims)
                }
            items = []
            for claim in claims:
                raw_item = raw_by_id.get(claim.claim_id) or {}
                returned_ids = [
                    str(value) for value in raw_item.get("citation_ids", [])
                ]
                verdict = str(raw_item.get("verdict", "unsupported")).lower()
                reason = str(raw_item.get("reason", ""))
                if claim.citation_ids and returned_ids != claim.citation_ids:
                    verdict = "unsupported"
                    reason = "核验返回的引用绑定不一致"
                elif not claim.citation_ids:
                    if any(value not in available for value in returned_ids):
                        verdict = "unsupported"
                        reason = "核验返回了不存在的引用"
                    elif verdict == "supported" and not returned_ids:
                        verdict = "uncited"
                        reason = "结论没有可绑定证据"
                effective_ids = (
                    claim.citation_ids if claim.citation_ids else returned_ids
                )
                inferred_sq_ids = {
                    sq_id
                    for citation_id in effective_ids
                    for sq_id in (
                        available[citation_id].subquestion_ids
                        if citation_id in available else []
                    )
                }
                raw_subquestion = str(
                    raw_item.get("subquestion_id") or ""
                ).strip()
                if raw_subquestion.lower() in {"unknown", "none", "null"}:
                    raw_subquestion = ""
                items.append(
                    CitationVerificationItem(
                        claim_id=claim.claim_id,
                        claim=claim.text,
                        subquestion_id=(
                            claim.subquestion_id
                            or (
                                next(iter(inferred_sq_ids))
                                if len(inferred_sq_ids) == 1 else None
                            )
                            or raw_subquestion
                            or None
                        ),
                        citation_ids=effective_ids,
                        verdict=verdict,
                        reason=reason,
                    )
                )
            if subquestions:
                allowed = {
                    item.subquestion_id
                    for item in subquestions
                    if item.status != "not_found"
                }
                for item in items:
                    if item.subquestion_id not in allowed:
                        item.verdict = "unsupported"
                        item.reason = "子问题无证据或编号无效"
            uncited = [
                item.claim
                for item in items
                if item.verdict != "supported" and not item.citation_ids
            ]
            if not items:
                return CitationVerification(
                    status="unverified",
                    message="核验模型没有返回可解析的事实结论",
                )
            failed = bool(uncited) or any(
                item.verdict not in {"supported"} for item in items
            )
            has_supported = any(item.verdict == "supported" for item in items)
            supported_count = sum(item.verdict == "supported" for item in items)
            verification_status = "failed" if failed else "verified"
            # Claim-level strict mode salvages the supported subset instead of
            # hiding an otherwise useful answer because one independent claim
            # failed verification.  The rejected claims are removed below and
            # the partial state remains visible in Trace/UI.
            if failed and has_supported:
                verification_status = "partially_verified"
            return CitationVerification(
                status=verification_status,
                items=items,
                uncited_claims=uncited,
                total_claims=len(items),
                supported_claims=supported_count,
                claim_coverage_rate=(supported_count / len(items) if items else None),
                message="存在未被原文完整支持的结论" if failed else "所有事实结论均有原文支持",
            )
        except Exception as exc:
            return CitationVerification(
                status="unverified",
                message=f"引用核验执行失败：{type(exc).__name__}",
            )

    def apply_policy(
        self,
        answer: str,
        result: CitationVerification,
        answer_status: str = "answerable",
    ) -> str:
        """严格模式失败关闭；宽松模式保留答案并仅返回核验状态。"""
        if not self.strict or result.status in {"verified", "skipped"}:
            return answer
        if result.status == "partially_verified":
            sanitized = self._remove_unsupported_claims(answer, result)
            if sanitized and re.search(r"\[S[1-9]\d*\]", sanitized):
                return sanitized
            return self.SAFE_FAILURE_ANSWER
        if result.status == "unverified":
            return self.SAFE_UNVERIFIED_ANSWER
        return self.SAFE_FAILURE_ANSWER

    @staticmethod
    def _remove_unsupported_claims(
        answer: str, result: CitationVerification
    ) -> str:
        """Remove only lines containing rejected claims; keep verified branches."""
        rejected = [
            item.claim.strip()
            for item in result.items
            if item.verdict != "supported" and item.claim.strip()
        ] + [claim.strip() for claim in result.uncited_claims if claim.strip()]
        if not rejected:
            return answer
        kept = []
        for line in (answer or "").splitlines():
            normalized = re.sub(r"\s+", "", line)
            if any(
                re.sub(r"\s+", "", claim) in normalized
                or normalized in re.sub(r"\s+", "", claim)
                for claim in rejected
                if normalized
            ):
                continue
            kept.append(line)
        return "\n".join(kept).strip()

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
