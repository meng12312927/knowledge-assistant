"""Generate reviewable Golden Dataset candidates from the versioned test corpus.

This command is intentionally separate from the regression runner. A benchmark
run never rewrites its own labels. Regenerate candidates only when the corpus or
label policy changes, review the diff, then commit ``questions.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from ingestion.loaders.factory import DocumentLoaderFactory
from ingestion.splitters.factory import get_splitter
from rag.chunk_identity import normalize_chunk_text, stable_chunk_id
from rag.llm.factory import _provider_config


DEFAULT_NOT_FOUND = [
    {
        "id": "not-found-001",
        "question": "公司员工食堂早餐几点供应？",
        "answer": "根据现有知识库无法确定。",
        "expected_chunks": [],
        "expected_sources": [],
        "evidence": [],
        "expected_answer_status": "not_found",
        "dimension": "not_found",
        "difficulty": "easy",
        "tags": ["boundary", "hallucination"],
    },
    {
        "id": "not-found-002",
        "question": "公司停车场每月收费多少钱？",
        "answer": "根据现有知识库无法确定。",
        "expected_chunks": [],
        "expected_sources": [],
        "evidence": [],
        "expected_answer_status": "not_found",
        "dimension": "not_found",
        "difficulty": "easy",
        "tags": ["boundary", "hallucination"],
    },
    {
        "id": "not-found-003",
        "question": "员工健身房周末开放到几点？",
        "answer": "根据现有知识库无法确定。",
        "expected_chunks": [],
        "expected_sources": [],
        "evidence": [],
        "expected_answer_status": "not_found",
        "dimension": "not_found",
        "difficulty": "easy",
        "tags": ["boundary", "hallucination"],
    },
    {
        "id": "not-found-004",
        "question": "公司班车经过哪些地铁站？",
        "answer": "根据现有知识库无法确定。",
        "expected_chunks": [],
        "expected_sources": [],
        "evidence": [],
        "expected_answer_status": "not_found",
        "dimension": "not_found",
        "difficulty": "easy",
        "tags": ["boundary", "hallucination"],
    },
    {
        "id": "not-found-005",
        "question": "员工宿舍允许饲养宠物吗？",
        "answer": "根据现有知识库无法确定。",
        "expected_chunks": [],
        "expected_sources": [],
        "evidence": [],
        "expected_answer_status": "not_found",
        "dimension": "not_found",
        "difficulty": "easy",
        "tags": ["boundary", "hallucination"],
    },
]


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "policy"


def load_document_chunks(path: Path, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    loader = DocumentLoaderFactory.get_loader(str(path))
    raw_chunks = loader.load(str(path))
    splitter = get_splitter(
        strategy="markdown" if path.suffix.lower() in {".md", ".markdown"} else "recursive",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split(raw_chunks)
    result = []
    for chunk in chunks:
        metadata = {**chunk.metadata, "source_file": path.name}
        result.append(
            {
                "stable_chunk_id": stable_chunk_id(chunk.content, metadata),
                "content": chunk.content,
            }
        )
    return result


def generate_for_document(
    client,
    model: str,
    path: Path,
    chunks: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    chunk_text = "\n\n".join(
        f"CHUNK_ID: {item['stable_chunk_id']}\n{item['content']}" for item in chunks
    )
    prompt = f"""请为企业制度 RAG 回归测试生成 {count} 条互不重复的黄金问答。

来源文件：{path.name}

已分块原文：
{chunk_text}

只输出 JSON 对象，格式：
{{"questions":[{{
  "question":"具体、自然的员工问题",
  "answer":"仅依据原文的简洁标准答案",
  "expected_chunks":["上方真实 CHUNK_ID"],
  "evidence":["从对应 chunk 原文逐字复制的关键短句"],
  "dimension":"factual|procedure|numeric_reasoning|conditional|exception|multi_fact|faithfulness",
  "difficulty":"easy|medium|hard",
  "tags":["主题标签"]
}}]}}

要求：
1. 每题必须能被 expected_chunks 直接支持，不得推断制度之间未写明的先后关系。
2. evidence 必须逐字复制原文中的连续短句，不得改写。
3. expected_chunks 只能使用上方真实 ID；一般使用 1 个，确需综合时最多 2 个。
4. 覆盖不同章节，至少包含 1 个金额/期限/数量问题（若原文存在）和 1 个流程或例外问题。
5. 不生成知识库外问题，不输出 Markdown。"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是严格的 RAG 黄金测试集标注员，只能依据给定原文生成标签。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2500,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    payload = json.loads(response.choices[0].message.content)
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError("模型响应缺少 questions 数组")
    return questions


def validate_generated(
    raw: dict[str, Any],
    path: Path,
    chunks: list[dict[str, Any]],
) -> tuple[bool, str]:
    by_id = {item["stable_chunk_id"]: item["content"] for item in chunks}
    expected = raw.get("expected_chunks")
    evidence = raw.get("evidence")
    if not str(raw.get("question") or "").strip() or not str(raw.get("answer") or "").strip():
        return False, "question/answer 不能为空"
    if not isinstance(expected, list) or not expected:
        return False, "expected_chunks 必须是非空数组"
    if any(chunk_id not in by_id for chunk_id in expected):
        return False, "包含不存在的 expected_chunk"
    if not isinstance(evidence, list) or not evidence:
        return False, "evidence 必须是非空数组"
    expected_text = "\n".join(by_id[chunk_id] for chunk_id in expected)
    normalized_text = normalize_chunk_text(expected_text)
    if any(normalize_chunk_text(value) not in normalized_text for value in evidence):
        return False, f"evidence 不是 {path.name} 对应 chunk 的逐字片段"
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="生成待人工审阅的固定 Golden Dataset")
    parser.add_argument("--corpus", type=Path, default=Path("tests/corpus"))
    parser.add_argument(
        "--output", type=Path, default=Path("tests/benchmark/questions.json")
    )
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    files = sorted(args.corpus.rglob("*.txt"))
    if not files:
        raise SystemExit(f"未找到语料：{args.corpus}")
    answerable_target = args.target - len(DEFAULT_NOT_FOUND)
    if answerable_target < len(files):
        raise SystemExit("target 太小，无法覆盖全部制度文件")

    settings = get_settings()
    provider_name = settings.llm_primary_provider or settings.llm_default_provider
    provider = _provider_config(provider_name, settings)
    from openai import OpenAI

    client = OpenAI(
        api_key=provider.api_key,
        base_url=provider.base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
    )

    base_count, remainder = divmod(answerable_target, len(files))
    result: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    domain_counters: dict[str, int] = {}

    for file_index, path in enumerate(files):
        requested = base_count + (1 if file_index < remainder else 0)
        chunks = load_document_chunks(path, settings.chunk_size, settings.chunk_overlap)
        accepted: list[dict[str, Any]] = []
        for attempt in range(1, args.max_attempts + 1):
            generated = generate_for_document(
                client, provider.model, path, chunks, requested + 2
            )
            for raw in generated:
                valid, _ = validate_generated(raw, path, chunks)
                question = str(raw.get("question") or "").strip()
                if not valid or question in seen_questions:
                    continue
                seen_questions.add(question)
                domain = path.parent.name
                domain_counters[domain] = domain_counters.get(domain, 0) + 1
                accepted.append(
                    {
                        "id": f"{slugify(domain)}-{domain_counters[domain]:03d}",
                        "question": question,
                        "answer": str(raw["answer"]).strip(),
                        "expected_chunks": list(dict.fromkeys(raw["expected_chunks"])),
                        "expected_sources": [path.name],
                        "evidence": [str(value).strip() for value in raw["evidence"]],
                        "expected_answer_status": "answerable",
                        "dimension": str(raw.get("dimension") or "faithfulness"),
                        "difficulty": str(raw.get("difficulty") or "medium"),
                        "tags": [str(value) for value in raw.get("tags", [])],
                    }
                )
                if len(accepted) == requested:
                    break
            if len(accepted) == requested:
                break
            print(
                f"[RETRY] {path.name} attempt={attempt} "
                f"accepted={len(accepted)}/{requested}"
            )
        if len(accepted) != requested:
            raise SystemExit(
                f"{path.name} 仅生成 {len(accepted)}/{requested} 条有效问题"
            )
        result.extend(accepted)
        print(f"[OK] {path.name}: {len(accepted)}")

    result.extend(DEFAULT_NOT_FOUND)
    if len(result) != args.target:
        raise SystemExit(f"最终数量错误：{len(result)} != {args.target}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] Golden Dataset candidates: {args.output} ({len(result)})")


if __name__ == "__main__":
    main()
