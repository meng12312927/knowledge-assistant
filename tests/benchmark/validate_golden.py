"""Validate Golden Dataset schema and evidence against the versioned corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from tests.benchmark.generate_golden import load_document_chunks
from rag.chunk_identity import normalize_chunk_text


def build_catalog(corpus: Path) -> dict[str, dict[str, Any]]:
    settings = get_settings()
    catalog: dict[str, dict[str, Any]] = {}
    for path in sorted(corpus.rglob("*.txt")):
        for chunk in load_document_chunks(
            path, settings.chunk_size, settings.chunk_overlap
        ):
            chunk_id = chunk["stable_chunk_id"]
            if chunk_id in catalog:
                raise ValueError(f"稳定 Chunk ID 冲突：{chunk_id}")
            catalog[chunk_id] = {
                "source_file": path.name,
                "content": chunk["content"],
            }
    return catalog


def validate_questions(
    questions: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    expected_count: int | None = 100,
) -> list[str]:
    errors: list[str] = []
    if expected_count is not None and len(questions) != expected_count:
        errors.append(f"题目数量应为 {expected_count}，实际 {len(questions)}")

    ids = [str(item.get("id") or "") for item in questions]
    query_texts = [str(item.get("question") or "").strip() for item in questions]
    for value, count in Counter(ids).items():
        if not value:
            errors.append("存在空 id")
        elif count > 1:
            errors.append(f"重复 id：{value}")
    for value, count in Counter(query_texts).items():
        if not value:
            errors.append("存在空 question")
        elif count > 1:
            errors.append(f"重复 question：{value}")

    for index, item in enumerate(questions, 1):
        label = str(item.get("id") or f"#{index}")
        status = item.get("expected_answer_status")
        expected_chunks = item.get("expected_chunks")
        if status not in {"answerable", "not_found"}:
            errors.append(f"{label}: expected_answer_status 非法")
            continue
        if not str(item.get("answer") or "").strip():
            errors.append(f"{label}: answer 不能为空")
        if not isinstance(expected_chunks, list):
            errors.append(f"{label}: expected_chunks 必须是数组")
            continue
        if status == "not_found":
            if expected_chunks:
                errors.append(f"{label}: not_found 不应包含 expected_chunks")
            continue
        if not expected_chunks:
            errors.append(f"{label}: answerable 必须包含 expected_chunks")
            continue

        missing = [value for value in expected_chunks if value not in catalog]
        if missing:
            errors.append(f"{label}: expected_chunks 不存在：{missing}")
            continue
        expected_sources = set(item.get("expected_sources") or [])
        actual_sources = {catalog[value]["source_file"] for value in expected_chunks}
        if expected_sources and not actual_sources.issubset(expected_sources):
            errors.append(
                f"{label}: expected_sources={sorted(expected_sources)} "
                f"与 Chunk 来源={sorted(actual_sources)} 不一致"
            )
        combined = normalize_chunk_text(
            "\n".join(catalog[value]["content"] for value in expected_chunks)
        )
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{label}: evidence 必须是非空数组")
        else:
            for snippet in evidence:
                if normalize_chunk_text(str(snippet)) not in combined:
                    errors.append(f"{label}: evidence 不是 expected_chunks 的逐字片段")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="校验固定 Golden Dataset")
    parser.add_argument(
        "--questions", type=Path, default=Path("tests/benchmark/questions.json")
    )
    parser.add_argument("--corpus", type=Path, default=Path("tests/corpus"))
    parser.add_argument("--expected-count", type=int, default=100)
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise SystemExit("questions.json 顶层必须是数组")
    catalog = build_catalog(args.corpus)
    errors = validate_questions(questions, catalog, args.expected_count)
    if errors:
        print(f"[FAIL] Golden Dataset 校验失败：{len(errors)} 项")
        for error in errors[:50]:
            print(f"  - {error}")
        raise SystemExit(1)
    status_counts = Counter(item["expected_answer_status"] for item in questions)
    print(
        f"[OK] questions={len(questions)} chunks={len(catalog)} "
        f"answerable={status_counts['answerable']} "
        f"not_found={status_counts['not_found']}"
    )


if __name__ == "__main__":
    main()
