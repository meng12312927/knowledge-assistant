"""Validate Golden Dataset schema and evidence against the versioned corpus."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from rag.chunk_identity import normalize_chunk_text
from tests.benchmark.generate_golden import load_document_chunks
from tests.benchmark.splits import (
    EXPECTED_SPLIT_COUNTS,
    SPLIT_ORDER,
    load_split,
)


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
    *,
    allow_unlabeled_answerable: bool = False,
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
        if status not in {"answerable", "partially_answerable", "not_found"}:
            errors.append(f"{label}: expected_answer_status 非法")
            continue
        expected_subquestions = item.get("expected_subquestions")
        if expected_subquestions is not None:
            if not isinstance(expected_subquestions, list):
                errors.append(f"{label}: expected_subquestions 必须是数组")
            else:
                subquestion_ids = []
                for subquestion in expected_subquestions:
                    subquestion_id = str(subquestion.get("id") or "")
                    subquestion_ids.append(subquestion_id)
                    sub_status = subquestion.get("status")
                    sub_chunks = subquestion.get("expected_chunks")
                    if not re.fullmatch(r"SQ[1-9]\d*", subquestion_id):
                        errors.append(f"{label}: 子问题 id 非法：{subquestion_id}")
                    if sub_status not in {"answerable", "not_found"}:
                        errors.append(
                            f"{label}/{subquestion_id}: status 非法"
                        )
                    if not isinstance(sub_chunks, list):
                        errors.append(
                            f"{label}/{subquestion_id}: expected_chunks 必须是数组"
                        )
                    elif sub_status == "not_found" and sub_chunks:
                        errors.append(
                            f"{label}/{subquestion_id}: not_found 不应有 Chunk"
                        )
                    elif sub_status == "answerable" and not sub_chunks:
                        errors.append(
                            f"{label}/{subquestion_id}: answerable 必须有 Chunk"
                        )
                    elif isinstance(sub_chunks, list):
                        missing_sub = [
                            value for value in sub_chunks if value not in catalog
                        ]
                        if missing_sub:
                            errors.append(
                                f"{label}/{subquestion_id}: Chunk 不存在：{missing_sub}"
                            )
                if len(subquestion_ids) != len(set(subquestion_ids)):
                    errors.append(f"{label}: 子问题 id 不应重复")
        if status == "partially_answerable" and not expected_subquestions:
            errors.append(
                f"{label}: partially_answerable 必须标注 expected_subquestions"
            )
        if (
            not allow_unlabeled_answerable
            and not str(item.get("answer") or "").strip()
        ):
            errors.append(f"{label}: answer 不能为空")
        if item.get("difficulty") not in {"easy", "medium", "hard"}:
            errors.append(f"{label}: difficulty 非法")
        if not str(item.get("dimension") or "").strip():
            errors.append(f"{label}: dimension 不能为空")
        if not isinstance(item.get("tags"), list):
            errors.append(f"{label}: tags 必须是数组")
        if not isinstance(expected_chunks, list):
            errors.append(f"{label}: expected_chunks 必须是数组")
            continue
        if len(expected_chunks) != len(set(expected_chunks)):
            errors.append(f"{label}: expected_chunks 不应包含重复 ID")
        if status == "not_found":
            if expected_chunks:
                errors.append(f"{label}: not_found 不应包含 expected_chunks")
            continue
        if not expected_chunks and allow_unlabeled_answerable:
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


def validate_split_suite(
    catalog: dict[str, dict[str, Any]],
) -> list[str]:
    """校验三个分片的数量、内部标签和跨分片隔离。"""
    errors: list[str] = []
    seen_ids: dict[str, str] = {}
    seen_questions: dict[str, str] = {}
    for split_name in SPLIT_ORDER:
        questions = load_split(split_name)
        split_errors = validate_questions(
            questions,
            catalog,
            expected_count=EXPECTED_SPLIT_COUNTS[split_name],
            allow_unlabeled_answerable=split_name == "calibration",
        )
        errors.extend(f"{split_name}: {error}" for error in split_errors)
        for item in questions:
            item_id = str(item.get("id") or "")
            question = "".join(str(item.get("question") or "").split())
            if item_id in seen_ids:
                errors.append(
                    f"跨分片重复 id：{item_id} "
                    f"({seen_ids[item_id]}, {split_name})"
                )
            else:
                seen_ids[item_id] = split_name
            if question in seen_questions:
                errors.append(
                    f"跨分片重复 question：{item.get('question')} "
                    f"({seen_questions[question]}, {split_name})"
                )
            else:
                seen_questions[question] = split_name
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 Golden Dataset")
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--questions", type=Path, default=Path("tests/benchmark/questions.json")
    )
    dataset_group.add_argument(
        "--split", choices=list(SPLIT_ORDER), help="校验一个命名分片"
    )
    dataset_group.add_argument(
        "--all-splits", action="store_true", help="校验全部分片及相互隔离"
    )
    parser.add_argument("--corpus", type=Path, default=Path("tests/corpus"))
    parser.add_argument("--expected-count", type=int, default=100)
    args = parser.parse_args()

    if args.all_splits:
        catalog = build_catalog(args.corpus)
        errors = validate_split_suite(catalog)
        if errors:
            print(f"[FAIL] Golden Dataset 分片校验失败：{len(errors)} 项")
            for error in errors[:100]:
                print(f"  - {error}")
            raise SystemExit(1)
        print(
            f"[OK] splits={len(SPLIT_ORDER)} "
            f"questions={sum(EXPECTED_SPLIT_COUNTS.values())} "
            f"chunks={len(catalog)}"
        )
        return

    if args.split:
        questions = load_split(args.split)
        expected_count = EXPECTED_SPLIT_COUNTS[args.split]
        allow_unlabeled = args.split == "calibration"
    else:
        questions = json.loads(args.questions.read_text(encoding="utf-8"))
        expected_count = args.expected_count
        allow_unlabeled = False
    if not isinstance(questions, list):
        raise SystemExit("questions.json 顶层必须是数组")
    catalog = build_catalog(args.corpus)
    errors = validate_questions(
        questions,
        catalog,
        expected_count,
        allow_unlabeled_answerable=allow_unlabeled,
    )
    if errors:
        print(f"[FAIL] Golden Dataset 校验失败：{len(errors)} 项")
        for error in errors[:50]:
            print(f"  - {error}")
        raise SystemExit(1)
    status_counts = Counter(item["expected_answer_status"] for item in questions)
    print(
        f"[OK] questions={len(questions)} chunks={len(catalog)} "
        f"answerable={status_counts['answerable']} "
        f"partially_answerable={status_counts['partially_answerable']} "
        f"not_found={status_counts['not_found']}"
    )


if __name__ == "__main__":
    main()
