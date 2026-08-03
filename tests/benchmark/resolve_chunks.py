"""使用生产摄取链离线解析并校验 Golden Dataset 的稳定 Chunk ID。

该脚本直接复用 ``DocumentLoaderFactory``、``get_splitter`` 和
``stable_chunk_id``，避免评测脚本复制一份分块算法后与生产实现漂移。

常用命令：

    python tests/benchmark/resolve_chunks.py
    python tests/benchmark/resolve_chunks.py --validate tests/benchmark/splits/blind_test.json
    python tests/benchmark/resolve_chunks.py --resolve tests/benchmark/splits/blind_test.json --refresh
"""

from __future__ import annotations

import argparse
import json
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


CORPUS_ROOT = PROJECT_ROOT / "tests" / "corpus"


def build_catalog(
    corpus_root: Path = CORPUS_ROOT,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, dict[str, Any]]:
    """按生产摄取实现构建 ``stable_chunk_id -> chunk`` 目录。"""
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    overlap = (
        settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    )
    catalog: dict[str, dict[str, Any]] = {}
    for file_path in sorted(corpus_root.rglob("*.txt")):
        loader = DocumentLoaderFactory.get_loader(str(file_path))
        raw_chunks = loader.load(str(file_path))
        splitter = get_splitter(
            strategy="recursive",
            chunk_size=size,
            chunk_overlap=overlap,
        )
        for chunk in splitter.split(raw_chunks):
            metadata = {**chunk.metadata, "source_file": file_path.name}
            chunk_id = stable_chunk_id(chunk.content, metadata)
            if chunk_id in catalog:
                raise ValueError(f"稳定 Chunk ID 冲突：{chunk_id}")
            catalog[chunk_id] = {
                "stable_chunk_id": chunk_id,
                "source_file": file_path.name,
                "content": chunk.content,
                "metadata": metadata,
            }
    return catalog


def find_chunk_for_evidence(
    evidence: str,
    source_files: list[str],
    catalog: dict[str, dict[str, Any]],
) -> list[str]:
    """返回包含证据逐字片段的全部候选 Chunk，顺序稳定。"""
    normalized_evidence = normalize_chunk_text(evidence)
    matches = []
    for chunk_id, info in catalog.items():
        if info["source_file"] not in source_files:
            continue
        if normalized_evidence in normalize_chunk_text(info["content"]):
            matches.append(chunk_id)
    return matches


def resolve_questions(
    questions: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    *,
    refresh: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """根据 ``evidence`` 解析标签；``refresh`` 可替换失效的旧标签。"""
    resolved: list[dict[str, Any]] = []
    errors: list[str] = []
    for original in questions:
        question = dict(original)
        if question.get("expected_answer_status") == "not_found":
            question["expected_chunks"] = []
            resolved.append(question)
            continue
        if question.get("expected_chunks") and not refresh:
            resolved.append(question)
            continue

        evidence_list = question.get("evidence") or []
        source_files = question.get("expected_sources") or []
        if not evidence_list or not source_files:
            errors.append(
                f"{question.get('id', '?')}: 缺少 evidence 或 expected_sources"
            )
            resolved.append(question)
            continue

        chunk_ids: list[str] = []
        for evidence in evidence_list:
            matches = find_chunk_for_evidence(
                str(evidence), list(source_files), catalog
            )
            if not matches:
                errors.append(
                    f"{question.get('id', '?')}: 找不到证据片段 "
                    f"{str(evidence)[:60]!r}，来源={source_files}"
                )
                continue
            # 一个证据片段通常只对应一个块；若多个块都包含它，全部保留，
            # 但稳定去重，避免多跳题重复标签扭曲 Recall。
            for chunk_id in matches:
                if chunk_id not in chunk_ids:
                    chunk_ids.append(chunk_id)
        question["expected_chunks"] = chunk_ids
        resolved.append(question)
    return resolved, errors


def validate_chunk_ids(
    questions: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for question in questions:
        label = question.get("id", "?")
        chunk_ids = question.get("expected_chunks") or []
        if len(chunk_ids) != len(set(chunk_ids)):
            errors.append(f"{label}: expected_chunks 包含重复 ID")
        for chunk_id in chunk_ids:
            if chunk_id not in catalog:
                errors.append(f"{label}: Chunk 不存在：{chunk_id}")
                continue
            sources = question.get("expected_sources") or []
            if sources and catalog[chunk_id]["source_file"] not in sources:
                errors.append(
                    f"{label}: {chunk_id} 来源不在 expected_sources 中"
                )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用生产分块实现解析 Golden Dataset Chunk ID"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--json", action="store_true", help="输出完整 Chunk 目录")
    mode.add_argument("--resolve", type=Path, help="解析指定题集的 expected_chunks")
    mode.add_argument("--validate", type=Path, help="只校验题集中的 Chunk ID")
    parser.add_argument("--in-place", action="store_true", help="覆盖 --resolve 输入文件")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="忽略已有 expected_chunks，按 evidence 全量重新解析",
    )
    parser.add_argument("--corpus", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    args = parser.parse_args()

    catalog = build_catalog(
        args.corpus,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.validate:
        questions = json.loads(args.validate.read_text(encoding="utf-8"))
        errors = validate_chunk_ids(questions, catalog)
        if errors:
            for error in errors:
                print(f"[FAIL] {error}", file=sys.stderr)
            raise SystemExit(1)
        print(f"[OK] questions={len(questions)} chunks={len(catalog)}")
        return

    if args.resolve:
        questions = json.loads(args.resolve.read_text(encoding="utf-8"))
        resolved, errors = resolve_questions(
            questions, catalog, refresh=args.refresh
        )
        if errors:
            for error in errors:
                print(f"[FAIL] {error}", file=sys.stderr)
            raise SystemExit(1)
        output_path = (
            args.resolve
            if args.in_place
            else args.resolve.with_suffix(".resolved.json")
        )
        output_path.write_text(
            json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[OK] resolved={len(resolved)} output={output_path}")
        return

    if args.json:
        print(json.dumps(catalog, ensure_ascii=False, indent=2))
        return

    by_file: dict[str, int] = {}
    for info in catalog.values():
        source = info["source_file"]
        by_file[source] = by_file.get(source, 0) + 1
    print(f"Catalog: {len(catalog)} chunks from {len(by_file)} documents")
    for source, count in sorted(by_file.items()):
        print(f"  {source}: {count}")


if __name__ == "__main__":
    main()
