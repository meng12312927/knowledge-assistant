"""End-to-end RAG regression benchmark with Golden Dataset comparison.

Default usage:

    python tests/benchmark/benchmark.py --version v1.4

The command runs the blind-test split by default, writes JSON + Markdown
reports, and compares them with ``tests/benchmark/baselines/latest.json`` when
present.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.chunk_identity import stable_chunk_id
from tests.benchmark.splits import SPLIT_PATHS, load_split
from tests.smoke.smoke_test import (
    DEFAULT_CASES as DEFAULT_SMOKE_CASES,
    DEFAULT_OUTPUT_DIR as DEFAULT_SMOKE_OUTPUT_DIR,
    run_smoke_suite,
    write_report as write_smoke_report,
)


DEFAULT_SPLIT = "blind_test"
DEFAULT_QUESTIONS = SPLIT_PATHS[DEFAULT_SPLIT]
DEFAULT_BASELINE = PROJECT_ROOT / "tests/benchmark/baselines/latest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests/benchmark/results"
DEFAULT_THRESHOLDS = PROJECT_ROOT / "tests/benchmark/regression_thresholds.json"
DEFAULT_PRICING = PROJECT_ROOT / "tests/benchmark/pricing.json"
LEGACY_QUESTIONS = PROJECT_ROOT / "tests/benchmark/questions.json"
SCHEMA_VERSION = 3
BASELINE_SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def percentile(values: Iterable[Any], ratio: float) -> float | None:
    clean = sorted(
        number for value in values if (number := finite_number(value)) is not None
    )
    if not clean:
        return None
    index = min(len(clean) - 1, max(0, math.ceil(ratio * len(clean)) - 1))
    return clean[index]


def stats(values: Iterable[Any]) -> dict[str, Any]:
    clean = [
        number for value in values if (number := finite_number(value)) is not None
    ]
    if not clean:
        return {
            "count": 0,
            "avg": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(clean),
        "avg": round(statistics.fmean(clean), 4),
        "p50": round(percentile(clean, 0.50) or 0, 4),
        "p95": round(percentile(clean, 0.95) or 0, 4),
        "p99": round(percentile(clean, 0.99) or 0, 4),
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
    }


def recall_at(ranking: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    return len(set(ranking[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float | None:
    if not relevant:
        return None
    for rank, chunk_id in enumerate(ranking, 1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at(ranking: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranking[:k], 1)
        if chunk_id in relevant
    )
    ideal_count = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_subquestions(
    expected: list[Mapping[str, Any]],
    actual: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate branch-level status and evidence coverage when labels exist."""
    if not expected:
        return {
            "evaluated": 0,
            "status_accuracy": None,
            "evidence_recall": None,
            "complete_evidence": None,
        }
    actual_by_id = {
        str(item.get("subquestion_id")): item for item in actual
    }
    status_correct = 0
    evidence_scores: list[float] = []
    complete = True
    for item in expected:
        subquestion_id = str(item.get("id") or item.get("subquestion_id") or "")
        actual_item = actual_by_id.get(subquestion_id, {})
        expected_status = str(item.get("status") or "answerable")
        if actual_item.get("status") == expected_status:
            status_correct += 1
        relevant = {str(value) for value in item.get("expected_chunks") or []}
        if relevant:
            selected = {
                str(value)
                for value in actual_item.get("selected_stable_chunk_ids") or []
            }
            score = len(selected & relevant) / len(relevant)
            evidence_scores.append(score)
            complete = complete and score == 1.0
        elif expected_status != "not_found":
            complete = False
    return {
        "evaluated": len(expected),
        "status_accuracy": status_correct / len(expected),
        "evidence_recall": (
            statistics.fmean(evidence_scores) if evidence_scores else None
        ),
        "complete_evidence": complete,
    }


def mean_metric(records: list[dict[str, Any]], key: str) -> float | None:
    values = [
        finite_number(record.get("retrieval_metrics", {}).get(key))
        for record in records
    ]
    clean = [value for value in values if value is not None]
    return round(statistics.fmean(clean), 6) if clean else None


def get_path(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def load_json(path: Path, *, required: bool = True) -> Any:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_pricing(path: Path) -> dict[str, Any]:
    payload = load_json(path, required=False)
    if not isinstance(payload, dict):
        return {"currency": "CNY", "per_million_tokens": {}}
    return payload


def canonical_sha256(value: Any) -> str:
    """Return a stable fingerprint for JSON-compatible benchmark metadata."""
    import hashlib

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_provenance(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Capture source identity without making a dirty tree unbenchmarkable."""
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip() or None
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
    }


def service_binding(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Extract model, KB and calibrated threshold identity from service status."""
    models = metadata.get("models") or {}
    stats_payload = metadata.get("stats") or {}
    binding = {
        "knowledge_base_version": stats_payload.get("knowledge_base_version"),
        "models": {
            "primary": models.get("primary"),
            "fast": models.get("fast"),
            "verifier": models.get("verifier"),
            "fallback": models.get("fallback"),
            "embedding": models.get("embedding"),
            "reranker": {
                key: (models.get("reranker") or {}).get(key)
                for key in ("provider", "model", "candidate_k", "top_n")
            },
        },
        "thresholds": {
            "simple_query_min_rrf_score": (
                models.get("retrieval") or {}
            ).get("simple_query_min_rrf_score"),
            "reranker_not_found_threshold": (
                models.get("reranker") or {}
            ).get("not_found_threshold"),
            "answer_status_threshold_low": (
                models.get("retrieval") or {}
            ).get("answer_status_threshold_low"),
            "answer_status_threshold_high": (
                models.get("retrieval") or {}
            ).get("answer_status_threshold_high"),
        },
    }
    binding["model_fingerprint"] = canonical_sha256(binding["models"])
    binding["threshold_fingerprint"] = canonical_sha256(binding["thresholds"])
    return binding


def build_provenance(
    dataset: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    binding = service_binding(metadata)
    return {
        "git": git_provenance(),
        "dataset_sha256": dataset.get("effective_sha256"),
        **binding,
    }


def baseline_snapshot(
    report: Mapping[str, Any], *, tier: str = "combined"
) -> dict[str, Any]:
    """只保留回归比较所需信息，避免把逐题 Trace 提交到 Git。"""
    return {
        "baseline_schema_version": BASELINE_SCHEMA_VERSION,
        "report_schema_version": report.get("schema_version"),
        "generated_at": report.get("generated_at"),
        "version": report.get("version"),
        "tier": tier,
        "dataset": report.get("dataset"),
        "provenance": report.get("provenance"),
        "config": report.get("config"),
        "environment": report.get("environment"),
        "summary": report.get("summary"),
    }


def write_baseline(
    report: Mapping[str, Any], path: Path, *, tier: str = "combined"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            baseline_snapshot(report, tier=tier), ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )


def component_cost(
    stage: Mapping[str, Any],
    prices: Mapping[str, Any],
    input_key: str,
    output_key: str | None = None,
    *,
    input_field: str = "prompt_tokens",
) -> float | None:
    prompt = int(stage.get(input_field, 0) or 0)
    completion = int(stage.get("completion_tokens", 0) or 0)
    if prompt == 0 and completion == 0:
        return 0.0
    input_price = finite_number(prices.get(input_key))
    output_price = finite_number(prices.get(output_key)) if output_key else 0.0
    if input_price is None or (completion and output_price is None):
        return None
    return (prompt * input_price + completion * (output_price or 0)) / 1_000_000


def calculate_costs(
    stage_usage: Mapping[str, Any], pricing: Mapping[str, Any]
) -> dict[str, Any]:
    prices = pricing.get("per_million_tokens", {})
    if not isinstance(prices, Mapping):
        prices = {}
    components = {
        "embedding": component_cost(
            stage_usage.get("embedding", {}), prices, "embedding_input"
        ),
        "query_rewrite": component_cost(
            stage_usage.get("query_rewrite", {}),
            prices,
            "query_rewrite_input",
            "query_rewrite_output",
        ),
        "generation": component_cost(
            stage_usage.get("generation", {}),
            prices,
            "generation_input",
            "generation_output",
        ),
        "verification": component_cost(
            stage_usage.get("citation_verification", {}),
            prices,
            "verification_input",
            "verification_output",
        ),
        "reranker": component_cost(
            stage_usage.get("rerank", {}),
            prices,
            "reranker_input",
            input_field="reranker_tokens",
        ),
    }
    complete = all(value is not None for value in components.values())
    return {
        **components,
        "total": sum(value for value in components.values() if value is not None)
        if complete
        else None,
        "complete": complete,
    }


def unsupported_claim_count(verification: Mapping[str, Any]) -> int:
    bad_verdicts = {"partial", "unsupported", "uncited"}
    items = verification.get("items") or []
    count = sum(
        1
        for item in items
        if isinstance(item, Mapping)
        and str(item.get("verdict") or "").lower() in bad_verdicts
    )
    count += len(verification.get("invalid_citation_ids") or [])
    item_uncited_claims = {
        str(item.get("claim") or "")
        for item in items
        if isinstance(item, Mapping)
        and str(item.get("verdict") or "").lower() == "uncited"
    }
    count += sum(
        1
        for claim in verification.get("uncited_claims") or []
        if str(claim) not in item_uncited_claims
    )
    return count


class RegressionRunner:
    def __init__(
        self,
        endpoint: str,
        timeout: float,
        top_k: int,
        pricing: Mapping[str, Any],
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.top_k = top_k
        self.pricing = pricing
        self._local = threading.local()
        self.run_id = uuid.uuid4().hex[:10]

    def session(self) -> requests.Session:
        value = getattr(self._local, "session", None)
        if value is None:
            value = requests.Session()
            value.headers.update(
                {
                    "Accept": "text/event-stream",
                    "User-Agent": "knowledge-assistant-regression/1",
                }
            )
            self._local.session = value
        return value

    def run_one(
        self, item: Mapping[str, Any], index: int, *, warmup: bool = False
    ) -> dict[str, Any]:
        started = time.perf_counter()
        request_id = f"reg-{self.run_id}-{index}"
        record: dict[str, Any] = {
            "index": index,
            "id": item.get("id"),
            "question": item["question"],
            "expected_answer": item.get("answer"),
            "expected_chunks": item.get("expected_chunks", []),
            "expected_answer_status": item.get(
                "expected_answer_status", "answerable"
            ),
            "expected_subquestions": item.get("expected_subquestions", []),
            "dimension": item.get("dimension"),
            "difficulty": item.get("difficulty"),
            "tags": item.get("tags", []),
            "warmup": warmup,
            "request_id": request_id,
            "success": False,
            "error_type": None,
            "error": None,
            "event_order": [],
        }
        answer_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        verification: dict[str, Any] = {}
        trace: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        client_ttft_ms = None
        client_done_ms = None
        done_received = False

        try:
            response = self.session().post(
                f"{self.endpoint}/api/v1/chat/stream",
                json={
                    "query": item["question"],
                    "session_id": None,
                    "top_k": self.top_k,
                    "enable_agent": False,
                },
                headers={"X-Request-ID": request_id},
                stream=True,
                timeout=(min(10.0, self.timeout), self.timeout),
            )
            with response:
                record["http_status"] = response.status_code
                response.raise_for_status()
                for raw_line in response.iter_lines(
                    chunk_size=64, decode_unicode=True
                ):
                    if not raw_line:
                        continue
                    line = (
                        raw_line.decode("utf-8")
                        if isinstance(raw_line, bytes)
                        else raw_line
                    )
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:].lstrip())
                    event_type = str(event.get("type") or "unknown")
                    record["event_order"].append(event_type)
                    if event_type == "token":
                        content = str(event.get("content") or "")
                        if content:
                            answer_parts.append(content)
                            if client_ttft_ms is None:
                                client_ttft_ms = (
                                    time.perf_counter() - started
                                ) * 1000
                    elif event_type == "sources":
                        value = event.get("sources")
                        sources = value if isinstance(value, list) else []
                        record["answer_status"] = event.get("answer_status")
                    elif event_type == "citations":
                        value = event.get("citations")
                        citations = value if isinstance(value, list) else []
                    elif event_type == "citation_verification":
                        value = event.get("verification")
                        verification = value if isinstance(value, dict) else {}
                    elif event_type == "rag_trace":
                        value = event.get("trace")
                        trace = value if isinstance(value, dict) else {}
                    elif event_type == "metrics":
                        value = event.get("metrics")
                        if isinstance(value, dict):
                            metrics.update(value)
                    elif event_type == "error":
                        record["error_type"] = "sse_error"
                        record["error"] = str(event.get("message") or "unknown")
                        break
                    elif event_type == "done":
                        value = event.get("metrics")
                        if isinstance(value, dict):
                            metrics.update(value)
                        client_done_ms = (time.perf_counter() - started) * 1000
                        done_received = True
                        break
        except requests.Timeout as exc:
            record["error_type"] = "timeout"
            record["error"] = str(exc)
        except requests.RequestException as exc:
            record["error_type"] = "request_error"
            record["error"] = str(exc)
        except Exception as exc:
            record["error_type"] = "unexpected_error"
            record["error"] = f"{type(exc).__name__}: {exc}"

        if record["error_type"] is None and not done_received:
            record["error_type"] = "missing_done"
            record["error"] = "SSE 流结束但没有 done"

        record["success"] = record["error_type"] is None and done_received
        record["answer"] = "".join(answer_parts)
        record["sources"] = sources
        record["citations"] = citations
        record["citation_verification"] = verification
        record["trace"] = trace
        record["server_metrics"] = metrics
        record["source_count"] = len(sources)
        record["citation_count"] = len(citations)
        record["verification_status"] = verification.get("status")
        record["unsupported_claims"] = unsupported_claim_count(verification)
        record["client_user_visible_ttft_ms"] = (
            round(client_ttft_ms, 3) if client_ttft_ms is not None else None
        )
        record["client_done_latency_ms"] = (
            round(client_done_ms, 3) if client_done_ms is not None else None
        )
        record["generation_ttft_ms"] = trace.get("generation_ttft_ms")
        record["verified_ttft_ms"] = trace.get("verified_ttft_ms")
        record["server_done_emit_ms"] = (
            metrics.get("server_done_emit_ms") or trace.get("sse_total_latency_ms")
        )
        record["knowledge_base_version"] = trace.get("knowledge_base_version")
        record["query_strategy"] = trace.get("query_strategy")
        record["subquestions"] = trace.get("subquestions") or []
        record["evidence_coverage"] = trace.get("evidence_coverage")
        record["cache_hits"] = trace.get("cache_hits") or {}
        record["token_usage"] = trace.get("token_usage") or {}
        stage_usage = trace.get("stage_token_usage") or {}
        record["stage_token_usage"] = stage_usage
        record["cost"] = calculate_costs(stage_usage, self.pricing)

        candidate_ranking = [
            str(entry.get("stable_chunk_id"))
            for entry in trace.get("candidate_rankings") or []
            if entry.get("stable_chunk_id")
        ]
        rerank_ranking = [
            str(entry.get("stable_chunk_id"))
            for entry in trace.get("rerank_rankings") or []
            if entry.get("stable_chunk_id")
        ]
        if not rerank_ranking:
            rerank_ranking = [
                stable_chunk_id(
                    str(source.get("content") or ""),
                    source.get("metadata") or {},
                )
                for source in sources
            ]
        relevant = {str(value) for value in item.get("expected_chunks") or []}
        record["candidate_ranking"] = candidate_ranking
        record["rerank_ranking"] = rerank_ranking
        record["retrieval_metrics"] = {
            "recall_at_5": recall_at(candidate_ranking, relevant, 5),
            "recall_at_10": recall_at(candidate_ranking, relevant, 10),
            "mrr": reciprocal_rank(candidate_ranking, relevant),
            "ndcg_at_10": ndcg_at(candidate_ranking, relevant, 10),
            "rerank_recall_at_5": recall_at(rerank_ranking, relevant, 5),
            "rerank_mrr": reciprocal_rank(rerank_ranking, relevant),
        }
        record["subquestion_metrics"] = evaluate_subquestions(
            record["expected_subquestions"], record["subquestions"]
        )
        expected_status = record["expected_answer_status"]
        actual_status = record.get("answer_status")
        record["answer_status_correct"] = actual_status == expected_status
        record["hallucination"] = bool(
            record["unsupported_claims"]
            or (
                expected_status in {"answerable", "partially_answerable"}
                and record["verification_status"] == "failed"
            )
            or (expected_status == "not_found" and actual_status != "not_found")
        )
        record["attempt_latency_ms"] = round(
            (time.perf_counter() - started) * 1000, 3
        )
        return record


def endpoint_metadata(endpoint: str, timeout: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with requests.Session() as session:
        for key, path in {
            "health": "/health",
            "models": "/api/v1/models/status",
            "stats": "/api/v1/stats",
        }.items():
            try:
                response = session.get(
                    f"{endpoint.rstrip('/')}{path}",
                    timeout=(min(10.0, timeout), min(15.0, timeout)),
                )
                response.raise_for_status()
                result[key] = response.json()
            except Exception as exc:
                result[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def run_benchmark(
    questions: list[dict[str, Any]],
    endpoint: str,
    concurrency: int,
    timeout: float,
    top_k: int,
    warmup: int,
    pricing: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    runner = RegressionRunner(endpoint, timeout, top_k, pricing)
    warmup_item = {
        "id": "warmup",
        "question": "知识库可以回答哪些企业制度问题？",
        "answer": "",
        "expected_chunks": [],
        "expected_answer_status": "answerable",
    }
    for index in range(warmup):
        runner.run_one(warmup_item, -(index + 1), warmup=True)

    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    if concurrency == 1:
        for index, item in enumerate(questions, 1):
            print(f"[{index}/{len(questions)}] {item['id']} {item['question']}")
            records.append(runner.run_one(item, index))
    else:
        with ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="regression"
        ) as pool:
            futures = {
                pool.submit(runner.run_one, item, index): index
                for index, item in enumerate(questions, 1)
            }
            completed = 0
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                completed += 1
                print(
                    f"[{completed}/{len(questions)}] "
                    f"{record.get('id')} success={record.get('success')}"
                )
        records.sort(key=lambda value: value["index"])
    return records, time.perf_counter() - started


def _dimension_breakdown(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-dimension metrics breakdown."""
    by_dim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        dim = record.get("dimension") or "unknown"
        by_dim[dim].append(record)

    breakdown: dict[str, Any] = {}
    for dim, dim_records in sorted(by_dim.items()):
        answerable = [
            r for r in dim_records
            if r.get("expected_answer_status") in {"answerable", "partially_answerable"}
        ]
        verified = [
            r for r in answerable
            if r.get("verification_status") in {"verified", "partially_verified"}
        ]
        correct_status = sum(bool(r.get("answer_status_correct")) for r in dim_records)
        hallucinations = sum(bool(r.get("hallucination")) for r in dim_records)
        latency_values = [
            r["client_done_latency_ms"]
            for r in dim_records
            if r.get("success")
            and r.get("client_done_latency_ms") is not None
        ]
        breakdown[dim] = {
            "count": len(dim_records),
            "success_rate": round(
                sum(bool(r.get("success")) for r in dim_records) / len(dim_records), 6
            ) if dim_records else 0,
            "answer_status_accuracy": round(
                correct_status / len(dim_records), 6
            ) if dim_records else None,
            "verification_pass_rate": round(
                len(verified) / len(answerable), 6
            ) if answerable else None,
            "hallucination_rate": round(
                hallucinations / len(dim_records), 6
            ) if dim_records else None,
            "avg_latency_ms": (
                round(statistics.fmean(latency_values), 2)
                if latency_values else None
            ),
            "recall_at_10": mean_metric(answerable, "recall_at_10"),
        }
    return breakdown


def _difficulty_breakdown(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-difficulty metrics breakdown."""
    by_diff: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        diff = record.get("difficulty") or "unknown"
        by_diff[diff].append(record)

    breakdown: dict[str, Any] = {}
    for diff, diff_records in sorted(by_diff.items()):
        correct_status = sum(bool(r.get("answer_status_correct")) for r in diff_records)
        hallucinations = sum(bool(r.get("hallucination")) for r in diff_records)
        latency_values = [
            r["client_done_latency_ms"]
            for r in diff_records
            if r.get("success")
            and r.get("client_done_latency_ms") is not None
        ]
        breakdown[diff] = {
            "count": len(diff_records),
            "success_rate": round(
                sum(bool(r.get("success")) for r in diff_records) / len(diff_records), 6
            ) if diff_records else 0,
            "answer_status_accuracy": round(
                correct_status / len(diff_records), 6
            ) if diff_records else None,
            "hallucination_rate": round(
                hallucinations / len(diff_records), 6
            ) if diff_records else None,
            "avg_latency_ms": (
                round(statistics.fmean(latency_values), 2)
                if latency_values else None
            ),
        }
    return breakdown


def summarize(
    records: list[dict[str, Any]],
    wall_seconds: float,
    pricing: Mapping[str, Any],
) -> dict[str, Any]:
    successful = [record for record in records if record.get("success")]
    answerable = [
        record
        for record in successful
        if record.get("expected_answer_status") in {
            "answerable", "partially_answerable"
        }
    ]
    verified = [
        record
        for record in answerable
        if record.get("verification_status") in {
            "verified", "partially_verified"
        }
    ]
    subquestion_records = [
        record
        for record in successful
        if (record.get("subquestion_metrics") or {}).get("evaluated")
    ]
    cost_components = [
        "embedding", "query_rewrite", "generation", "verification", "reranker", "total"
    ]
    cost_summary: dict[str, Any] = {
        "currency": pricing.get("currency", "CNY"),
        "pricing_complete": all(
            bool(record.get("cost", {}).get("complete")) for record in successful
        )
        if successful
        else False,
    }
    for component in cost_components:
        values = [
            record.get("cost", {}).get(component) for record in successful
        ]
        # 只要有一次真实调用缺少价格，整轮该阶段金额就必须显示 N/A；
        # 不能让未调用阶段的 0 元样本掩盖未知价格。
        if any(value is None for value in values):
            component_stats = stats([])
            component_stats["sum"] = None
        else:
            component_stats = stats(values)
            component_stats["sum"] = (
                round(sum(float(value) for value in values), 8)
                if component_stats["count"]
                else None
            )
        cost_summary[component] = component_stats

    claim_total = sum(
        int((record.get("citation_verification") or {}).get("total_claims", 0) or 0)
        for record in successful
    )
    claim_supported = sum(
        int((record.get("citation_verification") or {}).get("supported_claims", 0) or 0)
        for record in successful
    )
    return {
        "samples": len(records),
        "successful": len(successful),
        "success_rate": round(len(successful) / len(records), 6) if records else 0,
        "error_rate": round(
            (len(records) - len(successful)) / len(records), 6
        )
        if records
        else 0,
        "wall_seconds": round(wall_seconds, 4),
        "successful_rps": round(len(successful) / wall_seconds, 6)
        if wall_seconds
        else None,
        "latency": {
            "client_done_ms": stats(
                record.get("client_done_latency_ms") for record in successful
            ),
            "generation_ttft_ms": stats(
                record.get("generation_ttft_ms") for record in successful
            ),
            "user_visible_ttft_ms": stats(
                record.get("client_user_visible_ttft_ms") for record in successful
            ),
        },
        "retrieval": {
            "evaluated": len(answerable),
            "recall_at_5": mean_metric(answerable, "recall_at_5"),
            "recall_at_10": mean_metric(answerable, "recall_at_10"),
            "mrr": mean_metric(answerable, "mrr"),
            "ndcg_at_10": mean_metric(answerable, "ndcg_at_10"),
            "rerank_recall_at_5": mean_metric(
                answerable, "rerank_recall_at_5"
            ),
            "rerank_mrr": mean_metric(answerable, "rerank_mrr"),
            "evidence_coverage": stats(
                record.get("evidence_coverage") for record in successful
            ).get("avg"),
            "subquestion_evidence_recall": (
                statistics.fmean(
                    record["subquestion_metrics"]["evidence_recall"]
                    for record in subquestion_records
                    if record["subquestion_metrics"].get("evidence_recall") is not None
                )
                if any(
                    record["subquestion_metrics"].get("evidence_recall") is not None
                    for record in subquestion_records
                )
                else None
            ),
            "complete_evidence_rate": (
                sum(
                    bool(record["subquestion_metrics"].get("complete_evidence"))
                    for record in subquestion_records
                ) / len(subquestion_records)
                if subquestion_records else None
            ),
        },
        "generation": {
            "verification_pass_rate": round(
                len(verified) / len(answerable), 6
            )
            if answerable
            else None,
            "verification_status": dict(
                Counter(
                    record.get("verification_status") or "unknown"
                    for record in successful
                )
            ),
            "total_claims": claim_total,
            "supported_claims": claim_supported,
            "claim_supported_rate": (
                round(claim_supported / claim_total, 6)
                if claim_total else None
            ),
            "answer_status_accuracy": round(
                sum(bool(record.get("answer_status_correct")) for record in successful)
                / len(successful),
                6,
            )
            if successful
            else None,
            "subquestion_status_accuracy": (
                statistics.fmean(
                    record["subquestion_metrics"]["status_accuracy"]
                    for record in subquestion_records
                    if record["subquestion_metrics"].get("status_accuracy") is not None
                )
                if any(
                    record["subquestion_metrics"].get("status_accuracy") is not None
                    for record in subquestion_records
                )
                else None
            ),
            "unsupported_claims": sum(
                int(record.get("unsupported_claims", 0)) for record in successful
            ),
            "hallucination_requests": sum(
                bool(record.get("hallucination")) for record in successful
            ),
            "hallucination_rate": round(
                sum(bool(record.get("hallucination")) for record in successful)
                / len(successful),
                6,
            )
            if successful
            else None,
        },
        "tokens": {
            "total": stats(
                (record.get("token_usage") or {}).get("total_tokens")
                for record in successful
            ),
            "reranker": stats(
                (record.get("token_usage") or {}).get("reranker_tokens")
                for record in successful
            ),
        },
        "cost": cost_summary,
        "query_strategy": dict(
            Counter(
                record.get("query_strategy") or "unknown" for record in successful
            )
        ),
        "knowledge_base_versions": dict(
            Counter(
                record.get("knowledge_base_version") or "unknown"
                for record in successful
            )
        ),
        "by_dimension": _dimension_breakdown(successful),
        "by_difficulty": _difficulty_breakdown(successful),
    }


COMPARISON_FIELDS = {
    "Latency Avg": ("latency.client_done_ms.avg", "lower"),
    "Latency P95": ("latency.client_done_ms.p95", "lower"),
    "Latency P99": ("latency.client_done_ms.p99", "lower"),
    "Generation TTFT Avg": ("latency.generation_ttft_ms.avg", "lower"),
    "User-visible TTFT Avg": ("latency.user_visible_ttft_ms.avg", "lower"),
    "Recall@5": ("retrieval.recall_at_5", "higher"),
    "Recall@10": ("retrieval.recall_at_10", "higher"),
    "MRR": ("retrieval.mrr", "higher"),
    "nDCG@10": ("retrieval.ndcg_at_10", "higher"),
    "Rerank Recall@5": ("retrieval.rerank_recall_at_5", "higher"),
    "Evidence Coverage": ("retrieval.evidence_coverage", "higher"),
    "Subquestion Evidence Recall": (
        "retrieval.subquestion_evidence_recall", "higher"
    ),
    "Complete Evidence Rate": ("retrieval.complete_evidence_rate", "higher"),
    "Verification Pass": ("generation.verification_pass_rate", "higher"),
    "Claim Supported Rate": ("generation.claim_supported_rate", "higher"),
    "Answer Status Accuracy": ("generation.answer_status_accuracy", "higher"),
    "Subquestion Status Accuracy": (
        "generation.subquestion_status_accuracy", "higher"
    ),
    "Unsupported Claims": ("generation.unsupported_claims", "lower"),
    "Hallucination Rate": ("generation.hallucination_rate", "lower"),
    "Error Rate": ("error_rate", "lower"),
    "Cost / Query": ("cost.total.avg", "lower"),
}


def compare(
    baseline: Mapping[str, Any] | None, current: Mapping[str, Any]
) -> dict[str, Any]:
    if not baseline:
        return {}
    result: dict[str, Any] = {}
    for label, (path, direction) in COMPARISON_FIELDS.items():
        old = finite_number(get_path(baseline, path))
        new = finite_number(get_path(current, path))
        delta = new - old if old is not None and new is not None else None
        delta_pct = (
            delta / old * 100 if delta is not None and old not in (None, 0) else None
        )
        result[label] = {
            "path": path,
            "direction": direction,
            "baseline": old,
            "current": new,
            "delta": round(delta, 8) if delta is not None else None,
            "delta_pct": round(delta_pct, 4) if delta_pct is not None else None,
        }
    return result


def evaluate_thresholds(
    comparison: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_path = {value.get("path"): value for value in comparison.values()}
    failures = []
    for path, rule in thresholds.items():
        metric = by_path.get(path)
        if not metric or metric.get("current") is None:
            continue
        new = float(metric["current"])
        minimum = finite_number(rule.get("minimum"))
        maximum = finite_number(rule.get("maximum"))
        if minimum is not None and new < minimum:
            failures.append({
                "path": path,
                "baseline": metric.get("baseline"),
                "current": new,
                "regression": new - minimum,
                "reason": "below_minimum",
                "rule": dict(rule),
            })
            continue
        if maximum is not None and new > maximum:
            failures.append({
                "path": path,
                "baseline": metric.get("baseline"),
                "current": new,
                "regression": new - maximum,
                "reason": "above_maximum",
                "rule": dict(rule),
            })
            continue
        if metric.get("baseline") is None:
            continue
        old = float(metric["baseline"])
        direction = rule.get("direction")
        mode = rule.get("mode", "absolute")
        tolerance = float(rule.get("tolerance", 0))
        if mode == "relative":
            if old == 0:
                continue
            regression = (new - old) / abs(old)
            violated = (
                regression > tolerance
                if direction == "lower"
                else regression < -tolerance
            )
        else:
            regression = new - old
            violated = (
                regression > tolerance
                if direction == "lower"
                else regression < -tolerance
            )
        if violated:
            failures.append(
                {
                    "path": path,
                    "baseline": old,
                    "current": new,
                    "regression": regression,
                    "reason": "baseline_regression",
                    "rule": dict(rule),
                }
            )
    return failures


def fmt(value: Any, decimals: int = 4) -> str:
    number = finite_number(value)
    return "N/A" if number is None else f"{number:.{decimals}f}"


def fmt_change(metric: Mapping[str, Any], *, percent: bool = False) -> str:
    delta = finite_number(metric.get("delta"))
    delta_pct = finite_number(metric.get("delta_pct"))
    if delta is None:
        return "N/A"
    if percent and delta_pct is not None:
        return f"{delta_pct:+.2f}%"
    return f"{delta:+.4f}"


def pair(
    baseline: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    path: str,
    decimals: int = 4,
    suffix: str = "",
) -> str:
    old = get_path(baseline or {}, path)
    new = get_path(current, path)
    if baseline is None:
        return f"{fmt(new, decimals)}{suffix}"
    old_number = finite_number(old)
    new_number = finite_number(new)
    if old_number is None or new_number is None:
        return f"{fmt(old, decimals)} → {fmt(new, decimals)}"
    delta_pct = (
        (new_number - old_number) / old_number * 100 if old_number != 0 else None
    )
    change = f"{delta_pct:+.2f}%" if delta_pct is not None else f"{new_number-old_number:+.{decimals}f}"
    return (
        f"{fmt(old_number, decimals)}{suffix} → "
        f"{fmt(new_number, decimals)}{suffix} ({change})"
    )


def markdown_report(report: Mapping[str, Any]) -> str:
    current = report["summary"]
    baseline = report.get("baseline_summary")
    decision = report["decision"]
    currency = current["cost"].get("currency", "CNY")
    currency_symbol = "¥" if currency == "CNY" else f"{currency} "
    lines = [
        "# Regression Benchmark",
        "",
        f"- Baseline: `{report.get('baseline_version') or 'N/A'}`",
        f"- Current: `{report['version']}`",
        f"- Dataset: `{report['dataset'].get('split') or 'custom'}` / "
        f"`{report['dataset']['sha256'][:12]}` "
        f"({report['dataset']['questions']} questions)",
        f"- Decision: **{decision}**",
        "",
        "## Pre-regression Smoke Gate",
        "",
    ]
    smoke_gate = report.get("smoke_gate")
    if smoke_gate:
        lines.append(
            f"**PASS** — {smoke_gate.get('passed')}/"
            f"{smoke_gate.get('cases')} cases passed."
        )
    else:
        lines.append("Skipped by explicit `--skip-smoke` diagnostic option.")
    lines.extend([
        "",
        "## Latency",
        "",
        "| Metric | Baseline → Current |",
        "|---|---:|",
        f"| Avg | {pair(baseline, current, 'latency.client_done_ms.avg', 2, ' ms')} |",
        f"| P50 | {pair(baseline, current, 'latency.client_done_ms.p50', 2, ' ms')} |",
        f"| P95 | {pair(baseline, current, 'latency.client_done_ms.p95', 2, ' ms')} |",
        f"| P99 | {pair(baseline, current, 'latency.client_done_ms.p99', 2, ' ms')} |",
        f"| Generation TTFT Avg | {pair(baseline, current, 'latency.generation_ttft_ms.avg', 2, ' ms')} |",
        f"| User-visible TTFT Avg | {pair(baseline, current, 'latency.user_visible_ttft_ms.avg', 2, ' ms')} |",
        "",
        "## Retrieval",
        "",
        "| Metric | Baseline → Current |",
        "|---|---:|",
        f"| Recall@5 | {pair(baseline, current, 'retrieval.recall_at_5')} |",
        f"| Recall@10 | {pair(baseline, current, 'retrieval.recall_at_10')} |",
        f"| MRR | {pair(baseline, current, 'retrieval.mrr')} |",
        f"| nDCG@10 | {pair(baseline, current, 'retrieval.ndcg_at_10')} |",
        f"| Rerank Recall@5 | {pair(baseline, current, 'retrieval.rerank_recall_at_5')} |",
        f"| Evidence Coverage | {pair(baseline, current, 'retrieval.evidence_coverage')} |",
        f"| Subquestion Evidence Recall | {pair(baseline, current, 'retrieval.subquestion_evidence_recall')} |",
        f"| Complete Evidence Rate | {pair(baseline, current, 'retrieval.complete_evidence_rate')} |",
        "",
        "## Generation",
        "",
        "| Metric | Baseline → Current |",
        "|---|---:|",
        f"| Verification Pass | {pair(baseline, current, 'generation.verification_pass_rate', 4)} |",
        f"| Claim Supported Rate | {pair(baseline, current, 'generation.claim_supported_rate', 4)} |",
        f"| Answer Status Accuracy | {pair(baseline, current, 'generation.answer_status_accuracy', 4)} |",
        f"| Subquestion Status Accuracy | {pair(baseline, current, 'generation.subquestion_status_accuracy', 4)} |",
        f"| Unsupported Claims | {pair(baseline, current, 'generation.unsupported_claims', 0)} |",
        f"| Hallucination Requests | {pair(baseline, current, 'generation.hallucination_requests', 0)} |",
        f"| Error Rate | {pair(baseline, current, 'error_rate', 4)} |",
        "",
        "## Cost",
        "",
        f"Currency: `{currency}`. Prices are read from `tests/benchmark/pricing.json`.",
        "",
        "| Component | Baseline → Current Avg / Query | Current Run Total |",
        "|---|---:|---:|",
    ])
    for key, label in [
        ("embedding", "Embedding"),
        ("query_rewrite", "Query Rewrite"),
        ("generation", "Generation"),
        ("verification", "Verification"),
        ("reranker", "Reranker"),
        ("total", "Total"),
    ]:
        component = current["cost"][key]
        avg = component.get("avg")
        total = component.get("sum")
        average_comparison = pair(
            baseline,
            current,
            f"cost.{key}.avg",
            6,
            f" {currency}",
        )
        lines.append(
            f"| {label} | "
            f"{average_comparison if avg is not None else 'N/A'} | "
            f"{currency_symbol + fmt(total, 6) if total is not None else 'N/A'} |"
        )

    # Per-dimension breakdown
    by_dim = current.get("by_dimension") or {}
    if by_dim:
        lines.extend([
            "",
            "## Per-Dimension Breakdown",
            "",
            "| Dimension | Count | Success | Status Acc | Verif. Pass | Halluc. Rate | Recall@10 | Avg Latency |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for dim, metrics in sorted(by_dim.items()):
            lines.append(
                f"| {dim} | {metrics.get('count', 0)} | "
                f"{fmt(metrics.get('success_rate'), 4)} | "
                f"{fmt(metrics.get('answer_status_accuracy'), 4)} | "
                f"{fmt(metrics.get('verification_pass_rate'), 4)} | "
                f"{fmt(metrics.get('hallucination_rate'), 4)} | "
                f"{fmt(metrics.get('recall_at_10'), 4)} | "
                f"{fmt(metrics.get('avg_latency_ms'), 0)} ms |"
            )

    # Per-difficulty breakdown
    by_diff = current.get("by_difficulty") or {}
    if by_diff:
        lines.extend([
            "",
            "## Per-Difficulty Breakdown",
            "",
            "| Difficulty | Count | Success | Status Acc | Halluc. Rate | Avg Latency |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for diff, metrics in sorted(by_diff.items()):
            lines.append(
                f"| {diff} | {metrics.get('count', 0)} | "
                f"{fmt(metrics.get('success_rate'), 4)} | "
                f"{fmt(metrics.get('answer_status_accuracy'), 4)} | "
                f"{fmt(metrics.get('hallucination_rate'), 4)} | "
                f"{fmt(metrics.get('avg_latency_ms'), 0)} ms |"
            )

    lines.extend(["", "## Regression Gate", ""])
    failures = report.get("threshold_failures") or []
    if failures:
        lines.append("| Metric | Baseline | Current | Rule |")
        lines.append("|---|---:|---:|---|")
        for failure in failures:
            rule = failure["rule"]
            lines.append(
                f"| `{failure['path']}` | {fmt(failure['baseline'])} | "
                f"{fmt(failure['current'])} | {rule.get('direction')} "
                f"{rule.get('mode', 'absolute')} tolerance={rule.get('tolerance')} |"
            )
    elif baseline is None:
        reason = report.get("baseline_compatibility", {}).get("reason")
        if reason:
            lines.append(f"Baseline comparison was skipped: {reason}.")
        else:
            lines.append("No baseline was found. Review this run, then promote it explicitly.")
    else:
        lines.append("No configured regression threshold was violated.")

    failed_records = [
        record
        for record in report.get("records", [])
        if not record.get("success")
        or not record.get("answer_status_correct")
        or record.get("hallucination")
        or (
            record.get("expected_answer_status") in {
                "answerable", "partially_answerable"
            }
            and record.get("retrieval_metrics", {}).get("recall_at_10") == 0
        )
    ]
    lines.extend(["", "## Samples Requiring Review", ""])
    if not failed_records:
        lines.append("None.")
    else:
        lines.append("| ID | Question | Issue |")
        lines.append("|---|---|---|")
        for record in failed_records[:30]:
            issues = []
            if not record.get("success"):
                issues.append(record.get("error_type") or "technical error")
            if not record.get("answer_status_correct"):
                issues.append("answer status mismatch")
            if record.get("hallucination"):
                issues.append(
                    f"unsupported={record.get('unsupported_claims', 0)}"
                )
            if record.get("retrieval_metrics", {}).get("recall_at_10") == 0:
                issues.append("Recall@10=0")
            question = str(record.get("question") or "").replace("|", "\\|")
            lines.append(
                f"| `{record.get('id')}` | {question} | {', '.join(issues)} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="分片式 Golden Dataset RAG 回归基准")
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--questions",
        type=Path,
        help="显式题集路径；未指定时默认运行 blind_test 分片",
    )
    dataset_group.add_argument(
        "--split",
        choices=["train_dev", "calibration", "blind_test", "all", "legacy"],
        help="命名题集分片（默认：blind_test）",
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--version", default=os.getenv("BENCHMARK_VERSION", "working-tree")
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--dimension",
        action="append",
        help="只运行指定 dimension；可重复，仅用于定向诊断",
    )
    parser.add_argument(
        "--id",
        action="append",
        dest="question_ids",
        help="只运行指定题目 ID；可重复，仅用于定向诊断",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="仅用于诊断；默认必须先通过 5 类冒烟门禁",
    )
    parser.add_argument(
        "--smoke-cases", type=Path, default=DEFAULT_SMOKE_CASES
    )
    parser.add_argument(
        "--smoke-output-dir",
        type=Path,
        default=DEFAULT_SMOKE_OUTPUT_DIR,
    )
    parser.add_argument(
        "--smoke-concurrency",
        type=int,
        default=1,
        help="功能冒烟默认串行，容量与尾延迟由压力测试单独评估",
    )
    parser.add_argument("--promote-baseline", action="store_true")
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()

    split_name = args.split or (None if args.questions else DEFAULT_SPLIT)
    if split_name:
        try:
            all_questions = load_split(split_name)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(str(exc)) from exc
        dataset_path = (
            SPLIT_PATHS.get(split_name)
            if split_name in SPLIT_PATHS
            else LEGACY_QUESTIONS if split_name == "legacy" else None
        )
        questions_bytes = json.dumps(
            all_questions,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    else:
        dataset_path = args.questions
        if dataset_path is None:
            raise SystemExit("未指定 Golden Dataset")
        questions_bytes = dataset_path.read_bytes()
        all_questions = json.loads(questions_bytes.decode("utf-8"))

    if not isinstance(all_questions, list):
        raise SystemExit("Golden Dataset 顶层必须是数组")

    if (
        split_name != "blind_test" or args.limit or args.dimension
        or args.question_ids
    ) and (
        args.promote_baseline or args.fail_on_regression
    ):
        parser.error(
            "--promote-baseline/--fail-on-regression 只能用于未经筛选的 "
            "blind_test 分片"
        )
    questions = all_questions
    if args.dimension:
        dimensions = set(args.dimension)
        questions = [
            item for item in questions
            if item.get("dimension") in dimensions
        ]
    if args.question_ids:
        question_ids = set(args.question_ids)
        questions = [
            item for item in questions if item.get("id") in question_ids
        ]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise SystemExit("Golden Dataset 不能为空")
    if args.concurrency < 1 or args.timeout <= 0:
        parser.error("concurrency 和 timeout 必须大于0")
    if args.smoke_concurrency < 1:
        parser.error("smoke-concurrency 必须大于0")

    smoke_report = None
    if not args.skip_smoke:
        print("[SMOKE] running 5-case pre-regression gate")
        smoke_report = run_smoke_suite(
            endpoint=args.endpoint,
            cases_path=args.smoke_cases,
            timeout=args.timeout,
            concurrency=args.smoke_concurrency,
        )
        write_smoke_report(smoke_report, args.smoke_output_dir)
        print(
            f"[SMOKE] {smoke_report['passed']}/"
            f"{smoke_report['cases']} passed"
        )
        if not smoke_report["success"]:
            print(
                "[SMOKE] failed; 100-question regression was not started. "
                f"See {args.smoke_output_dir / 'smoke_report.md'}"
            )
            raise SystemExit(1)

    pricing = load_pricing(args.pricing)
    metadata_before = endpoint_metadata(args.endpoint, args.timeout)
    records, wall_seconds = run_benchmark(
        questions,
        args.endpoint,
        args.concurrency,
        args.timeout,
        args.top_k,
        args.warmup,
        pricing,
    )
    metadata_after = endpoint_metadata(args.endpoint, args.timeout)
    summary = summarize(records, wall_seconds, pricing)

    import hashlib

    effective_dataset_sha = hashlib.sha256(
        json.dumps(
            questions, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    baseline_report = load_json(args.baseline, required=False)
    baseline_compatibility = {"compatible": False, "reason": None}
    if isinstance(baseline_report, Mapping):
        baseline_dataset = baseline_report.get("dataset") or {}
        same_sha = baseline_dataset.get("effective_sha256") == effective_dataset_sha
        same_count = baseline_dataset.get("questions") == len(questions)
        if same_sha and same_count:
            baseline_compatibility["compatible"] = True
        else:
            baseline_compatibility["reason"] = (
                "Golden Dataset hash or evaluated question count differs"
            )
    baseline_summary = (
        baseline_report.get("summary")
        if isinstance(baseline_report, Mapping)
        and baseline_compatibility["compatible"]
        and isinstance(baseline_report.get("summary"), Mapping)
        else None
    )
    comparison = compare(baseline_summary, summary)
    thresholds = load_json(args.thresholds, required=False) or {}
    threshold_failures = evaluate_thresholds(comparison, thresholds)
    decision = (
        (
            "INCOMPARABLE"
            if isinstance(baseline_report, Mapping)
            and not baseline_compatibility["compatible"]
            else "NO_BASELINE"
        )
        if baseline_summary is None
        else ("REGRESSION" if threshold_failures else "PASS")
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "version": args.version,
        "baseline_version": baseline_report.get("version")
        if isinstance(baseline_report, Mapping)
        else None,
        "baseline_compatibility": baseline_compatibility,
        "decision": decision,
        "dataset": {
            "path": str(dataset_path) if dataset_path else "combined:splits/all",
            "split": split_name,
            "sha256": hashlib.sha256(questions_bytes).hexdigest(),
            "effective_sha256": effective_dataset_sha,
            "questions": len(questions),
            "full_dataset_questions": len(all_questions),
        },
        "config": {
            "endpoint": args.endpoint,
            "concurrency": args.concurrency,
            "timeout_seconds": args.timeout,
            "top_k": args.top_k,
            "warmup": args.warmup,
            "dimensions": args.dimension or [],
            "question_ids": args.question_ids or [],
            "smoke_concurrency": args.smoke_concurrency,
            "limit": args.limit,
            "split": split_name,
            "pricing": str(args.pricing),
            "thresholds": str(args.thresholds),
            "percentile_method": "nearest-rank",
            "smoke_gate_enabled": not args.skip_smoke,
            "smoke_cases": str(args.smoke_cases),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "service_before": metadata_before,
            "service_after": metadata_after,
        },
        "smoke_gate": smoke_report,
        "summary": summary,
        "baseline_summary": baseline_summary,
        "comparison": comparison,
        "threshold_failures": threshold_failures,
        "records": records,
    }
    report["provenance"] = build_provenance(
        report["dataset"], metadata_before
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_output = args.output_dir / "regression_report.json"
    markdown_output = args.output_dir / "regression_report.md"
    json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(markdown_report(report), encoding="utf-8")

    print(f"[{decision}] JSON: {json_output}")
    print(f"[{decision}] Markdown: {markdown_output}")
    if args.promote_baseline:
        write_baseline(report, args.baseline)
        print(f"[BASELINE] promoted: {args.baseline}")
    if args.fail_on_regression and threshold_failures:
        raise SystemExit(2)
    if args.fail_on_regression and decision in {"NO_BASELINE", "INCOMPARABLE"}:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
