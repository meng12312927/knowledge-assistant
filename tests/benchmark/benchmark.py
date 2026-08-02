"""End-to-end RAG regression benchmark with Golden Dataset comparison.

Default usage:

    python tests/benchmark/benchmark.py --version v1.4

The command runs all 100 fixed questions, writes JSON + Markdown reports, and
compares them with ``tests/benchmark/baselines/latest.json`` when present.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import statistics
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.chunk_identity import stable_chunk_id
from tests.smoke.smoke_test import (
    DEFAULT_CASES as DEFAULT_SMOKE_CASES,
    DEFAULT_OUTPUT_DIR as DEFAULT_SMOKE_OUTPUT_DIR,
    run_smoke_suite,
    write_report as write_smoke_report,
)


DEFAULT_QUESTIONS = PROJECT_ROOT / "tests/benchmark/questions.json"
DEFAULT_BASELINE = PROJECT_ROOT / "tests/benchmark/baselines/latest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests/benchmark/results"
DEFAULT_THRESHOLDS = PROJECT_ROOT / "tests/benchmark/regression_thresholds.json"
DEFAULT_PRICING = PROJECT_ROOT / "tests/benchmark/pricing.json"
SCHEMA_VERSION = 1


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
            "dimension": item.get("dimension"),
            "difficulty": item.get("difficulty"),
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
        expected_status = record["expected_answer_status"]
        actual_status = record.get("answer_status")
        record["answer_status_correct"] = actual_status == expected_status
        record["hallucination"] = bool(
            record["unsupported_claims"]
            or (
                expected_status == "answerable"
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


def summarize(
    records: list[dict[str, Any]],
    wall_seconds: float,
    pricing: Mapping[str, Any],
) -> dict[str, Any]:
    successful = [record for record in records if record.get("success")]
    answerable = [
        record
        for record in successful
        if record.get("expected_answer_status") == "answerable"
    ]
    verified = [
        record
        for record in answerable
        if record.get("verification_status") == "verified"
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
            "answer_status_accuracy": round(
                sum(bool(record.get("answer_status_correct")) for record in successful)
                / len(successful),
                6,
            )
            if successful
            else None,
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
    "Verification Pass": ("generation.verification_pass_rate", "higher"),
    "Answer Status Accuracy": ("generation.answer_status_accuracy", "higher"),
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
        if not metric or metric.get("baseline") is None or metric.get("current") is None:
            continue
        old = float(metric["baseline"])
        new = float(metric["current"])
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
        f"- Dataset: `{report['dataset']['sha256'][:12]}` ({report['dataset']['questions']} questions)",
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
        "",
        "## Generation",
        "",
        "| Metric | Baseline → Current |",
        "|---|---:|",
        f"| Verification Pass | {pair(baseline, current, 'generation.verification_pass_rate', 4)} |",
        f"| Answer Status Accuracy | {pair(baseline, current, 'generation.answer_status_accuracy', 4)} |",
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
            record.get("expected_answer_status") == "answerable"
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
    parser = argparse.ArgumentParser(description="固定100题的RAG回归基准")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
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
    parser.add_argument("--promote-baseline", action="store_true")
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()

    questions_bytes = args.questions.read_bytes()
    all_questions = json.loads(questions_bytes.decode("utf-8"))
    questions = all_questions
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise SystemExit("Golden Dataset 不能为空")
    if args.concurrency < 1 or args.timeout <= 0:
        parser.error("concurrency 和 timeout 必须大于0")

    smoke_report = None
    if not args.skip_smoke:
        print("[SMOKE] running 5-case pre-regression gate")
        smoke_report = run_smoke_suite(
            endpoint=args.endpoint,
            cases_path=args.smoke_cases,
            timeout=args.timeout,
            concurrency=min(5, args.concurrency),
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
            "path": str(args.questions),
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
            "limit": args.limit,
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
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(json_output, args.baseline)
        print(f"[BASELINE] promoted: {args.baseline}")
    if args.fail_on_regression and threshold_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
