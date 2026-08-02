"""RAG SSE 性能基准与优化前后对比工具。

该脚本只调用已经启动的服务，不负责启动后端。默认顺序执行 100 次，适合
观察单请求延迟；通过 ``--concurrency`` 可以额外进行有限并发测试。

示例：

    .venv/bin/python tests/performance_benchmark.py \
      --qa-file tests/qa_samples/all_qa.json \
      --requests 100 \
      --endpoint baseline=http://127.0.0.1:8001 \
      --endpoint optimized=http://127.0.0.1:8000 \
      --output tests/results/perf_compare.json

    .venv/bin/python tests/performance_benchmark.py \
      --qa-file tests/qa_samples/all_qa.json \
      --requests 100 \
      --endpoint optimized=http://127.0.0.1:8000 \
      --baseline-report tests/results/perf_baseline.json \
      --output tests/results/perf_optimized.json

TTFT 口径：

* generation_ttft_ms：服务端生成模型产生首 Token 的时间；
* verification_ttft_ms：服务端 Citation Verification 完成时间；
* user_visible_ttft_ms：客户端实际收到首个非空答案 Token 的时间；
* client_done_latency_ms：客户端实际收到 SSE ``done`` 的端到端时间。

严格引用核验会缓存生成模型输出，因此 generation TTFT 只能来自服务端 Trace，
客户端首 Token 对应的是核验后的 user-visible TTFT。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import statistics
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import requests


SCRIPT_VERSION = "1.0"
DEFAULT_WARMUP_QUERY = "请简要说明知识库可以回答哪些企业制度问题。"

GENERATION_TTFT_KEYS = (
    "generation_ttft_ms",
    "generation_first_token_at_ms",
    "model_ttft_ms",
    "llm_ttft_ms",
    "generation_first_token_ms",
)
VERIFICATION_TTFT_KEYS = (
    "verification_ttft_ms",
    "verified_ttft_ms",
    "citation_verification_completed_ms",
    "verification_completed_ms",
)
USER_VISIBLE_TTFT_KEYS = (
    "user_visible_ttft_ms",
    "verified_answer_ttft_ms",
    "actual_ttft_ms",
    "ttft_ms",  # 兼容旧版 Trace
)
SERVER_DONE_KEYS = (
    "server_done_emit_ms",
    "sse_total_latency_ms",
    "sse_complete_ms",
)
SERVER_TOTAL_KEYS = (
    "total_latency_ms",
    "api_total_latency_ms",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_number(value: Any) -> Optional[float]:
    """返回有限数值；布尔值不作为数值处理。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def first_number(sources: Iterable[Mapping[str, Any]], keys: Iterable[str]) -> Optional[float]:
    for source in sources:
        for key in keys:
            number = as_number(source.get(key))
            if number is not None:
                return number
    return None


def percentile(values: list[float], proportion: float) -> Optional[float]:
    """使用 nearest-rank 口径计算分位数，与现有实验脚本保持一致。"""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(proportion * len(ordered)) - 1))
    return ordered[index]


def metric_stats(values: Iterable[Any]) -> dict[str, Any]:
    clean = [number for value in values if (number := as_number(value)) is not None]
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
        "avg": round(statistics.fmean(clean), 2),
        "p50": round(percentile(clean, 0.50) or 0, 2),
        "p95": round(percentile(clean, 0.95) or 0, 2),
        "p99": round(percentile(clean, 0.99) or 0, 2),
        "min": round(min(clean), 2),
        "max": round(max(clean), 2),
    }


def token_stats(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [record.get("token_usage", {}).get(key) for record in records]
    stats = metric_stats(values)
    clean = [number for value in values if (number := as_number(value)) is not None]
    stats["sum"] = round(sum(clean), 2) if clean else None
    return stats


def llm_token_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        (record.get("token_usage", {}).get("prompt_tokens") or 0)
        + (record.get("token_usage", {}).get("completion_tokens") or 0)
        for record in records
    ]
    stats = metric_stats(values)
    stats["sum"] = round(sum(values), 2) if values else None
    return stats


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "endpoint"


def load_qa_schedule(path: Path, request_count: int, seed: int) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("QA 文件必须是非空 JSON 数组")

    qa_items: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or not str(item.get("query", "")).strip():
            raise ValueError(f"QA 文件第 {index + 1} 项缺少非空 query")
        qa_items.append({
            "qa_index": index,
            "query": str(item["query"]).strip(),
            "dimension": item.get("dimension"),
            "difficulty": item.get("difficulty"),
        })

    rng = random.Random(seed)
    schedule: list[dict[str, Any]] = []
    while len(schedule) < request_count:
        epoch = list(qa_items)
        rng.shuffle(epoch)
        schedule.extend(epoch)
    return schedule[:request_count], hashlib.sha256(raw).hexdigest()


def find_span(trace: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for span in trace.get("spans", []) or []:
        if isinstance(span, dict) and span.get("name") == name:
            return span
    return {}


def span_end_ms(span: Mapping[str, Any]) -> Optional[float]:
    start = as_number(span.get("start_offset_ms"))
    duration = as_number(span.get("duration_ms"))
    if start is None or duration is None:
        return None
    return start + duration


def generation_was_skipped(
    trace: Mapping[str, Any],
    answer_status: Optional[str],
) -> bool:
    """判断本次请求是否按设计跳过了生成模型，而非漏记 Generation TTFT。"""
    if answer_status == "not_found":
        return True
    generation_span = find_span(trace, "generation")
    attributes = generation_span.get("attributes", {})
    return isinstance(attributes, Mapping) and attributes.get("skipped") is True


def cache_state(attributes: Mapping[str, Any]) -> str:
    """把不同 Trace 版本中的缓存字段归一为 hit/miss/partial/skipped/unknown。"""
    if attributes.get("skipped") is True:
        return "skipped"

    hits = as_number(attributes.get("cache_hits"))
    misses = as_number(attributes.get("cache_misses"))
    if hits is not None or misses is not None:
        hits = hits or 0
        misses = misses or 0
        if hits > 0 and misses > 0:
            return "partial"
        if hits > 0:
            return "hit"
        if misses > 0:
            return "miss"

    for key in ("cache_hit", "cache_hit_all", "all_cache_hit"):
        value = attributes.get(key)
        if isinstance(value, bool):
            return "hit" if value else "miss"
    return "unknown"


def extract_cache(trace: Mapping[str, Any], span_name: str) -> dict[str, Any]:
    span = find_span(trace, span_name)
    attributes = span.get("attributes", {}) if isinstance(span, dict) else {}
    if not isinstance(attributes, dict):
        attributes = {}

    direct_cache = trace.get("cache", {})
    if isinstance(direct_cache, dict):
        component = direct_cache.get(span_name, {})
        if isinstance(component, dict):
            attributes = {**component, **attributes}

    # 当前 RAGTrace 使用顶层 cache_hits/cache_stats；同时兼容 query_embedding 命名。
    aliases = (span_name, "query_embedding") if span_name == "embedding" else (span_name,)
    direct_hits = trace.get("cache_hits", {})
    if isinstance(direct_hits, dict):
        for alias in aliases:
            value = direct_hits.get(alias)
            if isinstance(value, bool) and "cache_hit" not in attributes:
                attributes["cache_hit"] = value
                break
    direct_stats = trace.get("cache_stats", {})
    if isinstance(direct_stats, dict):
        for alias in aliases:
            for output_key, suffix in (("cache_hits", "hits"), ("cache_misses", "misses")):
                value = direct_stats.get(f"{alias}_{suffix}")
                if as_number(value) is not None and output_key not in attributes:
                    attributes[output_key] = value

    return {
        "state": cache_state(attributes),
        "hits": as_number(attributes.get("cache_hits")),
        "misses": as_number(attributes.get("cache_misses")),
        "skipped": attributes.get("skipped") is True,
        "reason": attributes.get("reason") or attributes.get("skip_reason"),
    }


def request_metadata(base_url: str, timeout: float) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    paths = {
        "health": "/health",
        "models_status": "/api/v1/models/status",
        "stats": "/api/v1/stats",
    }
    with requests.Session() as session:
        for key, path in paths.items():
            try:
                response = session.get(
                    f"{base_url.rstrip('/')}{path}",
                    timeout=(min(10.0, timeout), min(15.0, timeout)),
                )
                response.raise_for_status()
                metadata[key] = response.json()
            except Exception as exc:  # 元数据失败不应阻断性能测试
                metadata[key] = {"error": str(exc)}
    return metadata


class EndpointRunner:
    def __init__(self, label: str, base_url: str, timeout: float, top_k: int):
        self.label = label
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.top_k = top_k
        self._local = threading.local()
        self.run_id = uuid.uuid4().hex[:10]

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({
                "Accept": "text/event-stream",
                "User-Agent": f"rag-performance-benchmark/{SCRIPT_VERSION}",
            })
            self._local.session = session
        return session

    def run_one(self, item: Mapping[str, Any], index: int, *, warmup: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        request_id = f"perf-{safe_label(self.label)}-{self.run_id}-{index}"
        record: dict[str, Any] = {
            "index": index,
            "query": item["query"],
            "qa_index": item.get("qa_index"),
            "dimension": item.get("dimension"),
            "difficulty": item.get("difficulty"),
            "warmup": warmup,
            "request_id": request_id,
            "success": False,
            "observability_complete": False,
            "http_status": None,
            "event_order": [],
            "error_type": None,
            "error": None,
        }

        trace: dict[str, Any] = {}
        done_payload: dict[str, Any] = {}
        verification: dict[str, Any] = {}
        source_count = 0
        citation_count = 0
        token_event_count = 0
        answer_chars = 0
        client_ttft_ms: Optional[float] = None
        client_done_ms: Optional[float] = None
        done_received = False

        try:
            response = self._session().post(
                f"{self.base_url}/api/v1/chat/stream",
                json={
                    "query": item["query"],
                    # 性能样本不写入业务会话库，避免 50~100 次测试污染历史数据。
                    "session_id": None,
                    "top_k": self.top_k,
                    "enable_agent": False,
                },
                headers={"X-Request-ID": request_id},
                stream=True,
                timeout=(min(10.0, self.timeout), self.timeout),
            )
            with response:
                record["headers_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                record["http_status"] = response.status_code
                record["response_request_id"] = response.headers.get("X-Request-ID")
                if not response.ok:
                    record["error_type"] = "http_error"
                    record["error"] = response.text[:500]
                    return self._finish_record(record, started)

                # 64B 远小于 requests 默认 512B，可保持 TTFT 精度；逐字节读取会在
                # sources/citations 较大时显著放大客户端解析耗时，污染 SSE 总时延。
                for raw_line in response.iter_lines(chunk_size=64, decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if line.startswith(":") or not line.startswith("data:"):
                        continue
                    raw_data = line[5:].lstrip()
                    if not raw_data:
                        continue
                    try:
                        event = json.loads(raw_data)
                    except json.JSONDecodeError as exc:
                        record["error_type"] = "malformed_sse_json"
                        record["error"] = f"{exc}: {raw_data[:300]}"
                        break
                    if not isinstance(event, dict):
                        record["error_type"] = "invalid_sse_payload"
                        record["error"] = f"SSE data 不是 JSON object: {raw_data[:300]}"
                        break

                    event_type = str(event.get("type") or "unknown")
                    record["event_order"].append(event_type)
                    if event_type == "token":
                        content = str(event.get("content") or "")
                        if content:
                            token_event_count += 1
                            answer_chars += len(content)
                            if client_ttft_ms is None:
                                client_ttft_ms = (time.perf_counter() - started) * 1000
                    elif event_type == "sources":
                        sources = event.get("sources") or []
                        source_count = len(sources) if isinstance(sources, list) else 0
                        record["answer_status"] = event.get("answer_status")
                    elif event_type == "citations":
                        citations = event.get("citations") or []
                        citation_count = len(citations) if isinstance(citations, list) else 0
                    elif event_type == "citation_verification":
                        value = event.get("verification")
                        verification = value if isinstance(value, dict) else {}
                    elif event_type == "rag_trace":
                        value = event.get("trace")
                        trace = value if isinstance(value, dict) else {}
                    elif event_type in {"stream_metrics", "metrics"}:
                        metrics = event.get("metrics")
                        if isinstance(metrics, dict):
                            done_payload.setdefault("metrics", {}).update(metrics)
                    elif event_type == "error":
                        record["error_type"] = "sse_error"
                        record["error"] = str(event.get("message") or "unknown SSE error")
                        break
                    elif event_type == "done":
                        done_received = True
                        client_done_ms = (time.perf_counter() - started) * 1000
                        done_payload.update(event)
                        if not trace and isinstance(event.get("trace"), dict):
                            trace = event["trace"]
                        break
        except requests.Timeout as exc:
            record["error_type"] = "timeout"
            record["error"] = str(exc)
        except requests.ConnectionError as exc:
            record["error_type"] = "connection_error"
            record["error"] = str(exc)
        except requests.RequestException as exc:
            record["error_type"] = "request_error"
            record["error"] = str(exc)
        except Exception as exc:  # 保留单次样本，避免整批基准中断
            record["error_type"] = "unexpected_error"
            record["error"] = f"{type(exc).__name__}: {exc}"

        if record["error_type"] is None and not done_received:
            record["error_type"] = "missing_done"
            record["error"] = "SSE 流结束但未收到 done 事件"

        record["success"] = record["error_type"] is None and done_received
        record["client_user_visible_ttft_ms"] = round(client_ttft_ms, 2) if client_ttft_ms is not None else None
        record["user_visible_ttft_ms"] = record["client_user_visible_ttft_ms"]
        record["client_done_latency_ms"] = round(client_done_ms, 2) if client_done_ms is not None else None
        record["token_event_count"] = token_event_count
        record["answer_chars"] = answer_chars
        record["source_count"] = source_count
        record["citation_count"] = citation_count
        record["verification_status"] = verification.get("status")

        metrics = done_payload.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        metric_sources: list[Mapping[str, Any]] = [metrics, done_payload, trace]

        generation_ttft = first_number(metric_sources, GENERATION_TTFT_KEYS)
        verification_ttft = first_number(metric_sources, VERIFICATION_TTFT_KEYS)
        server_user_visible_ttft = first_number(metric_sources, USER_VISIBLE_TTFT_KEYS)
        server_done_emit = first_number(metric_sources, SERVER_DONE_KEYS)
        server_total = first_number(metric_sources, SERVER_TOTAL_KEYS)

        if verification_ttft is None:
            verification_ttft = span_end_ms(find_span(trace, "citation_verification"))

        record["generation_ttft_ms"] = generation_ttft
        record["verification_ttft_ms"] = verification_ttft
        record["server_user_visible_ttft_ms"] = server_user_visible_ttft
        record["server_done_emit_ms"] = server_done_emit
        record["server_total_latency_ms"] = server_total

        usage = trace.get("token_usage", {})
        if not isinstance(usage, dict):
            usage = metrics.get("token_usage", {}) if isinstance(metrics.get("token_usage"), dict) else {}
        record["token_usage"] = {
            "prompt_tokens": as_number(usage.get("prompt_tokens")),
            "completion_tokens": as_number(usage.get("completion_tokens")),
            "reranker_tokens": as_number(usage.get("reranker_tokens")),
            "total_tokens": as_number(usage.get("total_tokens")),
        }
        record["query_rewrite_cache"] = extract_cache(trace, "query_rewrite")
        record["embedding_cache"] = extract_cache(trace, "embedding")
        record["knowledge_base_version"] = trace.get("knowledge_base_version")
        record["query_strategy"] = trace.get("query_strategy")
        record["multiquery_triggered"] = trace.get("multiquery_triggered")
        record["generation_skipped"] = generation_was_skipped(
            trace,
            record.get("answer_status"),
        )

        observability_errors = []
        if record["success"]:
            if not trace:
                observability_errors.append("missing_trace")
            if client_ttft_ms is None:
                observability_errors.append("missing_token")
            if generation_ttft is None and not record["generation_skipped"]:
                observability_errors.append("missing_generation_ttft")
            if verification_ttft is None:
                observability_errors.append("missing_verification_ttft")
            if record["token_usage"]["total_tokens"] is None:
                observability_errors.append("missing_token_usage")
        record["observability_errors"] = observability_errors
        record["observability_complete"] = record["success"] and not observability_errors
        return self._finish_record(record, started)

    @staticmethod
    def _finish_record(record: dict[str, Any], started: float) -> dict[str, Any]:
        record["attempt_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return record


def cache_group_summary(records: list[dict[str, Any]], component: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for state in ("hit", "miss", "partial", "skipped", "unknown"):
        group = [record for record in records if record.get(component, {}).get("state") == state]
        result[state] = {
            "samples": len(group),
            "client_done_latency_ms": metric_stats(r.get("client_done_latency_ms") for r in group),
            "user_visible_ttft_ms": metric_stats(r.get("user_visible_ttft_ms") for r in group),
            "total_tokens": metric_stats(r.get("token_usage", {}).get("total_tokens") for r in group),
        }
    return result


def summarize(records: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    successful = [record for record in records if record.get("success")]
    failed = [record for record in records if not record.get("success")]
    total = len(records)
    verified_count = sum(r.get("verification_status") == "verified" for r in successful)
    return {
        "attempted_requests": total,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "success_rate": round(len(successful) / total, 6) if total else 0,
        "error_rate": round(len(failed) / total, 6) if total else 0,
        "observability_complete_rate": round(
            sum(bool(r.get("observability_complete")) for r in records) / total, 6
        ) if total else 0,
        "verification_verified_rate": round(
            verified_count / len(successful), 6
        ) if successful else 0,
        "error_types": dict(Counter(r.get("error_type") or "none" for r in records)),
        "observability_errors": dict(Counter(
            error for record in records for error in record.get("observability_errors", [])
        )),
        "wall_time_seconds": round(wall_seconds, 6),
        "attempted_rps": round(total / wall_seconds, 4) if wall_seconds > 0 else None,
        "successful_rps": round(len(successful) / wall_seconds, 4) if wall_seconds > 0 else None,
        "latency_ms": {
            "attempt": metric_stats(r.get("attempt_latency_ms") for r in records),
            "client_done": metric_stats(r.get("client_done_latency_ms") for r in successful),
            "server_done_emit": metric_stats(r.get("server_done_emit_ms") for r in successful),
            "server_total": metric_stats(r.get("server_total_latency_ms") for r in successful),
        },
        "ttft_ms": {
            "generation": metric_stats(r.get("generation_ttft_ms") for r in successful),
            "verification": metric_stats(r.get("verification_ttft_ms") for r in successful),
            "user_visible": metric_stats(r.get("user_visible_ttft_ms") for r in successful),
            "server_user_visible": metric_stats(r.get("server_user_visible_ttft_ms") for r in successful),
        },
        "token_usage": {
            "prompt_tokens": token_stats(successful, "prompt_tokens"),
            "completion_tokens": token_stats(successful, "completion_tokens"),
            "llm_tokens": llm_token_stats(successful),
            "reranker_tokens": token_stats(successful, "reranker_tokens"),
            "total_tokens": token_stats(successful, "total_tokens"),
        },
        "cache_groups": {
            "query_rewrite": cache_group_summary(successful, "query_rewrite_cache"),
            "embedding": cache_group_summary(successful, "embedding_cache"),
        },
        "answer_status": dict(Counter(r.get("answer_status") or "unknown" for r in successful)),
        "verification_status": dict(Counter(r.get("verification_status") or "unknown" for r in successful)),
        "query_strategy": dict(Counter(r.get("query_strategy") or "unknown" for r in successful)),
        "knowledge_base_versions": dict(Counter(
            r.get("knowledge_base_version") or "unknown" for r in successful
        )),
    }


def run_endpoint(
    label: str,
    url: str,
    schedule: list[dict[str, Any]],
    *,
    warmup_count: int,
    warmup_query: str,
    concurrency: int,
    interval_ms: int,
    timeout: float,
    top_k: int,
) -> dict[str, Any]:
    runner = EndpointRunner(label, url, timeout, top_k)
    metadata_before = request_metadata(url, timeout)

    warmups = []
    warmup_item = {"query": warmup_query, "qa_index": None, "dimension": "warmup", "difficulty": None}
    for index in range(warmup_count):
        warmups.append(runner.run_one(warmup_item, -(index + 1), warmup=True))

    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    if concurrency == 1:
        for index, item in enumerate(schedule, 1):
            records.append(runner.run_one(item, index))
            if interval_ms and index < len(schedule):
                time.sleep(interval_ms / 1000)
    else:
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=f"perf-{safe_label(label)}") as pool:
            futures = {}
            for index, item in enumerate(schedule, 1):
                future = pool.submit(runner.run_one, item, index)
                futures[future] = index
                if interval_ms and index < len(schedule):
                    time.sleep(interval_ms / 1000)
            for future in as_completed(futures):
                records.append(future.result())
        records.sort(key=lambda record: record["index"])
    wall_seconds = time.perf_counter() - started
    metadata_after = request_metadata(url, timeout)

    return {
        "label": label,
        "base_url": url.rstrip("/"),
        "service_metadata": {
            "before": metadata_before,
            "after": metadata_after,
        },
        "warmup": {
            "requests": warmup_count,
            "query": warmup_query,
            "successful": sum(bool(record.get("success")) for record in warmups),
            "records": warmups,
        },
        "summary": summarize(records, wall_seconds),
        "records": records,
    }


def get_path(mapping: Mapping[str, Any], path: str) -> Optional[float]:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return as_number(current)


COMPARISON_METRICS = {
    "client_done_avg_ms": ("latency_ms.client_done.avg", "lower"),
    "client_done_p95_ms": ("latency_ms.client_done.p95", "lower"),
    "client_done_p99_ms": ("latency_ms.client_done.p99", "lower"),
    "generation_ttft_avg_ms": ("ttft_ms.generation.avg", "lower"),
    "verification_ttft_avg_ms": ("ttft_ms.verification.avg", "lower"),
    "user_visible_ttft_avg_ms": ("ttft_ms.user_visible.avg", "lower"),
    "llm_tokens_avg": ("token_usage.llm_tokens.avg", "lower"),
    "reranker_tokens_avg": ("token_usage.reranker_tokens.avg", "lower"),
    "all_model_tokens_avg": ("token_usage.total_tokens.avg", "lower"),
    "error_rate": ("error_rate", "lower"),
    "verification_verified_rate": ("verification_verified_rate", "higher"),
    "successful_rps": ("successful_rps", "higher"),
}


def compare_summaries(
    baseline_label: str,
    baseline: Mapping[str, Any],
    current_label: str,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, (path, direction) in COMPARISON_METRICS.items():
        old = get_path(baseline, path)
        new = get_path(current, path)
        delta = new - old if old is not None and new is not None else None
        delta_pct = (delta / old * 100) if delta is not None and old not in (None, 0) else None
        improvement_pct = None
        if delta_pct is not None:
            improvement_pct = -delta_pct if direction == "lower" else delta_pct
        metrics[name] = {
            "baseline": old,
            "current": new,
            "delta": round(delta, 4) if delta is not None else None,
            "delta_pct": round(delta_pct, 2) if delta_pct is not None else None,
            "improvement_pct": round(improvement_pct, 2) if improvement_pct is not None else None,
            "better_when": direction,
        }
    return {
        "baseline_label": baseline_label,
        "current_label": current_label,
        "metrics": metrics,
    }


def load_baseline_experiments(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiments = payload.get("experiments") if isinstance(payload, dict) else None
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("baseline report 中没有非空 experiments 数组")
    normalized = []
    for item in experiments:
        if not isinstance(item, dict) or not isinstance(item.get("summary"), dict):
            continue
        copied = dict(item)
        records = item.get("records")
        if isinstance(records, list):
            wall_seconds = float(item["summary"].get("wall_time_seconds") or 0)
            copied["summary"] = summarize(records, wall_seconds)
        normalized.append(copied)
    return normalized


def build_comparisons(
    experiments: list[dict[str, Any]],
    baseline_experiments: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    comparisons = []
    if baseline_experiments:
        by_label = {item.get("label"): item for item in baseline_experiments}
        default_baseline = (
            by_label.get("baseline")
            or (baseline_experiments[0] if len(baseline_experiments) == 1 else None)
        )
        for current in experiments:
            baseline = by_label.get(current.get("label")) or default_baseline
            if baseline:
                comparisons.append(compare_summaries(
                    str(baseline.get("label", "baseline")),
                    baseline["summary"],
                    str(current.get("label", "current")),
                    current["summary"],
                ))
        return comparisons

    if len(experiments) > 1:
        baseline = experiments[0]
        for current in experiments[1:]:
            comparisons.append(compare_summaries(
                str(baseline["label"]), baseline["summary"],
                str(current["label"]), current["summary"],
            ))
    return comparisons


def format_value(value: Any, suffix: str = "") -> str:
    number = as_number(value)
    return "N/A" if number is None else f"{number:.2f}{suffix}"


def print_metric_row(name: str, stats: Mapping[str, Any]) -> None:
    print(
        f"  {name:26s} avg={format_value(stats.get('avg'), 'ms'):>11s} "
        f"P50={format_value(stats.get('p50'), 'ms'):>11s} "
        f"P95={format_value(stats.get('p95'), 'ms'):>11s} "
        f"P99={format_value(stats.get('p99'), 'ms'):>11s} "
        f"n={stats.get('count', 0)}"
    )


def print_report(report: Mapping[str, Any], output: Path) -> None:
    print(f"\nReport: {output}")
    print("Percentiles: nearest-rank（50 个样本时 P99 等同最大值，建议正式评测使用 100 次）")
    for experiment in report.get("experiments", []):
        summary = experiment["summary"]
        print(f"\n[{experiment['label']}] {experiment['base_url']}")
        print(
            f"  requests={summary['attempted_requests']} success={summary['success_rate']:.2%} "
            f"error={summary['error_rate']:.2%} RPS={format_value(summary.get('successful_rps'))}"
        )
        print_metric_row("Client done latency", summary["latency_ms"]["client_done"])
        print_metric_row("Generation TTFT", summary["ttft_ms"]["generation"])
        print_metric_row("Verification TTFT", summary["ttft_ms"]["verification"])
        print_metric_row("User-visible TTFT", summary["ttft_ms"]["user_visible"])
        tokens = summary["token_usage"]["total_tokens"]
        llm_tokens = summary["token_usage"]["llm_tokens"]
        reranker_tokens = summary["token_usage"]["reranker_tokens"]
        print(
            f"  {'Total tokens':26s} avg={format_value(tokens.get('avg')):>11s} "
            f"sum={format_value(tokens.get('sum')):>11s} n={tokens.get('count', 0)}"
        )
        print(
            f"  {'LLM / Reranker tokens':26s} avg={format_value(llm_tokens.get('avg'))} / "
            f"{format_value(reranker_tokens.get('avg'))}  "
            f"verified={summary.get('verification_verified_rate', 0):.2%}"
        )
        for component in ("query_rewrite", "embedding"):
            groups = summary["cache_groups"][component]
            counts = " ".join(f"{state}={groups[state]['samples']}" for state in groups)
            print(f"  {component + ' cache':26s} {counts}")

    for comparison in report.get("comparisons", []):
        print(f"\n[Compare] {comparison['baseline_label']} -> {comparison['current_label']}")
        for name, metric in comparison["metrics"].items():
            print(
                f"  {name:28s} {format_value(metric['baseline']):>10s} -> "
                f"{format_value(metric['current']):>10s} "
                f"improvement={format_value(metric['improvement_pct'], '%')}"
            )


def parse_endpoint(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("endpoint 格式必须为 LABEL=URL")
    label, url = value.split("=", 1)
    label = label.strip()
    url = url.strip().rstrip("/")
    if not label or not re.match(r"^https?://", url):
        raise argparse.ArgumentTypeError("endpoint 格式必须为非空 LABEL 和 http(s) URL")
    return label, url


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须大于等于 0")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="连续调用 RAG SSE 接口并输出可复现性能报告")
    parser.add_argument("--qa-file", required=True, type=Path, help="QA JSON 数组文件")
    parser.add_argument(
        "--endpoint", action="append", required=True, type=parse_endpoint, metavar="LABEL=URL",
        help="可重复传入；第一个 endpoint 默认作为同轮比较基线",
    )
    parser.add_argument("--requests", dest="request_count", type=positive_int, default=100)
    parser.add_argument("--warmup", type=non_negative_int, default=2)
    parser.add_argument("--warmup-query", default=DEFAULT_WARMUP_QUERY)
    parser.add_argument("--concurrency", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180.0, help="单次 SSE read timeout（秒）")
    parser.add_argument("--interval-ms", type=non_negative_int, default=0, help="请求提交间隔")
    parser.add_argument("--top-k", type=positive_int, default=15)
    parser.add_argument("--output", type=Path, default=Path("tests/results/performance_benchmark.json"))
    parser.add_argument("--baseline-report", type=Path, help="此前由本脚本生成的 JSON 报告")
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    if args.request_count < 50:
        print("[WARN] 少于 50 次仅适合冒烟验证，不适合作为正式 P95/P99 基准。")
    if args.request_count < 100:
        print("[WARN] 少于 100 次时 nearest-rank P99 接近或等于最大值。")

    schedule, qa_sha256 = load_qa_schedule(args.qa_file, args.request_count, args.seed)
    experiments = []
    for label, url in args.endpoint:
        print(
            f"[RUN] {label} {url} requests={args.request_count} "
            f"warmup={args.warmup} concurrency={args.concurrency}"
        )
        experiments.append(run_endpoint(
            label,
            url,
            schedule,
            warmup_count=args.warmup,
            warmup_query=args.warmup_query,
            concurrency=args.concurrency,
            interval_ms=args.interval_ms,
            timeout=args.timeout,
            top_k=args.top_k,
        ))

    baseline_experiments = None
    if args.baseline_report:
        baseline_experiments = load_baseline_experiments(args.baseline_report)

    report = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "generated_at": utc_now(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "benchmark_config": {
            "qa_file": str(args.qa_file),
            "qa_sha256": qa_sha256,
            "qa_unique_queries": len({item["query"] for item in schedule}),
            "requests": args.request_count,
            "warmup": args.warmup,
            "warmup_query": args.warmup_query,
            "concurrency": args.concurrency,
            "seed": args.seed,
            "timeout_seconds": args.timeout,
            "interval_ms": args.interval_ms,
            "top_k": args.top_k,
            "percentile_method": "nearest-rank",
            "baseline_report": str(args.baseline_report) if args.baseline_report else None,
        },
        "experiments": experiments,
        "comparisons": build_comparisons(experiments, baseline_experiments),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report, args.output)


if __name__ == "__main__":
    main()
