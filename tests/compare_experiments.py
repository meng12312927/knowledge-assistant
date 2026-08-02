"""对多个已启动的 RAG 配置执行相同 QA 集并输出可复现实验报告。"""

import argparse
import json
import math
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests


def percentile(values, percentile_value):
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def run_endpoint(label, base_url, qa_items, timeout):
    records = []
    for index, item in enumerate(qa_items, 1):
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/api/v1/chat",
                json={
                    "query": item["query"],
                    "session_id": f"experiment-{label}-{index}",
                    "top_k": 15,
                    "enable_agent": False,
                },
                timeout=timeout,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            response.raise_for_status()
            payload = response.json()
            records.append({
                "query": item["query"],
                "success": True,
                "latency_ms": latency_ms,
                "answer_status": payload.get("answer_status", "unknown"),
                "source_count": len(payload.get("sources", [])),
                "answer": payload.get("answer", ""),
            })
        except Exception as exc:
            records.append({
                "query": item["query"],
                "success": False,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "error": str(exc),
            })

    successful = [record for record in records if record["success"]]
    latencies = [record["latency_ms"] for record in successful]
    source_counts = [record["source_count"] for record in successful]
    return {
        "label": label,
        "base_url": base_url,
        "summary": {
            "samples": len(records),
            "success_rate": round(len(successful) / len(records), 4) if records else 0,
            "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0,
            "p95_latency_ms": percentile(latencies, 0.95),
            "avg_source_count": round(statistics.mean(source_counts), 2) if source_counts else 0,
            "answer_status": dict(Counter(r["answer_status"] for r in successful)),
        },
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-file", required=True)
    parser.add_argument(
        "--endpoint", action="append", required=True, metavar="LABEL=URL",
        help="可重复传入，例如 deepseek=http://localhost:8000",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", default="tests/results/experiment_comparison.json")
    args = parser.parse_args()

    qa_items = json.loads(Path(args.qa_file).read_text(encoding="utf-8"))
    if args.limit:
        qa_items = qa_items[:args.limit]

    endpoints = []
    for endpoint in args.endpoint:
        if "=" not in endpoint:
            parser.error(f"endpoint 格式错误: {endpoint}")
        endpoints.append(endpoint.split("=", 1))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qa_file": args.qa_file,
        "experiments": [run_endpoint(label, url, qa_items, args.timeout) for label, url in endpoints],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Report: {output}")
    for experiment in report["experiments"]:
        summary = experiment["summary"]
        print(
            f"{experiment['label']}: success={summary['success_rate']:.1%} "
            f"avg={summary['avg_latency_ms']}ms p95={summary['p95_latency_ms']}ms "
            f"sources={summary['avg_source_count']}"
        )


if __name__ == "__main__":
    main()
