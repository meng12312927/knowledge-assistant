"""Five-case production-path smoke gate executed before regression benchmark."""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = PROJECT_ROOT / "tests/smoke/cases.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests/smoke/results"


def _api_route(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    for span in trace.get("spans") or []:
        if span.get("name") == "api_total":
            return span.get("attributes") or {}
    return {}


def _observe(payload: Mapping[str, Any]) -> dict[str, Any]:
    trace = payload.get("trace") or {}
    route = _api_route(trace)
    citations = payload.get("citations") or []
    tools = payload.get("tool_results") or []
    first_tool = tools[0] if tools else {}
    verification = payload.get("citation_verification") or {}
    return {
        "answer_status": payload.get("answer_status"),
        "agent": route.get("agent"),
        "agent_reason": route.get("agent_route_reason"),
        "query_strategy": (
            trace.get("routing_probe_strategy")
            or trace.get("query_strategy")
        ),
        "multiquery_triggered": bool(
            trace.get("routing_probe_multiquery_triggered")
            or trace.get("multiquery_triggered")
        ),
        "retrieval_quality": trace.get("retrieval_quality"),
        "citation_count": len(citations),
        "citation_sources": sorted(
            {
                str(citation.get("source_file") or "")
                for citation in citations
                if citation.get("source_file")
            }
        ),
        "verification_status": verification.get("status"),
        "verification_message": verification.get("message"),
        "verification_failures": [
            {
                "claim": item.get("claim"),
                "verdict": item.get("verdict"),
                "reason": item.get("reason"),
            }
            for item in verification.get("items") or []
            if item.get("verdict") != "supported"
        ],
        "source_count": len(payload.get("sources") or []),
        "tool_name": first_tool.get("tool_name"),
        "tool_success": first_tool.get("success"),
        "tool_output": first_tool.get("output"),
        "answer_preview": str(payload.get("answer") or "")[:200],
    }


def _check(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    for key in (
        "agent",
        "agent_reason",
        "query_strategy",
        "multiquery_triggered",
        "retrieval_quality",
        "verification_status",
        "tool_name",
        "tool_success",
    ):
        if key in expected and observed.get(key) != expected[key]:
            failures.append(
                f"{key}: expected={expected[key]!r}, "
                f"actual={observed.get(key)!r}"
            )

    if "answer_status_in" in expected:
        allowed = expected["answer_status_in"]
        if observed.get("answer_status") not in allowed:
            failures.append(
                f"answer_status: expected one of {allowed!r}, "
                f"actual={observed.get('answer_status')!r}"
            )
    if "minimum_citations" in expected:
        if int(observed.get("citation_count") or 0) < int(
            expected["minimum_citations"]
        ):
            failures.append(
                f"citation_count: expected >= {expected['minimum_citations']}, "
                f"actual={observed.get('citation_count')}"
            )
    if "citation_source_any" in expected:
        actual = set(observed.get("citation_sources") or [])
        wanted = set(expected["citation_source_any"])
        if not actual.intersection(wanted):
            failures.append(
                f"citation_sources: expected any of {sorted(wanted)!r}, "
                f"actual={sorted(actual)!r}"
            )
    if expected.get("empty_sources") and observed.get("source_count") != 0:
        failures.append(
            f"source_count: expected=0, actual={observed.get('source_count')}"
        )
    if expected.get("empty_citations") and observed.get("citation_count") != 0:
        failures.append(
            f"citation_count: expected=0, actual={observed.get('citation_count')}"
        )
    if "tool_output" in expected:
        actual = observed.get("tool_output")
        wanted = expected["tool_output"]
        if not (
            isinstance(actual, (int, float))
            and math.isclose(float(actual), float(wanted), rel_tol=1e-9)
        ):
            failures.append(
                f"tool_output: expected={wanted!r}, actual={actual!r}"
            )
    return failures


def _run_one(
    endpoint: str,
    case: Mapping[str, Any],
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = {
        "id": case["id"],
        "type": case["type"],
        "question": case["question"],
        "passed": False,
        "failures": [],
    }
    try:
        response = requests.post(
            f"{endpoint.rstrip('/')}/api/v1/chat",
            json={"query": case["question"], "top_k": 6},
            timeout=(10, timeout),
        )
        result["http_status"] = response.status_code
        response.raise_for_status()
        observed = _observe(response.json())
        failures = _check(case["expected"], observed)
        result.update(
            {
                "observed": observed,
                "failures": failures,
                "passed": not failures,
            }
        )
    except Exception as exc:
        result["failures"] = [f"{type(exc).__name__}: {exc}"]
    result["latency_ms"] = round(
        (time.perf_counter() - started) * 1000, 3
    )
    return result


def run_smoke_suite(
    *,
    endpoint: str,
    cases_path: Path = DEFAULT_CASES,
    timeout: float = 180.0,
    concurrency: int = 5,
) -> dict[str, Any]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(_run_one, endpoint, case, timeout): case["id"]
            for case in cases
        }
        results = [future.result() for future in as_completed(futures)]
    order = {case["id"]: index for index, case in enumerate(cases)}
    results.sort(key=lambda item: order[item["id"]])
    passed = sum(bool(item["passed"]) for item in results)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "cases": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "success": passed == len(cases),
        "results": results,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# RAG Smoke Gate",
        "",
        f"- Result: **{'PASS' if report['success'] else 'FAIL'}**",
        f"- Passed: `{report['passed']}/{report['cases']}`",
        f"- Endpoint: `{report['endpoint']}`",
        "",
        "| Type | Question | Result | Latency |",
        "|---|---|---:|---:|",
    ]
    for result in report["results"]:
        question = str(result["question"]).replace("|", "\\|")
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(
            f"| {result['type']} | {question} | {status} | "
            f"{result['latency_ms']:.2f} ms |"
        )
        for failure in result["failures"]:
            lines.append(f"| ↳ | `{failure}` |  |  |")
    lines.append("")
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "smoke_report.md").write_text(
        markdown_report(report),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="5 类 RAG 回归前置冒烟")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    report = run_smoke_suite(
        endpoint=args.endpoint,
        cases_path=args.cases,
        timeout=args.timeout,
        concurrency=args.concurrency,
    )
    write_report(report, args.output_dir)
    for result in report["results"]:
        print(
            f"[{'PASS' if result['passed'] else 'FAIL'}] "
            f"{result['type']}: {result['question']}"
        )
        for failure in result["failures"]:
            print(f"  - {failure}")
    print(
        f"[{'PASS' if report['success'] else 'FAIL'}] "
        f"{report['passed']}/{report['cases']} smoke cases"
    )
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
