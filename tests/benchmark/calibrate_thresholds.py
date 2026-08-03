"""Threshold calibration using offline trace analysis.

Reads benchmark results from calibration.json and analyzes retrieval/reranker
trace data to recommend optimal thresholds for:
  - SIMPLE_QUERY_MIN_RRF_SCORE (MultiQuery trigger)
  - RERANKER_NOT_FOUND_THRESHOLD (OOD rejection)
  - ANSWER_STATUS_THRESHOLD_HIGH / ANSWER_STATUS_THRESHOLD_LOW

Usage:
    # First, run benchmark and save results:
    python tests/benchmark/benchmark.py --questions tests/benchmark/splits/calibration.json --version calibration-v1

    # Then analyze:
    python tests/benchmark/calibrate_thresholds.py --report tests/benchmark/results/regression_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    n = float(value)
    return n if math.isfinite(n) else None


def analyze_rrf_threshold(
    records: List[Dict[str, Any]],
    sweep_values: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Analyze optimal SIMPLE_QUERY_MIN_RRF_SCORE.

    ``calibration.json`` uses explicit ``should-trigger-multiquery`` and
    ``should-not-trigger-multiquery`` tags as routing ground truth. Boundary
    samples without either label are excluded from this particular sweep.
    """
    if sweep_values is None:
        sweep_values = [0.005, 0.010, 0.012, 0.015, 0.018, 0.020, 0.022, 0.025,
                        0.028, 0.030, 0.031, 0.032, 0.035, 0.040, 0.050, 0.060,
                        0.080, 0.100]

    points = []
    for record in records:
        top_score = finite_number(record.get("trace", {}).get("initial_retrieval_top_score"))
        if top_score is None:
            continue
        tags = {str(tag) for tag in record.get("tags") or []}
        if "should-trigger-multiquery" in tags:
            should_trigger = True
        elif "should-not-trigger-multiquery" in tags:
            should_trigger = False
        else:
            continue
        classification = next(
            (
                span.get("attributes") or {}
                for span in record.get("trace", {}).get("spans") or []
                if span.get("name") == "query_classification"
            ),
            {},
        )
        points.append({
            "top_score": top_score,
            "should_trigger": should_trigger,
            "simple": bool(classification.get("simple")),
            "channel_agreement": bool(
                classification.get("channel_top_source_agreement")
            ),
        })

    if not points:
        return {"error": "No records with initial_retrieval_top_score found"}

    results = []
    for threshold in sweep_values:
        tp = tn = fp = fn = 0
        for point in points:
            retrieval_sufficient = bool(
                point["top_score"] >= threshold
                and (point["simple"] or point["channel_agreement"])
            )
            predicted = not retrieval_sufficient
            expected = point["should_trigger"]
            if expected and predicted:
                tp += 1
            elif expected and not predicted:
                fn += 1
            elif not expected and predicted:
                fp += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        accuracy = (tp + tn) / len(points)

        results.append({
            "threshold": round(threshold, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "predicted_trigger_rate": round((tp + fp) / len(points), 4),
            "sample_count": len(points),
        })

    # 首先最大化触发判断 F1，再以准确率和较低误触发数打破平局。
    best = max(
        results,
        key=lambda result: (
            result["f1"], result["accuracy"], -result["fp"]
        ),
    )

    return {
        "sweep_results": results,
        "recommended_threshold": best["threshold"],
        "recommended_trigger_rate": best["predicted_trigger_rate"],
        "recommended_f1": best["f1"],
        "current_threshold": 0.031,
        "labeled_samples": len(points),
        "method": (
            "Maximize F1 against explicit routing labels using the production "
            "RRF + simple-query + channel-agreement decision"
        ),
    }


def analyze_reranker_threshold(
    records: List[Dict[str, Any]],
    sweep_values: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Analyze optimal RERANKER_NOT_FOUND_THRESHOLD.

    Uses reranker scores from trace data to find best OOD separation.
    """
    if sweep_values is None:
        sweep_values = [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28,
                        0.30, 0.32, 0.35, 0.38, 0.40, 0.45, 0.50, 0.60, 0.80]

    points = []
    excluded_fallbacks = 0
    missing_scores = 0
    for record in records:
        trace = record.get("trace", {})
        rerank_rankings = trace.get("rerank_rankings", [])
        expected_status = record.get("expected_answer_status", "answerable")
        actual_status = record.get("answer_status")

        rerank_span = next(
            (
                span
                for span in trace.get("spans") or []
                if span.get("name") == "rerank"
            ),
            {},
        )
        rerank_attributes = rerank_span.get("attributes") or {}
        if rerank_attributes.get("fallback"):
            excluded_fallbacks += 1
            continue

        top_rerank_score = None
        if rerank_rankings:
            top_rerank_score = finite_number(rerank_rankings[0].get("score"))

        if top_rerank_score is None:
            missing_scores += 1
            continue
        points.append({
            "top_rerank_score": top_rerank_score,
            "expected_status": expected_status,
            "actual_status": actual_status,
            "answer_status_correct": bool(record.get("answer_status_correct")),
        })

    if not points:
        return {"error": "No records with rerank rankings found"}

    results = []
    for threshold in sweep_values:
        tp = tn = fp = fn = 0  # OOD = positive class
        for p in points:
            is_ood = p["expected_status"] == "not_found"
            classified_ood = p["top_rerank_score"] < threshold
            if is_ood and classified_ood:
                tp += 1
            elif is_ood and not classified_ood:
                fn += 1
            elif not is_ood and classified_ood:
                fp += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(points) if points else 0

        results.append({
            "threshold": round(threshold, 4),
            "ood_precision": round(precision, 4),
            "ood_recall": round(recall, 4),
            "ood_f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })

    # Best threshold maximizes F1 for OOD detection, then accuracy and precision.
    best = max(
        results,
        key=lambda r: (r["ood_f1"], r["accuracy"], r["ood_precision"]),
    )
    # Also find threshold with best accuracy
    best_acc = max(results, key=lambda r: r["accuracy"])

    return {
        "sweep_results": results,
        "recommended_threshold": best["threshold"],
        "recommended_f1": best["ood_f1"],
        "recommended_accuracy": best_acc["accuracy"],
        "current_threshold": 0.50,
        "sample_count": len(points),
        "excluded_fallbacks": excluded_fallbacks,
        "missing_scores": missing_scores,
        "method": "Maximize F1 score for OOD vs answerable classification using reranker top score",
    }


def analyze_answer_status_thresholds(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze optimal ANSWER_STATUS_THRESHOLD_HIGH and ANSWER_STATUS_THRESHOLD_LOW.

    Uses RRF fusion scores to find best 3-class separation.
    """
    points = []
    for record in records:
        top_rrf = finite_number(
            record.get("trace", {}).get("final_retrieval_top_score")
        )
        expected = record.get("expected_answer_status", "answerable")

        # Also check candidate_rankings for RRF score
        if top_rrf is None:
            candidates = record.get("trace", {}).get("candidate_rankings", [])
            if candidates:
                top_rrf = finite_number(candidates[0].get("score"))

        if top_rrf is not None:
            points.append({
                "top_rrf": top_rrf,
                "expected_status": expected,
            })

    if not points:
        return {"error": "No records with RRF scores found"}

    answerable_scores = [
        p["top_rrf"] for p in points
        if p["expected_status"] != "not_found"
    ]
    not_found_scores = [p["top_rrf"] for p in points if p["expected_status"] == "not_found"]

    result = {
        "answerable_rrf_stats": {
            "count": len(answerable_scores),
            "mean": round(statistics.fmean(answerable_scores), 6) if answerable_scores else None,
            "p10": round(sorted(answerable_scores)[max(0, len(answerable_scores)//10)], 6) if answerable_scores else None,
            "p25": round(sorted(answerable_scores)[max(0, len(answerable_scores)//4)], 6) if answerable_scores else None,
            "p50": round(sorted(answerable_scores)[len(answerable_scores)//2], 6) if answerable_scores else None,
        },
        "not_found_rrf_stats": {
            "count": len(not_found_scores),
            "mean": round(statistics.fmean(not_found_scores), 6) if not_found_scores else None,
            "p90": round(sorted(not_found_scores)[min(len(not_found_scores)-1, len(not_found_scores)*9//10)], 6) if not_found_scores else None,
        },
    }

    # RRF 是相关片段的融合排名信号，并不天然是 OOD 概率。若 OOD 分布没有
    # 明确低于 answerable，拒绝用反向分布生成会导致大面积误拒答的阈值。
    if not_found_scores and answerable_scores:
        nf_p90 = sorted(not_found_scores)[min(len(not_found_scores)-1, len(not_found_scores)*9//10)]
        ans_p25 = sorted(answerable_scores)[max(0, len(answerable_scores)//4)]
        if nf_p90 < ans_p25:
            result["recommended_low"] = round(max(0.005, nf_p90 + 0.002), 4)
            result["recommended_high"] = round(
                max(result["recommended_low"] + 0.005, ans_p25), 4
            )
            result["method"] = "Separate lower OOD RRF tail from answerable distribution"
        else:
            result["recommended_low"] = 0.012
            result["recommended_high"] = 0.025
            result["method"] = "Keep current values; RRF distributions are not OOD-separable"
            result["warning"] = (
                "Not-found RRF scores overlap or exceed answerable scores; "
                "use the calibrated reranker threshold for OOD rejection."
            )
    else:
        result["recommended_low"] = 0.012
        result["recommended_high"] = 0.025

    return result


def generate_report(
    rrf_analysis: Dict[str, Any],
    reranker_analysis: Dict[str, Any],
    status_analysis: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> str:
    """Generate a Markdown calibration report."""
    lines = [
        "# Threshold Calibration Report",
        "",
        f"Analyzed {len(records)} calibration questions.",
        "",
        "## 1. MultiQuery Trigger Threshold (`SIMPLE_QUERY_MIN_RRF_SCORE`)",
        "",
        f"- **Current value**: {rrf_analysis.get('current_threshold', 'N/A')}",
        f"- **Recommended value**: **{rrf_analysis.get('recommended_threshold', 'N/A')}**",
        f"- **Expected MultiQuery trigger rate**: {rrf_analysis.get('recommended_trigger_rate', 'N/A')}",
        f"- **Routing-label F1**: {rrf_analysis.get('recommended_f1', 'N/A')}",
        f"- **Method**: {rrf_analysis.get('method', 'N/A')}",
        "",
        "| Threshold | Precision | Recall | F1 | Accuracy | Trigger Rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rrf_analysis.get("sweep_results", [])[::2]:  # every other for brevity
        lines.append(
            f"| {r['threshold']} | {r['precision']:.2%} | {r['recall']:.2%} | "
            f"{r['f1']:.2%} | {r['accuracy']:.2%} | {r['predicted_trigger_rate']:.2%} |"
        )

    lines.extend([
        "",
        "## 2. OOD Rejection Threshold (`RERANKER_NOT_FOUND_THRESHOLD`)",
        "",
        f"- **Current value**: {reranker_analysis.get('current_threshold', 'N/A')}",
        f"- **Recommended value**: **{reranker_analysis.get('recommended_threshold', 'N/A')}**",
        f"- **Best F1 score**: {reranker_analysis.get('recommended_f1', 'N/A')}",
        f"- **Method**: {reranker_analysis.get('method', 'N/A')}",
        "",
        "| Threshold | OOD Precision | OOD Recall | F1 | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ])
    for r in reranker_analysis.get("sweep_results", [])[::2]:
        lines.append(
            f"| {r['threshold']} | {r['ood_precision']:.2%} | {r['ood_recall']:.2%} | {r['ood_f1']:.2%} | {r['accuracy']:.2%} |"
        )

    lines.extend([
        "",
        "## 3. Answer Status Thresholds",
        "",
        "### RRF Score Distribution",
        "",
        f"- **Answerable** (n={status_analysis.get('answerable_rrf_stats', {}).get('count', 0)}): "
        f"mean={status_analysis.get('answerable_rrf_stats', {}).get('mean', 'N/A')}, "
        f"p25={status_analysis.get('answerable_rrf_stats', {}).get('p25', 'N/A')}, "
        f"p50={status_analysis.get('answerable_rrf_stats', {}).get('p50', 'N/A')}",
        f"- **Not Found** (n={status_analysis.get('not_found_rrf_stats', {}).get('count', 0)}): "
        f"mean={status_analysis.get('not_found_rrf_stats', {}).get('mean', 'N/A')}, "
        f"p90={status_analysis.get('not_found_rrf_stats', {}).get('p90', 'N/A')}",
        "",
        "### Recommendations",
        "",
        f"- **ANSWER_STATUS_THRESHOLD_LOW**: **{status_analysis.get('recommended_low', 'N/A')}** (current: 0.012)",
        f"- **ANSWER_STATUS_THRESHOLD_HIGH**: **{status_analysis.get('recommended_high', 'N/A')}** (current: 0.025)",
        f"- **Method**: {status_analysis.get('method', 'N/A')}",
        *(
            [f"- **Warning**: {status_analysis['warning']}"]
            if status_analysis.get("warning") else []
        ),
        "",
        "## 4. Summary of Recommended Changes",
        "",
        "```env",
        f"SIMPLE_QUERY_MIN_RRF_SCORE={rrf_analysis.get('recommended_threshold', 'N/A')}",
        f"RERANKER_NOT_FOUND_THRESHOLD={reranker_analysis.get('recommended_threshold', 'N/A')}",
        f"ANSWER_STATUS_THRESHOLD_LOW={status_analysis.get('recommended_low', 'N/A')}",
        f"ANSWER_STATUS_THRESHOLD_HIGH={status_analysis.get('recommended_high', 'N/A')}",
        "```",
        "",
        "> ⚠️ These recommendations are based on the calibration dataset only.",
        "> Do NOT tune further against blind_test.json.",
        "> Apply the recommended values to .env, restart the service, and run the blind test benchmark.",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate RAG thresholds from benchmark trace data"
    )
    parser.add_argument(
        "--report", type=Path,
        default=PROJECT_ROOT / "tests/benchmark/results/regression_report.json",
        help="Path to regression report JSON from benchmark run on calibration set",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "tests/benchmark/results",
    )
    parser.add_argument(
        "--simple", action="store_true",
        help="Simple mode: analyze RRF score distributions without requiring API calls",
    )
    args = parser.parse_args()

    if not args.report.exists():
        print(f"Error: Report not found: {args.report}")
        print("Run benchmark on calibration set first:")
        print("  python tests/benchmark/benchmark.py \\")
        print("    --questions tests/benchmark/splits/calibration.json \\")
        print("    --version calibration-v1")
        sys.exit(1)

    report = load_json(args.report)
    report_split = (report.get("dataset") or {}).get("split")
    if report_split != "calibration":
        print(
            "Error: threshold calibration requires a report generated with "
            "--split calibration; got " + repr(report_split)
        )
        sys.exit(1)
    records = report.get("records", [])
    if not records:
        print("Error: No records found in report")
        sys.exit(1)

    print(f"Analyzing {len(records)} calibration records...")

    rrf_analysis = analyze_rrf_threshold(records)
    reranker_analysis = analyze_reranker_threshold(records)
    status_analysis = analyze_answer_status_thresholds(records)
    analysis_errors = [
        result["error"]
        for result in (rrf_analysis, reranker_analysis, status_analysis)
        if result.get("error")
    ]
    if analysis_errors:
        print("Error: calibration trace is incomplete:")
        for error in analysis_errors:
            print(f"  - {error}")
        sys.exit(1)

    # Generate report
    md_report = generate_report(rrf_analysis, reranker_analysis, status_analysis, records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.output_dir / "calibration_report.md"
    md_path.write_text(md_report, encoding="utf-8")
    print(f"Calibration report: {md_path}")

    json_path = args.output_dir / "calibration_report.json"
    json_path.write_text(json.dumps({
        "rrf_analysis": rrf_analysis,
        "reranker_analysis": reranker_analysis,
        "answer_status_analysis": status_analysis,
        "recommendations": {
            "SIMPLE_QUERY_MIN_RRF_SCORE": rrf_analysis.get("recommended_threshold"),
            "RERANKER_NOT_FOUND_THRESHOLD": reranker_analysis.get("recommended_threshold"),
            "ANSWER_STATUS_THRESHOLD_LOW": status_analysis.get("recommended_low"),
            "ANSWER_STATUS_THRESHOLD_HIGH": status_analysis.get("recommended_high"),
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Calibration data: {json_path}")

    # Print summary
    print()
    print("=" * 50)
    print("RECOMMENDED THRESHOLDS")
    print("=" * 50)
    print(f"SIMPLE_QUERY_MIN_RRF_SCORE = {rrf_analysis.get('recommended_threshold')}")
    print(f"RERANKER_NOT_FOUND_THRESHOLD = {reranker_analysis.get('recommended_threshold')}")
    print(f"ANSWER_STATUS_THRESHOLD_LOW = {status_analysis.get('recommended_low')}")
    print(f"ANSWER_STATUS_THRESHOLD_HIGH = {status_analysis.get('recommended_high')}")
    print()
    print("Apply these to .env and restart the service.")
    print("Then run: python tests/benchmark/benchmark.py --split blind_test --version v2.0")


if __name__ == "__main__":
    main()
