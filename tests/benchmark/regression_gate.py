"""Layered CI gate for a completed blind-test regression report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmark.benchmark import (
    compare,
    evaluate_thresholds,
    load_json,
    write_baseline,
)

DEFAULT_REPORT = PROJECT_ROOT / "tests/benchmark/results/regression_report.json"
DEFAULT_PROFILES = PROJECT_ROOT / "tests/benchmark/regression_profiles.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "tests/benchmark/results/regression_gate_report.json"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def compatible_dataset(
    baseline: Mapping[str, Any], report: Mapping[str, Any]
) -> tuple[bool, str | None]:
    expected = (baseline.get("dataset") or {}).get("effective_sha256")
    actual = (report.get("dataset") or {}).get("effective_sha256")
    expected_count = (baseline.get("dataset") or {}).get("questions")
    actual_count = (report.get("dataset") or {}).get("questions")
    if not expected or not actual:
        return False, "baseline or report has no effective dataset hash"
    if expected != actual or expected_count != actual_count:
        return False, "dataset hash or evaluated question count differs"
    return True, None


def evaluate_tier(
    report: Mapping[str, Any],
    tier_name: str,
    tier: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_path = resolve_path(str(tier.get("baseline") or ""))
    baseline = load_json(baseline_path, required=False)
    used_fallback = False
    if not isinstance(baseline, Mapping):
        fallback = tier.get("fallback_baseline")
        fallback_path = resolve_path(str(fallback)) if fallback else None
        baseline = load_json(fallback_path, required=False) if fallback_path else None
        if isinstance(baseline, Mapping):
            baseline_path = fallback_path
            used_fallback = True
    if not isinstance(baseline, Mapping):
        return {
            "tier": tier_name,
            "decision": "NO_BASELINE",
            "baseline": str(baseline_path),
            "failures": [],
            "reason": "baseline file not found",
        }

    compatible, reason = compatible_dataset(baseline, report)
    if not compatible:
        return {
            "tier": tier_name,
            "decision": "INCOMPARABLE",
            "baseline": str(baseline_path),
            "fallback_baseline": used_fallback,
            "failures": [],
            "reason": reason,
        }
    comparison = compare(baseline.get("summary") or {}, report.get("summary") or {})
    failures = evaluate_thresholds(comparison, tier.get("metrics") or {})
    return {
        "tier": tier_name,
        "decision": "REGRESSION" if failures else "PASS",
        "baseline": str(baseline_path),
        "baseline_version": baseline.get("version"),
        "fallback_baseline": used_fallback,
        "baseline_provenance": baseline.get("provenance"),
        "current_provenance": report.get("provenance"),
        "configuration_changed": {
            "models": (
                (baseline.get("provenance") or {}).get("model_fingerprint")
                != (report.get("provenance") or {}).get("model_fingerprint")
            ),
            "thresholds": (
                (baseline.get("provenance") or {}).get("threshold_fingerprint")
                != (report.get("provenance") or {}).get("threshold_fingerprint")
            ),
        },
        "failures": failures,
        "reason": None,
    }


def run_gate(
    report: Mapping[str, Any], profiles: Mapping[str, Any], tiers: list[str]
) -> dict[str, Any]:
    configured = profiles.get("tiers") or {}
    results = []
    for name in tiers:
        if name not in configured:
            raise ValueError(f"unknown regression tier: {name}")
        results.append(evaluate_tier(report, name, configured[name]))
    smoke = report.get("smoke_gate")
    smoke_passed = smoke is None or bool(smoke.get("success"))
    provenance_complete = bool(
        (report.get("provenance") or {}).get("dataset_sha256")
        and (report.get("provenance") or {}).get("git", {}).get("commit")
        and (report.get("provenance") or {}).get("git", {}).get("dirty") is False
        and (report.get("provenance") or {}).get("model_fingerprint")
        and (report.get("provenance") or {}).get("threshold_fingerprint")
    )
    passed = (
        smoke_passed
        and provenance_complete
        and all(item["decision"] == "PASS" for item in results)
    )
    return {
        "decision": "PASS" if passed else "BLOCK",
        "smoke_passed": smoke_passed,
        "provenance_complete": provenance_complete,
        "tiers": results,
    }


def run_bootstrap_gate(
    report: Mapping[str, Any], profiles: Mapping[str, Any], tiers: list[str]
) -> dict[str, Any]:
    """Validate a first blind report against absolute floors before promotion."""
    dataset = report.get("dataset") or {}
    summary = report.get("summary") or {}
    smoke = report.get("smoke_gate") or {}
    provenance = report.get("provenance") or {}
    safety_failures = []
    if dataset.get("split") != "blind_test" or dataset.get("questions") != 111:
        safety_failures.append("bootstrap requires the complete 111-question blind_test")
    if summary.get("samples") != 111 or summary.get("successful") != 111:
        safety_failures.append("all 111 blind requests must succeed")
    if float(summary.get("error_rate", 1.0) or 0.0) != 0.0:
        safety_failures.append("error_rate must be zero")
    if not smoke.get("success"):
        safety_failures.append("the five-case smoke gate must pass")
    provenance_complete = bool(
        provenance.get("dataset_sha256")
        and (provenance.get("git") or {}).get("commit")
        and (provenance.get("git") or {}).get("dirty") is False
        and provenance.get("model_fingerprint")
        and provenance.get("threshold_fingerprint")
    )
    if not provenance_complete:
        safety_failures.append(
            "report provenance is incomplete or Git worktree is dirty"
        )

    comparison = compare(summary, summary)
    configured = profiles.get("tiers") or {}
    results = []
    for name in tiers:
        if name not in configured:
            raise ValueError(f"unknown regression tier: {name}")
        failures = evaluate_thresholds(
            comparison, configured[name].get("metrics") or {}
        )
        results.append({
            "tier": name,
            "decision": "REGRESSION" if failures else "PASS",
            "baseline": str(resolve_path(configured[name]["baseline"])),
            "baseline_version": None,
            "fallback_baseline": False,
            "baseline_provenance": None,
            "current_provenance": provenance,
            "configuration_changed": {"models": False, "thresholds": False},
            "failures": failures,
            "reason": "first baseline bootstrap",
        })
    passed = not safety_failures and all(
        item["decision"] == "PASS" for item in results
    )
    return {
        "decision": "PASS" if passed else "BLOCK",
        "mode": "bootstrap",
        "smoke_passed": bool(smoke.get("success")),
        "provenance_complete": provenance_complete,
        "safety_failures": safety_failures,
        "tiers": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="分层 RAG Regression Gate")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--tier", action="append", choices=["quality", "performance"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="人工审核通过后，把当前报告提升为所选分层 Baseline",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="仅首次建基线：要求完整盲测、Smoke、溯源和绝对质量下限全部通过",
    )
    args = parser.parse_args()
    report = load_json(args.report)
    profiles = load_json(args.profiles)
    tiers = args.tier or ["quality", "performance"]
    if args.bootstrap and not args.promote:
        raise SystemExit("--bootstrap 必须与 --promote 同时使用")
    result = (
        run_bootstrap_gate(report, profiles, tiers)
        if args.bootstrap
        else run_gate(report, profiles, tiers)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for item in result["tiers"]:
        print(
            f"[{item['decision']}] {item['tier']} baseline={item['baseline']} "
            f"failures={len(item['failures'])}"
        )
    if args.promote:
        if result["decision"] != "PASS":
            raise SystemExit("gate 未通过，拒绝提升 baseline")
        configured = profiles.get("tiers") or {}
        for name in tiers:
            target = resolve_path(configured[name]["baseline"])
            write_baseline(report, target, tier=name)
            print(f"[PROMOTED] {name}: {target}")
    if result["decision"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
