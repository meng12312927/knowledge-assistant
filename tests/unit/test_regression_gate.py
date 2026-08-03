from tests.benchmark.regression_gate import run_bootstrap_gate


def _report(recall_at_10=0.98):
    return {
        "dataset": {"split": "blind_test", "questions": 111},
        "smoke_gate": {"success": True},
        "provenance": {
            "dataset_sha256": "dataset",
            "git": {"commit": "abc123", "dirty": False},
            "model_fingerprint": "models",
            "threshold_fingerprint": "thresholds",
        },
        "summary": {
            "samples": 111,
            "successful": 111,
            "error_rate": 0.0,
            "retrieval": {"recall_at_10": recall_at_10},
        },
    }


def _profiles():
    return {
        "tiers": {
            "quality": {
                "baseline": "tests/benchmark/baselines/quality/latest.json",
                "metrics": {
                    "retrieval.recall_at_10": {
                        "direction": "higher",
                        "mode": "absolute",
                        "minimum": 0.97,
                    }
                },
            }
        }
    }


def test_bootstrap_gate_accepts_complete_blind_report_above_floor():
    result = run_bootstrap_gate(_report(), _profiles(), ["quality"])

    assert result["decision"] == "PASS"
    assert result["tiers"][0]["decision"] == "PASS"


def test_bootstrap_gate_rejects_metric_below_absolute_floor():
    result = run_bootstrap_gate(_report(0.96), _profiles(), ["quality"])

    assert result["decision"] == "BLOCK"
    assert result["tiers"][0]["failures"][0]["reason"] == "below_minimum"


def test_bootstrap_gate_rejects_partial_diagnostic_run():
    report = _report()
    report["dataset"]["questions"] = 2
    report["summary"]["samples"] = 2
    report["summary"]["successful"] = 2

    result = run_bootstrap_gate(report, _profiles(), ["quality"])

    assert result["decision"] == "BLOCK"
    assert result["safety_failures"]


def test_bootstrap_gate_rejects_dirty_git_provenance():
    report = _report()
    report["provenance"]["git"]["dirty"] = True

    result = run_bootstrap_gate(report, _profiles(), ["quality"])

    assert result["decision"] == "BLOCK"
    assert result["provenance_complete"] is False
