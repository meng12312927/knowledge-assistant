import json

from tests.benchmark.benchmark import (
    baseline_snapshot,
    calculate_costs,
    compare,
    evaluate_subquestions,
    evaluate_thresholds,
    ndcg_at,
    recall_at,
    reciprocal_rank,
    summarize,
    write_baseline,
)
from tests.benchmark.regression_gate import run_gate
from tests.benchmark.calibrate_thresholds import (
    analyze_reranker_threshold,
    analyze_rrf_threshold,
)


def test_retrieval_metrics():
    ranking = ["x", "expected", "y"]
    relevant = {"expected"}

    assert recall_at(ranking, relevant, 1) == 0
    assert recall_at(ranking, relevant, 5) == 1
    assert reciprocal_rank(ranking, relevant) == 0.5
    assert 0 < ndcg_at(ranking, relevant, 10) < 1


def test_subquestion_metrics_measure_status_and_complete_coverage():
    result = evaluate_subquestions(
        [
            {"id": "SQ1", "status": "answerable", "expected_chunks": ["a"]},
            {"id": "SQ2", "status": "not_found", "expected_chunks": []},
        ],
        [
            {
                "subquestion_id": "SQ1",
                "status": "answerable",
                "selected_stable_chunk_ids": ["a"],
            },
            {
                "subquestion_id": "SQ2",
                "status": "not_found",
                "selected_stable_chunk_ids": [],
            },
        ],
    )

    assert result == {
        "evaluated": 2,
        "status_accuracy": 1.0,
        "evidence_recall": 1.0,
        "complete_evidence": True,
    }


def test_cost_is_unknown_when_used_stage_has_no_price():
    result = calculate_costs(
        {
            "generation": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            }
        },
        {"currency": "CNY", "per_million_tokens": {}},
    )

    assert result["generation"] is None
    assert result["total"] is None
    assert result["complete"] is False


def test_cost_uses_per_million_token_prices():
    result = calculate_costs(
        {
            "generation": {
                "prompt_tokens": 1_000_000,
                "completion_tokens": 500_000,
                "total_tokens": 1_500_000,
            }
        },
        {
            "currency": "CNY",
            "per_million_tokens": {
                "generation_input": 2,
                "generation_output": 4,
            },
        },
    )

    assert result["generation"] == 4
    assert result["total"] == 4
    assert result["complete"] is True


def test_reranker_cost_uses_dedicated_token_field():
    result = calculate_costs(
        {"rerank": {"reranker_tokens": 1_000_000, "total_tokens": 1_000_000}},
        {
            "currency": "CNY",
            "per_million_tokens": {"reranker_input": 0.5},
        },
    )

    assert result["reranker"] == 0.5
    assert result["total"] == 0.5
    assert result["complete"] is True


def test_cost_summary_does_not_turn_unknown_price_into_zero():
    records = [
        {
            "success": True,
            "expected_answer_status": "answerable",
            "verification_status": "verified",
            "answer_status_correct": True,
            "unsupported_claims": 0,
            "hallucination": False,
            "retrieval_metrics": {},
            "cost": {"generation": None, "complete": False},
        },
        {
            "success": True,
            "expected_answer_status": "answerable",
            "verification_status": "verified",
            "answer_status_correct": True,
            "unsupported_claims": 0,
            "hallucination": False,
            "retrieval_metrics": {},
            "cost": {"generation": 0.0, "complete": True},
        },
    ]

    summary = summarize(records, wall_seconds=1, pricing={"currency": "CNY"})

    assert summary["cost"]["generation"]["avg"] is None
    assert summary["cost"]["generation"]["sum"] is None


def test_regression_gate_detects_latency_and_recall_regressions():
    baseline = {
        "latency": {"client_done_ms": {"p95": 1000}},
        "retrieval": {"recall_at_5": 0.95},
    }
    current = {
        "latency": {"client_done_ms": {"p95": 1200}},
        "retrieval": {"recall_at_5": 0.90},
    }
    comparison = compare(baseline, current)
    failures = evaluate_thresholds(
        comparison,
        {
            "latency.client_done_ms.p95": {
                "direction": "lower",
                "mode": "relative",
                "tolerance": 0.1,
            },
            "retrieval.recall_at_5": {
                "direction": "higher",
                "mode": "absolute",
                "tolerance": 0.01,
            },
        },
    )

    assert {item["path"] for item in failures} == {
        "latency.client_done_ms.p95",
        "retrieval.recall_at_5",
    }


def test_multiquery_threshold_uses_explicit_routing_labels():
    records = [
        {
            "tags": ["should-not-trigger-multiquery"],
            "trace": {
                "initial_retrieval_top_score": 0.04,
                "spans": [{
                    "name": "query_classification",
                    "attributes": {
                        "simple": True,
                        "channel_top_source_agreement": False,
                    },
                }],
            },
        },
        {
            "tags": ["should-trigger-multiquery"],
            "trace": {
                "initial_retrieval_top_score": 0.01,
                "spans": [{
                    "name": "query_classification",
                    "attributes": {
                        "simple": False,
                        "channel_top_source_agreement": False,
                    },
                }],
            },
        },
        {
            "tags": ["boundary-test"],
            "trace": {"initial_retrieval_top_score": 0.02},
        },
    ]

    result = analyze_rrf_threshold(records, sweep_values=[0.015, 0.05])

    assert result["recommended_threshold"] == 0.015
    assert result["recommended_f1"] == 1.0
    assert result["labeled_samples"] == 2


def test_reranker_threshold_ignores_missing_upstream_scores():
    records = [
        {
            "expected_answer_status": "answerable",
            "trace": {"rerank_rankings": [{"score": 0.8}]},
        },
        {
            "expected_answer_status": "not_found",
            "trace": {"rerank_rankings": [{"score": 0.1}]},
        },
        {
            "expected_answer_status": "not_found",
            "trace": {"rerank_rankings": []},
        },
    ]

    result = analyze_reranker_threshold(records, sweep_values=[0.2, 0.9])

    assert result["recommended_threshold"] == 0.2
    assert result["recommended_f1"] == 1.0


def test_promoted_baseline_excludes_per_question_traces(tmp_path):
    report = {
        "schema_version": 2,
        "generated_at": "2026-08-02T00:00:00Z",
        "version": "v2",
        "dataset": {"effective_sha256": "abc", "questions": 111},
        "config": {"concurrency": 5},
        "environment": {"python": "3.12"},
        "summary": {"error_rate": 0},
        "records": [{"trace": {"candidate_rankings": ["large"]}}],
    }
    path = tmp_path / "baseline.json"

    write_baseline(report, path)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved == baseline_snapshot(report)
    assert "records" not in saved
    assert saved["dataset"]["questions"] == 111


def test_layered_gate_blocks_floor_violation_and_binds_provenance(tmp_path):
    baseline = {
        "version": "v1",
        "dataset": {"effective_sha256": "abc", "questions": 111},
        "summary": {"retrieval": {"recall_at_10": 0.98}},
        "provenance": {
            "model_fingerprint": "model-v1",
            "threshold_fingerprint": "threshold-v1",
        },
    }
    baseline_path = tmp_path / "quality.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    report = {
        "dataset": {"effective_sha256": "abc", "questions": 111},
        "summary": {"retrieval": {"recall_at_10": 0.94}},
        "smoke_gate": {"success": True},
        "provenance": {
            "dataset_sha256": "abc",
            "git": {"commit": "deadbeef", "dirty": False},
            "model_fingerprint": "model-v2",
            "threshold_fingerprint": "threshold-v2",
        },
    }
    profiles = {
        "tiers": {
            "quality": {
                "baseline": str(baseline_path),
                "metrics": {
                    "retrieval.recall_at_10": {
                        "direction": "higher",
                        "mode": "absolute",
                        "tolerance": 0.01,
                        "minimum": 0.95,
                    }
                },
            }
        }
    }

    result = run_gate(report, profiles, ["quality"])

    assert result["decision"] == "BLOCK"
    assert result["provenance_complete"] is True
    assert result["tiers"][0]["configuration_changed"] == {
        "models": True,
        "thresholds": True,
    }
    assert result["tiers"][0]["failures"][0]["reason"] == "below_minimum"
