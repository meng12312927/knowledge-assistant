from tests.benchmark.benchmark import (
    calculate_costs,
    compare,
    evaluate_thresholds,
    ndcg_at,
    recall_at,
    reciprocal_rank,
    summarize,
)


def test_retrieval_metrics():
    ranking = ["x", "expected", "y"]
    relevant = {"expected"}

    assert recall_at(ranking, relevant, 1) == 0
    assert recall_at(ranking, relevant, 5) == 1
    assert reciprocal_rank(ranking, relevant) == 0.5
    assert 0 < ndcg_at(ranking, relevant, 10) < 1


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
