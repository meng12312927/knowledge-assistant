from tests.smoke.smoke_test import _check, _observe


def test_observe_extracts_route_citations_and_tool_result():
    payload = {
        "answer_status": "answerable",
        "sources": [{"content": "x"}],
        "citations": [{"source_file": "远程办公管理办法.txt"}],
        "citation_verification": {"status": "verified"},
        "tool_results": [
            {
                "tool_name": "calculator",
                "success": True,
                "output": 1000.0,
            }
        ],
        "trace": {
            "routing_probe_strategy": "adaptive_fallback",
            "routing_probe_multiquery_triggered": True,
            "retrieval_quality": "recoverable_low",
            "spans": [
                {
                    "name": "api_total",
                    "attributes": {
                        "agent": True,
                        "agent_route_reason": "recoverable_low_retrieval",
                    },
                }
            ],
        },
    }

    observed = _observe(payload)

    assert observed["agent"] is True
    assert observed["query_strategy"] == "adaptive_fallback"
    assert observed["multiquery_triggered"] is True
    assert observed["citation_sources"] == ["远程办公管理办法.txt"]
    assert observed["tool_output"] == 1000.0


def test_check_reports_mismatches_and_accepts_valid_result():
    expected = {
        "agent": False,
        "answer_status_in": ["not_found"],
        "empty_sources": True,
        "empty_citations": True,
    }
    valid = {
        "agent": False,
        "answer_status": "not_found",
        "source_count": 0,
        "citation_count": 0,
    }
    invalid = {
        "agent": True,
        "answer_status": "answerable",
        "source_count": 2,
        "citation_count": 1,
    }

    assert _check(expected, valid) == []
    failures = _check(expected, invalid)
    assert len(failures) == 4
