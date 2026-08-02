from app.web.main import format_trace_metric


def test_format_trace_metric_distinguishes_missing_and_zero_values():
    assert format_trace_metric(None, "ms") == "N/A"
    assert format_trace_metric(0, "ms") == "0 ms"
    assert format_trace_metric(12.5, "ms") == "12.5 ms"
    assert format_trace_metric(None) == "N/A"
