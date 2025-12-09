from scripts.watchdog import check_alerts, parse_metrics


def test_parse_metrics_simple():
    text = 'eigsh_timeouts_total 12\nann_latency_seconds_bucket{le="2"} 90\nann_latency_seconds_count 100\n'
    metrics = parse_metrics(text)
    assert metrics["eigsh_timeouts_total"] == 12
    assert metrics["ann_latency_seconds_count"] == 100


def test_check_alerts():
    metrics = {
        "eigsh_timeouts_total": 22,
        'ann_latency_seconds_bucket{le="2"}': 90,
        "ann_latency_seconds_count": 100,
    }
    alerts = check_alerts(metrics, {"eigsh_timeouts_total": 5}, 86.0)
    assert set(alerts) == {"eigsh_timeouts_total", "ann_latency", "disk_usage"}
