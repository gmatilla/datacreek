import pytest

pytest.importorskip("yaml")
import yaml


def test_jitter_alert_rule():
    with open("configs/alerts.yaml") as fh:
        data = yaml.safe_load(fh)
    rules = {}
    for r in data["groups"][0]["rules"]:
        key = r.get("alert") or r.get("record")
        rules[key] = r
    jitter = rules.get("SVGPJitterStorm")
    assert jitter is not None
    assert jitter["expr"] == "rate(gp_jitter_restarts_total[10m]) > 0.01"
    assert jitter["for"] == "10m"
    assert jitter["labels"]["severity"] == "warning"

    thresh = rules.get("redis_low_hit_ratio_threshold")
    assert thresh is not None
    assert thresh["expr"] == 0.3
    redis_rule = rules.get("RedisLowHitRatio")
    assert redis_rule is not None
    assert redis_rule["expr"] == "redis_hit_ratio < redis_low_hit_ratio_threshold"
    assert redis_rule["for"] == "5m"
    assert redis_rule["labels"]["severity"] == "warning"

    storm_thresh = rules.get("eigsh_timeout_storm_threshold")
    assert storm_thresh is not None
    assert storm_thresh["expr"] == 5
    critical_thresh = rules.get("eigsh_timeout_critical_threshold")
    assert critical_thresh is not None
    assert critical_thresh["expr"] == 2
    eigsh_rule = rules.get("EigshTimeoutCritical")
    assert eigsh_rule is not None
    assert (
        eigsh_rule["expr"]
        == "increase(eigsh_timeouts_total[1h]) > eigsh_timeout_critical_threshold"
    )
    assert eigsh_rule["for"] == "1m"
    assert eigsh_rule["labels"]["severity"] == "critical"

    cache_pressure = rules.get("CachePressure")
    assert cache_pressure is not None
    assert "ingest_queue_fill_ratio > 0.9" in cache_pressure["expr"]
    assert cache_pressure["labels"]["severity"] == "critical"

    gw_rule = rules.get("GraphwaveP95Slow")
    assert gw_rule is not None
    assert gw_rule["expr"] == "p95_graphwave_ms > 250"
    assert gw_rule["for"] == "10m"
    assert gw_rule["labels"]["severity"] == "warning"

    ingest_high = rules.get("IngestQueueHigh")
    assert ingest_high is not None
    assert ingest_high["expr"] == "ingest_queue_fill_ratio > 0.8"
    assert ingest_high["for"] == "10m"
    assert ingest_high["labels"]["severity"] == "critical"

    latency = rules.get("IngestLatencyP999High")
    assert latency is not None
    assert latency["expr"] == "ingest_latency_p999 > 5"
    assert latency["for"] == "15m"
    assert latency["labels"]["severity"] == "critical"

    burn_alert = rules.get("IngestLatencyBurnRate")
    assert burn_alert is not None
    assert (
        burn_alert["expr"]
        == "ingest_latency_burn_rate_1h > 1 and ingest_latency_burn_rate_6h > 1"
    )
    assert burn_alert["for"] == "10m"
    assert burn_alert["labels"]["severity"] == "critical"

    burn1 = rules.get("ingest_latency_burn_rate_1h")
    assert burn1 is not None
    assert burn1["expr"] == "avg_over_time(ingest_latency_p999[1h]) / 5"

    burn6 = rules.get("ingest_latency_burn_rate_6h")
    assert burn6 is not None
    assert burn6["expr"] == "avg_over_time(ingest_latency_p999[6h]) / 5"


def test_promtool_check_rules():
    import shutil
    import subprocess

    if shutil.which("promtool") is None:
        pytest.skip("promtool not available")
    subprocess.run(["promtool", "check", "rules", "configs/alerts.yaml"], check=True)


def test_promtool_test_rules():
    import shutil
    import subprocess

    if shutil.which("promtool") is None:
        pytest.skip("promtool not available")
    subprocess.run(
        [
            "promtool",
            "test",
            "rules",
            "ops/prometheus_rules_test.yml",
        ],
        check=True,
    )


def test_alertmanager_inhibition():
    with open("configs/alertmanager.yaml") as fh:
        cfg = yaml.safe_load(fh)
    inhibit = cfg.get("inhibit_rules")
    assert inhibit, "inhibit_rules missing"
    rule = inhibit[0]
    assert "critical" in rule["source_matchers"][0]
    assert "warning" in rule["target_matchers"][0]
    assert "alertname" in rule["equal"]
