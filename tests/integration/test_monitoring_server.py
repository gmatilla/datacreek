import importlib

import datacreek.analysis.monitoring as monitoring


def test_start_metrics_server_creates_gauges(monkeypatch):
    ports = {}
    monkeypatch.setattr(
        monitoring, "start_http_server", lambda p: ports.setdefault("port", p)
    )
    monitoring.start_metrics_server(9100)
    assert ports["port"] == 9100
    for name in [
        "tpl_w1",
        "sheaf_score",
        "gw_entropy",
        "autotune_cost",
        "bias_wasserstein_last",
        "haa_edges_total",
    ]:
        assert monitoring._METRICS[name] is not None
