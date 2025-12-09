import sys
import types
from importlib import reload

import pytest


def test_eta_callback_export():
    """Ensure EtaCallback is re-exported at package level."""
    from training import EtaCallback

    assert hasattr(EtaCallback, "update")


def test_init_wandb(monkeypatch):
    dummy = types.SimpleNamespace(
        init=lambda **kwargs: types.SimpleNamespace(kwargs=kwargs)
    )
    monkeypatch.setitem(sys.modules, "wandb", dummy)
    from training import monitoring

    reload(monitoring)
    run = monitoring.init_wandb(project="proj", entity="me")
    assert run.kwargs["project"] == "proj" and run.kwargs["entity"] == "me"


class DummyGauge:
    def __init__(self, name, desc):
        self.name = name
        self.desc = desc
        self.value = None

    def set(self, value):
        self.value = value


def test_prometheus_logger(monkeypatch):
    dummy = types.SimpleNamespace(Gauge=DummyGauge, start_http_server=lambda port: None)
    monkeypatch.setitem(sys.modules, "prometheus_client", dummy)
    from training import monitoring

    reload(monitoring)
    logger = monitoring.PrometheusLogger()
    logger.log(
        training_loss=0.5,
        val_metric=0.4,
        gpu_vram_bytes=1.0,
        reward_avg=0.7,
        training_eta_seconds=9.0,
        fractal_loss=0.1,
    )
    assert logger.training_loss.value == 0.5
    assert logger.val_metric.value == 0.4
    assert logger.gpu_vram_bytes.value == 1.0
    assert logger.reward_avg.value == 0.7
    assert logger.training_eta_seconds.value == 9.0
    assert logger.fractal_loss.value == 0.1


def test_early_stopping():
    from training.monitoring import EarlyStopping

    stopper = EarlyStopping(patience=2)
    assert not stopper.step(1.0)
    assert not stopper.step(1.1)
    assert stopper.step(1.2)


def test_eta_callback(monkeypatch):
    dummy = types.SimpleNamespace(Gauge=DummyGauge, start_http_server=lambda port: None)
    monkeypatch.setitem(sys.modules, "prometheus_client", dummy)
    from training import monitoring

    reload(monitoring)
    logger = monitoring.PrometheusLogger()
    times = iter([0.0, 10.0])
    monkeypatch.setattr(monitoring.time, "perf_counter", lambda: next(times))
    cb = monitoring.EtaCallback(steps_total=100, logger=logger)
    eta = cb.update(step_done=50)
    assert eta == 10.0
    assert logger.training_eta_seconds.value == 10.0


def test_fractal_dim_callback(monkeypatch):
    dummy = types.SimpleNamespace(Gauge=DummyGauge, start_http_server=lambda port: None)
    monkeypatch.setitem(sys.modules, "prometheus_client", dummy)
    from training import monitoring

    reload(monitoring)
    logger = monitoring.PrometheusLogger()
    # Patch fractal_dim_embedding to return a fixed dimension
    monkeypatch.setattr(monitoring, "fractal_dim_embedding", lambda e, r: 1.5)
    cb = monitoring.FractalDimCallback(target_dim=1.0, beta=2.0, logger=logger)
    # First epoch: no computation
    assert cb.update({1: [0.0]}) is None
    # Second epoch: dimension computed, loss logged
    dim = cb.update({1: [0.0]})
    assert dim == 1.5
    assert logger.fractal_loss.value == 1.0
