import sys
import types
from importlib import reload


def test_mlflow_callback_logs_params_metrics_and_artifact(tmp_path, monkeypatch):
    """The callback should forward hyperparameters, metrics and artifacts."""

    calls = {}

    def start_run():
        calls["start_run"] = True

    def log_params(params):
        calls["params"] = params

    def log_metrics(metrics, step=None):
        calls.setdefault("metrics", []).append((metrics, step))

    def log_artifact(path):
        calls["artifact"] = path

    def end_run():
        calls["end_run"] = True

    dummy_mlflow = types.SimpleNamespace(
        start_run=start_run,
        log_params=log_params,
        log_metrics=log_metrics,
        log_artifact=log_artifact,
        end_run=end_run,
    )
    monkeypatch.setitem(sys.modules, "mlflow", dummy_mlflow)
    from training import callbacks

    reload(callbacks)
    cb = callbacks.MlflowLoggingCallback(template_sha="abc123")
    args = types.SimpleNamespace(
        learning_rate=1e-4,
        num_train_epochs=2,
        output_dir=str(tmp_path),
    )
    state = types.SimpleNamespace(global_step=1)
    control = object()

    cb.on_train_begin(args, state, control)
    cb.on_log(args, state, control, logs={"val_loss": 0.5, "reward_avg": 0.8})
    model_path = tmp_path / "model.gguf"
    model_path.write_text("dummy")
    cb.on_train_end(args, state, control)

    assert calls["params"] == {
        "learning_rate": 1e-4,
        "epochs": 2,
        "template_sha": "abc123",
    }
    assert ({"val_loss": 0.5, "reward_avg": 0.8}, 1) in calls["metrics"]
    assert calls["artifact"] == str(model_path)
    assert calls["start_run"] and calls["end_run"]
