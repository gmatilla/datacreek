from __future__ import annotations

"""Monitoring utilities for training metrics and callbacks."""

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

from datacreek.analysis import fractal_dim_embedding

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore

try:
    from prometheus_client import Gauge, start_http_server
except Exception:  # pragma: no cover - optional dependency
    Gauge = None  # type: ignore
    start_http_server = None  # type: ignore


def init_wandb(project: str, **kwargs):
    """Initialize Weights & Biases logging.

    Parameters
    ----------
    project:
        Name of the W&B project.
    **kwargs:
        Additional arguments forwarded to :func:`wandb.init`.
    """
    if wandb is None:  # pragma: no cover - handled in tests
        raise ImportError("wandb is not installed")
    return wandb.init(project=project, **kwargs)


@dataclass
class PrometheusLogger:
    """Expose core training metrics through Prometheus gauges."""

    port: Optional[int] = None

    def __post_init__(self) -> None:
        if Gauge is None:  # pragma: no cover - optional dependency
            raise ImportError("prometheus_client is not installed")
        self.training_loss = Gauge("training_loss", "Current training loss")
        self.val_metric = Gauge("val_metric", "Validation metric")
        self.gpu_vram_bytes = Gauge("gpu_vram_bytes", "GPU VRAM usage in bytes")
        self.reward_avg = Gauge("reward_avg", "Average reward value")
        # Estimated time remaining for the training run (seconds).
        self.training_eta_seconds = Gauge(
            "training_eta_seconds", "Estimated time remaining in seconds"
        )
        # Fractal loss indicating deviation from target dimension.
        self.fractal_loss = Gauge("fractal_loss", "Fractal dimension loss")
        if self.port is not None and start_http_server:
            start_http_server(self.port)

    def log(
        self,
        training_loss: Optional[float] = None,
        val_metric: Optional[float] = None,
        gpu_vram_bytes: Optional[float] = None,
        reward_avg: Optional[float] = None,
        training_eta_seconds: Optional[float] = None,
        fractal_loss: Optional[float] = None,
    ) -> None:
        """Update metric gauges with new values."""
        if training_loss is not None:
            self.training_loss.set(training_loss)
        if val_metric is not None:
            self.val_metric.set(val_metric)
        if gpu_vram_bytes is not None:
            self.gpu_vram_bytes.set(gpu_vram_bytes)
        if reward_avg is not None:
            self.reward_avg.set(reward_avg)
        if training_eta_seconds is not None:
            self.training_eta_seconds.set(training_eta_seconds)
        if fractal_loss is not None:
            self.fractal_loss.set(fractal_loss)


class EtaCallback:
    """Compute and log an estimate of remaining training time.

    Parameters
    ----------
    steps_total:
        Total number of optimization steps expected for the run.
    logger:
        Optional :class:`PrometheusLogger` to record the metric.

    Notes
    -----
    The estimate is based on the average step rate since the callback
    creation:

    .. math::

        \text{ETA} = \frac{\text{steps}_\text{total} - \text{step}_\text{done}}
        {\text{steps}_\text{per\_sec}}

    where ``steps_per_sec = step_done / elapsed_time``.
    """

    def __init__(self, steps_total: int, logger: Optional[PrometheusLogger] = None):
        self.steps_total = steps_total
        self.logger = logger
        # Reference time used to compute steps per second.
        self._start_time = time.perf_counter()

    def update(self, step_done: int) -> float:
        """Update the ETA based on completed steps.

        Parameters
        ----------
        step_done:
            Number of steps completed so far.

        Returns
        -------
        float
            Estimated remaining time in seconds.
        """
        elapsed = time.perf_counter() - self._start_time
        # Avoid division by zero for the initial call.
        steps_per_sec = (
            step_done / elapsed if step_done > 0 and elapsed > 0 else float("inf")
        )
        eta = (
            (self.steps_total - step_done) / steps_per_sec
            if steps_per_sec > 0
            else float("inf")
        )
        if self.logger is not None:
            self.logger.log(training_eta_seconds=eta)
        return eta


class FractalDimCallback:
    """Estimate embedding fractal dimension every two epochs.

    The callback computes the dimension using :func:`fractal_dim_embedding` and
    logs the corresponding loss

    .. math:: \mathcal{L}_{frac} = \beta |\hat D_f - D_f^{target}|.
    """

    def __init__(
        self,
        target_dim: float,
        *,
        beta: float = 1.0,
        radii: Iterable[float] = (1.0, 2.0),
        logger: Optional[PrometheusLogger] = None,
    ) -> None:
        self.target_dim = target_dim
        self.beta = beta
        self.radii = tuple(radii)
        self.logger = logger
        self._epoch = 0

    def update(self, embeddings: Mapping[object, Iterable[float]]) -> Optional[float]:
        """Update the estimate and log the fractal loss.

        Parameters
        ----------
        embeddings:
            Mapping of node identifiers to embedding vectors.

        Returns
        -------
        float | None
            Dimension estimate when computed, otherwise ``None``.
        """

        self._epoch += 1
        if self._epoch % 2:
            return None
        dim = fractal_dim_embedding(embeddings, self.radii)
        loss = self.beta * abs(dim - self.target_dim)
        if self.logger is not None:
            self.logger.log(fractal_loss=loss)
        return dim


class EarlyStopping:
    """Stop training when a monitored metric fails to improve."""

    def __init__(self, patience: int = 3, mode: str = "min") -> None:
        self.patience = patience
        self.mode = mode
        self.best: Optional[float] = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Update internal state and indicate whether training should stop."""
        if self.best is None:
            self.best = value
            return False
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class CheckpointPruner:
    """Keep only the best ``k`` checkpoints based on a validation metric.

    Parameters
    ----------
    k:
        Number of checkpoints to retain. Defaults to ``2``.
    mode:
        Whether a larger (``"max"``) or smaller (``"min"``) metric value is
        considered better.

    Notes
    -----
    After saving a checkpoint, call :meth:`step` with its path and associated
    validation metric :math:`m`. The pruner keeps the checkpoints with the
    top-:math:`k` metric values and removes the others to limit disk usage.
    """

    def __init__(self, k: int = 2, mode: str = "max") -> None:
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.k = k
        self.mode = mode
        # Store pairs of (metric, path) for existing checkpoints.
        self._checkpoints: list[tuple[float, Path]] = []

    def step(self, path: str, metric: float) -> None:
        """Register a new checkpoint and prune excess ones.

        Parameters
        ----------
        path:
            Filesystem path where the checkpoint was saved.
        metric:
            Validation metric associated with this checkpoint.
        """

        p = Path(path)
        self._checkpoints.append((metric, p))
        # Sort according to metric quality.
        self._checkpoints.sort(key=lambda x: x[0], reverse=self.mode == "max")
        # Remove checkpoints beyond the top-k.
        for _, obsolete in self._checkpoints[self.k :]:
            shutil.rmtree(obsolete, ignore_errors=True)
        self._checkpoints = self._checkpoints[: self.k]
