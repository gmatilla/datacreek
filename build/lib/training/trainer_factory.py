"""Factory for selecting the appropriate TRL trainer based on the task.

The factory centralises the mapping between high level task identifiers and
`trl` trainer classes. It forwards arbitrary keyword arguments allowing callers
to configure batch size, learning rate, PEFT adapters, etc.
"""

from __future__ import annotations

from typing import Any, Dict, Type

try:
    # These imports are optional to keep the project lightweight when TRL is
    # not installed. They will be monkeypatched in unit tests.
    from trl import DPOTrainer, PPOTrainer, SFTTrainer
except Exception:  # pragma: no cover - handled in tests via monkeypatching
    DPOTrainer = PPOTrainer = SFTTrainer = None  # type: ignore


TRAINER_MAP: Dict[str, Type[Any]] = {
    "generation": SFTTrainer,
    "qa": SFTTrainer,  # QA uses the same supervised fine-tuning trainer
    "classification": SFTTrainer,
    "rlhf_ppo": PPOTrainer,
    "rlhf_dpo": DPOTrainer,
}


def build_trainer(
    task: str,
    model: Any,
    train_dataset: Any,
    eval_dataset: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Instantiate a TRL trainer suited for ``task``.

    Parameters
    ----------
    task:
        One of ``generation``, ``qa``, ``classification``, ``rlhf_ppo`` or
        ``rlhf_dpo``.
    model:
        The model to train.
    train_dataset:
        Dataset used for training.
    eval_dataset:
        Optional dataset for evaluation.
    **kwargs:
        Additional arguments forwarded to the trainer constructor such as
        ``batch_size``, ``learning_rate`` or PEFT configuration.

    Returns
    -------
    Any
        An instance of the appropriate trainer class.

    Raises
    ------
    ValueError
        If ``task`` is not recognised.
    ImportError
        If the required trainer class is unavailable because ``trl`` is not
        installed.
    """

    if task not in TRAINER_MAP:
        raise ValueError(f"Unsupported task: {task}")

    trainer_cls = TRAINER_MAP[task]
    if trainer_cls is None:
        # TRL is not installed; provide a clearer message.
        raise ImportError("TRL is required for trainer construction")

    return trainer_cls(
        model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, **kwargs
    )
