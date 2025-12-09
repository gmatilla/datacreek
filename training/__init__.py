"""Training utilities for the adaptive fine-tuning pipeline."""

from .augmenter import ActiveLearningAugmenter
from .auto_feedback import build_reward_fn, extract_triplets
from .callbacks import TrainingEtaSecondsCallback
from .curriculum_dataloader import CurriculumDataLoader, compute_difficulty
from .monitoring import EarlyStopping, EtaCallback, PrometheusLogger, init_wandb
from .reward_fn import build_alias_reward_fn
from .task_detector import detect_task, format_classif, format_rlhf, format_sft
from .trainer_factory import build_trainer
from .unsloth_loader import add_lora, load_model

__all__ = [
    "add_lora",
    "load_model",
    "detect_task",
    "build_trainer",
    "format_sft",
    "format_classif",
    "format_rlhf",
    "extract_triplets",
    "build_reward_fn",
    "build_alias_reward_fn",
    "compute_difficulty",
    "CurriculumDataLoader",
    "ActiveLearningAugmenter",
    "init_wandb",
    "PrometheusLogger",
    "EtaCallback",
    "EarlyStopping",
    "TrainingEtaSecondsCallback",
]
