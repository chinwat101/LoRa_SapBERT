"""
src/training/callbacks.py
Custom Hugging Face Trainer callbacks for logging and checkpoint management.
"""
import logging
from pathlib import Path

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

logger = logging.getLogger(__name__)


class LoggingCallback(TrainerCallback):
    """Log training metrics to Python logger at every log step."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs is None:
            return
        step = state.global_step
        metrics = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in logs.items()}
        logger.info(f"Step {step:6d} | {metrics}")


class BestModelCallback(TrainerCallback):
    """Log when a new best model checkpoint is saved."""

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.best_model_checkpoint:
            logger.info(f"Best checkpoint: {state.best_model_checkpoint} "
                        f"(metric={state.best_metric:.4f})")
