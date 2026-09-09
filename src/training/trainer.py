"""
src/training/trainer.py
MLM training pipeline with LoRA using Hugging Face Trainer.
"""
import math
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.training.callbacks import LoggingCallback, BestModelCallback

logger = get_logger(__name__)


def tokenize_dataset(dataset, tokenizer, max_length: int = 128):
    """
    Tokenise a HuggingFace Dataset of {"text": str} records.

    Args:
        dataset: HuggingFace Dataset.
        tokenizer: Loaded tokenizer.
        max_length: Maximum sequence length (truncate).

    Returns:
        Tokenised dataset.
    """
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,         # DataCollator handles dynamic padding
            return_special_tokens_mask=True,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenising",
    )
    return tokenized


def _compute_warmup_steps(cfg: dict) -> int:
    """
    Convert warmup_ratio to warmup_steps for transformers versions
    that don't support warmup_ratio natively.

    If 'warmup_steps' is explicitly set in config, use that directly.
    Otherwise, estimate total steps from num_epochs, batch_size, and
    gradient_accumulation_steps, then multiply by warmup_ratio.
    """
    # Explicit warmup_steps takes priority
    if "warmup_steps" in cfg and cfg["warmup_steps"] > 0:
        return int(cfg["warmup_steps"])

    warmup_ratio = cfg.get("warmup_ratio", 0.1)
    # Estimate total training steps (rough — Trainer will use exact value internally)
    num_epochs = cfg.get("num_train_epochs", 5)
    batch_size = cfg.get("per_device_train_batch_size", 16)
    grad_accum = cfg.get("gradient_accumulation_steps", 2)
    # Use train_dataset_size if available, otherwise fall back to a reasonable default
    train_size = cfg.get("train_dataset_size", 2_200_000)
    steps_per_epoch = max(1, train_size // (batch_size * grad_accum))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    logger.info(f"warmup_ratio={warmup_ratio} → warmup_steps={warmup_steps} "
                f"(estimated total_steps={total_steps})")
    return warmup_steps


def build_training_args(cfg: dict, output_dir: Path) -> TrainingArguments:
    """
    Build TrainingArguments from config dict.

    Automatically disables fp16 when CUDA is unavailable.
    """
    use_fp16 = cfg.get("fp16", True) and torch.cuda.is_available()
    if cfg.get("fp16", True) and not torch.cuda.is_available():
        logger.warning("fp16=True in config but CUDA not available. Disabling fp16.")

    warmup_steps = _compute_warmup_steps(cfg)

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.get("num_train_epochs", 5),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 2),
        learning_rate=cfg.get("learning_rate", 2e-4),
        warmup_steps=warmup_steps,
        weight_decay=cfg.get("weight_decay", 0.01),
        logging_steps=cfg.get("logging_steps", 50),
        eval_strategy=cfg.get("evaluation_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 200),
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg.get("greater_is_better", False),
        fp16=use_fp16,
        bf16=cfg.get("bf16", False),
        seed=cfg.get("seed", 42),
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=True,
    )


def run_training(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_cfg: dict,
    output_dir: Path,
    resume_from_checkpoint: Optional[str] = None,
) -> None:
    """
    Run the full MLM training loop.

    Args:
        model: PEFT-wrapped BertForMaskedLM.
        tokenizer: Loaded tokenizer.
        train_dataset: HuggingFace Dataset (tokenised).
        eval_dataset: HuggingFace Dataset (tokenised).
        training_cfg: Dict from training_config.yaml.
        output_dir: Directory for checkpoints.
        resume_from_checkpoint: Path to checkpoint dir or None.
    """
    set_seed(training_cfg.get("seed", 42))

    max_length = training_cfg.get("max_seq_length", 128)

    # Tokenise
    logger.info("Tokenising training split...")
    train_tok = tokenize_dataset(train_dataset, tokenizer, max_length)
    logger.info("Tokenising validation split...")
    eval_tok = tokenize_dataset(eval_dataset, tokenizer, max_length)

    # MLM Data Collator (randomly masks 15% of tokens)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=training_cfg.get("mlm_probability", 0.15),
    )

    training_args = build_training_args(training_cfg, output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        callbacks=[LoggingCallback(), BestModelCallback()],
    )

    logger.info("Starting MLM training with LoRA...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Compute final perplexity
    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results["eval_loss"])
    logger.info(f"Final eval loss: {eval_results['eval_loss']:.4f} | Perplexity: {perplexity:.2f}")


