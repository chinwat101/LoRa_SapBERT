#!/usr/bin/env python3
"""
scripts/train_lora_mlm.py
Step 7: Train BertForMaskedLM with LoRA on the Thai Lexitron corpus.

Features:
  - Loads base model as read-only
  - Applies PEFT LoRA adapter
  - Trains with DataCollatorForLanguageModeling (MLM)
  - Saves only LoRA adapter (not full model)
  - Supports resume from checkpoint
  - Auto-disables fp16 when CUDA unavailable

Usage:
    python scripts/train_lora_mlm.py
    python scripts/train_lora_mlm.py --resume outputs/checkpoints/checkpoint-400
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from datasets import Dataset

from src.models.model_loader import load_base_model_and_tokenizer
from src.models.lora import apply_lora, save_lora_adapter
from src.training.trainer import run_training
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__, log_dir=Path("outputs/logs"))


def load_jsonl_dataset(path: Path) -> Dataset:
    """Load a JSONL file as a HuggingFace Dataset."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA MLM on Thai Lexitron")
    parser.add_argument("--model-config",    default="configs/model_config.yaml")
    parser.add_argument("--lora-config",     default="configs/lora_config.yaml")
    parser.add_argument("--training-config", default="configs/training_config.yaml")
    parser.add_argument("--data-config",     default="configs/data_config.yaml")
    parser.add_argument("--resume",          default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # ── Load configs ─────────────────────────────────────────────────────────
    with open(Path(args.model_config), encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)["model"]
    with open(Path(args.lora_config), encoding="utf-8") as f:
        lora_cfg = yaml.safe_load(f)["lora"]
    with open(Path(args.training_config), encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)["training"]
    with open(Path(args.data_config), encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)["data"]

    # Allow CLI resume to override config
    resume = args.resume or train_cfg.get("resume_from_checkpoint")

    set_seed(train_cfg.get("seed", 42))

    # ── Load data ─────────────────────────────────────────────────────────────
    processed_dir = Path(data_cfg["processed_dir"])
    train_path = processed_dir / "train.jsonl"
    val_path   = processed_dir / "validation.jsonl"

    for p in [train_path, val_path]:
        if not p.exists():
            logger.error(
                f"Dataset file not found: {p}\n"
                "Run: python scripts/build_mlm_dataset.py first."
            )
            sys.exit(1)

    logger.info("Loading datasets...")
    train_dataset = load_jsonl_dataset(train_path)
    eval_dataset  = load_jsonl_dataset(val_path)
    logger.info(f"Train: {len(train_dataset):,} | Val: {len(eval_dataset):,}")

    # ── Load base model (read-only source) ───────────────────────────────────
    model_path = model_cfg["base_model_path"]
    logger.info(f"Loading base model from: {model_path}")
    model, tokenizer = load_base_model_and_tokenizer(model_path)

    # ── Apply LoRA ───────────────────────────────────────────────────────────
    logger.info("Applying LoRA adapters...")
    model = apply_lora(model, lora_cfg)

    # ── Train ─────────────────────────────────────────────────────────────────
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    run_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_cfg=train_cfg,
        output_dir=output_dir,
        resume_from_checkpoint=resume,
    )

    # ── Save adapter ──────────────────────────────────────────────────────────
    adapter_dir = Path(model_cfg["adapter_output_dir"])
    save_lora_adapter(model, adapter_dir)

    # Also save tokenizer alongside adapter for convenience
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info(f"Tokenizer saved alongside adapter: {adapter_dir}")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  LoRA adapter : {adapter_dir}")
    logger.info(f"  Checkpoints  : {output_dir}")
    logger.info(
        "  To merge LoRA into base model: "
        "python scripts/merge_lora.py"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
