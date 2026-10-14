#!/usr/bin/env python3
"""
scripts/train_lora_mlm_merge.py
Train BertForMaskedLM with LoRA — then automatically merge the adapter
into the base model and save a ready-to-use merged model.

Features:
  - Loads any base model (extended_base OR a previous merged_model)
  - Applies PEFT LoRA adapter
  - Trains with DataCollatorForLanguageModeling (MLM)
  - After training: auto-merges LoRA into base and saves merged model
  - Supports resume from checkpoint
  - Auto-disables fp16 when CUDA unavailable

Usage:
    # Default base (data/interim/extended_base)
    python scripts/train_lora_mlm_merge.py \\
        --train-data data/processed/medical_train.jsonl \\
        --val-data   data/processed/medical_val.jsonl

    # Continue training from a merged model
    python scripts/train_lora_mlm_merge.py \\
        --base-model outputs/merged_model_3 \\
        --train-data data/processed/qa_train.jsonl \\
        --val-data   data/processed/qa_val.jsonl

    # Resume interrupted training
    python scripts/train_lora_mlm_merge.py \\
        --base-model outputs/merged_model_3 \\
        --train-data data/processed/qa_train.jsonl \\
        --val-data   data/processed/qa_val.jsonl \\
        --resume outputs/checkpoints/checkpoint-1000
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from datasets import Dataset

from src.models.model_loader import load_base_model_and_tokenizer
from src.models.lora import apply_lora, save_lora_adapter, load_lora_adapter
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


def find_next_merged_dir(base_merged_dir: Path) -> Path:
    """Auto-increment output directory (e.g. merged_model_1 → merged_model_2)."""
    run_num = 1
    while True:
        candidate = base_merged_dir.with_name(f"{base_merged_dir.name}_{run_num}")
        if not candidate.exists():
            return candidate
        run_num += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LoRA MLM and auto-merge into base model"
    )
    parser.add_argument("--model-config",    default="configs/model_config.yaml")
    parser.add_argument("--lora-config",     default="configs/lora_config.yaml")
    parser.add_argument("--training-config", default="configs/training_config.yaml")
    parser.add_argument("--data-config",     default="configs/data_config.yaml")
    parser.add_argument("--resume",          default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument(
        "--train-data", default=None,
        help="Override training JSONL path (default: processed_dir/train.jsonl from data config)",
    )
    parser.add_argument(
        "--val-data", default=None,
        help="Override validation JSONL path (default: processed_dir/validation.jsonl from data config)",
    )
    parser.add_argument(
        "--base-model", default=None,
        help="Override base model path (e.g. outputs/merged_model_3). "
             "Default: base_model_path from model_config.yaml",
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Skip the auto-merge step (save adapter only, like train_lora_mlm.py)",
    )
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
    train_path = Path(args.train_data) if args.train_data else processed_dir / "train.jsonl"
    val_path   = Path(args.val_data)   if args.val_data   else processed_dir / "validation.jsonl"

    for p in [train_path, val_path]:
        if not p.exists():
            logger.error(
                f"Dataset file not found: {p}\n"
                "Run: python scripts/build_mlm_dataset.py   (for Lexitron/Wiki)\n"
                "     python scripts/build_medical_corpus.py (for medical data)\n"
                "     python scripts/build_qa_corpus.py      (for QA data)"
            )
            sys.exit(1)

    logger.info(f"Train data : {train_path}")
    logger.info(f"Val data   : {val_path}")
    train_dataset = load_jsonl_dataset(train_path)
    eval_dataset  = load_jsonl_dataset(val_path)
    logger.info(f"Train: {len(train_dataset):,} | Val: {len(eval_dataset):,}")

    # ── Load base model ───────────────────────────────────────────────────────
    model_path = args.base_model or model_cfg["base_model_path"]
    logger.info(f"Loading base model from: {model_path}")
    model, tokenizer = load_base_model_and_tokenizer(model_path)

    # ── Apply LoRA ───────────────────────────────────────────────────────────
    logger.info("Applying LoRA adapters...")
    model = apply_lora(model, lora_cfg)

    # ── Train ─────────────────────────────────────────────────────────────────
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = run_training(
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
    save_lora_adapter(trainer.model, adapter_dir)
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info(f"Tokenizer saved alongside adapter: {adapter_dir}")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  LoRA adapter : {adapter_dir}")
    logger.info(f"  Checkpoints  : {output_dir}")
    logger.info("=" * 60)

    # ── Auto-merge ────────────────────────────────────────────────────────────
    if args.no_merge:
        logger.info("Skipping merge (--no-merge flag set).")
        logger.info("To merge manually: python scripts/merge_lora.py")
        return

    logger.info("\n" + "=" * 60)
    logger.info("AUTO-MERGE: Merging LoRA into base model...")
    logger.info("=" * 60)

    # Reload clean base (no LoRA) then attach saved adapter
    logger.info(f"Reloading clean base model from: {model_path}")
    clean_base, tokenizer = load_base_model_and_tokenizer(model_path)

    logger.info(f"Loading adapter from: {adapter_dir}")
    lora_model = load_lora_adapter(clean_base, str(adapter_dir))

    logger.info("Merging weights...")
    merged_model = lora_model.merge_and_unload()

    # Auto-increment output directory
    base_merged_dir = Path(model_cfg["merged_model_dir"])
    merged_dir = find_next_merged_dir(base_merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    logger.info("\n" + "=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info(f"  Merged model : {merged_dir}")
    logger.info(f"  Base model   : {model_path}  (NOT modified)")
    logger.info("=" * 60)
    logger.info(f"\nTest with: python scripts/test_inference.py --merged {merged_dir}")


if __name__ == "__main__":
    main()
