#!/usr/bin/env python3
"""
scripts/evaluate_mlm.py
Evaluate MLM perplexity and report parameter counts.

Evaluates:
  1. MLM loss & perplexity on test set
  2. Trainable vs total parameter count
  3. Optionally compares base model vs LoRA-adapted model

Usage:
    # Default: base model vs current adapter, on test split
    python scripts/evaluate_mlm.py

    # Choose a specific merged model
    python scripts/evaluate_mlm.py --merged outputs/merged_model_3

    # Evaluate on medical test data instead of wiki/Lexitron test
    python scripts/evaluate_mlm.py --eval-data data/processed/medical_test.jsonl

    # Override base model path
    python scripts/evaluate_mlm.py --model outputs/merged_model_1_1

    # Full custom run
    python scripts/evaluate_mlm.py \\
        --model    data/interim/extended_base \\
        --adapter  outputs/adapters/lexitron_wiki_thai_lora_r32 \\
        --eval-data data/processed/medical_test.jsonl \\
        --split    test
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.models.model_loader import load_base_model_and_tokenizer
from src.models.lora import load_lora_adapter
from src.training.trainer import tokenize_dataset
from src.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("outputs/logs"))


def count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def load_jsonl_dataset(path: Path) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def evaluate_model(model, tokenizer, dataset: Dataset, max_length: int = 128, label: str = "") -> dict:
    """Compute eval loss and perplexity on a dataset."""
    tok_dataset = tokenize_dataset(dataset, tokenizer, max_length)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    args = TrainingArguments(
        output_dir="outputs/eval_tmp",
        per_device_eval_batch_size=16,
        use_cpu=not torch.cuda.is_available(),
        report_to="none",
        seed=42,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=tok_dataset,
        data_collator=data_collator,
    )

    results = trainer.evaluate()
    loss = results.get("eval_loss", float("nan"))
    ppl = math.exp(loss) if not math.isnan(loss) else float("nan")

    trainable, total = count_params(model)

    logger.info(f"\n{'='*50}")
    logger.info(f"EVALUATION RESULTS: {label}")
    logger.info(f"{'='*50}")
    logger.info(f"  Total parameters    : {total:>12,}")
    logger.info(f"  Trainable params    : {trainable:>12,}")
    logger.info(f"  Trainable %         : {100*trainable/total:>11.2f}%")
    logger.info(f"  MLM Loss            : {loss:>11.4f}")
    logger.info(f"  Perplexity          : {ppl:>11.2f}")

    return {"label": label, "loss": loss, "perplexity": ppl, "trainable": trainable, "total": total}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MLM model")
    parser.add_argument("--config",    default="configs/model_config.yaml")
    parser.add_argument("--training",  default="configs/training_config.yaml")
    parser.add_argument("--data",      default="configs/data_config.yaml")
    parser.add_argument("--split",     default="test", choices=["train", "validation", "test"])
    parser.add_argument("--adapter",   default=None,
                        help="Path to LoRA adapter dir. If None, uses adapter_output_dir from config.")
    parser.add_argument("--model",     default=None,
                        help="Override base model path (default: base_model_path from model_config.yaml)")
    parser.add_argument("--merged",    default=None,
                        help="Evaluate a fully merged model directly (skips adapter step)")
    parser.add_argument("--eval-data", default=None,
                        help="Override eval JSONL file path (e.g. data/processed/medical_test.jsonl)")
    args = parser.parse_args()

    with open(Path(args.config), encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)["model"]
    with open(Path(args.training), encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)["training"]
    with open(Path(args.data), encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)["data"]

    # ── Resolve paths ────────────────────────────────────────────────────────
    if args.eval_data:
        split_path = Path(args.eval_data)
    else:
        split_path = Path(data_cfg["processed_dir"]) / f"{args.split}.jsonl"

    if not split_path.exists():
        logger.error(f"Eval data file not found: {split_path}")
        sys.exit(1)

    dataset = load_jsonl_dataset(split_path)
    logger.info(f"Eval data : {split_path} ({len(dataset):,} samples)")

    max_length = train_cfg.get("max_seq_length", 128)
    model_path = args.model or model_cfg["base_model_path"]

    # ── Evaluate base model ──────────────────────────────────────────────────
    base_model, tokenizer = load_base_model_and_tokenizer(model_path)
    base_results = evaluate_model(base_model, tokenizer, dataset, max_length, label=f"Base ({Path(model_path).name})")

    # ── Evaluate fully merged model (if --merged given) ──────────────────────
    if args.merged:
        merged_path = Path(args.merged)
        if merged_path.exists() and (merged_path / "config.json").exists():
            logger.info(f"\nLoading merged model from: {merged_path}")
            merged_model, _ = load_base_model_and_tokenizer(str(merged_path))
            evaluate_model(merged_model, tokenizer, dataset, max_length,
                           label=f"Merged ({merged_path.name})")
        else:
            logger.error(f"Merged model not found: {merged_path}")
        return

    # ── Evaluate LoRA model (optional) ──────────────────────────────────────
    adapter_dir = args.adapter or str(Path(model_cfg["adapter_output_dir"]))
    if Path(adapter_dir).exists():
        logger.info(f"\nLoading LoRA adapter from: {adapter_dir}")
        lora_base, _ = load_base_model_and_tokenizer(model_path)
        lora_model = load_lora_adapter(lora_base, adapter_dir)
        lora_results = evaluate_model(lora_model, tokenizer, dataset, max_length,
                                      label=f"Base + LoRA ({Path(adapter_dir).name})")

        # Comparison
        logger.info("\n" + "=" * 50)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 50)
        logger.info(f"  {'Metric':<25} {'Base Model':>12} {'Base + LoRA':>12}")
        logger.info(f"  {'-'*50}")
        logger.info(f"  {'MLM Loss':<25} {base_results['loss']:>12.4f} {lora_results['loss']:>12.4f}")
        logger.info(f"  {'Perplexity':<25} {base_results['perplexity']:>12.2f} {lora_results['perplexity']:>12.2f}")
        logger.info(f"  {'Trainable %':<25} {100*base_results['trainable']/base_results['total']:>11.2f}% "
                    f"{100*lora_results['trainable']/lora_results['total']:>11.2f}%")
    else:
        logger.info(f"\nNo adapter found at {adapter_dir} — evaluating base model only.")


if __name__ == "__main__":
    main()
