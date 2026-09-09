#!/usr/bin/env python3
"""
scripts/evaluate_mlm.py
Step 9: Evaluate MLM perplexity and report parameter counts.

Evaluates:
  1. MLM loss & perplexity on test set
  2. Trainable vs total parameter count
  3. Optionally compares base model vs LoRA-adapted model

Usage:
    python scripts/evaluate_mlm.py
    python scripts/evaluate_mlm.py --config configs/model_config.yaml --split test
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
    parser.add_argument("--config",   default="configs/model_config.yaml")
    parser.add_argument("--training", default="configs/training_config.yaml")
    parser.add_argument("--data",     default="configs/data_config.yaml")
    parser.add_argument("--split",    default="test", choices=["train", "validation", "test"])
    parser.add_argument("--adapter",  default=None, help="Path to LoRA adapter dir. If None, evaluates base only.")
    args = parser.parse_args()

    with open(Path(args.config), encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)["model"]
    with open(Path(args.training), encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)["training"]
    with open(Path(args.data), encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)["data"]

    split_path = Path(data_cfg["processed_dir"]) / f"{args.split}.jsonl"
    if not split_path.exists():
        logger.error(f"Split file not found: {split_path}")
        sys.exit(1)

    dataset = load_jsonl_dataset(split_path)
    logger.info(f"Evaluating on {args.split} split: {len(dataset):,} samples")

    max_length = train_cfg.get("max_seq_length", 128)
    model_path = model_cfg["base_model_path"]

    # ── Evaluate base model ──────────────────────────────────────────────────
    base_model, tokenizer = load_base_model_and_tokenizer(model_path)
    base_results = evaluate_model(base_model, tokenizer, dataset, max_length, label="Base Model")

    # ── Evaluate LoRA model (optional) ──────────────────────────────────────
    adapter_dir = args.adapter or str(Path(model_cfg["adapter_output_dir"]))
    if Path(adapter_dir).exists():
        logger.info(f"\nLoading LoRA adapter from: {adapter_dir}")
        # Reload base (clean state) then attach adapter
        lora_base, _ = load_base_model_and_tokenizer(model_path)
        lora_model = load_lora_adapter(lora_base, adapter_dir)
        lora_results = evaluate_model(lora_model, tokenizer, dataset, max_length, label="Base + LoRA")

        # Comparison
        logger.info("\n" + "=" * 50)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 50)
        logger.info(f"  {'Metric':<25} {'Base Model':>12} {'Base + LoRA':>12}")
        logger.info(f"  {'-'*50}")
        logger.info(f"  {'MLM Loss':<25} {base_results['loss']:>12.4f} {lora_results['loss']:>12.4f}")
        logger.info(f"  {'Perplexity':<25} {base_results['perplexity']:>12.2f} {lora_results['perplexity']:>12.2f}")
        logger.info(f"  {'Trainable %':<25} {100*base_results['trainable']/base_results['total']:>11.2f}% {100*lora_results['trainable']/lora_results['total']:>11.2f}%")
    else:
        logger.info(f"\nNo adapter found at {adapter_dir} — evaluating base model only.")


if __name__ == "__main__":
    main()
