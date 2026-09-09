#!/usr/bin/env python3
"""
scripts/train_wangchan_mlm.py
Continued pre-training of WangchanBERTa with MLM (Masked Language Modeling) + LoRA.

Task  : Masked Language Modeling on Thai domain corpus
Model : airesearch/wangchanberta-base-att-spm-uncased (CamembertForMaskedLM)
Data  : data/processed/mlm_train.jsonl + mlm_val.jsonl
        (plain JSONL: each line {"text": "..."} )

        To build MLM corpus from NER data, run:
            python scripts/build_wangchan_mlm_corpus.py

Pipeline:
  1. Load WangchanBERTa as CamembertForMaskedLM
  2. Apply LoRA adapter (PEFT) — task_type=FEATURE_EXTRACTION
  3. Train with DataCollatorForLanguageModeling (15% random masking)
  4. Report perplexity after each evaluation
  5. Save best LoRA adapter + tokenizer

Usage:
    python scripts/train_wangchan_mlm.py
    python scripts/train_wangchan_mlm.py --config configs/wangchan_mlm_config.yaml
    python scripts/train_wangchan_mlm.py --resume outputs/wangchan_mlm/checkpoints/checkpoint-500
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    CamembertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__, log_dir=Path("outputs/wangchan_mlm/logs"))


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
def load_jsonl_texts(path: Path) -> Dataset:
    """Load a JSONL file of {\"text\": str} records into a HuggingFace Dataset."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Support both {"text": "..."} and NER-format {"text": "...", ...}
            if "text" in obj:
                records.append({"text": obj["text"]})
    logger.info(f"  Loaded {len(records):,} text records from {path}")
    return Dataset.from_list(records)


def tokenize_for_mlm(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    """Tokenise text for MLM (no labels — DataCollator handles masking)."""
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_special_tokens_mask=True,
        )

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenising",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Metric: perplexity
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics_mlm(eval_pred):
    """Return perplexity from eval_loss (Trainer computes loss automatically)."""
    # eval_pred is (logits, labels); loss is passed separately via eval_results
    # We return an empty dict here — perplexity is computed from eval_loss below.
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Model + LoRA
# ──────────────────────────────────────────────────────────────────────────────
def load_mlm_model(model_path: str, lora_cfg: dict):
    """Load CamembertForMaskedLM and wrap with LoRA."""
    logger.info(f"Loading base model: {model_path}")
    model = CamembertForMaskedLM.from_pretrained(model_path)

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("lora_alpha", 64),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get("lora_target_modules",
                                     ["query", "key", "value", "dense"]),
        inference_mode=False,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continued MLM pre-training of WangchanBERTa with LoRA"
    )
    parser.add_argument("--config", default="configs/wangchan_mlm_config.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume")
    args = parser.parse_args()

    # ─ Load config ─
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg    = cfg["model"]
    data_cfg     = cfg["data"]
    lora_cfg     = {**cfg.get("lora", {}), **model_cfg}
    training_cfg = cfg["training"]

    set_seed(training_cfg.get("seed", 42))

    # ─ Tokenizer ─
    model_path = model_cfg["base_model_path"]
    logger.info(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # ─ Load data ─
    train_path = Path(data_cfg["train_file"])
    val_path   = Path(data_cfg["val_file"])

    for p in [train_path, val_path]:
        if not p.exists():
            logger.error(
                f"MLM corpus not found: {p}\n"
                "Run: python scripts/build_wangchan_mlm_corpus.py"
            )
            sys.exit(1)

    logger.info("Loading MLM corpus...")
    max_len = training_cfg.get("max_seq_length", 128)

    train_raw = load_jsonl_texts(train_path)
    val_raw   = load_jsonl_texts(val_path)

    logger.info("Tokenising...")
    train_dataset = tokenize_for_mlm(train_raw, tokenizer, max_len)
    eval_dataset  = tokenize_for_mlm(val_raw,   tokenizer, max_len)
    logger.info(f"Train: {len(train_dataset):,} | Val: {len(eval_dataset):,}")

    # ─ Model + LoRA ─
    model = load_mlm_model(model_path, lora_cfg)

    # ─ Training arguments ─
    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = training_cfg.get("fp16", True) and torch.cuda.is_available()
    if training_cfg.get("fp16", True) and not torch.cuda.is_available():
        logger.warning("fp16=True but CUDA unavailable. Disabling fp16.")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 2),
        learning_rate=training_cfg.get("learning_rate", 2e-4),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        logging_dir=str(training_cfg.get("logging_dir", "outputs/wangchan_mlm/logs")),
        logging_steps=training_cfg.get("logging_steps", 100),
        eval_strategy="steps",
        eval_steps=training_cfg.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=training_cfg.get("save_steps", 500),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_fp16,
        bf16=training_cfg.get("bf16", False),
        seed=training_cfg.get("seed", 42),
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=True,
    )

    # DataCollator: randomly masks mlm_probability fraction of tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=training_cfg.get("mlm_probability", 0.15),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # No compute_metrics — perplexity computed from eval_loss below
    )

    # ─ Train ─
    resume = args.resume or training_cfg.get("resume_from_checkpoint")
    logger.info("Starting WangchanBERTa MLM continued pre-training...")
    trainer.train(resume_from_checkpoint=resume)

    # ─ Final eval + perplexity ─
    eval_results = trainer.evaluate()
    ppl = math.exp(eval_results["eval_loss"])
    logger.info(
        f"Final eval — Loss: {eval_results['eval_loss']:.4f} | Perplexity: {ppl:.2f}"
    )

    # ─ Save adapter ─
    adapter_dir = Path(model_cfg["adapter_output_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info(f"LoRA adapter saved: {adapter_dir}")

    logger.info("\n" + "=" * 60)
    logger.info("MLM PRE-TRAINING COMPLETE")
    logger.info(f"  Adapter     : {adapter_dir}")
    logger.info(f"  Checkpoints : {output_dir}")
    logger.info(f"  Perplexity  : {ppl:.2f}")
    logger.info(f"  To merge LoRA: python scripts/merge_lora.py "
                f"--base {model_path} --adapter {adapter_dir} "
                f"--output {model_cfg.get('merged_model_dir', 'outputs/wangchan_mlm/merged')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
