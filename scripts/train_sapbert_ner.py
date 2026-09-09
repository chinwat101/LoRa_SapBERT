#!/usr/bin/env python3
"""
scripts/train_sapbert_ner.py
Fine-tune merged SapBERT model for NER (token classification) with LoRA.

Task  : Token Classification — DRUG / DISEASE / SYMPTOM (BIO scheme, 7 labels)
Model : Latest merged SapBERT (outputs/merged_model_N)
Data  : data/processed/ner_bio_th.jsonl

Pipeline:
  1. Automatically locate the latest merged SapBERT model (outputs/merged_model_N)
  2. Load tokenizer and model as AutoModelForTokenClassification
  3. Apply LoRA adapter (PEFT)
  4. Train with seqeval F1 metric (entity-level)
  5. Merge LoRA adapter into base model and save to outputs/merged_model_N_ner_M (auto-incrementing)

Usage:
    python scripts/train_sapbert_ner.py
    python scripts/train_sapbert_ner.py --config configs/sapbert_ner_config.yaml
    python scripts/train_sapbert_ner.py --base-model outputs/merged_model_1
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__, log_dir=Path("outputs/logs"))

DEFAULT_LABEL2ID = {
    "O": 0, "B-DRUG": 1, "I-DRUG": 2,
    "B-DISEASE": 3, "I-DISEASE": 4,
    "B-SYMPTOM": 5, "I-SYMPTOM": 6,
}
DEFAULT_ID2LABEL = {v: k for k, v in DEFAULT_LABEL2ID.items()}


def find_latest_merged_model(output_root: str = "outputs") -> Path | None:
    """Find the latest outputs/merged_model_N directory based on the highest integer N."""
    base = Path(output_root)
    candidates = sorted(
        [p for p in base.glob("merged_model_*") if p.is_dir() and (p / "config.json").exists()],
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
    )
    if candidates:
        return candidates[-1]
    
    if (base / "merged_model" / "config.json").exists():
        return base / "merged_model"
        
    fallback = Path("data/interim/extended_base")
    if (fallback / "config.json").exists():
        return fallback
        
    return None


def get_next_ner_dirs(base_model_dir: Path, output_root: str = "outputs") -> tuple[Path, Path, Path]:
    """
    Generate auto-incrementing directory paths for adapter, merged model, and checkpoints.
    Example:
        If base_model_dir is 'outputs/merged_model_1'
        Returns:
            adapter_dir:    'outputs/adapters/merged_model_1_ner_1'
            merged_dir:     'outputs/merged_model_1_ner_1'
            checkpoint_dir: 'outputs/checkpoints/merged_model_1_ner_1'
    """
    root = Path(output_root)
    stem = base_model_dir.name  # e.g. "merged_model_1"
    
    run_num = 1
    while True:
        merged_dir = root / f"{stem}_ner_{run_num}"
        if not merged_dir.exists():
            break
        run_num += 1

    adapter_dir = root / "adapters" / f"{stem}_ner_{run_num}"
    checkpoint_dir = root / "checkpoints" / f"{stem}_ner_{run_num}"
    return adapter_dir, merged_dir, checkpoint_dir


def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def align_labels_to_subwords(
    example: dict,
    tokenizer,
    label2id: dict,
    max_length: int = 128,
) -> dict:
    text  = example["text"]
    spans = example.get("entities", [])

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )

    char_label: dict = {}
    span_starts: set = set()
    for span in spans:
        span_starts.add(span["start"])
        for i in range(span["start"], span["end"]):
            char_label.setdefault(i, span["label"])

    labels = []
    for cs, ce in enc["offset_mapping"]:
        if cs == 0 and ce == 0:  # special token
            labels.append(-100)
            continue
        tag_id = label2id["O"]
        for ci in range(cs, ce):
            if ci in char_label:
                lbl = char_label[ci]
                prefix = "B-" if ci in span_starts else "I-"
                tag_id = label2id.get(f"{prefix}{lbl}", label2id["O"])
                break
        labels.append(tag_id)

    return {
        "input_ids":      enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels":         labels,
    }


def build_hf_dataset(
    records: list,
    tokenizer,
    label2id: dict,
    max_length: int,
) -> Dataset:
    aligned = [
        align_labels_to_subwords(r, tokenizer, label2id, max_length)
        for r in records
    ]
    return Dataset.from_list(aligned)


def make_compute_metrics(id2label: dict):
    try:
        from seqeval.metrics import f1_score, precision_score, recall_score
        USE_SEQEVAL = True
    except ImportError:
        USE_SEQEVAL = False
        logger.warning("seqeval not installed. F1 metric will be token-level.")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_seqs, pred_seqs = [], []
        for pred_row, label_row in zip(preds, labels):
            true_seq, pred_seq = [], []
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                true_seq.append(id2label.get(l, "O"))
                pred_seq.append(id2label.get(p, "O"))
            true_seqs.append(true_seq)
            pred_seqs.append(pred_seq)

        if USE_SEQEVAL:
            return {
                "eval_f1":        f1_score(true_seqs, pred_seqs),
                "eval_precision": precision_score(true_seqs, pred_seqs),
                "eval_recall":    recall_score(true_seqs, pred_seqs),
            }
        else:
            correct = total = 0
            for ts, ps in zip(true_seqs, pred_seqs):
                for t, p in zip(ts, ps):
                    if t != "O":
                        total += 1
                        correct += t == p
            f1 = correct / total if total else 0.0
            return {"eval_f1": f1}

    return compute_metrics


def load_ner_model(model_path: str, label2id: dict, id2label: dict, lora_cfg: dict):
    """Load AutoModelForTokenClassification and wrap with LoRA."""
    logger.info(f"Loading base model for NER: {model_path}")
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.1),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get("lora_target_modules", ["query", "key", "value", "dense"]),
        inference_mode=False,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def _compute_warmup_steps(cfg: dict, train_dataset_len: int) -> int:
    if "warmup_steps" in cfg and cfg["warmup_steps"] > 0:
        return int(cfg["warmup_steps"])
    warmup_ratio = cfg.get("warmup_ratio", 0.1)
    num_epochs = cfg.get("num_train_epochs", 30)
    batch_size = cfg.get("per_device_train_batch_size", 8)
    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    steps_per_epoch = max(1, train_dataset_len // (batch_size * grad_accum))
    total_steps = steps_per_epoch * num_epochs
    return int(total_steps * warmup_ratio)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SapBERT for NER")
    parser.add_argument("--config", default="configs/sapbert_ner_config.yaml")
    parser.add_argument("--base-model", default=None, help="Path to base model (overrides config/autodetect)")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg    = cfg["model"]
    data_cfg     = cfg["data"]
    lora_cfg     = {**cfg.get("lora", {}), **model_cfg}
    training_cfg = cfg["training"]

    set_seed(training_cfg.get("seed", 42))

    # 1. Resolve base model path (autodetect latest merged SapBERT if not specified)
    output_root = model_cfg.get("output_root", "outputs")
    if args.base_model:
        model_path = Path(args.base_model)
    elif model_cfg.get("base_model_path"):
        model_path = Path(model_cfg["base_model_path"])
    else:
        model_path = find_latest_merged_model(output_root)

    if not model_path or not model_path.exists():
        logger.error("Could not find a valid base model directory!")
        logger.error("Please run merge_lora.py first or specify --base-model <path>")
        sys.exit(1)

    logger.info(f"Target Base Model: {model_path}")

    # 2. Determine auto-incremented NER output paths
    adapter_dir, merged_dir, checkpoint_dir = get_next_ner_dirs(model_path, output_root)
    logger.info(f"NER Output Target: {merged_dir}")
    logger.info(f"Adapter Output Target: {adapter_dir}")

    # 3. Label schema
    schema_path = Path(data_cfg.get("label_schema_file", "data/processed/ner_label_schema.json"))
    if schema_path.exists():
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        label2id = schema["label2id"]
        id2label = {int(k): v for k, v in schema["id2label"].items()}
    else:
        logger.warning("Label schema not found, using default.")
        label2id = DEFAULT_LABEL2ID
        id2label = DEFAULT_ID2LABEL

    logger.info(f"Labels ({len(label2id)}): {list(label2id.keys())}")

    # 4. Tokenizer
    logger.info(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # 5. Load & split data
    train_file = Path(data_cfg["train_file"])
    if not train_file.exists():
        logger.error(f"Train file not found: {train_file}")
        logger.error("Run: python scripts/convert_ner_dataset.py first.")
        sys.exit(1)

    records = load_jsonl(train_file)
    logger.info(f"Loaded {len(records)} examples from {train_file}")

    val_file = data_cfg.get("val_file")
    if val_file and Path(val_file).exists():
        val_records = load_jsonl(Path(val_file))
        train_records = records
    else:
        val_frac = data_cfg.get("val_split", 0.2)
        split = int(len(records) * (1 - val_frac))
        import random
        random.seed(training_cfg.get("seed", 42))
        random.shuffle(records)
        train_records = records[:split]
        val_records   = records[split:]
        logger.info(f"Auto-split: train={len(train_records)}, val={len(val_records)}")

    max_len = training_cfg.get("max_seq_length", 128)
    train_dataset = build_hf_dataset(train_records, tokenizer, label2id, max_len)
    eval_dataset  = build_hf_dataset(val_records,   tokenizer, label2id, max_len)
    logger.info(f"Train tokens: {len(train_dataset)} | Val tokens: {len(eval_dataset)}")

    # 6. Model + LoRA
    model = load_ner_model(str(model_path), label2id, id2label, lora_cfg)

    # 7. Training args
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    use_fp16 = training_cfg.get("fp16", True) and torch.cuda.is_available()
    warmup_steps = _compute_warmup_steps(training_cfg, len(train_dataset))

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=training_cfg.get("num_train_epochs", 30),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=training_cfg.get("learning_rate", 3e-4),
        warmup_steps=warmup_steps,
        weight_decay=training_cfg.get("weight_decay", 0.01),
        logging_steps=training_cfg.get("logging_steps", 5),
        eval_strategy="steps",
        eval_steps=training_cfg.get("eval_steps", 10),
        save_strategy="steps",
        save_steps=training_cfg.get("save_steps", 10),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=training_cfg.get("metric_for_best_model", "eval_f1"),
        greater_is_better=training_cfg.get("greater_is_better", True),
        fp16=use_fp16,
        bf16=training_cfg.get("bf16", False),
        seed=training_cfg.get("seed", 42),
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if use_fp16 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(id2label),
    )

    # 8. Train
    resume = args.resume or training_cfg.get("resume_from_checkpoint")
    logger.info("Starting SapBERT NER fine-tuning with LoRA...")
    trainer.train(resume_from_checkpoint=resume)

    # 9. Evaluation
    eval_results = trainer.evaluate()
    logger.info(f"Final eval — F1: {eval_results.get('eval_f1', 0):.4f} | "
                f"Loss: {eval_results.get('eval_loss', 0):.4f}")

    # 10. Save LoRA adapter
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    
    schema_dict = {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}
    with open(adapter_dir / "label_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"LoRA adapter saved: {adapter_dir}")

    # 11. Merge Adapter into Base Model (Auto-incremented output directory: outputs/<stem>_ner_<N>)
    logger.info(f"Merging LoRA adapter into base model → {merged_dir}")
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    with open(merged_dir / "label_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"Merged NER model saved to: {merged_dir}")

    logger.info("\n" + "=" * 60)
    logger.info("SAPBERT NER TRAINING COMPLETE")
    logger.info(f"  Base Model   : {model_path}")
    logger.info(f"  Merged Model : {merged_dir}")
    logger.info(f"  Adapter      : {adapter_dir}")
    logger.info(f"  Checkpoints  : {checkpoint_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
