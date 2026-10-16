#!/usr/bin/env python
"""Train an isolated SpERT-style model from the project's JSONL annotations."""

from __future__ import annotations

import argparse
import json
import logging
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.spert.dataset import SpertJsonlDataset, spert_collate_fn
from src.spert.model import SpERTModel
from src.spert.trainer import SpERTTrainer, linear_warmup_scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SpERT on the configured JSONL dataset")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "spert_config.yaml"),
        help="Path to SpERT YAML configuration",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    training = config["training"]
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("train_spert")
    logger.info("Device: %s", device)
    logger.info("Base model: %s", config["model_path"])
    logger.info("Dataset: %s", config["dataset_path"])

    tokenizer = AutoTokenizer.from_pretrained(
        config.get("tokenizer_path", config["model_path"]),
        use_fast=True,
    )
    dataset = SpertJsonlDataset(
        data_path=config["dataset_path"],
        tokenizer=tokenizer,
        entity_labels=config["entity_labels"],
        relation_labels=config["relation_labels"],
        max_seq_length=int(config["data"]["max_seq_length"]),
        max_span_size=int(config["data"]["max_span_size"]),
        neg_entity_count=int(config["data"]["neg_entity_count"]),
        neg_relation_count=int(config["data"]["neg_relation_count"]),
        seed=int(config.get("seed", 42)),
    )
    alignment_stats = dataset.alignment_stats
    logger.info("Entity alignment summary: %s", alignment_stats)
    if alignment_stats["token_span_label_collisions"]:
        logger.warning(
            "%d conflicting annotations collapsed to the same token span; "
            "the longer mention was retained and relations to discarded mentions were skipped.",
            alignment_stats["token_span_label_collisions"],
        )

    validation_fraction = float(config["data"]["validation_split"])
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("data.validation_split must be between 0 and 1.")
    validation_size = max(1, round(len(dataset) * validation_fraction))
    if validation_size >= len(dataset):
        raise ValueError("Dataset must contain at least two examples for train/validation split.")
    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    train_dataset, validation_dataset = random_split(
        dataset, [train_size, validation_size], generator=generator
    )
    collate = partial(spert_collate_fn, pad_token_id=tokenizer.pad_token_id or 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        num_workers=int(config["data"].get("num_workers", 0)),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        collate_fn=collate,
        num_workers=int(config["data"].get("num_workers", 0)),
    )
    logger.info(
        "Loaded %d examples: train=%d, validation=%d",
        len(dataset), train_size, validation_size,
    )

    model = SpERTModel(
        model_path=config["model_path"],
        entity_label_count=len(config["entity_labels"]),
        relation_label_count=len(config["relation_labels"]),
        max_span_size=int(config["data"]["max_span_size"]),
        size_embedding_dim=int(config["model"].get("size_embedding_dim", 25)),
        dropout=float(config["model"].get("dropout", 0.1)),
        freeze_encoder=bool(config["model"].get("freeze_encoder", False)),
        relation_chunk_size=int(config["model"].get("relation_chunk_size", 100)),
    )
    if device.type == "cuda" and bool(config["model"].get("gradient_checkpointing", True)):
        if hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable()
            logger.info("Encoder gradient checkpointing: enabled")
        else:
            logger.warning("Encoder does not support gradient checkpointing")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{trainable:,}")

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    total_steps = max(1, len(train_loader) * int(training["epochs"]))
    warmup_steps = int(total_steps * float(training.get("warmup_ratio", 0.1)))
    scheduler = linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    trainer = SpERTTrainer(
        model=model,
        entity_labels=config["entity_labels"],
        relation_labels=config["relation_labels"],
        device=device,
        output_dir=config["output_dir"],
        lambda_entity=float(training.get("lambda_entity", 1.0)),
        lambda_relation=float(training.get("lambda_relation", 1.0)),
        relation_threshold=float(config["evaluation"].get("relation_threshold", 0.4)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        use_amp=bool(training.get("mixed_precision", True)),
    )
    logger.info("CUDA mixed precision: %s", trainer.use_amp)

    history: list[dict] = []
    best_joint_f1 = -1.0
    patience_count = 0
    patience = int(training.get("early_stopping_patience", 10))
    for epoch in range(1, int(training["epochs"]) + 1):
        train_metrics = trainer.train_epoch(train_loader, optimizer, scheduler)
        validation_metrics = trainer.evaluate(validation_loader)
        row = {"epoch": epoch, **train_metrics, **validation_metrics}
        history.append(row)
        logger.info(
            "Epoch %d/%d | train loss %.4f (entity %.4f, relation %.4f) | "
            "val entity F1 %.4f | val relation F1 %.4f | joint F1 %.4f",
            epoch,
            int(training["epochs"]),
            train_metrics["loss"],
            train_metrics["entity_loss"],
            train_metrics["relation_loss"],
            validation_metrics["val_entity_f1"],
            validation_metrics["val_relation_f1"],
            validation_metrics["val_joint_f1"],
        )

        if validation_metrics["val_joint_f1"] > best_joint_f1:
            best_joint_f1 = validation_metrics["val_joint_f1"]
            patience_count = 0
            trainer.save_checkpoint(tokenizer, config, validation_metrics)
            logger.info("Saved best checkpoint to %s", config["output_dir"])
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info("Early stopping after %d epochs without improvement", patience)
                break

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, ensure_ascii=False, indent=2)
    with (output_dir / "run_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    logger.info("Training complete. Best validation joint F1: %.4f", best_joint_f1)


if __name__ == "__main__":
    main()
