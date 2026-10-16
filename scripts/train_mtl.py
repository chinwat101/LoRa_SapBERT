#!/usr/bin/env python3
"""
==============================================================================
Multi-Task Learning Training Script: NER + Relation Extraction
==============================================================================
ไฟล์นี้สร้างขึ้นเพื่อ:
  เป็น Main Script สำหรับฝึก JointNERAndREModel ด้วยข้อมูลทางการแพทย์ไทย
  อ่าน config จาก configs/mtl_config.yaml
  บันทึก best model ไปยัง outputs/mtl_model/

การใช้งาน:
  python scripts/train_mtl.py
  python scripts/train_mtl.py --config configs/mtl_config.yaml
  python scripts/train_mtl.py --config configs/mtl_config.yaml --epochs 50
==============================================================================
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

# เพิ่ม project root ใน PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mtl_dataset import MTLDataset, mtl_collate_fn
from src.models.mtl_export import export_ner_standalone
from src.models.mtl_model import build_model
from src.training.mtl_trainer import MTLTrainer, get_linear_warmup_scheduler
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("train_mtl")


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Multi-Task Learning (NER + RE) BERT model"
    )
    parser.add_argument(
        "--config", type=str, default="configs/mtl_config.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # --- โหลด Config ---
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Override จาก CLI
    train_cfg = cfg["training"]
    if args.epochs:
        train_cfg["epochs"] = args.epochs
    if args.lr:
        train_cfg["learning_rate"] = args.lr
    if args.batch_size:
        train_cfg["batch_size"] = args.batch_size
    if args.output_dir:
        cfg["model"]["output_dir"] = args.output_dir

    set_seed(cfg["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Label Maps ---
    ner_labels = cfg["ner_labels"]
    rel_labels = cfg["relation_labels"]
    ner2id = {lbl: i for i, lbl in enumerate(ner_labels)}
    rel2id = {lbl: i for i, lbl in enumerate(rel_labels)}
    id2ner = {i: lbl for lbl, i in ner2id.items()}
    id2rel = {i: lbl for lbl, i in rel2id.items()}

    logger.info(f"NER labels ({len(ner_labels)}): {ner_labels}")
    logger.info(f"RE labels ({len(rel_labels)}): {rel_labels}")

    # --- Tokenizer ---
    base_model_path = cfg["model"]["base_model_path"]
    logger.info(f"Loading tokenizer from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # --- Dataset ---
    data_cfg = cfg["data"]
    logger.info(f"Loading dataset: {data_cfg['train_file']}")
    full_dataset = MTLDataset(
        data_path=data_cfg["train_file"],
        tokenizer=tokenizer,
        ner2id=ner2id,
        rel2id=rel2id,
        max_seq_length=data_cfg["max_seq_length"],
        max_neg_pairs=data_cfg["max_neg_pairs"],
        seed=data_cfg["seed"],
    )

    # Train / Val split
    val_size = int(len(full_dataset) * data_cfg["val_split"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(data_cfg["seed"]),
    )
    logger.info(f"Dataset: {len(full_dataset)} total → train={train_size}, val={val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=mtl_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=mtl_collate_fn,
    )

    # --- Model ---
    logger.info(f"Building model from: {base_model_path}")
    model = build_model(
        base_model_path=base_model_path,
        num_ner_labels=len(ner_labels),
        num_rel_labels=len(rel_labels),
        lambda_ner=train_cfg["lambda_ner"],
        lambda_re=train_cfg["lambda_re"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # --- Optimizer & Scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    num_training_steps = len(train_loader) * train_cfg["epochs"]
    num_warmup_steps = int(num_training_steps * train_cfg["warmup_ratio"])
    scheduler = get_linear_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps)
    logger.info(f"Total training steps: {num_training_steps}, warmup: {num_warmup_steps}")

    # --- Trainer ---
    trainer = MTLTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        max_grad_norm=train_cfg["max_grad_norm"],
        id2ner=id2ner,
        id2rel=id2rel,
    )

    # --- Training Loop ---
    output_dir = cfg["model"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    best_joint_f1 = -1.0
    patience_counter = 0
    patience = train_cfg["early_stopping_patience"]
    history = []
    checkpoint_saved_this_run = False

    logger.info("=" * 60)
    logger.info("Starting Multi-Task Training (NER + RE)")
    logger.info("=" * 60)

    for epoch in range(1, train_cfg["epochs"] + 1):
        # Train
        train_metrics = trainer.train_epoch(train_loader)

        # Evaluate
        val_metrics = trainer.evaluate(val_loader)

        # Log
        joint_f1 = val_metrics.get("val_joint_f1", 0.0)
        ner_f1 = val_metrics.get("val_ner_f1", 0.0)
        re_f1 = val_metrics.get("val_re_f1", 0.0)

        logger.info(
            f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"(ner={train_metrics['ner_loss']:.4f}, re={train_metrics['re_loss']:.4f}) | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"NER F1={ner_f1:.4f} | RE F1={re_f1:.4f} | Joint F1={joint_f1:.4f}"
        )

        epoch_record = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(epoch_record)

        # Save best
        if joint_f1 > best_joint_f1:
            best_joint_f1 = joint_f1
            patience_counter = 0
            if train_cfg["save_best_only"]:
                trainer.save_checkpoint(output_dir)
                checkpoint_saved_this_run = True
                logger.info(f"  ✅ Best model saved (Joint F1={best_joint_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  ⏹  Early stopping at epoch {epoch} (patience={patience})")
                break

    # --- Save tokenizer & config alongside model ---
    tokenizer.save_pretrained(output_dir)

    # Save label maps
    with open(os.path.join(output_dir, "label_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "ner_labels": ner_labels,
            "relation_labels": rel_labels,
            "base_model_path": base_model_path,   # เก็บ path ไว้ให้ inference ใช้โหลด encoder
        }, f, ensure_ascii=False, indent=2)

    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # Reload the best joint checkpoint before exporting the NER-only HF model.
    checkpoint_path = Path(output_dir) / "model.pt"
    if not checkpoint_saved_this_run:
        # Save the final model if best-only checkpoint saving was disabled or no
        # validation improvement produced a checkpoint during this run.
        trainer.save_checkpoint(output_dir)
    best_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state)

    ner_export_dir = cfg["model"].get("ner_export_dir") or f"{output_dir}_ner"
    export_ner_standalone(
        mtl_model=model,
        tokenizer=tokenizer,
        base_model_path=base_model_path,
        ner_labels=ner_labels,
        output_dir=ner_export_dir,
        source_mtl_dir=output_dir,
    )

    logger.info("=" * 60)
    logger.info(f"Training complete! Best Joint F1: {best_joint_f1:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Standalone NER model saved to: {ner_export_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
