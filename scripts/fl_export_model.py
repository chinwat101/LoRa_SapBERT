#!/usr/bin/env python3
"""
scripts/fl_export_model.py

แปลง FL global adapter weights (numpy) → HuggingFace NER model
เพื่อให้ใช้กับ inference_ner.py ได้โดยตรง

Usage:
    # Export best round (default: round_04)
    python scripts/fl_export_model.py

    # Export specific round
    python scripts/fl_export_model.py --round 4

    # Override paths
    python scripts/fl_export_model.py \
        --base-model outputs/merged_model_3 \
        --fl-dir outputs/fl_ner \
        --round 4 \
        --output-dir outputs/fl_ner_exported
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("outputs/logs"))

DEFAULT_LABEL2ID = {
    "O": 0, "B-DRUG": 1, "I-DRUG": 2,
    "B-DISEASE": 3, "I-DISEASE": 4,
    "B-SYMPTOM": 5, "I-SYMPTOM": 6,
}
DEFAULT_ID2LABEL = {v: k for k, v in DEFAULT_LABEL2ID.items()}


def load_label_schema(schema_path: str) -> tuple:
    p = Path(schema_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            schema = json.load(f)
        label2id = schema["label2id"]
        id2label = {int(k): v for k, v in schema["id2label"].items()}
        return label2id, id2label
    return DEFAULT_LABEL2ID, DEFAULT_ID2LABEL


def find_best_round(fl_dir: Path) -> int:
    """อ่าน fl_training_log.json แล้วหา round ที่ global F1 สูงสุด"""
    log_path = fl_dir / "fl_training_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found: {log_path}")
    with open(log_path) as f:
        logs = json.load(f)
    best = max(logs, key=lambda x: x["global_f1"])
    logger.info(f"Best round from log: Round {best['round']} — F1: {best['global_f1']:.4f}")
    return best["round"]


def main():
    parser = argparse.ArgumentParser(description="Export FL adapter → HuggingFace NER model")
    parser.add_argument("--config",      default="configs/fl_config.yaml")
    parser.add_argument("--base-model",  default=None, help="Path to base model (merged_model_3)")
    parser.add_argument("--fl-dir",      default=None, help="FL output dir (outputs/fl_ner)")
    parser.add_argument("--round",       type=int, default=None, help="Round to export (default: best round)")
    parser.add_argument("--output-dir",  default=None, help="Where to save the exported model")
    args = parser.parse_args()

    # ─── โหลด Config ─────────────────────────────────────────────────────────
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_model = args.base_model or cfg["model"]["base_model_path"]
    fl_dir     = Path(args.fl_dir or cfg["model"]["output_dir"])
    lora_cfg   = cfg["lora"]
    data_cfg   = cfg["data"]

    label2id, id2label = load_label_schema(data_cfg["label_schema_file"])

    # ─── หา round ที่ดีที่สุด ─────────────────────────────────────────────────
    target_round = args.round or find_best_round(fl_dir)
    round_dir    = fl_dir / f"round_{target_round:02d}"
    weights_path = round_dir / "global_adapter_weights.npy"

    if not weights_path.exists():
        raise FileNotFoundError(f"Adapter weights not found: {weights_path}")

    output_dir = Path(args.output_dir or f"outputs/fl_ner_model_round{target_round:02d}")

    logger.info("=" * 55)
    logger.info("FL ADAPTER EXPORT")
    logger.info("=" * 55)
    logger.info(f"Base Model   : {base_model}")
    logger.info(f"FL Dir       : {fl_dir}")
    logger.info(f"Export Round : {target_round}")
    logger.info(f"Output Dir   : {output_dir}")
    logger.info("=" * 55)

    # ─── โหลด base model ─────────────────────────────────────────────────────
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ─── เพิ่ม LoRA adapter ──────────────────────────────────────────────────
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.1),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get("target_modules", ["query", "key", "value", "dense"]),
        inference_mode=True,
    )
    model = get_peft_model(model, peft_config)

    # ─── โหลด global FL weights เข้า adapter ────────────────────────────────
    logger.info(f"Loading FL adapter weights from: {weights_path}")
    global_weights = np.load(str(weights_path), allow_pickle=True)

    lora_state = get_peft_model_state_dict(model)
    if len(global_weights) != len(lora_state):
        raise ValueError(
            f"Weight count mismatch: FL={len(global_weights)}, model={len(lora_state)}\n"
            "Make sure --base-model and LoRA config match the training config."
        )

    params_dict = {
        k: torch.tensor(v.astype(np.float32))
        for k, v in zip(lora_state.keys(), global_weights)
    }
    set_peft_model_state_dict(model, params_dict)
    logger.info(f"✅ FL adapter weights loaded ({len(global_weights)} tensors)")

    # ─── บันทึก merged model (base + LoRA merged) ────────────────────────────
    logger.info("Merging LoRA adapter into base model...")
    merged_model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # ─── บันทึก metadata ─────────────────────────────────────────────────────
    meta = {
        "base_model": base_model,
        "fl_dir": str(fl_dir),
        "exported_round": target_round,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "lora_config": lora_cfg,
    }
    with open(output_dir / "fl_export_info.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 55)
    logger.info("✅ EXPORT COMPLETE")
    logger.info("=" * 55)
    logger.info(f"Model saved to: {output_dir}")
    logger.info("\nTest inference:")
    logger.info(f'  python scripts/inference_ner.py \\')
    logger.info(f'    --model {output_dir} \\')
    logger.info(f'    --text "อะบิราเทอโรน ใช้รักษามะเร็งต่อมลูกหมาก"')
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
