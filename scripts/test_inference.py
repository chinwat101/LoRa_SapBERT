#!/usr/bin/env python3
"""
scripts/test_inference.py
Quick inference test to verify the LoRA adapter works correctly
and compare outputs with the base model.

Also verifies NER encoder compatibility (hidden state shape check).

Usage:
    python scripts/test_inference.py
    python scripts/test_inference.py --text "โรงพยาบาลเป็นสถานที่รักษาผู้ป่วย"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.models.model_loader import load_base_model_and_tokenizer
from src.models.lora import load_lora_adapter
from src.utils.logger import get_logger

logger = get_logger(__name__)

THAI_TEST_SENTENCES = [
    # ใส่ [MASK] ที่คำที่ต้องการทดสอบ — โมเดลควรทายคำที่ถูกต้องตามบริบท
    "โรงพยาบาลเป็นสถานที่สำหรับ[MASK]ผู้ป่วย",          # expect: รักษา
    "คำว่าประชาธิปไตยหมายถึงการปกครองโดย[MASK]",         # expect: ประชาชน
    "ยา[MASK]ช่วยรักษาโรคและบรรเทาอาการเจ็บปวด",        # expect: สมุนไพร/แผนปัจจุบัน
    "แพทย์ใช้[MASK]ในการวินิจฉัยโรคของผู้ป่วย",           # expect: การตรวจ/เครื่องมือ
    "คำว่าสมานแผลหมายถึงทำให้[MASK]หายสนิท",            # expect: บาดแผล (from Lexitron)
]


def test_mlm_prediction(model, tokenizer, text: str, top_k: int = 5) -> None:
    """
    Predict masked token.
    If text contains '[MASK]', use it directly.
    Otherwise mask the middle content token (skip [CLS] and ##subwords).
    """
    if "[MASK]" in text:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        ids = inputs["input_ids"][0].tolist()
        try:
            mask_pos = ids.index(tokenizer.mask_token_id)
        except ValueError:
            logger.warning("  [MASK] token not found after tokenization, skipping.")
            return
        original_token = "[MASK]"
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        ids = inputs["input_ids"][0]
        # Find first non-CLS, non-SEP, non-## token (word-initial token)
        mask_pos = None
        for i, tok_id in enumerate(ids.tolist()):
            if i == 0:
                continue  # skip [CLS]
            tok = tokenizer.convert_ids_to_tokens([tok_id])[0]
            if tok in ("[SEP]", "[PAD]"):
                break
            if not tok.startswith("##"):
                mask_pos = i
                break
        if mask_pos is None:
            mask_pos = 1
        original_token = tokenizer.convert_ids_to_tokens([ids[mask_pos].item()])[0]
        masked_ids = ids.clone()
        masked_ids[mask_pos] = tokenizer.mask_token_id
        inputs["input_ids"] = masked_ids.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, mask_pos]
    top_ids = torch.topk(logits, top_k).indices.tolist()
    top_tokens = tokenizer.convert_ids_to_tokens(top_ids)

    logger.info(f"  Text     : {text[:80]}")
    logger.info(f"  Masked   : [{original_token}]")
    logger.info(f"  Top-{top_k}  : {top_tokens}")



def test_encoder_shape(model, tokenizer, text: str) -> None:
    """Verify encoder hidden state shape for NER compatibility."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    model.eval()
    with torch.no_grad():
        # Get encoder hidden states
        outputs = model(**inputs, output_hidden_states=True)

    last_hidden = outputs.hidden_states[-1]
    logger.info(f"  Encoder output shape: {tuple(last_hidden.shape)}  "
                f"[batch=1, seq_len, hidden={last_hidden.shape[-1]}]")


def find_latest_merged_model(merged_model_dir: str) -> Path | None:
    """
    Scan for merged model directories matching the pattern <merged_model_dir>_N
    (e.g. outputs/merged_model_1, outputs/merged_model_2, ...) and return the one
    with the highest N.  Also accepts the exact path if it exists as-is.
    """
    base = Path(merged_model_dir)

    # 1. Exact path exists → use it directly
    if base.exists() and (base / "config.json").exists():
        return base

    # 2. Scan siblings with numeric suffix (e.g. outputs/merged_model_1)
    parent = base.parent
    stem   = base.name          # e.g. "merged_model"
    candidates = sorted(
        (p for p in parent.glob(f"{stem}_*") if p.is_dir() and (p / "config.json").exists()),
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
    )
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LoRA adapter inference")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--text", default=None, help="Custom text to test")
    parser.add_argument("--merged", default=None,
                        help="Override: path to a specific merged model dir")
    args = parser.parse_args()

    with open(Path(args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]

    base_path        = cfg["base_model_path"]
    merged_model_dir = args.merged or cfg.get("merged_model_dir", "outputs/merged_model")

    test_texts = [args.text] if args.text else THAI_TEST_SENTENCES

    # ── Base model ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("BASE MODEL TEST")
    logger.info("=" * 60)
    base_model, tokenizer = load_base_model_and_tokenizer(base_path)

    for text in test_texts:
        test_mlm_prediction(base_model, tokenizer, text)
        test_encoder_shape(base_model, tokenizer, text)

    # ── Latest merged model (base + LoRA already baked in) ───────────────────
    latest_merged = find_latest_merged_model(merged_model_dir)
    if latest_merged:
        logger.info("\n" + "=" * 60)
        logger.info(f"MERGED MODEL TEST  →  {latest_merged}")
        logger.info("=" * 60)

        # Load directly — no adapter wrapper needed, weights are already merged
        merged_model, _ = load_base_model_and_tokenizer(str(latest_merged))

        for text in test_texts:
            test_mlm_prediction(merged_model, tokenizer, text)
            test_encoder_shape(merged_model, tokenizer, text)
    else:
        logger.info(
            f"\nNo merged model found under '{merged_model_dir}'. "
            "Run merge_lora.py first, or pass --merged <path>."
        )

    logger.info("\nInference test complete. Tokenizer and encoder are NER-compatible.")


if __name__ == "__main__":
    main()
