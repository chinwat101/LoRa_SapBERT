#!/usr/bin/env python3
"""
==============================================================================
Multi-Task Learning Inference Script: NER + Relation Extraction
==============================================================================
ไฟล์นี้สร้างขึ้นเพื่อ:
  รับข้อความทางการแพทย์ภาษาไทยเป็น Input
  ทำนาย Named Entities (DRUG, DISEASE, SYMPTOM) ด้วย NER Head
  ทำนาย Relations (TREATS, INDICATED_FOR) ระหว่าง entity pairs ด้วย RE Head
  แสดงผลเป็น JSON ที่อ่านง่าย

การใช้งาน:
  python scripts/inference_mtl.py --model outputs/mtl_model \
      --text "ลอราทาดีน ใช้บรรเทาอาการของโรคภูมิแพ้"

  python scripts/inference_mtl.py --model outputs/mtl_model \
      --text "ลอราทาดีน ใช้บรรเทาอาการของโรคภูมิแพ้" --threshold 0.5
==============================================================================
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.mtl_model import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_label_config(model_dir: str) -> Tuple[List[str], List[str], str]:
    """โหลด NER labels, RE labels และ base_model_path จาก label_config.json"""
    config_path = Path(model_dir) / "label_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        base_model_path = cfg.get("base_model_path", model_dir)
        return cfg["ner_labels"], cfg["relation_labels"], base_model_path
    # Default fallback
    return (
        ["O", "B-DRUG", "I-DRUG", "B-DISEASE", "I-DISEASE", "B-SYMPTOM", "I-SYMPTOM"],
        ["NO_RELATION", "TREATS", "INDICATED_FOR"],
        model_dir,
    )


def extract_entities(
    text: str,
    tokens: List[str],
    offsets: List[Tuple[int, int]],
    ner_tags: List[str],
) -> List[Dict[str, Any]]:
    """
    แปลง BIO token tags → entity spans ในระดับ character
    Returns list of {"text", "label", "start", "end", "tok_start", "tok_end"}
    """
    entities = []
    current = None

    for tok_idx, (token, (c_start, c_end), tag) in enumerate(
        zip(tokens, offsets, ner_tags)
    ):
        if c_start == c_end:  # special token
            if current:
                entities.append(current)
                current = None
            continue

        if tag == "O":
            if current:
                entities.append(current)
                current = None
            continue

        prefix, label = tag.split("-", 1) if "-" in tag else ("", tag)

        if prefix == "B":
            if current:
                entities.append(current)
            current = {
                "text": text[c_start:c_end],
                "label": label,
                "start": c_start,
                "end": c_end,
                "tok_start": tok_idx,
                "tok_end": tok_idx,
            }
        elif prefix == "I" and current and current["label"] == label:
            current["end"] = c_end
            current["text"] = text[current["start"]: c_end]
            current["tok_end"] = tok_idx
        else:
            if current:
                entities.append(current)
            current = None

    if current:
        entities.append(current)
    return entities


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(
    model,
    tokenizer,
    text: str,
    id2ner: Dict[int, str],
    id2rel: Dict[int, str],
    device: torch.device,
    rel_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    รัน end-to-end inference บนข้อความหนึ่งประโยค
    Returns dict {"text", "entities", "relations"}
    """
    model.eval()

    # --- Tokenize ---
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    offset_mapping = encoding["offset_mapping"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    num_tokens = input_ids.shape[1]

    # --- NER Forward (ไม่มี pairs ก่อน) ---
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    ner_logits = outputs["ner_logits"][0]   # [L, C]
    ner_pred_ids = ner_logits.argmax(dim=-1).tolist()

    # แปลง id → tag
    ner_tags = []
    for tok_idx, pred_id in enumerate(ner_pred_ids):
        c_start, c_end = offset_mapping[tok_idx]
        if c_start == c_end:
            ner_tags.append("O")
        else:
            ner_tags.append(id2ner.get(pred_id, "O"))

    # --- Extract entities ---
    entities = extract_entities(text, tokens, offset_mapping, ner_tags)

    # --- RE: สร้าง pairs จาก entities ที่พบ ---
    relations = []
    if len(entities) >= 2:
        pair_list = []
        pair_entity_idx = []

        for i in range(len(entities)):
            for j in range(len(entities)):
                if i == j:
                    continue
                tok_h = entities[i]["tok_start"]
                tok_t = entities[j]["tok_start"]
                pair_list.append([tok_h, tok_t])
                pair_entity_idx.append((i, j))

        if pair_list:
            pairs_tensor = torch.tensor(pair_list, dtype=torch.long).to(device)
            pair_sample_idx = torch.zeros(len(pair_list), dtype=torch.long).to(device)

            re_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pairs=pairs_tensor,
                pair_sample_idx=pair_sample_idx,
            )

            if re_outputs["re_logits"] is not None:
                re_probs = torch.softmax(re_outputs["re_logits"], dim=-1)  # [P, C]
                re_pred_ids = re_probs.argmax(dim=-1).tolist()
                re_pred_conf = re_probs.max(dim=-1).values.tolist()

                seen_pairs = set()
                for p_idx, (ent_i, ent_j) in enumerate(pair_entity_idx):
                    rel_id = re_pred_ids[p_idx]
                    rel_name = id2rel.get(rel_id, "NO_RELATION")
                    confidence = re_pred_conf[p_idx]

                    if rel_name == "NO_RELATION":
                        continue
                    if confidence < rel_threshold:
                        continue

                    # Deduplicate exact directed relations while preserving head → tail.
                    pair_key = (ent_i, ent_j, rel_name)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    relations.append({
                        "head": entities[ent_i]["text"],
                        "head_label": entities[ent_i]["label"],
                        "tail": entities[ent_j]["text"],
                        "tail_label": entities[ent_j]["label"],
                        "type": rel_name,
                        "confidence": round(confidence, 4),
                    })

    return {
        "text": text,
        "entities": [
            {k: v for k, v in e.items() if k not in ("tok_start", "tok_end")}
            for e in entities
        ],
        "relations": relations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="MTL Inference: NER + Relation Extraction"
    )
    parser.add_argument(
        "--model", type=str, default="outputs/mtl_model",
        help="Path to trained MTL model directory"
    )
    parser.add_argument(
        "--text", type=str, required=True,
        help="Thai medical text to analyze"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Minimum confidence threshold for relations (0.0 = all)"
    )
    parser.add_argument(
        "--pretty", action="store_true", default=True,
        help="Pretty-print JSON output"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Labels ---
    ner_labels, rel_labels, base_model_path = load_label_config(args.model)
    ner2id = {lbl: i for i, lbl in enumerate(ner_labels)}
    rel2id = {lbl: i for i, lbl in enumerate(rel_labels)}
    id2ner = {i: lbl for lbl, i in ner2id.items()}
    id2rel = {i: lbl for lbl, i in rel2id.items()}

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # --- Model ---
    # โหลด encoder จาก base_model_path (มี config.json ครบ)
    # แล้วค่อยโหลด fine-tuned weights จาก outputs/mtl_model/model.pt
    print(f"Loading encoder architecture from: {base_model_path}")
    model = build_model(
        base_model_path=base_model_path,
        num_ner_labels=len(ner_labels),
        num_rel_labels=len(rel_labels),
    )

    checkpoint = Path(args.model) / "model.pt"
    if checkpoint.exists():
        state_dict = torch.load(str(checkpoint), map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print(f"⚠️  No checkpoint found at {checkpoint}. Running with random weights.")

    model.to(device)

    # --- Inference ---
    result = run_inference(
        model=model,
        tokenizer=tokenizer,
        text=args.text,
        id2ner=id2ner,
        id2rel=id2rel,
        device=device,
        rel_threshold=args.threshold,
    )

    # --- Output ---
    indent = 2 if args.pretty else None
    print("\n" + "=" * 50)
    print("Inference Result")
    print("=" * 50)
    print(json.dumps(result, ensure_ascii=False, indent=indent))
    print("=" * 50)


if __name__ == "__main__":
    main()
