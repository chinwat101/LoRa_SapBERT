#!/usr/bin/env python3
"""
scripts/inference_ner.py
Run NER inference with fine-tuned NER models (SapBERT or WangchanBERTa).

Supports:
1. Standalone merged NER models (e.g. outputs/merged_model_1_ner_1)
2. Base model + LoRA adapter

Usage:
    # Auto-detect latest merged NER model (or adapter)
    python scripts/inference_ner.py --text "อะบิราเทอโรน ใช้รักษามะเร็งต่อมลูกหมาก"

    # Specific merged model path
    python scripts/inference_ner.py --model outputs/merged_model_1_ner_1 --text "พาราเซตามอล แก้ปวดลดไข้"

    # Interactive mode
    python scripts/inference_ner.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def find_latest_ner_model(output_root: str = "outputs") -> Path | None:
    """Find the latest merged NER model directory (e.g. outputs/merged_model_1_ner_1)."""
    base = Path(output_root)
    candidates = sorted(
        [p for p in base.glob("*_ner_*") if p.is_dir() and (p / "config.json").exists()],
        key=lambda p: (
            int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
            p.stat().st_mtime
        )
    )
    if candidates:
        return candidates[-1]
    
    # Fallback to wangchan_ner merged model
    wc_merged = base / "wangchan_ner" / "merged_model"
    if wc_merged.exists() and (wc_merged / "config.json").exists():
        return wc_merged

    return None


def load_ner_model(model_dir_or_adapter: str, base_model_dir: str = None):
    """
    Load NER model.
    If model_dir contains a standalone model, load it directly.
    Otherwise treat as LoRA adapter with base_model_dir.
    """
    target_path = Path(model_dir_or_adapter)

    # 1. Check label_schema.json
    schema_path = target_path / "label_schema.json"
    if not schema_path.exists() and base_model_dir:
        schema_path = Path(base_model_dir) / "label_schema.json"

    if schema_path.exists():
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        label2id = schema["label2id"]
        id2label = {int(k): v for k, v in schema["id2label"].items()}
    else:
        label2id = {"O": 0, "B-DRUG": 1, "I-DRUG": 2, "B-DISEASE": 3, "I-DISEASE": 4, "B-SYMPTOM": 5, "I-SYMPTOM": 6}
        id2label = {v: k for k, v in label2id.items()}

    # 2. Check if adapter vs standalone model
    is_adapter = (target_path / "adapter_config.json").exists()

    if is_adapter and base_model_dir:
        from peft import PeftModel
        print(f"Loading tokenizer from: {target_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(target_path))
        print(f"Loading base model from: {base_model_dir}")
        base_model = AutoModelForTokenClassification.from_pretrained(
            base_model_dir,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        print(f"Loading LoRA adapter from: {target_path}")
        model = PeftModel.from_pretrained(base_model, str(target_path))
    else:
        print(f"Loading standalone NER model & tokenizer from: {target_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(target_path))
        model = AutoModelForTokenClassification.from_pretrained(
            str(target_path),
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    model.eval()
    print("Model ready [OK]\n")
    return model, tokenizer, id2label


def predict(text: str, model, tokenizer, id2label: dict, max_length: int = 256) -> list:
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    offset_mapping = enc.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        logits = model(**enc).logits[0]  # (seq_len, num_labels)
    pred_ids = logits.argmax(dim=-1).tolist()

    token_tags = []
    input_ids_list = enc["input_ids"][0].tolist()
    tok_idx = 0
    for pred_id, (cs, ce) in zip(pred_ids, offset_mapping):
        if cs == 0 and ce == 0:  # special token
            tok_idx += 1
            continue
        token_str = tokenizer.convert_ids_to_tokens([input_ids_list[tok_idx]])[0]
        tag = id2label.get(pred_id, "O")
        token_tags.append((token_str, tag, cs, ce))
        tok_idx += 1

    entities = []
    current_label = None
    current_start = None
    current_end   = None

    for _, tag, cs, ce in token_tags:
        if tag.startswith("B-"):
            label = tag[2:]
            if current_label and current_start == cs:
                current_label = label
                current_end   = ce
            else:
                if current_label:
                    entities.append({
                        "text":  text[current_start:current_end],
                        "label": current_label,
                        "start": current_start,
                        "end":   current_end,
                    })
                current_label = label
                current_start = cs
                current_end   = ce
        elif tag.startswith("I-") and current_label == tag[2:]:
            current_end = ce
        else:
            if current_label:
                entities.append({
                    "text":  text[current_start:current_end],
                    "label": current_label,
                    "start": current_start,
                    "end":   current_end,
                })
            current_label = None
            current_start = None
            current_end   = None

    if current_label:
        entities.append({
            "text":  text[current_start:current_end],
            "label": current_label,
            "start": current_start,
            "end":   current_end,
        })

    return entities


LABEL_COLOR = {
    "DRUG":    "\033[94m",   # blue
    "DISEASE": "\033[91m",   # red
    "SYMPTOM": "\033[93m",   # yellow
}
RESET = "\033[0m"


def print_result(text: str, entities: list) -> None:
    print(f"\nText: {text}")
    if not entities:
        print("  (ไม่พบ entity)")
        return
    for e in entities:
        color = LABEL_COLOR.get(e["label"], "")
        print(f"  {color}[{e['label']:8s}]{RESET}  '{e['text']}'  (pos {e['start']}–{e['end']})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="NER inference for SapBERT / WangchanBERTa")
    parser.add_argument("--model", default=None, help="Path to standalone merged NER model directory")
    parser.add_argument("--adapter", default=None, help="Path to saved LoRA adapter directory")
    parser.add_argument("--base-model", default=None, help="Path to base model when using adapter")
    parser.add_argument("--text", default=None, help="Input text to run NER on")
    args = parser.parse_args()

    target_dir = args.model or args.adapter
    if not target_dir:
        latest = find_latest_ner_model()
        if latest:
            target_dir = str(latest)
            print(f"Auto-detected latest NER model: {target_dir}")
        else:
            target_dir = "outputs/wangchan_ner/adapter"

    model, tokenizer, id2label = load_ner_model(target_dir, args.base_model)

    if args.text:
        entities = predict(args.text, model, tokenizer, id2label)
        print_result(args.text, entities)
    else:
        print("=" * 60)
        print("Clinical NER — Interactive Mode")
        print("พิมพ์ข้อความภาษาไทย แล้วกด Enter (หรือพิมพ์ 'quit' เพื่อออก)")
        print("=" * 60)
        EXAMPLE_TEXTS = [
            "อะบิราเทอโรน ใช้รักษามะเร็งต่อมลูกหมาก โดยมีอาการแพร่กระจาย",
            "อะดาลิมูแมบ ใช้รักษาข้ออักเสบรูมาตอยด์และโรคโครห์น โดยมีอาการอักเสบ",
            "อัลเบนดาโซล ใช้รักษาโรคพยาธิตัวตืด",
        ]
        print("\nตัวอย่าง:")
        for t in EXAMPLE_TEXTS:
            print(f"  > {t}")
        print()

        while True:
            try:
                text = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not text or text.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            entities = predict(text, model, tokenizer, id2label)
            print_result(text, entities)


if __name__ == "__main__":
    main()
