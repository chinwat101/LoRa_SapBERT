#!/usr/bin/env python3
"""
scripts/convert_ner_dataset.py

แปลง data_EN.csv เป็น NER dataset สำหรับ fine-tune WangchanBERTa
Entity types:
    DRUG    -> drug_name_EN, drug_name_TH
    DISEASE -> disease_EN, disease_TH
    SYMPTOM -> symptom_EN, symptom_TH

Fixes:
    [1] Entity overlap: DRUG > DISEASE > SYMPTOM priority; nested spans removed
    [2] Proper subword tokenisation via WangchanBERTa tokenizer + offset_mapping
    [3] Deduplication: identical sentences (same text + lang) are merged

Output:
    data/processed/ner_spans.jsonl   -- all examples with character-level spans
    data/processed/ner_bio_th.jsonl  -- Thai examples with subword BIO tags
    data/processed/ner_bio_en.jsonl  -- English examples with subword BIO tags
    data/processed/ner_label_schema.json

Usage:
    python scripts/convert_ner_dataset.py
    python scripts/convert_ner_dataset.py --input data/raw/data_EN.csv --output data/processed/
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ──────────────────────────────────────────────────────────────────────────────
# Label schema
# ──────────────────────────────────────────────────────────────────────────────
LABEL2ID = {
    "O":         0,
    "B-DRUG":    1,
    "I-DRUG":    2,
    "B-DISEASE": 3,
    "I-DISEASE": 4,
    "B-SYMPTOM": 5,
    "I-SYMPTOM": 6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Priority when spans overlap: higher wins
ENTITY_PRIORITY = {"DRUG": 3, "DISEASE": 2, "SYMPTOM": 1}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def split_multi_value(text: str) -> list:
    if not isinstance(text, str) or not text.strip():
        return []
    return [clean_text(p) for p in text.split(",") if clean_text(p)]


def find_spans(text: str, mentions: list, label: str) -> list:
    """Return all character-level occurrences of each mention."""
    spans = []
    for mention in mentions:
        if not mention:
            continue
        pos = 0
        while True:
            idx = text.find(mention, pos)
            if idx == -1:
                break
            spans.append({"start": idx, "end": idx + len(mention), "label": label, "text": mention})
            pos = idx + len(mention)
    return spans


# ──────────────────────────────────────────────────────────────────────────────
# FIX 1 — Resolve overlapping spans
# ──────────────────────────────────────────────────────────────────────────────
def resolve_overlapping_spans(spans: list) -> list:
    """
    Remove overlapping spans using entity priority (DRUG > DISEASE > SYMPTOM).
    Among same priority, longer span wins; ties broken by earlier start.
    """
    if not spans:
        return []

    # Sort: highest priority first, longest first, earliest first
    ordered = sorted(
        spans,
        key=lambda s: (
            ENTITY_PRIORITY.get(s["label"], 0),
            s["end"] - s["start"],
            -s["start"],
        ),
        reverse=True,
    )

    kept: list = []
    occupied: list[tuple] = []

    for span in ordered:
        s, e = span["start"], span["end"]
        if not any(not (e <= ks or s >= ke) for ks, ke in occupied):
            kept.append(span)
            occupied.append((s, e))

    return sorted(kept, key=lambda s: s["start"])


# ──────────────────────────────────────────────────────────────────────────────
# FIX 2 — Subword-aligned BIO tags via WangchanBERTa tokenizer
# ──────────────────────────────────────────────────────────────────────────────
def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        print(f"  Tokenizer loaded (cache): {type(tok).__name__}")
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name)
        print(f"  Tokenizer downloaded: {type(tok).__name__}")
    return tok


def spans_to_bio(text: str, spans: list, tokenizer) -> dict:
    """
    Tokenise `text` with WangchanBERTa and produce subword-aligned BIO tags
    using offset_mapping. Special tokens (<s>, </s>) are excluded.
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )

    # Build char -> label and set of span-start positions
    char_label: dict = {}
    span_starts: set = set()
    for span in spans:
        span_starts.add(span["start"])
        for i in range(span["start"], span["end"]):
            char_label.setdefault(i, span["label"])

    tokens:   list = []
    ner_tags: list = []

    for tok_id, (cs, ce) in zip(enc["input_ids"], enc["offset_mapping"]):
        if cs == 0 and ce == 0:          # special token
            continue
        tok_str = tokenizer.convert_ids_to_tokens([tok_id])[0]
        tokens.append(tok_str)

        tag = "O"
        for ci in range(cs, ce):
            if ci in char_label:
                prefix = "B-" if ci in span_starts else "I-"
                tag = f"{prefix}{char_label[ci]}"
                break

        # Continuity fix: if previous token already carries this entity label
        # and this char is NOT a span-start, demote B- → I-
        if tag.startswith("B-") and ner_tags:
            entity = tag[2:]
            prev_entity = ner_tags[-1][2:] if ner_tags[-1] != "O" else ""
            if prev_entity == entity:
                first_ci = next((ci for ci in range(cs, ce) if ci in char_label), None)
                if first_ci is not None and first_ci not in span_starts:
                    tag = f"I-{entity}"

        ner_tags.append(tag)

    return {
        "tokens":   tokens,
        "ner_tags": ner_tags,
        "ner_ids":  [LABEL2ID.get(t, 0) for t in ner_tags],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sentence builders
# ──────────────────────────────────────────────────────────────────────────────
def build_thai_sentence(drug: str, disease: str, symptom: str, description: str) -> str:
    parts = []
    if drug:
        parts.append(drug)
    if disease:
        parts.append(f"ใช้รักษา {disease}")
    if symptom:
        parts.append(f"โดยมีอาการ {symptom}")
    if description:
        parts.append(description)
    return " ".join(parts)


def build_english_sentence(drug: str, disease: str, symptom: str) -> str:
    parts = []
    if drug:
        parts.append(drug)
    if disease:
        parts.append(f"is used to treat {disease}")
    if symptom:
        parts.append(f"with symptoms of {symptom}")
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Row → examples
# ──────────────────────────────────────────────────────────────────────────────
def row_to_examples(row: "pd.Series", tokenizer) -> list:
    source = clean_text(str(row.get("source", "")))
    examples = []

    # ─ Thai ─
    drug_th  = clean_text(row.get("drug_name_TH", ""))
    dis_th   = split_multi_value(row.get("disease_TH", ""))
    sym_th   = split_multi_value(row.get("symptom_TH", ""))
    desc     = clean_text(row.get("description", ""))

    text_th = build_thai_sentence(drug_th, "และ".join(dis_th), "และ".join(sym_th), desc)
    if text_th:
        raw = (find_spans(text_th, [drug_th], "DRUG")
               + [s for d in dis_th for s in find_spans(text_th, [d], "DISEASE")]
               + [s for sm in sym_th for s in find_spans(text_th, [sm], "SYMPTOM")])
        clean = resolve_overlapping_spans(raw)       # Fix 1
        bio   = spans_to_bio(text_th, clean, tokenizer)  # Fix 2
        examples.append({"id": f"{source}_th", "text": text_th, "lang": "th",
                         "source": source, "entities": clean, **bio})

    # ─ English ─
    drug_en = clean_text(row.get("drug_name_EN", ""))
    dis_en  = split_multi_value(row.get("disease_EN", ""))
    sym_en  = split_multi_value(row.get("symptom_EN", ""))

    text_en = build_english_sentence(drug_en, " and ".join(dis_en), " and ".join(sym_en))
    if text_en:
        raw = (find_spans(text_en, [drug_en], "DRUG")
               + [s for d in dis_en for s in find_spans(text_en, [d], "DISEASE")]
               + [s for sm in sym_en for s in find_spans(text_en, [sm], "SYMPTOM")])
        clean = resolve_overlapping_spans(raw)
        bio   = spans_to_bio(text_en, clean, tokenizer)
        examples.append({"id": f"{source}_en", "text": text_en, "lang": "en",
                         "source": source, "entities": clean, **bio})

    return examples


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw/data_EN.csv")
    parser.add_argument("--output", default="data/processed/")
    parser.add_argument("--model",  default="airesearch/wangchanberta-base-att-spm-uncased")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {args.input}")
    df = pd.read_csv(args.input, encoding="utf-8")
    df = df[[c for c in df.columns if not c.startswith("Unnamed")]]
    print(f"  Rows: {len(df)}")

    print("Loading tokenizer...")
    tok = load_tokenizer(args.model)

    all_ex, th_ex, en_ex = [], [], []
    seen: set = set()   # Fix 3: deduplicate

    for _, row in df.iterrows():
        for ex in row_to_examples(row, tok):
            key = (ex["text"], ex["lang"])
            if key in seen:
                continue
            seen.add(key)
            all_ex.append(ex)
            (th_ex if ex["lang"] == "th" else en_ex).append(ex)

    print(f"\n  Total (after dedup): {len(all_ex)}  ({len(th_ex)} TH + {len(en_ex)} EN)")

    def save(path: Path, data: list) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Saved {path}  ({len(data)} records)")

    save(output_dir / "ner_spans.jsonl",  all_ex)
    save(output_dir / "ner_bio_th.jsonl", th_ex)
    save(output_dir / "ner_bio_en.jsonl", en_ex)

    with open(output_dir / "ner_label_schema.json", "w", encoding="utf-8") as f:
        json.dump({
            "entity_types": ["DRUG", "DISEASE", "SYMPTOM"],
            "bio_labels":   list(LABEL2ID.keys()),
            "label2id":     LABEL2ID,
            "id2label":     {str(k): v for k, v in ID2LABEL.items()},
            "num_labels":   len(LABEL2ID),
            "tokenizer":    args.model,
        }, f, ensure_ascii=False, indent=2)
    print(f"  Saved {output_dir / 'ner_label_schema.json'}")

    # ─ Preview ─
    print("\n" + "=" * 60)
    print("PREVIEW — adalimumab (overlap fix demo):")
    print("=" * 60)
    ex7 = next((e for e in th_ex if "อักเสบ" in e["text"] and len(e["entities"]) > 4), None)
    if ex7:
        print(f"  text: {ex7['text'][:100]}...")
        print(f"  entities ({len(ex7['entities'])}) — no nested SYMPTOM inside DISEASE:")
        for e in ex7["entities"]:
            print(f"    [{e['label']:8s}] '{e['text']}'")
        print(f"  tokens  : {ex7['tokens'][:12]}")
        print(f"  ner_tags: {ex7['ner_tags'][:12]}")

    print("\nPREVIEW — simple Thai example (subword tokenisation):")
    simple = next((e for e in th_ex if len(e["entities"]) == 3), None)
    if simple:
        print(f"  text: {simple['text']}")
        for e in simple["entities"]:
            print(f"    [{e['label']:8s}] '{e['text']}'")
        for tok_str, tag in zip(simple["tokens"], simple["ner_tags"]):
            print(f"    {tok_str:<30} {tag}")

    print("\nDone! ✅")


if __name__ == "__main__":
    main()
