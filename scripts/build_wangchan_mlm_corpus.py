#!/usr/bin/env python3
"""
scripts/build_wangchan_mlm_corpus.py
Build MLM training corpus for WangchanBERTa from available text sources.

Sources (combined):
  1. data/processed/ner_spans.jsonl  — NER dataset sentences (TH + EN)
  2. data/raw/data_EN.csv            — description field (Thai clinical text)

Output:
  data/processed/mlm_train.jsonl    — {"text": "..."} per line
  data/processed/mlm_val.jsonl      — 10% held-out

Usage:
    python scripts/build_wangchan_mlm_corpus.py
"""
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def main():
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = 42
    random.seed(seed)

    texts: list[str] = []

    # ─ Source 1: NER spans (text field from all examples) ─
    ner_path = output_dir / "ner_spans.jsonl"
    if ner_path.exists():
        with open(ner_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    t = clean(obj.get("text", ""))
                    if t:
                        texts.append(t)
        print(f"  NER spans: {len(texts)} texts")
    else:
        print("  ner_spans.jsonl not found, skipping.")

    # ─ Source 2: CSV description + disease + symptom fields ─
    csv_path = Path("data/raw/data_EN.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding="utf-8")
        before = len(texts)
        for _, row in df.iterrows():
            for col in ["description", "disease_TH", "symptom_TH", "disease_EN", "symptom_EN"]:
                t = clean(str(row.get(col, "")))
                if t and len(t) > 10:
                    texts.append(t)
        print(f"  CSV fields: +{len(texts) - before} texts")
    else:
        print("  data_EN.csv not found, skipping.")

    # ─ Deduplicate and shuffle ─
    texts = list({t for t in texts if t})
    random.shuffle(texts)
    print(f"  Total unique texts: {len(texts)}")

    # ─ Split 90/10 ─
    split = int(len(texts) * 0.9)
    train_texts = texts[:split]
    val_texts   = texts[split:]

    def save(path: Path, data: list[str]):
        with open(path, "w", encoding="utf-8") as f:
            for t in data:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        print(f"  Saved {path}  ({len(data)} records)")

    save(output_dir / "mlm_train.jsonl", train_texts)
    save(output_dir / "mlm_val.jsonl",   val_texts)
    print("Done! ✅")


if __name__ == "__main__":
    main()
