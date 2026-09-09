#!/usr/bin/env python3
"""
scripts/preprocess_lexitron.py
Step 2: Clean and structure Lexitron CSV into JSONL.

Usage:
    python scripts/preprocess_lexitron.py
    python scripts/preprocess_lexitron.py --config configs/data_config.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.data.loader import load_lexitron, validate_thai_columns
from src.data.cleaner import clean_text, clean_list_field
from src.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("outputs/logs"))


def process_record(row: dict, col_map: dict, min_len: int = 2) -> dict | None:
    """
    Convert a raw CSV row into a structured record dict.

    Returns None if the entry word is empty after cleaning.
    """
    def get(key: str) -> str:
        col = col_map.get(key, key)
        return str(row.get(col, "")).strip()

    word = clean_text(get("entry"), min_length=min_len)
    if not word:
        return None

    definition = clean_text(get("definition"), min_length=2) or ""
    synonym_raw = clean_list_field(get("synonym"), sep=",")
    antonym_raw = clean_list_field(get("antonym"), sep=",")
    sample_raw = clean_text(get("sample"), min_length=5) or ""
    category = clean_text(get("category"), min_length=1) or ""

    examples = [s.strip() for s in sample_raw.split("  ") if s.strip()] if sample_raw else []

    return {
        "word": word,
        "category": category,
        "definition": definition,
        "synonyms": synonym_raw,
        "antonyms": antonym_raw,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Lexitron CSV to JSONL")
    parser.add_argument("--config", default="configs/data_config.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]

    csv_path = Path(cfg["raw_dir"]) / cfg["thai_file"]
    interim_dir = Path(cfg["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_path = interim_dir / "lexitron_clean.jsonl"

    col_map: dict = cfg.get("column_map", {})
    min_len: int = cfg.get("min_text_length", 2)

    # Load CSV
    df = load_lexitron(csv_path)
    logger.info(f"Raw records: {len(df):,}")

    # Drop full-row duplicates
    initial = len(df)
    df = df.drop_duplicates()
    logger.info(f"After dedup: {len(df):,} (removed {initial - len(df):,})")

    # Process rows
    records = []
    skipped = 0
    for _, row in df.iterrows():
        rec = process_record(row.to_dict(), col_map, min_len)
        if rec:
            records.append(rec)
        else:
            skipped += 1

    logger.info(f"Valid records: {len(records):,} | Skipped (no word): {skipped:,}")

    # Save JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Saved → {output_path}")

    # Preview
    logger.info("Preview (first 2 records):")
    for rec in records[:2]:
        logger.info(f"  {rec}")


if __name__ == "__main__":
    main()
