#!/usr/bin/env python3
"""
scripts/inspect_dataset.py
Step 1: Inspect the Lexitron CSV dataset.

Usage:
    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --config configs/data_config.yaml
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("outputs/logs"))


def inspect_csv(csv_path: Path, encoding: str = "utf-8-sig") -> None:
    logger.info("=" * 60)
    logger.info(f"DATASET INSPECTION: {csv_path.name}")
    logger.info("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, encoding=encoding, dtype=str, keep_default_na=False)

    # Strip BOM from column names (utf-8-sig handles it but just in case)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]

    # ── 2. Columns ───────────────────────────────────────────────────────────
    logger.info(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        logger.info(f"  {i:2d}. '{col}'")

    # ── 3. Record count ──────────────────────────────────────────────────────
    logger.info(f"\nTotal records: {len(df):,}")

    # ── 4. Missing values ────────────────────────────────────────────────────
    logger.info("\nMissing / empty values per column:")
    for col in df.columns:
        empty = (df[col].str.strip() == "").sum()
        pct = 100 * empty / len(df)
        logger.info(f"  {col:<20s}: {empty:6,} ({pct:.1f}%)")

    # ── 5. Sample rows ───────────────────────────────────────────────────────
    logger.info("\nSample rows (first 3):")
    for _, row in df.head(3).iterrows():
        logger.info(f"  {dict(row)}")

    # ── 6. Thai column analysis ──────────────────────────────────────────────
    thai_cols = ["t-entry", "t-def", "t-syn", "t-ant", "t-sample"]
    logger.info("\nThai column fill rates:")
    for col in thai_cols:
        if col in df.columns:
            filled = (df[col].str.strip() != "").sum()
            logger.info(f"  {col:<15s}: {filled:6,} / {len(df):,} ({100*filled/len(df):.1f}%)")
        else:
            logger.warning(f"  {col:<15s}: NOT FOUND")

    # ── 7. Encoding check ────────────────────────────────────────────────────
    logger.info(f"\nFile encoding assumed: utf-8-sig (auto-BOM-strip)")
    raw_bytes = csv_path.read_bytes()[:4]
    if raw_bytes[:3] == b'\xef\xbb\xbf':
        logger.info("  BOM detected: YES (UTF-8 BOM)")
    else:
        logger.info("  BOM detected: NO")

    # ── 8. Duplicates ────────────────────────────────────────────────────────
    if "t-entry" in df.columns:
        dup_count = df["t-entry"].duplicated().sum()
        logger.info(f"\nDuplicate t-entry values: {dup_count:,}")
    else:
        dup_all = df.duplicated().sum()
        logger.info(f"\nFull-row duplicates: {dup_all:,}")

    logger.info("\nInspection complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Lexitron dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data_config.yaml",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    thai_file = cfg["data"]["thai_file"]
    csv_path = raw_dir / thai_file

    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        logger.error(f"Please place the Lexitron CSV at: {csv_path}")
        sys.exit(1)

    inspect_csv(csv_path, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
