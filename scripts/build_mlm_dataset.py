#!/usr/bin/env python3
"""
scripts/build_mlm_dataset.py
Step 3+4: Build MLM corpus and split into train / val / test JSONL files.

Supports multiple corpus sources:
  - Lexitron (default)
  - Thai Wikipedia (--include-wiki)

Usage:
    python scripts/build_mlm_dataset.py
    python scripts/build_mlm_dataset.py --include-wiki
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.data.corpus_builder import build_corpus
from src.data.splitter import split_by_word, save_jsonl
from src.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("outputs/logs"))


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLM dataset from Lexitron JSONL")
    parser.add_argument("--config",        default="configs/data_config.yaml")
    parser.add_argument("--include-wiki",  action="store_true",
                        help="Merge Thai Wikipedia corpus (requires build_wiki_corpus.py first)")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]

    interim_path  = Path(cfg["interim_dir"]) / "lexitron_clean.jsonl"
    wiki_path     = Path(cfg["processed_dir"]) / "wiki_corpus.jsonl"
    processed_dir = Path(cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not interim_path.exists():
        logger.error(
            f"Interim file not found: {interim_path}\n"
            "Please run: python scripts/preprocess_lexitron.py first."
        )
        sys.exit(1)

    # ── Load Lexitron ─────────────────────────────────────────────────────────
    records = load_jsonl(interim_path)
    logger.info(f"Loaded {len(records):,} Lexitron records from {interim_path}")

    # Build corpus (multiple samples per Lexitron record)
    corpus = build_corpus(records, min_length=cfg.get("min_text_length", 5))
    logger.info(f"Lexitron corpus: {len(corpus):,} samples")

    # ── Merge Wikipedia (optional) ────────────────────────────────────────────
    if args.include_wiki:
        if not wiki_path.exists():
            logger.error(
                f"Wikipedia corpus not found: {wiki_path}\n"
                "Please run: python scripts/build_wiki_corpus.py first."
            )
            sys.exit(1)
        wiki_corpus = load_jsonl(wiki_path)
        logger.info(f"Wikipedia corpus: {len(wiki_corpus):,} samples")
        corpus = corpus + wiki_corpus
        random.seed(args.seed)
        random.shuffle(corpus)
        logger.info(f"Merged corpus: {len(corpus):,} samples total")

    # ── Split ─────────────────────────────────────────────────────────────────
    # For Wikipedia we split corpus directly (no word-level leakage concern since
    # wiki sentences are independent paragraphs)
    if args.include_wiki:
        random.seed(args.seed)
        random.shuffle(corpus)
        n = len(corpus)
        n_train = int(n * cfg.get("train_ratio", 0.90))
        n_val   = int(n * cfg.get("val_ratio",   0.05))
        train   = corpus[:n_train]
        val     = corpus[n_train:n_train + n_val]
        test    = corpus[n_train + n_val:]
    else:
        train, val, test = split_by_word(
            records=records,
            corpus=corpus,
            train_ratio=cfg.get("train_ratio", 0.90),
            val_ratio=cfg.get("val_ratio", 0.05),
            test_ratio=cfg.get("test_ratio", 0.05),
            seed=args.seed,
        )

    # ── Save splits ───────────────────────────────────────────────────────────
    save_jsonl(train, processed_dir / "train.jsonl")
    save_jsonl(val,   processed_dir / "validation.jsonl")
    save_jsonl(test,  processed_dir / "test.jsonl")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("CORPUS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"  Sources    : Lexitron" + (" + Thai Wikipedia" if args.include_wiki else ""))
    logger.info(f"  Train      : {len(train):>7,} samples")
    logger.info(f"  Validation : {len(val):>7,} samples")
    logger.info(f"  Test       : {len(test):>7,} samples")
    logger.info(f"  Total      : {len(train)+len(val)+len(test):>7,} samples")

    logger.info("\nSample training texts:")
    for item in train[:3]:
        logger.info(f"  › {item['text'][:100]}")


if __name__ == "__main__":
    main()

