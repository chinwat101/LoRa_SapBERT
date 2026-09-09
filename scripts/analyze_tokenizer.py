#!/usr/bin/env python3
"""
scripts/analyze_tokenizer.py
Step 8: Analyse tokenizer coverage on Thai text corpus.

Reports:
  - UNK count and rate
  - Average tokens per sentence
  - Max tokens
  - Percentile sequence lengths
  - Example tokenizations

Usage:
    python scripts/analyze_tokenizer.py
    python scripts/analyze_tokenizer.py --config configs/model_config.yaml \
        --data configs/data_config.yaml --split train
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

from src.models.model_loader import load_base_model_and_tokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("outputs/logs"))


def load_texts(jsonl_path: Path, max_samples: int = 5000) -> list[str]:
    texts = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            obj = json.loads(line.strip())
            if obj.get("text"):
                texts.append(obj["text"])
    return texts


def analyze(tokenizer, texts: list[str], max_length: int = 512) -> None:
    logger.info(f"Analysing {len(texts):,} texts...")

    unk_id = tokenizer.unk_token_id
    token_lengths = []
    unk_counts = []
    example_tokens = []

    for i, text in enumerate(texts):
        enc = tokenizer(text, truncation=False, add_special_tokens=True)
        ids = enc["input_ids"]
        token_lengths.append(len(ids))
        unk_counts.append(ids.count(unk_id) if unk_id is not None else 0)

        if i < 5:
            tokens = tokenizer.convert_ids_to_tokens(ids)
            example_tokens.append((text, tokens))

    lengths = np.array(token_lengths)
    unks = np.array(unk_counts)
    total_tokens = lengths.sum()
    total_unk = unks.sum()

    logger.info("\n" + "=" * 60)
    logger.info("TOKENIZER ANALYSIS REPORT")
    logger.info("=" * 60)
    logger.info(f"  Vocab size         : {tokenizer.vocab_size:,}")
    logger.info(f"  UNK token          : {tokenizer.unk_token!r} (id={unk_id})")
    logger.info(f"  Samples analysed   : {len(texts):,}")
    logger.info(f"  Total tokens       : {total_tokens:,}")
    logger.info(f"  Total [UNK]        : {int(total_unk):,}")
    logger.info(f"  UNK rate           : {100*total_unk/total_tokens:.2f}%")
    logger.info(f"\n  Sequence length stats:")
    logger.info(f"    Mean             : {lengths.mean():.1f}")
    logger.info(f"    Std              : {lengths.std():.1f}")
    logger.info(f"    Min              : {int(lengths.min())}")
    logger.info(f"    Max              : {int(lengths.max())}")
    for pct in [50, 75, 90, 95, 99]:
        logger.info(f"    P{pct:<2d}             : {int(np.percentile(lengths, pct))}")

    exceed = (lengths > max_length).sum()
    logger.info(f"\n  Sequences > {max_length} tokens: {int(exceed):,} ({100*exceed/len(texts):.1f}%)")

    logger.info("\n  Example tokenisations (first 5):")
    for text, tokens in example_tokens:
        logger.info(f"  Text  : {text[:80]!r}")
        logger.info(f"  Tokens: {tokens}")
        logger.info("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse tokenizer on Thai corpus")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--data", default="configs/data_config.yaml")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    with open(Path(args.config), encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)["model"]
    with open(Path(args.data), encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)["data"]

    split_file = Path(data_cfg["processed_dir"]) / f"{args.split}.jsonl"
    if not split_file.exists():
        logger.error(
            f"Split file not found: {split_file}\n"
            "Run: python scripts/build_mlm_dataset.py first."
        )
        sys.exit(1)

    _, tokenizer = load_base_model_and_tokenizer(model_cfg["base_model_path"])
    texts = load_texts(split_file, max_samples=args.max_samples)
    analyze(tokenizer, texts, max_length=args.max_length)


if __name__ == "__main__":
    main()
