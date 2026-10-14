#!/usr/bin/env python3
"""
scripts/build_medical_corpus.py
Preprocess train_original.jsonl into token-aware MLM chunks.

What this script does
---------------------
1. Load raw Thai medical articles (train_original.jsonl)
2. Clean each article:
     - Strip footer / contact / doctor-byline blocks
     - Remove markdown image syntax and bare URLs
3. Chunk each article using the REAL tokenizer of the base model:
     - Group paragraphs greedily up to max_content_tokens (default 510)
     - Paragraphs > 510 tokens → sliding window with configurable stride
4. Save chunks as JSONL to data/interim/medical_chunks.jsonl
5. Save train / validation / test splits to data/processed/ as JSONL
   with the same {"text": ...} format used by train_lora_mlm.py

Output files (do NOT overwrite existing train/val/test unless --merge-main)
---------------------------------------------------------------------------
  data/interim/medical_chunks.jsonl     full chunk set (with metadata)
  data/processed/medical_train.jsonl    training split  ({"text": ...})
  data/processed/medical_val.jsonl      validation split
  data/processed/medical_test.jsonl     test split

Usage
-----
  python scripts/build_medical_corpus.py
  python scripts/build_medical_corpus.py --max-tokens 510 --stride 128
  python scripts/build_medical_corpus.py --raw data/raw/train_original.jsonl
  python scripts/build_medical_corpus.py --dry-run          # stats only, no write
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from transformers import AutoTokenizer

from src.data.medical_chunker import ChunkConfig, process_corpus
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__, log_dir=Path("outputs/logs"))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records):,} records → {path}")


def chunks_to_text_only(chunks: list[dict]) -> list[dict]:
    """Convert full chunk dicts to the simple {"text": ...} format for the Trainer."""
    return [{"text": c["text"]} for c in chunks if c.get("text")]


def print_stats(chunks: list[dict]) -> None:
    if not chunks:
        logger.warning("No chunks to summarise.")
        return
    token_counts = [c["token_count"] for c in chunks]
    src_counts: dict[str, int] = {}
    for c in chunks:
        src_counts[c["source_id"]] = src_counts.get(c["source_id"], 0) + 1

    logger.info("=" * 60)
    logger.info("CHUNK STATISTICS")
    logger.info("=" * 60)
    logger.info(f"  Total chunks   : {len(chunks):>8,}")
    logger.info(f"  Min tokens     : {min(token_counts):>8,}")
    logger.info(f"  Max tokens     : {max(token_counts):>8,}")
    logger.info(f"  Mean tokens    : {sum(token_counts)/len(token_counts):>8.1f}")
    logger.info("")
    logger.info("  Chunks by source:")
    for src, cnt in sorted(src_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {cnt:>7,}  {src}")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean & chunk Thai medical articles for MLM pre-training"
    )
    parser.add_argument(
        "--raw",
        default="data/raw/train_original.jsonl",
        help="Input JSONL file (default: data/raw/train_original.jsonl)",
    )
    parser.add_argument(
        "--model-config", default="configs/model_config.yaml",
        help="Model config YAML (used to locate base model / tokenizer)",
    )
    parser.add_argument(
        "--data-config", default="configs/data_config.yaml",
        help="Data config YAML (used for split ratios and output dirs)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=510,
        help="Max content tokens per chunk (default: 510)",
    )
    parser.add_argument(
        "--stride", type=int, default=128,
        help="Sliding-window stride for oversized paragraphs (default: 128)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val/test split (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process and print stats but do NOT write any files",
    )
    parser.add_argument(
        "--tokenizer-path", default=None,
        help="Override tokenizer path (default: read from model_config.yaml)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # ── Load configs ──────────────────────────────────────────────────────────
    with open(args.model_config, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)["model"]
    with open(args.data_config, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)["data"]

    # Tokenizer: prefer extended base (has Thai vocab), fall back to original
    tokenizer_path = (
        args.tokenizer_path
        or model_cfg.get("base_model_path", "data/interim/extended_base")
    )

    raw_path      = Path(args.raw)
    interim_dir   = Path(data_cfg.get("interim_dir", "data/interim"))
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    chunk_file    = interim_dir / "medical_chunks.jsonl"

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not raw_path.exists():
        logger.error(f"Raw file not found: {raw_path}")
        sys.exit(1)

    if not Path(tokenizer_path).exists():
        logger.error(
            f"Tokenizer path not found: {tokenizer_path}\n"
            "Run scripts/extend_tokenizer.py first, or pass --tokenizer-path."
        )
        sys.exit(1)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    logger.info(f"Vocab size: {len(tokenizer):,}")

    # ── Load raw articles ─────────────────────────────────────────────────────
    logger.info(f"Loading articles from: {raw_path}")
    records = load_jsonl(raw_path)
    logger.info(f"Loaded {len(records):,} articles")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    cfg = ChunkConfig(
        max_content_tokens=args.max_tokens,
        stride=args.stride,
    )
    logger.info(
        f"Chunking with max_content_tokens={cfg.max_content_tokens}, "
        f"stride={cfg.stride}"
    )
    chunks = process_corpus(records, tokenizer=tokenizer, cfg=cfg)

    print_stats(chunks)

    if args.dry_run:
        logger.info("--dry-run: no files written.")
        return

    # ── Save full chunk set (with metadata) ───────────────────────────────────
    save_jsonl(chunks, chunk_file)

    # ── Split into train / val / test ─────────────────────────────────────────
    random.seed(args.seed)
    random.shuffle(chunks)

    n       = len(chunks)
    n_train = int(n * data_cfg.get("train_ratio", 0.90))
    n_val   = int(n * data_cfg.get("val_ratio",   0.05))

    train_chunks = chunks[:n_train]
    val_chunks   = chunks[n_train : n_train + n_val]
    test_chunks  = chunks[n_train + n_val :]

    # Save full-metadata splits (useful for analysis)
    save_jsonl(train_chunks, interim_dir / "medical_train_full.jsonl")
    save_jsonl(val_chunks,   interim_dir / "medical_val_full.jsonl")
    save_jsonl(test_chunks,  interim_dir / "medical_test_full.jsonl")

    # Save {"text": ...} splits → ready for train_lora_mlm.py
    save_jsonl(chunks_to_text_only(train_chunks), processed_dir / "medical_train.jsonl")
    save_jsonl(chunks_to_text_only(val_chunks),   processed_dir / "medical_val.jsonl")
    save_jsonl(chunks_to_text_only(test_chunks),  processed_dir / "medical_test.jsonl")

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("OUTPUT SUMMARY")
    logger.info(f"  Full chunks    : {chunk_file}")
    logger.info(f"  Train split    : {processed_dir / 'medical_train.jsonl'}  "
                f"({len(train_chunks):,} chunks)")
    logger.info(f"  Val split      : {processed_dir / 'medical_val.jsonl'}  "
                f"({len(val_chunks):,} chunks)")
    logger.info(f"  Test split     : {processed_dir / 'medical_test.jsonl'}  "
                f"({len(test_chunks):,} chunks)")
    logger.info("")
    logger.info(
        "To include medical data in MLM training, pass the paths above\n"
        "to train_lora_mlm.py via --train-data and --val-data flags,\n"
        "or merge them with the existing Lexitron/Wiki splits."
    )


if __name__ == "__main__":
    main()
