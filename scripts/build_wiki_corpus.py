#!/usr/bin/env python3
"""
scripts/build_wiki_corpus.py
ดึง Thai Wikipedia จาก HuggingFace Datasets แล้วแปลงเป็น JSONL format
เหมือนกับ Lexitron corpus เพื่อใช้สำหรับ MLM pretraining

Usage:
    python scripts/build_wiki_corpus.py
    python scripts/build_wiki_corpus.py --max-articles 50000
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("outputs/logs"))


# ──────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_wiki_text(text: str) -> str:
    """Clean Wikipedia markup and normalize whitespace."""
    # Remove section headers (== Header ==)
    text = re.sub(r"={2,}[^=]+={2,}", "", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove [[File:...]] and [[Image:...]]
    text = re.sub(r"\[\[(File|Image|ไฟล์|รูป):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    # Remove wiki links but keep display text [[link|display]] → display
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    # Remove external links
    text = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    # Remove wiki templates {{...}}
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    # Remove references
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/?>", "", text)
    # Remove tables
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_into_sentences(text: str, min_chars: int = 20) -> list[str]:
    """Split text into sentence-like chunks for MLM."""
    # Split on newlines first (paragraphs)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sentences = []
    for para in paragraphs:
        # Split very long paragraphs on Thai sentence endings
        parts = re.split(r"(?<=[ๆ。.!?])\s+", para)
        for part in parts:
            part = part.strip()
            if len(part) >= min_chars:
                sentences.append(part)
    return sentences


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Thai Wikipedia MLM corpus")
    parser.add_argument("--data-config",   default="configs/data_config.yaml")
    parser.add_argument("--max-articles",  type=int, default=None,
                        help="Limit number of Wikipedia articles (default: all)")
    parser.add_argument("--min-chars",     type=int, default=30,
                        help="Min characters per sentence to keep (default: 30)")
    args = parser.parse_args()

    with open(Path(args.data_config), encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)["data"]

    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_file = processed_dir / "wiki_corpus.jsonl"

    # ── Download Thai Wikipedia ───────────────────────────────────────────────
    logger.info("Downloading Thai Wikipedia from HuggingFace...")
    logger.info("(This may take a few minutes on first run — will be cached afterwards)")

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed. Run: pip install datasets")
        sys.exit(1)

    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.th",
        split="train",
        trust_remote_code=True,
    )
    total = len(dataset)
    logger.info(f"Downloaded {total:,} Thai Wikipedia articles")

    if args.max_articles:
        dataset = dataset.select(range(min(args.max_articles, total)))
        logger.info(f"Limited to {len(dataset):,} articles")

    # ── Process and write ─────────────────────────────────────────────────────
    sentence_count = 0
    skipped = 0

    with open(output_file, "w", encoding="utf-8") as out:
        for i, article in enumerate(dataset):
            title = article.get("title", "")
            text  = article.get("text", "")

            if not text.strip():
                skipped += 1
                continue

            cleaned = clean_wiki_text(text)
            sentences = split_into_sentences(cleaned, min_chars=args.min_chars)

            for sent in sentences:
                record = {
                    "text":   sent,
                    "source": "thai_wikipedia",
                    "title":  title,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                sentence_count += 1

            if (i + 1) % 5000 == 0:
                logger.info(f"  Processed {i+1:,}/{len(dataset):,} articles | "
                            f"Sentences so far: {sentence_count:,}")

    logger.info(f"\n✅ Done!")
    logger.info(f"   Articles processed : {len(dataset) - skipped:,}")
    logger.info(f"   Sentences written  : {sentence_count:,}")
    logger.info(f"   Output             : {output_file}")
    logger.info(f"\nNext step: python scripts/build_mlm_dataset.py --include-wiki")


if __name__ == "__main__":
    main()
