"""
src/data/corpus_builder.py
Convert structured Lexitron records into MLM training sentences.

Each dictionary entry can generate multiple training text samples
to maximise data diversity for Masked Language Modeling.
"""
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_samples_from_record(record: dict) -> list[str]:
    """
    Generate multiple natural-language text samples from a single Lexitron record.

    Patterns:
      1. คำว่า {entry}
      2. คำว่า {entry} หมายถึง {definition}
      3. คำว่า {entry} มีคำที่มีความหมายใกล้เคียงคือ {synonym}
      4. คำว่า {entry} มีคำตรงข้ามคือ {antonym}
      5. {example_sentence}   (verbatim)
      6. Combined: entry + definition + synonyms + example

    Args:
        record: Dict with keys: word, definition, synonyms, antonyms, examples.

    Returns:
        List of text strings ready for MLM tokenisation.
    """
    word: str = record.get("word", "").strip()
    definition: str = record.get("definition", "").strip()
    synonyms: list[str] = record.get("synonyms", [])
    antonyms: list[str] = record.get("antonyms", [])
    examples: list[str] = record.get("examples", [])

    samples: list[str] = []

    if not word:
        return samples

    # ── Pattern 1: word only ────────────────────────────────────────────────
    samples.append(f"คำว่า{word}")

    # ── Pattern 2: word + definition ───────────────────────────────────────
    if definition:
        samples.append(f"คำว่า{word}หมายถึง{definition}")

    # ── Pattern 3: word + synonyms ─────────────────────────────────────────
    if synonyms:
        syn_str = "และ".join(synonyms)
        samples.append(f"คำว่า{word}มีคำที่มีความหมายใกล้เคียงคือ{syn_str}")

    # ── Pattern 4: word + antonyms ─────────────────────────────────────────
    if antonyms:
        ant_str = "และ".join(antonyms)
        samples.append(f"คำว่า{word}มีคำตรงข้ามคือ{ant_str}")

    # ── Pattern 5: example sentences (verbatim) ─────────────────────────────
    for ex in examples:
        if ex.strip():
            samples.append(ex.strip())

    # ── Pattern 6: combined context ─────────────────────────────────────────
    parts: list[str] = []
    if definition:
        parts.append(f"คำว่า{word}หมายถึง{definition}")
    if synonyms:
        parts.append(f"คำที่มีความหมายใกล้เคียงได้แก่{'และ'.join(synonyms)}")
    if examples:
        parts.append(f"ตัวอย่างการใช้งาน: {examples[0]}")

    if len(parts) >= 2:
        samples.append(" ".join(parts))

    return samples


def build_corpus(records: list[dict], min_length: int = 5) -> list[dict]:
    """
    Build the full MLM corpus from a list of Lexitron records.

    Args:
        records: List of preprocessed record dicts.
        min_length: Skip samples shorter than this many characters.

    Returns:
        List of {"text": str} dicts ready for JSONL export.
    """
    corpus: list[dict] = []
    skipped = 0

    for rec in records:
        samples = build_samples_from_record(rec)
        for text in samples:
            if len(text) >= min_length:
                corpus.append({"text": text})
            else:
                skipped += 1

    logger.info(
        f"Built {len(corpus):,} MLM samples from {len(records):,} records "
        f"(skipped {skipped} too-short samples)"
    )
    return corpus
