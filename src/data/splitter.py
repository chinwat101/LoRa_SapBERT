"""
src/data/splitter.py
Train / Validation / Test split with word-level data leakage prevention.

Each unique t-entry (dictionary headword) is assigned to exactly one split,
so all MLM samples derived from the same headword stay together.
"""
import json
import random
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def split_by_word(
    records: list[dict],
    corpus: list[dict],
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split MLM corpus samples such that all samples from the same source word
    land in the same split (prevents data leakage).

    Args:
        records: Preprocessed Lexitron records (each has 'word' key).
        corpus: MLM corpus samples (each has 'text' key).
        train_ratio: Fraction of words for training set.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        seed: Random seed.

    Returns:
        (train_corpus, val_corpus, test_corpus) tuple of sample lists.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    set_seed(seed)

    # Index corpus by word (position in records → list of corpus indices)
    # Build a mapping: word → list of text samples
    word_to_samples: dict[str, list[dict]] = {}

    # We need to know which corpus sample came from which word.
    # Build a fresh per-word sample list by re-running the builder logic.
    from src.data.corpus_builder import build_samples_from_record

    for rec in records:
        word = rec.get("word", "").strip()
        if not word:
            continue
        texts = build_samples_from_record(rec)
        samples = [{"text": t} for t in texts if len(t) >= 5]
        if word not in word_to_samples:
            word_to_samples[word] = []
        word_to_samples[word].extend(samples)

    words = list(word_to_samples.keys())
    random.shuffle(words)

    n_total = len(words)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_words = words[:n_train]
    val_words = words[n_train : n_train + n_val]
    test_words = words[n_train + n_val :]

    def collect(word_list: list[str]) -> list[dict]:
        out = []
        for w in word_list:
            out.extend(word_to_samples.get(w, []))
        return out

    train_corpus = collect(train_words)
    val_corpus = collect(val_words)
    test_corpus = collect(test_words)

    logger.info(
        f"Split (by unique word): "
        f"train={len(train_words)} words / {len(train_corpus)} samples | "
        f"val={len(val_words)} words / {len(val_corpus)} samples | "
        f"test={len(test_words)} words / {len(test_corpus)} samples"
    )

    return train_corpus, val_corpus, test_corpus


def save_jsonl(samples: list[dict], path: Path) -> None:
    """Save a list of sample dicts as JSONL (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(samples):,} samples → {path}")
