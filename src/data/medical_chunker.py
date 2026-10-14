"""
src/data/medical_chunker.py
Clean and chunk Thai medical articles from train_original.jsonl.

Pipeline per article:
  1. Clean markdown → remove footer / contact blocks
  2. Split into paragraphs
  3. Group consecutive paragraphs into chunks <= max_content_tokens
  4. If a single paragraph exceeds the limit -> sliding window

Output record schema:
  {
    "article_id":  str,
    "chunk_id":    int,
    "source_id":   str,
    "title":       str,
    "token_count": int,
    "text":        str,
  }
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Footer / contact-info patterns to strip (Thai medical sites)
# ──────────────────────────────────────────────────────────────────────────────
_FOOTER_PATTERNS: list[re.Pattern] = [
    # Doctor bylines: "[นายแพทย์..." or "**นพ.**..." lines
    re.compile(r"^\[?(นายแพทย์|แพทย์หญิง|นพ\.|พญ\.)[^\n]*$", re.MULTILINE),
    # Hospital / clinic names on their own line
    re.compile(
        r"^[^\n]*(โรงพยาบาล|คลินิก|ศูนย์การแพทย์|Bumrungrad|BDMS|Samitivej|Bangkok Hospital)[^\n]*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # Markdown image syntax  ![alt](url)
    re.compile(r"!\[[^\]]*\]\([^)]*\)", re.MULTILINE),
    # Bare URLs on their own line
    re.compile(r"^https?://\S+$", re.MULTILINE),
    # Contact / address blocks
    re.compile(
        r"(ติดต่อ|สอบถาม|โทรศัพท์|โทร\.?|Tel|Fax|Line ID|LINE|Facebook|FB|Email|อีเมล)[^\n]*",
        re.IGNORECASE,
    ),
    # "Read more" / call-to-action lines
    re.compile(
        r"^(อ่านเพิ่มเติม|ดูเพิ่มเติม|Read more|บทความที่เกี่ยวข้อง|Related)[^\n]*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # Markdown horizontal rules
    re.compile(r"^---+$", re.MULTILINE),
    # Markdown hyperlinks → keep label text, drop URL
    re.compile(r"\[([^\]]+)\]\(https?://[^)]+\)"),
]

# Strip heading markers (## Heading → Heading)
_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
# Collapse 3+ blank lines → 2
_MULTI_BLANK = re.compile(r"\n{3,}")
# Trailing whitespace per line
_LINE_SPACES = re.compile(r"[ \t]+$", re.MULTILINE)


# ──────────────────────────────────────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_markdown(text: str) -> str:
    """
    Strip footer / contact info from a raw markdown article body.

    Steps
    -----
    1. Remove footer patterns (contacts, doctor bylines, bare URLs, images)
    2. Strip markdown heading markers (keep heading text)
    3. Collapse excessive blank lines
    """
    if not text:
        return ""

    for pat in _FOOTER_PATTERNS:
        # For the hyperlink pattern preserve the label group
        if pat.groups:
            text = pat.sub(r"\1", text)
        else:
            text = pat.sub(" ", text)

    # Strip heading markers but keep the text
    text = _HEADING.sub("", text)

    # Normalise line endings and whitespace
    text = _LINE_SPACES.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_BLANK.sub("\n\n", text)

    return text.strip()


def split_paragraphs(text: str, min_length: int = 10) -> list[str]:
    """
    Split cleaned text into non-empty paragraphs (double-newline separated).

    Args:
        text:       Cleaned article text.
        min_length: Discard paragraphs shorter than this many characters.

    Returns:
        List of paragraph strings.
    """
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if len(p) >= min_length]


# ──────────────────────────────────────────────────────────────────────────────
# Token-aware chunking
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkConfig:
    max_content_tokens: int = 510   # [CLS] + tokens + [SEP] <= 512
    stride: int = 128               # overlap window for long paragraphs
    min_chunk_tokens: int = 20      # drop chunks shorter than this (footer noise)


def _token_len(text: str, tokenizer) -> int:
    """Count subword tokens without adding special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _sliding_window(
    text: str,
    tokenizer,
    cfg: ChunkConfig,
) -> Iterator[str]:
    """
    Yield token-window slices over a paragraph that exceeds the limit.

    Window size = cfg.max_content_tokens
    Step        = cfg.max_content_tokens - cfg.stride
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    step = cfg.max_content_tokens - cfg.stride

    if step <= 0:
        raise ValueError(
            f"stride ({cfg.stride}) must be < max_content_tokens ({cfg.max_content_tokens})"
        )

    start = 0
    while start < len(ids):
        window = ids[start : start + cfg.max_content_tokens]
        chunk_text = tokenizer.decode(window, skip_special_tokens=True).strip()
        if chunk_text:
            yield chunk_text
        if start + cfg.max_content_tokens >= len(ids):
            break
        start += step


def chunk_article(
    article_id: str,
    source_id: str,
    title: str,
    markdown: str,
    tokenizer,
    cfg: ChunkConfig | None = None,
) -> list[dict]:
    """
    Clean and chunk a single article into token-bounded records.

    Algorithm
    ---------
    * Clean markdown (remove footer/contacts)
    * Split into paragraphs
    * Greedily merge consecutive paragraphs until the chunk would exceed
      max_content_tokens, then flush and start a new chunk
    * If a single paragraph already exceeds the limit -> sliding window

    Returns
    -------
    List of chunk dicts with keys:
        article_id, chunk_id, source_id, title, token_count, text
    """
    if cfg is None:
        cfg = ChunkConfig()

    cleaned = clean_markdown(markdown)
    paragraphs = split_paragraphs(cleaned)

    if not paragraphs:
        return []

    chunks: list[dict] = []
    chunk_id = 0
    buffer: list[str] = []
    buffer_tokens = 0

    def _flush(buf: list[str]) -> None:
        nonlocal chunk_id
        if not buf:
            return
        joined = "\n\n".join(buf)
        tc = _token_len(joined, tokenizer)
        if tc < cfg.min_chunk_tokens:
            # Too short → likely a footer fragment; discard
            return
        chunks.append({
            "article_id":  article_id,
            "chunk_id":    chunk_id,
            "source_id":   source_id,
            "title":       title,
            "token_count": tc,
            "text":        joined,
        })
        chunk_id += 1

    for para in paragraphs:
        para_tokens = _token_len(para, tokenizer)

        # ── Case 1: paragraph alone exceeds limit → sliding window ────────────
        if para_tokens > cfg.max_content_tokens:
            _flush(buffer)
            buffer = []
            buffer_tokens = 0

            for window_text in _sliding_window(para, tokenizer, cfg):
                tc = _token_len(window_text, tokenizer)
                if tc < cfg.min_chunk_tokens:
                    continue
                chunks.append({
                    "article_id":  article_id,
                    "chunk_id":    chunk_id,
                    "source_id":   source_id,
                    "title":       title,
                    "token_count": tc,
                    "text":        window_text,
                })
                chunk_id += 1
            continue

        # ── Case 2: adding paragraph would overflow → flush, start fresh ──────
        sep_tokens = 2 if buffer else 0  # "\n\n" costs ~2 tokens
        if buffer_tokens + sep_tokens + para_tokens > cfg.max_content_tokens:
            _flush(buffer)
            buffer = [para]
            buffer_tokens = para_tokens
        else:
            buffer.append(para)
            buffer_tokens += sep_tokens + para_tokens

    # Flush remaining
    _flush(buffer)
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────────────────────────────────────

def process_corpus(
    records: list[dict],
    tokenizer,
    cfg: ChunkConfig | None = None,
    log_every: int = 500,
) -> list[dict]:
    """
    Process a list of raw article records into MLM-ready chunks.

    Args:
        records:   List of dicts with keys: aid, markdown, metadata, source_id.
        tokenizer: Loaded HuggingFace tokenizer (must match base model).
        cfg:       ChunkConfig instance.
        log_every: Log progress every N articles.

    Returns:
        List of chunk dicts.
    """
    if cfg is None:
        cfg = ChunkConfig()

    all_chunks: list[dict] = []
    skipped = 0

    for i, rec in enumerate(records):
        if i > 0 and i % log_every == 0:
            logger.info(
                f"  [{i:,}/{len(records):,}] chunks so far: {len(all_chunks):,}"
            )

        article_id = rec.get("aid", f"article_{i}")
        source_id  = rec.get("source_id", "unknown")
        markdown   = rec.get("markdown", "")
        title      = (rec.get("metadata") or {}).get("title", "") or ""

        if not markdown.strip():
            skipped += 1
            continue

        article_chunks = chunk_article(
            article_id=article_id,
            source_id=source_id,
            title=title,
            markdown=markdown,
            tokenizer=tokenizer,
            cfg=cfg,
        )

        if not article_chunks:
            skipped += 1
            continue

        all_chunks.extend(article_chunks)

    logger.info(
        f"Done | {len(records):,} articles → {len(all_chunks):,} chunks "
        f"({skipped} articles skipped as empty)"
    )
    return all_chunks
