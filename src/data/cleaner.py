"""
src/data/cleaner.py
Text cleaning utilities for Thai and mixed-language text.
"""
import re
import html
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Characters to keep: Thai, ASCII alphanumeric, common punctuation
_THAI_RANGE = "\u0e00-\u0e7f"
_KEEP_PATTERN = re.compile(
    rf"[^{_THAI_RANGE}a-zA-Z0-9\s\.,;:!?\(\)\[\]{{}}\"\'\-\/\+\=\@\#\%\&\*]"
)
_MULTI_SPACE = re.compile(r"\s+")
_HTML_TAG = re.compile(r"<[^>]+>")


def remove_html(text: str) -> str:
    """Strip HTML tags and decode HTML entities."""
    if not text:
        return text
    text = _HTML_TAG.sub(" ", text)
    text = html.unescape(text)
    return text


def clean_whitespace(text: str) -> str:
    """Normalise whitespace: collapse multiple spaces/newlines to single space."""
    if not text:
        return text
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def clean_text(
    text: str,
    remove_html_tags: bool = True,
    min_length: int = 2,
) -> Optional[str]:
    """
    Full text cleaning pipeline.

    1. Remove HTML
    2. Normalise whitespace
    3. Return None if text is too short after cleaning

    NOTE: We intentionally do NOT use whitespace splitting for Thai because
    Thai does not use spaces between words. The BERT tokenizer handles this.

    Args:
        text: Raw text string.
        remove_html_tags: Whether to strip HTML (default True).
        min_length: Minimum character length after cleaning.

    Returns:
        Cleaned string, or None if the result is too short / empty.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    if remove_html_tags:
        text = remove_html(text)

    text = clean_whitespace(text)

    if len(text) < min_length:
        return None

    return text


def clean_list_field(raw: str, sep: str = ",") -> list[str]:
    """
    Parse a delimited field (e.g., synonyms) into a list of cleaned strings.

    Args:
        raw: Delimited string from CSV.
        sep: Delimiter character.

    Returns:
        List of non-empty cleaned strings.
    """
    if not raw or not raw.strip():
        return []
    parts = [clean_text(p) for p in raw.split(sep)]
    return [p for p in parts if p]
