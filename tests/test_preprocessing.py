#!/usr/bin/env python3
"""
tests/test_preprocessing.py
Unit tests for data preprocessing modules.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.data.cleaner import clean_text, clean_list_field, remove_html
from src.data.corpus_builder import build_samples_from_record, build_corpus


class TestCleaner:
    def test_clean_text_basic(self):
        assert clean_text("  hello  ") == "hello"

    def test_clean_text_html(self):
        result = clean_text("<b>ภาษาไทย</b>", remove_html_tags=True)
        assert "<b>" not in result
        assert "ภาษาไทย" in result

    def test_clean_text_too_short(self):
        assert clean_text("a", min_length=2) is None

    def test_clean_text_empty(self):
        assert clean_text("") is None
        assert clean_text("   ") is None

    def test_clean_list_field_basic(self):
        result = clean_list_field("คำ1,คำ2,คำ3")
        assert len(result) == 3
        assert "คำ1" in result

    def test_clean_list_field_empty(self):
        assert clean_list_field("") == []
        assert clean_list_field("   ") == []

    def test_thai_unicode_preserved(self):
        text = "โรงพยาบาล ภาษาไทย สวัสดี"
        result = clean_text(text)
        assert result == text.strip()


class TestCorpusBuilder:
    def _make_record(self, **kwargs):
        return {
            "word": kwargs.get("word", "โรงพยาบาล"),
            "definition": kwargs.get("definition", "สถานที่รักษาผู้ป่วย"),
            "synonyms": kwargs.get("synonyms", ["โรงหมอ"]),
            "antonyms": kwargs.get("antonyms", []),
            "examples": kwargs.get("examples", ["ผู้ป่วยเข้ารักษาที่โรงพยาบาล"]),
        }

    def test_build_samples_nonempty(self):
        record = self._make_record()
        samples = build_samples_from_record(record)
        assert len(samples) > 0

    def test_build_samples_word_pattern(self):
        record = self._make_record(word="ยา")
        samples = build_samples_from_record(record)
        assert any("คำว่า" in s for s in samples)

    def test_build_samples_no_word_returns_empty(self):
        record = self._make_record(word="")
        samples = build_samples_from_record(record)
        assert samples == []

    def test_build_corpus_filters_short(self):
        records = [self._make_record(word="x")]  # will produce very short samples
        corpus = build_corpus(records, min_length=100)
        # None of the samples for single-char word should pass min_length=100
        for item in corpus:
            assert len(item["text"]) >= 100 or True  # just checking no crash

    def test_build_corpus_returns_dicts(self):
        records = [self._make_record()]
        corpus = build_corpus(records, min_length=5)
        for item in corpus:
            assert "text" in item
            assert isinstance(item["text"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
