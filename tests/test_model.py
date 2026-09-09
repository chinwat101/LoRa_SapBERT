#!/usr/bin/env python3
"""
tests/test_model.py
Unit tests for model loading and LoRA module logic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


class TestModelLoader:
    """Tests for src/models/model_loader.py"""

    def test_validate_missing_dir(self):
        from src.models.model_loader import validate_model_dir
        with pytest.raises(FileNotFoundError):
            validate_model_dir(Path("/nonexistent/path"))

    def test_get_architecture(self, tmp_path):
        import json
        from src.models.model_loader import get_model_architecture
        cfg = {"architectures": ["BertForMaskedLM"], "model_type": "bert", "hidden_size": 768}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        result = get_model_architecture(tmp_path)
        assert result["model_type"] == "bert"
        assert "BertForMaskedLM" in result["architectures"]


class TestLoRA:
    """Tests for src/models/lora.py"""

    def test_get_bert_linear_modules_has_attention(self):
        """Verify we can find linear layers in a tiny BERT-like model."""
        import torch.nn as nn
        from src.models.lora import get_bert_linear_modules

        # Build a tiny mock model with named Linear modules
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.ModuleDict({
                    "query": nn.Linear(32, 32),
                    "key":   nn.Linear(32, 32),
                    "value": nn.Linear(32, 32),
                })

        model = TinyModel()
        names = get_bert_linear_modules(model)
        assert any("query" in n for n in names)
        assert any("key"   in n for n in names)
        assert any("value" in n for n in names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
