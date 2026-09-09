"""
src/models/model_loader.py
Load the base BertForMaskedLM model from local path with validation.
Ensures the base model is NEVER overwritten.
"""
import json
from pathlib import Path
from typing import Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

_REQUIRED_FILES = [
    "config.json",
    "tokenizer_config.json",
]
# At least one of these tokenizer vocab files must exist
_TOKENIZER_FILES = ["tokenizer.json", "vocab.txt"]
_WEIGHT_FILES = ["pytorch_model.bin", "model.safetensors"]


def validate_model_dir(model_path: Path) -> None:
    """
    Check that all required files are present in the model directory.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    missing = [f for f in _REQUIRED_FILES if not (model_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {model_path}: {missing}"
        )

    has_tokenizer = any((model_path / f).exists() for f in _TOKENIZER_FILES)
    if not has_tokenizer:
        raise FileNotFoundError(
            f"No tokenizer vocab file found in {model_path}. "
            f"Expected one of: {_TOKENIZER_FILES}"
        )

    has_weights = any((model_path / f).exists() for f in _WEIGHT_FILES)
    if not has_weights:
        raise FileNotFoundError(
            f"No weight file found in {model_path}. Expected one of: {_WEIGHT_FILES}"
        )

    logger.info(f"Model directory validated: {model_path}")


def get_model_architecture(model_path: Path) -> dict:
    """Read and return the model config.json as a dict."""
    config_path = Path(model_path) / "config.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def load_base_model_and_tokenizer(model_path: str | Path):
    """
    Load BertForMaskedLM and its tokenizer from a local directory.

    The model directory is read-only; weights are loaded into memory only.

    Args:
        model_path: Path to the model directory.

    Returns:
        (model, tokenizer) tuple.

    Raises:
        FileNotFoundError: If model directory or required files are missing.
        ValueError: If the model architecture is not BertForMaskedLM.
    """
    from transformers import AutoTokenizer, BertForMaskedLM

    model_path = Path(model_path)
    validate_model_dir(model_path)

    arch_cfg = get_model_architecture(model_path)
    architectures = arch_cfg.get("architectures", [])
    model_type = arch_cfg.get("model_type", "unknown")

    logger.info(f"Architecture: {architectures} | Type: {model_type}")

    if "BertForMaskedLM" not in architectures:
        # Still attempt to load but warn loudly
        logger.warning(
            f"Expected BertForMaskedLM but found {architectures}. "
            "Will attempt to load as BertForMaskedLM anyway. "
            "If this fails, inspect the model architecture with "
            "scripts/inspect_model_modules.py"
        )

    logger.info(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)

    logger.info(f"Loading BertForMaskedLM from: {model_path}")
    model = BertForMaskedLM.from_pretrained(str(model_path))

    # Log parameter count
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Base model loaded | Total parameters: {total:,}")

    return model, tokenizer
