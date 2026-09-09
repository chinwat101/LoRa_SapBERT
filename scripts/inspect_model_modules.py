#!/usr/bin/env python3
"""
scripts/inspect_model_modules.py
List all Linear layers in the base model to identify LoRA target_modules.

Usage:
    python scripts/inspect_model_modules.py
    python scripts/inspect_model_modules.py --config configs/model_config.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch.nn as nn

from src.models.model_loader import load_base_model_and_tokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect model layers for LoRA targets")
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()

    with open(Path(args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]

    model_path = cfg["base_model_path"]
    model, _ = load_base_model_and_tokenizer(model_path)

    logger.info("\n" + "=" * 60)
    logger.info("ALL Linear LAYERS (candidate LoRA targets):")
    logger.info("=" * 60)

    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_names.append((name, module.in_features, module.out_features))

    for name, in_f, out_f in linear_names:
        logger.info(f"  {name:<60s}  in={in_f}, out={out_f}")

    # Summarise unique suffixes (useful for target_modules config)
    suffixes = sorted(set(name.split(".")[-1] for name, _, _ in linear_names))
    logger.info("\nUnique layer name suffixes (these are valid target_modules values):")
    for s in suffixes:
        logger.info(f"  - {s}")

    logger.info("\nSuggested lora_config.yaml target_modules:")
    logger.info("  target_modules: ['query', 'key', 'value', 'dense']")


if __name__ == "__main__":
    main()
