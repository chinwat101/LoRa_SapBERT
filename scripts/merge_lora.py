#!/usr/bin/env python3
"""
scripts/merge_lora.py
Merge LoRA adapter weights into the base model and save a full merged model.

IMPORTANT:
  - The original base model is NEVER modified.
  - The merged model is saved to outputs/merged_model/.
  - Use this only if you need a self-contained deployable model.
  - For inference with adapter, use load_lora_adapter() instead.

Usage:
    python scripts/merge_lora.py
    python scripts/merge_lora.py --config configs/model_config.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.models.model_loader import load_base_model_and_tokenizer
from src.models.lora import load_lora_adapter
from src.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("outputs/logs"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()

    with open(Path(args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]

    base_path    = cfg["base_model_path"]
    adapter_dir  = Path(cfg["adapter_output_dir"])
    base_merged_dir = Path(cfg["merged_model_dir"])
    
    # Auto-increment the output directory to separate each run
    run_num = 1
    while True:
        merged_dir = base_merged_dir.with_name(f"{base_merged_dir.name}_{run_num}")
        if not merged_dir.exists():
            break
        run_num += 1

    if not adapter_dir.exists():
        logger.error(
            f"Adapter not found: {adapter_dir}\n"
            "Run training first: python scripts/train_lora_mlm.py"
        )
        sys.exit(1)

    # Load base model (clean, unmodified)
    logger.info(f"Loading base model from: {base_path}")
    base_model, tokenizer = load_base_model_and_tokenizer(base_path)

    # Load LoRA adapter
    logger.info(f"Loading adapter from: {adapter_dir}")
    lora_model = load_lora_adapter(base_model, adapter_dir)

    # Merge LoRA weights into base
    logger.info("Merging LoRA into base model...")
    merged_model = lora_model.merge_and_unload()

    # Save merged model
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    logger.info(f"Merged model saved to: {merged_dir}")
    logger.info("NOTE: The original base model at:")
    logger.info(f"  {base_path}")
    logger.info("  has NOT been modified.")


if __name__ == "__main__":
    main()
