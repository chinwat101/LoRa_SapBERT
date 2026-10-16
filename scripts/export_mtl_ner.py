#!/usr/bin/env python3
"""Export an existing MTL checkpoint as a standalone Hugging Face NER model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.mtl_export import export_ner_standalone
from src.models.mtl_model import build_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/mtl_model", help="MTL checkpoint directory")
    parser.add_argument("--output", default=None, help="Output directory (default: <model>_ner)")
    args = parser.parse_args()

    model_dir = Path(args.model)
    label_path = model_dir / "label_config.json"
    checkpoint_path = model_dir / "model.pt"
    if not label_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Expected an MTL checkpoint with label_config.json and model.pt in {model_dir}"
        )

    labels = json.loads(label_path.read_text(encoding="utf-8"))
    ner_labels = labels["ner_labels"]
    base_model_path = labels["base_model_path"]
    model = build_model(
        base_model_path=base_model_path,
        num_ner_labels=len(ner_labels),
        num_rel_labels=len(labels["relation_labels"]),
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    output_dir = args.output or f"{model_dir}_ner"
    saved_to = export_ner_standalone(
        mtl_model=model,
        tokenizer=tokenizer,
        base_model_path=base_model_path,
        ner_labels=ner_labels,
        output_dir=output_dir,
        source_mtl_dir=model_dir,
    )
    print(f"Standalone NER model saved to: {saved_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
