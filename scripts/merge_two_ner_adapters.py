#!/usr/bin/env python3
"""Merge adapter_qa and then adapter_drug into the NER base model.

adapter_qa is applied to outputs/merged_model_3 first. adapter_drug is then
applied to the first merged result. Both adapters must use the same label schema.
Existing output directories are never overwritten.

Run from the repository root:
    python scripts/merge_two_ner_adapters.py

Optional paths can be supplied to use a different base, adapters, or outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    outputs = repo_root / "outputs"

    parser = argparse.ArgumentParser(
        description="Sequentially merge adapter_qa, then adapter_drug."
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=outputs / "merged_model_3",
        help="Starting standalone model (default: outputs/merged_model_3).",
    )
    parser.add_argument(
        "--adapter-1",
        type=Path,
        default=outputs / "adapters" / "adapter_qa",
        help="First adapter to merge (default: outputs/adapters/adapter_qa).",
    )
    parser.add_argument(
        "--adapter-2",
        type=Path,
        default=outputs / "adapters" / "adapter_drug",
        help="Second adapter to merge (default: outputs/adapters/adapter_drug).",
    )
    parser.add_argument(
        "--output-1",
        type=Path,
        default=outputs / "merged_model_3_adapter_qa_trial",
        help="Output directory for base model + adapter 1.",
    )
    parser.add_argument(
        "--output-2",
        type=Path,
        default=outputs / "merged_model_3_adapter_qa_drug_trial",
        help="Output directory for output 1 + adapter 2.",
    )
    return parser.parse_args()


def merge_one_stage(
    base_model_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    tokenizer,
    label2id: dict[str, int],
    id2label: dict[int, str],
) -> None:
    if not base_model_dir.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite it: {output_dir}"
        )

    print(f"Loading base model: {base_model_dir}", flush=True)
    # The base checkpoint is an MLM model and may not have an NER classifier.
    # Build the classifier with the adapter's label schema before loading PEFT
    # weights (whose saved classifier is included via modules_to_save).
    model = AutoModelForTokenClassification.from_pretrained(
        base_model_dir,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    print(f"Loading adapter: {adapter_dir}", flush=True)
    model = PeftModel.from_pretrained(model, adapter_dir)

    print(f"Merging and saving to: {output_dir}", flush=True)
    merged_model = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=False)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved: {output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    paths = [
        args.base_model,
        args.adapter_1,
        args.adapter_2,
        args.output_1,
        args.output_2,
    ]
    # Resolve paths up front so relative command-line paths are predictable.
    base_model, adapter_1, adapter_2, output_1, output_2 = [
        path.resolve() for path in paths
    ]

    # Fail before loading large model weights if either output already exists.
    for output_dir in (output_1, output_2):
        if output_dir.exists():
            raise FileExistsError(
                f"Output already exists; refusing to overwrite it: {output_dir}"
            )

    print(f"Loading tokenizer: {base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    schema_paths = [adapter_1 / "label_schema.json", adapter_2 / "label_schema.json"]
    schemas = []
    for schema_path in schema_paths:
        if not schema_path.is_file():
            raise FileNotFoundError(f"Adapter label schema not found: {schema_path}")
        with schema_path.open(encoding="utf-8") as f:
            schemas.append(json.load(f))
    if schemas[0]["label2id"] != schemas[1]["label2id"]:
        raise ValueError("Adapters use different label2id schemas; cannot merge their classifiers safely.")
    label2id = schemas[0]["label2id"]
    id2label = {int(k): v for k, v in schemas[0]["id2label"].items()}

    merge_one_stage(base_model, adapter_1, output_1, tokenizer, label2id, id2label)
    merge_one_stage(output_1, adapter_2, output_2, tokenizer, label2id, id2label)

    print("Sequential adapter merge completed.", flush=True)
    print(f"First-stage model: {output_1}")
    print(f"Final model:       {output_2}")


if __name__ == "__main__":
    main()
