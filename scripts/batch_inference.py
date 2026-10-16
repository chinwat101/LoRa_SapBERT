#!/usr/bin/env python3
"""
scripts/batch_inference.py
Run batch NER inference on a JSON dataset (e.g. data/raw/thai_medical_ner_100_text_only.json).

Usage:
    # Default (auto-detect model, input: data/raw/thai_medical_ner_100_text_only.json)
    python scripts/batch_inference.py

    # Specify custom input and output paths
    python scripts/batch_inference.py --input-file data/raw/thai_medical_ner_100_text_only.json --output-file outputs/ner_predictions_100.json

    # Specify model path
    python scripts/batch_inference.py --model outputs/merged_model_1_ner_base --output-file outputs/results.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.inference_ner import find_latest_ner_model, load_ner_model, predict, print_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch NER inference for JSON datasets")
    parser.add_argument(
        "--input-file",
        default="data/raw/thai_medical_ner_100_text_only.json",
        help="Path to JSON file containing texts to predict (default: data/raw/thai_medical_ner_100_text_only.json)",
    )
    parser.add_argument(
        "--output-file",
        default="outputs/batch_ner_predictions.json",
        help="Path to save output JSON predictions (default: outputs/batch_ner_predictions.json)",
    )
    parser.add_argument("--model", default=None, help="Path to standalone merged NER model directory")
    parser.add_argument("--adapter", default=None, help="Path to saved LoRA adapter directory")
    parser.add_argument("--base-model", default=None, help="Path to base model when using adapter")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed print output for each sample")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found!")
        sys.exit(1)

    # 1. Determine model directory
    target_dir = args.model or args.adapter
    if not target_dir:
        latest = find_latest_ner_model()
        if latest:
            target_dir = str(latest)
            print(f"Auto-detected latest NER model: {target_dir}")
        else:
            target_dir = "outputs/wangchan_ner/adapter"

    # 2. Load model
    model, tokenizer, id2label = load_ner_model(target_dir, args.base_model)

    # 3. Read input JSON
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: Input JSON must be a list of objects or strings.")
        sys.exit(1)

    print(f"Loaded {len(data)} items from '{input_path}'")
    print("Running batch NER inference...\n" + "=" * 60)

    results = []
    entity_counts = Counter()
    total_entities = 0

    for idx, item in enumerate(data, 1):
        if isinstance(item, dict):
            text = item.get("text", "")
        else:
            text = str(item)

        if not text:
            continue

        entities = predict(text, model, tokenizer, id2label)
        total_entities += len(entities)

        for ent in entities:
            entity_counts[ent["label"]] += 1

        results.append({
            "id": idx,
            "text": text,
            "entity_count": len(entities),
            "entities": entities,
        })

        if not args.quiet:
            print(f"[{idx}/{len(data)}]")
            print_result(text, entities)

    # 4. Save results to output JSON
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=1, exist_ok=True)

    summary_metadata = {
        "total_samples": len(results),
        "total_entities_found": total_entities,
        "entity_counts_by_type": dict(entity_counts),
        "model_used": target_dir,
    }

    final_output = {
        "metadata": summary_metadata,
        "predictions": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    # 5. Print Summary
    print("=" * 60)
    print("Batch Inference Summary")
    print("=" * 60)
    print(f"Total Texts Processed : {len(results)}")
    print(f"Total Entities Found  : {total_entities}")
    print("Entities Breakdown    :")
    for label, count in entity_counts.items():
        print(f"  - {label:10s} : {count}")
    print(f"\nSaved output to       : {out_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
