"""Evaluate NER predictions with exact entity matching.

Exact Match rules:
  * The predicted entity text must exactly equal the Gold entity text.
  * The entity label must also match.
  * Matching is done within the same sentence.
  * Duplicate entities are counted one-to-one using Counter intersection.

Supported evaluation modes:
  1. Direct Model Inference (End-to-End):
     python scripts/evaluate_extact_match.py --model outputs/merged_model_5_30 --data data/processed/ner_spans.jsonl

  2. Pre-computed Prediction File:
     python scripts/evaluate_extact_match.py --pred outputs/batch_ner_predictions.json --data data/processed/ner_spans.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.relation_eval_utils import gold_relations, print_relation_report, score_relations

LABELS = ("DRUG", "DISEASE", "SYMPTOM")

# Example: [DISEASE ]  'โรคไข้หวัด'  (pos 20–30)
ENTITY_LINE_RE = re.compile(
    r"^\s*\[(DRUG|DISEASE|SYMPTOM)\s*\]\s*'(.*?)'\s*"
    r"\(pos\s*\d+\s*[–—-]\s*\d+\)",
    re.MULTILINE,
)
BLOCK_START_RE = re.compile(r"(?m)^\[(\d+)\s*/\s*(\d+)\]\s*$")
TEXT_LINE_RE = re.compile(r"(?m)^Text:\s*(.*)$")


def load_gold(path: Path, lang: str | None = None) -> list[dict[str, Any]]:
    """Load and validate the Gold dataset from .json or .jsonl file."""
    if not path.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์ Gold: {path}")

    items: list[dict[str, Any]] = []
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    items.append(obj)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSONL บรรทัดที่ {line_no} ไม่ถูกต้อง: {e}")
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            items = data["data"]
        elif isinstance(data, dict) and "gold" in data and isinstance(data["gold"], list):
            items = data["gold"]
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError("Gold JSON ต้องเป็น list ของตัวอย่างข้อความ หรือ dict ที่มี key 'data'/'gold'")

    gold: list[dict[str, Any]] = []
    for i, item in enumerate(items, start=1):
        if not isinstance(item, dict) or "text" not in item:
            raise ValueError(f"Gold item #{i} ต้องมี key 'text'")

        # Optional language filter if specified and present in dataset
        if lang and item.get("lang") and item["lang"] != lang:
            continue

        text = item["text"]
        raw_entities = item.get("entities", [])
        if not isinstance(raw_entities, list):
            raise ValueError(f"Gold item #{i}: 'entities' ต้องเป็น list")

        entities = []
        for j, ent in enumerate(raw_entities, start=1):
            if not isinstance(ent, dict) or "label" not in ent:
                raise ValueError(f"Gold item #{i}, entity #{j} ต้องมี key 'label'")

            ent_text = ent.get("text")
            if not ent_text and "start" in ent and "end" in ent:
                ent_text = text[ent["start"]:ent["end"]]

            if not ent_text:
                continue

            start = ent.get("start")
            end = ent.get("end")
            if start is None or end is None:
                start = text.find(ent_text)
                end = start + len(ent_text) if start >= 0 else None
            entities.append({
                "text": ent_text,
                "label": ent["label"],
                "start": start,
                "end": end,
            })

        gold.append({"text": text, "entities": entities, "relations": gold_relations(item)})

    if not gold:
        raise ValueError(f"ไม่พบข้อมูลในไฟล์ Gold: {path} (อาจถูกกรองด้วยเงื่อนไข)")

    return gold


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Parse sentence blocks and entity lines from .json or .txt prediction file."""
    if not path.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์ Prediction: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "predictions" in data and isinstance(data["predictions"], list):
            raw_list = data["predictions"]
        elif isinstance(data, list):
            raw_list = data
        else:
            raise ValueError("Prediction JSON ต้องเป็น list หรือ dict ที่มี key 'predictions'")

        predictions: list[dict[str, Any]] = []
        for item in raw_list:
            if isinstance(item, dict) and "text" in item:
                prediction = {
                    "text": item["text"],
                    "entities": [
                        {k: e[k] for k in ("text", "label", "start", "end") if k in e}
                        for e in item.get("entities", [])
                        if isinstance(e, dict) and "text" in e and "label" in e
                    ],
                }
                if "relations" in item:
                    prediction["relations"] = item["relations"]
                predictions.append(prediction)
        if not predictions:
            raise ValueError(f"ไม่พบข้อมูลประโยคในไฟล์ Prediction JSON: {path}")
        return predictions

    # Parse as batch inference .txt
    content = path.read_text(encoding="utf-8")
    starts = list(BLOCK_START_RE.finditer(content))
    predictions = []

    for index, match in enumerate(starts):
        block_start = match.end()
        block_end = starts[index + 1].start() if index + 1 < len(starts) else len(content)
        block = content[block_start:block_end]

        text_match = TEXT_LINE_RE.search(block)
        if not text_match:
            continue

        sentence = text_match.group(1).strip()
        entities = [
            {"text": entity_text, "label": label}
            for label, entity_text in ENTITY_LINE_RE.findall(block)
        ]
        predictions.append({"text": sentence, "entities": entities})

    if not predictions:
        raise ValueError(
            "ไม่พบตัวอย่างข้อความในไฟล์ Prediction: ตรวจสอบรูปแบบ [1/100], Text: และบรรทัด entity"
        )
    return predictions


def run_model_inference(
    gold: list[dict[str, Any]],
    model_dir_or_adapter: str,
    base_model_dir: str | None = None,
    max_length: int = 256,
    quiet: bool = False,
) -> list[dict[str, Any]]:
    """Run NER model inference directly on gold sentences."""
    from scripts.inference_ner import load_ner_model, predict

    print(f"Loading model: {model_dir_or_adapter} ...")
    model_dir = Path(model_dir_or_adapter)
    label_config_path = model_dir / "label_config.json"
    checkpoint_path = model_dir / "model.pt"
    run_config_path = model_dir / "run_config.yaml"
    if label_config_path.is_file() and checkpoint_path.is_file() and run_config_path.is_file():
        from scripts.relation_eval_utils import run_spert_inference

        print("Detected SpERT checkpoint; loading entity and relation model.")
        return run_spert_inference(gold, model_dir_or_adapter, quiet=quiet)

    if label_config_path.is_file() and checkpoint_path.is_file():
        with label_config_path.open(encoding="utf-8") as stream:
            checkpoint_config = json.load(stream)
        if "ner_labels" in checkpoint_config and "relation_labels" in checkpoint_config:
            from scripts.relation_eval_utils import run_mtl_inference

            print("Detected MTL checkpoint; loading its encoder and joint NER/RE heads.")
            return run_mtl_inference(gold, model_dir_or_adapter, quiet=quiet)

    model, tokenizer, id2label = load_ner_model(model_dir_or_adapter, base_model_dir)

    predictions: list[dict[str, Any]] = []
    total = len(gold)

    try:
        from tqdm import tqdm
        iterator = tqdm(gold, desc="Evaluating", unit="sample", disable=quiet)
    except ImportError:
        iterator = gold

    for idx, item in enumerate(iterator, start=1):
        text = item["text"]
        pred_entities = predict(text, model, tokenizer, id2label, max_length=max_length)
        predictions.append({
            "text": text,
            "entities": [
                {k: e[k] for k in ("text", "label", "start", "end") if k in e}
                for e in pred_entities
            ],
        })
        if not quiet and "tqdm" not in sys.modules and (idx % 10 == 0 or idx == total):
            print(f"  Processed [{idx}/{total}]")

    return predictions


def score_exact_match(
    predictions: list[dict[str, Any]], gold: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Calculate TP/FP/FN by exact (entity text, label) matches per sentence."""
    if len(predictions) != len(gold):
        raise ValueError(
            f"จำนวนประโยคไม่เท่ากัน: Prediction={len(predictions)}, Gold={len(gold)}"
        )

    counts = {label: {"TP": 0, "FP": 0, "FN": 0} for label in LABELS}

    for index, (pred_item, gold_item) in enumerate(zip(predictions, gold), start=1):
        if pred_item["text"] != gold_item["text"]:
            raise ValueError(
                f"ข้อความในลำดับที่ {index} ไม่ตรงกัน\n"
                f"Prediction: {pred_item['text']}\n"
                f"Gold      : {gold_item['text']}"
            )

        pred_counter = Counter(
            (ent["text"], ent["label"])
            for ent in pred_item["entities"]
            if ent["label"] in LABELS
        )
        gold_counter = Counter(
            (ent["text"], ent["label"])
            for ent in gold_item["entities"]
            if ent["label"] in LABELS
        )

        matched = pred_counter & gold_counter
        extra_pred = pred_counter - gold_counter
        missed_gold = gold_counter - pred_counter

        for (_, label), amount in matched.items():
            counts[label]["TP"] += amount
        for (_, label), amount in extra_pred.items():
            counts[label]["FP"] += amount
        for (_, label), amount in missed_gold.items():
            counts[label]["FN"] += amount

    # Calculate per-label metrics.
    for label in LABELS:
        tp, fp, fn = counts[label]["TP"], counts[label]["FP"], counts[label]["FN"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        counts[label].update({
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })

    total = {
        key: sum(counts[label][key] for label in LABELS)
        for key in ("TP", "FP", "FN")
    }
    tp, fp, fn = total["TP"], total["FP"], total["FN"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total.update({"Precision": precision, "Recall": recall, "F1": f1})
    counts["TOTAL"] = total
    return counts


def print_report(
    predictions: list[dict[str, Any]], gold: list[dict[str, Any]],
    results: dict[str, dict[str, float | int]],
) -> None:
    pred_count = sum(len(item["entities"]) for item in predictions)
    gold_count = sum(len(item["entities"]) for item in gold)

    print("\nExact Match Evaluation")
    print("=" * 74)
    print(f"Sentences             : {len(predictions)}")
    print(f"Predicted entities    : {pred_count}")
    print(f"Gold entities         : {gold_count}")
    if gold_count == 0:
        print("⚠️  คำเตือน: ชุดข้อมูลนี้ไม่มีเฉลย Entity (Gold = 0) ทุกคำทำนายจึงเป็น FP ทั้งหมด")
        print("   (ไฟล์เป็นข้อความดิบ/Text-only เช่น 100_text.json หากจะประเมินผลควรใช้ไฟล์ที่มีเฉลย)")
    print("-" * 74)
    print(f"{'Label':<10}{'TP':>7}{'FP':>7}{'FN':>7}{'Precision':>13}{'Recall':>11}{'F1':>11}")
    print("-" * 74)
    for label in (*LABELS, "TOTAL"):
        row = results[label]
        print(
            f"{label:<10}"
            f"{row['TP']:>7}"
            f"{row['FP']:>7}"
            f"{row['FN']:>7}"
            f"{row['Precision'] * 100:>12.2f}%"
            f"{row['Recall'] * 100:>10.2f}%"
            f"{row['F1'] * 100:>10.2f}%"
        )
    print("-" * 74)
    print("\nFormulas:")
    print("  Precision = TP / (TP + FP)")
    print("  Recall    = TP / (TP + FN)")
    print("  F1        = 2 * Precision * Recall / (Precision + Recall)")
    print("  A zero denominator is reported as 0.00%.")


def print_entity_details(predictions: list[dict[str, Any]], gold: list[dict[str, Any]]) -> None:
    """Print each sentence with its gold and predicted entities and character spans."""
    print("\nEntity Details (Gold vs Prediction)")
    print("=" * 82)
    for index, (pred_item, gold_item) in enumerate(zip(predictions, gold), start=1):
        print(f"[{index}/{len(gold)}] {gold_item['text']}")
        for title, item in (("Gold", gold_item), ("Pred", pred_item)):
            entities = item.get("entities", [])
            print(f"  {title}:")
            if not entities:
                print("    (none)")
                continue
            for entity in entities:
                span = ""
                if entity.get("start") is not None and entity.get("end") is not None:
                    span = f" (pos {entity['start']}-{entity['end']})"
                print(f"    [{entity['label']}] '{entity['text']}'{span}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate NER Exact Match scores directly from model & dataset, or pre-computed prediction file."
    )
    # Model inference arguments
    parser.add_argument("--model", "--weights", "--weight", "--weigth", "-m", default=None, help="Path to standalone merged NER model directory or adapter")
    parser.add_argument("--adapter", default=None, help="Path to saved LoRA adapter directory")
    parser.add_argument("--base-model", default=None, help="Path to base model when using LoRA adapter")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length for tokenizer (default: 256)")

    # Data arguments
    parser.add_argument("--data", "--gold", "-d", dest="data", default=None, type=Path,
                        help="Path to Gold dataset (.json or .jsonl)")
    parser.add_argument("--lang", default=None, choices=["th", "en"],
                        help="Optional language filter for dataset with 'lang' field")

    # Pre-computed prediction argument (optional)
    parser.add_argument("--pred", "-p", default=None, type=Path,
                        help="Optional path to pre-computed prediction file (.json or .txt)")

    # Output & logging
    parser.add_argument("--output", "-o", default=None, type=Path,
                        help="Optional path to save evaluation report and predictions (.json)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    parser.add_argument("--show-entities", action="store_true", help="Print gold and predicted entities for every sentence")

    args = parser.parse_args()

    if not args.data:
        print("Error: ต้องระบุ path ของไฟล์เฉลยด้วย --data หรือ --gold", file=sys.stderr)
        return 2

    try:
        gold = load_gold(args.data, lang=args.lang)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Error loading gold data: {exc}", file=sys.stderr)
        return 1

    if args.pred:
        try:
            predictions = load_predictions(args.pred)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"Error loading predictions: {exc}", file=sys.stderr)
            return 1
    else:
        target_dir = args.model or args.adapter
        if not target_dir:
            from scripts.inference_ner import find_latest_ner_model
            latest = find_latest_ner_model()
            if latest:
                target_dir = str(latest)
                print(f"Auto-detected latest NER model: {target_dir}")
            else:
                target_dir = "outputs/wangchan_ner/adapter"
                print(f"Defaulting to model/adapter: {target_dir}")

        try:
            predictions = run_model_inference(
                gold=gold,
                model_dir_or_adapter=target_dir,
                base_model_dir=args.base_model,
                max_length=args.max_length,
                quiet=args.quiet,
            )
        except Exception as exc:
            print(f"Error running model inference: {exc}", file=sys.stderr)
            return 1

    try:
        results = score_exact_match(predictions, gold)
    except ValueError as exc:
        print(f"Error scoring exact match: {exc}", file=sys.stderr)
        return 1

    print_report(predictions, gold, results)
    if args.show_entities:
        print_entity_details(predictions, gold)
    if any("relations" in item for item in predictions):
        print_relation_report(score_relations(predictions, gold, partial=False))
    elif any(item.get("relations") for item in gold):
        print("\nRelation evaluation skipped: this NER model/prediction file has no relation outputs.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_samples": len(predictions),
            "metrics": results,
            "predictions": predictions,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved evaluation results to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
