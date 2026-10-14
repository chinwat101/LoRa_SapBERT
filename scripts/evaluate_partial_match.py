#!/usr/bin/env python3
"""Evaluate NER predictions with Partial Match (one-to-one span overlap).

Matching rules:
  * Predicted entity label and Gold label must be equal.
  * Predicted and Gold spans must overlap by at least one character.
  * Matching is performed within the same sentence.
  * One prediction can match at most one Gold entity, and vice versa.
  * Maximum bipartite matching is used to maximize the number of TP matches.

Supported evaluation modes:
  1. Direct Model Inference (End-to-End):
     python scripts/evaluate_partial_match.py --model outputs/merged_model_3_ner_1 --data data/raw/text100_answer.json

  2. Pre-computed Prediction File:
     python scripts/evaluate_partial_match.py --pred outputs/predictions_100.json --data data/raw/text100_answer.json
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

LABELS = ("DRUG", "DISEASE", "SYMPTOM")
BLOCK_START_RE = re.compile(r"(?m)^\[(\d+)\s*/\s*(\d+)\]\s*$")
TEXT_LINE_RE = re.compile(r"(?m)^Text:\s*(.*)$")
ENTITY_LINE_RE = re.compile(
    r"(?m)^\s*\[(DRUG|DISEASE|SYMPTOM)\s*\]\s*'(.*?)'\s*"
    r"\(pos\s*(\d+)\s*[–—-]\s*(\d+)\s*\)"
)


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
            if ent["label"] not in LABELS:
                continue

            ent_text = ent.get("text")
            start = ent.get("start")
            end = ent.get("end")

            if not ent_text and start is not None and end is not None:
                ent_text = text[start:end]

            if not ent_text and (start is None or end is None):
                continue

            entities.append({
                "text": ent_text,
                "label": ent["label"],
                "start": int(start) if start is not None else None,
                "end": int(end) if end is not None else None,
            })

        gold.append({"text": text, "entities": entities})

    if not gold:
        raise ValueError(f"ไม่พบข้อมูลในไฟล์ Gold: {path}")

    return gold


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Parse sentence blocks and entity spans from .json or .txt prediction file."""
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
                text = item["text"]
                entities = []
                for e in item.get("entities", []):
                    if not isinstance(e, dict) or "label" not in e:
                        continue
                    if e["label"] not in LABELS:
                        continue
                    ent_text = e.get("text", "")
                    start = e.get("start")
                    end = e.get("end")

                    # Recover start/end if missing
                    if (start is None or end is None) and ent_text:
                        pos = text.find(ent_text)
                        if pos >= 0:
                            start = pos
                            end = pos + len(ent_text)

                    if start is not None and end is not None:
                        entities.append({
                            "text": ent_text,
                            "label": e["label"],
                            "start": int(start),
                            "end": int(end),
                        })
                predictions.append({"text": text, "entities": entities})

        if not predictions:
            raise ValueError(f"ไม่พบข้อมูลประโยคในไฟล์ Prediction JSON: {path}")
        return predictions

    # Otherwise parse as batch inference .txt
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
        entities = []
        for label, entity_text, start, end in ENTITY_LINE_RE.findall(block):
            start_i, end_i = int(start), int(end)
            if end_i < start_i:
                raise ValueError(f"Span ไม่ถูกต้อง: {entity_text!r} ({start_i}, {end_i})")
            entities.append({
                "text": entity_text,
                "label": label,
                "start": start_i,
                "end": end_i,
            })
        predictions.append({"text": sentence, "entities": entities})
    if not predictions:
        raise ValueError("ไม่พบรายการ Prediction: ตรวจสอบรูปแบบ [1/100], Text: และบรรทัด entity")
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
    model, tokenizer, id2label = load_ner_model(model_dir_or_adapter, base_model_dir)

    predictions: list[dict[str, Any]] = []
    total = len(gold)

    try:
        from tqdm import tqdm
        iterator = tqdm(gold, desc="Evaluating (Partial Match)", unit="sample", disable=quiet)
    except ImportError:
        iterator = gold

    for idx, item in enumerate(iterator, start=1):
        text = item["text"]
        pred_entities = predict(text, model, tokenizer, id2label, max_length=max_length)
        predictions.append({
            "text": text,
            "entities": [
                {
                    "text": e["text"],
                    "label": e["label"],
                    "start": int(e["start"]),
                    "end": int(e["end"]),
                }
                for e in pred_entities
                if "start" in e and "end" in e
            ],
        })
        if not quiet and "tqdm" not in sys.modules and (idx % 10 == 0 or idx == total):
            print(f"  Processed [{idx}/{total}]")

    return predictions


def all_occurrences(sentence: str, phrase: str) -> list[tuple[int, int]]:
    """Return every exact occurrence as [start, end), including overlapping ones."""
    if not phrase:
        return []
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = sentence.find(phrase, cursor)
        if start < 0:
            break
        spans.append((start, start + len(phrase)))
        cursor = start + 1
    return spans


def intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def recover_gold_spans(sentence: str, entities: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Recover Gold offsets by assigning exact occurrences to non-overlapping spans.

    If entities already have valid integer offsets, they are preserved directly.
    Otherwise, backtracking finds an assignment covering as many Gold entities as
    possible without assigning overlapping spans to different Gold mentions.
    """
    if all(ent.get("start") is not None and ent.get("end") is not None for ent in entities):
        return [
            {"text": ent["text"], "label": ent["label"], "start": ent["start"], "end": ent["end"]}
            for ent in entities
        ], []

    candidates = [all_occurrences(sentence, str(ent["text"])) for ent in entities]
    n = len(entities)
    order = sorted(range(n), key=lambda i: (len(candidates[i]), -len(str(entities[i]["text"])), i))
    best_assignment: dict[int, tuple[int, int]] = {}
    best_score: tuple[int, int] = (-1, 10**9)  # assigned count, inversions
    current: dict[int, tuple[int, int]] = {}

    def inversions(assignment: dict[int, tuple[int, int]]) -> int:
        ordered_indices = sorted(assignment)
        starts = [assignment[i][0] for i in ordered_indices]
        return sum(starts[a] > starts[b] for a in range(len(starts)) for b in range(a + 1, len(starts)))

    def search(depth: int) -> None:
        nonlocal best_assignment, best_score
        if len(current) + (n - depth) < best_score[0]:
            return
        if depth == n:
            inv = inversions(current)
            score = (len(current), inv)
            if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
                best_score = score
                best_assignment = dict(current)
            return

        entity_index = order[depth]
        for span in candidates[entity_index]:
            if all(not intervals_overlap(span, used_span) for used_span in current.values()):
                current[entity_index] = span
                search(depth + 1)
                del current[entity_index]
        search(depth + 1)

    search(0)
    recovered: list[dict[str, Any]] = []
    missing: list[str] = []
    for i, ent in enumerate(entities):
        item = {"text": ent["text"], "label": ent["label"]}
        if i in best_assignment:
            item["start"], item["end"] = best_assignment[i]
        elif ent.get("start") is not None and ent.get("end") is not None:
            item["start"], item["end"] = ent["start"], ent["end"]
        else:
            item["start"], item["end"] = None, None
            missing.append(f"{ent['label']}:{ent['text']}")
        recovered.append(item)
    return recovered, missing


def maximum_bipartite_matching(pred_entities: list[dict[str, Any]], gold_entities: list[dict[str, Any]]) -> dict[int, int]:
    """Return prediction-index -> Gold-index for a maximum 1:1 overlap matching."""
    adjacency: dict[int, list[int]] = {}
    for pi, pred in enumerate(pred_entities):
        edges = []
        pspan = (pred["start"], pred["end"])
        for gi, gold in enumerate(gold_entities):
            if gold["start"] is None or pred["label"] != gold["label"]:
                continue
            gspan = (gold["start"], gold["end"])
            if intervals_overlap(pspan, gspan):
                edges.append(gi)
        edges.sort(key=lambda gi: -max(0, min(pred["end"], gold_entities[gi]["end"]) - max(pred["start"], gold_entities[gi]["start"])))
        adjacency[pi] = edges

    gold_to_pred: dict[int, int] = {}

    def augment(pi: int, seen_gold: set[int]) -> bool:
        for gi in adjacency[pi]:
            if gi in seen_gold:
                continue
            seen_gold.add(gi)
            if gi not in gold_to_pred or augment(gold_to_pred[gi], seen_gold):
                gold_to_pred[gi] = pi
                return True
        return False

    for pi in range(len(pred_entities)):
        augment(pi, set())
    return {pi: gi for gi, pi in gold_to_pred.items()}


def calculate_metrics(counts: dict[str, dict[str, int]]) -> None:
    for label in counts:
        tp, fp, fn = counts[label]["TP"], counts[label]["FP"], counts[label]["FN"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        counts[label].update({"Precision": p, "Recall": r, "F1": f1})


def score_partial_match(predictions: list[dict[str, Any]], gold: list[dict[str, Any]]) -> tuple[dict[str, dict[str, float | int]], int, list[str]]:
    if len(predictions) != len(gold):
        raise ValueError(f"จำนวนประโยคไม่เท่ากัน: Prediction={len(predictions)}, Gold={len(gold)}")

    counts: dict[str, dict[str, int]] = {label: {"TP": 0, "FP": 0, "FN": 0} for label in LABELS}
    unresolved_total = 0
    warnings: list[str] = []

    for sent_index, (pred_item, gold_item) in enumerate(zip(predictions, gold), start=1):
        if pred_item["text"] != gold_item["text"]:
            raise ValueError(
                f"ข้อความในลำดับที่ {sent_index} ไม่ตรงกัน\n"
                f"Prediction: {pred_item['text']}\nGold      : {gold_item['text']}"
            )

        pred_entities = pred_item["entities"]
        gold_entities, missing = recover_gold_spans(gold_item["text"], gold_item["entities"])
        unresolved_total += len(missing)
        if missing:
            warnings.append(f"ประโยค {sent_index}: วางตำแหน่ง Gold ไม่ได้ {len(missing)} entity(s): {', '.join(missing)}")

        matching = maximum_bipartite_matching(pred_entities, gold_entities)
        matched_gold = set(matching.values())
        matched_pred = set(matching.keys())

        for pi, pred in enumerate(pred_entities):
            if pi in matched_pred:
                counts[pred["label"]]["TP"] += 1
            else:
                counts[pred["label"]]["FP"] += 1
        for gi, g in enumerate(gold_entities):
            if gi not in matched_gold:
                counts[g["label"]]["FN"] += 1

    calculate_metrics(counts)
    total = {k: sum(counts[label][k] for label in LABELS) for k in ("TP", "FP", "FN")}
    total_p = total["TP"] / (total["TP"] + total["FP"]) if total["TP"] + total["FP"] else 0.0
    total_r = total["TP"] / (total["TP"] + total["FN"]) if total["TP"] + total["FN"] else 0.0
    total_f1 = 2 * total_p * total_r / (total_p + total_r) if total_p + total_r else 0.0
    total.update({"Precision": total_p, "Recall": total_r, "F1": total_f1})
    counts["TOTAL"] = total
    return counts, unresolved_total, warnings


def print_report(predictions: list[dict[str, Any]], gold: list[dict[str, Any]], results: dict[str, dict[str, float | int]], unresolved: int) -> None:
    pred_count = sum(len(item["entities"]) for item in predictions)
    gold_count = sum(len(item["entities"]) for item in gold)
    print("\nPartial Match Evaluation (1:1 span overlap)")
    print("=" * 82)
    print(f"Sentences              : {len(predictions)}")
    print(f"Predicted entities     : {pred_count}")
    print(f"Gold entities          : {gold_count}")
    print(f"Gold spans unresolved  : {unresolved}")
    print("-" * 82)
    print(f"{'Label':<10}{'TP':>7}{'FP':>7}{'FN':>7}{'Precision':>13}{'Recall':>11}{'F1':>11}")
    print("-" * 82)
    for label in (*LABELS, "TOTAL"):
        row = results[label]
        print(
            f"{label:<10}{row['TP']:>7}{row['FP']:>7}{row['FN']:>7}"
            f"{row['Precision'] * 100:>12.2f}%{row['Recall'] * 100:>10.2f}%{row['F1'] * 100:>10.2f}%"
        )
    print("-" * 82)
    print("\nRules and formulas:")
    print("  A match requires the same label and at least one overlapping character.")
    print("  Matching is one-to-one; maximum bipartite matching maximizes TP.")
    print("  Precision = TP / (TP + FP)")
    print("  Recall    = TP / (TP + FN)")
    print("  F1        = 2 * Precision * Recall / (Precision + Recall)")
    print("  A zero denominator is reported as 0.00%.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate NER Partial Match scores using 1:1 span overlap directly from model or prediction file."
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
    parser.add_argument("--show-warnings", action="store_true", help="Show warnings for Gold entities without recoverable spans")

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
        results, unresolved, warnings = score_partial_match(predictions, gold)
    except ValueError as exc:
        print(f"Error scoring partial match: {exc}", file=sys.stderr)
        return 1

    print_report(predictions, gold, results, unresolved)

    if args.show_warnings and warnings:
        print("\nGold span recovery warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    elif unresolved:
        print("\nWarning: บาง Gold span กู้คืนตำแหน่งไม่ได้ ใช้ --show-warnings เพื่อดูรายละเอียด")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_samples": len(predictions),
            "unresolved_gold_spans": unresolved,
            "metrics": results,
            "predictions": predictions,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved evaluation results to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
