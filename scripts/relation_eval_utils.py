"""Shared relation extraction evaluation helpers for the NER evaluators."""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path


def gold_relations(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve dataset relation entity indices to text, labels, and offsets."""
    entities = item.get("entities", [])
    text = item.get("text", "")
    result = []
    for index, relation in enumerate(item.get("relations", [])):
        try:
            head = entities[int(relation["head"])]
            tail = entities[int(relation["tail"])]
        except (KeyError, IndexError, TypeError, ValueError):
            raise ValueError(f"Relation #{index + 1} has invalid head/tail entity index")
        head_start, head_end = head.get("start"), head.get("end")
        tail_start, tail_end = tail.get("start"), tail.get("end")
        if head_start is None and head.get("text"):
            found = text.find(head["text"])
            if found >= 0:
                head_start, head_end = found, found + len(head["text"])
        if tail_start is None and tail.get("text"):
            found = text.find(tail["text"])
            if found >= 0:
                tail_start, tail_end = found, found + len(tail["text"])
        result.append({
            "head": head.get("text", ""),
            "head_label": head.get("label", ""),
            "head_start": head_start,
            "head_end": head_end,
            "tail": tail.get("text", ""),
            "tail_label": tail.get("label", ""),
            "tail_start": tail_start,
            "tail_end": tail_end,
            "type": relation.get("type", ""),
        })
    return result


def recover_relation_offsets(relations: list[dict[str, Any]], entities: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    """Fill endpoint spans from entity predictions or text occurrence when absent."""
    normalized = []
    for relation in relations:
        relation = dict(relation)
        for endpoint in ("head", "tail"):
            start_key, end_key = f"{endpoint}_start", f"{endpoint}_end"
            if relation.get(start_key) is not None and relation.get(end_key) is not None:
                continue
            matches = [
                entity for entity in entities
                if entity.get("text") == relation.get(endpoint)
                and entity.get("label") == relation.get(f"{endpoint}_label")
            ]
            if matches:
                relation[start_key], relation[end_key] = matches[0].get("start"), matches[0].get("end")
            elif relation.get(endpoint):
                start = text.find(relation[endpoint])
                if start >= 0:
                    relation[start_key], relation[end_key] = start, start + len(relation[endpoint])
        normalized.append(relation)
    return normalized


def score_relations(predictions: list[dict[str, Any]], gold: list[dict[str, Any]], partial: bool) -> dict[str, dict[str, float | int]] | None:
    """Score directed relations; partial requires both endpoint spans to overlap."""
    if not any(item.get("relations") for item in gold):
        return None
    labels = sorted({
        relation.get("type", "")
        for item in (*predictions, *gold)
        for relation in item.get("relations", [])
        if relation.get("type")
    })
    counts = {label: {"TP": 0, "FP": 0, "FN": 0} for label in labels}
    if not labels:
        labels = ["TREATS", "INDICATED_FOR"]
        counts = {label: {"TP": 0, "FP": 0, "FN": 0} for label in labels}

    def endpoint_overlap(pred: dict, target: dict, side: str) -> bool:
        if pred.get(f"{side}_label") != target.get(f"{side}_label"):
            return False
        ps, pe = pred.get(f"{side}_start"), pred.get(f"{side}_end")
        gs, ge = target.get(f"{side}_start"), target.get(f"{side}_end")
        if None not in (ps, pe, gs, ge):
            if partial:
                return max(int(ps), int(gs)) < min(int(pe), int(ge))
            return int(ps) == int(gs) and int(pe) == int(ge)
        return pred.get(side) == target.get(side)

    for pred_item, gold_item in zip(predictions, gold):
        pred_rels = pred_item.get("relations", [])
        gold_rels = gold_item.get("relations", [])
        adjacency: dict[int, list[int]] = {}
        for pi, pred in enumerate(pred_rels):
            adjacency[pi] = [
                gi for gi, target in enumerate(gold_rels)
                if pred.get("type") == target.get("type")
                and endpoint_overlap(pred, target, "head")
                and endpoint_overlap(pred, target, "tail")
            ]
        gold_to_pred: dict[int, int] = {}

        def augment(pi: int, seen: set[int]) -> bool:
            for gi in adjacency[pi]:
                if gi in seen:
                    continue
                seen.add(gi)
                if gi not in gold_to_pred or augment(gold_to_pred[gi], seen):
                    gold_to_pred[gi] = pi
                    return True
            return False

        for pi in range(len(pred_rels)):
            augment(pi, set())
        matched_pred = set(gold_to_pred.values())
        matched_gold = set(gold_to_pred)
        for pi, relation in enumerate(pred_rels):
            label = relation.get("type")
            if label in counts:
                counts[label]["TP" if pi in matched_pred else "FP"] += 1
        for gi, relation in enumerate(gold_rels):
            label = relation.get("type")
            if label in counts and gi not in matched_gold:
                counts[label]["FN"] += 1

    for row in counts.values():
        tp, fp, fn = row["TP"], row["FP"], row["FN"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        row.update({"Precision": p, "Recall": r, "F1": 2 * p * r / (p + r) if p + r else 0.0})
    total = {key: sum(counts[label][key] for label in labels) for key in ("TP", "FP", "FN")}
    p = total["TP"] / (total["TP"] + total["FP"]) if total["TP"] + total["FP"] else 0.0
    r = total["TP"] / (total["TP"] + total["FN"]) if total["TP"] + total["FN"] else 0.0
    total.update({"Precision": p, "Recall": r, "F1": 2 * p * r / (p + r) if p + r else 0.0})
    counts["TOTAL"] = total
    return counts


def print_relation_report(results: dict[str, dict[str, float | int]] | None) -> None:
    if results is None:
        print("\nRelation evaluation skipped: Gold dataset contains no relation annotations.")
        return
    print("\nRelation Extraction Evaluation")
    print("=" * 82)
    print(f"{'Relation':<18}{'TP':>7}{'FP':>7}{'FN':>7}{'Precision':>13}{'Recall':>11}{'F1':>11}")
    print("-" * 82)
    for label, row in results.items():
        print(f"{label:<18}{row['TP']:>7}{row['FP']:>7}{row['FN']:>7}"
              f"{row['Precision'] * 100:>12.2f}%{row['Recall'] * 100:>10.2f}%{row['F1'] * 100:>10.2f}%")


def run_spert_inference(items: list[dict[str, Any]], model_dir: str, quiet: bool = False) -> list[dict[str, Any]]:
    """Batch-infer a SpERT checkpoint using the same decoder as inference_spert.py."""
    import torch
    import yaml
    from transformers import AutoTokenizer
    from scripts.inference_spert import predict
    from src.spert.model import SpERTModel

    directory = Path(model_dir)
    with (directory / "label_config.json").open(encoding="utf-8") as stream:
        labels = json.load(stream)
    with (directory / "run_config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg, data_cfg = config.get("model", {}), config.get("data", {})
    model = SpERTModel(
        model_path=labels.get("model_path", config["model_path"]),
        entity_label_count=len(labels["entity_labels"]),
        relation_label_count=len(labels["relation_labels"]),
        max_span_size=int(data_cfg.get("max_span_size", 10)),
        size_embedding_dim=int(model_cfg.get("size_embedding_dim", 25)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
        relation_chunk_size=int(model_cfg.get("relation_chunk_size", 100)),
    )
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=True)
    threshold = float(labels.get("relation_threshold", config.get("evaluation", {}).get("relation_threshold", 0.4)))
    predictions = []
    try:
        from tqdm import tqdm
        iterator = tqdm(items, desc="Evaluating SpERT", unit="sample", disable=quiet)
    except ImportError:
        iterator = items
    for item in iterator:
        predictions.append(predict(
            model, tokenizer, item["text"], labels["entity_labels"], labels["relation_labels"],
            int(data_cfg.get("max_seq_length", 256)), int(data_cfg.get("max_span_size", 10)),
            threshold, device,
        ))
    return predictions


def run_mtl_inference(items: list[dict[str, Any]], model_dir: str, quiet: bool = False) -> list[dict[str, Any]]:
    """Batch-infer the project's BIO-based MTL NER+RE checkpoint."""
    import torch
    from transformers import AutoTokenizer
    from scripts.inference_mtl import run_inference
    from src.models.mtl_model import build_model

    directory = Path(model_dir)
    with (directory / "label_config.json").open(encoding="utf-8") as stream:
        labels = json.load(stream)
    ner_labels = labels["ner_labels"]
    relation_labels = labels["relation_labels"]
    base_path = labels.get("base_model_path", str(directory))
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = build_model(
        base_model_path=base_path,
        num_ner_labels=len(ner_labels),
        num_rel_labels=len(relation_labels),
    )
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    id2ner = dict(enumerate(ner_labels))
    id2rel = dict(enumerate(relation_labels))
    predictions = []
    try:
        from tqdm import tqdm
        iterator = tqdm(items, desc="Evaluating MTL", unit="sample", disable=quiet)
    except ImportError:
        iterator = items
    for item in iterator:
        result = run_inference(model, tokenizer, item["text"], id2ner, id2rel, device)
        result["relations"] = recover_relation_offsets(
            result.get("relations", []), result.get("entities", []), item["text"]
        )
        predictions.append(result)
    return predictions
