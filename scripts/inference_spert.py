#!/usr/bin/env python
"""Run inference with a checkpoint produced by scripts/train_spert.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.spert.model import SpERTModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpERT NER + relation inference")
    parser.add_argument("--model", required=True, help="Trained SpERT output directory")
    parser.add_argument("--text", required=True, help="Input text")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Relation sigmoid threshold; defaults to checkpoint config")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return parser.parse_args()


@torch.no_grad()
def predict(model, tokenizer, text: str, entity_labels: list[str],
            relation_labels: list[str], max_seq_length: int, max_span_size: int,
            threshold: float, device: torch.device) -> dict:
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    valid_positions = [
        i for i, (start, end) in enumerate(offsets)
        if end > start and attention_mask[0, i].item() == 1
    ]

    spans: list[tuple[int, int]] = []
    for start_index, start_pos in enumerate(valid_positions):
        for end_index in range(start_index, min(len(valid_positions), start_index + max_span_size)):
            end_pos = valid_positions[end_index]
            if end_pos - start_pos + 1 != end_index - start_index + 1:
                break
            spans.append((start_pos, end_pos))
    if not spans:
        return {"text": text, "entities": [], "relations": []}

    spans_tensor = torch.tensor(spans, dtype=torch.long, device=device).unsqueeze(0)
    span_widths = (spans_tensor[:, :, 1] - spans_tensor[:, :, 0] + 1)
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "spans": spans_tensor,
        "span_widths": span_widths,
        "span_mask": torch.ones((1, len(spans)), dtype=torch.bool, device=device),
        "relation_pairs": [torch.empty((0, 2), dtype=torch.long, device=device)],
    }
    model.eval()
    outputs = model(batch, calculate_inference_relations=True)
    entity_ids = outputs["entity_logits"][0].argmax(dim=-1).tolist()

    entities = []
    candidate_to_entity: dict[int, int] = {}
    for candidate_idx, class_id in enumerate(entity_ids):
        if class_id == 0:
            continue
        token_start, token_end = spans[candidate_idx]
        char_start = offsets[token_start][0]
        char_end = offsets[token_end][1]
        if char_end <= char_start:
            continue
        candidate_to_entity[candidate_idx] = len(entities)
        entities.append({
            "text": text[char_start:char_end],
            "label": entity_labels[class_id - 1],
            "start": char_start,
            "end": char_end,
        })

    relations = []
    pairs = outputs["inference_pairs"][0]
    logits = outputs["inference_relation_logits"][0]
    if pairs.numel():
        scores = torch.sigmoid(logits)
        for pair_idx, (head_candidate, tail_candidate) in enumerate(pairs.tolist()):
            if head_candidate not in candidate_to_entity or tail_candidate not in candidate_to_entity:
                continue
            head = entities[candidate_to_entity[head_candidate]]
            tail = entities[candidate_to_entity[tail_candidate]]
            for rel_idx, score in enumerate(scores[pair_idx].tolist()):
                if score >= threshold:
                    relations.append({
                        "head": head["text"],
                        "head_label": head["label"],
                        "head_start": head["start"],
                        "head_end": head["end"],
                        "tail": tail["text"],
                        "tail_label": tail["label"],
                        "tail_start": tail["start"],
                        "tail_end": tail["end"],
                        "type": relation_labels[rel_idx],
                        "confidence": round(score, 4),
                    })

    return {"text": text, "entities": entities, "relations": relations}


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model)
    label_config_path = model_dir / "label_config.json"
    run_config_path = model_dir / "run_config.yaml"
    checkpoint_path = model_dir / "model.pt"
    for required in (label_config_path, run_config_path, checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(f"Not a complete SpERT checkpoint; missing {required}")

    with label_config_path.open(encoding="utf-8") as stream:
        labels = json.load(stream)
    with run_config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    entity_labels = labels["entity_labels"]
    relation_labels = labels["relation_labels"]
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    base_model_path = labels.get("model_path", config["model_path"])
    threshold = args.threshold
    if threshold is None:
        threshold = float(labels.get("relation_threshold", config.get("evaluation", {}).get("relation_threshold", 0.4)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = SpERTModel(
        model_path=base_model_path,
        entity_label_count=len(entity_labels),
        relation_label_count=len(relation_labels),
        max_span_size=int(data_cfg.get("max_span_size", 10)),
        size_embedding_dim=int(model_cfg.get("size_embedding_dim", 25)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
        relation_chunk_size=int(model_cfg.get("relation_chunk_size", 100)),
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Device: {device}")

    result = predict(
        model, tokenizer, args.text, entity_labels, relation_labels,
        int(data_cfg.get("max_seq_length", 256)),
        int(data_cfg.get("max_span_size", 10)),
        threshold, device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
