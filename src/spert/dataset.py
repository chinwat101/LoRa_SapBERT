"""Dataset adapter for the project's character-offset JSONL annotations."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class SpertJsonlDataset(Dataset):
    """Tokenize annotated text and enumerate candidate token spans.

    Entity offsets in the source JSONL are character offsets with exclusive
    ends. Conversion to token spans happens in memory; the source file is
    never modified and no converted dataset is written to disk.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        entity_labels: list[str],
        relation_labels: list[str],
        max_seq_length: int = 256,
        max_span_size: int = 10,
        neg_entity_count: int = 100,
        neg_relation_count: int = 100,
        seed: int = 42,
    ) -> None:
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.entity_labels = entity_labels
        self.relation_labels = relation_labels
        self.entity2id = {label: i + 1 for i, label in enumerate(entity_labels)}
        self.relation2id = {label: i for i, label in enumerate(relation_labels)}
        self.max_seq_length = max_seq_length
        self.max_span_size = max_span_size
        self.neg_entity_count = neg_entity_count
        self.neg_relation_count = neg_relation_count
        self.seed = seed

        if not tokenizer.is_fast:
            raise ValueError("SpERT data alignment requires a fast tokenizer with offset mapping.")
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        with self.data_path.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if not records:
            raise ValueError(f"Dataset contains no JSONL records: {self.data_path}")

        self.alignment_stats = {
            "unrepresentable_entities": 0,
            "token_span_label_collisions": 0,
        }
        self.features = [self._encode_record(record, i) for i, record in enumerate(records)]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]

    @staticmethod
    def _char_span_to_tokens(
        char_start: int,
        char_end: int,
        offsets: list[tuple[int, int]],
        valid_token_positions: list[int],
    ) -> tuple[int, int] | None:
        overlapping = [
            token_pos
            for token_pos in valid_token_positions
            if offsets[token_pos][0] < char_end and offsets[token_pos][1] > char_start
        ]
        if not overlapping:
            return None
        return overlapping[0], overlapping[-1]

    def _encode_record(self, record: dict[str, Any], record_index: int) -> dict[str, Any]:
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Record {record_index} has no string 'text' field.")

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        input_ids = encoding["input_ids"]
        offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
        attention_mask = encoding["attention_mask"]
        valid_positions = [
            i for i, (start, end) in enumerate(offsets)
            if end > start and attention_mask[i] == 1
        ]
        if not valid_positions:
            raise ValueError(f"Record {record_index} produced no text tokens.")

        # Enumerate every contiguous span of real tokens up to max_span_size.
        # Special tokens are excluded; each span is represented by inclusive
        # token start/end indices, matching the original SpERT formulation.
        spans: list[tuple[int, int]] = []
        for start_index, start_pos in enumerate(valid_positions):
            for end_index in range(start_index, min(len(valid_positions), start_index + self.max_span_size)):
                end_pos = valid_positions[end_index]
                # Refuse spans crossing special/padded positions.
                if end_pos - start_pos + 1 != end_index - start_index + 1:
                    break
                spans.append((start_pos, end_pos))
        span_to_index = {span: i for i, span in enumerate(spans)}

        entities = record.get("entities", [])
        entity_span_indices: dict[int, int] = {}
        entity_labels = [0] * len(spans)  # class 0 means "not an entity"
        span_owners: dict[int, tuple[int, str, int]] = {}
        for entity_index, entity in enumerate(entities):
            label = entity.get("label")
            if label not in self.entity2id:
                raise ValueError(
                    f"Record {record_index}, entity {entity_index}: unknown label {label!r}."
                )
            char_start, char_end = int(entity["start"]), int(entity["end"])
            if char_end > offsets[valid_positions[-1]][1]:
                # The entity is cut off by max_seq_length; do not label a
                # truncated fragment as a complete gold entity.
                self.alignment_stats["unrepresentable_entities"] += 1
                continue
            token_span = self._char_span_to_tokens(
                char_start, char_end, offsets, valid_positions
            )
            if token_span is None or token_span not in span_to_index:
                # Entity fell outside the truncated sequence or is longer than
                # the configured span width. It cannot contribute to this run.
                self.alignment_stats["unrepresentable_entities"] += 1
                continue
            span_index = span_to_index[token_span]
            previous = span_owners.get(span_index)
            if previous is not None and previous[1] != label:
                # A subword can cover a nested/overlapping character mention
                # entirely (e.g. a symptom inside a disease term). SpERT's
                # token-span representation cannot distinguish both bounds.
                # Keep the longer annotation deterministically; relations to
                # a discarded mention will be skipped below.
                self.alignment_stats["token_span_label_collisions"] += 1
                if char_end - char_start <= previous[2]:
                    continue
                entity_span_indices.pop(previous[0], None)
            elif previous is not None:
                entity_span_indices[entity_index] = span_index
                continue

            span_owners[span_index] = (entity_index, label, char_end - char_start)
            entity_span_indices[entity_index] = span_index
            entity_labels[span_index] = self.entity2id[label]

        gold_entity_records = [
            (spans[span_index][0], spans[span_index][1], label)
            for span_index, (_, label, _) in span_owners.items()
        ]

        # For entity classification, train on every positive span and a fixed
        # random sample of non-entity spans. All candidates remain available
        # for validation-time decoding.
        positive_indices = [i for i, label in enumerate(entity_labels) if label > 0]
        negative_indices = [i for i, label in enumerate(entity_labels) if label == 0]
        rng = random.Random(self.seed + record_index)
        sampled_negatives = rng.sample(
            negative_indices, min(self.neg_entity_count, len(negative_indices))
        )
        entity_loss_mask = [False] * len(spans)
        for span_index in positive_indices + sampled_negatives:
            entity_loss_mask[span_index] = True

        # Build directed relation pairs over annotated entities. Each relation
        # target is multi-hot; unannotated ordered pairs are sampled as hard
        # negatives (NO_RELATION is represented by an all-zero vector).
        relation_targets_by_pair: dict[tuple[int, int], list[float]] = {}
        positive_pair_keys: set[tuple[int, int]] = set()
        for relation in record.get("relations", []):
            head_entity = int(relation["head"])
            tail_entity = int(relation["tail"])
            relation_type = relation.get("type")
            if relation_type not in self.relation2id:
                raise ValueError(
                    f"Record {record_index}: unknown relation type {relation_type!r}."
                )
            if head_entity not in entity_span_indices or tail_entity not in entity_span_indices:
                continue
            pair = (entity_span_indices[head_entity], entity_span_indices[tail_entity])
            target = relation_targets_by_pair.setdefault(
                pair, [0.0] * len(self.relation_labels)
            )
            target[self.relation2id[relation_type]] = 1.0
            positive_pair_keys.add(pair)

        valid_entity_indices = [
            entity_span_indices[i] for i in range(len(entities)) if i in entity_span_indices
        ]
        candidate_negative_pairs = list({
            (head, tail)
            for head in valid_entity_indices
            for tail in valid_entity_indices
            if head != tail and (head, tail) not in positive_pair_keys
        })
        rng.shuffle(candidate_negative_pairs)
        for pair in candidate_negative_pairs[: self.neg_relation_count]:
            relation_targets_by_pair[pair] = [0.0] * len(self.relation_labels)

        relation_pairs = list(relation_targets_by_pair)
        relation_targets = [relation_targets_by_pair[pair] for pair in relation_pairs]

        # Gold relations are kept in span coordinates for strict end-to-end
        # validation metrics, independent of the sampled training negatives.
        gold_relations: list[tuple[int, int, str, str, int, int, str]] = []
        for relation in record.get("relations", []):
            h_idx, t_idx = int(relation["head"]), int(relation["tail"])
            if h_idx not in entity_span_indices or t_idx not in entity_span_indices:
                continue
            h_span = spans[entity_span_indices[h_idx]]
            t_span = spans[entity_span_indices[t_idx]]
            gold_relations.append((
                h_span[0], h_span[1], entities[h_idx]["label"],
                relation["type"],
                t_span[0], t_span[1], entities[t_idx]["label"],
            ))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "spans": spans,
            "span_widths": [end - start + 1 for start, end in spans],
            "entity_labels": entity_labels,
            "entity_loss_mask": entity_loss_mask,
            "relation_pairs": relation_pairs,
            "relation_targets": relation_targets,
            "relation_label_count": len(self.relation_labels),
            "gold_entities": gold_entity_records,
            "gold_relations": gold_relations,
        }


def spert_collate_fn(batch: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
    """Pad token and span tensors while keeping variable relation pairs per item."""
    batch_size = len(batch)
    max_tokens = max(len(item["input_ids"]) for item in batch)
    max_spans = max(len(item["spans"]) for item in batch)

    input_ids = torch.full((batch_size, max_tokens), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_tokens), dtype=torch.long)
    spans = torch.zeros((batch_size, max_spans, 2), dtype=torch.long)
    span_widths = torch.ones((batch_size, max_spans), dtype=torch.long)
    span_mask = torch.zeros((batch_size, max_spans), dtype=torch.bool)
    entity_labels = torch.zeros((batch_size, max_spans), dtype=torch.long)
    entity_loss_mask = torch.zeros((batch_size, max_spans), dtype=torch.bool)

    relation_pairs: list[torch.Tensor] = []
    relation_targets: list[torch.Tensor] = []
    for b, item in enumerate(batch):
        token_count = len(item["input_ids"])
        span_count = len(item["spans"])
        input_ids[b, :token_count] = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask[b, :token_count] = torch.tensor(item["attention_mask"], dtype=torch.long)
        spans[b, :span_count] = torch.tensor(item["spans"], dtype=torch.long)
        span_widths[b, :span_count] = torch.tensor(item["span_widths"], dtype=torch.long)
        span_mask[b, :span_count] = True
        entity_labels[b, :span_count] = torch.tensor(item["entity_labels"], dtype=torch.long)
        entity_loss_mask[b, :span_count] = torch.tensor(item["entity_loss_mask"], dtype=torch.bool)
        relation_pairs.append(torch.tensor(item["relation_pairs"], dtype=torch.long).reshape(-1, 2))
        relation_label_count = item["relation_label_count"]
        if item["relation_targets"]:
            relation_targets.append(
                torch.tensor(item["relation_targets"], dtype=torch.float32).reshape(-1, relation_label_count)
            )
        else:
            relation_targets.append(torch.empty((0, relation_label_count), dtype=torch.float32))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "spans": spans,
        "span_widths": span_widths,
        "span_mask": span_mask,
        "entity_labels": entity_labels,
        "entity_loss_mask": entity_loss_mask,
        "relation_pairs": relation_pairs,
        "relation_targets": relation_targets,
        "gold_entities": [item["gold_entities"] for item in batch],
        "gold_relations": [item["gold_relations"] for item in batch],
    }
