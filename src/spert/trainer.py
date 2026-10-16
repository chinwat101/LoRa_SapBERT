"""Training, strict span/relation metrics, checkpointing for SpERTModel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


def linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    def scale(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, scale)


def _f1_from_sets(gold: set, predicted: set, prefix: str) -> dict[str, float]:
    true_positives = len(gold & predicted)
    false_positives = len(predicted - gold)
    false_negatives = len(gold - predicted)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {f"{prefix}_precision": precision, f"{prefix}_recall": recall, f"{prefix}_f1": f1}


class SpERTTrainer:
    def __init__(
        self,
        model: nn.Module,
        entity_labels: list[str],
        relation_labels: list[str],
        device: torch.device,
        output_dir: str | Path,
        lambda_entity: float = 1.0,
        lambda_relation: float = 1.0,
        relation_threshold: float = 0.4,
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.entity_labels = entity_labels
        self.relation_labels = relation_labels
        self.device = device
        self.output_dir = Path(output_dir)
        self.lambda_entity = lambda_entity
        self.lambda_relation = lambda_relation
        self.relation_threshold = relation_threshold
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved = dict(batch)
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device)
        return moved

    def _loss(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entity_mask = batch["entity_loss_mask"] & batch["span_mask"]
        if entity_mask.any():
            entity_loss = nn.functional.cross_entropy(
                outputs["entity_logits"][entity_mask],
                batch["entity_labels"][entity_mask],
            )
        else:
            entity_loss = outputs["entity_logits"].sum() * 0.0

        relation_losses = []
        for logits, targets in zip(outputs["train_relation_logits"], batch["relation_targets"]):
            targets = targets.to(self.device)
            if logits.shape[0] and targets.numel():
                relation_losses.append(
                    nn.functional.binary_cross_entropy_with_logits(logits, targets)
                )
        relation_loss = (
            torch.stack(relation_losses).mean()
            if relation_losses
            else outputs["entity_logits"].sum() * 0.0
        )
        total_loss = self.lambda_entity * entity_loss + self.lambda_relation * relation_loss
        return total_loss, entity_loss, relation_loss

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
    ) -> dict[str, float]:
        self.model.train()
        total = entity_total = relation_total = 0.0
        batch_count = 0
        for raw_batch in dataloader:
            batch = self._move_batch(raw_batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                outputs = self.model(batch)
                loss, entity_loss, relation_loss = self._loss(outputs, batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
            if scheduler is not None:
                scheduler.step()
            total += float(loss.detach())
            entity_total += float(entity_loss.detach())
            relation_total += float(relation_loss.detach())
            batch_count += 1

        divisor = max(1, batch_count)
        return {
            "loss": total / divisor,
            "entity_loss": entity_total / divisor,
            "relation_loss": relation_total / divisor,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total = entity_total = relation_total = 0.0
        batch_count = 0
        gold_entities: set[tuple] = set()
        predicted_entities: set[tuple] = set()
        gold_relations: set[tuple] = set()
        predicted_relations: set[tuple] = set()
        sentence_offset = 0

        for raw_batch in dataloader:
            batch = self._move_batch(raw_batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                outputs = self.model(batch, calculate_inference_relations=True)
                loss, entity_loss, relation_loss = self._loss(outputs, batch)
            total += float(loss)
            entity_total += float(entity_loss)
            relation_total += float(relation_loss)
            batch_count += 1

            predicted_labels = outputs["entity_logits"].argmax(dim=-1)
            for b in range(predicted_labels.shape[0]):
                sid = sentence_offset + b
                for start, end, label in raw_batch["gold_entities"][b]:
                    gold_entities.add((sid, start, end, label))

                spans = raw_batch["spans"][b]
                for span_index, class_id in enumerate(predicted_labels[b].tolist()):
                    if not raw_batch["span_mask"][b, span_index] or class_id == 0:
                        continue
                    start, end = spans[span_index].tolist()
                    label = self.entity_labels[class_id - 1]
                    predicted_entities.add((sid, start, end, label))
                for start, end, head_label, relation, tail_start, tail_end, tail_label in raw_batch["gold_relations"][b]:
                    gold_relations.add((
                        sid, start, end, head_label, relation,
                        tail_start, tail_end, tail_label,
                    ))

                pair_tensor = outputs["inference_pairs"][b]
                relation_logits = outputs["inference_relation_logits"][b]
                if pair_tensor.numel():
                    relation_scores = torch.sigmoid(relation_logits)
                    for pair_index, (head_index, tail_index) in enumerate(pair_tensor.tolist()):
                        head_start, head_end = spans[head_index].tolist()
                        tail_start, tail_end = spans[tail_index].tolist()
                        head_class = int(predicted_labels[b, head_index])
                        tail_class = int(predicted_labels[b, tail_index])
                        head_label = self.entity_labels[head_class - 1]
                        tail_label = self.entity_labels[tail_class - 1]
                        for relation_index, score in enumerate(relation_scores[pair_index].tolist()):
                            if score >= self.relation_threshold:
                                predicted_relations.add((
                                    sid, head_start, head_end, head_label,
                                    self.relation_labels[relation_index],
                                    tail_start, tail_end, tail_label,
                                ))
            sentence_offset += predicted_labels.shape[0]

        divisor = max(1, batch_count)
        metrics = {
            "val_loss": total / divisor,
            "val_entity_loss": entity_total / divisor,
            "val_relation_loss": relation_total / divisor,
        }
        metrics.update(_f1_from_sets(gold_entities, predicted_entities, "val_entity"))
        metrics.update(_f1_from_sets(gold_relations, predicted_relations, "val_relation"))
        metrics["val_joint_f1"] = (metrics["val_entity_f1"] + metrics["val_relation_f1"]) / 2
        return metrics

    def save_checkpoint(self, tokenizer, config: dict, metrics: dict[str, float]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.output_dir / "model.pt")
        tokenizer.save_pretrained(self.output_dir)
        with (self.output_dir / "label_config.json").open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "entity_labels": self.entity_labels,
                    "relation_labels": self.relation_labels,
                    "model_path": config["model_path"],
                    "relation_threshold": self.relation_threshold,
                    "best_validation_metrics": metrics,
                },
                stream,
                ensure_ascii=False,
                indent=2,
            )
