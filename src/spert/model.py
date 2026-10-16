"""SpERT-style model with span pooling and pair-local relation context."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class SpERTModel(nn.Module):
    """Span-based joint entity/relation model using a pretrained BERT encoder."""

    def __init__(
        self,
        model_path: str,
        entity_label_count: int,
        relation_label_count: int,
        max_span_size: int = 10,
        size_embedding_dim: int = 25,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        relation_chunk_size: int = 100,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.encoder.config.hidden_size
        self.max_span_size = max_span_size
        self.relation_chunk_size = max(1, relation_chunk_size)
        self.size_embeddings = nn.Embedding(max_span_size + 1, size_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.entity_classifier = nn.Linear(
            self.hidden_size * 2 + size_embedding_dim,
            entity_label_count + 1,  # class 0 is "not an entity"
        )
        self.relation_classifier = nn.Linear(
            self.hidden_size * 3 + size_embedding_dim * 2,
            relation_label_count,
        )
        # Match the initialization used by the original SpERT implementation
        # without reinitializing the pretrained encoder.
        nn.init.normal_(self.size_embeddings.weight, mean=0.0, std=0.02)
        for classifier in (self.entity_classifier, self.relation_classifier):
            nn.init.normal_(classifier.weight, mean=0.0, std=0.02)
            nn.init.zeros_(classifier.bias)

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def _pool_spans(
        self,
        hidden: torch.Tensor,
        spans: torch.Tensor,
        span_widths: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Max-pool each token span without materializing a BxSxLxH tensor."""
        batch_size, span_count = span_widths.shape
        pooled = hidden.new_zeros((batch_size, span_count, self.hidden_size))
        for width in range(1, self.max_span_size + 1):
            positions = ((span_widths == width) & span_mask).nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            window_vectors = hidden.unfold(1, width, 1).amax(dim=-1)
            batch_indices, span_indices = positions[:, 0], positions[:, 1]
            starts = spans[batch_indices, span_indices, 0]
            pooled[batch_indices, span_indices] = window_vectors[batch_indices, starts]
        return pooled

    def _entity_logits(
        self,
        hidden: torch.Tensor,
        spans: torch.Tensor,
        span_widths: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        span_vectors = self._pool_spans(hidden, spans, span_widths, span_mask)
        widths = span_widths.clamp(min=1, max=self.max_span_size)
        size_vectors = self.size_embeddings(widths)
        cls_vectors = hidden[:, :1, :].expand(-1, spans.shape[1], -1)
        features = torch.cat((cls_vectors, span_vectors, size_vectors), dim=-1)
        logits = self.entity_classifier(self.dropout(features))
        return logits, span_vectors, size_vectors

    def _relation_logits_for_pairs(
        self,
        hidden: torch.Tensor,
        spans: torch.Tensor,
        span_vectors: torch.Tensor,
        size_vectors: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        if pairs.numel() == 0:
            return hidden.new_zeros((0, self.relation_classifier.out_features))

        pair_logits: list[torch.Tensor] = []
        for chunk_start in range(0, pairs.shape[0], self.relation_chunk_size):
            chunk = pairs[chunk_start:chunk_start + self.relation_chunk_size]
            heads, tails = chunk[:, 0], chunk[:, 1]
            head_vectors = span_vectors[heads]
            tail_vectors = span_vectors[tails]
            head_sizes = size_vectors[heads]
            tail_sizes = size_vectors[tails]

            contexts = []
            for head_index, tail_index in chunk.tolist():
                head_start, head_end = spans[head_index].tolist()
                tail_start, tail_end = spans[tail_index].tolist()
                if head_end < tail_start:
                    left, right = head_end + 1, tail_start
                elif tail_end < head_start:
                    left, right = tail_end + 1, head_start
                else:
                    left, right = 0, 0
                if right > left:
                    contexts.append(hidden[left:right].amax(dim=0))
                else:
                    contexts.append(hidden.new_zeros(self.hidden_size))
            context_vectors = torch.stack(contexts)
            features = torch.cat(
                (context_vectors, head_vectors, tail_vectors, head_sizes, tail_sizes),
                dim=-1,
            )
            pair_logits.append(self.relation_classifier(self.dropout(features)))
        return torch.cat(pair_logits, dim=0)

    def forward(
        self,
        batch: dict,
        calculate_inference_relations: bool = False,
    ) -> dict:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        entity_logits, span_vectors, size_vectors = self._entity_logits(
            hidden,
            batch["spans"],
            batch["span_widths"],
            batch["span_mask"],
        )

        train_relation_logits = []
        for batch_index, pairs in enumerate(batch["relation_pairs"]):
            pairs = pairs.to(input_ids.device)
            train_relation_logits.append(
                self._relation_logits_for_pairs(
                    hidden[batch_index],
                    batch["spans"][batch_index],
                    span_vectors[batch_index],
                    size_vectors[batch_index],
                    pairs,
                )
            )

        inference_pairs: list[torch.Tensor] = []
        inference_relation_logits: list[torch.Tensor] = []
        if calculate_inference_relations:
            predicted_entity_labels = entity_logits.argmax(dim=-1)
            for batch_index in range(input_ids.shape[0]):
                entity_indices = (
                    (predicted_entity_labels[batch_index] > 0)
                    & batch["span_mask"][batch_index]
                ).nonzero(as_tuple=False).flatten()
                pair_list = [
                    (int(head), int(tail))
                    for head in entity_indices.tolist()
                    for tail in entity_indices.tolist()
                    if head != tail
                ]
                pairs = torch.tensor(pair_list, dtype=torch.long, device=input_ids.device).reshape(-1, 2)
                inference_pairs.append(pairs)
                inference_relation_logits.append(
                    self._relation_logits_for_pairs(
                        hidden[batch_index],
                        batch["spans"][batch_index],
                        span_vectors[batch_index],
                        size_vectors[batch_index],
                        pairs,
                    )
                )

        return {
            "entity_logits": entity_logits,
            "train_relation_logits": train_relation_logits,
            "inference_pairs": inference_pairs,
            "inference_relation_logits": inference_relation_logits,
        }
