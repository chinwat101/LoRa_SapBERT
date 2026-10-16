"""
==============================================================================
Multi-Task Learning Model: JointNERAndREModel
==============================================================================
ไฟล์นี้สร้างขึ้นเพื่อ:
  กำหนดสถาปัตยกรรมหลักของ Multi-Task Model ที่ประกอบด้วย:
    1. Shared BERT Encoder  — ใช้ร่วมกันระหว่าง NER และ RE
    2. NER Head             — Linear(hidden_size → num_ner_labels)
                              ทำนาย BIO tag ต่อ token
    3. RE Head              — Linear(hidden_size*3 → num_rel_labels)
                              ทำนาย relation ต่อ entity pair
                              feature = [h_i ; h_j ; h_i ⊙ h_j]

Joint Loss:
    L_total = lambda_ner × L_NER + lambda_re × L_RE
==============================================================================
"""

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel


class JointNERAndREModel(nn.Module):
    """
    Multi-Task BERT Model สำหรับ NER + Relation Extraction พร้อมกัน

    Args:
        encoder (PreTrainedModel): BERT encoder ที่โหลดมาแล้ว
        num_ner_labels (int): จำนวน NER labels (default 7: O + BIO×3)
        num_rel_labels (int): จำนวน Relation labels (default 3)
        dropout (float): dropout rate ก่อน classification heads
        lambda_ner (float): weight ของ NER loss
        lambda_re (float): weight ของ RE loss
    """

    def __init__(
        self,
        encoder: PreTrainedModel,
        num_ner_labels: int = 7,
        num_rel_labels: int = 3,
        dropout: float = 0.1,
        lambda_ner: float = 1.0,
        lambda_re: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size

        self.lambda_ner = lambda_ner
        self.lambda_re = lambda_re

        self.dropout = nn.Dropout(dropout)

        # NER Head: token-level classification
        self.ner_head = nn.Linear(hidden_size, num_ner_labels)

        # RE Head: entity-pair level classification
        # input = [h_i ; h_j ; h_i ⊙ h_j]  → ขนาด hidden_size * 3
        self.re_head = nn.Linear(hidden_size * 3, num_rel_labels)

        # Loss functions
        self.ner_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.re_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,              # [B, L]
        attention_mask: torch.Tensor,          # [B, L]
        ner_labels: torch.Tensor = None,       # [B, L]  -100 = ignore
        pairs: torch.Tensor = None,            # [P, 2]  (tok_head, tok_tail)
        rel_labels: torch.Tensor = None,       # [P]
        pair_sample_idx: torch.Tensor = None,  # [P]  sample index ใน batch
    ) -> dict:
        """
        Returns dict ประกอบด้วย:
            loss         : scalar (ถ้า labels ถูกส่งมา)
            ner_loss     : scalar
            re_loss      : scalar
            ner_logits   : Tensor[B, L, num_ner_labels]
            re_logits    : Tensor[P, num_rel_labels]  (ถ้ามี pairs)
        """
        # ─── Shared BERT Encoder ─────────────────────────────────────────
        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # sequence_output: [B, L, H]
        sequence_output = self.dropout(encoder_out.last_hidden_state)

        # ─── NER Head ────────────────────────────────────────────────────
        ner_logits = self.ner_head(sequence_output)   # [B, L, num_ner_labels]

        # ─── RE Head ─────────────────────────────────────────────────────
        re_logits = None
        if pairs is not None and pairs.shape[0] > 0:
            re_logits = self._compute_re_logits(
                sequence_output, pairs, pair_sample_idx
            )   # [P, num_rel_labels]

        # ─── Loss Computation ─────────────────────────────────────────────
        ner_loss = torch.tensor(0.0, device=input_ids.device)
        re_loss = torch.tensor(0.0, device=input_ids.device)
        total_loss = torch.tensor(0.0, device=input_ids.device)

        if ner_labels is not None:
            ner_loss = self.ner_loss_fn(
                ner_logits.view(-1, ner_logits.size(-1)),  # [B*L, C]
                ner_labels.view(-1),                        # [B*L]
            )

        if re_logits is not None and rel_labels is not None:
            re_loss = self.re_loss_fn(re_logits, rel_labels)

        if ner_labels is not None or rel_labels is not None:
            total_loss = self.lambda_ner * ner_loss + self.lambda_re * re_loss

        return {
            "loss": total_loss,
            "ner_loss": ner_loss,
            "re_loss": re_loss,
            "ner_logits": ner_logits,
            "re_logits": re_logits,
        }

    def _compute_re_logits(
        self,
        sequence_output: torch.Tensor,  # [B, L, H]
        pairs: torch.Tensor,             # [P, 2] — (tok_head, tok_tail)
        pair_sample_idx: torch.Tensor,   # [P]    — sample index
    ) -> torch.Tensor:
        """
        สร้าง RE feature สำหรับแต่ละ entity pair:
            feature = [h_i ; h_j ; h_i ⊙ h_j]   ขนาด 3H
        Returns: Tensor[P, num_rel_labels]
        """
        h_head_list = []
        h_tail_list = []

        for p_idx in range(pairs.shape[0]):
            sample_idx = pair_sample_idx[p_idx].item()
            tok_head = pairs[p_idx, 0].item()
            tok_tail = pairs[p_idx, 1].item()

            h_head = sequence_output[sample_idx, tok_head, :]   # [H]
            h_tail = sequence_output[sample_idx, tok_tail, :]   # [H]

            h_head_list.append(h_head)
            h_tail_list.append(h_tail)

        h_heads = torch.stack(h_head_list, dim=0)  # [P, H]
        h_tails = torch.stack(h_tail_list, dim=0)  # [P, H]

        # Concatenate: [h_i ; h_j ; h_i ⊙ h_j]
        element_wise = h_heads * h_tails            # [P, H]
        re_features = torch.cat([h_heads, h_tails, element_wise], dim=-1)  # [P, 3H]

        re_logits = self.re_head(self.dropout(re_features))  # [P, num_rel_labels]
        return re_logits


def build_model(
    base_model_path: str,
    num_ner_labels: int = 7,
    num_rel_labels: int = 3,
    dropout: float = 0.1,
    lambda_ner: float = 1.0,
    lambda_re: float = 1.0,
) -> JointNERAndREModel:
    """
    Helper function: โหลด BERT encoder จาก path แล้วสร้าง JointNERAndREModel
    """
    encoder = AutoModel.from_pretrained(base_model_path, ignore_mismatched_sizes=True)
    model = JointNERAndREModel(
        encoder=encoder,
        num_ner_labels=num_ner_labels,
        num_rel_labels=num_rel_labels,
        dropout=dropout,
        lambda_ner=lambda_ner,
        lambda_re=lambda_re,
    )
    return model
