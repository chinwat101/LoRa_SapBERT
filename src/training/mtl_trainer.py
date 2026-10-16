"""
==============================================================================
Multi-Task Learning Trainer & Evaluator
==============================================================================
ไฟล์นี้สร้างขึ้นเพื่อ:
  จัดการ Training Loop และ Evaluation สำหรับ JointNERAndREModel
  โดยคำนวณ metrics สำหรับทั้ง NER (entity-level F1) และ RE (pair-level F1)

Metrics ที่คำนวณ:
  - NER: Micro F1, Per-class F1 (DRUG / DISEASE / SYMPTOM) — entity-level span
  - RE:  Micro F1, Per-class F1 (TREATS / INDICATED_FOR) — pair-level
  - Joint F1: mean(NER_F1, RE_F1)
==============================================================================
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from sklearn.metrics import f1_score, precision_recall_fscore_support


IGNORE_INDEX = -100


class MTLTrainer:
    """
    Trainer สำหรับ JointNERAndREModel (Multi-Task: NER + RE)

    Args:
        model           : JointNERAndREModel instance
        optimizer       : torch optimizer
        scheduler       : LR scheduler (optional)
        device          : torch.device
        max_grad_norm   : gradient clipping (default 1.0)
        id2ner          : dict{int → str} สำหรับ decode NER labels
        id2rel          : dict{int → str} สำหรับ decode RE labels
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler=None,
        max_grad_norm: float = 1.0,
        id2ner: Dict[int, str] = None,
        id2rel: Dict[int, str] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.id2ner = id2ner or {}
        self.id2rel = id2rel or {}

    # ─────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """ฝึกสอน 1 epoch, returns average losses"""
        self.model.train()
        total_loss = total_ner_loss = total_re_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            ner_labels = batch["ner_labels"].to(self.device)
            pairs = batch["pairs"].to(self.device)
            rel_labels = batch["rel_labels"].to(self.device)
            pair_sample_idx = batch["pair_sample_idx"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ner_labels=ner_labels,
                pairs=pairs,
                rel_labels=rel_labels,
                pair_sample_idx=pair_sample_idx,
            )

            loss = outputs["loss"]
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            total_ner_loss += outputs["ner_loss"].item()
            total_re_loss += outputs["re_loss"].item()
            num_batches += 1

        n = max(1, num_batches)
        return {
            "loss": total_loss / n,
            "ner_loss": total_ner_loss / n,
            "re_loss": total_re_loss / n,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        ประเมินผลบน dataloader:
          - NER: entity-level span F1 (micro, exclude O)
          - RE:  pair-level F1 (micro, exclude NO_RELATION)
        """
        self.model.eval()
        total_loss = total_ner_loss = total_re_loss = 0.0
        num_batches = 0

        # สะสม predictions สำหรับ NER (token-level → แปลงเป็น span)
        all_ner_true: List[List[int]] = []
        all_ner_pred: List[List[int]] = []

        # สะสม predictions สำหรับ RE
        all_re_true: List[int] = []
        all_re_pred: List[int] = []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            ner_labels = batch["ner_labels"].to(self.device)
            pairs = batch["pairs"].to(self.device)
            rel_labels = batch["rel_labels"].to(self.device)
            pair_sample_idx = batch["pair_sample_idx"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ner_labels=ner_labels,
                pairs=pairs,
                rel_labels=rel_labels,
                pair_sample_idx=pair_sample_idx,
            )

            total_loss += outputs["loss"].item()
            total_ner_loss += outputs["ner_loss"].item()
            total_re_loss += outputs["re_loss"].item()
            num_batches += 1

            # --- NER predictions ---
            ner_logits = outputs["ner_logits"]   # [B, L, C]
            ner_pred = ner_logits.argmax(dim=-1)  # [B, L]

            for b in range(ner_labels.shape[0]):
                true_seq, pred_seq = [], []
                for t in range(ner_labels.shape[1]):
                    lbl = ner_labels[b, t].item()
                    if lbl == IGNORE_INDEX:
                        continue
                    true_seq.append(lbl)
                    pred_seq.append(ner_pred[b, t].item())
                all_ner_true.append(true_seq)
                all_ner_pred.append(pred_seq)

            # --- RE predictions ---
            if outputs["re_logits"] is not None:
                re_pred = outputs["re_logits"].argmax(dim=-1)  # [P]
                all_re_true.extend(rel_labels.cpu().tolist())
                all_re_pred.extend(re_pred.cpu().tolist())

        n = max(1, num_batches)
        avg_metrics = {
            "val_loss": total_loss / n,
            "val_ner_loss": total_ner_loss / n,
            "val_re_loss": total_re_loss / n,
        }

        # --- คำนวณ NER F1 (entity-level span) ---
        ner_f1_scores = self._compute_ner_entity_f1(all_ner_true, all_ner_pred)
        avg_metrics.update(ner_f1_scores)

        # --- คำนวณ RE F1 ---
        re_f1_scores = self._compute_re_f1(all_re_true, all_re_pred)
        avg_metrics.update(re_f1_scores)

        # --- Joint F1 ---
        avg_metrics["val_joint_f1"] = (
            avg_metrics.get("val_ner_f1", 0.0) + avg_metrics.get("val_re_f1", 0.0)
        ) / 2.0

        return avg_metrics

    # ─────────────────────────────────────────────────────────────────────
    # Metric Helpers
    # ─────────────────────────────────────────────────────────────────────
    def _compute_ner_entity_f1(
        self,
        all_true: List[List[int]],
        all_pred: List[List[int]],
    ) -> Dict[str, float]:
        """
        คำนวณ entity-level (span-level) F1 สำหรับ NER
        Entity ถือว่าถูกต้องเมื่อ label + boundary ตรงทั้งคู่
        """
        def extract_spans(seq: List[int]) -> List[Tuple[int, int, str]]:
            """แปลง BIO id sequence → list of (start, end, label)"""
            spans = []
            cur_start, cur_label = None, None
            for i, tag_id in enumerate(seq):
                tag = self.id2ner.get(tag_id, "O")
                if tag.startswith("B-"):
                    if cur_start is not None:
                        spans.append((cur_start, i - 1, cur_label))
                    cur_start = i
                    cur_label = tag[2:]
                elif tag.startswith("I-") and cur_label == tag[2:]:
                    pass  # ต่อ entity เดิม
                else:
                    if cur_start is not None:
                        spans.append((cur_start, i - 1, cur_label))
                    cur_start, cur_label = None, None
            if cur_start is not None:
                spans.append((cur_start, len(seq) - 1, cur_label))
            return spans

        tp = fp = fn = 0
        label_tp: Dict[str, int] = {}
        label_fp: Dict[str, int] = {}
        label_fn: Dict[str, int] = {}

        for true_seq, pred_seq in zip(all_true, all_pred):
            true_spans = set(extract_spans(true_seq))
            pred_spans = set(extract_spans(pred_seq))

            for span in pred_spans:
                if span in true_spans:
                    tp += 1
                    label_tp[span[2]] = label_tp.get(span[2], 0) + 1
                else:
                    fp += 1
                    label_fp[span[2]] = label_fp.get(span[2], 0) + 1
            for span in true_spans:
                if span not in pred_spans:
                    fn += 1
                    label_fn[span[2]] = label_fn.get(span[2], 0) + 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        micro_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        result = {
            "val_ner_precision": prec,
            "val_ner_recall": rec,
            "val_ner_f1": micro_f1,
        }

        # Per-class F1
        for label in set(list(label_tp.keys()) + list(label_fp.keys()) + list(label_fn.keys())):
            t = label_tp.get(label, 0)
            f = label_fp.get(label, 0)
            n = label_fn.get(label, 0)
            p_cls = t / (t + f) if (t + f) > 0 else 0.0
            r_cls = t / (t + n) if (t + n) > 0 else 0.0
            f1_cls = 2 * p_cls * r_cls / (p_cls + r_cls) if (p_cls + r_cls) > 0 else 0.0
            result[f"val_ner_f1_{label.lower()}"] = f1_cls

        return result

    def _compute_re_f1(
        self,
        all_true: List[int],
        all_pred: List[int],
    ) -> Dict[str, float]:
        """คำนวณ RE F1 (micro) exclude NO_RELATION (id=0)"""
        if not all_true:
            return {"val_re_f1": 0.0, "val_re_precision": 0.0, "val_re_recall": 0.0}

        # กรอง NO_RELATION ออก
        filtered_true, filtered_pred = [], []
        for t, p in zip(all_true, all_pred):
            if t != 0 or p != 0:  # เอาเฉพาะที่ true หรือ pred ไม่ใช่ NO_REL
                filtered_true.append(t)
                filtered_pred.append(p)

        if not filtered_true:
            return {"val_re_f1": 0.0, "val_re_precision": 0.0, "val_re_recall": 0.0}

        labels = sorted(set(filtered_true + filtered_pred) - {0})
        prec, rec, f1, _ = precision_recall_fscore_support(
            filtered_true, filtered_pred,
            labels=labels,
            average="micro",
            zero_division=0,
        )
        result = {
            "val_re_precision": float(prec),
            "val_re_recall": float(rec),
            "val_re_f1": float(f1),
        }

        # Per-class F1
        per_cls_prec, per_cls_rec, per_cls_f1, _ = precision_recall_fscore_support(
            filtered_true, filtered_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )
        for i, lbl_id in enumerate(labels):
            lbl_name = self.id2rel.get(lbl_id, str(lbl_id)).lower()
            result[f"val_re_f1_{lbl_name}"] = float(per_cls_f1[i])

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────────────────
    def save_checkpoint(self, output_dir: str):
        """บันทึก model state_dict ไปยัง output_dir"""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))

    def load_checkpoint(self, checkpoint_path: str):
        """โหลด model weights จาก checkpoint"""
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)


# ─────────────────────────────────────────────────────────────────────────────
# LR Scheduler Helper
# ─────────────────────────────────────────────────────────────────────────────
def get_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Linear warmup แล้ว linear decay ถึง 0
    ใช้แทน HuggingFace get_linear_schedule_with_warmup
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)
