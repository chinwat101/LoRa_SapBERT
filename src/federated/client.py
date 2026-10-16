"""
src/federated/client.py

Flower NERLoRAClient — ใช้ LoRA adapter เทรน NER บน local data
ส่งกลับแค่ LoRA adapter weights ไปที่ aggregator server (~2-5 MB/round)
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import flwr as fl
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

DEFAULT_LABEL2ID = {
    "O": 0, "B-DRUG": 1, "I-DRUG": 2,
    "B-DISEASE": 3, "I-DISEASE": 4,
    "B-SYMPTOM": 5, "I-SYMPTOM": 6,
}
DEFAULT_ID2LABEL = {v: k for k, v in DEFAULT_LABEL2ID.items()}


# ─── Dataset helpers (reuse logic from train_sapbert_ner.py) ─────────────────

def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def align_labels_to_subwords(example: dict, tokenizer, label2id: dict, max_length: int = 128) -> dict:
    text = example["text"]
    spans = example.get("entities", [])

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )

    char_label: dict = {}
    span_starts: set = set()
    for span in spans:
        span_starts.add(span["start"])
        for i in range(span["start"], span["end"]):
            char_label.setdefault(i, span["label"])

    labels = []
    for cs, ce in enc["offset_mapping"]:
        if cs == 0 and ce == 0:
            labels.append(-100)
            continue
        tag_id = label2id["O"]
        for ci in range(cs, ce):
            if ci in char_label:
                lbl = char_label[ci]
                prefix = "B-" if ci in span_starts else "I-"
                tag_id = label2id.get(f"{prefix}{lbl}", label2id["O"])
                break
        labels.append(tag_id)

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


def build_hf_dataset(records: list, tokenizer, label2id: dict, max_length: int) -> Dataset:
    aligned = [align_labels_to_subwords(r, tokenizer, label2id, max_length) for r in records]
    return Dataset.from_list(aligned)


def make_compute_metrics(id2label: dict):
    try:
        from seqeval.metrics import f1_score, precision_score, recall_score
        USE_SEQEVAL = True
    except ImportError:
        USE_SEQEVAL = False

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        true_seqs, pred_seqs = [], []
        for pred_row, label_row in zip(preds, labels):
            true_seq, pred_seq = [], []
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                true_seq.append(id2label.get(l, "O"))
                pred_seq.append(id2label.get(p, "O"))
            true_seqs.append(true_seq)
            pred_seqs.append(pred_seq)

        if USE_SEQEVAL:
            return {
                "eval_f1": f1_score(true_seqs, pred_seqs),
                "eval_precision": precision_score(true_seqs, pred_seqs),
                "eval_recall": recall_score(true_seqs, pred_seqs),
            }
        else:
            correct = total = 0
            for ts, ps in zip(true_seqs, pred_seqs):
                for t, p in zip(ts, ps):
                    if t != "O":
                        total += 1
                        correct += t == p
            return {"eval_f1": correct / total if total else 0.0}

    return compute_metrics


# ─── FL Client ────────────────────────────────────────────────────────────────

class NERLoRAClient(fl.client.NumPyClient):
    """
    Flower FL Client ที่เทรน NER LoRA Adapter บน local dataset
    ส่งกลับแค่ LoRA weights (~5 MB) แทนโมเดลทั้งก้อน (~450 MB)
    """

    def __init__(
        self,
        client_id: int,
        client_name: str,
        model_path: str,
        train_file: str,
        lora_cfg: dict,
        training_cfg: dict,
        label2id: dict,
        id2label: dict,
        seed: int = 42,
    ):
        self.client_id = client_id
        self.client_name = client_name
        self.model_path = model_path
        self.train_file = train_file
        self.lora_cfg = lora_cfg
        self.training_cfg = training_cfg
        self.label2id = label2id
        self.id2label = id2label
        self.seed = seed

        set_seed(seed)

        logger.info(f"[Client {client_id} — {client_name}] Loading tokenizer from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # โหลดและแบ่ง dataset
        records = load_jsonl(Path(train_file))
        random.seed(seed)
        random.shuffle(records)
        val_frac = training_cfg.get("val_split", 0.2)
        split = int(len(records) * (1 - val_frac))
        train_records, val_records = records[:split], records[split:]
        logger.info(f"[Client {client_id}] Dataset: {train_file} → train={len(train_records)}, val={len(val_records)}")

        max_len = training_cfg.get("max_seq_length", 128)
        self.train_dataset = build_hf_dataset(train_records, self.tokenizer, label2id, max_len)
        self.eval_dataset = build_hf_dataset(val_records, self.tokenizer, label2id, max_len)

        # โหลด model + LoRA
        self.model = self._load_model_with_lora()

    def _load_model_with_lora(self):
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_path,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=self.lora_cfg.get("r", 16),
            lora_alpha=self.lora_cfg.get("lora_alpha", 32),
            lora_dropout=self.lora_cfg.get("lora_dropout", 0.1),
            bias=self.lora_cfg.get("bias", "none"),
            target_modules=self.lora_cfg.get("target_modules", ["query", "key", "value", "dense"]),
            inference_mode=False,
        )
        return get_peft_model(model, peft_config)

    def get_parameters(self, config) -> List[np.ndarray]:
        """ส่งเฉพาะ LoRA adapter weights (ไม่ส่ง base model weights)"""
        lora_state = get_peft_model_state_dict(self.model)
        return [val.cpu().numpy() for val in lora_state.values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """โหลด global LoRA weights จาก server เข้า local model"""
        lora_state = get_peft_model_state_dict(self.model)
        params_dict = {
            k: torch.tensor(v)
            for k, v in zip(lora_state.keys(), parameters)
        }
        set_peft_model_state_dict(self.model, params_dict)

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        """รับ global adapter → เทรน local → ส่ง updated adapter กลับ"""
        fl_round = config.get("fl_round", "?")
        logger.info(f"[Client {self.client_id} — {self.client_name}] fit() Round {fl_round}")

        # 1. รับ global weights จาก server
        self.set_parameters(parameters)

        # 2. สร้าง training args
        local_epochs = self.training_cfg.get("local_epochs", 2)
        batch_size = self.training_cfg.get("per_device_train_batch_size", 8)
        use_fp16 = self.training_cfg.get("fp16", True) and torch.cuda.is_available()

        checkpoint_dir = Path(f"outputs/fl_ner/checkpoints/client_{self.client_id}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=local_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=self.training_cfg.get("per_device_eval_batch_size", 8),
            learning_rate=self.training_cfg.get("learning_rate", 3e-4),
            fp16=use_fp16,
            logging_steps=max(1, len(self.train_dataset) // batch_size // 2),
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
            seed=self.seed,
            disable_tqdm=False,
        )

        # 3. เทรน
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
            compute_metrics=make_compute_metrics(self.id2label),
        )
        trainer.train()

        # 4. ส่ง updated LoRA weights กลับ
        metrics = trainer.evaluate()
        f1 = metrics.get("eval_f1", 0.0)
        logger.info(f"[Client {self.client_id}] fit() done — local F1: {f1:.4f}")

        return self.get_parameters({}), len(self.train_dataset), {"eval_f1": f1}

    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        """ประเมิน global model บน local val set"""
        self.set_parameters(parameters)
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir="outputs/fl_ner/eval_tmp",
                report_to="none",
                disable_tqdm=True,
            ),
            eval_dataset=self.eval_dataset,
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
            compute_metrics=make_compute_metrics(self.id2label),
        )
        metrics = trainer.evaluate()
        loss = metrics.get("eval_loss", 0.0)
        f1 = metrics.get("eval_f1", 0.0)
        logger.info(f"[Client {self.client_id}] evaluate() — F1: {f1:.4f}, Loss: {loss:.4f}")
        return loss, len(self.eval_dataset), {"eval_f1": f1}
