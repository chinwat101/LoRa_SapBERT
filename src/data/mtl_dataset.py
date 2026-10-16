"""
==============================================================================
Multi-Task Learning Dataset Module: NER + Relation Extraction
==============================================================================
ไฟล์นี้สร้างขึ้นเพื่อ:
  โหลดและแปลงข้อมูลจาก ner_and_Relation.jsonl ให้อยู่ในรูปแบบที่
  JointNERAndREModel ใช้งานได้ โดยรองรับทั้ง NER labels (BIO)
  และ Relation pairs (positive + negative sampling)

โครงสร้าง batch output:
  - input_ids          : Tensor[B, L]    — token ids
  - attention_mask     : Tensor[B, L]    — 1=real token, 0=padding
  - ner_labels         : Tensor[B, L]    — BIO label ids, -100=padding/special
  - pairs              : Tensor[P, 2]    — [tok_start_head, tok_start_tail]
  - rel_labels         : Tensor[P]       — 0=NO_REL, 1=TREATS, 2=INDICATED_FOR
  - pair_sample_idx    : Tensor[P]       — index ของ sample ใน batch
==============================================================================
"""

import json
import random
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Label Mappings (Default)
# ---------------------------------------------------------------------------
DEFAULT_NER2ID = {
    "O": 0,
    "B-DRUG": 1, "I-DRUG": 2,
    "B-DISEASE": 3, "I-DISEASE": 4,
    "B-SYMPTOM": 5, "I-SYMPTOM": 6,
}

DEFAULT_REL2ID = {
    "NO_RELATION": 0,
    "TREATS": 1,
    "INDICATED_FOR": 2,
}

IGNORE_INDEX = -100   # สำหรับ CrossEntropyLoss (padding / special tokens)


# ---------------------------------------------------------------------------
# MTLDataset
# ---------------------------------------------------------------------------
class MTLDataset(Dataset):
    """
    Dataset สำหรับ Multi-Task Learning (NER + Relation Extraction)

    อ่านข้อมูลจาก JSONL format:
      { "text", "entities": [{start, end, label}],
        "relations": [{head, tail, type}], ... }

    แปลงเป็น tokenized tensors พร้อม NER labels และ Relation pairs
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        ner2id: Dict[str, int] = None,
        rel2id: Dict[str, int] = None,
        max_seq_length: int = 256,
        max_neg_pairs: int = 10,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.ner2id = ner2id or DEFAULT_NER2ID
        self.rel2id = rel2id or DEFAULT_REL2ID
        self.max_seq_length = max_seq_length
        self.max_neg_pairs = max_neg_pairs
        self.rng = random.Random(seed)

        # โหลด raw data
        with open(data_path, "r", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]

        # แปลงเป็น features
        self.features = [self._convert(sample) for sample in raw]
        # กรอง sample ที่ tokenize แล้วได้ผิดปกติ
        self.features = [f for f in self.features if f is not None]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict:
        return self.features[idx]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _convert(self, sample: Dict) -> Optional[Dict]:
        """
        แปลง 1 sample เป็น feature dict พร้อม:
          - input_ids, attention_mask
          - ner_labels (aligned กับ subword tokens)
          - pairs + rel_labels (positive + negative)
        """
        text = sample["text"]
        entities = sample.get("entities", [])
        relations = sample.get("relations", [])

        # --- Tokenize พร้อม offset mapping ---
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offsets = encoding["offset_mapping"]   # [(char_start, char_end), ...]
        num_tokens = len(input_ids)

        # --- Align NER labels กับ subword tokens ---
        ner_labels = self._align_ner_labels(text, entities, offsets, num_tokens)

        # --- Map entities → token positions ---
        entity_token_spans = self._map_entity_token_spans(entities, offsets, num_tokens)

        # --- สร้าง Relation pairs (positive + negative) ---
        pairs, rel_labels = self._build_pairs(
            entities, relations, entity_token_spans
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ner_labels": ner_labels,
            "pairs": pairs,                    # List[Tuple[int,int]]
            "rel_labels": rel_labels,          # List[int]
        }

    def _align_ner_labels(
        self,
        text: str,
        entities: List[Dict],
        offsets: List[Tuple[int, int]],
        num_tokens: int,
    ) -> List[int]:
        """
        สร้าง BIO label per token โดย align กับ character offsets
        Returns list ขนาด num_tokens โดย special tokens และ padding = IGNORE_INDEX
        """
        # สร้าง char-level label map
        char_label = {}
        for ent in entities:
            c_start, c_end, label = ent["start"], ent["end"], ent["label"]
            for i, c in enumerate(range(c_start, c_end)):
                prefix = "B" if i == 0 else "I"
                char_label[c] = f"{prefix}-{label}"

        ner_labels = []
        for tok_idx, (c_start, c_end) in enumerate(offsets):
            if c_start == c_end:
                # special token ([CLS], [SEP]) หรือ padding
                ner_labels.append(IGNORE_INDEX)
                continue

            # ใช้ label ของ character แรกใน subword
            label_str = char_label.get(c_start, "O")
            ner_labels.append(self.ner2id.get(label_str, 0))

        return ner_labels

    def _map_entity_token_spans(
        self,
        entities: List[Dict],
        offsets: List[Tuple[int, int]],
        num_tokens: int,
    ) -> List[Optional[int]]:
        """
        แปลง character span ของแต่ละ entity → token index ของ subword แรก
        Returns list ขนาด len(entities), None ถ้าหา token ไม่เจอ (truncated)
        """
        spans = []
        for ent in entities:
            c_start = ent["start"]
            tok_idx = None
            for i, (cs, ce) in enumerate(offsets):
                if cs <= c_start < ce:
                    tok_idx = i
                    break
            spans.append(tok_idx)
        return spans

    def _build_pairs(
        self,
        entities: List[Dict],
        relations: List[Dict],
        entity_token_spans: List[Optional[int]],
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        สร้าง (head_tok, tail_tok, label) pairs สำหรับ RE:
        - Positive: จาก ground-truth relations
        - Negative: สุ่ม entity pairs ที่ไม่มีใน relations (NO_RELATION)
        """
        no_rel_id = self.rel2id["NO_RELATION"]

        # สร้าง set ของ positive pairs
        pos_set = set()
        pairs = []
        rel_labels = []

        for rel in relations:
            h_idx, t_idx, rel_type = rel["head"], rel["tail"], rel["type"]
            h_tok = entity_token_spans[h_idx] if h_idx < len(entity_token_spans) else None
            t_tok = entity_token_spans[t_idx] if t_idx < len(entity_token_spans) else None

            if h_tok is None or t_tok is None:
                continue  # entity ถูก truncate

            pairs.append((h_tok, t_tok))
            rel_labels.append(self.rel2id.get(rel_type, no_rel_id))
            pos_set.add((h_idx, t_idx))

        # สุ่ม negative pairs
        n_ent = len(entities)
        neg_candidates = [
            (i, j)
            for i in range(n_ent)
            for j in range(n_ent)
            if i != j and (i, j) not in pos_set
        ]
        self.rng.shuffle(neg_candidates)
        neg_candidates = neg_candidates[: self.max_neg_pairs]

        for h_idx, t_idx in neg_candidates:
            h_tok = entity_token_spans[h_idx]
            t_tok = entity_token_spans[t_idx]
            if h_tok is None or t_tok is None:
                continue
            pairs.append((h_tok, t_tok))
            rel_labels.append(no_rel_id)

        return pairs, rel_labels


# ---------------------------------------------------------------------------
# Collate Function
# ---------------------------------------------------------------------------
def mtl_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function สำหรับ DataLoader
    รวม samples ใน batch และ pad ให้มีความยาวเท่ากัน
    Relation pairs จากทุก sample ถูกรวมกัน พร้อม pair_sample_idx บอกว่า pair นี้มาจาก sample ไหน
    """
    pad_id = 0

    # --- Pad input_ids & attention_mask & ner_labels ---
    max_len = max(len(s["input_ids"]) for s in batch)

    input_ids_list = []
    attention_mask_list = []
    ner_labels_list = []

    for s in batch:
        seq_len = len(s["input_ids"])
        pad_len = max_len - seq_len

        input_ids_list.append(s["input_ids"] + [pad_id] * pad_len)
        attention_mask_list.append(s["attention_mask"] + [0] * pad_len)
        ner_labels_list.append(s["ner_labels"] + [IGNORE_INDEX] * pad_len)

    # --- รวม Relation pairs จากทุก samples ---
    all_pairs = []
    all_rel_labels = []
    all_pair_sample_idx = []

    for batch_idx, s in enumerate(batch):
        for pair, rel_label in zip(s["pairs"], s["rel_labels"]):
            all_pairs.append(list(pair))
            all_rel_labels.append(rel_label)
            all_pair_sample_idx.append(batch_idx)

    result = {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        "ner_labels": torch.tensor(ner_labels_list, dtype=torch.long),
    }

    if all_pairs:
        result["pairs"] = torch.tensor(all_pairs, dtype=torch.long)
        result["rel_labels"] = torch.tensor(all_rel_labels, dtype=torch.long)
        result["pair_sample_idx"] = torch.tensor(all_pair_sample_idx, dtype=torch.long)
    else:
        # ถ้า batch ไม่มี entity pair เลย
        result["pairs"] = torch.zeros((0, 2), dtype=torch.long)
        result["rel_labels"] = torch.zeros(0, dtype=torch.long)
        result["pair_sample_idx"] = torch.zeros(0, dtype=torch.long)

    return result
