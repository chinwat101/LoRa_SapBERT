#!/usr/bin/env python3
"""
scripts/prepare_medical_qa_ner.py

สคริปต์สกัด Entity (DRUG / DISEASE / SYMPTOM) จากไฟล์ QA / ข้อความการแพทย์
และแปลงเป็น BIO Scheme Token Classification สำหรับเทรน BERT NER Model
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Label Schema (Standard 7 Labels)
# ──────────────────────────────────────────────────────────────────────────────
LABEL2ID = {
    "O":         0,
    "B-DRUG":    1,
    "I-DRUG":    2,
    "B-DISEASE": 3,
    "I-DISEASE": 4,
    "B-SYMPTOM": 5,
    "I-SYMPTOM": 6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
ENTITY_PRIORITY = {"DRUG": 3, "DISEASE": 2, "SYMPTOM": 1}

# ──────────────────────────────────────────────────────────────────────────────
# Medical Pattern Regex Rules
# ──────────────────────────────────────────────────────────────────────────────
SYMPTOM_PATTERNS = [
    r"(?:มีอาการ|อาการ|เกิดอาการ)\s*([ก-๙a-zA-Z0-9\s]+?)(?=\s*(?:และ|ซึ่ง|โดย|ร่วมกับ|เนื่องจาก|จาก|\.|\,|$))",
    r"(?:ปวด|ไข้|อาเจียน|ท้องร่วง|คลื่นไส้|อ่อนแรง|ซึม|ไอ|เวียนศีรษะ|ผื่น|แน่นหน้าอก|หายใจลำบาก|เหนื่อยหอบ|เสมหะเหนียวข้น|ท้องเสีย|บวม)[ก-๙a-zA-Z0-9]*",
]

DISEASE_PATTERNS = [
    r"(?:โรค|ภาวะ|อาการป่วยจากโรค|มะเร็ง|กลุ่มอาการ)\s*([ก-๙a-zA-Z0-9\s]+?)(?=\s*(?:และ|ซึ่ง|โดย|ร่วมกับ|เนื่องจาก|จาก|\.|\,|$))",
    r"(?:ข้อเสื่อม|ตับอักเสบ|ไทรอยด์|ความดันโลหิตสูง|เบาหวาน|ปอดอักเสบ|ติดเชื้อ|colitis|dementia|pneumonia|ischemia|DCS)[ก-๙a-zA-Z0-9]*",
]

DRUG_PATTERNS = [
    r"(?:ยา|ยากลุ่ม|ใช้อย่าง|ทานยา|รับประทานยา|ยาฉีด)\s*([a-zA-Z0-9\s\-]+?)(?=\s*(?:และ|ซึ่ง|โดย|ร่วมกับ|เพื่อ|\.|\,|$))",
    r"(?:naproxen|thioamide|aspirin|paracetamol|ibuprofen|prednisolone|penicillin|amoxicillin)[a-zA-Z0-9]*",
]

def extract_entities_regex(text: str) -> List[Dict]:
    """สกัด Entity เบื้องต้นด้วย Pattern Matching Rules"""
    spans = []
    
    # 1. DRUG
    for pat in DRUG_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            s, e = m.span()
            matched_text = text[s:e].strip()
            if len(matched_text) >= 2:
                spans.append({"start": s, "end": e, "label": "DRUG", "text": matched_text})
                
    # 2. DISEASE
    for pat in DISEASE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            s, e = m.span()
            matched_text = text[s:e].strip()
            if len(matched_text) >= 2:
                spans.append({"start": s, "end": e, "label": "DISEASE", "text": matched_text})
                
    # 3. SYMPTOM
    for pat in SYMPTOM_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            s, e = m.span()
            matched_text = text[s:e].strip()
            if len(matched_text) >= 2:
                spans.append({"start": s, "end": e, "label": "SYMPTOM", "text": matched_text})
                
    return resolve_overlapping_spans(spans)

def resolve_overlapping_spans(spans: List[Dict]) -> List[Dict]:
    """เคลียร์ span ที่ทับซ้อนกัน โดยยึดตาม Priority: DRUG > DISEASE > SYMPTOM"""
    if not spans:
        return []
    ordered = sorted(
        spans,
        key=lambda s: (
            ENTITY_PRIORITY.get(s["label"], 0),
            s["end"] - s["start"],
            -s["start"],
        ),
        reverse=True,
    )
    kept = []
    occupied = []
    for span in ordered:
        s, e = span["start"], span["end"]
        if not any(not (e <= ks or s >= ke) for ks, ke in occupied):
            kept.append(span)
            occupied.append((s, e))
    return sorted(kept, key=lambda s: s["start"])

if __name__ == "__main__":
    print("Medical QA to NER Processor Script Ready.")
