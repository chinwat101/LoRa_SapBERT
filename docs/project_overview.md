# Project Overview: Thai Clinical NER (SapBERT) + MLM Fine-tuning
**Path:** `I:\NECTEC\thai_bert_lora`  
**Updated:** 2026-09-09 — Task 2 เปลี่ยนโมเดลหลักจาก WangchanBERTa เป็น **SapBERT Thai** (WangchanBERTa = baseline)

---

## 1. โครงสร้างไฟล์ (File Structure)

```
thai_bert_lora/
├── configs/                          # Config files แยกตาม task
│   ├── model_config.yaml             # Base model path (SapBERT/extended)
│   ├── lora_config.yaml              # LoRA config สำหรับ MLM (original)
│   ├── training_config.yaml          # Training config สำหรับ MLM (original)
│   ├── data_config.yaml              # Data paths (original pipeline)
│   ├── sapbert_ner_config.yaml       # ★ Config หลักสำหรับ NER (SapBERT Thai, 100 epochs)
│   ├── ner_config.yaml               # Config สำหรับ WangchanBERTa NER (baseline)
│   └── wangchan_mlm_config.yaml      # Config สำหรับ WangchanBERTa MLM (optional)
│
├── data/
│   ├── raw/
│   │   ├── data_EN.csv               # ★ ต้นทาง: ฉลากยา (49 rows)
│   │   └── lexitron/telex.csv        # พจนานุกรม Lexitron (original)
│   ├── interim/
│   │   ├── wangchanberta/            # ★ WangchanBERTa base model (404 MB)
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── tokenizer_config.json
│   │   │   └── tokenizer.json
│   │   ├── extended_base/            # SapBERT + Thai vocab (original)
│   │   ├── lexitron_clean.jsonl      # Lexitron cleaned (original)
│   │   ├── thai_spm.model            # SentencePiece model (original)
│   │   └── thai_spm.vocab
│   └── processed/
│       ├── ner_spans.jsonl           # ★ NER dataset: character-level spans (87 examples)
│       ├── ner_bio_th.jsonl          # ★ NER Thai: subword BIO tags (44 examples)
│       ├── ner_bio_en.jsonl          # ★ NER English: subword BIO tags (43 examples)
│       ├── ner_label_schema.json     # ★ Label schema: DRUG/DISEASE/SYMPTOM
│       ├── mlm_train.jsonl           # ★ MLM corpus train (194 texts)
│       ├── mlm_val.jsonl             # ★ MLM corpus val (22 texts)
│       ├── train.jsonl               # MLM dataset train (original)
│       ├── validation.jsonl          # MLM dataset val (original)
│       └── test.jsonl                # MLM dataset test (original)
│
├── scripts/                          # Runnable scripts
│   │
│   ├── [Data Preparation]
│   ├── download_wangchanberta.py     # ★ Download WangchanBERTa model
│   ├── convert_ner_dataset.py        # ★ แปลง data_EN.csv → NER JSONL
│   ├── build_wangchan_mlm_corpus.py  # ★ Build MLM corpus จาก NER data + CSV
│   ├── preprocess_lexitron.py        # Preprocess Lexitron (original)
│   ├── build_mlm_dataset.py          # Build MLM dataset (original)
│   ├── build_wiki_corpus.py          # Wiki corpus builder (original, ล้มเหลว)
│   │
│   ├── [Training]
│   ├── train_sapbert_ner.py          # ★ หลัก: Fine-tune SapBERT Thai NER (LoRA + auto-merge)
│   ├── train_wangchan_ner.py         # Fine-tune WangchanBERTa NER (baseline)
│   ├── train_wangchan_mlm.py         # Continued pre-training MLM (optional)
│   ├── train_lora_mlm.py             # MLM training (SapBERT pipeline — Task 1)
│   │
│   ├── [Inference & Evaluation]
│   ├── inference_ner.py              # ★ NER inference (interactive + CLI)
│   ├── test_inference.py             # MLM inference test (original)
│   ├── evaluate_mlm.py               # MLM evaluation (original)
│   ├── merge_lora.py                 # Merge LoRA adapter into base model
│   │
│   └── [Analysis]
│       ├── inspect_model_modules.py  # ดู module names ใน model
│       ├── inspect_dataset.py        # ดู dataset stats
│       └── analyze_tokenizer.py     # วิเคราะห์ tokenizer
│
├── src/                              # Library code
│   ├── data/
│   │   ├── loader.py                 # JSONL dataset loader
│   │   ├── cleaner.py                # Text cleaning utilities
│   │   ├── corpus_builder.py         # Corpus building logic
│   │   └── splitter.py               # Train/val/test split
│   ├── models/
│   │   ├── model_loader.py           # Load base model + tokenizer
│   │   └── lora.py                   # Apply LoRA, save adapter
│   └── training/
│       ├── trainer.py                # MLM training loop (original)
│       └── callbacks.py              # Logging + best model callbacks
│
├── outputs/                          # Training outputs (ไม่ commit)
│   ├── merged_model_1/               # ★ SapBERT Thai MLM merged (Task 1, PPL 19.27)
│   ├── merged_model_1_ner_1/         # NER รอบแรก (F1 = 0.7714)
│   ├── merged_model_1_ner_base/      # ★ NER รอบสอง (F1 = 0.8000) — โมเดลที่ดีที่สุด
│   ├── adapters/
│   │   ├── lexitron_wiki_thai_lora_r32/    # MLM LoRA adapter (Task 1)
│   │   ├── merged_model_1_ner_1/           # NER adapter รอบแรก
│   │   └── merged_model_1_ner_1_ner_1/     # NER adapter รอบสอง
│   ├── checkpoints/                  # Training checkpoints
│   └── logs/                         # ไฟล์ log + TensorBoard
│   # หมายเหตุ: โฟลเดอร์ wangchan_ner/ (baseline เดิม) ถูกลบไปแล้ว — เหลือตัวเลข F1 อ้างอิง
│
└── docs/
    ├── task_1.md                     # บันทึก SapBERT MLM pipeline (Task 1)
    ├── task_2.md                     # ★ บันทึก Clinical NER (SapBERT หลัก + Wangchan baseline)
    ├── result.md                     # สรุปผลลัพธ์เชิงลึกทั้งหมด
    ├── training_plan.md              # แผนอบรม 1 ชั่วโมงของโปรเจกต์
    └── project_overview.md           # ★ ไฟล์นี้
```

---

## 2. สองเส้นทางการทำงาน (Two Pipelines)

### 🔵 Pipeline A — SapBERT + Thai Vocabulary (Original)
> เป้าหมาย: สอนโมเดลภาษาอังกฤษ (SapBERT) ให้เข้าใจภาษาไทยด้วย vocab expansion + MLM

```
telex.csv (Lexitron)
    │
    ▼ scripts/preprocess_lexitron.py
data/interim/lexitron_clean.jsonl
    │
    ▼ scripts/extend_tokenizer.py
data/interim/extended_base/      (SapBERT + 20,000 Thai tokens)
    │
    ▼ scripts/build_mlm_dataset.py
data/processed/train.jsonl + validation.jsonl
    │
    ▼ scripts/train_lora_mlm.py   (config: lora_config.yaml + training_config.yaml)
outputs/checkpoints/ + outputs/adapter/
```

**สถานะ:** ทำงานได้ แต่โมเดล overfit กับรูปแบบพจนานุกรม

---

### 🟢 Pipeline B — SapBERT Thai + NER (★ หลักปัจจุบัน)
> เป้าหมาย: Fine-tune SapBERT Thai (merged model จาก Task 1) สำหรับ Named Entity Recognition ข้อมูลฉลากยา

```
data/raw/data_EN.csv                  (ฉลากยา 49 rows)
    │
    ▼ scripts/convert_ner_dataset.py
data/processed/ner_bio_th.jsonl       (44 examples ภาษาไทย, character-level spans)
data/processed/ner_bio_en.jsonl       (43 examples ภาษาอังกฤษ)
data/processed/ner_label_schema.json
    │
    ▼ scripts/train_sapbert_ner.py    (config: sapbert_ner_config.yaml)
    │   base = outputs/merged_model_1 (auto-detect merged_model_N ล่าสุด)
    │   LoRA TOKEN_CLS (r=16, α=32) → train 100 epochs → auto-merge
    ▼
outputs/merged_model_1_ner_base/      (★ F1 = 0.8000, standalone ~450 MB)
    │
    ▼ scripts/inference_ner.py --model outputs/merged_model_1_ner_base
→ รู้จัก DRUG / DISEASE / SYMPTOM ในข้อความภาษาไทย
```

**สถานะ:** ✅ เทรนเสร็จ | **F1 = 0.8000** (100 epochs) | รอบแรกที่เทรนจาก `merged_model_1` ตรงๆ ได้ F1 = 0.7714 (`merged_model_1_ner_1`)

---

### ⚪ Pipeline B0 — WangchanBERTa + NER (Baseline — เชิงประวัติ)
> เคยเป็นทางออกชั่วคราวช่วงที่ Task 1 ติด Frozen Embedding Bug — ปัจจุบันเก็บผลไว้เป็นเกณฑ์เปรียบเทียบ

```
data/raw/data_EN.csv              (ฉลากยา 49 rows)
    │
    ▼ scripts/convert_ner_dataset.py
data/processed/ner_bio_th.jsonl   (44 examples ภาษาไทย)
    │
    ▼ scripts/train_wangchan_ner.py   (config: ner_config.yaml)
outputs/wangchan_ner/adapter/     (LoRA adapter 11 MB — โฟลเดอร์ถูกลบแล้ว)
    │
    ▼ scripts/inference_ner.py
→ รู้จัก DRUG / DISEASE / SYMPTOM ในข้อความภาษาไทย
```

**สถานะ:** ✅ เทรนเสร็จ | F1 = 0.8269 (30 epochs, 12 วินาที) — เป็น baseline ให้ Pipeline B เปรียบเทียบ

---

### 🟡 Pipeline C — WangchanBERTa + Domain MLM (Optional)
> เป้าหมาย: Continued pre-training บน domain text (biomedical Thai)

```
data/processed/ner_spans.jsonl + data/raw/data_EN.csv (description field)
    │
    ▼ scripts/build_wangchan_mlm_corpus.py
data/processed/mlm_train.jsonl (194 texts)
data/processed/mlm_val.jsonl   (22 texts)
    │
    ▼ scripts/train_wangchan_mlm.py   (config: wangchan_mlm_config.yaml)
outputs/wangchan_mlm/adapter/
```

**สถานะ:** พร้อมรัน (corpus เล็ก แนะนำเพิ่มข้อมูลก่อน)

---

## 3. Entity Types และ Mapping

| คอลัมน์ใน CSV | Entity Type | Label BIO |
|---|---|---|
| `drug_name_TH`, `drug_name_EN` | DRUG | B-DRUG, I-DRUG |
| `disease_TH`, `disease_EN` | DISEASE | B-DISEASE, I-DISEASE |
| `symptom_TH`, `symptom_EN` | SYMPTOM | B-SYMPTOM, I-SYMPTOM |
| (ข้อความอื่น) | — | O |

**Label schema** (`ner_label_schema.json`):
```json
{ "O": 0, "B-DRUG": 1, "I-DRUG": 2,
  "B-DISEASE": 3, "I-DISEASE": 4,
  "B-SYMPTOM": 5, "I-SYMPTOM": 6 }
```

---

## 4. การจัดการ Data Quality Issues

### Fix 1 — Entity Overlap (ใน `convert_ner_dataset.py`)
ปัญหา: "อักเสบ" ถูก tag ทั้ง DISEASE (ส่วนหนึ่งของ "ข้ออักเสบรูมาตอยด์") และ SYMPTOM พร้อมกัน  
แก้: Priority DRUG > DISEASE > SYMPTOM — spans ที่ overlap จะใช้ priority สูงกว่า

### Fix 2 — Subword Tokenization (ใน `convert_ner_dataset.py`)
ปัญหา: ตัดด้วย whitespace ทำให้ "โรคที่ทางเดินหายใจ" = 1 token (ผิดสำหรับ SentencePiece)  
แก้: ใช้ WangchanBERTa tokenizer + `return_offsets_mapping=True` align character spans → subword tokens

### Fix 3 — Deduplication (ใน `convert_ner_dataset.py`)
ปัญหา: ยาชนิดเดียวกันปรากฏหลายแถวใน CSV (เช่น albendazole 6 แถว)  
แก้: Deduplicate ด้วย `(text, lang)` key → จาก 98 เหลือ 87 unique examples

---

## 5. คำสั่ง Quick Reference

```bash
cd I:\NECTEC\thai_bert_lora
source .venv/bin/activate        # Linux/WSL (Windows: .venv\Scripts\activate)

# 1. สร้าง NER dataset (ถ้า data_EN.csv เปลี่ยน)
python scripts/convert_ner_dataset.py

# 2. Train NER (★ หลัก — SapBERT Thai, เทรนเสร็จ auto-merge ให้เลย)
python scripts/train_sapbert_ner.py
#    ถ้าต้องการระบุ base ชัดเจน (กัน auto-detect ไปเลือก NER model เดิมมาซ้อน):
python scripts/train_sapbert_ner.py --base-model outputs/merged_model_1

# 3. Inference (CLI mode — ระบุโมเดลที่ดีที่สุด F1=0.8000)
python scripts/inference_ner.py --model outputs/merged_model_1_ner_base --text "ชื่อยา ใช้รักษา โรค"

# 4. Inference (Interactive mode)
python scripts/inference_ner.py

# 5. (Baseline) Train WangchanBERTa NER เปรียบเทียบ
python scripts/train_wangchan_ner.py

# 6. Build MLM corpus + train MLM (optional — Pipeline C)
python scripts/build_wangchan_mlm_corpus.py
python scripts/train_wangchan_mlm.py
```

---

## 6. ผลการเทรน NER

### ★ Pipeline B — SapBERT Thai NER (โมเดลหลัก, 2026-09-08)

| รอบ | Base Model | Best F1 | Loss | Output |
|---|---|:---:|:---:|---|
| 1 | `merged_model_1` (MLM merged) | 0.7714 | 0.4283 | `outputs/merged_model_1_ner_1` |
| 2 | `merged_model_1_ner_1` (auto-detect) | **0.8000** | 0.4676 | `outputs/merged_model_1_ner_base` |

**Config:** 100 epochs, LoRA r=16 / α=32 (TOKEN_CLS), LR 3e-4, auto-split 35/9  
**ผลลัพธ์:** standalone merged model (`BertForTokenClassification` ~450 MB) — ใช้ inference ได้โดยไม่ต้องพึ่ง `peft`

### Pipeline B0 — WangchanBERTa NER (Baseline — เชิงประวัติ)

| Epoch | F1 | Precision | Recall | Loss |
|---|---|---|---|---|
| 6 | 0.196 | 0.220 | 0.177 | 0.978 |
| 10 | 0.604 | 0.644 | 0.569 | 0.424 |
| 16 | 0.811 | 0.782 | 0.843 | 0.235 |
| 20 | 0.822 | 0.786 | 0.863 | 0.182 |
| **28** | **0.827** | **0.811** | **0.843** | **0.237** |
| 30 | 0.819 | 0.796 | 0.843 | 0.222 |

**Best model:** Epoch 28 — F1 = **0.8269** | Adapter 11 MB (~2.7% parameters trained)

### สรุปเปรียบเทียบ
**SapBERT Thai (ข้ามภาษา + domain การแพทย์) F1 = 0.8000** ใกล้เคียง **WangchanBERTa (โมเดลไทยแท้) F1 = 0.8269** — ต่างกันเพียง ~0.027

---

## 7. Dependencies หลัก

```
transformers>=4.40
peft>=0.10           # LoRA
datasets
seqeval              # Entity-level F1 metric
torch
pandas
pyyaml
```
