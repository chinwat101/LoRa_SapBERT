# สรุปการดำเนินงาน: โครงการสร้าง Clinical NER สำหรับฉลากยา (Task 2)

เอกสารนี้รวบรวมแนวทางการจัดการข้อมูล โครงสร้างไฟล์ และขั้นตอนการปรับจูน (Fine-tune) โมเดลสำหรับงาน Named Entity Recognition (NER) โดยเน้นข้อมูลที่มาจาก **ฉลากยา** เป็นหลัก

> **📌 อัปเดต 2026-09-09:** Task 2 **เปลี่ยนโมเดลหลักจาก WangchanBERTa เป็น SapBERT Thai** (merged model จาก Task 1 — `outputs/merged_model_1`) เรียบร้อยแล้ว โดย WangchanBERTa (`airesearch/wangchanberta-base-att-spm-uncased`) ถูกเก็บผลไว้เป็น **baseline เปรียบเทียบ** (F1 = 0.8269)

---

## 1. จุดเริ่มต้นและเป้าหมาย

**เป้าหมายหลัก:** สร้างโมเดลที่สามารถสกัดข้อมูลสำคัญ (Entities) จากข้อความฉลากยาและคำอธิบายทางการแพทย์ที่เป็นภาษาไทย — **DRUG / DISEASE / SYMPTOM** (BIO scheme, 7 labels)

**โมเดลหลัก (ปัจจุบัน):** ⭐ **SapBERT Thai** — โมเดลการแพทย์ภาษาอังกฤษที่ผ่าน Vocabulary Expansion (คำไทย 20,000 คำ) + MLM LoRA pretraining จาก Task 1 จนเข้าใจภาษาไทย (Perplexity 407,962 → 19.27) แล้ว merge เป็น standalone model ที่ `outputs/merged_model_1` จากนั้นนำมา fine-tune Token Classification ต่อด้วย LoRA

**โมเดล Baseline (เดิม):** WangchanBERTa — เคยใช้เป็นทางออกชั่วคราวช่วงที่ Task 1 ติด Frozen Embedding Bug เทรนเสร็จที่ F1 = 0.8269 (epoch 28/30, adapter 11 MB) กลายเป็นเกณฑ์วัดว่า SapBERT ข้ามภาษา "สู้โมเดลไทยแท้ได้ไหม"

**เทคนิคที่ใช้:** LoRA (Low-Rank Adaptation) — `TaskType.TOKEN_CLS`, r=16, alpha=32, dropout=0.1, target modules: `query`, `key`, `value`, `dense`

---

## 2. โครงสร้างไฟล์ที่เกี่ยวข้อง (เฉพาะส่วนฉลากยา)

```text
thai_bert_lora/
├── configs/
│   ├── sapbert_ner_config.yaml       # ★ Config หลัก: Fine-tune SapBERT NER (100 epochs, LR 3e-4)
│   └── ner_config.yaml               # Config เดิมของ WangchanBERTa baseline
├── data/
│   ├── raw/
│   │   └── data_EN.csv               # ⚠️ ต้นฉบับข้อมูลฉลากยา (49 records)
│   ├── interim/
│   │   └── wangchanberta/            # Base model baseline (404 MB)
│   └── processed/
│       ├── ner_bio_th.jsonl          # ★ ข้อมูลเทรนภาษาไทย (44 examples, character-level spans)
│       ├── ner_bio_en.jsonl          # ข้อมูลภาษาอังกฤษ (43 examples)
│       └── ner_label_schema.json     # ตัวแมป Label (ID <-> Name)
├── scripts/
│   ├── convert_ner_dataset.py        # สคริปต์: แปลง CSV เป็น BIO/span Format
│   ├── train_sapbert_ner.py          # ★ สคริปต์หลัก: Fine-tune SapBERT Thai NER (LoRA + auto-merge)
│   ├── train_wangchan_ner.py         # สคริปต์ baseline: Fine-tune WangchanBERTa NER
│   └── inference_ner.py              # สคริปต์: ทดสอบโมเดล (Interactive + CLI, รองรับทั้งสองโมเดล)
└── outputs/
    ├── merged_model_1/               # SapBERT Thai MLM merged (ฐานของ NER)
    ├── merged_model_1_ner_1/         # NER รอบแรก (F1 = 0.7714)
    ├── merged_model_1_ner_base/      # ★ NER รอบสอง (F1 = 0.8000) — โมเดลที่ดีที่สุด
    └── adapters/
        ├── merged_model_1_ner_1/         # LoRA adapter รอบแรก
        └── merged_model_1_ner_1_ner_1/   # LoRA adapter รอบสอง
```

---

## 3. การจัดการข้อมูลและการแก้ปัญหา (Data Management)

ข้อมูลเริ่มต้นคือไฟล์ `data_EN.csv` ซึ่งมีข้อมูลแบบตาราง (Column-based) เราต้องแปลงให้เป็นรูปแบบลำดับ (Sequence-based) พร้อม Entity spans เพื่อใช้เทรน NER

### 3.1 การกำหนด Entity (Entity Mapping)
กฎการแปลงคอลัมน์ใน CSV ให้เป็น Entity Tags:
- คอลัมน์ `drug_name_TH`, `drug_name_EN` → **DRUG**
- คอลัมน์ `disease_TH`, `disease_EN` → **DISEASE**
- คอลัมน์ `symptom_TH`, `symptom_EN` → **SYMPTOM**

**Label schema** (`ner_label_schema.json`): `{O: 0, B-DRUG: 1, I-DRUG: 2, B-DISEASE: 3, I-DISEASE: 4, B-SYMPTOM: 5, I-SYMPTOM: 6}`

### 3.2 ปัญหาคุณภาพข้อมูลและวิธีแก้ (ใน `convert_ner_dataset.py`)

1. **ปัญหาคำซ้อนทับกัน (Entity Overlap):**
   - **ปัญหา:** คำบางคำมีความหมายทับซ้อนกัน เช่น "อักเสบ" เป็น SYMPTOM แต่ "ข้ออักเสบรูมาตอยด์" เป็น DISEASE → tag ชนกัน
   - **วิธีแก้:** กำหนด Priority: `DRUG > DISEASE > SYMPTOM` หากพบคำซ้อนทับกัน ให้ยึดตาม Priority ที่สูงกว่า

2. **ปัญหาข้อมูลซ้ำ (Deduplication):**
   - **ปัญหา:** ใน CSV มีข้อมูลยาตัวเดียวกันซ้ำหลายแถว (เช่น Albendazole มี 6 แถว)
   - **วิธีแก้:** กรองซ้ำด้วยข้อความ (`description`) เป็น Key → เหลือ 87 ตัวอย่างไม่ซ้ำ (TH 44 / EN 43)

### 3.3 การจัดแนว Token (Subword Alignment — ย้ายมาอยู่ใน `train_sapbert_ner.py`)

`ner_bio_th.jsonl` เก็บ entity เป็น **character-level spans** (`start`/`end`/`label`) ทำให้ไม่ผูกกับ tokenizer ใด tokenizer หนึ่ง

ตอนเทรน ฟังก์ชัน `align_labels_to_subwords()` ใน `train_sapbert_ner.py` จะ:
- tokenize ข้อความด้วย **tokenizer ของ SapBERT Thai (WordPiece)** พร้อม `return_offsets_mapping=True`
- แมป character span → subword tokens: subword แรกของ entity = `B-`, subword ถัดไป = `I-`, special tokens = `-100` (ข้ามการคำนวณ loss)

*(เดิมสมัย WangchanBERTa baseline การ align ทำกับ SentencePiece tokenizer ใน `convert_ner_dataset.py` — หลักการเดียวกันคนละ tokenizer)*

---

## 4. การเทรน SapBERT Thai NER (`train_sapbert_ner.py`)

### 4.1 กลไกสำคัญของสคริปต์
1. **Auto-Detect Base Model:** ค้นหา `outputs/merged_model_N` ล่าสุดมาเป็น base อัตโนมัติ (override ได้ด้วย `--base-model`)
2. **LoRA Token Classification:** สวม adapter `TaskType.TOKEN_CLS` บน attention/dense layers
3. **Entity-level Metric:** ประเมินด้วย `seqeval` (วัด F1 ระดับ entity ทั้งคำ ไม่ใช่ระดับ subword) — ⚠️ ถ้าไม่มี seqeval จะ fallback เป็น token-level accuracy ภายใต้ชื่อ `eval_f1` เหมือนกัน (ดู CLAUDE.md ข้อ 3.3)
4. **Auto-split:** 80/20 (35 train / 9 val, seed 42) — ⚠️ ไม่ stratified (ดู CLAUDE.md ข้อ 3.7)
5. **Auto-Merge + Auto-increment Output:** เทรนเสร็จแล้ว `merge_and_unload()` ทันที บันทึกเป็น `outputs/<base>_ner_<N>` รันเลขอัตโนมัติ

### 4.2 ผลการเทรนจริง (2026-09-08, 100 epochs)

| รอบ | Base Model | F1 | Loss | Output |
|---|---|:---:|:---:|---|
| 1 | `merged_model_1` (MLM merged) | 0.7714 | 0.4283 | `outputs/merged_model_1_ner_1` |
| 2 | `merged_model_1_ner_1` (auto-detect เลือกมา) | **0.8000** ⭐ | 0.4676 | `outputs/merged_model_1_ner_base` (เดิมบันทึกเป็น `merged_model_1_ner_1_ner_1` แล้ว rename) |

> **หมายเหตุ:** รอบ 2 เกิดจาก `find_latest_merged_model()` auto-detect ไปเลือก `merged_model_1_ner_1` (ซึ่งเป็น NER model อยู่แล้ว) มาเทรน NER ซ้อนอีกชั้น — ได้ F1 ดีขึ้นเป็น 0.8000 แต่ถ้าต้องการเทรนจาก MLM base ใหม่ ให้ระบุ `--base-model outputs/merged_model_1` เสมอ

### 4.3 เปรียบเทียบกับ Baseline

| โมเดล | Best F1 | หมายเหตุ |
|---|:---:|---|
| **SapBERT Thai NER** ⭐ (โมเดลหลัก) | **0.8000** | `outputs/merged_model_1_ner_base`, 100 epochs |
| WangchanBERTa NER (baseline) | 0.8269 | epoch 28/30, adapter 11 MB — ผลเชิงประวัติ (โฟลเดอร์ `outputs/wangchan_ner/` ถูกลบไปแล้ว) |

➡️ **สรุป:** โมเดลการแพทย์ข้ามภาษาที่สอนไทยด้วย MLM (SapBERT Thai) ทำ NER ได้ใกล้เคียงโมเดลไทยแท้ (WangchanBERTa) มาก ต่างกันเพียง ~0.027 F1

---

## 5. การใช้งาน (Inference)

โมเดลผลลัพธ์เป็น **standalone merged model** (`BertForTokenClassification` ~450 MB) โหลดตรงได้โดยไม่ต้องพึ่งไลบรารี `peft`:

```bash
# แนะนำ: ระบุโมเดลที่ดีที่สุด (F1 = 0.8000) ตรงๆ
python scripts/inference_ner.py --model outputs/merged_model_1_ner_base --text "อะบิราเทอโรน ใช้รักษามะเร็งต่อมลูกหมาก โดยมีอาการแพร่กระจาย"

# Interactive mode
python scripts/inference_ner.py
```

> ⚠️ **ระวัง auto-detect:** ถ้าไม่ใส่ `--model` สคริปต์จะเลือก `merged_model_1_ner_1` (F1 0.7714) เพราะ logic จัดลำดับตามตัวเลขท้ายชื่อ — โมเดล F1 0.8000 ถูก rename เป็น `merged_model_1_ner_base` ซึ่งลงท้ายไม่ใช่ตัวเลข

**ตัวอย่างผลลัพธ์ที่โมเดลทำได้:**
```text
  [DRUG    ]  'อะบิราเทอโรน'  (pos 0–12)
  [DISEASE ]  'มะเร็งต่อมลูกหมาก'  (pos 21–38)
  [SYMPTOM ]  'แพร่กระจาย'  (pos 49–59)
```

การแสดงผลใน Terminal ใช้สีแยกประเภท: สีฟ้า = DRUG, สีแดง = DISEASE, สีเหลือง = SYMPTOM

---

## 6. ข้อสังเกตและก้าวต่อไป

1. **ข้อมูลเล็กมาก** — training data ภาษาไทยมีเพียง 35 ประโยค (val 9) โมเดลอาจทำนายผิดในยา/โรคที่ไม่เคยเห็น → **ต้องเพิ่มข้อมูลฉลากยาใน `data_EN.csv` ให้หลากหลายขึ้น** นี่คือก้าวต่อไปที่สำคัญที่สุด
2. **Val set ไม่ stratified** — ด้วยข้อมูล 9 ประโยค val อาจขาด entity บางประเภท ทำให้ F1 แกว่งระหว่าง run
3. **การเทรน NER ซ้อน NER (รอบ 2)** ให้ผลดีกว่ารอบแรกเล็กน้อย — ควรทดลองอย่างเป็นระบบมากขึ้น (เช่น เทรนจาก `merged_model_1` ด้วย seed/epoch ที่ต่างกัน) ก่อนสรุป
4. Baseline WangchanBERTa ยังรันซ้ำได้ด้วย `scripts/train_wangchan_ner.py` + `configs/ner_config.yaml` (base model อยู่ใน `data/interim/wangchanberta/`) และ merge ด้วย `scripts/merge_wangchan_ner.py` ถ้าต้องการเปรียบเทียบใหม่
