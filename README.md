# LoRa_SapBERT: Cross-lingual Thai Adaptation for SapBERT & Clinical NER

โปรเจกต์ปรับจูนโมเดล **SapBERT** (โมเดลการแพทย์ภาษาอังกฤษ) ให้เข้าใจภาษาไทยผ่านเทคนิค **Vocabulary Expansion (คำไทย 20,000 คำ) + MLM LoRA Pre-training (Task 1)** และนำไป Fine-tune ต่องาน **Clinical Named Entity Recognition (Task 2)** สำหรับฉลากยาและข้อความทางการแพทย์

## 🛠️ การติดตั้งและการเตรียม Environment (.venv)

```bash
# 1. สร้าง Virtual Environment (.venv)
python -m venv .venv

# 2. เปิดใช้งาน Virtual Environment
# บน Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# บน Windows (CMD):
.\.venv\Scripts\activate.bat
# บน Linux / macOS:
source .venv/bin/activate

# 3. อัปเดต pip และติดตั้ง Dependencies ทั้งหมด
python -m pip install --upgrade pip
pip install -r requirements.txt
```
```bash

#ทุกไฟล์สามารถรันโดยผ่าน
main.py

```
---

## 🚀 ขั้นตอนการรันทั้งหมด (Step-by-Step Pipeline)

### 🔹 TASK 1: MLM Pre-training (สอน SapBERT ให้เข้าใจภาษาไทย)

#### Step 1: ดาวน์โหลดโมเดลและคลังข้อมูลตั้งต้น
```bash
# 1.1 ดาวน์โหลดโมเดล SapBERT ภาษาอังกฤษต้นฉบับ
python scripts/download_sapbert.py

# 1.2 ดาวน์โหลดคลังคำศัพท์ Lexitron 2.0 จาก NECTEC Open Data Portal
python scripts/download_lexitron.py

# 1.3 ดาวน์โหลดคลังบทความ Thai Wikipedia จาก HuggingFace
python scripts/build_wiki_corpus.py
```

#### Step 2: จัดการและเตรียมข้อมูล MLM
```bash
# 2.1 ทำความสะอาดข้อมูล Lexitron CSV
python scripts/preprocess_lexitron.py

# 2.2 รวมข้อมูล Lexitron + Wikipedia และสร้าง Train/Val/Test Sets
python scripts/build_mlm_dataset.py --include-wiki
```

#### Step 3: ขยายคำศัพท์ + เทรน MLM + รวมร่างโมเดล
```bash
# 3.1 ขยาย Tokenizer เพิ่มคำศัพท์ภาษาไทย 20,000 คำ
python scripts/extend_tokenizer.py

# 3.2 เทรน Masked Language Modeling (MLM) ด้วย LoRA
python scripts/train_lora_mlm.py

# 3.3 รวมร่าง LoRA Adapter เข้ากับ Base Model (ได้ outputs/merged_model_1)
python scripts/merge_lora.py

# 3.4 ทดสอบผลลัพธ์การเติมคำภาษาไทยในช่องว่าง [MASK]
python scripts/test_inference.py

# 3.5  ประเมินค่า Perplexity และ MLM Loss
python scripts/evaluate_mlm.py
```

---

### 🔹 TASK 2: Clinical NER Fine-tuning (สกัดชื่อยา / โรค / อาการ)

#### Step 4: เตรียมข้อมูลฉลากยาสำหรับ NER
```bash
# แปลงข้อมูลฉลากยา (data_EN.csv) เป็นรูปแบบ BIO / Character Spans
python scripts/convert_ner_dataset.py
```

#### Step 5: Fine-tune โมเดล SapBERT Thai NER
```bash
# Fine-tune Token Classification ด้วย LoRA (ใช้ base model จาก outputs/merged_model_1)
python scripts/train_sapbert_ner.py --base-model outputs/merged_model_1
```

#### Step 6: ทดสอบ Inference สกัดเอนทิตีจากข้อความฉลากยา
```bash
# ทดสอบโมเดลสกัดชื่อยา (DRUG), โรค (DISEASE), และอาการ (SYMPTOM)
python scripts/inference_ner.py --model outputs/merged_model_1_ner_base --text "อะบิราเทอโรน ใช้รักษามะเร็งต่อมลูกหมาก โดยมีอาการแพร่กระจาย"
```