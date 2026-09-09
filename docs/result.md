# 📊 สรุปผลการดำเนินงานและเทคนิคเชิงลึก (Project Operations & Technical Deep Dive)

เอกสารนี้รวบรวมสรุปกระบวนการทำงาน เทคนิคเชิงลึกเบื้องหลัง และผลลัพธ์การดำเนินงานทั้งหมดของโครงการปรับแต่งโมเดลภาษาการแพทย์ภาษาอังกฤษ (**SapBERT**) ให้รองรับภาษาไทย และนำไปพัฒนาเป็นโมเดลวิเคราะห์ชื่อเฉพาะทางคลินิกและฉลากยา (**Thai Clinical NER**) 

---

## 🔵 ส่วนที่ 1: สรุปการดำเนินงาน 4 ขั้นตอนหลักของ Task 1 (SapBERT Thai Adaptation)

การปรับจูนโมเดล SapBERT (ซึ่งเดิมมีเฉพาะคลังคำศัพท์ภาษาอังกฤษ) ให้สามารถเข้าใจภาษาไทยเชิงบริบทได้อย่างสมบูรณ์ แบ่งออกเป็น 4 Block หลัก ดังนี้:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TASK 1 PIPELINE BLOCK                            │
├───────────────────┬───────────────────┬───────────────────┬─────────────────┤
│      BLOCK 1      │      BLOCK 2      │      BLOCK 3      │     BLOCK 4     │
│   Vocab Expansion │  Corpus Building  │  MLM Pre-training │  Weight Merge   │
│   & Smart Init    │   & Balancing     │    with LoRA      │  & Export Base  │
└───────────────────┴───────────────────┴───────────────────┴─────────────────┘
```

### 🔹 Block 1: การขยายคำศัพท์และการกำหนดค่าเริ่มต้นสมาร์ท (Vocabulary Expansion & Smart Initialization)
* **สิ่งที่ทำ:** สกัดคำศัพท์ภาษาไทยที่พบบ่อยจำนวน **20,000 คำ** (ทั้ง bare words และ `##subwords`) และเพิ่มตัวอักษรไทยเดี่ยวเป็น Fallback ด้วยสคริปต์ `scripts/extend_tokenizer.py`
* **ปัญหาเดิม:** SapBERT มีเฉพาะศัพท์ภาษาอังกฤษ ทำให้เมื่อเจอภาษาไทย โมเดลจะมองเห็นเป็น Unknown Tokens (`[UNK]`) หรือสับเป็นตัวอักษรย่อยยิบจนไม่สามารถวิเคราะห์บริบทได้
* **เทคนิคเชิงลึกที่ใช้:**
  1. **BPE Subword Tokenization:** ใช้ `SentencePiece` ฝึกสอน BPE Tokenizer บนคลังข้อความไทย เพื่อขยาย Tokenizer Vocab จาก 30,522 คำ เป็น 40,075 คำ กำจัดปัญหา `[UNK]` ออกไป 100%
  2. **Smart Embedding Initialization:** แก้ไขบัค Circular Initialization โดยทำการ Capture `decomposition` (ดึง mapping คำไทยใหม่ → list ของ Subword IDs เดิม) ด้วย Tokenizer ตัวเก่าก่อน แล้วนำค่า Vector Embeddings ของ Subwords เดิมที่ผ่านการฝึกสอนมาแล้วมาหาค่าเฉลี่ย (Mean Vector) เป็นค่าเริ่มต้นของคำใหม่ (หากเป็น UNK ทั้งหมดจะ fallback ไปที่ UNK Embedding ที่ฝึกแล้ว) ช่วยให้โมเดลมีจุดเริ่มต้นของน้ำหนักที่ดีกว่าการสุ่มแบบ Random Initialization ส่งผลให้เทรน MLM ไวขึ้นและเสถียรขึ้นอย่างมาก

### 🔹 Block 2: การสร้างชุดข้อมูลฝึกสอนและการถ่วงสมดุลโครงสร้าง (Pre-training Corpus Building & Data Balancing)
* **สิ่งที่ทำ:** รวบรวมและเตรียมชุดข้อมูลข้อความภาษาไทยรวม **2.46 ล้านประโยค** ด้วยสคริปต์ `scripts/build_wiki_corpus.py` และ `scripts/preprocess_lexitron.py`
* **ปัญหาเดิม:** ใน Phase แรกใช้เฉพาะพจนานุกรม Lexitron (130,000 ประโยค) ส่งผลให้โมเดลเกิด **Dictionary Pattern Overfitting** (จดจำแพทเทิร์นหน้าพจนานุกรม เช่น `1. ความหมาย...` หรือ `คำว่า... หมายถึง...` เวลาทำนายผล โมเดลจึงคายเฉพาะตัวเลขและสัญลักษณ์ออกมา) และพบปัญหาลิงก์ Wikipedia Dump เดิมบน HuggingFace เสีย
* **เทคนิคเชิงลึกที่ใช้:**
  1. **Dataset Link Resolution:** เปลี่ยนไปใช้ Dataset `wikimedia/wikipedia` (เวอร์ชัน `20231101.th`) ซึ่งเก็บไฟล์บน Cloud Storage ของ HuggingFace โดยตรง แก้ปัญหา Broken Links 100% ดึงบทความมาได้ 159,719 บทความ หั่นเป็นประโยคธรรมชาติได้ 2.32 ล้านประโยค
  2. **Corpus Balancing & Shuffling:** นำประโยคธรรมชาติจาก Wikipedia มารวมกับประโยคพจนานุกรม Lexitron (สัดส่วน 94% vs 6%) แล้วทำการ Shuffle ข้อมูลทั้งหมด ทำให้โครงสร้างพจนานุกรมถูกเจือจาง บังคับให้โมเดลต้องเลิกจำแพทเทิร์นตัวเลขและหันมาเรียนรู้ความหมายทางภาษาจากบริบทธรรมชาติแทน

### 🔹 Block 3: การฝึกสอน MLM ร่วมกับการปลดล็อกน้ำหนักคำศัพท์ (MLM Pre-training with LoRA & Unfreezing Embeddings)
* **สิ่งที่ทำ:** รันสคริปต์ `scripts/train_lora_mlm.py` เพื่อฝึกสอนโมเดลด้วยเทคนิค Masked Language Modeling (MLM) บนข้อมูล 2.46 ล้านประโยค
* **ปัญหาเดิม (The Frozen Embedding Bug):** ในการเทรนรอบแรก แม้ค่า Perplexity จะลดลงเหลือ 3,960 แต่เวลาทดสอบทำนาย โมเดลก็ยังพ่นตัวเลขออกมาเหมือนเดิม เนื่องจาก PEFT (LoRA) ปกติจะ **แช่แข็ง (Freeze)** เลเยอร์พื้นฐานทั้งหมด ทำให้คำศัพท์ใหม่ 20,000 คำซึ่งเป็นค่าสุ่มไม่สามารถอัปเดตน้ำหนักได้เลยตลอดการเทรน
* **เทคนิคเชิงลึกที่ใช้:**
  1. **LoRA Layer Unfreezing:** เพิ่มคำสั่ง `modules_to_save: ["word_embeddings", "cls.predictions"]` เข้าไปใน `LoraConfig` เพื่อบังคับให้ PEFT ปลดล็อกตารางคำศัพท์ (Embedding Table) และหัวทำนายผล (LM Head) ให้น้ำหนักคำไทย 20,000 คำสามารถเรียนรู้และถูกอัปเดตไปพร้อมๆ กับตัว LoRA Adapter Matrices (`query`, `key`, `value`, `dense`)
  2. **Dynamic Masking & AdamW Optimization:** ใช้ `DataCollatorForLanguageModeling` สุ่มซ่อนคำ 15% (`[MASK]`) และใช้ AdamW Optimizer ร่วมกับ Linear Warmup Scheduler
  3. **ผลลัพธ์ที่ได้:** ค่า Perplexity ดิ่งลงจาก **407,962.30 (Base)** เหลือเพียง **19.27** (MLM Loss = `2.9584`) โมเดลทายคำตอบภาษาไทยตรงตามบริบทอย่างสมบูรณ์

### 🔹 Block 4: การหลอมรวมน้ำหนักโมเดลและการส่งออกเป็น Standalone Base (`scripts/merge_lora.py`)
* **สิ่งที่ทำ:** หลอมรวมน้ำหนักส่วนเสริม LoRA Adapter เข้ากับ Base Model หลัก แล้วบันทึกเป็นโมเดลสมบูรณ์ไว้ที่ `outputs/merged_model_1`
* **เป้าหมาย:** สร้างโมเดลภาษาไทยเดี่ยวๆ ที่พร้อมนำไปใช้งานต่อกับงานอื่นๆ (Downstream Tasks) โดยไม่ต้องยุ่งกับระบบ Adapter อีก
* **เทคนิคเชิงลึกที่ใช้:**
  1. **Weight Merging Equation:** คำนวณหลอมรวมน้ำหนักตามสูตร $W_{\text{merged}} = W_{\text{base}} + \frac{\alpha}{r} (B \times A)$ นำค่าน้ำหนัก LoRA มารวมเข้ากับ Base Weights หลักอย่างถาวรโดยไม่ทำให้ไฟล์ต้นฉบับเสียหาย
  2. **Auto-incrementing Output Directory:** ใช้ระบบตรวจสอบรันหมายเลขโฟลเดอร์อัตโนมัติ (`outputs/merged_model_N`) ป้องกันการเขียนไฟล์ทับ ได้โมเดลขนาด ~450MB ที่พร้อมนำไป Fine-tune งาน NER ต่อไป

---

## 🟢 ส่วนที่ 2: สรุปขั้นตอนและสคริปต์การทำ Clinical NER (NER Fine-tuning Pipeline)

การนำโมเดล SapBERT ภาษาไทยที่ได้จาก Task 1 มา Fine-tune เพื่อสกัดข้อมูลฉลากยา (DRUG, DISEASE, SYMPTOM) แบ่งตามสคริปต์ที่รันจริงได้ 4 Block ดังนี้:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TASK 2 NER PIPELINE BLOCK                         │
├───────────────────┬───────────────────┬───────────────────┬─────────────────┤
│      BLOCK 1      │      BLOCK 2      │      BLOCK 3      │     BLOCK 4     │
│ convert_ner_data  │ train_sapbert_ner │   inference_ner   │train_wangchan_ner│
│ (Data Format BIO) │  (Train & Merge)  │  (Test Extraction)│(Baseline Compare)│
└───────────────────┴───────────────────┴───────────────────┴─────────────────┘
```

### 🔸 Block 1: สคริปต์แปลงข้อมูลฉลากยาเป็น BIO Scheme (`scripts/convert_ner_dataset.py`)
* **หน้าที่ของสคริปต์:** อ่านไฟล์ตารางฉลากยา `data/raw/data_EN.csv` (49 records) แปลงเป็นรูปแบบข้อความพร้อม Tag กำกับแบบ BIO Scheme ไว้ที่ `data/processed/ner_bio_th.jsonl`
* **เทคนิคเชิงลึกที่ใช้:**
  1. **Priority Resolution for Entity Overlap:** แก้ปัญหาคำซ้อนทับกัน (เช่น คำว่า "อักเสบ" เป็น SYMPTOM แต่ "ข้ออักเสบรูมาตอยด์" เป็น DISEASE) โดยกำหนด Priority: `DRUG > DISEASE > SYMPTOM` หากเจอคำซ้อนกันจะยึดตาม Priority ที่สูงกว่าเสมอ
  2. **Subword Offset Mapping Alignment:** ใช้ `return_offsets_mapping=True` เพื่อแมปดัชนีตัวอักษร (Character Spans) เข้ากับ Subword Tokens ของ Tokenizer อย่างแม่นยำ โดย Subword แรกของ Entity กำหนดเป็น `B-`, Subword ถัดไปเป็น `I-` และ Special Tokens กำหนดเป็น `-100` (เพื่อข้ามการคำนวณ Loss)
  3. **Data Deduplication:** ทำการกรองประโยคซ้ำโดยใช้ข้อความบรรยายยาเป็น Key ทำให้ข้อมูลไม่ซ้ำกันเหลือ 87 ตัวอย่าง (แบ่ง Train 80% / Val 20%)

### 🔸 Block 2: สคริปต์ Fine-tune SapBERT NER และหลอมรวมอัตโนมัติ (`scripts/train_sapbert_ner.py`)
* **หน้าที่ของสคริปต์:** โหลด SapBERT Merged Model ล่าสุดจาก Task 1 (`outputs/merged_model_1`) มาทำ Token Classification ด้วย LoRA และเซฟเป็นโมเดลใหม่
* **เทคนิคเชิงลึกที่ใช้:**
  1. **Auto-Detect Latest Base Model:** ระบบจะค้นหาและดึงโมเดล `outputs/merged_model_N` ล่าสุดมาเป็น Base Model โดยอัตโนมัติ
  2. **LoRA Token Classification Architecture:** สวม LoRA Adapter (`TaskType.TOKEN_CLS`, $r=16, \alpha=32$) บน Attention & Dense Layers (`query`, `key`, `value`, `dense`) ใช้ `AutoModelForTokenClassification` เพื่อรองรับ BERT Architecture
  3. **Entity-level Metric Evaluation:** ประเมินผลระหว่างเทรนด้วย `seqeval` (วัด Precision, Recall, F1 ในระดับ Entity ทั้งคำ ไม่ใช่วัดแค่ระดับ Subword)
  4. **Sequential Auto-increment Output System:** เมื่อเทรนเสร็จ สคริปต์จะทำการ Merge น้ำหนัก LoRA เข้ากับโมเดลทันที และบันทึกเป็นชื่อโฟลเดอร์ใหม่รันเลขต่อท้าย `_ner_N` โดยอัตโนมัติ — รอบแรกได้ `outputs/merged_model_1_ner_1` (F1 = 0.7714) ส่วนรอบสอง Auto-detect ไปเลือก NER model รอบแรกมาเป็น base ซ้ำอีกชั้น จึงบันทึกเป็น `merged_model_1_ner_1_ner_1` ปัจจุบันถูก rename เป็น **`outputs/merged_model_1_ner_base`**
  5. **ผลการทดลอง:** เมื่อเทรน 100 Epochs ได้ค่า **Best F1 Score สูงถึง 0.8000** (Loss = 0.4676 → `outputs/merged_model_1_ner_base`)

### 🔸 Block 3: สคริปต์ทดสอบและสกัด Entity จากข้อความ (`scripts/inference_ner.py`)
* **หน้าที่ของสคริปต์:** ทดสอบป้อนประโยคภาษาไทยเพื่อสกัด Entity (DRUG, DISEASE, SYMPTOM) ออกมาจากโมเดล
* **เทคนิคเชิงลึกที่ใช้:**
  1. **Auto-Detect & Standalone Direct Loading:** ค้นหาโฟลเดอร์โมเดล `*_ner_*` ล่าสุดอัตโนมัติ และทำการโหลดโมเดลฉบับ Merged เข้าคลาส `AutoModelForTokenClassification` โดยตรงโดยไม่ต้องใช้ไลบรารี `peft` (⚠️ ด้วยลำดับชื่อปัจจุบัน auto-detect จะได้ `merged_model_1_ner_1` — ถ้าต้องการโมเดลที่ดีที่สุด F1 = 0.8000 ให้ระบุ `--model outputs/merged_model_1_ner_base`)
  2. **Subword to Entity Reconstruction:** แปลงค่าน้ำหนักผลลัพธ์ (Logits) กลับเป็น Tag B-/I- แล้วใช้ `offset_mapping` รวม Subwords ย่อยกลับมาเป็นคำภาษาไทยเต็มประโยค พร้อมระบุตำแหน่งดัชนีเริ่มต้น-สิ้นสุด (`pos start–end`)
  3. **Terminal Color-coded Visualization:** แสดงผลการจำแนก Entity ผ่านหน้าจอ Terminal ด้วยสีแยกชัดเจน (สีฟ้า = DRUG, สีแดง = DISEASE, สีเหลือง = SYMPTOM)

### 🔸 Block 4: สคริปต์เทรน Baseline เปรียบเทียบด้วย WangchanBERTa (`scripts/train_wangchan_ner.py`)
* **หน้าที่ของสคริปต์:** Fine-tune โมเดล WangchanBERTa (`airesearch/wangchanberta-base-att-spm-uncased`) สำหรับงาน NER เพื่อใช้เป็นเกณฑ์เปรียบเทียบ (Baseline)
* **เทคนิคเชิงลึกที่ใช้:**
  1. **RoBERTa/Camembert Adaptation:** ใช้ `CamembertForTokenClassification` ร่วมกับ LoRA
  2. **Baseline Performance:** ได้ F1 Score อยู่ที่ **0.8269** (ที่ Epoch 28) ซึ่งนำมาใช้เปรียบเทียบกับ SapBERT Thai Model (F1 = 0.8000) เพื่อดูประสิทธิภาพในการเรียนรู้ภาษาไทยของโมเดลการแพทย์ข้ามภาษา

---

## 📈 สรุปเปรียบเทียบผลลัพธ์ภาพรวม (Overall Results Summary)

| ขั้นตอน / โมเดล | ภารกิจหลัก (Task) | MLM Loss | Perplexity | NER F1 Score | สถานะผลลัพธ์ |
|---|---|:---:|:---:|:---:|---|
| **Base SapBERT** | ขยาย vocab ไทย 20k | `12.9189` | `407,962.30` | - | ไม่รู้จักภาษาไทย (สุ่ม init) |
| **SapBERT MLM (ก่อนแก้บัค)** | MLM 2.46M ประโยค | `8.2841` | `3,960.49` | - | คำไทยถูก Freeze ติดค่าสุ่ม ทายได้แค่ตัวเลข |
| **SapBERT MLM (หลังแก้บัค)** ⭐ | MLM 2.46M ประโยค | **`2.9584`** | **`19.27`** | - | ✅ **โมเดลเข้าใจภาษาไทยสมบูรณ์** (`merged_model_1`) |
| **WangchanBERTa Baseline** | Clinical NER | - | - | **`0.8269`** | ✅ Baseline ภาษาไทยเดิม (`wangchan_ner`) |
| **SapBERT Thai NER (100 Ep)** ⭐ | Clinical NER | - | - | **`0.8000`** | ✅ **SapBERT ข้ามภาษาทำ NER ได้ใกล้เคียง Baseline** (`merged_model_1_ner_base`; รอบแรก `merged_model_1_ner_1` = 0.7714) |
