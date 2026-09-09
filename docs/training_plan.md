# 🎓 แผนการอบรม 1 ชั่วโมง: Cross-Lingual Domain Adaptation & Clinical NER
## "จาก SapBERT โมเดลการแพทย์ภาษาอังกฤษ สู่งานวิเคราะห์ฉลากยาภาษาไทย"

---

## 📌 ข้อมูลการอบรม
* **ระยะเวลา:** 60 นาที (1 ชั่วโมง)
* **รูปแบบ:** บรรยายเชิงปฏิบัติการ (Lecture + Code Walkthrough + Live Demo)
* **กลุ่มเป้าหมาย:** นักพัฒนา AI/ML, Data Scientist, หรือผู้สนใจการปรับแต่งโมเดลภาษา (LLM / Domain Adaptation)
* **วัตถุประสงค์:** 
  1. เข้าใจกระบวนการสอนโมเดลภาษาการแพทย์ภาษาอังกฤษ (SapBERT) ให้เข้าใจภาษาไทย
  2. เรียนรู้ปัญหาเทคนิคสำคัญที่มักพบในการใช้ LoRA กับการเพิ่มคำศัพท์ใหม่ (เช่น Frozen Embedding Bug)
  3. เข้าใจขั้นตอนการสร้างระบบวิเคราะห์ชื่อเฉพาะทางคลินิกและฉลากยา (Clinical NER) ด้วย PyTorch & HuggingFace

---

## ⏱️ กำหนดการอบรม (60 นาที)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AGENDA BREAKDOWN (60 MINS)                       │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────────────┤
│ 00:00-00:10  │ 00:10-00:25  │ 00:25-00:40  │ 00:40-00:50  │ 00:50-01:00     │
│ Intro & Problem│ Task 1 MLM   │ Task 2 NER   │ Live Demo    │ Key Takeaways   │
│ Statement    │ Adaptation   │ Fine-tuning  │ & Deployment │ & Q&A           │
└──────────────┴──────────────┴──────────────┴──────────────┴─────────────────┘
```

---

### 🕒 ช่วงที่ 1: Introduction & Problem Statement (10 นาที | 00:00 - 00:10)
* **หัวข้อบรรยาย:**
  * **โจทย์และความท้าทาย:** ทำไมต้องใช้ SapBERT? (โมเดลการแพทย์ภาษาอังกฤษมีความรู้ชีวการแพทย์ลึกซึ้ง แต่ไม่เข้าใจภาษาไทย)
  * **ปัญหาคลาสสิกของ Cross-Lingual Adaptation:** เมื่อป้อนข้อความไทยเข้าโมเดลอังกฤษ จะเกิดอะไรขึ้น? (โมเดลมองเห็นเป็น `[UNK]` หรือตัดคำย่อยยิบจนสูญเสียความหมาย)
* **กิจกรรม / สไลด์ประกอบ:**
  * แสดงตัวอย่างข้อความฉลากยาไทยที่ถูก Tokenize ด้วย SapBERT ดั้งเดิม vs โมเดลที่ขยายคำศัพท์แล้ว

---

### 🕒 ช่วงที่ 2: Task 1 — SapBERT Thai Adaptation Pipeline (15 นาที | 00:10 - 00:25)
* **หัวข้อบรรยาย:** 4 ขั้นตอนการสอนภาษาไทยให้ SapBERT
  1. **Vocabulary Expansion (BPE & Smart Init):** การขยายคำศัพท์ไทย 20,000 คำ และเทคนิค **Smart Embedding Init** (ดึง mean vector จาก subwords เดิมเพื่อป้องกันการสุ่มน้ำหนัก)
  2. **Corpus Construction & Balancing:** การสร้างชุดข้อมูล 2.46 ล้านประโยค (Wikipedia 94% + Lexitron 6%) เพื่อแก้ปัญหา **Dictionary Pattern Overfitting**
  3. **MLM Pre-training with LoRA (บทเรียนสำคัญ):** เจาะลึกบัค **"The Frozen Embedding Trap"** — ทำไมเทรน LoRA แล้วโมเดลทายแต่ตัวเลข? (เพราะตารางคำศัพท์ใหม่ถูกแช่แข็ง ต้องปลดล็อกด้วย `modules_to_save: ["word_embeddings", "cls.predictions"]`)
  4. **Perplexity Evaluation:** การวัดผลลัพธ์ความงุนงงของโมเดลที่ดิ่งลงจาก **407,962 เหลือ 19.27**
* **กิจกรรม / Code Walkthrough:**
  * เปิดโค้ดสคริปต์ [extend_tokenizer.py](file:///home/aom/thai_bert_lora/scripts/extend_tokenizer.py) และ [train_lora_mlm.py](file:///home/aom/thai_bert_lora/scripts/train_lora_mlm.py) อธิบายจุดแก้บัคสำคัญ

---

### 🕒 ช่วงที่ 3: Task 2 — Clinical NER Fine-tuning on Drug Labels (15 นาที | 00:25 - 00:40)
* **หัวข้อบรรยาย:** การ Fine-tune โมเดลสกัดชื่อยา โรค และอาการ (DRUG / DISEASE / SYMPTOM)
  1. **Data Preprocessing & BIO Scheme:** การแปลง CSV ฉลากยา 49 แถวเป็น BIO Format และการจัดการ Entity Overlap ด้วย Priority Resolution (`DRUG > DISEASE > SYMPTOM`)
  2. **Subword Offset Alignment:** การใช้ `offset_mapping` จับคู่ดัชนีตัวอักษรเข้ากับ Subword Tokens
  3. **SapBERT NER Training:** การสวม LoRA Token Classification (`train_sapbert_ner.py`) และระบบบันทึกโมเดลอัตโนมัติรันเลขต่อท้าย `_ner_N`
  4. **Performance Comparison:** เปรียบเทียบผล F1 Score ระหว่าง SapBERT Thai NER (**0.8000**) vs WangchanBERTa Baseline (**0.8269**)
* **กิจกรรม / Code Walkthrough:**
  * อธิบายโค้ด [train_sapbert_ner.py](file:///home/aom/thai_bert_lora/scripts/train_sapbert_ner.py) และแสดงกราฟ/ตารางเปรียบเทียบผลลัพธ์ F1 Score

---

### 🕒 ช่วงที่ 4: Live Inference Demonstration & Model Deployment (10 นาที | 00:40 - 00:50)
* **หัวข้อบรรยาย & สาธิตสด:**
  * **Interactive Inference:** รันสคริปต์ [inference_ner.py](file:///home/aom/thai_bert_lora/scripts/inference_ner.py) ทดสอบป้อนประโยคฉลากยาไทยสดๆ บน Terminal (แสดงผลสีเน้นย้ำ Blue/Red/Yellow)
  * **Model Merging & Standalone Export:** แนวคิดการหลอมรวมน้ำหนัก LoRA เข้ากับ Base Model (`merge_and_unload()`) เพื่อแปลงเป็นโมเดล Standalone (~450MB) ที่นำไป Deploy ขึ้นระบบจริงได้ทันทีโดยไม่ต้องพึ่งพาไลบรารี `peft`
* **กิจกรรม / Live Demo:**
  * รันคำสั่ง `python scripts/inference_ner.py --model outputs/merged_model_1_ner_base --text "อะบิราเทอโรน ใช้รักษามะเร็งต่อมลูกหมาก โดยมีอาการแพร่กระจาย"` แสดงการทำงานจริง

---

### 🕒 ช่วงที่ 5: Key Takeaways, Lessons Learned & Q&A (10 นาที | 00:50 - 01:00)
* **หัวข้อบรรยาย:**
  * **สรุป 3 ข้อผิดพลาดสำคัญที่ต้องระวัง (Gotchas):**
    1. Circular Initialization ในการขยาย Vocabulary
    2. Frozen Embedding Trap เมื่อใช้ PEFT/LoRA กับศัพท์ใหม่
    3. Dictionary Format Overfitting จากโครงสร้างชุดข้อมูล
  * **แนวทางการพัฒนาต่อในอนาคต (Next Steps):** การเพิ่มปริมาณข้อมูลฉลากยาภาษาไทยให้มีความหลากหลายยิ่งขึ้น
* **กิจกรรม:**
  * เปิดโอกาสให้ผู้เข้าร่วมอบรมถาม-ตอบ (Q&A)

---

## 🛠️ อุปกรณ์และไฟล์ที่ต้องเตรียมสำหรับผู้สอน (Trainer Checklists)

1. **สไลด์ประกอบการสอน:**
   * สรุปตามตารางในไฟล์ [docs/result.md](file:///home/aom/thai_bert_lora/docs/result.md)
2. **เตรียมสภาพแวดล้อมและโมเดลสำหรับ Demo:**
   * ตรวจสอบว่ามีโมเดล `outputs/merged_model_1_ner_base` (F1 = 0.8000) พร้อมรัน — ⚠️ ถ้าไม่ระบุ `--model` auto-detect จะเลือก `merged_model_1_ner_1` (F1 = 0.7714) แทน
   * คำสั่งรัน Demo:
     ```bash
     source .venv/bin/activate
     python scripts/inference_ner.py --model outputs/merged_model_1_ner_base --text "อะดาลิมูแมบ ใช้รักษาข้ออักเสบรูมาตอยด์ มีอาการอักเสบ"
     ```
