# Medical Corpus Chunking Pipeline

เอกสารนี้อธิบายกระบวนการเตรียมข้อมูล Thai Medical Corpus
สำหรับการ pre-train ด้วย Masked Language Modeling (MLM) + LoRA

---

## 1. ภาพรวม (Overview)

```
train_original.jsonl          →   medical_chunks.jsonl   →   splits
(9,081 บทความ / 211 MB)           (82,204 chunks / 204 MB)
                                          ↓
                                  medical_train.jsonl   (73,983 chunks / 172 MB)
                                  medical_val.jsonl     ( 4,110 chunks /   9.6 MB)
                                  medical_test.jsonl    ( 4,111 chunks /   9.5 MB)
```

---

## 2. ข้อมูล Input

| รายการ | ค่า |
|--------|-----|
| ไฟล์ต้นทาง | `data/raw/train_original.jsonl` |
| จำนวนบทความ | 9,081 บทความ |
| ขนาดไฟล์ | 211 MB |
| Format | JSONL — แต่ละบรรทัดเป็น `{"aid": ..., "markdown": ..., "metadata": {...}, "source_id": ...}` |
| แหล่งที่มา | Thai medical websites (Medscape, Haamor, WebMD TH, Bumrungrad, ฯลฯ) |

---

## 3. ขั้นตอนการทำ Chunking

### Step 1 — Data Cleaning (ทำความสะอาด Markdown)

ก่อน chunk ต้องตัด noise ออกก่อน โดยฟังก์ชัน `clean_markdown()` ใน
[`src/data/medical_chunker.py`](../src/data/medical_chunker.py):

| สิ่งที่ตัดออก | ตัวอย่าง |
|---|---|
| Doctor bylines | `นายแพทย์สมชาย...`, `**นพ.**...` |
| ชื่อโรงพยาบาล/คลินิก | บรรทัดที่มี `โรงพยาบาล`, `Bumrungrad`, `BDMS` |
| Markdown images | `![alt](url)` |
| URL เดี่ยว | `https://www.example.com` |
| ข้อมูลติดต่อ | บรรทัดที่มี `ติดต่อ`, `โทร`, `Line ID`, `Facebook` |
| Call-to-action | `อ่านเพิ่มเติม`, `Read more`, `บทความที่เกี่ยวข้อง` |
| Markdown horizontal rule | `---` |
| Markdown hyperlinks | `[ข้อความ](url)` → เหลือแค่ข้อความ |
| Heading markers | `## หัวข้อ` → `หัวข้อ` |

หลังจากทำความสะอาด บทความจะถูกแบ่งด้วย `\n\n` เป็น **paragraphs**
โดย paragraph ที่สั้นกว่า 10 ตัวอักษรจะถูกกรองออก

---

### Step 2 — Token-aware Chunking

#### ทำไมต้องใช้ tokenizer จริง?

BERT มีข้อจำกัด **512 tokens** ต่อ input เท่านั้น
การนับโดยใช้จำนวนตัวอักษรหรือคำไม่แม่นยำ เพราะ:

- ภาษาไทยใช้ WordPiece subword tokenization
- คำหนึ่งคำอาจถูกตัดเป็นหลาย subwords (เช่น `รักษา` → `รักษา`, `##โรค`)
- จำนวน tokens จริงขึ้นอยู่กับ vocabulary ของโมเดลที่ใช้

ดังนั้นจึงใช้ tokenizer จาก `data/interim/extended_base` (vocab 40,075 tokens)
ในการคำนวณความยาวทุกครั้ง

#### ค่า Parameters

| Parameter | ค่า | เหตุผล |
|-----------|-----|--------|
| `max_content_tokens` | **510** | BERT max = 512; เผื่อ [CLS] + [SEP] = 2 tokens |
| `stride` | **128** | Overlap สำหรับ sliding window ป้องกัน context ขาดหาย |
| `min_chunk_tokens` | **20** | กรอง footer noise ที่อาจหลุดผ่าน cleaning |

#### Algorithm

```
สำหรับแต่ละ Article:
  1. clean_markdown()  → ตัด footer/contact/images
  2. split_paragraphs() → แบ่งด้วย \n\n

  สำหรับแต่ละ paragraph:
    ├── paragraph_tokens > 510?
    │     → Sliding Window (ดูด้านล่าง)
    └── paragraph_tokens ≤ 510?
          ├── buffer + paragraph_tokens > 510?
          │     → flush buffer เป็น chunk ใหม่
          │     → เริ่ม buffer ใหม่ด้วย paragraph นี้
          └── buffer + paragraph_tokens ≤ 510?
                → เพิ่ม paragraph เข้า buffer

  flush buffer สุดท้าย
```

#### Sliding Window (สำหรับ paragraph ที่ยาวเกิน 510 tokens)

```
Paragraph ยาว N tokens → แบ่งเป็นหน้าต่างขนาด 510 tokens
Step = 510 - 128 = 382 tokens

ตัวอย่าง (N = 1,200 tokens):
  Window 1: tokens[0   : 510]
  Window 2: tokens[382 : 892]
  Window 3: tokens[764 : 1,274]  (clipped at N)
```

การ overlap 128 tokens ช่วยให้โมเดลได้เห็น context ที่ต่อเนื่องกัน
ไม่ถูกตัดขาดตรงกลางประโยค

---

## 4. ผลลัพธ์และสถิติ

### ขนาดข้อมูล

| | บทความต้นทาง | Chunks ทั้งหมด |
|--|--|--|
| จำนวน records | 9,081 | **82,204** |
| ขนาดไฟล์ | 211 MB | 204 MB (interim) |
| เฉลี่ย chunks/article | — | **9.1 chunks** |

### การกระจายตัวของ Token Count

| ช่วง tokens | จำนวน chunks | สัดส่วน |
|------------|-------------|--------|
| 1 – 100   |  1,497 |  1.8% |
| 101 – 200 |  2,763 |  3.4% |
| 201 – 300 |  3,505 |  4.3% |
| 301 – 400 |  9,528 | 11.6% |
| **401 – 510** | **64,911** | **79.0%** |

> 79% ของ chunks มีความยาวใกล้เคียงกับ limit (401–510 tokens)
> หมายความว่าโมเดลจะได้รับ context ที่หนาแน่นและสมบูรณ์ในทุก chunk

**สรุปสถิติ:**

| รายการ | ค่า |
|--------|-----|
| Min tokens | 20 |
| Max tokens | 514* |
| Mean tokens | 422.1 |
| Median tokens | 452 |

> \*บาง chunk เกิน 510 เล็กน้อยเพราะ tokenizer decode อาจนับต่างกัน (±4 tokens)

### Chunks แยกตาม Source

| Source | Chunks | สัดส่วน |
|--------|--------|--------|
| medscape_drug_disease | 68,489 | 83.3% |
| haamor_article | 6,098 | 7.4% |
| webmd_disease | 3,543 | 4.3% |
| synpad_article | 1,755 | 2.1% |
| bumrungrad_article | 1,754 | 2.1% |
| bangkok_mental_hospital_article | 375 | 0.5% |
| hdmall_faq | 98 | 0.1% |
| canesten_article | 92 | 0.1% |

### Train / Val / Test Splits

Split ratio: **90% / 5% / 5%** (random shuffle seed=42)

| Split | Chunks | ขนาดไฟล์ | Path |
|-------|--------|---------|------|
| Train | 73,983 | 172 MB | `data/processed/medical_train.jsonl` |
| Val   |  4,110 | 9.6 MB | `data/processed/medical_val.jsonl` |
| Test  |  4,111 | 9.5 MB | `data/processed/medical_test.jsonl` |

---

## 5. ทำไมถึงได้ข้อมูลขนาดนี้?

### จาก 9,081 บทความ → 82,204 chunks (เพิ่มขึ้น ~9×)

สาเหตุที่จำนวน records เพิ่มขึ้นมากคือ:

**1. บทความทางการแพทย์มีเนื้อหายาว**
บทความ Medscape และ Haamor มักมีเนื้อหา 3,000–10,000+ tokens
เมื่อ chunk ด้วยหน้าต่างขนาด 510 tokens จะได้หลาย chunks ต่อบทความ
(เฉลี่ย **9.1 chunks/article**)

**2. Sliding Window สร้าง Overlap**
paragraph ที่ยาวเกิน 510 tokens จะถูกแบ่งด้วย sliding window stride=128
ทำให้ 1 paragraph ยาวมากอาจกลายเป็น 5–10 chunks

**3. Medscape มีบทความจำนวนมากและยาว**
Medscape drug/disease articles คิดเป็น **83.3%** ของ chunks ทั้งหมด
เพราะเป็น database ยาและโรคที่มีเนื้อหาครอบคลุมมาก

### ทำไม train.jsonl ถึงเล็กกว่า interim/medical_chunks.jsonl?

`interim/medical_chunks.jsonl` เก็บ **metadata ครบ** (article_id, chunk_id, source_id, title, token_count, text)
ส่วน `processed/medical_train.jsonl` เก็บแค่ `{"text": "..."}` สำหรับ Trainer เท่านั้น
จึงมีขนาดเล็กกว่า (~172 MB vs 204 MB)

---

## 6. ไฟล์ที่เกี่ยวข้อง

| ไฟล์ | หน้าที่ |
|------|--------|
| [`src/data/medical_chunker.py`](../src/data/medical_chunker.py) | Library หลัก: `clean_markdown()`, `chunk_article()`, `process_corpus()` |
| [`scripts/build_medical_corpus.py`](../scripts/build_medical_corpus.py) | Runner script: อ่าน JSONL → chunk → บันทึก splits |
| `data/raw/train_original.jsonl` | Input: บทความทางการแพทย์ดิบ |
| `data/interim/medical_chunks.jsonl` | Output ครบ: chunk + metadata |
| `data/processed/medical_train.jsonl` | Output สำหรับ Trainer: `{"text": ...}` |

---

## 7. การรัน Pipeline

```bash
# Dry-run (ดูสถิติโดยไม่เขียนไฟล์)
python scripts/build_medical_corpus.py --dry-run

# รัน pipeline เต็ม (default: max_tokens=510, stride=128)
python scripts/build_medical_corpus.py

# ปรับค่า parameters
python scripts/build_medical_corpus.py --max-tokens 510 --stride 128

# เทรนกับ medical data
python scripts/train_lora_mlm.py \
    --train-data data/processed/medical_train.jsonl \
    --val-data   data/processed/medical_val.jsonl
```

---

## 8. ข้อสังเกต

> **NOTE**: ไฟล์ `data/interim/medical_chunks.jsonl` มี field `chunk_id` ที่ reset เป็น 0
> สำหรับทุกบทความ — ถ้าต้องการ global unique ID ให้ใช้ `(article_id, chunk_id)` คู่กัน

> **TIP**: สามารถ filter ข้อมูลตาม `source_id` ได้จาก `medical_chunks.jsonl`
> เช่น เทรนเฉพาะ Medscape หรือ Haamor สำหรับงาน domain-specific

> **WARNING**: อย่าใช้ `data/processed/medical_train.jsonl` เพื่อ analysis
> เพราะ metadata ถูกตัดออกแล้ว ให้ใช้ `data/interim/medical_chunks.jsonl` แทน


Option A: ใช้แต่ละ QA pair เป็น 1 record ตรงๆ  ช่วยเตรียมข้อมูลหน่อยครับ