# CLAUDE.md — Thai BERT LoRA Project

เอกสารนี้สำหรับ Claude Code ให้เข้าใจบริบทโปรเจกต์ ปัญหาที่ค้นพบ และสิ่งที่ต้องทำก่อนดำเนินการต่อ
*(อัปเดตล่าสุด: 2026-09-09 — Task 1 retrain สำเร็จแล้ว และ **Task 2 เปลี่ยนมาใช้โมเดล SapBERT Thai เป็นโมเดลหลัก** แทน WangchanBERTa)*

---

## 1. บริบทโปรเจกต์ (Project Context)

โปรเจกต์นี้มี **2 งานหลัก (Tasks)** ที่ทำงานขนานกัน:

### 🔵 Task 1 — สอน SapBERT (โมเดลอังกฤษ) ให้เข้าใจภาษาไทย
- **Pipeline:** SapBERT → ขยาย vocab ไทย 20,000 คำ (`extend_tokenizer.py`) → MLM pretraining ด้วย LoRA บนข้อมูล Lexitron + Thai Wikipedia (2.46 ล้านประโยค)
- **เป้าหมายสุดท้าย:** ได้โมเดลที่เข้าใจภาษาไทย นำไปทำ Clinical NER ต่อ
- **สถานะปัจจุบัน:** ✅ **เสร็จสมบูรณ์** — แก้บัคทั้งหมด, retrain สำเร็จ (Perplexity 407,962 → 19.27) และ merge เป็น `outputs/merged_model_1` ให้ Task 2 ใช้ต่อแล้ว (ดูข้อ 2.4)
- **เอกสารอ้างอิง:** `docs/task_1.md`

### 🟢 Task 2 — Fine-tune **SapBERT Thai** ทำ Clinical NER (โมเดลหลักปัจจุบัน)
- **ประวัติ:** ช่วงที่ Task 1 ยังติด Frozen Embedding Bug เคย**พัก Task 1** แล้วใช้ WangchanBERTa (`airesearch/wangchanberta-base-att-spm-uncased`) fine-tune NER เป็นทางออกชั่วคราว — ผลรอบนั้น (F1 = 0.8269) ถูกรักษาไว้เป็น **baseline เปรียบเทียบ**
- **ปัจจุบัน:** ✅ **Task 2 เปลี่ยนมาใช้โมเดล SapBERT แล้ว** — เมื่อ Task 1 แก้บัคและ merge เสร็จ (`outputs/merged_model_1`) ก็นำโมเดลนั้นมา fine-tune NER บน dataset ฉลากยา (`data/raw/data_EN.csv`, 49 rows → 44 ประโยค TH) ด้วย `scripts/train_sapbert_ner.py` (config: `configs/sapbert_ner_config.yaml`)
- **Entity types:** DRUG / DISEASE / SYMPTOM (BIO scheme, 7 labels)
- **สถานะปัจจุบัน:** ✅ เทรนเสร็จ — **F1 = 0.8000** (100 epochs → `outputs/merged_model_1_ner_base`; รอบแรกที่เทรนจาก `merged_model_1` ตรงๆ ได้ F1 = 0.7714 → `outputs/merged_model_1_ner_1`)
- **Baseline:** WangchanBERTa F1 = 0.8269 → SapBERT ข้ามภาษาทำได้ใกล้เคียงมาก (ต่าง ~0.027)
- **เอกสารอ้างอิง:** `docs/task_2.md`, `docs/result.md`, `docs/project_overview.md`

### Timeline ความสัมพันธ์ระหว่าง Tasks

```
Task 1 (SapBERT + MLM)              Task 2 (Clinical NER)
─────────────────────               ─────────────────────────────
Phase 1: Vocab Expansion ✅          (ช่วง Task 1 ติดปัญหา)
Phase 2: MLM Lexitron ❌             WangchanBERTa NER ชั่วคราว
  → Dictionary Overfitting            → F1 = 0.8269 ✅ (baseline)
Phase 3: +Wikipedia ✅                       │
Phase 4: Frozen Embedding ❌                 │
  → แก้ด้วย modules_to_save ✅               │
  → Retrain สำเร็จ PPL 19.27 ✅              │
  → Merge → merged_model_1 ✅                │
        └──────────────────────────────────►│
                                    SapBERT Thai NER (โมเดลหลัก)
                                    → F1 = 0.8000 ✅ (ner_base)
```

**สำคัญ:** สคริปต์ฝั่ง NER/MLM ของ Task 2 (`train_sapbert_ner.py`, `train_wangchan_ner.py`, `train_wangchan_mlm.py`) เขียน LoRA config เองภายใน script **ไม่ได้ใช้** `src/models/lora.py` — ดังนั้น bug ใน `src/models/lora.py` (ดูข้อ 2.1) จึงไม่กระทบ แต่จะกระทบทันทีถ้า refactor ไปใช้ shared code (หมายเหตุ: `train_sapbert_ner.py` ก็มี fallback metric pattern แบบเดียวกับข้อ 3.3 และ auto-split ไม่ stratified แบบข้อ 3.7 ด้วย)

---

## 🗺️ Flow ระบบ Task 1 (แบบย่อ)

```
   📖 ข้อมูลดิบ                    🔤 ขยายคำศัพท์              🧠 เทรน                    ✅ ตรวจผล
  ─────────────                  ─────────────────           ─────────                ─────────────

  telex.csv       ┐                                       
  (Lexitron)      ├──► [1] ──► corpus ──► [3] ──► extended ──► [5] ──► adapter ──► [7] ──► merged
                  │    เตรียม     2.46M      ขยาย vocab    base     เทรน MLM    + LoRA    ทดสอบ     model
  Wikipedia (TH)  ┘    ข้อมูล    ประโยค                        โมเดล                       + merge
```

### ทีละขั้น

```
[1] preprocess_lexitron.py     CSV พจนานุกรม → JSONL สะอาด
        ⬇
[2] build_wiki_corpus.py       ดึง Wikipedia ไทย → ประโยคธรรมชาติ
    build_mlm_dataset.py       รวมสองแหล่ง → train / val / test
        ⬇
[3] extend_tokenizer.py        สอน tokenizer ให้รู้จักคำไทย 20,000 คำ
        ⬇                      ⚠️ มี bug — init คำใหม่ไม่ทำงาน (ดูข้อ 2.2)
[4] analyze_tokenizer.py       เช็คว่า UNK rate ต่ำพอหรือยัง
        ⬇
[5] train_lora_mlm.py          เทรน MLM + LoRA (base เดิมไม่ถูกแก้)
        ⬇
[6] evaluate_mlm.py            วัด perplexity
    test_inference.py          ทดสอบเติมคำ [MASK] → ต้องทายคำไทยได้
        ⬇
[7] merge_lora.py              รวม adapter → โมเดลพร้อมใช้
```

### หลักคิดสำคัญ 3 ข้อ

1. **Base model ไม่ถูกแตะ** — ทุกอย่าง save แยกใน `outputs/` ถอยกลับได้ทุกเมื่อ
2. **LoRA เทรนแค่น้ำหนักเสริม** — ประหยัด แต่มีเงื่อนไข: ต้องปลดล็อก `word_embeddings` ด้วย (`modules_to_save`) ไม่งั้นคำไทยใหม่เรียนรู้ไม่ได้ ← นี่คือ bug ใหญ่ที่เจอ (ดูข้อ 2.4)
3. **จบที่ merged model** — เป้าหมายคือเอาไปทำ NER ต่อ (เทียบผลกับ WangchanBERTa ของ Task 2)

### สถานะตอนนี้

| ขั้น | สถานะ |
|------|-------|
| [1]–[2] เตรียมข้อมูล | ✅ เสร็จแล้ว (corpus 2.46M ประโยคพร้อมใน `data/processed/`) |
| [3] ขยาย vocab | ✅ แก้ Smart Init Bug แล้วรันใหม่สำเร็จ (extended_base ใหม่ init จาก mean ของ subwords เดิม) |
| [4]–[7] เทรน/ประเมิน/merge | ✅ เสร็จทั้งหมด — PPL 19.27, inference ผ่าน, merged → `outputs/merged_model_1` → ใช้ทำ NER ต่อใน Task 2 แล้ว (F1 = 0.8000) |

---

## 2. 🔴 Critical Bugs (ต้องแก้ก่อน retrain Task 1)

### 2.1 TaskType Hardcode — `src/models/lora.py:67`

**ปัญหา:** ฟังก์ชัน `apply_lora()` hardcode `task_type=TaskType.FEATURE_EXTRACTION` เสมอ โดย**ไม่อ่านค่า `task_type` จาก config** แม้ `configs/lora_config.yaml` จะมี field `task_type` และ `configs/ner_config.yaml` กำหนด `task_type: "TOKEN_CLS"` ก็ตาม

**สาเหตุที่เกิด:** เดิมโค้ดนี้เขียนขึ้นเพื่อรองรับเฉพาะ Task 1 (MLM = FEATURE_EXTRACTION) ช่วงต่อมาเมื่อเพิ่ม Task 2 (NER = TOKEN_CLS) ผู้เขียนเลือก copy LoRA logic ไปไว้ใน script ของ Task 2 แทนที่จะแก้ shared library ให้รองรับ config → field `task_type` ใน YAML จึงกลายเป็น "config ที่ตายแล้ว" (dead config)

**ผลกระทบ:**
- ปลอดภัยแบบ "เงียบๆ" (silent) — ไม่ error แต่ทำงานผิด
- ถ้าอนาคตมีโค้ดใดเรียก `apply_lora()` ด้วย NER config จะได้ model ที่ PEFT wrap ด้วย task type ผิด → head ของ TokenClassification อาจไม่ถูกจัดการถูกต้อง

**วิธีแก้:**
```python
task_type_str = lora_cfg.get("task_type", "FEATURE_EXTRACTION")
config = LoraConfig(
    task_type=getattr(TaskType, task_type_str),
    ...
)
```

---

### 2.2 Smart Init Bug (Circular Initialization) — `scripts/extend_tokenizer.py:135-156`

**ปัญหา:** ฟังก์ชัน `smart_init_embeddings()` มีจุดประสงค์ init embedding ของคำไทยใหม่ = ค่าเฉลี่ย (mean) ของ subword embeddings เดิม เช่น "โรงพยาบาล" ควร init จาก mean ของ "โรง", "พยา", "บาล" ที่ tokenizer เดิมรู้จัก

**แต่สิ่งที่เกิดจริง:** ลำดับการทำงานใน `main()` คือ:
1. เขียน `vocab.txt` ใหม่ที่มีคำไทยทั้งหมด (บรรทัด 253-257)
2. **Reload tokenizer จาก vocab ใหม่** (บรรทัด 278) ← tokenizer ตัวนี้รู้จักคำไทยแล้ว
3. `resize_token_embeddings()` (บรรทัด 284)
4. เรียก `smart_init_embeddings(model, tokenizer, new_tokens)` (บรรทัด 288) ← ส่ง **tokenizer ตัวใหม่** เข้าไป

ภายในฟังก์ชัน บรรทัด 148:
```python
sub_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
```
เมื่อ tokenize คำ "โรงพยาบาล" ด้วย tokenizer ที่**รู้จักคำนี้อยู่แล้ว** → ได้ token เดียวคือ ID ของตัวมันเอง → บรรทัด 152-153 คำนวณ mean ของ embedding ตัวมันเอง (ซึ่ง ณ จุดนั้นเป็นค่า random จากการ resize) → **mean ของค่า random = ค่า random เท่าเดิม ไม่มีผลอะไรเลย**

**สาเหตุที่เกิด:** ความสับสนเรื่องลำดับเวลา (temporal ordering) — comment ในโค้ดเขียนว่า *"Tokenize the new token with the OLD vocab (before adding it)"* แต่โค้ดจริงรันหลัง tokenizer ถูก replace ไปแล้ว ผู้เขียนลืมว่า tokenizer ตัวเก่าถูกเขียนทับไปด้วย `BertTokenizer.from_pretrained(str(output_dir))` ในบรรทัด 278

**ผลกระทบ:** คำไทย 20,000 คำที่เพิ่มเข้าไปทั้งหมดมี embedding เป็น **ค่าสุ่ม (random init)** ไม่ใช่ mean ของ subwords ตามที่ออกแบบ → การเทรน MLM ต้องเรียนรู้ embedding ของคำทั้งหมดจากศูนย์ ซึ่งช้ากว่าและแย่กว่าที่ควร (เป็นเหตุผลหนึ่งที่ทำให้ Phase 4 ของ Task 1 ใช้เวลาเทรนนานแต่ผลยังไม่ดี)

**วิธีแก้:** เก็บ reference ของ tokenizer ตัวเก่าไว้ก่อน reload หรือ tokenize คำใหม่ด้วย tokenizer เดิมเก็บเป็น mapping ไว้ก่อน แล้วค่อยส่ง mapping นั้นเข้า `smart_init_embeddings()`:
```python
# ก่อนเขียน vocab.txt ใหม่ — ใช้ tokenizer เดิมหา subword decomposition
old_tokenizer = AutoTokenizer.from_pretrained(str(base_path), use_fast=False)
decomposition = {t: old_tokenizer(t, add_special_tokens=False)["input_ids"]
                 for t in new_tokens}
# ... (add tokens, resize) ...
# แล้ว init จาก decomposition แทนการ tokenize ใหม่
```

---

### 2.3 Hardcoded Absolute Paths

**ปัญหาและตำแหน่ง:**

| ไฟล์ | Path ที่ hardcode |
|------|------------------|
| `scripts/extend_tokenizer.py:169` | `/home/aom/NECTEC/NER/outputs/models/SapBERTThaiMLM_CRF` (default ของ `--source-model`) |
| `scripts/generate_task1_nb.py:193,196` | `i:/NECTEC/thai_bert_lora/notebooks` |
| `scripts/generate_colab_nb.py:220-221` | `i:/NECTEC/thai_bert_lora/notebooks` |
| `scripts/train_wangchan_ner.py:353` | `outputs/wangchan_ner/merged_model` (แทนที่จะอ่านจาก `ner_config.yaml`) |
| `scripts/build_wangchan_mlm_corpus.py:35,57` | `data/processed`, `data/raw/data_EN.csv` (ไม่ใช้ config) |
| `README.md`, `docs/*.md` | อ้างอิง path `/home/aom/...` หลายจุด |

**สาเหตุที่เกิด:** โปรเจกต์เริ่มพัฒนาบนเครื่อง Linux (`/home/aom/`) แล้วย้ายมา Windows (`I:\`) — path เดิมถูกฝังไว้ในโค้ดแทนที่จะอยู่ใน config ทำให้รันข้ามเครื่องไม่ได้ทันที

**ผลกระทบ:** รันบนเครื่องอื่น/ย้าย drive แล้วพังทันที, `train_wangchan_ner.py` เปลี่ยน output path ใน config ก็ไม่มีผลกับ merge step

---

### 2.4 Frozen Embedding Bug — ✅ แก้แล้ว และ Retrain สำเร็จ (ปัญหาหลักของ Task 1)

**ปัญหา (ประวัติ):** หลังเทรน MLM บนข้อมูล 2.46 ล้านประโยค ค่า Perplexity ลดจาก 407,962 → 3,960 (ดูดีมาก) แต่ตอน inference โมเดลยังทาย `[MASK]` เป็นตัวเลข `['2','3','1','-',':']` ไม่ใช่คำไทย

**สาเหตุที่เกิด (root cause):** พฤติกรรม default ของ PEFT/LoRA คือ **freeze น้ำหนัก base ทั้งหมด** แล้วเทรนเฉพาะ LoRA matrices (query/key/value/dense) + เฉพาะ layer ที่ระบุใน `modules_to_save` เท่านั้นที่ได้เทรน

เมื่อ Phase 1 เพิ่มคำไทย 20,000 คำผ่าน `resize_token_embeddings()` น้ำหนักของคำใหม่ถูกสร้างแบบ random — แต่เพราะ `word_embeddings` และ `cls.predictions` (LM head) **ไม่อยู่ใน `modules_to_save`** น้ำหนัก random เหล่านั้นจึงถูกแช่แข็งถาวรตลอดการเทรน

โมเดลจึง "ไม่มีทางเลือก" — มันใช้คำไทยไม่ได้ (embedding เป็น random ค้างอยู่) เลยต้องลด loss ด้วยการทายเฉพาะคำเก่าที่ embedding ยังดีอยู่ = ตัวเลขและสัญลักษณ์ → **Perplexity ลดจริงเพราะโมเดลจำ distribution ของ corpus ได้ แต่ output head ชี้ไปคำไทยไม่ได้**

**สถานะการแก้:** เพิ่มใน `configs/lora_config.yaml`:
```yaml
modules_to_save:
  - "word_embeddings"
  - "cls.predictions"
```
และ `src/models/lora.py:73` ส่ง `modules_to_save` เข้า `LoraConfig` แล้ว ✅

**✅ ผลการ retrain (ยืนยันแล้วว่า fix ใช้ได้จริง — 2026-09-08):**
1. แก้ **Smart Init Bug (ข้อ 2.2)** ก่อน แล้วรัน `extend_tokenizer.py` ใหม่ → คำไทย 20,000 คำเริ่มจาก mean ของ subwords เดิม (ไม่ใช่ random)
2. Retrain `train_lora_mlm.py` → **MLM Loss 2.9584 / Perplexity 19.27** (จาก base 407,962) — `test_inference.py` ผ่าน: top-5 เป็นคำไทยตามบริบท ไม่ใช่ตัวเลขแล้ว
3. ขนาด adapter โตขึ้นตามที่คาด เพราะ `modules_to_save` ทำให้ PEFT เก็บสำเนาเต็ม (full copy) ของ `word_embeddings` + `cls.predictions` — เป็นพฤติกรรมปกติ ไม่ใช่ bug → adapter อยู่ที่ `outputs/adapters/lexitron_wiki_thai_lora_r32`
4. Merge แล้ว → `outputs/merged_model_1` และถูกนำไปเทรน NER ต่อใน Task 2 สำเร็จ (F1 = 0.8000)

---

## 3. 🟠 High/Medium Issues (ควรแก้)

### 3.1 `state.best_metric` อาจเป็น None — `src/training/callbacks.py:43`

```python
logger.info(f"Best checkpoint: {state.best_model_checkpoint} "
            f"(metric={state.best_metric:.4f})")
```
**สาเหตุ:** `on_save` ถูกเรียกทุกครั้งที่ checkpoint ถูกเซฟ ซึ่งอาจเกิดก่อนการ evaluate ครั้งแรก → `state.best_metric` ยังเป็น `None` → `f"{None:.4f}"` จะ raise `TypeError`

**วิธีแก้:** `if state.best_model_checkpoint and state.best_metric is not None:`

### 3.2 requirements.txt ขาด dependencies

**ขาด:** `seqeval` (ใช้ใน `train_wangchan_ner.py` สำหรับ entity-level F1) และ `sentencepiece` (ใช้ใน `extend_tokenizer.py`)

**สาเหตุที่เกิด:** requirements.txt เขียนขึ้นช่วงต้นโปรเจกต์ (Task 1 pipeline) แล้วไม่อัปเดตเมื่อเพิ่ม Task 2 scripts

**ผลกระทบ:** `pip install -r requirements.txt` บนเครื่องใหม่ → รัน `train_wangchan_ner.py` ได้แต่ F1 จะกลายเป็น fallback (ดูข้อ 3.3) หรือ `extend_tokenizer.py` fail ทันที

### 3.3 Fallback Metric หลอกชื่อ — `scripts/train_wangchan_ner.py:167-176`

**ปัญหา:** ถ้า `seqeval` ไม่ได้ติดตั้ง โค้ด fallback ไปคำนวณ **token-level accuracy (นับเฉพาะ non-O tokens)** แต่ return ภายใต้ key `"eval_f1"` เหมือนเดิม

**สาเหตุที่เกิด:** ต้องการให้ script รันได้แม้ไม่มี seqeval แต่ไม่เปลี่ยนชื่อ metric

**ผลกระทบ (อันตรายแบบเงียบ):**
- ค่าที่รายงานเป็น accuracy ไม่ใช่ entity-level F1 จริง → **เทียบตัวเลขกับผลเดิม (0.8269) ไม่ได้**
- `load_best_model_at_end=True` + `metric_for_best_model="eval_f1"` → Trainer อาจเลือก **best model ผิดตัว** เพราะ optimize คนละ metric โดยไม่รู้ตัว

**วิธีแก้:** fallback ควร return key คนละชื่อ เช่น `"eval_token_acc"` และ log warning ตัวใหญ่ๆ

### 3.4 Test ไร้ความหมาย — `tests/test_preprocessing.py:74-79`

```python
def test_build_corpus_filters_short(self):
    ...
    for item in corpus:
        assert len(item["text"]) >= 100 or True  # ← ผ่านเสมอไม่ว่าอะไรเกิด
```
**สาเหตุ:** `or True` ทำให้ assertion เป็น tautology — test นี้ไม่ได้ตรวจสอบอะไรเลยนอกจาก "ไม่ crash"

**วิธีแก้:** ควร assert ว่า `corpus == []` (เพราะ word="x" สั้นเกิน min_length=100 ทุก sample)

### 3.5 README.md บรรทัด 213 มี shell command ตกค้าง

```
~/NECTEC/thai_bert_lora$ /home/aom/NECTEC/NER/.venv/bin/python scripts/inference_ner.py
```
**สาเหตุ:** paste จาก terminal ตอนเขียน doc แล้วลืมลบ — ควรลบทิ้ง (เป็น path ของ Task 2 venv บนเครื่อง Linux ด้วย ซึ่งสื่อผิดบริบท)

### 3.6 Docstring ผิด — `scripts/train_wangchan_ner.py:9`

Docstring บอก model เป็น `CamembertForMaskedLM` แต่โค้ดจริงใช้ `CamembertForTokenClassification` (บรรทัด 187) → ทำให้ผู้อ่านสับสน

### 3.7 Auto-split ไม่ Stratified — `scripts/train_wangchan_ner.py:265-273`

**ปัญหา:** shuffle แล้วแบ่ง 80/20 ตรงๆ โดยไม่สน label distribution

**สาเหตุที่เกิด:** ใช้ `random.shuffle` ธรรมดา แทน stratified split

**ผลกระทบ:** ข้อมูลมีแค่ 44 ประโยค TH → val set อาจขาด entity บางประเภท (เช่น ไม่มี B-SYMPTOM เลย) ทำให้ F1 ที่วัดได้แกว่งมากระหว่าง run

**วิธีแก้:** ใช้ `sklearn.model_selection.train_test_split(..., stratify=...)` โดย stratify ด้วยจำนวน entity ต่อประโยค หรืออย่างน้อย entity type หลัก

### 3.8 Dead Code / Dead Parameters

| ตำแหน่ง | ปัญหา |
|---------|-------|
| `scripts/extend_tokenizer.py:175-176` | `--min-freq` argument ยังอยู่ แต่โค้ดที่ใช้ถูก comment ออก (บรรทัด 224) — user ส่งค่ามาก็ไม่มีผล |
| `scripts/train_wangchan_mlm.py:92-96` | `compute_metrics_mlm()` ถูก define แต่ไม่ได้ส่งเข้า Trainer |
| `src/data/loader.py:5` | `import re` ไม่ถูกใช้ |
| `src/data/cleaner.py:7` | `from typing import Optional` — ใช้จริงใน type hint (line 44) แต่โปรเจกต์ใช้ style `X \| None` ที่อื่นแล้ว ควรทำให้ konsisten |

### 3.9 Code Duplication — `load_jsonl_dataset()` เขียนซ้ำ 3+ ที่

ปรากฏใน: `scripts/train_lora_mlm.py:37-45`, `scripts/evaluate_mlm.py:42-49`, `scripts/train_wangchan_mlm.py:54-67` (ชื่อ `load_jsonl_texts`), `scripts/build_mlm_dataset.py:31-38` (ชื่อ `load_jsonl`), `scripts/train_wangchan_ner.py:61-63`

**สาเหตุที่เกิด:** scripts ของ Task 2 ถูกเขียนแบบ standalone (ไม่ import จาก src) ทำให้ logic เดิมถูก copy

**วิธีแก้:** รวมเป็นฟังก์ชันเดียวใน `src/data/loader.py` แล้วให้ทุก script import

### 3.10 Wangchan Scripts ไม่ Reuse Shared Library

`train_wangchan_ner.py` และ `train_wangchan_mlm.py` เขียน TrainingArguments/tokenize/LoRA setup ใหม่ทั้งหมด แทนที่จะใช้ `src/training/trainer.py` และ `src/models/lora.py`

**ผลกระทบระยะยาว:** แก้ bug ที่หนึ่ง (เช่น fp16 handling, seeding) ต้องไล่แก้หลายที่ และพฤติกรรมของแต่ละ pipeline จะค่อยๆ แยกออกจากกัน (drift)

---

## 4. 🟡 Low Priority Issues

| # | ปัญหา | ตำแหน่ง | รายละเอียด |
|---|-------|---------|-----------|
| 1 | ไม่มี `.gitignore` | project root | `.venv/` (43,000+ ไฟล์!), `__pycache__/`, `outputs/`, `data/interim/` ควรถูก exclude — ถ้า `git init` ตอนนี้จะ commit ไฟล์ขยะมหาศาล |
| 2 | ใช้ `random.seed()` ตรงๆ แทน `set_seed()` | `scripts/build_mlm_dataset.py:84,92` | ไม่ครอบคลุม numpy/torch → reproducibility ไม่สมบูรณ์ |
| 3 | `download_wangchanberta.py` ไม่มี logger/argparse/error handling | ทั้งไฟล์ | ไม่สอดคล้องกับ convention ของ scripts อื่น (ใช้ print + hardcode path) |
| 4 | Deprecated `evaluation_strategy` | `notebooks/task_1_colab.ipynb` (cell 7) | transformers เวอร์ชันใหม่ใช้ `eval_strategy` — ใน `src/training/trainer.py:74` แก้แล้วแต่ notebook ยังไม่แก้ |
| 5 | Token conversion ช้าใน loop | `scripts/inference_ner.py:98` | เรียก `convert_ids_to_tokens` ทีละ token — ควร batch convert ครั้งเดียว |
| 6 | Test coverage น้อย | `tests/` | มีแค่ 2 ไฟล์ — ขาด test ของ `splitter`, `lora.apply_lora`, `callbacks`, `resolve_overlapping_spans` (ซึ่งเป็น logic สำคัญของ Task 2) |
| 7 | `trust_remote_code=True` | `scripts/build_wiki_corpus.py:107` | ความเสี่ยง supply chain เล็กน้อย — dataset เป็นทางการของ Wikimedia จึงยอมรับได้ แต่ควรรู้ไว้ |
| 8 | Notebook generator scripts เป็น one-off | `scripts/generate_*.py` | สร้าง notebook เสร็จแล้วควร archive หรือลบ เพราะ hardcode path Windows (`i:/NECTEC/...`) |

---

## 5. ✅ Checklist ก่อน Retrain Task 1 — **เสร็จครบทั้งหมดแล้ว (2026-09-08)**

- [x] **1. แก้ Smart Init Bug (ข้อ 2.2)** — capture subword decomposition ด้วย tokenizer เดิมก่อน reload ✅
- [x] **2. แก้ TaskType hardcode (ข้อ 2.1)** — อ่าน `task_type` จาก config แล้ว ✅
- [x] **3. แก้ hardcoded `--source-model` path (ข้อ 2.3)** — ย้ายเข้า `configs/model_config.yaml` (`original_base_model_path`) ✅
- [x] **4. เพิ่ม `sentencepiece` + `seqeval` ใน requirements.txt (ข้อ 3.2)** ✅
- [x] **5. รัน `extend_tokenizer.py` ใหม่** — ได้ extended_base ที่ init จาก mean-subword จริง ✅
- [x] **6. Retrain:** `train_lora_mlm.py` — MLM Loss 2.9584 / PPL 19.27, adapter `outputs/adapters/lexitron_wiki_thai_lora_r32` ✅
- [x] **7. ทดสอบ:** `test_inference.py` — top-5 เป็นคำไทยตามบริบท ✅
- [x] **8. Merge:** `merge_lora.py` → `outputs/merged_model_1` ✅
- [x] **9. (เป้าหมายสุดท้าย) นำ merged model ไปทำ NER เทียบกับ WangchanBERTa** — `train_sapbert_ner.py` → **F1 = 0.8000** (`outputs/merged_model_1_ner_base`) vs baseline WangchanBERTa 0.8269 ✅

**สิ่งที่เหลือ (ไม่ blocking):** issue หมวด 3 (High/Medium) และหมวด 4 (Low) ด้านบนที่ยังไม่แก้

---

## 6. ข้อมูลสำคัญสำหรับการทำงานกับโปรเจกต์นี้

### Environment
- โปรเจกต์เดิมพัฒนาบน Linux (`/home/aom/NECTEC/`) ปัจจุบันอยู่ที่ Windows (`I:\NECTEC\thai_bert_lora`)
- `.venv` ในโปรเจกต์เป็น Python 3.12
- Task 2 เคยรันด้วย venv ของโปรเจกต์อื่น: `/home/aom/NECTEC/NER/.venv`

### โครงสร้างข้อมูล
- `data/raw/lexitron/telex.csv` — พจนานุกรม Lexitron (คอลัมน์ไทย: `t-entry`, `t-def`, `t-syn`, `t-ant`, `t-sample`; ไฟล์มี UTF-8 BOM — ต้องอ่านด้วย `utf-8-sig`)
- `data/raw/data_EN.csv` — ฉลากยา 49 rows (คอลัมน์: `drug_name_TH/EN`, `disease_TH/EN`, `symptom_TH/EN`, `description`, `source`)
- `data/interim/extended_base/` — SapBERT + vocab ไทย (สร้างใหม่จาก `extend_tokenizer.py` หลังแก้ Smart Init Bug — init จาก mean ของ subwords เดิมแล้ว)
- `data/interim/wangchanberta/` — WangchanBERTa base (404 MB)

### Config ที่สำคัญ
- `configs/lora_config.yaml` — MLM LoRA (r=32, alpha=64, **มี `modules_to_save` แก้ Frozen Embedding แล้ว**)
- `configs/sapbert_ner_config.yaml` — ★ NER หลักปัจจุบัน (SapBERT Thai, r=16, alpha=32, 100 epochs, auto-detect base จาก `outputs/merged_model_N` ล่าสุด)
- `configs/ner_config.yaml` — NER ของ WangchanBERTa baseline (r=16, 30 epochs, `metric_for_best_model: eval_f1`)
- `configs/model_config.yaml` — `base_model_path` ชี้ไป `data/interim/extended_base` (+ `original_base_model_path`)

### โมเดลผลลัพธ์ใน outputs/ (สถานะปัจจุบัน 2026-09-09)
- `outputs/merged_model_1/` — SapBERT Thai MLM merged (Task 1, PPL 19.27)
- `outputs/merged_model_1_ner_1/` — NER รอบแรก เทรนจาก `merged_model_1` (F1 = 0.7714)
- `outputs/merged_model_1_ner_base/` — NER รอบสอง (F1 = 0.8000) ★ โมเดล NER ที่ดีที่สุด — log การเทรนบันทึกว่า save เป็น `merged_model_1_ner_1_ner_1` (เพราะ `find_latest_merged_model` auto-detect ไปเลือก `merged_model_1_ner_1` ที่เป็น NER แล้วมาซ้อนอีกชั้น) ปัจจุบันโฟลเดอร์ถูก rename เป็น `merged_model_1_ner_base`
- `outputs/adapters/` — `lexitron_wiki_thai_lora_r32` (MLM), `merged_model_1_ner_1`, `merged_model_1_ner_1_ner_1` (NER adapters)
- ⚠️ `inference_ner.py` แบบ auto-detect จะเลือก `merged_model_1_ner_1` (F1 0.7714) เพราะ logic จัดลำดับตามตัวเลขท้ายชื่อ — ถ้าต้องการโมเดล F1 0.8000 ให้ระบุ `--model outputs/merged_model_1_ner_base`
- โฟลเดอร์ `outputs/wangchan_ner/` (baseline เดิม) ไม่อยู่ใน outputs/ แล้ว — เหลือแค่ตัวเลข F1 = 0.8269 ไว้อ้างอิง

### Convention ของโปรเจกต์
- ทุก script ใส่ `sys.path.insert(0, project_root)` ก่อน import จาก `src`
- ใช้ `src/utils/logger.get_logger(__name__, log_dir=Path("outputs/logs"))`
- ใช้ `src/utils/seed.set_seed()` สำหรับ reproducibility
- ภาษาไทยใน docstring/comments/log messages เป็นเรื่องปกติของโปรเจกต์นี้
- **ห้ามใช้ whitespace word-splitting กับภาษาไทย** — ใช้ tokenizer เสมอ
- Base model ต้องเป็น read-only — ห้าม overwrite (`model_config.yaml` → merged/adapters ออกไปที่ `outputs/` เท่านั้น)

### ผลการเทรนที่ผ่าน (อ้างอิง)
- Task 1 MLM: Perplexity 407,962 (base) → 3,960 (รอบพัง Frozen Embedding) → **19.27** (รอบแก้บัค, MLM Loss 2.9584; eval ล่าสุดบน test split: Loss 2.8843 / PPL 17.89)
- Task 2 SapBERT Thai NER (ปัจจุบัน): Best F1 = **0.8000** (100 epochs, Loss 0.4676 → `merged_model_1_ner_base`), รอบแรก F1 = 0.7714 (→ `merged_model_1_ner_1`)
- Task 2 WangchanBERTa baseline (ประวัติ): Best F1 = **0.8269** (epoch 28/30), precision 0.811, recall 0.843
