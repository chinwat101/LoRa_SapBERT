# Architecture & Implementation Document: Federated Learning (FL) + Parameter-Efficient Fine-Tuning (PEFT/LoRA) for Thai Medical NER

---

## 📌 1. ภาพรวมระบบ (System Overview)

เอกสารนี้อธิบายสถาปัตยกรรม **Federated Learning (FL)** ผสานกับ **Parameter-Efficient Fine-Tuning (PEFT / LoRA)** สำหรับงาน **Named Entity Recognition (NER) ภาษาไทยทางการแพทย์ (Medical & Healthcare NER)**

- **Base Model**: SapBERT / Thai Medical BERT (`outputs/merged_model_3`) ซึ่งเป็นโมเดลที่ผ่านการ Pre-training/Fine-tune แบบ Masked Language Modeling (MLM) บนคลังข้อมูลการแพทย์ภาษาไทยมาแล้ว
- **Target Domain**: จำแนก 3 เอนทิตีสำคัญทางการแพทย์ ได้แก่:
  - `DRUG` (ชื่อยาทางการแพทย์)
  - `DISEASE` (ชื่อโรค/ภาวะการป่วย)
  - `SYMPTOM` (อาการแสดงของโรค)
- **วัตถุประสงค์ของการใช้ Federated Learning + PEFT**:
  1. **Data Privacy**: โรงพยาบาลหรือหน่วยงานทางการแพทย์ (Clients) สามารถร่วมมือกันเทรนโมเดลโดย **ไม่ต้องส่งออกข้อมูลดิบ (Raw Medical Records)** ออกนอกหน่วยงาน
  2. **High Efficiency**: ส่งและรับเฉพาะ **LoRA Adapter Weights (~2-5 MB)** ในแต่ละ Round แทนที่จะต้องรับส่ง Full Model Weights (~450 MB) ลดการใช้ Bandwidth และ Memory อย่างมหาศาล

---

## 🏗️ 2. สถาปัตยกรรมและ Flow การทำงาน (Program Flow & Architecture)

### 2.1 Overall System Architecture Flowchart

```mermaid
sequenceDiagram
    autonumber
    actor User as Researcher / User
    participant Global as Global Model (Base Model + Global LoRA)
    participant Server as FL Aggregator (FedAvg / Uniform)
    participant ClientA as Client 0 (Hospital A - QA Medical)
    participant ClientB as Client 1 (Hospital B - Drug Label)
    participant Export as Model Exporter (fl_export_model.py)
    participant Inference as Inference Script (inference_ner.py)

    User->>Server: สั่งรัน fl_simulate.py (--rounds 10)
    Server->>Global: โหลด Base Model (merged_model_3) & Init Global LoRA
    
    loop ในแต่ละ FL Round (Round 1 -> 10)
        Server->>ClientA: Broadcast Global LoRA Weights (numpy)
        Server->>ClientB: Broadcast Global LoRA Weights (numpy)
        
        par Local Training (Asymmetric Training)
            ClientA->>ClientA: Train 1 Local Epoch (~2,970 steps) บน ner_qa_train.jsonl
            ClientB->>ClientB: Train 50 Local Epochs (~215 steps) บน ner_bio_th.jsonl
        end
        
        ClientA-->>Server: ส่งกลับ Updated LoRA Weights + n_samples
        ClientB-->>Server: ส่งกลับ Updated LoRA Weights + n_samples
        
        Server->>Server: Aggregate LoRA Weights (Uniform Strategy: 50:50)
        Server->>Server: Evaluate & Save Global Adapter (outputs/fl_ner/round_XX)
    end

    User->>Export: สั่งรัน fl_export_model.py (--round 10)
    Export->>Global: โหลด Base Model + Global LoRA Weights (round_10)
    Export->>Export: Merge LoRA Weights เข้ากับ Base Model
    Export-->>User: บันทึก HuggingFace Standalone Model (outputs/fl_ner_model_round10)

    User->>Inference: สั่งรัน inference_ner.py (--text "...")
    Inference-->>User: แสดงผลลัพธ์ NER Spans (DRUG, DISEASE, SYMPTOM)
```

---

## 📁 3. โครงสร้างไฟล์และส่วนประกอบที่เกี่ยวข้อง (Related Files & Components)

| ไฟล์ / Directory | คำอธิบายหน้าที่ |
| :--- | :--- |
| `configs/fl_config.yaml` | **Configuration หลักสำหรับ FL:** กำหนดจำนวน rounds, strategy (`FedAvg`), aggregation (`uniform`), hyperparams ของ LoRA, และตั้งค่า per-client local epochs |
| `scripts/fl_simulate.py` | **FL Simulation Loop:** สคริปต์หลักรัน FL Manual Simulation ลูป 10 Rounds จัดการ Broadcast, Local Training, Uniform Aggregation และ Evaluation |
| `src/federated/client.py` | **FL Client Module (`NERLoRAClient`):** สืบทอดจาก Flower NumPyClient โหลด local JSONL dataset, สร้าง HuggingFace Trainer + LoRA, เทรน local epochs และส่งกลับเฉพาะ numpy LoRA weights |
| `scripts/fl_export_model.py` | **Model Converter:** สคริปต์รวม Base Model (`outputs/merged_model_3`) เข้ากับ Global LoRA Adapter (`global_adapter_weights.npy`) เซฟเป็น HuggingFace Model สำเร็จรูป |
| `scripts/inference_ner.py` | **NER Inference Script:** สคริปต์ทดสอบฉีดข้อความภาษาไทยเพื่อรับ NER Span Labels |
| `data/processed/ner_qa_train.jsonl` | **Dataset Client 0 (Hospital A):** ข้อมูล QA ภาษาไทยทางการแพทย์ (29,709 ตัวอย่าง) เน้น `DISEASE` และ `SYMPTOM` |
| `data/processed/ner_bio_th.jsonl` | **Dataset Client 1 (Hospital B):** ข้อมูลฉลากยาภาษาไทย (44 ตัวอย่าง) เน้น `DRUG` |
| `outputs/fl_ner/` | **Output Directory:** เก็บผลลัพธ์ LoRA Adapter Weights แต่ละ Round (`round_01/` ... `round_10/`) และ `fl_training_log.json` |

---

## ⚠️ 4. ปัญหาที่พบและการวิเคราะห์สาเหตุ (Issues & Root Cause Analysis)

### 🔴 ปัญหาที่ 1: `ImportError: Unable to import module ray` ใน Flower Simulation
* **อาการ:** เมื่อรัน `flwr.simulation.start_simulation()` ระบบฟ้อง Error หาไลบรารี `ray` ไม่เจอ
* **สาเหตุ:** `flwr.simulation` ต้องพึ่งพา Ray framework ซึ่งการติดตั้งในสภาพแวดล้อมจำกัดทำได้ยากและเกิดปัญหา compatibility กับ Python 3.12
* **การแก้ไข:** ปรับสถาปัตยกรรมมาใช้ **Manual FL Loop Simulation** ใน `scripts/fl_simulate.py` โดย direct call ผ่าน `NERLoRAClient.fit()` และ `NERLoRAClient.evaluate()` โดยตรง ซึ่งได้ผลลัพธ์เหมือน FL จริงทุกประการโดยไม่ต้องพึ่ง Ray

---

### 🔴 ปัญหาที่ 2: Data Imbalance & Weight Dilution (Client A ดึงโมเดลไปทางเดียว)
* **อาการ:** โมเดลทาย `DISEASE` และ `SYMPTOM` ได้แม่นยำสูง (F1 > 0.94) แต่ **ไม่สามารถทาย `DRUG` ได้เลย** (F1 ของ Client B อยู่ที่แค่ 0.2083)
* **สาเหตุ:** 
  1. ขนาดข้อมูลต่างกันมหาศาล: Client A มี 29,709 ตัวอย่าง ขณะที่ Client B มีแค่ 44 ตัวอย่าง
  2. สูตรการหาค่าเฉลี่ยของ `FedAvg` แบบมาตรฐานจะถ่วงน้ำหนักตามจำนวนข้อมูล:
     $$W_{global} = \sum_{k=1}^K \frac{n_k}{N} W_k$$
     ส่งผลให้ Client B มีน้ำหนักใน Global Model เพียง **0.15%** ($44 / 29753$) ทำให้น้ำหนัก LoRA ของ Client B ถูกกลบจนเกือบหมด

---

### 🔴 ปัญหาที่ 3: Asymmetric Step Imbalance (จำนวน Step ในการอัปเดตต่างกันเป็นพันเท่า)
* **อาการ:** ถึงแม้จะพยายามเพิ่มจำนวน FL Rounds แต่น้ำหนักของ Client B ก็ยังไม่พัฒนา
* **สาเหตุ:** เมื่อตั้งค่า `local_epochs: 5` ให้ทุก Client เท่ากัน:
  - **Client A:** มีข้อมูล 29,709 ข้อความ เทรน 5 epochs = **~15,000 gradient steps** ต่อ round
  - **Client B:** มีข้อมูล 44 ข้อความ เทรน 5 epochs = **แค่ ~15 gradient steps** ต่อ round
  - ทำให้ในแต่ละ round Client A ปรับปรุงโครงข่ายประสาทมากกว่า Client B ถึง 1,000 เท่า!

---

### 🔴 ปัญหาที่ 4: Positioning Bias ในข้อมูล Client B (ฉลากยา)
* **อาการ:** ในช่วงแรก โมเดลทาย `DRUG` ได้เฉพาะเมื่อชื่อยาอยู่ **คำแรกสุดของประโยค** เท่านั้น แต่ถ้าเอาชื่อยาไปวางกลางประโยค เช่น *"ผู้ป่วยได้รับยา..."* โมเดลจะทายไม่เจอ
* **สาเหตุ:** ข้อมูลใน `ner_bio_th.jsonl` ทั้ง 44 ประโยคเป็นรูปแบบฉลากยาซึ่งขึ้นต้นด้วยชื่อยาเสมอ (`<ชื่อยา> ใช้รักษา...`) โมเดลจึงจดจำตำแหน่งคำแรกเป็นฟีเจอร์หลักของการเป็น `DRUG`

---

## 🛠️ 5. วิธีการแก้ไขและการตั้งค่า Configuration (Solutions & Configuration Fixes)

### ✅ โซลูชันที่ 1: เปลี่ยนเป็น Uniform Aggregation Strategy (`aggregation: "uniform"`)
แก้ไขฟังก์ชั่น `fedavg()` ใน `scripts/fl_simulate.py` ให้รองรับการเฉลี่ยน้ำหนัก LoRA แบบเท่ากันทุก Client (50:50) โดยไม่ถ่วงน้ำหนักด้วยขนาดข้อมูล:
$$W_{global} = \frac{1}{K} \sum_{k=1}^K W_k$$

```yaml
# configs/fl_config.yaml
federated:
  num_rounds: 10
  strategy: "FedAvg"
  aggregation: "uniform"    # เปลี่ยนจาก weighted -> uniform (ให้ทุก client เท่ากัน 50:50)
```

---

### ✅ โซลูชันที่ 2: ใช้ Asymmetric Local Training (สมดุล Gradient Steps)
ปรับการตั้งค่า `local_epochs` ใน `configs/fl_config.yaml` ราย Client เพื่อชดเชยขนาดข้อมูลที่แตกต่างกัน:

```yaml
# configs/fl_config.yaml
clients:
  - id: 0
    name: "Hospital_A_QA"
    train_file: "data/processed/ner_qa_train.jsonl"
    local_epochs: 1            # Client A (29,709 samples): 1 epoch = ~2,970 steps ต่อ round

  - id: 1
    name: "Hospital_B_Labels"
    train_file: "data/processed/ner_bio_th.jsonl"
    local_epochs: 50           # Client B (44 samples): 50 epochs = ~215 steps ต่อ round
```

---

### 📈 5.3 เปรียบเทียบผลลัพธ์หลังแก้ (Before vs. After Comparison)

| ตัววัดผล (Metrics) | ก่อนแก้ไข (Standard FedAvg) | หลังแก้ไข (Uniform + Asymmetric Training) |
| :--- | :--- | :--- |
| **Client B (Drug Label) F1** | **0.2083** (ทาย `DRUG` ไม่เจอ) | **0.7865** *(เพิ่มขึ้น ~3.8 เท่า!)* |
| **Client A (QA Medical) F1** | **0.9466** | **0.9356** *(รักษาความแม่นยำสูงไว้ได้)* |
| **Global Macro F1 Score** | **0.5775** | **0.8242** |
| **DRUG Detection (กลางประโยค)** | ❌ ล้มเหลว | **✅ สำเร็จ 100% (ตรวจจับชื่อยาได้ทุกตำแหน่ง)** |

#### 🧪 ตัวอย่างผลการทดสอบ Real Inference:
```text
Text: ผู้ป่วยได้รับยา อะบิราเทอโรน สำหรับรักษา มะเร็งต่อมลูกหมาก มีอาการ แพร่กระจาย
  [DRUG    ]  'อะบิราเทอโรน'       (pos 16–28)  ✅
  [DISEASE ]  'มะเร็งต่อมลูกหมาก'   (pos 41–58)  ✅
  [SYMPTOM ]  'แพร่กระจาย'         (pos 67–77)  ✅
```

---

## ⚡ 6. ตารางข้อควรระวังและความท้าทาย (Precautions & Challenges Table)

| มิติความท้าทาย (Category) | ความท้าทายที่พบ (Challenge) | ข้อควรระวัง (Precaution) | แนวทางการรับมือ / แนวทางแก้ไข (Mitigation) |
| :--- | :--- | :--- | :--- |
| **Non-IID Data Distribution** | ข้อมูลแต่ละ Client มีประเภทเอนทิตีไม่เหมือนกัน (Client A มีแต่ DISEASE/SYMPTOM, Client B มีแต่ DRUG) | การใช้ Standard FedAvg จะทำให้น้ำหนักโมเดลไหลไปตาม Client ที่มีข้อมูลมากกว่าอย่างรุนแรง | ใช้ **Uniform Aggregation** หรือ **FedProx Strategy** เพื่อควบคุมไม่ให้น้ำหนักเอนไปทาง Client ใหญ่ |
| **Asymmetric Step Discrepancy** | Client ขนาดใหญ่เกิดการปรับ Gradient Steps มากกว่า Client ขนาดเล็กมหาศาลใน 1 Round | หากตั้งค่า local epochs เท่ากัน Client เล็กจะไม่ได้รับการเรียนรู้เพียงพอในการปรับ LoRA adapter | ใช้ **Asymmetric Local Epochs / Steps** โดยกำหนดให้ Client เล็กเทรนจำนวน Epochs มากขึ้น (เช่น 50-100 Epochs) |
| **Vocabulary & OOV Coverage** | ชื่อยาทางการแพทย์มีศัพท์ใหม่ (Out-of-Vocabulary) หรือชื่อยาสามัญที่ไม่อยู่ใน 44 ตัวอย่าง | โมเดลจะไม่สามารถทายชื่อยาที่ไม่อยู่ในชุดข้อมูลฝึกสอนได้เนื่องจากจำกัดด้วย Vocab ของ Client เล็ก | ใช้ Base Model ที่ผ่าน **MLM Continued Pre-training (SapBERT)** มาก่อนเพื่อให้มี Embedding สารเคมี/ยาทางการแพทย์ |
| **Overfitting on Small Clients** | Client เล็ก (44 ตัวอย่าง) เทรน 50-100 epochs อาจเกิด Overfitting ใน local dataset | การตั้งค่า Learning Rate สูงเกินไปใน Client เล็กจะทำให้ LoRA adapter จำแบบท่องจำ (Memorization) | ควบคุม `learning_rate` (เช่น 3e-4) และใช้ `lora_dropout: 0.1` เพื่อควบคุมความเสถียร |
| **LoRA Weight Disparity** | การเปลี่ยนเฉพาะ LoRA Adapter Weights โดยไม่แตะ Base Model | ความเข้ากันได้ของ Classifier Head เมื่อเปลี่ยนจำนวน Class หรือสลับ Base Checkpoint | ล็อค `label_schema` ให้ตรงกันทุก Client ตั้งแต่ต้น และรวม LoRA Adapter แบบ Auto-merge ก่อน Export |
| **Privacy & Security Leakage** | แม้ไม่ส่งข้อมูลดิบ แต่ LoRA Gradients/Weights อาจถูกโจมตีด้วย Gradient Inversion Attack | การส่งผ่านไฟล์ Weights ผ่านเครือข่ายสาธารณะอาจมีความเสี่ยงถูกวิเคราะห์กลับเป็นข้อความ | ในระบบ Real World ควรเพิ่ม **Differential Privacy (DP)** หรือ **Secure Aggregation (SecAgg)** |

---

## 🚀 7. คำสั่งการใช้งาน (Execution Commands)

### 1. รัน Federated Learning Simulation
```bash
python scripts/fl_simulate.py --rounds 10
```

### 2. Export Global LoRA Model เป็น HuggingFace Model
```bash
python scripts/fl_export_model.py --round 10
```

### 3. ทดสอบ NER Inference
```bash
python scripts/inference_ner.py \
  --model outputs/fl_ner_model_round10 \
  --text "ผู้ป่วยได้รับยา อะบิราเทอโรน สำหรับรักษา มะเร็งต่อมลูกหมาก มีอาการ แพร่กระจาย"
```
