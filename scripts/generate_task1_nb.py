import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Tutorial: การสอนโมเดลภาษาอังกฤษให้เข้าใจภาษาไทยด้วย SapBERT (Task 1)\n",
                "---\n",
                "โน้ตบุ๊กนี้ถูกแปลงมาจากเอกสาร `docs/task_1.md` เพื่อให้คุณสามารถอ่านทำความเข้าใจและรันคำสั่ง (Run Cells) ได้ตามลำดับขั้นตอนจริงของการทำงานครับ\n",
                "\n",
                "**เป้าหมายหลัก:** นำโมเดล **SapBERT** (โมเดลการแพทย์ภาษาอังกฤษ) มาปรับจูน (Fine-tune) เพื่อใช้กับงานวิเคราะห์ชื่อเฉพาะทางคลินิก (Clinical NER) ในภาษาไทย โดยใช้เทคนิค **LoRA (Low-Rank Adaptation)**"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🛠️ การตั้งค่าสภาพแวดล้อม (Environment Setup)\n",
                "ก่อนเริ่ม ควรแน่ใจว่าเราทำงานอยู่ในโฟลเดอร์ที่ถูกต้อง"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "\n",
                "# ย้าย path กลับไปที่ root ของโปรเจกต์\n",
                "if os.getcwd().endswith('notebooks'):\n",
                "    os.chdir('..')\n",
                "print(\"Current Directory:\", os.getcwd())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1️⃣ Phase 1: การขยายคำศัพท์ (Vocabulary Expansion)\n",
                "**ปัญหาตั้งต้น:** โมเดล SapBERT ไม่มีคำศัพท์ภาษาไทยในตัว (Tokenizer ไม่รู้จัก) ทำให้มองเห็นภาษาไทยเป็น Unknown Tokens (`[UNK]`)\n",
                "\n",
                "**วิธีแก้:** เราได้ทำการเพิ่มคำศัพท์ภาษาไทยที่พบบ่อยจำนวน 20,000 คำ (สกัดจาก Wikipedia และ Lexitron) เข้าไปใน Tokenizer พร้อมปรับขนาด Embedding ของโมเดลให้รองรับคำใหม่ (น้ำหนักช่วงแรกจะเป็นการสุ่มค่า Random)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ตัวอย่างคำสั่งที่ใช้ในการสร้าง Lexitron Dataset และขยาย Tokenizer\n",
                "# (ถ้าทำไปแล้ว ข้ามเซลล์นี้ได้ครับ)\n",
                "!python scripts/preprocess_lexitron.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2️⃣ Phase 2: ปัญหาโมเดลติดรูปแบบพจนานุกรม (Overfitting to Dictionary)\n",
                "เมื่อลองเทรนโมเดลด้วยเทคนิค **Masked Language Modeling (MLM)** บนข้อมูลพจนานุกรม Lexitron (130,000 ประโยค)\n",
                "พบว่าโมเดลเรียนรู้โครงสร้างดีเกินไป เช่น พยายามทายคำตอบเป็นตัวเลข (1., 2.) หรือเครื่องหมายวรรคตอน แทนที่จะเป็นความหมายของภาษา"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3️⃣ Phase 3: การเพิ่มข้อมูล Wikipedia (Corpus Expansion)\n",
                "เพื่อแก้ปัญหานี้ เราจึงดาวน์โหลดบทความจาก **Thai Wikipedia** มาผสม เพื่อให้โมเดลเห็นประโยคที่เป็นธรรมชาติมากขึ้น รวมได้ข้อมูลกว่า 2.46 ล้านประโยค!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# คำสั่งสร้างชุดข้อมูลรวม (สมมติว่ารันสคริปต์นี้เพื่อรวม wiki)\n",
                "# (ระวัง: การดาวน์โหลด Wikipedia อาจใช้เวลานาน)\n",
                "!python scripts/build_wiki_corpus.py\n",
                "\n",
                "# และเตรียมข้อมูลสำหรับการเทรน MLM\n",
                "!python scripts/build_mlm_dataset.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4️⃣ Phase 4 & 5: การปลดล็อก The Frozen Embedding Bug\n",
                "🚨 **ปัญหาใหญ่ที่พบ:** เราพบว่าแม้จะเทรนไปนานแค่ไหน โมเดลก็ยังทายคำไทยออกมาไม่ได้ (ได้เป็นเครื่องหมายขีด หรือตัวเลข) สาเหตุเกิดจาก **LoRA ทำการแช่แข็ง (Freeze) Layer ของคำศัพท์ที่เราเพิ่งเติมเข้าไป** ทำให้คำไทย 20,000 คำไม่มีโอกาสได้เรียนรู้และอัปเดตน้ำหนัก (Weights) ของมันเลย!\n",
                "\n",
                "✅ **วิธีแก้:** ต้องแก้ `configs/lora_config.yaml` โดยเพิ่ม `modules_to_save` เพื่อปลดล็อกตารางคำศัพท์และหัวทำนายผล:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import yaml\n",
                "\n",
                "config_path = \"configs/lora_config.yaml\"\n",
                "if os.path.exists(config_path):\n",
                "    with open(config_path, \"r\", encoding=\"utf-8\") as f:\n",
                "        lora_cfg = yaml.safe_load(f)\n",
                "    \n",
                "    print(\"--- ตรวจสอบ LoRA Config ---\")\n",
                "    print(\"Modules to save (ควรปลดล็อก word_embeddings และ cls.predictions):\")\n",
                "    print(lora_cfg.get(\"modules_to_save\", \"Not Set!\"))\n",
                "else:\n",
                "    print(f\"ไม่พบไฟล์ {config_path}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5️⃣ ขั้นตอนต่อไป (Next Steps)\n",
                "หลังจากแก้ Config แล้ว เราพร้อมที่จะเทรนโมเดลใหม่อีกครั้งให้สมบูรณ์!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. เริ่มเทรน MLM ใหม่ด้วย LoRA ที่ถูกต้อง\n",
                "!python scripts/train_lora_mlm.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. รวมร่างโมเดล (Merge Model) เพื่อนำไปใช้งานแบบ Standalone\n",
                "!python scripts/merge_lora.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. ทดสอบการทำงานของโมเดล (Inference Test)\n",
                "# หวังว่าคราวนี้โมเดลจะทายคำภาษาไทยออกมาอย่างมีเหตุผล!\n",
                "!python scripts/test_inference.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "🎯 **เป้าหมายสูงสุด:** เมื่อแน่ใจว่าโมเดลอ่านภาษาไทยออกแล้ว ให้นำโมเดลที่ Merge แล้วไปใช้เป็น Base Model สำหรับการเทรนงาน NER ต่อไป (ซึ่งอยู่ใน Task 2 นั่นเอง)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# สร้างโฟลเดอร์ notebooks ถ้ายังไม่มี
os.makedirs('i:/NECTEC/thai_bert_lora/notebooks', exist_ok=True)

# เขียนไฟล์
with open('i:/NECTEC/thai_bert_lora/notebooks/task_1_tutorial.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print("Notebook created successfully at notebooks/task_1_tutorial.ipynb")
