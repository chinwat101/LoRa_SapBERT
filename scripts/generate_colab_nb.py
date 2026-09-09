import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Thai BERT LoRA: MLM Training (Colab Version)\n",
                "---\n",
                "นี่คือ Notebook สำหรับนำไปรันเทรนโมเดล (Task 1: MLM) บน Google Colab ครับ โค้ดทั้งหมดที่จำเป็นถูกรวบรวมไว้ในนี้แล้ว เพื่อให้รันได้จบในหน้าเดียวโดยไม่ต้องพึ่งพาไฟล์โค้ด `.py` อื่นๆ ในโปรเจกต์\n",
                "\n",
                "⚠️ **ข้อควรระวังก่อนรัน:**\n",
                "คุณจำเป็นต้องอัปโหลดโฟลเดอร์โมเดลตั้งต้น (`data/interim/wangchanberta` หรือ SapBERT) และไฟล์ Dataset (`train.jsonl`, `validation.jsonl`) ขึ้น Google Drive ของคุณก่อน จากนั้นเมานท์ (Mount) Google Drive เข้ากับ Colab เพื่อให้โค้ดสามารถดึงไฟล์ไปใช้ได้"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. ติดตั้งไลบรารีที่จำเป็น\n",
                "!pip install -q transformers peft datasets seqeval pyyaml"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# (ทางเลือก) เมานท์ Google Drive หากข้อมูลของคุณอยู่ใน Google Drive\n",
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "# TODO: แก้ไข Path ให้ตรงกับที่คุณเก็บไฟล์ไว้ใน Drive\n",
                "BASE_MODEL_PATH = \"/content/drive/MyDrive/thai_bert_lora/data/interim/wangchanberta\" # แก้ path ตรงนี้\n",
                "TRAIN_DATA_PATH = \"/content/drive/MyDrive/thai_bert_lora/data/processed/mlm_train.jsonl\"\n",
                "VAL_DATA_PATH = \"/content/drive/MyDrive/thai_bert_lora/data/processed/mlm_val.jsonl\"\n",
                "OUTPUT_DIR = \"/content/drive/MyDrive/thai_bert_lora/outputs/lora_adapter\""
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. นำเข้าไลบรารี (Imports)\n",
                "import json\n",
                "import math\n",
                "import torch\n",
                "from pathlib import Path\n",
                "from datasets import Dataset\n",
                "from transformers import (\n",
                "    AutoTokenizer,\n",
                "    BertForMaskedLM,  # เปลี่ยนเป็น CamembertForMaskedLM หากใช้ WangchanBERTa ทำ MLM\n",
                "    DataCollatorForLanguageModeling,\n",
                "    Trainer,\n",
                "    TrainingArguments\n",
                ")\n",
                "from peft import LoraConfig, get_peft_model, TaskType"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. ตั้งค่าพารามิเตอร์ (Configurations)\n",
                "lora_cfg = {\n",
                "    \"r\": 32,\n",
                "    \"lora_alpha\": 64,\n",
                "    \"lora_dropout\": 0.05,\n",
                "    \"bias\": \"none\",\n",
                "    \"target_modules\": [\"query\", \"key\", \"value\", \"dense\"],\n",
                "    # ปลดล็อก Layer เหล่านี้เพื่อแก้ปัญหา The Frozen Embedding Bug!\n",
                "    \"modules_to_save\": [\"word_embeddings\", \"cls.predictions\"]\n",
                "}\n",
                "\n",
                "training_cfg = {\n",
                "    \"num_train_epochs\": 3,\n",
                "    \"per_device_train_batch_size\": 16,\n",
                "    \"per_device_eval_batch_size\": 16,\n",
                "    \"learning_rate\": 2e-4,\n",
                "    \"max_seq_length\": 128,\n",
                "    \"mlm_probability\": 0.15,\n",
                "    \"fp16\": True # ใช้ True ได้เลยเพราะรันบน Colab GPU\n",
                "}"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. ฟังก์ชันโหลด Dataset\n",
                "def load_jsonl_dataset(path):\n",
                "    records = []\n",
                "    with open(path, encoding=\"utf-8\") as f:\n",
                "        for line in f:\n",
                "            line = line.strip()\n",
                "            if line:\n",
                "                records.append(json.loads(line))\n",
                "    return Dataset.from_list(records)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. ฟังก์ชันเตรียมข้อมูล (Tokenization)\n",
                "def tokenize_dataset(dataset, tokenizer, max_length):\n",
                "    def tokenize_fn(examples):\n",
                "        return tokenizer(\n",
                "            examples[\"text\"],\n",
                "            truncation=True,\n",
                "            max_length=max_length,\n",
                "            padding=False,  # DataCollator จะจัดการให้เอง\n",
                "            return_special_tokens_mask=True,\n",
                "        )\n",
                "    return dataset.map(tokenize_fn, batched=True, remove_columns=[\"text\"])"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. ขั้นตอนโหลดโมเดลและเทรน (Main Execution)\n",
                "\n",
                "# --- โหลดข้อมูล ---\n",
                "print(\"Loading datasets...\")\n",
                "train_dataset = load_jsonl_dataset(TRAIN_DATA_PATH)\n",
                "eval_dataset = load_jsonl_dataset(VAL_DATA_PATH)\n",
                "print(f\"Train: {len(train_dataset)} | Val: {len(eval_dataset)}\")\n",
                "\n",
                "# --- โหลด Base Model และ Tokenizer ---\n",
                "print(f\"Loading model from {BASE_MODEL_PATH}...\")\n",
                "tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)\n",
                "\n",
                "# 💡 Note: หากใช้ WangchanBERTa ให้เปลี่ยนจาก BertForMaskedLM เป็น CamembertForMaskedLM นะครับ\n",
                "model = BertForMaskedLM.from_pretrained(BASE_MODEL_PATH)\n",
                "\n",
                "# --- ทำการครอบด้วย LoRA ---\n",
                "print(\"Applying LoRA...\")\n",
                "peft_config = LoraConfig(\n",
                "    task_type=TaskType.FEATURE_EXTRACTION,\n",
                "    r=lora_cfg[\"r\"],\n",
                "    lora_alpha=lora_cfg[\"lora_alpha\"],\n",
                "    lora_dropout=lora_cfg[\"lora_dropout\"],\n",
                "    bias=lora_cfg[\"bias\"],\n",
                "    target_modules=lora_cfg[\"target_modules\"],\n",
                "    modules_to_save=lora_cfg[\"modules_to_save\"]\n",
                ")\n",
                "model = get_peft_model(model, peft_config)\n",
                "model.print_trainable_parameters()\n",
                "\n",
                "# --- เตรียมข้อมูลก่อนเทรน ---\n",
                "print(\"Tokenizing data...\")\n",
                "train_tok = tokenize_dataset(train_dataset, tokenizer, training_cfg[\"max_seq_length\"])\n",
                "eval_tok = tokenize_dataset(eval_dataset, tokenizer, training_cfg[\"max_seq_length\"])\n",
                "\n",
                "data_collator = DataCollatorForLanguageModeling(\n",
                "    tokenizer=tokenizer,\n",
                "    mlm=True,\n",
                "    mlm_probability=training_cfg[\"mlm_probability\"],\n",
                ")\n",
                "\n",
                "# --- ตั้งค่า Trainer ---\n",
                "training_args = TrainingArguments(\n",
                "    output_dir=OUTPUT_DIR,\n",
                "    num_train_epochs=training_cfg[\"num_train_epochs\"],\n",
                "    per_device_train_batch_size=training_cfg[\"per_device_train_batch_size\"],\n",
                "    per_device_eval_batch_size=training_cfg[\"per_device_eval_batch_size\"],\n",
                "    learning_rate=training_cfg[\"learning_rate\"],\n",
                "    evaluation_strategy=\"epoch\",\n",
                "    save_strategy=\"epoch\",\n",
                "    fp16=training_cfg[\"fp16\"],\n",
                "    report_to=\"none\"\n",
                ")\n",
                "\n",
                "trainer = Trainer(\n",
                "    model=model,\n",
                "    args=training_args,\n",
                "    train_dataset=train_tok,\n",
                "    eval_dataset=eval_tok,\n",
                "    data_collator=data_collator,\n",
                ")\n",
                "\n",
                "# --- เริ่มเทรน ---\n",
                "print(\"🚀 Starting Training...\")\n",
                "trainer.train()\n",
                "\n",
                "# --- บันทึกผลลัพธ์ (LoRA Adapter) ---\n",
                "model.save_pretrained(OUTPUT_DIR)\n",
                "tokenizer.save_pretrained(OUTPUT_DIR)\n",
                "print(f\"✅ Training Complete! Adapter saved to {OUTPUT_DIR}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. ผสานโมเดล (Merge LoRA into Base Model)\n",
                "import os\n",
                "import gc\n",
                "from pathlib import Path\n",
                "from peft import PeftModel\n",
                "\n",
                "# ล้างหน่วยความจำก่อนโหลดโมเดลใหม่\n",
                "del trainer\n",
                "gc.collect()\n",
                "torch.cuda.empty_cache()\n",
                "\n",
                "print(\"Loading base model for merging...\")\n",
                "base_model = BertForMaskedLM.from_pretrained(BASE_MODEL_PATH)\n",
                "\n",
                "print(f\"Loading adapter from {OUTPUT_DIR}...\")\n",
                "lora_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)\n",
                "\n",
                "print(\"Merging weights...\")\n",
                "merged_model = lora_model.merge_and_unload()\n",
                "\n",
                "# หารันนัมเบอร์เพื่อแยกโฟลเดอร์โมเดลที่ผสานแล้ว\n",
                "base_merged_dir = Path(\"/content/drive/MyDrive/thai_bert_lora/outputs/merged_model\")\n",
                "run_num = 1\n",
                "while True:\n",
                "    merged_dir = base_merged_dir.with_name(f\"{base_merged_dir.name}_{run_num}\")\n",
                "    if not merged_dir.exists():\n",
                "        break\n",
                "    run_num += 1\n",
                "\n",
                "print(f\"Saving merged model to {merged_dir}...\")\n",
                "merged_dir.mkdir(parents=True, exist_ok=True)\n",
                "merged_model.save_pretrained(str(merged_dir))\n",
                "tokenizer.save_pretrained(str(merged_dir))\n",
                "print(\"✅ Merging complete! โมเดลต้นฉบับไม่ได้ถูกแก้ไขใดๆ\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

os.makedirs('i:/NECTEC/thai_bert_lora/notebooks', exist_ok=True)
with open('i:/NECTEC/thai_bert_lora/notebooks/task_1_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print("Colab Notebook created successfully at notebooks/task_1_colab.ipynb")
