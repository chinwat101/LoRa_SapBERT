#!/usr/bin/env python3
"""
scripts/merge_wangchan_ner.py
Merge the LoRA adapter for WangchanBERTa NER into the base model.

This will output a standalone HuggingFace model (404 MB) that can be
loaded directly without the `peft` library.

Usage:
    python scripts/merge_wangchan_ner.py
"""
import json
import sys
from pathlib import Path

from peft import PeftModel
from transformers import AutoTokenizer, CamembertForTokenClassification


def main():
    base_model_dir = "data/interim/wangchanberta"
    adapter_dir = "outputs/wangchan_ner/adapter"
    merged_dir = "outputs/wangchan_ner/merged_model"
    
    print(f"Loading tokenizer from: {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    
    # Load label schema
    schema_path = Path(adapter_dir) / "label_schema.json"
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    label2id = schema["label2id"]
    id2label = {int(k): v for k, v in schema["id2label"].items()}
    
    print(f"Loading base model from: {base_model_dir}")
    base_model = CamembertForTokenClassification.from_pretrained(
        base_model_dir,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    
    print(f"Loading LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    print("Merging weights... (This may take a moment)")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged standalone model to: {merged_dir}")
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    
    print("Done! ✅")
    print(f"You can now load this model directly from: {merged_dir}")
    print("without needing the 'peft' library.")

if __name__ == "__main__":
    main()
