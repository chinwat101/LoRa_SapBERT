import os
from transformers import AutoTokenizer, AutoModelForMaskedLM

def download_wangchanberta(output_dir="data/interim/wangchanberta"):
    model_name = "airesearch/wangchanberta-base-att-spm-uncased"
    print(f"Downloading {model_name}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    # Create directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and tokenizer
    print(f"Saving to {output_dir}...")
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    
    print("Download and save completed!")

if __name__ == "__main__":
    download_wangchanberta()
