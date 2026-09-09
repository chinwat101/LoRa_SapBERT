import os
from transformers import AutoTokenizer, AutoModelForMaskedLM

def download_sapbert(output_dir="data/interim/sapbert_raw"):
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
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
    download_sapbert()
