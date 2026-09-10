#!/usr/bin/env python3
"""
main.py
Run the entire SapBERT Thai training pipeline (Task 1 + Task 2) in one command.

Each step runs sequentially — later steps depend on outputs from earlier steps.

Usage:
    python main.py                  # Run everything (Task 1 + Task 2)
    python main.py --task 1         # Run only Task 1 (MLM Pre-training)
    python main.py --task 2         # Run only Task 2 (NER Fine-tuning)
    python main.py --start-step 3   # Resume from Step 3 onwards
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_step(step_num: str, description: str, command: list[str]) -> None:
    """Run a single pipeline step and abort if it fails."""
    print(f"\n{'='*70}")
    print(f"  Step {step_num}: {description}")
    print(f"  Command: {' '.join(command)}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run(command, cwd=str(Path(__file__).parent))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n❌ Step {step_num} FAILED (exit code {result.returncode})")
        print(f"   Command: {' '.join(command)}")
        sys.exit(result.returncode)

    print(f"\n✅ Step {step_num} completed in {elapsed:.1f}s")


def get_python() -> str:
    """Return the current Python executable path."""
    return sys.executable


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SapBERT Thai pipeline (Task 1 + Task 2)")
    parser.add_argument("--task", type=int, choices=[1, 2], default=None,
                        help="Run only Task 1 or Task 2 (default: both)")
    parser.add_argument("--start-step", type=float, default=1.0,
                        help="Start from this step number (e.g. 3.1 to skip downloads)")
    parser.add_argument("--base-model", default="outputs/merged_model_1",
                        help="Base model path for Task 2 NER (default: outputs/merged_model_1)")
    args = parser.parse_args()

    py = get_python()
    run_task1 = args.task is None or args.task == 1
    run_task2 = args.task is None or args.task == 2
    start = args.start_step

    print("=" * 70)
    print("  SapBERT Thai — Full Training Pipeline")
    print(f"  Python: {py}")
    print(f"  Task(s): {'1 + 2' if args.task is None else args.task}")
    print(f"  Start from step: {start}")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════
    # TASK 1: MLM Pre-training (สอน SapBERT ให้เข้าใจภาษาไทย)
    # ══════════════════════════════════════════════════════════════════════
    if run_task1:
        print("\n\n" + "█" * 70)
        print("  TASK 1: MLM Pre-training")
        print("█" * 70)

        # Step 1: Download
        if start <= 1.1:
            run_step("1.1", "Download SapBERT base model",
                     [py, "scripts/download_sapbert.py"])

        if start <= 1.2:
            run_step("1.2", "Download Lexitron 2.0 from NECTEC",
                     [py, "scripts/download_lexitron.py"])

        if start <= 1.3:
            run_step("1.3", "Download Thai Wikipedia from HuggingFace",
                     [py, "scripts/build_wiki_corpus.py"])

        # Step 2: Prepare data
        if start <= 2.1:
            run_step("2.1", "Preprocess Lexitron CSV",
                     [py, "scripts/preprocess_lexitron.py"])

        if start <= 2.2:
            run_step("2.2", "Build MLM dataset (Lexitron + Wikipedia)",
                     [py, "scripts/build_mlm_dataset.py", "--include-wiki"])

        # Step 3: Train
        if start <= 3.1:
            run_step("3.1", "Extend tokenizer with Thai vocabulary",
                     [py, "scripts/extend_tokenizer.py",
                      "--source-model", "data/interim/sapbert_raw"])

        if start <= 3.2:
            run_step("3.2", "Train MLM with LoRA",
                     [py, "scripts/train_lora_mlm.py"])

        if start <= 3.3:
            run_step("3.3", "Merge LoRA adapter into base model",
                     [py, "scripts/merge_lora.py"])

        if start <= 3.4:
            run_step("3.4", "Test MLM inference (fill-mask)",
                     [py, "scripts/test_inference.py"])

        if start <= 3.5:
            run_step("3.5", "Evaluate MLM perplexity",
                     [py, "scripts/evaluate_mlm.py"])

    # ══════════════════════════════════════════════════════════════════════
    # TASK 2: Clinical NER Fine-tuning (สกัดชื่อยา / โรค / อาการ)
    # ══════════════════════════════════════════════════════════════════════
    if run_task2:
        print("\n\n" + "█" * 70)
        print("  TASK 2: Clinical NER Fine-tuning")
        print("█" * 70)

        if start <= 4:
            run_step("4", "Convert NER dataset (data_EN.csv → BIO format)",
                     [py, "scripts/convert_ner_dataset.py"])

        if start <= 5:
            run_step("5", "Fine-tune SapBERT Thai NER with LoRA",
                     [py, "scripts/train_sapbert_ner.py",
                      "--base-model", args.base_model])

        if start <= 6:
            run_step("6", "Test NER inference",
                     [py, "scripts/inference_ner.py",
                      "--text", "อะบิราเทอโรน ใช้รักษามะเร็งต่อมลูกหมาก โดยมีอาการแพร่กระจาย"])

    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("  🎉 Pipeline completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
