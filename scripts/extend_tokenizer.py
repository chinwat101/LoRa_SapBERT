#!/usr/bin/env python3
"""
scripts/extend_tokenizer.py
Step 0 (Pre-Training): Extend the SapBERT tokenizer with Thai vocabulary from Lexitron.

เนื่องจาก SapBERT ใช้ tokenizer ที่ไม่มี vocabulary ภาษาไทย (UNK rate ~45%)
script นี้จะ:
  1. รวบรวม text ภาษาไทยทั้งหมดจาก MLM corpus ที่สร้างไว้แล้ว
  2. Train SentencePiece BPE vocabulary ใหม่สำหรับภาษาไทย
  3. Diff กับ vocab เดิม หาเฉพาะ token ที่ขาดอยู่
  4. เพิ่ม token เหล่านั้นลง tokenizer + resize embedding matrix
  5. Initialize embedding ใหม่ด้วยค่า mean ของ subword embeddings เดิม
     (ดีกว่า random init มาก)
  6. บันทึก extended model+tokenizer → data/interim/extended_base/

Usage:
    python scripts/extend_tokenizer.py
    python scripts/extend_tokenizer.py --new-vocab-size 6000
    python scripts/extend_tokenizer.py --source-model /path/to/original/sapbert
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("outputs/logs"))

# ──────────────────────────────────────────────────────────────────────────────
# Helper: Collect Thai texts from JSONL corpus
# ──────────────────────────────────────────────────────────────────────────────

def collect_texts(jsonl_paths: list[Path]) -> list[str]:
    texts = []
    for p in jsonl_paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    t = obj.get("text", "").strip()
                    if t:
                        texts.append(t)
    logger.info(f"Collected {len(texts):,} text samples from corpus")
    return texts


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Train SentencePiece BPE on Thai corpus
# ──────────────────────────────────────────────────────────────────────────────

def train_spm(texts: list[str], vocab_size: int, output_prefix: Path) -> list[str]:
    """Train SentencePiece BPE model and return the new vocabulary."""
    try:
        import sentencepiece as spm
    except ImportError:
        logger.error(
            "sentencepiece not installed.\n"
            "Run: pip install sentencepiece"
        )
        sys.exit(1)

    # Write corpus to temp file (SentencePiece needs a file path)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")
        corpus_file = f.name

    logger.info(f"Training SentencePiece BPE (vocab_size={vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=str(output_prefix),
        vocab_size=vocab_size,
        character_coverage=0.9995,  # high coverage for Thai Unicode range
        model_type="bpe",
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
    )

    # Load and extract vocabulary
    sp = spm.SentencePieceProcessor()
    sp.Load(str(output_prefix) + ".model")
    new_vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    logger.info(f"SentencePiece trained | vocab size: {len(new_vocab):,}")

    Path(corpus_file).unlink(missing_ok=True)
    return new_vocab


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Find truly new tokens (missing from existing tokenizer)
# ──────────────────────────────────────────────────────────────────────────────

def find_new_tokens(existing_vocab: set[str], new_vocab: list[str]) -> list[str]:
    # Strip SPM prefix marker U+2581 ( ) when comparing
    spm_prefix = chr(9601)  # ' '
    new_tokens = []
    for tok in new_vocab:
        tok_clean = tok.lstrip(spm_prefix)
        if not tok_clean:
            continue
        # For WordPiece, we want both the bare token and the ## subword token
        bare_tok = tok_clean
        sub_tok = f"##{tok_clean}"
        
        if bare_tok not in existing_vocab:
            new_tokens.append(bare_tok)
        if sub_tok not in existing_vocab:
            new_tokens.append(sub_tok)
            
    # Deduplicate while preserving order
    seen = set()
    result = []
    for t in new_tokens:
        if t not in seen:
            seen.add(t)
            result.append(t)
    logger.info(f"New Thai tokens to add (bare + ##): {len(result):,}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Step 3+4+5: Add tokens, resize embedding, smart initialize
# ──────────────────────────────────────────────────────────────────────────────

def smart_init_embeddings(
    model,
    tokenizer,
    new_tokens: list[str],
    decomposition: dict[str, list[int]],
) -> None:
    """
    Initialize new embedding rows as the mean of the subword pieces
    from the OLD tokenizer's decomposition.

    The decomposition mapping MUST be computed with the OLD tokenizer
    BEFORE it is replaced by the new one. Otherwise the new tokenizer
    already knows the new tokens and the decomposition becomes circular
    (mean of a single random vector = the same random vector).

    Args:
        model: Model with resized embeddings.
        tokenizer: NEW tokenizer (used only to look up new token IDs).
        new_tokens: List of new token strings.
        decomposition: token string → list of OLD tokenizer subword IDs.
    """
    embedding_matrix = model.get_input_embeddings().weight.data
    unk_id = tokenizer.unk_token_id

    initialized = 0
    unk_fallback = 0
    skipped = 0

    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        sub_ids = decomposition.get(token, [])

        # Filter out UNK IDs — they carry no useful signal
        meaningful_ids = [sid for sid in sub_ids if sid != unk_id]

        if meaningful_ids:
            # Mean of known sub-piece embeddings (from the OLD, trained vocab)
            sub_embeddings = embedding_matrix[meaningful_ids]
            embedding_matrix[token_id] = sub_embeddings.mean(dim=0)
            initialized += 1
        elif sub_ids:
            # Decomposition was all UNK — use UNK embedding as fallback
            # (still better than random: UNK is a trained vector)
            embedding_matrix[token_id] = embedding_matrix[unk_id].clone()
            unk_fallback += 1
        else:
            # No decomposition available — leave as random init
            skipped += 1

    logger.info(
        f"Smart embedding init complete: "
        f"{initialized} from subwords, {unk_fallback} from UNK fallback, "
        f"{skipped} skipped (no decomposition)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extend SapBERT tokenizer with Thai vocab")
    parser.add_argument("--model-config",   default="configs/model_config.yaml")
    parser.add_argument("--data-config",    default="configs/data_config.yaml")
    parser.add_argument(
        "--source-model",
        default=None,
        help="Path to the ORIGINAL base model to extend (never overwritten). "
             "Overrides 'original_base_model_path' in model_config.yaml.",
    )
    parser.add_argument("--new-vocab-size", type=int, default=6000,
                        help="Size of the new SentencePiece BPE vocabulary")
    args = parser.parse_args()

    with open(Path(args.model_config), encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)["model"]
    with open(Path(args.data_config), encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)["data"]

    # Resolve source model path: CLI > config > error
    source_model = args.source_model or model_cfg.get("original_base_model_path")
    if not source_model:
        logger.error(
            "No source model path specified.\n"
            "Set 'original_base_model_path' in configs/model_config.yaml\n"
            "  or pass --source-model <path>"
        )
        sys.exit(1)
    base_path = Path(source_model)
    if not base_path.exists():
        logger.error(f"Source model path does not exist: {base_path}")
        sys.exit(1)

    interim_dir   = Path(data_cfg["interim_dir"])
    processed_dir = Path(data_cfg["processed_dir"])
    output_dir    = interim_dir / "extended_base"
    spm_prefix    = interim_dir / "thai_spm"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Import ML libs ────────────────────────────────────────────────────────
    from transformers import AutoTokenizer, BertForMaskedLM

    # ── Load corpus texts ─────────────────────────────────────────────────────
    corpus_files = [
        processed_dir / "train.jsonl",
        processed_dir / "validation.jsonl",
        processed_dir / "test.jsonl",
    ]
    missing = [p for p in corpus_files if not p.exists()]
    if missing:
        logger.error(
            f"Corpus files not found: {missing}\n"
            "Run: python scripts/build_mlm_dataset.py first."
        )
        sys.exit(1)

    texts = collect_texts(corpus_files)

    # ── Train SentencePiece ───────────────────────────────────────────────────
    new_vocab = train_spm(texts, vocab_size=args.new_vocab_size, output_prefix=spm_prefix)

    # ── Load base tokenizer ───────────────────────────────────────────────────
    logger.info(f"Loading base tokenizer from: {base_path}")
    # use_fast=False avoids the fast-tokenizer conversion error for SapBERT
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), use_fast=False)
    existing_vocab = set(tokenizer.get_vocab().keys())
    logger.info(f"Existing vocab size: {len(existing_vocab):,}")

    # ── Find new tokens ───────────────────────────────────────────────────────
    new_tokens = find_new_tokens(existing_vocab, new_vocab)

    if not new_tokens:
        logger.warning("No new tokens found! Check corpus or --new-vocab-size.")
        sys.exit(0)

    # ── Load base model ───────────────────────────────────────────────────────
    logger.info(f"Loading BertForMaskedLM from: {base_path}")
    model = BertForMaskedLM.from_pretrained(str(base_path), ignore_mismatched_sizes=True)
    orig_vocab_size = len(tokenizer)

    # ── Add all individual Thai characters as fallbacks ───────────────────────
    for i in range(0x0E00, 0x0E80):
        char = chr(i)
        if char not in existing_vocab and char not in new_tokens:
            new_tokens.append(char)
        sub_char = f"##{char}"
        if sub_char not in existing_vocab and sub_char not in new_tokens:
            new_tokens.append(sub_char)
    logger.info(f"Total tokens after adding single Thai char fallbacks: {len(new_tokens):,}")

    # ── Capture subword decomposition with OLD tokenizer ─────────────────────
    # This MUST happen before the tokenizer is reloaded from the new vocab.txt.
    # The old tokenizer does NOT know the new Thai tokens, so it decomposes
    # them into smaller subwords that ARE in the original (trained) vocab.
    # If we wait until after the reload, the new tokenizer already knows the
    # token as a single piece → mean of one random vector = same random vector
    # (the circular initialization bug).
    logger.info("Computing subword decomposition with old tokenizer...")
    decomposition: dict[str, list[int]] = {}
    for token in new_tokens:
        # For ## continuation tokens, decompose the bare form
        bare = token[2:] if token.startswith("##") else token
        sub_ids = tokenizer(bare, add_special_tokens=False)["input_ids"]
        decomposition[token] = sub_ids
    logger.info(f"Decomposition captured for {len(decomposition):,} tokens")

    # ── Add new tokens directly to vocab.txt ──────────────────────────────────
    # We will write the extended vocab.txt directly to output_dir
    import shutil
    
    vocab = tokenizer.get_vocab()
    vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])
    
    vocab_file = output_dir / "vocab.txt"
    with open(vocab_file, "w", encoding="utf-8") as f:
        for tok, idx in vocab_sorted:
            f.write(tok + "\n")
        for tok in new_tokens:
            f.write(tok + "\n")
            
    # Copy and fix tokenizer config files from base
    import json
    for f_name in ["tokenizer_config.json", "special_tokens_map.json"]:
        src = base_path / f_name
        if src.exists():
            if f_name == "tokenizer_config.json":
                # Fix strip_accents for Thai! If it strips accents, Thai vowels/tones are destroyed.
                config_dict = json.loads(src.read_text(encoding="utf-8"))
                config_dict["strip_accents"] = False
                (output_dir / f_name).write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
            else:
                shutil.copy(src, output_dir / f_name)
            
    # Ensure no tokenizer.json exists from previous runs to force slow tokenizer
    if (output_dir / "tokenizer.json").exists():
        (output_dir / "tokenizer.json").unlink()
            
    # Reload tokenizer natively from the newly created vocab.txt
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(str(output_dir))
        
    logger.info(f"Added {len(new_tokens):,} new tokens directly to vocab.txt")
    logger.info(f"New vocab size: {len(tokenizer):,} (was {orig_vocab_size:,})")

    # ── Resize embedding matrix ───────────────────────────────────────────────
    model.resize_token_embeddings(len(tokenizer))
    logger.info("Embedding matrix resized")

    # ── Smart initialize new embeddings ───────────────────────────────────────
    smart_init_embeddings(model, tokenizer, new_tokens, decomposition)

    # ── Save Model ────────────────────────────────────────────────────────────
    # We only need to save the model, because the tokenizer files are already in output_dir
    model.save_pretrained(str(output_dir))
    
    logger.info(f"\n✅ Extended model + tokenizer saved → {output_dir}")

    # ── Update model_config.yaml to point to extended model ──────────────────
    logger.info("\n⚠️  NEXT STEP: Update configs/model_config.yaml")
    logger.info(f"   Change base_model_path to: {output_dir.resolve()}")

    # Quick UNK rate check after extension
    logger.info("\nQuick UNK rate check after extension (first 5 samples):")
    sample_texts = texts[:5]
    for t in sample_texts:
        tokens = tokenizer.tokenize(t)
        unk_count = tokens.count("[UNK]")
        logger.info(f"  Text: {t[:60]!r}")
        logger.info(f"  Tokens: {tokens[:12]}  (UNK: {unk_count}/{len(tokens)})")

    logger.info("\n✅ Done! Proceed to training:")
    logger.info("   python scripts/train_lora_mlm.py")


if __name__ == "__main__":
    main()
