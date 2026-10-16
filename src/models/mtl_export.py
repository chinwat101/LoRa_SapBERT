"""Export the NER portion of a joint MTL model as a Hugging Face checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from transformers import AutoModelForTokenClassification, PreTrainedTokenizerBase


def export_ner_standalone(
    mtl_model,
    tokenizer: PreTrainedTokenizerBase,
    base_model_path: str,
    ner_labels: Sequence[str],
    output_dir: str | Path,
    source_mtl_dir: str | Path | None = None,
) -> Path:
    """Save the MTL encoder and NER head as a standalone token-classification model.

    The relation head remains in the original MTL checkpoint and is not part of
    this NER-only export.
    """
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    label2id = {label: idx for idx, label in enumerate(ner_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    standalone = AutoModelForTokenClassification.from_pretrained(
        base_model_path,
        num_labels=len(ner_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Transfer the fine-tuned shared encoder and NER classification head.
    # Token-classification backbones may omit the unused BERT pooler, while the
    # MTL encoder was created through AutoModel and can contain it.
    encoder_load = standalone.base_model.load_state_dict(
        mtl_model.encoder.state_dict(), strict=False
    )
    unexpected = [key for key in encoder_load.unexpected_keys if not key.startswith("pooler.")]
    missing = [key for key in encoder_load.missing_keys if not key.startswith("pooler.")]
    if unexpected or missing:
        raise RuntimeError(
            "Encoder weights did not map cleanly to the standalone NER model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    classifier = getattr(standalone, "classifier", None)
    if classifier is None:
        raise ValueError(
            f"{standalone.__class__.__name__} has no 'classifier' module; "
            "cannot export this architecture as token classification."
        )
    classifier.load_state_dict(mtl_model.ner_head.state_dict())
    standalone.eval()

    standalone.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    with (target / "label_schema.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {str(idx): label for idx, label in id2label.items()},
                "source_mtl_dir": str(source_mtl_dir) if source_mtl_dir else None,
                "export_scope": "NER only; relation head remains in the MTL checkpoint",
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )

    return target
