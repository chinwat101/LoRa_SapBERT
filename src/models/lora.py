"""
src/models/lora.py
Apply LoRA adapters to a BertForMaskedLM model via Hugging Face PEFT.
"""
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_bert_linear_modules(model) -> list[str]:
    """
    Dynamically find all Linear layer names in the model.
    Used to help select LoRA target_modules.

    Returns:
        Sorted list of module names that are nn.Linear.
    """
    import torch.nn as nn

    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_names.append(name)
    return sorted(linear_names)


def apply_lora(model, lora_cfg: dict):
    """
    Wrap a BertForMaskedLM with LoRA adapters using PEFT.

    Args:
        model: BertForMaskedLM instance.
        lora_cfg: Dict from lora_config.yaml with keys:
                  r, lora_alpha, lora_dropout, bias, target_modules.

    Returns:
        PEFT-wrapped model with LoRA adapters.
    """
    from peft import LoraConfig, get_peft_model, TaskType

    target_modules = lora_cfg.get("target_modules", ["query", "key", "value"])

    # Validate that target modules actually exist in the model
    linear_modules = get_bert_linear_modules(model)
    valid_targets = []
    for tm in target_modules:
        matches = [m for m in linear_modules if m.endswith(tm) or tm in m]
        if matches:
            valid_targets.append(tm)
        else:
            logger.warning(
                f"LoRA target '{tm}' not found in model layers. "
                f"Run scripts/inspect_model_modules.py to see available layers."
            )

    if not valid_targets:
        raise ValueError(
            "None of the LoRA target_modules were found in the model. "
            "Please run scripts/inspect_model_modules.py and update lora_config.yaml."
        )

    logger.info(f"Applying LoRA to modules: {valid_targets}")

    # Read task_type from config (default: FEATURE_EXTRACTION for MLM)
    task_type_str = lora_cfg.get("task_type", "FEATURE_EXTRACTION")
    try:
        task_type = getattr(TaskType, task_type_str)
    except AttributeError:
        valid = [t.name for t in TaskType]
        raise ValueError(
            f"Invalid task_type '{task_type_str}' in lora_config.yaml. "
            f"Valid options: {valid}"
        )
    logger.info(f"LoRA task_type: {task_type_str}")

    config = LoraConfig(
        task_type=task_type,
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        target_modules=valid_targets,
        modules_to_save=lora_cfg.get("modules_to_save", None),
        inference_mode=lora_cfg.get("inference_mode", False),
    )

    model = get_peft_model(model, config)

    trainable, total = _count_params(model)
    logger.info(
        f"LoRA applied | Trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    model.print_trainable_parameters()

    return model


def _count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_lora_adapter(model, output_dir: str | Path) -> None:
    """Save only LoRA adapter weights to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    logger.info(f"LoRA adapter saved to: {output_dir}")


def load_lora_adapter(base_model, adapter_dir: str | Path):
    """
    Load saved LoRA adapter weights onto a base model.

    Args:
        base_model: BertForMaskedLM instance.
        adapter_dir: Path to saved adapter directory.

    Returns:
        PEFT model with loaded adapter weights.
    """
    from peft import PeftModel

    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    logger.info(f"LoRA adapter loaded from: {adapter_dir}")
    return model
