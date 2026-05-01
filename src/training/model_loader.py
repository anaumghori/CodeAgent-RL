import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.config import PipelineConfig


def _dtype_from_string(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]

def load_model_and_tokenizer(cfg: PipelineConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name, token=cfg.credentials.hf_token or None,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        dtype=_dtype_from_string(cfg.model.dtype),
        token=cfg.credentials.hf_token or None,
    )
    if cfg.training.activation_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tokenizer

def load_tokenizer_only(cfg: PipelineConfig):
    """Load just the Hermes 4 tokenizer (used by the orchestrator parent process)."""
    return AutoTokenizer.from_pretrained(
        cfg.model.model_name, token=cfg.credentials.hf_token or None,
    )
