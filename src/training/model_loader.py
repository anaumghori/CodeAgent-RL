import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.config import PipelineConfig
from src.helpers import log_event

def _dtype_from_string(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]

def _log_model_summary(model, cfg: PipelineConfig) -> None:
    """Print a concise model summary for startup visibility."""
    named_params = list(model.named_parameters())
    total_params = sum(param.numel() for _, param in named_params)
    trainable_params = sum(param.numel() for _, param in named_params if param.requires_grad)
    bytes_per_param = torch.tensor([], dtype=_dtype_from_string(cfg.model.dtype)).element_size()
    approx_model_size_gb = (total_params * bytes_per_param) / (1024 ** 3)
    log_event(
        "model_loader",
        f"Model summary: name={cfg.model.model_name}, dtype={cfg.model.dtype}, "
        f"total_params={total_params}, trainable_params={trainable_params}, "
        f"approx_parameter_memory_gb={approx_model_size_gb:.2f}.",
    )

def load_model_and_tokenizer(cfg: PipelineConfig):
    log_event("model_loader", f"Loading tokenizer for {cfg.model.model_name}.")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name, token=cfg.credentials.hf_token or None,
    )
    log_event("model_loader", f"Loading model weights for {cfg.model.model_name}.")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        dtype=_dtype_from_string(cfg.model.dtype),
        token=cfg.credentials.hf_token or None,
    )
    if cfg.training.activation_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    _log_model_summary(model, cfg)
    log_event("model_loader", "Model and tokenizer loaded.")
    return model, tokenizer

def load_tokenizer_only(cfg: PipelineConfig):
    """Load just the Hermes 4 tokenizer (used by the orchestrator parent process)."""
    log_event("model_loader", f"Loading tokenizer-only path for {cfg.model.model_name}.")
    return AutoTokenizer.from_pretrained(
        cfg.model.model_name, token=cfg.credentials.hf_token or None,
    )
