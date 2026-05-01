from src.config.config import PipelineConfig


def build_deepspeed_config(cfg: PipelineConfig) -> dict:
    """
    Construct the DeepSpeed ZeRO-3 JSON configuration dictionary from the
    centralized `PipelineConfig`. The returned dict can be passed directly
    to `deepspeed.initialize(config=...)`.

    :param cfg: The pipeline configuration.
    :returns: DeepSpeed JSON config dict.
    """
    return {
        "bf16": {"enabled": cfg.training.mixed_precision == "bf16"},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": int(5e7),
            "stage3_prefetch_bucket_size": int(5e7),
            "stage3_param_persistence_threshold": int(1e5),
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": cfg.training.learning_rate,
                "betas": [cfg.training.adam_beta1, cfg.training.adam_beta2],
                "weight_decay": cfg.training.weight_decay,
            },
        },
        "gradient_clipping": cfg.training.max_grad_norm,
        "train_micro_batch_size_per_gpu": cfg.training.micro_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "activation_checkpointing": {
            "partition_activations": cfg.training.activation_checkpointing,
            "contiguous_memory_optimization": cfg.training.activation_checkpointing,
            "cpu_checkpointing": False,
        },
        "wall_clock_breakdown": False,
    }
