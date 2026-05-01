from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import os
import yaml

from dotenv import load_dotenv


@dataclass
class ModelConfig:
    """
    Hyperparameters describing the base language model used by the pipeline.
    """
    model_name: str = "NousResearch/Hermes-4-14B"
    dtype: str = "bfloat16"
    max_position_embeddings: int = 40960
    vocab_size: int = 151936


@dataclass
class TrainingConfig:
    """
    Optimizer, schedule, batching, and precision parameters for the training loop.
    """
    total_steps: int = 2000
    warmup_steps: int = 100
    learning_rate: float = 1e-6
    lr_schedule: str = "cosine"
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 16
    micro_batch_size: int = 1
    mixed_precision: str = "bf16"
    activation_checkpointing: bool = True


@dataclass
class SequenceConfig:
    """Maximum lengths and self-summarization triggers."""
    max_training_seq_length: int = 4096
    max_rollout_length: int = 32768
    max_generation_tokens: int = 8192
    summary_soft_trigger_tokens: int = 10000
    summary_hard_trigger_tokens: int = 12000
    summary_context_turns: int = 2


@dataclass
class GRPOConfig:
    """RL algorithm parameters governing GRPO/Dr. GRPO loss and rollout grouping."""
    group_size: int = 4
    ppo_clip_epsilon: float = 0.2
    kl_coefficient: float = 0.02
    kl_estimator: str = "k1"
    normalize_advantages_by_std: bool = False
    overlong_masking: bool = False
    max_policy_staleness: int = 2


@dataclass
class SamplingConfig:
    """Decoding parameters for both training rollouts and evaluation."""
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    eval_temperature: float = 0.0


@dataclass
class RewardConfig:
    """Weights and parameters of every reward component."""
    correctness_weight: float = 1.0
    aux_syntax_weight: float = 0.05
    aux_no_todo_weight: float = 0.05
    aux_minimal_diff_weight: float = 0.05
    aux_tool_hygiene_weight: float = 0.05
    aux_test_discipline_weight: float = 0.05
    length_penalty_k: float = 0.15
    length_penalty_q: float = 1.5
    length_penalty_lambda: float = 0.05
    test_timeout_seconds: int = 60


@dataclass
class DataConfig:
    """Dataset identifiers, mix ratios, and curriculum-related defaults."""
    swe_rebench_v2_id: str = "nebius/SWE-rebench-V2"
    swe_rebench_v2_prs_id: str = "nebius/SWE-rebench-V2-PRs"
    codecontests_o_id: str = "caijanfeng/CodeContests-O"
    mix_swe_v2_weight: float = 0.55
    mix_swe_prs_weight: float = 0.15
    mix_codecontests_weight: float = 0.30
    hard_task_upsample_factor: float = 2.0
    eval_subset_size: int = 50
    eval_seed: int = 42


@dataclass
class CheckpointConfig:
    """Recovery checkpoint cadence and Hugging Face storage location."""
    recovery_checkpoint_interval: int = 100
    hf_checkpoint_repo: str = ""
    weight_sync_interval: int = 1
    checkpoint_dir: str = "checkpoints"


@dataclass
class EvalConfig:
    """Intermediate-evaluation cadence and held-out subset sizes."""
    eval_interval: int = 100
    eval_codecontests_count: int = 50
    eval_swe_count: int = 50


@dataclass
class LoggingConfig:
    """Weights & Biases tracking configuration."""
    wandb_project: str = "composer-rl"
    wandb_entity: str = ""
    log_interval: int = 1


@dataclass
class InfraConfig:
    """GPU layout, vLLM behaviour, and rollout-buffer/environment-pool sizes."""
    num_training_gpus: int = 2
    inference_gpu_id: int = 2
    num_environment_workers: int = 16
    vllm_gpu_memory_utilization: float = 0.90
    vllm_max_model_len: int = 32768
    vllm_enable_prefix_caching: bool = True
    rollout_buffer_max_groups: int = 50
    env_prewarming_pool_size: int = 8
    weight_sync_master_address: str = "127.0.0.1"
    weight_sync_master_port: int = 29600
    trainer_dist_master_port: int = 29500


@dataclass
class CurriculumConfig:
    """Stage boundaries and per-stage dataset mixes."""
    stage1_end_step: int = 200
    stage2_end_step: int = 900
    stage1_mix: dict = field(default_factory=lambda: {"codecontests": 0.5, "swe_v2": 0.5})
    stage2_mix: dict = field(
        default_factory=lambda: {"swe_v2": 0.55, "swe_prs": 0.15, "codecontests": 0.30}
    )
    seq_length_extension_step: int = 500
    extended_max_training_seq_length: int = 8192
    hard_task_success_threshold: float = 0.5


@dataclass
class CredentialsConfig:
    """Container holding credentials loaded from a `.env` file at process start."""
    hf_token: str = ""
    wandb_api_key: str = ""
    modal_token_id: str = ""
    modal_token_secret: str = ""
    hf_username: str = ""
    wandb_username: str = ""


@dataclass
class PipelineConfig:
    """
    Top-level configuration object that aggregates every subsystem's settings.
    Acts as the single source of truth for all tunable parameters in the codebase.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)


    @property
    def effective_batch_size(self) -> int:
        """Effective optimizer-step batch size: micro × accumulation × DP world size."""
        return (
            self.training.micro_batch_size
            * self.training.gradient_accumulation_steps
            * self.infra.num_training_gpus
        )


    def to_dict(self) -> dict:
        return asdict(self)


def _apply_overrides(cfg: PipelineConfig, overrides: dict[str, Any]) -> None:
    """Recursively apply nested override dict onto the dataclass-based config."""
    for section, values in overrides.items():
        if not hasattr(cfg, section):
            raise KeyError(f"Unknown config section: {section}")
        sub = getattr(cfg, section)
        if not isinstance(values, dict):
            raise TypeError(f"Section {section} requires a dict of overrides")
        for k, v in values.items():
            if not hasattr(sub, k):
                raise KeyError(f"Unknown key {k} in section {section}")
            setattr(sub, k, v)


def load_config(yaml_path: str | Path | None = None) -> PipelineConfig:
    """
    Build the pipeline config. Loads `.env` for credentials, then optionally
    overlays a YAML override file.

    :param yaml_path: Optional path to a YAML override file.
    :returns: Fully populated `PipelineConfig`.
    """
    load_dotenv()
    cfg = PipelineConfig()
    cfg.credentials.hf_token = os.environ.get("HF_TOKEN", "")
    cfg.credentials.wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    cfg.credentials.modal_token_id = os.environ.get("MODAL_TOKEN_ID", "")
    cfg.credentials.modal_token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")
    cfg.credentials.hf_username = os.environ.get("HF_USERNAME", "")
    cfg.credentials.wandb_username = os.environ.get("WANDB_USERNAME", "")
    cfg.logging.wandb_entity = os.environ.get("WANDB_USERNAME", "")
    if cfg.credentials.hf_username:
        cfg.checkpoint.hf_checkpoint_repo = (
            f"{cfg.credentials.hf_username}/composer-rl-checkpoints"
        )

    if yaml_path is not None:
        with open(yaml_path, "r") as f:
            overrides = yaml.safe_load(f) or {}
        _apply_overrides(cfg, overrides)

    return cfg
