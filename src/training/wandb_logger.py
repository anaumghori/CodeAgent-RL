import os
import wandb

from src.config.config import PipelineConfig
from src.helpers import log_event


class WandbLogger:
    """Thin wrapper around `wandb.init/log/finish` used by the orchestrator and evaluator."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.run = None


    def init(self, run_id: str | None = None) -> str:
        if self.cfg.credentials.wandb_api_key:
            os.environ["WANDB_API_KEY"] = self.cfg.credentials.wandb_api_key
        log_event(
            "wandb",
            f"Initializing W&B run for project={self.cfg.logging.wandb_project!r}, resume={run_id!r}.",
        )
        self.run = wandb.init(
            project=self.cfg.logging.wandb_project,
            id=run_id,
            resume="must" if run_id else None,
            config=self.cfg.to_dict(),
        )
        log_event("wandb", f"W&B run initialized with id={self.run.id}.")
        return self.run.id


    def log(self, metrics: dict, step: int | None = None) -> None:
        if self.run is None:
            return
        self.run.log(metrics, step=step)


    def finish(self) -> None:
        if self.run is not None:
            log_event("wandb", f"Finishing W&B run id={self.run.id}.")
            self.run.finish()
            self.run = None
