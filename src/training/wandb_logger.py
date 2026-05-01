import os
import wandb
from src.config.config import PipelineConfig

class WandbLogger:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.run = None


    def init(self, run_id: str | None = None) -> str:
        if self.cfg.credentials.wandb_api_key:
            os.environ["WANDB_API_KEY"] = self.cfg.credentials.wandb_api_key
        self.run = wandb.init(
            project=self.cfg.logging.wandb_project,
            entity=self.cfg.logging.wandb_entity,
            id=run_id,
            resume="must" if run_id else None,
            config=self.cfg.to_dict(),
        )
        return self.run.id

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self.run is None:
            return
        self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
