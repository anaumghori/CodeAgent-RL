import json
import os
from pathlib import Path
import torch.distributed as dist
from huggingface_hub import HfApi, snapshot_download

from src.config.config import PipelineConfig


CLIENT_STATE_FILE = "client_state.json"


class RecoveryCheckpointer:
    """
    Save / load full training checkpoints via DeepSpeed and back them up to
    a Hugging Face repository for durable recovery.

    Each call to `save` and `load` is collective across the trainer DP
    ranks: DeepSpeed's `save_checkpoint` / `load_checkpoint` partition
    the ZeRO-3 state across ranks. Only rank 0 talks to the Hugging Face Hub.
    """


    def __init__(self, cfg: PipelineConfig, engine) -> None:
        self.cfg = cfg
        self.engine = engine
        self.rank = int(os.environ.get("RANK", "0"))
        self.api = HfApi(token=cfg.credentials.hf_token or None)
        self.local_root = Path(cfg.checkpoint.checkpoint_dir) / "recovery"
        self.local_root.mkdir(parents=True, exist_ok=True)


    def save(
        self,
        checkpoint_number: int,
        client_state: dict,
        push_to_hub: bool = True,
    ) -> Path:
        """
        Save a recovery checkpoint locally (collective) and push to HF (rank 0 only).
        """
        tag = f"checkpoint-{checkpoint_number}"
        save_dir = self.local_root / tag
        if self.rank == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        self.engine.save_checkpoint(
            save_dir=str(self.local_root), tag=tag, client_state=client_state,
        )
        if self.rank == 0:
            with open(save_dir / CLIENT_STATE_FILE, "w") as f:
                json.dump(client_state, f, indent=2)
            if push_to_hub:
                self.api.create_repo(self.cfg.checkpoint.hf_checkpoint_repo,
                                     repo_type="model", exist_ok=True, private=True)
                self.api.upload_folder(
                    folder_path=str(save_dir),
                    path_in_repo=tag,
                    repo_id=self.cfg.checkpoint.hf_checkpoint_repo,
                    repo_type="model",
                )
        if dist.is_initialized():
            dist.barrier()
        return save_dir


    def load(self, checkpoint_tag: str) -> dict:
        """
        Download (rank 0) and restore (collective) the named checkpoint.

        :returns: The restored `client_state` dictionary.
        """
        local_dir = self.local_root / checkpoint_tag
        if self.rank == 0 and not local_dir.exists():
            snapshot_download(
                repo_id=self.cfg.checkpoint.hf_checkpoint_repo,
                allow_patterns=[f"{checkpoint_tag}/*"],
                local_dir=str(self.local_root.parent),
                token=self.cfg.credentials.hf_token or None,
            )
        if dist.is_initialized():
            dist.barrier()
        load_path, client_state = self.engine.load_checkpoint(
            load_dir=str(self.local_root), tag=checkpoint_tag,
        )
        if not client_state and (local_dir / CLIENT_STATE_FILE).exists():
            with open(local_dir / CLIENT_STATE_FILE, "r") as f:
                client_state = json.load(f)
        if dist.is_initialized():
            dist.barrier()
        return client_state or {}
