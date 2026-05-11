"""
Usage:
    modal run src/modal_app.py
    modal run src/modal_app.py --resume-from checkpoint-14
"""

import os
from pathlib import Path
import modal

from src.config.config import INFERENCE_GPU_ID

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "composer-rl"
GPU_SPEC = "H200:3"
VOLUME_NAME = "composer-rl-checkpoints"
TIMEOUT = 60 * 60 * 23  # one short of Modal's 24h hard cap; resume via --resume-from

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .env({"DEBIAN_FRONTEND": "noninteractive", "CUDA_HOME": "/usr/local/cuda"})
    .apt_install(
        "git", "build-essential", "wget", "ca-certificates", "curl",
    )
    .uv_pip_install(
        "accelerate>=1.13.0",
        "deepspeed>=0.18.0",
        "vllm>=0.12.0",
        "huggingface-hub>=0.30.0",
        "datasets>=3.5.0",
        "wandb>=0.18.0",
        "python-dotenv>=1.2.2",
        "torch>=2.5.0",
        "torchvision>=0.20.0",
        "transformers>=4.57.0",
        "pyyaml>=6.0",
        "numpy>=1.26.0",
    )
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/root/src", copy=True)
    .add_local_dir(str(PROJECT_ROOT / "resources"), remote_path="/root/resources", copy=True)
)

app = modal.App(APP_NAME, image=image)
checkpoint_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
dotenv_secret = modal.Secret.from_dotenv(PROJECT_ROOT)

@app.function(
    gpu=GPU_SPEC,
    timeout=TIMEOUT,
    volumes={"/root/checkpoints": checkpoint_volume},
    secrets=[
        dotenv_secret,
    ],
    cpu=16.0,
    memory=512 * 1024,
)
def train(resume_from: str | None = None) -> None:
    """Run the full RL training pipeline inside a Modal container."""
    os.chdir("/root")
    from src.helpers import log_event
    from src.config.config import load_config

    log_event("modal_app", f"train() starting with resume_from={resume_from!r}")
    cfg = load_config()
    cfg.checkpoint.checkpoint_dir = "/root/checkpoints"
    os.environ["COMPOSER_RL_MODAL_APP_NAME"] = APP_NAME
    os.environ["CUDA_VISIBLE_DEVICES"] = str(INFERENCE_GPU_ID)
    log_event(
        "modal_app",
        f"Configuration loaded; binding orchestrator to inference GPU {INFERENCE_GPU_ID}.",
    )
    log_event(
        "modal_app",
        f"GPU layout: training_gpus=2 mapped to ranks 0..1, "
        f"inference_gpu={INFERENCE_GPU_ID}, parent CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}.",
    )

    from src.orchestrator import Orchestrator

    orch = Orchestrator(cfg, resume_from=resume_from)
    log_event("modal_app", "Orchestrator constructed; entering setup().")
    orch.setup()
    log_event("modal_app", "Setup complete; entering run loop.")
    orch.run()

@app.local_entrypoint()
def main(resume_from: str | None = None) -> None:
    train.remote(resume_from=resume_from)
