import os
import subprocess
import sys
import tempfile
from multiprocessing.connection import Listener, Connection
from pathlib import Path

from src.config.config import PipelineConfig


IPC_SOCKET_ENV = "COMPOSER_RL_IPC_SOCKET"
TRAINER_RANK_ENV = "COMPOSER_RL_TRAINER_RANK"


class TrainerLauncher:
    """
    Spawn the DeepSpeed trainer worker subprocesses (one per rank). Each rank gets 
    its own `CUDA_VISIBLE_DEVICES` mapping so that rank N sees a single device 
    which is globally GPU N. The orchestrator parent must already have set 
    `CUDA_VISIBLE_DEVICES=2` before any CUDA import so vLLM ends up on global GPU 2
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.world_size = cfg.infra.num_training_gpus
        socket_dir = Path(tempfile.mkdtemp(prefix="composer-rl-ipc-"))
        self.socket_path = str(socket_dir / "trainer.sock")
        self.listener: Listener | None = None
        self.connection: Connection | None = None
        self.processes: list[subprocess.Popen] = []


    def _build_env(self, rank: int) -> dict:
        """Compose the per-rank environment variables required by DeepSpeed + the IPC client."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = "0"  # one CUDA device visible per process
        env["WORLD_SIZE"] = str(self.world_size)
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(self.cfg.infra.trainer_dist_master_port)
        env[TRAINER_RANK_ENV] = str(rank)
        env[IPC_SOCKET_ENV] = self.socket_path
        return env


    def start(self) -> Connection:
        """
        Spawn both trainer subprocesses, then accept the IPC connection from
        rank 0 on the Unix-domain socket. Returns the live `Connection` to
        rank 0; the caller must close it on shutdown.
        """
        self.listener = Listener(address=self.socket_path, family="AF_UNIX")
        for rank in range(self.world_size):
            proc = subprocess.Popen(
                [sys.executable, "-m", "src.training.trainer_subprocess"],
                env=self._build_env(rank),
            )
            self.processes.append(proc)
        # Only rank 0 connects to the IPC listener; other ranks operate purely
        # via torch.distributed collectives.
        self.connection = self.listener.accept()
        return self.connection


    def crashed_ranks(self) -> list[int]:
        """Return the ranks whose subprocess has already exited (with any code)."""
        return [r for r, proc in enumerate(self.processes) if proc.poll() is not None]


    def shutdown(self) -> None:
        """Tear down the IPC listener and reap the trainer subprocesses."""
        if self.connection is not None:
            try:
                self.connection.close()
            except OSError:
                pass
            self.connection = None
        if self.listener is not None:
            try:
                self.listener.close()
            except OSError:
                pass
            self.listener = None
        for proc in self.processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in self.processes:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.processes = []
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
