import os
from multiprocessing.connection import Client, Connection
import torch.distributed as dist

from src.checkpointing.recovery import RecoveryCheckpointer
from src.checkpointing.weight_sync import WeightSync, init_weight_transfer_group
from src.config.config import load_config
from src.inference.rollout import RolloutGroup
from src.helpers import format_exception, log_event
from src.training.launcher import IPC_SOCKET_ENV, TRAINER_RANK_ENV
from src.training.model_loader import load_model_and_tokenizer
from src.training.trainer import Trainer

def _broadcast_command(rank: int, command: tuple | None) -> tuple:
    """
    Rank 0 broadcasts only the command kind (and small scalar args) to other
    ranks; the bulky payloads (rollout groups, client state) stay on rank 0
    because every rank reconstructs the heavy data through later collective
    ops (microbatch broadcast, ZeRO-3 gather, DeepSpeed checkpoint load).
    """
    if rank == 0:
        kind = command[0]
        if kind == "save_checkpoint":
            light = (kind, command[1])  # checkpoint_number; client_state stays on rank 0
        elif kind == "load_checkpoint":
            light = (kind, command[1])  # checkpoint_tag (small string)
        else:
            light = (kind,)
        payload = [light]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _run() -> None:
    """Trainer subprocess main entry. Differentiates rank 0 (IPC) from rank > 0."""
    rank = int(os.environ[TRAINER_RANK_ENV])
    log_event("trainer_subprocess", f"Rank {rank} subprocess starting.")
    ipc: Connection | None = None
    try:
        cfg = load_config()
        model, tokenizer = load_model_and_tokenizer(cfg)
        trainer = Trainer(cfg=cfg, model=model, tokenizer=tokenizer)

        transfer_group = None
        if rank == 0:
            log_event("trainer_subprocess", "Rank 0 connecting to orchestrator IPC socket.")
            ipc = Client(address=os.environ[IPC_SOCKET_ENV], family="AF_UNIX")
            ipc.send(("ds_init_done",))
            # Wait for the parent to acknowledge; it then concurrently issues
            # init_inference_weight_transfer while we call init_weight_transfer_group.
            ack = ipc.recv()
            if ack != ("init_nccl",):
                raise RuntimeError(f"Trainer rank 0 expected init_nccl from parent, got {ack!r}")
            log_event("trainer_subprocess", "Rank 0 initializing NCCL weight transfer group.")
            transfer_group = init_weight_transfer_group(cfg)
            ipc.send(("nccl_done",))

        weight_sync = WeightSync(
            cfg,
            trainer.engine,
            transfer_group,
            trainer.weight_transfer_metadata,
            vllm_ipc=ipc,
        )
        recovery = RecoveryCheckpointer(cfg, trainer.engine)

        while True:
            full_command = ipc.recv() if rank == 0 else None
            broadcast_command = _broadcast_command(rank, full_command)
            # Rank 0 keeps the full command (with bulky args); other ranks use
            # the lightweight broadcast version.
            command = full_command if rank == 0 else broadcast_command
            kind = command[0]
            if kind == "shutdown":
                break

            if kind == "step":
                groups: list[RolloutGroup] | None = command[1] if rank == 0 else None
                metrics = trainer.execute_step(groups)
                if (trainer.step % cfg.checkpoint.weight_sync_interval) == 0:
                    if rank == 0:
                        log_event(
                            "trainer_subprocess",
                            f"Starting weight sync at step {trainer.step}.",
                        )
                    weight_sync.synchronize()
                    trainer.policy_version += 1
                if rank == 0:
                    ipc.send((
                        "step_done",
                        trainer.step,
                        trainer.policy_version,
                        metrics,
                    ))
                continue

            if kind == "save_checkpoint":
                checkpoint_number = command[1]
                client_state = command[2] if rank == 0 else {}
                if rank == 0:
                    log_event("trainer_subprocess", f"Saving checkpoint {checkpoint_number}.")
                save_path = recovery.save(checkpoint_number, client_state, push_to_hub=True)
                if rank == 0:
                    ipc.send(("save_checkpoint_done", str(save_path)))
                continue

            if kind == "load_checkpoint":
                checkpoint_tag = command[1]
                if rank == 0:
                    log_event("trainer_subprocess", f"Loading checkpoint {checkpoint_tag}.")
                client_state = recovery.load(checkpoint_tag)
                if rank == 0:
                    ipc.send(("load_checkpoint_done", client_state))
                continue

            raise RuntimeError(f"Trainer subprocess received unknown command: {kind!r}")
    except BaseException as exc:
        log_event(
            "trainer_subprocess",
            f"Rank {rank} failed with exception:\n{format_exception(exc)}",
        )
        raise
    finally:
        if ipc is not None:
            ipc.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        log_event("trainer_subprocess", f"Rank {rank} subprocess exiting.")


if __name__ == "__main__":
    _run()
