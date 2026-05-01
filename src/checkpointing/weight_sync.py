import os
import torch.distributed as dist

from src.config.config import PipelineConfig


def init_weight_transfer_group(cfg: PipelineConfig):
    """
    Initialise the trainer side of the NCCL weight transfer process group.
    The trainer is rank 0 in this transfer group, the single inference worker
    is rank 1, world size is 2.

    This must only be called from the rank-0 trainer subprocess.
    """
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        raise RuntimeError(
            f"init_weight_transfer_group must run on trainer rank 0; current RANK={rank}."
        )

    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

    return NCCLWeightTransferEngine.trainer_init(
        dict(
            master_address=cfg.infra.weight_sync_master_address,
            master_port=cfg.infra.weight_sync_master_port,
            world_size=2,
        )
    )


def init_inference_weight_transfer(cfg: PipelineConfig, llm) -> None:
    """
    Initialise the inference side of the NCCL weight transfer group on the
    given vLLM `LLM` instance. Called from the orchestrator parent process.
    """
    from vllm.distributed.weight_transfer.base import WeightTransferInitRequest

    llm.init_weight_transfer_engine(
        WeightTransferInitRequest(
            init_info=dict(
                master_address=cfg.infra.weight_sync_master_address,
                master_port=cfg.infra.weight_sync_master_port,
                rank_offset=1,
                world_size=2,
            )
        )
    )


class WeightSync:
    """
    Drive the NCCL-based weight broadcast from the DeepSpeed training engine
    to the running vLLM inference server which lives in the orchestrator
    parent process. This object is constructed inside each trainer subprocess
    and is collective across all trainer ranks:

      * All ranks call `synchronize` together; ZeRO-3 parameter gathers
        require participation from every rank.
      * Only rank 0 issues the IPC pause/update/resume calls to the parent
        and only rank 0 calls `trainer_send_weights` over NCCL.
    """


    def __init__(
        self,
        cfg: PipelineConfig,
        engine,
        transfer_group,
        weight_metadata: tuple[list[str], list[str], list[tuple[int, ...]]],
        vllm_ipc=None,
    ) -> None:
        self.cfg = cfg
        self.engine = engine
        self.group = transfer_group
        self.weight_names, self.weight_dtype_names, self.weight_shapes = weight_metadata
        self.vllm_ipc = vllm_ipc
        self.rank = int(os.environ.get("RANK", "0"))
        self._assert_parameter_order_matches_metadata()


    def _named_parameters(self):
        """Iterate `(name, param)` from the underlying module (DS engine or raw module)."""
        module = self.engine.module if hasattr(self.engine, "module") else self.engine
        return list(module.named_parameters())


    def _assert_parameter_order_matches_metadata(self) -> None:
        live_names = [name for name, _ in self._named_parameters()]
        if live_names != self.weight_names:
            raise RuntimeError(
                "DeepSpeed parameter order no longer matches the pre-wrap weight "
                "metadata required for vLLM weight transfer."
            )


    def _gathered_named_parameters(self):
        """
        Yield `(name, full_param_tensor)` tuples, all-gathering ZeRO-3
        sharded parameters across the trainer ranks. Must be called
        collectively on every rank because `GatheredParameters` is a
        collective context manager.
        """
        import deepspeed

        for name, param in self._named_parameters():
            if hasattr(param, "ds_id"):
                with deepspeed.zero.GatheredParameters([param], modifier_rank=None, enabled=True):
                    yield name, param.data
            else:
                yield name, param.data


    def _ipc_request(self, payload: tuple) -> None:
        if self.vllm_ipc is None:
            raise RuntimeError("WeightSync rank 0 needs an IPC channel to the orchestrator parent.")
        self.vllm_ipc.send(payload)
        ack = self.vllm_ipc.recv()
        if not (isinstance(ack, tuple) and ack and ack[0] == "ack"):
            raise RuntimeError(f"vLLM IPC request {payload[0]} failed: {ack!r}")


    def synchronize(self) -> None:
        """
        Collectively gather ZeRO-3 parameters and broadcast them to vLLM.
        Rank 0 also drives the parent's vLLM pause/update/resume sequence
        via IPC. Every rank must call this in lockstep.
        """
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLTrainerSendWeightsArgs,
            NCCLWeightTransferEngine,
        )

        if self.rank == 0:
            self._ipc_request((
                "vllm_pause_and_update",
                self.weight_names,
                self.weight_dtype_names,
                self.weight_shapes,
            ))

        dist.barrier()

        if self.rank == 0:
            trainer_args = NCCLTrainerSendWeightsArgs(group=self.group, packed=True)
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=self._gathered_named_parameters(),
                trainer_args=trainer_args,
            )
        else:
            # Participate in the GatheredParameters collectives even though
            # this rank does not call trainer_send_weights.
            for _ in self._gathered_named_parameters():
                pass

        dist.barrier()

        if self.rank == 0:
            self._ipc_request(("vllm_join_and_resume",))
