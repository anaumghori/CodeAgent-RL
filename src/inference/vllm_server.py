import threading

from src.config.config import PipelineConfig
from src.helpers import gpu_memory_snapshot, log_event


class VLLMServer:
    """
    Lifecycle wrapper around the offline vLLM `LLM` class. Construction is
    unchanged: in the orchestrator parent `CUDA_VISIBLE_DEVICES` is set to
    only the inference GPU, so the underlying `cuda:0` of the vLLM engine
    is the global inference GPU.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._llm = None
        self._lock = threading.Lock()
        self._update_thread: threading.Thread | None = None
        self._update_error: list[BaseException] = []


    def start(self) -> None:
        from vllm import LLM
        from vllm.config import WeightTransferConfig

        log_event(
            "vllm",
            f"Starting vLLM for model={self.cfg.model.model_name}, dtype={self.cfg.model.dtype}, "
            f"max_model_len={self.cfg.infra.vllm_max_model_len}, "
            f"gpu_memory_utilization={self.cfg.infra.vllm_gpu_memory_utilization:.2f}.",
        )
        self._llm = LLM(
            model=self.cfg.model.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=self.cfg.infra.vllm_gpu_memory_utilization,
            max_model_len=self.cfg.infra.vllm_max_model_len,
            dtype=self.cfg.model.dtype,
            enable_prefix_caching=self.cfg.infra.vllm_enable_prefix_caching,
            enable_sleep_mode=True,
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
        )
        log_event("vllm", "vLLM server started.")
        memory = gpu_memory_snapshot(0)
        log_event(
            "vllm",
            f"Startup GPU memory: allocated={memory['allocated_gb']:.2f} GiB, "
            f"reserved={memory['reserved_gb']:.2f} GiB.",
        )


    @property
    def llm(self):
        if self._llm is None:
            raise RuntimeError("vLLM engine not started — call start() first.")
        return self._llm


    def pause_for_weight_sync(self) -> None:
        """
        Pause offline inference scheduling and invalidate reusable KV prefix
        state before a weight broadcast.
        """
        with self._lock:
            log_event("vllm", "Pausing vLLM for weight sync.")
            self.llm.sleep(level=0, mode="keep")
            self.llm.reset_prefix_cache(reset_running_requests=True)


    def resume_after_weight_sync(self) -> None:
        with self._lock:
            log_event("vllm", "Resuming vLLM after weight sync.")
            self.llm.wake_up(tags=["scheduling"])


    def begin_weight_update(
        self,
        names: list[str],
        dtype_names: list[str],
        shapes: list[tuple[int, ...]],
    ) -> None:
        """
        Kick off the vLLM-side `update_weights` call in a background thread
        so the trainer's NCCL `trainer_send_weights` (the sender side) can
        proceed concurrently. The thread is joined by `end_weight_update`.
        """
        from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest

        with self._lock:
            if self._update_thread is not None:
                raise RuntimeError("vLLM weight update already in progress.")
            self._update_error = []
            log_event("vllm", f"Beginning weight update for {len(names)} tensors.")
            request = WeightTransferUpdateRequest(
                update_info=dict(
                    names=names,
                    dtype_names=dtype_names,
                    shapes=shapes,
                    packed=True,
                )
            )

            def _runner() -> None:
                try:
                    self.llm.update_weights(request)
                except BaseException as exc:
                    self._update_error.append(exc)

            self._update_thread = threading.Thread(target=_runner, daemon=True)
            self._update_thread.start()


    def end_weight_update(self) -> None:
        """Join the background update thread and re-raise any error from it."""
        with self._lock:
            thread = self._update_thread
            self._update_thread = None
        if thread is None:
            raise RuntimeError("end_weight_update called without a pending begin_weight_update.")
        thread.join()
        if self._update_error:
            raise self._update_error[0]
        log_event("vllm", "Weight update completed.")


    def shutdown(self) -> None:
        log_event("vllm", "Shutting down vLLM server wrapper.")
        self._llm = None
