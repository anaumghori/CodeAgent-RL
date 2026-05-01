import threading

from src.config.config import PipelineConfig


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
            self.llm.sleep(level=0, mode="keep")
            self.llm.reset_prefix_cache(reset_running_requests=True)


    def resume_after_weight_sync(self) -> None:
        with self._lock:
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


    def sleep_for_reload(self) -> None:
        """Drop weights + KV cache so new weights can be received without OOM."""
        self.llm.sleep(level=2)


    def wake_after_reload(self) -> None:
        """Restore weights then KV cache after a sleep-mode-style reload."""
        self.llm.wake_up(tags=["weights"])
        self.llm.collective_rpc("reload_weights")
        self.llm.wake_up(tags=["kv_cache"])


    def shutdown(self) -> None:
        self._llm = None
