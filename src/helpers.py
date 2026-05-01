import time
import zlib
from contextlib import contextmanager
from typing import Iterator
import torch

@contextmanager
def timed(store: dict, key: str) -> Iterator[None]:
    """
    Context manager that records the wall-clock duration of a code block
    into `store[key]` using `time.perf_counter()` for precision.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        store[key] = time.perf_counter() - start


def gpu_memory_gb(device: int) -> tuple[float, float]:
    """Return (allocated_gb, reserved_gb) for the given CUDA device."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    return allocated, reserved
