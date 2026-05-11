from datetime import datetime, timezone
import json
import time
import traceback
from contextlib import contextmanager
from typing import Iterator
import torch


def log_event(component: str, message: str) -> None:
    """Print a timestamped runtime event for coarse-grained observability."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] [{component}] {message}", flush=True)

def log_json(component: str, label: str, payload) -> None:
    """Print a JSON-formatted payload under a timestamped component label."""
    rendered = json.dumps(payload, indent=2, sort_keys=True, default=str)
    log_event(component, f"{label}:\n{rendered}")

def format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

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

def gpu_memory_snapshot(device: int) -> dict[str, float]:
    """Return current and peak CUDA memory stats in GB for the given device."""
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "max_reserved_gb": 0.0,
        }
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
        "reserved_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
    }

def reset_gpu_peak_memory(device: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
