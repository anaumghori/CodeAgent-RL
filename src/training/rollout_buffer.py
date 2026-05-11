import threading
import time
from collections import deque

from src.inference.rollout import RolloutGroup

class RolloutBuffer:
    """
    Bounded thread-safe FIFO buffer mediating between the inference and training processes. 
    Producers (inference + reward computation) push `RolloutGroup`s tagged with the policy 
    version that generated them; the trainer pulls groups whose policy version is within
    `max_staleness` of the current training version. Stale groups are discarded and counted 
    via `pop_staleness_drops()`.
    """

    def __init__(self, max_groups: int, max_staleness: int) -> None:
        self.max_groups = max_groups
        self.max_staleness = max_staleness
        self._buffer: deque[RolloutGroup] = deque()
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)
        self._dropped = 0
        self._underruns = 0


    def push(self, group: RolloutGroup, timeout: float | None = None) -> None:
        """Block until the buffer has space, then enqueue `group`."""
        with self._not_full:
            while len(self._buffer) >= self.max_groups:
                if not self._not_full.wait(timeout=timeout):
                    return
            self._buffer.append(group)
            self._not_empty.notify()


    def pull(self, count: int, current_version: int, timeout: float | None = None) -> list[RolloutGroup]:
        """
        Pull up to `count` non-stale groups. Stale groups (older than `max_staleness` versions) are silently 
        dropped from the front. Waits for the requested batch size whenever possible and
        only returns a partial batch once `timeout` expires.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._not_empty:
            while True:
                while self._buffer and self._buffer[0].policy_version + self.max_staleness < current_version:
                    self._buffer.popleft()
                    self._dropped += 1
                if len(self._buffer) >= count:
                    out = [self._buffer.popleft() for _ in range(count)]
                    self._not_full.notify_all()
                    return out
                if deadline is not None and time.monotonic() >= deadline:
                    if self._buffer:
                        out = []
                        while self._buffer and len(out) < count:
                            out.append(self._buffer.popleft())
                        self._not_full.notify_all()
                        self._underruns += 1
                        return out
                    self._underruns += 1
                    return []
                wait_timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
                self._not_empty.wait(timeout=wait_timeout)


    def depth(self) -> int:
        with self._lock:
            return len(self._buffer)


    def pop_staleness_drops(self) -> int:
        """Return and reset the count of staleness-dropped groups."""
        with self._lock:
            d = self._dropped
            self._dropped = 0
            return d


    def pop_underruns(self) -> int:
        """Return and reset the count of `pull` calls that timed out empty."""
        with self._lock:
            u = self._underruns
            self._underruns = 0
            return u


    def unblock_consumers(self) -> None:
        """Wake every blocked `pull()` so callers can observe a shutdown signal."""
        with self._lock:
            self._not_empty.notify_all()
            self._not_full.notify_all()
