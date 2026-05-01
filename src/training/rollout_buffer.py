import threading
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
        dropped from the front. Blocks until at least one valid group is available.
        """
        out: list[RolloutGroup] = []
        with self._not_empty:
            while True:
                while self._buffer and self._buffer[0].policy_version + self.max_staleness < current_version:
                    self._buffer.popleft()
                    self._dropped += 1
                while self._buffer and len(out) < count:
                    out.append(self._buffer.popleft())
                if out:
                    self._not_full.notify_all()
                    return out
                if not self._not_empty.wait(timeout=timeout):
                    return out


    def depth(self) -> int:
        with self._lock:
            return len(self._buffer)


    def pop_staleness_drops(self) -> int:
        """Return and reset the count of staleness-dropped groups."""
        with self._lock:
            d = self._dropped
            self._dropped = 0
            return d


    def unblock_consumers(self) -> None:
        """Wake every blocked `pull()` so callers can observe a shutdown signal."""
        with self._lock:
            self._not_empty.notify_all()
            self._not_full.notify_all()
