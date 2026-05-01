from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    """
    The structured result of a single tool invocation.
    `output` is the textual payload that gets wrapped into `<tool_response>`.
    """
    output: str
    success: bool = True
    metadata: dict = field(default_factory=dict)


class Environment(ABC):
    """
    Abstract execution context for a single rollout. Concrete subclasses
    (SWEEnvironment, CodeContestsEnvironment) implement task-specific
    setup, tool dispatch, and test execution for reward computation.
    """

    @abstractmethod
    def setup(self) -> None:
        """Provision the environment (clone repo, build container, etc.)."""


    @abstractmethod
    def dispatch_tool(self, name: str, arguments: dict) -> ToolResult:
        """Execute a tool call within this environment."""


    @abstractmethod
    def run_tests(self) -> dict:
        """
        Execute the task's verification tests and return a result dictionary
        consumable by `training.rewards.compute_correctness_reward`.
        """


    @abstractmethod
    def collect_metadata(self) -> dict:
        """Return rollout metadata used by auxiliary rewards (written files, modified files, etc.)."""


    @abstractmethod
    def teardown(self) -> None:
        """Release any resources held by the environment."""
