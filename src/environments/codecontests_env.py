import shutil
import subprocess
import tempfile
from pathlib import Path

from src.environments.base import Environment, ToolResult
from src.environments.tools import dispatch_tool_call


SOLUTION_FILE = "solution.py"


class CodeContestsEnvironment(Environment):
    """
    Sandboxed execution context for a single CodeContests-O problem.
    Provides a scratch directory where the agent writes its solution and
    runs it against the pre-generated `corner_cases` test suite.
    """

    def __init__(self, instance: dict, test_timeout: int = 60) -> None:
        self.instance = instance
        self.test_timeout = test_timeout
        self.workspace: Path | None = None
        self._written_files: dict[str, str] = {}
        self._modified_files: set[str] = set()
        self._tool_call_history: list[dict] = []
        self._ran_tests: bool = False


    def setup(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="cc-env-"))


    def dispatch_tool(self, name: str, arguments: dict) -> ToolResult:
        self._tool_call_history.append({"name": name, "arguments": arguments})
        result = dispatch_tool_call(self.workspace, name, arguments)
        if name == "write_file":
            written_path = result.metadata.get("written_path")
            content = result.metadata.get("content", "")
            if written_path:
                self._written_files[written_path] = content
                self._modified_files.add(written_path)
        if name == "run_command":
            self._ran_tests = self._ran_tests or "test" in str(arguments.get("command", ""))
        return result


    def _solution_path(self) -> Path:
        return self.workspace / SOLUTION_FILE


    def _execute_solution(self, stdin_text: str) -> tuple[int, str, str]:
        """Run the agent's solution.py with the provided stdin payload."""
        sol = self._solution_path()
        if not sol.exists():
            return 1, "", "no solution.py written"
        proc = subprocess.run(
            ["python", str(sol)],
            input=stdin_text, capture_output=True, text=True,
            timeout=self.test_timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr


    def _outputs_match(self, produced: str, expected: str) -> bool:
        """Whitespace-insensitive comparison used when no `checker` is provided."""
        return produced.strip().split() == expected.strip().split()


    def run_tests(self) -> dict:
        """Run the agent's solution against every `corner_cases` test."""
        self._ran_tests = True
        cases = self.instance.get("corner_cases") or []
        passed = 0
        for case in cases:
            stdin = (case.get("input") or {}).get("stdin", "")
            expected = (case.get("output") or {}).get("stdout", "")
            try:
                code, out, _ = self._execute_solution(stdin)
            except subprocess.TimeoutExpired:
                continue
            if code == 0 and self._outputs_match(out, expected):
                passed += 1
        return {"tests_passed": passed, "tests_total": len(cases)}


    def collect_metadata(self) -> dict:
        return {
            "written_files": dict(self._written_files),
            "modified_files": set(self._modified_files),
            "task_relevant_files": {str(self._solution_path())},
            "tool_call_history": list(self._tool_call_history),
            "ran_tests_before_finish": self._ran_tests,
        }


    def teardown(self) -> None:
        if self.workspace is not None and self.workspace.exists():
            shutil.rmtree(self.workspace, ignore_errors=True)
            self.workspace = None
