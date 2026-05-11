import shutil
import subprocess
import tempfile
from pathlib import Path

from src.environments.base import Environment, ToolResult
from src.environments.tools import dispatch_tool_call, normalize_tool_arguments
from src.helpers import log_event

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
        self._agent_ran_verification: bool = False
        self._final_response: str = ""
        self._solution_origin: str = "missing"
        self._solution_written_via_tool: bool = False
        self._verification_attempted_after_write: bool = False


    def setup(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="cc-env-"))
        log_event(
            "codecontests_env",
            f"Created workspace {self.workspace} for problem {self.instance.get('name', '<unknown>')}.",
        )


    def dispatch_tool(self, name: str, arguments: dict | str | None) -> ToolResult:
        normalized = normalize_tool_arguments(arguments)
        self._tool_call_history.append({"name": name, "arguments": normalized})
        result = dispatch_tool_call(self.workspace, name, normalized)
        if name == "write_file":
            written_path = result.metadata.get("written_path")
            content = result.metadata.get("content", "")
            if written_path:
                self._written_files[written_path] = content
                self._modified_files.add(written_path)
                if Path(written_path) == self._solution_path():
                    self._solution_origin = "tool_write"
                    self._solution_written_via_tool = True
        if name == "run_command":
            command = str(normalized.get("command", ""))
            if self._looks_like_verification_command(command):
                self._agent_ran_verification = True
                if self._solution_written_via_tool:
                    self._verification_attempted_after_write = True
        return result


    def record_final_response(self, final_response: str) -> None:
        """Persist the model's terminal visible response for diagnostics only."""
        self._final_response = final_response or ""


    def _solution_path(self) -> Path:
        return self.workspace / SOLUTION_FILE


    def _looks_like_verification_command(self, command: str) -> bool:
        """Heuristic signal that a `run_command` invocation was used for solution verification."""
        lowered = command.lower()
        markers = (
            "solution.py",
            "pytest",
            "unittest",
            "diff ",
            "cmp ",
            "stdin",
            "<<",
            "echo ",
            "cat ",
        )
        return any(marker in lowered for marker in markers)

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

    def run_tests(self) -> dict:
        """Run the agent's solution against every `corner_cases` test."""
        if not self._solution_path().exists():
            self._solution_origin = "missing"
        cases = self.instance.get("corner_cases") or []
        passed = 0
        first_failed_case: dict | None = None
        for idx, case in enumerate(cases, start=1):
            stdin = (case.get("input") or {}).get("stdin", "")
            expected = (case.get("output") or {}).get("stdout", "")
            try:
                code, out, _ = self._execute_solution(stdin)
            except subprocess.TimeoutExpired:
                if first_failed_case is None:
                    first_failed_case = {
                        "index": idx,
                        "status": "timeout",
                        "stdin_preview": self._preview_text(stdin),
                        "expected_stdout_preview": self._preview_text(expected),
                    }
                continue
            matched = code == 0 and out.strip().split() == expected.strip().split()
            if not matched and first_failed_case is None:
                first_failed_case = {
                    "index": idx,
                    "status": "failed",
                    "returncode": code,
                    "stdin_preview": self._preview_text(stdin),
                    "expected_stdout_preview": self._preview_text(expected),
                    "produced_stdout_preview": self._preview_text(out),
                }
            if matched:
                passed += 1
        results = {
            "tests_passed": passed,
            "tests_total": len(cases),
            "solution_origin": self._solution_origin,
            "solution_written_via_tool": self._solution_written_via_tool,
            "verification_attempted_after_write": self._verification_attempted_after_write,
            "required_workflow_followed": (
                self._solution_written_via_tool and self._verification_attempted_after_write
            ),
        }
        if first_failed_case is not None:
            results["first_failed_case"] = first_failed_case
        return results


    def collect_metadata(self) -> dict:
        return {
            "written_files": dict(self._written_files),
            "modified_files": set(self._modified_files),
            "task_relevant_files": {str(self._solution_path())},
            "tool_call_history": list(self._tool_call_history),
            "agent_ran_verification": self._agent_ran_verification,
            "solution_origin": self._solution_origin,
            "solution_written_via_tool": self._solution_written_via_tool,
            "verification_attempted_after_write": self._verification_attempted_after_write,
            "required_workflow_followed": (
                self._solution_written_via_tool and self._verification_attempted_after_write
            ),
            "final_response_present": bool(self._final_response.strip()),
        }

    def _preview_text(self, text: str, limit: int = 200) -> str:
        """Return a compact single-line preview suitable for logs."""
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[:limit] + "...<truncated>"


    def teardown(self) -> None:
        if self.workspace is not None and self.workspace.exists():
            shutil.rmtree(self.workspace, ignore_errors=True)
            self.workspace = None
