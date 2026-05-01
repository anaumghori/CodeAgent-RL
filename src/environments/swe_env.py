import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.environments.base import Environment, ToolResult
from src.environments.tools import dispatch_tool_call


class SWEEnvironment(Environment):
    """
    Docker-backed execution context for a single SWE-rebench-V2 or
    SWE-rebench-V2-PRs instance. Clones the target repository at
    `base_commit` into a temporary worktree mapped into a Docker container,
    then dispatches tool calls inside the container.
    """

    def __init__(self, instance: dict, source: str, test_timeout: int = 60) -> None:
        self.instance = instance
        self.source = source
        self.test_timeout = test_timeout
        self.workspace: Path | None = None
        self.container_id: str | None = None
        self._written_files: dict[str, str] = {}
        self._modified_files: set[str] = set()
        self._tool_call_history: list[dict] = []
        self._ran_tests: bool = False


    def _image_name(self) -> str:
        """Return the Docker image to spawn — pre-built for V2, built from install_config for PRs."""
        img = self.instance.get("image_name")
        if img:
            return img
        install = self.instance.get("install_config") or {}
        return install.get("base_image", "python:3.11-slim")


    def setup(self) -> None:
        """Provision a workspace directory and spawn the Docker container."""
        self.workspace = Path(tempfile.mkdtemp(prefix="swe-env-"))
        repo = self.instance.get("repo", "")
        commit = self.instance.get("base_commit", "")
        if repo:
            clone_proc = subprocess.run(
                ["git", "clone", f"https://github.com/{repo}.git", str(self.workspace)],
                check=True, capture_output=True, text=True,
            )
            if commit:
                subprocess.run(
                    ["git", "checkout", commit],
                    cwd=str(self.workspace),
                    check=True,
                    capture_output=True,
                    text=True,
                )
        # Spin up a long-lived container with the workspace mounted.
        result = subprocess.run(
            [
                "docker", "run", "-d", "--rm",
                "-v", f"{self.workspace}:/workspace",
                "-w", "/workspace",
                self._image_name(),
                "sleep", "infinity",
            ],
            capture_output=True, text=True, check=True,
        )
        self.container_id = result.stdout.strip()
        # Run install commands for PR instances that lack pre-built images.
        install = self.instance.get("install_config") or {}
        for cmd in install.get("install_commands", []):
            self._docker_exec(cmd, timeout=600)


    def _docker_exec(self, command: str, timeout: int) -> tuple[int, str, str]:
        """Run a shell command inside the container and capture its output."""
        if self.container_id is None:
            return 1, "", "no container"
        proc = subprocess.run(
            ["docker", "exec", self.container_id, "bash", "-lc", command],
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr


    def dispatch_tool(self, name: str, arguments: dict) -> ToolResult:
        """Route the tool call: file IO operates on the host workspace, run_command goes into Docker."""
        self._tool_call_history.append({"name": name, "arguments": arguments})
        if name == "run_command" and self.container_id is not None:
            timeout = int(arguments.get("timeout", 60))
            code, out, err = self._docker_exec(arguments["command"], timeout=timeout)
            return ToolResult(
                output=f"exit={code}\nstdout:\n{out}\nstderr:\n{err}",
                success=code == 0, metadata={"returncode": code},
            )
        result = dispatch_tool_call(self.workspace, name, arguments)
        if name == "write_file":
            written_path = result.metadata.get("written_path")
            content = result.metadata.get("content", "")
            if written_path:
                self._written_files[written_path] = content
                self._modified_files.add(written_path)
        return result


    def run_tests(self) -> dict:
        """Execute FAIL_TO_PASS and PASS_TO_PASS test sets and tally outcomes."""
        self._ran_tests = True
        install = self.instance.get("install_config") or {}
        test_cmd_template = install.get("test_command", "pytest -x {test_id}")

        def _run_tests(test_ids: list[str]) -> int:
            passed = 0
            for tid in test_ids:
                cmd = test_cmd_template.format(test_id=tid)
                code, _, _ = self._docker_exec(cmd, timeout=self.test_timeout)
                if code == 0:
                    passed += 1
            return passed

        f2p = self.instance.get("FAIL_TO_PASS") or []
        p2p = self.instance.get("PASS_TO_PASS") or []
        return {
            "fail_to_pass_passed": _run_tests(f2p),
            "fail_to_pass_total": len(f2p),
            "pass_to_pass_passed": _run_tests(p2p),
            "pass_to_pass_total": len(p2p),
        }


    def collect_metadata(self) -> dict:
        """Gather inputs needed by the auxiliary-reward functions."""
        return {
            "written_files": dict(self._written_files),
            "modified_files": set(self._modified_files),
            "task_relevant_files": set(),
            "tool_call_history": list(self._tool_call_history),
            "ran_tests_before_finish": self._ran_tests,
        }


    def teardown(self) -> None:
        """Stop the Docker container and remove the temporary workspace."""
        if self.container_id is not None:
            subprocess.run(["docker", "kill", self.container_id], capture_output=True, check=False)
            self.container_id = None
        if self.workspace is not None and self.workspace.exists():
            shutil.rmtree(self.workspace, ignore_errors=True)
            self.workspace = None
