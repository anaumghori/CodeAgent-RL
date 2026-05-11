import base64
import os
import threading
import time
import modal

from src.config.config import PipelineConfig
from src.data.prompt_queue import SOURCE_SWE_V2
from src.environments.base import Environment, ToolResult
from src.helpers import format_exception, log_event

MODAL_WORKSPACE = "/workspace"
_MODAL_IMAGE_CACHE: dict[str, modal.Image] = {}
_IMAGE_CACHE_LOCK = threading.Lock()
_MODAL_APP = None


class SWEEnvironmentMetrics:
    """
    Thread-safe accumulator for SWE environment setup timings and Modal image
    resolution timings. The orchestrator snapshots and resets these counters at
    log cadence so the runtime can expose environment setup behaviour without
    emitting per-environment trace spam.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._setup_times: list[float] = []
        self._image_resolve_times: list[float] = []
        self._setup_failures: int = 0
        self._setup_successes: int = 0


    def record_setup(self, seconds: float, success: bool) -> None:
        """Record one environment setup attempt."""
        with self._lock:
            if success:
                self._setup_successes += 1
                self._setup_times.append(seconds)
                return
            self._setup_failures += 1


    def record_image_resolution(self, seconds: float) -> None:
        """Record one Modal image resolution duration."""
        with self._lock:
            self._image_resolve_times.append(seconds)


    def snapshot_and_reset(self) -> dict:
        """Return aggregated stats and clear the internal buffers."""
        with self._lock:
            setup = list(self._setup_times)
            image_resolve = list(self._image_resolve_times)
            failures = self._setup_failures
            successes = self._setup_successes
            self._setup_times.clear()
            self._image_resolve_times.clear()
            self._setup_failures = 0
            self._setup_successes = 0
        out = {
            "env/setup_failures": failures,
            "env/setup_successes": successes,
        }
        out.update(_quantiles("env/setup_time_sec", setup))
        out.update(_quantiles("env/image_resolution_time_sec", image_resolve))
        return out


METRICS = SWEEnvironmentMetrics()


def _quantiles(prefix: str, samples: list[float]) -> dict:
    """Return p50/p95/max/count for a list of timing samples."""
    if not samples:
        return {
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_count": 0,
        }
    ordered = sorted(samples)
    n = len(ordered)
    return {
        f"{prefix}_p50": ordered[n // 2],
        f"{prefix}_p95": ordered[min(n - 1, int(n * 0.95))],
        f"{prefix}_max": ordered[-1],
        f"{prefix}_count": n,
    }


def _modal_app() -> modal.App:
    """Return the Modal App handle used to create child sandboxes."""
    global _MODAL_APP
    if _MODAL_APP is None:
        app_name = os.environ.get("COMPOSER_RL_MODAL_APP_NAME", "composer-rl")
        _MODAL_APP = modal.App.lookup(app_name, create_if_missing=True)
    return _MODAL_APP


def _resolve_registry_image(tag: str) -> modal.Image:
    """Resolve and cache a registry-backed Modal image by tag."""
    with _IMAGE_CACHE_LOCK:
        image = _MODAL_IMAGE_CACHE.get(tag)
        if image is not None:
            return image
        started = time.perf_counter()
        with modal.enable_output():
            image = modal.Image.from_registry(tag).entrypoint([])
        _MODAL_IMAGE_CACHE[tag] = image
    METRICS.record_image_resolution(time.perf_counter() - started)
    return image


class SWEEnvironment(Environment):
    """
    Execution context for a single SWE-rebench-V2 instance running inside a
    native Modal sandbox created directly from the dataset-provided registry
    image. The repository is materialized inside `/workspace`, and every tool
    call executes against that isolated sandbox state.
    """

    def __init__(self, instance: dict, source: str, cfg: PipelineConfig,
                 test_timeout: int = 60) -> None:
        self.instance = instance
        self.source = source
        self.cfg = cfg
        self.test_timeout = test_timeout
        self.sandbox = None
        self._written_files: dict[str, str] = {}
        self._modified_files: set[str] = set()
        self._tool_call_history: list[dict] = []
        self._agent_ran_verification: bool = False


    def _sandbox_exec(self, *args: str, timeout: int, workdir: str | None = None) -> tuple[int, str, str]:
        """Execute a command inside the Modal sandbox."""
        if self.sandbox is None:
            return 1, "", "no sandbox"
        proc = self.sandbox.exec(*args, timeout=timeout, workdir=workdir, text=True)
        stdout = proc.stdout.read()
        stderr = proc.stderr.read()
        proc.wait()
        return proc.returncode, stdout, stderr


    def _sandbox_workspace_path(self, path: str) -> str:
        """Resolve a tool path inside the sandbox workspace."""
        return path if path.startswith("/") else f"{MODAL_WORKSPACE}/{path}"


    def _download_repo_into_sandbox(self, repo: str, commit: str) -> None:
        """Download and extract the GitHub archive for `repo@commit` into `/workspace`."""
        command = (
            "set -eu\n"
            "repo=\"$1\"\n"
            "commit=\"$2\"\n"
            "dest=\"$3\"\n"
            "url=\"https://codeload.github.com/${repo}/tar.gz/${commit}\"\n"
            "tmpdir=$(mktemp -d)\n"
            "archive=\"$tmpdir/repo.tar.gz\"\n"
            "mkdir -p \"$dest\"\n"
            "if command -v curl >/dev/null 2>&1; then\n"
            "  curl -L --fail --silent --show-error \"$url\" -o \"$archive\"\n"
            "elif command -v wget >/dev/null 2>&1; then\n"
            "  wget -qO \"$archive\" \"$url\"\n"
            "else\n"
            "  echo \"Neither curl nor wget is available in the sandbox.\" >&2\n"
            "  exit 1\n"
            "fi\n"
            "tar -xzf \"$archive\" -C \"$tmpdir\"\n"
            "root_dir=$(find \"$tmpdir\" -mindepth 1 -maxdepth 1 -type d | head -n 1)\n"
            "if [ -z \"$root_dir\" ]; then\n"
            "  echo \"Failed to locate extracted repository directory.\" >&2\n"
            "  exit 1\n"
            "fi\n"
            "cp -a \"$root_dir\"/. \"$dest\"/\n"
            "rm -rf \"$tmpdir\"\n"
        )
        code, out, err = self._sandbox_exec(
            "sh",
            "-lc",
            command,
            "sh",
            repo,
            commit,
            MODAL_WORKSPACE,
            timeout=600,
            workdir="/",
        )
        if code != 0:
            raise RuntimeError(
                f"Failed to download repository archive for {repo}@{commit}: stdout={out}\nstderr={err}"
            )


    def _setup_modal_sandbox(self) -> None:
        """Provision a Modal sandbox from the dataset image and populate `/workspace`."""
        if self.source != SOURCE_SWE_V2:
            raise ValueError(f"Unknown SWE source: {self.source}")
        image_name = self.instance.get("image_name")
        if not image_name:
            raise RuntimeError(
                f"SWE-V2 instance {self.instance.get('instance_id')} has no image_name"
            )
        image = _resolve_registry_image(image_name)
        try:
            with modal.enable_output():
                self.sandbox = modal.Sandbox.create(
                    app=_modal_app(),
                    image=image,
                    timeout=self.cfg.modal.sandbox_timeout_seconds,
                    cpu=self.cfg.modal.sandbox_cpus,
                    memory=self.cfg.modal.sandbox_memory_mb,
                    workdir=MODAL_WORKSPACE,
                )
        except BaseException:
            log_event(
                "swe_env",
                "Modal sandbox creation failed. The Modal build output immediately above "
                f"should contain the underlying image-build error for {image_name}.",
            )
            raise
        self._sandbox_exec("mkdir", "-p", MODAL_WORKSPACE, timeout=60, workdir="/")
        repo = self.instance.get("repo", "")
        commit = self.instance.get("base_commit", "")
        if repo and commit:
            self._download_repo_into_sandbox(repo, commit)


    def setup(self) -> None:
        """Provision the Modal sandbox and materialize the target repository into it."""
        started = time.perf_counter()
        success = False
        try:
            self._setup_modal_sandbox()
            success = True
        except BaseException as exc:
            log_event("swe_env", f"Environment setup failed:\n{format_exception(exc)}")
            raise
        finally:
            METRICS.record_setup(time.perf_counter() - started, success=success)


    def _sandbox_tool(self, name: str, arguments: dict) -> ToolResult:
        """Execute a non-shell tool directly inside the Modal sandbox."""
        if name == "read_file":
            code, out, err = self._sandbox_exec(
                "sh",
                "-lc",
                "cat \"$1\"",
                "sh",
                self._sandbox_workspace_path(arguments["path"]),
                timeout=60,
                workdir=MODAL_WORKSPACE,
            )
            return ToolResult(output=out if code == 0 else err, success=code == 0)

        if name == "write_file":
            encoded = base64.b64encode(arguments["content"].encode()).decode()
            path = self._sandbox_workspace_path(arguments["path"])
            code, out, err = self._sandbox_exec(
                "sh",
                "-lc",
                "mkdir -p \"$(dirname \"$1\")\" && printf '%s' \"$2\" | base64 -d > \"$1\" && echo \"Wrote $1\"",
                "sh",
                path,
                encoded,
                timeout=60,
                workdir=MODAL_WORKSPACE,
            )
            return ToolResult(
                output=out if code == 0 else err,
                success=code == 0,
                metadata={"written_path": path, "content": arguments["content"]},
            )

        if name == "search_code":
            root = self._sandbox_workspace_path(arguments.get("path") or ".")
            code, out, err = self._sandbox_exec(
                "sh",
                "-lc",
                "grep -rnE -- \"$1\" \"$2\" || true",
                "sh",
                arguments["pattern"],
                root,
                timeout=120,
                workdir=MODAL_WORKSPACE,
            )
            payload = out if out.strip() else "(no matches)"
            return ToolResult(output=payload if code == 0 else err, success=code == 0)

        if name == "list_directory":
            code, out, err = self._sandbox_exec(
                "sh",
                "-lc",
                "ls -1Ap \"$1\"",
                "sh",
                self._sandbox_workspace_path(arguments["path"]),
                timeout=60,
                workdir=MODAL_WORKSPACE,
            )
            return ToolResult(output=out if code == 0 else err, success=code == 0)

        return ToolResult(output=f"Unknown tool: {name}", success=False)


    def dispatch_tool(self, name: str, arguments: dict) -> ToolResult:
        """Execute a tool call inside the Modal sandbox and track useful metadata."""
        self._tool_call_history.append({"name": name, "arguments": arguments})
        if name == "run_command":
            command = str(arguments.get("command", ""))
            if self._looks_like_verification_command(command):
                self._agent_ran_verification = True
            timeout = int(arguments.get("timeout", 60))
            code, out, err = self._sandbox_exec(
                "sh", "-lc", command,
                timeout=timeout,
                workdir=MODAL_WORKSPACE,
            )
            return ToolResult(
                output=f"exit={code}\nstdout:\n{out}\nstderr:\n{err}",
                success=code == 0,
                metadata={"returncode": code},
            )
        result = self._sandbox_tool(name, arguments)
        if name == "write_file":
            written_path = result.metadata.get("written_path")
            content = result.metadata.get("content", "")
            if written_path:
                self._written_files[written_path] = content
                self._modified_files.add(written_path)
        return result


    def _looks_like_verification_command(self, command: str) -> bool:
        """Heuristic signal that a command was intended to validate the current patch."""
        lowered = command.lower()
        markers = (
            "pytest",
            "unittest",
            "tox",
            "nox",
            "make test",
            "cargo test",
            "go test",
            "npm test",
            "pnpm test",
            "yarn test",
        )
        return any(marker in lowered for marker in markers)


    def run_tests(self) -> dict:
        """Execute FAIL_TO_PASS and PASS_TO_PASS test sets and tally outcomes."""
        install = self.instance.get("install_config") or {}
        test_cmd_template = install.get("test_command", "pytest -x {test_id}")

        def _run_tests(test_ids: list[str]) -> int:
            passed = 0
            for tid in test_ids:
                cmd = test_cmd_template.format(test_id=tid)
                code, _, _ = self._sandbox_exec(
                    "sh", "-lc", cmd,
                    timeout=self.test_timeout,
                    workdir=MODAL_WORKSPACE,
                )
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
            "agent_ran_verification": self._agent_ran_verification,
        }


    def teardown(self) -> None:
        """Terminate the Modal sandbox backing this environment."""
        if self.sandbox is not None:
            self.sandbox.terminate()
            self.sandbox.detach()
            self.sandbox = None
