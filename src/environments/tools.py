import subprocess
from pathlib import Path

from src.environments.base import ToolResult

# Tool schemas in the format expected by Hermes' chat template `tools=` argument.
TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path within the workspace."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file at the given path. Overwrites existing content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command in the workspace and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": "Seconds before the command is killed."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a regex pattern in the codebase using grep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Subdirectory to restrict the search."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories at the given path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


def _resolve(workspace: Path, path: str) -> Path:
    """Resolve a possibly-relative path against the workspace root."""
    p = Path(path)
    if not p.is_absolute():
        p = workspace / p
    return p


def tool_read_file(workspace: Path, path: str) -> ToolResult:
    target = _resolve(workspace, path)
    if not target.exists():
        return ToolResult(output=f"File not found: {path}", success=False)
    text = target.read_text(errors="replace")
    return ToolResult(output=text)


def tool_write_file(workspace: Path, path: str, content: str) -> ToolResult:
    target = _resolve(workspace, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return ToolResult(
        output=f"Wrote {len(content)} chars to {path}",
        metadata={"written_path": str(target), "content": content},
    )


def tool_run_command(workspace: Path, command: str, timeout: int = 60) -> ToolResult:
    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    payload = f"exit={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return ToolResult(output=payload, success=proc.returncode == 0,
                      metadata={"returncode": proc.returncode, "command": command})


def tool_search_code(workspace: Path, pattern: str, path: str | None = None) -> ToolResult:
    target = _resolve(workspace, path) if path else workspace
    cmd = ["grep", "-rnE", "--", pattern, str(target)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return ToolResult(output=proc.stdout or "(no matches)", success=True)


def tool_list_directory(workspace: Path, path: str) -> ToolResult:
    target = _resolve(workspace, path)
    if not target.exists():
        return ToolResult(output=f"Path not found: {path}", success=False)
    entries = []
    for entry in sorted(target.iterdir()):
        suffix = "/" if entry.is_dir() else ""
        entries.append(entry.name + suffix)
    return ToolResult(output="\n".join(entries))


def dispatch_tool_call(workspace: Path, name: str, arguments: dict) -> ToolResult:
    """
    Dispatch a parsed tool call to its implementation.

    :param workspace: Filesystem root where the tools operate.
    :param name: Tool name.
    :param arguments: Parsed JSON arguments.
    """
    if name == "read_file":
        return tool_read_file(workspace, arguments["path"])
    if name == "write_file":
        return tool_write_file(workspace, arguments["path"], arguments["content"])
    if name == "run_command":
        return tool_run_command(workspace, arguments["command"], int(arguments.get("timeout", 60)))
    if name == "search_code":
        return tool_search_code(workspace, arguments["pattern"], arguments.get("path"))
    if name == "list_directory":
        return tool_list_directory(workspace, arguments["path"])
    return ToolResult(output=f"Unknown tool: {name}", success=False)
