#!/usr/bin/env python3
"""
Install the Chonk MCP server entry into your MCP host config.

Usage (HTTP — enterprise, server already running centrally):
    python install_mcp.py --url http://chonk.internal:8000/mcp --api-key SECRET

Usage (stdio — local, server runs as a subprocess):
    python install_mcp.py --db /data/index.duckdb

Options:
    --name NAME          Entry name in the MCP config (default: chonk)
    --url URL            HTTP server URL (use for centralised deployment)
    --api-key KEY        Bearer token for HTTP transport
    --db PATH            DuckDB file path (stdio transport)
    --dim N              Embedding dimension (default: 1024)
    --db-config JSON     Multi-DB JSON config string (overrides --db)
    --python PATH        Python interpreter to use (default: auto-detect venv)
    --server PATH        Path to mcp_chonk_server.py (default: auto-detect)
    --host HOST          Config target (default: claude):

                           claude            Claude Desktop
                           claude-code       Claude Code CLI — project scope (.mcp.json)
                           claude-code-user  Claude Code CLI — user scope (~/.claude.json)
                           gemini            Google Gemini CLI (~/.gemini/settings.json)
                           copilot           GitHub Copilot in VS Code (.vscode/mcp.json)
                           cursor            Cursor
                           vscode            VS Code (settings.json)
                           <file path>       Any JSON config file

    --dry-run            Print what would be written without modifying anything
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config file locations
# ---------------------------------------------------------------------------

def _claude_config_path() -> Path:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if system == "Windows":
        return Path(os.environ["APPDATA"]) / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def _claude_code_project_path() -> Path:
    return Path.cwd() / ".mcp.json"


def _claude_code_user_path() -> Path:
    return Path.home() / ".claude.json"


def _gemini_config_path() -> Path:
    return Path.home() / ".gemini" / "settings.json"


def _copilot_config_path() -> Path:
    return Path.cwd() / ".vscode" / "mcp.json"


def _cursor_config_path() -> Path:
    system = platform.system()
    if system == "Darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Cursor"
            / "User"
            / "globalStorage"
            / "rooveterinaryinc.roo-cline"
            / "settings"
            / "cline_mcp_settings.json"
        )
    if system == "Windows":
        return (
            Path(os.environ["APPDATA"])
            / "Cursor"
            / "User"
            / "globalStorage"
            / "rooveterinaryinc.roo-cline"
            / "settings"
            / "cline_mcp_settings.json"
        )
    return (
        Path.home()
        / ".config"
        / "Cursor"
        / "User"
        / "globalStorage"
        / "rooveterinaryinc.roo-cline"
        / "settings"
        / "cline_mcp_settings.json"
    )


def _vscode_config_path() -> Path:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Code" / "User" / "settings.json"
    if system == "Windows":
        return Path(os.environ["APPDATA"]) / "Code" / "User" / "settings.json"
    return Path.home() / ".config" / "Code" / "User" / "settings.json"


_HOST_PATHS: dict[str, object] = {
    "claude": _claude_config_path,
    "claude-code": _claude_code_project_path,
    "claude-code-user": _claude_code_user_path,
    "gemini": _gemini_config_path,
    "copilot": _copilot_config_path,
    "cursor": _cursor_config_path,
    "vscode": _vscode_config_path,
}

# Hosts whose entry format requires an explicit "type" field
_TYPED_HOSTS = {"claude-code", "claude-code-user", "copilot", "vscode"}


# ---------------------------------------------------------------------------
# Server script auto-detection
# ---------------------------------------------------------------------------

def _find_server_script() -> str:
    candidate = Path(__file__).parent / "mcp_chonk_server.py"
    if candidate.exists():
        return str(candidate.resolve())
    raise FileNotFoundError(
        "Cannot find mcp_chonk_server.py. Pass --server /path/to/mcp_chonk_server.py"
    )


def _find_python() -> str:
    venv = Path(__file__).parent / ".venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return sys.executable


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _build_http_entry(url: str, api_key: str | None, typed: bool) -> dict:
    entry: dict = {"url": url}
    if typed:
        entry = {"type": "http", **entry}
    if api_key:
        entry["headers"] = {"Authorization": f"Bearer {api_key}"}
    return entry


def _build_stdio_entry(
    server_path: str,
    python: str,
    db_path: str | None,
    db_config: str | None,
    dim: int,
    typed: bool,
) -> dict:
    env: dict[str, str] = {"CHONK_EMBEDDING_DIM": str(dim)}
    if db_config:
        env["CHONK_DB_CONFIG"] = db_config
    elif db_path:
        env["CHONK_DB_PATH"] = db_path
    else:
        raise ValueError("Stdio transport requires --db or --db-config")
    entry: dict = {"command": python, "args": [server_path], "env": env}
    if typed:
        entry = {"type": "stdio", **entry}
    return entry


# ---------------------------------------------------------------------------
# Read / write config
# ---------------------------------------------------------------------------

def _read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Cannot parse {path}: {exc}") from exc


def _write_config(path: Path, data: dict, dry_run: bool) -> None:
    text = json.dumps(data, indent=2) + "\n"
    if dry_run:
        print(f"\n[dry-run] Would write to {path}:\n{text}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Written: {path}")


def _inject_entry(config: dict, name: str, entry: dict, host: str) -> dict:
    """Insert the MCP entry at the correct key structure for each host."""
    if host == "vscode":
        # VS Code settings.json: {"mcp": {"servers": {...}}}
        config.setdefault("mcp", {}).setdefault("servers", {})[name] = entry
    elif host == "copilot":
        # .vscode/mcp.json: {"servers": {...}}
        config.setdefault("servers", {})[name] = entry
    else:
        # Claude Desktop, Claude Code, Gemini, Cursor: {"mcpServers": {...}}
        config.setdefault("mcpServers", {})[name] = entry
    return config


def _existing_entry(config: dict, name: str, host: str) -> dict | None:
    if host == "vscode":
        return config.get("mcp", {}).get("servers", {}).get(name)
    if host == "copilot":
        return config.get("servers", {}).get(name)
    return config.get("mcpServers", {}).get(name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the Chonk MCP server into your MCP host config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", default="chonk", help="Entry name (default: chonk)")
    parser.add_argument("--url", help="HTTP server URL (centralised deployment)")
    parser.add_argument("--api-key", help="Bearer token for HTTP transport")
    parser.add_argument("--db", help="DuckDB file path (stdio transport)")
    parser.add_argument("--dim", type=int, default=1024, help="Embedding dimension (default: 1024)")
    parser.add_argument("--db-config", help="Multi-DB JSON config string (overrides --db)")
    parser.add_argument("--python", help="Python interpreter (default: auto-detect venv)")
    parser.add_argument("--server", help="Path to mcp_chonk_server.py (default: auto-detect)")
    parser.add_argument(
        "--host",
        default="claude",
        help=(
            "Target: claude, claude-code, claude-code-user, "
            "gemini, copilot, cursor, vscode, or a file path (default: claude)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    args = parser.parse_args()

    # Resolve config file path
    if args.host in _HOST_PATHS:
        config_path = _HOST_PATHS[args.host]()  # type: ignore[operator]
    else:
        config_path = Path(args.host)

    typed = args.host in _TYPED_HOSTS

    # Build entry
    if args.url:
        entry = _build_http_entry(args.url, args.api_key, typed=typed)
        transport_desc = f"http → {args.url}"
    else:
        server_path = args.server or _find_server_script()
        python = args.python or _find_python()
        entry = _build_stdio_entry(
            server_path, python, args.db, args.db_config, args.dim, typed=typed
        )
        transport_desc = f"stdio → {server_path}"

    # Read, patch, write
    config = _read_config(config_path)
    if _existing_entry(config, args.name, args.host) and not args.dry_run:
        print(f"Replacing existing '{args.name}' entry in {config_path}")

    config = _inject_entry(config, args.name, entry, args.host)
    _write_config(config_path, config, args.dry_run)

    if not args.dry_run:
        print(f"\nInstalled '{args.name}' ({transport_desc})")
        if args.host in ("claude-code", "claude-code-user"):
            print("Run `claude mcp list` to verify.")
        else:
            print("Restart your MCP host to pick up the change.")


if __name__ == "__main__":
    main()
