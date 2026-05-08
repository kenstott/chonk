#!/usr/bin/env bash
# Install the Chonk MCP server entry into your MCP host config.
# Requires: python3 (macOS built-in) or jq
#
# Usage:
#   ./install_mcp.sh --url http://chonk.internal:8000/mcp --api-key SECRET
#
# Options:
#   --url       URL of the Chonk MCP HTTP server (required)
#   --api-key   Bearer token (required when server has auth enabled)
#   --name      Entry name (default: chonk)
#   --host      claude | claude-code | gemini | copilot | vscode (default: claude)
#   --dry-run   Print the config change without writing it

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
URL=""
API_KEY=""
NAME="chonk"
HOST="claude"
DRY_RUN=false

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --url)       URL="$2";     shift 2 ;;
        --api-key)   API_KEY="$2"; shift 2 ;;
        --name)      NAME="$2";    shift 2 ;;
        --host)      HOST="$2";    shift 2 ;;
        --dry-run)   DRY_RUN=true; shift   ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 --url <url> [--api-key <key>] [--name chonk] [--host claude] [--dry-run]" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$URL" ]]; then
    echo "Error: --url is required." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve config file path
# ---------------------------------------------------------------------------
case "$HOST" in
    claude)
        case "$(uname)" in
            Darwin) CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json" ;;
            *)      CONFIG="$HOME/.config/Claude/claude_desktop_config.json" ;;
        esac
        TOP_KEY="mcpServers"
        ;;
    claude-code)
        CONFIG="$(pwd)/.mcp.json"
        TOP_KEY="mcpServers"
        ;;
    claude-code-user)
        CONFIG="$HOME/.claude.json"
        TOP_KEY="mcpServers"
        ;;
    gemini)
        CONFIG="$HOME/.gemini/settings.json"
        TOP_KEY="mcpServers"
        ;;
    copilot)
        CONFIG="$(pwd)/.vscode/mcp.json"
        TOP_KEY="servers"
        ;;
    vscode)
        case "$(uname)" in
            Darwin) CONFIG="$HOME/Library/Application Support/Code/User/settings.json" ;;
            *)      CONFIG="$HOME/.config/Code/User/settings.json" ;;
        esac
        TOP_KEY="mcp_servers"   # handled specially below
        ;;
    *)
        CONFIG="$HOST"
        TOP_KEY="mcpServers"
        ;;
esac

# ---------------------------------------------------------------------------
# Build the entry JSON
# ---------------------------------------------------------------------------
if [[ -n "$API_KEY" ]]; then
    ENTRY=$(printf '{"url":"%s","headers":{"Authorization":"Bearer %s"}}' "$URL" "$API_KEY")
else
    ENTRY=$(printf '{"url":"%s"}' "$URL")
fi

# Claude Code and Copilot/VSCode need a "type" field
case "$HOST" in
    claude-code|claude-code-user|copilot|vscode)
        ENTRY=$(printf '{"type":"http","url":"%s"' "$URL")
        if [[ -n "$API_KEY" ]]; then
            ENTRY+=",\"headers\":{\"Authorization\":\"Bearer $API_KEY\"}"
        fi
        ENTRY+="}"
        ;;
esac

# ---------------------------------------------------------------------------
# Merge using python3 (available on macOS/Linux without extra installs)
# ---------------------------------------------------------------------------
merge_config() {
    local config_file="$1"
    local top_key="$2"
    local entry_name="$3"
    local entry_json="$4"
    local host="$5"

    python3 - <<PYEOF
import json, os, sys

config_file = """$config_file"""
top_key     = """$top_key"""
entry_name  = """$entry_name"""
entry_json  = """$entry_json"""
host        = """$host"""

config = {}
if os.path.exists(config_file):
    try:
        config = json.loads(open(config_file).read())
    except json.JSONDecodeError as e:
        print(f"Error: cannot parse {config_file}: {e}", file=sys.stderr)
        sys.exit(1)

entry = json.loads(entry_json)

if host == "vscode":
    config.setdefault("mcp", {}).setdefault("servers", {})[entry_name] = entry
elif host == "copilot":
    config.setdefault("servers", {})[entry_name] = entry
else:
    config.setdefault("mcpServers", {})[entry_name] = entry

print(json.dumps(config, indent=2))
PYEOF
}

NEW_CONFIG=$(merge_config "$CONFIG" "$TOP_KEY" "$NAME" "$ENTRY" "$HOST")

# ---------------------------------------------------------------------------
# Write or print
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] Would write to: $CONFIG"
    echo "$NEW_CONFIG"
    exit 0
fi

mkdir -p "$(dirname "$CONFIG")"
echo "$NEW_CONFIG" > "$CONFIG"

echo "Installed '$NAME' → $CONFIG"
case "$HOST" in
    claude-code|claude-code-user)
        echo "Run: claude mcp list" ;;
    *)
        echo "Restart your MCP host to pick up the change." ;;
esac
