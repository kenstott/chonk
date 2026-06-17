#!/usr/bin/env bash
# Launch the aipa_test_mcp_server using the project venv.
#
# Usage:
#   ./launch_aipa_test_mcp_server.sh            # stdio (default)
#   CHONK_TRANSPORT=http ./launch_aipa_test_mcp_server.sh
#   CHONK_HOST=127.0.0.1 CHONK_PORT=8000 CHONK_TRANSPORT=http ./launch_aipa_test_mcp_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/aipa_test_mcp_server"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "Error: venv not found at $SCRIPT_DIR/.venv" >&2
    echo "Create it with:  python3 -m venv .venv && .venv/bin/pip install -e '.[storage,mcp]'" >&2
    exit 1
fi

if [[ ! -f "$SERVER_DIR/index.duckdb" ]]; then
    echo "Index not found. Building from $SERVER_DIR/index_config.yaml ..." >&2
    cd "$SERVER_DIR"
    "$VENV_PYTHON" -c "from chonk.ingest import build; build('index_config.yaml')"
    cd "$SCRIPT_DIR"
fi

cd "$SERVER_DIR"
exec "$VENV_PYTHON" server.py "$@"
