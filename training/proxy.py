"""
Truncating proxy for mlx_lm server.

Intercepts /v1/chat/completions, rewrites prompts to fit the model's training
context, and logs every request/response pair to training/proxy_logs/ as JSONL
for prompt efficiency analysis.

Usage:
    # Terminal 1 — mlx_lm on 8081
    python -m mlx_lm server \
        --model training/mlx_output/base_mlx \
        --adapter-path training/mlx_output/adapters \
        --port 8081 \
        --max-tokens 8192 \
        --prompt-cache-size 1 \
        --chat-template-args '{"enable_thinking":false}'

    # Terminal 2 — proxy on 8080
    python training/proxy.py

AIPA config: apiBase: http://localhost:8080/v1
"""

from __future__ import annotations

import datetime
import json
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

BACKEND = "http://127.0.0.1:8081"
PORT = 8080
LOG_DIR = Path(__file__).parent / "proxy_logs"

# The model was fine-tuned with this exact system prompt.
# AIPA's system prompt is ~18k tokens — far beyond the 4096-token training context.
SYSTEM_PROMPT = "You convert Python to idiomatic, compilable Go."

# Character budget for non-system content (tool + user messages).
CONTENT_CHAR_BUDGET = 13000

# High-priority roles — kept in full; truncated only as last resort
_KEEP_ROLES = {"user", "tool"}

_log_file: Path | None = None


def _get_log_file() -> Path:
    global _log_file
    if _log_file is None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_file = LOG_DIR / f"session_{ts}.jsonl"
        print(f"[proxy] Logging to {_log_file}")
    return _log_file


def _log(record: dict[str, Any]) -> None:
    with _get_log_file().open("a") as f:
        f.write(json.dumps(record) + "\n")


def _rewrite_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Replace AIPA system prompt with our training prompt; protect file content."""
    result: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    non_system = [m for m in messages if m["role"] != "system"]

    if not non_system:
        return result

    priority = [m for m in non_system if m["role"] in _KEEP_ROLES]
    droppable = [m for m in non_system if m["role"] not in _KEEP_ROLES]

    remaining_chars = CONTENT_CHAR_BUDGET

    kept_droppable = []
    for m in reversed(droppable):
        c = len(m.get("content", ""))
        if remaining_chars - c >= 0:
            kept_droppable.insert(0, m)
            remaining_chars -= c
        else:
            print(f"[proxy] Dropped assistant turn ({c} chars)")

    kept_priority: list[dict[str, str]] = []
    for i, m in enumerate(priority):
        content = m.get("content", "")
        c = len(content)
        is_last = i == len(priority) - 1
        if not is_last:
            if remaining_chars - c >= 0:
                kept_priority.append(m)
                remaining_chars -= c
            else:
                print(f"[proxy] Dropped {m['role']} turn ({c} chars, budget exhausted)")
        else:
            keep = max(200, min(c, remaining_chars))
            if c > keep:
                print(f"[proxy] Truncated {m['role']} message: {c} → {keep} chars")
            kept_priority.append({**m, "content": content[:keep]})

    ordered = sorted(
        kept_priority + kept_droppable,
        key=lambda m: non_system.index(m) if m in non_system else 999,
    )
    return result + ordered


class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        print(f"[proxy] {format % args}")

    def do_GET(self) -> None:
        self._forward(None)

    def _statement_response(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self) -> None:
        if self.path.startswith("/v1/statement"):
            self._statement_response()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""

        log_entry: dict[str, Any] = {
            "ts": datetime.datetime.now().isoformat(),
            "path": self.path,
        }

        if self.path == "/v1/chat/completions" and body:
            try:
                payload = json.loads(body)
                messages = payload.get("messages", [])
                tools = payload.get("tools", [])
                tool_names = [t.get("function", {}).get("name", "?") for t in tools]

                original_chars = {
                    "system": sum(
                        len(m.get("content", "")) for m in messages if m["role"] == "system"
                    ),
                    "tools": len(json.dumps(tools)),
                    "other": sum(
                        len(m.get("content", "")) for m in messages if m["role"] != "system"
                    ),
                }

                log_entry["original"] = {
                    "messages": messages,
                    "tools": tools,
                    "chars": original_chars,
                }

                print(
                    f"[proxy] Prompt: {sum(original_chars.values())} chars total — "
                    f"system={original_chars['system']} tools={original_chars['tools']} "
                    f"other={original_chars['other']}"
                )
                for i, m in enumerate(messages):
                    content = m.get("content", "")
                    preview = content[:120].replace("\n", "\\n")
                    print(f"[proxy]   [{i}] role={m['role']} chars={len(content)} | {preview!r}")
                print(f"[proxy] Tools ({len(tools)}): {tool_names}")

                payload["messages"] = _rewrite_messages(messages)

                rewritten_chars = sum(len(m.get("content", "")) for m in payload["messages"])
                rewritten_tools_chars = len(json.dumps(payload.get("tools", [])))
                log_entry["rewritten"] = {
                    "messages": payload["messages"],
                    "tools": payload.get("tools", []),
                    "chars": {"messages": rewritten_chars, "tools": rewritten_tools_chars},
                }

                body = json.dumps(payload).encode()
            except Exception as e:
                print(f"[proxy] Parse error: {e}")
                log_entry["error"] = str(e)

        t0 = time.monotonic()
        resp_body = self._forward(body)
        elapsed = time.monotonic() - t0

        if resp_body is not None:
            try:
                log_entry["response"] = json.loads(resp_body)
            except Exception:
                log_entry["response_raw"] = resp_body.decode(errors="replace")[:2000]
        log_entry["elapsed_s"] = round(elapsed, 2)
        _log(log_entry)

    def _forward(self, body: bytes | None) -> bytes | None:
        url = BACKEND + self.path
        headers = {
            k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")
        }
        if body is not None:
            headers["Content-Length"] = str(len(body))

        req = urllib.request.Request(url, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=1200) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() != "transfer-encoding":
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(resp_body)
                return resp_body
        except urllib.error.HTTPError as e:
            body = e.read()
            try:
                self.send_response(e.code)
                self.end_headers()
                self.wfile.write(body)
            except BrokenPipeError:
                pass
            return body
        except BrokenPipeError:
            print("[proxy] Client disconnected (broken pipe)")
            return None
        except Exception as e:
            print(f"[proxy] Backend error: {e}")
            try:
                self.send_response(502)
                self.end_headers()
                self.wfile.write(str(e).encode())
            except BrokenPipeError:
                pass
            return None


if __name__ == "__main__":
    print(f"[proxy] Listening :{PORT} → {BACKEND}  (budget {CONTENT_CHAR_BUDGET} chars)")
    HTTPServer(("127.0.0.1", PORT), ProxyHandler).serve_forever()
