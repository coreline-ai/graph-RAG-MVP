"""
Codex CLI Bridge Server

Bridges the app's CodexProxy HTTP contract (POST /generate, GET /health)
to the local Codex CLI using `codex exec`.

Usage:
    python scripts/codex_proxy.py [--port 8800] [--model gpt-5.4]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("codex-proxy")

CODEX_BIN: str | None = None
CLAUDE_BIN: str | None = None
DEFAULT_MODEL = "gpt-5.4"
stats = {"total": 0, "errors": 0, "total_time": 0.0}
stats_lock = threading.Lock()


def find_codex_bin() -> str:
    path = shutil.which("codex")
    if not path:
        log.error("codex CLI not found in PATH")
        sys.exit(1)
    return path


def find_claude_bin() -> str | None:
    path = shutil.which("claude")
    if path:
        return path
    # Known Windows install path
    import glob
    candidates = glob.glob(
        r"C:\Users\*\AppData\Local\Microsoft\WinGet\Packages\Anthropic.ClaudeCode_*\claude.exe"
    )
    if candidates:
        return candidates[0]
    return None


def is_claude_model(model: str) -> bool:
    return model.startswith("claude-")


def call_codex(model: str, prompt: str, timeout: float = 120) -> tuple[str, float]:
    """Run codex exec with the given prompt and return (text, elapsed)."""
    cmd = [CODEX_BIN, "exec", "-c", f'model="{model}"', "-"]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        input=prompt.encode("utf-8"),
        capture_output=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0

    stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
    stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""

    if proc.returncode != 0:
        err = stderr.strip() or stdout.strip() or "unknown error"
        raise RuntimeError(f"codex exec failed (rc={proc.returncode}): {err[:300]}")

    # codex exec outputs:
    #   header lines (model info, session id, etc.)
    #   "--------"
    #   "user"
    #   <prompt echo>
    #   blank line
    #   "codex"
    #   <answer>
    #   blank line
    #   "tokens used"
    #   <number>
    #   <answer repeated>
    lines = stdout.split("\n")

    # Find "codex" marker line — answer starts after it
    codex_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == "codex":
            codex_idx = i
            break

    if codex_idx >= 0:
        # Collect lines after "codex" until "tokens used"
        content_lines = []
        for line in lines[codex_idx + 1:]:
            if line.strip().startswith("tokens used"):
                break
            content_lines.append(line)
        text = "\n".join(content_lines).strip()
    else:
        # Fallback: return everything
        text = stdout.strip()

    if not text:
        log.warning("Empty codex response. stdout=%d stderr=%d", len(stdout), len(stderr))
        raise RuntimeError(f"Empty response from codex. stderr: {stderr[:200]}")

    return text, elapsed


def call_claude(model: str, prompt: str, timeout: float = 120) -> tuple[str, float]:
    """Run claude -p with the given prompt and return (text, elapsed)."""
    if not CLAUDE_BIN:
        raise RuntimeError("claude CLI not found")
    cmd = [CLAUDE_BIN, "-p", "--model", model, "--no-session-persistence"]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        input=prompt.encode("utf-8"),
        capture_output=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0

    stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
    stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""

    if proc.returncode != 0:
        err = stderr.strip() or stdout.strip() or "unknown error"
        raise RuntimeError(f"claude failed (rc={proc.returncode}): {err[:300]}")

    text = stdout.strip()
    if not text:
        raise RuntimeError(f"Empty response from claude. stderr: {stderr[:200]}")

    return text, elapsed


def call_llm(model: str, prompt: str, timeout: float = 120) -> tuple[str, float]:
    """Route to codex or claude based on model name."""
    if is_claude_model(model):
        return call_claude(model, prompt, timeout)
    return call_codex(model, prompt, timeout)


class ProxyHandler(BaseHTTPRequestHandler):
    model = DEFAULT_MODEL

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/health", "/api/v1/health"):
            with stats_lock:
                avg = stats["total_time"] / stats["total"] if stats["total"] else 0
                self._send_json(200, {
                    "status": "ok",
                    "backend": "codex-cli-exec",
                    "model": self.__class__.model,
                    "stats": {
                        "total_calls": stats["total"],
                        "errors": stats["errors"],
                        "avg_latency_s": round(avg, 1),
                    },
                })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(400, {"error": "empty body"})
            return

        try:
            body = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_json(400, {"error": f"invalid JSON: {e}"})
            return

        system_prompt = body.get("system_prompt", "")
        user_prompt = body.get("user_prompt", "")
        if not user_prompt:
            self._send_json(400, {"error": "user_prompt is required"})
            return

        # Combine system + user prompt
        if system_prompt:
            combined = f"[System Instructions]\n{system_prompt}\n[/System Instructions]\n\n{user_prompt}"
        else:
            combined = user_prompt

        model = body.get("model") or self.__class__.model
        log.info("POST /generate -> sys=%d user=%d model=%s",
                 len(system_prompt), len(user_prompt), model)

        try:
            text, elapsed = call_llm(model, combined, timeout=130)
            with stats_lock:
                stats["total"] += 1
                stats["total_time"] += elapsed
            log.info("Response: %.1fs (len=%d)", elapsed, len(text))
        except subprocess.TimeoutExpired:
            with stats_lock:
                stats["total"] += 1
                stats["errors"] += 1
            self._send_json(504, {"error": "codex exec timed out"})
            return
        except RuntimeError as e:
            with stats_lock:
                stats["total"] += 1
                stats["errors"] += 1
            self._send_json(502, {"error": str(e)})
            return
        except Exception as e:
            with stats_lock:
                stats["total"] += 1
                stats["errors"] += 1
            log.exception("Unexpected error")
            self._send_json(500, {"error": str(e)})
            return

        self._send_json(200, {
            "text": text,
            "model": model,
            "finish_reason": "stop",
        })


def main():
    global CODEX_BIN, CLAUDE_BIN

    parser = argparse.ArgumentParser(description="Codex CLI proxy server")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    CODEX_BIN = find_codex_bin()
    CLAUDE_BIN = find_claude_bin()
    if CLAUDE_BIN:
        log.info("Claude CLI found: %s", CLAUDE_BIN)
    else:
        log.warning("Claude CLI not found - claude-* models will be unavailable")
    ProxyHandler.model = args.model

    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    log.info("=" * 42)
    log.info("  Codex CLI Proxy")
    log.info("  http://%s:%d", args.host, args.port)
    log.info("  Model: %s", args.model)
    log.info("  Backend: codex exec (per-request)")
    log.info("=" * 42)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
