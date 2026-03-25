"""
Claude CLI Bridge Server — Stream-JSON Process Pool

Bridges the CodexProxy HTTP contract (POST /generate, GET /health)
to the local Claude CLI using persistent stream-json processes.

Each worker is a long-lived `claude -p --input-format stream-json
--output-format stream-json --verbose` process that stays alive
across requests, eliminating per-call Node.js boot overhead (~5s).

Usage:
    python scripts/claude_proxy.py [--port 8800] [--model sonnet] [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import select
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from queue import Queue, Empty

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("claude-proxy")

CLAUDE_BIN: str | None = None
DEFAULT_MODEL = "sonnet"


def find_claude_bin() -> str:
    path = shutil.which("claude")
    if not path:
        log.error("claude CLI not found in PATH")
        sys.exit(1)
    return path


# ── Stream-JSON Worker ──

class StreamWorker:
    """A persistent Claude CLI process using stream-json protocol."""

    def __init__(self, model: str, system_prompt: str | None = None):
        self.model = model
        self.system_prompt = system_prompt
        self.session_id = str(uuid.uuid4())
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self._spawn()

    def _spawn(self):
        cmd = [
            CLAUDE_BIN, "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--no-session-persistence",
            "--model", self.model,
        ]
        if self.system_prompt:
            cmd.extend(["--system-prompt", self.system_prompt])

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        os.set_blocking(self.proc.stdout.fileno(), False)
        os.set_blocking(self.proc.stderr.fileno(), False)
        self.session_id = str(uuid.uuid4())
        log.info("Spawned worker pid=%d model=%s", self.proc.pid, self.model)

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _ensure_alive(self):
        if not self.is_alive():
            log.warning("Worker pid=%s died, respawning...",
                        self.proc.pid if self.proc else "?")
            self._spawn()

    def call(self, user_prompt: str, timeout: float = 120) -> tuple[str, float]:
        """Send a prompt and wait for the result. Thread-safe via lock."""
        with self.lock:
            self._ensure_alive()
            # Drain any leftover output from previous call
            self._drain()

            msg = json.dumps({
                "type": "user",
                "message": {"role": "user", "content": user_prompt},
                "parent_tool_use_id": None,
                "session_id": self.session_id,
            }) + "\n"

            t0 = time.perf_counter()
            try:
                self.proc.stdin.write(msg.encode())
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                log.warning("Write failed (pid=%d): %s, respawning", self.proc.pid, e)
                self._spawn()
                self.proc.stdin.write(msg.encode())
                self.proc.stdin.flush()

            # Read until we get a result message
            buffer = b""
            start = time.perf_counter()
            while time.perf_counter() - start < timeout:
                ready, _, _ = select.select(
                    [self.proc.stdout, self.proc.stderr], [], [], 0.3
                )
                for fd in ready:
                    data = fd.read(65536)
                    if data:
                        if fd == self.proc.stdout:
                            buffer += data
                        else:
                            # Log stderr but don't fail
                            pass

                # Parse lines looking for result
                text = buffer.decode("utf-8", errors="replace")
                for line in text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if obj.get("type") == "result":
                            elapsed = time.perf_counter() - t0
                            result_text = obj.get("result", "")
                            if obj.get("is_error"):
                                raise RuntimeError(
                                    f"CLI error: {result_text[:300]}"
                                )
                            return result_text, elapsed
                    except json.JSONDecodeError:
                        continue

                # Check if process died
                if self.proc.poll() is not None:
                    raise RuntimeError("CLI process died during request")

            raise TimeoutError(f"No result within {timeout}s")

    def _drain(self):
        """Drain any leftover output from stdout/stderr."""
        for fd in (self.proc.stdout, self.proc.stderr):
            try:
                while True:
                    ready, _, _ = select.select([fd], [], [], 0)
                    if not ready:
                        break
                    data = fd.read(65536)
                    if not data:
                        break
            except Exception:
                pass

    def terminate(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class WorkerPool:
    """Pool of StreamWorker instances. Requests are dispatched round-robin."""

    def __init__(self, size: int, model: str):
        self.size = size
        self.model = model
        self.queue: Queue[StreamWorker] = Queue()
        for _ in range(size):
            w = StreamWorker(model)
            self.queue.put(w)
        log.info("Worker pool ready: %d workers", size)

    def call(self, system_prompt: str, user_prompt: str, timeout: float = 120) -> tuple[str, float]:
        """Acquire a worker, send prompt, return result, release worker."""
        # Combine system_prompt into user_prompt since stream-json workers
        # don't support per-request system prompts (set at spawn time).
        # We prepend as a system instruction block.
        if system_prompt:
            combined = f"[System Instructions]\n{system_prompt}\n[/System Instructions]\n\n{user_prompt}"
        else:
            combined = user_prompt

        worker = self.queue.get(timeout=timeout)
        try:
            result = worker.call(combined, timeout=timeout)
            return result
        finally:
            self.queue.put(worker)

    def shutdown(self):
        while not self.queue.empty():
            try:
                w = self.queue.get_nowait()
                w.terminate()
            except Empty:
                break


# ── Globals ──
pool: WorkerPool | None = None
stats = {"total": 0, "errors": 0, "total_time": 0.0}
stats_lock = threading.Lock()


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
        if self.path == "/health":
            with stats_lock:
                avg = stats["total_time"] / stats["total"] if stats["total"] else 0
                self._send_json(200, {
                    "status": "ok",
                    "backend": "claude-cli-stream-pool",
                    "pool_size": pool.size if pool else 0,
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

        model = self.__class__.model
        log.info("POST /generate → sys=%d user=%d", len(system_prompt), len(user_prompt))

        try:
            text, elapsed = pool.call(system_prompt, user_prompt, timeout=130)
            with stats_lock:
                stats["total"] += 1
                stats["total_time"] += elapsed
            log.info("Response: %.1fs (len=%d)", elapsed, len(text))
        except TimeoutError:
            with stats_lock:
                stats["total"] += 1
                stats["errors"] += 1
            self._send_json(504, {"error": "Claude CLI timed out"})
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
    global CLAUDE_BIN, pool

    parser = argparse.ArgumentParser(description="Claude CLI proxy server (stream-json pool)")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of persistent CLI workers")
    args = parser.parse_args()

    CLAUDE_BIN = find_claude_bin()
    ProxyHandler.model = args.model

    log.info("Spawning %d stream-json workers (model=%s)...", args.workers, args.model)
    pool = WorkerPool(args.workers, args.model)

    server = HTTPServer((args.host, args.port), ProxyHandler)
    log.info("══════════════════════════════════════════")
    log.info("  Claude CLI Proxy (Stream-JSON Pool)")
    log.info("  http://%s:%d", args.host, args.port)
    log.info("  Model: %s | Workers: %d", args.model, args.workers)
    log.info("══════════════════════════════════════════")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        pool.shutdown()
        server.shutdown()


if __name__ == "__main__":
    main()
