#!/usr/bin/env python3
import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_FEED_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "live_feed_tail.sh"
RUNTIME_DASHBOARD = PROJECT_ROOT / "governance" / "health" / "runtime_gate_dashboard_latest.json"
HEALTH_GATES = PROJECT_ROOT / "governance" / "health" / "health_gates_latest.json"


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>Live Feed Mirror</title>
  <style>
    :root {
      --bg: #08110b;
      --panel: #0d1811;
      --border: #21402b;
      --text: #d7ffd9;
      --muted: #87b992;
      --accent: #49ff81;
      --warn: #ffd166;
      --bad: #ff6b6b;
    }
    html, body {
      margin: 0;
      background: radial-gradient(circle at top, #0e1b12 0%%, var(--bg) 55%%);
      color: var(--text);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      min-height: 100%%;
    }
    .wrap {
      padding: 14px;
      max-width: 1100px;
      margin: 0 auto;
    }
    .card {
      background: rgba(13, 24, 17, 0.95);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
    }
    .header {
      padding: 14px 16px 10px;
      border-bottom: 1px solid rgba(33, 64, 43, 0.75);
    }
    .title {
      font-size: 16px;
      font-weight: 700;
      color: var(--accent);
      margin-bottom: 6px;
    }
    .meta, .statusline {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.45;
    }
    .terminal {
      padding: 12px 14px 18px;
    }
    #terminal {
      margin: 0;
      line-height: 1.35;
      font-size: 12px;
      min-height: 55vh;
      color: var(--text);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      padding: 12px 14px 0;
    }
    button {
      border: 1px solid var(--border);
      background: #122117;
      color: var(--text);
      border-radius: 8px;
      padding: 8px 10px;
      font: inherit;
    }
    .ok { color: var(--accent); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="header">
        <div class="title">All Sleeves Live Feed Mirror</div>
        <div class="meta" id="meta">Connecting...</div>
        <div class="statusline" id="statusline"></div>
      </div>
      <div class="toolbar">
        <button id="refreshBtn" type="button">Reconnect</button>
      </div>
      <div class="terminal">
        <pre id="terminal">Loading live feed...</pre>
      </div>
    </div>
  </div>
  <script>
    const token = new URLSearchParams(window.location.search).get("token") || "";
    const terminalEl = document.getElementById("terminal");
    const metaEl = document.getElementById("meta");
    const statusEl = document.getElementById("statusline");
    const refreshBtn = document.getElementById("refreshBtn");
    let statusFlight = false;
    let eventSource = null;
    let pendingChunks = [];
    let renderScheduled = false;
    let terminalBuffer = "";
    let maxBufferChars = 42000;

    function esc(text) {
      return String(text || "");
    }

    function atBottom() {
      return (window.innerHeight + window.scrollY) >= (document.body.offsetHeight - 120);
    }

    function trimBuffer(limit) {
      if (terminalBuffer.length <= limit) {
        return;
      }
      const overflow = terminalBuffer.length - limit;
      const trimAt = terminalBuffer.indexOf("\n", Math.max(overflow, 0));
      terminalBuffer = trimAt >= 0 ? terminalBuffer.slice(trimAt + 1) : terminalBuffer.slice(-limit);
    }

    function scheduleRender() {
      if (renderScheduled) return;
      renderScheduled = true;
      window.setTimeout(() => {
        renderScheduled = false;
        if (pendingChunks.length === 0) {
          return;
        }
        const shouldFollow = atBottom();
        terminalBuffer += pendingChunks.join("");
        pendingChunks = [];
        trimBuffer(maxBufferChars);
        terminalEl.textContent = terminalBuffer;
        if (shouldFollow) {
          window.scrollTo(0, document.body.scrollHeight);
        }
      }, 180);
    }

    function resetBuffer() {
      pendingChunks = [];
      terminalBuffer = "";
      terminalEl.textContent = "";
    }

    function appendLine(line) {
      pendingChunks.push(`${esc(line)}\n`);
      scheduleRender();
    }

    function appendLines(lines) {
      if (!Array.isArray(lines) || lines.length === 0) return;
      pendingChunks.push(lines.map((line) => esc(line)).join("\n") + "\n");
      scheduleRender();
    }

    async function refreshStatus() {
      if (statusFlight) return;
      statusFlight = true;
      try {
        const params = new URLSearchParams(window.location.search);
        const resp = await fetch(`/api/status?${params.toString()}`, {
          headers: token ? { "X-Live-Feed-Token": token } : {},
          cache: "no-store",
        });
        const payload = await resp.json();
        if (!resp.ok) {
          throw new Error(payload.error || `HTTP ${resp.status}`);
        }
        maxBufferChars = payload.include_decisions ? 36000 : 70000;
        const statusClass =
          payload.dashboard_status === "ok" ? "ok" :
          (payload.dashboard_status === "warn" ? "warn" : "bad");
        metaEl.innerHTML =
          `<span class="${statusClass}">dashboard=${esc(payload.dashboard_status)}</span> ` +
          `health=${esc(payload.data_quality_score)} ` +
          `updated=${new Date().toLocaleTimeString()}`;
        statusEl.textContent =
          `stream=${payload.stream_connected ? "live" : "idle"} ` +
          `source=${payload.source} lines=${payload.lines} include_decisions=${payload.include_decisions ? "1" : "0"} pid=${payload.server_pid}`;
      } catch (err) {
        statusEl.textContent = `status refresh failed: ${err}`;
      } finally {
        statusFlight = false;
      }
    }

    function connectStream() {
      if (eventSource) {
        eventSource.close();
      }
      resetBuffer();
      const params = new URLSearchParams(window.location.search);
      eventSource = new EventSource(`/api/feed/stream?${params.toString()}`);
      eventSource.addEventListener("meta", (event) => {
        try {
          const payload = JSON.parse(event.data);
          maxBufferChars = payload.include_decisions ? 36000 : 70000;
          statusEl.textContent =
            `stream=live source=${payload.source} lines=${payload.lines} include_decisions=${payload.include_decisions ? "1" : "0"} pid=${payload.server_pid}`;
        } catch (err) {
          statusEl.textContent = `stream meta error: ${err}`;
        }
      });
      eventSource.addEventListener("line", (event) => {
        try {
          const payload = JSON.parse(event.data);
          appendLine(payload.line || "");
        } catch (err) {
          appendLine(`stream parse error: ${err}`);
        }
      });
      eventSource.addEventListener("lines", (event) => {
        try {
          const payload = JSON.parse(event.data);
          appendLines(Array.isArray(payload.lines) ? payload.lines : []);
        } catch (err) {
          appendLine(`stream batch parse error: ${err}`);
        }
      });
      eventSource.addEventListener("error", () => {
        statusEl.textContent = "stream disconnected, retrying...";
      });
      eventSource.onopen = () => {
        refreshStatus();
      };
    }

    refreshBtn.addEventListener("click", () => {
      connectStream();
      refreshStatus();
    });
    connectStream();
    refreshStatus();
    setInterval(refreshStatus, 20000);
  </script>
</body>
</html>
"""


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_utc_iso() -> str:
    return _now_utc().isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_loopback_host(host: str) -> bool:
    host_text = str(host or "").strip().lower()
    return host_text in {"127.0.0.1", "localhost", "::1"}


def _effective_token(host: str, token: str) -> str:
    token_text = str(token or "").strip()
    if token_text:
        return token_text
    if _is_loopback_host(host):
        return ""
    return secrets.token_urlsafe(18)


def _candidate_host_ips() -> list[str]:
    candidates: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            ip = str(info[4][0] or "").strip()
            if ip and not ip.startswith("127."):
                candidates.add(ip)
    except Exception:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 80))
            ip = str(sock.getsockname()[0] or "").strip()
            if ip and not ip.startswith("127."):
                candidates.add(ip)
    except Exception:
        pass
    return sorted(candidates)


def _candidate_urls(host: str, port: int, token: str) -> list[str]:
    host_text = str(host or "").strip() or "127.0.0.1"
    query = f"?token={token}" if token else ""
    if host_text in {"0.0.0.0", "::"}:
        urls = [f"http://{ip}:{port}/{query}" for ip in _candidate_host_ips()]
        return urls or [f"http://127.0.0.1:{port}/{query}"]
    return [f"http://{host_text}:{port}/{query}"]


def _stream_profile(include_decisions: bool) -> dict[str, float | int]:
    if include_decisions:
        return {
            "max_line_chars": 320,
            "batch_line_limit": 12,
            "batch_char_limit": 2200,
            "batch_interval_seconds": 0.65,
        }
    return {
        "max_line_chars": 640,
        "batch_line_limit": 10,
        "batch_char_limit": 4200,
        "batch_interval_seconds": 0.35,
    }


def _shape_stream_line(line: str, *, include_decisions: bool) -> str:
    text = str(line or "")
    max_chars = int(_stream_profile(include_decisions).get("max_line_chars", 720) or 720)
    if len(text) <= max_chars:
        return text
    trimmed = max(len(text) - max_chars, 0)
    head = text[: max(max_chars - 20, 1)].rstrip()
    return f"{head} ... [trimmed {trimmed}c]"


def _build_feed_command(*, source: str, lines: int, symbol: str, include_decisions: bool, snapshot: bool) -> list[str]:
    cmd = [
        str(LIVE_FEED_SCRIPT),
        "--source",
        str(source or "all"),
        "--lines",
        str(max(int(lines), 10)),
    ]
    if snapshot:
        cmd.append("--snapshot")
    if symbol:
        cmd.extend(["--symbol", symbol])
    if include_decisions:
        cmd.append("--include-decisions")
    return cmd


def _read_bearer(handler: BaseHTTPRequestHandler) -> str:
    auth = str(handler.headers.get("Authorization", "") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return ""


def _request_token(handler: BaseHTTPRequestHandler) -> str:
    query = parse_qs(urlparse(handler.path).query)
    return (
        str((query.get("token") or [""])[0] or "").strip()
        or str(handler.headers.get("X-Live-Feed-Token", "") or "").strip()
        or _read_bearer(handler)
    )


def _authorized(handler: BaseHTTPRequestHandler, token: str) -> bool:
    if not token:
        return True
    return secrets.compare_digest(_request_token(handler), token)


def _feed_snapshot(*, source: str, lines: int, symbol: str, include_decisions: bool, timeout_seconds: int) -> dict[str, Any]:
    cmd = _build_feed_command(
        source=source,
        lines=lines,
        symbol=symbol,
        include_decisions=include_decisions,
        snapshot=True,
    )
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=max(int(timeout_seconds), 5),
        check=False,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
        "cmd": cmd,
    }


def _status_summary() -> dict[str, Any]:
    runtime = _load_json(RUNTIME_DASHBOARD)
    health = _load_json(HEALTH_GATES)
    overall = runtime.get("overall") if isinstance(runtime.get("overall"), dict) else {}
    return {
        "dashboard_status": str(overall.get("status", "unknown") or "unknown"),
        "dashboard_attention": overall.get("attention") if isinstance(overall.get("attention"), list) else [],
        "data_quality_score": float(health.get("data_quality_score", 0.0) or 0.0),
        "hard_gate_triggered": bool(health.get("hard_gate_triggered", False)),
    }


class _PhoneMirrorHandler(BaseHTTPRequestHandler):
    server_version = "LiveFeedPhoneServer/1.0"

    @property
    def state(self) -> dict[str, Any]:
        return getattr(self.server, "state")

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _write_html(self, status: int, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _write_sse_event(self, event_name: str, payload: dict[str, Any]) -> None:
        body = f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n".encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()

    def _require_auth(self) -> bool:
        token = str(self.state.get("token", "") or "")
        if _authorized(self, token):
            return True
        self._write_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
        return False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self._write_json(HTTPStatus.OK, {"ok": True, "timestamp_utc": _now_utc_iso(), "pid": os.getpid()})
            return

        if not self._require_auth():
            return

        if parsed.path == "/":
            self._write_html(HTTPStatus.OK, HTML_PAGE)
            return

        if parsed.path == "/api/status":
            summary = _status_summary()
            self._write_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "timestamp_utc": _now_utc_iso(),
                    "server_pid": os.getpid(),
                    "source": str(self.state.get("source", "all") or "all"),
                    "lines": int(self.state.get("lines", 80) or 80),
                    "include_decisions": bool(self.state.get("include_decisions", False)),
                    "stream_connected": True,
                    **summary,
                },
            )
            return

        if parsed.path == "/api/feed/stream":
            source = str(self.state.get("source", "all") or "all")
            lines = int(self.state.get("lines", 80) or 80)
            symbol = str(self.state.get("symbol", "") or "")
            include_decisions = bool(self.state.get("include_decisions", False))
            stream_profile = _stream_profile(include_decisions)
            cmd = _build_feed_command(
                source=source,
                lines=lines,
                symbol=symbol,
                include_decisions=include_decisions,
                snapshot=False,
            )
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                self._write_sse_event(
                    "meta",
                    {
                        "source": source,
                        "lines": lines,
                        "symbol": symbol,
                        "include_decisions": include_decisions,
                        "server_pid": os.getpid(),
                    },
                )
                if proc.stdout is not None:
                    batch: list[str] = []
                    batch_chars = 0
                    last_emit = time.monotonic()
                    for raw_line in proc.stdout:
                        line = _shape_stream_line(
                            str(raw_line or "").rstrip("\n"),
                            include_decisions=include_decisions,
                        )
                        batch.append(line)
                        batch_chars += len(line)
                        now = time.monotonic()
                        if (
                            len(batch) >= int(stream_profile.get("batch_line_limit", 12) or 12)
                            or batch_chars >= int(stream_profile.get("batch_char_limit", 2400) or 2400)
                            or (
                            now - last_emit
                        ) >= float(stream_profile.get("batch_interval_seconds", 0.25) or 0.25)
                        ):
                            self._write_sse_event("lines", {"lines": batch})
                            batch = []
                            batch_chars = 0
                            last_emit = now
                    if batch:
                        self._write_sse_event("lines", {"lines": batch})
                rc = proc.wait(timeout=1)
                self._write_sse_event("line", {"line": f"[stream ended rc={rc}]"})
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)
            return

        if parsed.path == "/api/feed":
            source = str(self.state.get("source", "all") or "all")
            lines = int(self.state.get("lines", 80) or 80)
            symbol = str(self.state.get("symbol", "") or "")
            include_decisions = bool(self.state.get("include_decisions", False))
            timeout_seconds = int(self.state.get("timeout_seconds", 12) or 12)
            result = _feed_snapshot(
                source=source,
                lines=lines,
                symbol=symbol,
                include_decisions=include_decisions,
                timeout_seconds=timeout_seconds,
            )
            summary = _status_summary()
            refreshed = datetime.now().astimezone().strftime("%Y-%m-%d %I:%M:%S %p %Z")
            status = HTTPStatus.OK if result["ok"] else HTTPStatus.SERVICE_UNAVAILABLE
            self._write_json(
                status,
                {
                    "ok": bool(result["ok"]),
                    "source": source,
                    "lines": lines,
                    "symbol": symbol,
                    "include_decisions": include_decisions,
                    "output": str(result["stdout"] or result["stderr"] or "").strip(),
                    "stderr": str(result["stderr"] or ""),
                    "returncode": int(result["returncode"]),
                    "refreshed_at_local": refreshed,
                    "server_pid": os.getpid(),
                    **summary,
                },
            )
            return

        self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the all-sleeves live feed in a phone-friendly terminal page.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--source", default="all")
    parser.add_argument("--lines", type=int, default=80)
    parser.add_argument("--symbol", default="")
    parser.add_argument("--include-decisions", action="store_true")
    parser.add_argument("--token", default=os.getenv("LIVE_FEED_PHONE_TOKEN", ""))
    parser.add_argument("--timeout-seconds", type=int, default=12)
    args = parser.parse_args()

    token = _effective_token(args.host, args.token)
    state = {
        "source": str(args.source or "all"),
        "lines": int(max(args.lines, 10)),
        "symbol": str(args.symbol or ""),
        "include_decisions": bool(args.include_decisions),
        "token": token,
        "timeout_seconds": int(max(args.timeout_seconds, 5)),
    }

    server = ThreadingHTTPServer((args.host, int(args.port)), _PhoneMirrorHandler)
    server.state = state

    print("live_feed_phone_server started")
    print(f" source={state['source']} lines={state['lines']} include_decisions={1 if state['include_decisions'] else 0}")
    if token:
        print(" token_protected=1")
    for url in _candidate_urls(args.host, int(args.port), token):
        print(f" url={url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
