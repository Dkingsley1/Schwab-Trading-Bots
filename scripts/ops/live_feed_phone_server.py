#!/usr/bin/env python3
import argparse
import json
import os
import secrets
import select
import shutil
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
QUANT_MODEL_CONTROL = PROJECT_ROOT / "governance" / "health" / "quant_model_control_latest.json"
MEMORY_EFFICIENCY = PROJECT_ROOT / "governance" / "health" / "memory_efficiency_control_latest.json"
GLOBAL_KILLSWITCH = PROJECT_ROOT / "governance" / "health" / "global_killswitch_latest.json"
SPACEX_IPO_WATCH = PROJECT_ROOT / "governance" / "health" / "spacex_ipo_downside_watch_latest.json"
MACRO_EVENT_INTELLIGENCE = PROJECT_ROOT / "governance" / "health" / "macro_event_intelligence_latest.json"
LIVE_MACRO = PROJECT_ROOT / "data" / "external_context" / "live_macro_latest.json"
MAC_NOTIFICATION_STATE = PROJECT_ROOT / "governance" / "health" / "mac_notification_watch_state.json"
REMOTE_ALERT_CONTROL = PROJECT_ROOT / "governance" / "health" / "remote_alert_control_latest.json"
PROCESS_WATCHDOG = PROJECT_ROOT / "governance" / "health" / "process_watchdog_latest.json"
INGESTION_STORAGE_CONTROL = PROJECT_ROOT / "governance" / "health" / "ingestion_storage_control_latest.json"
RUNTIME_THROTTLE_CONTROL = PROJECT_ROOT / "governance" / "health" / "runtime_throttle_control_latest.json"
PDF_REPORT_ROOT = PROJECT_ROOT / "output" / "pdf"


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
    button, input, a.reportlink {
      border: 1px solid var(--border);
      background: #122117;
      color: var(--text);
      border-radius: 8px;
      padding: 8px 10px;
      font: inherit;
    }
    a.reportlink {
      display: inline-flex;
      align-items: center;
      min-height: 18px;
      text-decoration: none;
    }
    input {
      min-width: 180px;
      flex: 1 1 180px;
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
        <div class="statusline" id="systemline"></div>
      </div>
      <div class="toolbar">
        <button id="refreshBtn" type="button">Reconnect</button>
        <a id="reportLink" class="reportlink" href="/reports/latest-system-update.pdf" target="_blank" rel="noopener">Open PDF</a>
        <input id="tokenInput" type="password" placeholder="feed token" autocomplete="off" autocapitalize="none" spellcheck="false" />
        <button id="tokenBtn" type="button">Use Token</button>
      </div>
      <div class="statusline" id="tokenline"></div>
      <div class="terminal">
        <pre id="terminal">Loading live feed...</pre>
      </div>
    </div>
  </div>
  <script>
    const params = new URLSearchParams(window.location.search);
    let token = params.get("token") || window.localStorage.getItem("live_feed_phone_token") || "";
    const terminalEl = document.getElementById("terminal");
    const metaEl = document.getElementById("meta");
    const statusEl = document.getElementById("statusline");
    const systemEl = document.getElementById("systemline");
    const tokenEl = document.getElementById("tokenline");
    const refreshBtn = document.getElementById("refreshBtn");
    const reportLink = document.getElementById("reportLink");
    const tokenInput = document.getElementById("tokenInput");
    const tokenBtn = document.getElementById("tokenBtn");
    tokenInput.value = token;
    let statusFlight = false;
    let snapshotFlight = false;
    let eventSource = null;
    let pendingChunks = [];
    let renderScheduled = false;
    let terminalBuffer = "";
    let maxBufferChars = 42000;
    let reconnectTimer = null;
    let reconnectAttempts = 0;
    let lastStreamEventMs = Date.now();
    let streamHasData = false;
    const reconnectBaseDelayMs = 1500;
    const staleReconnectMs = 25000;

    function esc(text) {
      return String(text || "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[ch]));
    }

    function atBottom() {
      return (window.innerHeight + window.scrollY) >= (document.body.offsetHeight - 120);
    }

    function trimBuffer(limit) {
      if (terminalBuffer.length <= limit) {
        return;
      }
      const overflow = terminalBuffer.length - limit;
      const trimAt = terminalBuffer.indexOf("\\n", Math.max(overflow, 0));
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
      streamHasData = false;
    }

    function appendLine(line) {
      pendingChunks.push(`${esc(line)}\\n`);
      scheduleRender();
    }

    function appendLines(lines) {
      if (!Array.isArray(lines) || lines.length === 0) return;
      pendingChunks.push(lines.map((line) => esc(line)).join("\\n") + "\\n");
      scheduleRender();
    }

    function replaceBuffer(text) {
      const shouldFollow = atBottom();
      terminalBuffer = String(text || "").replace(/\\r\\n/g, "\\n");
      if (terminalBuffer && !terminalBuffer.endsWith("\\n")) {
        terminalBuffer += "\\n";
      }
      trimBuffer(maxBufferChars);
      terminalEl.textContent = terminalBuffer || "Waiting for live feed...";
      if (shouldFollow) {
        window.scrollTo(0, document.body.scrollHeight);
      }
    }

    function markStreamActivity() {
      lastStreamEventMs = Date.now();
      reconnectAttempts = 0;
    }

    function pageRequiresToken() {
      const host = String(window.location.hostname || "").toLowerCase();
      return !["127.0.0.1", "localhost", "::1", "[::1]"].includes(host);
    }

    function persistToken(nextToken) {
      token = String(nextToken || "").trim();
      tokenInput.value = token;
      if (token) {
        window.localStorage.setItem("live_feed_phone_token", token);
      } else {
        window.localStorage.removeItem("live_feed_phone_token");
      }
      const nextParams = new URLSearchParams(window.location.search);
      if (token) {
        nextParams.set("token", token);
      } else {
        nextParams.delete("token");
      }
      const nextQuery = nextParams.toString();
      const nextUrl = `${window.location.pathname}${nextQuery ? `?${nextQuery}` : ""}`;
      window.history.replaceState({}, "", nextUrl);
      updateReportLink();
    }

    function authParams(extra = {}) {
      const params = new URLSearchParams(window.location.search);
      if (token && !params.get("token")) {
        params.set("token", token);
      }
      Object.entries(extra).forEach(([key, value]) => params.set(key, String(value)));
      return params;
    }

    function updateReportLink() {
      const params = authParams();
      const query = params.toString();
      reportLink.href = `/reports/latest-system-update.pdf${query ? `?${query}` : ""}`;
    }

    function setTokenMessage(message, level = "warn") {
      tokenEl.className = `statusline ${level}`;
      tokenEl.textContent = message;
    }

    function clearReconnectTimer() {
      if (reconnectTimer) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    }

    function scheduleReconnect(reason, delayMs = reconnectBaseDelayMs) {
      clearReconnectTimer();
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      statusEl.textContent = reason;
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        connectStream({ preserveBuffer: true });
        refreshStatus();
      }, Math.max(delayMs, 250));
    }

    async function refreshStatus() {
      if (statusFlight) return;
      if (pageRequiresToken() && !token) {
        setTokenMessage("Paste the phone-feed token from the terminal output, then tap Use Token.", "warn");
        statusEl.textContent = "token required before status can load";
        return;
      }
      statusFlight = true;
      try {
        const params = authParams();
        const resp = await fetch(`/api/status?${params.toString()}`, {
          headers: token ? { "X-Live-Feed-Token": token } : {},
          cache: "no-store",
        });
        const payload = await resp.json();
        if (resp.status === 401) {
          setTokenMessage("Token rejected. Paste the latest phone-feed token from the terminal and retry.", "bad");
          statusEl.textContent = "authorization required";
          return;
        }
        if (!resp.ok) {
          throw new Error(payload.error || `HTTP ${resp.status}`);
        }
        setTokenMessage(token ? "token loaded" : "loopback mode: token not required", "ok");
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
        const spcxQuote = payload.spcx_quote_error ? payload.spcx_quote_error : "ok";
        const notifyState = payload.imessage_ready ? "iMessage" : "no_iMessage";
        const spcxAlert = payload.spcx_alert_triggered ? "alert" : "watch";
        systemEl.textContent =
          `halt=${payload.halt_state || "unknown"} mode=${payload.operating_mode || "unknown"} ` +
          `quant=${payload.quant_status || "unknown"} q_pressure=${payload.quant_resource_pressure || 0} ` +
          `memory=${payload.memory_profile || "unknown"} | ` +
          `spcx=${payload.spcx_status || "unknown"} quote=${spcxQuote} ${spcxAlert} ` +
          `macro=${payload.macro_relevance || "unknown"}/${payload.macro_calendar_status || "unknown"} ` +
          `notify=${notifyState} watchdog=${payload.watchdog_status || "unknown"} issues=${payload.watchdog_active_issues || 0}`;
      } catch (err) {
        statusEl.textContent = `status refresh failed: ${err}`;
      } finally {
        statusFlight = false;
      }
    }

    async function loadSnapshot(options = {}) {
      if (snapshotFlight) return;
      const replace = Boolean(options.replace);
      if (pageRequiresToken() && !token) {
        if (replace && terminalBuffer.length === 0) {
          replaceBuffer("Token required before snapshot can load.");
        }
        return;
      }
      snapshotFlight = true;
      try {
        const params = authParams();
        const resp = await fetch(`/api/feed?${params.toString()}`, {
          headers: token ? { "X-Live-Feed-Token": token } : {},
          cache: "no-store",
        });
        const payload = await resp.json();
        if (resp.status === 401) {
          setTokenMessage("Token rejected. Paste the latest phone-feed token from the terminal and retry.", "bad");
          statusEl.textContent = "authorization required";
          if (replace && terminalBuffer.length === 0) {
            replaceBuffer("Snapshot authorization failed.");
          }
          return;
        }
        if (!resp.ok) {
          throw new Error(payload.error || `HTTP ${resp.status}`);
        }
        const output = String(payload.output || "").trim();
        if (output) {
          streamHasData = true;
          if (replace || terminalBuffer.length === 0) {
            replaceBuffer(output);
          }
        } else if (replace && terminalBuffer.length === 0) {
          replaceBuffer("Waiting for live feed...");
        }
      } catch (err) {
        if (replace && terminalBuffer.length === 0) {
          replaceBuffer(`Snapshot load failed: ${err}`);
        }
      } finally {
        snapshotFlight = false;
      }
    }

    function connectStream(options = {}) {
      const preserveBuffer = Boolean(options.preserveBuffer);
      clearReconnectTimer();
      if (eventSource) {
        eventSource.close();
      }
      eventSource = null;
      if (!preserveBuffer) {
        resetBuffer();
      }
      if (pageRequiresToken() && !token) {
        setTokenMessage("Paste the phone-feed token from the terminal output, then tap Use Token.", "warn");
        statusEl.textContent = "token required before stream can connect";
        return;
      }
      markStreamActivity();
      const params = authParams({ stream_nonce: String(Date.now()) });
      eventSource = new EventSource(`/api/feed/stream?${params.toString()}`);
      eventSource.addEventListener("meta", (event) => {
        try {
          const payload = JSON.parse(event.data);
          maxBufferChars = payload.include_decisions ? 36000 : 70000;
          markStreamActivity();
          statusEl.textContent =
            `stream=live source=${payload.source} lines=${payload.lines} include_decisions=${payload.include_decisions ? "1" : "0"} pid=${payload.server_pid}`;
        } catch (err) {
          statusEl.textContent = `stream meta error: ${err}`;
        }
      });
      eventSource.addEventListener("line", (event) => {
        try {
          const payload = JSON.parse(event.data);
          markStreamActivity();
          streamHasData = true;
          appendLine(payload.line || "");
        } catch (err) {
          appendLine(`stream parse error: ${err}`);
        }
      });
      eventSource.addEventListener("lines", (event) => {
        try {
          const payload = JSON.parse(event.data);
          markStreamActivity();
          streamHasData = true;
          appendLines(Array.isArray(payload.lines) ? payload.lines : []);
        } catch (err) {
          appendLine(`stream batch parse error: ${err}`);
        }
      });
      eventSource.addEventListener("ping", () => {
        markStreamActivity();
      });
      eventSource.addEventListener("error", () => {
        if (pageRequiresToken() && !token) {
          setTokenMessage("Paste the phone-feed token from the terminal output, then tap Use Token.", "warn");
          statusEl.textContent = "token required before stream can reconnect";
          return;
        }
        reconnectAttempts += 1;
        const delayMs = Math.min(10000, reconnectBaseDelayMs * reconnectAttempts);
        if (!streamHasData || terminalBuffer.length === 0) {
          loadSnapshot({ replace: true });
        }
        scheduleReconnect("stream disconnected, retrying...", delayMs);
      });
      eventSource.onopen = () => {
        markStreamActivity();
        refreshStatus();
      };
    }

    function applyToken() {
      persistToken(tokenInput.value);
      setTokenMessage(token ? "token saved, reconnecting..." : "token cleared", token ? "ok" : "warn");
      connectStream({ preserveBuffer: false });
      loadSnapshot({ replace: true });
      refreshStatus();
    }

    refreshBtn.addEventListener("click", () => {
      connectStream({ preserveBuffer: false });
      loadSnapshot({ replace: true });
      refreshStatus();
    });
    tokenBtn.addEventListener("click", applyToken);
    tokenInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        applyToken();
      }
    });
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) {
        scheduleReconnect("refreshing stream after wake...", 50);
      }
    });
    window.addEventListener("pageshow", () => {
      scheduleReconnect("refreshing stream after page restore...", 50);
    });
    window.addEventListener("online", () => {
      scheduleReconnect("network restored, reconnecting...", 50);
    });
    window.setInterval(() => {
      const idleMs = Date.now() - lastStreamEventMs;
      if (idleMs > staleReconnectMs) {
        loadSnapshot({ replace: terminalBuffer.length === 0 });
        scheduleReconnect(`stream stale for ${Math.round(idleMs / 1000)}s, reconnecting...`, 50);
      }
    }, 5000);
    if (pageRequiresToken()) {
      setTokenMessage(token ? "token loaded" : "Paste the phone-feed token from the terminal output, then tap Use Token.", token ? "ok" : "warn");
    } else {
      setTokenMessage("loopback mode: token not required", "ok");
    }
    updateReportLink();
    loadSnapshot({ replace: true });
    connectStream({ preserveBuffer: false });
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


def _latest_system_update_pdf(report_dir: Path = PDF_REPORT_ROOT) -> Path | None:
    try:
        candidates = [path for path in report_dir.glob("schwab_system_update_*.pdf") if path.is_file()]
    except Exception:
        return None

    def mtime(path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except Exception:
            return 0.0

    candidates.sort(key=mtime, reverse=True)
    return candidates[0] if candidates else None


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


def _tailscale_status() -> dict[str, Any]:
    tailscale_bin = shutil.which("tailscale")
    if not tailscale_bin:
        return {}
    try:
        proc = subprocess.run(
            [tailscale_bin, "status", "--json"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except Exception:
        return {}
    if proc.returncode != 0:
        return {}
    try:
        payload = json.loads(proc.stdout or "{}")
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _tailscale_candidate_urls(port: int, token: str) -> list[str]:
    status = _tailscale_status()
    query = f"?token={token}" if token else ""
    self_node = status.get("Self") if isinstance(status.get("Self"), dict) else {}
    urls: list[str] = []
    dns_name = str(self_node.get("DNSName", "") or "").strip().rstrip(".")
    if dns_name:
        urls.append(f"http://{dns_name}:{int(port)}/{query}")
    for ip in self_node.get("TailscaleIPs") or []:
        ip_text = str(ip or "").strip()
        if not ip_text:
            continue
        if ":" in ip_text:
            urls.append(f"http://[{ip_text}]:{int(port)}/{query}")
        else:
            urls.append(f"http://{ip_text}:{int(port)}/{query}")
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url not in seen:
            deduped.append(url)
            seen.add(url)
    return deduped


def _stream_profile(include_decisions: bool) -> dict[str, float | int]:
    if include_decisions:
        return {
            "max_line_chars": 320,
            "batch_line_limit": 12,
            "batch_char_limit": 2200,
            "batch_interval_seconds": 0.65,
            "heartbeat_seconds": 8.0,
            "retry_millis": 2500,
        }
    return {
        "max_line_chars": 640,
        "batch_line_limit": 10,
        "batch_char_limit": 4200,
        "batch_interval_seconds": 0.35,
        "heartbeat_seconds": 10.0,
        "retry_millis": 2200,
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
    quant = _load_json(QUANT_MODEL_CONTROL)
    memory = _load_json(MEMORY_EFFICIENCY)
    killswitch = _load_json(GLOBAL_KILLSWITCH)
    spcx = _load_json(SPACEX_IPO_WATCH)
    macro_intel = _load_json(MACRO_EVENT_INTELLIGENCE)
    live_macro = _load_json(LIVE_MACRO)
    mac_notification = _load_json(MAC_NOTIFICATION_STATE)
    remote_alert = _load_json(REMOTE_ALERT_CONTROL)
    process_watchdog = _load_json(PROCESS_WATCHDOG)
    storage = _load_json(INGESTION_STORAGE_CONTROL)
    throttle = _load_json(RUNTIME_THROTTLE_CONTROL)

    def intish(value: Any) -> int:
        try:
            return int(float(value or 0))
        except Exception:
            return 0

    def floatish(value: Any) -> float:
        try:
            return float(value or 0.0)
        except Exception:
            return 0.0

    overall = runtime.get("overall") if isinstance(runtime.get("overall"), dict) else {}
    quant_features = quant.get("features") if isinstance(quant.get("features"), dict) else {}
    spcx_quote = spcx.get("quote") if isinstance(spcx.get("quote"), dict) else {}
    spcx_alert = spcx.get("alert") if isinstance(spcx.get("alert"), dict) else {}
    macro_calendar = (
        macro_intel.get("calendar_verification")
        if isinstance(macro_intel.get("calendar_verification"), dict)
        else {}
    )
    macro_items = live_macro.get("items") if isinstance(live_macro.get("items"), list) else []
    macro_item = macro_items[0] if macro_items and isinstance(macro_items[0], dict) else {}
    remote_backlog = (
        remote_alert.get("critical_backlog")
        if isinstance(remote_alert.get("critical_backlog"), dict)
        else {}
    )
    remote_channels = remote_alert.get("channels") if isinstance(remote_alert.get("channels"), dict) else {}
    watchdog_intel = (
        process_watchdog.get("watchdog_intelligence")
        if isinstance(process_watchdog.get("watchdog_intelligence"), dict)
        else {}
    )
    return {
        "dashboard_status": str(overall.get("status", "unknown") or "unknown"),
        "dashboard_attention": overall.get("attention") if isinstance(overall.get("attention"), list) else [],
        "data_quality_score": floatish(health.get("data_quality_score", 0.0)),
        "hard_gate_triggered": bool(health.get("hard_gate_triggered", False)),
        "halt_state": str(killswitch.get("halt_state") or "unknown"),
        "operating_mode": str(killswitch.get("operating_mode") or "unknown"),
        "expansion_pressure_score": floatish(killswitch.get("expansion_pressure_score", 0.0)),
        "quant_status": str(quant.get("overall_status") or "unknown"),
        "quant_resource_pressure": round(floatish(quant_features.get("quant_model_resource_pressure_norm", 0.0)), 3),
        "memory_profile": str(memory.get("recommended_profile") or "unknown"),
        "spcx_status": str(spcx.get("overall_status") or "unknown"),
        "spcx_symbol": str(spcx.get("symbol") or "SPCX"),
        "spcx_quote_error": str(spcx_quote.get("error") or ""),
        "spcx_alert_triggered": bool(spcx_alert.get("triggered", False)),
        "spcx_policy": str(spcx.get("policy") or ""),
        "macro_status": str(macro_intel.get("overall_status") or "unknown"),
        "macro_relevance": str(macro_intel.get("market_relevance") or "unknown"),
        "macro_calendar_status": str(macro_calendar.get("status") or "unknown"),
        "macro_headline": str(macro_item.get("headline") or live_macro.get("headline") or ""),
        "imessage_ready": bool(
            mac_notification.get("imessage_enabled")
            and mac_notification.get("imessage_recipient_configured")
        ),
        "imessage_min_severity": str(mac_notification.get("imessage_min_severity") or "unknown"),
        "remote_alert_status": str(remote_alert.get("overall_status") or "unknown"),
        "remote_alert_imessage": bool(remote_channels.get("imessage_bridge", False)),
        "remote_alert_unsent_count": intish(remote_backlog.get("unsent_count", 0)),
        "remote_alert_unacked_count": intish(remote_backlog.get("unacked_count", 0)),
        "watchdog_status": str(process_watchdog.get("overall_status") or "unknown"),
        "watchdog_grade": str(watchdog_intel.get("grade") or ""),
        "watchdog_active_issues": intish(watchdog_intel.get("active_issue_count", 0)),
        "storage_status": str(storage.get("overall_status") or "unknown"),
        "storage_pressure": floatish(storage.get("pressure_index", 0.0)),
        "throttle_status": str(throttle.get("overall_status") or "unknown"),
        "throttle_profile": str(throttle.get("throttle_profile") or "unknown"),
        "phone_mirror_profile": "expanded_system_safe",
    }


class _PhoneMirrorServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, request: object, client_address: object) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


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

    def _write_file(self, status: int, path: Path, content_type: str, filename: str) -> None:
        try:
            payload = path.read_bytes()
        except OSError:
            self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "report_not_found"})
            return
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Content-Disposition", f'inline; filename="{filename}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _write_sse_event(self, event_name: str, payload: dict[str, Any]) -> None:
        body = f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n".encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()

    def _write_sse_retry(self, retry_millis: int) -> None:
        body = f"retry: {max(int(retry_millis), 250)}\n\n".encode("utf-8")
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

        if parsed.path == "/":
            self._write_html(HTTPStatus.OK, HTML_PAGE)
            return

        if not self._require_auth():
            return

        if parsed.path == "/reports/latest-system-update.pdf":
            report_path = _latest_system_update_pdf()
            if report_path is None:
                self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "report_not_found"})
                return
            self._write_file(
                HTTPStatus.OK,
                report_path,
                "application/pdf",
                "schwab_system_update_latest.pdf",
            )
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
            retry_millis = int(stream_profile.get("retry_millis", 2500) or 2500)
            heartbeat_seconds = float(stream_profile.get("heartbeat_seconds", 10.0) or 10.0)
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
                self.send_header("Cache-Control", "no-store, no-transform")
                self.send_header("Connection", "keep-alive")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                self._write_sse_retry(retry_millis)
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
                    while True:
                        ready, _, _ = select.select([proc.stdout], [], [], heartbeat_seconds)
                        now = time.monotonic()
                        if ready:
                            raw_line = proc.stdout.readline()
                            if raw_line == "":
                                if batch:
                                    self._write_sse_event("lines", {"lines": batch})
                                    batch = []
                                    batch_chars = 0
                                if proc.poll() is not None:
                                    break
                                continue
                            line = _shape_stream_line(
                                str(raw_line or "").rstrip("\n"),
                                include_decisions=include_decisions,
                            )
                            batch.append(line)
                            batch_chars += len(line)
                            if (
                                len(batch) >= int(stream_profile.get("batch_line_limit", 12) or 12)
                                or batch_chars >= int(stream_profile.get("batch_char_limit", 2400) or 2400)
                                or (now - last_emit) >= float(stream_profile.get("batch_interval_seconds", 0.25) or 0.25)
                            ):
                                self._write_sse_event("lines", {"lines": batch})
                                batch = []
                                batch_chars = 0
                                last_emit = now
                            continue
                        if batch:
                            self._write_sse_event("lines", {"lines": batch})
                            batch = []
                            batch_chars = 0
                            last_emit = now
                        self._write_sse_event(
                            "ping",
                            {"timestamp_utc": _now_utc_iso(), "server_pid": os.getpid()},
                        )
                        if proc.poll() is not None:
                            break
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

    server = _PhoneMirrorServer((args.host, int(args.port)), _PhoneMirrorHandler)
    server.state = state

    print("live_feed_phone_server started")
    print(f" source={state['source']} lines={state['lines']} include_decisions={1 if state['include_decisions'] else 0}")
    if token:
        print(" token_protected=1")
    for url in _candidate_urls(args.host, int(args.port), token):
        print(f" url={url}")
    tailscale_status = _tailscale_status()
    backend_state = str(tailscale_status.get("BackendState", "") or "").strip()
    tailscale_urls = _tailscale_candidate_urls(int(args.port), token) if not _is_loopback_host(args.host) else []
    if backend_state:
        print(f" tailscale_state={backend_state}")
    if tailscale_urls and backend_state.lower() == "running":
        for url in tailscale_urls:
            print(f" remote_url={url}")
    elif tailscale_urls:
        print(" remote_url_unavailable_reason=tailscale_stopped")
        for url in tailscale_urls:
            print(f" remote_url_candidate={url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
