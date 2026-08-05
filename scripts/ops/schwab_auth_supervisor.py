#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.provider_access_guard import provider_access_status
    from scripts.brokers.schwab.common import token_needs_refresh, token_status
    from scripts.ops.long_runtime_common import iso_now, load_json, load_recent_jsonl, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.provider_access_guard import provider_access_status
    from scripts.brokers.schwab.common import token_needs_refresh, token_status
    from .long_runtime_common import iso_now, load_json, load_recent_jsonl, write_payload


DEFAULT_TOKEN_PATH = PROJECT_ROOT / "token.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "schwab_auth_supervisor_latest.json"
AUTH_ERROR_MARKERS = (
    "refresh_token_authentication_error",
    "unsupported_token_type",
    "OAuthError",
    "Access Denied",
    "errors.edgesuite.net",
)
CALLBACK_ERROR_MARKERS = (
    "Address already in use",
    "RedirectTimeoutError",
    "Timed out waiting for a post-authorization callback",
)


@dataclass(frozen=True)
class ProcessRow:
    pid: int
    ppid: int
    elapsed_seconds: int
    command: str


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _paper_soak_auth_operable(
    *,
    token: dict[str, Any],
    token_ready: bool,
    readiness_needed: bool,
    min_ready_expires_seconds: float,
    broker_readiness: dict[str, Any],
    auth_lease: dict[str, Any],
) -> bool:
    broker_state = _dict(auth_lease.get("broker_state"))
    lease_budget = _dict(auth_lease.get("lease_budget"))
    broker_preflight = _dict(broker_readiness.get("preflight_checks"))
    expires_in_seconds = max(
        _safe_float(token.get("expires_in_seconds"), 0.0),
        _safe_float(lease_budget.get("expires_in_seconds"), 0.0),
        _safe_float(broker_readiness.get("token_expires_in_seconds"), 0.0),
    )
    ready_floor = max(float(min_ready_expires_seconds), 900.0)
    critical_floor = max(_safe_float(lease_budget.get("critical_lease_seconds"), 0.0), 600.0)
    network_ok = bool(
        broker_readiness.get("network_ok", True) is not False
        and broker_state.get("network_ok", True) is not False
    )
    broker_operable = bool(
        bool(broker_readiness.get("ready_for_open", False))
        or bool(broker_state.get("broker_operable", False))
    )
    configured_for_refresh = bool(
        broker_state.get("configured_for_refresh", True) is not False
        and (bool(token) or bool(broker_preflight.get("token_exists", False)))
    )
    return bool(
        token_ready
        and not readiness_needed
        and expires_in_seconds >= max(ready_floor, critical_floor)
        and network_ok
        and broker_operable
        and configured_for_refresh
    )


def _parse_etime(raw: str) -> int:
    text = str(raw or "").strip()
    if not text:
        return 0
    days = 0
    if "-" in text:
        day_raw, text = text.split("-", 1)
        days = _safe_int(day_raw, 0)
    parts = [_safe_int(part, 0) for part in text.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 1:
        hours = 0
        minutes = 0
        seconds = parts[0]
    else:
        return 0
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _list_auth_processes() -> list[ProcessRow]:
    try:
        proc = subprocess.run(
            ["ps", "ax", "-o", "pid=,ppid=,etime=,command="],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except Exception:
        return []
    rows: list[ProcessRow] = []
    for line in (proc.stdout or "").splitlines():
        if "schwab_auth_refresh.py" not in line:
            continue
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        pid = _safe_int(parts[0], 0)
        if pid <= 0 or pid == os.getpid():
            continue
        rows.append(
            ProcessRow(
                pid=pid,
                ppid=_safe_int(parts[1], 0),
                elapsed_seconds=_parse_etime(parts[2]),
                command=parts[3],
            )
        )
    return rows


def _callback_port_open(host: str, port: int, timeout_seconds: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=max(float(timeout_seconds), 0.05)):
            return True
    except Exception:
        return False


def _read_tail(path: Path, *, max_bytes: int = 256_000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - max_bytes, 0), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _recent_auth_signals(project_root: Path) -> dict[str, Any]:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    events_root = project_root / "governance" / "events"
    logs_root = project_root / "logs"
    event_rows = (
        load_recent_jsonl(events_root / f"auth_events_{day}.jsonl", limit=250)
        + load_recent_jsonl(events_root / f"premarket_token_guard_{day}.jsonl", limit=100)
    )
    texts: list[str] = [json.dumps(row, ensure_ascii=True) for row in event_rows]
    for path in sorted(logs_root.glob(f"*{day}*.log"))[-30:]:
        texts.append(_read_tail(path))
    joined = "\n".join(texts)
    auth_markers = [marker for marker in AUTH_ERROR_MARKERS if marker in joined]
    callback_markers = [marker for marker in CALLBACK_ERROR_MARKERS if marker in joined]
    circuit_breaker_with_auth_error = bool(auth_markers and "CircuitBreaker" in joined and "market_data_error" in joined)
    return {
        "auth_error_markers": sorted(set(auth_markers)),
        "callback_error_markers": sorted(set(callback_markers)),
        "auth_error_count": sum(joined.count(marker) for marker in AUTH_ERROR_MARKERS),
        "callback_error_count": sum(joined.count(marker) for marker in CALLBACK_ERROR_MARKERS),
        "circuit_breaker_with_auth_error": circuit_breaker_with_auth_error,
    }


def _run_json(cmd: list[str], *, cwd: Path, timeout_sec: int = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        rc = int(proc.returncode)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        rc = 124
        timed_out = True
    payload: dict[str, Any] = {}
    for raw in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break
    return {
        "cmd": cmd,
        "rc": rc,
        "timed_out": timed_out,
        "stdout_tail": "\n".join(stdout.splitlines()[-8:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-8:]),
        "payload": payload,
    }


def _kill_process(pid: int) -> dict[str, Any]:
    try:
        os.kill(int(pid), signal.SIGTERM)
        return {"pid": int(pid), "ok": True, "signal": "TERM"}
    except ProcessLookupError:
        return {"pid": int(pid), "ok": True, "signal": "already_gone"}
    except Exception as exc:
        return {"pid": int(pid), "ok": False, "error": f"{type(exc).__name__}:{exc}"}


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    token_path: Path | None = None,
    min_expires_seconds: float = 1500.0,
    min_ready_expires_seconds: float = 900.0,
    callback_host: str = "127.0.0.1",
    callback_port: int = 8182,
    stale_auth_process_seconds: int = 120,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    token_path = (token_path or (project_root / "token.json")).expanduser().resolve()
    token = token_status(token_path)
    refresh_needed, refresh_reason = token_needs_refresh(
        token,
        min_expires_seconds=max(float(min_expires_seconds), 0.0),
        ready_reason="token_ready",
    )
    readiness_needed, readiness_reason = token_needs_refresh(
        token,
        min_expires_seconds=max(float(min_ready_expires_seconds), 0.0),
        ready_reason="token_ready",
    )
    token_ready = not bool(readiness_needed)
    token_expires_in = _safe_float(token.get("expires_in_seconds"), 0.0)

    premarket_guard = load_json(health_root / "premarket_token_guard_latest.json")
    broker_readiness = load_json(health_root / "broker_readiness_latest.json")
    auth_lease = load_json(health_root / "auth_lease_manager_latest.json")
    auth_refresh = load_json(health_root / "schwab_auth_refresh_latest.json")
    provider_access = provider_access_status(project_root, "schwab")
    signals = _recent_auth_signals(project_root)
    processes = _list_auth_processes()
    callback_port_in_use = _callback_port_open(callback_host, int(callback_port))
    stale_processes = [
        row
        for row in processes
        if row.elapsed_seconds >= int(stale_auth_process_seconds)
        or (token_ready and "--force" not in row.command)
    ]

    broker_ready = bool(broker_readiness.get("ready_for_open", premarket_guard.get("ok", False)))
    guard_ok = bool(premarket_guard.get("ok", token_ready))
    auth_lease_status = str(auth_lease.get("overall_status") or "").strip().lower()
    auth_lease_state = str(auth_lease.get("lease_state") or "").strip().lower()
    auth_refresh_reason = str(auth_refresh.get("reason") or "").strip()
    paper_soak_auth_operable = _paper_soak_auth_operable(
        token=token,
        token_ready=bool(token_ready),
        readiness_needed=bool(readiness_needed),
        min_ready_expires_seconds=float(min_ready_expires_seconds),
        broker_readiness=broker_readiness,
        auth_lease=auth_lease,
    )
    if bool(provider_access.get("active", False)):
        paper_soak_auth_operable = False

    status = "ready"
    findings: list[str] = []
    recovered_findings: list[str] = []
    operator_followups: list[str] = []
    repair_plan: list[dict[str, Any]] = []

    if not token_ready:
        status = "blocked"
        findings.append(f"token_not_ready:{readiness_reason}")
        operator_followups.append("./scripts/ops/opsctl.sh token-refresh-interactive --force --json")
    elif refresh_needed:
        findings.append(f"token_refresh_recommended:{refresh_reason}")
        if paper_soak_auth_operable:
            findings.append("token_refresh_watch_paper_soak_ready")
        elif status == "ready":
            status = "degraded"
    if not guard_ok or (broker_readiness and not broker_ready):
        status = "blocked"
        findings.append("broker_readiness_not_ready")
    if bool(provider_access.get("active", False)):
        if status == "ready":
            status = "degraded"
        findings.append(
            f"schwab_provider_cooldown_http_{int(provider_access.get('status_code', 0) or 0)}"
        )
    if auth_lease_status == "blocked" or auth_lease_state == "critical":
        if paper_soak_auth_operable:
            if status == "ready":
                status = "degraded"
            findings.append(f"auth_lease_{auth_lease_state or auth_lease_status}_paper_soak_grace")
        else:
            status = "blocked"
            findings.append(f"auth_lease_{auth_lease_state or auth_lease_status}")
    elif auth_lease_status == "degraded" or auth_lease_state == "warning":
        if paper_soak_auth_operable:
            findings.append(f"auth_lease_{auth_lease_state or auth_lease_status}_paper_soak_grace")
        else:
            if status == "ready":
                status = "degraded"
            findings.append(f"auth_lease_{auth_lease_state or auth_lease_status}")

    if stale_processes:
        if status == "ready":
            status = "degraded"
        findings.append("stale_schwab_auth_refresh_processes")
        repair_plan.append({"name": "cleanup_stale_auth_helpers", "action": "kill_stale_schwab_auth_refresh_processes"})

    if callback_port_in_use and not processes:
        if status == "ready":
            status = "degraded"
        findings.append("callback_port_in_use_by_unknown_process")
    elif callback_port_in_use and stale_processes:
        if status == "ready":
            status = "degraded"
        findings.append("callback_port_held_by_stale_auth_helper")

    active_contract_ok = bool(
        token_ready
        and broker_ready
        and guard_ok
        and auth_lease_status != "blocked"
        and auth_lease_state != "critical"
        and not stale_processes
        and not callback_port_in_use
        and not bool(provider_access.get("active", False))
    )
    if signals["auth_error_markers"]:
        if not token_ready or not broker_ready:
            status = "blocked"
            findings.append("recent_schwab_auth_errors")
        elif not active_contract_ok:
            if status == "ready":
                status = "degraded"
            findings.append("recent_schwab_auth_errors")
        else:
            recovered_findings.append("historical_schwab_auth_errors_after_current_recovery")

    if signals["callback_error_markers"] or auth_refresh_reason.startswith("auth_error:RedirectTimeoutError"):
        if active_contract_ok:
            recovered_findings.append("historical_callback_flow_errors_after_current_recovery")
        elif status == "ready":
            status = "degraded"
            findings.append("recent_callback_flow_errors")
        else:
            findings.append("recent_callback_flow_errors")

    if signals["circuit_breaker_with_auth_error"]:
        if active_contract_ok:
            recovered_findings.append("historical_auth_error_misclassified_as_symbol_data")
        else:
            findings.append("auth_error_misclassified_as_symbol_data")

    repair_plan.extend(
        [
            {
                "name": "refresh_auth_dependent_paper_truth",
                "cmd": ["./scripts/ops/opsctl.sh", "schwab-auth-post-refresh", "--json"],
            },
        ]
    )

    attempts: list[dict[str, Any]] = []
    if apply:
        initial_status = status
        initial_findings = sorted(set(findings))
        for row in stale_processes:
            attempts.append({"action": "kill_stale_auth_helper", **_kill_process(row.pid)})
        attempts.append(
            _run_json(
                ["./scripts/ops/opsctl.sh", "schwab-auth-post-refresh", "--json"],
                cwd=project_root,
                timeout_sec=540,
            )
        )
        failed_attempts = [
            row
            for row in attempts
            if row.get("ok") is False
            or bool(row.get("timed_out", False))
            or ("rc" in row and _safe_int(row.get("rc"), 1) not in {0, 2})
        ]
        refreshed = build_payload(
            project_root,
            apply=False,
            token_path=token_path,
            min_expires_seconds=min_expires_seconds,
            min_ready_expires_seconds=min_ready_expires_seconds,
            callback_host=callback_host,
            callback_port=callback_port,
            stale_auth_process_seconds=stale_auth_process_seconds,
        )
        refreshed["apply"] = True
        refreshed["attempts"] = attempts
        refreshed["post_repair_recheck"] = True
        refreshed["initial_evaluation"] = {
            "overall_status": initial_status,
            "findings": initial_findings,
        }
        cleared = sorted(set(initial_findings) - set(refreshed.get("findings") or []))
        refreshed["recovered_findings"] = sorted(
            set(refreshed.get("recovered_findings") or [])
            | {f"repaired_same_run:{item}" for item in cleared}
        )
        if failed_attempts:
            refreshed["ok"] = False
            refreshed["overall_status"] = "blocked"
            refreshed["findings"] = sorted(set(refreshed.get("findings") or []) | {"supervisor_apply_failed"})
        return refreshed

    summary = (
        f"token_ready={int(token_ready)} expires_in_seconds={round(token_expires_in, 1)} "
        f"broker_ready={int(broker_ready)} auth_lease={auth_lease_state or auth_lease_status or 'unknown'} "
        f"auth_helpers={len(processes)} stale_helpers={len(stale_processes)} callback_port_in_use={int(callback_port_in_use)}"
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": status == "ready",
        "overall_status": status,
        "apply": bool(apply),
        "summary": summary,
        "findings": sorted(set(findings)),
        "recovered_findings": sorted(set(recovered_findings)),
        "token": {
            **token,
            "ready": bool(token_ready),
            "refresh_needed": bool(refresh_needed),
            "refresh_reason": refresh_reason,
            "readiness_refresh_needed": bool(readiness_needed),
            "readiness_reason": readiness_reason,
            "min_expires_seconds": float(min_expires_seconds),
            "min_ready_expires_seconds": float(min_ready_expires_seconds),
        },
        "broker_readiness": {
            "ready_for_open": broker_readiness.get("ready_for_open"),
            "auth_ok": broker_readiness.get("auth_ok"),
            "network_ok": broker_readiness.get("network_ok"),
            "token_expires_in_seconds": broker_readiness.get("token_expires_in_seconds"),
        },
        "auth_lease": {
            "overall_status": auth_lease.get("overall_status"),
            "lease_state": auth_lease.get("lease_state"),
            "lease_budget": auth_lease.get("lease_budget") if isinstance(auth_lease.get("lease_budget"), dict) else {},
        },
        "auth_refresh": {
            "overall_status": auth_refresh.get("overall_status"),
            "ok": auth_refresh.get("ok"),
            "reason": auth_refresh_reason,
            "skipped": auth_refresh.get("skipped"),
        },
        "provider_access": {
            "active": bool(provider_access.get("active", False)),
            "state": provider_access.get("state"),
            "status_code": int(provider_access.get("status_code", 0) or 0),
            "remaining_seconds": int(provider_access.get("remaining_seconds", 0) or 0),
            "cooldown_until_utc": provider_access.get("cooldown_until_utc"),
            "reason": provider_access.get("reason"),
        },
        "callback": {
            "host": callback_host,
            "port": int(callback_port),
            "port_in_use": bool(callback_port_in_use),
        },
        "auth_processes": [
            {
                "pid": row.pid,
                "ppid": row.ppid,
                "elapsed_seconds": row.elapsed_seconds,
                "stale": row in stale_processes,
                "command": row.command,
            }
            for row in processes
        ],
        "recent_auth_signals": signals,
        "repair_plan": repair_plan,
        "attempts": attempts,
        "operator_followups": sorted(set(operator_followups)),
        "regression_contract": {
            "fresh_schwab_token_floor_seconds": float(min_expires_seconds),
            "schwab_token_ready_floor_seconds": float(min_ready_expires_seconds),
            "auth_lease_warning_floor_seconds": 1200,
            "do_not_open_browser_when_token_ready": True,
            "refresh_recommendation_above_ready_floor_is_advisory": True,
            "callback_port_conflict_is_infra_failure": True,
            "oauth_errors_are_broker_auth_failures_not_symbol_failures": True,
            "paper_soak_auth_grace_keeps_live_execution_locked": True,
            "apply_reloads_fresh_artifacts_before_reporting": True,
            "successful_auth_refresh_rebuilds_account_position_and_paper_truth": True,
            "historical_oauth_errors_clear_after_current_contract_proof": True,
        },
        "paper_soak_auth_operable": bool(paper_soak_auth_operable),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervise Schwab auth/token drift and callback-port regressions.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--token-path", default=str(DEFAULT_TOKEN_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--min-expires-seconds", type=float, default=float(os.getenv("SCHWAB_AUTH_MIN_EXPIRES_SECONDS", os.getenv("PREMARKET_TOKEN_MIN_EXPIRES_SECONDS", "1500"))))
    parser.add_argument("--min-ready-expires-seconds", type=float, default=float(os.getenv("SCHWAB_AUTH_READY_MIN_EXPIRES_SECONDS", os.getenv("PREMARKET_TOKEN_READY_MIN_EXPIRES_SECONDS", "900"))))
    parser.add_argument("--callback-host", default=os.getenv("SCHWAB_AUTH_CALLBACK_HOST", "127.0.0.1"))
    parser.add_argument("--callback-port", type=int, default=int(os.getenv("SCHWAB_AUTH_CALLBACK_PORT", "8182")))
    parser.add_argument("--stale-auth-process-seconds", type=int, default=int(os.getenv("SCHWAB_AUTH_STALE_PROCESS_SECONDS", "120")))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(
        project_root,
        apply=bool(args.apply),
        token_path=Path(args.token_path),
        min_expires_seconds=float(args.min_expires_seconds),
        min_ready_expires_seconds=float(args.min_ready_expires_seconds),
        callback_host=str(args.callback_host),
        callback_port=int(args.callback_port),
        stale_auth_process_seconds=int(args.stale_auth_process_seconds),
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"schwab_auth_supervisor status={payload['overall_status']} out={out_path}")
    return 0 if payload["overall_status"] in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
