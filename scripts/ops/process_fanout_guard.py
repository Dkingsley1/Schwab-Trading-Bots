#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, parse_iso_utc, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from .long_runtime_common import iso_now, load_json, parse_iso_utc, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "process_fanout_guard_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "process_fanout_guard_state.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.process_fanout_guard_override"
DEFAULT_PROJECT_MARKER = str(PROJECT_ROOT)
TARGETABLE_MARKERS = (
    "scripts/run_shadow_training_loop.py",
    "scripts/run_specialized_sleeve_shadow.py",
    "scripts/run_parallel_aggressive_modes.py",
    "scripts/run_parallel_shadows.py",
    "scripts/run_all_sleeves.py",
)
PROTECTED_MARKERS = (
    "scripts/run_execution_lane.py",
    "scripts/ops/process_fanout_guard.py",
    "scripts/ops/process_watchdog.py",
    "scripts/shadow_watchdog.py",
    "scripts/ops/schwab_auth_supervisor.py",
    "scripts/ops/premarket_token_guard.py",
    "scripts/ops/sql_link_shard_manager.py",
    "scripts/ops/sql_link_writer_service.py",
    "scripts/ops/run_sql_link_writer_launchd.sh",
    "scripts/ops/storage_",
)
PRESSURE_CORE_SLEEVE_MARKERS = (
    "scripts/run_all_sleeves.py",
    "scripts/run_parallel_shadows.py",
    "scripts/run_dividend_shadow.py",
    "scripts/run_bond_shadow.py",
    "scripts/run_fx_shadow.py",
)
PRESSURE_CORE_PROFILES = {
    "aggressive",
    "bond",
    "conservative",
    "default",
    "dividend",
    "fx",
    "schwab_futures",
}


@dataclass
class ProcRow:
    pid: int
    ppid: int
    cpu_percent: float
    rss_mb: float
    etimes: int
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


def _env_float(name: str, default: float) -> float:
    return _safe_float(os.getenv(name), default)


def _env_int(name: str, default: int) -> int:
    return _safe_int(os.getenv(name), default)


def _env_csv(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _run_capture(command: list[str], *, timeout: float = 5.0) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return ""
    return completed.stdout or ""


def collect_processes(*, project_marker: str = DEFAULT_PROJECT_MARKER) -> list[ProcRow]:
    text = _run_capture(["ps", "-axo", "pid,ppid,pcpu,rss,etimes,command"])
    rows: list[ProcRow] = []
    marker = str(project_marker or "").strip()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("pid "):
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        pid, ppid, cpu, rss_kb, etimes, command = parts
        if marker and marker not in command:
            continue
        rows.append(
            ProcRow(
                pid=_safe_int(pid),
                ppid=_safe_int(ppid),
                cpu_percent=round(_safe_float(cpu), 3),
                rss_mb=round(_safe_float(rss_kb) / 1024.0, 3),
                etimes=_safe_int(etimes),
                command=command,
            )
        )
    return [row for row in rows if row.pid > 0]


def _profile(command: str) -> str:
    parts = shlex.split(command)
    for idx, part in enumerate(parts):
        if part == "--profile" and idx + 1 < len(parts):
            return parts[idx + 1].strip().lower()
    return "default"


def _broker(command: str) -> str:
    parts = shlex.split(command)
    for idx, part in enumerate(parts):
        if part == "--broker" and idx + 1 < len(parts):
            return parts[idx + 1].strip().lower()
    return ""


def _is_protected(row: ProcRow, keep_profiles: set[str]) -> bool:
    command = row.command
    if any(marker in command for marker in PROTECTED_MARKERS):
        return True
    if _env_flag("PROCESS_FANOUT_GUARD_CORE_SLEEVE_RESTART_ALLOWED", "0"):
        if any(marker in command for marker in PRESSURE_CORE_SLEEVE_MARKERS):
            return True
        if "scripts/run_shadow_training_loop.py" in command and _broker(command) == "schwab" and _profile(command) in PRESSURE_CORE_PROFILES:
            return True
    if "scripts/run_shadow_training_loop.py" in command:
        if _broker(command) != "schwab":
            return True
        if _profile(command) in keep_profiles:
            return True
    return False


def _is_targetable(row: ProcRow, keep_profiles: set[str]) -> bool:
    if _is_protected(row, keep_profiles):
        return False
    return any(marker in row.command for marker in TARGETABLE_MARKERS)


def _priority(row: ProcRow) -> tuple[int, float, int]:
    command = row.command
    if "scripts/run_all_sleeves.py" in command or "scripts/run_parallel_aggressive_modes.py" in command:
        base = 0
    elif "scripts/run_parallel_shadows.py" in command:
        base = 1
    elif "scripts/run_specialized_sleeve_shadow.py" in command:
        base = 2
    elif "scripts/run_shadow_training_loop.py" in command:
        base = 3
    else:
        base = 5
    return (base, -float(row.rss_mb), -int(row.etimes))


def _status(triggered: bool, terminated_count: int) -> str:
    if triggered and terminated_count <= 0:
        return "degraded"
    if triggered:
        return "active"
    return "ready"


def _write_override(
    path: Path,
    *,
    active: bool,
    mode: str,
    reason: str,
    max_count: int,
    target_count: int,
    max_rss_mb: float,
    target_rss_mb: float,
    core_sleeve_restart_allowed: bool = False,
) -> bool:
    values = {
        "PROCESS_FANOUT_GUARD_ACTIVE": "1" if active else "0",
        "PROCESS_FANOUT_GUARD_CORE_SLEEVE_RESTART_ALLOWED": "1" if core_sleeve_restart_allowed else "0",
        "PROCESS_FANOUT_GUARD_MAX_COUNT": str(int(max_count)),
        "PROCESS_FANOUT_GUARD_MAX_RSS_MB": f"{float(max_rss_mb):.1f}",
        "PROCESS_FANOUT_GUARD_MODE": mode,
        "PROCESS_FANOUT_GUARD_REASON": reason,
        "PROCESS_FANOUT_GUARD_TARGET_COUNT": str(int(target_count)),
        "PROCESS_FANOUT_GUARD_TARGET_RSS_MB": f"{float(target_rss_mb):.1f}",
        "TRAINING_RUNTIME_PAUSED_FOR_FANOUT": "1" if active else "0",
        "SHADOW_RESEARCH_PAUSED_FOR_FANOUT": "1" if active else "0",
        "OPS_WATCHDOG_ALL_SLEEVES_WITH_AGGRESSIVE": "0" if active else "1",
        "RUN_ALL_SLEEVES_WITH_DIVIDEND_CAPTURE": "0" if active else "1",
        "RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES": "0" if active else "1",
        "SHADOW_LOOP_INTERVAL": "60" if active else os.getenv("SHADOW_LOOP_INTERVAL", "15"),
        "DIVIDEND_SHADOW_INTERVAL": "120" if active else os.getenv("DIVIDEND_SHADOW_INTERVAL", "60"),
        "BOND_SHADOW_INTERVAL": "180" if active else os.getenv("BOND_SHADOW_INTERVAL", "120"),
        "SPECIALIZED_SLEEVE_INTERVAL": "300" if active else os.getenv("SPECIALIZED_SLEEVE_INTERVAL", "120"),
        "SLEEVE_WORKERS_BASELINE": "1" if active else os.getenv("SLEEVE_WORKERS_BASELINE", os.getenv("ASYNC_PIPELINE_WORKERS", "4")),
        "SLEEVE_WORKERS_DIVIDEND": "1" if active else os.getenv("SLEEVE_WORKERS_DIVIDEND", "2"),
        "SLEEVE_WORKERS_BOND": "1" if active else os.getenv("SLEEVE_WORKERS_BOND", "2"),
    }
    lines = ["# Auto-managed by scripts/ops/process_fanout_guard.py"]
    lines.extend(f"{key}={shlex.quote(value)}" for key, value in sorted(values.items()))
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def build_payload(
    *,
    apply: bool = False,
    clear_hold: bool = False,
    project_marker: str = DEFAULT_PROJECT_MARKER,
    out_path: Path = DEFAULT_OUT_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    max_count = _env_int("PROCESS_FANOUT_GUARD_MAX_COUNT", 120)
    target_count = _env_int("PROCESS_FANOUT_GUARD_TARGET_COUNT", 80)
    max_rss_mb = _env_float("PROCESS_FANOUT_GUARD_MAX_RSS_MB", 5120.0)
    target_rss_mb = _env_float("PROCESS_FANOUT_GUARD_TARGET_RSS_MB", 4096.0)
    hold_seconds = _env_int("PROCESS_FANOUT_GUARD_HOLD_SECONDS", 600)
    keep_profiles = set(_env_csv("PROCESS_FANOUT_GUARD_KEEP_PROFILES", "schwab_futures,fx,dividend,bond"))
    state = load_json(state_path)
    preserve_clear_cooldown = _env_flag("PROCESS_FANOUT_GUARD_PRESERVE_CLEAR_COOLDOWN", "0")
    previous_hold_until = None if clear_hold or not preserve_clear_cooldown else parse_iso_utc(state.get("hold_until_utc"))

    rows = collect_processes(project_marker=project_marker)
    targetable = [row for row in rows if _is_targetable(row, keep_profiles)]
    protected = [row for row in rows if _is_protected(row, keep_profiles)]
    total_rss_mb = round(sum(row.rss_mb for row in rows), 3)
    targetable_rss_mb = round(sum(row.rss_mb for row in targetable), 3)
    triggered = bool(len(rows) > max_count or total_rss_mb > max_rss_mb)
    if triggered:
        hold_until = now + timedelta(seconds=max(hold_seconds, 0))
    else:
        hold_until = previous_hold_until if previous_hold_until and previous_hold_until > now else None
    hold_active = bool(hold_until is not None and hold_until > now)
    override_active = bool(triggered or hold_active)

    kill_plan: list[ProcRow] = []
    projected_count = len(rows)
    projected_rss = total_rss_mb
    if triggered:
        for row in sorted(targetable, key=_priority):
            if projected_count <= target_count and projected_rss <= target_rss_mb:
                break
            kill_plan.append(row)
            projected_count -= 1
            projected_rss = round(projected_rss - row.rss_mb, 3)
    core_sleeve_restart_allowed = bool(triggered and not targetable and not kill_plan)

    terminated: list[dict[str, Any]] = []
    if apply:
        for row in kill_plan:
            try:
                os.kill(row.pid, signal.SIGTERM)
                terminated.append({"pid": row.pid, "ok": True, "rss_mb": row.rss_mb, "command": row.command[:500]})
            except ProcessLookupError:
                terminated.append({"pid": row.pid, "ok": True, "rss_mb": row.rss_mb, "command": row.command[:500], "note": "already_exited"})
            except Exception as exc:
                terminated.append({"pid": row.pid, "ok": False, "rss_mb": row.rss_mb, "command": row.command[:500], "error": str(exc)})

    reason = "fanout_pressure" if triggered else "within_budget"
    override_changed = _write_override(
        override_path,
        active=override_active,
        mode="trimming" if triggered else ("cooldown_hold" if hold_active else "observe"),
        reason="fanout_cooldown_hold" if hold_active and not triggered else reason,
        max_count=max_count,
        target_count=target_count,
        max_rss_mb=max_rss_mb,
        target_rss_mb=target_rss_mb,
        core_sleeve_restart_allowed=core_sleeve_restart_allowed,
    )
    write_payload(
        state_path,
        {
            "timestamp_utc": now.isoformat(),
            "hold_until_utc": hold_until.isoformat() if hold_until else "",
            "last_triggered_utc": now.isoformat() if triggered else str(state.get("last_triggered_utc") or ""),
            "last_process_count": len(rows),
            "last_total_rss_mb": total_rss_mb,
        },
    )
    terminated_count = sum(1 for row in terminated if bool(row.get("ok", False)))
    payload = {
        "timestamp_utc": iso_now(),
        "overall_status": _status(triggered, terminated_count if apply else len(kill_plan)),
        "ok": not triggered or bool(kill_plan) or not apply,
        "apply": bool(apply),
        "thresholds": {
            "max_count": max_count,
            "target_count": target_count,
            "max_rss_mb": max_rss_mb,
            "target_rss_mb": target_rss_mb,
            "hold_seconds": hold_seconds,
            "keep_profiles": sorted(keep_profiles),
        },
        "fanout": {
            "process_count": len(rows),
            "total_rss_mb": total_rss_mb,
            "targetable_count": len(targetable),
            "targetable_rss_mb": targetable_rss_mb,
            "protected_count": len(protected),
        },
        "triggered": triggered,
        "override": {
            "path": str(override_path),
            "changed": override_changed,
            "active": override_active,
            "hold_active": hold_active,
            "hold_until_utc": hold_until.isoformat() if hold_until else "",
            "hold_cleared": bool(clear_hold),
        },
        "startup_policy": {
            "core_sleeve_restart_allowed": core_sleeve_restart_allowed,
            "all_sleeves_aggressive_enabled": not override_active,
            "all_sleeves_specialized_enabled": not override_active,
            "all_sleeves_dividend_capture_enabled": not override_active,
            "reason": "no_targetable_sleeve_processes_to_trim" if core_sleeve_restart_allowed else reason,
        },
        "kill_plan": [
            {
                "pid": row.pid,
                "rss_mb": row.rss_mb,
                "cpu_percent": row.cpu_percent,
                "profile": _profile(row.command),
                "broker": _broker(row.command),
                "command": row.command[:500],
            }
            for row in kill_plan
        ],
        "terminated_count": terminated_count,
        "terminated": terminated,
        "top_processes": [
            {
                "pid": row.pid,
                "rss_mb": row.rss_mb,
                "cpu_percent": row.cpu_percent,
                "profile": _profile(row.command),
                "broker": _broker(row.command),
                "protected": _is_protected(row, keep_profiles),
                "command": row.command[:500],
            }
            for row in sorted(rows, key=lambda item: item.rss_mb, reverse=True)[:20]
        ],
        "recommended_actions": [
            "process_fanout_guard wrote a runtime override to stop aggressive all-sleeves relaunch while pressure is active"
            if triggered
            else "process fanout is within configured budget",
            "run with --apply to terminate optional Schwab shadow research workers"
            if triggered and not apply
            else "optional Schwab shadow research workers were trimmed" if terminated_count else "no process trim was needed",
        ],
    }
    write_payload(out_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Govern schwab_trading_bot process fanout under memory pressure.")
    parser.add_argument("--apply", action="store_true", help="Terminate optional Schwab shadow/research workers until under target.")
    parser.add_argument("--clear-hold", action="store_true", help="Clear an active fanout cooldown hold before recomputing the override.")
    parser.add_argument("--json", action="store_true", help="Print JSON payload.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--override", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--project-marker", default=DEFAULT_PROJECT_MARKER)
    args = parser.parse_args()

    payload = build_payload(
        apply=bool(args.apply),
        clear_hold=bool(args.clear_hold),
        project_marker=str(args.project_marker),
        out_path=Path(args.out),
        state_path=Path(args.state),
        override_path=Path(args.override),
    )
    if args.json:
        import json

        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        fanout = payload["fanout"]
        print(
            f"status={payload['overall_status']} count={fanout['process_count']} "
            f"rss_mb={fanout['total_rss_mb']:.1f} terminated={payload['terminated_count']}"
        )


if __name__ == "__main__":
    main()
