#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_maintenance import maintenance_hold_snapshot

RUNTIME_ROOT = Path(os.getenv("MAINTENANCE_SLOT_RUNTIME_ROOT", str(PROJECT_ROOT / "runtime" / "maintenance_slots")))
LOCK_ROOT = RUNTIME_ROOT / "locks"
STATE_ROOT = RUNTIME_ROOT / "state"
HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "maintenance_slot_guard_latest.json"
RUNTIME_THROTTLE_HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_throttle_control_latest.json"
EXTERNAL_HEALTH_PATH = Path("/Volumes/BOT_LOGS/schwab_trading_bot/governance/health/maintenance_slot_guard_latest.json")
MACRO_STATUS_CANDIDATES = (
    PROJECT_ROOT / "governance" / "health" / "macro_auto_watch_status.json",
    Path("/Volumes/BOT_LOGS/schwab_trading_bot/governance/health/macro_auto_watch_status.json"),
)

SLOT_MIN_INTERVAL_SECONDS = {
    "one_numbers_refresh": 600,
    "one_numbers_regression_guard": 1800,
    "daily_auto_verify": 7200,
    "grade_regression_autopilot": 1800,
    "section_grade_autopilot": 1800,
    "system_drift_autopilot": 1800,
    "infrastructure_autofix": 1800,
    "project_timeline_autoupdate": 7200,
    "sqlite_maintenance": 14400,
    "storage_pressure_clearance": 1800,
    "storage_backpressure_autopilot": 1800,
    "sql_link_writer": 900,
}
DEFAULT_SMOOTH_GATE_EXEMPT_SLOTS = {
    "sql_link_writer",
    "storage_backpressure_autopilot",
    "storage_pressure_clearance",
    "storage_reconnect_infrabot",
    "storage_eject_guard",
    "runtime_smooth_mode",
    "failover_watch",
    "shadow_watchdog",
    "mac_notification_watch",
    "observability_exporter",
    "premarket_token_guard",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _csv_set(raw: str, default: set[str] | None = None) -> set[str]:
    values = {item.strip() for item in str(raw or "").split(",") if item.strip()}
    return values if values else set(default or set())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        print(f"maintenance_slot_guard warning=status_write_failed:{type(exc).__name__}:{exc}", file=sys.stderr)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _lock_age_seconds(path: Path) -> float | None:
    try:
        return max(time.time() - path.stat().st_mtime, 0.0)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _remove_lock(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    except Exception:
        pass


def _state_path(slot: str) -> Path:
    return STATE_ROOT / f"{slot}.json"


def _slot_min_interval(slot: str, explicit: float | None) -> float:
    if explicit is not None and explicit >= 0:
        return float(explicit)
    env_key = f"MAINTENANCE_SLOT_{slot.upper()}_MIN_INTERVAL_SECONDS"
    if env_key in os.environ:
        return max(_safe_float(os.getenv(env_key), 0.0), 0.0)
    return float(SLOT_MIN_INTERVAL_SECONDS.get(slot, _safe_float(os.getenv("MAINTENANCE_SLOT_MIN_INTERVAL_SECONDS"), 300.0)))


def _cooldown_blocked(slot: str, min_interval_seconds: float) -> tuple[bool, str, dict[str, Any]]:
    if min_interval_seconds <= 0:
        return False, "cooldown_disabled", {}
    state = _read_json(_state_path(slot))
    last_end = _safe_float(state.get("last_end_epoch"), 0.0)
    if last_end <= 0:
        return False, "cooldown_no_prior_run", state
    age = max(time.time() - last_end, 0.0)
    if age < min_interval_seconds:
        return True, f"slot_cooldown_age_seconds={int(age)}<{int(min_interval_seconds)}", state
    return False, f"slot_cooldown_age_seconds={int(age)}", state


def _load_macro_status() -> dict[str, Any]:
    candidates = sorted(
        [path for path in MACRO_STATUS_CANDIDATES if path.exists()],
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    for path in candidates:
        payload = _read_json(path)
        if payload:
            payload["_path"] = str(path)
            return payload
    return {}


def _process_running(needles: tuple[str, ...]) -> bool:
    try:
        import subprocess

        completed = subprocess.run(
            ["ps", "-axo", "command"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return False
    for raw_line in (completed.stdout or "").splitlines():
        line = raw_line.lower()
        if "run_guarded_maintenance.sh" in line:
            continue
        if any(needle.lower() in line for needle in needles):
            return True
    return False


def _macro_event_protected(status: dict[str, Any], *, protect_before_minutes: float, protect_after_minutes: float) -> tuple[bool, str]:
    if not status:
        return False, "macro_status_missing"
    if str(status.get("stream_state") or "") == "live" or bool(status.get("media_ingest_triggered")):
        return True, "macro_stream_active"
    event_raw = str(status.get("calendar_event_time_utc") or "").strip()
    if not event_raw:
        return False, "macro_event_time_missing"
    try:
        event_dt = datetime.fromisoformat(event_raw.replace("Z", "+00:00"))
    except Exception:
        return False, "macro_event_time_unparseable"
    if event_dt.tzinfo is None:
        event_dt = event_dt.replace(tzinfo=timezone.utc)
    delta_minutes = (event_dt.astimezone(timezone.utc).timestamp() - time.time()) / 60.0
    if -max(protect_after_minutes, 0.0) <= delta_minutes <= max(protect_before_minutes, 0.0):
        return True, f"macro_event_window_delta_minutes={delta_minutes:.1f}"
    return False, f"outside_macro_event_window_delta_minutes={delta_minutes:.1f}"


def _host_pressure(max_load_ratio: float, max_five_min_load_ratio: float, max_one_min_load: float | None) -> tuple[bool, dict[str, Any]]:
    cpu_count = max(os.cpu_count() or 1, 1)
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
    except Exception:
        load_1m = load_5m = load_15m = 0.0
    ratio_1m = load_1m / cpu_count
    ratio_5m = load_5m / cpu_count
    blocked = (
        ratio_1m > max_load_ratio
        or ratio_5m > max_five_min_load_ratio
        or (max_one_min_load is not None and load_1m > max_one_min_load)
    )
    return blocked, {
        "cpu_count": cpu_count,
        "load_averages": {"one_minute": round(load_1m, 3), "five_minutes": round(load_5m, 3), "fifteen_minutes": round(load_15m, 3)},
        "load_ratios": {"one_minute": round(ratio_1m, 3), "five_minutes": round(ratio_5m, 3), "fifteen_minutes": round(load_15m / cpu_count, 3)},
        "max_load_ratio": max_load_ratio,
        "max_five_min_load_ratio": max_five_min_load_ratio,
        "max_one_min_load": max_one_min_load,
    }


def _smooth_mode_blocked(
    slot: str,
    *,
    max_saturation_score: float,
    exempt_slots: set[str],
    runtime_path: Path = RUNTIME_THROTTLE_HEALTH_PATH,
) -> tuple[bool, str, dict[str, Any]]:
    normalized_slot = str(slot or "").strip()
    if normalized_slot in exempt_slots:
        return False, "smooth_gate_exempt", {"enabled": True, "exempt": True, "slot": normalized_slot}

    payload = _read_json(runtime_path)
    if not payload:
        return False, "runtime_throttle_missing", {"enabled": True, "runtime_health_path": str(runtime_path), "artifact_present": False}

    mac = payload.get("mac_fluidity_contract") if isinstance(payload.get("mac_fluidity_contract"), dict) else {}
    measurements = mac.get("measurements") if isinstance(mac.get("measurements"), dict) else {}
    governor = payload.get("runtime_saturation_governor_v2") if isinstance(payload.get("runtime_saturation_governor_v2"), dict) else {}
    host_saturation_score = _safe_float(
        payload.get("host_saturation_score"),
        _safe_float(governor.get("host_saturation_score"), _safe_float(measurements.get("host_saturation_score"), 0.0)),
    )
    fluidity_band = str(mac.get("fluidity_band") or os.getenv("MAC_FLUIDITY_BAND", "")).strip().lower()
    fluidity_status = str(mac.get("overall_status") or os.getenv("MAC_FLUIDITY_STATUS", "")).strip().lower()
    compute_pressure = str(payload.get("compute_pressure_level") or measurements.get("compute_pressure_level") or "").strip().lower()
    memory_pressure = str(payload.get("memory_pressure_level") or measurements.get("memory_pressure_level") or "").strip().lower()
    support_pause = bool(mac.get("support_pause_recommended", False)) or _env_flag("MAC_FLUIDITY_SUPPORT_PAUSE", False)

    reason = ""
    if fluidity_band in {"protect", "strained"}:
        reason = f"fluidity_band={fluidity_band}"
    elif fluidity_status == "needs_work":
        reason = "fluidity_status=needs_work"
    elif memory_pressure == "high":
        reason = "memory_pressure=high"
    elif host_saturation_score >= max(float(max_saturation_score), 0.0):
        reason = f"host_saturation_score={host_saturation_score:.2f}>={float(max_saturation_score):.2f}"
    elif compute_pressure == "high" and host_saturation_score >= max(float(max_saturation_score) * 0.85, 1.0):
        reason = f"compute_pressure=high host_saturation_score={host_saturation_score:.2f}"
    elif support_pause:
        reason = "support_pause_recommended"

    snapshot = {
        "enabled": True,
        "exempt": False,
        "slot": normalized_slot,
        "runtime_health_path": str(runtime_path),
        "artifact_present": True,
        "blocked": bool(reason),
        "reason": reason,
        "host_saturation_score": round(host_saturation_score, 3),
        "max_saturation_score": float(max_saturation_score),
        "fluidity_band": fluidity_band,
        "fluidity_status": fluidity_status,
        "compute_pressure_level": compute_pressure,
        "memory_pressure_level": memory_pressure,
        "support_pause_recommended": support_pause,
        "policy": "defer_nonessential_maintenance_when_runtime_smooth_mode_is_strained",
    }
    if reason:
        return True, f"runtime_smooth_gate:{reason}", snapshot
    return False, "runtime_smooth_gate_clear", snapshot


def _in_quiet_window(start_hour: int, end_hour: int) -> tuple[bool, dict[str, Any]]:
    now = datetime.now()
    hour = int(now.hour)
    start = max(min(int(start_hour), 23), 0)
    end = max(min(int(end_hour), 23), 0)
    if start == end:
        in_window = True
    elif start > end:
        in_window = bool(hour >= start or hour < end)
    else:
        in_window = bool(start <= hour < end)
    return in_window, {
        "enabled": True,
        "local_hour": hour,
        "start_hour": start,
        "end_hour": end,
        "in_window": in_window,
    }


def _begin(args: argparse.Namespace) -> int:
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    bundle_lock = LOCK_ROOT / "maintenance_bundle.lock"
    slot_lock = LOCK_ROOT / f"{args.slot}.lock"
    stale_seconds = max(float(args.stale_seconds), 60.0)

    for lock_path in (bundle_lock, slot_lock):
        age = _lock_age_seconds(lock_path)
        if age is not None and age > stale_seconds:
            _remove_lock(lock_path)

    pressure_blocked, pressure = _host_pressure(
        float(args.max_load_ratio),
        float(args.max_five_min_load_ratio),
        None if args.max_one_min_load <= 0 else float(args.max_one_min_load),
    )
    quiet_enabled = bool(args.quiet_windows_enabled)
    quiet_allowed = True
    quiet_payload: dict[str, Any] = {"enabled": False}
    if quiet_enabled:
        quiet_allowed, quiet_payload = _in_quiet_window(int(args.quiet_start_hour), int(args.quiet_end_hour))
    macro_status = _load_macro_status()
    macro_blocked, macro_reason = _macro_event_protected(
        macro_status,
        protect_before_minutes=float(args.protect_macro_before_minutes),
        protect_after_minutes=float(args.protect_macro_after_minutes),
    )
    min_interval_seconds = _slot_min_interval(args.slot, args.min_interval_seconds)
    cooldown_blocked, cooldown_reason, slot_state = _cooldown_blocked(args.slot, min_interval_seconds)
    maintenance_hold = maintenance_hold_snapshot(PROJECT_ROOT)
    reasons: list[str] = []
    if bool(maintenance_hold.get("active", False)):
        reasons.append("runtime_maintenance_hold")
    if pressure_blocked and args.slot != "sql_link_writer":
        reasons.append("host_pressure")
    if quiet_enabled and (not quiet_allowed) and bool(args.defer_outside_quiet_window) and args.slot != "sql_link_writer":
        reasons.append("outside_quiet_window")
    if macro_blocked and not args.allow_during_macro_event:
        reasons.append(macro_reason)
    if cooldown_blocked and args.slot != "sql_link_writer":
        reasons.append(cooldown_reason)
    if bool(args.defer_while_sql_link_active) and args.slot != "sql_link_writer" and _process_running(("scripts/ops/sql_link_shard_manager.py", "scripts/link_jsonl_to_sql.py", "scripts/ops/sql_link_writer_service.py")):
        reasons.append("sql_link_active")
    smooth_gate_payload: dict[str, Any] = {"enabled": bool(args.smooth_gate_enabled)}
    if bool(args.smooth_gate_enabled):
        smooth_blocked, smooth_reason, smooth_gate_payload = _smooth_mode_blocked(
            args.slot,
            max_saturation_score=float(args.smooth_gate_max_saturation_score),
            exempt_slots=_csv_set(str(args.smooth_gate_exempt_slots), DEFAULT_SMOOTH_GATE_EXEMPT_SLOTS),
        )
        if smooth_blocked:
            reasons.append(smooth_reason)
    for label, lock_path in (("bundle", bundle_lock), ("slot", slot_lock)):
        age = _lock_age_seconds(lock_path)
        if age is not None:
            reasons.append(f"{label}_lock_active_age_seconds={int(age)}")

    payload = {
        "timestamp_utc": _now_iso(),
        "slot": args.slot,
        "action": "begin",
        "allowed": not reasons,
        "reasons": reasons,
        "pressure": pressure,
        "quiet_window": quiet_payload,
        "macro": {
            "protected": macro_blocked,
            "reason": macro_reason,
            "status_path": macro_status.get("_path", ""),
            "event_time_utc": macro_status.get("calendar_event_time_utc", ""),
            "stream_state": macro_status.get("stream_state", ""),
        },
        "cooldown": {
            "min_interval_seconds": min_interval_seconds,
            "reason": cooldown_reason,
            "last_end_utc": slot_state.get("last_end_utc", ""),
        },
        "smooth_mode_gate": smooth_gate_payload,
        "runtime_maintenance_hold": maintenance_hold,
        "runtime_root": str(RUNTIME_ROOT),
        "defer_while_sql_link_active": bool(args.defer_while_sql_link_active),
        "pid": os.getpid(),
    }

    if reasons:
        _write_json(HEALTH_PATH, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(f"maintenance_slot_guard skip slot={args.slot} reasons={','.join(reasons)}")
        return int(args.skip_exit_code)

    try:
        bundle_lock.mkdir()
        slot_lock.mkdir()
    except FileExistsError:
        payload["allowed"] = False
        payload["reasons"] = ["lock_race"]
        _write_json(HEALTH_PATH, payload)
        print(f"maintenance_slot_guard skip slot={args.slot} reasons=lock_race")
        return int(args.skip_exit_code)

    lock_payload = {
        "slot": args.slot,
        "pid": os.getpid(),
        "created_utc": _now_iso(),
        "created_epoch": time.time(),
        "command": " ".join(sys.argv),
    }
    for lock_path in (bundle_lock, slot_lock):
        (lock_path / "owner.json").write_text(json.dumps(lock_payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    _write_json(HEALTH_PATH, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"maintenance_slot_guard allow slot={args.slot}")
    return 0


def _end(args: argparse.Namespace) -> int:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    end_epoch = time.time()
    state = _read_json(_state_path(args.slot))
    state.update(
        {
            "slot": args.slot,
            "last_end_epoch": end_epoch,
            "last_end_utc": _now_iso(),
            "last_pid": os.getpid(),
        }
    )
    _write_json(_state_path(args.slot), state)
    for lock_path in (LOCK_ROOT / f"{args.slot}.lock", LOCK_ROOT / "maintenance_bundle.lock"):
        _remove_lock(lock_path)
    payload = {
        "timestamp_utc": _now_iso(),
        "slot": args.slot,
        "action": "end",
        "allowed": True,
        "pid": os.getpid(),
    }
    _write_json(HEALTH_PATH, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard low-priority maintenance launchd jobs from overlapping or running during host pressure.")
    parser.add_argument("--slot", required=True)
    parser.add_argument("--begin", action="store_true")
    parser.add_argument("--end", action="store_true")
    parser.add_argument("--max-load-ratio", type=float, default=_safe_float(os.getenv("MAINTENANCE_SLOT_MAX_LOAD_RATIO"), 0.85))
    parser.add_argument("--max-five-min-load-ratio", type=float, default=_safe_float(os.getenv("MAINTENANCE_SLOT_MAX_FIVE_MIN_LOAD_RATIO"), 0.7))
    parser.add_argument("--max-one-min-load", type=float, default=_safe_float(os.getenv("MAINTENANCE_SLOT_MAX_ONE_MIN_LOAD"), 0.0))
    parser.add_argument("--min-interval-seconds", type=float, default=None)
    parser.add_argument("--stale-seconds", type=float, default=_safe_float(os.getenv("MAINTENANCE_SLOT_STALE_SECONDS"), 1800.0))
    parser.add_argument("--protect-macro-before-minutes", type=float, default=_safe_float(os.getenv("MAINTENANCE_SLOT_PROTECT_MACRO_BEFORE_MINUTES"), 180.0))
    parser.add_argument("--protect-macro-after-minutes", type=float, default=_safe_float(os.getenv("MAINTENANCE_SLOT_PROTECT_MACRO_AFTER_MINUTES"), 75.0))
    parser.add_argument("--allow-during-macro-event", action="store_true")
    parser.add_argument("--defer-while-sql-link-active", action=argparse.BooleanOptionalAction, default=os.getenv("MAINTENANCE_SLOT_DEFER_WHILE_SQL_LINK_ACTIVE", "1").strip().lower() not in {"0", "false", "no", "off"})
    parser.add_argument("--quiet-windows-enabled", action=argparse.BooleanOptionalAction, default=os.getenv("MAINTENANCE_SLOT_QUIET_WINDOWS_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--defer-outside-quiet-window", action=argparse.BooleanOptionalAction, default=os.getenv("MAINTENANCE_SLOT_DEFER_OUTSIDE_QUIET_WINDOW", "0").strip().lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--quiet-start-hour", type=int, default=_safe_int(os.getenv("MAINTENANCE_SLOT_QUIET_LOCAL_START_HOUR"), 21))
    parser.add_argument("--quiet-end-hour", type=int, default=_safe_int(os.getenv("MAINTENANCE_SLOT_QUIET_LOCAL_END_HOUR"), 6))
    parser.add_argument("--smooth-gate-enabled", action=argparse.BooleanOptionalAction, default=_env_flag("MAINTENANCE_SLOT_SMOOTH_GATE_ENABLED", False))
    parser.add_argument("--smooth-gate-max-saturation-score", type=float, default=_safe_float(os.getenv("MAINTENANCE_SLOT_SMOOTH_GATE_MAX_SATURATION_SCORE"), 68.0))
    parser.add_argument(
        "--smooth-gate-exempt-slots",
        default=os.getenv("MAINTENANCE_SLOT_SMOOTH_GATE_EXEMPT_SLOTS", ",".join(sorted(DEFAULT_SMOOTH_GATE_EXEMPT_SLOTS))),
    )
    parser.add_argument("--skip-exit-code", type=int, default=_safe_int(os.getenv("MAINTENANCE_SLOT_SKIP_EXIT_CODE"), 75))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.begin == args.end:
        raise SystemExit("pass exactly one of --begin or --end")
    if args.begin:
        return _begin(args)
    return _end(args)


if __name__ == "__main__":
    raise SystemExit(main())
