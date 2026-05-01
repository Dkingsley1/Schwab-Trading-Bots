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
RUNTIME_ROOT = Path(os.getenv("MAINTENANCE_SLOT_RUNTIME_ROOT", "/tmp/schwab_trading_bot/maintenance_slots"))
LOCK_ROOT = RUNTIME_ROOT / "locks"
STATE_ROOT = RUNTIME_ROOT / "state"
HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "maintenance_slot_guard_latest.json"
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    candidates = [path]
    if path != EXTERNAL_HEALTH_PATH:
        candidates.append(EXTERNAL_HEALTH_PATH)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            tmp = candidate.with_suffix(candidate.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
            tmp.replace(candidate)
            return
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        print(f"maintenance_slot_guard warning=status_write_failed:{type(last_error).__name__}:{last_error}", file=sys.stderr)


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
    macro_status = _load_macro_status()
    macro_blocked, macro_reason = _macro_event_protected(
        macro_status,
        protect_before_minutes=float(args.protect_macro_before_minutes),
        protect_after_minutes=float(args.protect_macro_after_minutes),
    )
    min_interval_seconds = _slot_min_interval(args.slot, args.min_interval_seconds)
    cooldown_blocked, cooldown_reason, slot_state = _cooldown_blocked(args.slot, min_interval_seconds)
    reasons: list[str] = []
    if pressure_blocked:
        reasons.append("host_pressure")
    if macro_blocked and not args.allow_during_macro_event:
        reasons.append(macro_reason)
    if cooldown_blocked:
        reasons.append(cooldown_reason)
    if bool(args.defer_while_sql_link_active) and args.slot != "sql_link_writer" and _process_running(("scripts/ops/sql_link_shard_manager.py", "scripts/link_jsonl_to_sql.py", "scripts/ops/sql_link_writer_service.py")):
        reasons.append("sql_link_active")
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
