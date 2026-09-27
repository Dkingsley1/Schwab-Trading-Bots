#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_maintenance import maintenance_hold_snapshot
from core.workload_admission import POLICIES, SLOT_WORKLOADS, current_lease, fresh, mapping
from scripts.ops.long_runtime_common import _process_tree_targets, _signal_process_tree_targets

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


def _pid_is_running(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _reap_abandoned_lock(path: Path, *, stale_seconds: float, owner_grace_seconds: float = 5.0) -> bool:
    age = _lock_age_seconds(path)
    if age is None:
        return False
    owner = _read_json(path / "owner.json") if path.is_dir() else {}
    owner_pid = _safe_int(owner.get("pid"), 0)
    owner_dead = owner_pid > 0 and not _pid_is_running(owner_pid)
    # Age alone cannot revoke ownership, and missing ownership is not proof of death.
    if not owner_dead:
        return False
    _remove_lock(path)
    return not path.exists()


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
        values = (load_1m, load_5m, load_15m, max_load_ratio, max_five_min_load_ratio)
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("invalid_load_observation_or_limit")
        if max_one_min_load is not None and (not math.isfinite(max_one_min_load) or max_one_min_load < 0):
            raise ValueError("invalid_absolute_load_limit")
    except Exception:
        return True, {"measurement_available": False, "reason": "load_observation_or_policy_unavailable"}
    ratio_1m = load_1m / cpu_count
    ratio_5m = load_5m / cpu_count
    blocked = (
        ratio_1m > max_load_ratio
        or ratio_5m > max_five_min_load_ratio
        or (max_one_min_load is not None and load_1m > max_one_min_load)
    )
    return blocked, {
        "measurement_available": True,
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


def _adaptive_slot_policy(args: argparse.Namespace) -> dict[str, Any]:
    workload = SLOT_WORKLOADS.get(args.slot)
    runtime = _read_json(RUNTIME_THROTTLE_HEALTH_PATH)
    lease = runtime.get("workload_admission", {})
    now = datetime.now(timezone.utc)
    active = bool(
        getattr(args, "execute", False) and getattr(args, "adaptive", True)
        and workload and current_lease(lease, workload, now=now)
        and runtime.get("input_evidence_ready") is True
        and runtime.get("protective_hold") is False
        and fresh(runtime.get("source_timestamp_utc"), now)
        and mapping(runtime.get("adaptive_safety_limits")).get("active") is False
        and runtime.get("memory_pressure_level") == "normal"
        and mapping(runtime.get("mac_fluidity_contract")).get("support_pause_recommended") is False
    )
    return {
        "active": active,
        "workload": workload or "legacy_heavy",
        "policy": dict(POLICIES[workload]) if workload else {},
        "source_timestamp_utc": lease.get("source_timestamp_utc") if isinstance(lease, dict) else None,
        "workload_reasons": mapping(mapping(mapping(lease).get("workloads")).get(workload)).get("reasons", []),
        "reason": "fresh_workload_lease" if active else "no_extra_admission",
    }


def _bundle_name(slot: str) -> str:
    return "observer_bundle" if SLOT_WORKLOADS.get(slot) == "observer" else "maintenance_bundle"


def _health_path(slot: str) -> Path:
    if SLOT_WORKLOADS.get(slot) == "observer":
        return HEALTH_PATH.with_name("maintenance_observer_latest.json")
    return HEALTH_PATH


def _begin(args: argparse.Namespace) -> int:
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    bundle_lock = LOCK_ROOT / f"{_bundle_name(args.slot)}.lock"
    slot_lock = LOCK_ROOT / f"{args.slot}.lock"
    stale_seconds = max(float(args.stale_seconds), 60.0)

    for lock_path in (bundle_lock, slot_lock):
        _reap_abandoned_lock(lock_path, stale_seconds=stale_seconds)

    adaptive = _adaptive_slot_policy(args)
    args.adaptive_admission = adaptive
    load_limit = adaptive["policy"].get("load", 0) if adaptive["active"] else 0
    pressure_blocked, pressure = _host_pressure(
        max(float(args.max_load_ratio), load_limit),
        max(float(args.max_five_min_load_ratio), load_limit),
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
    if adaptive["active"] and args.min_interval_seconds is None and f"MAINTENANCE_SLOT_{args.slot.upper()}_MIN_INTERVAL_SECONDS" not in os.environ:
        prior = _read_json(_state_path(args.slot))
        duration = max(_safe_float(prior.get("last_duration_seconds"), 0), 0)
        adaptive_interval = 60 if adaptive["workload"] == "observer" else max(300, duration * 3)
        min_interval_seconds = min(min_interval_seconds, adaptive_interval)
    cooldown_blocked, cooldown_reason, slot_state = _cooldown_blocked(args.slot, min_interval_seconds)
    maintenance_hold = maintenance_hold_snapshot(PROJECT_ROOT)
    reasons: list[str] = []
    if pressure.get("measurement_available") is False:
        reasons.append("load_evidence_unavailable")
    if args.slot == "infrastructure_observe" and not adaptive["active"]:
        reasons.append("observer_workload_lease_not_ready")
    bounded_risk_refresh = False
    if args.slot == "one_numbers_refresh" and os.getenv("ONE_NUMBERS_BOUNDED_RISK_REFRESH") == "1":
        from scripts.ops.one_numbers_refresh_policy import bounded_refresh_admitted

        bounded_risk_refresh = bounded_refresh_admitted(PROJECT_ROOT, _read_json(RUNTIME_THROTTLE_HEALTH_PATH))
    if bool(maintenance_hold.get("active", False)):
        reasons.append("runtime_maintenance_hold")
    if pressure_blocked and args.slot != "sql_link_writer" and not bounded_risk_refresh:
        reasons.append("host_pressure")
    if quiet_enabled and (not quiet_allowed) and bool(args.defer_outside_quiet_window) and args.slot != "sql_link_writer" and not (adaptive["active"] and adaptive["workload"] == "observer"):
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
        bounded_risk_label_only = (
            bounded_risk_refresh
            and smooth_reason == "runtime_smooth_gate:fluidity_band=protect"
            and smooth_gate_payload.get("support_pause_recommended") is False
        )
        if smooth_blocked and not bounded_risk_label_only and not (adaptive["active"] and adaptive["workload"] == "observer"):
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
        "bounded_overdue_risk_refresh": bounded_risk_refresh,
        "adaptive_admission": adaptive,
        "resource_pool": _bundle_name(args.slot),
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
        _write_json(_health_path(args.slot), payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(f"maintenance_slot_guard skip slot={args.slot} reasons={','.join(reasons)}")
        return int(args.skip_exit_code)

    created_locks = []
    try:
        for lock_path in (bundle_lock, slot_lock):
            lock_path.mkdir()
            created_locks.append(lock_path)
    except FileExistsError:
        for lock_path in created_locks:
            lock_path.rmdir()
        payload["allowed"] = False
        payload["reasons"] = ["lock_race"]
        _write_json(_health_path(args.slot), payload)
        print(f"maintenance_slot_guard skip slot={args.slot} reasons=lock_race")
        return int(args.skip_exit_code)

    lock_payload = {
        "slot": args.slot,
        "pid": os.getpid() if getattr(args, "execute", False) else os.getppid(),
        "created_utc": _now_iso(),
        "created_epoch": time.time(),
        "command": args.slot if getattr(args, "execute", False) else " ".join(sys.argv),
    }
    for lock_path in (bundle_lock, slot_lock):
        (lock_path / "owner.json").write_text(json.dumps(lock_payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    _write_json(_health_path(args.slot), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"maintenance_slot_guard allow slot={args.slot}")
    return 0


def _end(args: argparse.Namespace) -> int:
    for path in (LOCK_ROOT / f"{args.slot}.lock", LOCK_ROOT / f"{_bundle_name(args.slot)}.lock"):
        owner = _read_json(path / "owner.json")
        pid = _safe_int(owner.get("pid"), 0)
        if path.exists() and (owner.get("slot") != args.slot or pid <= 0 or (pid not in (os.getpid(), os.getppid()) and _pid_is_running(pid))):
            return int(getattr(args, "skip_exit_code", 75))
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
    for lock_path in (LOCK_ROOT / f"{args.slot}.lock", LOCK_ROOT / f"{_bundle_name(args.slot)}.lock"):
        _remove_lock(lock_path)
    payload = {
        "timestamp_utc": _now_iso(),
        "slot": args.slot,
        "action": "end",
        "allowed": True,
        "pid": os.getpid(),
    }
    _write_json(_health_path(args.slot), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    return 0


def _adaptive_continue(args: argparse.Namespace) -> bool:
    current = _adaptive_slot_policy(args)
    args.admission_recheck = {"allowed": False, "reason": "workload_lease_expired_or_withheld", "lease": current}
    if not current["active"]:
        return False
    policy = args.adaptive_admission["policy"]
    if maintenance_hold_snapshot(PROJECT_ROOT).get("active", True):
        args.admission_recheck["reason"] = "maintenance_hold"
        return False
    if any(path.exists() for path in (
        PROJECT_ROOT / "OPERATOR_STOP.flag",
        PROJECT_ROOT / "governance/health/OPERATOR_STOP.flag",
        PROJECT_ROOT / "RUNTIME_MAINTENANCE_HOLD.flag",
    )):
        args.admission_recheck["reason"] = "operator_hold"
        return False
    try:
        free_gib = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
        args.admission_recheck["local_free_gib"] = free_gib
        if free_gib < policy["disk"]:
            args.admission_recheck["reason"] = "disk_reserve"
            return False
    except OSError:
        args.admission_recheck["reason"] = "disk_measurement_unavailable"
        return False
    blocked, pressure = _host_pressure(policy["load"], policy["load"], None if args.max_one_min_load <= 0 else args.max_one_min_load)
    args.admission_recheck.update(allowed=not blocked, reason="host_load" if blocked else "admitted", pressure=pressure)
    return not blocked


def _stop_owned_group(proc: subprocess.Popen, grace: float = 2.0) -> None:
    targets = _process_tree_targets(proc.pid) if proc.poll() is None else (set(), {proc.pid})
    _signal_process_tree_targets(*targets, signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Also reap remaining same-group children if the leader exited first.
    _signal_process_tree_targets(*targets, signal.SIGKILL)
    proc.wait(timeout=max(grace, 0.1))


def _execute(args: argparse.Namespace) -> int:
    lease_wait = float(getattr(args, "lease_wait_seconds", 0))
    if not math.isfinite(lease_wait) or not 0 <= lease_wait <= 120:
        raise ValueError("lease_wait_seconds must be between 0 and 120")
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("maintenance command is required")
    if args.slot == "infrastructure_observe" and command[-4:] != [
        str(PROJECT_ROOT / "scripts/ops/infrastructure_autofix_bot.py"),
        "--timeout-sec", "45", "--json",
    ]:
        raise ValueError("observer_slot_requires_fixed_assessment_command")
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    # This kernel lease spans admission, execution, and cleanup; its inode is never removed.
    fd = os.open(LOCK_ROOT / f"{_bundle_name(args.slot)}.flock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a+") as lease:
        wait_started = time.monotonic()
        wait_announced = False
        while True:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = lease_wait - (time.monotonic() - wait_started)
                if remaining <= 0:
                    print(
                        f"maintenance_slot_guard skip slot={args.slot} reasons=kernel_lease_busy"
                    )
                    return int(args.skip_exit_code)
                if not wait_announced:
                    print(
                        f"maintenance_slot_guard waiting slot={args.slot} max_seconds={lease_wait}",
                        flush=True,
                    )
                    wait_announced = True
                time.sleep(min(0.25, remaining))
        waited_seconds = round(time.monotonic() - wait_started, 3)
        # Admission is evaluated only after ownership, never reused from before a wait.
        rc = _begin(args)
        if rc:
            return rc
        started = time.monotonic()
        proc = None
        result = "failed"
        active = args.adaptive_admission["active"]
        limit = max(args.runtime_limit, 0)
        if active:
            cap = args.adaptive_admission["policy"]["seconds"]
            limit = min(limit, cap) if limit else cap
        previous_handlers = {}

        def interrupted(signum, frame):
            raise InterruptedError(f"signal_{signum}")

        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[sig] = signal.signal(sig, interrupted)
            if active and not _adaptive_continue(args):
                result = "resource_lease_revoked_before_start"
                return int(args.skip_exit_code)
            env = os.environ.copy()
            if active:
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                    env[key] = "1"
            proc = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, start_new_session=True)
            next_check = time.monotonic() + 5
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                if limit and elapsed >= limit:
                    result = "deadline_reached"
                    return 124
                if active and time.monotonic() >= next_check:
                    if not _adaptive_continue(args):
                        result = "resource_lease_revoked"
                        return int(args.skip_exit_code)
                    next_check = time.monotonic() + 5
                time.sleep(0.25)
            result = "completed" if proc.returncode == 0 else "child_failed"
            return int(proc.returncode)
        except InterruptedError:
            result = "interrupted"
            return 130
        finally:
            try:
                if proc is not None:
                    _stop_owned_group(proc, grace=max(0.1, min(args.terminate_grace, 30)))
            finally:
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)
                _end(args)
                state = _read_json(_state_path(args.slot))
                state.update(last_duration_seconds=round(time.monotonic() - started, 3), last_result=result)
                _write_json(_state_path(args.slot), state)
                _write_json(_health_path(args.slot), {
                    "timestamp_utc": _now_iso(), "slot": args.slot, "action": "completed",
                    "allowed": result == "completed", "result": result,
                    "elapsed_seconds": state["last_duration_seconds"],
                    "lease_wait_seconds": waited_seconds,
                    "runtime_limit_seconds": limit, "adaptive_admission": args.adaptive_admission,
                    "last_admission_check": getattr(args, "admission_recheck", {}),
                })


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard low-priority maintenance launchd jobs from overlapping or running during host pressure.")
    parser.add_argument("--slot", required=True)
    parser.add_argument("--begin", action="store_true")
    parser.add_argument("--end", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime-limit", type=int, default=0)
    parser.add_argument(
        "--lease-wait-seconds",
        type=float,
        default=0,
        help="Wait up to 120 seconds for the kernel lease, then evaluate admission afresh.",
    )
    parser.add_argument("--terminate-grace", type=float, default=30)
    parser.add_argument("--command", nargs=argparse.REMAINDER, default=[])
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
    if sum((args.begin, args.end, args.execute)) != 1:
        raise SystemExit("pass exactly one of --begin, --end or --execute")
    if args.execute:
        return _execute(args)
    if args.begin:
        return _begin(args)
    return _end(args)


if __name__ == "__main__":
    raise SystemExit(main())
