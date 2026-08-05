from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


GIB = float(1024**3)
DEFAULT_TARGET_FREE_GB = 64.0
DEFAULT_PRESSURE_FREE_GB = 32.0
DEFAULT_HARD_FREE_GB = 16.0
DEFAULT_EMERGENCY_FREE_GB = 8.0

DiskUsageFn = Callable[[Path], Any]


def _safe_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _threshold(name: str, default: float, explicit: float | None) -> float:
    if explicit is not None:
        return max(float(explicit), 0.0)
    return max(_safe_float(os.getenv(name), default), 0.0)


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.expanduser()
    for probe in (candidate, *candidate.parents):
        if probe.exists():
            return probe
    return candidate


def live_disk_snapshot(path: Path, *, disk_usage_fn: DiskUsageFn = shutil.disk_usage) -> dict[str, Any]:
    requested = path.expanduser()
    probe = _nearest_existing_path(requested)
    try:
        usage = disk_usage_fn(probe)
    except Exception as exc:
        return {
            "path": str(requested),
            "probe_path": str(probe),
            "known": False,
            "exists": bool(requested.exists()),
            "free_gb": 0.0,
            "used_gb": 0.0,
            "total_gb": 0.0,
            "used_percent": 100.0,
            "error": f"{type(exc).__name__}:{exc}",
        }

    total = max(float(usage.total), 1.0)
    used = max(float(usage.used), 0.0)
    free = max(float(usage.free), 0.0)
    return {
        "path": str(requested),
        "probe_path": str(probe),
        "known": True,
        "exists": bool(requested.exists()),
        "free_gb": round(free / GIB, 3),
        "used_gb": round(used / GIB, 3),
        "total_gb": round(total / GIB, 3),
        "used_percent": round((used / total) * 100.0, 3),
        "error": "",
    }


def local_storage_reserve_contract(
    project_root: Path,
    *,
    target_free_gb: float | None = None,
    pressure_free_gb: float | None = None,
    hard_free_gb: float | None = None,
    emergency_free_gb: float | None = None,
    disk_usage_fn: DiskUsageFn = shutil.disk_usage,
) -> dict[str, Any]:
    target = _threshold("BOT_LOCAL_STORAGE_TARGET_FREE_GB", DEFAULT_TARGET_FREE_GB, target_free_gb)
    pressure = _threshold("BOT_LOCAL_STORAGE_PRESSURE_FREE_GB", DEFAULT_PRESSURE_FREE_GB, pressure_free_gb)
    hard = _threshold("BOT_LOCAL_STORAGE_HARD_FREE_GB", DEFAULT_HARD_FREE_GB, hard_free_gb)
    emergency = _threshold(
        "BOT_LOCAL_STORAGE_EMERGENCY_FREE_GB",
        DEFAULT_EMERGENCY_FREE_GB,
        emergency_free_gb,
    )
    pressure = min(pressure, target)
    hard = min(hard, pressure)
    emergency = min(emergency, hard)

    disk = live_disk_snapshot(project_root, disk_usage_fn=disk_usage_fn)
    known = bool(disk.get("known", False))
    free_gb = _safe_float(disk.get("free_gb"), 0.0)
    used_percent = _safe_float(disk.get("used_percent"), 100.0)
    emergency_active = bool(known and (free_gb < emergency or used_percent >= 99.0))
    hard_block = bool(not known or emergency_active or free_gb < hard)
    pressure_active = bool(known and free_gb < pressure)
    target_ready = bool(known and free_gb >= target)

    if not known:
        status = "blocked_unknown"
        grade = "F"
    elif emergency_active:
        status = "emergency"
        grade = "F"
    elif free_gb < hard:
        status = "blocked"
        grade = "D"
    elif free_gb < pressure:
        status = "degraded"
        grade = "C"
    elif free_gb < target:
        status = "watch"
        grade = "A"
    else:
        status = "ready"
        grade = "A+"

    control_env: dict[str, str] = {
        "BOT_LOCAL_STORAGE_RESERVE_STATE": status,
        "BOT_LOCAL_STORAGE_RESERVE_FREE_GB": f"{free_gb:.3f}",
        "BOT_LOCAL_STORAGE_TARGET_FREE_GB": f"{target:.3f}",
        "BOT_LOCAL_STORAGE_PRESSURE_FREE_GB": f"{pressure:.3f}",
        "BOT_LOCAL_STORAGE_HARD_FREE_GB": f"{hard:.3f}",
        "BOT_LOCAL_STORAGE_EMERGENCY_FREE_GB": f"{emergency:.3f}",
        "BOT_LOCAL_STORAGE_RESERVE_PRESSURE": "1" if pressure_active else "0",
        "BOT_LOCAL_STORAGE_RESERVE_HARD_BLOCK": "1" if hard_block else "0",
    }
    if pressure_active or hard_block:
        control_env.update(
            {
                "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
                "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.15" if not hard_block else "0.05",
                "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG": "1",
                "HEAVY_COLLECTORS_PAUSED_FOR_BACKLOG": "1",
                "REPORT_REFRESH_PAUSED_FOR_BACKLOG": "1",
                "SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE": "1",
                "PAPER_EXECUTION_RUNTIME_PAUSED_FOR_LOCAL_STORAGE": "1",
                "BOT_DATA_CAPTURE_MODE": "thin_digest_with_manifest",
                "BOT_RAW_PAYLOAD_STORAGE_MODE": "manifest_first",
                "BOT_STORAGE_SPACE_RECOVERY_REQUIRED": "1",
                "BOT_STORAGE_RESERVE_REBUILD_REQUIRED": "1",
            }
        )
    if emergency_active or not known:
        control_env.update(
            {
                "BOT_STORAGE_EMERGENCY_DISK_GUARD": "1",
                "LOG_API_CALLS": "0",
                "LOG_LOOP_STATE": "0",
                "LOG_GATE_EVALUATIONS": "0",
                "LOG_GATE_PASSES": "0",
                "LOG_MASTER_VARIANT_DECISIONS": "0",
                "LOG_GRAND_MASTER_DECISIONS": "0",
                "LOG_OPTIONS_MASTER_DECISIONS": "0",
                "LOG_FUTURES_MASTER_DECISIONS": "0",
                "LOG_DECISION_EXPLANATIONS": "0",
                "LOG_SHADOW_PNL_ATTRIBUTION": "0",
            }
        )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "status": status,
        "ready": target_ready,
        "ok": not hard_block,
        "grade": grade,
        "disk_source": "live_disk_usage",
        "disk": disk,
        "free_gb": round(free_gb, 3),
        "target_free_gb": round(target, 3),
        "pressure_free_gb": round(pressure, 3),
        "hard_free_gb": round(hard, 3),
        "emergency_free_gb": round(emergency, 3),
        "reserve_deficit_gb": round(max(target - free_gb, 0.0), 3),
        "pressure_active": pressure_active,
        "hard_block": hard_block,
        "emergency_active": emergency_active,
        "pause_nonessential_writers": bool(pressure_active or hard_block),
        "control_env": control_env,
        "next_action": (
            "continue normal hot-path collection"
            if target_ready
            else "restore the internal hot-storage reserve before unattended operation"
        ),
    }
