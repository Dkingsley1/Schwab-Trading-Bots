#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.storage_mounts import resolve_external_storage
    from core.local_storage_reserve import local_storage_reserve_contract
    from scripts.ops.long_runtime_common import iso_now, ordered_unique, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.storage_mounts import resolve_external_storage
    from core.local_storage_reserve import local_storage_reserve_contract
    from .long_runtime_common import iso_now, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "storage_retention_unison_latest.json"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "governance" / "health" / "storage_retention_unison_history.jsonl"
DEFAULT_FORECAST_PATH = PROJECT_ROOT / "governance" / "health" / "storage_growth_forecast_latest.json"
DEFAULT_CAPACITY_CONTROL_EPOCH_PATH = (
    PROJECT_ROOT / "governance" / "runtime" / "storage_capacity_control_epoch_latest.json"
)
PROTECTED_VOLUME_PREFIXES = ("/Volumes/VIDEO",)
DEFAULT_SECOND_COLD_CANDIDATES = (
    "/Volumes/BOT_COLD/schwab_trading_bot",
    "/Volumes/BOT_ARCHIVE/schwab_trading_bot",
    "/Volumes/BOT_RETENTION/schwab_trading_bot",
)
DEFAULT_CONTINUOUS_RUN_DAYS = 30.0
DEFAULT_CONTINUOUS_RUN_BUFFER_GB = 32.0
DEFAULT_CONTINUOUS_RUN_MIN_DAILY_GB = 0.5


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _grade(score: float) -> str:
    value = max(min(float(score), 100.0), 0.0)
    if value >= 99.0:
        return "A+"
    if value >= 97.0:
        return "A+"
    if value >= 93.0:
        return "A"
    if value >= 85.0:
        return "B"
    if value >= 75.0:
        return "C"
    if value >= 65.0:
        return "D"
    return "F"


def _grade_rank(raw: Any) -> int:
    text = str(raw or "").strip().upper()
    ranks = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4, "A+": 5, "A++": 5}
    return ranks.get(text, 0)


def _is_protected_volume(path: Path) -> bool:
    raw = str(path.expanduser())
    return any(raw == prefix or raw.startswith(f"{prefix}/") for prefix in PROTECTED_VOLUME_PREFIXES)


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "y", "on"}


def _nearest_existing_parent(path: Path) -> Path:
    current = path.expanduser()
    for candidate in (current, *current.parents):
        if candidate.exists():
            return candidate
    return current


def _disk_snapshot(path: Path) -> dict[str, Any]:
    candidate = path.expanduser()
    protected = _is_protected_volume(candidate)
    parent = _nearest_existing_parent(candidate)
    if protected:
        return {
            "path": str(candidate),
            "checked_path": str(parent),
            "exists": False,
            "protected": True,
            "total_gb": 0.0,
            "used_gb": 0.0,
            "free_gb": 0.0,
            "used_percent": 0.0,
        }
    try:
        usage = shutil.disk_usage(parent)
    except Exception:
        return {
            "path": str(candidate),
            "checked_path": str(parent),
            "exists": bool(candidate.exists()),
            "protected": False,
            "total_gb": 0.0,
            "used_gb": 0.0,
            "free_gb": 0.0,
            "used_percent": 0.0,
        }
    total = max(float(usage.total), 1.0)
    used = float(usage.used)
    return {
        "path": str(candidate),
        "checked_path": str(parent),
        "exists": bool(candidate.exists()),
        "protected": False,
        "total_gb": round(total / (1024.0**3), 3),
        "used_gb": round(used / (1024.0**3), 3),
        "free_gb": round(float(usage.free) / (1024.0**3), 3),
        "used_percent": round((used / total) * 100.0, 3),
    }


def _same_real_path(left: Path, right: Path) -> bool:
    try:
        return left.expanduser().resolve(strict=False) == right.expanduser().resolve(strict=False)
    except Exception:
        return str(left.expanduser()) == str(right.expanduser())


def _synthetic_step(*, command: list[str], overall_status: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "command": list(command),
        "returncode": 0,
        "timed_out": False,
        "ok": True,
        "overall_status": overall_status,
        "payload": payload,
        "stdout_tail": "",
        "stderr_tail": "",
    }


def _run_json(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = int(proc.returncode)
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
        "command": command,
        "returncode": rc,
        "timed_out": timed_out,
        "ok": bool(rc == 0 and payload),
        "overall_status": str(payload.get("overall_status") or payload.get("status") or ("ready" if rc == 0 else "error")),
        "payload": payload,
        "stdout_tail": "\n".join(stdout.splitlines()[-8:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-8:]),
    }


def _read_history(path: Path, *, limit: int = 40) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    rows.append(parsed)
    except Exception:
        return []
    return rows[-max(int(limit), 1):]


def _parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _hot_lane_control_epoch(project_root: Path) -> datetime | None:
    path = project_root / "config" / ".env.hot_lane_retention_override"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    active = False
    timestamp: datetime | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("# updated_at_utc="):
            timestamp = _parse_ts(line.split("=", 1)[1])
        elif line.startswith("HOT_LANE_RETENTION_ACTIVE="):
            active = line.split("=", 1)[1].strip().strip("'\"").lower() in {"1", "true", "yes", "on"}
    if not active:
        return None
    if timestamp is not None:
        return timestamp
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _device_id(path: Path) -> int | None:
    try:
        return int(_nearest_existing_parent(path).stat().st_dev)
    except OSError:
        return None


def _path_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((os.path.abspath(path), os.path.abspath(root))) == os.path.abspath(root)
    except (OSError, ValueError):
        return False


def _verified_cross_tier_capacity_event(
    project_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    deep_cold = _load_json_file(
        project_root / "governance" / "health" / "deep_cold_storage_layer_latest.json"
    )
    manifest_path = Path(str(deep_cold.get("manifest_path") or "")).expanduser()
    if not str(manifest_path) or _is_protected_volume(manifest_path) or not manifest_path.is_file():
        return {}

    source_device_id = _device_id(project_root)
    destination_device_id = _device_id(external_root)
    if (
        source_device_id is None
        or destination_device_id is None
        or source_device_id == destination_device_id
    ):
        return {}

    configured_cold_root = str(os.getenv("BOT_SECOND_COLD_ROOT", "") or "").strip()
    cold_root = Path(configured_cold_root).expanduser() if configured_cold_root else external_root
    if _is_protected_volume(cold_root):
        return {}

    latest_epoch: datetime | None = None
    moved_bytes = 0
    moved_files = 0
    try:
        handle = manifest_path.open("r", encoding="utf-8")
    except OSError:
        return {}
    with handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict) or not bool(row.get("source_replaced_with_symlink", False)):
                continue
            move = row.get("second_cold_move") if isinstance(row.get("second_cold_move"), dict) else {}
            if not (
                bool(move.get("source_replaced_with_symlink", False))
                and bool(move.get("verified_size_match", False))
                and bool(move.get("verified_sha256_match", False))
                and bool(move.get("source_stable", False))
                and str(move.get("source_sha256") or "")
                == str(move.get("target_sha256") or "")
            ):
                continue
            source = Path(str(move.get("source") or row.get("path") or "")).expanduser()
            target = Path(str(move.get("target") or row.get("second_cold_target") or "")).expanduser()
            if (
                not _path_within(source, project_root)
                or not _path_within(target, cold_root)
                or _is_protected_volume(source)
                or _is_protected_volume(target)
                or _safe_int(row.get("source_device_id"), -1) != source_device_id
                or _device_id(target) != destination_device_id
                or not source.is_symlink()
                or not target.is_file()
            ):
                continue
            try:
                link_target = Path(os.readlink(source))
                if not link_target.is_absolute():
                    link_target = source.parent / link_target
                if os.path.abspath(link_target) != os.path.abspath(target):
                    continue
                source_epoch = datetime.fromtimestamp(source.lstat().st_mtime, tz=timezone.utc)
                target_bytes = int(target.stat().st_size)
            except OSError:
                continue
            expected_bytes = _safe_int(move.get("bytes"), target_bytes)
            if expected_bytes <= 0 or target_bytes != expected_bytes:
                continue
            latest_epoch = source_epoch if latest_epoch is None else max(latest_epoch, source_epoch)
            moved_bytes += expected_bytes
            moved_files += 1

    if latest_epoch is None or moved_files <= 0:
        return {}
    return {
        "timestamp_utc": latest_epoch.isoformat(),
        "baseline_not_before_utc": latest_epoch.isoformat(),
        "event_type": "verified_cross_filesystem_deep_cold_move",
        "verified": True,
        "manifest_path": str(manifest_path),
        "source_device_id": source_device_id,
        "destination_device_id": destination_device_id,
        "moved_files": moved_files,
        "moved_gb": round(moved_bytes / float(1024**3), 3),
        "forecast_effect": "reset_external_growth_baseline_without_erasing_capacity_consumption",
    }


def _storage_growth_baseline_control(
    project_root: Path,
    external_root: Path,
    *,
    apply: bool,
    event_path: Path,
) -> dict[str, Any]:
    hot_lane_epoch = _hot_lane_control_epoch(project_root)
    persisted = _load_json_file(event_path)
    persisted_epoch = (
        _parse_ts(persisted.get("baseline_not_before_utc"))
        if bool(persisted.get("verified", False))
        else None
    )
    discovered = _verified_cross_tier_capacity_event(project_root, external_root)
    discovered_epoch = _parse_ts(discovered.get("baseline_not_before_utc"))
    capacity_epoch = max(
        [epoch for epoch in (persisted_epoch, discovered_epoch) if epoch is not None],
        default=None,
    )
    if apply and discovered_epoch is not None and (
        persisted_epoch is None or discovered_epoch > persisted_epoch
    ):
        write_payload(event_path, discovered)
        persisted = discovered

    candidates = [epoch for epoch in (hot_lane_epoch, capacity_epoch) if epoch is not None]
    selected_epoch = max(candidates, default=None)
    if selected_epoch is None:
        scope = "unbounded_history"
        reason = ""
    elif capacity_epoch is not None and selected_epoch == capacity_epoch:
        scope = "post_verified_cross_tier_capacity_event"
        reason = "verified archive movement is capacity redistribution, not live ingestion growth"
    else:
        scope = "post_hot_lane_control_epoch"
        reason = "hot-lane retention policy changed the measurable storage slope"
    return {
        "baseline_not_before_utc": selected_epoch.isoformat() if selected_epoch else "",
        "baseline_scope": scope,
        "reason": reason,
        "hot_lane_control_epoch_utc": hot_lane_epoch.isoformat() if hot_lane_epoch else "",
        "capacity_control_epoch_utc": capacity_epoch.isoformat() if capacity_epoch else "",
        "capacity_control_event": persisted if persisted_epoch is not None else discovered,
        "event_path": str(event_path),
    }
def _storage_growth_forecast(
    *,
    current_external: dict[str, Any],
    current_internal: dict[str, Any],
    history_rows: list[dict[str, Any]],
    target_free_gb: float,
    pressure_free_gb: float,
    baseline_not_before_utc: datetime | None = None,
    baseline_scope: str = "post_hot_lane_control_epoch",
) -> dict[str, Any]:
    current_ts = datetime.now(timezone.utc)
    current_free = _safe_float(current_external.get("free_gb"), 0.0)
    current_internal_free = _safe_float(current_internal.get("free_gb"), 0.0)
    sustained_baseline: dict[str, Any] = {}
    burst_baseline: dict[str, Any] = {}
    discarded_pre_control_samples = 0
    for row in reversed(history_rows):
        ts = _parse_ts(row.get("timestamp_utc"))
        disk = row.get("disk") if isinstance(row.get("disk"), dict) else {}
        external = disk.get("external") if isinstance(disk.get("external"), dict) else {}
        prior_free = _safe_float(external.get("free_gb"), -1.0)
        if ts is None or prior_free < 0.0:
            continue
        if baseline_not_before_utc is not None and ts < baseline_not_before_utc:
            discarded_pre_control_samples += 1
            continue
        age_seconds = (current_ts - ts).total_seconds()
        if not burst_baseline and age_seconds >= 300:
            burst_baseline = {"timestamp_utc": ts.isoformat(), "external_free_gb": prior_free}
        if age_seconds >= 1800:
            sustained_baseline = {"timestamp_utc": ts.isoformat(), "external_free_gb": prior_free}
            break

    def rate_from(baseline: dict[str, Any]) -> tuple[float, float]:
        if not baseline:
            return 0.0, 0.0
        parsed = _parse_ts(baseline.get("timestamp_utc"))
        if parsed is None:
            return 0.0, 0.0
        elapsed = max((current_ts - parsed).total_seconds() / 86400.0, 0.0)
        prior_free = _safe_float(baseline.get("external_free_gb"), current_free)
        return elapsed, max((prior_free - current_free) / max(elapsed, 1e-6), 0.0)

    sustained_elapsed_days, sustained_consumed_gb_per_day = rate_from(sustained_baseline)
    burst_elapsed_days, burst_consumed_gb_per_day = rate_from(burst_baseline)
    baseline = sustained_baseline or burst_baseline
    elapsed_days = sustained_elapsed_days if sustained_baseline else burst_elapsed_days
    consumed_gb_per_day = sustained_consumed_gb_per_day if sustained_baseline else burst_consumed_gb_per_day
    confidence = "sustained" if sustained_baseline else ("burst_low" if burst_baseline else "new_baseline")

    def days_until(floor: float, *, rate: float | None = None) -> float | None:
        effective_rate = consumed_gb_per_day if rate is None else float(rate)
        if effective_rate <= 0.0:
            return None
        if current_free <= floor:
            return 0.0
        return round((current_free - float(floor)) / effective_rate, 2)

    target_days = days_until(float(target_free_gb), rate=sustained_consumed_gb_per_day or consumed_gb_per_day)
    pressure_days = days_until(float(pressure_free_gb), rate=sustained_consumed_gb_per_day or consumed_gb_per_day)
    burst_pressure_days = days_until(float(pressure_free_gb), rate=burst_consumed_gb_per_day)
    if not sustained_baseline and not burst_baseline:
        status = "baseline_needed"
        score = 94.0 if current_free >= target_free_gb and current_internal_free >= 25.0 else 82.0
        next_action = "run storage-retention-unison again later to establish a real growth-rate slope"
    elif current_free < pressure_free_gb:
        status = "pressure"
        score = 72.0
        next_action = "clear space before training or expansion"
    elif current_free < target_free_gb:
        status = "target_floor_breach"
        score = 84.0
        next_action = "apply hot-lane retention and run bounded cleanup until BOT_LOGS is back above target"
    elif sustained_consumed_gb_per_day <= 0.0 and burst_consumed_gb_per_day <= 0.0:
        status = "stable_or_improving"
        score = 99.0
        next_action = "free space is stable or improving"
    elif not sustained_baseline and burst_consumed_gb_per_day > 0.0:
        status = "burst_watch"
        score = 92.0 if current_free >= target_free_gb else 84.0
        next_action = "apply hot-lane retention, then re-sample after the burst window before lowering the grade"
    elif pressure_days is not None and pressure_days <= 3.0:
        status = "near_pressure"
        score = 86.0
        next_action = "run cleanup/compaction and hot-lane retention before the next training wave"
    else:
        status = "forecast_ready"
        score = 97.0 if current_free >= target_free_gb else 88.0
        next_action = "watch projected pressure date and keep raw compaction bounded"

    return {
        "timestamp_utc": iso_now(),
        "status": status,
        "score": round(score, 2),
        "grade": _grade(score),
        "source": "storage_retention_unison_history" if baseline else "new_baseline",
        "confidence": confidence,
        "baseline_scope": str(baseline_scope) if baseline_not_before_utc else "unbounded_history",
        "baseline_not_before_utc": baseline_not_before_utc.isoformat() if baseline_not_before_utc else "",
        "discarded_pre_control_samples": discarded_pre_control_samples,
        "baseline": baseline,
        "sustained_baseline": sustained_baseline,
        "burst_baseline": burst_baseline,
        "current_external_free_gb": round(current_free, 3),
        "current_internal_free_gb": round(current_internal_free, 3),
        "target_free_gb": round(float(target_free_gb), 3),
        "pressure_free_gb": round(float(pressure_free_gb), 3),
        "elapsed_days": round(elapsed_days, 4),
        "consumed_gb_per_day": round(consumed_gb_per_day, 4),
        "sustained_consumed_gb_per_day": round(sustained_consumed_gb_per_day, 4),
        "burst_consumed_gb_per_day": round(burst_consumed_gb_per_day, 4),
        "days_until_target_free": target_days,
        "days_until_pressure_free": pressure_days,
        "burst_days_until_pressure_free": burst_pressure_days,
        "recommended_control": "hot_lane_retention_control" if status in {"burst_watch", "near_pressure", "target_floor_breach", "pressure"} else "",
        "next_action": next_action,
    }


def _continuous_run_contract(
    *,
    forecast: dict[str, Any],
    horizon_days: float = DEFAULT_CONTINUOUS_RUN_DAYS,
    pressure_free_gb: float = 64.0,
    safety_buffer_gb: float = DEFAULT_CONTINUOUS_RUN_BUFFER_GB,
    min_daily_growth_gb: float = DEFAULT_CONTINUOUS_RUN_MIN_DAILY_GB,
    duty_cycle_max_active_ratio: float | None = None,
    storage_controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    horizon = max(float(horizon_days), 1.0)
    pressure_floor = max(float(pressure_free_gb), 0.0)
    buffer_gb = max(float(safety_buffer_gb), 0.0)
    min_daily = max(float(min_daily_growth_gb), 0.0)
    current_free = _safe_float(forecast.get("current_external_free_gb"), 0.0)
    sustained = _safe_float(forecast.get("sustained_consumed_gb_per_day"), 0.0)
    burst = _safe_float(forecast.get("burst_consumed_gb_per_day"), 0.0)
    observed = _safe_float(forecast.get("consumed_gb_per_day"), 0.0)
    confidence = str(forecast.get("confidence") or "")
    slope_elapsed_days = _safe_float(forecast.get("elapsed_days"), 0.0)
    min_projection_slope_days = max(
        _safe_float(os.getenv("STORAGE_CONTINUOUS_RUN_MIN_SLOPE_DAYS"), 0.25),
        0.0,
    )
    sustained_baseline_ready = bool(forecast.get("sustained_baseline")) or confidence == "sustained"
    high_short_window_growth = bool(
        slope_elapsed_days > 0.0
        and slope_elapsed_days < min_projection_slope_days
        and max(sustained, burst, observed) > max(min_daily * 4.0, 2.0)
    )
    if high_short_window_growth:
        raw_effective_daily = min_daily
    elif sustained_baseline_ready:
        raw_effective_daily = max(sustained, min_daily)
    else:
        raw_effective_daily = max(burst, observed, min_daily)

    controls = storage_controls if isinstance(storage_controls, dict) else {}
    governed_slope_max_days = max(
        _safe_float(os.getenv("STORAGE_CONTINUOUS_RUN_GOVERNED_SLOPE_MAX_DAYS"), 1.0),
        min_projection_slope_days,
    )
    core_control_names = (
        "storage_efficiency_ready",
        "quota_ready",
        "route_verified",
        "resilience_ready",
        "steady_state_ready",
        "retention_debt_ok",
        "collector_intake_enforced",
        "raw_candidate_compaction_ok",
        "sparse_large_line_pending_bounded",
        "deep_cold_ready",
        "hot_lane_retention_active",
        "external_free_above_target",
    )
    storage_governed_core_ready = bool(controls) and all(bool(controls.get(name, False)) for name in core_control_names)
    manifest_first_ready = bool(controls.get("manifest_first_storage", False))
    storage_governed_control_ready = bool(storage_governed_core_ready and manifest_first_ready)
    storage_bounded_control_ready = bool(storage_governed_core_ready and not manifest_first_ready)
    short_controlled_slope = bool(
        slope_elapsed_days > 0.0
        and slope_elapsed_days < governed_slope_max_days
        and raw_effective_daily > max(min_daily * 4.0, 2.0)
    )
    storage_governed_projection = bool(storage_governed_control_ready and short_controlled_slope)
    storage_bounded_projection = bool(storage_bounded_control_ready and short_controlled_slope)
    storage_projection_override = bool(storage_governed_projection or storage_bounded_projection)
    duty_cycle_ratio = 1.0
    if duty_cycle_max_active_ratio is not None:
        duty_cycle_ratio = min(max(_safe_float(duty_cycle_max_active_ratio, 1.0), 0.01), 1.0)
    duty_cycle_adjusted = bool(duty_cycle_ratio < 0.999 and raw_effective_daily > min_daily)
    effective_daily = max(min_daily, raw_effective_daily * duty_cycle_ratio) if duty_cycle_adjusted else raw_effective_daily
    if storage_projection_override:
        effective_daily = min_daily
    projected_free = round(current_free - (effective_daily * horizon), 3)
    required_free = round(pressure_floor + buffer_gb + (effective_daily * horizon), 3)
    available_margin = round(current_free - required_free, 3)
    days_until_pressure = forecast.get("days_until_pressure_free")
    days_until_pressure_value = None
    if days_until_pressure is not None:
        days_until_pressure_value = _safe_float(days_until_pressure, 0.0)
    controlled_days_until_pressure = None
    if effective_daily > 0.0 and current_free > pressure_floor:
        controlled_days_until_pressure = round((current_free - pressure_floor) / effective_daily, 2)

    blockers: list[str] = []
    warnings: list[str] = []
    if current_free <= 0.0:
        blockers.append("external_free_space_unknown_or_zero")
    if current_free < required_free:
        blockers.append("insufficient_projected_free_space")
    if projected_free < pressure_floor:
        blockers.append("projected_below_pressure_floor")
    if (
        controlled_days_until_pressure is not None
        and controlled_days_until_pressure < horizon
        and not high_short_window_growth
        and not storage_projection_override
    ):
        blockers.append("forecast_pressure_inside_horizon")
    if (
        str(forecast.get("status") or "") in {"pressure", "target_floor_breach", "near_pressure"}
        and not high_short_window_growth
        and not storage_projection_override
        and not duty_cycle_adjusted
    ):
        blockers.append(f"forecast_status_{forecast.get('status')}")
    if high_short_window_growth:
        warnings.append("growth_rate_window_too_short_for_30_day_projection")
    if storage_governed_projection:
        warnings.append("storage_governed_controls_override_short_slope")
    if storage_bounded_projection:
        warnings.append("bounded_storage_controls_override_short_post_maintenance_slope")
        if not manifest_first_ready:
            warnings.append("manifest_first_storage_pending")
    if confidence in {"new_baseline", "burst_low"}:
        warnings.append("growth_rate_baseline_is_not_sustained_yet")
    if sustained > 0.0 and burst > max(sustained * 2.0, sustained + min_daily):
        warnings.append("burst_growth_above_sustained_rate")
    if duty_cycle_adjusted:
        warnings.append("collection_duty_cycle_controls_growth_projection")

    warning_set = set(warnings)
    short_window_warning_reclassified_ready = bool(
        not blockers
        and warning_set == {"growth_rate_window_too_short_for_30_day_projection"}
        and high_short_window_growth
        and storage_governed_core_ready
        and bool(controls.get("external_free_above_target", False))
        and available_margin >= buffer_gb
        and (
            controlled_days_until_pressure is None
            or controlled_days_until_pressure >= horizon * 2.0
        )
    )

    if blockers:
        status = "blocked"
        score = 72.0 if "forecast_pressure_inside_horizon" in blockers else 80.0
        next_action = "apply retention/cleanup until projected 30-day free-space margin is positive"
    elif warnings and not short_window_warning_reclassified_ready:
        status = "watch"
        score = 92.0
        next_action = "keep storage-retention-unison on cadence until sustained growth-rate history confirms the 30-day margin"
    else:
        status = "ready"
        score = 99.0
        next_action = (
            "storage controls and free-space margin absorb short-window growth-rate noise"
            if short_window_warning_reclassified_ready
            else "storage slope has enough free-space margin for the 30-day collection soak"
        )

    return {
        "active": True,
        "status": status,
        "ready": bool(not blockers),
        "score": round(score, 2),
        "grade": _grade(score),
        "horizon_days": round(horizon, 3),
        "pressure_free_gb": round(pressure_floor, 3),
        "safety_buffer_gb": round(buffer_gb, 3),
        "min_daily_growth_gb": round(min_daily, 3),
        "min_projection_slope_days": round(min_projection_slope_days, 4),
        "storage_governed_slope_max_days": round(governed_slope_max_days, 4),
        "raw_effective_daily_growth_gb": round(raw_effective_daily, 4),
        "effective_daily_growth_gb": round(effective_daily, 4),
        "storage_governed_projection": storage_governed_projection,
        "storage_bounded_projection": storage_bounded_projection,
        "storage_projection_override": storage_projection_override,
        "storage_governed_core_ready": storage_governed_core_ready,
        "storage_governed_control_ready": storage_governed_control_ready,
        "storage_bounded_control_ready": storage_bounded_control_ready,
        "short_window_warning_reclassified_ready": short_window_warning_reclassified_ready,
        "storage_controls": controls,
        "duty_cycle_adjusted": duty_cycle_adjusted,
        "duty_cycle_max_active_ratio": round(duty_cycle_ratio, 4),
        "sustained_daily_growth_gb": round(sustained, 4),
        "burst_daily_growth_gb": round(burst, 4),
        "current_external_free_gb": round(current_free, 3),
        "projected_external_free_gb_after_horizon": projected_free,
        "required_external_free_gb": required_free,
        "available_margin_gb": available_margin,
        "days_until_pressure_free": days_until_pressure,
        "raw_days_until_pressure_free": days_until_pressure_value,
        "controlled_days_until_pressure_free": controlled_days_until_pressure,
        "blockers": blockers,
        "warnings": warnings,
        "control_env": {
            "BOT_CONTINUOUS_COLLECTION_SOAK_DAYS": str(round(horizon, 3)),
            "BOT_CONTINUOUS_COLLECTION_PRESSURE_FREE_GB": str(round(pressure_floor, 3)),
            "BOT_CONTINUOUS_COLLECTION_FREE_SPACE_MARGIN_GB": str(available_margin),
            "BOT_CONTINUOUS_COLLECTION_READY": "1" if not blockers else "0",
            "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
            "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": str(round(duty_cycle_ratio if duty_cycle_adjusted else (0.24 if not blockers else 0.16), 4)),
        },
        "next_action": next_action,
    }


def _second_cold_preflight() -> dict[str, Any]:
    configured = os.getenv("BOT_SECOND_COLD_ROOT", "").strip()
    candidates = [configured] if configured else list(DEFAULT_SECOND_COLD_CANDIDATES)
    candidate_rows: list[dict[str, Any]] = []
    ready = False
    configured_protected_hit = False
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        protected = _is_protected_volume(path)
        configured_protected_hit = configured_protected_hit or bool(configured and raw == configured and protected)
        snapshot = _disk_snapshot(path)
        exists = bool(path.exists() and not protected)
        row = {
            "path": str(path),
            "configured": bool(configured and raw == configured),
            "exists": exists,
            "protected": protected,
            "free_gb": snapshot.get("free_gb", 0.0),
            "used_percent": snapshot.get("used_percent", 0.0),
            "ready": bool(exists and _safe_float(snapshot.get("free_gb"), 0.0) >= 64.0),
        }
        if row["ready"]:
            ready = True
        candidate_rows.append(row)
    status = "ready" if ready else ("blocked_protected_target" if configured_protected_hit else "prewired_waiting_for_drive")
    score = 100.0 if ready else (50.0 if configured_protected_hit else 96.0)
    return {
        "status": status,
        "score": score,
        "grade": _grade(score),
        "ready": ready,
        "configured_path": configured,
        "candidates": candidate_rows,
        "recommended_volume_names": ["BOT_COLD", "BOT_ARCHIVE", "BOT_RETENTION"],
        "recommended_format": "APFS",
        "never_touch_protected_volumes": list(PROTECTED_VOLUME_PREFIXES),
        "approved_video_cold_archive": {
            "enabled": False,
            "root": "",
            "scope": "forbidden",
        },
        "next_action": (
            "second cold target is ready"
            if ready
            else "set BOT_SECOND_COLD_ROOT to an approved cold archive target, or mount BOT_COLD/BOT_ARCHIVE"
        ),
    }


def _cold_archive_spillover_capacity_gb(second_cold: dict[str, Any]) -> float:
    if not bool(second_cold.get("ready", False)):
        return 0.0
    reserve_gb = max(_safe_float(os.getenv("BOT_COLD_ARCHIVE_RESERVE_GB"), 64.0), 0.0)
    max_credit_gb = max(_safe_float(os.getenv("BOT_COLD_ARCHIVE_SPILLOVER_MAX_CREDIT_GB"), 0.0), 0.0)
    best_free = 0.0
    for row in second_cold.get("candidates") if isinstance(second_cold.get("candidates"), list) else []:
        if isinstance(row, dict) and bool(row.get("ready", False)):
            best_free = max(best_free, _safe_float(row.get("free_gb"), 0.0))
    usable_headroom = max(best_free - reserve_gb, 0.0)
    if max_credit_gb > 0.0:
        usable_headroom = min(usable_headroom, max_credit_gb)
    return round(usable_headroom, 3)


def _apply_cold_archive_spillover_contract(continuous_run: dict[str, Any], second_cold: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(continuous_run, dict) or not bool(second_cold.get("ready", False)):
        return continuous_run
    blockers = [str(item) for item in continuous_run.get("blockers") if str(item)] if isinstance(continuous_run.get("blockers"), list) else []
    managed_projection_blockers = {
        "insufficient_projected_free_space",
        "projected_below_pressure_floor",
        "forecast_pressure_inside_horizon",
        "forecast_status_target_floor_breach",
    }
    unmanaged = [item for item in blockers if item not in managed_projection_blockers]
    current_free = _safe_float(continuous_run.get("current_external_free_gb"), 0.0)
    pressure_free = _safe_float(continuous_run.get("pressure_free_gb"), 64.0)
    primary_guard_buffer = max(_safe_float(os.getenv("BOT_COLD_ARCHIVE_PRIMARY_PRESSURE_BUFFER_GB"), 16.0), 0.0)
    margin = _safe_float(continuous_run.get("available_margin_gb"), 0.0)
    spillover_capacity = _cold_archive_spillover_capacity_gb(second_cold)
    adjusted_margin = round(margin + spillover_capacity, 3)
    out = dict(continuous_run)
    required_spillover = round(max(-margin, 0.0), 3)
    capacity_shortfall = round(max(-adjusted_margin, 0.0), 3)
    primary_guard_ready = bool(current_free >= pressure_free + primary_guard_buffer)
    projection_only_blockers = bool(blockers and not unmanaged)
    spillover_ready = bool(
        projection_only_blockers
        and primary_guard_ready
        and spillover_capacity > 0.0
        and adjusted_margin >= 0.0
    )
    if not blockers:
        archive_status = "available_not_needed"
    elif unmanaged:
        archive_status = "blocked_by_non_projection_controls"
    elif not primary_guard_ready:
        archive_status = "primary_pressure_guard_not_ready"
    elif adjusted_margin < 0.0:
        archive_status = "insufficient_capacity_for_horizon"
    else:
        archive_status = "ready_for_projection_spillover"

    out.update(
        {
            "cold_archive_spillover_available": bool(spillover_capacity > 0.0),
            "cold_archive_spillover_ready": spillover_ready,
            "cold_archive_spillover_status": archive_status,
            "cold_archive_spillover_capacity_gb": spillover_capacity,
            "cold_archive_required_spillover_gb": required_spillover,
            "cold_archive_capacity_shortfall_gb": capacity_shortfall,
            "cold_archive_adjusted_margin_gb": adjusted_margin,
            "cold_archive_primary_pressure_buffer_gb": primary_guard_buffer,
            "cold_archive_primary_pressure_guard_ready": primary_guard_ready,
            "cold_archive_capacity_policy": "live_destination_free_minus_reserve_with_optional_configured_cap",
        }
    )
    control_env = dict(out.get("control_env") if isinstance(out.get("control_env"), dict) else {})
    control_env.update(
        {
            "BOT_COLD_ARCHIVE_SPILLOVER_READY": "1" if spillover_ready else "0",
            "BOT_COLD_ARCHIVE_SPILLOVER_CAPACITY_GB": str(spillover_capacity),
            "BOT_COLD_ARCHIVE_REQUIRED_SPILLOVER_GB": str(required_spillover),
            "BOT_COLD_ARCHIVE_CAPACITY_SHORTFALL_GB": str(capacity_shortfall),
            "BOT_COLD_ARCHIVE_ADJUSTED_MARGIN_GB": str(adjusted_margin),
        }
    )
    out["control_env"] = control_env
    if not spillover_ready:
        if archive_status == "insufficient_capacity_for_horizon":
            out["next_action"] = (
                f"reduce sustained storage growth; cold archive headroom={spillover_capacity:.3f} GiB "
                f"leaves horizon shortfall={capacity_shortfall:.3f} GiB"
            )
        return out

    warnings = ordered_unique(
        [str(item) for item in out.get("warnings") if str(item)] if isinstance(out.get("warnings"), list) else []
    )
    warnings.append("second_cold_archive_spillover_covers_30_day_margin")
    out.update(
        {
            "status": "watch" if warnings else "ready",
            "ready": True,
            "score": max(_safe_float(out.get("score"), 0.0), 94.0),
            "grade": _grade(max(_safe_float(out.get("score"), 0.0), 94.0)),
            "blockers": [],
            "managed_blockers": blockers,
            "warnings": ordered_unique(warnings),
            "cold_archive_spillover_ready": True,
            "cold_archive_spillover_capacity_gb": spillover_capacity,
            "cold_archive_adjusted_margin_gb": adjusted_margin,
            "cold_archive_primary_pressure_buffer_gb": primary_guard_buffer,
            "next_action": "primary BOT_LOGS stays above pressure floor; approved cold archive spillover covers the 30-day margin",
        }
    )
    control_env = dict(out.get("control_env") if isinstance(out.get("control_env"), dict) else {})
    control_env.update(
        {
            "BOT_CONTINUOUS_COLLECTION_READY": "1",
            "BOT_COLD_ARCHIVE_SPILLOVER_READY": "1",
            "BOT_COLD_ARCHIVE_SPILLOVER_CAPACITY_GB": str(spillover_capacity),
            "BOT_COLD_ARCHIVE_ADJUSTED_MARGIN_GB": str(adjusted_margin),
        }
    )
    out["control_env"] = control_env
    return out


def _sql_soft_quota_managed_by_cold_spillover(quota_payload: dict[str, Any], continuous_run: dict[str, Any]) -> bool:
    summary = quota_payload.get("quota_summary") if isinstance(quota_payload.get("quota_summary"), dict) else {}
    degraded = {str(item) for item in summary.get("degraded_families") if str(item)} if isinstance(summary.get("degraded_families"), list) else set()
    blocked = {str(item) for item in summary.get("blocked_families") if str(item)} if isinstance(summary.get("blocked_families"), list) else set()
    if blocked or degraded - {"sql_link_shards"}:
        return False
    if _safe_int(summary.get("hard_breaches"), 0) > 0:
        return False
    if _safe_int(summary.get("soft_breaches"), 0) <= 0:
        return False
    if not bool(continuous_run.get("cold_archive_spillover_ready", False)):
        return False
    current_free = _safe_float(continuous_run.get("current_external_free_gb"), 0.0)
    pressure_free = _safe_float(continuous_run.get("pressure_free_gb"), 64.0)
    primary_guard_buffer = _safe_float(continuous_run.get("cold_archive_primary_pressure_buffer_gb"), 16.0)
    return bool(current_free >= pressure_free + primary_guard_buffer)


def _section(label: str, status: str, score: float, evidence: dict[str, Any], next_action: str) -> dict[str, Any]:
    return {
        "label": label,
        "status": status,
        "score": round(float(score), 2),
        "grade": _grade(score),
        "evidence": evidence,
        "next_action": next_action,
    }


def _step_payload(step: dict[str, Any]) -> dict[str, Any]:
    payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}
    return payload


def _step_summary(step: dict[str, Any]) -> dict[str, Any]:
    payload = _step_payload(step)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return summary


def _step_reduction_gb(step: dict[str, Any]) -> float:
    summary = _step_summary(step)
    return max(
        _safe_float(summary.get("estimated_hot_reduction_gb"), 0.0),
        _safe_float(summary.get("estimated_reduction_gb"), 0.0),
        _safe_float(summary.get("raw_compacted_gb"), 0.0),
        _safe_float(summary.get("raw_archived_gb"), 0.0),
        0.0,
    )


def _compaction_step_ok(step: dict[str, Any]) -> bool:
    status = str(step.get("overall_status") or "")
    if status == "busy":
        return True
    if int(step.get("returncode", 1)) != 0:
        return False
    return status in {"applied", "planned", "nothing_to_do", "ready", "watch", "watching"}


def _deep_cold_needs_data_is_advisory(name: str, step: dict[str, Any]) -> bool:
    if name != "retention_freshness_deep_cold":
        return False
    if str(step.get("overall_status") or "") != "needs_data":
        return False
    payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return (
        _safe_int(summary.get("candidate_count"), 0) <= 0
        and _safe_float(summary.get("candidate_gb"), 0.0) <= 0.0
        and _safe_int(summary.get("managed_count"), 0) <= 0
    )


def _compaction_lane_evidence(step: dict[str, Any]) -> dict[str, Any]:
    summary = _step_summary(step)
    return {
        "status": str(step.get("overall_status") or ""),
        "candidate_count": _safe_int(summary.get("candidate_count"), 0),
        "selected_gb": _safe_float(summary.get("selected_gb"), 0.0),
        "estimated_reduction_gb": _step_reduction_gb(step),
    }


def _hot_plane_compaction_contract(*, steps_by_lane: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summaries = {name: _step_summary(step) for name, step in steps_by_lane.items()}
    selected_gb = round(sum(_safe_float(summary.get("selected_gb"), 0.0) for summary in summaries.values()), 3)
    reduction_gb = round(sum(_step_reduction_gb(step) for step in steps_by_lane.values()), 3)
    errors = [name for name, step in steps_by_lane.items() if not _compaction_step_ok(step)]
    busy_lanes = [
        name
        for name, step in steps_by_lane.items()
        if str(step.get("overall_status") or "") == "busy"
    ]
    candidate_count = sum(_safe_int(summary.get("candidate_count"), 0) for summary in summaries.values())
    if errors:
        status = "degraded"
        score = 82.0
        next_action = "fix compactor failures before widening ingestion or training"
    elif busy_lanes:
        status = "in_progress"
        score = 97.0
        next_action = "let the lock-owning compactor finish, then refresh storage-tier-policy"
    elif reduction_gb > 0.0:
        status = "applied"
        score = 99.0
        next_action = "refresh storage-tier-policy after the bounded compaction wave"
    elif selected_gb > 0.0:
        status = "planned"
        score = 94.0
        next_action = "run storage-retention-unison --apply to compress selected hot-plane artifacts"
    else:
        status = "ready"
        score = 99.0
        next_action = "hot-plane compaction has no oversized eligible files right now"
    return {
        "status": status,
        "score": score,
        "grade": _grade(score),
        "errors": errors,
        "busy_lanes": busy_lanes,
        "candidate_count": candidate_count,
        "selected_gb": selected_gb,
        "estimated_reduction_gb": reduction_gb,
        "lanes": {name: _compaction_lane_evidence(step) for name, step in steps_by_lane.items()},
        "policy": "run compactors as part of storage-retention-unison so hot telemetry is rotated before disk pressure turns into a blocker",
        "next_action": next_action,
    }


def _manifest_backed_offload_evidence(tier_payload: dict[str, Any]) -> dict[str, Any]:
    contract = (
        tier_payload.get("manifest_backed_offload_contract")
        if isinstance(tier_payload.get("manifest_backed_offload_contract"), dict)
        else {}
    )
    summary = (
        tier_payload.get("offload_manifest_summary")
        if isinstance(tier_payload.get("offload_manifest_summary"), dict)
        else {}
    )
    status = str(contract.get("status") or "missing")
    eligible_files = _safe_int(contract.get("eligible_offload_files"), 0)
    eligible_gb = _safe_float(contract.get("eligible_offload_gb"), 0.0)
    compaction_files = _safe_int(contract.get("compaction_only_files"), 0)
    compaction_gb = _safe_float(contract.get("compaction_only_gb"), 0.0)
    manifest_path = str(contract.get("manifest_path") or "")
    if not contract:
        score = 72.0
        section_status = "missing"
        next_action = "refresh storage-tier-policy so offload and compaction lanes have a manifest-backed safety contract"
    elif status == "planned":
        score = 99.0
        section_status = "planned"
        next_action = str(contract.get("next_action") or "run bounded retention-unison apply lanes")
    else:
        score = 97.0
        section_status = status
        next_action = str(contract.get("next_action") or "keep refreshing storage-tier-policy")
    return {
        "status": section_status,
        "score": score,
        "grade": _grade(score),
        "manifest_path": manifest_path,
        "eligible_offload_files": eligible_files,
        "eligible_offload_gb": eligible_gb,
        "compaction_only_files": compaction_files,
        "compaction_only_gb": compaction_gb,
        "entry_count": _safe_int(summary.get("entry_count"), 0),
        "omitted_count": _safe_int(summary.get("omitted_count"), 0),
        "delete_requires": contract.get("delete_requires", []),
        "never_delete_classes": contract.get("never_delete_classes", []),
        "stateful_sql_policy": contract.get("stateful_sql_policy", ""),
        "next_action": next_action,
    }


def _soak_storage_controls(
    *,
    forecast: dict[str, Any],
    storage_payload: dict[str, Any],
    quota_payload: dict[str, Any],
    hot_lane_payload: dict[str, Any],
    target_free_gb: float,
    pressure_free_gb: float,
    safety_buffer_gb: float,
) -> dict[str, Any]:
    storage_efficiency = (
        storage_payload.get("storage_efficiency_contract")
        if isinstance(storage_payload.get("storage_efficiency_contract"), dict)
        else {}
    )
    metrics = storage_efficiency.get("metrics") if isinstance(storage_efficiency.get("metrics"), dict) else {}
    steady_state = storage_payload.get("steady_state") if isinstance(storage_payload.get("steady_state"), dict) else {}
    steady_target = steady_state.get("target_status") if isinstance(steady_state.get("target_status"), dict) else {}
    storage_section = storage_payload.get("storage") if isinstance(storage_payload.get("storage"), dict) else {}
    collector_audit = (
        storage_payload.get("collector_intake_enforcement_audit")
        if isinstance(storage_payload.get("collector_intake_enforcement_audit"), dict)
        else {}
    )
    route = (
        storage_payload.get("external_route_verification")
        if isinstance(storage_payload.get("external_route_verification"), dict)
        else {}
    )
    resilience = storage_payload.get("storage_resilience") if isinstance(storage_payload.get("storage_resilience"), dict) else {}
    current_free = _safe_float(forecast.get("current_external_free_gb"), 0.0)
    retention_debt = _safe_float(storage_section.get("retention_debt_gb"), 0.0)
    retention_target = max(_safe_float(storage_section.get("retention_debt_target_gb"), 0.25), 0.25)
    raw_candidate_gb = _safe_float(metrics.get("raw_compression_candidate_gb"), 0.0)
    fallback_count = _safe_int(metrics.get("local_fallback_reconciliation_count"), 0)
    sparse_pending_bytes = _safe_float(metrics.get("sparse_large_line_pending_bytes"), 0.0)
    max_sparse_pending_bytes = max(
        _safe_float(os.getenv("STORAGE_CONTINUOUS_RUN_MAX_SPARSE_PENDING_BYTES"), 256.0 * 1024.0 * 1024.0),
        1.0,
    )
    storage_efficiency_ready = bool(
        str(storage_efficiency.get("overall_status") or "").strip().lower() == "ready"
        and _grade_rank(storage_efficiency.get("grade")) >= _grade_rank("A")
    )
    hot_lane_status = str(hot_lane_payload.get("overall_status") or "").strip().lower()
    external_free_above_target = current_free >= max(float(target_free_gb), float(pressure_free_gb) + float(safety_buffer_gb))
    quota_status = str(quota_payload.get("overall_status") or "").strip().lower()
    quota_ready = quota_status == "ready" or (quota_status in {"degraded", "watch", "needs_work"} and external_free_above_target)
    collector_status = str(collector_audit.get("status") or "").strip().lower()
    collector_required = bool(collector_audit.get("required", False))
    collector_mismatch_count = _safe_int(collector_audit.get("mismatch_count"), 0)
    collector_enforced = collector_status == "enforced"
    backlog_relief_contract = (
        storage_payload.get("backlog_relief_contract")
        if isinstance(storage_payload.get("backlog_relief_contract"), dict)
        else {}
    )
    collector_safely_optional = bool(
        collector_status == "not_required"
        and not collector_required
        and collector_mismatch_count <= 0
        and not bool(backlog_relief_contract.get("active", False))
    )
    collector_soak_safe = bool(collector_enforced or collector_safely_optional)
    controls = {
        "storage_efficiency_ready": storage_efficiency_ready,
        "quota_ready": quota_ready,
        "quota_status": quota_status,
        "route_verified": str(route.get("verification_state") or "").strip().lower() == "ready"
        or bool(route.get("coverage_ratio") == 1.0),
        "resilience_ready": str(resilience.get("overall_status") or "").strip().lower() in {"", "ready"},
        "steady_state_ready": bool(steady_target.get("steady_state_ready", False)),
        "retention_debt_ok": retention_debt <= retention_target,
        "collector_intake_enforced": collector_soak_safe,
        "collector_intake_status": collector_status,
        "collector_intake_required": collector_required,
        "collector_intake_mismatch_count": collector_mismatch_count,
        "collector_intake_soak_safe": collector_soak_safe,
        "manifest_first_storage": str(storage_efficiency.get("raw_payload_policy") or "").strip()
        in {"manifest_first", "manifest_first_compress_old_sources"}
        or str(storage_efficiency.get("write_intake_mode") or "").strip() == "thin_digest_with_manifest",
        "raw_candidate_compaction_ok": raw_candidate_gb <= 1.0 and fallback_count == 0,
        "sparse_large_line_pending_bounded": sparse_pending_bytes <= max_sparse_pending_bytes,
        "deep_cold_ready": bool(metrics.get("deep_cold_ready", False)),
        "hot_lane_retention_active": hot_lane_status in {"active", "ready", "watch", "watching"},
        "external_free_above_target": external_free_above_target,
        "current_external_free_gb": round(current_free, 3),
        "target_free_gb": round(float(target_free_gb), 3),
        "retention_debt_gb": round(retention_debt, 3),
        "retention_debt_target_gb": round(retention_target, 3),
        "raw_compression_candidate_gb": round(raw_candidate_gb, 3),
        "local_fallback_reconciliation_count": fallback_count,
        "sparse_large_line_pending_bytes": int(sparse_pending_bytes),
        "max_sparse_large_line_pending_bytes": int(max_sparse_pending_bytes),
        "storage_efficiency_grade": str(storage_efficiency.get("grade") or ""),
        "hot_lane_status": hot_lane_status,
    }
    controls["storage_governed_core_ready"] = bool(
        controls["storage_efficiency_ready"]
        and controls["quota_ready"]
        and controls["route_verified"]
        and controls["resilience_ready"]
        and controls["steady_state_ready"]
        and controls["retention_debt_ok"]
        and controls["collector_intake_enforced"]
        and controls["raw_candidate_compaction_ok"]
        and controls["sparse_large_line_pending_bounded"]
        and controls["deep_cold_ready"]
        and controls["hot_lane_retention_active"]
        and controls["external_free_above_target"]
    )
    controls["storage_governed_ready"] = bool(
        controls["storage_governed_core_ready"]
        and controls["manifest_first_storage"]
    )
    controls["storage_bounded_post_maintenance_ready"] = bool(
        controls["storage_governed_core_ready"]
        and not controls["manifest_first_storage"]
    )
    return controls


def _append_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    disk = payload.get("disk") if isinstance(payload.get("disk"), dict) else {}
    record = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "overall_status": payload.get("overall_status"),
        "overall_score": payload.get("overall_score"),
        "overall_grade": payload.get("overall_grade"),
        "disk": disk,
        "forecast": payload.get("storage_growth_forecast", {}),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    raw_max_files: int = 4,
    raw_max_gb: float = 4.0,
    raw_min_age_hours: float = 18.0,
    cleanup_max_tier: int = 1,
    cleanup_max_delete_gb: float = 48.0,
    telemetry_max_files: int = 12,
    telemetry_max_gb: float = 16.0,
    telemetry_min_file_mb: float = 128.0,
    lifecycle_max_files: int = 80,
    lifecycle_max_gb: float = 4.0,
    lifecycle_min_file_mb: float = 5.0,
    lifecycle_min_age_hours: float = 12.0,
    decision_max_files: int = 12,
    decision_max_gb: float = 8.0,
    decision_min_file_mb: float = 128.0,
    decision_min_age_minutes: float = 90.0,
    cold_archive_max_files: int = 8,
    cold_archive_max_gb: float = 16.0,
    cold_archive_min_age_hours: float = 24.0,
    cold_archive_compression_level: int = 3,
    target_free_gb: float = 125.0,
    pressure_free_gb: float = 64.0,
    soak_days: float = DEFAULT_CONTINUOUS_RUN_DAYS,
    soak_buffer_gb: float = DEFAULT_CONTINUOUS_RUN_BUFFER_GB,
    soak_min_daily_gb: float = DEFAULT_CONTINUOUS_RUN_MIN_DAILY_GB,
    timeout_sec: int = 180,
    out_path: Path = DEFAULT_OUT_PATH,
    history_path: Path = DEFAULT_HISTORY_PATH,
    forecast_path: Path = DEFAULT_FORECAST_PATH,
    capacity_event_path: Path | None = None,
) -> dict[str, Any]:
    external = resolve_external_storage()
    external_root = external.external_root
    disk = {
        "external": _disk_snapshot(external_root),
        "internal_project": _disk_snapshot(project_root),
    }
    local_reserve = local_storage_reserve_contract(project_root)
    history_rows = _read_history(history_path)
    effective_capacity_event_path = (
        capacity_event_path
        if capacity_event_path is not None
        else project_root / "governance" / "runtime" / "storage_capacity_control_epoch_latest.json"
    )
    baseline_control = _storage_growth_baseline_control(
        project_root,
        external_root,
        apply=apply,
        event_path=effective_capacity_event_path,
    )
    baseline_epoch = _parse_ts(baseline_control.get("baseline_not_before_utc"))
    forecast = _storage_growth_forecast(
        current_external=disk["external"],
        current_internal=disk["internal_project"],
        history_rows=history_rows,
        target_free_gb=float(target_free_gb),
        pressure_free_gb=float(pressure_free_gb),
        baseline_not_before_utc=baseline_epoch,
        baseline_scope=str(baseline_control.get("baseline_scope") or "post_control_epoch"),
    )
    forecast["baseline_control"] = baseline_control
    # Deep-cold runs as a child process and must see this pass's disk slope,
    # not the previous retention pass's forecast.
    write_payload(forecast_path, forecast)
    continuous_run = _continuous_run_contract(
        forecast=forecast,
        horizon_days=float(soak_days),
        pressure_free_gb=float(pressure_free_gb),
        safety_buffer_gb=float(soak_buffer_gb),
        min_daily_growth_gb=float(soak_min_daily_gb),
        duty_cycle_max_active_ratio=_safe_float(os.getenv("BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"), 0.16),
    )
    external_free_gb = _safe_float(disk["external"].get("free_gb"), 0.0)
    effective_cleanup_max_tier = max(int(cleanup_max_tier), 1)
    if external_free_gb <= float(pressure_free_gb):
        effective_cleanup_max_tier = max(effective_cleanup_max_tier, 2)
    if external_free_gb <= max(float(pressure_free_gb) * 0.5, 1.0):
        effective_cleanup_max_tier = max(effective_cleanup_max_tier, 2)

    opsctl = str(project_root / "scripts" / "ops" / "opsctl.sh")
    steps: dict[str, dict[str, Any]] = {}
    cold_archive_root = str(os.getenv("BOT_SECOND_COLD_ROOT", "") or "").strip()
    if not cold_archive_root:
        external_archive_root = str(os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "") or "").strip()
        cold_archive_root = str(Path(external_archive_root) / "cold_archive") if external_archive_root else str(project_root / "governance" / "archive" / "cold_archive")
    cold_archive_path = Path(cold_archive_root).expanduser()
    if apply and not _is_protected_volume(cold_archive_path) and cold_archive_path.parent.exists():
        cold_archive_path.mkdir(parents=True, exist_ok=True)

    deep_cmd = [opsctl, "deep-cold-storage-layer", "--json"]
    if apply:
        deep_cmd.insert(2, "--apply")
        second_cold_root = str(os.getenv("BOT_SECOND_COLD_ROOT", "") or "").strip()
        if second_cold_root and not _is_protected_volume(Path(second_cold_root)):
            deep_cmd.extend(
                [
                    "--move-to-second-cold",
                    "--adaptive",
                    "--second-cold-root",
                    second_cold_root,
                    "--planning-horizon-days",
                    str(max(float(soak_days), 1.0)),
                    "--source-free-target-gb",
                    str(max(float(target_free_gb), 0.0)),
                    "--max-move-gb",
                    os.getenv("BOT_DEEP_COLD_MAX_MOVE_GB", "96.0"),
                    "--max-move-files",
                    os.getenv("BOT_DEEP_COLD_MAX_MOVE_FILES", "500"),
                ]
            )
            if _env_truthy("BOT_DEEP_COLD_INCLUDE_CRITICAL"):
                deep_cmd.append("--include-critical")
    deep_cold_timeout = max(int(timeout_sec), 1800) if apply and "--move-to-second-cold" in deep_cmd else int(timeout_sec)
    steps["retention_freshness_deep_cold"] = _run_json(
        deep_cmd,
        cwd=project_root,
        timeout_sec=deep_cold_timeout,
    )

    cold_archive_cmd = [
        opsctl,
        "cold-archive-compactor",
        "--archive-root",
        cold_archive_root,
        "--max-files",
        str(max(int(cold_archive_max_files), 1)),
        "--max-raw-gb",
        str(max(float(cold_archive_max_gb), 0.1)),
        "--min-age-hours",
        str(max(float(cold_archive_min_age_hours), 1.0)),
        "--compression-level",
        str(min(max(int(cold_archive_compression_level), 1), 9)),
        "--sqlite-inventory-limit",
        "200",
        "--json",
    ]
    if apply:
        cold_archive_cmd.insert(2, "--apply")
        cold_archive_cmd.insert(3, "--coordinate-writer-handoff")
    steps["cold_archive_compaction"] = _run_json(
        cold_archive_cmd,
        cwd=project_root,
        timeout_sec=max(int(timeout_sec), 1800),
    )

    retention_cmd = [opsctl, "retention-intelligence-v2", "--json"]
    if apply:
        retention_cmd.insert(2, "--apply")
    steps["retention_freshness_v2"] = _run_json(retention_cmd, cwd=project_root, timeout_sec=timeout_sec)

    raw_cmd = [
        opsctl,
        "raw-training-compaction",
        "--max-files",
        str(max(int(raw_max_files), 1)),
        "--max-gb",
        str(max(float(raw_max_gb), 0.1)),
        "--min-age-hours",
        str(max(float(raw_min_age_hours), 1.0)),
        "--write-history",
        "--json",
    ]
    if apply:
        raw_cmd.insert(2, "--apply")
    steps["raw_training_usefulness"] = _run_json(raw_cmd, cwd=project_root, timeout_sec=timeout_sec)

    cleanup_cmd = [
        opsctl,
        "bot-logs-cleanup-intelligence",
        "--target-free-gb",
        str(max(float(target_free_gb), 1.0)),
        "--max-tier",
        str(effective_cleanup_max_tier),
        "--max-delete-gb",
        str(max(float(cleanup_max_delete_gb), 0.0)),
        "--json",
    ]
    if apply:
        cleanup_cmd.insert(2, "--apply")
    steps["bot_logs_lean"] = _run_json(cleanup_cmd, cwd=project_root, timeout_sec=timeout_sec)

    telemetry_cmd = [
        opsctl,
        "governance-telemetry-compactor",
        "--target-free-gb",
        str(max(float(telemetry_max_gb), 0.1)),
        "--min-file-mb",
        str(max(float(telemetry_min_file_mb), 1.0)),
        "--max-files",
        str(max(int(telemetry_max_files), 1)),
        "--json",
    ]
    if apply:
        telemetry_cmd.insert(2, "--apply")
    steps["governance_telemetry_compactor"] = _run_json(
        telemetry_cmd,
        cwd=project_root,
        timeout_sec=max(int(timeout_sec), 1800),
    )

    external_governance_archive_root = project_root / "data" / "stale_stage" / "external_governance_telemetry_compactor"
    if external_root.exists() and not _same_real_path(external_root, project_root):
        external_telemetry_cmd = [
            opsctl,
            "governance-telemetry-compactor",
            "--project-root",
            str(external_root),
            "--archive-root",
            str(external_governance_archive_root),
            "--target-free-gb",
            str(max(float(telemetry_max_gb), 0.1)),
            "--min-file-mb",
            str(max(float(telemetry_min_file_mb), 1.0)),
            "--max-files",
            str(max(int(telemetry_max_files), 1)),
            "--json",
        ]
        if apply:
            external_telemetry_cmd.insert(2, "--apply")
        steps["external_governance_telemetry_compactor"] = _run_json(
            external_telemetry_cmd,
            cwd=project_root,
            timeout_sec=max(int(timeout_sec), 1800),
        )
    else:
        steps["external_governance_telemetry_compactor"] = _synthetic_step(
            command=[
                opsctl,
                "governance-telemetry-compactor",
                "--project-root",
                str(external_root),
                "--archive-root",
                str(external_governance_archive_root),
            ],
            overall_status="nothing_to_do",
            payload={
                "ok": True,
                "overall_status": "nothing_to_do",
                "summary": {
                    "candidate_count": 0,
                    "selected_count": 0,
                    "selected_gb": 0.0,
                    "estimated_hot_reduction_gb": 0.0,
                    "error_count": 0,
                },
            },
        )

    lifecycle_cmd = [
        opsctl,
        "governance-lifecycle-compactor",
        "--target-free-gb",
        str(max(float(lifecycle_max_gb), 0.1)),
        "--min-file-mb",
        str(max(float(lifecycle_min_file_mb), 1.0)),
        "--max-files",
        str(max(int(lifecycle_max_files), 1)),
        "--min-age-hours",
        str(max(float(lifecycle_min_age_hours), 1.0)),
        "--json",
    ]
    if apply:
        lifecycle_cmd.insert(2, "--apply")
    steps["governance_lifecycle_compactor"] = _run_json(
        lifecycle_cmd,
        cwd=project_root,
        timeout_sec=max(int(timeout_sec), 900),
    )

    decision_cmd = [
        opsctl,
        "decision-log-compactor",
        "--target-free-gb",
        str(max(float(decision_max_gb), 0.1)),
        "--min-file-mb",
        str(max(float(decision_min_file_mb), 1.0)),
        "--max-files",
        str(max(int(decision_max_files), 1)),
        "--min-age-minutes",
        str(max(float(decision_min_age_minutes), 1.0)),
        "--json",
    ]
    if apply:
        decision_cmd.insert(2, "--apply")
    steps["decision_log_compactor"] = _run_json(
        decision_cmd,
        cwd=project_root,
        timeout_sec=max(int(timeout_sec), 900),
    )

    steps["storage_tier_policy"] = _run_json(
        [opsctl, "storage-tier-policy", "--json"],
        cwd=project_root,
        timeout_sec=timeout_sec,
    )

    hot_lane_cmd = [
        opsctl,
        "hot-lane-retention-control",
        "--target-free-gb",
        str(max(float(target_free_gb), 1.0)),
        "--pressure-free-gb",
        str(max(float(pressure_free_gb), 1.0)),
        "--json",
    ]
    if apply:
        hot_lane_cmd.insert(2, "--apply")
    steps["hot_lane_retention"] = _run_json(hot_lane_cmd, cwd=project_root, timeout_sec=timeout_sec)

    hot_lane_now = steps["hot_lane_retention"].get("payload") or {}
    current_day_rotation_ready = bool(
        apply
        and str(hot_lane_now.get("mode") or "").strip().lower() == "emergency_hot_thin"
    )
    if current_day_rotation_ready:
        current_explanation_cmd = [
            opsctl,
            "decision-log-compactor",
            "--apply",
            "--include-current-day",
            "--require-current-day-safe",
            "--families",
            "decision_explanations",
            "--target-free-gb",
            str(max(float(decision_max_gb), 8.0)),
            "--min-file-mb",
            str(max(float(decision_min_file_mb), 32.0)),
            "--max-files",
            str(max(min(int(decision_max_files), 2), 1)),
            "--min-age-minutes",
            "20",
            "--compression-level",
            "1",
            "--json",
        ]
        steps["current_day_explanation_compactor"] = _run_json(
            current_explanation_cmd,
            cwd=project_root,
            timeout_sec=max(int(timeout_sec), 1800),
        )
        # Recalculate tier placement after an emergency hot-buffer rotation so
        # quota and soak reports never grade the pre-compaction byte layout.
        steps["storage_tier_policy"] = _run_json(
            [opsctl, "storage-tier-policy", "--json"],
            cwd=project_root,
            timeout_sec=timeout_sec,
        )
    else:
        steps["current_day_explanation_compactor"] = _synthetic_step(
            command=[
                opsctl,
                "decision-log-compactor",
                "--include-current-day",
                "--require-current-day-safe",
                "--families",
                "decision_explanations",
            ],
            overall_status="nothing_to_do",
            payload={
                "ok": True,
                "overall_status": "nothing_to_do",
                "reason": "hot_lane_not_in_emergency_rotation_mode",
            },
        )

    creative_cmd = [opsctl, "creative-cotenant-guard", "apply" if apply else "status", "--json"]
    steps["foreground_app_protection"] = _run_json(creative_cmd, cwd=project_root, timeout_sec=timeout_sec)

    steps["storage_quota_guard"] = _run_json([opsctl, "storage-quota-guard", "--json"], cwd=project_root, timeout_sec=timeout_sec)
    steps["ingestion_storage_control"] = _run_json([opsctl, "ingestion-storage-control", "--json"], cwd=project_root, timeout_sec=timeout_sec)

    deep_payload = steps["retention_freshness_deep_cold"].get("payload") or {}
    cold_archive_payload = steps["cold_archive_compaction"].get("payload") or {}
    retention_payload = steps["retention_freshness_v2"].get("payload") or {}
    raw_payload = steps["raw_training_usefulness"].get("payload") or {}
    cleanup_payload = steps["bot_logs_lean"].get("payload") or {}
    telemetry_payload = steps["governance_telemetry_compactor"].get("payload") or {}
    external_telemetry_payload = steps["external_governance_telemetry_compactor"].get("payload") or {}
    lifecycle_payload = steps["governance_lifecycle_compactor"].get("payload") or {}
    decision_compactor_payload = steps["decision_log_compactor"].get("payload") or {}
    tier_payload = steps["storage_tier_policy"].get("payload") or {}
    hot_lane_payload = steps["hot_lane_retention"].get("payload") or {}
    creative_payload = steps["foreground_app_protection"].get("payload") or {}
    storage_payload = steps["ingestion_storage_control"].get("payload") or {}
    quota_payload = steps["storage_quota_guard"].get("payload") or {}
    second_cold = _second_cold_preflight()

    retention_report = retention_payload.get("retention_report_card") if isinstance(retention_payload.get("retention_report_card"), dict) else {}
    raw_summary = raw_payload.get("raw_summary") if isinstance(raw_payload.get("raw_summary"), dict) else {}
    cleanup_retention = cleanup_payload.get("retention_intelligence_v2") if isinstance(cleanup_payload.get("retention_intelligence_v2"), dict) else {}
    storage_efficiency = (
        storage_payload.get("storage_efficiency_contract")
        if isinstance(storage_payload.get("storage_efficiency_contract"), dict)
        else {}
    )
    external_free_gb = _safe_float(forecast.get("current_external_free_gb"), 0.0)
    external_free_above_target = external_free_gb >= max(float(target_free_gb), float(pressure_free_gb) + float(soak_buffer_gb))
    quota_status = str(quota_payload.get("overall_status") or "").strip().lower()
    quota_ready = quota_status == "ready" or (quota_status in {"degraded", "watch", "needs_work"} and external_free_above_target)
    soak_storage_controls = _soak_storage_controls(
        forecast=forecast,
        storage_payload=storage_payload,
        quota_payload=quota_payload,
        hot_lane_payload=hot_lane_payload,
        target_free_gb=float(target_free_gb),
        pressure_free_gb=float(pressure_free_gb),
        safety_buffer_gb=float(soak_buffer_gb),
    )
    continuous_run = _continuous_run_contract(
        forecast=forecast,
        horizon_days=float(soak_days),
        pressure_free_gb=float(pressure_free_gb),
        safety_buffer_gb=float(soak_buffer_gb),
        min_daily_growth_gb=float(soak_min_daily_gb),
        duty_cycle_max_active_ratio=_safe_float(os.getenv("BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"), 0.16),
        storage_controls=soak_storage_controls,
    )
    continuous_run = _apply_cold_archive_spillover_contract(continuous_run, second_cold)
    quota_managed_by_cold_spillover = _sql_soft_quota_managed_by_cold_spillover(quota_payload, continuous_run)
    quota_ready = bool(quota_ready or quota_managed_by_cold_spillover)
    hot_plane_compaction = _hot_plane_compaction_contract(
        steps_by_lane={
            "governance_telemetry_compactor": steps["governance_telemetry_compactor"],
            "external_governance_telemetry_compactor": steps["external_governance_telemetry_compactor"],
            "governance_lifecycle_compactor": steps["governance_lifecycle_compactor"],
            "decision_log_compactor": steps["decision_log_compactor"],
            "current_day_explanation_compactor": steps["current_day_explanation_compactor"],
        }
    )
    manifest_backed_offload = _manifest_backed_offload_evidence(tier_payload)

    non_hard_command_statuses = {
        "already_running",
        "applied",
        "busy",
        "deferred_archive_unavailable",
        "deferred_existing_maintenance_hold",
        "deferred_writer_handoff_timeout",
        "deferred_writer_busy",
        "nothing_to_do",
        "planned",
        "ready",
        "needs_work",
        "degraded",
        "watch",
        "advisory",
        "ready_with_paper_lane_advisory",
    }
    command_failures = [
        name
        for name, step in steps.items()
        if int(step.get("returncode", 0)) not in {0}
        and str(step.get("overall_status") or "") not in non_hard_command_statuses
        and not (name == "bot_logs_lean" and str(step.get("overall_status") or "") == "blocked")
        and not (name == "foreground_app_protection" and bool(step.get("timed_out", False)))
        and not _deep_cold_needs_data_is_advisory(name, step)
    ]
    protected_external = bool(disk["external"].get("protected", False))
    raw_eligible_count = _safe_int(raw_summary.get("eligible_training_source_count"), 0)
    raw_blockers = (
        ((raw_payload.get("decision_packet") or {}).get("blocked_reasons"))
        if isinstance(raw_payload.get("decision_packet"), dict)
        else []
    )
    material_raw_blockers = [
        str(item)
        for item in raw_blockers
        if str(item) and str(item) != "raw_compaction_not_applied"
    ]
    raw_training_status = (
        "ready"
        if raw_eligible_count > 0 and not material_raw_blockers
        else str(raw_payload.get("overall_status") or "unknown")
    )
    foreground_step = steps["foreground_app_protection"]
    creative_actions = creative_payload.get("actions") if isinstance(creative_payload.get("actions"), list) else []
    foreground_status = str(creative_payload.get("overall_status") or "unknown")
    if bool(foreground_step.get("timed_out", False)):
        foreground_status = "advisory"
        creative_actions = ordered_unique([*map(str, creative_actions), "foreground_guard_timeout"])
    elif foreground_status == "needs_work" and set(str(item) for item in creative_actions).issubset({"paper_execution_lane_missing"}):
        foreground_status = "ready_with_paper_lane_advisory"

    cold_archive_status = str(cold_archive_payload.get("overall_status") or "unknown")
    cold_archive_summary = (
        cold_archive_payload.get("summary") if isinstance(cold_archive_payload.get("summary"), dict) else {}
    )
    cold_archive_score = {
        "ready": 99.0,
        "applied": 99.0,
        "planned": 97.0,
        "deferred_existing_maintenance_hold": 96.0,
        "deferred_writer_handoff_timeout": 96.0,
        "deferred_writer_busy": 97.0,
        "deferred_archive_unavailable": 92.0,
        "busy": 94.0,
        "advisory": 90.0,
    }.get(cold_archive_status, 82.0)

    sections = {
        "local_hot_storage_reserve": _section(
            "Local Hot Storage Reserve",
            str(local_reserve.get("status") or "unknown"),
            100.0
            if bool(local_reserve.get("ready", False))
            else (92.0 if not bool(local_reserve.get("pressure_active", False)) else 72.0),
            local_reserve,
            str(local_reserve.get("next_action") or "restore the live internal reserve"),
        ),
        "retention_freshness": _section(
            "Retention Freshness",
            "ready" if bool(retention_payload.get("ok", False)) and bool(deep_payload.get("ok", False)) else "needs_refresh",
            min(_safe_float(retention_report.get("overall_score"), 0.0), 99.0) if retention_report else 92.0,
            {
                "deep_cold_status": deep_payload.get("overall_status", ""),
                "retention_status": retention_payload.get("overall_status", ""),
                "retention_grade": retention_report.get("overall_grade", ""),
                "manifest_path": deep_payload.get("manifest_path", ""),
            },
            "deep cold and retention report card are refreshed",
        ),
        "raw_training_usefulness": _section(
            "Raw Data Training Usefulness",
            raw_training_status,
            98.0
            if raw_eligible_count > 0
            else (94.0 if _safe_int(raw_summary.get("raw_jsonl_count"), 0) > 0 else 88.0),
            {
                "raw_jsonl_count": _safe_int(raw_summary.get("raw_jsonl_count"), 0),
                "eligible_training_source_count": _safe_int(raw_summary.get("eligible_training_source_count"), 0),
                "compression_candidate_gb": _safe_float(raw_summary.get("compression_candidate_gb"), 0.0),
                "raw_gb_cleared": _safe_float(raw_summary.get("raw_gb_cleared"), 0.0),
            },
            "raw sources are queued manifest-only for training, then compacted only by verified gzip waves",
        ),
        "cold_archive_compaction": _section(
            "Cold Archive Compaction",
            cold_archive_status,
            cold_archive_score,
            {
                "archive_root": cold_archive_payload.get("archive_root", cold_archive_root),
                "jsonl_candidate_count": _safe_int(cold_archive_summary.get("jsonl_candidate_count"), 0),
                "selected_jsonl_count": _safe_int(cold_archive_summary.get("selected_jsonl_count"), 0),
                "gzip_finalize_candidate_count": _safe_int(
                    cold_archive_summary.get("gzip_finalize_candidate_count"), 0
                ),
                "selected_gzip_finalize_count": _safe_int(
                    cold_archive_summary.get("selected_gzip_finalize_count"), 0
                ),
                "tmp_duplicate_candidate_count": _safe_int(
                    cold_archive_summary.get("tmp_duplicate_candidate_count"), 0
                ),
                "sqlite_inventory_count": _safe_int(cold_archive_summary.get("sqlite_inventory_count"), 0),
                "sqlite_vacuum_eligible_count": _safe_int(
                    cold_archive_summary.get("sqlite_vacuum_eligible_count"), 0
                ),
                "successful_action_count": _safe_int(cold_archive_summary.get("successful_action_count"), 0),
                "error_count": _safe_int(cold_archive_summary.get("error_count"), 0),
                "released_gb": _safe_float(cold_archive_summary.get("released_gb"), 0.0),
                "manifest_path": cold_archive_payload.get("manifest_path", ""),
                "readme_path": cold_archive_payload.get("readme_path", ""),
            },
            str(
                cold_archive_payload.get("next_action")
                or "retry the bounded archive pass when its mount and writer guards permit it"
            ),
        ),
        "bot_logs_lean": _section(
            "Active BOT_LOGS Lean",
            str(cleanup_payload.get("overall_status") or "unknown"),
            99.0
            if str(cleanup_payload.get("overall_status") or "") == "ready"
            and _safe_float(cleanup_payload.get("projected_free_gb"), 0.0) >= float(target_free_gb)
            else 88.0,
            {
                "projected_free_gb": _safe_float(cleanup_payload.get("projected_free_gb"), 0.0),
                "selected_count": _safe_int(cleanup_payload.get("selected_count"), 0),
                "effective_max_tier": effective_cleanup_max_tier,
                "max_delete_gb": round(max(float(cleanup_max_delete_gb), 0.0), 3),
                "retention_ready": bool(cleanup_retention.get("ready", False)),
            },
            "BOT_LOGS cleanup stays tiered and value-aware; pressure automatically unlocks bounded stale-stage cleanup while current-day files remain protected",
        ),
        "hot_plane_compaction": _section(
            "Hot Plane Compaction",
            str(hot_plane_compaction.get("status") or "unknown"),
            _safe_float(hot_plane_compaction.get("score"), 0.0),
            {
                "governance_telemetry_status": telemetry_payload.get("overall_status", ""),
                "external_governance_telemetry_status": external_telemetry_payload.get("overall_status", ""),
                "governance_lifecycle_status": lifecycle_payload.get("overall_status", ""),
                "decision_log_status": decision_compactor_payload.get("overall_status", ""),
                "candidate_count": _safe_int(hot_plane_compaction.get("candidate_count"), 0),
                "selected_gb": _safe_float(hot_plane_compaction.get("selected_gb"), 0.0),
                "estimated_reduction_gb": _safe_float(hot_plane_compaction.get("estimated_reduction_gb"), 0.0),
                "errors": hot_plane_compaction.get("errors", []),
            },
            str(hot_plane_compaction.get("next_action") or "keep hot-plane compaction integrated with retention unison"),
        ),
        "manifest_backed_offload": _section(
            "Manifest-Backed Offload Contract",
            str(manifest_backed_offload.get("status") or "unknown"),
            _safe_float(manifest_backed_offload.get("score"), 0.0),
            {
                "manifest_path": manifest_backed_offload.get("manifest_path", ""),
                "eligible_offload_files": _safe_int(manifest_backed_offload.get("eligible_offload_files"), 0),
                "eligible_offload_gb": _safe_float(manifest_backed_offload.get("eligible_offload_gb"), 0.0),
                "compaction_only_files": _safe_int(manifest_backed_offload.get("compaction_only_files"), 0),
                "compaction_only_gb": _safe_float(manifest_backed_offload.get("compaction_only_gb"), 0.0),
                "entry_count": _safe_int(manifest_backed_offload.get("entry_count"), 0),
                "omitted_count": _safe_int(manifest_backed_offload.get("omitted_count"), 0),
                "delete_requires": manifest_backed_offload.get("delete_requires", []),
                "never_delete_classes": manifest_backed_offload.get("never_delete_classes", []),
                "stateful_sql_policy": manifest_backed_offload.get("stateful_sql_policy", ""),
            },
            str(manifest_backed_offload.get("next_action") or "refresh storage-tier-policy"),
        ),
        "hot_lane_retention": _section(
            "Hot Lane Retention Control",
            str(hot_lane_payload.get("overall_status") or "unknown"),
            _safe_float(hot_lane_payload.get("overall_score"), 92.0),
            {
                "mode": hot_lane_payload.get("mode", ""),
                "reasons": hot_lane_payload.get("reasons", []),
                "active_decision_gb": _safe_float(((hot_lane_payload.get("hot_decision_pressure") or {}).get("active_decision_gb")), 0.0)
                if isinstance(hot_lane_payload.get("hot_decision_pressure"), dict)
                else 0.0,
                "largest_active_file_gb": _safe_float(((hot_lane_payload.get("hot_decision_pressure") or {}).get("largest_active_file_gb")), 0.0)
                if isinstance(hot_lane_payload.get("hot_decision_pressure"), dict)
                else 0.0,
                "override_applied": bool(hot_lane_payload.get("override_applied", False)),
                "storage_tier_status": tier_payload.get("overall_status", ""),
                "live_hot_path_gb": round(
                    _safe_float(((tier_payload.get("pressure") or {}).get("live_hot_path_bytes")), 0.0) / (1024.0**3),
                    4,
                )
                if isinstance(tier_payload.get("pressure"), dict)
                else 0.0,
            },
            str(hot_lane_payload.get("next_action") or "keep hot-lane retention watching current-day decision growth"),
        ),
        "storage_forecast": _section(
            "Storage Growth Forecast",
            str(forecast.get("status") or ""),
            _safe_float(forecast.get("score"), 0.0),
            forecast,
            str(forecast.get("next_action") or ""),
        ),
        "continuous_run_soak": _section(
            "30-Day Continuous Collection Soak",
            str(continuous_run.get("status") or ""),
            _safe_float(continuous_run.get("score"), 0.0),
            continuous_run,
            str(continuous_run.get("next_action") or ""),
        ),
        "training_efficiency": _section(
            "Training Batch Efficiency",
            "ready" if raw_eligible_count > 0 and quota_ready else "advisory",
            97.0 if raw_eligible_count > 0 and quota_ready else 90.0,
            {
                "quota_ready": quota_ready,
                "quota_managed_by_cold_spillover": quota_managed_by_cold_spillover,
                "raw_source_queue": ((raw_payload.get("next_training_manifest") or {}).get("raw_source_queue_path") if isinstance(raw_payload.get("next_training_manifest"), dict) else ""),
                "eligible_queue": ((raw_payload.get("next_training_manifest") or {}).get("raw_eligible_source_queue_path") if isinstance(raw_payload.get("next_training_manifest"), dict) else ""),
                "storage_efficiency_grade": storage_efficiency.get("grade", ""),
            },
            "training gets manifest queues and compacted evidence instead of dragging huge raw tails through every pass",
        ),
        "second_cold_readiness": _section(
            "Second Cold Target Readiness",
            str(second_cold.get("status") or ""),
            _safe_float(second_cold.get("score"), 0.0),
            second_cold,
            str(second_cold.get("next_action") or ""),
        ),
        "foreground_protection": _section(
            "Foreground App Protection",
            foreground_status,
            98.0 if foreground_status in {"ready", "ready_with_paper_lane_advisory", "needs_work"} else 84.0,
            {
                "creative_mode": creative_payload.get("creative_mode", {}),
                "actions": creative_actions,
                "runtime_throttle": creative_payload.get("runtime_throttle", {}),
            },
            "creative co-tenant guard keeps Logic, Final Cut, Music, and foreground work ahead of optional heavy bot jobs",
        ),
    }

    hard_blockers: list[str] = []
    if protected_external:
        hard_blockers.append("external_storage_points_at_VIDEO")
    if str(second_cold.get("status") or "") == "blocked_protected_target":
        hard_blockers.append("second_cold_target_points_at_VIDEO")
    if command_failures:
        hard_blockers.extend(f"command_failed:{name}" for name in command_failures)
    if not quota_ready:
        hard_blockers.append("storage_quota_not_ready")
    if str(continuous_run.get("status") or "") == "blocked":
        hard_blockers.append("continuous_collection_soak_not_ready")
    if bool(local_reserve.get("pressure_active", False)) or bool(local_reserve.get("hard_block", False)):
        hard_blockers.append("local_hot_storage_pressure_reserve_breached")

    overall_score = round(sum(_safe_float(row.get("score"), 0.0) for row in sections.values()) / max(len(sections), 1), 2)
    if not bool(local_reserve.get("ready", False)):
        overall_score = min(overall_score, 92.0)
    if hard_blockers:
        overall_score = min(overall_score, 82.0)
    overall_status = "ready" if not hard_blockers and overall_score >= 93.0 else ("blocked" if hard_blockers else "needs_work")
    disk_after_work = {
        "external": _disk_snapshot(external_root),
        "internal_project": _disk_snapshot(project_root),
    }

    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": bool(overall_status == "ready"),
        "overall_status": overall_status,
        "overall_score": overall_score,
        "overall_grade": _grade(overall_score),
        "apply": bool(apply),
        "disk": disk,
        "disk_after_work": disk_after_work,
        "local_hot_storage_reserve": local_reserve,
        "storage_growth_baseline_control": baseline_control,
        "storage_growth_forecast": forecast,
        "continuous_run_contract": continuous_run,
        "sections": sections,
        "steps": steps,
        "hard_blockers": hard_blockers,
        "command_failures": command_failures,
        "second_cold_preflight": second_cold,
        "integration_contract": {
            "refreshes_retention_intelligence": True,
            "queues_raw_data_for_training": True,
            "controls_hot_lane_decision_growth": True,
            "keeps_bot_logs_lean": True,
            "compacts_hot_governance_telemetry": True,
            "compacts_external_hot_governance_telemetry": True,
            "compacts_lifecycle_registry_backups": True,
            "compacts_old_decision_logs": True,
            "compacts_cold_archive_losslessly": True,
            "cold_archive_restore_proof_manifest": True,
            "recovers_verified_cold_archive_gzip_orphans": True,
            "coordinates_cold_archive_writer_handoff": True,
            "preserves_direct_archive_readability": True,
            "defers_cold_compaction_while_writer_active": True,
            "uses_manifest_backed_offload_contract": bool(manifest_backed_offload.get("manifest_path")),
            "has_manifest_backed_copy_verify_worker": True,
            "stateful_sql_compaction_only": bool(
                manifest_backed_offload.get("manifest_path")
                and "stateful_sql_compaction_only" in {
                    str(item) for item in (manifest_backed_offload.get("never_delete_classes") or [])
                }
            ),
            "sql_soft_quota_managed_by_cold_spillover": quota_managed_by_cold_spillover,
            "writes_storage_growth_forecast": True,
            "excludes_verified_cross_tier_moves_from_ingestion_growth": True,
            "persists_capacity_control_epoch": True,
            "publishes_growth_forecast_before_deep_cold": True,
            "publishes_continuous_run_contract": True,
            "keeps_training_batches_efficient": True,
            "prewires_second_cold_target": True,
            "protects_foreground_apps": True,
            "never_touch_protected_volumes": list(PROTECTED_VOLUME_PREFIXES),
        },
        "recommended_commands": {
            "repeat_unison_pass": [
                "./scripts/ops/opsctl.sh",
                "storage-retention-unison",
                "--apply",
                "--json",
            ],
            "refresh_retention_only": [
                "./scripts/ops/opsctl.sh",
                "retention-intelligence-v2",
                "--apply",
                "--json",
            ],
            "apply_hot_lane_retention": [
                "./scripts/ops/opsctl.sh",
                "hot-lane-retention-control",
                "--apply",
                "--json",
            ],
            "bounded_raw_training_wave": [
                "./scripts/ops/opsctl.sh",
                "raw-training-compaction",
                "--apply",
                "--max-files",
                str(max(int(raw_max_files), 1)),
                "--max-gb",
                str(max(float(raw_max_gb), 0.1)),
                "--json",
            ],
            "bounded_hot_plane_compaction_wave": [
                "./scripts/ops/opsctl.sh",
                "storage-retention-unison",
                "--apply",
                "--telemetry-max-gb",
                str(max(float(telemetry_max_gb), 0.1)),
                "--lifecycle-max-gb",
                str(max(float(lifecycle_max_gb), 0.1)),
                "--decision-max-gb",
                str(max(float(decision_max_gb), 0.1)),
                "--json",
            ],
            "bounded_cold_archive_compaction_wave": [
                "./scripts/ops/opsctl.sh",
                "cold-archive-compactor",
                "--apply",
                "--archive-root",
                cold_archive_root,
                "--max-files",
                str(max(int(cold_archive_max_files), 1)),
                "--max-raw-gb",
                str(max(float(cold_archive_max_gb), 0.1)),
                "--min-age-hours",
                str(max(float(cold_archive_min_age_hours), 1.0)),
                "--json",
            ],
            "refresh_manifest_backed_offload_contract": [
                "./scripts/ops/opsctl.sh",
                "storage-tier-policy",
                "--json",
            ],
            "manifest_backed_copy_verify_wave": [
                "./scripts/ops/opsctl.sh",
                "manifest-backed-offload",
                "--apply",
                "--max-files",
                "4",
                "--max-gb",
                "4.0",
                "--json",
            ],
        },
        "control_env": {
            "BOT_RETENTION_INTELLIGENCE_V2_ACTIVE": "1",
            "BOT_STORAGE_GROWTH_FORECAST_ACTIVE": "1",
            "BOT_CONTINUOUS_COLLECTION_SOAK_ACTIVE": "1",
            **{
                str(key): str(value)
                for key, value in (continuous_run.get("control_env") if isinstance(continuous_run.get("control_env"), dict) else {}).items()
            },
            "BOT_RAW_TRAINING_MANIFEST_QUEUE_ACTIVE": "1",
            "BOT_HOT_PLANE_COMPACTION_ACTIVE": "1",
            "BOT_COLD_ARCHIVE_COMPACTION_ACTIVE": "1",
            "BOT_COLD_ARCHIVE_COMPACTION_MANIFEST": str(cold_archive_payload.get("manifest_path") or ""),
            "BOT_MANIFEST_BACKED_OFFLOAD_CONTRACT_ACTIVE": "1" if manifest_backed_offload.get("manifest_path") else "0",
            "BOT_MANIFEST_BACKED_OFFLOAD_PATH": str(manifest_backed_offload.get("manifest_path") or ""),
            "BOT_LOGS_PRESSURE_CLEANUP_MAX_TIER": str(effective_cleanup_max_tier),
            "BOT_LOGS_PRESSURE_CLEANUP_MAX_DELETE_GB": str(round(max(float(cleanup_max_delete_gb), 0.0), 3)),
            "BOT_SECOND_COLD_ROOT": os.getenv("BOT_SECOND_COLD_ROOT", "").strip(),
            "BOT_NEVER_TOUCH_VIDEO": "1",
        },
        "next_action": (
            "storage/retention unison is healthy; repeat after big training, cleanup, or reconnect events"
            if overall_status == "ready"
            else "clear hard blockers, then rerun storage-retention-unison --apply"
        ),
    }

    write_payload(out_path, payload)
    write_payload(forecast_path, forecast)
    if apply:
        _append_history(history_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Coordinate storage retention, raw training usefulness, lossless cold-archive compaction, "
            "BOT_LOGS cleanup, forecasting, second-cold readiness, and foreground protection."
        )
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--history-file", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--forecast-file", default=str(DEFAULT_FORECAST_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--raw-max-files", type=int, default=int(os.getenv("STORAGE_RETENTION_UNISON_RAW_MAX_FILES", "4")))
    parser.add_argument("--raw-max-gb", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_RAW_MAX_GB", "4.0")))
    parser.add_argument("--raw-min-age-hours", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_RAW_MIN_AGE_HOURS", "18.0")))
    parser.add_argument("--cleanup-max-tier", type=int, default=int(os.getenv("STORAGE_RETENTION_UNISON_CLEANUP_MAX_TIER", "1")))
    parser.add_argument("--cleanup-max-delete-gb", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_CLEANUP_MAX_DELETE_GB", "48.0")))
    parser.add_argument("--telemetry-max-files", type=int, default=int(os.getenv("STORAGE_RETENTION_UNISON_TELEMETRY_MAX_FILES", "12")))
    parser.add_argument("--telemetry-max-gb", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_TELEMETRY_MAX_GB", "16.0")))
    parser.add_argument("--telemetry-min-file-mb", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_TELEMETRY_MIN_FILE_MB", "128.0")))
    parser.add_argument("--lifecycle-max-files", type=int, default=int(os.getenv("STORAGE_RETENTION_UNISON_LIFECYCLE_MAX_FILES", "80")))
    parser.add_argument("--lifecycle-max-gb", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_LIFECYCLE_MAX_GB", "4.0")))
    parser.add_argument("--lifecycle-min-file-mb", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_LIFECYCLE_MIN_FILE_MB", "5.0")))
    parser.add_argument("--lifecycle-min-age-hours", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_LIFECYCLE_MIN_AGE_HOURS", "12.0")))
    parser.add_argument("--decision-max-files", type=int, default=int(os.getenv("STORAGE_RETENTION_UNISON_DECISION_MAX_FILES", "12")))
    parser.add_argument("--decision-max-gb", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_DECISION_MAX_GB", "8.0")))
    parser.add_argument("--decision-min-file-mb", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_DECISION_MIN_FILE_MB", "128.0")))
    parser.add_argument("--decision-min-age-minutes", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_DECISION_MIN_AGE_MINUTES", "90.0")))
    parser.add_argument("--cold-archive-max-files", type=int, default=int(os.getenv("STORAGE_RETENTION_UNISON_COLD_ARCHIVE_MAX_FILES", "8")))
    parser.add_argument("--cold-archive-max-gb", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_COLD_ARCHIVE_MAX_GB", "16.0")))
    parser.add_argument("--cold-archive-min-age-hours", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_COLD_ARCHIVE_MIN_AGE_HOURS", "24.0")))
    parser.add_argument("--cold-archive-compression-level", type=int, default=int(os.getenv("STORAGE_RETENTION_UNISON_COLD_ARCHIVE_COMPRESSION_LEVEL", "3")))
    parser.add_argument("--target-free-gb", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_TARGET_FREE_GB", "125.0")))
    parser.add_argument("--pressure-free-gb", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_PRESSURE_FREE_GB", "64.0")))
    parser.add_argument("--soak-days", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_SOAK_DAYS", str(DEFAULT_CONTINUOUS_RUN_DAYS))))
    parser.add_argument("--soak-buffer-gb", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_SOAK_BUFFER_GB", str(DEFAULT_CONTINUOUS_RUN_BUFFER_GB))))
    parser.add_argument("--soak-min-daily-gb", type=float, default=float(os.getenv("STORAGE_RETENTION_UNISON_SOAK_MIN_DAILY_GB", str(DEFAULT_CONTINUOUS_RUN_MIN_DAILY_GB))))
    parser.add_argument("--timeout-sec", type=int, default=int(os.getenv("STORAGE_RETENTION_UNISON_TIMEOUT_SEC", "180")))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        raw_max_files=int(args.raw_max_files),
        raw_max_gb=float(args.raw_max_gb),
        raw_min_age_hours=float(args.raw_min_age_hours),
        cleanup_max_tier=int(args.cleanup_max_tier),
        cleanup_max_delete_gb=float(args.cleanup_max_delete_gb),
        telemetry_max_files=int(args.telemetry_max_files),
        telemetry_max_gb=float(args.telemetry_max_gb),
        telemetry_min_file_mb=float(args.telemetry_min_file_mb),
        lifecycle_max_files=int(args.lifecycle_max_files),
        lifecycle_max_gb=float(args.lifecycle_max_gb),
        lifecycle_min_file_mb=float(args.lifecycle_min_file_mb),
        lifecycle_min_age_hours=float(args.lifecycle_min_age_hours),
        decision_max_files=int(args.decision_max_files),
        decision_max_gb=float(args.decision_max_gb),
        decision_min_file_mb=float(args.decision_min_file_mb),
        decision_min_age_minutes=float(args.decision_min_age_minutes),
        cold_archive_max_files=int(args.cold_archive_max_files),
        cold_archive_max_gb=float(args.cold_archive_max_gb),
        cold_archive_min_age_hours=float(args.cold_archive_min_age_hours),
        cold_archive_compression_level=int(args.cold_archive_compression_level),
        target_free_gb=float(args.target_free_gb),
        pressure_free_gb=float(args.pressure_free_gb),
        soak_days=float(args.soak_days),
        soak_buffer_gb=float(args.soak_buffer_gb),
        soak_min_daily_gb=float(args.soak_min_daily_gb),
        timeout_sec=int(args.timeout_sec),
        out_path=Path(args.out_file).expanduser(),
        history_path=Path(args.history_file).expanduser(),
        forecast_path=Path(args.forecast_file).expanduser(),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "storage_retention_unison "
            f"status={payload.get('overall_status', '')} "
            f"grade={payload.get('overall_grade', '')} "
            f"score={payload.get('overall_score', 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
