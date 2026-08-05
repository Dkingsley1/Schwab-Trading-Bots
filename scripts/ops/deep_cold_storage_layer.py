#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_maintenance import maintenance_hold_snapshot
    from core.storage_mounts import resolve_external_storage
    from scripts.ops.long_runtime_common import iso_now, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_maintenance import maintenance_hold_snapshot
    from core.storage_mounts import resolve_external_storage
    from .long_runtime_common import iso_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "deep_cold_storage_layer_latest.json"
DEFAULT_MANIFEST_NAME = "deep_cold_manifest.jsonl"
DEFAULT_VIDEO_COLD_ARCHIVE_ROOT = "/Volumes/VIDEO/schwab_trading_bot_cold"
PROTECTED_VOLUME_NAMES = {"VIDEO"}
DEFAULT_ADAPTIVE_FREE_FLOOR_GB = 96.0
DEFAULT_ADAPTIVE_FREE_RATIO = 0.12
DEFAULT_ADAPTIVE_HORIZON_DAYS = 30.0
DEFAULT_ADAPTIVE_GROWTH_FLOOR_GB_PER_DAY = 0.5
DEFAULT_ADAPTIVE_OPERATING_BUFFER_GB = 32.0
DEFAULT_DESTINATION_RESERVE_GB = 64.0
VALUE_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


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


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "y", "on"}


def _lower_runtime_priority() -> dict[str, Any]:
    target = max(_safe_int(os.getenv("BOT_DEEP_COLD_RUNTIME_NICE"), 15), 0)
    try:
        previous = int(os.nice(0))
        current = int(os.nice(max(target - previous, 0))) if target > previous else previous
        return {"applied": current >= target, "previous_nice": previous, "target_nice": target, "current_nice": current}
    except Exception as exc:
        return {
            "applied": False,
            "target_nice": target,
            "error": f"{type(exc).__name__}:{exc}",
        }


def _gb(value: int | float) -> float:
    return round(float(value) / float(1024**3), 3)


def _disk_usage_snapshot(path: Path) -> dict[str, int | None]:
    probe = path.expanduser()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except Exception:
        return {"total_bytes": None, "used_bytes": None, "free_bytes": None, "device_id": None}
    try:
        device_id = int(probe.stat().st_dev)
    except Exception:
        device_id = None
    return {
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "device_id": device_id,
    }


def _disk_free_bytes(path: Path) -> int | None:
    return _disk_usage_snapshot(path).get("free_bytes")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_timestamp(raw: Any) -> datetime | None:
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


def _path_under(path: Path, root: Path) -> bool:
    try:
        resolved_path = path.expanduser().resolve(strict=False)
        resolved_root = root.expanduser().resolve(strict=False)
    except Exception:
        resolved_path = path.expanduser()
        resolved_root = root.expanduser()
    return bool(resolved_path == resolved_root or resolved_root in resolved_path.parents)


def _active_capacity_source(project_root: Path, external_root: Path) -> tuple[Path, str]:
    local_root = project_root / "local_fallback_storage"
    configured_floor_bytes = int(
        max(
            _safe_float(os.getenv("BOT_DEEP_COLD_SOURCE_FREE_FLOOR_GB"), DEFAULT_ADAPTIVE_FREE_FLOOR_GB),
            0.0,
        )
        * (1024**3)
    )
    pressure_rows: list[tuple[Path, dict[str, Any], int]] = []
    seen_devices: set[int] = set()
    for candidate in (project_root, external_root):
        usage = _disk_usage_snapshot(candidate)
        device_id = _safe_int(usage.get("device_id"), -1)
        if device_id >= 0 and device_id in seen_devices:
            continue
        if device_id >= 0:
            seen_devices.add(device_id)
        free_bytes = _safe_int(usage.get("free_bytes"), 0)
        pressure_rows.append((candidate, usage, max(configured_floor_bytes - free_bytes, 0)))
    breached = [row for row in pressure_rows if row[2] > 0]
    if breached:
        breached.sort(
            key=lambda item: (
                -item[2],
                _safe_int(item[1].get("free_bytes"), 0) / max(_safe_int(item[1].get("total_bytes"), 0), 1),
                str(item[0]),
            )
        )
        selected = breached[0][0]
        return selected, (
            "external_filesystem_hard_reserve_breach"
            if _path_under(selected, external_root)
            else "project_filesystem_hard_reserve_breach"
        )

    route_paths = (
        project_root / "data" / "jsonl_link.sqlite3",
        project_root / "data" / "bot_channel_queue.sqlite3",
        project_root / "data" / "snapshot_context.sqlite3",
    )
    resolved_routes: list[Path] = []
    for route_path in route_paths:
        try:
            resolved_routes.append(route_path.resolve(strict=False))
        except Exception:
            continue
    if any(_path_under(path, local_root) for path in resolved_routes):
        return project_root, "active_sqlite_local_filesystem"
    if any(_path_under(path, external_root) for path in resolved_routes):
        return external_root, "active_sqlite_external_filesystem"

    project_usage = _disk_usage_snapshot(project_root)
    external_usage = _disk_usage_snapshot(external_root)
    project_free = project_usage.get("free_bytes")
    external_free = external_usage.get("free_bytes")
    if project_free is not None and (external_free is None or int(project_free) <= int(external_free)):
        return project_root, "lowest_free_healthy_filesystem_fallback"
    return external_root, "external_filesystem_fallback"


def _fresh_growth_signal(project_root: Path, *, now: datetime) -> dict[str, Any]:
    path = project_root / "governance" / "health" / "storage_growth_forecast_latest.json"
    payload = _load_json(path)
    timestamp = _parse_timestamp(payload.get("timestamp_utc"))
    age_minutes = None if timestamp is None else max((now - timestamp).total_seconds() / 60.0, 0.0)
    max_age_minutes = max(_safe_float(os.getenv("BOT_DEEP_COLD_GROWTH_MAX_AGE_MINUTES"), 360.0), 1.0)
    fresh = bool(payload and age_minutes is not None and age_minutes <= max_age_minutes)
    floor = max(
        _safe_float(
            os.getenv("BOT_DEEP_COLD_GROWTH_FLOOR_GB_PER_DAY"),
            DEFAULT_ADAPTIVE_GROWTH_FLOOR_GB_PER_DAY,
        ),
        0.0,
    )
    sustained = _safe_float(payload.get("sustained_consumed_gb_per_day"), 0.0)
    observed = _safe_float(payload.get("consumed_gb_per_day"), 0.0)
    confidence = str(payload.get("confidence") or "")
    elapsed_days = _safe_float(payload.get("elapsed_days"), 0.0)
    min_slope_days = max(
        _safe_float(os.getenv("BOT_DEEP_COLD_MIN_GROWTH_SLOPE_DAYS"), 0.25),
        0.0,
    )
    short_window_spike = bool(
        fresh
        and elapsed_days > 0.0
        and elapsed_days < min_slope_days
        and max(sustained, observed) > max(floor * 4.0, 2.0)
    )
    if short_window_spike:
        effective = floor
        source = "fresh_short_window_growth_floor"
    elif fresh and (bool(payload.get("sustained_baseline")) or confidence == "sustained"):
        effective = max(sustained, floor)
        source = "fresh_sustained_forecast"
    elif fresh:
        max_unsustained = max(_safe_float(os.getenv("BOT_DEEP_COLD_UNSUSTAINED_GROWTH_CAP_GB_PER_DAY"), 4.0), floor)
        effective = min(max(observed, floor), max_unsustained)
        source = "fresh_bounded_unsustained_forecast"
    else:
        effective = floor
        source = "configured_growth_floor"
    return {
        "artifact_path": str(path),
        "artifact_present": bool(payload),
        "artifact_fresh": fresh,
        "artifact_age_minutes": None if age_minutes is None else round(age_minutes, 3),
        "max_age_minutes": round(max_age_minutes, 3),
        "confidence": confidence,
        "elapsed_days": round(elapsed_days, 4),
        "minimum_slope_days": round(min_slope_days, 4),
        "short_window_spike_clamped": short_window_spike,
        "sustained_gb_per_day": round(sustained, 4),
        "observed_gb_per_day": round(observed, 4),
        "effective_gb_per_day": round(effective, 4),
        "source": source,
    }


def _adaptive_release_policy(
    *,
    project_root: Path,
    source_path: Path,
    destination_path: Path,
    explicit_source_free_target_gb: float,
    explicit_release_target_gb: float,
    planning_horizon_days: float,
    max_move_gb: float,
    destination_reserve_gb: float,
    now: datetime,
) -> dict[str, Any]:
    source_usage = _disk_usage_snapshot(source_path)
    destination_usage = _disk_usage_snapshot(destination_path)
    source_total = _safe_int(source_usage.get("total_bytes"), 0)
    source_free = _safe_int(source_usage.get("free_bytes"), 0)
    destination_free = _safe_int(destination_usage.get("free_bytes"), 0)
    free_floor_gb = max(
        _safe_float(os.getenv("BOT_DEEP_COLD_SOURCE_FREE_FLOOR_GB"), DEFAULT_ADAPTIVE_FREE_FLOOR_GB),
        0.0,
    )
    free_ratio = min(
        max(_safe_float(os.getenv("BOT_DEEP_COLD_SOURCE_FREE_RATIO"), DEFAULT_ADAPTIVE_FREE_RATIO), 0.0),
        0.95,
    )
    operating_buffer_gb = max(
        _safe_float(os.getenv("BOT_DEEP_COLD_OPERATING_BUFFER_GB"), DEFAULT_ADAPTIVE_OPERATING_BUFFER_GB),
        0.0,
    )
    horizon_days = max(float(planning_horizon_days), 1.0)
    growth = _fresh_growth_signal(project_root, now=now)
    ratio_target_bytes = int(source_total * free_ratio)
    floor_target_bytes = int(free_floor_gb * (1024**3))
    growth_target_bytes = int(
        (operating_buffer_gb + (_safe_float(growth.get("effective_gb_per_day"), 0.0) * horizon_days))
        * (1024**3)
    )
    explicit_target_bytes = int(max(float(explicit_source_free_target_gb), 0.0) * (1024**3))
    hard_target_free_bytes = max(floor_target_bytes, growth_target_bytes, explicit_target_bytes)
    target_free_bytes = max(hard_target_free_bytes, ratio_target_bytes)
    if source_total > 0:
        hard_target_free_bytes = min(hard_target_free_bytes, source_total)
        target_free_bytes = min(target_free_bytes, source_total)
    hard_deficit_bytes = max(hard_target_free_bytes - source_free, 0)
    preferred_deficit_bytes = max(target_free_bytes - source_free, 0)
    explicit_release_bytes = int(max(float(explicit_release_target_gb), 0.0) * (1024**3))
    requested_release_bytes = max(
        preferred_deficit_bytes if hard_deficit_bytes > 0 else 0,
        explicit_release_bytes,
    )
    minimum_release_bytes = int(
        max(_safe_float(os.getenv("BOT_DEEP_COLD_MIN_ADAPTIVE_RELEASE_GB"), 1.0), 0.0) * (1024**3)
    )
    if explicit_release_bytes <= 0 and requested_release_bytes < minimum_release_bytes:
        requested_release_bytes = 0

    destination_reserve_bytes = int(max(float(destination_reserve_gb), 0.0) * (1024**3))
    destination_headroom_bytes = max(destination_free - destination_reserve_bytes, 0)
    configured_cap_bytes = int(max(float(max_move_gb), 0.0) * (1024**3))
    effective_cap_bytes = destination_headroom_bytes
    if configured_cap_bytes > 0:
        effective_cap_bytes = min(effective_cap_bytes, configured_cap_bytes)
    achievable_release_bytes = min(requested_release_bytes, effective_cap_bytes)
    same_filesystem = bool(
        source_usage.get("device_id") is not None
        and source_usage.get("device_id") == destination_usage.get("device_id")
    )
    if same_filesystem:
        achievable_release_bytes = 0
    if requested_release_bytes <= 0:
        status = (
            "source_hard_reserve_satisfied_soft_headroom_watch"
            if preferred_deficit_bytes > 0
            else "source_reserve_satisfied"
        )
    elif same_filesystem:
        status = "blocked_destination_same_filesystem"
    elif destination_free <= 0:
        status = "blocked_destination_capacity_unknown"
    elif achievable_release_bytes <= 0:
        status = "blocked_destination_reserve"
    elif achievable_release_bytes < requested_release_bytes:
        status = "bounded_by_destination_or_wave_cap"
    else:
        status = "release_required"
    return {
        "enabled": True,
        "status": status,
        "source_path": str(source_path),
        "source_device_id": source_usage.get("device_id"),
        "source_total_gb": _gb(source_total),
        "source_free_gb_before": _gb(source_free),
        "source_free_ratio_before": round((source_free / source_total), 5) if source_total > 0 else None,
        "hard_target_free_gb": _gb(hard_target_free_bytes),
        "target_free_gb": _gb(target_free_bytes),
        "target_components_gb": {
            "absolute_floor": round(free_floor_gb, 3),
            "filesystem_ratio": _gb(ratio_target_bytes),
            "growth_horizon": _gb(growth_target_bytes),
            "explicit": _gb(explicit_target_bytes),
        },
        "planning_horizon_days": round(horizon_days, 3),
        "growth_signal": growth,
        "hard_deficit_gb": _gb(hard_deficit_bytes),
        "preferred_headroom_deficit_gb": _gb(preferred_deficit_bytes),
        "explicit_release_target_gb": _gb(explicit_release_bytes),
        "requested_release_gb": _gb(requested_release_bytes),
        "achievable_release_this_wave_gb": _gb(achievable_release_bytes),
        "destination_path": str(destination_path),
        "destination_device_id": destination_usage.get("device_id"),
        "destination_free_gb_before": _gb(destination_free),
        "destination_reserve_gb": round(max(float(destination_reserve_gb), 0.0), 3),
        "destination_headroom_gb": _gb(destination_headroom_bytes),
        "configured_wave_cap_gb": round(max(float(max_move_gb), 0.0), 3),
        "effective_wave_cap_gb": _gb(effective_cap_bytes),
        "same_filesystem": same_filesystem,
        "policy": "live_capacity_growth_horizon_and_destination_reserve",
    }


def _file_age_days(path: Path, *, now: datetime) -> float:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return 0.0
    return max((now - mtime).total_seconds() / 86400.0, 0.0)


def _volume_name(path: Path) -> str:
    parts = path.expanduser().parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        return parts[2]
    return ""


def _is_protected_volume(path: Path) -> bool:
    volume = _volume_name(path)
    if volume == "VIDEO" and _approved_video_cold_archive(path):
        return False
    return volume in PROTECTED_VOLUME_NAMES


def _approved_video_cold_archive(path: Path) -> bool:
    if not _env_truthy("BOT_ALLOW_VIDEO_COLD_ARCHIVE"):
        return False
    allowed_root = Path(os.getenv("BOT_VIDEO_COLD_ARCHIVE_ROOT", DEFAULT_VIDEO_COLD_ARCHIVE_ROOT)).expanduser()
    try:
        raw = path.expanduser().resolve(strict=False)
        allowed = allowed_root.resolve(strict=False)
    except Exception:
        raw = path.expanduser()
        allowed = allowed_root
    return bool(raw == allowed or allowed in raw.parents)


def _relative_to_any(path: Path, roots: list[tuple[str, Path]]) -> str:
    for label, root in sorted(roots, key=lambda item: len(str(item[1])), reverse=True):
        try:
            return str(Path(label) / path.relative_to(root))
        except Exception:
            continue
    return str(path)


def _economic_value_from_stale_path(rel: str) -> str:
    lowered = rel.lower()
    if "/decisions/" in lowered or lowered.startswith("data/stale_stage/decisions/"):
        return "critical"
    if "/decision_explanations/" in lowered:
        return "high"
    if "/governance/" in lowered or "governance_telemetry_compactor" in lowered:
        return "medium"
    if "/exports/" in lowered or "/logs/" in lowered:
        return "low"
    return "medium"


def _retention_days_for_value(value: str) -> int:
    return {
        "low": 3,
        "medium": 14,
        "high": 30,
        "critical": 90,
    }.get(str(value or "medium"), 14)


def _deep_cold_state(*, rel: str, value: str, age_days: float, suffix: str) -> str:
    if value == "critical":
        return "nearline_retained_critical"
    if suffix == ".gz":
        return "manifest_indexed_compressed"
    if age_days >= _retention_days_for_value(value):
        return "retention_mature_review"
    return "manifest_indexed_retention_locked"


def _iter_candidate_files(stale_root: Path, *, min_size_bytes: int) -> list[Path]:
    if not stale_root.exists():
        return []
    rows: list[Path] = []
    for path in stale_root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            size = path.stat().st_size
        except Exception:
            continue
        if size < min_size_bytes:
            continue
        if path.name == DEFAULT_MANIFEST_NAME:
            continue
        rows.append(path)
    return sorted(rows, key=lambda item: (-_safe_int(item.stat().st_size if item.exists() else 0), str(item)))


def _second_cold_target_for_row(row: dict[str, Any], *, second_cold_root: Path) -> Path:
    rel = str(row.get("relative_path") or row.get("path") or "").strip().lstrip("/")
    clean_parts = [part for part in Path(rel).parts if part not in {"", ".", ".."}]
    return second_cold_root / "deep_cold" / "stale_stage" / Path(*clean_parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_with_sha256(source: Path, target: Path) -> tuple[str, int]:
    """Copy with a verified resumable prefix and return (source hash, resumed bytes)."""
    chunk_size = 8 * 1024 * 1024
    source_size = _safe_int(source.stat().st_size)
    resumed_bytes = 0
    digest = hashlib.sha256()

    if target.exists():
        partial_size = _safe_int(target.stat().st_size)
        if 0 < partial_size <= source_size:
            target_prefix_digest = hashlib.sha256()
            remaining = partial_size
            prefix_matches = True
            with source.open("rb") as source_handle, target.open("rb") as target_handle:
                while remaining > 0:
                    read_size = min(chunk_size, remaining)
                    source_chunk = source_handle.read(read_size)
                    target_chunk = target_handle.read(read_size)
                    if source_chunk != target_chunk or len(source_chunk) != read_size:
                        prefix_matches = False
                        break
                    digest.update(source_chunk)
                    target_prefix_digest.update(target_chunk)
                    remaining -= len(source_chunk)
            prefix_matches = bool(
                prefix_matches
                and remaining == 0
                and digest.digest() == target_prefix_digest.digest()
            )
            if prefix_matches:
                resumed_bytes = partial_size
            else:
                digest = hashlib.sha256()

    target_mode = "ab" if resumed_bytes else "wb"
    with source.open("rb") as source_handle, target.open(target_mode) as target_handle:
        if resumed_bytes:
            source_handle.seek(resumed_bytes)
        for chunk in iter(lambda: source_handle.read(chunk_size), b""):
            digest.update(chunk)
            target_handle.write(chunk)
        target_handle.flush()
        os.fsync(target_handle.fileno())
    try:
        shutil.copystat(source, target)
    except OSError:
        pass
    return digest.hexdigest(), resumed_bytes


def _copy_verify_then_symlink(source: Path, target: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": str(source),
        "target": str(target),
        "copied": False,
        "source_replaced_with_symlink": False,
        "verified_size_match": False,
        "verified_sha256_match": False,
        "source_sha256": "",
        "target_sha256": "",
        "resumed_bytes": 0,
        "skipped": False,
        "reason": "",
        "bytes": 0,
    }
    if source.is_symlink():
        result.update({"skipped": True, "reason": "source_already_symlink"})
        return result
    if not source.exists() or not source.is_file():
        result.update({"skipped": True, "reason": "source_missing_or_not_file"})
        return result

    source_stat_before = source.stat()
    source_size = _safe_int(source_stat_before.st_size)
    result["bytes"] = source_size
    target.parent.mkdir(parents=True, exist_ok=True)
    final_target = target
    if final_target.exists():
        target_size = _safe_int(final_target.stat().st_size)
        if target_size == source_size:
            source_hash = _sha256(source)
            target_hash = _sha256(final_target)
            result.update(
                {
                    "verified_size_match": True,
                    "verified_sha256_match": source_hash == target_hash,
                    "source_sha256": source_hash,
                    "target_sha256": target_hash,
                }
            )
            if source_hash != target_hash:
                result["verified_size_match"] = False
        else:
            stamp = iso_now().replace(":", "").replace("+", "_")
            final_target = target.with_name(f"{target.stem}.{stamp}{target.suffix}")

    if not result["verified_sha256_match"]:
        tmp = final_target.with_name(f".{final_target.name}.tmp")
        try:
            source_hash, resumed_bytes = _copy_with_sha256(source, tmp)
            source_stat_after = source.stat()
            copied_size = _safe_int(tmp.stat().st_size)
            target_hash = _sha256(tmp)
            source_stable = bool(
                source_stat_after.st_size == source_stat_before.st_size
                and source_stat_after.st_mtime_ns == source_stat_before.st_mtime_ns
            )
            if copied_size != source_size or source_hash != target_hash or not source_stable:
                try:
                    tmp.unlink()
                except Exception:
                    pass
                result.update(
                    {
                        "reason": "copy_verification_failed",
                        "verified_size_match": copied_size == source_size,
                        "verified_sha256_match": source_hash == target_hash,
                        "source_stable": source_stable,
                        "source_sha256": source_hash,
                        "target_sha256": target_hash,
                        "resumed_bytes": resumed_bytes,
                    }
                )
                return result
            os.replace(tmp, final_target)
            result.update(
                {
                    "copied": True,
                    "verified_size_match": True,
                    "verified_sha256_match": True,
                    "source_stable": True,
                    "source_sha256": source_hash,
                    "target_sha256": target_hash,
                    "resumed_bytes": resumed_bytes,
                    "target": str(final_target),
                }
            )
        except Exception as exc:
            result.update({"reason": f"copy_failed:{exc}"})
            return result

    if not bool(result.get("verified_sha256_match", False)):
        result["reason"] = "sha256_verification_required_before_source_release"
        return result

    try:
        source.unlink()
        source.symlink_to(final_target)
        result["source_replaced_with_symlink"] = True
    except Exception as exc:
        result["reason"] = f"symlink_replace_failed:{exc}"
    return result


def _apply_second_cold_moves(
    rows: list[dict[str, Any]],
    *,
    second_cold_root: Path,
    max_move_gb: float,
    max_move_files: int,
    include_critical: bool,
    release_target_gb: float = 0.0,
    adaptive: bool = False,
    source_device_id: int | None = None,
    maintenance_hold_active: bool = False,
) -> dict[str, Any]:
    if _is_protected_volume(second_cold_root):
        return {
            "enabled": True,
            "status": "blocked",
            "reason": "second_cold_root_protected_without_approved_subtree",
            "second_cold_root": str(second_cold_root),
            "moved_files": 0,
            "moved_gb": 0.0,
            "actions": [],
        }

    max_bytes = int(max(float(max_move_gb), 0.0) * (1024**3))
    release_target_bytes = int(max(float(release_target_gb), 0.0) * (1024**3))
    max_files = max(int(max_move_files), 0)
    actions: list[dict[str, Any]] = []
    moved_bytes = 0
    skipped_over_cap: list[dict[str, Any]] = []
    scope_candidates = [
        row
        for row in rows
        if (include_critical or str(row.get("economic_value") or "") != "critical")
        and str(row.get("path") or "").strip()
        and not bool(row.get("source_replaced_with_symlink", False))
        and (
            not adaptive
            or source_device_id is None
            or _safe_int(row.get("source_device_id"), -1) == int(source_device_id)
        )
    ]
    maintenance_blocked = [
        row
        for row in scope_candidates
        if bool(row.get("requires_maintenance_hold", False)) and not maintenance_hold_active
    ]
    candidates = [row for row in scope_candidates if row not in maintenance_blocked]
    candidates.sort(
        key=lambda row: (
            VALUE_RANK.get(str(row.get("economic_value") or "medium"), 1),
            bool(row.get("retention_locked", False)),
            -_safe_float(row.get("age_days"), 0.0),
            -_safe_int(row.get("size_bytes"), 0),
            str(row.get("path") or ""),
        )
    )
    selected_candidates, skipped_over_cap = _select_move_candidates(
        candidates,
        release_target_bytes=release_target_bytes,
        max_bytes=max_bytes,
        max_files=max_files,
    )
    for row in selected_candidates:
        if max_files and len(actions) >= max_files:
            break
        size = _safe_int(row.get("size_bytes"), 0)
        source = Path(str(row.get("path") or ""))
        target = _second_cold_target_for_row(row, second_cold_root=second_cold_root)
        action = _copy_verify_then_symlink(source, target)
        actions.append(action)
        row["second_cold_target"] = str(action.get("target") or target)
        row["second_cold_move"] = action
        row["source_replaced_with_symlink"] = bool(action.get("source_replaced_with_symlink", False))
        if bool(action.get("source_replaced_with_symlink", False)):
            moved_bytes += _safe_int(action.get("bytes"), size)

    failed = [row for row in actions if not bool(row.get("source_replaced_with_symlink", False)) and not bool(row.get("skipped", False))]
    release_target_met = bool(release_target_bytes <= 0 or moved_bytes >= release_target_bytes)
    if failed:
        status = "partial"
        reason = "one_or_more_moves_failed"
    elif not release_target_met and maintenance_blocked:
        status = "blocked"
        reason = "adaptive_release_waiting_for_maintenance_hold"
    elif not release_target_met:
        status = "partial"
        reason = "adaptive_release_target_unmet"
    else:
        status = "ready"
        reason = ""
    return {
        "enabled": True,
        "status": status,
        "reason": reason,
        "second_cold_root": str(second_cold_root),
        "include_critical": bool(include_critical),
        "max_move_gb": round(float(max_move_gb), 3),
        "max_move_files": int(max_files),
        "adaptive": bool(adaptive),
        "source_device_id": source_device_id,
        "scope_candidate_files": len(scope_candidates),
        "candidate_files": len(candidates),
        "maintenance_blocked_files": len(maintenance_blocked),
        "maintenance_blocked_gb": _gb(
            sum(_safe_int(row.get("size_bytes"), 0) for row in maintenance_blocked)
        ),
        "selected_candidate_files": len(selected_candidates),
        "attempted_files": len(actions),
        "moved_files": sum(1 for row in actions if bool(row.get("source_replaced_with_symlink", False))),
        "moved_gb": _gb(moved_bytes),
        "failed_files": len(failed),
        "release_target_gb": _gb(release_target_bytes),
        "release_target_met": release_target_met,
        "release_target_unmet_gb": _gb(max(release_target_bytes - moved_bytes, 0)),
        "skipped_over_cap_files": len(skipped_over_cap),
        "skipped_over_cap_gb": _gb(sum(_safe_int(row.get("size_bytes"), 0) for row in skipped_over_cap)),
        "selected_value_counts": {
            value: sum(1 for row in selected_candidates if str(row.get("economic_value") or "") == value)
            for value in ("low", "medium", "high", "critical")
        },
        "actions": actions[:50],
    }


def _select_move_candidates(
    candidates: list[dict[str, Any]],
    *,
    release_target_bytes: int,
    max_bytes: int,
    max_files: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    limit_files = max(int(max_files), 1)
    usable = [row for row in candidates if _safe_int(row.get("size_bytes"), 0) > 0]
    skipped = [
        {"path": str(row.get("path") or ""), "size_bytes": _safe_int(row.get("size_bytes"), 0)}
        for row in usable
        if max_bytes and _safe_int(row.get("size_bytes"), 0) > max_bytes
    ]
    within_cap = [row for row in usable if not max_bytes or _safe_int(row.get("size_bytes"), 0) <= max_bytes]
    if release_target_bytes <= 0:
        selected: list[dict[str, Any]] = []
        selected_bytes = 0
        for row in within_cap:
            size = _safe_int(row.get("size_bytes"), 0)
            if len(selected) >= limit_files:
                break
            if max_bytes and selected_bytes + size > max_bytes:
                skipped.append({"path": str(row.get("path") or ""), "size_bytes": size})
                continue
            selected.append(row)
            selected_bytes += size
        return selected, skipped

    def valid(plan: list[dict[str, Any]]) -> bool:
        total = sum(_safe_int(row.get("size_bytes"), 0) for row in plan)
        return bool(plan and len(plan) <= limit_files and (not max_bytes or total <= max_bytes))

    plans: list[list[dict[str, Any]]] = []
    for row in within_cap:
        size = _safe_int(row.get("size_bytes"), 0)
        if size >= release_target_bytes:
            plans.append([row])

    for ordered in (
        sorted(within_cap, key=lambda row: _safe_int(row.get("size_bytes"), 0), reverse=True),
        sorted(within_cap, key=lambda row: _safe_int(row.get("size_bytes"), 0)),
    ):
        plan: list[dict[str, Any]] = []
        total = 0
        for row in ordered:
            size = _safe_int(row.get("size_bytes"), 0)
            if len(plan) >= limit_files or (max_bytes and total + size > max_bytes):
                continue
            plan.append(row)
            total += size
            if total >= release_target_bytes:
                break
        if valid(plan):
            plans.append(plan)

    pair_pool = sorted(within_cap, key=lambda row: _safe_int(row.get("size_bytes"), 0))[:128]
    if limit_files >= 2:
        for index, left in enumerate(pair_pool):
            left_size = _safe_int(left.get("size_bytes"), 0)
            for right in pair_pool[index + 1 :]:
                total = left_size + _safe_int(right.get("size_bytes"), 0)
                if total >= release_target_bytes and (not max_bytes or total <= max_bytes):
                    plans.append([left, right])
                    break

    meeting = [
        plan
        for plan in plans
        if sum(_safe_int(row.get("size_bytes"), 0) for row in plan) >= release_target_bytes
    ]
    if meeting:
        selected = min(
            meeting,
            key=lambda plan: (
                sum(_safe_int(row.get("size_bytes"), 0) for row in plan),
                max(VALUE_RANK.get(str(row.get("economic_value") or "medium"), 1) for row in plan),
                sum(VALUE_RANK.get(str(row.get("economic_value") or "medium"), 1) for row in plan),
                len(plan),
                tuple(str(row.get("path") or "") for row in plan),
            ),
        )
    elif plans:
        selected = max(
            plans,
            key=lambda plan: sum(_safe_int(row.get("size_bytes"), 0) for row in plan),
        )
    else:
        selected = []
    selected_ids = {id(row) for row in selected}
    for row in within_cap:
        if id(row) not in selected_ids:
            skipped.append({"path": str(row.get("path") or ""), "size_bytes": _safe_int(row.get("size_bytes"), 0)})
    return selected, skipped


def _candidate_row(
    path: Path,
    *,
    rel_roots: list[tuple[str, Path]],
    now: datetime,
    artifact_class: str,
    requires_maintenance_hold: bool,
    economic_value: str = "",
) -> dict[str, Any]:
    size = _safe_int(path.stat().st_size if path.exists() else 0)
    rel = _relative_to_any(path, rel_roots)
    value = str(economic_value or _economic_value_from_stale_path(rel))
    age_days = _file_age_days(path, now=now)
    retention_days = _retention_days_for_value(value)
    suffix = path.suffix.lower()
    state = _deep_cold_state(rel=rel, value=value, age_days=age_days, suffix=suffix)
    try:
        source_device_id = int(path.stat().st_dev)
    except Exception:
        source_device_id = None
    return {
        "relative_path": rel,
        "path": str(path),
        "size_bytes": size,
        "size_gb": _gb(size),
        "age_days": round(age_days, 3),
        "economic_value": value,
        "retention_days": retention_days,
        "deep_cold_state": state,
        "retention_locked": bool(age_days < retention_days),
        "compressed": suffix == ".gz",
        "artifact_class": artifact_class,
        "requires_maintenance_hold": bool(requires_maintenance_hold),
        "source_device_id": source_device_id,
        "eligible_for_delete": False,
        "reason": "deep_cold_manifest_index_only_no_delete",
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    min_size_mb: float = 25.0,
    top_n: int = 25,
    manifest_path: Path | None = None,
    move_to_second_cold: bool = False,
    second_cold_root: Path | None = None,
    max_move_gb: float = 64.0,
    max_move_files: int = 250,
    include_critical: bool = False,
    include_local_quarantine: bool = False,
    include_failover_backups: bool = False,
    adaptive: bool = False,
    adaptive_release_target_gb: float = 0.0,
    source_free_target_gb: float = 0.0,
    source_free_path: Path | None = None,
    planning_horizon_days: float = DEFAULT_ADAPTIVE_HORIZON_DAYS,
    destination_reserve_gb: float = DEFAULT_DESTINATION_RESERVE_GB,
) -> dict[str, Any]:
    external = resolve_external_storage()
    external_root = external.external_root
    if _is_protected_volume(external_root):
        payload = {
            "timestamp_utc": iso_now(),
            "schema_version": 1,
            "ok": False,
            "overall_status": "blocked",
            "apply": bool(apply),
            "blocked_reason": "protected_volume_refused",
            "protected_volume": _volume_name(external_root),
            "never_touch_protected_volumes": sorted(PROTECTED_VOLUME_NAMES),
        }
        return payload

    now = datetime.now(timezone.utc)
    maintenance_hold = maintenance_hold_snapshot(project_root)
    if (
        apply
        and move_to_second_cold
        and include_local_quarantine
        and not adaptive
        and not bool(maintenance_hold.get("active", False))
    ):
        return {
            "timestamp_utc": iso_now(),
            "schema_version": 1,
            "ok": False,
            "overall_status": "blocked",
            "apply": True,
            "blocked_reason": "runtime_maintenance_hold_required_for_local_quarantine_move",
            "runtime_maintenance_hold": maintenance_hold,
        }

    target_root = (
        second_cold_root
        or Path(
            os.getenv("BOT_SECOND_COLD_ROOT", "")
            or os.getenv("BOT_VIDEO_COLD_ARCHIVE_ROOT", DEFAULT_VIDEO_COLD_ARCHIVE_ROOT)
        )
    ).expanduser()
    source_reason = "explicit_source_path"
    if source_free_path is not None:
        adaptive_source_path = Path(source_free_path).expanduser()
    elif adaptive:
        adaptive_source_path, source_reason = _active_capacity_source(project_root, external_root)
    else:
        adaptive_source_path = project_root / "local_fallback_storage"

    if adaptive:
        adaptive_release = _adaptive_release_policy(
            project_root=project_root,
            source_path=adaptive_source_path,
            destination_path=target_root,
            explicit_source_free_target_gb=source_free_target_gb,
            explicit_release_target_gb=adaptive_release_target_gb,
            planning_horizon_days=planning_horizon_days,
            max_move_gb=max_move_gb,
            destination_reserve_gb=destination_reserve_gb,
            now=now,
        )
        adaptive_release["source_selection_reason"] = source_reason
        adaptive_release_bytes = int(_safe_float(adaptive_release.get("requested_release_gb"), 0.0) * (1024**3))
        effective_max_move_gb = _safe_float(adaptive_release.get("effective_wave_cap_gb"), 0.0)
    else:
        source_free_before = _disk_free_bytes(adaptive_source_path)
        source_free_target_bytes = int(max(float(source_free_target_gb), 0.0) * (1024**3))
        floor_deficit_bytes = (
            max(source_free_target_bytes - source_free_before, 0)
            if source_free_before is not None
            else 0
        )
        explicit_release_bytes = int(max(float(adaptive_release_target_gb), 0.0) * (1024**3))
        adaptive_release_bytes = max(floor_deficit_bytes, explicit_release_bytes)
        effective_max_move_gb = float(max_move_gb)
        adaptive_release = {
            "enabled": False,
            "status": "manual_bounded_policy",
            "source_path": str(adaptive_source_path),
            "source_selection_reason": source_reason,
            "source_free_bytes_before": source_free_before,
            "source_free_gb_before": _gb(source_free_before or 0),
            "source_free_target_gb": _gb(source_free_target_bytes),
            "explicit_release_target_gb": _gb(explicit_release_bytes),
            "requested_release_gb": _gb(adaptive_release_bytes),
            "mode": "demand_driven" if adaptive_release_bytes > 0 else "bounded_batch",
        }

    candidate_roots: list[tuple[str, Path, bool]] = [
        ("external_stale_stage", external_root / "data" / "stale_stage", False),
        ("project_stale_stage", project_root / "data" / "stale_stage", False),
    ]
    if include_local_quarantine:
        candidate_roots.append(
            ("local_quarantine", project_root / "local_fallback_storage" / "quarantine", True)
        )
    seen_roots: dict[str, tuple[str, Path, bool]] = {}
    for artifact_class, root, requires_hold in candidate_roots:
        try:
            seen_roots[str(root.resolve())] = (artifact_class, root, requires_hold)
        except Exception:
            seen_roots[str(root)] = (artifact_class, root, requires_hold)
    roots = list(seen_roots.values())
    rel_roots = [("external", external_root), ("project", project_root)]
    min_size_bytes = max(int(float(min_size_mb) * 1024 * 1024), 1)
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for artifact_class, candidate_root, requires_hold in roots:
        for path in _iter_candidate_files(candidate_root, min_size_bytes=min_size_bytes):
            try:
                real = str(path.absolute())
            except Exception:
                real = str(path)
            if real in seen_paths:
                continue
            seen_paths.add(real)
            rows.append(
                _candidate_row(
                    path,
                    rel_roots=rel_roots,
                    now=now,
                    artifact_class=artifact_class,
                    requires_maintenance_hold=requires_hold,
                )
            )
    if include_failover_backups:
        failover_root = project_root / "local_fallback_storage" / "data"
        for path in sorted(failover_root.glob("*.pre_local_failover_*.bak")):
            if not path.is_file() or path.is_symlink() or _safe_int(path.stat().st_size, 0) < min_size_bytes:
                continue
            raw_path = str(path.absolute())
            if raw_path in seen_paths:
                continue
            seen_paths.add(raw_path)
            rows.append(
                _candidate_row(
                    path,
                    rel_roots=rel_roots,
                    now=now,
                    artifact_class="superseded_verified_failover_backup",
                    requires_maintenance_hold=False,
                    economic_value="high",
                )
            )

    adaptive_source_device_id = _safe_int(adaptive_release.get("source_device_id"), -1) if adaptive else None
    for row in rows:
        row["matches_adaptive_source_filesystem"] = bool(
            not adaptive
            or adaptive_source_device_id is None
            or adaptive_source_device_id < 0
            or _safe_int(row.get("source_device_id"), -2) == adaptive_source_device_id
        )

    rows.sort(key=lambda row: (-_safe_int(row.get("size_bytes"), 0), str(row.get("relative_path") or "")))
    managed_rows = [
        row
        for row in rows
        if str(row.get("deep_cold_state") or "") in {"manifest_indexed_compressed", "manifest_indexed_retention_locked"}
    ]
    critical_rows = [row for row in rows if str(row.get("economic_value") or "") == "critical"]
    retention_locked_rows = [row for row in rows if bool(row.get("retention_locked", False))]
    managed_bytes = sum(_safe_int(row.get("size_bytes"), 0) for row in managed_rows)
    total_bytes = sum(_safe_int(row.get("size_bytes"), 0) for row in rows)
    retention_locked_bytes = sum(_safe_int(row.get("size_bytes"), 0) for row in retention_locked_rows)
    deep_root = manifest_path.parent if manifest_path is not None else external_root / "data" / "deep_cold"
    final_manifest_path = manifest_path or deep_root / DEFAULT_MANIFEST_NAME
    second_cold_move: dict[str, Any] = {
        "enabled": bool(move_to_second_cold),
        "status": "planned" if move_to_second_cold else "disabled",
        "reason": "apply_required" if move_to_second_cold else "",
        "second_cold_root": str(target_root if move_to_second_cold else second_cold_root or ""),
        "release_target_gb": _gb(adaptive_release_bytes),
        "moved_files": 0,
        "moved_gb": 0.0,
        "actions": [],
    }
    if apply and move_to_second_cold:
        if adaptive and adaptive_release_bytes <= 0:
            second_cold_move.update(
                {
                    "status": "ready",
                    "reason": "source_reserve_satisfied_no_move",
                    "release_target_met": True,
                    "release_target_unmet_gb": 0.0,
                }
            )
        elif adaptive and _safe_float(adaptive_release.get("achievable_release_this_wave_gb"), 0.0) <= 0:
            second_cold_move.update(
                {
                    "status": "blocked",
                    "reason": str(adaptive_release.get("status") or "adaptive_destination_blocked"),
                    "release_target_met": False,
                    "release_target_unmet_gb": _gb(adaptive_release_bytes),
                }
            )
        else:
            second_cold_move = _apply_second_cold_moves(
                rows,
                second_cold_root=target_root,
                max_move_gb=effective_max_move_gb,
                max_move_files=max_move_files,
                include_critical=include_critical,
                release_target_gb=float(adaptive_release_bytes) / float(1024**3),
                adaptive=adaptive,
                source_device_id=adaptive_source_device_id,
                maintenance_hold_active=bool(maintenance_hold.get("active", False)),
            )

    source_free_after = _disk_free_bytes(adaptive_source_path)
    adaptive_release["source_free_gb_after"] = _gb(source_free_after or 0)
    if adaptive:
        target_free_bytes = int(_safe_float(adaptive_release.get("target_free_gb"), 0.0) * (1024**3))
        hard_target_free_bytes = int(
            _safe_float(adaptive_release.get("hard_target_free_gb"), 0.0) * (1024**3)
        )
        hard_reserve_met_after = bool(
            source_free_after is not None and _safe_int(source_free_after, 0) >= hard_target_free_bytes
        )
        adaptive_release["hard_reserve_met_after"] = hard_reserve_met_after
        adaptive_release["remaining_hard_deficit_gb_after"] = _gb(
            max(hard_target_free_bytes - _safe_int(source_free_after, 0), 0)
        )
        adaptive_release["remaining_deficit_gb_after"] = _gb(
            max(target_free_bytes - _safe_int(source_free_after, 0), 0)
        )
        if (
            str(second_cold_move.get("status") or "") == "partial"
            and hard_reserve_met_after
            and _safe_int(second_cold_move.get("failed_files"), 0) == 0
        ):
            second_cold_move.update(
                {
                    "status": "ready",
                    "reason": "hard_reserve_restored_preferred_headroom_advisory",
                    "hard_reserve_target_met": True,
                    "preferred_headroom_target_met": False,
                }
            )

    write_result: dict[str, Any] = {
        "applied": False,
        "manifest_path": str(final_manifest_path),
        "manifest_rows": 0,
        "error": "",
    }
    if apply:
        try:
            final_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with final_manifest_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            write_result.update({"applied": True, "manifest_rows": len(rows)})
        except Exception as exc:
            write_result.update({"applied": False, "error": str(exc)})

    manifest_ready = bool((rows or adaptive) and (not apply or write_result.get("applied", False)))
    move_status = str(second_cold_move.get("status") or "")
    move_ready = bool(
        not (apply and move_to_second_cold)
        or move_status == "ready"
    )
    ready = bool(manifest_ready and move_ready)
    if ready:
        overall_status = (
            "planned"
            if not apply and move_to_second_cold and adaptive_release_bytes > 0
            else "ready"
        )
    elif move_status == "blocked":
        overall_status = "blocked"
    elif move_status == "partial":
        overall_status = "needs_attention"
    else:
        overall_status = "needs_data"
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": bool(ready),
        "overall_status": overall_status,
        "apply": bool(apply),
        "adaptive": bool(adaptive),
        "include_local_quarantine": bool(include_local_quarantine),
        "include_failover_backups": bool(include_failover_backups),
        "runtime_maintenance_hold": maintenance_hold,
        "adaptive_release": adaptive_release,
        "external_root": str(external_root),
        "stale_roots": [str(root) for _artifact_class, root, _requires_hold in roots],
        "candidate_sources": [
            {
                "artifact_class": artifact_class,
                "path": str(root),
                "requires_maintenance_hold": requires_hold,
            }
            for artifact_class, root, requires_hold in roots
        ]
        + (
            [
                {
                    "artifact_class": "superseded_verified_failover_backup",
                    "path": str(project_root / "local_fallback_storage" / "data"),
                    "requires_maintenance_hold": False,
                    "glob": "*.pre_local_failover_*.bak",
                }
            ]
            if include_failover_backups
            else []
        ),
        "deep_cold_root": str(final_manifest_path.parent),
        "manifest_path": str(final_manifest_path),
        "min_size_mb": float(min_size_mb),
        "summary": {
            "candidate_count": len(rows),
            "candidate_gb": _gb(total_bytes),
            "managed_count": len(managed_rows),
            "managed_gb": _gb(managed_bytes),
            "retention_locked_count": len(retention_locked_rows),
            "retention_locked_gb": _gb(retention_locked_bytes),
            "critical_nearline_count": len(critical_rows),
            "critical_nearline_gb": _gb(sum(_safe_int(row.get("size_bytes"), 0) for row in critical_rows)),
            "adaptive_source_candidate_count": sum(
                1 for row in rows if bool(row.get("matches_adaptive_source_filesystem", False))
            ),
            "adaptive_source_candidate_gb": _gb(
                sum(
                    _safe_int(row.get("size_bytes"), 0)
                    for row in rows
                    if bool(row.get("matches_adaptive_source_filesystem", False))
                )
            ),
            "failover_backup_count": sum(
                1 for row in rows if str(row.get("artifact_class") or "") == "superseded_verified_failover_backup"
            ),
            "failover_backup_gb": _gb(
                sum(
                    _safe_int(row.get("size_bytes"), 0)
                    for row in rows
                    if str(row.get("artifact_class") or "") == "superseded_verified_failover_backup"
                )
            ),
        },
        "policy": {
            "mode": "manifest_indexed_deep_cold_no_delete",
            "purpose": "keep evidence discoverable while moving protected stale-stage archives out of active hot-path scoring",
            "delete_policy": "never delete from this layer; stale-reaper keeps owning retention deletion",
            "never_touch_protected_volumes": sorted(PROTECTED_VOLUME_NAMES),
            "protected_volume_checked": _volume_name(external_root),
            "second_cold_move_policy": (
                "copy-verify-retain-via-original-path-symlink"
                if move_to_second_cold
                else "manifest-only"
            ),
            "adaptive_policy": "move only the live deficit on the constrained source filesystem",
            "destination_policy": "preserve configured cold-volume reserve and refuse same-filesystem offload",
            "critical_evidence_policy": "excluded unless explicitly enabled",
            "maintenance_policy": "local quarantine waits for an active maintenance hold; immutable failover backups may move after source-stability proof",
        },
        "second_cold_move": second_cold_move,
        "write_result": write_result,
        "top_rows": rows[: max(int(top_n), 1)],
        "control_env": {
            "BOT_DEEP_COLD_LAYER_ACTIVE": "1" if ready else "0",
            "BOT_DEEP_COLD_ROOT": str(final_manifest_path.parent),
            "BOT_DEEP_COLD_MANIFEST_PATH": str(final_manifest_path),
            "BOT_DEEP_COLD_MANAGED_GB": str(_gb(managed_bytes)),
            "BOT_DEEP_COLD_DELETE_POLICY": "never_delete_manifest_index_only",
            "BOT_DEEP_COLD_SECOND_COLD_MOVE_GB": str(second_cold_move.get("moved_gb", 0.0)),
        },
        "next_action": (
            "source reserve is satisfied; no cold movement is required"
            if adaptive and adaptive_release_bytes <= 0
            else "hard source reserve is restored; keep the remaining preferred headroom as an advisory"
            if str(second_cold_move.get("reason") or "") == "hard_reserve_restored_preferred_headroom_advisory"
            else "apply the bounded adaptive cold-archive wave; selection is limited to the constrained filesystem"
            if not apply and move_to_second_cold and adaptive_release_bytes > 0
            else "engage a runtime maintenance hold to release the remaining local-quarantine deficit"
            if str(second_cold_move.get("reason") or "") == "adaptive_release_waiting_for_maintenance_hold"
            else "resolve cold destination capacity before retrying adaptive offload"
            if str(second_cold_move.get("status") or "") == "blocked"
            else "deep cold manifest is current; approved second-cold moves preserve original paths with symlinks"
            if move_to_second_cold and _safe_int(second_cold_move.get("moved_files"), 0) > 0
            else "deep cold manifest is current; storage control can treat retention-locked stale-stage archives as managed cold evidence"
            if ready
            else "run with --apply after stale-stage archives exist"
        ),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the deep-cold manifest layer for protected BOT_LOGS archives.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--min-size-mb", type=float, default=25.0)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--move-to-second-cold", action="store_true")
    parser.add_argument("--second-cold-root", default="")
    parser.add_argument("--max-move-gb", type=float, default=float(os.getenv("BOT_DEEP_COLD_MAX_MOVE_GB", "64.0")))
    parser.add_argument("--max-move-files", type=int, default=int(os.getenv("BOT_DEEP_COLD_MAX_MOVE_FILES", "250")))
    parser.add_argument("--include-critical", action="store_true", default=_env_truthy("BOT_DEEP_COLD_INCLUDE_CRITICAL"))
    parser.add_argument("--adaptive", action="store_true", default=_env_truthy("BOT_DEEP_COLD_ADAPTIVE"))
    parser.add_argument("--include-local-quarantine", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-failover-backups", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--adaptive-release-target-gb", type=float, default=0.0)
    parser.add_argument("--source-free-target-gb", type=float, default=0.0)
    parser.add_argument("--source-free-path", default="")
    parser.add_argument(
        "--planning-horizon-days",
        type=float,
        default=float(os.getenv("BOT_DEEP_COLD_PLANNING_HORIZON_DAYS", str(DEFAULT_ADAPTIVE_HORIZON_DAYS))),
    )
    parser.add_argument(
        "--destination-reserve-gb",
        type=float,
        default=float(os.getenv("BOT_COLD_ARCHIVE_RESERVE_GB", str(DEFAULT_DESTINATION_RESERVE_GB))),
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path).expanduser() if str(args.manifest_path or "").strip() else None
    second_cold_root = Path(args.second_cold_root).expanduser() if str(args.second_cold_root or "").strip() else None
    include_local_quarantine = bool(args.adaptive) if args.include_local_quarantine is None else bool(args.include_local_quarantine)
    include_failover_backups = bool(args.adaptive) if args.include_failover_backups is None else bool(args.include_failover_backups)
    runtime_priority = (
        _lower_runtime_priority()
        if bool(args.apply) and bool(args.move_to_second_cold)
        else {"applied": False, "reason": "no_copy_wave"}
    )
    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        min_size_mb=float(args.min_size_mb),
        top_n=int(args.top_n),
        manifest_path=manifest_path,
        move_to_second_cold=bool(args.move_to_second_cold),
        second_cold_root=second_cold_root,
        max_move_gb=float(args.max_move_gb),
        max_move_files=int(args.max_move_files),
        include_critical=bool(args.include_critical),
        include_local_quarantine=include_local_quarantine,
        include_failover_backups=include_failover_backups,
        adaptive=bool(args.adaptive),
        adaptive_release_target_gb=float(args.adaptive_release_target_gb),
        source_free_target_gb=float(args.source_free_target_gb),
        source_free_path=Path(args.source_free_path).expanduser() if str(args.source_free_path or "").strip() else None,
        planning_horizon_days=float(args.planning_horizon_days),
        destination_reserve_gb=float(args.destination_reserve_gb),
    )
    payload["runtime_priority"] = runtime_priority
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        print(
            "deep_cold_storage_layer "
            f"overall_status={payload.get('overall_status', '')} "
            f"managed_gb={summary.get('managed_gb', 0)} "
            f"manifest={payload.get('manifest_path', '')}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
