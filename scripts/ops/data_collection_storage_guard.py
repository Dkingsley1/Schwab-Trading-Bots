#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "data_collection_storage_guard_latest.json"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_EXTERNAL_ROOT = Path("/Volumes/BOT_LOGS/schwab_trading_bot")
PROTECTED_VOLUME_PREFIXES = ("/Volumes/VIDEO",)
SAFE_STALE_SUFFIXES = (".tmp", ".temp", ".part", ".partial", ".incomplete", ".download", ".swap", ".swp")
SAFE_METADATA_NAMES = {".DS_Store"}
DEFAULT_SPACE_RECOVERY_MAX_DELETE_GB = 8.0
DEFAULT_SPACE_RECOVERY_TARGET_FREE_GB = 64.0
DEFAULT_SPACE_RECOVERY_MIN_AGE_HOURS = 6.0
DEFAULT_SPACE_RECOVERY_CANDIDATE_LIMIT = 20000
DEFAULT_SPACE_RECOVERY_SCAN_FILE_LIMIT = 200000
DEFAULT_SPACE_RECOVERY_JUMBO_DUPLICATE_GB = 12.0


def _disk_usage(path: Path) -> dict[str, Any]:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = os.statvfs(str(probe))
    except Exception:
        return {
            "path": str(path),
            "probe_path": str(probe),
            "available_bytes": 0,
            "total_bytes": 0,
            "used_bytes": 0,
            "used_ratio": 1.0,
            "ok": False,
        }
    total = int(usage.f_frsize * usage.f_blocks)
    available = int(usage.f_frsize * usage.f_bavail)
    used = max(total - available, 0)
    return {
        "path": str(path),
        "probe_path": str(probe),
        "available_bytes": available,
        "total_bytes": total,
        "used_bytes": used,
        "used_ratio": (float(used) / float(total)) if total > 0 else 1.0,
        "ok": True,
    }


def _gb(raw: int | float) -> float:
    return float(raw) / float(1024**3)


def _is_protected_volume(path: Path) -> bool:
    text = str(path.expanduser())
    return any(text == prefix or text.startswith(prefix + "/") for prefix in PROTECTED_VOLUME_PREFIXES)


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except Exception:
        return False


def _mode_for_space(*, available_gb: float, used_ratio: float, warn_gb: float, throttle_gb: float, critical_gb: float) -> str:
    if available_gb <= critical_gb or used_ratio >= 0.98:
        return "critical"
    if available_gb <= throttle_gb or used_ratio >= 0.94:
        return "throttle"
    if available_gb <= warn_gb or used_ratio >= 0.90:
        return "watch"
    return "normal"


def _collector_kind(row: dict[str, Any]) -> str:
    kind = str(row.get("slot_kind") or "").strip().lower()
    role = str(row.get("bot_role") or "").strip().lower()
    label_contract = str(row.get("data_label_contract_version") or "").strip().lower()
    collections = ",".join(str(item or "").strip().lower() for item in list(row.get("data_intake_collections") or []))
    if (
        "quant" in kind
        or label_contract.startswith("quant_")
        or any(token in collections for token in ("mlx_library", "mlx_graph", "mlx_snn", "mlx_vision", "esig_rough_path"))
    ):
        return "quant_research"
    if "aggressive_intraday" in kind:
        return "aggressive_intraday"
    if role == "options_sub_bot" or "options" in kind:
        return "options"
    if role == "infrastructure_sub_bot":
        return "infrastructure"
    return "standard"


def _guard_profile(mode: str, kind: str) -> dict[str, Any]:
    if mode == "normal":
        if kind == "quant_research":
            return {
                "capture_mode": "sampled",
                "max_daily_storage_mb": 80,
                "freshness_floor_seconds": 900,
                "retention_profile": "hot_quant_sampled_2d_warm_30d",
                "sample_rate": 0.35,
            }
        return {
            "capture_mode": "full",
            "max_daily_storage_mb": 250 if kind == "aggressive_intraday" else 150,
            "freshness_floor_seconds": 60 if kind == "aggressive_intraday" else (180 if kind == "options" else 300),
            "retention_profile": "",
            "sample_rate": 1.0,
        }
    if mode == "watch":
        if kind == "quant_research":
            return {
                "capture_mode": "thin_sample",
                "max_daily_storage_mb": 45,
                "freshness_floor_seconds": 1200,
                "retention_profile": "hot_quant_thin_1d_warm_21d",
                "sample_rate": 0.18,
            }
        return {
            "capture_mode": "sampled",
            "max_daily_storage_mb": 100 if kind == "aggressive_intraday" else 80,
            "freshness_floor_seconds": 180 if kind == "aggressive_intraday" else (300 if kind == "options" else 600),
            "retention_profile": "hot_sampled_3d_warm_45d",
            "sample_rate": 0.5,
        }
    if mode == "throttle":
        if kind == "quant_research":
            return {
                "capture_mode": "metadata_only",
                "max_daily_storage_mb": 20,
                "freshness_floor_seconds": 2400,
                "retention_profile": "hot_quant_metadata_12h_warm_14d",
                "sample_rate": 0.08,
            }
        return {
            "capture_mode": "thin_sample",
            "max_daily_storage_mb": 50 if kind == "aggressive_intraday" else 40,
            "freshness_floor_seconds": 600 if kind == "aggressive_intraday" else (900 if kind == "options" else 1200),
            "retention_profile": "hot_thin_1d_warm_30d",
            "sample_rate": 0.2,
        }
    return {
        "capture_mode": "metadata_only",
        "max_daily_storage_mb": 10 if kind == "quant_research" else (15 if kind == "aggressive_intraday" else 20),
        "freshness_floor_seconds": 1800 if kind == "aggressive_intraday" else (1800 if kind == "options" else 3600),
        "retention_profile": "hot_quant_metadata_6h_warm_7d" if kind == "quant_research" else "hot_metadata_12h_warm_14d",
        "sample_rate": 0.03 if kind == "quant_research" else 0.05,
    }


def _compute_guard_floor(row: dict[str, Any]) -> dict[str, Any] | None:
    mode = str(row.get("data_collection_compute_guard_mode") or "").strip().lower()
    if mode == "protect_live":
        return {
            "capture_mode": "thin_sample",
            "sample_rate": 0.15,
            "max_daily_storage_mb": 35,
            "freshness_floor_seconds": 1800,
            "reason": "compute_guard=protect_live",
        }
    if mode == "sustain":
        return {
            "capture_mode": "sampled",
            "sample_rate": 0.3,
            "max_daily_storage_mb": 60,
            "freshness_floor_seconds": 900,
            "reason": "compute_guard=sustain",
        }
    if mode == "soft_cap":
        return {
            "capture_mode": "sampled",
            "sample_rate": 0.5,
            "max_daily_storage_mb": 90,
            "freshness_floor_seconds": 600,
            "reason": "compute_guard=soft_cap",
        }
    return None


def _registry_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _refresh_summary(payload: dict[str, Any]) -> None:
    rows = _registry_rows(payload)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    active_rows = [row for row in rows if bool(row.get("active", False))]
    summary["total_bots"] = len(rows)
    summary["active_bots"] = len(active_rows)
    summary["inactive_bots"] = max(len(rows) - len(active_rows), 0)
    summary["active_signal_sub_bots"] = sum(1 for row in active_rows if str(row.get("bot_role") or "") == "signal_sub_bot")
    summary["active_infrastructure_sub_bots"] = sum(1 for row in active_rows if str(row.get("bot_role") or "") == "infrastructure_sub_bot")
    summary["active_options_sub_bots"] = sum(1 for row in active_rows if str(row.get("bot_role") or "") == "options_sub_bot")
    summary["inactive_signal_sub_bots"] = sum(
        1 for row in rows if not bool(row.get("active", False)) and str(row.get("bot_role") or "") == "signal_sub_bot"
    )
    summary["inactive_infrastructure_sub_bots"] = sum(
        1 for row in rows if not bool(row.get("active", False)) and str(row.get("bot_role") or "") == "infrastructure_sub_bot"
    )
    summary["inactive_options_sub_bots"] = sum(
        1 for row in rows if not bool(row.get("active", False)) and str(row.get("bot_role") or "") == "options_sub_bot"
    )
    summary["data_collection_only_bots"] = sum(1 for row in rows if str(row.get("lifecycle_state") or "") == "data_collection_only")
    summary["training_excluded_bots"] = sum(1 for row in rows if bool(row.get("training_excluded", False)))
    summary["storage_guarded_collectors"] = sum(1 for row in rows if bool(row.get("data_collection_storage_guarded", False)))
    summary["storage_guard_metadata_only_collectors"] = sum(
        1 for row in rows if str(row.get("data_collection_capture_mode") or "") == "metadata_only"
    )
    payload["summary"] = summary
    payload["updated_at_utc"] = iso_now()


def _duplicate_fallback_files(root: Path, *, limit: int = 50000) -> list[Path]:
    if not root.exists() or _is_protected_volume(root):
        return []
    out: list[Path] = []
    for path in root.rglob("*.local_fallback*"):
        if path.is_file() and not path.is_symlink() and _is_within_root(path, root) and not _is_protected_volume(path):
            out.append(path)
            if len(out) >= limit:
                break
    return out


def _space_candidate_record(path: Path, root: Path, *, reason: str, priority: int, now_ts: float) -> dict[str, Any] | None:
    if path.is_symlink() or _is_protected_volume(path) or not _is_within_root(path, root):
        return None
    try:
        stat = path.stat()
    except Exception:
        return None
    if not path.is_file():
        return None
    size_bytes = max(int(stat.st_size), 0)
    age_hours = max((float(now_ts) - float(stat.st_mtime)) / 3600.0, 0.0)
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(root)) if _is_within_root(path, root) else path.name,
        "reason": reason,
        "priority": int(priority),
        "size_bytes": size_bytes,
        "size_gb": round(_gb(size_bytes), 6),
        "age_hours": round(age_hours, 3),
    }


def _canonical_sibling_for_local_fallback(path: Path) -> Path | None:
    marker = ".local_fallback"
    name = path.name
    if marker not in name:
        return None
    canonical_name = name.split(marker, 1)[0]
    if not canonical_name:
        return None
    canonical = path.with_name(canonical_name)
    if canonical.exists():
        return canonical
    compressed = path.with_name(f"{canonical_name}.gz")
    if compressed.exists():
        return compressed
    return canonical


def _safe_space_recovery_candidates(
    root: Path,
    *,
    duplicate_files: list[Path],
    min_age_hours: float,
    candidate_limit: int,
    scan_file_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not root.exists() or _is_protected_volume(root):
        return [], {
            "scan_root_exists": bool(root.exists()),
            "protected_volume_blocked": bool(_is_protected_volume(root)),
            "scanned_files": 0,
            "scan_limit_reached": False,
        }

    now_ts = datetime.now(timezone.utc).timestamp()
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    unbacked_duplicate_count = 0
    unbacked_duplicate_bytes = 0
    min_age = max(float(min_age_hours), 0.0)

    for path in duplicate_files:
        key = str(path)
        if key in seen:
            continue
        canonical = _canonical_sibling_for_local_fallback(path)
        if (
            canonical is None
            or not canonical.exists()
            or canonical.is_symlink()
            or _is_protected_volume(canonical)
            or not _is_within_root(canonical, root)
        ):
            try:
                unbacked_duplicate_bytes += int(path.stat().st_size)
            except Exception:
                pass
            unbacked_duplicate_count += 1
            continue
        record = _space_candidate_record(path, root, reason="duplicate_local_fallback_artifact", priority=100, now_ts=now_ts)
        if record is not None:
            if float(record.get("age_hours") or 0.0) < min_age:
                continue
            record["canonical_path"] = str(canonical)
            record["canonical_relative_path"] = str(canonical.relative_to(root)) if _is_within_root(canonical, root) else canonical.name
            record["canonical_exists"] = True
            record["canonical_compressed"] = str(canonical.name).endswith(".gz")
            candidates.append(record)
            seen.add(key)
        if len(candidates) >= max(int(candidate_limit), 1):
            return candidates, {
                "scan_root_exists": True,
                "protected_volume_blocked": False,
                "scanned_files": 0,
                "scan_limit_reached": True,
                "unbacked_duplicate_count": unbacked_duplicate_count,
                "unbacked_duplicate_gb": round(_gb(unbacked_duplicate_bytes), 3),
            }

    scanned = 0
    scan_limit_reached = False
    excluded_dirs = {".git", ".Spotlight-V100", ".Trashes", ".fseventsd", "__pycache__", "node_modules"}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        if _is_protected_volume(current):
            dirnames[:] = []
            continue
        dirnames[:] = [
            name
            for name in dirnames
            if name not in excluded_dirs
            and not (current / name).is_symlink()
            and not _is_protected_volume(current / name)
        ]
        for name in filenames:
            scanned += 1
            if scanned > max(int(scan_file_limit), 1):
                scan_limit_reached = True
                break
            path = current / name
            key = str(path)
            if key in seen or path.is_symlink():
                continue
            lower_name = name.lower()
            reason = ""
            priority = 0
            if name in SAFE_METADATA_NAMES or name.startswith("._"):
                reason = "safe_os_metadata_artifact"
                priority = 60
            elif lower_name.endswith(SAFE_STALE_SUFFIXES):
                try:
                    age_hours = max((now_ts - path.stat().st_mtime) / 3600.0, 0.0)
                except Exception:
                    continue
                if age_hours < min_age:
                    continue
                reason = "stale_partial_or_temp_artifact"
                priority = 80
            else:
                continue
            record = _space_candidate_record(path, root, reason=reason, priority=priority, now_ts=now_ts)
            if record is not None:
                candidates.append(record)
                seen.add(key)
            if len(candidates) >= max(int(candidate_limit), 1):
                scan_limit_reached = True
                break
        if scan_limit_reached:
            break

    candidates.sort(key=lambda row: (int(row.get("priority", 0)), int(row.get("size_bytes", 0)), float(row.get("age_hours", 0.0))), reverse=True)
    return candidates, {
        "scan_root_exists": True,
        "protected_volume_blocked": False,
        "scanned_files": scanned,
        "scan_limit_reached": bool(scan_limit_reached),
        "unbacked_duplicate_count": unbacked_duplicate_count,
        "unbacked_duplicate_gb": round(_gb(unbacked_duplicate_bytes), 3),
    }


def _effective_space_recovery_delete_gb(*, available_gb: float, target_free_gb: float, max_delete_gb: float) -> float:
    max_delete = max(float(max_delete_gb), 0.0)
    target = max(float(target_free_gb), 0.0)
    if target <= 0.0:
        return round(max_delete, 3)
    deficit = max(target - max(float(available_gb), 0.0), 0.0)
    return round(min(max_delete, deficit), 3) if deficit > 0.0 else 0.0


def _select_space_recovery_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_delete_gb: float,
    jumbo_duplicate_gb: float,
) -> list[dict[str, Any]]:
    max_bytes = int(max(float(max_delete_gb), 0.0) * (1024**3))
    jumbo_duplicate_bytes = int(max(float(jumbo_duplicate_gb), 0.0) * (1024**3))
    if max_bytes <= 0:
        return []
    selected: list[dict[str, Any]] = []
    selected_bytes = 0
    for row in candidates:
        size_bytes = max(int(row.get("size_bytes") or 0), 0)
        if size_bytes <= 0:
            selected.append(row)
            continue
        if selected_bytes + size_bytes > max_bytes and selected:
            continue
        if selected_bytes + size_bytes > max_bytes and not selected:
            if (
                str(row.get("reason") or "") == "duplicate_local_fallback_artifact"
                and jumbo_duplicate_bytes > 0
                and size_bytes <= jumbo_duplicate_bytes
            ):
                row["selected_over_wave_cap"] = True
                row["selection_reason"] = "single_jumbo_duplicate_fallback_artifact"
                selected.append(row)
                selected_bytes += size_bytes
            continue
        selected.append(row)
        selected_bytes += size_bytes
    return selected


def _reason_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        reason = str(row.get("reason") or "unknown")
        size = max(int(row.get("size_bytes") or 0), 0)
        bucket = out.setdefault(reason, {"count": 0, "bytes": 0, "gb": 0.0})
        bucket["count"] = int(bucket["count"]) + 1
        bucket["bytes"] = int(bucket["bytes"]) + size
        bucket["gb"] = round(_gb(int(bucket["bytes"])), 6)
    return out


def build_payload(
    *,
    external_root: Path,
    registry_path: Path,
    warn_gb: float,
    throttle_gb: float,
    critical_gb: float,
    apply: bool,
    cleanup_duplicates: bool,
    space_recovery: bool = False,
    space_recovery_max_delete_gb: float = DEFAULT_SPACE_RECOVERY_MAX_DELETE_GB,
    space_recovery_target_free_gb: float = DEFAULT_SPACE_RECOVERY_TARGET_FREE_GB,
    space_recovery_min_age_hours: float = DEFAULT_SPACE_RECOVERY_MIN_AGE_HOURS,
    space_recovery_candidate_limit: int = DEFAULT_SPACE_RECOVERY_CANDIDATE_LIMIT,
    space_recovery_scan_file_limit: int = DEFAULT_SPACE_RECOVERY_SCAN_FILE_LIMIT,
    space_recovery_jumbo_duplicate_gb: float = DEFAULT_SPACE_RECOVERY_JUMBO_DUPLICATE_GB,
) -> dict[str, Any]:
    disk = _disk_usage(external_root)
    available_gb = _gb(int(disk.get("available_bytes") or 0))
    used_ratio = float(disk.get("used_ratio") or 1.0)
    mode = _mode_for_space(
        available_gb=available_gb,
        used_ratio=used_ratio,
        warn_gb=warn_gb,
        throttle_gb=throttle_gb,
        critical_gb=critical_gb,
    )
    registry = load_json(registry_path)
    rows = _registry_rows(registry)
    collectors = [
        row
        for row in rows
        if bool(row.get("active", False))
        and bool(row.get("data_collection_active", False))
        and str(row.get("lifecycle_state") or "") == "data_collection_only"
    ]
    changes: list[dict[str, Any]] = []
    now = iso_now()
    for row in collectors:
        kind = _collector_kind(row)
        profile = _guard_profile(mode, kind)
        compute_floor = _compute_guard_floor(row)
        if compute_floor:
            profile = {
                **profile,
                "capture_mode": compute_floor["capture_mode"],
                "sample_rate": min(float(profile["sample_rate"]), float(compute_floor["sample_rate"])),
                "max_daily_storage_mb": min(int(profile["max_daily_storage_mb"]), int(compute_floor["max_daily_storage_mb"])),
                "freshness_floor_seconds": max(int(profile["freshness_floor_seconds"]), int(compute_floor["freshness_floor_seconds"])),
            }
        desired = {
            "data_collection_storage_guarded": True,
            "data_collection_storage_guard_mode": mode,
            "data_collection_capture_mode": profile["capture_mode"],
            "data_collection_sample_rate": profile["sample_rate"],
            "data_collection_max_daily_storage_mb": profile["max_daily_storage_mb"],
            "data_collection_storage_guard_updated_utc": now,
            "data_collection_runtime_dependency_profile": (
                "mlx_optional_research_only" if kind == "quant_research" else str(row.get("data_collection_runtime_dependency_profile") or "")
            ),
            "storage_pressure_capture_reason": (
                f"external_available_gb={available_gb:.2f};mode={mode};{compute_floor['reason']}"
                if compute_floor
                else f"external_available_gb={available_gb:.2f};mode={mode}"
            ),
            "freshness_slo_seconds": max(int(row.get("freshness_slo_seconds") or 0), int(profile["freshness_floor_seconds"])),
        }
        if profile["retention_profile"]:
            desired["retention_profile"] = profile["retention_profile"]
        delta = {key: value for key, value in desired.items() if row.get(key) != value}
        if delta:
            changes.append({"bot_id": str(row.get("bot_id") or ""), "kind": kind, "updates": delta})
            if apply:
                row.update(delta)

    duplicate_files = _duplicate_fallback_files(external_root) if cleanup_duplicates or space_recovery else []
    duplicate_bytes = 0
    deleted_duplicates: list[str] = []
    for path in duplicate_files:
        try:
            duplicate_bytes += int(path.stat().st_size)
        except Exception:
            continue

    need_safe_recovery_scan = bool(space_recovery or cleanup_duplicates)
    space_candidates, space_scan = _safe_space_recovery_candidates(
        external_root,
        duplicate_files=duplicate_files,
        min_age_hours=float(space_recovery_min_age_hours),
        candidate_limit=max(int(space_recovery_candidate_limit), 1),
        scan_file_limit=max(int(space_recovery_scan_file_limit), 1),
    ) if need_safe_recovery_scan else ([], {"scan_root_exists": bool(external_root.exists()), "protected_volume_blocked": bool(_is_protected_volume(external_root)), "scanned_files": 0, "scan_limit_reached": False})
    target_free_gb = max(float(space_recovery_target_free_gb), 0.0)
    target_free_deficit_gb = max(target_free_gb - float(available_gb), 0.0) if target_free_gb > 0.0 else 0.0
    effective_max_delete_gb = _effective_space_recovery_delete_gb(
        available_gb=float(available_gb),
        target_free_gb=target_free_gb,
        max_delete_gb=float(space_recovery_max_delete_gb),
    )
    selected_space_candidates = _select_space_recovery_candidates(
        space_candidates,
        max_delete_gb=float(effective_max_delete_gb),
        jumbo_duplicate_gb=float(space_recovery_jumbo_duplicate_gb),
    )
    deleted_space: list[dict[str, Any]] = []
    delete_errors: list[dict[str, Any]] = []
    if apply and space_recovery:
        for row in selected_space_candidates:
            path = Path(str(row.get("path") or "")).expanduser()
            if _is_protected_volume(path) or not _is_within_root(path, external_root) or path.is_symlink():
                delete_errors.append({"path": str(path), "reason": "safety_check_failed"})
                continue
            try:
                path.unlink()
                deleted_space.append(row)
                if str(row.get("reason") or "") == "duplicate_local_fallback_artifact":
                    deleted_duplicates.append(str(path))
            except Exception as exc:
                delete_errors.append({"path": str(path), "reason": type(exc).__name__})
    elif apply and cleanup_duplicates:
        selected_duplicate_candidates = _select_space_recovery_candidates(
            [
                row
                for row in space_candidates
                if str(row.get("reason") or "") == "duplicate_local_fallback_artifact"
            ],
            max_delete_gb=float(space_recovery_max_delete_gb),
            jumbo_duplicate_gb=float(space_recovery_jumbo_duplicate_gb),
        )
        for row in selected_duplicate_candidates:
            path = Path(str(row.get("path") or "")).expanduser()
            if _is_protected_volume(path) or not _is_within_root(path, external_root) or path.is_symlink():
                continue
            try:
                path.unlink()
                deleted_space.append(row)
                deleted_duplicates.append(str(path))
            except Exception:
                continue

    backup_path = ""
    if apply and changes:
        backup = registry_path.parent / "governance" / "lifecycle" / f"master_bot_registry.data_collection_storage_guard_backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if registry_path.exists():
            backup.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
            backup_path = str(backup)
        _refresh_summary(registry)
        registry_path.write_text(json.dumps(registry, ensure_ascii=True, indent=2), encoding="utf-8")

    status = "ready" if mode == "normal" else ("degraded" if mode in {"watch", "throttle"} else "blocked")
    return {
        "timestamp_utc": now,
        "schema_version": 1,
        "ok": mode != "critical",
        "overall_status": status,
        "apply_requested": bool(apply),
        "external_root": str(external_root),
        "disk": {**disk, "available_gb": round(available_gb, 3), "used_percent": round(used_ratio * 100.0, 3)},
        "thresholds": {"warn_gb": warn_gb, "throttle_gb": throttle_gb, "critical_gb": critical_gb},
        "guard_mode": mode,
        "collector_count": len(collectors),
        "planned_changes": changes[:200],
        "changed_count": len(changes),
        "registry_backup_path": backup_path,
        "duplicate_cleanup": {
            "enabled": bool(cleanup_duplicates),
            "candidate_count": len(duplicate_files),
            "candidate_bytes": duplicate_bytes,
            "candidate_gb": round(_gb(duplicate_bytes), 3),
            "deleted_count": len(deleted_duplicates),
            "deleted_gb": round(sum(float(row.get("size_gb") or 0.0) for row in deleted_space if str(row.get("reason") or "") == "duplicate_local_fallback_artifact"), 3)
            if apply
            else (round(_gb(duplicate_bytes), 3) if apply else 0.0),
        },
        "safe_space_recovery": {
            "enabled": bool(space_recovery),
            "apply_requested": bool(apply and space_recovery),
            "root": str(external_root),
            "max_delete_gb": round(max(float(space_recovery_max_delete_gb), 0.0), 3),
            "jumbo_duplicate_gb": round(max(float(space_recovery_jumbo_duplicate_gb), 0.0), 3),
            "target_free_gb": round(target_free_gb, 3),
            "target_free_deficit_gb": round(target_free_deficit_gb, 3),
            "effective_max_delete_gb": round(float(effective_max_delete_gb), 3),
            "reserve_rebuild_required": bool(target_free_deficit_gb > 0.25 and selected_space_candidates),
            "min_age_hours": round(max(float(space_recovery_min_age_hours), 0.0), 3),
            "candidate_limit": max(int(space_recovery_candidate_limit), 1),
            "scan_file_limit": max(int(space_recovery_scan_file_limit), 1),
            "scan": space_scan,
            "candidate_count": len(space_candidates),
            "candidate_bytes": sum(max(int(row.get("size_bytes") or 0), 0) for row in space_candidates),
            "candidate_gb": round(_gb(sum(max(int(row.get("size_bytes") or 0), 0) for row in space_candidates)), 3),
            "selected_count": len(selected_space_candidates),
            "selected_bytes": sum(max(int(row.get("size_bytes") or 0), 0) for row in selected_space_candidates),
            "selected_gb": round(_gb(sum(max(int(row.get("size_bytes") or 0), 0) for row in selected_space_candidates)), 3),
            "deleted_count": len(deleted_space),
            "deleted_bytes": sum(max(int(row.get("size_bytes") or 0), 0) for row in deleted_space),
            "deleted_gb": round(_gb(sum(max(int(row.get("size_bytes") or 0), 0) for row in deleted_space)), 3),
            "delete_error_count": len(delete_errors),
            "delete_errors": delete_errors[:20],
            "by_reason": _reason_counts(space_candidates),
            "selected_by_reason": _reason_counts(selected_space_candidates),
            "top_candidates": [
                {
                    "relative_path": str(row.get("relative_path") or ""),
                    "reason": str(row.get("reason") or ""),
                    "size_gb": round(float(row.get("size_gb") or 0.0), 6),
                    "age_hours": round(float(row.get("age_hours") or 0.0), 3),
                }
                for row in space_candidates[:20]
            ],
        },
        "recommended_actions": [
            "keep data-collection-only bots in metadata_only or thin_sample mode until external free space is above the throttle threshold"
            if mode in {"critical", "throttle"}
            else "",
            "run safe BOT_LOGS space recovery in bounded apply waves; it only targets duplicate fallback, stale partial/temp, and OS metadata artifacts"
            if space_recovery and selected_space_candidates and not apply
            else "",
            "remove duplicate .local_fallback files from the external route; they are fallback-copy artifacts, not canonical live files"
            if cleanup_duplicates and duplicate_files and not apply
            else "",
            "run storage-tier-policy after cleanup to find the next archive/compact targets",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard new data-collection bots from exhausting external storage.")
    parser.add_argument("--external-root", default=str(DEFAULT_EXTERNAL_ROOT))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--warn-gb", type=float, default=120.0)
    parser.add_argument("--throttle-gb", type=float, default=80.0)
    parser.add_argument("--critical-gb", type=float, default=40.0)
    parser.add_argument("--cleanup-duplicates", action="store_true")
    parser.add_argument("--space-recovery", action="store_true")
    parser.add_argument("--space-recovery-max-delete-gb", type=float, default=float(os.getenv("BOT_LOGS_SPACE_RECOVERY_MAX_DELETE_GB", str(DEFAULT_SPACE_RECOVERY_MAX_DELETE_GB))))
    parser.add_argument("--space-recovery-target-free-gb", type=float, default=float(os.getenv("BOT_LOGS_SPACE_RECOVERY_TARGET_FREE_GB", str(DEFAULT_SPACE_RECOVERY_TARGET_FREE_GB))))
    parser.add_argument("--space-recovery-min-age-hours", type=float, default=float(os.getenv("BOT_LOGS_SPACE_RECOVERY_MIN_AGE_HOURS", str(DEFAULT_SPACE_RECOVERY_MIN_AGE_HOURS))))
    parser.add_argument("--space-recovery-candidate-limit", type=int, default=int(os.getenv("BOT_LOGS_SPACE_RECOVERY_CANDIDATE_LIMIT", str(DEFAULT_SPACE_RECOVERY_CANDIDATE_LIMIT))))
    parser.add_argument("--space-recovery-scan-file-limit", type=int, default=int(os.getenv("BOT_LOGS_SPACE_RECOVERY_SCAN_FILE_LIMIT", str(DEFAULT_SPACE_RECOVERY_SCAN_FILE_LIMIT))))
    parser.add_argument("--space-recovery-jumbo-duplicate-gb", type=float, default=float(os.getenv("BOT_LOGS_SPACE_RECOVERY_JUMBO_DUPLICATE_GB", str(DEFAULT_SPACE_RECOVERY_JUMBO_DUPLICATE_GB))))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        external_root=Path(args.external_root).expanduser(),
        registry_path=Path(args.registry).expanduser(),
        warn_gb=float(args.warn_gb),
        throttle_gb=float(args.throttle_gb),
        critical_gb=float(args.critical_gb),
        apply=bool(args.apply),
        cleanup_duplicates=bool(args.cleanup_duplicates),
        space_recovery=bool(args.space_recovery),
        space_recovery_max_delete_gb=float(args.space_recovery_max_delete_gb),
        space_recovery_target_free_gb=float(args.space_recovery_target_free_gb),
        space_recovery_min_age_hours=float(args.space_recovery_min_age_hours),
        space_recovery_candidate_limit=int(args.space_recovery_candidate_limit),
        space_recovery_scan_file_limit=int(args.space_recovery_scan_file_limit),
        space_recovery_jumbo_duplicate_gb=float(args.space_recovery_jumbo_duplicate_gb),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "data_collection_storage_guard "
            f"overall_status={payload.get('overall_status')} "
            f"guard_mode={payload.get('guard_mode')} "
            f"collector_count={payload.get('collector_count')} "
            f"changed_count={payload.get('changed_count')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
