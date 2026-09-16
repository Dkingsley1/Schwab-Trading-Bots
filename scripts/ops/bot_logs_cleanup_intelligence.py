#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import shutil
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.storage_mounts import configured_external_project_root, external_mount_candidates, external_project_dir
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.storage_mounts import configured_external_project_root, external_mount_candidates, external_project_dir
    from .long_runtime_common import iso_now, load_json, ordered_unique, write_payload

from scripts.ops import verified_duplicate_cleanup as verified


def _verified_external_root():
    configured = configured_external_project_root()
    candidates = ([configured] if configured is not None else []) + [
        mount / external_project_dir() for mount in external_mount_candidates()
    ]
    for candidate in candidates:
        route = verified.safety.inspect_storage_path(candidate)
        if route.get("status") not in {"present", "missing"} or route.get("symlinks"):
            raise RuntimeError("protected_or_unavailable_external_route")
        if route["status"] == "present":
            return candidate
    raise verified.safety.Deferred("external_root_unavailable")


def _safe_present(path):
    route = verified.safety.inspect_storage_path(path)
    return route.get("status") == "present" and not route.get("symlinks")


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "bot_logs_cleanup_intelligence_latest.json"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "governance" / "health" / "bot_logs_cleanup_intelligence_history.jsonl"
DEFAULT_TARGET_FREE_GB = 125.0
DEFAULT_MIN_AGE_HOURS = 12.0
DEFAULT_PREFIX_VERIFY_BYTES = 65536
DEFAULT_FALLBACK_QUARANTINE_ROOT = PROJECT_ROOT / "local_fallback_storage" / "quarantine" / "bot_logs_cleanup"
DEFAULT_INTERNAL_QUARANTINE_MIN_FREE_GB = float(
    os.getenv(
        "BOT_LOGS_CLEANUP_INTERNAL_QUARANTINE_MIN_FREE_GB",
        os.getenv("BOT_LOCAL_STORAGE_TARGET_FREE_GB", str(DEFAULT_TARGET_FREE_GB)),
    )
)
DEFAULT_CORRUPT_SQLITE_QUARANTINE_MIN_AGE_HOURS = float(
    os.getenv("BOT_LOGS_CLEANUP_CORRUPT_SQLITE_MIN_AGE_HOURS", "24")
)


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


def _gb(value: int | float) -> float:
    return round(float(value) / float(1024**3), 3)


def _disk_snapshot(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
    except Exception:
        return {
            "path": str(path),
            "exists": bool(path.exists()),
            "total_bytes": 0,
            "used_bytes": 0,
            "free_bytes": 0,
            "free_gb": 0.0,
            "used_gb": 0.0,
            "capacity_pct": 0.0,
        }
    capacity_pct = 100.0 * float(usage.used) / max(float(usage.total), 1.0)
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "free_gb": _gb(usage.free),
        "used_gb": _gb(usage.used),
        "capacity_pct": round(capacity_pct, 3),
    }


def _nearest_existing_parent(path: Path) -> Path:
    current = path.expanduser()
    for candidate in (current, *current.parents):
        if candidate.exists():
            return candidate
    return current


def _quarantine_disk_snapshot(path: Path) -> dict[str, Any]:
    return _disk_snapshot(_nearest_existing_parent(path))


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def _file_allocated_size(path: Path) -> int:
    try:
        stat = path.stat()
    except Exception:
        return 0
    logical_size = max(int(stat.st_size), 0)
    blocks = getattr(stat, "st_blocks", None)
    if blocks is None:
        return logical_size
    return min(logical_size, max(int(blocks), 0) * 512)


def _file_identity(path: Path) -> dict[str, int]:
    try:
        verified.identity(path)
        stat = path.lstat()
    except Exception:
        return {}
    return {
        "inode": int(stat.st_ino),
        "device": int(stat.st_dev),
        "ctime_ns": int(stat.st_ctime_ns),
        "nlink": int(stat.st_nlink),
        "size_bytes": int(stat.st_size),
        "allocated_bytes": _file_allocated_size(path),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _file_age_hours(path: Path, *, now: datetime | None = None) -> float:
    try:
        mt = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return 0.0
    current = now or datetime.now(timezone.utc)
    return max((current - mt).total_seconds() / 3600.0, 0.0)


def _today_tokens(now: datetime | None = None) -> set[str]:
    current = now or datetime.now(timezone.utc)
    tokens = {current.strftime("%Y%m%d")}
    try:
        local = current.astimezone()
        tokens.add(local.strftime("%Y%m%d"))
    except Exception:
        pass
    return tokens


def _protects_current_day(path: Path, *, now: datetime | None = None) -> bool:
    name = path.name
    return any(token in name for token in _today_tokens(now))


def _gzip_duplicate_verification(raw_path: Path, gz_path: Path, *, prefix_bytes: int) -> dict[str, Any]:
    # The legacy argument remains accepted but can no longer weaken verification.
    try:
        return verified.verify_pair(raw_path, gz_path)
    except (OSError, RuntimeError, EOFError, zlib.error) as exc:
        return {"ok": False, "state": "verification_failed", "reason": str(exc)}


def _candidate_family(path: Path, root: Path) -> str:
    try:
        rel = str(path.relative_to(root))
    except Exception:
        rel = str(path)
    if rel.startswith("data/stale_stage/"):
        return "stale_stage"
    if rel.startswith("decision_explanations/"):
        return "decision_explanations"
    if rel.startswith("decisions/"):
        return "decisions"
    if rel.startswith("governance/execution_lanes/"):
        return "execution_lanes"
    if rel.startswith("governance/channels/"):
        return "governance_channels"
    if rel.startswith("governance/"):
        return "governance"
    if rel.startswith("data/"):
        return "data"
    return "other"


def _risk_score(*, tier: int, family: str, current_day: bool, age_hours: float) -> int:
    score = 0
    if tier >= 3:
        score += 4
    if family in {"decisions", "data"}:
        score += 2
    if current_day:
        score += 5
    if age_hours < 24.0:
        score += 2
    return score


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _scan_duplicate_jsonl_gzip(
    root: Path,
    *,
    min_age_hours: float,
    protect_current_day: bool,
    prefix_verify_bytes: int,
    now: datetime | None = None,
    paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    current = now or datetime.now(timezone.utc)
    for raw_path in (paths if paths is not None else verified.inventory(root)):
        if raw_path.suffix != ".jsonl":
            continue
        gz_path = raw_path.with_suffix(raw_path.suffix + ".gz")
        try:
            verified.identity(raw_path)
            verified.identity(gz_path)
        except (OSError, RuntimeError):
            continue
        age_hours = min(_file_age_hours(raw_path, now=current), _file_age_hours(gz_path, now=current))
        current_day = _protects_current_day(raw_path, now=current)
        family = _candidate_family(raw_path, root)
        verification = {"ok": False, "state": "full_verification_required_at_apply"}
        eligible = True
        blocked_reasons = []
        try:
            verified.require_log_source(root, raw_path)
        except RuntimeError as exc:
            eligible = False
            blocked_reasons.append(str(exc))
        if age_hours < float(min_age_hours):
            eligible = False
            blocked_reasons.append("too_recent")
        if current_day:
            eligible = False
            blocked_reasons.append("current_day_protected")
        dated = re.search(r"(?:^|_)(\d{8})(?:\.|_)", raw_path.name)
        try:
            closed_date = datetime.strptime(dated.group(1), "%Y%m%d").date() if dated else None
        except ValueError:
            closed_date = None
        if not current_day and (closed_date is None or closed_date >= current.date()):
            eligible = False
            blocked_reasons.append("closed_date_required")
        if family == "stale_stage":
            eligible = False
            blocked_reasons.append("manifest_retention_owner_required")
        if (any(part in {"cold_archive", "cold_archives", "deep_cold", "quarantine", "training"}
                for part in raw_path.relative_to(root).parts) or "_latest" in raw_path.name):
            eligible = False
            blocked_reasons.append("retained_artifact_owner_required")
        raw_size = _file_size(raw_path)
        gz_size = _file_size(gz_path)
        raw_allocated_size = _file_allocated_size(raw_path)
        rows.append(
            {
                "tier": 1,
                "tier_name": "lossless_duplicate_raw_jsonl",
                "family": family,
                "relative_path": _relative(raw_path, root),
                "path": str(raw_path),
                "compressed_path": str(gz_path),
                "compressed_identity": _file_identity(gz_path),
                "size_bytes": int(raw_size),
                "allocated_bytes": int(raw_allocated_size),
                "compressed_size_bytes": int(gz_size),
                "reclaimable_bytes": int(raw_allocated_size),
                "age_hours": round(age_hours, 3),
                "current_day": bool(current_day),
                "eligible": bool(eligible),
                "verification": verification,
                "blocked_reasons": ordered_unique(blocked_reasons),
                "risk_score": _risk_score(tier=1, family=family, current_day=current_day, age_hours=age_hours),
            }
        )
    return rows


def _stale_stage_value(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root / "data" / "stale_stage")
        label = str(rel.parts[0] if rel.parts else "")
    except Exception:
        label = ""
    if label.startswith("decision_explanations") or label == "decision_explanations":
        return "high"
    if label.startswith("decisions") or label == "decisions":
        return "critical"
    if label.startswith("governance") or label == "governance":
        return "medium"
    return "low"


def _value_window_hours(value: str) -> float:
    return {
        "low": 24.0,
        "medium": 5.0 * 24.0,
        "high": 14.0 * 24.0,
        "critical": 45.0 * 24.0,
    }.get(str(value or "").strip().lower(), 14.0 * 24.0)


def _scan_stale_stage(root: Path, *, now: datetime | None = None, paths: list[Path] | None = None) -> list[dict[str, Any]]:
    stale_root = root / "data" / "stale_stage"
    rows: list[dict[str, Any]] = []
    if paths is None and not verified.safety.allowed(stale_root, missing=True).exists():
        return rows
    current = now or datetime.now(timezone.utc)
    for path in (paths if paths is not None else verified.inventory(stale_root)):
        if not path.is_relative_to(stale_root):
            continue
        if not _file_identity(path) or path.name == "stale_manifest.jsonl":
            continue
        value = _stale_stage_value(path, root)
        age_hours = _file_age_hours(path, now=current)
        min_age = _value_window_hours(value)
        age_eligible = age_hours >= min_age
        eligible = False
        blocked = ["manifest_retention_owner_required"]
        if not age_eligible:
            blocked.append(f"value_window_not_met:{value}")
        size_bytes = _file_size(path)
        allocated_bytes = _file_allocated_size(path)
        rows.append(
            {
                "tier": 2,
                "tier_name": "stale_stage_reaper",
                "family": "stale_stage",
                "economic_value": value,
                "relative_path": _relative(path, root),
                "path": str(path),
                "size_bytes": int(size_bytes),
                "allocated_bytes": int(allocated_bytes),
                "reclaimable_bytes": int(allocated_bytes),
                "age_hours": round(age_hours, 3),
                "min_age_hours": min_age,
                "eligible": bool(eligible),
                "age_eligible": age_eligible,
                "verification": {"ok": False, "state": "manifest_retention_owner_required", "reason": "Age is advisory; data-retention owns manifest, hash, protected-evidence and expiry checks"},
                "blocked_reasons": blocked,
                "risk_score": _risk_score(tier=2, family="stale_stage", current_day=False, age_hours=age_hours),
            }
        )
    return rows


def _local_fallback_canonical_name(name: str) -> str:
    marker = ".local_fallback"
    if marker not in name:
        return name
    return name.split(marker, 1)[0]


def _scan_external_local_fallback_copies(
    root: Path,
    *,
    project_root: Path,
    fallback_quarantine_root: Path,
    min_quarantine_free_gb: float = DEFAULT_INTERNAL_QUARANTINE_MIN_FREE_GB,
    now: datetime | None = None,
    paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows

    current = now or datetime.now(timezone.utc)
    try:
        quarantine_resolved = fallback_quarantine_root.resolve(strict=False)
        root_resolved = root.resolve(strict=False)
        quarantine_inside_external = str(quarantine_resolved).startswith(str(root_resolved))
    except Exception:
        quarantine_inside_external = False
    quarantine_disk = _quarantine_disk_snapshot(fallback_quarantine_root)
    quarantine_free_bytes = _safe_int(quarantine_disk.get("free_bytes"), 0)
    min_quarantine_free_bytes = int(max(float(min_quarantine_free_gb), 0.0) * (1024**3))

    for path in (paths if paths is not None else verified.inventory(root)):
        if ".local_fallback" not in path.name or not _file_identity(path):
            continue
        size_bytes = _file_size(path)
        allocated_bytes = _file_allocated_size(path)
        rel_path = _relative(path, root)
        canonical_name = _local_fallback_canonical_name(path.name)
        canonical_rel = str(Path(rel_path).with_name(canonical_name))
        local_preservation_path = project_root / "local_fallback_storage" / canonical_rel
        external_canonical_path = root / canonical_rel
        age_hours = _file_age_hours(path, now=current)
        current_day = _protects_current_day(path, now=current)
        family = _candidate_family(external_canonical_path, root)
        destination_path = fallback_quarantine_root / rel_path
        blocked_reasons: list[str] = []
        eligible = True
        verification_state = "quarantine_preserves_copy"
        verification_reason = (
            "external failback conflict copy can be moved to local quarantine before removal from BOT_LOGS"
        )
        if quarantine_inside_external:
            eligible = False
            blocked_reasons.append("quarantine_root_inside_external")
            verification_state = "unsafe_quarantine_root"
            verification_reason = "quarantine root must be outside BOT_LOGS to reclaim space"
        if quarantine_free_bytes and quarantine_free_bytes < size_bytes + min_quarantine_free_bytes:
            eligible = False
            blocked_reasons.append("quarantine_root_low_free_space")
            verification_state = "unsafe_quarantine_capacity"
            verification_reason = (
                "local quarantine free space is below the reserve needed to preserve this copy without "
                "pressuring the internal SSD"
            )
        rows.append(
            {
                "tier": 2,
                "tier_name": "external_failback_conflict_quarantine",
                "family": family,
                "relative_path": rel_path,
                "path": str(path),
                "action": "quarantine",
                "destination_path": str(destination_path),
                "canonical_relative_path": canonical_rel,
                "local_preservation_path": str(local_preservation_path),
                "local_preservation_exists": bool(_safe_present(local_preservation_path)),
                "external_canonical_exists": bool(_safe_present(external_canonical_path)),
                "size_bytes": int(size_bytes),
                "allocated_bytes": int(allocated_bytes),
                "reclaimable_bytes": int(allocated_bytes),
                "quarantine_disk": quarantine_disk,
                "min_quarantine_free_gb": round(float(min_quarantine_free_gb), 3),
                "age_hours": round(age_hours, 3),
                "current_day": bool(current_day),
                "eligible": bool(eligible),
                "verification": {
                    "ok": bool(eligible),
                    "state": verification_state,
                    "reason": verification_reason,
                },
                "blocked_reasons": ordered_unique(blocked_reasons),
                "risk_score": 1,
            }
        )
    return rows


def _scan_stateful_corrupt_quarantine(
    root: Path,
    *,
    fallback_quarantine_root: Path,
    min_age_hours: float = DEFAULT_CORRUPT_SQLITE_QUARANTINE_MIN_AGE_HOURS,
    min_quarantine_free_gb: float = DEFAULT_INTERNAL_QUARANTINE_MIN_FREE_GB,
    now: datetime | None = None,
    paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data_root = root / "data"
    if paths is None and not verified.safety.allowed(data_root, missing=True).exists():
        return rows

    current = now or datetime.now(timezone.utc)
    try:
        quarantine_resolved = fallback_quarantine_root.resolve(strict=False)
        root_resolved = root.resolve(strict=False)
        quarantine_inside_external = str(quarantine_resolved).startswith(str(root_resolved))
    except Exception:
        quarantine_inside_external = False
    quarantine_disk = _quarantine_disk_snapshot(fallback_quarantine_root)
    quarantine_free_bytes = _safe_int(quarantine_disk.get("free_bytes"), 0)
    min_quarantine_free_bytes = int(max(float(min_quarantine_free_gb), 0.0) * (1024**3))

    for path in (paths if paths is not None else verified.inventory(data_root)):
        if path.parent != data_root or ".corrupt-" not in path.name:
            continue
        if not _file_identity(path):
            continue
        lower_name = path.name.lower()
        if ".sqlite" not in lower_name and ".db" not in lower_name:
            continue
        size_bytes = _file_size(path)
        allocated_bytes = _file_allocated_size(path)
        rel_path = _relative(path, root)
        canonical_name = path.name.split(".corrupt-", 1)[0]
        active_sibling = path.with_name(canonical_name)
        age_hours = _file_age_hours(path, now=current)
        destination_path = fallback_quarantine_root / "stateful_corrupt" / rel_path
        blocked_reasons: list[str] = []
        eligible = True
        verification_state = "corrupt_stateful_copy_quarantine_preserves_evidence"
        verification_reason = "old corrupt SQLite quarantine copy can move off BOT_LOGS while the active sibling remains in place"

        if age_hours < max(float(min_age_hours), 0.0):
            eligible = False
            blocked_reasons.append("corrupt_sqlite_min_age_not_met")
            verification_state = "age_policy"
            verification_reason = "corrupt SQLite copy is still inside the quarantine hold window"
        if not _safe_present(active_sibling):
            eligible = False
            blocked_reasons.append("active_stateful_sibling_not_verified")
            verification_state = "active_sibling_missing"
            verification_reason = "keep corrupt copy until the active SQLite sibling is visible on BOT_LOGS"
        if quarantine_inside_external:
            eligible = False
            blocked_reasons.append("quarantine_root_inside_external")
            verification_state = "unsafe_quarantine_root"
            verification_reason = "quarantine root must be outside BOT_LOGS to reclaim space"
        if quarantine_free_bytes and quarantine_free_bytes < size_bytes + min_quarantine_free_bytes:
            eligible = False
            blocked_reasons.append("quarantine_root_low_free_space")
            verification_state = "unsafe_quarantine_capacity"
            verification_reason = "local quarantine free space is below the reserve needed to preserve the corrupt copy"

        rows.append(
            {
                "tier": 2,
                "tier_name": "stateful_corrupt_sqlite_quarantine",
                "family": "stateful_corrupt_sqlite",
                "economic_value": "medium",
                "relative_path": rel_path,
                "path": str(path),
                "action": "quarantine",
                "destination_path": str(destination_path),
                "active_sibling_path": str(active_sibling),
                "active_sibling_exists": bool(active_sibling.exists() and not active_sibling.is_symlink()),
                "size_bytes": int(size_bytes),
                "allocated_bytes": int(allocated_bytes),
                "reclaimable_bytes": int(allocated_bytes),
                "quarantine_disk": quarantine_disk,
                "min_quarantine_free_gb": round(float(min_quarantine_free_gb), 3),
                "age_hours": round(age_hours, 3),
                "min_age_hours": round(float(min_age_hours), 3),
                "current_day": False,
                "eligible": bool(eligible),
                "verification": {
                    "ok": bool(eligible),
                    "state": verification_state,
                    "reason": verification_reason,
                },
                "blocked_reasons": ordered_unique(blocked_reasons),
                "risk_score": _risk_score(tier=2, family="stateful_corrupt_sqlite", current_day=False, age_hours=age_hours),
            }
        )
    return rows


def _select_candidates(
    candidates: list[dict[str, Any]],
    *,
    free_bytes: int,
    target_free_bytes: int,
    max_tier: int,
    max_delete_bytes: int,
) -> list[dict[str, Any]]:
    needed = max(int(target_free_bytes) - int(free_bytes), 0)
    if needed <= 0:
        return []
    eligible = [
        row for row in candidates
        if bool(row.get("eligible", False)) and _safe_int(row.get("tier"), 99) <= int(max_tier)
    ]
    eligible.sort(
        key=lambda row: (
            _safe_int(row.get("tier"), 99),
            0 if str(row.get("action") or "delete") == "delete" else 1,
            _safe_int(row.get("risk_score"), 99),
            0 if _safe_int(row.get("reclaimable_bytes"), 0) >= needed else 1,
            (
                _safe_int(row.get("reclaimable_bytes"), 0)
                if _safe_int(row.get("reclaimable_bytes"), 0) >= needed
                else -_safe_int(row.get("reclaimable_bytes"), 0)
            ),
            str(row.get("relative_path") or ""),
        )
    )
    selected: list[dict[str, Any]] = []
    selected_bytes = 0
    quarantine_selected_by_disk: dict[str, int] = {}
    max_bytes = max(int(max_delete_bytes), 0)
    for row in eligible:
        reclaimable = _safe_int(row.get("reclaimable_bytes"), 0)
        if reclaimable <= 0:
            continue
        if str(row.get("action") or "") == "quarantine":
            quarantine_disk = row.get("quarantine_disk") if isinstance(row.get("quarantine_disk"), dict) else {}
            disk_key = str(quarantine_disk.get("path") or row.get("destination_path") or "local_quarantine")
            quarantine_free_bytes = _safe_int(quarantine_disk.get("free_bytes"), 0)
            min_quarantine_free_bytes = int(max(_safe_float(row.get("min_quarantine_free_gb"), 0.0), 0.0) * (1024**3))
            quarantine_budget = max(quarantine_free_bytes - min_quarantine_free_bytes, 0)
            already_selected = int(quarantine_selected_by_disk.get(disk_key, 0))
            destination_bytes = _safe_int(row.get("size_bytes"), reclaimable)
            if already_selected + destination_bytes > quarantine_budget:
                continue
        if max_bytes and selected_bytes + reclaimable > max_bytes:
            continue
        selected_row = dict(row)
        selected_row["selected"] = True
        selected.append(selected_row)
        selected_bytes += reclaimable
        if str(row.get("action") or "") == "quarantine":
            quarantine_disk = row.get("quarantine_disk") if isinstance(row.get("quarantine_disk"), dict) else {}
            disk_key = str(quarantine_disk.get("path") or row.get("destination_path") or "local_quarantine")
            quarantine_selected_by_disk[disk_key] = int(quarantine_selected_by_disk.get(disk_key, 0)) + _safe_int(
                row.get("size_bytes"), reclaimable
            )
        if int(free_bytes) + selected_bytes >= int(target_free_bytes):
            break
        if max_bytes and selected_bytes >= max_bytes:
            break
    return selected


def _apply_selected(rows: list[dict[str, Any]], *, project_root=None, budget=None, source_root=None) -> dict[str, Any]:
    deleted_files = 0
    deleted_bytes = 0
    offloaded_files = 0
    offloaded_bytes = 0
    errors: list[dict[str, str]] = []
    skipped_rows: list[dict[str, str]] = []
    deleted_rows: list[dict[str, Any]] = []
    offloaded_rows: list[dict[str, Any]] = []
    for row in rows:
        path = Path(str(row.get("path") or "")).expanduser()
        expected_identity = row.get("source_identity") if isinstance(row.get("source_identity"), dict) else {}
        current_identity = _file_identity(path)
        if not current_identity:
            skipped_rows.append({"path": str(path), "reason": "source_missing_or_route_unverifiable"})
            continue
        if expected_identity and current_identity != expected_identity:
            skipped_rows.append({"path": str(path), "reason": "source_changed_since_scan"})
            continue
        size_bytes = _safe_int(row.get("reclaimable_bytes"), _file_allocated_size(path))
        action = str(row.get("action") or "delete")
        if action == "delete":
            if (row.get("tier_name") != "lossless_duplicate_raw_jsonl" or
                    project_root is None or budget is None or not expected_identity):
                skipped_rows.append({"path": str(path), "reason": "verified_duplicate_owner_required"})
                continue
            if _file_identity(Path(row["compressed_path"])) != row.get("compressed_identity"):
                skipped_rows.append({"path": str(path), "reason": "archive_changed_since_scan"})
                continue
            try:
                budget.check()
                identity_keys = ("device", "inode", "size_bytes", "mtime_ns", "ctime_ns")
                expected_pair = tuple(tuple(info[key] for key in identity_keys) for info in
                                      (expected_identity, row["compressed_identity"]))
                proof = verified.remove_pair(project_root, path, Path(row["compressed_path"]), budget,
                                             expected=expected_pair, source_root=source_root)
            except (OSError, RuntimeError, EOFError, zlib.error) as exc:
                skipped_rows.append({"path": str(path), "reason": str(exc)})
                continue
            deleted_files += 1
            deleted_bytes += size_bytes
            deleted_rows.append({"relative_path": str(row.get("relative_path") or ""),
                                 "deleted_bytes": size_bytes, "verification": proof})
            for error in proof.get("persistence_errors", []):
                errors.append({"path": str(path), "source_removed": True, "error": error})
            continue
        if action == "quarantine":
            skipped_rows.append({"path": str(path), "reason": "verified_offload_owner_required"})
            continue
        skipped_rows.append({"path": str(path), "reason": "unsupported_cleanup_action"})
    reclaimed_bytes = int(deleted_bytes + offloaded_bytes)
    return {
        "deleted_files": int(deleted_files),
        "deleted_bytes": int(deleted_bytes),
        "deleted_gb": _gb(deleted_bytes),
        "offloaded_files": int(offloaded_files),
        "offloaded_bytes": int(offloaded_bytes),
        "offloaded_gb": _gb(offloaded_bytes),
        "reclaimed_bytes": int(reclaimed_bytes),
        "reclaimed_gb": _gb(reclaimed_bytes),
        "errors": errors,
        "skipped_files": len(skipped_rows),
        "skipped_rows": skipped_rows[:50],
        "deleted_rows": deleted_rows[:50],
        "offloaded_rows": offloaded_rows[:50],
    }


def _merge_apply_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    merged = {
        "deleted_files": 0,
        "deleted_bytes": 0,
        "offloaded_files": 0,
        "offloaded_bytes": 0,
        "errors": [],
        "skipped_files": 0,
        "skipped_rows": [],
        "deleted_rows": [],
        "offloaded_rows": [],
    }
    for result in results:
        for key in ("deleted_files", "deleted_bytes", "offloaded_files", "offloaded_bytes", "skipped_files"):
            merged[key] += _safe_int(result.get(key), 0)
        for key in ("errors", "skipped_rows", "deleted_rows", "offloaded_rows"):
            values = result.get(key) if isinstance(result.get(key), list) else []
            merged[key].extend(values)
    estimated_reclaimed_bytes = int(merged["deleted_bytes"] + merged["offloaded_bytes"])
    return {
        **merged,
        "deleted_gb": _gb(merged["deleted_bytes"]),
        "offloaded_gb": _gb(merged["offloaded_bytes"]),
        "reclaimed_bytes": estimated_reclaimed_bytes,
        "reclaimed_gb": _gb(estimated_reclaimed_bytes),
        "errors": merged["errors"][:50],
        "skipped_rows": merged["skipped_rows"][:50],
        "deleted_rows": merged["deleted_rows"][:50],
        "offloaded_rows": merged["offloaded_rows"][:50],
    }


def _summarize_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_tier: dict[str, dict[str, Any]] = {}
    by_family: dict[str, dict[str, Any]] = {}
    eligible_bytes = 0
    blocked_count = 0
    for row in rows:
        tier_key = f"tier_{_safe_int(row.get('tier'), 0)}"
        family = str(row.get("family") or "unknown")
        reclaimable = _safe_int(row.get("reclaimable_bytes"), 0)
        for bucket, key in ((by_tier, tier_key), (by_family, family)):
            entry = bucket.setdefault(key, {"files": 0, "eligible_files": 0, "bytes": 0, "eligible_bytes": 0})
            entry["files"] += 1
            entry["bytes"] += reclaimable
            if bool(row.get("eligible", False)):
                entry["eligible_files"] += 1
                entry["eligible_bytes"] += reclaimable
        if bool(row.get("eligible", False)):
            eligible_bytes += reclaimable
        else:
            blocked_count += 1
    return {
        "candidate_count": len(rows),
        "eligible_count": sum(1 for row in rows if bool(row.get("eligible", False))),
        "blocked_count": blocked_count,
        "eligible_bytes": int(eligible_bytes),
        "eligible_gb": _gb(eligible_bytes),
        "by_tier": by_tier,
        "by_family": by_family,
    }


def _top_rows(rows: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    out = []
    for row in sorted(rows, key=lambda item: (-_safe_int(item.get("reclaimable_bytes"), 0), str(item.get("relative_path") or "")))[: max(int(limit), 1)]:
        out.append(
            {
                "tier": _safe_int(row.get("tier"), 0),
                "tier_name": str(row.get("tier_name") or ""),
                "family": str(row.get("family") or ""),
                "relative_path": str(row.get("relative_path") or ""),
                "reclaimable_gb": _gb(_safe_int(row.get("reclaimable_bytes"), 0)),
                "eligible": bool(row.get("eligible", False)),
                "blocked_reasons": list(row.get("blocked_reasons") or []),
                "verification_state": str((row.get("verification") or {}).get("state") or ""),
            }
        )
    return out


def _build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    bot_logs_root: Path | None = None,
    apply: bool = False,
    target_free_gb: float = DEFAULT_TARGET_FREE_GB,
    max_tier: int = 1,
    max_delete_gb: float = 0.0,
    min_age_hours: float = DEFAULT_MIN_AGE_HOURS,
    protect_current_day: bool = True,
    prefix_verify_bytes: int = DEFAULT_PREFIX_VERIFY_BYTES,
    fallback_quarantine_root: Path = DEFAULT_FALLBACK_QUARANTINE_ROOT,
    out_path: Path = DEFAULT_OUT_PATH,
    history_path: Path = DEFAULT_HISTORY_PATH,
    inventory_paths: list[Path] | None = None,
    verification_budget=None,
    max_files: int = 4,
) -> dict[str, Any]:
    external_root = bot_logs_root or _verified_external_root()
    verification_budget = verification_budget or verified.Budget()
    def checked_paths():
        paths = inventory_paths if inventory_paths is not None else verified.inventory(external_root)
        for path in paths:
            verification_budget.check()
            yield path
        verification_budget.check()

    verification_budget.check()
    disk_before = _disk_snapshot(external_root)
    target_free_bytes = int(max(float(target_free_gb), 0.0) * (1024**3))
    free_bytes = _safe_int(disk_before.get("free_bytes"), 0)
    max_delete_bytes = int(max(float(max_delete_gb), 0.0) * (1024**3))
    if max_delete_bytes <= 0:
        max_delete_bytes = max(target_free_bytes - free_bytes + int(10 * 1024**3), 0)

    duplicate_rows = _scan_duplicate_jsonl_gzip(
        external_root,
        min_age_hours=float(min_age_hours),
        protect_current_day=bool(protect_current_day),
        prefix_verify_bytes=max(int(prefix_verify_bytes), 1),
        paths=checked_paths(),
    )
    fallback_rows = _scan_external_local_fallback_copies(
        external_root,
        project_root=project_root,
        fallback_quarantine_root=fallback_quarantine_root,
        paths=checked_paths(),
    ) if max_tier >= 2 else []
    corrupt_rows = _scan_stateful_corrupt_quarantine(
        external_root,
        fallback_quarantine_root=fallback_quarantine_root,
        paths=checked_paths(),
    ) if max_tier >= 2 else []
    stale_rows = _scan_stale_stage(external_root, paths=checked_paths())
    verification_budget.check()
    deep_cold_layer = load_json(verified.safety.allowed(project_root / "governance" / "health" / "deep_cold_storage_layer_latest.json", missing=True))
    deep_cold_summary = (
        deep_cold_layer.get("summary")
        if isinstance(deep_cold_layer.get("summary"), dict)
        else {}
    )
    retention_v2 = load_json(verified.safety.allowed(project_root / "governance" / "health" / "retention_intelligence_v2_latest.json", missing=True))
    retention_report = (
        retention_v2.get("retention_report_card")
        if isinstance(retention_v2.get("retention_report_card"), dict)
        else {}
    )
    all_candidates = duplicate_rows + fallback_rows + corrupt_rows + stale_rows
    for row in fallback_rows + corrupt_rows:
        row["eligible"] = False
        row["blocked_reasons"].append("verified_offload_owner_required")
    for row in all_candidates:
        verification_budget.check()
        path = Path(str(row.get("path") or "")).expanduser()
        identity = _file_identity(path)
        row["source_identity"] = identity
        if identity:
            row["size_bytes"] = int(identity["size_bytes"])
            row["allocated_bytes"] = int(identity["allocated_bytes"])
            row["reclaimable_bytes"] = int(identity["allocated_bytes"])
    selected = _select_candidates(
        all_candidates,
        free_bytes=free_bytes,
        target_free_bytes=target_free_bytes,
        max_tier=max(int(max_tier), 1),
        max_delete_bytes=max_delete_bytes,
    )[:max_files]
    verification_budget.check()
    selected_bytes = sum(_safe_int(row.get("reclaimable_bytes"), 0) for row in selected)
    projected_free_bytes = int(free_bytes + selected_bytes)
    cleanup_needed = free_bytes < target_free_bytes
    disk_after = dict(disk_before)
    apply_result = {
        "applied": False,
        "deleted_files": 0,
        "deleted_bytes": 0,
        "deleted_gb": 0.0,
        "offloaded_files": 0,
        "offloaded_bytes": 0,
        "offloaded_gb": 0.0,
        "reclaimed_bytes": 0,
        "reclaimed_gb": 0.0,
        "errors": [],
        "skipped_files": 0,
        "skipped_rows": [],
        "deleted_rows": [],
        "offloaded_rows": [],
    }
    if apply and selected:
        selected_paths: set[str] = set()
        apply_rounds: list[dict[str, Any]] = []
        selected_round = list(selected)
        estimated_attempted_bytes = 0
        while selected_round:
            apply_rounds.append(_apply_selected(selected_round, project_root=project_root, budget=verification_budget,
                                               source_root=external_root))
            for row in selected_round:
                selected_paths.add(str(row.get("path") or ""))
                estimated_attempted_bytes += _safe_int(row.get("reclaimable_bytes"), 0)
            disk_after = _disk_snapshot(external_root)
            actual_free_bytes = _safe_int(disk_after.get("free_bytes"), free_bytes)
            if (actual_free_bytes >= target_free_bytes or estimated_attempted_bytes >= max_delete_bytes
                    or len(selected_paths) >= max_files):
                break
            remaining_delete_candidates = [
                row
                for row in all_candidates
                if str(row.get("path") or "") not in selected_paths
                and str(row.get("action") or "delete") == "delete"
            ]
            remaining_budget = max(max_delete_bytes - estimated_attempted_bytes, 0)
            if not remaining_delete_candidates or remaining_budget <= 0:
                break
            selected_round = _select_candidates(
                remaining_delete_candidates,
                free_bytes=actual_free_bytes,
                target_free_bytes=target_free_bytes,
                max_tier=max(int(max_tier), 1),
                max_delete_bytes=remaining_budget,
            )[:max_files - len(selected_paths)]
            selected.extend(selected_round)
        apply_result = {"applied": True, **_merge_apply_results(apply_rounds)}
        actual_reclaimed_bytes = max(_safe_int(disk_after.get("free_bytes"), free_bytes) - free_bytes, 0)
        apply_result["actual_reclaimed_bytes"] = int(actual_reclaimed_bytes)
        apply_result["actual_reclaimed_gb"] = _gb(actual_reclaimed_bytes)
        apply_result["apply_rounds"] = len(apply_rounds)
        selected_bytes = sum(_safe_int(row.get("reclaimable_bytes"), 0) for row in selected)
        projected_free_bytes = int(free_bytes + selected_bytes)
    elif apply:
        apply_result["applied"] = True
        apply_result["actual_reclaimed_bytes"] = 0
        apply_result["actual_reclaimed_gb"] = 0.0
        apply_result["apply_rounds"] = 0

    actual_free_bytes = _safe_int(disk_after.get("free_bytes"), projected_free_bytes if not apply else free_bytes)
    comparison_free_bytes = actual_free_bytes
    still_needed_bytes = max(target_free_bytes - comparison_free_bytes, 0)
    if comparison_free_bytes >= target_free_bytes:
        status = "ready"
    elif selected:
        status = "degraded"
    else:
        status = "blocked" if cleanup_needed else "ready"

    selected_top = _top_rows(selected, limit=30)
    candidates_summary = _summarize_candidates(all_candidates)
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": status == "ready",
        "overall_status": status,
        "assessment_complete": True,
        "cleanup_pass_complete": not apply_result["errors"] and not apply_result["skipped_files"],
        "live_execution_authority": False,
        "apply_requested": bool(apply),
        "bot_logs_root": str(external_root),
        "target_free_gb": round(float(target_free_gb), 3),
        "max_tier": int(max_tier),
        "guardrails": {
            "tier_1": "closed-date logs/ raw/gzip pairs require full SHA-256 and length verification, idle single-link stable files, storage ownership and a durable pre-release receipt; SQL payload and unknown roots retain their retirement owners",
            "tier_2": "advisory only; data-retention owns manifest/hash/protected-evidence expiry and verified offload owners preserve conflict/quarantine copies",
            "tier_3": "recommend SQL compaction or offload; do not delete stateful SQLite files here",
            "fallback_quarantine_root": str(fallback_quarantine_root),
            "internal_quarantine_min_free_gb": round(float(DEFAULT_INTERNAL_QUARANTINE_MIN_FREE_GB), 3),
            "protect_current_day": True,
            "min_age_hours": float(min_age_hours),
            "prefix_verify_bytes": int(prefix_verify_bytes),
            "prefix_option_deprecated": True,
            "max_files": max_files,
            "preview_is_deletion_proof": False,
        },
        "disk_before": disk_before,
        "disk_after": disk_after,
        "cleanup_needed": bool(cleanup_needed),
        "selected_count": len(selected),
        "selected_reclaimable_bytes": int(selected_bytes),
        "selected_reclaimable_gb": _gb(selected_bytes),
        "projected_free_gb": _gb(projected_free_bytes),
        "remaining_to_target_gb": _gb(still_needed_bytes),
        "candidate_summary": candidates_summary,
        "deep_cold_layer": {
            "ready": bool(deep_cold_layer.get("ok", False)),
            "managed_gb": _safe_float(deep_cold_summary.get("managed_gb"), 0.0),
            "retention_locked_gb": _safe_float(deep_cold_summary.get("retention_locked_gb"), 0.0),
            "manifest_path": str(deep_cold_layer.get("manifest_path") or ""),
            "policy": "manifest-index retention-locked evidence; data-retention owns actual expiry and this duplicate lane cannot purge staged files",
        },
            "retention_intelligence_v2": {
            "ready": bool(retention_v2.get("ok", False)),
            "overall_status": str(retention_v2.get("overall_status") or ""),
            "overall_grade": str(retention_report.get("overall_grade") or ""),
            "overall_score": _safe_float(retention_report.get("overall_score"), 0.0),
            "registry_path": str(((retention_v2.get("retention_class_registry") or {}).get("registry_path")) or ""),
            "policy": "value-based retention report card guides cleanup before broad deletes",
        },
        "corrupt_sqlite_quarantine": {
            "candidate_count": len(corrupt_rows),
            "eligible_count": sum(1 for row in corrupt_rows if bool(row.get("eligible", False))),
            "candidate_gb": _gb(sum(_safe_int(row.get("reclaimable_bytes"), 0) for row in corrupt_rows)),
            "eligible_gb": _gb(
                sum(_safe_int(row.get("reclaimable_bytes"), 0) for row in corrupt_rows if bool(row.get("eligible", False)))
            ),
            "policy": "preserve corrupt SQLite copies for a verified offload owner; active SQLite siblings remain untouched",
        },
        "selected_candidates": selected_top,
        "top_candidates": _top_rows(all_candidates, limit=30),
        "apply_result": apply_result,
        "intelligence_layer": {
            "decision": (
                "target_reached" if comparison_free_bytes >= target_free_bytes and apply
                else "ready_to_apply_selected_tiers" if comparison_free_bytes >= target_free_bytes
                else "run_next_tier_or_compact_stateful_sql" if still_needed_bytes > 0 and int(max_tier) < 2
                else "manual_review_required"
            ),
            "pressure_level": (
                "critical" if _safe_float(disk_before.get("capacity_pct"), 0.0) >= 98.0
                else "elevated" if _safe_float(disk_before.get("capacity_pct"), 0.0) >= 90.0
                else "normal"
            ),
            "self_updates": [
                "history rows record applied/deleted bytes so future cleanup can measure which tier actually helped",
                "current-day raw JSONL protection prevents the cleanup layer from racing active writers",
                "external failback conflict and corrupt SQLite copies remain advisory until a verified offload owner handles them",
                "tier selection stops as soon as the target free-space floor is projected or achieved",
            ],
            "next_actions": ordered_unique(
                [
                    "refresh storage-tier-policy and storage-quota-guard after cleanup",
                    "refresh retention-intelligence-v2 so cleanup keeps value-based retention context current"
                    if not bool(retention_v2.get("ok", False)) else "",
                    "refresh deep-cold-storage-layer when stale-stage archives are retained but not deletion-eligible"
                    if len(stale_rows) > 0 else "",
                    "tier 2 is advisory; use the manifest retention or verified offload owner for remaining capacity work",
                    "keep autosync disabled or space-gated until BOT_LOGS has enough free space"
                    if len(fallback_rows) > 0 else "",
                    "request verified offload of corrupt SQLite copies without deleting active databases"
                    if len(corrupt_rows) > 0 and still_needed_bytes > 0 else "",
                    "checkpoint and compact jsonl_link.sqlite3 separately; it is stateful and intentionally outside this delete lane"
                    if still_needed_bytes > 0 else "",
                ]
            ),
        },
    }
    try:
        verification_budget.check()
    except RuntimeError as exc:
        # Preserve actual releases even if the final assessment outlives its budget.
        payload.update(assessment_complete=False, cleanup_pass_complete=False,
                       overall_status="deferred", ok=False, reason=str(exc))
    write_payload(out_path, payload)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp_utc": payload["timestamp_utc"],
                    "apply_requested": bool(apply),
                    "overall_status": payload["overall_status"],
                    "assessment_complete": payload["assessment_complete"],
                    "cleanup_pass_complete": payload["cleanup_pass_complete"],
                    "bot_logs_root": str(external_root),
                    "free_gb_before": disk_before.get("free_gb"),
                    "free_gb_after": disk_after.get("free_gb"),
                    "selected_reclaimable_gb": payload["selected_reclaimable_gb"],
                    "deleted_gb": apply_result.get("deleted_gb", 0.0),
                    "offloaded_gb": apply_result.get("offloaded_gb", 0.0),
                    "reclaimed_gb": apply_result.get("reclaimed_gb", 0.0),
                    "selected_count": len(selected),
                    "deleted_files": apply_result.get("deleted_files", 0),
                    "offloaded_files": apply_result.get("offloaded_files", 0),
                    "max_tier": int(max_tier),
                },
                ensure_ascii=True,
            )
            + "\n"
        )
    return payload


def build_payload(project_root: Path = PROJECT_ROOT, **kwargs) -> dict[str, Any]:
    seconds = float(kwargs.pop("seconds", 90))
    max_verify_gb = float(kwargs.pop("max_verify_gb", 1))
    max_files = int(kwargs.get("max_files", 4))
    if (not math.isfinite(seconds) or not 1 <= seconds <= 300 or
            not math.isfinite(max_verify_gb) or not 0 < max_verify_gb <= 4 or
            not 1 <= max_files <= 32):
        raise ValueError("invalid_cleanup_budget")
    out = kwargs.get("out_path", DEFAULT_OUT_PATH)
    lock = None
    try:
        root = kwargs.get("bot_logs_root") or _verified_external_root()
        kwargs["bot_logs_root"] = root
        verified.safety.allowed(root)
        verified.safety.allowed(project_root)
        verified.safety.allowed(out, missing=True)
        verified.safety.allowed(kwargs.get("history_path", DEFAULT_HISTORY_PATH), missing=True)
        verified.safety.allowed(kwargs.get("fallback_quarantine_root", DEFAULT_FALLBACK_QUARANTINE_ROOT), missing=True)
        guard = None
        if kwargs.get("apply", False):
            lock_path = verified.safety.allowed(project_root / "governance/locks/storage_maintenance.lock", missing=True)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = os.fdopen(os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), "a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise verified.safety.Deferred("storage_maintenance_lock_busy") from exc
            verified.safety.background_policy()
            guard = verified.safety.Guard(project_root, seconds)
            held = os.fstat(lock.fileno())
            guard.lock_anchor = (lock_path, (held.st_dev, held.st_ino))
            guard.check()
        budget = verified.Budget(seconds, int(max_verify_gb * 1024**3), guard)
        paths = verified.inventory(root, seconds=min(seconds, 15))
        budget.check()
        return _build_payload(project_root, inventory_paths=paths, verification_budget=budget, **kwargs)
    except (OSError, RuntimeError) as exc:
        payload = {"timestamp_utc": iso_now(), "overall_status": "deferred", "ok": False,
                   "apply_requested": bool(kwargs.get("apply", False)), "reason": str(exc),
                   "assessment_complete": False, "cleanup_pass_complete": False,
                   "live_execution_authority": False}
        verified.safety.allowed(out, missing=True)
        write_payload(out, payload)
        return payload
    finally:
        if lock is not None:
            lock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiered BOT_LOGS cleanup with guarded cleanup intelligence.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--bot-logs-root", default="")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--history-file", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--target-free-gb", type=float, default=DEFAULT_TARGET_FREE_GB)
    parser.add_argument("--max-tier", type=int, default=1)
    parser.add_argument("--max-delete-gb", type=float, default=0.0)
    parser.add_argument("--max-files", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--max-verify-gb", type=float, default=1)
    parser.add_argument("--min-age-hours", type=float, default=DEFAULT_MIN_AGE_HOURS)
    parser.add_argument("--protect-current-day", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefix-verify-bytes", type=int, default=DEFAULT_PREFIX_VERIFY_BYTES)
    parser.add_argument("--fallback-quarantine-root", default=str(DEFAULT_FALLBACK_QUARANTINE_ROOT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).absolute()
    bot_logs_root = Path(args.bot_logs_root).expanduser() if str(args.bot_logs_root or "").strip() else None
    payload = build_payload(
        project_root,
        bot_logs_root=bot_logs_root,
        apply=bool(args.apply),
        target_free_gb=float(args.target_free_gb),
        max_tier=max(int(args.max_tier), 1),
        max_delete_gb=float(args.max_delete_gb),
        max_files=args.max_files,
        seconds=args.seconds,
        max_verify_gb=args.max_verify_gb,
        min_age_hours=float(args.min_age_hours),
        protect_current_day=bool(args.protect_current_day),
        prefix_verify_bytes=max(int(args.prefix_verify_bytes), 1),
        fallback_quarantine_root=Path(args.fallback_quarantine_root).expanduser(),
        out_path=Path(args.out_file).expanduser(),
        history_path=Path(args.history_file).expanduser(),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "bot_logs_cleanup_intelligence "
            f"overall_status={payload.get('overall_status', '')} "
            f"selected_gb={payload.get('selected_reclaimable_gb', 0)} "
            f"free_after_gb={((payload.get('disk_after') or {}).get('free_gb', 0))}"
        )
    return 0 if payload.get("cleanup_pass_complete") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
