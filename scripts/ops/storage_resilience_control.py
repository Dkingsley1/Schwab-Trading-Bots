#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.sqlite_runtime import sqlite_integrity_summary
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.sqlite_runtime import sqlite_integrity_summary

DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "storage_resilience_control_latest.json"
CHECKSUM_PATH = PROJECT_ROOT / "governance" / "storage" / "checksum_scrub_latest.json"


def _local_fallback_equivalent(path: Path, *, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    local_fallback_root = project_root / "local_fallback_storage"
    try:
        rel = candidate.relative_to(project_root)
    except ValueError:
        return candidate
    if rel.parts and rel.parts[0] == local_fallback_root.name:
        return candidate
    return local_fallback_root / rel


def _is_broken_symlink(path: Path) -> bool:
    try:
        return path.is_symlink() and not path.exists()
    except OSError:
        return False


def _routed_or_local_fallback_path(path: Path, *, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if _is_broken_symlink(candidate):
        return _local_fallback_equivalent(candidate, project_root=project_root)
    return candidate


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_iso(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _integrity_summary(
    path: Path,
    *,
    project_root: Path,
    fast: bool,
    max_quick_check_db_gb: float,
) -> dict[str, Any]:
    if path.exists() and fast:
        size_bytes = int(path.stat().st_size)
        size_gb = float(size_bytes) / float(1024**3)
        if max_quick_check_db_gb <= 0.0 or size_gb > max_quick_check_db_gb:
            wal_path = Path(f"{path}-wal")
            shm_path = Path(f"{path}-shm")
            return {
                "db_path": str(path),
                "present": True,
                "ok": True,
                "quick_check": "skipped_fast_mode_large_db",
                "check_mode": "fast_skip_large_db",
                "db_size_bytes": size_bytes,
                "wal_size_bytes": int(wal_path.stat().st_size) if wal_path.exists() else 0,
                "shm_size_bytes": int(shm_path.stat().st_size) if shm_path.exists() else 0,
            }
    payload = sqlite_integrity_summary(path, project_root=project_root)
    payload["check_mode"] = "full"
    return payload


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    fast: bool = False,
    max_quick_check_db_gb: float = 4.0,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    mount_guard = _load_json(health_root / "storage_mount_guard_latest.json")
    failback = _load_json(health_root / "storage_failback_sync_latest.json")
    split_brain = _load_json(health_root / "storage_split_brain_reconciler_latest.json")
    retention = _load_json(health_root / "storage_retention_unison_latest.json")
    state_snapshot = _load_json(project_root / "exports" / "state_snapshot_drills" / "latest.json")
    backup_restore_event_files = sorted((project_root / "governance" / "watchdog").glob("backup_restore_events.jsonl*"))
    checksum_targets = [
        health_root / "storage_mount_guard_latest.json",
        health_root / "storage_failback_sync_latest.json",
        health_root / "storage_split_brain_reconciler_latest.json",
        health_root / "daily_auto_verify_latest.json",
    ]
    sqlite_targets = [
        _routed_or_local_fallback_path(project_root / "data" / "jsonl_link.sqlite3", project_root=project_root),
        project_root / "governance" / "ops_data_plane.sqlite3",
        _routed_or_local_fallback_path(project_root / "data" / "bot_channel_queue.sqlite3", project_root=project_root),
        _routed_or_local_fallback_path(project_root / "data" / "snapshot_context.sqlite3", project_root=project_root),
    ]
    checksum_rows = []
    for path in checksum_targets:
        if path.exists() and path.is_file():
            checksum_rows.append({"path": str(path), "sha256": _sha(path)})

    CHECKSUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    checksum_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "targets": checksum_rows,
    }
    CHECKSUM_PATH.write_text(json.dumps(checksum_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    database_integrity_checks = [
        _integrity_summary(
            path,
            project_root=project_root,
            fast=fast,
            max_quick_check_db_gb=max_quick_check_db_gb,
        )
        for path in sqlite_targets
    ]
    wal_health_checks = [
        {
            "db_path": row.get("db_path", ""),
            "wal_size_bytes": int(row.get("wal_size_bytes", 0) or 0),
            "db_size_bytes": int(row.get("db_size_bytes", 0) or 0),
            "wal_pressure_norm": round(
                float(row.get("wal_size_bytes", 0) or 0) / max(float(row.get("db_size_bytes", 1) or 1), 1.0),
                6,
            ),
        }
        for row in database_integrity_checks
        if bool(row.get("present", False))
    ]

    snapshot_ts = _parse_iso(state_snapshot.get("timestamp_utc") or state_snapshot.get("generated_utc"))
    snapshot_age_hours = max((datetime.now(timezone.utc) - snapshot_ts).total_seconds() / 3600.0, 0.0) if snapshot_ts else None
    warm_standby_ready = (project_root / "local_fallback_storage").exists()
    retention_contract = retention.get("continuous_run_contract") if isinstance(retention.get("continuous_run_contract"), dict) else {}
    retention_disk = retention.get("disk") if isinstance(retention.get("disk"), dict) else {}
    external_disk = retention_disk.get("external") if isinstance(retention_disk.get("external"), dict) else {}
    external_archive_disk_ready = bool(
        external_disk.get("exists", False)
        and not external_disk.get("protected", False)
        and _safe_float(external_disk.get("free_gb"), 0.0) > 0.0
    )
    archive_standby_ready = bool(
        external_archive_disk_ready
        or (
            retention_contract.get("cold_archive_spillover_available", False)
            and _safe_float(retention_contract.get("current_external_free_gb"), 0.0) > 0.0
            and _safe_float(retention_contract.get("cold_archive_capacity_shortfall_gb"), 0.0) <= 0.0
        )
    )
    local_hot_policy_ready = bool(
        mount_guard.get("external_required_for_hot_path") is False
        and mount_guard.get("hot_storage_available", False)
        and mount_guard.get("probe_skipped_external_io", False)
    )
    active_route_ready = bool(mount_guard.get("external_available", False) or local_hot_policy_ready)
    dual_root_ready = bool(
        warm_standby_ready
        and (mount_guard.get("external_available", False) or archive_standby_ready)
    )
    restore_drill_fresh = bool(
        state_snapshot.get("ok", False)
        and snapshot_age_hours is not None
        and snapshot_age_hours <= 168.0
    )
    unresolved_split_brain = int(((split_brain.get("summary") or {}).get("unresolved_conflicts", 0) or 0))
    reliability_score = sum(
        [
            25 if dual_root_ready else 0,
            20 if warm_standby_ready else 0,
            20 if restore_drill_fresh else 0,
            20 if unresolved_split_brain == 0 else 0,
            15 if bool(checksum_rows) else 0,
        ]
    )
    database_integrity_ready = bool(
        not database_integrity_checks
        or all(bool(row.get("ok", False)) for row in database_integrity_checks if bool(row.get("present", False)))
    )
    if not database_integrity_ready:
        reliability_score = max(reliability_score - 20, 0)
    ready = bool(
        reliability_score >= 75
        and active_route_ready
        and dual_root_ready
        and warm_standby_ready
        and restore_drill_fresh
        and unresolved_split_brain == 0
        and database_integrity_ready
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": ready,
        "overall_status": "ready" if ready else "needs_work",
        "resilience_score": reliability_score,
        "dual_root_ready": dual_root_ready,
        "warm_standby_ready": warm_standby_ready,
        "archive_standby_ready": archive_standby_ready,
        "active_route_ready": active_route_ready,
        "local_hot_policy_ready": local_hot_policy_ready,
        "restore_drill_fresh": restore_drill_fresh,
        "snapshot_age_hours": round(float(snapshot_age_hours), 3) if snapshot_age_hours is not None else None,
        "unresolved_split_brain_conflicts": unresolved_split_brain,
        "backup_restore_event_files": len(backup_restore_event_files),
        "checksum_scrub": checksum_payload,
        "integrity_mode": ("fast" if fast else "full"),
        "max_quick_check_db_gb": round(float(max_quick_check_db_gb), 3),
        "database_integrity_checks": database_integrity_checks,
        "wal_health_checks": wal_health_checks,
        "route_topology": {
            "policy": "local_hot_external_archive" if local_hot_policy_ready else "external_hot_with_local_standby",
            "active_route_ready": active_route_ready,
            "local_hot_policy_ready": local_hot_policy_ready,
            "archive_standby_ready": archive_standby_ready,
            "external_archive_disk_ready": external_archive_disk_ready,
            "dual_root_ready": dual_root_ready,
            "external_required_for_hot_path": bool(mount_guard.get("external_required_for_hot_path", True)),
        },
        "mount_guard": mount_guard,
        "failback": failback,
        "top_actions": [
            "keep local_fallback_storage warm as the standing standby root",
            "treat checksum_scrub_latest.json as the baseline for post-incident integrity review",
            "treat quick_check and WAL pressure on the primary SQLite files as first-class storage health signals",
            "run restore drills on a schedule tight enough to keep snapshot_age_hours below one week",
        ],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish storage resilience controls for BOT_LOGS failover, checksums, and restore freshness.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max-quick-check-db-gb", type=float, default=4.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        fast=bool(args.fast),
        max_quick_check_db_gb=max(_safe_float(args.max_quick_check_db_gb, 4.0), 0.0),
    )
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "storage_resilience_control "
            f"overall_status={payload.get('overall_status', '')} "
            f"resilience_score={int(payload.get('resilience_score', 0) or 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
