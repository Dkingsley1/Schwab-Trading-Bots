#!/usr/bin/env python3
"""Prepare a primary data SSD without claiming, formatting, or writing to it."""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.storage_router import inspect_storage_path

GIB = 1024**3
REPORT_LIMIT = 2 * 1024**2


def _inventory(project_root: Path, now: datetime) -> dict:
    path = project_root / "governance/health/storage_tier_policy_latest.json"
    result = {
        "state": "unavailable",
        "source": str(path),
        "scope": "existing_tier_report_not_a_complete_or_verified_migration_manifest",
        "reclaimable_bytes": None,
    }
    route = inspect_storage_path(path, boundary_root=project_root, allow_external=False)
    if route.get("status") != "present":
        return result
    try:
        with path.open("rb") as source:
            raw = source.read(REPORT_LIMIT + 1)
        if len(raw) > REPORT_LIMIT:
            return {**result, "state": "report_exceeds_read_budget"}
        payload = json.loads(raw)
        timestamp = datetime.fromisoformat(
            payload["timestamp_utc"].replace("Z", "+00:00")
        )
        if timestamp.tzinfo is None:
            raise ValueError("timezone_required")
        age = (now - timestamp).total_seconds()
        families = payload["by_family"]
        if not isinstance(families, dict):
            raise ValueError("families_required")
        rows = []
        for name, row in families.items():
            count = row["bytes"]
            if type(count) is not int or count < 0:
                raise ValueError("invalid_byte_count")
            rows.append({"family": name, "reported_bytes": count})
        return {
            **result,
            "state": "fresh" if 0 <= age <= 3600 else "stale_or_future",
            "source_timestamp_utc": timestamp.isoformat(),
            "age_seconds": round(age, 1),
            "reported_bytes": sum(row["reported_bytes"] for row in rows),
            "largest_families": sorted(rows, key=lambda row: -row["reported_bytes"])[
                :12
            ],
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {**result, "state": "invalid_or_unreadable_report"}


def _disk_info(mount: Path) -> dict:
    proc = subprocess.run(
        ["/usr/sbin/diskutil", "info", "-plist", str(mount)],
        capture_output=True,
        check=False,
        timeout=10,
    )
    if proc.returncode != 0 or len(proc.stdout) > REPORT_LIMIT:
        raise ValueError("disk_info_unavailable")
    info = plistlib.loads(proc.stdout)
    if not isinstance(info, dict):
        raise ValueError("disk_info_invalid")
    return info


def _uuid(value: str) -> str:
    return str(UUID(value)).upper()


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    mount: str = "",
    expected_uuid: str = "",
    now: datetime | None = None,
) -> dict:
    now = now or datetime.now(timezone.utc)
    result = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "state": "awaiting_drive_selection",
        "preflight_ok": False,
        "activation_ready": False,
        "role": "primary_platform_data_with_cold_archives",
        "suggested_volume_name": "BOT_DATA",
        "planned_external_families": [
            "data/sql_link_shards",
            "data/analytics_and_training_datasets",
            "logs",
            "decisions",
            "decision_explanations",
            "exports",
            "models",
            "closed_compressed_history",
            "eligible_cold_archives",
        ],
        "keep_internal": [
            "source_and_git",
            "python_environments",
            "credentials_and_keychain",
            "active_control_flags_and_small_health_records",
            "bounded_recovery_buffer",
        ],
        "migration_order": [
            "identify_and_operator_confirm_new_ssd",
            "verify_apfs_capacity_and_durable_io",
            "build_fresh_per_file_manifest_and_check_capacity",
            "copy_and_verify_closed_history_without_deleting_sources",
            "quiesce_writers_and_verify_database_backup_restore",
            "review_and_switch_exact_data_routes_with_rollback",
            "verify_reads_writes_ingestion_and_reconnect_behavior",
            "release_verified_duplicates_through_existing_owners",
        ],
        "required_before_activation": [
            "durable_write_read_fsync_and_rename_probe",
            "representative_sqlite_integrity_restore_and_latency_test",
            "stable_connection_and_disconnect_recovery_test",
            "reviewed_exact_manifest_and_destination_capacity_budget",
            "operator_approved_route_cutover_and_retained_rollback",
        ],
        "authority": {
            "format": False,
            "mkdir_on_drive": False,
            "copy": False,
            "delete": False,
            "route_change": False,
            "restart": False,
            "automatic_drive_adoption": False,
            "live_order": False,
        },
        "existing_bot_logs_target_unchanged": True,
        "video_volume_untouched": True,
        "physical_backup_independence_verified": False,
        "inventory": _inventory(project_root, now),
        "blockers": ["drive_not_selected"],
    }
    if not mount:
        return result
    path = Path(mount)
    # Reject protected/old volumes and ambiguous paths before any target metadata I/O.
    if (
        not path.is_absolute()
        or len(path.parts) != 3
        or path.parts[1] != "Volumes"
        or path.name.casefold() in {"video", "bot_logs", ".", ".."}
    ):
        result.update(state="blocked", blockers=["new_direct_volume_path_required"])
        return result
    route = inspect_storage_path(path)
    if (
        route.get("status") != "present"
        or route.get("symlinks")
        or route.get("kind") != "directory"
    ):
        result.update(
            state="blocked", blockers=["selected_volume_missing_or_unsafe_route"]
        )
        return result
    try:
        if not path.is_mount():
            raise ValueError("selected_path_is_not_mounted_volume")
        before = path.stat()
        info = _disk_info(path)
        blockers = []
        if info.get("Internal") is not False:
            blockers.append("external_device_not_confirmed")
        if info.get("MountPoint") != str(path):
            blockers.append("mount_identity_mismatch")
        if str(info.get("FilesystemType", "")).lower() != "apfs":
            blockers.append("apfs_required_for_primary_data_plan")
        if info.get("WritableVolume") is not True:
            blockers.append("writable_volume_not_confirmed")
        observed_uuid = _uuid(str(info.get("VolumeUUID", "")))
        if not expected_uuid:
            blockers.append("operator_uuid_confirmation_required")
        elif _uuid(expected_uuid) != observed_uuid:
            blockers.append("volume_uuid_mismatch")
        existing_uuid = os.getenv("BOT_LOGS_EXTERNAL_VOLUME_UUID", "").strip()
        if existing_uuid and _uuid(existing_uuid) == observed_uuid:
            blockers.append("existing_bot_logs_volume_is_not_the_new_drive")
        usage = shutil.disk_usage(path)
        reserve = max(150 * GIB, usage.total // 10)
        if usage.total < 1_500_000_000_000:
            blockers.append("capacity_smaller_than_expected_2tb_drive")
        if usage.free < reserve + GIB:
            blockers.append("insufficient_destination_headroom")
        after_route = inspect_storage_path(path)
        if after_route.get("status") != "present" or after_route.get("symlinks"):
            raise ValueError("mount_changed_during_probe")
        after = path.stat()
        if (before.st_dev, before.st_ino) != (
            after.st_dev,
            after.st_ino,
        ) or not path.is_mount():
            raise ValueError("mount_changed_during_probe")
        result.update(
            state="metadata_preflight_passed" if not blockers else "blocked",
            preflight_ok=not blockers,
            blockers=blockers,
            volume={
                "mount": str(path),
                "uuid": observed_uuid,
                "filesystem": info.get("FilesystemType"),
                "total_bytes": usage.total,
                "free_bytes": usage.free,
                "planned_reserve_bytes": reserve,
                "planned_root": str(path / "schwab_trading_bot"),
                "planned_cold_root": str(path / "schwab_trading_bot/cold_archive"),
                "write_test_performed": False,
                "performance_verified": False,
            },
        )
    except (
        OSError,
        ValueError,
        TypeError,
        subprocess.TimeoutExpired,
        plistlib.InvalidFileException,
    ):
        result.update(
            state="blocked",
            preflight_ok=False,
            blockers=["volume_probe_failed_or_identity_invalid"],
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mount", default="", help="Explicit /Volumes/NAME; never auto-discovered."
    )
    parser.add_argument(
        "--expected-uuid", default="", help="Operator-reviewed volume UUID."
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = build_payload(mount=args.mount, expected_uuid=args.expected_uuid)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"external_drive_preflight state={payload['state']} activation_ready=false"
        )
        print(
            "Plan: primary platform data plus eligible cold archives; no routes changed."
        )
        print("Blockers: " + ", ".join(payload["blockers"]))
        print("Run with --json for the staged migration checklist.")
    return 2 if payload["state"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
