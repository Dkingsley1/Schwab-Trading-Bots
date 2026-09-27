"""Explicit receipt-bound retirement of independently backed-up SQLite standbys."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import subprocess
import time
import uuid

from core import sqlite_primary_storage as primary
from core.write_path_recovery import durable_json


def _owned_json(root, path):
    from core.storage_router import inspect_storage_path

    state = inspect_storage_path(
        path, boundary_root=root / "governance/storage_recovery", allow_external=False
    )
    if state["status"] != "present" or state["symlinks"]:
        raise ValueError("retirement_receipt_not_owned")
    with path.open("rb") as stream:
        data = stream.read(4 * 1024**2 + 1)
    if len(data) > 4 * 1024**2:
        raise ValueError("retirement_receipt_oversized")
    return json.loads(data)


def _backup_volume(row, target, source):
    mount = Path(row["backup_mount"])
    backup = Path(row["backup"])
    # Reject protected/unknown namespaces before any volume inspection.
    if mount.parent != Path("/Volumes") or mount.name in {"VIDEO", "", ".", ".."}:
        raise ValueError("retirement_backup_mount_invalid")
    expected = mount / "schwab_trading_bot/cold_archive/sqlite_primary_recovery"
    if backup.parent != expected or backup.name != source.name + ".zst":
        raise ValueError("retirement_backup_path_invalid")
    primary._physical(expected)
    info = plistlib.loads(
        subprocess.check_output(
            ["/usr/sbin/diskutil", "info", "-plist", str(mount)], timeout=20
        )
    )
    if (
        info.get("VolumeUUID") != row["backup_volume_uuid"]
        or info.get("MountPoint") != str(mount)
        or info.get("Internal") is not False
        or not os.path.ismount(mount)
    ):
        raise ValueError("retirement_backup_volume_mismatch")
    # Require different physical stores, not merely different mounted partitions.
    active = plistlib.loads(
        subprocess.check_output(
            ["/usr/sbin/diskutil", "info", "-plist", str(target.parent)], timeout=20
        )
    )
    stores = active.get("APFSPhysicalStores", [])
    active_disks = {
        re.sub(r"s\d+$", "", str(item.get("APFSPhysicalStore", ""))) for item in stores
    }
    backup_stores = info.get("APFSPhysicalStores", [])
    backup_disks = (
        {
            re.sub(r"s\d+$", "", str(item.get("APFSPhysicalStore", "")))
            for item in backup_stores
        }
        if backup_stores
        else {info.get("ParentWholeDisk", "")}
    )
    if (
        not active_disks
        or "" in active_disks
        or not backup_disks
        or "" in backup_disks
        or bool(backup_disks & active_disks)
        or mount.stat().st_dev in {target.stat().st_dev, source.stat().st_dev}
    ):
        raise ValueError("retirement_backup_independence_unproven")
    return backup


def _restored_hash(path, expected_bytes, deadline):
    import zstandard

    digest, count = hashlib.sha256(), 0
    with path.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(
        raw
    ) as stream:
        while chunk := stream.read(8 * 1024**2):
            count += len(chunk)
            if count > expected_bytes or time.monotonic() > deadline:
                raise ValueError("retirement_restore_budget_exceeded")
            digest.update(chunk)
    if count != expected_bytes:
        raise ValueError("retirement_restore_size_mismatch")
    return digest.hexdigest()


def _idle(path):
    result = subprocess.run(["lsof", str(path)], capture_output=True, timeout=30)
    if result.returncode != 1 or result.stdout or result.stderr:
        raise RuntimeError("retirement_source_handles_present_or_unknown")
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(path) + suffix)
        if os.path.lexists(sidecar) and primary._identity(sidecar)[2]:
            raise RuntimeError("retirement_source_journal_present")


def reviewed_source_identity(root, relative, prepared):
    if relative != "jsonl_link.sqlite3":
        return prepared[relative]["source_identity"], []
    reconciled = _owned_json(root, root / "governance/storage_recovery/sqlite_primary_recovery_reconciliation.json")
    if (reconciled.get("purpose") != "sqlite_primary_route_recovery_reconciliation"
            or reconciled.get("all_standby_payloads_preserved") is not True
            or reconciled.get("primary_cursors_not_regressed") is not True):
        raise ValueError("central_retirement_prior_reconciliation_missing")
    reports = [r for r in reconciled.get("tables", []) if r.get("table") == "one_numbers_snapshots"]
    if len(reports) != 1:
        raise ValueError("central_retirement_report_mapping_missing")
    return reconciled["source_identity"], reports[0]["report_id_mapping"]


def _central_idle(root, source, target, relative):
    active = target / "data" / relative
    shared_memory = Path(str(active) + "-shm")
    if os.path.lexists(shared_memory) and primary._identity(shared_memory)[2]:
        from core.sqlite_primary_recovery import _quiet

        # Reuse the recovery owner's SQLite-only cleanup, including its second
        # quiet point and unchanged database identity checks. Never unlink SHM.
        primary._hold(root)
        _quiet(root / "local_fallback_storage/data", target,
               [{"relative": relative}], {relative: primary._identity(source)})
    _idle(active)


def retire_standbys(root: Path, receipt_path: Path, *, apply: bool = False,
                    allow_central_databases: bool = False) -> dict:
    root = Path(root).absolute()
    if not primary.enabled(root):
        raise RuntimeError("retirement_requires_sqlite_primary")
    primary._hold(root)
    primary.require_ready(root)
    target = primary._validated_target()
    directory = root / "governance/storage_recovery"
    receipt = _owned_json(root, Path(receipt_path))
    cutover = _owned_json(root, directory / "sqlite_primary_cutover_reviewed.json")
    handoff = _owned_json(root, directory / "sqlite_primary_handoff_result.json")
    io = _owned_json(root, directory / "sqlite_primary_io_verification.json")
    if (
        receipt.get("purpose") != "sqlite_primary_independent_backups"
        or cutover.get("target_root") != str(target)
        or cutover.get("source_root") != str(root / "local_fallback_storage/data")
        or cutover.get("volume_uuid") != primary._getenv("BOT_LOGS_EXTERNAL_VOLUME_UUID")
        or handoff.get("target_root") != str(target)
        or handoff.get("ok") is not True
        or handoff.get("route_mutation_performed") is not True
        or handoff.get("payload_hashes_verified") != len(cutover.get("files", []))
        or io.get("ok") is not True
        or io.get("queue_write_read_ack_verified") is not True
        or io.get("standby_unchanged") is not True
        or io.get("target_root") != str(target)
        or io.get("volume_uuid") != primary._getenv("BOT_LOGS_EXTERNAL_VOLUME_UUID")
    ):
        raise ValueError("retirement_handoff_or_io_proof_missing")
    rows = receipt.get("files", [])
    if not isinstance(rows, list) or not 1 <= len(rows) <= 2:
        raise ValueError("retirement_bounded_selection_required")
    prepared = {row["relative"]: row for row in cutover["files"]}
    if len({row["relative"] for row in rows}) != len(rows):
        raise ValueError("retirement_duplicate_selection")
    deadline = time.monotonic() + 1800
    primary._physical(root / "governance/locks")
    with (root / "governance/locks/sqlite_primary_route.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verified = []
        for row in rows:
            relative = row["relative"]
            central = allow_central_databases and relative in primary.DATABASES
            if not central and not re.fullmatch(
                r"sql_link_shards/jsonl_link_[A-Za-z0-9_]+\.sqlite3", relative
            ):
                raise ValueError("retirement_only_named_standby_shards")
            source = root / "local_fallback_storage/data" / relative
            primary._physical(source.parent)
            if relative not in prepared:
                raise ValueError("retirement_source_not_in_cutover")
            expected_identity, mapping = (
                reviewed_source_identity(root, relative, prepared) if central
                else (prepared[relative]["source_identity"], [])
            )
            if (
                row.get("source") != str(source)
                or relative not in prepared
                or primary._identity(source) != row["source_identity"]
                or row["source_identity"] != expected_identity
                or row.get("restored_bytes") != row["source_identity"][2]
                or row.get("full_restore_hash_verified") is not True
            ):
                raise ValueError("retirement_source_custody_mismatch")
            _idle(source)
            active = target / "data" / relative
            active_identity = primary._identity(active)
            if central:
                from core.sqlite_standby_reconciliation import verify_records

                _central_idle(root, source, target, relative)
                custody = verify_records(source, active, report_mapping=mapping,
                    guard=lambda: primary._hold(root))
                row = {**row, "central_reconciliation": custody,
                       "reconciled_target_identity": active_identity}
            backup = _backup_volume(row, target, source)
            backup_identity = primary._identity(backup)
            if backup_identity != row["backup_identity"]:
                raise ValueError("retirement_backup_changed")
            if (
                primary._digest(source, deadline) != row["source_sha256"]
                or _restored_hash(backup, row["restored_bytes"], deadline)
                != row["source_sha256"]
            ):
                raise ValueError("retirement_full_hash_mismatch")
            if (
                primary._identity(source) != row["source_identity"]
                or primary._identity(backup) != backup_identity
            ):
                raise ValueError("retirement_changed_during_verification")
            primary._hold(root)
            primary.require_ready(root)
            if central:
                _central_idle(root, source, target, relative)
                if primary._identity(active) != active_identity:
                    raise ValueError("central_retirement_target_changed")
            verified.append(row)
        result = {
            "purpose": "sqlite_primary_standby_retirement",
            "ok": True,
            "apply": apply,
            "verified": verified,
            "retired": [],
            "live_execution_authority": False,
        }
        if not apply:
            return result
        journal = directory / (
            "sqlite_primary_retirement_" + uuid.uuid4().hex + ".json"
        )
        durable_json(root, journal, {**result, "phase": "verified_before_retirement"})
        for row in verified:
            primary._hold(root)
            primary.require_ready(root)
            source = Path(row["source"])
            primary._identity(target / "data" / row["relative"])
            if row.get("central_reconciliation"):
                active = target / "data" / row["relative"]
                _central_idle(root, source, target, row["relative"])
                if primary._identity(active) != row["reconciled_target_identity"]:
                    raise ValueError("central_retirement_target_changed_before_unlink")
            backup = _backup_volume(row, target, source)
            _idle(source)
            if (
                primary._identity(source) != row["source_identity"]
                or primary._identity(backup) != row["backup_identity"]
            ):
                raise ValueError("retirement_changed_before_unlink")
            source.unlink()
            primary._sync_directory(source.parent)
            result["retired"].append(row["relative"])
            durable_json(root, journal, {**result, "phase": "retiring"})
        result["retired_logical_bytes"] = sum(
            row["source_identity"][2] for row in verified
        )
        result["journal"] = str(journal)
        durable_json(root, journal, {**result, "phase": "complete"})
        return result
