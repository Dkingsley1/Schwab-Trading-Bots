"""Measured compact snapshot budgets and fail-closed APFS restore clones."""

from contextlib import closing
import ctypes
import gzip
import hashlib
import math
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import sys
import tempfile
import time

from core.background_work_budget import WorkBudget
from core.storage_router import inspect_storage_path

GIB = 1024**3


def complete_restore_evidence(payload):
    if not isinstance(payload, dict):
        return False
    checked = payload.get("files_checked")
    verified = payload.get("files_restore_verified")
    return bool(
        payload.get("ok") is True
        and type(checked) is int
        and checked > 0
        and type(verified) is int
        and verified == checked
        and payload.get("latest_write_verified") is True
        and payload.get("published_latest_write_verified") is True
        and not payload.get("accepted_metadata_only_large_files", 0)
        and not payload.get("missing_files", [])
    )


def load_policy(root):
    import json

    path = root / "config/state_snapshot_drill_v1.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if (
        data.get("schema_version") != 1
        or data.get("mode") != "compact_sqlite_apfs_clone_gzip"
        or type(data.get("max_snapshot_bytes")) is not int
        or not 1 <= data["max_snapshot_bytes"] <= 64 * GIB
        or type(data.get("operation_seconds")) is not int
        or not 30 <= data["operation_seconds"] <= 1800
        or type(data.get("command_timeout_seconds")) is not int
        or not data["operation_seconds"] < data["command_timeout_seconds"] <= 3600
    ):
        raise ValueError("invalid_state_snapshot_policy")
    allowed(Path(data["out_root"]), missing=True, stage="policy_out_root")
    return data


def policy_arguments(root):
    policy = load_policy(root)
    if not policy:
        return []
    return [
        "--compact-sqlite",
        "--clone-restore",
        "--out-root",
        policy["out_root"],
        "--max-copy-bytes",
        str(policy["max_snapshot_bytes"]),
        "--operation-seconds",
        str(policy["operation_seconds"]),
    ]


class SnapshotRouteError(ValueError):
    def __init__(self, path, route, stage):
        self.diagnostic = {
            "path": str(path),
            "stage": stage,
            "route_status": route["status"],
            "resolved_path": route.get("resolved_path", ""),
        }
        super().__init__(
            f"snapshot_route_unavailable:stage={stage}:path={path}:"
            f"status={route['status']}:resolved={route.get('resolved_path', '')}"
        )


def allowed(path, *, missing=False, stage="snapshot_route", within=None):
    route = inspect_storage_path(path, boundary_root=within)
    if route["status"] not in ({"present", "missing"} if missing else {"present"}):
        raise SnapshotRouteError(path, route, stage)
    return Path(route["resolved_path"])


def existing_parent(path):
    path = allowed(path, missing=True)
    while not path.exists():
        path = path.parent
    return path


def reserve_bytes(project_root, output):
    local = (
        existing_parent(project_root).stat().st_dev
        == existing_parent(output).stat().st_dev
    )
    key = (
        "BOT_LOCAL_STORAGE_TARGET_FREE_GB"
        if local
        else "SNAPSHOT_DRILL_EXTERNAL_RESERVE_GB"
    )
    raw = float(os.environ.get(key, "64"))
    if not math.isfinite(raw) or raw < 64:
        raise ValueError("snapshot_reserve_below_64_gib_or_invalid")
    return int(raw * GIB), (
        "local_configured_reserve" if local else "separate_archive_volume_reserve"
    )


def plan_target(path, max_bytes):
    path = allowed(path)
    before = path.stat()
    result = {
        "path": str(path),
        "source_bytes": before.st_size,
        "source_identity": [before.st_dev, before.st_ino],
        "sqlite": False,
    }
    with path.open("rb") as file:
        is_sqlite = file.read(16) == b"SQLite format 3\x00"
    if is_sqlite:
        for suffix in ("-wal", "-shm"):
            allowed(Path(str(path) + suffix), missing=True)
        with closing(
            sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2)
        ) as db:
            page_size = db.execute("PRAGMA page_size").fetchone()[0]
            pages = db.execute("PRAGMA page_count").fetchone()[0]
            free = db.execute("PRAGMA freelist_count").fetchone()[0]
            occupied = max(pages - free, 0) * page_size
        # This is an enforced output ceiling, not a promise that an estimate is exact.
        cap = min(max_bytes, occupied + 64 * 1024**2)
        result.update(
            sqlite=True,
            occupied_bytes=occupied,
            logical_bytes=pages * page_size,
            freelist_bytes=free * page_size,
            output_limit_bytes=cap,
        )
        if occupied > max_bytes:
            raise ValueError("occupied_database_exceeds_snapshot_cap")
    else:
        if before.st_size > max_bytes:
            raise ValueError("file_exceeds_snapshot_cap")
        result["output_limit_bytes"] = before.st_size
    result["archive_limit_bytes"] = min(
        4 * GIB, int(result["output_limit_bytes"] * 1.01) + 65536
    )
    return result


def capacity(project_root, output, plans, clone_restore):
    reserve, scope = reserve_bytes(project_root, output)
    # Clones must succeed; no unreserved physical-copy fallback is permitted.
    clones = (
        len(plans) * 1024**2
        if clone_restore
        else sum(p["output_limit_bytes"] for p in plans)
    )
    allocation = (
        sum(p["output_limit_bytes"] + p.get("archive_limit_bytes", 0) for p in plans)
        + clones
    )
    free = shutil.disk_usage(existing_parent(output)).free
    return {
        "known": True,
        "sufficient": free >= reserve + allocation,
        "required_free_bytes": reserve + allocation,
        "allocation_budget_bytes": allocation,
        "reserve_bytes": reserve,
        "reserve_scope": scope,
        "free_bytes": free,
        "copy_count": len(plans),
        "clone_restore": clone_restore,
        "allocation_scope": "enforced_per_target_caps_plus_restore_budget",
        "reservation_scope": "cooperating_storage_maintenance_lock_owners_only",
    }


class CopyGuard:
    def __init__(self, root, output, reserve, seconds, *, operator_approved=False):
        self.root, self.output, self.reserve = root, output, reserve
        self.deadline = time.monotonic() + seconds
        self.pace = WorkBudget()
        self.last_probe = 0
        self.operator_approved = operator_approved
        self.admission_wait_seconds = 0.0

    def _wait_for_recovery_resources(self):
        from scripts.ops.approved_storage_recovery import (
            recovery_hold_active, resources_admitted,
        )

        started = time.monotonic()
        stable = None
        while True:
            now = time.monotonic()
            if now >= self.deadline or now - started >= 120:
                raise RuntimeError(
                    "approved_recovery_hard_resource_admission_withdrawn"
                )
            if recovery_hold_active(self.root):
                raise RuntimeError("maintenance_or_operator_hold")
            if shutil.disk_usage(self.output).free < self.reserve:
                raise RuntimeError("snapshot_reserve_consumed")
            admitted = resources_admitted(self.root)
            stable = (stable if stable is not None else now) if admitted else None
            if stable is not None and now - stable >= 10:
                self.admission_wait_seconds += now - started
                return
            # No copy, hashing, compression, or SQLite work proceeds during a hold.
            time.sleep(min(1.0, max(self.deadline - now, 0.0)))

    def check(self):
        self.pace.tick()
        now = time.monotonic()
        if now >= self.deadline:
            raise TimeoutError("snapshot_operation_deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024
        if rss >= 256 * 1024**2:
            raise RuntimeError("snapshot_resident_memory_budget_reached")
        if now - self.last_probe >= 1:
            from scripts.ops.support_maintenance_gate import (
                support_maintenance_freeze_contract,
            )

            if self.operator_approved:
                from scripts.ops.approved_storage_recovery import resources_admitted

                if not resources_admitted(self.root):
                    self._wait_for_recovery_resources()
            elif support_maintenance_freeze_contract(self.root, "state_snapshot_drill")[
                "active"
            ]:
                raise RuntimeError("support_maintenance_frozen_for_mac_fluidity")
            if shutil.disk_usage(self.output).free < self.reserve:
                raise RuntimeError("snapshot_reserve_consumed")
            from scripts.ops.approved_storage_recovery import recovery_hold_active

            if recovery_hold_active(self.root):
                raise RuntimeError("maintenance_or_operator_hold")
            self.last_probe = now


def compact_snapshot(source, target, plan, guard):
    source, target = allowed(source), allowed(target, missing=True)
    if target.exists():
        raise ValueError("snapshot_destination_exists")
    if [source.stat().st_dev, source.stat().st_ino] != plan["source_identity"]:
        raise ValueError("snapshot_source_replaced")
    failure = []
    wal = Path(str(source) + "-wal")
    initial_wal = inspect_storage_path(wal).get("size_bytes") or 0
    last_check = 0

    def progress():
        nonlocal last_check
        try:
            guard.check()
            now = time.monotonic()
            if now - last_check >= 0.25:
                if (
                    target.exists()
                    and target.stat().st_size > plan["output_limit_bytes"]
                ):
                    raise RuntimeError("compact_snapshot_exceeds_output_cap")
                growth = (
                    inspect_storage_path(wal).get("size_bytes") or 0
                ) - initial_wal
                if growth > GIB:
                    raise RuntimeError("snapshot_source_wal_growth_limit")
                last_check = now
            return 0
        except Exception as exc:
            failure.append(exc)
            return 1

    with closing(
        sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=2)
    ) as db:
        db.execute("PRAGMA cache_size=-4096")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA threads=1")
        db.set_progress_handler(progress, 10000)
        try:
            db.execute("VACUUM INTO ?", (str(target),))
        except sqlite3.Error:
            if failure:
                raise failure[0]
            raise
    if target.stat().st_size > plan["output_limit_bytes"]:
        raise RuntimeError("compact_snapshot_exceeds_output_cap")
    with target.open("rb") as file:
        os.fsync(file.fileno())


def clone_restore(source, target):
    source, target = allowed(source), allowed(target, missing=True)
    if sys.platform != "darwin":
        raise RuntimeError("apfs_clone_unavailable")
    if target.exists() or source.stat().st_dev != target.parent.stat().st_dev:
        raise RuntimeError("clone_requires_new_same_volume_target")
    library = ctypes.CDLL(None, use_errno=True)
    clone = library.clonefile
    clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    clone.restype = ctypes.c_int
    if clone(os.fsencode(source), os.fsencode(target), 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "apfs_clone_failed_no_copy_fallback")
    if source.stat().st_ino == target.stat().st_ino:
        raise RuntimeError("restore_must_be_independent_inode")


def probe_clone(output):
    output = allowed(output)
    with tempfile.TemporaryDirectory(prefix=".clone_probe_", dir=output) as directory:
        source, target = Path(directory) / "source", Path(directory) / "restore"
        source.write_bytes(b"snapshot-clone-probe")
        clone_restore(source, target)
        if target.read_bytes() != source.read_bytes():
            raise RuntimeError("clone_probe_bytes_differ")
        target.write_bytes(b"changed-clone")
        if source.read_bytes() != b"snapshot-clone-probe":
            raise RuntimeError("clone_probe_not_write_isolated")


def verify_snapshot(snapshot, restored, guard, sqlite):
    def sha(path):
        h = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024**2), b""):
                guard.check()
                h.update(block)
            os.fsync(file.fileno())
        return h.hexdigest()

    a, b = sha(snapshot), sha(restored)
    if a != b:
        raise RuntimeError("snapshot_restore_hash_mismatch")
    if sqlite:
        failure = []

        def progress():
            try:
                guard.check()
                return 0
            except Exception as exc:
                failure.append(exc)
                return 1

        with closing(sqlite3.connect(restored.as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("PRAGMA cache_size=-4096")
            db.set_progress_handler(progress, 10000)
            try:
                if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    raise RuntimeError("restored_sqlite_integrity_failure")
            except sqlite3.Error:
                if failure:
                    raise failure[0]
                raise
    return a, b


def seal_compressed_snapshot(
    snapshot, restored, expected_sha, guard, max_archive_bytes, *, archive_path=None
):
    """Retain a restorable gzip only after a full decoded-byte verification."""
    archive = archive_path or snapshot.with_suffix(snapshot.suffix + ".gz")
    original_size = snapshot.stat().st_size
    with archive.open("xb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0
        ) as zipped:
            with snapshot.open("rb") as source:
                for block in iter(lambda: source.read(1024**2), b""):
                    guard.check()
                    zipped.write(block)
                    if raw.tell() > max_archive_bytes:
                        raise RuntimeError(
                            "compressed_snapshot_archive_budget_exceeded"
                        )
        raw.flush()
        os.fsync(raw.fileno())
    if archive.stat().st_size > max_archive_bytes:
        raise RuntimeError("compressed_snapshot_archive_budget_exceeded")
    restored_sha, size = hashlib.sha256(), 0
    with gzip.open(archive, "rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            guard.check()
            restored_sha.update(block)
            size += len(block)
    if restored_sha.hexdigest() != expected_sha or size != original_size:
        raise RuntimeError("compressed_restore_verification_failed_keep_snapshots")
    archive_sha = hashlib.sha256()
    with archive.open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            guard.check()
            archive_sha.update(block)
    proof = {
        "archive_path": str(archive),
        "archive_sha256": archive_sha.hexdigest(),
        "decoded_sha256": expected_sha,
        "decoded_bytes": original_size,
        "archive_bytes": archive.stat().st_size,
        "verification": "full_decoded_bytes_match_verified_sqlite_or_file_restore",
    }
    import json

    with archive.with_suffix(archive.suffix + ".receipt.json").open("x") as receipt:
        json.dump(proof, receipt)
        receipt.flush()
        os.fsync(receipt.fileno())
    fd = os.open(archive.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    # Only this run's verified temporary snapshot and restore probe are released.
    # The live source, other runs, and any failed copies are never removed here.
    restored.unlink()
    snapshot.unlink()
    return proof
