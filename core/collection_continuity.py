"""Durable collection-only outbox. Recovery archives evidence, never queue actions."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import math
import os
import shutil
import sqlite3
import stat
import tempfile
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from core import sqlite_primary_storage as primary
from core.storage_router import inspect_storage_path
from core.write_path_recovery import durable_json, local_path

RELATIVE_DB = "governance/collection_buffer/outbox.sqlite3"
REPORT = "governance/health/collection_continuity_latest.json"
MAX_BATCH_BYTES = 8 * 1024 * 1024
MAX_PENDING_BYTES = 4 * 1024**3
CHANNELS = frozenset(
    {"runtime", "gate", "ingress", "api", "loop_state", "decision", "risk"}
)


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _reserve_bytes() -> int:
    configured = float(os.getenv("BOT_LOCAL_STORAGE_TARGET_FREE_GB", "125"))
    if not math.isfinite(configured) or configured <= 0:
        raise ValueError("collection_buffer_invalid_reserve")
    return int(max(125.0, configured) * 1024**3)


def _outbox_path(root: Path) -> Path:
    path = root / RELATIVE_DB
    route = inspect_storage_path(path, boundary_root=root, allow_external=False)
    if route["status"] not in {"present", "missing"} or route.get("symlinks"):
        raise ValueError("collection_buffer_requires_physical_internal_path")
    return path


def collection_start_allowed(root: Path) -> None:
    from core.live_execution_switch import switch_status
    from core.runtime_maintenance import maintenance_hold_snapshot

    switch = switch_status(root)
    if (
        not switch.get("ok")
        or switch.get("switch") != "OFF"
        or switch.get("requested_on")
    ):
        raise RuntimeError("collection_only_start_requires_execution_off")
    if maintenance_hold_snapshot(root).get("active"):
        raise RuntimeError("runtime_maintenance_hold_active")
    for relative in (
        "governance/channels",
        "decisions",
        "decision_explanations",
        "logs",
    ):
        local_path(root, root / relative)
    if _free_bytes(root) < _reserve_bytes():
        raise RuntimeError("collection_buffer_internal_reserve")


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _connect(root: Path) -> sqlite3.Connection:
    path = _outbox_path(root)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    primary._physical(path.parent)
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(str(path) + suffix)
        if os.path.lexists(candidate) and not stat.S_ISREG(candidate.lstat().st_mode):
            raise ValueError("collection_buffer_nonregular_file")
    conn = sqlite3.connect(path, timeout=2)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pending (id TEXT PRIMARY KEY, body BLOB NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS receipts (id TEXT NOT NULL, target TEXT NOT NULL, verified_utc TEXT NOT NULL, PRIMARY KEY(id,target))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS capacity (singleton INTEGER PRIMARY KEY CHECK(singleton=1), pending_bytes INTEGER NOT NULL CHECK(pending_bytes>=0))"
        )
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM capacity WHERE singleton=1").fetchone() is None:
            conn.execute(
                "INSERT INTO capacity SELECT 1, COALESCE(SUM(length(body)),0) FROM pending"
            )
        conn.commit()
        _sync_directory(path.parent)
        return conn
    except BaseException:
        conn.close()
        raise


def preserve(
    root: Path, *, channel: str, source_path: str, payloads: list[dict]
) -> str:
    """ACK only after FULL-sync local commit; duplicate batches keep the same ID."""
    root = root.absolute()
    if not primary.enabled(root) or channel not in CHANNELS or not payloads:
        raise ValueError("collection_buffer_scope_rejected")
    source = Path(os.path.abspath(source_path))
    if not source.is_relative_to(root) or source.suffix != ".jsonl":
        raise ValueError("collection_buffer_source_rejected")
    if not all(isinstance(row, dict) and row.get("message_id") for row in payloads):
        raise ValueError("collection_buffer_message_identity_required")
    body = json.dumps(
        {
            "schema_version": 1,
            "purpose": "collection_evidence_only",
            "channel": channel,
            "source_relative": str(source.relative_to(root)),
            "payloads": payloads,
            "queue_replay_authorized": False,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    if len(body) > MAX_BATCH_BYTES:
        raise ValueError("collection_buffer_batch_budget")
    if _free_bytes(root) - len(body) < _reserve_bytes():
        raise ValueError("collection_buffer_internal_reserve")
    batch_id = hashlib.sha256(body).hexdigest()
    with closing(_connect(root)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT body FROM pending WHERE id=?", (batch_id,)
        ).fetchone()
        if existing is not None:
            if existing[0] != body:
                raise ValueError("collection_buffer_identity_conflict")
            return batch_id
        pending = conn.execute(
            "SELECT pending_bytes FROM capacity WHERE singleton=1"
        ).fetchone()[0]
        if pending + len(body) > MAX_PENDING_BYTES:
            raise ValueError("collection_buffer_capacity")
        # A prior archive receipt cannot prove the drive still retains those bytes.
        conn.execute("INSERT INTO pending VALUES (?,?)", (batch_id, body))
        conn.execute(
            "UPDATE capacity SET pending_bytes=pending_bytes+? WHERE singleton=1",
            (len(body),),
        )
    return batch_id


def _readback(path: Path, expected: bytes) -> None:
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_BATCH_BYTES + 65536:
            raise ValueError("collection_buffer_archive_invalid")
        with gzip.GzipFile(fileobj=handle) as archive:
            restored = archive.read(len(expected) + 1)
        if restored != expected:
            raise ValueError("collection_buffer_archive_conflict")
        if primary._identity(path) != [
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ]:
            raise ValueError("collection_buffer_archive_changed")


def _publish(target: Path, batch_id: str, body: bytes) -> Path:
    directory = target / "cold_archive/collection_buffer"
    # Never create a replacement mount directory after the device disappears.
    primary._physical(target)
    device = target.stat().st_dev
    for child in (target / "cold_archive", directory):
        primary._physical(child.parent)
        child.mkdir(exist_ok=True)
        primary._physical(child)
        if child.stat().st_dev != device:
            raise ValueError("collection_buffer_device_boundary")
    destination = directory / f"{batch_id}.json.gz"
    if not os.path.lexists(destination):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=directory, prefix=".collection-", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(gzip.compress(body, mtime=0))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
            _sync_directory(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    _readback(destination, body)
    return destination


def reconcile(root: Path, **budgets) -> dict:
    root = root.absolute()
    path = _outbox_path(root)
    if not os.path.lexists(path):
        return _reconcile_locked(root, **budgets)
    lock_path = local_path(root, path.parent / "drain.lock")
    with os.fdopen(
        os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "a+"
    ) as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "ok": False,
                "overall_status": "deferred",
                "reason": "collection_buffer_drain_busy",
            }
        return _reconcile_locked(root, **budgets)


def _reconcile_locked(
    root: Path,
    *,
    max_batches: int = 64,
    max_bytes: int = 32 * 1024**2,
    max_seconds: float = 10,
) -> dict:
    """Bounded archival shared by existing failback and SQL writer owners."""
    from core.runtime_maintenance import maintenance_hold_snapshot

    root = root.absolute()
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": False,
        "overall_status": "deferred",
        "archived_batches": 0,
        "archived_bytes": 0,
        "pending_batches": None,
        "queue_replay_authorized": False,
        "live_execution_authority": False,
        "database_routes_changed": False,
        "canonical_ingestion_verified": False,
    }
    try:
        if not primary.enabled(root):
            raise ValueError("collection_buffer_requires_selected_primary")
        if maintenance_hold_snapshot(root).get("active"):
            raise ValueError("runtime_maintenance_hold_active")
        target = primary.require_ready(root)
        if not os.path.lexists(_outbox_path(root)):
            result.update(ok=True, overall_status="ready", pending_batches=0)
        else:
            deadline = time.monotonic() + min(max(float(max_seconds), 0), 20)
            with closing(_connect(root)) as conn:
                for _ in range(min(max(int(max_batches), 0), 128)):
                    if time.monotonic() >= deadline:
                        break
                    row = conn.execute(
                        "SELECT id,body FROM pending ORDER BY rowid LIMIT 1"
                    ).fetchone()
                    if row is None:
                        break
                    batch_id, body = row
                    if result["archived_bytes"] + len(body) > max_bytes:
                        break
                    if hashlib.sha256(body).hexdigest() != batch_id:
                        raise ValueError("collection_buffer_local_integrity")
                    if primary.require_ready(root) != target:
                        raise ValueError("collection_buffer_target_changed")
                    destination = _publish(target, batch_id, body)
                    if primary.require_ready(
                        root
                    ) != target or maintenance_hold_snapshot(root).get("active"):
                        raise ValueError(
                            "collection_buffer_post_copy_admission_changed"
                        )
                    _readback(destination, body)
                    with conn:
                        conn.execute(
                            "INSERT OR IGNORE INTO receipts VALUES (?,?,?)",
                            (
                                batch_id,
                                str(destination),
                                datetime.now(timezone.utc).isoformat(),
                            ),
                        )
                        deleted = conn.execute(
                            "DELETE FROM pending WHERE id=? AND body=?",
                            (batch_id, body),
                        ).rowcount
                        if deleted != 1:
                            raise ValueError("collection_buffer_pending_changed")
                        conn.execute(
                            "UPDATE capacity SET pending_bytes=pending_bytes-? WHERE singleton=1",
                            (len(body),),
                        )
                    result["archived_batches"] += 1
                    result["archived_bytes"] += len(body)
                pending = conn.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
                result.update(
                    ok=True,
                    overall_status="ready" if not pending else "catching_up",
                    pending_batches=pending,
                )
    except (OSError, EOFError, ValueError, RuntimeError, sqlite3.Error) as exc:
        result["reason"] = str(exc)
    durable_json(root, root / REPORT, result)
    return result
