"""Opt-in, UUID-bound SQLite routing; no automatic migration or retirement."""

from __future__ import annotations

import fcntl
from contextvars import ContextVar
import hashlib
import json
import os
import plistlib
import shutil
import shlex
import stat
import subprocess
import time
import uuid
from pathlib import Path

PROFILE = "sqlite_primary"
_managed_settings: ContextVar[dict | None] = ContextVar(
    "sqlite_primary_settings", default=None
)
DATABASES = (
    "jsonl_link.sqlite3",
    "bot_channel_queue.sqlite3",
    "snapshot_context.sqlite3",
)
LINKS = ("data/sql_link_shards",) + tuple(
    f"data/{name}{suffix}" for name in DATABASES for suffix in ("", "-wal", "-shm")
)


def _load_managed_profile(project_root: Path) -> dict | None:
    path = Path(project_root).absolute() / "config/.env.storage_target_override"
    if not os.path.lexists(path):
        return
    _physical(path.parent)
    before = _identity(path)
    if before[2] > 16384:
        raise ValueError("sqlite_primary_target_config_oversized")
    values = {}
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, raw = line.partition("=")
            if not separator or key in values:
                raise ValueError("sqlite_primary_target_config_invalid")
            parts = shlex.split(raw, comments=True)
            if len(parts) > 1:
                raise ValueError("sqlite_primary_target_config_invalid")
            values[key] = parts[0] if parts else ""
    if _identity(path) != before:
        raise ValueError("sqlite_primary_target_config_changed")
    profile = values.get("BOT_STORAGE_ROUTE_PROFILE", "")
    if profile not in {"", PROFILE}:
        raise ValueError("unsupported_storage_route_profile")
    if profile == PROFILE:
        required = {
            "BOT_STORAGE_ROUTE_PROFILE",
            "BOT_LOGS_EXTERNAL_MOUNT",
            "BOT_LOGS_EXTERNAL_PROJECT_DIR",
            "BOT_LOGS_EXTERNAL_PROJECT_ROOT",
            "BOT_LOGS_EXTERNAL_VOLUME_UUID",
        }
        if any(not values.get(key) for key in required):
            raise ValueError("sqlite_primary_target_config_incomplete")
        # Persisted operator selection outranks stale launchd environments.
        # Loading the contract never adopts, repairs or merges database routes.
        allowed = required | {
            "BOT_LOGS_EXTERNAL_VOLUME_NAME",
            "BOT_LOGS_EXTERNAL_MOUNT_CANDIDATES",
            "BOT_LOGS_EXTERNAL_DISK_IDENTIFIER",
        }
        return {key: value for key, value in values.items() if key in allowed}
    return None


def _getenv(key: str, default: str = "") -> str:
    settings = _managed_settings.get() or {}
    return settings.get(key, os.getenv(key, default))


def enabled(project_root: Path | str | None = None) -> bool:
    _managed_settings.set(None)
    _managed_settings.set(
        _load_managed_profile(Path(project_root)) if project_root is not None else None
    )
    profile = _getenv("BOT_STORAGE_ROUTE_PROFILE").strip()
    if profile not in {"", PROFILE}:
        raise ValueError("unsupported_storage_route_profile")
    return profile == PROFILE


def configured_target() -> Path:
    mount = Path(_getenv("BOT_LOGS_EXTERNAL_MOUNT"))
    project = _getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", "schwab_trading_bot")
    if (
        not mount.is_absolute()
        or mount.parent != Path("/Volumes")
        or mount.name.casefold() in {"video", ".", ".."}
        or project in {"", ".", ".."}
        or Path(project).name != project
    ):
        raise ValueError("sqlite_primary_invalid_target")
    target = mount / project
    explicit = _getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT")
    if explicit and Path(explicit) != target:
        raise ValueError("sqlite_primary_target_binding_mismatch")
    return target


def _physical(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if not stat.S_ISDIR(current.lstat().st_mode):
            raise ValueError(f"sqlite_primary_nonphysical_directory:{current}")


def _validated_target() -> Path:
    target = configured_target()
    mount = target.parent
    expected = str(uuid.UUID(_getenv("BOT_LOGS_EXTERNAL_VOLUME_UUID"))).upper()
    _physical(target)
    if not os.path.ismount(mount):
        raise ValueError("sqlite_primary_mount_missing")
    proc = subprocess.run(
        ["/usr/sbin/diskutil", "info", "-plist", str(mount)],
        capture_output=True,
        timeout=10,
        check=True,
    )
    if len(proc.stdout) > 1024 * 1024:
        raise ValueError("sqlite_primary_device_report_oversized")
    info = plistlib.loads(proc.stdout)
    if (
        str(info.get("VolumeUUID", "")).upper() != expected
        or info.get("MountPoint") != str(mount)
        or info.get("FilesystemType") != "apfs"
        or info.get("Internal") is not False
        or not info.get("WritableVolume")
    ):
        raise ValueError("sqlite_primary_device_identity_mismatch")
    if target.stat().st_dev != mount.stat().st_dev:
        raise ValueError("sqlite_primary_device_boundary_mismatch")
    from core.storage_router import _external_min_free_bytes

    if shutil.disk_usage(target).free < max(200 * 1024**3, _external_min_free_bytes()):
        raise ValueError("sqlite_primary_destination_reserve")
    return target


def logical_database_path(project_root: Path, path: Path | str) -> Path:
    """Resolve declared aliases without I/O; admission precedes opening a DB."""
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    root = Path(project_root).absolute()
    enabled(root)
    for base in (root, root / "local_fallback_storage", configured_target()):
        try:
            rel = candidate.relative_to(base)
        except ValueError:
            continue
        if rel.as_posix() in {f"data/{name}" for name in DATABASES} or (
            rel.parts[:2] == ("data", "sql_link_shards")
        ):
            return root / rel
    return candidate


def managed_path(project_root: Path, path: Path) -> bool:
    root = Path(project_root).absolute()
    return path in {root / "data" / name for name in DATABASES} or path.is_relative_to(
        root / "data/sql_link_shards"
    )


def observe(project_root: Path) -> dict:
    from core.storage_router import inspect_storage_path

    root = Path(project_root).absolute()
    enabled(root)
    result = {
        "profile": PROFILE,
        "ok": False,
        "mode": "sqlite_primary_unavailable",
        "blockers": [],
        "mismatches": [],
        "route_mutation_performed": False,
        "integrity_verified": False,
        "delete_authority": False,
    }
    try:
        target = _validated_target()
        result["target_root"] = str(target)
        for control in (
            "governance",
            "governance/runtime",
            "governance/health",
            "governance/execution_lanes",
        ):
            check = inspect_storage_path(
                root / control, boundary_root=root, allow_external=False
            )
            if check["status"] not in {"present", "missing"}:
                raise ValueError("sqlite_primary_control_route_external_or_invalid")
        _physical(root / "data")
        _physical(target / "data/sql_link_shards")
        for name in DATABASES:
            path = target / "data" / name
            if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size < 100:
                raise ValueError("sqlite_primary_payload_missing_or_alias")
        for rel in LINKS:
            path = root / rel
            wanted = target / rel
            if (
                not path.is_symlink()
                or Path(os.path.abspath(path.parent / os.readlink(path))) != wanted
            ):
                result["mismatches"].append(rel)
        result["mode"] = "sqlite_primary_pending" if result["mismatches"] else PROFILE
        if result["mismatches"]:
            result["blockers"].append("sqlite_primary_handoff_required")
        result["ok"] = not result["blockers"]
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        result["blockers"].append(str(exc))
    result["route_verification"] = {
        "verification_state": "ready" if result["ok"] else "blocked",
        "certified_mode": result["mode"],
        "scope": "declared_sqlite_routes_only",
        "tracked_count": len(LINKS),
        "ready_count": len(LINKS) if result["ok"] else 0,
        "coverage_ratio": 1.0 if result["ok"] else 0.0,
        "mismatches": list(result["mismatches"]),
        "blockers": list(result["blockers"]),
        "integrity_verified": False,
        "ingestion_verified": False,
    }
    return result


def require_ready(project_root: Path) -> Path:
    observation = observe(project_root)
    if not observation["ok"]:
        raise RuntimeError(
            "sqlite_primary_deferred:" + ",".join(observation["blockers"])
        )
    return Path(observation["target_root"])


def check_database_open(
    project_root: Path, path: Path | str, *, readonly: bool = False
) -> None:
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    logical = logical_database_path(project_root, candidate)
    if not managed_path(project_root, logical):
        return
    if candidate.is_relative_to(
        Path(project_root).absolute() / "local_fallback_storage"
    ):
        if not readonly:
            raise RuntimeError("sqlite_primary_standby_write_forbidden")
        return  # Explicit read-only standby inspection must not read the active DB instead.
    require_ready(project_root)


def protected_owner_observation(project_root: Path, owner: str) -> dict:
    from datetime import datetime, timezone

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": False,
        "overall_status": "blocked",
        "reason": "sqlite_primary_requires_verified_handoff_owner",
        "owner": owner,
        "route_verification": observe(project_root),
        "route_mutation_performed": False,
        "source_files_removed": 0,
    }


def _identity(path: Path) -> list[int]:
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError("sqlite_primary_payload_alias")
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def _hold(root: Path) -> None:
    from core.runtime_maintenance import (
        maintenance_hold_snapshot,
        maintenance_hold_token_authorized,
    )

    _physical(root / "governance/runtime")
    if not maintenance_hold_token_authorized(maintenance_hold_snapshot(root)):
        raise RuntimeError("sqlite_primary_authorized_maintenance_required")
    state = json.loads(
        (root / "governance/runtime/live_execution_switch_state.json").read_text()
    )
    if state.get("requested_on") is not False:
        raise RuntimeError("sqlite_primary_trading_off_required")


def _digest(path: Path, deadline: float) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            if time.monotonic() > deadline:
                raise TimeoutError("sqlite_primary_handoff_deadline")
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _shard_inventory(base: Path, deadline: float) -> set[str]:
    _physical(base)
    found = set()
    entries = 0

    def scan_error(error: OSError) -> None:
        raise error

    for current, dirs, files in os.walk(base, followlinks=False, onerror=scan_error):
        entries += len(dirs) + len(files)
        if entries > 10000 or time.monotonic() > deadline:
            raise ValueError("sqlite_primary_inventory_budget")
        for name in dirs:
            _physical(Path(current) / name)
        for name in files:
            path = Path(current) / name
            identity = _identity(path)
            if path.name.endswith(("-wal", "-shm", "-journal")):
                if identity[2]:
                    raise ValueError("sqlite_primary_journal_not_quiesced")
                continue
            found.add("sql_link_shards/" + path.relative_to(base).as_posix())
    return found


def commit_routes(project_root: Path, receipt_path: Path) -> dict:
    """Explicit operator handoff; recheck full target hashes under native hold.

    Receipt rows bind fresh quiesced source identities to previously integrity-
    checked target files. Missing inventory, aliases, open handles and journals
    block adoption. No payload or original is copied, merged, or deleted here.
    """
    root = Path(project_root).absolute()
    enabled(root)
    deadline = time.monotonic() + 900
    from core.storage_router import inspect_storage_path

    _hold(root)
    target = _validated_target()
    source = root / "local_fallback_storage/data"
    _physical(source)
    route = inspect_storage_path(
        receipt_path,
        boundary_root=root / "governance/storage_recovery",
        allow_external=False,
    )
    if route["status"] != "present" or route["symlinks"]:
        raise ValueError("sqlite_primary_receipt_not_owned")
    with receipt_path.open("rb") as stream:
        raw = stream.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2:
        raise ValueError("sqlite_primary_receipt_oversized")
    receipt = json.loads(raw)
    if (
        receipt.get("purpose") != "sqlite_primary_cutover"
        or receipt.get("schema_version") != 1
        or receipt.get("target_root") != str(target)
        or receipt.get("source_root") != str(source)
        or receipt.get("volume_uuid") != _getenv("BOT_LOGS_EXTERNAL_VOLUME_UUID")
    ):
        raise ValueError("sqlite_primary_receipt_binding_mismatch")
    rows = receipt["files"]
    if not isinstance(rows, list) or not 3 < len(rows) <= 10000:
        raise ValueError("sqlite_primary_receipt_inventory_invalid")
    expected = set(DATABASES)
    # Every shard entry must have custody, including historical lookup files.
    for base in (source / "sql_link_shards", target / "data/sql_link_shards"):
        found = _shard_inventory(base, deadline)
        if base == source / "sql_link_shards":
            expected |= found
        elif found != expected - set(DATABASES):
            raise ValueError("sqlite_primary_shard_inventory_mismatch")
    if (
        len({r["relative"] for r in rows}) != len(rows)
        or {r["relative"] for r in rows} != expected
    ):
        raise ValueError("sqlite_primary_receipt_inventory_mismatch")
    _physical(root / "governance/locks")
    with (root / "governance/locks/sqlite_primary_route.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = observe(root)
        if observed["mode"] == "sqlite_primary_unavailable":
            raise RuntimeError(str(observed["blockers"]))
        for base in (source, target / "data"):
            handles = subprocess.run(
                ["lsof", "+D", str(base)], capture_output=True, timeout=30
            )
            if handles.returncode != 1 or handles.stdout or handles.stderr:
                raise RuntimeError("sqlite_primary_handles_present_or_unknown")
        target_identities = {}
        for row in rows:
            rel = Path(row["relative"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("sqlite_primary_invalid_receipt_path")
            old, new = source / rel, target / "data" / rel
            if _identity(old) != row["source_identity"]:
                raise ValueError("sqlite_primary_source_changed")
            for path in (old, new):
                for suffix in ("-wal", "-shm", "-journal"):
                    journal = Path(str(path) + suffix)
                    if os.path.lexists(journal) and _identity(journal)[2]:
                        raise ValueError("sqlite_primary_journal_not_quiesced")
            before = _identity(new)
            digest = _digest(new, deadline)
            if digest != row["sha256"] or before != _identity(new):
                raise ValueError("sqlite_primary_target_hash_mismatch")
            target_identities[row["relative"]] = before
            if rel.suffix == ".sqlite3" and row.get("quick_check") != "ok":
                raise ValueError("sqlite_primary_integrity_receipt_required")
            _hold(root)
        _validated_target()
        for base in (source / "sql_link_shards", target / "data/sql_link_shards"):
            if _shard_inventory(base, deadline) != expected - set(DATABASES):
                raise ValueError("sqlite_primary_inventory_changed_before_commit")
        for row in rows:
            if _identity(source / row["relative"]) != row["source_identity"]:
                raise ValueError("sqlite_primary_source_changed_before_commit")
            if (
                _identity(target / "data" / row["relative"])
                != target_identities[row["relative"]]
            ):
                raise ValueError("sqlite_primary_target_changed_before_commit")
            for base in (source, target / "data"):
                for suffix in ("-wal", "-shm", "-journal"):
                    journal = Path(str(base / row["relative"]) + suffix)
                    if os.path.lexists(journal) and _identity(journal)[2]:
                        raise ValueError("sqlite_primary_journal_changed_before_commit")
        for base in (source, target / "data"):
            handles = subprocess.run(
                ["lsof", "+D", str(base)], capture_output=True, timeout=30
            )
            if handles.returncode != 1 or handles.stdout or handles.stderr:
                raise RuntimeError("sqlite_primary_handles_changed_before_commit")
        _publish_verified_links(root, target, hashlib.sha256(raw).hexdigest())
    return {
        **observe(root),
        "route_mutation_performed": True,
        "source_retired": False,
        "payload_hashes_verified": len(rows),
    }


def _publish_verified_links(root: Path, target: Path, receipt_sha256: str) -> Path:
    """Caller holds the route lock and has verified the complete quiet point."""
    originals = {}
    for rel in LINKS:
        path = root / rel
        if os.path.lexists(path) and not path.is_symlink():
            raise ValueError("sqlite_primary_existing_nonlink_route")
        originals[rel] = os.readlink(path) if path.is_symlink() else None
        if originals[rel] is not None:
            actual = Path(os.path.abspath(path.parent / originals[rel]))
            if actual not in {root / "local_fallback_storage" / rel, target / rel}:
                raise ValueError("sqlite_primary_source_route_mismatch")
    changed = []
    from core.write_path_recovery import durable_json

    journal = (
        root
        / "governance/storage_recovery"
        / ("sqlite_primary_transaction_" + uuid.uuid4().hex + ".json")
    )
    transaction = {
        "purpose": "sqlite_primary_route_transaction",
        "phase": "prepared",
        "receipt_sha256": receipt_sha256,
        "target_root": str(target),
        "original_links": originals,
        "source_retired": False,
    }
    durable_json(root, journal, transaction)
    try:
        for rel in LINKS:
            _hold(root)
            path = root / rel
            current = os.readlink(path) if path.is_symlink() else None
            if current != originals[rel] or (
                os.path.lexists(path) and not path.is_symlink()
            ):
                raise RuntimeError("sqlite_primary_route_changed_during_handoff")
            temp = path.with_name("." + path.name + "." + uuid.uuid4().hex)
            try:
                temp.symlink_to(target / rel)
                os.replace(temp, path)
                changed.append(rel)
                _sync_directory(path.parent)
            finally:
                if temp.is_symlink():
                    temp.unlink()
        require_ready(root)
        _hold(root)
        durable_json(root, journal, {**transaction, "phase": "committed"})
    except BaseException:
        conflicts = []
        for rel in reversed(changed):
            path = root / rel
            if not path.is_symlink() or os.readlink(path) != str(target / rel):
                conflicts.append(rel)
                continue
            path.unlink()
            if originals[rel] is not None:
                path.symlink_to(originals[rel])
            _sync_directory(path.parent)
        durable_json(
            root,
            journal,
            {
                **transaction,
                "phase": "rollback_incomplete" if conflicts else "rolled_back",
                "rollback_conflicts": conflicts,
            },
        )
        raise
    return journal
