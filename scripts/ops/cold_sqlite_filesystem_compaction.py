"""Transparent, verified compression of inactive cold SQLite files on APFS."""

from __future__ import annotations

import hashlib
from contextlib import closing, nullcontext
import json
import math
import os
from pathlib import Path
import plistlib
import shutil
import sqlite3
import stat
import sys
import tempfile
import time
from typing import Callable, ContextManager

from core.storage_router import inspect_storage_path
from scripts.ops.long_runtime_common import iso_now, run_bounded_process_group
from scripts.ops import cold_sqlite_streaming_compressor as streaming

GIB = 1024**3
MAX_NATIVE_FILE_BYTES = 2 * GIB - 1


def select_compressor(requested: str) -> str:
    if requested == "auto":
        if streaming.installed():
            return "applesauce"
        return "afsctool" if shutil.which("afsctool") else "ditto"
    if requested not in {"ditto", "afsctool", "applesauce"}:
        raise ValueError("unknown_filesystem_compressor")
    return requested


def require_compressor(compressor: str) -> str:
    if compressor not in {"ditto", "afsctool", "applesauce"}:
        raise ValueError("unknown_filesystem_compressor")
    # Every backend works on an isolated physical copy made by ditto.
    if not os.access("/usr/bin/ditto", os.X_OK):
        raise RuntimeError("ditto_not_installed")
    if compressor == "applesauce":
        return streaming.require_executable()
    executable = (
        shutil.which("afsctool") if compressor == "afsctool" else "/usr/bin/ditto"
    )
    if not executable:
        raise RuntimeError("afsctool_not_installed")
    return executable


def select_inactive_archives(
    root: Path,
    *,
    max_files: int,
    max_raw_gb: float,
    min_age_hours: float,
    compressor: str = "ditto",
) -> list[Path]:
    _allowed(root)
    selected = []
    total = 0
    inspected = 0
    deadline = time.monotonic() + 5
    for base, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name
            for name in directories
            if inspect_storage_path(Path(base) / name).get("status") == "present"
            and not (Path(base) / name).is_symlink()
            and not name.startswith(".")
        )
        for name in sorted(files):
            inspected += 1
            if inspected > 5000 or time.monotonic() >= deadline:
                return selected
            if not name.endswith((".sqlite3", ".sqlite")):
                continue
            path = Path(base) / name
            if (
                inspect_storage_path(path).get("status") != "present"
                or path.is_symlink()
            ):
                continue
            info = path.stat()
            if (
                info.st_size < 100 * 1024**2
                or info.st_size
                > (
                    streaming.MAX_FILE_BYTES
                    if compressor == "applesauce"
                    else MAX_NATIVE_FILE_BYTES
                )
                or getattr(info, "st_flags", 0) & stat.UF_COMPRESSED
                or info.st_nlink != 1
                or time.time() - info.st_mtime < max(min_age_hours, 24) * 3600
                or total + info.st_size > max_raw_gb * GIB
            ):
                continue
            try:
                _require_idle(path, deadline)
            except (RuntimeError, TimeoutError, OSError, ValueError):
                continue
            selected.append(path)
            total += info.st_size
            if len(selected) >= max(max_files, 1):
                return selected
    return selected


def _allowed(path: Path) -> None:
    if inspect_storage_path(path).get("status") not in {"present", "missing"}:
        raise ValueError("protected_or_unavailable_path")


def _identity(path: Path) -> tuple[int, ...]:
    _allowed(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("requires_single_link_regular_file")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _remaining(deadline: float) -> int:
    remaining = int(deadline - time.monotonic())
    if remaining < 1:
        raise TimeoutError("filesystem_compaction_deadline")
    return remaining


def _hash(path: Path, deadline: float) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            _remaining(deadline)
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _run(command: list[str], parent: Path, deadline: float) -> dict:
    return run_bounded_process_group(
        command, cwd=parent, timeout_seconds=_remaining(deadline)
    )


def _require_idle(path: Path, deadline: float) -> None:
    result = _run(["/usr/sbin/lsof", "-t", "--", str(path)], path.parent, deadline)
    if result["rc"] != 1 or result["stdout"].strip() or result["stderr"].strip():
        raise RuntimeError("cold_database_open_or_process_probe_failed")
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(path) + suffix)
        _allowed(sidecar)
        if sidecar.is_symlink() or (sidecar.exists() and sidecar.stat().st_size):
            raise RuntimeError("cold_database_has_uncheckpointed_sidecar")


def _require_apfs(device: str, parent: Path, deadline: float) -> None:
    probe = _run(["/usr/sbin/diskutil", "info", "-plist", device], parent, deadline)
    if probe["rc"] != 0 or probe.get("timed_out"):
        raise RuntimeError("filesystem_type_probe_failed")
    try:
        info = plistlib.loads(probe["stdout"].encode())
        filesystem = info.get("FilesystemType") if isinstance(info, dict) else None
    except (ValueError, TypeError, AttributeError, plistlib.InvalidFileException):
        filesystem = None
    if not isinstance(filesystem, str) or not filesystem.strip():
        raise RuntimeError("filesystem_type_probe_incomplete")
    if filesystem != "apfs":
        raise RuntimeError("transparent_compression_requires_apfs")


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _receipt(path: Path, record: dict) -> None:
    _allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(path.parent)


def compact_one(
    path: Path,
    *,
    archive_root: Path,
    manifest: Path,
    deadline: float,
    min_age_hours: float = 24.0,
    compressor: str = "ditto",
    publication_guard: Callable[[float], ContextManager] | None = None,
) -> dict:
    result = {
        "path": str(path),
        "timestamp_utc": iso_now(),
        "status": "failed",
        "original_replaced": False,
        "allocated_bytes_reclaimed": 0,
    }
    temporary = None
    try:
        for candidate in (path, archive_root, manifest):
            _allowed(candidate)
        root = archive_root.resolve()
        if not path.absolute().is_relative_to(
            archive_root.absolute()
        ) or not path.resolve().is_relative_to(root):
            raise ValueError("source_outside_cold_archive")
        identity = _identity(path)
        file_limit = (
            streaming.MAX_FILE_BYTES
            if compressor == "applesauce"
            else MAX_NATIVE_FILE_BYTES
        )
        if identity[2] > file_limit:
            raise ValueError("archive_exceeds_bounded_native_compression_size")
        before = path.stat()
        if path.suffix not in {".sqlite3", ".sqlite", ".db"}:
            raise ValueError("not_sqlite_archive")
        if time.time() - before.st_mtime < max(float(min_age_hours), 24.0) * 3600:
            raise ValueError("archive_not_old_enough")
        if sys.platform != "darwin":
            raise RuntimeError("transparent_compression_requires_macos_apfs")
        backend = require_compressor(compressor)
        executable = backend if compressor == "afsctool" else None
        if before.st_flags & stat.UF_COMPRESSED:
            return {**result, "status": "already_compressed"}
        mounted = _run(["/bin/df", "-P", str(path.parent)], path.parent, deadline)
        if mounted["rc"] != 0 or len(mounted["stdout"].splitlines()) != 2:
            raise RuntimeError("filesystem_device_probe_failed")
        device = mounted["stdout"].splitlines()[1].split()[0]
        if not device.startswith("/dev/disk"):
            raise RuntimeError("not_a_local_disk_device")
        _require_apfs(device, path.parent, deadline)
        reserve = float(os.getenv("BOT_LOCAL_STORAGE_EMERGENCY_FREE_GB", "16"))
        if not math.isfinite(reserve) or reserve < 0:
            raise ValueError("invalid_recovery_reserve")
        external = path.stat().st_dev != Path(__file__).resolve().stat().st_dev
        reserve_bytes = int(max(reserve, 64.0 if external else 16.0) * GIB)
        extra_scratch = (
            streaming.extra_scratch_bytes(identity[2])
            if compressor == "applesauce"
            else 0
        )
        result.update(
            compressor=compressor,
            reserve_bytes=reserve_bytes,
            extra_scratch_bytes=extra_scratch,
        )
        if (
            shutil.disk_usage(path.parent).free
            < identity[2] * 1.01 + extra_scratch + reserve_bytes
        ):
            raise RuntimeError("insufficient_recovery_scratch")
        _require_idle(path, deadline)
        source_hash = _hash(path, deadline)
        temporary = Path(
            tempfile.mkdtemp(prefix=".filesystem_compaction_", dir=path.parent)
        )
        target = temporary / "verified.sqlite3"
        copied = _run(
            [
                "/usr/bin/ditto",
                "--hfsCompression" if compressor == "ditto" else "--nohfsCompression",
                "--noclone",
                "--nocache",
                str(path),
                str(target),
            ],
            path.parent,
            deadline,
        )
        if copied["rc"] != 0 or copied.get("timed_out"):
            raise RuntimeError("filesystem_compression_failed_or_timed_out")
        if executable:
            compressed = _run(
                [executable, "-c", "-1", "-m", str(MAX_NATIVE_FILE_BYTES), str(target)],
                path.parent,
                deadline,
            )
            if compressed["rc"] != 0 or compressed.get("timed_out"):
                raise RuntimeError("afsctool_failed_or_timed_out")
        elif compressor == "applesauce":
            result["streaming_compression"] = streaming.compress(
                backend, target, deadline=deadline, reserve_bytes=reserve_bytes
            )
            if result["streaming_compression"]["rc"] != 0:
                raise RuntimeError("applesauce_failed")
        after = target.stat()
        result.update(
            copy_logical_bytes=after.st_size,
            copy_allocated_bytes=after.st_blocks * 512,
            copy_filesystem_compressed=bool(after.st_flags & stat.UF_COMPRESSED),
        )
        reclaimed = (before.st_blocks - after.st_blocks) * 512
        if (
            after.st_size != before.st_size
            or not (after.st_flags & stat.UF_COMPRESSED)
            or reclaimed <= 0
        ):
            raise RuntimeError("no_verified_physical_space_saving")
        target_hash = _hash(target, deadline)
        if target_hash != source_hash:
            raise RuntimeError("full_content_hash_mismatch")
        with closing(
            sqlite3.connect(target.as_uri() + "?mode=ro&immutable=1", uri=True)
        ) as conn:
            conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 10000)
            if conn.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise RuntimeError("compressed_sqlite_integrity_failed")
        with target.open("rb") as handle:
            os.fsync(handle.fileno())
        proof = {
            **result,
            "logical_bytes": before.st_size,
            "source_sha256": source_hash,
            "verified_copy_sha256": target_hash,
            "sqlite_quick_check": "ok",
            "allocated_bytes_before": before.st_blocks * 512,
            "allocated_bytes_after": after.st_blocks * 512,
            "temporary_path": str(target),
            "status": "verified_pending_replace",
        }
        guard = (
            publication_guard(deadline)
            if publication_guard
            else nullcontext((deadline, {}))
        )
        with guard as (publication_deadline, publication):
            _remaining(publication_deadline)
            _require_idle(path, publication_deadline)
            if _identity(path) != identity:
                raise RuntimeError("source_changed_during_compaction")
            if shutil.disk_usage(path.parent).free < reserve_bytes:
                raise RuntimeError("recovery_reserve_consumed_during_compaction")
            if publication:
                proof["publication_handoff"] = publication
            _receipt(manifest, proof)
            _remaining(publication_deadline)
            if _identity(path) != identity:
                raise RuntimeError("source_changed_before_replace")
            os.replace(target, path)
            result = {
                **proof,
                "status": "filesystem_compressed_verified",
                "original_replaced": True,
                "allocated_bytes_reclaimed": reclaimed,
                "completed_at_utc": iso_now(),
            }
            _sync_directory(path.parent)
            _receipt(manifest, result)
        if publication and publication.get("released") is False:
            result.update(
                status="publication_hold_release_failed",
                error="owned_hold_not_released",
            )
        return result
    except Exception as exc:
        return {**result, "status": "failed", "error": f"{type(exc).__name__}:{exc}"}
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def build_payload(
    *,
    paths: list[Path],
    archive_root: Path,
    manifest: Path,
    apply: bool,
    max_files: int,
    max_raw_gb: float,
    timeout_seconds: int,
    min_age_hours: float,
    compressor: str = "ditto",
    publication_guard: Callable[[float], ContextManager] | None = None,
) -> dict:
    deadline = time.monotonic() + max(int(timeout_seconds), 1)
    if (
        not math.isfinite(max_raw_gb)
        or max_raw_gb <= 0
        or not math.isfinite(min_age_hours)
        or min_age_hours < 0
    ):
        raise ValueError("invalid_filesystem_compaction_budget")
    requested_max_raw_gb = max_raw_gb
    if compressor == "applesauce":
        # Leave room for the paced compressor, physical copy and two read proofs.
        max_raw_gb = min(max_raw_gb, max(int(timeout_seconds), 1) / 150.0)
    selected = []
    total = 0
    for path in paths:
        _allowed(path)
        size = _identity(path)[2]
        if (
            len(selected) >= max(int(max_files), 1)
            or total + size > max(float(max_raw_gb), 0) * GIB
        ):
            continue
        selected.append(path)
        total += size
    actions = (
        [
            compact_one(
                path,
                archive_root=archive_root,
                manifest=manifest,
                deadline=deadline,
                min_age_hours=min_age_hours,
                compressor=compressor,
                publication_guard=publication_guard,
            )
            for path in selected
        ]
        if apply
        else []
    )
    ok = all(
        row["status"] in {"filesystem_compressed_verified", "already_compressed"}
        for row in actions
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "apply": apply,
        "ok": ok,
        "overall_status": "ready" if ok else "blocked",
        "mode": "transparent_cold_sqlite_compression",
        "compressor": compressor,
        "requested_max_raw_gb": requested_max_raw_gb,
        "effective_max_raw_gb": max_raw_gb,
        "maximum_file_bytes": (
            streaming.MAX_FILE_BYTES
            if compressor == "applesauce"
            else MAX_NATIVE_FILE_BYTES
        ),
        "selected_paths": [str(path) for path in selected],
        "selected_logical_bytes": total,
        "actions": actions,
        "manifest_path": str(manifest),
        "allocated_bytes_reclaimed": sum(
            row["allocated_bytes_reclaimed"] for row in actions
        ),
        "direct_sqlite_readability_preserved": True,
        "source_records_deleted": False,
    }
