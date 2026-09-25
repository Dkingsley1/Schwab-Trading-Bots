"""Bounded online candle cache; immutable captures are not an audit archive."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from core.decision_price_evidence import digest, timestamp
from core.storage_router import inspect_storage_path

DIRECTORY = Path("governance/rehearsals/schd")
MAX_BYTES = 6 * 1024 * 1024
MAX_FILES = 128
MAX_TOTAL_BYTES = 64 * 1024 * 1024
RETENTION_SECONDS = 1800


def checked(root, path):
    route = inspect_storage_path(path, boundary_root=root, allow_external=False)
    if route.get("status") not in {"present", "missing"} or route.get("symlinks"):
        raise ValueError("unsafe_schd_capture_route")
    return Path(path)


def read_capture(root, path):
    path = checked(root, path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
            raise ValueError("bounded_regular_schd_capture_required")
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("schd_capture_size_exceeded")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("schd_capture_object_required")
    digest(value)  # Reject NaN and infinity before hashing or publishing evidence.
    return value


def latest_capture(root):
    path = checked(root, Path(root) / DIRECTORY / "market_latest.json")
    return read_capture(root, path) if path.exists() else {}


def decision_capture(root, row):
    context = row.get("metadata", {}).get("schd_candle_context", {})
    identity = context.get("capture_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", str(identity)):
        # Diagnostic context only; native_packet will reject the missing binding.
        return latest_capture(root)
    path = checked(root, Path(root) / DIRECTORY / "captures" / f"{identity}.json")
    if path.exists():
        value = read_capture(root, path)
    else:
        value = latest_capture(root)
    if not value or digest(value) != identity:
        return {}
    return value


def _inventory(root):
    directory = checked(root, Path(root) / DIRECTORY / "captures")
    if not directory.exists():
        return []
    paths = []
    with os.scandir(directory) as entries:
        for entry in entries:
            paths.append(checked(root, directory / entry.name))
            if len(paths) > MAX_FILES:
                raise ValueError("schd_capture_file_budget_exceeded")
    return paths


def preserve_capture(root, value):
    """Caller holds writer.lock; publish complete bytes without replacing a hash."""
    identity = digest(value)
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("schd_capture_size_exceeded")
    directory = checked(root, Path(root) / DIRECTORY / "captures")
    directory.mkdir(parents=True, exist_ok=True)
    path = checked(root, directory / f"{identity}.json")
    if path.exists():
        if digest(read_capture(root, path)) != identity:
            raise ValueError("schd_capture_hash_collision_or_corruption")
        return identity
    files = _inventory(root)
    if (
        len(files) >= MAX_FILES
        or sum(p.lstat().st_size for p in files) + len(raw) > MAX_TOTAL_BYTES
    ):
        raise ValueError("schd_capture_cache_budget_exceeded")
    fd, temporary = tempfile.mkstemp(prefix=".capture-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        os.unlink(temporary)
    return identity


def publish_capture(root, value):
    from scripts.ops.long_runtime_common import write_text_atomic

    previous = latest_capture(root)
    if previous:
        preserve_capture(root, previous)
    identity = preserve_capture(root, value)
    write_text_atomic(
        checked(root, Path(root) / DIRECTORY / "market_latest.json"),
        json.dumps(value, sort_keys=True, allow_nan=False),
    )
    return identity


def prune_captures(root, *, now=None):
    """Delete only verified expired cache entries, never latest or audit records."""
    now = now or datetime.now(timezone.utc)
    latest = latest_capture(root)
    protected = digest(latest) if latest else ""
    deleted, skipped, bytes_read = [], [], 0
    for path in sorted(_inventory(root)):
        named_capture = bool(re.fullmatch(r"[0-9a-f]{64}\.json", path.name))
        abandoned_build = bool(re.fullmatch(r"\.capture-[a-z0-9_]{8}", path.name))
        if not (named_capture or abandoned_build) or path.stem == protected:
            continue
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            skipped.append(path.name)
            continue
        if len(deleted) >= 8 or bytes_read + before.st_size > 8 * 1024 * 1024:
            break
        try:
            value = read_capture(root, path)
            bytes_read += before.st_size
            source = value["source"]
            observed = timestamp(source["fetch_started_at_utc"])
            if (
                (named_capture and digest(value) != path.stem)
                or source.get("provider") != "schwab"
                or source.get("symbol") != "SCHD"
            ):
                skipped.append(path.name)
                continue
            if (now - observed).total_seconds() <= RETENTION_SECONDS or (
                abandoned_build
                and now.timestamp() - before.st_mtime <= RETENTION_SECONDS
            ):
                continue
            after = checked(root, path).lstat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                skipped.append(path.name)
                continue
            path.unlink()
            deleted.append(path.name)
        except (OSError, ValueError, KeyError, TypeError):
            skipped.append(path.name)
    return {
        "deleted": deleted,
        "skipped": skipped,
        "bytes_verified": bytes_read,
        "scope": "expired_online_candle_cache_only_not_decisions_or_audit_history",
    }
