"""Bounded full-content proof for retiring inactive raw/gzip duplicates."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from scripts.ops import cold_evidence_compactor as safety
from scripts.ops.long_runtime_common import iso_now

CHUNK = 1024 * 1024
RETAINED_DIRECTORIES = {
    "exports",
    "cold_archive",
    "cold_archives",
    "deep_cold",
    "quarantine",
    "training",
}


class Budget:
    def __init__(self, seconds=90, max_bytes=1024**3, guard=None):
        self.deadline = time.monotonic() + seconds
        self.remaining = int(max_bytes)
        self.guard = guard

    def check(self):
        if time.monotonic() >= self.deadline:
            raise safety.Deferred("verification_deadline")
        if self.guard is not None:
            self.guard.check()

    def consume(self, count):
        self.check()
        self.remaining -= count
        if self.remaining < 0:
            raise safety.Deferred("verification_byte_budget")


class MeteredReader:
    def __init__(self, stream, budget):
        self.stream, self.budget = stream, budget

    def read(self, size=-1):
        self.budget.check()
        data = self.stream.read(CHUNK if size < 0 else min(size, CHUNK))
        self.budget.consume(len(data))
        return data


def identity(path):
    safety.allowed(path)
    info = Path(path).lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError("regular_single_link_file_required")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def require_log_source(root, path):
    """SQL payload retirement belongs to its writer-aware compaction owner."""
    root, path = Path(os.path.abspath(root)), Path(os.path.abspath(path))
    safety.allowed(root)
    safety.allowed(path)
    if not path.is_relative_to(root) or path.relative_to(root).parts[:1] != ("logs",):
        raise safety.Deferred("sql_or_unknown_retirement_owner_required")


def inventory(root, *, seconds=15, max_entries=20000):
    """Do not follow directory links, including protected external aliases."""
    root = safety.allowed(Path(root))
    deadline = time.monotonic() + seconds
    pending, paths, count = [root], [], 0
    while pending:
        directory = pending.pop()
        safety.allowed(directory)
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                if count > max_entries or time.monotonic() >= deadline:
                    raise safety.Deferred("inventory_budget_exhausted")
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if (
                        entry.name not in RETAINED_DIRECTORIES
                        and not entry.name.startswith(".")
                    ):
                        pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    paths.append(Path(entry.path))
    return sorted(paths)


def verify_pair(raw, archive, budget=None):
    budget = budget or Budget()
    raw, archive = Path(raw), Path(archive)
    before = (identity(raw), identity(archive))
    size = before[0][2]
    if size <= 0 or before[1][2] <= 0:
        raise RuntimeError("empty_pair")
    if size * 2 + before[1][2] > budget.remaining:
        raise safety.Deferred("verification_byte_budget")
    digests, counts = [], []
    for path, expected, compressed in (
        (raw, before[0], False),
        (archive, before[1], True),
    ):
        budget.check()
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as source:
            opened = os.fstat(source.fileno())
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            ) != expected or opened.st_nlink != 1:
                raise RuntimeError("pair_changed_before_read")
            stream = (
                gzip.GzipFile(fileobj=MeteredReader(source, budget))
                if compressed
                else source
            )
            digest, count = hashlib.sha256(), 0
            try:
                while True:
                    budget.check()
                    data = stream.read(min(CHUNK, size - count + 1))
                    if not data:
                        break
                    count += len(data)
                    budget.consume(len(data))
                    if count > size:
                        raise RuntimeError("restored_length_mismatch")
                    digest.update(data)
            finally:
                if compressed:
                    stream.close()
            digests.append(digest.hexdigest())
            counts.append(count)
    if before != (identity(raw), identity(archive)):
        raise RuntimeError("pair_changed_during_verification")
    if counts != [size, size] or digests[0] != digests[1]:
        raise RuntimeError("full_content_mismatch")
    return {
        "ok": True,
        "state": "full_sha256_match",
        "sha256": digests[0],
        "verified_bytes": size,
        "raw_identity": before[0],
        "archive_identity": before[1],
    }


def receipt(root, payload):
    folder = safety.allowed(Path(root) / "governance/storage_recovery", missing=True)
    folder.mkdir(parents=True, exist_ok=True)
    path = safety.allowed(folder / "verified_duplicate_cleanup.jsonl", missing=True)
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600),
        "a",
    ) as handle:
        handle.write(
            json.dumps({"timestamp_utc": iso_now(), **payload}, sort_keys=True) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    safety.sync_dir(folder)


def remove_pair(root, raw, archive, budget, *, expected=None, source_root=None):
    raw, archive = Path(raw), Path(archive)
    require_log_source(source_root or root, raw)
    before = (identity(raw), identity(archive))
    if expected is not None and before != expected:
        raise RuntimeError("pair_changed_since_scan")
    safety.idle(raw)
    safety.idle(archive)
    proof = verify_pair(raw, archive, budget)
    if (proof["raw_identity"], proof["archive_identity"]) != before:
        raise RuntimeError("pair_changed_during_idle_probe")
    safety.idle(raw)
    safety.idle(archive)
    budget.check()
    with os.fdopen(os.open(archive, os.O_RDONLY | os.O_NOFOLLOW), "rb") as retained:
        os.fsync(retained.fileno())
    record = {"source": str(raw), "archive": str(archive), **proof}
    receipt(root, {"phase": "verified_before_release", **record})
    budget.check()
    parent_fd = os.open(
        safety.allowed(raw.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    persistence_errors = []
    removed = False
    try:
        if (
            identity(raw) != proof["raw_identity"]
            or identity(archive) != proof["archive_identity"]
        ):
            raise RuntimeError("pair_changed_before_release")
        held = os.fstat(parent_fd)
        current = raw.parent.stat()
        if (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError("parent_changed_before_release")
        os.unlink(raw.name, dir_fd=parent_fd)
        removed = True
        try:
            os.fsync(parent_fd)
        except OSError as exc:
            persistence_errors.append(f"release_directory_fsync:{exc}")
    finally:
        try:
            os.close(parent_fd)
        except OSError as exc:
            if not removed:
                raise
            persistence_errors.append(f"release_directory_close:{exc}")
    try:
        receipt(root, {"phase": "source_released", **record})
    except (OSError, RuntimeError) as exc:
        persistence_errors.append(f"release_receipt:{exc}")
    return {
        **proof,
        "source_removed": True,
        "release_persistence_complete": not persistence_errors,
        "persistence_errors": persistence_errors,
    }
