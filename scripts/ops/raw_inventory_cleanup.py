#!/usr/bin/env python3
"""Frozen-inventory verification and bounded historical placeholder retirement.

Nonempty duplicates retain both lookup paths through a verified .payload in
deep-cold custody. No scheduler or automatic payload deletion is provided.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import verified_duplicate_cleanup as native
from scripts.ops.long_runtime_common import iso_now

safety = native.safety
SOURCE_ROOT = Path("/Volumes/BOT_LOGS/schwab_trading_bot")
REVIEWED_AUDIT_SHA256 = (
    "170be3859a0882ad9e1d22a0c8a5d8565abdee0f90c602aea094a1584fca3e35"
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
MAX_AUDIT_BYTES = 2 * 1024**2
MAX_FILES = 414
MAX_SECONDS = 1800
MAX_VERIFY_BYTES = 64 * 1024**3
OWNER = "raw_inventory_cleanup"
PLACEHOLDER = re.compile(
    r"(?:local_fallback_storage/)?governance/(?:"
    r"channels/decision/[a-z0-9_]+/decision_|"
    r"shadow_[a-z0-9_]+/master_control_)(\d{8})\.jsonl$"
)
RESERVE_BYTES = 32 * 1024**3
PAYLOAD_REL = Path("data/deep_cold/reviewed_raw_inventory")


def lexical_path(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != str(value):
        raise ValueError("absolute_canonical_path_required")
    # Reject lexically, before any stat, resolve, or storage probe.
    if (
        str(path).casefold().startswith("/volumes/video/")
        or str(path).casefold() == "/volumes/video"
    ):
        raise ValueError("protected_volume")
    return path


def source_path(value):
    path = lexical_path(value)
    if not path.is_relative_to(SOURCE_ROOT) or path.suffix != ".jsonl":
        raise ValueError("outside_reviewed_source_boundary")
    return path


def identity_info(info):
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError("regular_single_link_file_required")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@contextmanager
def anchored_parent(path):
    """Walk with dirfds and O_NOFOLLOW, so a swapped ancestor cannot redirect IO."""
    path = lexical_path(str(path))
    safety.allowed(path.parent)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def read_audit(path):
    path = lexical_path(str(path))
    safety.allowed(path)
    with anchored_parent(path) as parent:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        with os.fdopen(fd, "rb") as handle:
            before = identity_info(os.fstat(handle.fileno()))
            if before[2] > MAX_AUDIT_BYTES:
                raise ValueError("audit_too_large")
            raw = handle.read(MAX_AUDIT_BYTES + 1)
            if identity_info(os.fstat(handle.fileno())) != before:
                raise RuntimeError("audit_changed")
    if hashlib.sha256(raw).hexdigest() != REVIEWED_AUDIT_SHA256:
        raise ValueError("audit_not_exact_reviewed_inventory")
    audit = json.loads(raw)
    rows = audit["files"]
    groups = audit["summary"]["matching_sample_groups"]
    if len(rows) != 748 or len(groups) != 36:
        raise ValueError("reviewed_inventory_shape_changed")
    files = {}
    for row in rows:
        path = str(source_path(row["path"]))
        if path in files:
            raise ValueError("duplicate_inventory_path")
        if row["status"] != "present" or row["stable_during_read"] is not True:
            raise ValueError("unverified_inventory_identity")
        files[path] = row
    empties = [row for row in rows if row["current_bytes"] == 0]
    if len(empties) != 342:
        raise ValueError("reviewed_empty_count_changed")
    seen = set()
    for group in groups:
        paths = group["paths"]
        if len(paths) != 2 or len(set(paths)) != 2:
            raise ValueError("invalid_duplicate_group")
        for value in paths:
            source_path(value)
            if (
                value not in files
                or value in seen
                or files[value]["current_bytes"] != group["bytes_each"]
            ):
                raise ValueError("duplicate_group_not_bound_to_inventory")
            seen.add(value)
    return files, empties, groups


def frozen_identity(row):
    path = source_path(row["path"])
    actual = native.identity(path)
    stamp = datetime.fromtimestamp(path.lstat().st_mtime, timezone.utc).isoformat()
    if (
        actual[:3] != (row["device"], row["inode"], row["current_bytes"])
        or stamp != row["mtime_utc"]
    ):
        raise RuntimeError("source_changed_since_frozen_inventory")
    if native.identity(path) != actual:
        raise RuntimeError("source_changed_during_identity_check")
    return actual


def hash_source(row, budget):
    budget.check()
    path = source_path(row["path"])
    before = frozen_identity(row)
    if before[2] > budget.remaining:
        raise safety.Deferred("verification_byte_budget")
    safety.idle(path)
    budget.check()
    with anchored_parent(path) as parent:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        with os.fdopen(fd, "rb") as handle:
            if identity_info(os.fstat(handle.fileno())) != before:
                raise RuntimeError("source_changed_before_read")
            digest, count = hashlib.sha256(), 0
            while True:
                budget.check()
                chunk = handle.read(min(native.CHUNK, before[2] - count + 1))
                if not chunk:
                    break
                budget.consume(len(chunk))
                count += len(chunk)
                if count > before[2]:
                    raise RuntimeError("source_grew_during_read")
                digest.update(chunk)
            if identity_info(os.fstat(handle.fileno())) != before or count != before[2]:
                raise RuntimeError("source_changed_during_read")
    safety.idle(path)
    budget.check()
    if native.identity(path) != before:
        raise RuntimeError("source_changed_after_read")
    return {
        "path": str(path),
        "identity": list(before),
        "sha256": digest.hexdigest(),
        "verified_bytes": count,
        "full_stream_read": True,
        "handle_probes": "idle_before_and_after",
        "observed_at": iso_now(),
    }


def verify_pair(rows, budget):
    if sum(row["current_bytes"] for row in rows) > budget.remaining:
        raise safety.Deferred("verification_byte_budget")
    proof = [hash_source(row, budget) for row in rows]
    for item in proof:
        if list(native.identity(Path(item["path"]))) != item["identity"]:
            raise RuntimeError("pair_changed_during_verification")
    if proof[0]["identity"][:2] == proof[1]["identity"][:2]:
        raise RuntimeError("same_inode_not_independent_duplicate")
    if (
        proof[0]["sha256"] != proof[1]["sha256"]
        or proof[0]["verified_bytes"] != proof[1]["verified_bytes"]
    ):
        raise RuntimeError("full_content_mismatch")
    return proof


def empty_blocker(row, now=None, *, min_age_seconds=7 * 86400):
    path = source_path(row["path"])
    relative = path.relative_to(SOURCE_ROOT)
    parts = set(relative.parts)
    if "manifest" in path.name:
        return "retention_or_archive_custody_manifest"
    if (
        "independent_fill_inbox" in parts
        or "current" in path.stem
        or "latest" in path.stem
    ):
        return "active_inbox_or_pointer"
    if parts & {
        "cold_archive",
        "cold_archives",
        "deep_cold",
        "stale_stage",
        "quarantine",
        "manifest_backed",
    }:
        return "manifest_owned_or_retention_locked_archive"
    if "execution_lanes" in parts or "trade_logs" in parts:
        return "durable_execution_or_financial_evidence"
    match = PLACEHOLDER.fullmatch(relative.as_posix())
    if not match:
        return "unassigned_empty_retirement_owner"
    now = now or datetime.now(timezone.utc)
    date = datetime.strptime(match[1], "%Y%m%d").replace(tzinfo=timezone.utc)
    stamp = datetime.fromisoformat(row["mtime_utc"])
    if (
        stamp.tzinfo is None
        or (now - date).total_seconds() < min_age_seconds
        or (now - stamp).total_seconds() < min_age_seconds
    ):
        return "not_closed_historical_placeholder"
    if row["current_bytes"] != 0 or row.get("full_sha256") != EMPTY_SHA256:
        return "not_verified_empty_in_frozen_inventory"
    if set(row.get("inventory_blockers", [])) - {"empty_source"}:
        return "inventory_owner_blocker"
    return ""


@contextmanager
def owner_locks(root):
    """Use the existing retention and storage maintenance singleton paths."""
    handles = []
    try:
        for name in ("data_retention.lock", "storage_maintenance.lock"):
            path = safety.allowed(root / "governance/locks" / name, missing=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            safety.allowed(path.parent)
            with anchored_parent(path) as parent:
                fd = os.open(
                    path.name,
                    os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent,
                )
            handle = os.fdopen(fd, "a+")
            handles.append((path, handle, identity_info(os.fstat(fd))[:2]))
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise safety.Deferred(f"{name}_busy") from exc

        def check():
            for path, handle, anchor in handles:
                if (
                    native.identity(path)[:2] != anchor
                    or identity_info(os.fstat(handle.fileno()))[:2] != anchor
                ):
                    raise safety.Deferred("owner_lock_anchor_changed")

        check()
        yield check
    finally:
        for _, handle, _ in reversed(handles):
            handle.close()


def retire_empty(root, row, budget, check_locks, *, audit):
    files, _, _ = read_audit(audit)
    if files.get(row["path"]) != row:
        raise ValueError("source_not_exact_reviewed_row")
    blocker = empty_blocker(row)
    if blocker:
        raise safety.Deferred(blocker)
    check_locks()
    proof = hash_source(row, budget)
    if proof["verified_bytes"] != 0 or proof["sha256"] != EMPTY_SHA256:
        raise RuntimeError("empty_proof_required")
    path = source_path(row["path"])
    info = path.lstat()
    record = {
        "owner": OWNER,
        "audit_sha256": REVIEWED_AUDIT_SHA256,
        "source": str(path),
        "source_root": str(SOURCE_ROOT),
        "operation": "retire_reviewed_closed_empty_placeholder",
        "proof": proof,
        "frozen_source": row,
        "restore": {
            "content_hex": "",
            "mode": stat.S_IMODE(info.st_mode),
            "uid": info.st_uid,
            "gid": info.st_gid,
            "mtime_ns": info.st_mtime_ns,
            "atime_ns": info.st_atime_ns,
        },
    }
    # Durable native write-ahead tombstone is mandatory before the sole unlink.
    native.receipt(root, {"phase": "empty_retirement_prepared", **record})
    safety.idle(path)
    check_locks()
    budget.check()
    if empty_blocker(row):
        raise safety.Deferred("placeholder_protection_changed")
    with anchored_parent(path) as parent:
        current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        held = os.fstat(parent)
        visible = path.parent.stat()
        if (held.st_dev, held.st_ino) != (visible.st_dev, visible.st_ino):
            raise RuntimeError("parent_changed_before_retirement")
        if (
            list(identity_info(current)) != proof["identity"]
            or list(native.identity(path)) != proof["identity"]
        ):
            raise RuntimeError("source_changed_before_retirement")
        os.unlink(path.name, dir_fd=parent)
        errors = []
        try:
            os.fsync(parent)
        except OSError as exc:
            errors.append(f"directory_fsync:{exc}")
        try:
            native.receipt(root, {"phase": "empty_retired", **record})
        except (OSError, RuntimeError) as exc:
            errors.append(f"completion_receipt:{exc}")
    return {
        "source_removed": True,
        "released_payload_bytes": 0,
        "persistence_complete": not errors,
        "persistence_errors": errors,
        "proof": proof,
    }


def require_reserve(extra=0):
    safety.allowed(SOURCE_ROOT)
    if shutil.disk_usage(SOURCE_ROOT).free - extra < RESERVE_BYTES:
        raise safety.Deferred("external_32gib_reserve_required")


def hash_payload(path, budget):
    before = native.identity(path)
    if before[2] > budget.remaining:
        raise safety.Deferred("verification_byte_budget")
    digest, count = hashlib.sha256(), 0
    with anchored_parent(path) as parent:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            if identity_info(os.fstat(stream.fileno())) != before:
                raise RuntimeError("payload_changed_before_read")
            while True:
                budget.check()
                chunk = stream.read(min(native.CHUNK, before[2] - count + 1))
                if not chunk:
                    break
                budget.consume(len(chunk))
                count += len(chunk)
                if count > before[2]:
                    raise RuntimeError("payload_grew_during_read")
                digest.update(chunk)
            if identity_info(os.fstat(stream.fileno())) != before:
                raise RuntimeError("payload_changed_during_read")
    if count != before[2] or native.identity(path) != before:
        raise RuntimeError("payload_changed_after_read")
    return {
        "identity": list(before),
        "sha256": digest.hexdigest(),
        "verified_bytes": count,
    }


def custody_receipt(target, record):
    """Same fsynced pre-replacement receipt contract as the deep-cold owner."""
    path = safety.allowed(
        target.with_name(target.name + ".restore_proofs.receipt"), missing=True
    )
    with anchored_parent(path) as parent:
        fd = os.open(
            path.name,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        with os.fdopen(fd, "a") as handle:
            identity_info(os.fstat(handle.fileno()))
            handle.write(
                json.dumps({"timestamp_utc": iso_now(), **record}, sort_keys=True)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(parent)


def has_custody(target, rows, digest):
    path = target.with_name(target.name + ".restore_proofs.receipt")
    safety.allowed(path)
    with anchored_parent(path) as parent:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        with os.fdopen(fd, "rb") as handle:
            before = identity_info(os.fstat(handle.fileno()))
            if before[2] > MAX_AUDIT_BYTES:
                raise RuntimeError("custody_receipt_read_budget")
            records = [
                json.loads(line)
                for line in handle.read(MAX_AUDIT_BYTES + 1).splitlines()
            ]
            if identity_info(os.fstat(handle.fileno())) != before:
                raise RuntimeError("custody_receipt_changed")
    for record in records:
        if (
            record.get("owner") == OWNER
            and record.get("audit_sha256") == REVIEWED_AUDIT_SHA256
            and record.get("phase") == "verified_before_atomic_source_replacement"
            and record.get("canonical") == str(target)
            and record.get("sha256") == digest
            and record.get("paths") == [r["path"] for r in rows]
            and record.get("original_sources") == rows
        ):
            return True
    return False


def publish_payload(source, expected, target, digest, budget, check_locks):
    """Bounded version of deep-cold copy/verify/no-clobber publication semantics."""
    size = expected[2]
    require_reserve(size)
    safety.allowed(target.parent)
    if target.parent.stat().st_dev != expected[0]:
        raise safety.Deferred("canonical_must_share_source_filesystem")
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.building")
    created = False
    created_identity = None
    published = False
    try:
        with anchored_parent(source) as parent:
            fd = os.open(source.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(fd, "rb") as src:
                if identity_info(os.fstat(src.fileno())) != tuple(expected):
                    raise RuntimeError("source_changed_before_copy")
                with anchored_parent(temporary) as output_parent:
                    output_fd = os.open(
                        temporary.name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=output_parent,
                    )
                created = True
                created_identity = identity_info(os.fstat(output_fd))[:2]
                with os.fdopen(output_fd, "wb") as dst:
                    copied = 0
                    while True:
                        budget.check()
                        check_locks()
                        chunk = src.read(min(native.CHUNK, size - copied + 1))
                        if not chunk:
                            break
                        budget.consume(len(chunk))
                        copied += len(chunk)
                        if copied > size:
                            raise RuntimeError("source_grew_during_copy")
                        require_reserve(len(chunk))
                        dst.write(chunk)
                    dst.flush()
                    os.fsync(dst.fileno())
                if copied != size or identity_info(os.fstat(src.fileno())) != tuple(
                    expected
                ):
                    raise RuntimeError("source_changed_during_copy")
        proof = hash_payload(temporary, budget)
        if (
            proof["sha256"] != digest
            or proof["verified_bytes"] != size
            or native.identity(source) != tuple(expected)
        ):
            raise RuntimeError("canonical_copy_verification_failed")
        budget.check()
        check_locks()
        require_reserve()
        # The native publication owner uses this same exclusive hard-link method.
        # Unsupported filesystems defer; no unbounded copy fallback is allowed.
        with anchored_parent(target) as parent:
            current = identity_info(
                os.stat(temporary.name, dir_fd=parent, follow_symlinks=False)
            )
            if list(current) != proof["identity"]:
                raise RuntimeError("temporary_changed_before_publication")
            held, visible = os.fstat(parent), target.parent.stat()
            if (held.st_dev, held.st_ino) != (visible.st_dev, visible.st_ino):
                raise RuntimeError("canonical_parent_changed_before_publication")
            os.link(
                temporary.name,
                target.name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )
            published = True
            os.unlink(temporary.name, dir_fd=parent)
        created = False
        proof["identity"] = list(native.identity(target))
        safety.sync_dir(target.parent)
        return proof
    except (OSError, RuntimeError) as exc:
        if published:
            return {**proof, "publication_persistence_errors": [str(exc)]}
        raise
    finally:
        if created:
            safety.allowed(temporary, missing=True)
            try:
                with anchored_parent(temporary) as parent:
                    current = os.stat(
                        temporary.name, dir_fd=parent, follow_symlinks=False
                    )
                    if (
                        stat.S_ISREG(current.st_mode)
                        and (current.st_dev, current.st_ino) == created_identity
                    ):
                        os.unlink(temporary.name, dir_fd=parent)
            except FileNotFoundError:
                pass


def replace_alias(path, target, expected, canonical_identity, budget, check_locks):
    safety.idle(path)
    safety.idle(target)
    budget.check()
    check_locks()
    require_reserve()
    with anchored_parent(path) as parent:
        link = f".{path.name}.offload_link_{uuid.uuid4().hex}"
        os.symlink(str(target), link, dir_fd=parent)
        try:
            if native.identity(path) != tuple(expected) or native.identity(
                target
            ) != tuple(canonical_identity):
                raise RuntimeError("source_or_payload_changed_before_alias")
            if identity_info(
                os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            ) != tuple(expected):
                raise RuntimeError("source_parent_changed_before_alias")
            held, visible = os.fstat(parent), path.parent.stat()
            if (held.st_dev, held.st_ino) != (visible.st_dev, visible.st_ino):
                raise RuntimeError("source_parent_changed_before_alias")
            os.replace(link, path.name, src_dir_fd=parent, dst_dir_fd=parent)
            try:
                os.fsync(parent)
            except OSError as exc:
                return [f"alias_directory_fsync:{exc}"]
        finally:
            try:
                os.unlink(link, dir_fd=parent)
            except FileNotFoundError:
                pass
    return []


def consolidate_pair(root, rows, budget, check_locks, *, audit):
    files, _, groups = read_audit(audit)
    paths = [row["path"] for row in rows]
    if (
        len(rows) != 2
        or any(files.get(r["path"]) != r for r in rows)
        or paths not in [g["paths"] for g in groups]
    ):
        raise ValueError("pair_not_exact_reviewed_group")
    if any(
        not PLACEHOLDER.fullmatch(source_path(p).relative_to(SOURCE_ROOT).as_posix())
        for p in paths
    ):
        raise safety.Deferred("raw_duplicate_owner_boundary")
    if any(
        not p.startswith(str(SOURCE_ROOT / "local_fallback_storage/governance") + "/")
        for p in paths
    ):
        raise safety.Deferred("raw_duplicate_owner_boundary")
    for row in rows:
        # Same closed-date gate as placeholders, while retaining every evidence byte.
        closed = {**row, "current_bytes": 0, "full_sha256": EMPTY_SHA256}
        if empty_blocker(closed, min_age_seconds=86400):
            raise safety.Deferred("duplicate_not_closed_or_inventory_blocked")
    size = rows[0]["current_bytes"]
    if size <= 0 or rows[1]["current_bytes"] != size:
        raise ValueError("nonempty_equal_size_pair_required")
    payload_root = SOURCE_ROOT / PAYLOAD_REL
    aliases, ordinary = {}, {}
    digest = None
    for row in rows:
        path = source_path(row["path"])
        safety.allowed(path.parent)
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            target = lexical_path(os.readlink(path))
            if target.parent != payload_root or not re.fullmatch(
                r"[0-9a-f]{64}\.payload", target.name
            ):
                raise RuntimeError("unrecognized_source_alias")
            if digest is not None and digest != target.stem:
                raise RuntimeError("aliases_disagree")
            digest = target.stem
            aliases[str(path)] = (
                info.st_dev,
                info.st_ino,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
        else:
            ordinary[str(path)] = frozen_identity(row)
    require_reserve()
    # Charge all read/copy passes before starting; byte budgets are per invocation.
    needed = size * (len(ordinary) + (2 if not aliases else 1))
    if needed > budget.remaining:
        raise safety.Deferred("verification_byte_budget")
    proofs = [hash_source(row, budget) for row in rows if row["path"] in ordinary]
    if proofs:
        hashes = {p["sha256"] for p in proofs}
        if len(hashes) != 1 or (digest is not None and digest not in hashes):
            raise RuntimeError("full_content_mismatch")
        digest = proofs[0]["sha256"]
    target = payload_root / f"{digest}.payload"
    if aliases and not has_custody(target, rows, digest):
        raise RuntimeError("source_alias_requires_matching_custody_receipt")
    safety.allowed(payload_root, missing=True)
    payload_root.mkdir(parents=True, exist_ok=True)
    safety.allowed(payload_root)
    safety.sync_dir(payload_root.parent)
    check_locks()
    canonical_created = not safety.allowed(target, missing=True).exists()
    if not canonical_created:
        safety.idle(target)
        canonical = hash_payload(target, budget)
        if canonical["sha256"] != digest or canonical["verified_bytes"] != size:
            raise RuntimeError("existing_canonical_content_mismatch")
    else:
        if not ordinary:
            raise RuntimeError("canonical_missing_without_raw_source")
        source = Path(next(iter(ordinary)))
        canonical = publish_payload(
            source, ordinary[str(source)], target, digest, budget, check_locks
        )
    record = {
        "owner": OWNER,
        "audit_sha256": REVIEWED_AUDIT_SHA256,
        "phase": "verified_before_atomic_source_replacement",
        "paths": paths,
        "original_sources": rows,
        "canonical": str(target),
        "sha256": digest,
        "canonical_proof": canonical,
        "raw_proofs": proofs,
        "retention": "retained_deep_cold_payload_no_automatic_delete",
    }
    result = {
        "canonical": str(target),
        "sha256": digest,
        "canonical_proof": canonical,
        "canonical_created": canonical_created,
        "aliases_replaced": [],
        "already_linked": list(aliases),
        "source_removed": False,
        "released_payload_bytes": 0,
        "persistence_complete": True,
        "persistence_errors": [],
    }
    try:
        if canonical.get("publication_persistence_errors"):
            raise RuntimeError(
                "canonical_publication_persistence_failed:"
                + ";".join(canonical["publication_persistence_errors"])
            )
        if canonical["identity"][0] != SOURCE_ROOT.stat().st_dev:
            raise RuntimeError("canonical_filesystem_changed")
        # Complete target durability and custody before replacing even the first raw.
        with anchored_parent(target) as parent:
            fd = os.open(target.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(fd, "rb") as handle:
                if (
                    list(identity_info(os.fstat(handle.fileno())))
                    != canonical["identity"]
                ):
                    raise RuntimeError("canonical_changed_before_custody")
                os.fsync(handle.fileno())
        custody_receipt(target, record)
        native.receipt(root, record)
        for row in rows:
            path = source_path(row["path"])
            if row["path"] in aliases:
                info = path.lstat()
                if (
                    info.st_dev,
                    info.st_ino,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                ) != aliases[row["path"]] or os.readlink(path) != str(target):
                    raise RuntimeError("existing_alias_changed")
                continue
            errors = replace_alias(
                path,
                target,
                ordinary[str(path)],
                canonical["identity"],
                budget,
                check_locks,
            )
            result["aliases_replaced"].append(str(path))
            result["persistence_errors"].extend(errors)
            completed = {
                **record,
                "phase": "source_replaced_with_canonical_alias",
                "source": str(path),
            }
            custody_receipt(target, completed)
            native.receipt(root, completed)
            if errors:
                break
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        result["persistence_errors"].append(str(exc))
    result["released_payload_bytes"] = size * (
        len(result["aliases_replaced"]) - int(canonical_created)
    )
    result["space_accounting"] = "logical_net_payload_bytes_not_measured_free_space"
    result["persistence_complete"] = not result["persistence_errors"]
    result["retained_unaliased_payload"] = (
        not aliases and not result["aliases_replaced"]
    )
    if not result["persistence_complete"]:
        result["recovery_action"] = (
            "Rerun this exact reviewed pair; retain the payload and all remaining raw sources."
        )
        failure = {**record, "phase": "alias_transaction_incomplete", "outcome": result}
        # Independent native receipt survives failure of the payload-side receipt.
        try:
            native.receipt(root, failure)
        except (OSError, RuntimeError) as exc:
            result["persistence_errors"].append(f"failure_receipt:{exc}")
    return result


def run(
    audit,
    *,
    kind="empty",
    apply=False,
    start=0,
    limit=4,
    seconds=90,
    max_bytes=10 * 1024**3,
    root=PROJECT_ROOT,
):
    if kind not in {"empty", "duplicates"} or type(start) is not int or start < 0:
        raise ValueError("invalid_selection")
    if type(limit) is not int or not 1 <= limit <= MAX_FILES:
        raise ValueError("invalid_file_limit")
    if (
        not math.isfinite(seconds)
        or not 0 < seconds <= MAX_SECONDS
        or type(max_bytes) is not int
        or not 0 < max_bytes <= MAX_VERIFY_BYTES
    ):
        raise ValueError("invalid_verification_budget")
    files, empties, groups = read_audit(audit)
    candidates = empties if kind == "empty" else groups
    selected = candidates[start : start + limit]
    budget = native.Budget(seconds=seconds, max_bytes=max_bytes)
    results = []
    report = {
        "owner": OWNER,
        "audit_sha256": REVIEWED_AUDIT_SHA256,
        "timestamp_utc": iso_now(),
        "mode": "apply" if apply else "verify_only",
        "kind": kind,
        "start": start,
        "selected": len(selected),
        "total_candidates": len(candidates),
        "results": results,
        "source_files_removed": 0,
        "released_payload_bytes": 0,
        "raw_files_replaced_with_aliases": 0,
        "duplicate_replacement_authority": "exact_reviewed_pairs_to_retained_payload_only",
    }

    def process(check_locks=lambda: None):
        for item in selected:
            result = {
                "paths": [item["path"]] if kind == "empty" else item["paths"],
                "source_removed": False,
            }
            results.append(result)
            try:
                budget.check()
                if kind == "duplicates":
                    if apply:
                        result.update(
                            consolidate_pair(
                                root,
                                [files[p] for p in item["paths"]],
                                budget,
                                check_locks,
                                audit=audit,
                            ),
                            status="consolidated",
                        )
                        report["released_payload_bytes"] += result[
                            "released_payload_bytes"
                        ]
                        report["raw_files_replaced_with_aliases"] += len(
                            result["aliases_replaced"]
                        )
                        if not result["persistence_complete"]:
                            result["status"] = "partial"
                            report["stopped_reason"] = "alias_transaction_incomplete"
                            break
                    else:
                        result["proof"] = verify_pair(
                            [files[p] for p in item["paths"]], budget
                        )
                        result["status"] = "full_sha256_match"
                else:
                    blocker = empty_blocker(item)
                    if blocker:
                        result.update(status="preserved", blocker=blocker)
                    elif apply:
                        result.update(
                            retire_empty(root, item, budget, check_locks, audit=audit),
                            status="retired",
                        )
                        report["source_files_removed"] += int(result["source_removed"])
                        if not result["persistence_complete"]:
                            report["stopped_reason"] = (
                                "post_retirement_persistence_failure"
                            )
                            break
                    else:
                        result.update(
                            status="verified_empty", proof=hash_source(item, budget)
                        )
            except (
                OSError,
                RuntimeError,
                ValueError,
                subprocess.SubprocessError,
            ) as exc:
                result.update(
                    status="deferred" if isinstance(exc, safety.Deferred) else "failed",
                    blocker=str(exc),
                )
                if apply:
                    report["stopped_reason"] = "apply_failed_closed"
                    break

    if apply:
        root = lexical_path(str(root))
        safety.allowed(root)
        ready, reason = safety.admission(root)
        if not ready:
            raise safety.Deferred(reason)
        with owner_locks(root) as check_locks:
            budget.guard = safety.Guard(root, seconds)
            process(check_locks)
    else:
        process()
    report["verification_bytes_read"] = max_bytes - budget.remaining
    report["processed"] = len(results)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="Default; no receipts, locks, or source writes",
    )
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--kind", choices=("empty", "duplicates"), default="empty")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--max-bytes", type=int, default=10 * 1024**3)
    args = parser.parse_args()
    try:
        report = run(
            args.audit,
            kind=args.kind,
            apply=args.apply,
            start=args.start,
            limit=args.limit,
            seconds=args.seconds,
            max_bytes=args.max_bytes,
        )
        print(json.dumps(report, indent=2))
        return (
            2
            if report.get("stopped_reason")
            or any(r["status"] in {"failed", "deferred"} for r in report["results"])
            else 0
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"owner": OWNER, "status": "blocked", "blocker": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
