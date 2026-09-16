"""Bounded metadata-only organization of retained archives; never changes data paths."""

from __future__ import annotations

import csv
import fcntl
import html
import io
import json
import math
import os
import re
import stat
import time
import uuid
from collections import defaultdict
from datetime import date
from pathlib import Path
from urllib.parse import quote

from core.storage_router import inspect_storage_path
from scripts.ops.long_runtime_common import iso_now

OUTPUTS = ("ARCHIVE_INDEX.md", "ARCHIVE_CATALOG.csv", "ARCHIVE_CATALOG.json")
MANIFEST = "cold_archive_compaction_manifest.jsonl"
LOCK_NAME = ".archive_catalog.flock"
DATE = re.compile(r"(?<!\d)(20\d{2})[-_]?(\d{2})[-_]?(\d{2})(?!\d)")
HASH = re.compile(r"[0-9a-f]{64}")
GIB = 1024**3


def _label(value: str) -> str:
    value = html.escape(value).replace("\n", " ").replace("\r", " ")
    for character in ("\\", "[", "]", "|", "`", "*", "_"):
        value = value.replace(character, "\\" + character)
    return value


def _filename_date(relative: Path) -> str:
    for part in reversed(relative.parts):
        for match in DATE.finditer(part):
            try:
                return date(*(int(value) for value in match.groups())).isoformat()
            except ValueError:
                continue
    return "undated"


def _dataset(relative: Path) -> str:
    from scripts.ops.cold_archive_compactor import _archive_data_family

    parts = relative.parts
    if len(parts) >= 4 and parts[:2] == ("sql_link_shards", "archives"):
        return f"sql_link_{parts[2]}"
    if parts[0] == "sql_link_primary":
        return "sql_link_primary"
    if parts[0].startswith("local_primary_legacy_"):
        return "sql_link_primary_legacy"
    family = _archive_data_family(relative.as_posix())
    return "failover_backups" if family == "verified_failover_backups" else family


def _lifecycle(relative: Path, kind: str) -> str:
    parts = relative.parts
    if any(
        "partial_dataset" in part or "filesystem_compaction_" in part for part in parts
    ):
        return "incomplete_maintenance"
    if (
        any(
            part.lower() in {"quarantine", "storage_split_brain", "stateful_corrupt"}
            for part in parts
        )
        or "quarantined" in kind
        or ".corrupt" in relative.name
    ):
        return "quarantined"
    if kind in {"sqlite_wal", "sqlite_shm"}:
        return "sqlite_sidecar_keep_with_database"
    if (
        re.search(r"\.tmp(?:\.|$)", relative.name)
        or ".compact_pending_" in relative.name
    ):
        return "incomplete_maintenance"
    if relative.name.endswith(".bak"):
        return "retained_backup"
    return "retained_archive"


def _receipt_index(root_fd: int, root: Path, deadline: float) -> tuple[dict, bool]:
    receipts = {}
    try:
        fd = os.open(
            MANIFEST, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd
        )
    except FileNotFoundError:
        return receipts, True
    except OSError:
        return receipts, False
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            return receipts, False
        remaining = 8 * 1024**2
        complete = True
        while remaining > 0 and time.monotonic() < deadline:
            line = handle.readline(min(65537, remaining + 1))
            if not line:
                return receipts, complete
            remaining -= len(line)
            if len(line) > 65536 or not line.endswith(b"\n") or remaining < 0:
                return receipts, False
            try:
                row = json.loads(line)
                raw = Path(row["path"])
                relative = raw.relative_to(root) if raw.is_absolute() else raw
                if ".." in relative.parts or not relative.parts:
                    continue
                if (
                    row.get("status") == "filesystem_compressed_verified"
                    and row.get("original_replaced") is True
                    and HASH.fullmatch(str(row.get("source_sha256", "")))
                    and row["source_sha256"] == row.get("verified_copy_sha256")
                    and row.get("sqlite_quick_check") == "ok"
                ):
                    receipts[relative.as_posix()] = row
            except (ValueError, KeyError, TypeError):
                complete = False
        return receipts, False


def _publish(root_fd: int, name: str, content: str) -> None:
    try:
        if not stat.S_ISREG(
            os.stat(name, dir_fd=root_fd, follow_symlinks=False).st_mode
        ):
            raise ValueError(f"catalog_output_not_regular:{name}")
    except FileNotFoundError:
        pass
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=root_fd,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        os.fsync(root_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=root_fd)
        except FileNotFoundError:
            pass


def build_catalog(
    root: Path, *, apply: bool, max_entries: int = 50000, max_seconds: float = 20
) -> dict:
    from scripts.ops.cold_archive_compactor import _archive_format

    if type(max_entries) is not int or not 1 <= max_entries <= 50000:
        raise ValueError("catalog entry limit must be between 1 and 50000")
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 30:
        raise ValueError(
            "catalog deadline must be greater than zero and at most 30 seconds"
        )
    raw_root = Path(root).expanduser()
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "apply": bool(apply),
        "mode": "cold_archive_metadata_catalog",
        "archive_root": str(raw_root),
        "ok": False,
        "complete": False,
        "files": [],
        "groups": [],
        "policy": "Metadata organization only; no data move, deletion, restore proof, retention or readiness authority.",
        "date_policy": "Dates are parsed from filenames/parent names, not verified record coverage.",
    }
    # Reject protected aliases before walking or opening anything beneath the root.
    if inspect_storage_path(raw_root).get("status") not in {"present", "missing"}:
        return {**payload, "overall_status": "blocked_protected_root"}
    root = raw_root.resolve(strict=False)
    deadline = time.monotonic() + max_seconds
    visited = 0
    skipped_links = 0
    path_bytes = 0
    scan_complete = True
    errors = []
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        return {**payload, "overall_status": "unavailable", "error": str(exc)}
    lease_fd = None
    try:
        if apply:
            lease_fd = os.open(
                LOCK_NAME, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=root_fd
            )
            if not stat.S_ISREG(os.fstat(lease_fd).st_mode):
                raise ValueError("catalog_lock_not_regular")
            try:
                fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {**payload, "overall_status": "catalog_busy"}
        receipts, receipt_complete = _receipt_index(root_fd, root, deadline)

        def walk(directory_fd: int, relative: Path, depth: int = 0) -> None:
            nonlocal visited, skipped_links, scan_complete, path_bytes
            if depth > 32:
                scan_complete = False
                return
            try:
                with os.scandir(directory_fd) as entries:
                    for entry in entries:
                        if visited >= max_entries or time.monotonic() >= deadline:
                            scan_complete = False
                            return
                        visited += 1
                        rel = relative / entry.name
                        if entry.name.startswith("._") or (
                            relative == Path(".")
                            and entry.name in (*OUTPUTS, LOCK_NAME)
                        ):
                            continue
                        path_bytes += len(rel.as_posix().encode("utf-8"))
                        if path_bytes > 8 * 1024**2:
                            scan_complete = False
                            return
                        meta = entry.stat(follow_symlinks=False)
                        if stat.S_ISLNK(meta.st_mode):
                            skipped_links += 1
                            continue
                        if stat.S_ISDIR(meta.st_mode):
                            child = os.open(
                                entry.name,
                                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=directory_fd,
                            )
                            try:
                                walk(child, rel, depth + 1)
                            finally:
                                os.close(child)
                        elif stat.S_ISREG(meta.st_mode):
                            kind = (
                                "parquet"
                                if rel.suffix == ".parquet"
                                else _archive_format(rel)
                            )
                            dated = _filename_date(rel)
                            compressed = bool(
                                getattr(meta, "st_flags", 0)
                                & getattr(stat, "UF_COMPRESSED", 0x20)
                            )
                            receipt = receipts.get(rel.as_posix(), {})
                            matched = bool(
                                compressed
                                and receipt.get("logical_bytes") == meta.st_size
                                and type(receipt.get("allocated_bytes_after")) is int
                                and meta.st_blocks * 512
                                <= receipt["allocated_bytes_after"]
                            )
                            payload["files"].append(
                                {
                                    "dataset": _dataset(rel),
                                    "filename_date": dated,
                                    "month": (
                                        dated[:7] if dated != "undated" else "undated"
                                    ),
                                    "relative_path": rel.as_posix(),
                                    "format": kind,
                                    "lifecycle": _lifecycle(rel, kind),
                                    "logical_bytes": meta.st_size,
                                    "allocated_bytes": meta.st_blocks * 512,
                                    "filesystem_compressed": compressed,
                                    "verification": (
                                        "recorded_compression_verified_metadata_matches"
                                        if matched
                                        else "not_verified_by_catalog"
                                    ),
                                    "verification_timestamp_utc": (
                                        receipt.get("completed_at_utc", "")
                                        if matched
                                        else ""
                                    ),
                                }
                            )
            except OSError as exc:
                scan_complete = False
                if len(errors) < 20:
                    errors.append({"directory": relative.as_posix(), "error": str(exc)})

        walk(root_fd, Path("."))
        payload["files"].sort(
            key=lambda row: (row["dataset"], row["month"], row["relative_path"])
        )
        groups = defaultdict(
            lambda: {
                "file_count": 0,
                "logical_bytes": 0,
                "allocated_bytes": 0,
                "recorded_verified_count": 0,
                "incomplete_count": 0,
            }
        )
        for row in payload["files"]:
            group = groups[(row["dataset"], row["month"])]
            group["file_count"] += 1
            group["logical_bytes"] += row["logical_bytes"]
            group["allocated_bytes"] += row["allocated_bytes"]
            group["recorded_verified_count"] += row["verification"].startswith(
                "recorded_"
            )
            group["incomplete_count"] += row["lifecycle"] == "incomplete_maintenance"
        payload.update(
            archive_root=str(root),
            complete=scan_complete,
            ok=scan_complete,
            overall_status="indexed" if scan_complete else "partial_inventory",
            groups=[
                {"dataset": dataset, "month": month, **group}
                for (dataset, month), group in sorted(groups.items())
            ],
            file_count=len(payload["files"]),
            entries_observed=visited,
            symlinks_skipped=skipped_links,
            errors=errors,
            receipt_index_complete=receipt_complete,
            content_reverified=False,
            outputs=[str(root / name) for name in OUTPUTS] if apply else [],
        )
        if apply:
            text = [
                "# Cold Archive Index",
                "",
                f"Updated: {payload['timestamp_utc']}",
                "",
                f"Scan: {'complete' if scan_complete else 'PARTIAL'}; {len(payload['files'])} files observed.",
                "",
                "Files remain at their existing paths. This index does not authorize deletion or restore readiness.",
                "Dates below come from filenames, not verified record coverage. Verification is historical receipt metadata, not a new content check.",
                "",
                f"[Full sorted catalog (CSV)]({quote(str(root / OUTPUTS[1]), safe='/')}) | [Machine-readable catalog (JSON)]({quote(str(root / OUTPUTS[2]), safe='/')})",
                "",
                "| Dataset | Month | Files | Physical GiB | Logical GiB | Recorded Verified | Incomplete |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            anchors = {
                (group["dataset"], group["month"]): f"group-{number}"
                for number, group in enumerate(payload["groups"])
            }
            for group in payload["groups"]:
                anchor = anchors[(group["dataset"], group["month"])]
                text.append(
                    f"| [{_label(group['dataset'])}](#{anchor}) | {group['month']} | {group['file_count']} | {group['allocated_bytes']/GIB:.3f} | {group['logical_bytes']/GIB:.3f} | {group['recorded_verified_count']} | {group['incomplete_count']} |"
                )
            current = None
            for row in payload["files"]:
                key = (row["dataset"], row["month"])
                if key != current:
                    text.extend(
                        [
                            "",
                            f'<a id="{anchors[key]}"></a>',
                            f"## {_label(key[0])} / {key[1]}",
                            "",
                        ]
                    )
                    current = key
                path = row["relative_path"]
                text.append(
                    f"- [{_label(path)}]({quote(str(root / path), safe='/')}) | {row['lifecycle']} | {row['allocated_bytes']/GIB:.3f} GiB | {row['verification']}"
                )
            output = io.StringIO(newline="")
            fields = (
                list(payload["files"][0])
                if payload["files"]
                else ["dataset", "filename_date", "relative_path"]
            )
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for row in payload["files"]:
                # A filename must never become a spreadsheet formula on opening the CSV.
                writer.writerow(
                    {
                        key: (
                            "'" + value
                            if isinstance(value, str)
                            and value.lstrip().startswith(("=", "+", "-", "@"))
                            else value
                        )
                        for key, value in row.items()
                    }
                )
            _publish(root_fd, OUTPUTS[0], "\n".join(text) + "\n")
            _publish(root_fd, OUTPUTS[1], output.getvalue())
            _publish(
                root_fd,
                OUTPUTS[2],
                json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            )
        return payload
    finally:
        if lease_fd is not None:
            os.close(lease_fd)
        os.close(root_fd)
