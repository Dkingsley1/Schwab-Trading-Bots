#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops.long_runtime_common import iso_now, write_payload  # noqa: E402
from scripts.ops.writer_cycle_coordinator import writer_state_snapshot  # noqa: E402
from core.runtime_maintenance import (  # noqa: E402
    engage_maintenance_hold,
    maintenance_hold_snapshot,
    release_maintenance_hold,
)

DEFAULT_ARCHIVE_ROOT = PROJECT_ROOT / "governance" / "archive" / "cold_archive"
PROTECTED_VOLUME_PREFIXES = ("/Volumes/VIDEO",)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "cold_archive_compactor_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "cold_archive_compactor.lock"
DEFAULT_MANIFEST_NAME = "cold_archive_compaction_manifest.jsonl"
DEFAULT_README_NAME = "COLD_ARCHIVE_README.txt"
DEFAULT_CORRUPT_GZIP_QUARANTINE = Path("quarantine") / "corrupt_gzip_orphans"


def writer_blocks_compaction(writer_state: dict[str, Any], *, allow_active_writer: bool = False) -> bool:
    return bool(writer_state.get("active")) and not bool(allow_active_writer)


def _is_protected(path: Path) -> bool:
    raw = str(path.expanduser())
    return any(raw == prefix or raw.startswith(f"{prefix}/") for prefix in PROTECTED_VOLUME_PREFIXES)


def _default_archive_root() -> Path:
    configured = str(os.getenv("BOT_SECOND_COLD_ROOT", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    external = str(os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "") or "").strip()
    return Path(external).expanduser() / "cold_archive" if external else DEFAULT_ARCHIVE_ROOT


def archive_root_available(path: Path) -> bool:
    candidate = path.expanduser()
    return bool(not _is_protected(candidate) and candidate.exists() and candidate.is_dir())


def wait_for_writer_handoff(
    project_root: Path,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + max(float(timeout_seconds), 0.0)
    polls = 0
    while True:
        polls += 1
        state = writer_state_snapshot(project_root)
        if not writer_blocks_compaction(state):
            return {
                "ready": True,
                "status": "writer_handoff_complete",
                "waited_seconds": round(max(time.monotonic() - started, 0.0), 3),
                "poll_count": polls,
                "writer_state": state,
            }
        if time.monotonic() >= deadline:
            return {
                "ready": False,
                "status": "writer_handoff_timeout",
                "waited_seconds": round(max(time.monotonic() - started, 0.0), 3),
                "poll_count": polls,
                "writer_state": state,
            }
        time.sleep(max(min(float(poll_seconds), max(deadline - time.monotonic(), 0.0)), 0.1))


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _gb(raw_bytes: int) -> float:
    return round(float(raw_bytes) / float(1024**3), 3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _age_hours(path: Path, *, now: datetime) -> float:
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return 0.0
    return max((now - modified).total_seconds() / 3600.0, 0.0)


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path)


def _archive_format(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".corrupt-gzip-fragment"):
        return "gzip_quarantined_corrupt_fragment"
    if name.endswith(".corrupt-gzip-fragment.metadata.json"):
        return "gzip_quarantine_metadata"
    if name.endswith(".gz.tmp.tmp"):
        return "gzip_temporary_duplicate"
    if name.endswith(".jsonl.gz.tmp"):
        return "gzip_pending_finalize"
    if ".jsonl.compact_pending_" in name:
        return "jsonl_pending_compaction"
    if name.endswith(".jsonl.gz"):
        return "jsonl_gzip"
    if name.endswith(".jsonl"):
        return "jsonl_plain"
    if name.endswith((".sqlite-wal", ".sqlite3-wal", ".db-wal")):
        return "sqlite_wal"
    if name.endswith((".sqlite-shm", ".sqlite3-shm", ".db-shm")):
        return "sqlite_shm"
    if path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
        return "sqlite_database"
    if ".sqlite" in name and ".pre_local_failover_" in name and name.endswith(".bak"):
        return "sqlite_failover_backup"
    if ".sqlite" in name and ".corrupt-" in name:
        return "sqlite_quarantined_corrupt"
    if ".sqlite" in name and name.endswith(".local_fallback"):
        return "sqlite_local_fallback_snapshot"
    if name.endswith(".jsonl.local_fallback"):
        return "jsonl_local_fallback_snapshot"
    if name.endswith(".log.local_fallback"):
        return "log_local_fallback_snapshot"
    if name.endswith(".log"):
        return "line_log"
    if name.endswith(".gz"):
        return "gzip_other"
    if name.endswith(".json"):
        return "json_document"
    if name.endswith((".txt", ".md")):
        return "archive_documentation"
    if len(path.name) == 64 and all(char in "0123456789abcdef" for char in path.name.lower()):
        return "content_addressed_blob"
    return "other"


def _archive_data_family(relative_path: str) -> str:
    rel = str(relative_path or "").lower().replace("\\", "/")
    if "/quarantine/corrupt_gzip_orphans/" in f"/{rel.lstrip('/')}":
        return "quarantined_corrupt_gzip_fragments"
    if ".pre_local_failover_" in rel:
        return "verified_failover_backups"
    if "/stateful_corrupt/" in rel:
        return "quarantined_corrupt_state"
    if "/storage_split_brain/" in rel:
        return "split_brain_quarantine"
    if "/bot_logs_cleanup/" in rel:
        return "quarantined_runtime_artifacts"
    sql_families = {
        "jsonl_link_crypto_explanations": "sql_link_crypto_explanations",
        "jsonl_link_explanations": "sql_link_explanations",
        "jsonl_link_governance": "sql_link_governance",
        "jsonl_link_runtime": "sql_link_runtime",
        "jsonl_link_aggressive_trading": "sql_link_aggressive_trading",
        "jsonl_link_trading": "sql_link_trading",
        "/risk_support/": "sql_link_risk_support",
    }
    for marker, family in sql_families.items():
        if marker in rel:
            return family
    if "/decisions/" in rel or Path(rel).name.startswith("trade_decisions_"):
        return "trade_decisions"
    if "/decision_explanations/" in rel:
        return "decision_explanations"
    if "execution_intents" in rel or "/execution_lanes/" in rel:
        return "execution_intents"
    if "master_control" in rel:
        return "master_control_telemetry"
    if "killswitch" in rel or "/governance/watchdog/" in rel:
        return "watchdog_and_killswitch_events"
    if "/governance/channels/risk/" in rel or (
        "/governance_channels/" in rel and "/risk/" in rel
    ):
        return "risk_governance_events"
    if "/governance/channels/decision/" in rel or (
        "/governance_channels/" in rel and "/decision/" in rel
    ):
        return "decision_governance_events"
    if "/content_store/" in rel:
        return "content_addressed_evidence"
    if "/governance/" in rel:
        return "other_governance_evidence"
    if Path(rel).name in {DEFAULT_MANIFEST_NAME.lower(), DEFAULT_README_NAME.lower()}:
        return "archive_control"
    return "other_archived_data"


def _archive_tier(relative_path: str) -> str:
    normalized = str(relative_path or "").lower().replace("\\", "/")
    rel = f"/{normalized.lstrip('/')}"
    if "/data/deep_cold/manifest_backed/" in rel:
        return "manifest_backed_evidence"
    if "/data/sql_hot_archive/" in rel:
        return "sql_hot_archive"
    if "/sql_link_shards/" in rel:
        return "sql_link_shards"
    if "/deep_cold/stale_stage/" in rel:
        return "stale_stage_and_quarantine"
    if "/quarantine/" in rel:
        return "stale_stage_and_quarantine"
    if "/sqlite_tmp/" in rel:
        return "sqlite_temporary_support"
    if Path(rel).name in {DEFAULT_MANIFEST_NAME.lower(), DEFAULT_README_NAME.lower()}:
        return "archive_control"
    return "other"


def _inventory_rows(buckets: dict[str, dict[str, int]], *, key_name: str) -> list[dict[str, Any]]:
    return [
        {
            key_name: key,
            "file_count": int(values["file_count"]),
            "bytes": int(values["bytes"]),
            "gb": _gb(int(values["bytes"])),
        }
        for key, values in sorted(
            buckets.items(),
            key=lambda item: (-int(item[1]["bytes"]), item[0]),
        )
    ]


def _archive_inventory(root: Path, files: list[Path]) -> dict[str, Any]:
    formats: dict[str, dict[str, int]] = {}
    families: dict[str, dict[str, int]] = {}
    tiers: dict[str, dict[str, int]] = {}
    total_bytes = 0

    def add(bucket: dict[str, dict[str, int]], key: str, size: int) -> None:
        row = bucket.setdefault(key, {"file_count": 0, "bytes": 0})
        row["file_count"] += 1
        row["bytes"] += size

    for path in files:
        try:
            size = max(int(path.stat().st_size), 0)
        except OSError:
            continue
        relative = _relative(root, path)
        total_bytes += size
        add(formats, _archive_format(path), size)
        add(families, _archive_data_family(relative), size)
        add(tiers, _archive_tier(relative), size)

    return {
        "indexed_file_count": sum(int(row["file_count"]) for row in formats.values()),
        "indexed_bytes": total_bytes,
        "indexed_gb": _gb(total_bytes),
        "formats": _inventory_rows(formats, key_name="format"),
        "data_families": _inventory_rows(families, key_name="data_family"),
        "tiers": _inventory_rows(tiers, key_name="tier"),
        "excluded_from_inventory": ["symlinks", "AppleDouble metadata files prefixed with ._"],
    }


def _canonical_jsonl_target(path: Path) -> Path:
    name = path.name.split(".compact_pending_", 1)[0]
    if not name.endswith(".jsonl"):
        raise ValueError(f"unsupported_jsonl_name:{path.name}")
    return path.with_name(f"{name}.gz")


def _iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    rows: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.name.startswith("._"):
            continue
        rows.append(path)
    return rows


def stable_file_work_candidates(
    root: Path,
    *,
    min_age_hours: float,
    include_plain_jsonl: bool,
    excluded_paths: set[Path] | None = None,
) -> list[Path]:
    now = datetime.now(timezone.utc)
    excluded = {path.resolve(strict=False) for path in (excluded_paths or set())}
    candidates: list[Path] = []
    for path in _iter_files(root):
        if path.resolve(strict=False) in excluded:
            continue
        if _age_hours(path, now=now) < max(float(min_age_hours), 0.0):
            continue
        name = path.name
        size = _safe_int(path.stat().st_size if path.exists() else 0)
        if (
            (size > 0 and ".jsonl.compact_pending_" in name)
            or (size > 0 and include_plain_jsonl and name.endswith(".jsonl"))
            or name.endswith(".jsonl.gz.tmp")
            or name.endswith(".gz.tmp.tmp")
        ):
            candidates.append(path)
    return candidates


def _sqlite_inventory(path: Path, *, check_integrity: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "ok": False,
        "quick_check": "",
        "page_count": 0,
        "freelist_count": 0,
        "page_size": 0,
        "reclaimable_bytes": 0,
        "reclaimable_ratio": 0.0,
        "integrity_checked": bool(check_integrity),
        "error": "",
    }
    try:
        uri = f"file:{path.resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=5.0) as conn:
            if check_integrity:
                quick_check_row = conn.execute("PRAGMA quick_check(1)").fetchone()
                quick_check = str(quick_check_row[0] if quick_check_row else "")
            else:
                quick_check = "not_run_inventory_only"
            page_count = _safe_int(conn.execute("PRAGMA page_count").fetchone()[0])
            freelist_count = _safe_int(conn.execute("PRAGMA freelist_count").fetchone()[0])
            page_size = _safe_int(conn.execute("PRAGMA page_size").fetchone()[0])
        reclaimable_bytes = max(freelist_count * page_size, 0)
        row.update(
            {
                "ok": quick_check.lower() == "ok" if check_integrity else True,
                "quick_check": quick_check,
                "page_count": page_count,
                "freelist_count": freelist_count,
                "page_size": page_size,
                "reclaimable_bytes": reclaimable_bytes,
                "reclaimable_ratio": round(freelist_count / max(page_count, 1), 6),
            }
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}:{exc}"
    return row


def _stream_hash(handle: Any) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total = 0
    lines = 0
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
        total += len(chunk)
        lines += chunk.count(b"\n")
    return digest.hexdigest(), total, lines


def _compress_jsonl(source: Path, target: Path, *, compression_level: int) -> dict[str, Any]:
    started = iso_now()
    source_stat = source.stat()
    source_size = int(source_stat.st_size)
    if target.exists():
        try:
            with gzip.open(target, "rb") as existing:
                target_hash, target_raw_bytes, target_lines = _stream_hash(existing)
            with source.open("rb") as current:
                source_hash, current_bytes, current_lines = _stream_hash(current)
            if (
                target_hash == source_hash
                and target_raw_bytes == current_bytes
                and target_lines == current_lines
            ):
                source.unlink()
                return {
                    "status": "released_verified_duplicate",
                    "source": str(source),
                    "target": str(target),
                    "raw_bytes": source_size,
                    "archive_bytes": int(target.stat().st_size),
                    "released_bytes": source_size,
                    "sha256_uncompressed": source_hash,
                    "line_count": current_lines,
                    "started_utc": started,
                    "completed_utc": iso_now(),
                }
            return {
                "status": "blocked_target_conflict",
                "source": str(source),
                "target": str(target),
                "raw_bytes": source_size,
                "error": "existing_target_content_differs",
            }
        except Exception as exc:
            return {
                "status": "blocked_target_invalid",
                "source": str(source),
                "target": str(target),
                "raw_bytes": source_size,
                "error": f"{type(exc).__name__}:{exc}",
            }

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.cold_compact_{os.getpid()}.tmp")
    source_hash = hashlib.sha256()
    raw_bytes = 0
    line_count = 0
    try:
        with source.open("rb") as src, tmp.open("wb") as raw_out:
            with gzip.GzipFile(
                filename=target.name.removesuffix(".gz"),
                mode="wb",
                compresslevel=max(min(int(compression_level), 9), 1),
                fileobj=raw_out,
                mtime=0,
            ) as dst:
                for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                    source_hash.update(chunk)
                    raw_bytes += len(chunk)
                    line_count += chunk.count(b"\n")
                    dst.write(chunk)
            raw_out.flush()
            os.fsync(raw_out.fileno())

        if raw_bytes != source_size or source.stat().st_size != source_size:
            raise RuntimeError("source_changed_during_compaction")
        with gzip.open(tmp, "rb") as verify:
            verify_hash, verify_bytes, verify_lines = _stream_hash(verify)
        digest = source_hash.hexdigest()
        if verify_hash != digest or verify_bytes != raw_bytes or verify_lines != line_count:
            raise RuntimeError("gzip_restore_proof_mismatch")

        os.replace(tmp, target)
        os.utime(target, (source_stat.st_atime, source_stat.st_mtime))
        source.unlink()
        return {
            "status": "compacted_verified",
            "source": str(source),
            "target": str(target),
            "raw_bytes": raw_bytes,
            "archive_bytes": int(target.stat().st_size),
            "released_bytes": max(raw_bytes - int(target.stat().st_size), 0),
            "sha256_uncompressed": digest,
            "line_count": line_count,
            "codec": "gzip",
            "read_command": f"gzip -cd -- {target}",
            "started_utc": started,
            "completed_utc": iso_now(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "source": str(source),
            "target": str(target),
            "raw_bytes": source_size,
            "error": f"{type(exc).__name__}:{exc}",
        }
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _release_verified_tmp_duplicate(path: Path) -> dict[str, Any]:
    final_name = path.name.removesuffix(".tmp.tmp")
    target = path.with_name(final_name)
    if not target.exists() and final_name.startswith("."):
        target = path.with_name(final_name[1:])
    row = {"source": str(path), "target": str(target), "status": "blocked_missing_final"}
    if not target.exists() or not target.is_file():
        return row
    try:
        source_size = int(path.stat().st_size)
        target_size = int(target.stat().st_size)
        if source_size != target_size:
            return dict(row, status="blocked_size_mismatch", source_bytes=source_size, target_bytes=target_size)
        source_hash = _sha256(path)
        target_hash = _sha256(target)
        if source_hash != target_hash:
            return dict(row, status="blocked_hash_mismatch", source_bytes=source_size, target_bytes=target_size)
        with gzip.open(target, "rb") as handle:
            handle.read(1024 * 1024)
        path.unlink()
        return {
            **row,
            "status": "released_verified_duplicate",
            "released_bytes": source_size,
            "sha256_compressed": source_hash,
        }
    except Exception as exc:
        return dict(row, status="error", error=f"{type(exc).__name__}:{exc}")


def _recover_gzip_tmp(path: Path) -> dict[str, Any]:
    target = path.with_name(path.name.removesuffix(".tmp"))
    row = {"source": str(path), "target": str(target), "status": "blocked_invalid_name"}
    if not path.name.endswith(".jsonl.gz.tmp"):
        return row
    try:
        source_size = int(path.stat().st_size)
        if target.exists():
            target_size = int(target.stat().st_size)
            if source_size != target_size:
                return dict(
                    row,
                    status="blocked_target_conflict",
                    source_bytes=source_size,
                    target_bytes=target_size,
                )
            source_hash = _sha256(path)
            target_hash = _sha256(target)
            if source_hash != target_hash:
                return dict(
                    row,
                    status="blocked_target_conflict",
                    source_bytes=source_size,
                    target_bytes=target_size,
                )
            with gzip.open(target, "rb") as handle:
                _stream_hash(handle)
            path.unlink()
            return {
                **row,
                "status": "released_verified_duplicate",
                "released_bytes": source_size,
                "sha256_compressed": source_hash,
            }

        with gzip.open(path, "rb") as handle:
            raw_hash, raw_bytes, line_count = _stream_hash(handle)
        os.replace(path, target)
        return {
            **row,
            "status": "recovered_verified_orphan",
            "archive_bytes": source_size,
            "raw_bytes": raw_bytes,
            "released_bytes": 0,
            "sha256_uncompressed": raw_hash,
            "line_count": line_count,
            "codec": "gzip",
            "read_command": f"gzip -cd -- {target}",
        }
    except Exception as exc:
        return dict(row, status="error", error=f"{type(exc).__name__}:{exc}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _quarantine_corrupt_gzip_tmp(
    path: Path,
    *,
    archive_root: Path,
    validation_error: str,
) -> dict[str, Any]:
    target = path.with_name(path.name.removesuffix(".tmp"))
    row = {
        "source": str(path),
        "target": str(target),
        "status": "blocked_invalid_name",
        "validation_error": validation_error,
    }
    if not path.name.endswith(".jsonl.gz.tmp"):
        return row
    try:
        root = archive_root.resolve(strict=False)
        relative = path.resolve(strict=False).relative_to(root)
        source_size = int(path.stat().st_size)
        source_hash = _sha256(path)
        quarantine_dir = root / DEFAULT_CORRUPT_GZIP_QUARANTINE / relative.parent
        quarantine_path = quarantine_dir / (
            f"{path.name}.{source_hash[:16]}.corrupt-gzip-fragment"
        )
        metadata_path = quarantine_path.with_name(f"{quarantine_path.name}.metadata.json")
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        if quarantine_path.exists():
            if int(quarantine_path.stat().st_size) != source_size or _sha256(quarantine_path) != source_hash:
                return dict(
                    row,
                    status="blocked_quarantine_conflict",
                    quarantine_path=str(quarantine_path),
                    archive_bytes=source_size,
                )
            path.unlink()
            action_status = "released_verified_quarantine_duplicate"
            released_bytes = source_size
        else:
            os.replace(path, quarantine_path)
            action_status = "quarantined_corrupt_orphan"
            released_bytes = 0

        metadata = {
            "schema_version": 1,
            "detected_utc": iso_now(),
            "source_path": str(path),
            "source_relative_path": str(relative).replace("\\", "/"),
            "intended_target_path": str(target),
            "quarantine_path": str(quarantine_path),
            "archive_bytes": source_size,
            "sha256_compressed_fragment": source_hash,
            "validation_error": validation_error,
            "recovery_policy": "preserve_fragment_for_forensic_or_manual_recovery",
        }
        _write_json_atomic(metadata_path, metadata)
        return {
            **row,
            "status": action_status,
            "quarantine_path": str(quarantine_path),
            "metadata_path": str(metadata_path),
            "archive_bytes": source_size,
            "released_bytes": released_bytes,
            "sha256_compressed_fragment": source_hash,
        }
    except Exception as exc:
        return dict(row, status="error", error=f"{type(exc).__name__}:{exc}")


def _vacuum_sqlite(path: Path, inventory: dict[str, Any]) -> dict[str, Any]:
    before_bytes = int(path.stat().st_size)
    try:
        before_check = _sqlite_inventory(path, check_integrity=True)
        if not before_check.get("ok"):
            raise RuntimeError(
                f"pre_vacuum_integrity_failed:{before_check.get('quick_check') or before_check.get('error')}"
            )
        with sqlite3.connect(str(path), timeout=60.0) as conn:
            conn.execute("PRAGMA busy_timeout=60000")
            conn.execute("VACUUM")
        after = _sqlite_inventory(path, check_integrity=True)
        if not after.get("ok"):
            raise RuntimeError(f"post_vacuum_integrity_failed:{after.get('quick_check') or after.get('error')}")
        after_bytes = int(path.stat().st_size)
        return {
            "status": "vacuumed_verified",
            "path": str(path),
            "before_bytes": before_bytes,
            "after_bytes": after_bytes,
            "released_bytes": max(before_bytes - after_bytes, 0),
            "quick_check": after.get("quick_check"),
            "reclaimable_bytes_before": _safe_int(inventory.get("reclaimable_bytes")),
        }
    except Exception as exc:
        return {
            "status": "error",
            "path": str(path),
            "before_bytes": before_bytes,
            "error": f"{type(exc).__name__}:{exc}",
        }


def _append_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_readme(path: Path, *, manifest_path: Path) -> None:
    content = (
        "Schwab Trading Bot Cold Archive\n"
        "================================\n\n"
        "Contents\n"
        "--------\n"
        "- Decision explanations from paper and shadow sleeves.\n"
        "- Decision, risk, execution-intent, watchdog, kill-switch, and master-control evidence.\n"
        "- Dated SQLite link archives for runtime, governance, trading, crypto explanations, and risk support.\n"
        "- Manifest-backed stale-stage, quarantine, and content-addressed evidence retained by policy.\n\n"
        "JSONL evidence is stored as lossless gzip and remains line-readable.\n"
        "Read: gzip -cd -- path/to/file.jsonl.gz | less\n"
        "Search: gzip -cd -- path/to/file.jsonl.gz | rg 'pattern'\n"
        "SQLite archives remain ordinary SQLite databases.\n"
        "Inspect immutable snapshot: sqlite3 'file:/absolute/path/archive.sqlite3?immutable=1' '.tables'\n"
        f"Compaction and restore proofs: {manifest_path}\n"
        "Files are removed only after checksum or SQLite integrity verification.\n"
        "Interrupted .jsonl.gz.tmp files are finalized only after a full gzip restore proof.\n"
        "Truncated gzip fragments are hash-preserved under quarantine/corrupt_gzip_orphans with provenance metadata.\n"
        "Symlinks and macOS AppleDouble metadata are not compacted as evidence.\n"
    )
    path.write_text(content, encoding="utf-8")


def build_payload(
    *,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    apply: bool = False,
    min_age_hours: float = 24.0,
    max_files: int = 8,
    max_raw_gb: float = 16.0,
    compression_level: int = 3,
    include_plain_jsonl: bool = True,
    vacuum_sqlite: bool = False,
    sqlite_min_reclaim_mb: float = 256.0,
    sqlite_min_reclaim_ratio: float = 0.08,
    sqlite_inventory_limit: int = 200,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    raw_root = Path(archive_root).expanduser()
    raw_manifest = Path(manifest_path).expanduser() if manifest_path is not None else None
    if _is_protected(raw_root) or (raw_manifest is not None and _is_protected(raw_manifest)):
        return {
            "timestamp_utc": iso_now(),
            "schema_version": 1,
            "ok": False,
            "overall_status": "blocked_protected_volume",
            "apply": bool(apply),
            "archive_root": str(raw_root),
            "blockers": ["protected_archive_volume_rejected"],
            "never_touch_protected_volumes": list(PROTECTED_VOLUME_PREFIXES),
        }
    root = raw_root.resolve(strict=False)
    now = datetime.now(timezone.utc)
    manifest = Path(manifest_path or (root / DEFAULT_MANIFEST_NAME)).expanduser()
    manifest_resolved = manifest.resolve(strict=False)
    files = [path for path in _iter_files(root) if path.resolve(strict=False) != manifest_resolved]

    jsonl_candidates: list[dict[str, Any]] = []
    gzip_finalize_candidates: list[dict[str, Any]] = []
    tmp_candidates: list[dict[str, Any]] = []
    sqlite_candidates: list[dict[str, Any]] = []
    for path in files:
        age_hours = _age_hours(path, now=now)
        if age_hours < max(float(min_age_hours), 0.0):
            continue
        size = _safe_int(path.stat().st_size if path.exists() else 0)
        if ".jsonl.compact_pending_" in path.name or (include_plain_jsonl and path.name.endswith(".jsonl")):
            if size > 0:
                jsonl_candidates.append(
                    {
                        "path": str(path),
                        "relative_path": _relative(root, path),
                        "size_bytes": size,
                        "age_hours": round(age_hours, 3),
                        "orphaned_compaction": ".compact_pending_" in path.name,
                    }
                )
        elif path.name.endswith(".jsonl.gz.tmp"):
            gzip_finalize_candidates.append(
                {
                    "path": str(path),
                    "relative_path": _relative(root, path),
                    "size_bytes": size,
                    "age_hours": round(age_hours, 3),
                }
            )
        elif path.name.endswith(".gz.tmp.tmp"):
            tmp_candidates.append(
                {
                    "path": str(path),
                    "relative_path": _relative(root, path),
                    "size_bytes": size,
                    "age_hours": round(age_hours, 3),
                }
            )
        elif path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
            sqlite_candidates.append(
                {
                    "path": str(path),
                    "relative_path": _relative(root, path),
                    "size_bytes": size,
                    "age_hours": round(age_hours, 3),
                }
            )

    jsonl_candidates.sort(key=lambda row: (-int(bool(row["orphaned_compaction"])), -row["size_bytes"], row["path"]))
    gzip_finalize_candidates.sort(key=lambda row: (-row["size_bytes"], row["path"]))
    tmp_candidates.sort(key=lambda row: (-row["size_bytes"], row["path"]))
    sqlite_candidates.sort(key=lambda row: (-row["size_bytes"], row["path"]))

    selected_jsonl: list[dict[str, Any]] = []
    selected_bytes = 0
    raw_cap = max(int(float(max_raw_gb) * 1024**3), 0)
    for row in jsonl_candidates:
        if max_files > 0 and len(selected_jsonl) >= max_files:
            break
        size = _safe_int(row.get("size_bytes"))
        if selected_jsonl and raw_cap > 0 and selected_bytes + size > raw_cap:
            continue
        selected_jsonl.append(row)
        selected_bytes += size

    selected_gzip_finalize: list[dict[str, Any]] = []
    selected_gzip_finalize_bytes = 0
    for row in gzip_finalize_candidates:
        if max_files > 0 and len(selected_gzip_finalize) >= max_files:
            break
        size = _safe_int(row.get("size_bytes"))
        if (
            selected_gzip_finalize
            and raw_cap > 0
            and selected_gzip_finalize_bytes + size > raw_cap
        ):
            continue
        selected_gzip_finalize.append(row)
        selected_gzip_finalize_bytes += size

    sqlite_inventory: list[dict[str, Any]] = []
    for row in sqlite_candidates[: max(int(sqlite_inventory_limit), 0)]:
        inventory = _sqlite_inventory(Path(row["path"]))
        inventory.update({"relative_path": row["relative_path"], "size_bytes": row["size_bytes"]})
        inventory["eligible"] = bool(
            inventory.get("ok")
            and _safe_int(inventory.get("reclaimable_bytes")) >= int(max(float(sqlite_min_reclaim_mb), 0.0) * 1024**2)
            and float(inventory.get("reclaimable_ratio") or 0.0) >= max(float(sqlite_min_reclaim_ratio), 0.0)
        )
        sqlite_inventory.append(inventory)

    actions: list[dict[str, Any]] = []
    if apply:
        root.mkdir(parents=True, exist_ok=True)
        for row in selected_jsonl:
            source = Path(row["path"])
            actions.append(_compress_jsonl(source, _canonical_jsonl_target(source), compression_level=compression_level))
        for row in selected_gzip_finalize:
            recovery = _recover_gzip_tmp(Path(row["path"]))
            if recovery.get("status") == "error":
                recovery = _quarantine_corrupt_gzip_tmp(
                    Path(row["path"]),
                    archive_root=root,
                    validation_error=str(recovery.get("error") or "gzip_restore_proof_failed"),
                )
            actions.append(recovery)
        for row in tmp_candidates:
            actions.append(_release_verified_tmp_duplicate(Path(row["path"])))
        if vacuum_sqlite:
            for inventory in sqlite_inventory:
                if inventory.get("eligible"):
                    actions.append(_vacuum_sqlite(Path(inventory["path"]), inventory))
        manifest_records = [
            {"timestamp_utc": iso_now(), "archive_root": str(root), **row}
            for row in actions
            if row.get("status")
            in {
                "compacted_verified",
                "recovered_verified_orphan",
                "released_verified_duplicate",
                "released_verified_quarantine_duplicate",
                "quarantined_corrupt_orphan",
                "vacuumed_verified",
            }
        ]
        _append_manifest(manifest, manifest_records)
        _write_readme(root / DEFAULT_README_NAME, manifest_path=manifest)

    inventory_files = [path for path in _iter_files(root) if path.resolve(strict=False) != manifest_resolved]
    archive_inventory = _archive_inventory(root, inventory_files)

    errors = [row for row in actions if row.get("status") == "error"]
    blocked = [row for row in actions if str(row.get("status") or "").startswith("blocked_")]
    successful = [
        row
        for row in actions
        if row.get("status")
        in {
            "compacted_verified",
            "recovered_verified_orphan",
            "released_verified_duplicate",
            "released_verified_quarantine_duplicate",
            "quarantined_corrupt_orphan",
            "vacuumed_verified",
        }
    ]
    released_bytes = sum(_safe_int(row.get("released_bytes")) for row in successful)
    if errors:
        overall_status = "degraded"
    elif blocked:
        overall_status = "advisory"
    elif apply and successful:
        overall_status = "applied"
    elif (
        jsonl_candidates
        or gzip_finalize_candidates
        or tmp_candidates
        or any(row.get("eligible") for row in sqlite_inventory)
    ):
        overall_status = "planned"
    else:
        overall_status = "ready"

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": not errors,
        "overall_status": overall_status,
        "apply": bool(apply),
        "archive_root": str(root),
        "manifest_path": str(manifest),
        "readme_path": str(root / DEFAULT_README_NAME),
        "policy": {
            "codec": "gzip",
            "lossless_restore_proof": True,
            "remove_source_only_after_verify": True,
            "remove_tmp_only_after_exact_hash_match": True,
            "invalid_gzip_fragment_quarantine_is_lossless": True,
            "sqlite_remains_directly_readable": True,
            "sqlite_vacuum_requires_quick_check_and_reclaim_floor": True,
            "min_age_hours": float(min_age_hours),
            "max_files": int(max_files),
            "max_raw_gb": float(max_raw_gb),
            "compression_level": int(compression_level),
        },
        "summary": {
            "jsonl_candidate_count": len(jsonl_candidates),
            "jsonl_candidate_gb": _gb(sum(_safe_int(row.get("size_bytes")) for row in jsonl_candidates)),
            "selected_jsonl_count": len(selected_jsonl),
            "selected_jsonl_gb": _gb(selected_bytes),
            "gzip_finalize_candidate_count": len(gzip_finalize_candidates),
            "gzip_finalize_candidate_gb": _gb(
                sum(_safe_int(row.get("size_bytes")) for row in gzip_finalize_candidates)
            ),
            "selected_gzip_finalize_count": len(selected_gzip_finalize),
            "selected_gzip_finalize_gb": _gb(selected_gzip_finalize_bytes),
            "tmp_duplicate_candidate_count": len(tmp_candidates),
            "tmp_duplicate_candidate_gb": _gb(sum(_safe_int(row.get("size_bytes")) for row in tmp_candidates)),
            "sqlite_inventory_count": len(sqlite_inventory),
            "sqlite_vacuum_eligible_count": sum(1 for row in sqlite_inventory if row.get("eligible")),
            "successful_action_count": len(successful),
            "blocked_action_count": len(blocked),
            "error_count": len(errors),
            "quarantined_corrupt_orphan_count": sum(
                1 for row in successful if row.get("status") == "quarantined_corrupt_orphan"
            ),
            "quarantined_corrupt_orphan_gb": _gb(
                sum(
                    _safe_int(row.get("archive_bytes"))
                    for row in successful
                    if row.get("status") == "quarantined_corrupt_orphan"
                )
            ),
            "released_gb": _gb(released_bytes),
        },
        "selected_jsonl": selected_jsonl,
        "gzip_finalize_candidates": gzip_finalize_candidates,
        "selected_gzip_finalize": selected_gzip_finalize,
        "tmp_duplicate_candidates": tmp_candidates,
        "sqlite_inventory": sqlite_inventory,
        "archive_inventory": archive_inventory,
        "actions": actions,
        "reader_commands": {
            "jsonl": "gzip -cd -- FILE.jsonl.gz | less",
            "search_jsonl": "gzip -cd -- FILE.jsonl.gz | rg 'PATTERN'",
            "sqlite": "sqlite3 'file:/ABSOLUTE/PATH/FILE.sqlite3?immutable=1' '.tables'",
        },
        "next_action": (
            "inspect blocked or failed archive records before another bounded wave"
            if errors or blocked
            else "run another bounded wave if candidates remain"
            if apply
            and (
                len(jsonl_candidates) > len(selected_jsonl)
                or len(gzip_finalize_candidates) > len(selected_gzip_finalize)
            )
            else "cold archive is compact and directly readable"
            if apply
            else "run with --apply after the active SQL writer is idle"
        ),
    }


def _acquire_lock(path: Path) -> tuple[Any | None, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={iso_now()}\n")
        handle.flush()
        return handle, ""
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip()
        handle.close()
        return None, owner


def main() -> int:
    parser = argparse.ArgumentParser(description="Losslessly compact and index the directly readable cold archive.")
    parser.add_argument("--archive-root", default=str(_default_archive_root()))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-path", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--min-age-hours", type=float, default=24.0)
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--max-raw-gb", type=float, default=16.0)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--include-plain-jsonl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vacuum-sqlite", action="store_true")
    parser.add_argument("--allow-active-writer", action="store_true")
    parser.add_argument("--coordinate-writer-handoff", action="store_true")
    parser.add_argument(
        "--writer-handoff-timeout-seconds",
        type=float,
        default=float(os.getenv("BOT_COLD_ARCHIVE_WRITER_HANDOFF_TIMEOUT_SECONDS", "900")),
    )
    parser.add_argument(
        "--writer-handoff-poll-seconds",
        type=float,
        default=float(os.getenv("BOT_COLD_ARCHIVE_WRITER_HANDOFF_POLL_SECONDS", "2")),
    )
    parser.add_argument(
        "--maintenance-hold-ttl-seconds",
        type=int,
        default=int(os.getenv("BOT_COLD_ARCHIVE_MAINTENANCE_HOLD_TTL_SECONDS", "7200")),
    )
    parser.add_argument("--sqlite-min-reclaim-mb", type=float, default=256.0)
    parser.add_argument("--sqlite-min-reclaim-ratio", type=float, default=0.08)
    parser.add_argument("--sqlite-inventory-limit", type=int, default=200)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        os.nice(max(15 - os.nice(0), 0))
    except Exception:
        pass

    lock_handle = None
    maintenance_hold_token = ""
    maintenance_hold_result: dict[str, Any] = {}
    writer_handoff: dict[str, Any] = {}
    payload: dict[str, Any] | None = None
    coordinate_writer_handoff = bool(args.coordinate_writer_handoff and not args.allow_active_writer)

    if args.apply:
        archive_root = Path(args.archive_root).expanduser()
        if _is_protected(archive_root):
            payload = {
                "timestamp_utc": iso_now(),
                "schema_version": 1,
                "ok": False,
                "overall_status": "blocked_protected_volume",
                "apply": True,
                "archive_root": str(archive_root),
                "blockers": ["protected_archive_volume_rejected"],
                "never_touch_protected_volumes": list(PROTECTED_VOLUME_PREFIXES),
            }
            write_payload(Path(args.out_file).expanduser(), payload)
            print(json.dumps(payload, ensure_ascii=True))
            return 2
        if not archive_root_available(archive_root):
            payload = {
                "timestamp_utc": iso_now(),
                "schema_version": 1,
                "ok": True,
                "overall_status": "deferred_archive_unavailable",
                "apply": True,
                "archive_root": str(archive_root),
                "next_action": "retry after the configured cold archive is mounted",
            }
            write_payload(Path(args.out_file).expanduser(), payload)
            print(json.dumps(payload, ensure_ascii=True))
            return 0
        if coordinate_writer_handoff and not args.vacuum_sqlite:
            configured_manifest = (
                Path(args.manifest_path).expanduser()
                if args.manifest_path
                else archive_root / DEFAULT_MANIFEST_NAME
            )
            stable_candidates = stable_file_work_candidates(
                archive_root,
                min_age_hours=float(args.min_age_hours),
                include_plain_jsonl=bool(args.include_plain_jsonl),
                excluded_paths={configured_manifest},
            )
            if not stable_candidates:
                payload = build_payload(
                    archive_root=archive_root,
                    apply=False,
                    min_age_hours=float(args.min_age_hours),
                    max_files=int(args.max_files),
                    max_raw_gb=float(args.max_raw_gb),
                    compression_level=int(args.compression_level),
                    include_plain_jsonl=bool(args.include_plain_jsonl),
                    vacuum_sqlite=False,
                    sqlite_min_reclaim_mb=float(args.sqlite_min_reclaim_mb),
                    sqlite_min_reclaim_ratio=float(args.sqlite_min_reclaim_ratio),
                    sqlite_inventory_limit=int(args.sqlite_inventory_limit),
                    manifest_path=Path(args.manifest_path).expanduser() if args.manifest_path else None,
                )
                payload.update(
                    {
                        "apply": True,
                        "overall_status": "ready",
                        "ok": True,
                        "no_op_reason": "no_stable_file_candidates",
                        "writer_handoff": {
                            "ready": True,
                            "status": "not_needed_no_file_work",
                            "waited_seconds": 0.0,
                            "poll_count": 0,
                        },
                        "next_action": "wait for the next bounded cold-archive retention cadence",
                    }
                )
        writer_state = writer_state_snapshot(PROJECT_ROOT)
        if payload is None and writer_blocks_compaction(
            writer_state,
            allow_active_writer=bool(args.allow_active_writer),
        ) and not coordinate_writer_handoff:
            payload = {
                "timestamp_utc": iso_now(),
                "schema_version": 1,
                "ok": True,
                "overall_status": "deferred_writer_busy",
                "apply": True,
                "writer_state": writer_state,
                "next_action": "retry after the active single-writer cycle completes",
            }
            write_payload(Path(args.out_file).expanduser(), payload)
            print(json.dumps(payload, ensure_ascii=True))
            return 0
        if payload is None:
            lock_handle, owner = _acquire_lock(Path(args.lock_path).expanduser())
            if lock_handle is None:
                payload = {
                    "timestamp_utc": iso_now(),
                    "schema_version": 1,
                    "ok": False,
                    "overall_status": "busy",
                    "lock_owner": owner,
                }
                write_payload(Path(args.out_file).expanduser(), payload)
                print(json.dumps(payload, ensure_ascii=True))
                return 2

    try:
        if payload is None and args.apply and coordinate_writer_handoff:
            existing_hold = maintenance_hold_snapshot(PROJECT_ROOT)
            if bool(existing_hold.get("active", False)):
                payload = {
                    "timestamp_utc": iso_now(),
                    "schema_version": 1,
                    "ok": True,
                    "overall_status": "deferred_existing_maintenance_hold",
                    "apply": True,
                    "maintenance_hold": {
                        key: existing_hold.get(key)
                        for key in ("path", "reason", "owner", "engaged_at_utc", "expires_at_utc")
                    },
                    "next_action": "retry after the existing runtime maintenance hold is released",
                }
            else:
                engaged = engage_maintenance_hold(
                    PROJECT_ROOT,
                    reason="cold_archive_compaction",
                    owner="cold_archive_compactor",
                    ttl_seconds=max(int(args.maintenance_hold_ttl_seconds), 60),
                )
                maintenance_hold_token = str(engaged.get("token") or "")
                writer_handoff = wait_for_writer_handoff(
                    PROJECT_ROOT,
                    timeout_seconds=max(float(args.writer_handoff_timeout_seconds), 0.0),
                    poll_seconds=max(float(args.writer_handoff_poll_seconds), 0.1),
                )
                if not bool(writer_handoff.get("ready", False)):
                    payload = {
                        "timestamp_utc": iso_now(),
                        "schema_version": 1,
                        "ok": True,
                        "overall_status": "deferred_writer_handoff_timeout",
                        "apply": True,
                        "writer_handoff": writer_handoff,
                        "next_action": "retry the bounded maintenance handoff on the next storage cadence",
                    }

        if payload is None:
            payload = build_payload(
                archive_root=Path(args.archive_root),
                apply=bool(args.apply),
                min_age_hours=float(args.min_age_hours),
                max_files=int(args.max_files),
                max_raw_gb=float(args.max_raw_gb),
                compression_level=int(args.compression_level),
                include_plain_jsonl=bool(args.include_plain_jsonl),
                vacuum_sqlite=bool(args.vacuum_sqlite),
                sqlite_min_reclaim_mb=float(args.sqlite_min_reclaim_mb),
                sqlite_min_reclaim_ratio=float(args.sqlite_min_reclaim_ratio),
                sqlite_inventory_limit=int(args.sqlite_inventory_limit),
                manifest_path=Path(args.manifest_path).expanduser() if args.manifest_path else None,
            )
            if writer_handoff:
                payload["writer_handoff"] = writer_handoff
    finally:
        if maintenance_hold_token:
            released = release_maintenance_hold(PROJECT_ROOT, expected_token=maintenance_hold_token)
            maintenance_hold_result = {
                "engaged": True,
                "released": bool(released.get("released", False)),
                "reason": "cold_archive_compaction",
                "release_error": str(released.get("release_error") or ""),
            }
        if lock_handle is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            lock_handle.close()

    assert payload is not None
    if maintenance_hold_result:
        payload["maintenance_hold"] = maintenance_hold_result
        if not bool(maintenance_hold_result.get("released", False)):
            payload["ok"] = False
            payload["overall_status"] = "degraded"
            payload["next_action"] = "release the expired cold-archive maintenance hold before resuming writers"
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        print(
            "cold_archive_compactor "
            f"status={payload.get('overall_status', '')} "
            f"selected_gb={summary.get('selected_jsonl_gb', 0)} "
            f"released_gb={summary.get('released_gb', 0)}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
