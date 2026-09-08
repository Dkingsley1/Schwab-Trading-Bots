#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.storage_router import inspect_storage_path

DEFAULT_HEALTH_PATH = (
    PROJECT_ROOT
    / "governance"
    / "health"
    / "raw_training_compaction_intelligence_latest.json"
)
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "governance"
    / "training"
    / "raw_training_compaction_manifest_latest.json"
)
DEFAULT_SOURCE_QUEUE_PATH = (
    PROJECT_ROOT / "governance" / "training" / "raw_training_source_queue_latest.jsonl"
)
DEFAULT_ELIGIBLE_QUEUE_PATH = (
    PROJECT_ROOT
    / "governance"
    / "training"
    / "raw_training_eligible_source_queue_latest.jsonl"
)
DEFAULT_HISTORY_DIR = (
    PROJECT_ROOT / "governance" / "training" / "raw_training_compaction_history"
)
DEFAULT_BOT_LOGS_ROOT = Path(
    os.environ.get("BOT_LOGS_ROOT", "/Volumes/BOT_LOGS/schwab_trading_bot")
)
LOCAL_FALLBACK_DIR_NAME = "local_fallback_storage"
DEFAULT_MATERIAL_COMPACTION_GB = 1.0
PROTECTED_VOLUME_PREFIXES = ("/Volumes/VIDEO",)
EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv314",
    "__pycache__",
    "node_modules",
}
TRAINING_PATH_HINTS = {
    "attribution",
    "bot",
    "coinbase",
    "collector",
    "context",
    "crypto",
    "decision",
    "execution",
    "feature",
    "fill",
    "intent",
    "market",
    "micro",
    "order",
    "paper",
    "quote",
    "schwab",
    "shadow",
    "signal",
    "sleeve",
    "strategy",
    "trade",
    "training",
}
BLOCKED_PATH_HINTS = {
    "local_fallback",
    "fallback_local",
    "reconciliation_debt",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _grade(score: float) -> str:
    if score >= 99:
        return "A+"
    if score >= 97:
        return "A+"
    if score >= 93:
        return "A"
    if score >= 90:
        return "A-"
    if score >= 87:
        return "B+"
    if score >= 83:
        return "B"
    if score >= 80:
        return "B-"
    if score >= 77:
        return "C+"
    if score >= 73:
        return "C"
    if score >= 70:
        return "C-"
    if score >= 60:
        return "D"
    return "F"


def _status_from_score(score: float) -> str:
    if score >= 90:
        return "ready"
    if score >= 75:
        return "needs_work"
    return "blocked"


def _is_under_protected_volume(path: Path) -> bool:
    return inspect_storage_path(path).get("status") not in {"present", "missing"}


def _path_parts_lower(path: Path) -> set[str]:
    return {part.lower() for part in path.parts}


def _contains_hint(path: Path, hints: set[str]) -> bool:
    text = str(path).lower().replace("-", "_")
    parts = _path_parts_lower(path)
    for hint in hints:
        if hint in parts or hint in text:
            return True
    return False


def _is_live_local_fallback_artifact(path: Path) -> bool:
    name = path.name.lower()
    text = str(path).lower().replace("-", "_")
    parts = _path_parts_lower(path)
    if ".local_fallback" in name or name.endswith(".local_fallback"):
        return True
    if "local_fallback" in parts:
        return True
    if "fallback_local" in parts or "fallback_local" in text:
        return True
    if "local_fallback_storage" in parts:
        # Route repair backups are cold evidence from a completed failback, not a live writer fallback lane.
        return not any("route_repair_backup" in part for part in parts)
    return "reconciliation_debt" in parts or "reconciliation_debt" in text


def _is_archived_fallback_evidence(path: Path, scan_root: Path) -> bool:
    """Identify fallback evidence already preserved beneath the scanned storage root."""
    try:
        relative = path.relative_to(scan_root)
    except ValueError:
        return False
    parts = tuple(part.lower() for part in relative.parts)
    return bool(parts and parts[0] == LOCAL_FALLBACK_DIR_NAME)


def _date_token_matches_current_day(path: Path, today: str) -> bool:
    dashed = f"{today[:4]}-{today[4:6]}-{today[6:]}"
    text = str(path)
    return today in text or dashed in text


def _iter_jsonl_files(root: Path) -> Iterable[Path]:
    if _is_under_protected_volume(root):
        return
    if not root.exists():
        return
    pending = [root]
    while pending:
        current = pending.pop()
        if _is_under_protected_volume(current):
            continue
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if (
                            entry.name not in EXCLUDED_DIR_NAMES
                            and not entry.name.startswith(".")
                        ):
                            pending.append(Path(entry.path))
                    elif entry.name.endswith(".jsonl") and entry.is_file(
                        follow_symlinks=False
                    ):
                        yield Path(entry.path)
        except OSError:
            continue


def _default_scan_roots() -> list[Path]:
    raw_env = os.environ.get("RAW_TRAINING_SCAN_ROOTS", "").strip()
    roots: list[Path] = []
    if raw_env:
        roots.extend(
            Path(part).expanduser()
            for part in raw_env.split(os.pathsep)
            if part.strip()
        )
    if DEFAULT_BOT_LOGS_ROOT.exists():
        roots.append(DEFAULT_BOT_LOGS_ROOT)
    for candidate in (
        PROJECT_ROOT / "bot_logs",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "data" / "raw",
    ):
        if candidate.exists():
            roots.append(candidate)
    if not roots:
        roots.append(DEFAULT_BOT_LOGS_ROOT)
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def _prefix_sha256(path: Path, sample_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as handle:
            chunk = handle.read(max(0, sample_bytes))
    except (OSError, PermissionError):
        return "", 0
    digest.update(chunk)
    total += len(chunk)
    return digest.hexdigest(), total


def _compressed_sibling(path: Path) -> Path:
    return path.with_name(path.name + ".gz")


def _raw_training_sibling(path: Path) -> Path:
    return path.with_name(path.name + ".raw-training.gz")


def _classify_row(
    path: Path,
    *,
    scan_root: Path,
    now_ts: float,
    today: str,
    min_age_hours: float,
    sample_bytes: int,
) -> dict[str, Any]:
    stat = path.stat()
    size_bytes = int(stat.st_size)
    age_seconds = max(0.0, now_ts - float(stat.st_mtime))
    age_hours = age_seconds / 3600.0
    current_day = _date_token_matches_current_day(path, today) or age_hours < min(
        6.0, min_age_hours
    )
    active_latest_artifact = path.name.lower().endswith("_latest.jsonl")
    archived_fallback_evidence = _is_archived_fallback_evidence(path, scan_root)
    local_fallback = bool(
        _is_live_local_fallback_artifact(path) and not archived_fallback_evidence
    )
    protected = _is_under_protected_volume(path)
    training_candidate = bool(
        _contains_hint(path, TRAINING_PATH_HINTS) or size_bytes > 0
    )
    sibling = _compressed_sibling(path)
    already_compressed_sibling = sibling.exists() and sibling.stat().st_size > 0
    old_enough = age_hours >= min_age_hours
    queue_blockers: list[str] = []
    if protected:
        queue_blockers.append("protected_volume")
    if local_fallback:
        queue_blockers.append("local_fallback_reconciliation_required")
    if size_bytes <= 0:
        queue_blockers.append("empty_source")
    if not training_candidate:
        queue_blockers.append("low_training_relevance")
    compaction_blockers = list(queue_blockers)
    if current_day:
        compaction_blockers.append("current_day_or_recent_source_protected")
    if active_latest_artifact:
        compaction_blockers.append("active_latest_artifact_protected")
    if not old_enough:
        compaction_blockers.append("min_age_not_met")
    prefix_hash, hashed_bytes = _prefix_sha256(path, sample_bytes)
    training_eligible = bool(
        training_candidate and not protected and not local_fallback and size_bytes > 0
    )
    compression_candidate = bool(
        training_eligible
        and not current_day
        and not active_latest_artifact
        and old_enough
    )
    clear_action = "compress_then_remove_raw"
    if already_compressed_sibling and compression_candidate:
        clear_action = "remove_raw_duplicate_of_compressed_sibling"
    elif not compression_candidate:
        clear_action = "queue_only"
    queue_state = (
        "queued_training_source" if training_eligible else "blocked_training_source"
    )
    if local_fallback:
        queue_state = "blocked_local_fallback_reconciliation"
    if protected:
        queue_state = "blocked_protected_volume"
    if current_day and training_eligible:
        queue_state = "queued_training_source_current_day_protected"
    return {
        "queue_kind": "raw_training_source_manifest",
        "queue_state": queue_state,
        "path": str(path),
        "compressed_path": str(sibling),
        "copy_raw_payload": False,
        "manifest_only": True,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 6),
        "mtime_utc": datetime.fromtimestamp(
            float(stat.st_mtime), timezone.utc
        ).isoformat(),
        "age_hours": round(age_hours, 3),
        "current_day_protected": bool(current_day),
        "active_latest_artifact_protected": bool(active_latest_artifact),
        "training_candidate": bool(training_candidate),
        "training_eligible": bool(training_eligible),
        "compression_candidate": bool(compression_candidate),
        "already_compressed_sibling": bool(already_compressed_sibling),
        "local_fallback_reconciliation_required": bool(local_fallback),
        "archived_fallback_evidence": bool(archived_fallback_evidence),
        "protected_volume": bool(protected),
        "clear_action": clear_action,
        "queue_blockers": queue_blockers,
        "compaction_blockers": compaction_blockers,
        "prefix_sha256": prefix_hash,
        "prefix_hashed_bytes": hashed_bytes,
        "training_ingress": "runtime_training_snapshot",
        "labeling_ingress": "training_labeling_intelligence",
        "storage_ingress": "raw_training_compaction_intelligence",
    }


def _select_batch(
    rows: list[dict[str, Any]], max_files: int, max_gb: float, jumbo_gb: float = 0.0
) -> list[dict[str, Any]]:
    max_files = max(0, int(max_files))
    max_bytes = max(0, int(max_gb * 1024 * 1024 * 1024))
    jumbo_bytes = max(0, int(jumbo_gb * 1024 * 1024 * 1024))
    candidates = [
        row
        for row in rows
        if row.get("compression_candidate")
        and not row.get("protected_volume")
        and not row.get("current_day_protected")
        and not row.get("local_fallback_reconciliation_required")
    ]
    candidates.sort(
        key=lambda row: (
            _safe_float(row.get("size_bytes"), 0.0),
            bool(row.get("already_compressed_sibling")),
            _safe_float(row.get("age_hours"), 0.0),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    used_bytes = 0
    for row in candidates:
        if len(selected) >= max_files:
            break
        size_bytes = _safe_int(row.get("size_bytes"), 0)
        if max_bytes and used_bytes + size_bytes > max_bytes:
            if not selected and jumbo_bytes > 0 and size_bytes <= jumbo_bytes:
                row["selected_over_wave_cap"] = True
                row["selection_reason"] = "single_jumbo_raw_compaction_candidate"
                selected.append(row)
                used_bytes += size_bytes
                break
            continue
        selected.append(row)
        used_bytes += size_bytes
    if not selected and max_files > 0 and max_bytes > 0:
        for row in sorted(
            candidates, key=lambda item: _safe_int(item.get("size_bytes"), 0)
        ):
            size_bytes = _safe_int(row.get("size_bytes"), 0)
            if size_bytes <= max_bytes:
                selected.append(row)
                break
    return selected


def _verify_gzip(path: Path) -> bool:
    try:
        if _is_under_protected_volume(path):
            return False
        with gzip.open(path, "rb") as handle:
            _digest_stream(handle, time.monotonic() + 300)
        return path.exists() and path.stat().st_size > 0
    except Exception:
        return False


def _gzip_prefix_sha256(path: Path, sample_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    try:
        with gzip.open(path, "rb") as handle:
            chunk = handle.read(max(0, sample_bytes))
    except Exception:
        return "", 0
    digest.update(chunk)
    return digest.hexdigest(), len(chunk)


def _file_identity(path: Path) -> tuple[int, ...]:
    if _is_under_protected_volume(path):
        raise ValueError("protected_or_unavailable_route")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("not_regular_file")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _digest_stream(handle: Any, deadline: float) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError("compaction_verification_deadline")
        chunk = handle.read(1024 * 1024)
        if not chunk:
            return digest.hexdigest(), size
        digest.update(chunk)
        size += len(chunk)


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _compaction_reserve_bytes() -> int:
    reserve = float(os.getenv("BOT_LOCAL_STORAGE_EMERGENCY_FREE_GB", "16"))
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("invalid_compaction_reserve")
    return int(max(reserve, 16.0) * 1024**3)


def _compress_and_clear(
    path: Path, compressed_path: Path, *, compress_level: int, keep_raw: bool
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    tmp_path = None
    try:
        identity = _file_identity(path)
        before_bytes = identity[2]
        if _is_under_protected_volume(compressed_path):
            raise ValueError("protected_or_unavailable_route")
        if compressed_path.exists():
            return _remove_duplicate_raw(
                path,
                compressed_path,
                expected_prefix_sha256="",
                sample_bytes=0,
                keep_raw=keep_raw,
            )
        compressed_path.parent.mkdir(parents=True, exist_ok=True)
        # Recovery compaction must retain emergency headroom even for incompressible input.
        if (
            shutil.disk_usage(compressed_path.parent).free
            < before_bytes * 1.01 + _compaction_reserve_bytes()
        ):
            raise RuntimeError("insufficient_compaction_scratch_reserve")
        deadline = time.monotonic() + 300
        fd, name = tempfile.mkstemp(
            prefix=".raw_compact_", suffix=".tmp", dir=compressed_path.parent
        )
        tmp_path = Path(name)
        digest = hashlib.sha256()
        copied = 0
        with os.fdopen(fd, "wb") as raw_out, path.open("rb") as src:
            with gzip.GzipFile(
                fileobj=raw_out, mode="wb", compresslevel=compress_level, mtime=0
            ) as dst:
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError("compaction_deadline")
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > before_bytes:
                        raise RuntimeError("source_changed_during_compaction")
                    digest.update(chunk)
                    dst.write(chunk)
            raw_out.flush()
            os.fsync(raw_out.fileno())
        with gzip.open(tmp_path, "rb") as verify:
            restored_digest, restored_bytes = _digest_stream(verify, deadline)
        if (restored_digest, restored_bytes) != (
            digest.hexdigest(),
            before_bytes,
        ) or copied != before_bytes:
            raise RuntimeError("gzip_restore_proof_mismatch")
        if _file_identity(path) != identity:
            raise RuntimeError("source_changed_during_compaction")
        # Atomic no-clobber publication preserves any competing or divergent archive.
        os.link(tmp_path, compressed_path)
        _sync_directory(compressed_path.parent)
        target_identity = _file_identity(compressed_path)
        raw_removed = False
        if not keep_raw:
            if (
                _file_identity(path) != identity
                or _file_identity(compressed_path) != target_identity
            ):
                raise RuntimeError("source_or_target_changed_before_release")
            path.unlink()
            raw_removed = True
        after_bytes = compressed_path.stat().st_size
        duration_seconds = round(
            (datetime.now(timezone.utc) - started).total_seconds(), 3
        )
        return {
            "path": str(path),
            "compressed_path": str(compressed_path),
            "status": "ok",
            "action": (
                "compress_then_remove_raw" if raw_removed else "compress_keep_raw"
            ),
            "raw_removed": raw_removed,
            "before_bytes": before_bytes,
            "compressed_bytes": after_bytes,
            "sha256_uncompressed": restored_digest,
            "verified_raw_bytes": restored_bytes,
            "verification_basis": "full_gzip_restore_sha256_and_stable_source_identity",
            "estimated_raw_bytes_cleared": before_bytes if raw_removed else 0,
            "duration_seconds": duration_seconds,
        }
    except Exception as exc:
        try:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return {
            "path": str(path),
            "compressed_path": str(compressed_path),
            "status": "failed",
            "reason": f"{type(exc).__name__}:{exc}",
            "raw_removed": False,
            "estimated_raw_bytes_cleared": 0,
        }
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _remove_duplicate_raw(
    path: Path,
    compressed_path: Path,
    *,
    expected_prefix_sha256: str,
    sample_bytes: int,
    keep_raw: bool = False,
) -> dict[str, Any]:
    try:
        identity = _file_identity(path)
        target_identity = _file_identity(compressed_path)
        before_bytes = identity[2]
        deadline = time.monotonic() + 300
        with path.open("rb") as source:
            source_digest, source_bytes = _digest_stream(source, deadline)
        with gzip.open(compressed_path, "rb") as restored:
            restored_digest, restored_bytes = _digest_stream(restored, deadline)
        if (
            _file_identity(path) != identity
            or _file_identity(compressed_path) != target_identity
        ):
            raise RuntimeError("source_or_target_changed_during_verification")
        if (source_digest, source_bytes) != (restored_digest, restored_bytes):
            reason = "compressed_sibling_content_mismatch"
            if expected_prefix_sha256 and sample_bytes > 0:
                prefix, _ = _gzip_prefix_sha256(compressed_path, sample_bytes)
                if prefix != expected_prefix_sha256:
                    reason = "compressed_sibling_prefix_mismatch"
            return {
                "path": str(path),
                "compressed_path": str(compressed_path),
                "status": "skipped",
                "reason": reason,
                "raw_removed": False,
                "estimated_raw_bytes_cleared": 0,
            }
        if not keep_raw:
            with compressed_path.open("rb") as durable:
                os.fsync(durable.fileno())
            _sync_directory(compressed_path.parent)
            if (
                _file_identity(path) != identity
                or _file_identity(compressed_path) != target_identity
            ):
                raise RuntimeError("source_or_target_changed_before_release")
            path.unlink()
    except Exception as exc:
        return {
            "path": str(path),
            "compressed_path": str(compressed_path),
            "status": "failed",
            "reason": f"compressed_sibling_failed_verification:{type(exc).__name__}:{exc}",
            "raw_removed": False,
            "estimated_raw_bytes_cleared": 0,
        }
    return {
        "path": str(path),
        "compressed_path": str(compressed_path),
        "status": "ok",
        "action": (
            "compress_keep_raw"
            if keep_raw
            else "remove_raw_duplicate_of_compressed_sibling"
        ),
        "raw_removed": not keep_raw,
        "before_bytes": before_bytes,
        "compressed_bytes": compressed_path.stat().st_size,
        "estimated_raw_bytes_cleared": 0 if keep_raw else before_bytes,
        "sha256_uncompressed": source_digest,
        "verified_raw_bytes": source_bytes,
        "verification_basis": "full_gzip_restore_sha256_and_stable_source_identity",
        "duration_seconds": 0.0,
    }


def _apply_batch(
    rows: list[dict[str, Any]],
    *,
    compress_level: int,
    keep_raw: bool,
    compaction_workers: int = 1,
) -> list[dict[str, Any]]:
    records_by_index: dict[int, dict[str, Any]] = {}
    compression_jobs: list[tuple[int, Path, Path]] = []
    for index, row in enumerate(rows):
        path = Path(str(row.get("path", "")))
        compressed_path = Path(str(row.get("compressed_path", "")))
        if _is_under_protected_volume(path) or _is_under_protected_volume(
            compressed_path
        ):
            records_by_index[index] = {
                "path": str(path),
                "status": "skipped",
                "reason": "protected_volume",
                "raw_removed": False,
            }
            continue
        if not path.exists():
            records_by_index[index] = {
                "path": str(path),
                "compressed_path": str(compressed_path),
                "status": "skipped",
                "reason": "raw_source_missing",
                "raw_removed": False,
                "estimated_raw_bytes_cleared": 0,
            }
            continue
        if _is_under_protected_volume(path):
            records_by_index[index] = {
                "path": str(path),
                "compressed_path": str(compressed_path),
                "status": "skipped",
                "reason": "protected_volume",
                "raw_removed": False,
                "estimated_raw_bytes_cleared": 0,
            }
            continue
        if row.get("already_compressed_sibling"):
            duplicate_record = _remove_duplicate_raw(
                path,
                compressed_path,
                expected_prefix_sha256=str(row.get("prefix_sha256", "")),
                sample_bytes=_safe_int(row.get("prefix_hashed_bytes"), 0),
                keep_raw=keep_raw,
            )
            if duplicate_record.get("reason") in {
                "compressed_sibling_prefix_mismatch",
                "compressed_sibling_content_mismatch",
            }:
                repacked_path = _raw_training_sibling(path)
                repack_record = _compress_and_clear(
                    path,
                    repacked_path,
                    compress_level=compress_level,
                    keep_raw=keep_raw,
                )
                repack_record["original_compressed_path"] = str(compressed_path)
                repack_record["original_compressed_sibling_reason"] = duplicate_record[
                    "reason"
                ]
                if repack_record.get("status") == "ok":
                    repack_record["action"] = (
                        "repack_mismatched_sibling_then_remove_raw"
                        if repack_record.get("raw_removed")
                        else "repack_mismatched_sibling_keep_raw"
                    )
                records_by_index[index] = repack_record
            else:
                records_by_index[index] = duplicate_record
            continue
        compression_jobs.append((index, path, compressed_path))

    worker_count = max(int(compaction_workers), 1)
    if worker_count <= 1 or len(compression_jobs) <= 1:
        for index, path, compressed_path in compression_jobs:
            records_by_index[index] = _compress_and_clear(
                path,
                compressed_path,
                compress_level=compress_level,
                keep_raw=keep_raw,
            )
    else:
        worker_count = min(worker_count, len(compression_jobs))
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(
                    _compress_and_clear,
                    path,
                    compressed_path,
                    compress_level=compress_level,
                    keep_raw=keep_raw,
                ): index
                for index, path, compressed_path in compression_jobs
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    records_by_index[index] = future.result()
                except Exception as exc:
                    _, path, compressed_path = next(
                        job for job in compression_jobs if job[0] == index
                    )
                    records_by_index[index] = {
                        "path": str(path),
                        "compressed_path": str(compressed_path),
                        "status": "failed",
                        "reason": f"{type(exc).__name__}:{exc}",
                        "raw_removed": False,
                        "estimated_raw_bytes_cleared": 0,
                    }
    return [records_by_index[index] for index in sorted(records_by_index)]


def _line_count(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for _line in handle)
    except Exception:
        return 0


def _training_clearance_snapshot(
    scan_roots: list[Path] | None = None,
    *,
    snapshot_health_path: Path | None = None,
) -> dict[str, Any]:
    health_path = (
        snapshot_health_path
        or PROJECT_ROOT
        / "governance"
        / "health"
        / "runtime_training_snapshot_latest.json"
    )
    snapshot_health: dict[str, Any] = {}
    if health_path.exists():
        try:
            loaded = json.loads(health_path.read_text(encoding="utf-8"))
            snapshot_health = loaded if isinstance(loaded, dict) else {}
        except Exception:
            snapshot_health = {}
    contracted_candidate = Path(
        str(snapshot_health.get("rows_path") or "")
    ).expanduser()
    explicit_root_candidates = [
        root.expanduser()
        / "exports"
        / "training"
        / "runtime_training_snapshot_latest.jsonl"
        for root in (scan_roots or [])
    ]
    fallback_candidates = [
        PROJECT_ROOT
        / "exports"
        / "training"
        / "runtime_training_snapshot_latest.jsonl",
        DEFAULT_BOT_LOGS_ROOT
        / "exports"
        / "training"
        / "runtime_training_snapshot_latest.jsonl",
    ]
    candidates = [contracted_candidate, *explicit_root_candidates, *fallback_candidates]
    resolved_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not str(candidate):
            continue
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        resolved_candidates.append(candidate)
    existing = [
        path
        for path in resolved_candidates
        if path.is_file() and path.stat().st_size > 0
    ]
    contracted_existing = (
        contracted_candidate if contracted_candidate in existing else None
    )
    explicit_existing = [path for path in explicit_root_candidates if path in existing]
    fallback_existing = [path for path in fallback_candidates if path in existing]
    if contracted_existing is not None:
        runtime_path = contracted_existing
    elif explicit_existing:
        runtime_path = max(explicit_existing, key=lambda path: path.stat().st_mtime)
    elif fallback_existing:
        runtime_path = max(fallback_existing, key=lambda path: path.stat().st_mtime)
    else:
        runtime_path = resolved_candidates[0]
    contracted_rows = _safe_int(snapshot_health.get("row_count"), 0)
    runtime_rows = (
        contracted_rows
        if contracted_rows > 0 and existing
        else (_line_count(runtime_path) if runtime_path.exists() else 0)
    )
    quality_path = (
        PROJECT_ROOT / "governance" / "health" / "training_quality_control_latest.json"
    )
    if not quality_path.exists():
        quality_path = (
            PROJECT_ROOT / "governance" / "health" / "training_quality_latest.json"
        )
    quality_payload: dict[str, Any] = {}
    if quality_path.exists():
        try:
            loaded = json.loads(quality_path.read_text(encoding="utf-8"))
            quality_payload = loaded if isinstance(loaded, dict) else {}
        except Exception:
            quality_payload = {}
    return {
        "runtime_snapshot_path": str(runtime_path),
        "runtime_snapshot_ready": runtime_rows > 0,
        "runtime_snapshot_row_count": runtime_rows,
        "runtime_snapshot_sequence_count": _safe_int(
            snapshot_health.get("sequence_count"), 0
        ),
        "runtime_snapshot_health_path": str(health_path),
        "runtime_snapshot_candidate_paths": [str(path) for path in resolved_candidates],
        "runtime_snapshot_source": (
            "health_contract_and_resolved_rows"
            if contracted_rows > 0 and existing
            else "bounded_row_probe"
        ),
        "training_quality_status": quality_payload.get("overall_status", ""),
        "training_quality_score": quality_payload.get(
            "training_quality_score",
            quality_payload.get("overall_score", quality_payload.get("score", 0)),
        ),
        "training_quality_path": str(quality_path),
        "raw_hard_delete_allowed": False,
        "gzip_compaction_allowed": True,
        "raw_hard_delete_reason": "raw evidence is only cleared by verified gzip compaction or verified compressed sibling removal",
    }


def _summary(
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    apply_records: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_count = len(rows)
    raw_bytes = sum(_safe_int(row.get("size_bytes"), 0) for row in rows)
    training_candidate_rows = [row for row in rows if row.get("training_candidate")]
    eligible_rows = [row for row in rows if row.get("training_eligible")]
    compression_rows = [row for row in rows if row.get("compression_candidate")]
    sibling_rows = [
        row for row in compression_rows if row.get("already_compressed_sibling")
    ]
    current_day_rows = [row for row in rows if row.get("current_day_protected")]
    active_latest_rows = [
        row for row in rows if row.get("active_latest_artifact_protected")
    ]
    local_fallback_rows = [
        row for row in rows if row.get("local_fallback_reconciliation_required")
    ]
    archived_fallback_rows = [
        row for row in rows if row.get("archived_fallback_evidence")
    ]
    protected_rows = [row for row in rows if row.get("protected_volume")]
    cleared_bytes = sum(
        _safe_int(record.get("estimated_raw_bytes_cleared"), 0)
        for record in apply_records
        if record.get("status") == "ok"
    )
    failed_apply_count = sum(
        1 for record in apply_records if record.get("status") == "failed"
    )
    return {
        "raw_jsonl_count": raw_count,
        "raw_jsonl_gb": round(raw_bytes / (1024**3), 6),
        "training_candidate_count": len(training_candidate_rows),
        "training_candidate_gb": round(
            sum(_safe_int(row.get("size_bytes"), 0) for row in training_candidate_rows)
            / (1024**3),
            6,
        ),
        "eligible_training_source_count": len(eligible_rows),
        "eligible_training_source_gb": round(
            sum(_safe_int(row.get("size_bytes"), 0) for row in eligible_rows)
            / (1024**3),
            6,
        ),
        "compression_candidate_count": len(compression_rows),
        "compression_candidate_gb": round(
            sum(_safe_int(row.get("size_bytes"), 0) for row in compression_rows)
            / (1024**3),
            6,
        ),
        "already_compressed_sibling_count": len(sibling_rows),
        "current_day_protected_count": len(current_day_rows),
        "active_latest_artifact_protected_count": len(active_latest_rows),
        "local_fallback_reconciliation_count": len(local_fallback_rows),
        "archived_fallback_evidence_count": len(archived_fallback_rows),
        "protected_volume_count": len(protected_rows),
        "selected_compaction_count": len(selected),
        "selected_compaction_gb": round(
            sum(_safe_int(row.get("size_bytes"), 0) for row in selected) / (1024**3), 6
        ),
        "apply_record_count": len(apply_records),
        "apply_failed_count": failed_apply_count,
        "raw_bytes_cleared": cleared_bytes,
        "raw_gb_cleared": round(cleared_bytes / (1024**3), 6),
    }


def _score_report(
    summary: dict[str, Any], apply_requested: bool
) -> tuple[float, str, list[str], list[str]]:
    raw_count = _safe_int(summary.get("raw_jsonl_count"), 0)
    queued_count = _safe_int(summary.get("raw_source_queue_count"), raw_count)
    compression_count = _safe_int(summary.get("compression_candidate_count"), 0)
    compression_gb = _safe_float(summary.get("compression_candidate_gb"), 0.0)
    material_floor_gb = max(
        _safe_float(
            os.getenv("BOT_RAW_TRAINING_MATERIAL_COMPACTION_GB"),
            DEFAULT_MATERIAL_COMPACTION_GB,
        ),
        0.0,
    )
    submaterial_tail = bool(
        compression_count > 0 and compression_gb < material_floor_gb
    )
    selected_count = _safe_int(summary.get("selected_compaction_count"), 0)
    failed_count = _safe_int(summary.get("apply_failed_count"), 0)
    blockers: list[str] = []
    next_actions: list[str] = []
    if raw_count <= 0:
        score = 100.0
        next_actions.append(
            "no raw JSONL sources found; training queue is empty and raw backlog is clear"
        )
        return score, "queued_and_clear", blockers, next_actions
    queue_coverage = min(queued_count / max(1, raw_count), 1.0)
    score = 82.0 + min(10.0, queue_coverage * 10.0)
    if compression_count <= 0:
        score += 8.0
        next_actions.append("no eligible old raw sources need compaction right now")
    elif apply_requested and selected_count > 0 and failed_count == 0:
        score += 7.0
        next_actions.append(
            "continue bounded compaction waves until compression candidates reach zero"
        )
    elif submaterial_tail:
        score += 8.0
        next_actions.append(
            "keep the sub-material compaction tail manifest-indexed until it crosses the bounded wave floor"
        )
    elif apply_requested and selected_count <= 0:
        score -= 15.0
        blockers.append("no_compaction_batch_selected_under_current_caps")
        next_actions.append(
            "increase --max-gb or lower active intake before clearing larger raw files"
        )
    else:
        score -= 12.0
        blockers.append("raw_compaction_not_applied")
        next_actions.append(
            "run with --apply to clear the queued old raw sources by verified gzip compaction"
        )
    if failed_count > 0:
        score -= min(15.0, failed_count * 3.0)
        blockers.append("compaction_apply_failures")
    return (
        max(0.0, min(100.0, score)),
        _status_from_score(score),
        blockers,
        next_actions,
    )


def _build_rows(
    scan_roots: list[Path], *, min_age_hours: float, sample_bytes: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    now_ts = datetime.now(timezone.utc).timestamp()
    today = _utc_day()
    roots_payload: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in scan_roots:
        root = root.expanduser()
        protected = _is_under_protected_volume(root)
        roots_payload.append(
            {
                "path": str(root),
                "exists": False if protected else root.exists(),
                "protected": protected,
            }
        )
        if protected or not root.exists():
            continue
        for path in _iter_jsonl_files(root):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            try:
                rows.append(
                    _classify_row(
                        path,
                        scan_root=root,
                        now_ts=now_ts,
                        today=today,
                        min_age_hours=min_age_hours,
                        sample_bytes=sample_bytes,
                    )
                )
            except (OSError, PermissionError):
                continue
    rows.sort(
        key=lambda row: (
            _safe_int(row.get("size_bytes"), 0),
            _safe_float(row.get("age_hours"), 0.0),
        ),
        reverse=True,
    )
    return rows, roots_payload


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    scan_roots: list[Path]
    if args.scan_root:
        scan_roots = [Path(raw).expanduser() for raw in args.scan_root]
    elif args.bot_logs_root:
        scan_roots = [Path(args.bot_logs_root).expanduser()]
    else:
        scan_roots = _default_scan_roots()

    rows, roots_payload = _build_rows(
        scan_roots, min_age_hours=args.min_age_hours, sample_bytes=args.sample_bytes
    )
    eligible_rows = [row for row in rows if row.get("training_eligible")]
    selected = _select_batch(
        rows, max_files=args.max_files, max_gb=args.max_gb, jumbo_gb=args.jumbo_gb
    )
    compaction_workers = max(_safe_int(getattr(args, "compaction_workers", 1), 1), 1)
    apply_records: list[dict[str, Any]] = []
    if args.apply:
        apply_records = _apply_batch(
            selected,
            compress_level=args.compress_level,
            keep_raw=args.keep_raw_after_compress,
            compaction_workers=compaction_workers,
        )

    source_count = _write_jsonl(Path(args.source_queue_path), rows)
    eligible_count = _write_jsonl(Path(args.eligible_queue_path), eligible_rows)
    summary = _summary(rows, selected, apply_records)
    material_floor_gb = max(
        _safe_float(
            os.getenv("BOT_RAW_TRAINING_MATERIAL_COMPACTION_GB"),
            DEFAULT_MATERIAL_COMPACTION_GB,
        ),
        0.0,
    )
    compression_candidate_gb = _safe_float(summary.get("compression_candidate_gb"), 0.0)
    summary.update(
        {
            "raw_source_queue_count": source_count,
            "raw_source_queue_coverage_ratio": round(
                source_count / max(len(rows), 1), 6
            ),
            "eligible_source_selection_ratio": round(
                eligible_count / max(len(rows), 1), 6
            ),
            "material_compaction_floor_gb": round(material_floor_gb, 6),
            "submaterial_compaction_tail": bool(
                _safe_int(summary.get("compression_candidate_count"), 0) > 0
                and compression_candidate_gb < material_floor_gb
            ),
        }
    )
    score, status, blockers, next_actions = _score_report(summary, bool(args.apply))
    grade = _grade(score)
    timestamp = _utc_now()
    training_clearance = _training_clearance_snapshot(scan_roots)
    decision_packet = {
        "action": (
            "queue_all_raw_sources_then_clear_eligible_by_verified_gzip"
            if args.apply
            else "queue_all_raw_sources_dry_run"
        ),
        "raw_sources_queued": source_count,
        "eligible_sources_queued": eligible_count,
        "raw_source_queue_manifest_only": True,
        "raw_source_queue_copies_payload": False,
        "raw_hard_delete_allowed_now": False,
        "gzip_compaction_allowed_now": True,
        "apply_requested": bool(args.apply),
        "bounded_apply_caps": {
            "max_files": int(args.max_files),
            "max_gb": float(args.max_gb),
            "jumbo_gb": float(args.jumbo_gb),
            "min_age_hours": float(args.min_age_hours),
            "compress_level": int(args.compress_level),
            "keep_raw_after_compress": bool(args.keep_raw_after_compress),
            "compaction_workers": compaction_workers,
        },
        "blocked_reasons": blockers,
        "managed_debts": (
            ["submaterial_raw_compaction_tail"]
            if bool(summary.get("submaterial_compaction_tail", False)) and not blockers
            else []
        ),
        "risk_flags": [
            "manifest_only_queue_does_not_copy_raw_payload",
            "current_day_sources_protected",
            "local_fallback_sources_require_reconciliation",
            "externally_archived_fallback_evidence_is_not_live_reconciliation_debt",
            "protected_volume_VIDEO_never_touched",
            "raw_evidence_preserved_as_gzip_before_raw_removal",
            "parallel_workers_only_touch_independent_raw_files",
        ],
        "next_actions": next_actions,
    }
    manifest = {
        "timestamp_utc": timestamp,
        "schema_version": 1,
        "overall_status": status,
        "overall_score": round(score, 3),
        "overall_grade": grade,
        "apply_requested": bool(args.apply),
        "scan_roots": roots_payload,
        "policy": {
            "copy_raw_payload": False,
            "manifest_only_training_queue": True,
            "hard_delete_raw_without_compressed_evidence": False,
            "protected_volumes": list(PROTECTED_VOLUME_PREFIXES),
            "current_day_protected": True,
            "active_latest_jsonl_protected": True,
            "local_fallback_requires_reconciliation": True,
            "clearance_method": "verified_gzip_compaction",
            "material_compaction_floor_gb": round(material_floor_gb, 6),
            "live_execution_unchanged": "paper_read_only",
        },
        "raw_summary": summary,
        "training_clearance": training_clearance,
        "decision_packet": decision_packet,
        "next_training_manifest": {
            "raw_source_queue_path": str(args.source_queue_path),
            "raw_source_queue_count": source_count,
            "raw_source_queue_manifest_only": True,
            "raw_source_queue_copies_payload": False,
            "raw_eligible_source_queue_path": str(args.eligible_queue_path),
            "raw_eligible_source_queue_count": eligible_count,
            "training_ingress": "runtime_training_snapshot",
            "labeling_ingress": "training_labeling_intelligence",
            "storage_ingress": "raw_training_compaction_intelligence",
        },
        "selected_compaction_batch": selected[: int(args.max_files)],
        "apply_records": apply_records,
        "top_training_sources": eligible_rows[:25],
        "top_compaction_candidates": [
            row for row in rows if row.get("compression_candidate")
        ][:25],
        "recommended_commands": {
            "dry_run": [
                "./scripts/ops/opsctl.sh",
                "raw-training-compaction",
                "--json",
            ],
            "bounded_apply": [
                "./scripts/ops/opsctl.sh",
                "raw-training-compaction",
                "--apply",
                "--max-files",
                str(args.max_files),
                "--max-gb",
                str(args.max_gb),
                "--compaction-workers",
                str(compaction_workers),
                "--json",
            ],
            "refresh_raw_backlog_refiner": [
                "./scripts/ops/opsctl.sh",
                "raw-backlog-refiner",
                "--apply",
                "--skip-drain",
                "--json",
            ],
        },
    }
    _write_json(Path(args.manifest_path), manifest)
    _write_json(Path(args.health_path), manifest)
    if args.write_history:
        history_dir = Path(args.history_dir)
        history_path = (
            history_dir
            / f"raw_training_compaction_{timestamp.replace(':', '').replace('+', 'Z')}.json"
        )
        _write_json(history_path, manifest)
        manifest["history_path"] = str(history_path)
        _write_json(Path(args.manifest_path), manifest)
        _write_json(Path(args.health_path), manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Queue raw JSONL data for training, then safely clear eligible raw files by verified gzip compaction."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply a bounded compaction wave after writing the training queues.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument(
        "--bot-logs-root", default="", help="Primary raw bot logs root to scan."
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        default=[],
        help="Additional or replacement scan root. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=12,
        help="Maximum raw files to compact in one apply wave.",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=8.0,
        help="Maximum raw source GB to compact in one apply wave.",
    )
    parser.add_argument(
        "--jumbo-gb",
        type=float,
        default=float(os.getenv("BOT_RAW_TRAINING_JUMBO_COMPACTION_GB", "12.0")),
        help="Allow one old eligible raw file up to this size when normal wave caps would otherwise pick tiny files.",
    )
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=24.0,
        help="Minimum source age before raw compaction is allowed.",
    )
    parser.add_argument(
        "--sample-bytes",
        type=int,
        default=4096,
        help="Bytes to hash from each raw source for lineage.",
    )
    parser.add_argument(
        "--compress-level",
        type=int,
        default=1,
        choices=range(1, 10),
        help="gzip compression level for compaction.",
    )
    parser.add_argument(
        "--compaction-workers",
        type=int,
        default=int(os.getenv("BOT_RAW_TRAINING_COMPACTION_WORKERS", "1")),
        help="Bounded worker count for independent raw-file gzip compaction.",
    )
    parser.add_argument(
        "--keep-raw-after-compress",
        action="store_true",
        help="Write gzip files but do not remove raw files.",
    )
    parser.add_argument("--health-path", default=str(DEFAULT_HEALTH_PATH))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--source-queue-path", default=str(DEFAULT_SOURCE_QUEUE_PATH))
    parser.add_argument(
        "--eligible-queue-path", default=str(DEFAULT_ELIGIBLE_QUEUE_PATH)
    )
    parser.add_argument("--write-history", action="store_true")
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = build_report(args)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "raw_training_compaction "
            f"status={payload.get('overall_status')} "
            f"grade={payload.get('overall_grade')} "
            f"raw={payload.get('raw_summary', {}).get('raw_jsonl_count')} "
            f"eligible={payload.get('raw_summary', {}).get('eligible_training_source_count')} "
            f"cleared_gb={payload.get('raw_summary', {}).get('raw_gb_cleared')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
