#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, write_payload

from scripts.ops import cold_evidence_compactor as verified

DEFAULT_OUT_PATH = (
    PROJECT_ROOT
    / "governance"
    / "health"
    / "governance_lifecycle_compactor_latest.json"
)
DEFAULT_LOCK_PATH = (
    PROJECT_ROOT / "governance" / "locks" / "governance_lifecycle_compactor.lock"
)
BACKUP_NAME = re.compile(r"master_bot_registry\..*backup.*\.json$")


def _gb(raw_bytes: int) -> float:
    return round(float(raw_bytes) / float(1024**3), 3)


def _today_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _relative(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _file_day(path: Path) -> str:
    name = path.name
    for token in name.replace("-", "").replace("T", "_").replace(".", "_").split("_"):
        if len(token) == 8 and token.startswith("20") and token.isdigit():
            return token
    return ""


def _path_age_hours(path: Path) -> float:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return 0.0
    return max((datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0, 0.0)


def _iter_lifecycle_json(project_root: Path) -> list[Path]:
    root = verified.allowed(project_root / "governance" / "lifecycle", missing=True)
    if not root.exists():
        return []
    return sorted(
        (
            verified.allowed(path)
            for path in root.glob("*.json")
            if not path.is_symlink()
            and BACKUP_NAME.fullmatch(path.name)
            and path.is_file()
        ),
        key=lambda path: str(path),
    )


def _candidate_rows(
    *,
    project_root: Path,
    min_file_bytes: int,
    include_current_day: bool,
    min_age_hours: float,
    keep_latest: int,
) -> list[dict[str, Any]]:
    today = _today_stamp()
    files = _iter_lifecycle_json(project_root)
    newest_keep = set(
        sorted(
            files,
            key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
            reverse=True,
        )[: max(int(keep_latest), 0)]
    )
    rows: list[dict[str, Any]] = []
    for path in files:
        if path in newest_keep:
            continue
        try:
            source_identity = verified.identity(path)
            size_bytes = source_identity[2]
        except OSError:
            continue
        if size_bytes < min_file_bytes:
            continue
        age_hours = (
            datetime.now(timezone.utc).timestamp() - source_identity[3] / 1e9
        ) / 3600
        if age_hours < float(min_age_hours):
            continue
        day = _file_day(path)
        is_current_day = bool(day and day >= today)
        if is_current_day and not include_current_day:
            continue
        rows.append(
            {
                "relative_path": _relative(project_root, path),
                "size_bytes": size_bytes,
                "size_gb": _gb(size_bytes),
                "day": day,
                "current_day": is_current_day,
                "age_hours": round(age_hours, 3),
                "source_identity": source_identity,
                "action": "gzip_compact_lifecycle_backup_in_place",
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row.get("size_bytes", 0) or 0),
            str(row.get("relative_path") or ""),
        )
    )
    return rows


def _select_rows(
    rows: list[dict[str, Any]], *, target_free_bytes: int, max_files: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_bytes = 0
    for row in rows:
        if max_files > 0 and len(selected) >= max_files:
            break
        selected.append(dict(row))
        selected_bytes += int(row.get("size_bytes", 0) or 0)
        if target_free_bytes > 0 and selected_bytes >= target_free_bytes:
            break
    return selected


def _compact_one(
    *, project_root: Path, source_rel: str, compression_level: int, guard, expected
) -> dict[str, Any]:
    source_path = project_root / source_rel
    try:
        if (
            source_path.parent != project_root / "governance/lifecycle"
            or not BACKUP_NAME.fullmatch(source_path.name)
        ):
            raise ValueError("not_a_lifecycle_registry_backup")
        proof = verified.compact_one(
            project_root,
            source_path,
            tuple(expected),
            guard,
            codec="gzip",
            compression_level=compression_level,
        )
        raw_bytes, archive_bytes = proof["source_bytes"], proof["compressed_bytes"]
        return {
            "relative_path": source_rel,
            "status": "compacted",
            "archive_replaced": False,
            "raw_bytes": raw_bytes,
            "raw_gb": _gb(raw_bytes),
            "archive_path": proof["compressed"],
            "archive_bytes": archive_bytes,
            "archive_gb": _gb(archive_bytes),
            "estimated_reduction_bytes": max(raw_bytes - archive_bytes, 0),
            "estimated_reduction_gb": _gb(max(raw_bytes - archive_bytes, 0)),
            "restore_proof": proof,
        }
    except verified.Deferred as exc:
        return {"relative_path": source_rel, "status": "deferred", "reason": str(exc)}
    except Exception as exc:
        return {
            "relative_path": source_rel,
            "status": "error",
            "error": str(exc),
        }


def build_payload(
    *,
    project_root: Path = PROJECT_ROOT,
    apply: bool = False,
    min_file_mb: float = 5.0,
    target_free_gb: float = 2.0,
    max_files: int = 120,
    include_current_day: bool = False,
    min_age_hours: float = 24.0,
    keep_latest: int = 12,
    compression_level: int = 1,
    seconds: int = 720,
) -> dict[str, Any]:
    project_root = verified.allowed(Path(project_root).absolute())
    min_file_bytes = max(int(float(min_file_mb) * 1024 * 1024), 1)
    target_free_bytes = max(int(float(target_free_gb) * 1024 * 1024 * 1024), 0)
    candidates = _candidate_rows(
        project_root=project_root,
        min_file_bytes=min_file_bytes,
        include_current_day=bool(include_current_day),
        min_age_hours=float(min_age_hours),
        keep_latest=int(keep_latest),
    )
    selected = _select_rows(
        candidates,
        target_free_bytes=target_free_bytes,
        max_files=max(int(max_files), 0),
    )
    if apply and selected:
        lock = verified.allowed(
            project_root / "governance/locks/storage_maintenance.lock", missing=True
        )
        lock.parent.mkdir(parents=True, exist_ok=True)
        records = []
        with os.fdopen(
            os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), "a+"
        ) as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                records.append(
                    {"status": "deferred", "reason": "storage_maintenance_lock_busy"}
                )
            else:
                verified.background_policy()
                guard = verified.Guard(project_root, min(max(int(seconds), 1), 840))
                held = os.fstat(handle.fileno())
                guard.lock_anchor = (lock, (held.st_dev, held.st_ino))
                for row in selected:
                    record = _compact_one(
                        project_root=project_root,
                        source_rel=row["relative_path"],
                        compression_level=int(compression_level),
                        guard=guard,
                        expected=row["source_identity"],
                    )
                    records.append(record)
                    if record["status"] != "compacted":
                        break
    else:
        records = [dict(row, status="planned") for row in selected]

    compacted = [row for row in records if str(row.get("status") or "") == "compacted"]
    errors = [row for row in records if str(row.get("status") or "") == "error"]
    deferred = [row for row in records if row.get("status") == "deferred"]
    selected_bytes = sum(
        int(row.get("size_bytes", row.get("raw_bytes", 0)) or 0) for row in selected
    )
    raw_compacted_bytes = sum(int(row.get("raw_bytes", 0) or 0) for row in compacted)
    archive_bytes = sum(int(row.get("archive_bytes", 0) or 0) for row in compacted)
    reduction_bytes = sum(
        int(row.get("estimated_reduction_bytes", 0) or 0) for row in compacted
    )
    if errors:
        overall_status = "degraded"
    elif deferred:
        overall_status = "deferred"
    elif apply and compacted:
        overall_status = "applied"
    elif selected:
        overall_status = "planned"
    else:
        overall_status = "nothing_to_do"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": not errors,
        "overall_status": overall_status,
        "apply": bool(apply),
        "assessment_complete": True,
        "batch_complete": bool(apply and len(compacted) == len(selected)),
        "policy": {
            "min_file_mb": float(min_file_mb),
            "target_free_gb": float(target_free_gb),
            "max_files": int(max_files),
            "include_current_day": bool(include_current_day),
            "min_age_hours": float(min_age_hours),
            "keep_latest": int(keep_latest),
            "compression_level": int(compression_level),
            "work_deadline_seconds": min(max(int(seconds), 1), 840),
            "full_restore_sha256_required": True,
            "shared_storage_lock_required": True,
            "disk_recovery_cpu_policy": dict(verified.RECOVERY_CPU_POLICY),
            "existing_archive_overwrite_allowed": False,
            "compaction_policy": "gzip_old_governance_lifecycle_backups_keep_latest_and_current_day_hot",
        },
        "summary": {
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "selected_bytes": int(selected_bytes),
            "selected_gb": _gb(selected_bytes),
            "compacted_count": len(compacted),
            "raw_compacted_bytes": int(raw_compacted_bytes),
            "raw_compacted_gb": _gb(raw_compacted_bytes),
            "archive_bytes": int(archive_bytes),
            "archive_gb": _gb(archive_bytes),
            "estimated_reduction_bytes": int(reduction_bytes),
            "estimated_reduction_gb": _gb(reduction_bytes),
            "error_count": len(errors),
            "deferred_count": len(deferred),
        },
        "records": records,
        "next_action": (
            "monitor governance lifecycle backup growth"
            if not selected
            else "refresh governance directory usage after compaction"
        ),
    }


def _acquire_lock(path: Path) -> tuple[Any | None, str]:
    verified.allowed(path, missing=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = os.fdopen(
        os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600),
        "a+",
        encoding="utf-8",
    )
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()} started={iso_now()}\n")
        fh.flush()
        return fh, ""
    except BlockingIOError:
        try:
            fh.seek(0)
            owner = fh.read().strip()
        except Exception:
            owner = ""
        try:
            fh.close()
        except Exception:
            pass
        return None, owner


def main() -> int:
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    parser = argparse.ArgumentParser(
        description="Gzip-compact old governance lifecycle registry backups while keeping recent backups readable."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-path", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument(
        "--min-file-mb",
        type=float,
        default=float(os.getenv("GOVERNANCE_LIFECYCLE_COMPACTOR_MIN_FILE_MB", "5")),
    )
    parser.add_argument(
        "--target-free-gb",
        type=float,
        default=float(os.getenv("GOVERNANCE_LIFECYCLE_COMPACTOR_TARGET_FREE_GB", "2")),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=int(os.getenv("GOVERNANCE_LIFECYCLE_COMPACTOR_MAX_FILES", "120")),
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=int(os.getenv("GOVERNANCE_LIFECYCLE_COMPACTOR_GZIP_LEVEL", "1")),
    )
    parser.add_argument(
        "--include-current-day",
        action=argparse.BooleanOptionalAction,
        default=os.getenv(
            "GOVERNANCE_LIFECYCLE_COMPACTOR_INCLUDE_CURRENT_DAY", "0"
        ).strip()
        == "1",
    )
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=float(os.getenv("GOVERNANCE_LIFECYCLE_COMPACTOR_MIN_AGE_HOURS", "24")),
    )
    parser.add_argument(
        "--keep-latest",
        type=int,
        default=int(os.getenv("GOVERNANCE_LIFECYCLE_COMPACTOR_KEEP_LATEST", "12")),
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--seconds", type=int, default=720)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    verified.allowed(Path(args.project_root).expanduser().absolute())
    out_path = verified.allowed(Path(args.out_file).expanduser(), missing=True)
    lock_fh = None
    if args.apply:
        lock_fh, owner = _acquire_lock(Path(args.lock_path).expanduser())
        if lock_fh is None:
            payload = {
                "timestamp_utc": iso_now(),
                "schema_version": 1,
                "ok": False,
                "overall_status": "busy",
                "apply": True,
                "lock_owner": owner,
            }
            write_payload(out_path, payload)
            print(json.dumps(payload, ensure_ascii=True))
            return 2
    try:
        payload = build_payload(
            project_root=Path(args.project_root).expanduser().absolute(),
            apply=bool(args.apply),
            min_file_mb=float(args.min_file_mb),
            target_free_gb=float(args.target_free_gb),
            max_files=int(args.max_files),
            include_current_day=bool(args.include_current_day),
            min_age_hours=float(args.min_age_hours),
            keep_latest=int(args.keep_latest),
            compression_level=int(args.compression_level),
            seconds=int(args.seconds),
        )
        write_payload(out_path, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            summary = (
                payload.get("summary")
                if isinstance(payload.get("summary"), dict)
                else {}
            )
            print(
                "governance_lifecycle_compactor "
                f"overall_status={payload.get('overall_status', '')} "
                f"selected_gb={summary.get('selected_gb', 0)} "
                f"compacted_gb={summary.get('raw_compacted_gb', 0)}"
            )
        return 0 if payload.get("ok", False) else 2
    finally:
        if lock_fh is not None:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                lock_fh.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
