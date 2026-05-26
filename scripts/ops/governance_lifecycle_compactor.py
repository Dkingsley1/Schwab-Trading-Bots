#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import os
import shutil
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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "governance_lifecycle_compactor_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "governance_lifecycle_compactor.lock"


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
    root = project_root / "governance" / "lifecycle"
    if not root.exists():
        return []
    return sorted((path for path in root.glob("*.json") if path.is_file()), key=lambda path: str(path))


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
    newest_keep = set(sorted(files, key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)[: max(int(keep_latest), 0)])
    rows: list[dict[str, Any]] = []
    for path in files:
        if path in newest_keep:
            continue
        try:
            size_bytes = int(path.stat().st_size)
        except OSError:
            continue
        if size_bytes < min_file_bytes:
            continue
        age_hours = _path_age_hours(path)
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
                "action": "gzip_compact_lifecycle_backup_in_place",
            }
        )
    rows.sort(key=lambda row: (-int(row.get("size_bytes", 0) or 0), str(row.get("relative_path") or "")))
    return rows


def _select_rows(rows: list[dict[str, Any]], *, target_free_bytes: int, max_files: int) -> list[dict[str, Any]]:
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


def _compact_one(*, project_root: Path, source_rel: str, compression_level: int) -> dict[str, Any]:
    source_path = project_root / source_rel
    if not source_path.exists():
        return {"relative_path": source_rel, "status": "missing", "error": "source_missing"}
    archive_path = source_path.with_name(f"{source_path.name}.gz")
    archive_preexisting = archive_path.exists()
    try:
        raw_bytes = int(source_path.stat().st_size)
    except OSError as exc:
        return {"relative_path": source_rel, "status": "error", "error": str(exc)}
    tmp_archive = archive_path.with_name(f"{archive_path.name}.tmp.{os.getpid()}")
    try:
        with source_path.open("rb") as src, gzip.open(tmp_archive, "wb", compresslevel=max(min(int(compression_level), 9), 1)) as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        tmp_archive.replace(archive_path)
        archive_bytes = int(archive_path.stat().st_size)
        source_path.unlink()
        return {
            "relative_path": source_rel,
            "status": "compacted",
            "archive_replaced": bool(archive_preexisting),
            "raw_bytes": raw_bytes,
            "raw_gb": _gb(raw_bytes),
            "archive_path": _relative(project_root, archive_path),
            "archive_bytes": archive_bytes,
            "archive_gb": _gb(archive_bytes),
            "estimated_reduction_bytes": max(raw_bytes - archive_bytes, 0),
            "estimated_reduction_gb": _gb(max(raw_bytes - archive_bytes, 0)),
        }
    except Exception as exc:
        try:
            tmp_archive.unlink(missing_ok=True)
        except Exception:
            pass
        return {
            "relative_path": source_rel,
            "status": "error",
            "raw_bytes": raw_bytes,
            "raw_gb": _gb(raw_bytes),
            "archive_path": _relative(project_root, archive_path),
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
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    min_file_bytes = max(int(float(min_file_mb) * 1024 * 1024), 1)
    target_free_bytes = max(int(float(target_free_gb) * 1024 * 1024 * 1024), 0)
    candidates = _candidate_rows(
        project_root=project_root,
        min_file_bytes=min_file_bytes,
        include_current_day=bool(include_current_day),
        min_age_hours=float(min_age_hours),
        keep_latest=int(keep_latest),
    )
    selected = _select_rows(candidates, target_free_bytes=target_free_bytes, max_files=max(int(max_files), 0))
    if apply and selected:
        records = [
            _compact_one(project_root=project_root, source_rel=str(row.get("relative_path") or ""), compression_level=int(compression_level))
            for row in selected
        ]
    else:
        records = [dict(row, status="planned") for row in selected]

    compacted = [row for row in records if str(row.get("status") or "") == "compacted"]
    errors = [row for row in records if str(row.get("status") or "") == "error"]
    selected_bytes = sum(int(row.get("size_bytes", row.get("raw_bytes", 0)) or 0) for row in selected)
    raw_compacted_bytes = sum(int(row.get("raw_bytes", 0) or 0) for row in compacted)
    archive_bytes = sum(int(row.get("archive_bytes", 0) or 0) for row in compacted)
    reduction_bytes = sum(int(row.get("estimated_reduction_bytes", 0) or 0) for row in compacted)
    if errors:
        overall_status = "degraded"
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
        "policy": {
            "min_file_mb": float(min_file_mb),
            "target_free_gb": float(target_free_gb),
            "max_files": int(max_files),
            "include_current_day": bool(include_current_day),
            "min_age_hours": float(min_age_hours),
            "keep_latest": int(keep_latest),
            "compression_level": int(compression_level),
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
        },
        "records": records,
        "next_action": "monitor governance lifecycle backup growth" if not selected else "refresh governance directory usage after compaction",
    }


def _acquire_lock(path: Path) -> tuple[Any | None, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a+", encoding="utf-8")
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
    parser = argparse.ArgumentParser(description="Gzip-compact old governance lifecycle registry backups while keeping recent backups readable.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-path", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--min-file-mb", type=float, default=float(os.getenv("GOVERNANCE_LIFECYCLE_COMPACTOR_MIN_FILE_MB", "5")))
    parser.add_argument("--target-free-gb", type=float, default=float(os.getenv("GOVERNANCE_LIFECYCLE_COMPACTOR_TARGET_FREE_GB", "2")))
    parser.add_argument("--max-files", type=int, default=int(os.getenv("GOVERNANCE_LIFECYCLE_COMPACTOR_MAX_FILES", "120")))
    parser.add_argument("--compression-level", type=int, default=int(os.getenv("GOVERNANCE_LIFECYCLE_COMPACTOR_GZIP_LEVEL", "1")))
    parser.add_argument("--include-current-day", action=argparse.BooleanOptionalAction, default=os.getenv("GOVERNANCE_LIFECYCLE_COMPACTOR_INCLUDE_CURRENT_DAY", "0").strip() == "1")
    parser.add_argument("--min-age-hours", type=float, default=float(os.getenv("GOVERNANCE_LIFECYCLE_COMPACTOR_MIN_AGE_HOURS", "24")))
    parser.add_argument("--keep-latest", type=int, default=int(os.getenv("GOVERNANCE_LIFECYCLE_COMPACTOR_KEEP_LATEST", "12")))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out_file).expanduser()
    lock_fh = None
    if args.apply:
        lock_fh, owner = _acquire_lock(Path(args.lock_path).expanduser())
        if lock_fh is None:
            payload = {"timestamp_utc": iso_now(), "schema_version": 1, "ok": False, "overall_status": "busy", "apply": True, "lock_owner": owner}
            write_payload(out_path, payload)
            print(json.dumps(payload, ensure_ascii=True))
            return 2
    try:
        payload = build_payload(
            project_root=Path(args.project_root).resolve(),
            apply=bool(args.apply),
            min_file_mb=float(args.min_file_mb),
            target_free_gb=float(args.target_free_gb),
            max_files=int(args.max_files),
            include_current_day=bool(args.include_current_day),
            min_age_hours=float(args.min_age_hours),
            keep_latest=int(args.keep_latest),
            compression_level=int(args.compression_level),
        )
        write_payload(out_path, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
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
