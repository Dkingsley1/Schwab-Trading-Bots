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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "governance_telemetry_compactor_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "governance_telemetry_compactor.lock"
DEFAULT_ARCHIVE_ROOT = PROJECT_ROOT / "data" / "stale_stage" / "governance_telemetry_compactor"
DEFAULT_CHANNELS = ("decision",)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _gb(raw_bytes: int) -> float:
    return round(float(raw_bytes) / float(1024**3), 3)


def _today_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_csv(raw: str | None, default: tuple[str, ...]) -> list[str]:
    rows = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    return rows or list(default)


def _relative(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _file_day(path: Path) -> str:
    stem = path.name
    for token in stem.replace(".", "_").split("_"):
        if len(token) == 8 and token.startswith("20") and token.isdigit():
            return token
    return ""


def _iter_channel_files(project_root: Path, channels: list[str]) -> list[Path]:
    files: list[Path] = []
    base = project_root / "governance" / "channels"
    for channel in channels:
        root = base / channel
        if not root.exists():
            continue
        files.extend(path for path in root.rglob("*.jsonl") if path.is_file())
    return sorted(files, key=lambda path: (-_safe_int(path.stat().st_size if path.exists() else 0), str(path)))


def _candidate_rows(
    *,
    project_root: Path,
    channels: list[str],
    min_file_bytes: int,
    include_current_day: bool,
) -> list[dict[str, Any]]:
    today = _today_stamp()
    rows: list[dict[str, Any]] = []
    for path in _iter_channel_files(project_root, channels):
        try:
            size_bytes = int(path.stat().st_size)
        except OSError:
            continue
        if size_bytes < min_file_bytes:
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
                "action": "rotate_and_archive_oversized_governance_channel",
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


def _archive_one(
    *,
    project_root: Path,
    source_rel: str,
    archive_root: Path,
    compression_level: int,
    stamp: str,
) -> dict[str, Any]:
    source_path = project_root / source_rel
    if not source_path.exists():
        return {"relative_path": source_rel, "status": "missing", "error": "source_missing"}

    try:
        raw_bytes = int(source_path.stat().st_size)
    except OSError as exc:
        return {"relative_path": source_rel, "status": "error", "error": str(exc)}

    pending_path = source_path.with_name(f"{source_path.name}.compact_pending_{stamp}_{os.getpid()}")
    archive_path = archive_root / stamp / source_rel
    archive_path = archive_path.with_name(f"{archive_path.name}.gz")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_archive = archive_path.with_name(f"{archive_path.name}.tmp")

    try:
        source_path.rename(pending_path)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.touch(exist_ok=True)
        with pending_path.open("rb") as src, gzip.open(tmp_archive, "wb", compresslevel=max(min(int(compression_level), 9), 1)) as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        tmp_archive.replace(archive_path)
        archive_bytes = int(archive_path.stat().st_size)
        pending_path.unlink(missing_ok=True)
        return {
            "relative_path": source_rel,
            "status": "archived",
            "raw_bytes": raw_bytes,
            "raw_gb": _gb(raw_bytes),
            "archive_path": _relative(project_root, archive_path),
            "archive_bytes": archive_bytes,
            "archive_gb": _gb(archive_bytes),
            "estimated_hot_reduction_bytes": max(raw_bytes - archive_bytes, 0),
            "estimated_hot_reduction_gb": _gb(max(raw_bytes - archive_bytes, 0)),
        }
    except Exception as exc:
        try:
            if pending_path.exists() and not source_path.exists():
                pending_path.rename(source_path)
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
    finally:
        try:
            tmp_archive.unlink(missing_ok=True)
        except Exception:
            pass


def build_payload(
    *,
    project_root: Path = PROJECT_ROOT,
    apply: bool = False,
    channels: list[str] | None = None,
    min_file_mb: float = 256.0,
    target_free_gb: float = 9.0,
    max_files: int = 8,
    include_current_day: bool = True,
    archive_root: Path | None = None,
    compression_level: int = 1,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    channel_list = [part for part in (channels or list(DEFAULT_CHANNELS)) if part]
    min_file_bytes = max(int(float(min_file_mb) * 1024 * 1024), 1)
    target_free_bytes = max(int(float(target_free_gb) * 1024 * 1024 * 1024), 0)
    archive_base = Path(archive_root or (project_root / "data" / "stale_stage" / "governance_telemetry_compactor")).expanduser()

    candidates = _candidate_rows(
        project_root=project_root,
        channels=channel_list,
        min_file_bytes=min_file_bytes,
        include_current_day=bool(include_current_day),
    )
    selected = _select_rows(candidates, target_free_bytes=target_free_bytes, max_files=max(int(max_files), 0))

    records: list[dict[str, Any]] = []
    if apply and selected:
        stamp = _timestamp_slug()
        for row in selected:
            records.append(
                _archive_one(
                    project_root=project_root,
                    source_rel=str(row.get("relative_path") or ""),
                    archive_root=archive_base,
                    compression_level=int(compression_level),
                    stamp=stamp,
                )
            )
    else:
        records = [dict(row, status="planned") for row in selected]

    archived_records = [row for row in records if str(row.get("status") or "") == "archived"]
    error_records = [row for row in records if str(row.get("status") or "") == "error"]
    selected_bytes = sum(int(row.get("size_bytes", row.get("raw_bytes", 0)) or 0) for row in selected)
    raw_archived_bytes = sum(int(row.get("raw_bytes", 0) or 0) for row in archived_records)
    archive_bytes = sum(int(row.get("archive_bytes", 0) or 0) for row in archived_records)
    estimated_reduction_bytes = sum(int(row.get("estimated_hot_reduction_bytes", 0) or 0) for row in archived_records)

    if error_records:
        overall_status = "degraded"
    elif apply and archived_records:
        overall_status = "applied"
    elif selected:
        overall_status = "planned"
    else:
        overall_status = "nothing_to_do"

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": not error_records,
        "overall_status": overall_status,
        "apply": bool(apply),
        "policy": {
            "channels": channel_list,
            "min_file_mb": float(min_file_mb),
            "target_free_gb": float(target_free_gb),
            "max_files": int(max_files),
            "include_current_day": bool(include_current_day),
            "archive_root": _relative(project_root, archive_base),
            "compression_level": int(compression_level),
            "rotation_policy": "rename_active_file_then_touch_fresh_path",
        },
        "summary": {
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "selected_bytes": int(selected_bytes),
            "selected_gb": _gb(selected_bytes),
            "archived_count": len(archived_records),
            "raw_archived_bytes": int(raw_archived_bytes),
            "raw_archived_gb": _gb(raw_archived_bytes),
            "archive_bytes": int(archive_bytes),
            "archive_gb": _gb(archive_bytes),
            "estimated_hot_reduction_bytes": int(estimated_reduction_bytes),
            "estimated_hot_reduction_gb": _gb(estimated_reduction_bytes),
            "error_count": len(error_records),
        },
        "records": records,
        "next_action": (
            "refresh storage-tier-policy and storage-quota-guard, then recheck training-runtime-control"
            if apply and archived_records
            else "run with --apply to rotate oversized governance telemetry files"
            if selected and not apply
            else "monitor governance telemetry quota"
        ),
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
    parser = argparse.ArgumentParser(description="Rotate and compress oversized governance channel telemetry out of the hot quota lane.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-path", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    parser.add_argument("--channels", default=",".join(DEFAULT_CHANNELS))
    parser.add_argument("--min-file-mb", type=float, default=float(os.getenv("GOVERNANCE_TELEMETRY_COMPACTOR_MIN_FILE_MB", "256")))
    parser.add_argument("--target-free-gb", type=float, default=float(os.getenv("GOVERNANCE_TELEMETRY_COMPACTOR_TARGET_FREE_GB", "9")))
    parser.add_argument("--max-files", type=int, default=int(os.getenv("GOVERNANCE_TELEMETRY_COMPACTOR_MAX_FILES", "8")))
    parser.add_argument("--compression-level", type=int, default=int(os.getenv("GOVERNANCE_TELEMETRY_COMPACTOR_GZIP_LEVEL", "1")))
    parser.add_argument("--include-current-day", action=argparse.BooleanOptionalAction, default=os.getenv("GOVERNANCE_TELEMETRY_COMPACTOR_INCLUDE_CURRENT_DAY", "1").strip() == "1")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_path = Path(args.out_file).expanduser()
    lock_path = Path(args.lock_path).expanduser()

    lock_fh = None
    if args.apply:
        lock_fh, owner = _acquire_lock(lock_path)
        if lock_fh is None:
            payload = {
                "timestamp_utc": iso_now(),
                "schema_version": 1,
                "ok": False,
                "overall_status": "busy",
                "apply": True,
                "lock_path": str(lock_path),
                "lock_owner": owner,
            }
            write_payload(out_path, payload)
            print(json.dumps(payload, ensure_ascii=True))
            return 2

    try:
        payload = build_payload(
            project_root=project_root,
            apply=bool(args.apply),
            channels=_parse_csv(args.channels, DEFAULT_CHANNELS),
            min_file_mb=float(args.min_file_mb),
            target_free_gb=float(args.target_free_gb),
            max_files=int(args.max_files),
            include_current_day=bool(args.include_current_day),
            archive_root=Path(args.archive_root).expanduser(),
            compression_level=int(args.compression_level),
        )
        write_payload(out_path, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
            print(
                "governance_telemetry_compactor "
                f"overall_status={payload.get('overall_status', '')} "
                f"selected_gb={summary.get('selected_gb', 0)} "
                f"archived_gb={summary.get('raw_archived_gb', 0)}"
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
