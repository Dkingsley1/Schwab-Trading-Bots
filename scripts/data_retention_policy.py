import argparse
import fcntl
import json
import math
import os
import re
import signal
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.training_guard import check_confirmed_training_success
from sql_hot_retention import _prune_archive_storage
from snapshot_health_sql import debug_snapshot_ingest_coverage, sync_raw_debug_snapshots_to_sqlite

TIMELINE_STAMP_RE = re.compile(r"^project_timeline(?:_print)?_(\d{8}_\d{6})\.(?:md|pdf|html)$")
CRASH_REPORT_STAMP_RE = re.compile(r"^crash_report_digest(?:_print)?_(\d{8}_\d{6})\.(?:md|pdf|html)$")
TRAINING_REPORT_STAMP_RE = re.compile(r"^training_report(?:_print)?_(\d{8}_\d{6})\.(?:md|pdf|html)$")
DAILY_OPS_REPORT_RE = re.compile(r"^daily_ops_report_(\d{8})\.(?:md|json|pdf)$")
ONE_NUMBERS_STAMP_RE = re.compile(r"^one_numbers_\d{8}_(\d{8}_\d{6})\.(?:md|csv|pdf)$")

DEFAULT_EXTERNAL_MOUNT = "/Volumes/BOT_LOGS"
DEFAULT_EXTERNAL_PROJECT = "schwab_trading_bot"
LOCAL_FALLBACK_STORAGE_MODES = {"local_fallback", "local_fallback_split_brain"}
DEFAULT_STALE_STAGE_DIRNAME = "stale_stage"
DEFAULT_STALE_STAGE_SECTION_TOKENS = ("logs", "governance", "exports")


def _acquire_singleton_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            fh.seek(0)
            owner = fh.read().strip()
        except Exception:
            owner = "unknown"
        fh.close()
        return None, owner or "unknown"

    fh.seek(0)
    fh.truncate(0)
    fh.write(f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()} cmd={' '.join(sys.argv)}")
    fh.flush()
    return fh, ""


def _collect_old_files(base: Path, older_than_days: int) -> list[Path]:
    if not base.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(int(older_than_days), 0))
    out: list[Path] = []
    for root, _, files in os.walk(base):
        for name in files:
            p = Path(root) / name
            try:
                mt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
            if mt < cutoff:
                out.append(p)
    return out


def _path_size_bytes(path: Path) -> int:
    try:
        if path.is_file():
            return int(path.stat().st_size)
    except OSError:
        return 0

    total = 0
    if not path.exists():
        return total
    for root, _, files in os.walk(path):
        for name in files:
            p = Path(root) / name
            try:
                total += int(p.stat().st_size)
            except OSError:
                continue
    return total


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def _default_stale_stage_root(project_root: Path) -> Path:
    return project_root / "data" / DEFAULT_STALE_STAGE_DIRNAME


def _parse_csv_tokens(raw: str, *, default: tuple[str, ...] = ()) -> set[str]:
    text = str(raw or "").strip()
    if not text:
        return {str(item).strip().lower() for item in default if str(item).strip()}
    return {
        str(item).strip().lower()
        for item in text.split(",")
        if str(item).strip()
    }


def _label_matches_stale_stage(label: str, sections: set[str]) -> bool:
    raw = str(label or "").strip()
    token = raw.lower()
    if not token or not sections:
        return False
    if token in sections:
        return True
    if token == "logs" and "logs" in sections:
        return True
    if token.startswith("governance_") and "governance" in sections:
        return True
    if (token.startswith("exports_") or token == "backup_drills") and "exports" in sections:
        return True
    if token.startswith("debug_snapshot") and "debug_snapshots" in sections:
        return True
    return False


def _stale_manifest_path(stale_root: Path, manifest_override: str) -> Path:
    text = str(manifest_override or "").strip()
    if text:
        return Path(text).expanduser()
    return stale_root / "stale_manifest.jsonl"


def _stale_relative_path(path: Path, *, project_root: Path, external_root: Path) -> Path:
    roots: list[tuple[str, Path]] = [
        ("local_fallback", project_root / "local_fallback_storage"),
        ("external", external_root),
        ("project", project_root),
    ]
    roots.sort(key=lambda item: len(str(item[1])), reverse=True)
    for prefix, base in roots:
        try:
            rel = path.relative_to(base)
            return Path(prefix) / rel
        except ValueError:
            continue
    return Path("misc") / path.name


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    seq = 1
    while True:
        candidate = parent / f"{stem}.dup{seq}{suffix}"
        if not candidate.exists():
            return candidate
        seq += 1


def _move_paths_to_stale_stage(
    *,
    paths: list[Path],
    label: str,
    project_root: Path,
    external_root: Path,
    stale_root: Path,
    manifest_path: Path,
) -> dict[str, object]:
    moved = 0
    moved_bytes = 0
    moved_paths: list[str] = []
    errors: list[str] = []
    for path in _dedupe_paths(paths):
        if not path.exists():
            continue
        size_bytes = _path_size_bytes(path)
        rel = _stale_relative_path(path, project_root=project_root, external_root=external_root)
        dest = _unique_destination(stale_root / str(label or "unknown") / rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(path), str(dest))
        except OSError as exc:
            errors.append(f"{path}:{exc}")
            continue
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": "staged",
            "label": str(label),
            "original_path": str(path),
            "staged_path": str(dest),
            "size_bytes": int(size_bytes),
        }
        _append_jsonl(manifest_path, row)
        moved += 1
        moved_bytes += int(size_bytes)
        moved_paths.append(str(dest))
    return {
        "moved": int(moved),
        "moved_bytes": int(moved_bytes),
        "moved_paths": moved_paths,
        "errors": errors,
    }


def _purge_old_stale_stage(
    *,
    stale_root: Path,
    manifest_path: Path,
    older_than_days: int,
) -> dict[str, object]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(int(older_than_days), 0))
    rows: list[Path] = []
    if stale_root.exists():
        for root, dirs, files in os.walk(stale_root):
            dirs.sort()
            for name in files:
                path = Path(root) / name
                if path == manifest_path:
                    continue
                try:
                    mt = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                except OSError:
                    continue
                if mt < cutoff:
                    rows.append(path)
    bytes_total = sum(_path_size_bytes(path) for path in rows)
    deleted, errors = _delete_paths(rows)
    if stale_root.exists():
        for root, dirs, files in os.walk(stale_root, topdown=False):
            if dirs or files:
                continue
            path = Path(root)
            if path == stale_root:
                continue
            try:
                path.rmdir()
            except OSError:
                continue
    return {
        "candidate_files": int(len(rows)),
        "candidate_bytes": int(bytes_total),
        "deleted_files": int(deleted),
        "delete_errors": int(errors),
        "older_than_days": int(older_than_days),
    }


def _resolve_external_project_root() -> Path:
    configured = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    mount_root = Path(os.getenv("BOT_LOGS_EXTERNAL_MOUNT", DEFAULT_EXTERNAL_MOUNT)).expanduser()
    project_dir = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", DEFAULT_EXTERNAL_PROJECT).strip() or DEFAULT_EXTERNAL_PROJECT
    return mount_root / project_dir


def _external_mount_root() -> Path:
    return Path(os.getenv("BOT_LOGS_EXTERNAL_MOUNT", DEFAULT_EXTERNAL_MOUNT)).expanduser()


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve(strict=False) == b.resolve(strict=False)
    except Exception:
        return str(a) == str(b)


def _external_min_free_bytes() -> int:
    raw_bytes = os.getenv("BOT_LOGS_EXTERNAL_MIN_FREE_BYTES", "").strip()
    if raw_bytes:
        try:
            return max(int(float(raw_bytes)), 0)
        except Exception:
            return 0

    raw_gb = os.getenv("BOT_LOGS_EXTERNAL_MIN_FREE_GB", "").strip()
    if raw_gb:
        try:
            return max(int(float(raw_gb) * (1024 ** 3)), 0)
        except Exception:
            return 0

    return 0


def _external_low_space_autoprune_min_free_bytes() -> int:
    raw_bytes = os.getenv("BOT_LOGS_LOW_SPACE_AUTOPRUNE_MIN_FREE_BYTES", "").strip()
    if raw_bytes:
        try:
            return max(int(float(raw_bytes)), 0)
        except Exception:
            return 0

    raw_gb = os.getenv("BOT_LOGS_LOW_SPACE_AUTOPRUNE_MIN_FREE_GB", "").strip()
    if raw_gb:
        try:
            return max(int(float(raw_gb) * (1024 ** 3)), 0)
        except Exception:
            return 0

    return _external_min_free_bytes()


def _disk_free_bytes(path: Path) -> int | None:
    try:
        return int(shutil.disk_usage(path).free)
    except Exception:
        return None


def _probe_external_storage_pressure(external_root: Path) -> dict[str, object]:
    mount_root = _external_mount_root()
    mount_present = bool(mount_root.exists() and mount_root.is_dir())
    external_root_exists = bool(external_root.exists() and external_root.is_dir())
    external_root_writable = bool(external_root_exists and os.access(external_root, os.W_OK))
    probe_root = external_root if external_root_exists else mount_root
    external_free_bytes = _disk_free_bytes(probe_root) if mount_present else None
    external_min_free_bytes = _external_low_space_autoprune_min_free_bytes()
    external_low_space = bool(
        external_root_exists
        and external_root_writable
        and external_min_free_bytes > 0
        and external_free_bytes is not None
        and external_free_bytes < external_min_free_bytes
    )
    return {
        "mount_root": str(mount_root),
        "external_root": str(external_root),
        "mount_present": mount_present,
        "external_root_exists": external_root_exists,
        "external_root_writable": external_root_writable,
        "external_free_bytes": external_free_bytes,
        "external_min_free_bytes": int(external_min_free_bytes),
        "external_low_space": external_low_space,
    }


def _path_is_older_than(path: Path, older_than_days: int) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(int(older_than_days), 0))
    try:
        mt = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return False
    return mt < cutoff


def _current_storage_mode(project_root: Path) -> str:
    for candidate in (
        project_root / "governance",
        project_root / "exports",
        project_root / "data",
    ):
        try:
            resolved = candidate.resolve(strict=False)
        except Exception:
            resolved = candidate
        if "local_fallback_storage" in str(resolved):
            return "local_fallback"

    path = project_root / "governance" / "health" / "process_watchdog_latest.json"
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    storage_mode = str(payload.get("storage_mode") or "").strip()
    if storage_mode:
        return storage_mode
    nested = (
        payload.get("storage_failback_sync", {})
        .get("payload", {})
        .get("mode")
    )
    return str(nested or "").strip()


def _collect_external_live_sqlite_rows(
    project_root: Path,
    external_root: Path,
    *,
    older_than_days: int,
    require_local_fallback: bool,
) -> tuple[list[Path], dict[str, object]]:
    storage_mode = _current_storage_mode(project_root)
    details: dict[str, object] = {
        "external_root": str(external_root),
        "storage_mode": storage_mode,
        "older_than_days": int(older_than_days),
        "require_local_fallback": bool(require_local_fallback),
    }
    if require_local_fallback and storage_mode not in LOCAL_FALLBACK_STORAGE_MODES:
        details["skipped_reason"] = "storage_mode_not_local_fallback"
        return [], details

    external_data_root = external_root / "data"
    local_data_root = project_root / "local_fallback_storage" / "data"
    details["local_data_root"] = str(local_data_root)

    candidates: list[Path] = []
    local_mirror_required = {
        "jsonl_link.sqlite3": local_data_root / "jsonl_link.sqlite3",
        "jsonl_link.sqlite3-wal": local_data_root / "jsonl_link.sqlite3-wal",
        "jsonl_link.sqlite3-shm": local_data_root / "jsonl_link.sqlite3-shm",
        "bot_channel_queue.sqlite3": local_data_root / "bot_channel_queue.sqlite3",
    }
    for name, local_path in local_mirror_required.items():
        external_path = external_data_root / name
        if not external_path.exists():
            continue
        if not local_path.exists():
            continue
        if _path_is_older_than(external_path, older_than_days):
            candidates.append(external_path)

    external_shard_root = external_data_root / "sql_link_shards"
    local_shard_root = local_data_root / "sql_link_shards"
    shard_rows: list[Path] = []
    if external_shard_root.exists() and local_shard_root.exists():
        shard_rows = _collect_old_files(external_shard_root, older_than_days)
        candidates.extend(shard_rows)
    details["shard_file_candidates"] = int(len(shard_rows))
    return candidates, details


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _delete_paths(paths: list[Path]) -> tuple[int, int]:
    deleted = 0
    errors = 0
    for path in sorted(_dedupe_paths(paths), key=lambda item: len(str(item)), reverse=True):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            deleted += 1
        except OSError:
            errors += 1
    return deleted, errors


def _collect_external_live_sqlite_pressure_rows(
    project_root: Path,
    external_root: Path,
    *,
    require_local_fallback: bool,
) -> tuple[list[Path], dict[str, object]]:
    storage_mode = _current_storage_mode(project_root)
    pressure = _probe_external_storage_pressure(external_root)
    details: dict[str, object] = {
        **pressure,
        "storage_mode": storage_mode,
        "require_local_fallback": bool(require_local_fallback),
    }
    if not bool(pressure.get("external_low_space", False)):
        details["skipped_reason"] = "external_not_low_space"
        return [], details
    if require_local_fallback and storage_mode not in LOCAL_FALLBACK_STORAGE_MODES:
        details["skipped_reason"] = "storage_mode_not_local_fallback"
        return [], details

    external_data_root = external_root / "data"
    local_data_root = project_root / "local_fallback_storage" / "data"
    details["local_data_root"] = str(local_data_root)

    candidates: list[Path] = []
    mirrored_top_level = {
        "jsonl_link.sqlite3": local_data_root / "jsonl_link.sqlite3",
        "jsonl_link.sqlite3-wal": local_data_root / "jsonl_link.sqlite3-wal",
        "jsonl_link.sqlite3-shm": local_data_root / "jsonl_link.sqlite3-shm",
        "bot_channel_queue.sqlite3": local_data_root / "bot_channel_queue.sqlite3",
    }
    mirrored_count = 0
    for name, local_path in mirrored_top_level.items():
        external_path = external_data_root / name
        if not external_path.exists():
            continue
        if not local_path.exists():
            continue
        candidates.append(external_path)
        mirrored_count += 1

    local_fallback_copies = [
        path for path in external_data_root.glob("*.local_fallback*")
        if path.is_file()
    ]
    candidates.extend(local_fallback_copies)

    shard_rows: list[Path] = []
    shard_local_fallback_rows: list[Path] = []
    external_shard_root = external_data_root / "sql_link_shards"
    local_shard_root = local_data_root / "sql_link_shards"
    if external_shard_root.exists() and local_shard_root.exists():
        shard_rows = [
            path for path in external_shard_root.rglob("*")
            if path.is_file() and ".local_fallback" not in path.name
        ]
        candidates.extend(shard_rows)
    if external_shard_root.exists():
        shard_local_fallback_rows = [path for path in external_shard_root.glob("*.local_fallback*") if path.is_file()]
        candidates.extend(shard_local_fallback_rows)

    candidates = _dedupe_paths(candidates)
    details["pressure_candidates"] = int(len(candidates))
    details["pressure_mirrored_top_level_candidates"] = int(mirrored_count)
    details["pressure_local_fallback_copy_candidates"] = int(len(local_fallback_copies))
    details["pressure_shard_file_candidates"] = int(len(shard_rows))
    details["pressure_shard_local_fallback_candidates"] = int(len(shard_local_fallback_rows))
    return candidates, details


def _collect_old_top_level_pattern_files(base: Path, pattern: str, older_than_days: int) -> list[Path]:
    if not base.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(int(older_than_days), 0))
    out: list[Path] = []
    for p in base.glob(pattern):
        if not p.is_file():
            continue
        try:
            mt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mt < cutoff:
            out.append(p)
    return out


def _collect_old_nested_pattern_files(base: Path, pattern: str, older_than_days: int) -> list[Path]:
    if not base.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(int(older_than_days), 0))
    out: list[Path] = []
    for p in base.rglob(pattern):
        if not p.is_file():
            continue
        try:
            mt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mt < cutoff:
            out.append(p)
    return out


def _parse_timeline_stamp(stamp: str) -> datetime | None:
    try:
        dt = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _parse_day_stamp(stamp: str) -> datetime | None:
    try:
        dt = datetime.strptime(stamp, "%Y%m%d")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _collect_old_stamped_files(
    base: Path,
    stamp_re: re.Pattern[str],
    *,
    older_than_days: int,
    keep_latest_runs: int,
    parse_stamp_fn: Callable[[str], datetime | None],
) -> tuple[list[Path], int, int]:
    if not base.exists():
        return [], 0, 0

    rows: list[tuple[Path, str, datetime]] = []
    total_files = 0
    for p in base.iterdir():
        if not p.is_file():
            continue
        total_files += 1
        match = stamp_re.match(p.name)
        if not match:
            continue
        stamp = match.group(1)
        dt = parse_stamp_fn(stamp)
        if dt is None:
            continue
        rows.append((p, stamp, dt))

    keep_latest_runs = max(int(keep_latest_runs), 0)
    unique_stamps = sorted({stamp: dt for _, stamp, dt in rows}.items(), key=lambda item: item[1])
    keep_stamps = {stamp for stamp, _ in unique_stamps[-keep_latest_runs:]} if keep_latest_runs > 0 else set()

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(int(older_than_days), 0))
    out: list[Path] = []
    for p, stamp, dt in rows:
        if stamp in keep_stamps:
            continue
        if dt < cutoff:
            out.append(p)

    for p in base.glob("*.local_fallback*"):
        try:
            mt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mt < cutoff:
            out.append(p)

    return out, total_files, len(unique_stamps)


def _collect_old_timeline_files(base: Path, older_than_days: int, keep_latest_runs: int) -> tuple[list[Path], int, int]:
    if not base.exists():
        return [], 0, 0

    rows: list[tuple[Path, str, datetime]] = []
    total_files = 0
    for p in base.iterdir():
        if not p.is_file():
            continue
        total_files += 1
        match = TIMELINE_STAMP_RE.match(p.name)
        if not match:
            continue
        stamp = match.group(1)
        dt = _parse_timeline_stamp(stamp)
        if dt is None:
            continue
        rows.append((p, stamp, dt))

    keep_latest_runs = max(int(keep_latest_runs), 0)
    unique_stamps = sorted({stamp: dt for _, stamp, dt in rows}.items(), key=lambda item: item[1])
    keep_stamps = {stamp for stamp, _ in unique_stamps[-keep_latest_runs:]} if keep_latest_runs > 0 else set()

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(int(older_than_days), 0))
    out: list[Path] = []
    for p, stamp, dt in rows:
        if stamp in keep_stamps:
            continue
        if dt < cutoff:
            out.append(p)

    for p in base.glob("*.local_fallback"):
        try:
            mt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mt < cutoff:
            out.append(p)

    return out, total_files, len(unique_stamps)


def _parse_snapshot_dir_ts(path: Path) -> datetime:
    name = str(path.name)
    try:
        dt = datetime.strptime(name, "%Y%m%d_%H%M%S")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return datetime.now(timezone.utc)


def _snapshot_file_count(path: Path) -> int:
    n = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            n += 1
    return n


def _collect_old_snapshot_dirs(base: Path, older_than_days: int, keep_latest: int) -> tuple[list[Path], int]:
    if not base.exists():
        return [], 0

    dirs = [
        p for p in base.iterdir()
        if p.is_dir() and p.name != "latest"
    ]
    dirs.sort(key=_parse_snapshot_dir_ts)

    total_dirs = len(dirs)
    keep_latest = max(int(keep_latest), 0)
    keep_names = {d.name for d in dirs[-keep_latest:]} if keep_latest > 0 else set()

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(int(older_than_days), 0))
    out: list[Path] = []
    for d in dirs:
        if d.name in keep_names:
            continue
        if _parse_snapshot_dir_ts(d) < cutoff:
            out.append(d)

    return out, total_dirs


def _sqlite_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 ** 3)


def _vacuum_sqlite(path: Path) -> bool:
    conn = None
    try:
        conn = sqlite3.connect(str(path))
        conn.execute("VACUUM")
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _snapshot_training_coverage_ready(project_root: Path, *, max_age_hours: float) -> tuple[bool, str, dict[str, object]]:
    path = project_root / "governance" / "health" / "snapshot_training_coverage_latest.json"
    details: dict[str, object] = {
        "coverage_file": str(path),
        "max_age_hours": float(max(max_age_hours, 0.0)),
    }
    if not path.exists():
        return False, "missing_snapshot_training_coverage_artifact", details

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        details["error"] = str(exc)
        return False, "invalid_snapshot_training_coverage_artifact", details

    ts_raw = payload.get("timestamp_utc")
    try:
        ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts = ts.astimezone(timezone.utc)
    except Exception:
        return False, "invalid_snapshot_training_coverage_timestamp", details

    age_hours = max((datetime.now(timezone.utc) - ts).total_seconds() / 3600.0, 0.0)
    details.update(
        {
            "timestamp_utc": ts.isoformat(),
            "age_hours": round(age_hours, 4),
            "all_snapshot_data_incorporated": bool(payload.get("all_snapshot_data_incorporated", False)),
            "snapshot_raw_sql_ingest_ratio": float(payload.get("snapshot_raw_sql_ingest_ratio", 0.0) or 0.0),
            "snapshot_cov_fill_ratio": float(payload.get("snapshot_cov_fill_ratio", 0.0) or 0.0),
            "snapshot_feature_coverage_ratio": float(payload.get("snapshot_feature_coverage_ratio", 0.0) or 0.0),
            "reason": str(payload.get("reason") or ""),
        }
    )
    if age_hours > max(float(max_age_hours), 0.0):
        return False, "stale_snapshot_training_coverage", details
    if not bool(payload.get("all_snapshot_data_incorporated", False)):
        return False, "snapshot_training_not_fully_incorporated", details
    return True, "ok", details


def _invoke_with_timeout(timeout_seconds: float, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    timeout_seconds = float(timeout_seconds or 0.0)
    if timeout_seconds <= 0:
        try:
            return {"ok": True, "timed_out": False, "timeout_seconds": 0.0, "result": fn(*args, **kwargs)}
        except Exception as exc:
            return {"ok": False, "timed_out": False, "timeout_seconds": 0.0, "error": str(exc)}

    if not hasattr(signal, "SIGALRM"):
        try:
            return {
                "ok": True,
                "timed_out": False,
                "timeout_seconds": timeout_seconds,
                "timeout_supported": False,
                "result": fn(*args, **kwargs),
            }
        except Exception as exc:
            return {
                "ok": False,
                "timed_out": False,
                "timeout_seconds": timeout_seconds,
                "timeout_supported": False,
                "error": str(exc),
            }

    alarm_seconds = max(int(math.ceil(timeout_seconds)), 1)

    class _OperationTimeout(Exception):
        pass

    def _alarm_handler(_signum, _frame):
        raise _OperationTimeout(f"timed_out_after_{alarm_seconds}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(alarm_seconds)
    try:
        result = fn(*args, **kwargs)
        return {"ok": True, "timed_out": False, "timeout_seconds": float(alarm_seconds), "result": result}
    except _OperationTimeout as exc:
        return {"ok": False, "timed_out": True, "timeout_seconds": float(alarm_seconds), "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "timed_out": False, "timeout_seconds": float(alarm_seconds), "error": str(exc)}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune old data artifacts by retention policy.")
    parser.add_argument("--decisions-days", type=int, default=int(os.getenv("RETENTION_DECISIONS_DAYS", "30")))
    parser.add_argument("--decision-explanations-days", type=int, default=int(os.getenv("RETENTION_DECISION_EXPLANATIONS_DAYS", "30")))
    parser.add_argument("--governance-days", type=int, default=int(os.getenv("RETENTION_GOVERNANCE_DAYS", "45")))
    parser.add_argument("--exports-days", type=int, default=int(os.getenv("RETENTION_EXPORTS_DAYS", "30")))
    parser.add_argument("--backup-drills-days", type=int, default=int(os.getenv("RETENTION_BACKUP_DRILLS_DAYS", "14")))
    parser.add_argument("--csv-days", type=int, default=int(os.getenv("RETENTION_CSV_DAYS", "10")))
    parser.add_argument("--logs-days", type=int, default=int(os.getenv("RETENTION_LOGS_DAYS", "14")))
    parser.add_argument("--watchdog-events-days", type=int, default=int(os.getenv("RETENTION_WATCHDOG_EVENTS_DAYS", "30")))
    parser.add_argument("--governance-channels-days", type=int, default=int(os.getenv("RETENTION_GOVERNANCE_CHANNELS_DAYS", "7")))
    parser.add_argument("--governance-shadow-days", type=int, default=int(os.getenv("RETENTION_GOVERNANCE_SHADOW_DAYS", "7")))
    parser.add_argument("--governance-health-days", type=int, default=int(os.getenv("RETENTION_GOVERNANCE_HEALTH_DAYS", "14")))
    parser.add_argument("--data-local-fallback-days", type=int, default=int(os.getenv("RETENTION_DATA_LOCAL_FALLBACK_DAYS", "1")))
    parser.add_argument(
        "--external-live-sqlite-days",
        type=int,
        default=int(os.getenv("RETENTION_EXTERNAL_LIVE_SQLITE_DAYS", "1")),
    )
    parser.add_argument(
        "--external-live-sqlite-require-local-fallback",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("RETENTION_EXTERNAL_LIVE_SQLITE_REQUIRE_LOCAL_FALLBACK", "1").strip() == "1",
    )
    parser.add_argument(
        "--external-live-sqlite-low-space-force",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("RETENTION_EXTERNAL_LIVE_SQLITE_LOW_SPACE_FORCE", "1").strip() == "1",
    )
    parser.add_argument("--project-timeline-days", type=int, default=int(os.getenv("RETENTION_PROJECT_TIMELINE_DAYS", "30")))
    parser.add_argument("--project-timeline-keep-runs", type=int, default=int(os.getenv("RETENTION_PROJECT_TIMELINE_KEEP_RUNS", "40")))
    parser.add_argument("--debug-snapshots-days", type=int, default=int(os.getenv("RETENTION_DEBUG_SNAPSHOTS_DAYS", "3")))
    parser.add_argument("--debug-snapshots-keep", type=int, default=int(os.getenv("RETENTION_DEBUG_SNAPSHOTS_KEEP", "24")))
    parser.add_argument("--prune-debug-snapshots", action=argparse.BooleanOptionalAction, default=os.getenv("RETENTION_PRUNE_DEBUG_SNAPSHOTS", "1").strip() == "1")
    parser.add_argument("--require-debug-snapshot-sync", action=argparse.BooleanOptionalAction, default=os.getenv("RETENTION_REQUIRE_DEBUG_SNAPSHOT_SYNC", "1").strip() == "1")
    parser.add_argument(
        "--require-training-success",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("RETENTION_REQUIRE_TRAINING_SUCCESS", "0").strip() == "1",
        help="Only purge debug snapshots after confirmed successful training with snapshot coverage incorporated.",
    )
    parser.add_argument(
        "--training-success-max-age-hours",
        type=float,
        default=float(os.getenv("RETENTION_TRAINING_SUCCESS_MAX_AGE_HOURS", "168")),
    )
    parser.add_argument(
        "--require-snapshot-training-coverage",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("RETENTION_REQUIRE_SNAPSHOT_TRAINING_COVERAGE", "0").strip() == "1",
        help="Only purge debug snapshots after the latest snapshot-training coverage artifact confirms full incorporation.",
    )
    parser.add_argument(
        "--snapshot-training-max-age-hours",
        type=float,
        default=float(os.getenv("RETENTION_SNAPSHOT_TRAINING_MAX_AGE_HOURS", "168")),
    )
    parser.add_argument("--debug-snapshot-sync-timeout-seconds", type=float, default=float(os.getenv("RETENTION_DEBUG_SNAPSHOT_SYNC_TIMEOUT_SECONDS", "600")))
    parser.add_argument("--sqlite-path", default=os.getenv("SNAPSHOT_CONTEXT_SQLITE_PATH", str(PROJECT_ROOT / "data" / "snapshot_context.sqlite3")))
    parser.add_argument("--sqlite-vacuum-over-gb", type=float, default=6.0)
    parser.add_argument("--sqlite-vacuum-timeout-seconds", type=float, default=float(os.getenv("RETENTION_SQLITE_VACUUM_TIMEOUT_SECONDS", "900")))
    parser.add_argument("--skip-sqlite-vacuum", action=argparse.BooleanOptionalAction, default=os.getenv("RETENTION_SKIP_SQLITE_VACUUM", "0").strip() == "1")
    parser.add_argument("--archive-prune", action=argparse.BooleanOptionalAction, default=os.getenv("RETENTION_ARCHIVE_PRUNE_ENABLED", "1").strip() == "1")
    parser.add_argument("--archive-db", default=os.getenv("RETENTION_ARCHIVE_DB", str(PROJECT_ROOT / "data" / "jsonl_link_archive.sqlite3")))
    parser.add_argument("--archive-root", default=os.getenv("RETENTION_ARCHIVE_ROOT", str(PROJECT_ROOT / "data" / "jsonl_link_archives")))
    parser.add_argument("--archive-retention-days", type=int, default=int(os.getenv("RETENTION_ARCHIVE_RETENTION_DAYS", os.getenv("SQL_LINK_SERVICE_HOT_ARCHIVE_RETENTION_DAYS", "7"))))
    parser.add_argument("--archive-prune-vacuum", action=argparse.BooleanOptionalAction, default=os.getenv("RETENTION_ARCHIVE_PRUNE_VACUUM", os.getenv("SQL_LINK_SERVICE_HOT_ARCHIVE_PRUNE_VACUUM", "1")).strip() == "1")
    parser.add_argument("--archive-cold-export-root", default=os.getenv("RETENTION_ARCHIVE_COLD_EXPORT_ROOT", os.getenv("SQL_LINK_SERVICE_HOT_COLD_ARCHIVE_ROOT", "")))
    parser.add_argument("--archive-cold-export-format", default=os.getenv("RETENTION_ARCHIVE_COLD_EXPORT_FORMAT", os.getenv("SQL_LINK_SERVICE_HOT_COLD_ARCHIVE_FORMAT", "parquet")))
    parser.add_argument("--archive-cold-export-batch-size", type=int, default=int(os.getenv("RETENTION_ARCHIVE_COLD_EXPORT_BATCH_SIZE", os.getenv("SQL_LINK_SERVICE_HOT_COLD_ARCHIVE_BATCH_SIZE", "50000"))))
    parser.add_argument("--archive-cold-export-compression", default=os.getenv("RETENTION_ARCHIVE_COLD_EXPORT_COMPRESSION", os.getenv("SQL_LINK_SERVICE_HOT_COLD_ARCHIVE_COMPRESSION", "zstd")))
    parser.add_argument("--stale-stage", action=argparse.BooleanOptionalAction, default=os.getenv("RETENTION_STALE_STAGE_ENABLED", "0").strip() == "1")
    parser.add_argument("--stale-stage-root", default=os.getenv("RETENTION_STALE_STAGE_ROOT", str(_default_stale_stage_root(PROJECT_ROOT))))
    parser.add_argument("--stale-stage-manifest", default=os.getenv("RETENTION_STALE_STAGE_MANIFEST", ""))
    parser.add_argument("--stale-stage-sections", default=os.getenv("RETENTION_STALE_STAGE_SECTIONS", ",".join(DEFAULT_STALE_STAGE_SECTION_TOKENS)))
    parser.add_argument("--stale-purge", action=argparse.BooleanOptionalAction, default=os.getenv("RETENTION_STALE_PURGE_ENABLED", "0").strip() == "1")
    parser.add_argument("--stale-purge-days", type=int, default=int(os.getenv("RETENTION_STALE_PURGE_DAYS", "30")))
    parser.add_argument("--json", action="store_true", help="Accepted for orchestrator compatibility; JSON is always printed.")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    out = PROJECT_ROOT / "governance" / "health" / "data_retention_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(
        os.getenv(
            "DATA_RETENTION_LOCK_PATH",
            str(PROJECT_ROOT / "governance" / "locks" / "data_retention.lock"),
        )
    )
    lock_fh, lock_owner = _acquire_singleton_lock(lock_path)
    if lock_fh is None:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "apply": bool(args.apply),
            "busy": True,
            "skipped_reason": "lock_busy",
            "lock_path": str(lock_path),
            "lock_owner": lock_owner,
            "archive_pruning": {
                "enabled": bool(args.archive_prune),
                "ran": False,
                "details": {},
            },
        }
        out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=True))
        return 0

    targets = {
        "decisions": (PROJECT_ROOT / "decisions", args.decisions_days),
        "decision_explanations": (PROJECT_ROOT / "decision_explanations", args.decision_explanations_days),
        "governance_watchdog": (PROJECT_ROOT / "governance" / "watchdog", args.governance_days),
        "governance_events": (PROJECT_ROOT / "governance" / "events", args.watchdog_events_days),
        "governance_channels": (PROJECT_ROOT / "governance" / "channels", args.governance_channels_days),
        "governance_health": (PROJECT_ROOT / "governance" / "health", args.governance_health_days),
        "exports_sql_reports": (PROJECT_ROOT / "exports" / "sql_reports", args.exports_days),
        "exports_csv": (PROJECT_ROOT / "exports" / "csv", args.csv_days),
        "backup_drills": (PROJECT_ROOT / "exports" / "backup_drills", args.backup_drills_days),
        "logs": (PROJECT_ROOT / "logs", args.logs_days),
    }

    external_csv_root = _resolve_external_project_root() / "exports" / "csv"
    project_csv_root = PROJECT_ROOT / "exports" / "csv"
    if not _same_path(external_csv_root, project_csv_root):
        targets["exports_csv_external"] = (external_csv_root, args.csv_days)

    governance_root = PROJECT_ROOT / "governance"
    if governance_root.exists():
        for child in sorted(governance_root.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith("shadow"):
                continue
            targets[f"governance_{name}"] = (child, args.governance_shadow_days)

    to_delete: list[Path] = []
    candidate_rows_by_label: dict[str, list[Path]] = {}
    summary: dict[str, dict[str, int]] = {}
    for label, (base, days) in targets.items():
        rows = _collect_old_files(base, days)
        to_delete.extend(rows)
        candidate_rows_by_label[label] = list(rows)
        summary[label] = {"candidates": len(rows), "older_than_days": int(days)}

    data_local_fallback_rows = _collect_old_top_level_pattern_files(
        PROJECT_ROOT / "data",
        "*.local_fallback*",
        args.data_local_fallback_days,
    )
    to_delete.extend(data_local_fallback_rows)
    candidate_rows_by_label["data_local_fallback"] = list(data_local_fallback_rows)
    summary["data_local_fallback"] = {
        "candidates": len(data_local_fallback_rows),
        "older_than_days": int(args.data_local_fallback_days),
    }

    shard_local_fallback_rows = _collect_old_nested_pattern_files(
        PROJECT_ROOT / "data" / "sql_link_shards",
        "*.local_fallback*",
        args.data_local_fallback_days,
    )
    to_delete.extend(shard_local_fallback_rows)
    candidate_rows_by_label["data_sql_link_shard_local_fallback"] = list(shard_local_fallback_rows)
    summary["data_sql_link_shard_local_fallback"] = {
        "candidates": len(shard_local_fallback_rows),
        "older_than_days": int(args.data_local_fallback_days),
    }

    external_csv_local_fallback_rows = _collect_old_top_level_pattern_files(
        external_csv_root,
        "*.local_fallback*",
        args.data_local_fallback_days,
    )
    to_delete.extend(external_csv_local_fallback_rows)
    candidate_rows_by_label["exports_csv_external_local_fallback"] = list(external_csv_local_fallback_rows)
    summary["exports_csv_external_local_fallback"] = {
        "candidates": len(external_csv_local_fallback_rows),
        "older_than_days": int(args.data_local_fallback_days),
        "base": str(external_csv_root),
    }

    external_live_sqlite_rows, external_live_sqlite_meta = _collect_external_live_sqlite_rows(
        PROJECT_ROOT,
        _resolve_external_project_root(),
        older_than_days=args.external_live_sqlite_days,
        require_local_fallback=bool(args.external_live_sqlite_require_local_fallback),
    )
    external_live_sqlite_pressure_rows: list[Path] = []
    external_live_sqlite_pressure_meta: dict[str, object] = {}
    if args.external_live_sqlite_low_space_force:
        external_live_sqlite_pressure_rows, external_live_sqlite_pressure_meta = _collect_external_live_sqlite_pressure_rows(
            PROJECT_ROOT,
            _resolve_external_project_root(),
            require_local_fallback=bool(args.external_live_sqlite_require_local_fallback),
        )
        external_live_sqlite_rows = _dedupe_paths(external_live_sqlite_rows + external_live_sqlite_pressure_rows)
    to_delete.extend(external_live_sqlite_rows)
    candidate_rows_by_label["external_live_sqlite"] = list(external_live_sqlite_rows)
    summary["external_live_sqlite"] = {
        "candidates": len(external_live_sqlite_rows),
        "older_than_days": int(args.external_live_sqlite_days),
        "low_space_force_enabled": bool(args.external_live_sqlite_low_space_force),
        "low_space_force_candidates": int(len(external_live_sqlite_pressure_rows)),
        **external_live_sqlite_meta,
        **external_live_sqlite_pressure_meta,
    }

    timeline_dir = PROJECT_ROOT / "exports" / "reports" / "project_timeline"
    timeline_rows, timeline_total_files, timeline_total_runs = _collect_old_timeline_files(
        timeline_dir,
        older_than_days=args.project_timeline_days,
        keep_latest_runs=args.project_timeline_keep_runs,
    )
    to_delete.extend(timeline_rows)
    candidate_rows_by_label["exports_project_timeline"] = list(timeline_rows)
    summary["exports_project_timeline"] = {
        "candidates": len(timeline_rows),
        "older_than_days": int(args.project_timeline_days),
        "keep_latest_runs": int(args.project_timeline_keep_runs),
        "total_files": int(timeline_total_files),
        "total_runs": int(timeline_total_runs),
    }

    crash_dir = PROJECT_ROOT / "exports" / "reports" / "crash_reports"
    crash_rows, crash_total_files, crash_total_runs = _collect_old_stamped_files(
        crash_dir,
        CRASH_REPORT_STAMP_RE,
        older_than_days=args.exports_days,
        keep_latest_runs=0,
        parse_stamp_fn=_parse_timeline_stamp,
    )
    to_delete.extend(crash_rows)
    candidate_rows_by_label["exports_crash_reports"] = list(crash_rows)
    summary["exports_crash_reports"] = {
        "candidates": len(crash_rows),
        "older_than_days": int(args.exports_days),
        "total_files": int(crash_total_files),
        "total_runs": int(crash_total_runs),
    }

    training_dir = PROJECT_ROOT / "exports" / "reports" / "training_reports"
    training_rows, training_total_files, training_total_runs = _collect_old_stamped_files(
        training_dir,
        TRAINING_REPORT_STAMP_RE,
        older_than_days=args.exports_days,
        keep_latest_runs=0,
        parse_stamp_fn=_parse_timeline_stamp,
    )
    to_delete.extend(training_rows)
    candidate_rows_by_label["exports_training_reports"] = list(training_rows)
    summary["exports_training_reports"] = {
        "candidates": len(training_rows),
        "older_than_days": int(args.exports_days),
        "total_files": int(training_total_files),
        "total_runs": int(training_total_runs),
    }

    daily_ops_dir = PROJECT_ROOT / "exports" / "reports"
    daily_ops_rows, daily_ops_total_files, daily_ops_total_runs = _collect_old_stamped_files(
        daily_ops_dir,
        DAILY_OPS_REPORT_RE,
        older_than_days=args.exports_days,
        keep_latest_runs=0,
        parse_stamp_fn=_parse_day_stamp,
    )
    to_delete.extend(daily_ops_rows)
    candidate_rows_by_label["exports_daily_ops_reports"] = list(daily_ops_rows)
    summary["exports_daily_ops_reports"] = {
        "candidates": len(daily_ops_rows),
        "older_than_days": int(args.exports_days),
        "total_files": int(daily_ops_total_files),
        "total_runs": int(daily_ops_total_runs),
    }

    one_numbers_dir = PROJECT_ROOT / "exports" / "one_numbers"
    one_numbers_rows, one_numbers_total_files, one_numbers_total_runs = _collect_old_stamped_files(
        one_numbers_dir,
        ONE_NUMBERS_STAMP_RE,
        older_than_days=args.exports_days,
        keep_latest_runs=0,
        parse_stamp_fn=_parse_timeline_stamp,
    )
    to_delete.extend(one_numbers_rows)
    candidate_rows_by_label["exports_one_numbers"] = list(one_numbers_rows)
    summary["exports_one_numbers"] = {
        "candidates": len(one_numbers_rows),
        "older_than_days": int(args.exports_days),
        "total_files": int(one_numbers_total_files),
        "total_runs": int(one_numbers_total_runs),
    }

    external_root = _resolve_external_project_root()
    stale_stage_sections = _parse_csv_tokens(
        args.stale_stage_sections,
        default=DEFAULT_STALE_STAGE_SECTION_TOKENS,
    )
    stale_stage_root = Path(args.stale_stage_root).expanduser()
    stale_stage_manifest = _stale_manifest_path(stale_stage_root, args.stale_stage_manifest)
    stale_stage_rows_by_label: dict[str, list[Path]] = {}
    hard_delete_rows: list[Path] = []
    for label, rows in candidate_rows_by_label.items():
        if args.stale_stage and _label_matches_stale_stage(label, stale_stage_sections):
            stale_stage_rows_by_label[label] = list(rows)
        else:
            hard_delete_rows.extend(rows)

    stale_stage_payload: dict[str, object] = {
        "enabled": bool(args.stale_stage),
        "root": str(stale_stage_root),
        "manifest_path": str(stale_stage_manifest),
        "sections": sorted(stale_stage_sections),
        "candidate_files": int(sum(len(rows) for rows in stale_stage_rows_by_label.values())),
        "candidate_bytes": int(sum(_path_size_bytes(path) for rows in stale_stage_rows_by_label.values() for path in rows)),
        "staged_files": 0,
        "staged_bytes": 0,
        "delete_errors": 0,
        "staged_by_label": {},
        "purge_enabled": bool(args.stale_purge),
        "purge": {},
    }

    file_deleted = 0
    delete_errors = 0
    if args.apply:
        if args.stale_stage:
            staged_by_label: dict[str, dict[str, object]] = {}
            staged_files = 0
            staged_bytes = 0
            stage_errors = 0
            for label, rows in stale_stage_rows_by_label.items():
                result = _move_paths_to_stale_stage(
                    paths=rows,
                    label=label,
                    project_root=PROJECT_ROOT,
                    external_root=external_root,
                    stale_root=stale_stage_root,
                    manifest_path=stale_stage_manifest,
                )
                staged_by_label[label] = {
                    "candidate_files": int(len(rows)),
                    "staged_files": int(result.get("moved", 0) or 0),
                    "staged_bytes": int(result.get("moved_bytes", 0) or 0),
                    "errors": list(result.get("errors", []) or []),
                }
                staged_files += int(result.get("moved", 0) or 0)
                staged_bytes += int(result.get("moved_bytes", 0) or 0)
                stage_errors += len(result.get("errors", []) or [])
            stale_stage_payload["staged_by_label"] = staged_by_label
            stale_stage_payload["staged_files"] = int(staged_files)
            stale_stage_payload["staged_bytes"] = int(staged_bytes)
            stale_stage_payload["delete_errors"] = int(stage_errors)
        if args.stale_purge:
            stale_stage_payload["purge"] = _purge_old_stale_stage(
                stale_root=stale_stage_root,
                manifest_path=stale_stage_manifest,
                older_than_days=int(args.stale_purge_days),
            )
        file_deleted, delete_errors = _delete_paths(hard_delete_rows)

    sqlite_path = Path(args.sqlite_path)

    debug_root = PROJECT_ROOT / "exports" / "debug_snapshots"
    debug_candidates, debug_total_dirs = _collect_old_snapshot_dirs(
        debug_root,
        older_than_days=args.debug_snapshots_days,
        keep_latest=args.debug_snapshots_keep,
    )

    debug_sync_meta: dict[str, object] = {}
    debug_cov_meta: dict[str, object] = {}
    debug_training_guard: dict[str, object] = {}
    debug_snapshot_training_guard: dict[str, object] = {}
    debug_dirs_deleted = 0
    debug_files_deleted = 0
    debug_dirs_skipped_not_ready = 0
    debug_dirs_skipped_training_guard = 0
    debug_dirs_staged = 0
    debug_files_staged = 0

    if args.apply and args.prune_debug_snapshots and debug_candidates:
        if args.require_training_success:
            guard_ok, guard_reason, guard_details = check_confirmed_training_success(
                project_root=str(PROJECT_ROOT),
                max_age_hours=float(args.training_success_max_age_hours),
                require_snapshot_training_complete=True,
            )
            debug_training_guard = {
                "ok": bool(guard_ok),
                "reason": str(guard_reason),
                "details": guard_details if isinstance(guard_details, dict) else {},
            }
            if not guard_ok:
                debug_dirs_skipped_training_guard = len(debug_candidates)
                debug_candidates = []

        if args.require_snapshot_training_coverage and debug_candidates:
            snap_ok, snap_reason, snap_details = _snapshot_training_coverage_ready(
                PROJECT_ROOT,
                max_age_hours=float(args.snapshot_training_max_age_hours),
            )
            debug_snapshot_training_guard = {
                "ok": bool(snap_ok),
                "reason": str(snap_reason),
                "details": snap_details if isinstance(snap_details, dict) else {},
            }
            if not snap_ok:
                debug_dirs_skipped_training_guard += len(debug_candidates)
                debug_candidates = []

        if args.require_debug_snapshot_sync:
            sync_result = _invoke_with_timeout(
                args.debug_snapshot_sync_timeout_seconds,
                sync_raw_debug_snapshots_to_sqlite,
                project_root=PROJECT_ROOT,
                sqlite_path=sqlite_path,
                snapshot_dirs=debug_candidates,
            )
            if bool(sync_result.get("ok", False)):
                value = sync_result.get("result")
                if isinstance(value, dict):
                    debug_sync_meta = value
                else:
                    debug_sync_meta = {"result": value}
            else:
                debug_sync_meta = {
                    "error": str(sync_result.get("error", "unknown_sync_error")),
                    "timed_out": bool(sync_result.get("timed_out", False)),
                    "timeout_seconds": float(sync_result.get("timeout_seconds", 0.0)),
                }
                if "timeout_supported" in sync_result:
                    debug_sync_meta["timeout_supported"] = bool(sync_result.get("timeout_supported"))

        ready_ids: set[str] = set()
        try:
            cov = debug_snapshot_ingest_coverage(
                project_root=PROJECT_ROOT,
                sqlite_path=sqlite_path,
                snapshot_dirs=debug_candidates,
            )
            rows = cov.get("rows") if isinstance(cov.get("rows"), list) else []
            for row in rows:
                if isinstance(row, dict) and bool(row.get("ready", False)):
                    ready_ids.add(str(row.get("snapshot_id", "")))
            debug_cov_meta = {
                "snapshot_total": int(cov.get("snapshot_total", 0) or 0),
                "ready_total": int(cov.get("ready_total", 0) or 0),
                "coverage_ratio": float(cov.get("coverage_ratio", 0.0) or 0.0),
            }
        except Exception as exc:
            debug_cov_meta = {"error": str(exc)}

        for d in debug_candidates:
            if args.require_debug_snapshot_sync and d.name not in ready_ids:
                debug_dirs_skipped_not_ready += 1
                continue
            try:
                file_count = _snapshot_file_count(d)
                if args.stale_stage and _label_matches_stale_stage("debug_snapshots", stale_stage_sections):
                    result = _move_paths_to_stale_stage(
                        paths=[d],
                        label="debug_snapshots",
                        project_root=PROJECT_ROOT,
                        external_root=external_root,
                        stale_root=stale_stage_root,
                        manifest_path=stale_stage_manifest,
                    )
                    if int(result.get("moved", 0) or 0) > 0:
                        debug_dirs_staged += 1
                        debug_files_staged += int(file_count)
                        stale_stage_payload["staged_files"] = int(stale_stage_payload.get("staged_files", 0) or 0) + int(result.get("moved", 0) or 0)
                        stale_stage_payload["staged_bytes"] = int(stale_stage_payload.get("staged_bytes", 0) or 0) + int(result.get("moved_bytes", 0) or 0)
                        staged_by_label = stale_stage_payload.get("staged_by_label")
                        if not isinstance(staged_by_label, dict):
                            staged_by_label = {}
                            stale_stage_payload["staged_by_label"] = staged_by_label
                        entry = staged_by_label.setdefault(
                            "debug_snapshots",
                            {
                                "candidate_files": 0,
                                "staged_files": 0,
                                "staged_bytes": 0,
                                "errors": [],
                            },
                        )
                        entry["candidate_files"] = int(entry.get("candidate_files", 0) or 0) + 1
                        entry["staged_files"] = int(entry.get("staged_files", 0) or 0) + int(result.get("moved", 0) or 0)
                        entry["staged_bytes"] = int(entry.get("staged_bytes", 0) or 0) + int(result.get("moved_bytes", 0) or 0)
                        if result.get("errors"):
                            entry["errors"] = list(entry.get("errors", []) or []) + list(result.get("errors", []) or [])
                            stale_stage_payload["delete_errors"] = int(stale_stage_payload.get("delete_errors", 0) or 0) + len(result.get("errors", []) or [])
                    else:
                        stale_stage_payload["delete_errors"] = int(stale_stage_payload.get("delete_errors", 0) or 0) + len(result.get("errors", []) or [])
                else:
                    shutil.rmtree(d)
                    debug_dirs_deleted += 1
                    debug_files_deleted += int(file_count)
            except OSError:
                continue

    size_before = _sqlite_size_gb(sqlite_path)
    vacuum_ran = False
    vacuum_ok = None
    vacuum_skipped = bool(args.skip_sqlite_vacuum)
    vacuum_meta: dict[str, object] = {}
    if args.apply and (not args.skip_sqlite_vacuum) and sqlite_path.exists() and size_before >= args.sqlite_vacuum_over_gb:
        vacuum_ran = True
        vacuum_result = _invoke_with_timeout(
            args.sqlite_vacuum_timeout_seconds,
            _vacuum_sqlite,
            sqlite_path,
        )
        if bool(vacuum_result.get("ok", False)):
            vacuum_ok = bool(vacuum_result.get("result", False))
        else:
            vacuum_ok = False
        vacuum_meta = {
            "timed_out": bool(vacuum_result.get("timed_out", False)),
            "timeout_seconds": float(vacuum_result.get("timeout_seconds", 0.0)),
        }
        if "timeout_supported" in vacuum_result:
            vacuum_meta["timeout_supported"] = bool(vacuum_result.get("timeout_supported"))
        if vacuum_result.get("error") is not None:
            vacuum_meta["error"] = str(vacuum_result.get("error"))
    size_after = _sqlite_size_gb(sqlite_path)

    archive_pruning: dict[str, object] = {
        "enabled": bool(args.archive_prune),
        "ran": False,
        "details": {},
    }
    if args.apply and args.archive_prune:
        archive_db = Path(args.archive_db)
        archive_root_raw = str(args.archive_root or "").strip()
        archive_root = Path(archive_root_raw).expanduser() if archive_root_raw else None
        cold_export_root_raw = str(args.archive_cold_export_root or "").strip()
        cold_export_root = Path(cold_export_root_raw).expanduser() if cold_export_root_raw else None
        archive_pruning["ran"] = True
        try:
            archive_pruning["details"] = _prune_archive_storage(
                archive_db=archive_db,
                archive_root=archive_root,
                archive_retention_days=int(args.archive_retention_days),
                archive_prune_vacuum=bool(args.archive_prune_vacuum),
                cold_export_root=cold_export_root,
                cold_export_format=str(args.archive_cold_export_format or "parquet"),
                cold_export_batch_size=int(args.archive_cold_export_batch_size),
                cold_export_compression=str(args.archive_cold_export_compression or "zstd"),
            )
        except Exception as exc:
            archive_pruning["details"] = {"error": str(exc)}

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "apply": bool(args.apply),
        "lock_path": str(lock_path),
        "targets": summary,
        "total_candidates": int(len(to_delete) + len(debug_candidates)),
        "deleted": int(file_deleted + debug_files_deleted),
        "deleted_files": int(file_deleted),
        "delete_errors": int(delete_errors),
        "debug_snapshots": {
            "enabled": bool(args.prune_debug_snapshots),
            "require_sync": bool(args.require_debug_snapshot_sync),
            "retention_days": int(args.debug_snapshots_days),
            "keep_latest": int(args.debug_snapshots_keep),
            "sync_timeout_seconds": float(args.debug_snapshot_sync_timeout_seconds),
            "total_dirs": int(debug_total_dirs),
            "candidate_dirs": int(len(debug_candidates)),
            "deleted_dirs": int(debug_dirs_deleted),
            "deleted_files": int(debug_files_deleted),
            "staged_dirs": int(debug_dirs_staged),
            "staged_files": int(debug_files_staged),
            "skipped_not_ready": int(debug_dirs_skipped_not_ready),
            "skipped_training_guard": int(debug_dirs_skipped_training_guard),
            "sql_sync": debug_sync_meta,
            "ingest_coverage": debug_cov_meta,
            "training_success_guard": debug_training_guard,
            "snapshot_training_guard": debug_snapshot_training_guard,
        },
        "sqlite": {
            "path": str(sqlite_path),
            "size_gb_before": round(size_before, 3),
            "size_gb_after": round(size_after, 3),
            "vacuum_threshold_gb": args.sqlite_vacuum_over_gb,
            "vacuum_timeout_seconds": float(args.sqlite_vacuum_timeout_seconds),
            "vacuum_skipped": bool(vacuum_skipped),
            "vacuum_ran": vacuum_ran,
            "vacuum_ok": vacuum_ok,
            "vacuum_meta": vacuum_meta,
        },
        "archive_pruning": archive_pruning,
        "stale_stage": stale_stage_payload,
    }

    try:
        out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=True))
        return 0
    finally:
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
