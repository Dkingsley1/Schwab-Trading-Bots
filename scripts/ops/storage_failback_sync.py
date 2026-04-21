import argparse
import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _external_project_root() -> Path:
    configured = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    mount_root = Path(os.getenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS")).expanduser()
    project_dir = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", "schwab_trading_bot").strip() or "schwab_trading_bot"
    return mount_root / project_dir


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
        return int(os.statvfs(path).f_bavail * os.statvfs(path).f_frsize)
    except Exception:
        return None


def _probe_external_storage(external_root: Path) -> dict[str, object]:
    mount_root = Path(os.getenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS")).expanduser()
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


def _maybe_autoprune_external_low_space(project_root: Path, external_root: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "enabled": _env_flag("BOT_LOGS_LOW_SPACE_AUTOPRUNE_ENABLED", "1"),
        "attempted": False,
    }
    pressure_before = _probe_external_storage(external_root)
    payload.update(
        {
            "external_root": str(external_root),
            "external_free_bytes_before": pressure_before.get("external_free_bytes"),
            "external_min_free_bytes": pressure_before.get("external_min_free_bytes"),
            "external_low_space_before": pressure_before.get("external_low_space"),
        }
    )
    if not payload["enabled"]:
        payload["skipped_reason"] = "autoprune_disabled"
        return payload
    if not bool(pressure_before.get("external_low_space", False)):
        payload["skipped_reason"] = "external_not_low_space"
        return payload

    scripts_dir = project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import data_retention_policy as drp

    candidates, details = drp._collect_external_live_sqlite_pressure_rows(
        project_root,
        external_root,
        require_local_fallback=(
            os.getenv("RETENTION_EXTERNAL_LIVE_SQLITE_REQUIRE_LOCAL_FALLBACK", "1").strip() == "1"
        ),
    )
    payload["attempted"] = True
    payload["candidate_count"] = int(len(candidates))
    payload["details"] = details
    if not candidates:
        payload["skipped_reason"] = str(details.get("skipped_reason") or "no_pressure_candidates")
        return payload

    deleted, errors = drp._delete_paths(candidates)
    pressure_after = _probe_external_storage(external_root)
    payload.update(
        {
            "deleted_count": int(deleted),
            "error_count": int(errors),
            "external_free_bytes_after": pressure_after.get("external_free_bytes"),
            "external_low_space_after": pressure_after.get("external_low_space"),
        }
    )
    return payload


def _path_metadata(path: Path) -> dict[str, object]:
    exists = bool(path.exists())
    out: dict[str, object] = {
        "path": str(path),
        "exists": exists,
        "size_bytes": 0,
        "mtime_utc": "",
    }
    if not exists:
        return out

    try:
        stat = path.stat()
        out["size_bytes"] = int(stat.st_size)
        out["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        pass
    return out


def _sqlite_sidecars(path: Path) -> list[str]:
    out: list[str] = []
    for suffix in ("-wal", "-shm"):
        candidate = Path(f"{path}{suffix}")
        if candidate.exists():
            out.append(candidate.name)
    return out


def _default_local_queue_db(project_root: Path, local_root: Path) -> Path:
    override = str(os.getenv("BOT_CHANNEL_QUEUE_DB", "") or "").strip()
    if override:
        return Path(override).expanduser()
    local_override = str(os.getenv("BOT_CHANNEL_QUEUE_LOCAL_ROOT", "") or "").strip()
    if local_override:
        return Path(local_override).expanduser() / "data" / "bot_channel_queue.sqlite3"
    return local_root / "data" / "bot_channel_queue.sqlite3"


def _build_sqlite_skip_report(
    project_root: Path,
    external_root: Path,
    *,
    mode: str,
    active_root: Path,
) -> dict[str, object]:
    local_root = Path(
        os.getenv(
            "BOT_LOGS_LOCAL_FALLBACK_ROOT",
            str(project_root / "local_fallback_storage"),
        )
    ).expanduser()
    queue_db_path = _default_local_queue_db(project_root, local_root)
    try:
        queue_db_realpath = queue_db_path.resolve(strict=False)
    except Exception:
        queue_db_realpath = queue_db_path

    entries: list[dict[str, object]] = []
    tracked = (
        "data/jsonl_link.sqlite3",
        "data/bot_channel_queue.sqlite3",
        "data/snapshot_context.sqlite3",
    )
    local_bytes_total = 0
    active_local_count = 0
    warm_standby_count = 0
    local_present_count = 0

    for rel in tracked:
        local_path = local_root / rel
        external_path = external_root / rel
        local_meta = _path_metadata(local_path)
        external_meta = _path_metadata(external_path)
        local_exists = bool(local_meta.get("exists", False))
        external_exists = bool(external_meta.get("exists", False))
        local_bytes = int(local_meta.get("size_bytes", 0) or 0)
        external_bytes = int(external_meta.get("size_bytes", 0) or 0)
        if local_exists:
            local_present_count += 1
            local_bytes_total += local_bytes

        local_sidecars = _sqlite_sidecars(local_path)
        external_sidecars = _sqlite_sidecars(external_path)
        try:
            local_realpath = local_path.resolve(strict=False)
        except Exception:
            local_realpath = local_path

        active_path = ""
        classification = "not_present"
        reason = "No retained local fallback SQLite file is present for this skip path."

        if local_exists and rel == "data/bot_channel_queue.sqlite3" and queue_db_realpath == local_realpath:
            classification = "active_local_queue"
            reason = (
                "The channel queue DB is currently pinned to the internal fallback root, "
                "so this file is intentionally retained and should not be pruned during external failback."
            )
            active_path = str(local_path)
            active_local_count += 1
        elif local_exists and mode == "external":
            classification = "warm_standby_retained"
            reason = (
                "This SQLite file is intentionally skipped during failback so the internal fallback root "
                "keeps a warm standby copy instead of forcing a live SQLite copy-back."
            )
            active_path = str(external_path if external_exists else active_root / rel)
            warm_standby_count += 1
        elif local_exists:
            classification = "active_local_route"
            reason = "Storage is not currently routed to the external root, so the local fallback copy remains active."
            active_path = str(local_path)
            active_local_count += 1

        entries.append(
            {
                "relative_path": rel,
                "classification": classification,
                "reason": reason,
                "prune_eligible": False,
                "active_path": active_path,
                "local": {
                    **local_meta,
                    "sidecars": local_sidecars,
                },
                "external": {
                    **external_meta,
                    "sidecars": external_sidecars,
                },
                "external_at_least_as_large": bool(external_exists and local_exists and external_bytes >= local_bytes),
            }
        )

    return {
        "mode": str(mode or ""),
        "active_root": str(active_root),
        "queue_db_path": str(queue_db_path),
        "summary": {
            "tracked_entries": len(entries),
            "local_present_count": int(local_present_count),
            "active_local_count": int(active_local_count),
            "warm_standby_count": int(warm_standby_count),
            "prune_eligible_count": 0,
            "local_bytes_total": int(local_bytes_total),
        },
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Re-evaluate storage route and auto-sync local backlog when drive is back.')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from core.storage_router import describe_storage_routing, route_runtime_storage

    lock_path = Path(
        os.getenv(
            "STORAGE_FAILBACK_SYNC_LOCK_PATH",
            str(PROJECT_ROOT / "governance" / "locks" / "storage_failback_sync.lock"),
        )
    )
    lock_fh, lock_owner = _acquire_singleton_lock(lock_path)

    out = PROJECT_ROOT / 'governance' / 'health' / 'storage_failback_sync_latest.json'
    compat = PROJECT_ROOT / 'governance' / 'health' / 'storage_route_status_latest.json'
    out.parent.mkdir(parents=True, exist_ok=True)

    if lock_fh is None:
        payload = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'ok': True,
            'busy': True,
            'lock_path': str(lock_path),
            'lock_owner': lock_owner,
            'skipped_reason': 'lock_busy',
        }
        encoded = json.dumps(payload, ensure_ascii=True, indent=2)
        out.write_text(encoded, encoding='utf-8')
        compat.write_text(encoded, encoding='utf-8')
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(f"[StorageRoute] busy lock_path={lock_path} owner={lock_owner}")
        return 0

    external_root = _external_project_root()
    try:
        low_space_autoprune = _maybe_autoprune_external_low_space(PROJECT_ROOT, external_root)
        routing = route_runtime_storage(PROJECT_ROOT)

        payload = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'mode': routing.mode,
            'active_root': str(routing.active_root),
            'switched_links': list(routing.switched_links),
            'passthrough_paths': list(routing.passthrough_paths),
            'autosync': {
                'copied_files': int(routing.autosync_copied_files),
                'copy_errors': int(routing.autosync_copy_errors),
                'pruned_files': int(routing.autosync_pruned_files),
                'error_details': list(routing.autosync_error_details),
            },
            'split_brain_conflicts': int(routing.split_brain_conflicts),
            'low_space_autoprune': low_space_autoprune,
            'sqlite_skip_report': _build_sqlite_skip_report(
                PROJECT_ROOT,
                external_root,
                mode=routing.mode,
                active_root=routing.active_root,
            ),
            'lock_path': str(lock_path),
        }

        encoded = json.dumps(payload, ensure_ascii=True, indent=2)
        out.write_text(encoded, encoding='utf-8')
        compat.write_text(encoded, encoding='utf-8')

        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(describe_storage_routing(routing))

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


if __name__ == '__main__':
    raise SystemExit(main())
