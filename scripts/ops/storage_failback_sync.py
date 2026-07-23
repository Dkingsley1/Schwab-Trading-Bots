import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.channel_queue import default_queue_db_path
from core.storage_mounts import find_target_external_volume, resolve_external_storage
from scripts.ops.support_maintenance_gate import frozen_health_payload, support_maintenance_freeze_contract


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _external_project_root() -> Path:
    return resolve_external_storage().external_root


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
    resolution = resolve_external_storage()
    mount_root = resolution.mount_root
    target_volume = find_target_external_volume()
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
        "configured_mount_root": str(resolution.configured_mount_root),
        "configured_project_root": str(resolution.configured_project_root) if resolution.configured_project_root else "",
        "candidate_mount_roots": [str(path) for path in resolution.candidate_mount_roots],
        "matched_mount_root": str(resolution.matched_mount_root) if resolution.matched_mount_root else "",
        "match_reason": str(resolution.match_reason),
        "target_volume_device_identifier": str(target_volume.device_identifier) if target_volume else "",
        "target_volume_name": str(target_volume.volume_name) if target_volume else "",
        "target_volume_uuid": str(target_volume.volume_uuid) if target_volume else "",
        "target_volume_mount_point": str(target_volume.mount_point) if target_volume else "",
        "target_volume_present": bool(target_volume is not None),
        "target_volume_mounted": bool(target_volume.is_mounted) if target_volume else False,
        "mount_present": mount_present,
        "external_root_exists": external_root_exists,
        "external_root_writable": external_root_writable,
        "external_free_bytes": external_free_bytes,
        "external_min_free_bytes": int(external_min_free_bytes),
        "external_low_space": external_low_space,
    }


def _support_freeze_bypass_reason(previous_path: Path, external_root: Path) -> str:
    try:
        decoded = json.loads(previous_path.read_text(encoding="utf-8"))
    except Exception:
        decoded = {}
    previous = decoded if isinstance(decoded, dict) else {}
    previous_mode = str(previous.get("certified_mode") or previous.get("mode") or "").strip().lower()
    if previous_mode not in {"external", "external_curated"}:
        return f"previous_route_not_external:{previous_mode or 'unknown'}"
    try:
        if int(float(previous.get("split_brain_conflicts", 0) or 0)) > 0:
            return "previous_split_brain_conflicts"
    except Exception:
        return "previous_split_brain_conflicts_unreadable"

    probe = _probe_external_storage(external_root)
    if not bool(probe.get("mount_present", False)):
        return "external_mount_missing"
    if not bool(probe.get("external_root_exists", False)):
        return "external_root_missing"
    if not bool(probe.get("external_root_writable", False)):
        return "external_root_not_writable"
    return ""


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


def _metadata_mtime_utc(meta: dict[str, object]) -> datetime | None:
    raw = str(meta.get("mtime_utc") or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _sync_bot_logs_finder_shortcuts(project_root: Path) -> dict[str, object]:
    helper = project_root / "scripts" / "ops" / "bot_logs_finder_sync.py"
    if not helper.exists():
        return {
            "attempted": False,
            "ok": False,
            "error": "helper_missing",
        }

    create_desktop_shortcut = os.getenv("BOT_LOGS_FINDER_DESKTOP_SHORTCUTS", "1").strip().lower() in {"1", "true", "yes", "on"}
    cmd = [sys.executable, str(helper), "--json"]
    if not create_desktop_shortcut:
        cmd.append("--no-desktop-shortcut")
    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    payload: dict[str, object] = {}
    for raw in reversed([line.strip() for line in str(proc.stdout or "").splitlines() if line.strip()]):
        try:
            decoded = json.loads(raw)
        except Exception:
            continue
        if isinstance(decoded, dict):
            payload = decoded
            break
    if not payload:
        out_file = project_root / "governance" / "health" / "bot_logs_finder_sync_latest.json"
        try:
            decoded = json.loads(out_file.read_text(encoding="utf-8"))
        except Exception:
            decoded = {}
        payload = decoded if isinstance(decoded, dict) else {}
    return {
        "attempted": True,
        "ok": int(proc.returncode) == 0 and bool(payload.get("ok", False)),
        "rc": int(proc.returncode),
        "desktop_shortcut_enabled": create_desktop_shortcut,
        "payload": payload,
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-8:]),
    }


def _sqlite_sidecars(path: Path) -> list[str]:
    out: list[str] = []
    for suffix in ("-wal", "-shm"):
        candidate = Path(f"{path}{suffix}")
        if candidate.exists():
            out.append(candidate.name)
    return out


def _default_local_queue_db(project_root: Path, local_root: Path) -> Path:
    del local_root
    return Path(default_queue_db_path(project_root)).expanduser()


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
    active_external_count = 0
    warm_standby_count = 0
    local_present_count = 0
    external_ready_count = 0
    verified_count = 0
    curated_standby_count = 0
    active_passthrough_count = 0
    verification_mismatches: list[str] = []

    for rel in tracked:
        repo_path = project_root / rel
        local_path = local_root / rel
        external_path = external_root / rel
        repo_meta = _path_metadata(repo_path)
        local_meta = _path_metadata(local_path)
        external_meta = _path_metadata(external_path)
        repo_exists = bool(repo_meta.get("exists", False))
        local_exists = bool(local_meta.get("exists", False))
        external_exists = bool(external_meta.get("exists", False))
        repo_bytes = int(repo_meta.get("size_bytes", 0) or 0)
        local_bytes = int(local_meta.get("size_bytes", 0) or 0)
        external_bytes = int(external_meta.get("size_bytes", 0) or 0)
        if local_exists:
            local_present_count += 1
            local_bytes_total += local_bytes

        repo_sidecars = _sqlite_sidecars(repo_path)
        local_sidecars = _sqlite_sidecars(local_path)
        external_sidecars = _sqlite_sidecars(external_path)
        try:
            repo_realpath = repo_path.resolve(strict=False)
        except Exception:
            repo_realpath = repo_path
        try:
            local_realpath = local_path.resolve(strict=False)
        except Exception:
            local_realpath = local_path
        try:
            external_realpath = external_path.resolve(strict=False)
        except Exception:
            external_realpath = external_path
        repo_is_external_route = bool(
            (repo_path.is_symlink() or repo_exists)
            and external_exists
            and repo_realpath == external_realpath
        )
        repo_is_local_route = bool(
            (repo_path.is_symlink() or repo_exists)
            and local_exists
            and repo_realpath == local_realpath
        )

        active_path = ""
        classification = "not_present"
        reason = "No retained local fallback SQLite file is present for this skip path."

        if repo_is_local_route and local_exists:
            classification = "active_local_route"
            reason = (
                "The repo SQLite link is routed to local_fallback_storage, "
                "so the local fallback copy is the active database for this path."
            )
            active_path = str(repo_path)
            active_local_count += 1
        elif repo_is_external_route and external_exists:
            classification = "active_external_route"
            reason = (
                "The repo SQLite link is routed to the external BOT_LOGS root, "
                "so the external copy is the active database for this path."
            )
            active_path = str(repo_path)
            active_external_count += 1
        elif (
            rel == "data/bot_channel_queue.sqlite3"
            and repo_exists
            and queue_db_realpath == repo_realpath
            and not repo_is_external_route
        ):
            classification = "active_repo_queue_passthrough"
            reason = (
                "The channel queue DB is currently pinned to the repo data passthrough route, "
                "so this active live queue is verified in place instead of requiring an external copy."
            )
            active_path = str(repo_path)
            active_passthrough_count += 1
        elif local_exists and rel == "data/bot_channel_queue.sqlite3" and queue_db_realpath == local_realpath:
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

        verification_state = "missing_external_copy"
        verification_reason = "The external route does not currently have a verified SQLite copy for this tracked path."
        if classification == "active_local_route" and local_bytes > 0:
            verification_state = "active_local_ready"
            verification_reason = (
                "The active route is local fallback and the repo link resolves to a present local SQLite copy."
            )
            external_ready_count += 1
            verified_count += 1
        elif classification == "active_repo_queue_passthrough" and repo_bytes > 0:
            verification_state = "active_passthrough"
            verification_reason = (
                "The active queue DB is intentionally routed through the repo passthrough data path; "
                "the failback route is considered ready because the live queue is present and not copy-back eligible."
            )
            external_ready_count += 1
            verified_count += 1
        elif classification == "active_external_route" and external_exists and external_bytes > 0:
            if not local_exists or external_bytes >= local_bytes:
                verification_state = "verified"
                verification_reason = "The active external route carries a present SQLite copy that is at least as large as the retained local copy."
                external_ready_count += 1
                verified_count += 1
            elif (
                _metadata_mtime_utc(external_meta) is not None
                and _metadata_mtime_utc(local_meta) is not None
                and _metadata_mtime_utc(external_meta) >= _metadata_mtime_utc(local_meta)
            ):
                verification_state = "active_external_newer_than_standby"
                verification_reason = (
                    "The active external route is newer than the retained local fallback copy; "
                    "SQLite file size alone is not treated as lagging evidence for a live routed DB."
                )
                external_ready_count += 1
                verified_count += 1
            else:
                verification_state = "lagging_external_copy"
                verification_reason = "The active external route copy is present but smaller than the retained local fallback copy."
                verification_mismatches.append(rel)
        elif str(mode or "") == "external" and classification in {"warm_standby_retained", "active_local_queue"} and (
            not external_exists or external_bytes <= 0 or (local_exists and external_bytes < local_bytes)
        ):
            verification_state = "curated_standby"
            verification_reason = (
                "This SQLite path is intentionally retained as a local warm standby while the rebuilt external BOT_LOGS route "
                "is certified from curated artifacts instead of a full live SQLite copy-back."
            )
            external_ready_count += 1
            curated_standby_count += 1
        elif external_exists and external_bytes > 0:
            if not local_exists or external_bytes >= local_bytes:
                verification_state = "verified"
                verification_reason = "The external route carries a present SQLite copy that is at least as large as the retained local copy."
                external_ready_count += 1
                verified_count += 1
            else:
                verification_state = "lagging_external_copy"
                verification_reason = "The external route copy is present but smaller than the retained local fallback copy."
                verification_mismatches.append(rel)
        else:
            verification_mismatches.append(rel)

        entries.append(
            {
                "relative_path": rel,
                "classification": classification,
                "reason": reason,
                "prune_eligible": False,
                "active_path": active_path,
                "route_verification": {
                    "state": verification_state,
                    "reason": verification_reason,
                },
                "active_repo": {
                    **repo_meta,
                    "sidecars": repo_sidecars,
                },
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

    verification_state = "ready"
    if verification_mismatches:
        verification_state = "warning" if external_ready_count > 0 else "blocked"
    elif curated_standby_count > 0:
        verification_state = "curated_ready"
    elif active_local_count > 0 and str(mode or "").startswith("local_fallback"):
        verification_state = "active_local_ready"

    certified_mode = str(mode or "")
    if str(mode or "") == "external" and verification_state == "curated_ready":
        certified_mode = "external_curated"

    return {
        "mode": str(mode or ""),
        "certified_mode": certified_mode,
        "active_root": str(active_root),
        "queue_db_path": str(queue_db_path),
        "summary": {
            "tracked_entries": len(entries),
            "local_present_count": int(local_present_count),
            "active_local_count": int(active_local_count),
            "active_external_count": int(active_external_count),
            "active_passthrough_count": int(active_passthrough_count),
            "warm_standby_count": int(warm_standby_count),
            "external_ready_count": int(external_ready_count),
            "verified_count": int(verified_count),
            "curated_standby_count": int(curated_standby_count),
            "verification_mismatch_count": len(verification_mismatches),
            "verification_state": verification_state,
            "prune_eligible_count": 0,
            "local_bytes_total": int(local_bytes_total),
        },
        "route_verification": {
            "verification_state": verification_state,
            "certified_mode": certified_mode,
            "ready_count": int(external_ready_count),
            "verified_count": int(verified_count),
            "curated_standby_count": int(curated_standby_count),
            "tracked_count": len(entries),
            "coverage_ratio": round(external_ready_count / max(len(entries), 1), 6),
            "mismatches": verification_mismatches,
        },
        "entries": entries,
    }


def _refresh_frozen_sqlite_skip_report(payload: dict[str, object], external_root: Path) -> dict[str, object]:
    refreshed = dict(payload)
    mode = str(refreshed.get("mode") or refreshed.get("certified_mode") or "")
    if not mode or mode == "support_maintenance_frozen":
        return refreshed

    active_root_raw = str(refreshed.get("active_root") or "").strip()
    if active_root_raw:
        active_root = Path(active_root_raw).expanduser()
    elif mode.startswith("local_fallback"):
        active_root = Path(
            os.getenv(
                "BOT_LOGS_LOCAL_FALLBACK_ROOT",
                str(PROJECT_ROOT / "local_fallback_storage"),
            )
        ).expanduser()
    else:
        active_root = external_root

    try:
        sqlite_skip_report = _build_sqlite_skip_report(
            PROJECT_ROOT,
            external_root,
            mode=mode,
            active_root=active_root,
        )
    except Exception as exc:
        refreshed["frozen_lightweight_refresh"] = {
            "sqlite_skip_report": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        return refreshed

    refreshed["sqlite_skip_report"] = sqlite_skip_report
    refreshed["certified_mode"] = str(
        sqlite_skip_report.get("certified_mode") or refreshed.get("certified_mode") or mode
    )
    route_verification = sqlite_skip_report.get("route_verification")
    if isinstance(route_verification, dict):
        refreshed["route_verification"] = route_verification
    refreshed["frozen_lightweight_refresh"] = {
        "sqlite_skip_report": True,
        "policy": "refresh_sqlite_route_metadata_while_skipping_heavy_failback_work",
    }
    return refreshed


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
        freeze_contract = support_maintenance_freeze_contract(PROJECT_ROOT, "storage_failback_sync")
        freeze_bypass_reason = _support_freeze_bypass_reason(out, external_root)
        if bool(freeze_contract.get("active", False)) and not freeze_bypass_reason:
            payload = frozen_health_payload(out, freeze_contract)
            payload.setdefault("mode", "support_maintenance_frozen")
            payload.setdefault("certified_mode", payload.get("mode", "support_maintenance_frozen"))
            payload.setdefault("split_brain_conflicts", 0)
            payload = _refresh_frozen_sqlite_skip_report(payload, external_root)
            encoded = json.dumps(payload, ensure_ascii=True, indent=2)
            out.write_text(encoded, encoding='utf-8')
            compat.write_text(encoded, encoding='utf-8')
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("[StorageRoute] skipped support_maintenance_frozen_for_mac_fluidity")
            return 0

        low_space_autoprune = _maybe_autoprune_external_low_space(PROJECT_ROOT, external_root)
        routing = route_runtime_storage(PROJECT_ROOT)

        payload = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'mode': routing.mode,
            'certified_mode': routing.mode,
            'active_root': str(routing.active_root),
            'switched_links': list(routing.switched_links),
            'passthrough_paths': list(routing.passthrough_paths),
            'autosync': {
                'copied_files': int(routing.autosync_copied_files),
                'copy_errors': int(routing.autosync_copy_errors),
                'pruned_files': int(routing.autosync_pruned_files),
                'error_details': list(routing.autosync_error_details),
                'skipped_reason': str(getattr(routing, 'autosync_skipped_reason', '') or ''),
                'free_bytes': getattr(routing, 'autosync_free_bytes', None),
                'min_free_bytes': int(getattr(routing, 'autosync_min_free_bytes', 0) or 0),
            },
            'split_brain_conflicts': int(routing.split_brain_conflicts),
            'low_space_autoprune': low_space_autoprune,
            'sqlite_skip_report': _build_sqlite_skip_report(
                PROJECT_ROOT,
                external_root,
                mode=routing.mode,
                active_root=routing.active_root,
            ),
            'finder_sync': _sync_bot_logs_finder_shortcuts(PROJECT_ROOT),
            'lock_path': str(lock_path),
        }
        if bool(freeze_contract.get("active", False)) and freeze_bypass_reason:
            payload["support_maintenance_freeze_bypassed"] = True
            payload["support_maintenance_freeze_bypass_reason"] = freeze_bypass_reason
            payload["support_maintenance_freeze_contract"] = freeze_contract
        sqlite_skip_report = payload.get('sqlite_skip_report') if isinstance(payload.get('sqlite_skip_report'), dict) else {}
        payload['certified_mode'] = str(sqlite_skip_report.get('certified_mode') or payload.get('certified_mode') or payload.get('mode') or "")
        payload['route_verification'] = sqlite_skip_report.get('route_verification') if isinstance(sqlite_skip_report.get('route_verification'), dict) else {}

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
