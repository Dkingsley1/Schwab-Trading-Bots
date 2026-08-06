#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
    from core.storage_mounts import find_target_external_volume, resolve_external_storage
    from core.storage_target_override import DEFAULT_STORAGE_TARGET_OVERRIDE_PATH, write_storage_target_override
    from scripts.ops import writer_cycle_coordinator as writer_src
    from scripts.ops.long_runtime_common import payload_age_minutes, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_python import resolve_runtime_python
    from core.storage_mounts import find_target_external_volume, resolve_external_storage
    from core.storage_target_override import DEFAULT_STORAGE_TARGET_OVERRIDE_PATH, write_storage_target_override
    from scripts.ops import writer_cycle_coordinator as writer_src
    from scripts.ops.long_runtime_common import payload_age_minutes, write_payload


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "storage_disaster_recovery_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "storage_disaster_recovery_state.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "storage_disaster_recovery.lock"
DEFAULT_RECOVERY_ROOT = Path.home() / "Documents" / "BOT_LOGS_recovery_auto"
DEFAULT_LOCAL_ROOT = PROJECT_ROOT / "local_fallback_storage"
DEFAULT_ROUTE_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.storage_override"
DEFAULT_TARGET_OVERRIDE_PATH = DEFAULT_STORAGE_TARGET_OVERRIDE_PATH
IMPORTANT_DIRS: tuple[str, ...] = (
    "governance",
    "logs",
    "exports",
    "decision_explanations",
    "decisions",
    "models",
)
IMPORTANT_FILES: tuple[str, ...] = (
    "data/snapshot_context.sqlite3",
    "data/snapshot_context.sqlite3-wal",
    "data/snapshot_context.sqlite3-shm",
)
GIB = float(1024**3)
LOCAL_MODES = {"local_fallback", "local_fallback_split_brain"}
EXTERNAL_CERTIFIED_MODES = {"external", "external_curated"}
EXTERNAL_RECOVERY_MODES = EXTERNAL_CERTIFIED_MODES | {"external_available_unverified", "unknown"}
TRACKED_SQLITE_ROUTES: tuple[str, ...] = (
    "data/jsonl_link.sqlite3",
    "data/bot_channel_queue.sqlite3",
    "data/snapshot_context.sqlite3",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _project_dir_from_env() -> str:
    return str(os.getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", "schwab_trading_bot") or "schwab_trading_bot").strip() or "schwab_trading_bot"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_payload(path, payload)


def _parse_json_output(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(text or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _run_command(cmd: list[str], *, cwd: Path, timeout_sec: int = 180) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        rc = int(proc.returncode)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        rc = 124
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        timed_out = True

    duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
    return {
        "cmd": list(cmd),
        "rc": rc,
        "timed_out": timed_out,
        "duration_ms": duration_ms,
        "payload": _parse_json_output(stdout),
        "stdout_tail": "\n".join(stdout.splitlines()[-12:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-12:]),
    }


def _probe_storage() -> dict[str, Any]:
    prefer_external = str(os.getenv("BOT_LOGS_PREFER_EXTERNAL", "1") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if not prefer_external:
        mount_root = Path(os.getenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS")).expanduser()
        configured_root = str(os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "") or "").strip()
        project_dir = str(os.getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", "schwab_trading_bot") or "schwab_trading_bot").strip()
        external_root = Path(configured_root).expanduser() if configured_root else mount_root / project_dir
        return {
            "mount_root": str(mount_root),
            "external_root": str(external_root),
            "configured_mount_root": str(mount_root),
            "configured_project_root": str(external_root),
            "candidate_mount_roots": [str(mount_root)],
            "matched_mount_root": "",
            "match_reason": "external_io_probe_skipped_local_hot_storage_policy",
            "target_volume_device_identifier": str(os.getenv("BOT_LOGS_EXTERNAL_DISK_IDENTIFIER", "") or ""),
            "target_volume_name": str(os.getenv("BOT_LOGS_EXTERNAL_VOLUME_NAME", "BOT_LOGS") or "BOT_LOGS"),
            "target_volume_uuid": str(os.getenv("BOT_LOGS_EXTERNAL_VOLUME_UUID", "") or ""),
            "target_volume_mount_point": "",
            "target_volume_present": False,
            "target_volume_mounted": False,
            "mount_present": False,
            "external_root_exists": False,
            "external_root_writable": False,
            "external_available": False,
            "external_unavailable_reason": "cold_archive_only_local_hot_storage_policy",
            "external_required_for_hot_path": False,
            "hot_storage_available": True,
            "probe_skipped_external_io": True,
        }
    resolution = resolve_external_storage()
    mount_root = resolution.mount_root
    external_root = resolution.external_root
    target_volume = find_target_external_volume()
    mount_present = bool(mount_root.exists() and mount_root.is_dir())
    external_root_exists = bool(external_root.exists() and external_root.is_dir())
    external_root_writable = bool(external_root_exists and os.access(external_root, os.W_OK))

    if not mount_present:
        if target_volume is not None and not target_volume.is_mounted:
            unavailable_reason = "volume_unmounted"
        else:
            unavailable_reason = "mount_missing"
    elif target_volume is not None and not target_volume.is_mounted:
        unavailable_reason = "volume_unmounted"
    elif not external_root_exists:
        unavailable_reason = "root_missing"
    elif not external_root_writable:
        unavailable_reason = "not_writable"
    else:
        unavailable_reason = "ok"

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
        "external_available": bool(mount_present and external_root_exists and external_root_writable),
        "external_unavailable_reason": unavailable_reason,
    }


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        Path(os.path.abspath(str(path.expanduser()))).relative_to(Path(os.path.abspath(str(root.expanduser()))))
    except ValueError:
        return False
    return True


def _physical_sqlite_route_mode(project_root: Path, probe: dict[str, Any] | None = None) -> str:
    live_probe = probe if isinstance(probe, dict) else _probe_storage()
    local_root = Path(os.getenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(project_root / "local_fallback_storage"))).expanduser()
    external_text = str(live_probe.get("external_root") or "").strip()
    external_root = Path(external_text).expanduser() if external_text else None
    families: list[str] = []
    for relative_path in TRACKED_SQLITE_ROUTES:
        route = project_root / relative_path
        if not route.is_symlink():
            return ""
        try:
            raw_target = Path(os.readlink(route))
        except OSError:
            return ""
        target = raw_target if raw_target.is_absolute() else route.parent / raw_target
        if _path_is_within(target, local_root):
            families.append("local")
        elif external_root is not None and _path_is_within(target, external_root):
            families.append("external")
        else:
            return ""
    if families and all(family == "local" for family in families):
        return "local_fallback"
    if families and all(family == "external" for family in families):
        return "external"
    if families:
        return "local_fallback_split_brain"
    return ""


def _current_storage_mode(project_root: Path, probe: dict[str, Any] | None = None) -> str:
    physical_mode = _physical_sqlite_route_mode(project_root, probe)
    if physical_mode:
        return physical_mode
    failback = _load_json(project_root / "governance" / "health" / "storage_failback_sync_latest.json")
    mount_guard = _load_json(project_root / "governance" / "health" / "storage_mount_guard_latest.json")
    mode = str(failback.get("certified_mode") or failback.get("mode") or mount_guard.get("storage_mode") or "").strip()
    if mode:
        return mode
    live_probe = probe if isinstance(probe, dict) else _probe_storage()
    if bool(live_probe.get("external_available", False)):
        return "external_available_unverified"
    return "unknown"


def _route_policy(project_root: Path) -> dict[str, Any]:
    override_path = project_root / "config" / ".env.storage_override"
    try:
        override_body = override_path.read_text(encoding="utf-8")
    except Exception:
        override_body = ""
    local_pinned = any(
        line.strip().lower() in {
            "bot_logs_prefer_external=0",
            "bot_logs_prefer_external=false",
            "bot_logs_prefer_external=no",
            "bot_logs_prefer_external=off",
        }
        for line in override_body.splitlines()
    )
    return {
        "override_path": str(override_path),
        "local_route_pinned": local_pinned,
        "automatic_external_failback_enabled": _env_flag("BOT_LOGS_RECOVERY_AUTO_FAILBACK_EXTERNAL", "0"),
        "policy": "preserve_explicit_local_route_and_keep_external_as_standby",
    }


def _recovery_selected_paths(local_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel_path, kind in (
        *((rel_path, "dir") for rel_path in IMPORTANT_DIRS),
        *((rel_path, "file") for rel_path in IMPORTANT_FILES),
    ):
        src = local_root / rel_path
        exists = bool(src.exists() and (src.is_dir() if kind == "dir" else src.is_file()))
        try:
            resolved = src.resolve(strict=exists)
        except (OSError, RuntimeError):
            resolved = src.absolute()
        local_physical_source = bool(exists and _path_is_within(resolved, local_root.resolve()))
        rows.append(
            {
                "rel_path": rel_path,
                "kind": kind,
                "source": str(src),
                "resolved_source": str(resolved),
                "is_symlink": src.is_symlink(),
                "exists": exists,
                "eligible": local_physical_source,
                "skip_reason": "" if local_physical_source else "missing" if not exists else "outside_local_fallback_root",
            }
        )
    return rows


def _selected_size_bytes(source_root: Path, selected_rows: list[dict[str, Any]]) -> int:
    total = 0
    seen: set[tuple[int, int]] = set()
    physical_root = source_root.resolve()
    for row in selected_rows:
        if not bool(row.get("eligible", row.get("exists", False))):
            continue
        src = source_root / str(row.get("rel_path") or "")
        candidates = src.rglob("*") if str(row.get("kind") or "") == "dir" else (src,)
        for candidate in candidates:
            try:
                if not candidate.is_file():
                    continue
                if not _path_is_within(candidate.resolve(strict=True), physical_root):
                    continue
                stat = candidate.stat()
            except (FileNotFoundError, OSError):
                continue
            identity = (int(stat.st_dev), int(stat.st_ino))
            if identity in seen:
                continue
            seen.add(identity)
            total += max(int(stat.st_size), 0)
    return total


def _cleanup_snapshot_workspace(recovery_root: Path, *, apply: bool) -> dict[str, Any]:
    if apply:
        recovery_root.mkdir(parents=True, exist_ok=True)
    staging_roots = sorted(recovery_root.glob(".latest_staging_*"))
    previous_root = recovery_root / ".latest_previous"
    latest_root = recovery_root / "latest"
    previous_present = previous_root.exists()
    deleted_bytes = 0
    deleted_paths: list[str] = []
    errors: list[str] = []

    if apply and previous_root.exists() and not latest_root.exists():
        try:
            previous_root.rename(latest_root)
        except Exception as exc:
            errors.append(f"{previous_root}:{type(exc).__name__}:{exc}")
    elif apply and previous_root.exists():
        try:
            shutil.rmtree(previous_root)
            deleted_paths.append(str(previous_root))
        except Exception as exc:
            errors.append(f"{previous_root}:{type(exc).__name__}:{exc}")

    for path in staging_roots:
        try:
            path_bytes = _selected_size_bytes(path, [{"rel_path": ".", "kind": "dir", "eligible": True}])
            if apply:
                shutil.rmtree(path)
                deleted_paths.append(str(path))
                deleted_bytes += path_bytes
        except Exception as exc:
            errors.append(f"{path}:{type(exc).__name__}:{exc}")

    return {
        "apply_requested": bool(apply),
        "candidate_count": len(staging_roots) + int(previous_present),
        "deleted_count": len(deleted_paths),
        "deleted_bytes": int(deleted_bytes),
        "deleted_gb": round(float(deleted_bytes) / GIB, 3),
        "deleted_paths": deleted_paths,
        "error_count": len(errors),
        "errors": errors,
        "ok": not errors,
    }


def _writer_quiet_point(
    project_root: Path,
    *,
    apply: bool,
    poll_seconds: float | None = None,
    wait_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    before = writer_src.writer_state_snapshot(project_root)
    poll = float(poll_seconds if poll_seconds is not None else os.getenv("BOT_LOGS_RECOVERY_QUIET_POLL_SECONDS", "2.0"))
    wait_timeout = float(
        wait_timeout_seconds if wait_timeout_seconds is not None else os.getenv("BOT_LOGS_RECOVERY_QUIET_WAIT_SECONDS", "90.0")
    )
    payload: dict[str, Any] = {
        "attempted": False,
        "ok": not bool(before.get("active", False)),
        "writer_state_before": before,
        "wait_for_writer": {
            "requested": False,
            "completed": not bool(before.get("active", False)),
            "timed_out": False,
            "attempts": 0,
            "waited_seconds": 0.0,
            "final_state": before,
        },
    }
    if not apply:
        payload["skipped_reason"] = "apply_disabled"
        return payload
    payload["attempted"] = True
    if not bool(before.get("active", False)):
        return payload
    wait = writer_src._wait_for_writer_idle(
        project_root,
        poll_seconds=max(float(poll), 0.1),
        wait_timeout_seconds=max(float(wait_timeout), 1.0),
    )
    final_state = wait.get("final_state") if isinstance(wait.get("final_state"), dict) else before
    payload["wait_for_writer"] = wait
    payload["writer_state_after_wait"] = final_state
    payload["ok"] = bool(wait.get("completed", False)) and not bool(final_state.get("active", False))
    if not payload["ok"]:
        payload["skipped_reason"] = "writer_not_quiet"
    return payload


def _mount_target_volume(probe: dict[str, Any], *, apply: bool, state: dict[str, Any], cooldown_seconds: float) -> dict[str, Any]:
    attempted = False
    target_device = str(probe.get("target_volume_device_identifier") or "").strip()
    last_epoch = _safe_float(state.get("last_mount_attempt_epoch"), 0.0)
    cooldown_remaining = max(last_epoch + max(float(cooldown_seconds), 0.0) - time.time(), 0.0)
    should_attempt = (
        bool(apply)
        and not bool(probe.get("external_available", False))
        and bool(probe.get("target_volume_present", False))
        and not bool(probe.get("target_volume_mounted", False))
        and bool(target_device)
        and cooldown_remaining <= 0.0
    )
    payload: dict[str, Any] = {
        "attempted": False,
        "ok": False,
        "device_identifier": target_device,
        "cooldown_remaining_seconds": round(float(cooldown_remaining), 3),
    }
    if not should_attempt:
        payload["skipped_reason"] = (
            "apply_disabled"
            if not apply
            else "mount_cooldown"
            if cooldown_remaining > 0.0
            else "mount_not_needed"
        )
        return payload

    attempted = True
    state["last_mount_attempt_epoch"] = time.time()
    result = _run_command(["/usr/sbin/diskutil", "mount", target_device], cwd=PROJECT_ROOT, timeout_sec=60)
    payload.update(
        {
            "attempted": attempted,
            "ok": int(result.get("rc", 1)) == 0,
            "rc": int(result.get("rc", 1)),
            "duration_ms": float(result.get("duration_ms", 0.0) or 0.0),
            "stdout_tail": str(result.get("stdout_tail") or ""),
            "stderr_tail": str(result.get("stderr_tail") or ""),
        }
    )
    return payload


def _sync_storage_target_override(project_root: Path, probe: dict[str, Any], *, apply: bool) -> dict[str, Any]:
    mount_root = str(probe.get("mount_root") or probe.get("configured_mount_root") or "").strip()
    override_path = project_root / "config" / ".env.storage_target_override"
    payload = {
        "attempted": False,
        "ok": False,
        "path": str(override_path),
        "mount_root": mount_root,
    }
    if not mount_root:
        payload["skipped_reason"] = "missing_mount_root"
        return payload
    if not apply:
        payload["skipped_reason"] = "apply_disabled"
        return payload
    result = write_storage_target_override(
        mount_root=mount_root,
        project_dir=_project_dir_from_env(),
        mount_candidates=tuple(str(item) for item in list(probe.get("candidate_mount_roots") or []) if str(item).strip()),
        volume_name=str(probe.get("target_volume_name") or "").strip(),
        volume_uuid=str(probe.get("target_volume_uuid") or "").strip(),
        disk_identifier=str(probe.get("target_volume_device_identifier") or "").strip(),
        override_path=override_path,
    )
    return {
        "attempted": True,
        "ok": True,
        **result,
    }


def _switch_storage_mode(project_root: Path, target_mode: str, *, apply: bool) -> dict[str, Any]:
    if not apply:
        return {
            "attempted": False,
            "ok": False,
            "target_mode": target_mode,
            "skipped_reason": "apply_disabled",
        }
    result = _run_command(
        [
            str(PY),
            str(project_root / "scripts" / "ops" / "storage_switch_orchestrator.py"),
            "--target-mode",
            str(target_mode),
        ],
        cwd=project_root,
        timeout_sec=240,
    )
    return {
        "attempted": True,
        "ok": int(result.get("rc", 1)) == 0,
        "target_mode": target_mode,
        "rc": int(result.get("rc", 1)),
        "duration_ms": float(result.get("duration_ms", 0.0) or 0.0),
        "payload": result.get("payload") if isinstance(result.get("payload"), dict) else {},
        "stdout_tail": str(result.get("stdout_tail") or ""),
        "stderr_tail": str(result.get("stderr_tail") or ""),
    }


def _sync_finder_shortcuts(project_root: Path, *, apply: bool) -> dict[str, Any]:
    if not apply:
        return {
            "attempted": False,
            "ok": False,
            "skipped_reason": "apply_disabled",
        }
    result = _run_command(
        [
            str(PY),
            str(project_root / "scripts" / "ops" / "bot_logs_finder_sync.py"),
            "--json",
        ],
        cwd=project_root,
        timeout_sec=60,
    )
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    return {
        "attempted": True,
        "ok": int(result.get("rc", 1)) == 0 and bool(payload.get("ok", False)),
        "rc": int(result.get("rc", 1)),
        "payload": payload,
        "stdout_tail": str(result.get("stdout_tail") or ""),
        "stderr_tail": str(result.get("stderr_tail") or ""),
    }


def _copy_file_transactional(
    src: Path,
    dst: Path,
    *,
    compare_existing: bool,
    max_attempts: int = 3,
) -> tuple[str, str]:
    for attempt in range(max(int(max_attempts), 1)):
        try:
            if compare_existing and dst.exists():
                src_stat = src.stat()
                dst_stat = dst.stat()
                if dst_stat.st_size >= src_stat.st_size and dst_stat.st_mtime >= src_stat.st_mtime:
                    return "skipped", ""
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.parent / f".{dst.name}.storage_recovery_tmp"
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            return "copied", ""
        except FileNotFoundError as exc:
            if attempt + 1 < max(int(max_attempts), 1):
                time.sleep(0.1)
                continue
            return "transient_missing", f"{src}:{type(exc).__name__}:{exc}"
        except Exception as exc:
            return "error", f"{src}:{type(exc).__name__}:{exc}"
    return "error", f"{src}:copy_attempts_exhausted"


def _stage_selected_paths(
    source_root: Path,
    target_root: Path,
    selected_rows: list[dict[str, Any]],
    *,
    compare_existing: bool,
) -> dict[str, Any]:
    copied_paths: list[str] = []
    skipped_paths: list[str] = []
    transient_missing: list[str] = []
    errors: list[str] = []
    unsafe_skipped_paths: list[str] = []
    physical_source_root = source_root.resolve()
    for row in selected_rows:
        rel_path = str(row.get("rel_path") or "").strip()
        kind = str(row.get("kind") or "").strip()
        if not rel_path:
            continue
        src = source_root / rel_path
        dst = target_root / rel_path
        if kind == "dir":
            if not src.exists() or not src.is_dir():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for child in src.rglob("*"):
                if not child.is_file():
                    continue
                rel_child = child.relative_to(source_root)
                try:
                    child_is_local = _path_is_within(child.resolve(strict=True), physical_source_root)
                except (FileNotFoundError, OSError, RuntimeError):
                    child_is_local = False
                if not child_is_local:
                    unsafe_skipped_paths.append(str(rel_child))
                    continue
                state, detail = _copy_file_transactional(child, target_root / rel_child, compare_existing=compare_existing)
                if state == "copied":
                    copied_paths.append(str(rel_child))
                elif state == "skipped":
                    skipped_paths.append(str(rel_child))
                elif state == "transient_missing":
                    transient_missing.append(detail)
                else:
                    errors.append(detail)
            continue
        try:
            src_is_local = _path_is_within(src.resolve(strict=True), physical_source_root)
        except (FileNotFoundError, OSError, RuntimeError):
            src_is_local = False
        if not src_is_local:
            unsafe_skipped_paths.append(rel_path)
            continue
        state, detail = _copy_file_transactional(src, dst, compare_existing=compare_existing)
        if state == "copied":
            copied_paths.append(rel_path)
        elif state == "skipped":
            skipped_paths.append(rel_path)
        elif state == "transient_missing":
            transient_missing.append(detail)
        else:
            errors.append(detail)
    return {
        "copied_paths": copied_paths,
        "skipped_paths": skipped_paths,
        "transient_missing": transient_missing,
        "errors": errors,
        "unsafe_skipped_paths": unsafe_skipped_paths,
    }


def _promote_latest_snapshot(staging_root: Path, latest_root: Path) -> None:
    backup_root = latest_root.parent / ".latest_previous"
    if backup_root.exists():
        shutil.rmtree(backup_root)
    if latest_root.exists():
        latest_root.rename(backup_root)
    staging_root.rename(latest_root)
    if backup_root.exists():
        shutil.rmtree(backup_root)


def _backup_sqlite_online(src: Path, dst: Path) -> tuple[bool, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.storage_recovery_tmp"
    try:
        if tmp.exists():
            tmp.unlink()
        with sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=5.0) as source:
            with sqlite3.connect(tmp, timeout=5.0) as target:
                source.backup(target)
                integrity = str(target.execute("PRAGMA integrity_check").fetchone()[0]).strip().lower()
                if integrity != "ok":
                    raise sqlite3.DatabaseError(f"integrity_check={integrity}")
        os.replace(tmp, dst)
        return True, ""
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False, f"{src}:{type(exc).__name__}:{exc}"


def _take_curated_snapshot(
    local_root: Path,
    recovery_root: Path,
    *,
    apply: bool,
    state: dict[str, Any],
    cooldown_seconds: float,
    project_root: Path = PROJECT_ROOT,
    require_writer_quiet: bool = True,
) -> dict[str, Any]:
    selected = _recovery_selected_paths(local_root)
    available = [row for row in selected if bool(row.get("eligible", row.get("exists", False)))]
    last_epoch = _safe_float(state.get("last_snapshot_epoch"), 0.0)
    latest_root = recovery_root / "latest"
    cooldown_remaining = max(last_epoch + max(float(cooldown_seconds), 0.0) - time.time(), 0.0)
    workspace_cleanup = _cleanup_snapshot_workspace(recovery_root, apply=apply)

    payload: dict[str, Any] = {
        "attempted": False,
        "ok": False,
        "recovery_root": str(recovery_root),
        "latest_snapshot_root": str(latest_root),
        "selected_paths": selected,
        "available_path_count": int(len(available)),
        "cooldown_remaining_seconds": round(float(cooldown_remaining), 3),
        "workspace_cleanup": workspace_cleanup,
        "snapshot_mode": "writer_quiet" if require_writer_quiet else "online_sqlite_backup",
    }
    if not apply:
        payload["skipped_reason"] = "apply_disabled"
        return payload
    if not available:
        payload["skipped_reason"] = "no_curated_paths_available"
        return payload
    if not bool(workspace_cleanup.get("ok", False)):
        payload["skipped_reason"] = "snapshot_workspace_cleanup_failed"
        return payload
    if latest_root.exists() and cooldown_remaining > 0.0:
        payload["ok"] = True
        payload["skipped_reason"] = "snapshot_cooldown_active"
        return payload

    quiet_point = {"attempted": False, "ok": True, "skipped_reason": "online_snapshot_mode"}
    if require_writer_quiet:
        quiet_point = _writer_quiet_point(project_root, apply=apply)
        if not bool(quiet_point.get("ok", False)):
            payload["quiet_point"] = quiet_point
            payload["skipped_reason"] = str(quiet_point.get("skipped_reason") or "writer_not_quiet")
            return payload
    payload["quiet_point"] = quiet_point

    estimated_bytes = _selected_size_bytes(local_root, available)
    min_free_after_gb = max(
        _safe_float(
            os.getenv("BOT_LOGS_RECOVERY_MIN_FREE_AFTER_SNAPSHOT_GB"),
            _safe_float(os.getenv("BOT_LOCAL_STORAGE_TARGET_FREE_GB"), 64.0),
        ),
        0.0,
    )
    headroom_ratio = max(_safe_float(os.getenv("BOT_LOGS_RECOVERY_SNAPSHOT_HEADROOM_RATIO"), 1.10), 1.0)
    try:
        usage = shutil.disk_usage(recovery_root)
        free_bytes = int(usage.free)
        capacity_known = True
        capacity_error = ""
    except Exception as exc:
        free_bytes = 0
        capacity_known = False
        capacity_error = f"{type(exc).__name__}:{exc}"
    required_free_bytes = int((min_free_after_gb * GIB) + (estimated_bytes * headroom_ratio))
    capacity = {
        "known": capacity_known,
        "free_bytes": free_bytes,
        "free_gb": round(float(free_bytes) / GIB, 3),
        "estimated_snapshot_bytes": int(estimated_bytes),
        "estimated_snapshot_gb": round(float(estimated_bytes) / GIB, 3),
        "headroom_ratio": round(float(headroom_ratio), 3),
        "min_free_after_snapshot_gb": round(float(min_free_after_gb), 3),
        "required_free_bytes": required_free_bytes,
        "sufficient": bool(capacity_known and free_bytes >= required_free_bytes),
        "error": capacity_error,
    }
    payload["capacity_preflight"] = capacity
    if not bool(capacity["sufficient"]):
        payload["skipped_reason"] = "insufficient_local_capacity"
        return payload

    staging_root = recovery_root / f".latest_staging_{int(time.time() * 1000)}"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    stage_rows = available
    if not require_writer_quiet:
        stage_rows = [row for row in available if str(row.get("rel_path") or "") not in IMPORTANT_FILES]
    staged = _stage_selected_paths(local_root, staging_root, stage_rows, compare_existing=False)
    if not require_writer_quiet:
        sqlite_src = local_root / "data" / "snapshot_context.sqlite3"
        sqlite_dst = staging_root / "data" / "snapshot_context.sqlite3"
        if sqlite_src.is_file():
            backed_up, backup_error = _backup_sqlite_online(sqlite_src, sqlite_dst)
            if backed_up:
                staged["copied_paths"].append("data/snapshot_context.sqlite3")
            else:
                staged["errors"].append(backup_error)

    manifest = {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "source_root": str(local_root),
        "snapshot_root": str(latest_root),
        "staging_root": str(staging_root),
        "snapshot_mode": "writer_quiet" if require_writer_quiet else "online_sqlite_backup",
        "copied_paths": staged["copied_paths"],
        "skipped_paths": staged["skipped_paths"],
        "transient_missing": staged["transient_missing"],
        "errors": staged["errors"],
    }
    manifest_path = recovery_root / "recovery_manifest_latest.json"
    if not staged["errors"]:
        _promote_latest_snapshot(staging_root, latest_root)
        state["last_snapshot_epoch"] = time.time()
        state["last_snapshot_root"] = str(latest_root)
    else:
        shutil.rmtree(staging_root, ignore_errors=True)
    _write_json(manifest_path, manifest)

    payload.update(
        {
            "attempted": True,
            "ok": not staged["errors"],
            "copied_paths": staged["copied_paths"],
            "skipped_paths": staged["skipped_paths"],
            "transient_missing_count": int(len(staged["transient_missing"])),
            "transient_missing": staged["transient_missing"],
            "error_count": int(len(staged["errors"])),
            "errors": staged["errors"],
            "unsafe_skipped_paths": staged["unsafe_skipped_paths"],
            "manifest_path": str(manifest_path),
        }
    )
    return payload


def _transactional_curated_restore(
    source_root: Path,
    external_root: Path,
    *,
    apply: bool,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    selected = _recovery_selected_paths(source_root)
    available = [row for row in selected if bool(row.get("eligible", row.get("exists", False)))]
    payload: dict[str, Any] = {
        "attempted": False,
        "ok": False,
        "source_root": str(source_root),
        "target_root": str(external_root),
        "selected_paths": selected,
    }
    if not apply:
        payload["skipped_reason"] = "apply_disabled"
        return payload
    if not available:
        payload["skipped_reason"] = "no_curated_paths_available"
        return payload
    quiet_point = _writer_quiet_point(project_root, apply=apply)
    payload["quiet_point"] = quiet_point
    if not bool(quiet_point.get("ok", False)):
        payload["skipped_reason"] = str(quiet_point.get("skipped_reason") or "writer_not_quiet")
        return payload
    restored = _stage_selected_paths(source_root, external_root, available, compare_existing=True)
    payload.update(
        {
            "attempted": True,
            "ok": not restored["errors"],
            "copied_paths": restored["copied_paths"],
            "skipped_paths": restored["skipped_paths"],
            "transient_missing_count": int(len(restored["transient_missing"])),
            "transient_missing": restored["transient_missing"],
            "error_count": int(len(restored["errors"])),
            "errors": restored["errors"],
            "unsafe_skipped_paths": restored["unsafe_skipped_paths"],
        }
    )
    return payload


def _active_model_rows(project_root: Path) -> list[dict[str, str]]:
    registry = _load_json(project_root / "master_bot_registry.json")
    rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    active: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict) or not bool(row.get("active", False)):
            continue
        bot_id = str(row.get("bot_id") or "").strip()
        model_path = str(row.get("model_path") or "").strip()
        model_name = Path(model_path).name if model_path else ""
        key = (bot_id, model_name)
        if not bot_id or not model_name or key in seen:
            continue
        seen.add(key)
        active.append({"bot_id": bot_id, "model_name": model_name})
    return active


def _promotion_model_ids(project_root: Path) -> set[str]:
    packet = _load_json(project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json")
    scope = packet.get("promotion_scope") if isinstance(packet.get("promotion_scope"), dict) else {}
    return {
        str(bot_id or "").strip()
        for bot_id in (scope.get("trained_bot_ids") if isinstance(scope.get("trained_bot_ids"), list) else [])
        if str(bot_id or "").strip()
    }


def _model_route_contract(project_root: Path, local_root: Path) -> dict[str, Any]:
    rows = _active_model_rows(project_root)
    promotion_ids = _promotion_model_ids(project_root)
    available: list[str] = []
    missing: list[str] = []
    promotion_missing: list[str] = []
    for row in rows:
        model_path = local_root / "models" / row["model_name"]
        present = bool(model_path.is_file() and model_path.stat().st_size > 0)
        target = available if present else missing
        target.append(row["bot_id"])
        if not present and row["bot_id"] in promotion_ids:
            promotion_missing.append(row["bot_id"])
    return {
        "active_model_count": len(rows),
        "available_active_model_count": len(available),
        "missing_active_model_count": len(missing),
        "active_model_coverage_ratio": round(len(available) / max(len(rows), 1), 6),
        "missing_active_model_ids": missing[:100],
        "promotion_model_count": len(promotion_ids),
        "missing_promotion_model_ids": promotion_missing,
        "promotion_model_coverage_ready": not promotion_missing,
        "paper_collection_model_gaps_advisory": bool(missing and not promotion_missing),
        "policy": "missing active-model artifacts remain visible and block promotion only when the bot enters an explicit promotion packet",
    }


def _configured_external_model_root() -> Path:
    configured = str(os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "") or "").strip()
    if configured:
        return Path(configured).expanduser() / "models"
    mount_root = Path(os.getenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS")).expanduser()
    return mount_root / _project_dir_from_env() / "models"


def _hydrate_local_models(
    project_root: Path,
    local_root: Path,
    *,
    apply: bool,
    state: dict[str, Any],
) -> dict[str, Any]:
    source_root = _configured_external_model_root()
    destination_root = local_root / "models"
    cooldown_seconds = max(_safe_float(os.getenv("BOT_LOGS_MODEL_HYDRATION_COOLDOWN_SECONDS"), 21600.0), 0.0)
    last_epoch = _safe_float(state.get("last_model_hydration_epoch"), 0.0)
    cooldown_remaining = max(last_epoch + cooldown_seconds - time.time(), 0.0)
    rows = _active_model_rows(project_root)
    missing_rows = [row for row in rows if not (destination_root / row["model_name"]).is_file()]
    payload: dict[str, Any] = {
        "attempted": False,
        "ok": not missing_rows,
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "requested_model_count": len(rows),
        "missing_before_count": len(missing_rows),
        "copied_model_count": 0,
        "source_missing_count": 0,
        "copy_error_count": 0,
        "copied_bytes": 0,
        "cooldown_remaining_seconds": round(cooldown_remaining, 3),
    }
    if not apply:
        payload["skipped_reason"] = "apply_disabled"
        return payload
    if not missing_rows:
        payload["skipped_reason"] = "local_models_complete"
        return payload
    if cooldown_remaining > 0.0:
        payload["skipped_reason"] = "hydration_cooldown_active"
        return payload
    if not source_root.is_dir():
        payload["skipped_reason"] = "external_model_source_unavailable"
        return payload

    max_total_bytes = max(int(_safe_float(os.getenv("BOT_LOGS_MODEL_HYDRATION_MAX_BYTES"), float(1024**3))), 0)
    destination_root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    source_missing: list[str] = []
    errors: list[str] = []
    copied_bytes = 0
    payload["attempted"] = True
    state["last_model_hydration_epoch"] = time.time()
    for row in missing_rows:
        src = source_root / row["model_name"]
        dst = destination_root / row["model_name"]
        if not src.is_file():
            source_missing.append(row["bot_id"])
            continue
        try:
            size_bytes = max(int(src.stat().st_size), 0)
        except OSError as exc:
            errors.append(f"{row['bot_id']}:{type(exc).__name__}:{exc}")
            continue
        if max_total_bytes > 0 and copied_bytes + size_bytes > max_total_bytes:
            errors.append(f"{row['bot_id']}:hydration_byte_budget_exceeded")
            continue
        copy_state, detail = _copy_file_transactional(src, dst, compare_existing=False)
        if copy_state == "copied":
            copied.append(row["bot_id"])
            copied_bytes += size_bytes
        else:
            errors.append(detail or f"{row['bot_id']}:{copy_state}")
    remaining = [row for row in rows if not (destination_root / row["model_name"]).is_file()]
    payload.update(
        {
            "ok": not errors and not remaining,
            "copied_model_count": len(copied),
            "copied_model_ids": copied[:100],
            "source_missing_count": len(source_missing),
            "source_missing_model_ids": source_missing[:100],
            "copy_error_count": len(errors),
            "copy_errors": errors[:100],
            "copied_bytes": copied_bytes,
            "missing_after_count": len(remaining),
            "missing_after_model_ids": [row["bot_id"] for row in remaining[:100]],
        }
    )
    return payload


def _recovery_snapshot_contract(project_root: Path, recovery_root: Path) -> dict[str, Any]:
    manifest_path = recovery_root / "recovery_manifest_latest.json"
    manifest = _load_json(manifest_path)
    latest_root = recovery_root / "latest"
    max_age_minutes = max(
        _safe_float(os.getenv("BOT_LOGS_RECOVERY_MAX_SNAPSHOT_AGE_HOURS"), 36.0) * 60.0,
        60.0,
    )
    age_minutes = payload_age_minutes(manifest, manifest_path)
    copied_paths = manifest.get("copied_paths") if isinstance(manifest.get("copied_paths"), list) else []
    snapshot_db_present = bool((latest_root / "data" / "snapshot_context.sqlite3").is_file())
    manifest_clean = bool(manifest and not list(manifest.get("errors") or []))
    snapshot_fresh = bool(age_minutes is not None and age_minutes <= max_age_minutes)

    content_path = project_root / "governance" / "content_store" / "latest.json"
    content = _load_json(content_path)
    content_age = payload_age_minutes(content, content_path)
    content_fresh = bool(content_age is not None and content_age <= 24.0 * 60.0)
    content_ready = bool(content_fresh and content.get("ok", False) and str(content.get("manifest_hash") or ""))
    blockers = []
    if not latest_root.is_dir():
        blockers.append("recovery_snapshot_missing")
    if not manifest_clean:
        blockers.append("recovery_manifest_missing_or_unclean")
    if not snapshot_fresh:
        blockers.append("recovery_snapshot_stale")
    if not snapshot_db_present:
        blockers.append("snapshot_context_backup_missing")
    if not content_ready:
        blockers.append("immutable_control_plane_evidence_not_current")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "manifest_path": str(manifest_path),
        "latest_snapshot_root": str(latest_root),
        "age_minutes": round(float(age_minutes), 3) if age_minutes is not None else None,
        "max_age_minutes": max_age_minutes,
        "manifest_clean": manifest_clean,
        "copied_path_count": len(copied_paths),
        "snapshot_context_backup_present": snapshot_db_present,
        "content_store": {
            "path": str(content_path),
            "ready": content_ready,
            "age_minutes": round(float(content_age), 3) if content_age is not None else None,
            "manifest_hash": str(content.get("manifest_hash") or ""),
        },
    }


def _durability_contract(
    project_root: Path,
    recovery_root: Path,
    probe: dict[str, Any],
    current_mode: str,
    route_policy: dict[str, Any],
    local_root: Path,
) -> dict[str, Any]:
    snapshot = _recovery_snapshot_contract(project_root, recovery_root)
    models = _model_route_contract(project_root, local_root)
    hot_path_ready = bool(probe.get("hot_storage_available", False) or current_mode in LOCAL_MODES)
    local_route_certified = bool(current_mode in LOCAL_MODES and route_policy.get("local_route_pinned", False))
    blockers = []
    if not hot_path_ready:
        blockers.append("local_hot_path_unavailable")
    if not local_route_certified:
        blockers.append("local_hot_route_not_pinned")
    blockers.extend(str(item) for item in snapshot.get("blockers") or [])
    if not bool(models.get("promotion_model_coverage_ready", False)):
        blockers.append("promotion_model_artifacts_missing_from_local_route")
    return {
        "ready": not blockers,
        "status": "ready_local_durable" if not blockers else "degraded_local_durability",
        "blockers": list(dict.fromkeys(blockers)),
        "hot_path_ready": hot_path_ready,
        "external_required_for_hot_path": bool(probe.get("external_required_for_hot_path", False)),
        "local_route_certified": local_route_certified,
        "recovery_snapshot": snapshot,
        "model_route": models,
        "policy": "a pinned local hot route is production-ready only when its recovery snapshot, immutable control-plane evidence, and current promotion models are independently recoverable",
    }


def _recommended_actions(
    probe: dict[str, Any],
    current_mode: str,
    route_policy: dict[str, Any],
    durability: dict[str, Any] | None = None,
) -> list[str]:
    actions: list[str] = []
    if not bool(probe.get("external_available", False)):
        actions.append("keep BOT_LOGS routed to local fallback until the target APFS volume is mounted and writable again")
        if bool(probe.get("target_volume_present", False)) and not bool(probe.get("target_volume_mounted", False)):
            actions.append("let the storage disaster recovery bot keep attempting exact-volume remounts for the configured BOT_LOGS target")
        actions.append("maintain the curated internal BOT_LOGS recovery mirror so governance, decisions, exports, and models stay recoverable")
    if current_mode == "external_available_unverified":
        actions.append("refresh the storage failback artifacts if you want the route controller to certify the rebuilt BOT_LOGS volume as the active live SQLite route")
    if current_mode in LOCAL_MODES:
        if bool(route_policy.get("local_route_pinned", False)):
            actions.append("keep the verified local hot route pinned; use an explicit certified storage switch when an external live route is wanted")
        elif not bool(route_policy.get("automatic_external_failback_enabled", False)):
            actions.append("keep the local hot route until an operator explicitly certifies an external failback")
    durability = durability or {}
    for blocker in durability.get("blockers") or []:
        if blocker == "recovery_snapshot_stale":
            actions.append("refresh the bounded local recovery snapshot at the next writer quiet point")
        elif blocker == "immutable_control_plane_evidence_not_current":
            actions.append("refresh the content-addressed control-plane evidence manifest")
        elif blocker == "promotion_model_artifacts_missing_from_local_route":
            actions.append("hydrate every promoted model artifact onto the pinned local route before live consideration")
    model_route = durability.get("model_route") if isinstance(durability.get("model_route"), dict) else {}
    if bool(model_route.get("paper_collection_model_gaps_advisory", False)):
        actions.append("retrain or restore missing paper-only model artifacts before those bots enter promotion scope")
    return actions


def _overall_status(
    probe: dict[str, Any],
    current_mode: str,
    route_policy: dict[str, Any],
    durability: dict[str, Any] | None = None,
) -> str:
    if current_mode in LOCAL_MODES:
        return "ready" if bool((durability or {}).get("ready", False)) else "degraded"
    if bool(probe.get("external_available", False)) and current_mode in EXTERNAL_CERTIFIED_MODES:
        return "ready"
    if bool(probe.get("external_available", False)) and current_mode == "external_available_unverified":
        return "degraded"
    if bool(probe.get("external_available", False)):
        return "degraded"
    return "blocked"


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
    fh.write(f"pid={os.getpid()} started={_utc_now()} cmd={' '.join(sys.argv)}")
    fh.flush()
    return fh, ""


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool,
    recovery_root: Path,
    state_path: Path,
    mount_cooldown_seconds: float,
    snapshot_cooldown_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _load_json(state_path)
    route_policy = _route_policy(project_root)
    initial_probe = _probe_storage()
    current_mode = _current_storage_mode(project_root, probe=initial_probe)
    local_root = Path(os.getenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(DEFAULT_LOCAL_ROOT))).expanduser()
    snapshot_workspace_cleanup = _cleanup_snapshot_workspace(recovery_root, apply=apply)

    mount_attempt = _mount_target_volume(
        initial_probe,
        apply=apply,
        state=state,
        cooldown_seconds=float(mount_cooldown_seconds),
    )

    probe_after_mount = _probe_storage()
    current_mode_after_mount = _current_storage_mode(project_root, probe=probe_after_mount)

    switch_local = {
        "attempted": False,
        "ok": False,
        "target_mode": "local",
        "skipped_reason": "not_required",
    }
    if (not bool(probe_after_mount.get("external_available", False))) and current_mode_after_mount not in LOCAL_MODES:
        switch_local = _switch_storage_mode(project_root, "local", apply=apply)
        current_mode_after_mount = _current_storage_mode(project_root, probe=probe_after_mount)

    model_hydration = {
        "attempted": False,
        "ok": False,
        "skipped_reason": "not_local_hot_route",
    }
    if current_mode_after_mount in LOCAL_MODES:
        model_hydration = _hydrate_local_models(
            project_root,
            local_root,
            apply=apply,
            state=state,
        )

    snapshot = {
        "attempted": False,
        "ok": False,
        "skipped_reason": "not_required",
        "recovery_root": str(recovery_root),
        "snapshot_workspace_cleanup": snapshot_workspace_cleanup,
        "latest_snapshot_root": str(recovery_root / "latest"),
        "selected_paths": _recovery_selected_paths(local_root),
    }
    if current_mode_after_mount in LOCAL_MODES:
        pinned_local_route = bool(route_policy.get("local_route_pinned", False))
        effective_snapshot_cooldown = float(snapshot_cooldown_seconds)
        if pinned_local_route:
            effective_snapshot_cooldown = max(
                effective_snapshot_cooldown,
                _safe_float(os.getenv("BOT_LOGS_LOCAL_PINNED_SNAPSHOT_COOLDOWN_SECONDS"), 43200.0),
            )
        snapshot = _take_curated_snapshot(
            local_root,
            recovery_root,
            apply=apply,
            state=state,
            cooldown_seconds=effective_snapshot_cooldown,
            project_root=project_root,
            require_writer_quiet=not pinned_local_route,
        )

    probe_after_snapshot = _probe_storage()
    current_mode_after_snapshot = _current_storage_mode(project_root, probe=probe_after_snapshot)

    curated_restore = {
        "attempted": False,
        "ok": False,
        "skipped_reason": "not_required",
    }
    if bool(probe_after_snapshot.get("external_available", False)) and current_mode_after_snapshot in (LOCAL_MODES | EXTERNAL_RECOVERY_MODES):
        source_root = recovery_root / "latest"
        if not source_root.exists():
            source_root = local_root
        curated_restore = _transactional_curated_restore(
            source_root,
            Path(str(probe_after_snapshot.get("external_root") or "")).expanduser(),
            apply=apply,
            project_root=project_root,
        )

    restore_external = {
        "attempted": False,
        "ok": False,
        "target_mode": "external",
        "skipped_reason": "not_required",
    }
    external_failback_allowed = bool(route_policy.get("automatic_external_failback_enabled", False)) and not bool(
        route_policy.get("local_route_pinned", False)
    )
    if (
        bool(probe_after_snapshot.get("external_available", False))
        and current_mode_after_snapshot in (LOCAL_MODES | EXTERNAL_RECOVERY_MODES)
        and external_failback_allowed
    ):
        restore_external = _switch_storage_mode(project_root, "external", apply=apply)
        current_mode_after_snapshot = _current_storage_mode(project_root, probe=probe_after_snapshot)
    elif bool(probe_after_snapshot.get("external_available", False)) and current_mode_after_snapshot in LOCAL_MODES:
        restore_external["skipped_reason"] = (
            "local_route_pinned"
            if bool(route_policy.get("local_route_pinned", False))
            else "automatic_external_failback_disabled"
        )

    final_probe = _probe_storage()
    final_mode = _current_storage_mode(project_root, probe=final_probe)
    target_override = _sync_storage_target_override(project_root, final_probe, apply=apply)

    finder_sync = {
        "attempted": False,
        "ok": False,
        "skipped_reason": "not_required",
    }
    if any(bool(step.get("attempted", False)) for step in (mount_attempt, switch_local, snapshot, curated_restore, restore_external)) or bool(target_override.get("changed", False)):
        finder_sync = _sync_finder_shortcuts(project_root, apply=apply)

    durability = _durability_contract(
        project_root,
        recovery_root,
        final_probe,
        final_mode,
        route_policy,
        local_root,
    )
    overall_status = _overall_status(final_probe, final_mode, route_policy, durability)
    payload = {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "project_root": str(project_root),
        "apply_requested": bool(apply),
        "overall_status": overall_status,
        "ok": overall_status != "blocked",
        "current_storage_mode": final_mode,
        "route_policy": route_policy,
        "local_fallback_root": str(local_root),
        "recovery_root": str(recovery_root),
        "storage_probe": final_probe,
        "initial_storage_probe": initial_probe,
        "recommended_actions": _recommended_actions(final_probe, final_mode, route_policy, durability),
        "mount_attempt": mount_attempt,
        "switch_local": switch_local,
        "recovery_snapshot": snapshot,
        "durability_contract": durability,
        "model_hydration": model_hydration,
        "curated_restore": curated_restore,
        "restore_external": restore_external,
        "target_override": target_override,
        "finder_sync": finder_sync,
        "automation_contract": {
            "launchd_label": "com.dankingsley.storage_disaster_recovery",
            "run_command": "./scripts/ops/opsctl.sh storage-disaster-recovery --apply --json",
            "interval_seconds": max(int(float(os.getenv("BOT_LOGS_RECOVERY_AUTO_INTERVAL_SECONDS", "300") or 300)), 60),
            "enabled_by_default": True,
            "automatic_external_failback_enabled_by_default": False,
            "pinned_local_online_snapshot_cooldown_seconds": max(
                int(_safe_float(os.getenv("BOT_LOGS_LOCAL_PINNED_SNAPSHOT_COOLDOWN_SECONDS"), 43200.0)),
                3600,
            ),
            "pinned_local_online_snapshot_uses_sqlite_backup_api": True,
        },
        "upgrade_track": {
            "upgradeable": True,
            "family": "infrabots",
            "install_command": "./scripts/install_storage_disaster_recovery_launchd.sh",
        },
    }
    return payload, state


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-recover BOT_LOGS routing and curate a recovery mirror when the external APFS volume disappears.")
    parser.add_argument("--apply", action="store_true", help="Attempt mount, route-switch, finder-sync, and recovery snapshot actions.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--recovery-root", default=str(Path(os.getenv("BOT_LOGS_RECOVERY_AUTO_ROOT", str(DEFAULT_RECOVERY_ROOT))).expanduser()))
    parser.add_argument("--mount-cooldown-seconds", type=float, default=float(os.getenv("BOT_LOGS_RECOVERY_MOUNT_COOLDOWN_SECONDS", "120")))
    parser.add_argument("--snapshot-cooldown-seconds", type=float, default=float(os.getenv("BOT_LOGS_RECOVERY_SNAPSHOT_COOLDOWN_SECONDS", "3600")))
    args = parser.parse_args()

    lock_handle, owner = _acquire_singleton_lock(Path(args.lock_file).expanduser())
    if lock_handle is None:
        payload = {
            "timestamp_utc": _utc_now(),
            "schema_version": 1,
            "ok": False,
            "overall_status": "busy",
            "lock_owner": owner,
        }
        _write_json(Path(args.out_file).expanduser(), payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(f"storage_disaster_recovery busy owner={owner}")
        return 0

    try:
        payload, state = build_payload(
            PROJECT_ROOT,
            apply=bool(args.apply),
            recovery_root=Path(args.recovery_root).expanduser(),
            state_path=Path(args.state_file).expanduser(),
            mount_cooldown_seconds=float(args.mount_cooldown_seconds),
            snapshot_cooldown_seconds=float(args.snapshot_cooldown_seconds),
        )
        _write_json(Path(args.out_file).expanduser(), payload)
        _write_json(Path(args.state_file).expanduser(), state)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(
                "storage_disaster_recovery "
                f"status={payload['overall_status']} "
                f"mode={payload['current_storage_mode']} "
                f"external_available={int(bool((payload.get('storage_probe') or {}).get('external_available', False)))}"
            )
        return 0 if bool(payload.get("ok", False)) else 1
    finally:
        try:
            lock_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
