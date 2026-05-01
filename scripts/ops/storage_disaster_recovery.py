#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
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
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_python import resolve_runtime_python
    from core.storage_mounts import find_target_external_volume, resolve_external_storage
    from core.storage_target_override import DEFAULT_STORAGE_TARGET_OVERRIDE_PATH, write_storage_target_override
    from scripts.ops import writer_cycle_coordinator as writer_src


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "storage_disaster_recovery_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "storage_disaster_recovery_state.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "storage_disaster_recovery.lock"
DEFAULT_RECOVERY_ROOT = Path.home() / "Documents" / "BOT_LOGS_recovery_auto"
DEFAULT_LOCAL_ROOT = PROJECT_ROOT / "local_fallback_storage"
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
LOCAL_MODES = {"local_fallback", "local_fallback_split_brain"}
EXTERNAL_CERTIFIED_MODES = {"external", "external_curated"}
EXTERNAL_RECOVERY_MODES = EXTERNAL_CERTIFIED_MODES | {"external_available_unverified", "unknown"}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


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


def _current_storage_mode(project_root: Path, probe: dict[str, Any] | None = None) -> str:
    failback = _load_json(project_root / "governance" / "health" / "storage_failback_sync_latest.json")
    mount_guard = _load_json(project_root / "governance" / "health" / "storage_mount_guard_latest.json")
    mode = str(failback.get("certified_mode") or failback.get("mode") or mount_guard.get("storage_mode") or "").strip()
    if mode:
        return mode
    live_probe = probe if isinstance(probe, dict) else _probe_storage()
    if bool(live_probe.get("external_available", False)):
        return "external_available_unverified"
    return "unknown"


def _recovery_selected_paths(local_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel_path in IMPORTANT_DIRS:
        src = local_root / rel_path
        rows.append(
            {
                "rel_path": rel_path,
                "kind": "dir",
                "source": str(src),
                "exists": bool(src.exists() and src.is_dir()),
            }
        )
    for rel_path in IMPORTANT_FILES:
        src = local_root / rel_path
        rows.append(
            {
                "rel_path": rel_path,
                "kind": "file",
                "source": str(src),
                "exists": bool(src.exists() and src.is_file()),
            }
        )
    return rows


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
            "--no-restart",
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


def _take_curated_snapshot(
    local_root: Path,
    recovery_root: Path,
    *,
    apply: bool,
    state: dict[str, Any],
    cooldown_seconds: float,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    selected = _recovery_selected_paths(local_root)
    available = [row for row in selected if bool(row.get("exists", False))]
    last_epoch = _safe_float(state.get("last_snapshot_epoch"), 0.0)
    latest_root = recovery_root / "latest"
    cooldown_remaining = max(last_epoch + max(float(cooldown_seconds), 0.0) - time.time(), 0.0)

    payload: dict[str, Any] = {
        "attempted": False,
        "ok": False,
        "recovery_root": str(recovery_root),
        "latest_snapshot_root": str(latest_root),
        "selected_paths": selected,
        "available_path_count": int(len(available)),
        "cooldown_remaining_seconds": round(float(cooldown_remaining), 3),
    }
    if not apply:
        payload["skipped_reason"] = "apply_disabled"
        return payload
    if not available:
        payload["skipped_reason"] = "no_curated_paths_available"
        return payload
    if latest_root.exists() and cooldown_remaining > 0.0:
        payload["ok"] = True
        payload["skipped_reason"] = "snapshot_cooldown_active"
        return payload

    quiet_point = _writer_quiet_point(project_root, apply=apply)
    payload["quiet_point"] = quiet_point
    if not bool(quiet_point.get("ok", False)):
        payload["skipped_reason"] = str(quiet_point.get("skipped_reason") or "writer_not_quiet")
        return payload

    staging_root = recovery_root / f".latest_staging_{int(time.time() * 1000)}"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    staged = _stage_selected_paths(local_root, staging_root, available, compare_existing=False)

    manifest = {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "source_root": str(local_root),
        "snapshot_root": str(latest_root),
        "staging_root": str(staging_root),
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
    available = [row for row in selected if bool(row.get("exists", False))]
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
        }
    )
    return payload


def _recommended_actions(probe: dict[str, Any], current_mode: str) -> list[str]:
    actions: list[str] = []
    if not bool(probe.get("external_available", False)):
        actions.append("keep BOT_LOGS routed to local fallback until the target APFS volume is mounted and writable again")
        if bool(probe.get("target_volume_present", False)) and not bool(probe.get("target_volume_mounted", False)):
            actions.append("let the storage disaster recovery bot keep attempting exact-volume remounts for the configured BOT_LOGS target")
        actions.append("maintain the curated internal BOT_LOGS recovery mirror so governance, decisions, exports, and models stay recoverable")
    if current_mode == "external_available_unverified":
        actions.append("refresh the storage failback artifacts if you want the route controller to certify the rebuilt BOT_LOGS volume as the active live SQLite route")
    if current_mode in LOCAL_MODES:
        actions.append("when BOT_LOGS returns, the bot can switch the runtime back to the external route automatically without touching VIDEO")
    return actions


def _overall_status(probe: dict[str, Any], current_mode: str) -> str:
    if bool(probe.get("external_available", False)) and current_mode in EXTERNAL_CERTIFIED_MODES:
        return "ready"
    if bool(probe.get("external_available", False)) and current_mode == "external_available_unverified":
        return "degraded"
    if current_mode in LOCAL_MODES:
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
    initial_probe = _probe_storage()
    current_mode = _current_storage_mode(project_root, probe=initial_probe)
    local_root = Path(os.getenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(DEFAULT_LOCAL_ROOT))).expanduser()

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

    snapshot = {
        "attempted": False,
        "ok": False,
        "skipped_reason": "not_required",
        "recovery_root": str(recovery_root),
        "latest_snapshot_root": str(recovery_root / "latest"),
        "selected_paths": _recovery_selected_paths(local_root),
    }
    if not bool(probe_after_mount.get("external_available", False)) and current_mode_after_mount in LOCAL_MODES:
        snapshot = _take_curated_snapshot(
            local_root,
            recovery_root,
            apply=apply,
            state=state,
            cooldown_seconds=float(snapshot_cooldown_seconds),
            project_root=project_root,
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
    if bool(probe_after_snapshot.get("external_available", False)) and current_mode_after_snapshot in (LOCAL_MODES | EXTERNAL_RECOVERY_MODES):
        restore_external = _switch_storage_mode(project_root, "external", apply=apply)
        current_mode_after_snapshot = _current_storage_mode(project_root, probe=probe_after_snapshot)

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

    payload = {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "project_root": str(project_root),
        "apply_requested": bool(apply),
        "overall_status": _overall_status(final_probe, final_mode),
        "ok": _overall_status(final_probe, final_mode) != "blocked",
        "current_storage_mode": final_mode,
        "local_fallback_root": str(local_root),
        "recovery_root": str(recovery_root),
        "storage_probe": final_probe,
        "initial_storage_probe": initial_probe,
        "recommended_actions": _recommended_actions(final_probe, final_mode),
        "mount_attempt": mount_attempt,
        "switch_local": switch_local,
        "recovery_snapshot": snapshot,
        "curated_restore": curated_restore,
        "restore_external": restore_external,
        "target_override": target_override,
        "finder_sync": finder_sync,
        "automation_contract": {
            "launchd_label": "com.dankingsley.storage_disaster_recovery",
            "run_command": "./scripts/ops/opsctl.sh storage-disaster-recovery --apply --json",
            "interval_seconds": max(int(float(os.getenv("BOT_LOGS_RECOVERY_AUTO_INTERVAL_SECONDS", "300") or 300)), 60),
            "enabled_by_default": True,
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
