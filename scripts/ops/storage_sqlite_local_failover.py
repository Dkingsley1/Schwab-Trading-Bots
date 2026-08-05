#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_maintenance import maintenance_hold_snapshot
    from core.storage_mounts import resolve_external_storage
    from scripts.ops import writer_cycle_coordinator as writer_state
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_maintenance import maintenance_hold_snapshot
    from core.storage_mounts import resolve_external_storage
    from scripts.ops import writer_cycle_coordinator as writer_state


TRACKED_DATABASES = (
    "data/jsonl_link.sqlite3",
    "data/bot_channel_queue.sqlite3",
    "data/snapshot_context.sqlite3",
)
RUNTIME_PROCESS_MARKERS = (
    "scripts/shadow_watchdog.py",
    "scripts/run_all_sleeves.py",
    "scripts/run_parallel_shadows.py",
    "scripts/run_parallel_aggressive_modes.py",
    "scripts/run_shadow_training_loop.py",
    "scripts/ops/sql_link_shard_manager.py",
    "scripts/ops/sql_link_writer_service.py",
    "scripts/link_jsonl_to_sql.py",
    "scripts/ops/storage_failback_sync.py",
    "scripts/ops/storage_switch_orchestrator.py",
    "scripts/ops/storage_split_brain_reconciler.py",
    "scripts/ops/storage_transition_coordinator.py",
    "scripts/sqlite_performance_maintenance.py",
    "scripts/ops/external_backlog_drain.py",
)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "storage_sqlite_local_failover_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "storage_sqlite_local_failover.lock"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.storage_override"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _disk_free_bytes(path: Path) -> int | None:
    try:
        stat = os.statvfs(path)
        return int(stat.f_bavail * stat.f_frsize)
    except Exception:
        return None


def _size_bytes(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def _stat_signature(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except Exception:
        return {"exists": False, "size_bytes": 0, "mtime_ns": 0}
    return {
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _source_signature(path: Path) -> dict[str, Any]:
    wal = _stat_signature(Path(f"{path}-wal"))
    if int(wal.get("size_bytes", 0) or 0) == 0:
        wal = {"exists": False, "size_bytes": 0, "mtime_ns": 0}
    return {
        "primary": _stat_signature(path),
        "wal": wal,
    }


def _connect_readonly(path: Path, *, timeout_seconds: float) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=max(float(timeout_seconds), 1.0))
    conn.execute(f"PRAGMA busy_timeout={int(max(float(timeout_seconds), 1.0) * 1000)}")
    conn.execute("PRAGMA query_only=ON")
    return conn


def _pragma_int(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute(f"PRAGMA {name}").fetchone()
    return int(row[0] if row else 0)


def _database_identity(conn: sqlite3.Connection) -> dict[str, Any]:
    schema_rows = [
        tuple(str(value or "") for value in row)
        for row in conn.execute(
            """
            SELECT type, name, tbl_name, rootpage, sql
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
    ]
    schema_encoded = json.dumps(schema_rows, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return {
        "page_count": _pragma_int(conn, "page_count"),
        "page_size": _pragma_int(conn, "page_size"),
        "freelist_count": _pragma_int(conn, "freelist_count"),
        "schema_version": _pragma_int(conn, "schema_version"),
        "user_version": _pragma_int(conn, "user_version"),
        "application_id": _pragma_int(conn, "application_id"),
        "schema_object_count": len(schema_rows),
        "schema_sha256": hashlib.sha256(schema_encoded).hexdigest(),
    }


def _quick_check(path: Path, *, timeout_seconds: float) -> dict[str, Any]:
    try:
        with _connect_readonly(path, timeout_seconds=timeout_seconds) as conn:
            rows = conn.execute("PRAGMA quick_check").fetchall()
        results = [str(row[0] if row else "") for row in rows]
    except Exception as exc:
        return {"ok": False, "result": str(exc), "error_type": type(exc).__name__}
    ok = results == ["ok"]
    return {"ok": ok, "result": "ok" if ok else "; ".join(results[:8])}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _runtime_processes() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception as exc:
        return [{"pid": 0, "command": f"process_scan_failed:{type(exc).__name__}:{exc}"}]

    matches: list[dict[str, Any]] = []
    for raw in (proc.stdout or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except Exception:
            continue
        if pid == os.getpid() or "storage_sqlite_local_failover.py" in command:
            continue
        if any(marker in command for marker in RUNTIME_PROCESS_MARKERS):
            matches.append({"pid": pid, "command": command[:500]})
    return matches


def _writer_is_active(snapshot: dict[str, Any]) -> bool:
    if bool(snapshot.get("active", False)):
        return True
    if bool(snapshot.get("writer_lock_held", False)):
        return True
    if bool(snapshot.get("writer_owner_pid_live", False)):
        return True
    if bool(snapshot.get("child_writer_active", False)):
        return True
    if int(snapshot.get("active_child_writer_count", 0) or 0) > 0:
        return True
    return False


def _normalize_relative_paths(relative_paths: list[str] | tuple[str, ...]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in relative_paths:
        rel = str(raw or "").replace("\\", "/").strip().lstrip("./")
        path = Path(rel)
        if not rel or path.is_absolute() or ".." in path.parts or not rel.endswith(".sqlite3"):
            raise ValueError(f"unsafe SQLite relative path: {raw}")
        if rel not in seen:
            normalized.append(rel)
            seen.add(rel)
    if not normalized:
        raise ValueError("at least one SQLite relative path is required")
    return normalized


def _route_snapshot(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        return {"state": "symlink", "target": os.readlink(path)}
    if path.exists():
        return {"state": "regular", "target": ""}
    return {"state": "missing", "target": ""}


def _atomic_symlink(link: Path, target: Path | str) -> None:
    if link.exists() and not link.is_symlink():
        raise RuntimeError(f"route path is not a symlink: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp = link.with_name(f".{link.name}.{os.getpid()}.route_tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target)
    os.replace(tmp, link)


def _restore_route(path: Path, snapshot: dict[str, Any]) -> None:
    state = str(snapshot.get("state") or "missing")
    if state == "symlink":
        _atomic_symlink(path, str(snapshot.get("target") or ""))
    elif state == "missing":
        if path.exists() or path.is_symlink():
            path.unlink()
    else:
        raise RuntimeError(f"cannot restore non-symlink route: {path}")


def _identity_matches(source: dict[str, Any], target: dict[str, Any]) -> tuple[bool, list[str]]:
    fields = (
        "page_count",
        "page_size",
        "freelist_count",
        "user_version",
        "application_id",
        "schema_object_count",
        "schema_sha256",
    )
    mismatches = [name for name in fields if source.get(name) != target.get(name)]
    return not mismatches, mismatches


def _verify_existing_database(
    source: Path,
    target: Path,
    *,
    timeout_seconds: float,
    prior_entry: dict[str, Any] | None,
    require_prior_proof: bool,
) -> dict[str, Any]:
    source_before = _source_signature(source)
    try:
        with _connect_readonly(source, timeout_seconds=timeout_seconds) as source_conn:
            source_identity = _database_identity(source_conn)
        with _connect_readonly(target, timeout_seconds=timeout_seconds) as target_conn:
            target_identity = _database_identity(target_conn)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
    source_after = _source_signature(source)
    identity_ok, identity_mismatches = _identity_matches(source_identity, target_identity)
    source_stable = source_before == source_after
    quick_check = _quick_check(target, timeout_seconds=timeout_seconds)
    size_ok = _size_bytes(source) == _size_bytes(target)

    prior_stage = prior_entry.get("stage") if isinstance(prior_entry, dict) and isinstance(prior_entry.get("stage"), dict) else {}
    prior_commit = prior_entry.get("commit") if isinstance(prior_entry, dict) and isinstance(prior_entry.get("commit"), dict) else {}
    prior_source_signature = prior_stage.get("source_signature_after") if isinstance(prior_stage.get("source_signature_after"), dict) else {}
    prior_target_identity = prior_stage.get("target_identity") if isinstance(prior_stage.get("target_identity"), dict) else {}
    prior_identity_ok, prior_identity_mismatches = _identity_matches(prior_target_identity, target_identity)
    prior_proof_ok = bool(
        prior_entry
        and prior_stage.get("ok", False)
        and prior_stage.get("source_stable", False)
        and prior_stage.get("identity_ok", False)
        and isinstance(prior_stage.get("quick_check"), dict)
        and bool(prior_stage["quick_check"].get("ok", False))
        and prior_commit.get("target_installed", False)
        and prior_source_signature == source_after
        and prior_identity_ok
        and int(prior_commit.get("target_size_bytes", 0) or 0) == _size_bytes(target)
    )
    ok = bool(
        source_stable
        and identity_ok
        and size_ok
        and quick_check.get("ok", False)
        and (prior_proof_ok or not require_prior_proof)
    )
    return {
        "mode": "activate_existing_local",
        "source": str(source),
        "target": str(target),
        "source_signature_before": source_before,
        "source_signature_after": source_after,
        "source_identity": source_identity,
        "target_identity": target_identity,
        "source_stable": source_stable,
        "identity_ok": identity_ok,
        "identity_mismatches": identity_mismatches,
        "size_ok": size_ok,
        "target_size_bytes": _size_bytes(target),
        "quick_check": quick_check,
        "require_prior_proof": bool(require_prior_proof),
        "prior_proof_ok": prior_proof_ok,
        "prior_identity_mismatches": prior_identity_mismatches,
        "ok": ok,
    }


def _text_file_snapshot(path: Path) -> dict[str, Any]:
    try:
        return {"exists": True, "content": path.read_text(encoding="utf-8")}
    except FileNotFoundError:
        return {"exists": False, "content": ""}


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _install_local_storage_override(path: Path) -> dict[str, Any]:
    body = "# Auto-managed by storage_sqlite_local_failover.py\nBOT_LOGS_PREFER_EXTERNAL=0\n"
    before = _text_file_snapshot(path)
    _atomic_write_text(path, body)
    return {"path": str(path), "before": before, "installed": True, "content": body}


def _restore_text_file(path: Path, snapshot: dict[str, Any]) -> None:
    if bool(snapshot.get("exists", False)):
        _atomic_write_text(path, str(snapshot.get("content") or ""))
    elif path.exists() or path.is_symlink():
        path.unlink()


def _stage_database(
    source: Path,
    temp: Path,
    *,
    page_batch: int,
    timeout_seconds: float,
    progress: Callable[[int, int, int], None] | None,
) -> dict[str, Any]:
    temp.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (temp, Path(f"{temp}-wal"), Path(f"{temp}-shm")):
        if candidate.exists() or candidate.is_symlink():
            candidate.unlink()

    source_before = _source_signature(source)
    with _connect_readonly(source, timeout_seconds=timeout_seconds) as src:
        source_identity = _database_identity(src)
        dest = sqlite3.connect(str(temp), timeout=max(float(timeout_seconds), 1.0))
        try:
            dest.execute(f"PRAGMA busy_timeout={int(max(float(timeout_seconds), 1.0) * 1000)}")
            src.backup(
                dest,
                pages=max(int(page_batch), 1),
                progress=progress,
                sleep=0.05,
            )
            dest.commit()
        finally:
            dest.close()
        source_identity_after = _database_identity(src)

    source_after = _source_signature(source)
    with _connect_readonly(temp, timeout_seconds=timeout_seconds) as target_conn:
        target_identity = _database_identity(target_conn)
    identity_ok, identity_mismatches = _identity_matches(source_identity_after, target_identity)
    stable_source = source_before == source_after and source_identity == source_identity_after
    quick_check = _quick_check(temp, timeout_seconds=timeout_seconds)
    return {
        "source": str(source),
        "temp": str(temp),
        "source_signature_before": source_before,
        "source_signature_after": source_after,
        "source_identity": source_identity,
        "source_identity_after": source_identity_after,
        "target_identity": target_identity,
        "target_size_bytes": _size_bytes(temp),
        "source_stable": stable_source,
        "identity_ok": identity_ok,
        "identity_mismatches": identity_mismatches,
        "quick_check": quick_check,
        "ok": bool(stable_source and identity_ok and quick_check.get("ok", False)),
    }


def _acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate(0)
    handle.write(f"pid={os.getpid()} started={_utc_now()}")
    handle.flush()
    return handle


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    source_root: Path,
    target_root: Path,
    relative_paths: list[str] | tuple[str, ...] = TRACKED_DATABASES,
    apply: bool,
    min_free_after_bytes: int,
    page_batch: int,
    timeout_seconds: float,
    require_writer_idle: bool = True,
    require_runtime_stopped: bool = True,
    require_maintenance_hold: bool = True,
    preserve_existing: bool = True,
    activate_existing_local: bool = False,
    require_prior_activation_proof: bool = True,
    prior_proof_path: Path | None = None,
    storage_override_path: Path | None = None,
    lock_path: Path | None = None,
    progress_writer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    source_root = Path(source_root).expanduser().resolve()
    target_root = Path(target_root).expanduser().resolve()
    relative_paths = _normalize_relative_paths(relative_paths)
    run_id = _run_id()
    lock_path = Path(lock_path or (project_root / "governance" / "locks" / "storage_sqlite_local_failover.lock"))
    prior_proof_path = Path(prior_proof_path or (project_root / "governance" / "health" / "storage_sqlite_local_failover_latest.json"))
    prior_proof = _load_json(prior_proof_path) if activate_existing_local else {}
    prior_entries = {
        str(row.get("relative_path") or ""): row
        for row in (prior_proof.get("entries") or [])
        if isinstance(row, dict)
    }
    prior_proof_header_ok = bool(
        prior_proof.get("ok", False)
        and str(prior_proof.get("overall_status") or "")
        in {"switched_local_verified", "activated_existing_local_verified"}
        and isinstance(prior_proof.get("route_switch"), dict)
        and bool(prior_proof["route_switch"].get("ok", False))
    )
    storage_override_path = Path(storage_override_path or (project_root / "config" / ".env.storage_override"))
    writer_snapshot = writer_state.writer_state_snapshot(project_root)
    runtime_processes = _runtime_processes()
    maintenance_hold = maintenance_hold_snapshot(project_root)
    target_free_before = _disk_free_bytes(target_root)

    entries: list[dict[str, Any]] = []
    required_stage_bytes = 0
    blockers: list[str] = []
    for rel in relative_paths:
        source = source_root / rel
        target = target_root / rel
        repo = project_root / rel
        temp = target.with_name(f".{target.name}.local_failover_{run_id}.tmp")
        source_size = _size_bytes(source) + _size_bytes(Path(f"{source}-wal"))
        if not activate_existing_local:
            required_stage_bytes += source_size
        repo_route = _route_snapshot(repo)
        repo_realpath = repo.resolve(strict=False) if repo.is_symlink() else repo
        source_realpath = source.resolve(strict=False)
        entry_blockers: list[str] = []
        if not source.exists():
            entry_blockers.append("source_missing")
        elif source_size <= 0:
            entry_blockers.append("source_empty")
        if repo_route["state"] != "symlink":
            entry_blockers.append("repo_route_not_symlink")
        elif repo_realpath != source_realpath and not (
            activate_existing_local and repo_realpath == target.resolve(strict=False)
        ):
            entry_blockers.append("repo_route_not_source")
        if activate_existing_local and not target.exists():
            entry_blockers.append("existing_local_target_missing")
        elif activate_existing_local and _size_bytes(target) <= 0:
            entry_blockers.append("existing_local_target_empty")
        if activate_existing_local and require_prior_activation_proof:
            if not prior_proof_header_ok:
                entry_blockers.append("prior_verified_failover_header_missing")
            elif rel not in prior_entries:
                entry_blockers.append("prior_verified_failover_entry_missing")
        for suffix in ("-wal", "-shm"):
            sidecar_route = Path(f"{repo}{suffix}")
            if sidecar_route.exists() and not sidecar_route.is_symlink():
                entry_blockers.append(f"repo_sidecar_not_symlink:{suffix}")
        blockers.extend(f"{rel}:{reason}" for reason in entry_blockers)
        entries.append(
            {
                "relative_path": rel,
                "source": str(source),
                "target": str(target),
                "repo": str(repo),
                "temp": str(temp),
                "source_size_bytes": int(source_size),
                "repo_route": repo_route,
                "blockers": entry_blockers,
                "stage": {},
                "commit": {},
            }
        )

    stage_headroom_bytes = 0 if activate_existing_local else max(int(required_stage_bytes * 0.10), 1024**3)
    required_with_headroom = int(required_stage_bytes + stage_headroom_bytes)
    projected_free_after = None if target_free_before is None else int(target_free_before - required_with_headroom)
    required_free_before_stage = int(required_with_headroom + max(int(min_free_after_bytes), 0))
    source_release_deficit = (
        None
        if target_free_before is None
        else max(required_free_before_stage - int(target_free_before), 0)
    )
    if source_root == target_root:
        blockers.append("source_and_target_roots_match")
    if not target_root.exists() or not os.access(target_root, os.W_OK):
        blockers.append("target_root_not_writable")
    if require_writer_idle and _writer_is_active(writer_snapshot):
        blockers.append("writer_not_idle")
    if require_runtime_stopped and runtime_processes:
        blockers.append("runtime_processes_active")
    if require_maintenance_hold and not bool(maintenance_hold.get("active", False)):
        blockers.append("runtime_maintenance_hold_inactive")
    if target_free_before is None:
        blockers.append("target_free_space_unknown")
    elif projected_free_after is not None and projected_free_after < max(int(min_free_after_bytes), 0):
        blockers.append("target_free_after_stage_below_reserve")

    payload: dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "run_id": run_id,
        "ok": False,
        "overall_status": "blocked" if blockers else "ready_to_apply",
        "apply": bool(apply),
        "project_root": str(project_root),
        "source_root": str(source_root),
        "target_root": str(target_root),
        "relative_paths": relative_paths,
        "entries": entries,
        "writer_state": writer_snapshot,
        "runtime_processes": runtime_processes,
        "runtime_maintenance_hold": maintenance_hold,
        "require_writer_idle": bool(require_writer_idle),
        "require_runtime_stopped": bool(require_runtime_stopped),
        "require_maintenance_hold": bool(require_maintenance_hold),
        "preserve_existing": bool(preserve_existing),
        "activate_existing_local": bool(activate_existing_local),
        "require_prior_activation_proof": bool(require_prior_activation_proof),
        "prior_proof_path": str(prior_proof_path),
        "prior_proof_header_ok": prior_proof_header_ok,
        "storage_override_path": str(storage_override_path),
        "storage_override": {"installed": False, "rolled_back": False},
        "target_free_bytes_before": target_free_before,
        "required_stage_bytes": int(required_stage_bytes),
        "stage_headroom_bytes": int(stage_headroom_bytes),
        "required_with_headroom_bytes": required_with_headroom,
        "projected_free_after_stage_bytes": projected_free_after,
        "min_free_after_bytes": int(min_free_after_bytes),
        "required_free_before_stage_bytes": required_free_before_stage,
        "cold_archive_recommendation": {
            "needed": bool((source_release_deficit or 0) > 0),
            "required_source_release_bytes": source_release_deficit,
            "required_source_release_gb": round(float(source_release_deficit or 0) / float(1024**3), 3),
            "source_free_target_gb": round(float(required_free_before_stage) / float(1024**3), 3),
            "policy": "demand_driven_smallest_verified_noncritical_set",
        },
        "blockers": blockers,
        "current_database": "",
        "progress": {},
        "route_switch": {"attempted": False, "ok": False, "rolled_back": False},
        "lock_path": str(lock_path),
    }
    if blockers:
        return payload
    if not apply:
        payload["ok"] = True
        payload["overall_status"] = "activation_dry_run_ready" if activate_existing_local else "dry_run_ready"
        return payload

    lock_handle = _acquire_lock(lock_path)
    if lock_handle is None:
        payload["blockers"].append("failover_lock_busy")
        payload["overall_status"] = "blocked"
        return payload

    last_progress_write = 0.0

    def publish_progress(*, force: bool = False) -> None:
        nonlocal last_progress_write
        now = time.monotonic()
        if progress_writer is not None and (force or now - last_progress_write >= 5.0):
            payload["timestamp_utc"] = _utc_now()
            progress_writer(payload)
            last_progress_write = now

    staged_temps: list[Path] = []
    try:
        payload["overall_status"] = "verifying_existing_local" if activate_existing_local else "staging"
        publish_progress(force=True)
        for entry in entries:
            rel = str(entry["relative_path"])
            source = Path(str(entry["source"]))
            temp = Path(str(entry["temp"]))
            payload["current_database"] = rel

            def on_backup_progress(status: int, remaining: int, total: int) -> None:
                completed = max(int(total) - int(remaining), 0)
                payload["progress"] = {
                    "sqlite_status": int(status),
                    "remaining_pages": int(remaining),
                    "total_pages": int(total),
                    "completed_pages": completed,
                    "percent": round((completed / max(int(total), 1)) * 100.0, 3),
                }
                publish_progress()

            try:
                if activate_existing_local:
                    stage = _verify_existing_database(
                        source,
                        Path(str(entry["target"])),
                        timeout_seconds=timeout_seconds,
                        prior_entry=prior_entries.get(rel),
                        require_prior_proof=require_prior_activation_proof,
                    )
                else:
                    stage = _stage_database(
                        source,
                        temp,
                        page_batch=page_batch,
                        timeout_seconds=timeout_seconds,
                        progress=on_backup_progress,
                    )
            except Exception as exc:
                stage = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
            entry["stage"] = stage
            if not activate_existing_local and temp.exists():
                staged_temps.append(temp)
            publish_progress(force=True)
            if not bool(stage.get("ok", False)):
                payload["blockers"].append(f"{rel}:stage_verification_failed")
                payload["overall_status"] = "stage_failed"
                return payload

        writer_after_stage = writer_state.writer_state_snapshot(project_root)
        runtime_after_stage = _runtime_processes()
        payload["writer_state_after_stage"] = writer_after_stage
        payload["runtime_processes_after_stage"] = runtime_after_stage
        if require_writer_idle and _writer_is_active(writer_after_stage):
            payload["blockers"].append("writer_started_during_stage")
        if require_runtime_stopped and runtime_after_stage:
            payload["blockers"].append("runtime_started_during_stage")
        if payload["blockers"]:
            payload["overall_status"] = "quiescence_lost"
            return payload

        payload["overall_status"] = "activating_existing_local" if activate_existing_local else "committing"
        publish_progress(force=True)
        if activate_existing_local:
            for entry in entries:
                target = Path(str(entry["target"]))
                entry["commit"] = {
                    "target_installed": True,
                    "existing_target_reused": True,
                    "preserved_existing": [],
                    "target_size_bytes": _size_bytes(target),
                }
        else:
            for entry in entries:
                target = Path(str(entry["target"]))
                temp = Path(str(entry["temp"]))
                target.parent.mkdir(parents=True, exist_ok=True)
                preserved: list[str] = []
                for candidate in (target, Path(f"{target}-wal"), Path(f"{target}-shm")):
                    if not candidate.exists() and not candidate.is_symlink():
                        continue
                    if preserve_existing:
                        backup = candidate.with_name(f"{candidate.name}.pre_local_failover_{run_id}.bak")
                        os.replace(candidate, backup)
                        preserved.append(str(backup))
                    else:
                        candidate.unlink()
                os.replace(temp, target)
                entry["commit"] = {
                    "target_installed": True,
                    "preserved_existing": preserved,
                    "target_size_bytes": _size_bytes(target),
                }

        try:
            override_result = _install_local_storage_override(storage_override_path)
            payload["storage_override"] = override_result
        except Exception as exc:
            payload["blockers"].append(f"storage_override_install_failed:{type(exc).__name__}:{exc}")
            payload["overall_status"] = "storage_override_install_failed"
            return payload

        route_snapshots: dict[str, dict[str, Any]] = {}
        route_paths: list[tuple[Path, Path]] = []
        for entry in entries:
            repo = Path(str(entry["repo"]))
            target = Path(str(entry["target"]))
            route_paths.append((repo, target))
            route_paths.extend(
                (Path(f"{repo}{suffix}"), Path(f"{target}{suffix}"))
                for suffix in ("-wal", "-shm")
            )
        for link, _ in route_paths:
            route_snapshots[str(link)] = _route_snapshot(link)

        payload["route_switch"]["attempted"] = True
        switched: list[str] = []
        try:
            for link, target in route_paths:
                _atomic_symlink(link, target)
                switched.append(str(link))
            mismatches = [
                str(link)
                for link, target in route_paths
                if not link.is_symlink() or link.resolve(strict=False) != target.resolve(strict=False)
            ]
            if mismatches:
                raise RuntimeError(f"route verification failed: {','.join(mismatches)}")
        except Exception as exc:
            rollback_errors: list[str] = []
            for link, _ in reversed(route_paths):
                try:
                    _restore_route(link, route_snapshots[str(link)])
                except Exception as rollback_exc:
                    rollback_errors.append(f"{link}:{type(rollback_exc).__name__}:{rollback_exc}")
            payload["route_switch"].update(
                {
                    "ok": False,
                    "rolled_back": True,
                    "error": f"{type(exc).__name__}:{exc}",
                    "rollback_errors": rollback_errors,
                    "switched_before_rollback": switched,
                }
            )
            try:
                _restore_text_file(storage_override_path, override_result.get("before") or {})
                payload["storage_override"]["rolled_back"] = True
            except Exception as override_exc:
                payload["storage_override"]["rollback_error"] = f"{type(override_exc).__name__}:{override_exc}"
            payload["overall_status"] = "route_switch_failed"
            return payload

        payload["route_switch"].update(
            {
                "ok": True,
                "rolled_back": False,
                "switched_paths": switched,
            }
        )
        payload["target_free_bytes_after"] = _disk_free_bytes(target_root)
        payload["current_database"] = ""
        payload["progress"] = {}
        payload["ok"] = True
        payload["overall_status"] = (
            "activated_existing_local_verified"
            if activate_existing_local
            else "switched_local_verified"
        )
        return payload
    finally:
        if not bool(payload.get("ok", False)):
            for temp in staged_temps:
                for candidate in (temp, Path(f"{temp}-wal"), Path(f"{temp}-shm")):
                    try:
                        if candidate.exists() or candidate.is_symlink():
                            candidate.unlink()
                    except Exception:
                        pass
        payload["timestamp_utc"] = _utc_now()
        publish_progress(force=True)
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transactionally back up active external SQLite databases to local fallback and switch repo routes."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--source-root", default="")
    parser.add_argument("--target-root", default="")
    parser.add_argument("--relative-path", action="append", default=[])
    parser.add_argument("--min-free-after-gb", type=float, default=float(os.getenv("BOT_LOGS_LOCAL_SQLITE_MIN_FREE_AFTER_GB", "50")))
    parser.add_argument("--page-batch", type=int, default=int(os.getenv("BOT_LOGS_LOCAL_SQLITE_BACKUP_PAGE_BATCH", "4096")))
    parser.add_argument("--sqlite-timeout-seconds", type=float, default=float(os.getenv("BOT_LOGS_LOCAL_SQLITE_TIMEOUT_SECONDS", "900")))
    parser.add_argument("--no-require-writer-idle", action="store_true")
    parser.add_argument("--no-require-runtime-stopped", action="store_true")
    parser.add_argument("--no-require-maintenance-hold", action="store_true")
    parser.add_argument("--no-preserve-existing", action="store_true")
    parser.add_argument("--activate-existing-local", action="store_true")
    parser.add_argument("--no-require-prior-activation-proof", action="store_true")
    parser.add_argument("--prior-proof-file", default="")
    parser.add_argument("--storage-override-file", default="")
    parser.add_argument("--lock-file", default="")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    source_root = (
        Path(args.source_root).expanduser().resolve()
        if str(args.source_root or "").strip()
        else resolve_external_storage().external_root.resolve()
    )
    target_root = (
        Path(args.target_root).expanduser().resolve()
        if str(args.target_root or "").strip()
        else (project_root / "local_fallback_storage").resolve()
    )
    out_file = Path(args.out_file).expanduser()
    relative_paths = list(args.relative_path or TRACKED_DATABASES)

    def progress_writer(payload: dict[str, Any]) -> None:
        _atomic_write_json(out_file, payload)

    payload = build_payload(
        project_root,
        source_root=source_root,
        target_root=target_root,
        relative_paths=relative_paths,
        apply=bool(args.apply),
        min_free_after_bytes=int(max(float(args.min_free_after_gb), 0.0) * (1024**3)),
        page_batch=max(int(args.page_batch), 1),
        timeout_seconds=max(float(args.sqlite_timeout_seconds), 1.0),
        require_writer_idle=not bool(args.no_require_writer_idle),
        require_runtime_stopped=not bool(args.no_require_runtime_stopped),
        require_maintenance_hold=not bool(args.no_require_maintenance_hold),
        preserve_existing=not bool(args.no_preserve_existing),
        activate_existing_local=bool(args.activate_existing_local),
        require_prior_activation_proof=not bool(args.no_require_prior_activation_proof),
        prior_proof_path=Path(args.prior_proof_file).expanduser() if str(args.prior_proof_file or "").strip() else None,
        storage_override_path=Path(args.storage_override_file).expanduser() if str(args.storage_override_file or "").strip() else None,
        lock_path=Path(args.lock_file).expanduser() if str(args.lock_file or "").strip() else None,
        progress_writer=progress_writer,
    )
    _atomic_write_json(out_file, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "storage_sqlite_local_failover "
            f"status={payload.get('overall_status')} "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"databases={len(payload.get('entries') or [])}"
        )
    return 0 if bool(payload.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
