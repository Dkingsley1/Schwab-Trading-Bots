#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_python import resolve_runtime_python


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "storage_standby_prune_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "storage_standby_prune.lock"
DEFAULT_ROUTE_SOAK_HOURS = float(os.getenv("BOT_LOGS_STANDBY_PRUNE_MIN_ROUTE_SOAK_HOURS", "2"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts_utc(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _run_failback_sync(project_root: Path) -> dict[str, Any]:
    cmd = [str(PY), str(project_root / "scripts" / "ops" / "storage_failback_sync.py"), "--json"]
    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = _parse_json_output(proc.stdout)
    return {
        "rc": int(proc.returncode),
        "payload": payload if isinstance(payload, dict) else {},
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-12:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-12:]),
    }


def _route_soak_summary(project_root: Path, *, min_route_soak_hours: float) -> dict[str, Any]:
    switch_payload = _load_json(project_root / "governance" / "health" / "storage_switch_orchestrator_latest.json")
    switch_timestamp = str(switch_payload.get("timestamp_utc") or "")
    dt = _parse_ts_utc(switch_timestamp)
    if dt is None:
        return {
            "source": "unknown",
            "switch_timestamp_utc": "",
            "age_hours": None,
            "min_route_soak_hours": float(max(min_route_soak_hours, 0.0)),
            "ok": False,
            "reason": "No external storage switch artifact timestamp is available yet.",
        }

    age_hours = round(max((datetime.now(timezone.utc) - dt).total_seconds() / 3600.0, 0.0), 6)
    ok = age_hours >= float(max(min_route_soak_hours, 0.0))
    return {
        "source": "storage_switch_orchestrator_latest",
        "switch_timestamp_utc": switch_timestamp,
        "age_hours": age_hours,
        "min_route_soak_hours": float(max(min_route_soak_hours, 0.0)),
        "ok": bool(ok),
        "reason": (
            "The external route has satisfied the standby prune soak window."
            if ok
            else "The external route switch is too recent to prune standby copies safely."
        ),
    }


def _file_size_bytes(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def _build_candidate_rows(
    failback_payload: dict[str, Any],
    *,
    include_curated_standby: bool,
    relative_paths: set[str],
    route_soak_ok: bool,
) -> tuple[list[dict[str, Any]], int]:
    sqlite_report = failback_payload.get("sqlite_skip_report")
    if not isinstance(sqlite_report, dict):
        return [], 0

    allowed_states = {"verified", "active_external_newer_than_standby"}
    if include_curated_standby:
        allowed_states.add("curated_standby")

    rows: list[dict[str, Any]] = []
    reclaimable_bytes_total = 0
    for raw in list(sqlite_report.get("entries") or []):
        if not isinstance(raw, dict):
            continue
        relative_path = str(raw.get("relative_path") or "").strip()
        if not relative_path:
            continue
        if relative_paths and relative_path not in relative_paths:
            continue

        local = raw.get("local")
        route_verification = raw.get("route_verification")
        if not isinstance(local, dict) or not isinstance(route_verification, dict):
            continue

        local_path_text = str(local.get("path") or "").strip()
        local_exists = bool(local.get("exists", False))
        classification = str(raw.get("classification") or "").strip()
        verification_state = str(route_verification.get("state") or "").strip()
        eligible_classification = classification in {"warm_standby_retained", "active_external_route"}
        eligible = (
            eligible_classification
            and local_exists
            and verification_state in allowed_states
            and route_soak_ok
        )

        delete_paths: list[str] = []
        reclaimable_bytes = 0
        if local_path_text:
            local_path = Path(local_path_text)
            if local_path.exists():
                delete_paths.append(str(local_path))
                reclaimable_bytes += _file_size_bytes(local_path)
                for sidecar_name in list(local.get("sidecars") or []):
                    sidecar_path = local_path.parent / str(sidecar_name)
                    if sidecar_path.exists():
                        delete_paths.append(str(sidecar_path))
                        reclaimable_bytes += _file_size_bytes(sidecar_path)

        if verification_state not in allowed_states:
            reason = "Standby pruning is limited to externally verified copies by default."
        elif not eligible_classification:
            reason = "This tracked path is not a retained local standby copy behind the external route."
        elif not local_exists:
            reason = "No retained local standby copy exists for this tracked path."
        elif not route_soak_ok:
            reason = "The external route has not soaked long enough yet for standby pruning."
        else:
            reason = "This retained local standby copy is eligible for guarded pruning."

        row = {
            "relative_path": relative_path,
            "classification": classification,
            "verification_state": verification_state,
            "eligible": bool(eligible),
            "reason": reason,
            "local_path": local_path_text,
            "delete_paths": delete_paths,
            "reclaimable_bytes": int(reclaimable_bytes),
        }
        rows.append(row)
        if eligible:
            reclaimable_bytes_total += int(reclaimable_bytes)

    rows.sort(key=lambda row: str(row.get("relative_path") or ""))
    return rows, reclaimable_bytes_total


def _delete_paths(paths: list[str]) -> tuple[int, int, list[str], int]:
    deleted_files = 0
    error_count = 0
    deleted_paths: list[str] = []
    reclaimed_bytes = 0
    for raw_path in paths:
        path = Path(str(raw_path)).expanduser()
        if not path.exists():
            continue
        reclaimed_bytes += _file_size_bytes(path)
        try:
            path.unlink()
            deleted_files += 1
            deleted_paths.append(str(path))
        except Exception:
            error_count += 1
    return deleted_files, error_count, deleted_paths, reclaimed_bytes


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool,
    include_curated_standby: bool = False,
    min_route_soak_hours: float = DEFAULT_ROUTE_SOAK_HOURS,
    relative_paths: set[str] | None = None,
    failback_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    relative_paths = {str(item).strip() for item in (relative_paths or set()) if str(item).strip()}

    failback_result = (
        {"rc": 0, "payload": failback_payload, "stdout_tail": "", "stderr_tail": ""}
        if isinstance(failback_payload, dict)
        else _run_failback_sync(project_root)
    )
    payload = failback_result.get("payload") if isinstance(failback_result.get("payload"), dict) else {}
    sqlite_report = payload.get("sqlite_skip_report") if isinstance(payload.get("sqlite_skip_report"), dict) else {}
    route_verification = sqlite_report.get("route_verification") if isinstance(sqlite_report.get("route_verification"), dict) else {}
    summary = sqlite_report.get("summary") if isinstance(sqlite_report.get("summary"), dict) else {}
    certified_mode = str(payload.get("certified_mode") or payload.get("mode") or "").strip()
    verification_state = str(route_verification.get("verification_state") or "").strip()
    verification_mismatches = list(route_verification.get("mismatches") or [])
    active_local_count = int(summary.get("active_local_count") or 0)
    active_external_count = int(summary.get("active_external_count") or 0)

    route_guard_ok = (
        int(failback_result.get("rc", 1)) == 0
        and certified_mode in {"external", "external_curated"}
        and verification_state in {"ready", "curated_ready"}
        and not verification_mismatches
        and active_local_count == 0
    )
    active_local_route_idle = bool(
        int(failback_result.get("rc", 1)) == 0
        and certified_mode in {"local_fallback", "local_fallback_split_brain"}
        and verification_state == "active_local_ready"
        and not verification_mismatches
        and active_local_count > 0
        and active_external_count == 0
    )
    route_guard_reason = "External route is certified for standby pruning."
    if int(failback_result.get("rc", 1)) != 0:
        route_guard_reason = "The live storage failback report could not be refreshed successfully."
    elif certified_mode not in {"external", "external_curated"}:
        route_guard_reason = "Standby pruning is only allowed when BOT_LOGS is the active external route."
    elif verification_state not in {"ready", "curated_ready"}:
        route_guard_reason = "The external route is not in a standby-prune-ready verification state."
    elif verification_mismatches:
        route_guard_reason = "The external route still has SQLite verification mismatches."
    elif active_local_count > 0:
        route_guard_reason = "A tracked SQLite path is still actively pinned to the local fallback root."
    if active_local_route_idle:
        route_guard_reason = "Standby pruning is safely not applicable while the verified local fallback route is active."

    route_soak = _route_soak_summary(project_root, min_route_soak_hours=min_route_soak_hours)
    candidate_rows, reclaimable_bytes_total = _build_candidate_rows(
        payload,
        include_curated_standby=include_curated_standby,
        relative_paths=relative_paths,
        route_soak_ok=bool(route_guard_ok and route_soak.get("ok", False)),
    )
    eligible_rows = [row for row in candidate_rows if bool(row.get("eligible", False))]

    deleted_files = 0
    delete_errors = 0
    deleted_paths: list[str] = []
    reclaimed_bytes = 0
    if apply and eligible_rows:
        delete_targets: list[str] = []
        for row in eligible_rows:
            delete_targets.extend([str(path) for path in list(row.get("delete_paths") or []) if str(path).strip()])
        deleted_files, delete_errors, deleted_paths, reclaimed_bytes = _delete_paths(delete_targets)
        if deleted_files > 0 and delete_errors == 0 and failback_payload is None:
            failback_result = _run_failback_sync(project_root)
            payload = failback_result.get("payload") if isinstance(failback_result.get("payload"), dict) else payload

    if active_local_route_idle:
        overall_status = "ready_idle_active_local_route"
    elif not route_guard_ok:
        overall_status = "blocked_by_route_guard"
    elif not bool(route_soak.get("ok", False)):
        overall_status = "deferred_route_soak"
    elif not eligible_rows:
        overall_status = "no_eligible_standby"
    elif not apply:
        overall_status = "dry_run"
    elif delete_errors:
        overall_status = "partial_prune"
    else:
        overall_status = "pruned"

    return {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "ok": overall_status in {"dry_run", "pruned", "no_eligible_standby", "ready_idle_active_local_route"}
        or (overall_status == "deferred_route_soak" and route_guard_ok),
        "overall_status": overall_status,
        "apply": bool(apply),
        "include_curated_standby": bool(include_curated_standby),
        "min_route_soak_hours": float(max(min_route_soak_hours, 0.0)),
        "route_guard": {
            "ok": bool(route_guard_ok),
            "reason": route_guard_reason,
            "certified_mode": certified_mode,
            "verification_state": verification_state,
            "verification_mismatches": verification_mismatches,
            "active_local_count": active_local_count,
            "active_external_count": active_external_count,
            "active_local_route_idle": active_local_route_idle,
        },
        "route_soak": route_soak,
        "filters": {
            "relative_paths": sorted(relative_paths),
        },
        "summary": {
            "candidate_count": len(candidate_rows),
            "eligible_count": len(eligible_rows),
            "reclaimable_bytes_total": int(reclaimable_bytes_total),
            "deleted_files": int(deleted_files),
            "deleted_paths_count": len(deleted_paths),
            "delete_errors": int(delete_errors),
            "reclaimed_bytes": int(reclaimed_bytes),
        },
        "candidates": candidate_rows,
        "deleted_paths": deleted_paths,
        "failback_sync": {
            "rc": int(failback_result.get("rc", 1)),
            "stdout_tail": str(failback_result.get("stdout_tail") or ""),
            "stderr_tail": str(failback_result.get("stderr_tail") or ""),
            "payload": payload,
        },
        "automation_contract": {
            "launchd_label": "com.dankingsley.ops.storage_standby_prune",
            "run_command": "./scripts/ops/opsctl.sh storage-prune-standby --apply --json",
            "interval_seconds": max(int(float(os.getenv("BOT_LOGS_STANDBY_PRUNE_INTERVAL_SECONDS", "300") or 300)), 60),
            "enabled_by_default": True,
            "default_min_route_soak_hours": float(max(min_route_soak_hours, 0.0)),
        },
        "upgrade_track": {
            "upgradeable": True,
            "family": "infrabots",
            "install_command": "./scripts/ops/install_ops_automation_launchd.sh",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Guardedly prune retained local BOT_LOGS standby SQLite copies after the external route has soaked.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--include-curated-standby", action="store_true")
    parser.add_argument("--min-route-soak-hours", type=float, default=DEFAULT_ROUTE_SOAK_HOURS)
    parser.add_argument("--relative-path", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    out_file = Path(args.out_file).expanduser()
    lock_path = Path(args.lock_file).expanduser()

    lock_fh, lock_owner = _acquire_singleton_lock(lock_path)
    if lock_fh is None:
        payload = {
            "timestamp_utc": _utc_now(),
            "schema_version": 1,
            "ok": True,
            "overall_status": "lock_busy",
            "lock_path": str(lock_path),
            "lock_owner": lock_owner,
        }
        _write_json(out_file, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(f"storage_standby_prune_status=lock_busy lock_path={lock_path} owner={lock_owner}")
        return 0

    try:
        payload = build_payload(
            project_root=project_root,
            apply=bool(args.apply),
            include_curated_standby=bool(args.include_curated_standby),
            min_route_soak_hours=float(max(args.min_route_soak_hours, 0.0)),
            relative_paths={str(item).strip() for item in list(args.relative_path or []) if str(item).strip()},
        )
        _write_json(out_file, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
            print(
                "storage_standby_prune_status="
                f"{payload.get('overall_status')} eligible={summary.get('eligible_count', 0)} "
                f"deleted={summary.get('deleted_paths_count', 0)}"
            )
        return 0 if bool(payload.get("ok", False)) else 1
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
