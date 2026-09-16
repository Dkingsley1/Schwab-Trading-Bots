#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from .long_runtime_common import write_payload


DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "local_sql_shard_standby_prune_latest.json"
)
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "local_sql_shard_standby_prune.lock"
DEFAULT_MANIFEST_ROOT = PROJECT_ROOT / "governance" / "storage_recovery"
GIB = 1024**3
SQL_SHARD_RE = re.compile(r"^jsonl_link_[A-Za-z0-9_]+\.sqlite3(?:-(?:wal|shm))?$")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _external_project_root(project_root: Path, raw: str = "") -> Path:
    if raw:
        return Path(raw).expanduser()
    env_root = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser()
    mount = Path(os.getenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS")).expanduser()
    project_dir = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", "schwab_trading_bot").strip()
    return mount / (project_dir or project_root.name)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except Exception:
        return str(left) == str(right)


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except Exception:
        return False


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _base_sqlite_name(name: str) -> str:
    if name.endswith("-wal"):
        return name[:-4]
    if name.endswith("-shm"):
        return name[:-4]
    return name


def _stateful_route_ready(project_root: Path) -> tuple[bool, list[str], dict[str, Any]]:
    payload = _load_json(
        project_root / "governance" / "health" / "stateful_storage_regression_guard_latest.json"
    )
    if not payload:
        return False, ["stateful_storage_regression_guard_missing"], {}
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    shard_check = next(
        (
            row
            for row in checks
            if isinstance(row, dict) and str(row.get("name") or "") == "sql_link_shards"
        ),
        {},
    )
    blockers: list[str] = []
    if str(payload.get("stateful_target_mode") or "") not in {"external", "external_curated"}:
        blockers.append("stateful_target_not_external")
    if str(shard_check.get("status") or "") != "ready":
        blockers.append("sql_link_shard_route_not_ready")
    if not bool(shard_check.get("target_match", False)):
        blockers.append("sql_link_shard_route_target_mismatch")
    return not blockers, blockers, {"payload": payload, "sql_link_shards": shard_check}


def _failback_route_ready(project_root: Path) -> tuple[bool, list[str], dict[str, Any]]:
    payload = _load_json(project_root / "governance" / "health" / "storage_failback_sync_latest.json")
    if not payload:
        return False, ["storage_failback_sync_missing"], {}
    route = payload.get("route_verification") if isinstance(payload.get("route_verification"), dict) else {}
    sqlite_report = payload.get("sqlite_skip_report") if isinstance(payload.get("sqlite_skip_report"), dict) else {}
    if not route:
        route = (
            sqlite_report.get("route_verification")
            if isinstance(sqlite_report.get("route_verification"), dict)
            else {}
        )
    certified_mode = str(payload.get("certified_mode") or payload.get("mode") or "")
    verification_state = str(route.get("verification_state") or "")
    mismatches = route.get("mismatches") if isinstance(route.get("mismatches"), list) else []
    blockers: list[str] = []
    if certified_mode not in {"external", "external_curated"}:
        blockers.append("storage_failback_mode_not_external")
    if verification_state not in {"ready", "curated_ready"}:
        blockers.append("storage_failback_route_not_ready")
    if mismatches:
        blockers.append("storage_failback_route_mismatch")
    return not blockers, blockers, {"payload": payload, "route_verification": route}


def _open_handles(path: Path, timeout_seconds: float) -> dict[str, Any]:
    if os.getenv("LOCAL_SQL_SHARD_STANDBY_PRUNE_CHECK_OPEN_HANDLES", "1").strip() in {
        "0",
        "false",
        "False",
        "no",
    }:
        return {"checked": False, "open": False, "rows": [], "reason": "disabled_by_env"}
    try:
        proc = subprocess.run(
            ["lsof", "+D", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(float(timeout_seconds), 1.0),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"checked": True, "open": True, "rows": [], "reason": "lsof_timeout"}
    except OSError as exc:
        return {
            "checked": True,
            "open": True,
            "rows": [],
            "reason": f"lsof_failed:{type(exc).__name__}:{exc}",
        }
    rows = [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode == 1 and not rows:
        return {"checked": True, "open": False, "rows": [], "reason": "none"}
    data_rows = rows[1:] if rows and rows[0].startswith("COMMAND") else rows
    return {
        "checked": True,
        "open": bool(data_rows),
        "rows": data_rows[:20],
        "reason": "open_handles" if data_rows else "none",
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-500:],
    }


def _candidate_rows(
    local_root: Path,
    active_root: Path,
    *,
    require_external_counterpart: bool,
    min_age_minutes: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).timestamp()
    try:
        rows = sorted(local_root.iterdir(), key=lambda item: item.name)
    except OSError:
        return [], []
    for path in rows:
        if not path.is_file() or path.is_symlink():
            continue
        if not SQL_SHARD_RE.match(path.name):
            skipped.append({"path": str(path), "reason": "name_not_sql_shard_cache"})
            continue
        base_name = _base_sqlite_name(path.name)
        external_base = active_root / base_name
        external_exists = external_base.exists()
        if require_external_counterpart and not external_exists:
            skipped.append(
                {
                    "path": str(path),
                    "reason": "external_active_counterpart_missing",
                    "external_counterpart": str(external_base),
                }
            )
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            skipped.append({"path": str(path), "reason": "stat_failed"})
            continue
        age_minutes = max((now - float(mtime)) / 60.0, 0.0)
        if age_minutes < min_age_minutes:
            skipped.append(
                {
                    "path": str(path),
                    "reason": "younger_than_min_age",
                    "age_minutes": round(age_minutes, 3),
                    "min_age_minutes": float(min_age_minutes),
                }
            )
            continue
        candidates.append(
            {
                "path": str(path),
                "name": path.name,
                "base_name": base_name,
                "size_bytes": _file_size(path),
                "mtime_utc": datetime.fromtimestamp(mtime, timezone.utc).isoformat(),
                "age_minutes": round(age_minutes, 3),
                "external_counterpart": str(external_base),
                "external_counterpart_exists": bool(external_exists),
                "external_counterpart_size_bytes": _file_size(external_base),
            }
        )
    return candidates, skipped


def _select_with_budget(
    candidates: list[dict[str, Any]], max_delete_gb: float
) -> list[dict[str, Any]]:
    budget = int(max(float(max_delete_gb), 0.0) * GIB)
    if budget <= 0:
        return []
    selected: list[dict[str, Any]] = []
    used = 0
    for row in sorted(candidates, key=lambda item: _safe_int(item.get("size_bytes")), reverse=True):
        size = _safe_int(row.get("size_bytes"))
        if size <= 0:
            continue
        if used + size > budget:
            continue
        selected.append(row)
        used += size
    return selected


def _write_manifest(payload: dict[str, Any]) -> str:
    run_id = str(payload.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    manifest = DEFAULT_MANIFEST_ROOT / f"local_sql_shard_standby_prune_{run_id}.json"
    write_payload(manifest, payload)
    return str(manifest)


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    external_root: str = "",
    apply: bool = False,
    max_delete_gb: float = 512.0,
    min_age_minutes: float = 0.0,
    require_external_counterpart: bool = True,
    lsof_timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    external = _external_project_root(project_root, external_root)
    active_link = project_root / "data" / "sql_link_shards"
    local_root = project_root / "local_fallback_storage" / "data" / "sql_link_shards"
    active_root = active_link.resolve(strict=False)
    external_shard_root = (external / "data" / "sql_link_shards").resolve(strict=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")

    blockers: list[str] = []
    warnings: list[str] = []
    if not local_root.exists():
        warnings.append("local_fallback_sql_shard_root_missing")
    if not active_link.exists():
        blockers.append("active_sql_shard_route_missing")
    if _same_path(active_root, local_root):
        blockers.append("active_route_is_local_fallback")
    if not _same_path(active_root, external_shard_root):
        blockers.append("active_route_not_configured_external_shard_root")
    if not _is_under(active_root, external):
        blockers.append("active_route_not_under_external_root")
    if not external_shard_root.exists():
        blockers.append("external_sql_shard_root_missing")

    stateful_ok, stateful_blockers, stateful_context = _stateful_route_ready(project_root)
    failback_ok, failback_blockers, failback_context = _failback_route_ready(project_root)
    if not stateful_ok:
        blockers.extend(stateful_blockers)
    if not failback_ok:
        blockers.extend(failback_blockers)

    handles = _open_handles(local_root, lsof_timeout_seconds) if local_root.exists() else {}
    if bool(handles.get("open", False)):
        blockers.append(str(handles.get("reason") or "open_handles_present"))

    candidates, skipped = (
        _candidate_rows(
            local_root,
            active_root,
            require_external_counterpart=require_external_counterpart,
            min_age_minutes=max(float(min_age_minutes), 0.0),
        )
        if local_root.exists()
        else ([], [])
    )
    selected = _select_with_budget(candidates, max_delete_gb)
    selected_bytes = sum(_safe_int(row.get("size_bytes")) for row in selected)
    candidate_bytes = sum(_safe_int(row.get("size_bytes")) for row in candidates)

    delete_errors: list[dict[str, str]] = []
    deleted_rows: list[dict[str, Any]] = []
    manifest_path = ""
    if apply and not blockers:
        manifest_payload = {
            "timestamp_utc": iso_now(),
            "schema_version": 1,
            "run_id": run_id,
            "project_root": str(project_root),
            "active_sql_shard_root": str(active_root),
            "local_standby_sql_shard_root": str(local_root),
            "selected": selected,
            "policy": "inactive_derived_sqlite_shard_cache_only; active route must be external; raw JSONL evidence is untouched",
        }
        manifest_path = _write_manifest(manifest_payload)
        for row in selected:
            path = Path(str(row.get("path") or ""))
            try:
                path.unlink()
            except OSError as exc:
                delete_errors.append({"path": str(path), "error": str(exc)})
                continue
            deleted = dict(row)
            deleted["deleted"] = True
            deleted_rows.append(deleted)

    deleted_bytes = sum(_safe_int(row.get("size_bytes")) for row in deleted_rows)
    if blockers:
        overall_status = "blocked"
    elif apply and delete_errors:
        overall_status = "degraded"
    elif apply and deleted_rows:
        overall_status = "applied"
    elif candidates:
        overall_status = "ready"
    else:
        overall_status = "no_candidates"

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status in {"ready", "applied", "no_candidates"},
        "overall_status": overall_status,
        "apply": bool(apply),
        "run_id": run_id,
        "project_root": str(project_root),
        "active_sql_shard_link": str(active_link),
        "active_sql_shard_root": str(active_root),
        "external_sql_shard_root": str(external_shard_root),
        "local_standby_sql_shard_root": str(local_root),
        "route_contract": {
            "active_route_external": (
                "active_route_not_under_external_root" not in blockers
                and "active_route_is_local_fallback" not in blockers
            ),
            "require_external_counterpart": bool(require_external_counterpart),
            "stateful_route_ready": bool(stateful_ok),
            "failback_route_ready": bool(failback_ok),
        },
        "stateful_route_context": stateful_context,
        "failback_route_context": failback_context,
        "open_handle_check": handles,
        "candidate_count": len(candidates),
        "candidate_bytes": int(candidate_bytes),
        "candidate_gb": round(candidate_bytes / GIB, 3),
        "selected_count": len(selected),
        "selected_bytes": int(selected_bytes),
        "selected_gb": round(selected_bytes / GIB, 3),
        "deleted_count": len(deleted_rows),
        "deleted_bytes": int(deleted_bytes),
        "deleted_gb": round(deleted_bytes / GIB, 3),
        "manifest_path": manifest_path,
        "delete_errors": delete_errors,
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:20],
        "selected": selected[:50],
        "deleted": deleted_rows[:50],
        "blockers": blockers,
        "warnings": warnings,
        "policy": "Prune only inactive local_fallback SQL shard cache files after external shard routing is certified; never delete raw JSONL evidence or active SQLite routes.",
        "recommended_actions": [
            "run stateful-storage-regression-guard before this pruner so active shard routing is certified",
            "run local-storage-reserve-guard after this pruner so local reserve, ingestion storage, and dashboard gates rescore",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prune inactive local_fallback SQL shard cache files after the active shard route is external."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--external-root", default="")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-delete-gb", type=float, default=512.0)
    parser.add_argument("--min-age-minutes", type=float, default=0.0)
    parser.add_argument("--allow-unmirrored", action="store_true")
    parser.add_argument("--lsof-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    lock_path = Path(args.lock_file).expanduser()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            payload = {
                "timestamp_utc": iso_now(),
                "schema_version": 1,
                "ok": False,
                "overall_status": "busy",
                "apply": bool(args.apply),
                "lock_path": str(lock_path),
            }
            write_payload(Path(args.out_file).expanduser(), payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("local_sql_shard_standby_prune overall_status=busy")
            return 2

        payload = build_payload(
            Path(args.project_root),
            external_root=str(args.external_root or ""),
            apply=bool(args.apply),
            max_delete_gb=float(args.max_delete_gb),
            min_age_minutes=float(args.min_age_minutes),
            require_external_counterpart=not bool(args.allow_unmirrored),
            lsof_timeout_seconds=float(args.lsof_timeout_seconds),
        )
        payload["lock_path"] = str(lock_path)
        write_payload(Path(args.out_file).expanduser(), payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "local_sql_shard_standby_prune "
            f"overall_status={payload.get('overall_status', '')} "
            f"selected_gb={payload.get('selected_gb', 0.0)} "
            f"deleted_gb={payload.get('deleted_gb', 0.0)}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
