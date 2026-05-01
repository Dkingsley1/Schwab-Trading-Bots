#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from .long_runtime_common import iso_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "stateful_storage_regression_guard_latest.json"
DEFAULT_LAUNCHD_LOG_ROOT = Path("/tmp/schwab_trading_bot/launchd_ops")
SQL_WRITER_PLIST = Path.home() / "Library" / "LaunchAgents" / "com.dankingsley.ops.sql_link_writer.plist"
TEXT_MERGE_SUFFIXES = {".jsonl", ".log", ".txt"}


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _external_project_root(project_root: Path, raw: str = "") -> Path:
    if raw:
        return Path(raw).expanduser()
    env_root = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser()
    mount = Path(os.getenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS")).expanduser()
    project_dir = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", "schwab_trading_bot").strip() or project_root.name
    return mount / project_dir


def _path_size_bytes(path: Path) -> int:
    try:
        if path.is_symlink():
            return 0
        if path.is_file():
            return int(path.stat().st_size)
        if not path.exists():
            return 0
        total = 0
        for child in path.rglob("*"):
            try:
                if child.is_file() and not child.is_symlink():
                    total += int(child.stat().st_size)
            except Exception:
                continue
        return total
    except Exception:
        return 0


def _has_open_handles(path: Path) -> bool:
    if not path.exists() or not _env_flag("STATEFUL_STORAGE_REGRESSION_CHECK_OPEN_HANDLES", "1"):
        return False
    try:
        proc = subprocess.run(
            ["lsof", "+D", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
    except Exception:
        return False
    return proc.returncode == 0 and bool((proc.stdout or "").strip())


def _active_process(patterns: tuple[str, ...]) -> bool:
    try:
        proc = subprocess.run(["ps", "ax", "-o", "command="], capture_output=True, text=True, check=False, timeout=5)
    except Exception:
        return False
    text = proc.stdout or ""
    return any(pattern in text for pattern in patterns)


def _same_target(link: Path, target: Path) -> bool:
    try:
        return link.is_symlink() and link.resolve(strict=False) == target.resolve(strict=False)
    except Exception:
        return False


def _append_then_unlink(src: Path, dest: Path, actions: list[dict[str, Any]]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("ab") as out_handle, src.open("rb") as in_handle:
        if dest.stat().st_size > 0:
            out_handle.write(b"\n")
        shutil.copyfileobj(in_handle, out_handle)
    actions.append({"action": "append_merge", "source": str(src), "target": str(dest)})
    src.unlink()


def _move_conflict(src: Path, target_root: Path, actions: list[dict[str, Any]]) -> None:
    conflict_root = target_root / "_local_conflicts" / iso_now().replace(":", "").replace("+", "_")
    conflict_root.mkdir(parents=True, exist_ok=True)
    dest = conflict_root / src.name
    shutil.move(str(src), str(dest))
    actions.append({"action": "move_conflict", "source": str(src), "target": str(dest)})


def _merge_path(src: Path, dest: Path, target_root: Path, actions: list[dict[str, Any]]) -> None:
    if src.is_symlink():
        actions.append({"action": "skip_symlink", "source": str(src), "target": os.readlink(src)})
        return
    if src.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        for child in sorted(src.iterdir()):
            _merge_path(child, dest / child.name, target_root, actions)
        try:
            src.rmdir()
            actions.append({"action": "remove_empty_dir", "path": str(src)})
        except OSError:
            pass
        return
    if not src.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.move(str(src), str(dest))
        actions.append({"action": "move", "source": str(src), "target": str(dest)})
        return
    try:
        same_size = src.stat().st_size == dest.stat().st_size
    except Exception:
        same_size = False
    if same_size:
        src.unlink()
        actions.append({"action": "remove_duplicate", "source": str(src), "target": str(dest)})
    elif src.suffix in TEXT_MERGE_SUFFIXES and dest.suffix in TEXT_MERGE_SUFFIXES:
        _append_then_unlink(src, dest, actions)
    else:
        _move_conflict(src, target_root, actions)


def _repair_stateful_path(
    *,
    name: str,
    local: Path,
    target: Path,
    apply: bool,
    max_local_bytes: int,
    active_patterns: tuple[str, ...],
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    target.mkdir(parents=True, exist_ok=True)

    if local.is_symlink():
        status = "ready" if _same_target(local, target) else "degraded"
        return {
            "name": name,
            "local_path": str(local),
            "target_path": str(target),
            "local_bytes": 0,
            "is_symlink": True,
            "target_match": _same_target(local, target),
            "active_process": False,
            "open_handles": False,
            "status": status,
            "actions": actions,
        }

    local_bytes = _path_size_bytes(local)
    active = _active_process(active_patterns)
    open_handles = _has_open_handles(local)

    if apply and local.exists() and local.is_dir():
        nested = local / local.name
        if nested.is_symlink() and _same_target(nested, target):
            nested.unlink()
            actions.append({"action": "remove_nested_self_symlink", "path": str(nested)})
        for child in sorted(local.iterdir()):
            _merge_path(child, target / child.name, target, actions)
        local_bytes = _path_size_bytes(local)
        allow_active_empty_relink = _env_flag("STATEFUL_STORAGE_REGRESSION_ALLOW_ACTIVE_EMPTY_RELINK", "1")
        if (not open_handles and (not active or allow_active_empty_relink) and local.exists() and local.is_dir() and local_bytes == 0):
            local.rmdir()
            local.symlink_to(target, target_is_directory=True)
            actions.append({"action": "replace_local_dir_with_symlink", "local_path": str(local), "target_path": str(target)})

        local_bytes = _path_size_bytes(local)
        if not open_handles and local.exists() and local.is_dir() and not local.is_symlink() and not _same_target(local, target):
            handoff = local.with_name(f"{local.name}.handoff_{iso_now().replace(':', '').replace('+', '_')}")
            local.rename(handoff)
            local.symlink_to(target, target_is_directory=True)
            actions.append({"action": "atomic_handoff_symlink", "handoff_path": str(handoff), "local_path": str(local), "target_path": str(target)})
            for child in sorted(handoff.iterdir()):
                _merge_path(child, target / child.name, target, actions)
            try:
                handoff.rmdir()
                actions.append({"action": "remove_empty_handoff_dir", "path": str(handoff)})
            except OSError:
                pass

    if apply and not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        local.symlink_to(target, target_is_directory=True)
        actions.append({"action": "create_symlink", "local_path": str(local), "target_path": str(target)})

    target_match = _same_target(local, target)
    local_bytes = _path_size_bytes(local)
    if target_match:
        status = "ready"
    elif local_bytes > max_local_bytes:
        status = "blocked"
    else:
        status = "degraded"

    return {
        "name": name,
        "local_path": str(local),
        "target_path": str(target),
        "local_bytes": local_bytes,
        "local_gb": round(local_bytes / (1024**3), 3),
        "max_local_bytes": max_local_bytes,
        "is_symlink": local.is_symlink(),
        "target_match": target_match,
        "active_process": active,
        "open_handles": open_handles,
        "status": status,
        "actions": actions,
    }


def _read_plist(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            payload = plistlib.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_plist(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=False)


def _launchd_log_check(*, apply: bool) -> dict[str, Any]:
    payload = _read_plist(SQL_WRITER_PLIST)
    if not payload:
        return {
            "name": "sql_link_writer_launchd_logs",
            "plist": str(SQL_WRITER_PLIST),
            "status": "degraded",
            "summary": "sql writer LaunchAgent plist is missing or unreadable",
            "actions": [],
        }
    desired_out = DEFAULT_LAUNCHD_LOG_ROOT / "ops_sql_link_writer.out.log"
    desired_err = DEFAULT_LAUNCHD_LOG_ROOT / "ops_sql_link_writer.err.log"
    actions: list[dict[str, Any]] = []
    current_out = str(payload.get("StandardOutPath") or "")
    current_err = str(payload.get("StandardErrorPath") or "")
    ok = current_out == str(desired_out) and current_err == str(desired_err)
    if apply and not ok:
        DEFAULT_LAUNCHD_LOG_ROOT.mkdir(parents=True, exist_ok=True)
        payload["StandardOutPath"] = str(desired_out)
        payload["StandardErrorPath"] = str(desired_err)
        _write_plist(SQL_WRITER_PLIST, payload)
        actions.append({"action": "rewrite_launchd_log_paths", "stdout": str(desired_out), "stderr": str(desired_err)})
        ok = True
    return {
        "name": "sql_link_writer_launchd_logs",
        "plist": str(SQL_WRITER_PLIST),
        "stdout_path": str(payload.get("StandardOutPath") or ""),
        "stderr_path": str(payload.get("StandardErrorPath") or ""),
        "desired_stdout_path": str(desired_out),
        "desired_stderr_path": str(desired_err),
        "status": "ready" if ok else "degraded",
        "summary": "sql writer launchd logs are outside the project/external log symlink" if ok else "sql writer launchd logs still point at an unsafe project/external log path",
        "actions": actions,
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    external_root: str = "",
    apply: bool = False,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    external = _external_project_root(project_root, external_root)
    max_sql_bytes = _safe_int(os.getenv("STATEFUL_STORAGE_SQL_LOCAL_MAX_BYTES"), 64 * 1024 * 1024)
    max_lane_bytes = _safe_int(os.getenv("STATEFUL_STORAGE_EXECUTION_LANE_LOCAL_MAX_BYTES"), 256 * 1024 * 1024)
    route_checks = [
        _repair_stateful_path(
            name="sql_link_shards",
            local=project_root / "data" / "sql_link_shards",
            target=external / "data" / "sql_link_shards",
            apply=apply,
            max_local_bytes=max_sql_bytes,
            active_patterns=("scripts/ops/sql_link_shard_manager.py", "scripts/ops/sql_link_writer_service.py", "scripts/link_jsonl_to_sql.py"),
        ),
        _repair_stateful_path(
            name="execution_lanes",
            local=project_root / "governance" / "execution_lanes",
            target=external / "governance" / "execution_lanes",
            apply=apply,
            max_local_bytes=max_lane_bytes,
            active_patterns=("scripts/run_execution_lane.py",),
        ),
    ]
    launchd_check = _launchd_log_check(apply=apply)
    checks = [*route_checks, launchd_check]
    blocked = [row for row in checks if row.get("status") == "blocked"]
    degraded = [row for row in checks if row.get("status") == "degraded"]
    overall_status = "blocked" if blocked else ("degraded" if degraded else "ready")
    local_total_bytes = sum(_safe_int(row.get("local_bytes"), 0) for row in route_checks)
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply": bool(apply),
        "project_root": str(project_root),
        "external_project_root": str(external),
        "checks": checks,
        "metrics": {
            "local_stateful_bytes": local_total_bytes,
            "local_stateful_gb": round(local_total_bytes / (1024**3), 3),
            "blocked_check_count": len(blocked),
            "degraded_check_count": len(degraded),
        },
        "infra_owner": "infrastructure_autofix_bot",
        "recommended_actions": [
            "run ./scripts/ops/opsctl.sh stateful-storage-regression-guard --apply --json if local stateful lanes reappear",
            "restart the execution lane after market hours if active writers keep a small local fallback directory open",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect and repair local stateful-storage regressions after BOT_LOGS routing.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--external-root", default="")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), external_root=str(args.external_root or ""), apply=bool(args.apply))
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "stateful_storage_regression_guard "
            f"overall_status={payload.get('overall_status', '')} "
            f"local_stateful_gb={((payload.get('metrics') or {}).get('local_stateful_gb', 0.0))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
