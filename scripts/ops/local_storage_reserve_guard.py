#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.local_storage_reserve import local_storage_reserve_contract


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "local_storage_reserve_guard_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.local_storage_reserve_override"
DEFAULT_LOG_ROOT = Path("/private/tmp/schwab_trading_bot/launchd_ops")
DEFAULT_MAX_LOG_BYTES = 16 * 1024 * 1024
DEFAULT_TAIL_BYTES = 1024 * 1024
TELEMETRY_ROUTE_PATHS = (
    "local_fallback_storage/decisions",
    "local_fallback_storage/decision_explanations",
    "local_fallback_storage/governance",
    "governance/channels/decision",
)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, path)
    except OSError:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        # A tiny in-place fallback still works when the filesystem cannot
        # allocate metadata for an atomic replacement during severe pressure.
        with path.open("w", encoding="utf-8") as handle:
            handle.write(content)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def _write_override(path: Path, control_env: dict[str, str]) -> bool:
    lines = [
        "# Managed by local_storage_reserve_guard.py. Manual edits will be replaced.",
        "# Pressure-only keys disappear automatically after the live reserve recovers.",
    ]
    lines.extend(f"{key}={value}" for key, value in sorted(control_env.items()))
    content = "\n".join(lines) + "\n"
    try:
        previous = path.read_text(encoding="utf-8")
    except OSError:
        previous = ""
    if previous == content:
        return False
    _atomic_write_text(path, content)
    return True


def _cap_log_file(path: Path, *, max_bytes: int, tail_bytes: int, apply: bool) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError as exc:
        return {"path": str(path), "status": "error", "error": f"{type(exc).__name__}:{exc}"}
    original = int(stat.st_size)
    if original <= max_bytes:
        return {"path": str(path), "status": "within_limit", "bytes_before": original, "bytes_after": original}
    if not apply:
        return {"path": str(path), "status": "would_cap", "bytes_before": original, "bytes_after": original}

    keep = max(min(int(tail_bytes), int(max_bytes), original), 0)
    try:
        with path.open("r+b", buffering=0) as handle:
            handle.seek(max(original - keep, 0))
            tail = handle.read(keep)
            handle.seek(0)
            handle.write(tail)
            handle.truncate(len(tail))
        after = int(path.stat().st_size)
    except OSError as exc:
        return {
            "path": str(path),
            "status": "error",
            "bytes_before": original,
            "bytes_after": original,
            "error": f"{type(exc).__name__}:{exc}",
        }
    return {
        "path": str(path),
        "status": "capped",
        "bytes_before": original,
        "bytes_after": after,
        "bytes_reclaimed": max(original - after, 0),
    }


def cap_launchd_logs(
    log_root: Path,
    *,
    max_bytes: int = DEFAULT_MAX_LOG_BYTES,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
    apply: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if log_root.exists():
        candidates = sorted(
            path
            for pattern in ("*.out.log", "*.err.log")
            for path in log_root.glob(pattern)
            if path.is_file() and not path.is_symlink()
        )
        rows = [
            _cap_log_file(path, max_bytes=max(int(max_bytes), 1), tail_bytes=max(int(tail_bytes), 0), apply=apply)
            for path in candidates
        ]
    oversized = [row for row in rows if row.get("status") in {"would_cap", "capped", "error"}]
    errors = [row for row in rows if row.get("status") == "error"]
    return {
        "log_root": str(log_root),
        "exists": log_root.exists(),
        "apply": bool(apply),
        "max_file_bytes": int(max_bytes),
        "tail_bytes": int(tail_bytes),
        "file_count": len(rows),
        "oversized_count": len(oversized),
        "capped_count": sum(1 for row in rows if row.get("status") == "capped"),
        "error_count": len(errors),
        "bytes_reclaimed": sum(int(row.get("bytes_reclaimed", 0) or 0) for row in rows),
        "rows": oversized[:40],
    }


def telemetry_route_contract(project_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for relative in TELEMETRY_ROUTE_PATHS:
        path = project_root / relative
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            resolved = path
        external = str(resolved).startswith("/Volumes/BOT_LOGS/")
        quarantine_backed = "/quarantine/" in str(resolved).replace("\\", "/")
        rows.append(
            {
                "relative_path": relative,
                "path": str(path),
                "is_symlink": path.is_symlink(),
                "exists": path.exists(),
                "resolved_path": str(resolved),
                "external_bot_logs": external,
                "quarantine_backed": quarantine_backed,
                "ready": bool(path.exists() and external and not quarantine_backed),
            }
        )
    ready_count = sum(1 for row in rows if row["ready"])
    return {
        "status": "ready" if ready_count == len(rows) else "degraded",
        "ready": ready_count == len(rows),
        "ready_count": ready_count,
        "tracked_count": len(rows),
        "rows": rows,
    }


def _reconcile_storage_governor(project_root: Path, *, timeout_seconds: int) -> dict[str, Any]:
    command = [
        str(project_root / "scripts" / "ops" / "opsctl.sh"),
        "ingestion-storage-governor",
        "apply",
        "--json",
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(int(timeout_seconds), 1),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"attempted": True, "ok": False, "returncode": 124, "error": "timeout"}
    except OSError as exc:
        return {"attempted": True, "ok": False, "returncode": 1, "error": f"{type(exc).__name__}:{exc}"}
    return {
        "attempted": True,
        "ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stderr_tail": (proc.stderr or "")[-1000:],
    }


def build_payload(
    project_root: Path,
    *,
    apply: bool,
    override_path: Path,
    log_root: Path,
    max_log_bytes: int,
    tail_bytes: int,
    reconcile_governor: bool = True,
) -> dict[str, Any]:
    logs = cap_launchd_logs(
        log_root,
        max_bytes=max_log_bytes,
        tail_bytes=tail_bytes,
        apply=apply,
    )
    reserve = local_storage_reserve_contract(project_root)
    changed = _write_override(override_path, reserve["control_env"]) if apply else False
    governor = {"attempted": False, "ok": True}
    if apply and reconcile_governor and not bool(reserve.get("emergency_active", False)):
        governor = _reconcile_storage_governor(project_root, timeout_seconds=45)
    routes = telemetry_route_contract(project_root)
    hard_blockers: list[str] = []
    warnings: list[str] = []
    if bool(reserve.get("hard_block", False)):
        hard_blockers.append("local_hot_storage_below_hard_reserve")
    elif not bool(reserve.get("ready", False)):
        warnings.append("local_hot_storage_below_unattended_target")
    if int(logs.get("error_count", 0) or 0) > 0:
        warnings.append("launchd_log_cap_errors")
    if not bool(routes.get("ready", False)):
        hard_blockers.append("external_telemetry_spill_route_not_ready")
    if bool(governor.get("attempted", False)) and not bool(governor.get("ok", False)):
        warnings.append("storage_governor_reconciliation_failed")
    status = "blocked" if hard_blockers else ("watch" if warnings else "ready")
    return {
        "timestamp_utc": reserve.get("timestamp_utc"),
        "schema_version": 1,
        "ok": not hard_blockers,
        "overall_status": status,
        "grade": "F" if hard_blockers else ("A" if warnings else "A+"),
        "apply": bool(apply),
        "local_storage_reserve": reserve,
        "launchd_log_guard": logs,
        "telemetry_route_contract": routes,
        "storage_governor_reconciliation": governor,
        "override_path": str(override_path),
        "override_changed": bool(changed),
        "hard_blockers": hard_blockers,
        "warnings": warnings,
        "next_action": (
            "continue unattended collection with the live reserve guard active"
            if status == "ready"
            else "restore reserve or telemetry routing before unattended collection"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce live local-disk reserve and bounded launchd logs.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--max-log-bytes", type=int, default=int(os.getenv("BOT_LAUNCHD_LOG_MAX_BYTES", DEFAULT_MAX_LOG_BYTES)))
    parser.add_argument("--tail-bytes", type=int, default=int(os.getenv("BOT_LAUNCHD_LOG_TAIL_BYTES", DEFAULT_TAIL_BYTES)))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--skip-governor-reconcile", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(
        project_root,
        apply=bool(args.apply),
        override_path=Path(args.override_file).expanduser(),
        log_root=Path(args.log_root).expanduser(),
        max_log_bytes=max(int(args.max_log_bytes), 1),
        tail_bytes=max(int(args.tail_bytes), 0),
        reconcile_governor=not bool(args.skip_governor_reconcile),
    )
    _write_json(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    else:
        reserve = payload.get("local_storage_reserve", {})
        print(
            "local_storage_reserve_guard "
            f"status={payload.get('overall_status')} "
            f"free_gb={reserve.get('free_gb', 0)} "
            f"logs_capped={payload.get('launchd_log_guard', {}).get('capped_count', 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
