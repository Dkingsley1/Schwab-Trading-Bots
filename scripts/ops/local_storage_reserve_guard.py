#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.local_storage_reserve import local_storage_reserve_contract  # noqa: E402

DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "local_storage_reserve_guard_latest.json"
)
DEFAULT_HISTORY_PATH = (
    PROJECT_ROOT / "governance" / "health" / "local_storage_reserve_guard_history.jsonl"
)
DEFAULT_LOCK_PATH = (
    PROJECT_ROOT / "governance" / "locks" / "local_storage_reserve_guard.lock"
)
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.local_storage_reserve_override"
DEFAULT_LOG_ROOT = Path("/private/tmp/schwab_trading_bot/launchd_ops")
DEFAULT_USER_LOG_ROOT = Path.home() / "Library" / "Logs" / "schwab_trading_bot"
DEFAULT_MAX_LOG_BYTES = 16 * 1024 * 1024
DEFAULT_TAIL_BYTES = 1024 * 1024
DEFAULT_HISTORY_MAX_LINES = 2048
DEFAULT_TARGET_FREE_GB = 125.0
DEFAULT_RECOVERY_HEADROOM_GB = 10.0
DEFAULT_PRESSURE_FREE_GB = 64.0
DEFAULT_HARD_FREE_GB = 32.0
DEFAULT_EMERGENCY_FREE_GB = 16.0
DEFAULT_TREE_SIZE_FILE_LIMIT = 10000
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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _history_signature(payload: dict[str, Any]) -> tuple[Any, ...]:
    reserve = (
        payload.get("local_storage_reserve")
        if isinstance(payload.get("local_storage_reserve"), dict)
        else {}
    )
    recovery = (
        payload.get("recovery_request")
        if isinstance(payload.get("recovery_request"), dict)
        else {}
    )
    return (
        str(payload.get("overall_status") or ""),
        str(payload.get("grade") or ""),
        tuple(str(item) for item in list(payload.get("hard_blockers") or [])),
        tuple(str(item) for item in list(payload.get("warnings") or [])),
        str(reserve.get("status") or ""),
        float(reserve.get("target_free_gb", 0.0) or 0.0),
        float(reserve.get("pressure_free_gb", 0.0) or 0.0),
        float(reserve.get("hard_free_gb", 0.0) or 0.0),
        float(reserve.get("emergency_free_gb", 0.0) or 0.0),
        bool(recovery.get("active", False)),
        str(recovery.get("severity") or ""),
        bool(recovery.get("paper_pause_required", False)),
    )


def _history_event_required(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    if not previous or _history_signature(previous) != _history_signature(current):
        return True
    logs = (
        current.get("launchd_log_guard")
        if isinstance(current.get("launchd_log_guard"), dict)
        else {}
    )
    return bool(
        int(logs.get("bytes_reclaimed", 0) or 0) > 0
        or int(logs.get("capped_count", 0) or 0) > 0
    )


def _append_history(
    path: Path, payload: dict[str, Any], *, max_lines: int
) -> dict[str, Any]:
    reserve = (
        payload.get("local_storage_reserve")
        if isinstance(payload.get("local_storage_reserve"), dict)
        else {}
    )
    cleanup = (
        payload.get("cleanup_verification")
        if isinstance(payload.get("cleanup_verification"), dict)
        else {}
    )
    recovery = (
        payload.get("recovery_request")
        if isinstance(payload.get("recovery_request"), dict)
        else {}
    )
    row = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "overall_status": payload.get("overall_status"),
        "grade": payload.get("grade"),
        "free_gb": reserve.get("free_gb"),
        "target_free_gb": reserve.get("target_free_gb"),
        "pressure_free_gb": reserve.get("pressure_free_gb"),
        "hard_free_gb": reserve.get("hard_free_gb"),
        "emergency_free_gb": reserve.get("emergency_free_gb"),
        "bytes_reclaimed": cleanup.get("file_bytes_reclaimed"),
        "cleanup_verified": cleanup.get("verified"),
        "recovery_active": recovery.get("active"),
        "recovery_severity": recovery.get("severity"),
        "paper_pause_required": recovery.get("paper_pause_required"),
        "hard_blockers": list(payload.get("hard_blockers") or []),
        "warnings": list(payload.get("warnings") or []),
    }
    limit = max(int(max_lines), 1)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        return {
            "ok": True,
            "appended": True,
            "path": str(path),
            "max_lines": limit,
            "retention_policy": "append_only_cold_archive_managed",
        }
    except OSError as exc:
        return {
            "ok": False,
            "appended": False,
            "path": str(path),
            "max_lines": limit,
            "error": f"{type(exc).__name__}:{exc}",
        }


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


def _cap_log_file(
    path: Path, *, max_bytes: int, tail_bytes: int, apply: bool
) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError as exc:
        return {
            "path": str(path),
            "status": "error",
            "error": f"{type(exc).__name__}:{exc}",
        }
    original = int(stat.st_size)
    inode_before = int(stat.st_ino)
    if original <= max_bytes:
        return {
            "path": str(path),
            "status": "within_limit",
            "bytes_before": original,
            "bytes_after": original,
        }
    if not apply:
        return {
            "path": str(path),
            "status": "would_cap",
            "bytes_before": original,
            "bytes_after": original,
        }

    keep = max(min(int(tail_bytes), int(max_bytes), original), 0)
    try:
        with path.open("r+b", buffering=0) as handle:
            handle.seek(max(original - keep, 0))
            tail = handle.read(keep)
            handle.seek(0)
            handle.write(tail)
            handle.truncate(len(tail))
        after_stat = path.stat()
        after = int(after_stat.st_size)
        inode_after = int(after_stat.st_ino)
        with path.open("rb") as handle:
            tail_content_preserved = handle.read(len(tail)) == tail
    except OSError as exc:
        return {
            "path": str(path),
            "status": "error",
            "bytes_before": original,
            "bytes_after": original,
            "error": f"{type(exc).__name__}:{exc}",
        }
    inode_preserved = inode_after == inode_before
    within_limit = after <= max_bytes
    verified = inode_preserved and within_limit and tail_content_preserved
    return {
        "path": str(path),
        "status": "capped" if verified else "verification_failed",
        "bytes_before": original,
        "bytes_after": after,
        "bytes_reclaimed": max(original - after, 0),
        "inode_before": inode_before,
        "inode_after": inode_after,
        "inode_preserved": inode_preserved,
        "within_limit": within_limit,
        "tail_content_preserved": tail_content_preserved,
        "tail_bytes_preserved": min(keep, after),
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
            {
                path
                for pattern in ("*.out.log", "*.err.log")
                for path in log_root.rglob(pattern)
                if path.is_file() and not path.is_symlink()
            }
        )
        rows = [
            _cap_log_file(
                path,
                max_bytes=max(int(max_bytes), 1),
                tail_bytes=max(int(tail_bytes), 0),
                apply=apply,
            )
            for path in candidates
        ]
    oversized = [
        row
        for row in rows
        if row.get("status") in {"would_cap", "capped", "verification_failed", "error"}
    ]
    errors = [
        row for row in rows if row.get("status") in {"verification_failed", "error"}
    ]
    return {
        "log_root": str(log_root),
        "exists": log_root.exists(),
        "apply": bool(apply),
        "max_file_bytes": int(max_bytes),
        "tail_bytes": int(tail_bytes),
        "file_count": len(rows),
        "oversized_count": len(oversized),
        "capped_count": sum(1 for row in rows if row.get("status") == "capped"),
        "verification_failed_count": sum(
            1 for row in rows if row.get("status") == "verification_failed"
        ),
        "error_count": len(errors),
        "bytes_reclaimed": sum(int(row.get("bytes_reclaimed", 0) or 0) for row in rows),
        "rows": oversized[:40],
    }


def cap_launchd_log_roots(
    log_roots: list[Path],
    *,
    max_bytes: int = DEFAULT_MAX_LOG_BYTES,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
    apply: bool = False,
) -> dict[str, Any]:
    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in log_roots:
        normalized = str(root.expanduser().resolve(strict=False))
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_roots.append(Path(normalized))

    root_reports = [
        cap_launchd_logs(
            root,
            max_bytes=max_bytes,
            tail_bytes=tail_bytes,
            apply=apply,
        )
        for root in unique_roots
    ]
    return {
        "log_root": str(unique_roots[0]) if unique_roots else "",
        "log_roots": [str(root) for root in unique_roots],
        "exists": any(bool(report.get("exists", False)) for report in root_reports),
        "apply": bool(apply),
        "max_file_bytes": int(max_bytes),
        "tail_bytes": int(tail_bytes),
        "file_count": sum(
            int(report.get("file_count", 0) or 0) for report in root_reports
        ),
        "oversized_count": sum(
            int(report.get("oversized_count", 0) or 0) for report in root_reports
        ),
        "capped_count": sum(
            int(report.get("capped_count", 0) or 0) for report in root_reports
        ),
        "verification_failed_count": sum(
            int(report.get("verification_failed_count", 0) or 0)
            for report in root_reports
        ),
        "error_count": sum(
            int(report.get("error_count", 0) or 0) for report in root_reports
        ),
        "bytes_reclaimed": sum(
            int(report.get("bytes_reclaimed", 0) or 0) for report in root_reports
        ),
        "rows": [
            row for report in root_reports for row in list(report.get("rows") or [])
        ][:80],
        "roots": root_reports,
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


def _disk_free_gb(path: Path) -> float:
    try:
        usage = os.statvfs(path)
    except OSError:
        return 0.0
    return round(float(usage.f_bavail * usage.f_frsize) / (1024**3), 3)


def _bounded_tree_size(
    path: Path, *, max_files: int = DEFAULT_TREE_SIZE_FILE_LIMIT
) -> dict[str, Any]:
    files = 0
    size_bytes = 0
    errors = 0
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": 0,
            "size_gb": 0.0,
            "size_kind": "complete",
            "files_counted": 0,
            "truncated": False,
            "errors": 0,
        }
    for root, _, names in os.walk(path):
        root_path = Path(root)
        for name in names:
            if files >= max(max_files, 1):
                return {
                    "path": str(path),
                    "exists": True,
                    "size_bytes": int(size_bytes),
                    "size_gb": round(float(size_bytes) / (1024**3), 3),
                    "size_kind": "lower_bound",
                    "files_counted": int(files),
                    "truncated": True,
                    "errors": int(errors),
                }
            item = root_path / name
            try:
                if item.is_symlink():
                    continue
                size_bytes += int(item.stat().st_size)
                files += 1
            except OSError:
                errors += 1
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(size_bytes),
        "size_gb": round(float(size_bytes) / (1024**3), 3),
        "size_kind": "complete",
        "files_counted": int(files),
        "truncated": False,
        "errors": int(errors),
    }


def _external_project_root_from_env() -> Path:
    configured = str(os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    mount_root = Path(
        os.getenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS")
    ).expanduser()
    project_dir = str(
        os.getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", "schwab_trading_bot")
        or "schwab_trading_bot"
    ).strip()
    return mount_root / project_dir


def _pipeline_stage(
    *,
    order: int,
    stage: str,
    owner: str,
    authority: str,
    command: list[str],
    entry_condition: str,
    release_condition: str,
    blocked_by: list[str],
    inputs: list[str],
    outputs: list[str],
    auto_execute: bool,
    resource_lock: str = "none",
    failure_mode: str = "",
    next_on_success: str = "",
) -> dict[str, Any]:
    return {
        "order": int(order),
        "stage": stage,
        "owner": owner,
        "authority": authority,
        "auto_execute_allowed": bool(auto_execute),
        "resource_lock": resource_lock,
        "entry_condition": entry_condition,
        "release_condition": release_condition,
        "blocked_by": blocked_by,
        "inputs": inputs,
        "outputs": outputs,
        "command": command,
        "failure_mode": failure_mode,
        "next_on_success": next_on_success,
    }


def _storage_recovery_pipeline(project_root: Path) -> list[dict[str, Any]]:
    opsctl = str(project_root / "scripts" / "ops" / "opsctl.sh")
    return [
        _pipeline_stage(
            order=10,
            stage="detect",
            owner="local_storage_reserve_guard",
            authority="automatic_readiness_probe",
            auto_execute=True,
            resource_lock="local_storage_guard",
            entry_condition="internal free space is below target, pressure, hard, or emergency reserve",
            release_condition="local_storage_reserve.ready is true or a specific route/storage blocker is named",
            blocked_by=["local_storage_reserve_guard.lock"],
            inputs=["local disk usage", "storage_failback_sync_latest.json"],
            outputs=[
                "local_storage_reserve_guard_latest.json",
                "config/.env.local_storage_reserve_override",
            ],
            command=[opsctl, "local-storage-reserve-guard", "--apply", "--json"],
            failure_mode="reserve_unknown_or_guard_history_write_failed",
            next_on_success="contain",
        ),
        _pipeline_stage(
            order=20,
            stage="contain",
            owner="ingestion_storage_governor",
            authority="automatic_pause_nonessential_writers",
            auto_execute=True,
            resource_lock="storage_governor",
            entry_condition="local pressure is active or telemetry route is not external",
            release_condition="paper/order/training writers are paused or duty-cycled before heavier repair",
            blocked_by=["runtime_maintenance_hold_without_token"],
            inputs=[
                "local_storage_reserve_guard_latest.json",
                "ingestion_storage_control_latest.json",
            ],
            outputs=["ingestion_storage_governor_latest.json"],
            command=[opsctl, "ingestion-storage-governor", "apply", "--json"],
            failure_mode="writers_continue_while_local_pressure_active",
            next_on_success="compact",
        ),
        _pipeline_stage(
            order=30,
            stage="compact",
            owner="backlog_pcore_accelerator",
            authority="automatic_bounded_p_core_file_writes",
            auto_execute=True,
            resource_lock="p_core_file_compaction",
            entry_condition="raw or cold file candidates exist and external scratch space is above floor",
            release_condition="raw/cold compaction reports zero verification failures",
            blocked_by=["storage_compaction_scratch_space_below_floor"],
            inputs=[
                "raw_training_compaction_intelligence_latest.json",
                "data_collection_storage_guard_latest.json",
            ],
            outputs=[
                "backlog_pcore_accelerator_latest.json",
                "compressed raw evidence files",
            ],
            command=[opsctl, "backlog-pcore-accelerator", "--apply", "--json"],
            failure_mode="file_compaction_verification_failure",
            next_on_success="checkpoint",
        ),
        _pipeline_stage(
            order=40,
            stage="checkpoint",
            owner="storage_pressure_clearance",
            authority="automatic_single_sqlite_writer_checkpoint",
            auto_execute=True,
            resource_lock="single_sqlite_writer",
            entry_condition="WAL growth, stale gate, or pressure remains after file compaction",
            release_condition="active_pressure is false and checkpoint attempts report zero errors",
            blocked_by=["duplicate_sqlite_writer_blocks_checkpoint"],
            inputs=[
                "storage_pressure_clearance_latest.json",
                "sqlite_maintenance_latest.json",
            ],
            outputs=["storage_pressure_clearance_latest.json"],
            command=[
                opsctl,
                "storage-pressure-clearance",
                "--apply",
                "--force-clear-stale-gate",
                "--checkpoint-mode",
                "passive",
                "--json",
            ],
            failure_mode="checkpoint_does_not_reduce_wal_or_pressure",
            next_on_success="rehome",
        ),
        _pipeline_stage(
            order=50,
            stage="rehome",
            owner="storage_switch_orchestrator",
            authority="operator_route_mutation",
            auto_execute=False,
            resource_lock="storage_route",
            entry_condition="active route is local_fallback, BOT_LOGS is mounted, and local reserve remains under pressure",
            release_condition="storage_failback_sync certifies external or external_curated with zero mismatches",
            blocked_by=[
                "external_root_unavailable",
                "route_verification_mismatch",
                "active_writer_not_quiesced",
            ],
            inputs=[
                "storage_failback_sync_latest.json",
                "storage_switch_orchestrator_latest.json",
            ],
            outputs=[
                "storage_switch_orchestrator_latest.json",
                "storage_failback_sync_latest.json",
            ],
            command=[opsctl, "storage-switch-external"],
            failure_mode="route_rehome_incomplete_or_external_copy_not_certified",
            next_on_success="prune",
        ),
        _pipeline_stage(
            order=60,
            stage="prune",
            owner="storage_standby_prune",
            authority="automatic_verified_standby_delete",
            auto_execute=True,
            resource_lock="storage_standby_prune",
            entry_condition="external route is certified, active local count is zero, and route soak window passed",
            release_condition="eligible local standby copies are deleted or no eligible standby remains",
            blocked_by=[
                "active_local_count_positive",
                "route_soak_not_satisfied",
                "verification_mismatch_count_positive",
            ],
            inputs=[
                "storage_failback_sync_latest.json",
                "storage_switch_orchestrator_latest.json",
            ],
            outputs=["storage_standby_prune_latest.json"],
            command=[opsctl, "storage-prune-standby", "--apply", "--json"],
            failure_mode="standby_delete_error_or_route_guard_refusal",
            next_on_success="verify",
        ),
        _pipeline_stage(
            order=70,
            stage="verify",
            owner="storage_retention_unison",
            authority="automatic_readiness_verification",
            auto_execute=True,
            resource_lock="storage_readiness",
            entry_condition="compact, checkpoint, rehome, and prune have either completed or named blockers",
            release_condition="reserve, route, backlog, and retention surfaces agree on ready/watch state",
            blocked_by=[
                "local_hot_storage_pressure_reserve_breached",
                "second_cold_same_filesystem",
            ],
            inputs=[
                "local_storage_reserve_guard_latest.json",
                "backlog_pcore_accelerator_latest.json",
                "storage_pressure_clearance_latest.json",
                "storage_retention_unison_latest.json",
            ],
            outputs=["storage_retention_unison_latest.json"],
            command=[opsctl, "storage-retention-unison", "--json"],
            failure_mode="retention_unison_reports_blocked_after_pipeline",
            next_on_success="release",
        ),
        _pipeline_stage(
            order=80,
            stage="release",
            owner="operator_cockpit",
            authority="automatic_status_release_no_trade_authority",
            auto_execute=True,
            resource_lock="none",
            entry_condition="storage pipeline verifies ready/watch and no hot-path block remains",
            release_condition="operator cockpit no longer reports storage as an uncontained blocker",
            blocked_by=[
                "paper_execution_safety_guard_active",
                "profitability_evidence_debt",
            ],
            inputs=[
                "operator_cockpit_latest.json",
                "runtime_gate_dashboard_latest.json",
            ],
            outputs=[
                "operator_cockpit_latest.json",
                "runtime_gate_dashboard_latest.json",
            ],
            command=[opsctl, "runtime-gate-dashboard", "--json"],
            failure_mode="dashboard_still_reports_storage_degradation",
            next_on_success="monitor",
        ),
        _pipeline_stage(
            order=90,
            stage="monitor",
            owner="degradation_swarm_coordinator",
            authority="automatic_containment_watch",
            auto_execute=True,
            resource_lock="degradation_swarm",
            entry_condition="pipeline has released or named a stable external blocker",
            release_condition="no repeated storage degradation incident crosses the recurrence threshold",
            blocked_by=["same_incident_repeats_after_max_attempts"],
            inputs=[
                "degradation_swarm_state.json",
                "local_storage_reserve_guard_history.jsonl",
            ],
            outputs=[
                "degradation_swarm_coordinator_latest.json",
                "degradation_swarm_repair_ledger.jsonl",
            ],
            command=[
                opsctl,
                "degradation-swarm",
                "--apply",
                "--execute-safe-repairs",
                "--json",
            ],
            failure_mode="swarm_cooldown_or_retry_budget_exhausted",
            next_on_success="steady_state",
        ),
    ]


def fallback_route_pressure_contract(
    project_root: Path,
    reserve: dict[str, Any],
) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    failback = _load_json(health / "storage_failback_sync_latest.json")
    sqlite_report = (
        failback.get("sqlite_skip_report")
        if isinstance(failback.get("sqlite_skip_report"), dict)
        else {}
    )
    summary = (
        sqlite_report.get("summary")
        if isinstance(sqlite_report.get("summary"), dict)
        else {}
    )
    route_verification = (
        failback.get("route_verification")
        if isinstance(failback.get("route_verification"), dict)
        else {}
    )
    if not route_verification:
        route_verification = (
            sqlite_report.get("route_verification")
            if isinstance(sqlite_report.get("route_verification"), dict)
            else {}
        )
    active_mode = str(
        failback.get("certified_mode") or failback.get("mode") or ""
    ).strip()
    active_root = str(failback.get("active_root") or "").strip()
    route_state = str(route_verification.get("verification_state") or "").strip()
    mismatches = [
        str(item)
        for item in list(route_verification.get("mismatches") or [])
        if str(item).strip()
    ]
    active_local_count = _safe_int(summary.get("active_local_count"), 0)
    active_external_count = _safe_int(summary.get("active_external_count"), 0)
    warm_standby_count = _safe_int(summary.get("warm_standby_count"), 0)
    tracked_local_sqlite_gb = round(
        float(_safe_int(summary.get("local_bytes_total"), 0)) / (1024**3),
        3,
    )
    pressure_active = bool(reserve.get("pressure_active", False))
    route_is_local_fallback = bool(active_mode.startswith("local_fallback"))
    local_root = project_root / "local_fallback_storage"
    external_root = _external_project_root_from_env()
    external_available = bool(
        external_root.exists() and os.access(external_root, os.W_OK)
    )
    external_free_gb = _disk_free_gb(external_root) if external_available else 0.0
    route_rehome_required = bool(route_is_local_fallback and pressure_active)
    route_rehome_ready = bool(
        route_rehome_required and external_available and not mismatches
    )
    standby_prune_ready = bool(
        active_mode in {"external", "external_curated"}
        and route_state in {"ready", "curated_ready"}
        and active_local_count == 0
        and not mismatches
    )
    pipeline = _storage_recovery_pipeline(project_root)
    if route_rehome_ready:
        status = "route_rehome_ready"
    elif route_rehome_required:
        status = "route_rehome_blocked"
    elif standby_prune_ready:
        status = "standby_prune_ready"
    else:
        status = "ready"

    return {
        "status": status,
        "active_mode": active_mode,
        "active_root": active_root,
        "route_state": route_state,
        "route_is_local_fallback": route_is_local_fallback,
        "route_rehome_required": route_rehome_required,
        "route_rehome_ready": route_rehome_ready,
        "standby_prune_ready": standby_prune_ready,
        "external_root": str(external_root),
        "external_available": external_available,
        "external_free_gb": external_free_gb,
        "active_local_count": int(active_local_count),
        "active_external_count": int(active_external_count),
        "warm_standby_count": int(warm_standby_count),
        "tracked_local_sqlite_gb": tracked_local_sqlite_gb,
        "local_fallback_data_tree": _bounded_tree_size(local_root / "data"),
        "local_fallback_sql_link_shards": _bounded_tree_size(
            local_root / "data" / "sql_link_shards"
        ),
        "verification_mismatches": mismatches,
        "ordered_recovery_pipeline": pipeline,
        "pipeline_summary": {
            "stage_count": len(pipeline),
            "automatic_stage_count": sum(
                1 for row in pipeline if bool(row.get("auto_execute_allowed", False))
            ),
            "operator_route_mutation_stage_count": sum(
                1
                for row in pipeline
                if str(row.get("authority") or "") == "operator_route_mutation"
            ),
            "single_writer_stage_count": sum(
                1
                for row in pipeline
                if str(row.get("resource_lock") or "") == "single_sqlite_writer"
            ),
        },
        "policy": "compact and checkpoint automatically; route mutation is explicit; standby deletion requires route verification",
    }


def _reconcile_storage_governor(
    project_root: Path, *, timeout_seconds: int
) -> dict[str, Any]:
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
        return {
            "attempted": True,
            "ok": False,
            "returncode": 1,
            "error": f"{type(exc).__name__}:{exc}",
        }
    return {
        "attempted": True,
        "ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stderr_tail": (proc.stderr or "")[-1000:],
    }


def _recovery_request(
    reserve: dict[str, Any],
    project_root: Path,
    *,
    recovery_headroom_gb: float = DEFAULT_RECOVERY_HEADROOM_GB,
) -> dict[str, Any]:
    known = bool((reserve.get("disk") or {}).get("known", False))
    active = bool(not known or not reserve.get("ready", False))
    if not known:
        severity = "unknown"
    elif bool(reserve.get("emergency_active", False)):
        severity = "emergency"
    elif bool(reserve.get("hard_block", False)):
        severity = "hard"
    elif bool(reserve.get("pressure_active", False)):
        severity = "pressure"
    elif active:
        severity = "proactive"
    else:
        severity = "none"
    paper_pause_required = bool(
        reserve.get("pressure_active", False) or reserve.get("hard_block", False)
    )
    reasons = {
        "unknown": "local_hot_storage_free_space_unknown",
        "emergency": "local_hot_storage_emergency_reserve_breached",
        "hard": "local_hot_storage_hard_reserve_breached",
        "pressure": "local_hot_storage_pressure_reserve_breached",
        "proactive": "local_hot_storage_below_unattended_target",
        "none": "",
    }
    target_free_gb = max(
        float(reserve.get("target_free_gb", DEFAULT_TARGET_FREE_GB) or 0.0), 0.0
    )
    recovery_target_free_gb = target_free_gb + max(float(recovery_headroom_gb), 0.0)
    return {
        "active": active,
        "severity": severity,
        "reason": reasons[severity],
        "reserve_deficit_gb": reserve.get("reserve_deficit_gb", 0.0),
        "warning_target_free_gb": round(target_free_gb, 3),
        "recovery_headroom_gb": round(max(float(recovery_headroom_gb), 0.0), 3),
        "recovery_target_free_gb": round(recovery_target_free_gb, 3),
        "paper_pause_required": paper_pause_required,
        "collection_may_continue": bool(known and not paper_pause_required),
        "delegated_controller": "soak_self_healing_control",
        "command": [
            str(project_root / "scripts" / "ops" / "opsctl.sh"),
            "soak-self-heal",
            "--apply",
            "--storage-target-free-gb",
            str(round(recovery_target_free_gb, 3)),
            "--json",
        ],
        "safe_action_classes": [
            "same_inode_bot_log_tail_compaction",
            "manifest_verified_raw_training_compaction",
            "verified_ops_data_plane_schema_drift_rollup_compaction",
            "retention_policy_expiry",
            "manifest_verified_cold_archive_offload",
        ],
        "forbidden_action_classes": [
            "active_sqlite_deletion",
            "unverified_artifact_deletion",
            "credential_or_auth_state_deletion",
            "operating_system_or_user_cache_deletion",
        ],
    }


def build_payload(
    project_root: Path,
    *,
    apply: bool,
    override_path: Path,
    log_root: Path,
    additional_log_roots: list[Path] | None,
    max_log_bytes: int,
    tail_bytes: int,
    reconcile_governor: bool = True,
    target_free_gb: float = DEFAULT_TARGET_FREE_GB,
    recovery_headroom_gb: float = DEFAULT_RECOVERY_HEADROOM_GB,
    pressure_free_gb: float = DEFAULT_PRESSURE_FREE_GB,
    hard_free_gb: float = DEFAULT_HARD_FREE_GB,
    emergency_free_gb: float = DEFAULT_EMERGENCY_FREE_GB,
) -> dict[str, Any]:
    reserve_kwargs = {
        "target_free_gb": max(float(target_free_gb), 0.0),
        "pressure_free_gb": max(float(pressure_free_gb), 0.0),
        "hard_free_gb": max(float(hard_free_gb), 0.0),
        "emergency_free_gb": max(float(emergency_free_gb), 0.0),
    }
    reserve_before = local_storage_reserve_contract(project_root, **reserve_kwargs)
    logs = cap_launchd_log_roots(
        [log_root, *(additional_log_roots or [])],
        max_bytes=max_log_bytes,
        tail_bytes=tail_bytes,
        apply=apply,
    )
    reserve = local_storage_reserve_contract(project_root, **reserve_kwargs)
    changed = _write_override(override_path, reserve["control_env"]) if apply else False
    governor = {"attempted": False, "ok": True}
    if (
        apply
        and reconcile_governor
        and not bool(reserve.get("emergency_active", False))
    ):
        governor = _reconcile_storage_governor(project_root, timeout_seconds=45)
    routes = telemetry_route_contract(project_root)
    route_pressure = fallback_route_pressure_contract(project_root, reserve)
    hard_blockers: list[str] = []
    warnings: list[str] = []
    pressure_active = bool(reserve.get("pressure_active", False))
    if bool(reserve.get("hard_block", False)):
        hard_blockers.append("local_hot_storage_below_hard_reserve")
    elif pressure_active:
        warnings.append("local_hot_storage_pressure_reserve_breached")
    elif not bool(reserve.get("ready", False)):
        warnings.append("local_hot_storage_below_unattended_target")
    if int(logs.get("error_count", 0) or 0) > 0:
        warnings.append("launchd_log_cap_errors")
    if not bool(routes.get("ready", False)):
        hard_blockers.append("external_telemetry_spill_route_not_ready")
    if bool(route_pressure.get("route_rehome_required", False)):
        warnings.append("active_local_fallback_route_under_local_storage_pressure")
        if not bool(route_pressure.get("route_rehome_ready", False)):
            hard_blockers.append("active_local_fallback_route_rehome_not_ready")
    if bool(governor.get("attempted", False)) and not bool(governor.get("ok", False)):
        warnings.append("storage_governor_reconciliation_failed")
    status = (
        "blocked"
        if hard_blockers
        else ("degraded" if pressure_active else ("watch" if warnings else "ready"))
    )
    recovery = _recovery_request(
        reserve,
        project_root,
        recovery_headroom_gb=max(float(recovery_headroom_gb), 0.0),
    )
    recovery["storage_route_recovery"] = route_pressure
    recovery["ordered_recovery_pipeline"] = list(
        route_pressure.get("ordered_recovery_pipeline") or []
    )
    if bool(route_pressure.get("route_rehome_required", False)):
        recovery["delegated_controller"] = "storage_switch_orchestrator"
        recovery["command"] = [
            str(project_root / "scripts" / "ops" / "opsctl.sh"),
            "storage-switch-external",
        ]
        recovery["follow_up_commands"] = [
            [
                str(project_root / "scripts" / "ops" / "opsctl.sh"),
                "storage-prune-standby",
                "--apply",
                "--json",
            ],
            [
                str(project_root / "scripts" / "ops" / "opsctl.sh"),
                "storage-retention-unison",
                "--json",
            ],
        ]
        recovery["recovery_reason"] = (
            "local fallback is the active storage route while internal disk reserve is under pressure"
        )
    free_before = float(reserve_before.get("free_gb", 0.0) or 0.0)
    free_after = float(reserve.get("free_gb", 0.0) or 0.0)
    cleanup_verification = {
        "verified": int(logs.get("error_count", 0) or 0) == 0,
        "free_gb_before": round(free_before, 3),
        "free_gb_after": round(free_after, 3),
        "free_gb_delta": round(free_after - free_before, 3),
        "file_bytes_reclaimed": int(logs.get("bytes_reclaimed", 0) or 0),
        "capped_file_count": int(logs.get("capped_count", 0) or 0),
        "verification_failed_count": int(logs.get("verification_failed_count", 0) or 0),
        "verification_basis": "post-write size and inode checks; disk free delta is advisory under concurrent writes",
    }
    return {
        "timestamp_utc": reserve.get("timestamp_utc"),
        "schema_version": 2,
        "ok": not hard_blockers,
        "overall_status": status,
        "grade": (
            "F"
            if hard_blockers
            else ("C" if pressure_active else ("A" if warnings else "A+"))
        ),
        "apply": bool(apply),
        "local_storage_reserve": reserve,
        "launchd_log_guard": logs,
        "cleanup_verification": cleanup_verification,
        "recovery_request": recovery,
        "safety_contract": {
            "automatic_mutation_scope": "configured bot-owned launchd log roots only",
            "log_symlinks_followed": False,
            "active_log_inode_preservation_required": True,
            "log_mutation_verified": bool(cleanup_verification["verified"]),
            "retained_tail_bytes": int(tail_bytes),
            "stateful_artifact_deletion_delegated_to": "manifest_and_retention_backed_storage_controllers",
            "os_and_user_cache_cleanup": "operator-approved only",
            "reserve_hysteresis": {
                "warning_target_free_gb": round(max(float(target_free_gb), 0.0), 3),
                "recovery_headroom_gb": round(max(float(recovery_headroom_gb), 0.0), 3),
                "recovery_target_free_gb": round(
                    max(float(target_free_gb), 0.0)
                    + max(float(recovery_headroom_gb), 0.0),
                    3,
                ),
            },
        },
        "telemetry_route_contract": routes,
        "fallback_route_pressure_contract": route_pressure,
        "storage_governor_reconciliation": governor,
        "override_path": str(override_path),
        "override_changed": bool(changed),
        "hard_blockers": hard_blockers,
        "warnings": warnings,
        "next_action": (
            "continue unattended collection with the live reserve guard active"
            if status == "ready"
            else (
                "switch active storage back to BOT_LOGS, then prune verified local standby"
                if bool(route_pressure.get("route_rehome_required", False))
                else (
                    "run bounded storage recovery while paper collection continues"
                    if recovery.get("active")
                    and not recovery.get("paper_pause_required")
                    and not hard_blockers
                    else "restore reserve or telemetry routing before unattended collection"
                )
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce live local-disk reserve and bounded launchd logs."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--history-file", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument(
        "--history-max-lines", type=int, default=DEFAULT_HISTORY_MAX_LINES
    )
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument(
        "--additional-log-root",
        action="append",
        default=None,
        help="Additional recursively scanned bot launchd log root; repeat as needed.",
    )
    parser.add_argument(
        "--max-log-bytes",
        type=int,
        default=int(os.getenv("BOT_LAUNCHD_LOG_MAX_BYTES", DEFAULT_MAX_LOG_BYTES)),
    )
    parser.add_argument(
        "--tail-bytes",
        type=int,
        default=int(os.getenv("BOT_LAUNCHD_LOG_TAIL_BYTES", DEFAULT_TAIL_BYTES)),
    )
    parser.add_argument(
        "--target-free-gb",
        type=float,
        default=_env_float("BOT_LOCAL_STORAGE_TARGET_FREE_GB", DEFAULT_TARGET_FREE_GB),
    )
    parser.add_argument(
        "--pressure-free-gb",
        type=float,
        default=_env_float(
            "BOT_LOCAL_STORAGE_PRESSURE_FREE_GB", DEFAULT_PRESSURE_FREE_GB
        ),
    )
    parser.add_argument(
        "--recovery-headroom-gb",
        type=float,
        default=_env_float(
            "BOT_LOCAL_STORAGE_RECOVERY_HEADROOM_GB", DEFAULT_RECOVERY_HEADROOM_GB
        ),
    )
    parser.add_argument(
        "--hard-free-gb",
        type=float,
        default=_env_float("BOT_LOCAL_STORAGE_HARD_FREE_GB", DEFAULT_HARD_FREE_GB),
    )
    parser.add_argument(
        "--emergency-free-gb",
        type=float,
        default=_env_float(
            "BOT_LOCAL_STORAGE_EMERGENCY_FREE_GB", DEFAULT_EMERGENCY_FREE_GB
        ),
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--skip-governor-reconcile", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    configured_additional = args.additional_log_root
    if configured_additional is None:
        env_roots = [
            part.strip()
            for part in os.getenv("BOT_ADDITIONAL_LAUNCHD_LOG_ROOTS", "").split(
                os.pathsep
            )
            if part.strip()
        ]
        configured_additional = env_roots or [str(DEFAULT_USER_LOG_ROOT)]

    project_root = Path(args.project_root).expanduser().resolve()
    lock_path = Path(args.lock_file).expanduser()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        busy_payload = {
            "ok": True,
            "busy": True,
            "overall_status": "already_running",
            "reason": "non_overlapping_guard_lock_held",
            "lock_path": str(lock_path),
        }
        if args.json:
            print(json.dumps(busy_payload, ensure_ascii=True, separators=(",", ":")))
        else:
            print("local_storage_reserve_guard status=already_running")
        lock_handle.close()
        return 0

    out_path = Path(args.out_file).expanduser()
    previous = _load_json(out_path)
    try:
        payload = build_payload(
            project_root,
            apply=bool(args.apply),
            override_path=Path(args.override_file).expanduser(),
            log_root=Path(args.log_root).expanduser(),
            additional_log_roots=[
                Path(path).expanduser() for path in configured_additional
            ],
            max_log_bytes=max(int(args.max_log_bytes), 1),
            tail_bytes=max(int(args.tail_bytes), 0),
            reconcile_governor=not bool(args.skip_governor_reconcile),
            target_free_gb=max(float(args.target_free_gb), 0.0),
            recovery_headroom_gb=max(float(args.recovery_headroom_gb), 0.0),
            pressure_free_gb=max(float(args.pressure_free_gb), 0.0),
            hard_free_gb=max(float(args.hard_free_gb), 0.0),
            emergency_free_gb=max(float(args.emergency_free_gb), 0.0),
        )
        payload["busy"] = False
        payload["lock_path"] = str(lock_path)
        history_event = bool(args.apply and _history_event_required(previous, payload))
        history_result = {
            "ok": True,
            "appended": False,
            "path": str(Path(args.history_file).expanduser()),
            "max_lines": max(int(args.history_max_lines), 1),
        }
        if history_event:
            history_result = _append_history(
                Path(args.history_file).expanduser(),
                payload,
                max_lines=max(int(args.history_max_lines), 1),
            )
        payload["history"] = history_result
        if not bool(history_result.get("ok", False)):
            payload["warnings"] = [
                *list(payload.get("warnings") or []),
                "storage_guard_history_write_failed",
            ]
            if payload.get("overall_status") == "ready":
                payload["overall_status"] = "watch"
                payload["grade"] = "A"
        _write_json(out_path, payload)
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
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
