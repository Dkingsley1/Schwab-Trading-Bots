#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "ingestion_storage_governor_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.storage_pressure_override"


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


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _queue_watermarks(
    *,
    core_pending_lines: int,
    deferred_pending_lines: int,
    cold_pending_lines: int,
    support_pending_lines: int,
    stale_stage_pending_lines: int,
) -> dict[str, Any]:
    rows = {
        "core": {
            "pending_lines": int(core_pending_lines),
            "target": 5000,
            "elevated_threshold": 15000,
            "hard_threshold": 50000,
        },
        "deferred": {
            "pending_lines": int(deferred_pending_lines),
            "target": 25000,
            "elevated_threshold": 100000,
            "hard_threshold": 250000,
        },
        "cold": {
            "pending_lines": int(cold_pending_lines),
            "target": 5000,
            "elevated_threshold": 10000,
            "hard_threshold": 100000,
        },
        "support_telemetry": {
            "pending_lines": int(support_pending_lines),
            "target": 5000,
            "elevated_threshold": 50000,
            "hard_threshold": 150000,
        },
        "stale_stage": {
            "pending_lines": int(stale_stage_pending_lines),
            "target": 0,
            "elevated_threshold": 10000,
            "hard_threshold": 100000,
        },
    }
    breached = {"hard": [], "elevated": [], "target": []}
    for lane, row in rows.items():
        pending = int(row["pending_lines"])
        target = int(row["target"])
        elevated_threshold = int(row["elevated_threshold"])
        hard_threshold = int(row["hard_threshold"])
        row["target_breached"] = pending > target
        row["elevated_breached"] = pending >= elevated_threshold
        row["hard_breached"] = pending >= hard_threshold
        row["distance_to_target"] = max(pending - target, 0)
        if row["hard_breached"]:
            breached["hard"].append(lane)
        if row["elevated_breached"]:
            breached["elevated"].append(lane)
        if row["target_breached"]:
            breached["target"].append(lane)
    overall = "ready"
    if breached["hard"]:
        overall = "blocked"
    elif breached["elevated"]:
        overall = "degraded"
    elif breached["target"]:
        overall = "watch"
    return {
        "overall_status": overall,
        "lanes": rows,
        "breaches": breached,
    }


def _writer_shedding_contract(
    *,
    profile_name: str,
    route_drift: bool,
    queue_watermarks: dict[str, Any],
) -> dict[str, Any]:
    hard_breaches = list(((queue_watermarks.get("breaches") or {}).get("hard") or []))
    elevated_breaches = list(((queue_watermarks.get("breaches") or {}).get("elevated") or []))
    level = "normal"
    if profile_name == "critical_backpressure":
        level = "protect_core"
    elif profile_name == "elevated_backpressure":
        level = "trim_background"
    active = level != "normal" or bool(route_drift)
    notes = []
    if route_drift:
        notes.append("writer route drift keeps live writes pinned to the routed repo DB path until storage alignment is restored")
    if "support_telemetry" in hard_breaches or "support_telemetry" in elevated_breaches:
        notes.append("support telemetry should stay shard-isolated so watchdog chatter cannot crowd the core writer")
    if "stale_stage" in hard_breaches or "stale_stage" in elevated_breaches:
        notes.append("stale-stage backlog should be reaped or archived instead of treated as hot-path ingestion")
    return {
        "active": active,
        "level": level,
        "freeze_cold_lanes": profile_name == "critical_backpressure",
        "throttle_deferred_lanes": profile_name in {"critical_backpressure", "elevated_backpressure"},
        "shed_support_telemetry": "support_telemetry" in hard_breaches or "support_telemetry" in elevated_breaches,
        "suppress_verbose_decision_logs": profile_name in {"critical_backpressure", "elevated_backpressure"},
        "route_drift_override": bool(route_drift),
        "hard_breaches": hard_breaches,
        "elevated_breaches": elevated_breaches,
        "notes": notes,
    }


def _override_lines(profile_name: str, env_overrides: dict[str, str]) -> list[str]:
    lines = [
        "# Auto-managed by scripts/ops/ingestion_storage_governor.py",
        f"BOT_INGESTION_STORAGE_PROFILE={profile_name}",
    ]
    for key, value in sorted(env_overrides.items()):
        lines.append(f"{key}={value}")
    return lines


def _write_override(path: Path, profile_name: str, env_overrides: dict[str, str]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(_override_lines(profile_name, env_overrides)) + "\n"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _storage_profile(
    *,
    hard_gate: bool,
    core_pending_lines: int,
    deferred_pending_lines: int,
    cold_pending_lines: int,
    support_pending_lines: int,
    stale_stage_pending_lines: int,
    retention_debt_gb: float,
    route_drift: bool,
    pressure_index: float = 0.0,
    storage_severity: str = "",
) -> str:
    effective_deferred_pending_lines = max(
        int(deferred_pending_lines) - int(stale_stage_pending_lines) - int(support_pending_lines),
        0,
    )
    effective_cold_pending_lines = max(int(cold_pending_lines) - int(stale_stage_pending_lines), 0)
    if (
        hard_gate
        or pressure_index >= 1.0
        or (str(storage_severity).strip().lower() in {"high", "critical", "blocked"} and core_pending_lines >= 15000)
        or core_pending_lines >= 50000
        or effective_deferred_pending_lines >= 250000
        or effective_cold_pending_lines >= 100000
        or retention_debt_gb >= 5.0
        or route_drift
    ):
        return "critical_backpressure"
    if (
        core_pending_lines >= 15000
        or effective_deferred_pending_lines >= 100000
        or effective_cold_pending_lines >= 10000
        or support_pending_lines >= 100000
        or stale_stage_pending_lines >= 100000
        or retention_debt_gb > 0.0
    ):
        return "elevated_backpressure"
    return "steady_state"


def _critical_deferred_budget(*, core_pending_lines: int, deferred_pending_lines: int, route_drift: bool) -> int:
    if route_drift or deferred_pending_lines <= 0:
        return 0
    if core_pending_lines <= 5000:
        return 2
    if core_pending_lines <= 15000:
        return 1
    return 0


def _profile_env(
    profile_name: str,
    project_root: Path,
    *,
    core_pending_lines: int = 0,
    deferred_pending_lines: int = 0,
    route_drift: bool = False,
) -> dict[str, str]:
    routed_primary_db = str(project_root / "data" / "jsonl_link.sqlite3")
    routed_queue_db = str(project_root / "data" / "bot_channel_queue.sqlite3")
    base = {
        "SQL_LINK_SERVICE_PRIMARY_DB": routed_primary_db,
        "BOT_CHANNEL_QUEUE_DB": routed_queue_db,
        "SQL_LINK_SERVICE_QUEUE_DB": routed_queue_db,
        "SQL_LINK_SERVICE_QUEUE_PRUNE_ORPHANS": "1",
        "SQL_LINK_SERVICE_QUEUE_ORPHAN_DAYS": "14",
        "SQL_LINK_SERVICE_AUTO_LOCAL_FALLBACK_PRUNE": "1",
        "SQL_LINK_SERVICE_LOCAL_FALLBACK_PRUNE_OLDER_THAN_SECONDS": "21600",
        "SQL_LINK_SERVICE_LOCAL_FALLBACK_PRUNE_MAX_FILES": "400",
        "SQL_LINK_SERVICE_QUEUE_MAX_DB_GB": "12",
        "SQL_LINK_SERVICE_QUEUE_MAX_ROWS": "240000",
        "RETENTION_STALE_STAGE_ENABLED": "1",
        "RETENTION_STALE_PURGE_ENABLED": "1",
        "RETENTION_STALE_PURGE_DAYS": "30",
        "RETENTION_STALE_PURGE_LOW_VALUE_DAYS": "7",
        "RETENTION_STALE_PURGE_MEDIUM_VALUE_DAYS": "21",
        "RETENTION_STALE_PURGE_HIGH_VALUE_DAYS": "45",
        "RETENTION_STALE_PURGE_CRITICAL_VALUE_DAYS": "90",
        "RETENTION_STALE_PURGE_MAX_FILES": "5000",
        "RETENTION_STALE_PURGE_MAX_GB": "10",
    }
    if profile_name == "critical_backpressure":
        deferred_budget = _critical_deferred_budget(
            core_pending_lines=core_pending_lines,
            deferred_pending_lines=deferred_pending_lines,
            route_drift=route_drift,
        )
        explanation_max_files = "4" if deferred_budget >= 2 else "3" if deferred_budget == 1 else "2"
        base.update(
            {
                "INGEST_MAX_DEFERRED_FILES": str(deferred_budget),
                "JSONL_SQL_MAX_COLD_LANE_FILES": "0",
                "LOG_DATA_INGRESS": "0",
                "LOG_API_CALLS": "0",
                "LOG_LOOP_STATE": "0",
                "LOG_GATE_EVALUATIONS": "0",
                "LOG_GATE_PASSES": "0",
                "LOG_SUB_BOT_DECISIONS": "0",
                "LOG_MASTER_VARIANT_DECISIONS": "0",
                "LOG_DECISION_EXPLANATIONS": "0",
                "LOG_SHADOW_PNL_ATTRIBUTION": "0",
                "CHANNEL_LOG_PRIMARY_MODE": "channel",
                "LEGACY_HOT_CHANNEL_MIRROR_ENABLED": "0",
                "SQL_LINK_SERVICE_INTERVAL_SECONDS": "20",
                "SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB": "0.5",
                "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB": "0.5",
                "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_ROWS": "250000",
                "SQL_LINK_SERVICE_WAL_CHECKPOINT_MIN_INTERVAL_SECONDS": "60",
                "SQL_LINK_SERVICE_WAL_TRUNCATE_MAX_GB": "2",
                "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "60",
                "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "180000",
                "SQL_LINK_SERVICE_HOT_MAX_ROWS": "1800000",
                "SQL_LINK_SERVICE_HOT_DAYS": "3",
                "SQL_LINK_SERVICE_QUEUE_MAX_DB_GB": "8",
                "SQL_LINK_SERVICE_QUEUE_MAX_ROWS": "180000",
                "SQL_LINK_SERVICE_QUEUE_ORPHAN_DAYS": "7",
                "SQL_LINK_SERVICE_LOCAL_FALLBACK_PRUNE_MAX_FILES": "800",
                "RETENTION_STALE_PURGE_LOW_VALUE_DAYS": "3",
                "RETENTION_STALE_PURGE_MEDIUM_VALUE_DAYS": "14",
                "RETENTION_STALE_PURGE_HIGH_VALUE_DAYS": "30",
                "RETENTION_STALE_PURGE_CRITICAL_VALUE_DAYS": "90",
                "RETENTION_STALE_PURGE_MAX_FILES": "8000",
                "RETENTION_STALE_PURGE_MAX_GB": "20",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_FILES": explanation_max_files,
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_FILES": explanation_max_files,
                "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_MAX_LINES_PER_FILE": "96000",
                "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_STATE_CHECKPOINT_LINES": "4000",
                "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_MAX_FILES": "1",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_MAX_FILES": "1",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MAX_DB_GB": "4",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_TRIGGER_GROWTH_GB": "0.5",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_TRIGGER_ROWS": "120000",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_HOT_DAYS": "2",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_BATCH_SIZE": "180000",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MAX_ROWS": "1200000",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MIN_INTERVAL_SECONDS": "60",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_ARCHIVE_PERIOD": "day",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_ARCHIVE_RETENTION_DAYS": "30",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_VACUUM_THRESHOLD_GB": "2",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_MAX_DB_GB": "3",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_TRIGGER_GROWTH_GB": "0.4",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_TRIGGER_ROWS": "100000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_HOT_DAYS": "2",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_BATCH_SIZE": "160000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_MAX_ROWS": "900000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_MIN_INTERVAL_SECONDS": "60",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_ARCHIVE_PERIOD": "day",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_ARCHIVE_RETENTION_DAYS": "21",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_VACUUM_THRESHOLD_GB": "1.5",
                "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_BATCH_SIZE": "220000",
                "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_MAX_ROWS": "1500000",
                "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_MIN_INTERVAL_SECONDS": "60",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_BATCH_SIZE": "220000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_MAX_ROWS": "1500000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_MIN_INTERVAL_SECONDS": "60",
            }
        )
        return base
    if profile_name == "elevated_backpressure":
        base.update(
            {
                "INGEST_MAX_DEFERRED_FILES": "1",
                "JSONL_SQL_MAX_COLD_LANE_FILES": "0",
                "LOG_DATA_INGRESS": "1",
                "LOG_API_CALLS": "1",
                "LOG_LOOP_STATE": "1",
                "LOG_GATE_EVALUATIONS": "0",
                "LOG_GATE_PASSES": "0",
                "LOG_SHADOW_PNL_ATTRIBUTION": "0",
                "CHANNEL_LOG_PRIMARY_MODE": "channel",
                "LEGACY_HOT_CHANNEL_MIRROR_ENABLED": "0",
                "SQL_LINK_SERVICE_INTERVAL_SECONDS": "30",
                "SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB": "0.75",
                "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB": "0.75",
                "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_ROWS": "400000",
                "SQL_LINK_SERVICE_WAL_CHECKPOINT_MIN_INTERVAL_SECONDS": "120",
                "SQL_LINK_SERVICE_WAL_TRUNCATE_MAX_GB": "3",
                "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "120",
                "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "140000",
                "SQL_LINK_SERVICE_HOT_MAX_ROWS": "1200000",
                "SQL_LINK_SERVICE_HOT_DAYS": "4",
                "SQL_LINK_SERVICE_QUEUE_MAX_DB_GB": "10",
                "SQL_LINK_SERVICE_QUEUE_MAX_ROWS": "220000",
                "SQL_LINK_SERVICE_QUEUE_ORPHAN_DAYS": "10",
                "RETENTION_STALE_PURGE_LOW_VALUE_DAYS": "5",
                "RETENTION_STALE_PURGE_MEDIUM_VALUE_DAYS": "21",
                "RETENTION_STALE_PURGE_HIGH_VALUE_DAYS": "45",
                "RETENTION_STALE_PURGE_CRITICAL_VALUE_DAYS": "90",
                "RETENTION_STALE_PURGE_MAX_FILES": "6500",
                "RETENTION_STALE_PURGE_MAX_GB": "12",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_FILES": "4",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_FILES": "4",
                "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_MAX_LINES_PER_FILE": "96000",
                "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_STATE_CHECKPOINT_LINES": "4000",
                "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_MAX_FILES": "2",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_MAX_FILES": "2",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MAX_DB_GB": "6",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_TRIGGER_GROWTH_GB": "0.75",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_TRIGGER_ROWS": "160000",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_HOT_DAYS": "3",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_BATCH_SIZE": "140000",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MAX_ROWS": "900000",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MIN_INTERVAL_SECONDS": "90",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_ARCHIVE_PERIOD": "day",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_ARCHIVE_RETENTION_DAYS": "45",
                "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_VACUUM_THRESHOLD_GB": "2.5",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_MAX_DB_GB": "4.5",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_TRIGGER_GROWTH_GB": "0.5",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_TRIGGER_ROWS": "120000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_HOT_DAYS": "3",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_BATCH_SIZE": "120000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_MAX_ROWS": "700000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_MIN_INTERVAL_SECONDS": "90",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_ARCHIVE_PERIOD": "day",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_ARCHIVE_RETENTION_DAYS": "30",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_VACUUM_THRESHOLD_GB": "2",
                "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_BATCH_SIZE": "180000",
                "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_MAX_ROWS": "1000000",
                "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_MIN_INTERVAL_SECONDS": "90",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_BATCH_SIZE": "180000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_MAX_ROWS": "1000000",
                "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_MIN_INTERVAL_SECONDS": "90",
            }
        )
        return base
    base.update(
        {
            "INGEST_MAX_DEFERRED_FILES": "2",
            "JSONL_SQL_MAX_COLD_LANE_FILES": "1",
            "LOG_DATA_INGRESS": "1",
            "LOG_API_CALLS": "1",
            "LOG_LOOP_STATE": "1",
            "LOG_SHADOW_PNL_ATTRIBUTION": "1",
            "SQL_LINK_SERVICE_INTERVAL_SECONDS": "45",
            "SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB": "2",
            "RETENTION_STALE_PURGE_MAX_GB": "8",
        }
    )
    return base


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    action: str,
    changed: bool = False,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    storage_control = _load_json(health_root / "ingestion_storage_control_latest.json")
    backpressure = _load_json(health_root / "ingestion_backpressure_latest.json")
    queue = _load_json(health_root / "ingestion_priority_queue_latest.json")
    mount = _load_json(health_root / "storage_mount_guard_latest.json")
    failback = _load_json(health_root / "storage_failback_sync_latest.json")
    split_brain = _load_json(health_root / "storage_split_brain_reconciler_latest.json")
    sql_service = _load_json(health_root / "sql_link_service_latest.json")
    sql_progress = _load_json(health_root / "sql_link_service_progress_latest.json")
    health_gates = _load_json(health_root / "health_gates_latest.json")

    core_pending_lines = _safe_int(backpressure.get("pending_lines"), _safe_int(((queue.get("lane_counts") or {}).get("core") or {}).get("pending_lines"), 0))
    deferred_pending_lines = _safe_int(backpressure.get("pending_lines_deferred"), _safe_int(((queue.get("lane_counts") or {}).get("deferred") or {}).get("pending_lines"), 0))
    cold_pending_lines = _safe_int(backpressure.get("pending_lines_cold"), _safe_int(((queue.get("lane_counts") or {}).get("cold") or {}).get("pending_lines"), 0))
    support_pending_lines = _safe_int(backpressure.get("pending_lines_support_telemetry"), 0)
    stale_stage_pending_lines = _safe_int(backpressure.get("pending_lines_stale_stage"), 0)
    retention_debt_gb = _safe_float(((storage_control.get("storage") or {}).get("retention_debt_gb")), _safe_float(((health_gates.get("storage_pressure") or {}).get("retention_debt_gb")), 0.0))
    pressure_index = _safe_float(storage_control.get("pressure_index"), 0.0)
    storage_severity = str(storage_control.get("severity") or "")
    hard_gate_flags = health_gates.get("hard_gates") if isinstance(health_gates.get("hard_gates"), dict) else {}
    storage_hard_gate = any(
        bool(hard_gate_flags.get(key, False))
        for key in (
            "ingestion_pending_lines",
            "ingestion_oldest_age",
            "ingestion_invalid_lines",
            "ingestion_backpressure_overload",
            "priority_shard_storage",
            "sql_progress_stall",
            "sql_wal_pressure",
        )
    )
    hard_gate = bool(
        str(storage_control.get("overall_status") or "") == "blocked"
        or (storage_hard_gate if hard_gate_flags else bool(health_gates.get("hard_gate_triggered", False)))
    )

    current_primary_db = str(sql_progress.get("primary_db") or sql_service.get("primary_db") or "").strip()
    current_primary_db_realpath = str(sql_progress.get("primary_db_realpath") or sql_service.get("primary_db_realpath") or current_primary_db)
    split_brain_conflicts = max(
        _safe_int(failback.get("split_brain_conflicts"), 0),
        _safe_int(((split_brain.get("summary") or {}).get("unresolved_conflicts")), 0),
    )
    storage_mode = str(mount.get("storage_mode") or failback.get("certified_mode") or failback.get("mode") or "")
    storage_external = bool(mount.get("external_available", False)) and storage_mode in {"external", "external_curated"}
    route_drift = bool(
        storage_external
        and split_brain_conflicts == 0
        and (
            "/local_fallback_storage/" in current_primary_db
            or "/local_fallback_storage/" in current_primary_db_realpath
        )
    )

    profile_name = _storage_profile(
        hard_gate=hard_gate,
        core_pending_lines=core_pending_lines,
        deferred_pending_lines=deferred_pending_lines,
        cold_pending_lines=cold_pending_lines,
        support_pending_lines=support_pending_lines,
        stale_stage_pending_lines=stale_stage_pending_lines,
        retention_debt_gb=retention_debt_gb,
        route_drift=route_drift,
        pressure_index=pressure_index,
        storage_severity=storage_severity,
    )
    env_overrides = _profile_env(
        profile_name,
        project_root,
        core_pending_lines=core_pending_lines,
        deferred_pending_lines=deferred_pending_lines,
        route_drift=route_drift,
    )
    queue_watermarks = _queue_watermarks(
        core_pending_lines=core_pending_lines,
        deferred_pending_lines=deferred_pending_lines,
        cold_pending_lines=cold_pending_lines,
        support_pending_lines=support_pending_lines,
        stale_stage_pending_lines=stale_stage_pending_lines,
    )
    writer_shedding = _writer_shedding_contract(
        profile_name=profile_name,
        route_drift=route_drift,
        queue_watermarks=queue_watermarks,
    )
    top_actions: list[str] = []
    if route_drift:
        top_actions.append("normalize SQL linker back to the routed primary DB path and restart the writer service")
    if deferred_pending_lines > 0:
        top_actions.append("keep deferred ingestion quota-limited until core drain stays under 30 minutes")
    if support_pending_lines > 0:
        top_actions.append("route watchdog failover and pager telemetry through the support shard so it stops crowding governance backlog")
    if cold_pending_lines > 0:
        top_actions.append("hold shadow PnL attribution cold-lane ingestion at zero until the cold backlog clears")
    if stale_stage_pending_lines > 0:
        top_actions.append("treat stale-stage debt as archive or reap work instead of generic hot-path ingestion pressure")
    if retention_debt_gb > 0.0:
        top_actions.append("run aggressive explanation and attribution shard hot retention until retention debt is near zero")
    if hard_gate:
        top_actions.append("treat storage pressure as maintenance priority over new training and research work")
    if bool(writer_shedding.get("shed_support_telemetry", False)):
        top_actions.append("shed support telemetry into shard-isolated writes until support backlog drops under its elevated watermark")
    if bool(writer_shedding.get("freeze_cold_lanes", False)):
        top_actions.append("keep cold-lane ingestion frozen while the core queue remains under active protection")

    notes = [
        "the governor writes a dedicated storage-pressure override so manual storage route switches can still live in config/.env.storage_override",
        "SQL_LINK_SERVICE_PRIMARY_DB is pinned to the routed repo path instead of a resolved fallback path so future failback transitions can follow symlink routing cleanly",
    ]
    if profile_name == "critical_backpressure":
        notes.append(
            "critical profile keeps cold lanes at zero and only reopens a small deferred trickle once the core queue is nearly clear and routing is healthy"
        )
    elif profile_name == "elevated_backpressure":
        notes.append("elevated profile keeps core ingestion moving while limiting shadow PnL attribution and reducing deferred fan-in")
    else:
        notes.append("steady-state profile preserves the routed primary DB path and relaxed deferred/cold quotas")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": True,
        "action": action,
        "profile": profile_name,
        "changed": bool(changed),
        "override_path": str(override_path),
        "override_exists": bool(override_path.exists()),
        "storage": {
            "external_available": bool(mount.get("external_available", False)),
            "storage_mode": str(mount.get("storage_mode") or failback.get("mode") or ""),
            "split_brain_conflicts": int(split_brain_conflicts),
        },
        "pressure": {
            "hard_gate": bool(hard_gate),
            "core_pending_lines": int(core_pending_lines),
            "deferred_pending_lines": int(deferred_pending_lines),
            "cold_pending_lines": int(cold_pending_lines),
            "support_pending_lines": int(support_pending_lines),
            "stale_stage_pending_lines": int(stale_stage_pending_lines),
            "retention_debt_gb": round(float(retention_debt_gb), 3),
            "pressure_index": round(float(pressure_index), 3),
            "storage_severity": storage_severity,
        },
        "queue_watermarks": queue_watermarks,
        "sql_primary_db": {
            "current_path": current_primary_db,
            "current_realpath": current_primary_db_realpath,
            "target_path": str(project_root / "data" / "jsonl_link.sqlite3"),
            "route_drift": bool(route_drift),
        },
        "writer_shedding": writer_shedding,
        "throttle_controls": {
            "deferred_files_budget": _safe_int(env_overrides.get("INGEST_MAX_DEFERRED_FILES"), 0),
            "cold_files_budget": _safe_int(env_overrides.get("JSONL_SQL_MAX_COLD_LANE_FILES"), 0),
            "queue_prune_orphans": env_overrides.get("SQL_LINK_SERVICE_QUEUE_PRUNE_ORPHANS"),
            "queue_orphan_days": _safe_int(env_overrides.get("SQL_LINK_SERVICE_QUEUE_ORPHAN_DAYS"), 0),
            "queue_max_db_gb": _safe_float(env_overrides.get("SQL_LINK_SERVICE_QUEUE_MAX_DB_GB"), 0.0),
            "queue_max_rows": _safe_int(env_overrides.get("SQL_LINK_SERVICE_QUEUE_MAX_ROWS"), 0),
            "local_fallback_prune_enabled": env_overrides.get("SQL_LINK_SERVICE_AUTO_LOCAL_FALLBACK_PRUNE"),
            "local_fallback_prune_max_files": _safe_int(env_overrides.get("SQL_LINK_SERVICE_LOCAL_FALLBACK_PRUNE_MAX_FILES"), 0),
            "stale_stage_enabled": env_overrides.get("RETENTION_STALE_STAGE_ENABLED"),
            "stale_purge_enabled": env_overrides.get("RETENTION_STALE_PURGE_ENABLED"),
            "stale_purge_low_value_days": _safe_int(env_overrides.get("RETENTION_STALE_PURGE_LOW_VALUE_DAYS"), 0),
            "stale_purge_medium_value_days": _safe_int(env_overrides.get("RETENTION_STALE_PURGE_MEDIUM_VALUE_DAYS"), 0),
            "stale_purge_high_value_days": _safe_int(env_overrides.get("RETENTION_STALE_PURGE_HIGH_VALUE_DAYS"), 0),
            "stale_purge_critical_value_days": _safe_int(env_overrides.get("RETENTION_STALE_PURGE_CRITICAL_VALUE_DAYS"), 0),
            "stale_purge_max_files": _safe_int(env_overrides.get("RETENTION_STALE_PURGE_MAX_FILES"), 0),
            "stale_purge_max_gb": _safe_float(env_overrides.get("RETENTION_STALE_PURGE_MAX_GB"), 0.0),
            "log_api_calls": env_overrides.get("LOG_API_CALLS"),
            "log_loop_state": env_overrides.get("LOG_LOOP_STATE"),
            "log_data_ingress": env_overrides.get("LOG_DATA_INGRESS"),
            "log_gate_evaluations": env_overrides.get("LOG_GATE_EVALUATIONS"),
            "log_gate_passes": env_overrides.get("LOG_GATE_PASSES"),
            "log_sub_bot_decisions": env_overrides.get("LOG_SUB_BOT_DECISIONS"),
            "log_master_variant_decisions": env_overrides.get("LOG_MASTER_VARIANT_DECISIONS"),
            "log_decision_explanations": env_overrides.get("LOG_DECISION_EXPLANATIONS"),
            "log_shadow_pnl_attribution": env_overrides.get("LOG_SHADOW_PNL_ATTRIBUTION"),
        },
        "env_overrides": env_overrides,
        "top_actions": top_actions,
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply authoritative ingestion/storage throttles and normalize SQL primary DB routing.")
    parser.add_argument("action", choices=("status", "apply"))
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    override_path = Path(args.override_file).expanduser()
    changed = False
    if args.action == "apply":
        payload_preview = build_payload(project_root, override_path=override_path, action=args.action, changed=False)
        changed = _write_override(override_path, str(payload_preview.get("profile") or "steady_state"), payload_preview.get("env_overrides") if isinstance(payload_preview.get("env_overrides"), dict) else {})
    payload = build_payload(project_root, override_path=override_path, action=args.action, changed=changed)

    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "ingestion_storage_governor "
            f"profile={payload.get('profile', '')} "
            f"changed={int(bool(payload.get('changed', False)))} "
            f"route_drift={int(bool(((payload.get('sql_primary_db') or {}).get('route_drift', False))))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
