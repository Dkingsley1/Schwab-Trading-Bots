#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shlex
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops.long_runtime_common import (
    evidence_freshness,
    parse_iso_utc,
    iso_now,
    load_json,
    run_bounded_process_group,
    write_payload,
)
from scripts.ops.scheduled_lifecycle_common import lifecycle_receipt

SCHEMA_VERSION = 1
HEALTH_DIR = Path("governance/health")
DEFAULT_OUT_REL = HEALTH_DIR / "adaptive_ops_recovery_policy_latest.json"
DEFAULT_OVERRIDE_REL = Path("config/.env.adaptive_ops_recovery_policy_override")
PROTECTED_VOLUME = "/Volumes/VIDEO"
CONTROL_REFRESH_AGE_BUFFER_MINUTES = 5.0

CONTROL_PLANE_REFRESH_STEPS: dict[str, dict[str, Any]] = {
    "schd_evidence_maintenance": {
        "command": ["./scripts/ops/opsctl.sh", "schd-decision-rehearsal", "maintain", "--json"],
        "timeout_seconds": 45,
        "reason": "refresh read-only SCHD candle context and prune expired owned cache; never orders or bot decisions",
    },
    "risk_service_boundary": {
        "command": ["./scripts/ops/opsctl.sh", "risk-service-boundary", "--refresh-inputs", "--json"],
        "timeout_seconds": 30,
        "reason": "refresh watchdog-derived budget evidence before risk inputs expire; no execution authority",
    },
    "system_role_contract": {
        "command": ["./scripts/ops/opsctl.sh", "system-role-contract", "--json"],
        "timeout_seconds": 45,
        "reason": "refresh responsibility and authority contract before soak/dashboard gating",
    },
    "capability_materialization": {
        "command": ["./scripts/ops/opsctl.sh", "capability-materialization", "--json"],
        "timeout_seconds": 45,
        "reason": "refresh materialized capability proofs before unattended soak gating",
    },
    "collector_capability_control": {
        "command": [
            "./scripts/ops/opsctl.sh",
            "collector-capability-control",
            "--json",
        ],
        "timeout_seconds": 45,
        "reason": "refresh collector capability authority before unattended soak gating",
    },
    "health_gates": {
        "command": ["./scripts/ops/opsctl.sh", "health-gates", "--json"],
        "timeout_seconds": 150,
        "reason": "refresh canonical health gates before halt-trigger evaluation",
    },
    "halt_trigger_status": {
        "command": ["./scripts/ops/opsctl.sh", "halt-trigger-status", "--json"],
        "timeout_seconds": 30,
        "reason": "refresh halt-trigger status after health-gates freshness changes",
    },
    "global_halt_refresh": {
        "command": ["./scripts/ops/opsctl.sh", "global-halt-refresh", "--json"],
        "timeout_seconds": 30,
        "reason": "recompute halt blockers from current evidence without clearing the halt or enabling execution",
    },
    "coordination_state": {
        "command": ["./scripts/ops/opsctl.sh", "coordination-status", "--json"],
        "timeout_seconds": 45,
        "reason": "refresh fast coordination state before soak sentinel evaluation",
    },
    "ingestion_storage_control": {
        "command": ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"],
        "timeout_seconds": 45,
        "reason": "refresh storage and ingestion soak contract before soak gates",
    },
    "system_plumbing_control": {
        "command": ["./scripts/ops/opsctl.sh", "system-plumbing-control", "--json"],
        "timeout_seconds": 30,
        "reason": "refresh fast-health plumbing evidence after storage recovery without applying controls",
    },
    "system_architecture_hardening": {
        "command": [
            "./scripts/ops/opsctl.sh",
            "system-architecture-hardening",
            "--json",
        ],
        "timeout_seconds": 30,
        "reason": "refresh architecture evidence after storage and plumbing without applying controls",
    },
    "soak_reliability_sentinel": {
        "command": [
            "./scripts/ops/opsctl.sh",
            "soak-reliability-sentinel",
            "--json",
        ],
        "timeout_seconds": 45,
        "reason": "refresh soak sentinel status after upstream storage or safety control changes",
    },
    "runtime_paper_regression_guard": {
        "command": [
            "./scripts/ops/opsctl.sh",
            "runtime-paper-regression-guard",
            "--json",
        ],
        "timeout_seconds": 30,
        "reason": "recheck paper regression evidence before the soak sentinel without changing execution controls",
    },
    "unattended_soak_readiness": {
        "command": ["./scripts/ops/opsctl.sh", "unattended-soak-readiness", "--json"],
        "timeout_seconds": 45,
        "reason": "refresh unattended soak certificate after upstream readiness changes",
    },
    "runtime_gate_dashboard": {
        "command": ["./scripts/ops/opsctl.sh", "runtime-gate-dashboard", "--json"],
        "timeout_seconds": 45,
        "reason": "refresh runtime dashboard after dependency refreshes",
    },
    "bot_profitability_scalability_control": {
        "command": [
            "./scripts/ops/opsctl.sh",
            "bot-profitability-scalability",
            "--json",
        ],
        "timeout_seconds": 30,
        "reason": "refresh stale advisory profitability evidence without applying portfolio changes",
    },
    "sleeve_scalability_selector": {
        "command": ["./scripts/ops/opsctl.sh", "sleeve-scalability-selector", "--json"],
        "timeout_seconds": 30,
        "reason": "refresh stale advisory sleeve selection after profitability evidence",
    },
    "master_grandmaster_evidence_v2": {
        "command": ["./scripts/ops/opsctl.sh", "master-grandmaster-evidence", "--json"],
        "timeout_seconds": 30,
        "reason": "refresh stale advisory master evidence without teaching or execution authority",
    },
}

SAFETY_ENV = {
    "ADAPTIVE_OPS_RECOVERY_POLICY_ACTIVE": "1",
    "ALLOW_ORDER_EXECUTION": "0",
    "BOT_NEVER_TOUCH_VIDEO": "1",
    "BOT_PROTECTED_VOLUME_DENYLIST": PROTECTED_VOLUME,
    "EXECUTION_LANE_LIVE_ENABLED": "0",
    "MARKET_DATA_ONLY": "1",
    "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR": "0",
    "TOP_BOT_ENABLE_LIVE_EXECUTION": "0",
}

TIER_PROFILES: dict[str, dict[str, str]] = {
    "observe": {
        "ADAPTIVE_OPS_RECOVERY_POLICY_REASON": "platform_within_recovery_band",
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.95",
        "BOT_LOGS_SPACE_RECOVERY_MAX_DELETE_GB": "4",
        "BOT_LOGS_SPACE_RECOVERY_MIN_AGE_HOURS": "24",
        "BOT_LOGS_SPACE_RECOVERY_TARGET_FREE_GB": "110",
        "MAINTENANCE_SLOT_STORAGE_BACKPRESSURE_AUTOPILOT_MIN_INTERVAL_SECONDS": "600",
        "MAINTENANCE_SLOT_STORAGE_PRESSURE_CLEARANCE_MIN_INTERVAL_SECONDS": "600",
        "MAINTENANCE_SLOT_STORAGE_RECOVERY_DEFER_OUTSIDE_QUIET_WINDOW": "1",
        "MAINTENANCE_SLOT_STORAGE_RECOVERY_DEFER_WHILE_SQL_LINK_ACTIVE": "1",
        "MAINTENANCE_SLOT_STORAGE_RECOVERY_FAST_MODE": "0",
        "SOAK_SELF_HEAL_STORAGE_CLEANUP_MAX_DELETE_GB": "4",
        "STORAGE_BACKPRESSURE_AUTOPILOT_BACKPRESSURE_TIMEOUT_SECONDS": "300",
        "STORAGE_BACKPRESSURE_AUTOPILOT_MAX_CYCLES": "1",
        "STORAGE_BACKPRESSURE_AUTOPILOT_POLL_SECONDS": "20",
        "STORAGE_BACKPRESSURE_AUTOPILOT_TIMEOUT_SECONDS": "1200",
        "STORAGE_BACKPRESSURE_AUTOPILOT_WAIT_TIMEOUT_SECONDS": "300",
        "STORAGE_PRESSURE_CLEARANCE_MAX_CYCLES": "1",
        "STORAGE_PRESSURE_CLEARANCE_POLL_SECONDS": "10",
        "STORAGE_PRESSURE_CLEARANCE_TIMEOUT_SECONDS": "900",
        "STORAGE_PRESSURE_CLEARANCE_WAIT_TIMEOUT_SECONDS": "180",
    },
    "watch": {
        "ADAPTIVE_OPS_RECOVERY_POLICY_REASON": "early_pressure_detected",
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.70",
        "BOT_LOGS_SPACE_RECOVERY_MAX_DELETE_GB": "8",
        "BOT_LOGS_SPACE_RECOVERY_MIN_AGE_HOURS": "18",
        "BOT_LOGS_SPACE_RECOVERY_TARGET_FREE_GB": "120",
        "MAINTENANCE_SLOT_STORAGE_BACKPRESSURE_AUTOPILOT_MIN_INTERVAL_SECONDS": "300",
        "MAINTENANCE_SLOT_STORAGE_PRESSURE_CLEARANCE_MIN_INTERVAL_SECONDS": "180",
        "MAINTENANCE_SLOT_STORAGE_RECOVERY_DEFER_OUTSIDE_QUIET_WINDOW": "1",
        "MAINTENANCE_SLOT_STORAGE_RECOVERY_DEFER_WHILE_SQL_LINK_ACTIVE": "1",
        "MAINTENANCE_SLOT_STORAGE_RECOVERY_FAST_MODE": "1",
        "SOAK_SELF_HEAL_STORAGE_CLEANUP_MAX_DELETE_GB": "6",
        "STORAGE_BACKPRESSURE_AUTOPILOT_BACKPRESSURE_TIMEOUT_SECONDS": "240",
        "STORAGE_BACKPRESSURE_AUTOPILOT_MAX_CYCLES": "1",
        "STORAGE_BACKPRESSURE_AUTOPILOT_POLL_SECONDS": "12",
        "STORAGE_BACKPRESSURE_AUTOPILOT_TIMEOUT_SECONDS": "900",
        "STORAGE_BACKPRESSURE_AUTOPILOT_WAIT_TIMEOUT_SECONDS": "240",
        "STORAGE_PRESSURE_CLEARANCE_MAX_CYCLES": "1",
        "STORAGE_PRESSURE_CLEARANCE_POLL_SECONDS": "8",
        "STORAGE_PRESSURE_CLEARANCE_TIMEOUT_SECONDS": "750",
        "STORAGE_PRESSURE_CLEARANCE_WAIT_TIMEOUT_SECONDS": "150",
    },
    "hot_recovery": {
        "ADAPTIVE_OPS_RECOVERY_POLICY_REASON": "hot_path_or_storage_pressure_detected",
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.35",
        "BOT_LOGS_SPACE_RECOVERY_MAX_DELETE_GB": "12",
        "BOT_LOGS_SPACE_RECOVERY_MIN_AGE_HOURS": "12",
        "BOT_LOGS_SPACE_RECOVERY_TARGET_FREE_GB": "125",
        "MAINTENANCE_SLOT_STORAGE_BACKPRESSURE_AUTOPILOT_MIN_INTERVAL_SECONDS": "120",
        "MAINTENANCE_SLOT_STORAGE_PRESSURE_CLEARANCE_MIN_INTERVAL_SECONDS": "90",
        "MAINTENANCE_SLOT_STORAGE_RECOVERY_DEFER_OUTSIDE_QUIET_WINDOW": "0",
        "MAINTENANCE_SLOT_STORAGE_RECOVERY_DEFER_WHILE_SQL_LINK_ACTIVE": "1",
        "MAINTENANCE_SLOT_STORAGE_RECOVERY_FAST_MODE": "1",
        "SOAK_SELF_HEAL_STORAGE_CLEANUP_MAX_DELETE_GB": "10",
        "STORAGE_BACKPRESSURE_AUTOPILOT_BACKPRESSURE_TIMEOUT_SECONDS": "180",
        "STORAGE_BACKPRESSURE_AUTOPILOT_MAX_CYCLES": "1",
        "STORAGE_BACKPRESSURE_AUTOPILOT_POLL_SECONDS": "8",
        "STORAGE_BACKPRESSURE_AUTOPILOT_TIMEOUT_SECONDS": "750",
        "STORAGE_BACKPRESSURE_AUTOPILOT_WAIT_TIMEOUT_SECONDS": "180",
        "STORAGE_PRESSURE_CLEARANCE_MAX_CYCLES": "1",
        "STORAGE_PRESSURE_CLEARANCE_POLL_SECONDS": "5",
        "STORAGE_PRESSURE_CLEARANCE_TIMEOUT_SECONDS": "600",
        "STORAGE_PRESSURE_CLEARANCE_WAIT_TIMEOUT_SECONDS": "120",
    },
    "soak_recovery": {
        "ADAPTIVE_OPS_RECOVERY_POLICY_REASON": "soak_blocked_by_storage_or_control_plane_freshness",
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.25",
        "BOT_LOGS_SPACE_RECOVERY_MAX_DELETE_GB": "16",
        "BOT_LOGS_SPACE_RECOVERY_MIN_AGE_HOURS": "8",
        "BOT_LOGS_SPACE_RECOVERY_TARGET_FREE_GB": "130",
        "MAINTENANCE_SLOT_STORAGE_BACKPRESSURE_AUTOPILOT_MIN_INTERVAL_SECONDS": "90",
        "MAINTENANCE_SLOT_STORAGE_PRESSURE_CLEARANCE_MIN_INTERVAL_SECONDS": "75",
        "MAINTENANCE_SLOT_STORAGE_RECOVERY_DEFER_OUTSIDE_QUIET_WINDOW": "0",
        "MAINTENANCE_SLOT_STORAGE_RECOVERY_DEFER_WHILE_SQL_LINK_ACTIVE": "1",
        "MAINTENANCE_SLOT_STORAGE_RECOVERY_FAST_MODE": "1",
        "SOAK_SELF_HEAL_STORAGE_CLEANUP_MAX_DELETE_GB": "12",
        "STORAGE_BACKPRESSURE_AUTOPILOT_BACKPRESSURE_TIMEOUT_SECONDS": "120",
        "STORAGE_BACKPRESSURE_AUTOPILOT_MAX_CYCLES": "2",
        "STORAGE_BACKPRESSURE_AUTOPILOT_POLL_SECONDS": "6",
        "STORAGE_BACKPRESSURE_AUTOPILOT_TIMEOUT_SECONDS": "180",
        "STORAGE_BACKPRESSURE_AUTOPILOT_WAIT_TIMEOUT_SECONDS": "75",
        "STORAGE_BACKPRESSURE_AUTOPILOT_TARGET_PENDING_LINES": "5000",
        "STORAGE_PRESSURE_CLEARANCE_MAX_CYCLES": "1",
        "STORAGE_PRESSURE_CLEARANCE_POLL_SECONDS": "5",
        "STORAGE_PRESSURE_CLEARANCE_TIMEOUT_SECONDS": "600",
        "STORAGE_PRESSURE_CLEARANCE_WAIT_TIMEOUT_SECONDS": "105",
    },
    "critical_recovery": {
        "ADAPTIVE_OPS_RECOVERY_POLICY_REASON": "critical_storage_memory_or_hot_path_pressure",
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.10",
        "BOT_LOGS_SPACE_RECOVERY_MAX_DELETE_GB": "20",
        "BOT_LOGS_SPACE_RECOVERY_MIN_AGE_HOURS": "6",
        "BOT_LOGS_SPACE_RECOVERY_TARGET_FREE_GB": "140",
        "MAINTENANCE_SLOT_STORAGE_BACKPRESSURE_AUTOPILOT_MIN_INTERVAL_SECONDS": "60",
        "MAINTENANCE_SLOT_STORAGE_PRESSURE_CLEARANCE_MIN_INTERVAL_SECONDS": "60",
        "MAINTENANCE_SLOT_STORAGE_RECOVERY_DEFER_OUTSIDE_QUIET_WINDOW": "0",
        "MAINTENANCE_SLOT_STORAGE_RECOVERY_DEFER_WHILE_SQL_LINK_ACTIVE": "1",
        "MAINTENANCE_SLOT_STORAGE_RECOVERY_FAST_MODE": "1",
        "SOAK_SELF_HEAL_STORAGE_CLEANUP_MAX_DELETE_GB": "16",
        "STORAGE_BACKPRESSURE_AUTOPILOT_BACKPRESSURE_TIMEOUT_SECONDS": "120",
        "STORAGE_BACKPRESSURE_AUTOPILOT_MAX_CYCLES": "1",
        "STORAGE_BACKPRESSURE_AUTOPILOT_POLL_SECONDS": "5",
        "STORAGE_BACKPRESSURE_AUTOPILOT_TIMEOUT_SECONDS": "600",
        "STORAGE_BACKPRESSURE_AUTOPILOT_WAIT_TIMEOUT_SECONDS": "120",
        "STORAGE_PRESSURE_CLEARANCE_MAX_CYCLES": "1",
        "STORAGE_PRESSURE_CLEARANCE_POLL_SECONDS": "5",
        "STORAGE_PRESSURE_CLEARANCE_TIMEOUT_SECONDS": "600",
        "STORAGE_PRESSURE_CLEARANCE_WAIT_TIMEOUT_SECONDS": "90",
    },
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _status(value: Any) -> str:
    return str(value or "unknown").strip().lower()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "blocked"}


def _parse_iso_utc(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _artifact_age_minutes(payload: dict[str, Any], path: Path) -> float | None:
    timestamp = (
        payload.get("timestamp_utc")
        or payload.get("generated_at")
        or payload.get("generated_utc")
        or payload.get("completed_utc")
    )
    parsed = _parse_iso_utc(timestamp)
    if parsed is None or parsed.tzinfo is None:
        return None
    age = (
        datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)
    ).total_seconds() / 60.0
    return age if age >= 0.0 else None


def _load_health(project_root: Path, filename: str) -> dict[str, Any]:
    path = project_root / HEALTH_DIR / filename
    return _as_dict(load_json(path))


def _health_artifact_metric(
    project_root: Path,
    name: str,
    rel_path: Path,
    *,
    max_age_minutes: float,
) -> dict[str, Any]:
    path = project_root / rel_path
    payload = _as_dict(load_json(path))
    age = _artifact_age_minutes(payload, path) if payload else None
    refresh_lead_minutes = min(
        CONTROL_REFRESH_AGE_BUFFER_MINUTES, max(float(max_age_minutes) * 0.25, 0.0)
    )
    stale = bool(age is None or float(age) > float(max_age_minutes))
    refresh_due = bool(
        stale
        or (
            age is not None
            and float(age) >= max(float(max_age_minutes) - refresh_lead_minutes, 0.0)
        )
    )
    return {
        "name": name,
        "path": str(path.resolve()),
        "exists": path.exists(),
        "age_minutes": round(float(age), 3) if age is not None else None,
        "max_age_minutes": float(max_age_minutes),
        "refresh_lead_minutes": round(float(refresh_lead_minutes), 3),
        "stale": stale,
        "refresh_due": refresh_due,
        "status": _status(payload.get("overall_status") or payload.get("status")),
        "ok": payload.get("ok"),
    }


def _risk_evidence_metric(project_root: Path) -> dict[str, Any]:
    from scripts.sleeve_slo_guard import read_local_json

    path = project_root / "governance/risk/risk_service_boundary_latest.json"
    try:
        payload = read_local_json(project_root, path)
    except (ValueError, OSError):
        payload = {}
    freshness = evidence_freshness(payload, max_age_minutes=3)
    age = freshness.get("age_minutes")
    deadline = parse_iso_utc(payload.get("valid_until_utc"))
    input_expiry_due = "valid_until_utc" in payload and (
        deadline is None or deadline <= datetime.now(timezone.utc) + timedelta(minutes=1)
    )
    return {
        **freshness,
        "refresh_due": not freshness["fresh"] or (age is not None and age >= 2.25) or input_expiry_due,
        "input_expiry_due": input_expiry_due,
        "path": str(path),
    }


def collect_metrics(project_root: Path) -> dict[str, Any]:
    from scripts.ops.schd_evidence_maintenance import maintenance_metric

    dashboard = _load_health(project_root, "runtime_gate_dashboard_latest.json")
    ingestion = _load_health(project_root, "ingestion_storage_control_latest.json")
    local_reserve = _load_health(
        project_root, "local_storage_reserve_guard_latest.json"
    )
    memory = _load_health(project_root, "memory_efficiency_control_latest.json")
    cleanup = _load_health(project_root, "bot_logs_cleanup_intelligence_latest.json")
    soak = _load_health(project_root, "soak_reliability_sentinel_latest.json")
    unattended = _load_health(project_root, "unattended_soak_readiness_latest.json")
    halt = _load_health(project_root, "halt_trigger_control_plane_latest.json")

    overall = _as_dict(dashboard.get("overall"))
    storage = _as_dict(dashboard.get("storage"))
    containment = _as_dict(overall.get("degradation_containment"))
    soak_context = _as_dict(overall.get("soak_management_context"))
    attention_tiers = _as_dict(overall.get("attention_tiers"))

    local_hard_blockers = _as_list(local_reserve.get("hard_blockers"))
    soak_blockers = _as_list(soak.get("blockers"))
    unattended_blockers = _as_list(unattended.get("blockers"))
    dashboard_attention = [str(item) for item in _as_list(overall.get("attention"))]
    degraded_attention = [
        str(item) for item in _as_list(attention_tiers.get("degraded"))
    ]
    critical_attention = [
        str(item) for item in _as_list(attention_tiers.get("critical"))
    ]
    pressure_index = max(
        _safe_float(storage.get("pressure_index")),
        _safe_float(ingestion.get("pressure_index")),
        _safe_float(cleanup.get("pressure_index")),
    )
    total_pending_lines = max(
        _safe_int(storage.get("total_pending_lines")),
        _safe_int(ingestion.get("total_pending_lines")),
        _safe_int(_as_dict(ingestion.get("backpressure")).get("total_pending_lines")),
        _safe_int(cleanup.get("total_pending_lines")),
    )

    storage_status = _status(
        storage.get("status")
        or ingestion.get("overall_status")
        or ingestion.get("status")
    )
    ingestion_status = _status(
        ingestion.get("overall_status") or ingestion.get("status")
    )
    local_status = _status(
        local_reserve.get("status") or local_reserve.get("overall_status")
    )
    memory_status = _status(
        memory.get("overall_status")
        or memory.get("status")
        or _as_dict(dashboard.get("memory")).get("status")
    )
    soak_status = _status(
        soak.get("overall_status")
        or soak.get("status")
        or soak_context.get("soak_status")
    )
    unattended_status = _status(
        unattended.get("overall_status") or unattended.get("status")
    )
    dashboard_status = _status(dashboard.get("overall_status") or overall.get("status"))

    storage_blocked = storage_status in {
        "blocked",
        "critical",
        "halt",
        "red",
    } or ingestion_status in {
        "blocked",
        "critical",
        "halt",
        "red",
    }
    local_blocked = local_status in {"blocked", "critical", "halt", "red"} or bool(
        local_hard_blockers
    )
    memory_blocked = memory_status in {"blocked", "critical", "halt", "red"}
    hot_path_blocked = _truthy(
        containment.get("hot_path_blocked") or dashboard.get("hot_path_blocked")
    )
    soak_blocked = (
        soak_status in {"blocked", "critical", "halt", "red"}
        or unattended_status in {"blocked", "critical", "halt", "red"}
        or bool(soak_blockers or unattended_blockers)
    )

    storage_or_freshness_soak_blocker = any(
        any(
            token in str(blocker).lower()
            for token in (
                "storage",
                "reserve",
                "ingestion",
                "refresh_due",
                "role_contract",
                "capability",
                "collector",
                "halt_trigger",
                "health_gates",
                "coordination",
            )
        )
        for blocker in [*soak_blockers, *unattended_blockers]
    )
    freshness_artifacts = {
        "risk_service_boundary": _risk_evidence_metric(project_root),
        "schd_evidence_maintenance": maintenance_metric(project_root),
        "ingestion_storage_control": _health_artifact_metric(
            project_root,
            "ingestion_storage_control",
            HEALTH_DIR / "ingestion_storage_control_latest.json",
            max_age_minutes=15.0,
        ),
        "system_role_contract": _health_artifact_metric(
            project_root,
            "system_role_contract",
            HEALTH_DIR / "system_role_contract_latest.json",
            max_age_minutes=30.0,
        ),
        "health_gates": _health_artifact_metric(
            project_root,
            "health_gates",
            HEALTH_DIR / "health_gates_latest.json",
            max_age_minutes=15.0,
        ),
        "halt_trigger_control_plane": _health_artifact_metric(
            project_root,
            "halt_trigger_control_plane",
            HEALTH_DIR / "halt_trigger_control_plane_latest.json",
            max_age_minutes=15.0,
        ),
        "coordination_state": _health_artifact_metric(
            project_root,
            "coordination_state",
            HEALTH_DIR / "coordination_state_latest.json",
            max_age_minutes=4.0,
        ),
        "capability_materialization": _health_artifact_metric(
            project_root,
            "capability_materialization",
            Path(
                "governance/collector_capabilities/materialized_capabilities_latest.json"
            ),
            max_age_minutes=30.0,
        ),
        "collector_capability_control": _health_artifact_metric(
            project_root,
            "collector_capability_control",
            HEALTH_DIR / "collector_capability_control_latest.json",
            max_age_minutes=30.0,
        ),
        "unattended_soak_readiness": _health_artifact_metric(
            project_root,
            "unattended_soak_readiness",
            HEALTH_DIR / "unattended_soak_readiness_latest.json",
            max_age_minutes=15.0,
        ),
        "runtime_gate_dashboard": _health_artifact_metric(
            project_root,
            "runtime_gate_dashboard",
            HEALTH_DIR / "runtime_gate_dashboard_latest.json",
            max_age_minutes=15.0,
        ),
    }

    return {
        "dashboard_status": dashboard_status,
        "storage_status": storage_status,
        "ingestion_status": ingestion_status,
        "local_storage_status": local_status,
        "memory_status": memory_status,
        "soak_status": soak_status,
        "unattended_soak_status": unattended_status,
        "pressure_index": pressure_index,
        "total_pending_lines": total_pending_lines,
        "core_pending_lines": _safe_int(storage.get("core_pending_lines")),
        "hot_path_blocked": hot_path_blocked,
        "storage_blocked": storage_blocked,
        "local_storage_blocked": local_blocked,
        "memory_blocked": memory_blocked,
        "soak_blocked": soak_blocked,
        "soak_storage_or_freshness_blocked": storage_or_freshness_soak_blocker,
        "local_storage_hard_blockers": [str(item) for item in local_hard_blockers],
        "soak_blockers": [str(item) for item in soak_blockers],
        "unattended_soak_blockers": [str(item) for item in unattended_blockers],
        "halt_clear_blockers": _as_list(
            _as_dict(halt.get("blockers")).get("halt_clear")
        ),
        "dashboard_attention": dashboard_attention,
        "dashboard_degraded_attention": degraded_attention,
        "dashboard_critical_attention": critical_attention,
        "control_plane_freshness": freshness_artifacts,
        "bot_logs_remaining_to_target_gb": _safe_float(
            cleanup.get("remaining_to_target_gb")
        ),
        "source_files": {
            "runtime_gate_dashboard": str(
                (
                    project_root / HEALTH_DIR / "runtime_gate_dashboard_latest.json"
                ).resolve()
            ),
            "ingestion_storage_control": str(
                (
                    project_root / HEALTH_DIR / "ingestion_storage_control_latest.json"
                ).resolve()
            ),
            "local_storage_reserve_guard": str(
                (
                    project_root
                    / HEALTH_DIR
                    / "local_storage_reserve_guard_latest.json"
                ).resolve()
            ),
            "memory_efficiency_control": str(
                (
                    project_root / HEALTH_DIR / "memory_efficiency_control_latest.json"
                ).resolve()
            ),
            "bot_logs_cleanup_intelligence": str(
                (
                    project_root
                    / HEALTH_DIR
                    / "bot_logs_cleanup_intelligence_latest.json"
                ).resolve()
            ),
            "soak_reliability_sentinel": str(
                (
                    project_root / HEALTH_DIR / "soak_reliability_sentinel_latest.json"
                ).resolve()
            ),
            "unattended_soak_readiness": str(
                (
                    project_root / HEALTH_DIR / "unattended_soak_readiness_latest.json"
                ).resolve()
            ),
            "system_role_contract": str(
                (
                    project_root / HEALTH_DIR / "system_role_contract_latest.json"
                ).resolve()
            ),
            "health_gates": str(
                (project_root / HEALTH_DIR / "health_gates_latest.json").resolve()
            ),
            "halt_trigger_control_plane": str(
                (
                    project_root / HEALTH_DIR / "halt_trigger_control_plane_latest.json"
                ).resolve()
            ),
            "coordination_state": str(
                (project_root / HEALTH_DIR / "coordination_state_latest.json").resolve()
            ),
            "capability_materialization": str(
                (
                    project_root
                    / "governance"
                    / "collector_capabilities"
                    / "materialized_capabilities_latest.json"
                ).resolve()
            ),
            "collector_capability_control": str(
                (
                    project_root
                    / HEALTH_DIR
                    / "collector_capability_control_latest.json"
                ).resolve()
            ),
        },
    }


def choose_tier(metrics: dict[str, Any]) -> str:
    pressure = _safe_float(metrics.get("pressure_index"))
    pending = _safe_int(metrics.get("total_pending_lines"))

    critical = (
        _truthy(metrics.get("memory_blocked"))
        or (
            _truthy(metrics.get("hot_path_blocked"))
            and _truthy(metrics.get("storage_blocked"))
        )
        or pressure >= 5.0
        or pending >= 25_000
        or (_truthy(metrics.get("local_storage_blocked")) and pressure >= 2.0)
    )
    if critical:
        return "critical_recovery"

    if _truthy(metrics.get("soak_blocked")) and _truthy(
        metrics.get("soak_storage_or_freshness_blocked")
    ):
        return "soak_recovery"

    hot = (
        _truthy(metrics.get("storage_blocked"))
        or _truthy(metrics.get("hot_path_blocked"))
        or _truthy(metrics.get("local_storage_blocked"))
        or pressure >= 1.0
        or pending >= 15_000
    )
    if hot:
        return "hot_recovery"

    watch = (
        pressure >= 0.25
        or pending >= 5_000
        or _status(metrics.get("storage_status"))
        in {
            "needs_work",
            "degraded",
            "watch",
        }
    )
    if watch:
        return "watch"

    return "observe"


def _metric_refresh_due(metrics: dict[str, Any], name: str) -> bool:
    freshness = _as_dict(metrics.get("control_plane_freshness"))
    row = _as_dict(freshness.get(name))
    return bool(row.get("refresh_due", False))


def _status_blocked(value: Any) -> bool:
    return _status(value) in {"blocked", "critical", "degraded", "halt", "red"}


def _control_signal_text(metrics: dict[str, Any]) -> str:
    signals: list[str] = []
    for key in (
        "soak_blockers",
        "unattended_soak_blockers",
        "dashboard_degraded_attention",
        "dashboard_critical_attention",
        "halt_clear_blockers",
    ):
        signals.extend(str(item) for item in _as_list(metrics.get(key)))
    return " ".join(signals).lower()


def build_control_plane_refresh_plan(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    signal_text = _control_signal_text(metrics)
    plan: list[dict[str, Any]] = []
    added: set[str] = set()

    def add(step_id: str) -> None:
        if step_id in added:
            return
        step = CONTROL_PLANE_REFRESH_STEPS[step_id]
        plan.append(
            {
                "id": step_id,
                "command": list(step["command"]),
                "timeout_seconds": int(step["timeout_seconds"]),
                "reason": str(step["reason"]),
            }
        )
        added.add(step_id)

    role_needed = bool(
        "system_role_contract" in signal_text
        or "role_contract" in signal_text
        or _metric_refresh_due(metrics, "system_role_contract")
    )
    capability_needed = bool(
        "capability_materialization" in signal_text
        or _metric_refresh_due(metrics, "capability_materialization")
    )
    collector_needed = bool(
        "collector_capability" in signal_text
        or _metric_refresh_due(metrics, "collector_capability_control")
    )
    health_needed = bool(
        "health_gates" in signal_text
        or "health_gate" in signal_text
        or _metric_refresh_due(metrics, "health_gates")
    )
    storage_needed = bool(
        "ingestion" in signal_text
        or "storage" in signal_text
        or _truthy(metrics.get("storage_blocked"))
        or _status_blocked(metrics.get("ingestion_status"))
        or _metric_refresh_due(metrics, "ingestion_storage_control")
        or _safe_float(metrics.get("pressure_index")) >= 0.25
        or _safe_int(metrics.get("total_pending_lines")) >= 5000
    )
    halt_needed = bool(
        "halt_trigger" in signal_text
        or _metric_refresh_due(metrics, "halt_trigger_control_plane")
        or health_needed
        or storage_needed
        or _as_list(metrics.get("halt_clear_blockers"))
    )
    coordination_needed = bool(
        "coordination" in signal_text
        or _metric_refresh_due(metrics, "coordination_state")
    )
    paper_guard_needed = "runtime_paper_regression_guard" in signal_text
    soak_needed = bool(
        "soak" in signal_text
        or health_needed
        or halt_needed
        or coordination_needed
        or paper_guard_needed
        or storage_needed
        or _truthy(metrics.get("soak_blocked"))
    )
    unattended_needed = bool(
        role_needed
        or capability_needed
        or collector_needed
        or storage_needed
        or soak_needed
        or _status_blocked(metrics.get("unattended_soak_status"))
        or _metric_refresh_due(metrics, "unattended_soak_readiness")
    )
    dashboard_needed = bool(
        unattended_needed
        or _status_blocked(metrics.get("dashboard_status"))
        or _metric_refresh_due(metrics, "runtime_gate_dashboard")
    )
    stale_advisories = {
        str(signal)
        for key in ("dashboard_degraded_attention", "dashboard_critical_attention")
        for signal in _as_list(metrics.get(key))
    }
    advisory_steps = [
        name
        for name in (
            "bot_profitability_scalability_control",
            "sleeve_scalability_selector",
            "master_grandmaster_evidence_v2",
        )
        if f"{name}_stale" in stale_advisories
    ]

    if _metric_refresh_due(metrics, "risk_service_boundary"):
        add("risk_service_boundary")
    if _metric_refresh_due(metrics, "schd_evidence_maintenance"):
        add("schd_evidence_maintenance")
    if role_needed:
        add("system_role_contract")
    if capability_needed:
        add("capability_materialization")
    if collector_needed:
        add("collector_capability_control")
    if health_needed:
        add("health_gates")
    if storage_needed:
        add("ingestion_storage_control")
    if halt_needed:
        add("global_halt_refresh")
        add("halt_trigger_status")
    if storage_needed:
        add("system_plumbing_control")
        add("system_architecture_hardening")
    if coordination_needed:
        add("coordination_state")
    if paper_guard_needed:
        add("runtime_paper_regression_guard")
    if soak_needed:
        add("soak_reliability_sentinel")
    if unattended_needed:
        add("unattended_soak_readiness")
    for name in advisory_steps:
        add(name)
    if dashboard_needed or advisory_steps:
        add("runtime_gate_dashboard")

    return plan


def _parse_json_output(stdout: str) -> dict[str, Any]:
    text = (stdout or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return _as_dict(parsed)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return _as_dict(json.loads(text[start : end + 1]))
        except json.JSONDecodeError:
            return {}
    return {}


def _refresh_payload_ok(payload: dict[str, Any], rc: int, *, step_id: str = "") -> bool:
    if rc != 0 or not payload:
        return False
    status = _status(
        payload.get("overall_status")
        or payload.get("status")
        or _as_dict(payload.get("overall")).get("status")
    )
    if step_id == "halt_trigger_status":
        blockers = _as_dict(payload.get("blockers"))
        return blockers.get("halt_clear") == []
    if step_id == "health_gates":
        return payload.get("hard_gate_triggered") is False
    if step_id == "global_halt_refresh":
        return payload.get("halt") is False and payload.get("clear_blockers") == []
    if step_id == "ingestion_storage_control":
        soak = _as_dict(payload.get("continuous_run_soak_contract"))
        if _status_blocked(soak.get("status")) or soak.get("blockers"):
            return False
    if payload.get("ok") is False:
        return False
    if payload.get("ok") is True:
        return True
    return status in {"ready", "ok", "healthy", "guarded", "watch", "advisory"}


def run_control_plane_refresh_plan(
    project_root: Path,
    plan: list[dict[str, Any]],
    *,
    env_overrides: dict[str, str],
    max_steps: int,
    budget_seconds: int,
) -> list[dict[str, Any]]:
    opsctl = project_root / "scripts" / "ops" / "opsctl.sh"
    results: list[dict[str, Any]] = []
    start = time.monotonic()
    env = dict(os.environ)
    env.update(env_overrides)
    env.update(SAFETY_ENV)
    env["PYTHONUNBUFFERED"] = "1"
    env["OPSCTL_SELF_MODEL_AUTO_REFRESH"] = "0"

    for index, step in enumerate(plan):
        elapsed = time.monotonic() - start
        remaining = max(float(budget_seconds) - elapsed, 0.0)
        skip_reason = ""
        if index >= max(int(max_steps), 0):
            skip_reason = "max_refresh_steps_reached"
        elif remaining < 3.0:
            skip_reason = "refresh_budget_exhausted"
        if skip_reason:
            results.append(
                {
                    "id": step.get("id"),
                    "command": step.get("command"),
                    "ok": False,
                    "skipped": True,
                    "skip_reason": skip_reason,
                }
            )
            continue

        command = [str(item) for item in _as_list(step.get("command"))]
        if command[:1] == ["./scripts/ops/opsctl.sh"] and not opsctl.exists():
            results.append(
                {
                    "id": step.get("id"),
                    "command": command,
                    "ok": False,
                    "skipped": True,
                    "skip_reason": "opsctl_missing",
                }
            )
            continue

        timeout = max(1, min(int(step.get("timeout_seconds") or 30), int(remaining)))
        step_start = time.monotonic()
        try:
            result = run_bounded_process_group(
                command,
                cwd=project_root,
                env=env,
                timeout_seconds=timeout,
            )
        except OSError as exc:
            result = {"rc": 127, "stdout": "", "stderr": str(exc), "timed_out": False}
        stdout = str(result.get("stdout") or "")
        stderr = str(result.get("stderr") or "")
        rc = int(result["rc"])
        timed_out = bool(result.get("timed_out"))

        parsed = _parse_json_output(stdout)
        results.append(
            {
                "id": step.get("id"),
                "command": command,
                "rc": rc,
                "timed_out": timed_out,
                "timeout_cleanup": result.get("timeout_cleanup"),
                "ok": bool(
                    _refresh_payload_ok(parsed, rc, step_id=str(step.get("id") or ""))
                )
                and not timed_out,
                "duration_seconds": round(time.monotonic() - step_start, 3),
                "parsed_status": _status(
                    parsed.get("overall_status")
                    or parsed.get("status")
                    or _as_dict(parsed.get("overall")).get("status")
                ),
                "parsed_ok": parsed.get("ok"),
                "stdout_tail": "\n".join(stdout.splitlines()[-6:])[-1200:],
                "stderr_tail": "\n".join(stderr.splitlines()[-6:])[-1200:],
            }
        )

    return results


def build_recommended_actions(metrics: dict[str, Any], tier: str) -> list[str]:
    actions: list[str] = []
    if tier in {"hot_recovery", "soak_recovery", "critical_recovery"}:
        actions.append(
            "Run storage pressure clearance and backpressure autopilot on the tightened adaptive cadence."
        )
    if _truthy(metrics.get("memory_blocked")):
        actions.append(
            "Keep memory-efficiency controls in collection-shedding mode until the next ready dashboard."
        )
    if _truthy(metrics.get("local_storage_blocked")):
        actions.append(
            "Prioritize local hot-storage reserve recovery before widening collection duty cycle."
        )
    if _truthy(metrics.get("soak_blocked")):
        actions.append(
            "Keep soak blocked until storage, ingestion, and freshness sentinels publish ready snapshots."
        )
    if not actions:
        actions.append(
            "Keep the adaptive policy in observe mode and preserve normal launchd cadence."
        )
    return actions


def env_for_tier(tier: str, *, project_root: Path | None = None) -> dict[str, str]:
    env = dict(SAFETY_ENV)
    env["STORAGE_BACKPRESSURE_AUTOPILOT_TARGET_PENDING_LINES"] = "20000"
    env.update(TIER_PROFILES[tier])
    env["ADAPTIVE_OPS_RECOVERY_TIER"] = tier
    if project_root is not None:
        ratio_key = "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"
        ceilings = [float(env[ratio_key])]
        pause_keys = (
            "BOT_COLLECTION_DUTY_CYCLE_ENABLED",
            "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG",
            "HEAVY_COLLECTORS_PAUSED_FOR_BACKLOG",
            "REPORT_REFRESH_PAUSED_FOR_BACKLOG",
        )
        owner_flags: dict[str, list[str]] = {}
        for name in (
            "storage_pressure",
            "runtime_resource_guard",
            "computer_task",
            "operator_mode",
        ):
            path = project_root / "config" / f".env.{name}_override"
            values = dotenv_values(path, interpolate=False) if path.is_file() else {}
            try:
                ceiling = float(values.get(ratio_key) or "")
                if math.isfinite(ceiling) and 0.0 <= ceiling <= 1.0:
                    ceilings.append(ceiling)
            except ValueError:
                pass
            for key in pause_keys:
                if values.get(key) in {"0", "1"}:
                    owner_flags.setdefault(key, []).append(str(values[key]))
        env[ratio_key] = str(min(ceilings))
        for key, values in owner_flags.items():
            env[key] = "1" if "1" in values else "0"
    return env


def _env_line(key: str, value: str) -> str:
    return f"export {key}={shlex.quote(str(value))}"


def render_override(env: dict[str, str], *, tier: str, generated_at: str) -> str:
    lines = [
        "# Auto-managed by scripts/ops/adaptive_ops_recovery_policy.py",
        "# Loaded last by scripts/ops/load_runtime_env.sh so platform pressure can tighten or relax ops controls.",
        f"# generated_at={generated_at}",
        f"# tier={tier}",
    ]
    for key in sorted(env):
        lines.append(_env_line(key, env[key]))
    return "\n".join(lines) + "\n"


def _write_if_changed(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        old = None
    if old == text:
        return False
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return True


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    out_file: Path | None = None,
    override_file: Path | None = None,
    now: str | None = None,
    refresh_control_plane: bool = True,
    max_refresh_steps: int = len(CONTROL_PLANE_REFRESH_STEPS),
    refresh_budget_seconds: int = 210,
) -> dict[str, Any]:
    project_root = Path(project_root)
    started_utc = datetime.now(timezone.utc)
    out_file = (
        Path(out_file) if out_file is not None else project_root / DEFAULT_OUT_REL
    )
    override_file = (
        Path(override_file)
        if override_file is not None
        else project_root / DEFAULT_OVERRIDE_REL
    )

    pre_refresh_metrics = collect_metrics(project_root)
    refresh_plan = build_control_plane_refresh_plan(pre_refresh_metrics)
    refresh_results: list[dict[str, Any]] = []
    if apply and refresh_control_plane and refresh_plan:
        initial_tier = choose_tier(pre_refresh_metrics)
        _write_if_changed(
            override_file,
            render_override(
                env_for_tier(initial_tier, project_root=project_root),
                tier=initial_tier,
                generated_at=now or iso_now(),
            ),
        )
        refresh_results = run_control_plane_refresh_plan(
            project_root,
            refresh_plan,
            env_overrides=env_for_tier(initial_tier, project_root=project_root),
            max_steps=max_refresh_steps,
            budget_seconds=refresh_budget_seconds,
        )
        metrics = collect_metrics(project_root)
    else:
        metrics = pre_refresh_metrics
    tier = choose_tier(metrics)
    generated_at = now or iso_now()
    env = env_for_tier(tier, project_root=project_root)
    override_text = render_override(env, tier=tier, generated_at=generated_at)
    override_changed = False
    if apply:
        override_changed = _write_if_changed(override_file, override_text)
    refresh_failures = [
        row for row in refresh_results if not bool(_as_dict(row).get("ok", False))
    ]
    policy_ok = not refresh_failures

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "timestamp_utc": generated_at,
        "ok": policy_ok,
        "platform_ok": bool(
            policy_ok
            and tier == "observe"
            and metrics.get("dashboard_status") in {"ready", "ok", "healthy"}
            and metrics.get("unattended_soak_status") in {"ready", "ok", "healthy"}
            and not _truthy(metrics.get("soak_blocked"))
            and not any(
                row["stale"] for row in metrics["control_plane_freshness"].values()
            )
        ),
        "overall_status": "refresh_findings" if refresh_failures else tier,
        "tier": tier,
        "pre_refresh_tier": choose_tier(pre_refresh_metrics),
        "control_plane_refresh_status": (
            "not_run"
            if not apply or not refresh_control_plane
            else "attention" if refresh_failures else "ready"
        ),
        "control_plane_refresh_failure_count": len(refresh_failures),
        "control_plane_refresh_plan": refresh_plan,
        "control_plane_refresh_results": refresh_results,
        "control_plane_refresh_enabled": bool(refresh_control_plane),
        "metrics": metrics,
        "pre_refresh_metrics": pre_refresh_metrics,
        "env_overrides": env,
        "override_file": str(override_file),
        "override_changed": override_changed,
        "recommended_actions": build_recommended_actions(metrics, tier),
    }
    if apply:
        ts = datetime.now(timezone.utc)
        payload["job_lifecycle"] = lifecycle_receipt(
            job_id="adaptive_ops_recovery_policy",
            scheduled=False,
            started_utc=started_utc,
            completed_utc=ts,
            schedule_interval_seconds=60,
            rc=0 if policy_ok else 2,
            terminal_status="completed" if policy_ok else "completed_with_findings",
            ok=policy_ok,
            artifact_present_before=out_file.exists(),
            artifact_present_after=True,
            deadline_seconds=240,
            command=[
                "scripts/ops/adaptive_ops_recovery_policy.py",
                "--apply",
                "--json",
            ],
            source="adaptive_ops_recovery_policy_direct",
        )
        write_payload(out_file, payload)
    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish adaptive ops recovery policy overrides."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-file", type=Path)
    parser.add_argument("--override-file", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the override env file and latest health artifact.",
    )
    parser.add_argument(
        "--no-control-plane-refresh",
        action="store_true",
        help="Do not run the bounded health/control-plane refresh cascade during --apply.",
    )
    parser.add_argument(
        "--max-control-refresh-steps",
        type=int,
        default=int(
            os.getenv(
                "ADAPTIVE_OPS_MAX_CONTROL_REFRESH_STEPS",
                str(len(CONTROL_PLANE_REFRESH_STEPS)),
            )
        ),
        help="Maximum allowlisted control-plane refresh steps to run during --apply.",
    )
    parser.add_argument(
        "--control-refresh-budget-seconds",
        type=int,
        default=int(os.getenv("ADAPTIVE_OPS_CONTROL_REFRESH_BUDGET_SECONDS", "210")),
        help="Total bounded runtime budget for control-plane refresh during --apply.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the computed policy payload as JSON."
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    lock_handle = None
    if args.apply:
        lock_path = (
            args.project_root / "governance/locks/adaptive_ops_recovery_policy.lock"
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_handle.close()
            print(
                "adaptive_ops_recovery_policy status=deferred reason=policy_writer_busy",
                file=sys.stderr,
            )
            if args.json:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "overall_status": "deferred",
                            "reason": "policy_writer_busy",
                        }
                    )
                )
            return 0
    try:
        payload = build_payload(
            args.project_root,
            apply=args.apply,
            out_file=args.out_file,
            override_file=args.override_file,
            refresh_control_plane=not args.no_control_plane_refresh,
            max_refresh_steps=args.max_control_refresh_steps,
            refresh_budget_seconds=args.control_refresh_budget_seconds,
        )
    finally:
        if lock_handle is not None:
            lock_handle.close()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"adaptive_ops_recovery_policy tier={payload['tier']} "
            f"pressure={payload['metrics']['pressure_index']:.3f} "
            f"pending={payload['metrics']['total_pending_lines']} "
            f"override_changed={str(payload['override_changed']).lower()}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
