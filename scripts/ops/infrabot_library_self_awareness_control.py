#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


SCHEMA_VERSION = 1
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "infrabot_library_self_awareness_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "infrabot_library_self_awareness_control_latest.json"
DEFAULT_EXTERNAL_CONTEXT_PATH = PROJECT_ROOT / "exports" / "external_context" / "infrabot_library_self_awareness_control_latest.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "exports" / "reports" / "operator" / "infrabot_library_self_awareness_control_latest.md"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.infrabot_library_self_awareness_override"

READY_STATUSES = {"ready", "ok", "active", "guarded", "advisory", "watch", "needs_action"}
BAD_STATUSES = {"blocked", "critical", "failed", "error", "missing"}
SINGLE_WRITER_HINTS = ("storage", "writer", "sql", "backpressure", "stateful", "quota")
AUTONOMIC_REFLEX_PHASES = [
    {
        "phase_id": "sense",
        "title": "Sense",
        "severity_floor": "watch",
        "allowed_action": "read_fresh_health_artifacts_and_collect_symptoms",
        "cooldown_seconds": 60,
        "escalation_target": "self_awareness_need_brief",
    },
    {
        "phase_id": "classify",
        "title": "Classify",
        "severity_floor": "watch",
        "allowed_action": "separate_hard_blocker_managed_advisory_and_cosmetic_noise",
        "cooldown_seconds": 90,
        "escalation_target": "runtime_gate_dashboard",
    },
    {
        "phase_id": "refresh",
        "title": "Refresh",
        "severity_floor": "advisory",
        "allowed_action": "run_lightweight_owner_refresh_without_mutating_dependencies",
        "cooldown_seconds": 180,
        "escalation_target": "owning_health_artifact",
    },
    {
        "phase_id": "repair",
        "title": "Repair",
        "severity_floor": "degraded",
        "allowed_action": "run_one_bounded_safe_apply_command_for_the_owner_lane",
        "cooldown_seconds": 300,
        "escalation_target": "infrabot_adaptive_governor",
    },
    {
        "phase_id": "verify",
        "title": "Verify",
        "severity_floor": "watch",
        "allowed_action": "recheck_stop_condition_and_publish_replayable_proof",
        "cooldown_seconds": 120,
        "escalation_target": "production_level_upgrade_hardener_control",
    },
    {
        "phase_id": "escalate_or_hold",
        "title": "Escalate Or Hold",
        "severity_floor": "critical",
        "allowed_action": "escalate_to_operator_or_hold_visible_if_soak_manager_proves_it_is_safe",
        "cooldown_seconds": 600,
        "escalation_target": "operator_notification_and_unattended_soak_readiness",
    },
]
LANE_SENSORY_ARTIFACTS = {
    "storage_writer": [
        "governance/health/runtime_gate_dashboard_latest.json",
        "governance/health/ingestion_storage_control_latest.json",
        "governance/health/storage_quota_guard_latest.json",
        "governance/health/writer_cycle_coordinator_latest.json",
    ],
    "raw_profitability_recovery": [
        "governance/health/paper_profitability_control_latest.json",
        "governance/health/paper_runtime_profitability_controls_latest.json",
        "governance/health/live_canary_readiness_contract_latest.json",
        "governance/health/system_needs_intelligence_latest.json",
    ],
    "source_truth": [
        "governance/health/provider_mesh_latest.json",
        "governance/health/source_verification_latest.json",
        "governance/health/collector_contracts_latest.json",
    ],
    "runtime_memory": [
        "governance/health/runtime_gate_dashboard_latest.json",
        "governance/health/runtime_throttle_control_latest.json",
        "governance/health/memory_efficiency_control_latest.json",
        "governance/health/mlx_intelligence_router_latest.json",
    ],
    "governance_regression": [
        "governance/health/adaptive_regression_guard_latest.json",
        "governance/health/runtime_paper_regression_guard_latest.json",
        "governance/health/grade_regression_guard_latest.json",
        "governance/health/production_level_upgrade_hardener_control_latest.json",
    ],
    "auth_live_lock": [
        "governance/health/schwab_auth_supervisor_latest.json",
        "governance/health/auth_lease_manager_latest.json",
        "governance/health/production_readiness_control_latest.json",
        "governance/health/global_killswitch_latest.json",
    ],
    "general_infrabot": [
        "governance/health/runtime_gate_dashboard_latest.json",
        "governance/health/system_needs_intelligence_latest.json",
        "governance/health/infrabot_adaptive_governor_latest.json",
    ],
}
LANE_HEALING_CAPABILITIES = {
    "storage_writer": {
        "primary_capability": "writer_cycle_coordinator",
        "secondary_capabilities": ["storage_backpressure_autopilot", "stateful_storage_regression_guard"],
        "verify_command": ["./scripts/ops/opsctl.sh", "ingestion-storage", "--json"],
        "max_attempts": 1,
        "cooldown_seconds": 300,
        "allowed_under_pressure": True,
    },
    "raw_profitability_recovery": {
        "primary_capability": "paper_profitability_control",
        "secondary_capabilities": ["paper_performance_refresh", "master_grandmaster_profitability_trainer", "runtime_paper_regression_guard"],
        "verify_command": ["./scripts/ops/opsctl.sh", "runtime-paper-regression-guard", "--json"],
        "max_attempts": 2,
        "cooldown_seconds": 600,
        "allowed_under_pressure": False,
    },
    "source_truth": {
        "primary_capability": "source_verification_autorefresh",
        "secondary_capabilities": ["provider_mesh_refresh", "market_explanation_evidence"],
        "verify_command": ["./scripts/ops/opsctl.sh", "source-verification", "--json"],
        "max_attempts": 2,
        "cooldown_seconds": 300,
        "allowed_under_pressure": False,
    },
    "runtime_memory": {
        "primary_capability": "pressure_relief_control",
        "secondary_capabilities": ["runtime_throttle_control", "memory_pressure_intelligence"],
        "verify_command": ["./scripts/ops/opsctl.sh", "runtime-throttle", "--json"],
        "max_attempts": 2,
        "cooldown_seconds": 240,
        "allowed_under_pressure": True,
    },
    "governance_regression": {
        "primary_capability": "adaptive_regression_guard",
        "secondary_capabilities": ["runtime_paper_regression_guard", "grade_regression_guard", "production_quality_slo_guard"],
        "verify_command": ["./scripts/ops/opsctl.sh", "grade-regression-guard", "--json"],
        "max_attempts": 2,
        "cooldown_seconds": 420,
        "allowed_under_pressure": False,
    },
    "auth_live_lock": {
        "primary_capability": "broker_auth_supervisor",
        "secondary_capabilities": ["global_halt_refresh", "paper_ramp_guard", "production_quality_control"],
        "verify_command": ["./scripts/ops/opsctl.sh", "production-readiness", "--json"],
        "max_attempts": 1,
        "cooldown_seconds": 300,
        "allowed_under_pressure": True,
    },
}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _string_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "ready", "ok", "active", "guarded"}


def _path_value(payload: Any, dotted: str, default: Any = None) -> Any:
    current = payload
    for part in str(dotted or "").split("."):
        if not part:
            continue
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        return default
    return current


def _path_exists(payload: Any, dotted: str) -> bool:
    sentinel = object()
    return _path_value(payload, dotted, sentinel) is not sentinel


def _extract_status(payload: dict[str, Any], status_path: str = "") -> str:
    if status_path:
        status = str(_path_value(payload, status_path, "") or "").strip().lower()
        if status:
            return status
    for key in ("overall_status", "status", "state"):
        status = str(payload.get(key) or "").strip().lower()
        if status:
            return status
    if "ok" in payload:
        return "ready" if bool(payload.get("ok")) else "blocked"
    return "present" if payload else "missing"


def _project_path(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else project_root / path


def _artifact_row(project_root: Path, requirement: dict[str, Any]) -> dict[str, Any]:
    name = str(requirement.get("name") or "").strip()
    path = _project_path(project_root, requirement.get("path"))
    required = bool(requirement.get("required", False))
    exists = path.exists()
    payload = load_json(path) if exists else {}
    status = _extract_status(payload, str(requirement.get("status_path") or "")) if exists else "missing"
    ready_statuses = set(_string_list(requirement.get("ready_statuses")) or READY_STATUSES)
    blockers: list[str] = []
    if required and not exists:
        blockers.append("artifact_missing")
    elif exists and status not in ready_statuses:
        blockers.append(f"status_not_ready:{status}")

    selected: dict[str, Any] = {}
    for dotted in _string_list(requirement.get("truthy_paths")):
        value = _path_value(payload, dotted)
        selected[dotted] = value
        if not _bool(value):
            blockers.append(f"truthy_path_failed:{dotted}")
    for dotted in _string_list(requirement.get("falsey_paths")):
        value = _path_value(payload, dotted)
        selected[dotted] = value
        if _bool(value):
            blockers.append(f"falsey_path_failed:{dotted}")
    for dotted in _string_list(requirement.get("required_paths")):
        selected[dotted] = _path_value(payload, dotted)
        if not _path_exists(payload, dotted):
            blockers.append(f"required_path_missing:{dotted}")
    for dotted in _string_list(requirement.get("zero_count_paths")):
        value = _path_value(payload, dotted)
        selected[dotted] = value
        ok = len(value) == 0 if isinstance(value, list) else _safe_float(value, 999999.0) == 0.0
        if not ok:
            blockers.append(f"zero_count_failed:{dotted}")

    max_values = requirement.get("max_value_by_path") if isinstance(requirement.get("max_value_by_path"), dict) else {}
    for dotted, ceiling in max_values.items():
        value = _path_value(payload, str(dotted), 0)
        selected[str(dotted)] = value
        if _safe_float(value, 0.0) > _safe_float(ceiling):
            blockers.append(f"max_value_failed:{dotted}")
    min_values = requirement.get("min_value_by_path") if isinstance(requirement.get("min_value_by_path"), dict) else {}
    for dotted, floor in min_values.items():
        value = _path_value(payload, str(dotted))
        selected[str(dotted)] = value
        if _safe_float(value, -999999.0) < _safe_float(floor):
            blockers.append(f"min_value_failed:{dotted}")
    expected = requirement.get("expected_values") if isinstance(requirement.get("expected_values"), dict) else {}
    for dotted, expected_value in expected.items():
        value = _path_value(payload, str(dotted))
        selected[str(dotted)] = value
        if str(value) != str(expected_value):
            blockers.append(f"expected_value_failed:{dotted}")

    blockers = ordered_unique(blockers)
    return {
        "name": name,
        "path": str(path),
        "required": required,
        "exists": exists,
        "status": status,
        "ready": not blockers,
        "blocking": bool(required and blockers),
        "blockers": blockers,
        "selected_values": selected,
        "summary_keys": sorted(payload.keys())[:24] if payload else [],
    }


def _health(project_root: Path, filename: str) -> dict[str, Any]:
    return load_json(project_root / "governance" / "health" / filename)


def _command_key(command: list[Any]) -> tuple[str, ...]:
    return tuple(str(part) for part in command)


def _command_text(command: list[Any]) -> str:
    return " ".join(str(part) for part in command)


def _command_lane(command: list[Any], configured_lanes: list[dict[str, Any]]) -> str:
    text = _command_text(command).lower()
    for lane in configured_lanes:
        lane_name = str(lane.get("lane") or "")
        if lane_name and lane_name.replace("_", "-") in text:
            return lane_name
        for owner in _string_list(lane.get("owner_controls")):
            if owner.replace("_", "-") in text or owner in text:
                return lane_name
    if "source-verification" in text or "collector-contract" in text:
        return "source_truth"
    if "auth" in text or "production-readiness" in text:
        return "auth_live_lock"
    if "runtime" in text or "memory" in text or "mlx" in text or "library" in text:
        return "runtime_memory"
    if "regression" in text or "drift" in text or "grade" in text:
        return "governance_regression"
    if any(hint in text for hint in SINGLE_WRITER_HINTS):
        return "storage_writer"
    return "general_infrabot"


def _lane_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for raw in _as_list(config.get("infrabot_efficiency_lanes")):
        if not isinstance(raw, dict):
            continue
        lane = str(raw.get("lane") or "").strip()
        if lane:
            specs[lane] = raw
    return specs


def _collect_default_lane_commands(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane in _as_list(config.get("infrabot_efficiency_lanes")):
        if not isinstance(lane, dict):
            continue
        for command in _as_list(lane.get("default_commands")):
            if isinstance(command, list) and command:
                rows.append({"source": "config_default", "reason": "default_lane_command", "command": [str(part) for part in command]})
    return rows


def _collect_payload_commands(name: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in (
        "repair_plan",
        "advisory_repair_plan",
        "pre_apply_repair_plan",
        "pre_apply_advisory_repair_plan",
        "recommended_commands",
        "ordered_repair_commands",
    ):
        for raw in _as_list(payload.get(field)):
            command = raw.get("cmd") if isinstance(raw, dict) else raw
            if isinstance(command, list) and command:
                rows.append(
                    {
                        "source": name,
                        "reason": str(raw.get("reason") or field) if isinstance(raw, dict) else field,
                        "command": [str(part) for part in command],
                    }
                )
    for check in _as_list(payload.get("checks")):
        if not isinstance(check, dict):
            continue
        for command in _as_list(check.get("repair_commands")):
            if isinstance(command, list) and command:
                rows.append(
                    {
                        "source": name,
                        "reason": f"check:{check.get('name', '')}",
                        "command": [str(part) for part in command],
                    }
                )
    return rows


def _infrabot_efficiency_plan(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    lane_spec_list = [raw for raw in _as_list(config.get("infrabot_efficiency_lanes")) if isinstance(raw, dict)]
    specs = _lane_specs(config)
    rows = _collect_default_lane_commands(config)
    for filename, name in (
        ("infrastructure_autofix_bot_latest.json", "infrastructure_autofix_bot"),
        ("master_infrastructure_supervisor_latest.json", "master_infrastructure_supervisor"),
        ("system_drift_autopilot_latest.json", "system_drift_autopilot"),
        ("infrabot_adaptive_governor_latest.json", "infrabot_adaptive_governor"),
    ):
        rows.extend(_collect_payload_commands(name, _health(project_root, filename)))

    seen: set[tuple[str, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        command = [str(part) for part in _as_list(row.get("command"))]
        if not command:
            continue
        key = _command_key(command)
        if key in seen:
            continue
        seen.add(key)
        lane = _command_lane(command, lane_spec_list)
        spec = specs.get(lane, {})
        max_parallel = max(1, _safe_int(spec.get("max_parallel"), 1 if lane == "storage_writer" else 2))
        single_writer = bool(max_parallel == 1 or lane == "storage_writer" or any(hint in _command_text(command).lower() for hint in SINGLE_WRITER_HINTS))
        deduped.append(
            {
                "lane": lane,
                "source": str(row.get("source") or ""),
                "reason": str(row.get("reason") or ""),
                "command": command,
                "max_parallel": 1 if single_writer else max_parallel,
                "single_writer_or_pressure_sensitive": single_writer,
                "pressure_policy": str(spec.get("pressure_policy") or "run_with_default_infrabot_rate_limits"),
                "stop_condition": str(spec.get("stop_condition") or "owning health artifact returns ready or advisory-only"),
                "authority_boundary": "safe_repair_or_read_only_no_live_execution_no_dependency_mutation",
            }
        )
    lane_counts = Counter(str(row.get("lane") or "general_infrabot") for row in deduped)
    return {
        "mode": "dedupe_group_and_rate_limit_infrabot_commands",
        "command_count": len(deduped),
        "lane_counts": dict(sorted(lane_counts.items())),
        "single_writer_command_count": sum(1 for row in deduped if row.get("single_writer_or_pressure_sensitive")),
        "max_parallel_writer_commands": 1,
        "commands": deduped,
        "efficiency_contract": {
            "deduplicates_repair_commands": True,
            "storage_and_writer_repairs_single_parallel": True,
            "uses_existing_owner_controls": True,
            "does_not_run_repairs_directly": True,
        },
    }


def _first_command(plan: dict[str, Any], lane: str) -> list[str]:
    for row in _as_list(plan.get("commands")):
        if isinstance(row, dict) and str(row.get("lane") or "") == lane and isinstance(row.get("command"), list):
            return [str(part) for part in row["command"]]
    return []


def _lane_artifacts(lane: str) -> list[str]:
    return list(LANE_SENSORY_ARTIFACTS.get(lane) or LANE_SENSORY_ARTIFACTS["general_infrabot"])


def _configured_reflex_phases(config: dict[str, Any]) -> list[dict[str, Any]]:
    nervous_config = _as_dict(config.get("autonomic_nervous_system"))
    configured = [row for row in _as_list(nervous_config.get("reflex_phases")) if isinstance(row, dict)]
    phases = configured or [dict(row) for row in AUTONOMIC_REFLEX_PHASES]
    minimum = max(1, _safe_int(nervous_config.get("minimum_reflexes_per_lane"), len(AUTONOMIC_REFLEX_PHASES)))
    while len(phases) < minimum:
        index = len(phases) + 1
        phases.append(
            {
                "phase_id": f"extended_reflex_{index:02d}",
                "title": f"Extended Reflex {index:02d}",
                "severity_floor": "watch",
                "allowed_action": "preserve_visible_state_and_route_to_owner_without_live_execution",
                "cooldown_seconds": 300,
                "escalation_target": "self_awareness_need_brief",
            }
        )
    return phases[:minimum]


def _autonomic_nervous_system(config: dict[str, Any], plan: dict[str, Any], artifact_rows: list[dict[str, Any]], library_scope: dict[str, Any]) -> dict[str, Any]:
    nervous_config = _as_dict(config.get("autonomic_nervous_system"))
    enabled = _bool(nervous_config.get("enabled", True))
    lane_specs = [row for row in _as_list(config.get("infrabot_efficiency_lanes")) if isinstance(row, dict)]
    lane_names = [str(row.get("lane") or "").strip() for row in lane_specs if str(row.get("lane") or "").strip()]
    if not lane_names:
        lane_names = sorted(str(lane) for lane in _as_dict(plan.get("lane_counts")).keys())
    phases = _configured_reflex_phases(config)
    reflexes: list[dict[str, Any]] = []
    for lane in lane_names:
        command = _first_command(plan, lane) or ["./scripts/ops/opsctl.sh", "system-needs", "--json"]
        artifacts = _lane_artifacts(lane)
        single_writer_cap = 1 if lane == "storage_writer" or any(hint in lane for hint in SINGLE_WRITER_HINTS) else _safe_int(
            next((row.get("max_parallel") for row in lane_specs if str(row.get("lane") or "") == lane), 1),
            1,
        )
        for phase in phases:
            phase_id = str(phase.get("phase_id") or phase.get("title") or "reflex").strip().lower().replace(" ", "_")
            reflexes.append(
                {
                    "reflex_id": f"{lane}.{phase_id}",
                    "lane": lane,
                    "phase": phase_id,
                    "title": str(phase.get("title") or phase_id.replace("_", " ").title()),
                    "severity_floor": str(phase.get("severity_floor") or "watch"),
                    "sensing_artifacts": artifacts,
                    "owner_command": command,
                    "allowed_action": str(phase.get("allowed_action") or "read_and_route_only"),
                    "cooldown_seconds": max(0, _safe_int(phase.get("cooldown_seconds"), 300)),
                    "proof_artifacts": artifacts,
                    "stop_condition": (
                        next((str(row.get("stop_condition") or "") for row in lane_specs if str(row.get("lane") or "") == lane), "")
                        or "owning health artifacts return ready or managed advisory with visible evidence"
                    ),
                    "escalation_target": str(phase.get("escalation_target") or "self_awareness_need_brief"),
                    "authority_boundary": "safe_repair_or_read_only_no_live_execution_no_dependency_mutation",
                    "live_execution_authority": False,
                    "dependency_mutation_authority": False,
                    "max_parallel": max(1, single_writer_cap),
                }
            )
    reflex_ids = [str(row.get("reflex_id") or "") for row in reflexes]
    lane_counts = Counter(str(row.get("lane") or "general_infrabot") for row in reflexes)
    incomplete_reflexes = [
        row
        for row in reflexes
        if not _as_list(row.get("owner_command"))
        or not _string_list(row.get("proof_artifacts"))
        or not str(row.get("stop_condition") or "").strip()
        or not str(row.get("escalation_target") or "").strip()
        or bool(row.get("live_execution_authority"))
        or bool(row.get("dependency_mutation_authority"))
    ]
    artifact_statuses = {
        str(row.get("name") or Path(str(row.get("path") or "")).name): {
            "status": row.get("status"),
            "ready": row.get("ready"),
            "blocking": row.get("blocking"),
        }
        for row in artifact_rows
    }
    complete = bool(
        enabled
        and lane_names
        and reflexes
        and not incomplete_reflexes
        and len(reflex_ids) == len(set(reflex_ids))
        and all(lane_counts.get(lane, 0) >= len(phases) for lane in lane_names)
    )
    return {
        "enabled": enabled,
        "mode": "autonomic_reflex_matrix_v1",
        "lane_count": len(lane_names),
        "lanes": lane_names,
        "reflex_phase_count": len(phases),
        "reflex_count": len(reflexes),
        "unique_reflex_count": len(set(reflex_ids)),
        "lane_reflex_counts": dict(sorted(lane_counts.items())),
        "minimum_reflexes_per_lane": len(phases),
        "incomplete_reflex_count": len(incomplete_reflexes),
        "incomplete_reflex_ids": [str(row.get("reflex_id") or "") for row in incomplete_reflexes],
        "all_reflex_ids_unique": len(reflex_ids) == len(set(reflex_ids)),
        "all_lanes_have_reflex_minimum": all(lane_counts.get(lane, 0) >= len(phases) for lane in lane_names),
        "all_reflexes_have_owner_commands": all(_as_list(row.get("owner_command")) for row in reflexes),
        "all_reflexes_have_proof_artifacts": all(_string_list(row.get("proof_artifacts")) for row in reflexes),
        "all_reflexes_have_escalation_targets": all(str(row.get("escalation_target") or "").strip() for row in reflexes),
        "all_reflexes_have_stop_conditions": all(str(row.get("stop_condition") or "").strip() for row in reflexes),
        "live_execution_authority": False,
        "dependency_mutation_authority": False,
        "authority_boundary": "safe_repair_or_read_only_no_live_execution_no_dependency_mutation",
        "sensory_bus": {
            "artifact_statuses": artifact_statuses,
            "library_route_count": library_scope.get("existing_library_route_count"),
            "library_blocked_routes": library_scope.get("existing_library_blocked_routes"),
            "library_degraded_routes": library_scope.get("existing_library_degraded_routes"),
            "infrabot_lane_counts": plan.get("lane_counts"),
        },
        "grade": "A+" if complete else "F",
        "reflexes": reflexes,
    }


def _need_with_reflex_route(need: dict[str, Any], nervous_system: dict[str, Any]) -> dict[str, Any]:
    lane = str(need.get("reflex_lane") or need.get("owner") or "general_infrabot")
    target_phase = "repair" if str(need.get("status") or "") == "hard_blocker" else "refresh"
    reflexes = [row for row in _as_list(nervous_system.get("reflexes")) if isinstance(row, dict) and str(row.get("lane") or "") == lane]
    selected = next((row for row in reflexes if str(row.get("phase") or "") == target_phase), None)
    if selected is None:
        selected = next(iter(reflexes), {})
    row = dict(need)
    row["reflex_route"] = {
        "reflex_id": selected.get("reflex_id"),
        "lane": selected.get("lane", lane),
        "phase": selected.get("phase"),
        "owner_command": selected.get("owner_command") or need.get("safe_next_command"),
        "proof_artifacts": selected.get("proof_artifacts") or [],
        "cooldown_seconds": selected.get("cooldown_seconds"),
        "escalation_target": selected.get("escalation_target"),
        "authority_boundary": selected.get("authority_boundary") or "safe_repair_or_read_only_no_live_execution_no_dependency_mutation",
    }
    return row


def _reflex_by_lane_phase(nervous_system: dict[str, Any], lane: str, phase: str) -> dict[str, Any]:
    for row in _as_list(nervous_system.get("reflexes")):
        if not isinstance(row, dict):
            continue
        if str(row.get("lane") or "") == lane and str(row.get("phase") or "") == phase:
            return row
    return {}


def _self_healing_playbooks(config: dict[str, Any], nervous_system: dict[str, Any], need_rows: list[dict[str, Any]]) -> dict[str, Any]:
    healing_config = _as_dict(config.get("self_healing_contract"))
    enabled = _bool(healing_config.get("enabled", True))
    default_max_attempts = max(1, _safe_int(healing_config.get("default_max_attempts_per_incident"), 2))
    default_cooldown = max(60, _safe_int(healing_config.get("default_cooldown_seconds"), 300))
    lanes = _string_list(nervous_system.get("lanes"))
    need_count_by_lane = Counter(
        str(_as_dict(row.get("reflex_route")).get("lane") or row.get("reflex_lane") or row.get("owner") or "general_infrabot")
        for row in need_rows
        if isinstance(row, dict)
    )
    playbooks: list[dict[str, Any]] = []
    for lane in lanes:
        profile = dict(LANE_HEALING_CAPABILITIES.get(lane) or {})
        repair_reflex = _reflex_by_lane_phase(nervous_system, lane, "repair")
        refresh_reflex = _reflex_by_lane_phase(nervous_system, lane, "refresh")
        verify_reflex = _reflex_by_lane_phase(nervous_system, lane, "verify")
        owner_command = _as_list(repair_reflex.get("owner_command")) or _as_list(refresh_reflex.get("owner_command"))
        verify_command = _as_list(profile.get("verify_command")) or _as_list(verify_reflex.get("owner_command")) or owner_command
        proof_artifacts = ordered_unique(_string_list(verify_reflex.get("proof_artifacts")) or _string_list(repair_reflex.get("proof_artifacts")) or _lane_artifacts(lane))
        cooldown = max(60, _safe_int(profile.get("cooldown_seconds"), default_cooldown))
        max_attempts = max(1, _safe_int(profile.get("max_attempts"), default_max_attempts))
        hold_condition = (
            "stop automatic attempts, keep the issue visible, and escalate through the reflex target "
            "when retry budget is exhausted, proof artifacts fail verification, or safety guard removes apply authority"
        )
        complete = bool(
            enabled
            and owner_command
            and verify_command
            and proof_artifacts
            and str(repair_reflex.get("stop_condition") or verify_reflex.get("stop_condition") or "").strip()
            and not bool(repair_reflex.get("live_execution_authority"))
            and not bool(repair_reflex.get("dependency_mutation_authority"))
        )
        playbooks.append(
            {
                "playbook_id": f"{lane}.self_heal",
                "lane": lane,
                "active_need_count": int(need_count_by_lane.get(lane, 0)),
                "primary_capability": str(profile.get("primary_capability") or lane),
                "secondary_capabilities": _string_list(profile.get("secondary_capabilities")),
                "owner_command": [str(part) for part in owner_command],
                "verify_command": [str(part) for part in verify_command],
                "proof_artifacts": proof_artifacts,
                "max_attempts_per_incident": max_attempts,
                "cooldown_seconds": cooldown,
                "retry_backoff_seconds": [cooldown, min(cooldown * 2, 3600), min(cooldown * 4, 7200)][:max_attempts],
                "allowed_under_pressure": bool(profile.get("allowed_under_pressure", False)),
                "requires_single_writer_idle": lane == "storage_writer",
                "stop_condition": str(repair_reflex.get("stop_condition") or verify_reflex.get("stop_condition") or ""),
                "hold_condition": hold_condition,
                "escalation_target": str(repair_reflex.get("escalation_target") or "infrabot_adaptive_governor"),
                "proof_obligation": "verify_command exits cleanly and proof_artifacts show ready, guarded, or managed-visible state before another attempt is allowed",
                "authority_boundary": "safe_repair_or_read_only_no_live_execution_no_dependency_mutation",
                "live_execution_authority": False,
                "dependency_mutation_authority": False,
                "budget_consumption_policy": "consume_budget_only_after_exact_allowlisted_apply_command_runs_or_times_out",
                "cooldown_policy": "backoff_after_retryable_blocked_timeout_or_failed_result",
                "rollback_policy": "no source dependency or live-order rollback authority; publish hold/escalation if verification fails",
                "complete": complete,
                "grade": "A+" if complete else "F",
            }
        )
    playbook_ids = [str(row.get("playbook_id") or "") for row in playbooks]
    complete_count = sum(1 for row in playbooks if row.get("complete"))
    by_lane = {str(row.get("lane") or ""): row for row in playbooks}
    route_complete = all(
        bool(by_lane.get(str(_as_dict(row.get("reflex_route")).get("lane") or row.get("reflex_lane") or row.get("owner") or "")))
        for row in need_rows
    )
    authority_safe = bool(
        all(not bool(row.get("live_execution_authority")) and not bool(row.get("dependency_mutation_authority")) for row in playbooks)
    )
    complete = bool(
        enabled
        and playbooks
        and len(playbook_ids) == len(set(playbook_ids))
        and complete_count == len(playbooks)
        and route_complete
        and authority_safe
    )
    return {
        "enabled": enabled,
        "mode": "bounded_self_healing_playbooks_v1",
        "playbook_count": len(playbooks),
        "complete_playbook_count": complete_count,
        "unique_playbook_count": len(set(playbook_ids)),
        "lane_count": len(lanes),
        "all_playbook_ids_unique": len(playbook_ids) == len(set(playbook_ids)),
        "all_lanes_have_playbooks": len(playbooks) == len(lanes),
        "all_needs_have_playbooks": route_complete,
        "all_playbooks_complete": complete_count == len(playbooks),
        "authority_safe": authority_safe,
        "max_attempts_default": default_max_attempts,
        "cooldown_default_seconds": default_cooldown,
        "live_execution_authority": False,
        "dependency_mutation_authority": False,
        "grade": "A+" if complete else "F",
        "playbooks": playbooks,
    }


def _need_with_healing_playbook(need: dict[str, Any], healing: dict[str, Any]) -> dict[str, Any]:
    lane = str(_as_dict(need.get("reflex_route")).get("lane") or need.get("reflex_lane") or need.get("owner") or "general_infrabot")
    selected = next(
        (row for row in _as_list(healing.get("playbooks")) if isinstance(row, dict) and str(row.get("lane") or "") == lane),
        {},
    )
    row = dict(need)
    row["healing_playbook"] = {
        "playbook_id": selected.get("playbook_id"),
        "lane": selected.get("lane", lane),
        "primary_capability": selected.get("primary_capability"),
        "max_attempts_per_incident": selected.get("max_attempts_per_incident"),
        "cooldown_seconds": selected.get("cooldown_seconds"),
        "verify_command": selected.get("verify_command") or [],
        "proof_artifacts": selected.get("proof_artifacts") or [],
        "hold_condition": selected.get("hold_condition") or "",
        "authority_boundary": selected.get("authority_boundary") or "safe_repair_or_read_only_no_live_execution_no_dependency_mutation",
    }
    return row


def _managed_dashboard_needs(project_root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    dashboard = _health(project_root, "runtime_gate_dashboard_latest.json")
    overall = _as_dict(dashboard.get("overall"))
    rows: list[dict[str, Any]] = []
    for raw in _as_list(overall.get("managed_controls")):
        if not isinstance(raw, dict):
            continue
        attention = str(raw.get("attention") or "managed_attention").strip()
        lane = "storage_writer" if "storage" in attention else "governance_regression"
        if "coordination" in attention:
            lane = "auth_live_lock"
        if "infrastructure" in attention:
            lane = "governance_regression"
        rows.append(
            {
                "need_id": attention,
                "status": "managed_soak_advisory",
                "owner": str(raw.get("managed_by") or "unattended_soak_readiness"),
                "reflex_lane": lane,
                "symptom": attention,
                "why_it_matters": "This is visible production telemetry, but the soak manager has verified paper/auth/storage/livefeed contracts are still clean.",
                "safe_next_command": _first_command(plan, lane),
                "stop_condition": str(raw.get("when_to_unmanage") or "item returns ready or stops being managed by soak readiness"),
                "authority_boundary": "visibility_and_safe_repair_only_no_live_execution",
                "urgency": "watch",
                "soak_impact": "does_not_block_guarded_paper_soak_while_soak_management_context_remains_ready",
                "evidence": raw,
            }
        )
    return rows


def _artifact_needs(artifact_rows: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in artifact_rows:
        if row.get("ready"):
            continue
        lane = "runtime_memory"
        name = str(row.get("name") or "")
        if "library" in name or "mlx" in name:
            lane = "runtime_memory"
        elif "storage" in name or "writer" in name:
            lane = "storage_writer"
        elif "auth" in name or "production" in name:
            lane = "auth_live_lock"
        elif "source" in name:
            lane = "source_truth"
        elif "regression" in name or "drift" in name or "infra" in name:
            lane = "governance_regression"
        rows.append(
            {
                "need_id": name,
                "status": "hard_blocker" if row.get("blocking") else "owned_advisory",
                "owner": lane,
                "reflex_lane": lane,
                "symptom": ",".join(_string_list(row.get("blockers"))) or f"{name}_not_ready",
                "why_it_matters": "Required control artifacts must stay current and truthful so the soak does not hide paper trading, storage, auth, or routing faults.",
                "safe_next_command": _first_command(plan, lane),
                "stop_condition": "artifact row reports ready and all required paths pass",
                "authority_boundary": "safe_repair_or_read_only_no_live_execution_no_dependency_mutation",
                "urgency": "critical" if row.get("blocking") else "watch",
                "soak_impact": "blocks_soak_confidence" if row.get("blocking") else "visible_but_managed",
                "evidence": row,
            }
        )
    return rows


def _library_upgrade_scope(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    library_utilization = _health(project_root, "library_utilization_router_latest.json")
    library_upgrade = _health(project_root, "library_upgrade_route_control_latest.json")
    mlx_router = _health(project_root, "mlx_intelligence_router_latest.json")
    coverage = _as_dict(library_utilization.get("coverage"))
    candidate_matrix = _as_dict(library_utilization.get("candidate_library_matrix"))
    upgrade_plan = _as_dict(library_upgrade.get("upgrade_plan"))
    route_matrix = _as_dict(library_upgrade.get("route_matrix"))
    runtime_family_counts = _as_dict(candidate_matrix.get("runtime_family_counts"))
    return {
        "mode": str(upgrade_plan.get("mode") or "route_now_plan_upgrades_without_mutating_dependencies"),
        "dependency_mutation_allowed_during_soak": bool(upgrade_plan.get("soak_dependency_mutation_allowed", False)),
        "existing_python_packages_managed": _safe_int(coverage.get("managed_non_mlx_package_count"), 0),
        "existing_python_coverage_ratio": _safe_float(coverage.get("coverage_ratio"), 0.0),
        "existing_library_route_count": _safe_int(route_matrix.get("route_count"), 0),
        "existing_library_blocked_routes": _safe_int(route_matrix.get("blocked_route_count"), 0),
        "existing_library_degraded_routes": _safe_int(route_matrix.get("degraded_route_count"), 0),
        "hard_upgrade_blockers": _safe_int(upgrade_plan.get("hard_blocker_count"), 0),
        "actionable_upgrade_packages": _safe_int(upgrade_plan.get("actionable_package_count"), 0),
        "candidate_package_count": _safe_int(candidate_matrix.get("candidate_package_count"), 0),
        "candidate_route_coverage_ratio": _safe_float(candidate_matrix.get("mapped_candidate_ratio"), 0.0),
        "candidate_python_packages": _safe_int(runtime_family_counts.get("python"), 0),
        "candidate_mlx_packages": _safe_int(runtime_family_counts.get("mlx"), 0),
        "configured_candidate_additions": _string_list(config.get("candidate_library_additions")),
        "mlx_status": str(mlx_router.get("overall_status") or ""),
        "mlx_runtime_caps": _as_dict(mlx_router.get("runtime_caps")),
        "upgrade_safety_contract": {
            "route_existing_python_and_mlx_libraries": True,
            "stage_new_candidates_without_installing": True,
            "pip_install_or_lock_rewrite_ran": False,
            "maintenance_required_for_dependency_mutation": True,
        },
    }


def _quality_checks(
    artifact_rows: list[dict[str, Any]],
    library_scope: dict[str, Any],
    infrabot_plan: dict[str, Any],
    nervous_system: dict[str, Any],
    self_healing: dict[str, Any],
    need_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, bool]:
    contract = _as_dict(config.get("control_contract"))
    comms = _as_dict(config.get("communication_contract"))
    required_fields = _string_list(comms.get("required_fields"))
    required_ready = all(not row.get("blocking") for row in artifact_rows)
    single_writer_ok = bool(_safe_int(infrabot_plan.get("max_parallel_writer_commands"), 99) <= 1)
    return {
        "required_artifacts_ready": required_ready,
        "existing_python_libraries_fully_routed": _safe_float(library_scope.get("existing_python_coverage_ratio"), 0.0) >= 1.0,
        "library_upgrade_hard_blockers_zero": _safe_int(library_scope.get("hard_upgrade_blockers"), 0) == 0,
        "library_routes_not_blocked_or_degraded": _safe_int(library_scope.get("existing_library_blocked_routes"), 0) == 0
        and _safe_int(library_scope.get("existing_library_degraded_routes"), 0) == 0,
        "dependency_mutation_disabled": not bool(library_scope.get("dependency_mutation_allowed_during_soak")),
        "live_execution_authority_false": not bool(contract.get("live_execution_authority", True)),
        "paper_soak_safe_contract": bool(contract.get("paper_soak_safe", False)),
        "infrabot_command_dedupe_present": _safe_int(infrabot_plan.get("command_count"), 0) > 0,
        "single_writer_storage_cap": single_writer_ok,
        "communication_contract_complete": all(field in required_fields for field in ("owner", "symptom", "safe_next_command", "stop_condition", "authority_boundary", "soak_impact")),
        "autonomic_nervous_system_enabled": bool(nervous_system.get("enabled", False)),
        "autonomic_lane_reflex_coverage_complete": bool(nervous_system.get("all_lanes_have_reflex_minimum", False)),
        "autonomic_reflex_ids_unique": bool(nervous_system.get("all_reflex_ids_unique", False)),
        "autonomic_reflexes_have_owner_commands": bool(nervous_system.get("all_reflexes_have_owner_commands", False)),
        "autonomic_reflexes_have_proof_artifacts": bool(nervous_system.get("all_reflexes_have_proof_artifacts", False)),
        "autonomic_reflexes_have_stop_conditions": bool(nervous_system.get("all_reflexes_have_stop_conditions", False)),
        "autonomic_reflexes_have_escalation_targets": bool(nervous_system.get("all_reflexes_have_escalation_targets", False)),
        "autonomic_need_brief_routes_present": all(bool(_as_dict(row.get("reflex_route")).get("reflex_id")) for row in need_rows),
        "autonomic_authority_safe": not bool(nervous_system.get("live_execution_authority", True))
        and not bool(nervous_system.get("dependency_mutation_authority", True))
        and _safe_int(nervous_system.get("incomplete_reflex_count"), 1) == 0,
        "self_healing_playbooks_enabled": bool(self_healing.get("enabled", False)),
        "self_healing_lane_coverage_complete": bool(self_healing.get("all_lanes_have_playbooks", False)),
        "self_healing_playbooks_complete": bool(self_healing.get("all_playbooks_complete", False)),
        "self_healing_playbook_ids_unique": bool(self_healing.get("all_playbook_ids_unique", False)),
        "self_healing_need_playbooks_present": bool(self_healing.get("all_needs_have_playbooks", False)),
        "self_healing_authority_safe": bool(self_healing.get("authority_safe", False))
        and not bool(self_healing.get("live_execution_authority", True))
        and not bool(self_healing.get("dependency_mutation_authority", True)),
    }


def _grade(ready_count: int, total_count: int) -> str:
    if total_count <= 0:
        return "F"
    if ready_count == total_count:
        return "A+"
    ratio = ready_count / total_count
    if ratio >= 0.9:
        return "A"
    if ratio >= 0.75:
        return "B"
    if ratio >= 0.5:
        return "C"
    if ratio >= 0.25:
        return "D"
    return "F"


def _write_env_override(path: Path, payload: dict[str, Any]) -> bool:
    env = {
        "INFRABOT_LIBRARY_SELF_AWARENESS_ENABLED": "1",
        "INFRABOT_AUTONOMIC_NERVOUS_SYSTEM_ENABLED": "1",
        "INFRABOT_AUTONOMIC_REFLEX_MIN_PER_LANE": str(
            _safe_int(_as_dict(payload.get("autonomic_nervous_system")).get("minimum_reflexes_per_lane"), len(AUTONOMIC_REFLEX_PHASES))
        ),
        "INFRABOT_SELF_HEALING_PLAYBOOKS_ENABLED": "1",
        "INFRABOT_SELF_HEALING_PLAYBOOK_COUNT": str(_safe_int(_as_dict(payload.get("self_healing_playbooks")).get("playbook_count"), 0)),
        "INFRABOT_REPAIR_COMMAND_DEDUPE": "1",
        "INFRABOT_STORAGE_WRITER_MAX_PARALLEL": "1",
        "INFRABOT_NEEDS_COMMUNICATION_DEPTH": "owner_command_stop_condition_authority",
        "LIBRARY_UPGRADE_ROUTE_DEPENDENCY_MUTATION_ALLOWED": "0",
        "PRIMARY_ML_RUNTIME_BACKEND": "mlx",
    }
    lines = ["# Auto-managed by scripts/ops/infrabot_library_self_awareness_control.py"]
    for key, value in sorted(env.items()):
        safe_value = str(value).replace("'", "'\"'\"'")
        lines.append(f"{key}='{safe_value}'")
    lines.append(f"INFRABOT_LIBRARY_SELF_AWARENESS_GRADE='{payload.get('grade', '')}'")
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def render_markdown(payload: dict[str, Any]) -> str:
    library_scope = _as_dict(payload.get("library_upgrade_scope"))
    plan = _as_dict(payload.get("infrabot_efficiency_plan"))
    nervous = _as_dict(payload.get("autonomic_nervous_system"))
    healing = _as_dict(payload.get("self_healing_playbooks"))
    lines = [
        "# Infrabot Library Self Awareness Control",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        f"Status: `{payload.get('overall_status', '')}`",
        f"Grade: `{payload.get('grade', '')}`",
        "",
        "## Library Routing",
        "",
        f"- Existing Python packages managed: `{library_scope.get('existing_python_packages_managed', 0)}`",
        f"- Existing Python coverage ratio: `{library_scope.get('existing_python_coverage_ratio', 0.0)}`",
        f"- Existing library routes: `{library_scope.get('existing_library_route_count', 0)}`",
        f"- Hard upgrade blockers: `{library_scope.get('hard_upgrade_blockers', 0)}`",
        f"- Dependency mutation during soak: `{library_scope.get('dependency_mutation_allowed_during_soak', False)}`",
        f"- Candidate packages staged: `{library_scope.get('candidate_package_count', 0)}`",
        f"- Candidate MLX packages staged: `{library_scope.get('candidate_mlx_packages', 0)}`",
        "",
        "## Infrabot Efficiency",
        "",
        f"- Deduped commands: `{plan.get('command_count', 0)}`",
        f"- Single-writer or pressure-sensitive commands: `{plan.get('single_writer_command_count', 0)}`",
        f"- Lane counts: `{json.dumps(plan.get('lane_counts') or {}, sort_keys=True)}`",
        "",
        "## Autonomic Nervous System",
        "",
        f"- Grade: `{nervous.get('grade', '')}`",
        f"- Lanes: `{nervous.get('lane_count', 0)}`",
        f"- Reflexes: `{nervous.get('reflex_count', 0)}`",
        f"- Reflexes per lane: `{nervous.get('minimum_reflexes_per_lane', 0)}`",
        f"- Authority boundary: `{nervous.get('authority_boundary', '')}`",
        "",
        "## Self Healing Playbooks",
        "",
        f"- Grade: `{healing.get('grade', '')}`",
        f"- Playbooks: `{healing.get('playbook_count', 0)}`",
        f"- Complete: `{healing.get('complete_playbook_count', 0)}/{healing.get('playbook_count', 0)}`",
        f"- Authority safe: `{healing.get('authority_safe', False)}`",
        "",
        "## Needs Brief",
        "",
    ]
    for need in _as_list(payload.get("self_awareness_need_brief"))[:20]:
        if not isinstance(need, dict):
            continue
        command = " ".join(str(part) for part in _as_list(need.get("safe_next_command"))) or "none"
        route = _as_dict(need.get("reflex_route"))
        playbook = _as_dict(need.get("healing_playbook"))
        lines.append(f"- `{need.get('status', '')}` `{need.get('need_id', '')}` owner=`{need.get('owner', '')}` urgency=`{need.get('urgency', '')}`")
        lines.append(f"  command: `{command}`")
        lines.append(f"  reflex: `{route.get('reflex_id', '')}` escalate=`{route.get('escalation_target', '')}`")
        lines.append(
            f"  healing: `{playbook.get('playbook_id', '')}` attempts=`{playbook.get('max_attempts_per_incident', '')}` cooldown=`{playbook.get('cooldown_seconds', '')}`"
        )
        lines.append(f"  stop: {need.get('stop_condition', '')}")
    lines.extend(["", "## Control Contract", "", "- Live execution authority: `false`", "- Dependency mutation during soak: `false`", "- Raw profitability truth remains external and evidence-based."])
    return "\n".join(lines).rstrip() + "\n"


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path = DEFAULT_CONFIG,
    apply: bool = False,
) -> dict[str, Any]:
    config = load_json(config_path)
    artifact_rows = [
        _artifact_row(project_root, raw)
        for raw in _as_list(config.get("artifact_requirements"))
        if isinstance(raw, dict)
    ]
    infrabot_plan = _infrabot_efficiency_plan(project_root, config)
    library_scope = _library_upgrade_scope(project_root, config)
    nervous_system = _autonomic_nervous_system(config, infrabot_plan, artifact_rows, library_scope)
    need_rows = [
        _need_with_reflex_route(row, nervous_system)
        for row in (_managed_dashboard_needs(project_root, infrabot_plan) + _artifact_needs(artifact_rows, infrabot_plan))
    ]
    self_healing = _self_healing_playbooks(config, nervous_system, need_rows)
    need_rows = [_need_with_healing_playbook(row, self_healing) for row in need_rows]
    need_rows = sorted(
        need_rows,
        key=lambda row: (
            0 if str(row.get("status") or "") == "hard_blocker" else 1,
            str(row.get("urgency") or ""),
            str(row.get("need_id") or ""),
        ),
    )
    quality_checks = _quality_checks(artifact_rows, library_scope, infrabot_plan, nervous_system, self_healing, need_rows, config)
    ready_quality_count = sum(1 for value in quality_checks.values() if value)
    blockers = ordered_unique(
        [
            f"{row.get('name')}:{blocker}"
            for row in artifact_rows
            if row.get("blocking")
            for blocker in _string_list(row.get("blockers"))
        ]
        + [f"quality:{name}" for name, ok in quality_checks.items() if not ok]
    )
    overall_ready = bool(not blockers and ready_quality_count == len(quality_checks))
    contract = _as_dict(config.get("control_contract"))
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": SCHEMA_VERSION,
        "source": "infrabot_library_self_awareness_control",
        "ok": overall_ready,
        "overall_status": "ready" if overall_ready else "needs_work",
        "grade": _grade(ready_quality_count, len(quality_checks)),
        "target_grade": str(config.get("target_grade") or "A+"),
        "quality_checks": quality_checks,
        "ready_quality_count": ready_quality_count,
        "quality_check_count": len(quality_checks),
        "blockers": blockers,
        "artifact_rows": artifact_rows,
        "library_upgrade_scope": library_scope,
        "infrabot_efficiency_plan": infrabot_plan,
        "autonomic_nervous_system": nervous_system,
        "self_healing_playbooks": self_healing,
        "self_awareness_need_brief": need_rows,
        "communications_contract": _as_dict(config.get("communication_contract")),
        "control_contract": {
            "live_execution_authority": False,
            "live_orders_must_remain_disabled": True,
            "paper_soak_safe": bool(contract.get("paper_soak_safe", True)),
            "dependency_mutation_allowed_during_soak": False,
            "pip_install_or_lock_rewrite_ran": False,
            "raw_profitability_truth_must_remain_visible": bool(contract.get("raw_profitability_truth_must_remain_visible", True)),
            "managed_dashboard_attention_does_not_fake_green": True,
            "autonomic_nervous_system_enabled": bool(nervous_system.get("enabled", False)),
            "autonomic_reflex_count": nervous_system.get("reflex_count"),
            "autonomic_reflex_grade": nervous_system.get("grade"),
            "self_healing_playbooks_enabled": bool(self_healing.get("enabled", False)),
            "self_healing_playbook_count": self_healing.get("playbook_count"),
            "self_healing_playbook_grade": self_healing.get("grade"),
        },
        "recommended_actions": ordered_unique(
            [
                "./scripts/ops/opsctl.sh library-utilization-router --apply --json",
                "./scripts/ops/opsctl.sh library-upgrade-route --apply --json",
                "./scripts/ops/opsctl.sh mlx-intelligence-router --apply --json",
                "./scripts/ops/opsctl.sh infrabot-library-self-awareness --apply --json",
                "use autonomic_nervous_system.reflexes to route degradation through sense/classify/refresh/repair/verify/escalate phases",
                "use self_healing_playbooks.playbooks for retry budgets, verification commands, proof artifacts, and safe hold escalation",
                "keep dependency installs in a maintenance window; this control only routes and stages during soak",
                "run the first safe_next_command for any hard_blocker need, then refresh this control",
            ]
        ),
        "apply_result": {
            "applied": bool(apply),
            "dependency_mutation_ran": False,
            "live_execution_authority_changed": False,
        },
        "artifact_paths": {
            "json": str(DEFAULT_OUT_PATH),
            "external_context": str(DEFAULT_EXTERNAL_CONTEXT_PATH),
            "markdown": str(DEFAULT_MARKDOWN_PATH),
            "config": str(config_path),
            "env_override": str(DEFAULT_OVERRIDE_PATH),
        },
    }
    return payload


def write_outputs(
    payload: dict[str, Any],
    *,
    out_path: Path = DEFAULT_OUT_PATH,
    external_context_path: Path = DEFAULT_EXTERNAL_CONTEXT_PATH,
    markdown_path: Path = DEFAULT_MARKDOWN_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    apply: bool = False,
) -> None:
    if apply:
        payload["apply_result"]["env_override_changed"] = _write_env_override(override_path, payload)
    write_payload(out_path, payload)
    write_payload(external_context_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Coordinate infrabot efficiency, library upgrade routing, and deep needs communication.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--external-context-file", type=Path, default=DEFAULT_EXTERNAL_CONTEXT_PATH)
    parser.add_argument("--markdown-file", type=Path, default=DEFAULT_MARKDOWN_PATH)
    parser.add_argument("--override-file", type=Path, default=DEFAULT_OVERRIDE_PATH)
    parser.add_argument("--apply", action="store_true", help="Write health, external context, markdown, and env override artifacts.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero unless the control is ready.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(args.project_root.resolve(), config_path=args.config.resolve(), apply=bool(args.apply))
    write_outputs(
        payload,
        out_path=args.out_file.expanduser(),
        external_context_path=args.external_context_file.expanduser(),
        markdown_path=args.markdown_file.expanduser(),
        override_path=args.override_file.expanduser(),
        apply=bool(args.apply),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "infrabot_library_self_awareness_control "
            f"status={payload.get('overall_status', '')} "
            f"grade={payload.get('grade', '')} "
            f"needs={len(payload.get('self_awareness_need_brief') or [])}"
        )
    return 2 if args.check and not payload.get("ok") else 0


if __name__ == "__main__":
    raise SystemExit(main())
