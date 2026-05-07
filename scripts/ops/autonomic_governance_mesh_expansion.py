#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_VERSION = 1558
TARGET_PLATFORM_TOTAL_BOTS = 1604
PACK_VERSION = "autonomic_governance_mesh_v1"
PACK_SLUG = "autonomic_governance_mesh"
PACK_DISPLAY_NAME = "Autonomic Governance Mesh Pack"
SLEEVE_FAMILY = "autonomic_governance_mesh"
LABEL_CONTRACT_VERSION = "autonomic_governance_mesh_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 70000
MINIMUM_COLLECTION_DAYS = 180
SAMPLE_RATE = 0.012
MAX_DAILY_MB_PER_BOT = 2


GOVERNANCE_SYSTEMS: list[dict[str, Any]] = [
    {
        "slug": "system_governor_council",
        "layer": "system_governor",
        "display_name": "System Governor Council",
        "objective": "Fuse surface status, pressure, dependencies, and evidence into one governor vote.",
        "outputs": ["governor_vote", "surface_risk_rank", "safe_mode_recommendation"],
    },
    {
        "slug": "sleeve_budget_market",
        "layer": "sleeve_economy",
        "display_name": "Sleeve Budget Market",
        "objective": "Allocate collection, freshness, compute, and storage budgets by value, risk, and pressure.",
        "outputs": ["sleeve_budget_curve", "budget_exception_packet", "sleep_or_wake_vote"],
    },
    {
        "slug": "evidence_packet_court",
        "layer": "evidence_court",
        "display_name": "Evidence Packet Court",
        "objective": "Require auditable promotion packets before a strategy leaves collect-only status.",
        "outputs": ["evidence_packet_status", "promotion_blocker_list", "packet_quality_score"],
    },
    {
        "slug": "promotion_gate_witness",
        "layer": "evidence_court",
        "display_name": "Promotion Gate Witness",
        "objective": "Witness every promotion request against leakage, duplicate-alpha, paper/live, and cost gates.",
        "outputs": ["promotion_witness_attestation", "unsafe_diff_alert", "gate_reason_codes"],
    },
    {
        "slug": "memory_storage_triage_router",
        "layer": "memory_storage_triage",
        "display_name": "Memory Storage Triage Router",
        "objective": "Route collectors between raw, digest, heartbeat, and parked capture tiers.",
        "outputs": ["capture_tier_route", "storage_pressure_tradeoff", "collector_downgrade_packet"],
    },
    {
        "slug": "collector_value_decay_scorer",
        "layer": "memory_storage_triage",
        "display_name": "Collector Value Decay Scorer",
        "objective": "Score stale or duplicative collectors for heartbeat, parking, or deletion review.",
        "outputs": ["collector_value_score", "decay_reason_codes", "parking_candidate_list"],
    },
    {
        "slug": "backlog_effect_ledger",
        "layer": "backlog_outcome_learning",
        "display_name": "Backlog Effect Ledger",
        "objective": "Measure whether drainers, organizers, and writers actually reduce pressure after action.",
        "outputs": ["backlog_effect_delta", "drainer_outcome_verdict", "playbook_reward_signal"],
    },
    {
        "slug": "drainer_playbook_optimizer",
        "layer": "backlog_outcome_learning",
        "display_name": "Drainer Playbook Optimizer",
        "objective": "Choose the least risky backlog drain sequence for current memory, storage, and market window.",
        "outputs": ["drainer_playbook_rank", "safe_window_vote", "rollback_condition_packet"],
    },
    {
        "slug": "dependency_blast_radius_mapper",
        "layer": "self_model_upgrade",
        "display_name": "Dependency Blast Radius Mapper",
        "objective": "Map which launchers, reports, writers, and sleeves are affected by each degraded surface.",
        "outputs": ["blast_radius_graph", "blocked_edge_packet", "dependency_repair_order"],
    },
    {
        "slug": "freshness_conflict_arbitrator",
        "layer": "self_model_upgrade",
        "display_name": "Freshness Conflict Arbitrator",
        "objective": "Detect stale or contradictory health artifacts before they mislead the governor.",
        "outputs": ["freshness_conflict_vote", "stale_surface_rank", "last_good_artifact_map"],
    },
    {
        "slug": "operator_decision_digestor",
        "layer": "operator_interface",
        "display_name": "Operator Decision Digestor",
        "objective": "Compress the governor state into a short operator-facing packet with next safe actions.",
        "outputs": ["operator_decision_digest", "attention_queue_delta", "approval_needed_list"],
    },
    {
        "slug": "codex_handoff_negotiator",
        "layer": "operator_interface",
        "display_name": "Codex Handoff Negotiator",
        "objective": "Format system state and safe action proposals so Codex can reason over them without live authority.",
        "outputs": ["codex_handoff_packet", "assistant_context_delta", "bounded_command_options"],
    },
    {
        "slug": "safe_action_simulator",
        "layer": "system_governor",
        "display_name": "Safe Action Simulator",
        "objective": "Simulate bounded ops commands before they are recommended to the operator.",
        "outputs": ["preflight_result", "postcondition_prediction", "do_not_do_guardrail"],
    },
    {
        "slug": "expansion_stability_stress_oracle",
        "layer": "sleeve_economy",
        "display_name": "Expansion Stability Stress Oracle",
        "objective": "Stress future expansion against memory, storage, fanout, freshness, and materialization limits.",
        "outputs": ["expansion_stability_score", "max_safe_wave_size", "capacity_debt_list"],
    },
]


ROLE_TEMPLATES: list[dict[str, Any]] = [
    {"suffix": "telemetry_collector", "label": "Telemetry Collector", "bot_role": "infrastructure_sub_bot", "priority": "high"},
    {"suffix": "value_scorer", "label": "Value Scorer", "bot_role": "signal_sub_bot", "priority": "high"},
    {"suffix": "policy_guard", "label": "Policy Guard", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
    {"suffix": "governor_bridge", "label": "Governor Bridge", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
]


BASE_DATA_INTAKES = [
    "whole_system_governor_trace",
    "system_self_model_trace",
    "quant_operational_intelligence_trace",
    "memory_efficiency_trace",
    "backpressure_drainer_trace",
    "operator_cockpit_trace",
    "codex_handoff_trace",
]


REQUIRED_LABELS = [
    "governor_action_effect_bucket",
    "sleeve_budget_effect_bucket",
    "evidence_packet_status",
    "memory_triage_effect_bucket",
    "backlog_drain_outcome_bucket",
    "operator_attention_quality_bucket",
    "paper_live_separation_status",
]


STORAGE_TARGETS = [
    "governance/autonomic_governance_mesh",
    *[f"governance/autonomic_governance_mesh/{system['slug']}" for system in GOVERNANCE_SYSTEMS],
    "governance/health/autonomic_governance_mesh_latest.json",
]


def _bot_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for system in GOVERNANCE_SYSTEMS:
        for role in ROLE_TEMPLATES:
            role_slug = f"{system['slug']}_{role['suffix']}"
            specs.append(
                {
                    "role_slug": role_slug,
                    "slug": f"autonomic_governance_{role_slug}_bot",
                    "label": f"{system['display_name']} {role['label']}",
                    "system": system["slug"],
                    "layer": system["layer"],
                    "bot_role": role["bot_role"],
                    "priority": role["priority"],
                    "objective": f"{role['label']} for {system['objective']}",
                    "target_functions": list(system["outputs"]),
                }
            )
    return specs


BOTS = _bot_specs()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _ensure_storage_targets(project_root: Path) -> list[str]:
    ready: list[str] = []
    for target in STORAGE_TARGETS:
        path = project_root / target
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            ready.append(str(path.parent.relative_to(project_root)))
        else:
            path.mkdir(parents=True, exist_ok=True)
            ready.append(str(path.relative_to(project_root)))
    return sorted(set(ready))


def _version_from_bot_id(bot_id: str) -> int | None:
    match = re.match(r"^brain_refinery_v(?P<version>\d+)", bot_id)
    return int(match.group("version")) if match else None


def _next_available_version(used_versions: set[int], start: int) -> int:
    version = start
    while version in used_versions:
        version += 1
    used_versions.add(version)
    return version


def _slot_kind(bot: dict[str, Any]) -> str:
    return f"{PACK_SLUG}_{bot['role_slug']}"


def _assign_bot_ids(rows: list[dict[str, Any]]) -> dict[str, str]:
    existing_by_slot = {
        str(row.get("slot_kind") or ""): str(row.get("bot_id") or "")
        for row in rows
        if str(row.get("slot_kind") or "") and str(row.get("bot_id") or "")
    }
    used_versions = {
        version
        for row in rows
        for version in [_version_from_bot_id(str(row.get("bot_id") or ""))]
        if version is not None
    }
    assigned: dict[str, str] = {}
    for index, bot in enumerate(BOTS):
        slot = _slot_kind(bot)
        if slot in existing_by_slot:
            assigned[slot] = existing_by_slot[slot]
            continue
        desired = BASE_VERSION + index
        if desired not in used_versions:
            version = desired
            used_versions.add(version)
        else:
            version = _next_available_version(used_versions, max(max(used_versions, default=BASE_VERSION - 1) + 1, desired))
        assigned[slot] = f"brain_refinery_v{version}_{bot['slug']}"
    return assigned


def _system(bot: dict[str, Any]) -> dict[str, Any]:
    for system in GOVERNANCE_SYSTEMS:
        if system["slug"] == bot["system"]:
            return system
    return {"slug": bot["system"], "layer": bot.get("layer", ""), "display_name": bot["system"], "outputs": []}


def _threshold_progress() -> dict[str, Any]:
    return {
        "observations": 0,
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "observations_ready": False,
        "collection_age_days": 0.0,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "days_ready": False,
        "training_ready": False,
    }


def _pack_contract(assigned_ids: dict[str, str]) -> dict[str, Any]:
    return {
        "contract_version": PACK_VERSION,
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "sleeve_family": SLEEVE_FAMILY,
        "display_name": PACK_DISPLAY_NAME,
        "system_count": len(GOVERNANCE_SYSTEMS),
        "bot_count": len(BOTS),
        "bot_pack_size_rule": "14_governance_systems_4_bots_each_56_bot_autonomic_mesh",
        "governance_layers": sorted({system["layer"] for system in GOVERNANCE_SYSTEMS}),
        "dedicated_data_intake": list(BASE_DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "autonomic_governance_hot_3d_warm_120d_cold_540d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": MAX_DAILY_MB_PER_BOT,
            "capture_mode": "thin_digest_and_heartbeat_first",
            "sample_rate": SAMPLE_RATE,
            "dedupe_required": True,
            "self_accommodation": "downgrade_to_heartbeat_under_memory_storage_or_fanout_pressure",
        },
        "paper_only_floor": {
            "paper_trade_lock_required": True,
            "paper_trading_enabled": False,
            "live_trading_enabled": False,
            "execution_enabled": False,
            "allocation_enabled": False,
            "graduation_requires_minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "graduation_requires_collection_days": MINIMUM_COLLECTION_DAYS,
        },
        "anchor_bot_ids": {
            bot["system"]: assigned_ids.get(_slot_kind(bot), "")
            for bot in BOTS
            if bot["role_slug"].endswith("telemetry_collector")
        },
        "governor_command": "./scripts/ops/opsctl.sh whole-system-governor --apply --json",
        "codex_communication_surface": "governance/health/codex_handoff_latest.json",
        "authority_boundary": "advisory_collect_only_no_execution_no_allocation_no_halt_clearance",
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    system = _system(bot)
    contract = _pack_contract(assigned_ids)
    system_slug = str(system["slug"])
    layer = str(system["layer"])
    data_intakes = list(BASE_DATA_INTAKES) + [
        f"{system_slug}_effect_trace",
        f"{system_slug}_label_quality_trace",
    ]
    return {
        "bot_id": bot_id,
        "bot_role": bot["bot_role"],
        "active": True,
        "reason": "autonomic_governance_mesh_expansion_slot",
        "weight": 0.0,
        "preference_score": 0.0,
        "quality_score": 0.0,
        "test_accuracy": None,
        "candidate_test_accuracy": None,
        "candidate_quality_score": 0.0,
        "previous_best_accuracy": None,
        "no_improvement_streak": 0,
        "deleted_from_rotation": False,
        "delete_reason": "",
        "promoted": False,
        "promotion_reason": "autonomic_governance_mesh_expansion_slot",
        "model_path": "",
        "log_file": "",
        "candidate_log_file": "",
        "lifecycle_state": "data_collection_only",
        "slot_label": bot["label"],
        "slot_kind": _slot_kind(bot),
        "slot_priority": bot["priority"],
        "slot_objective": bot["objective"],
        "target_functions": list(bot["target_functions"]),
        "preferred_regimes": [
            "protective_pressure",
            "constrained_collection",
            "normal_collection",
            "overnight_drain",
            "post_expansion_settlement",
            "operator_review",
        ],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v1478_quant_operational_alpha_factor_court_telemetry_collector_bot",
            "brain_refinery_v1522_quant_operational_backlog_outcome_verifier_telemetry_collector_bot",
            "brain_refinery_v1526_quant_operational_safe_command_router_telemetry_collector_bot",
            "brain_refinery_v1554_quant_operational_operator_decision_packet_builder_telemetry_collector_bot",
        ],
        "data_intake_collections": data_intakes,
        "storage_targets": [
            "governance/autonomic_governance_mesh",
            f"governance/autonomic_governance_mesh/{system_slug}",
            "governance/health/autonomic_governance_mesh_latest.json",
        ],
        "freshness_slo_seconds": 1800,
        "retention_profile": "autonomic_governance_hot_3d_warm_120d_cold_540d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "autonomic_governance_mesh_collect_only_until_evidence_resource_runtime_and_safety_gates_clear",
        "trading_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "allocation_enabled": False,
        "execution_enabled": False,
        "rotation_blocked": True,
        "rotation_block_reason": "autonomic_governance_mesh_collection_only_zero_weight",
        "training_excluded": True,
        "exclude_from_training": True,
        "training_candidate_after_threshold": True,
        "training_exclusion_reason": "collecting_governance_effect_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "training_threshold_policy": {
            "minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "minimum_collection_days": MINIMUM_COLLECTION_DAYS,
            "requires_whole_system_governor_clearance": True,
            "requires_runtime_pressure_clearance": True,
            "requires_backpressure_clearance": True,
            "requires_data_quality_clearance": True,
            "requires_paper_live_separation_clearance": True,
            "requires_global_halt_clear": True,
        },
        "data_collection_storage_guarded": True,
        "data_collection_capture_mode": "thin_digest_with_heartbeat_fallback",
        "data_collection_sample_rate": SAMPLE_RATE,
        "data_collection_max_daily_storage_mb": MAX_DAILY_MB_PER_BOT,
        "data_collection_max_daily_mb": float(MAX_DAILY_MB_PER_BOT),
        "data_collection_compute_guard_mode": "pressure_self_accommodating",
        "data_collection_resource_guard_reason": "autonomic_governance_mesh_defaults_to_low_storage_low_compute_advisory_capture",
        "self_accommodating_policy": {
            "pressure_default": "thin_digest",
            "high_pressure_fallback": "heartbeat",
            "critical_pressure_fallback": "parked_until_operator_review",
            "raw_trace_allowed": False,
        },
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "paper_execution_queue_policy": "blocked_until_governance_evidence_thresholds_clear",
        "paper_runtime_control_refresh_seconds": 360,
        "sleeve_profile": system_slug,
        "sleeve_family": SLEEVE_FAMILY,
        "governance_layer": layer,
        "intelligence_system": system_slug,
        "strategy_family": "autonomic_system_governance",
        "correlation_peer_sleeves": [
            "whole_system_governor",
            "quant_operational_intelligence",
            "system_self_model",
            "system_self_intelligence",
            "platform_organ_systems",
            "operator_cockpit_v2",
            "codex_handoff",
        ],
        "correlation_dependencies": [
            "whole_system_governor",
            "memory_efficiency_control",
            "backpressure_drainer_fleet",
            "paper_trade_lock_guard",
            "global_halt_guard",
        ],
        "provider_capability_profile": "internal_governance_and_codex_handoff_collect_only",
        "direct_market_data_available": False,
        "direct_execution_allowed": False,
        "proxy_data_sources": [
            "master_bot_registry",
            "governance_health",
            "whole_system_governor",
            "quant_operational_intelligence",
            "system_self_model",
            "codex_handoff",
        ],
        "schwab_direct_inputs": [],
        "proxy_only_reason": "governance_mesh_advises_budgets_evidence_and_operator_packets_without_execution_authority",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(REQUIRED_LABELS),
            "primary_horizon": f"{system_slug}_governance_effect_after_bounded_action",
            "required_context": data_intakes,
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.89,
            "freshness_slo_seconds": 1800,
            "regression_guard_bot_id": contract["anchor_bot_ids"].get(system_slug, ""),
        },
        "data_label_contract_version": LABEL_CONTRACT_VERSION,
        "labeling_tags": [
            "research_only",
            "collection_only",
            "execution_blocked",
            "paper_only_floor",
            f"sleeve_family:{SLEEVE_FAMILY}",
            f"governance_layer:{layer}",
            f"capability_pack:{PACK_SLUG}",
            "autonomic_governance_mesh",
            "point_in_time_only",
            "training_after_threshold",
            "global_halt_aware",
            "pressure_safe",
        ],
        "execution_policy_label": "collection_only_autonomic_governance_no_execution",
        "eligible_for_master_vote": False,
        "founder_bot_id": "brain_refinery_v1",
        "founder_dna_version": "founder_dna_v1",
        "founder_dna_traits": [
            "paper_first_safety",
            "global_halt_awareness",
            "resource_throttle_awareness",
            "decision_explanation_contract",
            "registry_auditable_identity",
            "operator_packet_contract",
            "point_in_time_labeling",
        ],
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "paper_runtime_stability_mode": "thin_digest_autonomic_governance",
        "paper_trade_lock_required": True,
        "capability_pack_version": PACK_VERSION,
        "capability_pack_slug": PACK_SLUG,
        "capability_pack_display_name": PACK_DISPLAY_NAME,
        "autonomic_governance_mesh_version": PACK_VERSION,
        "capability_pack_contract": contract,
        "autonomic_governance_mesh_contract": {
            "contract_version": "autonomic_governance_layers_v1",
            "capability_pack": PACK_SLUG,
            "governance_layer": layer,
            "intelligence_system": system_slug,
            "system_display_name": system["display_name"],
            "system_outputs": list(system["outputs"]),
            "authority_boundary": "collection_only_advisory_no_execution_no_allocation_no_halt_clearance",
            "integration_contract": "feeds_whole_system_governor_quant_operational_intelligence_and_codex_handoff",
        },
    }


def _refresh_summary(registry: dict[str, Any]) -> None:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    active = [row for row in rows if bool(row.get("active"))]
    inactive = [row for row in rows if not bool(row.get("active"))]
    signal_active = [row for row in active if str(row.get("bot_role") or "") == "signal_sub_bot"]
    infra_active = [row for row in active if str(row.get("bot_role") or "") == "infrastructure_sub_bot"]
    structured = [row for row in rows if str(row.get("capability_pack_version") or "")]
    pack_rows = [row for row in rows if str(row.get("autonomic_governance_mesh_version") or "") == PACK_VERSION]
    versions = [
        int(match.group(1))
        for row in rows
        for match in [re.match(r"^brain_refinery_v(\d+)", str(row.get("bot_id") or ""))]
        if match
    ]
    summary = dict(registry.get("summary") or {})
    summary.update(
        {
            "total_bots": len(rows),
            "active_bots": len(active),
            "inactive_bots": len(inactive),
            "active_signal_sub_bots": len(signal_active),
            "active_infrastructure_sub_bots": len(infra_active),
            "data_collection_active_bots": sum(1 for row in rows if bool(row.get("data_collection_active"))),
            "training_excluded_bots": sum(1 for row in rows if bool(row.get("training_excluded"))),
            "structured_capability_pack_bot_count": len(structured),
            "autonomic_governance_mesh_bot_count": len(pack_rows),
            "latest_autonomic_governance_mesh": PACK_VERSION,
            "max_bot_version": max(versions) if versions else None,
            "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
            "target_platform_total_bots_met": len(rows) >= TARGET_PLATFORM_TOTAL_BOTS,
        }
    )
    registry["summary"] = summary


def _pack_summary(assigned_ids: dict[str, str]) -> dict[str, Any]:
    contract = _pack_contract(assigned_ids)
    return {
        "slug": PACK_SLUG,
        "display_name": PACK_DISPLAY_NAME,
        "sleeve_family": SLEEVE_FAMILY,
        "objective": "Add a 56-bot collect-only mesh that makes the whole-system governor self-accommodating across evidence, memory, backlog, sleeve budgets, self-model, and operator handoff.",
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "system_count": len(GOVERNANCE_SYSTEMS),
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "governance_systems": list(GOVERNANCE_SYSTEMS),
        "dedicated_data_intake": list(BASE_DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "anchor_bot_ids": contract["anchor_bot_ids"],
    }


def plan_registry_expansion(registry: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    existing_slot_kinds = {str(row.get("slot_kind") or "") for row in rows}
    assigned_ids = _assign_bot_ids(rows)
    now = _utc_now()
    planned_rows: list[dict[str, Any]] = []
    skipped_existing: list[str] = []
    for bot in BOTS:
        slot = _slot_kind(bot)
        if slot in existing_slot_kinds:
            skipped_existing.append(slot)
            continue
        planned_rows.append(_row_for_bot(bot, assigned_ids[slot], assigned_ids, now))
    return {
        "generated_at_utc": now,
        "autonomic_governance_mesh_version": PACK_VERSION,
        "system_count": len(GOVERNANCE_SYSTEMS),
        "bot_count": len(BOTS),
        "planned_bot_count": len(planned_rows),
        "skipped_existing_count": len(skipped_existing),
        "planned_total_after_apply": len(rows) + len(planned_rows),
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "planned_reaches_target_total": len(rows) + len(planned_rows) >= TARGET_PLATFORM_TOTAL_BOTS,
        "planned_rows": planned_rows,
        "skipped_existing_slot_kinds": skipped_existing,
        "pack": _pack_summary(assigned_ids),
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    registry = _load_json(project_root / "master_bot_registry.json")
    plan = plan_registry_expansion(registry)
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    return {
        "ok": True,
        "generated_at_utc": plan["generated_at_utc"],
        "mode": "dry_run",
        "registry_path": str((project_root / "master_bot_registry.json").resolve()),
        "current_total_bots": len(rows),
        "current_active_bots": sum(1 for row in rows if bool(row.get("active"))),
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "planned_total_after_apply": plan["planned_total_after_apply"],
        "planned_reaches_target_total": plan["planned_reaches_target_total"],
        "autonomic_governance_mesh_version": PACK_VERSION,
        "system_count": plan["system_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "pack": plan["pack"],
        "recommended_apply_command": "./scripts/ops/opsctl.sh autonomic-governance-mesh --apply --json",
        "paired_governor_command": "./scripts/ops/opsctl.sh whole-system-governor --apply --json",
    }


def apply_registry(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    registry_path = project_root / "master_bot_registry.json"
    registry = _load_json(registry_path)
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    plan = plan_registry_expansion(registry)
    added_rows = list(plan["planned_rows"])
    storage_targets_ready = _ensure_storage_targets(project_root)
    backup_path = ""
    if added_rows:
        backup_dir = project_root / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"master_bot_registry_before_autonomic_governance_mesh_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy2(registry_path, backup)
        backup_path = str(backup)
        rows.extend(added_rows)
        registry["sub_bots"] = rows
        registry["updated_at_utc"] = _utc_now()
        _refresh_summary(registry)
        _write_json(registry_path, registry)

    payload = build_payload(project_root)
    payload.update(
        {
            "mode": "applied",
            "added_bot_count": len(added_rows),
            "added_bot_ids": [str(row.get("bot_id") or "") for row in added_rows],
            "backup_path": backup_path,
            "new_total_bots": len(rows),
            "new_active_bots": sum(1 for row in rows if bool(row.get("active"))),
            "target_platform_total_bots_met": len(rows) >= TARGET_PLATFORM_TOTAL_BOTS,
            "storage_targets_ready": storage_targets_ready,
        }
    )
    _write_json(
        project_root / "config" / "autonomic_governance_mesh_v1.json",
        {
            "generated_at_utc": _utc_now(),
            "autonomic_governance_mesh_version": PACK_VERSION,
            "pack": payload["pack"],
        },
    )
    _write_json(project_root / "governance" / "health" / "autonomic_governance_mesh_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add the 56-bot autonomic governance mesh collect-only pack.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = apply_registry(project_root) if args.apply else build_payload(project_root)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "autonomic_governance_mesh "
            f"mode={payload['mode']} systems={payload['system_count']} "
            f"bots={payload['bot_count']} planned={payload['planned_bot_count']} "
            f"added={payload.get('added_bot_count', 0)} "
            f"target_total={payload.get('planned_total_after_apply') or payload.get('new_total_bots')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
