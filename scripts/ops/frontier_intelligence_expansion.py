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
BASE_VERSION = 1038
TARGET_PLATFORM_TOTAL_BOTS = 1086
PACK_VERSION = "frontier_intelligence_v1"
PACK_SLUG = "frontier_intelligence"
PACK_DISPLAY_NAME = "Frontier Intelligence Pack"
SLEEVE_FAMILY = "advanced_intelligence_control_plane"
SLEEVE_PROFILE = "frontier_intelligence"
LABEL_CONTRACT_VERSION = "frontier_intelligence_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 48000
MINIMUM_COLLECTION_DAYS = 180
PAPER_RUNTIME_CAPACITY_FLOOR = 1000

SYSTEMS: list[dict[str, Any]] = [
    {
        "slug": "counterfactual_causal_lab",
        "display_name": "Counterfactual Causal Lab",
        "objective": "Model what would have happened under alternate runtime, feed, queue, halt, and expansion choices.",
        "outputs": ["counterfactual_trace", "causal_intervention_rank", "blocked_action_explanation"],
    },
    {
        "slug": "hierarchical_memory_retrieval",
        "display_name": "Hierarchical Memory Retrieval",
        "objective": "Route hot reflex, experience, incident, and long-horizon memories into the right decision context.",
        "outputs": ["memory_route_packet", "stale_advice_flag", "lesson_reuse_score"],
    },
    {
        "slug": "multi_agent_debate_critique",
        "display_name": "Multi-Agent Debate And Critique",
        "objective": "Run structured pressure, data, safety, alpha, execution, and operator-headroom debates before major actions.",
        "outputs": ["debate_vote_packet", "critic_disagreement_score", "operator_review_reason"],
    },
    {
        "slug": "uncertainty_risk_calibration",
        "display_name": "Uncertainty And Risk Calibration",
        "objective": "Calibrate confidence, tail risk, false-positive halt risk, and data-quality uncertainty before promotion or expansion.",
        "outputs": ["uncertainty_score", "tail_risk_bucket", "confidence_discount"],
    },
    {
        "slug": "active_learning_data_value",
        "display_name": "Active Learning Data Value",
        "objective": "Ask for the highest-information observations instead of collecting more raw volume when pressure is high.",
        "outputs": ["next_best_observation", "label_gap_rank", "data_value_score"],
    },
    {
        "slug": "alpha_thesis_factory",
        "display_name": "Alpha Thesis Factory",
        "objective": "Convert research themes into testable, non-duplicative, gated alpha hypotheses across sleeves.",
        "outputs": ["alpha_thesis_card", "novelty_review_packet", "thesis_retirement_reason"],
    },
    {
        "slug": "execution_microstructure_sandbox",
        "display_name": "Execution Microstructure Sandbox",
        "objective": "Stress paper trading with queue position, VPIN, LOB imbalance, spread, latency, and partial-fill realism.",
        "outputs": ["microstructure_fill_surface", "toxicity_discount", "paper_realism_adjustment"],
    },
    {
        "slug": "macro_event_world_model",
        "display_name": "Macro Event World Model",
        "objective": "Anticipate macro, Fed, Treasury, CPI, labor, funding, and geopolitical event pressure across sleeves.",
        "outputs": ["event_pressure_forecast", "macro_priming_packet", "calendar_risk_vote"],
    },
    {
        "slug": "resource_allocation_market",
        "display_name": "Resource Allocation Market",
        "objective": "Treat compute, memory, SQL writer time, report rendering, training, and feed attention as budgeted markets.",
        "outputs": ["resource_bid_packet", "capacity_price", "duty_cycle_vote"],
    },
    {
        "slug": "formal_safety_verification",
        "display_name": "Formal Safety Verification",
        "objective": "Keep live execution, credentials, halt clearing, cleanup, and training behind explicit safety invariants.",
        "outputs": ["safety_invariant_check", "blocked_action_proof", "regression_guard_spec"],
    },
    {
        "slug": "bot_genome_lineage_evolution",
        "display_name": "Bot Genome Lineage Evolution",
        "objective": "Track founder DNA, pack inheritance, sleeve lineage, and mutation safety across every new bot wave.",
        "outputs": ["lineage_audit", "mutation_budget", "founder_dna_revalidation"],
    },
    {
        "slug": "operator_copilot_intent_bridge",
        "display_name": "Operator Copilot Intent Bridge",
        "objective": "Translate operator goals into safe command sequences, calm-mode decisions, and report-ready explanations.",
        "outputs": ["operator_intent_state", "safe_command_sequence", "narrative_packet"],
    },
]

ROLE_TEMPLATES: list[dict[str, Any]] = [
    {"suffix": "state_builder", "label": "State Builder", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
    {"suffix": "forecast_scorer", "label": "Forecast Scorer", "bot_role": "signal_sub_bot", "priority": "high"},
    {"suffix": "policy_guard", "label": "Policy Guard", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
    {"suffix": "evidence_writer", "label": "Evidence Writer", "bot_role": "infrastructure_sub_bot", "priority": "high"},
]

DATA_INTAKES = [
    "frontier_counterfactual_trace",
    "frontier_intervention_result_trace",
    "frontier_hierarchical_memory_trace",
    "frontier_debate_vote_trace",
    "frontier_uncertainty_calibration_trace",
    "frontier_data_value_trace",
    "frontier_alpha_thesis_trace",
    "frontier_microstructure_sandbox_trace",
    "frontier_macro_event_world_trace",
    "frontier_resource_market_trace",
    "frontier_safety_verification_trace",
    "frontier_lineage_evolution_trace",
    "frontier_operator_intent_trace",
]

STORAGE_TARGETS = [
    "governance/frontier_intelligence",
    "governance/frontier_intelligence/counterfactual_causal_lab",
    "governance/frontier_intelligence/hierarchical_memory_retrieval",
    "governance/frontier_intelligence/multi_agent_debate_critique",
    "governance/frontier_intelligence/uncertainty_risk_calibration",
    "governance/frontier_intelligence/active_learning_data_value",
    "governance/frontier_intelligence/alpha_thesis_factory",
    "governance/frontier_intelligence/execution_microstructure_sandbox",
    "governance/frontier_intelligence/macro_event_world_model",
    "governance/frontier_intelligence/resource_allocation_market",
    "governance/frontier_intelligence/formal_safety_verification",
    "governance/frontier_intelligence/bot_genome_lineage_evolution",
    "governance/frontier_intelligence/operator_copilot_intent_bridge",
    "governance/health/frontier_intelligence_latest.json",
]

REQUIRED_LABELS = [
    "counterfactual_decision_bucket",
    "intervention_expected_payoff_bucket",
    "memory_route_quality_bucket",
    "debate_consensus_status",
    "uncertainty_calibration_bucket",
    "data_value_rank_bucket",
    "alpha_thesis_novelty_bucket",
    "microstructure_realism_bucket",
    "macro_event_pressure_bucket",
    "resource_bid_priority_bucket",
    "safety_invariant_status",
    "lineage_mutation_budget_bucket",
    "operator_intent_mode",
]


def _bot_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for role in ROLE_TEMPLATES:
            role_slug = f"{system['slug']}_{role['suffix']}"
            specs.append(
                {
                    "role_slug": role_slug,
                    "slug": f"frontier_{role_slug}_bot",
                    "label": f"{system['display_name']} {role['label']}",
                    "system": system["slug"],
                    "bot_role": role["bot_role"],
                    "priority": role["priority"],
                    "objective": f"{role['label']} for {system['objective']}",
                    "target_functions": list(system.get("outputs", [])),
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
        version = desired if desired not in used_versions else _next_available_version(used_versions, max(max(used_versions, default=BASE_VERSION - 1) + 1, desired))
        used_versions.add(version)
        assigned[slot] = f"brain_refinery_v{version}_{bot['slug']}"
    return assigned


def _system(bot: dict[str, Any]) -> dict[str, Any]:
    for system in SYSTEMS:
        if system["slug"] == bot["system"]:
            return system
    return {"slug": bot["system"], "display_name": bot["system"], "objective": bot["objective"], "outputs": []}


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
        "new_sleeve_or_subsleeve": {
            "sleeve_family": SLEEVE_FAMILY,
            "sleeve_profile": SLEEVE_PROFILE,
            "display_name": PACK_DISPLAY_NAME,
        },
        "system_count": len(SYSTEMS),
        "bot_count": len(BOTS),
        "bot_pack_size_rule": "48_bots_frontier_intelligence_layer_after_deep_recursive_awareness",
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "frontier_intelligence_hot_7d_warm_150d_cold_720d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": 10,
            "capture_mode": "sampled_digest_first_frontier_trace",
            "sample_rate": 0.08,
            "dedupe_required": True,
            "stale_deletion_policy": "retain_decision_digests_stage_raw_frontier_traces",
        },
        "paper_only_floor": {
            "paper_trade_lock_required": True,
            "live_trading_enabled": False,
            "execution_enabled": False,
            "allocation_enabled": False,
            "graduation_requires_minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "graduation_requires_collection_days": MINIMUM_COLLECTION_DAYS,
        },
        "anchor_bot_ids": {
            bot["system"]: assigned_ids.get(_slot_kind(bot), "")
            for bot in BOTS
            if bot["role_slug"].endswith("state_builder")
        },
        "runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "intelligence_advancements": [system["slug"] for system in SYSTEMS],
        "global_halt_contract": "frontier_intelligence_can_explain_preempt_and_rehearse_halt_paths_but_never_force_clear",
        "paper_lock_contract": "no_execution_no_allocation_no_training_until_180_days_48000_observations_and_v6_gates_clear",
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    system = _system(bot)
    contract = _pack_contract(assigned_ids)
    frontier_contract = {
        "contract_version": "frontier_intelligence_layers_v1",
        "capability_pack": PACK_SLUG,
        "system": bot["system"],
        "system_display_name": system["display_name"],
        "system_outputs": list(system.get("outputs", [])),
        "platform_brain_v6_dependency": "platform_brain_v6_foresight_cortex",
        "frontier_boundary": "advanced_advisory_self_modeling_collect_only_no_consciousness_claim_no_execution_authority",
    }
    return {
        "bot_id": bot_id,
        "bot_role": bot["bot_role"],
        "active": True,
        "reason": "frontier_intelligence_expansion_slot",
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
        "promotion_reason": "frontier_intelligence_expansion_slot",
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
            "rapid_expansion",
            "global_halt_recovery",
            "high_uncertainty",
            "macro_event_window",
            "backpressure_spike",
            "paper_realism_review",
            "operator_headroom_mode",
        ],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v964_apex_self_model_state_vector_builder_bot",
            "brain_refinery_v1010_recursive_awareness_causal_incident_root_cause_builder_bot",
            "brain_refinery_v1014_recursive_awareness_runtime_pressure_forecaster_bot",
            "brain_refinery_v1030_recursive_awareness_overconfidence_challenge_board_bot",
        ],
        "data_intake_collections": list(DATA_INTAKES),
        "storage_targets": list(STORAGE_TARGETS),
        "freshness_slo_seconds": 900,
        "retention_profile": "frontier_intelligence_hot_7d_warm_150d_cold_720d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "frontier_intelligence_collect_only_until_v6_foresight_memory_debate_uncertainty_and_safety_gates_clear",
        "trading_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "allocation_enabled": False,
        "execution_enabled": False,
        "rotation_blocked": True,
        "rotation_block_reason": "frontier_intelligence_collection_only_zero_weight",
        "training_excluded": True,
        "exclude_from_training": True,
        "training_candidate_after_threshold": True,
        "training_exclusion_reason": "collecting_frontier_intelligence_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "training_threshold_policy": {
            "minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "minimum_collection_days": MINIMUM_COLLECTION_DAYS,
            "requires_platform_brain_v6_clearance": True,
            "requires_counterfactual_quality_clearance": True,
            "requires_uncertainty_calibration_clearance": True,
            "requires_formal_safety_clearance": True,
            "requires_data_value_clearance": True,
        },
        "data_collection_storage_guarded": True,
        "data_collection_capture_mode": "sampled",
        "data_collection_sample_rate": 0.08,
        "data_collection_max_daily_storage_mb": 10,
        "data_collection_max_daily_mb": 10.0,
        "data_collection_compute_guard_mode": "thin_digest",
        "data_collection_resource_guard_reason": "frontier_intelligence_uses_digest_first_traces_for_1000_plus_bot_platform",
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "paper_execution_queue_policy": "buffered_jsonl_batching",
        "paper_runtime_control_refresh_seconds": 240,
        "sleeve_profile": SLEEVE_PROFILE,
        "sleeve_family": SLEEVE_FAMILY,
        "correlation_peer_sleeves": [
            "deep_recursive_awareness",
            "apex_self_awareness_intelligence",
            "adaptive_intelligence_kernel",
            "intelligence_layer_advancement",
            "coordination_intelligence",
        ],
        "correlation_dependencies": [
            "platform_brain_v6",
            "platform_brain_v5",
            "platform_stabilization_quality",
            "system_self_model",
            "runtime_throttle",
            "memory_efficiency",
            "backpressure_drainer_fleet",
        ],
        "provider_capability_profile": "internal_frontier_intelligence_control_plane_only",
        "direct_market_data_available": False,
        "direct_execution_allowed": False,
        "proxy_data_sources": [
            "master_bot_registry",
            "governance_health",
            "platform_brain_v6",
            "platform_brain_v5",
            "deep_recursive_awareness",
            "system_self_model",
            "decision_provenance",
        ],
        "schwab_direct_inputs": [],
        "proxy_only_reason": "frontier_intelligence_has_no_direct_broker_dependency_or_execution_authority",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(REQUIRED_LABELS),
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.84,
            "freshness_slo_seconds": 900,
            "regression_guard_bot_id": contract["anchor_bot_ids"].get("formal_safety_verification", ""),
        },
        "data_label_contract_version": LABEL_CONTRACT_VERSION,
        "labeling_tags": [
            "research_only",
            "collection_only",
            "execution_blocked",
            "paper_only_floor",
            f"sleeve_family:{SLEEVE_FAMILY}",
            f"sleeve_profile:{SLEEVE_PROFILE}",
            f"capability_pack:{PACK_SLUG}",
            f"frontier_system:{bot['system']}",
            "frontier_intelligence",
            "platform_brain_v6_aware",
            "counterfactual_reasoning",
            "uncertainty_calibrated",
            "training_after_threshold",
            "global_halt_aware",
            "operator_context_aware",
            "1000_plus_bot_platform",
        ],
        "execution_policy_label": "collection_only_frontier_intelligence_no_execution",
        "eligible_for_master_vote": False,
        "founder_bot_id": "brain_refinery_v1",
        "founder_dna_version": "founder_dna_v1",
        "founder_dna_traits": [
            "market_data_observation",
            "paper_first_safety",
            "global_halt_awareness",
            "resource_throttle_awareness",
            "decision_explanation_contract",
            "registry_auditable_identity",
            "system_self_model_awareness",
            "deep_recursive_awareness",
            "frontier_counterfactual_reasoning",
            "formal_safety_boundary",
        ],
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "paper_runtime_stability_mode": "digest_first_frontier_intelligence",
        "paper_trade_lock_required": True,
        "paper_runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "capability_pack_version": PACK_VERSION,
        "capability_pack_slug": PACK_SLUG,
        "capability_pack_display_name": PACK_DISPLAY_NAME,
        "frontier_intelligence_version": PACK_VERSION,
        "capability_pack_contract": contract,
        "frontier_intelligence_contract": frontier_contract,
    }


def _refresh_summary(registry: dict[str, Any]) -> None:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    active = [row for row in rows if bool(row.get("active"))]
    inactive = [row for row in rows if not bool(row.get("active"))]
    signal_active = [row for row in active if str(row.get("bot_role") or "") == "signal_sub_bot"]
    infra_active = [row for row in active if str(row.get("bot_role") or "") == "infrastructure_sub_bot"]
    structured = [row for row in rows if str(row.get("capability_pack_version") or "")]
    pack_rows = [row for row in rows if str(row.get("frontier_intelligence_version") or "") == PACK_VERSION]
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
            "frontier_intelligence_bot_count": len(pack_rows),
            "latest_frontier_intelligence": PACK_VERSION,
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
        "sleeve_profile": SLEEVE_PROFILE,
        "objective": "Add a larger frontier intelligence layer for counterfactuals, memory routing, debate, uncertainty, active learning, alpha thesis design, microstructure realism, macro anticipation, resource markets, formal safety, lineage, and operator copilot intent.",
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "system_count": len(SYSTEMS),
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "systems": list(SYSTEMS),
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "runtime_capacity_floor": contract["runtime_capacity_floor"],
        "intelligence_advancements": list(contract["intelligence_advancements"]),
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
        "frontier_intelligence_version": PACK_VERSION,
        "system_count": len(SYSTEMS),
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
        "frontier_intelligence_version": PACK_VERSION,
        "system_count": plan["system_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "pack": plan["pack"],
        "recommended_apply_command": "./scripts/ops/opsctl.sh frontier-intelligence --apply --json",
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
        backup = backup_dir / f"master_bot_registry_before_frontier_intelligence_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
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
        project_root / "config" / "frontier_intelligence_v1.json",
        {"generated_at_utc": _utc_now(), "frontier_intelligence_version": PACK_VERSION, "pack": payload["pack"]},
    )
    _write_json(project_root / "governance" / "health" / "frontier_intelligence_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add the 48-bot frontier intelligence collect-only control-plane pack.")
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
            "frontier_intelligence "
            f"mode={payload['mode']} systems={payload['system_count']} "
            f"bots={payload['bot_count']} planned={payload['planned_bot_count']} "
            f"added={payload.get('added_bot_count', 0)} "
            f"target_total={payload.get('planned_total_after_apply') or payload.get('new_total_bots')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
