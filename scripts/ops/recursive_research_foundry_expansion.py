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
BASE_VERSION = 834
FOUNDRY_VERSION = "recursive_research_foundry_v1"
LABEL_CONTRACT_VERSION = "recursive_foundry_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 12000
MINIMUM_COLLECTION_DAYS = 45
PAPER_RUNTIME_CAPACITY_FLOOR = 700

FOUNDRY_SLUG = "recursive_research_foundry"
FOUNDRY_DISPLAY_NAME = "Recursive Research Foundry"
FOUNDRY_FAMILY = "recursive_research"
FOUNDRY_PROFILE = "recursive_research_foundry"

DATA_INTAKES = [
    "literature_to_hypothesis_trace",
    "theorem_proof_obligation_trace",
    "symbolic_factor_grammar_trace",
    "market_ontology_evolution_trace",
    "self_supervised_representation_trace",
    "synthetic_market_validity_trace",
    "agent_architecture_search_trace",
    "causal_mechanism_discovery_trace",
    "adversarial_red_team_scenario_trace",
    "teacher_student_distillation_trace",
    "micro_experiment_design_trace",
    "benchmark_replay_suite_trace",
    "proof_report_narrative_trace",
    "research_compute_budget_trace",
    "grandmaster_foundry_packet_trace",
]

STORAGE_TARGETS = [
    "governance/recursive_research_foundry",
    "governance/recursive_research_foundry/hypotheses",
    "governance/recursive_research_foundry/proofs",
    "governance/recursive_research_foundry/benchmarks",
    "governance/health/recursive_research_foundry",
    "data/jsonl_link.sqlite3",
]

REQUIRED_LABELS = [
    "hypothesis_novelty_grade",
    "proof_obligation_status",
    "factor_grammar_family",
    "synthetic_validity_grade",
    "architecture_search_status",
    "red_team_severity_bucket",
    "benchmark_replay_integrity",
]

FOUNDRY_CONTRACT = {
    "contract_version": FOUNDRY_VERSION,
    "purpose": "recursive_research_generation_validation_and_distillation_before_strategy_admission",
    "research_depth": [
        "literature_to_hypothesis_compilation",
        "proof_obligation_generation",
        "symbolic_factor_grammar_mining",
        "market_ontology_evolution",
        "self_supervised_representation_learning",
        "synthetic_market_validation",
        "agent_architecture_search",
        "causal_mechanism_discovery",
        "adversarial_red_team_generation",
        "teacher_student_distillation",
        "micro_experiment_design",
        "benchmark_replay_curation",
        "proof_to_report_narration",
        "research_compute_budgeting",
        "grandmaster_foundry_bridge",
    ],
    "graduation_policy": "collection_only_until_replay_proof_distillation_and_research_debt_checks_clear",
    "resource_contract": "foundry_runs_cold_lane_only_and_distills_raw_research_traces",
    "global_halt_contract": "foundry_must_degrade_to_digest_only_before_hard_halt",
    "paper_lock_contract": "no_execution_path_and_no_paper_trade_until_admission_guard_promotes",
}

BOTS: list[dict[str, Any]] = [
    ("literature_hypothesis_compiler", "recursive_foundry_literature_hypothesis_compiler_bot", "Recursive Foundry Literature Hypothesis Compiler", "infrastructure_sub_bot", "hypothesis_compilation", "critical", "Compile papers, reports, and prior notes into traceable research hypotheses.", ["research_pipeline", "hypothesis_retrieval_rag", "source_verification"]),
    ("proof_obligation_generator", "recursive_foundry_proof_obligation_generator_bot", "Recursive Foundry Proof Obligation Generator", "infrastructure_sub_bot", "proof_generation", "critical", "Turn research candidates into explicit safety, data, and replay proof obligations.", ["formal_verification", "safety_case_builder", "regression_guard"]),
    ("symbolic_factor_grammar_miner", "recursive_foundry_symbolic_factor_grammar_miner_bot", "Recursive Foundry Symbolic Factor Grammar Miner", "signal_sub_bot", "symbolic_factor_mining", "high", "Mine reusable symbolic factor grammars from successful and failed sleeve behavior.", ["neuro_symbolic_rule_bridge", "feature_store", "strategy_coverage"]),
    ("market_ontology_evolution_curator", "recursive_foundry_market_ontology_evolution_curator_bot", "Recursive Foundry Market Ontology Evolution Curator", "infrastructure_sub_bot", "ontology_evolution", "high", "Curate ontology changes so new sleeves inherit clean labels and shared meaning.", ["semantic_feature_ontology", "label_taxonomy", "commands_hygiene"]),
    ("self_supervised_representation_scout", "recursive_foundry_self_supervised_representation_scout_bot", "Recursive Foundry Self-Supervised Representation Scout", "signal_sub_bot", "representation_learning", "high", "Scout representation-learning candidates using unlabeled traces without touching training lanes.", ["model_lifecycle", "feature_quality", "research_automation"]),
    ("synthetic_market_validity_auditor", "recursive_foundry_synthetic_market_validity_auditor_bot", "Recursive Foundry Synthetic Market Validity Auditor", "infrastructure_sub_bot", "synthetic_validation", "critical", "Audit synthetic market generators for realism, leakage, and regime coverage.", ["stress_lab", "golden_replay_guard", "model_lifecycle"]),
    ("agent_architecture_search_planner", "recursive_foundry_agent_architecture_search_planner_bot", "Recursive Foundry Agent Architecture Search Planner", "signal_sub_bot", "architecture_search", "high", "Plan agent architecture experiments under compute, memory, and evidence constraints.", ["cognitive_control_plane", "mlx_batch_size_runtime_governor", "experiment_ledger"]),
    ("causal_mechanism_discovery_adjudicator", "recursive_foundry_causal_mechanism_discovery_adjudicator_bot", "Recursive Foundry Causal Mechanism Discovery Adjudicator", "signal_sub_bot", "causal_mechanism_discovery", "critical", "Adjudicate candidate mechanisms before correlation-only ideas become strategies.", ["causal_counterfactual_evidence", "multiple_testing_guard", "research_pipeline"]),
    ("adversarial_red_team_scenario_crafter", "recursive_foundry_adversarial_red_team_scenario_crafter_bot", "Recursive Foundry Adversarial Red-Team Scenario Crafter", "infrastructure_sub_bot", "red_team_generation", "critical", "Craft hostile scenarios for data sources, labels, execution assumptions, and model brittleness.", ["adversarial_robustness_guard", "source_verification", "stress_lab"]),
    ("teacher_student_distillation_steward", "recursive_foundry_teacher_student_distillation_steward_bot", "Recursive Foundry Teacher Student Distillation Steward", "infrastructure_sub_bot", "distillation", "high", "Distill high-value lessons from mature bots into safe student-bot evidence packets.", ["teacher_quality", "bot_quality_autopilot", "model_lifecycle"]),
    ("micro_experiment_designer", "recursive_foundry_micro_experiment_designer_bot", "Recursive Foundry Micro Experiment Designer", "signal_sub_bot", "micro_experiment_design", "high", "Design tiny, bounded experiments that answer one research question without runtime churn.", ["experiment_ledger", "active_learning_query_planner", "coverage_gap_closer"]),
    ("benchmark_replay_suite_curator", "recursive_foundry_benchmark_replay_suite_curator_bot", "Recursive Foundry Benchmark Replay Suite Curator", "infrastructure_sub_bot", "benchmark_curation", "critical", "Curate benchmark replays that every future strategy pack must survive.", ["replay_hash_registry", "golden_replay_guard", "stress_lab"]),
    ("proof_report_narrator", "recursive_foundry_proof_report_narrator_bot", "Recursive Foundry Proof Report Narrator", "infrastructure_sub_bot", "proof_reporting", "medium", "Translate proof, replay, and research evidence into concise report-ready narratives.", ["reporting_layer", "system_summary", "decision_provenance"]),
    ("research_compute_budget_sentinel", "recursive_foundry_research_compute_budget_sentinel_bot", "Recursive Foundry Research Compute Budget Sentinel", "infrastructure_sub_bot", "compute_budgeting", "critical", "Throttle research foundry work around swap, CPU, storage, and live runtime pressure.", ["runtime_throttle", "memory_efficiency", "swap_pressure_governor", "ingestion_storage_control"]),
    ("grandmaster_foundry_bridge", "recursive_foundry_grandmaster_foundry_bridge_bot", "Recursive Foundry Grandmaster Foundry Bridge", "infrastructure_sub_bot", "grandmaster_foundry_bridge", "critical", "Package foundry outputs into grandmaster-ready research admission decisions.", ["grand_master_reporting", "platform_control_plane", "operator_cockpit"]),
]
BOTS = [
    {
        "role_slug": role_slug,
        "slug": slug,
        "label": label,
        "bot_role": bot_role,
        "foundry_layer": layer,
        "priority": priority,
        "objective": objective,
        "target_functions": targets,
    }
    for role_slug, slug, label, bot_role, layer, priority, objective, targets in BOTS
]


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


def _version_from_bot_id(bot_id: str) -> int | None:
    match = re.match(r"^brain_refinery_v(?P<version>\d+)", bot_id)
    return int(match.group("version")) if match else None


def _slot_kind(bot: dict[str, Any]) -> str:
    return f"{FOUNDRY_SLUG}_{bot['role_slug']}"


def _next_available_version(used_versions: set[int], start: int) -> int:
    version = start
    while version in used_versions:
        version += 1
    used_versions.add(version)
    return version


def _assign_bot_ids(rows: list[dict[str, Any]]) -> dict[str, str]:
    existing_by_slot_kind = {
        str(row.get("slot_kind") or ""): str(row.get("bot_id") or "")
        for row in rows
        if isinstance(row, dict) and str(row.get("slot_kind") or "") and str(row.get("bot_id") or "")
    }
    used_versions = {
        version
        for row in rows
        if isinstance(row, dict)
        for version in [_version_from_bot_id(str(row.get("bot_id") or ""))]
        if version is not None
    }
    assigned: dict[str, str] = {}
    for index, bot in enumerate(BOTS):
        slot = _slot_kind(bot)
        if slot in existing_by_slot_kind:
            assigned[slot] = existing_by_slot_kind[slot]
            continue
        desired = BASE_VERSION + index
        version = desired if desired not in used_versions else _next_available_version(used_versions, max(max(used_versions, default=BASE_VERSION - 1) + 1, desired))
        used_versions.add(version)
        assigned[slot] = f"brain_refinery_v{version}_{bot['slug']}"
    return assigned


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


def _foundry_contract(assigned_ids: dict[str, str]) -> dict[str, Any]:
    return {
        "contract_version": FOUNDRY_VERSION,
        "new_sleeve_or_subsleeve": {
            "sleeve_family": FOUNDRY_FAMILY,
            "sleeve_profile": FOUNDRY_PROFILE,
            "display_name": FOUNDRY_DISPLAY_NAME,
        },
        "bot_pack_size": len(BOTS),
        "bot_pack_size_rule": "5_to_15_bots_max",
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "foundry_hot_7d_warm_60d_cold_365d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": 35,
            "capture_mode": "sampled_digest_first",
            "sample_rate": 0.2,
            "dedupe_required": True,
            "stale_deletion_policy": "distill_research_digest_then_stage_low_value_raw_traces",
        },
        "paper_only_floor": {
            "paper_trade_lock_required": True,
            "live_trading_enabled": False,
            "execution_enabled": False,
            "allocation_enabled": False,
            "graduation_requires_minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "graduation_requires_collection_days": MINIMUM_COLLECTION_DAYS,
        },
        "sleeve_master_bot_id": assigned_ids.get(f"{FOUNDRY_SLUG}_literature_hypothesis_compiler", ""),
        "regression_guard_bot_id": assigned_ids.get(f"{FOUNDRY_SLUG}_proof_obligation_generator", ""),
        "capacity_check": {
            "active_bot_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
            "runtime_stability_mode": "full_force_buffered",
            "runtime_control_refresh_seconds": 240,
            "queue_policy": "buffered_jsonl_batching",
            "compute_guard_mode": "sustain",
            "global_halt_mode": "digest_only_before_hard_halt",
            "heavy_reasoning_mode": "cold_lane_digest_only_until_host_pressure_clear",
        },
        "recursive_foundry_contract": FOUNDRY_CONTRACT,
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    contract = _foundry_contract(assigned_ids)
    advanced_contract = {
        "contract_version": "advanced_intelligence_layers_v5",
        "recursive_foundry_version": FOUNDRY_VERSION,
        "capability_pack": FOUNDRY_SLUG,
        "bot_intelligence_layer": bot["foundry_layer"],
        "reports_to_sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "critic_guard_bot_id": contract["regression_guard_bot_id"],
        "hypothesis_compilation": "literature_to_hypothesis_trace",
        "proof_obligations": "theorem_proof_obligation_trace",
        "symbolic_grammar": "symbolic_factor_grammar_trace",
        "ontology": "market_ontology_evolution_trace",
        "synthetic_validation": "synthetic_market_validity_trace",
        "red_team": "adversarial_red_team_scenario_trace",
        "distillation": "teacher_student_distillation_trace",
        "benchmarking": "benchmark_replay_suite_trace",
        "resource_budget": "research_compute_budget_trace",
    }
    return {
        "bot_id": bot_id,
        "bot_role": bot["bot_role"],
        "active": True,
        "reason": "planned_roster_expansion_slot",
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
        "promotion_reason": "planned_roster_expansion_slot",
        "model_path": "",
        "log_file": "",
        "candidate_log_file": "",
        "lifecycle_state": "data_collection_only",
        "slot_label": bot["label"],
        "slot_kind": _slot_kind(bot),
        "slot_priority": bot["priority"],
        "slot_objective": bot["objective"],
        "target_functions": list(bot["target_functions"]),
        "preferred_regimes": ["research_cycle", "low_pressure", "off_hours", "model_uncertainty", "regime_shift"],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v819_cognitive_planning_orchestrator_master_bot",
            "brain_refinery_v822_cognitive_formal_objective_alignment_guard_bot",
            "brain_refinery_v833_cognitive_grandmaster_cognition_bridge_bot",
        ],
        "data_intake_collections": list(DATA_INTAKES),
        "storage_targets": list(STORAGE_TARGETS),
        "freshness_slo_seconds": 1800,
        "retention_profile": "foundry_hot_7d_warm_60d_cold_365d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "recursive_foundry_digest_observer_until_replay_proof_thresholds_clear",
        "trading_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "allocation_enabled": False,
        "execution_enabled": False,
        "rotation_blocked": True,
        "rotation_block_reason": "data_collection_only_zero_weight",
        "training_excluded": True,
        "exclude_from_training": True,
        "training_candidate_after_threshold": True,
        "training_exclusion_reason": "collecting_foundry_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "training_threshold_policy": "eligible_when_observations_days_replay_proof_and_research_debt_checks_clear",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "promotion_blocked_until": "minimum_data_collection_threshold_met",
        "promotion_block_reason": "recursive_foundry_collection_only_no_training_yet",
        "data_collection_storage_guarded": True,
        "data_collection_storage_guard_mode": "normal",
        "data_collection_capture_mode": "sampled_digest_first",
        "data_collection_sample_rate": 0.2,
        "data_collection_max_daily_storage_mb": 35,
        "data_collection_storage_guard_updated_utc": now,
        "storage_pressure_capture_reason": "recursive_foundry_distilled_cold_lane_collection",
        "data_collection_compute_guard_mode": "sustain",
        "data_collection_resource_guard_reason": "foundry_heavy_reasoning_cold_lane_only",
        "data_collection_max_daily_mb": 35,
        "collected_observation_count": 0,
        "data_collection_last_counted_utc": "",
        "data_collection_observation_rollup_source": "awaiting_first_incremental_rollup",
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "sleeve_profile": FOUNDRY_PROFILE,
        "sleeve_family": FOUNDRY_FAMILY,
        "correlation_peer_sleeves": [
            "cognitive_control_plane",
            "advanced_intelligence_mesh",
            "research_automation",
            "model_lifecycle",
            "stress_lab",
        ],
        "correlation_dependencies": [
            "research_pipeline",
            "formal_verification",
            "experiment_ledger",
            "golden_replay_guard",
            "runtime_throttle",
        ],
        "provider_capability_profile": "internal_recursive_research_only",
        "direct_market_data_available": False,
        "direct_execution_allowed": False,
        "proxy_data_sources": [
            "research_pipeline",
            "experiment_ledger",
            "decision_provenance",
            "cognitive_control_plane",
            "advanced_intelligence_mesh",
            "stress_replay_artifacts",
        ],
        "schwab_direct_inputs": [],
        "proxy_only_reason": "recursive_foundry_has_no_direct_execution_or_broker_dependency",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(REQUIRED_LABELS),
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.72,
            "freshness_slo_seconds": 1800,
            "regression_guard_bot_id": contract["regression_guard_bot_id"],
        },
        "data_label_contract_version": LABEL_CONTRACT_VERSION,
        "labeling_tags": [
            "research_only",
            "collection_only",
            "execution_blocked",
            "paper_only_floor",
            f"sleeve_family:{FOUNDRY_FAMILY}",
            f"sleeve_profile:{FOUNDRY_PROFILE}",
            f"capability_pack:{FOUNDRY_SLUG}",
            f"foundry_layer:{bot['foundry_layer']}",
            f"data_floor:{MINIMUM_TRAINING_OBSERVATIONS}",
            "training_after_threshold",
            "global_halt_aware",
            "recursive_research",
        ],
        "execution_policy_label": "collection_only_recursive_foundry_no_execution",
        "eligible_for_master_vote": False,
        "data_collection_runtime_dependency_profile": "cold_lane_digest_first_sampled",
        "founder_bot_id": "brain_refinery_v1",
        "founder_dna_version": "founder_dna_v1",
        "founder_dna_scope": "brain_refinery_v1_to_full_fleet",
        "founder_dna_source": "registry_inferred_full_fleet_contract",
        "founder_dna_confidence": 0.8,
        "founder_dna_traits": [
            "market_data_observation",
            "paper_first_safety",
            "global_halt_awareness",
            "resource_throttle_awareness",
            "training_lineage",
            "decision_explanation_contract",
            "data_collection_before_training",
            "registry_auditable_identity",
            "recursive_research_governance",
        ],
        "founder_dna_inheritance_mode": "explicit_contract_metadata",
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "lineage_regression_guard": "fail_if_founder_dna_missing_or_stale",
        "lineage_generation": 5,
        "paper_runtime_stability_mode": "full_force_buffered",
        "paper_execution_queue_policy": "buffered_jsonl_batching",
        "paper_runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "paper_trade_lock_required": True,
        "paper_runtime_control_refresh_seconds": 240,
        "capability_pack_version": FOUNDRY_VERSION,
        "capability_pack_slug": FOUNDRY_SLUG,
        "capability_pack_display_name": FOUNDRY_DISPLAY_NAME,
        "recursive_foundry_version": FOUNDRY_VERSION,
        "capability_pack_contract": contract,
        "advanced_intelligence_layer_contract": advanced_contract,
    }


def _refresh_summary(registry: dict[str, Any]) -> None:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    active = [row for row in rows if bool(row.get("active"))]
    inactive = [row for row in rows if not bool(row.get("active"))]
    summary = dict(registry.get("summary") or {})
    summary.update(
        {
            "total_bots": len(rows),
            "active_bots": len(active),
            "inactive_bots": len(inactive),
            "active_signal_sub_bots": sum(1 for row in active if str(row.get("bot_role") or "") == "signal_sub_bot"),
            "active_infrastructure_sub_bots": sum(1 for row in active if str(row.get("bot_role") or "") == "infrastructure_sub_bot"),
            "inactive_signal_sub_bots": sum(1 for row in inactive if str(row.get("bot_role") or "") == "signal_sub_bot"),
            "inactive_infrastructure_sub_bots": sum(1 for row in inactive if str(row.get("bot_role") or "") == "infrastructure_sub_bot"),
            "data_collection_active_bots": sum(1 for row in rows if bool(row.get("data_collection_active"))),
            "training_excluded_bots": sum(1 for row in rows if bool(row.get("training_excluded"))),
            "structured_capability_pack_bot_count": sum(1 for row in rows if str(row.get("capability_pack_version") or "")),
            "advanced_intelligence_mesh_bot_count": sum(1 for row in rows if str(row.get("advanced_mesh_version") or "")),
            "cognitive_control_plane_bot_count": sum(1 for row in rows if str(row.get("cognitive_control_version") or "")),
            "recursive_research_foundry_bot_count": sum(1 for row in rows if str(row.get("recursive_foundry_version") or "") == FOUNDRY_VERSION),
            "latest_recursive_research_foundry": FOUNDRY_VERSION,
        }
    )
    registry["summary"] = summary


def _foundry_summary(assigned_ids: dict[str, str]) -> dict[str, Any]:
    contract = _foundry_contract(assigned_ids)
    return {
        "slug": FOUNDRY_SLUG,
        "display_name": FOUNDRY_DISPLAY_NAME,
        "sleeve_family": FOUNDRY_FAMILY,
        "sleeve_profile": FOUNDRY_PROFILE,
        "objective": "Generate, prove, red-team, distill, and package new research ideas before strategy admission.",
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "regression_guard_bot_id": contract["regression_guard_bot_id"],
        "capacity_check": contract["capacity_check"],
        "research_depth": list(FOUNDRY_CONTRACT["research_depth"]),
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
        "recursive_foundry_version": FOUNDRY_VERSION,
        "pack_count": 1,
        "bot_count": len(BOTS),
        "planned_bot_count": len(planned_rows),
        "skipped_existing_count": len(skipped_existing),
        "planned_rows": planned_rows,
        "skipped_existing_slot_kinds": skipped_existing,
        "foundry": _foundry_summary(assigned_ids),
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
        "recursive_foundry_version": FOUNDRY_VERSION,
        "pack_count": plan["pack_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "foundry": plan["foundry"],
        "recursive_foundry_contract": FOUNDRY_CONTRACT,
        "recommended_apply_command": "./scripts/ops/opsctl.sh recursive-research-foundry --apply --json",
    }


def apply_registry(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    registry_path = project_root / "master_bot_registry.json"
    registry = _load_json(registry_path)
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    plan = plan_registry_expansion(registry)
    added_rows = list(plan["planned_rows"])
    backup_path = ""
    if added_rows:
        backup_dir = project_root / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"master_bot_registry_before_recursive_research_foundry_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
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
        }
    )
    _write_json(
        project_root / "config" / "recursive_research_foundry_v1.json",
        {
            "generated_at_utc": _utc_now(),
            "recursive_foundry_version": FOUNDRY_VERSION,
            "foundry": payload["foundry"],
            "recursive_foundry_contract": payload["recursive_foundry_contract"],
        },
    )
    _write_json(project_root / "governance" / "health" / "recursive_research_foundry_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add a 15-bot recursive research foundry expansion.")
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
            "recursive_research_foundry "
            f"mode={payload['mode']} bots={payload['bot_count']} "
            f"planned={payload['planned_bot_count']} added={payload.get('added_bot_count', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
