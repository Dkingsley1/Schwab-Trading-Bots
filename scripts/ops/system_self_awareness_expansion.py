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
BASE_VERSION = 894
PACK_VERSION = "system_self_awareness_v1"
PACK_SLUG = "system_self_awareness"
PACK_DISPLAY_NAME = "System Self-Awareness Infrabots"
SLEEVE_FAMILY = "self_awareness_infrastructure"
SLEEVE_PROFILE = "system_self_awareness"
LABEL_CONTRACT_VERSION = "system_self_awareness_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 5000
MINIMUM_COLLECTION_DAYS = 21

DATA_INTAKES = [
    "system_self_model_snapshot",
    "operator_cockpit_adaptive_posture",
    "memory_cotenant_awareness_trace",
    "registry_identity_surface",
    "health_surface_status_matrix",
    "incident_failure_memory_trace",
    "dependency_edge_trace",
    "expansion_pressure_trace",
    "self_optimization_action_trace",
    "grandmaster_self_model_packet",
]

STORAGE_TARGETS = [
    "governance/self_model",
    "governance/self_model/dependency_graph",
    "governance/self_model/failure_memory",
    "governance/self_model/optimization",
    "governance/health/system_self_model_latest.json",
    "exports/reports/operator/system_self_model_latest.md",
]

REQUIRED_LABELS = [
    "self_model_domain_status",
    "resource_pressure_context",
    "dependency_edge_status",
    "incident_cause_bucket",
    "expansion_pressure_bucket",
    "optimization_priority_bucket",
]

BOTS: list[dict[str, Any]] = [
    {
        "role_slug": "self_model_identity_cartographer",
        "slug": "self_model_identity_cartographer_bot",
        "label": "Self-Model Identity Cartographer",
        "priority": "critical",
        "objective": "Maintain the platform identity map: bot counts, sleeves, lifecycle states, and capability packs.",
        "target_functions": ["master_bot_registry", "core_bot_catalog", "operator_cockpit"],
    },
    {
        "role_slug": "resource_cotenant_awareness_guard",
        "slug": "resource_cotenant_awareness_guard_bot",
        "label": "Resource Co-Tenant Awareness Guard",
        "priority": "critical",
        "objective": "Track foreground apps, memory, swap, storage, and throttle mode so open apps become context instead of false alarms.",
        "target_functions": ["memory_efficiency", "runtime_throttle", "resource_guard"],
    },
    {
        "role_slug": "bot_roster_drift_mapper",
        "slug": "bot_roster_drift_mapper_bot",
        "label": "Bot Roster Drift Mapper",
        "priority": "high",
        "objective": "Diff registry changes across expansions and flag missing, duplicate, or unmapped bot identities.",
        "target_functions": ["master_bot_registry", "core_bot_materialization_guard", "bot_founder_dna"],
    },
    {
        "role_slug": "incident_failure_memory_linker",
        "slug": "incident_failure_memory_linker_bot",
        "label": "Incident Failure Memory Linker",
        "priority": "critical",
        "objective": "Join halts, tripwires, feed cuts, margin guards, and backlog pressure into replayable incident memory.",
        "target_functions": ["global_halt", "incident_timeline", "watchdog_events"],
    },
    {
        "role_slug": "dependency_graph_surface_mapper",
        "slug": "dependency_graph_surface_mapper_bot",
        "label": "Dependency Graph Surface Mapper",
        "priority": "high",
        "objective": "Map upstream and downstream dependencies between health files, feeds, writers, sleeves, and reports.",
        "target_functions": ["operator_cockpit", "system_self_model", "artifact_freshness_slo"],
    },
    {
        "role_slug": "growth_pressure_forecaster",
        "slug": "growth_pressure_forecaster_bot",
        "label": "Growth Pressure Forecaster",
        "priority": "critical",
        "objective": "Forecast when bot growth requires rollups, cold lanes, storage pruning, or throttled collection windows.",
        "target_functions": ["expansion_capacity", "memory_efficiency", "ingestion_storage_control"],
    },
    {
        "role_slug": "self_reporting_narrator",
        "slug": "self_reporting_narrator_bot",
        "label": "Self-Reporting Narrator",
        "priority": "high",
        "objective": "Generate concise self-briefs explaining current state, what changed, why it downshifted, and safe next commands.",
        "target_functions": ["system_self_model", "operator_cockpit", "system_summary"],
    },
    {
        "role_slug": "optimization_recommendation_ranker",
        "slug": "optimization_recommendation_ranker_bot",
        "label": "Optimization Recommendation Ranker",
        "priority": "high",
        "objective": "Rank self-improvement actions by risk, payoff, staleness, resource cost, and impact on live collection.",
        "target_functions": ["system_self_model", "master_infrastructure_supervisor", "daily_verify_auto_remediation"],
    },
    {
        "role_slug": "self_model_regression_guard",
        "slug": "self_model_regression_guard_bot",
        "label": "Self-Model Regression Guard",
        "priority": "critical",
        "objective": "Fail loudly when self-model outputs become stale, contradictory, or disconnected from registry truth.",
        "target_functions": ["system_self_model", "commands_verify", "regression_guard"],
    },
    {
        "role_slug": "grandmaster_self_awareness_bridge",
        "slug": "grandmaster_self_awareness_bridge_bot",
        "label": "Grand Master Self-Awareness Bridge",
        "priority": "critical",
        "objective": "Compress the self-model into Grand Master packets for safer routing, throttling, expansion, and reporting decisions.",
        "target_functions": ["grand_master_reporting", "operator_cockpit", "adaptive_intelligence_kernel"],
    },
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
        version = desired if desired not in used_versions else _next_available_version(used_versions, max(used_versions, default=desired) + 1)
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


def _pack_contract(assigned_ids: dict[str, str]) -> dict[str, Any]:
    return {
        "contract_version": PACK_VERSION,
        "new_sleeve_or_subsleeve": {
            "sleeve_family": SLEEVE_FAMILY,
            "sleeve_profile": SLEEVE_PROFILE,
            "display_name": PACK_DISPLAY_NAME,
        },
        "bot_pack_size": len(BOTS),
        "bot_pack_size_rule": "5_to_15_bots_max",
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "self_model_hot_14d_warm_120d_cold_365d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": 25,
            "capture_mode": "diffed_snapshot_summary",
            "sample_rate": 0.25,
            "dedupe_required": True,
        },
        "paper_only_floor": {
            "paper_trade_lock_required": True,
            "live_trading_enabled": False,
            "execution_enabled": False,
            "allocation_enabled": False,
            "graduation_requires_minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "graduation_requires_collection_days": MINIMUM_COLLECTION_DAYS,
        },
        "sleeve_master_bot_id": assigned_ids.get(f"{PACK_SLUG}_self_model_identity_cartographer", ""),
        "regression_guard_bot_id": assigned_ids.get(f"{PACK_SLUG}_self_model_regression_guard", ""),
        "self_awareness_depth": [
            "resource_awareness",
            "bot_identity_awareness",
            "failure_memory",
            "dependency_graph",
            "growth_pressure",
            "self_reporting",
            "optimization_ranking",
            "grandmaster_self_model_bridge",
        ],
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    contract = _pack_contract(assigned_ids)
    return {
        "bot_id": bot_id,
        "bot_role": "infrastructure_sub_bot",
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
        "preferred_regimes": ["all_weather", "expansion_pressure", "resource_pressure", "incident_recovery", "low_pressure"],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v724_global_halt_clearance_verifier_bot",
            "brain_refinery_v804_advanced_mesh_metacognitive_state_router_master_bot",
            "brain_refinery_v879_adaptive_kernel_online_meta_learning_master_bot",
        ],
        "data_intake_collections": list(DATA_INTAKES),
        "storage_targets": list(STORAGE_TARGETS),
        "freshness_slo_seconds": 900,
        "retention_profile": "self_model_hot_14d_warm_120d_cold_365d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "system_self_awareness_observer_until_self_model_history_and_regression_guards_clear",
        "trading_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "allocation_enabled": False,
        "execution_enabled": False,
        "rotation_blocked": True,
        "rotation_block_reason": "self_awareness_collection_only_zero_weight",
        "training_excluded": True,
        "exclude_from_training": True,
        "training_candidate_after_threshold": True,
        "training_exclusion_reason": "collecting_self_model_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "data_collection_storage_guarded": True,
        "data_collection_capture_mode": "diffed_snapshot_summary",
        "data_collection_sample_rate": 0.25,
        "data_collection_max_daily_storage_mb": 25,
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "sleeve_profile": SLEEVE_PROFILE,
        "sleeve_family": SLEEVE_FAMILY,
        "correlation_peer_sleeves": [
            "adaptive_intelligence_kernel",
            "advanced_intelligence_mesh",
            "coordination_intelligence",
            "system_governor",
            "memory_efficiency",
            "runtime_throttle",
        ],
        "correlation_dependencies": [
            "system_self_model",
            "operator_cockpit",
            "memory_efficiency",
            "runtime_throttle",
            "global_halt_status",
            "core_bot_materialization_guard",
        ],
        "provider_capability_profile": "internal_self_model_governance_only",
        "direct_market_data_available": False,
        "direct_execution_allowed": False,
        "proxy_data_sources": [
            "master_bot_registry",
            "operator_cockpit",
            "memory_efficiency",
            "runtime_throttle",
            "global_halt",
            "incident_memory",
            "core_bot_catalog",
        ],
        "schwab_direct_inputs": [],
        "proxy_only_reason": "system_self_awareness_has_no_direct_execution_or_broker_dependency",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(REQUIRED_LABELS),
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.74,
            "freshness_slo_seconds": 900,
            "regression_guard_bot_id": contract["regression_guard_bot_id"],
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
            "system_self_awareness",
            "training_after_threshold",
            "global_halt_aware",
        ],
        "execution_policy_label": "collection_only_self_awareness_no_execution",
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
        ],
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "paper_runtime_stability_mode": "full_force_buffered",
        "paper_trade_lock_required": True,
        "paper_runtime_capacity_floor": 700,
        "capability_pack_version": PACK_VERSION,
        "capability_pack_slug": PACK_SLUG,
        "capability_pack_display_name": PACK_DISPLAY_NAME,
        "system_self_awareness_version": PACK_VERSION,
        "capability_pack_contract": contract,
    }


def _refresh_summary(registry: dict[str, Any]) -> None:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    active = [row for row in rows if bool(row.get("active"))]
    inactive = [row for row in rows if not bool(row.get("active"))]
    signal_active = [row for row in active if str(row.get("bot_role") or "") == "signal_sub_bot"]
    infra_active = [row for row in active if str(row.get("bot_role") or "") == "infrastructure_sub_bot"]
    structured = [row for row in rows if str(row.get("capability_pack_version") or "")]
    self_awareness = [row for row in rows if str(row.get("system_self_awareness_version") or "") == PACK_VERSION]
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
            "system_self_awareness_bot_count": len(self_awareness),
            "latest_system_self_awareness": PACK_VERSION,
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
        "objective": "Add operational self-awareness infrabots that maintain identity, resource, dependency, failure-memory, growth, and self-reporting context.",
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "regression_guard_bot_id": contract["regression_guard_bot_id"],
        "self_awareness_depth": list(contract["self_awareness_depth"]),
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
        "system_self_awareness_version": PACK_VERSION,
        "pack_count": 1,
        "bot_count": len(BOTS),
        "planned_bot_count": len(planned_rows),
        "skipped_existing_count": len(skipped_existing),
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
        "system_self_awareness_version": PACK_VERSION,
        "pack_count": plan["pack_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "pack": plan["pack"],
        "recommended_apply_command": "./scripts/ops/opsctl.sh self-awareness-infrabots --apply --json",
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
        backup = backup_dir / f"master_bot_registry_before_system_self_awareness_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
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
        project_root / "config" / "system_self_awareness_v1.json",
        {
            "generated_at_utc": _utc_now(),
            "system_self_awareness_version": PACK_VERSION,
            "pack": payload["pack"],
        },
    )
    _write_json(project_root / "governance" / "health" / "system_self_awareness_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add system self-awareness infrabots.")
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
            "system_self_awareness "
            f"mode={payload['mode']} bots={payload['bot_count']} "
            f"planned={payload['planned_bot_count']} added={payload.get('added_bot_count', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
