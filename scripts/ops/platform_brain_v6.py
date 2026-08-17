#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, status_rank, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, status_rank, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "platform_brain_v6_latest.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "platform_brain_v6_foresight.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.platform_brain_v6_override"
DEFAULT_MEMORY_PATH = PROJECT_ROOT / "governance" / "platform_brain_v6" / "foresight_memory" / "foresight_events.jsonl"

SECTION_KEYS: tuple[str, ...] = (
    "counterfactual_world_model",
    "causal_intervention_planner",
    "hierarchical_memory_router",
    "multi_agent_debate_chamber",
    "uncertainty_risk_calibrator",
    "alpha_research_thesis_factory",
    "market_microstructure_world_model",
    "macro_event_anticipator",
    "execution_policy_sandbox",
    "adaptive_resource_governor",
    "data_value_active_learner",
    "formal_safety_guard",
    "self_upgrade_experiment_designer",
    "bot_genome_lineage_planner",
    "operator_narrative_synthesizer",
)

CONTROLS: tuple[dict[str, str], ...] = (
    {"id": "counterfactual_world_model", "title": "Counterfactual world model", "env_key": "PLATFORM_BRAIN_V6_COUNTERFACTUAL_WORLD_MODEL_ENABLED"},
    {"id": "causal_intervention_planner", "title": "Causal intervention planner", "env_key": "PLATFORM_BRAIN_V6_CAUSAL_INTERVENTION_ENABLED"},
    {"id": "hierarchical_memory_router", "title": "Hierarchical memory router", "env_key": "PLATFORM_BRAIN_V6_MEMORY_ROUTER_ENABLED"},
    {"id": "multi_agent_debate_chamber", "title": "Multi-agent debate chamber", "env_key": "PLATFORM_BRAIN_V6_DEBATE_CHAMBER_ENABLED"},
    {"id": "uncertainty_risk_calibrator", "title": "Uncertainty and risk calibrator", "env_key": "PLATFORM_BRAIN_V6_UNCERTAINTY_CALIBRATOR_ENABLED"},
    {"id": "alpha_research_thesis_factory", "title": "Alpha research thesis factory", "env_key": "PLATFORM_BRAIN_V6_ALPHA_THESIS_FACTORY_ENABLED"},
    {"id": "market_microstructure_world_model", "title": "Market microstructure world model", "env_key": "PLATFORM_BRAIN_V6_MICROSTRUCTURE_WORLD_MODEL_ENABLED"},
    {"id": "macro_event_anticipator", "title": "Macro event anticipator", "env_key": "PLATFORM_BRAIN_V6_MACRO_EVENT_ANTICIPATOR_ENABLED"},
    {"id": "execution_policy_sandbox", "title": "Execution policy sandbox", "env_key": "PLATFORM_BRAIN_V6_EXECUTION_SANDBOX_ENABLED"},
    {"id": "adaptive_resource_governor", "title": "Adaptive resource governor", "env_key": "PLATFORM_BRAIN_V6_RESOURCE_GOVERNOR_ENABLED"},
    {"id": "data_value_active_learner", "title": "Data-value active learner", "env_key": "PLATFORM_BRAIN_V6_ACTIVE_LEARNER_ENABLED"},
    {"id": "formal_safety_guard", "title": "Formal safety guard", "env_key": "PLATFORM_BRAIN_V6_FORMAL_SAFETY_ENABLED"},
    {"id": "self_upgrade_experiment_designer", "title": "Self-upgrade experiment designer", "env_key": "PLATFORM_BRAIN_V6_EXPERIMENT_DESIGNER_ENABLED"},
    {"id": "bot_genome_lineage_planner", "title": "Bot genome lineage planner", "env_key": "PLATFORM_BRAIN_V6_GENOME_LINEAGE_ENABLED"},
    {"id": "operator_narrative_synthesizer", "title": "Operator narrative synthesizer", "env_key": "PLATFORM_BRAIN_V6_OPERATOR_NARRATIVE_ENABLED"},
)

SAFE_REFLEX_PREFIXES = (
    "./scripts/ops/opsctl.sh health-fast",
    "./scripts/ops/opsctl.sh pressure-relief",
    "./scripts/ops/opsctl.sh runtime-throttle",
    "./scripts/ops/opsctl.sh memory-efficiency",
    "./scripts/ops/opsctl.sh platform-stabilization",
    "./scripts/ops/opsctl.sh platform-settlement-stabilization",
    "./scripts/ops/opsctl.sh backpressure-drainers",
    "./scripts/ops/opsctl.sh storage-backpressure-autopilot",
)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on", "active", "enabled"}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _health(project_root: Path, name: str) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / name)
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path, *, limit: int = 300) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    return rows[-max(limit, 1):]


def _registry_summary(project_root: Path) -> dict[str, Any]:
    payload = load_json(project_root / "master_bot_registry.json")
    rows = [row for row in payload.get("sub_bots", []) if isinstance(row, dict)] if isinstance(payload.get("sub_bots"), list) else []
    active = sum(1 for row in rows if _bool(row.get("active")))
    collecting = sum(1 for row in rows if _bool(row.get("data_collection_active")) or str(row.get("lifecycle_state") or "") == "data_collection_only")
    training_excluded = sum(1 for row in rows if _bool(row.get("training_excluded")) or _bool(row.get("exclude_from_training")))
    trainable = sum(1 for row in rows if _bool(row.get("data_collection_training_ready")) and not (_bool(row.get("training_excluded")) or _bool(row.get("exclude_from_training"))))
    sleeves = {str(row.get("sleeve_profile") or row.get("sleeve") or row.get("profile") or "default") for row in rows}
    pack_counts = Counter(str(row.get("capability_pack_slug") or "unpacked") for row in rows)
    return {
        "total_bots": len(rows),
        "active_bots": active,
        "collecting_bots": collecting,
        "training_excluded_bots": training_excluded,
        "trainable_bots": trainable,
        "sleeve_count": len([sleeve for sleeve in sleeves if sleeve]),
        "capability_pack_counts": dict(sorted(pack_counts.items())),
    }


def _status_rows(sections: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in SECTION_KEYS:
        status = str(_as_dict(sections.get(key)).get("overall_status") or "missing")
        rows.append({"section": key, "overall_status": status, "rank": status_rank(status)})
    return rows


def _worst_status(rows: list[dict[str, Any]]) -> str:
    statuses = {str(row.get("overall_status") or "") for row in rows}
    if statuses & {"blocked", "critical"}:
        return "blocked"
    if statuses & {"degraded"}:
        return "degraded"
    if statuses & {"needs_work", "watch", "thin", "missing"}:
        return "needs_work"
    return "ready"


def _pressure_context(project_root: Path) -> dict[str, Any]:
    runtime = _health(project_root, "runtime_throttle_control_latest.json")
    memory = _health(project_root, "memory_efficiency_control_latest.json")
    swap = _health(project_root, "swap_pressure_governor_latest.json")
    storage = _health(project_root, "ingestion_storage_control_latest.json")
    halt = _health(project_root, "global_halt_auto_clear_latest.json") or _health(project_root, "global_killswitch_latest.json")
    storage_bp = _as_dict(storage.get("backpressure"))
    swap_payload = _as_dict(swap.get("swap_pressure"))
    total_pending = max(
        _safe_int(storage_bp.get("total_pending_lines"), 0),
        _safe_int(storage_bp.get("core_pending_lines"), 0) + _safe_int(storage_bp.get("deferred_pending_lines"), 0) + _safe_int(storage_bp.get("cold_pending_lines"), 0),
    )
    threshold = max(_safe_int(storage_bp.get("pending_lines_threshold"), 15000), 1)
    return {
        "runtime_status": str(runtime.get("overall_status") or "missing"),
        "host_saturation_score": round(_safe_float(runtime.get("host_saturation_score"), 0.0), 3),
        "compute_pressure_level": str(runtime.get("compute_pressure_level") or "unknown"),
        "memory_pressure_level": str(runtime.get("memory_pressure_level") or "unknown"),
        "memory_status": str(memory.get("overall_status") or "missing"),
        "swap_tier": str(swap_payload.get("tier") or swap.get("tier") or "normal"),
        "swap_used_gb": round(_safe_float(swap_payload.get("swap_used_gb") or swap.get("swap_used_gb"), 0.0), 3),
        "storage_severity": str(storage.get("severity") or "unknown"),
        "storage_pressure_index": round(_safe_float(storage.get("pressure_index"), 0.0), 6),
        "pending_lines_total": total_pending,
        "pending_ratio": round(total_pending / float(threshold), 6),
        "halt_active": _bool(halt.get("halt")) or _bool(halt.get("global_halt_active")),
        "halt_state": str(halt.get("halt_state") or halt.get("state") or "unknown"),
        "clear_blockers": [str(item) for item in _as_list(halt.get("clear_blockers"))],
    }


def _gate_blockers(pressure: dict[str, Any], stabilization: dict[str, Any]) -> list[str]:
    sections = _as_dict(stabilization.get("sections"))
    expansion = _as_dict(sections.get("expansion_rehearsal_gate"))
    blockers = [str(item) for item in _as_list(expansion.get("gate_closed_reasons"))]
    if _safe_float(pressure.get("pending_ratio"), 0.0) >= 1.0 and "queue_backpressure_active" not in blockers:
        blockers.append("queue_backpressure_active")
    if _safe_float(pressure.get("host_saturation_score"), 0.0) >= 65.0 and "runtime_not_calm" not in blockers:
        blockers.append("runtime_not_calm")
    if _bool(pressure.get("halt_active")) and "global_halt_active" not in blockers:
        blockers.append("global_halt_active")
    for blocker in _as_list(pressure.get("clear_blockers")):
        if blocker and blocker not in blockers:
            blockers.append(str(blocker))
    return blockers


def _counterfactual_world_model(v5: dict[str, Any], pressure: dict[str, Any], blockers: list[str]) -> dict[str, Any]:
    scenarios = [
        {"counterfactual": "add_50_bots_now", "expected_effect": "backpressure_and_runtime_pressure_worse" if blockers else "guarded_collect_only_possible", "decision": "defer" if blockers else "rehearse"},
        {"counterfactual": "train_now", "expected_effect": "swap_and_cpu_spike" if _safe_float(pressure.get("host_saturation_score"), 0.0) >= 65 else "small_off_hours_batch_possible", "decision": "hold" if blockers else "microbatch_only"},
        {"counterfactual": "heavy_feed_view", "expected_effect": "operator_visible_churn", "decision": "ttl_digest_mode"},
        {"counterfactual": "provider_429_burst", "expected_effect": "cooldown_not_halt", "decision": "use_damper_and_last_good_cache"},
        {"counterfactual": "external_drive_reconnect", "expected_effect": "route_verification_then_drain", "decision": "settlement_guard_first"},
    ]
    return {
        "overall_status": "needs_work" if blockers else "ready",
        "v5_status": v5.get("overall_status", "missing"),
        "blocker_count": len(blockers),
        "counterfactuals": scenarios,
        "world_model_contract": [
            "simulate_before_expanding_or_training",
            "prefer_counterfactual_evidence_over_snapshot_confidence",
            "provider_denials_degrade_collection_instead_of_global_halting",
        ],
    }


def _causal_intervention_planner(blockers: list[str]) -> dict[str, Any]:
    command_by_blocker = {
        "queue_backpressure_active": "./scripts/ops/opsctl.sh backpressure-drainers --apply --ttl-seconds 1200 --json",
        "storage_or_queue_not_settled": "./scripts/ops/opsctl.sh platform-settlement-stabilization --apply --json",
        "runtime_not_calm": "./scripts/ops/opsctl.sh pressure-relief --apply --json",
        "global_clear_blockers_present": "./scripts/ops/opsctl.sh global-halt-auto-clear --json",
        "provider_degraded": "./scripts/ops/opsctl.sh provider-mesh --json",
        "collection_floor_not_clean": "./scripts/ops/opsctl.sh data-collection-observation-rollup --json",
    }
    interventions = []
    for blocker in blockers[:10]:
        command = command_by_blocker.get(blocker, "./scripts/ops/opsctl.sh health-fast --json")
        interventions.append({"blocker": blocker, "command": command, "safe_auto": command.startswith(SAFE_REFLEX_PREFIXES)})
    if not interventions:
        interventions.append({"blocker": "none", "command": "./scripts/ops/opsctl.sh health-fast --json", "safe_auto": True})
    return {
        "overall_status": "needs_work" if blockers else "ready",
        "intervention_count": len(interventions),
        "interventions": interventions,
        "intervention_contract": [
            "one_guarded_intervention_per_pressure_class",
            "do_not_force_clear_halts_or_start_execution",
            "prefer_existing_opsctl_commands",
        ],
    }


def _hierarchical_memory_router(project_root: Path) -> dict[str, Any]:
    v4_events = _read_jsonl(project_root / "governance" / "platform_brain_v4" / "experience_memory" / "experience_memory_events.jsonl")
    v5_events = _read_jsonl(project_root / "governance" / "platform_brain_v5" / "reflex_memory" / "reflex_events.jsonl")
    v6_events = _read_jsonl(project_root / "governance" / "platform_brain_v6" / "foresight_memory" / "foresight_events.jsonl")
    routes = [
        {"memory_tier": "hot_reflex", "event_count": len(v5_events), "use_for": "next_command_and_recent_regret"},
        {"memory_tier": "experience", "event_count": len(v4_events), "use_for": "what_worked_or_repeated"},
        {"memory_tier": "foresight", "event_count": len(v6_events), "use_for": "counterfactual_and_intervention_outcomes"},
    ]
    return {
        "overall_status": "ready" if len(v4_events) + len(v5_events) > 0 else "thin",
        "memory_event_count": sum(row["event_count"] for row in routes),
        "routes": routes,
        "memory_contract": [
            "route_fast_reflex_memory_separately_from_longer_experience",
            "append_compact_foresight_events_on_apply",
            "prevent_self_model_outputs_from_becoming_their_own_evidence",
        ],
    }


def _multi_agent_debate(blockers: list[str], pressure: dict[str, Any]) -> dict[str, Any]:
    votes = [
        {"agent": "pressure_critic", "vote": "hold_expansion" if blockers else "allow_rehearsal", "reason": "runtime_and_queue_first"},
        {"agent": "data_value_critic", "vote": "collect_more_quality", "reason": "new_bots_need_observation_depth"},
        {"agent": "safety_critic", "vote": "advisory_only", "reason": "paper_lock_and_no_execution"},
        {"agent": "alpha_builder", "vote": "prepare_collect_only_pack", "reason": "safe_to_expand_intelligence_without_training_or_execution"},
        {"agent": "operator_headroom_critic", "vote": "keep_foreground_safe" if _safe_float(pressure.get("host_saturation_score"), 0.0) >= 50 else "normal", "reason": "leave_mac_headroom"},
    ]
    hold_count = sum(1 for row in votes if str(row.get("vote")).startswith("hold"))
    return {"overall_status": "needs_work" if hold_count else "ready", "vote_count": len(votes), "hold_count": hold_count, "votes": votes}


def _uncertainty_risk(pressure: dict[str, Any], registry: dict[str, Any], blockers: list[str]) -> dict[str, Any]:
    score = min(100.0, len(blockers) * 11.0 + _safe_float(pressure.get("pending_ratio"), 0.0) * 18.0 + max(0.0, _safe_float(pressure.get("host_saturation_score"), 0.0) - 50.0) * 0.7)
    confidence = max(0.05, 1.0 - score / 140.0)
    return {
        "overall_status": "needs_work" if score >= 45 else "ready",
        "uncertainty_score": round(score, 3),
        "decision_confidence": round(confidence, 4),
        "active_bots": registry.get("active_bots", 0),
        "calibration_policy": "discount_recommendations_when_pressure_or_backlog_is_high",
    }


def _alpha_research_factory(blockers: list[str]) -> dict[str, Any]:
    lanes = [
        {"lane": "cross_sleeve_alpha_novelty", "mode": "digest_only", "promotion_allowed": False},
        {"lane": "execution_cost_edge", "mode": "paper_realism_research", "promotion_allowed": False},
        {"lane": "macro_event_prepositioning", "mode": "calendar_and_source_quality", "promotion_allowed": False},
        {"lane": "microstructure_toxicity", "mode": "vpins_lob_imbalance_digest", "promotion_allowed": False},
        {"lane": "model_decay_rescue", "mode": "candidate_requalification", "promotion_allowed": False},
    ]
    return {
        "overall_status": "ready",
        "research_mode": "collect_only_until_calm" if blockers else "guarded_research",
        "thesis_count": len(lanes),
        "thesis_lanes": lanes,
    }


def _microstructure_world_model() -> dict[str, Any]:
    return {
        "overall_status": "ready",
        "features": ["spread_regime", "queue_depth_decay", "order_flow_imbalance", "vpin_toxicity", "fill_probability", "slippage_surface"],
        "routing_policy": "paper_trade_realism_receives_microstructure_discount_before_any_promotion",
    }


def _macro_event_anticipator(project_root: Path) -> dict[str, Any]:
    macro = _health(project_root, "live_macro_auto_watch_latest.json") or _health(project_root, "macro_auto_watch_latest.json")
    events = _as_list(macro.get("events")) or _as_list(macro.get("upcoming_events"))
    return {
        "overall_status": "ready" if macro else "thin",
        "macro_watch_status": macro.get("overall_status", "missing") if macro else "missing",
        "event_count": len(events),
        "anticipation_policy": "prime_feeds_and_calm_training_around_high_impact_events",
    }


def _execution_sandbox() -> dict[str, Any]:
    return {
        "overall_status": "ready",
        "paper_trade_lock_required": True,
        "live_execution_allowed": False,
        "allocation_allowed": False,
        "sandbox_checks": ["spread_slippage_latency", "queue_position", "partial_fill", "halt_resume", "provider_cooldown"],
    }


def _resource_governor(pressure: dict[str, Any]) -> dict[str, Any]:
    hot = _safe_float(pressure.get("host_saturation_score"), 0.0) >= 65 or _safe_float(pressure.get("pending_ratio"), 0.0) >= 1.0
    budget = {"live_collection": 68, "sql_drain": 22, "intelligence": 5, "reports": 3, "training": 0, "research": 2} if hot else {"live_collection": 50, "sql_drain": 18, "intelligence": 10, "reports": 7, "training": 5, "research": 10}
    return {"overall_status": "ready", "pressure_hot": hot, "resource_budget_percent": budget, "governor_contract": "live_collection_and_sql_drain_buy_capacity_first"}


def _data_value_active_learner(registry: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall_status": "ready",
        "collecting_bots": registry.get("collecting_bots", 0),
        "active_learning_targets": ["missing_labels", "thin_sleeves", "provider_conflict", "execution_realism_gaps", "duplicate_alpha_clusters"],
        "sample_policy": "ask_for_information_value_not_more_raw_volume",
    }


def _formal_safety_guard() -> dict[str, Any]:
    blocked = ["live_order_execution", "credential_entry", "manual_halt_force_clear", "destructive_cleanup", "training_without_green_gates"]
    return {"overall_status": "ready", "autonomy_mode": "advisory_guarded_reflex_only", "blocked_actions": blocked, "mlx_default": True, "consciousness_claim_allowed": False}


def _experiment_designer(blockers: list[str]) -> dict[str, Any]:
    experiments = [
        {"name": "pressure_relief_delta", "metric": "host_saturation_and_pending_ratio", "safe_to_run": True},
        {"name": "drain_contract_replay", "metric": "pending_lines_per_minute", "safe_to_run": True},
        {"name": "provider_damper_replay", "metric": "false_halt_reduction", "safe_to_run": True},
        {"name": "bot_quality_lift", "metric": "quality_score_and_zero_obs", "safe_to_run": not blockers},
    ]
    return {"overall_status": "ready", "experiment_count": len(experiments), "experiments": experiments, "default_mode": "shadow_replay"}


def _genome_lineage(registry: dict[str, Any]) -> dict[str, Any]:
    packs = _as_dict(registry.get("capability_pack_counts"))
    return {
        "overall_status": "ready",
        "founder_bot_id": "brain_refinery_v1",
        "lineage_policy": "all_new_bots_inherit_founder_dna_and_current_safety_contracts",
        "capability_pack_counts": packs,
    }


def _operator_narrative(blockers: list[str], interventions: dict[str, Any], pressure: dict[str, Any]) -> dict[str, Any]:
    first_command = "./scripts/ops/opsctl.sh health-fast --json"
    rows = _as_list(interventions.get("interventions"))
    if rows:
        first_command = str(_as_dict(rows[0]).get("command") or first_command)
    summary = "hold_expansion_and_drain_pressure" if blockers else "calm_enough_for_guarded_rehearsal"
    return {
        "overall_status": "needs_work" if blockers else "ready",
        "operator_summary": summary,
        "next_best_command": first_command,
        "pressure_snapshot": pressure,
    }


def _env_overrides(payload: dict[str, Any]) -> dict[str, str]:
    sections = _as_dict(payload.get("sections"))
    narrative = _as_dict(sections.get("operator_narrative_synthesizer"))
    blockers = _as_list(payload.get("gate_blockers"))
    env = {
        "PLATFORM_BRAIN_V6_ENABLED": "1",
        "PLATFORM_BRAIN_V6_MODE": "foresight_advisory_collect_only",
        "PLATFORM_BRAIN_V6_SECTION_COUNT": str(len(SECTION_KEYS)),
        "PLATFORM_BRAIN_V6_NEXT_BEST_COMMAND": str(narrative.get("next_best_command") or ""),
        "PLATFORM_BRAIN_V6_EXPANSION_ALLOWED_NOW": "0" if blockers else "1",
        "PLATFORM_BRAIN_V6_GATE_BLOCKERS": ",".join(str(item) for item in blockers) or "none",
        "INTELLIGENCE_EXPANSION_DEFAULT_MODE": "collect_only",
        "INTELLIGENCE_EXPANSION_REQUIRE_STABILIZATION_GREEN": "1",
        "INTELLIGENCE_EXPANSION_TRAINING_EXCLUDED_DEFAULT": "1",
        "PRIMARY_ML_RUNTIME_BACKEND": "mlx",
        "LIBRARY_DEFAULT_ML_BACKEND": "mlx",
        "PAPER_TRADE_LOCK": "1",
        "ALLOW_ORDER_EXECUTION": "0",
    }
    for control in CONTROLS:
        env[control["env_key"]] = "1"
    return env


def _write_env_override(path: Path, env: dict[str, str]) -> bool:
    lines = ["# Auto-managed by scripts/ops/platform_brain_v6.py"]
    for key in sorted(env):
        lines.append(f"{key}={shlex.quote(str(env[key]))}")
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _write_config(path: Path, payload: dict[str, Any]) -> bool:
    config = {
        "schema_version": 1,
        "updated_utc": payload.get("timestamp_utc"),
        "layer": "platform_brain_v6_foresight_cortex",
        "section_keys": list(SECTION_KEYS),
        "controls": list(CONTROLS),
        "artifacts": payload.get("section_artifacts", {}),
    }
    content = json.dumps(config, ensure_ascii=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _append_memory_event(path: Path, event: dict[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
    return True


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    v5 = _health(project_root, "platform_brain_v5_latest.json")
    stabilization = _health(project_root, "platform_stabilization_quality_latest.json")
    platform = _health(project_root, "platform_intelligence_expansion_latest.json")
    deep = _health(project_root, "deep_recursive_awareness_latest.json")
    registry = _registry_summary(project_root)
    pressure = _pressure_context(project_root)
    blockers = _gate_blockers(pressure, stabilization)
    counterfactual = _counterfactual_world_model(v5, pressure, blockers)
    intervention = _causal_intervention_planner(blockers)
    memory = _hierarchical_memory_router(project_root)
    debate = _multi_agent_debate(blockers, pressure)
    uncertainty = _uncertainty_risk(pressure, registry, blockers)
    alpha = _alpha_research_factory(blockers)
    micro = _microstructure_world_model()
    macro = _macro_event_anticipator(project_root)
    execution = _execution_sandbox()
    resources = _resource_governor(pressure)
    data_value = _data_value_active_learner(registry)
    safety = _formal_safety_guard()
    experiments = _experiment_designer(blockers)
    genome = _genome_lineage(registry)
    narrative = _operator_narrative(blockers, intervention, pressure)
    sections = {
        "counterfactual_world_model": counterfactual,
        "causal_intervention_planner": intervention,
        "hierarchical_memory_router": memory,
        "multi_agent_debate_chamber": debate,
        "uncertainty_risk_calibrator": uncertainty,
        "alpha_research_thesis_factory": alpha,
        "market_microstructure_world_model": micro,
        "macro_event_anticipator": macro,
        "execution_policy_sandbox": execution,
        "adaptive_resource_governor": resources,
        "data_value_active_learner": data_value,
        "formal_safety_guard": safety,
        "self_upgrade_experiment_designer": experiments,
        "bot_genome_lineage_planner": genome,
        "operator_narrative_synthesizer": narrative,
    }
    rows = _status_rows(sections)
    overall = _worst_status(rows)
    latest_event = {
        "timestamp_utc": iso_now(),
        "overall_status": overall,
        "blockers": blockers,
        "next_best_command": narrative.get("next_best_command"),
        "uncertainty_score": uncertainty.get("uncertainty_score"),
    }
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall in {"ready", "needs_work", "degraded"},
        "overall_status": overall,
        "brain_name": "Platform Brain v6 Foresight Cortex",
        "mode": "foresight_advisory_collect_only",
        "section_count": len(SECTION_KEYS),
        "section_keys": list(SECTION_KEYS),
        "control_count": len(CONTROLS),
        "controls": [{**control, "enabled": True} for control in CONTROLS],
        "registry_summary": registry,
        "v5_status": v5.get("overall_status", "missing"),
        "platform_stabilization_status": stabilization.get("overall_status", "missing"),
        "platform_intelligence_status": platform.get("overall_status", "missing"),
        "deep_recursive_awareness_status": deep.get("mode", "missing"),
        "pressure_context": pressure,
        "gate_blockers": blockers,
        "sections": sections,
        "section_statuses": rows,
        "latest_foresight_event": latest_event,
        "recommended_env_overrides": {},
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "platform-brain-v6", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "platform-brain-v5", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "platform-stabilization", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"],
        ],
        "source_files": {
            "platform_brain_v5": str(project_root / "governance" / "health" / "platform_brain_v5_latest.json"),
            "platform_stabilization": str(project_root / "governance" / "health" / "platform_stabilization_quality_latest.json"),
            "primary_artifact": str(DEFAULT_OUT_PATH),
        },
    }
    payload["recommended_env_overrides"] = _env_overrides(payload)
    return payload


def write_section_artifacts(project_root: Path, payload: dict[str, Any]) -> dict[str, str]:
    root = project_root / "governance" / "platform_brain_v6"
    sections = _as_dict(payload.get("sections"))
    written: dict[str, str] = {}
    for key in SECTION_KEYS:
        section = _as_dict(sections.get(key))
        if not section:
            continue
        path = root / f"{key}_latest.json"
        write_payload(path, {"timestamp_utc": payload.get("timestamp_utc"), "schema_version": 1, **section})
        written[key] = str(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Platform Brain v6 Foresight Cortex advisory layer.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--memory-file", default=str(DEFAULT_MEMORY_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root)
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    payload["section_artifacts"] = write_section_artifacts(project_root, payload)
    if args.apply:
        env = {str(k): str(v) for k, v in _as_dict(payload.get("recommended_env_overrides")).items()}
        payload["apply_result"] = {
            "applied": True,
            "override_path": str(Path(args.override_file).expanduser()),
            "override_changed": _write_env_override(Path(args.override_file).expanduser(), env),
            "config_path": str(Path(args.config_file).expanduser()),
            "config_changed": _write_config(Path(args.config_file).expanduser(), payload),
            "memory_path": str(Path(args.memory_file).expanduser()),
            "foresight_event_appended": _append_memory_event(Path(args.memory_file).expanduser(), _as_dict(payload.get("latest_foresight_event"))),
        }
    else:
        payload["apply_result"] = {"applied": False}
    write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        narrative = _as_dict(_as_dict(payload.get("sections")).get("operator_narrative_synthesizer"))
        print(
            "platform_brain_v6 "
            f"overall_status={payload.get('overall_status')} "
            f"sections={payload.get('section_count')} "
            f"next_best_command={narrative.get('next_best_command', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
