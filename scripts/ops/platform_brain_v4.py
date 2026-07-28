#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, status_rank, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, status_rank, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "platform_brain_v4_latest.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "platform_brain_v4_grande.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.platform_brain_v4_override"
DEFAULT_MEMORY_EVENTS_PATH = PROJECT_ROOT / "governance" / "platform_brain_v4" / "experience_memory" / "experience_memory_events.jsonl"

SECTION_KEYS: tuple[str, ...] = (
    "executive_meta_orchestrator",
    "causal_world_model",
    "experience_memory_core_v2",
    "predictive_expansion_simulator",
    "autonomous_priority_ranker",
    "self_upgrade_planner",
    "critic_council",
    "outcome_verification_loop",
    "bot_portfolio_economist",
    "data_value_engine",
    "training_scheduler_brain",
    "operator_intent_model",
)

PLATFORM_BRAIN_CONTROLS: tuple[dict[str, str], ...] = (
    {"id": "executive_meta_orchestrator", "title": "Executive meta-orchestrator", "env_key": "PLATFORM_BRAIN_EXECUTIVE_ORCHESTRATOR_ENABLED"},
    {"id": "causal_world_model", "title": "Causal world model", "env_key": "PLATFORM_BRAIN_CAUSAL_WORLD_MODEL_ENABLED"},
    {"id": "experience_memory_core_v2", "title": "Experience memory core v2", "env_key": "PLATFORM_BRAIN_EXPERIENCE_MEMORY_V2_ENABLED"},
    {"id": "predictive_expansion_simulator", "title": "Predictive expansion simulator", "env_key": "PLATFORM_BRAIN_EXPANSION_SIMULATOR_ENABLED"},
    {"id": "autonomous_priority_ranker", "title": "Autonomous priority ranker", "env_key": "PLATFORM_BRAIN_PRIORITY_RANKER_ENABLED"},
    {"id": "self_upgrade_planner", "title": "Self-upgrade planner", "env_key": "PLATFORM_BRAIN_SELF_UPGRADE_PLANNER_ENABLED"},
    {"id": "critic_council", "title": "Critic council", "env_key": "PLATFORM_BRAIN_CRITIC_COUNCIL_ENABLED"},
    {"id": "outcome_verification_loop", "title": "Outcome verification loop", "env_key": "PLATFORM_BRAIN_OUTCOME_VERIFICATION_ENABLED"},
    {"id": "bot_portfolio_economist", "title": "Bot portfolio economist", "env_key": "PLATFORM_BRAIN_BOT_PORTFOLIO_ECONOMIST_ENABLED"},
    {"id": "data_value_engine", "title": "Data value engine", "env_key": "PLATFORM_BRAIN_DATA_VALUE_ENGINE_ENABLED"},
    {"id": "training_scheduler_brain", "title": "Training scheduler brain", "env_key": "PLATFORM_BRAIN_TRAINING_SCHEDULER_ENABLED"},
    {"id": "operator_intent_model", "title": "Operator intent model", "env_key": "PLATFORM_BRAIN_OPERATOR_INTENT_MODEL_ENABLED"},
)

SUMMARY_OR_ALIAS_SECTIONS = {
    "professional_system_dashboard",
    "bot_quality_score_system",
    "execution_realism_engine",
}


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
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on", "enabled", "active"}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _health(project_root: Path, name: str) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / name)
    return payload if isinstance(payload, dict) else {}


def _registry_rows(project_root: Path) -> list[dict[str, Any]]:
    payload = load_json(project_root / "master_bot_registry.json")
    rows = payload.get("sub_bots")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _registry_summary(project_root: Path) -> dict[str, Any]:
    rows = _registry_rows(project_root)
    active = sum(1 for row in rows if _bool(row.get("active")))
    collecting = sum(1 for row in rows if _bool(row.get("data_collection_active")) or str(row.get("lifecycle_state") or "") == "data_collection_only")
    training_excluded = sum(1 for row in rows if _bool(row.get("training_excluded")) or _bool(row.get("exclude_from_training")))
    sleeves = {
        str(row.get("sleeve") or row.get("sleeve_profile") or row.get("profile") or row.get("slot_kind") or "default").strip()
        for row in rows
    }
    return {
        "total_bots": len(rows),
        "active_bots": active,
        "collecting_bots": collecting,
        "training_excluded_bots": training_excluded,
        "sleeve_count": len([sleeve for sleeve in sleeves if sleeve]),
    }


def _platform_payload(project_root: Path) -> dict[str, Any]:
    return _health(project_root, "platform_intelligence_expansion_latest.json")


def _platform_sections(platform: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sections = _as_dict(platform.get("sections"))
    return {str(key): value for key, value in sections.items() if isinstance(value, dict)}


def _pressure_snapshot(project_root: Path, platform: dict[str, Any]) -> dict[str, Any]:
    pressure = _as_dict(platform.get("pressure_snapshot"))
    if pressure:
        return pressure
    runtime = _health(project_root, "runtime_throttle_control_latest.json")
    swap = _health(project_root, "swap_pressure_governor_latest.json")
    memory = _health(project_root, "memory_efficiency_control_latest.json")
    ingestion = _health(project_root, "ingestion_storage_control_latest.json")
    swap_pressure = _as_dict(swap.get("swap_pressure"))
    return {
        "overall_status": runtime.get("overall_status") or "missing",
        "host_saturation_score": _safe_float(runtime.get("host_saturation_score"), 0.0),
        "swap_tier": swap_pressure.get("tier", "unknown"),
        "swap_used_gb": _safe_float(swap_pressure.get("swap_used_gb"), 0.0),
        "memory_status": memory.get("overall_status"),
        "storage_status": ingestion.get("overall_status") or ingestion.get("severity"),
        "storage_pressure_index": _safe_float(ingestion.get("pressure_index"), 0.0),
        "compute_policy": "sustain" if str(runtime.get("overall_status")) == "degraded" else "normal",
    }


def _section_status_rows(sections: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, payload in sorted(sections.items()):
        status = str(payload.get("overall_status") or "missing")
        rows.append({"section": key, "overall_status": status, "rank": status_rank(status)})
    return rows


def _worst_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "missing"
    worst = max(rows, key=lambda row: _safe_int(row.get("rank"), 1))
    status = str(worst.get("overall_status") or "missing")
    if status in {"critical", "blocked"}:
        return "blocked"
    if status == "degraded":
        return "degraded"
    if any(str(row.get("overall_status")) == "needs_work" for row in rows):
        return "needs_work"
    if any(str(row.get("overall_status")) in {"watch", "thin"} for row in rows):
        return "watch"
    return "ready"


def _priority_for_section(section: str, status: str) -> dict[str, Any]:
    command_by_section = {
        "provider_rotation_failover_mesh": "./scripts/ops/opsctl.sh pressure-relief --apply --json",
        "backpressure_prediction_engine": "./scripts/ops/opsctl.sh storage-backpressure-autopilot --apply --json",
        "execution_paper_trade_realism_layer": "./scripts/ops/opsctl.sh execution-lab --json",
        "duplicate_alpha_overlap_detector": "./scripts/ops/opsctl.sh platform-intelligence --json",
        "bot_data_quality_scores": "./scripts/ops/opsctl.sh bot-quality-autopilot --json",
        "training_readiness_board": "./scripts/ops/opsctl.sh training-quality --json",
        "self_healing_incident_playbooks": "./scripts/ops/opsctl.sh infrastructure-autofix --apply --json",
        "cross_sleeve_correlation_governor": "./scripts/ops/opsctl.sh platform-intelligence --json",
        "model_decay_detector": "./scripts/ops/opsctl.sh decay-monitor --json",
        "swap_cpu_capacity_planner": "./scripts/ops/opsctl.sh pressure-relief --apply --json",
    }
    severity = {"blocked": 100, "critical": 95, "degraded": 80, "needs_work": 60, "watch": 45, "thin": 25}.get(status, 10)
    return {
        "section": section,
        "status": status,
        "priority_score": severity,
        "recommended_command": command_by_section.get(section, "./scripts/ops/opsctl.sh health-fast --json"),
    }


def _executive_meta_orchestrator(platform: dict[str, Any], sections: dict[str, dict[str, Any]], pressure: dict[str, Any]) -> dict[str, Any]:
    section_rows = _section_status_rows(sections)
    priorities = sorted(
        [
            _priority_for_section(str(row.get("section")), str(row.get("overall_status")))
            for row in section_rows
            if str(row.get("overall_status")) not in {"ready", "ok", "active"}
            and str(row.get("section")) not in SUMMARY_OR_ALIAS_SECTIONS
        ],
        key=lambda row: (-_safe_float(row.get("priority_score"), 0.0), str(row.get("section"))),
    )
    top_actions = [str(item) for item in _as_list(platform.get("top_actions")) if str(item).strip()]
    next_action = priorities[0]["recommended_command"] if priorities else "./scripts/ops/opsctl.sh health-fast --json"
    mode = "protect_collection" if str(pressure.get("overall_status")) in {"blocked", "degraded"} else "scale_carefully"
    return {
        "overall_status": "ready" if priorities else "ready",
        "mode": mode,
        "next_best_command": next_action,
        "top_platform_actions": top_actions[:8],
        "ranked_priority_count": len(priorities),
        "ranked_priorities": priorities[:12],
        "decision_contract": [
            "read_all_platform_intelligence_sections",
            "prioritize_pressure_provider_quality_and_execution_realism_before_expansion",
            "emit_one_next_best_command_without_live_execution",
        ],
    }


def _causal_world_model(sections: dict[str, dict[str, Any]], pressure: dict[str, Any]) -> dict[str, Any]:
    provider = _as_dict(sections.get("provider_rotation_failover_mesh"))
    backpressure = _as_dict(sections.get("backpressure_prediction_engine"))
    quality = _as_dict(sections.get("bot_data_quality_scores"))
    duplicate = _as_dict(sections.get("duplicate_alpha_overlap_detector"))
    execution = _as_dict(sections.get("execution_paper_trade_realism_layer"))
    edges: list[dict[str, Any]] = []
    if _safe_int(provider.get("degraded_provider_count"), 0) > 0:
        edges.append({"cause": "provider_denial_or_degradation", "effect": "cooldown_or_fallback_collection", "confidence": 0.78})
        edges.append({"cause": "provider_cooldown", "effect": "lower_false_global_halt_risk", "confidence": 0.66})
    if str(backpressure.get("overall_status")) in {"watch", "needs_work"}:
        edges.append({"cause": "queue_backpressure", "effect": "thin_sampling_and_drain_priority", "confidence": 0.74})
    if _safe_int(quality.get("label_counts", {}).get("cold_start"), 0) > 0:
        edges.append({"cause": "cold_start_bot_quality", "effect": "training_deferred_until_collection_floor", "confidence": 0.82})
    if _safe_int(duplicate.get("overlap_cluster_count"), 0) > 0:
        edges.append({"cause": "duplicate_alpha_overlap", "effect": "promotion_requires_novelty_review", "confidence": 0.71})
    if str(execution.get("overall_status")) != "ready":
        edges.append({"cause": "paper_execution_realism_gap", "effect": "paper_pnl_trust_discount", "confidence": 0.69})
    edges.append({"cause": f"pressure_policy_{pressure.get('compute_policy', 'normal')}", "effect": "support_jobs_downshift_before_collection", "confidence": 0.63})
    return {
        "overall_status": "ready" if len(edges) >= 3 else "thin",
        "causal_edge_count": len(edges),
        "causal_edges": edges,
        "current_world_state": {
            "pressure_status": pressure.get("overall_status"),
            "compute_policy": pressure.get("compute_policy"),
            "provider_status": provider.get("overall_status"),
            "data_quality_status": quality.get("overall_status"),
        },
        "world_model_contract": [
            "convert_warnings_into_cause_effect_edges",
            "separate_provider_failures_from_platform_failures",
            "discount_training_and_paper_confidence_when upstream evidence is weak",
        ],
    }


def _experience_memory_core(project_root: Path, platform: dict[str, Any], sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    memory_events_path = project_root / "governance" / "platform_brain_v4" / "experience_memory" / "experience_memory_events.jsonl"
    existing_events = 0
    if memory_events_path.exists():
        try:
            existing_events = sum(1 for _line in memory_events_path.open("r", encoding="utf-8"))
        except Exception:
            existing_events = 0
    event = {
        "timestamp_utc": iso_now(),
        "platform_status": platform.get("overall_status"),
        "bot_count": platform.get("bot_count"),
        "top_actions": _as_list(platform.get("top_actions"))[:5],
        "section_statuses": {key: value.get("overall_status") for key, value in sections.items() if isinstance(value, dict)},
    }
    return {
        "overall_status": "ready",
        "mode": "append_on_apply",
        "memory_event_count_before_apply": existing_events,
        "latest_memory_event": event,
        "memory_store": str(memory_events_path),
        "memory_contract": [
            "store_fix_expansion_pressure_and_outcome_context",
            "keep_memory_compact_jsonl_not_raw_log_heavy",
            "use_memory_for_future_upgrade_and_regression_prioritization",
        ],
    }


def _predictive_expansion_simulator(pressure: dict[str, Any], registry: dict[str, Any], sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base_bots = _safe_int(registry.get("active_bots"), 0)
    host = _safe_float(pressure.get("host_saturation_score"), 0.0)
    storage_index = _safe_float(pressure.get("storage_pressure_index"), 0.0)
    provider_penalty = 12.0 if str(_as_dict(sections.get("provider_rotation_failover_mesh")).get("overall_status")) in {"needs_work", "degraded"} else 0.0
    quality_penalty = 10.0 if str(_as_dict(sections.get("bot_data_quality_scores")).get("overall_status")) == "needs_work" else 0.0
    simulations = []
    for add_bots in (25, 100, 250):
        projected_pressure = min(100.0, host + (add_bots / max(base_bots, 1)) * 35.0 + storage_index * 18.0 + provider_penalty + quality_penalty)
        if projected_pressure >= 85.0:
            recommendation = "defer"
        elif projected_pressure >= 68.0:
            recommendation = "guarded_collect_only"
        else:
            recommendation = "allowed_collect_only"
        simulations.append(
            {
                "additional_bots": add_bots,
                "projected_active_bots": base_bots + add_bots,
                "projected_pressure_score": round(projected_pressure, 3),
                "recommendation": recommendation,
            }
        )
    return {
        "overall_status": "ready",
        "base_active_bots": base_bots,
        "simulations": simulations,
        "expansion_contract": [
            "simulate_cpu_memory_swap_storage_provider_and_quality_impact_before_adding_bots",
            "new_expansion_defaults_to_collect_only",
            "training_and_live_execution_stay_separate_gates",
        ],
    }


def _priority_ranker(sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    priorities = sorted(
        [
            _priority_for_section(key, str(value.get("overall_status") or "missing"))
            for key, value in sections.items()
            if str(value.get("overall_status") or "missing") not in {"ready", "ok", "active"}
            and key not in SUMMARY_OR_ALIAS_SECTIONS
        ],
        key=lambda row: (-_safe_float(row.get("priority_score"), 0.0), str(row.get("section"))),
    )
    buckets = Counter("fix_now" if _safe_float(row.get("priority_score"), 0.0) >= 75 else "monitor" for row in priorities)
    return {
        "overall_status": "ready" if priorities else "ready",
        "priority_count": len(priorities),
        "bucket_counts": dict(sorted(buckets.items())),
        "ranked_priorities": priorities[:20],
        "ranking_contract": [
            "fix_now_for_blocked_degraded_pressure_or_provider_items",
            "monitor_for_needs_work_or_thin_items",
            "ignore_nothing_but_defer_low_impact_noise",
        ],
    }


def _self_upgrade_planner(priority: dict[str, Any]) -> dict[str, Any]:
    plans = []
    for row in _as_list(priority.get("ranked_priorities"))[:10]:
        section = str(row.get("section") or "")
        plans.append(
            {
                "target_section": section,
                "upgrade_type": "regression_guard_plus_control_loop",
                "proposed_change": f"tighten {section} evidence, thresholds, and post-change verification",
                "required_guard": f"tests/test_{section}.py",
                "expected_payoff": "reduce repeated warnings before next expansion",
            }
        )
    return {
        "overall_status": "ready" if plans else "thin",
        "planned_upgrade_count": len(plans),
        "upgrade_plans": plans,
        "upgrade_contract": [
            "every_upgrade_has_expected_payoff",
            "every_upgrade_gets_a_regression_guard",
            "do_not_auto_apply_code_changes_from_brain_without_operator_request",
        ],
    }


def _critic_council(sections: dict[str, dict[str, Any]], pressure: dict[str, Any]) -> dict[str, Any]:
    votes = [
        {
            "critic": "resource_critic",
            "vote": "caution" if str(pressure.get("overall_status")) in {"degraded", "blocked"} else "clear",
            "reason": "host/runtime pressure should shape expansion cadence",
        },
        {
            "critic": "data_critic",
            "vote": "caution" if str(_as_dict(sections.get("bot_data_quality_scores")).get("overall_status")) != "ready" else "clear",
            "reason": "cold-start and probation bots need collection before training",
        },
        {
            "critic": "execution_critic",
            "vote": "caution" if str(_as_dict(sections.get("execution_paper_trade_realism_layer")).get("overall_status")) != "ready" else "clear",
            "reason": "paper PnL needs realism discounts until execution lab is clean",
        },
        {
            "critic": "overlap_critic",
            "vote": "caution" if str(_as_dict(sections.get("duplicate_alpha_overlap_detector")).get("overall_status")) != "ready" else "clear",
            "reason": "duplicate alpha can waste compute and overstate diversification",
        },
        {
            "critic": "autonomy_critic",
            "vote": "clear",
            "reason": "brain remains advisory/read-only and live execution stays separately gated",
        },
    ]
    caution_count = sum(1 for vote in votes if vote["vote"] == "caution")
    hard_pressure = str(pressure.get("overall_status") or "").strip().lower() in {"blocked", "critical"}
    status = "needs_work" if hard_pressure else "watch" if caution_count else "ready"
    return {
        "overall_status": status,
        "critic_count": len(votes),
        "caution_count": caution_count,
        "severity_policy": (
            "blocked_or_critical_pressure_keeps_critic_council_hard"
            if hard_pressure
            else "caution_votes_hold_expansion_without_blocking_guarded_collection_or_paper"
            if caution_count
            else "critics_clear_for_guarded_iteration"
        ),
        "votes": votes,
        "critic_contract": [
            "risk_data_execution_resource_overlap_and_autonomy_critics_review_the_plan",
            "caution_votes_do_not_stop_collection_but_do_block_blind_expansion",
        ],
    }


def _outcome_verification_loop(priority: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for minutes in (15, 60, 480):
        checks.append(
            {
                "checkpoint_minutes": minutes,
                "metrics": ["health_fast_status", "global_halt_state", "backpressure_pending_lines", "provider_degraded_count", "platform_brain_priority_count"],
                "pass_condition": "status_not_worse_and_top_priority_count_not_increased",
            }
        )
    return {
        "overall_status": "ready",
        "checkpoint_count": len(checks),
        "checkpoints": checks,
        "active_priority_count": _safe_int(priority.get("priority_count"), 0),
        "verification_contract": [
            "verify_15min_1h_and_overnight_after_changes",
            "score_fix_effectiveness_against_same_artifacts",
            "write_compact_outcome_memory",
        ],
    }


def _bot_portfolio_economist(registry: dict[str, Any], sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    lifecycle_counts = _as_dict(_as_dict(sections.get("bot_lifecycle_manager")).get("lifecycle_counts"))
    quality = _as_dict(sections.get("bot_data_quality_scores"))
    label_counts = _as_dict(quality.get("label_counts"))
    duplicate = _as_dict(sections.get("duplicate_alpha_overlap_detector"))
    active = _safe_int(registry.get("active_bots"), 0)
    collecting = _safe_int(lifecycle_counts.get("collecting"), _safe_int(registry.get("collecting_bots"), 0))
    trainable = _safe_int(lifecycle_counts.get("trainable"), 0)
    cold_start = _safe_int(label_counts.get("cold_start"), 0)
    overlap_clusters = _safe_int(duplicate.get("overlap_cluster_count"), 0)
    maturity_debt = bool(cold_start > max(active * 0.35, 20) or overlap_clusters)
    return {
        "overall_status": "watch" if maturity_debt else "ready",
        "active_bots": active,
        "collecting_bots": collecting,
        "trainable_bots": trainable,
        "cold_start_bots": cold_start,
        "overlap_cluster_count": overlap_clusters,
        "severity_policy": (
            "cold_start_and_duplicate_alpha_debt_is_soak_watch_debt_while_collection_continues"
            if maturity_debt
            else "portfolio_ready_for_guarded_collection"
        ),
        "portfolio_actions": {
            "protect_compute_for": ["trainable_bots", "high_quality_unique_sleeves", "provider_health_bots"],
            "keep_collecting": ["cold_start_bots", "new_expansion_bots", "deep_awareness_bots"],
            "review_for_merge_or_retire": ["duplicate_alpha_clusters", "inactive_low_quality_legacy_bots"],
        },
        "economist_contract": [
            "compute_is_a_budget",
            "unique_high_quality_bots_get_priority",
            "redundant_or_cold_bots_collect_thin_until_evidence_improves",
        ],
    }


def _data_value_engine(sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    quality = _as_dict(sections.get("bot_data_quality_scores"))
    provider = _as_dict(sections.get("provider_rotation_failover_mesh"))
    execution = _as_dict(sections.get("execution_paper_trade_realism_layer"))
    avg_quality = _safe_float(quality.get("average_quality_score"), 0.0)
    provider_degraded = _safe_int(provider.get("degraded_provider_count"), 0)
    provider_status = str(provider.get("overall_status") or "").strip().lower()
    realism_ready = str(execution.get("overall_status")) == "ready"
    value_score = max(0.0, min(100.0, avg_quality - provider_degraded * 6.0 + (8.0 if realism_ready else -8.0)))
    if value_score < 25.0 and provider_status in {"blocked", "critical"}:
        status = "needs_work"
        severity_policy = "low_data_value_with_hard_provider_failure_requires_repair"
    elif value_score < 55.0:
        status = "watch"
        severity_policy = "low_data_value_is_soak_watch_debt_until_quality_and_realism_mature"
    else:
        status = "ready"
        severity_policy = "data_value_ready_for_guarded_iteration"
    return {
        "overall_status": status,
        "data_value_score": round(value_score, 3),
        "average_bot_quality_score": avg_quality,
        "provider_degraded_count": provider_degraded,
        "provider_status": provider_status,
        "execution_realism_ready": realism_ready,
        "severity_policy": severity_policy,
        "high_value_data_next": ["provider_health", "execution_realism", "label_quality", "source_confidence", "unique_alpha_features"],
        "data_value_contract": [
            "score_data_by_model_usefulness_not_raw_volume",
            "discount_stale_or_disagreed_sources",
            "prioritize_features_that_help_training_readiness_and_execution_realism",
        ],
    }


def _training_scheduler_brain(sections: dict[str, dict[str, Any]], pressure: dict[str, Any]) -> dict[str, Any]:
    readiness = _as_dict(sections.get("training_readiness_board"))
    train_allowed = _safe_int(readiness.get("train_allowed_count"), 0)
    sample_debt = _safe_int(readiness.get("sample_debt_count"), 0)
    pressure_status = str(pressure.get("overall_status") or "missing")
    if pressure_status == "blocked":
        policy = "paused"
    elif pressure_status == "degraded":
        policy = "off_hours_micro_batches"
    elif train_allowed > 0:
        policy = "small_batches_allowed"
    else:
        policy = "collect_more_data"
    return {
        "overall_status": "ready",
        "training_policy": policy,
        "train_allowed_count": train_allowed,
        "sample_debt_count": sample_debt,
        "preferred_window": "quiet_window_21_to_06_local",
        "scheduler_contract": [
            "train_only_when_data_readiness_and_pressure_clear",
            "prefer_off_hours_when_operator_or_creative_apps_are_open",
            "do_not_train_new_bots_until_collection_floor_clears",
        ],
    }


def _operator_intent_model(project_root: Path, pressure: dict[str, Any]) -> dict[str, Any]:
    creative = _health(project_root, "creative_cotenant_guard_latest.json")
    memory = _health(project_root, "memory_efficiency_control_latest.json")
    local_hour = datetime.now().hour
    creative_status = str(creative.get("overall_status") or "")
    memory_status = str(memory.get("overall_status") or "")
    if creative_status in {"needs_work", "degraded"} or memory_status in {"needs_work", "degraded"}:
        inferred_mode = "foreground_app_headroom"
    elif local_hour >= 21 or local_hour < 6:
        inferred_mode = "overnight_collection"
    elif str(pressure.get("overall_status")) == "degraded":
        inferred_mode = "calm_expansion_review"
    else:
        inferred_mode = "normal_operator_mode"
    return {
        "overall_status": "ready",
        "inferred_operator_mode": inferred_mode,
        "local_hour": local_hour,
        "creative_cotenant_status": creative_status or "missing",
        "memory_efficiency_status": memory_status or "missing",
        "runtime_intensity": "calm" if inferred_mode in {"foreground_app_headroom", "calm_expansion_review"} else "normal",
        "operator_contract": [
            "preserve_headroom_for_pycharm_browser_logic_final_cut_modes",
            "overnight_can_collect_more_but_training_still_needs_gates",
            "operator_intent_shapes_support_jobs_not_safety_rules",
        ],
    }


def _env_overrides(payload: dict[str, Any]) -> dict[str, str]:
    executive = _as_dict(_as_dict(payload.get("sections")).get("executive_meta_orchestrator"))
    training = _as_dict(_as_dict(payload.get("sections")).get("training_scheduler_brain"))
    operator = _as_dict(_as_dict(payload.get("sections")).get("operator_intent_model"))
    env = {
        "PLATFORM_BRAIN_V4_ENABLED": "1",
        "PLATFORM_BRAIN_V4_MODE": "advisory_read_only",
        "PLATFORM_BRAIN_V4_SECTION_COUNT": str(len(SECTION_KEYS)),
        "PLATFORM_BRAIN_V4_NEXT_BEST_COMMAND": str(executive.get("next_best_command") or ""),
        "PLATFORM_BRAIN_V4_TRAINING_POLICY": str(training.get("training_policy") or "collect_more_data"),
        "PLATFORM_BRAIN_V4_OPERATOR_MODE": str(operator.get("inferred_operator_mode") or "normal_operator_mode"),
        "PRIMARY_ML_RUNTIME_BACKEND": "mlx",
        "LIBRARY_DEFAULT_ML_BACKEND": "mlx",
        "ALLOW_ORDER_EXECUTION": "0",
        "PAPER_TRADE_LOCK": "1",
    }
    for control in PLATFORM_BRAIN_CONTROLS:
        env[control["env_key"]] = "1"
    return env


def _write_env_override(path: Path, env: dict[str, str]) -> bool:
    lines = ["# Auto-managed by scripts/ops/platform_brain_v4.py"]
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
        "layer": "platform_brain_v4_grande",
        "section_keys": list(SECTION_KEYS),
        "controls": PLATFORM_BRAIN_CONTROLS,
        "artifacts": payload.get("section_artifacts", {}),
        "recommended_commands": payload.get("recommended_commands", []),
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
    platform = _platform_payload(project_root)
    platform_sections = _platform_sections(platform)
    pressure = _pressure_snapshot(project_root, platform)
    registry = _registry_summary(project_root)

    executive = _executive_meta_orchestrator(platform, platform_sections, pressure)
    causal = _causal_world_model(platform_sections, pressure)
    memory = _experience_memory_core(project_root, platform, platform_sections)
    simulator = _predictive_expansion_simulator(pressure, registry, platform_sections)
    priority = _priority_ranker(platform_sections)
    upgrade = _self_upgrade_planner(priority)
    critics = _critic_council(platform_sections, pressure)
    verification = _outcome_verification_loop(priority)
    economist = _bot_portfolio_economist(registry, platform_sections)
    data_value = _data_value_engine(platform_sections)
    training = _training_scheduler_brain(platform_sections, pressure)
    operator = _operator_intent_model(project_root, pressure)

    sections = {
        "executive_meta_orchestrator": executive,
        "causal_world_model": causal,
        "experience_memory_core_v2": memory,
        "predictive_expansion_simulator": simulator,
        "autonomous_priority_ranker": priority,
        "self_upgrade_planner": upgrade,
        "critic_council": critics,
        "outcome_verification_loop": verification,
        "bot_portfolio_economist": economist,
        "data_value_engine": data_value,
        "training_scheduler_brain": training,
        "operator_intent_model": operator,
    }
    status_rows = _section_status_rows(sections)
    overall = _worst_status(status_rows)
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall in {"ready", "watch", "needs_work", "degraded"},
        "overall_status": overall,
        "mode": "advisory_read_only_decision_brain",
        "brain_name": "Platform Brain v4 Grande",
        "section_count": len(SECTION_KEYS),
        "section_keys": list(SECTION_KEYS),
        "control_count": len(PLATFORM_BRAIN_CONTROLS),
        "controls": [{**control, "enabled": True} for control in PLATFORM_BRAIN_CONTROLS],
        "registry_summary": registry,
        "pressure_snapshot": pressure,
        "platform_intelligence_status": platform.get("overall_status", "missing"),
        "sections": sections,
        "section_statuses": status_rows,
        "recommended_env_overrides": {},
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "platform-brain-v4", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "platform-intelligence", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "health-fast", "--json"],
            ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"],
        ],
        "source_files": {
            "platform_intelligence": str(project_root / "governance" / "health" / "platform_intelligence_expansion_latest.json"),
            "master_bot_registry": str(project_root / "master_bot_registry.json"),
            "primary_artifact": str(DEFAULT_OUT_PATH),
        },
    }
    payload["recommended_env_overrides"] = _env_overrides(payload)
    return payload


def write_section_artifacts(project_root: Path, payload: dict[str, Any]) -> dict[str, str]:
    root = project_root / "governance" / "platform_brain_v4"
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
    parser = argparse.ArgumentParser(description="Build Platform Brain v4 Grande advisory decision brain.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--memory-events-file", default=str(DEFAULT_MEMORY_EVENTS_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root)
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    written = write_section_artifacts(project_root, payload)
    payload["section_artifacts"] = written
    if args.apply:
        env = _as_dict(payload.get("recommended_env_overrides"))
        memory_event = _as_dict(_as_dict(payload.get("sections")).get("experience_memory_core_v2")).get("latest_memory_event")
        payload["apply_result"] = {
            "applied": True,
            "override_path": str(Path(args.override_file).expanduser()),
            "override_changed": _write_env_override(Path(args.override_file).expanduser(), {str(k): str(v) for k, v in env.items()}),
            "config_path": str(Path(args.config_file).expanduser()),
            "config_changed": _write_config(Path(args.config_file).expanduser(), payload),
            "memory_events_path": str(Path(args.memory_events_file).expanduser()),
            "memory_event_appended": _append_memory_event(Path(args.memory_events_file).expanduser(), _as_dict(memory_event)),
        }
    else:
        payload["apply_result"] = {"applied": False}
    write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "platform_brain_v4 "
            f"overall_status={payload.get('overall_status')} "
            f"sections={payload.get('section_count')} "
            f"next_best_command={_as_dict(_as_dict(payload.get('sections')).get('executive_meta_orchestrator')).get('next_best_command', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
