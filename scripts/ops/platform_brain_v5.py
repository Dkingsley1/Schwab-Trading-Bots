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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, status_rank, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, status_rank, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "platform_brain_v5_latest.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "platform_brain_v5_reflex.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.platform_brain_v5_override"
DEFAULT_REFLEX_EVENTS_PATH = PROJECT_ROOT / "governance" / "platform_brain_v5" / "reflex_memory" / "reflex_events.jsonl"

SECTION_KEYS: tuple[str, ...] = (
    "temporal_self_model",
    "reflex_action_router",
    "regret_and_outcome_ledger",
    "scenario_rehearsal_lab",
    "adaptive_cadence_controller",
    "safe_autonomy_boundary",
    "critic_ensemble_fusion",
    "resource_budget_market_maker",
    "data_contract_negotiator",
    "bot_curriculum_builder",
    "dependency_reflex_map",
    "strategic_roadmap_synthesizer",
)

CONTROLS: tuple[dict[str, str], ...] = (
    {"id": "temporal_self_model", "title": "Temporal self-model", "env_key": "PLATFORM_BRAIN_V5_TEMPORAL_SELF_MODEL_ENABLED"},
    {"id": "reflex_action_router", "title": "Reflex action router", "env_key": "PLATFORM_BRAIN_V5_REFLEX_ROUTER_ENABLED"},
    {"id": "regret_and_outcome_ledger", "title": "Regret and outcome ledger", "env_key": "PLATFORM_BRAIN_V5_REGRET_LEDGER_ENABLED"},
    {"id": "scenario_rehearsal_lab", "title": "Scenario rehearsal lab", "env_key": "PLATFORM_BRAIN_V5_SCENARIO_REHEARSAL_ENABLED"},
    {"id": "adaptive_cadence_controller", "title": "Adaptive cadence controller", "env_key": "PLATFORM_BRAIN_V5_CADENCE_CONTROLLER_ENABLED"},
    {"id": "safe_autonomy_boundary", "title": "Safe autonomy boundary", "env_key": "PLATFORM_BRAIN_V5_SAFE_AUTONOMY_ENABLED"},
    {"id": "critic_ensemble_fusion", "title": "Critic ensemble fusion", "env_key": "PLATFORM_BRAIN_V5_CRITIC_FUSION_ENABLED"},
    {"id": "resource_budget_market_maker", "title": "Resource budget market maker", "env_key": "PLATFORM_BRAIN_V5_RESOURCE_MARKET_ENABLED"},
    {"id": "data_contract_negotiator", "title": "Data contract negotiator", "env_key": "PLATFORM_BRAIN_V5_DATA_CONTRACT_ENABLED"},
    {"id": "bot_curriculum_builder", "title": "Bot curriculum builder", "env_key": "PLATFORM_BRAIN_V5_BOT_CURRICULUM_ENABLED"},
    {"id": "dependency_reflex_map", "title": "Dependency reflex map", "env_key": "PLATFORM_BRAIN_V5_DEPENDENCY_REFLEX_ENABLED"},
    {"id": "strategic_roadmap_synthesizer", "title": "Strategic roadmap synthesizer", "env_key": "PLATFORM_BRAIN_V5_ROADMAP_ENABLED"},
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


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on", "active", "enabled"}


def _health(project_root: Path, name: str) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / name)
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path, *, limit: int = 200) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    return rows[-max(limit, 1):]


def _status_rows(sections: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key, value in sorted(sections.items()):
        status = str(value.get("overall_status") or "missing")
        rows.append({"section": key, "overall_status": status, "rank": status_rank(status)})
    return rows


def _worst_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "missing"
    if any(str(row.get("overall_status")) in {"blocked", "critical"} for row in rows):
        return "blocked"
    if any(str(row.get("overall_status")) == "degraded" for row in rows):
        return "degraded"
    if any(str(row.get("overall_status")) == "needs_work" for row in rows):
        return "needs_work"
    if any(str(row.get("overall_status")) in {"watch", "thin"} for row in rows):
        return "watch"
    return "ready"


def _registry_summary(project_root: Path) -> dict[str, Any]:
    payload = load_json(project_root / "master_bot_registry.json")
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    active = sum(1 for row in rows if isinstance(row, dict) and _bool(row.get("active")))
    collecting = sum(
        1
        for row in rows
        if isinstance(row, dict) and (_bool(row.get("data_collection_active")) or str(row.get("lifecycle_state") or "") == "data_collection_only")
    )
    excluded = sum(1 for row in rows if isinstance(row, dict) and (_bool(row.get("training_excluded")) or _bool(row.get("exclude_from_training"))))
    return {"total_bots": len(rows), "active_bots": active, "collecting_bots": collecting, "training_excluded_bots": excluded}


def _v4(project_root: Path) -> dict[str, Any]:
    return _health(project_root, "platform_brain_v4_latest.json")


def _platform(project_root: Path) -> dict[str, Any]:
    return _health(project_root, "platform_intelligence_expansion_latest.json")


def _temporal_self_model(project_root: Path, v4: dict[str, Any], platform: dict[str, Any]) -> dict[str, Any]:
    v4_events = _read_jsonl(project_root / "governance" / "platform_brain_v4" / "experience_memory" / "experience_memory_events.jsonl")
    v5_events = _read_jsonl(project_root / "governance" / "platform_brain_v5" / "reflex_memory" / "reflex_events.jsonl")
    statuses = [str(event.get("platform_status") or "") for event in v4_events if event.get("platform_status")]
    repeated = Counter()
    for event in v4_events:
        for action in _as_list(event.get("top_actions")):
            repeated[str(action)] += 1
    return {
        "overall_status": "ready" if v4 else "thin",
        "v4_status": v4.get("overall_status", "missing"),
        "platform_intelligence_status": platform.get("overall_status", "missing"),
        "v4_memory_event_count": len(v4_events),
        "v5_reflex_event_count": len(v5_events),
        "recent_status_counts": dict(sorted(Counter(statuses).items())),
        "repeated_action_themes": [{"action": key, "count": count} for key, count in repeated.most_common(8)],
        "temporal_contract": [
            "compare_current_brain_state_against_recent_memory",
            "detect_repeated_actions_before_they_become_stale_advice",
            "prefer_trend_awareness_over_single_snapshot_reaction",
        ],
    }


def _reflex_action_router(v4: dict[str, Any]) -> dict[str, Any]:
    executive = _as_dict(_as_dict(v4.get("sections")).get("executive_meta_orchestrator"))
    priority = _as_dict(_as_dict(v4.get("sections")).get("autonomous_priority_ranker"))
    next_best = str(executive.get("next_best_command") or "./scripts/ops/opsctl.sh health-fast --json")
    allowed_prefixes = (
        "./scripts/ops/opsctl.sh health-fast",
        "./scripts/ops/opsctl.sh pressure-relief",
        "./scripts/ops/opsctl.sh platform-intelligence",
        "./scripts/ops/opsctl.sh platform-brain-v4",
        "./scripts/ops/opsctl.sh storage-backpressure-autopilot",
    )
    reflexes = []
    for row in _as_list(priority.get("ranked_priorities"))[:8]:
        command = str(row.get("recommended_command") or "")
        auto_allowed = command.startswith(allowed_prefixes)
        reflexes.append(
            {
                "section": row.get("section"),
                "priority_score": row.get("priority_score"),
                "command": command,
                "auto_allowed": auto_allowed,
                "mode": "safe_advisory" if auto_allowed else "operator_review",
            }
        )
    return {
        "overall_status": "ready",
        "next_best_command": next_best,
        "safe_reflex_count": sum(1 for row in reflexes if row["auto_allowed"]),
        "operator_review_count": sum(1 for row in reflexes if not row["auto_allowed"]),
        "reflex_queue": reflexes,
        "reflex_contract": [
            "only_route_safe_read_only_or_guarded_pressure_commands",
            "never_clear_halts_or_start_live_execution_as_a_reflex",
            "operator_review_for_training_auth_decay_and_execution_lab_actions",
        ],
    }


def _regret_ledger(project_root: Path, temporal: dict[str, Any], v4: dict[str, Any]) -> dict[str, Any]:
    priority = _as_dict(_as_dict(v4.get("sections")).get("autonomous_priority_ranker"))
    priority_count = _safe_int(priority.get("priority_count"), 0)
    priority_rows = _as_list(priority.get("ranked_priorities"))
    hard_priority_count = sum(
        1
        for row in priority_rows
        if str(_as_dict(row).get("status") or "").strip().lower() in {"blocked", "critical", "degraded", "failed", "fatal"}
    )
    memory_count = _safe_int(temporal.get("v5_reflex_event_count"), 0)
    repeated_count = len(_as_list(temporal.get("repeated_action_themes")))
    regret_score = min(100.0, priority_count * 6.0 + repeated_count * 4.0)
    v4_status = str(v4.get("overall_status") or "").strip().lower()
    current_ready_clears_historical_regret = bool(v4_status == "ready" and hard_priority_count == 0)
    if current_ready_clears_historical_regret:
        status = "ready"
        severity_policy = "historical_advisory_regret_cleared_by_current_ready_state"
    elif regret_score >= 50.0 and hard_priority_count:
        status = "needs_work"
        severity_policy = "high_regret_with_hard_priority_rows_requires_operator_repair"
    elif regret_score >= 50.0:
        status = "watch"
        severity_policy = "high_regret_from_advisory_repetition_is_soak_watch_debt"
    else:
        status = "ready"
        severity_policy = "regret_within_guarded_soak_budget"
    event = {
        "timestamp_utc": iso_now(),
        "priority_count": priority_count,
        "hard_priority_count": hard_priority_count,
        "v4_status": v4.get("overall_status"),
        "regret_score": round(regret_score, 3),
        "next_best_command": _as_dict(_as_dict(v4.get("sections")).get("executive_meta_orchestrator")).get("next_best_command"),
    }
    return {
        "overall_status": status,
        "mode": "append_on_apply",
        "reflex_memory_event_count_before_apply": memory_count,
        "regret_score": round(regret_score, 3),
        "hard_priority_count": hard_priority_count,
        "current_ready_clears_historical_regret": current_ready_clears_historical_regret,
        "managed_historical_advisory_count": repeated_count if current_ready_clears_historical_regret else 0,
        "severity_policy": severity_policy,
        "latest_reflex_event": event,
        "ledger_contract": [
            "track_unresolved_priority_count_as_regret",
            "append_compact_reflex_events_on_apply",
            "measure_whether_recommendations_reduce_future_priority_count",
        ],
    }


def _scenario_rehearsal(v4: dict[str, Any]) -> dict[str, Any]:
    simulator = _as_dict(_as_dict(v4.get("sections")).get("predictive_expansion_simulator"))
    expansion_recs = [str(row.get("recommendation")) for row in _as_list(simulator.get("simulations")) if isinstance(row, dict)]
    scenarios = [
        {"scenario": "add_25_bots_now", "recommendation": "defer" if "defer" in expansion_recs else "guarded_collect_only"},
        {"scenario": "run_heavy_feed_view_now", "recommendation": "ttl_limited_only"},
        {"scenario": "train_now", "recommendation": str(_as_dict(_as_dict(v4.get("sections")).get("training_scheduler_brain")).get("training_policy") or "collect_more_data")},
        {"scenario": "provider_403_429_burst", "recommendation": "cooldown_and_cache_fallback"},
        {"scenario": "external_drive_reconnect", "recommendation": "verify_route_then_resume_drain"},
    ]
    return {
        "overall_status": "ready",
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "rehearsal_contract": [
            "rehearse_before_expanding",
            "prefer_defer_or_collect_only_when_pressure_is_elevated",
            "treat_provider_denial_as_cooldown_not_platform_failure",
        ],
    }


def _adaptive_cadence(v4: dict[str, Any]) -> dict[str, Any]:
    operator = _as_dict(_as_dict(v4.get("sections")).get("operator_intent_model"))
    mode = str(operator.get("inferred_operator_mode") or "normal_operator_mode")
    if mode == "foreground_app_headroom":
        cadence = {"health_seconds": 90, "platform_brain_seconds": 300, "heavy_reports": "off_hours_only", "training": "off_hours_micro_batches"}
    elif mode == "overnight_collection":
        cadence = {"health_seconds": 60, "platform_brain_seconds": 240, "heavy_reports": "allowed_ttl", "training": "small_batches_if_clear"}
    else:
        cadence = {"health_seconds": 60, "platform_brain_seconds": 300, "heavy_reports": "manual_ttl", "training": "gate_controlled"}
    return {
        "overall_status": "ready",
        "operator_mode": mode,
        "cadence": cadence,
        "cadence_contract": [
            "operator_mode_shapes_cadence",
            "foreground_apps_keep_headroom",
            "brain_refreshes_stay_lightweight",
        ],
    }


def _safe_autonomy_boundary(v4: dict[str, Any], reflex: dict[str, Any]) -> dict[str, Any]:
    blocked_actions = [
        "live_order_execution",
        "manual_halt_force_clear",
        "credential_entry_or_storage",
        "training_start_without_readiness_gate",
        "destructive_cleanup_without_operator_request",
    ]
    return {
        "overall_status": "ready",
        "autonomy_mode": "advisory_read_only_safe_reflexes",
        "paper_trade_lock_required": True,
        "live_execution_allowed": False,
        "safe_reflex_count": reflex.get("safe_reflex_count", 0),
        "blocked_actions": blocked_actions,
        "boundary_contract": [
            "safe_reflexes_can_recommend_guarded_commands",
            "execution_auth_halt_and_destructive_actions_stay_operator_gated",
            "mlx_remains_default_backend",
        ],
    }


def _critic_fusion(v4: dict[str, Any], reflex: dict[str, Any]) -> dict[str, Any]:
    council = _as_dict(_as_dict(v4.get("sections")).get("critic_council"))
    council_status = str(council.get("overall_status") or "").strip().lower()
    caution_count = _safe_int(council.get("caution_count"), 0)
    votes = _as_list(council.get("votes"))
    hard_vote_count = sum(1 for row in votes if str(_as_dict(row).get("vote") or "").strip().lower() in {"block", "blocked", "critical"})
    fusion_vote = "hold_expansion" if caution_count >= 3 else "allow_guarded_iteration"
    if council_status in {"blocked", "critical", "failed", "fatal"} or hard_vote_count:
        status = "needs_work"
        severity_policy = "blocked_or_critical_critic_votes_require_operator_repair"
    elif caution_count >= 3:
        status = "watch"
        severity_policy = "caution_votes_hold_expansion_without_degrading_guarded_collection_or_paper"
    else:
        status = "ready"
        severity_policy = "critics_clear_for_guarded_iteration"
    return {
        "overall_status": status,
        "fusion_vote": fusion_vote,
        "critic_count": len(votes),
        "caution_count": caution_count,
        "hard_vote_count": hard_vote_count,
        "severity_policy": severity_policy,
        "safe_reflex_count": reflex.get("safe_reflex_count", 0),
        "critic_contract": [
            "fuse_resource_data_execution_overlap_and_autonomy_critics",
            "hold_expansion_when_multiple_critics_warn",
            "allow_safe_reflexes_even_when_expansion_is_held",
        ],
    }


def _resource_budget(v4: dict[str, Any]) -> dict[str, Any]:
    pressure = _as_dict(v4.get("pressure_snapshot"))
    status = str(pressure.get("overall_status") or "")
    if status == "blocked":
        budgets = {"live_collection": 70, "sql_drain": 20, "reports": 5, "training": 0, "research": 5}
    elif status == "degraded":
        budgets = {"live_collection": 58, "sql_drain": 22, "reports": 6, "training": 4, "research": 10}
    else:
        budgets = {"live_collection": 45, "sql_drain": 20, "reports": 10, "training": 10, "research": 15}
    return {
        "overall_status": "ready",
        "pressure_status": status or "missing",
        "resource_budget_percent": budgets,
        "market_contract": [
            "allocate_compute_like_a_budget",
            "protect_collection_and_drain_first",
            "reports_training_and_research_buy_remaining_capacity",
        ],
    }


def _data_contract(v4: dict[str, Any], platform: dict[str, Any]) -> dict[str, Any]:
    data_value = _as_dict(_as_dict(v4.get("sections")).get("data_value_engine"))
    provider = _as_dict(_as_dict(v4.get("sections")).get("causal_world_model")).get("current_world_state", {})
    score = _safe_float(data_value.get("data_value_score"), 0.0)
    if not v4:
        status = "thin"
    elif score < 25.0 and str(_as_dict(provider).get("provider_status") or "") in {"blocked", "critical"}:
        status = "needs_work"
    elif score < 55.0:
        status = "watch"
    else:
        status = "ready"
    contracts = [
        {"source_family": "provider_health", "priority": "high", "reason": "reduces false halts and failed collection"},
        {"source_family": "execution_realism", "priority": "high", "reason": "discounts paper PnL before promotion"},
        {"source_family": "label_quality", "priority": "high", "reason": "unblocks training readiness"},
        {"source_family": "unique_alpha_features", "priority": "medium", "reason": "reduces duplicate-alpha load"},
    ]
    return {
        "overall_status": status,
        "data_value_score": round(score, 3),
        "provider_status": _as_dict(provider).get("provider_status", "unknown"),
        "platform_intelligence_status": platform.get("overall_status", "missing"),
        "data_contracts": contracts,
        "contract_policy": "collect_high_value_quality_evidence_before_more_volume",
    }


def _bot_curriculum(v4: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    economist = _as_dict(_as_dict(v4.get("sections")).get("bot_portfolio_economist"))
    readiness = _as_dict(_as_dict(v4.get("sections")).get("training_scheduler_brain"))
    curriculum = [
        {"cohort": "trainable", "count": _safe_int(economist.get("trainable_bots"), 0), "assignment": "protect_for_off_hours_micro_batch_review"},
        {"cohort": "cold_start", "count": _safe_int(economist.get("cold_start_bots"), 0), "assignment": "thin_collect_until_quality_floor"},
        {"cohort": "duplicate_overlap", "count": _safe_int(economist.get("overlap_cluster_count"), 0), "assignment": "novelty_review_before_promotion"},
        {"cohort": "collecting", "count": _safe_int(registry.get("collecting_bots"), 0), "assignment": "continue_live_data_and_paper_only"},
    ]
    return {
        "overall_status": "ready",
        "training_policy": readiness.get("training_policy", "collect_more_data"),
        "curriculum": curriculum,
        "curriculum_contract": [
            "separate_bot_cohorts_by_readiness_and_compute_value",
            "do_not_train_cold_start_bots",
            "turn_duplicate_clusters_into_merge_or_novelty_review",
        ],
    }


def _dependency_map(v4: dict[str, Any]) -> dict[str, Any]:
    priority = _as_dict(_as_dict(v4.get("sections")).get("autonomous_priority_ranker"))
    dependencies = []
    for row in _as_list(priority.get("ranked_priorities"))[:10]:
        command = str(row.get("recommended_command") or "")
        dependencies.append(
            {
                "section": row.get("section"),
                "command": command,
                "dependency_type": "opsctl_command" if command.startswith("./scripts/ops/opsctl.sh") else "manual_or_external",
                "operator_review": not command.startswith("./scripts/ops/opsctl.sh"),
            }
        )
    return {
        "overall_status": "ready",
        "dependency_count": len(dependencies),
        "dependencies": dependencies,
        "dependency_contract": [
            "map_recommendations_to_commands_and_artifacts",
            "flag_manual_or_external_dependencies",
            "prefer_existing_opsctl_surfaces_over_new_one_off_work",
        ],
    }


def _roadmap(v4: dict[str, Any], reflex: dict[str, Any], scenarios: dict[str, Any]) -> dict[str, Any]:
    steps = [
        {"step": 1, "title": "Stabilize pressure and provider cooldown surfaces", "command": reflex.get("next_best_command")},
        {"step": 2, "title": "Improve data quality and duplicate-alpha review before more bots", "command": "./scripts/ops/opsctl.sh platform-intelligence --json"},
        {"step": 3, "title": "Rehearse expansion again after pressure and provider warnings clear", "command": "./scripts/ops/opsctl.sh platform-brain-v5 --json"},
    ]
    expansion_allowed = not any(str(row.get("recommendation")) == "defer" for row in _as_list(scenarios.get("scenarios")) if "add_" in str(row.get("scenario")))
    return {
        "overall_status": "ready",
        "expansion_allowed_now": expansion_allowed,
        "roadmap": steps,
        "roadmap_contract": [
            "stabilize_then_evaluate_then_expand",
            "brain_can_plan_but_operator_approves_implementation",
            "all_future_expansion_rehearsed_before_apply",
        ],
    }


def _env_overrides(payload: dict[str, Any]) -> dict[str, str]:
    sections = _as_dict(payload.get("sections"))
    reflex = _as_dict(sections.get("reflex_action_router"))
    cadence = _as_dict(sections.get("adaptive_cadence_controller"))
    roadmap = _as_dict(sections.get("strategic_roadmap_synthesizer"))
    env = {
        "PLATFORM_BRAIN_V5_ENABLED": "1",
        "PLATFORM_BRAIN_V5_MODE": "reflex_advisory_read_only",
        "PLATFORM_BRAIN_V5_SECTION_COUNT": str(len(SECTION_KEYS)),
        "PLATFORM_BRAIN_V5_NEXT_BEST_COMMAND": str(reflex.get("next_best_command") or ""),
        "PLATFORM_BRAIN_V5_EXPANSION_ALLOWED_NOW": "1" if _bool(roadmap.get("expansion_allowed_now")) else "0",
        "PLATFORM_BRAIN_V5_HEALTH_SECONDS": str(_as_dict(cadence.get("cadence")).get("health_seconds", "90")),
        "PRIMARY_ML_RUNTIME_BACKEND": "mlx",
        "LIBRARY_DEFAULT_ML_BACKEND": "mlx",
        "PAPER_TRADE_LOCK": "1",
        "ALLOW_ORDER_EXECUTION": "0",
    }
    for control in CONTROLS:
        env[control["env_key"]] = "1"
    return env


def _write_env_override(path: Path, env: dict[str, str]) -> bool:
    lines = ["# Auto-managed by scripts/ops/platform_brain_v5.py"]
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
        "layer": "platform_brain_v5_reflex_cortex",
        "section_keys": list(SECTION_KEYS),
        "controls": CONTROLS,
        "artifacts": payload.get("section_artifacts", {}),
    }
    content = json.dumps(config, ensure_ascii=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _append_reflex_event(path: Path, event: dict[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
    return True


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    v4 = _v4(project_root)
    platform = _platform(project_root)
    registry = _registry_summary(project_root)
    temporal = _temporal_self_model(project_root, v4, platform)
    reflex = _reflex_action_router(v4)
    regret = _regret_ledger(project_root, temporal, v4)
    scenarios = _scenario_rehearsal(v4)
    cadence = _adaptive_cadence(v4)
    boundary = _safe_autonomy_boundary(v4, reflex)
    critics = _critic_fusion(v4, reflex)
    resources = _resource_budget(v4)
    data_contracts = _data_contract(v4, platform)
    curriculum = _bot_curriculum(v4, registry)
    dependencies = _dependency_map(v4)
    roadmap = _roadmap(v4, reflex, scenarios)
    sections = {
        "temporal_self_model": temporal,
        "reflex_action_router": reflex,
        "regret_and_outcome_ledger": regret,
        "scenario_rehearsal_lab": scenarios,
        "adaptive_cadence_controller": cadence,
        "safe_autonomy_boundary": boundary,
        "critic_ensemble_fusion": critics,
        "resource_budget_market_maker": resources,
        "data_contract_negotiator": data_contracts,
        "bot_curriculum_builder": curriculum,
        "dependency_reflex_map": dependencies,
        "strategic_roadmap_synthesizer": roadmap,
    }
    rows = _status_rows(sections)
    overall = _worst_status(rows)
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall in {"ready", "watch", "needs_work", "degraded"},
        "overall_status": overall,
        "brain_name": "Platform Brain v5 Reflex Cortex",
        "mode": "reflex_advisory_read_only",
        "section_count": len(SECTION_KEYS),
        "section_keys": list(SECTION_KEYS),
        "control_count": len(CONTROLS),
        "controls": [{**control, "enabled": True} for control in CONTROLS],
        "registry_summary": registry,
        "v4_status": v4.get("overall_status", "missing"),
        "platform_intelligence_status": platform.get("overall_status", "missing"),
        "sections": sections,
        "section_statuses": rows,
        "recommended_env_overrides": {},
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "platform-brain-v5", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "platform-brain-v4", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "platform-intelligence", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"],
        ],
        "source_files": {
            "platform_brain_v4": str(project_root / "governance" / "health" / "platform_brain_v4_latest.json"),
            "platform_intelligence": str(project_root / "governance" / "health" / "platform_intelligence_expansion_latest.json"),
            "primary_artifact": str(DEFAULT_OUT_PATH),
        },
    }
    payload["recommended_env_overrides"] = _env_overrides(payload)
    return payload


def write_section_artifacts(project_root: Path, payload: dict[str, Any]) -> dict[str, str]:
    root = project_root / "governance" / "platform_brain_v5"
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
    parser = argparse.ArgumentParser(description="Build Platform Brain v5 Reflex Cortex advisory layer.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--reflex-events-file", default=str(DEFAULT_REFLEX_EVENTS_PATH))
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
        event = _as_dict(_as_dict(payload.get("sections")).get("regret_and_outcome_ledger")).get("latest_reflex_event")
        payload["apply_result"] = {
            "applied": True,
            "override_path": str(Path(args.override_file).expanduser()),
            "override_changed": _write_env_override(Path(args.override_file).expanduser(), {str(k): str(v) for k, v in env.items()}),
            "config_path": str(Path(args.config_file).expanduser()),
            "config_changed": _write_config(Path(args.config_file).expanduser(), payload),
            "reflex_events_path": str(Path(args.reflex_events_file).expanduser()),
            "reflex_event_appended": _append_reflex_event(Path(args.reflex_events_file).expanduser(), _as_dict(event)),
        }
    else:
        payload["apply_result"] = {"applied": False}
    write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        reflex = _as_dict(_as_dict(payload.get("sections")).get("reflex_action_router"))
        print(
            "platform_brain_v5 "
            f"overall_status={payload.get('overall_status')} "
            f"sections={payload.get('section_count')} "
            f"next_best_command={reflex.get('next_best_command', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
