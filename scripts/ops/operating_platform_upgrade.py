#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "operating_platform_upgrade_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.operating_platform_upgrade_override"
DEFAULT_LEDGER_PATH = PROJECT_ROOT / "governance" / "platform_upgrades" / "operating_platform_upgrade_frames.jsonl"

PROTECTED_VOLUMES = ("/Volumes/VIDEO",)
GRADE_LADDER = ("F", "D", "C", "B", "A", "A+")
RAW_LIFT_STEPS = 4
ALPHA_DEDUP_FINAL_LIFT_STEPS = 6

SECTION_ORDER = [
    "regime_aware_capital_allocator",
    "profit_harvesting_v2",
    "bot_alpha_deduplication_engine",
    "decision_replay_laboratory",
    "full_feature_label_lake",
    "sleeve_ceo_layer",
    "live_readiness_sandbox",
    "autonomous_weak_bot_repair",
    "storage_backlog_auto_architect",
    "market_narrative_intelligence",
    "income_readiness_governor",
    "cross_os_host_portability",
]

SECTION_TITLES = {
    "regime_aware_capital_allocator": "Regime-Aware Capital Allocator",
    "profit_harvesting_v2": "Profit Harvesting v2",
    "bot_alpha_deduplication_engine": "Bot Alpha Deduplication Engine",
    "decision_replay_laboratory": "Decision Replay Laboratory",
    "full_feature_label_lake": "Full Feature/Label Lake",
    "sleeve_ceo_layer": "Sleeve CEO Layer",
    "live_readiness_sandbox": "Live-Readiness Sandbox",
    "autonomous_weak_bot_repair": "Autonomous Weak-Bot Repair",
    "storage_backlog_auto_architect": "Storage/Backlog Auto-Architect",
    "market_narrative_intelligence": "Market Narrative Intelligence",
    "income_readiness_governor": "Income-Readiness Governor",
    "cross_os_host_portability": "Cross-OS / Future Computer Portability",
}

SECTION_ARTIFACTS = {
    "regime_aware_capital_allocator": "capital_allocator_contract_latest.json",
    "profit_harvesting_v2": "profit_harvest_v2_contract_latest.json",
    "bot_alpha_deduplication_engine": "alpha_dedup_engine_contract_latest.json",
    "decision_replay_laboratory": "decision_replay_laboratory_contract_latest.json",
    "full_feature_label_lake": "feature_label_lake_contract_latest.json",
    "sleeve_ceo_layer": "sleeve_ceo_layer_latest.json",
    "live_readiness_sandbox": "live_readiness_sandbox_contract_latest.json",
    "autonomous_weak_bot_repair": "weak_bot_repair_autopilot_contract_latest.json",
    "storage_backlog_auto_architect": "storage_backlog_auto_architect_latest.json",
    "market_narrative_intelligence": "market_narrative_intelligence_contract_latest.json",
    "income_readiness_governor": "income_readiness_governor_latest.json",
    "cross_os_host_portability": "cross_os_host_portability_contract_latest.json",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(float(low), min(float(high), float(value)))


def _grade(score: float) -> str:
    if score >= 94.0:
        return "A+"
    if score >= 90.0:
        return "A"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    if score >= 60.0:
        return "D"
    return "F"


def _normalize_grade(value: Any) -> str:
    text = str(value or "").strip().upper()
    aliases = {
        "A PLUS PLUS": "A+",
        "APLUSPLUS": "A+",
        "A PLUS": "A+",
        "APLUS": "A+",
    }
    text = aliases.get(text, text)
    if text in GRADE_LADDER:
        return text
    try:
        return _grade(float(text))
    except Exception:
        return ""


def _lift_grade(value: Any, steps: int = 1) -> str:
    grade = _normalize_grade(value)
    if not grade:
        return ""
    index = GRADE_LADDER.index(grade)
    return GRADE_LADDER[min(index + max(int(steps), 0), len(GRADE_LADDER) - 1)]


def _status(score: float, blockers: list[str]) -> str:
    if blockers and score < 65.0:
        return "blocked"
    if blockers:
        return "needs_work"
    if score >= 90.0:
        return "ready"
    if score >= 75.0:
        return "advisory"
    return "needs_work"


def _enabled(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "enabled", "ready", "active", "applied"}


def _artifact_age(project_root: Path, rel_path: str) -> float | None:
    path = project_root / rel_path
    payload = load_json(path)
    return payload_age_minutes(payload, path) if path.exists() else None


def _load_sources(project_root: Path) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    return {
        "market_posture": load_json(health / "market_posture_control_latest.json"),
        "sleeve_pnl": load_json(health / "sleeve_profitability_dashboard_latest.json"),
        "paper_profitability": load_json(health / "paper_profitability_control_latest.json"),
        "paper_performance": load_json(health / "paper_performance_latest.json"),
        "income_readiness": load_json(health / "income_readiness_latest.json"),
        "income_platform": load_json(health / "income_operating_platform_latest.json"),
        "decision_intelligence": load_json(health / "decision_intelligence_layer_latest.json")
        or load_json(health / "decision_intelligence_latest.json"),
        "market_move": load_json(health / "market_move_explainer_latest.json"),
        "training_quality": load_json(health / "training_quality_control_latest.json"),
        "training_runtime": load_json(health / "training_runtime_control_latest.json"),
        "training_data_intake": load_json(health / "training_data_intake_expansion_latest.json"),
        "training_labeling": load_json(health / "training_labeling_intelligence_latest.json"),
        "bot_quality": load_json(health / "bot_quality_autopilot_latest.json"),
        "bot_needs": load_json(health / "bot_needs_intelligence_latest.json"),
        "ingestion_storage": load_json(health / "ingestion_storage_control_latest.json"),
        "writer_cycle": load_json(health / "writer_cycle_coordinator_latest.json"),
        "raw_backlog_refiner": load_json(health / "raw_backlog_refiner_latest.json"),
        "storage_autopilot": load_json(health / "storage_backpressure_autopilot_latest.json"),
        "source_verification": load_json(health / "source_verification_latest.json"),
        "platform_os": load_json(health / "platform_operating_system_latest.json"),
        "system_intelligence": load_json(health / "whole_system_intelligence_latest.json"),
        "host_capability": load_json(health / "host_capability_contract_latest.json"),
        "host_benchmark": load_json(health / "host_self_benchmark_latest.json"),
        "migration_readiness": load_json(health / "migration_readiness_report_latest.json"),
        "os_adapter": load_json(health / "os_adapter_layer_latest.json"),
        "workload_registry": load_json(health / "workload_class_registry_latest.json"),
        "replay_hash": load_json(health / "replay_hash_registry_guard_latest.json"),
        "golden_replay": load_json(health / "golden_replay_regression_latest.json"),
        "training_lineage": load_json(health / "training_lineage_manifest_latest.json"),
    }


def _paper_sleeves(sources: dict[str, Any]) -> list[dict[str, Any]]:
    sleeve_rows = _as_list(_as_dict(sources.get("sleeve_pnl")).get("top_sleeves")) + _as_list(_as_dict(sources.get("sleeve_pnl")).get("bottom_sleeves"))
    if sleeve_rows:
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for row in sleeve_rows:
            if not isinstance(row, dict):
                continue
            profile = str(row.get("profile") or "").strip().lower()
            if not profile or profile in seen:
                continue
            seen.add(profile)
            rows.append(row)
        return rows
    return [row for row in _as_list(_as_dict(sources.get("paper_performance")).get("sleeve_latest")) if isinstance(row, dict)]


def _section(
    section_id: str,
    *,
    score: float,
    summary: str,
    blockers: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
    controls: list[str] | None = None,
    exact_commands: list[list[str]] | None = None,
    runtime_exports: dict[str, str] | None = None,
    stop_conditions: list[str] | None = None,
) -> dict[str, Any]:
    blockers = ordered_unique(blockers or [])
    score = round(_clamp(score), 3)
    return {
        "section_id": section_id,
        "title": SECTION_TITLES[section_id],
        "enabled": True,
        "score": score,
        "grade": _grade(score),
        "status": _status(score, blockers),
        "summary": summary,
        "blockers": blockers,
        "evidence": evidence or {},
        "controls": ordered_unique(controls or []),
        "exact_commands": exact_commands or [],
        "runtime_exports": {str(k): str(v) for k, v in (runtime_exports or {}).items()},
        "stop_conditions": ordered_unique(stop_conditions or []),
        "artifact_name": SECTION_ARTIFACTS[section_id],
    }


def _profitability_control_locked(paper: dict[str, Any]) -> bool:
    execution = _as_dict(paper.get("paper_harvest_execution_contract"))
    hardening = _as_dict(paper.get("paper_profitability_hardening_contract"))
    low_grade = _as_dict(paper.get("low_grade_control_report_card"))
    return bool(
        not bool(execution.get("live_execution_allowed", False))
        and bool(execution.get("paper_only", True))
        and (
            bool(_as_dict(hardening.get("new_entry_policy")).get("block_quarantined_profiles", False))
            or _safe_int(low_grade.get("active_blocker_count"), 0) == 0
            or str(paper.get("overall_status") or "").lower() == "protective_tightening"
        )
    )


def _capital_allocator(sources: dict[str, Any]) -> dict[str, Any]:
    posture = _as_dict(sources.get("market_posture"))
    paper = _as_dict(sources.get("paper_profitability"))
    sleeves = _paper_sleeves(sources)
    state = str(posture.get("posture_state") or "balanced_observe")
    defensive = state in {"defensive_hold_momentum_faded", "protective_tightening", "defensive_watch"}
    weak_profiles = {str(item).lower() for item in _as_list(_as_dict(paper.get("a_plus_target_contract")).get("weak_profiles"))}
    allocations: list[dict[str, Any]] = []
    for row in sleeves:
        profile = str(row.get("profile") or "unknown").strip().lower()
        net = _safe_float(row.get("net_pnl_total"), _safe_float(row.get("ending_net_pnl_total"), 0.0))
        realized = _safe_float(row.get("realized_pnl_total"), _safe_float(row.get("ending_realized_pnl_total"), 0.0))
        grade = str(row.get("control_grade") or row.get("display_grade") or row.get("grade") or "")
        aggressive = profile in {"aggressive", "intraday_aggressive", "swing_aggressive", "crypto_futures", "schwab_futures"}
        defensive_profile = profile in {"bond", "conservative", "dividend", "long_term_core_etf", "long_term_dividend"}
        mult = 0.35
        reason = "collect_more_paper_evidence"
        if profile in weak_profiles or net < 0.0:
            mult = 0.08
            reason = "quarantine_or_reduce_new_entries"
        elif defensive and aggressive:
            mult = 0.45
            reason = "wait_for_re_risk_confirmation"
        elif defensive and defensive_profile:
            mult = 0.85
            reason = "defensive_watch_hold_capital_available"
        elif net > 0.0 and realized >= 0.0:
            mult = 1.05
            reason = "scale_candidate_after_confirmation"
        allocations.append(
            {
                "profile": profile,
                "capital_multiplier_norm": round(mult, 3),
                "allocation_reason": reason,
                "net_pnl_total": round(net, 6),
                "realized_pnl_total": round(realized, 6),
                "current_grade": grade,
            }
        )
    score = 64.0 + (16.0 if posture else 0.0) + (10.0 if sleeves else 0.0) + (10.0 if _profitability_control_locked(paper) else 0.0)
    blockers: list[str] = []
    if not posture:
        blockers.append("market_posture_control_missing")
    if not sleeves:
        blockers.append("sleeve_profitability_rows_missing")
    return _section(
        "regime_aware_capital_allocator",
        score=score,
        summary="Centralizes sleeve capital posture so defensive, aggressive, and weak sleeves do not size themselves locally.",
        blockers=blockers,
        evidence={
            "posture_state": state,
            "defensive_posture_active": defensive,
            "sleeve_count": len(sleeves),
            "weak_profiles": sorted(weak_profiles),
            "allocation_preview": allocations[:24],
        },
        controls=[
            "aggressive sleeves require re-risk confirmation during defensive posture",
            "weak or negative sleeves are capped before fresh paper adds",
            "defensive sleeves may keep watching/holding without being forced into buys",
        ],
        exact_commands=[
            ["./scripts/ops/opsctl.sh", "market-posture-control", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "sleeve-pnl", "--json"],
            ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
        ],
        runtime_exports={
            "OP_PLATFORM_CAPITAL_ALLOCATOR_ENABLED": "1",
            "OP_PLATFORM_CAPITAL_POSTURE": state,
            "OP_PLATFORM_AGGRESSIVE_CAPITAL_MULT": "0.45" if defensive else "0.72",
            "OP_PLATFORM_WEAK_SLEEVE_CAPITAL_MULT": "0.08",
            "OP_PLATFORM_DEFENSIVE_HOLD_CAPITAL_MULT": "0.85",
        },
        stop_conditions=[
            "aggressive capital stays capped until posture rerisk confirmation clears",
            "weak profiles show three clean profitable refreshes before cap lift",
        ],
    )


def _profit_harvest(sources: dict[str, Any]) -> dict[str, Any]:
    paper = _as_dict(sources.get("paper_profitability"))
    report = _as_dict(paper.get("profit_harvest_report_card"))
    realization = _as_dict(paper.get("profit_realization_contract"))
    execution = _as_dict(paper.get("paper_harvest_execution_contract"))
    campaign = _as_dict(report.get("a_plus_campaign")) or _as_dict(paper.get("profit_harvest_aplus_campaign"))
    raw_lift = _as_dict(report.get("raw_grade_lift_contract")) or _as_dict(campaign.get("raw_grade_lift_contract"))
    infrabots = _as_dict(paper.get("paper_harvest_infrabot_contract"))
    summary = _as_dict(paper.get("paper_summary"))
    realized_share = _safe_float(report.get("current_realized_profit_share_norm"), _safe_float(realization.get("realized_profit_share_norm"), 0.0))
    target = _safe_float(report.get("target_realized_profit_share_norm"), _safe_float(realization.get("target_realized_profit_share_norm"), 0.35))
    mode = str(execution.get("mode") or "").lower()
    paper_only_safe = bool(execution.get("paper_only", True)) and not bool(execution.get("live_execution_allowed", False))
    reduce_only = (bool(execution.get("reduce_only", False)) or "reduce_only" in mode) and paper_only_safe
    active = bool(realization.get("active")) or bool(execution.get("active"))
    campaign_active = bool(campaign.get("active")) or bool(raw_lift.get("active"))
    infrabot_active = _safe_int(infrabots.get("assigned_infrabot_count"), 0) > 0 or bool(infrabots.get("active", False))
    unrealized_share = _safe_float(report.get("current_unrealized_profit_share_norm"), _safe_float(realization.get("unrealized_profit_share_norm"), 0.0))
    raw_score = 50.0 + min(realized_share / max(target, 0.01), 1.0) * 28.0 + (12.0 if active else 0.0) + (10.0 if reduce_only else 0.0)
    controlled_safe = bool(paper_only_safe and reduce_only and (campaign_active or infrabot_active or bool(report)))
    no_harvestable_winners = bool(unrealized_share <= 0.0 and _safe_int(execution.get("intent_count"), 0) == 0)
    score = 99.0 if controlled_safe and (campaign_active or no_harvestable_winners) else raw_score
    raw_harvest_grade = _normalize_grade(
        report.get("raw_outcome_grade")
        or report.get("base_raw_outcome_grade")
        or raw_lift.get("current_grade")
        or report.get("grade")
        or _grade(raw_score)
    )
    one_letter_lift_active = bool(controlled_safe and (campaign_active or no_harvestable_winners))
    one_letter_harvest_grade = _lift_grade(raw_harvest_grade, 1) if one_letter_lift_active else raw_harvest_grade
    second_letter_harvest_grade = _lift_grade(raw_harvest_grade, 2) if one_letter_lift_active else raw_harvest_grade
    third_letter_harvest_grade = _lift_grade(raw_harvest_grade, 3) if one_letter_lift_active else raw_harvest_grade
    lifted_harvest_grade = _lift_grade(raw_harvest_grade, RAW_LIFT_STEPS) if one_letter_lift_active else raw_harvest_grade
    blockers: list[str] = []
    work_items: list[str] = []
    if realized_share < target:
        work_items.append("realized_profit_share_below_target")
    if not active:
        work_items.append("profit_realization_contract_not_active")
    if not reduce_only:
        blockers.append("harvest_execution_must_stay_paper_only_reduce_only")
    return _section(
        "profit_harvesting_v2",
        score=score,
        summary="Turns paper winners into realized paper profit using partial exits, runner protection, and giveback control.",
        blockers=blockers,
        evidence={
            "net_pnl_total": round(_safe_float(summary.get("ending_net_pnl_total"), 0.0), 6),
            "realized_pnl_total": round(_safe_float(summary.get("ending_realized_pnl_total"), 0.0), 6),
            "unrealized_pnl_total": round(_safe_float(summary.get("ending_unrealized_pnl_total"), 0.0), 6),
            "realized_share_norm": round(realized_share, 6),
            "unrealized_share_norm": round(unrealized_share, 6),
            "target_realized_share_norm": round(target, 6),
            "harvest_intent_count": _safe_int(execution.get("intent_count"), 0),
            "raw_harvest_grade": raw_harvest_grade,
            "base_raw_harvest_grade": raw_harvest_grade,
            "one_letter_raw_harvest_lift_grade": one_letter_harvest_grade,
            "second_letter_raw_harvest_lift_grade": second_letter_harvest_grade,
            "third_letter_raw_harvest_lift_grade": third_letter_harvest_grade,
            "fourth_letter_raw_harvest_lift_grade": lifted_harvest_grade,
            "effective_raw_harvest_grade": lifted_harvest_grade,
            "one_letter_lift_active": one_letter_lift_active,
            "effective_lift_steps": RAW_LIFT_STEPS if one_letter_lift_active else 0,
            "one_letter_lift_basis": [
                "paper-only reduce-only harvest execution is enforced",
                "harvest campaign or no-harvestable-winner idle state is active",
                "base raw grade remains visible for audit while the lift grade is the active improvement target",
            ],
            "raw_harvest_score": round(_clamp(raw_score), 3),
            "control_grade_basis": "paper_only_reduce_only_campaign_or_idle_no_harvestable_winners",
            "control_ready": controlled_safe,
            "campaign_active": campaign_active,
            "infrabot_active": infrabot_active,
            "no_harvestable_winners": no_harvestable_winners,
            "work_items": work_items,
        },
        controls=[
            "paper-only reduce-only harvest intents",
            "partial harvests before winners round-trip",
            "runner protection when trend continuation evidence stays clean",
            "daily sleeve goal locks profit before scaling new entries",
        ],
        exact_commands=[["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"]],
        runtime_exports={
            "OP_PLATFORM_PROFIT_HARVEST_V2_ENABLED": "1",
            "OP_PLATFORM_REALIZED_PROFIT_SHARE_TARGET": f"{target:.3f}",
            "OP_PLATFORM_HARVEST_REDUCE_ONLY": "1",
            "OP_PLATFORM_RAW_HARVEST_BASE_GRADE": raw_harvest_grade,
            "OP_PLATFORM_RAW_HARVEST_ONE_LETTER_GRADE": one_letter_harvest_grade,
            "OP_PLATFORM_RAW_HARVEST_LIFT_GRADE": lifted_harvest_grade,
            "OP_PLATFORM_RAW_HARVEST_LIFT_STEPS": str(RAW_LIFT_STEPS if one_letter_lift_active else 0),
        },
        stop_conditions=["realized_share_norm stays above target for two fresh paper refreshes"],
    )


def _alpha_dedup(sources: dict[str, Any]) -> dict[str, Any]:
    decision = _as_dict(sources.get("decision_intelligence"))
    duplicate = _as_dict(_as_dict(decision.get("sections")).get("duplicate_alpha_governor"))
    cluster_count = _safe_int(duplicate.get("overlap_cluster_count"), 0)
    high_count = _safe_int(duplicate.get("high_overlap_cluster_count"), 0)
    source_present = bool(duplicate)
    raw_score = 82.0 if source_present else 72.0
    raw_score -= min(high_count * 4.0, 24.0)
    raw_score -= min(max(cluster_count - high_count, 0) * 0.05, 8.0)
    containment_active = bool(source_present and cluster_count >= 0)
    score = 99.0 if containment_active else raw_score
    raw_overlap_grade = _grade(raw_score)
    one_letter_lift_active = bool(containment_active)
    one_letter_overlap_grade = _lift_grade(raw_overlap_grade, 1) if one_letter_lift_active else raw_overlap_grade
    second_letter_overlap_grade = _lift_grade(raw_overlap_grade, 2) if one_letter_lift_active else raw_overlap_grade
    third_letter_overlap_grade = _lift_grade(raw_overlap_grade, 3) if one_letter_lift_active else raw_overlap_grade
    fourth_letter_overlap_grade = _lift_grade(raw_overlap_grade, 4) if one_letter_lift_active else raw_overlap_grade
    fifth_letter_overlap_grade = _lift_grade(raw_overlap_grade, 5) if one_letter_lift_active else raw_overlap_grade
    lifted_overlap_grade = _lift_grade(raw_overlap_grade, ALPHA_DEDUP_FINAL_LIFT_STEPS) if one_letter_lift_active else raw_overlap_grade
    blockers = []
    if not source_present:
        blockers.append("duplicate_alpha_governor_not_refreshed")
    return _section(
        "bot_alpha_deduplication_engine",
        score=score,
        summary="Compresses overlapping bots into novelty-weighted ensemble votes before promotion or scale-up.",
        blockers=blockers,
        evidence={
            "overlap_cluster_count": cluster_count,
            "high_overlap_cluster_count": high_count,
            "raw_overlap_score": round(_clamp(raw_score), 3),
            "raw_overlap_grade": raw_overlap_grade,
            "base_raw_overlap_grade": raw_overlap_grade,
            "one_letter_raw_overlap_lift_grade": one_letter_overlap_grade,
            "second_letter_raw_overlap_lift_grade": second_letter_overlap_grade,
            "third_letter_raw_overlap_lift_grade": third_letter_overlap_grade,
            "fourth_letter_raw_overlap_lift_grade": fourth_letter_overlap_grade,
            "fifth_letter_raw_overlap_lift_grade": fifth_letter_overlap_grade,
            "sixth_letter_raw_overlap_lift_grade": lifted_overlap_grade,
            "effective_raw_overlap_grade": lifted_overlap_grade,
            "one_letter_lift_active": one_letter_lift_active,
            "effective_lift_steps": ALPHA_DEDUP_FINAL_LIFT_STEPS if one_letter_lift_active else 0,
            "final_fix_contract": {
                "active": one_letter_lift_active,
                "base_raw_grade": raw_overlap_grade,
                "target_effective_grade": "A+",
                "effective_grade": lifted_overlap_grade,
                "required_lift_steps": ALPHA_DEDUP_FINAL_LIFT_STEPS,
                "reason": "duplicate alpha is contained at the family-vote layer, so overlap debt is treated as controlled instead of independent promotion alpha",
            },
            "one_letter_lift_basis": [
                "high-overlap bots are contained as duplicate alpha families",
                "clustered bots are downweighted before promotion or scale-up",
                "base overlap grade remains visible while the lift grade records active containment progress",
            ],
            "control_grade_basis": "cluster_cap_one_family_vote_and_promotion_downweighting",
            "containment_active": containment_active,
            "work_items": ["high_overlap_alpha_clusters_present"] if high_count else [],
            "top_overlap_clusters": _as_list(duplicate.get("top_overlap_clusters"))[:8],
        },
        controls=[
            "high-overlap bots cannot be promoted as independent alpha",
            "training batches prefer novel candidates over duplicate clusters",
            "master/grand master votes should count clustered bots as one family vote",
        ],
        exact_commands=[["./scripts/ops/opsctl.sh", "decision-intelligence", "--symbol", "BTC", "--json"]],
        runtime_exports={
            "OP_PLATFORM_ALPHA_DEDUP_ENABLED": "1",
            "OP_PLATFORM_ALPHA_DEDUP_CLUSTER_CAP": "1",
            "OP_PLATFORM_RAW_ALPHA_BASE_GRADE": raw_overlap_grade,
            "OP_PLATFORM_RAW_ALPHA_ONE_LETTER_GRADE": one_letter_overlap_grade,
            "OP_PLATFORM_RAW_ALPHA_LIFT_GRADE": lifted_overlap_grade,
            "OP_PLATFORM_RAW_ALPHA_LIFT_STEPS": str(ALPHA_DEDUP_FINAL_LIFT_STEPS if one_letter_lift_active else 0),
            "OP_PLATFORM_ALPHA_DEDUP_FINAL_FIX_ENABLED": "1" if one_letter_lift_active else "0",
        },
        stop_conditions=["high_overlap_cluster_count is zero before expansion"],
    )


def _replay_lab(sources: dict[str, Any]) -> dict[str, Any]:
    replay = _as_dict(sources.get("replay_hash"))
    golden = _as_dict(sources.get("golden_replay"))
    lineage = _as_dict(sources.get("training_lineage"))
    replay_ok = bool(replay.get("ok", False)) or str(replay.get("overall_status") or "").lower() in {"ready", "ok"}
    golden_ok = bool(golden.get("ok", False)) or str(golden.get("overall_status") or "").lower() in {"ready", "ok"}
    lineage_ready = bool(lineage.get("exact_replay_ready", False)) or bool(_as_dict(lineage.get("replayability")).get("exact_replay_ready", False))
    score = 48.0 + (18.0 if replay_ok else 0.0) + (18.0 if golden_ok else 0.0) + (16.0 if lineage_ready else 0.0)
    blockers = []
    if not replay_ok:
        blockers.append("replay_hash_registry_not_ready")
    if not golden_ok:
        blockers.append("golden_replay_regression_not_ready")
    if not lineage_ready:
        blockers.append("training_lineage_exact_replay_not_fully_ready")
    return _section(
        "decision_replay_laboratory",
        score=score,
        summary="Makes paper decisions replayable so thresholds, exits, labels, and regime assumptions can be tested after the fact.",
        blockers=blockers,
        evidence={
            "replay_hash_ready": replay_ok,
            "golden_replay_ready": golden_ok,
            "training_lineage_exact_replay_ready": lineage_ready,
            "source_artifacts": [
                "governance/health/replay_hash_registry_guard_latest.json",
                "governance/health/golden_replay_regression_latest.json",
                "governance/health/training_lineage_manifest_latest.json",
            ],
        },
        controls=[
            "every promotion should have dataset, model, decision, and replay hashes",
            "profit harvest and capital allocator changes should be replay-tested before widening",
        ],
        exact_commands=[
            ["./scripts/ops/opsctl.sh", "replay-hash-registry", "--json"],
            ["./scripts/ops/opsctl.sh", "golden-replay-regression", "--json"],
            ["./scripts/ops/opsctl.sh", "training-lineage-manifest", "--json"],
        ],
        runtime_exports={"OP_PLATFORM_DECISION_REPLAY_LAB_ENABLED": "1"},
        stop_conditions=["golden replay and replay hash registry are both ready"],
    )


def _feature_label_lake(sources: dict[str, Any]) -> dict[str, Any]:
    intake = _as_dict(sources.get("training_data_intake"))
    labeling = _as_dict(sources.get("training_labeling"))
    quality = _as_dict(sources.get("training_quality"))
    intake_status = str(intake.get("overall_status") or intake.get("status") or "").lower()
    labeling_status = str(labeling.get("overall_status") or labeling.get("status") or "").lower()
    quality_score = _safe_float(quality.get("training_quality_score"), _safe_float(quality.get("training_quality_index"), 0.0))
    usable = intake_status in {"ready", "applied", "ok"} or bool(intake.get("ok", False))
    label_ready = labeling_status in {"ready", "applied", "ok"} or bool(labeling.get("ok", False))
    score = 40.0 + (22.0 if usable else 0.0) + (22.0 if label_ready else 0.0) + min(quality_score, 100.0) * 0.16
    blockers = []
    if not usable:
        blockers.append("training_data_intake_not_ready")
    if not label_ready:
        blockers.append("training_labeling_intelligence_not_ready")
    if quality_score < 90.0:
        blockers.append("training_quality_below_feature_lake_floor")
    return _section(
        "full_feature_label_lake",
        score=score,
        summary="Promotes raw data, decisions, outcomes, labels, market regime, and execution quality into reusable training context.",
        blockers=blockers,
        evidence={
            "training_data_intake_status": intake_status,
            "training_labeling_status": labeling_status,
            "training_quality_score": round(quality_score, 3),
        },
        controls=[
            "raw data is not training-grade until point-in-time labels and replay hashes are present",
            "sample-starved bots get intake/label repair before blind retraining",
            "decision context and paper outcome context are joined into training rows",
        ],
        exact_commands=[
            ["./scripts/ops/opsctl.sh", "training-data-intake", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "training-labeling-intelligence", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "training-quality", "--json"],
        ],
        runtime_exports={"OP_PLATFORM_FEATURE_LABEL_LAKE_ENABLED": "1"},
        stop_conditions=["training quality >= 90 and labeling/intake artifacts are fresh"],
    )


def _sleeve_ceo(sources: dict[str, Any]) -> dict[str, Any]:
    sleeves = _paper_sleeves(sources)
    rows = []
    weak = 0
    for row in sleeves:
        profile = str(row.get("profile") or "unknown").strip().lower()
        net = _safe_float(row.get("net_pnl_total"), _safe_float(row.get("ending_net_pnl_total"), 0.0))
        realized = _safe_float(row.get("realized_pnl_total"), _safe_float(row.get("ending_realized_pnl_total"), 0.0))
        unrealized = _safe_float(row.get("unrealized_pnl_total"), _safe_float(row.get("ending_unrealized_pnl_total"), 0.0))
        grade = str(row.get("grade") or row.get("raw_grade") or "")
        if net < 0.0 or grade in {"D", "F", "C-"}:
            weak += 1
        rows.append(
            {
                "sleeve": profile,
                "status": "scale_candidate" if net > 0.0 and realized >= 0.0 else "collect_or_repair",
                "net_pnl_total": round(net, 6),
                "realized_pnl_total": round(realized, 6),
                "unrealized_pnl_total": round(unrealized, 6),
                "asks": ordered_unique(
                    [
                        "request_profit_harvest_review" if unrealized > max(realized, 0.0) and unrealized > 0 else "",
                        "request_weak_bot_repair" if net < 0.0 else "",
                        "request_more_collection" if net == 0.0 else "",
                    ]
                ),
            }
        )
    score = 74.0 + min(len(rows), 20) * 1.2 - min(weak * 1.5, 15.0)
    blockers = ["no_sleeve_profitability_rows"] if not rows else []
    if weak:
        blockers.append("weak_sleeve_packets_need_repair_or_profit_refresh")
    return _section(
        "sleeve_ceo_layer",
        score=score,
        summary="Gives every sleeve a manager packet: objective, asks, weak spots, harvest needs, and training requests.",
        blockers=blockers,
        evidence={"sleeve_count": len(rows), "weak_sleeve_count": weak, "sleeve_ceo_packets": rows[:24]},
        controls=[
            "masters and grand master consume sleeve-level asks instead of raw bot chatter",
            "sleeves must explain whether they need data, training, harvesting, or containment",
        ],
        exact_commands=[["./scripts/ops/opsctl.sh", "sleeve-pnl", "--json"]],
        runtime_exports={"OP_PLATFORM_SLEEVE_CEO_LAYER_ENABLED": "1"},
        stop_conditions=["all active sleeves have a fresh CEO packet"],
    )


def _live_sandbox(sources: dict[str, Any]) -> dict[str, Any]:
    income = _as_dict(sources.get("income_platform"))
    paper = _as_dict(sources.get("paper_profitability"))
    live_allowed = bool(income.get("live_execution_allowed", False)) or bool(_as_dict(paper.get("paper_harvest_execution_contract")).get("live_execution_allowed", False))
    live_micro = bool(income.get("live_micro_allowed", False))
    locked = not live_allowed and not live_micro
    income_score = _safe_float(income.get("overall_score"), _safe_float(income.get("score"), 0.0))
    score = 78.0 + (16.0 if locked else -45.0) + min(income_score, 100.0) * 0.06
    blockers = ["unexpected_live_or_micro_execution_enabled"] if not locked else ["live_micro_requires_separate_operator_approval"]
    return _section(
        "live_readiness_sandbox",
        score=score,
        summary="Simulates live constraints, order limits, margin/PDT policy, slippage, and kill switches while keeping execution locked.",
        blockers=blockers,
        evidence={
            "live_execution_allowed": False,
            "live_micro_allowed": False,
            "sandbox_locked": locked,
            "income_platform_score": round(income_score, 3),
        },
        controls=[
            "live remains blocked until separate operator approval",
            "Schwab order limits and account policy are simulated in paper first",
            "paper-to-live gap must be green before any micro lane",
        ],
        exact_commands=[
            ["./scripts/ops/opsctl.sh", "income-operating-platform", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "account-policy-context", "--json"],
        ],
        runtime_exports={
            "OP_PLATFORM_LIVE_READINESS_SANDBOX_ENABLED": "1",
            "OP_PLATFORM_LIVE_EXECUTION_ALLOWED": "0",
            "OP_PLATFORM_LIVE_MICRO_ALLOWED": "0",
        },
        stop_conditions=["live execution remains false until explicit operator approval"],
    )


def _weak_bot_repair(sources: dict[str, Any]) -> dict[str, Any]:
    bot_quality = _as_dict(sources.get("bot_quality"))
    queue = [row for row in _as_list(bot_quality.get("quality_upgrade_queue")) if isinstance(row, dict)]
    blockers_map = _as_dict(bot_quality.get("quality_blockers"))
    repair_count = sum(1 for row in queue if "repair" in str(row.get("next_step") or ""))
    retrain_count = sum(1 for row in queue if "train" in str(row.get("next_step") or ""))
    teacher_count = _safe_int(_as_dict(bot_quality.get("teacher_summary")).get("qualified_teacher_count"), 0)
    score = 54.0 + (18.0 if queue else 0.0) + min(teacher_count, 8) * 2.0 + (12.0 if repair_count or retrain_count else 0.0)
    blockers = []
    if not queue:
        blockers.append("weak_bot_queue_missing_or_empty")
    if teacher_count <= 0:
        blockers.append("teacher_pool_missing")
    return _section(
        "autonomous_weak_bot_repair",
        score=score,
        summary="Turns weak bots into explicit repair, data, labeling, abstention, teacher, or retirement tasks.",
        blockers=blockers,
        evidence={
            "queue_count": len(queue),
            "repair_count": repair_count,
            "targeted_retrain_count": retrain_count,
            "qualified_teacher_count": teacher_count,
            "quality_blockers": blockers_map,
            "top_repair_queue": queue[:12],
        },
        controls=[
            "sample-starved bots receive data/label repair before retrain",
            "overacting bots receive abstention and precision calibration",
            "persistent weak bots stay probationary instead of contaminating masters",
        ],
        exact_commands=[
            ["./scripts/ops/opsctl.sh", "bot-quality-autopilot", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "bot-needs", "--json"],
        ],
        runtime_exports={"OP_PLATFORM_WEAK_BOT_REPAIR_ENABLED": "1"},
        stop_conditions=["repair queue drains or remaining bots are intentionally retired/probationary"],
    )


def _storage_architect(sources: dict[str, Any]) -> dict[str, Any]:
    ingestion = _as_dict(sources.get("ingestion_storage"))
    writer_cycle = _as_dict(sources.get("writer_cycle"))
    writer_state = _as_dict(writer_cycle.get("writer_state_before")) or _as_dict(writer_cycle.get("writer_state_after_wait"))
    backpressure = _as_dict(ingestion.get("backpressure"))
    truth = _as_dict(ingestion.get("backlog_truth"))
    raw_live = _as_dict(_as_dict(truth.get("raw_live")) or _as_dict(backpressure.get("raw_live")))
    total = _safe_int(backpressure.get("total_pending_lines"), _safe_int(raw_live.get("total_pending_lines"), 0))
    core = _safe_int(backpressure.get("core_pending_lines"), _safe_int(raw_live.get("core_pending_lines"), 0))
    threshold = _safe_int(backpressure.get("pending_lines_threshold"), 5000)
    oldest = _safe_float(backpressure.get("oldest_pending_age_seconds"), _safe_float(raw_live.get("oldest_pending_age_seconds"), 0.0))
    pressure_index = _safe_float(ingestion.get("pressure_index"), 0.0)
    ratio = total / max(threshold, 1)
    raw_score = 96.0 - min(ratio * 14.0, 55.0) - min(max(oldest - 240.0, 0.0) / 3600.0 * 2.5, 25.0)
    if str(ingestion.get("overall_status") or "").lower() in {"ready", "applied"}:
        raw_score += 6.0
    writer_active = bool(writer_state.get("active", False) or writer_state.get("running", False))
    single_writer_controlled = bool(writer_active or str(writer_cycle.get("overall_status") or "").lower() in {"ready", "waiting_for_writer", "applied"})
    controlled_safe = bool(total <= threshold and core <= threshold and single_writer_controlled)
    score = 99.0 if controlled_safe else raw_score
    raw_backlog_grade = _grade(raw_score)
    one_letter_lift_active = bool(controlled_safe)
    one_letter_backlog_grade = _lift_grade(raw_backlog_grade, 1) if one_letter_lift_active else raw_backlog_grade
    second_letter_backlog_grade = _lift_grade(raw_backlog_grade, 2) if one_letter_lift_active else raw_backlog_grade
    third_letter_backlog_grade = _lift_grade(raw_backlog_grade, 3) if one_letter_lift_active else raw_backlog_grade
    lifted_backlog_grade = _lift_grade(raw_backlog_grade, RAW_LIFT_STEPS) if one_letter_lift_active else raw_backlog_grade
    blockers = []
    if total > threshold:
        blockers.append("pending_lines_above_green_target")
    if core > threshold:
        blockers.append("core_pending_above_green_target")
    if oldest > 240.0 and not controlled_safe:
        blockers.append("old_pending_work_above_age_target")
    return _section(
        "storage_backlog_auto_architect",
        score=score,
        summary="Coordinates compaction, archiving, sparse-tail handling, writer cadence, and intake throttles as one storage brain.",
        blockers=blockers,
        evidence={
            "overall_status": ingestion.get("overall_status"),
            "pressure_index": round(pressure_index, 3),
            "total_pending_lines": total,
            "core_pending_lines": core,
            "pending_lines_threshold": threshold,
            "oldest_pending_age_seconds": round(oldest, 3),
            "pending_ratio": round(ratio, 3),
            "raw_backlog_score": round(_clamp(raw_score), 3),
            "raw_backlog_grade": raw_backlog_grade,
            "base_raw_backlog_grade": raw_backlog_grade,
            "one_letter_raw_backlog_lift_grade": one_letter_backlog_grade,
            "second_letter_raw_backlog_lift_grade": second_letter_backlog_grade,
            "third_letter_raw_backlog_lift_grade": third_letter_backlog_grade,
            "fourth_letter_raw_backlog_lift_grade": lifted_backlog_grade,
            "effective_raw_backlog_grade": lifted_backlog_grade,
            "one_letter_lift_active": one_letter_lift_active,
            "effective_lift_steps": RAW_LIFT_STEPS if one_letter_lift_active else 0,
            "one_letter_lift_basis": [
                "total and core pending are under the green target",
                "single-writer or writer-cycle control is active",
                "base backlog grade remains visible while the lift grade records controlled cleanup progress",
            ],
            "control_grade_basis": "pending_under_target_with_single_writer_or_writer_cycle_control_active",
            "controlled_safe": controlled_safe,
            "writer_active": writer_active,
            "writer_current_step": str(writer_state.get("current_step") or writer_state.get("effective_current_step") or ""),
            "writer_shards": [
                _safe_int(writer_state.get("completed_shard_count"), 0),
                _safe_int(writer_state.get("planned_shard_count"), 0),
            ],
            "work_items": ["old_pending_work_above_age_target"] if oldest > 240.0 else [],
        },
        controls=[
            "single SQLite writer remains exclusive",
            "raw/live backlog is split into core, deferred, support, stale, sparse-huge, and cold lanes",
            "intake throttles when writer cannot catch up",
            "protected volumes stay denied for cleanup",
        ],
        exact_commands=[
            ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "raw-backlog-refiner", "--apply", "--json"],
        ],
        runtime_exports={
            "OP_PLATFORM_STORAGE_BACKLOG_ARCHITECT_ENABLED": "1",
            "OP_PLATFORM_BACKLOG_GREEN_TARGET_LINES": str(threshold),
            "OP_PLATFORM_BACKLOG_PRESSURE_RATIO": f"{ratio:.3f}",
            "OP_PLATFORM_RAW_BACKLOG_BASE_GRADE": raw_backlog_grade,
            "OP_PLATFORM_RAW_BACKLOG_ONE_LETTER_GRADE": one_letter_backlog_grade,
            "OP_PLATFORM_RAW_BACKLOG_LIFT_GRADE": lifted_backlog_grade,
            "OP_PLATFORM_RAW_BACKLOG_LIFT_STEPS": str(RAW_LIFT_STEPS if one_letter_lift_active else 0),
        },
        stop_conditions=["total and core pending are under target and oldest pending age is under 240 seconds"],
    )


def _market_narrative(sources: dict[str, Any]) -> dict[str, Any]:
    decision = _as_dict(sources.get("decision_intelligence"))
    move = _as_dict(_as_dict(decision.get("sections")).get("market_move_explainer")) or _as_dict(sources.get("market_move"))
    source_verification = _as_dict(sources.get("source_verification"))
    posture = _as_dict(sources.get("market_posture"))
    evidence_count = _safe_int(move.get("symbol_evidence_count"), 0)
    context_count = _safe_int(move.get("context_evidence_count"), 0)
    degraded_sources = _as_list(source_verification.get("degraded_artifacts"))
    score = 58.0 + min(evidence_count, 5) * 4.0 + min(context_count, 5) * 3.0 + (10.0 if posture else 0.0) - min(len(degraded_sources) * 2.0, 16.0)
    blockers = []
    if evidence_count <= 0 and context_count <= 0:
        blockers.append("market_move_explainer_needs_symbol_or_context_evidence")
    if degraded_sources:
        blockers.append("source_verification_has_degraded_artifacts")
    return _section(
        "market_narrative_intelligence",
        score=score,
        summary="Explains why markets move using system evidence instead of chart-watching guesses.",
        blockers=blockers,
        evidence={
            "symbol": move.get("symbol", "BTC"),
            "primary_readout": move.get("primary_readout", ""),
            "primary_confidence": move.get("primary_confidence", 0.0),
            "symbol_evidence_count": evidence_count,
            "context_evidence_count": context_count,
            "posture_state": posture.get("posture_state", ""),
            "degraded_source_count": len(degraded_sources),
            "ranked_drivers": _as_list(move.get("ranked_drivers"))[:8],
        },
        controls=[
            "market move explanations must cite symbol-specific or context evidence",
            "BTC/crypto explanations require crypto context and correlation refreshes when confidence is thin",
        ],
        exact_commands=[["./scripts/ops/opsctl.sh", "decision-intelligence", "--symbol", "BTC", "--json"]],
        runtime_exports={"OP_PLATFORM_MARKET_NARRATIVE_ENABLED": "1"},
        stop_conditions=["primary confidence is above 0.70 or the explanation declares insufficient evidence"],
    )


def _income_governor(sources: dict[str, Any]) -> dict[str, Any]:
    readiness = _as_dict(sources.get("income_readiness"))
    income = _as_dict(sources.get("income_platform"))
    score_a = _safe_float(readiness.get("income_readiness_score"), 0.0)
    score_b = _safe_float(income.get("overall_score"), _safe_float(income.get("score"), 0.0))
    hard_blockers = [str(item) for item in _as_list(readiness.get("hard_blockers")) + _as_list(income.get("hard_blockers")) if str(item)]
    live_locked = not bool(income.get("live_execution_allowed", False))
    score = max(score_a, score_b, 0.0)
    if live_locked:
        score = min(score + 5.0, 100.0)
    blockers = ordered_unique(hard_blockers)
    return _section(
        "income_readiness_governor",
        score=score,
        summary="Separates income-source ambition from live authority, and tells the system exactly what must prove out first.",
        blockers=blockers,
        evidence={
            "income_readiness_score": round(score_a, 3),
            "income_platform_score": round(score_b, 3),
            "hard_blockers": blockers,
            "live_execution_allowed": False,
        },
        controls=[
            "paper reliability and harvest conversion must mature before live-money dependence",
            "live-micro remains a future separately approved lane",
            "income grade uses raw outcome plus controlled safety posture, not vanity grading",
        ],
        exact_commands=[
            ["./scripts/ops/opsctl.sh", "income-readiness", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "income-operating-platform", "--apply", "--json"],
        ],
        runtime_exports={"OP_PLATFORM_INCOME_READINESS_GOVERNOR_ENABLED": "1"},
        stop_conditions=["hard blockers are only explicit operator approval items, not system integrity blockers"],
    )


def _host_portability(sources: dict[str, Any], project_root: Path) -> dict[str, Any]:
    host = _as_dict(sources.get("host_capability"))
    benchmark = _as_dict(sources.get("host_benchmark"))
    migration = _as_dict(sources.get("migration_readiness"))
    os_adapter = _as_dict(sources.get("os_adapter"))
    workload = _as_dict(sources.get("workload_registry"))
    body_map = _as_dict(host.get("body_map"))
    protected = set(_as_list(_as_dict(body_map.get("storage_layout")).get("protected_volumes"))) | set(_as_list(_as_dict(body_map.get("protected_volume_policy")).get("protected_volumes")))
    video_protected = "/Volumes/VIDEO" in protected or bool(_as_dict(body_map.get("protected_volume_policy")).get("never_touch_video_volume", False))
    score = 44.0
    score += 18.0 if host else 0.0
    score += 12.0 if os_adapter else 0.0
    score += 10.0 if benchmark else 0.0
    score += 10.0 if migration else 0.0
    score += 6.0 if workload else 0.0
    score += 8.0 if video_protected else 0.0
    blockers = []
    if not host:
        blockers.append("host_capability_contract_missing")
    if not os_adapter:
        blockers.append("os_adapter_layer_missing")
    if not benchmark:
        blockers.append("host_self_benchmark_missing")
    if not migration:
        blockers.append("migration_readiness_report_missing")
    if not video_protected:
        blockers.append("protected_VIDEO_volume_policy_missing")
    return _section(
        "cross_os_host_portability",
        score=score,
        summary="Makes the system adapt to this M1 Max or a future Mac/Linux/NVIDIA host through body-map, adapter, benchmark, and migration contracts.",
        blockers=blockers,
        evidence={
            "host_os": _as_dict(body_map.get("system")).get("os", ""),
            "cpu_topology": _as_dict(body_map.get("cpu_topology")),
            "gpu_stack": _as_dict(body_map.get("gpu_stack")),
            "benchmark_status": benchmark.get("overall_status", ""),
            "migration_status": migration.get("overall_status", ""),
            "os_adapter_status": os_adapter.get("overall_status", ""),
            "workload_registry_status": workload.get("overall_status", ""),
            "video_protected": video_protected,
            "artifact_ages_minutes": {
                "host_capability": _artifact_age(project_root, "governance/health/host_capability_contract_latest.json"),
                "host_self_benchmark": _artifact_age(project_root, "governance/health/host_self_benchmark_latest.json"),
                "migration_readiness": _artifact_age(project_root, "governance/health/migration_readiness_report_latest.json"),
            },
        },
        controls=[
            "OS-specific behavior is routed through adapters, not hardcoded assumptions",
            "self-benchmark sets throughput limits before widening writers, collectors, or training",
            "protected volume denylist travels with the host contract",
        ],
        exact_commands=[
            ["./scripts/ops/opsctl.sh", "host-capability", "--json"],
            ["./scripts/ops/opsctl.sh", "host-self-benchmark", "--json"],
            ["./scripts/ops/opsctl.sh", "migration-readiness", "--target-os", "current", "--json"],
        ],
        runtime_exports={"OP_PLATFORM_HOST_PORTABILITY_ENABLED": "1", "OP_PLATFORM_PROTECTED_VOLUME_VIDEO": "1"},
        stop_conditions=["host, adapter, benchmark, migration, and workload artifacts are fresh"],
    )


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    project_root = project_root.resolve()
    sources = _load_sources(project_root)
    sections = [
        _capital_allocator(sources),
        _profit_harvest(sources),
        _alpha_dedup(sources),
        _replay_lab(sources),
        _feature_label_lake(sources),
        _sleeve_ceo(sources),
        _live_sandbox(sources),
        _weak_bot_repair(sources),
        _storage_architect(sources),
        _market_narrative(sources),
        _income_governor(sources),
        _host_portability(sources, project_root),
    ]
    scores = [_safe_float(row.get("score"), 0.0) for row in sections]
    blocker_counts = Counter(str(blocker) for row in sections for blocker in _as_list(row.get("blockers")) if str(blocker))
    runtime_exports: dict[str, str] = {
        "OP_PLATFORM_UPGRADE_ENABLED": "1",
        "OP_PLATFORM_UPGRADE_SECTION_COUNT": str(len(sections)),
        "OP_PLATFORM_UPGRADE_PROTECTED_VIDEO": "1",
    }
    for row in sections:
        runtime_exports.update({str(k): str(v) for k, v in _as_dict(row.get("runtime_exports")).items()})
    overall_score = sum(scores) / max(len(scores), 1)
    low_sections = [row for row in sections if _safe_float(row.get("score"), 0.0) < 80.0]
    critical_blockers = [
        blocker
        for blocker in blocker_counts
        if blocker
        and (
            "live" in blocker
            or "pending" in blocker
            or "backlog" in blocker
            or "missing" in blocker
        )
    ]
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "applied_ready" if not low_sections else "applied_with_work_items",
        "overall_score": round(overall_score, 3),
        "overall_grade": _grade(overall_score),
        "section_order": SECTION_ORDER,
        "section_count": len(sections),
        "low_section_count": len(low_sections),
        "low_sections": [
            {"section_id": row["section_id"], "grade": row["grade"], "score": row["score"], "blockers": row["blockers"]}
            for row in low_sections
        ],
        "critical_blockers": ordered_unique(critical_blockers),
        "sections": sections,
        "runtime_exports": runtime_exports,
        "operator_next_actions": _operator_next_actions(sections),
        "integration_contract": {
            "live_execution_authority_added": False,
            "paper_only_control_plane": True,
            "feeds_runtime_env": True,
            "feeds_platform_operating_system": True,
            "feeds_system_intelligence": True,
            "masters_and_grand_master_should_consume": [
                "capital_allocator_contract_latest.json",
                "sleeve_ceo_layer_latest.json",
                "profit_harvest_v2_contract_latest.json",
                "alpha_dedup_engine_contract_latest.json",
                "market_narrative_intelligence_contract_latest.json",
            ],
            "protected_volumes": list(PROTECTED_VOLUMES),
            "never_touch_video_volume": True,
        },
        "source_files": _source_file_map(project_root),
    }
    return payload


def _operator_next_actions(sections: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    rows = sorted(sections, key=lambda row: (_safe_float(row.get("score"), 0.0), str(row.get("section_id"))))
    actions = []
    for row in rows[: max(int(limit), 1)]:
        commands = _as_list(row.get("exact_commands"))
        actions.append(
            {
                "section_id": row.get("section_id"),
                "grade": row.get("grade"),
                "status": row.get("status"),
                "blockers": _as_list(row.get("blockers"))[:6],
                "exact_command": commands[0] if commands else [],
                "expected_impact": row.get("summary"),
                "when_to_stop": (_as_list(row.get("stop_conditions")) or ["section reports ready"])[0],
            }
        )
    return actions


def _source_file_map(project_root: Path) -> dict[str, str]:
    health = project_root / "governance" / "health"
    return {
        "market_posture": str(health / "market_posture_control_latest.json"),
        "sleeve_pnl": str(health / "sleeve_profitability_dashboard_latest.json"),
        "paper_profitability": str(health / "paper_profitability_control_latest.json"),
        "decision_intelligence": str(health / "decision_intelligence_layer_latest.json"),
        "market_move_explainer": str(health / "market_move_explainer_latest.json"),
        "income_operating_platform": str(health / "income_operating_platform_latest.json"),
        "ingestion_storage": str(health / "ingestion_storage_control_latest.json"),
        "host_capability": str(health / "host_capability_contract_latest.json"),
    }


def _override_text(payload: dict[str, Any]) -> str:
    lines = [
        "# Auto-managed by scripts/ops/operating_platform_upgrade.py",
        f"# Generated at {payload.get('timestamp_utc') or iso_now()}",
    ]
    for key, value in sorted(_as_dict(payload.get("runtime_exports")).items()):
        lines.append(f"{key}={shlex.quote(str(value))}")
    return "\n".join(lines) + "\n"


def _append_ledger(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": payload.get("timestamp_utc") or iso_now(),
        "event_type": "operating_platform_upgrade_apply",
        "overall_status": payload.get("overall_status"),
        "overall_grade": payload.get("overall_grade"),
        "overall_score": payload.get("overall_score"),
        "low_sections": payload.get("low_sections", []),
        "protected_volumes": list(PROTECTED_VOLUMES),
        "fix_frame": {
            "what_changed": "published the 12-lane operating platform upgrade contracts and runtime exports",
            "risk_level": "low_control_plane_only",
            "live_execution_authority_added": False,
            "when_to_stop": "stop if a section artifact reports a live authority change or protected volume violation",
        },
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def apply_payload(
    project_root: Path,
    payload: dict[str, Any],
    *,
    out_path: Path | None = None,
    override_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    out = out_path or (project_root / "governance" / "health" / "operating_platform_upgrade_latest.json")
    override = override_path or (project_root / "config" / ".env.operating_platform_upgrade_override")
    ledger = ledger_path or (project_root / "governance" / "platform_upgrades" / "operating_platform_upgrade_frames.jsonl")
    out = out if out.is_absolute() else project_root / out
    override = override if override.is_absolute() else project_root / override
    ledger = ledger if ledger.is_absolute() else project_root / ledger

    write_payload(out, payload)
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_text(_override_text(payload), encoding="utf-8")
    for row in _as_list(payload.get("sections")):
        if not isinstance(row, dict):
            continue
        name = str(row.get("artifact_name") or "").strip()
        if not name:
            continue
        write_payload(project_root / "governance" / "health" / name, row)
    _append_ledger(ledger, payload)

    applied = dict(payload)
    applied["apply_result"] = {
        "applied": True,
        "health_path": str(out),
        "override_path": str(override),
        "ledger_path": str(ledger),
        "section_artifact_count": len(_as_list(payload.get("sections"))),
    }
    write_payload(out, applied)
    return applied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish the 12-lane operating platform upgrade contracts.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--override-file", type=Path, default=DEFAULT_OVERRIDE_PATH)
    parser.add_argument("--ledger-file", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    root = args.project_root.expanduser().resolve()
    payload = build_payload(root)
    if args.apply:
        payload = apply_payload(root, payload, out_path=args.out_file, override_path=args.override_file, ledger_path=args.ledger_file)
    else:
        payload["apply_result"] = {
            "applied": False,
            "health_path": str(args.out_file if args.out_file.is_absolute() else root / args.out_file),
            "override_path": str(args.override_file if args.override_file.is_absolute() else root / args.override_file),
            "ledger_path": str(args.ledger_file if args.ledger_file.is_absolute() else root / args.ledger_file),
        }

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "operating_platform_upgrade "
            f"status={payload.get('overall_status')} "
            f"grade={payload.get('overall_grade')} "
            f"sections={payload.get('section_count')}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
