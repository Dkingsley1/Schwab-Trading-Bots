#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "income_operating_platform_latest.json"
DEFAULT_CONTROL_PATH = PROJECT_ROOT / "governance" / "health" / "income_operating_platform_runtime_controls_latest.json"
DEFAULT_DASHBOARD_PATH = PROJECT_ROOT / "governance" / "health" / "income_operator_dashboard_latest.json"

SECTION_ORDER = [
    "income_promotion_gate",
    "realized_profit_engine",
    "drawdown_governor",
    "paper_to_live_gap_model",
    "live_micro_lane",
    "withdrawal_simulator",
    "account_rules_layer",
    "sleeve_profitability_ranking",
    "failure_mode_drills",
    "human_income_dashboard",
]

SECTION_WEIGHTS = {
    "income_promotion_gate": 0.13,
    "realized_profit_engine": 0.13,
    "drawdown_governor": 0.13,
    "paper_to_live_gap_model": 0.12,
    "live_micro_lane": 0.10,
    "withdrawal_simulator": 0.09,
    "account_rules_layer": 0.08,
    "sleeve_profitability_ranking": 0.10,
    "failure_mode_drills": 0.07,
    "human_income_dashboard": 0.05,
}

REGULATORY_REFERENCES = [
    {
        "source": "FINRA Rule 4210",
        "url": "https://www.finra.org/rules-guidance/rulebooks/finra-rules/4210",
        "reason": "margin, day-trading buying power, and account equity constraints",
    },
    {
        "source": "FINRA Regulatory Notice 26-10",
        "url": "https://business.cch.com/srd/RegulatoryNotice26-10_FINRAorg042126.pdf",
        "reason": "new intraday margin standards replacing PDT/day-trading margin requirements effective 2026-06-04 with broker phase-in",
    },
    {
        "source": "SEC Release No. 34-105226",
        "url": "https://www.sec.gov/files/rules/sro/finra/2026/34-105226.pdf",
        "reason": "SEC approval order for FINRA Rule 4210 intraday margin amendments",
    },
    {
        "source": "Schwab day-trading rule update",
        "url": "https://www.schwab.com/learn/story/schwab-changes-rules-around-day-trading",
        "reason": "Schwab-specific June 8 implementation and intraday buying-power treatment",
    },
    {
        "source": "FINRA Day Trading",
        "url": "https://www.finra.org/investors/investing/investment-products/stocks/day-trading",
        "reason": "day-trading risk and broker execution awareness",
    },
    {
        "source": "SEC Day Trading: Your Dollars at Risk",
        "url": "https://www.sec.gov/about/reports-publications/investor-publications/day-trading-your-dollars-at-risk",
        "reason": "risk, break-even discipline, and execution-system dependence",
    },
]


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


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _grade(score: float) -> str:
    score = _safe_float(score)
    if score >= 97.0:
        return "A+"
    if score >= 92.0:
        return "A+"
    if score >= 85.0:
        return "A"
    if score >= 75.0:
        return "B"
    if score >= 65.0:
        return "C"
    if score >= 50.0:
        return "D"
    return "F"


def _status(score: float, blockers: list[str]) -> str:
    if blockers and score < 65.0:
        return "blocked"
    if blockers:
        return "degraded"
    if score >= 85.0:
        return "ready"
    if score >= 75.0:
        return "needs_work"
    return "degraded"


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def _section(
    section_id: str,
    *,
    title: str,
    score: float,
    summary: str,
    blockers: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
    controls: list[str] | None = None,
    exact_commands: list[list[str]] | None = None,
    stop_conditions: list[str] | None = None,
) -> dict[str, Any]:
    blockers = ordered_unique(blockers or [])
    return {
        "section_id": section_id,
        "title": title,
        "score": round(_safe_float(score), 3),
        "grade": _grade(score),
        "status": _status(score, blockers),
        "summary": summary,
        "blockers": blockers,
        "evidence": evidence or {},
        "controls": ordered_unique(controls or []),
        "exact_commands": exact_commands or [],
        "stop_conditions": ordered_unique(stop_conditions or []),
    }


def _load_sources(project_root: Path) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    return {
        "income_readiness": load_json(health / "income_readiness_latest.json"),
        "paper_profitability": load_json(health / "paper_profitability_control_latest.json"),
        "paper_performance": load_json(health / "paper_performance_latest.json"),
        "paper_calibration": load_json(health / "paper_execution_calibration_latest.json")
        or load_json(health / "paper_calibration_latest.json")
        or load_json(health / "paper_execution_reconciliation_latest.json"),
        "training_runtime": load_json(health / "training_runtime_control_latest.json"),
        "training_quality": load_json(health / "training_quality_control_latest.json"),
        "promotion_quality": load_json(health / "promotion_quality_gate_latest.json"),
        "account_policy": load_json(health / "account_policy_context_latest.json"),
        "chaos_drills": load_json(health / "chaos_drill_coordinator_latest.json"),
        "process_watchdog": load_json(health / "process_watchdog_latest.json"),
        "runtime_gate": load_json(health / "runtime_gate_dashboard_latest.json"),
        "storage_quota": load_json(health / "storage_quota_guard_latest.json"),
        "backpressure": load_json(health / "ingestion_backpressure_latest.json"),
        "bot_logs_cleanup": load_json(health / "bot_logs_cleanup_intelligence_latest.json"),
    }


def _paper_history(sources: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in _as_list(sources["paper_performance"].get("history_daily_series")) if isinstance(row, dict)]
    rows.sort(key=lambda row: str(row.get("day_utc") or row.get("day") or ""))
    return rows


def _paper_sleeves(sources: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in _as_list(sources["paper_performance"].get("sleeve_latest")) if isinstance(row, dict)]


def _profit_summary(sources: dict[str, Any]) -> dict[str, Any]:
    return _nested(sources["paper_profitability"], "paper_summary", default={}) or {}


def _grade_at_least(grade: Any, target: str) -> bool:
    ranks = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4, "A+": 5, "A++": 5}
    return ranks.get(str(grade or "").strip().upper(), -1) >= ranks.get(target, 999)


def _profitability_control_posture(sources: dict[str, Any]) -> dict[str, Any]:
    profitability = sources["paper_profitability"]
    target = _nested(profitability, "a_plus_target_contract", default={}) or {}
    low_grade = _nested(profitability, "low_grade_control_report_card", default={}) or {}
    containment = _nested(profitability, "raw_operational_containment_filter", default={}) or {}
    hardening = _nested(profitability, "paper_profitability_hardening_contract", default={}) or {}
    execution = _nested(profitability, "paper_harvest_execution_contract", default={}) or {}
    current = _nested(target, "current", default={}) or {}
    current_weak_profiles = _safe_int(current.get("weak_profile_count"), len(_as_list(target.get("weak_profiles"))))
    unprotected_weak = _safe_int(current.get("unprotected_weak_profile_count"), 0)
    unprotected_strategy = _safe_int(current.get("unprotected_strategy_control_count"), 0)
    active_blockers = _safe_int(low_grade.get("active_blocker_count"), 0)
    contained_profiles = _safe_int(containment.get("contained_weak_profile_count"), len(_as_list(containment.get("contained_profiles"))))
    contained_strategy_pairs = _safe_int(
        containment.get("contained_strategy_control_count"),
        len(_as_list(containment.get("contained_strategy_pairs"))),
    )
    weak_contained = unprotected_weak == 0 and unprotected_strategy == 0 and current_weak_profiles >= 0
    active_new_entry_block = bool(_nested(hardening, "new_entry_policy", "block_quarantined_profiles", default=False))
    reduce_only = bool(execution.get("reduce_only")) and bool(execution.get("paper_only", True)) and not bool(execution.get("live_execution_allowed"))
    control_grade = str(low_grade.get("control_posture_grade") or containment.get("contained_grade") or "")
    a_plus_control_ready = bool(low_grade.get("a_plus_control_ready", False)) or _grade_at_least(control_grade, "A+")
    return {
        "active": bool(profitability),
        "overall_status": profitability.get("overall_status"),
        "control_posture_grade": control_grade,
        "a_plus_control_ready": a_plus_control_ready,
        "low_grade_active_blocker_count": active_blockers,
        "weak_profile_count": current_weak_profiles,
        "unprotected_weak_profile_count": unprotected_weak,
        "unprotected_strategy_control_count": unprotected_strategy,
        "contained_weak_profile_count": contained_profiles,
        "contained_strategy_control_count": contained_strategy_pairs,
        "weak_exposure_contained": weak_contained,
        "new_entry_quarantine_active": active_new_entry_block,
        "reduce_only_harvest_active": reduce_only,
        "paper_only": bool(execution.get("paper_only", True)),
        "live_execution_allowed": bool(execution.get("live_execution_allowed")),
    }


def _drawdown_stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    peak: float | None = None
    latest = 0.0
    max_drawdown = 0.0
    worst_day = 0.0
    negative_days = 0
    for row in history:
        value = _safe_float(row.get("ending_net_pnl_total"), 0.0)
        latest = value
        peak = value if peak is None else max(peak, value)
        max_drawdown = min(max_drawdown, value - (peak or value))
        change = _safe_float(row.get("change_vs_previous_day"), 0.0)
        worst_day = min(worst_day, change)
        if change < 0.0:
            negative_days += 1
    return {
        "history_days": len(history),
        "latest_net_pnl_total": round(latest, 6),
        "max_drawdown_total": round(max_drawdown, 6),
        "worst_day_change_total": round(worst_day, 6),
        "negative_day_count": negative_days,
    }


def _section_income_promotion_gate(sources: dict[str, Any]) -> dict[str, Any]:
    readiness = sources["income_readiness"]
    promotion = sources["promotion_quality"]
    training_runtime = sources["training_runtime"]
    training_quality = sources["training_quality"]
    readiness_score = _safe_float(readiness.get("income_readiness_score"), 0.0)
    promotion_ok = bool(promotion.get("ok", False))
    failed_checks = [str(item) for item in _as_list(promotion.get("failed_checks"))]
    details = promotion.get("details") if isinstance(promotion.get("details"), dict) else {}
    runtime_status = str(training_runtime.get("overall_status") or "").lower()
    runtime_ready = runtime_status in {"ready", "ok", "cleared"}
    runtime_prep_ready = bool(_nested(training_runtime, "training_launch_contract", "prep_allowed", default=False))
    runtime_buffer_safe = bool(_nested(training_runtime, "pretraining_drain_buffer", "safe_to_launch_now", default=False))
    quality_score = _safe_float(training_quality.get("training_quality_score"), _safe_float(training_quality.get("training_quality_index"), 0.0))
    partial_promotion_checks = [
        "feature_store_manifest_ready",
        "retrain_schema_compatibility_ok",
        "golden_replay_regression_ok",
        "cohort_drift_baseline_ok",
        "leak_overfit_ok",
        "replay_ok",
        "replay_hash_registry_ok",
        "champion_challenger_probation_ok",
        "reconciliation_slo_ok",
        "snapshot_coverage_ok",
        "data_source_divergence_ok",
        "artifact_freshness_ok",
    ]
    partial_pass_count = sum(1 for key in partial_promotion_checks if bool(details.get(key, False)))
    partial_promotion_score = 8.0 * _clamp(partial_pass_count / max(len(partial_promotion_checks), 1))
    raw_score = 20.0 + min(readiness_score * 0.28, 28.0) + min(quality_score * 0.25, 25.0)
    raw_score += 15.0 if promotion_ok else 0.0
    raw_score += 12.0 if runtime_ready else 10.0 if runtime_prep_ready and runtime_buffer_safe else 6.0 if runtime_prep_ready or runtime_buffer_safe else 0.0
    raw_score += partial_promotion_score if not promotion_ok else 0.0
    controlled_promotion_ready = (
        promotion_ok
        and quality_score >= 100.0
        and readiness_score >= 89.0
        and (runtime_ready or (runtime_prep_ready and runtime_buffer_safe))
        and partial_pass_count >= len(partial_promotion_checks)
    )
    score = 100.0 if controlled_promotion_ready else raw_score
    blockers: list[str] = []
    if readiness_score < 90.0:
        blockers.append("income_readiness_below_money_promotion_floor")
    if not promotion_ok:
        blockers.append("promotion_quality_gate_not_ready")
    if not runtime_ready:
        blockers.append("training_runtime_launch_headroom_not_clear")
    if quality_score < 90.0:
        blockers.append("training_quality_below_money_floor")
    return _section(
        "income_promotion_gate",
        title="Income Promotion Gate",
        score=min(score, 100.0),
        summary="Decides whether any bot, sleeve, or strategy can graduate toward money-grade evidence.",
        blockers=blockers,
        evidence={
            "income_readiness_score": round(readiness_score, 3),
            "income_readiness_grade": readiness.get("income_readiness_grade"),
            "promotion_quality_ok": promotion_ok,
            "promotion_failed_checks": failed_checks[:20],
            "training_runtime_status": runtime_status,
            "training_runtime_prep_ready": runtime_prep_ready,
            "training_runtime_backlog_buffer_safe": runtime_buffer_safe,
            "training_quality_score": round(quality_score, 3),
            "partial_promotion_pass_count": partial_pass_count,
            "partial_promotion_check_count": len(partial_promotion_checks),
            "partial_promotion_score": round(partial_promotion_score, 3),
            "raw_income_promotion_score": round(max(min(raw_score, 100.0), 0.0), 3),
            "controlled_money_promotion_ready": controlled_promotion_ready,
            "controlled_score_basis": "perfect paper-money gate control when promotion quality, training quality, and prep/backlog buffers are fully clean",
        },
        controls=[
            "money eligibility requires readiness, promotion quality, training quality, and runtime readiness",
            "probationary or weak bots remain capped even when paper PnL is positive",
        ],
        exact_commands=[
            ["./scripts/ops/opsctl.sh", "income-readiness", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "promotion-quality-gate", "--json"],
            ["./scripts/ops/opsctl.sh", "training-runtime-control", "--json"],
        ],
        stop_conditions=[
            "promotion_quality_ok is true",
            "training_runtime_status is ready",
            "income_readiness_score >= 90",
        ],
    )


def _section_realized_profit_engine(sources: dict[str, Any]) -> dict[str, Any]:
    profitability = sources["paper_profitability"]
    report = _nested(profitability, "profit_harvest_report_card", default={}) or {}
    realization = _nested(profitability, "profit_realization_contract", default={}) or {}
    execution = _nested(profitability, "paper_harvest_execution_contract", default={}) or {}
    summary = _profit_summary(sources)
    realized_share = _safe_float(report.get("current_realized_profit_share_norm"), _safe_float(realization.get("realized_profit_share_norm"), 0.0))
    unrealized_share = _safe_float(report.get("current_unrealized_profit_share_norm"), _safe_float(realization.get("unrealized_profit_share_norm"), 0.0))
    target_share = _safe_float(report.get("target_realized_profit_share_norm"), _safe_float(realization.get("target_realized_profit_share_norm"), 0.35))
    active = bool(realization.get("active")) or bool(execution.get("active"))
    reduce_only = bool(execution.get("reduce_only"))
    paper_only = bool(execution.get("paper_only", True))
    live_allowed = bool(execution.get("live_execution_allowed"))
    intent_count = _safe_int(execution.get("intent_count"), 0)
    score = 30.0 + 35.0 * _clamp(realized_share / max(target_share, 0.01))
    score += 12.0 if active else 0.0
    score += 10.0 if reduce_only and paper_only and not live_allowed else 0.0
    score += 8.0 if intent_count > 0 else 0.0
    score += 5.0 * _clamp((1.0 - unrealized_share) / 0.35)
    blockers: list[str] = []
    if realized_share < target_share:
        blockers.append("realized_profit_share_below_target")
    if not active:
        blockers.append("paper_profit_harvest_not_active")
    if live_allowed:
        blockers.append("harvest_contract_unexpectedly_allows_live_execution")
    return _section(
        "realized_profit_engine",
        title="Realized Profit Engine",
        score=min(score, 100.0),
        summary="Converts paper winners into realized paper profit without letting winners round-trip.",
        blockers=blockers,
        evidence={
            "net_pnl_total": round(_safe_float(summary.get("ending_net_pnl_total"), 0.0), 6),
            "realized_pnl_total": round(_safe_float(summary.get("ending_realized_pnl_total"), 0.0), 6),
            "unrealized_pnl_total": round(_safe_float(summary.get("ending_unrealized_pnl_total"), 0.0), 6),
            "realized_share_norm": round(realized_share, 6),
            "unrealized_share_norm": round(unrealized_share, 6),
            "target_realized_share_norm": round(target_share, 6),
            "paper_harvest_intent_count": intent_count,
            "harvest_grade": report.get("grade"),
            "raw_harvest_grade": report.get("raw_outcome_grade") or report.get("base_raw_outcome_grade"),
        },
        controls=[
            "profit harvesting remains paper-only and reduce-only",
            "sleeves with daily targets block fresh adds until profit conversion improves",
            "runner protection stays active so harvesting does not crush trend continuation",
        ],
        exact_commands=[["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"]],
        stop_conditions=[
            "realized_share_norm >= target_realized_share_norm",
            "paper_harvest_intent_count is nonzero when unrealized profit is high",
        ],
    )


def _section_drawdown_governor(sources: dict[str, Any]) -> dict[str, Any]:
    history = _paper_history(sources)
    stats = _drawdown_stats(history)
    control = _profitability_control_posture(sources)
    latest = _safe_float(stats.get("latest_net_pnl_total"), 0.0)
    max_drawdown = abs(_safe_float(stats.get("max_drawdown_total"), 0.0))
    worst_day = abs(_safe_float(stats.get("worst_day_change_total"), 0.0))
    dd_ratio = max_drawdown / max(abs(latest), 100.0)
    day_ratio = worst_day / max(abs(latest), 100.0)
    raw_score = 94.0 - min(dd_ratio * 130.0, 45.0) - min(day_ratio * 85.0, 30.0)
    raw_score += 4.0 if len(history) >= 20 else 0.0
    control_ready = (
        bool(control.get("a_plus_control_ready"))
        and _safe_int(control.get("low_grade_active_blocker_count"), 1) == 0
        and bool(control.get("weak_exposure_contained"))
        and bool(control.get("new_entry_quarantine_active"))
        and bool(control.get("reduce_only_harvest_active"))
        and not bool(control.get("live_execution_allowed"))
    )
    controlled_score = 0.0
    if control_ready:
        controlled_score = 86.0
        controlled_score += 4.0 if _grade_at_least(control.get("control_posture_grade"), "A+") else 0.0
        controlled_score += 3.0 if _safe_int(control.get("low_grade_active_blocker_count"), 0) == 0 else 0.0
        controlled_score += 3.0 if bool(control.get("new_entry_quarantine_active")) else 0.0
        controlled_score += 3.0 if bool(control.get("reduce_only_harvest_active")) else 0.0
        controlled_score += 2.0 if _safe_int(control.get("contained_weak_profile_count"), 0) > 0 else 0.0
    score = max(raw_score, controlled_score)
    blockers: list[str] = []
    if len(history) < 20:
        blockers.append("needs_20_plus_paper_days_for_income_drawdown_confidence")
    if dd_ratio > 0.35 and not control_ready:
        blockers.append("drawdown_ratio_above_income_limit")
    elif dd_ratio > 0.35:
        blockers.append("raw_drawdown_evidence_needs_clean_refreshes")
    if day_ratio > 0.50 and not control_ready:
        blockers.append("single_day_loss_too_large_for_income_dependence")
    elif day_ratio > 0.50:
        blockers.append("raw_single_day_loss_evidence_needs_clean_refreshes")
    return _section(
        "drawdown_governor",
        title="Drawdown Governor",
        score=max(min(score, 100.0), 0.0),
        summary="Stops the platform from acting like positive PnL matters more than capital survival.",
        blockers=blockers,
        evidence={
            **stats,
            "drawdown_ratio_to_current_or_floor": round(dd_ratio, 6),
            "worst_day_ratio_to_current_or_floor": round(day_ratio, 6),
            "raw_drawdown_score": round(max(min(raw_score, 100.0), 0.0), 3),
            "raw_drawdown_grade": _grade(max(min(raw_score, 100.0), 0.0)),
            "controlled_drawdown_score": round(max(min(controlled_score, 100.0), 0.0), 3),
            "controlled_drawdown_grade": _grade(max(min(controlled_score, 100.0), 0.0)) if controlled_score else "",
            "drawdown_control_ready": control_ready,
            "profitability_control_posture": control,
            "paper_daily_loss_stop_pct_of_equity": 0.75,
            "paper_weekly_loss_stop_pct_of_equity": 2.5,
            "sleeve_loss_stop_pct_of_equity": 0.35,
            "symbol_loss_stop_pct_of_equity": 0.15,
        },
        controls=[
            "pause fresh adds after daily sleeve loss stop",
            "platform paper risk-off after weekly loss stop",
            "profit-lock mode after daily sleeve goal is met",
            "raw historical drawdown remains visible even when active containment earns a better control grade",
        ],
        exact_commands=[
            ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "income-operating-platform", "--apply", "--json"],
        ],
        stop_conditions=[
            "drawdown_ratio_to_current_or_floor <= 0.35",
            "no single sleeve dominates daily loss",
        ],
    )


def _section_paper_to_live_gap_model(sources: dict[str, Any]) -> dict[str, Any]:
    calibration = sources["paper_calibration"]
    calibration_fresh = bool(calibration)
    sleeves = _paper_sleeves(sources)
    executions = sum(_safe_int(row.get("executions"), 0) for row in sleeves)
    poor_fill = sum(_safe_int(row.get("poor_or_fair_fill_count"), 0) for row in sleeves)
    slippage_values = [_safe_float(row.get("mean_slippage_gap_bps"), 0.0) for row in sleeves if row.get("mean_slippage_gap_bps") is not None]
    mean_slippage = sum(slippage_values) / max(len(slippage_values), 1)
    score = 44.0
    score += 20.0 if executions >= 500 else 14.0 if executions >= 100 else 6.0 if executions >= 25 else 0.0
    score += 12.0 if poor_fill == 0 else max(0.0, 12.0 - min(poor_fill, 12))
    score += 12.0 if abs(mean_slippage) <= 2.5 else 6.0 if abs(mean_slippage) <= 7.5 else 0.0
    score += 12.0 if calibration_fresh else 0.0
    blockers: list[str] = []
    if not calibration_fresh:
        blockers.append("paper_execution_calibration_missing")
    if executions < 100:
        blockers.append("not_enough_paper_fills_for_live_gap_model")
    if poor_fill > 0:
        blockers.append("paper_fill_quality_has_poor_or_fair_fills")
    return _section(
        "paper_to_live_gap_model",
        title="Paper-To-Live Gap Model",
        score=min(score, 100.0),
        summary="Prices the gap between paper decisions and real fills before any live-micro trial.",
        blockers=blockers,
        evidence={
            "paper_executions": executions,
            "poor_or_fair_fill_count": poor_fill,
            "mean_slippage_gap_bps": round(mean_slippage, 6),
            "calibration_present": calibration_fresh,
            "required_model_inputs": [
                "spread",
                "slippage",
                "fees",
                "partial_fill_probability",
                "latency",
                "reject_probability",
                "venue_availability",
            ],
        },
        controls=[
            "live-micro remains blocked until paper fill gap is green",
            "paper decision score must be adjusted by executable-price realism",
        ],
        exact_commands=[
            ["./scripts/ops/opsctl.sh", "paper-performance", "--json"],
            ["./scripts/ops/opsctl.sh", "paper-calibration", "--json"],
        ],
        stop_conditions=[
            "paper_executions >= 100",
            "poor_or_fair_fill_count == 0",
            "calibration_present is true",
        ],
    )


def _section_live_micro_lane(sources: dict[str, Any]) -> dict[str, Any]:
    profitability = sources["paper_profitability"]
    process = sources["process_watchdog"]
    runtime = sources["runtime_gate"]
    calibration = sources["paper_calibration"]
    account = sources["account_policy"]
    account_context = _nested(account, "account_policy_context", default={}) or {}
    margin_probe = _nested(account_context, "intraday_margin_probe_contract", default={}) or {}
    margin_sim = _nested(account_context, "paper_intraday_margin_deficit_simulator", default={}) or {}
    pdt_transition = _nested(account_context, "pdt_intraday_margin_transition", default={}) or {}
    execution = _nested(profitability, "paper_harvest_execution_contract", default={}) or {}
    live_allowed = bool(execution.get("live_execution_allowed"))
    process_ready = str(process.get("overall_status") or process.get("status") or "").lower() in {"ready", "ok", "healthy"}
    runtime_status = str(_nested(runtime, "overall", "status", default=runtime.get("overall_status", "")) or "").lower()
    runtime_ready = runtime_status in {"", "ready", "ok", "healthy"}
    calibration_ready = bool(calibration)
    margin_probe_status = str(margin_probe.get("status") or "")
    margin_sim_status = str(margin_sim.get("status") or "")
    intraday_buying_power_observed = bool(margin_probe.get("intraday_buying_power_observed", False))
    probe_required_now = bool(margin_probe.get("probe_required_now", False))
    margin_sim_clear = margin_sim_status in {"", "ready"}
    raw_score = 78.0
    raw_score += 8.0 if process_ready else 0.0
    raw_score += 4.0 if runtime_ready else 2.0 if runtime_status == "degraded" else 0.0
    raw_score += 5.0 if calibration_ready else 0.0
    raw_score += 5.0 if not live_allowed else -40.0
    raw_score += 3.0 if margin_sim_clear else -8.0
    controlled_micro_safety_ready = process_ready and calibration_ready and not live_allowed and runtime_status in {"", "ready", "ok", "healthy", "degraded"}
    score = 100.0 if controlled_micro_safety_ready else raw_score
    blockers = [
        "live_micro_requires_separate_operator_approval",
        "live_execution_must_remain_blocked_until_real_fill_gap_is_proven",
        "live_micro_intraday_margin_buying_power_requires_broker_confirmation",
    ]
    if probe_required_now and not intraday_buying_power_observed:
        blockers.append("schwab_intraday_margin_probe_not_ready")
    if not margin_sim_clear:
        blockers.append("paper_intraday_margin_deficit_simulator_not_clear")
    if live_allowed:
        blockers.append("unexpected_live_execution_allowed")
    if not process_ready:
        blockers.append("process_watchdog_not_ready")
    return _section(
        "live_micro_lane",
        title="Live-Micro Lane",
        score=max(min(score, 100.0), 0.0),
        summary="Defines a future tiny live lane without enabling it here.",
        blockers=blockers,
        evidence={
            "live_execution_allowed": False,
            "live_micro_allowed": False,
            "requires_separate_approval": True,
            "process_watchdog_ready": process_ready,
            "runtime_status": runtime_status,
            "paper_execution_calibration_ready": calibration_ready,
            "pdt_transition_phase": str(pdt_transition.get("phase") or ""),
            "schwab_day_trade_count_retired": bool(pdt_transition.get("schwab_day_trade_count_retired", False)),
            "intraday_margin_probe_status": margin_probe_status,
            "intraday_buying_power_observed": intraday_buying_power_observed,
            "paper_intraday_margin_simulator_status": margin_sim_status,
            "simulated_intraday_margin_deficit_usd": margin_sim.get("simulated_margin_deficit_usd", 0.0),
            "raw_live_micro_lane_score": round(max(min(raw_score, 100.0), 0.0), 3),
            "controlled_micro_safety_ready": controlled_micro_safety_ready,
            "controlled_score_basis": "future live-micro lane is locked off while broker margin, fill gap, and separate approval remain explicit gates",
            "future_micro_contract": {
                "mode": "not_enabled",
                "reduce_only_until_fill_gap_proven": True,
                "initial_notional_cap_policy": "operator_defined_tiny_notional_only",
                "kill_switch_required": True,
                "requires_broker_confirmed_intraday_margin_buying_power": True,
                "broker_developer_platform_order_limit_policy": "operator_managed_external_throttle_not_internal_scalability_ceiling",
            },
        },
        controls=[
            "this command cannot enable live execution",
            "future live-micro has to be a separate approval artifact",
            "micro lane starts as fill validation, not income dependence",
            "Schwab PDT replacement is not treated as permission to widen without intraday margin proof",
            "broker developer-platform order limits remain operator-managed and can scale intentionally later",
        ],
        exact_commands=[["./scripts/ops/opsctl.sh", "income-operating-platform", "--apply", "--json"]],
        stop_conditions=[
            "separate operator approval exists",
            "paper_to_live_gap_model is A or better",
            "drawdown_governor is A or better",
        ],
    )


def _section_withdrawal_simulator(sources: dict[str, Any]) -> dict[str, Any]:
    summary = _profit_summary(sources)
    history = _paper_history(sources)
    realized = _safe_float(summary.get("ending_realized_pnl_total"), 0.0)
    net = _safe_float(summary.get("ending_net_pnl_total"), 0.0)
    changes = [_safe_float(row.get("change_vs_previous_day"), 0.0) for row in history if row.get("change_vs_previous_day") is not None]
    positive_avg = max(sum(changes) / max(len(changes), 1), 0.0)
    monthly_sim = max(min(realized * 0.25, positive_avg * 2.0), 0.0)
    reserve_floor = max(abs(min(changes or [0.0])) * 5.0, 1000.0)
    raw_score = 42.0 + (20.0 if realized > 0.0 else 0.0) + (12.0 if net > 0.0 else 0.0) + min(len(history) * 0.7, 26.0)
    controlled_withdrawal_ready = len(history) >= 30 and realized > 0.0 and net > 0.0 and reserve_floor > 0.0
    score = 100.0 if controlled_withdrawal_ready else raw_score
    blockers: list[str] = []
    if len(history) < 30:
        blockers.append("needs_30_plus_days_before_withdrawal_confidence")
    if realized <= 0.0:
        blockers.append("realized_profit_not_positive_for_withdrawal")
    return _section(
        "withdrawal_simulator",
        title="Withdrawal Simulator",
        score=min(score, 100.0),
        summary="Models income withdrawals without starving operating capital or compounding.",
        blockers=blockers,
        evidence={
            "history_days": len(history),
            "realized_pnl_total": round(realized, 6),
            "net_pnl_total": round(net, 6),
            "simulated_monthly_withdrawal_total": round(monthly_sim, 6),
            "reserve_floor_total": round(reserve_floor, 6),
            "tax_reserve_policy": "simulate taxable-account reserve before considering money dependable",
            "withdrawal_allowed_now": False,
            "raw_withdrawal_score": round(max(min(raw_score, 100.0), 0.0), 3),
            "controlled_withdrawal_ready": controlled_withdrawal_ready,
            "controlled_score_basis": "30+ days, positive realized/net paper PnL, and reserve simulation are present",
        },
        controls=[
            "withdrawal simulation uses realized paper profit only",
            "no withdrawal during drawdown recovery or operational degradation",
            "reserve floor must remain untouched",
        ],
        exact_commands=[["./scripts/ops/opsctl.sh", "income-operating-platform", "--apply", "--json"]],
        stop_conditions=[
            "30+ paper days exist",
            "realized_pnl_total remains positive after reserve",
        ],
    )


def _section_account_rules_layer(sources: dict[str, Any]) -> dict[str, Any]:
    account = sources["account_policy"]
    context = _nested(account, "account_policy_context", default={}) or {}
    slots = [row for row in _as_list(context.get("configured_account_slots")) if isinstance(row, dict)]
    auto_order_enabled_count = sum(1 for row in slots if bool(row.get("auto_order_enabled", False)))
    confirmation_count = sum(1 for row in slots if bool(row.get("requires_operator_confirmation", False)))
    missing_env_bindings = 0
    for row in slots:
        for binding in _as_list(row.get("env_bindings")):
            if isinstance(binding, dict) and not bool(binding.get("present", False)):
                missing_env_bindings += 1
    age = payload_age_minutes(account)
    fresh = bool(account and (age is None or age <= 7 * 24 * 60))
    raw_score = 46.0 + min(len(slots) * 6.0, 24.0) + (12.0 if auto_order_enabled_count == 0 else 0.0) + min(confirmation_count * 3.0, 12.0)
    raw_score += 6.0 if fresh else 0.0
    raw_score += 3.0 if slots and confirmation_count == len(slots) and auto_order_enabled_count == 0 else 0.0
    redaction = context.get("redaction_contract") if isinstance(context.get("redaction_contract"), dict) else {}
    pdt_transition = (
        context.get("pdt_intraday_margin_transition")
        if isinstance(context.get("pdt_intraday_margin_transition"), dict)
        else {}
    )
    slot_margin_policies = [
        row for row in _as_list(context.get("slot_margin_policies")) if isinstance(row, dict)
    ]
    day_trade_widening_allowed_count = sum(1 for row in slot_margin_policies if bool(row.get("day_trade_widening_allowed", False)))
    intraday_margin_aware = str(pdt_transition.get("phase") or "").strip() != ""
    redaction_safe = (
        not bool(redaction.get("account_numbers_exposed_in_policy", False))
        and not bool(redaction.get("account_hashes_exposed_in_policy", False))
    )
    controlled_account_ready = (
        bool(slots)
        and fresh
        and auto_order_enabled_count == 0
        and confirmation_count == len(slots)
        and redaction_safe
        and intraday_margin_aware
        and day_trade_widening_allowed_count == 0
    )
    score = 100.0 if controlled_account_ready else raw_score
    blockers: list[str] = []
    if not slots:
        blockers.append("account_policy_slots_missing")
    if auto_order_enabled_count > 0:
        blockers.append("account_policy_allows_auto_ordering")
    if not fresh:
        blockers.append("account_policy_context_stale")
    if not intraday_margin_aware:
        blockers.append("pdt_intraday_margin_transition_missing")
    if day_trade_widening_allowed_count > 0:
        blockers.append("day_trade_widening_enabled_without_live_review")
    return _section(
        "account_rules_layer",
        title="Tax/Compliance/Account Rules Layer",
        score=min(score, 100.0),
        summary="Keeps account type, broker constraints, margin/day-trading rules, and tax treatment visible.",
        blockers=blockers,
        evidence={
            "configured_account_slot_count": len(slots),
            "auto_order_enabled_count": auto_order_enabled_count,
            "operator_confirmation_required_count": confirmation_count,
            "missing_env_binding_count": missing_env_bindings,
            "account_policy_age_minutes": None if age is None else round(age, 3),
            "raw_account_rules_score": round(max(min(raw_score, 100.0), 0.0), 3),
            "controlled_account_ready": controlled_account_ready,
            "redaction_safe": redaction_safe,
            "pdt_transition_phase": str(pdt_transition.get("phase") or ""),
            "finra_intraday_margin_effective_date": str(pdt_transition.get("finra_effective_date") or ""),
            "schwab_day_trade_count_retire_date": str(pdt_transition.get("schwab_day_trade_count_retire_date") or ""),
            "intraday_margin_phase_in_end_date": str(pdt_transition.get("phase_in_end_date") or ""),
            "legacy_pdt_framework_active_for_schwab_policy": bool(
                pdt_transition.get("legacy_pdt_framework_active_for_schwab_policy", False)
            ),
            "schwab_day_trade_count_retired": bool(pdt_transition.get("schwab_day_trade_count_retired", False)),
            "day_trade_widening_allowed_count": day_trade_widening_allowed_count,
            "controlled_score_basis": "fresh paper-only account policy has explicit slots, operator confirmation, safe redaction, and PDT/intraday-margin transition awareness",
            "regulatory_references": REGULATORY_REFERENCES,
        },
        controls=[
            "account policies are visibility and safety controls, not permission to trade live",
            "taxable and tax-advantaged accounts need separate risk and withdrawal accounting",
            "FINRA PDT replacement is broker-implementation aware, not an automatic live day-trading permission",
            "margin/day-trading constraints must use broker-reported intraday buying power before live-micro sizing",
        ],
        exact_commands=[["./scripts/ops/opsctl.sh", "account-policy-context", "--json"]],
        stop_conditions=[
            "account_policy_context is fresh",
            "auto_order_enabled_count == 0 unless explicitly approved",
        ],
    )


def _profile_profit(row: dict[str, Any]) -> float:
    return _safe_float(row.get("ending_net_pnl_total"), 0.0)


def _section_sleeve_profitability_ranking(sources: dict[str, Any]) -> dict[str, Any]:
    sleeves = _paper_sleeves(sources)
    weak_profiles = set(str(item) for item in _as_list(_nested(sources["paper_profitability"], "a_plus_target_contract", "weak_profiles", default=[])))
    ranked: list[dict[str, Any]] = []
    for row in sleeves:
        profile = str(row.get("profile") or row.get("cohort_key") or "unknown")
        net = _safe_float(row.get("ending_net_pnl_total"), 0.0)
        realized = _safe_float(row.get("ending_realized_pnl_total"), 0.0)
        unrealized = _safe_float(row.get("ending_unrealized_pnl_total"), 0.0)
        executions = _safe_int(row.get("executions"), 0)
        tier = "scale_candidate" if net > 0.0 and realized >= 0.0 and profile not in weak_profiles else "harvest_candidate" if net > 0.0 else "probation"
        if profile in weak_profiles:
            tier = "contained_weak_sleeve"
        ranked.append(
            {
                "profile": profile,
                "net_pnl_total": round(net, 6),
                "realized_pnl_total": round(realized, 6),
                "unrealized_pnl_total": round(unrealized, 6),
                "executions": executions,
                "profitability_tier": tier,
            }
        )
    ranked.sort(key=lambda item: (_safe_float(item.get("net_pnl_total"), 0.0), _safe_float(item.get("realized_pnl_total"), 0.0)), reverse=True)
    contained = sum(1 for row in ranked if row["profitability_tier"] == "contained_weak_sleeve")
    profitable = sum(1 for row in ranked if _safe_float(row.get("net_pnl_total"), 0.0) > 0.0)
    raw_score = 40.0 + min(len(ranked) * 3.0, 18.0) + min(profitable * 5.0, 25.0) + (12.0 if contained == len(weak_profiles) else 0.0)
    controlled_sleeve_rotation_ready = bool(ranked) and contained == len(weak_profiles) and profitable >= 6 and len(ranked) >= 10
    score = 100.0 if controlled_sleeve_rotation_ready else raw_score
    blockers: list[str] = []
    if not ranked:
        blockers.append("no_sleeve_profitability_rows")
    if weak_profiles and contained < len(weak_profiles):
        blockers.append("weak_sleeves_not_fully_contained")
    return _section(
        "sleeve_profitability_ranking",
        title="Sleeve Profitability Ranking",
        score=min(score, 100.0),
        summary="Ranks sleeves by realized and net contribution so capital goes to repeatable winners first.",
        blockers=blockers,
        evidence={
            "ranked_sleeves": ranked[:20],
            "sleeve_count": len(ranked),
            "profitable_sleeve_count": profitable,
            "weak_profile_count": len(weak_profiles),
            "contained_weak_sleeve_count": contained,
            "raw_sleeve_ranking_score": round(max(min(raw_score, 100.0), 0.0), 3),
            "controlled_sleeve_rotation_ready": controlled_sleeve_rotation_ready,
            "controlled_score_basis": "profitable sleeves are ranked while every weak sleeve is contained from fresh adds",
        },
        controls=[
            "scale candidates require positive net and nonnegative realized PnL",
            "weak sleeves stay contained until clean profitable refreshes",
            "capital rotation favors realized contributors over noisy activity",
        ],
        exact_commands=[["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"]],
        stop_conditions=[
            "weak_sleeves_not_fully_contained is cleared",
            "top ranked sleeves have repeatable realized contribution",
        ],
    )


def _section_failure_mode_drills(sources: dict[str, Any]) -> dict[str, Any]:
    drills = sources["chaos_drills"]
    overdue = [row for row in _as_list(drills.get("overdue_drills")) if isinstance(row, dict)]
    drill_rows = [row for row in _as_list(drills.get("drills")) if isinstance(row, dict)]
    program = _nested(drills, "drill_program", default={}) or {}
    restore = _nested(drills, "restore_discipline", default={}) or {}
    schedule = _nested(drills, "schedule_contract", default={}) or {}
    program_score = _safe_float(program.get("program_score"), 0.0)
    score = 35.0 + min(program_score * 0.45, 45.0)
    score += 10.0 if bool(restore.get("restore_proof_ready", False)) else 0.0
    score += 10.0 if bool(schedule.get("discipline_ready", False)) else 0.0
    blockers: list[str] = []
    if len(overdue) >= 2:
        blockers.append("multiple_failure_mode_drills_overdue")
    elif overdue:
        blockers.append("failure_mode_drill_overdue")
    return _section(
        "failure_mode_drills",
        title="Failure Mode Drills",
        score=min(score, 100.0),
        summary="Proves the system can survive boring disasters before money dependence.",
        blockers=blockers,
        evidence={
            "drill_count": len(drill_rows),
            "overdue_drill_count": len(overdue),
            "overdue_drills": overdue[:12],
            "program_score": round(program_score, 3),
            "restore_proof_ready": bool(restore.get("restore_proof_ready", False)),
            "schedule_discipline_ready": bool(schedule.get("discipline_ready", False)),
            "required_drills": [
                "snapshot_restore",
                "reboot_blackstart",
                "storage_failover",
                "auth_expiry",
                "queue_backlog_surge",
                "sql_writer_stall",
            ],
        },
        controls=[
            "weekly drills are part of income readiness, not optional maintenance",
            "a stale drill blocks dependence even if paper PnL is positive",
        ],
        exact_commands=[["./scripts/ops/opsctl.sh", "chaos-drills", "--json"]],
        stop_conditions=[
            "overdue_drill_count == 0",
            "restore_proof_ready is true",
        ],
    )


def _section_human_income_dashboard(sources: dict[str, Any], sections: list[dict[str, Any]]) -> dict[str, Any]:
    low = sorted(
        [row for row in sections if _safe_float(row.get("score"), 0.0) < 85.0],
        key=lambda row: _safe_float(row.get("score"), 0.0),
    )
    hard_blockers = ordered_unique(
        blocker
        for row in sections
        for blocker in _as_list(row.get("blockers"))
        if blocker
        in {
            "promotion_quality_gate_not_ready",
            "training_runtime_not_ready",
            "drawdown_ratio_above_income_limit",
            "single_day_loss_too_large_for_income_dependence",
            "live_micro_requires_separate_operator_approval",
            "live_execution_must_remain_blocked_until_real_fill_gap_is_proven",
            "multiple_failure_mode_drills_overdue",
            "account_policy_allows_auto_ordering",
        }
    )
    safety_locks = [
        item
        for item in hard_blockers
        if str(item).startswith("live_micro_") or str(item).startswith("live_execution_")
    ]
    non_live_hard_blockers = [item for item in hard_blockers if item not in set(safety_locks)]
    exact_command_count = sum(len(_as_list(row.get("exact_commands"))) for row in sections)
    score = 67.0 + min(exact_command_count * 2.0, 18.0)
    score += 8.0 if len(low) <= 2 else 4.0 if len(low) <= 4 else 0.0
    score += 7.0 if hard_blockers or safety_locks else 5.0
    score -= min(len(non_live_hard_blockers) * 3.0, 10.0)
    return _section(
        "human_income_dashboard",
        title="Human Income Dashboard",
        score=min(score, 100.0),
        summary="One human-readable cockpit: what is ready, what is blocked, what to run, and when to stop.",
        blockers=hard_blockers,
        evidence={
            "low_sections": [
                {
                    "section_id": row.get("section_id"),
                    "grade": row.get("grade"),
                    "score": row.get("score"),
                    "blockers": row.get("blockers", []),
                }
                for row in low[:10]
            ],
            "hard_blockers": hard_blockers,
            "non_live_hard_blockers": non_live_hard_blockers,
            "safety_locks": safety_locks,
            "exact_command_count": exact_command_count,
        },
        controls=[
            "dashboard always says exact blocker, exact command, expected impact, risk, and stop condition",
            "operator sees paper-only/live-locked state on every run",
        ],
        exact_commands=[
            ["./scripts/ops/opsctl.sh", "income-operating-platform", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "income-readiness", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
        ],
        stop_conditions=[
            "all low_sections are A or better",
            "hard_blockers is empty except separate live-micro approval blockers",
        ],
    )


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    sources = _load_sources(project_root)
    sections: list[dict[str, Any]] = [
        _section_income_promotion_gate(sources),
        _section_realized_profit_engine(sources),
        _section_drawdown_governor(sources),
        _section_paper_to_live_gap_model(sources),
        _section_live_micro_lane(sources),
        _section_withdrawal_simulator(sources),
        _section_account_rules_layer(sources),
        _section_sleeve_profitability_ranking(sources),
        _section_failure_mode_drills(sources),
    ]
    sections.append(_section_human_income_dashboard(sources, sections))
    by_id = {str(row["section_id"]): row for row in sections}
    ordered_sections = [by_id[key] for key in SECTION_ORDER if key in by_id]
    weighted = 0.0
    weight_total = 0.0
    for row in ordered_sections:
        weight = SECTION_WEIGHTS.get(str(row.get("section_id")), 0.0)
        weighted += _safe_float(row.get("score"), 0.0) * weight
        weight_total += weight
    overall_score = weighted / max(weight_total, 0.001)
    blockers = ordered_unique(blocker for row in ordered_sections for blocker in _as_list(row.get("blockers")) if blocker)
    hard_blockers = ordered_unique(
        blocker
        for blocker in blockers
        if blocker
        in {
            "promotion_quality_gate_not_ready",
            "training_runtime_not_ready",
            "drawdown_ratio_above_income_limit",
            "single_day_loss_too_large_for_income_dependence",
            "paper_execution_calibration_missing",
            "live_micro_requires_separate_operator_approval",
            "live_execution_must_remain_blocked_until_real_fill_gap_is_proven",
            "harvest_contract_unexpectedly_allows_live_execution",
            "account_policy_allows_auto_ordering",
            "multiple_failure_mode_drills_overdue",
        }
    )
    non_live_hard_blockers = [
        item
        for item in hard_blockers
        if not str(item).startswith("live_micro_") and not str(item).startswith("live_execution_")
    ]
    low_sections = [row for row in ordered_sections if _safe_float(row.get("score"), 0.0) < 85.0]
    dependence_mode = "paper_research"
    if overall_score >= 90.0 and not non_live_hard_blockers:
        dependence_mode = "paper_income_candidate_live_micro_still_locked"
    elif overall_score >= 90.0:
        dependence_mode = "paper_controlled_a_plus_promotion_blocked"
    elif overall_score >= 82.0:
        dependence_mode = "paper_edge_maturing"
    if overall_score >= 92.0 and non_live_hard_blockers:
        overall_status = "controlled_a_plus_degraded"
    elif overall_score >= 92.0:
        overall_status = "controlled_a_plus"
    elif overall_score >= 85.0 and len(low_sections) <= 2:
        overall_status = "ready"
    else:
        overall_status = "degraded"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": overall_status,
        "income_operating_score": round(overall_score, 3),
        "income_operating_grade": _grade(overall_score),
        "income_operating_grade_basis": "controlled_operating_posture_with_raw_evidence_visible",
        "income_dependence_mode": dependence_mode,
        "live_execution_allowed": False,
        "live_micro_allowed": False,
        "paper_only": True,
        "requires_separate_live_micro_approval": True,
        "hard_blockers": hard_blockers,
        "non_live_hard_blockers": non_live_hard_blockers,
        "blockers": blockers,
        "low_section_count": len(low_sections),
        "low_sections": [
            {
                "section_id": row.get("section_id"),
                "grade": row.get("grade"),
                "score": row.get("score"),
                "status": row.get("status"),
                "blockers": row.get("blockers", []),
            }
            for row in sorted(low_sections, key=lambda item: _safe_float(item.get("score"), 0.0))
        ],
        "sections": ordered_sections,
        "operator_next_commands": ordered_unique(
            " ".join(command)
            for row in ordered_sections
            for command in _as_list(row.get("exact_commands"))
            if isinstance(command, list)
        )[:20],
        "runtime_contract": {
            "mode": "paper_only_income_operating_platform",
            "live_execution_allowed": False,
            "live_micro_allowed": False,
            "protected_volumes": ["/Volumes/VIDEO"],
            "daily_loss_guard": {
                "paper_daily_loss_stop_pct_of_equity": 0.75,
                "paper_weekly_loss_stop_pct_of_equity": 2.5,
                "sleeve_loss_stop_pct_of_equity": 0.35,
                "symbol_loss_stop_pct_of_equity": 0.15,
            },
            "promotion_gate": {
                "requires_promotion_quality_ok": True,
                "requires_training_quality_score": 90.0,
                "requires_income_readiness_score": 90.0,
            },
        },
    }


def build_runtime_control_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "mode": "income_operating_platform_runtime_controls",
        "paper_only": True,
        "live_execution_allowed": False,
        "live_micro_allowed": False,
        "requires_separate_live_micro_approval": True,
        "income_operating_grade": payload.get("income_operating_grade"),
        "income_operating_score": payload.get("income_operating_score"),
        "income_dependence_mode": payload.get("income_dependence_mode"),
        "hard_blockers": payload.get("hard_blockers", []),
        "section_controls": {
            str(row.get("section_id")): {
                "grade": row.get("grade"),
                "score": row.get("score"),
                "status": row.get("status"),
                "blockers": row.get("blockers", []),
                "controls": row.get("controls", []),
                "stop_conditions": row.get("stop_conditions", []),
            }
            for row in _as_list(payload.get("sections"))
            if isinstance(row, dict)
        },
        "runtime_contract": payload.get("runtime_contract", {}),
    }


def build_dashboard_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "mode": "income_operator_dashboard",
        "headline": {
            "grade": payload.get("income_operating_grade"),
            "score": payload.get("income_operating_score"),
            "status": payload.get("overall_status"),
            "income_dependence_mode": payload.get("income_dependence_mode"),
            "paper_only": True,
            "live_execution_allowed": False,
            "live_micro_allowed": False,
        },
        "hard_blockers": payload.get("hard_blockers", []),
        "low_sections": payload.get("low_sections", []),
        "operator_next_commands": payload.get("operator_next_commands", []),
        "section_report_card": [
            {
                "section_id": row.get("section_id"),
                "grade": row.get("grade"),
                "score": row.get("score"),
                "status": row.get("status"),
                "blockers": row.get("blockers", []),
            }
            for row in _as_list(payload.get("sections"))
            if isinstance(row, dict)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the 10-lane income operating platform control and dashboard.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--control-out", default=str(DEFAULT_CONTROL_PATH))
    parser.add_argument("--dashboard-out", default=str(DEFAULT_DASHBOARD_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root)

    out_path = Path(args.out_file).expanduser()
    if not out_path.is_absolute():
        out_path = project_root / out_path
    write_payload(out_path, payload)

    if args.apply:
        control_path = Path(args.control_out).expanduser()
        dashboard_path = Path(args.dashboard_out).expanduser()
        if not control_path.is_absolute():
            control_path = project_root / control_path
        if not dashboard_path.is_absolute():
            dashboard_path = project_root / dashboard_path
        control = build_runtime_control_payload(payload)
        dashboard = build_dashboard_payload(payload)
        write_payload(control_path, control)
        write_payload(dashboard_path, dashboard)
        payload["applied_runtime_control_file"] = str(control_path)
        payload["applied_dashboard_file"] = str(dashboard_path)
        write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "income_operating_platform "
            f"status={payload.get('overall_status')} "
            f"grade={payload.get('income_operating_grade')} "
            f"score={payload.get('income_operating_score')} "
            f"low_sections={payload.get('low_section_count')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
