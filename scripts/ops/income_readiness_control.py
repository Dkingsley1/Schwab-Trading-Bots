#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "income_readiness_latest.json"
DEFAULT_CONTROL_PATH = PROJECT_ROOT / "governance" / "health" / "income_readiness_runtime_controls_latest.json"
DEFAULT_BOT_LOGS_ROOT = Path("/Volumes/BOT_LOGS/schwab_trading_bot")

SECTION_ORDER = [
    "income_readiness_scorecard",
    "paper_vs_real_fill_gap",
    "realized_profit_discipline",
    "drawdown_governor",
    "bot_attribution",
    "regime_proof",
    "live_micro_readiness",
    "withdrawal_simulation",
    "operational_boringness",
    "promotion_rules_for_money",
]

SECTION_WEIGHTS = {
    "income_readiness_scorecard": 0.08,
    "paper_vs_real_fill_gap": 0.12,
    "realized_profit_discipline": 0.14,
    "drawdown_governor": 0.12,
    "bot_attribution": 0.10,
    "regime_proof": 0.10,
    "live_micro_readiness": 0.10,
    "withdrawal_simulation": 0.08,
    "operational_boringness": 0.10,
    "promotion_rules_for_money": 0.06,
}

MIN_BOT_LOGS_FREE_GB = 125.0
WARN_BOT_LOGS_FREE_GB = 150.0


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


def _round(value: Any, digits: int = 3) -> float:
    return round(_safe_float(value), digits)


def _grade(score: float) -> str:
    score = _safe_float(score)
    if score >= 97.0:
        return "A++"
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
    if blockers:
        return "blocked" if score < 65.0 else "degraded"
    if score >= 85.0:
        return "ready"
    if score >= 70.0:
        return "needs_work"
    if score >= 50.0:
        return "degraded"
    return "blocked"


def _section(
    section_id: str,
    *,
    score: float,
    title: str,
    summary: str,
    blockers: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
    controls: list[str] | None = None,
    next_actions: list[str] | None = None,
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
        "next_actions": ordered_unique(next_actions or []),
    }


def _load_sources(project_root: Path, bot_logs_root: Path) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    candidate_calibration_files = [
        health / "paper_execution_calibration_latest.json",
        health / "paper_calibration_latest.json",
        health / "paper_execution_reconciliation_latest.json",
    ]
    calibration = {}
    calibration_path = ""
    for path in candidate_calibration_files:
        payload = load_json(path)
        if payload:
            calibration = payload
            calibration_path = str(path)
            break
    return {
        "paper_profitability": load_json(health / "paper_profitability_control_latest.json"),
        "paper_performance": load_json(health / "paper_performance_latest.json"),
        "paper_calibration": calibration,
        "paper_calibration_path": calibration_path,
        "training_runtime": load_json(health / "training_runtime_control_latest.json"),
        "training_quality": load_json(health / "training_quality_control_latest.json"),
        "promotion_quality": load_json(health / "promotion_quality_gate_latest.json"),
        "backpressure": load_json(health / "ingestion_backpressure_latest.json"),
        "process_watchdog": load_json(health / "process_watchdog_latest.json"),
        "runtime_gate": load_json(health / "runtime_gate_dashboard_latest.json"),
        "storage_quota": load_json(health / "storage_quota_guard_latest.json"),
        "artifact_freshness": load_json(health / "artifact_freshness_slo_latest.json"),
        "bot_logs_cleanup": load_json(health / "bot_logs_cleanup_intelligence_latest.json"),
        "global_halt": load_json(health / "global_killswitch_latest.json"),
        "bot_logs_disk": _disk_snapshot(bot_logs_root),
    }


def _disk_snapshot(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
    except Exception:
        return {
            "path": str(path),
            "exists": path.exists(),
            "mounted": False,
            "free_gb": 0.0,
            "total_gb": 0.0,
            "used_gb": 0.0,
            "capacity_pct": 100.0,
        }
    total_gb = usage.total / (1024.0**3)
    free_gb = usage.free / (1024.0**3)
    used_gb = usage.used / (1024.0**3)
    capacity = 100.0 * used_gb / max(total_gb, 0.001)
    return {
        "path": str(path),
        "exists": path.exists(),
        "mounted": True,
        "free_gb": round(free_gb, 3),
        "total_gb": round(total_gb, 3),
        "used_gb": round(used_gb, 3),
        "capacity_pct": round(capacity, 3),
    }


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def _paper_series(paper_performance: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in _as_list(paper_performance.get("history_daily_series")) if isinstance(row, dict)]
    rows.sort(key=lambda row: str(row.get("day_utc") or row.get("day") or ""))
    return rows


def _drawdown_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    peak = None
    max_drawdown = 0.0
    worst_day_change = 0.0
    latest = 0.0
    for row in rows:
        value = _safe_float(row.get("ending_net_pnl_total"), 0.0)
        latest = value
        peak = value if peak is None else max(peak, value)
        max_drawdown = min(max_drawdown, value - (peak if peak is not None else value))
        change = _safe_float(row.get("change_vs_previous_day"), 0.0)
        worst_day_change = min(worst_day_change, change)
    return {
        "history_points": len(rows),
        "latest_net_pnl_total": round(latest, 6),
        "max_drawdown_pnl_total": round(max_drawdown, 6),
        "worst_day_change_total": round(worst_day_change, 6),
    }


def _grade_at_least(grade: Any, target: str) -> bool:
    ranks = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4, "A+": 5, "A++": 6}
    return ranks.get(str(grade or "").strip().upper(), -1) >= ranks.get(target, 999)


def _profitability_control_posture(sources: dict[str, Any]) -> dict[str, Any]:
    profitability = sources["paper_profitability"]
    low_grade = _nested(profitability, "low_grade_control_report_card", default={}) or {}
    containment = _nested(profitability, "raw_operational_containment_filter", default={}) or {}
    hardening = _nested(profitability, "paper_profitability_hardening_contract", default={}) or {}
    execution = _nested(profitability, "paper_harvest_execution_contract", default={}) or {}
    target = _nested(profitability, "a_plus_target_contract", "current", default={}) or {}
    unprotected_weak = _safe_int(target.get("unprotected_weak_profile_count"), 0)
    unprotected_strategy = _safe_int(target.get("unprotected_strategy_control_count"), 0)
    control_grade = str(low_grade.get("control_posture_grade") or containment.get("contained_grade") or "")
    return {
        "active": bool(profitability),
        "control_posture_grade": control_grade,
        "a_plus_control_ready": bool(low_grade.get("a_plus_control_ready", False)) or _grade_at_least(control_grade, "A+"),
        "low_grade_active_blocker_count": _safe_int(low_grade.get("active_blocker_count"), 0),
        "contained_weak_profile_count": _safe_int(containment.get("contained_weak_profile_count"), 0),
        "contained_strategy_control_count": _safe_int(containment.get("contained_strategy_control_count"), 0),
        "weak_exposure_contained": unprotected_weak == 0 and unprotected_strategy == 0,
        "new_entry_quarantine_active": bool(_nested(hardening, "new_entry_policy", "block_quarantined_profiles", default=False)),
        "reduce_only_harvest_active": bool(execution.get("reduce_only")) and bool(execution.get("paper_only", True)) and not bool(execution.get("live_execution_allowed")),
        "live_execution_allowed": bool(execution.get("live_execution_allowed")),
    }


def _training_runtime_control_ready(training_runtime: dict[str, Any]) -> bool:
    status = str(training_runtime.get("overall_status") or "").lower()
    if status in {"ready", "cleared", "ok"}:
        return True
    prep_ready = bool(_nested(training_runtime, "training_launch_contract", "prep_allowed", default=False))
    buffer_safe = bool(_nested(training_runtime, "pretraining_drain_buffer", "safe_to_launch_now", default=False))
    return prep_ready and buffer_safe


def _section_paper_vs_real_fill_gap(sources: dict[str, Any]) -> dict[str, Any]:
    paper = sources["paper_performance"]
    calibration = sources["paper_calibration"]
    sleeves = [row for row in _as_list(paper.get("sleeve_latest")) if isinstance(row, dict)]
    executions = sum(_safe_int(row.get("executions"), 0) for row in sleeves)
    poor_fill = sum(_safe_int(row.get("poor_or_fair_fill_count"), 0) for row in sleeves)
    slippage_values = [_safe_float(row.get("mean_slippage_gap_bps"), 0.0) for row in sleeves if row.get("mean_slippage_gap_bps") is not None]
    mean_slippage = sum(slippage_values) / max(len(slippage_values), 1)
    calibration_age = payload_age_minutes(calibration, Path(sources["paper_calibration_path"])) if calibration else None
    calibration_fresh = bool(calibration and (calibration_age is None or calibration_age <= 24.0 * 60.0))
    score = 56.0
    score += 18.0 if executions >= 100 else 9.0 if executions >= 25 else 0.0
    score += 12.0 if poor_fill == 0 else max(0.0, 12.0 - min(poor_fill, 12))
    score += 8.0 if abs(mean_slippage) <= 2.5 else 3.0 if abs(mean_slippage) <= 7.5 else 0.0
    score += 6.0 if calibration_fresh else 0.0
    blockers: list[str] = []
    if not calibration_fresh:
        blockers.append("paper_execution_calibration_not_fresh_or_missing")
    if executions < 25:
        blockers.append("not_enough_current_paper_fills_for_fill_gap_confidence")
    return _section(
        "paper_vs_real_fill_gap",
        score=min(score, 100.0),
        title="Paper vs Real Fill Gap",
        summary="Measures whether paper fills are credible enough to become live-micro evidence later.",
        blockers=blockers,
        evidence={
            "paper_executions": executions,
            "poor_or_fair_fill_count": poor_fill,
            "mean_slippage_gap_bps": round(mean_slippage, 6),
            "calibration_source": sources["paper_calibration_path"],
            "calibration_age_minutes": None if calibration_age is None else round(calibration_age, 3),
        },
        controls=[
            "paper fills must reconcile before live-micro sizing can increase",
            "slippage and fill-quality labels are required training inputs",
        ],
        next_actions=[
            "refresh paper calibration/reconciliation after active paper sessions",
            "compare paper intended prices against executable broker marks before live micro",
        ],
    )


def _section_realized_profit_discipline(sources: dict[str, Any]) -> dict[str, Any]:
    profitability = sources["paper_profitability"]
    report = _nested(profitability, "profit_harvest_report_card", default={})
    realization = _nested(profitability, "profit_realization_contract", default={})
    execution = _nested(profitability, "paper_harvest_execution_contract", default={})
    paper_summary = _nested(profitability, "paper_summary", default={})
    realized_share = _safe_float(report.get("current_realized_profit_share_norm"), _safe_float(realization.get("realized_profit_share_norm"), 0.0))
    unrealized_share = _safe_float(report.get("current_unrealized_profit_share_norm"), _safe_float(realization.get("unrealized_profit_share_norm"), 0.0))
    target_share = _safe_float(report.get("target_realized_profit_share_norm"), _safe_float(realization.get("target_realized_profit_share_norm"), 0.35))
    harvest_active = bool(realization.get("active")) or bool(execution.get("active"))
    intent_count = _safe_int(execution.get("intent_count"), 0)
    reduce_only = bool(execution.get("reduce_only"))
    live_allowed = bool(execution.get("live_execution_allowed"))
    raw_harvest_grade = str(report.get("raw_outcome_grade") or "")
    score = 35.0
    score += 30.0 * _clamp(realized_share / max(target_share, 0.01))
    score += 15.0 * _clamp((1.0 - unrealized_share) / max(1.0 - _safe_float(realization.get("max_unrealized_profit_share_norm"), 0.70), 0.01))
    score += 10.0 if harvest_active else 0.0
    score += 5.0 if intent_count > 0 else 0.0
    score += 5.0 if reduce_only and not live_allowed else 0.0
    blockers: list[str] = []
    if realized_share < target_share:
        blockers.append("realized_profit_share_below_target")
    if raw_harvest_grade in {"D", "F"}:
        blockers.append("raw_harvest_outcome_low")
    if live_allowed:
        blockers.append("harvest_contract_unexpectedly_allows_live_execution")
    return _section(
        "realized_profit_discipline",
        score=min(score, 100.0),
        title="Realized Profit Discipline",
        summary="Pushes the system to convert paper winners into realized profit without choking runners.",
        blockers=blockers,
        evidence={
            "net_pnl_total": _round(paper_summary.get("ending_net_pnl_total"), 6),
            "realized_pnl_total": _round(paper_summary.get("ending_realized_pnl_total"), 6),
            "unrealized_pnl_total": _round(paper_summary.get("ending_unrealized_pnl_total"), 6),
            "realized_share_norm": round(realized_share, 6),
            "unrealized_share_norm": round(unrealized_share, 6),
            "target_realized_share_norm": round(target_share, 6),
            "harvest_active": harvest_active,
            "paper_reduce_only_intent_count": intent_count,
            "harvest_grade": report.get("grade"),
            "raw_harvest_grade": raw_harvest_grade,
        },
        controls=[
            "harvest intents remain paper-only and reduce-only",
            "fresh adds stay blocked in sleeves with active daily harvest goals",
        ],
        next_actions=[
            "keep running paper-profitability-control --apply after paper fills",
            "raise daily sleeve targets only after previous targets are met cleanly",
        ],
    )


def _section_drawdown_governor(sources: dict[str, Any]) -> dict[str, Any]:
    rows = _paper_series(sources["paper_performance"])
    stats = _drawdown_stats(rows)
    control = _profitability_control_posture(sources)
    net = _safe_float(stats.get("latest_net_pnl_total"), 0.0)
    max_dd = abs(_safe_float(stats.get("max_drawdown_pnl_total"), 0.0))
    worst_day = abs(_safe_float(stats.get("worst_day_change_total"), 0.0))
    dd_ratio = max_dd / max(abs(net), 100.0)
    day_ratio = worst_day / max(abs(net), 100.0)
    raw_score = 92.0
    raw_score -= min(dd_ratio * 120.0, 35.0)
    raw_score -= min(day_ratio * 80.0, 25.0)
    raw_score += 5.0 if len(rows) >= 7 else 0.0
    controlled_drawdown_ready = (
        bool(control.get("a_plus_control_ready"))
        and _safe_int(control.get("low_grade_active_blocker_count"), 1) == 0
        and bool(control.get("weak_exposure_contained"))
        and bool(control.get("new_entry_quarantine_active"))
        and bool(control.get("reduce_only_harvest_active"))
        and not bool(control.get("live_execution_allowed"))
    )
    score = 100.0 if controlled_drawdown_ready else raw_score
    blockers: list[str] = []
    if len(rows) < 7:
        blockers.append("not_enough_recent_days_for_drawdown_confidence")
    if dd_ratio > 0.35:
        blockers.append("raw_paper_drawdown_ratio_needs_clean_refreshes" if controlled_drawdown_ready else "paper_drawdown_ratio_too_high_for_income_dependence")
    return _section(
        "drawdown_governor",
        score=max(min(score, 100.0), 0.0),
        title="Drawdown Governor",
        summary="Defines when the platform must stop adding risk and protect capital.",
        blockers=blockers,
        evidence={
            **stats,
            "drawdown_ratio_to_current_or_floor": round(dd_ratio, 6),
            "worst_day_ratio_to_current_or_floor": round(day_ratio, 6),
            "raw_drawdown_score": round(max(min(raw_score, 100.0), 0.0), 3),
            "raw_drawdown_grade": _grade(max(min(raw_score, 100.0), 0.0)),
            "controlled_drawdown_ready": controlled_drawdown_ready,
            "profitability_control_posture": control,
            "proposed_daily_loss_stop_pct_of_equity": 0.75,
            "proposed_weekly_loss_stop_pct_of_equity": 2.5,
        },
        controls=[
            "daily sleeve loss stops before new entries",
            "weekly platform loss stop pauses training-to-live promotion",
            "profit-lock mode after daily target is reached",
        ],
        next_actions=[
            "keep drawdown limits in paper until live-micro evidence exists",
            "audit every sleeve whose worst-day change dominates portfolio drawdown",
        ],
    )


def _section_bot_attribution(sources: dict[str, Any]) -> dict[str, Any]:
    profitability = sources["paper_profitability"]
    strategy_controls = _nested(profitability, "profit_harvest_strategy_controls", default={})
    regular_controls = _as_list(profitability.get("strategy_controls"))
    position_ledger = _nested(profitability, "profit_harvest_position_ledger", default={})
    positions = _as_list(position_ledger.get("positions"))
    profile_counts = Counter()
    for row in positions:
        if isinstance(row, dict):
            profile_counts[str(row.get("profile") or "unknown")] += 1
    strategy_count = len(strategy_controls) if isinstance(strategy_controls, dict) else 0
    score = 45.0 + min(strategy_count * 7.0, 30.0) + min(len(positions) * 3.0, 15.0) + min(len(regular_controls) * 1.5, 10.0)
    blockers: list[str] = []
    if strategy_count < 3:
        blockers.append("thin_strategy_level_profit_attribution")
    if len(positions) < 10:
        blockers.append("thin_position_ledger_for_income_grade_attribution")
    return _section(
        "bot_attribution",
        score=min(score, 100.0),
        title="Bot Attribution",
        summary="Every dollar should map back to bot, sleeve, symbol, strategy, regime, and data source.",
        blockers=blockers,
        evidence={
            "profit_harvest_strategy_control_count": strategy_count,
            "active_strategy_control_count": len(regular_controls),
            "position_ledger_count": len(positions),
            "position_profiles": dict(sorted(profile_counts.items())),
        },
        controls=[
            "no capital increase without bot-level attribution",
            "losing profile-strategy pairs stay quarantined until replay improves",
        ],
        next_actions=[
            "expand position ledger coverage for paper winners and losers",
            "write bot-level PnL attribution into daily paper reports",
        ],
    )


def _section_regime_proof(sources: dict[str, Any]) -> dict[str, Any]:
    paper = sources["paper_performance"]
    sleeves = [row for row in _as_list(paper.get("sleeve_latest")) if isinstance(row, dict)]
    history = _paper_series(paper)
    active_profiles = _as_list(paper.get("active_paper_profiles_today"))
    latest_statuses = Counter(str(row.get("data_status") or "unknown") for row in sleeves)
    score = 38.0 + min(len(history) * 4.0, 28.0) + min(len(active_profiles) * 4.0, 18.0) + min(len(sleeves) * 1.5, 16.0)
    blockers: list[str] = []
    if len(history) < 10:
        blockers.append("needs_more_market_days_for_regime_proof")
    if len(active_profiles) < 5:
        blockers.append("too_few_active_paper_profiles_today")
    return _section(
        "regime_proof",
        score=min(score, 100.0),
        title="Regime Proof",
        summary="Checks whether profits survive more than one market condition.",
        blockers=blockers,
        evidence={
            "history_days": len(history),
            "active_paper_profile_count_today": len(active_profiles),
            "sleeve_latest_count": len(sleeves),
            "data_status_counts": dict(sorted(latest_statuses.items())),
            "required_regimes": [
                "trend",
                "chop",
                "selloff",
                "crypto_sideways",
                "high_volatility",
                "low_volume",
                "earnings_or_event",
                "boring_day",
            ],
        },
        controls=[
            "promotion requires cross-regime paper evidence",
            "single-regime winners stay capped until they repeat elsewhere",
        ],
        next_actions=[
            "tag paper trades with regime labels and event context",
            "keep collecting across boring and high-volatility sessions before live dependence",
        ],
    )


def _section_live_micro_readiness(sources: dict[str, Any]) -> dict[str, Any]:
    global_halt = sources["global_halt"]
    profitability = sources["paper_profitability"]
    runtime_gate = sources["runtime_gate"]
    process = sources["process_watchdog"]
    live_execution_allowed = bool(_nested(profitability, "paper_harvest_execution_contract", "live_execution_allowed", default=False))
    halt_active = bool(global_halt.get("halt") or global_halt.get("global_halt") or global_halt.get("global_kill_triggered"))
    process_ready = str(process.get("overall_status") or process.get("status") or "").lower() in {"ready", "ok", "healthy"}
    runtime_overall = str(_nested(runtime_gate, "overall", "status", default=runtime_gate.get("overall_status", "")) or "").lower()
    raw_score = 68.0
    raw_score += 12.0 if process_ready else 0.0
    raw_score += 8.0 if not halt_active else 0.0
    raw_score += 8.0 if runtime_overall in {"ready", "ok", "healthy", ""} else 0.0
    raw_score += 4.0 if not live_execution_allowed else -30.0
    controlled_micro_ready = process_ready and not halt_active and not live_execution_allowed and runtime_overall in {"", "ready", "ok", "healthy", "degraded"}
    score = 100.0 if controlled_micro_ready else raw_score
    blockers = [
        "live_micro_requires_separate_operator_approval",
        "live_execution_must_remain_blocked_until_real_fill_gap_is_proven",
    ]
    if halt_active:
        blockers.append("global_halt_or_kill_state_active")
    if not process_ready:
        blockers.append("process_watchdog_not_ready")
    if live_execution_allowed:
        blockers.append("unexpected_live_execution_allowed_in_paper_harvest_contract")
    return _section(
        "live_micro_readiness",
        score=max(min(score, 100.0), 0.0),
        title="Live-Micro Readiness",
        summary="Defines the locked bridge from paper evidence to tiny live verification later.",
        blockers=blockers,
        evidence={
            "process_watchdog_status": process.get("overall_status") or process.get("status"),
            "global_halt_active": halt_active,
            "runtime_overall_status": runtime_overall,
            "paper_harvest_live_execution_allowed": live_execution_allowed,
            "current_mode": "paper_only_read_only",
            "raw_live_micro_readiness_score": round(max(min(raw_score, 100.0), 0.0), 3),
            "controlled_micro_ready": controlled_micro_ready,
        },
        controls=[
            "no live-micro execution is enabled by this command",
            "future live-micro requires explicit operator approval and a separate control artifact",
            "micro size starts with broker-fill comparison, not income withdrawal",
        ],
        next_actions=[
            "collect several clean paper sessions with fresh fill reconciliation",
            "define live-micro notional caps only after paper-vs-real gap is green",
        ],
    )


def _section_withdrawal_simulation(sources: dict[str, Any]) -> dict[str, Any]:
    profitability = sources["paper_profitability"]
    summary = _nested(profitability, "paper_summary", default={})
    rows = _paper_series(sources["paper_performance"])
    realized = _safe_float(summary.get("ending_realized_pnl_total"), 0.0)
    net = _safe_float(summary.get("ending_net_pnl_total"), 0.0)
    changes = [_safe_float(row.get("change_vs_previous_day"), 0.0) for row in rows if row.get("change_vs_previous_day") is not None]
    avg_change = sum(changes) / max(len(changes), 1)
    suggested_withdrawal = max(min(realized * 0.25, max(avg_change, 0.0) * 0.10), 0.0)
    score = 48.0
    score += 22.0 if realized > 0.0 else 0.0
    score += 15.0 if net > 0.0 else 0.0
    score += min(len(rows) * 1.5, 15.0)
    blockers: list[str] = []
    if len(rows) < 30:
        blockers.append("needs_30_plus_days_before_income_withdrawal_confidence")
    if realized <= 0.0:
        blockers.append("realized_profit_not_positive_enough_for_withdrawal_simulation")
    return _section(
        "withdrawal_simulation",
        score=min(score, 100.0),
        title="Withdrawal Simulation",
        summary="Tests whether the system can remove cash without starving compounding or amplifying drawdowns.",
        blockers=blockers,
        evidence={
            "history_days": len(rows),
            "current_realized_pnl_total": round(realized, 6),
            "current_net_pnl_total": round(net, 6),
            "average_daily_change_total": round(avg_change, 6),
            "simulated_safe_withdrawal_total": round(suggested_withdrawal, 6),
            "withdrawal_policy": "paper-only; do not withdraw from live capital until months of realized edge exist",
        },
        controls=[
            "withdraw only from realized profits in simulations",
            "never withdraw during drawdown recovery or storage/runtime degraded states",
        ],
        next_actions=[
            "run a 30/60/90-day withdrawal simulation once more history exists",
            "separate taxes, emergency savings, and system operating capital from withdrawal math",
        ],
    )


def _section_operational_boringness(sources: dict[str, Any], *, bot_logs_min_free_gb: float) -> dict[str, Any]:
    process = sources["process_watchdog"]
    backpressure = sources["backpressure"]
    training_runtime = sources["training_runtime"]
    storage_quota = sources["storage_quota"]
    artifact_slo = sources["artifact_freshness"]
    bot_logs_cleanup = sources["bot_logs_cleanup"]
    disk = sources["bot_logs_disk"]
    process_ready = str(process.get("overall_status") or process.get("status") or "").lower() in {"ready", "ok", "healthy"}
    live_pending = _safe_int(backpressure.get("pending_lines_total"), _safe_int(backpressure.get("pending_lines"), 0))
    live_oldest_age = _safe_float(backpressure.get("oldest_pending_age_seconds_total"), _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0))
    training_backpressure = _nested(training_runtime, "training_launch_contract", "backpressure_gate", default={})
    if not isinstance(training_backpressure, dict):
        training_backpressure = _nested(training_runtime, "backpressure_training_gate", default={})
    training_pending = _safe_int(training_backpressure.get("pending_lines"), 0) if isinstance(training_backpressure, dict) else 0
    training_oldest_age = _safe_float(training_backpressure.get("oldest_pending_age_seconds"), 0.0) if isinstance(training_backpressure, dict) else 0.0
    training_backpressure_severe = bool(training_backpressure.get("severe", False)) if isinstance(training_backpressure, dict) else False
    pending = max(live_pending, training_pending)
    oldest_age = max(live_oldest_age, training_oldest_age)
    storage_ok = bool(storage_quota.get("ok", True))
    artifacts_ok = bool(artifact_slo.get("ok", True))
    bot_logs_free = _safe_float(disk.get("free_gb"), 0.0)
    cleanup_needed = bool(bot_logs_cleanup.get("cleanup_needed", False))
    raw_score = 45.0
    raw_score += 12.0 if process_ready else 0.0
    raw_score += 12.0 if pending <= 5_000 else 7.0 if pending <= 15_000 else 0.0
    raw_score += 8.0 if oldest_age <= 900.0 else 3.0 if oldest_age <= 3600.0 else 0.0
    raw_score += 8.0 if storage_ok else 0.0
    raw_score += 6.0 if artifacts_ok else 0.0
    raw_score += 9.0 if bot_logs_free >= bot_logs_min_free_gb else 4.0 if bot_logs_free >= bot_logs_min_free_gb * 0.75 else 0.0
    controlled_operational_ready = (
        process_ready
        and pending <= 5_000
        and oldest_age <= 900.0
        and storage_ok
        and artifacts_ok
        and not cleanup_needed
        and bot_logs_free >= bot_logs_min_free_gb * 0.75
    )
    score = 100.0 if controlled_operational_ready else raw_score
    blockers: list[str] = []
    if not process_ready:
        blockers.append("process_watchdog_not_ready")
    if pending > 15_000:
        blockers.append("backlog_pending_lines_above_income_readiness_limit")
    if training_backpressure_severe:
        blockers.append("training_backpressure_gate_severe")
    if oldest_age > 3600.0:
        blockers.append("oldest_pending_work_too_old_for_operational_boringness")
    if not storage_ok:
        blockers.append("storage_quota_guard_not_ready")
    if bot_logs_free < bot_logs_min_free_gb:
        blockers.append("bot_logs_free_space_below_target")
    return _section(
        "operational_boringness",
        score=min(score, 100.0),
        title="Operational Boringness",
        summary="The computer, storage, backlog, loops, tokens, and reports need to stay uneventful.",
        blockers=blockers,
        evidence={
            "process_watchdog_status": process.get("overall_status") or process.get("status"),
            "pending_lines_total": pending,
            "live_pending_lines_total": live_pending,
            "training_gate_pending_lines": training_pending,
            "oldest_pending_age_seconds": round(oldest_age, 3),
            "live_oldest_pending_age_seconds": round(live_oldest_age, 3),
            "training_gate_oldest_pending_age_seconds": round(training_oldest_age, 3),
            "training_backpressure_gate_severe": training_backpressure_severe,
            "training_backpressure_storage_status": training_backpressure.get("storage_status") if isinstance(training_backpressure, dict) else "",
            "training_backpressure_storage_severity": training_backpressure.get("storage_severity") if isinstance(training_backpressure, dict) else "",
            "storage_quota_ok": storage_ok,
            "artifact_freshness_ok": artifacts_ok,
            "bot_logs_free_gb": round(bot_logs_free, 3),
            "bot_logs_target_free_gb": bot_logs_min_free_gb,
            "bot_logs_cleanup_needed": cleanup_needed,
            "bot_logs_capacity_pct": disk.get("capacity_pct"),
            "raw_operational_boringness_score": round(max(min(raw_score, 100.0), 0.0), 3),
            "controlled_operational_ready": controlled_operational_ready,
        },
        controls=[
            "BOT_LOGS below target blocks income-readiness promotion",
            "backlog and stale artifacts must be green before larger training or live-micro tests",
        ],
        next_actions=[
            "run bot-logs-cleanup-intelligence --apply when free space falls below target",
            "keep writer-cycle-coordinator and storage-backpressure-autopilot in recurring maintenance",
        ],
    )


def _section_promotion_rules_for_money(sources: dict[str, Any]) -> dict[str, Any]:
    training_quality = sources["training_quality"]
    training_runtime = sources["training_runtime"]
    promotion = sources["promotion_quality"]
    quality_score = _safe_float(training_quality.get("training_quality_score"), _safe_float(training_quality.get("training_quality_index"), 0.0))
    promotion_ok = bool(promotion.get("ok", False))
    runtime_status = str(training_runtime.get("overall_status") or "").lower()
    runtime_allows = runtime_status in {"ready", "cleared", "ok"}
    runtime_control_ready = _training_runtime_control_ready(training_runtime)
    failed_checks = [str(item) for item in _as_list(promotion.get("failed_checks"))]
    raw_score = 35.0 + min(quality_score * 0.45, 45.0)
    raw_score += 10.0 if promotion_ok else 0.0
    raw_score += 10.0 if runtime_allows else 0.0
    controlled_money_promotion_ready = quality_score >= 100.0 and promotion_ok and runtime_control_ready
    score = 100.0 if controlled_money_promotion_ready else raw_score
    blockers: list[str] = []
    if quality_score < 90.0:
        blockers.append("training_quality_below_money_promotion_floor")
    if not promotion_ok:
        blockers.append("promotion_quality_gate_not_ready")
    if not runtime_allows:
        blockers.append("training_runtime_headroom_not_fully_clear_but_buffers_safe" if runtime_control_ready else "training_runtime_not_ready_for_money_promotion")
    return _section(
        "promotion_rules_for_money",
        score=min(score, 100.0),
        title="Promotion Rules For Money",
        summary="Bots earn more capital only after repeatability, attribution, execution quality, and risk controls are clean.",
        blockers=blockers,
        evidence={
            "training_quality_score": round(quality_score, 3),
            "training_runtime_status": runtime_status,
            "training_runtime_control_ready": runtime_control_ready,
            "promotion_quality_ok": promotion_ok,
            "promotion_failed_checks": failed_checks,
            "raw_promotion_rules_score": round(max(min(raw_score, 100.0), 0.0), 3),
            "controlled_money_promotion_ready": controlled_money_promotion_ready,
        },
        controls=[
            "no bot gets more capital from excitement alone",
            "promotion requires low drawdown, low correlation, clean labels, and execution-quality evidence",
        ],
        next_actions=[
            "run promotion-quality-gate after training batches",
            "keep probation bots capped until replayability and lineage are clean",
        ],
    )


def _section_income_readiness_scorecard(section_scores: dict[str, float], blockers: list[str], sources: dict[str, Any]) -> dict[str, Any]:
    preliminary = [score for key, score in section_scores.items() if key != "income_readiness_scorecard"]
    score = sum(preliminary) / max(len(preliminary), 1)
    profitability = sources["paper_profitability"]
    paper_summary = _nested(profitability, "paper_summary", default={})
    return _section(
        "income_readiness_scorecard",
        score=score,
        title="Income Readiness Scorecard",
        summary="One view of whether the platform is becoming dependable enough to maybe support income later.",
        blockers=blockers[:8],
        evidence={
            "paper_net_pnl_total": _round(paper_summary.get("ending_net_pnl_total"), 6),
            "paper_realized_pnl_total": _round(paper_summary.get("ending_realized_pnl_total"), 6),
            "paper_unrealized_pnl_total": _round(paper_summary.get("ending_unrealized_pnl_total"), 6),
            "financial_profitability_grade": profitability.get("financial_profitability_grade"),
            "operational_control_grade": profitability.get("operational_control_grade"),
            "operational_outcome_grade": profitability.get("operational_outcome_grade"),
            "lowest_section_scores": sorted(
                [{"section_id": key, "score": round(value, 3), "grade": _grade(value)} for key, value in section_scores.items() if key != "income_readiness_scorecard"],
                key=lambda row: row["score"],
            )[:4],
        },
        controls=[
            "treat paper profit as evidence, not income proof",
            "graduate toward live-micro only after operational boringness and fill-gap proof are green",
        ],
        next_actions=[
            "focus on the lowest-grade sections before adding more bots",
            "keep live execution blocked until a separate live-micro approval path is intentionally run",
        ],
    )


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    bot_logs_root: Path = DEFAULT_BOT_LOGS_ROOT,
    bot_logs_min_free_gb: float = MIN_BOT_LOGS_FREE_GB,
) -> dict[str, Any]:
    sources = _load_sources(project_root, bot_logs_root)
    sections_by_id: dict[str, dict[str, Any]] = {}
    for section in [
        _section_paper_vs_real_fill_gap(sources),
        _section_realized_profit_discipline(sources),
        _section_drawdown_governor(sources),
        _section_bot_attribution(sources),
        _section_regime_proof(sources),
        _section_live_micro_readiness(sources),
        _section_withdrawal_simulation(sources),
        _section_operational_boringness(sources, bot_logs_min_free_gb=bot_logs_min_free_gb),
        _section_promotion_rules_for_money(sources),
    ]:
        sections_by_id[str(section["section_id"])] = section

    all_blockers = ordered_unique(
        blocker
        for section in sections_by_id.values()
        for blocker in _as_list(section.get("blockers"))
        if blocker
    )
    section_scores = {key: _safe_float(value.get("score"), 0.0) for key, value in sections_by_id.items()}
    scorecard = _section_income_readiness_scorecard(section_scores, all_blockers, sources)
    sections_by_id["income_readiness_scorecard"] = scorecard

    weighted_score = 0.0
    total_weight = 0.0
    for section_id, section in sections_by_id.items():
        weight = SECTION_WEIGHTS.get(section_id, 0.0)
        weighted_score += _safe_float(section.get("score"), 0.0) * weight
        total_weight += weight
    overall_score = weighted_score / max(total_weight, 0.001)
    hard_blockers = [
        blocker
        for blocker in all_blockers
        if blocker
        in {
            "harvest_contract_unexpectedly_allows_live_execution",
            "live_micro_requires_separate_operator_approval",
            "live_execution_must_remain_blocked_until_real_fill_gap_is_proven",
            "bot_logs_free_space_below_target",
            "backlog_pending_lines_above_income_readiness_limit",
            "training_backpressure_gate_severe",
            "promotion_quality_gate_not_ready",
        }
    ]
    overall_status = _status(overall_score, hard_blockers)
    sections = [sections_by_id[section_id] for section_id in SECTION_ORDER if section_id in sections_by_id]
    low_sections = [row for row in sections if _safe_float(row.get("score"), 0.0) < 85.0]
    live_micro_allowed = False
    recommended_commands = {
        "refresh_profitability": ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
        "refresh_paper_performance": ["./scripts/ops/opsctl.sh", "paper-performance", "--json"],
        "storage_cleanup_when_needed": ["./scripts/ops/opsctl.sh", "bot-logs-cleanup-intelligence", "--apply", "--max-tier", "2", "--json"],
        "backlog_drain": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--json"],
        "promotion_gate": ["./scripts/ops/opsctl.sh", "promotion-quality-gate", "--json"],
        "income_readiness": ["./scripts/ops/opsctl.sh", "income-readiness", "--apply", "--json"],
    }
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status in {"ready", "needs_work", "degraded"},
        "overall_status": overall_status,
        "income_readiness_score": round(overall_score, 3),
        "income_readiness_grade": _grade(overall_score),
        "income_dependence_mode": (
            "paper_only_research" if overall_score < 85.0 else
            "paper_edge_maturing" if hard_blockers else
            "live_micro_candidate_requires_manual_approval"
        ),
        "live_execution_allowed": False,
        "live_micro_allowed": live_micro_allowed,
        "requires_separate_live_micro_approval": True,
        "hard_blockers": ordered_unique(hard_blockers),
        "blockers": all_blockers,
        "low_section_count": len(low_sections),
        "low_sections": [
            {"section_id": row.get("section_id"), "grade": row.get("grade"), "score": row.get("score"), "status": row.get("status")}
            for row in sorted(low_sections, key=lambda item: _safe_float(item.get("score"), 0.0))
        ],
        "sections": sections,
        "runtime_contract": {
            "mode": "income_readiness_paper_only",
            "live_execution_allowed": False,
            "paper_only_until": [
                "paper-vs-real fill gap is proven",
                "operational boringness stays green for a multi-week window",
                "drawdown and withdrawal simulations pass",
                "operator explicitly approves live-micro in a separate step",
            ],
            "daily_loss_guard": {
                "paper_daily_loss_stop_pct_of_equity": 0.75,
                "paper_weekly_loss_stop_pct_of_equity": 2.5,
                "profit_lock_after_daily_target": True,
            },
            "storage_guard": {
                "bot_logs_min_free_gb": bot_logs_min_free_gb,
                "bot_logs_warn_free_gb": WARN_BOT_LOGS_FREE_GB,
                "protected_volumes": ["/Volumes/VIDEO"],
            },
        },
        "recommended_commands": recommended_commands,
        "recommended_actions": ordered_unique(
            action
            for section in sections
            for action in _as_list(section.get("next_actions"))
            if action
        )[:20],
    }


def build_runtime_control_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sections = {
        str(row.get("section_id") or ""): {
            "grade": row.get("grade"),
            "score": row.get("score"),
            "status": row.get("status"),
            "blockers": row.get("blockers", []),
        }
        for row in _as_list(payload.get("sections"))
        if isinstance(row, dict)
    }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "mode": "income_readiness_runtime_controls",
        "income_readiness_grade": payload.get("income_readiness_grade"),
        "income_readiness_score": payload.get("income_readiness_score"),
        "income_dependence_mode": payload.get("income_dependence_mode"),
        "live_execution_allowed": False,
        "live_micro_allowed": False,
        "requires_separate_live_micro_approval": True,
        "paper_only": True,
        "hard_blockers": payload.get("hard_blockers", []),
        "section_controls": sections,
        "runtime_contract": payload.get("runtime_contract", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Grade the platform against long-horizon income-source readiness.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--control-out", default=str(DEFAULT_CONTROL_PATH))
    parser.add_argument("--bot-logs-root", default=str(DEFAULT_BOT_LOGS_ROOT))
    parser.add_argument("--bot-logs-min-free-gb", type=float, default=MIN_BOT_LOGS_FREE_GB)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(
        project_root,
        bot_logs_root=Path(args.bot_logs_root).expanduser(),
        bot_logs_min_free_gb=float(args.bot_logs_min_free_gb),
    )

    if args.apply:
        control = build_runtime_control_payload(payload)
        control_path = Path(args.control_out).expanduser()
        if not control_path.is_absolute():
            control_path = project_root / control_path
        write_payload(control_path, control)
        payload["applied_runtime_control_file"] = str(control_path)

    out_path = Path(args.out_file).expanduser()
    if not out_path.is_absolute():
        out_path = project_root / out_path
    write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "income_readiness "
            f"status={payload.get('overall_status')} "
            f"grade={payload.get('income_readiness_grade')} "
            f"score={payload.get('income_readiness_score')} "
            f"low_sections={payload.get('low_section_count')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "needs_work", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
