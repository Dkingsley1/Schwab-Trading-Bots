#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "capital_growth_intelligence_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.capital_growth_intelligence_override"

RISK_PROFILES = {
    "conservative": {
        "paper_scale": 1.0,
        "live_micro_cap_pct": 0.0025,
        "per_sleeve_cap_pct": 0.035,
        "daily_loss_pct": 0.0025,
        "drawdown_pause_pct": 0.01,
        "profit_reinvest_pct": 0.25,
    },
    "balanced": {
        "paper_scale": 1.0,
        "live_micro_cap_pct": 0.005,
        "per_sleeve_cap_pct": 0.06,
        "daily_loss_pct": 0.005,
        "drawdown_pause_pct": 0.02,
        "profit_reinvest_pct": 0.35,
    },
    "aggressive": {
        "paper_scale": 1.0,
        "live_micro_cap_pct": 0.01,
        "per_sleeve_cap_pct": 0.09,
        "daily_loss_pct": 0.0075,
        "drawdown_pause_pct": 0.03,
        "profit_reinvest_pct": 0.45,
    },
}


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


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _round(raw: Any, digits: int = 4) -> float:
    return round(_safe_float(raw), digits)


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


def _grade_rank(raw: Any) -> int:
    ranks = {"F": 0, "D": 1, "C": 2, "C-": 2, "B": 3, "A": 4, "A+": 5, "A++": 5}
    return ranks.get(str(raw or "").strip().upper(), -1)


def _load_sources(project_root: Path) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    return {
        "income_readiness": load_json(health / "income_readiness_latest.json"),
        "income_operating_platform": load_json(health / "income_operating_platform_latest.json"),
        "paper_profitability": load_json(health / "paper_profitability_control_latest.json"),
        "paper_performance": load_json(health / "paper_performance_latest.json"),
        "sleeve_dashboard": load_json(health / "sleeve_profitability_dashboard_latest.json"),
        "market_posture": load_json(health / "market_posture_control_latest.json"),
        "account_policy": load_json(health / "account_policy_context_latest.json"),
        "training_runtime": load_json(health / "training_runtime_control_latest.json"),
        "promotion_quality": load_json(health / "promotion_quality_gate_latest.json"),
        "storage_quota": load_json(health / "storage_quota_guard_latest.json"),
        "ingestion_storage": load_json(health / "ingestion_storage_control_latest.json"),
        "runtime_gate": load_json(health / "runtime_gate_dashboard_latest.json"),
    }


def _nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def _paper_summary(sources: dict[str, Any]) -> dict[str, Any]:
    profitability = _as_dict(sources.get("paper_profitability"))
    summary = _as_dict(profitability.get("paper_summary"))
    if summary:
        return summary
    dashboard = _as_dict(sources.get("sleeve_dashboard"))
    totals = _as_dict(dashboard.get("totals"))
    return {
        "executions": totals.get("execution_count"),
        "ending_realized_pnl_total": totals.get("realized_pnl_total"),
        "ending_unrealized_pnl_total": totals.get("unrealized_pnl_total"),
        "ending_net_pnl_total": totals.get("net_pnl_total"),
    }


def _paper_sleeves(sources: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in _as_list(_as_dict(sources.get("paper_performance")).get("sleeve_latest")) if isinstance(row, dict)]
    if rows:
        return rows
    dashboard = _as_dict(sources.get("sleeve_dashboard"))
    merged = _as_list(dashboard.get("top_sleeves")) + _as_list(dashboard.get("bottom_sleeves"))
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in merged:
        if not isinstance(row, dict):
            continue
        key = str(row.get("profile") or row.get("cohort_key") or "")
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _control_profiles(sources: dict[str, Any]) -> tuple[set[str], set[str]]:
    profitability = _as_dict(sources.get("paper_profitability"))
    weak_profiles = {
        str(item).strip().lower()
        for item in _as_list(_nested(profitability, "a_plus_target_contract", "weak_profiles", default=[]))
        if str(item).strip()
    }
    active_controls = _as_dict(profitability.get("active_profile_controls"))
    quarantined = {
        str(profile).strip().lower()
        for profile, control in active_controls.items()
        if isinstance(control, dict)
        and (
            bool(control.get("block_new_entries"))
            or "quarantine" in str(control.get("action") or "").lower()
            or _safe_float(control.get("position_size_multiplier"), 1.0) <= 0.1
        )
    }
    return weak_profiles, quarantined


def _sleeve_name(row: dict[str, Any]) -> str:
    return str(row.get("profile") or row.get("cohort_key") or row.get("family") or "unknown").strip().lower() or "unknown"


def _sleeve_metrics(row: dict[str, Any]) -> dict[str, Any]:
    realized = _safe_float(row.get("realized_pnl_total"), _safe_float(row.get("ending_realized_pnl_total"), 0.0))
    unrealized = _safe_float(row.get("unrealized_pnl_total"), _safe_float(row.get("ending_unrealized_pnl_total"), 0.0))
    net = _safe_float(row.get("net_pnl_total"), _safe_float(row.get("ending_net_pnl_total"), realized + unrealized))
    executions = _safe_int(row.get("executions"), 0)
    win_rate = row.get("win_rate")
    return {
        "profile": _sleeve_name(row),
        "executions": executions,
        "realized_pnl": round(realized, 6),
        "unrealized_pnl": round(unrealized, 6),
        "net_pnl": round(net, 6),
        "win_rate": None if win_rate is None else round(_safe_float(win_rate), 6),
        "grade": str(row.get("display_grade") or row.get("control_grade") or row.get("grade") or ""),
    }


def _growth_score(row: dict[str, Any], *, weak_profiles: set[str], quarantined: set[str]) -> tuple[float, str]:
    profile = _sleeve_name(row)
    metrics = _sleeve_metrics(row)
    net = _safe_float(metrics["net_pnl"])
    realized = _safe_float(metrics["realized_pnl"])
    unrealized = _safe_float(metrics["unrealized_pnl"])
    executions = _safe_int(metrics["executions"])
    win_rate = metrics.get("win_rate")
    score = 40.0
    score += min(executions, 50) * 0.45
    score += _clamp(net / 25.0, -1.0, 1.0) * 18.0
    score += _clamp(realized / 10.0, -1.0, 1.0) * 12.0
    if unrealized < 0.0:
        score -= _clamp(abs(unrealized) / 25.0, 0.0, 1.0) * 16.0
    if win_rate is not None:
        score += (_clamp(_safe_float(win_rate), 0.0, 1.0) - 0.5) * 12.0
    score += max(_grade_rank(metrics.get("grade")), 0) * 1.5
    if profile in weak_profiles or profile in quarantined:
        score = min(score, 42.0)
        return max(score, 0.0), "quarantined_or_weak_profile"
    if net < 0.0:
        score = min(score, 48.0)
        return max(score, 0.0), "negative_net_paper_outcome"
    if realized < 0.0:
        score = min(score, 55.0)
        return max(score, 0.0), "realized_pnl_not_confirmed"
    if executions < 10:
        score = min(score, 62.0)
        return max(score, 0.0), "thin_execution_sample"
    if net > 0.0 and realized >= 0.0:
        return min(score + 8.0, 100.0), "scale_candidate_after_repeatability"
    return min(max(score, 0.0), 100.0), "collect_more_profit_evidence"


def _stage_plan(capital: float, profile: dict[str, float], live_allowed: bool) -> list[dict[str, Any]]:
    live_micro_budget = capital * profile["live_micro_cap_pct"] if live_allowed else 0.0
    daily_loss = capital * profile["daily_loss_pct"]
    drawdown_pause = capital * profile["drawdown_pause_pct"]
    return [
        {
            "stage": "stage_0_paper_truth",
            "mode": "paper_only",
            "capital_at_risk_usd": 0.0,
            "purpose": "prove that paper decisions, fills, exits, and attribution are repeatable before money scales",
            "advance_when": [
                "income readiness is A or better",
                "paper net PnL is positive across multiple refreshes",
                "realized profit share is at or above target",
                "storage, runtime, and promotion gates are clean",
            ],
        },
        {
            "stage": "stage_1_live_micro_rehearsal",
            "mode": "blocked_until_operator_approval" if not live_allowed else "live_micro_candidate",
            "capital_at_risk_usd": round(live_micro_budget, 2),
            "max_daily_loss_usd": round(daily_loss, 2),
            "purpose": "test tiny real-fill behavior without trusting paper PnL as income proof",
            "advance_when": [
                "real fill gap is proven small",
                "broker buying-power and account-rule checks are current",
                "no new-entry quarantine is active for the target sleeve",
            ],
        },
        {
            "stage": "stage_2_controlled_compounding",
            "mode": "future_plan",
            "capital_at_risk_usd": round(capital * min(profile["live_micro_cap_pct"] * 3.0, 0.03), 2) if live_allowed else 0.0,
            "max_drawdown_pause_usd": round(drawdown_pause, 2),
            "purpose": "increase only from realized, attributed, repeatable profits",
            "advance_when": [
                "30/60/90-day realized-profit evidence is positive",
                "sleeve-level capacity curves show no crowding",
                "weak sleeves remain capped automatically",
            ],
        },
        {
            "stage": "stage_3_scaled_income_candidate",
            "mode": "future_plan",
            "capital_at_risk_usd": round(capital * 0.0, 2),
            "purpose": "only after months of real fill and withdrawal simulation evidence",
            "advance_when": [
                "live micro history survives drawdown and choppy regimes",
                "income withdrawals are simulated from realized profit only",
                "tax, cash reserve, and operating capital buckets are separated",
            ],
        },
    ]


def _storage_hard_breaches(storage_quota: dict[str, Any]) -> int:
    summary = _as_dict(storage_quota.get("quota_summary"))
    return _safe_int(summary.get("hard_breaches"), 0)


def _storage_growth_ready(storage_quota: dict[str, Any]) -> bool:
    return _storage_hard_breaches(storage_quota) <= 0


def _training_growth_ready(training_runtime: dict[str, Any]) -> bool:
    status = str(training_runtime.get("overall_status") or "").lower()
    if status in {"ready", "cleared", "ok", "advisory", "constrained"}:
        return True
    quota_gate = _as_dict(training_runtime.get("storage_quota_training_gate"))
    launch_contract = _as_dict(training_runtime.get("training_launch_contract"))
    launch_blockers = {str(item or "").strip() for item in _as_list(launch_contract.get("launch_blockers")) if str(item or "").strip()}
    host_gate = _as_dict(launch_contract.get("host_training_headroom_gate"))
    backpressure_gate = _as_dict(launch_contract.get("backpressure_gate"))
    host_ready = bool(host_gate.get("safe_for_training", False)) or str(host_gate.get("status") or "").lower() == "ready"
    storage_ready = _safe_int(quota_gate.get("hard_breaches"), 0) <= 0
    bounded_backpressure = (
        "backpressure_overload_severe" in launch_blockers
        and _safe_int(backpressure_gate.get("pending_lines"), 0) <= 2_500
        and _safe_float(backpressure_gate.get("oldest_pending_age_seconds"), 0.0) <= 600.0
        and _safe_float(backpressure_gate.get("pressure_index"), 999.0) <= 2.0
    )
    tolerated_blockers = {"runtime_snapshot_not_fresh"}
    if bounded_backpressure:
        tolerated_blockers.add("backpressure_overload_severe")
    snapshot_or_bounded_tail_only = launch_blockers <= tolerated_blockers
    return storage_ready and (
        status in {"degraded", "guarded"}
        or (status == "blocked" and snapshot_or_bounded_tail_only and host_ready)
    )


def _capital_growth_control_blockers(
    sources: dict[str, Any],
    *,
    capital: float,
    sleeve_plan: list[dict[str, Any]],
) -> list[str]:
    storage_quota = _as_dict(sources.get("storage_quota"))
    training_runtime = _as_dict(sources.get("training_runtime"))
    blockers: list[str] = []
    if capital <= 0.0:
        blockers.append("capital_amount_missing")
    if not sleeve_plan:
        blockers.append("sleeve_growth_plan_missing")
    if not _storage_growth_ready(storage_quota):
        blocked = ",".join(str(item) for item in _as_list(_nested(storage_quota, "quota_summary", "blocked_families", default=[])))
        blockers.append(f"storage_quota_hard_breach:{blocked or 'unknown'}")
    if not _training_growth_ready(training_runtime):
        blockers.append("training_runtime_growth_headroom_not_ready")
    return ordered_unique(blockers)


def _capital_growth_control_score(
    *,
    capital: float,
    sleeve_plan: list[dict[str, Any]],
    control_blockers: list[str],
    live_blockers: list[str],
    weak_profiles: set[str],
    quarantined: set[str],
) -> float:
    score = 88.0
    score += 4.0 if capital > 0.0 else 0.0
    score += 3.0 if sleeve_plan else 0.0
    score += 3.0 if (weak_profiles or quarantined) else 2.0
    score += 2.0 if live_blockers else 0.0  # live remains guarded instead of silently widening.
    if control_blockers:
        score -= 18.0
        score -= min(len(control_blockers), 4) * 4.0
    return max(0.0, min(score, 100.0))


def _live_money_score(live_blockers: list[str], summary: dict[str, Any]) -> float:
    score = 100.0
    score -= min(len(live_blockers), 8) * 8.0
    if _safe_float(summary.get("ending_net_pnl_total"), summary.get("net_pnl_total") or 0.0) <= 0.0:
        score -= 12.0
    if _safe_float(summary.get("ending_realized_pnl_total"), summary.get("realized_pnl_total") or 0.0) <= 0.0:
        score -= 8.0
    return max(0.0, min(score, 100.0))


def _live_scaling_blockers(sources: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    income = _as_dict(sources.get("income_readiness"))
    platform = _as_dict(sources.get("income_operating_platform"))
    profitability = _as_dict(sources.get("paper_profitability"))
    storage_quota = _as_dict(sources.get("storage_quota"))
    training_runtime = _as_dict(sources.get("training_runtime"))
    promotion = _as_dict(sources.get("promotion_quality"))
    blockers: list[str] = []
    if _safe_float(income.get("income_readiness_score"), income.get("overall_score") or 0.0) < 85.0:
        blockers.append("income_readiness_below_money_growth_floor")
    if str(platform.get("overall_status") or "").lower() not in {"ready", "advisory"}:
        blockers.append("income_operating_platform_not_ready")
    if _safe_float(summary.get("ending_net_pnl_total"), summary.get("net_pnl_total") or 0.0) <= 0.0:
        blockers.append("paper_net_pnl_not_positive")
    realized = _safe_float(summary.get("ending_realized_pnl_total"), summary.get("realized_pnl_total") or 0.0)
    unrealized = _safe_float(summary.get("ending_unrealized_pnl_total"), summary.get("unrealized_pnl_total") or 0.0)
    if realized <= 0.0 or realized < max(abs(unrealized), 1.0) * 0.25:
        blockers.append("realized_profit_conversion_too_low")
    if bool(_nested(profitability, "paper_harvest_execution_contract", "live_execution_allowed", default=False)):
        blockers.append("unexpected_live_execution_flag_review_required")
    else:
        blockers.append("live_execution_still_requires_separate_operator_approval")
    if not bool(storage_quota.get("ok", False)):
        blockers.append("storage_quota_guard_not_ready")
    if str(training_runtime.get("overall_status") or "").lower() not in {"ready", "cleared", "ok"}:
        blockers.append("training_runtime_not_ready_for_growth_promotion")
    if not bool(promotion.get("ok", False)):
        blockers.append("promotion_quality_gate_not_ready")
    return ordered_unique(blockers)


def _build_sleeve_growth_plan(
    sleeves: list[dict[str, Any]],
    *,
    capital: float,
    risk_profile: dict[str, float],
    weak_profiles: set[str],
    quarantined: set[str],
    live_allowed: bool,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in sleeves:
        score, reason = _growth_score(row, weak_profiles=weak_profiles, quarantined=quarantined)
        metrics = _sleeve_metrics(row)
        scored.append({**metrics, "growth_score": round(score, 3), "growth_grade": _grade(score), "budget_reason": reason})
    positive = [row for row in scored if _safe_float(row["growth_score"]) >= 60.0]
    total_score = sum(_safe_float(row["growth_score"]) for row in positive) or 1.0
    max_sleeve_cap = risk_profile["per_sleeve_cap_pct"]
    result: list[dict[str, Any]] = []
    for row in sorted(scored, key=lambda item: _safe_float(item["growth_score"]), reverse=True):
        score = _safe_float(row["growth_score"])
        if score < 45.0:
            paper_pct = 0.005
        elif score < 60.0:
            paper_pct = 0.015
        else:
            paper_pct = min(max_sleeve_cap, (score / total_score) * min(0.35, max_sleeve_cap * max(len(positive), 1)))
        live_pct = paper_pct * 0.25 if live_allowed and score >= 75.0 else 0.0
        result.append(
            {
                **row,
                "paper_sim_budget_pct": round(paper_pct, 5),
                "paper_sim_budget_usd": round(capital * paper_pct, 2),
                "live_micro_budget_pct": round(live_pct, 5),
                "live_micro_budget_usd": round(capital * live_pct, 2),
                "capital_action": "candidate_for_growth" if score >= 75.0 else "observe_or_repair" if score >= 50.0 else "cap_or_quarantine",
            }
        )
    return result


def _write_env(path: Path, payload: dict[str, Any]) -> None:
    contract = _as_dict(payload.get("runtime_contract"))
    lines = [
        "# Generated by capital_growth_intelligence.py",
        f"CAPITAL_GROWTH_INTELLIGENCE_READY={1 if payload.get('ok') else 0}",
        f"CAPITAL_GROWTH_RISK_PROFILE={payload.get('risk_profile')}",
        f"CAPITAL_GROWTH_TOTAL_CAPITAL_USD={contract.get('total_capital_usd', 0)}",
        f"CAPITAL_GROWTH_PAPER_SIMULATION_ENABLED={1 if contract.get('paper_simulation_enabled') else 0}",
        f"CAPITAL_GROWTH_LIVE_SCALING_ALLOWED={1 if contract.get('live_money_scaling_allowed') else 0}",
        f"CAPITAL_GROWTH_MAX_LIVE_MICRO_BUDGET_USD={contract.get('max_live_micro_budget_usd', 0)}",
        f"CAPITAL_GROWTH_MAX_DAILY_LOSS_USD={contract.get('max_daily_loss_usd', 0)}",
        f"CAPITAL_GROWTH_PROFIT_REINVEST_PCT={contract.get('profit_reinvest_pct', 0)}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    capital: float = 10_000.0,
    risk_profile_name: str = "balanced",
    monthly_contribution: float = 0.0,
    apply: bool = False,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
) -> dict[str, Any]:
    sources = _load_sources(project_root)
    risk_profile_name = risk_profile_name.lower().strip()
    risk_profile = RISK_PROFILES.get(risk_profile_name, RISK_PROFILES["balanced"])
    capital = max(_safe_float(capital, 0.0), 0.0)
    monthly_contribution = max(_safe_float(monthly_contribution, 0.0), 0.0)
    summary = _paper_summary(sources)
    sleeves = _paper_sleeves(sources)
    weak_profiles, quarantined = _control_profiles(sources)
    live_blockers = _live_scaling_blockers(sources, summary)
    live_allowed = not live_blockers
    sleeve_plan = _build_sleeve_growth_plan(
        sleeves,
        capital=capital,
        risk_profile=risk_profile,
        weak_profiles=weak_profiles,
        quarantined=quarantined,
        live_allowed=live_allowed,
    )
    candidate_sleeves = [row for row in sleeve_plan if row["capital_action"] == "candidate_for_growth"]
    growth_control_blockers = _capital_growth_control_blockers(sources, capital=capital, sleeve_plan=sleeve_plan)
    score = _capital_growth_control_score(
        capital=capital,
        sleeve_plan=sleeve_plan,
        control_blockers=growth_control_blockers,
        live_blockers=live_blockers,
        weak_profiles=weak_profiles,
        quarantined=quarantined,
    )
    live_score = _live_money_score(live_blockers, summary)
    max_live_micro = capital * risk_profile["live_micro_cap_pct"] if live_allowed else 0.0
    max_daily_loss = capital * risk_profile["daily_loss_pct"]
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": not growth_control_blockers,
        "overall_status": "capital_growth_controls_ready" if not growth_control_blockers else "capital_growth_controls_blocked",
        "overall_score": round(score, 3),
        "overall_grade": _grade(score),
        "grade_basis": "capital_growth_control_readiness; live-money approval and profit evidence are graded separately",
        "risk_profile": risk_profile_name,
        "capital_context": {
            "total_capital_usd": round(capital, 2),
            "monthly_contribution_usd": round(monthly_contribution, 2),
            "works_with_any_capital_amount": True,
            "money_scaling_principle": "capital grows only after realized, attributed, repeatable profit evidence; paper evidence is not treated as income proof",
        },
        "readiness": {
            "paper_simulation_allowed": True,
            "capital_growth_controls_ready": not growth_control_blockers,
            "capital_growth_control_blockers": growth_control_blockers,
            "live_money_scaling_allowed": live_allowed,
            "live_money_scaling_blockers": live_blockers,
            "candidate_growth_sleeve_count": len(candidate_sleeves),
            "quarantined_or_weak_profiles": sorted(weak_profiles | quarantined),
        },
        "capital_growth_control": {
            "score": round(score, 3),
            "grade": _grade(score),
            "blockers": growth_control_blockers,
            "allows_paper_money_tree_simulation": not growth_control_blockers,
            "live_blockers_are_expected_controls": bool(live_blockers),
            "iron_clad_principle": "the system may plan and simulate growth at any account size, but capital can only scale after realized, attributed, repeatable edge clears the live-money gate",
        },
        "live_money_scaling": {
            "score": round(live_score, 3),
            "grade": _grade(live_score),
            "allowed": live_allowed,
            "blockers": live_blockers,
            "operator_approval_required": True,
        },
        "paper_profit_snapshot": {
            "executions": _safe_int(summary.get("executions"), 0),
            "realized_pnl_total": _round(summary.get("ending_realized_pnl_total"), 6),
            "unrealized_pnl_total": _round(summary.get("ending_unrealized_pnl_total"), 6),
            "net_pnl_total": _round(summary.get("ending_net_pnl_total"), 6),
        },
        "stage_plan": _stage_plan(capital, risk_profile, live_allowed),
        "sleeve_growth_plan": sleeve_plan,
        "money_tree_growth_policy": {
            "capital_scaling_unit": "sleeve",
            "works_from_small_to_large_accounts": True,
            "increase_budget_when": [
                "sleeve has positive realized contribution",
                "unrealized drag is below stop-rule tolerance",
                "paper fills and attribution are fresh",
                "storage, runtime, and training gates are not hard-blocked",
            ],
            "decrease_budget_when": [
                "sleeve is quarantined or weak",
                "realized profit conversion falls below target",
                "drawdown or daily-loss stop rules trip",
                "confirmation-bias, overlap, or one-sided precision checks fail",
            ],
        },
        "profit_compounding_policy": {
            "paper_only_until_live_micro_approved": True,
            "realized_profit_reinvest_pct": risk_profile["profit_reinvest_pct"],
            "realized_profit_buckets": [
                "first refill drawdown reserve",
                "then fund more paper sampling for winning sleeves",
                "then allow tiny sleeve cap increases only for realized contributors",
                "never increase a weak sleeve because portfolio-level PnL looks good",
            ],
            "monthly_contribution_policy": "new contributions enter simulation capital first; live deployment remains gated by readiness",
        },
        "stop_rules": [
            f"pause growth if one-day paper/live loss exceeds ${max_daily_loss:.2f}",
            f"pause growth if drawdown reaches ${capital * risk_profile['drawdown_pause_pct']:.2f}",
            "pause growth if realized-profit share falls below target",
            "pause growth if storage quota, runtime, token, or fill-gap gates degrade",
            "pause growth if a sleeve becomes overlap-heavy, one-sided, or confirmation-biased",
        ],
        "runtime_contract": {
            "paper_simulation_enabled": True,
            "live_money_scaling_allowed": live_allowed,
            "total_capital_usd": round(capital, 2),
            "max_live_micro_budget_usd": round(max_live_micro, 2),
            "max_daily_loss_usd": round(max_daily_loss, 2),
            "profit_reinvest_pct": risk_profile["profit_reinvest_pct"],
            "requires_operator_approval_for_live_money": True,
            "live_execution_unchanged": True,
        },
        "recommended_commands": {
            "refresh_profitability": ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
            "refresh_sleeves": ["./scripts/ops/opsctl.sh", "sleeve-pnl", "--json"],
            "refresh_income": ["./scripts/ops/opsctl.sh", "income-readiness", "--apply", "--json"],
            "refresh_training_snapshot_light": [
                "./scripts/ops/opsctl.sh",
                "runtime-training-snapshot",
                "--lookback-days",
                "1",
                "--light-refresh-existing",
                "--json",
            ],
            "rerun_growth_plan": [
                "./scripts/ops/opsctl.sh",
                "capital-growth-intelligence",
                "--capital",
                str(round(capital, 2)),
                "--risk-profile",
                risk_profile_name,
                "--json",
            ],
        },
        "source_artifacts": [
            str(project_root / "governance" / "health" / "paper_profitability_control_latest.json"),
            str(project_root / "governance" / "health" / "paper_performance_latest.json"),
            str(project_root / "governance" / "health" / "income_readiness_latest.json"),
            str(project_root / "governance" / "health" / "income_operating_platform_latest.json"),
        ],
    }
    if apply:
        _write_env(override_path, payload)
        payload["write_result"] = {"override_path": str(override_path), "applied": True}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a capital growth plan from paper profit, risk, and operating readiness evidence.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--monthly-contribution", type=float, default=0.0)
    parser.add_argument("--risk-profile", choices=sorted(RISK_PROFILES), default="balanced")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--override-path", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(
        project_root,
        capital=args.capital,
        risk_profile_name=args.risk_profile,
        monthly_contribution=args.monthly_contribution,
        apply=bool(args.apply),
        override_path=Path(args.override_path).expanduser(),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        readiness = _as_dict(payload.get("readiness"))
        print(
            "capital_growth_intelligence "
            f"status={payload.get('overall_status')} "
            f"grade={payload.get('overall_grade')} "
            f"capital={_safe_float(_nested(payload, 'capital_context', 'total_capital_usd'), 0.0):.2f} "
            f"live_allowed={int(bool(readiness.get('live_money_scaling_allowed')))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
