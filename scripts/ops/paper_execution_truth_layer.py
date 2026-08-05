#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "paper_execution_truth_layer_latest.json"
DEFAULT_PLATFORM_OS_OUT_PATH = PROJECT_ROOT / "governance" / "platform_os" / "paper_execution_truth_layer_latest.json"
BROKER_RECONCILIATION_CORE_SOURCE_IDS = {
    "market_quote_profiles",
    "options_context_mesh",
    "crypto_market_context",
    "free_equity_reference_context",
    "fx_market_context",
    "official_macro_context",
    "schwab_education_context",
    "market_micro_context",
    "public_policy_context",
    "fed_2026_supervisory_stress_scenario",
}
MIN_BLOCKING_COUNTERFACTUAL_KEPT_COUNT = 50
NON_GRADE_BLOCKING_REPLAY_REASONS = {
    "counterfactual_candidates_pending_collecting",
    "counterfactual_outcome_attribution_pending",
    "paper_replay_rows_low_collecting",
    "counterfactual_low_sample_win_rate_below_floor",
    "counterfactual_low_sample_aggregate_nonpositive",
    "counterfactual_low_sample_outcome_attribution_pending",
}
NON_GRADE_BLOCKING_OPTIONS_REASONS = {
    "covered_call_watch_critical",
    "covered_call_alerts_present",
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


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(float(low), min(float(value), float(high)))


def _gate(status: str, score: float, reasons: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    reasons = reasons or []
    return {
        "ok": status == "ready",
        "status": status,
        "score": round(_clamp(float(score)), 6),
        "reasons": ordered_unique(reasons),
        **extra,
    }


def _grade_from_score(score: float, status: str) -> str:
    if status == "blocked":
        return "F" if score < 60.0 else "C"
    if score >= 97.0:
        return "A+"
    if score >= 93.0:
        return "A"
    if score >= 90.0:
        return "A-"
    if score >= 87.0:
        return "B+"
    if score >= 83.0:
        return "B"
    if score >= 80.0:
        return "B-"
    if score >= 70.0:
        return "C"
    return "D"


def _grade_blocking_warning(name: str, gate: dict[str, Any]) -> bool:
    if str(gate.get("status") or "") != "warn":
        return False
    if "grade_blocking" in gate:
        return bool(gate.get("grade_blocking", True))
    return True


def _advisory_warning(name: str, gate: dict[str, Any]) -> bool:
    return bool(str(gate.get("status") or "") == "warn" and not _grade_blocking_warning(name, gate))


def _artifact_ok(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    if "ok" in payload:
        return bool(payload.get("ok", False))
    status = str(payload.get("overall_status") or payload.get("status") or "").strip().lower()
    return status in {"ready", "ok", "watch", "active"}


def _promotion_quality_self_referential(payload: dict[str, Any]) -> bool:
    failed = {
        str(item or "").strip()
        for item in (payload.get("failed_checks") if isinstance(payload.get("failed_checks"), list) else [])
        if str(item or "").strip()
    }
    if not failed or not failed.issubset({"daily_verify_not_ok", "paper_execution_truth_layer_not_ok"}):
        return False
    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    truth_failed = {
        str(item or "").strip()
        for item in (
            details.get("paper_execution_truth_layer_failed_checks")
            if isinstance(details.get("paper_execution_truth_layer_failed_checks"), list)
            else []
        )
        if str(item or "").strip()
    }
    unresolved_daily = {
        str(item or "").strip()
        for item in (
            details.get("daily_verify_unresolved_failed_checks")
            if isinstance(details.get("daily_verify_unresolved_failed_checks"), list)
            else []
        )
        if str(item or "").strip()
    }
    truth_loop_only = not truth_failed or truth_failed.issubset({"promotion_gate_hardening"})
    daily_loop_only = not unresolved_daily or unresolved_daily.issubset(
        {"paper_execution_calibration_report", "promotion_quality_gate", "paper_execution_truth_layer"}
    )
    return bool(truth_loop_only and daily_loop_only)


def _promotion_quality_blocking_failed_checks(payload: dict[str, Any]) -> list[str]:
    failed = {
        str(item or "").strip()
        for item in (payload.get("failed_checks") if isinstance(payload.get("failed_checks"), list) else [])
        if str(item or "").strip()
    }
    promotion_only = {
        "promotion_gate_blocked",
        "insufficient_considered_bots",
        "daily_verify_not_ok",
        "new_bot_graduation_not_ok",
        "new_bot_admission_not_ok",
        "feature_store_manifest_not_ready",
        "retrain_schema_compatibility_not_ok",
        "promotion_packet_not_ready",
        "paper_execution_truth_layer_not_ok",
    }
    return sorted(failed - promotion_only)


def _profile_calibration(calibration: dict[str, Any], profile: str) -> dict[str, Any]:
    by_profile = calibration.get("by_profile") if isinstance(calibration.get("by_profile"), dict) else {}
    return by_profile.get(str(profile or "").strip().lower(), {}) if isinstance(by_profile.get(str(profile or "").strip().lower()), dict) else {}


def _build_calibration_gate(calibration: dict[str, Any], *, max_mae_bps: float, max_p95_bps: float) -> dict[str, Any]:
    metrics = calibration.get("metrics") if isinstance(calibration.get("metrics"), dict) else {}
    calibration_window = calibration.get("calibration_window") if isinstance(calibration.get("calibration_window"), dict) else {}
    reset_active = bool(calibration_window.get("reset_active", False))
    samples = _safe_int(calibration.get("independent_samples"), _safe_int(calibration.get("samples"), 0))
    model_derived_samples = _safe_int(calibration.get("model_derived_samples"), 0)
    minimum_independent_samples = max(_safe_int(calibration.get("minimum_independent_samples"), 1), 1)
    independent_evidence_ready = bool(
        calibration.get("independent_evidence_ready", samples >= minimum_independent_samples)
    )
    mae = _safe_float(metrics.get("mae_bps"), 0.0)
    p95 = _safe_float(metrics.get("p95_bps"), 0.0)
    reasons: list[str] = []
    if not calibration:
        reasons.append("calibration_artifact_missing")
    elif not independent_evidence_ready:
        if samples <= 0 and model_derived_samples > 0:
            reasons.append("model_derived_fills_are_not_independent_calibration_evidence")
        if reset_active and samples <= 0:
            reasons.append("calibration_window_reset_waiting_for_samples")
        else:
            reasons.append("independent_calibration_evidence_pending")
    if samples > 0 and mae > max_mae_bps:
        reasons.append("calibration_mae_above_limit")
    if samples > 0 and p95 > max_p95_bps:
        reasons.append("calibration_p95_above_limit")
    metric_failure = bool({"calibration_mae_above_limit", "calibration_p95_above_limit"}.intersection(reasons))
    if "calibration_artifact_missing" in reasons or metric_failure:
        status = "blocked"
    elif not reasons:
        status = "ready"
    else:
        status = "warn"
    score = 100.0 - (mae / max(max_mae_bps, 1.0)) * 35.0 - (p95 / max(max_p95_bps, 1.0)) * 20.0
    evidence_pending = bool(status == "warn" and not independent_evidence_ready)
    if evidence_pending:
        score = 82.0
    return _gate(
        status,
        score,
        reasons,
        samples=samples,
        independent_samples=samples,
        model_derived_samples=model_derived_samples,
        minimum_independent_samples=minimum_independent_samples,
        independent_evidence_ready=independent_evidence_ready,
        promotion_evidence_eligible=bool(status == "ready" and independent_evidence_ready),
        calibration_window=calibration_window,
        metrics={
            "mae_bps": round(mae, 6),
            "p95_bps": round(p95, 6),
            "mean_bias_bps": round(_safe_float(metrics.get("mean_bias_bps"), 0.0), 6),
        },
        recommendations=calibration.get("recommendations", {}),
        grade_blocking=not evidence_pending,
        advisory_only=evidence_pending,
        advisory_policy=(
            "independent fill evidence debt remains visible while paper collection continues, but model-derived fills "
            "cannot satisfy the live-money promotion contract"
        ),
    )


def _build_post_cost_expectancy_gate(paper_performance: dict[str, Any]) -> dict[str, Any]:
    expectancy = (
        paper_performance.get("post_cost_expectancy")
        if isinstance(paper_performance.get("post_cost_expectancy"), dict)
        else {}
    )
    sample_count = _safe_int(expectancy.get("sample_count"), 0)
    minimum_samples = max(_safe_int(expectancy.get("minimum_samples"), 30), 1)
    evidence_sufficient = bool(expectancy.get("evidence_sufficient", sample_count >= minimum_samples))
    positive_lcb = bool(expectancy.get("positive_lower_confidence_bound_95", False))
    mean_pnl = _safe_float(expectancy.get("mean_post_cost_pnl_delta"), 0.0)
    mean_return_bps = _safe_float(expectancy.get("mean_post_cost_return_bps"), 0.0)
    pnl_lcb = _safe_float(expectancy.get("lower_confidence_bound_95_post_cost_pnl_delta"), 0.0)
    return_lcb = _safe_float(expectancy.get("lower_confidence_bound_95_post_cost_return_bps"), 0.0)
    reasons: list[str] = []
    if not expectancy or not bool(expectancy.get("available", False)):
        reasons.append("post_cost_expectancy_evidence_missing")
    elif not evidence_sufficient:
        reasons.append("post_cost_expectancy_samples_pending")
    elif not positive_lcb:
        if mean_pnl <= 0.0 or mean_return_bps <= 0.0:
            reasons.append("post_cost_expectancy_nonpositive")
        else:
            reasons.append("post_cost_expectancy_confidence_pending")
    status = "ready" if not reasons else "warn"
    if status == "ready":
        score = 100.0
    elif evidence_sufficient and (mean_pnl <= 0.0 or mean_return_bps <= 0.0):
        score = 60.0
    else:
        score = 84.0
    return _gate(
        status,
        score,
        reasons,
        sample_count=sample_count,
        minimum_samples=minimum_samples,
        evidence_sufficient=evidence_sufficient,
        promotion_evidence_eligible=bool(evidence_sufficient and positive_lcb),
        mean_post_cost_pnl_delta=round(mean_pnl, 6),
        mean_post_cost_return_bps=round(mean_return_bps, 6),
        lower_confidence_bound_95_post_cost_pnl_delta=round(pnl_lcb, 6),
        lower_confidence_bound_95_post_cost_return_bps=round(return_lcb, 6),
        grade_blocking=False,
        advisory_only=status == "warn",
        advisory_policy=(
            "paper collection continues while post-cost expectancy matures; live-money promotion requires enough "
            "samples and positive 95% lower confidence bounds for both PnL delta and return"
        ),
    )


def _sleeve_rows(paper_performance: dict[str, Any], calibration: dict[str, Any], *, max_slippage_gap_bps: float) -> list[dict[str, Any]]:
    rows = paper_performance.get("sleeve_latest") if isinstance(paper_performance.get("sleeve_latest"), list) else []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile") or "default").strip().lower() or "default"
        data_status = str(row.get("data_status") or "").strip().lower()
        current_day_available_raw = row.get("current_day_available")
        stale_latest_for_current_truth = bool(
            data_status in {"latest_available", "no_data"}
            or (current_day_available_raw is False and data_status not in {"current", "current_live_no_fills"})
        )
        tca = row.get("tca_summary") if isinstance(row.get("tca_summary"), dict) else {}
        execs = _safe_int(row.get("executions"), 0)
        pnl = _safe_float(row.get("ending_net_pnl_total"), 0.0)
        change = _safe_float(row.get("change_vs_previous_day"), 0.0)
        slippage_gap = _safe_float(tca.get("mean_slippage_gap_bps"), 0.0)
        expected_slip = _safe_float(tca.get("mean_expected_slippage_bps"), 0.0)
        realized_slip = _safe_float(tca.get("mean_realized_slippage_bps"), 0.0)
        partial = _safe_float(tca.get("mean_partial_fill_ratio"), 1.0)
        poor_count = _safe_int(tca.get("poor_or_fair_fill_count"), 0)
        win_rate = _safe_float(row.get("win_rate"), 0.0)
        calibration_row = _profile_calibration(calibration, profile)
        calibration_mae = _safe_float(calibration_row.get("mae_bps"), 0.0)
        pnl_per_execution = pnl / max(execs, 1) if execs > 0 else 0.0
        reasons: list[str] = []
        if abs(slippage_gap) > max_slippage_gap_bps:
            reasons.append("slippage_gap_high")
        if partial < 0.90:
            reasons.append("partial_fill_drag")
        if poor_count > 0:
            reasons.append("poor_or_fair_fills")
        if pnl < 0.0:
            reasons.append("negative_net_pnl")
        if execs > 0 and pnl_per_execution <= 0.0:
            reasons.append("nonpositive_pnl_per_execution")
        score = (
            100.0
            - min(abs(slippage_gap) / max(max_slippage_gap_bps, 1.0), 3.0) * 18.0
            - min(max(calibration_mae, 0.0) / 75.0, 2.0) * 12.0
            - (1.0 - min(max(partial, 0.0), 1.0)) * 25.0
            - min(poor_count, 25) * 1.2
            - (18.0 if pnl < 0.0 else 0.0)
            + min(max(win_rate, 0.0), 1.0) * 8.0
        )
        status = "ready" if _clamp(score) >= 70.0 and not {"negative_net_pnl", "partial_fill_drag"}.intersection(reasons) else "watch"
        if _clamp(score) < 45.0:
            status = "blocked"
        if stale_latest_for_current_truth:
            status = "watch"
            reasons.append("stale_latest_available_for_current_truth")
        if execs <= 0 and status == "blocked" and data_status in {"current_live_no_fills", "latest_available", "no_data"}:
            status = "watch"
            reasons.append("no_current_fills_for_blocking_execution_truth")
        out.append(
            {
                "profile": profile,
                "status": status,
                "data_status": data_status,
                "current_day_available": current_day_available_raw,
                "day_utc": str(row.get("day_utc") or ""),
                "execution_realism_score": round(_clamp(score), 6),
                "executions": execs,
                "ending_net_pnl_total": round(pnl, 6),
                "change_vs_previous_day": round(change, 6),
                "pnl_per_execution": round(pnl_per_execution, 8),
                "win_rate": row.get("win_rate"),
                "tca": {
                    "mean_expected_slippage_bps": round(expected_slip, 6),
                    "mean_realized_slippage_bps": round(realized_slip, 6),
                    "mean_slippage_gap_bps": round(slippage_gap, 6),
                    "mean_partial_fill_ratio": round(partial, 6),
                    "poor_or_fair_fill_count": poor_count,
                },
                "calibration": {
                    "samples": _safe_int(calibration_row.get("samples"), 0),
                    "mae_bps": round(calibration_mae, 6),
                    "recommended_slippage_scale": _safe_float(calibration_row.get("recommended_slippage_scale"), 1.0),
                },
                "reasons": ordered_unique(reasons),
            }
        )
    out.sort(key=lambda item: (item["execution_realism_score"], -abs(_safe_float(item.get("ending_net_pnl_total"), 0.0)), item["profile"]))
    return out


def _build_sleeve_gate(scorecards: list[dict[str, Any]], *, min_sleeve_score: float) -> dict[str, Any]:
    if not scorecards:
        return _gate("blocked", 0.0, ["no_sleeve_scorecards"], scorecards=[])
    advisory_no_fill = {
        row["profile"]
        for row in scorecards
        if "no_current_fills_for_blocking_execution_truth"
        in {str(reason or "") for reason in (row.get("reasons") if isinstance(row.get("reasons"), list) else [])}
    }
    advisory_stale_latest = {
        row["profile"]
        for row in scorecards
        if "stale_latest_available_for_current_truth"
        in {str(reason or "") for reason in (row.get("reasons") if isinstance(row.get("reasons"), list) else [])}
    }
    advisory_profiles = advisory_no_fill | advisory_stale_latest
    failing = [
        row["profile"]
        for row in scorecards
        if _safe_float(row.get("execution_realism_score"), 0.0) < min_sleeve_score
        and row["profile"] not in advisory_profiles
    ]
    blocked = [row["profile"] for row in scorecards if str(row.get("status") or "") == "blocked" and row["profile"] not in advisory_profiles]
    mean_score = sum(_safe_float(row.get("execution_realism_score"), 0.0) for row in scorecards) / max(len(scorecards), 1)
    reasons: list[str] = []
    if failing:
        reasons.append("sleeve_score_below_floor")
    if blocked:
        reasons.append("sleeve_blocked")
    status = "ready" if not reasons else ("warn" if not blocked else "blocked")
    return _gate(
        status,
        mean_score,
        reasons,
        min_sleeve_score=float(min_sleeve_score),
        failing_profiles=failing,
        advisory_no_fill_profiles=sorted(advisory_no_fill),
        advisory_stale_latest_profiles=sorted(advisory_stale_latest),
        blocked_profiles=blocked,
        scorecards=scorecards,
    )


def _build_replay_gate(counterfactual: dict[str, Any], paper_replay: dict[str, Any], *, min_win_rate: float) -> dict[str, Any]:
    top = counterfactual.get("top_candidates") if isinstance(counterfactual.get("top_candidates"), list) else []
    best = top[0] if top and isinstance(top[0], dict) else {}
    win_rate_raw = best.get("win_rate")
    has_win_rate = win_rate_raw is not None and str(win_rate_raw).strip() != ""
    win_rate = _safe_float(win_rate_raw, 0.0)
    aggregate = _safe_float(best.get("aggregate_net_pnl_total"), 0.0)
    kept_count = _safe_int(best.get("kept_count"), 0)
    low_sample = bool(top and 0 < kept_count < MIN_BLOCKING_COUNTERFACTUAL_KEPT_COUNT)
    reasons: list[str] = []
    warnings: list[str] = []
    advisory_reasons: list[str] = []
    paper_replay_failed = {
        str(item or "").strip()
        for item in (paper_replay.get("failed_checks") if isinstance(paper_replay.get("failed_checks"), list) else [])
        if str(item or "").strip()
    }
    paper_replay_collecting_only = bool(
        paper_replay
        and not _artifact_ok(paper_replay)
        and paper_replay_failed
        and paper_replay_failed.issubset({"paper_rows_low"})
    )
    if not _artifact_ok(counterfactual):
        reasons.append("counterfactual_replay_not_ok")
    if not top:
        if _artifact_ok(counterfactual) and (_artifact_ok(paper_replay) or paper_replay_collecting_only):
            warnings.append("counterfactual_candidates_pending_collecting")
        else:
            reasons.append("no_counterfactual_candidates")
    if has_win_rate and win_rate < min_win_rate:
        if aggregate < 0.0 and not low_sample:
            reasons.append("counterfactual_win_rate_below_floor")
        elif aggregate < 0.0:
            warnings.append("counterfactual_low_sample_win_rate_below_floor")
        else:
            advisory_reasons.append("counterfactual_win_rate_below_floor_attributed_nonnegative")
    if top and aggregate < 0.0 and not low_sample:
        reasons.append("counterfactual_aggregate_nonpositive")
    elif top and aggregate < 0.0:
        warnings.append("counterfactual_low_sample_aggregate_nonpositive")
    elif top and aggregate == 0.0 and not has_win_rate and low_sample:
        warnings.append("counterfactual_low_sample_outcome_attribution_pending")
    elif top and aggregate == 0.0 and not has_win_rate:
        warnings.append("counterfactual_outcome_attribution_pending")
    if paper_replay and not _artifact_ok(paper_replay):
        if "stale_execution_skips_only" in paper_replay_failed:
            reasons.append("paper_replay_stale_skips_only")
        elif paper_replay_collecting_only and (low_sample or not top):
            warnings.append("paper_replay_rows_low_collecting")
        else:
            reasons.append("paper_replay_drill_not_ok")
    status = "ready" if not reasons and not warnings else ("blocked" if reasons else "warn")
    grade_blocking = not bool(
        status == "warn"
        and warnings
        and set(warnings).issubset(NON_GRADE_BLOCKING_REPLAY_REASONS)
    )
    score = 55.0 + min(max(win_rate, 0.0), 1.0) * 35.0 + (10.0 if aggregate > 0.0 else 0.0)
    if status == "warn":
        score = max(score, 82.0)
    return _gate(
        status,
        score,
        reasons + warnings,
        best_candidate=best,
        min_blocking_kept_count=MIN_BLOCKING_COUNTERFACTUAL_KEPT_COUNT,
        low_sample=low_sample,
        grade_blocking=grade_blocking,
        advisory_only=not grade_blocking,
        advisory_reasons=advisory_reasons,
        paper_replay_ok=(_artifact_ok(paper_replay) if paper_replay else None),
    )


def _build_options_account_gate(account_study: dict[str, Any], covered_call_watch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    position_count = _safe_int(account_study.get("position_count"), 0)
    account_count = _safe_int(account_study.get("account_count"), 0)
    underlying_count = _safe_int(account_study.get("underlying_count"), 0)
    cc = account_study.get("covered_call_roll_watch") if isinstance(account_study.get("covered_call_roll_watch"), dict) else {}
    watch = covered_call_watch if covered_call_watch else cc
    covered_call_count = _safe_int(watch.get("covered_call_count"), _safe_int(cc.get("covered_call_count"), 0))
    alert_count = _safe_int(watch.get("alert_count"), _safe_int(cc.get("alert_count"), 0))
    status_text = str(watch.get("overall_status") or cc.get("overall_status") or "").strip().lower()

    account_reasons: list[str] = []
    if not _artifact_ok(account_study):
        account_reasons.append("account_study_not_ok")
    if account_count <= 0:
        account_reasons.append("no_accounts_visible")
    if position_count <= 0:
        account_reasons.append("no_positions_visible")
    account_gate = _gate(
        "ready" if not account_reasons else "blocked",
        100.0 - len(account_reasons) * 35.0,
        account_reasons,
        account_count=account_count,
        position_count=position_count,
        underlying_count=underlying_count,
        notes=[
            "Uses redacted local account labels only.",
            "This gate studies positions and roll windows; it does not place orders.",
        ],
    )

    option_reasons: list[str] = []
    if covered_call_count <= 0:
        option_reasons.append("no_covered_calls_detected")
    if status_text in {"critical", "blocked"}:
        option_reasons.append("covered_call_watch_critical")
    if alert_count > 0:
        option_reasons.append("covered_call_alerts_present")
    option_status = "ready" if not option_reasons or option_reasons == ["no_covered_calls_detected"] else "warn"
    operator_advisory_only = bool(
        option_status == "warn"
        and set(option_reasons).issubset(NON_GRADE_BLOCKING_OPTIONS_REASONS)
    )
    option_score = (
        100.0
        if operator_advisory_only
        else 100.0 - alert_count * 10.0 - (25.0 if "covered_call_watch_critical" in option_reasons else 0.0)
    )
    return account_gate, _gate(
        option_status,
        option_score,
        option_reasons,
        covered_call_count=covered_call_count,
        alert_count=alert_count,
        overall_status=status_text,
        grade_blocking=not operator_advisory_only,
        operator_advisory_only=operator_advisory_only,
        advisory_policy=(
            "covered-call roll watch stays visible for operator review but does not downgrade paper execution realism "
            "while it remains advisory-only and does not authorize automatic orders"
        ),
    )


def _build_stress_gate(execution_lab: dict[str, Any], *, max_worst_slippage_bps: float) -> dict[str, Any]:
    worst = execution_lab.get("top_worst_case_scenarios") if isinstance(execution_lab.get("top_worst_case_scenarios"), list) else []
    worst_slip = max((_safe_float(row.get("slippage_bps"), 0.0) for row in worst if isinstance(row, dict)), default=0.0)
    caps = execution_lab.get("capabilities") if isinstance(execution_lab.get("capabilities"), dict) else {}
    required_caps = {
        "fee_spread_slippage_haircut",
        "partial_fill_modeling",
        "queue_priority_modeling",
        "market_impact_modeling",
        "reject_cancel_stale_quote_modeling",
        "realistic_option_fills",
        "execution_quality_scoring",
        "sleeve_specific_friction",
        "live_shadow_calibration_inputs",
    }
    missing = [cap for cap in sorted(required_caps) if not bool(caps.get(cap, False))]
    reasons: list[str] = []
    if not _artifact_ok(execution_lab):
        reasons.append("execution_lab_not_ok")
    if missing:
        reasons.append("stress_capabilities_missing")
    if worst_slip > max_worst_slippage_bps:
        reasons.append("stress_slippage_above_limit")
    status = "ready" if not reasons else ("warn" if reasons == ["stress_slippage_above_limit"] else "blocked")
    score = 100.0 - (worst_slip / max(max_worst_slippage_bps, 1.0)) * 30.0 - len(missing) * 7.0
    return _gate(
        status,
        score,
        reasons,
        max_worst_slippage_bps=max_worst_slippage_bps,
        worst_slippage_bps=round(worst_slip, 6),
        missing_capabilities=missing,
        top_worst_case_scenarios=worst[:6],
    )


def _build_live_transition_gate(live_readiness: dict[str, Any], execution_lab: dict[str, Any]) -> dict[str, Any]:
    caps = execution_lab.get("capabilities") if isinstance(execution_lab.get("capabilities"), dict) else {}
    required_caps = {
        "fee_spread_slippage_haircut",
        "partial_fill_modeling",
        "queue_priority_modeling",
        "market_impact_modeling",
        "reject_cancel_stale_quote_modeling",
        "realistic_option_fills",
        "execution_quality_scoring",
        "live_shadow_calibration_inputs",
    }
    missing_caps = [cap for cap in sorted(required_caps) if not bool(caps.get(cap, False))]
    hard_blocks = live_readiness.get("hard_blocks") if isinstance(live_readiness.get("hard_blocks"), list) else []
    warnings = live_readiness.get("warnings") if isinstance(live_readiness.get("warnings"), list) else []
    submit_enabled = bool(live_readiness.get("submit_path_enabled", False))
    readiness_score = _safe_float(live_readiness.get("readiness_score"), 0.0 if not live_readiness else 100.0)
    reasons: list[str] = []
    advisories: list[str] = []
    if not live_readiness:
        advisories.append("live_readiness_smoke_missing")
    if missing_caps:
        reasons.append("live_execution_realism_capabilities_missing")
    if live_readiness and not _artifact_ok(live_readiness) and submit_enabled:
        reasons.append("live_readiness_not_ok")
    elif live_readiness and not _artifact_ok(live_readiness):
        advisories.append("validate_only_live_readiness_not_ok")
    if hard_blocks and submit_enabled:
        reasons.append("live_submit_hard_blocks_present")
    elif hard_blocks:
        advisories.append("validate_only_live_hard_blocks_present")
    if warnings and submit_enabled:
        reasons.append("live_submit_warnings_present")
    elif warnings:
        advisories.append("validate_only_live_warnings_present")

    if missing_caps or "live_submit_hard_blocks_present" in reasons:
        status = "blocked"
    elif reasons:
        status = "warn"
    else:
        status = "ready"

    if submit_enabled:
        score = readiness_score - len(missing_caps) * 12.0 - len(hard_blocks) * 4.0 - len(warnings) * 1.5
    else:
        score = 96.0 - len(missing_caps) * 12.0
        if not live_readiness:
            score -= 8.0
    return _gate(
        status,
        score,
        reasons,
        mode=str(live_readiness.get("mode") or "missing"),
        readiness_score=round(readiness_score, 6),
        submit_path_enabled=submit_enabled,
        hard_blocks=hard_blocks,
        live_readiness_warnings=warnings,
        advisory_reasons=ordered_unique(advisories),
        missing_capabilities=missing_caps,
        parity_requirements=[
            "live guard uses the execution simulator before broker submit",
            "stale quote, reject, cancel, low score, and weak effective-fill paths block live submit",
            "paper/live share calibration, sleeve friction, account awareness, and promotion gating",
        ],
    )


def _effective_pnl_per_execution(score_row: dict[str, Any], weekly_executions: int) -> tuple[float, str]:
    current_execs = _safe_int(score_row.get("executions"), 0)
    current_pnl_per_exec = _safe_float(score_row.get("pnl_per_execution"), 0.0)
    if current_execs > 0:
        return current_pnl_per_exec, "current_scorecard_pnl_per_execution"

    net_pnl = _safe_float(score_row.get("ending_net_pnl_total"), 0.0)
    if weekly_executions > 0 and net_pnl != 0.0:
        return net_pnl / max(float(weekly_executions), 1.0), "latest_net_pnl_over_weekly_executions"

    latest_change = _safe_float(score_row.get("change_vs_previous_day"), 0.0)
    if weekly_executions > 0 and latest_change != 0.0:
        return latest_change / max(float(weekly_executions), 1.0), "latest_change_over_weekly_executions"

    return current_pnl_per_exec, "current_scorecard_default"


def _build_throttle_gate(paper_performance: dict[str, Any], scorecards: list[dict[str, Any]], *, max_weekly_execs: int, min_pnl_per_exec: float) -> dict[str, Any]:
    week = paper_performance.get("week") if isinstance(paper_performance.get("week"), dict) else {}
    week_profiles = week.get("top_profiles") if isinstance(week.get("top_profiles"), list) else []
    score_by_profile = {str(row.get("profile") or ""): row for row in scorecards}
    actions: list[dict[str, Any]] = []
    for row in week_profiles:
        if not isinstance(row, dict):
            continue
        profile = str(row.get("name") or row.get("profile") or "").strip().lower()
        executions = _safe_int(row.get("executions"), 0)
        score_row = score_by_profile.get(profile, {})
        pnl_per_exec, pnl_basis = _effective_pnl_per_execution(score_row, executions)
        score = _safe_float(score_row.get("execution_realism_score"), 75.0)
        net_pnl = _safe_float(score_row.get("ending_net_pnl_total"), 0.0)
        weak_quality = bool((pnl_per_exec < min_pnl_per_exec and net_pnl <= 0.0) or score < 65.0)
        if executions > max_weekly_execs and weak_quality:
            actions.append(
                {
                    "profile": profile,
                    "action": "throttle_new_entries",
                    "reason": "high_execution_low_quality",
                    "weekly_executions": executions,
                    "pnl_per_execution": round(pnl_per_exec, 8),
                    "pnl_per_execution_basis": pnl_basis,
                    "ending_net_pnl_total": round(net_pnl, 6),
                    "execution_realism_score": round(score, 6),
                }
            )
    status = "ready" if not actions else "warn"
    return _gate(
        status,
        100.0 - min(len(actions) * 12.0, 60.0),
        ["overtrading_throttle_actions_present"] if actions else [],
        max_weekly_executions_per_profile=max_weekly_execs,
        min_pnl_per_execution=min_pnl_per_exec,
        throttle_actions=actions,
    )


def _build_haircut_ledger(paper_performance: dict[str, Any], scorecards: list[dict[str, Any]]) -> dict[str, Any]:
    week = paper_performance.get("week") if isinstance(paper_performance.get("week"), dict) else {}
    raw_week_pnl = _safe_float(week.get("rolling_change"), _safe_float(week.get("week_to_date_change"), 0.0))
    rows: list[dict[str, Any]] = []
    total_drag = 0.0
    for row in scorecards:
        executions = _safe_int(row.get("executions"), 0)
        tca = row.get("tca") if isinstance(row.get("tca"), dict) else {}
        expected_slip = max(_safe_float(tca.get("mean_expected_slippage_bps"), 0.0), 0.0)
        gap = abs(_safe_float(tca.get("mean_slippage_gap_bps"), 0.0))
        partial = min(max(_safe_float(tca.get("mean_partial_fill_ratio"), 1.0), 0.0), 1.0)
        drag = executions * (expected_slip + gap) * 0.001 + executions * (1.0 - partial) * 0.01
        total_drag += drag
        rows.append(
            {
                "profile": row.get("profile"),
                "executions": executions,
                "raw_net_pnl_total": row.get("ending_net_pnl_total"),
                "estimated_realism_drag": round(drag, 6),
                "realism_adjusted_net_pnl_total": round(_safe_float(row.get("ending_net_pnl_total"), 0.0) - drag, 6),
            }
        )
    adjusted = raw_week_pnl - total_drag
    return {
        "raw_week_pnl": round(raw_week_pnl, 6),
        "estimated_realism_drag": round(total_drag, 6),
        "realism_adjusted_week_pnl": round(adjusted, 6),
        "drag_model": "executions * (expected_slippage_bps + abs(slippage_gap_bps)) * 0.001 + partial_fill_drag",
        "by_sleeve": rows,
    }


def _build_ingestion_gate(ingestion_storage: dict[str, Any], ingestion_backpressure: dict[str, Any]) -> dict[str, Any]:
    storage_bp = ingestion_storage.get("backpressure") if isinstance(ingestion_storage.get("backpressure"), dict) else {}
    total_pending = _safe_int(storage_bp.get("total_pending_lines"), _safe_int(ingestion_backpressure.get("pending_lines_total"), _safe_int(ingestion_backpressure.get("pending_lines"), 0)))
    core_pending = _safe_int(storage_bp.get("core_pending_lines"), _safe_int(ingestion_backpressure.get("pending_lines"), 0))
    threshold = _safe_int(storage_bp.get("pending_lines_threshold"), _safe_int(ingestion_backpressure.get("pending_lines_threshold"), 15000))
    oldest_age = _safe_float(storage_bp.get("oldest_pending_age_seconds"), _safe_float(ingestion_backpressure.get("oldest_pending_age_seconds_total"), 0.0))
    storage_ready_clear = bool(
        str(ingestion_storage.get("overall_status") or "").strip().lower() == "ready"
        and str(ingestion_storage.get("severity") or "").strip().lower() in {"", "stable"}
        and total_pending <= threshold
    )
    reasons: list[str] = []
    if ingestion_storage and not _artifact_ok(ingestion_storage):
        reasons.append("ingestion_storage_control_not_ok")
    if ingestion_backpressure and bool(ingestion_backpressure.get("overload", False)) and not storage_ready_clear:
        reasons.append("ingestion_backpressure_overload")
    if threshold > 0 and total_pending > threshold:
        reasons.append("ingestion_pending_above_threshold")
    status = "ready" if not reasons else "warn"
    return _gate(
        status,
        100.0 - min(total_pending / max(threshold, 1), 3.0) * 20.0,
        reasons,
        total_pending_lines=total_pending,
        core_pending_lines=core_pending,
        pending_lines_threshold=threshold,
        oldest_pending_age_seconds=round(oldest_age, 3),
        storage_ready_clear=storage_ready_clear,
    )


def _build_paper_broker_reconciliation_gate(
    paper_performance: dict[str, Any],
    broker_truth: dict[str, Any],
    source_verification: dict[str, Any],
) -> dict[str, Any]:
    week = paper_performance.get("week") if isinstance(paper_performance.get("week"), dict) else {}
    sleeve_latest = paper_performance.get("sleeve_latest") if isinstance(paper_performance.get("sleeve_latest"), list) else []
    weekly_executions = sum(_safe_int(row.get("executions"), 0) for row in week.get("top_profiles", []) if isinstance(row, dict))
    latest_executions = sum(_safe_int(row.get("executions"), 0) for row in sleeve_latest if isinstance(row, dict))
    decision_activity = max(weekly_executions, latest_executions)
    broker_v2 = broker_truth.get("broker_truth_reconcile_v2") if isinstance(broker_truth.get("broker_truth_reconcile_v2"), dict) else {}
    broker_ok = bool(broker_truth.get("ok", False) and broker_truth.get("broker_truth_ok", True))
    broker_score = _safe_float(broker_v2.get("truth_score"), _safe_float(broker_truth.get("broker_truth_v2_score"), 0.0))
    broker_grade = str(broker_v2.get("truth_grade") or broker_truth.get("broker_truth_v2_grade") or "")
    sources_ok = bool(source_verification.get("ok", False))
    source_rows = source_verification.get("sources") if isinstance(source_verification.get("sources"), list) else []
    source_rows_by_id = {
        str(row.get("source_id") or "").strip(): row
        for row in source_rows
        if isinstance(row, dict) and str(row.get("source_id") or "").strip()
    }
    core_source_status: dict[str, bool] = {}
    for source_id in sorted(BROKER_RECONCILIATION_CORE_SOURCE_IDS):
        source_row = source_rows_by_id.get(source_id, {})
        verification_status = str(source_row.get("verification_status") or "").strip()
        source_confidence = _safe_float(source_row.get("source_confidence_score"), 0.0)
        core_source_status[source_id] = bool(
            source_row
            and bool(source_row.get("ok", False))
            and verification_status != "single_source_unverified"
            and source_confidence >= 0.70
        )
    missing_or_unverified_core_sources = [
        source_id for source_id, verified in core_source_status.items() if not verified
    ]
    source_confidence_scores = [
        _safe_float(row.get("source_confidence_score"), 0.0)
        for row in source_rows
        if isinstance(row, dict)
    ]
    mean_source_confidence = (
        sum(source_confidence_scores) / max(len(source_confidence_scores), 1)
        if source_confidence_scores
        else 0.0
    )
    mismatch_count = _safe_int(broker_truth.get("broker_truth_mismatch_count"), _safe_int(broker_truth.get("mismatch_count"), 0))
    account_count = _safe_int(broker_truth.get("account_count"), 0)
    position_rows = _safe_int(broker_truth.get("position_rows"), _safe_int(broker_truth.get("broker_truth_position_count"), 0))
    broker_truth_clean_for_source_advisory = bool(
        broker_ok
        and broker_score >= 0.90
        and mismatch_count == 0
        and account_count > 0
        and (position_rows > 0 or decision_activity <= 0)
    )
    reasons: list[str] = []
    advisory_reasons: list[str] = []
    if decision_activity > 0 and not broker_ok:
        reasons.append("paper_activity_without_clean_broker_truth")
    if broker_ok and broker_score and broker_score < 0.78:
        reasons.append("broker_truth_v2_score_below_floor")
    if mismatch_count > 0:
        reasons.append("paper_or_manual_position_delta_present")
    if account_count <= 0 and decision_activity > 0:
        reasons.append("no_broker_accounts_visible_for_paper_reconcile")
    core_sources_ready = bool(core_source_status) and not missing_or_unverified_core_sources
    if not sources_ok and not core_sources_ready:
        if broker_truth_clean_for_source_advisory:
            advisory_reasons.append("source_verification_context_debt_not_blocking_clean_broker_truth")
        else:
            reasons.append("source_verification_not_ready")
    elif not sources_ok:
        advisory_reasons.append("optional_source_verification_lanes_not_ready")
    if mean_source_confidence and mean_source_confidence < 0.70:
        if broker_truth_clean_for_source_advisory:
            advisory_reasons.append("source_confidence_thin_context_advisory")
        else:
            reasons.append("source_confidence_thin")
    status = "ready" if not reasons else ("warn" if decision_activity <= 0 or reasons == ["source_confidence_thin"] else "blocked")
    score = 100.0
    score -= 35.0 if not broker_ok and decision_activity > 0 else 0.0
    score -= max(0.0, 0.78 - broker_score) * 50.0 if broker_score else 8.0
    score -= min(mismatch_count, 10) * 5.0
    score -= 15.0 if not sources_ok and not core_sources_ready else 0.0
    score -= 3.0 if advisory_reasons else 0.0
    score += min(mean_source_confidence, 1.0) * 5.0
    return _gate(
        status,
        score,
        reasons,
        weekly_executions=weekly_executions,
        latest_executions=latest_executions,
        broker_truth_ok=broker_ok,
        broker_truth_v2_score=round(broker_score, 6),
        broker_truth_v2_grade=broker_grade,
        account_count=account_count,
        position_rows=position_rows,
        mismatch_count=mismatch_count,
        source_verification_ok=sources_ok,
        mean_source_confidence=round(mean_source_confidence, 6),
        broker_truth_clean_for_source_advisory=broker_truth_clean_for_source_advisory,
        broker_reconciliation_core_sources_ready=core_sources_ready,
        broker_reconciliation_core_source_status=core_source_status,
        missing_or_unverified_core_sources=missing_or_unverified_core_sources,
        advisory_reasons=ordered_unique(advisory_reasons),
        reconciliation_policy="paper activity must remain explainable by current broker truth and verified market context",
    )


def evaluate_truth_layer(
    *,
    paper_performance: dict[str, Any],
    calibration: dict[str, Any],
    counterfactual: dict[str, Any],
    paper_replay: dict[str, Any],
    account_study: dict[str, Any],
    covered_call_watch: dict[str, Any],
    execution_lab: dict[str, Any],
    ingestion_storage: dict[str, Any],
    ingestion_backpressure: dict[str, Any],
    promotion_quality: dict[str, Any],
    live_readiness: dict[str, Any] | None = None,
    broker_truth: dict[str, Any] | None = None,
    source_verification: dict[str, Any] | None = None,
    max_calibration_mae_bps: float = 35.0,
    max_calibration_p95_bps: float = 175.0,
    min_sleeve_score: float = 65.0,
    max_slippage_gap_bps: float = 25.0,
    min_replay_win_rate: float = 0.52,
    max_stress_slippage_bps: float = 90.0,
    max_weekly_executions_per_profile: int = 12000,
    min_pnl_per_execution: float = 0.001,
) -> dict[str, Any]:
    calibration_gate = _build_calibration_gate(
        calibration,
        max_mae_bps=max_calibration_mae_bps,
        max_p95_bps=max_calibration_p95_bps,
    )
    post_cost_expectancy_gate = _build_post_cost_expectancy_gate(paper_performance)
    scorecards = _sleeve_rows(paper_performance, calibration, max_slippage_gap_bps=max_slippage_gap_bps)
    sleeve_gate = _build_sleeve_gate(scorecards, min_sleeve_score=min_sleeve_score)
    replay_gate = _build_replay_gate(counterfactual, paper_replay, min_win_rate=min_replay_win_rate)
    account_gate, options_gate = _build_options_account_gate(account_study, covered_call_watch)
    stress_gate = _build_stress_gate(execution_lab, max_worst_slippage_bps=max_stress_slippage_bps)
    live_transition_gate = _build_live_transition_gate(live_readiness or {}, execution_lab)
    throttle_gate = _build_throttle_gate(
        paper_performance,
        scorecards,
        max_weekly_execs=max_weekly_executions_per_profile,
        min_pnl_per_exec=min_pnl_per_execution,
    )
    haircut_ledger = _build_haircut_ledger(paper_performance, scorecards)
    ingestion_gate = _build_ingestion_gate(ingestion_storage, ingestion_backpressure)
    paper_broker_reconciliation_gate = _build_paper_broker_reconciliation_gate(
        paper_performance,
        broker_truth or {},
        source_verification or {},
    )

    promotion_reasons: list[str] = []
    promotion_warnings: list[str] = []
    gate_map = {
        "live_quote_fill_calibration": calibration_gate,
        "post_cost_expectancy_evidence": post_cost_expectancy_gate,
        "sleeve_execution_scorecards": sleeve_gate,
        "decision_replay_harness": replay_gate,
        "options_specific_realism": options_gate,
        "account_position_awareness": account_gate,
        "market_regime_stress_mode": stress_gate,
        "live_execution_transition_parity": live_transition_gate,
        "auto_throttle_overtrading": throttle_gate,
        "data_ingestion_quality_gate": ingestion_gate,
        "paper_broker_truth_reconciliation": paper_broker_reconciliation_gate,
    }
    for name, gate in gate_map.items():
        if str(gate.get("status") or "") == "blocked":
            promotion_reasons.append(f"{name}_blocked")
        elif gate.get("promotion_evidence_eligible") is False:
            promotion_reasons.append(f"{name}_promotion_evidence_not_ready")
    promotion_quality_effective_ok = _artifact_ok(promotion_quality) or _promotion_quality_self_referential(promotion_quality)
    promotion_quality_blockers = _promotion_quality_blocking_failed_checks(promotion_quality) if promotion_quality else []
    if promotion_quality and not promotion_quality_effective_ok:
        if promotion_quality_blockers:
            promotion_reasons.append("promotion_quality_gate_not_ok")
        elif not any(str(gate.get("status") or "") == "blocked" for gate in gate_map.values()):
            promotion_warnings.append("promotion_quality_gate_advisory_only")
        else:
            promotion_warnings.append("promotion_quality_gate_watch")
    promotion_gate = _gate(
        (
            "ready"
            if not promotion_reasons
            and (not promotion_warnings or promotion_warnings == ["promotion_quality_gate_advisory_only"])
            else ("blocked" if promotion_reasons else "warn")
        ),
        min((_safe_float(gate.get("score"), 0.0) for gate in gate_map.values()), default=0.0),
        promotion_reasons
        + [warning for warning in promotion_warnings if warning != "promotion_quality_gate_advisory_only"],
        promotion_quality_gate_ok=(promotion_quality_effective_ok if promotion_quality else None),
        promotion_quality_advisory_only=bool(promotion_warnings == ["promotion_quality_gate_advisory_only"]),
        promotion_quality_failed_checks=(
            promotion_quality.get("failed_checks", []) if isinstance(promotion_quality.get("failed_checks"), list) else []
        ),
        promotion_quality_blocking_failed_checks=promotion_quality_blockers,
        hardened_requirements=[
            "calibration",
            "confidence_bounded_post_cost_expectancy",
            "sleeve_scorecards",
            "counterfactual_replay",
            "options_realism",
            "account_awareness",
            "stress_scenarios",
            "live_execution_transition_parity",
            "overtrading_throttle",
            "haircut_ledger",
            "ingestion_quality",
            "paper_broker_truth_reconciliation",
            "promotion_quality",
        ],
        promotion_only=True,
    )

    all_gates = {**gate_map, "paper_pnl_haircut_ledger": {"ok": True, "status": "ready", **haircut_ledger}, "promotion_gate_hardening": promotion_gate}
    operational_gates = {
        name: gate for name, gate in all_gates.items() if not bool(gate.get("promotion_only", False))
    }
    blocked = [name for name, gate in operational_gates.items() if str(gate.get("status") or "") == "blocked"]
    grade_blocking_warnings = [
        name for name, gate in operational_gates.items() if _grade_blocking_warning(name, gate)
    ]
    advisory_warnings = [
        name for name, gate in operational_gates.items() if _advisory_warning(name, gate)
    ]
    warnings = grade_blocking_warnings
    ready_count = len(operational_gates) - len(blocked) - len(grade_blocking_warnings)
    raw_metric_score = sum(_safe_float(gate.get("score"), 85.0) for gate in operational_gates.values()) / max(len(operational_gates), 1)
    status = "ready" if not blocked else "blocked"
    if grade_blocking_warnings and not blocked:
        status = "watch"
    overall_score = _clamp(raw_metric_score)
    if status == "ready":
        overall_score = max(overall_score, 97.0)
    elif status == "watch":
        overall_score = min(overall_score, 89.9)
    grade = _grade_from_score(overall_score, status)
    a_plus_ready = bool(status == "ready" and grade == "A+")
    promotion_ready = bool(not blocked and str(promotion_gate.get("status") or "") == "ready")
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 2,
        "ok": not blocked,
        "overall_status": status,
        "score": round(_clamp(overall_score), 6),
        "raw_metric_score": round(_clamp(raw_metric_score), 6),
        "grade": grade,
        "a_plus_ready": a_plus_ready,
        "promotion_ready": promotion_ready,
        "live_money_promotion_ready": promotion_ready,
        "promotion_status": str(promotion_gate.get("status") or "blocked"),
        "promotion_failed_checks": list(promotion_gate.get("reasons") or []),
        "ready_gates": ready_count,
        "warning_gates": len(grade_blocking_warnings),
        "advisory_warning_gates": len(advisory_warnings),
        "blocked_gates": len(blocked),
        "failed_checks": blocked,
        "warnings": warnings,
        "grade_blocking_warnings": grade_blocking_warnings,
        "advisory_warnings": advisory_warnings,
        "operator_advisories": [
            {
                "gate": name,
                "reasons": gate.get("reasons", []),
                "policy": str(gate.get("advisory_policy") or "visible_advisory_not_grade_blocking"),
            }
            for name, gate in all_gates.items()
            if _advisory_warning(name, gate)
        ],
        "gates": all_gates,
        "sleeve_scorecards": scorecards,
        "paper_pnl_haircut_ledger": haircut_ledger,
        "recommended_actions": ordered_unique(
            [
                action["action"] + ":" + action["profile"]
                for action in throttle_gate.get("throttle_actions", [])
                if isinstance(action, dict)
            ]
            + calibration_gate.get("reasons", [])
            + post_cost_expectancy_gate.get("reasons", [])
            + sleeve_gate.get("reasons", [])
            + replay_gate.get("reasons", [])
            + options_gate.get("reasons", [])
            + live_transition_gate.get("reasons", [])
            + ingestion_gate.get("reasons", [])
            + paper_broker_reconciliation_gate.get("reasons", [])
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a consolidated paper execution truth layer for real-world paper realism.")
    parser.add_argument("--paper-performance-file", default=str(PROJECT_ROOT / "governance" / "health" / "paper_performance_latest.json"))
    parser.add_argument("--calibration-file", default=str(PROJECT_ROOT / "governance" / "health" / "paper_execution_calibration_latest.json"))
    parser.add_argument("--counterfactual-file", default=str(PROJECT_ROOT / "governance" / "health" / "counterfactual_replay_latest.json"))
    parser.add_argument("--paper-replay-file", default=str(PROJECT_ROOT / "governance" / "health" / "paper_replay_drill_latest.json"))
    parser.add_argument("--account-study-file", default=str(PROJECT_ROOT / "governance" / "health" / "account_position_study_latest.json"))
    parser.add_argument("--covered-call-watch-file", default=str(PROJECT_ROOT / "governance" / "health" / "covered_call_roll_watch_latest.json"))
    parser.add_argument("--execution-lab-file", default=str(PROJECT_ROOT / "governance" / "health" / "execution_lab_latest.json"))
    parser.add_argument("--live-readiness-file", default=str(PROJECT_ROOT / "governance" / "health" / "live_readiness_smoke_latest.json"))
    parser.add_argument("--ingestion-storage-file", default=str(PROJECT_ROOT / "governance" / "health" / "ingestion_storage_control_latest.json"))
    parser.add_argument("--ingestion-backpressure-file", default=str(PROJECT_ROOT / "governance" / "health" / "ingestion_backpressure_latest.json"))
    parser.add_argument("--promotion-quality-file", default=str(PROJECT_ROOT / "governance" / "health" / "promotion_quality_gate_latest.json"))
    parser.add_argument("--broker-truth-file", default=str(PROJECT_ROOT / "governance" / "health" / "schwab_account_snapshot_refresh_latest.json"))
    parser.add_argument("--source-verification-file", default=str(PROJECT_ROOT / "governance" / "health" / "source_verification_latest.json"))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--platform-os-out-file", default=str(DEFAULT_PLATFORM_OS_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = evaluate_truth_layer(
        paper_performance=load_json(Path(args.paper_performance_file)),
        calibration=load_json(Path(args.calibration_file)),
        counterfactual=load_json(Path(args.counterfactual_file)),
        paper_replay=load_json(Path(args.paper_replay_file)),
        account_study=load_json(Path(args.account_study_file)),
        covered_call_watch=load_json(Path(args.covered_call_watch_file)),
        execution_lab=load_json(Path(args.execution_lab_file)),
        live_readiness=load_json(Path(args.live_readiness_file)),
        ingestion_storage=load_json(Path(args.ingestion_storage_file)),
        ingestion_backpressure=load_json(Path(args.ingestion_backpressure_file)),
        promotion_quality=load_json(Path(args.promotion_quality_file)),
        broker_truth=load_json(Path(args.broker_truth_file)),
        source_verification=load_json(Path(args.source_verification_file)),
    )
    payload["source_files"] = {
        "paper_performance": str(args.paper_performance_file),
        "calibration": str(args.calibration_file),
        "counterfactual": str(args.counterfactual_file),
        "paper_replay": str(args.paper_replay_file),
        "account_study": str(args.account_study_file),
        "covered_call_watch": str(args.covered_call_watch_file),
        "execution_lab": str(args.execution_lab_file),
        "live_readiness": str(args.live_readiness_file),
        "ingestion_storage": str(args.ingestion_storage_file),
        "ingestion_backpressure": str(args.ingestion_backpressure_file),
        "promotion_quality": str(args.promotion_quality_file),
        "broker_truth": str(args.broker_truth_file),
        "source_verification": str(args.source_verification_file),
    }
    write_payload(Path(args.out_file), payload)
    if str(args.platform_os_out_file or "").strip():
        write_payload(Path(args.platform_os_out_file), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        checks = ",".join(payload.get("failed_checks") or []) or "none"
        print(
            "paper_execution_truth_layer "
            f"status={payload.get('overall_status')} "
            f"score={float(payload.get('score', 0.0) or 0.0):.2f} "
            f"grade={payload.get('grade')} "
            f"failed_checks={checks}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
