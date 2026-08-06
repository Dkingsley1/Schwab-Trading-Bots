#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.profitability_statistics import benjamini_hochberg, clustered_post_cost_statistics
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from core.profitability_statistics import benjamini_hochberg, clustered_post_cost_statistics
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "profitability_evidence_firewall_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "profitability_evidence_firewall_latest.json"


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _grade(score: float, *, complete: bool = False) -> str:
    if complete and score >= 100.0:
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


def _control(control_id: str, title: str, implemented: bool, evidence_ready: bool, evidence: Any) -> dict[str, Any]:
    return {
        "control_id": control_id,
        "title": title,
        "implemented": bool(implemented),
        "implementation_status": "ready" if implemented else "blocked",
        "evidence_ready": bool(evidence_ready),
        "evidence_status": "ready" if evidence_ready else "collecting",
        "evidence": evidence,
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    if left_var <= 1e-18 or right_var <= 1e-18:
        return 0.0 if left == right else None
    return numerator / math.sqrt(left_var * right_var)


def _allocation_contract(performance: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    sleeves = _as_list(performance.get("sleeve_latest"))
    daily = _as_dict(performance.get("sleeve_daily_series"))
    qualified: list[str] = []
    for row in sleeves:
        if not isinstance(row, dict):
            continue
        expectancy = _as_dict(row.get("post_cost_expectancy"))
        if bool(expectancy.get("positive_clustered_lower_confidence_bound_95", False)):
            qualified.append(str(row.get("profile") or ""))
    series: dict[str, dict[str, float]] = {}
    volatility: dict[str, float] = {}
    for profile in qualified:
        values: dict[str, float] = {}
        for index, row in enumerate(_as_list(daily.get(profile))):
            if not isinstance(row, dict):
                continue
            key = str(row.get("day_utc") or row.get("day") or index)
            values[key] = _safe_float(row.get("change_vs_previous_day"), 0.0)
        series[profile] = values
        raw_values = list(values.values())
        mean = sum(raw_values) / max(len(raw_values), 1)
        volatility[profile] = math.sqrt(sum((value - mean) ** 2 for value in raw_values) / max(len(raw_values) - 1, 1))
    pairs: list[dict[str, Any]] = []
    max_abs_correlation = 0.0
    minimum_observations = int(policy.get("minimum_daily_observations", 7) or 7)
    correlation_evidence_ready = True
    for left_index, left in enumerate(sorted(qualified)):
        for right in sorted(qualified)[left_index + 1 :]:
            common = sorted(set(series.get(left, {})) & set(series.get(right, {})))
            correlation = _pearson(
                [series[left][key] for key in common],
                [series[right][key] for key in common],
            )
            if len(common) < minimum_observations or correlation is None:
                correlation_evidence_ready = False
            if correlation is not None:
                max_abs_correlation = max(max_abs_correlation, abs(correlation))
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "common_days": len(common),
                    "correlation": round(correlation, 8) if correlation is not None else None,
                }
            )
    inverse_volatility = {
        profile: 1.0 / max(volatility.get(profile, 0.0), 1e-9)
        for profile in qualified
    }
    total_inverse = sum(inverse_volatility.values())
    maximum_weight = _safe_float(policy.get("maximum_single_sleeve_weight"), 0.4)
    weights = {
        profile: min(inverse_volatility[profile] / max(total_inverse, 1e-9), maximum_weight)
        for profile in qualified
    }
    weight_total = sum(weights.values())
    if weight_total > 0.0:
        weights = {profile: round(value / weight_total, 8) for profile, value in weights.items()}
    minimum_sleeves = int(policy.get("minimum_profitable_sleeves", 3) or 3)
    maximum_correlation = _safe_float(policy.get("maximum_pairwise_correlation"), 0.65)
    ready = bool(
        len(qualified) >= minimum_sleeves
        and correlation_evidence_ready
        and max_abs_correlation <= maximum_correlation
        and weights
        and max(weights.values(), default=1.0) <= maximum_weight + 1e-9
    )
    return {
        "ready": ready,
        "qualified_sleeves": qualified,
        "qualified_sleeve_count": len(qualified),
        "pairwise_correlations": pairs,
        "maximum_absolute_correlation": round(max_abs_correlation, 8),
        "suggested_inverse_volatility_weights": weights,
        "automatic_allocation_allowed": False,
        "thresholds": policy,
        "policy": "only independently profitable, sufficiently observed, low-correlation sleeves may enter a conservative allocation proposal; application remains explicit",
    }


def _stress_contract(expectancy: dict[str, Any], stress_levels: list[float]) -> dict[str, Any]:
    robust = _as_dict(expectancy.get("robust_statistics"))
    count = max(int(expectancy.get("sample_count", 0) or 0), 1)
    notional_per_sample = _safe_float(expectancy.get("execution_notional_total"), 0.0) / count
    pnl_lcb = robust.get("promotion_lower_confidence_bound_95_post_cost_pnl_delta")
    return_lcb = robust.get("promotion_lower_confidence_bound_95_post_cost_return_bps")
    rows: list[dict[str, Any]] = []
    for stress_bps in stress_levels:
        stressed_pnl = None if pnl_lcb is None else _safe_float(pnl_lcb) - notional_per_sample * stress_bps / 10000.0
        stressed_return = None if return_lcb is None else _safe_float(return_lcb) - stress_bps
        rows.append(
            {
                "stress_cost_bps": stress_bps,
                "stressed_pnl_lcb": round(stressed_pnl, 8) if stressed_pnl is not None else None,
                "stressed_return_lcb_bps": round(stressed_return, 8) if stressed_return is not None else None,
                "passes": bool(stressed_pnl is not None and stressed_return is not None and stressed_pnl > 0.0 and stressed_return > 0.0),
            }
        )
    return {"ready": bool(rows and all(row["passes"] for row in rows)), "scenarios": rows}


def build_payload(project_root: Path = PROJECT_ROOT, *, config_path: Path | None = None) -> dict[str, Any]:
    config = load_json(config_path or project_root / "config" / DEFAULT_CONFIG_PATH.name)
    health = project_root / "governance" / "health"
    source = load_json(health / "source_verification_latest.json")
    fill = load_json(health / "paper_execution_calibration_latest.json")
    performance = load_json(health / "paper_performance_latest.json")
    paper_control = load_json(health / "paper_runtime_profitability_controls_latest.json") or load_json(
        health / "paper_profitability_control_latest.json"
    )
    counterfactual = load_json(health / "counterfactual_replay_latest.json")
    multiple_testing = load_json(project_root / "governance" / "research" / "multiple_testing_guard_latest.json")
    expectancy = _as_dict(performance.get("post_cost_expectancy"))
    robust = _as_dict(expectancy.get("robust_statistics"))
    source_policy = _as_dict(config.get("source_verification"))
    source_overall = _as_dict(source.get("overall"))
    source_ready = bool(
        source.get("ok", False)
        and (not source_policy.get("require_all_verified", True) or source_overall.get("all_verified", False))
        and _safe_float(source_overall.get("mean_source_confidence_score"), 0.0)
        >= _safe_float(source_policy.get("minimum_mean_confidence"), 0.9)
        and _safe_float(source_overall.get("min_source_confidence_score"), 0.0)
        >= _safe_float(source_policy.get("minimum_source_confidence"), 0.7)
    )
    fill_policy = _as_dict(config.get("independent_fill_evidence"))
    fill_ready = bool(
        int(fill.get("independent_samples", 0) or 0) >= int(fill_policy.get("minimum_samples", 100) or 100)
        and (not fill_policy.get("require_independent_evidence_ready", True) or fill.get("independent_evidence_ready", False))
    )
    improvement = _as_dict(paper_control.get("raw_profitability_improvement_contract"))
    clean_gate = _as_dict(improvement.get("clean_sleeve_strict_buy_gate_contract"))
    entry_ready = bool(clean_gate.get("active", False) and clean_gate.get("enforced", False) and clean_gate.get("allow_buy_only_when_all_gates_pass", False))
    zero_entry = _as_dict(improvement.get("weak_sleeve_zero_entry_contract"))
    profile_rows = {
        str(row.get("profile") or ""): row
        for row in _as_list(zero_entry.get("profiles"))
        if isinstance(row, dict)
    }
    mandatory_quarantine = [str(item) for item in _as_list(config.get("mandatory_quarantine_profiles"))]
    quarantine_ready = bool(
        mandatory_quarantine
        and all(
            bool(profile_rows.get(profile, {}).get("block_new_entries", False))
            and _safe_float(profile_rows.get(profile, {}).get("new_entry_cap"), 1.0) == 0.0
            for profile in mandatory_quarantine
        )
    )
    hardening = _as_dict(paper_control.get("paper_profitability_hardening_contract"))
    scout = _as_dict(hardening.get("scout_collection_contract")) or _as_dict(paper_control.get("scout_collection_contract"))
    required_labels = {str(item) for item in _as_list(scout.get("required_label_outputs"))}
    configured_labels = {str(item) for item in _as_list(config.get("counterfactual_labels"))}
    counterfactual_ready = bool(
        counterfactual.get("ok", False)
        and int(counterfactual.get("candidate_count", 0) or 0) > 0
        and configured_labels.issubset(required_labels)
    )
    stress = _stress_contract(expectancy, [_safe_float(item) for item in _as_list(config.get("stress_cost_bps"))])
    cluster_ready = bool(robust.get("promotion_evidence_sufficient", False))
    statistical_ready = bool(multiple_testing.get("statistical_evidence_ready", False))
    stat_policy = _as_dict(config.get("statistical_evidence"))
    dsr = _as_dict(robust.get("deflated_sharpe"))
    oos_ready = bool(
        robust.get("positive_clustered_lower_confidence_bound_95", False)
        and int(robust.get("unique_day_count", 0) or 0) >= int(stat_policy.get("minimum_independent_days", 7) or 7)
        and int(robust.get("unique_regime_count", 0) or 0) >= int(stat_policy.get("minimum_regimes", 2) or 2)
        and _safe_float(dsr.get("probability"), 0.0) >= _safe_float(stat_policy.get("minimum_deflated_sharpe_probability"), 0.95)
    )
    allocation = _allocation_contract(performance, _as_dict(config.get("allocation")))

    implementation_self_test = clustered_post_cost_statistics(
        [
            {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "SPY",
                "strategy": "self_test",
                "post_cost_pnl_delta": 1.0,
                "post_cost_return_bps": 1.0,
            }
            for _index in range(100)
        ]
    )
    fdr_self_test = benjamini_hochberg({"one": 0.001, "two": 0.5})
    oos_control_implemented = bool(
        all(
            key in implementation_self_test
            for key in (
                "unique_day_count",
                "unique_regime_count",
                "deflated_sharpe",
                "promotion_lower_confidence_bound_95_post_cost_pnl_delta",
                "positive_clustered_lower_confidence_bound_95",
            )
        )
        and all(
            key in stat_policy
            for key in (
                "minimum_independent_days",
                "minimum_regimes",
                "minimum_deflated_sharpe_probability",
            )
        )
    )
    source_code = (project_root / "core" / "base_trader.py").read_text(encoding="utf-8") if (project_root / "core" / "base_trader.py").is_file() else ""
    controls = [
        _control("01_source_verification", "Verified and confidence-scored point-in-time sources", bool(config and source_policy), source_ready, source_overall),
        _control("02_independent_fills", "Independent fills and explicit cost calibration", bool(config and fill_policy), fill_ready, fill),
        _control("03_fail_closed_entry_quality", "Unknown spread, source, fill, session, event, or tradeability evidence blocks entries", "paper_profitability_clean_profile_evidence_block" in source_code, entry_ready, clean_gate),
        _control("04_weak_sleeve_quarantine", "Known weak event and aggressive futures sleeves remain collect-only", bool(mandatory_quarantine), quarantine_ready, {"required": mandatory_quarantine, "profiles": {key: profile_rows.get(key, {}) for key in mandatory_quarantine}}),
        _control("05_counterfactual_path_labels", "No-trade, MAE, MFE, exit timing, and post-entry regime labels feed training", bool(configured_labels), counterfactual_ready, {"required": sorted(configured_labels), "present": sorted(required_labels), "counterfactual_candidate_count": counterfactual.get("candidate_count", 0)}),
        _control("06_stressed_post_cost_expectancy", "Expectancy remains positive after explicit cost stress", bool(config.get("stress_cost_bps")), bool(stress.get("ready", False)), stress),
        _control("07_cluster_effective_samples", "Effective samples are clustered by independent evidence units", bool(implementation_self_test.get("available") and not implementation_self_test.get("promotion_evidence_sufficient")), cluster_ready, robust),
        _control("08_multiple_testing_firewall", "Actual FDR, deflated Sharpe, and PBO control selection bias", bool(fdr_self_test.get("hypothesis_count") == 2), statistical_ready, multiple_testing),
        _control("09_oos_regime_lcb", "Positive lower bound persists across independent days and regimes", oos_control_implemented, oos_ready, {"robust_statistics": robust, "thresholds": stat_policy}),
        _control("10_conservative_allocation", "Only low-correlation independently profitable sleeves receive proposed weight", True, bool(allocation.get("ready", False)), allocation),
    ]
    implemented_count = sum(1 for row in controls if row["implemented"])
    evidence_count = sum(1 for row in controls if row["evidence_ready"])
    control_ready = implemented_count == len(controls)
    economic_ready = evidence_count == len(controls)
    control_score = 100.0 * implemented_count / max(len(controls), 1)
    economic_score = 100.0 * evidence_count / max(len(controls), 1)
    blockers = ordered_unique(
        [row["control_id"] for row in controls if not row["implemented"]]
        + [row["control_id"] for row in controls if not row["evidence_ready"]]
    )
    raw_grade = str(paper_control.get("raw_profitability_grade") or "unknown")
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": control_ready,
        "overall_status": "ready" if control_ready else "blocked",
        "control_grade": _grade(control_score, complete=control_ready),
        "control_score": round(control_score, 3),
        "economic_evidence_grade": _grade(economic_score, complete=economic_ready),
        "economic_evidence_score": round(economic_score, 3),
        "promotion_evidence_ready": economic_ready,
        "raw_profitability_grade": raw_grade,
        "raw_profitability_grade_overridden": False,
        "implemented_control_count": implemented_count,
        "evidence_ready_control_count": evidence_count,
        "control_count": len(controls),
        "controls": controls,
        "blockers": blockers,
        "allocation_proposal": allocation,
        "grading_contract": {
            "control_A_plus_means_all_ten_fail_closed_controls_are_implemented": True,
            "economic_A_plus_requires_all_ten_current_evidence_checks": True,
            "negative_or_insufficient_raw_results_can_never_be_relabelled": True,
            "paper_controls_do_not_authorize_live_money": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the ten-control profitability evidence firewall.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    out_path = args.out_file if args.out_file.is_absolute() else project_root / args.out_file
    payload = build_payload(project_root, config_path=config_path)
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "profitability_evidence_firewall "
            f"control_grade={payload['control_grade']} economic_grade={payload['economic_evidence_grade']} "
            f"promotion_ready={int(bool(payload['promotion_evidence_ready']))}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
