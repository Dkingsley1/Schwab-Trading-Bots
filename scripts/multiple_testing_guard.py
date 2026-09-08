#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.profitability_statistics import benjamini_hochberg, probability_of_backtest_overfitting


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "research" / "multiple_testing_guard_latest.json"
STRATEGY_ROLES = {
    "crypto_sub_bot",
    "futures_sub_bot",
    "macro_sub_bot",
    "options_sub_bot",
    "signal_sub_bot",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _aligned_candidate_period_returns(
    raw_series: Any,
    *,
    minimum_periods: int = 8,
) -> tuple[dict[str, list[float]], list[str]]:
    if not isinstance(raw_series, dict):
        return {}, []
    by_profile: dict[str, dict[str, float]] = {}
    for raw_profile, rows in raw_series.items():
        profile = str(raw_profile or "").strip().lower()
        if not profile or not isinstance(rows, list):
            continue
        daily: dict[str, float] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            day = str(row.get("day_utc") or "").strip()
            if not day:
                continue
            daily[day] = _safe_float(
                row.get("post_cost_return_bps_total"),
                0.0,
            )
        if len(daily) >= max(int(minimum_periods), 4):
            by_profile[profile] = daily
    if len(by_profile) < 2:
        return {}, []
    common_days = sorted(
        set.intersection(*(set(values) for values in by_profile.values()))
    )
    if len(common_days) < max(int(minimum_periods), 4):
        return {}, common_days
    return {
        profile: [daily[day] for day in common_days]
        for profile, daily in sorted(by_profile.items())
    }, common_days


def _purged_walk_forward_evaluation(
    raw_series: Any,
    policy: dict[str, Any],
) -> dict[str, Any]:
    enabled = bool(policy.get("enabled", False))
    fold_count = max(_safe_int(policy.get("fold_count"), 5), 1)
    purge_periods = max(_safe_int(policy.get("purge_periods"), 1), 0)
    embargo_periods = max(_safe_int(policy.get("embargo_periods"), 1), 0)
    minimum_train = max(_safe_int(policy.get("minimum_train_periods"), 12), 2)
    minimum_test = max(_safe_int(policy.get("minimum_test_periods"), 3), 1)
    minimum_total = max(
        _safe_int(policy.get("minimum_total_periods"), 30),
        minimum_train + purge_periods + minimum_test,
    )
    minimum_folds = max(_safe_int(policy.get("minimum_completed_folds"), 3), 1)
    minimum_positive_fold_rate = min(
        max(_safe_float(policy.get("minimum_positive_fold_rate"), 0.6), 0.0),
        1.0,
    )
    implementation_ready = bool(
        enabled
        and fold_count >= minimum_folds
        and purge_periods >= 1
        and embargo_periods >= 1
        and minimum_total > minimum_train
    )
    strategy_rows: list[dict[str, Any]] = []
    if isinstance(raw_series, dict):
        for strategy_id, raw_rows in sorted(raw_series.items()):
            if not isinstance(raw_rows, list):
                continue
            daily = sorted(
                (
                    str(row.get("day_utc") or "").strip(),
                    _safe_float(row.get("post_cost_return_bps_total"), 0.0),
                )
                for row in raw_rows
                if isinstance(row, dict) and str(row.get("day_utc") or "").strip()
            )
            deduped = {day: value for day, value in daily}
            observations = sorted(deduped.items())
            folds: list[dict[str, Any]] = []
            cursor = minimum_train + purge_periods
            while (
                cursor + minimum_test <= len(observations) and len(folds) < fold_count
            ):
                train_end = cursor - purge_periods
                test_start = cursor
                test_end = min(test_start + minimum_test, len(observations))
                train = observations[:train_end]
                test = observations[test_start:test_end]
                if len(train) < minimum_train or len(test) < minimum_test:
                    break
                train_values = [value for _day, value in train]
                test_values = [value for _day, value in test]
                folds.append(
                    {
                        "fold": len(folds) + 1,
                        "train_period_count": len(train_values),
                        "test_period_count": len(test_values),
                        "purge_periods": purge_periods,
                        "embargo_periods": embargo_periods,
                        "train_end_day": train[-1][0],
                        "test_start_day": test[0][0],
                        "test_end_day": test[-1][0],
                        "train_mean_return_bps": round(
                            sum(train_values) / len(train_values), 8
                        ),
                        "test_mean_return_bps": round(
                            sum(test_values) / len(test_values), 8
                        ),
                        "test_positive": sum(test_values) > 0.0,
                    }
                )
                cursor = test_end + embargo_periods
            test_values_all = [
                value
                for fold in folds
                for _day, value in observations[
                    next(
                        index
                        for index, (day, _value) in enumerate(observations)
                        if day == fold["test_start_day"]
                    ) : next(
                        index
                        for index, (day, _value) in enumerate(observations)
                        if day == fold["test_end_day"]
                    )
                    + 1
                ]
            ]
            positive_fold_rate = (
                sum(1 for fold in folds if fold["test_positive"]) / len(folds)
                if folds
                else 0.0
            )
            oos_mean = (
                sum(test_values_all) / len(test_values_all) if test_values_all else 0.0
            )
            evidence_available = bool(
                len(observations) >= minimum_total and len(folds) >= minimum_folds
            )
            passes = bool(
                evidence_available
                and oos_mean > 0.0
                and positive_fold_rate >= minimum_positive_fold_rate
            )
            strategy_rows.append(
                {
                    "strategy_id": str(strategy_id),
                    "period_count": len(observations),
                    "completed_fold_count": len(folds),
                    "evidence_available": evidence_available,
                    "oos_mean_return_bps": round(oos_mean, 8),
                    "positive_fold_rate": round(positive_fold_rate, 8),
                    "passes": passes,
                    "folds": folds,
                }
            )
    evaluated = [row for row in strategy_rows if row["evidence_available"]]
    passing = [row for row in evaluated if row["passes"]]
    evidence_ready = bool(
        implementation_ready and evaluated and len(passing) == len(evaluated)
    )
    blockers: list[str] = []
    if not implementation_ready:
        blockers.append("purged_walk_forward_policy_invalid")
    if not evaluated:
        blockers.append("minimum_purged_oos_periods_pending")
    elif len(passing) != len(evaluated):
        blockers.append("purged_oos_expectancy_not_positive")
    return {
        "implementation_ready": implementation_ready,
        "evidence_ready": evidence_ready,
        "evaluated_strategy_count": len(evaluated),
        "passing_strategy_count": len(passing),
        "strategies": strategy_rows,
        "thresholds": {
            "fold_count": fold_count,
            "purge_periods": purge_periods,
            "embargo_periods": embargo_periods,
            "minimum_train_periods": minimum_train,
            "minimum_test_periods": minimum_test,
            "minimum_total_periods": minimum_total,
            "minimum_completed_folds": minimum_folds,
            "minimum_positive_fold_rate": minimum_positive_fold_rate,
        },
        "blockers": blockers,
        "policy": "chronological test folds are separated from training by purge gaps and from subsequent folds by embargo gaps; only out-of-sample post-cost returns count",
    }


def _experiment_ledger_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    handle = (
        gzip.open(path, "rt", encoding="utf-8", errors="replace")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8", errors="replace")
    )
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            value = str(
                row.get("hypothesis_id")
                or row.get("experiment_id")
                or row.get("trial_id")
                or ""
            ).strip()
            if value:
                out.add(value)
            elif row:
                out.add(f"anonymous-line-{line_number}")
    return out


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"

    ablation = _load_json(health_root / "replay_feature_ablation_latest.json")
    counterfactual = _load_json(health_root / "counterfactual_replay_latest.json")
    promotion_readiness = _load_json(walk_root / "promotion_readiness_latest.json")
    paper_performance = _load_json(health_root / "paper_performance_latest.json")
    registry = _load_json(project_root / "master_bot_registry.json")
    hardening_config = _load_json(
        project_root / "config" / "profitability_evidence_firewall_v1.json"
    )
    self_assessment = _load_json(
        project_root / "config" / "profitability_self_assessment_v1.json"
    )
    validation_protocol = (
        self_assessment.get("validation_protocol")
        if isinstance(self_assessment.get("validation_protocol"), dict)
        else {}
    )
    purged_policy = (
        validation_protocol.get("purged_walk_forward")
        if isinstance(validation_protocol.get("purged_walk_forward"), dict)
        else {}
    )

    ablation_block = ablation.get("ablation") if isinstance(ablation.get("ablation"), dict) else {}
    strict_checks = ablation.get("strict_checks") if isinstance(ablation.get("strict_checks"), dict) else {}
    failed_checks = ablation.get("failed_checks") if isinstance(ablation.get("failed_checks"), list) else []
    profiles_reviewed = counterfactual.get("profiles_reviewed") if isinstance(counterfactual.get("profiles_reviewed"), list) else []

    feature_hypotheses = 0
    for key, value in ablation_block.items():
        if key == "baseline":
            continue
        if isinstance(value, dict):
            feature_hypotheses += 1
    if feature_hypotheses <= 0:
        feature_hypotheses = max(_safe_int(ablation_block.get("e2e_feature_count"), 0) + _safe_int(ablation_block.get("paper_feature_count"), 0), 0)

    counterfactual_candidates = _safe_int(counterfactual.get("candidate_count"), 0)
    considered_bots = _safe_int(promotion_readiness.get("considered_bots"), 0)
    derived_family_size = max(feature_hypotheses + counterfactual_candidates + considered_bots, 0)
    lineage_policy = hardening_config.get("experiment_lineage") if isinstance(hardening_config.get("experiment_lineage"), dict) else {}
    configured_roles = {
        str(item or "").strip()
        for item in lineage_policy.get("strategy_roles") or []
        if str(item or "").strip()
    }
    strategy_roles = configured_roles or STRATEGY_ROLES
    registry_rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    registry_hypothesis_ids = {
        str(row.get("bot_id") or "").strip()
        for row in registry_rows
        if isinstance(row, dict)
        and str(row.get("bot_role") or "").strip() in strategy_roles
        and str(row.get("bot_id") or "").strip()
    }
    experiment_ledger_paths = [
        project_root / "governance" / "research" / "experiment_ledger.jsonl",
        project_root / "governance" / "experiments" / "immutable_experiment_ledger.jsonl",
        project_root / "governance" / "experiments" / "immutable_experiment_ledger.jsonl.gz",
    ]
    experiment_ledger_ids_by_path = {
        str(path): _experiment_ledger_ids(path)
        for path in experiment_ledger_paths
        if path.is_file()
    }
    experiment_ledger_ids = set().union(*experiment_ledger_ids_by_path.values()) if experiment_ledger_ids_by_path else set()
    family_size = max(derived_family_size, len(registry_hypothesis_ids), len(experiment_ledger_ids), 0)
    registry_floor_required = bool(lineage_policy.get("require_complete_registry_floor", True))
    lineage_complete = bool(
        family_size > 0
        and (not registry_floor_required or not registry_hypothesis_ids or family_size >= len(registry_hypothesis_ids))
    )
    method = "benjamini_hochberg_fdr" if family_size >= 10 else "bonferroni" if family_size > 0 else "not_applicable"
    base_alpha = 0.05
    corrected_alpha = base_alpha if method == "benjamini_hochberg_fdr" else round(base_alpha / max(family_size, 1), 6) if family_size > 0 else 0.0
    regime_segments = sorted({str(profile).strip().lower() for profile in profiles_reviewed if str(profile).strip()})
    if not regime_segments:
        regime_segments = ["global"]

    hypotheses = [
        {
            "family": "feature_ablation",
            "hypothesis_count": feature_hypotheses,
            "evidence_path": str(health_root / "replay_feature_ablation_latest.json"),
        },
        {
            "family": "counterfactual_threshold_search",
            "hypothesis_count": counterfactual_candidates,
            "evidence_path": str(health_root / "counterfactual_replay_latest.json"),
        },
        {
            "family": "promotion_candidates",
            "hypothesis_count": considered_bots,
            "evidence_path": str(walk_root / "promotion_readiness_latest.json"),
        },
    ]

    ablation_contract_present = bool(ablation and (ablation_block or strict_checks or ablation.get("delta")))
    counterfactual_contract_present = bool(counterfactual and (counterfactual_candidates > 0 or profiles_reviewed))
    promotion_contract_present = bool(promotion_readiness and (considered_bots > 0 or promotion_readiness.get("coverage_shortfall_bots") is not None))
    ablation_ready = bool(ablation.get("ok", False) or (ablation_contract_present and not failed_checks and feature_hypotheses > 0))
    counterfactual_ready = bool(counterfactual.get("ok", False) or counterfactual_contract_present)
    contract_present = bool(family_size > 0 and (ablation_contract_present or counterfactual_contract_present or promotion_contract_present))

    ok = bool(ablation_ready and counterfactual_ready and family_size > 0 and lineage_complete and not failed_checks)
    overall_status = "ready" if ok else "needs_work"
    if family_size <= 0 or not contract_present:
        overall_status = "blocked"

    sleeve_rows = paper_performance.get("sleeve_latest") if isinstance(paper_performance.get("sleeve_latest"), list) else []
    evidence_window = (
        paper_performance.get("profitability_evidence_window")
        if isinstance(paper_performance.get("profitability_evidence_window"), dict)
        else {}
    )
    candidate_id = str(evidence_window.get("candidate_id") or "").strip()
    candidate_binding_required = bool(
        evidence_window.get("candidate_binding_required", False)
    )
    candidate_binding_mismatches = max(
        _safe_int(
            evidence_window.get("candidate_binding_mismatch_rows_excluded"),
            0,
        ),
        0,
    )
    candidate_bound = bool(
        not candidate_binding_required
        or (
            candidate_id
            and evidence_window.get("candidate_filter_active", False)
            and candidate_binding_mismatches == 0
        )
    )
    sleeve_p_values: dict[str, float] = {}
    for row in sleeve_rows:
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile") or "").strip()
        expectancy = row.get("post_cost_expectancy") if isinstance(row.get("post_cost_expectancy"), dict) else {}
        robust = expectancy.get("robust_statistics") if isinstance(expectancy.get("robust_statistics"), dict) else {}
        raw_p = robust.get("one_sided_positive_expectancy_p_value")
        if raw_p is None:
            continue
        try:
            sleeve_p_values[profile] = float(raw_p)
        except Exception:
            continue
    actual_fdr = benjamini_hochberg(sleeve_p_values, alpha=base_alpha)
    common_candidate_days: list[str] = []
    if candidate_binding_required:
        strategy_period_returns, common_candidate_days = (
            _aligned_candidate_period_returns(
                paper_performance.get("candidate_post_cost_daily_series")
            )
        )
        pbo_series_scope = "candidate_forward_profile_daily_post_cost_returns"
    else:
        daily_series = paper_performance.get("sleeve_daily_series") if isinstance(paper_performance.get("sleeve_daily_series"), dict) else {}
        strategy_period_returns = {}
        for profile, rows in daily_series.items():
            if not isinstance(rows, list):
                continue
            values = [
                _safe_float(row.get("change_vs_previous_day"), 0.0)
                for row in rows
                if isinstance(row, dict)
            ]
            if values:
                strategy_period_returns[str(profile)] = values
        pbo_series_scope = "legacy_lifetime_sleeve_daily_returns"
    pbo = probability_of_backtest_overfitting(strategy_period_returns)
    purged_walk_forward = _purged_walk_forward_evaluation(
        paper_performance.get("candidate_strategy_post_cost_daily_series"),
        purged_policy,
    )
    statistical_evidence_ready = bool(
        lineage_complete
        and candidate_bound
        and
        actual_fdr.get("hypothesis_count", 0) >= 2
        and actual_fdr.get("passing_hypotheses")
        and pbo.get("available", False)
        and pbo.get("passes", False)
        and purged_walk_forward.get("evidence_ready", False)
    )
    statistical_blockers: list[str] = []
    if not candidate_bound:
        statistical_blockers.append("candidate_binding_not_ready")
    if not lineage_complete:
        statistical_blockers.append("complete_experiment_lineage_pending")
    if actual_fdr.get("hypothesis_count", 0) < 2:
        statistical_blockers.append("actual_sleeve_p_values_pending")
    elif not actual_fdr.get("passing_hypotheses"):
        statistical_blockers.append("no_sleeve_passes_fdr")
    if not pbo.get("available", False):
        statistical_blockers.extend(str(item) for item in pbo.get("blockers", []) if str(item))
    elif not pbo.get("passes", False):
        statistical_blockers.append("probability_of_backtest_overfitting_above_ceiling")
    statistical_blockers.extend(
        str(item) for item in purged_walk_forward.get("blockers", []) if str(item)
    )

    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 4,
        "ok": ok,
        "overall_status": overall_status,
        "contract_present": contract_present,
        "ablation_contract_ready": ablation_ready,
        "counterfactual_contract_ready": counterfactual_ready,
        "promotion_contract_present": promotion_contract_present,
        "base_alpha": base_alpha,
        "correction_method": method,
        "corrected_alpha": corrected_alpha,
        "family_size": family_size,
        "family_size_components": {
            "derived_research_family_size": derived_family_size,
            "registry_strategy_hypothesis_count": len(registry_hypothesis_ids),
            "experiment_ledger_hypothesis_count": len(experiment_ledger_ids),
            "experiment_ledger_counts_by_path": {
                path: len(ids)
                for path, ids in sorted(experiment_ledger_ids_by_path.items())
            },
            "conservative_family_size": family_size,
        },
        "experiment_lineage": {
            "complete": lineage_complete,
            "registry_floor_required": registry_floor_required,
            "strategy_roles": sorted(strategy_roles),
            "registry_hypothesis_count": len(registry_hypothesis_ids),
            "experiment_ledger_hypothesis_count": len(experiment_ledger_ids),
            "experiment_ledger_paths": sorted(experiment_ledger_ids_by_path),
            "compressed_immutable_ledger_supported": True,
            "policy": "the statistical family is never smaller than the complete registered strategy or immutable experiment lineage; deleted, excluded, and failed strategies remain in the family",
        },
        "hypotheses": hypotheses,
        "regime_segments": regime_segments,
        "strict_checks": strict_checks,
        "baseline_metrics": ablation_block.get("baseline") if isinstance(ablation_block.get("baseline"), dict) else {},
        "delta_metrics": ablation.get("delta") if isinstance(ablation.get("delta"), dict) else {},
        "failed_checks": failed_checks,
        "candidate_binding": {
            "candidate_id": candidate_id,
            "generation": _safe_int(
                evidence_window.get("candidate_generation"),
                0,
            ),
            "cutoff_utc": str(evidence_window.get("candidate_cutoff_utc") or ""),
            "evidence_through_utc": str(
                evidence_window.get("evidence_through_utc") or ""
            ),
            "required": candidate_binding_required,
            "bound": candidate_bound,
            "mismatch_rows_excluded": candidate_binding_mismatches,
            "pbo_series_scope": pbo_series_scope,
            "pbo_common_period_days": common_candidate_days,
        },
        "actual_statistical_correction": actual_fdr,
        "deflated_sharpe_available_by_sleeve": {
            str(row.get("profile") or ""): (
                (row.get("post_cost_expectancy") or {}).get("robust_statistics", {}).get("deflated_sharpe", {})
                if isinstance(row.get("post_cost_expectancy"), dict)
                else {}
            )
            for row in sleeve_rows
            if isinstance(row, dict) and str(row.get("profile") or "").strip()
        },
        "probability_of_backtest_overfitting": pbo,
        "purged_walk_forward": purged_walk_forward,
        "statistical_evidence_ready": statistical_evidence_ready,
        "statistical_evidence_blockers": statistical_blockers,
        "grading_contract": {
            "ok_measures_structural_research_control": True,
            "statistical_evidence_ready_requires_actual_p_values_and_pbo": True,
            "candidate_bound_pbo_periods_required": True,
            "purged_and_embargoed_oos_folds_required": True,
            "declared_correction_method_is_not_profitability_evidence": True,
            "all_registered_strategy_hypotheses_count_toward_selection_bias": True,
            "discarded_experiments_cannot_disappear_from_the_family_size": True,
        },
        "recommendations": [
            "Keep correction families stable across feature ablation, counterfactual threshold search, and promotion review batches.",
            "Segment research verdicts by lane or regime when profiles_reviewed spans materially different sleeves.",
        ],
        "source_files": {
            "replay_feature_ablation": str(health_root / "replay_feature_ablation_latest.json"),
            "counterfactual_replay": str(health_root / "counterfactual_replay_latest.json"),
            "promotion_readiness": str(walk_root / "promotion_readiness_latest.json"),
            "master_bot_registry": str(project_root / "master_bot_registry.json"),
            "validation_protocol": str(
                project_root / "config" / "profitability_self_assessment_v1.json"
            ),
            "experiment_ledgers": sorted(experiment_ledger_ids_by_path),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a multiple-testing control artifact for replay and promotion research.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "multiple_testing_guard "
            f"status={payload['overall_status']} "
            f"family_size={int(payload.get('family_size', 0) or 0)} "
            f"method={payload.get('correction_method', '')}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
