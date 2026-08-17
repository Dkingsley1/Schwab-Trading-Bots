#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def _experiment_ledger_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
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
    hardening_config = _load_json(project_root / "config" / "profitability_evidence_firewall_v1.json")

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
    experiment_ledger_path = project_root / "governance" / "research" / "experiment_ledger.jsonl"
    experiment_ledger_ids = _experiment_ledger_ids(experiment_ledger_path)
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
    daily_series = paper_performance.get("sleeve_daily_series") if isinstance(paper_performance.get("sleeve_daily_series"), dict) else {}
    strategy_period_returns: dict[str, list[float]] = {}
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
    pbo = probability_of_backtest_overfitting(strategy_period_returns)
    statistical_evidence_ready = bool(
        lineage_complete
        and
        actual_fdr.get("hypothesis_count", 0) >= 2
        and actual_fdr.get("passing_hypotheses")
        and pbo.get("available", False)
        and pbo.get("passes", False)
    )
    statistical_blockers: list[str] = []
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

    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 2,
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
            "conservative_family_size": family_size,
        },
        "experiment_lineage": {
            "complete": lineage_complete,
            "registry_floor_required": registry_floor_required,
            "strategy_roles": sorted(strategy_roles),
            "registry_hypothesis_count": len(registry_hypothesis_ids),
            "experiment_ledger_hypothesis_count": len(experiment_ledger_ids),
            "experiment_ledger_path": str(experiment_ledger_path),
            "policy": "the statistical family is never smaller than the complete registered strategy or immutable experiment lineage; deleted, excluded, and failed strategies remain in the family",
        },
        "hypotheses": hypotheses,
        "regime_segments": regime_segments,
        "strict_checks": strict_checks,
        "baseline_metrics": ablation_block.get("baseline") if isinstance(ablation_block.get("baseline"), dict) else {},
        "delta_metrics": ablation.get("delta") if isinstance(ablation.get("delta"), dict) else {},
        "failed_checks": failed_checks,
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
        "statistical_evidence_ready": statistical_evidence_ready,
        "statistical_evidence_blockers": statistical_blockers,
        "grading_contract": {
            "ok_measures_structural_research_control": True,
            "statistical_evidence_ready_requires_actual_p_values_and_pbo": True,
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
            "experiment_ledger": str(experiment_ledger_path),
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
