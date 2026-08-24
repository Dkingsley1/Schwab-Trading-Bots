#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.alpha_evidence_contract import build_cross_sleeve_alpha_map
    from scripts.ops.long_runtime_common import load_json, write_payload
else:
    from core.alpha_evidence_contract import build_cross_sleeve_alpha_map
    from .long_runtime_common import PROJECT_ROOT, load_json, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "alpha_generation_control_v1.json"
DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "alpha_generation_control_latest.json"
)
DEFAULT_MARKDOWN_PATH = (
    PROJECT_ROOT
    / "exports"
    / "reports"
    / "operator"
    / "alpha_generation_control_latest.md"
)
SOURCE_PATHS = {
    "candidate": "governance/runtime/production_candidate_state.json",
    "profitability": "governance/health/profitability_self_assessment_latest.json",
    "paper_performance": "governance/health/paper_performance_latest.json",
    "execution_calibration": "governance/health/paper_execution_calibration_latest.json",
    "multiple_testing": "governance/research/multiple_testing_guard_latest.json",
    "training_quality": "governance/health/training_quality_control_latest.json",
    "label_materialization": "governance/training_labeling_intelligence/all_bot_label_materialization_latest.json",
    "regime": "governance/health/regime_control_plane_latest.json",
    "bot_profitability": "governance/health/bot_profitability_scalability_latest.json",
    "strategy_generation": "governance/health/strategy_generation_control_latest.json",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return (
        parsed if parsed == parsed and abs(parsed) != float("inf") else float(default)
    )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


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


def _receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False, "sha256": ""}
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        digest = ""
    return {"path": str(path), "present": True, "sha256": digest}


def _control(
    control_id: str,
    title: str,
    *,
    implementation_ready: bool,
    evidence_ready: bool,
    measurement: dict[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "control_id": control_id,
        "title": title,
        "implementation_ready": bool(implementation_ready),
        "implementation_status": "ready" if implementation_ready else "blocked",
        "evidence_ready": bool(evidence_ready),
        "evidence_status": "ready" if evidence_ready else "collecting",
        "measurement": measurement,
        "blockers": sorted({str(item) for item in blockers if str(item)}),
        "changes_runtime_decisions": False,
        "live_execution_authority": False,
    }


def _candidate_series_days(series: Mapping[str, Any]) -> int:
    days = {
        str(row.get("day_utc") or "")
        for rows in series.values()
        for row in _as_list(rows)
        if isinstance(row, dict) and str(row.get("day_utc") or "")
    }
    return len(days)


def _candidate_symbols(paper: dict[str, Any]) -> int:
    expectancy = _as_dict(paper.get("post_cost_expectancy"))
    robust = _as_dict(expectancy.get("robust_statistics"))
    return _safe_int(
        robust.get("unique_symbol_count"),
        _safe_int(robust.get("symbol_count"), 0),
    )


def _statistically_qualified_sleeves(
    paper: dict[str, Any],
    multiple_testing: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[set[str], dict[str, list[str]]]:
    statistical_policy = _as_dict(policy.get("statistical_validation"))
    minimum_dsr = _safe_float(
        statistical_policy.get("minimum_deflated_sharpe_probability"), 0.95
    )
    maximum_pbo = _safe_float(
        statistical_policy.get("maximum_probability_of_backtest_overfitting"), 0.2
    )
    expectancy_ready: set[str] = set()
    reasons: dict[str, list[str]] = {}
    for row in _as_list(paper.get("sleeve_latest")):
        if not isinstance(row, dict):
            continue
        sleeve = str(row.get("profile") or "").strip()
        if not sleeve:
            continue
        expectancy = _as_dict(row.get("post_cost_expectancy"))
        sleeve_reasons: list[str] = []
        if not bool(expectancy.get("promotion_evidence_sufficient", False)):
            sleeve_reasons.append("independent_post_cost_evidence_pending")
        if not bool(
            expectancy.get("positive_clustered_lower_confidence_bound_95", False)
        ):
            sleeve_reasons.append("positive_clustered_lcb_pending")
        if not sleeve_reasons:
            expectancy_ready.add(sleeve)
        reasons[sleeve] = sleeve_reasons

    correction = _as_dict(multiple_testing.get("actual_statistical_correction"))
    passing = {
        str(item)
        for item in _as_list(correction.get("passing_hypotheses"))
        if str(item)
    }
    passing_sleeves: set[str] = set()
    for row in _as_list(correction.get("rows")):
        if not isinstance(row, dict):
            continue
        identity = str(
            row.get("sleeve")
            or row.get("profile")
            or row.get("hypothesis_id")
            or row.get("hypothesis")
            or ""
        )
        if not identity:
            continue
        if bool(row.get("passes", row.get("rejected", identity in passing))):
            passing_sleeves.add(identity)
    passing_sleeves.update(item for item in passing if item in expectancy_ready)

    dsr_by_sleeve = _as_dict(
        multiple_testing.get("deflated_sharpe_available_by_sleeve")
    )
    dsr_ready: set[str] = set()
    for sleeve, raw in dsr_by_sleeve.items():
        row = _as_dict(raw)
        probability = _safe_float(
            row.get("deflated_sharpe_probability"),
            _safe_float(
                row.get("probability_sharpe_above_zero"),
                _safe_float(row.get("probability"), -1.0),
            ),
        )
        if probability >= minimum_dsr:
            dsr_ready.add(str(sleeve))

    pbo = _as_dict(multiple_testing.get("probability_of_backtest_overfitting"))
    pbo_value = pbo.get("pbo")
    pbo_ready = bool(
        pbo.get("available", False)
        and pbo_value is not None
        and _safe_float(pbo_value, 1.0) <= maximum_pbo
    )
    qualified = expectancy_ready.intersection(passing_sleeves, dsr_ready)
    if not pbo_ready:
        qualified.clear()
    for sleeve in expectancy_ready:
        if sleeve not in passing_sleeves:
            reasons.setdefault(sleeve, []).append("fdr_correction_pending")
        if sleeve not in dsr_ready:
            reasons.setdefault(sleeve, []).append("deflated_sharpe_pending")
        if not pbo_ready:
            reasons.setdefault(sleeve, []).append("pbo_evidence_pending")
    return qualified, reasons


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    policy_path = config_path or root / "config" / DEFAULT_CONFIG_PATH.name
    policy = load_json(policy_path)
    sources = {
        name: load_json(root / relative) for name, relative in SOURCE_PATHS.items()
    }
    receipts = {
        name: _receipt(root / relative) for name, relative in SOURCE_PATHS.items()
    }
    candidate = sources["candidate"]
    profitability = sources["profitability"]
    paper = sources["paper_performance"]
    execution = sources["execution_calibration"]
    multiple_testing = sources["multiple_testing"]
    training = sources["training_quality"]
    labels = sources["label_materialization"]
    regime = sources["regime"]
    bot_profitability = sources["bot_profitability"]

    binding = _as_dict(profitability.get("candidate_binding"))
    measurement = _as_dict(profitability.get("measurement"))
    economic_grades = _as_dict(profitability.get("grades"))
    candidate_id = str(
        candidate.get("candidate_id") or binding.get("candidate_id") or ""
    )
    profitability_candidate_id = str(binding.get("candidate_id") or "")
    candidate_identity_matches = bool(
        candidate_id
        and profitability_candidate_id
        and profitability_candidate_id == candidate_id
    )
    candidate_identity_complete = bool(
        binding.get("identity_complete", False) and candidate_identity_matches
    )
    candidate_identity_consistent = bool(
        binding.get("identity_consistent", False) and candidate_identity_matches
    )
    candidate_binding_blockers = list(binding.get("missing_required_bindings") or [])
    candidate_binding_blockers.extend(binding.get("mismatch_sources") or [])
    if candidate_id and profitability_candidate_id and not candidate_identity_matches:
        candidate_binding_blockers.append("profitability_assessment_candidate_mismatch")
    elif candidate_id and not profitability_candidate_id:
        candidate_binding_blockers.append("profitability_assessment_candidate_missing")
    sample_count = _safe_int(measurement.get("candidate_post_cost_sample_count"), 0)
    series = _as_dict(paper.get("candidate_post_cost_daily_series"))
    candidate_days = _candidate_series_days(series)
    symbol_count = _candidate_symbols(paper)
    stats_policy = _as_dict(policy.get("statistical_validation"))

    qualified_sleeves, sleeve_statistical_reasons = _statistically_qualified_sleeves(
        paper, multiple_testing, policy
    )
    cross_policy = _as_dict(policy.get("cross_sleeve_alpha"))
    cross_sleeve = build_cross_sleeve_alpha_map(
        series,
        statistically_qualified_sleeves=qualified_sleeves,
        minimum_profitable_sleeves=_safe_int(
            cross_policy.get("minimum_profitable_sleeves"), 4
        ),
        minimum_common_days=_safe_int(cross_policy.get("minimum_common_days"), 30),
        maximum_pairwise_correlation=_safe_float(
            cross_policy.get("maximum_pairwise_correlation"), 0.5
        ),
        maximum_single_sleeve_weight=_safe_float(
            cross_policy.get("maximum_single_sleeve_weight"), 0.25
        ),
    )
    cross_sleeve["qualification_reasons_by_sleeve"] = sleeve_statistical_reasons

    label_coverage = _safe_float(labels.get("route_coverage_ratio"), 0.0)
    label_policy = _as_dict(policy.get("training_labels"))
    label_route_ready = bool(
        labels
        and label_coverage
        >= _safe_float(label_policy.get("minimum_route_coverage_ratio"), 1.0)
        and _safe_int(labels.get("misrouted_directional_infrastructure_after_count"), 0)
        == 0
        and _safe_int(labels.get("misrouted_market_signal_guards_after_count"), 0) == 0
    )
    training_ready = str(training.get("overall_status") or "").lower() in {
        "ready",
        "healthy",
        "ok",
    }

    regime_policy = _as_dict(policy.get("regime_evidence"))
    depth = _as_dict(regime.get("data_depth"))
    daily_points = _safe_int(depth.get("daily_points"), 0)
    distinct_regimes = _safe_int(depth.get("distinct_regimes"), 0)
    if distinct_regimes <= 0 and str(regime.get("regime_state") or ""):
        distinct_regimes = 1
    regime_ready = bool(
        daily_points >= _safe_int(regime_policy.get("minimum_daily_points"), 30)
        and distinct_regimes
        >= _safe_int(regime_policy.get("minimum_distinct_regimes"), 3)
        and str(regime.get("overall_status") or "").lower()
        not in {
            "thin",
            "blocked",
            "degraded",
        }
    )

    controls = [
        _control(
            "a01",
            "Candidate-bound alpha objective",
            implementation_ready=bool(policy and candidate),
            evidence_ready=candidate_identity_complete,
            measurement={
                "candidate_id": candidate_id,
                "profitability_candidate_id": profitability_candidate_id,
                "generation": _safe_int(candidate.get("generation"), 0),
                "identity_complete": candidate_identity_complete,
                "identity_consistent": candidate_identity_consistent,
                "historical_relabeling_allowed": False,
            },
            blockers=candidate_binding_blockers,
        ),
        _control(
            "a02",
            "Mature post-cost outcome accrual",
            implementation_ready=bool(
                "candidate_post_cost_daily_series" in paper
                and "accounting_views" in paper
            ),
            evidence_ready=bool(
                sample_count
                >= _safe_int(stats_policy.get("minimum_post_cost_samples"), 200)
                and candidate_days
                >= _safe_int(stats_policy.get("minimum_independent_days"), 30)
                and symbol_count >= _safe_int(stats_policy.get("minimum_symbols"), 10)
            ),
            measurement={
                "post_cost_samples": sample_count,
                "minimum_post_cost_samples": _safe_int(
                    stats_policy.get("minimum_post_cost_samples"), 200
                ),
                "independent_days": candidate_days,
                "minimum_independent_days": _safe_int(
                    stats_policy.get("minimum_independent_days"), 30
                ),
                "symbols": symbol_count,
                "minimum_symbols": _safe_int(stats_policy.get("minimum_symbols"), 10),
            },
            blockers=list(
                _as_dict(paper.get("post_cost_expectancy")).get("promotion_blockers")
                or []
            ),
        ),
        _control(
            "a03",
            "Causal alpha and execution attribution",
            implementation_ready=bool(
                (root / "core" / "causal_attribution.py").is_file()
                and (root / "core" / "alpha_evidence_contract.py").is_file()
            ),
            evidence_ready=sample_count > 0,
            measurement={
                "candidate_outcome_count": sample_count,
                "decomposition_stages": [
                    "common_context",
                    "sleeve_residual",
                    "portfolio_overlay",
                    "execution_cost",
                    "realized_net",
                ],
            },
            blockers=[] if sample_count > 0 else ["candidate_outcomes_pending"],
        ),
        _control(
            "a04",
            "Multiple-testing and overfit correction",
            implementation_ready=bool(
                multiple_testing
                and _as_dict(multiple_testing.get("actual_statistical_correction")).get(
                    "method"
                )
                == "benjamini_hochberg_fdr"
            ),
            evidence_ready=bool(
                multiple_testing.get("statistical_evidence_ready", False)
            ),
            measurement={
                "family_size": _safe_int(multiple_testing.get("family_size"), 0),
                "corrected_hypothesis_count": _safe_int(
                    _as_dict(multiple_testing.get("actual_statistical_correction")).get(
                        "hypothesis_count"
                    ),
                    0,
                ),
                "qualified_sleeve_count": len(qualified_sleeves),
                "pbo": _as_dict(
                    multiple_testing.get("probability_of_backtest_overfitting")
                ).get("pbo"),
            },
            blockers=list(multiple_testing.get("statistical_evidence_blockers") or []),
        ),
        _control(
            "a05",
            "Sleeve-specific labels and horizons",
            implementation_ready=label_route_ready,
            evidence_ready=bool(label_route_ready and training_ready),
            measurement={
                "total_bots": _safe_int(labels.get("total_bot_count"), 0),
                "routed_bots": _safe_int(labels.get("routed_bot_count"), 0),
                "route_coverage_ratio": label_coverage,
                "training_quality_status": str(
                    training.get("overall_status") or "missing"
                ),
            },
            blockers=(
                []
                if label_route_ready and training_ready
                else ["training_quality_or_label_maturity_pending"]
            ),
        ),
        _control(
            "a06",
            "Regime-conditioned alpha depth",
            implementation_ready=bool(regime and regime_policy),
            evidence_ready=regime_ready,
            measurement={
                "status": str(regime.get("overall_status") or "missing"),
                "daily_points": daily_points,
                "minimum_daily_points": _safe_int(
                    regime_policy.get("minimum_daily_points"), 30
                ),
                "distinct_regimes": distinct_regimes,
                "minimum_distinct_regimes": _safe_int(
                    regime_policy.get("minimum_distinct_regimes"), 3
                ),
            },
            blockers=[] if regime_ready else ["regime_depth_pending"],
        ),
        _control(
            "a07",
            "Conservative net-edge gating",
            implementation_ready=bool(_as_dict(policy.get("net_edge"))),
            evidence_ready=bool(
                _as_dict(paper.get("post_cost_expectancy")).get("available", False)
                and execution.get("independent_evidence_ready", False)
            ),
            measurement={
                "net_edge_policy": "gross_or_explicit_post_cost_minus_observed_costs_and_uncertainty",
                "unknown_cost_defaults_allowed": False,
                "model_score_to_edge_conversion_allowed": False,
                "independent_execution_samples": _safe_int(
                    execution.get("independent_samples"), 0
                ),
            },
            blockers=list(execution.get("failed_checks") or [])
            + (
                []
                if _as_dict(paper.get("post_cost_expectancy")).get("available", False)
                else ["post_cost_edge_samples_pending"]
            ),
        ),
        _control(
            "a08",
            "Champion-challenger retirement loop",
            implementation_ready=bool(
                bot_profitability.get("control_grade") == "A+"
                and not bot_profitability.get("automatic_allocation_allowed", True)
            ),
            evidence_ready=bool(
                _safe_int(bot_profitability.get("ranked_bot_count"), 0) > 0
                and _safe_int(bot_profitability.get("persistent_bot_count"), 0) > 0
            ),
            measurement={
                "candidate_observed_bots": _safe_int(
                    bot_profitability.get("candidate_observed_bot_count"), 0
                ),
                "ranked_bots": _safe_int(bot_profitability.get("ranked_bot_count"), 0),
                "persistent_bots": _safe_int(
                    bot_profitability.get("persistent_bot_count"), 0
                ),
                "automatic_replacement_allowed": False,
            },
            blockers=list(bot_profitability.get("evidence_debt") or []),
        ),
        _control(
            "a09",
            "Cross-sleeve residual-alpha allocation",
            implementation_ready=bool(cross_policy),
            evidence_ready=bool(cross_sleeve.get("evidence_ready", False)),
            measurement={
                "qualified_sleeves": len(qualified_sleeves),
                "independently_positive_sleeves": _safe_int(
                    cross_sleeve.get("independently_positive_sleeve_count"), 0
                ),
                "selected_sleeves": _safe_int(
                    cross_sleeve.get("selected_sleeve_count"), 0
                ),
                "cash_weight": cross_sleeve.get("cash_weight"),
                "shared_trade_logic_allowed": False,
            },
            blockers=list(cross_sleeve.get("blockers") or []),
        ),
        _control(
            "a10",
            "Alpha expansion freeze",
            implementation_ready=bool(
                _as_dict(policy.get("strategy_expansion_freeze")).get("enabled", False)
                and (
                    root / "scripts" / "ops" / "strategy_generation_control.py"
                ).is_file()
            ),
            evidence_ready=bool(
                economic_grades.get("economic_evidence_ready", False)
                and multiple_testing.get("statistical_evidence_ready", False)
                and regime_ready
                and cross_sleeve.get("evidence_ready", False)
            ),
            measurement={
                "freeze_enabled": bool(
                    _as_dict(policy.get("strategy_expansion_freeze")).get(
                        "enabled", False
                    )
                ),
                "existing_collection_continues": True,
                "new_offspring_blocked_until_evidence_ready": True,
            },
            blockers=(
                []
                if economic_grades.get("economic_evidence_ready", False)
                else ["economic_evidence_grade_a_plus_pending"]
            ),
        ),
    ]
    implementation_ready = sum(
        1 for row in controls if bool(row.get("implementation_ready", False))
    )
    evidence_ready = sum(
        1 for row in controls if bool(row.get("evidence_ready", False))
    )
    implementation_score = round(100.0 * implementation_ready / len(controls), 3)
    evidence_score = round(100.0 * evidence_ready / len(controls), 3)
    implementation_complete = implementation_ready == len(controls)
    evidence_complete = evidence_ready == len(controls)
    overall_status = (
        "alpha_evidence_ready"
        if evidence_complete
        else (
            "collecting_candidate_alpha_evidence"
            if implementation_complete
            else "implementation_blocked"
        )
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "policy_id": str(policy.get("policy_id") or ""),
        "ok": implementation_complete,
        "overall_status": overall_status,
        "candidate_binding": {
            "candidate_id": candidate_id,
            "generation": _safe_int(candidate.get("generation"), 0),
            "accepted_at_utc": str(candidate.get("accepted_at_utc") or ""),
            "profitability_candidate_id": profitability_candidate_id,
            "identity_complete": candidate_identity_complete,
            "identity_consistent": candidate_identity_consistent,
            "blockers": candidate_binding_blockers,
            "historical_evidence_relabeling_allowed": False,
        },
        "grades": {
            "implementation_grade": _grade(
                implementation_score, complete=implementation_complete
            ),
            "implementation_score": implementation_score,
            "implementation_ready_controls": implementation_ready,
            "implementation_control_count": len(controls),
            "economic_evidence_grade": _grade(
                evidence_score, complete=evidence_complete
            ),
            "economic_evidence_score": evidence_score,
            "economic_evidence_ready_controls": evidence_ready,
            "economic_evidence_control_count": len(controls),
            "economic_evidence_ready": evidence_complete,
            "grade_separation_policy": "implementation completeness never upgrades economic alpha evidence",
        },
        "controls": controls,
        "cross_sleeve_alpha": cross_sleeve,
        "strategy_expansion_freeze": {
            "active": not evidence_complete,
            "reason": "new strategy volume cannot substitute for candidate-bound alpha evidence",
            "existing_collection_and_paper_observation_continue": True,
            "current_strategy_generation_status": str(
                sources["strategy_generation"].get("overall_status") or "missing"
            ),
        },
        "authority_contract": {
            **_as_dict(policy.get("authority_contract")),
            "automatic_allocation": False,
            "automatic_promotion": False,
            "live_execution_authority": False,
        },
        "soak_contract": _as_dict(policy.get("soak_contract")),
        "source_receipts": receipts,
        "recommended_actions": [
            "continue candidate-bound paper collection until mature post-cost outcomes exist",
            "share regime, liquidity, macro, factor, and cost context across sleeves without sharing trade ownership",
            "allocate only after positive residual LCB, FDR, DSR, PBO, and correlation gates pass",
            "keep strategy offspring frozen while current candidate alpha evidence is incomplete",
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    grades = _as_dict(payload.get("grades"))
    lines = [
        "# Alpha Generation Control",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        f"Candidate: `{_as_dict(payload.get('candidate_binding')).get('candidate_id', '')}`",
        f"Status: `{payload.get('overall_status', '')}`",
        f"Implementation: `{grades.get('implementation_grade', '')}` ({grades.get('implementation_ready_controls', 0)}/{grades.get('implementation_control_count', 0)})",
        f"Economic evidence: `{grades.get('economic_evidence_grade', '')}` ({grades.get('economic_evidence_ready_controls', 0)}/{grades.get('economic_evidence_control_count', 0)})",
        "",
        "## Controls",
        "",
    ]
    for row in _as_list(payload.get("controls")):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('control_id', '')}` {row.get('title', '')}: implementation `{row.get('implementation_status', '')}`, evidence `{row.get('evidence_status', '')}`"
        )
    cross = _as_dict(payload.get("cross_sleeve_alpha"))
    lines.extend(
        [
            "",
            "## Cross-Sleeve Alpha",
            "",
            f"- Status: `{cross.get('status', '')}`",
            f"- Selected sleeves: `{', '.join(cross.get('selected_sleeves') or []) or 'none'}`",
            f"- Cash weight: `{cross.get('cash_weight', 1.0)}`",
            "- Shared inputs are context only; a single evidence-qualified owner controls each symbol/horizon/alpha-family exposure.",
            "- This report has no allocation, promotion, sizing, order, or live authority.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Candidate-bound alpha evidence and cross-sleeve residual allocation control."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default="")
    parser.add_argument("--out-file", default="")
    parser.add_argument("--markdown-file", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else root / "config" / DEFAULT_CONFIG_PATH.name
    )
    out_path = (
        Path(args.out_file).expanduser().resolve()
        if args.out_file
        else root / "governance" / "health" / DEFAULT_OUT_PATH.name
    )
    markdown_path = (
        Path(args.markdown_file).expanduser().resolve()
        if args.markdown_file
        else root / "exports" / "reports" / "operator" / DEFAULT_MARKDOWN_PATH.name
    )
    payload = build_payload(root, config_path=config_path)
    write_payload(out_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        grades = _as_dict(payload.get("grades"))
        print(
            "alpha_generation_control "
            f"status={payload.get('overall_status')} "
            f"implementation={grades.get('implementation_grade')} "
            f"economic={grades.get('economic_evidence_grade')} "
            f"candidate={_as_dict(payload.get('candidate_binding')).get('candidate_id')}"
        )
    return 0 if bool(payload.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
