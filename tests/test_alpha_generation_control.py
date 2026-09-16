from __future__ import annotations

import json
from pathlib import Path

from scripts.ops.alpha_generation_control import build_payload
from scripts.ops.strategy_generation_control import _alpha_expansion_gate


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _series() -> dict[str, list[dict]]:
    sleeves = ["a", "b", "c", "d", "e"]
    result: dict[str, list[dict]] = {sleeve: [] for sleeve in sleeves}
    for day_index in range(30):
        leader = sleeves[day_index % len(sleeves)]
        for sleeve in sleeves:
            result[sleeve].append(
                {
                    "day_utc": f"202609{day_index + 1:02d}",
                    "sample_count": 8,
                    "mean_post_cost_return_bps": 20.0 if sleeve == leader else 5.0,
                }
            )
    return result


def _complete_root(tmp_path: Path) -> Path:
    policy = {
        "policy_id": "test_alpha",
        "net_edge": {
            "required_cost_components": [
                "half_spread",
                "fees",
                "slippage",
                "market_impact",
            ]
        },
        "statistical_validation": {
            "minimum_post_cost_samples": 200,
            "minimum_independent_days": 30,
            "minimum_symbols": 10,
            "minimum_deflated_sharpe_probability": 0.95,
            "maximum_probability_of_backtest_overfitting": 0.2,
        },
        "training_labels": {"minimum_route_coverage_ratio": 1.0},
        "regime_evidence": {"minimum_daily_points": 30, "minimum_distinct_regimes": 3},
        "champion_challenger": {},
        "cross_sleeve_alpha": {
            "minimum_profitable_sleeves": 4,
            "minimum_common_days": 30,
            "maximum_pairwise_correlation": 0.5,
            "maximum_single_sleeve_weight": 0.25,
        },
        "strategy_expansion_freeze": {"enabled": True},
        "authority_contract": {"live_execution_authority": False},
        "soak_contract": {"preserve_cumulative_runtime_history": True},
    }
    _write(tmp_path / "config" / "alpha_generation_control_v1.json", policy)
    _write(
        tmp_path / "governance" / "runtime" / "production_candidate_state.json",
        {
            "candidate_id": "pc-complete",
            "generation": 4,
            "accepted_at_utc": "2026-09-01T00:00:00+00:00",
        },
    )
    _write(
        tmp_path
        / "governance"
        / "health"
        / "profitability_self_assessment_latest.json",
        {
            "candidate_binding": {
                "candidate_id": "pc-complete",
                "identity_complete": True,
                "identity_consistent": True,
            },
            "measurement": {"candidate_post_cost_sample_count": 200},
            "grades": {
                "economic_evidence_grade": "A+",
                "economic_evidence_ready": True,
            },
        },
    )
    sleeve_latest = [
        {
            "profile": sleeve,
            "post_cost_expectancy": {
                "promotion_evidence_sufficient": True,
                "positive_clustered_lower_confidence_bound_95": True,
            },
        }
        for sleeve in _series()
    ]
    _write(
        tmp_path / "governance" / "health" / "paper_performance_latest.json",
        {
            "candidate_post_cost_daily_series": _series(),
            "accounting_views": {},
            "sleeve_latest": sleeve_latest,
            "post_cost_expectancy": {
                "available": True,
                "robust_statistics": {"unique_symbol_count": 10},
            },
        },
    )
    _write(
        tmp_path / "governance" / "health" / "paper_execution_calibration_latest.json",
        {"independent_evidence_ready": True, "independent_samples": 100},
    )
    correction_rows = [{"sleeve": sleeve, "passes": True} for sleeve in _series()]
    _write(
        tmp_path / "governance" / "research" / "multiple_testing_guard_latest.json",
        {
            "statistical_evidence_ready": True,
            "family_size": 5,
            "actual_statistical_correction": {
                "method": "benjamini_hochberg_fdr",
                "hypothesis_count": 5,
                "rows": correction_rows,
                "passing_hypotheses": list(_series()),
            },
            "deflated_sharpe_available_by_sleeve": {
                sleeve: {"deflated_sharpe_probability": 0.99} for sleeve in _series()
            },
            "probability_of_backtest_overfitting": {"available": True, "pbo": 0.1},
        },
    )
    _write(
        tmp_path / "governance" / "health" / "training_quality_control_latest.json",
        {"overall_status": "ready"},
    )
    _write(
        tmp_path
        / "governance"
        / "training_labeling_intelligence"
        / "all_bot_label_materialization_latest.json",
        {
            "total_bot_count": 5,
            "routed_bot_count": 5,
            "route_coverage_ratio": 1.0,
            "misrouted_directional_infrastructure_after_count": 0,
            "misrouted_market_signal_guards_after_count": 0,
        },
    )
    _write(
        tmp_path / "governance" / "health" / "regime_control_plane_latest.json",
        {
            "overall_status": "ready",
            "regime_state": "risk_on",
            "data_depth": {"daily_points": 30, "distinct_regimes": 3},
        },
    )
    _write(
        tmp_path
        / "governance"
        / "health"
        / "bot_profitability_scalability_latest.json",
        {
            "control_grade": "A+",
            "automatic_allocation_allowed": False,
            "candidate_observed_bot_count": 5,
            "ranked_bot_count": 5,
            "persistent_bot_count": 5,
            "evidence_debt": [],
        },
    )
    _write(
        tmp_path / "governance" / "health" / "strategy_generation_control_latest.json",
        {"overall_status": "bounded_idle"},
    )
    (tmp_path / "core").mkdir(parents=True, exist_ok=True)
    (tmp_path / "core" / "causal_attribution.py").touch()
    (tmp_path / "core" / "alpha_evidence_contract.py").touch()
    (tmp_path / "scripts" / "ops").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "ops" / "strategy_generation_control.py").touch()
    return tmp_path


def test_all_ten_alpha_controls_can_clear_only_with_organic_evidence(
    tmp_path: Path,
) -> None:
    root = _complete_root(tmp_path)

    payload = build_payload(
        root, config_path=root / "config" / "alpha_generation_control_v1.json"
    )

    assert payload["grades"]["implementation_grade"] == "A+"
    assert payload["grades"]["economic_evidence_grade"] == "A+"
    assert payload["grades"]["economic_evidence_ready_controls"] == 10
    assert payload["cross_sleeve_alpha"]["evidence_ready"] is True
    assert payload["authority_contract"]["automatic_allocation"] is False
    assert payload["authority_contract"]["live_execution_authority"] is False


def test_candidate_rollover_fails_closed_until_profitability_sources_refresh(
    tmp_path: Path,
) -> None:
    root = _complete_root(tmp_path)
    _write(
        root / "governance" / "runtime" / "production_candidate_state.json",
        {
            "candidate_id": "pc-next",
            "generation": 5,
            "accepted_at_utc": "2026-09-02T00:00:00+00:00",
        },
    )

    payload = build_payload(
        root, config_path=root / "config" / "alpha_generation_control_v1.json"
    )
    binding = payload["candidate_binding"]
    candidate_control = next(
        row for row in payload["controls"] if row["control_id"] == "a01"
    )

    assert binding["candidate_id"] == "pc-next"
    assert binding["profitability_candidate_id"] == "pc-complete"
    assert binding["identity_complete"] is False
    assert binding["identity_consistent"] is False
    assert "profitability_assessment_candidate_mismatch" in binding["blockers"]
    assert candidate_control["evidence_ready"] is False


def test_strategy_generation_alpha_gate_blocks_weak_evidence(tmp_path: Path) -> None:
    config = {
        "alpha_expansion_gate": {
            "enabled": True,
            "artifact": "governance/health/alpha_generation_control_latest.json",
            "fallback_artifact": "governance/health/profitability_self_assessment_latest.json",
            "require_implementation_grade": "A+",
            "require_economic_evidence_grade": "A+",
            "require_statistical_evidence_ready": True,
            "require_regime_evidence_ready": True,
            "require_cross_sleeve_alpha_ready": True,
        }
    }
    _write(
        tmp_path / "governance" / "health" / "alpha_generation_control_latest.json",
        {
            "ok": True,
            "grades": {
                "implementation_grade": "A+",
                "economic_evidence_grade": "F",
            },
            "controls": [
                {"control_id": "a04", "evidence_ready": False},
                {"control_id": "a06", "evidence_ready": False},
            ],
            "cross_sleeve_alpha": {"evidence_ready": False},
        },
    )

    gate = _alpha_expansion_gate(tmp_path, config)

    assert gate["ready"] is False
    assert "alpha_economic_evidence_grade_pending" in gate["blockers"]
    assert "alpha_statistical_evidence_pending" in gate["blockers"]
    assert "alpha_regime_evidence_pending" in gate["blockers"]
    assert "cross_sleeve_residual_alpha_map_pending" in gate["blockers"]
    assert gate["existing_offspring_collection_continues"] is True
