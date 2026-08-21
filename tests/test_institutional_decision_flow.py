from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.ops.production_excellence_control import _scope_files
from scripts.run_specialized_sleeve_shadow import SLEEVE_DEFAULTS
from core.sleeve_strategy_specialization import attach_strategy_specialization
from shadow_research.institutional_decision_flow.evaluator import (
    apply_decision_flow_control,
    apply_paper_decision_flow_control,
    build_candidate_bound_quantitative_evidence,
    build_report,
    evaluate_decision,
    evaluate_execution_policy_guard,
    load_policy,
    resolve_sleeve_policy,
)
from shadow_research.institutional_decision_flow.runner import load_recent_decisions
from shadow_research.institutional_decision_flow.launchd import LABEL, build_plist


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _decision_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp_utc": "2026-08-18T16:00:00+00:00",
        "message_id": "message-1",
        "run_id": "run-1",
        "snapshot_id": "snapshot-1",
        "broker": "schwab",
        "shadow_profile": "dividend",
        "routing_lane": "schwab_equities",
        "symbol": "SCHD",
        "action": "BUY",
        "master_action": "BUY",
        "master_intent_action": "BUY",
        "master_intent_score": 0.82,
        "source_quality_score": 1.0,
        "ingestion_route": {
            "status": "ready",
            "route_state": "ready",
            "runtime_profile": "dividend",
            "requested_runtime_profile": "dividend",
            "decision_policy_family_id": "long_horizon_income",
            "ingestion_lane": "core",
            "cadence": "daily",
            "profile_id": "route_dividend",
            "average_route_score": 0.94,
            "paper_required_capability_coverage_ratio": 1.0,
            "live_required_capability_coverage_ratio": 0.8,
            "independent_failover_coverage_ratio": 0.75,
            "selected_producer_count": 4,
            "paper_decision_data_ready": True,
            "live_decision_data_ready": False,
            "artifact_age_minutes": 0.5,
            "artifact_fresh": True,
            "receipt_valid": True,
            "routing_receipt_sha256": "a" * 64,
            "route_receipt_sha256": "b" * 64,
            "route_summary_receipt_sha256": "c" * 64,
        },
        "feature_freshness": {"ok": True, "age_seconds": 0.2},
        "data_quality_features": {
            "data_quality_quote_agreement_norm": 1.0,
            "data_quality_missing_feature_ratio_norm": 0.0,
        },
        "market": {
            "last_price": 100.0,
            "spread_bps": 1.0,
            "market_data_latency_ms": 20.0,
            "market_impact_curve": {"1000": 0.4},
        },
        "market_micro_features": {
            "market_micro_tradeability_score_norm": 0.95,
            "market_micro_trend_persistence_norm": 0.80,
            "market_micro_post_event_drift_norm": 0.75,
            "market_micro_reversal_risk_norm": 0.10,
        },
        "grand_master_meta": {
            "specialist_consensus": 0.80,
            "sleeve_master_consensus": 0.75,
            "directional_alignment": 0.70,
            "master_disagreement": 0.10,
            "quant_strategy_fit": 0.85,
            "quant_data_confidence": 1.0,
        },
        "allocation_confidence": {
            "allocation_confidence_norm": 0.90,
            "allocation_conflict_norm": 0.10,
            "portfolio_overlap_pressure_norm": 0.10,
        },
        "execution_guard": {"ok": True},
        "execution_sim": {
            "slippage_bps": 0.5,
            "impact_bps": 0.2,
            "fee_bps": 0.1,
        },
        "portfolio": {"lane_budget_mult": 1.0},
        "portfolio_risk_engine": {"blocked": False},
        "long_term_turnover_policy": {"blocked": False},
        "circuit_breakers": {
            "kill_switch_active": False,
            "vol_shock_pause_active": False,
            "liquidity_pause_active": False,
            "symbol_circuit_active": False,
            "lane_kill_switch_active": False,
        },
        "broker_truth_reconcile": {"ok": True},
        "position_context": {
            "truth_available": True,
            "current_quantity": 0.0,
            "short_permission_confirmed": True,
            "linked_leg_truth_ready": True,
            "defined_risk_structure_ready": True,
        },
        "quantitative_evidence": {
            "selection_bias_control": 0.90,
            "independent_samples": 0.90,
            "uncertainty_calibration": 0.90,
            "signal_decay_fit": 0.90,
            "payoff_asymmetry": 0.90,
            "capacity_headroom": 0.90,
            "crowding_residual": 0.90,
            "tail_survival": 0.90,
            "regime_stability": 0.90,
        },
        "predicted_edge_lower_confidence_bound_bps": 40.0,
        "post_cost_samples": 100,
        "post_cost_lower_confidence_bound": 0.01,
    }
    row.update(overrides)
    return row


def test_qualified_shadow_candidate_never_receives_order_authority() -> None:
    result = evaluate_decision(_decision_row(), load_policy())

    assert result["classification"] == "qualified_shadow_candidate"
    assert result["qualified_shadow_candidate"] is True
    assert result["first_failed_stage"] == ""
    assert result["capital_scale_evidence"]["status"] == "bounded_evidence_available"
    assert len(result["decision_playbook"]["playbook_sha256"]) == 64
    assert result["decision_trace"]["mode_quality_gates"] == {
        "paper": "fully_qualified",
        "live": "decision_quality_qualified",
    }
    assert result["decision_trace"]["stage_progress"]["live"]["complete"] is True
    assert result["decision_trace"]["data_route"]["paper_ready"] is True
    assert result["decision_trace"]["data_route"]["quality_norm"] == 0.94
    assert len(result["ingestion_route"]["decision_route_receipt_sha256"]) == 64
    assert result["authority"] == {
        "changes_active_action": False,
        "changes_position_size": False,
        "paper_order_authority": False,
        "live_order_authority": False,
        "promotion_authority": False,
    }


def test_protected_hold_keeps_guard_attribution_and_fails_risk_stage() -> None:
    result = evaluate_decision(
        _decision_row(
            action="HOLD",
            decision_disposition="protected_hold",
            decision_blocking_stage="risk",
            decision_guard_categories=["risk", "portfolio"],
            decision_guard_reasons=["portfolio_risk_engine_qty_capped_to_zero"],
        ),
        load_policy(),
    )

    assert result["classification"] == "protected_hold"
    assert result["protected_hold"] is True
    assert result["active_guard_categories"] == ["risk", "portfolio"]
    assert result["active_guard_reasons"] == ["portfolio_risk_engine_qty_capped_to_zero"]
    risk_stage = next(stage for stage in result["stages"] if stage["stage_id"] == "08_non_bypassable_risk")
    assert risk_stage["passed"] is False


def test_intentional_hold_is_not_mislabeled_as_protective_blocking() -> None:
    result = evaluate_decision(
        _decision_row(
            action="HOLD",
            master_action="HOLD",
            master_intent_action="HOLD",
            master_intent_score=0.505,
            predicted_edge_lower_confidence_bound_bps=None,
        ),
        load_policy(),
    )

    assert result["classification"] == "no_edge_hold"
    assert result["protected_hold"] is False
    assert result["first_failed_stage"] == "03_signal_formation"
    assert result["decision_trace"]["blocking"]["reason_code"] == (
        "intentional_no_edge_hold"
    )
    signal_stage = next(
        stage
        for stage in result["stages"]
        if stage["stage_id"] == "03_signal_formation"
    )
    assert signal_stage["outcome"] == "block"
    assert signal_stage["required_for_live"] is True


def test_report_is_deterministic_and_changes_no_active_decisions() -> None:
    policy = load_policy()
    rows = [_decision_row(), _decision_row(message_id="message-2", symbol="VIG")]
    first = build_report(rows, policy, generated_at_utc="2026-08-18T16:10:00+00:00")
    second = build_report(reversed(rows), policy, generated_at_utc="2026-08-18T16:10:00+00:00")

    assert first["report_id"] == second["report_id"]
    assert first["soak_contract"]["active_action_change_count"] == 0
    assert first["soak_contract"]["position_size_change_count"] == 0
    assert first["soak_contract"]["order_submission_count"] == 0
    assert first["soak_contract"]["candidate_mutation_count"] == 0
    assert first["strategy_definition_coverage"]["complete_rate"] == 1.0
    assert first["quantitative_evidence_readiness"]["live_ready_rate"] == 1.0
    assert first["decision_efficiency"]["decision_horizon_counts"] == {
        "weeks_to_years": 2
    }


def test_policy_rejects_any_requested_execution_authority(tmp_path: Path) -> None:
    policy = load_policy()
    policy["authority"]["submit_paper_order"] = True
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden authority"):
        load_policy(path)


def test_policy_rejects_non_monotonic_active_control(tmp_path: Path) -> None:
    policy = load_policy()
    policy["active_paper_control"]["can_increase_quantity"] = True
    path = tmp_path / "unsafe_active.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="non-monotonic authority"):
        load_policy(path)


def test_active_control_passes_fully_qualified_paper_order_without_size_increase() -> None:
    policy = load_policy()
    evaluation = evaluate_decision(_decision_row(), policy)

    action, quantity, metadata = apply_paper_decision_flow_control(
        target_mode="paper",
        current_action="BUY",
        quantity=7.0,
        evaluation=evaluation,
        policy=policy,
    )

    assert action == "BUY"
    assert quantity == 7.0
    assert metadata["disposition"] == "qualified_passthrough"
    assert metadata["quantity_multiplier"] <= 1.0
    assert metadata["authority_contract"]["live_execution_authority"] is False


def test_active_control_caps_missing_edge_to_bounded_paper_probe() -> None:
    policy = load_policy()
    evaluation = evaluate_decision(
        _decision_row(predicted_edge_lower_confidence_bound_bps=None),
        policy,
    )

    action, quantity, metadata = apply_paper_decision_flow_control(
        target_mode="paper",
        current_action="BUY",
        quantity=10.0,
        evaluation=evaluation,
        policy=policy,
    )

    assert action == "BUY"
    assert quantity == 1.0
    assert metadata["disposition"] == "bounded_evidence_probe"
    assert metadata["quantity_reduced"] is True
    summary = metadata["operator_summary"]
    assert summary["control_outcome"] == "bounded_evidence_probe"
    assert summary["blocking_stage"] == "05_post_cost_edge"
    assert summary["edge_state"] == "missing"
    assert summary["position_transition"] == "enter_long"
    assert summary["stage_progress"]["paper"]["complete"] is True
    assert summary["stage_progress"]["live"]["complete"] is False
    assert len(summary["decision_playbook_sha256"]) == 64
    assert len(summary["summary_sha256"]) == 64
    assert summary["ingestion_route_status"] == "ready"
    assert summary["ingestion_route_quality_norm"] == 0.94
    assert summary["ingestion_paper_coverage_norm"] == 1.0
    assert summary["ingestion_live_coverage_norm"] == 0.8
    assert summary["ingestion_selected_producer_count"] == 4
    assert summary["ingestion_route_receipt_valid"] is True
    assert len(summary["ingestion_route_summary_receipt_sha256"]) == 64


def test_ingestion_route_receipt_is_bound_to_decision_identity() -> None:
    policy = load_policy()
    first = evaluate_decision(_decision_row(), policy)
    changed_route = deepcopy(_decision_row()["ingestion_route"])
    changed_route["average_route_score"] = 0.73
    second = evaluate_decision(
        _decision_row(ingestion_route=changed_route),
        policy,
    )

    assert first["evaluation_id"] != second["evaluation_id"]
    assert (
        first["ingestion_route"]["decision_route_receipt_sha256"]
        != second["ingestion_route"]["decision_route_receipt_sha256"]
    )


def test_active_control_vetoes_explicit_nonpositive_post_cost_edge() -> None:
    policy = load_policy()
    evaluation = evaluate_decision(
        _decision_row(predicted_edge_lower_confidence_bound_bps=0.0),
        policy,
    )

    action, quantity, metadata = apply_paper_decision_flow_control(
        target_mode="paper",
        current_action="BUY",
        quantity=10.0,
        evaluation=evaluation,
        policy=policy,
    )

    assert action == "HOLD"
    assert quantity == 0.0
    assert metadata["blocking_stage"] == "05_post_cost_edge"
    assert metadata["action_vetoed"] is True


def test_active_control_never_resurrects_hold_or_reverses_direction() -> None:
    policy = load_policy()
    evaluation = evaluate_decision(_decision_row(), policy)

    hold_action, hold_quantity, _ = apply_paper_decision_flow_control(
        target_mode="paper",
        current_action="HOLD",
        quantity=10.0,
        evaluation=evaluation,
        policy=policy,
    )
    reverse_action, reverse_quantity, reverse_metadata = apply_paper_decision_flow_control(
        target_mode="paper",
        current_action="SELL",
        quantity=10.0,
        evaluation=evaluation,
        policy=policy,
    )

    assert (hold_action, hold_quantity) == ("HOLD", 0.0)
    assert (reverse_action, reverse_quantity) == ("HOLD", 0.0)
    assert reverse_metadata["disposition"] == "veto_direction_mismatch"


def test_active_control_has_no_live_mode_authority() -> None:
    policy = load_policy()
    evaluation = evaluate_decision(_decision_row(), policy)

    action, quantity, metadata = apply_paper_decision_flow_control(
        target_mode="live",
        current_action="BUY",
        quantity=3.0,
        evaluation=evaluation,
        policy=policy,
    )

    assert (action, quantity) == ("BUY", 3.0)
    assert metadata["authorized_mode"] is False
    assert metadata["disposition"] == "unauthorized_mode_passthrough"


def test_sleeve_resolver_selects_distinct_versioned_policy_families() -> None:
    policy = load_policy()
    dividend_policy, dividend_receipt = resolve_sleeve_policy("dividend", policy)
    intraday_policy, intraday_receipt = resolve_sleeve_policy(
        "intraday_aggressive", policy
    )
    macro_policy, macro_receipt = resolve_sleeve_policy("fx", policy)

    assert dividend_receipt["policy_family_id"] == "long_horizon_income"
    assert intraday_receipt["policy_family_id"] == "intraday_momentum"
    assert macro_receipt["policy_family_id"] == "macro_rates_fx"
    assert dividend_receipt["resolved_policy_sha256"] != intraday_receipt[
        "resolved_policy_sha256"
    ]
    assert dividend_policy["component_weights"] != intraday_policy[
        "component_weights"
    ]
    assert intraday_policy["market_quality"]["latency_ceiling_ms"] < macro_policy[
        "market_quality"
    ]["latency_ceiling_ms"]
    assert dividend_receipt["paper_live_policy_parity"] is True
    assert dividend_receipt["strategy_definition_complete"] is True
    assert intraday_receipt["decision_horizon"] == "seconds_to_hours"
    assert macro_receipt["portfolio_role"] == "macro_carry_duration_and_currency_alpha"
    assert dividend_receipt["decision_playbook_sha256"] != intraday_receipt[
        "decision_playbook_sha256"
    ]
    assert dividend_policy["decision_playbook"]["paper_live_same_thesis"] is True
    assert dividend_policy["decision_playbook"]["stage_priority"][0]["stage_id"] == (
        "07_portfolio_fit"
    )
    assert intraday_policy["decision_playbook"]["stage_priority"][0]["stage_id"] == (
        "04_consensus_and_regime"
    )


def test_default_crypto_domain_uses_digital_asset_policy() -> None:
    policy = load_policy()
    resolved, receipt = resolve_sleeve_policy(
        "default",
        policy,
        domain="crypto",
    )

    assert receipt["policy_family_id"] == "digital_asset_basis"
    assert receipt["match_source"] == "domain_rule"
    assert resolved["resolved_sleeve_policy"] == receipt


def test_profile_strategy_override_distinguishes_dividend_capture_from_core_income() -> None:
    policy = load_policy()
    dividend_policy, dividend_receipt = resolve_sleeve_policy("dividend", policy)
    capture_policy, capture_receipt = resolve_sleeve_policy(
        "dividend_capture", policy
    )

    assert dividend_policy["strategy_definition"]["decision_horizon"] == "weeks_to_years"
    assert capture_policy["strategy_definition"]["decision_horizon"] == (
        "days_around_ex_dividend_event"
    )
    assert capture_policy["strategy_definition"]["primary_edge"] == (
        "net_dividend_after_price_drop_tax_cost_and_recovery"
    )
    assert dividend_receipt["strategy_definition_sha256"] != capture_receipt[
        "strategy_definition_sha256"
    ]
    assert dividend_receipt["strategy_variant_id"] != capture_receipt[
        "strategy_variant_id"
    ]


def test_collect_only_lifecycle_keeps_specialization_but_is_not_executable() -> None:
    _, receipt = resolve_sleeve_policy(
        "volatility",
        load_policy(),
        domain="exotic_derivatives",
        lifecycle_state="data_collection_only",
    )

    assert receipt["policy_family_id"] == "volatility_derivatives"
    assert receipt["execution_eligible"] is False
    assert receipt["lifecycle_state"] == "data_collection_only"


def test_every_specialized_sleeve_has_an_explicit_policy_family() -> None:
    policy = load_policy()
    unresolved: list[str] = []
    for profile, defaults in SLEEVE_DEFAULTS.items():
        _, receipt = resolve_sleeve_policy(
            profile,
            policy,
            domain=str(defaults.get("domain") or "equities"),
            lifecycle_state="data_collection_only",
        )
        if receipt["match_source"] == "default_fallback":
            unresolved.append(profile)
        assert receipt["execution_eligible"] is False
        assert receipt["strategy_definition_complete"] is True
        assert receipt["strategy_definition_sha256"]
        assert receipt["decision_playbook_sha256"]
        assert len(receipt["decision_stage_priority"]) == 8
        assert receipt["required_quantitative_evidence"]

    assert len(SLEEVE_DEFAULTS) >= 90
    assert unresolved == []


def test_policy_rejects_incomplete_strategy_definition(tmp_path: Path) -> None:
    policy = load_policy()
    del policy["strategy_definitions"]["balanced_directional"]["exit_style"]
    path = tmp_path / "incomplete_strategy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete strategy definition"):
        load_policy(path)


def test_policy_rejects_weak_live_quantitative_evidence_contract(
    tmp_path: Path,
) -> None:
    policy = load_policy()
    policy["active_live_control"]["require_quantitative_evidence_ready"] = False
    path = tmp_path / "weak_live_quant.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="strict receipt and qualification"):
        load_policy(path)


def test_missing_quantitative_evidence_remains_visible_without_overblocking_paper() -> None:
    policy = load_policy()
    evaluation = evaluate_decision(
        _decision_row(quantitative_evidence=None),
        policy,
    )
    action, quantity, metadata = apply_decision_flow_control(
        target_mode="paper",
        current_action="BUY",
        quantity=4.0,
        evaluation=evaluation,
        policy=policy,
    )

    assert (action, quantity) == ("BUY", 4.0)
    assert metadata["disposition"] == "qualified_passthrough"
    assert evaluation["quantitative_evidence"]["live_ready"] is False
    assert evaluation["quantitative_evidence"]["proxy_only_required_axes"]
    assert evaluation["quantitative_evidence"]["explicit_adverse"] is False


def test_candidate_bound_profitability_artifacts_build_direct_quantitative_evidence() -> None:
    candidate_id = "pc-candidate-g63"
    performance = {
        "profitability_evidence_window": {
            "candidate_id": candidate_id,
            "candidate_generation": 63,
            "candidate_cutoff_utc": "2026-08-18T20:00:00+00:00",
            "evidence_through_utc": "2026-08-18T22:00:00+00:00",
            "candidate_filter_active": True,
            "candidate_binding_required": True,
            "candidate_binding_mismatch_rows_excluded": 0,
        },
        "sleeve_latest": [
            {
                "profile": "dividend",
                "post_cost_expectancy": {
                    "robust_statistics": {
                        "available": True,
                        "effective_sample_size": 15.0,
                        "thresholds": {"minimum_effective_samples": 20.0},
                        "deflated_sharpe": {
                            "available": True,
                            "probability": 0.96,
                        },
                    },
                    "payoff_asymmetry": {
                        "available": True,
                        "positive_sample_count": 12,
                        "negative_sample_count": 8,
                        "average_win_to_average_loss_ratio": 2.0,
                    },
                },
            }
        ],
    }
    candidate_binding = {
        "candidate_id": candidate_id,
        "bound": True,
    }
    multiple_testing = {
        "timestamp_utc": "2026-08-18T21:00:00+00:00",
        "candidate_binding": candidate_binding,
        "actual_statistical_correction": {
            "rows": [{"hypothesis_id": "dividend", "q_value": 0.04}]
        },
        "probability_of_backtest_overfitting": {
            "available": True,
            "pbo": 0.10,
        },
        "deflated_sharpe_available_by_sleeve": {
            "dividend": {"available": True, "probability": 0.96}
        },
    }
    validator = {
        "timestamp_utc": "2026-08-18T21:05:00+00:00",
        "candidate_binding": candidate_binding,
        "risk_of_ruin": {
            "available": True,
            "ruin_probability": 0.01,
            "drawdown_breach_probability": 0.04,
            "day_count": 30,
        },
    }
    decay = {
        "timestamp_utc": "2026-08-18T21:10:00+00:00",
        "candidate_binding": candidate_binding,
        "edge_decay_contract": {
            "profiles": [
                {
                    "profile": "dividend",
                    "history_days": 12,
                    "mean_decay_fraction": 0.10,
                    "decayed": False,
                }
            ]
        },
    }
    challengers = {
        "timestamp_utc": "2026-08-18T21:15:00+00:00",
        "candidate_binding": candidate_binding,
        "overall_status": "collecting",
        "implemented_concept_count": 8,
        "evidence_ready_concept_count": 2,
        "supported_concept_count": 1,
        "authority_contract": {
            "changes_active_action": False,
            "changes_position_size": False,
            "submits_live_orders": False,
        },
        "decision_metadata_by_profile": {
            "dividend": {
                "policy_id": "candidate_bound_quantitative_challengers_v1",
                "candidate_id": candidate_id,
                "candidate_bound": True,
                "status": "evidence_available",
                "available_method_count": 2,
                "supported_method_count": 1,
                "method_count": 6,
                "report_receipt_sha256": "a" * 64,
                "authority": "read_only_metadata_no_decision_authority",
            }
        },
    }

    packet = build_candidate_bound_quantitative_evidence(
        "dividend",
        paper_performance=performance,
        multiple_testing=multiple_testing,
        independent_validator=validator,
        decay_monitor=decay,
        quantitative_challengers=challengers,
        expected_candidate_id=candidate_id,
    )

    assert packet["independent_samples"] == 0.75
    assert packet["payoff_asymmetry"] == 0.66666667
    assert packet["selection_bias_control"] == 0.9
    assert packet["tail_survival"] == 0.96
    assert packet["signal_decay_fit"] == 0.9
    assert packet["_bridge"]["status"] == "bound"
    assert packet["_bridge"]["direct_axes"] == [
        "independent_samples",
        "payoff_asymmetry",
        "selection_bias_control",
        "signal_decay_fit",
        "tail_survival",
    ]
    assert packet["_bridge"]["challenger_report_bound"] is True
    assert packet["_challengers"]["implemented_concept_count"] == 8
    assert packet["_challengers"]["changes_active_evidence_axes"] is False
    assert not any(packet["_challengers"]["authority_contract"].values())
    assert len(packet["_bridge"]["receipt_sha256"]) == 64

    evaluation = evaluate_decision(
        _decision_row(quantitative_evidence=packet),
        load_policy(),
    )
    assert evaluation["research_challengers"]["status"] == "evidence_available"
    assert evaluation["research_challengers"]["changes_decision_utility"] is False


def test_research_challenger_metadata_cannot_change_decision_or_size() -> None:
    policy = load_policy()
    baseline_row = _decision_row()
    baseline = evaluate_decision(baseline_row, policy)
    challenger_row = _decision_row(
        quantitative_evidence={
            **baseline_row["quantitative_evidence"],
            "_challengers": {
                "status": "evidence_available",
                "implemented_concept_count": 8,
                "available_method_count": 8,
                "supported_method_count": 8,
                "changes_active_evidence_axes": False,
                "changes_decision_utility": False,
                "changes_order_quantity": False,
                "authority_contract": {
                    "changes_active_action": False,
                    "changes_position_size": False,
                    "submits_paper_orders": False,
                    "submits_live_orders": False,
                    "grants_promotion": False,
                },
            },
        }
    )
    challenged = evaluate_decision(challenger_row, policy)

    for key in (
        "classification",
        "qualified_shadow_candidate",
        "final_action",
        "decision_quality_utility_norm",
        "raw_weighted_utility_norm",
        "uncertainty_penalty_norm",
        "components",
        "stages",
    ):
        assert challenged[key] == baseline[key]
    assert challenged["research_challengers"]["supported_method_count"] == 8
    assert not any(challenged["authority"].values())


def test_quantitative_evidence_bridge_rejects_candidate_mismatch() -> None:
    packet = build_candidate_bound_quantitative_evidence(
        "dividend",
        paper_performance={
            "profitability_evidence_window": {
                "candidate_id": "pc-old-g62",
                "candidate_cutoff_utc": "2026-08-18T20:00:00+00:00",
                "evidence_through_utc": "2026-08-18T22:00:00+00:00",
                "candidate_filter_active": True,
                "candidate_binding_required": True,
                "candidate_binding_mismatch_rows_excluded": 0,
            }
        },
        expected_candidate_id="pc-current-g63",
    )

    assert packet["_bridge"]["status"] == "unbound"
    assert "candidate_id_mismatch" in packet["_bridge"]["reasons"]
    assert not set(packet).intersection(
        {
            "selection_bias_control",
            "independent_samples",
            "payoff_asymmetry",
            "signal_decay_fit",
            "tail_survival",
        }
    )


def test_explicit_adverse_quantitative_evidence_downsizes_paper() -> None:
    policy = load_policy()
    packet = dict(_decision_row()["quantitative_evidence"])
    packet["tail_survival"] = 0.10
    evaluation = evaluate_decision(
        _decision_row(quantitative_evidence=packet),
        policy,
    )
    action, quantity, metadata = apply_decision_flow_control(
        target_mode="paper",
        current_action="BUY",
        quantity=10.0,
        evaluation=evaluation,
        policy=policy,
    )

    assert (action, quantity) == ("BUY", 2.5)
    assert metadata["disposition"] == "adverse_quantitative_evidence_downsize"
    assert "tail_survival" in evaluation["quantitative_evidence"][
        "critical_adverse_axes"
    ]


def test_live_control_fails_closed_on_proxy_only_quantitative_evidence() -> None:
    policy = load_policy()
    evaluation = evaluate_decision(
        _decision_row(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            quantitative_evidence=None,
        ),
        policy,
    )
    action, quantity, metadata = apply_decision_flow_control(
        target_mode="live",
        current_action="BUY",
        quantity=2.0,
        evaluation=evaluation,
        policy=policy,
    )

    assert (action, quantity) == ("HOLD", 0.0)
    assert metadata["disposition"] == "veto_quantitative_evidence_not_ready"
    assert "live_requires_direct_passing_quantitative_evidence" in metadata["reasons"]


def test_long_only_strategy_blocks_flat_account_sell_transition() -> None:
    evaluation = evaluate_decision(
        _decision_row(
            action="SELL",
            master_action="SELL",
            master_intent_action="SELL",
            master_intent_score=0.18,
            grand_master_meta={
                "specialist_consensus": -0.80,
                "sleeve_master_consensus": -0.75,
                "directional_alignment": -0.70,
                "master_disagreement": 0.10,
                "quant_strategy_fit": 0.85,
                "quant_data_confidence": 1.0,
            },
        ),
        load_policy(),
    )

    assert evaluation["action_semantics"]["semantic"] == "enter_short"
    assert evaluation["action_semantics"]["ready"] is False
    assert "short_entry_forbidden_by_strategy_definition" in evaluation[
        "action_semantics"
    ]["reasons"]


def test_live_control_uses_same_policy_and_requires_full_qualification() -> None:
    policy = load_policy()
    timestamp = datetime.now(timezone.utc).isoformat()
    qualified = evaluate_decision(_decision_row(timestamp_utc=timestamp), policy)
    action, quantity, metadata = apply_decision_flow_control(
        target_mode="live",
        current_action="BUY",
        quantity=5.0,
        evaluation=qualified,
        policy=policy,
    )

    assert (action, quantity) == ("BUY", 5.0)
    assert metadata["disposition"] == "qualified_passthrough"
    assert metadata["policy_receipt"] == qualified["policy_receipt"]
    assert metadata["quantity_multiplier"] <= 1.0
    assert metadata["authority_contract"]["live_execution_authority"] is False

    unqualified = evaluate_decision(
        _decision_row(
            timestamp_utc=timestamp,
            predicted_edge_lower_confidence_bound_bps=None,
        ),
        policy,
    )
    blocked_action, blocked_quantity, blocked = apply_decision_flow_control(
        target_mode="live",
        current_action="BUY",
        quantity=5.0,
        evaluation=unqualified,
        policy=policy,
    )

    assert (blocked_action, blocked_quantity) == ("HOLD", 0.0)
    assert blocked["blocking_stage"] == "05_post_cost_edge"
    assert blocked["action_vetoed"] is True


def test_execution_guard_revalidates_receipt_for_paper_and_live() -> None:
    policy = load_policy()
    evaluation = evaluate_decision(
        _decision_row(timestamp_utc=datetime.now(timezone.utc).isoformat()),
        policy,
    )
    _, quantity, control = apply_paper_decision_flow_control(
        target_mode="paper",
        current_action="BUY",
        quantity=3.0,
        evaluation=evaluation,
        policy=policy,
    )
    intent = {
        "symbol": "SCHD",
        "action": "BUY",
        "quantity": quantity,
        "strategy": "grand_master_bot",
        "metadata": {
            "layer": "grand_master",
            "source_profile": "dividend",
            "shadow_domain": "",
            "lifecycle_state": "",
            "allow_live_promotion": True,
            "institutional_decision_flow": {
                "policy_receipt": evaluation["policy_receipt"],
                "evaluation": evaluation,
                "control": control,
            },
        },
    }
    intent["metadata"] = attach_strategy_specialization(
        {
            **intent["metadata"],
            "production_candidate_id": "pc-test-g1",
        },
        profile="dividend",
        raw_strategy="grand_master_bot",
        features={"market_regime_snapshot": {"regime_state": "mixed_transition"}},
        action="BUY",
        quantity=quantity,
    )

    paper_guard = evaluate_execution_policy_guard(
        intent=intent,
        target_mode="paper",
        policy=policy,
    )
    live_guard = evaluate_execution_policy_guard(
        intent=intent,
        target_mode="live",
        policy=policy,
    )

    assert paper_guard["allow_execute"] is True
    assert live_guard["allow_execute"] is True
    assert live_guard["policy_receipt"] == paper_guard["policy_receipt"]

    disabled_live_policy = deepcopy(policy)
    disabled_live_policy["active_live_control"]["enabled"] = False
    disabled_evaluation = evaluate_decision(
        _decision_row(timestamp_utc=datetime.now(timezone.utc).isoformat()),
        disabled_live_policy,
    )
    _, disabled_quantity, disabled_control = apply_paper_decision_flow_control(
        target_mode="paper",
        current_action="BUY",
        quantity=3.0,
        evaluation=disabled_evaluation,
        policy=disabled_live_policy,
    )
    disabled_intent = deepcopy(intent)
    disabled_intent["quantity"] = disabled_quantity
    disabled_intent["metadata"]["institutional_decision_flow"] = {
        "policy_receipt": disabled_evaluation["policy_receipt"],
        "evaluation": disabled_evaluation,
        "control": disabled_control,
    }
    disabled_live_guard = evaluate_execution_policy_guard(
        intent=disabled_intent,
        target_mode="live",
        policy=disabled_live_policy,
    )
    assert disabled_live_guard["allow_execute"] is False
    assert "decision_flow_live_control_not_authorized" in disabled_live_guard[
        "reasons"
    ]

    tampered = deepcopy(intent)
    tampered["metadata"]["institutional_decision_flow"]["evaluation"][
        "classification"
    ] = "no_edge_hold"
    tampered_guard = evaluate_execution_policy_guard(
        intent=tampered,
        target_mode="live",
        policy=policy,
    )

    assert tampered_guard["allow_execute"] is False
    assert "decision_flow_evaluation_digest_mismatch" in tampered_guard["reasons"]

    tampered_receipt = deepcopy(intent)
    tampered_receipt["metadata"]["institutional_decision_flow"]["policy_receipt"][
        "strategy_definition_sha256"
    ] = "0" * 64
    tampered_receipt_guard = evaluate_execution_policy_guard(
        intent=tampered_receipt,
        target_mode="live",
        policy=policy,
    )

    assert tampered_receipt_guard["allow_execute"] is False
    assert "decision_flow_receipt_strategy_definition_sha256_mismatch" in (
        tampered_receipt_guard["reasons"]
    )

    tampered_playbook = deepcopy(intent)
    tampered_playbook["metadata"]["institutional_decision_flow"]["policy_receipt"][
        "decision_playbook_sha256"
    ] = "0" * 64
    tampered_playbook_guard = evaluate_execution_policy_guard(
        intent=tampered_playbook,
        target_mode="live",
        policy=policy,
    )

    assert tampered_playbook_guard["allow_execute"] is False
    assert "decision_flow_receipt_decision_playbook_sha256_mismatch" in (
        tampered_playbook_guard["reasons"]
    )


def test_live_execution_guard_fails_closed_without_policy_receipt() -> None:
    guard = evaluate_execution_policy_guard(
        intent={
            "symbol": "SPY",
            "action": "BUY",
            "quantity": 1.0,
            "strategy": "grand_master_bot",
            "metadata": {"source_profile": "default"},
        },
        target_mode="live",
        policy=load_policy(),
    )

    assert guard["allow_execute"] is False
    assert "decision_flow_metadata_missing" in guard["reasons"]


def test_policy_rejects_non_monotonic_live_control(tmp_path: Path) -> None:
    policy = load_policy()
    policy["active_live_control"]["can_increase_quantity"] = True
    path = tmp_path / "unsafe_live.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="non-monotonic authority"):
        load_policy(path)


def test_policy_rejects_weak_live_qualification_contract(tmp_path: Path) -> None:
    policy = load_policy()
    policy["active_live_control"]["require_qualified_candidate"] = False
    path = tmp_path / "weak_live.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="strict receipt and qualification"):
        load_policy(path)


def test_active_control_is_inserted_before_idempotency_and_paper_queue() -> None:
    source = (PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py").read_text(
        encoding="utf-8"
    )
    active_hook = source.index("institutional_decision_flow_paper_veto")
    idempotency = source.index('idempotency_key = ""', active_hook)
    queue_publish = source.index("exec_queue.enqueue(req)", idempotency)

    assert active_hook < idempotency < queue_publish
    assert 'target_mode="paper"' in source[active_hook - 1000 : idempotency]
    assert "live_execution_authority=False" in source[active_hook:idempotency]


def test_runtime_evidence_lookup_uses_unconditionally_initialized_profile() -> None:
    source = (PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py").read_text(
        encoding="utf-8"
    )

    assert 'current_profile = (_shadow_profile_name() or "default").strip().lower()' in source
    assert '"quantitative_evidence": _profile_quantitative_evidence(current_profile)' in source
    assert "_profile_quantitative_evidence(profile)" not in source
    assert "long_term_profile = current_profile" in source


def test_recent_decision_reader_is_bounded_and_deduplicated(tmp_path: Path) -> None:
    path = tmp_path / "governance" / "shadow_test" / "master_control_20260818.jsonl"
    path.parent.mkdir(parents=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = [
        _decision_row(timestamp_utc=timestamp, message_id="same", symbol="SPY"),
        _decision_row(timestamp_utc=timestamp, message_id="same", symbol="QQQ"),
        _decision_row(timestamp_utc=timestamp, message_id="unique", symbol="IWM"),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    loaded = load_recent_decisions(tmp_path, max_rows=10, tail_bytes_per_file=1024 * 1024)

    assert len(loaded) == 2
    assert {row["message_id"] for row in loaded} == {"same", "unique"}


def test_shadow_research_sources_are_outside_candidate_fingerprint_scopes() -> None:
    config = json.loads((PROJECT_ROOT / "config" / "production_excellence_v1.json").read_text(encoding="utf-8"))
    scopes = config["candidate"]["scope_globs"]

    included = {
        str(path.relative_to(PROJECT_ROOT))
        for patterns in scopes.values()
        for path in _scope_files(PROJECT_ROOT, patterns)
    }

    assert not any(path.startswith("shadow_research/") for path in included)
    assert "core/institutional_decision_flow.py" in included
    assert "config/institutional_decision_flow_v1.json" in included


def test_launchd_sidecar_is_bounded_background_read_only() -> None:
    payload = build_plist(PROJECT_ROOT, interval_seconds=300)

    assert payload["Label"] == LABEL
    assert payload["StartInterval"] == 300
    assert payload["Nice"] == 15
    assert payload["LowPriorityIO"] is True
    assert payload["EnvironmentVariables"]["INSTITUTIONAL_DECISION_FLOW_AUTHORITY"] == "shadow_read_only"
    joined = " ".join(payload["ProgramArguments"])
    assert "institutional_decision_flow.runner" in joined
    assert "chrome" not in joined.lower()


def test_livefeed_surfaces_strategy_and_quantitative_evidence_context() -> None:
    source = (PROJECT_ROOT / "scripts" / "ops" / "live_feed_tail.sh").read_text(
        encoding="utf-8"
    )

    for token in (
        "flow_strategy=",
        "flow_horizon=",
        "flow_role=",
        "flow_edge=",
        "flow_action_semantic=",
        "flow_quant_ready=",
        "flow_quant_gaps=",
        "flow_challengers=",
        "flow_challenger_supported=",
    ):
        assert token in source
