import scripts.run_shadow_training_loop as loop


def _master_output(vote: float, *, action: str = "BUY", score: float = 0.66, threshold: float = 0.58) -> dict:
    return {
        "action": action,
        "score": score,
        "threshold": threshold,
        "vote": vote,
    }


def test_grand_master_vote_holds_when_infra_stress_and_deployability_are_poor() -> None:
    action, score, threshold, reasons, meta = loop._grand_master_vote(
        {
            "trend": _master_output(0.34),
            "mean_revert": _master_output(0.18, score=0.60),
            "shock": _master_output(0.12, score=0.58),
        },
        {"trend": 0.45, "mean_revert": 0.30, "shock": 0.25},
        {
            "infra_vote": -0.25,
            "infra_risk_throttle_norm": 0.94,
            "infra_veto_active": 1.0,
            "infra_confidence_calibrator_scale_norm": 0.22,
            "options_specialist_vote": 0.05,
            "futures_specialist_vote": -0.08,
            "flow_direction_signed": -0.10,
            "flow_conviction_norm": 0.18,
            "flow_stress_norm": 0.86,
            "lead_lag_signal_signed": -0.18,
            "lead_lag_confidence_norm": 0.20,
            "lead_lag_break_norm": 0.88,
            "market_micro_order_flow_imbalance_norm": 0.34,
            "execution_fitness_norm": 0.16,
            "market_micro_tradeability_score_norm": 0.18,
            "cross_bot_conflict_norm": 0.84,
        },
    )

    assert action == "HOLD"
    assert score >= 0.0
    assert threshold > 0.0
    assert meta["deployability"] < 0.35
    assert meta["master_disagreement"] >= 0.0
    assert any("deployability=" in reason for reason in reasons)


def test_grand_master_vote_uses_specialist_alignment_when_conditions_support_deployment() -> None:
    action, score, threshold, reasons, meta = loop._grand_master_vote(
        {
            "trend": _master_output(0.66, score=0.78),
            "mean_revert": _master_output(0.18, action="HOLD", score=0.56),
            "shock": _master_output(0.28, score=0.63),
        },
        {"trend": 0.50, "mean_revert": 0.15, "shock": 0.35},
        {
            "infra_vote": 0.18,
            "infra_risk_throttle_norm": 0.18,
            "infra_veto_active": 0.0,
            "infra_confidence_calibrator_scale_norm": 0.82,
            "options_specialist_vote": 0.62,
            "futures_specialist_vote": 0.58,
            "flow_direction_signed": 0.52,
            "flow_conviction_norm": 0.74,
            "flow_stress_norm": 0.16,
            "lead_lag_signal_signed": 0.48,
            "lead_lag_confidence_norm": 0.78,
            "lead_lag_break_norm": 0.12,
            "market_micro_order_flow_imbalance_norm": 0.78,
            "execution_fitness_norm": 0.84,
            "market_micro_tradeability_score_norm": 0.82,
            "cross_bot_conflict_norm": 0.10,
        },
    )

    assert action == "BUY"
    assert score > 0.5
    assert threshold > 0.0
    assert meta["deployability"] > 0.45
    assert meta["specialist_consensus"] > 0.40
    assert meta["directional_alignment"] > 0.30
    assert any("specialist_consensus=" in reason for reason in reasons)


def test_grand_master_vote_integrates_quant_strategy_conviction_router() -> None:
    master_outputs = {
        "trend": _master_output(0.34, score=0.67),
        "mean_revert": _master_output(0.12, action="HOLD", score=0.56),
        "shock": _master_output(0.08, score=0.56),
    }
    weights = {"trend": 0.52, "mean_revert": 0.18, "shock": 0.30}
    base_features = {
        "infra_vote": 0.08,
        "infra_risk_throttle_norm": 0.10,
        "infra_veto_active": 0.0,
        "infra_confidence_calibrator_scale_norm": 0.78,
        "options_specialist_vote": 0.25,
        "futures_specialist_vote": 0.22,
        "flow_direction_signed": 0.50,
        "flow_conviction_norm": 0.68,
        "flow_stress_norm": 0.10,
        "lead_lag_signal_signed": 0.45,
        "lead_lag_confidence_norm": 0.72,
        "lead_lag_break_norm": 0.08,
        "market_micro_order_flow_imbalance_norm": 0.73,
        "execution_fitness_norm": 0.75,
        "market_micro_tradeability_score_norm": 0.77,
        "cross_bot_conflict_norm": 0.05,
        "quant_model_data_confidence_norm": 0.70,
        "quant_model_resource_pressure_norm": 0.08,
        "quant_cvar_tail_risk_norm": 0.12,
        "quant_copula_dependency_norm": 0.10,
    }

    _base_action, base_score, _base_threshold, _base_reasons, base_meta = loop._grand_master_vote(
        master_outputs,
        weights,
        base_features,
    )
    action, score, threshold, reasons, meta = loop._grand_master_vote(
        master_outputs,
        weights,
        {
            **base_features,
            "quant_strategy_selection_confidence_norm": 0.78,
            "quant_strategy_portfolio_fit_norm": 0.80,
            "quant_strategy_execution_alignment_norm": 0.82,
            "quant_strategy_risk_adjusted_conviction_norm": 0.74,
            "quant_strategy_allocation_bias_norm": 0.76,
        },
    )

    assert action in {"BUY", "HOLD"}
    assert score > base_score
    assert threshold > 0.0
    assert meta["quant_strategy_conviction"] == 0.74
    assert meta["quant_strategy_vote"] > 0.20
    assert meta["directional_trigger"] <= base_meta["directional_trigger"]
    assert any("quant_strategy_conviction=" in reason for reason in reasons)
    assert any("quant_strategy_vote=" in reason for reason in reasons)


def test_master_vote_variant_uses_paper_profitability_awareness() -> None:
    rows = [
        {
            "bot_id": "brain_refinery_v10",
            "action": "BUY",
            "direction": 1.0,
            "weight": 0.7,
            "eligible_for_master_vote": True,
        },
        {
            "bot_id": "brain_refinery_v50",
            "action": "BUY",
            "direction": 1.0,
            "weight": 0.3,
            "eligible_for_master_vote": True,
        },
    ]
    base_action, base_score, _base_threshold, _base_reasons, base_meta = loop._master_vote_variant(
        "trend",
        rows,
        {
            "mom_5m": 0.004,
            "pct_from_close": 0.006,
            "market_micro_tradeability_score_norm": 0.75,
            "execution_fitness_norm": 0.74,
        },
    )
    action, score, _threshold, reasons, meta = loop._master_vote_variant(
        "trend",
        rows,
        {
            "mom_5m": 0.004,
            "pct_from_close": 0.006,
            "market_micro_tradeability_score_norm": 0.75,
            "execution_fitness_norm": 0.74,
            "paper_profitability_master_awareness_active_norm": 1.0,
            "paper_profitability_master_profit_score_norm": 0.24,
            "paper_profitability_master_drag_norm": 0.82,
            "paper_profitability_master_risk_norm": 0.78,
            "paper_profitability_master_size_multiplier_norm": 0.18,
        },
    )

    assert base_action in {"BUY", "HOLD"}
    assert action in {"BUY", "HOLD"}
    assert score < base_score
    assert abs(meta["vote"]) < abs(base_meta["vote"])
    assert meta["paper_profitability_master_awareness"] == 1.0
    assert any("paper_profitability_master_risk=" in reason for reason in reasons)


def test_grand_master_vote_uses_paper_profitability_risk() -> None:
    master_outputs = {
        "trend": _master_output(0.44, score=0.70),
        "mean_revert": _master_output(0.18, action="HOLD", score=0.57),
        "shock": _master_output(0.14, score=0.56),
    }
    weights = {"trend": 0.55, "mean_revert": 0.20, "shock": 0.25}
    clean_features = {
        "infra_vote": 0.08,
        "infra_risk_throttle_norm": 0.08,
        "infra_confidence_calibrator_scale_norm": 0.80,
        "options_specialist_vote": 0.30,
        "futures_specialist_vote": 0.24,
        "flow_direction_signed": 0.34,
        "flow_conviction_norm": 0.62,
        "flow_stress_norm": 0.08,
        "lead_lag_signal_signed": 0.30,
        "lead_lag_confidence_norm": 0.70,
        "lead_lag_break_norm": 0.06,
        "market_micro_order_flow_imbalance_norm": 0.68,
        "execution_fitness_norm": 0.76,
        "market_micro_tradeability_score_norm": 0.76,
        "cross_bot_conflict_norm": 0.08,
    }
    _clean_action, clean_score, _clean_threshold, _clean_reasons, clean_meta = loop._grand_master_vote(
        master_outputs,
        weights,
        clean_features,
    )
    action, score, threshold, reasons, meta = loop._grand_master_vote(
        master_outputs,
        weights,
        {
            **clean_features,
            "paper_profitability_grandmaster_awareness_active_norm": 1.0,
            "paper_profitability_grandmaster_profit_score_norm": 0.18,
            "paper_profitability_grandmaster_drag_norm": 0.88,
            "paper_profitability_grandmaster_risk_norm": 0.86,
            "paper_profitability_grandmaster_exit_pressure_norm": 0.76,
            "paper_profitability_grandmaster_execution_discount_norm": 0.42,
            "paper_profitability_grandmaster_size_multiplier_norm": 0.12,
        },
    )

    assert action == "HOLD"
    assert score < clean_score
    assert threshold >= _clean_threshold
    assert meta["paper_profitability_grandmaster_awareness"] == 1.0
    assert meta["paper_profitability_grandmaster_risk"] == 0.86
    assert meta["deployability"] < clean_meta["deployability"]
    assert any("paper_profitability_grandmaster_risk=" in reason for reason in reasons)


def test_decision_driver_feature_snapshot_carries_market_move_explainer_inputs() -> None:
    snapshot = loop._decision_driver_feature_snapshot(
        {
            "mom_5m": -0.006,
            "flow_direction_signed": -0.52,
            "market_micro_order_flow_imbalance_norm": 0.31,
            "market_micro_tradeability_score_norm": 0.78,
            "quant_strategy_crypto_basis_edge_norm": 0.69,
            "quant_cvar_tail_risk_norm": 0.44,
            "paper_profitability_grandmaster_risk_norm": 0.62,
            "paper_profitability_master_training_weight_norm": 0.71,
        },
        {
            "quant_strategy_conviction": 0.74,
            "quant_strategy_fit": 0.80,
            "quant_strategy_execution_alignment": 0.82,
            "vote": -0.36,
            "paper_profitability_grandmaster_profit_score": 0.33,
        },
        action="SELL",
        score=0.38,
        threshold=0.58,
        return_1m=-0.004,
    )

    assert snapshot["decision_action_norm"] == -1.0
    assert snapshot["return_1m"] == -0.004
    assert snapshot["quant_strategy_crypto_basis_edge_norm"] == 0.69
    assert snapshot["quant_strategy_risk_adjusted_conviction_norm"] == 0.74
    assert snapshot["quant_strategy_portfolio_fit_norm"] == 0.80
    assert snapshot["quant_strategy_execution_alignment_norm"] == 0.82
    assert snapshot["grand_master_vote"] == -0.36
    assert snapshot["paper_profitability_grandmaster_risk_norm"] == 0.62
    assert snapshot["paper_profitability_grandmaster_profit_score_norm"] == 0.33
    assert snapshot["paper_profitability_master_training_weight_norm"] == 0.71
    assert snapshot["decision_driver_sell_pressure_norm"] > snapshot["decision_driver_buy_pressure_norm"]


def test_publish_master_plan_intent_emits_options_plan(monkeypatch) -> None:
    calls = []

    def _capture_publish(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(loop, "_publish_execution_lane_intent", _capture_publish)

    published = loop._publish_master_plan_intent(
        broker="schwab",
        symbol="SPY",
        decision={
            "action": "BUY",
            "score": 0.71,
            "threshold": 0.60,
            "reasons": ["options_edge"],
            "plan": {
                "options_style": "CALL_DEBIT_SPREAD",
                "contracts": 2,
                "legs": [
                    {"side": "BUY", "option_type": "CALL", "strike": 500.0, "expiry_days": 21, "quantity": 2},
                    {"side": "SELL", "option_type": "CALL", "strike": 510.0, "expiry_days": 21, "quantity": 2},
                ],
            },
        },
        features={"grand_master_vote": 0.52},
        gates={"market_data_ok": True},
        strategy="master_options_bot",
        layer="master_options",
        snapshot_id="snap-1",
        source_profile="live",
        shadow_domain="equities",
        plan_key="options_plan",
        intent_kind="master_options",
        extra_metadata={"asset_type": "OPTION"},
    )

    assert published is True
    assert len(calls) == 1
    assert calls[0]["intent_kind"] == "master_options"
    assert calls[0]["metadata"]["options_plan"]["contracts"] == 2
    assert calls[0]["metadata"]["allow_live_promotion"] is False


def test_publish_master_plan_intent_skips_non_trade_plan(monkeypatch) -> None:
    calls = []

    def _capture_publish(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(loop, "_publish_execution_lane_intent", _capture_publish)

    published = loop._publish_master_plan_intent(
        broker="schwab",
        symbol="SPY",
        decision={
            "action": "HOLD",
            "score": 0.50,
            "threshold": 0.60,
            "reasons": ["no_edge"],
            "plan": {
                "options_style": "NONE",
                "contracts": 0,
                "legs": [],
            },
        },
        features={"grand_master_vote": 0.0},
        gates={"market_data_ok": True},
        strategy="master_options_bot",
        layer="master_options",
        snapshot_id="snap-1",
        source_profile="live",
        shadow_domain="equities",
        plan_key="options_plan",
        intent_kind="master_options",
    )

    assert published is False
    assert calls == []


def test_collect_only_rows_do_not_vote_or_create_conflict() -> None:
    rows = [
        {
            "bot_id": "brain_refinery_v247_market_neutral_pairs_execution_bot",
            "action": "BUY",
            "direction": 1.0,
            "weight": 0.0,
            "lifecycle_state": "data_collection_only",
            "training_excluded": True,
            "eligible_for_master_vote": False,
        },
        {
            "bot_id": "brain_refinery_v4_simple",
            "action": "SELL",
            "direction": -1.0,
            "weight": 0.4,
            "eligible_for_master_vote": True,
        },
    ]

    assert loop._weighted_direction_vote(rows) == -1.0
    assert loop._cross_bot_conflict_norm(rows) == 0.0


def test_sleeve_family_aux_tracks_only_eligible_votes() -> None:
    rows = [
        {
            "bot_id": "brain_refinery_v227_day_trading_opening_range_breakout_bot",
            "slot_kind": "day_trading_signal",
            "action": "BUY",
            "direction": 1.0,
            "weight": 0.3,
            "eligible_for_master_vote": True,
        },
        {
            "bot_id": "brain_refinery_v247_market_neutral_pairs_execution_bot",
            "slot_kind": "market_neutral_signal",
            "action": "SELL",
            "direction": -1.0,
            "weight": 0.0,
            "lifecycle_state": "data_collection_only",
            "eligible_for_master_vote": False,
        },
    ]

    aux = loop._derive_sleeve_family_aux_features(rows)

    assert aux["day_trading_sleeve_active"] == 1.0
    assert aux["day_trading_sleeve_vote"] == 1.0
    assert aux["market_neutral_sleeve_active"] == 0.0
    assert aux["market_neutral_sleeve_vote"] == 0.0
    assert aux["sleeve_family_collect_only_pressure_norm"] == 0.5


def test_grand_master_label_quality_and_sleeve_pressure_gate_weak_edges() -> None:
    action, score, threshold, reasons, meta = loop._grand_master_vote(
        {
            "trend": _master_output(0.30, score=0.63),
            "mean_revert": _master_output(0.10, action="HOLD", score=0.55),
            "shock": _master_output(0.08, score=0.56),
        },
        {"trend": 0.50, "mean_revert": 0.25, "shock": 0.25},
        {
            "infra_vote": 0.02,
            "infra_risk_throttle_norm": 0.22,
            "infra_confidence_calibrator_scale_norm": 0.50,
            "options_specialist_vote": 0.18,
            "futures_specialist_vote": 0.10,
            "sleeve_family_consensus_vote": 0.25,
            "sleeve_family_collect_only_pressure_norm": 0.80,
            "label_contract_quality_norm": 0.25,
            "flow_direction_signed": 0.20,
            "flow_conviction_norm": 0.30,
            "flow_stress_norm": 0.20,
            "lead_lag_signal_signed": 0.15,
            "lead_lag_confidence_norm": 0.35,
            "lead_lag_break_norm": 0.20,
            "market_micro_order_flow_imbalance_norm": 0.58,
            "execution_fitness_norm": 0.45,
            "market_micro_tradeability_score_norm": 0.48,
            "cross_bot_conflict_norm": 0.12,
        },
    )

    assert action == "HOLD"
    assert score >= 0.0
    assert threshold > 0.0
    assert meta["label_contract_quality"] == 0.25
    assert meta["collect_only_pressure"] == 0.80
    assert any("label_contract_quality=" in reason for reason in reasons)


def test_sleeve_master_rollups_feed_dynamic_sleeves_to_grand_master() -> None:
    rows = [
        {
            "bot_id": "brain_refinery_v327_options_on_futures_defined_risk_hedge_bot",
            "sleeve_profile": "options_on_futures",
            "action": "BUY",
            "direction": 1.0,
            "weight": 0.4,
            "eligible_for_master_vote": True,
        },
        {
            "bot_id": "brain_refinery_v542_credit_derivatives_single_name_cds_proxy_bot",
            "sleeve_profile": "credit_derivatives_cdx_cds",
            "action": "SELL",
            "direction": -1.0,
            "weight": 0.2,
            "eligible_for_master_vote": True,
        },
        {
            "bot_id": "brain_refinery_v619_quantum_barrier_path_amplitude_options_bot",
            "sleeve_profile": "barrier_lookback_options",
            "action": "HOLD",
            "direction": 0.0,
            "weight": 0.0,
            "lifecycle_state": "data_collection_only",
            "eligible_for_master_vote": False,
        },
    ]

    aux = loop._derive_sleeve_family_aux_features(rows)

    assert aux["sleeve_master_rollup_enabled"] == 1.0
    assert aux["options_on_futures_sleeve_master_active"] == 1.0
    assert aux["credit_derivatives_cdx_cds_sleeve_master_active"] == 1.0
    assert aux["sleeve_master_collect_only_pressure_norm"] > 0.0

    action, _score, _threshold, reasons, meta = loop._grand_master_vote(
        {
            "trend": _master_output(0.42, score=0.70),
            "mean_revert": _master_output(0.22, action="HOLD", score=0.58),
            "shock": _master_output(0.18, score=0.60),
        },
        {"trend": 0.50, "mean_revert": 0.25, "shock": 0.25},
        {
            **aux,
            "infra_vote": 0.10,
            "infra_risk_throttle_norm": 0.10,
            "infra_veto_active": 0.0,
            "infra_confidence_calibrator_scale_norm": 0.80,
            "options_specialist_vote": 0.40,
            "futures_specialist_vote": 0.35,
            "flow_direction_signed": 0.36,
            "flow_conviction_norm": 0.66,
            "flow_stress_norm": 0.12,
            "lead_lag_signal_signed": 0.32,
            "lead_lag_confidence_norm": 0.70,
            "lead_lag_break_norm": 0.08,
            "market_micro_order_flow_imbalance_norm": 0.68,
            "execution_fitness_norm": 0.78,
            "market_micro_tradeability_score_norm": 0.74,
            "cross_bot_conflict_norm": 0.12,
        },
    )

    assert action in {"BUY", "HOLD"}
    assert meta["sleeve_master_rollup_enabled"] == 1.0
    assert "sleeve_master_consensus" in meta
    assert any("sleeve_master_consensus=" in reason for reason in reasons)
