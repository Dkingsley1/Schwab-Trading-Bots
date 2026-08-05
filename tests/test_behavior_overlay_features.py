import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import scripts.build_behavior_dataset_from_decisions as behavior_ds
import scripts.run_shadow_training_loop as loop
import scripts.train_trade_behavior_bot as trainer


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_behavior_feature_schema_appends_lane_overlay_features() -> None:
    assert behavior_ds.BEHAVIOR_LANE_FEATURE_NAMES == loop._BEHAVIOR_LANE_FEATURE_NAMES
    assert behavior_ds.PAPER_CONTEXT_FEATURE_NAMES == loop._BEHAVIOR_PAPER_FEATURE_NAMES
    lane_start = behavior_ds.FEATURE_NAMES.index(behavior_ds.BEHAVIOR_LANE_FEATURE_NAMES[0])
    loop_lane_start = loop._BEHAVIOR_FEATURE_NAMES_V2.index(loop._BEHAVIOR_LANE_FEATURE_NAMES[0])
    paper_start = behavior_ds.FEATURE_NAMES.index(behavior_ds.PAPER_CONTEXT_FEATURE_NAMES[0])
    loop_paper_start = loop._BEHAVIOR_FEATURE_NAMES_V2.index(loop._BEHAVIOR_PAPER_FEATURE_NAMES[0])
    capital_start = behavior_ds.FEATURE_NAMES.index(behavior_ds.BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES[0])
    loop_capital_start = loop._BEHAVIOR_FEATURE_NAMES_V2.index(loop._BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES[0])
    assert behavior_ds.FEATURE_NAMES[lane_start : lane_start + len(behavior_ds.BEHAVIOR_LANE_FEATURE_NAMES)] == behavior_ds.BEHAVIOR_LANE_FEATURE_NAMES
    assert loop._BEHAVIOR_FEATURE_NAMES_V2[loop_lane_start : loop_lane_start + len(loop._BEHAVIOR_LANE_FEATURE_NAMES)] == loop._BEHAVIOR_LANE_FEATURE_NAMES
    assert behavior_ds.FEATURE_NAMES[paper_start : paper_start + len(behavior_ds.PAPER_CONTEXT_FEATURE_NAMES)] == behavior_ds.PAPER_CONTEXT_FEATURE_NAMES
    assert loop._BEHAVIOR_FEATURE_NAMES_V2[loop_paper_start : loop_paper_start + len(loop._BEHAVIOR_PAPER_FEATURE_NAMES)] == loop._BEHAVIOR_PAPER_FEATURE_NAMES
    assert behavior_ds.FEATURE_NAMES[capital_start : capital_start + len(behavior_ds.BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES)] == behavior_ds.BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES
    assert loop._BEHAVIOR_FEATURE_NAMES_V2[loop_capital_start : loop_capital_start + len(loop._BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES)] == loop._BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES
    assert behavior_ds.BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES == loop._BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES


def test_behavior_dataset_vector_matches_declared_feature_schema() -> None:
    features = {
        "dividend_compounding_quality_norm": 0.81,
        "dividend_capture_timing_quality_norm": 0.72,
        "dividend_payout_stress_gate_norm": 0.63,
        "dividend_growth_persistence_norm": 0.54,
        "dividend_capture_ex_date_hazard_norm": 0.45,
    }
    vector, _, _ = behavior_ds._decision_feature_vector(
        row={
            "features": features,
            "ts_utc": datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc),
            "symbol": "SPY",
            "action": "HOLD",
            "role_idx": 0.0,
            "quantity": 0.0,
        },
        gov={},
        lag_exec=(0.0, 0.0, 0.0),
        paper_snapshot={},
        lag_paper=(0.0, 0.0, 0.0),
        snapshot_context={},
        external_context={},
        external_meta={},
        event_windows=[],
    )

    assert len(vector) == len(behavior_ds.FEATURE_NAMES)
    for name, expected in features.items():
        assert vector[behavior_ds.FEATURE_NAMES.index(name)] == expected


def test_behavior_dataset_failed_build_preserves_last_valid_artifact(tmp_path: Path) -> None:
    out_path = tmp_path / "trade_learning_dataset.json"
    failure_path = tmp_path / "build_failure.json"
    previous = {"rows": 125, "feature_dim": len(behavior_ds.FEATURE_NAMES)}
    _write_json(out_path, previous)

    result = behavior_ds._publish_dataset(
        {
            "timestamp_utc": "2026-07-31T12:00:00+00:00",
            "rows": 0,
            "feature_dim": len(behavior_ds.FEATURE_NAMES),
            "label_counts": {},
            "skipped": {"low_symbol_rows": 12},
            "source": {"decision_files": 1},
        },
        out_path=out_path,
        failure_path=failure_path,
        min_output_rows=50,
    )

    assert result["published"] is False
    assert result["preserved_previous"] is True
    assert json.loads(out_path.read_text(encoding="utf-8")) == previous
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "insufficient_rows_preserved_previous_dataset"
    assert failure["previous_dataset_preserved"] is True


def test_behavior_dataset_local_route_rewrites_external_hot_inputs(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    local_root = project_root / "local_fallback_storage"
    monkeypatch.setenv("BOT_LOGS_PREFER_EXTERNAL", "0")
    monkeypatch.setenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(local_root))

    routed = behavior_ds._routed_input_pattern(
        str(project_root / "governance" / "shadow*" / "shadow_pnl_attribution_*.jsonl"),
        project_root=project_root,
    )

    assert routed == str(local_root / "governance" / "shadow*" / "shadow_pnl_attribution_*.jsonl")
    assert behavior_ds._routed_input_pattern(
        "/Volumes/BOT_LOGS/schwab_trading_bot/governance/shadow*/master_control_*.jsonl",
        project_root=project_root,
    ) == ""


def test_behavior_dataset_tail_reader_bounds_large_auxiliary_inputs(tmp_path: Path) -> None:
    path = tmp_path / "paper.jsonl"
    rows = [{"row": idx, "payload": "x" * 64} for idx in range(6)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    observed = list(behavior_ds._iter_jsonl([path], tail_bytes=220))

    assert observed
    assert observed[-1]["row"] == 5
    assert observed[0]["row"] > 0


def test_behavior_feature_schema_includes_execution_realism_and_conflict_features() -> None:
    keys = [
        "execution_fitness_norm",
        "stop_target_realism_norm",
        "symbol_cooldown_memory_norm",
        "cross_bot_conflict_norm",
    ]
    for key in keys:
        assert key in behavior_ds.FEATURE_NAMES
        assert key in loop._BEHAVIOR_FEATURE_NAMES_V2


def test_behavior_feature_schema_includes_core_sleeve_overlay_features() -> None:
    keys = [
        "core_default_dependency_norm",
        "core_conservative_quality_gate_norm",
        "core_aggressive_breakout_conviction_norm",
        "core_cross_sectional_rank_norm",
        "core_regime_specialist_blend_norm",
        "core_event_reaction_norm",
        "core_fx_macro_confirmation_norm",
        "core_futures_regime_edge_norm",
        "core_crypto_unwind_risk_norm",
        "aggressive_relative_strength_burst_norm",
    ]
    for key in keys:
        assert key in behavior_ds.BEHAVIOR_LANE_FEATURE_NAMES
        assert key in behavior_ds.FEATURE_NAMES
        assert key in loop._BEHAVIOR_LANE_FEATURE_NAMES
        assert key in loop._BEHAVIOR_FEATURE_NAMES_V2


def test_channel_decision_rows_are_canonicalized_for_behavior_dataset() -> None:
    row = {
        "timestamp_utc": "2026-05-28T12:00:00+00:00",
        "symbol": "SPY",
        "snapshot_id": "snap-123",
        "shadow_profile": "aggressive",
        "market": {"last_price": 500.0, "pct_from_close": 0.01, "spread_bps": 1.5},
        "master_action": "BUY",
        "master_score": 0.71,
        "master_outputs": {
            "trend": {
                "master_meta": {
                    "paper_profitability_master_awareness": 1.0,
                    "paper_profitability_master_profit_score": 0.72,
                    "paper_profitability_master_drag": 0.18,
                    "paper_profitability_master_risk": 0.22,
                    "paper_profitability_master_size_multiplier": 0.44,
                }
            },
            "shock": {
                "master_meta": {
                    "paper_profitability_master_awareness": 1.0,
                    "paper_profitability_master_profit_score": 0.62,
                    "paper_profitability_master_drag": 0.28,
                    "paper_profitability_master_risk": 0.32,
                    "paper_profitability_master_size_multiplier": 0.34,
                }
            },
        },
        "grand_master_meta": {
            "paper_profitability_grandmaster_awareness": 1.0,
            "paper_profitability_grandmaster_profit_score": 0.69,
            "paper_profitability_grandmaster_drag": 0.21,
            "paper_profitability_grandmaster_risk": 0.25,
            "paper_profitability_grandmaster_size_multiplier": 0.39,
            "paper_profitability_grandmaster_exit_pressure": 0.14,
            "paper_profitability_grandmaster_execution_discount": 0.08,
            "specialist_conflict": 0.17,
        },
    }

    canonical = behavior_ds._canonical_behavior_decision_row(row)

    assert canonical is not None
    assert canonical["strategy"] == "grand_master_bot"
    assert canonical["action"] == "BUY"
    assert canonical["mode"] == "aggressive"
    assert canonical["metadata"]["snapshot_id"] == "snap-123"
    assert canonical["features"]["last_price"] == 500.0
    assert canonical["features"]["paper_profitability_master_awareness_active_norm"] == 1.0
    assert round(canonical["features"]["paper_profitability_master_profit_score_norm"], 4) == 0.67
    assert canonical["features"]["paper_profitability_grandmaster_profit_score_norm"] == 0.69
    assert canonical["features"]["paper_profitability_grandmaster_conflict_cap_norm"] == 0.83


def test_load_paper_trade_context_joins_snapshot_and_symbol_history() -> None:
    since_utc = datetime.now(timezone.utc) - timedelta(hours=2)
    ts = datetime.now(timezone.utc)
    by_snapshot, by_symbol = behavior_ds._load_paper_trade_context(
        [
            {
                "timestamp_utc": ts.isoformat(),
                "symbol": "SPY",
                "action": "BUY",
                "fill_price": 100.10,
                "reference_price": 100.00,
                "mark_price": 100.40,
                "metadata": {"snapshot_id": "snap-1"},
            }
        ],
        since_utc=since_utc,
    )

    assert by_snapshot["snap-1"]["count"] == 1.0
    assert by_snapshot["snap-1"]["mean_slippage_bps"] > 0.0
    assert by_snapshot["snap-1"]["mean_return_proxy_bps"] > 0.0
    assert "SPY" in by_symbol


def test_behavior_feature_vector_v2_accepts_paper_context_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "SPY",
        "BUY",
        {
            "pct_from_close": 0.003,
            "mom_5m": 0.001,
            "vol_30m": 0.004,
            "paper_snapshot_trade_count_norm": 0.50,
            "paper_snapshot_slippage_bps_norm": 0.20,
            "paper_snapshot_return_proxy_signed_scaled": 0.35,
            "paper_recent_trade_count_norm": 0.75,
            "paper_recent_slippage_bps_norm": 0.10,
            "paper_recent_return_proxy_signed_scaled": -0.25,
        },
        {},
    )

    assert vec is not None
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("paper_snapshot_trade_count_norm")] == 0.50
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("paper_snapshot_return_proxy_signed_scaled")] == 0.35
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("paper_recent_return_proxy_signed_scaled")] == -0.25


def test_behavior_feature_vector_v2_accepts_execution_realism_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "SPY",
        "BUY",
        {
            "pct_from_close": 0.003,
            "mom_5m": 0.001,
            "vol_30m": 0.004,
            "execution_fitness_norm": 0.81,
            "stop_target_realism_norm": 0.73,
            "symbol_cooldown_memory_norm": 0.16,
            "cross_bot_conflict_norm": 0.22,
        },
        {},
    )

    assert vec is not None
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("execution_fitness_norm")] == 0.81
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("stop_target_realism_norm")] == 0.73
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("cross_bot_conflict_norm")] == 0.22


def test_behavior_feature_vector_v2_accepts_core_sleeve_overlay_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "ES",
        "BUY",
        {
            "pct_from_close": 0.003,
            "mom_5m": 0.001,
            "vol_30m": 0.004,
            "core_default_dependency_norm": 0.48,
            "core_conservative_quality_gate_norm": 0.81,
            "core_aggressive_breakout_conviction_norm": 0.77,
            "core_cross_sectional_rank_norm": 0.72,
            "core_regime_specialist_blend_norm": 0.68,
            "core_event_reaction_norm": 0.74,
            "core_fx_macro_confirmation_norm": 0.69,
            "core_futures_regime_edge_norm": 0.73,
            "core_crypto_unwind_risk_norm": 0.21,
            "aggressive_relative_strength_burst_norm": 0.70,
        },
        {},
    )

    assert vec is not None
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("core_default_dependency_norm")] == 0.48
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("core_aggressive_breakout_conviction_norm")] == 0.77
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("core_cross_sectional_rank_norm")] == 0.72
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("core_futures_regime_edge_norm")] == 0.73


def test_behavior_feature_schema_includes_new_day_swing_and_long_term_features() -> None:
    keys = [
        "day_failed_breakout_risk_norm",
        "day_closing_squeeze_norm",
        "swing_weekly_pullback_quality_norm",
        "bond_equity_contamination_norm",
        "long_term_factor_exposure_control_norm",
        "long_term_overlap_rebalance_norm",
    ]
    for key in keys:
        assert key in behavior_ds.BEHAVIOR_LANE_FEATURE_NAMES
        assert key in loop._BEHAVIOR_LANE_FEATURE_NAMES
        assert key in behavior_ds.FEATURE_NAMES
        assert key in loop._BEHAVIOR_FEATURE_NAMES_V2


def test_tradeability_realism_control_blocks_directional_trade_when_conflict_is_high() -> None:
    action, score, reasons = loop._apply_tradeability_realism_control(
        action="BUY",
        score=0.78,
        threshold=0.60,
        reasons=["base_signal"],
        features={
            "execution_fitness_norm": 0.18,
            "stop_target_realism_norm": 0.20,
            "symbol_cooldown_memory_norm": 0.18,
            "cross_bot_conflict_norm": 0.86,
        },
    )

    assert action == "HOLD"
    assert score < 0.78
    assert any("execution_realism_block" in reason for reason in reasons)


def test_behavior_feature_vector_v2_accepts_tastytrade_context_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "SPY",
        "BUY",
        {
            "pct_from_close": 0.002,
            "mom_5m": 0.001,
            "vol_30m": 0.004,
            "tasty_iv_rank_norm": 0.61,
            "tasty_implied_volatility_index_norm": 0.57,
            "tasty_liquidity_rating_norm": 0.83,
            "tasty_expected_move_norm": 0.29,
            "tasty_beta_norm": 0.54,
            "tasty_watchlist_presence_norm": 1.0,
        },
        {},
    )

    assert vec is not None
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("tasty_iv_rank_norm")] == 0.61
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("tasty_liquidity_rating_norm")] == 0.83
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("tasty_watchlist_presence_norm")] == 1.0


def test_behavior_feature_vector_v2_accepts_new_options_and_fx_context_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "EURUSD",
        "BUY",
        {
            "pct_from_close": 0.002,
            "mom_5m": 0.001,
            "vol_30m": 0.004,
            "options_iv_crush_risk_norm": 0.64,
            "options_assignment_risk_norm": 0.52,
            "options_zero_dte_regime_norm": 0.71,
            "options_vol_of_vol_change_norm": 0.68,
            "options_spread_execution_risk_norm": 0.43,
            "fx_session_london_norm": 1.0,
            "fx_rollover_risk_norm": 0.0,
            "fx_dxy_yield_confirmation_norm": 0.77,
            "fx_carry_proxy_norm": 0.63,
        },
        {},
    )

    assert vec is not None
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("options_iv_crush_risk_norm")] == 0.64
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("fx_session_london_norm")] == 1.0
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("fx_dxy_yield_confirmation_norm")] == 0.77


def test_behavior_feature_vector_v2_accepts_crypto_context_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "BTC-USD",
        "BUY",
        {
            "pct_from_close": 0.004,
            "mom_5m": 0.002,
            "vol_30m": 0.009,
            "crypto_deribit_mark_iv_norm": 0.71,
            "crypto_hyperliquid_funding_norm": 0.57,
            "crypto_coinmetrics_tx_count_norm": 0.63,
            "crypto_coingecko_momentum_norm": 0.69,
            "crypto_cross_provider_price_agreement_norm": 0.93,
            "crypto_defillama_stablecoin_growth_norm": 0.59,
            "crypto_etherscan_gas_norm": 0.04,
        },
        {},
    )

    assert vec is not None
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("crypto_deribit_mark_iv_norm")] == 0.71
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("crypto_hyperliquid_funding_norm")] == 0.57
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("crypto_cross_provider_price_agreement_norm")] == 0.93


def test_behavior_feature_schema_includes_plumbed_context_and_source_quality_features() -> None:
    keys = [
        "live_macro_gate_active_norm",
        "live_macro_gate_confidence_norm",
        "sec_context_signal_norm",
        "extended_quant_signal_norm",
        "official_macro_signal_norm",
        "schwab_education_signal_norm",
        "market_breadth_signal_norm",
        "bond_reference_signal_norm",
        "source_quality_average_score_norm",
        "source_quality_required_failure_ratio_norm",
        "source_quality_soft_failure_ratio_norm",
        "source_quality_unverified_ratio_norm",
        "source_quality_cross_verified_ratio_norm",
        "source_quality_market_micro_score_norm",
        "source_quality_official_macro_score_norm",
        "source_quality_crypto_context_score_norm",
    ]
    for key in keys:
        assert key in behavior_ds.FEATURE_NAMES


def test_behavior_feature_vector_v2_accepts_market_crypto_correlation_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "BTC-USD",
        "BUY",
        {
            "pct_from_close": 0.004,
            "mom_5m": 0.002,
            "vol_30m": 0.009,
            "market_crypto_risk_corr_norm": 0.58,
            "market_crypto_spy_corr_norm": 0.61,
            "market_crypto_qqq_corr_norm": 0.55,
            "market_crypto_tlt_corr_norm": 0.50,
            "market_crypto_uup_inverse_corr_norm": 0.54,
            "market_crypto_gold_corr_norm": 0.46,
            "market_crypto_current_alignment_norm": 0.24,
            "market_crypto_divergence_norm": 0.69,
            "market_crypto_corr_confidence_norm": 1.0,
            "market_crypto_sleeve_coverage_norm": 0.72,
            "market_crypto_sleeve_avg_abs_corr_norm": 0.63,
            "market_crypto_sleeve_dispersion_norm": 0.31,
            "market_crypto_sleeve_confidence_norm": 0.84,
            "market_crypto_risk_on_crypto_alignment_norm": 0.67,
            "market_crypto_fx_crypto_inverse_corr_norm": 0.73,
            "market_crypto_rates_crypto_corr_norm": 0.44,
            "market_crypto_energy_crypto_corr_norm": 0.59,
        },
        {},
    )

    assert vec is not None
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("market_crypto_risk_corr_norm")] == 0.58
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("market_crypto_current_alignment_norm")] == 0.24
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("market_crypto_corr_confidence_norm")] == 1.0
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("market_crypto_sleeve_coverage_norm")] == 0.72
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("market_crypto_fx_crypto_inverse_corr_norm")] == 0.73


def test_behavior_feature_schema_includes_market_crypto_correlation_features() -> None:
    keys = [
        "market_crypto_risk_corr_norm",
        "market_crypto_spy_corr_norm",
        "market_crypto_qqq_corr_norm",
        "market_crypto_tlt_corr_norm",
        "market_crypto_uup_inverse_corr_norm",
        "market_crypto_gold_corr_norm",
        "market_crypto_current_alignment_norm",
        "market_crypto_divergence_norm",
        "market_crypto_corr_confidence_norm",
        "market_crypto_sleeve_coverage_norm",
        "market_crypto_sleeve_avg_abs_corr_norm",
        "market_crypto_sleeve_dispersion_norm",
        "market_crypto_sleeve_confidence_norm",
        "market_crypto_risk_on_crypto_alignment_norm",
        "market_crypto_fx_crypto_inverse_corr_norm",
        "market_crypto_rates_crypto_corr_norm",
        "market_crypto_energy_crypto_corr_norm",
    ]
    for key in keys:
        assert key in behavior_ds.FEATURE_NAMES
        assert key in loop._BEHAVIOR_FEATURE_NAMES_V2


def test_behavior_feature_schema_includes_dividend_drip_features() -> None:
    keys = [
        "dividend_drip_active_norm",
        "dividend_drip_recent_reinvest_norm",
        "dividend_drip_cash_only_norm",
        "dividend_drip_share_credit_norm",
        "dividend_drip_event_recency_norm",
        "dividend_drip_confidence_norm",
    ]
    for key in keys:
        assert key in behavior_ds.FEATURE_NAMES
        assert key in loop._BEHAVIOR_FEATURE_NAMES_V2


def test_dividend_feature_registry_includes_underwriting_keys() -> None:
    keys = [
        "dividend_streak_quality_norm",
        "dividend_cut_freeze_risk_norm",
        "dividend_fcf_coverage_norm",
        "dividend_structure_aware_quality_norm",
        "dividend_income_quality_norm",
        "dividend_trap_internal_risk_norm",
        "dividend_total_return_income_norm",
        "dividend_tax_friction_norm",
        "dividend_capture_vs_hold_edge_norm",
        "dividend_reinvest_cadence_norm",
        "dividend_payout_stress_forward_norm",
    ]
    for key in keys:
        assert key in loop._DIVIDEND_FEATURE_KEYS


def test_long_term_feature_registry_includes_underwriting_keys() -> None:
    keys = [
        "long_term_valuation_anchor_norm",
        "long_term_capital_allocation_quality_norm",
        "long_term_quality_persistence_norm",
        "long_term_accumulation_discipline_norm",
        "long_term_overlap_crowding_norm",
        "long_term_total_return_income_norm",
        "long_term_tax_friction_norm",
        "long_term_downside_preservation_norm",
        "long_term_factor_quality_mix_norm",
        "long_term_rebalance_overlap_penalty_norm",
    ]
    for key in keys:
        assert key in loop._LONG_TERM_FEATURE_KEYS


def test_dividend_underwriting_features_emit_quality_and_trap_signals() -> None:
    out = loop._dividend_underwriting_features(
        symbol="SCHD",
        features={
            "last_price": 80.0,
            "dividend_quality_score_norm": 0.78,
            "dividend_safety_composite_norm": 0.74,
            "dividend_payout_ratio_norm": 0.46,
            "dividend_growth_momentum_norm": 0.71,
            "dividend_compound_growth_norm": 0.66,
            "dividend_compound_drawdown_norm": 0.12,
            "dividend_drip_active_norm": 0.82,
            "dividend_drip_recent_reinvest_norm": 0.68,
            "dividend_tax_qualified_hold_norm": 0.58,
            "dividend_yield_norm": 0.42,
            "calendar_dividend_quality_signal_norm": 0.76,
            "sec_estimate_revision_drift_norm": 0.72,
            "sec_earnings_whisper_surprise_norm": 0.69,
            "sec_insider_buy_30d_norm": 0.61,
            "sec_insider_sell_30d_norm": 0.08,
            "short_borrow_availability_norm": 0.94,
            "short_borrow_fee_norm": 0.07,
            "short_utilization_norm": 0.12,
            "swing_sector_relative_strength_norm": 0.62,
        },
        iter_count=14,
        state={},
    )

    assert out["dividend_fcf_coverage_norm"] > 0.40
    assert out["dividend_total_return_income_norm"] > 0.40
    assert out["dividend_trap_internal_risk_norm"] < 0.45
    assert out["dividend_capture_vs_hold_edge_norm"] > 0.0
    assert out["dividend_reinvest_cadence_norm"] > 0.0


def test_long_term_underwriting_features_emit_accumulation_controls() -> None:
    out = loop._long_term_underwriting_features(
        symbol="SCHD",
        features={
            "last_price": 82.0,
            "pct_from_close": -0.014,
            "mom_5m": 0.0012,
            "vol_30m": 0.007,
            "range_pos": 0.42,
            "dividend_quality_score_norm": 0.80,
            "dividend_payout_ratio_norm": 0.44,
            "dividend_compound_growth_norm": 0.63,
            "dividend_compound_drawdown_norm": 0.10,
            "dividend_yield_norm": 0.39,
            "sec_estimate_revision_drift_norm": 0.74,
            "sec_earnings_whisper_surprise_norm": 0.67,
            "sec_insider_buy_30d_norm": 0.58,
            "bond_duration_years_norm": 0.22,
            "bond_duration_regime_norm": 0.34,
            "sofr_term_pressure_norm": 0.30,
            "etf_fund_family_flow_norm": 0.28,
            "calendar_index_rebalance_window_norm": 0.10,
        },
        profile="long_term_dividend",
        iter_count=22,
        state={},
    )

    assert out["long_term_valuation_anchor_norm"] > 0.35
    assert out["long_term_accumulation_discipline_norm"] > 0.30
    assert out["long_term_downside_preservation_norm"] > 0.30
    assert out["long_term_factor_quality_mix_norm"] > 0.30
    assert out["long_term_rebalance_overlap_penalty_norm"] >= 0.0


def test_behavior_regime_index_marks_dividend_defensive_context_mean_revert() -> None:
    _, regime = behavior_ds._regime_index(
        "SCHD",
        {
            "pct_from_close": 0.0004,
            "mom_5m": 0.0002,
            "vol_30m": 0.003,
            "dividend_yield_norm": 0.72,
            "dividend_quality_score_norm": 0.81,
            "dividend_drip_active_norm": 0.84,
        },
    )

    assert regime == "mean_revert"


def test_behavior_regime_index_marks_futures_event_risk_context_shock() -> None:
    _, regime = behavior_ds._regime_index(
        "ES=F",
        {
            "pct_from_close": 0.001,
            "mom_5m": 0.0005,
            "vol_30m": 0.004,
            "calendar_event_proximity_norm": 0.68,
            "futures_order_book_imbalance_norm": 0.77,
            "futures_term_structure_norm": 0.61,
        },
    )

    assert regime == "shock"


def test_behavior_feature_vector_v2_accepts_dividend_drip_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "SCHD",
        "BUY",
        {
            "pct_from_close": 0.002,
            "mom_5m": 0.001,
            "vol_30m": 0.003,
            "dividend_drip_active_norm": 0.84,
            "dividend_drip_recent_reinvest_norm": 0.65,
            "dividend_drip_cash_only_norm": 0.18,
            "dividend_drip_share_credit_norm": 0.57,
            "dividend_drip_event_recency_norm": 0.92,
            "dividend_drip_confidence_norm": 0.88,
        },
        {},
    )

    assert vec is not None
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("dividend_drip_active_norm")] == 0.84
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("dividend_drip_recent_reinvest_norm")] == 0.65
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("dividend_drip_confidence_norm")] == 0.88


def test_load_governance_index_captures_lane_strategy_features() -> None:
    since_utc = datetime.now(timezone.utc) - timedelta(hours=1)
    rows = [
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_id": "snap-1",
            "lane_strategy_features": {
                "day_regime_trend_norm": 0.81,
                "swing_regime_chop_norm": 0.24,
                "bond_curve_steepener_norm": 0.67,
            },
        }
    ]

    out = behavior_ds._load_governance_index(rows, since_utc=since_utc)

    assert out["snap-1"]["day_regime_trend_norm"] == 0.81
    assert out["snap-1"]["swing_regime_chop_norm"] == 0.24
    assert out["snap-1"]["bond_curve_steepener_norm"] == 0.67
    assert out["snap-1"]["day_execution_cost_risk_norm"] == 0.0


def test_load_governance_index_captures_capital_flow_features() -> None:
    since_utc = datetime.now(timezone.utc) - timedelta(hours=1)
    rows = [
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_id": "snap-2",
            "capital_flow": {
                "capital_flow_signed_scaled": -0.72,
                "capital_flow_inflow_norm": 0.0,
                "capital_flow_outflow_norm": 0.61,
            },
        }
    ]

    out = behavior_ds._load_governance_index(rows, since_utc=since_utc)

    assert out["snap-2"]["capital_flow_signed_scaled"] == -0.72
    assert out["snap-2"]["capital_flow_inflow_norm"] == 0.0
    assert out["snap-2"]["capital_flow_outflow_norm"] == 0.61


def test_behavior_feature_vector_v2_accepts_appended_lane_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "NVDA",
        "BUY",
        {
            "pct_from_close": 0.01,
            "mom_5m": 0.004,
            "vol_30m": 0.008,
            "range_pos": 0.8,
            "spread_bps": 3.0,
            "day_regime_trend_norm": 0.83,
            "swing_regime_alignment_norm": 0.66,
            "bond_carry_roll_norm": 0.41,
        },
        {},
    )

    assert vec is not None
    assert vec.shape[1] == len(loop._BEHAVIOR_FEATURE_NAMES_V2)
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("day_regime_trend_norm")] == 0.83
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("swing_regime_alignment_norm")] == 0.66
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("bond_carry_roll_norm")] == 0.41


def test_behavior_feature_vector_v2_accepts_capital_flow_features() -> None:
    vec = loop._behavior_feature_vector_v2(
        "SPY",
        "SELL",
        {
            "pct_from_close": -0.004,
            "mom_5m": -0.002,
            "vol_30m": 0.005,
            "capital_flow_signed_scaled": -0.88,
            "capital_flow_inflow_norm": 0.0,
            "capital_flow_outflow_norm": 0.74,
        },
        {},
    )

    assert vec is not None
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("capital_flow_signed_scaled")] == -0.88
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("capital_flow_inflow_norm")] == 0.0
    assert vec[0, loop._BEHAVIOR_FEATURE_NAMES_V2.index("capital_flow_outflow_norm")] == 0.74


def test_effective_account_equity_proxy_prefers_fresh_broker_truth() -> None:
    effective, meta = loop._effective_account_equity_proxy(
        {
            "status": "mismatch",
            "age_iters": 1,
            "account_metrics": {"equity": 152500.0, "cash_balance": 48000.0},
        },
        fallback_equity_proxy=100000.0,
    )

    assert effective == 152500.0
    assert meta["source"] == "broker_truth_account_metrics"


def test_estimate_capital_flow_state_detects_large_outflow() -> None:
    flow = loop._estimate_capital_flow_state(
        {"equity": 87000.0, "cash_balance": 22000.0},
        {"equity": 120000.0, "cash_balance": 55000.0},
    )

    assert flow["detected"] is True
    assert flow["estimated_amount"] < 0.0
    assert flow["capital_flow_signed_scaled"] < 0.0
    assert flow["capital_flow_outflow_norm"] > 0.0
    assert flow["capital_flow_inflow_norm"] == 0.0


def test_rollback_schema_compatible_allows_prefix_compatible_extension() -> None:
    ok, reason = trainer._rollback_schema_compatible(
        {
            "load_ok": True,
            "effective_dim": 3,
            "feature_names": ["a", "b", "c"],
        },
        dataset_feature_dim=5,
        dataset_feature_names=["a", "b", "c", "d", "e"],
        require_feature_names=True,
    )

    assert ok is True
    assert reason.startswith("prefix_compatible")


def test_curated_dataset_guard_accepts_curated_behavior_dataset() -> None:
    ok, reason, summary = trainer._curated_dataset_guard(
        {
            "dataset_kind": "curated_decision_governance",
            "source": {
                "decision_files": 3,
                "decision_sql_files": 2,
                "governance_files": 2,
                "governance_sql_files": 1,
                "pnl_attribution_files": 1,
                "pnl_sql_files": 1,
            },
        }
    )

    assert ok is True
    assert reason == "ok"
    assert summary["decision_sources"] == 5
    assert summary["governance_sources"] == 3


def test_curated_dataset_guard_rejects_legacy_dataset_kind() -> None:
    ok, reason, summary = trainer._curated_dataset_guard(
        {
            "dataset_kind": "legacy_trade_history",
            "source": {
                "decision_files": 3,
                "governance_files": 2,
            },
        }
    )

    assert ok is False
    assert reason == "dataset_kind_not_curated"
    assert summary["dataset_kind"] == "legacy_trade_history"


def test_trade_behavior_data_quality_gate_blocks_on_health_and_thin_paper_feedback(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(health_root / "snapshot_coverage_latest.json", {"timestamp_utc": now.isoformat(), "coverage_ratio": 1.0})
    _write_json(health_root / "data_source_divergence_latest.json", {"timestamp_utc": now.isoformat(), "worst_relative_spread": 0.0})
    _write_json(
        health_root / "preopen_replay_drift_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "drift": {
                "decision_rows": 0.0,
                "governance_rows": 0.0,
                "decision_stale": 0.0,
                "governance_stale": 0.0,
            },
        },
    )
    _write_json(health_root / "replay_preopen_sanity_latest.json", {"timestamp_utc": now.isoformat(), "ok": True})
    _write_json(health_root / "health_gates_latest.json", {"timestamp_utc": now.isoformat(), "hard_gate_triggered": True})
    _write_json(
        health_root / "paper_performance_latest.json",
        {"timestamp_utc": now.isoformat(), "sleeve_latest": [{"profile": "default", "executions": 2, "non_flat_strategy_count": 1}]},
    )
    _write_json(walk_root / "promotion_readiness_latest.json", {"promote_ok": True})

    ok, reasons, summary = trainer._data_quality_gate(tmp_path, require_walk_forward_ok=True)

    assert ok is False
    assert "health_gate_triggered" in reasons
    assert "paper_feedback_executions=2 < min=24" in reasons
    assert "paper_feedback_active_sleeves=1 < min=3" in reasons
    assert summary["health_gate_triggered"] is True
    assert summary["paper_feedback_total_executions"] == 2


def test_trade_behavior_data_quality_gate_accepts_healthy_paper_feedback(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(health_root / "snapshot_coverage_latest.json", {"timestamp_utc": now.isoformat(), "coverage_ratio": 1.0})
    _write_json(health_root / "data_source_divergence_latest.json", {"timestamp_utc": now.isoformat(), "worst_relative_spread": 0.0})
    _write_json(
        health_root / "preopen_replay_drift_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "drift": {
                "decision_rows": 0.0,
                "governance_rows": 0.0,
                "decision_stale": 0.0,
                "governance_stale": 0.0,
            },
        },
    )
    _write_json(health_root / "replay_preopen_sanity_latest.json", {"timestamp_utc": now.isoformat(), "ok": True})
    _write_json(health_root / "health_gates_latest.json", {"timestamp_utc": now.isoformat(), "hard_gate_triggered": False})
    _write_json(
        health_root / "paper_performance_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sleeve_latest": [
                {"profile": "default", "executions": 20, "non_flat_strategy_count": 1},
                {"profile": "aggressive", "executions": 18, "non_flat_strategy_count": 1},
                {"profile": "bond", "executions": 12, "non_flat_strategy_count": 1},
            ],
        },
    )
    _write_json(walk_root / "promotion_readiness_latest.json", {"promote_ok": True})

    ok, reasons, summary = trainer._data_quality_gate(tmp_path, require_walk_forward_ok=True)

    assert ok is True
    assert reasons == []
    assert summary["paper_feedback_total_executions"] == 50
    assert summary["paper_feedback_active_sleeves"] == 3
