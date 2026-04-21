import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.collect_market_micro_context import _aggregate_local_micro_context, _parse_nasdaq_trade_halt_rows


def test_aggregate_local_micro_context_emits_richer_session_features(tmp_path: Path) -> None:
    day_stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    day_path = tmp_path / "decision_explanations" / "shadow_conservative_equities" / f"decision_explanations_{day_stamp}.jsonl"
    day_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp_utc": "2026-03-20T12:45:00+00:00",
            "strategy": "grand_master_bot",
            "symbol": "SPY",
            "action": "BUY",
            "quantity": 3,
            "features": {
                "pct_from_close": 0.011,
                "mom_5m": 0.004,
                "vol_30m": 0.018,
                "spread_bps": 8.0,
                "bid_size": 1200.0,
                "ask_size": 900.0,
                "queue_depth": 2100.0,
                "options_gamma_exposure_norm": 0.55,
                "options_unusual_flow_norm": 0.35,
                "options_put_call_oi_ratio_norm": 0.72,
                "bond_hy_ig_flow_norm": 0.62,
                "bond_nav_stress_norm": 0.18,
                "dark_pool_print_norm": 0.61,
                "off_exchange_share_norm": 0.57,
                "etf_nav_premium_discount_norm": 0.22,
                "etf_creation_redemption_stress_norm": 0.41,
            },
        },
        {
            "timestamp_utc": "2026-03-20T13:35:00+00:00",
            "strategy": "grand_master_bot",
            "symbol": "SPY",
            "action": "BUY",
            "quantity": 6,
            "features": {
                "pct_from_close": 0.016,
                "mom_5m": 0.006,
                "vol_30m": 0.022,
                "spread_bps": 11.0,
                "bid_size": 700.0,
                "ask_size": 550.0,
                "queue_depth": 1250.0,
                "options_gamma_exposure_norm": 0.60,
                "options_unusual_flow_norm": 0.32,
                "options_put_call_oi_ratio_norm": 0.69,
                "bond_hy_ig_flow_norm": 0.66,
                "bond_nav_stress_norm": 0.20,
                "dark_pool_imbalance_norm": 0.52,
                "ats_volume_share_norm": 0.48,
                "etf_primary_secondary_liquidity_norm": 0.77,
            },
        },
        {
            "timestamp_utc": "2026-03-20T16:30:00+00:00",
            "strategy": "options_master_bot",
            "symbol": "SPY",
            "action": "SELL",
            "quantity": 7,
            "features": {
                "pct_from_close": -0.013,
                "mom_5m": 0.004,
                "vol_30m": 0.014,
                "spread_bps": 13.0,
                "bid_size": 380.0,
                "ask_size": 340.0,
                "queue_depth": 720.0,
                "options_gamma_exposure_norm": 0.48,
                "options_unusual_flow_norm": 0.40,
                "options_put_call_oi_ratio_norm": 0.28,
                "bond_hy_ig_flow_norm": 0.58,
                "bond_nav_stress_norm": 0.24,
                "pct_from_close": -0.113,
                "etf_underlying_basket_stress_norm": 0.58,
            },
        },
        {
            "timestamp_utc": "2026-03-20T19:40:00+00:00",
            "strategy": "futures_master_bot",
            "symbol": "SPY",
            "action": "BUY",
            "quantity": 8,
            "features": {
                "pct_from_close": 0.019,
                "mom_5m": 0.007,
                "vol_30m": 0.021,
                "spread_bps": 10.0,
                "bid_size": 410.0,
                "ask_size": 390.0,
                "queue_depth": 800.0,
                "options_gamma_exposure_norm": 0.52,
                "options_unusual_flow_norm": 0.44,
                "options_put_call_oi_ratio_norm": 0.76,
                "bond_hy_ig_flow_norm": 0.71,
                "bond_nav_stress_norm": 0.22,
            },
        },
    ]
    day_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    out = _aggregate_local_micro_context(tmp_path, lookback_days=5, symbols={"SPY"})
    spy = out["SPY"]

    assert spy["market_micro_premarket_pressure_norm"] > 0.0
    assert spy["market_micro_opening_auction_norm"] > 0.0
    assert spy["market_micro_opening_auction_imbalance_norm"] > 0.5
    assert spy["market_micro_opening_drive_pressure_norm"] > 0.0
    assert spy["market_micro_power_hour_pressure_norm"] > 0.0
    assert spy["market_micro_closing_auction_imbalance_norm"] > 0.5
    assert spy["market_micro_closing_cross_pressure_norm"] > 0.0
    assert spy["market_micro_auction_print_pressure_norm"] > 0.0
    assert spy["market_micro_gap_continuation_norm"] > 0.0
    assert spy["market_micro_reversal_risk_norm"] > 0.0
    assert spy["market_micro_trend_persistence_norm"] > 0.0
    assert spy["market_micro_range_expansion_norm"] > 0.0
    assert spy["market_micro_dark_pool_pressure_norm"] > 0.0
    assert spy["market_micro_off_exchange_share_norm"] > 0.0
    assert spy["market_micro_spread_regime_norm"] > 0.0
    assert spy["market_micro_spread_widening_norm"] > 0.0
    assert spy["market_micro_queue_depth_decay_norm"] > 0.0
    assert spy["market_micro_depth_collapse_norm"] > 0.0
    assert spy["market_micro_quote_fade_rate_norm"] > 0.0
    assert spy["market_micro_tradeability_score_norm"] > 0.0
    assert spy["market_micro_session_open_norm"] > 0.0
    assert spy["market_micro_session_midday_norm"] > 0.0
    assert spy["market_micro_session_power_hour_norm"] > 0.0
    assert spy["market_micro_overnight_gap_norm"] > 0.0
    assert spy["market_micro_post_event_drift_norm"] > 0.0
    assert spy["market_micro_lunch_chop_norm"] > 0.0
    assert spy["market_micro_open_close_imbalance_regime_norm"] > 0.0
    assert spy["market_micro_symbol_cooldown_pressure_norm"] > 0.0
    assert spy["market_micro_gap_fade_risk_norm"] > 0.0
    assert spy["market_micro_overnight_event_hazard_norm"] > 0.0
    assert spy["market_micro_ssr_active_norm"] > 0.0
    assert spy["etf_creation_redemption_stress_norm"] > 0.0
    assert spy["etf_primary_secondary_liquidity_norm"] > 0.0
    assert spy["etf_fund_family_flow_norm"] > 0.5
    assert spy["etf_fund_family_creation_pressure_norm"] > 0.0


def test_parse_nasdaq_trade_halt_rows_reads_pipe_delimited_feed() -> None:
    raw = "\n".join(
        [
            "Symbol|Issue Name|Reason Code|Pause Threshold Price|Halt Time|Resume Date|Resume Time",
            "ABCD|Example Corp|LUDP|25.01|13:14:00|03/20/2026|13:19:00",
            "WXYZ|Another Corp|T1|0.00|09:31:00|03/20/2026|10:15:00",
        ]
    )

    rows = _parse_nasdaq_trade_halt_rows(raw)

    assert rows[0]["symbol"] == "ABCD"
    assert rows[0]["reason"] == "LUDP"
    assert rows[0]["resume_time"] == "13:19:00"
    assert rows[1]["symbol"] == "WXYZ"
