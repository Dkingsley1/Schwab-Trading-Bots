import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.collect_market_micro_context import _aggregate_local_micro_context


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
                "options_gamma_exposure_norm": 0.55,
                "options_unusual_flow_norm": 0.35,
                "options_put_call_oi_ratio_norm": 0.72,
                "bond_hy_ig_flow_norm": 0.62,
                "bond_nav_stress_norm": 0.18,
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
                "options_gamma_exposure_norm": 0.60,
                "options_unusual_flow_norm": 0.32,
                "options_put_call_oi_ratio_norm": 0.69,
                "bond_hy_ig_flow_norm": 0.66,
                "bond_nav_stress_norm": 0.20,
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
                "options_gamma_exposure_norm": 0.48,
                "options_unusual_flow_norm": 0.40,
                "options_put_call_oi_ratio_norm": 0.28,
                "bond_hy_ig_flow_norm": 0.58,
                "bond_nav_stress_norm": 0.24,
            },
        },
        {
            "timestamp_utc": "2026-03-20T19:10:00+00:00",
            "strategy": "futures_master_bot",
            "symbol": "SPY",
            "action": "BUY",
            "quantity": 8,
            "features": {
                "pct_from_close": 0.019,
                "mom_5m": 0.007,
                "vol_30m": 0.021,
                "spread_bps": 10.0,
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
    assert spy["market_micro_power_hour_pressure_norm"] > 0.0
    assert spy["market_micro_gap_continuation_norm"] > 0.0
    assert spy["market_micro_reversal_risk_norm"] > 0.0
    assert spy["market_micro_trend_persistence_norm"] > 0.0
    assert spy["market_micro_range_expansion_norm"] > 0.0
