from datetime import datetime, timedelta, timezone

from core.derivatives_features import (
    summarize_calendar_payload,
    summarize_futures_quote_features,
    summarize_option_chain,
)


def test_summarize_option_chain_emits_bias_roll_vwap_term_and_vol_expectation() -> None:
    payload = {
        "items": [
            {
                "symbol": "XYZ_20260417C00100000",
                "putCall": "CALL",
                "strikePrice": 100.0,
                "impliedVolatility": 0.38,
                "openInterest": 1200,
                "totalVolume": 900,
                "delta": 0.52,
                "gamma": 0.08,
                "theta": -0.04,
                "vega": 0.12,
                "bid": 4.8,
                "ask": 5.2,
                "mark": 5.1,
                "daysToExpiration": 14,
            },
            {
                "symbol": "XYZ_20260417P00100000",
                "putCall": "PUT",
                "strikePrice": 100.0,
                "impliedVolatility": 0.46,
                "openInterest": 2100,
                "totalVolume": 1400,
                "delta": -0.49,
                "gamma": 0.09,
                "theta": -0.05,
                "vega": 0.14,
                "bid": 5.4,
                "ask": 5.9,
                "mark": 5.7,
                "daysToExpiration": 14,
            },
            {
                "symbol": "XYZ_20260719C00110000",
                "putCall": "CALL",
                "strikePrice": 110.0,
                "impliedVolatility": 0.34,
                "openInterest": 800,
                "totalVolume": 500,
                "bid": 2.1,
                "ask": 2.4,
                "mark": 2.3,
                "daysToExpiration": 90,
            },
            {
                "symbol": "XYZ_20260719P00090000",
                "putCall": "PUT",
                "strikePrice": 90.0,
                "impliedVolatility": 0.43,
                "openInterest": 1600,
                "totalVolume": 700,
                "bid": 2.5,
                "ask": 2.9,
                "mark": 2.8,
                "daysToExpiration": 90,
            },
        ]
    }

    out = summarize_option_chain(
        payload,
        symbol="XYZ",
        underlying_price=100.0,
        realized_vol=0.22,
        now_ts=datetime.now(timezone.utc).timestamp(),
    )

    assert out["options_chain_available"] == 1.0
    assert 0.0 <= out["options_negative_bias_norm"] <= 1.0
    assert 0.0 <= out["options_roll_yield_norm"] <= 1.0
    assert 0.0 <= out["options_vwap_bias_norm"] <= 1.0
    assert 0.0 <= out["options_vol_expectation_norm"] <= 1.0
    assert 0.0 <= out["options_gamma_exposure_norm"] <= 1.0
    assert 0.0 <= out["options_call_wall_distance_norm"] <= 1.0
    assert 0.0 <= out["options_put_wall_distance_norm"] <= 1.0
    assert 0.0 <= out["options_oi_concentration_norm"] <= 1.0
    assert 0.0 <= out["options_unusual_flow_norm"] <= 1.0
    assert 0.0 <= out["options_0dte_share_norm"] <= 1.0
    assert 0.0 <= out["options_net_call_premium_bias_norm"] <= 1.0
    assert 0.0 <= out["options_sweep_flow_norm"] <= 1.0
    assert 0.0 <= out["options_block_flow_norm"] <= 1.0
    assert 0.0 <= out["options_iv_percentile_norm"] <= 1.0
    assert 0.0 <= out["options_iv_realized_spread_norm"] <= 1.0
    assert 0.0 <= out["options_gamma_front_share_norm"] <= 1.0
    assert 0.0 <= out["options_gamma_expiry_skew_norm"] <= 1.0
    assert "options_iv_term_structure" in out


def test_summarize_futures_quote_emits_funding_basis_term_vwap_negative_bias_and_roll() -> None:
    payload = {
        "bidPrice": 100.0,
        "askPrice": 100.2,
        "bidSize": 2200,
        "askSize": 1400,
        "openInterest": 54000,
        "fundingRate": 0.0008,
        "markPrice": 100.5,
        "indexPrice": 100.0,
        "vwap": 99.6,
        "daysToExpiration": 35,
        "takerBuyVolume": 18000,
        "takerSellVolume": 12000,
        "longShortRatio": 1.35,
        "sessionVolumeProfile": 1.1,
        "liquidationVolume": 125000,
        "curve": [
            {"daysToExpiration": 35, "markPrice": 100.5, "indexPrice": 100.0},
            {"daysToExpiration": 70, "markPrice": 101.7, "indexPrice": 100.1},
        ],
    }

    out = summarize_futures_quote_features(payload, last_price=100.1)

    assert out["futures_spread_bps"] > 0.0
    assert 0.0 <= out["futures_funding_rate_norm"] <= 1.0
    assert 0.0 <= out["futures_term_structure_norm"] <= 1.0
    assert 0.0 <= out["futures_negative_bias_norm"] <= 1.0
    assert 0.0 <= out["futures_roll_yield_norm"] <= 1.0
    assert 0.0 <= out["futures_vwap_bias_norm"] <= 1.0
    assert 0.0 <= out["futures_taker_imbalance_norm"] <= 1.0
    assert 0.0 <= out["futures_cvd_norm"] <= 1.0
    assert 0.0 <= out["futures_liquidation_risk_norm"] <= 1.0
    assert 0.0 <= out["futures_long_short_ratio_norm"] <= 1.0
    assert 0.0 <= out["futures_basis_divergence_norm"] <= 1.0
    assert 0.0 <= out["futures_mark_index_dislocation_norm"] <= 1.0
    assert 0.0 <= out["futures_session_volume_profile_norm"] <= 1.0
    assert 0.0 <= out["futures_calendar_spread_curve_norm"] <= 1.0


def test_summarize_calendar_payload_emits_event_proximity_and_expiry_week() -> None:
    now = datetime.now(timezone.utc)
    payload = {
        "events": [
            {
                "title": "FOMC Rate Decision",
                "eventDate": (now + timedelta(hours=8)).isoformat(),
                "importance": "high",
            },
            {
                "title": "Options Expiration",
                "eventDate": (now + timedelta(days=2)).isoformat(),
            },
        ]
    }

    out = summarize_calendar_payload(payload, now_ts=now.timestamp())

    assert out["calendar_feed_available"] == 1.0
    assert out["calendar_next_event_norm"] > 0.0
    assert out["calendar_high_impact_24h_norm"] > 0.0
    assert out["calendar_options_expiry_week_norm"] > 0.0



def test_summarize_calendar_payload_supports_tradingeconomics_style_fields() -> None:
    now = datetime.now(timezone.utc)
    payload = [
        {
            "Date": (now + timedelta(hours=3)).isoformat(),
            "Country": "United States",
            "Event": "CPI YoY",
            "Importance": 3,
            "Actual": "3.6%",
            "Forecast": "3.2%",
            "Previous": "3.0%",
        },
        {
            "Date": (now + timedelta(days=1)).isoformat(),
            "Country": "United States",
            "Event": "10-Year Treasury Auction",
            "Importance": "Medium",
            "Actual": "4.18",
            "Forecast": "4.11",
            "Previous": "4.07",
        },
    ]

    out = summarize_calendar_payload(payload, now_ts=now.timestamp())

    assert out["calendar_feed_available"] == 1.0
    assert out["calendar_next_event_norm"] > 0.0
    assert out["calendar_high_impact_24h_norm"] > 0.0
    assert out["calendar_macro_event_norm"] > 0.0
    assert out["calendar_macro_surprise_norm"] != 0.5
    assert out["calendar_macro_abs_surprise_norm"] > 0.0
    assert out["calendar_cpi_event_norm"] > 0.0
    assert out["calendar_treasury_auction_norm"] > 0.0


def test_summarize_calendar_payload_emits_dividend_capture_and_payout_features() -> None:
    now = datetime.now(timezone.utc)
    payload = {
        "events": [
            {
                "title": "Quality Dividend Ex-Dividend Date",
                "symbol": "SCHD",
                "eventDate": (now + timedelta(days=2)).isoformat(),
                "importance": "medium",
            },
            {
                "title": "Dividend Payment Date",
                "symbol": "SCHD",
                "eventDate": (now + timedelta(days=9)).isoformat(),
                "importance": "medium",
            },
            {
                "title": "Ex-Dividend Date",
                "symbol": "VIG",
                "eventDate": (now - timedelta(days=1)).isoformat(),
                "importance": "low",
            },
        ]
    }

    out = summarize_calendar_payload(payload, now_ts=now.timestamp())

    assert out["calendar_feed_available"] == 1.0
    assert out["calendar_dividend_events_30d_norm"] > 0.0
    assert out["calendar_dividend_exdate_proximity_norm"] > 0.0
    assert out["calendar_dividend_payout_proximity_norm"] > 0.0
    assert out["calendar_dividend_recent_exdate_norm"] > 0.0
    assert out["calendar_dividend_quality_signal_norm"] > 0.0
