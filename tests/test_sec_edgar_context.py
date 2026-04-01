from datetime import datetime, timezone

from scripts.collect_sec_edgar_context import _aggregate_features, _derive_symbol_summary, _ticker_map


def test_ticker_map_zero_pads_cik() -> None:
    payload = {
        "0": {"ticker": "AAPL", "cik_str": 320193},
        "1": {"ticker": "MSFT", "cik_str": "789019"},
    }
    out = _ticker_map(payload)
    assert out["AAPL"] == "0000320193"
    assert out["MSFT"] == "0000789019"


def test_derive_symbol_summary_counts_recent_filings() -> None:
    now = datetime(2026, 3, 20, 14, 0, tzinfo=timezone.utc)
    rows = [
        {
            "form": "8-K",
            "filing_date": "2026-03-20",
            "accepted_at": "2026-03-20T12:00:00+00:00",
            "description": "Company raises guidance after earnings",
            "market_session": "premarket",
        },
        {
            "form": "SC 13G",
            "filing_date": "2026-03-18",
            "accepted_at": "2026-03-18T20:00:00+00:00",
            "description": "Beneficial ownership report",
            "market_session": "after_hours",
        },
    ]
    out = _derive_symbol_summary("AAPL", "0000320193", rows, now)
    assert out["filings_1d"] == 1
    assert out["high_impact_7d"] == 2
    assert out["guidance_7d"] == 1
    assert out["ownership_30d"] == 1
    assert out["features"]["sec_recent_proximity_norm"] > 0.0


def test_aggregate_features_builds_calendar_and_news_context() -> None:
    row = {
        "symbol": "AAPL",
        "filings_1d": 1,
        "filings_7d": 2,
        "high_impact_1d": 1,
        "high_impact_7d": 2,
        "features": {
            "sec_earnings_7d_norm": 0.8,
            "sec_guidance_7d_norm": 0.6,
            "sec_regulatory_7d_norm": 0.2,
            "sec_ownership_30d_norm": 0.5,
            "sec_insider_30d_norm": 0.4,
            "sec_recent_proximity_norm": 0.9,
            "news_premarket_norm": 0.7,
            "news_intraday_norm": 0.0,
            "news_after_hours_norm": 0.4,
        },
    }
    out = _aggregate_features([row], request_count=5)
    assert out["calendar_features"]["calendar_feed_available"] == 1.0
    assert out["news_features"]["news_topic_earnings_norm"] == 0.8
    assert out["global_features"]["sec_recent_filings_1d_norm"] > 0.0
    assert out["symbol_features"]["AAPL"]["sec_guidance_7d_norm"] == 0.6
