from datetime import datetime, timezone
from typing import Any

from scripts import collect_sec_edgar_context as sec_edgar
from scripts.collect_sec_edgar_context import _aggregate_features, _derive_symbol_summary, _filing_text_signals, _ticker_map


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
            "text_signals": {
                "guidance_raise": 1.0,
                "guidance_cut": 0.0,
                "offering": 0.0,
                "dilution": 0.0,
                "mna": 0.0,
                "restatement": 0.0,
                "financing_stress": 0.0,
            },
        },
        {
            "form": "SC 13G",
            "filing_date": "2026-03-18",
            "accepted_at": "2026-03-18T20:00:00+00:00",
            "description": "Beneficial ownership report",
            "market_session": "after_hours",
            "text_signals": {
                "guidance_raise": 0.0,
                "guidance_cut": 0.0,
                "offering": 0.0,
                "dilution": 0.0,
                "mna": 1.0,
                "restatement": 0.0,
                "financing_stress": 0.0,
            },
        },
        {
            "form": "4",
            "filing_date": "2026-03-17",
            "accepted_at": "2026-03-17T13:00:00+00:00",
            "description": "Director purchased shares after stock split and special dividend announcement",
            "market_session": "premarket",
            "text_signals": {
                "guidance_raise": 0.0,
                "guidance_cut": 0.0,
                "offering": 0.0,
                "dilution": 0.0,
                "mna": 0.0,
                "restatement": 0.0,
                "financing_stress": 0.0,
                "insider_buy": 1.0,
                "insider_sell": 0.0,
                "estimate_raise": 1.0,
                "estimate_cut": 0.0,
                "whisper_beat": 1.0,
                "whisper_miss": 0.0,
                "split_hazard": 1.0,
                "special_dividend": 1.0,
                "offering_priced": 0.0,
                "lockup_secondary": 0.0,
            },
        },
    ]
    out = _derive_symbol_summary("AAPL", "0000320193", rows, now)
    assert out["filings_1d"] == 1
    assert out["high_impact_7d"] == 3
    assert out["guidance_7d"] == 1
    assert out["mna_7d"] == 1
    assert out["ownership_30d"] == 1
    assert out["insider_buy_30d"] == 1
    assert out["estimate_raise_30d"] == 1
    assert out["split_hazard_30d"] == 1
    assert out["features"]["sec_recent_proximity_norm"] > 0.0
    assert out["features"]["sec_mna_7d_norm"] > 0.0
    assert out["features"]["sec_insider_buy_30d_norm"] > 0.0
    assert out["features"]["sec_estimate_revision_drift_norm"] > 0.5
    assert out["features"]["sec_earnings_whisper_surprise_norm"] > 0.5


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
            "sec_mna_7d_norm": 0.75,
            "sec_offering_7d_norm": 0.4,
            "sec_ownership_30d_norm": 0.5,
            "sec_insider_30d_norm": 0.4,
            "sec_insider_buy_30d_norm": 0.7,
            "sec_insider_sell_30d_norm": 0.1,
            "sec_estimate_revision_drift_norm": 0.8,
            "sec_earnings_whisper_surprise_norm": 0.75,
            "sec_split_hazard_30d_norm": 0.6,
            "sec_special_dividend_30d_norm": 0.4,
            "sec_offering_priced_30d_norm": 0.3,
            "sec_lockup_secondary_30d_norm": 0.2,
            "sec_recent_proximity_norm": 0.9,
            "news_premarket_norm": 0.7,
            "news_intraday_norm": 0.0,
            "news_after_hours_norm": 0.4,
        },
    }
    out = _aggregate_features([row], request_count=5)
    assert out["calendar_features"]["calendar_feed_available"] == 1.0
    assert out["news_features"]["news_topic_earnings_norm"] == 0.8
    assert out["news_features"]["news_topic_mna_norm"] == 0.75
    assert out["news_features"]["news_topic_guidance_norm"] >= 0.6
    assert out["global_features"]["sec_recent_filings_1d_norm"] > 0.0
    assert out["global_features"]["sec_estimate_revision_drift_norm"] == 0.8
    assert out["global_features"]["sec_split_hazard_30d_norm"] == 0.6
    assert out["symbol_features"]["AAPL"]["sec_guidance_7d_norm"] == 0.6


def test_filing_text_signals_detects_dilution_and_financing_stress() -> None:
    out = _filing_text_signals(
        "The company entered into a registered direct offering, warned about going concern risk, disclosed a restatement, "
        "priced a secondary offering, announced a special dividend, and said an insider purchased shares before a stock split."
    )

    assert out["offering"] == 1.0
    assert out["financing_stress"] == 1.0
    assert out["restatement"] == 1.0
    assert out["offering_priced"] == 1.0
    assert out["special_dividend"] == 1.0
    assert out["insider_buy"] == 1.0
    assert out["split_hazard"] == 1.0


def test_collect_sec_edgar_context_treats_archive_text_as_optional(monkeypatch: Any) -> None:
    recent_ts = datetime.now(timezone.utc).replace(microsecond=0)

    def fake_http_json_result(url: str, **kwargs: Any) -> dict[str, Any]:
        if "company_tickers" in url:
            return {
                "ok": True,
                "json": {"0": {"ticker": "AAPL", "cik_str": 320193}},
                "fetched_utc": "2026-03-20T14:00:00+00:00",
                "source_confidence_norm": 0.99,
                "schema_confidence_norm": 0.97,
                "freshness_norm": 1.0,
            }
        return {
            "ok": True,
            "json": {
                "filings": {
                    "recent": {
                        "form": ["8-K"],
                        "filingDate": [recent_ts.date().isoformat()],
                        "acceptanceDateTime": [recent_ts.isoformat()],
                        "primaryDocDescription": ["Company raises guidance"],
                        "primaryDocument": ["aapl-20260320.htm"],
                        "accessionNumber": ["0000320193-26-000001"],
                    }
                }
            },
            "fetched_utc": "2026-03-20T14:00:00+00:00",
            "source_confidence_norm": 0.99,
            "schema_confidence_norm": 0.95,
            "freshness_norm": 1.0,
        }

    monkeypatch.setattr(sec_edgar, "_http_json_result", fake_http_json_result)
    monkeypatch.setattr(sec_edgar, "_http_text_result", lambda url, **kwargs: {"ok": False, "error": "archive_timeout"})

    payload, status = sec_edgar.collect_sec_edgar_context(
        symbols=["AAPL"],
        user_agent="test/1.0",
        timeout=1.0,
        pause_seconds=0.0,
        max_runtime_seconds=5.0,
        max_archive_fetches=1,
    )

    assert status["ok"] is True
    assert status["ticker_map_ok"] is True
    assert status["tracked_symbols"] == 1
    assert status["error_count"] == 0
    assert status["warning_count"] == 1
    assert payload["symbol_rows"][0]["symbol"] == "AAPL"
