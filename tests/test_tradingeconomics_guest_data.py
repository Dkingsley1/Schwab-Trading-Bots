from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import scripts.build_behavior_dataset_from_decisions as behavior_ds
import scripts.collect_tradingeconomics_guest_data as te_guest
import scripts.run_shadow_training_loop as loop


def test_derive_macro_backfill_prefers_primary_country_indicator_rows() -> None:
    indicators = [
        {
            "Country": "United States",
            "Category": "Unemployment Rate",
            "Title": "United States Unemployment Rate",
            "LatestValue": 4.2,
            "Unit": "percent",
        },
        {
            "Country": "United States",
            "Category": "Inflation Rate Mom",
            "Title": "United States Inflation Rate Mom",
            "LatestValue": 0.3,
            "Unit": "percent",
        },
        {
            "Country": "United States",
            "Category": "GDP Growth Rate",
            "Title": "United States GDP Growth Rate",
            "LatestValue": 2.4,
            "Unit": "percent",
        },
    ]

    out = te_guest._derive_macro_backfill(indicators, [], country="United States")

    assert out["unemployment_rate_latest"] == 4.2
    assert round(float(out["inflation_mom_ratio"]), 6) == 0.003
    assert round(float(out["gdp_qoq_ratio"]), 6) == 0.024
    assert out["unemployment_source"] == "indicators"


def test_derive_market_breadth_builds_snapshot_from_quote_rows() -> None:
    rows = [
        {"Symbol": "SPY:US", "PercentChange": 1.2, "Volume": 1000},
        {"Symbol": "QQQ:US", "PercentChange": 0.8, "Volume": 1200},
        {"Symbol": "IWM:US", "PercentChange": -0.4, "Volume": 800},
        {"Symbol": "DIA:US", "PercentChange": 0.1, "Volume": 700},
        {"Symbol": "XLK:US", "PercentChange": 1.4, "Volume": 650},
        {"Symbol": "XLF:US", "PercentChange": -0.6, "Volume": 620},
    ]

    out = te_guest._derive_market_breadth(rows)

    assert out["row_count"] == 6
    assert out["advancers"] == 4.0
    assert out["decliners"] == 2.0
    assert out["up_volume"] > out["down_volume"]
    assert out["sector_dispersion"] > 0.0
    assert out["sector_advancers"] == 1.0
    assert out["sector_decliners"] == 1.0
    assert "technology" in out["sector_average_moves"]
    assert out["index_alignment_score"] > 0.0


def test_external_feeds_context_backfills_from_tradingeconomics_latest(tmp_path: Path) -> None:
    ext_root = tmp_path / "exports" / "external_feeds" / "tradingeconomics"
    ext_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime(2026, 3, 18, 18, 0, tzinfo=timezone.utc).isoformat(),
        "status": {"ok": True, "datasets_ok_count": 4},
        "derived": {
            "macro_backfill": {
                "unemployment_rate_latest": 4.1,
                "inflation_mom_ratio": 0.0025,
                "gdp_qoq_ratio": 0.018,
            },
            "calendar_rows": [],
            "news_features": {},
            "market_breadth": {},
            "bond_reference": {},
        },
    }
    (ext_root / "latest.json").write_text(json.dumps(payload), encoding="utf-8")

    context, meta = behavior_ds._external_feeds_context(
        tmp_path,
        datetime(2026, 3, 18, 20, 0, tzinfo=timezone.utc),
    )

    assert context["external_feeds_ok"] == 1.0
    assert context["external_fred_unrate_norm"] > 0.0
    assert context["external_fred_cpi_mom_norm"] != 0.5
    assert context["external_fred_gdp_qoq_norm"] != 0.5
    assert meta["raw"]["tradingeconomics_backfill_used"]["fred_unrate"] is True
    assert meta["provider_ok"]["tradingeconomics"] is True


def test_external_feeds_context_plumbs_context_quality_and_new_sources(tmp_path: Path) -> None:
    ext_root = tmp_path / "exports" / "external_feeds" / "tradingeconomics"
    ext_root.mkdir(parents=True, exist_ok=True)
    (ext_root / "latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime(2026, 3, 18, 18, 0, tzinfo=timezone.utc).isoformat(),
                "status": {"ok": True, "datasets_ok_count": 4},
                "derived": {
                    "macro_backfill": {},
                    "calendar_rows": [],
                    "news_features": {},
                    "market_breadth": {"index_alignment_score": 0.72},
                    "bond_reference": {"curve_regime_score": 0.61},
                },
            }
        ),
        encoding="utf-8",
    )
    external_root = tmp_path / "exports" / "external_context"
    external_root.mkdir(parents=True, exist_ok=True)
    (external_root / "sec_edgar_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime(2026, 3, 18, 19, 0, tzinfo=timezone.utc).isoformat(),
                "derived": {"global_features": {"sec_recent_high_impact_1d_norm": 0.9, "sec_mna_7d_norm": 0.6}},
            }
        ),
        encoding="utf-8",
    )
    (external_root / "extended_quant_context_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime(2026, 3, 18, 19, 5, tzinfo=timezone.utc).isoformat(),
                "derived": {"global_features": {"cboe_put_call_stress_norm": 0.7, "sofr_funding_stress_norm": 0.5}},
            }
        ),
        encoding="utf-8",
    )
    (external_root / "official_macro_context_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime(2026, 3, 18, 19, 10, tzinfo=timezone.utc).isoformat(),
                "derived": {"calendar_features": {"calendar_high_impact_24h_norm": 0.8, "calendar_macro_event_norm": 0.6}},
            }
        ),
        encoding="utf-8",
    )
    (external_root / "schwab_education_context_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime(2026, 3, 18, 19, 15, tzinfo=timezone.utc).isoformat(),
                "derived": {"global_features": {"schwab_education_recent_activity_norm": 0.9}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "data" / "external_context").mkdir(parents=True, exist_ok=True)
    ((tmp_path / "data" / "external_context") / "live_macro_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime(2026, 3, 18, 19, 30, tzinfo=timezone.utc).isoformat(),
                "derived": {
                    "news_features": {"news_source_quality_norm": 0.9, "news_entity_relevance_norm": 0.8},
                    "calendar_features": {"calendar_high_impact_24h_norm": 0.7, "calendar_macro_event_norm": 0.6},
                },
            }
        ),
        encoding="utf-8",
    )
    health_root = tmp_path / "governance" / "health"
    health_root.mkdir(parents=True, exist_ok=True)
    (health_root / "collector_contracts_latest.json").write_text(
        json.dumps(
            {
                "collector_count": 4,
                "required_failure_count": 1,
                "soft_failure_count": 1,
                "average_quality_score": 0.8,
                "rows": [
                    {"name": "market_micro_context", "quality_score": 0.7},
                    {"name": "official_macro_context", "quality_score": 0.9},
                    {"name": "crypto_market_context", "quality_score": 0.6},
                ],
            }
        ),
        encoding="utf-8",
    )
    (health_root / "source_verification_latest.json").write_text(
        json.dumps(
            {
                "overall": {
                    "total_sources": 4,
                    "unverified_sources": ["official_macro_context"],
                    "counts": {"cross_verified": 2},
                }
            }
        ),
        encoding="utf-8",
    )

    context, _meta = behavior_ds._external_feeds_context(
        tmp_path,
        datetime(2026, 3, 18, 20, 0, tzinfo=timezone.utc),
    )

    assert context["live_macro_gate_active_norm"] == 1.0
    assert context["live_macro_gate_confidence_norm"] > 0.0
    assert context["sec_context_signal_norm"] > 0.0
    assert context["extended_quant_signal_norm"] > 0.0
    assert context["official_macro_signal_norm"] > 0.0
    assert context["schwab_education_signal_norm"] > 0.0
    assert context["market_breadth_signal_norm"] > 0.0
    assert context["bond_reference_signal_norm"] > 0.0
    assert context["source_quality_average_score_norm"] == 0.8
    assert context["source_quality_market_micro_score_norm"] == 0.7


def test_external_macro_calendar_proxy_features_merges_tradingeconomics_calendar(monkeypatch) -> None:
    future_ts = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()

    def _fake_external_feeds_context(project_root: Path, now_utc: datetime):
        return (
            {
                "external_fred_unrate_norm": 0.0,
                "external_fred_cpi_mom_norm": 0.5,
                "external_fred_gdp_qoq_norm": 0.5,
                "external_bls_unrate_norm": 0.0,
                "external_bls_cpi_mom_norm": 0.5,
            },
            {
                "fred": {},
                "bls": {},
                "tradingeconomics": {
                    "calendar_rows": [
                        {
                            "Country": "United States",
                            "Event": "FOMC Press Conference",
                            "Date": future_ts,
                            "Importance": "High",
                        }
                    ]
                },
            },
        )

    monkeypatch.setattr(behavior_ds, "_external_feeds_context", _fake_external_feeds_context)
    out = loop._external_macro_calendar_proxy_features(str(PROJECT_ROOT))

    assert out["calendar_feed_available"] > 0.0
    assert out["calendar_high_impact_24h_norm"] > 0.0
    assert out["calendar_fomc_event_norm"] > 0.0


def test_broker_context_env_helpers_prefer_generic_over_broker_specific(monkeypatch) -> None:
    monkeypatch.setenv("SCHWAB_NEWS_CACHE_TTL_SECONDS", "180")
    assert loop._broker_context_env_float("schwab", "NEWS_CACHE_TTL_SECONDS", 90.0) == 180.0

    monkeypatch.setenv("BROKER_NEWS_CACHE_TTL_SECONDS", "75")
    assert loop._broker_context_env_float("schwab", "NEWS_CACHE_TTL_SECONDS", 90.0) == 75.0

    monkeypatch.delenv("BROKER_NEWS_CACHE_TTL_SECONDS")
    monkeypatch.setenv("COINBASE_CALENDAR_CONTEXT_ENABLED", "1")
    assert loop._broker_context_env_flag("coinbase", "CALENDAR_CONTEXT_ENABLED", "0") is True


def test_external_context_merges_accept_official_macro_and_tradingeconomics_payloads() -> None:
    news_base = loop._default_news_features()
    calendar_base = loop.default_calendar_features()

    official_macro_snapshot = {
        "derived": {
            "news_features": {
                "news_available": 1.0,
                "news_source_quality_norm": 0.9,
            },
            "calendar_features": {
                "calendar_high_impact_24h_norm": 0.8,
                "calendar_macro_event_norm": 0.7,
            },
        }
    }
    tradingeconomics_snapshot = {
        "derived": {
            "news_features": {
                "news_items_24h": 0.6,
            },
            "calendar_features": {
                "calendar_feed_available": 1.0,
                "calendar_macro_abs_surprise_norm": 0.55,
            },
        }
    }

    merged_news = loop._merge_external_context_news_features(news_base, official_macro_snapshot, symbol="SPY")
    merged_news = loop._merge_external_context_news_features(merged_news, tradingeconomics_snapshot, symbol="SPY")
    merged_calendar = loop._merge_external_context_calendar_features(calendar_base, official_macro_snapshot)
    merged_calendar = loop._merge_external_context_calendar_features(merged_calendar, tradingeconomics_snapshot)

    assert merged_news["news_available"] == 1.0
    assert merged_news["news_source_quality_norm"] == 0.9
    assert merged_news["news_items_24h"] == 0.6
    assert merged_calendar["calendar_high_impact_24h_norm"] == 0.8
    assert merged_calendar["calendar_macro_event_norm"] == 0.7
    assert merged_calendar["calendar_feed_available"] == 1.0
