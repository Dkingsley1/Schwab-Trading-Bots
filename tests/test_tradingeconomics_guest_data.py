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
    ]

    out = te_guest._derive_market_breadth(rows)

    assert out["row_count"] == 4
    assert out["advancers"] == 3.0
    assert out["decliners"] == 1.0
    assert out["up_volume"] > out["down_volume"]
    assert out["sector_dispersion"] > 0.0


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
