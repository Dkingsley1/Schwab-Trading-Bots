import json
from datetime import datetime, timezone

from scripts.collect_bls_census_data import (
    DEFAULT_FRED_SERIES_IDS,
    _bea_rss_payload,
    _cached_static_census_payload,
    _derive_central_bank_liquidity_context,
    _derive_fred_macro_context,
    _fred_csv_to_payload,
    _usable_api_key,
)
from scripts.collect_official_macro_context import (
    _cached_federal_reserve_calendar_rows,
    _calendar_rows_from_news,
    _parse_bls_ics,
    _parse_federal_reserve_calendar_text,
    _parse_news_links_from_html,
)


def test_derive_fred_macro_context_extracts_rates_and_cross_asset():
    payload = {
        "timestamp_utc": "2026-03-19T15:06:32+00:00",
        "responses": {
            "DGS2": {"observations": [{"value": "4.01"}]},
            "DGS5": {"observations": [{"value": "4.02"}]},
            "DGS10": {"observations": [{"value": "4.03"}]},
            "DGS30": {"observations": [{"value": "4.04"}]},
            "DFII10": {"observations": [{"value": "1.82"}]},
            "VIXCLS": {"observations": [{"value": "19.2"}]},
            "DTWEXBGS": {"observations": [{"value": "121.4"}]},
            "GOLDAMGBD228NLBM": {"observations": [{"value": "3021.5"}]},
            "DCOILWTICO": {"observations": [{"value": "77.8"}]},
            "BAMLH0A0HYM2": {"observations": [{"value": "366"}]},
        },
    }
    out = _derive_fred_macro_context(payload)
    assert out["treasury_yields"]["10y"] == 4.03
    assert out["treasury_yields"]["real_10y"] == 1.82
    assert out["cross_asset"]["vix"] == 19.2
    assert out["bond_reference_overlay"]["credit_spread_bps"] == 366.0


def test_derive_fred_macro_context_uses_pm_gold_alias():
    payload = {
        "timestamp_utc": "2026-03-20T14:00:00+00:00",
        "responses": {
            "DGS10": {"observations": [{"value": "4.03"}]},
            "GOLDPMGBD228NLBM": {"observations": [{"value": "3017.4"}]},
        },
    }
    out = _derive_fred_macro_context(payload)
    assert out["cross_asset"]["gold_fix"] == 3017.4


def test_central_bank_liquidity_context_covers_balance_sheet_funding_and_stress():
    def observations(*values: float):
        return {
            "observations": [
                {"date": f"2026-08-{14 - index:02d}", "value": str(value)}
                for index, value in enumerate(values)
            ]
        }

    payload = {
        "timestamp_utc": "2026-08-15T00:18:02+00:00",
        "responses": {
            "WALCL": observations(7_000_000, 6_800_000),
            "WRESBAL": observations(3_000_000, 2_950_000),
            "RRPONTSYD": observations(100, 90, 80, 70, 60, 50),
            "RPONTSYD": observations(2, 1),
            "WTREGEN": observations(700_000, 650_000),
            "SWPT": observations(1_000, 900),
            "SOFR": observations(5.4, 5.35),
            "EFFR": observations(5.3, 5.3),
            "OBFR": observations(5.31, 5.31),
            "IORB": observations(5.4, 5.4),
            "DFEDTARL": observations(5.25, 5.25),
            "DFEDTARU": observations(5.5, 5.5),
            "NFCI": observations(0.2, 0.1),
            "ANFCI": observations(0.1, 0.0),
            "STLFSI4": observations(0.5, 0.4),
        },
    }

    out = _derive_central_bank_liquidity_context(payload)

    assert out["coverage"]["required_coverage_ratio"] == 1.0
    assert out["balance_sheet"]["net_liquidity_proxy_millions"] == 6_200_000
    assert out["balance_sheet"]["net_liquidity_proxy_change_millions"] == 100_000
    assert round(out["funding_rates"]["sofr_minus_effr_bps"], 6) == 10.0
    assert out["global_features"]["fed_net_liquidity_impulse_norm"] > 0.5
    assert out["global_features"]["fed_funding_stress_norm"] > 0.5
    assert out["methodology"]["classification"] == "heuristic_market_liquidity_proxy_not_official_accounting_identity"
    assert all(series_id in DEFAULT_FRED_SERIES_IDS for series_id in ("WALCL", "WRESBAL", "RRPONTSYD", "WTREGEN", "SOFR", "EFFR"))


def test_central_bank_liquidity_excludes_future_effective_dates() -> None:
    payload = {
        "timestamp_utc": "2026-08-15T12:00:00+00:00",
        "responses": {
            "IORB": {
                "observations": [
                    {"date": "2026-08-17", "value": "9.99"},
                    {"date": "2026-08-14", "value": "3.65"},
                ]
            }
        },
    }

    out = _derive_central_bank_liquidity_context(payload)

    assert out["funding_rates"]["iorb_percent"] == 3.65
    assert out["coverage"]["latest_observation_dates"]["IORB"] == "2026-08-14"
    assert out["coverage"]["future_observations_excluded"] == {"IORB": ["2026-08-17"]}
    assert out["coverage"]["future_observation_selected"] is False


def test_central_bank_liquidity_marks_stale_required_series_unusable() -> None:
    payload = {
        "timestamp_utc": "2026-08-15T12:00:00+00:00",
        "responses": {
            "WALCL": {"observations": [{"date": "2026-07-01", "value": "6800000"}]},
        },
    }

    out = _derive_central_bank_liquidity_context(payload)

    assert "WALCL" in out["coverage"]["stale_required_series"]
    assert "WALCL" in out["coverage"]["unusable_required_series"]
    assert out["coverage"]["required_coverage_ratio"] < out["coverage"]["required_availability_ratio"]


def test_fred_public_csv_fallback_builds_observations():
    payload = _fred_csv_to_payload(
        "observation_date,GDP\n2026-01-01,28000.5\n2026-04-01,28100.2\n",
        series_id="GDP",
        limit=1,
    )

    assert payload["file_type"] == "csv_public_graph_fallback"
    assert payload["observations"][0]["date"] == "2026-04-01"
    assert payload["observations"][0]["value"] == "28100.2"


def test_bea_rss_fallback_extracts_items():
    payload = _bea_rss_payload(
        """
        <rss><channel>
          <item><title>GDP release</title><link>https://apps.bea.gov/news</link><pubDate>Thu, 25 Jun 2026 12:00:00 GMT</pubDate></item>
        </channel></rss>
        """
    )

    assert payload["items"][0]["title"] == "GDP release"


def test_placeholder_api_key_is_ignored():
    assert _usable_api_key("YOUR_REAL_KEY") == ""
    assert _usable_api_key("replace-me") == ""
    assert _usable_api_key("live-real-token") == "live-real-token"


def test_cached_static_census_payload_accepts_matching_acs_snapshot(tmp_path):
    census_root = tmp_path / "census"
    census_root.mkdir()
    (census_root / "latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-06-15T16:39:40+00:00",
                "request": {
                    "dataset": "2023/acs/acs5",
                    "get": "NAME,B01001_001E",
                    "for": "us:1",
                },
                "response": [["NAME", "B01001_001E", "us"], ["United States", "334000000", "1"]],
            }
        ),
        encoding="utf-8",
    )

    payload = _cached_static_census_payload(
        census_root,
        census_dataset="2023/acs/acs5",
        census_get="NAME,B01001_001E",
        census_for="us:1",
    )

    assert payload is not None
    assert payload["response"][1][1] == "334000000"


def test_parse_bls_ics_extracts_event_rows():
    rows = _parse_bls_ics(
        "\n".join(
            [
                "BEGIN:VCALENDAR",
                "BEGIN:VEVENT",
                "DTSTART:20260320T123000Z",
                "SUMMARY:Consumer Price Index",
                "END:VEVENT",
                "END:VCALENDAR",
            ]
        )
    )
    assert len(rows) == 1
    assert rows[0]["event"] == "Consumer Price Index"
    assert rows[0]["source"] == "Bureau of Labor Statistics"
    assert rows[0]["macro_event_type"] == "inflation"
    assert rows[0]["importance"] == "High"


def test_parse_bls_ics_extracts_tz_start_rows():
    rows = _parse_bls_ics(
        "\n".join(
            [
                "BEGIN:VCALENDAR",
                "BEGIN:VEVENT",
                "DTSTART;TZID=America/New_York:20260321T133000",
                "SUMMARY:Chair Jerome H. Powell remarks",
                "END:VEVENT",
                "END:VCALENDAR",
            ]
        )
    )
    assert len(rows) == 1
    assert rows[0]["datetime"] is not None
    assert "Powell" in rows[0]["event"]


def test_calendar_rows_from_news_keeps_macro_headlines():
    rows = _calendar_rows_from_news(
        [
            {"headline": "Treasury to auction 10-year notes on March 25, 2026", "published": "2026-03-19T00:00:00+00:00", "source": "U.S. Treasury"},
            {"headline": "Unrelated website update", "published": "2026-03-19T00:00:00+00:00", "source": "Other"},
        ]
    )
    assert len(rows) == 1
    assert "auction" in rows[0]["event"].lower()


def test_parse_federal_reserve_calendar_text_extracts_powell_event():
    html = """
    <html><body>
    <div>1:30 p.m.</div>
    <div>Speech - Chair Jerome H. Powell</div>
    <div>Brief Award Acceptance Remarks</div>
    <div>At the American Society for Public Administration Annual Conference</div>
    <div>21</div>
    </body></html>
    """
    rows = _parse_federal_reserve_calendar_text(html, year=2026, month=3)
    assert len(rows) == 1
    assert rows[0]["source"] == "Federal Reserve"
    assert "Powell" in rows[0]["title"]
    assert rows[0]["datetime"] is not None
    assert rows[0]["macro_event_type"] == "fed_speech"
    assert rows[0]["importance"] == "High"
    assert rows[0]["speaker"] == "Jerome H. Powell"


def test_parse_news_links_from_html_extracts_treasury_press_release_rows():
    html = """
    <html><body>
    <a href="/news/press-releases/jy1234">Treasury Announces 10-Year Note Auction for March 25, 2026</a>
    <a href="/about/general-information/role-of-the-treasury">About Treasury</a>
    </body></html>
    """
    rows = _parse_news_links_from_html(html, "treasury", "https://home.treasury.gov/news/press-releases")
    assert len(rows) == 1
    assert rows[0]["source"] == "U.S. Treasury"
    assert "auction" in rows[0]["headline"].lower()
    assert rows[0]["macro_event_type"] == "treasury_auction"
    assert rows[0]["importance"] == "High"


def test_cached_federal_reserve_calendar_rows_reuses_recent_rows(tmp_path):
    payload_path = tmp_path / "official_macro_context_latest.json"
    payload_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-17T12:00:00+00:00",
                "derived": {
                    "calendar_rows": [
                        {"source": "Federal Reserve", "title": "Speech - Chair Powell"},
                        {"source": "U.S. Treasury", "title": "Auction"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    rows = _cached_federal_reserve_calendar_rows(
        payload_path,
        now=datetime(2026, 4, 17, 15, 0, tzinfo=timezone.utc),
        max_age_hours=72.0,
    )

    assert len(rows) == 1
    assert rows[0]["source"] == "Federal Reserve"
