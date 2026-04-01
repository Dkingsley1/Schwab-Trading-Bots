from scripts.collect_bls_census_data import _derive_fred_macro_context
from scripts.collect_official_macro_context import (
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
