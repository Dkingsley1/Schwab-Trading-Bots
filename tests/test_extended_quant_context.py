from scripts.collect_extended_quant_context import (
    _derive_cftc_features,
    _derive_sofr_features,
    _discover_first_zip_link,
    _extract_sofr_snapshot,
    _parse_cboe_market_stats_html,
    _parse_cboe_vix_spot,
    _parse_cftc_financial_long_report,
    _parse_nasdaq_threshold_rows,
    _parse_sec_ftd_rows,
)


def test_parse_cftc_financial_long_report_extracts_contract_rows() -> None:
    text = """
    Traders in Financial Futures - Futures Only Positions as of March 17, 2026
    ---------------------------------------------------------------------------
    E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE   (CONTRACTS OF $50 X INDEX)
    CFTC Code #13874A                                                    Open Interest is   120,000
    Positions
        10,000      9,000          0     35,000     12,000      2,000     30,000     18,000      1,000      4,000      3,000          0     41,000     76,000
    Changes from:       March 10, 2026                                   Total Change is:      500

    Percent of Open Interest Represented by Each Category of Trader
           8.3        7.5        0.0       29.2       10.0        1.7       25.0       15.0        0.8        3.3        2.5        0.0       34.2       63.3
    Number of Traders in Each Category                                    Total Traders:       120
    """
    rows = _parse_cftc_financial_long_report(text)
    assert len(rows) == 1
    assert rows[0]["name"].startswith("E-MINI S&P 500")
    assert rows[0]["positions"]["asset_manager_long"] == 35000.0
    assert rows[0]["percent_of_oi"]["leveraged_short"] == 15.0


def test_derive_cftc_features_builds_risk_metrics() -> None:
    rows = [
        {
            "name": "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
            "percent_of_oi": {
                "asset_manager_long": 28.0,
                "asset_manager_short": 10.0,
                "leveraged_long": 21.0,
                "leveraged_short": 11.0,
            },
        },
        {
            "name": "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
            "percent_of_oi": {
                "asset_manager_long": 18.0,
                "asset_manager_short": 8.0,
                "leveraged_long": 9.0,
                "leveraged_short": 14.0,
            },
        },
        {
            "name": "U.S. DOLLAR INDEX - ICE FUTURES U.S.",
            "percent_of_oi": {
                "asset_manager_long": 0.0,
                "asset_manager_short": 0.0,
                "leveraged_long": 16.0,
                "leveraged_short": 7.0,
            },
        },
    ]
    out = _derive_cftc_features(rows)
    assert out["cot_equity_risk_on_norm"] > 0.5
    assert out["cot_usd_bullish_norm"] > 0.5
    assert out["cot_macro_positioning_stress_norm"] > 0.0


def test_extract_sofr_snapshot_reads_rate_and_averages() -> None:
    rate_payload = {"refRates": [{"type": "SOFR", "percentRate": "4.31"}]}
    averages_payload = {"refRates": [{"type": "SOFR", "average30Day": "4.29", "average90Day": "4.18", "average180Day": "4.09", "index": "1.234"}]}
    snapshot = _extract_sofr_snapshot(rate_payload, averages_payload)
    assert snapshot["rate"] == 4.31
    assert snapshot["avg30"] == 4.29
    assert snapshot["avg90"] == 4.18
    assert snapshot["index"] == 1.234


def test_derive_sofr_features_builds_funding_stress() -> None:
    features, overlay = _derive_sofr_features({"rate": 4.35, "avg30": 4.24, "avg90": 4.12, "avg180": 4.01, "index": 1.20})
    assert features["sofr_level_norm"] > 0.0
    assert features["sofr_funding_stress_norm"] > 0.0
    assert overlay["reference_sofr"] == 4.35


def test_parse_cboe_market_stats_html_extracts_section_ratios() -> None:
    html = """
    <html><body>
    <h3>Total</h3>
    <table>
      <tr><th>Time</th><th>Calls</th><th>Puts</th><th>Total</th><th>P/C Ratio</th></tr>
      <tr><td>09:00 AM</td><td>1,000</td><td>900</td><td>1,900</td><td>0.90</td></tr>
      <tr><td>04:15 PM</td><td>2,000</td><td>2,400</td><td>4,400</td><td>1.20</td></tr>
    </table>
    <h3>Index Options</h3>
    <table>
      <tr><td>04:15 PM</td><td>1,500</td><td>2,250</td><td>3,750</td><td>1.50</td></tr>
    </table>
    <h3>Equity Options</h3>
    <table>
      <tr><td>04:15 PM</td><td>1,100</td><td>990</td><td>2,090</td><td>0.90</td></tr>
    </table>
    </body></html>
    """
    out = _parse_cboe_market_stats_html(html)
    assert out["total_ratio"] == 1.2
    assert out["index_ratio"] == 1.5
    assert out["equity_ratio"] == 0.9


def test_parse_cboe_vix_spot_reads_nearby_value() -> None:
    html = """
    <html><body>
      <div>24.95</div>
      <div>VIX Spot Price</div>
    </body></html>
    """
    assert _parse_cboe_vix_spot(html) == 24.95


def test_parse_nasdaq_threshold_rows_extracts_symbols() -> None:
    html = """
    <html><body>
      <h2>Threshold Security List</h2>
      <table>
        <tr><th>Symbol</th><th>Security Name</th><th>Market Category</th><th>Reg SHO Threshold Flag</th><th>Rule 3210</th></tr>
        <tr><td>BOXL</td><td>Boxlight Corp</td><td>S</td><td>Y</td><td>N</td></tr>
        <tr><td>GXAI</td><td>Gaxos AI Inc</td><td>S</td><td>Y</td><td>Y</td></tr>
      </table>
    </body></html>
    """
    rows = _parse_nasdaq_threshold_rows(html)
    assert [row["symbol"] for row in rows] == ["BOXL", "GXAI"]
    assert rows[1]["rule3210"] == "Y"


def test_discover_first_zip_link_prefers_first_zip() -> None:
    html = """
    <html><body>
      <a href="/files/ftd_jan.zip">January 2026</a>
      <a href="/files/ftd_dec.zip">December 2025</a>
    </body></html>
    """
    out = _discover_first_zip_link("https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data", html)
    assert out == "https://www.sec.gov/files/ftd_jan.zip"


def test_parse_sec_ftd_rows_extracts_tracked_symbols() -> None:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "ftd.txt",
            "\n".join(
                [
                    "20260314|000000001|BOXL|125000|Boxlight Corp|2.15",
                    "20260315|000000002|SPY|2500|SPDR S&P 500 ETF|510.10",
                ]
            ),
        )
    rows = _parse_sec_ftd_rows(buffer.getvalue(), {"BOXL"})
    assert len(rows) == 1
    assert rows[0]["symbol"] == "BOXL"
    assert rows[0]["quantity"] == 125000.0
