import json
import math

from scripts.collect_fx_market_context import (
    _alpha_vantage_intraday,
    _parse_fed_h10_current,
    _latest_pair_history,
    _normalize_pair_symbol,
    _pair_levels,
    _proxy_agreement,
    _twelve_data_time_series,
)


def test_pair_level_derivation():
    rates = {
        "USD": 1.10,
        "JPY": 165.0,
        "GBP": 0.85,
        "CHF": 0.96,
        "CAD": 1.48,
        "AUD": 1.68,
    }
    pairs = _pair_levels(rates)
    assert math.isclose(pairs["EURUSD"], 1.10, rel_tol=1e-9)
    assert math.isclose(pairs["USDJPY"], 150.0, rel_tol=1e-9)
    assert math.isclose(pairs["GBPUSD"], 1.10 / 0.85, rel_tol=1e-9)
    assert math.isclose(pairs["USDCHF"], 0.96 / 1.10, rel_tol=1e-9)


def test_pair_history_and_proxy_agreement():
    rows = [
        {"date": "2026-03-20", "rates": {"USD": 1.08, "JPY": 162.0, "GBP": 0.84, "CAD": 1.46, "AUD": 1.66}},
        {"date": "2026-03-21", "rates": {"USD": 1.10, "JPY": 170.0, "GBP": 0.85, "CAD": 1.48, "AUD": 1.68}},
    ]
    latest, previous = _latest_pair_history(rows)
    assert latest["EURUSD"] > previous["EURUSD"]
    assert latest["USDJPY"] > previous["USDJPY"]

    pair_changes = {
        "EURUSD": 0.01,
        "USDJPY": 0.02,
        "GBPUSD": 0.01,
        "AUDUSD": 0.01,
        "USDCAD": 0.01,
    }
    latest_market = {
        "FXE": {"pct_from_close": 0.02},
        "FXY": {"pct_from_close": -0.01},
        "FXB": {"pct_from_close": 0.01},
        "FXA": {"pct_from_close": 0.02},
        "FXC": {"pct_from_close": -0.01},
        "UUP": {"pct_from_close": 0.01},
    }
    agreement, checks = _proxy_agreement(pair_changes, latest_market, usd_strength_raw=0.02)
    assert agreement > 0.8
    assert checks["EURUSD_FXE"] is True
    assert checks["USDJPY_FXY"] is True
    assert checks["USD_UUP"] is True


def test_alpha_vantage_intraday_parsing(monkeypatch):
    payload = {
        "Meta Data": {"1. Information": "FX Intraday"},
        "Time Series FX (5min)": {
            "2026-03-23 15:55:00": {
                "1. open": "1.0820",
                "2. high": "1.0822",
                "3. low": "1.0818",
                "4. close": "1.0821",
            },
            "2026-03-23 16:00:00": {
                "1. open": "1.0821",
                "2. high": "1.0826",
                "3. low": "1.0820",
                "4. close": "1.0825",
            },
        },
    }

    monkeypatch.setattr(
        "scripts.collect_fx_market_context._http_text",
        lambda url, timeout=20.0: json.dumps(payload),
    )

    result = _alpha_vantage_intraday(
        api_key="demo",
        from_symbol="EUR",
        to_symbol="USD",
        interval="5min",
        timeout=5.0,
    )
    assert result["ok"] is True
    assert result["rows"] == 2
    assert math.isclose(result["latest_close"], 1.0825, rel_tol=1e-9)
    assert math.isclose(result["previous_close"], 1.0821, rel_tol=1e-9)


def test_twelve_data_time_series_parsing(monkeypatch):
    payload = {
        "meta": {"symbol": "EUR/USD", "interval": "5min"},
        "values": [
            {"datetime": "2026-03-23 21:50:00", "close": "1.0812", "high": "1.0814", "low": "1.0811"},
            {"datetime": "2026-03-23 21:55:00", "close": "1.0816", "high": "1.0818", "low": "1.0814"},
            {"datetime": "2026-03-23 22:00:00", "close": "1.0820", "high": "1.0822", "low": "1.0817"},
        ],
    }

    monkeypatch.setattr(
        "scripts.collect_fx_market_context._http_text",
        lambda url, timeout=20.0: json.dumps(payload),
    )

    result = _twelve_data_time_series(
        api_key="demo",
        pair_symbol="EURUSD",
        interval="5min",
        outputsize=12,
        timeout=5.0,
    )
    assert result["ok"] is True
    assert result["pair_symbol"] == "EURUSD"
    assert result["rows"] == 3
    assert math.isclose(result["latest_close"], 1.0820, rel_tol=1e-9)
    assert math.isclose(result["session_close"], 1.0812, rel_tol=1e-9)


def test_normalize_pair_symbol():
    assert _normalize_pair_symbol("EUR/USD") == "EURUSD"
    assert _normalize_pair_symbol("usd-jpy") == "USDJPY"
    assert _normalize_pair_symbol("SPY") == ""


def test_fed_h10_current_parsing():
    html = """
    Release Date: March 23, 2026
    *AUSTRALIA  DOLLAR  0.7055 0.7100 0.7078 0.7036 0.7033
    CANADA  DOLLAR  1.3683 1.3706 1.3697 1.3721 1.3731
    *EMU MEMBERS  EURO  1.1487 1.1525 1.1513 1.1515 1.1543
    JAPAN  YEN  159.3000 159.0300 159.4800 158.1900 159.2600
    SWITZERLAND  FRANC  0.7887 0.7857 0.7892 0.7924 0.7890
    *UNITED KINGDOM  POUND  1.3302 1.3345 1.3323 1.3361 1.3303
    1) BROAD  JAN06=100 120.0970 119.8328 119.9276 120.1802 120.2757
    """
    parsed = _parse_fed_h10_current(html)
    assert parsed["ok"] is True
    assert parsed["pair_count"] >= 6
    assert math.isclose(parsed["pair_values"]["EURUSD"], 1.1543, rel_tol=1e-9)
    assert math.isclose(parsed["pair_values"]["USDJPY"], 159.2600, rel_tol=1e-9)
    assert math.isclose(parsed["pair_values"]["GBPUSD"], 1.3303, rel_tol=1e-9)
    assert math.isclose(parsed["broad_index"], 120.2757, rel_tol=1e-9)


def test_fed_h10_current_parsing_split_lines():
    html = """
    <tr>
    <th>*EMU MEMBERS</th>
    <td>EURO</td>
    <td>1.1487</td>
    <td>1.1525</td>
    <td>1.1513</td>
    <td>1.1515</td>
    <td>1.1543</td>
    </tr>
    <tr>
    <th>JAPAN</th>
    <td>YEN</td>
    <td>159.3000</td>
    <td>159.0300</td>
    <td>159.4800</td>
    <td>158.1900</td>
    <td>159.2600</td>
    </tr>
    <tr>
    <th>*UNITED KINGDOM</th>
    <td>POUND</td>
    <td>1.3302</td>
    <td>1.3345</td>
    <td>1.3323</td>
    <td>1.3361</td>
    <td>1.3303</td>
    </tr>
    <tr>
    <th>1) BROAD</th>
    <td>JAN06=100</td>
    <td>120.0970</td>
    <td>119.8328</td>
    <td>119.9276</td>
    <td>120.1802</td>
    <td>120.2757</td>
    </tr>
    """
    parsed = _parse_fed_h10_current(html)
    assert parsed["ok"] is True
    assert math.isclose(parsed["pair_values"]["EURUSD"], 1.1543, rel_tol=1e-9)
    assert math.isclose(parsed["pair_values"]["USDJPY"], 159.2600, rel_tol=1e-9)
    assert math.isclose(parsed["pair_values"]["GBPUSD"], 1.3303, rel_tol=1e-9)
    assert math.isclose(parsed["broad_index"], 120.2757, rel_tol=1e-9)
