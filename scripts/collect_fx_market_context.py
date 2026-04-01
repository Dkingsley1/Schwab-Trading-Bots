#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.market_context_features import load_latest_external_context


ECB_FX_HIST_90D_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist-90d.xml"
FED_H10_CURRENT_URL = "https://www.federalreserve.gov/releases/h10/current/default.htm"
ALPHA_VANTAGE_FX_INTRADAY_URL = "https://www.alphavantage.co/query"
TWELVE_DATA_TIME_SERIES_URL = "https://api.twelvedata.com/time_series"
USER_AGENT = "schwab-trading-bot/1.0"

FEATURE_KEYS = [
    "fx_official_data_available",
    "fx_eurusd_level_norm",
    "fx_eurusd_momentum_norm",
    "fx_usdjpy_level_norm",
    "fx_usdjpy_momentum_norm",
    "fx_gbpusd_level_norm",
    "fx_gbpusd_momentum_norm",
    "fx_usd_strength_norm",
    "fx_usd_broad_index_norm",
    "fx_proxy_agreement_norm",
    "fx_risk_on_alignment_norm",
    "fx_crypto_alignment_norm",
    "fx_macro_dispersion_norm",
    "fx_corr_confidence_norm",
]

PAIR_SYMBOLS = ("EURUSD", "USDJPY", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD")
PROXY_SYMBOLS = ("UUP", "FXE", "FXY", "FXB", "FXC", "FXA", "CYB", "SPY", "QQQ", "TLT", "GLD", "BTC-USD", "ETH-USD", "SOL-USD")
PAIR_TWELVE_DATA_SYMBOLS = {
    "EURUSD": "EUR/USD",
    "USDJPY": "USD/JPY",
    "GBPUSD": "GBP/USD",
    "USDCHF": "USD/CHF",
    "USDCAD": "USD/CAD",
    "AUDUSD": "AUD/USD",
}
FED_H10_PAIR_MARKERS = {
    "EURUSD": "*EMU MEMBERS",
    "USDJPY": "JAPAN",
    "GBPUSD": "*UNITED KINGDOM",
    "USDCHF": "SWITZERLAND",
    "USDCAD": "CANADA",
    "AUDUSD": "*AUSTRALIA",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _signed_norm(value: float, scale: float) -> float:
    return _clamp01(0.5 + (float(value) / max(float(scale), 1e-8)))


def _pct_change(current: float | None, previous: float | None) -> float:
    if current is None or previous is None:
        return 0.0
    if abs(previous) <= 1e-12:
        return 0.0
    return (float(current) - float(previous)) / abs(float(previous))


def _http_text(url: str, *, timeout: float = 20.0) -> str:
    req = Request(url=url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _normalize_pair_symbol(raw: Any) -> str:
    token = str(raw or "").strip().upper().replace("/", "").replace("-", "")
    return token if token in PAIR_SYMBOLS else ""


def _twelve_data_time_series(
    *,
    api_key: str,
    pair_symbol: str,
    interval: str,
    outputsize: int,
    timeout: float,
) -> dict[str, Any]:
    normalized = _normalize_pair_symbol(pair_symbol)
    feed_symbol = PAIR_TWELVE_DATA_SYMBOLS.get(normalized, "")
    if not normalized or not feed_symbol:
        return {"ok": False, "error": "unsupported_pair", "pair_symbol": normalized}
    query = (
        f"{TWELVE_DATA_TIME_SERIES_URL}"
        f"?symbol={feed_symbol}&interval={interval}&outputsize={max(int(outputsize), 2)}"
        f"&apikey={api_key}&format=JSON"
    )
    raw = _http_text(query, timeout=timeout)
    try:
        payload = json.loads(raw)
    except Exception:
        return {"ok": False, "error": "invalid_json", "pair_symbol": normalized, "payload": {}}
    if not isinstance(payload, Mapping):
        return {"ok": False, "error": "invalid_payload", "pair_symbol": normalized, "payload": {}}
    if str(payload.get("status", "")).strip().lower() == "error":
        code = str(payload.get("code", "")).strip()
        message = str(payload.get("message", "")).strip() or "twelve_data_error"
        return {
            "ok": False,
            "error": f"{code}:{message}" if code else message,
            "pair_symbol": normalized,
            "payload": dict(payload),
        }
    values = payload.get("values")
    if not isinstance(values, list):
        return {"ok": False, "error": "missing_values", "pair_symbol": normalized, "payload": dict(payload)}
    rows: list[tuple[str, float]] = []
    for row in values:
        if not isinstance(row, Mapping):
            continue
        ts = str(row.get("datetime") or "").strip()
        close_value = _to_float(row.get("close"), math.nan)
        if ts and math.isfinite(close_value) and close_value > 0.0:
            rows.append((ts, close_value))
    rows.sort(key=lambda item: item[0])
    if len(rows) < 2:
        return {"ok": False, "error": "insufficient_intraday_rows", "pair_symbol": normalized, "payload": dict(payload)}
    latest_ts, latest_close = rows[-1]
    previous_ts, previous_close = rows[-2]
    session_ts, session_close = rows[0]
    return {
        "ok": True,
        "error": None,
        "pair_symbol": normalized,
        "feed_symbol": feed_symbol,
        "rows": len(rows),
        "latest_ts": latest_ts,
        "previous_ts": previous_ts,
        "session_ts": session_ts,
        "latest_close": latest_close,
        "previous_close": previous_close,
        "session_close": session_close,
    }


def _alpha_vantage_intraday(
    *,
    api_key: str,
    from_symbol: str,
    to_symbol: str,
    interval: str,
    timeout: float,
) -> dict[str, Any]:
    query = (
        f"{ALPHA_VANTAGE_FX_INTRADAY_URL}"
        f"?function=FX_INTRADAY&from_symbol={from_symbol}&to_symbol={to_symbol}"
        f"&interval={interval}&outputsize=compact&apikey={api_key}"
    )
    raw = _http_text(query, timeout=timeout)
    try:
        payload = json.loads(raw)
    except Exception:
        return {"ok": False, "error": "invalid_json", "payload": {}}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "invalid_payload", "payload": {}}
    if payload.get("Note"):
        return {"ok": False, "error": f"rate_limited:{payload.get('Note')}", "payload": payload}
    if payload.get("Information"):
        return {"ok": False, "error": str(payload.get("Information")), "payload": payload}
    if payload.get("Error Message"):
        return {"ok": False, "error": str(payload.get("Error Message")), "payload": payload}
    series_key = next((key for key in payload.keys() if str(key).startswith("Time Series FX")), "")
    series = payload.get(series_key) if isinstance(payload.get(series_key), Mapping) else {}
    rows: list[tuple[str, float]] = []
    for ts, row in series.items():
        if not isinstance(row, Mapping):
            continue
        close_value = _to_float(row.get("4. close"), math.nan)
        if math.isfinite(close_value) and close_value > 0.0:
            rows.append((str(ts), close_value))
    rows.sort(key=lambda item: item[0])
    if len(rows) < 2:
        return {"ok": False, "error": "insufficient_intraday_rows", "payload": payload}
    latest_ts, latest_close = rows[-1]
    previous_ts, previous_close = rows[-2]
    return {
        "ok": True,
        "error": None,
        "rows": len(rows),
        "latest_ts": latest_ts,
        "previous_ts": previous_ts,
        "latest_close": latest_close,
        "previous_close": previous_close,
    }


def _parse_fed_h10_current(html_text: str) -> dict[str, Any]:
    lines = []
    for raw in str(html_text or "").splitlines():
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            lines.append(text)
    pair_values: dict[str, float] = {}
    previous_values: dict[str, float] = {}
    broad_index = 0.0

    def _extract_tail_floats(text: str) -> list[float]:
        matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        values = []
        for token in matches[-5:]:
            value = _to_float(token, math.nan)
            if math.isfinite(value):
                values.append(value)
        return values

    marker_starts = set(FED_H10_PAIR_MARKERS.values()) | {"1) BROAD"}

    def _extract_marker_values(start_idx: int) -> list[float]:
        # The live Fed H10 page often splits a logical row across multiple lines:
        # marker, currency/unit, then one numeric cell per line. Collect just that row.
        values: list[float] = []
        started_numeric = False
        for idx in range(start_idx, min(len(lines), start_idx + 10)):
            line = lines[idx]
            if idx > start_idx and line in marker_starts:
                break
            numeric_tokens = re.findall(r"[-+]?\d+\.\d+", line)
            if not numeric_tokens:
                if started_numeric:
                    break
                continue
            for token in numeric_tokens:
                value = _to_float(token, math.nan)
                if math.isfinite(value):
                    values.append(value)
            started_numeric = True
            if len(values) >= 5:
                break
        return values[-5:]

    for pair, marker in FED_H10_PAIR_MARKERS.items():
        for idx, line in enumerate(lines):
            if line.startswith(marker):
                values = _extract_marker_values(idx)
                if len(values) >= 2:
                    previous_values[pair] = float(values[-2])
                    pair_values[pair] = float(values[-1])
                break

    for idx, line in enumerate(lines):
        if line.startswith("1) BROAD"):
            values = _extract_marker_values(idx)
            if values:
                broad_index = float(values[-1])
            break

    ok = len(pair_values) >= 3
    return {
        "ok": ok,
        "pair_values": pair_values,
        "previous_pair_values": previous_values,
        "broad_index": broad_index,
        "pair_count": len(pair_values),
        "error": None if ok else "insufficient_h10_pairs",
    }


def _parse_ecb_hist_90d(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    rows: list[dict[str, Any]] = []
    for cube_time in root.findall(".//{*}Cube[@time]"):
        day = str(cube_time.attrib.get("time") or "").strip()
        if not day:
            continue
        rates: dict[str, float] = {}
        for cube in cube_time.findall("{*}Cube[@currency][@rate]"):
            currency = str(cube.attrib.get("currency") or "").strip().upper()
            rate = _to_float(cube.attrib.get("rate"), math.nan)
            if currency and math.isfinite(rate) and rate > 0.0:
                rates[currency] = rate
        if rates:
            rows.append({"date": day, "rates": rates})
    rows.sort(key=lambda row: str(row.get("date") or ""))
    return rows


def _pair_levels(rates: Mapping[str, float]) -> dict[str, float]:
    usd = _to_float(rates.get("USD"), math.nan)
    jpy = _to_float(rates.get("JPY"), math.nan)
    gbp = _to_float(rates.get("GBP"), math.nan)
    chf = _to_float(rates.get("CHF"), math.nan)
    cad = _to_float(rates.get("CAD"), math.nan)
    aud = _to_float(rates.get("AUD"), math.nan)
    out: dict[str, float] = {}
    if math.isfinite(usd) and usd > 0.0:
        out["EURUSD"] = usd
        if math.isfinite(jpy) and jpy > 0.0:
            out["USDJPY"] = jpy / usd
        if math.isfinite(gbp) and gbp > 0.0:
            out["GBPUSD"] = usd / gbp
        if math.isfinite(chf) and chf > 0.0:
            out["USDCHF"] = chf / usd
        if math.isfinite(cad) and cad > 0.0:
            out["USDCAD"] = cad / usd
        if math.isfinite(aud) and aud > 0.0:
            out["AUDUSD"] = usd / aud
    return out


def _latest_pair_history(rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    if not rows:
        return {}, {}
    latest = _pair_levels(rows[-1].get("rates") if isinstance(rows[-1].get("rates"), Mapping) else {})
    previous = {}
    if len(rows) >= 2:
        previous = _pair_levels(rows[-2].get("rates") if isinstance(rows[-2].get("rates"), Mapping) else {})
    return latest, previous


def _latest_market_snapshot() -> dict[str, dict[str, float]]:
    snapshot = load_latest_external_context(PROJECT_ROOT, "market_crypto_correlation")
    derived = snapshot.get("derived") if isinstance(snapshot.get("derived"), Mapping) else {}
    latest_market = derived.get("latest_market") if isinstance(derived.get("latest_market"), Mapping) else {}
    if not latest_market:
        latest_market = derived.get("latest_snapshots") if isinstance(derived.get("latest_snapshots"), Mapping) else {}
    out: dict[str, dict[str, float]] = {}
    for symbol, row in latest_market.items():
        if not isinstance(row, Mapping):
            continue
        token = str(symbol or "").strip().upper()
        if token not in PROXY_SYMBOLS:
            continue
        out[token] = {
            "pct_from_close": _to_float(row.get("pct_from_close"), 0.0),
            "mom_5m": _to_float(row.get("mom_5m"), 0.0),
            "last_price": _to_float(row.get("last_price"), 0.0),
            "ts": _to_float(row.get("ts"), 0.0),
        }
    return out


def _macro_cross_asset_context() -> dict[str, Any]:
    return _safe_load_json(PROJECT_ROOT / "exports" / "external_context" / "macro_cross_asset_latest.json")


def _proxy_agreement(
    pair_changes: Mapping[str, float],
    latest_market: Mapping[str, Mapping[str, float]],
    usd_strength_raw: float,
) -> tuple[float, dict[str, bool]]:
    checks: dict[str, bool] = {}

    def _direction(value: float, *, invert: bool = False) -> bool | None:
        if abs(value) <= 1e-6:
            return None
        return (value < 0.0) if invert else (value > 0.0)

    comparisons = [
        ("EURUSD", "FXE", False),
        ("USDJPY", "FXY", True),
        ("GBPUSD", "FXB", False),
        ("AUDUSD", "FXA", False),
        ("USDCAD", "FXC", True),
    ]
    matches = 0
    total = 0
    for pair, proxy, invert in comparisons:
        if proxy not in latest_market:
            continue
        pair_dir = _direction(_to_float(pair_changes.get(pair), 0.0))
        proxy_dir = _direction(_to_float((latest_market.get(proxy) or {}).get("pct_from_close"), 0.0), invert=invert)
        if pair_dir is None or proxy_dir is None:
            continue
        ok = bool(pair_dir == proxy_dir)
        checks[f"{pair}_{proxy}"] = ok
        total += 1
        matches += 1 if ok else 0

    if "UUP" in latest_market:
        usd_dir = _direction(usd_strength_raw)
        proxy_dir = _direction(_to_float((latest_market.get("UUP") or {}).get("pct_from_close"), 0.0))
        if usd_dir is not None and proxy_dir is not None:
            ok = bool(usd_dir == proxy_dir)
            checks["USD_UUP"] = ok
            total += 1
            matches += 1 if ok else 0

    if total <= 0:
        return 0.0, checks
    return matches / total, checks


def _risk_alignment(latest_market: Mapping[str, Mapping[str, float]], usd_strength_raw: float) -> tuple[float, float]:
    risk_symbols = ("SPY", "QQQ")
    crypto_symbols = ("BTC-USD", "ETH-USD", "SOL-USD")
    risk_values = [_to_float((latest_market.get(symbol) or {}).get("pct_from_close"), 0.0) for symbol in risk_symbols if symbol in latest_market]
    crypto_values = [_to_float((latest_market.get(symbol) or {}).get("pct_from_close"), 0.0) for symbol in crypto_symbols if symbol in latest_market]
    risk_avg = sum(risk_values) / len(risk_values) if risk_values else 0.0
    crypto_avg = sum(crypto_values) / len(crypto_values) if crypto_values else 0.0
    risk_align = _clamp01(0.5 + (-usd_strength_raw * risk_avg * 120.0))
    crypto_align = _clamp01(0.5 + (-usd_strength_raw * crypto_avg * 120.0))
    return risk_align, crypto_align


def collect_fx_market_context(*, timeout: float = 20.0) -> tuple[dict[str, Any], dict[str, Any]]:
    now = datetime.now(timezone.utc)
    warnings: list[str] = []
    twelve_data_enabled = str(os.getenv("FX_MARKET_CONTEXT_TWELVE_DATA_ENABLED", "1")).strip().lower() in {"1", "true", "yes", "on"}
    twelve_data_api_key = str(os.getenv("TWELVE_DATA_API_KEY", "")).strip()
    twelve_data_pairs = [
        pair
        for pair in (
            _normalize_pair_symbol(token)
            for token in str(os.getenv("FX_REALTIME_SYMBOLS", ",".join(PAIR_SYMBOLS))).split(",")
        )
        if pair
    ] or list(PAIR_SYMBOLS)
    twelve_data_interval = str(os.getenv("FX_TWELVE_DATA_INTERVAL", "5min") or "5min").strip() or "5min"
    twelve_data_outputsize = max(int(str(os.getenv("FX_TWELVE_DATA_OUTPUTSIZE", "72") or "72")), 4)
    alpha_vantage_enabled = str(os.getenv("FX_MARKET_CONTEXT_ALPHA_VANTAGE_ENABLED", "0")).strip().lower() in {"1", "true", "yes", "on"}
    alpha_vantage_api_key = str(os.getenv("ALPHA_VANTAGE_API_KEY", "")).strip()
    source_status: dict[str, Any] = {
        "ecb": {"ok": False, "url": ECB_FX_HIST_90D_URL, "rows": 0, "error": None},
        "fed_h10": {"ok": False, "url": FED_H10_CURRENT_URL, "pair_count": 0, "error": None},
        "macro_cross_asset": {"ok": False, "path": str(PROJECT_ROOT / "exports" / "external_context" / "macro_cross_asset_latest.json"), "error": None},
        "market_proxy": {"ok": False, "symbols": 0, "error": None},
    }
    if twelve_data_enabled:
        source_status["twelve_data"] = {
            "ok": False,
            "configured": bool(twelve_data_api_key),
            "pairs_requested": list(twelve_data_pairs),
            "pairs_ok": 0,
            "interval": twelve_data_interval,
            "outputsize": twelve_data_outputsize,
            "error": None,
        }
    if alpha_vantage_enabled:
        source_status["alpha_vantage"] = {
            "ok": False,
            "configured": bool(alpha_vantage_api_key),
            "pair": "EURUSD",
            "interval": "5min",
            "rows": 0,
            "error": None,
        }

    ecb_rows: list[dict[str, Any]] = []
    try:
        ecb_rows = _parse_ecb_hist_90d(_http_text(ECB_FX_HIST_90D_URL, timeout=timeout))
        source_status["ecb"]["ok"] = len(ecb_rows) >= 2
        source_status["ecb"]["rows"] = len(ecb_rows)
        if not source_status["ecb"]["ok"]:
            source_status["ecb"]["error"] = "insufficient_rows"
    except (HTTPError, URLError, TimeoutError, ET.ParseError, OSError) as exc:
        source_status["ecb"]["error"] = str(exc)

    latest_pairs, previous_pairs = _latest_pair_history(ecb_rows)
    pair_changes = {pair: _pct_change(latest_pairs.get(pair), previous_pairs.get(pair)) for pair in PAIR_SYMBOLS}

    alpha_vantage_intraday: dict[str, Any] = {}
    twelve_data_intraday: dict[str, Any] = {}
    fed_h10 = {}
    try:
        fed_h10 = _parse_fed_h10_current(_http_text(FED_H10_CURRENT_URL, timeout=timeout))
        source_status["fed_h10"]["ok"] = bool(fed_h10.get("ok"))
        source_status["fed_h10"]["pair_count"] = int(fed_h10.get("pair_count", 0) or 0)
        source_status["fed_h10"]["error"] = fed_h10.get("error")
        for pair, value in (fed_h10.get("pair_values") or {}).items():
            latest_pairs[str(pair)] = _to_float(value, latest_pairs.get(str(pair), 0.0))
        for pair, value in (fed_h10.get("previous_pair_values") or {}).items():
            previous_pairs[str(pair)] = _to_float(value, previous_pairs.get(str(pair), 0.0))
        pair_changes = {pair: _pct_change(latest_pairs.get(pair), previous_pairs.get(pair)) for pair in PAIR_SYMBOLS}
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        source_status["fed_h10"]["error"] = str(exc)

    if alpha_vantage_enabled and alpha_vantage_api_key:
        alpha_vantage_intraday = _alpha_vantage_intraday(
            api_key=alpha_vantage_api_key,
            from_symbol="EUR",
            to_symbol="USD",
            interval="5min",
            timeout=timeout,
        )
        source_status["alpha_vantage"]["ok"] = bool(alpha_vantage_intraday.get("ok"))
        source_status["alpha_vantage"]["rows"] = int(alpha_vantage_intraday.get("rows", 0) or 0)
        source_status["alpha_vantage"]["error"] = alpha_vantage_intraday.get("error")
        if alpha_vantage_intraday.get("ok"):
            av_change = _pct_change(
                _to_float(alpha_vantage_intraday.get("latest_close"), 0.0),
                _to_float(alpha_vantage_intraday.get("previous_close"), 0.0),
            )
            if abs(av_change) > abs(pair_changes.get("EURUSD", 0.0)):
                pair_changes["EURUSD"] = av_change
            if _to_float(alpha_vantage_intraday.get("latest_close"), 0.0) > 0.0:
                latest_pairs["EURUSD"] = _to_float(alpha_vantage_intraday.get("latest_close"), 0.0)
            if _to_float(alpha_vantage_intraday.get("previous_close"), 0.0) > 0.0:
                previous_pairs["EURUSD"] = _to_float(alpha_vantage_intraday.get("previous_close"), 0.0)
            pair_changes["EURUSD"] = _pct_change(latest_pairs.get("EURUSD"), previous_pairs.get("EURUSD"))
    elif alpha_vantage_enabled:
        source_status["alpha_vantage"]["error"] = "api_key_missing"

    if twelve_data_enabled and twelve_data_api_key:
        twelve_data_errors: list[str] = []
        for pair in twelve_data_pairs:
            result = _twelve_data_time_series(
                api_key=twelve_data_api_key,
                pair_symbol=pair,
                interval=twelve_data_interval,
                outputsize=twelve_data_outputsize,
                timeout=timeout,
            )
            twelve_data_intraday[pair] = dict(result)
            if result.get("ok"):
                latest_pairs[pair] = _to_float(result.get("latest_close"), latest_pairs.get(pair, 0.0))
                previous_pairs[pair] = _to_float(result.get("session_close"), previous_pairs.get(pair, 0.0))
                pair_changes[pair] = _pct_change(latest_pairs.get(pair), previous_pairs.get(pair))
            else:
                twelve_data_errors.append(f"{pair}:{result.get('error')}")
        ok_pairs = sum(1 for row in twelve_data_intraday.values() if bool(row.get("ok")))
        source_status["twelve_data"]["ok"] = ok_pairs > 0
        source_status["twelve_data"]["pairs_ok"] = ok_pairs
        source_status["twelve_data"]["error"] = None if ok_pairs > 0 else (";".join(twelve_data_errors[:3]) or "no_pairs_ok")
    elif twelve_data_enabled:
        source_status["twelve_data"]["error"] = "api_key_missing"

    usd_components = [
        -pair_changes.get("EURUSD", 0.0),
        pair_changes.get("USDJPY", 0.0),
        -pair_changes.get("GBPUSD", 0.0),
        pair_changes.get("USDCHF", 0.0),
        pair_changes.get("USDCAD", 0.0),
        -pair_changes.get("AUDUSD", 0.0),
    ]
    usd_strength_raw = sum(usd_components) / len([x for x in usd_components if math.isfinite(x)]) if usd_components else 0.0
    macro_dispersion_raw = 0.0
    if pair_changes:
        centered = [float(value) for value in pair_changes.values() if math.isfinite(float(value))]
        if centered:
            mean = sum(centered) / len(centered)
            macro_dispersion_raw = math.sqrt(sum((value - mean) ** 2 for value in centered) / max(len(centered), 1))

    macro_cross_asset = _macro_cross_asset_context()
    macro_cross_ok = bool(macro_cross_asset)
    source_status["macro_cross_asset"]["ok"] = macro_cross_ok
    if not macro_cross_ok:
        source_status["macro_cross_asset"]["error"] = "missing_macro_cross_asset_latest"
    dollar_index = 0.0
    if macro_cross_ok:
        cross_asset = macro_cross_asset.get("cross_asset") if isinstance(macro_cross_asset.get("cross_asset"), Mapping) else {}
        dollar_index = _to_float(fed_h10.get("broad_index"), 0.0) or _to_float(cross_asset.get("dollar_index_broad"), 0.0)

    latest_market = _latest_market_snapshot()
    source_status["market_proxy"]["ok"] = len(latest_market) > 0
    source_status["market_proxy"]["symbols"] = len(latest_market)
    if not latest_market:
        source_status["market_proxy"]["error"] = "missing_market_proxy_snapshot"

    proxy_agreement_raw, proxy_checks = _proxy_agreement(pair_changes, latest_market, usd_strength_raw)
    risk_alignment, crypto_alignment = _risk_alignment(latest_market, usd_strength_raw)

    confidence = 0.0
    if source_status["ecb"]["ok"]:
        confidence = 0.55
    if source_status["fed_h10"]["ok"]:
        confidence += 0.15
    if twelve_data_enabled and source_status.get("twelve_data", {}).get("ok"):
        confidence += 0.20
    if alpha_vantage_enabled and source_status.get("alpha_vantage", {}).get("ok"):
        confidence += 0.10
    if source_status["market_proxy"]["ok"]:
        confidence += 0.25
    if macro_cross_ok:
        confidence += 0.20
    confidence = _clamp01(confidence)

    if not source_status["ecb"]["ok"]:
        warnings.append("ecb_fx_feed_unavailable")
    if not source_status["fed_h10"]["ok"]:
        warnings.append("fed_h10_fx_feed_unavailable")
    if twelve_data_enabled and source_status["twelve_data"]["configured"] and not source_status["twelve_data"]["ok"]:
        warnings.append("twelve_data_fx_intraday_unavailable")
    if alpha_vantage_enabled and source_status["alpha_vantage"]["configured"] and not source_status["alpha_vantage"]["ok"]:
        warnings.append("alpha_vantage_fx_intraday_unavailable")
    if not source_status["market_proxy"]["ok"]:
        warnings.append("market_proxy_snapshot_missing")
    elif proxy_agreement_raw <= 0.0:
        warnings.append("proxy_agreement_sparse")

    global_features = {
        "fx_official_data_available": 1.0 if source_status["ecb"]["ok"] else 0.0,
        "fx_eurusd_level_norm": _clamp01(_to_float(latest_pairs.get("EURUSD"), 0.0) / 2.0),
        "fx_eurusd_momentum_norm": _signed_norm(pair_changes.get("EURUSD", 0.0), 0.05),
        "fx_usdjpy_level_norm": _clamp01(_to_float(latest_pairs.get("USDJPY"), 0.0) / 200.0),
        "fx_usdjpy_momentum_norm": _signed_norm(pair_changes.get("USDJPY", 0.0), 0.05),
        "fx_gbpusd_level_norm": _clamp01(_to_float(latest_pairs.get("GBPUSD"), 0.0) / 2.0),
        "fx_gbpusd_momentum_norm": _signed_norm(pair_changes.get("GBPUSD", 0.0), 0.05),
        "fx_usd_strength_norm": _signed_norm(usd_strength_raw, 0.04),
        "fx_usd_broad_index_norm": _clamp01((dollar_index - 70.0) / 60.0) if dollar_index > 0.0 else _signed_norm(usd_strength_raw, 0.04),
        "fx_proxy_agreement_norm": _clamp01(proxy_agreement_raw),
        "fx_risk_on_alignment_norm": _clamp01(risk_alignment),
        "fx_crypto_alignment_norm": _clamp01(crypto_alignment),
        "fx_macro_dispersion_norm": _clamp01(macro_dispersion_raw / 0.03),
        "fx_corr_confidence_norm": confidence,
    }

    symbol_features = {
        symbol: dict(global_features)
        for symbol in set(PROXY_SYMBOLS) | set(PAIR_SYMBOLS) | {"SPY", "QQQ", "BTC-USD", "ETH-USD", "SOL-USD"}
    }
    for symbol, row in symbol_features.items():
        if symbol in {"UUP", "EUO", "YCS"}:
            row["fx_usd_strength_norm"] = _clamp01(min(global_features["fx_usd_strength_norm"] + 0.08, 1.0))
        elif symbol in {"FXE", "FXB", "FXA", "CYB"}:
            row["fx_usd_strength_norm"] = _clamp01(max(global_features["fx_usd_strength_norm"] - 0.08, 0.0))
        elif symbol == "FXY":
            row["fx_usdjpy_momentum_norm"] = _clamp01(1.0 - global_features["fx_usdjpy_momentum_norm"])

    payload = {
        "timestamp_utc": now.isoformat(),
        "provider": "fx_market_context",
        "sources": source_status,
        "derived": {
            "calendar_features": {},
            "news_features": {},
            "global_features": global_features,
            "symbol_features": symbol_features,
            "pair_values": {key: round(_to_float(value), 6) for key, value in latest_pairs.items()},
            "pair_changes": {key: round(_to_float(value), 6) for key, value in pair_changes.items()},
            "pair_intraday_quotes": {
                key: {
                    "ok": bool(value.get("ok")),
                    "latest_ts": value.get("latest_ts"),
                    "latest_close": round(_to_float(value.get("latest_close"), 0.0), 6),
                    "session_close": round(_to_float(value.get("session_close"), 0.0), 6),
                    "rows": int(value.get("rows", 0) or 0),
                }
                for key, value in twelve_data_intraday.items()
                if isinstance(value, Mapping)
            },
            "intraday_reference": {
                "fed_h10": {
                    "ok": bool(fed_h10.get("ok")),
                    "pair_count": int(fed_h10.get("pair_count", 0) or 0),
                },
                "twelve_data": (
                    {
                        "ok": bool(source_status.get("twelve_data", {}).get("ok")),
                        "pairs_ok": int(source_status.get("twelve_data", {}).get("pairs_ok", 0) or 0),
                        "pairs_requested": list(twelve_data_pairs),
                        "interval": twelve_data_interval,
                    }
                    if twelve_data_enabled
                    else {"enabled": False}
                ),
                "alpha_vantage": (
                    {
                        "ok": bool(alpha_vantage_intraday.get("ok")),
                        "latest_ts": alpha_vantage_intraday.get("latest_ts"),
                        "rows": int(alpha_vantage_intraday.get("rows", 0) or 0),
                    }
                    if alpha_vantage_enabled
                    else {"enabled": False}
                ),
            },
            "proxy_checks": proxy_checks,
            "latest_market": latest_market,
        },
    }
    health = {
        "timestamp_utc": now.isoformat(),
        "ok": bool(source_status["ecb"]["ok"] or source_status.get("twelve_data", {}).get("ok")),
        "source_count": len(source_status),
        "ok_source_count": sum(1 for row in source_status.values() if isinstance(row, Mapping) and bool(row.get("ok", False))),
        "official_pairs": len(latest_pairs),
        "proxy_symbols_observed": len(latest_market),
        "proxy_agreement_norm": round(proxy_agreement_raw, 6),
        "direct_forex_execution_supported": False,
        "direct_forex_execution_reason": "schwab_official_api_forex_unverified",
        "warning_count": len(warnings),
        "warnings": warnings,
        "sources": source_status,
    }
    return payload, health


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect free FX context from official feeds plus live proxy markets.")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload, health = collect_fx_market_context(timeout=max(float(args.timeout), 5.0))

    external_path = PROJECT_ROOT / "exports" / "external_context" / "fx_market_context_latest.json"
    health_path = PROJECT_ROOT / "governance" / "health" / "fx_market_context_sync_latest.json"
    _write_json(external_path, payload)
    _write_json(health_path, health)

    if args.json:
        print(json.dumps(health, ensure_ascii=True))
    else:
        print(f"fx_market_context ok={health.get('ok')} proxy_symbols={health.get('proxy_symbols_observed')}")
        print(f"fx_market_context_latest={external_path}")
        print(f"fx_market_context_sync_latest={health_path}")
    return 0 if bool(health.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
