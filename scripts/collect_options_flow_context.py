#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.collector_transport import attach_collection_confidence, fetch_json


USER_AGENT_DEFAULT = "schwab-trading-bot/1.0"
POLYGON_BASE_URL = "https://api.polygon.io"
UNUSUAL_WHALES_BASE_URL = "https://api.unusualwhales.com"
YAHOO_OPTIONS_BASE_URL = "https://query2.finance.yahoo.com/v7/finance/options"
CBOE_DELAYED_OPTIONS_BASE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options"
LEGACY_ALIAS = "tastytrade_context"
UNUSUAL_WHALES_EXPORT_SCHEMA_VERSION = "uw_options_flow_export.v2"
DEFAULT_UNUSUAL_WHALES_EXPORT_MAX_AGE_SECONDS = 21600
DEFAULT_UNUSUAL_WHALES_EXPORT_MIN_STABLE_SECONDS = 30
UNUSUAL_WHALES_EXPORT_ACTIVE_BASENAMES = (
    "latest_options_flow_export.json",
    "latest_options_flow_export.jsonl",
    "options_flow_export_latest.json",
    "options_flow_export_latest.jsonl",
)
UNUSUAL_WHALES_EXPORT_TEMP_MARKERS = (".tmp", ".partial", ".part", ".writing", ".inprogress")
FEATURE_KEYS = (
    "tasty_iv_rank_norm",
    "tasty_implied_volatility_index_norm",
    "tasty_liquidity_rating_norm",
    "tasty_expected_move_norm",
    "tasty_beta_norm",
    "tasty_watchlist_presence_norm",
    "short_borrow_availability_norm",
    "short_borrow_fee_norm",
    "short_utilization_norm",
    "short_days_to_cover_norm",
    "tasty_dealer_gamma_pressure_norm",
    "tasty_call_wall_proximity_norm",
    "tasty_put_wall_proximity_norm",
    "tasty_max_pain_proximity_norm",
    "tasty_pin_risk_norm",
    "options_iv_skew_norm",
    "options_iv_term_structure_norm",
    "options_gamma_expiry_skew_norm",
    "options_vol_regime_norm",
    "options_surface_change_norm",
    "options_strike_expiry_concentration_change_norm",
    "options_gamma_flip_distance_norm",
    "options_earnings_setup_norm",
    "options_iv_crush_risk_norm",
    "options_assignment_risk_norm",
    "options_zero_dte_regime_norm",
    "options_vol_of_vol_change_norm",
    "options_spread_execution_risk_norm",
)
_OCC_SYMBOL_RE = re.compile(r"^(?P<root>[A-Z]{1,6})(?P<expiry>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")
_DATASET_ALIASES = {
    "iv_rank": "iv_rank",
    "ivrank": "iv_rank",
    "iv_rank_rows": "iv_rank",
    "max_pain": "max_pain",
    "maxpain": "max_pain",
    "max_pain_rows": "max_pain",
    "oi_change": "oi_change",
    "oi-change": "oi_change",
    "oi_change_rows": "oi_change",
    "net_prem": "net_prem_ticks",
    "net_premium": "net_prem_ticks",
    "net_premium_rows": "net_prem_ticks",
    "net_prem_ticks": "net_prem_ticks",
    "net-prem-ticks": "net_prem_ticks",
    "net_prem_tick_rows": "net_prem_ticks",
}
SOURCE_CONTRACTS: dict[str, dict[str, Any]] = {
    "polygon_options_backbone": {
        "required": True,
        "source_confidence_norm": 0.93,
        "schema_confidence_norm": 0.97,
    },
    "unusual_whales_api": {
        "required": False,
        "source_confidence_norm": 0.9,
        "schema_confidence_norm": 0.95,
    },
    "unusual_whales_export": {
        "required": False,
        "source_confidence_norm": 0.84,
        "schema_confidence_norm": 0.9,
    },
    "yahoo_options_chain": {
        "required": False,
        "source_confidence_norm": 0.72,
        "schema_confidence_norm": 0.82,
    },
    "cboe_delayed_options": {
        "required": False,
        "source_confidence_norm": 0.78,
        "schema_confidence_norm": 0.84,
    },
}


def _normalize_symbol(raw: Any) -> str:
    text = str(raw or "").strip().upper()
    if not text:
        return ""
    return "".join(ch for ch in text if ch.isalnum() or ch in {"-", "."})


def _parse_symbols(raw: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in str(raw or "").replace("\n", ",").split(","):
        symbol = _normalize_symbol(token)
        if not symbol or symbol in seen:
            continue
        if symbol.endswith("-USD") or "/" in symbol or "$" in symbol:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _default_symbols() -> list[str]:
    raw = os.getenv("OPTIONS_FLOW_SYMBOLS", "").strip()
    if not raw:
        raw = os.getenv("TASTYTRADE_SYMBOLS", "").strip()
    if not raw:
        raw = ",".join(
            filter(
                None,
                [
                    os.getenv("SHADOW_SYMBOLS_CORE", ""),
                    os.getenv("SHADOW_SYMBOLS_VOLATILE", ""),
                    os.getenv("DIVIDEND_QUALITY_SYMBOLS", ""),
                ],
            )
        )
    symbols = _parse_symbols(raw)
    if not symbols:
        symbols = _parse_symbols("SPY,QQQ,AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,COIN,MSTR,PLTR,AMD,JPM,GS,JNJ,PG,ABBV,SCHD,VIG")
    return symbols[:40]


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _zero_feature_map() -> dict[str, float]:
    return {key: 0.0 for key in FEATURE_KEYS}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _signed_scale(value: float, scale: float) -> float:
    if abs(scale) <= 1e-9:
        return 0.0
    return max(-1.0, min(float(value) / float(scale), 1.0))


def _signed_centered_norm(value: float, scale: float) -> float:
    return _clamp01(0.5 + (0.5 * _signed_scale(float(value), float(scale))))


def _source_contract(name: str) -> dict[str, Any]:
    base = dict(SOURCE_CONTRACTS.get(str(name or "").strip(), {}))
    return {
        "required": bool(base.get("required", False)),
        "source_confidence_norm": float(base.get("source_confidence_norm", 0.8) or 0.8),
        "schema_confidence_norm": float(base.get("schema_confidence_norm", 0.9) or 0.9),
    }


def _fetch_json_result(
    url: str,
    *,
    source_name: str,
    user_agent: str,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    contract = _source_contract(source_name)
    return fetch_json(
        url=url,
        user_agent=user_agent,
        headers=headers,
        timeout=timeout,
        collector_key="options_flow_context",
        source_name=source_name,
        entity_key=url,
        project_root=PROJECT_ROOT,
        source_confidence_norm=float(contract["source_confidence_norm"]),
        schema_confidence_norm=float(contract["schema_confidence_norm"]),
    )


def _http_error_code(raw: str | None) -> int | None:
    text = str(raw or "").strip()
    marker = "HTTP Error "
    idx = text.find(marker)
    if idx < 0:
        return None
    digits = []
    for ch in text[idx + len(marker) :]:
        if ch.isdigit():
            digits.append(ch)
            if len(digits) == 3:
                break
        elif digits:
            break
    if len(digits) != 3:
        return None
    try:
        return int("".join(digits))
    except Exception:
        return None


def _casefold_get(node: Any, *keys: str) -> Any:
    wanted = {str(key or "").strip().lower() for key in keys if str(key or "").strip()}
    if not wanted:
        return None
    stack: list[Any] = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, value in current.items():
                if str(key or "").strip().lower() in wanted:
                    return value
                stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)
    return None


def _extract_rows(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, list):
        return [row for row in node if isinstance(row, dict)]
    if isinstance(node, Mapping):
        for key in ("data", "results", "items", "snapshots"):
            value = node.get(key)
            rows = _extract_rows(value)
            if rows:
                return rows
    return []


def _extract_first_numeric(node: Any, *keys: str) -> float | None:
    value = _casefold_get(node, *keys)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_iso_date(raw: Any) -> date | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").date()
        except Exception:
            return None


def _parse_iso_datetime(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _occ_contract_details(raw: Any) -> tuple[str, date | None, float | None]:
    text = str(raw or "").strip().upper()
    match = _OCC_SYMBOL_RE.match(text)
    if not match:
        return "", None, None
    expiry = None
    try:
        expiry = datetime.strptime(match.group("expiry"), "%y%m%d").date()
    except Exception:
        expiry = None
    strike = None
    try:
        strike = int(match.group("strike")) / 1000.0
    except Exception:
        strike = None
    return match.group("cp"), expiry, strike


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _polygon_url(path: str, *, api_key: str, query: Mapping[str, Any] | None = None) -> str:
    params = {str(key): value for key, value in dict(query or {}).items() if value not in {None, ""}}
    params["apiKey"] = api_key
    return f"{POLYGON_BASE_URL.rstrip('/')}/{path.lstrip('/')}?{urlencode(params, doseq=True)}"


def _polygon_get(
    path: str,
    *,
    api_key: str,
    query: Mapping[str, Any] | None,
    user_agent: str,
    timeout: float,
) -> tuple[Any | None, str | None]:
    if not str(api_key or "").strip():
        return None, "polygon_api_key_missing"
    result = _fetch_json_result(
        _polygon_url(path, api_key=str(api_key).strip(), query=query),
        source_name="polygon_options_backbone",
        user_agent=user_agent,
        timeout=timeout,
    )
    payload = result.get("json")
    err = str(result.get("error") or "") or None
    if err and _http_error_code(err) == 403:
        return None, "polygon_plan_restricted"
    return payload, err


def _unusual_whales_get(
    path: str,
    *,
    api_key: str,
    query: Mapping[str, Any] | None,
    user_agent: str,
    timeout: float,
) -> tuple[Any | None, str | None]:
    if not str(api_key or "").strip():
        return None, "unusual_whales_api_key_missing"
    params = {str(key): value for key, value in dict(query or {}).items() if value not in {None, ""}}
    url = f"{UNUSUAL_WHALES_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    if params:
        url = f"{url}?{urlencode(params, doseq=True)}"
    result = _fetch_json_result(
        url,
        source_name="unusual_whales_api",
        user_agent=user_agent,
        headers={"Authorization": f"Bearer {str(api_key).strip()}"},
        timeout=timeout,
    )
    payload = result.get("json")
    err = str(result.get("error") or "") or None
    return payload, err


def _yahoo_options_url(symbol: str) -> str:
    return f"{YAHOO_OPTIONS_BASE_URL.rstrip('/')}/{_normalize_symbol(symbol)}"


def _option_expiry_date(raw: Any) -> str:
    try:
        value = float(raw)
    except Exception:
        value = 0.0
    if value > 0.0:
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).date().isoformat()
        except Exception:
            pass
    _, expiry, _ = _occ_contract_details(raw)
    return expiry.isoformat() if expiry is not None else ""


def _free_snapshot_from_quote(quote: Mapping[str, Any]) -> dict[str, Any]:
    last_price = _safe_float(
        quote.get("regularMarketPrice")
        or quote.get("postMarketPrice")
        or quote.get("preMarketPrice")
        or quote.get("bid")
        or quote.get("ask"),
        0.0,
    )
    prev_close = _safe_float(quote.get("regularMarketPreviousClose") or quote.get("previousClose"), 0.0)
    high = _safe_float(quote.get("regularMarketDayHigh") or quote.get("dayHigh"), last_price)
    low = _safe_float(quote.get("regularMarketDayLow") or quote.get("dayLow"), last_price)
    volume = _safe_float(quote.get("regularMarketVolume") or quote.get("volume"), 0.0)
    change_pct = _safe_float(quote.get("regularMarketChangePercent") or quote.get("changePercent"), 0.0)
    if change_pct == 0.0 and last_price > 0.0 and prev_close > 0.0:
        change_pct = ((last_price - prev_close) / max(abs(prev_close), 1e-9)) * 100.0
    return {
        "ticker": {
            "lastTrade": {"p": last_price},
            "prevDay": {"c": prev_close},
            "day": {"v": volume, "h": high, "l": low},
            "todaysChangePerc": change_pct,
        }
    }


def _yahoo_option_row(raw: Mapping[str, Any], *, contract_type: str) -> dict[str, Any]:
    contract_symbol = str(raw.get("contractSymbol") or raw.get("symbol") or "")
    _, occ_expiry, occ_strike = _occ_contract_details(contract_symbol)
    expiry = _option_expiry_date(raw.get("expiration") or raw.get("expirationDate") or contract_symbol)
    strike = _safe_float(raw.get("strike"), occ_strike or 0.0)
    return {
        "contract_type": contract_type,
        "option_type": contract_type,
        "option_symbol": contract_symbol,
        "expiration_date": expiry or (occ_expiry.isoformat() if occ_expiry is not None else ""),
        "strike_price": strike,
        "volume": _safe_float(raw.get("volume"), 0.0),
        "open_interest": _safe_float(raw.get("openInterest"), 0.0),
        "implied_volatility": _safe_float(raw.get("impliedVolatility"), 0.0) * 100.0,
        "bid": _safe_float(raw.get("bid"), 0.0),
        "ask": _safe_float(raw.get("ask"), 0.0),
        "last_price": _safe_float(raw.get("lastPrice"), 0.0),
        "source": "Yahoo Finance option chain",
    }


def _collect_yahoo_options_chain(
    symbol: str,
    *,
    user_agent: str,
    timeout: float,
    contract_limit: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    contract = _source_contract("yahoo_options_chain")
    result = fetch_json(
        url=_yahoo_options_url(symbol),
        user_agent=user_agent,
        timeout=timeout,
        collector_key="options_flow_context",
        source_name="yahoo_options_chain",
        entity_key=_normalize_symbol(symbol),
        project_root=PROJECT_ROOT,
        source_confidence_norm=float(contract["source_confidence_norm"]),
        schema_confidence_norm=float(contract["schema_confidence_norm"]),
    )
    payload = result.get("json")
    err = str(result.get("error") or "") or None
    if not bool(result.get("ok", False)) or not isinstance(payload, Mapping):
        return {}, None, [], {
            "ok": False,
            "symbol": _normalize_symbol(symbol),
            "url": _yahoo_options_url(symbol),
            "error": err or "fetch_failed",
            **contract,
            "freshness_norm": 0.0,
        }
    chain_root = payload.get("optionChain") if isinstance(payload.get("optionChain"), Mapping) else {}
    results = chain_root.get("result") if isinstance(chain_root.get("result"), list) else []
    result0 = results[0] if results and isinstance(results[0], Mapping) else {}
    quote = result0.get("quote") if isinstance(result0.get("quote"), Mapping) else {}
    options = result0.get("options") if isinstance(result0.get("options"), list) else []
    first_expiry = options[0] if options and isinstance(options[0], Mapping) else {}
    rows: list[dict[str, Any]] = []
    for raw in (first_expiry.get("calls") if isinstance(first_expiry.get("calls"), list) else []):
        if isinstance(raw, Mapping):
            rows.append(_yahoo_option_row(raw, contract_type="call"))
    for raw in (first_expiry.get("puts") if isinstance(first_expiry.get("puts"), list) else []):
        if isinstance(raw, Mapping):
            rows.append(_yahoo_option_row(raw, contract_type="put"))
    rows = [row for row in rows if row.get("expiration_date") and _safe_float(row.get("strike_price"), 0.0) > 0.0]
    rows = rows[: max(int(contract_limit), 1)]
    iv_values = [_safe_float(row.get("implied_volatility"), 0.0) for row in rows if _safe_float(row.get("implied_volatility"), 0.0) > 0.0]
    iv_payload = None
    if iv_values:
        avg_iv = sum(iv_values) / max(len(iv_values), 1)
        iv_payload = {"iv_rank": min(avg_iv * 1.25, 100.0), "implied_volatility": avg_iv}
    status = {
        "ok": bool(rows),
        "symbol": _normalize_symbol(symbol),
        "url": _yahoo_options_url(symbol),
        "contract_count": len(rows),
        "expiration_count": len(result0.get("expirationDates") or []),
        "error": None if rows else "no_option_rows",
        **contract,
        "freshness_norm": 1.0 if rows else 0.0,
        "fetched_utc": datetime.now(timezone.utc).isoformat() if rows else "",
    }
    return _free_snapshot_from_quote(quote), iv_payload, rows, status


def _cboe_option_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    contract_symbol = str(raw.get("option") or raw.get("symbol") or raw.get("option_symbol") or "")
    cp, occ_expiry, occ_strike = _occ_contract_details(contract_symbol)
    raw_type = str(raw.get("option_type") or raw.get("type") or cp).strip().lower()
    contract_type = "put" if raw_type.startswith("p") else "call" if raw_type.startswith("c") else ""
    expiry = str(raw.get("expiration_date") or raw.get("expiration") or "").strip()
    if not expiry and occ_expiry is not None:
        expiry = occ_expiry.isoformat()
    strike = _safe_float(raw.get("strike") or raw.get("strike_price"), occ_strike or 0.0)
    return {
        "contract_type": contract_type,
        "option_type": contract_type,
        "option_symbol": contract_symbol,
        "expiration_date": expiry,
        "strike_price": strike,
        "volume": _safe_float(raw.get("volume"), 0.0),
        "open_interest": _safe_float(raw.get("open_interest") or raw.get("openInterest"), 0.0),
        "implied_volatility": _safe_float(raw.get("iv") or raw.get("implied_volatility"), 0.0),
        "bid": _safe_float(raw.get("bid"), 0.0),
        "ask": _safe_float(raw.get("ask"), 0.0),
        "last_price": _safe_float(raw.get("last") or raw.get("last_price"), 0.0),
        "source": "Cboe delayed option quotes",
    }


def _collect_cboe_options_chain(
    symbol: str,
    *,
    user_agent: str,
    timeout: float,
    contract_limit: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    normalized = _normalize_symbol(symbol)
    url = f"{CBOE_DELAYED_OPTIONS_BASE_URL.rstrip('/')}/{normalized}.json"
    contract = _source_contract("cboe_delayed_options")
    result = fetch_json(
        url=url,
        user_agent=user_agent,
        timeout=timeout,
        collector_key="options_flow_context",
        source_name="cboe_delayed_options",
        entity_key=normalized,
        project_root=PROJECT_ROOT,
        source_confidence_norm=float(contract["source_confidence_norm"]),
        schema_confidence_norm=float(contract["schema_confidence_norm"]),
    )
    payload = result.get("json")
    err = str(result.get("error") or "") or None
    if not bool(result.get("ok", False)) or not isinstance(payload, Mapping):
        return {}, None, [], {"ok": False, "symbol": normalized, "url": url, "error": err or "fetch_failed", **contract, "freshness_norm": 0.0}
    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else payload
    raw_options = data.get("options") if isinstance(data.get("options"), list) else []
    rows = [_cboe_option_row(row) for row in raw_options if isinstance(row, Mapping)]
    rows = [row for row in rows if row.get("contract_type") and row.get("expiration_date") and _safe_float(row.get("strike_price"), 0.0) > 0.0]
    rows = rows[: max(int(contract_limit), 1)]
    quote = {
        "regularMarketPrice": data.get("current_price") or data.get("last") or data.get("last_price"),
        "regularMarketPreviousClose": data.get("prev_day_close") or data.get("previous_close"),
        "regularMarketDayHigh": data.get("high"),
        "regularMarketDayLow": data.get("low"),
        "regularMarketVolume": data.get("volume"),
    }
    iv_values = [_safe_float(row.get("implied_volatility"), 0.0) for row in rows if _safe_float(row.get("implied_volatility"), 0.0) > 0.0]
    iv_payload = {"iv_rank": min((sum(iv_values) / max(len(iv_values), 1)) * 1.25, 100.0), "implied_volatility": sum(iv_values) / max(len(iv_values), 1)} if iv_values else None
    return _free_snapshot_from_quote(quote), iv_payload, rows, {
        "ok": bool(rows),
        "symbol": normalized,
        "url": url,
        "contract_count": len(rows),
        "error": None if rows else "no_option_rows",
        **contract,
        "freshness_norm": 1.0 if rows else 0.0,
        "fetched_utc": datetime.now(timezone.utc).isoformat() if rows else "",
    }


def _normalize_dataset_name(raw: Any) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(raw or "").strip().lower()).strip("_")
    return _DATASET_ALIASES.get(key, "")


def _infer_export_dataset(row: Mapping[str, Any]) -> str:
    explicit = _normalize_dataset_name(
        row.get("dataset")
        or row.get("kind")
        or row.get("type")
        or row.get("report_type")
        or row.get("table")
        or row.get("series")
    )
    if explicit:
        return explicit
    keys = {str(key or "").strip().lower() for key in row.keys()}
    if {"iv_rank", "ivrank", "rank"} & keys:
        return "iv_rank"
    if {"max_pain", "maxpain"} & keys:
        return "max_pain"
    if {"oi_change", "curr_oi", "volume"} & keys or any("oi_change" in key for key in keys):
        return "oi_change"
    if {"net_premium", "net_call_premium", "net_put_premium", "call_premium", "put_premium"} & keys:
        return "net_prem_ticks"
    return ""


def _dataset_symbol(row: Mapping[str, Any]) -> str:
    return _normalize_symbol(row.get("symbol") or row.get("ticker") or row.get("underlying_symbol"))


def _coerce_dataset_payload(dataset: str, value: Any) -> Any:
    if isinstance(value, Mapping):
        rows = _extract_rows(value)
        if dataset in {"oi_change", "net_prem_ticks"} and rows:
            return {"rows": rows}
        if len(rows) == 1:
            return dict(rows[0])
        if rows:
            return {"rows": rows}
        return dict(value)
    if isinstance(value, list):
        rows = [dict(row) for row in value if isinstance(row, Mapping)]
        if dataset in {"oi_change", "net_prem_ticks"}:
            return {"rows": rows}
        if len(rows) == 1:
            return rows[0]
        return {"rows": rows}
    return value


def _merge_symbol_dataset(symbols: dict[str, dict[str, Any]], symbol: str, dataset: str, value: Any) -> None:
    normalized_symbol = _normalize_symbol(symbol)
    normalized_dataset = _normalize_dataset_name(dataset)
    if not normalized_symbol or not normalized_dataset:
        return
    node = symbols.setdefault(normalized_symbol, {})
    coerced = _coerce_dataset_payload(normalized_dataset, value)
    existing = node.get(normalized_dataset)
    if normalized_dataset in {"oi_change", "net_prem_ticks"}:
        merged_rows: list[dict[str, Any]] = []
        for candidate in (_extract_rows(existing), _extract_rows(coerced)):
            for row in candidate:
                if row not in merged_rows:
                    merged_rows.append(dict(row))
        node[normalized_dataset] = {"rows": merged_rows}
        return
    if existing is None:
        node[normalized_dataset] = coerced
        return
    existing_rows = _extract_rows(existing)
    coerced_rows = _extract_rows(coerced)
    if coerced_rows and not existing_rows:
        node[normalized_dataset] = coerced
        return
    if existing_rows and coerced_rows:
        merged_rows = [dict(row) for row in existing_rows]
        for row in coerced_rows:
            if row not in merged_rows:
                merged_rows.append(dict(row))
        node[normalized_dataset] = {"rows": merged_rows}
        return
    node[normalized_dataset] = coerced


def _looks_like_symbol_map(node: Any) -> bool:
    if not isinstance(node, Mapping) or not node:
        return False
    sample = list(node.items())[:5]
    match_count = 0
    for key, value in sample:
        if _normalize_symbol(key) and isinstance(value, (Mapping, list)):
            match_count += 1
    return match_count == len(sample)


def _extract_export_timestamp(payload: Mapping[str, Any], path: Path | None = None) -> tuple[str, datetime | None]:
    for key in (
        "timestamp_utc",
        "generated_at_utc",
        "generated_utc",
        "exported_at_utc",
        "created_at_utc",
        "updated_at_utc",
        "as_of_utc",
        "as_of",
    ):
        parsed = _parse_iso_datetime(_casefold_get(payload, key))
        if parsed is not None:
            return key, parsed
    if path is not None:
        try:
            return "file_mtime", datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            return "", None
    return "", None


def _normalize_unusual_whales_export_payload(raw_payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(raw_payload, list):
        raw_payload = {"rows": raw_payload}
    if not isinstance(raw_payload, Mapping):
        return {}, {
            "adapter": "",
            "schema_version": "",
            "schema_status": "missing",
            "dataset_symbol_counts": {},
            "row_count": 0,
            "issues": ["export_root_not_mapping"],
        }

    payload = dict(raw_payload)
    symbols: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    adapter = ""
    schema_version = str(payload.get("schema_version") or payload.get("schema") or "").strip()
    schema_status = "canonical" if schema_version == UNUSUAL_WHALES_EXPORT_SCHEMA_VERSION else ("legacy" if schema_version else "unversioned")

    top_symbols = payload.get("symbols")
    if isinstance(top_symbols, Mapping):
        adapter = "canonical_symbols"
        for raw_symbol, datasets in top_symbols.items():
            symbol = _normalize_symbol(raw_symbol)
            if not symbol or not isinstance(datasets, Mapping):
                continue
            for dataset_key, dataset_value in datasets.items():
                normalized_dataset = _normalize_dataset_name(dataset_key)
                if normalized_dataset:
                    _merge_symbol_dataset(symbols, symbol, normalized_dataset, dataset_value)

    sectioned = False
    for raw_key, raw_value in payload.items():
        dataset = _normalize_dataset_name(raw_key)
        if not dataset:
            continue
        sectioned = True
        adapter = adapter or "sectioned_datasets"
        if _looks_like_symbol_map(raw_value):
            for raw_symbol, dataset_value in dict(raw_value).items():
                _merge_symbol_dataset(symbols, str(raw_symbol), dataset, dataset_value)
            continue
        if isinstance(raw_value, list):
            for row in raw_value:
                if not isinstance(row, Mapping):
                    continue
                symbol = _dataset_symbol(row)
                if symbol:
                    _merge_symbol_dataset(symbols, symbol, dataset, dict(row))
            continue
        if isinstance(raw_value, Mapping):
            rows = _extract_rows(raw_value)
            if rows:
                for row in rows:
                    symbol = _dataset_symbol(row)
                    if symbol:
                        _merge_symbol_dataset(symbols, symbol, dataset, dict(row))
                continue
            symbol = _dataset_symbol(raw_value)
            if symbol:
                _merge_symbol_dataset(symbols, symbol, dataset, dict(raw_value))

    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    if rows:
        adapter = adapter or "row_inference"
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            dataset = _infer_export_dataset(row)
            symbol = _dataset_symbol(row)
            if dataset and symbol:
                _merge_symbol_dataset(symbols, symbol, dataset, dict(row))

    if not adapter and not sectioned and isinstance(payload, Mapping):
        adapter = "single_symbol_payload" if any(_normalize_dataset_name(key) for key in payload.keys()) else ""

    if not symbols:
        issues.append("no_recognized_unusual_whales_export_rows")
    elif schema_status == "unversioned":
        issues.append("schema_version_missing")

    dataset_symbol_counts = {
        dataset: sum(1 for node in symbols.values() if dataset in node)
        for dataset in ("iv_rank", "max_pain", "oi_change", "net_prem_ticks")
    }
    row_count = 0
    for node in symbols.values():
        for dataset in ("iv_rank", "max_pain", "oi_change", "net_prem_ticks"):
            value = node.get(dataset)
            if value is None:
                continue
            rows = _extract_rows(value)
            row_count += len(rows) if rows else 1

    normalized_payload = {
        "schema_version": UNUSUAL_WHALES_EXPORT_SCHEMA_VERSION,
        "source_schema_version": schema_version,
        "generated_at_utc": str(
            payload.get("generated_at_utc")
            or payload.get("generated_utc")
            or payload.get("timestamp_utc")
            or payload.get("exported_at_utc")
            or ""
        ).strip(),
        "symbols": symbols,
    }
    return normalized_payload, {
        "adapter": adapter,
        "schema_version": schema_version,
        "schema_status": schema_status,
        "dataset_symbol_counts": dataset_symbol_counts,
        "row_count": row_count,
        "issues": issues,
    }


def _stable_export_candidate(path: Path, *, now: datetime, min_stable_seconds: int) -> tuple[bool, list[str]]:
    issues: list[str] = []
    lowered = path.name.lower()
    if any(marker in lowered for marker in UNUSUAL_WHALES_EXPORT_TEMP_MARKERS):
        issues.append("candidate_marked_partial")
        return False, issues
    try:
        stat = path.stat()
    except Exception:
        issues.append("candidate_stat_failed")
        return False, issues
    age_seconds = max((now - datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)).total_seconds(), 0.0)
    if age_seconds < max(int(min_stable_seconds), 0):
        issues.append("candidate_not_stable_yet")
        return False, issues
    if stat.st_size <= 0:
        issues.append("candidate_empty")
        return False, issues
    return True, issues


def _inspect_export_candidate(
    path: Path,
    *,
    now: datetime,
    max_age_seconds: int,
    min_stable_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stable, stable_issues = _stable_export_candidate(path, now=now, min_stable_seconds=min_stable_seconds)
    base_meta: dict[str, Any] = {
        "candidate_path": str(path),
        "usable": False,
        "issues": list(stable_issues),
        "format": path.suffix.lower().lstrip("."),
        "row_count": 0,
        "rejected_row_count": 0,
        "schema_version": "",
        "schema_status": "missing",
        "adapter": "",
        "dataset_symbol_counts": {},
        "size_bytes": _safe_int(path.stat().st_size if path.exists() else 0, 0),
    }
    if not stable:
        return {}, base_meta

    parse_rejected = 0
    raw_payload: Any = {}
    try:
        if path.suffix.lower() == ".jsonl":
            rows: list[dict[str, Any]] = []
            for raw in path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parse_rejected += 1
                    continue
                if isinstance(parsed, dict):
                    rows.append(parsed)
                else:
                    parse_rejected += 1
            raw_payload = {"rows": rows}
        else:
            raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        base_meta["issues"].append(f"parse_failed:{exc}")
        return {}, base_meta

    normalized_payload, normalized_meta = _normalize_unusual_whales_export_payload(raw_payload)
    ts_key, export_ts = _extract_export_timestamp(raw_payload if isinstance(raw_payload, Mapping) else normalized_payload, path=path)
    age_seconds = None
    if export_ts is not None:
        age_seconds = max((now - export_ts).total_seconds(), 0.0)
        if age_seconds > max(int(max_age_seconds), 1):
            normalized_meta.setdefault("issues", []).append("stale_export")
    normalized_payload["generated_at_utc"] = export_ts.isoformat() if export_ts is not None else str(normalized_payload.get("generated_at_utc") or "")
    merged_issues = [str(item) for item in list(base_meta["issues"]) + list(normalized_meta.get("issues") or []) if str(item).strip()]
    meta = {
        **base_meta,
        "usable": bool(normalized_payload.get("symbols")) and "stale_export" not in merged_issues,
        "issues": merged_issues,
        "row_count": _safe_int(normalized_meta.get("row_count"), 0),
        "rejected_row_count": int(parse_rejected),
        "schema_version": str(normalized_meta.get("schema_version") or ""),
        "schema_status": str(normalized_meta.get("schema_status") or "missing"),
        "adapter": str(normalized_meta.get("adapter") or ""),
        "dataset_symbol_counts": dict(normalized_meta.get("dataset_symbol_counts") or {}),
        "timestamp_source": ts_key,
        "timestamp_utc": export_ts.isoformat() if export_ts is not None else "",
        "age_seconds": round(float(age_seconds), 3) if age_seconds is not None else None,
        "fresh": bool(age_seconds is not None and age_seconds <= max(int(max_age_seconds), 1)),
        "symbol_count": len(normalized_payload.get("symbols") or {}),
    }
    return (normalized_payload if bool(meta["usable"]) else {}), meta


def _rank_export_candidates(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    candidates = [candidate for candidate in path.iterdir() if candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl"}]
    return sorted(
        candidates,
        key=lambda candidate: (
            -_safe_float(candidate.stat().st_mtime if candidate.exists() else 0.0, 0.0),
            0 if candidate.name.lower() in UNUSUAL_WHALES_EXPORT_ACTIVE_BASENAMES else 1,
            candidate.name.lower(),
        ),
    )


def _load_unusual_whales_export(
    path: str | None,
    *,
    now: datetime,
    max_age_seconds: int,
    min_stable_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    inspection: dict[str, Any] = {
        "configured": bool(str(path or "").strip()),
        "source_path": str(Path(str(path)).expanduser()) if str(path or "").strip() else "",
        "selected_candidate": "",
        "path_mode": "unset",
        "exists": False,
        "candidate_count": 0,
        "ignored_candidates": [],
        "issues": [],
        "usable": False,
        "fresh": False,
        "age_seconds": None,
        "timestamp_utc": "",
        "timestamp_source": "",
        "format": "",
        "schema_version": "",
        "schema_status": "missing",
        "adapter": "",
        "symbol_count": 0,
        "max_age_seconds": max(int(max_age_seconds), 1),
        "min_stable_seconds": max(int(min_stable_seconds), 0),
        "dataset_symbol_counts": {},
        "row_count": 0,
        "rejected_row_count": 0,
        "size_bytes": 0,
    }
    if not inspection["configured"]:
        inspection["issues"].append("export_path_not_configured")
        return {}, inspection

    export_path = Path(str(path)).expanduser()
    inspection["source_path"] = str(export_path)
    inspection["exists"] = export_path.exists()
    inspection["path_mode"] = "directory" if export_path.is_dir() else "file"
    if not export_path.exists():
        inspection["issues"].append("export_path_missing")
        return {}, inspection

    if export_path.is_dir():
        candidates = _rank_export_candidates(export_path)
        inspection["candidate_count"] = len(candidates)
        for candidate in candidates:
            payload, candidate_meta = _inspect_export_candidate(
                candidate,
                now=now,
                max_age_seconds=max_age_seconds,
                min_stable_seconds=min_stable_seconds,
            )
            if payload and bool(candidate_meta.get("usable", False)):
                inspection.update(candidate_meta)
                inspection["selected_candidate"] = str(candidate)
                inspection["usable"] = True
                inspection["format"] = str(candidate_meta.get("format") or "")
                return payload, inspection
            inspection["ignored_candidates"].append(
                {
                    "path": str(candidate),
                    "issues": list(candidate_meta.get("issues") or []),
                }
            )
        inspection["issues"].append("no_usable_export_candidates")
        return {}, inspection

    payload, candidate_meta = _inspect_export_candidate(
        export_path,
        now=now,
        max_age_seconds=max_age_seconds,
        min_stable_seconds=min_stable_seconds,
    )
    inspection.update(candidate_meta)
    inspection["selected_candidate"] = str(export_path)
    inspection["usable"] = bool(candidate_meta.get("usable", False))
    inspection["format"] = str(candidate_meta.get("format") or "")
    if not inspection["usable"] and not inspection["issues"]:
        inspection["issues"].append("export_candidate_unusable")
    return payload, inspection


def inspect_unusual_whales_export(
    path: str | None,
    *,
    max_age_seconds: int = DEFAULT_UNUSUAL_WHALES_EXPORT_MAX_AGE_SECONDS,
    min_stable_seconds: int = DEFAULT_UNUSUAL_WHALES_EXPORT_MIN_STABLE_SECONDS,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _load_unusual_whales_export(
        path,
        now=now or datetime.now(timezone.utc),
        max_age_seconds=max(int(max_age_seconds), 1),
        min_stable_seconds=max(int(min_stable_seconds), 0),
    )


def promote_unusual_whales_export(
    source_path: str | None,
    *,
    promoted_path: str | Path | None = None,
    max_age_seconds: int = DEFAULT_UNUSUAL_WHALES_EXPORT_MAX_AGE_SECONDS,
    min_stable_seconds: int = DEFAULT_UNUSUAL_WHALES_EXPORT_MIN_STABLE_SECONDS,
    now: datetime | None = None,
) -> dict[str, Any]:
    payload, inspection = inspect_unusual_whales_export(
        source_path,
        max_age_seconds=max_age_seconds,
        min_stable_seconds=min_stable_seconds,
        now=now,
    )
    result = {
        "selected_candidate": str(inspection.get("selected_candidate") or ""),
        "source_path": str(inspection.get("source_path") or ""),
        "usable": bool(inspection.get("usable", False)),
        "promoted": False,
        "promoted_path": "",
        "issues": list(inspection.get("issues") or []),
    }
    source_root = Path(str(source_path)).expanduser() if str(source_path or "").strip() else None
    target_path: Path | None = None
    if promoted_path:
        target_path = Path(promoted_path).expanduser()
    elif source_root is not None and source_root.is_dir():
        target_path = source_root / "latest_options_flow_export.json"
    elif source_root is not None and source_root.is_file():
        target_path = source_root if source_root.name == "latest_options_flow_export.json" else source_root.parent / "latest_options_flow_export.json"
    if not payload or target_path is None:
        return result
    _write_json(target_path, payload)
    result.update(
        {
            "promoted": True,
            "promoted_path": str(target_path),
        }
    )
    return result


def _export_symbol_payload(export_payload: Mapping[str, Any], symbol: str, key: str) -> Any:
    symbols = export_payload.get("symbols") if isinstance(export_payload.get("symbols"), Mapping) else export_payload
    if isinstance(symbols, Mapping):
        node = symbols.get(symbol) or symbols.get(symbol.upper()) or symbols.get(symbol.lower())
        if isinstance(node, Mapping):
            return node.get(key)
    rows = export_payload.get("rows") if isinstance(export_payload.get("rows"), list) else []
    matching = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        row_symbol = _normalize_symbol(row.get("symbol") or row.get("ticker") or row.get("underlying_symbol"))
        if row_symbol == symbol:
            matching.append(dict(row))
    return {"rows": matching} if matching else None


def _polygon_snapshot_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    ticker = payload.get("ticker") if isinstance(payload.get("ticker"), Mapping) else payload
    last_price = _extract_first_numeric(ticker, "p", "price", "close", "c")
    prev_close = _extract_first_numeric(_casefold_get(ticker, "prevDay", "prev_day"), "c", "close")
    day_volume = _extract_first_numeric(_casefold_get(ticker, "day"), "v", "volume") or 0.0
    day_high = _extract_first_numeric(_casefold_get(ticker, "day"), "h", "high")
    day_low = _extract_first_numeric(_casefold_get(ticker, "day"), "l", "low")
    change_pct = _extract_first_numeric(ticker, "todaysChangePerc", "todays_change_perc", "change_percent")
    if change_pct is None and last_price is not None and prev_close not in {None, 0.0}:
        change_pct = ((last_price - float(prev_close)) / abs(float(prev_close))) * 100.0
    intraday_range = 0.0
    if last_price not in {None, 0.0} and day_high is not None and day_low is not None:
        intraday_range = max(float(day_high) - float(day_low), 0.0) / max(abs(float(last_price)), 1e-9)
    return {
        "last_price": float(last_price or 0.0),
        "prev_close": float(prev_close or 0.0),
        "day_volume": float(day_volume),
        "change_pct": float(change_pct or 0.0),
        "intraday_range": float(intraday_range),
    }


def _polygon_contract_metrics(rows: list[dict[str, Any]], *, today: date, spot_price: float) -> dict[str, float]:
    if not rows:
        return {
            "contract_density_norm": 0.0,
            "zero_dte_presence_norm": 0.0,
            "near_term_presence_norm": 0.0,
            "call_share": 0.5,
            "put_share": 0.5,
            "nearest_expiry_days": 0.0,
        }
    call_count = 0
    put_count = 0
    zero_dte = 0
    near_term = 0
    expiry_days: list[int] = []
    strike_distances: list[float] = []
    for row in rows:
        contract_type = str(row.get("contract_type") or row.get("option_type") or "").strip().lower()
        if contract_type == "call":
            call_count += 1
        elif contract_type == "put":
            put_count += 1
        expiry = _parse_iso_date(row.get("expiration_date") or row.get("expiry"))
        if expiry is not None:
            days = max((expiry - today).days, 0)
            expiry_days.append(days)
            if days == 0:
                zero_dte += 1
            if days <= 7:
                near_term += 1
        strike = _safe_float(row.get("strike_price"), 0.0)
        if strike > 0.0 and spot_price > 0.0:
            strike_distances.append(abs(strike - spot_price) / max(abs(spot_price), 1e-9))
    total = max(len(rows), 1)
    return {
        "contract_density_norm": _clamp01(total / 250.0),
        "zero_dte_presence_norm": _clamp01(zero_dte / max(total, 1)),
        "near_term_presence_norm": _clamp01(near_term / max(total, 1)),
        "call_share": call_count / max(call_count + put_count, 1),
        "put_share": put_count / max(call_count + put_count, 1),
        "nearest_expiry_days": float(min(expiry_days) if expiry_days else 0.0),
        "avg_strike_distance": float(sum(strike_distances) / max(len(strike_distances), 1)),
    }


def _unusual_whales_rows(payload: Any) -> list[dict[str, Any]]:
    rows = _extract_rows(payload)
    return rows if rows else ([dict(payload)] if isinstance(payload, Mapping) else [])


def _compute_symbol_features(
    *,
    symbol: str,
    spot_price: float,
    polygon_snapshot: Mapping[str, Any] | None,
    polygon_contracts: list[dict[str, Any]],
    uw_iv_rank_payload: Any,
    uw_max_pain_payload: Any,
    uw_oi_change_payload: Any,
    uw_net_prem_payload: Any,
    requested_symbols: set[str],
    today: date,
) -> tuple[dict[str, float], dict[str, Any]]:
    snapshot_metrics = _polygon_snapshot_metrics(polygon_snapshot or {})
    contract_metrics = _polygon_contract_metrics(polygon_contracts, today=today, spot_price=spot_price or snapshot_metrics["last_price"])
    iv_rank = _extract_first_numeric(uw_iv_rank_payload, "iv_rank", "ivrank", "rank") or 0.0
    iv_current = _extract_first_numeric(uw_iv_rank_payload, "implied_volatility", "iv", "current_iv") or iv_rank
    max_pain = _extract_first_numeric(uw_max_pain_payload, "max_pain", "maxpain", "price")

    oi_rows = _unusual_whales_rows(uw_oi_change_payload)
    net_prem_rows = _unusual_whales_rows(uw_net_prem_payload)
    total_abs_oi_change = 0.0
    total_curr_oi = 0.0
    total_volume = 0.0
    call_oi_change = 0.0
    put_oi_change = 0.0
    strongest_call_distance = 1.0
    strongest_put_distance = 1.0
    concentration_ratio = 0.0
    top_abs_oi = 0.0
    for row in oi_rows:
        option_symbol = str(row.get("option_symbol") or row.get("symbol") or "")
        cp, expiry, strike = _occ_contract_details(option_symbol)
        oi_change = abs(_safe_float(row.get("oi_change"), 0.0))
        curr_oi = max(_safe_float(row.get("curr_oi"), 0.0), 0.0)
        volume = max(_safe_float(row.get("volume"), 0.0), 0.0)
        total_abs_oi_change += oi_change
        total_curr_oi += curr_oi
        total_volume += volume
        top_abs_oi = max(top_abs_oi, oi_change)
        if cp == "C":
            call_oi_change += oi_change
        elif cp == "P":
            put_oi_change += oi_change
        if strike and (spot_price or snapshot_metrics["last_price"]) > 0.0:
            rel_distance = abs(strike - (spot_price or snapshot_metrics["last_price"])) / max(abs(spot_price or snapshot_metrics["last_price"]), 1e-9)
            if cp == "C":
                strongest_call_distance = min(strongest_call_distance, rel_distance)
            elif cp == "P":
                strongest_put_distance = min(strongest_put_distance, rel_distance)
    if total_abs_oi_change > 0.0:
        concentration_ratio = top_abs_oi / total_abs_oi_change

    call_premium = 0.0
    put_premium = 0.0
    total_abs_premium = 0.0
    for row in net_prem_rows:
        call_premium += _safe_float(row.get("net_call_premium"), 0.0) + _safe_float(row.get("call_premium"), 0.0)
        put_premium += _safe_float(row.get("net_put_premium"), 0.0) + _safe_float(row.get("put_premium"), 0.0)
        total_abs_premium += abs(_safe_float(row.get("net_premium"), 0.0))
    dealer_gamma_pressure = call_premium - put_premium

    max_pain_proximity = 0.0
    gamma_flip_distance = 0.5
    pin_risk = 0.0
    reference_price = spot_price or snapshot_metrics["last_price"]
    if reference_price > 0.0 and max_pain not in {None, 0.0}:
        rel_distance = abs(float(max_pain) - reference_price) / max(abs(reference_price), 1e-9)
        max_pain_proximity = _clamp01(1.0 - min(rel_distance / 0.1, 1.0))
        gamma_flip_distance = _clamp01(min(rel_distance / 0.08, 1.0))
        pin_risk = max_pain_proximity * max(contract_metrics["zero_dte_presence_norm"], contract_metrics["near_term_presence_norm"])

    liquidity_norm = _clamp01(math.log1p(snapshot_metrics["day_volume"]) / math.log(10_000_000.0 + 1.0))
    expected_move_norm = _clamp01(max(snapshot_metrics["intraday_range"], abs(snapshot_metrics["change_pct"]) / 100.0) / 0.1)
    beta_norm = _clamp01(0.35 + min(abs(snapshot_metrics["change_pct"]) / 6.0, 0.65))
    vol_regime_norm = _clamp01(math.log1p(total_volume + total_abs_oi_change) / math.log(250_000.0 + 1.0))
    surface_change_norm = _clamp01(total_abs_oi_change / 100_000.0)
    iv_skew_norm = _signed_centered_norm(call_oi_change - put_oi_change, max(total_abs_oi_change, 1.0))
    iv_term_structure_norm = _clamp01(1.0 - min(contract_metrics["nearest_expiry_days"] / 30.0, 1.0))
    gamma_expiry_skew_norm = _signed_centered_norm(call_oi_change - put_oi_change, max(total_curr_oi, 1.0))
    assignment_risk_norm = _clamp01(max(contract_metrics["zero_dte_presence_norm"], max_pain_proximity * 0.9))
    zero_dte_regime_norm = contract_metrics["zero_dte_presence_norm"]
    iv_crush_risk_norm = _clamp01((iv_rank / 100.0) * max(contract_metrics["near_term_presence_norm"], 0.25))
    vol_of_vol_change_norm = _clamp01((abs(dealer_gamma_pressure) / max(total_abs_premium, 1.0)) if total_abs_premium > 0.0 else (iv_rank / 100.0))
    spread_execution_risk_norm = _clamp01(1.0 - liquidity_norm)
    call_wall_proximity_norm = _clamp01(1.0 - min(strongest_call_distance / 0.08, 1.0))
    put_wall_proximity_norm = _clamp01(1.0 - min(strongest_put_distance / 0.08, 1.0))

    features = _zero_feature_map()
    features.update(
        {
            "tasty_iv_rank_norm": _clamp01(iv_rank / 100.0),
            "tasty_implied_volatility_index_norm": _clamp01(iv_current / 100.0),
            "tasty_liquidity_rating_norm": liquidity_norm,
            "tasty_expected_move_norm": expected_move_norm,
            "tasty_beta_norm": beta_norm,
            "tasty_watchlist_presence_norm": 1.0 if symbol in requested_symbols else 0.0,
            "short_borrow_availability_norm": _clamp01(0.55 + (liquidity_norm * 0.35)),
            "short_borrow_fee_norm": _clamp01((1.0 - liquidity_norm) * 0.6 + vol_regime_norm * 0.4),
            "short_utilization_norm": _clamp01(surface_change_norm * 0.7 + vol_regime_norm * 0.3),
            "short_days_to_cover_norm": _clamp01((1.0 - liquidity_norm) * 0.5 + contract_metrics["near_term_presence_norm"] * 0.5),
            "tasty_dealer_gamma_pressure_norm": _signed_centered_norm(dealer_gamma_pressure, max(total_abs_premium, 1.0)),
            "tasty_call_wall_proximity_norm": call_wall_proximity_norm,
            "tasty_put_wall_proximity_norm": put_wall_proximity_norm,
            "tasty_max_pain_proximity_norm": max_pain_proximity,
            "tasty_pin_risk_norm": _clamp01(pin_risk),
            "options_iv_skew_norm": iv_skew_norm,
            "options_iv_term_structure_norm": iv_term_structure_norm,
            "options_gamma_expiry_skew_norm": gamma_expiry_skew_norm,
            "options_vol_regime_norm": vol_regime_norm,
            "options_surface_change_norm": surface_change_norm,
            "options_strike_expiry_concentration_change_norm": _clamp01(concentration_ratio),
            "options_gamma_flip_distance_norm": gamma_flip_distance,
            "options_earnings_setup_norm": _clamp01(contract_metrics["near_term_presence_norm"] * 0.75 + vol_regime_norm * 0.25),
            "options_iv_crush_risk_norm": iv_crush_risk_norm,
            "options_assignment_risk_norm": assignment_risk_norm,
            "options_zero_dte_regime_norm": zero_dte_regime_norm,
            "options_vol_of_vol_change_norm": vol_of_vol_change_norm,
            "options_spread_execution_risk_norm": spread_execution_risk_norm,
        }
    )
    meta = {
        "polygon_contract_count": len(polygon_contracts),
        "unusual_whales_oi_rows": len(oi_rows),
        "unusual_whales_net_premium_rows": len(net_prem_rows),
        "spot_price": round(reference_price, 6) if reference_price else 0.0,
        "max_pain": round(float(max_pain), 6) if max_pain not in {None, 0.0} else None,
    }
    return features, meta


def _mean_feature(values: list[dict[str, float]], key: str) -> float:
    if not values:
        return 0.0
    return round(sum(float(row.get(key, 0.0)) for row in values) / max(len(values), 1), 6)


def collect_options_flow_context(
    *,
    polygon_api_key: str,
    unusual_whales_api_key: str,
    unusual_whales_export_path: str,
    symbols: list[str],
    user_agent: str,
    timeout_seconds: float,
    polygon_contract_limit: int,
    unusual_whales_export_max_age_seconds: int = DEFAULT_UNUSUAL_WHALES_EXPORT_MAX_AGE_SECONDS,
    unusual_whales_export_min_stable_seconds: int = DEFAULT_UNUSUAL_WHALES_EXPORT_MIN_STABLE_SECONDS,
    free_sources_enabled: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = datetime.now(timezone.utc)
    today = now.date()
    requested_symbols = {symbol.upper() for symbol in symbols}
    polygon_contract = _source_contract("polygon_options_backbone")
    unusual_whales_api_contract = _source_contract("unusual_whales_api")
    unusual_whales_export_contract = _source_contract("unusual_whales_export")
    yahoo_options_contract = _source_contract("yahoo_options_chain")
    cboe_options_contract = _source_contract("cboe_delayed_options")
    export_payload, export_source = inspect_unusual_whales_export(
        unusual_whales_export_path,
        max_age_seconds=max(int(unusual_whales_export_max_age_seconds), 1),
        min_stable_seconds=max(int(unusual_whales_export_min_stable_seconds), 0),
        now=now,
    )
    export_source_payload = {
        "ok": False,
        "symbol_count": 0,
        "available_symbol_count": int(export_source.get("symbol_count", 0) or 0),
        "errors": list(export_source.get("issues") or []),
        "configured": bool(export_source.get("configured", False)),
        "source_path": str(export_source.get("source_path") or ""),
        "selected_candidate": str(export_source.get("selected_candidate") or ""),
        "path_mode": str(export_source.get("path_mode") or ""),
        "fresh": bool(export_source.get("fresh", False)),
        "timestamp_utc": str(export_source.get("timestamp_utc") or ""),
        "timestamp_source": str(export_source.get("timestamp_source") or ""),
        "age_seconds": export_source.get("age_seconds"),
        "format": str(export_source.get("format") or ""),
        "schema_version": str(export_source.get("schema_version") or ""),
        "schema_status": str(export_source.get("schema_status") or "missing"),
        "adapter": str(export_source.get("adapter") or ""),
        "max_age_seconds": int(export_source.get("max_age_seconds", unusual_whales_export_max_age_seconds) or unusual_whales_export_max_age_seconds),
        "min_stable_seconds": int(export_source.get("min_stable_seconds", unusual_whales_export_min_stable_seconds) or unusual_whales_export_min_stable_seconds),
        "candidate_count": int(export_source.get("candidate_count", 0) or 0),
        "row_count": int(export_source.get("row_count", 0) or 0),
        "rejected_row_count": int(export_source.get("rejected_row_count", 0) or 0),
        "dataset_symbol_counts": dict(export_source.get("dataset_symbol_counts") or {}),
        "ignored_candidates": list(export_source.get("ignored_candidates") or [])[:5],
        "size_bytes": int(export_source.get("size_bytes", 0) or 0),
        "source_confidence_norm": float(unusual_whales_export_contract["source_confidence_norm"]),
        "schema_confidence_norm": float(unusual_whales_export_contract["schema_confidence_norm"]),
        "freshness_norm": 1.0 if bool(export_source.get("fresh", False)) else 0.0,
        "fetched_utc": str(export_source.get("timestamp_utc") or ""),
    }
    unusual_whales_api_expected = bool(str(unusual_whales_api_key or "").strip())
    unusual_whales_export_expected = bool(export_source_payload.get("configured", False))
    unusual_whales_expected = bool(unusual_whales_api_expected or unusual_whales_export_expected)
    payload: dict[str, Any] = {
        "timestamp_utc": now.isoformat(),
        "provider": "options_flow_context",
        "legacy_feature_namespace": "tastytrade_compat",
        "providers": ["polygon", "unusual_whales"],
        "symbols": symbols,
        "derived": {
            "global_features": {key: 0.0 for key in FEATURE_KEYS},
            "symbol_features": {},
        },
        "sources": {
            "polygon": {
                "ok": False,
                "symbol_count": 0,
                "errors": [],
                "required": True,
                "expected": True,
                "contract_participates": True,
                "source_confidence_norm": float(polygon_contract["source_confidence_norm"]),
                "schema_confidence_norm": float(polygon_contract["schema_confidence_norm"]),
                "freshness_norm": 0.0,
                "fetched_utc": "",
            },
            "unusual_whales_api": {
                "ok": False,
                "symbol_count": 0,
                "errors": [],
                "required": False,
                "expected": unusual_whales_api_expected,
                "contract_participates": unusual_whales_api_expected,
                "source_confidence_norm": float(unusual_whales_api_contract["source_confidence_norm"]),
                "schema_confidence_norm": float(unusual_whales_api_contract["schema_confidence_norm"]),
                "freshness_norm": 0.0,
                "fetched_utc": "",
            },
            "unusual_whales_export": {
                **dict(export_source_payload),
                "required": False,
                "expected": unusual_whales_export_expected,
                "contract_participates": unusual_whales_export_expected,
            },
            "yahoo_options_chain": {
                "ok": False,
                "symbol_count": 0,
                "contract_count": 0,
                "errors": [],
                "required": False,
                "expected": bool(free_sources_enabled),
                "contract_participates": bool(free_sources_enabled),
                "source_confidence_norm": float(yahoo_options_contract["source_confidence_norm"]),
                "schema_confidence_norm": float(yahoo_options_contract["schema_confidence_norm"]),
                "freshness_norm": 0.0,
                "fetched_utc": "",
            },
            "cboe_delayed_options": {
                "ok": False,
                "symbol_count": 0,
                "contract_count": 0,
                "errors": [],
                "required": False,
                "expected": bool(free_sources_enabled),
                "contract_participates": bool(free_sources_enabled),
                "source_confidence_norm": float(cboe_options_contract["source_confidence_norm"]),
                "schema_confidence_norm": float(cboe_options_contract["schema_confidence_norm"]),
                "freshness_norm": 0.0,
                "fetched_utc": "",
            },
        },
    }
    status: dict[str, Any] = {
        "timestamp_utc": now.isoformat(),
        "provider": "options_flow_context",
        "legacy_alias": LEGACY_ALIAS,
        "ok": False,
        "overall_status": "blocked",
        "symbols_requested": len(symbols),
        "symbols_with_chain": 0,
        "symbols_with_metrics": 0,
        "symbols_with_polygon": 0,
        "symbols_with_polygon_chain": 0,
        "symbols_with_unusual_whales": 0,
        "symbols_with_unusual_whales_export": 0,
        "symbols_with_free_options": 0,
        "alignment_ok": True,
        "alignment_compared": 0,
        "alignment_reference_only": 0,
        "sandbox": False,
        "requested_sandbox": False,
        "errors": [],
        "degraded_reasons": [],
        "sources": {
            "polygon": {
                "ok": False,
                "symbol_count": 0,
                "errors": [],
                "required": True,
                "expected": True,
                "contract_participates": True,
                "source_confidence_norm": float(polygon_contract["source_confidence_norm"]),
                "schema_confidence_norm": float(polygon_contract["schema_confidence_norm"]),
                "freshness_norm": 0.0,
                "fetched_utc": "",
            },
            "unusual_whales_api": {
                "ok": False,
                "symbol_count": 0,
                "errors": [],
                "required": False,
                "expected": unusual_whales_api_expected,
                "contract_participates": unusual_whales_api_expected,
                "source_confidence_norm": float(unusual_whales_api_contract["source_confidence_norm"]),
                "schema_confidence_norm": float(unusual_whales_api_contract["schema_confidence_norm"]),
                "freshness_norm": 0.0,
                "fetched_utc": "",
            },
            "unusual_whales_export": {
                **dict(export_source_payload),
                "required": False,
                "expected": unusual_whales_export_expected,
                "contract_participates": unusual_whales_export_expected,
            },
            "yahoo_options_chain": dict(payload["sources"]["yahoo_options_chain"]),
            "cboe_delayed_options": dict(payload["sources"]["cboe_delayed_options"]),
        },
    }

    if not polygon_api_key and not unusual_whales_api_expected and not export_payload and not free_sources_enabled:
        status["auth_issue"] = (
            "options_flow_export_unusable"
            if bool(export_source_payload.get("configured", False))
            else "options_flow_credentials_missing"
        )
        status["operator_action_required"] = True
        status["errors"].append(str(status["auth_issue"]))
        payload["sources"]["session"] = {
            "ok": False,
            "error": str(status["auth_issue"]),
            "operator_action_required": True,
            "recommended_action": (
                "repair_unusual_whales_export_or_set_polygon_api_key"
                if bool(export_source_payload.get("configured", False))
                else "set_polygon_api_key"
            ),
        }
        return payload, status

    symbol_feature_rows: list[dict[str, float]] = []
    polygon_errors: set[str] = set()
    uw_errors: set[str] = set()
    yahoo_errors: set[str] = set()
    cboe_errors: set[str] = set()

    for symbol in symbols:
        polygon_snapshot, polygon_snapshot_err = _polygon_get(
            f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}",
            api_key=polygon_api_key,
            query=None,
            user_agent=user_agent,
            timeout=timeout_seconds,
        )
        polygon_contracts_payload, polygon_contracts_err = _polygon_get(
            "/v3/reference/options/contracts",
            api_key=polygon_api_key,
            query={"underlying_ticker": symbol, "limit": max(int(polygon_contract_limit), 1), "sort": "expiration_date", "order": "asc"},
            user_agent=user_agent,
            timeout=timeout_seconds,
        )
        if polygon_snapshot is not None or polygon_contracts_payload is not None:
            status["symbols_with_polygon"] += 1
        if polygon_snapshot is not None:
            status["alignment_compared"] += 1
        if polygon_snapshot_err:
            polygon_errors.add(str(polygon_snapshot_err))
        if polygon_contracts_err:
            polygon_errors.add(str(polygon_contracts_err))

        polygon_contract_rows = _extract_rows(polygon_contracts_payload)
        if polygon_contract_rows:
            status["symbols_with_polygon_chain"] += 1
        free_snapshot: dict[str, Any] = {}
        free_iv_payload: dict[str, Any] | None = None
        free_contract_rows: list[dict[str, Any]] = []
        if free_sources_enabled:
            yahoo_snapshot, yahoo_iv, yahoo_rows, yahoo_status = _collect_yahoo_options_chain(
                symbol,
                user_agent=user_agent,
                timeout=timeout_seconds,
                contract_limit=polygon_contract_limit,
            )
            if yahoo_status:
                if yahoo_status.get("ok"):
                    payload["sources"]["yahoo_options_chain"]["symbol_count"] = int(payload["sources"]["yahoo_options_chain"].get("symbol_count", 0) or 0) + 1
                    payload["sources"]["yahoo_options_chain"]["contract_count"] = int(payload["sources"]["yahoo_options_chain"].get("contract_count", 0) or 0) + int(yahoo_status.get("contract_count", 0) or 0)
                    payload["sources"]["yahoo_options_chain"]["freshness_norm"] = 1.0
                    payload["sources"]["yahoo_options_chain"]["fetched_utc"] = now.isoformat()
                elif yahoo_status.get("error"):
                    yahoo_errors.add(str(yahoo_status.get("error")))
            if yahoo_rows:
                free_contract_rows.extend(yahoo_rows)
                free_snapshot = yahoo_snapshot or free_snapshot
                free_iv_payload = yahoo_iv or free_iv_payload

            cboe_snapshot, cboe_iv, cboe_rows, cboe_status = _collect_cboe_options_chain(
                symbol,
                user_agent=user_agent,
                timeout=timeout_seconds,
                contract_limit=polygon_contract_limit,
            )
            if cboe_status:
                if cboe_status.get("ok"):
                    payload["sources"]["cboe_delayed_options"]["symbol_count"] = int(payload["sources"]["cboe_delayed_options"].get("symbol_count", 0) or 0) + 1
                    payload["sources"]["cboe_delayed_options"]["contract_count"] = int(payload["sources"]["cboe_delayed_options"].get("contract_count", 0) or 0) + int(cboe_status.get("contract_count", 0) or 0)
                    payload["sources"]["cboe_delayed_options"]["freshness_norm"] = 1.0
                    payload["sources"]["cboe_delayed_options"]["fetched_utc"] = now.isoformat()
                elif cboe_status.get("error"):
                    cboe_errors.add(str(cboe_status.get("error")))
            if cboe_rows:
                free_contract_rows.extend(cboe_rows)
                free_snapshot = free_snapshot or cboe_snapshot
                free_iv_payload = free_iv_payload or cboe_iv

        option_contract_rows = polygon_contract_rows or free_contract_rows
        if option_contract_rows:
            status["symbols_with_chain"] += 1
        if free_contract_rows:
            status["symbols_with_free_options"] += 1

        uw_iv_rank_payload, uw_iv_rank_err = _unusual_whales_get(
            f"/api/stock/{symbol}/iv-rank",
            api_key=unusual_whales_api_key,
            query=None,
            user_agent=user_agent,
            timeout=timeout_seconds,
        )
        uw_max_pain_payload, uw_max_pain_err = _unusual_whales_get(
            f"/api/stock/{symbol}/max-pain",
            api_key=unusual_whales_api_key,
            query=None,
            user_agent=user_agent,
            timeout=timeout_seconds,
        )
        uw_oi_change_payload, uw_oi_change_err = _unusual_whales_get(
            f"/api/stock/{symbol}/oi-change",
            api_key=unusual_whales_api_key,
            query={"limit": 200},
            user_agent=user_agent,
            timeout=timeout_seconds,
        )
        uw_net_prem_payload, uw_net_prem_err = _unusual_whales_get(
            f"/api/stock/{symbol}/net-prem-ticks",
            api_key=unusual_whales_api_key,
            query=None,
            user_agent=user_agent,
            timeout=timeout_seconds,
        )
        if unusual_whales_api_key and any(
            payload_node is not None
            for payload_node in (uw_iv_rank_payload, uw_max_pain_payload, uw_oi_change_payload, uw_net_prem_payload)
        ):
            status["symbols_with_unusual_whales"] += 1
        for err in (uw_iv_rank_err, uw_max_pain_err, uw_oi_change_err, uw_net_prem_err):
            if err and err != "unusual_whales_api_key_missing":
                uw_errors.add(str(err))

        export_iv_rank = _export_symbol_payload(export_payload, symbol, "iv_rank")
        export_max_pain = _export_symbol_payload(export_payload, symbol, "max_pain")
        export_oi_change = _export_symbol_payload(export_payload, symbol, "oi_change")
        export_net_prem = _export_symbol_payload(export_payload, symbol, "net_prem_ticks")

        symbol_features, symbol_meta = _compute_symbol_features(
            symbol=symbol,
            spot_price=_polygon_snapshot_metrics(polygon_snapshot or free_snapshot or {}).get("last_price", 0.0),
            polygon_snapshot=polygon_snapshot if isinstance(polygon_snapshot, Mapping) else free_snapshot,
            polygon_contracts=option_contract_rows,
            uw_iv_rank_payload=uw_iv_rank_payload or export_iv_rank or free_iv_payload,
            uw_max_pain_payload=uw_max_pain_payload or export_max_pain,
            uw_oi_change_payload=uw_oi_change_payload or export_oi_change,
            uw_net_prem_payload=uw_net_prem_payload or export_net_prem,
            requested_symbols=requested_symbols,
            today=today,
        )
        if any(value > 0.0 for value in symbol_features.values()):
            symbol_source_confidence = max(
                float(polygon_contract["source_confidence_norm"]) if (polygon_snapshot is not None or polygon_contracts_payload is not None) else 0.0,
                float(unusual_whales_api_contract["source_confidence_norm"])
                if any(payload_node is not None for payload_node in (uw_iv_rank_payload, uw_max_pain_payload, uw_oi_change_payload, uw_net_prem_payload))
                else 0.0,
                float(unusual_whales_export_contract["source_confidence_norm"])
                if any(node is not None for node in (export_iv_rank, export_max_pain, export_oi_change, export_net_prem))
                else 0.0,
                max(float(yahoo_options_contract["source_confidence_norm"]), float(cboe_options_contract["source_confidence_norm"]))
                if free_contract_rows
                else 0.0,
            )
            symbol_schema_confidence = max(
                float(polygon_contract["schema_confidence_norm"]) if (polygon_snapshot is not None or polygon_contracts_payload is not None) else 0.0,
                float(unusual_whales_api_contract["schema_confidence_norm"])
                if any(payload_node is not None for payload_node in (uw_iv_rank_payload, uw_max_pain_payload, uw_oi_change_payload, uw_net_prem_payload))
                else 0.0,
                float(unusual_whales_export_contract["schema_confidence_norm"])
                if any(node is not None for node in (export_iv_rank, export_max_pain, export_oi_change, export_net_prem))
                else 0.0,
                max(float(yahoo_options_contract["schema_confidence_norm"]), float(cboe_options_contract["schema_confidence_norm"]))
                if free_contract_rows
                else 0.0,
            )
            symbol_freshness = max(
                1.0 if (polygon_snapshot is not None or polygon_contracts_payload is not None) else 0.0,
                1.0 if any(payload_node is not None for payload_node in (uw_iv_rank_payload, uw_max_pain_payload, uw_oi_change_payload, uw_net_prem_payload)) else 0.0,
                float(payload["sources"]["unusual_whales_export"].get("freshness_norm", 0.0) or 0.0)
                if any(node is not None for node in (export_iv_rank, export_max_pain, export_oi_change, export_net_prem))
                else 0.0,
                1.0 if free_contract_rows else 0.0,
            )
            symbol_feature_rows.append(symbol_features)
            payload["derived"]["symbol_features"][symbol] = attach_collection_confidence(
                symbol_features,
                source_confidence_norm=symbol_source_confidence,
                schema_confidence_norm=symbol_schema_confidence,
                freshness_norm=symbol_freshness,
                fetched_utc=str(now.isoformat()),
            )
            status["symbols_with_metrics"] += 1
            payload.setdefault("symbol_meta", {})[symbol] = symbol_meta
            payload["symbol_meta"][symbol]["free_option_contract_count"] = len(free_contract_rows)
        if export_iv_rank or export_max_pain or export_oi_change or export_net_prem:
            payload["sources"]["unusual_whales_export"]["symbol_count"] = int(payload["sources"]["unusual_whales_export"].get("symbol_count", 0) or 0) + 1
            status["symbols_with_unusual_whales_export"] += 1

    payload["sources"]["polygon"]["ok"] = status["symbols_with_polygon"] > 0
    payload["sources"]["polygon"]["symbol_count"] = status["symbols_with_polygon"]
    payload["sources"]["polygon"]["errors"] = sorted(polygon_errors)
    payload["sources"]["polygon"]["freshness_norm"] = 1.0 if payload["sources"]["polygon"]["ok"] else 0.0
    payload["sources"]["polygon"]["fetched_utc"] = now.isoformat() if payload["sources"]["polygon"]["ok"] else ""
    payload["sources"]["unusual_whales_api"]["ok"] = status["symbols_with_unusual_whales"] > 0
    payload["sources"]["unusual_whales_api"]["symbol_count"] = status["symbols_with_unusual_whales"]
    payload["sources"]["unusual_whales_api"]["errors"] = sorted(uw_errors)
    payload["sources"]["unusual_whales_api"]["freshness_norm"] = 1.0 if payload["sources"]["unusual_whales_api"]["ok"] else 0.0
    payload["sources"]["unusual_whales_api"]["fetched_utc"] = now.isoformat() if payload["sources"]["unusual_whales_api"]["ok"] else ""
    payload["sources"]["yahoo_options_chain"]["ok"] = int(payload["sources"]["yahoo_options_chain"].get("symbol_count", 0) or 0) > 0
    payload["sources"]["yahoo_options_chain"]["errors"] = sorted(yahoo_errors)[:10]
    payload["sources"]["cboe_delayed_options"]["ok"] = int(payload["sources"]["cboe_delayed_options"].get("symbol_count", 0) or 0) > 0
    payload["sources"]["cboe_delayed_options"]["errors"] = sorted(cboe_errors)[:10]
    if bool(payload["sources"]["unusual_whales_export"].get("configured", False)) and int(payload["sources"]["unusual_whales_export"].get("symbol_count", 0) or 0) <= 0:
        export_errors = list(payload["sources"]["unusual_whales_export"].get("errors") or [])
        if bool(export_payload) and "requested_symbols_missing_from_export" not in export_errors:
            export_errors.append("requested_symbols_missing_from_export")
        payload["sources"]["unusual_whales_export"]["errors"] = export_errors

    for key in FEATURE_KEYS:
        payload["derived"]["global_features"][key] = _mean_feature(symbol_feature_rows, key)

    payload["sources"]["unusual_whales_export"]["ok"] = bool(
        int(payload["sources"]["unusual_whales_export"].get("symbol_count", 0) or 0) > 0
        and bool(payload["sources"]["unusual_whales_export"].get("fresh", False))
    )
    polygon_ok = bool(payload["sources"]["polygon"]["ok"])
    polygon_backbone_ok = int(status.get("symbols_with_polygon_chain", 0) or 0) > 0
    uw_api_ok = bool(payload["sources"]["unusual_whales_api"]["ok"])
    uw_export_ok = bool(payload["sources"]["unusual_whales_export"]["ok"])
    uw_ok = bool(uw_api_ok or uw_export_ok)
    free_options_ok = bool(payload["sources"]["yahoo_options_chain"]["ok"] or payload["sources"]["cboe_delayed_options"]["ok"])
    context_profile = (
        "multi_provider_full"
        if polygon_backbone_ok and uw_ok
        else "polygon_primary_only"
        if polygon_backbone_ok and not unusual_whales_expected
        else "polygon_backbone_only"
        if polygon_backbone_ok
        else "free_options_chain_plus_overlay"
        if free_options_ok and uw_ok
        else "free_options_chain_only"
        if free_options_ok
        else "unusual_whales_overlay_only"
        if uw_ok
        else "unavailable"
    )
    coverage_score = (
        1.0
        if context_profile == "multi_provider_full"
        else 0.9
        if context_profile == "polygon_primary_only"
        else 0.75
        if context_profile == "polygon_backbone_only"
        else 0.68
        if context_profile == "free_options_chain_plus_overlay"
        else 0.62
        if context_profile == "free_options_chain_only"
        else 0.45
        if context_profile == "unusual_whales_overlay_only"
        else 0.0
    )
    status["ok"] = bool(payload["derived"]["symbol_features"]) and (polygon_ok or uw_ok or free_options_ok)
    if not status["ok"]:
        if not polygon_ok and not uw_ok and not free_options_ok:
            status["auth_issue"] = "options_flow_sources_unavailable"
            status["operator_action_required"] = True
            status["errors"].append("options_flow_sources_unavailable")
        elif not polygon_ok and uw_ok:
            status["auth_issue"] = sorted(polygon_errors)[0] if polygon_errors else "polygon_source_unavailable"
            status["operator_action_required"] = True
            status["errors"].append(f"polygon:{status['auth_issue']}")
        elif unusual_whales_expected:
            status["errors"].append("unusual_whales_optional_source_unavailable")
    elif context_profile == "unusual_whales_overlay_only":
        status["degraded_reasons"].append("polygon_backbone_missing")
        status["operator_action_required"] = True
    status["overall_status"] = "ready" if status["ok"] and (polygon_backbone_ok or free_options_ok) else ("degraded" if status["ok"] else "blocked")
    status["context_profile"] = context_profile
    status["coverage_score"] = round(float(coverage_score), 3)
    status["coverage"] = {
        "context_profile": context_profile,
        "coverage_score": round(float(coverage_score), 3),
        "polygon_backbone_ok": bool(polygon_backbone_ok),
        "free_options_chain_ok": bool(free_options_ok),
        "symbols_with_polygon_chain": int(status.get("symbols_with_polygon_chain", 0) or 0),
        "polygon_signal_ok": bool(polygon_ok),
        "unusual_whales_any_ok": bool(uw_ok),
        "unusual_whales_api_ok": bool(uw_api_ok),
        "unusual_whales_export_ok": bool(uw_export_ok),
        "export_available_symbol_count": int(payload["sources"]["unusual_whales_export"].get("available_symbol_count", 0) or 0),
    }
    status["sources"] = {
        "polygon": dict(payload["sources"]["polygon"]),
        "unusual_whales_api": dict(payload["sources"]["unusual_whales_api"]),
        "unusual_whales_export": dict(payload["sources"]["unusual_whales_export"]),
        "yahoo_options_chain": dict(payload["sources"]["yahoo_options_chain"]),
        "cboe_delayed_options": dict(payload["sources"]["cboe_delayed_options"]),
    }
    status["source_contracts"] = dict(SOURCE_CONTRACTS)
    participating_sources = [
        row
        for row in status["sources"].values()
        if isinstance(row, Mapping) and bool(row.get("ok", False))
    ]
    provider_confidence_norm = (
        sum(float(row.get("source_confidence_norm", 0.0) or 0.0) for row in participating_sources)
        / max(len(participating_sources), 1)
    )
    payload["collection_contract"] = {
        "provider": "options_flow_context",
        "source_contracts": dict(SOURCE_CONTRACTS),
        "provider_confidence_norm": round(float(provider_confidence_norm), 6),
    }
    status["collection_contract"] = dict(payload["collection_contract"])

    payload["sources"]["session"] = {
        "ok": status["ok"],
        "polygon_ok": polygon_ok,
        "polygon_backbone_ok": bool(polygon_backbone_ok),
        "free_options_chain_ok": bool(free_options_ok),
        "symbols_with_polygon_chain": int(status.get("symbols_with_polygon_chain", 0) or 0),
        "unusual_whales_ok": uw_ok,
        "unusual_whales_expected": unusual_whales_expected,
        "context_profile": context_profile,
        "coverage_score": round(float(coverage_score), 3),
        "overall_status": str(status.get("overall_status") or ""),
        "operator_action_required": bool(status.get("operator_action_required", False)),
        "recommended_action": (
            "set_polygon_api_key"
            if status.get("auth_issue") in {"polygon_api_key_missing", "polygon_plan_restricted", "polygon_source_unavailable"}
            else ""
            if context_profile.startswith("free_options_chain")
            else "set_polygon_api_key"
            if context_profile == "unusual_whales_overlay_only"
            else "repair_unusual_whales_export_or_set_polygon_api_key"
            if status.get("auth_issue") == "options_flow_export_unusable"
            else "set_polygon_api_key"
            if status.get("operator_action_required", False) and not unusual_whales_expected
            else "set_polygon_api_key_or_unusual_whales_source"
            if status.get("operator_action_required", False)
            else ""
        ),
    }
    return payload, status


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect options-flow context from Polygon and optional Unusual Whales sources.")
    parser.add_argument("--polygon-api-key", default=os.getenv("POLYGON_API_KEY", ""))
    parser.add_argument("--unusual-whales-api-key", default=os.getenv("UNUSUAL_WHALES_API_KEY", ""))
    parser.add_argument("--unusual-whales-export-path", default=os.getenv("UNUSUAL_WHALES_EXPORT_PATH", ""))
    parser.add_argument(
        "--unusual-whales-export-max-age-seconds",
        type=int,
        default=int(os.getenv("UNUSUAL_WHALES_EXPORT_MAX_AGE_SECONDS", str(DEFAULT_UNUSUAL_WHALES_EXPORT_MAX_AGE_SECONDS))),
    )
    parser.add_argument(
        "--unusual-whales-export-min-stable-seconds",
        type=int,
        default=int(os.getenv("UNUSUAL_WHALES_EXPORT_MIN_STABLE_SECONDS", str(DEFAULT_UNUSUAL_WHALES_EXPORT_MIN_STABLE_SECONDS))),
    )
    parser.add_argument("--symbols", default=os.getenv("OPTIONS_FLOW_SYMBOLS", ""))
    parser.add_argument("--timeout-seconds", type=float, default=float(os.getenv("OPTIONS_FLOW_TIMEOUT_SECONDS", "20")))
    parser.add_argument("--polygon-contract-limit", type=int, default=int(os.getenv("OPTIONS_FLOW_POLYGON_CONTRACT_LIMIT", "250")))
    parser.add_argument("--user-agent", default=os.getenv("OPTIONS_FLOW_USER_AGENT", USER_AGENT_DEFAULT))
    parser.add_argument("--disable-free-sources", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols) if str(args.symbols or "").strip() else _default_symbols()
    payload, status = collect_options_flow_context(
        polygon_api_key=str(args.polygon_api_key or ""),
        unusual_whales_api_key=str(args.unusual_whales_api_key or ""),
        unusual_whales_export_path=str(args.unusual_whales_export_path or ""),
        symbols=symbols,
        user_agent=str(args.user_agent or USER_AGENT_DEFAULT),
        timeout_seconds=float(args.timeout_seconds),
        polygon_contract_limit=int(args.polygon_contract_limit),
        unusual_whales_export_max_age_seconds=int(args.unusual_whales_export_max_age_seconds),
        unusual_whales_export_min_stable_seconds=int(args.unusual_whales_export_min_stable_seconds),
        free_sources_enabled=not bool(args.disable_free_sources),
    )

    options_flow_payload_path = PROJECT_ROOT / "exports" / "external_context" / "options_flow_context_latest.json"
    options_flow_status_path = PROJECT_ROOT / "governance" / "health" / "options_flow_context_sync_latest.json"
    legacy_payload_path = PROJECT_ROOT / "exports" / "external_context" / f"{LEGACY_ALIAS}_latest.json"
    legacy_status_path = PROJECT_ROOT / "governance" / "health" / f"{LEGACY_ALIAS}_sync_latest.json"
    _write_json(options_flow_payload_path, payload)
    _write_json(options_flow_status_path, status)
    _write_json(legacy_payload_path, payload)
    _write_json(legacy_status_path, status)

    if args.json:
        print(json.dumps(status, ensure_ascii=True))
    else:
        print(
            "options_flow_context ok={ok} symbols={symbols} polygon={polygon} unusual_whales={uw} metrics={metrics}".format(
                ok=str(bool(status.get("ok", False))).lower(),
                symbols=len(symbols),
                polygon=int(status.get("symbols_with_polygon", 0) or 0),
                uw=int(status.get("symbols_with_unusual_whales", 0) or 0),
                metrics=int(status.get("symbols_with_metrics", 0) or 0),
            )
        )
    return 0 if status.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
