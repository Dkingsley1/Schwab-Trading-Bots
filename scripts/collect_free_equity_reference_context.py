#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote, urlencode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.collector_transport import fetch_json, fetch_text


HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "free_equity_reference_context_latest.json"
EXTERNAL_CONTEXT_PATH = PROJECT_ROOT / "exports" / "external_context" / "free_equity_reference_context_latest.json"
USER_AGENT_DEFAULT = "schwab-trading-bot/1.0"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
NASDAQ_QUOTE_URL = "https://api.nasdaq.com/api/quote"
NASDAQ_USER_AGENT_DEFAULT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
NASDAQ_ETF_SYMBOLS = {"DIA", "GLD", "IWM", "QQQ", "SLV", "SPY", "TLT", "TQQQ", "UVXY", "VXX"}
FEATURE_KEYS = (
    "free_equity_quote_available_norm",
    "free_equity_yahoo_volume_norm",
    "free_equity_nasdaq_volume_norm",
    "free_equity_momentum_norm",
    "free_equity_cross_provider_agreement_norm",
)
DEFAULT_FETCH_TIMEOUT_SECONDS = 2.5
DEFAULT_MAX_SYMBOLS = 20
DEFAULT_MAX_RUNTIME_SECONDS = 45.0
MIN_FETCH_TIMEOUT_SECONDS = 1.0


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _normalize_symbol(raw: Any) -> str:
    return "".join(ch for ch in str(raw or "").strip().upper() if ch.isalnum() or ch in {".", "-"})


def _parse_symbols(raw: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in str(raw or "").replace("\n", ",").split(","):
        symbol = _normalize_symbol(token)
        if not symbol or symbol.endswith("-USD") or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _default_symbols() -> list[str]:
    raw = os.getenv("FREE_EQUITY_REFERENCE_SYMBOLS", "").strip()
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
        symbols = _parse_symbols("SPY,QQQ,DIA,IWM,AAPL,MSFT,NVDA,AMD,AMZN,GOOGL,META,TSLA,COIN,MSTR,HUT,IREN,RIOT,CORZ,WULF,APLD,CIFR,MARA,BTDR,BITF,HIVE,BTBT,DGHI")
    return symbols


def _to_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _signed_centered_norm(value: float, scale: float) -> float:
    return _clamp01(0.5 + (float(value) / max(float(scale), 1e-8)))


def _safe_log_norm(value: float, *, denom: float) -> float:
    if value <= 0.0:
        return 0.0
    return _clamp01(math.log1p(float(value)) / max(float(denom), 1e-8))


def _parse_market_number(raw: Any, default: float = 0.0) -> float:
    text = str(raw or "").strip()
    if not text or text.upper() in {"N/A", "NA", "--"}:
        return float(default)
    for token in ("$", "%", ",", "+"):
        text = text.replace(token, "")
    return _to_float(text, default)


def _parse_market_percent(raw: Any) -> float:
    return _parse_market_number(raw, 0.0) / 100.0


def _agreement(prices: list[float], *, max_relative_spread: float = 0.03) -> float:
    clean = [price for price in prices if price > 0.0]
    if len(clean) < 2:
        return 0.0
    spread = (max(clean) - min(clean)) / max(min(clean), 1e-8)
    return _clamp01(1.0 - (spread / max(float(max_relative_spread), 1e-8)))


def _bounded_timeout(timeout: float, deadline: float | None) -> float:
    base = max(float(timeout), MIN_FETCH_TIMEOUT_SECONDS)
    if deadline is None:
        return base
    remaining = max(float(deadline) - time.monotonic(), 0.0)
    return max(min(base, remaining), MIN_FETCH_TIMEOUT_SECONDS)


def _has_fetch_budget(deadline: float | None) -> bool:
    return deadline is None or (float(deadline) - time.monotonic()) >= MIN_FETCH_TIMEOUT_SECONDS


def _is_deadline_skip(error: Any) -> bool:
    return str(error or "") in {"deadline_deferred", "deadline_exceeded"}


def _fetch_yahoo_chart(symbol: str, *, user_agent: str, timeout: float, retries: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    url = f"{YAHOO_CHART_URL.rstrip('/')}/{quote(symbol)}?{urlencode({'range': '5d', 'interval': '1d'})}"
    result = fetch_json(
        url=url,
        user_agent=user_agent,
        timeout=timeout,
        collector_key="free_equity_reference_context",
        source_name="yahoo_chart",
        entity_key=symbol,
        project_root=PROJECT_ROOT,
        source_confidence_norm=0.70,
        schema_confidence_norm=0.84,
        retries=max(int(retries), 0),
    )
    payload = result.get("json")
    if not bool(result.get("ok", False)) or not isinstance(payload, Mapping):
        return {}, {"ok": False, "error": str(result.get("error") or "fetch_failed"), "url": url}
    chart = payload.get("chart") if isinstance(payload.get("chart"), Mapping) else {}
    rows = chart.get("result") if isinstance(chart.get("result"), list) else []
    row = rows[0] if rows and isinstance(rows[0], Mapping) else {}
    meta = row.get("meta") if isinstance(row.get("meta"), Mapping) else {}
    quote_rows = ((row.get("indicators") or {}).get("quote") if isinstance(row.get("indicators"), Mapping) else [])
    quote0 = quote_rows[0] if quote_rows and isinstance(quote_rows[0], Mapping) else {}
    closes = [value for value in (quote0.get("close") or []) if isinstance(value, (int, float)) and value and math.isfinite(float(value))]
    volumes = [value for value in (quote0.get("volume") or []) if isinstance(value, (int, float)) and value and math.isfinite(float(value))]
    price = _to_float(meta.get("regularMarketPrice"), closes[-1] if closes else 0.0)
    prev = _to_float(meta.get("chartPreviousClose"), closes[-2] if len(closes) >= 2 else 0.0)
    volume = _to_float(meta.get("regularMarketVolume"), volumes[-1] if volumes else 0.0)
    change = ((price - prev) / max(abs(prev), 1e-9)) if price > 0.0 and prev > 0.0 else 0.0
    return {
        "price": price,
        "previous_close": prev,
        "volume": volume,
        "change": change,
        "currency": str(meta.get("currency") or ""),
    }, {"ok": price > 0.0, "url": url, "price": price, "volume": volume, "error": None if price > 0.0 else "missing_price"}


def _nasdaq_asset_class(symbol: str) -> str:
    return "etf" if _normalize_symbol(symbol) in NASDAQ_ETF_SYMBOLS else "stocks"


def _nasdaq_user_agent(user_agent: str) -> str:
    text = str(user_agent or "").strip()
    return text if "Mozilla/" in text else NASDAQ_USER_AGENT_DEFAULT


def _fetch_nasdaq_quote(symbol: str, *, user_agent: str, timeout: float, retries: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    asset_class = _nasdaq_asset_class(symbol)
    url = f"{NASDAQ_QUOTE_URL.rstrip('/')}/{quote(_normalize_symbol(symbol))}/info?{urlencode({'assetclass': asset_class})}"
    result = fetch_json(
        url=url,
        user_agent=_nasdaq_user_agent(user_agent),
        timeout=timeout,
        collector_key="free_equity_reference_context",
        source_name="nasdaq_quote",
        entity_key=symbol,
        project_root=PROJECT_ROOT,
        source_confidence_norm=0.76,
        schema_confidence_norm=0.84,
        retries=max(int(retries), 0),
    )
    payload = result.get("json")
    if not bool(result.get("ok", False)) or not isinstance(payload, Mapping):
        return {}, {"ok": False, "url": url, "error": str(result.get("error") or "fetch_failed")}
    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
    primary = data.get("primaryData") if isinstance(data.get("primaryData"), Mapping) else {}
    price = _parse_market_number(primary.get("lastSalePrice"), 0.0)
    bid = _parse_market_number(primary.get("bidPrice"), 0.0)
    ask = _parse_market_number(primary.get("askPrice"), 0.0)
    if price <= 0.0 and bid > 0.0 and ask > 0.0:
        price = (bid + ask) / 2.0
    volume = _parse_market_number(primary.get("volume"), 0.0)
    change = _parse_market_percent(primary.get("percentageChange"))
    return {
        "price": price,
        "bid": bid,
        "ask": ask,
        "volume": volume,
        "change": change,
        "market_status": str(data.get("marketStatus") or ""),
        "asset_class": str(data.get("assetClass") or "").upper(),
        "is_realtime": bool(primary.get("isRealTime", False)),
    }, {"ok": price > 0.0, "url": url, "price": price, "volume": volume, "error": None if price > 0.0 else "missing_price"}


def _symbol_features(yahoo: Mapping[str, Any], nasdaq: Mapping[str, Any]) -> dict[str, float]:
    yahoo_price = _to_float(yahoo.get("price"), 0.0)
    nasdaq_price = _to_float(nasdaq.get("price"), 0.0)
    changes = [_to_float(row.get("change"), 0.0) for row in (yahoo, nasdaq) if isinstance(row, Mapping)]
    momentum = sum(changes) / max(len(changes), 1)
    return {
        "free_equity_quote_available_norm": 1.0 if yahoo_price > 0.0 or nasdaq_price > 0.0 else 0.0,
        "free_equity_yahoo_volume_norm": _safe_log_norm(_to_float(yahoo.get("volume"), 0.0), denom=18.0),
        "free_equity_nasdaq_volume_norm": _safe_log_norm(_to_float(nasdaq.get("volume"), 0.0), denom=18.0),
        "free_equity_momentum_norm": _signed_centered_norm(momentum, 0.08),
        "free_equity_cross_provider_agreement_norm": _agreement([yahoo_price, nasdaq_price]),
    }


def build_payload(
    *,
    symbols: list[str],
    user_agent: str = USER_AGENT_DEFAULT,
    timeout: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    max_runtime_seconds: float = DEFAULT_MAX_RUNTIME_SECONDS,
    enable_nasdaq: bool = False,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    runtime_budget = max(float(max_runtime_seconds), 0.0)
    deadline = started_monotonic + runtime_budget if runtime_budget > 0.0 else None
    requested = [_normalize_symbol(symbol) for symbol in symbols if _normalize_symbol(symbol)][: max(int(max_symbols), 1)]
    sources = {
        "yahoo_chart": {"ok": False, "symbol_count": 0, "errors": [], "source_confidence_norm": 0.70, "schema_confidence_norm": 0.84},
    }
    if enable_nasdaq:
        sources["nasdaq_quote"] = {"ok": False, "symbol_count": 0, "errors": [], "source_confidence_norm": 0.76, "schema_confidence_norm": 0.84}
    symbols_out: dict[str, Any] = {}
    global_feature_values: dict[str, list[float]] = {key: [] for key in FEATURE_KEYS}
    attempted_symbol_count = 0
    deadline_exceeded = False
    for symbol in requested:
        if not _has_fetch_budget(deadline):
            deadline_exceeded = True
            break
        attempted_symbol_count += 1
        yahoo, yahoo_status = _fetch_yahoo_chart(
            symbol,
            user_agent=user_agent,
            timeout=_bounded_timeout(timeout, deadline),
            retries=0,
        )
        if enable_nasdaq and _has_fetch_budget(deadline):
            nasdaq, nasdaq_status = _fetch_nasdaq_quote(
                symbol,
                user_agent=user_agent,
                timeout=_bounded_timeout(timeout, deadline),
                retries=0,
            )
        elif not enable_nasdaq:
            nasdaq, nasdaq_status = {}, {"ok": False, "error": "nasdaq_disabled"}
        else:
            deadline_exceeded = True
            nasdaq, nasdaq_status = {}, {"ok": False, "error": "deadline_deferred"}
        if yahoo_status.get("ok"):
            sources["yahoo_chart"]["symbol_count"] = int(sources["yahoo_chart"].get("symbol_count", 0) or 0) + 1
        elif yahoo_status.get("error") and not _is_deadline_skip(yahoo_status.get("error")):
            sources["yahoo_chart"]["errors"].append(f"{symbol}:{yahoo_status.get('error')}")
        if enable_nasdaq and nasdaq_status.get("ok"):
            sources["nasdaq_quote"]["symbol_count"] = int(sources["nasdaq_quote"].get("symbol_count", 0) or 0) + 1
        elif enable_nasdaq and nasdaq_status.get("error") and not _is_deadline_skip(nasdaq_status.get("error")):
            sources["nasdaq_quote"]["errors"].append(f"{symbol}:{nasdaq_status.get('error')}")
        features = _symbol_features(yahoo, nasdaq)
        for key, value in features.items():
            global_feature_values.setdefault(key, []).append(float(value))
        symbols_out[symbol] = {
            "ok": bool(features["free_equity_quote_available_norm"] > 0.0),
            "yahoo_chart": yahoo,
            "nasdaq_quote": nasdaq,
            "features": features,
        }
    for row in sources.values():
        row["ok"] = int(row.get("symbol_count", 0) or 0) > 0
        row["errors"] = list(row.get("errors") or [])[:10]
        row["freshness_norm"] = 1.0 if row["ok"] else 0.0
    ok_sources = sum(1 for row in sources.values() if bool(row.get("ok")))
    symbols_with_reference = sum(1 for row in symbols_out.values() if bool(row.get("ok")))
    global_features = {
        key: round(sum(values) / max(len(values), 1), 6) if values else 0.0
        for key, values in global_feature_values.items()
    }
    return {
        "timestamp_utc": now.isoformat(),
        "provider": "free_equity_reference_context",
        "ok": bool(symbols_with_reference > 0 and ok_sources > 0),
        "overall_status": "ready" if symbols_with_reference > 0 and ok_sources > 0 else "degraded",
        "requested_symbol_count": len(requested),
        "attempted_symbol_count": attempted_symbol_count,
        "deferred_symbol_count": max(len(requested) - attempted_symbol_count, 0),
        "deadline_exceeded": bool(deadline_exceeded),
        "max_runtime_seconds": round(runtime_budget, 3),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
        "symbols_with_reference": symbols_with_reference,
        "source_count": len(sources),
        "ok_source_count": ok_sources,
        "sources": sources,
        "symbols": symbols_out,
        "features": global_features,
        "derived": {
            "global_features": global_features,
            "symbol_features": {symbol: row["features"] for symbol, row in symbols_out.items()},
        },
        "safety_contract": {
            "market_data_only": True,
            "live_execution_allowed": False,
            "writes_orders": False,
            "source_policy": "free_public_yahoo_chart_with_optional_nasdaq_reference",
            "nasdaq_enabled": bool(enable_nasdaq),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect free equity reference quotes from no-key public sources.")
    parser.add_argument("--symbols", default=os.getenv("FREE_EQUITY_REFERENCE_SYMBOLS", ""))
    parser.add_argument("--max-symbols", type=int, default=int(os.getenv("FREE_EQUITY_REFERENCE_MAX_SYMBOLS", str(DEFAULT_MAX_SYMBOLS)) or DEFAULT_MAX_SYMBOLS))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("FREE_EQUITY_REFERENCE_TIMEOUT_SECONDS", str(DEFAULT_FETCH_TIMEOUT_SECONDS)) or DEFAULT_FETCH_TIMEOUT_SECONDS))
    parser.add_argument(
        "--enable-nasdaq",
        action="store_true",
        default=str(os.getenv("FREE_EQUITY_REFERENCE_ENABLE_NASDAQ", "0") or "").strip().lower() in {"1", "true", "yes", "on"},
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=float(os.getenv("FREE_EQUITY_REFERENCE_MAX_RUNTIME_SECONDS", str(DEFAULT_MAX_RUNTIME_SECONDS)) or DEFAULT_MAX_RUNTIME_SECONDS),
    )
    parser.add_argument("--user-agent", default=os.getenv("FREE_EQUITY_REFERENCE_USER_AGENT", USER_AGENT_DEFAULT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    symbols = _parse_symbols(args.symbols) if str(args.symbols or "").strip() else _default_symbols()
    payload = build_payload(
        symbols=symbols,
        user_agent=str(args.user_agent or USER_AGENT_DEFAULT),
        timeout=float(args.timeout),
        max_symbols=int(args.max_symbols),
        max_runtime_seconds=float(args.max_runtime_seconds),
        enable_nasdaq=bool(args.enable_nasdaq),
    )
    _write_json(HEALTH_PATH, payload)
    _write_json(EXTERNAL_CONTEXT_PATH, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "free_equity_reference_context "
            f"status={payload.get('overall_status')} "
            f"symbols_with_reference={payload.get('symbols_with_reference')}/{payload.get('requested_symbol_count')} "
            f"sources={payload.get('ok_source_count')}/{payload.get('source_count')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
