#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.accountability import safe_append_jsonl, safe_write_json_atomic
from scripts.brokers.schwab.common import build_schwab_trader


DEFAULT_SYMBOL = "SPCX"
DEFAULT_PROXY_SYMBOLS = "TSLA,RKLB,ASTS,LUNR,ARKX,XAR,ITA,QQQ,SMH,VIXY,UUP"
DEFAULT_DRAWDOWN_BANDS = "0.05,0.10,0.15,0.20"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "spacex_ipo_downside_watch_state.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "spacex_ipo_downside_watch_latest.json"
DEFAULT_ALERT_LATEST_PATH = PROJECT_ROOT / "governance" / "alerts" / "critical_latest_spacex_ipo_downside_watch.json"
DEFAULT_UNTIL_UTC = "2026-06-13T01:00:00+00:00"
_SCHWAB_TRADER: Any | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_utc(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _parse_csv(value: str) -> list[str]:
    out: list[str] = []
    for part in str(value or "").split(","):
        token = str(part or "").strip().upper()
        if token and token not in out:
            out.append(token)
    return out


def _parse_bands(value: str) -> list[float]:
    bands: list[float] = []
    for part in str(value or "").split(","):
        raw = str(part or "").strip()
        if not raw:
            continue
        try:
            band = float(raw)
        except Exception:
            continue
        if band > 1.0:
            band = band / 100.0
        if 0.0 < band < 1.0:
            bands.append(float(band))
    return sorted(set(round(band, 6) for band in bands))


def _write_json(path: Path, payload: Mapping[str, Any], *, source: str) -> None:
    safe_write_json_atomic(path, dict(payload), project_root=str(PROJECT_ROOT), source=source)


def _append_alert_event(row: Mapping[str, Any]) -> None:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = PROJECT_ROOT / "governance" / "alerts" / f"critical_events_{day}.jsonl"
    safe_append_jsonl(path, dict(row), project_root=str(PROJECT_ROOT), source="spacex_ipo_downside_watch")


def _quote_from_snapshot(symbol: str, payload: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    bid = _safe_float(payload.get("bid_price", payload.get("bidPrice", payload.get("bid"))), 0.0)
    ask = _safe_float(payload.get("ask_price", payload.get("askPrice", payload.get("ask"))), 0.0)
    mark = _safe_float(payload.get("mark_price", payload.get("markPrice", payload.get("mark"))), 0.0)
    last = _safe_float(
        payload.get(
            "last_price",
            payload.get(
                "lastPrice",
                payload.get("regularMarketLastPrice", payload.get("price", payload.get("closePrice"))),
            ),
        ),
        0.0,
    )
    if last <= 0.0:
        last = max(mark, bid, ask, 0.0)
    spread_bps = 0.0
    if bid > 0.0 and ask > bid and last > 0.0:
        spread_bps = ((ask - bid) / max(last, ask, bid, 1e-8)) * 10000.0
    return {
        "ok": bool(last > 0.0),
        "source": source,
        "symbol": symbol.upper(),
        "last_price": round(float(last), 6),
        "bid_price": round(float(bid), 6),
        "ask_price": round(float(ask), 6),
        "mark_price": round(float(mark), 6),
        "spread_bps": round(float(spread_bps), 3),
        "timestamp_utc": _now_iso(),
    }


def _matches_symbol_key(key: Any, symbol: str) -> bool:
    if not isinstance(key, str):
        return False
    normalized_key = "".join(ch for ch in key.upper() if ch.isalnum())
    normalized_symbol = "".join(ch for ch in symbol.upper() if ch.isalnum())
    return normalized_key == normalized_symbol


def _find_symbol_quote(payload: Any, symbol: str, *, source: str, depth: int = 0) -> dict[str, Any] | None:
    if depth > 7:
        return None
    if isinstance(payload, Mapping):
        symbol_field = str(payload.get("symbol") or payload.get("ticker") or "").strip().upper()
        if symbol_field == symbol.upper():
            quote = _quote_from_snapshot(symbol, payload, source=source)
            if quote["ok"]:
                return quote
        for key, value in payload.items():
            if _matches_symbol_key(key, symbol) and isinstance(value, Mapping):
                quote = _quote_from_snapshot(symbol, value, source=source)
                if quote["ok"]:
                    return quote
        for value in payload.values():
            if isinstance(value, (Mapping, list)):
                quote = _find_symbol_quote(value, symbol, source=source, depth=depth + 1)
                if quote:
                    return quote
    elif isinstance(payload, list):
        for item in payload[:200]:
            quote = _find_symbol_quote(item, symbol, source=source, depth=depth + 1)
            if quote:
                return quote
    return None


def _fallback_quote_paths(project_root: Path) -> list[Path]:
    return [
        project_root / "exports" / "external_context" / "market_crypto_correlation_latest.json",
        project_root / "exports" / "external_context" / "market_crypto_correlation_cache_latest.json",
        project_root / "exports" / "external_context" / "market_micro_latest.json",
        project_root / "exports" / "external_context" / "ticker_news_context_latest.json",
        project_root / "governance" / "health" / "market_posture_control_latest.json",
    ]


def fetch_fallback_quote(symbol: str, *, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    for path in _fallback_quote_paths(project_root):
        payload = _read_json(path)
        if not payload:
            continue
        quote = _find_symbol_quote(payload, symbol, source=f"snapshot:{path.name}")
        if quote:
            return quote
    return {"ok": False, "source": "snapshot", "symbol": symbol.upper(), "error": "quote_not_found"}


def _authenticate_trader(trader: Any, *, quiet_auth: bool) -> None:
    if not quiet_auth:
        trader.authenticate()
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            trader.authenticate()


def fetch_broker_quote(symbol: str, *, broker: str = "schwab", quiet_auth: bool = True) -> dict[str, Any]:
    global _SCHWAB_TRADER
    if broker.lower() != "schwab":
        return {"ok": False, "source": broker, "symbol": symbol.upper(), "error": f"unsupported_broker:{broker}"}
    old_allow = os.environ.get("ALLOW_ORDER_EXECUTION")
    old_market_data = os.environ.get("MARKET_DATA_ONLY")
    os.environ["ALLOW_ORDER_EXECUTION"] = "0"
    os.environ["MARKET_DATA_ONLY"] = "1"
    try:
        trader = _SCHWAB_TRADER
        if trader is None:
            trader = build_schwab_trader(PROJECT_ROOT, mode="shadow", missing_credentials_message="Schwab credentials are required for SPCX quote watch")
            _authenticate_trader(trader, quiet_auth=quiet_auth)
            _SCHWAB_TRADER = trader
        fetched = trader._fetch_live_quote(symbol=symbol)
        if not bool(fetched.get("ok", False)) and str(fetched.get("error") or "") == "client_not_authenticated":
            _authenticate_trader(trader, quiet_auth=quiet_auth)
            fetched = trader._fetch_live_quote(symbol=symbol)
    except Exception as exc:
        return {"ok": False, "source": "schwab_quote", "symbol": symbol.upper(), "error": f"{type(exc).__name__}:{exc}"}
    finally:
        if old_allow is None:
            os.environ.pop("ALLOW_ORDER_EXECUTION", None)
        else:
            os.environ["ALLOW_ORDER_EXECUTION"] = old_allow
        if old_market_data is None:
            os.environ.pop("MARKET_DATA_ONLY", None)
        else:
            os.environ["MARKET_DATA_ONLY"] = old_market_data

    if not bool(fetched.get("ok", False)):
        return {
            "ok": False,
            "source": "schwab_quote",
            "symbol": symbol.upper(),
            "error": str(fetched.get("error") or "quote_fetch_failed"),
        }
    snapshot = fetched.get("quote_snapshot") if isinstance(fetched.get("quote_snapshot"), Mapping) else {}
    quote = _quote_from_snapshot(symbol, snapshot, source="schwab_quote")
    if not quote["ok"]:
        quote["error"] = "quote_missing_price"
    return quote


def select_quote(symbol: str, *, broker: str, disable_broker: bool = False) -> dict[str, Any]:
    if not disable_broker:
        broker_quote = fetch_broker_quote(symbol, broker=broker)
        if broker_quote.get("ok"):
            return broker_quote
    fallback = fetch_fallback_quote(symbol)
    if fallback.get("ok"):
        if not disable_broker:
            fallback["broker_error"] = broker_quote.get("error", "broker_quote_unavailable")
        return fallback
    if disable_broker:
        return fallback
    return {
        "ok": False,
        "source": "none",
        "symbol": symbol.upper(),
        "error": str(broker_quote.get("error") or fallback.get("error") or "quote_unavailable"),
        "broker_error": str(broker_quote.get("error") or ""),
        "fallback_error": str(fallback.get("error") or ""),
    }


def _drop_ratio(reference: float, last_price: float) -> float:
    if reference <= 0.0 or last_price <= 0.0:
        return 0.0
    return max((reference - last_price) / max(reference, 1e-8), 0.0)


def _crossed_band(value: float, bands: Iterable[float], alerted: set[str], prefix: str) -> float:
    crossed = [band for band in bands if value >= band and f"{prefix}:{band:.6f}" not in alerted]
    return max(crossed) if crossed else 0.0


def evaluate_watch(
    *,
    symbol: str,
    quote: Mapping[str, Any],
    state: Mapping[str, Any],
    bands: list[float],
    ipo_price: float = 0.0,
    spread_bps_alert: float = 500.0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    now = _now_iso()
    symbol = symbol.strip().upper() or DEFAULT_SYMBOL
    previous_state = dict(state or {})
    new_state = dict(previous_state)
    new_state.setdefault("symbol", symbol)
    alerted = {str(item) for item in new_state.get("alerted") or [] if str(item).strip()}

    if not bool(quote.get("ok", False)):
        payload = {
            "timestamp_utc": now,
            "ok": True,
            "overall_status": "waiting_for_first_quote",
            "symbol": symbol,
            "quote": dict(quote),
            "alert": {"triggered": False},
            "policy": "monitoring_only_no_order_instruction",
            "recommended_actions": [
                "wait for first broker quote before trusting SPCX-specific downside metrics",
                "keep the macro bulletin active so proxy symbols remain in context",
            ],
        }
        new_state["last_status"] = payload["overall_status"]
        new_state["last_quote_error"] = str(quote.get("error") or "quote_unavailable")
        return payload, new_state, None

    last_price = _safe_float(quote.get("last_price"), 0.0)
    first_print = _safe_float(new_state.get("first_print_price"), 0.0)
    if first_print <= 0.0 and last_price > 0.0:
        first_print = last_price
        new_state["first_print_price"] = round(first_print, 6)
        new_state["first_print_timestamp_utc"] = now

    high_price = max(_safe_float(new_state.get("high_price"), 0.0), last_price)
    if high_price != _safe_float(new_state.get("high_price"), 0.0):
        new_state["high_price"] = round(high_price, 6)
        new_state["high_timestamp_utc"] = now

    drop_from_first = _drop_ratio(first_print, last_price)
    drop_from_high = _drop_ratio(high_price, last_price)
    drop_from_ipo = _drop_ratio(ipo_price, last_price)
    spread_bps = _safe_float(quote.get("spread_bps"), 0.0)

    checks = {
        "from_first_print": _crossed_band(drop_from_first, bands, alerted, "from_first_print"),
        "from_high": _crossed_band(drop_from_high, bands, alerted, "from_high"),
        "from_ipo_price": _crossed_band(drop_from_ipo, bands, alerted, "from_ipo_price") if ipo_price > 0.0 else 0.0,
    }
    spread_trigger = bool(spread_bps_alert > 0.0 and spread_bps >= spread_bps_alert and "spread_bps" not in alerted)
    crossed_checks = [(name, band) for name, band in checks.items() if band > 0.0]
    triggered = bool(crossed_checks or spread_trigger)

    metrics = {
        "last_price": round(last_price, 6),
        "first_print_price": round(first_print, 6),
        "high_price": round(high_price, 6),
        "ipo_price": round(float(ipo_price), 6),
        "drop_from_first_print_pct": round(drop_from_first * 100.0, 3),
        "drop_from_high_pct": round(drop_from_high * 100.0, 3),
        "drop_from_ipo_price_pct": round(drop_from_ipo * 100.0, 3),
        "spread_bps": round(spread_bps, 3),
    }
    status = "alert" if triggered else "armed"
    if first_print == last_price and not previous_state.get("first_print_price"):
        status = "first_quote_seen"

    alert: dict[str, Any] | None = None
    if triggered:
        reasons: list[str] = []
        for check_name, crossed_band in crossed_checks:
            alerted.add(f"{check_name}:{crossed_band:.6f}")
            reasons.append(f"{check_name}_drop_ge_{crossed_band * 100.0:.0f}pct")
        if spread_trigger:
            alerted.add("spread_bps")
            reasons.append(f"spread_bps_ge_{spread_bps_alert:.0f}")
        new_state["alerted"] = sorted(alerted)
        headline_metric = max(metrics["drop_from_first_print_pct"], metrics["drop_from_high_pct"], metrics["drop_from_ipo_price_pct"])
        message = (
            f"SPCX downside watch fired: last={last_price:.2f}, "
            f"max_drop={headline_metric:.1f}%, spread={spread_bps:.0f}bps. "
            "Observation only; no automatic order."
        )
        alert = {
            "timestamp_utc": now,
            "severity": "critical",
            "event": "spacex_ipo_downside_watch",
            "message": message,
            "broker": "schwab",
            "profile": "event_intelligence",
            "domain": "equities",
            "details": {
                "symbol": symbol,
                "reasons": reasons,
                "metrics": metrics,
                "quote": dict(quote),
                "policy": "monitoring_only_no_order_instruction",
            },
        }

    payload = {
        "timestamp_utc": now,
        "ok": True,
        "overall_status": status,
        "symbol": symbol,
        "quote": dict(quote),
        "metrics": metrics,
        "bands": [round(band, 6) for band in bands],
        "spread_bps_alert": round(float(spread_bps_alert), 3),
        "alert": {"triggered": bool(alert is not None), "payload": alert or {}},
        "policy": "monitoring_only_no_order_instruction",
        "recommended_actions": [
            "verify the first reliable broker quote before interpreting early SPCX indicators",
            "watch first-print, high-watermark, VWAP/opening-range, halt/reopen, spread, and proxy-basket weakness",
            "do not use this watcher as an automatic short, buy, or execution signal",
        ],
    }
    new_state["last_status"] = payload["overall_status"]
    new_state["last_quote"] = dict(quote)
    new_state["last_metrics"] = dict(metrics)
    new_state["updated_at_utc"] = now
    return payload, new_state, alert


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    symbol = str(args.symbol or DEFAULT_SYMBOL).strip().upper()
    state_path = Path(args.state_file).expanduser()
    out_path = Path(args.out_file).expanduser()
    alert_path = Path(args.alert_latest_file).expanduser()
    state = _read_json(state_path)
    bands = _parse_bands(args.drawdown_bands)
    if not bands:
        bands = _parse_bands(DEFAULT_DRAWDOWN_BANDS)
    if args.quote_json:
        try:
            quote_obj = json.loads(str(args.quote_json))
        except Exception as exc:
            quote_obj = {"ok": False, "source": "quote_json", "symbol": symbol, "error": f"invalid_quote_json:{exc}"}
        quote = quote_obj if isinstance(quote_obj, dict) else {"ok": False, "source": "quote_json", "symbol": symbol, "error": "invalid_quote_json"}
    else:
        quote = select_quote(symbol, broker=str(args.broker or "schwab"), disable_broker=bool(args.disable_broker))

    payload, new_state, alert = evaluate_watch(
        symbol=symbol,
        quote=quote,
        state=state,
        bands=bands,
        ipo_price=_safe_float(args.ipo_price, 0.0),
        spread_bps_alert=_safe_float(args.spread_bps_alert, 500.0),
    )
    payload["proxy_symbols"] = _parse_csv(str(args.proxy_symbols or DEFAULT_PROXY_SYMBOLS))
    _write_json(state_path, new_state, source="spacex_ipo_downside_watch.state")
    _write_json(out_path, payload, source="spacex_ipo_downside_watch.latest")
    if alert is not None and not bool(args.no_alert_write):
        _write_json(alert_path, alert, source="spacex_ipo_downside_watch.alert_latest")
        _append_alert_event(alert)
    return payload


def expired_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "timestamp_utc": _now_iso(),
        "ok": True,
        "overall_status": "expired",
        "symbol": str(args.symbol or DEFAULT_SYMBOL).strip().upper(),
        "policy": "monitoring_only_no_order_instruction",
        "until_utc": str(args.until_utc or ""),
        "alert": {"triggered": False},
        "recommended_actions": ["disable the launchd agent or leave it idle until another IPO watch is configured"],
    }
    _write_json(Path(args.out_file).expanduser(), payload, source="spacex_ipo_downside_watch.expired")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor SpaceX/SPCX IPO downside behavior and publish critical observation alerts.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--proxy-symbols", default=DEFAULT_PROXY_SYMBOLS)
    parser.add_argument("--broker", default="schwab")
    parser.add_argument("--disable-broker", action="store_true")
    parser.add_argument("--drawdown-bands", default=os.getenv("SPACEX_IPO_DRAWDOWN_BANDS", DEFAULT_DRAWDOWN_BANDS))
    parser.add_argument("--ipo-price", type=float, default=float(os.getenv("SPACEX_IPO_PRICE", "0") or 0.0))
    parser.add_argument("--spread-bps-alert", type=float, default=float(os.getenv("SPACEX_IPO_SPREAD_BPS_ALERT", "500") or 500.0))
    parser.add_argument("--poll-seconds", type=float, default=float(os.getenv("SPACEX_IPO_WATCH_POLL_SECONDS", "30") or 30.0))
    parser.add_argument("--until-utc", default=os.getenv("SPACEX_IPO_WATCH_UNTIL_UTC", DEFAULT_UNTIL_UTC))
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--alert-latest-file", default=str(DEFAULT_ALERT_LATEST_PATH))
    parser.add_argument("--quote-json", default="", help="Testing hook: provide a quote object instead of fetching broker/snapshot data.")
    parser.add_argument("--no-alert-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    until_dt = _parse_utc(str(args.until_utc or ""))
    if until_dt is not None and datetime.now(timezone.utc) >= until_dt:
        payload = expired_payload(args)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(f"spacex_ipo_watch status=expired symbol={payload.get('symbol')}")
        return 0

    if not args.loop:
        payload = run_once(args)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(
                f"spacex_ipo_watch status={payload.get('overall_status')} "
                f"symbol={payload.get('symbol')} alert={bool((payload.get('alert') or {}).get('triggered'))}"
            )
        return 0

    while True:
        if until_dt is not None and datetime.now(timezone.utc) >= until_dt:
            payload = expired_payload(args)
            print(json.dumps(payload, ensure_ascii=True), flush=True)
            return 0
        payload = run_once(args)
        print(json.dumps(payload, ensure_ascii=True), flush=True)
        time.sleep(max(float(args.poll_seconds), 5.0))


if __name__ == "__main__":
    raise SystemExit(main())
