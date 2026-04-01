#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


USER_AGENT_DEFAULT = "schwab-trading-bot/1.0"
LIVE_BASE_URL = "https://api.tastyworks.com"
SANDBOX_BASE_URL = "https://api.cert.tastyworks.com"


FEATURE_KEYS = [
    "tasty_iv_rank_norm",
    "tasty_implied_volatility_index_norm",
    "tasty_liquidity_rating_norm",
    "tasty_expected_move_norm",
    "tasty_beta_norm",
    "tasty_watchlist_presence_norm",
]

_DEFAULT_ALIGNMENT_TAIL_BYTES = 8 * 1024 * 1024
_ENDPOINT_UNAVAILABLE_ERR = "endpoint_unavailable"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _parse_ts(raw: Any) -> datetime | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_symbol(raw: str) -> str:
    return str(raw or "").strip().upper().replace(".", "-")


def _parse_symbols(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in str(raw or "").replace("\n", ",").split(","):
        symbol = _normalize_symbol(token)
        if not symbol or symbol in seen:
            continue
        if any(ch in symbol for ch in ("/", "$")):
            continue
        if symbol.endswith("-USD"):
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _default_symbols() -> list[str]:
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


def _zero_feature_map() -> dict[str, float]:
    return {key: 0.0 for key in FEATURE_KEYS}


def _signed_centered_norm(value: float, scale: float) -> float:
    return _clamp01(0.5 + (float(value) / max(float(scale), 1e-8)))


def _http_json(
    url: str,
    *,
    method: str = "GET",
    user_agent: str,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: float = 20.0,
) -> Any:
    payload: bytes | None = None
    req_headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }
    if headers:
        req_headers.update(headers)
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = Request(url=url, headers=req_headers, data=payload, method=method.upper())
    with urlopen(req, timeout=max(float(timeout), 1.0)) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _safe_http_json(**kwargs: Any) -> tuple[Any | None, str | None]:
    try:
        return _http_json(**kwargs), None
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return None, str(exc)


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
    wanted = {str(k).strip().lower() for k in keys if str(k).strip()}
    if not wanted:
        return None
    stack: list[Any] = [node]
    while stack:
        current = stack.pop(0)
        if isinstance(current, dict):
            lowered = {str(k).strip().lower(): v for k, v in current.items()}
            for key in wanted:
                if key in lowered:
                    return lowered[key]
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return None


def _normalize_percentish(raw: Any, *, default: float = 0.0) -> float:
    value = _to_float(raw, default)
    if value <= 0.0:
        return 0.0
    if value <= 1.0:
        return _clamp01(value)
    return _clamp01(value / 100.0)


def _normalize_liquidity_rating(raw: Any) -> float:
    value = _to_float(raw, 0.0)
    if value <= 0.0:
        return 0.0
    if value <= 1.0:
        return _clamp01(value)
    if value <= 5.0:
        return _clamp01(value / 5.0)
    return _clamp01(value / 100.0)


def _derive_option_chain_metrics(payload: Any) -> dict[str, float]:
    items = (((payload or {}).get("data") or {}).get("items")) if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return {
            "chain_available": 0.0,
            "contract_density_norm": 0.0,
            "zero_dte_presence_norm": 0.0,
            "near_term_presence_norm": 0.0,
        }

    contract_count = 0.0
    zero_dte = 0.0
    near_term = 0.0
    for item in items:
        expirations = item.get("expirations") if isinstance(item, dict) else None
        if not isinstance(expirations, list):
            continue
        for expiry in expirations:
            if not isinstance(expiry, dict):
                continue
            dte = _to_float(expiry.get("days-to-expiration"), -1.0)
            strikes = expiry.get("strikes")
            strike_count = float(len(strikes)) if isinstance(strikes, list) else 0.0
            contract_count += strike_count * 2.0
            if 0.0 <= dte <= 1.0 and strike_count > 0.0:
                zero_dte = 1.0
            if 0.0 <= dte <= 7.0 and strike_count > 0.0:
                near_term = 1.0

    return {
        "chain_available": 1.0 if contract_count > 0.0 else 0.0,
        "contract_density_norm": _clamp01(contract_count / 600.0),
        "zero_dte_presence_norm": zero_dte,
        "near_term_presence_norm": near_term,
    }


def _derive_symbol_features(
    *,
    symbol: str,
    option_chain_payload: Any,
    market_metrics_payload: Any,
    instrument_payload: Any,
    watchlist_symbols: set[str],
) -> dict[str, float]:
    chain = _derive_option_chain_metrics(option_chain_payload)
    last_price = _to_float(
        _casefold_get(
            market_metrics_payload,
            "underlying-price",
            "underlying_price",
            "price",
            "mark",
            "close",
            "close-price",
        ),
        0.0,
    )
    expected_move_raw = _casefold_get(market_metrics_payload, "expected-move", "expected_move", "expectedMove")
    expected_move = _to_float(expected_move_raw, 0.0)
    expected_move_norm = 0.0
    if expected_move > 0.0:
        if last_price > 0.0 and expected_move > 1.0:
            expected_move_norm = _clamp01(expected_move / max(last_price * 0.15, 1e-6))
        elif expected_move <= 1.0:
            expected_move_norm = _clamp01(expected_move / 0.10)
        else:
            expected_move_norm = _clamp01(expected_move / 10.0)

    out = {
        "tasty_iv_rank_norm": _normalize_percentish(
            _casefold_get(
                market_metrics_payload,
                "iv-rank",
                "iv_rank",
                "ivRank",
                "implied-volatility-rank",
                "implied_volatility_rank",
            )
        ),
        "tasty_implied_volatility_index_norm": _normalize_percentish(
            _casefold_get(
                market_metrics_payload,
                "implied-volatility-index",
                "implied_volatility_index",
                "impliedVolatilityIndex",
                "implied-volatility",
                "implied_volatility",
            )
        ),
        "tasty_liquidity_rating_norm": _normalize_liquidity_rating(
            _casefold_get(
                market_metrics_payload,
                "liquidity-rating",
                "liquidity_rating",
                "liquidityRating",
            )
        ),
        "tasty_expected_move_norm": expected_move_norm,
        "tasty_beta_norm": _signed_centered_norm(
            _to_float(_casefold_get(market_metrics_payload, "beta"), 0.0),
            3.0,
        ),
        "tasty_watchlist_presence_norm": 1.0 if symbol in watchlist_symbols else 0.0,
    }

    if out["tasty_expected_move_norm"] <= 0.0:
        out["tasty_expected_move_norm"] = max(
            0.0,
            0.55 * chain["contract_density_norm"] + 0.25 * out["tasty_iv_rank_norm"] + 0.20 * chain["near_term_presence_norm"],
        )
    if out["tasty_implied_volatility_index_norm"] <= 0.0:
        out["tasty_implied_volatility_index_norm"] = max(
            0.0,
            0.65 * out["tasty_iv_rank_norm"] + 0.35 * chain["zero_dte_presence_norm"],
        )
    if out["tasty_liquidity_rating_norm"] <= 0.0:
        out["tasty_liquidity_rating_norm"] = max(
            0.0,
            0.70 * chain["contract_density_norm"] + 0.30 * out["tasty_watchlist_presence_norm"],
        )
    if out["tasty_liquidity_rating_norm"] <= 0.0 and isinstance(instrument_payload, dict):
        lendability = str(
            _casefold_get(
                instrument_payload,
                "lendability",
            )
            or ""
        ).strip().lower()
        is_illiquid = bool(
            _casefold_get(
                instrument_payload,
                "is-illiquid",
                "is_illiquid",
            )
        )
        if "easy to borrow" in lendability:
            out["tasty_liquidity_rating_norm"] = 0.9 if not is_illiquid else 0.45
        elif lendability:
            out["tasty_liquidity_rating_norm"] = 0.6 if not is_illiquid else 0.3
    out["tasty_underlying_price"] = max(last_price, 0.0)
    return out


def _mean_feature(items: Iterable[dict[str, float]], key: str) -> float:
    values = [float(row.get(key, 0.0) or 0.0) for row in items]
    if not values:
        return 0.0
    return float(sum(values) / max(len(values), 1))


def _recent_master_control_paths(project_root: Path, since: datetime) -> list[Path]:
    governance_root = project_root / "governance"
    min_date = since.date() - timedelta(days=1)
    out: list[Path] = []
    for path in governance_root.glob("shadow*/master_control_*.jsonl"):
        raw_day = path.stem.rsplit("_", 1)[-1]
        try:
            day = datetime.strptime(raw_day, "%Y%m%d").date()
        except Exception:
            out.append(path)
            continue
        if day >= min_date:
            out.append(path)
    return sorted(out, reverse=True)


def _iter_tail_lines_reverse(path: Path, *, max_bytes: int, chunk_bytes: int = 262144) -> Iterable[str]:
    limit = max(int(max_bytes), 1024)
    step = max(int(chunk_bytes), 1024)
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        position = fh.tell()
        if position <= 0:
            return
        remaining = min(position, limit)
        buffer = b""
        while remaining > 0:
            read_size = min(step, remaining)
            position -= read_size
            fh.seek(position)
            chunk = fh.read(read_size)
            remaining -= read_size
            buffer = chunk + buffer
            parts = buffer.split(b"\n")
            buffer = parts[0]
            for raw in reversed(parts[1:]):
                line = raw.decode("utf-8", "replace").strip()
                if line:
                    yield line
        tail = buffer.decode("utf-8", "replace").strip()
        if tail:
            yield tail


def _row_is_simulated(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    simulate_raw = row.get("simulate")
    if isinstance(simulate_raw, bool):
        return simulate_raw
    return str(simulate_raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_recent_schwab_price_history(
    project_root: Path,
    *,
    since: datetime,
    tail_bytes: int,
    symbols: set[str] | None = None,
) -> dict[str, list[tuple[datetime, float]]]:
    target_symbols = {_normalize_symbol(symbol) for symbol in (symbols or set()) if _normalize_symbol(symbol)}
    history: dict[str, list[tuple[datetime, float]]] = {}
    for path in _recent_master_control_paths(project_root, since):
        try:
            for line in _iter_tail_lines_reverse(path, max_bytes=tail_bytes):
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if _row_is_simulated(row):
                    continue
                ts = _parse_ts(row.get("timestamp_utc"))
                if ts is None:
                    continue
                if ts < since:
                    break
                symbol = _normalize_symbol(row.get("symbol", ""))
                if target_symbols and symbol not in target_symbols:
                    continue
                if not symbol:
                    continue
                market = row.get("market") if isinstance(row.get("market"), dict) else {}
                last_price = _to_float(market.get("last_price"), 0.0)
                if last_price <= 0.0:
                    continue
                history.setdefault(symbol, []).append((ts, last_price))
                if target_symbols and len(history) >= len(target_symbols):
                    break
        except Exception:
            continue
        if target_symbols and len(history) >= len(target_symbols):
            break
    for rows in history.values():
        rows.sort(key=lambda item: item[0])
    return history


def _resolve_reference_price(
    series: list[tuple[datetime, float]],
    *,
    target_ts: datetime,
    tolerance_seconds: float,
) -> tuple[float, datetime] | tuple[None, None]:
    best_price: float | None = None
    best_ts: datetime | None = None
    best_delta: float | None = None
    for row_ts, row_price in series:
        delta = abs((row_ts - target_ts).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_ts = row_ts
            best_price = row_price
    if best_delta is None or best_delta > max(float(tolerance_seconds), 1.0):
        return None, None
    return best_price, best_ts


def _align_symbol_features_with_schwab(
    *,
    symbol: str,
    features: dict[str, float],
    schwab_history: dict[str, list[tuple[datetime, float]]],
    now_utc: datetime,
    sandbox: bool,
    max_relative_spread: float,
    tolerance_minutes: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    series = schwab_history.get(symbol, [])
    target_ts = now_utc - timedelta(minutes=15 if sandbox else 0)
    ref_price, ref_ts = _resolve_reference_price(
        series,
        target_ts=target_ts,
        tolerance_seconds=max(float(tolerance_minutes), 1.0) * 60.0,
    )
    tasty_price = _to_float(features.get("tasty_underlying_price"), 0.0)
    if ref_price is None:
        return _zero_feature_map(), {
            "symbol": symbol,
            "ok": False,
            "reason": "missing_schwab_reference",
            "schwab_price": ref_price,
            "tasty_price": tasty_price,
        }
    if tasty_price <= 0.0:
        out = dict(features)
        out.pop("tasty_underlying_price", None)
        return out, {
            "symbol": symbol,
            "ok": True,
            "reason": "schwab_reference_only",
            "reference_only": True,
            "schwab_price": round(ref_price, 6),
            "tasty_price": tasty_price,
            "schwab_timestamp_utc": ref_ts.isoformat() if isinstance(ref_ts, datetime) else "",
        }
    rel_spread = abs(tasty_price - ref_price) / max(min(tasty_price, ref_price), 1e-8)
    if rel_spread > max(float(max_relative_spread), 1e-6):
        return _zero_feature_map(), {
            "symbol": symbol,
            "ok": False,
            "reason": "relative_spread_exceeded",
            "relative_spread": round(rel_spread, 6),
            "max_relative_spread": float(max_relative_spread),
            "schwab_price": round(ref_price, 6),
            "tasty_price": round(tasty_price, 6),
            "schwab_timestamp_utc": ref_ts.isoformat() if isinstance(ref_ts, datetime) else "",
        }
    out = dict(features)
    out.pop("tasty_underlying_price", None)
    return out, {
        "symbol": symbol,
        "ok": True,
        "relative_spread": round(rel_spread, 6),
        "schwab_price": round(ref_price, 6),
        "tasty_price": round(tasty_price, 6),
        "schwab_timestamp_utc": ref_ts.isoformat() if isinstance(ref_ts, datetime) else "",
    }


def _post_session(base_url: str, *, user_agent: str, login: str, password: str, timeout: float) -> tuple[str | None, str | None]:
    payload, err = _safe_http_json(
        url=f"{base_url}/sessions",
        method="POST",
        user_agent=user_agent,
        body={"login": login, "password": password, "remember-me": True},
        timeout=timeout,
    )
    if err or not isinstance(payload, dict):
        return None, err or "invalid_response"
    token = _casefold_get(payload, "session-token", "session_token")
    if not isinstance(token, str) or not token.strip():
        return None, "session_token_missing"
    return token.strip(), None


def _auth_headers(session_token: str) -> dict[str, str]:
    return {"Authorization": session_token}


def _fetch_public_watchlists(base_url: str, *, user_agent: str, session_token: str, timeout: float) -> tuple[Any | None, str | None]:
    return _safe_http_json(
        url=f"{base_url}/public-watchlists",
        user_agent=user_agent,
        headers=_auth_headers(session_token),
        timeout=timeout,
    )


def _watchlist_symbols(payload: Any) -> set[str]:
    out: set[str] = set()
    items = (((payload or {}).get("data") or {}).get("items")) if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return out
    for item in items:
        stack = [item]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                symbol = current.get("symbol")
                if isinstance(symbol, str) and symbol.strip():
                    out.add(_normalize_symbol(symbol))
                stack.extend(current.values())
            elif isinstance(current, list):
                stack.extend(current)
    return out


def _fetch_option_chain_nested(base_url: str, *, symbol: str, user_agent: str, session_token: str, timeout: float) -> tuple[Any | None, str | None]:
    return _safe_http_json(
        url=f"{base_url}/option-chains/{quote(symbol)}/nested",
        user_agent=user_agent,
        headers=_auth_headers(session_token),
        timeout=timeout,
    )


def _fetch_equity_instrument(base_url: str, *, symbol: str, user_agent: str, session_token: str, timeout: float) -> tuple[Any | None, str | None]:
    return _safe_http_json(
        url=f"{base_url}/instruments/equities/{quote(symbol)}",
        user_agent=user_agent,
        headers=_auth_headers(session_token),
        timeout=timeout,
    )


def _fetch_market_metrics(
    base_url: str,
    *,
    symbol: str,
    user_agent: str,
    session_token: str,
    timeout: float,
    capability_state: dict[str, Any] | None = None,
) -> tuple[Any | None, str | None]:
    if isinstance(capability_state, dict) and capability_state.get("unsupported"):
        return None, _ENDPOINT_UNAVAILABLE_ERR
    headers = _auth_headers(session_token)
    candidates = [
        f"{base_url}/market-metrics/{quote(symbol)}",
        f"{base_url}/market-metrics?symbol={quote(symbol)}",
        f"{base_url}/market-metrics?symbols[]={quote(symbol)}",
        f"{base_url}/market-metrics?underlying-symbol={quote(symbol)}",
    ]
    last_err = "not_attempted"
    errs: list[str] = []
    for url in candidates:
        payload, err = _safe_http_json(url=url, user_agent=user_agent, headers=headers, timeout=timeout)
        if payload is not None:
            if isinstance(capability_state, dict):
                capability_state["unsupported"] = False
            return payload, None
        if err:
            last_err = err
            errs.append(err)
    if errs and all(_http_error_code(err) == 404 for err in errs):
        if isinstance(capability_state, dict):
            capability_state["unsupported"] = True
            capability_state["last_error"] = errs[-1]
        return None, _ENDPOINT_UNAVAILABLE_ERR
    return None, last_err


def collect_tastytrade_context(
    *,
    login: str,
    password: str,
    symbols: list[str],
    user_agent: str,
    timeout_seconds: float,
    sandbox: bool,
    schwab_alignment_hours: float,
    max_schwab_relative_spread: float,
    schwab_tolerance_minutes: float,
    schwab_alignment_max_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = datetime.now(timezone.utc)
    base_url = SANDBOX_BASE_URL if sandbox else LIVE_BASE_URL
    status: dict[str, Any] = {
        "timestamp_utc": now.isoformat(),
        "provider": "tastytrade_context",
        "base_url": base_url,
        "sandbox": bool(sandbox),
        "ok": False,
        "symbols_requested": len(symbols),
        "symbols_with_chain": 0,
        "symbols_with_metrics": 0,
        "alignment_compared": 0,
        "alignment_missing_reference": 0,
        "alignment_reference_only": 0,
        "alignment_worst_relative_spread": 0.0,
        "alignment_max_relative_spread": float(max_schwab_relative_spread),
        "alignment_ok": True,
        "alignment_offenders": [],
        "errors": [],
    }
    payload: dict[str, Any] = {
        "timestamp_utc": now.isoformat(),
        "provider": "tastytrade_context",
        "symbols": symbols,
        "derived": {"global_features": {k: 0.0 for k in FEATURE_KEYS}, "symbol_features": {}},
        "sources": {},
    }

    if not login.strip() or not password.strip():
        status["errors"].append("credentials_missing")
        payload["sources"]["session"] = {"ok": False, "error": "credentials_missing"}
        return payload, status

    session_token, session_err = _post_session(
        base_url,
        user_agent=user_agent,
        login=login,
        password=password,
        timeout=timeout_seconds,
    )
    payload["sources"]["session"] = {"ok": session_token is not None, "error": session_err}
    if session_token is None:
        status["errors"].append(f"session:{session_err}")
        return payload, status

    watchlists_payload, watchlists_err = _fetch_public_watchlists(
        base_url,
        user_agent=user_agent,
        session_token=session_token,
        timeout=timeout_seconds,
    )
    watchlist_symbols = _watchlist_symbols(watchlists_payload)
    payload["sources"]["watchlists"] = {
        "ok": watchlists_payload is not None,
        "error": watchlists_err,
        "symbol_count": len(watchlist_symbols),
    }

    schwab_history = _load_recent_schwab_price_history(
        PROJECT_ROOT,
        since=now - timedelta(hours=max(float(schwab_alignment_hours), 1.0)),
        tail_bytes=max(int(schwab_alignment_max_bytes), 1024),
        symbols=set(symbols),
    )
    payload["sources"]["schwab_alignment"] = {
        "ok": bool(schwab_history),
        "reference_symbol_count": len(schwab_history),
        "tail_bytes": max(int(schwab_alignment_max_bytes), 1024),
    }

    symbol_features: dict[str, dict[str, float]] = {}
    market_metrics_capability: dict[str, Any] = {}
    for symbol in symbols:
        option_chain_payload, option_chain_err = _fetch_option_chain_nested(
            base_url,
            symbol=symbol,
            user_agent=user_agent,
            session_token=session_token,
            timeout=timeout_seconds,
        )
        metrics_payload, metrics_err = _fetch_market_metrics(
            base_url,
            symbol=symbol,
            user_agent=user_agent,
            session_token=session_token,
            timeout=timeout_seconds,
            capability_state=market_metrics_capability,
        )
        instrument_payload, instrument_err = _fetch_equity_instrument(
            base_url,
            symbol=symbol,
            user_agent=user_agent,
            session_token=session_token,
            timeout=timeout_seconds,
        )
        if option_chain_payload is not None:
            status["symbols_with_chain"] = int(status["symbols_with_chain"]) + 1
        if metrics_payload is not None or instrument_payload is not None:
            status["symbols_with_metrics"] = int(status["symbols_with_metrics"]) + 1
        if option_chain_payload is None and metrics_payload is None and instrument_payload is None:
            payload["sources"][symbol] = {
                "option_chain_ok": False,
                "option_chain_error": option_chain_err,
                "market_metrics_ok": False,
                "market_metrics_error": metrics_err,
                "instrument_ok": False,
                "instrument_error": instrument_err,
            }
            continue

        feats = _derive_symbol_features(
            symbol=symbol,
            option_chain_payload=option_chain_payload,
            market_metrics_payload=metrics_payload,
            instrument_payload=instrument_payload,
            watchlist_symbols=watchlist_symbols,
        )
        aligned_feats, alignment = _align_symbol_features_with_schwab(
            symbol=symbol,
            features=feats,
            schwab_history=schwab_history,
            now_utc=now,
            sandbox=sandbox,
            max_relative_spread=max_schwab_relative_spread,
            tolerance_minutes=schwab_tolerance_minutes,
        )
        if alignment.get("ok"):
            if alignment.get("reference_only"):
                status["alignment_reference_only"] = int(status.get("alignment_reference_only", 0) or 0) + 1
            else:
                status["alignment_compared"] = int(status.get("alignment_compared", 0) or 0) + 1
                status["alignment_worst_relative_spread"] = max(
                    float(status.get("alignment_worst_relative_spread", 0.0) or 0.0),
                    float(alignment.get("relative_spread", 0.0) or 0.0),
                )
        else:
            reason = str(alignment.get("reason", "") or "")
            if reason == "missing_schwab_reference":
                status["alignment_missing_reference"] = int(status.get("alignment_missing_reference", 0) or 0) + 1
            else:
                status["alignment_ok"] = False
                offenders = status.get("alignment_offenders")
                if not isinstance(offenders, list):
                    offenders = []
                    status["alignment_offenders"] = offenders
                offenders.append(alignment)
        symbol_features[symbol] = aligned_feats
        payload["sources"][symbol] = {
            "option_chain_ok": option_chain_payload is not None,
            "option_chain_error": option_chain_err,
            "market_metrics_ok": metrics_payload is not None,
            "market_metrics_error": metrics_err,
            "instrument_ok": instrument_payload is not None,
            "instrument_error": instrument_err,
            "alignment": alignment,
        }

    payload["derived"]["symbol_features"] = symbol_features
    payload["derived"]["global_features"] = {key: _mean_feature(symbol_features.values(), key) for key in FEATURE_KEYS}
    payload["sources"]["market_metrics"] = {
        "ok": not bool(market_metrics_capability.get("unsupported")),
        "error": market_metrics_capability.get("last_error") if market_metrics_capability.get("unsupported") else None,
    }
    status["alignment_offenders"] = list(status.get("alignment_offenders", []))[:50]
    status["ok"] = bool(symbol_features) and bool(status.get("alignment_ok", True))
    return payload, status


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect tastytrade context for options/dividend enrichment.")
    parser.add_argument("--login", default=os.getenv("TASTYTRADE_LOGIN", ""))
    parser.add_argument("--password", default=os.getenv("TASTYTRADE_PASSWORD", ""))
    parser.add_argument("--symbols", default=os.getenv("TASTYTRADE_SYMBOLS", ""))
    parser.add_argument("--timeout-seconds", type=float, default=float(os.getenv("TASTYTRADE_TIMEOUT_SECONDS", "20")))
    parser.add_argument("--user-agent", default=os.getenv("TASTYTRADE_USER_AGENT", USER_AGENT_DEFAULT))
    parser.add_argument("--sandbox", action="store_true", default=os.getenv("TASTYTRADE_SANDBOX", "0").strip() == "1")
    parser.add_argument("--schwab-alignment-hours", type=float, default=float(os.getenv("TASTYTRADE_SCHWAB_ALIGNMENT_HOURS", "6")))
    parser.add_argument("--max-schwab-relative-spread", type=float, default=float(os.getenv("TASTYTRADE_MAX_SCHWAB_REL_SPREAD", "0.05")))
    parser.add_argument("--schwab-tolerance-minutes", type=float, default=float(os.getenv("TASTYTRADE_SCHWAB_TOLERANCE_MINUTES", "25")))
    parser.add_argument(
        "--schwab-alignment-max-bytes",
        type=int,
        default=int(os.getenv("TASTYTRADE_SCHWAB_ALIGNMENT_MAX_BYTES", str(_DEFAULT_ALIGNMENT_TAIL_BYTES))),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols) if str(args.symbols or "").strip() else _default_symbols()
    payload, status = collect_tastytrade_context(
        login=str(args.login or ""),
        password=str(args.password or ""),
        symbols=symbols,
        user_agent=str(args.user_agent or USER_AGENT_DEFAULT),
        timeout_seconds=float(args.timeout_seconds),
        sandbox=bool(args.sandbox),
        schwab_alignment_hours=float(args.schwab_alignment_hours),
        max_schwab_relative_spread=float(args.max_schwab_relative_spread),
        schwab_tolerance_minutes=float(args.schwab_tolerance_minutes),
        schwab_alignment_max_bytes=int(args.schwab_alignment_max_bytes),
    )

    _write_json(PROJECT_ROOT / "exports" / "external_context" / "tastytrade_context_latest.json", payload)
    _write_json(PROJECT_ROOT / "governance" / "health" / "tastytrade_context_sync_latest.json", status)

    if args.json:
        print(json.dumps(status, ensure_ascii=True))
    else:
        print(
            "tastytrade_context ok={ok} symbols={symbols} chains={chains} metrics={metrics}".format(
                ok=str(bool(status.get("ok", False))).lower(),
                symbols=len(symbols),
                chains=int(status.get("symbols_with_chain", 0) or 0),
                metrics=int(status.get("symbols_with_metrics", 0) or 0),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
