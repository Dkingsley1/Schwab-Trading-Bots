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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.collector_transport import attach_collection_confidence, fetch_json
from core.log_tail import iter_tail_lines_reverse as _iter_tail_lines_reverse


USER_AGENT_DEFAULT = "schwab-trading-bot/1.0"
LIVE_BASE_URL = "https://api.tastyworks.com"
SANDBOX_BASE_URL = "https://api.cert.tastyworks.com"
_PLACEHOLDER_SECRET_TOKENS = {
    "YOUR_REAL_LOGIN",
    "YOUR_REAL_PASSWORD",
    "YOUR_REAL_CLIENT_ID",
    "YOUR_REAL_KEY",
    "CHANGEME",
    "CHANGE_ME",
    "REPLACE_ME",
}


FEATURE_KEYS = [
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
]

_DEFAULT_ALIGNMENT_TAIL_BYTES = 8 * 1024 * 1024
_ENDPOINT_UNAVAILABLE_ERR = "endpoint_unavailable"
SOURCE_CONTRACTS = {
    "tastytrade_session": {"source_confidence_norm": 0.96, "schema_confidence_norm": 0.95},
    "tastytrade_public_watchlists": {"source_confidence_norm": 0.9, "schema_confidence_norm": 0.92},
    "tastytrade_option_chain": {"source_confidence_norm": 0.95, "schema_confidence_norm": 0.94},
    "tastytrade_equity_instrument": {"source_confidence_norm": 0.93, "schema_confidence_norm": 0.93},
    "tastytrade_market_metrics": {"source_confidence_norm": 0.94, "schema_confidence_norm": 0.93},
    "schwab_alignment_reference": {"source_confidence_norm": 0.9, "schema_confidence_norm": 0.9},
}


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


def _source_contract(source_name: str) -> dict[str, float]:
    row = SOURCE_CONTRACTS.get(str(source_name or "").strip(), {})
    return {
        "source_confidence_norm": float(row.get("source_confidence_norm", 0.9) or 0.9),
        "schema_confidence_norm": float(row.get("schema_confidence_norm", 0.9) or 0.9),
    }


def _source_contract_name(url: str, *, method: str = "GET") -> str:
    text = str(url or "")
    upper_method = str(method or "GET").upper()
    if text.endswith("/sessions") and upper_method == "POST":
        return "tastytrade_session"
    if "/public-watchlists" in text:
        return "tastytrade_public_watchlists"
    if "/option-chains/" in text:
        return "tastytrade_option_chain"
    if "/instruments/equities/" in text:
        return "tastytrade_equity_instrument"
    if "/market-metrics" in text:
        return "tastytrade_market_metrics"
    return "tastytrade_market_metrics"


def _http_json(
    url: str,
    *,
    method: str = "GET",
    user_agent: str,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: float = 20.0,
) -> Any:
    source_name = _source_contract_name(url, method=method)
    contract = _source_contract(source_name)
    req_headers = dict(headers or {})
    if body is not None and not any(str(key).lower() == "content-type" for key in req_headers):
        req_headers["Content-Type"] = "application/json"
    result = fetch_json(
        url=url,
        method=method,
        user_agent=user_agent,
        headers=req_headers,
        body=(json.dumps(body).encode("utf-8") if body is not None else None),
        timeout=timeout,
        collector_key="tastytrade_context",
        source_name=source_name,
        entity_key=url,
        project_root=PROJECT_ROOT,
        source_confidence_norm=contract["source_confidence_norm"],
        schema_confidence_norm=contract["schema_confidence_norm"],
    )
    if not bool(result.get("ok", False)):
        raise RuntimeError(str(result.get("error") or "http_json_failed"))
    return result.get("json")


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


def _looks_placeholder_secret(raw: Any) -> bool:
    text = str(raw or "").strip()
    if not text:
        return True
    upper = text.upper()
    if upper in _PLACEHOLDER_SECRET_TOKENS:
        return True
    return upper.startswith("YOUR_REAL_")


def _is_unauthorized_error(raw: str | None) -> bool:
    code = _http_error_code(raw)
    if code == 401:
        return True
    text = str(raw or "").strip().lower()
    return "unauthorized" in text or "authentication" in text


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


def _iter_chain_contracts(payload: Any) -> Iterable[dict[str, float]]:
    items = (((payload or {}).get("data") or {}).get("items")) if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []

    rows: list[dict[str, float]] = []
    for item in items:
        expirations = item.get("expirations") if isinstance(item, dict) else None
        if not isinstance(expirations, list):
            continue
        for expiry in expirations:
            if not isinstance(expiry, dict):
                continue
            dte = _to_float(expiry.get("days-to-expiration") or expiry.get("days_to_expiration"), -1.0)
            strikes = expiry.get("strikes")
            if not isinstance(strikes, list):
                continue
            for strike_row in strikes:
                if isinstance(strike_row, (int, float, str)):
                    strike = _to_float(strike_row, 0.0)
                    if strike > 0.0:
                        rows.append({"strike": strike, "dte": dte, "side": 0.0, "open_interest": 1.0, "gamma": 0.0, "volume": 0.0})
                    continue
                if not isinstance(strike_row, dict):
                    continue
                strike = _to_float(strike_row.get("strike-price") or strike_row.get("strike_price") or strike_row.get("strike"), 0.0)
                if strike <= 0.0:
                    continue
                for side_name, side_token in (("call", 1.0), ("put", -1.0)):
                    node = strike_row.get(side_name)
                    if not isinstance(node, dict):
                        node = strike_row
                    open_interest = _to_float(
                        _casefold_get(
                            node,
                            f"{side_name}-open-interest",
                            f"{side_name}_open_interest",
                            "open-interest",
                            "open_interest",
                            "openInterest",
                        ),
                        0.0,
                    )
                    gamma = _to_float(
                        _casefold_get(
                            node,
                            f"{side_name}-gamma",
                            f"{side_name}_gamma",
                            "gamma",
                            "gamma-exposure",
                            "gamma_exposure",
                        ),
                        0.0,
                    )
                    volume = _to_float(
                        _casefold_get(
                            node,
                            f"{side_name}-volume",
                            f"{side_name}_volume",
                            "volume",
                            "trade-volume",
                            "trade_volume",
                        ),
                        0.0,
                    )
                    if open_interest <= 0.0 and gamma <= 0.0 and volume <= 0.0 and node is strike_row and side_name == "put":
                        continue
                    rows.append(
                        {
                            "strike": strike,
                            "dte": dte,
                            "side": side_token,
                            "open_interest": max(open_interest, 1.0 if node is strike_row else 0.0),
                            "gamma": max(gamma, 0.0),
                            "volume": max(volume, 0.0),
                        }
                    )
    return rows


def _normalize_percent_metric(raw: Any, *, scale: float = 100.0) -> float:
    value = _to_float(raw, 0.0)
    if value <= 0.0:
        return 0.0
    if value <= 1.0:
        return _clamp01(value)
    return _clamp01(value / max(scale, 1.0))


def _derive_shortability_features(market_metrics_payload: Any, instrument_payload: Any) -> dict[str, float]:
    borrow_fee_norm = _normalize_percent_metric(
        _casefold_get(
            instrument_payload,
            "borrow-fee-rate",
            "borrow_fee_rate",
            "stock-borrow-rate",
            "stock_borrow_rate",
            "hard-to-borrow-rate",
            "hard_to_borrow_rate",
        ),
        scale=60.0,
    )
    utilization_norm = _normalize_percent_metric(
        _casefold_get(
            market_metrics_payload,
            "utilization",
            "borrow-utilization",
            "borrow_utilization",
        ),
        scale=100.0,
    )
    days_to_cover_norm = _clamp01(
        _to_float(
            _casefold_get(
                market_metrics_payload,
                "days-to-cover",
                "days_to_cover",
                "short-interest-days-to-cover",
                "short_interest_days_to_cover",
            ),
            0.0,
        )
        / 7.0
    )
    lendability = str(_casefold_get(instrument_payload, "lendability") or "").strip().lower()
    is_hard_to_borrow = bool(_casefold_get(instrument_payload, "is-hard-to-borrow", "is_hard_to_borrow"))
    shares_available = _to_float(
        _casefold_get(
            instrument_payload,
            "shares-available-to-borrow",
            "shares_available_to_borrow",
            "borrow-availability",
            "borrow_availability",
        ),
        0.0,
    )
    availability_norm = 0.0
    if shares_available > 0.0:
        availability_norm = _clamp01(math.log10(max(shares_available, 1.0)) / 6.0)
    elif "easy to borrow" in lendability:
        availability_norm = 0.9
    elif "hard to borrow" in lendability or is_hard_to_borrow:
        availability_norm = 0.2
    elif lendability:
        availability_norm = 0.55
    return {
        "short_borrow_availability_norm": availability_norm,
        "short_borrow_fee_norm": borrow_fee_norm,
        "short_utilization_norm": utilization_norm,
        "short_days_to_cover_norm": days_to_cover_norm,
    }


def _derive_strike_wall_features(option_chain_payload: Any, *, last_price: float, expected_move: float) -> dict[str, float]:
    contracts = list(_iter_chain_contracts(option_chain_payload))
    if not contracts or last_price <= 0.0:
        return {
            "tasty_dealer_gamma_pressure_norm": 0.0,
            "tasty_call_wall_proximity_norm": 0.0,
            "tasty_put_wall_proximity_norm": 0.0,
            "tasty_max_pain_proximity_norm": 0.0,
            "tasty_pin_risk_norm": 0.0,
        }

    call_weights: dict[float, float] = {}
    put_weights: dict[float, float] = {}
    candidate_strikes: list[float] = []
    near_term_oi = 0.0
    total_oi = 0.0
    gamma_near = 0.0
    gamma_total = 0.0
    for row in contracts:
        strike = float(row.get("strike", 0.0) or 0.0)
        if strike <= 0.0:
            continue
        candidate_strikes.append(strike)
        weight = max(float(row.get("open_interest", 0.0) or 0.0), 0.0) + (0.25 * max(float(row.get("volume", 0.0) or 0.0), 0.0))
        if weight <= 0.0:
            weight = 1.0
        dte = float(row.get("dte", -1.0) or -1.0)
        is_near = 0.0 <= dte <= 7.0
        total_oi += weight
        if is_near:
            near_term_oi += weight
            gamma_near += abs(float(row.get("gamma", 0.0) or 0.0))
        gamma_total += abs(float(row.get("gamma", 0.0) or 0.0))
        if float(row.get("side", 0.0) or 0.0) > 0.0:
            call_weights[strike] = call_weights.get(strike, 0.0) + weight
        else:
            put_weights[strike] = put_weights.get(strike, 0.0) + weight

    if not candidate_strikes:
        return {
            "tasty_dealer_gamma_pressure_norm": 0.0,
            "tasty_call_wall_proximity_norm": 0.0,
            "tasty_put_wall_proximity_norm": 0.0,
            "tasty_max_pain_proximity_norm": 0.0,
            "tasty_pin_risk_norm": 0.0,
        }

    call_wall = max(call_weights.items(), key=lambda item: item[1])[0] if call_weights else last_price
    put_wall = max(put_weights.items(), key=lambda item: item[1])[0] if put_weights else last_price
    scale = max(expected_move, last_price * 0.10, 1.0)
    call_wall_proximity = _clamp01(1.0 - (abs(call_wall - last_price) / scale))
    put_wall_proximity = _clamp01(1.0 - (abs(last_price - put_wall) / scale))

    candidate_strikes = sorted(set(candidate_strikes))
    best_max_pain = candidate_strikes[0]
    best_cost = None
    for candidate in candidate_strikes:
        payout = 0.0
        for strike, weight in call_weights.items():
            payout += max(candidate - strike, 0.0) * weight
        for strike, weight in put_weights.items():
            payout += max(strike - candidate, 0.0) * weight
        if best_cost is None or payout < best_cost:
            best_cost = payout
            best_max_pain = candidate
    max_pain_proximity = _clamp01(1.0 - (abs(best_max_pain - last_price) / scale))
    near_term_share = near_term_oi / max(total_oi, 1.0)
    gamma_pressure = _clamp01((0.55 * near_term_share) + (0.45 * _clamp01(gamma_near / max(gamma_total, 1.0))))
    pin_risk = _clamp01((0.65 * near_term_share) + (0.35 * max(max_pain_proximity, call_wall_proximity, put_wall_proximity)))
    return {
        "tasty_dealer_gamma_pressure_norm": gamma_pressure,
        "tasty_call_wall_proximity_norm": call_wall_proximity,
        "tasty_put_wall_proximity_norm": put_wall_proximity,
        "tasty_max_pain_proximity_norm": max_pain_proximity,
        "tasty_pin_risk_norm": pin_risk,
    }


def _derive_vol_surface_features(option_chain_payload: Any, *, last_price: float) -> dict[str, float]:
    contracts = list(_iter_chain_contracts(option_chain_payload))
    if not contracts or last_price <= 0.0:
        return {
            "options_iv_skew_norm": 0.5,
            "options_iv_term_structure_norm": 0.5,
            "options_gamma_expiry_skew_norm": 0.5,
            "options_vol_regime_norm": 0.0,
        }

    front_weight = 0.0
    back_weight = 0.0
    front_gamma = 0.0
    back_gamma = 0.0
    call_atm_weight = 0.0
    put_atm_weight = 0.0
    for row in contracts:
        strike = max(float(row.get("strike", 0.0) or 0.0), 0.0)
        if strike <= 0.0:
            continue
        dte = float(row.get("dte", -1.0) or -1.0)
        side = float(row.get("side", 0.0) or 0.0)
        gamma = abs(float(row.get("gamma", 0.0) or 0.0))
        weight = max(float(row.get("open_interest", 0.0) or 0.0), 0.0) + (0.25 * max(float(row.get("volume", 0.0) or 0.0), 0.0))
        if weight <= 0.0:
            weight = 1.0
        distance_ratio = abs(strike - last_price) / max(last_price, 1e-6)
        if 0.0 <= dte <= 7.0:
            front_weight += weight
            front_gamma += gamma
        elif 21.0 <= dte <= 60.0:
            back_weight += weight
            back_gamma += gamma
        if distance_ratio <= 0.08:
            if side > 0.0:
                call_atm_weight += weight
            elif side < 0.0:
                put_atm_weight += weight

    term_balance = (back_weight - front_weight) / max(back_weight + front_weight, 1.0)
    skew_balance = (put_atm_weight - call_atm_weight) / max(put_atm_weight + call_atm_weight, 1.0)
    gamma_expiry_balance = (front_gamma - back_gamma) / max(front_gamma + back_gamma, 1.0)
    vol_regime = _clamp01(
        (0.40 * min((front_weight + back_weight) / 4000.0, 1.0))
        + (0.35 * abs(term_balance))
        + (0.25 * abs(gamma_expiry_balance))
    )
    return {
        "options_iv_skew_norm": _signed_centered_norm(skew_balance, 1.0),
        "options_iv_term_structure_norm": _signed_centered_norm(term_balance, 1.0),
        "options_gamma_expiry_skew_norm": _signed_centered_norm(gamma_expiry_balance, 1.0),
        "options_vol_regime_norm": vol_regime,
    }


def _derive_symbol_features(
    *,
    symbol: str,
    option_chain_payload: Any,
    market_metrics_payload: Any,
    instrument_payload: Any = None,
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
    out.update(_derive_shortability_features(market_metrics_payload, instrument_payload))

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
    out.update(
        _derive_strike_wall_features(
            option_chain_payload,
            last_price=max(last_price, 0.0),
            expected_move=max(expected_move, 0.0),
        )
    )
    out.update(
        _derive_vol_surface_features(
            option_chain_payload,
            last_price=max(last_price, 0.0),
        )
    )
    out["options_surface_change_norm"] = _clamp01(
        (0.40 * abs(float(out.get("options_iv_term_structure_norm", 0.5) or 0.5) - 0.5) * 2.0)
        + (0.35 * abs(float(out.get("options_iv_skew_norm", 0.5) or 0.5) - 0.5) * 2.0)
        + (0.25 * float(out.get("options_vol_regime_norm", 0.0) or 0.0))
    )
    out["options_strike_expiry_concentration_change_norm"] = _clamp01(
        (0.35 * chain["contract_density_norm"])
        + (0.25 * chain["zero_dte_presence_norm"])
        + (0.20 * float(out.get("tasty_dealer_gamma_pressure_norm", 0.0) or 0.0))
        + (0.20 * float(out.get("tasty_pin_risk_norm", 0.0) or 0.0))
    )
    out["options_gamma_flip_distance_norm"] = _clamp01(
        1.0
        - max(
            float(out.get("tasty_call_wall_proximity_norm", 0.0) or 0.0),
            float(out.get("tasty_put_wall_proximity_norm", 0.0) or 0.0),
            float(out.get("tasty_max_pain_proximity_norm", 0.0) or 0.0),
        )
    )
    out["options_earnings_setup_norm"] = _clamp01(
        (0.30 * expected_move_norm)
        + (0.25 * float(out.get("options_vol_regime_norm", 0.0) or 0.0))
        + (0.20 * float(out.get("options_surface_change_norm", 0.0) or 0.0))
        + (0.15 * chain["near_term_presence_norm"])
        + (0.10 * out["tasty_watchlist_presence_norm"])
    )
    out["options_iv_crush_risk_norm"] = _clamp01(
        (0.32 * expected_move_norm)
        + (0.26 * float(out.get("options_vol_regime_norm", 0.0) or 0.0))
        + (0.22 * float(out.get("options_surface_change_norm", 0.0) or 0.0))
        + (0.20 * chain["near_term_presence_norm"])
    )
    out["options_assignment_risk_norm"] = _clamp01(
        (0.40 * float(out.get("tasty_pin_risk_norm", 0.0) or 0.0))
        + (0.25 * chain["zero_dte_presence_norm"])
        + (0.20 * chain["near_term_presence_norm"])
        + (0.15 * float(out.get("options_strike_expiry_concentration_change_norm", 0.0) or 0.0))
    )
    out["options_zero_dte_regime_norm"] = _clamp01(
        (0.55 * chain["zero_dte_presence_norm"])
        + (0.25 * chain["contract_density_norm"])
        + (0.20 * float(out.get("tasty_pin_risk_norm", 0.0) or 0.0))
    )
    out["options_vol_of_vol_change_norm"] = _clamp01(
        (0.45 * float(out.get("options_surface_change_norm", 0.0) or 0.0))
        + (0.30 * abs(float(out.get("options_iv_term_structure_norm", 0.5) or 0.5) - 0.5) * 2.0)
        + (0.25 * abs(float(out.get("options_iv_skew_norm", 0.5) or 0.5) - 0.5) * 2.0)
    )
    out["options_spread_execution_risk_norm"] = _clamp01(
        (0.35 * max(1.0 - out["tasty_liquidity_rating_norm"], 0.0))
        + (0.25 * chain["zero_dte_presence_norm"])
        + (0.20 * float(out.get("tasty_pin_risk_norm", 0.0) or 0.0))
        + (0.20 * expected_move_norm)
    )
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


def _establish_session(
    *,
    sandbox: bool,
    user_agent: str,
    login: str,
    password: str,
    timeout: float,
) -> tuple[str | None, str, bool, str | None, list[dict[str, Any]]]:
    candidates = [(SANDBOX_BASE_URL, True), (LIVE_BASE_URL, False)] if sandbox else [(LIVE_BASE_URL, False)]
    attempts: list[dict[str, Any]] = []
    last_error = "session_not_attempted"
    selected_url = candidates[0][0]
    selected_sandbox = candidates[0][1]
    for idx, (base_url, sandbox_mode) in enumerate(candidates):
        token, err = _post_session(
            base_url,
            user_agent=user_agent,
            login=login,
            password=password,
            timeout=timeout,
        )
        attempt = {
            "base_url": base_url,
            "sandbox": sandbox_mode,
            "ok": token is not None,
            "error": err,
        }
        attempts.append(attempt)
        selected_url = base_url
        selected_sandbox = sandbox_mode
        if token is not None:
            return token, base_url, sandbox_mode, None, attempts
        last_error = err or "session_failed"
        if idx == 0 and sandbox and _is_unauthorized_error(last_error):
            continue
        break
    return None, selected_url, selected_sandbox, last_error, attempts


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
    requested_sandbox = bool(sandbox)
    base_url = SANDBOX_BASE_URL if requested_sandbox else LIVE_BASE_URL
    session_contract = _source_contract("tastytrade_session")
    watchlist_contract = _source_contract("tastytrade_public_watchlists")
    option_chain_contract = _source_contract("tastytrade_option_chain")
    instrument_contract = _source_contract("tastytrade_equity_instrument")
    market_metrics_contract = _source_contract("tastytrade_market_metrics")
    schwab_alignment_contract = _source_contract("schwab_alignment_reference")
    status: dict[str, Any] = {
        "timestamp_utc": now.isoformat(),
        "provider": "tastytrade_context",
        "base_url": base_url,
        "sandbox": requested_sandbox,
        "requested_sandbox": requested_sandbox,
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
        "sources": {
            "session": {
                "ok": False,
                "source_confidence_norm": float(session_contract["source_confidence_norm"]),
                "schema_confidence_norm": float(session_contract["schema_confidence_norm"]),
                "freshness_norm": 0.0,
                "fetched_utc": "",
            },
            "watchlists": {
                "ok": False,
                "source_confidence_norm": float(watchlist_contract["source_confidence_norm"]),
                "schema_confidence_norm": float(watchlist_contract["schema_confidence_norm"]),
                "freshness_norm": 0.0,
                "fetched_utc": "",
            },
            "schwab_alignment": {
                "ok": False,
                "source_confidence_norm": float(schwab_alignment_contract["source_confidence_norm"]),
                "schema_confidence_norm": float(schwab_alignment_contract["schema_confidence_norm"]),
                "freshness_norm": 0.0,
                "fetched_utc": "",
            },
        },
    }

    if _looks_placeholder_secret(login) or _looks_placeholder_secret(password):
        status["auth_issue"] = "credentials_missing"
        status["operator_action_required"] = True
        status["errors"].append("credentials_missing")
        payload["sources"]["session"] = {
            "ok": False,
            "error": "credentials_missing",
            "operator_action_required": True,
            "recommended_action": "set_live_tastytrade_credentials",
            "source_confidence_norm": float(session_contract["source_confidence_norm"]),
            "schema_confidence_norm": float(session_contract["schema_confidence_norm"]),
            "freshness_norm": 0.0,
            "fetched_utc": "",
        }
        return payload, status

    session_token, base_url, effective_sandbox, session_err, session_attempts = _establish_session(
        sandbox=requested_sandbox,
        user_agent=user_agent,
        login=login,
        password=password,
        timeout=timeout_seconds,
    )
    status["base_url"] = base_url
    status["sandbox"] = bool(effective_sandbox)
    payload["sources"]["session"] = {
        "ok": session_token is not None,
        "error": session_err,
        "attempts": session_attempts,
        "fallback_used": bool(requested_sandbox and not effective_sandbox),
        "source_confidence_norm": float(session_contract["source_confidence_norm"]),
        "schema_confidence_norm": float(session_contract["schema_confidence_norm"]),
        "freshness_norm": 1.0 if session_token is not None else 0.0,
        "fetched_utc": now.isoformat() if session_token is not None else "",
    }
    if session_token is None:
        if _is_unauthorized_error(session_err):
            status["auth_issue"] = "sandbox_credentials_rejected" if bool(effective_sandbox) else "live_credentials_rejected"
            status["operator_action_required"] = True
            payload["sources"]["session"]["operator_action_required"] = True
            payload["sources"]["session"]["recommended_action"] = (
                "refresh_tastytrade_sandbox_credentials" if bool(effective_sandbox) else "refresh_tastytrade_live_credentials"
            )
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
        "source_confidence_norm": float(watchlist_contract["source_confidence_norm"]),
        "schema_confidence_norm": float(watchlist_contract["schema_confidence_norm"]),
        "freshness_norm": 1.0 if watchlists_payload is not None else 0.0,
        "fetched_utc": now.isoformat() if watchlists_payload is not None else "",
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
        "source_confidence_norm": float(schwab_alignment_contract["source_confidence_norm"]),
        "schema_confidence_norm": float(schwab_alignment_contract["schema_confidence_norm"]),
        "freshness_norm": 1.0 if schwab_history else 0.0,
        "fetched_utc": now.isoformat() if schwab_history else "",
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
                "source_confidence_norm": max(
                    float(option_chain_contract["source_confidence_norm"]),
                    float(market_metrics_contract["source_confidence_norm"]),
                    float(instrument_contract["source_confidence_norm"]),
                ),
                "schema_confidence_norm": max(
                    float(option_chain_contract["schema_confidence_norm"]),
                    float(market_metrics_contract["schema_confidence_norm"]),
                    float(instrument_contract["schema_confidence_norm"]),
                ),
                "freshness_norm": 0.0,
                "fetched_utc": "",
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
            sandbox=bool(effective_sandbox),
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
            "source_confidence_norm": max(
                float(option_chain_contract["source_confidence_norm"]) if option_chain_payload is not None else 0.0,
                float(market_metrics_contract["source_confidence_norm"]) if metrics_payload is not None else 0.0,
                float(instrument_contract["source_confidence_norm"]) if instrument_payload is not None else 0.0,
            ),
            "schema_confidence_norm": max(
                float(option_chain_contract["schema_confidence_norm"]) if option_chain_payload is not None else 0.0,
                float(market_metrics_contract["schema_confidence_norm"]) if metrics_payload is not None else 0.0,
                float(instrument_contract["schema_confidence_norm"]) if instrument_payload is not None else 0.0,
            ),
            "freshness_norm": 1.0 if (option_chain_payload is not None or metrics_payload is not None or instrument_payload is not None) else 0.0,
            "fetched_utc": now.isoformat(),
        }
        symbol_features[symbol] = attach_collection_confidence(
            symbol_features[symbol],
            source_confidence_norm=float(payload["sources"][symbol]["source_confidence_norm"]),
            schema_confidence_norm=float(payload["sources"][symbol]["schema_confidence_norm"]),
            freshness_norm=float(payload["sources"][symbol]["freshness_norm"]),
            fetched_utc=str(payload["sources"][symbol]["fetched_utc"]),
        )

    payload["derived"]["symbol_features"] = symbol_features
    payload["derived"]["global_features"] = {key: _mean_feature(symbol_features.values(), key) for key in FEATURE_KEYS}
    payload["sources"]["market_metrics"] = {
        "ok": not bool(market_metrics_capability.get("unsupported")),
        "error": market_metrics_capability.get("last_error") if market_metrics_capability.get("unsupported") else None,
        "source_confidence_norm": float(market_metrics_contract["source_confidence_norm"]),
        "schema_confidence_norm": float(market_metrics_contract["schema_confidence_norm"]),
        "freshness_norm": 1.0 if not bool(market_metrics_capability.get("unsupported")) else 0.0,
        "fetched_utc": now.isoformat() if not bool(market_metrics_capability.get("unsupported")) else "",
    }
    source_rows = [
        row
        for key, row in payload["sources"].items()
        if key in {"session", "watchlists", "schwab_alignment", "market_metrics"} and isinstance(row, dict) and bool(row.get("ok"))
    ]
    payload["collection_contract"] = {
        "provider": "tastytrade_context",
        "source_contracts": dict(SOURCE_CONTRACTS),
        "provider_confidence_norm": round(
            sum(float(row.get("source_confidence_norm", 0.0) or 0.0) for row in source_rows) / max(len(source_rows), 1),
            6,
        ),
    }
    status["sources"] = {
        key: dict(value)
        for key, value in payload["sources"].items()
        if key in {"session", "watchlists", "schwab_alignment", "market_metrics"}
    }
    status["source_contracts"] = dict(SOURCE_CONTRACTS)
    status["collection_contract"] = dict(payload["collection_contract"])
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
