import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


OPTIONS_FEATURE_KEYS = [
    "options_chain_available",
    "options_contract_count_norm",
    "options_iv_atm",
    "options_iv_atm_norm",
    "options_iv_mean",
    "options_iv_skew",
    "options_iv_skew_norm",
    "options_iv_term_structure",
    "options_iv_term_structure_norm",
    "options_open_interest_total",
    "options_open_interest_norm",
    "options_put_call_oi_ratio",
    "options_put_call_oi_ratio_norm",
    "options_delta_abs_mean_norm",
    "options_gamma_mean_norm",
    "options_theta_abs_mean_norm",
    "options_vega_mean_norm",
    "options_spread_bps_mean",
    "options_spread_bps_norm",
    "options_spread_skew_norm",
    "options_negative_bias_norm",
    "options_roll_yield_norm",
    "options_vwap_bias_norm",
    "options_near_expiry_days",
    "options_far_expiry_days",
    "options_near_expiry_norm",
    "options_far_expiry_norm",
    "options_atm_strike",
    "options_strike_dispersion_norm",
    "options_vol_expectation_norm",
    "options_gamma_exposure_norm",
    "options_call_wall_distance_norm",
    "options_put_wall_distance_norm",
    "options_oi_concentration_norm",
    "options_unusual_flow_norm",
]

FUTURES_FEATURE_KEYS = [
    "futures_spread_bps",
    "futures_spread_bps_norm",
    "futures_bid_size",
    "futures_ask_size",
    "futures_order_book_imbalance",
    "futures_order_book_imbalance_norm",
    "futures_depth_ratio",
    "futures_depth_ratio_norm",
    "futures_open_interest",
    "futures_open_interest_norm",
    "futures_funding_rate",
    "futures_funding_rate_norm",
    "futures_basis_bps",
    "futures_basis_bps_norm",
    "futures_vwap_bias_norm",
    "futures_term_structure_norm",
    "futures_negative_bias_norm",
    "futures_roll_yield_norm",
    "futures_expiry_days",
    "futures_expiry_norm",
]

CALENDAR_FEATURE_KEYS = [
    "calendar_feed_available",
    "calendar_events_24h_norm",
    "calendar_high_impact_24h_norm",
    "calendar_earnings_7d_norm",
    "calendar_event_proximity_norm",
    "calendar_next_event_norm",
    "calendar_macro_event_norm",
    "calendar_macro_surprise_norm",
    "calendar_macro_abs_surprise_norm",
    "calendar_macro_revision_norm",
    "calendar_fomc_event_norm",
    "calendar_cpi_event_norm",
    "calendar_labor_event_norm",
    "calendar_treasury_auction_norm",
    "calendar_options_expiry_week_norm",
    "calendar_dividend_events_30d_norm",
    "calendar_dividend_exdate_proximity_norm",
    "calendar_dividend_payout_proximity_norm",
    "calendar_dividend_recent_exdate_norm",
    "calendar_dividend_quality_signal_norm",
]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, str):
            text = value.strip().replace(",", "")
            if not text or text.lower() in {"n/a", "na", "none", "null", "--", "-"}:
                return default
            mult = 1.0
            tail = text[-1:].lower()
            if tail in {"k", "m", "b"}:
                mult = {"k": 1.0e3, "m": 1.0e6, "b": 1.0e9}[tail]
                text = text[:-1]
            if text.endswith("%"):
                text = text[:-1]
            return float(text) * mult
        return float(value)
    except Exception:
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _signed_centered_norm(value: float, scale: float) -> float:
    return _clamp01(0.5 + (float(value) / max(float(scale), 1e-8)))


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / max(len(values), 1))


def _safe_stdev(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = _safe_mean(values)
    var = sum((x - mu) ** 2 for x in values) / max(n - 1, 1)
    return float(math.sqrt(max(var, 0.0)))


def _parse_epoch(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        v = float(raw)
        if v > 1e12:
            v /= 1000.0
        if v > 1e9:
            return v
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.isdigit():
        v = float(s)
        if v > 1e12:
            v /= 1000.0
        if v > 1e9:
            return v
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).timestamp()
    except Exception:
        return None


def _days_to_expiry(raw: Any, *, now_ts: float) -> Optional[float]:
    dte = _to_float(raw, -1.0)
    if dte >= 0.0:
        return dte
    epoch = _parse_epoch(raw)
    if epoch is None:
        return None
    return max((epoch - now_ts) / 86400.0, 0.0)


def _iter_dict_nodes(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            for child in _iter_dict_nodes(v):
                yield child
    elif isinstance(node, list):
        for item in node:
            for child in _iter_dict_nodes(item):
                yield child


def _first_numeric(node: Any, keys: Iterable[str]) -> Optional[float]:
    key_set = {str(k).lower() for k in keys}
    for d in _iter_dict_nodes(node):
        for k, v in d.items():
            if str(k).lower() in key_set:
                try:
                    return float(v)
                except Exception:
                    continue
    return None


def default_options_features() -> Dict[str, float]:
    return {k: 0.0 for k in OPTIONS_FEATURE_KEYS}


def default_futures_features() -> Dict[str, float]:
    return {k: 0.0 for k in FUTURES_FEATURE_KEYS}


def default_calendar_features() -> Dict[str, float]:
    return {k: 0.0 for k in CALENDAR_FEATURE_KEYS}


def _option_side(row: Dict[str, Any], default: str = "") -> str:
    raw = str(
        row.get("putCall")
        or row.get("optionType")
        or row.get("type")
        or row.get("right")
        or default
    ).upper()
    if raw in {"C", "CALL"} or "CALL" in raw:
        return "CALL"
    if raw in {"P", "PUT"} or "PUT" in raw:
        return "PUT"
    return ""


def _option_row_strike(row: Dict[str, Any], strike_hint: Optional[float]) -> float:
    strike = _to_float(
        row.get("strikePrice")
        if row.get("strikePrice") is not None
        else row.get("strike"),
        0.0,
    )
    if strike <= 0.0 and strike_hint is not None:
        strike = max(float(strike_hint), 0.0)
    return strike


def _extract_option_rows(payload: Any, *, now_ts: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if isinstance(payload, dict):
        for side_key, side_name in (("callExpDateMap", "CALL"), ("putExpDateMap", "PUT")):
            exp_map = payload.get(side_key)
            if not isinstance(exp_map, dict):
                continue
            for exp_key, by_strike in exp_map.items():
                dte_hint: Optional[float] = None
                if isinstance(exp_key, str):
                    parts = exp_key.split(":")
                    if len(parts) > 1:
                        dte_hint = _to_float(parts[-1], -1.0)
                        if dte_hint < 0.0:
                            dte_hint = None
                if not isinstance(by_strike, dict):
                    continue
                for strike_key, contracts in by_strike.items():
                    strike_hint = _to_float(strike_key, 0.0)
                    if strike_hint <= 0.0:
                        strike_hint = None
                    if not isinstance(contracts, list):
                        continue
                    for raw in contracts:
                        if not isinstance(raw, dict):
                            continue
                        row = dict(raw)
                        if "putCall" not in row:
                            row["putCall"] = side_name
                        if strike_hint is not None and row.get("strikePrice") is None:
                            row["strikePrice"] = strike_hint
                        if dte_hint is not None and row.get("daysToExpiration") is None:
                            row["daysToExpiration"] = dte_hint
                        rows.append(row)

    if isinstance(payload, dict):
        for key in ("options", "contracts", "items", "data", "results"):
            val = payload.get(key)
            if not isinstance(val, list):
                continue
            for raw in val:
                if isinstance(raw, dict):
                    rows.append(dict(raw))

    for node in _iter_dict_nodes(payload):
        if not any(k in node for k in ("impliedVolatility", "delta", "openInterest", "strikePrice")):
            continue
        rows.append(dict(node))

    uniq: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        side = _option_side(row)
        strike = _option_row_strike(row, None)
        expiry_days = _days_to_expiry(
            row.get("daysToExpiration")
            if row.get("daysToExpiration") is not None
            else row.get("expirationDate"),
            now_ts=now_ts,
        )
        key = "|".join(
            [
                str(row.get("symbol") or row.get("description") or ""),
                side,
                f"{strike:.4f}",
                f"{(expiry_days if expiry_days is not None else -1.0):.4f}",
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
    return uniq


def summarize_option_chain(
    payload: Any,
    *,
    symbol: str = "",
    underlying_price: float = 0.0,
    now_ts: Optional[float] = None,
    max_contracts: int = 1200,
) -> Dict[str, float]:
    out = default_options_features()
    ts_now = float(now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp())
    rows = _extract_option_rows(payload, now_ts=ts_now)
    if not rows:
        return out

    rows = rows[: max(max_contracts, 50)]
    under = float(max(underlying_price, 0.0))
    if under <= 0.0:
        u = _first_numeric(payload, ("underlyingPrice", "underlying", "mark", "last"))
        if u is not None:
            under = max(float(u), 0.0)

    iv_all: List[float] = []
    iv_call: List[float] = []
    iv_put: List[float] = []
    dte_all: List[float] = []
    dte_iv_pairs: List[Tuple[float, float]] = []
    moneyness_all: List[float] = []
    spread_bps_all: List[float] = []
    spread_bps_call: List[float] = []
    spread_bps_put: List[float] = []
    mark_mid_bias: List[float] = []

    call_oi = 0.0
    put_oi = 0.0
    call_volume = 0.0
    put_volume = 0.0
    delta_abs_vals: List[float] = []
    gamma_vals: List[float] = []
    theta_abs_vals: List[float] = []
    vega_vals: List[float] = []
    unusual_flow_ratios: List[float] = []
    strike_open_interest: Dict[float, float] = {}
    call_wall_oi: Dict[float, float] = {}
    put_wall_oi: Dict[float, float] = {}
    gamma_exposure = 0.0

    atm_strike = 0.0
    atm_iv = 0.0
    atm_dist = 1e9

    for row in rows:
        side = _option_side(row)
        strike = _option_row_strike(row, None)
        if strike > 0.0 and under > 0.0:
            rel_m = (strike - under) / max(under, 1e-8)
            moneyness_all.append(rel_m)
            dist = abs(strike - under)
            if dist < atm_dist:
                atm_dist = dist
                atm_strike = strike

        iv = _to_float(
            row.get("impliedVolatility")
            if row.get("impliedVolatility") is not None
            else row.get("volatility"),
            0.0,
        )
        if iv > 2.5:
            iv /= 100.0
        if 0.0 < iv < 5.0:
            iv_all.append(iv)
            if side == "CALL":
                iv_call.append(iv)
            elif side == "PUT":
                iv_put.append(iv)
            if strike > 0.0 and under > 0.0:
                dist = abs(strike - under)
                if dist <= max(under * 0.01, 0.25):
                    if atm_iv <= 0.0:
                        atm_iv = iv
                    else:
                        atm_iv = 0.5 * atm_iv + 0.5 * iv

        dte = _days_to_expiry(
            row.get("daysToExpiration")
            if row.get("daysToExpiration") is not None
            else row.get("expirationDate"),
            now_ts=ts_now,
        )
        if dte is not None:
            dte_all.append(dte)
            if iv > 0.0:
                dte_iv_pairs.append((dte, iv))

        oi = max(
            _to_float(
                row.get("openInterest")
                if row.get("openInterest") is not None
                else row.get("open_interest"),
                0.0,
            ),
            0.0,
        )
        if side == "CALL":
            call_oi += oi
        elif side == "PUT":
            put_oi += oi
        if strike > 0.0 and oi > 0.0:
            strike_open_interest[strike] = strike_open_interest.get(strike, 0.0) + oi
            if side == "CALL":
                call_wall_oi[strike] = call_wall_oi.get(strike, 0.0) + oi
            elif side == "PUT":
                put_wall_oi[strike] = put_wall_oi.get(strike, 0.0) + oi

        volume = max(
            _to_float(
                row.get("totalVolume")
                if row.get("totalVolume") is not None
                else row.get("volume"),
                0.0,
            ),
            0.0,
        )
        if side == "CALL":
            call_volume += volume
        elif side == "PUT":
            put_volume += volume
        if volume > 0.0:
            unusual_flow_ratios.append(min(volume / max(oi, 1.0), 4.0))

        d = row.get("delta")
        if d is not None:
            d_val = abs(_to_float(d, 0.0))
            if d_val <= 2.0:
                delta_abs_vals.append(min(d_val, 1.0))
        g = row.get("gamma")
        if g is not None:
            g_val = abs(_to_float(g, 0.0))
            if g_val <= 5.0:
                gamma_vals.append(g_val)
                gamma_exposure += g_val * max(oi, 0.0)
        t = row.get("theta")
        if t is not None:
            t_val = abs(_to_float(t, 0.0))
            if t_val <= 20.0:
                theta_abs_vals.append(t_val)
        v = row.get("vega")
        if v is not None:
            v_val = abs(_to_float(v, 0.0))
            if v_val <= 20.0:
                vega_vals.append(v_val)

        bid = _to_float(row.get("bid"), 0.0)
        ask = _to_float(row.get("ask"), 0.0)
        if bid > 0.0 and ask > bid:
            mid = 0.5 * (bid + ask)
            spread_bps = ((ask - bid) / max(mid, 1e-8)) * 10000.0
            spread_bps_all.append(spread_bps)
            if side == "CALL":
                spread_bps_call.append(spread_bps)
            elif side == "PUT":
                spread_bps_put.append(spread_bps)
            mark = _to_float(
                row.get("mark")
                if row.get("mark") is not None
                else row.get("last"),
                mid,
            )
            mark_mid_bias.append((mark - mid) / max(mid, 1e-8))

    iv_mean = _safe_mean(iv_all)
    call_iv_mean = _safe_mean(iv_call)
    put_iv_mean = _safe_mean(iv_put)
    iv_skew = put_iv_mean - call_iv_mean

    near_ivs = [iv for dte, iv in dte_iv_pairs if dte <= 14.0]
    far_ivs = [iv for dte, iv in dte_iv_pairs if dte >= 30.0]
    near_iv = _safe_mean(near_ivs)
    far_iv = _safe_mean(far_ivs)
    if near_iv <= 0.0:
        near_iv = iv_mean
    if far_iv <= 0.0:
        far_iv = iv_mean
    iv_term = far_iv - near_iv

    if atm_iv <= 0.0:
        atm_iv = iv_mean

    put_call_ratio = (put_oi / max(call_oi, 1e-6)) if call_oi > 0.0 else 0.0
    spread_bps_mean = _safe_mean(spread_bps_all)
    spread_skew = _safe_mean(spread_bps_put) - _safe_mean(spread_bps_call)

    near_exp = min(dte_all) if dte_all else 0.0
    far_exp = max(dte_all) if dte_all else 0.0
    strike_dispersion = _safe_stdev(moneyness_all)
    total_oi = max(call_oi + put_oi, 0.0)
    oi_concentration = (max(strike_open_interest.values()) / total_oi) if strike_open_interest and total_oi > 0.0 else 0.0

    call_wall_distance = 0.0
    put_wall_distance = 0.0
    if under > 0.0 and call_wall_oi:
        call_wall_strike = max(call_wall_oi.items(), key=lambda kv: kv[1])[0]
        call_wall_distance = abs(call_wall_strike - under) / max(under, 1e-8)
    if under > 0.0 and put_wall_oi:
        put_wall_strike = max(put_wall_oi.items(), key=lambda kv: kv[1])[0]
        put_wall_distance = abs(put_wall_strike - under) / max(under, 1e-8)

    negative_bias = _clamp01(
        (0.55 * _clamp01((put_call_ratio - 1.0) / 1.5))
        + (0.25 * _clamp01(iv_skew / 0.15))
        + (0.20 * _clamp01(-_safe_mean(mark_mid_bias) * 6.0))
    )

    roll_yield = _clamp01((near_iv - far_iv + 0.25) / 0.50)
    vwap_bias = _clamp01(abs(_safe_mean(mark_mid_bias)) * 12.0)
    vol_expect = _clamp01((0.65 * _clamp01(atm_iv / 1.20)) + (0.35 * _clamp01((near_iv + max(iv_term, 0.0)) / 1.20)))

    out.update(
        {
            "options_chain_available": 1.0,
            "options_contract_count_norm": _clamp01(len(rows) / 500.0),
            "options_iv_atm": float(max(atm_iv, 0.0)),
            "options_iv_atm_norm": _clamp01(max(atm_iv, 0.0) / 1.20),
            "options_iv_mean": float(max(iv_mean, 0.0)),
            "options_iv_skew": float(iv_skew),
            "options_iv_skew_norm": _clamp01((iv_skew + 0.25) / 0.50),
            "options_iv_term_structure": float(iv_term),
            "options_iv_term_structure_norm": _clamp01((iv_term + 0.25) / 0.50),
            "options_open_interest_total": float(max(call_oi + put_oi, 0.0)),
            "options_open_interest_norm": _clamp01(math.log1p(max(call_oi + put_oi, 0.0)) / 14.0),
            "options_put_call_oi_ratio": float(max(put_call_ratio, 0.0)),
            "options_put_call_oi_ratio_norm": _clamp01(min(put_call_ratio, 4.0) / 4.0),
            "options_delta_abs_mean_norm": _clamp01(_safe_mean(delta_abs_vals)),
            "options_gamma_mean_norm": _clamp01(_safe_mean(gamma_vals) / 0.25),
            "options_theta_abs_mean_norm": _clamp01(_safe_mean(theta_abs_vals) / 2.0),
            "options_vega_mean_norm": _clamp01(_safe_mean(vega_vals) / 2.0),
            "options_spread_bps_mean": float(max(spread_bps_mean, 0.0)),
            "options_spread_bps_norm": _clamp01(spread_bps_mean / 200.0),
            "options_spread_skew_norm": _clamp01((spread_skew + 80.0) / 160.0),
            "options_negative_bias_norm": negative_bias,
            "options_roll_yield_norm": roll_yield,
            "options_vwap_bias_norm": vwap_bias,
            "options_near_expiry_days": float(max(near_exp, 0.0)),
            "options_far_expiry_days": float(max(far_exp, 0.0)),
            "options_near_expiry_norm": _clamp01(max(near_exp, 0.0) / 45.0),
            "options_far_expiry_norm": _clamp01(max(far_exp, 0.0) / 180.0),
            "options_atm_strike": float(max(atm_strike, 0.0)),
            "options_strike_dispersion_norm": _clamp01(strike_dispersion / 0.35),
            "options_vol_expectation_norm": vol_expect,
            "options_gamma_exposure_norm": _clamp01(math.log1p(max(gamma_exposure, 0.0)) / 10.0),
            "options_call_wall_distance_norm": _clamp01(call_wall_distance / 0.20),
            "options_put_wall_distance_norm": _clamp01(put_wall_distance / 0.20),
            "options_oi_concentration_norm": _clamp01(oi_concentration),
            "options_unusual_flow_norm": _clamp01(_safe_mean(unusual_flow_ratios) / 4.0),
        }
    )
    return out


def summarize_order_book(
    payload: Any,
    *,
    last_price: float = 0.0,
    top_n: int = 5,
) -> Dict[str, float]:
    out = default_futures_features()
    if not isinstance(payload, dict):
        return out

    bids = payload.get("bids")
    asks = payload.get("asks")
    if not isinstance(bids, list) or not isinstance(asks, list):
        return out

    best_bid = 0.0
    best_ask = 0.0
    bid_size = 0.0
    ask_size = 0.0
    bid_notional = 0.0
    ask_notional = 0.0

    for i, row in enumerate(bids[: max(top_n, 1)]):
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        px = _to_float(row[0], 0.0)
        sz = _to_float(row[1], 0.0)
        if i == 0:
            best_bid = px
            bid_size = max(sz, 0.0)
        bid_notional += max(px, 0.0) * max(sz, 0.0)

    for i, row in enumerate(asks[: max(top_n, 1)]):
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        px = _to_float(row[0], 0.0)
        sz = _to_float(row[1], 0.0)
        if i == 0:
            best_ask = px
            ask_size = max(sz, 0.0)
        ask_notional += max(px, 0.0) * max(sz, 0.0)

    ref_price = max(float(last_price), best_bid, best_ask, 1e-8)
    spread_bps = 0.0
    if best_bid > 0.0 and best_ask > best_bid:
        spread_bps = ((best_ask - best_bid) / ref_price) * 10000.0

    total_notional = bid_notional + ask_notional
    imbalance = (bid_notional - ask_notional) / max(total_notional, 1e-8)
    depth_ratio = _clamp01(total_notional / max(ref_price * 500.0, 1e-8))

    out.update(
        {
            "futures_spread_bps": float(max(spread_bps, 0.0)),
            "futures_spread_bps_norm": _clamp01(spread_bps / 80.0),
            "futures_bid_size": float(max(bid_size, 0.0)),
            "futures_ask_size": float(max(ask_size, 0.0)),
            "futures_order_book_imbalance": float(max(min(imbalance, 1.0), -1.0)),
            "futures_order_book_imbalance_norm": _clamp01((imbalance + 1.0) * 0.5),
            "futures_depth_ratio": float(depth_ratio),
            "futures_depth_ratio_norm": float(depth_ratio),
        }
    )
    return out


def summarize_futures_quote_features(
    payload: Any,
    *,
    last_price: float = 0.0,
    now_ts: Optional[float] = None,
) -> Dict[str, float]:
    out = default_futures_features()
    ts_now = float(now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp())

    spread_bps = 0.0
    bid = _first_numeric(payload, ("bidPrice", "bid", "bestBid"))
    ask = _first_numeric(payload, ("askPrice", "ask", "bestAsk"))
    bid_size = _first_numeric(payload, ("bidSize", "bestBidSize", "bidQty"))
    ask_size = _first_numeric(payload, ("askSize", "bestAskSize", "askQty"))

    if bid is not None and ask is not None and ask > bid:
        ref = max(float(last_price), bid, ask, 1e-8)
        spread_bps = ((ask - bid) / ref) * 10000.0

    oi = _first_numeric(payload, ("openInterest", "open_interest", "openInterestQty", "open_interest_qty"))
    funding = _first_numeric(payload, ("fundingRate", "funding_rate", "nextFundingRate", "interestRate"))
    basis_mark = _first_numeric(payload, ("markPrice", "mark", "mark_price"))
    basis_index = _first_numeric(payload, ("indexPrice", "index", "index_price", "spotPrice", "spot_price"))
    vwap = _first_numeric(payload, ("vwap", "VWAP", "volumeWeightedAveragePrice"))

    basis_bps = 0.0
    if basis_mark is not None and basis_index is not None and basis_index > 0.0:
        basis_bps = ((basis_mark - basis_index) / basis_index) * 10000.0
    elif basis_mark is not None and last_price > 0.0:
        basis_bps = ((basis_mark - last_price) / max(last_price, 1e-8)) * 10000.0

    expiry_days = 0.0
    for key in ("daysToExpiration", "days_to_expiration", "daysToExpiry", "expiry"):
        if isinstance(payload, dict) and key in payload:
            dte = _days_to_expiry(payload.get(key), now_ts=ts_now)
            if dte is not None:
                expiry_days = max(float(dte), 0.0)
                break
    if expiry_days <= 0.0:
        exp_epoch = _first_numeric(payload, ("expirationDate", "expiryDate", "maturityDate"))
        if exp_epoch is not None:
            dte2 = _days_to_expiry(exp_epoch, now_ts=ts_now)
            if dte2 is not None:
                expiry_days = max(float(dte2), 0.0)

    imbalance = 0.0
    if bid_size is not None or ask_size is not None:
        bsz = max(float(bid_size or 0.0), 0.0)
        asz = max(float(ask_size or 0.0), 0.0)
        imbalance = (bsz - asz) / max(bsz + asz, 1e-8)
    depth_ratio = _clamp01((max(float(bid_size or 0.0), 0.0) + max(float(ask_size or 0.0), 0.0)) / 2000.0)
    vwap_bias = 0.0
    if vwap is not None and vwap > 0.0 and last_price > 0.0:
        vwap_bias = (last_price - vwap) / vwap

    funding_val = float(funding or 0.0)
    if abs(funding_val) > 0.5:
        funding_val /= 100.0

    term_structure = _clamp01((basis_bps + 300.0) / 600.0)
    negative_bias = _clamp01(
        (0.45 * _clamp01((-funding_val * 100.0 + 2.0) / 4.0))
        + (0.35 * _clamp01((-imbalance + 1.0) * 0.5))
        + (0.20 * _clamp01((-basis_bps + 250.0) / 500.0))
    )
    roll_yield = _clamp01((basis_bps / max(expiry_days, 1.0) + 20.0) / 40.0)

    out.update(
        {
            "futures_spread_bps": float(max(spread_bps, 0.0)),
            "futures_spread_bps_norm": _clamp01(spread_bps / 80.0),
            "futures_bid_size": float(max(float(bid_size or 0.0), 0.0)),
            "futures_ask_size": float(max(float(ask_size or 0.0), 0.0)),
            "futures_order_book_imbalance": float(max(min(imbalance, 1.0), -1.0)),
            "futures_order_book_imbalance_norm": _clamp01((imbalance + 1.0) * 0.5),
            "futures_depth_ratio": float(depth_ratio),
            "futures_depth_ratio_norm": float(depth_ratio),
            "futures_open_interest": float(max(float(oi or 0.0), 0.0)),
            "futures_open_interest_norm": _clamp01(math.log1p(max(float(oi or 0.0), 0.0)) / 14.0),
            "futures_funding_rate": float(funding_val),
            "futures_funding_rate_norm": _clamp01((math.tanh(funding_val * 500.0) + 1.0) * 0.5),
            "futures_basis_bps": float(basis_bps),
            "futures_basis_bps_norm": _clamp01((basis_bps + 300.0) / 600.0),
            "futures_vwap_bias_norm": _clamp01((vwap_bias + 0.05) / 0.10),
            "futures_term_structure_norm": term_structure,
            "futures_negative_bias_norm": negative_bias,
            "futures_roll_yield_norm": roll_yield,
            "futures_expiry_days": float(max(expiry_days, 0.0)),
            "futures_expiry_norm": _clamp01(max(expiry_days, 0.0) / 120.0),
        }
    )
    return out


def _row_get_ci(row: Dict[str, Any], *keys: str) -> Any:
    if not isinstance(row, dict):
        return None
    key_map = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key.lower() in key_map:
            return key_map[key.lower()]
    return None


def _event_text(row: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for key in (
        "title",
        "name",
        "event",
        "eventtype",
        "category",
        "description",
        "headline",
        "symbol",
        "ticker",
        "country",
        "importance",
        "impact",
        "type",
        "eventclass",
    ):
        val = _row_get_ci(row, key)
        if isinstance(val, str) and val.strip():
            chunks.append(val.strip())
    return " ".join(chunks).lower()


def summarize_calendar_payload(
    payload: Any,
    *,
    now_ts: Optional[float] = None,
    max_items: int = 500,
) -> Dict[str, float]:
    out = default_calendar_features()
    ts_now = float(now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp())

    rows: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        rows.extend([r for r in payload if isinstance(r, dict)])
    elif isinstance(payload, dict):
        for key, val in payload.items():
            lk = str(key).lower()
            if lk not in {"events", "items", "calendar", "data", "results", "scheduledevents", "economicevents", "earnings"}:
                continue
            if isinstance(val, list):
                rows.extend([r for r in val if isinstance(r, dict)])
    for node in _iter_dict_nodes(payload):
        key_set = {str(k).lower() for k in node.keys()}
        if key_set.intersection({"eventdate", "date", "timestamp", "startdate", "datetime", "time", "dateutc"}):
            rows.append(node)

    if not rows:
        return out

    rows = rows[: max(max_items, 50)]
    next_minutes = 1e9
    events_24h = 0
    high_impact_24h = 0
    earnings_7d = 0
    macro_24h = 0
    options_expiry_week = 0
    macro_surprise = 0.0
    macro_abs_surprise = 0.0
    macro_revision = 0.0
    macro_surprise_n = 0
    fomc_7d = 0
    cpi_7d = 0
    labor_7d = 0
    treasury_auction_7d = 0

    macro_tokens = (
        "fomc",
        "fed",
        "cpi",
        "pce",
        "pmi",
        "ism",
        "gdp",
        "payroll",
        "unemployment",
        "rates",
        "treasury",
    )
    cpi_tokens = ("cpi", "pce", "inflation")
    labor_tokens = ("payroll", "nonfarm", "unemployment", "jobless", "labor")
    treasury_auction_tokens = ("treasury auction", "10-year auction", "30-year auction", "2-year auction", "bond auction", "note auction")
    high_tokens = ("high", "critical", "red", "major")
    earnings_tokens = ("earnings", "guidance", "revenue")
    options_expiry_tokens = ("options expiration", "opex", "triple witching")
    dividend_tokens = ("dividend", "distribution", "cash dividend")
    dividend_exdate_tokens = ("ex-dividend", "ex dividend", "ex-date", "ex date", "exdividend")
    dividend_payout_tokens = ("payout", "pay date", "payment date", "payable date", "distribution date")
    dividend_quality_tokens = (
        "quality dividend",
        "dividend aristocrat",
        "dividend king",
        "dividend champion",
        "blue chip dividend",
    )

    dividend_events_30d = 0
    dividend_recent_exdate = 0
    dividend_quality_events = 0
    next_dividend_ex_minutes = 1e9
    next_dividend_payout_minutes = 1e9

    for row in rows:
        ts = None
        for key in ("eventdate", "startdate", "datetime", "date", "timestamp", "time", "dateutc"):
            ts = _parse_epoch(_row_get_ci(row, key))
            if ts is not None:
                break
        if ts is None:
            continue

        delta_s = ts - ts_now
        text = _event_text(row)
        impact_raw = _row_get_ci(row, "impact", "importance")
        impact = str(impact_raw or "").lower()
        impact_score = _to_float(impact_raw, 0.0)
        is_high = any(tok in impact for tok in high_tokens) or ("high impact" in text) or (impact_score >= 3.0)
        is_macro = any(tok in text for tok in macro_tokens)
        is_fomc = ("fomc" in text) or ("fed" in text) or ("rate decision" in text)
        is_cpi = any(tok in text for tok in cpi_tokens)
        is_labor = any(tok in text for tok in labor_tokens)
        is_treasury_auction = any(tok in text for tok in treasury_auction_tokens)
        is_earnings = any(tok in text for tok in earnings_tokens)
        is_opex = any(tok in text for tok in options_expiry_tokens)
        is_dividend = any(tok in text for tok in dividend_tokens)
        is_dividend_exdate = is_dividend and any(tok in text for tok in dividend_exdate_tokens)
        is_dividend_payout = is_dividend and any(tok in text for tok in dividend_payout_tokens)
        is_dividend_quality = is_dividend and any(tok in text for tok in dividend_quality_tokens)

        if delta_s >= 0:
            mins = delta_s / 60.0
            next_minutes = min(next_minutes, mins)

            if delta_s <= 24 * 3600:
                events_24h += 1
                if is_high:
                    high_impact_24h += 1
                if is_macro:
                    macro_24h += 1
            if delta_s <= 7 * 24 * 3600 and is_earnings:
                earnings_7d += 1
            if delta_s <= 7 * 24 * 3600 and is_opex:
                options_expiry_week += 1
            if delta_s <= 7 * 24 * 3600 and is_fomc:
                fomc_7d += 1
            if delta_s <= 7 * 24 * 3600 and is_cpi:
                cpi_7d += 1
            if delta_s <= 7 * 24 * 3600 and is_labor:
                labor_7d += 1
            if delta_s <= 7 * 24 * 3600 and is_treasury_auction:
                treasury_auction_7d += 1

            if delta_s <= 30 * 24 * 3600 and is_dividend:
                dividend_events_30d += 1
                if is_dividend_quality:
                    dividend_quality_events += 1
            if is_dividend_exdate:
                next_dividend_ex_minutes = min(next_dividend_ex_minutes, mins)
            if is_dividend_payout:
                next_dividend_payout_minutes = min(next_dividend_payout_minutes, mins)
        elif delta_s >= -(3 * 24 * 3600):
            if is_dividend_exdate:
                dividend_recent_exdate += 1

        if is_macro:
            actual = _to_float(_row_get_ci(row, "actual", "actualvalue", "last"), 0.0)
            forecast = _to_float(_row_get_ci(row, "forecast", "consensus", "estimate", "teforecast"), 0.0)
            previous = _to_float(_row_get_ci(row, "previous", "prior"), 0.0)
            revised = _to_float(_row_get_ci(row, "revised", "revision", "revisedfromprevious"), 0.0)

            baseline = forecast if abs(forecast) > 0.0 else previous
            if abs(actual) > 0.0 and abs(baseline) > 0.0:
                scale = max(abs(baseline), 1.0)
                surprise = max(min((actual - baseline) / scale, 3.0), -3.0)
                macro_surprise += surprise
                macro_abs_surprise += abs(surprise)
                macro_surprise_n += 1
            if abs(revised) > 0.0 and abs(previous) > 0.0:
                revision = max(min((revised - previous) / max(abs(previous), 1.0), 3.0), -3.0)
                macro_revision += revision

    if next_minutes >= 1e8:
        next_minutes = 0.0
    if next_dividend_ex_minutes >= 1e8:
        next_dividend_ex_minutes = 0.0
    if next_dividend_payout_minutes >= 1e8:
        next_dividend_payout_minutes = 0.0
    if macro_surprise_n <= 0:
        macro_surprise_n = 1

    out.update(
        {
            "calendar_feed_available": 1.0,
            "calendar_events_24h_norm": _clamp01(events_24h / 30.0),
            "calendar_high_impact_24h_norm": _clamp01(high_impact_24h / 12.0),
            "calendar_earnings_7d_norm": _clamp01(earnings_7d / 24.0),
            "calendar_event_proximity_norm": _clamp01(1.0 - (next_minutes / 240.0)) if next_minutes > 0 else 0.0,
            "calendar_next_event_norm": _clamp01(next_minutes / 1440.0) if next_minutes > 0 else 0.0,
            "calendar_macro_event_norm": _clamp01(macro_24h / 12.0),
            "calendar_macro_surprise_norm": _signed_centered_norm(macro_surprise / macro_surprise_n, 3.0),
            "calendar_macro_abs_surprise_norm": _clamp01((macro_abs_surprise / macro_surprise_n) / 3.0),
            "calendar_macro_revision_norm": _signed_centered_norm(macro_revision / macro_surprise_n, 3.0),
            "calendar_fomc_event_norm": _clamp01(fomc_7d / 3.0),
            "calendar_cpi_event_norm": _clamp01(cpi_7d / 4.0),
            "calendar_labor_event_norm": _clamp01(labor_7d / 4.0),
            "calendar_treasury_auction_norm": _clamp01(treasury_auction_7d / 5.0),
            "calendar_options_expiry_week_norm": _clamp01(options_expiry_week / 4.0),
            "calendar_dividend_events_30d_norm": _clamp01(dividend_events_30d / 20.0),
            "calendar_dividend_exdate_proximity_norm": _clamp01(1.0 - (next_dividend_ex_minutes / (7.0 * 1440.0))) if next_dividend_ex_minutes > 0 else 0.0,
            "calendar_dividend_payout_proximity_norm": _clamp01(1.0 - (next_dividend_payout_minutes / (14.0 * 1440.0))) if next_dividend_payout_minutes > 0 else 0.0,
            "calendar_dividend_recent_exdate_norm": _clamp01(dividend_recent_exdate / 3.0),
            "calendar_dividend_quality_signal_norm": _clamp01(dividend_quality_events / 8.0),
        }
    )
    return out



def parse_expiry_key_days(raw: str, *, now_ts: Optional[float] = None) -> Optional[float]:
    s = str(raw or "").strip()
    if not s:
        return None
    parts = s.split(":")
    if len(parts) > 1:
        d = _to_float(parts[-1], -1.0)
        if d >= 0.0:
            return d

    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    if not m:
        return None
    dt = _parse_epoch(m.group(1) + "T00:00:00+00:00")
    if dt is None:
        return None
    ts_now = float(now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp())
    return max((dt - ts_now) / 86400.0, 0.0)
