from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


NEWS_STRUCTURED_FEATURE_KEYS = [
    "news_source_quality_norm",
    "news_entity_relevance_norm",
    "news_topic_earnings_norm",
    "news_topic_guidance_norm",
    "news_topic_mna_norm",
    "news_topic_regulatory_norm",
    "news_novelty_norm",
    "news_duplicate_cluster_norm",
    "news_premarket_norm",
    "news_intraday_norm",
    "news_after_hours_norm",
]

BREADTH_FEATURE_KEYS = [
    "breadth_snapshot_available",
    "breadth_advance_decline_norm",
    "breadth_up_down_volume_norm",
    "breadth_new_high_low_norm",
    "breadth_sector_dispersion_norm",
    "breadth_thrust_norm",
    "breadth_risk_off_norm",
]

BOND_REFERENCE_FEATURE_KEYS = [
    "bond_yield_2y_norm",
    "bond_yield_5y_norm",
    "bond_yield_10y_norm",
    "bond_yield_30y_norm",
    "bond_curve_2s10s_norm",
    "bond_curve_5s30s_norm",
    "bond_real_yield_10y_norm",
    "bond_duration_years_norm",
    "bond_convexity_norm",
    "bond_nav_discount_norm",
    "bond_etf_flow_5d_norm",
    "bond_ytm_norm",
    "bond_auction_window_norm",
    "bond_auction_tail_norm",
]

CREDIT_CONTEXT_FEATURE_KEYS = [
    "bond_credit_spread_level_norm",
    "bond_credit_spread_change_norm",
    "bond_hy_ig_flow_norm",
    "bond_nav_stress_norm",
]

DATA_QUALITY_FEATURE_KEYS = [
    "data_quality_quote_agreement_norm",
    "data_quality_quote_deviation_norm",
    "data_quality_stale_streak_norm",
    "data_quality_fail_streak_norm",
    "data_quality_quarantine_hits_norm",
    "data_quality_missing_feature_ratio_norm",
    "data_quality_bid_ask_imbalance_norm",
    "data_quality_market_data_latency_norm",
]

EXECUTION_LAG_FEATURE_KEYS = [
    "lag_slippage_bps",
    "lag_latency_ms",
    "lag_impact_bps",
    "lag_fee_bps",
    "lag_expected_fill_delta_bps",
    "lag_adjusted_return_1m",
    "lag_trade_action_norm",
]

_ET_ZONE = ZoneInfo("America/New_York") if ZoneInfo is not None else None

_NEWS_SOURCE_QUALITY = {
    "reuters": 1.0,
    "bloomberg": 1.0,
    "associated press": 0.98,
    "ap": 0.96,
    "wsj": 0.94,
    "wall street journal": 0.94,
    "financial times": 0.93,
    "dow jones": 0.92,
    "benzinga": 0.82,
    "marketwatch": 0.80,
    "seeking alpha": 0.74,
    "yahoo finance": 0.70,
    "business wire": 0.69,
    "globenewswire": 0.66,
    "accesswire": 0.64,
    "pr newswire": 0.62,
}

_NEWS_TOPIC_TOKENS = {
    "news_topic_earnings_norm": ("earnings", "eps", "revenue", "results", "beat", "miss"),
    "news_topic_guidance_norm": ("guidance", "outlook", "raises", "cuts", "forecast"),
    "news_topic_mna_norm": ("merger", "acquisition", "buyout", "takeover", "deal"),
    "news_topic_regulatory_norm": ("sec", "investigation", "lawsuit", "fda", "recall", "probe", "antitrust"),
}

_BOND_DURATION_ROLE_DEFAULTS = {
    "long_duration": 15.0,
    "short_duration": 2.0,
    "inflation": 7.0,
    "credit": 6.0,
}

_BOND_CONVEXITY_ROLE_DEFAULTS = {
    "long_duration": 0.70,
    "short_duration": 0.18,
    "inflation": 0.42,
    "credit": 0.30,
}


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


def _signed_centered_norm(value: float, scale: float) -> float:
    return _clamp01(0.5 + (float(value) / max(float(scale), 1e-8)))


def default_structured_news_features() -> Dict[str, float]:
    return {k: 0.0 for k in NEWS_STRUCTURED_FEATURE_KEYS}


def default_breadth_features() -> Dict[str, float]:
    return {k: 0.0 for k in BREADTH_FEATURE_KEYS}


def default_bond_reference_features() -> Dict[str, float]:
    out = {k: 0.0 for k in BOND_REFERENCE_FEATURE_KEYS}
    out["bond_nav_discount_norm"] = 0.5
    out["bond_auction_tail_norm"] = 0.5
    return out


def default_credit_context_features() -> Dict[str, float]:
    out = {k: 0.0 for k in CREDIT_CONTEXT_FEATURE_KEYS}
    out["bond_nav_stress_norm"] = 0.5
    return out


def default_data_quality_features() -> Dict[str, float]:
    return {k: 0.0 for k in DATA_QUALITY_FEATURE_KEYS}


def default_execution_lag_features() -> Dict[str, float]:
    return {k: 0.0 for k in EXECUTION_LAG_FEATURE_KEYS}


def load_latest_external_context(project_root: str | Path, category: str) -> Dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    token = str(category or "").strip().lower()
    if not token:
        return {}

    candidates = [
        root / "data" / "external_context" / f"{token}_latest.json",
        root / "exports" / "external_context" / f"{token}_latest.json",
        root / "governance" / "health" / f"{token}_latest.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        return payload if isinstance(payload, dict) else {"items": payload}
    return {}


def _parse_ts(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        if val > 1e12:
            val /= 1000.0
        if val > 1e9:
            return val
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.isdigit():
        val = float(text)
        if val > 1e12:
            val /= 1000.0
        if val > 1e9:
            return val
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp()


def _headline_text(row: Mapping[str, Any]) -> str:
    chunks: List[str] = []
    for key in ("headline", "title", "summary", "description", "content"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            chunks.append(val.strip())
    return " ".join(chunks)


def _publisher_text(row: Mapping[str, Any]) -> str:
    for key in ("source", "publisher", "provider", "channel"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return ""


def _headline_symbols(row: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for key in ("symbol", "ticker"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            out.add(val.strip().upper())
    for key in ("symbols", "tickers", "relatedSymbols", "relatedTickers", "securities"):
        val = row.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip():
                    out.add(item.strip().upper())
                elif isinstance(item, dict):
                    for sub_key in ("symbol", "ticker"):
                        sub_val = item.get(sub_key)
                        if isinstance(sub_val, str) and sub_val.strip():
                            out.add(sub_val.strip().upper())
        elif isinstance(val, str) and val.strip():
            for token in val.replace("|", ",").split(","):
                token = token.strip().upper()
                if token:
                    out.add(token)
    return out


def summarize_structured_news_items(
    items: Iterable[Mapping[str, Any]],
    *,
    symbol: str,
    now_ts: float,
    max_items: int,
) -> Dict[str, float]:
    out = default_structured_news_features()
    sym = str(symbol or "").strip().upper()
    rows: List[tuple[float, Mapping[str, Any], str]] = []
    for raw in items:
        if not isinstance(raw, Mapping):
            continue
        ts = None
        for key in ("publishedDate", "published", "dateTime", "datetime", "timestamp", "time", "displayDate"):
            ts = _parse_ts(raw.get(key))
            if ts is not None:
                break
        if ts is None or ts > now_ts:
            continue
        age = now_ts - ts
        if age > 48.0 * 3600.0:
            continue
        text = _headline_text(raw).strip()
        if not text:
            continue
        rows.append((age, raw, text))

    if not rows:
        return out

    rows.sort(key=lambda x: x[0])
    rows = rows[: max(max_items, 1)]
    total_weight = 0.0
    source_sum = 0.0
    relevance_sum = 0.0
    topic_scores = {key: 0.0 for key in _NEWS_TOPIC_TOKENS}
    session_scores = {"premarket": 0.0, "intraday": 0.0, "after_hours": 0.0}
    unique_heads: set[str] = set()

    for age, row, headline in rows:
        weight = math.exp(-age / 7200.0)
        total_weight += weight
        source = _publisher_text(row)
        source_quality = 0.55
        for token, score in _NEWS_SOURCE_QUALITY.items():
            if token in source:
                source_quality = score
                break
        source_sum += weight * source_quality

        related = _headline_symbols(row)
        text_lower = headline.lower()
        relevance = 0.0
        if sym and sym in related:
            relevance = 1.0
        elif sym and sym.lower() in text_lower:
            relevance = 0.8
        elif related:
            relevance = 0.35
        else:
            relevance = 0.45
        relevance_sum += weight * relevance

        for key, tokens in _NEWS_TOPIC_TOKENS.items():
            if any(tok in text_lower for tok in tokens):
                topic_scores[key] += weight

        ts = _parse_ts(
            row.get("publishedDate")
            or row.get("published")
            or row.get("dateTime")
            or row.get("datetime")
            or row.get("timestamp")
            or row.get("time")
            or row.get("displayDate")
        )
        if ts is not None and _ET_ZONE is not None:
            dt_et = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(_ET_ZONE)
            hm = (dt_et.hour * 60) + dt_et.minute
            if 4 * 60 <= hm < 9 * 60 + 30:
                session_scores["premarket"] += weight
            elif 9 * 60 + 30 <= hm < 16 * 60:
                session_scores["intraday"] += weight
            elif 16 * 60 <= hm < 20 * 60:
                session_scores["after_hours"] += weight

        unique_heads.add(text_lower)

    total_weight = max(total_weight, 1e-8)
    unique_ratio = len(unique_heads) / max(len(rows), 1)
    out.update(
        {
            "news_source_quality_norm": _clamp01(source_sum / total_weight),
            "news_entity_relevance_norm": _clamp01(relevance_sum / total_weight),
            "news_topic_earnings_norm": _clamp01(topic_scores["news_topic_earnings_norm"] / total_weight),
            "news_topic_guidance_norm": _clamp01(topic_scores["news_topic_guidance_norm"] / total_weight),
            "news_topic_mna_norm": _clamp01(topic_scores["news_topic_mna_norm"] / total_weight),
            "news_topic_regulatory_norm": _clamp01(topic_scores["news_topic_regulatory_norm"] / total_weight),
            "news_novelty_norm": _clamp01(unique_ratio),
            "news_duplicate_cluster_norm": _clamp01(1.0 - unique_ratio),
            "news_premarket_norm": _clamp01(session_scores["premarket"] / total_weight),
            "news_intraday_norm": _clamp01(session_scores["intraday"] / total_weight),
            "news_after_hours_norm": _clamp01(session_scores["after_hours"] / total_weight),
        }
    )
    return out


def _snapshot_symbols(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("symbols", "tickers", "etfs", "items_by_symbol"):
        node = snapshot.get(key)
        if isinstance(node, Mapping):
            return node
    return {}


def _snapshot_symbol_row(snapshot: Mapping[str, Any], symbol: str) -> Mapping[str, Any]:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return {}
    rows = _snapshot_symbols(snapshot)
    for key, row in rows.items():
        if str(key).strip().upper() == sym and isinstance(row, Mapping):
            return row
    direct = snapshot.get(sym)
    if isinstance(direct, Mapping):
        return direct
    return {}


def summarize_breadth_context(
    *,
    symbol: str,
    market_snapshot: Mapping[str, Any],
    context_market: Mapping[str, Mapping[str, Any]],
    external_snapshot: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    out = default_breadth_features()
    snapshot = external_snapshot if isinstance(external_snapshot, Mapping) else {}

    adv = _to_float(snapshot.get("advancers"), 0.0)
    dec = _to_float(snapshot.get("decliners"), 0.0)
    up_vol = _to_float(snapshot.get("up_volume"), 0.0)
    down_vol = _to_float(snapshot.get("down_volume"), 0.0)
    new_highs = _to_float(snapshot.get("new_highs"), 0.0)
    new_lows = _to_float(snapshot.get("new_lows"), 0.0)
    sector_dispersion = _to_float(snapshot.get("sector_dispersion"), 0.0)
    snapshot_available = 0.0

    if adv <= 0.0 and dec <= 0.0:
        pcts: List[float] = []
        weights: List[float] = []
        curr_pct = _to_float(market_snapshot.get("pct_from_close"), 0.0)
        if math.isfinite(curr_pct):
            pcts.append(curr_pct)
            weights.append(max(_to_float(market_snapshot.get("queue_depth"), 1.0), 1.0))
        for row in context_market.values():
            pct = _to_float(row.get("pct_from_close"), 0.0)
            if not math.isfinite(pct):
                continue
            pcts.append(pct)
            weights.append(max(_to_float(row.get("queue_depth"), 1.0), 1.0))
        if pcts:
            adv = float(sum(1 for pct in pcts if pct > 0.0))
            dec = float(sum(1 for pct in pcts if pct < 0.0))
            up_vol = float(sum(w for pct, w in zip(pcts, weights) if pct > 0.0))
            down_vol = float(sum(w for pct, w in zip(pcts, weights) if pct < 0.0))
            sector_dispersion = float(math.sqrt(sum((pct - (sum(pcts) / len(pcts))) ** 2 for pct in pcts) / max(len(pcts), 1)))
            new_highs = float(sum(1 for pct in pcts if pct >= 0.015))
            new_lows = float(sum(1 for pct in pcts if pct <= -0.015))
    else:
        snapshot_available = 1.0

    ad_ratio = (adv - dec) / max(adv + dec, 1.0)
    uv_ratio = (up_vol - down_vol) / max(up_vol + down_vol, 1.0)
    nh_nl_ratio = (new_highs - new_lows) / max(new_highs + new_lows, 1.0) if (new_highs + new_lows) > 0.0 else ad_ratio
    thrust = max((adv / max(adv + dec, 1.0)) - 0.50, 0.0) * 2.0
    risk_off = _clamp01((0.55 * (1.0 - _signed_centered_norm(ad_ratio, 1.0))) + (0.45 * _clamp01(sector_dispersion / 0.03)))

    out.update(
        {
            "breadth_snapshot_available": snapshot_available,
            "breadth_advance_decline_norm": _signed_centered_norm(ad_ratio, 1.0),
            "breadth_up_down_volume_norm": _signed_centered_norm(uv_ratio, 1.0),
            "breadth_new_high_low_norm": _signed_centered_norm(nh_nl_ratio, 1.0),
            "breadth_sector_dispersion_norm": _clamp01(sector_dispersion / 0.03),
            "breadth_thrust_norm": _clamp01(thrust),
            "breadth_risk_off_norm": risk_off,
        }
    )
    return out


def _ctx_pct(context_market: Mapping[str, Mapping[str, Any]], symbol: str) -> float:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return 0.0
    row = context_market.get(sym)
    if isinstance(row, Mapping):
        return _to_float(row.get("pct_from_close"), 0.0)
    return 0.0


def summarize_bond_quote_reference_features(
    *,
    symbol: str,
    quote_payload: Mapping[str, Any],
    last_price: float,
) -> Dict[str, float]:
    out = default_bond_reference_features()
    sym = str(symbol or "").strip().upper()
    nav = 0.0
    duration_years = 0.0
    convexity = 0.0
    ytm = 0.0
    flow_5d = 0.0
    aum = 0.0
    premium_discount = None

    for key in ("nav", "navPrice", "fundNav", "navPerShare"):
        nav = max(nav, _to_float(quote_payload.get(key), 0.0))
    for key in ("effectiveDuration", "duration", "avgDuration", "averageDurationYears", "fundDurationYears"):
        duration_years = max(duration_years, _to_float(quote_payload.get(key), 0.0))
    for key in ("convexity", "effectiveConvexity", "fundConvexity"):
        convexity = max(convexity, _to_float(quote_payload.get(key), 0.0))
    for key in ("yieldToMaturity", "ytm", "distributionYield", "secYield", "thirtyDayYield"):
        ytm = max(ytm, _to_float(quote_payload.get(key), 0.0))
    for key in ("netFlows5d", "fundFlow5d", "flow5d", "fundNetFlow5d"):
        flow_5d = _to_float(quote_payload.get(key), flow_5d)
    for key in ("aum", "totalAssets", "fundAssets", "netAssets"):
        aum = max(aum, _to_float(quote_payload.get(key), 0.0))
    for key in ("premiumDiscount", "premium_discount", "navPremiumDiscountPct"):
        raw = quote_payload.get(key)
        if raw is not None:
            premium_discount = _to_float(raw, 0.0)
            break

    if ytm > 1.0:
        ytm /= 100.0
    if premium_discount is None and nav > 0.0 and last_price > 0.0:
        premium_discount = (last_price - nav) / nav
    if premium_discount is not None and abs(premium_discount) > 1.0:
        premium_discount = premium_discount / 100.0

    role = "credit"
    if sym in {"TLT", "TLH", "VGLT", "EDV", "ZROZ", "IEF", "VGIT"}:
        role = "long_duration"
    elif sym in {"SHY", "FLOT", "VGSH", "SCHO"}:
        role = "short_duration"
    elif sym in {"TIP", "VTIP", "SCHP"}:
        role = "inflation"

    if duration_years <= 0.0:
        duration_years = _BOND_DURATION_ROLE_DEFAULTS.get(role, 5.0)
    if convexity <= 0.0:
        convexity = _BOND_CONVEXITY_ROLE_DEFAULTS.get(role, 0.30)

    flow_norm = 0.0
    if abs(flow_5d) > 0.0 and aum > 0.0:
        flow_norm = _clamp01(abs(flow_5d) / max(aum * 0.03, 1e-8))

    out.update(
        {
            "bond_duration_years_norm": _clamp01(duration_years / 20.0),
            "bond_convexity_norm": _clamp01(convexity),
            "bond_nav_discount_norm": _signed_centered_norm(-(premium_discount or 0.0), 0.05) if premium_discount is not None else 0.5,
            "bond_etf_flow_5d_norm": flow_norm,
            "bond_ytm_norm": _clamp01(max(ytm, 0.0) / 0.10),
        }
    )
    return out


def summarize_bond_reference_context(
    *,
    symbol: str,
    market_snapshot: Mapping[str, Any],
    context_market: Mapping[str, Mapping[str, Any]],
    calendar_features: Mapping[str, Any],
    external_snapshot: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    out = default_bond_reference_features()
    snapshot = external_snapshot if isinstance(external_snapshot, Mapping) else {}
    symbol_row = _snapshot_symbol_row(snapshot, symbol)
    yields = snapshot.get("treasury_yields") if isinstance(snapshot.get("treasury_yields"), Mapping) else {}
    if not isinstance(yields, Mapping):
        yields = {}

    curve_2s10s_proxy = _ctx_pct(context_market, "TLT") - _ctx_pct(context_market, "SHY")
    curve_5s30s_proxy = _ctx_pct(context_market, "TLT") - _ctx_pct(context_market, "IEF")
    inflation_proxy = _ctx_pct(context_market, "TIP") - _ctx_pct(context_market, "IEF")

    y2 = _to_float(yields.get("2y", yields.get("two_year", 0.0)), 0.0)
    y5 = _to_float(yields.get("5y", yields.get("five_year", 0.0)), 0.0)
    y10 = _to_float(yields.get("10y", yields.get("ten_year", 0.0)), 0.0)
    y30 = _to_float(yields.get("30y", yields.get("thirty_year", 0.0)), 0.0)
    real_10 = _to_float(yields.get("real_10y", yields.get("ten_year_real", 0.0)), 0.0)

    if y2 <= 0.0:
        y2 = 3.5 - (_ctx_pct(context_market, "SHY") * 80.0)
    if y5 <= 0.0:
        y5 = 3.8 - (_ctx_pct(context_market, "IEF") * 80.0)
    if y10 <= 0.0:
        y10 = 4.0 - (_ctx_pct(context_market, "TLT") * 85.0)
    if y30 <= 0.0:
        y30 = 4.2 - (_ctx_pct(context_market, "TLT") * 95.0)
    if real_10 <= 0.0:
        real_10 = 1.5 - (inflation_proxy * 40.0)

    curve_2s10s = _to_float(y10 - y2, curve_2s10s_proxy * 100.0)
    curve_5s30s = _to_float(y30 - y5, curve_5s30s_proxy * 100.0)
    duration_norm = _to_float(symbol_row.get("duration_years_norm"), _to_float(market_snapshot.get("bond_duration_years_norm"), 0.0))
    convexity_norm = _to_float(symbol_row.get("convexity_norm"), _to_float(market_snapshot.get("bond_convexity_norm"), 0.0))
    nav_discount_norm = _to_float(symbol_row.get("nav_discount_norm"), _to_float(market_snapshot.get("bond_nav_discount_norm"), 0.5))
    flow_5d_norm = _to_float(symbol_row.get("flow_5d_norm"), _to_float(market_snapshot.get("bond_etf_flow_5d_norm"), 0.0))
    ytm_norm = _to_float(symbol_row.get("ytm_norm"), _to_float(market_snapshot.get("bond_ytm_norm"), 0.0))
    auction_tail_bps = _to_float(snapshot.get("auction_tail_bps", symbol_row.get("auction_tail_bps", 0.0)), 0.0)
    auction_window_norm = _to_float(calendar_features.get("calendar_treasury_auction_norm"), 0.0)

    out.update(
        {
            "bond_yield_2y_norm": _clamp01(max(y2, 0.0) / 8.0),
            "bond_yield_5y_norm": _clamp01(max(y5, 0.0) / 8.0),
            "bond_yield_10y_norm": _clamp01(max(y10, 0.0) / 8.0),
            "bond_yield_30y_norm": _clamp01(max(y30, 0.0) / 8.0),
            "bond_curve_2s10s_norm": _signed_centered_norm(curve_2s10s, 3.0),
            "bond_curve_5s30s_norm": _signed_centered_norm(curve_5s30s, 3.0),
            "bond_real_yield_10y_norm": _clamp01(max(real_10, 0.0) / 4.0),
            "bond_duration_years_norm": duration_norm if duration_norm > 0.0 else out["bond_duration_years_norm"],
            "bond_convexity_norm": convexity_norm if convexity_norm > 0.0 else out["bond_convexity_norm"],
            "bond_nav_discount_norm": nav_discount_norm,
            "bond_etf_flow_5d_norm": flow_5d_norm,
            "bond_ytm_norm": ytm_norm,
            "bond_auction_window_norm": _clamp01(auction_window_norm),
            "bond_auction_tail_norm": _signed_centered_norm(auction_tail_bps, 6.0),
        }
    )
    return out


def summarize_credit_context(
    *,
    symbol: str,
    market_snapshot: Mapping[str, Any],
    context_market: Mapping[str, Mapping[str, Any]],
    external_snapshot: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    out = default_credit_context_features()
    snapshot = external_snapshot if isinstance(external_snapshot, Mapping) else {}
    symbol_row = _snapshot_symbol_row(snapshot, symbol)

    spread_level = _to_float(snapshot.get("credit_spread_bps", symbol_row.get("credit_spread_bps", 0.0)), 0.0)
    spread_change = _to_float(snapshot.get("credit_spread_change_bps", symbol_row.get("credit_spread_change_bps", 0.0)), 0.0)
    hy_ig_flow = _to_float(snapshot.get("hy_ig_flow_ratio", symbol_row.get("hy_ig_flow_ratio", 0.0)), 0.0)

    if spread_level <= 0.0:
        spread_level = 100.0 + ((_ctx_pct(context_market, "HYG") - _ctx_pct(context_market, "LQD")) * 9000.0)
    if spread_change == 0.0:
        spread_change = (_to_float(context_market.get("HYG", {}).get("mom_5m"), 0.0) - _to_float(context_market.get("LQD", {}).get("mom_5m"), 0.0)) * 5000.0

    nav_stress = max(
        abs(_to_float(symbol_row.get("nav_discount_pct"), 0.0)),
        abs((_to_float(market_snapshot.get("bond_nav_discount_norm"), 0.5) - 0.5) * 0.10),
    )

    out.update(
        {
            "bond_credit_spread_level_norm": _clamp01(max(spread_level, 0.0) / 800.0),
            "bond_credit_spread_change_norm": _signed_centered_norm(spread_change, 150.0),
            "bond_hy_ig_flow_norm": _signed_centered_norm(hy_ig_flow, 1.5),
            "bond_nav_stress_norm": _clamp01(nav_stress / 0.05),
        }
    )
    return out


def summarize_data_quality_context(
    *,
    market_snapshot: Mapping[str, Any],
    freshness_ok: bool,
    freshness_age_seconds: float,
    symbol_fail_count: int,
    symbol_stale_count: int,
    symbol_circuit_hits: int,
    quarantine_seconds: float,
    missing_feature_count: int,
    required_feature_count: int,
) -> Dict[str, float]:
    out = default_data_quality_features()
    quote_agreement = _to_float(market_snapshot.get("quote_history_agreement_norm"), 0.0)
    quote_dev = _to_float(market_snapshot.get("quote_history_relative_deviation"), 0.0)
    bid_size = max(_to_float(market_snapshot.get("bid_size"), 0.0), 0.0)
    ask_size = max(_to_float(market_snapshot.get("ask_size"), 0.0), 0.0)
    imbalance = (bid_size - ask_size) / max(bid_size + ask_size, 1e-8) if (bid_size > 0.0 or ask_size > 0.0) else 0.0
    missing_ratio = (float(missing_feature_count) / max(float(required_feature_count), 1.0)) if required_feature_count > 0 else 0.0

    if quote_agreement <= 0.0:
        quote_agreement = max(1.0 - min(quote_dev / 0.05, 1.0), 0.0)

    out.update(
        {
            "data_quality_quote_agreement_norm": _clamp01(quote_agreement),
            "data_quality_quote_deviation_norm": _clamp01(quote_dev / 0.05),
            "data_quality_stale_streak_norm": _clamp01(float(symbol_stale_count) / 8.0),
            "data_quality_fail_streak_norm": _clamp01(float(symbol_fail_count) / 5.0),
            "data_quality_quarantine_hits_norm": max(
                _clamp01(float(symbol_circuit_hits) / 5.0),
                _clamp01(float(quarantine_seconds) / 900.0),
            ),
            "data_quality_missing_feature_ratio_norm": _clamp01(missing_ratio),
            "data_quality_bid_ask_imbalance_norm": _signed_centered_norm(imbalance, 1.0),
            "data_quality_market_data_latency_norm": _clamp01(max(_to_float(market_snapshot.get("market_data_latency_ms"), 0.0), freshness_age_seconds if not freshness_ok else 0.0) / 2500.0),
        }
    )
    return out
