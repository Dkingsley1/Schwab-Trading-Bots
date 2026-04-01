from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from market_context_features import (
    BOND_REFERENCE_FEATURE_KEYS,
    BREADTH_FEATURE_KEYS,
    CREDIT_CONTEXT_FEATURE_KEYS,
    NEWS_STRUCTURED_FEATURE_KEYS,
    load_latest_external_context,
    summarize_bond_reference_context,
    summarize_breadth_context,
    summarize_credit_context,
)

try:
    from derivatives_features import summarize_calendar_payload
except Exception:
    summarize_calendar_payload = None


RuntimeObservation = Dict[str, Any]
RuntimeSequenceMap = Dict[Tuple[str, str], List[RuntimeObservation]]
RuntimeFeatureBuilder = Callable[[Sequence[RuntimeObservation], int], np.ndarray]
RuntimeLabelBuilder = Callable[[Sequence[RuntimeObservation], int, int], Optional[float]]
RuntimeSampleFilter = Callable[[Sequence[RuntimeObservation], int, int], bool]
RuntimeConfidenceBuilder = Callable[[Sequence[RuntimeObservation], int, int], float]

_DEFAULT_RUNTIME_LABEL_BALANCE_MAX_RATIO = 4.0
_DEFAULT_RUNTIME_LABEL_BALANCE_MIN_MINORITY_SAMPLES = 6
_DEFAULT_RUNTIME_LABEL_BALANCE_MIN_TOTAL_SAMPLES = 64

_ROOT_STRATEGY_PRIORITY = {
    "grand_master_bot": 0,
    "grand_master_intent_bot": 1,
}

_RUNTIME_NEWS_EVENT_KEYS = {
    "news_available",
    "news_items_30m",
    "news_items_2h",
    "news_items_24h",
    "news_sentiment",
    "news_negative_share",
    "news_positive_share",
    "news_shock_rate",
    "news_recent_impact",
    "news_novelty_norm",
}

_RUNTIME_CALENDAR_EVENT_KEYS = {
    "calendar_feed_available",
    "calendar_event_proximity_norm",
    "calendar_next_event_norm",
    "calendar_events_24h_norm",
    "calendar_high_impact_24h_norm",
    "calendar_macro_event_norm",
    "calendar_macro_surprise_norm",
    "calendar_macro_abs_surprise_norm",
    "calendar_macro_revision_norm",
    "calendar_fomc_event_norm",
    "calendar_cpi_event_norm",
    "calendar_labor_event_norm",
    "calendar_treasury_auction_norm",
}

_RUNTIME_MARKET_MICRO_KEYS = {
    "market_micro_premarket_pressure_norm",
    "market_micro_opening_auction_norm",
    "market_micro_power_hour_pressure_norm",
    "market_micro_closing_auction_norm",
    "market_micro_relative_volume_norm",
    "market_micro_order_flow_imbalance_norm",
    "market_micro_options_flow_norm",
    "market_micro_short_pressure_norm",
    "market_micro_credit_flow_norm",
    "market_micro_gap_continuation_norm",
    "market_micro_reversal_risk_norm",
    "market_micro_trend_persistence_norm",
    "market_micro_range_expansion_norm",
    "market_micro_block_trade_norm",
}

_RUNTIME_SEC_EDGAR_KEYS = {
    "sec_filing_count_7d_norm",
    "sec_high_impact_7d_norm",
    "sec_earnings_7d_norm",
    "sec_guidance_7d_norm",
    "sec_regulatory_7d_norm",
    "sec_ownership_30d_norm",
    "sec_insider_30d_norm",
    "sec_recent_proximity_norm",
    "sec_recent_symbols_norm",
    "sec_recent_filings_1d_norm",
    "sec_recent_high_impact_1d_norm",
}

_RUNTIME_EXTENDED_QUANT_KEYS = {
    "cot_equity_risk_on_norm",
    "cot_equity_crowding_norm",
    "cot_bond_risk_off_norm",
    "cot_usd_bullish_norm",
    "cot_macro_positioning_stress_norm",
    "cot_risk_on_norm",
    "sofr_level_norm",
    "sofr_30d_avg_norm",
    "sofr_90d_avg_norm",
    "sofr_180d_avg_norm",
    "sofr_term_pressure_norm",
    "sofr_funding_stress_norm",
    "sofr_index_norm",
    "cboe_total_put_call_norm",
    "cboe_index_put_call_norm",
    "cboe_equity_put_call_norm",
    "cboe_put_call_stress_norm",
    "cboe_vix_spot_norm",
    "short_threshold_listed_norm",
    "short_threshold_rule3210_norm",
    "short_threshold_symbol_share_norm",
    "short_threshold_total_listed_norm",
    "short_threshold_recency_norm",
    "short_ftd_presence_norm",
    "short_ftd_quantity_norm",
    "short_ftd_symbol_share_norm",
    "short_ftd_total_hits_norm",
}

_RUNTIME_CRYPTO_MARKET_KEYS = {
    "crypto_deribit_futures_oi_norm",
    "crypto_deribit_options_oi_norm",
    "crypto_deribit_mark_iv_norm",
    "crypto_deribit_basis_norm",
    "crypto_kraken_volume_norm",
    "crypto_kraken_range_norm",
    "crypto_hyperliquid_funding_norm",
    "crypto_hyperliquid_open_interest_norm",
    "crypto_hyperliquid_basis_norm",
    "crypto_coinmetrics_tx_count_norm",
    "crypto_coinmetrics_active_addr_norm",
    "crypto_coingecko_volume_norm",
    "crypto_coingecko_momentum_norm",
    "crypto_cross_provider_price_agreement_norm",
    "crypto_defillama_stablecoin_growth_norm",
    "crypto_defillama_dex_volume_growth_norm",
    "crypto_etherscan_gas_norm",
}

_RUNTIME_MARKET_CRYPTO_CORRELATION_KEYS = {
    "market_crypto_risk_corr_norm",
    "market_crypto_spy_corr_norm",
    "market_crypto_qqq_corr_norm",
    "market_crypto_tlt_corr_norm",
    "market_crypto_uup_inverse_corr_norm",
    "market_crypto_gold_corr_norm",
    "market_crypto_current_alignment_norm",
    "market_crypto_divergence_norm",
    "market_crypto_corr_confidence_norm",
}

_RUNTIME_FX_MARKET_KEYS = {
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
}

_RUNTIME_DIVIDEND_DRIP_KEYS = {
    "dividend_drip_active_norm",
    "dividend_drip_recent_reinvest_norm",
    "dividend_drip_cash_only_norm",
    "dividend_drip_share_credit_norm",
    "dividend_drip_event_recency_norm",
    "dividend_drip_confidence_norm",
}

_RUNTIME_GAP_FILL_KEYS = set(BREADTH_FEATURE_KEYS) | set(BOND_REFERENCE_FEATURE_KEYS) | set(CREDIT_CONTEXT_FEATURE_KEYS) | set(NEWS_STRUCTURED_FEATURE_KEYS) | _RUNTIME_NEWS_EVENT_KEYS | _RUNTIME_CALENDAR_EVENT_KEYS | _RUNTIME_MARKET_MICRO_KEYS | _RUNTIME_SEC_EDGAR_KEYS | _RUNTIME_EXTENDED_QUANT_KEYS | _RUNTIME_CRYPTO_MARKET_KEYS | _RUNTIME_MARKET_CRYPTO_CORRELATION_KEYS | _RUNTIME_FX_MARKET_KEYS | _RUNTIME_DIVIDEND_DRIP_KEYS


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _parse_ts(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    text = str(raw).strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _path_day_utc(path: Path) -> Optional[datetime]:
    parts = path.stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    stamp = parts[-1]
    if len(stamp) != 8 or (not stamp.isdigit()):
        return None
    try:
        return datetime.strptime(stamp, "%Y%m%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _is_missing_feature(features: Mapping[str, Any], name: str) -> bool:
    if name not in features:
        return True
    try:
        value = float(features.get(name))
    except Exception:
        return True
    return not math.isfinite(value)


def _set_missing_feature(features: Dict[str, Any], name: str, value: Any) -> None:
    try:
        coerced = float(value)
    except Exception:
        return
    if not math.isfinite(coerced):
        return
    if _is_missing_feature(features, name):
        features[name] = coerced


def _feature_subset(payload: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in keys:
        try:
            value = float(payload.get(key))  # type: ignore[arg-type]
        except Exception:
            continue
        if math.isfinite(value):
            out[str(key)] = value
    return out


def _symbol_feature_subset(payload: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for symbol, row in payload.items():
        if not isinstance(row, Mapping):
            continue
        subset = _feature_subset(row, keys)
        if subset:
            out[str(symbol).strip().upper()] = subset
    return out


def _live_macro_gap_fill_features(payload: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not isinstance(payload, Mapping):
        return {}, {}

    active = bool(payload.get("active"))
    shock_hint = max(0.0, min(_safe_float(payload.get("shock_hint"), 0.0), 1.0))
    sentiment_hint = max(-1.0, min(_safe_float(payload.get("sentiment_hint"), 0.0), 1.0))
    stance = str(payload.get("stance") or "").strip().lower()
    template = str(payload.get("template") or "").strip().lower()
    source = str(payload.get("source") or "").strip().lower()
    strength = max(shock_hint, 1.0 if active else 0.0)
    if strength <= 0.0:
        return {}, {}

    macro_event = 1.0 if (template in {"powell", "fed", "fomc"} or "federal reserve" in source or "powell" in source) else min(0.7, strength)
    calendar_features = {
        "calendar_feed_available": strength,
        "calendar_event_proximity_norm": strength,
        "calendar_next_event_norm": strength,
        "calendar_events_24h_norm": strength,
        "calendar_high_impact_24h_norm": strength,
        "calendar_macro_event_norm": macro_event,
        "calendar_macro_surprise_norm": max(0.0, min(1.0, 0.5 + (sentiment_hint * 0.5))),
        "calendar_macro_abs_surprise_norm": abs(sentiment_hint),
        "calendar_fomc_event_norm": macro_event,
    }
    if "auction" in stance or "auction" in template:
        calendar_features["calendar_treasury_auction_norm"] = strength

    news_features = {
        "news_available": max(0.35, strength),
        "news_items_30m": min(strength, 1.0) * 0.6,
        "news_items_2h": min(strength, 1.0) * 0.75,
        "news_items_24h": min(strength, 1.0),
        "news_sentiment": sentiment_hint,
        "news_negative_share": max(-sentiment_hint, 0.0),
        "news_positive_share": max(sentiment_hint, 0.0),
        "news_shock_rate": strength,
        "news_recent_impact": strength,
        "news_source_quality_norm": 0.9,
        "news_entity_relevance_norm": 0.9 if bool(payload.get("broad_market")) else 0.65,
        "news_novelty_norm": min(0.45 + (strength * 0.5), 1.0),
    }
    return calendar_features, news_features


def _load_runtime_gap_fill_context(project_root: Path) -> Dict[str, Any]:
    tradingeconomics = load_latest_external_context(project_root, "tradingeconomics")
    market_breadth = load_latest_external_context(project_root, "market_breadth")
    bond_reference = load_latest_external_context(project_root, "bond_reference")
    live_macro = load_latest_external_context(project_root, "live_macro")
    official_macro = load_latest_external_context(project_root, "official_macro_context")
    market_micro = load_latest_external_context(project_root, "market_micro")
    sec_edgar = load_latest_external_context(project_root, "sec_edgar")
    extended_quant = load_latest_external_context(project_root, "extended_quant_context")
    crypto_market = load_latest_external_context(project_root, "crypto_market_context")
    market_crypto_correlation = load_latest_external_context(project_root, "market_crypto_correlation")
    fx_market_context = load_latest_external_context(project_root, "fx_market_context")
    dividend_drip_state = load_latest_external_context(project_root, "dividend_drip_state")

    te_derived = tradingeconomics.get("derived") if isinstance(tradingeconomics.get("derived"), Mapping) else {}
    official_derived = official_macro.get("derived") if isinstance(official_macro.get("derived"), Mapping) else {}
    sec_derived = sec_edgar.get("derived") if isinstance(sec_edgar.get("derived"), Mapping) else {}
    extended_derived = extended_quant.get("derived") if isinstance(extended_quant.get("derived"), Mapping) else {}
    crypto_derived = crypto_market.get("derived") if isinstance(crypto_market.get("derived"), Mapping) else {}
    market_crypto_corr_derived = market_crypto_correlation.get("derived") if isinstance(market_crypto_correlation.get("derived"), Mapping) else {}
    fx_market_derived = fx_market_context.get("derived") if isinstance(fx_market_context.get("derived"), Mapping) else {}
    dividend_drip_derived = dividend_drip_state.get("derived") if isinstance(dividend_drip_state.get("derived"), Mapping) else {}
    te_calendar = te_derived.get("calendar_features") if isinstance(te_derived.get("calendar_features"), Mapping) else {}
    te_news = te_derived.get("news_features") if isinstance(te_derived.get("news_features"), Mapping) else {}
    te_calendar_rows = te_derived.get("calendar_rows") if isinstance(te_derived.get("calendar_rows"), list) else []
    official_calendar = official_derived.get("calendar_features") if isinstance(official_derived.get("calendar_features"), Mapping) else {}
    official_news = official_derived.get("news_features") if isinstance(official_derived.get("news_features"), Mapping) else {}
    official_calendar_rows = official_derived.get("calendar_rows") if isinstance(official_derived.get("calendar_rows"), list) else []
    official_bond_overlay = official_derived.get("bond_reference_overlay") if isinstance(official_derived.get("bond_reference_overlay"), Mapping) else {}
    sec_calendar = sec_derived.get("calendar_features") if isinstance(sec_derived.get("calendar_features"), Mapping) else {}
    sec_news = sec_derived.get("news_features") if isinstance(sec_derived.get("news_features"), Mapping) else {}
    sec_global = sec_derived.get("global_features") if isinstance(sec_derived.get("global_features"), Mapping) else {}
    sec_symbol = sec_derived.get("symbol_features") if isinstance(sec_derived.get("symbol_features"), Mapping) else {}
    extended_calendar = extended_derived.get("calendar_features") if isinstance(extended_derived.get("calendar_features"), Mapping) else {}
    extended_news = extended_derived.get("news_features") if isinstance(extended_derived.get("news_features"), Mapping) else {}
    extended_global = extended_derived.get("global_features") if isinstance(extended_derived.get("global_features"), Mapping) else {}
    extended_symbol = extended_derived.get("symbol_features") if isinstance(extended_derived.get("symbol_features"), Mapping) else {}
    extended_bond_overlay = extended_derived.get("bond_reference_overlay") if isinstance(extended_derived.get("bond_reference_overlay"), Mapping) else {}
    crypto_news = crypto_derived.get("news_features") if isinstance(crypto_derived.get("news_features"), Mapping) else {}
    crypto_global = crypto_derived.get("global_features") if isinstance(crypto_derived.get("global_features"), Mapping) else {}
    crypto_symbol = crypto_derived.get("symbol_features") if isinstance(crypto_derived.get("symbol_features"), Mapping) else {}
    market_crypto_corr_global = market_crypto_corr_derived.get("global_features") if isinstance(market_crypto_corr_derived.get("global_features"), Mapping) else {}
    market_crypto_corr_symbol = market_crypto_corr_derived.get("symbol_features") if isinstance(market_crypto_corr_derived.get("symbol_features"), Mapping) else {}
    fx_market_global = fx_market_derived.get("global_features") if isinstance(fx_market_derived.get("global_features"), Mapping) else {}
    fx_market_symbol = fx_market_derived.get("symbol_features") if isinstance(fx_market_derived.get("symbol_features"), Mapping) else {}
    dividend_drip_global = dividend_drip_derived.get("global_features") if isinstance(dividend_drip_derived.get("global_features"), Mapping) else {}
    dividend_drip_symbol = dividend_drip_derived.get("symbol_features") if isinstance(dividend_drip_derived.get("symbol_features"), Mapping) else {}

    calendar_features = _feature_subset(te_calendar, _RUNTIME_CALENDAR_EVENT_KEYS)
    if te_calendar_rows and callable(summarize_calendar_payload):
        try:
            summarized = summarize_calendar_payload(te_calendar_rows, now_ts=datetime.now(timezone.utc).timestamp(), max_items=600)
        except Exception:
            summarized = {}
        for key, value in _feature_subset(summarized, _RUNTIME_CALENDAR_EVENT_KEYS).items():
            if key not in calendar_features:
                calendar_features[key] = value
    for key, value in _feature_subset(official_calendar, _RUNTIME_CALENDAR_EVENT_KEYS).items():
        calendar_features[key] = max(calendar_features.get(key, 0.0), value)
    for key, value in _feature_subset(sec_calendar, _RUNTIME_CALENDAR_EVENT_KEYS).items():
        calendar_features[key] = max(calendar_features.get(key, 0.0), value)
    for key, value in _feature_subset(extended_calendar, _RUNTIME_CALENDAR_EVENT_KEYS).items():
        calendar_features[key] = max(calendar_features.get(key, 0.0), value)
    if official_calendar_rows and callable(summarize_calendar_payload):
        try:
            official_summarized = summarize_calendar_payload(official_calendar_rows, now_ts=datetime.now(timezone.utc).timestamp(), max_items=600)
        except Exception:
            official_summarized = {}
        for key, value in _feature_subset(official_summarized, _RUNTIME_CALENDAR_EVENT_KEYS).items():
            calendar_features[key] = max(calendar_features.get(key, 0.0), value)

    news_features = _feature_subset(te_news, set(NEWS_STRUCTURED_FEATURE_KEYS) | _RUNTIME_NEWS_EVENT_KEYS)
    if te_news:
        news_features.setdefault("news_available", 0.35)
        news_features.setdefault("news_items_24h", 0.4)
    for key, value in _feature_subset(official_news, set(NEWS_STRUCTURED_FEATURE_KEYS) | _RUNTIME_NEWS_EVENT_KEYS).items():
        if key == "news_sentiment":
            if abs(value) > abs(news_features.get(key, 0.0)):
                news_features[key] = value
        else:
            news_features[key] = max(news_features.get(key, 0.0), value)
    for extra_news in (sec_news, extended_news):
        for key, value in _feature_subset(extra_news, set(NEWS_STRUCTURED_FEATURE_KEYS) | _RUNTIME_NEWS_EVENT_KEYS).items():
            if key == "news_sentiment":
                if abs(value) > abs(news_features.get(key, 0.0)):
                    news_features[key] = value
            else:
                news_features[key] = max(news_features.get(key, 0.0), value)
    for key, value in _feature_subset(crypto_news, set(NEWS_STRUCTURED_FEATURE_KEYS) | _RUNTIME_NEWS_EVENT_KEYS).items():
        if key == "news_sentiment":
            if abs(value) > abs(news_features.get(key, 0.0)):
                news_features[key] = value
        else:
            news_features[key] = max(news_features.get(key, 0.0), value)

    live_macro_calendar, live_macro_news = _live_macro_gap_fill_features(live_macro if isinstance(live_macro, Mapping) else {})
    breadth_features = summarize_breadth_context(
        symbol="SPY",
        market_snapshot={},
        context_market={},
        external_snapshot=market_breadth if isinstance(market_breadth, Mapping) else {},
    )
    merged_bond_reference = dict(bond_reference) if isinstance(bond_reference, Mapping) else {}
    for overlay in (official_bond_overlay, extended_bond_overlay):
        if not isinstance(overlay, Mapping):
            continue
        for key, value in overlay.items():
            if isinstance(value, Mapping) and isinstance(merged_bond_reference.get(key), Mapping):
                nested = dict(merged_bond_reference[key])
                nested.update(value)
                merged_bond_reference[key] = nested
            else:
                merged_bond_reference[key] = value
    market_micro_features = {}
    market_micro_derived = market_micro.get("derived") if isinstance(market_micro.get("derived"), Mapping) else {}
    market_micro_global = market_micro_derived.get("global_features") if isinstance(market_micro_derived.get("global_features"), Mapping) else {}
    for key, value in _feature_subset(market_micro_global, _RUNTIME_MARKET_MICRO_KEYS).items():
        market_micro_features[key] = value
    external_global_features = {}
    external_global_features.update(_feature_subset(sec_global, _RUNTIME_SEC_EDGAR_KEYS))
    external_global_features.update(_feature_subset(extended_global, _RUNTIME_EXTENDED_QUANT_KEYS))
    external_global_features.update(_feature_subset(crypto_global, _RUNTIME_CRYPTO_MARKET_KEYS))
    external_global_features.update(_feature_subset(market_crypto_corr_global, _RUNTIME_MARKET_CRYPTO_CORRELATION_KEYS))
    external_global_features.update(_feature_subset(fx_market_global, _RUNTIME_FX_MARKET_KEYS))
    external_global_features.update(_feature_subset(dividend_drip_global, _RUNTIME_DIVIDEND_DRIP_KEYS))
    external_symbol_features = _symbol_feature_subset(sec_symbol, _RUNTIME_SEC_EDGAR_KEYS)
    for symbol, subset in _symbol_feature_subset(extended_symbol, _RUNTIME_EXTENDED_QUANT_KEYS).items():
        current = external_symbol_features.setdefault(symbol, {})
        current.update(subset)
    for symbol, subset in _symbol_feature_subset(crypto_symbol, _RUNTIME_CRYPTO_MARKET_KEYS).items():
        current = external_symbol_features.setdefault(symbol, {})
        current.update(subset)
    for symbol, subset in _symbol_feature_subset(market_crypto_corr_symbol, _RUNTIME_MARKET_CRYPTO_CORRELATION_KEYS).items():
        current = external_symbol_features.setdefault(symbol, {})
        current.update(subset)
    for symbol, subset in _symbol_feature_subset(fx_market_symbol, _RUNTIME_FX_MARKET_KEYS).items():
        current = external_symbol_features.setdefault(symbol, {})
        current.update(subset)
    for symbol, subset in _symbol_feature_subset(dividend_drip_symbol, _RUNTIME_DIVIDEND_DRIP_KEYS).items():
        current = external_symbol_features.setdefault(symbol, {})
        current.update(subset)

    return {
        "calendar_features": calendar_features,
        "news_features": news_features,
        "live_macro_calendar": live_macro_calendar,
        "live_macro_news": live_macro_news,
        "breadth_features": breadth_features,
        "bond_reference": merged_bond_reference,
        "market_micro_features": market_micro_features,
        "external_global_features": external_global_features,
        "external_symbol_features": external_symbol_features,
    }


def _enrich_runtime_observation(
    obs: RuntimeObservation,
    *,
    carry_forward_features: Mapping[str, float],
    gap_fill_context: Mapping[str, Any],
) -> RuntimeObservation:
    enriched = dict(obs)
    features = dict(obs.get("features") if isinstance(obs.get("features"), Mapping) else {})

    for key, value in carry_forward_features.items():
        _set_missing_feature(features, key, value)

    calendar_features = gap_fill_context.get("calendar_features") if isinstance(gap_fill_context.get("calendar_features"), Mapping) else {}
    news_features = gap_fill_context.get("news_features") if isinstance(gap_fill_context.get("news_features"), Mapping) else {}
    live_macro_calendar = gap_fill_context.get("live_macro_calendar") if isinstance(gap_fill_context.get("live_macro_calendar"), Mapping) else {}
    live_macro_news = gap_fill_context.get("live_macro_news") if isinstance(gap_fill_context.get("live_macro_news"), Mapping) else {}
    breadth_features = gap_fill_context.get("breadth_features") if isinstance(gap_fill_context.get("breadth_features"), Mapping) else {}
    bond_reference = gap_fill_context.get("bond_reference") if isinstance(gap_fill_context.get("bond_reference"), Mapping) else {}
    market_micro_features = gap_fill_context.get("market_micro_features") if isinstance(gap_fill_context.get("market_micro_features"), Mapping) else {}
    external_global_features = gap_fill_context.get("external_global_features") if isinstance(gap_fill_context.get("external_global_features"), Mapping) else {}
    external_symbol_features = gap_fill_context.get("external_symbol_features") if isinstance(gap_fill_context.get("external_symbol_features"), Mapping) else {}

    for key, value in calendar_features.items():
        _set_missing_feature(features, str(key), value)
    for key, value in news_features.items():
        _set_missing_feature(features, str(key), value)
    for key, value in live_macro_calendar.items():
        _set_missing_feature(features, str(key), value)
    for key, value in live_macro_news.items():
        _set_missing_feature(features, str(key), value)
    for key, value in breadth_features.items():
        _set_missing_feature(features, str(key), value)
    for key, value in market_micro_features.items():
        _set_missing_feature(features, str(key), value)
    symbol = str(obs.get("symbol") or "").strip().upper()
    symbol_feature_map = external_symbol_features.get(symbol) if isinstance(external_symbol_features.get(symbol), Mapping) else {}
    for key, value in symbol_feature_map.items():
        _set_missing_feature(features, str(key), value)
    for key, value in external_global_features.items():
        _set_missing_feature(features, str(key), value)

    bond_features = summarize_bond_reference_context(
        symbol=symbol,
        market_snapshot=features,
        context_market={},
        calendar_features=features,
        external_snapshot=bond_reference,
    )
    credit_features = summarize_credit_context(
        symbol=symbol,
        market_snapshot=features,
        context_market={},
        external_snapshot=bond_reference,
    )
    for key, value in bond_features.items():
        _set_missing_feature(features, str(key), value)
    for key, value in credit_features.items():
        _set_missing_feature(features, str(key), value)

    enriched["features"] = features
    return enriched


def _recent_decision_paths(project_root: Path, *, lookback_days: int) -> List[Path]:
    root = Path(project_root).expanduser().resolve()
    since_utc = datetime.now(timezone.utc) - timedelta(days=max(int(lookback_days), 1))
    cutoff_day = (since_utc - timedelta(days=1)).date()
    out: List[Path] = []
    for raw in glob.glob(str(root / "decision_explanations" / "shadow*" / "decision_explanations_*.jsonl")):
        path = Path(raw)
        day_utc = _path_day_utc(path)
        if day_utc is not None and day_utc.date() >= cutoff_day:
            out.append(path)
            continue
        try:
            mtime_utc = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            continue
        if mtime_utc >= since_utc - timedelta(days=1):
            out.append(path)
    return sorted({p.resolve() for p in out})


def observation_feature(obs: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    token = str(name or "").strip()
    if not token:
        return default

    if token == "last_price":
        return _safe_float(obs.get("price"), default)
    if token == "symbol_hash":
        return _stable_hash01(str(obs.get("symbol") or ""))
    if token == "mode_hash":
        return _stable_hash01(str(obs.get("mode") or ""))

    features = obs.get("features") if isinstance(obs.get("features"), dict) else {}
    return _safe_float(features.get(token), default)


def _stable_hash01(text: str) -> float:
    if not text:
        return 0.0
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h / 0xFFFFFFFF


def price_change(sequence: Sequence[RuntimeObservation], idx: int, lookback: int = 1) -> float:
    back = max(int(lookback), 1)
    if idx <= 0:
        return 0.0
    j = max(0, idx - back)
    prev_price = observation_feature(sequence[j], "last_price", 0.0)
    curr_price = observation_feature(sequence[idx], "last_price", 0.0)
    if prev_price <= 0.0 or curr_price <= 0.0:
        return 0.0
    return (curr_price / max(prev_price, 1e-8)) - 1.0


def feature_std(sequence: Sequence[RuntimeObservation], idx: int, name: str, window: int = 6) -> float:
    w = max(int(window), 1)
    start = max(0, idx - w + 1)
    vals = [observation_feature(sequence[j], name, 0.0) for j in range(start, idx + 1)]
    if not vals:
        return 0.0
    return float(np.std(np.asarray(vals, dtype=np.float64)))


def feature_mean(sequence: Sequence[RuntimeObservation], idx: int, name: str, window: int = 6) -> float:
    w = max(int(window), 1)
    start = max(0, idx - w + 1)
    vals = [observation_feature(sequence[j], name, 0.0) for j in range(start, idx + 1)]
    if not vals:
        return 0.0
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def feature_ema(sequence: Sequence[RuntimeObservation], idx: int, name: str, span: int = 6) -> float:
    use_span = max(int(span), 1)
    start = max(0, idx - (use_span * 6))
    alpha = 2.0 / (use_span + 1.0)
    out = 0.0
    initialized = False
    for j in range(start, idx + 1):
        val = observation_feature(sequence[j], name, 0.0)
        if not initialized:
            out = val
            initialized = True
        else:
            out = alpha * val + (1.0 - alpha) * out
    return float(out)


def rolling_drawdown(sequence: Sequence[RuntimeObservation], idx: int, window: int = 20) -> float:
    w = max(int(window), 1)
    start = max(0, idx - w + 1)
    prices = [observation_feature(sequence[j], "last_price", 0.0) for j in range(start, idx + 1)]
    prices = [p for p in prices if p > 0.0]
    if not prices:
        return 0.0
    peak = max(prices)
    curr = prices[-1]
    return (curr / max(peak, 1e-8)) - 1.0


def future_return(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> float:
    h = max(int(horizon), 1)
    curr_price = observation_feature(sequence[idx], "last_price", 0.0)
    fut_price = observation_feature(sequence[idx + h], "last_price", 0.0)
    if curr_price <= 0.0 or fut_price <= 0.0:
        return 0.0
    return (fut_price / max(curr_price, 1e-8)) - 1.0


def future_max_drawdown(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> float:
    h = max(int(horizon), 1)
    curr_price = observation_feature(sequence[idx], "last_price", 0.0)
    if curr_price <= 0.0:
        return 0.0
    worst = 0.0
    for j in range(idx + 1, idx + h + 1):
        price = observation_feature(sequence[j], "last_price", 0.0)
        if price <= 0.0:
            continue
        ret = (price / max(curr_price, 1e-8)) - 1.0
        worst = min(worst, ret)
    return worst


def future_realized_vol(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> float:
    h = max(int(horizon), 1)
    prices = [observation_feature(sequence[j], "last_price", 0.0) for j in range(idx, idx + h + 1)]
    prices = [p for p in prices if p > 0.0]
    if len(prices) < 3:
        return 0.0
    arr = np.asarray(prices, dtype=np.float64)
    rets = np.diff(np.log(np.maximum(arr, 1e-8)))
    if rets.size == 0:
        return 0.0
    return float(np.std(rets))


def symbol_role_features(symbol: str, role_map: Mapping[str, Sequence[str]]) -> Dict[str, float]:
    sym = str(symbol or "").strip().upper()
    out: Dict[str, float] = {}
    for role, symbols in role_map.items():
        token = str(role or "").strip().lower()
        key = f"role_{token}"
        out[key] = 1.0 if sym in {str(s).strip().upper() for s in symbols} else 0.0
    return out


def direction_label_builder(*, min_return: float = 0.0) -> RuntimeLabelBuilder:
    threshold = float(min_return)

    def _label(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> Optional[float]:
        ret = future_return(sequence, idx, horizon)
        return 1.0 if ret > threshold else 0.0

    return _label


def selective_direction_label_builder(*, min_abs_return: float = 0.0) -> RuntimeLabelBuilder:
    threshold = abs(float(min_abs_return))

    def _label(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> Optional[float]:
        ret = future_return(sequence, idx, horizon)
        if abs(ret) <= threshold:
            return None
        return 1.0 if ret > 0.0 else 0.0

    return _label


def multi_horizon_direction_label_builder(
    *,
    horizons: Sequence[int],
    min_return: float = 0.0,
) -> RuntimeLabelBuilder:
    threshold = abs(float(min_return))
    eval_horizons = sorted({max(int(h), 1) for h in horizons if int(h) > 0})

    def _label(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> Optional[float]:
        used_horizons = eval_horizons or [max(int(horizon), 1)]
        votes: List[float] = []
        for step_h in used_horizons:
            if (idx + step_h) >= len(sequence):
                return None
            ret = future_return(sequence, idx, step_h)
            if abs(ret) <= threshold:
                return None
            votes.append(1.0 if ret > 0.0 else 0.0)
        if not votes or len(set(votes)) != 1:
            return None
        return votes[0]

    return _label


def risk_support_label_builder(
    *,
    min_return: float = -0.001,
    max_drawdown: float = 0.015,
    max_realized_vol: float = 0.02,
    vol_multiplier: float = 3.0,
) -> RuntimeLabelBuilder:
    min_ret = float(min_return)
    max_dd = abs(float(max_drawdown))
    max_vol = abs(float(max_realized_vol))
    mult = max(float(vol_multiplier), 1.0)

    def _label(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> Optional[float]:
        fwd_ret = future_return(sequence, idx, horizon)
        dd = abs(future_max_drawdown(sequence, idx, horizon))
        realized = future_realized_vol(sequence, idx, horizon)
        curr_vol = abs(observation_feature(sequence[idx], "vol_30m", 0.0))
        allowed_vol = max(max_vol, curr_vol * mult)
        return 1.0 if (fwd_ret >= min_ret and dd <= max_dd and realized <= allowed_vol) else 0.0

    return _label


def load_runtime_observation_sequences(
    project_root: Path,
    *,
    lookback_days: int = 14,
    mode_allowlist: Optional[Sequence[str]] = None,
    symbol_allowlist: Optional[Sequence[str]] = None,
) -> RuntimeSequenceMap:
    root = Path(project_root).expanduser().resolve()
    since_utc = datetime.now(timezone.utc) - timedelta(days=max(int(lookback_days), 1))
    mode_allow = {str(x).strip().lower() for x in (mode_allowlist or []) if str(x).strip()}
    symbol_allow = {str(x).strip().upper() for x in (symbol_allowlist or []) if str(x).strip()}
    gap_fill_context = _load_runtime_gap_fill_context(root)

    best_by_snapshot: Dict[Tuple[str, str, str], RuntimeObservation] = {}
    for path in _recent_decision_paths(root, lookback_days=max(int(lookback_days), 1)):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                    if str(metadata.get("layer") or "").strip().lower() != "grand_master":
                        continue
                    strategy = str(row.get("strategy") or "").strip().lower()
                    if strategy not in _ROOT_STRATEGY_PRIORITY:
                        continue

                    ts = _parse_ts(row.get("timestamp_utc"))
                    if ts is None or ts < since_utc:
                        continue
                    mode = str(row.get("mode") or "").strip().lower()
                    symbol = str(row.get("symbol") or "").strip().upper()
                    if not mode or not symbol:
                        continue
                    if mode_allow and mode not in mode_allow:
                        continue
                    if symbol_allow and symbol not in symbol_allow:
                        continue

                    gates = row.get("gates") if isinstance(row.get("gates"), dict) else {}
                    if ("market_data_ok" in gates) and (not bool(gates.get("market_data_ok"))):
                        continue

                    features = row.get("features") if isinstance(row.get("features"), dict) else {}
                    price = _safe_float(features.get("last_price"), 0.0)
                    if price <= 0.0:
                        continue

                    snapshot_id = str(metadata.get("snapshot_id") or row.get("snapshot_id") or row.get("parent_decision_id") or "").strip()
                    if not snapshot_id:
                        snapshot_id = f"{symbol}:{ts.isoformat()}"

                    obs = {
                        "mode": mode,
                        "symbol": symbol,
                        "strategy": strategy,
                        "strategy_priority": _ROOT_STRATEGY_PRIORITY[strategy],
                        "snapshot_id": snapshot_id,
                        "ts_epoch": float(ts.timestamp()),
                        "timestamp_utc": ts.isoformat(),
                        "price": price,
                        "features": dict(features),
                    }
                    key = (mode, symbol, snapshot_id)
                    prev = best_by_snapshot.get(key)
                    if prev is None or int(obs["strategy_priority"]) < int(prev.get("strategy_priority", 99)):
                        best_by_snapshot[key] = obs
        except Exception:
            continue

    grouped: RuntimeSequenceMap = defaultdict(list)
    for obs in best_by_snapshot.values():
        grouped[(str(obs["mode"]), str(obs["symbol"]))].append(obs)

    out: RuntimeSequenceMap = {}
    for key, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda x: (float(x.get("ts_epoch", 0.0)), int(x.get("strategy_priority", 99)), str(x.get("snapshot_id") or "")))
        deduped: List[RuntimeObservation] = []
        seen_snapshot_ids: set[str] = set()
        carry_forward_features: Dict[str, float] = {}
        for row in rows_sorted:
            sid = str(row.get("snapshot_id") or "")
            if sid in seen_snapshot_ids:
                continue
            seen_snapshot_ids.add(sid)
            enriched = _enrich_runtime_observation(
                row,
                carry_forward_features=carry_forward_features,
                gap_fill_context=gap_fill_context,
            )
            deduped.append(enriched)
            next_carry: Dict[str, float] = {}
            feature_map = enriched.get("features") if isinstance(enriched.get("features"), Mapping) else {}
            for name in _RUNTIME_GAP_FILL_KEYS:
                if name not in feature_map:
                    continue
                try:
                    value = float(feature_map.get(name))
                except Exception:
                    continue
                if math.isfinite(value):
                    next_carry[name] = value
            carry_forward_features.update(next_carry)
        if deduped:
            out[key] = deduped
    return out


def make_runtime_windowed_dataset(
    *,
    sequences: RuntimeSequenceMap,
    feature_builder: RuntimeFeatureBuilder,
    label_builder: RuntimeLabelBuilder,
    sample_filter: Optional[RuntimeSampleFilter] = None,
    confidence_builder: Optional[RuntimeConfidenceBuilder] = None,
    min_confidence: float = 0.0,
    sample_stride: int = 1,
    window: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    w = max(int(window), 1)
    h = max(int(horizon), 1)
    min_conf = max(0.0, min(float(min_confidence), 1.0))
    stride = max(int(sample_stride), 1)

    samples: List[np.ndarray] = []
    labels: List[float] = []
    anchor_ts: List[float] = []
    sample_confidence: List[float] = []
    eligible_sequences = 0
    skipped_labels = 0
    skipped_filtered = 0
    skipped_low_confidence = 0
    feature_dim = 0

    for rows in sequences.values():
        if len(rows) < (w + h):
            continue
        eligible_sequences += 1
        for idx in range(w - 1, len(rows) - h, stride):
            if sample_filter is not None:
                try:
                    include_sample = bool(sample_filter(rows, idx, h))
                except Exception:
                    include_sample = False
                if not include_sample:
                    skipped_filtered += 1
                    continue

            confidence = 1.0
            if confidence_builder is not None:
                try:
                    confidence = _safe_float(confidence_builder(rows, idx, h), 0.0)
                except Exception:
                    confidence = 0.0
                confidence = min(max(confidence, 0.0), 1.0)
                if confidence < min_conf:
                    skipped_low_confidence += 1
                    continue

            per_step: List[np.ndarray] = []
            for step_idx in range(idx - w + 1, idx + 1):
                vec = np.asarray(feature_builder(rows, step_idx), dtype=np.float32).reshape(-1)
                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                if vec.size == 0:
                    per_step = []
                    break
                if feature_dim == 0:
                    feature_dim = int(vec.size)
                per_step.append(vec)
            if not per_step:
                continue

            label = label_builder(rows, idx, h)
            if label is None or (not math.isfinite(float(label))):
                skipped_labels += 1
                continue

            sample = np.concatenate(per_step, axis=0)
            samples.append(sample.astype(np.float32))
            labels.append(float(label))
            anchor_ts.append(float(rows[idx].get("ts_epoch", 0.0)))
            sample_confidence.append(float(confidence))

    if not samples:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 1), dtype=np.float32), {
            "sequence_count": len(sequences),
            "eligible_sequences": eligible_sequences,
            "sample_count": 0,
            "feature_dim": 0,
            "window": w,
            "horizon": h,
            "sample_stride": stride,
            "positive_rate": 0.0,
            "skipped_labels": skipped_labels,
            "skipped_filtered": skipped_filtered,
            "skipped_low_confidence": skipped_low_confidence,
            "confidence_mean": 0.0,
            "confidence_min": 0.0,
            "confidence_max": 0.0,
            "min_confidence": float(min_conf),
            "_sample_confidence": np.zeros((0,), dtype=np.float32),
        }

    order = np.argsort(np.asarray(anchor_ts, dtype=np.float64))
    X = np.asarray([samples[i] for i in order], dtype=np.float32)
    y = np.asarray([[labels[i]] for i in order], dtype=np.float32)
    conf = np.asarray([sample_confidence[i] for i in order], dtype=np.float32)
    anchor_ordered = np.asarray([anchor_ts[i] for i in order], dtype=np.float64)
    X, y, conf, balance_meta = _rebalance_binary_runtime_dataset(
        X,
        y,
        conf,
        anchor_ordered,
    )
    positive_rate = float(np.mean(y[:, 0])) if y.size else 0.0
    return X, y, {
        "sequence_count": len(sequences),
        "eligible_sequences": eligible_sequences,
        "sample_count": int(X.shape[0]),
        "feature_dim": int(feature_dim),
        "window": w,
        "horizon": h,
        "sample_stride": stride,
        "positive_rate": positive_rate,
        "skipped_labels": skipped_labels,
        "skipped_filtered": skipped_filtered,
        "skipped_low_confidence": skipped_low_confidence,
        "confidence_mean": float(np.mean(conf)) if conf.size else 0.0,
        "confidence_min": float(np.min(conf)) if conf.size else 0.0,
        "confidence_max": float(np.max(conf)) if conf.size else 0.0,
        "min_confidence": float(min_conf),
        "_sample_confidence": conf,
        **balance_meta,
    }


def _rebalance_binary_runtime_dataset(
    X: np.ndarray,
    y: np.ndarray,
    conf: np.ndarray,
    anchor_ts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    labels = np.asarray(y[:, 0], dtype=np.float32).reshape(-1) if y.ndim == 2 else np.asarray([], dtype=np.float32)
    total_samples = int(labels.size)
    positive_count = int(np.sum(labels >= 0.5))
    negative_count = int(total_samples - positive_count)
    base_meta = {
        "label_balance_applied": False,
        "label_balance_reason": "not_needed",
        "label_balance_original_sample_count": total_samples,
        "label_balance_original_positive_rate": float(np.mean(labels)) if labels.size else 0.0,
        "label_balance_kept_positive": positive_count,
        "label_balance_kept_negative": negative_count,
        "label_balance_max_ratio": float(
            max(_safe_float(os.getenv("RUNTIME_TRAIN_LABEL_BALANCE_MAX_RATIO", _DEFAULT_RUNTIME_LABEL_BALANCE_MAX_RATIO), _DEFAULT_RUNTIME_LABEL_BALANCE_MAX_RATIO), 1.0)
        ),
    }
    if total_samples == 0 or positive_count == 0 or negative_count == 0:
        base_meta["label_balance_reason"] = "single_class"
        return X, y, conf, base_meta

    min_total_samples = max(
        int(_safe_float(os.getenv("RUNTIME_TRAIN_LABEL_BALANCE_MIN_TOTAL_SAMPLES", _DEFAULT_RUNTIME_LABEL_BALANCE_MIN_TOTAL_SAMPLES), _DEFAULT_RUNTIME_LABEL_BALANCE_MIN_TOTAL_SAMPLES)),
        1,
    )
    min_minority_samples = max(
        int(_safe_float(os.getenv("RUNTIME_TRAIN_LABEL_BALANCE_MIN_MINORITY_SAMPLES", _DEFAULT_RUNTIME_LABEL_BALANCE_MIN_MINORITY_SAMPLES), _DEFAULT_RUNTIME_LABEL_BALANCE_MIN_MINORITY_SAMPLES)),
        1,
    )
    max_ratio = float(base_meta["label_balance_max_ratio"])
    if total_samples < min_total_samples:
        base_meta["label_balance_reason"] = "sample_count_below_floor"
        return X, y, conf, base_meta

    if positive_count >= negative_count:
        majority_idx = np.flatnonzero(labels >= 0.5)
        minority_idx = np.flatnonzero(labels < 0.5)
        majority_label = "positive"
    else:
        majority_idx = np.flatnonzero(labels < 0.5)
        minority_idx = np.flatnonzero(labels >= 0.5)
        majority_label = "negative"

    if minority_idx.size < min_minority_samples:
        base_meta["label_balance_reason"] = "minority_below_floor"
        return X, y, conf, base_meta

    if majority_idx.size <= int(math.ceil(minority_idx.size * max_ratio)):
        base_meta["label_balance_reason"] = "already_within_ratio"
        return X, y, conf, base_meta

    target_majority = min(int(math.ceil(minority_idx.size * max_ratio)), int(majority_idx.size))
    majority_by_time = majority_idx[np.argsort(anchor_ts[majority_idx], kind="stable")]
    anchor_positions = np.unique(np.linspace(0, max(majority_by_time.size - 1, 0), num=target_majority, dtype=np.int64))
    majority_keep = majority_by_time[anchor_positions]
    if majority_keep.size < target_majority:
        remaining_needed = int(target_majority - majority_keep.size)
        majority_set = {int(i) for i in majority_keep.tolist()}
        extras_ranked = sorted(
            [int(i) for i in majority_idx.tolist() if int(i) not in majority_set],
            key=lambda idx: (-float(conf[idx]), -float(anchor_ts[idx])),
        )
        if remaining_needed > 0 and extras_ranked:
            majority_keep = np.concatenate([majority_keep, np.asarray(extras_ranked[:remaining_needed], dtype=np.int64)])

    selected_idx = np.sort(np.concatenate([minority_idx, majority_keep]))
    X_out = np.asarray(X[selected_idx], dtype=np.float32)
    y_out = np.asarray(y[selected_idx], dtype=np.float32)
    conf_out = np.asarray(conf[selected_idx], dtype=np.float32)
    labels_out = np.asarray(y_out[:, 0], dtype=np.float32)
    base_meta.update(
        {
            "label_balance_applied": True,
            "label_balance_reason": f"downsampled_{majority_label}",
            "label_balance_kept_positive": int(np.sum(labels_out >= 0.5)),
            "label_balance_kept_negative": int(np.sum(labels_out < 0.5)),
            "label_balance_rebalanced_sample_count": int(labels_out.size),
            "label_balance_rebalanced_positive_rate": float(np.mean(labels_out)) if labels_out.size else 0.0,
        }
    )
    return X_out, y_out, conf_out, base_meta
