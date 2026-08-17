from __future__ import annotations

import glob
import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sql_dataset_io import (
    iter_sqlite_jsonl_rows,
    iter_sqlite_jsonl_rows_by_like_patterns,
    resolve_sqlite_path,
    split_paths_by_sqlite_coverage,
)

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
from central_bank_liquidity import (
    CENTRAL_BANK_LIQUIDITY_FEATURE_KEYS,
    central_bank_liquidity_context_ready,
)
from global_central_bank_context import (
    CENTRAL_BANK_CROSS_SOURCE_FEATURE_KEYS,
    GLOBAL_CENTRAL_BANK_FEATURE_KEYS,
    central_bank_cross_source_context_ready,
    global_central_bank_context_ready,
)
from decision_context_mesh import (
    DECISION_CONTEXT_MESH_FEATURE_KEYS,
    decision_context_mesh_ready,
)

try:
    from derivatives_features import summarize_calendar_payload
except Exception:
    summarize_calendar_payload = None

try:
    from advanced_quant_models import QUANT_MODEL_FEATURE_KEYS
except Exception:
    QUANT_MODEL_FEATURE_KEYS = ()


RuntimeObservation = Dict[str, Any]
RuntimeSequenceMap = Dict[Tuple[str, str], List[RuntimeObservation]]
RuntimeFeatureBuilder = Callable[[Sequence[RuntimeObservation], int], np.ndarray]
RuntimeLabelBuilder = Callable[[Sequence[RuntimeObservation], int, int], Optional[float]]
RuntimeSampleFilter = Callable[[Sequence[RuntimeObservation], int, int], bool]
RuntimeConfidenceBuilder = Callable[[Sequence[RuntimeObservation], int, int], float]

_DEFAULT_RUNTIME_LABEL_BALANCE_MAX_RATIO = 4.0
_DEFAULT_RUNTIME_LABEL_BALANCE_MIN_MINORITY_SAMPLES = 6
_DEFAULT_RUNTIME_LABEL_BALANCE_MIN_TOTAL_SAMPLES = 64
_DEFAULT_RUNTIME_SNAPSHOT_HEALTH = "governance/health/runtime_training_snapshot_latest.json"
_DEFAULT_HDF5_CACHE_HEALTH = "governance/health/hdf5_training_cache_latest.json"
_DEFAULT_SQL_PROGRESS_HEALTH = "governance/health/sql_link_service_progress_latest.json"
_HDF5_CACHE_SCHEMA_VERSION = 2

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
    "calendar_opex_week_norm",
    "calendar_month_end_rebalance_norm",
    "calendar_quarter_end_rebalance_norm",
    "calendar_futures_roll_window_norm",
    "calendar_index_rebalance_window_norm",
}

_RUNTIME_MARKET_MICRO_KEYS = {
    "market_micro_premarket_pressure_norm",
    "market_micro_opening_auction_norm",
    "market_micro_opening_auction_imbalance_norm",
    "market_micro_opening_drive_pressure_norm",
    "market_micro_power_hour_pressure_norm",
    "market_micro_closing_auction_norm",
    "market_micro_closing_auction_imbalance_norm",
    "market_micro_closing_cross_pressure_norm",
    "market_micro_auction_print_pressure_norm",
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
    "market_micro_trade_halt_norm",
    "market_micro_luld_pause_norm",
    "market_micro_ssr_active_norm",
    "market_micro_resume_window_norm",
    "market_micro_dark_pool_pressure_norm",
    "market_micro_off_exchange_share_norm",
    "market_micro_spread_regime_norm",
    "market_micro_spread_widening_norm",
    "market_micro_queue_depth_decay_norm",
    "market_micro_depth_collapse_norm",
    "market_micro_quote_fade_rate_norm",
    "market_micro_tradeability_score_norm",
    "market_micro_session_open_norm",
    "market_micro_session_midday_norm",
    "market_micro_session_power_hour_norm",
    "market_micro_overnight_gap_norm",
    "market_micro_post_event_drift_norm",
    "market_micro_lunch_chop_norm",
    "market_micro_open_close_imbalance_regime_norm",
    "market_micro_symbol_cooldown_pressure_norm",
    "market_micro_gap_fade_risk_norm",
    "market_micro_overnight_event_hazard_norm",
    "etf_nav_premium_discount_norm",
    "etf_creation_redemption_stress_norm",
    "etf_primary_secondary_liquidity_norm",
    "etf_underlying_basket_stress_norm",
    "etf_fund_family_flow_norm",
    "etf_fund_family_creation_pressure_norm",
}

_RUNTIME_SEC_EDGAR_KEYS = {
    "sec_filing_count_7d_norm",
    "sec_high_impact_7d_norm",
    "sec_earnings_7d_norm",
    "sec_guidance_7d_norm",
    "sec_regulatory_7d_norm",
    "sec_offering_7d_norm",
    "sec_dilution_7d_norm",
    "sec_mna_7d_norm",
    "sec_restatement_7d_norm",
    "sec_financing_stress_7d_norm",
    "sec_ownership_30d_norm",
    "sec_insider_30d_norm",
    "sec_insider_buy_30d_norm",
    "sec_insider_sell_30d_norm",
    "sec_estimate_revision_drift_norm",
    "sec_earnings_whisper_surprise_norm",
    "sec_split_hazard_30d_norm",
    "sec_special_dividend_30d_norm",
    "sec_offering_priced_30d_norm",
    "sec_lockup_secondary_30d_norm",
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
    "calendar_opex_week_norm",
    "calendar_month_end_rebalance_norm",
    "calendar_quarter_end_rebalance_norm",
    "calendar_futures_roll_window_norm",
    "calendar_index_rebalance_window_norm",
}

_RUNTIME_CENTRAL_BANK_LIQUIDITY_KEYS = set(CENTRAL_BANK_LIQUIDITY_FEATURE_KEYS)
_RUNTIME_GLOBAL_CENTRAL_BANK_KEYS = set(GLOBAL_CENTRAL_BANK_FEATURE_KEYS)
_RUNTIME_CENTRAL_BANK_CROSS_SOURCE_KEYS = set(CENTRAL_BANK_CROSS_SOURCE_FEATURE_KEYS)
_RUNTIME_DECISION_CONTEXT_MESH_KEYS = set(DECISION_CONTEXT_MESH_FEATURE_KEYS)

_RUNTIME_TASTYTRADE_KEYS = {
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
    "options_vanna_mean_norm",
    "options_charm_abs_mean_norm",
    "options_vomma_mean_norm",
    "options_speed_abs_mean_norm",
    "options_color_abs_mean_norm",
    "options_zomma_abs_mean_norm",
    "options_ultima_abs_mean_norm",
    "options_higher_order_greek_pressure_norm",
    "options_barrier_touch_risk_norm",
    "options_lookback_path_dependency_norm",
    "options_variance_swap_proxy_norm",
    "options_volatility_swap_proxy_norm",
    "options_gamma_scalping_pressure_norm",
    "options_vanna_volga_hedge_pressure_norm",
    "options_dispersion_trade_proxy_norm",
    "options_volatility_arbitrage_proxy_norm",
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
    "market_crypto_sleeve_coverage_norm",
    "market_crypto_sleeve_avg_abs_corr_norm",
    "market_crypto_sleeve_dispersion_norm",
    "market_crypto_sleeve_confidence_norm",
    "market_crypto_risk_on_crypto_alignment_norm",
    "market_crypto_fx_crypto_inverse_corr_norm",
    "market_crypto_rates_crypto_corr_norm",
    "market_crypto_energy_crypto_corr_norm",
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
    "fx_session_asia_norm",
    "fx_session_london_norm",
    "fx_session_ny_norm",
    "fx_rollover_risk_norm",
    "fx_dxy_yield_confirmation_norm",
    "fx_carry_proxy_norm",
}

_RUNTIME_DIVIDEND_DRIP_KEYS = {
    "dividend_drip_active_norm",
    "dividend_drip_recent_reinvest_norm",
    "dividend_drip_cash_only_norm",
    "dividend_drip_share_credit_norm",
    "dividend_drip_event_recency_norm",
    "dividend_drip_confidence_norm",
}

_RUNTIME_QUANT_MODEL_KEYS = set(QUANT_MODEL_FEATURE_KEYS)

_RUNTIME_SCHWAB_EDUCATION_KEYS = {
    "schwab_education_item_density_norm",
    "schwab_education_recent_activity_norm",
    "schwab_education_symbol_coverage_norm",
    "schwab_education_video_share_norm",
    "schwab_education_stream_share_norm",
    "schwab_education_network_share_norm",
    "schwab_education_symbol_frequency_norm",
    "schwab_education_symbol_recency_norm",
    "schwab_education_symbol_stream_share_norm",
}

_RUNTIME_GAP_FILL_KEYS = set(BREADTH_FEATURE_KEYS) | set(BOND_REFERENCE_FEATURE_KEYS) | set(CREDIT_CONTEXT_FEATURE_KEYS) | set(NEWS_STRUCTURED_FEATURE_KEYS) | _RUNTIME_NEWS_EVENT_KEYS | _RUNTIME_CALENDAR_EVENT_KEYS | _RUNTIME_MARKET_MICRO_KEYS | _RUNTIME_SEC_EDGAR_KEYS | _RUNTIME_EXTENDED_QUANT_KEYS | _RUNTIME_CENTRAL_BANK_LIQUIDITY_KEYS | _RUNTIME_GLOBAL_CENTRAL_BANK_KEYS | _RUNTIME_CENTRAL_BANK_CROSS_SOURCE_KEYS | _RUNTIME_DECISION_CONTEXT_MESH_KEYS | _RUNTIME_TASTYTRADE_KEYS | _RUNTIME_CRYPTO_MARKET_KEYS | _RUNTIME_MARKET_CRYPTO_CORRELATION_KEYS | _RUNTIME_FX_MARKET_KEYS | _RUNTIME_DIVIDEND_DRIP_KEYS | _RUNTIME_SCHWAB_EDUCATION_KEYS | _RUNTIME_QUANT_MODEL_KEYS


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    out = _safe_float(value, float(default))
    try:
        return int(out)
    except Exception:
        return int(default)


def _runtime_row_metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = row.get("metadata") if isinstance(row, Mapping) else {}
    if isinstance(metadata, Mapping):
        return metadata
    if str(row.get("channel") or "").strip().lower() == "decision" and isinstance(row.get("grand_master_meta"), Mapping):
        return {
            "layer": "grand_master",
            "strategy": "grand_master_bot",
            "mode": _runtime_row_mode(row, {}),
            "snapshot_id": row.get("snapshot_id"),
        }
    return {}


def _runtime_row_strategy(row: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> str:
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if str(row.get("channel") or "").strip().lower() == "decision" and isinstance(row.get("grand_master_meta"), Mapping):
        return "grand_master_bot"
    return str(row.get("strategy") or metadata.get("strategy") or "").strip().lower()


def _runtime_strategy_priority(strategy: Any, metadata: Mapping[str, Any] | None = None) -> Optional[int]:
    strategy_key = str(strategy or "").strip().lower()
    metadata = metadata if isinstance(metadata, Mapping) else {}
    layer = str(metadata.get("layer") or "").strip().lower()
    if strategy_key in _ROOT_STRATEGY_PRIORITY:
        return int(_ROOT_STRATEGY_PRIORITY[strategy_key])
    if layer == "grand_master":
        return 0
    if "master" in layer:
        return 20
    if strategy_key:
        return 50
    return None


def _runtime_row_mode(row: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> str:
    metadata = metadata if isinstance(metadata, Mapping) else {}
    explicit = str(row.get("mode") or metadata.get("mode") or "").strip().lower()
    profile = str(row.get("shadow_profile") or metadata.get("source_profile") or "").strip().lower()
    shadow_domain = str(metadata.get("shadow_domain") or "").strip().lower()
    if explicit == "shadow" and shadow_domain == "crypto":
        if "future" in profile:
            return "shadow_crypto_futures_crypto"
        return "shadow_crypto"
    if explicit:
        return explicit
    broker = str(row.get("broker") or "").strip().lower()
    if broker == "coinbase":
        if "future" in profile:
            return "shadow_crypto_futures_crypto"
        return "shadow_crypto"
    return ""


_CHANNEL_FEATURE_MAP_KEYS = (
    "market",
    "data_quality_features",
    "flow_awareness_features",
    "lead_lag_features",
    "breadth_features",
    "calendar_features",
    "credit_context_features",
    "bond_reference_features",
    "market_micro_features",
    "news_features",
    "futures_features",
    "options_chain_features",
    "dividend_features",
    "external_context_features",
    "long_term_features",
    "quant_model_features",
    "lane_strategy_features",
    "execution_lag_features",
    "allocation_confidence",
    "capital_flow",
)


def _set_missing_numeric_feature(features: Dict[str, Any], key: str, value: Any) -> None:
    if key in features:
        return
    try:
        numeric = float(value)
    except Exception:
        return
    if math.isfinite(numeric):
        features[key] = numeric


def _derive_channel_features(row: Mapping[str, Any], features: Dict[str, Any]) -> None:
    master_vote = _safe_float(row.get("master_vote"), _safe_float(row.get("master_score"), 0.5) - 0.5)
    _set_missing_numeric_feature(features, "behavior_prior", max(min(master_vote, 1.0), -1.0))

    queue_depth_norm = _safe_float(features.get("queue_depth_norm"), _safe_float(features.get("queue_depth"), 0.0) / 4.0)
    spread_norm = _safe_float(features.get("futures_spread_bps_norm"), _safe_float(features.get("spread_bps"), 0.0) / 80.0)
    liquidity = max(0.0, min(1.0, (0.72 * queue_depth_norm) + (0.28 * (1.0 - min(max(spread_norm, 0.0), 1.0)))))
    _set_missing_numeric_feature(features, "crypto_exchange_liquidity_norm", liquidity)

    symbol = str(row.get("symbol") or "").strip().upper()
    if symbol.startswith("SOL"):
        eth_strength = _safe_float(features.get("crypto_eth_btc_relative_strength_norm"), 0.5)
        sol_strength = max(0.0, min(1.0, 0.5 + (_safe_float(features.get("mom_15m"), 0.0) * 120.0) + (_safe_float(features.get("mom_5m"), 0.0) * 80.0) - ((eth_strength - 0.5) * 0.25)))
        _set_missing_numeric_feature(features, "crypto_solana_relative_strength_norm", sol_strength)
    if symbol.startswith("ETH"):
        eth_strength = max(0.0, min(1.0, 0.5 + (_safe_float(features.get("mom_15m"), 0.0) * 90.0) + (_safe_float(features.get("mom_5m"), 0.0) * 70.0)))
        _set_missing_numeric_feature(features, "crypto_eth_btc_relative_strength_norm", eth_strength)

    divergence = max(
        0.0,
        min(
            1.0,
            abs(1.0 - _safe_float(features.get("crypto_cross_provider_price_agreement_norm"), 1.0))
            + (0.35 * min(max(spread_norm, 0.0), 1.0))
            + (0.20 * abs(_safe_float(features.get("futures_order_book_imbalance_norm"), 0.5) - 0.5)),
        ),
    )
    _set_missing_numeric_feature(features, "crypto_exchange_price_divergence_norm", divergence)
    volume_dispersion = max(0.0, min(1.0, abs(_safe_float(features.get("volume_zscore"), 0.0)) / 3.0))
    _set_missing_numeric_feature(features, "crypto_exchange_volume_dispersion_norm", volume_dispersion)

    now_utc = _parse_ts(row.get("timestamp_utc"))
    weekend = 1.0 if now_utc is not None and now_utc.weekday() >= 5 else 0.0
    _set_missing_numeric_feature(features, "crypto_weekend_session_norm", weekend)
    weekend_gap = max(0.0, min(1.0, abs(_safe_float(features.get("pct_from_close"), 0.0)) / 0.025))
    _set_missing_numeric_feature(features, "crypto_weekend_gap_norm", weekend_gap if weekend > 0 else weekend_gap * 0.25)

    _set_missing_numeric_feature(features, "crypto_liquidation_pressure_norm", _safe_float(features.get("futures_liquidation_risk_norm"), 0.0))
    _set_missing_numeric_feature(features, "infra_risk_throttle_norm", _safe_float(features.get("risk_throttle_norm"), _safe_float(features.get("quant_model_resource_pressure_norm"), 0.0)))
    _set_missing_numeric_feature(features, "fx_dollar_funding_stress_norm", _safe_float(features.get("fx_dollar_pressure_norm"), 0.0))
    _set_missing_numeric_feature(features, "macro_event_proximity_norm", _safe_float(features.get("calendar_event_proximity_norm"), 0.0))


def _runtime_row_features(row: Mapping[str, Any]) -> Dict[str, Any]:
    features = row.get("features") if isinstance(row, Mapping) else {}
    out = dict(features) if isinstance(features, Mapping) else {}
    if str(row.get("channel") or "").strip().lower() == "decision":
        for key in _CHANNEL_FEATURE_MAP_KEYS:
            values = row.get(key)
            if isinstance(values, Mapping):
                for feature_key, feature_value in values.items():
                    out.setdefault(str(feature_key), feature_value)
        _derive_channel_features(row, out)
    return out


def _runtime_snapshot_id_candidates(row: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> List[str]:
    metadata = metadata if isinstance(metadata, Mapping) else {}
    candidates = [
        metadata.get("snapshot_id"),
        row.get("snapshot_id"),
        row.get("parent_decision_id"),
        row.get("decision_id"),
        row.get("iter_id"),
        row.get("run_id"),
    ]
    out: List[str] = []
    seen: set[str] = set()
    for raw in candidates:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _runtime_row_price(row: Mapping[str, Any], features: Mapping[str, Any] | None = None) -> float:
    features = features if isinstance(features, Mapping) else {}
    market = row.get("market") if isinstance(row, Mapping) else {}
    quote = row.get("quote") if isinstance(row, Mapping) else {}
    for raw in (
        row.get("price"),
        row.get("last_price"),
        features.get("last_price"),
        features.get("price"),
        market.get("last_price") if isinstance(market, Mapping) else None,
        market.get("price") if isinstance(market, Mapping) else None,
        quote.get("last_price") if isinstance(quote, Mapping) else None,
        quote.get("price") if isinstance(quote, Mapping) else None,
    ):
        price = _safe_float(raw, 0.0)
        if price > 0.0:
            return price
    return 0.0


def _iter_runtime_price_sidecar_rows(paths: Sequence[Path], *, max_rows: int = 0) -> Iterable[Dict[str, Any]]:
    yielded = 0
    for raw_path in paths:
        path = Path(raw_path)
        try:
            handle_cm = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")
            with handle_cm as handle:
                for line in handle:
                    if max_rows > 0 and yielded >= max_rows:
                        return
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        yielded += 1
                        yield row
        except Exception:
            continue


def _build_runtime_price_sidecar_from_rows(rows: Iterable[Mapping[str, Any]], *, max_rows: int = 0) -> Dict[str, Any]:
    by_snapshot: Dict[str, Dict[str, Any]] = {}
    by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    consumed = 0
    for row in rows:
        if max_rows > 0 and consumed >= max_rows:
            break
        if not isinstance(row, Mapping):
            continue
        consumed += 1
        metadata = _runtime_row_metadata(row)
        symbol = str(row.get("symbol") or "").strip().upper()
        ts = _parse_ts(row.get("timestamp_utc"))
        snapshot_ids = _runtime_snapshot_id_candidates(row, metadata)
        features = _runtime_row_features(row)
        price = _runtime_row_price(row, features)
        if price <= 0.0 or not symbol:
            continue
        entry = {
            "symbol": symbol,
            "timestamp_utc": ts.isoformat() if ts is not None else "",
            "ts_epoch": float(ts.timestamp()) if ts is not None else 0.0,
            "price": price,
            "features": features,
        }
        for snapshot_id in snapshot_ids:
            by_snapshot[snapshot_id] = entry
        by_symbol[symbol].append(entry)
    for entries in by_symbol.values():
        entries.sort(key=lambda row: float(row.get("ts_epoch", 0.0)))
    return {
        "by_snapshot": by_snapshot,
        "by_symbol": dict(by_symbol),
        "row_count": consumed,
    }


def _lookup_runtime_sidecar_context(
    sidecar: Mapping[str, Any],
    *,
    symbol: str,
    snapshot_ids: Sequence[str],
    ts: datetime | None = None,
) -> Dict[str, Any]:
    by_snapshot = sidecar.get("by_snapshot") if isinstance(sidecar, Mapping) else {}
    if isinstance(by_snapshot, Mapping):
        for snapshot_id in snapshot_ids:
            entry = by_snapshot.get(str(snapshot_id))
            if isinstance(entry, Mapping):
                return dict(entry)
    by_symbol = sidecar.get("by_symbol") if isinstance(sidecar, Mapping) else {}
    entries = by_symbol.get(str(symbol or "").strip().upper()) if isinstance(by_symbol, Mapping) else None
    if not isinstance(entries, Sequence):
        return {}
    candidates = [entry for entry in entries if isinstance(entry, Mapping)]
    if not candidates:
        return {}
    if ts is None:
        return dict(candidates[-1])
    target = float(ts.timestamp())
    return dict(min(candidates, key=lambda row: abs(_safe_float(row.get("ts_epoch"), target) - target)))


def _runtime_sidecar_entry_price(entry: Mapping[str, Any] | None) -> float:
    if not isinstance(entry, Mapping):
        return 0.0
    features = entry.get("features") if isinstance(entry.get("features"), Mapping) else {}
    return _runtime_row_price(entry, features)


def _runtime_features_with_sidecar_context(features: Mapping[str, Any], entry: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(features) if isinstance(features, Mapping) else {}
    sidecar_features = entry.get("features") if isinstance(entry, Mapping) and isinstance(entry.get("features"), Mapping) else {}
    for key, value in sidecar_features.items():
        merged.setdefault(str(key), value)
    sidecar_price = _runtime_sidecar_entry_price(entry)
    if sidecar_price > 0.0 and _safe_float(merged.get("last_price"), 0.0) <= 0.0:
        merged["last_price"] = sidecar_price
        merged["price_recovered_from_sidecar"] = 1.0
    return merged


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
    name = path.name
    for suffix in (".jsonl.gz", ".jsonl", ".log.gz", ".log"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    parts = name.rsplit("_", 1)
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
    central_bank_cross_source = load_latest_external_context(project_root, "central_bank_cross_source")
    decision_context_mesh = load_latest_external_context(project_root, "decision_context_mesh")
    schwab_education = load_latest_external_context(project_root, "schwab_education_context")
    market_micro = load_latest_external_context(project_root, "market_micro")
    sec_edgar = load_latest_external_context(project_root, "sec_edgar")
    extended_quant = load_latest_external_context(project_root, "extended_quant_context")
    options_flow = load_latest_external_context(project_root, "options_flow_context")
    crypto_market = load_latest_external_context(project_root, "crypto_market_context")
    market_crypto_correlation = load_latest_external_context(project_root, "market_crypto_correlation")
    fx_market_context = load_latest_external_context(project_root, "fx_market_context")
    dividend_drip_state = load_latest_external_context(project_root, "dividend_drip_state")
    quant_model_control = load_latest_external_context(project_root, "quant_model_control")

    te_derived = tradingeconomics.get("derived") if isinstance(tradingeconomics.get("derived"), Mapping) else {}
    official_derived = official_macro.get("derived") if isinstance(official_macro.get("derived"), Mapping) else {}
    central_bank_cross_derived = central_bank_cross_source.get("derived") if isinstance(central_bank_cross_source.get("derived"), Mapping) else {}
    decision_context_mesh_derived = decision_context_mesh.get("derived") if isinstance(decision_context_mesh.get("derived"), Mapping) else {}
    schwab_derived = schwab_education.get("derived") if isinstance(schwab_education.get("derived"), Mapping) else {}
    sec_derived = sec_edgar.get("derived") if isinstance(sec_edgar.get("derived"), Mapping) else {}
    extended_derived = extended_quant.get("derived") if isinstance(extended_quant.get("derived"), Mapping) else {}
    options_flow_derived = options_flow.get("derived") if isinstance(options_flow.get("derived"), Mapping) else {}
    crypto_derived = crypto_market.get("derived") if isinstance(crypto_market.get("derived"), Mapping) else {}
    market_crypto_corr_derived = market_crypto_correlation.get("derived") if isinstance(market_crypto_correlation.get("derived"), Mapping) else {}
    fx_market_derived = fx_market_context.get("derived") if isinstance(fx_market_context.get("derived"), Mapping) else {}
    dividend_drip_derived = dividend_drip_state.get("derived") if isinstance(dividend_drip_state.get("derived"), Mapping) else {}
    quant_model_derived = quant_model_control.get("derived") if isinstance(quant_model_control.get("derived"), Mapping) else {}
    te_calendar = te_derived.get("calendar_features") if isinstance(te_derived.get("calendar_features"), Mapping) else {}
    te_news = te_derived.get("news_features") if isinstance(te_derived.get("news_features"), Mapping) else {}
    te_calendar_rows = te_derived.get("calendar_rows") if isinstance(te_derived.get("calendar_rows"), list) else []
    official_calendar = official_derived.get("calendar_features") if isinstance(official_derived.get("calendar_features"), Mapping) else {}
    official_news = official_derived.get("news_features") if isinstance(official_derived.get("news_features"), Mapping) else {}
    official_global = official_derived.get("global_features") if isinstance(official_derived.get("global_features"), Mapping) else {}
    central_bank_cross_global = central_bank_cross_derived.get("global_features") if isinstance(central_bank_cross_derived.get("global_features"), Mapping) else {}
    central_bank_cross_symbol = central_bank_cross_derived.get("symbol_features") if isinstance(central_bank_cross_derived.get("symbol_features"), Mapping) else {}
    decision_context_mesh_global = decision_context_mesh_derived.get("global_features") if isinstance(decision_context_mesh_derived.get("global_features"), Mapping) else {}
    decision_context_mesh_symbol = decision_context_mesh_derived.get("symbol_features") if isinstance(decision_context_mesh_derived.get("symbol_features"), Mapping) else {}
    schwab_news = schwab_derived.get("news_features") if isinstance(schwab_derived.get("news_features"), Mapping) else {}
    schwab_global = schwab_derived.get("global_features") if isinstance(schwab_derived.get("global_features"), Mapping) else {}
    schwab_symbol = schwab_derived.get("symbol_features") if isinstance(schwab_derived.get("symbol_features"), Mapping) else {}
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
    options_flow_global = options_flow_derived.get("global_features") if isinstance(options_flow_derived.get("global_features"), Mapping) else {}
    options_flow_symbol = options_flow_derived.get("symbol_features") if isinstance(options_flow_derived.get("symbol_features"), Mapping) else {}
    crypto_news = crypto_derived.get("news_features") if isinstance(crypto_derived.get("news_features"), Mapping) else {}
    crypto_global = crypto_derived.get("global_features") if isinstance(crypto_derived.get("global_features"), Mapping) else {}
    crypto_symbol = crypto_derived.get("symbol_features") if isinstance(crypto_derived.get("symbol_features"), Mapping) else {}
    market_crypto_corr_global = market_crypto_corr_derived.get("global_features") if isinstance(market_crypto_corr_derived.get("global_features"), Mapping) else {}
    market_crypto_corr_symbol = market_crypto_corr_derived.get("symbol_features") if isinstance(market_crypto_corr_derived.get("symbol_features"), Mapping) else {}
    fx_market_global = fx_market_derived.get("global_features") if isinstance(fx_market_derived.get("global_features"), Mapping) else {}
    fx_market_symbol = fx_market_derived.get("symbol_features") if isinstance(fx_market_derived.get("symbol_features"), Mapping) else {}
    dividend_drip_global = dividend_drip_derived.get("global_features") if isinstance(dividend_drip_derived.get("global_features"), Mapping) else {}
    dividend_drip_symbol = dividend_drip_derived.get("symbol_features") if isinstance(dividend_drip_derived.get("symbol_features"), Mapping) else {}
    quant_model_global = quant_model_derived.get("global_features") if isinstance(quant_model_derived.get("global_features"), Mapping) else {}
    quant_model_symbol = quant_model_derived.get("symbol_features") if isinstance(quant_model_derived.get("symbol_features"), Mapping) else {}

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
    for extra_news in (schwab_news, sec_news, extended_news):
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
    market_micro_symbol = market_micro_derived.get("symbol_features") if isinstance(market_micro_derived.get("symbol_features"), Mapping) else {}
    for key, value in _feature_subset(market_micro_global, _RUNTIME_MARKET_MICRO_KEYS).items():
        market_micro_features[key] = value
    external_global_features = {}
    if central_bank_liquidity_context_ready(official_macro):
        external_global_features.update(_feature_subset(official_global, _RUNTIME_CENTRAL_BANK_LIQUIDITY_KEYS))
    if global_central_bank_context_ready(official_macro):
        external_global_features.update(_feature_subset(official_global, _RUNTIME_GLOBAL_CENTRAL_BANK_KEYS))
    if central_bank_cross_source_context_ready(central_bank_cross_source):
        external_global_features.update(
            _feature_subset(central_bank_cross_global, _RUNTIME_CENTRAL_BANK_CROSS_SOURCE_KEYS)
        )
    if decision_context_mesh_ready(decision_context_mesh):
        external_global_features.update(
            _feature_subset(decision_context_mesh_global, _RUNTIME_DECISION_CONTEXT_MESH_KEYS)
        )
    external_global_features.update(_feature_subset(schwab_global, _RUNTIME_SCHWAB_EDUCATION_KEYS))
    external_global_features.update(_feature_subset(sec_global, _RUNTIME_SEC_EDGAR_KEYS))
    external_global_features.update(_feature_subset(extended_global, _RUNTIME_EXTENDED_QUANT_KEYS))
    external_global_features.update(_feature_subset(options_flow_global, _RUNTIME_TASTYTRADE_KEYS))
    external_global_features.update(_feature_subset(crypto_global, _RUNTIME_CRYPTO_MARKET_KEYS))
    external_global_features.update(_feature_subset(market_crypto_corr_global, _RUNTIME_MARKET_CRYPTO_CORRELATION_KEYS))
    external_global_features.update(_feature_subset(fx_market_global, _RUNTIME_FX_MARKET_KEYS))
    external_global_features.update(_feature_subset(dividend_drip_global, _RUNTIME_DIVIDEND_DRIP_KEYS))
    external_global_features.update(_feature_subset(quant_model_global, _RUNTIME_QUANT_MODEL_KEYS))
    external_symbol_features = _symbol_feature_subset(sec_symbol, _RUNTIME_SEC_EDGAR_KEYS)
    if central_bank_cross_source_context_ready(central_bank_cross_source):
        for symbol, subset in _symbol_feature_subset(
            central_bank_cross_symbol,
            _RUNTIME_CENTRAL_BANK_CROSS_SOURCE_KEYS,
        ).items():
            current = external_symbol_features.setdefault(symbol, {})
            current.update(subset)
    if decision_context_mesh_ready(decision_context_mesh):
        for symbol, subset in _symbol_feature_subset(
            decision_context_mesh_symbol,
            _RUNTIME_DECISION_CONTEXT_MESH_KEYS,
        ).items():
            current = external_symbol_features.setdefault(symbol, {})
            current.update(subset)
    for symbol, subset in _symbol_feature_subset(
        schwab_symbol,
        set(NEWS_STRUCTURED_FEATURE_KEYS) | _RUNTIME_NEWS_EVENT_KEYS | _RUNTIME_SCHWAB_EDUCATION_KEYS,
    ).items():
        current = external_symbol_features.setdefault(symbol, {})
        current.update(subset)
    for symbol, subset in _symbol_feature_subset(market_micro_symbol, _RUNTIME_MARKET_MICRO_KEYS).items():
        current = external_symbol_features.setdefault(symbol, {})
        current.update(subset)
    for symbol, subset in _symbol_feature_subset(extended_symbol, _RUNTIME_EXTENDED_QUANT_KEYS).items():
        current = external_symbol_features.setdefault(symbol, {})
        current.update(subset)
    for symbol, subset in _symbol_feature_subset(options_flow_symbol, _RUNTIME_TASTYTRADE_KEYS).items():
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
    for symbol, subset in _symbol_feature_subset(quant_model_symbol, _RUNTIME_QUANT_MODEL_KEYS).items():
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
    symbol = str(obs.get("symbol") or "").strip().upper()
    symbol_feature_map = external_symbol_features.get(symbol) if isinstance(external_symbol_features.get(symbol), Mapping) else {}
    for key, value in symbol_feature_map.items():
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
    patterns = [
        root / "decision_explanations" / "shadow*" / "decision_explanations_*.jsonl",
        root / "decision_explanations" / "shadow*" / "decision_explanations_*.jsonl.gz",
        root / "decision_explanations" / "shadow*" / "latest_decisions.log",
        root / "decision_explanations" / "shadow*" / "latest_decisions.log.gz",
        root / "decisions" / "*" / "trade_decisions_*.jsonl",
        root / "decisions" / "*" / "trade_decisions_*.jsonl.gz",
        root / "governance" / "channels" / "decision" / "*" / "decision_*.jsonl",
        root / "governance" / "channels" / "decision" / "*" / "decision_*.jsonl.gz",
    ]
    for pattern in patterns:
        for raw in glob.glob(str(pattern)):
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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _runtime_sqlite_read_allowed(project_root: Path) -> bool:
    if _env_flag("RUNTIME_TRAIN_FORCE_SQLITE", False):
        return True

    progress_path = Path(project_root).expanduser().resolve() / _DEFAULT_SQL_PROGRESS_HEALTH
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    if not isinstance(progress, dict):
        return True

    status = str(progress.get("status") or "").strip().lower()
    current_step = str(progress.get("current_step") or "").strip().lower()
    running = bool(progress.get("running", False)) or status == "running"
    timestamp_raw = str(progress.get("timestamp_utc") or "").strip().replace("Z", "+00:00")
    age_seconds = None
    if timestamp_raw:
        try:
            ts = datetime.fromisoformat(timestamp_raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_seconds = max((datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds(), 0.0)
        except Exception:
            age_seconds = None

    if not running:
        return True
    if age_seconds is not None and age_seconds > 6 * 3600:
        return True
    if current_step in {"merge_primary", "merge_shards", "merge_shard", "hot_retention", "checkpoint_primary"}:
        return False
    return True


def _load_hdf5_snapshot_rows(
    project_root: Path,
    *,
    lookback_days: int,
    mode_allowlist: Optional[Sequence[str]],
    symbol_allowlist: Optional[Sequence[str]],
) -> RuntimeSequenceMap:
    root = Path(project_root).expanduser().resolve()
    health_path = root / _DEFAULT_HDF5_CACHE_HEALTH
    runtime_health_path = root / _DEFAULT_RUNTIME_SNAPSHOT_HEALTH
    try:
        cache_health = json.loads(health_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    try:
        runtime_health = json.loads(runtime_health_path.read_text(encoding="utf-8"))
    except Exception:
        runtime_health = {}
    if not isinstance(cache_health, dict):
        return {}
    if not isinstance(runtime_health, dict):
        runtime_health = {}
    if str(cache_health.get("overall_status") or "") != "ready":
        return {}
    freshness = cache_health.get("freshness_gate") if isinstance(cache_health.get("freshness_gate"), dict) else {}
    schema = cache_health.get("schema_validation") if isinstance(cache_health.get("schema_validation"), dict) else {}
    if not bool(freshness.get("fresh")) or not bool(schema.get("ok")):
        return {}
    source = cache_health.get("source_snapshot") if isinstance(cache_health.get("source_snapshot"), dict) else {}
    runtime_rows_sha = str(runtime_health.get("rows_sha256") or "")
    cache_rows_sha = str(source.get("rows_sha256") or "")
    if runtime_rows_sha and cache_rows_sha and runtime_rows_sha != cache_rows_sha:
        return {}
    cache = cache_health.get("cache") if isinstance(cache_health.get("cache"), dict) else {}
    h5_path = Path(str(cache.get("h5_path") or "")).expanduser()
    if not h5_path.is_absolute():
        h5_path = root / h5_path
    if not h5_path.exists():
        return {}
    try:
        import h5py  # type: ignore
    except Exception:
        return {}

    since_utc = datetime.now(timezone.utc) - timedelta(days=max(int(lookback_days), 1))
    mode_allow = {str(x).strip().lower() for x in (mode_allowlist or []) if str(x).strip()}
    symbol_allow = {str(x).strip().upper() for x in (symbol_allowlist or []) if str(x).strip()}
    grouped: RuntimeSequenceMap = defaultdict(list)
    try:
        with h5py.File(h5_path, "r") as h5:
            if int(h5.attrs.get("schema_version", 0) or 0) < _HDF5_CACHE_SCHEMA_VERSION:
                return {}
            if str(h5.attrs.get("source_rows_sha256", "") or "") != cache_rows_sha:
                return {}
            if "immutable_research_snapshots/raw_rows_json" not in h5:
                return {}
            raw_rows = h5["immutable_research_snapshots/raw_rows_json"][:]
            for raw in raw_rows:
                text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                try:
                    row = json.loads(text)
                except Exception:
                    continue
                if not isinstance(row, dict):
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
                grouped[(mode, symbol)].append(dict(row))
    except Exception:
        return {}
    return grouped


def _load_runtime_snapshot_rows(
    project_root: Path,
    *,
    lookback_days: int,
    mode_allowlist: Optional[Sequence[str]],
    symbol_allowlist: Optional[Sequence[str]],
    snapshot_file: Optional[Path] = None,
) -> RuntimeSequenceMap:
    root = Path(project_root).expanduser().resolve()
    summary_path = Path(snapshot_file or (root / _DEFAULT_RUNTIME_SNAPSHOT_HEALTH)).expanduser()
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(summary, dict):
        return {}
    if int(summary.get("lookback_days", 0) or 0) < max(int(lookback_days), 1):
        return {}
    rows_path = Path(str(summary.get("rows_path") or "")).expanduser()
    if not rows_path.exists():
        return {}

    since_utc = datetime.now(timezone.utc) - timedelta(days=max(int(lookback_days), 1))
    mode_allow = {str(x).strip().lower() for x in (mode_allowlist or []) if str(x).strip()}
    symbol_allow = {str(x).strip().upper() for x in (symbol_allowlist or []) if str(x).strip()}
    grouped: RuntimeSequenceMap = defaultdict(list)
    try:
        with rows_path.open("r", encoding="utf-8") as handle:
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
                grouped[(mode, symbol)].append(dict(row))
    except Exception:
        return {}
    return grouped


def _runtime_sqlite_like_patterns(*, lookback_days: int) -> List[str]:
    day_count = max(int(lookback_days), 1) + 2
    now_utc = datetime.now(timezone.utc)
    patterns = [
        f"governance/channels/decision/%/decision_{(now_utc - timedelta(days=offset)).strftime('%Y%m%d')}.jsonl%"
        for offset in range(day_count)
    ]
    patterns.extend(
        f"decision_explanations/%/decision_explanations_{(now_utc - timedelta(days=offset)).strftime('%Y%m%d')}.jsonl%"
        for offset in range(day_count)
    )
    patterns.extend(
        f"decisions/%/trade_decisions_{(now_utc - timedelta(days=offset)).strftime('%Y%m%d')}.jsonl%"
        for offset in range(day_count)
    )
    patterns.append("decision_explanations/%/latest_decisions.log%")
    return patterns


def _iter_runtime_observation_rows(
    project_root: Path,
    *,
    lookback_days: int,
    prefer_sqlite: bool,
) -> Iterable[dict[str, Any]]:
    root = Path(project_root).expanduser().resolve()
    paths = _recent_decision_paths(root, lookback_days=max(int(lookback_days), 1))
    effective_prefer_sqlite = bool(prefer_sqlite and _runtime_sqlite_read_allowed(root))
    sqlite_path = resolve_sqlite_path(os.getenv("RUNTIME_TRAIN_SQLITE_PATH", "").strip() or None)
    sql_source_rels: List[str] = []
    file_fallbacks = paths
    sqlite_had_runtime_history = False
    if effective_prefer_sqlite and sqlite_path.exists():
        try:
            for row in iter_sqlite_jsonl_rows_by_like_patterns(
                sqlite_path=sqlite_path,
                like_patterns=_runtime_sqlite_like_patterns(lookback_days=max(int(lookback_days), 1)),
            ):
                if isinstance(row, dict):
                    sqlite_had_runtime_history = True
                    yield row
        except Exception:
            sqlite_had_runtime_history = False

        sql_source_rels, file_fallbacks = split_paths_by_sqlite_coverage(
            project_root=root,
            paths=paths,
            sqlite_path=sqlite_path,
        )
        if sqlite_had_runtime_history:
            sql_source_rels = []
    elif not effective_prefer_sqlite:
        file_fallbacks = paths

    if effective_prefer_sqlite and sql_source_rels:
        for row in iter_sqlite_jsonl_rows(sqlite_path=sqlite_path, source_rels=sql_source_rels):
            if isinstance(row, dict):
                yield row

    for path in file_fallbacks:
        try:
            if path.suffix == ".gz":
                handle_cm = gzip.open(path, "rt", encoding="utf-8")
            else:
                handle_cm = path.open("r", encoding="utf-8")
            with handle_cm as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        yield row
        except Exception:
            continue


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


def _runtime_observation_timestamp(observation: Mapping[str, Any]) -> Optional[datetime]:
    timestamp = _parse_ts(observation.get("timestamp_utc"))
    if timestamp is not None:
        return timestamp
    try:
        epoch = float(observation.get("ts_epoch"))
    except Exception:
        return None
    if epoch <= 0.0 or not math.isfinite(epoch):
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _runtime_contract_outcome_horizon(
    sequence: Sequence[RuntimeObservation],
    idx: int,
    configured_horizon: int,
    *,
    minimum_maturity_seconds: float,
    maximum_maturity_seconds: float,
) -> Tuple[Optional[int], str]:
    base_horizon = max(int(configured_horizon), 1)
    minimum_seconds = max(float(minimum_maturity_seconds), 0.0)
    maximum_seconds = max(float(maximum_maturity_seconds), 0.0)
    if minimum_seconds <= 0.0:
        return base_horizon, ""

    anchor_ts = _runtime_observation_timestamp(sequence[idx])
    if anchor_ts is None:
        return base_horizon, ""
    for outcome_idx in range(idx + base_horizon, len(sequence)):
        outcome_ts = _runtime_observation_timestamp(sequence[outcome_idx])
        if outcome_ts is None:
            continue
        maturity_seconds = float((outcome_ts - anchor_ts).total_seconds())
        if maturity_seconds < minimum_seconds:
            continue
        if maximum_seconds > 0.0 and maturity_seconds > maximum_seconds:
            return None, "label_maturity_after_contract_maximum"
        return outcome_idx - idx, ""
    return None, "label_horizon_not_mature_for_contract"


def runtime_label_evidence(
    sequence: Sequence[RuntimeObservation],
    idx: int,
    horizon: int,
    *,
    expected_mode: str = "",
    expected_symbol: str = "",
    label_owner_id: str = "",
    semantic_horizon: str = "",
    minimum_maturity_seconds: float = 0.0,
    maximum_maturity_seconds: float = 0.0,
    label_contract_sha256: str = "",
) -> Dict[str, Any]:
    """Validate and identify the point-in-time evidence behind one label."""
    h = max(int(horizon), 1)
    reasons: List[str] = []
    if idx < 0 or (idx + h) >= len(sequence):
        return {
            "eligible": False,
            "reasons": ["label_horizon_not_mature"],
            "lineage_sha256": "",
            "maturity_seconds": 0.0,
        }

    anchor = sequence[idx]
    outcome = sequence[idx + h]
    anchor_ts = _runtime_observation_timestamp(anchor)
    outcome_ts = _runtime_observation_timestamp(outcome)
    if anchor_ts is None:
        reasons.append("missing_feature_timestamp")
    if outcome_ts is None:
        reasons.append("missing_label_maturity_timestamp")
    maturity_seconds = 0.0
    if anchor_ts is not None and outcome_ts is not None:
        maturity_seconds = float((outcome_ts - anchor_ts).total_seconds())
        if maturity_seconds <= 0.0:
            reasons.append("noncausal_label_maturity")
        minimum_seconds = max(float(minimum_maturity_seconds), 0.0)
        maximum_seconds = max(float(maximum_maturity_seconds), 0.0)
        if minimum_seconds > 0.0 and maturity_seconds < minimum_seconds:
            reasons.append("label_maturity_before_contract_minimum")
        if maximum_seconds > 0.0 and maturity_seconds > maximum_seconds:
            reasons.append("label_maturity_after_contract_maximum")

    expected_mode_norm = str(expected_mode or "").strip().lower()
    expected_symbol_norm = str(expected_symbol or "").strip().upper()
    anchor_mode = str(anchor.get("mode") or expected_mode_norm).strip().lower()
    outcome_mode = str(outcome.get("mode") or expected_mode_norm).strip().lower()
    anchor_symbol = str(anchor.get("symbol") or expected_symbol_norm).strip().upper()
    outcome_symbol = str(outcome.get("symbol") or expected_symbol_norm).strip().upper()
    if expected_mode_norm and (anchor_mode != expected_mode_norm or outcome_mode != expected_mode_norm):
        reasons.append("cross_mode_label_join")
    if expected_symbol_norm and (anchor_symbol != expected_symbol_norm or outcome_symbol != expected_symbol_norm):
        reasons.append("cross_symbol_label_join")
    if anchor_mode and outcome_mode and anchor_mode != outcome_mode:
        reasons.append("cross_mode_label_join")
    if anchor_symbol and outcome_symbol and anchor_symbol != outcome_symbol:
        reasons.append("cross_symbol_label_join")

    anchor_snapshot_id = str(anchor.get("snapshot_id") or "").strip()
    outcome_snapshot_id = str(outcome.get("snapshot_id") or "").strip()
    if not anchor_snapshot_id:
        reasons.append("missing_feature_snapshot_id")
    if not outcome_snapshot_id:
        reasons.append("missing_label_snapshot_id")
    if anchor_snapshot_id and outcome_snapshot_id and anchor_snapshot_id == outcome_snapshot_id:
        reasons.append("duplicate_snapshot_label_join")

    anchor_price = observation_feature(anchor, "last_price", 0.0)
    outcome_price = observation_feature(outcome, "last_price", 0.0)
    if anchor_price <= 0.0:
        reasons.append("invalid_feature_price")
    if outcome_price <= 0.0:
        reasons.append("invalid_label_price")

    reasons = sorted(set(reasons))
    lineage_payload = {
        "schema_version": "runtime_label_evidence_v2",
        "label_owner_id": str(label_owner_id or "").strip().lower(),
        "label_contract_sha256": str(label_contract_sha256 or "").strip().lower(),
        "semantic_horizon": str(semantic_horizon or "").strip().lower(),
        "minimum_maturity_seconds": round(max(float(minimum_maturity_seconds), 0.0), 6),
        "maximum_maturity_seconds": round(max(float(maximum_maturity_seconds), 0.0), 6),
        "mode": anchor_mode or expected_mode_norm,
        "symbol": anchor_symbol or expected_symbol_norm,
        "feature_timestamp_utc": anchor_ts.isoformat() if anchor_ts is not None else "",
        "label_matured_at_utc": outcome_ts.isoformat() if outcome_ts is not None else "",
        "feature_snapshot_id": anchor_snapshot_id,
        "label_snapshot_id": outcome_snapshot_id,
        "horizon_rows": h,
    }
    lineage_sha256 = hashlib.sha256(
        json.dumps(lineage_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **lineage_payload,
        "eligible": not reasons,
        "reasons": reasons,
        "lineage_sha256": lineage_sha256,
        "maturity_seconds": round(max(maturity_seconds, 0.0), 6),
    }


def _runtime_label_evidence_summary(
    *,
    candidate_count: int,
    valid_candidate_count: int,
    accepted_count: int,
    rejected_candidate_count: int,
    selected_training_sample_count: int,
    rejection_counts: Mapping[str, int],
    maturity_seconds: Sequence[float],
    lineage_hashes: Sequence[str],
    effective_horizon_rows: Sequence[int],
    label_contract: Mapping[str, Any] | None,
    label_owner_id: str,
) -> Dict[str, Any]:
    maturity = np.asarray([float(value) for value in maturity_seconds if float(value) >= 0.0], dtype=np.float64)
    hashes = [str(value) for value in lineage_hashes if str(value)]
    effective_horizons = np.asarray([int(value) for value in effective_horizon_rows if int(value) > 0], dtype=np.int64)
    objective_class = str((label_contract or {}).get("objective_class") or "market_outcome")
    horizon_policy = (label_contract or {}).get("label_horizon_policy")
    horizon_policy = dict(horizon_policy) if isinstance(horizon_policy, Mapping) else {}
    rejection_reason_occurrence_count = int(sum(int(value) for value in rejection_counts.values()))
    return {
        "schema_version": "runtime_label_evidence_v2",
        "label_owner_id": str(label_owner_id or "").strip().lower(),
        "objective_class": objective_class,
        "semantic_horizon": str((label_contract or {}).get("primary_horizon") or horizon_policy.get("semantic_horizon") or ""),
        "horizon_enforcement_mode": str(horizon_policy.get("enforcement_mode") or "configured_row_horizon"),
        "minimum_maturity_seconds_required": int((label_contract or {}).get("minimum_label_maturity_seconds", horizon_policy.get("minimum_maturity_seconds", 0)) or 0),
        "maximum_maturity_seconds_allowed": int((label_contract or {}).get("maximum_label_maturity_seconds", horizon_policy.get("maximum_maturity_seconds", 0)) or 0),
        "candidate_count": int(candidate_count),
        "point_in_time_valid_candidate_count": int(valid_candidate_count),
        "accepted_label_count": int(accepted_count),
        "accepted_materialized_label_count": int(accepted_count),
        "selected_training_sample_count": int(selected_training_sample_count),
        "unmaterialized_after_evidence_count": max(int(valid_candidate_count) - int(accepted_count), 0),
        "rejected_evidence_count": int(rejected_candidate_count),
        "rejected_evidence_candidate_count": int(rejected_candidate_count),
        "rejection_reason_occurrence_count": rejection_reason_occurrence_count,
        "rejection_counts": dict(sorted((str(key), int(value)) for key, value in rejection_counts.items())),
        "lineage_record_count": len(hashes),
        "unique_lineage_count": len(set(hashes)),
        "lineage_collision_count": max(len(hashes) - len(set(hashes)), 0),
        "maturity_seconds_min": round(float(np.min(maturity)), 6) if maturity.size else 0.0,
        "maturity_seconds_median": round(float(np.median(maturity)), 6) if maturity.size else 0.0,
        "maturity_seconds_max": round(float(np.max(maturity)), 6) if maturity.size else 0.0,
        "effective_horizon_rows_min": int(np.min(effective_horizons)) if effective_horizons.size else 0,
        "effective_horizon_rows_median": round(float(np.median(effective_horizons)), 6) if effective_horizons.size else 0.0,
        "effective_horizon_rows_max": int(np.max(effective_horizons)) if effective_horizons.size else 0,
        "point_in_time_guard_enforced": True,
        "invalid_evidence_admitted": 0,
        "training_eligible": bool(accepted_count > 0 and objective_class == "market_outcome"),
        "policy": "reject invalid label evidence; never convert missing or noncausal outcomes into class zero",
    }


def _label_repair_direction_label(
    sequence: Sequence[RuntimeObservation],
    idx: int,
    horizon: int,
    *,
    min_abs_return: float,
) -> Optional[float]:
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    move_floor = max(float(min_abs_return), 0.00012)
    if abs(fwd_ret) < move_floor and realized < (move_floor * 3.0) and drawdown < (move_floor * 4.0):
        return None
    return 1.0 if fwd_ret >= 0.0 else 0.0


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


def cost_adjusted_direction_label_builder(
    *,
    min_edge: float = 0.0,
    transaction_cost_bps: float = 6.0,
    spread_cost_weight: float = 0.35,
    tradeability_weight: float = 0.002,
    vwap_bias_weight: float = 0.002,
) -> RuntimeLabelBuilder:
    threshold = abs(float(min_edge))
    cost_floor = max(float(transaction_cost_bps), 0.0) / 10000.0
    spread_weight = max(float(spread_cost_weight), 0.0)
    tradeability_weight = max(float(tradeability_weight), 0.0)
    vwap_bias_weight = max(float(vwap_bias_weight), 0.0)

    def _label(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> Optional[float]:
        gross_ret = future_return(sequence, idx, horizon)
        spread_cost = max(observation_feature(sequence[idx], "spread_bps", 0.0), 0.0) / 10000.0
        tradeability = _safe_float(observation_feature(sequence[idx], "market_micro_tradeability_score_norm", 1.0), 1.0)
        tradeability_penalty = max(1.0 - tradeability, 0.0) * tradeability_weight
        vwap_bias = abs(
            observation_feature(
                sequence[idx],
                "futures_vwap_bias_norm",
                observation_feature(sequence[idx], "options_vwap_bias_norm", 0.0),
            )
        )
        total_cost = cost_floor + (spread_weight * spread_cost) + tradeability_penalty + (vwap_bias_weight * vwap_bias)
        if gross_ret > 0.0:
            net_edge = gross_ret - total_cost
            if net_edge <= threshold:
                return None
            return 1.0
        if gross_ret < 0.0:
            net_edge = gross_ret + total_cost
            if abs(net_edge) <= threshold:
                return None
            return 0.0
        if abs(gross_ret) <= threshold:
            return None
        return None

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


def fill_adjusted_outcome_label_builder(
    *,
    min_net_return: float = 0.0,
    transaction_cost_bps: float = 6.0,
    spread_cost_weight: float = 0.25,
    slippage_cost_weight: float = 0.45,
    impact_cost_weight: float = 0.25,
    fee_cost_weight: float = 1.0,
    tradeability_weight: float = 0.002,
    stop_target_realism_weight: float = 0.0015,
) -> RuntimeLabelBuilder:
    threshold = abs(float(min_net_return))
    cost_floor = max(float(transaction_cost_bps), 0.0) / 10000.0
    spread_cost_weight = max(float(spread_cost_weight), 0.0)
    slippage_cost_weight = max(float(slippage_cost_weight), 0.0)
    impact_cost_weight = max(float(impact_cost_weight), 0.0)
    fee_cost_weight = max(float(fee_cost_weight), 0.0)
    tradeability_weight = max(float(tradeability_weight), 0.0)
    stop_target_realism_weight = max(float(stop_target_realism_weight), 0.0)

    def _label(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> Optional[float]:
        gross_ret = future_return(sequence, idx, horizon)
        spread_cost = max(observation_feature(sequence[idx], "spread_bps", 0.0), 0.0) / 10000.0
        slippage_cost = max(
            observation_feature(
                sequence[idx],
                "lag_slippage_bps",
                observation_feature(sequence[idx], "paper_recent_slippage_bps_norm", 0.0) * 10.0,
            ),
            0.0,
        ) / 10000.0
        impact_cost = max(observation_feature(sequence[idx], "lag_impact_bps", 0.0), 0.0) / 10000.0
        fee_cost = max(observation_feature(sequence[idx], "lag_fee_bps", 0.0), 0.0) / 10000.0
        tradeability = _safe_float(
            observation_feature(
                sequence[idx],
                "execution_fitness_norm",
                observation_feature(sequence[idx], "market_micro_tradeability_score_norm", 1.0),
            ),
            1.0,
        )
        tradeability_penalty = max(1.0 - tradeability, 0.0) * tradeability_weight
        stop_target_realism = _safe_float(observation_feature(sequence[idx], "stop_target_realism_norm", 1.0), 1.0)
        realism_penalty = max(1.0 - stop_target_realism, 0.0) * stop_target_realism_weight
        total_cost = (
            cost_floor
            + (spread_cost_weight * spread_cost)
            + (slippage_cost_weight * slippage_cost)
            + (impact_cost_weight * impact_cost)
            + (fee_cost_weight * fee_cost)
            + tradeability_penalty
            + realism_penalty
        )
        if gross_ret > 0.0:
            net_ret = gross_ret - total_cost
            if net_ret <= threshold:
                return None
            return 1.0
        if gross_ret < 0.0:
            net_ret = gross_ret + total_cost
            if abs(net_ret) <= threshold:
                return None
            return 0.0
        return None

    return _label


def event_followthrough_label_builder(
    *,
    checkpoints: Sequence[float] = (0.25, 0.5, 1.0),
    min_return: float = 0.0,
    min_followthrough_share: float = 0.66,
    max_reversal_share: float = 0.50,
) -> RuntimeLabelBuilder:
    threshold = abs(float(min_return))
    min_followthrough_share = max(0.0, min(float(min_followthrough_share), 1.0))
    max_reversal_share = max(0.0, float(max_reversal_share))
    checkpoint_fracs = sorted(
        {
            max(0.1, min(float(raw), 1.0))
            for raw in checkpoints
            if float(raw) > 0.0
        }
    ) or [1.0]

    def _label(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> Optional[float]:
        eval_horizon = max(int(horizon), 1)
        returns: List[float] = []
        for frac in checkpoint_fracs:
            step = max(1, int(round(eval_horizon * frac)))
            if (idx + step) >= len(sequence):
                return None
            returns.append(future_return(sequence, idx, step))
        final_ret = returns[-1]
        if abs(final_ret) <= threshold:
            return None
        direction = 1.0 if final_ret > 0.0 else -1.0
        aligned = sum(1 for ret in returns if (ret * direction) > threshold)
        if (aligned / max(len(returns), 1)) < min_followthrough_share:
            return None
        reversal = max((max(-ret, 0.0) if direction > 0.0 else max(ret, 0.0)) for ret in returns)
        if reversal > (abs(final_ret) * max_reversal_share):
            return None
        return 1.0 if direction > 0.0 else 0.0

    return _label


def abstain_quality_label_builder(
    *,
    max_abs_return: float = 0.0015,
    min_stress_score: float = 0.45,
    max_spread_bps: float = 18.0,
    min_tradeability: float = 0.45,
    max_execution_fitness: float = 0.45,
) -> RuntimeLabelBuilder:
    flat_threshold = abs(float(max_abs_return))
    min_stress_score = _safe_float(min_stress_score, 0.45)
    max_spread_bps = max(float(max_spread_bps), 1.0)
    min_tradeability = _safe_float(min_tradeability, 0.45)
    max_execution_fitness = _safe_float(max_execution_fitness, 0.45)

    def _label(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> Optional[float]:
        realized_move = abs(future_return(sequence, idx, horizon))
        spread_norm = min(max(observation_feature(sequence[idx], "spread_bps", 0.0), 0.0) / max_spread_bps, 1.0)
        tradeability = _safe_float(observation_feature(sequence[idx], "market_micro_tradeability_score_norm", 1.0), 1.0)
        execution_fitness = _safe_float(observation_feature(sequence[idx], "execution_fitness_norm", tradeability), tradeability)
        halt_risk = max(
            observation_feature(sequence[idx], "market_micro_trade_halt_norm", 0.0),
            observation_feature(sequence[idx], "market_micro_luld_pause_norm", 0.0),
        )
        stress_score = _safe_float(
            0.34 * max(1.0 - tradeability, 0.0)
            + 0.28 * max(1.0 - execution_fitness, 0.0)
            + 0.22 * spread_norm
            + 0.16 * halt_risk,
            0.0,
        )
        structural_drag = (
            tradeability <= min_tradeability
            or execution_fitness <= max_execution_fitness
            or spread_norm >= 0.75
            or halt_risk >= 0.50
        )
        if stress_score >= min_stress_score and structural_drag and realized_move <= flat_threshold:
            return 1.0
        if realized_move > flat_threshold and stress_score < min_stress_score:
            return 0.0
        return None

    return _label


def regime_specific_label_builder(
    *,
    regime: str,
    min_return: float = 0.0,
    regime_threshold: float = 0.55,
) -> RuntimeLabelBuilder:
    regime_key = str(regime or "trend").strip().lower() or "trend"
    threshold = abs(float(min_return))
    regime_threshold = _safe_float(regime_threshold, 0.55)

    def _regime_score(obs: RuntimeObservation) -> float:
        if regime_key == "trend":
            return max(
                observation_feature(obs, "day_regime_trend_norm", 0.0),
                observation_feature(obs, "market_micro_trend_persistence_norm", 0.0),
                observation_feature(obs, "swing_weekly_trend_confirm_norm", 0.0),
            )
        if regime_key in {"chop", "mean_revert"}:
            return max(
                observation_feature(obs, "day_regime_chop_norm", 0.0),
                observation_feature(obs, "market_micro_reversal_risk_norm", 0.0),
                observation_feature(obs, "market_micro_range_expansion_norm", 0.0) * 0.5,
            )
        if regime_key == "shock":
            return max(
                observation_feature(obs, "calendar_event_proximity_norm", 0.0),
                observation_feature(obs, "market_micro_post_event_drift_norm", 0.0),
                observation_feature(obs, "market_micro_range_expansion_norm", 0.0),
                observation_feature(obs, "regime_dislocation_norm", 0.0),
            )
        return observation_feature(obs, "day_regime_trend_norm", 0.0)

    def _label(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> Optional[float]:
        obs = sequence[idx]
        if _regime_score(obs) < regime_threshold:
            return None
        ret = future_return(sequence, idx, horizon)
        if abs(ret) <= threshold:
            return None
        recent_bias = _safe_float(
            observation_feature(
                obs,
                "pct_from_close",
                observation_feature(obs, "mom_5m", 0.0),
            ),
            0.0,
        )
        if regime_key == "trend":
            if recent_bias != 0.0 and (ret * recent_bias) <= 0.0:
                return None
        elif regime_key in {"chop", "mean_revert"}:
            if abs(recent_bias) <= threshold or (ret * recent_bias) >= 0.0:
                return None
        return 1.0 if ret > 0.0 else 0.0

    return _label


def income_total_return_label_builder(
    *,
    min_total_return: float = 0.0,
    min_income_quality: float = 0.52,
    max_payout_stress: float = 0.74,
    income_yield_weight: float = 0.0025,
    compounding_weight: float = 0.0015,
) -> RuntimeLabelBuilder:
    threshold = abs(float(min_total_return))
    min_income_quality = _safe_float(min_income_quality, 0.52)
    max_payout_stress = _safe_float(max_payout_stress, 0.74)
    income_yield_weight = max(float(income_yield_weight), 0.0)
    compounding_weight = max(float(compounding_weight), 0.0)

    def _label(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> Optional[float]:
        obs = sequence[idx]
        price_return = future_return(sequence, idx, horizon)
        income_quality = max(
            observation_feature(obs, "dividend_income_quality_norm", 0.0),
            observation_feature(obs, "long_term_total_return_income_norm", 0.0),
        )
        compounding_quality = max(
            observation_feature(obs, "dividend_compounding_quality_norm", 0.0),
            observation_feature(obs, "long_term_compounder_conviction_norm", 0.0),
        )
        payout_stress = max(
            observation_feature(obs, "dividend_payout_stress_forward_norm", 0.0),
            observation_feature(obs, "dividend_forward_hazard_norm", 0.0),
            observation_feature(obs, "long_term_corporate_action_hazard_norm", 0.0),
        )
        total_return = price_return + (
            observation_feature(obs, "dividend_yield_norm", 0.0) * income_yield_weight
        ) + (compounding_quality * compounding_weight)
        if income_quality < min_income_quality or payout_stress > max_payout_stress:
            if total_return < -threshold:
                return 0.0
            return None
        if total_return > threshold:
            return 1.0
        if total_return < -threshold:
            return 0.0
        return None

    return _label


def derivatives_structure_label_builder(
    *,
    min_return: float = 0.0,
    min_structure_score: float = 0.54,
    min_directional_edge: float = 0.08,
) -> RuntimeLabelBuilder:
    threshold = abs(float(min_return))
    min_structure_score = _safe_float(min_structure_score, 0.54)
    min_directional_edge = abs(float(min_directional_edge))

    def _label(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> Optional[float]:
        obs = sequence[idx]
        options_edge = (
            (observation_feature(obs, "options_net_call_premium_bias_norm", 0.5) - 0.5)
            + (observation_feature(obs, "options_gamma_expiry_skew_norm", 0.5) - 0.5)
            + (observation_feature(obs, "options_surface_change_norm", 0.5) - 0.5)
        ) / 3.0
        futures_edge = (
            (observation_feature(obs, "futures_order_book_imbalance_norm", 0.5) - 0.5)
            + (observation_feature(obs, "futures_basis_bps_norm", 0.5) - 0.5)
            + (observation_feature(obs, "futures_term_structure_norm", 0.5) - 0.5)
        ) / 3.0
        flow_edge = _safe_float(observation_feature(obs, "flow_direction_signed", 0.0), 0.0)
        structure_signal = _safe_float((0.40 * options_edge) + (0.40 * futures_edge) + (0.20 * flow_edge), 0.0)
        structure_score = max(
            observation_feature(obs, "core_options_structure_edge_norm", 0.0),
            observation_feature(obs, "core_futures_curve_alignment_norm", 0.0),
            observation_feature(obs, "core_futures_regime_edge_norm", 0.0),
        )
        future_ret = future_return(sequence, idx, horizon)
        if structure_score < min_structure_score or abs(structure_signal) < min_directional_edge or abs(future_ret) <= threshold:
            return None
        if (future_ret * structure_signal) > 0.0:
            return 1.0 if future_ret > 0.0 else 0.0
        return None

    return _label


def _mode_family_label(mode: Any) -> str:
    text = str(mode or "").strip().lower()
    if "dividend" in text:
        return "dividend"
    if "long_term" in text:
        return "long_term"
    if "intraday" in text or "day" in text:
        return "intraday"
    if "swing" in text:
        return "swing"
    if "bond" in text:
        return "bond"
    if "options" in text:
        return "options"
    if "futures" in text:
        return "futures"
    if "fx" in text:
        return "fx"
    if "crypto" in text:
        return "crypto"
    if "aggressive" in text or "default" in text or "conservative" in text:
        return "trading"
    return "other"


def load_runtime_observation_sequences(
    project_root: Path,
    *,
    lookback_days: int = 14,
    mode_allowlist: Optional[Sequence[str]] = None,
    symbol_allowlist: Optional[Sequence[str]] = None,
    prefer_sqlite: Optional[bool] = None,
    allow_snapshot: bool = True,
    snapshot_file: Optional[Path] = None,
    max_observation_rows: Optional[int] = None,
) -> RuntimeSequenceMap:
    root = Path(project_root).expanduser().resolve()
    since_utc = datetime.now(timezone.utc) - timedelta(days=max(int(lookback_days), 1))
    mode_allow = {str(x).strip().lower() for x in (mode_allowlist or []) if str(x).strip()}
    symbol_allow = {str(x).strip().upper() for x in (symbol_allowlist or []) if str(x).strip()}
    gap_fill_context = _load_runtime_gap_fill_context(root)
    sidecar_paths = _recent_decision_paths(root, lookback_days=max(int(lookback_days), 1))
    sidecar_max_rows = max(int(os.getenv("RUNTIME_TRAIN_PRICE_SIDECAR_MAX_ROWS", "200000") or 200000), 1000)
    price_sidecar = _build_runtime_price_sidecar_from_rows(
        _iter_runtime_price_sidecar_rows(sidecar_paths, max_rows=sidecar_max_rows),
        max_rows=sidecar_max_rows,
    )
    effective_prefer_sqlite = _env_flag("RUNTIME_TRAIN_PREFER_SQLITE", False) if prefer_sqlite is None else bool(prefer_sqlite)

    if allow_snapshot and _env_flag("RUNTIME_TRAIN_USE_SNAPSHOT", False):
        env_snapshot_file = Path(str(os.getenv("RUNTIME_TRAIN_SNAPSHOT_FILE", "")).strip()).expanduser() if str(os.getenv("RUNTIME_TRAIN_SNAPSHOT_FILE", "")).strip() else None
        snapshot_rows: RuntimeSequenceMap = {}
        if _env_flag("RUNTIME_TRAIN_USE_HDF5_CACHE", True):
            snapshot_rows = _load_hdf5_snapshot_rows(
                root,
                lookback_days=max(int(lookback_days), 1),
                mode_allowlist=mode_allowlist,
                symbol_allowlist=symbol_allowlist,
            )
        if not snapshot_rows:
            snapshot_rows = _load_runtime_snapshot_rows(
                root,
                lookback_days=max(int(lookback_days), 1),
                mode_allowlist=mode_allowlist,
                symbol_allowlist=symbol_allowlist,
                snapshot_file=snapshot_file or env_snapshot_file,
            )
        if snapshot_rows:
            out: RuntimeSequenceMap = {}
            for key, rows in snapshot_rows.items():
                rows_sorted = sorted(
                    rows,
                    key=lambda x: (
                        float(x.get("ts_epoch", 0.0)),
                        int(x.get("strategy_priority", 99)),
                        str(x.get("snapshot_id") or ""),
                    ),
                )
                carry_forward_features: Dict[str, float] = {}
                deduped: List[RuntimeObservation] = []
                seen_snapshot_ids: set[str] = set()
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
                    for feature_key, feature_value in feature_map.items():
                        try:
                            numeric_value = float(feature_value)
                        except Exception:
                            continue
                        if math.isfinite(numeric_value):
                            next_carry[str(feature_key)] = numeric_value
                    carry_forward_features = next_carry
                if deduped:
                    out[key] = deduped
            if out:
                return out
        if _env_flag("RUNTIME_TRAIN_SNAPSHOT_ONLY", False):
            return {}

    max_source_rows = max(int(max_observation_rows or 0), 0)
    source_rows_seen = 0
    best_by_snapshot: Dict[Tuple[str, str, str], RuntimeObservation] = {}
    for row in _iter_runtime_observation_rows(root, lookback_days=max(int(lookback_days), 1), prefer_sqlite=effective_prefer_sqlite):
        source_rows_seen += 1
        if max_source_rows and source_rows_seen > max_source_rows:
            break
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if not metadata:
            metadata = dict(_runtime_row_metadata(row))
        layer = str(metadata.get("layer") or "").strip().lower()
        trusted_runtime_layer = (
            layer == "grand_master"
            or "master" in layer
            or layer == "sub_bot_paper_mirror"
        )
        if not trusted_runtime_layer:
            continue
        strategy = _runtime_row_strategy(row, metadata)
        strategy_priority = _runtime_strategy_priority(strategy, metadata)
        if strategy_priority is None:
            continue

        ts = _parse_ts(row.get("timestamp_utc"))
        if ts is None or ts < since_utc:
            continue
        mode = _runtime_row_mode(row, metadata)
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

        snapshot_ids = _runtime_snapshot_id_candidates(row, metadata)
        features = _runtime_row_features(row)
        price = _safe_float(features.get("last_price"), 0.0)
        if price <= 0.0:
            price = _runtime_row_price(row, features)
        if price <= 0.0:
            sidecar_entry = _lookup_runtime_sidecar_context(
                price_sidecar,
                symbol=symbol,
                snapshot_ids=snapshot_ids,
                ts=ts,
            )
            features = _runtime_features_with_sidecar_context(features, sidecar_entry)
            price = _runtime_row_price(row, features)
        if price <= 0.0:
            continue

        snapshot_id = str(snapshot_ids[0] if snapshot_ids else "").strip()
        if not snapshot_id:
            snapshot_id = f"{symbol}:{ts.isoformat()}"

        obs = {
            "strategy": strategy,
            "strategy_priority": int(strategy_priority),
            "snapshot_id": snapshot_id,
            "ts_epoch": float(ts.timestamp()),
            "price": price,
            "features": features,
        }
        key = (mode, symbol, snapshot_id)
        prev = best_by_snapshot.get(key)
        if prev is None or int(obs["strategy_priority"]) < int(prev.get("strategy_priority", 99)):
            best_by_snapshot[key] = obs

    grouped: RuntimeSequenceMap = defaultdict(list)
    for (mode, symbol, _snapshot_id), obs in best_by_snapshot.items():
        grouped[(str(mode), str(symbol))].append(obs)

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
            row_for_enrich = dict(row)
            row_for_enrich.setdefault("mode", key[0])
            row_for_enrich.setdefault("symbol", key[1])
            if "timestamp_utc" not in row_for_enrich:
                try:
                    row_for_enrich["timestamp_utc"] = datetime.fromtimestamp(
                        float(row_for_enrich.get("ts_epoch", 0.0)),
                        tz=timezone.utc,
                    ).isoformat()
                except Exception:
                    row_for_enrich["timestamp_utc"] = ""
            enriched = _enrich_runtime_observation(
                row_for_enrich,
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


def _sample_regime_label(row: RuntimeObservation) -> str:
    features = row.get("features") if isinstance(row.get("features"), Mapping) else {}
    shock_score = max(
        _safe_float(features.get("news_shock_rate"), 0.0),
        _safe_float(features.get("calendar_macro_surprise_norm"), 0.0),
        _safe_float(features.get("market_micro_trade_halt_norm"), 0.0),
        _safe_float(features.get("market_micro_range_expansion_norm"), 0.0),
    )
    trend_score = max(
        _safe_float(features.get("day_regime_trend_norm"), 0.0),
        _safe_float(features.get("market_micro_trend_persistence_norm"), 0.0),
        _safe_float(features.get("futures_curve_shift_velocity_norm"), 0.0),
    )
    mean_revert_score = max(
        _safe_float(features.get("day_regime_mean_revert_norm"), 0.0),
        _safe_float(features.get("market_micro_reversal_risk_norm"), 0.0),
    )
    if shock_score >= 0.60:
        return "shock"
    if trend_score >= max(0.58, mean_revert_score + 0.08):
        return "trend"
    if mean_revert_score >= 0.55:
        return "mean_revert"
    return "chop"


def _sample_session_label(row: RuntimeObservation) -> str:
    features = row.get("features") if isinstance(row.get("features"), Mapping) else {}
    if _safe_float(features.get("market_micro_overnight_gap_norm"), 0.0) >= 0.55:
        return "overnight_gap"
    if _safe_float(features.get("market_micro_session_open_norm"), 0.0) >= 0.55:
        return "open"
    if _safe_float(features.get("market_micro_session_power_hour_norm"), 0.0) >= 0.55:
        return "power_hour"
    if _safe_float(features.get("market_micro_post_event_drift_norm"), 0.0) >= 0.55:
        return "post_event_drift"
    if _safe_float(features.get("market_micro_session_midday_norm"), 0.0) >= 0.55:
        return "midday"
    return "regular"


def _select_evenly_spaced_indices(indices: np.ndarray, keep_count: int, conf: np.ndarray, anchor_ts: np.ndarray) -> np.ndarray:
    if indices.size <= keep_count:
        return np.asarray(indices, dtype=np.int64)
    indices_sorted = np.asarray(indices[np.argsort(anchor_ts[indices], kind="stable")], dtype=np.int64)
    anchor_positions = np.unique(np.linspace(0, max(indices_sorted.size - 1, 0), num=keep_count, dtype=np.int64))
    selected = indices_sorted[anchor_positions]
    if selected.size < keep_count:
        selected_set = {int(i) for i in selected.tolist()}
        extras_ranked = sorted(
            [int(i) for i in indices_sorted.tolist() if int(i) not in selected_set],
            key=lambda idx: (-float(conf[idx]), -float(anchor_ts[idx])),
        )
        need = int(keep_count - selected.size)
        if need > 0 and extras_ranked:
            selected = np.concatenate([selected, np.asarray(extras_ranked[:need], dtype=np.int64)])
    return np.sort(np.asarray(selected, dtype=np.int64))


def _apply_symbol_and_regime_balance(
    X: np.ndarray,
    y: np.ndarray,
    conf: np.ndarray,
    anchor_ts: np.ndarray,
    symbols: np.ndarray,
    modes: np.ndarray,
    regimes: np.ndarray,
    sessions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    total_samples = int(y.shape[0]) if y.ndim == 2 else 0
    meta: Dict[str, Any] = {
        "symbol_cap_applied": False,
        "symbol_cap_reason": "not_needed",
        "symbol_cap_max_share": float(_safe_float(os.getenv("RUNTIME_TRAIN_SYMBOL_MAX_SHARE", "0.35"), 0.35)),
        "regime_balance_applied": False,
        "regime_balance_reason": "not_needed",
        "regime_balance_max_ratio": float(_safe_float(os.getenv("RUNTIME_TRAIN_REGIME_MAX_RATIO", "2.5"), 2.5)),
    }
    if total_samples <= 0:
        return X, y, conf, symbols, modes, regimes, sessions, meta

    selected_idx = np.arange(total_samples, dtype=np.int64)
    symbol_max_share = min(max(float(meta["symbol_cap_max_share"]), 0.10), 0.95)
    symbol_cap_floor = max(int(_safe_float(os.getenv("RUNTIME_TRAIN_SYMBOL_CAP_MIN_SAMPLES", "96"), 96)), 16)
    if total_samples >= symbol_cap_floor:
        max_per_symbol = max(int(math.ceil(total_samples * symbol_max_share)), 8)
        keep_chunks: List[np.ndarray] = []
        symbol_counts: Dict[str, int] = {}
        symbol_capped = False
        for symbol in sorted({str(item) for item in symbols.tolist()}):
            sym_idx = np.flatnonzero(symbols == symbol)
            symbol_counts[symbol] = int(sym_idx.size)
            keep_count = min(int(sym_idx.size), max_per_symbol)
            if keep_count < int(sym_idx.size):
                symbol_capped = True
            keep_chunks.append(_select_evenly_spaced_indices(sym_idx, keep_count, conf, anchor_ts))
        if keep_chunks:
            selected_idx = np.sort(np.concatenate(keep_chunks))
            if symbol_capped:
                meta["symbol_cap_applied"] = True
                meta["symbol_cap_reason"] = "capped_dominant_symbols"
        meta["symbol_counts_before"] = symbol_counts

    X_sel = np.asarray(X[selected_idx], dtype=np.float32)
    y_sel = np.asarray(y[selected_idx], dtype=np.float32)
    conf_sel = np.asarray(conf[selected_idx], dtype=np.float32)
    symbols_sel = np.asarray(symbols[selected_idx])
    modes_sel = np.asarray(modes[selected_idx])
    regimes_sel = np.asarray(regimes[selected_idx])
    sessions_sel = np.asarray(sessions[selected_idx])
    anchor_sel = np.asarray(anchor_ts[selected_idx], dtype=np.float64)

    regime_cap_floor = max(int(_safe_float(os.getenv("RUNTIME_TRAIN_REGIME_BALANCE_MIN_SAMPLES", "72"), 72)), 12)
    if y_sel.shape[0] >= regime_cap_floor:
        regime_counts_before = {str(key): int(np.sum(regimes_sel == key)) for key in sorted({str(item) for item in regimes_sel.tolist()})}
        non_zero_counts = [count for count in regime_counts_before.values() if count > 0]
        if len(non_zero_counts) >= 2:
            max_ratio = min(max(float(meta["regime_balance_max_ratio"]), 1.0), 6.0)
            minority_count = min(non_zero_counts)
            dominant_cap = max(int(math.ceil(minority_count * max_ratio)), minority_count)
            keep_chunks = []
            regime_capped = False
            for regime in sorted(regime_counts_before):
                reg_idx = np.flatnonzero(regimes_sel == regime)
                keep_count = min(int(reg_idx.size), dominant_cap)
                if keep_count < int(reg_idx.size):
                    regime_capped = True
                keep_chunks.append(_select_evenly_spaced_indices(reg_idx, keep_count, conf_sel, anchor_sel))
            if keep_chunks:
                reg_selected = np.sort(np.concatenate(keep_chunks))
                X_sel = np.asarray(X_sel[reg_selected], dtype=np.float32)
                y_sel = np.asarray(y_sel[reg_selected], dtype=np.float32)
                conf_sel = np.asarray(conf_sel[reg_selected], dtype=np.float32)
                symbols_sel = np.asarray(symbols_sel[reg_selected])
                modes_sel = np.asarray(modes_sel[reg_selected])
                regimes_sel = np.asarray(regimes_sel[reg_selected])
                sessions_sel = np.asarray(sessions_sel[reg_selected])
                if regime_capped:
                    meta["regime_balance_applied"] = True
                    meta["regime_balance_reason"] = "capped_dominant_regimes"
        meta["regime_counts_before"] = regime_counts_before

    meta["symbol_counts_after"] = {
        str(key): int(np.sum(symbols_sel == key)) for key in sorted({str(item) for item in symbols_sel.tolist()})
    }
    meta["regime_counts_after"] = {
        str(key): int(np.sum(regimes_sel == key)) for key in sorted({str(item) for item in regimes_sel.tolist()})
    }
    return X_sel, y_sel, conf_sel, symbols_sel, modes_sel, regimes_sel, sessions_sel, meta


def _build_label_audit(
    labels: np.ndarray,
    symbols: np.ndarray,
    modes: np.ndarray,
    regimes: np.ndarray,
    sessions: np.ndarray,
) -> Dict[str, Any]:
    labels_flat = np.asarray(labels[:, 0], dtype=np.float32).reshape(-1) if labels.ndim == 2 else np.asarray([], dtype=np.float32)

    def _group(values: np.ndarray) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for key in sorted({str(item) for item in values.tolist()}):
            mask = values == key
            count = int(np.sum(mask))
            if count <= 0:
                continue
            selected = labels_flat[mask]
            rows.append(
                {
                    "name": key,
                    "sample_count": count,
                    "positive_rate": round(float(np.mean(selected)) if selected.size else 0.0, 6),
                    "positive_count": int(np.sum(selected >= 0.5)),
                    "negative_count": int(np.sum(selected < 0.5)),
                }
            )
        rows.sort(key=lambda row: (-int(row["sample_count"]), row["name"]))
        return rows[:20]

    return {
        "by_symbol": _group(symbols),
        "by_sleeve": _group(modes),
        "by_family": _group(np.asarray([_mode_family_label(item) for item in modes.tolist()], dtype=object)),
        "by_regime": _group(regimes),
        "by_session": _group(sessions),
    }


def make_runtime_windowed_dataset(
    *,
    sequences: RuntimeSequenceMap,
    feature_builder: RuntimeFeatureBuilder,
    label_builder: RuntimeLabelBuilder,
    label_contract: Optional[Mapping[str, Any]] = None,
    label_owner_id: str = "",
    sample_filter: Optional[RuntimeSampleFilter] = None,
    confidence_builder: Optional[RuntimeConfidenceBuilder] = None,
    min_confidence: float = 0.0,
    sample_stride: int = 1,
    max_samples: int = 0,
    bypass_sample_filter: bool = False,
    fallback_direction_label: bool = False,
    fallback_min_abs_return: float = 0.00035,
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
    sample_symbols: List[str] = []
    sample_modes: List[str] = []
    sample_regimes: List[str] = []
    sample_sessions: List[str] = []
    eligible_sequences = 0
    skipped_labels = 0
    skipped_filtered = 0
    skipped_low_confidence = 0
    repaired_labels = 0
    feature_dim = 0
    evidence_candidate_count = 0
    evidence_valid_candidate_count = 0
    evidence_rejected_candidate_count = 0
    evidence_rejection_counts: Counter[str] = Counter()
    accepted_maturity_seconds: List[float] = []
    accepted_lineage_hashes: List[str] = []
    accepted_lineage_set: set[str] = set()
    accepted_effective_horizons: List[int] = []
    objective_class = str((label_contract or {}).get("objective_class") or "market_outcome").strip().lower()
    horizon_policy = (label_contract or {}).get("label_horizon_policy")
    horizon_policy = dict(horizon_policy) if isinstance(horizon_policy, Mapping) else {}
    semantic_horizon = str((label_contract or {}).get("primary_horizon") or horizon_policy.get("semantic_horizon") or "")
    minimum_maturity_seconds = max(
        _safe_float((label_contract or {}).get("minimum_label_maturity_seconds", horizon_policy.get("minimum_maturity_seconds", 0)), 0.0),
        0.0,
    )
    maximum_maturity_seconds = max(
        _safe_float((label_contract or {}).get("maximum_label_maturity_seconds", horizon_policy.get("maximum_maturity_seconds", 0)), 0.0),
        0.0,
    )
    label_contract_sha256 = str((label_contract or {}).get("contract_sha256") or "")

    for (mode_key, symbol_key), rows in sequences.items():
        if len(rows) < (w + h):
            continue
        eligible_sequences += 1
        for idx in range(w - 1, len(rows) - h, stride):
            evidence_candidate_count += 1
            if objective_class != "market_outcome":
                evidence_rejected_candidate_count += 1
                evidence_rejection_counts["objective_requires_non_market_outcome"] += 1
                continue
            effective_horizon, horizon_rejection_reason = _runtime_contract_outcome_horizon(
                rows,
                idx,
                h,
                minimum_maturity_seconds=minimum_maturity_seconds,
                maximum_maturity_seconds=maximum_maturity_seconds,
            )
            if effective_horizon is None:
                evidence_rejected_candidate_count += 1
                evidence_rejection_counts[horizon_rejection_reason or "label_horizon_not_mature_for_contract"] += 1
                continue
            evidence = runtime_label_evidence(
                rows,
                idx,
                effective_horizon,
                expected_mode=str(mode_key),
                expected_symbol=str(symbol_key),
                label_owner_id=label_owner_id,
                semantic_horizon=semantic_horizon,
                minimum_maturity_seconds=minimum_maturity_seconds,
                maximum_maturity_seconds=maximum_maturity_seconds,
                label_contract_sha256=label_contract_sha256,
            )
            if not bool(evidence.get("eligible", False)):
                evidence_rejected_candidate_count += 1
                reasons = evidence.get("reasons") if isinstance(evidence.get("reasons"), list) else []
                for reason in reasons or ["invalid_label_evidence"]:
                    evidence_rejection_counts[str(reason)] += 1
                continue
            lineage_sha256 = str(evidence.get("lineage_sha256") or "")
            if not lineage_sha256 or lineage_sha256 in accepted_lineage_set:
                evidence_rejected_candidate_count += 1
                evidence_rejection_counts["duplicate_label_lineage"] += 1
                continue
            evidence_valid_candidate_count += 1
            if sample_filter is not None and not bypass_sample_filter:
                try:
                    include_sample = bool(sample_filter(rows, idx, effective_horizon))
                except Exception:
                    include_sample = False
                if not include_sample:
                    skipped_filtered += 1
                    continue

            confidence = 1.0
            if confidence_builder is not None:
                try:
                    confidence = _safe_float(confidence_builder(rows, idx, effective_horizon), 0.0)
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

            label = label_builder(rows, idx, effective_horizon)
            if label is None or (not math.isfinite(float(label))):
                if fallback_direction_label:
                    label = _label_repair_direction_label(
                        rows,
                        idx,
                        effective_horizon,
                        min_abs_return=fallback_min_abs_return,
                    )
                    if label is not None and math.isfinite(float(label)):
                        repaired_labels += 1
                if label is None or (not math.isfinite(float(label))):
                    skipped_labels += 1
                    continue

            sample = np.concatenate(per_step, axis=0)
            samples.append(sample.astype(np.float32))
            labels.append(float(label))
            accepted_maturity_seconds.append(float(evidence.get("maturity_seconds", 0.0) or 0.0))
            accepted_lineage_hashes.append(lineage_sha256)
            accepted_lineage_set.add(lineage_sha256)
            accepted_effective_horizons.append(int(effective_horizon))
            anchor_datetime = _runtime_observation_timestamp(rows[idx])
            anchor_ts.append(float(anchor_datetime.timestamp()) if anchor_datetime is not None else 0.0)
            sample_confidence.append(float(confidence))
            sample_symbols.append(str(rows[idx].get("symbol") or symbol_key or "").strip().upper())
            sample_modes.append(str(rows[idx].get("mode") or mode_key or "").strip().lower())
            sample_regimes.append(_sample_regime_label(rows[idx]))
            sample_sessions.append(_sample_session_label(rows[idx]))

    if not samples:
        evidence_audit = _runtime_label_evidence_summary(
            candidate_count=evidence_candidate_count,
            valid_candidate_count=evidence_valid_candidate_count,
            accepted_count=0,
            rejected_candidate_count=evidence_rejected_candidate_count,
            selected_training_sample_count=0,
            rejection_counts=evidence_rejection_counts,
            maturity_seconds=accepted_maturity_seconds,
            lineage_hashes=accepted_lineage_hashes,
            effective_horizon_rows=accepted_effective_horizons,
            label_contract=label_contract,
            label_owner_id=label_owner_id,
        )
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
            "label_repair_enabled": bool(fallback_direction_label),
            "label_repair_bypassed_filter": bool(bypass_sample_filter),
            "label_repaired": int(repaired_labels),
            "confidence_mean": 0.0,
            "confidence_min": 0.0,
            "confidence_max": 0.0,
            "min_confidence": float(min_conf),
            "label_contract": dict(label_contract or {}),
            "label_evidence_audit": evidence_audit,
            "_sample_confidence": np.zeros((0,), dtype=np.float32),
        }

    order = np.argsort(np.asarray(anchor_ts, dtype=np.float64))
    X = np.asarray([samples[i] for i in order], dtype=np.float32)
    y = np.asarray([[labels[i]] for i in order], dtype=np.float32)
    conf = np.asarray([sample_confidence[i] for i in order], dtype=np.float32)
    anchor_ordered = np.asarray([anchor_ts[i] for i in order], dtype=np.float64)
    symbols = np.asarray([sample_symbols[i] for i in order], dtype=object)
    modes = np.asarray([sample_modes[i] for i in order], dtype=object)
    regimes = np.asarray([sample_regimes[i] for i in order], dtype=object)
    sessions = np.asarray([sample_sessions[i] for i in order], dtype=object)
    X, y, conf, balance_meta = _rebalance_binary_runtime_dataset(
        X,
        y,
        conf,
        anchor_ordered,
    )
    selected_idx = np.asarray(balance_meta.pop("_selected_idx", np.arange(int(y.shape[0]), dtype=np.int64)), dtype=np.int64)
    symbols = np.asarray(symbols[selected_idx])
    modes = np.asarray(modes[selected_idx])
    regimes = np.asarray(regimes[selected_idx])
    sessions = np.asarray(sessions[selected_idx])
    anchor_ordered = np.asarray(anchor_ordered[selected_idx], dtype=np.float64)
    X, y, conf, symbols, modes, regimes, sessions, context_balance_meta = _apply_symbol_and_regime_balance(
        X,
        y,
        conf,
        anchor_ordered,
        symbols,
        modes,
        regimes,
        sessions,
    )
    memory_sample_cap_limit = max(int(max_samples), 0)
    memory_sample_cap_applied = False
    memory_sample_cap_original_count = int(X.shape[0])
    if memory_sample_cap_limit > 0 and int(X.shape[0]) > memory_sample_cap_limit:
        selected_idx = np.linspace(0, int(X.shape[0]) - 1, num=memory_sample_cap_limit, dtype=np.int64)
        selected_idx = np.unique(selected_idx)
        X = np.asarray(X[selected_idx], dtype=np.float32)
        y = np.asarray(y[selected_idx], dtype=np.float32)
        conf = np.asarray(conf[selected_idx], dtype=np.float32)
        symbols = np.asarray(symbols[selected_idx])
        modes = np.asarray(modes[selected_idx])
        regimes = np.asarray(regimes[selected_idx])
        sessions = np.asarray(sessions[selected_idx])
        memory_sample_cap_applied = True
    positive_rate = float(np.mean(y[:, 0])) if y.size else 0.0
    label_audit = _build_label_audit(y, symbols, modes, regimes, sessions)
    evidence_audit = _runtime_label_evidence_summary(
        candidate_count=evidence_candidate_count,
        valid_candidate_count=evidence_valid_candidate_count,
        accepted_count=len(accepted_lineage_hashes),
        rejected_candidate_count=evidence_rejected_candidate_count,
        selected_training_sample_count=int(X.shape[0]),
        rejection_counts=evidence_rejection_counts,
        maturity_seconds=accepted_maturity_seconds,
        lineage_hashes=accepted_lineage_hashes,
        effective_horizon_rows=accepted_effective_horizons,
        label_contract=label_contract,
        label_owner_id=label_owner_id,
    )
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
        "label_repair_enabled": bool(fallback_direction_label),
        "label_repair_bypassed_filter": bool(bypass_sample_filter),
        "label_repaired": int(repaired_labels),
        "confidence_mean": float(np.mean(conf)) if conf.size else 0.0,
        "confidence_min": float(np.min(conf)) if conf.size else 0.0,
        "confidence_max": float(np.max(conf)) if conf.size else 0.0,
        "min_confidence": float(min_conf),
        "memory_sample_cap_limit": int(memory_sample_cap_limit),
        "memory_sample_cap_applied": bool(memory_sample_cap_applied),
        "memory_sample_cap_original_count": int(memory_sample_cap_original_count),
        "label_contract": dict(label_contract or {}),
        "label_evidence_audit": evidence_audit,
        "_sample_confidence": conf,
        "label_audit": label_audit,
        **balance_meta,
        **context_balance_meta,
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
        "_selected_idx": np.arange(total_samples, dtype=np.int64),
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
            "_selected_idx": selected_idx,
        }
    )
    return X_out, y_out, conf_out, base_meta
