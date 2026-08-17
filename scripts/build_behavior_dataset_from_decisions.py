import argparse
import glob
import json
import math
import os
import re
import sys
from itertools import chain
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

try:
    import orjson as _fast_json
except Exception:
    _fast_json = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_FAILURE_PATH = PROJECT_ROOT / "governance" / "health" / "trade_behavior_dataset_build_failure_latest.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sql_dataset_io import iter_sqlite_jsonl_rows, resolve_sqlite_path, split_paths_by_sqlite_coverage
from core.central_bank_liquidity import (
    CENTRAL_BANK_LIQUIDITY_FEATURE_KEYS,
    assess_central_bank_liquidity_context,
)
from core.global_central_bank_context import (
    CENTRAL_BANK_CROSS_SOURCE_FEATURE_KEYS,
    GLOBAL_CENTRAL_BANK_FEATURE_KEYS,
    assess_central_bank_cross_source_context,
    assess_global_central_bank_context,
)
from core.decision_context_mesh import (
    DECISION_CONTEXT_MESH_FEATURE_KEYS,
    assess_decision_context_mesh,
)
from core.profitability_hardening import post_cost_adjusted_forward_return

SHOCK_SYMBOLS = {"UVXY", "VIXY", "SOXL", "SOXS", "MSTR", "SMCI", "COIN", "TSLA"}
MEAN_REVERT_SYMBOLS = {"TLT", "IEF", "SHY", "BND", "AGG", "GLD", "XLU", "XLP"}
DEFENSIVE_DIVIDEND_SYMBOLS = {"SCHD", "VYM", "DVY", "XLP", "XLU", "PG", "MO", "XOM", "O", "CVX", "KO", "PEP", "JNJ"}

DIVIDEND_DRIP_FEATURE_NAMES = [
    "dividend_drip_active_norm",
    "dividend_drip_recent_reinvest_norm",
    "dividend_drip_cash_only_norm",
    "dividend_drip_share_credit_norm",
    "dividend_drip_event_recency_norm",
    "dividend_drip_confidence_norm",
]

PLUMBED_CONTEXT_FEATURE_NAMES = [
    "live_macro_gate_active_norm",
    "live_macro_gate_confidence_norm",
    "sec_context_signal_norm",
    "extended_quant_signal_norm",
    "official_macro_signal_norm",
    "schwab_education_signal_norm",
    "market_breadth_signal_norm",
    "bond_reference_signal_norm",
    *CENTRAL_BANK_LIQUIDITY_FEATURE_KEYS,
    *GLOBAL_CENTRAL_BANK_FEATURE_KEYS,
    *CENTRAL_BANK_CROSS_SOURCE_FEATURE_KEYS,
    *DECISION_CONTEXT_MESH_FEATURE_KEYS,
]

SOURCE_QUALITY_FEATURE_NAMES = [
    "source_quality_average_score_norm",
    "source_quality_required_failure_ratio_norm",
    "source_quality_soft_failure_ratio_norm",
    "source_quality_unverified_ratio_norm",
    "source_quality_cross_verified_ratio_norm",
    "source_quality_market_micro_score_norm",
    "source_quality_official_macro_score_norm",
    "source_quality_crypto_context_score_norm",
]

FEATURE_NAMES = [
    "pnl_proxy",
    "qty_log",
    "role_idx",
    "symbol_hash",
    "action_hash",
    "dow",
    "hour",
    "regime_idx",
    "label_confidence_proxy",
    "pct_from_close_scaled",
    "mom_5m_scaled",
    "vol_30m_scaled",
    "range_pos",
    "spread_bps_norm",
    "ctx_vix_pct_scaled",
    "ctx_uup_pct_scaled",
    "lag_slippage_bps_norm",
    "lag_latency_ms_norm",
    "lag_impact_bps_norm",
    "active_sub_bots_norm",
    "queue_depth_norm",
    "dispatch_qty_norm",
    "session_bucket_norm",
    "mins_from_open_norm",
    "mins_to_close_norm",
    "event_window_proximity",
    "feature_freshness_ok",
    "feature_freshness_age_ratio",
    "master_latency_slo_ok",
    "master_latency_ratio",
    "risk_pause_active",
    "options_chain_available",
    "options_iv_atm_norm",
    "options_iv_skew_norm",
    "options_iv_term_structure_norm",
    "options_put_call_oi_ratio_norm",
    "options_negative_bias_norm",
    "options_roll_yield_norm",
    "options_vwap_bias_norm",
    "options_vol_expectation_norm",
    "calendar_event_proximity_norm",
    "calendar_high_impact_24h_norm",
    "calendar_options_expiry_week_norm",
    "calendar_dividend_events_30d_norm",
    "calendar_dividend_exdate_proximity_norm",
    "calendar_dividend_payout_proximity_norm",
    "calendar_dividend_recent_exdate_norm",
    "calendar_dividend_quality_signal_norm",
    "dividend_yield_norm",
    "dividend_payout_ratio_norm",
    "dividend_ex_date_proximity_norm",
    "dividend_pay_date_proximity_norm",
    "dividend_quality_score_norm",
    "dividend_capture_entry_signal_norm",
    "dividend_capture_exit_signal_norm",
    "dividend_compound_bias_norm",
    "dividend_compound_growth_norm",
    "dividend_compound_drawdown_norm",
    "dividend_compound_steps_norm",
    "dividend_compounding_quality_norm",
    "dividend_capture_timing_quality_norm",
    "dividend_payout_stress_gate_norm",
    "dividend_growth_persistence_norm",
    "dividend_capture_ex_date_hazard_norm",
    "dividend_strategy_mode_capture",
    "dividend_strategy_mode_compound",
    "dividend_strategy_mode_hybrid",
    *DIVIDEND_DRIP_FEATURE_NAMES,
    "futures_order_book_imbalance_norm",
    "futures_funding_rate_norm",
    "futures_basis_bps_norm",
    "futures_term_structure_norm",
    "futures_negative_bias_norm",
    "futures_roll_yield_norm",
    "futures_vwap_bias_norm",
    "options_specialist_active",
    "futures_specialist_active",
    "options_specialist_vote",
    "futures_specialist_vote",
    "active_options_sub_bots_norm",
    "active_futures_sub_bots_norm",
    "snapshot_cov_ok",
    "snapshot_cov_log_ratio",
    "snapshot_replay_stale_ratio",
    "snapshot_replay_drift_ratio",
    "snapshot_divergence_ratio",
    "snapshot_triprate_ratio",
    "snapshot_queue_pressure_ratio",
    "snapshot_drill_ok",
    "snapshot_drill_restore_fail_ratio",
    "snapshot_drill_missing_ratio",
    "snapshot_drill_recency_norm",
    "canary_weight_cap_norm",
    "snapshot_raw_sql_ingest_ratio",
    "snapshot_raw_count_norm",
    "snapshot_raw_file_count_norm",
    "snapshot_raw_bytes_norm",
    "snapshot_raw_json_ratio",
    "snapshot_raw_event_file_ratio",
    "snapshot_raw_lock_file_ratio",
    "snapshot_raw_recency_norm",
    "snapshot_cov_fill_ratio",
    "snapshot_replay_ok",
    "snapshot_e2e_replay_ok",
    "snapshot_e2e_hash_match",
    "snapshot_paper_replay_ok",
    "snapshot_paper_replay_hash_match",
    "external_feeds_ok",
    "external_feeds_recency_norm",
    "external_fred_unrate_norm",
    "external_fred_cpi_mom_norm",
    "external_fred_gdp_qoq_norm",
    "external_bls_unrate_norm",
    "external_bls_cpi_mom_norm",
    "external_census_population_log_norm",
    "external_bea_dataset_count_norm",
    "external_micro_auction_norm",
    "external_micro_relative_volume_norm",
    "external_micro_options_flow_norm",
    "external_micro_short_pressure_norm",
    "external_micro_credit_flow_norm",
    "external_micro_block_trade_norm",
    "tasty_iv_rank_norm",
    "tasty_implied_volatility_index_norm",
    "tasty_liquidity_rating_norm",
    "tasty_expected_move_norm",
    "tasty_beta_norm",
    "tasty_watchlist_presence_norm",
    "options_iv_crush_risk_norm",
    "options_assignment_risk_norm",
    "options_zero_dte_regime_norm",
    "options_vol_of_vol_change_norm",
    "options_spread_execution_risk_norm",
    "options_higher_order_greek_pressure_norm",
    "options_barrier_touch_risk_norm",
    "options_lookback_path_dependency_norm",
    "options_variance_swap_proxy_norm",
    "options_volatility_swap_proxy_norm",
    "options_gamma_scalping_pressure_norm",
    "options_vanna_volga_hedge_pressure_norm",
    "options_dispersion_trade_proxy_norm",
    "options_volatility_arbitrage_proxy_norm",
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
]

FEATURE_NAMES.extend(PLUMBED_CONTEXT_FEATURE_NAMES)
FEATURE_NAMES.extend(SOURCE_QUALITY_FEATURE_NAMES)

BEHAVIOR_LANE_FEATURE_NAMES = [
    "core_default_dependency_norm",
    "core_bot_concentration_norm",
    "core_aggressive_identity_norm",
    "profile_symbol_drag_penalty_norm",
    "profile_kill_switch_norm",
    "core_conservative_quality_gate_norm",
    "core_aggressive_breakout_conviction_norm",
    "core_options_structure_edge_norm",
    "core_fx_macro_confirmation_norm",
    "core_futures_regime_edge_norm",
    "core_futures_curve_alignment_norm",
    "core_crypto_unwind_risk_norm",
    "core_cross_sectional_rank_norm",
    "core_regime_specialist_blend_norm",
    "core_exit_persistence_norm",
    "core_portfolio_overlap_pressure_norm",
    "core_event_reaction_norm",
    "core_cross_asset_confirmation_norm",
    "core_champion_challenger_gap_norm",
    "aggressive_relative_strength_burst_norm",
    "options_skew_dislocation_norm",
    "options_gamma_wall_reaction_norm",
    "futures_basis_dislocation_norm",
    "futures_overnight_inventory_norm",
    "day_opening_auction_signal_norm",
    "day_halt_resume_risk_norm",
    "day_liquidity_vacuum_risk_norm",
    "day_execution_cost_risk_norm",
    "day_session_open_norm",
    "day_session_midday_norm",
    "day_session_power_hour_norm",
    "day_regime_trend_norm",
    "day_regime_chop_norm",
    "day_regime_alignment_norm",
    "day_lunch_chop_norm",
    "day_open_close_imbalance_regime_norm",
    "day_symbol_cooldown_pressure_norm",
    "day_open_drive_conviction_norm",
    "day_failed_breakout_risk_norm",
    "day_closing_squeeze_norm",
    "intraday_allowlist_score_norm",
    "swing_post_earnings_drift_norm",
    "swing_gap_continuation_norm",
    "swing_gap_fade_norm",
    "swing_vol_compression_breakout_norm",
    "swing_sector_relative_strength_norm",
    "swing_weekly_trend_confirm_norm",
    "swing_weekly_pullback_quality_norm",
    "swing_regime_trend_norm",
    "swing_regime_chop_norm",
    "swing_regime_alignment_norm",
    "swing_overnight_event_hazard_norm",
    "swing_event_blackout_norm",
    "bond_duration_regime_norm",
    "bond_curve_steepener_norm",
    "bond_curve_flattener_norm",
    "bond_carry_roll_norm",
    "bond_credit_risk_on_norm",
    "bond_credit_risk_off_norm",
    "bond_inflation_breakeven_norm",
    "bond_bot_roster_alignment_norm",
    "bond_equity_contamination_norm",
    "long_term_factor_exposure_control_norm",
    "long_term_overlap_rebalance_norm",
    "long_term_valuation_reserve_norm",
    "long_term_compounder_conviction_norm",
]

FEATURE_NAMES.extend(BEHAVIOR_LANE_FEATURE_NAMES)

PAPER_CONTEXT_FEATURE_NAMES = [
    "paper_snapshot_trade_count_norm",
    "paper_snapshot_slippage_bps_norm",
    "paper_snapshot_return_proxy_signed_scaled",
    "paper_recent_trade_count_norm",
    "paper_recent_slippage_bps_norm",
    "paper_recent_return_proxy_signed_scaled",
]

FEATURE_NAMES.extend(PAPER_CONTEXT_FEATURE_NAMES)

PAPER_PROFITABILITY_FEATURE_NAMES = [
    "paper_profitability_master_awareness_active_norm",
    "paper_profitability_master_profit_score_norm",
    "paper_profitability_master_drag_norm",
    "paper_profitability_master_training_weight_norm",
    "paper_profitability_master_size_multiplier_norm",
    "paper_profitability_master_risk_norm",
    "paper_profitability_grandmaster_awareness_active_norm",
    "paper_profitability_grandmaster_profit_score_norm",
    "paper_profitability_grandmaster_drag_norm",
    "paper_profitability_grandmaster_training_weight_norm",
    "paper_profitability_grandmaster_size_multiplier_norm",
    "paper_profitability_grandmaster_risk_norm",
    "paper_profitability_grandmaster_exit_pressure_norm",
    "paper_profitability_grandmaster_execution_discount_norm",
    "paper_profitability_grandmaster_conflict_cap_norm",
]

FEATURE_NAMES.extend(PAPER_PROFITABILITY_FEATURE_NAMES)

BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES = [
    "capital_flow_signed_scaled",
    "capital_flow_inflow_norm",
    "capital_flow_outflow_norm",
]

FEATURE_NAMES.extend(BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES)

BEHAVIOR_FLOW_AWARENESS_FEATURE_NAMES = [
    "flow_direction_signed",
    "flow_conviction_norm",
    "flow_defensive_rotation_norm",
    "flow_risk_on_norm",
    "flow_risk_off_norm",
    "flow_stress_norm",
]

FEATURE_NAMES.extend(BEHAVIOR_FLOW_AWARENESS_FEATURE_NAMES)

BEHAVIOR_LEAD_LAG_FEATURE_NAMES = [
    "lead_lag_signal_signed",
    "lead_lag_alignment_norm",
    "lead_lag_confirmation_norm",
    "lead_lag_break_norm",
    "lead_lag_gap_norm",
    "lead_lag_confidence_norm",
]

FEATURE_NAMES.extend(BEHAVIOR_LEAD_LAG_FEATURE_NAMES)

BEHAVIOR_ALLOCATION_FEATURE_NAMES = [
    "allocation_trade_edge_norm",
    "allocation_confidence_norm",
    "allocation_confidence_scale",
    "allocation_conflict_norm",
    "execution_fitness_norm",
    "stop_target_realism_norm",
    "symbol_cooldown_memory_norm",
    "cross_bot_conflict_norm",
    "regime_dislocation_norm",
]

FEATURE_NAMES.extend(BEHAVIOR_ALLOCATION_FEATURE_NAMES)


def _json_loads(raw: Any) -> Any:
    if _fast_json is not None:
        try:
            if isinstance(raw, str):
                return _fast_json.loads(raw)
            return _fast_json.loads(raw or b"")
        except Exception:
            pass
    return json.loads(raw)


def _json_dumps(payload: Any, *, pretty: bool = False) -> str:
    if _fast_json is not None:
        option = 0
        if pretty:
            option |= _fast_json.OPT_INDENT_2
        return _fast_json.dumps(payload, option=option).decode("utf-8")
    if pretty:
        return json.dumps(payload, ensure_ascii=True, indent=2)
    return json.dumps(payload, ensure_ascii=True)


def _safe_load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        obj = _json_loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else default
    except Exception:
        return default


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(_json_dumps(payload, pretty=True), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _publish_dataset(
    payload: Dict[str, Any],
    *,
    out_path: Path,
    failure_path: Path,
    min_output_rows: int,
) -> Dict[str, Any]:
    rows = max(int(payload.get("rows", 0) or 0), 0)
    required_rows = max(int(min_output_rows), 1)
    if rows < required_rows:
        failure_payload = {
            "timestamp_utc": str(payload.get("timestamp_utc") or datetime.now(timezone.utc).isoformat()),
            "ok": False,
            "status": "insufficient_rows_preserved_previous_dataset",
            "rows": rows,
            "min_output_rows": required_rows,
            "out_file": str(out_path),
            "previous_dataset_preserved": bool(out_path.exists()),
            "feature_dim": int(payload.get("feature_dim", 0) or 0),
            "label_counts": payload.get("label_counts") if isinstance(payload.get("label_counts"), dict) else {},
            "skipped": payload.get("skipped") if isinstance(payload.get("skipped"), dict) else {},
            "source": payload.get("source") if isinstance(payload.get("source"), dict) else {},
        }
        _write_json_atomic(failure_path, failure_payload)
        return {
            "published": False,
            "preserved_previous": bool(out_path.exists()),
            "failure_file": str(failure_path),
            "min_output_rows": required_rows,
        }

    _write_json_atomic(out_path, payload)
    return {
        "published": True,
        "preserved_previous": False,
        "failure_file": "",
        "min_output_rows": required_rows,
    }


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(float(value), hi))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _signed_scale(value: float, gain: float) -> float:
    return math.tanh(float(value) * float(gain))


def _hash01(text: str) -> float:
    if not text:
        return 0.0
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h / 0xFFFFFFFF


def _parse_ts(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    s = str(raw).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _fresh_health_payload(payload: Dict[str, Any], *, max_age_hours: float) -> Tuple[Dict[str, Any], bool]:
    if not isinstance(payload, dict) or not payload:
        return {}, False
    ts = _parse_ts(payload.get("timestamp_utc"))
    if ts is None:
        return payload, False
    age_hours = max((datetime.now(timezone.utc) - ts).total_seconds(), 0.0) / 3600.0
    return payload, age_hours <= max(max_age_hours, 0.0)


def _iter_jsonl(paths: Iterable[Path], *, tail_bytes: int = 0) -> Iterable[Dict[str, Any]]:
    for p in paths:
        try:
            tail_limit = max(int(tail_bytes or 0), 0)
            if tail_limit > 0 and p.stat().st_size > tail_limit:
                with p.open("rb") as f:
                    start = max(p.stat().st_size - tail_limit, 0)
                    f.seek(start)
                    raw_lines = f.read().splitlines()
                if start > 0 and raw_lines:
                    raw_lines = raw_lines[1:]
                for raw_line in raw_lines:
                    try:
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                    except Exception:
                        continue
                    if not line:
                        continue
                    try:
                        obj = _json_loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        yield obj
                continue

            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = _json_loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except Exception:
            continue



def _resolve_glob_paths(pattern: str, *, root: Path) -> List[Path]:
    pat = str(pattern or "").strip()
    if not pat:
        return []

    matches: List[Path] = []
    for raw in glob.glob(pat, recursive=True):
        p = Path(raw)
        if p.is_file():
            matches.append(p.resolve())

    if not os.path.isabs(pat):
        for raw in glob.glob(str(root / pat), recursive=True):
            p = Path(raw)
            if p.is_file():
                matches.append(p.resolve())

    uniq = {str(p): p for p in matches}
    return [uniq[k] for k in sorted(uniq.keys())]


def _prefer_external_storage() -> bool:
    return os.getenv("BOT_LOGS_PREFER_EXTERNAL", "1").strip().lower() not in {"0", "false", "no", "off"}


def _routed_input_pattern(pattern: str, *, project_root: Path) -> str:
    raw = str(pattern or "").strip()
    if not raw or _prefer_external_storage():
        return raw
    if os.getenv("BEHAVIOR_DATASET_ALLOW_EXTERNAL_INPUTS", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return raw

    local_root = Path(
        os.getenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(project_root / "local_fallback_storage"))
    ).expanduser()
    project_text = str(project_root.absolute()).rstrip("/")
    normalized = raw
    if os.path.isabs(normalized):
        if normalized == project_text:
            relative = ""
        elif normalized.startswith(f"{project_text}/"):
            relative = normalized[len(project_text) + 1 :]
        elif normalized.startswith("/Volumes/"):
            return ""
        else:
            return normalized
    else:
        relative = normalized.lstrip("./")

    routed_prefixes = (
        "decision_explanations/",
        "decisions/",
        "exports/",
        "governance/shadow",
    )
    if relative.startswith(routed_prefixes):
        return str(local_root / relative)
    return normalized


def _path_day_utc(path: Path) -> Optional[datetime]:
    m = re.search(r"_(\d{8})\.jsonl$", path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _filter_recent_paths(paths: List[Path], since_utc: datetime) -> List[Path]:
    cutoff_day = (since_utc - timedelta(days=1)).date()
    out: List[Path] = []
    for path in paths:
        day_utc = _path_day_utc(path)
        if day_utc is not None and day_utc.date() >= cutoff_day:
            out.append(path)
            continue
        try:
            mtime_utc = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            mtime_utc = None
        if mtime_utc is not None and mtime_utc >= (since_utc - timedelta(days=1)):
            out.append(path)
    return out


def _limit_recent_paths(paths: List[Path], *, max_files: int) -> List[Path]:
    if max_files <= 0 or len(paths) <= max_files:
        return paths

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except Exception:
            return 0.0

    return sorted(sorted(paths, key=_mtime, reverse=True)[:max_files])


def _merge_dict_features(features: Dict[str, Any], payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            continue
        features[str(key)] = value


def _mean_nested_float(rows: Iterable[Dict[str, Any]], key: str) -> float:
    values: List[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _to_float(row.get(key), float("nan"))
        if math.isfinite(value):
            values.append(value)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _channel_paper_profitability_features(row: Dict[str, Any], features: Dict[str, Any]) -> None:
    master_outputs = row.get("master_outputs") if isinstance(row.get("master_outputs"), dict) else {}
    master_meta_rows: List[Dict[str, Any]] = []
    for output in master_outputs.values():
        if not isinstance(output, dict):
            continue
        meta = output.get("master_meta")
        if isinstance(meta, dict):
            master_meta_rows.append(meta)

    master_awareness = _mean_nested_float(master_meta_rows, "paper_profitability_master_awareness")
    master_profit = _mean_nested_float(master_meta_rows, "paper_profitability_master_profit_score")
    master_drag = _mean_nested_float(master_meta_rows, "paper_profitability_master_drag")
    master_risk = _mean_nested_float(master_meta_rows, "paper_profitability_master_risk")
    master_size = _mean_nested_float(master_meta_rows, "paper_profitability_master_size_multiplier")

    if master_meta_rows:
        features["paper_profitability_master_awareness_active_norm"] = _clamp01(master_awareness)
        features["paper_profitability_master_profit_score_norm"] = _clamp01(master_profit)
        features["paper_profitability_master_drag_norm"] = _clamp01(master_drag)
        features["paper_profitability_master_risk_norm"] = _clamp01(master_risk)
        features["paper_profitability_master_size_multiplier_norm"] = _clamp01(master_size)
        features["paper_profitability_master_training_weight_norm"] = _clamp01(master_profit * (1.0 - min(master_drag, 1.0)))

    gm = row.get("grand_master_meta") if isinstance(row.get("grand_master_meta"), dict) else {}
    if gm:
        gm_awareness = _to_float(gm.get("paper_profitability_grandmaster_awareness"), 0.0)
        gm_profit = _to_float(gm.get("paper_profitability_grandmaster_profit_score"), 0.0)
        gm_drag = _to_float(gm.get("paper_profitability_grandmaster_drag"), 0.0)
        features["paper_profitability_grandmaster_awareness_active_norm"] = _clamp01(gm_awareness)
        features["paper_profitability_grandmaster_profit_score_norm"] = _clamp01(gm_profit)
        features["paper_profitability_grandmaster_drag_norm"] = _clamp01(gm_drag)
        features["paper_profitability_grandmaster_risk_norm"] = _clamp01(_to_float(gm.get("paper_profitability_grandmaster_risk"), 0.0))
        features["paper_profitability_grandmaster_exit_pressure_norm"] = _clamp01(_to_float(gm.get("paper_profitability_grandmaster_exit_pressure"), 0.0))
        features["paper_profitability_grandmaster_execution_discount_norm"] = _clamp01(_to_float(gm.get("paper_profitability_grandmaster_execution_discount"), 0.0))
        features["paper_profitability_grandmaster_size_multiplier_norm"] = _clamp01(_to_float(gm.get("paper_profitability_grandmaster_size_multiplier"), 0.0))
        features["paper_profitability_grandmaster_training_weight_norm"] = _clamp01(gm_profit * (1.0 - min(gm_drag, 1.0)))
        features["paper_profitability_grandmaster_conflict_cap_norm"] = _clamp01(
            _to_float(gm.get("paper_profitability_grandmaster_conflict_cap"), 1.0 - _to_float(gm.get("specialist_conflict"), 0.0))
        )


def _canonical_behavior_decision_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return None
    if str(row.get("strategy") or "") == "grand_master_bot":
        return row
    if not any(key in row for key in ("master_action", "master_intent_action", "grand_master_meta", "master_outputs")):
        return None

    features: Dict[str, Any] = {}
    _merge_dict_features(features, row.get("features"))
    for key in (
        "market",
        "data_quality_features",
        "execution_lag_features",
        "capital_flow",
        "flow_awareness_features",
        "lead_lag_features",
        "allocation_confidence",
        "lane_strategy_features",
    ):
        _merge_dict_features(features, row.get(key))
    _channel_paper_profitability_features(row, features)

    for key in ("active_sub_bots", "active_options_sub_bots", "active_futures_sub_bots"):
        if key in row:
            features[key] = row.get(key)

    last_price = _to_float(features.get("last_price"), _to_float(row.get("last_price"), 0.0))
    if last_price <= 0.0:
        return None
    features["last_price"] = last_price

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    snapshot_id = str(row.get("snapshot_id") or metadata.get("snapshot_id") or "").strip()
    canonical_meta = dict(metadata)
    if snapshot_id:
        canonical_meta["snapshot_id"] = snapshot_id

    portfolio = row.get("portfolio") if isinstance(row.get("portfolio"), dict) else {}
    return {
        "timestamp_utc": row.get("timestamp_utc"),
        "symbol": row.get("symbol"),
        "strategy": "grand_master_bot",
        "action": row.get("master_action") or row.get("master_intent_action") or row.get("action") or "HOLD",
        "quantity": row.get("quantity", portfolio.get("dispatch_qty", 0.0)),
        "mode": row.get("mode") or row.get("shadow_profile") or row.get("profile") or row.get("broker") or "",
        "features": features,
        "gates": row.get("gates") or row.get("execution_guard") or {},
        "metadata": canonical_meta,
    }


def _role_index(mode_label: str) -> float:
    m = (mode_label or "").lower()
    if "swing" in m:
        return 2.0 / 3.0
    if "dividend" in m or "bond" in m:
        return 0.0
    return 1.0 / 3.0


def _regime_index(symbol: str, features: Dict[str, Any]) -> Tuple[float, str]:
    s = (symbol or "").upper()
    pct = abs(_to_float(features.get("pct_from_close"), 0.0))
    mom = abs(_to_float(features.get("mom_5m"), 0.0))
    vol = abs(_to_float(features.get("vol_30m"), 0.0))
    event_proximity = _clamp01(_to_float(features.get("calendar_event_proximity_norm"), 0.0))
    dividend_signal = max(
        _clamp01(_to_float(features.get("calendar_dividend_quality_signal_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_yield_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_quality_score_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_capture_entry_signal_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_compound_bias_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_drip_active_norm"), 0.0)),
    )
    futures_event_signal = max(
        abs(_to_float(features.get("futures_order_book_imbalance_norm"), 0.0)),
        _clamp01(_to_float(features.get("futures_term_structure_norm"), 0.0)),
        _clamp01(abs(_to_float(features.get("futures_basis_bps_norm"), 0.0))),
        _clamp01(_to_float(features.get("futures_specialist_active"), 0.0)),
        _clamp01(abs(_to_float(features.get("futures_specialist_vote"), 0.0))),
    )
    guard_heavy_signal = max(
        _clamp01(_to_float(features.get("risk_pause_active"), 0.0)),
        _clamp01(_to_float(features.get("snapshot_triprate_ratio"), 0.0)),
        _clamp01(_to_float(features.get("snapshot_queue_pressure_ratio"), 0.0)),
    )

    if (
        s in SHOCK_SYMBOLS
        or vol >= 0.03
        or pct >= 0.04
        or event_proximity >= 0.55
        or futures_event_signal >= 0.55
        or guard_heavy_signal >= 0.70
    ):
        return 2.0 / 3.0, "shock"
    if s in MEAN_REVERT_SYMBOLS or s in DEFENSIVE_DIVIDEND_SYMBOLS or dividend_signal >= 0.42:
        return 1.0 / 3.0, "mean_revert"
    if mom >= 0.001 or pct >= 0.0015:
        return 0.0, "trend"
    return 1.0, "other"


def _event_windows_from_env() -> List[Tuple[int, int]]:
    raw = os.getenv("EVENT_LOCK_WINDOWS_ET", os.getenv("EVENT_BLACKOUT_WINDOWS_ET", "08:29-08:36,09:59-10:06,13:58-14:05")).strip()
    windows: List[Tuple[int, int]] = []
    if not raw:
        return windows
    for seg in raw.split(","):
        seg = seg.strip()
        if "-" not in seg:
            continue
        a, b = seg.split("-", 1)
        try:
            ah, am = [int(x) for x in a.split(":", 1)]
            bh, bm = [int(x) for x in b.split(":", 1)]
            windows.append((ah * 60 + am, bh * 60 + bm))
        except Exception:
            continue
    return windows


def _session_event_context(ts_utc: datetime, windows: List[Tuple[int, int]]) -> Dict[str, float]:
    if ZoneInfo is not None:
        ts_et = ts_utc.astimezone(ZoneInfo("America/New_York"))
    else:
        ts_et = ts_utc

    now_min = ts_et.hour * 60 + ts_et.minute
    open_min = 9 * 60 + 30
    close_min = 16 * 60

    if now_min < open_min:
        bucket = 0.0
    elif now_min <= close_min:
        bucket = 0.5
    else:
        bucket = 1.0

    mins_from_open = _clamp01((now_min - open_min) / 390.0)
    mins_to_close = _clamp01((close_min - now_min) / 390.0)

    proximity = 0.0
    for start_min, end_min in windows:
        if start_min <= end_min:
            if start_min <= now_min <= end_min:
                proximity = 1.0
                break
            dist = min(abs(now_min - start_min), abs(now_min - end_min))
        else:
            in_window = now_min >= start_min or now_min <= end_min
            if in_window:
                proximity = 1.0
                break
            dist = min(abs(now_min - start_min), abs(now_min - end_min))
        proximity = max(proximity, _clamp01(1.0 - (dist / 30.0)))

    return {
        "session_bucket_norm": bucket,
        "mins_from_open_norm": mins_from_open,
        "mins_to_close_norm": mins_to_close,
        "event_window_proximity": proximity,
    }


def _snapshot_health_context(project_root: Path) -> Tuple[Dict[str, float], Dict[str, Any]]:
    try:
        from snapshot_health_sql import load_snapshot_context

        sqlite_override = str(os.getenv("SNAPSHOT_CONTEXT_SQLITE_PATH", "")).strip()
        sqlite_path = Path(sqlite_override).expanduser() if sqlite_override else None
        prefer_sql = str(os.getenv("SNAPSHOT_CONTEXT_PREFER_SQL", "1")).strip() == "1"
        persist_sql = str(os.getenv("SNAPSHOT_CONTEXT_PERSIST_TO_SQL", "1")).strip() == "1"

        context, meta = load_snapshot_context(
            project_root=project_root,
            sqlite_path=sqlite_path,
            prefer_sql=prefer_sql,
            persist_files_to_sql=persist_sql,
        )
        if context:
            context.setdefault("snapshot_e2e_replay_ok", 1.0)
            context.setdefault("snapshot_e2e_hash_match", 1.0)
            context.setdefault("snapshot_paper_replay_ok", 1.0)
            context.setdefault("snapshot_paper_replay_hash_match", 1.0)
            return context, meta
    except Exception:
        pass

    health = project_root / "governance" / "health"

    coverage = _safe_load_json(health / "snapshot_coverage_latest.json", default={})
    replay = _safe_load_json(health / "replay_preopen_sanity_latest.json", default={})
    replay_e2e = _safe_load_json(health / "replay_end_to_end_latest.json", default={})
    paper_replay = _safe_load_json(health / "paper_replay_drill_latest.json", default={})
    drift = _safe_load_json(health / "preopen_replay_drift_latest.json", default={})
    divergence, _ = _fresh_health_payload(
        _safe_load_json(health / "data_source_divergence_latest.json", default={}),
        max_age_hours=float(os.getenv("TRADE_BEHAVIOR_DIVERGENCE_MAX_AGE_HOURS", "8")),
    )
    triprate = _safe_load_json(health / "guardrail_triprate_latest.json", default={})
    queue_stress = _safe_load_json(health / "execution_queue_stress_latest.json", default={})
    drill = _safe_load_json(project_root / "exports" / "state_snapshot_drills" / "latest.json", default={})

    coverage_ratio = _to_float(coverage.get("coverage_ratio"), 0.0)
    coverage_log_ratio = _clamp01(math.log1p(max(coverage_ratio, 0.0)) / 6.0)

    replay_decision_stale = _to_float((replay.get("decision") or {}).get("stale_windows"), 0.0)
    replay_governance_stale = _to_float((replay.get("governance") or {}).get("stale_windows"), 0.0)
    replay_max_decision_stale = max(_to_float((replay.get("thresholds") or {}).get("max_decision_stale_windows"), 12.0), 1.0)
    replay_max_governance_stale = max(_to_float((replay.get("thresholds") or {}).get("max_governance_stale_windows"), 12.0), 1.0)
    replay_stale_ratio = _clamp01((replay_decision_stale + replay_governance_stale) / (replay_max_decision_stale + replay_max_governance_stale))

    drift_obj = drift.get("drift") or {}
    thresholds_obj = drift.get("thresholds") or {}
    row_drift = max(abs(_to_float(drift_obj.get("decision_rows"), 0.0)), abs(_to_float(drift_obj.get("governance_rows"), 0.0)))
    stale_drift = max(abs(_to_float(drift_obj.get("decision_stale"), 0.0)), abs(_to_float(drift_obj.get("governance_stale"), 0.0)))
    max_row_drift = max(_to_float(thresholds_obj.get("max_row_drift"), 1.2), 1e-6)
    max_stale_drift = max(_to_float(thresholds_obj.get("max_stale_drift"), 1.0), 1e-6)
    replay_drift_ratio = _clamp01((0.6 * (row_drift / max_row_drift)) + (0.4 * (stale_drift / max_stale_drift)))

    worst_spread = _to_float(divergence.get("worst_relative_spread"), 0.0)
    max_spread = max(_to_float(divergence.get("max_relative_spread"), 0.03), 1e-6)
    divergence_ratio = _clamp01(worst_spread / max_spread)

    trip_rate = _to_float(triprate.get("trip_rate"), 0.0)
    max_trip_rate = max(_to_float(triprate.get("max_trip_rate"), 0.4), 1e-6)
    triprate_ratio = _clamp01(trip_rate / max_trip_rate)

    depth_seen = _to_float(queue_stress.get("max_queue_depth_seen"), 0.0)
    depth_max = max(_to_float(queue_stress.get("max_queue_depth"), 2000.0), 1.0)
    depth_ratio = _clamp01(depth_seen / depth_max)
    breach_rate = _to_float(queue_stress.get("queue_breach_rate"), 0.0)
    breach_rate_max = max(_to_float(queue_stress.get("max_queue_breach_rate"), 0.25), 1e-6)
    breach_ratio = _clamp01(breach_rate / breach_rate_max)
    queue_pressure_ratio = _clamp01(max(depth_ratio, breach_ratio))

    drill_files_checked = max(_to_float(drill.get("files_checked"), 0.0), 0.0)
    drill_missing_files = drill.get("missing_files") if isinstance(drill.get("missing_files"), list) else []
    drill_missing_count = float(len(drill_missing_files))
    drill_missing_ratio = _clamp01(drill_missing_count / max(drill_files_checked + drill_missing_count, 1.0))

    drill_rows = drill.get("rows") if isinstance(drill.get("rows"), list) else []
    drill_restore_total = float(len(drill_rows))
    drill_restore_ok = float(
        sum(1 for row in drill_rows if isinstance(row, dict) and bool(row.get("restore_ok", False)))
    )
    drill_restore_fail_ratio = _clamp01((drill_restore_total - drill_restore_ok) / max(drill_restore_total, 1.0))

    drill_ts = _parse_ts(drill.get("timestamp_utc"))
    if drill_ts is not None:
        drill_age_hours = max((datetime.now(timezone.utc) - drill_ts).total_seconds() / 3600.0, 0.0)
        drill_recency_norm = 1.0 - _clamp01(drill_age_hours / 72.0)
    else:
        drill_recency_norm = 0.0

    canary_weight_cap_norm = _clamp01(_to_float(os.getenv("CANARY_MAX_WEIGHT", "0.08"), 0.08) / 0.20)

    e2e_hash_match = replay_e2e.get("hash_match")
    if e2e_hash_match is None:
        e2e_hash_match = True
    paper_hash_match = paper_replay.get("hash_match")
    if paper_hash_match is None:
        paper_hash_match = True

    context = {
        "snapshot_cov_ok": 1.0 if bool(coverage.get("ok", False)) else 0.0,
        "snapshot_cov_log_ratio": coverage_log_ratio,
        "snapshot_replay_stale_ratio": replay_stale_ratio,
        "snapshot_replay_drift_ratio": replay_drift_ratio,
        "snapshot_divergence_ratio": divergence_ratio,
        "snapshot_triprate_ratio": triprate_ratio,
        "snapshot_queue_pressure_ratio": queue_pressure_ratio,
        "snapshot_drill_ok": 1.0 if bool(drill.get("ok", False)) else 0.0,
        "snapshot_drill_restore_fail_ratio": drill_restore_fail_ratio,
        "snapshot_drill_missing_ratio": drill_missing_ratio,
        "snapshot_drill_recency_norm": drill_recency_norm,
        "canary_weight_cap_norm": canary_weight_cap_norm,
        "snapshot_e2e_replay_ok": 1.0 if bool(replay_e2e.get("ok", True)) else 0.0,
        "snapshot_e2e_hash_match": 1.0 if bool(e2e_hash_match) else 0.0,
        "snapshot_paper_replay_ok": 1.0 if bool(paper_replay.get("ok", True)) else 0.0,
        "snapshot_paper_replay_hash_match": 1.0 if bool(paper_hash_match) else 0.0,
    }
    meta = {
        "coverage_ts": coverage.get("timestamp_utc"),
        "replay_ts": replay.get("timestamp_utc"),
        "replay_end_to_end_ts": replay_e2e.get("timestamp_utc"),
        "paper_replay_ts": paper_replay.get("timestamp_utc"),
        "drift_ts": drift.get("timestamp_utc"),
        "divergence_ts": divergence.get("timestamp_utc"),
        "triprate_ts": triprate.get("timestamp_utc"),
        "queue_stress_ts": queue_stress.get("timestamp_utc"),
        "state_snapshot_drill_ts": drill.get("timestamp_utc"),
    }
    return context, meta


def _try_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
        if math.isfinite(num):
            return num
    except Exception:
        pass
    return None


def _latest_two_numeric(rows: Any, *, value_key: str = "value") -> Tuple[Optional[float], Optional[float]]:
    latest: Optional[float] = None
    prev: Optional[float] = None
    if not isinstance(rows, list):
        return latest, prev
    for row in rows:
        if not isinstance(row, dict):
            continue
        num = _try_float(row.get(value_key))
        if num is None:
            continue
        if latest is None:
            latest = num
            continue
        prev = num
        break
    return latest, prev


def _pct_change(latest: Optional[float], prev: Optional[float]) -> float:
    if latest is None or prev is None:
        return 0.0
    if abs(prev) <= 1e-9:
        return 0.0
    return (latest - prev) / abs(prev)


def _signed_pct_norm(pct_change: float, gain: float) -> float:
    return _clamp01(0.5 + (0.5 * _signed_scale(float(pct_change), float(gain))))


def _safe_mean(values: List[Optional[float]]) -> float:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not nums:
        return 0.0
    return sum(nums) / float(len(nums))


def _payload_recency_norm(payload: Dict[str, Any], *, now_utc: datetime, max_age_hours: float) -> float:
    ts = _parse_ts(payload.get("timestamp_utc"))
    if ts is None:
        return 0.0
    age_hours = max((now_utc - ts).total_seconds() / 3600.0, 0.0)
    return 1.0 - _clamp01(age_hours / max(max_age_hours, 1e-6))


def _load_latest_context_file(project_root: Path, category: str) -> Dict[str, Any]:
    token = str(category or "").strip().lower()
    if not token:
        return {}
    aliases = {
        "options_flow_context": ["options_flow_context", "tastytrade_context"],
        "tastytrade_context": ["options_flow_context", "tastytrade_context"],
    }
    candidates = aliases.get(token, [token])
    for candidate in candidates:
        for path in (
            project_root / "data" / "external_context" / f"{candidate}_latest.json",
            project_root / "exports" / "external_context" / f"{candidate}_latest.json",
            project_root / "governance" / "health" / f"{candidate}_latest.json",
        ):
            payload = _safe_load_json(path, default={})
            if payload:
                return payload
    return {}


def _global_feature_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    derived = payload.get("derived") if isinstance(payload.get("derived"), dict) else {}
    return derived.get("global_features") if isinstance(derived.get("global_features"), dict) else {}


def _feature_signal(node: Dict[str, Any], keys: List[str], *, scale: float = 1.0) -> float:
    values = [_try_float(node.get(key)) for key in keys]
    return _clamp01(_safe_mean(values) / max(scale, 1e-6))


def _collector_quality_score(rows: List[Dict[str, Any]], name: str) -> float:
    token = str(name or "").strip().lower()
    if not token:
        return 0.0
    for row in rows:
        if str(row.get("name") or "").strip().lower() != token:
            continue
        return _clamp01(_to_float(row.get("quality_score"), 0.0))
    return 0.0


def _verification_status_score(status: str) -> float:
    token = str(status or "").strip().lower()
    if token == "cross_verified":
        return 1.0
    if token == "single_source_verified":
        return 0.75
    if token == "single_source_unverified":
        return 0.25
    return 0.0


def _external_feeds_context(project_root: Path, now_utc: datetime) -> Tuple[Dict[str, float], Dict[str, Any]]:
    root = project_root / "exports" / "external_feeds"
    status = _safe_load_json(root / "latest_status.json", default={})
    bls = _safe_load_json(root / "bls" / "latest.json", default={})
    census = _safe_load_json(root / "census" / "latest.json", default={})
    fred = _safe_load_json(root / "fred" / "latest.json", default={})
    bea = _safe_load_json(root / "bea" / "latest.json", default={})
    tradingeconomics = _safe_load_json(root / "tradingeconomics" / "latest.json", default={})
    market_micro = _load_latest_context_file(project_root, "market_micro")
    options_flow = _load_latest_context_file(project_root, "options_flow_context")
    crypto_market = _load_latest_context_file(project_root, "crypto_market_context")
    market_crypto_correlation = _load_latest_context_file(project_root, "market_crypto_correlation")
    fx_market = _load_latest_context_file(project_root, "fx_market_context")
    dividend_drip = _load_latest_context_file(project_root, "dividend_drip_state")
    sec_edgar = _load_latest_context_file(project_root, "sec_edgar")
    extended_quant = _load_latest_context_file(project_root, "extended_quant_context")
    official_macro = _load_latest_context_file(project_root, "official_macro_context")
    central_bank_cross_source = _load_latest_context_file(project_root, "central_bank_cross_source")
    decision_context_mesh = _load_latest_context_file(project_root, "decision_context_mesh")
    schwab_education = _load_latest_context_file(project_root, "schwab_education_context")
    live_macro = _load_latest_context_file(project_root, "live_macro")
    market_breadth_snapshot = _load_latest_context_file(project_root, "market_breadth")
    bond_reference_snapshot = _load_latest_context_file(project_root, "bond_reference")
    collector_contracts = _safe_load_json(project_root / "governance" / "health" / "collector_contracts_latest.json", default={})
    source_verification = _safe_load_json(project_root / "governance" / "health" / "source_verification_latest.json", default={})

    provider_names = ("bls", "census", "fred", "bea")
    provider_ok = {}
    for name in provider_names:
        node = status.get(name) if isinstance(status.get(name), dict) else {}
        provider_ok[name] = bool(node.get("ok", False))

    status_ts = _parse_ts(status.get("timestamp_utc"))
    te_status = tradingeconomics.get("status") if isinstance(tradingeconomics.get("status"), dict) else {}
    te_ts = _parse_ts(tradingeconomics.get("timestamp_utc"))
    te_ok = bool(te_status.get("ok", False)) or int(_to_float(te_status.get("datasets_ok_count"), 0.0)) > 0
    provider_ok["tradingeconomics"] = te_ok
    latest_status_ts = max([ts for ts in (status_ts, te_ts) if ts is not None], default=None)
    if latest_status_ts is not None:
        age_hours = max((now_utc - latest_status_ts).total_seconds() / 3600.0, 0.0)
        recency_norm = 1.0 - _clamp01(age_hours / 72.0)
    else:
        recency_norm = 0.0

    fred_responses = fred.get("responses") if isinstance(fred.get("responses"), dict) else {}
    fred_unrate_latest, _ = _latest_two_numeric((fred_responses.get("UNRATE") or {}).get("observations"))
    fred_cpi_latest, fred_cpi_prev = _latest_two_numeric((fred_responses.get("CPIAUCSL") or {}).get("observations"))
    fred_gdp_latest, fred_gdp_prev = _latest_two_numeric((fred_responses.get("GDP") or {}).get("observations"))
    fred_cpi_mom = _pct_change(fred_cpi_latest, fred_cpi_prev)
    fred_gdp_qoq = _pct_change(fred_gdp_latest, fred_gdp_prev)

    bls_series = ((bls.get("response") or {}).get("Results") or {}).get("series")
    bls_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(bls_series, list):
        for row in bls_series:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("seriesID") or "").strip().upper()
            if sid:
                bls_map[sid] = row
    bls_unrate_latest, _ = _latest_two_numeric((bls_map.get("LNS14000000") or {}).get("data"))
    bls_cpi_latest, bls_cpi_prev = _latest_two_numeric((bls_map.get("CUUR0000SA0") or {}).get("data"))
    bls_cpi_mom = _pct_change(bls_cpi_latest, bls_cpi_prev)

    te_derived = tradingeconomics.get("derived") if isinstance(tradingeconomics.get("derived"), dict) else {}
    te_macro_backfill = te_derived.get("macro_backfill") if isinstance(te_derived.get("macro_backfill"), dict) else {}
    te_unrate_latest = _try_float(te_macro_backfill.get("unemployment_rate_latest"))
    te_inflation_mom = _try_float(te_macro_backfill.get("inflation_mom_ratio"))
    te_gdp_qoq = _try_float(te_macro_backfill.get("gdp_qoq_ratio"))

    te_backfill_used = {
        "fred_unrate": False,
        "fred_cpi_mom": False,
        "fred_gdp_qoq": False,
        "bls_unrate": False,
        "bls_cpi_mom": False,
    }

    if fred_unrate_latest is None and te_unrate_latest is not None:
        fred_unrate_latest = te_unrate_latest
        te_backfill_used["fred_unrate"] = True
    if abs(fred_cpi_mom) <= 1e-12 and te_inflation_mom is not None:
        fred_cpi_mom = te_inflation_mom
        te_backfill_used["fred_cpi_mom"] = True
    if abs(fred_gdp_qoq) <= 1e-12 and te_gdp_qoq is not None:
        fred_gdp_qoq = te_gdp_qoq
        te_backfill_used["fred_gdp_qoq"] = True
    if bls_unrate_latest is None and te_unrate_latest is not None:
        bls_unrate_latest = te_unrate_latest
        te_backfill_used["bls_unrate"] = True
    if abs(bls_cpi_mom) <= 1e-12 and te_inflation_mom is not None:
        bls_cpi_mom = te_inflation_mom
        te_backfill_used["bls_cpi_mom"] = True

    census_population: Optional[float] = None
    census_rows = census.get("response")
    if isinstance(census_rows, list) and len(census_rows) >= 2:
        header = census_rows[0] if isinstance(census_rows[0], list) else []
        values = census_rows[1] if isinstance(census_rows[1], list) else []
        if header and values:
            try:
                idx = [str(x) for x in header].index("B01001_001E")
                if idx < len(values):
                    census_population = _try_float(values[idx])
            except ValueError:
                census_population = None

    bea_dataset_rows = ((((bea.get("response") or {}).get("BEAAPI") or {}).get("Results") or {}).get("Dataset"))
    bea_dataset_count = float(len(bea_dataset_rows)) if isinstance(bea_dataset_rows, list) else 0.0
    micro_derived = market_micro.get("derived") if isinstance(market_micro.get("derived"), dict) else {}
    micro_global = micro_derived.get("global_features") if isinstance(micro_derived.get("global_features"), dict) else {}
    tasty_derived = options_flow.get("derived") if isinstance(options_flow.get("derived"), dict) else {}
    tasty_global = tasty_derived.get("global_features") if isinstance(tasty_derived.get("global_features"), dict) else {}
    crypto_derived = crypto_market.get("derived") if isinstance(crypto_market.get("derived"), dict) else {}
    crypto_global = crypto_derived.get("global_features") if isinstance(crypto_derived.get("global_features"), dict) else {}
    market_crypto_derived = market_crypto_correlation.get("derived") if isinstance(market_crypto_correlation.get("derived"), dict) else {}
    market_crypto_global = market_crypto_derived.get("global_features") if isinstance(market_crypto_derived.get("global_features"), dict) else {}
    fx_derived = fx_market.get("derived") if isinstance(fx_market.get("derived"), dict) else {}
    fx_global = fx_derived.get("global_features") if isinstance(fx_derived.get("global_features"), dict) else {}
    dividend_drip_derived = dividend_drip.get("derived") if isinstance(dividend_drip.get("derived"), dict) else {}
    dividend_drip_global = dividend_drip_derived.get("global_features") if isinstance(dividend_drip_derived.get("global_features"), dict) else {}
    dividend_drip_symbol = dividend_drip_derived.get("symbol_features") if isinstance(dividend_drip_derived.get("symbol_features"), dict) else {}
    sec_global = _global_feature_map(sec_edgar)
    extended_quant_global = _global_feature_map(extended_quant)
    official_macro_derived = official_macro.get("derived") if isinstance(official_macro.get("derived"), dict) else {}
    official_macro_news = official_macro_derived.get("news_features") if isinstance(official_macro_derived.get("news_features"), dict) else {}
    official_macro_calendar = official_macro_derived.get("calendar_features") if isinstance(official_macro_derived.get("calendar_features"), dict) else {}
    official_macro_global = official_macro_derived.get("global_features") if isinstance(official_macro_derived.get("global_features"), dict) else {}
    central_bank_cross_derived = central_bank_cross_source.get("derived") if isinstance(central_bank_cross_source.get("derived"), dict) else {}
    central_bank_cross_global = central_bank_cross_derived.get("global_features") if isinstance(central_bank_cross_derived.get("global_features"), dict) else {}
    central_bank_cross_symbol = central_bank_cross_derived.get("symbol_features") if isinstance(central_bank_cross_derived.get("symbol_features"), dict) else {}
    decision_context_mesh_derived = decision_context_mesh.get("derived") if isinstance(decision_context_mesh.get("derived"), dict) else {}
    decision_context_mesh_global = decision_context_mesh_derived.get("global_features") if isinstance(decision_context_mesh_derived.get("global_features"), dict) else {}
    decision_context_mesh_symbol = decision_context_mesh_derived.get("symbol_features") if isinstance(decision_context_mesh_derived.get("symbol_features"), dict) else {}
    education_global = _global_feature_map(schwab_education)
    live_macro_derived = live_macro.get("derived") if isinstance(live_macro.get("derived"), dict) else {}
    live_macro_news = live_macro_derived.get("news_features") if isinstance(live_macro_derived.get("news_features"), dict) else {}
    live_macro_calendar = live_macro_derived.get("calendar_features") if isinstance(live_macro_derived.get("calendar_features"), dict) else {}
    te_market_breadth = te_derived.get("market_breadth") if isinstance(te_derived.get("market_breadth"), dict) else {}
    te_bond_reference = te_derived.get("bond_reference") if isinstance(te_derived.get("bond_reference"), dict) else {}
    breadth_payload = market_breadth_snapshot if market_breadth_snapshot else {"derived": {"global_features": te_market_breadth}}
    bond_payload = bond_reference_snapshot if bond_reference_snapshot else {"derived": {"global_features": te_bond_reference}}
    breadth_global = _global_feature_map(breadth_payload)
    bond_global = _global_feature_map(bond_payload)
    collector_rows = collector_contracts.get("rows") if isinstance(collector_contracts.get("rows"), list) else []
    verification_overall = source_verification.get("overall") if isinstance(source_verification.get("overall"), dict) else {}
    verification_counts = verification_overall.get("counts") if isinstance(verification_overall.get("counts"), dict) else {}
    verification_rows = source_verification.get("sources") if isinstance(source_verification.get("sources"), list) else []
    verification_by_id = {
        str(row.get("source_id") or "").strip().lower(): row
        for row in verification_rows
        if isinstance(row, dict) and str(row.get("source_id") or "").strip()
    }
    total_verification_sources = max(int(_to_float(verification_overall.get("total_sources"), 0.0)), len(verification_rows), 1)
    live_macro_confidence = _safe_mean(
        [
            _try_float(live_macro_news.get("news_source_quality_norm")),
            _try_float(live_macro_news.get("news_entity_relevance_norm")),
            _try_float(live_macro_calendar.get("calendar_high_impact_24h_norm")),
            _try_float(live_macro_calendar.get("calendar_macro_event_norm")),
            _payload_recency_norm(live_macro, now_utc=now_utc, max_age_hours=48.0),
        ]
    )

    context = {
        "external_feeds_ok": 1.0 if (all(provider_ok.get(name, False) for name in provider_names) or te_ok) else 0.0,
        "external_feeds_recency_norm": recency_norm,
        "external_fred_unrate_norm": _clamp01((_try_float(fred_unrate_latest) or 0.0) / 12.0),
        "external_fred_cpi_mom_norm": _signed_pct_norm(fred_cpi_mom, 120.0),
        "external_fred_gdp_qoq_norm": _signed_pct_norm(fred_gdp_qoq, 30.0),
        "external_bls_unrate_norm": _clamp01((_try_float(bls_unrate_latest) or 0.0) / 12.0),
        "external_bls_cpi_mom_norm": _signed_pct_norm(bls_cpi_mom, 120.0),
        "external_census_population_log_norm": _clamp01(math.log1p(max((_try_float(census_population) or 0.0), 0.0)) / 21.0),
        "external_bea_dataset_count_norm": _clamp01(bea_dataset_count / 30.0),
        "external_micro_auction_norm": _clamp01(_to_float(micro_global.get("market_micro_opening_auction_norm"), 0.0)),
        "external_micro_relative_volume_norm": _clamp01(_to_float(micro_global.get("market_micro_relative_volume_norm"), 0.0)),
        "external_micro_options_flow_norm": _clamp01(_to_float(micro_global.get("market_micro_options_flow_norm"), 0.0)),
        "external_micro_short_pressure_norm": _clamp01(_to_float(micro_global.get("market_micro_short_pressure_norm"), 0.0)),
        "external_micro_credit_flow_norm": _clamp01(_to_float(micro_global.get("market_micro_credit_flow_norm"), 0.0)),
        "external_micro_block_trade_norm": _clamp01(_to_float(micro_global.get("market_micro_block_trade_norm"), 0.0)),
        "tasty_iv_rank_norm": _clamp01(_to_float(tasty_global.get("tasty_iv_rank_norm"), 0.0)),
        "tasty_implied_volatility_index_norm": _clamp01(_to_float(tasty_global.get("tasty_implied_volatility_index_norm"), 0.0)),
        "tasty_liquidity_rating_norm": _clamp01(_to_float(tasty_global.get("tasty_liquidity_rating_norm"), 0.0)),
        "tasty_expected_move_norm": _clamp01(_to_float(tasty_global.get("tasty_expected_move_norm"), 0.0)),
        "tasty_beta_norm": _clamp01(_to_float(tasty_global.get("tasty_beta_norm"), 0.0)),
        "tasty_watchlist_presence_norm": _clamp01(_to_float(tasty_global.get("tasty_watchlist_presence_norm"), 0.0)),
        "options_iv_crush_risk_norm": _clamp01(_to_float(tasty_global.get("options_iv_crush_risk_norm"), 0.0)),
        "options_assignment_risk_norm": _clamp01(_to_float(tasty_global.get("options_assignment_risk_norm"), 0.0)),
        "options_zero_dte_regime_norm": _clamp01(_to_float(tasty_global.get("options_zero_dte_regime_norm"), 0.0)),
        "options_vol_of_vol_change_norm": _clamp01(_to_float(tasty_global.get("options_vol_of_vol_change_norm"), 0.0)),
        "options_spread_execution_risk_norm": _clamp01(_to_float(tasty_global.get("options_spread_execution_risk_norm"), 0.0)),
        "options_higher_order_greek_pressure_norm": _clamp01(_to_float(tasty_global.get("options_higher_order_greek_pressure_norm"), 0.0)),
        "options_barrier_touch_risk_norm": _clamp01(_to_float(tasty_global.get("options_barrier_touch_risk_norm"), 0.0)),
        "options_lookback_path_dependency_norm": _clamp01(_to_float(tasty_global.get("options_lookback_path_dependency_norm"), 0.0)),
        "options_variance_swap_proxy_norm": _clamp01(_to_float(tasty_global.get("options_variance_swap_proxy_norm"), 0.0)),
        "options_volatility_swap_proxy_norm": _clamp01(_to_float(tasty_global.get("options_volatility_swap_proxy_norm"), 0.0)),
        "options_gamma_scalping_pressure_norm": _clamp01(_to_float(tasty_global.get("options_gamma_scalping_pressure_norm"), 0.0)),
        "options_vanna_volga_hedge_pressure_norm": _clamp01(_to_float(tasty_global.get("options_vanna_volga_hedge_pressure_norm"), 0.0)),
        "options_dispersion_trade_proxy_norm": _clamp01(_to_float(tasty_global.get("options_dispersion_trade_proxy_norm"), 0.0)),
        "options_volatility_arbitrage_proxy_norm": _clamp01(_to_float(tasty_global.get("options_volatility_arbitrage_proxy_norm"), 0.0)),
        "crypto_deribit_mark_iv_norm": _clamp01(_to_float(crypto_global.get("crypto_deribit_mark_iv_norm"), 0.0)),
        "crypto_hyperliquid_funding_norm": _clamp01(_to_float(crypto_global.get("crypto_hyperliquid_funding_norm"), 0.0)),
        "crypto_coingecko_momentum_norm": _clamp01(_to_float(crypto_global.get("crypto_coingecko_momentum_norm"), 0.0)),
        "crypto_cross_provider_price_agreement_norm": _clamp01(_to_float(crypto_global.get("crypto_cross_provider_price_agreement_norm"), 0.0)),
        "crypto_defillama_stablecoin_growth_norm": _clamp01(_to_float(crypto_global.get("crypto_defillama_stablecoin_growth_norm"), 0.0)),
        "crypto_defillama_dex_volume_growth_norm": _clamp01(_to_float(crypto_global.get("crypto_defillama_dex_volume_growth_norm"), 0.0)),
        "crypto_etherscan_gas_norm": _clamp01(_to_float(crypto_global.get("crypto_etherscan_gas_norm"), 0.0)),
        "market_crypto_risk_corr_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_risk_corr_norm"), 0.0)),
        "market_crypto_spy_corr_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_spy_corr_norm"), 0.0)),
        "market_crypto_qqq_corr_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_qqq_corr_norm"), 0.0)),
        "market_crypto_tlt_corr_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_tlt_corr_norm"), 0.0)),
        "market_crypto_uup_inverse_corr_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_uup_inverse_corr_norm"), 0.0)),
        "market_crypto_gold_corr_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_gold_corr_norm"), 0.0)),
        "market_crypto_current_alignment_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_current_alignment_norm"), 0.0)),
        "market_crypto_divergence_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_divergence_norm"), 0.0)),
        "market_crypto_corr_confidence_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_corr_confidence_norm"), 0.0)),
        "market_crypto_sleeve_coverage_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_sleeve_coverage_norm"), 0.0)),
        "market_crypto_sleeve_avg_abs_corr_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_sleeve_avg_abs_corr_norm"), 0.0)),
        "market_crypto_sleeve_dispersion_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_sleeve_dispersion_norm"), 0.0)),
        "market_crypto_sleeve_confidence_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_sleeve_confidence_norm"), 0.0)),
        "market_crypto_risk_on_crypto_alignment_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_risk_on_crypto_alignment_norm"), 0.0)),
        "market_crypto_fx_crypto_inverse_corr_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_fx_crypto_inverse_corr_norm"), 0.0)),
        "market_crypto_rates_crypto_corr_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_rates_crypto_corr_norm"), 0.0)),
        "market_crypto_energy_crypto_corr_norm": _clamp01(_to_float(market_crypto_global.get("market_crypto_energy_crypto_corr_norm"), 0.0)),
        "fx_official_data_available": _clamp01(_to_float(fx_global.get("fx_official_data_available"), 0.0)),
        "fx_eurusd_level_norm": _clamp01(_to_float(fx_global.get("fx_eurusd_level_norm"), 0.0)),
        "fx_eurusd_momentum_norm": _clamp01(_to_float(fx_global.get("fx_eurusd_momentum_norm"), 0.0)),
        "fx_usdjpy_level_norm": _clamp01(_to_float(fx_global.get("fx_usdjpy_level_norm"), 0.0)),
        "fx_usdjpy_momentum_norm": _clamp01(_to_float(fx_global.get("fx_usdjpy_momentum_norm"), 0.0)),
        "fx_gbpusd_level_norm": _clamp01(_to_float(fx_global.get("fx_gbpusd_level_norm"), 0.0)),
        "fx_gbpusd_momentum_norm": _clamp01(_to_float(fx_global.get("fx_gbpusd_momentum_norm"), 0.0)),
        "fx_usd_strength_norm": _clamp01(_to_float(fx_global.get("fx_usd_strength_norm"), 0.0)),
        "fx_usd_broad_index_norm": _clamp01(_to_float(fx_global.get("fx_usd_broad_index_norm"), 0.0)),
        "fx_proxy_agreement_norm": _clamp01(_to_float(fx_global.get("fx_proxy_agreement_norm"), 0.0)),
        "fx_risk_on_alignment_norm": _clamp01(_to_float(fx_global.get("fx_risk_on_alignment_norm"), 0.0)),
        "fx_crypto_alignment_norm": _clamp01(_to_float(fx_global.get("fx_crypto_alignment_norm"), 0.0)),
        "fx_macro_dispersion_norm": _clamp01(_to_float(fx_global.get("fx_macro_dispersion_norm"), 0.0)),
        "fx_corr_confidence_norm": _clamp01(_to_float(fx_global.get("fx_corr_confidence_norm"), 0.0)),
        "fx_session_asia_norm": _clamp01(_to_float(fx_global.get("fx_session_asia_norm"), 0.0)),
        "fx_session_london_norm": _clamp01(_to_float(fx_global.get("fx_session_london_norm"), 0.0)),
        "fx_session_ny_norm": _clamp01(_to_float(fx_global.get("fx_session_ny_norm"), 0.0)),
        "fx_rollover_risk_norm": _clamp01(_to_float(fx_global.get("fx_rollover_risk_norm"), 0.0)),
        "fx_dxy_yield_confirmation_norm": _clamp01(_to_float(fx_global.get("fx_dxy_yield_confirmation_norm"), 0.0)),
        "fx_carry_proxy_norm": _clamp01(_to_float(fx_global.get("fx_carry_proxy_norm"), 0.0)),
        "dividend_drip_active_norm": _clamp01(_to_float(dividend_drip_global.get("dividend_drip_active_norm"), 0.0)),
        "dividend_drip_recent_reinvest_norm": _clamp01(_to_float(dividend_drip_global.get("dividend_drip_recent_reinvest_norm"), 0.0)),
        "dividend_drip_cash_only_norm": _clamp01(_to_float(dividend_drip_global.get("dividend_drip_cash_only_norm"), 0.0)),
        "dividend_drip_share_credit_norm": _clamp01(_to_float(dividend_drip_global.get("dividend_drip_share_credit_norm"), 0.0)),
        "dividend_drip_event_recency_norm": _clamp01(_to_float(dividend_drip_global.get("dividend_drip_event_recency_norm"), 0.0)),
        "dividend_drip_confidence_norm": _clamp01(_to_float(dividend_drip_global.get("dividend_drip_confidence_norm"), 0.0)),
        "live_macro_gate_active_norm": 1.0 if live_macro_confidence > 0.0 else 0.0,
        "live_macro_gate_confidence_norm": _clamp01(live_macro_confidence),
        "sec_context_signal_norm": _feature_signal(
            sec_global,
            [
                "sec_recent_filings_1d_norm",
                "sec_recent_high_impact_1d_norm",
                "sec_offering_7d_norm",
                "sec_dilution_7d_norm",
                "sec_mna_7d_norm",
                "sec_financing_stress_7d_norm",
            ],
        ),
        "extended_quant_signal_norm": _feature_signal(
            extended_quant_global,
            [
                "cot_macro_positioning_stress_norm",
                "sofr_term_pressure_norm",
                "sofr_funding_stress_norm",
                "cboe_put_call_stress_norm",
                "short_threshold_total_listed_norm",
                "short_ftd_total_hits_norm",
            ],
        ),
        "official_macro_signal_norm": _clamp01(
            _safe_mean(
                [
                    _try_float(official_macro_calendar.get("calendar_high_impact_24h_norm")),
                    _try_float(official_macro_calendar.get("calendar_macro_event_norm")),
                    _try_float(official_macro_calendar.get("calendar_macro_abs_surprise_norm")),
                    _try_float(official_macro_news.get("news_source_quality_norm")),
                ]
            )
        ),
        "schwab_education_signal_norm": _feature_signal(
            education_global,
            [
                "schwab_education_item_density_norm",
                "schwab_education_recent_activity_norm",
                "schwab_education_symbol_coverage_norm",
                "schwab_education_video_share_norm",
                "schwab_education_stream_share_norm",
                "schwab_education_network_share_norm",
            ],
        ),
        "market_breadth_signal_norm": _clamp01(
            max(
                _to_float(breadth_global.get("breadth_thrust_norm"), 0.0),
                _to_float(breadth_global.get("breadth_sector_rotation_norm"), 0.0),
                _to_float(breadth_global.get("breadth_risk_off_norm"), 0.0),
                _clamp01(abs(_to_float(te_market_breadth.get("index_alignment_score"), 0.0))),
            )
        ),
        "bond_reference_signal_norm": _clamp01(
            max(
                _to_float(bond_global.get("bond_curve_2s10s_norm"), 0.0),
                _to_float(bond_global.get("bond_auction_window_norm"), 0.0),
                _to_float(bond_global.get("bond_auction_tail_norm"), 0.0),
                _clamp01(abs(_to_float(te_bond_reference.get("curve_regime_score"), 0.0))),
            )
        ),
        "source_quality_average_score_norm": _clamp01(_to_float(collector_contracts.get("average_quality_score"), 0.0)),
        "source_quality_required_failure_ratio_norm": _clamp01(
            _to_float(collector_contracts.get("required_failure_count"), 0.0) / max(_to_float(collector_contracts.get("collector_count"), 0.0), 1.0)
        ),
        "source_quality_soft_failure_ratio_norm": _clamp01(
            _to_float(collector_contracts.get("soft_failure_count"), 0.0) / max(_to_float(collector_contracts.get("collector_count"), 0.0), 1.0)
        ),
        "source_quality_unverified_ratio_norm": _clamp01(
            _to_float(len(verification_overall.get("unverified_sources") or []), 0.0) / float(total_verification_sources)
        ),
        "source_quality_cross_verified_ratio_norm": _clamp01(
            _to_float(verification_counts.get("cross_verified"), 0.0) / float(total_verification_sources)
        ),
        "source_quality_market_micro_score_norm": _collector_quality_score(collector_rows, "market_micro_context"),
        "source_quality_official_macro_score_norm": _collector_quality_score(collector_rows, "official_macro_context"),
        "source_quality_crypto_context_score_norm": _collector_quality_score(collector_rows, "crypto_market_context"),
    }
    meta = {
        "status_ts": status.get("timestamp_utc"),
        "provider_ok": provider_ok,
        "bls_ts": bls.get("timestamp_utc"),
        "census_ts": census.get("timestamp_utc"),
        "fred_ts": fred.get("timestamp_utc"),
        "bea_ts": bea.get("timestamp_utc"),
        "market_micro_ts": market_micro.get("timestamp_utc"),
        "tastytrade_ts": options_flow.get("timestamp_utc"),
        "crypto_market_ts": crypto_market.get("timestamp_utc"),
        "market_crypto_correlation_ts": market_crypto_correlation.get("timestamp_utc"),
        "fx_market_context_ts": fx_market.get("timestamp_utc"),
        "dividend_drip_state_ts": dividend_drip.get("timestamp_utc"),
        "sec_edgar_ts": sec_edgar.get("timestamp_utc"),
        "extended_quant_context_ts": extended_quant.get("timestamp_utc"),
        "official_macro_context_ts": official_macro.get("timestamp_utc"),
        "central_bank_cross_source_ts": central_bank_cross_source.get("timestamp_utc"),
        "decision_context_mesh_ts": decision_context_mesh.get("timestamp_utc"),
        "schwab_education_context_ts": schwab_education.get("timestamp_utc"),
        "live_macro_ts": live_macro.get("timestamp_utc"),
        "market_breadth_ts": market_breadth_snapshot.get("timestamp_utc") if isinstance(market_breadth_snapshot, dict) else None,
        "bond_reference_ts": bond_reference_snapshot.get("timestamp_utc") if isinstance(bond_reference_snapshot, dict) else None,
        "tradingeconomics_ts": tradingeconomics.get("timestamp_utc"),
        "market_micro": micro_global,
        "tastytrade": tasty_global,
        "crypto_market": crypto_global,
        "market_crypto_correlation": market_crypto_global,
        "fx_market": fx_global,
        "dividend_drip": dividend_drip_global,
        "dividend_drip_symbol_features": dividend_drip_symbol,
        "sec_edgar": sec_global,
        "extended_quant_context": extended_quant_global,
        "official_macro_context": {
            "news_features": official_macro_news,
            "calendar_features": official_macro_calendar,
            "global_features": official_macro_global,
        },
        "central_bank_cross_source": {
            "global_features": central_bank_cross_global,
            "symbol_features": central_bank_cross_symbol,
            "routing": central_bank_cross_source.get("routing", {}),
        },
        "decision_context_mesh": {
            "global_features": decision_context_mesh_global,
            "symbol_features": decision_context_mesh_symbol,
            "grade_summary": decision_context_mesh.get("grade_summary", {}),
            "routing": decision_context_mesh.get("routing", {}),
        },
        "schwab_education_context": education_global,
        "live_macro": {
            "news_features": live_macro_news,
            "calendar_features": live_macro_calendar,
            "confidence_norm": _clamp01(live_macro_confidence),
        },
        "market_breadth": breadth_global if breadth_global else te_market_breadth,
        "bond_reference": bond_global if bond_global else te_bond_reference,
        "collector_contracts": {
            "average_quality_score": _to_float(collector_contracts.get("average_quality_score"), 0.0),
            "required_failure_count": int(_to_float(collector_contracts.get("required_failure_count"), 0.0)),
            "soft_failure_count": int(_to_float(collector_contracts.get("soft_failure_count"), 0.0)),
            "collector_count": int(_to_float(collector_contracts.get("collector_count"), 0.0)),
        },
        "source_verification": {
            "counts": verification_counts,
            "total_sources": total_verification_sources,
            "statuses": {
                key: str((row or {}).get("verification_status") or "")
                for key, row in verification_by_id.items()
            },
            "status_scores": {
                key: _verification_status_score(str((row or {}).get("verification_status") or ""))
                for key, row in verification_by_id.items()
            },
        },
        "tradingeconomics": {
            "ok": te_ok,
            "datasets_ok_count": int(_to_float(te_status.get("datasets_ok_count"), 0.0)),
            "macro_backfill": te_macro_backfill,
            "calendar_rows": te_derived.get("calendar_rows") if isinstance(te_derived.get("calendar_rows"), list) else [],
            "news_features": te_derived.get("news_features") if isinstance(te_derived.get("news_features"), dict) else {},
            "market_breadth": te_market_breadth,
            "bond_reference": te_bond_reference,
        },
        "raw": {
            "fred_unrate_latest": fred_unrate_latest,
            "fred_cpi_mom": fred_cpi_mom,
            "fred_gdp_qoq": fred_gdp_qoq,
            "bls_unrate_latest": bls_unrate_latest,
            "bls_cpi_mom": bls_cpi_mom,
            "census_population": census_population,
            "bea_dataset_count": int(bea_dataset_count),
            "tradingeconomics_unrate_latest": te_unrate_latest,
            "tradingeconomics_inflation_mom_ratio": te_inflation_mom,
            "tradingeconomics_gdp_qoq_ratio": te_gdp_qoq,
            "tradingeconomics_backfill_used": te_backfill_used,
        },
    }
    central_bank_contract = assess_central_bank_liquidity_context(official_macro)
    if bool(central_bank_contract.get("ready", False)):
        for key in CENTRAL_BANK_LIQUIDITY_FEATURE_KEYS:
            context[key] = _clamp01(_to_float(official_macro_global.get(key), 0.0))
    meta["central_bank_liquidity_contract"] = central_bank_contract
    global_central_bank_contract = assess_global_central_bank_context(official_macro)
    if bool(global_central_bank_contract.get("ready", False)):
        for key in GLOBAL_CENTRAL_BANK_FEATURE_KEYS:
            context[key] = _clamp01(_to_float(official_macro_global.get(key), 0.0))
    meta["global_central_bank_contract"] = global_central_bank_contract
    cross_source_contract = assess_central_bank_cross_source_context(central_bank_cross_source)
    if bool(cross_source_contract.get("ready", False)):
        for key in CENTRAL_BANK_CROSS_SOURCE_FEATURE_KEYS:
            context[key] = _clamp01(_to_float(central_bank_cross_global.get(key), 0.0))
    meta["central_bank_cross_source_contract"] = cross_source_contract
    decision_context_mesh_contract = assess_decision_context_mesh(decision_context_mesh, now_utc=now_utc)
    if bool(decision_context_mesh_contract.get("ready", False)):
        for key in DECISION_CONTEXT_MESH_FEATURE_KEYS:
            context[key] = _clamp01(_to_float(decision_context_mesh_global.get(key), 0.0))
    meta["decision_context_mesh_contract"] = decision_context_mesh_contract
    return context, meta




def _load_governance_index(rows: Iterable[Dict[str, Any]], since_utc: datetime) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        ts = _parse_ts(row.get("timestamp_utc"))
        if ts is None or ts < since_utc:
            continue
        sid = str(row.get("snapshot_id") or "").strip()
        if not sid:
            continue

        freshness = row.get("feature_freshness") or {}
        latency = row.get("master_latency_slo") or {}
        portfolio = row.get("portfolio") or {}
        cb = row.get("circuit_breakers") or {}
        exec_sim = row.get("execution_sim") or {}
        spec_votes = row.get("specialist_votes") if isinstance(row.get("specialist_votes"), dict) else {}
        lane_features = row.get("lane_strategy_features") if isinstance(row.get("lane_strategy_features"), dict) else {}
        capital_flow = row.get("capital_flow") if isinstance(row.get("capital_flow"), dict) else {}
        flow_awareness = row.get("flow_awareness_features") if isinstance(row.get("flow_awareness_features"), dict) else {}
        lead_lag = row.get("lead_lag_features") if isinstance(row.get("lead_lag_features"), dict) else {}
        allocation_conf = row.get("allocation_confidence") if isinstance(row.get("allocation_confidence"), dict) else {}

        out[sid] = {
            "active_sub_bots": _to_float(row.get("active_sub_bots_total"), _to_float(row.get("active_sub_bots"), 0.0)),
            "active_options_sub_bots": _to_float(row.get("active_options_sub_bots"), 0.0),
            "active_futures_sub_bots": _to_float(row.get("active_futures_sub_bots"), 0.0),
            "options_specialist_vote": _to_float(spec_votes.get("options"), 0.0),
            "futures_specialist_vote": _to_float(spec_votes.get("futures"), 0.0),
            "queue_depth": _to_float(portfolio.get("queue_depth"), 0.0),
            "dispatch_qty": _to_float(portfolio.get("dispatch_qty"), 0.0),
            "feature_freshness_ok": 1.0 if bool(freshness.get("ok", True)) else 0.0,
            "feature_freshness_age_ratio": _clamp01(
                _to_float(freshness.get("age_seconds"), 0.0) / max(_to_float(freshness.get("max_age_seconds"), 20.0), 1e-6)
            ),
            "master_latency_slo_ok": 1.0 if bool(latency.get("ok", True)) else 0.0,
            "master_latency_ratio": _clamp01(
                _to_float(latency.get("elapsed_ms"), 0.0) / max(_to_float(latency.get("timeout_ms"), 900.0), 1e-6)
            ),
            "risk_pause_active": 1.0 if any(bool(cb.get(k, False)) for k in ("kill_switch_active", "vol_shock_pause_active", "liquidity_pause_active")) else 0.0,
            "exec_slippage_bps": _to_float(exec_sim.get("slippage_bps"), 0.0),
            "exec_latency_ms": _to_float(exec_sim.get("latency_ms"), 0.0),
            "exec_impact_bps": _to_float(exec_sim.get("impact_bps"), 0.0),
        }
        for key in BEHAVIOR_LANE_FEATURE_NAMES:
            out[sid][key] = _clamp01(_to_float(lane_features.get(key), 0.0))
        for key in BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES:
            raw = _to_float(capital_flow.get(key), 0.0)
            if key == "capital_flow_signed_scaled":
                out[sid][key] = _clamp(raw, -1.0, 1.0)
            else:
                out[sid][key] = _clamp01(raw)
        for key in BEHAVIOR_FLOW_AWARENESS_FEATURE_NAMES:
            raw = _to_float(flow_awareness.get(key), 0.0)
            if key == "flow_direction_signed":
                out[sid][key] = _clamp(raw, -1.0, 1.0)
            else:
                out[sid][key] = _clamp01(raw)
        for key in BEHAVIOR_LEAD_LAG_FEATURE_NAMES:
            raw = _to_float(lead_lag.get(key), 0.0)
            if key == "lead_lag_signal_signed":
                out[sid][key] = _clamp(raw, -1.0, 1.0)
            else:
                out[sid][key] = _clamp01(raw)
        for key in BEHAVIOR_ALLOCATION_FEATURE_NAMES:
            fallback = _to_float(cb.get(key), 0.0) if key == "regime_dislocation_norm" else 0.0
            raw = _to_float(allocation_conf.get(key), fallback)
            if key == "allocation_confidence_scale":
                out[sid][key] = _clamp01(raw / 1.25)
            else:
                out[sid][key] = _clamp01(raw)
    return out


def _load_exec_history(rows: Iterable[Dict[str, Any]], since_utc: datetime) -> Dict[str, List[Tuple[float, float, float, float]]]:
    by_symbol: Dict[str, List[Tuple[float, float, float, float]]] = defaultdict(list)
    for row in rows:
        if str(row.get("layer") or "") != "grand_master":
            continue
        ts = _parse_ts(row.get("timestamp_utc"))
        if ts is None or ts < since_utc:
            continue
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        by_symbol[symbol].append(
            (
                ts.timestamp(),
                _to_float(row.get("slippage_bps"), 0.0),
                _to_float(row.get("latency_ms"), 0.0),
                _to_float(row.get("impact_bps"), 0.0),
            )
        )

    for sym in by_symbol:
        by_symbol[sym].sort(key=lambda x: x[0])
    return by_symbol


def _paper_trade_slippage_bps(row: Dict[str, Any]) -> float:
    action = _normalize_action(str(row.get("action") or "HOLD"))
    fill = _to_float(row.get("fill_price"), 0.0)
    ref = _to_float(row.get("reference_price"), _to_float(row.get("intended_price"), 0.0))
    if fill <= 0.0 or ref <= 0.0:
        return 0.0
    if action == "BUY":
        return ((fill - ref) / ref) * 10000.0
    if action == "SELL":
        return ((ref - fill) / ref) * 10000.0
    return ((fill - ref) / ref) * 10000.0


def _paper_trade_return_proxy_bps(row: Dict[str, Any]) -> float:
    action = _normalize_action(str(row.get("action") or "HOLD"))
    fill = _to_float(row.get("fill_price"), 0.0)
    mark = _to_float(row.get("mark_price"), _to_float(row.get("reference_price"), 0.0))
    ref = _to_float(row.get("reference_price"), fill)
    if fill <= 0.0 or mark <= 0.0 or ref <= 0.0:
        return 0.0
    if action == "BUY":
        return ((mark - fill) / ref) * 10000.0
    if action == "SELL":
        return ((fill - mark) / ref) * 10000.0
    return 0.0


def _load_paper_trade_context(
    rows: Iterable[Dict[str, Any]],
    since_utc: datetime,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[Tuple[float, float, float]]]]:
    by_snapshot: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "slippage_sum": 0.0,
            "return_proxy_sum": 0.0,
            "entry_cost_sum": 0.0,
        }
    )
    by_symbol: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)

    for row in rows:
        ts = _parse_ts(row.get("timestamp_utc"))
        if ts is None or ts < since_utc:
            continue
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        snapshot_id = str(metadata.get("snapshot_id") or row.get("snapshot_id") or "").strip()
        slippage_bps = _paper_trade_slippage_bps(row)
        return_proxy_bps = _paper_trade_return_proxy_bps(row)
        modeled_cost_bps = max(
            abs(slippage_bps),
            max(_to_float(row.get("expected_slippage_bps"), 0.0), 0.0)
            + max(_to_float(row.get("expected_impact_bps"), 0.0), 0.0)
            + max(_to_float(row.get("expected_spread_jump_penalty_bps"), 0.0), 0.0),
        )

        if snapshot_id:
            agg = by_snapshot[snapshot_id]
            agg["count"] += 1.0
            agg["slippage_sum"] += slippage_bps
            agg["return_proxy_sum"] += return_proxy_bps
            agg["entry_cost_sum"] += modeled_cost_bps

        by_symbol[symbol].append((ts.timestamp(), slippage_bps, return_proxy_bps))

    snapshot_summary: Dict[str, Dict[str, float]] = {}
    for snapshot_id, agg in by_snapshot.items():
        count = max(_to_float(agg.get("count"), 0.0), 0.0)
        denom = max(count, 1.0)
        snapshot_summary[snapshot_id] = {
            "count": count,
            "mean_slippage_bps": _to_float(agg.get("slippage_sum"), 0.0) / denom,
            "mean_return_proxy_bps": _to_float(agg.get("return_proxy_sum"), 0.0) / denom,
            "mean_entry_cost_bps": _to_float(agg.get("entry_cost_sum"), 0.0) / denom,
        }

    for sym in by_symbol:
        by_symbol[sym].sort(key=lambda x: x[0])

    return snapshot_summary, by_symbol


def _find_last_exec_metrics(history: List[Tuple[float, float, float, float]], ts_epoch: float) -> Tuple[float, float, float]:
    lo = 0
    hi = len(history)
    while lo < hi:
        mid = (lo + hi) // 2
        if history[mid][0] < ts_epoch:
            lo = mid + 1
        else:
            hi = mid
    idx = lo - 1
    if idx < 0:
        return 0.0, 0.0, 0.0

    # Small rolling average over prior observations; available at decision time.
    start = max(0, idx - 24)
    window = history[start : idx + 1]
    n = max(len(window), 1)
    slip = sum(r[1] for r in window) / n
    lat = sum(r[2] for r in window) / n
    impact = sum(r[3] for r in window) / n
    return slip, lat, impact


def _find_last_paper_metrics(history: List[Tuple[float, float, float]], ts_epoch: float) -> Tuple[float, float, float]:
    lo = 0
    hi = len(history)
    while lo < hi:
        mid = (lo + hi) // 2
        if history[mid][0] < ts_epoch:
            lo = mid + 1
        else:
            hi = mid
    idx = lo - 1
    if idx < 0:
        return 0.0, 0.0, 0.0

    start = max(0, idx - 24)
    window = history[start : idx + 1]
    n = max(len(window), 1)
    trade_count_norm = _clamp01(len(window) / 8.0)
    slip = sum(r[1] for r in window) / n
    ret = sum(r[2] for r in window) / n
    return trade_count_norm, slip, ret


def _normalize_action(action: str) -> str:
    a = (action or "").upper().strip()
    if a in {"BUY", "SELL", "HOLD"}:
        return a
    return "HOLD"


def _direction_for_action(action: str) -> float:
    if action == "BUY":
        return 1.0
    if action == "SELL":
        return -1.0
    return 0.0


def _label_from_forward(
    *,
    action: str,
    forward_return: float,
    positive_thr: float,
    negative_thr: float,
    hold_pos_thr: float,
    hold_neg_thr: float,
) -> Tuple[str, float]:
    if action in {"BUY", "SELL"}:
        edge = _direction_for_action(action) * forward_return
        if edge >= positive_thr:
            conf = _clamp01(edge / max(positive_thr, 1e-6))
            return "positive", conf
        if edge <= -negative_thr:
            conf = _clamp01(abs(edge) / max(negative_thr, 1e-6))
            return "negative", conf
        conf = _clamp01(abs(edge) / max(max(positive_thr, negative_thr), 1e-6))
        return "neutral", conf

    abs_ret = abs(forward_return)
    if abs_ret <= hold_pos_thr:
        conf = _clamp01((hold_pos_thr - abs_ret) / max(hold_pos_thr, 1e-6))
        return "positive", conf
    if abs_ret >= hold_neg_thr:
        conf = _clamp01((abs_ret - hold_neg_thr) / max(hold_neg_thr, 1e-6))
        return "negative", conf
    conf = _clamp01(abs_ret / max(hold_neg_thr, 1e-6))
    return "neutral", conf


def _excursion_bucket(value: float, *, adverse: bool = False) -> str:
    bps = float(value) * 10000.0
    magnitude = abs(min(bps, 0.0)) if adverse else max(bps, 0.0)
    prefix = "adverse" if adverse else "favorable"
    if magnitude < 3.0:
        return f"{prefix}_flat"
    if magnitude < 10.0:
        return f"{prefix}_small"
    if magnitude < 25.0:
        return f"{prefix}_medium"
    return f"{prefix}_large"


def _path_dependent_labels(
    *,
    action: str,
    base_price: float,
    future_prices: List[float],
    post_cost_forward_return: float,
    hold_opportunity_threshold: float,
) -> Dict[str, Any]:
    prices = [float(price) for price in future_prices if float(price) > 0.0]
    if base_price <= 0.0 or not prices:
        return {
            "path_label_ready": False,
            "path_observation_count": 0,
            "no_trade_counterfactual_outcome": "path_unavailable",
        }
    raw_returns = [(price - base_price) / base_price for price in prices]
    direction = _direction_for_action(action)
    directional_returns = [direction * value for value in raw_returns] if direction else raw_returns
    mfe = max(directional_returns)
    mae = min(directional_returns)
    path_range = max(raw_returns) - min(raw_returns)
    final_raw = raw_returns[-1]
    if action in {"BUY", "SELL"}:
        no_trade_outcome = (
            "trade_outperformed_cash"
            if post_cost_forward_return > 0.0
            else "cash_outperformed_trade"
            if post_cost_forward_return < 0.0
            else "trade_matched_cash"
        )
        if post_cost_forward_return <= 0.0:
            exit_timing = "loss_or_cost_drag"
        else:
            capture = post_cost_forward_return / max(mfe, 1e-12)
            exit_timing = (
                "near_path_best"
                if capture >= 0.80
                else "reasonable_capture"
                if capture >= 0.40
                else "premature_or_faded"
            )
    else:
        best_counter_side = max(max(raw_returns), -min(raw_returns), 0.0)
        no_trade_outcome = (
            "hold_missed_large_move"
            if best_counter_side >= max(hold_opportunity_threshold, 0.0)
            else "hold_was_efficient"
        )
        exit_timing = "not_applicable_hold"

    directional_efficiency = abs(final_raw) / max(path_range, 1e-12)
    if path_range >= 0.005:
        path_regime = "shock_path"
    elif directional_efficiency >= 0.65:
        path_regime = "persistent_path"
    elif max(raw_returns) > 0.0 and min(raw_returns) < 0.0:
        path_regime = "two_sided_chop_path"
    else:
        path_regime = "low_motion_path"
    return {
        "path_label_ready": True,
        "path_observation_count": len(prices),
        "maximum_favorable_excursion": round(mfe, 8),
        "maximum_adverse_excursion": round(mae, 8),
        "mae_bucket": _excursion_bucket(mae, adverse=True),
        "mfe_bucket": _excursion_bucket(mfe),
        "exit_timing_bucket": exit_timing,
        "post_entry_regime_bucket": path_regime,
        "no_trade_counterfactual_return": 0.0,
        "trade_vs_no_trade_excess_return": round(post_cost_forward_return, 8),
        "no_trade_counterfactual_outcome": no_trade_outcome,
        "path_range_return": round(path_range, 8),
    }


def _decision_feature_vector(
    *,
    row: Dict[str, Any],
    gov: Dict[str, float],
    lag_exec: Tuple[float, float, float],
    paper_snapshot: Dict[str, float],
    lag_paper: Tuple[float, float, float],
    snapshot_context: Dict[str, float],
    external_context: Dict[str, float],
    external_meta: Dict[str, Any],
    event_windows: List[Tuple[int, int]],
) -> Tuple[List[float], str, float]:
    features = row.get("features") or {}
    ts = row["ts_utc"]

    symbol = row["symbol"]
    action = row["action"]
    role_idx = row["role_idx"]
    dividend_drip_symbol_features = external_meta.get("dividend_drip_symbol_features") if isinstance(external_meta.get("dividend_drip_symbol_features"), dict) else {}
    symbol_dividend_drip = dividend_drip_symbol_features.get(symbol) if isinstance(dividend_drip_symbol_features.get(symbol), dict) else {}
    central_bank_meta = external_meta.get("central_bank_cross_source") if isinstance(external_meta.get("central_bank_cross_source"), dict) else {}
    central_bank_symbol_features = central_bank_meta.get("symbol_features") if isinstance(central_bank_meta.get("symbol_features"), dict) else {}
    symbol_central_bank = central_bank_symbol_features.get(symbol) if isinstance(central_bank_symbol_features.get(symbol), dict) else {}
    context_mesh_meta = external_meta.get("decision_context_mesh") if isinstance(external_meta.get("decision_context_mesh"), dict) else {}
    context_mesh_symbol_features = context_mesh_meta.get("symbol_features") if isinstance(context_mesh_meta.get("symbol_features"), dict) else {}
    symbol_context_mesh = context_mesh_symbol_features.get(symbol) if isinstance(context_mesh_symbol_features.get(symbol), dict) else {}

    def _drip_value(name: str) -> float:
        return _clamp01(_to_float(features.get(name), _to_float(symbol_dividend_drip.get(name), _to_float(external_context.get(name), 0.0))))

    def _central_bank_value(name: str) -> float:
        return _clamp01(
            _to_float(
                features.get(name),
                _to_float(symbol_central_bank.get(name), _to_float(external_context.get(name), 0.0)),
            )
        )

    def _context_mesh_value(name: str) -> float:
        return _clamp01(
            _to_float(
                features.get(name),
                _to_float(symbol_context_mesh.get(name), _to_float(external_context.get(name), 0.0)),
            )
        )

    pct = _to_float(features.get("pct_from_close"), 0.0)
    mom = _to_float(features.get("mom_5m"), 0.0)
    vol = _to_float(features.get("vol_30m"), 0.0)
    range_pos = _clamp01(_to_float(features.get("range_pos"), 0.5))
    spread_bps = abs(_to_float(features.get("spread_bps"), 0.0))

    ctx_vix_pct = _to_float(features.get("ctx_VIX_X_pct_from_close"), _to_float(features.get("ctx_VIX_pct_from_close"), 0.0))
    ctx_uup_pct = _to_float(features.get("ctx_UUP_pct_from_close"), 0.0)

    pnl_proxy = _signed_scale((pct + (0.5 * mom) - (0.25 * vol)) * 100.0, 1.0)
    qty_log = math.log1p(abs(_to_float(row.get("quantity"), 0.0)))

    regime_idx, regime = _regime_index(symbol, features)
    label_confidence_proxy = _clamp01((abs(pct) + (0.5 * abs(mom)) + (0.25 * abs(vol))) * 25.0)

    slip, lat_ms, impact = lag_exec
    paper_recent_trade_count_norm, paper_recent_slip_bps, paper_recent_return_proxy_bps = lag_paper
    paper_snapshot_trade_count_norm = _clamp01(_to_float(paper_snapshot.get("count"), 0.0) / 8.0)
    paper_snapshot_slippage_bps_norm = _clamp01(abs(_to_float(paper_snapshot.get("mean_slippage_bps"), 0.0)) / 25.0)
    paper_snapshot_return_proxy_signed_scaled = _signed_scale(_to_float(paper_snapshot.get("mean_return_proxy_bps"), 0.0) / 10000.0, 80.0)
    paper_recent_slippage_bps_norm = _clamp01(abs(paper_recent_slip_bps) / 25.0)
    paper_recent_return_proxy_signed_scaled = _signed_scale(paper_recent_return_proxy_bps / 10000.0, 80.0)

    session = _session_event_context(ts, event_windows)

    vec = [
        pnl_proxy,
        qty_log,
        role_idx,
        _hash01(symbol),
        _hash01(action),
        ts.weekday() / 6.0,
        ts.hour / 23.0,
        regime_idx,
        label_confidence_proxy,
        _signed_scale(pct, 40.0),
        _signed_scale(mom, 120.0),
        _signed_scale(vol, 60.0),
        range_pos,
        _clamp01(spread_bps / 25.0),
        _signed_scale(ctx_vix_pct, 60.0),
        _signed_scale(ctx_uup_pct, 60.0),
        _clamp01(abs(slip) / 10.0),
        _clamp01(lat_ms / 350.0),
        _clamp01(abs(impact) / 10.0),
        _clamp01(_to_float(gov.get("active_sub_bots"), 0.0) / 60.0),
        _clamp01(_to_float(gov.get("queue_depth"), 0.0) / 1000.0),
        _clamp01(abs(_to_float(gov.get("dispatch_qty"), 0.0)) / 20.0),
        session["session_bucket_norm"],
        session["mins_from_open_norm"],
        session["mins_to_close_norm"],
        session["event_window_proximity"],
        _clamp01(_to_float(gov.get("feature_freshness_ok"), 1.0)),
        _clamp01(_to_float(gov.get("feature_freshness_age_ratio"), 0.0)),
        _clamp01(_to_float(gov.get("master_latency_slo_ok"), 1.0)),
        _clamp01(_to_float(gov.get("master_latency_ratio"), 0.0)),
        _clamp01(_to_float(gov.get("risk_pause_active"), 0.0)),
        _clamp01(_to_float(features.get("options_chain_available"), 0.0)),
        _clamp01(_to_float(features.get("options_iv_atm_norm"), 0.0)),
        _clamp01(_to_float(features.get("options_iv_skew_norm"), 0.0)),
        _clamp01(_to_float(features.get("options_iv_term_structure_norm"), 0.0)),
        _clamp01(_to_float(features.get("options_put_call_oi_ratio_norm"), 0.0)),
        _clamp01(_to_float(features.get("options_negative_bias_norm"), 0.0)),
        _clamp01(_to_float(features.get("options_roll_yield_norm"), 0.0)),
        _clamp01(_to_float(features.get("options_vwap_bias_norm"), 0.0)),
        _clamp01(_to_float(features.get("options_vol_expectation_norm"), 0.0)),
        _clamp01(_to_float(features.get("calendar_event_proximity_norm"), 0.0)),
        _clamp01(_to_float(features.get("calendar_high_impact_24h_norm"), 0.0)),
        _clamp01(_to_float(features.get("calendar_options_expiry_week_norm"), 0.0)),
        _clamp01(_to_float(features.get("calendar_dividend_events_30d_norm"), 0.0)),
        _clamp01(_to_float(features.get("calendar_dividend_exdate_proximity_norm"), 0.0)),
        _clamp01(_to_float(features.get("calendar_dividend_payout_proximity_norm"), 0.0)),
        _clamp01(_to_float(features.get("calendar_dividend_recent_exdate_norm"), 0.0)),
        _clamp01(_to_float(features.get("calendar_dividend_quality_signal_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_yield_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_payout_ratio_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_ex_date_proximity_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_pay_date_proximity_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_quality_score_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_capture_entry_signal_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_capture_exit_signal_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_compound_bias_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_compound_growth_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_compound_drawdown_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_compound_steps_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_compounding_quality_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_capture_timing_quality_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_payout_stress_gate_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_growth_persistence_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_capture_ex_date_hazard_norm"), 0.0)),
        _clamp01(_to_float(features.get("dividend_strategy_mode_capture"), 0.0)),
        _clamp01(_to_float(features.get("dividend_strategy_mode_compound"), 0.0)),
        _clamp01(_to_float(features.get("dividend_strategy_mode_hybrid"), 0.0)),
        _drip_value("dividend_drip_active_norm"),
        _drip_value("dividend_drip_recent_reinvest_norm"),
        _drip_value("dividend_drip_cash_only_norm"),
        _drip_value("dividend_drip_share_credit_norm"),
        _drip_value("dividend_drip_event_recency_norm"),
        _drip_value("dividend_drip_confidence_norm"),
        _clamp01(_to_float(features.get("futures_order_book_imbalance_norm"), 0.0)),
        _clamp01(_to_float(features.get("futures_funding_rate_norm"), 0.0)),
        _clamp01(_to_float(features.get("futures_basis_bps_norm"), 0.0)),
        _clamp01(_to_float(features.get("futures_term_structure_norm"), 0.0)),
        _clamp01(_to_float(features.get("futures_negative_bias_norm"), 0.0)),
        _clamp01(_to_float(features.get("futures_roll_yield_norm"), 0.0)),
        _clamp01(_to_float(features.get("futures_vwap_bias_norm"), 0.0)),
        _clamp01(_to_float(features.get("options_specialist_active"), _to_float(gov.get("active_options_sub_bots"), 0.0))),
        _clamp01(_to_float(features.get("futures_specialist_active"), _to_float(gov.get("active_futures_sub_bots"), 0.0))),
        _signed_scale(_to_float(features.get("options_specialist_vote"), _to_float(gov.get("options_specialist_vote"), 0.0)), 1.0),
        _signed_scale(_to_float(features.get("futures_specialist_vote"), _to_float(gov.get("futures_specialist_vote"), 0.0)), 1.0),
        _clamp01(_to_float(features.get("active_options_sub_bots"), _to_float(gov.get("active_options_sub_bots"), 0.0)) / 20.0),
        _clamp01(_to_float(features.get("active_futures_sub_bots"), _to_float(gov.get("active_futures_sub_bots"), 0.0)) / 20.0),
        _clamp01(_to_float(snapshot_context.get("snapshot_cov_ok"), 1.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_cov_log_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_replay_stale_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_replay_drift_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_divergence_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_triprate_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_queue_pressure_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_drill_ok"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_drill_restore_fail_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_drill_missing_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_drill_recency_norm"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("canary_weight_cap_norm"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_raw_sql_ingest_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_raw_count_norm"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_raw_file_count_norm"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_raw_bytes_norm"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_raw_json_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_raw_event_file_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_raw_lock_file_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_raw_recency_norm"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_cov_fill_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_replay_ok"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_e2e_replay_ok"), 1.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_e2e_hash_match"), 1.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_paper_replay_ok"), 1.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_paper_replay_hash_match"), 1.0)),
        _clamp01(_to_float(external_context.get("external_feeds_ok"), 0.0)),
        _clamp01(_to_float(external_context.get("external_feeds_recency_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("external_fred_unrate_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("external_fred_cpi_mom_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("external_fred_gdp_qoq_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("external_bls_unrate_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("external_bls_cpi_mom_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("external_census_population_log_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("external_bea_dataset_count_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("external_micro_auction_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("external_micro_relative_volume_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("external_micro_options_flow_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("external_micro_short_pressure_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("external_micro_credit_flow_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("external_micro_block_trade_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("tasty_iv_rank_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("tasty_implied_volatility_index_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("tasty_liquidity_rating_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("tasty_expected_move_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("tasty_beta_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("tasty_watchlist_presence_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("options_iv_crush_risk_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("options_assignment_risk_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("options_zero_dte_regime_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("options_vol_of_vol_change_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("options_spread_execution_risk_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("options_higher_order_greek_pressure_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("options_barrier_touch_risk_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("options_lookback_path_dependency_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("options_variance_swap_proxy_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("options_volatility_swap_proxy_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("options_gamma_scalping_pressure_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("options_vanna_volga_hedge_pressure_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("options_dispersion_trade_proxy_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("options_volatility_arbitrage_proxy_norm"), 0.0)),
        _clamp01(_to_float(features.get("crypto_deribit_futures_oi_norm"), _to_float(external_context.get("crypto_deribit_futures_oi_norm"), 0.0))),
        _clamp01(_to_float(features.get("crypto_deribit_options_oi_norm"), _to_float(external_context.get("crypto_deribit_options_oi_norm"), 0.0))),
        _clamp01(_to_float(features.get("crypto_deribit_mark_iv_norm"), _to_float(external_context.get("crypto_deribit_mark_iv_norm"), 0.0))),
        _clamp01(_to_float(features.get("crypto_deribit_basis_norm"), _to_float(external_context.get("crypto_deribit_basis_norm"), 0.0))),
        _clamp01(_to_float(features.get("crypto_kraken_volume_norm"), _to_float(external_context.get("crypto_kraken_volume_norm"), 0.0))),
        _clamp01(_to_float(features.get("crypto_kraken_range_norm"), _to_float(external_context.get("crypto_kraken_range_norm"), 0.0))),
        _clamp01(_to_float(features.get("crypto_hyperliquid_funding_norm"), _to_float(external_context.get("crypto_hyperliquid_funding_norm"), 0.0))),
        _clamp01(_to_float(features.get("crypto_hyperliquid_open_interest_norm"), _to_float(external_context.get("crypto_hyperliquid_open_interest_norm"), 0.0))),
        _clamp01(_to_float(features.get("crypto_hyperliquid_basis_norm"), _to_float(external_context.get("crypto_hyperliquid_basis_norm"), 0.0))),
        _clamp01(_to_float(features.get("crypto_coinmetrics_tx_count_norm"), _to_float(external_context.get("crypto_coinmetrics_tx_count_norm"), 0.0))),
        _clamp01(_to_float(features.get("crypto_coinmetrics_active_addr_norm"), _to_float(external_context.get("crypto_coinmetrics_active_addr_norm"), 0.0))),
        _clamp01(_to_float(features.get("crypto_coingecko_volume_norm"), _to_float(external_context.get("crypto_coingecko_volume_norm"), 0.0))),
        _clamp01(_to_float(features.get("crypto_coingecko_momentum_norm"), _to_float(external_context.get("crypto_coingecko_momentum_norm"), 0.0))),
        _clamp01(_to_float(features.get("crypto_cross_provider_price_agreement_norm"), _to_float(external_context.get("crypto_cross_provider_price_agreement_norm"), 0.0))),
        _clamp01(_to_float(features.get("crypto_defillama_stablecoin_growth_norm"), _to_float(external_context.get("crypto_defillama_stablecoin_growth_norm"), 0.0))),
        _clamp01(_to_float(features.get("crypto_defillama_dex_volume_growth_norm"), _to_float(external_context.get("crypto_defillama_dex_volume_growth_norm"), 0.0))),
        _clamp01(_to_float(features.get("crypto_etherscan_gas_norm"), _to_float(external_context.get("crypto_etherscan_gas_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_risk_corr_norm"), _to_float(external_context.get("market_crypto_risk_corr_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_spy_corr_norm"), _to_float(external_context.get("market_crypto_spy_corr_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_qqq_corr_norm"), _to_float(external_context.get("market_crypto_qqq_corr_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_tlt_corr_norm"), _to_float(external_context.get("market_crypto_tlt_corr_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_uup_inverse_corr_norm"), _to_float(external_context.get("market_crypto_uup_inverse_corr_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_gold_corr_norm"), _to_float(external_context.get("market_crypto_gold_corr_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_current_alignment_norm"), _to_float(external_context.get("market_crypto_current_alignment_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_divergence_norm"), _to_float(external_context.get("market_crypto_divergence_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_corr_confidence_norm"), _to_float(external_context.get("market_crypto_corr_confidence_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_sleeve_coverage_norm"), _to_float(external_context.get("market_crypto_sleeve_coverage_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_sleeve_avg_abs_corr_norm"), _to_float(external_context.get("market_crypto_sleeve_avg_abs_corr_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_sleeve_dispersion_norm"), _to_float(external_context.get("market_crypto_sleeve_dispersion_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_sleeve_confidence_norm"), _to_float(external_context.get("market_crypto_sleeve_confidence_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_risk_on_crypto_alignment_norm"), _to_float(external_context.get("market_crypto_risk_on_crypto_alignment_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_fx_crypto_inverse_corr_norm"), _to_float(external_context.get("market_crypto_fx_crypto_inverse_corr_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_rates_crypto_corr_norm"), _to_float(external_context.get("market_crypto_rates_crypto_corr_norm"), 0.0))),
        _clamp01(_to_float(features.get("market_crypto_energy_crypto_corr_norm"), _to_float(external_context.get("market_crypto_energy_crypto_corr_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_official_data_available"), _to_float(external_context.get("fx_official_data_available"), 0.0))),
        _clamp01(_to_float(features.get("fx_eurusd_level_norm"), _to_float(external_context.get("fx_eurusd_level_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_eurusd_momentum_norm"), _to_float(external_context.get("fx_eurusd_momentum_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_usdjpy_level_norm"), _to_float(external_context.get("fx_usdjpy_level_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_usdjpy_momentum_norm"), _to_float(external_context.get("fx_usdjpy_momentum_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_gbpusd_level_norm"), _to_float(external_context.get("fx_gbpusd_level_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_gbpusd_momentum_norm"), _to_float(external_context.get("fx_gbpusd_momentum_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_usd_strength_norm"), _to_float(external_context.get("fx_usd_strength_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_usd_broad_index_norm"), _to_float(external_context.get("fx_usd_broad_index_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_proxy_agreement_norm"), _to_float(external_context.get("fx_proxy_agreement_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_risk_on_alignment_norm"), _to_float(external_context.get("fx_risk_on_alignment_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_crypto_alignment_norm"), _to_float(external_context.get("fx_crypto_alignment_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_macro_dispersion_norm"), _to_float(external_context.get("fx_macro_dispersion_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_corr_confidence_norm"), _to_float(external_context.get("fx_corr_confidence_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_session_asia_norm"), _to_float(external_context.get("fx_session_asia_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_session_london_norm"), _to_float(external_context.get("fx_session_london_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_session_ny_norm"), _to_float(external_context.get("fx_session_ny_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_rollover_risk_norm"), _to_float(external_context.get("fx_rollover_risk_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_dxy_yield_confirmation_norm"), _to_float(external_context.get("fx_dxy_yield_confirmation_norm"), 0.0))),
        _clamp01(_to_float(features.get("fx_carry_proxy_norm"), _to_float(external_context.get("fx_carry_proxy_norm"), 0.0))),
        _clamp01(_to_float(external_context.get("live_macro_gate_active_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("live_macro_gate_confidence_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("sec_context_signal_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("extended_quant_signal_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("official_macro_signal_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("schwab_education_signal_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("market_breadth_signal_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("bond_reference_signal_norm"), 0.0)),
        *[
            _clamp01(_to_float(external_context.get(key), 0.0))
            for key in CENTRAL_BANK_LIQUIDITY_FEATURE_KEYS
        ],
        *[_central_bank_value(key) for key in GLOBAL_CENTRAL_BANK_FEATURE_KEYS],
        *[_central_bank_value(key) for key in CENTRAL_BANK_CROSS_SOURCE_FEATURE_KEYS],
        *[_context_mesh_value(key) for key in DECISION_CONTEXT_MESH_FEATURE_KEYS],
        _clamp01(_to_float(external_context.get("source_quality_average_score_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("source_quality_required_failure_ratio_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("source_quality_soft_failure_ratio_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("source_quality_unverified_ratio_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("source_quality_cross_verified_ratio_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("source_quality_market_micro_score_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("source_quality_official_macro_score_norm"), 0.0)),
        _clamp01(_to_float(external_context.get("source_quality_crypto_context_score_norm"), 0.0)),
    ]

    for key in BEHAVIOR_LANE_FEATURE_NAMES:
        vec.append(_clamp01(_to_float(features.get(key), _to_float(gov.get(key), 0.0))))
    vec.extend(
        [
            paper_snapshot_trade_count_norm,
            paper_snapshot_slippage_bps_norm,
            paper_snapshot_return_proxy_signed_scaled,
            paper_recent_trade_count_norm,
            paper_recent_slippage_bps_norm,
            paper_recent_return_proxy_signed_scaled,
        ]
    )
    for key in PAPER_PROFITABILITY_FEATURE_NAMES:
        vec.append(_clamp01(_to_float(features.get(key), _to_float(gov.get(key), 0.0))))
    for key in BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES:
        raw = _to_float(features.get(key), _to_float(gov.get(key), 0.0))
        if key == "capital_flow_signed_scaled":
            vec.append(_clamp(raw, -1.0, 1.0))
        else:
            vec.append(_clamp01(raw))
    for key in BEHAVIOR_FLOW_AWARENESS_FEATURE_NAMES:
        raw = _to_float(features.get(key), _to_float(gov.get(key), 0.0))
        if key == "flow_direction_signed":
            vec.append(_clamp(raw, -1.0, 1.0))
        else:
            vec.append(_clamp01(raw))
    for key in BEHAVIOR_LEAD_LAG_FEATURE_NAMES:
        raw = _to_float(features.get(key), _to_float(gov.get(key), 0.0))
        if key == "lead_lag_signal_signed":
            vec.append(_clamp(raw, -1.0, 1.0))
        else:
            vec.append(_clamp01(raw))
    for key in BEHAVIOR_ALLOCATION_FEATURE_NAMES:
        raw = _to_float(features.get(key), _to_float(gov.get(key), 0.0))
        if key == "allocation_confidence_scale":
            vec.append(_clamp01(raw / 1.25))
        else:
            vec.append(_clamp01(raw))

    return vec, regime, label_confidence_proxy


def main() -> int:
    parser = argparse.ArgumentParser(description="Build leak-free behavior dataset from shadow decisions (forward-return labels + rich context).")
    parser.add_argument("--decision-glob", default=str(PROJECT_ROOT / "decision_explanations" / "shadow*" / "decision_explanations_*.jsonl"))
    parser.add_argument("--channel-decision-glob", default=os.getenv("BEHAVIOR_DATASET_CHANNEL_DECISION_GLOB", str(PROJECT_ROOT / "governance" / "channels" / "decision" / "*" / "decision_*.jsonl")))
    parser.add_argument("--governance-glob", default=str(PROJECT_ROOT / "governance" / "shadow*" / "master_control_*.jsonl"))
    parser.add_argument("--pnl-attribution-glob", default=str(PROJECT_ROOT / "governance" / "shadow*" / "shadow_pnl_attribution_*.jsonl"))
    parser.add_argument("--paper-trades-glob", default=str(PROJECT_ROOT / "exports" / "trade_logs" / "**" / "paper_trades_*.jsonl"))
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "data" / "trade_history" / "trade_learning_dataset.json"))
    parser.add_argument("--policy", default=str(PROJECT_ROOT / "config" / "trade_learning_policy.json"))
    parser.add_argument("--lookback-hours", type=int, default=int(os.getenv("BEHAVIOR_DATASET_LOOKBACK_HOURS", "96")))
    parser.add_argument("--horizon-seconds", type=int, default=int(os.getenv("BEHAVIOR_DATASET_FORWARD_HORIZON_SECONDS", "300")))
    parser.add_argument("--aux-horizon-seconds", type=int, default=int(os.getenv("BEHAVIOR_DATASET_FORWARD_AUX_HORIZON_SECONDS", "900")))
    parser.add_argument("--horizon-blend-alpha", type=float, default=float(os.getenv("BEHAVIOR_DATASET_HORIZON_BLEND_ALPHA", "0.65")))
    parser.add_argument("--max-examples", type=int, default=int(os.getenv("BEHAVIOR_DATASET_MAX_EXAMPLES", "120000")))
    parser.add_argument("--min-per-symbol", type=int, default=int(os.getenv("BEHAVIOR_DATASET_MIN_PER_SYMBOL", "8")))
    parser.add_argument("--max-per-symbol", type=int, default=int(os.getenv("BEHAVIOR_DATASET_MAX_PER_SYMBOL", "3000")))
    parser.add_argument("--max-per-symbol-regime", type=int, default=int(os.getenv("BEHAVIOR_DATASET_MAX_PER_SYMBOL_REGIME", "1200")))
    parser.add_argument("--decision-tail-bytes", type=int, default=int(os.getenv("BEHAVIOR_DATASET_DECISION_TAIL_BYTES", str(16 * 1024 * 1024))))
    parser.add_argument("--governance-tail-bytes", type=int, default=int(os.getenv("BEHAVIOR_DATASET_GOVERNANCE_TAIL_BYTES", str(8 * 1024 * 1024))))
    parser.add_argument("--pnl-tail-bytes", type=int, default=int(os.getenv("BEHAVIOR_DATASET_PNL_TAIL_BYTES", str(64 * 1024 * 1024))))
    parser.add_argument("--paper-trades-tail-bytes", type=int, default=int(os.getenv("BEHAVIOR_DATASET_PAPER_TRADES_TAIL_BYTES", str(64 * 1024 * 1024))))
    parser.add_argument("--channel-decision-max-files", type=int, default=int(os.getenv("BEHAVIOR_DATASET_CHANNEL_DECISION_MAX_FILES", "96")))
    parser.add_argument("--sqlite-path", default=os.getenv("BEHAVIOR_DATASET_SQLITE_PATH", ""))
    parser.add_argument("--min-output-rows", type=int, default=int(os.getenv("BEHAVIOR_DATASET_MIN_OUTPUT_ROWS", "50")))
    parser.add_argument("--failure-file", default=os.getenv("BEHAVIOR_DATASET_FAILURE_FILE", str(DEFAULT_BUILD_FAILURE_PATH)))
    parser.add_argument(
        "--prefer-sql",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("BEHAVIOR_DATASET_PREFER_SQL", "1").strip() == "1",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    since_utc = now_utc - timedelta(hours=max(args.lookback_hours, 1))

    policy = _safe_load_json(Path(args.policy), default={})
    fw_cfg = policy.get("behavior_forward_labels", {})
    outcome_cfg = policy.get("outcome_learning", {})

    positive_bps = float(fw_cfg.get("positive_bps", 6.0))
    negative_bps = float(fw_cfg.get("negative_bps", 6.0))
    hold_positive_bps = float(fw_cfg.get("hold_positive_max_bps", 4.0))
    hold_negative_bps = float(fw_cfg.get("hold_negative_min_bps", 14.0))

    positive_thr = max(positive_bps, 0.1) / 10000.0
    negative_thr = max(negative_bps, 0.1) / 10000.0
    hold_pos_thr = max(hold_positive_bps, 0.1) / 10000.0
    hold_neg_thr = max(hold_negative_bps, 0.1) / 10000.0

    post_cost_cfg = fw_cfg.get("post_cost_labels") if isinstance(fw_cfg.get("post_cost_labels"), dict) else {}
    post_cost_labels_enabled = bool(post_cost_cfg.get("enabled", True))
    minimum_entry_cost_bps = max(
        float(post_cost_cfg.get("minimum_entry_cost_bps", os.getenv("BEHAVIOR_LABEL_MIN_ENTRY_COST_BPS", "2.0"))),
        0.0,
    )
    default_exit_cost_bps = max(
        float(post_cost_cfg.get("default_exit_cost_bps", os.getenv("BEHAVIOR_LABEL_DEFAULT_EXIT_COST_BPS", "2.0"))),
        0.0,
    )
    fee_bps = max(
        float(post_cost_cfg.get("fee_bps", os.getenv("BEHAVIOR_LABEL_FEE_BPS", "0.0"))),
        0.0,
    )
    path_label_cfg = (
        fw_cfg.get("path_dependent_labels")
        if isinstance(fw_cfg.get("path_dependent_labels"), dict)
        else {}
    )
    path_labels_enabled = bool(path_label_cfg.get("enabled", True))

    class_weights = outcome_cfg.get("class_weights", {"positive": 1.35, "neutral": 1.10, "negative": 0.95})
    regime_weights = outcome_cfg.get("regime_sample_weights", {"trend": 1.0, "mean_revert": 1.10, "shock": 1.25, "other": 1.0})

    weight_shaping_cfg = outcome_cfg.get("label_weight_shaping", {}) if isinstance(outcome_cfg.get("label_weight_shaping"), dict) else {}
    high_signal_positive_abs_bps = float(weight_shaping_cfg.get("high_signal_positive_abs_bps", os.getenv("BEHAVIOR_DATASET_HIGH_SIGNAL_POSITIVE_ABS_BPS", "12.0")))
    high_signal_positive_boost = float(weight_shaping_cfg.get("high_signal_positive_boost", os.getenv("BEHAVIOR_DATASET_HIGH_SIGNAL_POSITIVE_BOOST", "1.35")))
    high_signal_positive_event_boost = float(weight_shaping_cfg.get("high_signal_positive_event_boost", os.getenv("BEHAVIOR_DATASET_HIGH_SIGNAL_POSITIVE_EVENT_BOOST", "1.12")))
    high_signal_positive_event_proximity = float(weight_shaping_cfg.get("high_signal_positive_event_proximity", os.getenv("BEHAVIOR_DATASET_HIGH_SIGNAL_POSITIVE_EVENT_PROX", "0.55")))
    high_signal_positive_open_boost = float(weight_shaping_cfg.get("high_signal_positive_open_boost", os.getenv("BEHAVIOR_DATASET_HIGH_SIGNAL_POSITIVE_OPEN_BOOST", "1.08")))
    high_signal_positive_open_window_norm = float(weight_shaping_cfg.get("high_signal_positive_open_window_norm", os.getenv("BEHAVIOR_DATASET_HIGH_SIGNAL_POSITIVE_OPEN_WINDOW", "0.18")))

    neutral_noise_abs_bps_max = float(weight_shaping_cfg.get("neutral_noise_abs_bps_max", os.getenv("BEHAVIOR_DATASET_NEUTRAL_NOISE_ABS_BPS_MAX", "5.0")))
    neutral_noise_downweight = float(weight_shaping_cfg.get("neutral_noise_downweight", os.getenv("BEHAVIOR_DATASET_NEUTRAL_NOISE_DOWNWEIGHT", "0.68")))
    neutral_horizon_disagree_downweight = float(weight_shaping_cfg.get("neutral_horizon_disagree_downweight", os.getenv("BEHAVIOR_DATASET_NEUTRAL_HORIZON_DISAGREE_DOWNWEIGHT", "0.74")))
    neutral_noisy_regime_downweight = float(weight_shaping_cfg.get("neutral_noisy_regime_downweight", os.getenv("BEHAVIOR_DATASET_NEUTRAL_NOISY_REGIME_DOWNWEIGHT", "0.82")))
    neutral_event_downweight = float(weight_shaping_cfg.get("neutral_event_downweight", os.getenv("BEHAVIOR_DATASET_NEUTRAL_EVENT_DOWNWEIGHT", "0.76")))
    neutral_event_proximity = float(weight_shaping_cfg.get("neutral_event_proximity", os.getenv("BEHAVIOR_DATASET_NEUTRAL_EVENT_PROX", "0.65")))
    noisy_neutral_regimes = {
        str(x).strip().lower()
        for x in (weight_shaping_cfg.get("noisy_neutral_regimes") or ["mean_revert", "other"])
        if str(x).strip()
    }

    horizon_primary_s = max(int(args.horizon_seconds), 30)
    horizon_aux_s = max(int(args.aux_horizon_seconds), 0)
    aux_enabled = horizon_aux_s >= 30
    blend_alpha = _clamp(float(args.horizon_blend_alpha), 0.0, 1.0)

    decision_pattern = _routed_input_pattern(args.decision_glob, project_root=PROJECT_ROOT)
    channel_decision_pattern = _routed_input_pattern(args.channel_decision_glob, project_root=PROJECT_ROOT)
    governance_pattern = _routed_input_pattern(args.governance_glob, project_root=PROJECT_ROOT)
    pnl_pattern = _routed_input_pattern(args.pnl_attribution_glob, project_root=PROJECT_ROOT)
    paper_pattern = _routed_input_pattern(args.paper_trades_glob, project_root=PROJECT_ROOT)

    decision_paths = _resolve_glob_paths(decision_pattern, root=PROJECT_ROOT)
    if not decision_paths:
        fallback_pattern = _routed_input_pattern(
            str(PROJECT_ROOT / "decision_explanations" / "shadow*" / "decision_explanations_*.jsonl"),
            project_root=PROJECT_ROOT,
        )
        decision_paths = _resolve_glob_paths(fallback_pattern, root=PROJECT_ROOT)
    decision_paths = _filter_recent_paths(decision_paths, since_utc)

    channel_decision_paths = _resolve_glob_paths(channel_decision_pattern, root=PROJECT_ROOT)
    channel_decision_paths = _filter_recent_paths(channel_decision_paths, since_utc)
    channel_decision_paths = _limit_recent_paths(channel_decision_paths, max_files=max(int(args.channel_decision_max_files), 0))
    if channel_decision_paths:
        decision_by_path = {str(path): path for path in [*decision_paths, *channel_decision_paths]}
        decision_paths = [decision_by_path[key] for key in sorted(decision_by_path.keys())]

    governance_paths = _resolve_glob_paths(governance_pattern, root=PROJECT_ROOT)
    if not governance_paths:
        fallback_pattern = _routed_input_pattern(
            str(PROJECT_ROOT / "governance" / "shadow*" / "master_control_*.jsonl"),
            project_root=PROJECT_ROOT,
        )
        governance_paths = _resolve_glob_paths(fallback_pattern, root=PROJECT_ROOT)
    governance_paths = _filter_recent_paths(governance_paths, since_utc)

    pnl_paths = _resolve_glob_paths(pnl_pattern, root=PROJECT_ROOT)
    if not pnl_paths:
        fallback_pattern = _routed_input_pattern(
            str(PROJECT_ROOT / "governance" / "shadow*" / "shadow_pnl_attribution_*.jsonl"),
            project_root=PROJECT_ROOT,
        )
        pnl_paths = _resolve_glob_paths(fallback_pattern, root=PROJECT_ROOT)
    pnl_paths = _filter_recent_paths(pnl_paths, since_utc)

    paper_trade_paths = _resolve_glob_paths(paper_pattern, root=PROJECT_ROOT)
    if not paper_trade_paths:
        fallback_pattern = _routed_input_pattern(
            str(PROJECT_ROOT / "exports" / "trade_logs" / "**" / "paper_trades_*.jsonl"),
            project_root=PROJECT_ROOT,
        )
        paper_trade_paths = _resolve_glob_paths(fallback_pattern, root=PROJECT_ROOT)
    if not paper_trade_paths:
        paper_trade_paths = _resolve_glob_paths(str(PROJECT_ROOT / "paper_trades_*.jsonl"), root=PROJECT_ROOT)
    paper_trade_paths = _filter_recent_paths(paper_trade_paths, since_utc)

    sqlite_path = resolve_sqlite_path(args.sqlite_path) if bool(args.prefer_sql) else None

    decision_sql_rels, decision_file_fallbacks = split_paths_by_sqlite_coverage(
        project_root=PROJECT_ROOT,
        paths=decision_paths,
        sqlite_path=sqlite_path,
    )
    governance_sql_rels, governance_file_fallbacks = split_paths_by_sqlite_coverage(
        project_root=PROJECT_ROOT,
        paths=governance_paths,
        sqlite_path=sqlite_path,
    )
    pnl_sql_rels, pnl_file_fallbacks = split_paths_by_sqlite_coverage(
        project_root=PROJECT_ROOT,
        paths=pnl_paths,
        sqlite_path=sqlite_path,
    )
    paper_sql_rels, paper_file_fallbacks = split_paths_by_sqlite_coverage(
        project_root=PROJECT_ROOT,
        paths=paper_trade_paths,
        sqlite_path=sqlite_path,
    )

    snapshot_context, snapshot_meta = _snapshot_health_context(PROJECT_ROOT)
    external_context, external_meta = _external_feeds_context(PROJECT_ROOT, now_utc=now_utc)
    event_windows = _event_windows_from_env()

    governance_rows = chain(
        iter_sqlite_jsonl_rows(sqlite_path=sqlite_path, source_rels=governance_sql_rels) if governance_sql_rels else (),
        _iter_jsonl(governance_file_fallbacks, tail_bytes=max(int(args.governance_tail_bytes), 0)),
    )
    pnl_rows = chain(
        iter_sqlite_jsonl_rows(sqlite_path=sqlite_path, source_rels=pnl_sql_rels) if pnl_sql_rels else (),
        _iter_jsonl(pnl_file_fallbacks, tail_bytes=max(int(args.pnl_tail_bytes), 0)),
    )
    paper_rows = chain(
        iter_sqlite_jsonl_rows(sqlite_path=sqlite_path, source_rels=paper_sql_rels) if paper_sql_rels else (),
        _iter_jsonl(paper_file_fallbacks, tail_bytes=max(int(args.paper_trades_tail_bytes), 0)),
    )
    decision_rows = chain(
        iter_sqlite_jsonl_rows(sqlite_path=sqlite_path, source_rels=decision_sql_rels) if decision_sql_rels else (),
        _iter_jsonl(decision_file_fallbacks, tail_bytes=max(int(args.decision_tail_bytes), 0)),
    )

    gov_by_snapshot = _load_governance_index(governance_rows, since_utc=since_utc)
    exec_history = _load_exec_history(pnl_rows, since_utc=since_utc)
    paper_by_snapshot, paper_history = _load_paper_trade_context(paper_rows, since_utc=since_utc)

    by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    raw_rows = 0
    for raw_row in decision_rows:
        row = _canonical_behavior_decision_row(raw_row)
        if row is None:
            continue
        raw_rows += 1
        ts = _parse_ts(row.get("timestamp_utc"))
        if ts is None or ts < since_utc:
            continue
        if str(row.get("strategy") or "") != "grand_master_bot":
            continue

        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol:
            continue

        features = row.get("features") or {}
        last_price = _to_float(features.get("last_price"), 0.0)
        if last_price <= 0.0:
            continue

        action = _normalize_action(str(row.get("action") or "HOLD"))
        mode_label = str(row.get("mode") or "")
        snapshot_id = str((row.get("metadata") or {}).get("snapshot_id") or "").strip()

        by_symbol[symbol].append(
            {
                "timestamp_utc": ts.isoformat(),
                "ts_utc": ts,
                "ts_epoch": ts.timestamp(),
                "symbol": symbol,
                "action": action,
                "quantity": _to_float(row.get("quantity"), 0.0),
                "mode": mode_label,
                "role_idx": _role_index(mode_label),
                "snapshot_id": snapshot_id,
                "features": features,
                "last_price": last_price,
                "gates": row.get("gates") or {},
            }
        )

    examples: List[Dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    regime_counts: Dict[str, Counter] = defaultdict(Counter)
    per_symbol_kept: Counter[str] = Counter()
    per_symbol_regime_kept: Counter[Tuple[str, str]] = Counter()

    skipped_no_horizon = 0
    skipped_low_symbol_rows = 0
    skipped_symbol_cap = 0
    skipped_symbol_regime_cap = 0

    for symbol, rows in by_symbol.items():
        rows.sort(key=lambda r: r["ts_epoch"])
        if len(rows) < max(args.min_per_symbol, 2):
            skipped_low_symbol_rows += len(rows)
            continue

        j_primary = 1
        j_aux = 1
        n = len(rows)
        for i in range(n):
            base = rows[i]

            ret_primary: Optional[float] = None
            ret_aux: Optional[float] = None

            if j_primary <= i:
                j_primary = i + 1
            target_primary_ts = base["ts_epoch"] + horizon_primary_s
            while j_primary < n and rows[j_primary]["ts_epoch"] < target_primary_ts:
                j_primary += 1
            if j_primary < n:
                fut_primary = rows[j_primary]
                ret_primary = (fut_primary["last_price"] - base["last_price"]) / max(base["last_price"], 1e-6)

            if aux_enabled:
                if j_aux <= i:
                    j_aux = i + 1
                target_aux_ts = base["ts_epoch"] + horizon_aux_s
                while j_aux < n and rows[j_aux]["ts_epoch"] < target_aux_ts:
                    j_aux += 1
                if j_aux < n:
                    fut_aux = rows[j_aux]
                    ret_aux = (fut_aux["last_price"] - base["last_price"]) / max(base["last_price"], 1e-6)

            if ret_primary is None and ret_aux is None:
                skipped_no_horizon += 1
                continue

            horizon_indices = []
            if ret_primary is not None:
                horizon_indices.append(j_primary)
            if ret_aux is not None:
                horizon_indices.append(j_aux)
            path_end_index = max(horizon_indices) if horizon_indices else i
            future_prices = [
                _to_float(item.get("last_price"), 0.0)
                for item in rows[i + 1 : path_end_index + 1]
            ]

            if ret_primary is not None and ret_aux is not None:
                forward_return = (blend_alpha * ret_primary) + ((1.0 - blend_alpha) * ret_aux)
                horizon_profile = "blend"
                horizon_disagree = (abs(ret_primary) > 1e-8 and abs(ret_aux) > 1e-8 and ((ret_primary > 0.0) != (ret_aux > 0.0)))
            elif ret_primary is not None:
                forward_return = ret_primary
                horizon_profile = "primary_only"
                horizon_disagree = False
            else:
                forward_return = ret_aux if ret_aux is not None else 0.0
                horizon_profile = "aux_only"
                horizon_disagree = False

            sid = base.get("snapshot_id") or ""
            gov = gov_by_snapshot.get(sid, {})
            lag_exec = _find_last_exec_metrics(exec_history.get(symbol, []), base["ts_epoch"])
            lag_paper = _find_last_paper_metrics(paper_history.get(symbol, []), base["ts_epoch"])
            paper_snapshot = paper_by_snapshot.get(sid, {})
            observed_entry_cost_bps = max(
                _to_float(paper_snapshot.get("mean_entry_cost_bps"), 0.0),
                minimum_entry_cost_bps,
            )
            post_cost = post_cost_adjusted_forward_return(
                action=base["action"],
                forward_return=forward_return,
                entry_cost_bps=(observed_entry_cost_bps if post_cost_labels_enabled else 0.0),
                exit_cost_bps=(default_exit_cost_bps if post_cost_labels_enabled else 0.0),
                fee_bps=(fee_bps if post_cost_labels_enabled else 0.0),
            )
            post_cost_forward_return = _to_float(post_cost.get("post_cost_forward_return"), forward_return)
            path_labels = (
                _path_dependent_labels(
                    action=base["action"],
                    base_price=_to_float(base.get("last_price"), 0.0),
                    future_prices=future_prices,
                    post_cost_forward_return=post_cost_forward_return,
                    hold_opportunity_threshold=hold_neg_thr,
                )
                if path_labels_enabled
                else {
                    "path_label_ready": False,
                    "path_observation_count": 0,
                    "no_trade_counterfactual_outcome": "disabled_by_policy",
                }
            )
            label, label_conf = _label_from_forward(
                action=base["action"],
                forward_return=post_cost_forward_return,
                positive_thr=positive_thr,
                negative_thr=negative_thr,
                hold_pos_thr=hold_pos_thr,
                hold_neg_thr=hold_neg_thr,
            )

            feats, regime, label_conf_proxy = _decision_feature_vector(
                row=base,
                gov=gov,
                lag_exec=lag_exec,
                paper_snapshot=paper_snapshot,
                lag_paper=lag_paper,
                snapshot_context=snapshot_context,
                external_context=external_context,
                external_meta=external_meta,
                event_windows=event_windows,
            )

            if args.max_per_symbol > 0 and per_symbol_kept[symbol] >= args.max_per_symbol:
                skipped_symbol_cap += 1
                continue
            if args.max_per_symbol_regime > 0 and per_symbol_regime_kept[(symbol, regime)] >= args.max_per_symbol_regime:
                skipped_symbol_regime_cap += 1
                continue

            weight = (
                _to_float(class_weights.get(label), 1.0)
                * _to_float(regime_weights.get(regime), 1.0)
                * (0.5 + (0.5 * _clamp01(label_conf)))
            )
            if horizon_disagree:
                weight *= 0.88

            abs_forward_bps = abs(post_cost_forward_return) * 10000.0
            session_ctx = _session_event_context(base["ts_utc"], event_windows)
            event_proximity = _to_float(session_ctx.get("event_window_proximity"), 0.0)

            if label == "positive":
                if abs_forward_bps >= max(high_signal_positive_abs_bps, 0.0):
                    weight *= max(high_signal_positive_boost, 0.2)
                if event_proximity >= max(min(high_signal_positive_event_proximity, 1.0), 0.0):
                    weight *= max(high_signal_positive_event_boost, 0.2)
                if (
                    _to_float(session_ctx.get("session_bucket_norm"), 1.0) <= 0.5
                    and _to_float(session_ctx.get("mins_from_open_norm"), 1.0) <= max(min(high_signal_positive_open_window_norm, 1.0), 0.0)
                ):
                    weight *= max(high_signal_positive_open_boost, 0.2)
            elif label == "neutral":
                if abs_forward_bps <= max(neutral_noise_abs_bps_max, 0.0):
                    weight *= max(neutral_noise_downweight, 0.05)
                if horizon_disagree:
                    weight *= max(neutral_horizon_disagree_downweight, 0.05)
                if regime in noisy_neutral_regimes:
                    weight *= max(neutral_noisy_regime_downweight, 0.05)
                if event_proximity >= max(min(neutral_event_proximity, 1.0), 0.0):
                    weight *= max(neutral_event_downweight, 0.05)

            examples.append(
                {
                    "id": len(examples),
                    "timestamp_utc": base["timestamp_utc"],
                    "symbol": symbol,
                    "action": base["action"],
                    "regime": regime,
                    "label": label,
                    "label_confidence": round(label_conf, 6),
                    "label_confidence_proxy": round(label_conf_proxy, 6),
                    "forward_return": round(forward_return, 8),
                    "post_cost_forward_return": round(post_cost_forward_return, 8),
                    "round_trip_cost_bps": round(_to_float(post_cost.get("round_trip_cost_bps"), 0.0), 6),
                    "post_cost_label": bool(post_cost_labels_enabled and base["action"] in {"BUY", "SELL"}),
                    "forward_return_primary": (round(ret_primary, 8) if ret_primary is not None else None),
                    "forward_return_aux": (round(ret_aux, 8) if ret_aux is not None else None),
                    "horizon_seconds": int(horizon_primary_s),
                    "aux_horizon_seconds": int(horizon_aux_s if aux_enabled else 0),
                    "horizon_blend_alpha": round(blend_alpha, 4),
                    "horizon_profile": horizon_profile,
                    "horizon_disagree": bool(horizon_disagree),
                    **path_labels,
                    "sample_weight": round(max(weight, 0.05), 6),
                    "features": feats,
                }
            )
            label_counts[label] += 1
            regime_counts[regime][label] += 1
            per_symbol_kept[symbol] += 1
            per_symbol_regime_kept[(symbol, regime)] += 1

            if args.max_examples > 0 and len(examples) >= args.max_examples:
                break

        if args.max_examples > 0 and len(examples) >= args.max_examples:
            break

    payload = {
        "timestamp_utc": now_utc.isoformat(),
        "dataset_kind": "curated_decision_governance",
        "schema": "behavior_dataset_v7_post_cost_path_labels_point_in_time",
        "feature_schema_version": "trade_behavior_features_v6",
        "lookback_hours": int(args.lookback_hours),
        "horizons": {
            "primary_seconds": int(horizon_primary_s),
            "aux_seconds": int(horizon_aux_s if aux_enabled else 0),
            "blend_alpha": float(blend_alpha),
            "aux_enabled": bool(aux_enabled),
        },
        "caps": {
            "max_examples": int(args.max_examples),
            "max_per_symbol": int(args.max_per_symbol),
            "max_per_symbol_regime": int(args.max_per_symbol_regime),
            "decision_tail_bytes": int(args.decision_tail_bytes),
            "governance_tail_bytes": int(args.governance_tail_bytes),
            "pnl_tail_bytes": int(args.pnl_tail_bytes),
            "paper_trades_tail_bytes": int(args.paper_trades_tail_bytes),
            "channel_decision_max_files": int(args.channel_decision_max_files),
        },
        "source": {
            "decision_files": len(decision_paths),
            "channel_decision_files": len(channel_decision_paths),
            "governance_files": len(governance_paths),
            "pnl_attribution_files": len(pnl_paths),
            "since_utc": since_utc.isoformat(),
            "raw_decision_rows_scanned": int(raw_rows),
            "prefer_sql": bool(args.prefer_sql),
            "sqlite_path": str(sqlite_path) if sqlite_path else "",
            "decision_sql_files": len(decision_sql_rels),
            "decision_file_fallbacks": len(decision_file_fallbacks),
            "governance_sql_files": len(governance_sql_rels),
            "governance_file_fallbacks": len(governance_file_fallbacks),
            "pnl_sql_files": len(pnl_sql_rels),
            "pnl_file_fallbacks": len(pnl_file_fallbacks),
            "paper_trade_files": len(paper_trade_paths),
            "paper_trade_sql_files": len(paper_sql_rels),
            "paper_trade_file_fallbacks": len(paper_file_fallbacks),
            "storage_route": "external" if _prefer_external_storage() else "local_fallback",
            "input_patterns": {
                "decision": decision_pattern,
                "channel_decision": channel_decision_pattern,
                "governance": governance_pattern,
                "pnl_attribution": pnl_pattern,
                "paper_trades": paper_pattern,
            },
        },
        "thresholds": {
            "positive_bps": positive_bps,
            "negative_bps": negative_bps,
            "hold_positive_max_bps": hold_positive_bps,
            "hold_negative_min_bps": hold_negative_bps,
        },
        "sample_weight_shaping": {
            "high_signal_positive_abs_bps": float(high_signal_positive_abs_bps),
            "high_signal_positive_boost": float(high_signal_positive_boost),
            "high_signal_positive_event_boost": float(high_signal_positive_event_boost),
            "high_signal_positive_event_proximity": float(high_signal_positive_event_proximity),
            "high_signal_positive_open_boost": float(high_signal_positive_open_boost),
            "high_signal_positive_open_window_norm": float(high_signal_positive_open_window_norm),
            "neutral_noise_abs_bps_max": float(neutral_noise_abs_bps_max),
            "neutral_noise_downweight": float(neutral_noise_downweight),
            "neutral_horizon_disagree_downweight": float(neutral_horizon_disagree_downweight),
            "neutral_noisy_regime_downweight": float(neutral_noisy_regime_downweight),
            "neutral_event_downweight": float(neutral_event_downweight),
            "neutral_event_proximity": float(neutral_event_proximity),
            "noisy_neutral_regimes": sorted(noisy_neutral_regimes),
        },
        "feature_dim": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "retention_model": {
            "primary_training_inputs": [
                "decision_explanations",
                "governance/channels/decision",
                "governance/master_control",
                "governance/shadow_pnl_attribution",
                "exports/trade_logs/paper_trades",
            ],
            "raw_ingest_dependency": "bounded_sql_backing_only",
        },
        "lineage": {
            "feature_schema_version": "trade_behavior_features_v6",
            "builder_script": str(Path(__file__).resolve()),
        },
        "snapshot_context": {
            "features": {k: round(_to_float(v), 6) for k, v in snapshot_context.items()},
            "meta": snapshot_meta,
        },
        "external_context": {
            "features": {k: round(_to_float(v), 6) for k, v in external_context.items()},
            "meta": external_meta,
        },
        "rows": len(examples),
        "label_space": ["negative", "neutral", "positive"],
        "label_counts": dict(label_counts),
        "regime_label_counts": {k: dict(v) for k, v in regime_counts.items()},
        "label_contract": {
            "post_cost_labels_enabled": post_cost_labels_enabled,
            "minimum_entry_cost_bps": minimum_entry_cost_bps,
            "default_exit_cost_bps": default_exit_cost_bps,
            "fee_bps": fee_bps,
            "cost_source": "snapshot_realized_or_modeled_execution_cost_with_conservative_floor",
            "policy": "directional_labels_use_forward_outcomes_after_round_trip_execution_costs",
            "path_dependent_labels_enabled": path_labels_enabled,
            "path_dependent_outputs": [
                "maximum_favorable_excursion",
                "maximum_adverse_excursion",
                "mfe_bucket",
                "mae_bucket",
                "exit_timing_bucket",
                "post_entry_regime_bucket",
                "no_trade_counterfactual_outcome",
                "trade_vs_no_trade_excess_return",
            ],
            "no_trade_baseline": "cash_return_zero_before_financing_cost",
            "endpoint_only_training_allowed": False,
        },
        "skipped": {
            "no_horizon": int(skipped_no_horizon),
            "low_symbol_rows": int(skipped_low_symbol_rows),
            "symbol_cap": int(skipped_symbol_cap),
            "symbol_regime_cap": int(skipped_symbol_regime_cap),
        },
        "data": examples,
    }

    out_path = Path(args.out_file)
    publish_result = _publish_dataset(
        payload,
        out_path=out_path,
        failure_path=Path(args.failure_file),
        min_output_rows=max(int(args.min_output_rows), 1),
    )

    summary = {
        "timestamp_utc": payload["timestamp_utc"],
        "rows": payload["rows"],
        "feature_dim": payload["feature_dim"],
        "label_counts": payload["label_counts"],
        "regime_label_counts": payload["regime_label_counts"],
        "horizons": payload["horizons"],
        "caps": payload["caps"],
        "skipped": payload["skipped"],
        "source": payload["source"],
        "out_file": str(out_path),
        **publish_result,
    }

    if args.json:
        print(_json_dumps(summary, pretty=False))
    else:
        print(_json_dumps(summary, pretty=True))

    return 0 if bool(publish_result.get("published", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
