import argparse
import atexit
import copy
import fcntl
import glob
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

try:
    import numpy as np
except Exception:
    np = None


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HALT_FLAG_PATH = Path(PROJECT_ROOT) / "governance" / "health" / "GLOBAL_TRADING_HALT.flag"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.base_trader import BaseTrader
from core.execution_simulator import simulate_execution
from core.risk_engine import apply_risk_limits
from core.position_sizing import size_from_action
from core.portfolio_optimizer import allocate_quantity
from core.execution_queue import ExecutionQueue, OrderRequest
from core.coinbase_market_data import CoinbaseMarketDataClient, MarketDataAPIError
from core.derivatives_features import (
    CALENDAR_FEATURE_KEYS,
    FUTURES_FEATURE_KEYS,
    OPTIONS_FEATURE_KEYS,
    default_calendar_features,
    default_futures_features,
    default_options_features,
    parse_expiry_key_days,
    summarize_calendar_payload,
    summarize_futures_quote_features,
    summarize_option_chain,
)
from core.market_context_features import (
    BOND_REFERENCE_FEATURE_KEYS,
    BREADTH_FEATURE_KEYS,
    CREDIT_CONTEXT_FEATURE_KEYS,
    DATA_QUALITY_FEATURE_KEYS,
    EXECUTION_LAG_FEATURE_KEYS,
    NEWS_STRUCTURED_FEATURE_KEYS,
    default_bond_reference_features,
    default_breadth_features,
    default_credit_context_features,
    default_data_quality_features,
    default_execution_lag_features,
    default_structured_news_features,
    load_latest_external_context,
    summarize_bond_quote_reference_features,
    summarize_bond_reference_context,
    summarize_breadth_context,
    summarize_credit_context,
    summarize_data_quality_context,
    summarize_structured_news_items,
)
from core.runtime_layers import (
    BackpressureController,
    CanaryRollout,
    CheckpointStore,
    CircuitBreaker,
    StateCache,
    TelemetryEmitter,
    config_hash,
)
from core.accountability import enrich_log_row, safe_append_channel_batch, safe_append_jsonl_batch, safe_write_json_atomic
from core.path_registry import (
    build_shadow_context,
    channel_snapshot_path,
    classify_channel_from_path,
    default_channel_mirror_paths,
    governance_master_control_path,
    ingress_state_path,
    runtime_event_legacy_path,
)


@dataclass
class SubBot:
    bot_id: str
    weight: float
    active: bool
    reason: str
    test_accuracy: Optional[float]
    promoted: bool = False
    bot_role: str = "signal_sub_bot"


def _parse_bot_weight_boosts(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in (raw or "").split(","):
        chunk = str(part or "").strip()
        if not chunk or ":" not in chunk:
            continue
        bot_id, mult_raw = chunk.split(":", 1)
        bot_key = str(bot_id or "").strip().lower()
        if not bot_key:
            continue
        try:
            mult = float(mult_raw)
        except ValueError:
            continue
        if not math.isfinite(mult):
            continue
        out[bot_key] = min(max(mult, 0.10), 3.00)
    return out


def _bot_weight_boost_map() -> Dict[str, float]:
    raw = os.getenv(
        "MASTER_BOT_WEIGHT_BOOSTS",
        "brain_refinery_v10_seasonal:1.35,brain_refinery_v35_dmi_state_machine:1.30",
    ).strip()
    return _parse_bot_weight_boosts(raw)


def _apply_bot_weight_boost(bot_id: str, weight: float) -> float:
    boosts = _bot_weight_boost_map()
    if not boosts:
        return weight
    mult = boosts.get(str(bot_id or "").strip().lower())
    if mult is None:
        return weight
    return max(float(weight) * float(mult), 0.0)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _global_trading_halt_enabled() -> bool:
    return _env_flag("GLOBAL_TRADING_HALT", "0") or HALT_FLAG_PATH.exists()


def _route_storage_or_fail() -> bool:
    try:
        from core.storage_router import describe_storage_routing, route_runtime_storage

        routing = route_runtime_storage(PROJECT_ROOT)
        print(describe_storage_routing(routing))
        return True
    except Exception as exc:
        print(f"[StorageRoute] startup blocked err={exc}")
        return False


_SHADOW_SINGLETON_LOCK_FH: Optional[Any] = None


def _lock_safe_token(raw: str) -> str:
    out = []
    for ch in (raw or ""):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "default"


def _shadow_lock_key(broker: str) -> str:
    profile = _shadow_profile_name() or "default"
    return f"{_lock_safe_token((broker or 'unknown').lower())}_{_lock_safe_token(profile.lower())}"


def _release_shadow_singleton_lock() -> None:
    global _SHADOW_SINGLETON_LOCK_FH
    if _SHADOW_SINGLETON_LOCK_FH is None:
        return
    try:
        fcntl.flock(_SHADOW_SINGLETON_LOCK_FH.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        _SHADOW_SINGLETON_LOCK_FH.close()
    except Exception:
        pass
    _SHADOW_SINGLETON_LOCK_FH = None


def _acquire_shadow_singleton_lock(project_root: str, broker: str) -> bool:
    if os.getenv("ENABLE_SHADOW_SINGLETON_LOCK", "1").strip() != "1":
        return True

    key = _shadow_lock_key(broker)
    default_lock_path = os.path.join(project_root, "governance", "locks", f"shadow_loop_{key}.lock")
    lock_path = os.getenv("SHADOW_LOOP_LOCK_PATH", default_lock_path)

    try:
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        fh = open(lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                fh.seek(0)
                owner = fh.read().strip()
            except Exception:
                owner = "unknown"
            fh.close()
            print(f"[ShadowLock] busy lock_path={lock_path} owner={owner or 'unknown'}")
            return False

        fh.seek(0)
        fh.truncate(0)
        fh.write(f"pid={os.getpid()} started={time.time():.0f} broker={broker} profile={_shadow_profile_name() or 'default'} cmd={' '.join(sys.argv)}")
        fh.flush()

        global _SHADOW_SINGLETON_LOCK_FH
        _SHADOW_SINGLETON_LOCK_FH = fh
        atexit.register(_release_shadow_singleton_lock)
        print(f"[ShadowLock] acquired lock_path={lock_path} pid={os.getpid()} broker={broker} profile={_shadow_profile_name() or 'default'}")
        return True
    except Exception as exc:
        print(f"[ShadowLock] warning lock setup failed: {exc}")
        return True


def _enforce_data_only_lock() -> None:
    market_data_only = os.getenv("MARKET_DATA_ONLY", "1").strip()
    allow_order_execution = os.getenv("ALLOW_ORDER_EXECUTION", "0").strip()
    if market_data_only != "1" or allow_order_execution != "0":
        raise RuntimeError("This runner requires MARKET_DATA_ONLY=1 and ALLOW_ORDER_EXECUTION=0")


def _load_registry(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Master registry not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_sub_bots(registry: Dict[str, Any]) -> List[SubBot]:
    bots: List[SubBot] = []
    for row in registry.get("sub_bots", []):
        bot_id = str(row.get("bot_id"))
        base_weight = float(row.get("weight", 0.0) or 0.0)
        bots.append(
            SubBot(
                bot_id=bot_id,
                weight=_apply_bot_weight_boost(bot_id, base_weight),
                active=bool(row.get("active", False)),
                reason=str(row.get("reason", "unknown")),
                test_accuracy=float(row["test_accuracy"]) if row.get("test_accuracy") is not None else None,
                promoted=bool(row.get("promoted", False)),
                bot_role=str(row.get("bot_role", "signal_sub_bot")),
            )
        )
    if not bots:
        raise RuntimeError("No sub_bots found in master registry")
    return bots


def _fresh_registry(registry_path: str) -> tuple[Dict[str, Any], List[SubBot]]:
    registry = _load_registry(registry_path)
    return registry, _parse_sub_bots(registry)


def _bot_lane_key(bot: SubBot) -> str:
    role = str(bot.bot_role or "").strip().lower()
    bot_id = str(bot.bot_id or "").strip().lower()

    if role == "options_sub_bot" or any(tok in bot_id for tok in ("options", "greek", "iv_", "vol_surface", "put_call")):
        return "options"
    if role == "futures_sub_bot" or any(tok in bot_id for tok in ("futures", "funding", "basis", "order_book", "open_interest", "term_structure")):
        return "futures"
    if any(tok in bot_id for tok in ("long_term", "dividend_quality_compounder", "dividend_yield_trap_avoidance")):
        return "long_term"
    if any(tok in bot_id for tok in ("intraday", "scalp", "open_close", "ultrafast", "day_trade", "daytrading")):
        return "day"
    if any(tok in bot_id for tok in ("swing", "position_1m_3m", "1w_3w", "2d_5d")):
        return "swing"
    return "equities"


def _parse_lane_float_caps(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in (raw or "").split(","):
        seg = part.strip()
        if not seg or ":" not in seg:
            continue
        k, v = seg.split(":", 1)
        key = k.strip().lower()
        if not key:
            continue
        try:
            out[key] = max(float(v.strip()), 0.0)
        except Exception:
            continue
    return out


def _parse_lane_int_caps(raw: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for part in (raw or "").split(","):
        seg = part.strip()
        if not seg or ":" not in seg:
            continue
        k, v = seg.split(":", 1)
        key = k.strip().lower()
        if not key:
            continue
        try:
            out[key] = max(int(float(v.strip())), 0)
        except Exception:
            continue
    return out


def _canary_id_overrides() -> set[str]:
    raw = os.getenv("CANARY_BOT_IDS", "").strip()
    if not raw:
        return set()
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _is_canary_bot(bot: SubBot, canary_ids: set[str]) -> bool:
    if bool(bot.promoted):
        return True
    bot_id = str(bot.bot_id or "").strip().lower()
    if bot_id and bot_id in canary_ids:
        return True
    reason = str(bot.reason or "").strip().lower()
    return "canary" in reason


def _apply_canary_rollout_to_bots(bots: List[SubBot], canary_max_weight: float) -> List[SubBot]:
    if canary_max_weight <= 0.0:
        return bots

    active = [b for b in bots if b.active]
    if not active:
        return bots

    sandbox_enabled = os.getenv("CANARY_WEIGHT_SANDBOX_ENABLED", "1").strip() == "1"
    if not sandbox_enabled:
        total = sum(max(b.weight, 0.0) for b in active)
        if total > 0.0:
            for b in active:
                b.weight = max(b.weight, 0.0) / total
        return bots

    default_per_bot_cap = min(
        max(float(os.getenv("CANARY_WEIGHT_SANDBOX_MAX", str(canary_max_weight))), 0.0),
        max(canary_max_weight, 0.0),
    )
    lane_per_bot_caps = _parse_lane_float_caps(
        os.getenv(
            "CANARY_LANE_PER_BOT_CAPS",
            (
                f"equities:{default_per_bot_cap:.4f},"
                "day:0.0300,swing:0.0350,options:0.0250,futures:0.0250,long_term:0.0200"
            ),
        )
    )

    lane_share_caps = _parse_lane_float_caps(
        os.getenv(
            "CANARY_LANE_SHARE_CAPS",
            "equities:0.10,day:0.08,swing:0.08,options:0.06,futures:0.06,long_term:0.04,default:0.10",
        )
    )

    canary_ids = _canary_id_overrides()
    lane_map: Dict[str, List[SubBot]] = {}
    for b in active:
        lane = _bot_lane_key(b)
        lane_map.setdefault(lane, []).append(b)

    for lane, lane_bots in lane_map.items():
        canaries = [b for b in lane_bots if _is_canary_bot(b, canary_ids)]
        if not canaries:
            continue

        per_bot_cap = lane_per_bot_caps.get(lane, lane_per_bot_caps.get("default", default_per_bot_cap))
        per_bot_cap = min(max(per_bot_cap, 0.0), 1.0)
        for b in canaries:
            b.weight = min(max(b.weight, 0.0), per_bot_cap)

        lane_total = sum(max(b.weight, 0.0) for b in lane_bots)
        if lane_total <= 0.0:
            continue

        canary_total = sum(max(b.weight, 0.0) for b in canaries)
        share_cap = lane_share_caps.get(lane, lane_share_caps.get("default", 0.10))
        share_cap = min(max(share_cap, 0.0), 1.0)
        allowed = lane_total * share_cap

        if canary_total > allowed + 1e-12:
            scale = allowed / max(canary_total, 1e-8)
            for b in canaries:
                b.weight = max(b.weight, 0.0) * scale

            freed = canary_total - allowed
            non_canaries = [b for b in lane_bots if b not in canaries]
            non_total = sum(max(b.weight, 0.0) for b in non_canaries)
            if non_canaries and non_total > 0.0 and freed > 0.0:
                for b in non_canaries:
                    b.weight = max(b.weight, 0.0) + (max(b.weight, 0.0) / non_total) * freed

            capped_share = (sum(max(b.weight, 0.0) for b in canaries) / max(sum(max(b.weight, 0.0) for b in lane_bots), 1e-8))
            print(
                f"[CanaryLaneCap] lane={lane} canary_share_capped={capped_share:.3f} "
                f"max_share={share_cap:.3f}"
            )

    total = sum(max(b.weight, 0.0) for b in active)
    if total > 0.0:
        for b in active:
            b.weight = max(b.weight, 0.0) / total

    return bots

def _extract_quote_payload(raw: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    sym = symbol.upper()
    if sym in raw and isinstance(raw[sym], dict):
        return raw[sym]

    # Schwab keys can vary in case/syntax (e.g. "$VIX.X"). Try normalized matches.
    normalized = re.sub(r"[^A-Z0-9]", "", sym)
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        if key.upper() == sym:
            return value
        if re.sub(r"[^A-Z0-9]", "", key.upper()) == normalized:
            return value

    # Only fall back when there's exactly one quote-like dict payload.
    dict_children = [v for v in raw.values() if isinstance(v, dict)]
    if len(dict_children) == 1:
        return dict_children[0]
    return {}


def _quote_field(payload: Dict[str, Any], *keys: str) -> Any:
    if not isinstance(payload, dict):
        return None

    containers: List[Dict[str, Any]] = [payload]
    for nested in ("quote", "regular", "reference", "extended", "fundamental"):
        child = payload.get(nested)
        if isinstance(child, dict):
            containers.append(child)

    for container in containers:
        for key in keys:
            if key in container and container.get(key) is not None:
                return container.get(key)
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


_BASE_NEWS_FEATURE_KEYS = [
    "news_available",
    "news_items_30m",
    "news_items_2h",
    "news_items_24h",
    "news_sentiment",
    "news_negative_share",
    "news_positive_share",
    "news_shock_rate",
    "news_recent_impact",
]

_NEWS_FEATURE_KEYS = _BASE_NEWS_FEATURE_KEYS + list(NEWS_STRUCTURED_FEATURE_KEYS)

_NEWS_POSITIVE_TOKENS = {
    "beat", "beats", "upgrade", "upgrades", "outperform", "buy", "surge", "record", "growth", "raises", "strong", "bullish", "profit", "profits", "gain", "gains"
}

_NEWS_NEGATIVE_TOKENS = {
    "miss", "misses", "downgrade", "downgrades", "underperform", "sell", "drop", "plunge", "cut", "cuts", "weak", "bearish", "loss", "losses", "probe", "lawsuit", "bankruptcy"
}

_NEWS_SHOCK_TOKENS = {
    "guidance", "earnings", "fda", "recall", "investigation", "sec", "merger", "acquisition", "layoff", "default"
}

_LIVE_MACRO_SOURCE_TOKENS = (
    "federal reserve",
    "federalreserve.gov",
    "fomc",
    "jerome powell",
    "powell",
)


_DIVIDEND_FEATURE_KEYS = [
    "dividend_signal_available",
    "dividend_yield_pct",
    "dividend_yield_norm",
    "dividend_amount",
    "dividend_payout_ratio",
    "dividend_payout_ratio_norm",
    "dividend_ex_date_days",
    "dividend_ex_date_proximity_norm",
    "dividend_pay_date_days",
    "dividend_pay_date_proximity_norm",
    "dividend_quality_score_norm",
    "dividend_capture_entry_signal_norm",
    "dividend_capture_exit_signal_norm",
    "dividend_compound_bias_norm",
    "dividend_compound_equity",
    "dividend_compound_growth",
    "dividend_compound_drawdown_norm",
    "dividend_compound_growth_norm",
    "dividend_compound_steps_norm",
    "dividend_strategy_mode_capture",
    "dividend_strategy_mode_compound",
    "dividend_strategy_mode_hybrid",
    "dividend_safety_composite_norm",
    "dividend_ex_slippage_risk_norm",
    "dividend_tax_qualified_hold_norm",
    "dividend_growth_momentum_norm",
    "dividend_rebalance_due_norm",
]

_LONG_TERM_FEATURE_KEYS = [
    "long_term_profile_active",
    "long_term_strict_buy_hold",
    "long_term_horizon_years",
    "long_term_horizon_10y_plus",
    "long_term_10y_score_norm",
    "long_term_10y_buy_floor_norm",
    "long_term_10y_strong_buy_floor_norm",
    "long_term_quality_universe_hit",
    "long_term_quality_stability_norm",
    "long_term_quality_trend_norm",
    "long_term_quality_pullback_resilience_norm",
    "long_term_quality_range_balance_norm",
    "long_term_quality_dividend_norm",
    "long_term_quality_payout_safety_norm",
    "long_term_quality_compound_growth_norm",
    "long_term_quality_compound_safety_norm",
    "long_term_dca_interval_iters",
    "long_term_dca_iters_since_buy",
    "long_term_rebalance_band_norm",
]

_LANE_STRATEGY_FEATURE_KEYS = [
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
    "swing_post_earnings_drift_norm",
    "swing_gap_continuation_norm",
    "swing_gap_fade_norm",
    "swing_vol_compression_breakout_norm",
    "swing_sector_relative_strength_norm",
    "swing_weekly_trend_confirm_norm",
    "swing_regime_trend_norm",
    "swing_regime_chop_norm",
    "swing_regime_alignment_norm",
    "bond_duration_regime_norm",
    "bond_curve_steepener_norm",
    "bond_curve_flattener_norm",
    "bond_carry_roll_norm",
    "bond_credit_risk_on_norm",
    "bond_credit_risk_off_norm",
    "bond_inflation_breakeven_norm",
    "dividend_safety_composite_norm",
    "dividend_ex_slippage_risk_norm",
    "dividend_tax_qualified_hold_norm",
    "dividend_growth_momentum_norm",
    "dividend_rebalance_due_norm",
]

_CAPITAL_FLOW_FEATURE_KEYS = [
    "capital_flow_signed_scaled",
    "capital_flow_inflow_norm",
    "capital_flow_outflow_norm",
]

_LONG_TERM_CORE_QUALITY_DEFAULT = "SPY,VOO,VTI,IVV,QQQ,SCHX,IWB,RSP,SPLG,VUG,VTV"
_LONG_TERM_SECTOR_QUALITY_DEFAULT = "XLK,XLV,XLP,XLU,XLI,XLF,XLE,SMH,SOXX"


def _default_news_features() -> Dict[str, float]:
    out = {k: 0.0 for k in _BASE_NEWS_FEATURE_KEYS}
    out.update(default_structured_news_features())
    return out


def _default_dividend_features() -> Dict[str, float]:
    return {k: 0.0 for k in _DIVIDEND_FEATURE_KEYS}


def _default_lane_strategy_features() -> Dict[str, float]:
    return {k: 0.0 for k in _LANE_STRATEGY_FEATURE_KEYS}


def _default_capital_flow_features() -> Dict[str, float]:
    return {k: 0.0 for k in _CAPITAL_FLOW_FEATURE_KEYS}


def _clamp11(value: float) -> float:
    return max(-1.0, min(float(value), 1.0))


def _hint_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _news_ts_to_epoch(raw: Any) -> Optional[float]:
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


def _days_until_event(raw: Any, *, now_ts: float) -> float:
    ts = _news_ts_to_epoch(raw)
    if ts is None:
        s = str(raw or "").strip()
        if s and len(s) == 8 and s.isdigit():
            try:
                dt = datetime.strptime(s, "%Y%m%d").replace(tzinfo=timezone.utc)
                ts = dt.timestamp()
            except Exception:
                ts = None
    if ts is None:
        return 0.0
    return max((float(ts) - float(now_ts)) / 86400.0, 0.0)


def _extract_news_items(payload: Any, symbol: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    def _consume(value: Any) -> None:
        if isinstance(value, list):
            for row in value:
                if isinstance(row, dict):
                    items.append(row)
        elif isinstance(value, dict):
            for key in ("items", "stories", "articles", "results", "headlines", "data", "news"):
                sub = value.get(key)
                if isinstance(sub, list):
                    for row in sub:
                        if isinstance(row, dict):
                            items.append(row)

    if isinstance(payload, dict):
        for key in (symbol.upper(), symbol.lower(), "items", "stories", "articles", "results", "headlines", "data", "news"):
            if key in payload:
                _consume(payload.get(key))
        if not items:
            _consume(payload)
    else:
        _consume(payload)

    return items


def _headline_text(row: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for key in ("headline", "title", "summary", "description", "content"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            chunks.append(v.strip())
    return " ".join(chunks)


def _headline_sentiment(text: str) -> tuple[float, bool]:
    if not text:
        return 0.0, False

    toks = [t for t in re.split(r"[^a-z]+", text.lower()) if t]
    if not toks:
        return 0.0, False

    pos = sum(1 for t in toks if t in _NEWS_POSITIVE_TOKENS)
    neg = sum(1 for t in toks if t in _NEWS_NEGATIVE_TOKENS)
    shock = any(t in _NEWS_SHOCK_TOKENS for t in toks)

    if (pos + neg) <= 0:
        return 0.0, shock

    score = (pos - neg) / max(pos + neg, 1)
    return float(max(min(score, 1.0), -1.0)), shock


def _summarize_news_items(
    items: List[Dict[str, Any]],
    *,
    symbol: str,
    now_ts: float,
    lookback_seconds: float,
    max_items: int,
) -> Dict[str, float]:
    out = _default_news_features()
    if not items:
        return out

    rows: List[tuple[float, Dict[str, Any]]] = []
    for row in items:
        ts = None
        for key in ("publishedDate", "published", "dateTime", "datetime", "timestamp", "time", "displayDate"):
            ts = _news_ts_to_epoch(row.get(key))
            if ts is not None:
                break
        if ts is None:
            continue
        age = max(now_ts - ts, 0.0)
        if age > max(lookback_seconds, 60.0):
            continue
        rows.append((age, row))

    if not rows:
        return out

    rows.sort(key=lambda x: x[0])
    rows = rows[: max(max_items, 1)]

    c30 = 0
    c2h = 0
    c24h = 0
    pos_n = 0
    neg_n = 0
    shock_n = 0
    weight_sum = 0.0
    sent_sum = 0.0
    impact_sum = 0.0

    for age, row in rows:
        if age <= 30 * 60:
            c30 += 1
        if age <= 2 * 60 * 60:
            c2h += 1
        if age <= 24 * 60 * 60:
            c24h += 1

        sent, shock = _headline_sentiment(_headline_text(row))
        if sent > 0.0:
            pos_n += 1
        elif sent < 0.0:
            neg_n += 1
        if shock:
            shock_n += 1

        w = math.exp(-age / 3600.0)
        weight_sum += w
        sent_sum += w * sent
        impact_sum += w * abs(sent)

    n = len(rows)
    denom = float(max(max_items, 1))
    out.update(
        {
            "news_available": 1.0,
            "news_items_30m": min(c30 / denom, 1.0),
            "news_items_2h": min(c2h / denom, 1.0),
            "news_items_24h": min(c24h / denom, 1.0),
            "news_sentiment": (sent_sum / weight_sum) if weight_sum > 0 else 0.0,
            "news_negative_share": neg_n / max(n, 1),
            "news_positive_share": pos_n / max(n, 1),
            "news_shock_rate": shock_n / max(n, 1),
            "news_recent_impact": min(impact_sum / max(weight_sum, 1e-8), 1.0),
        }
    )
    out.update(
        summarize_structured_news_items(
            [row for _, row in rows],
            symbol=symbol,
            now_ts=now_ts,
            max_items=max_items,
        )
    )
    return out


def _live_macro_symbols(row: Dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for key in ("symbol", "ticker"):
        raw = row.get(key)
        if isinstance(raw, str) and raw.strip():
            out.add(raw.strip().upper())
    for key in ("symbols", "tickers", "relatedSymbols", "relatedTickers", "securities"):
        raw = row.get(key)
        if isinstance(raw, str) and raw.strip():
            for token in raw.replace("|", ",").split(","):
                token = token.strip().upper()
                if token:
                    out.add(token)
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    out.add(item.strip().upper())
                elif isinstance(item, dict):
                    for sub_key in ("symbol", "ticker"):
                        sub_val = item.get(sub_key)
                        if isinstance(sub_val, str) and sub_val.strip():
                            out.add(sub_val.strip().upper())
    return out


def _live_macro_active_rows(snapshot: Dict[str, Any], *, symbol: str, now_ts: float) -> List[Dict[str, Any]]:
    if not isinstance(snapshot, dict):
        return []
    active_raw = str(snapshot.get("active", "1")).strip().lower()
    if active_raw in {"0", "false", "no", "off"}:
        return []

    expires_ts = _news_ts_to_epoch(snapshot.get("expires_at_utc") or snapshot.get("expires_at"))
    if expires_ts is not None and expires_ts < now_ts:
        return []

    items = _extract_news_items(snapshot, symbol)
    if not items and any(str(snapshot.get(k) or "").strip() for k in ("headline", "title", "summary", "description", "content")):
        items = [dict(snapshot)]

    rows: List[Dict[str, Any]] = []
    sym = str(symbol or "").strip().upper()
    snapshot_broad = bool(snapshot.get("broad_market") or snapshot.get("macro_event"))
    snapshot_published = snapshot.get("published") or snapshot.get("timestamp_utc")
    for raw in items:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        if not row.get("published") and snapshot_published:
            row["published"] = snapshot_published
        if not row.get("expires_at_utc") and snapshot.get("expires_at_utc"):
            row["expires_at_utc"] = snapshot.get("expires_at_utc")

        row_expires = _news_ts_to_epoch(row.get("expires_at_utc") or row.get("expires_at"))
        if row_expires is not None and row_expires < now_ts:
            continue

        broad_market = bool(row.get("broad_market") or row.get("macro_event") or snapshot_broad)
        related = _live_macro_symbols(row)
        if sym and related and (sym not in related) and (not broad_market):
            continue
        rows.append(row)
    return rows


def _live_macro_source_quality(row: Dict[str, Any]) -> float:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("source", "publisher", "channel", "speaker", "headline", "title", "summary")
    ).lower()
    if any(token in text for token in _LIVE_MACRO_SOURCE_TOKENS):
        return 1.0
    if any(token in text for token in ("reuters", "bloomberg", "cnbc", "wall street journal", "wsj")):
        return 0.92
    return 0.75


def _merge_live_macro_news_features(
    news_features: Dict[str, float],
    *,
    symbol: str,
    now_ts: float,
    snapshot: Dict[str, Any],
) -> Dict[str, float]:
    out = dict(news_features) if isinstance(news_features, dict) else _default_news_features()
    rows = _live_macro_active_rows(snapshot, symbol=symbol, now_ts=now_ts)
    if not rows:
        return out

    manual_feats = _summarize_news_items(
        rows,
        symbol=symbol,
        now_ts=now_ts,
        lookback_seconds=48.0 * 3600.0,
        max_items=max(len(rows), 1),
    )
    for key, value in manual_feats.items():
        curr = float(out.get(key, 0.0) or 0.0)
        cand = float(value or 0.0)
        if key == "news_sentiment":
            if abs(cand) >= abs(curr):
                out[key] = _clamp11(cand)
        else:
            out[key] = max(curr, cand)

    snapshot_sentiment = _clamp11(_hint_float(snapshot.get("sentiment_hint"), 0.0))
    snapshot_shock = _clamp01(_hint_float(snapshot.get("shock_hint"), 0.0))
    for row in rows:
        ts = _news_ts_to_epoch(
            row.get("published")
            or row.get("publishedDate")
            or row.get("dateTime")
            or row.get("datetime")
            or row.get("timestamp")
            or row.get("time")
            or row.get("displayDate")
        )
        age_seconds = max(now_ts - ts, 0.0) if ts is not None else 0.0
        recency = math.exp(-age_seconds / 5400.0)

        sentiment_hint = _clamp11(_hint_float(row.get("sentiment_hint"), snapshot_sentiment))
        shock_hint = _clamp01(_hint_float(row.get("shock_hint"), snapshot_shock))
        broad_market = bool(row.get("broad_market") or row.get("macro_event") or snapshot.get("broad_market") or snapshot.get("macro_event"))
        relevance = 1.0 if broad_market else 0.85
        source_quality = _live_macro_source_quality(row)
        impact_floor = max(shock_hint, abs(sentiment_hint))
        impact_score = _clamp01(impact_floor * max(recency, 0.45))

        out["news_available"] = max(float(out.get("news_available", 0.0) or 0.0), 1.0)
        if age_seconds <= 30.0 * 60.0:
            out["news_items_30m"] = max(float(out.get("news_items_30m", 0.0) or 0.0), 1.0)
        if age_seconds <= 2.0 * 60.0 * 60.0:
            out["news_items_2h"] = max(float(out.get("news_items_2h", 0.0) or 0.0), 1.0)
        if age_seconds <= 24.0 * 60.0 * 60.0:
            out["news_items_24h"] = max(float(out.get("news_items_24h", 0.0) or 0.0), 1.0)

        current_sent = float(out.get("news_sentiment", 0.0) or 0.0)
        hinted_sent = _clamp11(sentiment_hint * max(recency, 0.55))
        if abs(hinted_sent) >= abs(current_sent):
            out["news_sentiment"] = hinted_sent
        if hinted_sent > 0.0:
            out["news_positive_share"] = max(float(out.get("news_positive_share", 0.0) or 0.0), max(recency, 0.65))
        elif hinted_sent < 0.0:
            out["news_negative_share"] = max(float(out.get("news_negative_share", 0.0) or 0.0), max(recency, 0.65))

        out["news_shock_rate"] = max(float(out.get("news_shock_rate", 0.0) or 0.0), impact_score)
        out["news_recent_impact"] = max(float(out.get("news_recent_impact", 0.0) or 0.0), impact_score)
        out["news_source_quality_norm"] = max(float(out.get("news_source_quality_norm", 0.0) or 0.0), source_quality)
        out["news_entity_relevance_norm"] = max(float(out.get("news_entity_relevance_norm", 0.0) or 0.0), relevance)
        out["news_novelty_norm"] = max(float(out.get("news_novelty_norm", 0.0) or 0.0), 1.0)

        headline = _headline_text(row).lower()
        if any(token in headline for token in ("powell", "federal reserve", "fed", "fomc", "inflation", "rates", "tariffs", "labor", "growth")):
            out["news_topic_regulatory_norm"] = max(float(out.get("news_topic_regulatory_norm", 0.0) or 0.0), 0.90 * max(recency, 0.5))

    return out


def _merge_live_macro_calendar_features(
    calendar_features: Dict[str, float],
    *,
    symbol: str,
    now_ts: float,
    snapshot: Dict[str, Any],
) -> Dict[str, float]:
    out = dict(calendar_features) if isinstance(calendar_features, dict) else default_calendar_features()
    rows = _live_macro_active_rows(snapshot, symbol=symbol, now_ts=now_ts)
    if not rows:
        return out

    max_event = 0.0
    signed_signal = 0.0
    snapshot_sentiment = _clamp11(_hint_float(snapshot.get("sentiment_hint"), 0.0))
    snapshot_shock = _clamp01(_hint_float(snapshot.get("shock_hint"), 0.0))
    for row in rows:
        ts = _news_ts_to_epoch(
            row.get("published")
            or row.get("publishedDate")
            or row.get("dateTime")
            or row.get("datetime")
            or row.get("timestamp")
            or row.get("time")
            or row.get("displayDate")
        )
        age_seconds = max(now_ts - ts, 0.0) if ts is not None else 0.0
        recency = math.exp(-age_seconds / 5400.0)

        sentiment_hint = _clamp11(_hint_float(row.get("sentiment_hint"), snapshot_sentiment))
        shock_hint = _clamp01(_hint_float(row.get("shock_hint"), snapshot_shock))
        event_score = max(0.70 * max(recency, 0.5), shock_hint * max(recency, 0.5))
        max_event = max(max_event, _clamp01(event_score))

        hinted = _clamp11(sentiment_hint * max(recency, 0.6))
        if abs(hinted) >= abs(signed_signal):
            signed_signal = hinted

    out["calendar_feed_available"] = max(float(out.get("calendar_feed_available", 0.0) or 0.0), 1.0)
    out["calendar_events_24h_norm"] = max(float(out.get("calendar_events_24h_norm", 0.0) or 0.0), max_event)
    out["calendar_high_impact_24h_norm"] = max(float(out.get("calendar_high_impact_24h_norm", 0.0) or 0.0), max_event)
    out["calendar_event_proximity_norm"] = max(float(out.get("calendar_event_proximity_norm", 0.0) or 0.0), max_event)
    out["calendar_next_event_norm"] = max(float(out.get("calendar_next_event_norm", 0.0) or 0.0), max_event)
    out["calendar_macro_event_norm"] = max(float(out.get("calendar_macro_event_norm", 0.0) or 0.0), max_event)
    out["calendar_macro_abs_surprise_norm"] = max(float(out.get("calendar_macro_abs_surprise_norm", 0.0) or 0.0), max_event)
    out["calendar_fomc_event_norm"] = max(float(out.get("calendar_fomc_event_norm", 0.0) or 0.0), max_event)

    current_signed = float(out.get("calendar_macro_surprise_norm", 0.5) or 0.5) - 0.5
    if abs(signed_signal) >= abs(current_signed):
        out["calendar_macro_surprise_norm"] = _clamp01(0.5 + (signed_signal * 0.5))

    return out


def _merge_calendar_feature_sets(
    base_features: Dict[str, float],
    extra_features: Dict[str, Any],
) -> Dict[str, float]:
    out = dict(base_features) if isinstance(base_features, dict) else default_calendar_features()
    if not isinstance(extra_features, dict):
        return out

    for key, raw_value in extra_features.items():
        if key not in CALENDAR_FEATURE_KEYS:
            continue
        value = _hint_float(raw_value, None)
        if value is None or not math.isfinite(value):
            continue

        current = _hint_float(out.get(key), 0.0)
        if key in {"calendar_macro_surprise_norm", "calendar_macro_revision_norm"}:
            current_delta = abs((current if current is not None else 0.5) - 0.5)
            value_delta = abs(value - 0.5)
            if value_delta >= current_delta:
                out[key] = _clamp01(value)
            continue

        if key == "calendar_next_event_norm":
            current_num = float(current or 0.0)
            if current_num <= 0.0:
                out[key] = _clamp01(value)
            elif value > 0.0:
                out[key] = _clamp01(min(current_num, value))
            continue

        out[key] = max(float(current or 0.0), _clamp01(value))

    return out


def _external_macro_calendar_proxy_features(project_root: str) -> Dict[str, float]:
    feats = default_calendar_features()
    try:
        try:
            from scripts.build_behavior_dataset_from_decisions import _external_feeds_context
        except Exception:
            from build_behavior_dataset_from_decisions import _external_feeds_context

        external_context, external_meta = _external_feeds_context(Path(project_root), datetime.now(timezone.utc))
        if not isinstance(external_context, dict):
            return feats

        fred_unrate = float(external_context.get("external_fred_unrate_norm", 0.0) or 0.0)
        fred_cpi = float(external_context.get("external_fred_cpi_mom_norm", 0.0) or 0.0)
        fred_gdp = float(external_context.get("external_fred_gdp_qoq_norm", 0.0) or 0.0)
        bls_unrate = float(external_context.get("external_bls_unrate_norm", 0.0) or 0.0)
        bls_cpi = float(external_context.get("external_bls_cpi_mom_norm", 0.0) or 0.0)

        macro_values = [fred_cpi, fred_gdp, bls_cpi]
        macro_values = [v for v in macro_values if math.isfinite(v) and abs(v) > 1e-9]
        macro_surprise = sum((v - 0.5) for v in macro_values) / max(len(macro_values), 1)
        labor_signal = 0.5 * ((bls_unrate - 0.5) + (fred_unrate - 0.5))
        any_macro = 1.0 if macro_values or abs(labor_signal) > 1e-6 else 0.0
        abs_surprise = min(abs(macro_surprise) * 2.0, 1.0)

        meta = external_meta if isinstance(external_meta, dict) else {}
        fred_map = meta.get("fred") if isinstance(meta.get("fred"), dict) else {}
        bls_map = meta.get("bls") if isinstance(meta.get("bls"), dict) else {}
        has_cpi = 1.0 if (fred_map.get("fred_cpi_mom") is not None or bls_map.get("bls_cpi_mom") is not None) else 0.0
        has_labor = 1.0 if (fred_map.get("fred_unrate_latest") is not None or bls_map.get("bls_unrate_latest") is not None) else 0.0

        feats.update(
            {
                "calendar_feed_available": 1.0 if any_macro > 0.0 else 0.0,
                "calendar_events_24h_norm": max(any_macro, abs_surprise),
                "calendar_high_impact_24h_norm": max(abs_surprise, min(abs(labor_signal) * 2.0, 1.0)),
                "calendar_event_proximity_norm": max(abs_surprise, min(abs(labor_signal) * 2.0, 1.0)),
                "calendar_next_event_norm": max(abs_surprise, min(abs(labor_signal) * 2.0, 1.0)),
                "calendar_macro_event_norm": any_macro,
                "calendar_macro_surprise_norm": max(min(0.5 + macro_surprise, 1.0), 0.0),
                "calendar_macro_abs_surprise_norm": abs_surprise,
                "calendar_macro_revision_norm": 0.5,
                "calendar_fomc_event_norm": 1.0 if abs(float(fred_gdp or 0.0) - 0.5) >= 0.05 else 0.0,
                "calendar_cpi_event_norm": has_cpi,
                "calendar_labor_event_norm": has_labor,
            }
        )

        te_meta = meta.get("tradingeconomics") if isinstance(meta.get("tradingeconomics"), dict) else {}
        te_calendar_rows = te_meta.get("calendar_rows") if isinstance(te_meta.get("calendar_rows"), list) else []
        if te_calendar_rows:
            te_calendar_features = summarize_calendar_payload(
                te_calendar_rows,
                now_ts=datetime.now(timezone.utc).timestamp(),
                max_items=600,
            )
            feats = _merge_calendar_feature_sets(feats, te_calendar_features)
    except Exception:
        return feats
    return feats


def _augment_news_features_with_event_proxy(
    news_features: Dict[str, float],
    *,
    market_snapshot: Dict[str, float],
    calendar_features: Dict[str, float],
) -> Dict[str, float]:
    out = dict(news_features or {})
    if float(out.get("news_available", 0.0) or 0.0) > 0.0:
        return out

    macro_abs = float(calendar_features.get("calendar_macro_abs_surprise_norm", 0.0) or 0.0)
    macro_signed = float(calendar_features.get("calendar_macro_surprise_norm", 0.0) or 0.0) - 0.5
    event_prox = float(calendar_features.get("calendar_event_proximity_norm", 0.0) or 0.0)
    high_impact = float(calendar_features.get("calendar_high_impact_24h_norm", 0.0) or 0.0)
    vol_30m = abs(float(market_snapshot.get("vol_30m", 0.0) or 0.0))
    pct_from_close = abs(float(market_snapshot.get("pct_from_close", 0.0) or 0.0))
    shock_intensity = max(macro_abs, high_impact, min((vol_30m / 0.02), 1.0), min((pct_from_close / 0.03), 1.0))

    if shock_intensity <= 0.0:
        return out

    out.update(
        {
            "news_available": max(float(out.get("news_available", 0.0) or 0.0), 0.35),
            "news_items_30m": max(float(out.get("news_items_30m", 0.0) or 0.0), min(shock_intensity, 1.0) * 0.6),
            "news_items_2h": max(float(out.get("news_items_2h", 0.0) or 0.0), min(max(shock_intensity, event_prox), 1.0) * 0.7),
            "news_items_24h": max(float(out.get("news_items_24h", 0.0) or 0.0), min(max(shock_intensity, event_prox), 1.0)),
            "news_shock_rate": max(float(out.get("news_shock_rate", 0.0) or 0.0), shock_intensity),
            "news_recent_impact": max(float(out.get("news_recent_impact", 0.0) or 0.0), shock_intensity),
            "news_sentiment": max(min(macro_signed * 2.0, 1.0), -1.0),
            "news_negative_share": max(float(out.get("news_negative_share", 0.0) or 0.0), max(-macro_signed * 2.0, 0.0)),
            "news_positive_share": max(float(out.get("news_positive_share", 0.0) or 0.0), max(macro_signed * 2.0, 0.0)),
            "news_source_quality_norm": max(float(out.get("news_source_quality_norm", 0.0) or 0.0), 0.55),
            "news_entity_relevance_norm": max(float(out.get("news_entity_relevance_norm", 0.0) or 0.0), 0.5),
            "news_novelty_norm": max(float(out.get("news_novelty_norm", 0.0) or 0.0), min(0.4 + shock_intensity * 0.5, 1.0)),
        }
    )
    return out


def _try_schwab_news_method(client: Any, method_name: str, symbol: str, limit: int) -> Optional[Any]:
    method = getattr(client, method_name, None)
    if not callable(method):
        return None

    candidates = [
        ((symbol,), {"limit": limit}),
        ((symbol,), {}),
        ((), {"symbol": symbol, "limit": limit}),
        ((), {"symbol": symbol}),
        ((), {"symbols": symbol, "limit": limit}),
        ((), {"symbols": symbol}),
    ]

    for args, kwargs in candidates:
        try:
            resp = method(*args, **kwargs)
        except TypeError:
            continue
        except Exception:
            continue

        if resp is None:
            continue
        if hasattr(resp, "json"):
            try:
                return resp.json()
            except Exception:
                continue
        return resp

    return None


def _try_schwab_option_chain_method(client: Any, method_name: str, symbol: str, strike_count: int) -> Optional[Any]:
    method = getattr(client, method_name, None)
    if not callable(method):
        return None

    candidates = [
        ((symbol,), {'strike_count': strike_count, 'include_quotes': True}),
        ((symbol,), {'strike_count': strike_count}),
        ((symbol,), {'include_quotes': True}),
        ((symbol,), {}),
        ((), {'symbol': symbol, 'strike_count': strike_count, 'include_quotes': True}),
        ((), {'symbol': symbol, 'strike_count': strike_count}),
        ((), {'symbol': symbol, 'include_quotes': True}),
        ((), {'symbol': symbol}),
    ]

    for args, kwargs in candidates:
        try:
            resp = method(*args, **kwargs)
        except TypeError:
            continue
        except Exception:
            continue

        if resp is None:
            continue
        if hasattr(resp, 'json'):
            try:
                return resp.json()
            except Exception:
                continue
        return resp

    return None


def _try_schwab_calendar_method(client: Any, method_name: str, symbol: str, days_ahead: int) -> Optional[Any]:
    method = getattr(client, method_name, None)
    if not callable(method):
        return None

    now_utc = datetime.now(timezone.utc)
    end_utc = now_utc + timedelta(days=max(days_ahead, 1))

    candidates = [
        ((), {'symbol': symbol, 'start_datetime': now_utc, 'end_datetime': end_utc}),
        ((), {'symbols': symbol, 'start_datetime': now_utc, 'end_datetime': end_utc}),
        ((), {'symbol': symbol}),
        ((), {'symbols': symbol}),
        ((), {'start_datetime': now_utc, 'end_datetime': end_utc}),
        ((symbol,), {}),
        ((), {}),
    ]

    for args, kwargs in candidates:
        try:
            resp = method(*args, **kwargs)
        except TypeError:
            continue
        except Exception:
            continue

        if resp is None:
            continue
        if hasattr(resp, 'json'):
            try:
                return resp.json()
            except Exception:
                continue
        return resp

    return None


def _row_get_ci(row: Dict[str, Any], *keys: str) -> Any:
    if not isinstance(row, dict):
        return None
    key_map = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key.lower() in key_map:
            return key_map[key.lower()]
    return None


def _try_tradingeconomics_calendar(
    *,
    days_ahead: int,
    country: str,
    importance_csv: str,
    timeout_seconds: float,
    auth_token: str,
    max_items: int,
) -> Tuple[Optional[Any], str, str]:
    base = "https://api.tradingeconomics.com/calendar"
    query = {"f": "json", "c": (auth_token or "guest:guest")}

    urls: List[Tuple[str, str]] = []
    country_clean = str(country or "").strip()
    if country_clean:
        urls.append(
            (
                f"{base}/country/{quote(country_clean, safe='')}?{urlencode(query)}",
                "tradingeconomics.calendar.country",
            )
        )
    urls.append((f"{base}?{urlencode(query)}", "tradingeconomics.calendar"))

    importance_filter = {x.strip().lower() for x in str(importance_csv or "").split(",") if x.strip()}
    now_ts = time.time()
    horizon_ts = now_ts + (max(int(days_ahead), 1) * 86400.0)

    last_endpoint = "tradingeconomics.calendar"
    last_error = "no_payload"

    for url, endpoint in urls:
        last_endpoint = endpoint
        try:
            req = Request(url, headers={"Accept": "application/json", "User-Agent": "schwab-trading-bot/1.0"})
            with urlopen(req, timeout=max(float(timeout_seconds), 1.0)) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except HTTPError as exc:
            last_error = f"HTTPError:{exc.code}:{getattr(exc, 'reason', exc)}"
            continue
        except URLError as exc:
            last_error = f"URLError:{getattr(exc, 'reason', exc)}"
            continue
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{exc}"
            continue

        try:
            payload = json.loads(raw)
        except Exception as exc:
            last_error = f"JSONDecodeError:{exc}"
            continue

        rows: Optional[List[Dict[str, Any]]] = None
        if isinstance(payload, list):
            rows = [r for r in payload if isinstance(r, dict)]
        elif isinstance(payload, dict):
            for k, v in payload.items():
                if str(k).lower() in {"data", "results", "calendar", "events", "items"} and isinstance(v, list):
                    rows = [r for r in v if isinstance(r, dict)]
                    break

        if rows is None:
            return payload, endpoint, ""

        filtered: List[Dict[str, Any]] = []
        country_lower = country_clean.lower()
        for row in rows:
            row_country = str(_row_get_ci(row, "country") or "").strip().lower()
            if country_lower and row_country and country_lower not in row_country and row_country not in country_lower:
                continue

            if importance_filter:
                imp_raw = _row_get_ci(row, "importance", "impact")
                imp = str(imp_raw or "").strip().lower()
                imp_num = ""
                try:
                    imp_num = str(int(float(imp_raw)))
                except Exception:
                    imp_num = ""
                if imp and imp not in importance_filter and (not imp_num or imp_num not in importance_filter):
                    continue

            ts = None
            for key in ("date", "eventdate", "startdate", "datetime", "time", "dateutc", "timestamp"):
                raw_ts = _row_get_ci(row, key)
                if raw_ts is None:
                    continue
                ts = _news_ts_to_epoch(raw_ts)
                if ts is not None:
                    break
            if ts is not None and (ts < now_ts - 6 * 3600 or ts > horizon_ts + 86400):
                continue

            filtered.append(row)
            if len(filtered) >= max(max_items, 50):
                break

        return filtered, endpoint, ""

    return None, last_endpoint, last_error


def _symbol_is_probably_futures(symbol: str) -> bool:
    s = (symbol or '').upper().strip()
    if not s:
        return False
    if s.startswith('/'):
        return True
    if s.endswith('=F') or s.endswith('-PERP'):
        return True
    return s in {
        'ES', 'MES', 'NQ', 'MNQ', 'YM', 'MYM', 'RTY', 'M2K', 'CL', 'MCL', 'GC', 'MGC', 'SI', 'SIL', 'HG', 'NG', 'RB', 'HO', 'ZB', 'ZN', 'ZF', 'ZT', '6E', '6J', '6B'
    }


def _is_usable_market_snapshot(mkt: Dict[str, float]) -> bool:
    # Premarket/after-hours feeds can temporarily miss last_price; allow sane fallbacks.
    if _to_float(mkt.get("last_price"), 0.0) > 0.0:
        return True
    if _to_float(mkt.get("prev_close"), 0.0) > 0.0:
        return True
    return False


def _market_snapshot_from_schwab(client: Any, symbol: str) -> Dict[str, float]:
    try:
        quote_resp = client.get_quote(symbol)
    except Exception as exc:
        raise RuntimeError(f"schwab_quote_failed symbol={symbol} err={type(exc).__name__}:{exc}") from exc

    quote_status = int(getattr(quote_resp, "status_code", 0) or 0)
    if quote_status >= 400:
        raise RuntimeError(f"schwab_quote_http_error symbol={symbol} status={quote_status}")

    try:
        quote_obj = quote_resp.json() if hasattr(quote_resp, "json") else {}
    except Exception as exc:
        raise RuntimeError(f"schwab_quote_parse_failed symbol={symbol} err={type(exc).__name__}:{exc}") from exc

    quote = _extract_quote_payload(quote_obj if isinstance(quote_obj, dict) else {}, symbol)

    now_utc = datetime.now(timezone.utc)
    now_ts = now_utc.timestamp()
    try:
        history_resp = client.get_price_history_every_minute(
            symbol,
            start_datetime=now_utc - timedelta(minutes=60),
            end_datetime=now_utc,
            need_extended_hours_data=False,
            need_previous_close=True,
        )
    except Exception as exc:
        raise RuntimeError(f"schwab_history_failed symbol={symbol} err={type(exc).__name__}:{exc}") from exc

    history_status = int(getattr(history_resp, "status_code", 0) or 0)
    if history_status >= 400:
        raise RuntimeError(f"schwab_history_http_error symbol={symbol} status={history_status}")

    try:
        hist_obj = history_resp.json() if hasattr(history_resp, "json") else {}
    except Exception as exc:
        raise RuntimeError(f"schwab_history_parse_failed symbol={symbol} err={type(exc).__name__}:{exc}") from exc

    candles = hist_obj.get("candles", []) if isinstance(hist_obj, dict) else []

    closes = [_to_float(c.get("close")) for c in candles if _to_float(c.get("close")) > 0]
    highs = [_to_float(c.get("high")) for c in candles if _to_float(c.get("high")) > 0]
    lows = [_to_float(c.get("low")) for c in candles if _to_float(c.get("low")) > 0]

    last_price = _to_float(_quote_field(quote, "lastPrice", "regularMarketLastPrice"), 0.0)
    if last_price <= 0:
        last_price = _to_float(_quote_field(quote, "mark"), 0.0)
    if last_price <= 0 and closes:
        last_price = closes[-1]

    prev_close = _to_float(_quote_field(quote, "closePrice", "previousClose"), 0.0)
    if prev_close <= 0 and len(closes) > 1:
        prev_close = closes[0]
    if prev_close <= 0:
        prev_close = max(last_price, 1.0)

    last_price, prev_close = _apply_bond_quote_quarantine(
        symbol=symbol,
        last_price=last_price,
        prev_close=prev_close,
        closes=closes,
    )

    quote_history_relative_deviation = 0.0
    quote_history_agreement_norm = 0.0
    if closes and last_price > 0.0:
        hist_last = closes[-1]
        if hist_last > 0.0:
            max_quote_dev = max(float(os.getenv("MARKET_SNAPSHOT_MAX_QUOTE_DEVIATION", "0.35")), 0.05)
            if str(symbol or "").strip().upper() in _bond_quote_symbol_universe():
                max_quote_dev = min(
                    max_quote_dev,
                    max(min(float(os.getenv("BOND_MARKET_SNAPSHOT_MAX_QUOTE_DEVIATION", "0.05")), 0.25), 0.005),
                )
            rel_dev = abs(last_price - hist_last) / max(hist_last, 1e-8)
            quote_history_relative_deviation = float(rel_dev)
            quote_history_agreement_norm = _clamp01(max(1.0 - (rel_dev / max(max_quote_dev, 1e-8)), 0.0))
            if rel_dev > max_quote_dev:
                last_price = hist_last

    pct_from_close = (last_price - prev_close) / max(prev_close, 1e-8)

    ret = []
    for i in range(1, len(closes)):
        p0 = closes[i - 1]
        p1 = closes[i]
        if p0 > 0 and p1 > 0:
            ret.append((p1 - p0) / p0)
    vol_30m = math.sqrt(sum(r * r for r in ret[-30:]) / max(len(ret[-30:]), 1))

    mom_5m = 0.0
    if len(closes) >= 6 and closes[-6] > 0:
        mom_5m = (closes[-1] - closes[-6]) / closes[-6]

    day_high = max(highs) if highs else max(last_price, prev_close)
    day_low = min(lows) if lows else min(last_price, prev_close)
    range_pos = 0.5
    if day_high > day_low:
        range_pos = (last_price - day_low) / (day_high - day_low)

    bid = _to_float(_quote_field(quote, "bidPrice", "bid"), 0.0)
    ask = _to_float(_quote_field(quote, "askPrice", "ask"), 0.0)
    bid_size = max(_to_float(_quote_field(quote, "bidSize", "bid_size", "bidQty"), 0.0), 0.0)
    ask_size = max(_to_float(_quote_field(quote, "askSize", "ask_size", "askQty"), 0.0), 0.0)
    spread_bps = 0.0
    if bid > 0.0 and ask > bid:
        spread_bps = ((ask - bid) / max(last_price, ask, bid, 1e-8)) * 10000.0

    futures_quote = summarize_futures_quote_features(
        quote,
        last_price=float(last_price),
        now_ts=now_ts,
    ) if _symbol_is_probably_futures(symbol) else default_futures_features()

    dividend_yield_pct = max(
        _to_float(
            _quote_field(
                quote,
                "divYield",
                "dividendYield",
                "forwardDividendYield",
                "trailingAnnualDividendYield",
                "yield",
            ),
            0.0,
        ),
        0.0,
    )
    if 0.0 < dividend_yield_pct <= 1.0:
        dividend_yield_pct *= 100.0
    if dividend_yield_pct > 100.0:
        dividend_yield_pct = dividend_yield_pct / 100.0

    dividend_amount = max(
        _to_float(
            _quote_field(
                quote,
                "divAmount",
                "dividendAmount",
                "annualDividendAmount",
                "dividendPayAmount",
                "nextDividendAmount",
            ),
            0.0,
        ),
        0.0,
    )

    payout_ratio = max(
        _to_float(_quote_field(quote, "payoutRatio", "dividendPayoutRatio", "payout"), 0.0),
        0.0,
    )
    if 1.5 < payout_ratio <= 100.0:
        payout_ratio = payout_ratio / 100.0

    ex_date_raw = _quote_field(quote, "exDivDate", "nextExDividendDate", "dividendExDate")
    pay_date_raw = _quote_field(quote, "nextPayDate", "nextDividendPayDate", "dividendPayDate", "payDate")
    ex_days = _days_until_event(ex_date_raw, now_ts=now_ts)
    pay_days = _days_until_event(pay_date_raw, now_ts=now_ts)

    out = {
        "last_price": last_price,
        "prev_close": prev_close,
        "pct_from_close": pct_from_close,
        "vol_30m": vol_30m,
        "mom_5m": mom_5m,
        "range_pos": range_pos,
        "spread_bps": spread_bps,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "snapshot_ts_utc": now_ts,
        "quote_history_relative_deviation": quote_history_relative_deviation,
        "quote_history_agreement_norm": quote_history_agreement_norm,
    }
    out.update(futures_quote)
    out.update(_default_dividend_features())
    out.update(default_bond_reference_features())
    out.update(
        {
            "dividend_signal_available": 1.0 if (dividend_yield_pct > 0.0 or dividend_amount > 0.0 or payout_ratio > 0.0 or ex_days > 0.0 or pay_days > 0.0) else 0.0,
            "dividend_yield_pct": float(dividend_yield_pct),
            "dividend_yield_norm": _clamp01(dividend_yield_pct / 8.0),
            "dividend_amount": float(dividend_amount),
            "dividend_payout_ratio": float(payout_ratio),
            "dividend_payout_ratio_norm": _clamp01(payout_ratio),
            "dividend_ex_date_days": float(ex_days),
            "dividend_ex_date_proximity_norm": _clamp01(1.0 - (ex_days / 10.0)) if ex_days > 0.0 else 0.0,
            "dividend_pay_date_days": float(pay_days),
            "dividend_pay_date_proximity_norm": _clamp01(1.0 - (pay_days / 20.0)) if pay_days > 0.0 else 0.0,
        }
    )
    out.update(
        summarize_bond_quote_reference_features(
            symbol=symbol,
            quote_payload=quote,
            last_price=float(last_price),
        )
    )
    return out



def _market_snapshot_from_coinbase(client: CoinbaseMarketDataClient, symbol: str) -> Dict[str, float]:
    try:
        snap = client.market_snapshot(symbol)
    except MarketDataAPIError as exc:
        raise RuntimeError(f"coinbase_market_data_failed symbol={symbol} err={exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"coinbase_market_data_failed symbol={symbol} err={type(exc).__name__}:{exc}") from exc

    out = dict(snap) if isinstance(snap, dict) else {}
    if float(out.get("snapshot_ts_utc", 0.0) or 0.0) <= 0.0:
        out["snapshot_ts_utc"] = time.time()
    out.update(_default_dividend_features())
    out.update(default_bond_reference_features())
    return out



def _market_snapshot_simulated(last_price: float) -> Dict[str, float]:
    t = time.time()
    drift = 0.0004 * math.sin(t / 200.0)
    shock = 0.0008 * math.cos(t / 31.0)
    new_price = max(1.0, last_price * (1.0 + drift + shock))
    prev_close = new_price * (1.0 - 0.0015)
    pct_from_close = (new_price - prev_close) / max(prev_close, 1e-8)
    mom_5m = 0.5 * drift + 0.5 * shock
    vol_30m = abs(shock) * 3.0
    range_pos = min(max(0.5 + 10.0 * drift, 0.0), 1.0)
    out = {
        "last_price": new_price,
        "prev_close": prev_close,
        "pct_from_close": pct_from_close,
        "vol_30m": vol_30m,
        "mom_5m": mom_5m,
        "range_pos": range_pos,
        "spread_bps": 8.0,
        "bid_size": 1000.0,
        "ask_size": 1000.0,
        "snapshot_ts_utc": time.time(),
    }
    out.update(default_futures_features())
    out.update(_default_dividend_features())
    out.update(default_bond_reference_features())
    return out



def _hash_unit(text: str) -> float:
    b = hashlib.sha256(text.encode("utf-8")).digest()[0]
    return (b / 255.0) * 2.0 - 1.0


def _calibrate_vote(raw_vote: float, scale: float = 0.9) -> float:
    return math.tanh(scale * raw_vote)


def _vote_to_score(vote: float) -> float:
    # Keep confidence bounded away from extreme saturation.
    score = 0.5 + 0.45 * vote
    return min(max(score, 0.01), 0.99)


def _is_day_trading_bot(bot: SubBot) -> bool:
    bot_id = str(bot.bot_id or "").lower()
    if not bot_id:
        return False
    # Intraday/scalp/open-close sleeves are treated as day-trading bots.
    day_tokens = ("intraday", "scalp", "open_close", "ultrafast", "day_trade", "daytrading")
    return any(tok in bot_id for tok in day_tokens)


def _is_swing_trading_bot(bot: SubBot) -> bool:
    bot_id = str(bot.bot_id or "").lower()
    if not bot_id:
        return False
    swing_tokens = ("swing", "position_1m_3m", "1w_3w", "2d_5d")
    return any(tok in bot_id for tok in swing_tokens)


def _is_short_horizon_trading_bot(bot: SubBot) -> bool:
    return _is_day_trading_bot(bot) or _is_swing_trading_bot(bot)


def _filter_long_term_registry_bots(active_bots: List[SubBot]) -> List[SubBot]:
    out: List[SubBot] = []
    for b in active_bots:
        if _is_short_horizon_trading_bot(b):
            continue
        if _is_options_sub_bot(b) or _is_futures_sub_bot(b):
            continue
        out.append(b)
    return out


def _is_options_sub_bot(bot: SubBot) -> bool:
    role = str(bot.bot_role or "").strip().lower()
    if role == "options_sub_bot":
        return True
    bot_id = str(bot.bot_id or "").strip().lower()
    return any(tok in bot_id for tok in ("options", "greek", "iv_", "vol_surface", "put_call"))


def _is_futures_sub_bot(bot: SubBot) -> bool:
    role = str(bot.bot_role or "").strip().lower()
    if role == "futures_sub_bot":
        return True
    bot_id = str(bot.bot_id or "").strip().lower()
    return any(tok in bot_id for tok in ("futures", "funding", "basis", "order_book", "open_interest", "term_structure"))


def _derivatives_super_trader_mode() -> bool:
    # Super-trader path is intentionally disabled for long-term profiles.
    return _env_flag("DERIVATIVES_SUPER_TRADER_MODE", "1") and (not _is_long_term_profile())


def _super_trader_env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _specialist_bot_signal(bot: SubBot, features: Dict[str, float], *, segment: str) -> tuple[str, float, float, List[str]]:
    acc = bot.test_accuracy if bot.test_accuracy is not None else 0.53
    super_mode = _derivatives_super_trader_mode()

    if segment == "options":
        mom = float(features.get("mom_5m", 0.0) or 0.0)
        pct = float(features.get("pct_from_close", 0.0) or 0.0)
        iv = float(features.get("options_iv_atm_norm", 0.0) or 0.0)
        skew = float(features.get("options_iv_skew_norm", 0.5) or 0.5) - 0.5
        pcr = float(features.get("options_put_call_oi_ratio_norm", 0.5) or 0.5) - 0.5
        neg_bias = float(features.get("options_negative_bias_norm", 0.0) or 0.0)
        event_prox = float(features.get("calendar_event_proximity_norm", 0.0) or 0.0)
        term = float(features.get("options_iv_term_structure_norm", 0.5) or 0.5) - 0.5

        edge = 0.55 * mom + 0.30 * pct - 0.18 * iv - 0.22 * pcr - 0.20 * neg_bias - 0.10 * event_prox - 0.08 * term + 0.12 * skew
        edge_scale = _super_trader_env_float("OPTIONS_SUPER_TRADER_EDGE_SCALE", 1.32) if super_mode else 1.0
        conf_gain = _super_trader_env_float("OPTIONS_SUPER_TRADER_CONFIDENCE_GAIN", 1.18) if super_mode else 1.0
        threshold_shift = _super_trader_env_float("OPTIONS_SUPER_TRADER_THRESHOLD_SHIFT", -0.035) if super_mode else 0.0

        edge = _calibrate_vote(edge * edge_scale, scale=0.95)
        score = _vote_to_score(edge)
        threshold = min(max(_shift_threshold(0.57) + threshold_shift, 0.45), 0.75)

        reasons = [
            "options_specialist_signal",
            f"iv={iv:.3f}",
            f"skew={skew:.3f}",
            f"pcr_norm={pcr + 0.5:.3f}",
        ]
        if super_mode:
            reasons = reasons + [
                f"options_super_trader edge_scale={edge_scale:.2f}",
                f"options_super_trader threshold_shift={threshold_shift:+.3f}",
            ]
    else:
        mom = float(features.get("mom_5m", 0.0) or 0.0)
        pct = float(features.get("pct_from_close", 0.0) or 0.0)
        imbalance = float(features.get("futures_order_book_imbalance", 0.0) or 0.0)
        depth = float(features.get("futures_depth_ratio_norm", 0.0) or 0.0)
        funding = (float(features.get("futures_funding_rate_norm", 0.5) or 0.5) - 0.5)
        basis = (float(features.get("futures_basis_bps_norm", 0.5) or 0.5) - 0.5)
        term = (float(features.get("futures_term_structure_norm", 0.5) or 0.5) - 0.5)
        neg_bias = float(features.get("futures_negative_bias_norm", 0.0) or 0.0)

        edge = 0.45 * mom + 0.30 * pct + 0.35 * imbalance + 0.10 * depth + 0.14 * basis + 0.10 * term - 0.20 * funding - 0.18 * neg_bias
        edge_scale = _super_trader_env_float("FUTURES_SUPER_TRADER_EDGE_SCALE", 1.35) if super_mode else 1.0
        conf_gain = _super_trader_env_float("FUTURES_SUPER_TRADER_CONFIDENCE_GAIN", 1.20) if super_mode else 1.0
        threshold_shift = _super_trader_env_float("FUTURES_SUPER_TRADER_THRESHOLD_SHIFT", -0.035) if super_mode else 0.0

        edge = _calibrate_vote(edge * edge_scale, scale=0.90)
        score = _vote_to_score(edge)
        threshold = min(max(_shift_threshold(0.56) + threshold_shift, 0.45), 0.75)

        reasons = [
            "futures_specialist_signal",
            f"imbalance={imbalance:.3f}",
            f"funding_norm={funding + 0.5:.3f}",
            f"basis_norm={basis + 0.5:.3f}",
        ]
        if super_mode:
            reasons = reasons + [
                f"futures_super_trader edge_scale={edge_scale:.2f}",
                f"futures_super_trader threshold_shift={threshold_shift:+.3f}",
            ]

    score = 0.5 + (score - 0.5) * min(max((0.70 + 2.0 * (acc - 0.5)), 0.55), 1.20)
    score = 0.5 + (score - 0.5) * conf_gain
    score = min(max(score, 0.01), 0.99)

    action_band = _super_trader_env_float("DERIVATIVES_SUPER_TRADER_ACTION_BAND", 0.015) if super_mode else 0.0
    buy_cut = max(threshold - action_band, 0.45)
    sell_cut = min((1.0 - threshold) + action_band, 0.55)

    if score >= buy_cut:
        action = "BUY"
    elif score <= sell_cut:
        action = "SELL"
    else:
        action = "HOLD"

    return action, score, threshold, reasons

def _specialist_signal_batch(
    bots: List[SubBot],
    features: Dict[str, float],
    *,
    segment: str,
) -> List[Tuple[SubBot, str, float, float, List[str]]]:
    out: List[Tuple[SubBot, str, float, float, List[str]]] = []
    for b in bots:
        action, score, threshold, reasons = _specialist_bot_signal(b, features, segment=segment)
        out.append((b, action, score, threshold, reasons))
    return out


def _derive_specialist_aux_features(options_rows: List[Dict[str, Any]], futures_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    def _vote(rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        num = 0.0
        den = 0.0
        for r in rows:
            w = max(float(r.get("weight", 0.0) or 0.0), 1e-6)
            a = str(r.get("action", "HOLD")).upper()
            v = 1.0 if a == "BUY" else (-1.0 if a == "SELL" else 0.0)
            num += w * v
            den += w
        return max(min(num / max(den, 1e-8), 1.0), -1.0)

    opt_vote = _vote(options_rows)
    fut_vote = _vote(futures_rows)

    if _derivatives_super_trader_mode():
        opt_gain = _super_trader_env_float("OPTIONS_SUPER_TRADER_VOTE_GAIN", 1.30)
        fut_gain = _super_trader_env_float("FUTURES_SUPER_TRADER_VOTE_GAIN", 1.35)
        opt_vote = _calibrate_vote(opt_vote * opt_gain, scale=1.00)
        fut_vote = _calibrate_vote(fut_vote * fut_gain, scale=1.00)

    return {
        "options_specialist_active": 1.0 if options_rows else 0.0,
        "futures_specialist_active": 1.0 if futures_rows else 0.0,
        "options_specialist_vote": opt_vote,
        "futures_specialist_vote": fut_vote,
    }

def _sub_bot_signal(bot: SubBot, mkt: Dict[str, float]) -> tuple[str, float, float, List[str]]:
    acc = bot.test_accuracy if bot.test_accuracy is not None else 0.50
    trend_alpha = 0.9 + 0.4 * _hash_unit(bot.bot_id + "_trend")
    mean_rev_alpha = 0.8 + 0.4 * _hash_unit(bot.bot_id + "_meanrev")
    vol_alpha = 0.7 + 0.6 * _hash_unit(bot.bot_id + "_vol")

    base = 0.5
    base += 0.28 * trend_alpha * mkt["mom_5m"]
    base += 0.18 * trend_alpha * mkt["pct_from_close"]
    base -= 0.16 * mean_rev_alpha * (mkt["range_pos"] - 0.5)
    base -= 0.08 * vol_alpha * mkt["vol_30m"]
    base += 0.10 * (acc - 0.5)

    score = min(max(base, 0.01), 0.99)
    # Per-bot confidence calibration keeps weaker/infra bots from saturating confidence.
    role_mult = 0.92 if bot.bot_role == "infrastructure_sub_bot" else 1.0
    confidence_cal = min(max((0.72 + 2.2 * (acc - 0.5)) * role_mult, 0.55), 1.20)
    score = 0.5 + (score - 0.5) * confidence_cal
    score = min(max(score, 0.01), 0.99)
    if _is_day_trading_bot(bot):
        threshold = 0.45
    else:
        threshold = _shift_threshold(0.55 if acc >= 0.52 else 0.58)

    if score >= threshold:
        action = "BUY"
        reasons = ["score_above_threshold", "trend_or_momentum_support"]
    elif score <= (1.0 - threshold):
        action = "SELL"
        reasons = ["score_below_inverse_threshold", "mean_reversion_or_weak_momentum"]
    else:
        action = "HOLD"
        reasons = ["inside_no_trade_band", "confidence_not_extreme"]

    reasons = reasons + [f"confidence_calibration={confidence_cal:.3f}"]
    return action, score, threshold, reasons


def _estimate_transaction_cost(features: Dict[str, float], action: str) -> float:
    if action == "HOLD":
        return 0.0
    vol = max(float(features.get("vol_30m", 0.0)), 0.0)
    spread_proxy = min(max(0.0005 + 2.0 * vol, 0.0005), 0.0300)
    fee_proxy = 0.0004
    return min(max(spread_proxy + fee_proxy, 0.0), 0.0350)


def _apply_transaction_cost_penalty(
    *,
    action: str,
    score: float,
    threshold: float,
    reasons: List[str],
    features: Dict[str, float],
) -> tuple[str, float, List[str]]:
    tx_cost = _estimate_transaction_cost(features, action)
    if action == "HOLD" or tx_cost <= 0.0:
        return action, score, reasons

    sensitivity = float(os.getenv("TX_COST_PENALTY_SENSITIVITY", "0.85"))
    edge = abs(score - 0.5)
    adj_edge = max(0.0, edge - (sensitivity * tx_cost))
    score = 0.5 + (1.0 if score >= 0.5 else -1.0) * adj_edge
    score = min(max(score, 0.01), 0.99)

    if score >= threshold:
        action = "BUY"
    elif score <= (1.0 - threshold):
        action = "SELL"
    else:
        action = "HOLD"

    return action, score, reasons + [f"tx_cost_penalty={tx_cost:.4f}"]


def _enforce_cross_symbol_exposure_cap(
    *,
    action: str,
    score: float,
    reasons: List[str],
    exposure_state: Dict[str, int],
    max_long: int,
    max_short: int,
) -> tuple[str, float, List[str]]:
    if action == "BUY":
        if exposure_state.get("BUY", 0) >= max_long:
            return "HOLD", 0.5 + 0.5 * (score - 0.5), reasons + ["exposure_cap_long_reached"]
        exposure_state["BUY"] = exposure_state.get("BUY", 0) + 1
    elif action == "SELL":
        if exposure_state.get("SELL", 0) >= max_short:
            return "HOLD", 0.5 + 0.5 * (score - 0.5), reasons + ["exposure_cap_short_reached"]
        exposure_state["SELL"] = exposure_state.get("SELL", 0) + 1
    return action, score, reasons


def _runtime_lane_key(*, symbol: str, broker: str, profile: str, symbol_is_futures: bool) -> str:
    prof = (profile or "").strip().lower()
    sym = (symbol or "").strip().upper()
    if "long_term" in prof:
        return "long_term"
    if symbol_is_futures or ("futures" in prof) or sym.startswith("/"):
        return "futures"
    if "options" in prof:
        return "options"
    if any(tok in prof for tok in ("swing", "position")):
        return "swing"
    if any(tok in prof for tok in ("day", "intraday", "aggressive", "scalp")):
        return "day"
    if broker == "coinbase":
        return "futures"
    return "equities"


def _lane_budget_multiplier(lane: str) -> float:
    caps = _parse_lane_float_caps(
        os.getenv(
            "PORTFOLIO_LANE_BUDGET_MULTIPLIERS",
            "equities:1.00,day:0.90,swing:0.95,options:0.75,futures:0.70,long_term:1.10,default:1.00",
        )
    )
    val = float(caps.get((lane or "").strip().lower(), caps.get("default", 1.0)))
    return min(max(val, 0.0), 2.5)


def _parse_symbol_label_map(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in (raw or "").split(","):
        seg = part.strip()
        if (not seg) or (":" not in seg):
            continue
        k, v = seg.split(":", 1)
        sym = k.strip().upper()
        label = v.strip().lower()
        if sym and label:
            out[sym] = label
    return out


def _load_manual_trade_overrides(project_root: str, broker: str) -> Dict[str, Any]:
    candidates = [
        os.path.join(project_root, "governance", _shadow_profile_subdir(broker=broker), "manual_trade_overrides_latest.json"),
        os.path.join(project_root, "governance", _shadow_profile_subdir(broker=broker), "manual_trade_overrides.json"),
        os.path.join(project_root, "governance", "health", "manual_trade_overrides_latest.json"),
    ]
    for path in candidates:
        try:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                payload["_path"] = path
                return payload
        except Exception:
            continue
    return {"symbols": {}}


def _apply_manual_trade_reconciler(
    *,
    symbol: str,
    action: str,
    score: float,
    threshold: float,
    reasons: List[str],
    manual_payload: Dict[str, Any],
) -> tuple[str, float, List[str], Dict[str, Any]]:
    symbols_map = manual_payload.get("symbols", {})
    if not isinstance(symbols_map, dict):
        symbols_map = {}
    row = symbols_map.get(symbol.upper(), {})
    if not isinstance(row, dict) or not row:
        return action, score, reasons, {
            "active": False,
            "mode": "none",
            "position_qty": 0.0,
            "covered_call_shares_override": None,
            "source_file": str(manual_payload.get("_path", "")),
        }

    mode = str(row.get("mode", "observe") or "observe").strip().lower()
    if not mode:
        mode = "observe"
    position_qty = float(row.get("position_qty", row.get("shares", 0.0)) or 0.0)
    covered_override = row.get("covered_call_shares")
    if covered_override is None and position_qty > 0:
        covered_override = int(max(position_qty, 0.0))

    new_action = str(action).upper()
    new_score = float(score)
    new_reasons = list(reasons)
    blocked = False

    freeze_new = bool(row.get("freeze_new_entries", False))
    block_buy = bool(row.get("block_buy", False))
    block_sell = bool(row.get("block_sell", False))
    block_open = bool(row.get("block_open", False))

    if mode in {"hold_only", "freeze", "manual_only"}:
        if new_action in {"BUY", "SELL"}:
            new_action, new_score = _force_action_score("HOLD", new_score, threshold)
            blocked = True
            new_reasons = new_reasons + [f"manual_trade_reconcile_mode={mode}"]
    elif mode == "bias_long" and new_action == "SELL":
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        blocked = True
        new_reasons = new_reasons + ["manual_trade_reconcile_bias_long"]
    elif mode == "bias_short" and new_action == "BUY":
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        blocked = True
        new_reasons = new_reasons + ["manual_trade_reconcile_bias_short"]
    elif mode == "flat_only" and new_action in {"BUY", "SELL"} and abs(position_qty) > 0.0:
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        blocked = True
        new_reasons = new_reasons + ["manual_trade_reconcile_flat_only"]

    if freeze_new and new_action in {"BUY", "SELL"}:
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        blocked = True
        new_reasons = new_reasons + ["manual_trade_reconcile_freeze_new_entries"]
    if block_open and new_action in {"BUY", "SELL"}:
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        blocked = True
        new_reasons = new_reasons + ["manual_trade_reconcile_block_open"]
    if block_buy and new_action == "BUY":
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        blocked = True
        new_reasons = new_reasons + ["manual_trade_reconcile_block_buy"]
    if block_sell and new_action == "SELL":
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        blocked = True
        new_reasons = new_reasons + ["manual_trade_reconcile_block_sell"]

    return new_action, new_score, new_reasons, {
        "active": True,
        "mode": mode,
        "position_qty": float(position_qty),
        "blocked": bool(blocked),
        "covered_call_shares_override": int(covered_override) if covered_override is not None else None,
        "source_file": str(manual_payload.get("_path", "")),
    }


def _apply_execution_guard(
    *,
    action: str,
    score: float,
    threshold: float,
    reasons: List[str],
    features: Dict[str, float],
    symbol_is_futures: bool,
    broker: str = "",
) -> tuple[str, float, List[str], Dict[str, Any]]:
    if action not in {"BUY", "SELL"}:
        return action, score, reasons, {
            "ok": True,
            "market_kind": "none",
            "spread_ok": True,
            "depth_ok": True,
            "tx_cost_ok": True,
            "latency_ok": True,
            "imbalance_ok": True,
            "reason": "not_trade_action",
        }

    market_kind = "crypto" if str(broker or "").strip().lower() == "coinbase" else ("futures" if symbol_is_futures else "equities")
    spread_bps = float(features.get("spread_bps", 0.0) or 0.0)
    if symbol_is_futures and spread_bps <= 0.0:
        spread_norm = float(features.get("futures_spread_bps_norm", 0.0) or 0.0)
        spread_bps = max(spread_norm, 0.0) * 40.0
    depth_norm = float(
        features.get(
            "futures_depth_ratio_norm",
            features.get("queue_depth_norm", features.get("queue_depth", 0.0)),
        )
        or 0.0
    )
    tx_cost_bps = _estimate_transaction_cost(features, action) * 10000.0
    market_data_latency_ms = float(features.get("market_data_latency_ms", 0.0) or 0.0)
    imbalance = float(features.get("futures_order_book_imbalance", 0.0) or 0.0)
    adverse_imbalance = max((-imbalance if action == "BUY" else imbalance), 0.0)

    max_spread = float(
        os.getenv(
            f"EXEC_GUARD_MAX_SPREAD_BPS_{market_kind.upper()}",
            "22" if market_kind == "crypto" else ("28" if market_kind == "futures" else "35"),
        )
    )
    min_depth = float(
        os.getenv(
            f"EXEC_GUARD_MIN_DEPTH_NORM_{market_kind.upper()}",
            "0.08" if market_kind == "crypto" else ("0.10" if market_kind == "futures" else "0.00"),
        )
    )
    max_tx_cost = float(
        os.getenv(
            f"EXEC_GUARD_MAX_TX_COST_BPS_{market_kind.upper()}",
            os.getenv("EXEC_GUARD_MAX_TX_COST_BPS", "24" if market_kind == "crypto" else "26"),
        )
    )
    max_market_data_latency_ms = float(
        os.getenv(
            f"EXEC_GUARD_MAX_MARKET_DATA_LATENCY_MS_{market_kind.upper()}",
            "1500" if market_kind == "crypto" else ("1200" if market_kind == "futures" else "900"),
        )
    )
    max_adverse_imbalance = float(
        os.getenv(
            f"EXEC_GUARD_MAX_ADVERSE_IMBALANCE_{market_kind.upper()}",
            "0.45" if market_kind in {"crypto", "futures"} else "1.10",
        )
    )

    spread_ok = (spread_bps <= max_spread) if spread_bps > 0.0 else True
    depth_ok = depth_norm >= min_depth
    tx_cost_ok = tx_cost_bps <= max_tx_cost
    latency_ok = (market_data_latency_ms <= max_market_data_latency_ms) if market_data_latency_ms > 0.0 else True
    imbalance_ok = adverse_imbalance <= max_adverse_imbalance
    ok = spread_ok and depth_ok and tx_cost_ok and latency_ok and imbalance_ok

    if ok:
        return action, score, reasons, {
            "ok": True,
            "market_kind": market_kind,
            "spread_ok": spread_ok,
            "depth_ok": depth_ok,
            "tx_cost_ok": tx_cost_ok,
            "latency_ok": latency_ok,
            "imbalance_ok": imbalance_ok,
            "spread_bps": spread_bps,
            "depth_norm": depth_norm,
            "tx_cost_bps": tx_cost_bps,
            "market_data_latency_ms": market_data_latency_ms,
            "max_market_data_latency_ms": max_market_data_latency_ms,
            "adverse_imbalance": adverse_imbalance,
            "max_adverse_imbalance": max_adverse_imbalance,
        }

    new_action, new_score = _force_action_score("HOLD", score, threshold)
    new_reasons = list(reasons) + [
        (
            "execution_guard_block "
            f"market_kind={market_kind} "
            f"spread_ok={int(spread_ok)} depth_ok={int(depth_ok)} tx_cost_ok={int(tx_cost_ok)} "
            f"latency_ok={int(latency_ok)} imbalance_ok={int(imbalance_ok)} "
            f"spread_bps={spread_bps:.2f}/{max_spread:.2f} "
            f"depth_norm={depth_norm:.3f}/{min_depth:.3f} "
            f"tx_cost_bps={tx_cost_bps:.2f}/{max_tx_cost:.2f} "
            f"market_data_latency_ms={market_data_latency_ms:.1f}/{max_market_data_latency_ms:.1f} "
            f"adverse_imbalance={adverse_imbalance:.3f}/{max_adverse_imbalance:.3f}"
        )
    ]
    return new_action, new_score, new_reasons, {
        "ok": False,
        "market_kind": market_kind,
        "spread_ok": spread_ok,
        "depth_ok": depth_ok,
        "tx_cost_ok": tx_cost_ok,
        "latency_ok": latency_ok,
        "imbalance_ok": imbalance_ok,
        "spread_bps": spread_bps,
        "depth_norm": depth_norm,
        "tx_cost_bps": tx_cost_bps,
        "market_data_latency_ms": market_data_latency_ms,
        "max_market_data_latency_ms": max_market_data_latency_ms,
        "adverse_imbalance": adverse_imbalance,
        "max_adverse_imbalance": max_adverse_imbalance,
    }


def _apply_portfolio_risk_engine_caps(
    *,
    symbol: str,
    broker: str,
    lane: str,
    action: str,
    qty: float,
    last_price: float,
    equity_proxy: float,
    state: Dict[str, Any],
    sector_map: Dict[str, str],
) -> tuple[float, Dict[str, Any]]:
    if action not in {"BUY", "SELL"} or qty <= 0.0 or last_price <= 0.0 or equity_proxy <= 0.0:
        return max(float(qty), 0.0), {
            "applied": False,
            "reason": "notional_not_required",
        }

    proposed_notional = max(float(qty), 0.0) * max(float(last_price), 0.0)
    if proposed_notional <= 0.0:
        return 0.0, {"applied": False, "reason": "zero_proposed_notional"}

    lane_key = (lane or "equities").strip().lower()
    sector = sector_map.get(symbol.upper(), "other")
    broker_key = (broker or "schwab").strip().lower()

    symbol_cap = float(os.getenv("PORTFOLIO_MAX_SYMBOL_NOTIONAL_PCT", "0.12"))
    sector_cap = float(os.getenv("PORTFOLIO_MAX_SECTOR_NOTIONAL_PCT", "0.30"))
    gross_cap = float(os.getenv("PORTFOLIO_MAX_GROSS_NOTIONAL_PCT", "0.95"))
    lane_caps = _parse_lane_float_caps(
        os.getenv(
            "PORTFOLIO_MAX_LANE_NOTIONAL_PCT",
            "equities:0.30,day:0.22,swing:0.24,options:0.14,futures:0.18,long_term:0.35,default:0.25",
        )
    )
    broker_caps = _parse_lane_float_caps(
        os.getenv(
            "PORTFOLIO_MAX_BROKER_NOTIONAL_PCT",
            "schwab:0.80,coinbase:0.45,default:0.80",
        )
    )

    sym_used = float((state.get("symbol_notional") or {}).get(symbol.upper(), 0.0) or 0.0)
    lane_used = float((state.get("lane_notional") or {}).get(lane_key, 0.0) or 0.0)
    sector_used = float((state.get("sector_notional") or {}).get(sector, 0.0) or 0.0)
    broker_used = float((state.get("broker_notional") or {}).get(broker_key, 0.0) or 0.0)
    gross_used = float(state.get("gross_notional", 0.0) or 0.0)

    lane_cap = float(lane_caps.get(lane_key, lane_caps.get("default", 0.25)))
    broker_cap = float(broker_caps.get(broker_key, broker_caps.get("default", 0.80)))

    sym_headroom = max((symbol_cap * equity_proxy) - sym_used, 0.0)
    lane_headroom = max((lane_cap * equity_proxy) - lane_used, 0.0)
    sector_headroom = max((sector_cap * equity_proxy) - sector_used, 0.0)
    broker_headroom = max((broker_cap * equity_proxy) - broker_used, 0.0)
    gross_headroom = max((gross_cap * equity_proxy) - gross_used, 0.0)

    allowed_notional = min(sym_headroom, lane_headroom, sector_headroom, broker_headroom, gross_headroom)
    if allowed_notional <= 0.0:
        return 0.0, {
            "applied": True,
            "blocked": True,
            "symbol_cap_pct": symbol_cap,
            "lane_cap_pct": lane_cap,
            "sector_cap_pct": sector_cap,
            "broker_cap_pct": broker_cap,
            "gross_cap_pct": gross_cap,
            "allowed_notional": 0.0,
            "proposed_notional": proposed_notional,
            "sector": sector,
            "lane": lane_key,
            "broker": broker_key,
        }

    scale = min(1.0, allowed_notional / max(proposed_notional, 1e-9))
    approved_qty = round(max(float(qty), 0.0) * scale, 6)
    approved_notional = approved_qty * max(float(last_price), 0.0)
    if approved_notional > 0.0:
        state.setdefault("symbol_notional", {})[symbol.upper()] = sym_used + approved_notional
        state.setdefault("lane_notional", {})[lane_key] = lane_used + approved_notional
        state.setdefault("sector_notional", {})[sector] = sector_used + approved_notional
        state.setdefault("broker_notional", {})[broker_key] = broker_used + approved_notional
        state["gross_notional"] = gross_used + approved_notional

    return approved_qty, {
        "applied": True,
        "blocked": approved_qty <= 0.0,
        "symbol_cap_pct": symbol_cap,
        "lane_cap_pct": lane_cap,
        "sector_cap_pct": sector_cap,
        "broker_cap_pct": broker_cap,
        "gross_cap_pct": gross_cap,
        "allowed_notional": allowed_notional,
        "proposed_notional": proposed_notional,
        "approved_notional": approved_notional,
        "scale": scale,
        "sector": sector,
        "lane": lane_key,
        "broker": broker_key,
    }


def _apply_long_term_turnover_cap(
    *,
    qty: float,
    last_price: float,
    equity_proxy: float,
    state: Dict[str, Any],
) -> tuple[float, Dict[str, Any]]:
    if (not _is_long_term_profile()) or qty <= 0.0 or last_price <= 0.0 or equity_proxy <= 0.0:
        return max(float(qty), 0.0), {"applied": False}

    day_key = datetime.now(timezone.utc).strftime("%Y%m%d")
    if str(state.get("day_key", "")) != day_key:
        state["day_key"] = day_key
        state["used_turnover_pct"] = 0.0

    used_turnover_pct = float(state.get("used_turnover_pct", 0.0) or 0.0)
    max_turnover_pct = max(float(os.getenv("LONG_TERM_MAX_TURNOVER_PCT", "0.08")), 0.01)

    proposed_notional = max(float(qty), 0.0) * max(float(last_price), 0.0)
    proposed_pct = proposed_notional / max(float(equity_proxy), 1e-9)
    headroom_pct = max(max_turnover_pct - used_turnover_pct, 0.0)

    if headroom_pct <= 0.0:
        return 0.0, {
            "applied": True,
            "blocked": True,
            "used_turnover_pct": used_turnover_pct,
            "max_turnover_pct": max_turnover_pct,
            "headroom_pct": 0.0,
            "proposed_pct": proposed_pct,
        }

    scale = min(1.0, headroom_pct / max(proposed_pct, 1e-9))
    approved_qty = round(max(float(qty), 0.0) * scale, 6)
    approved_notional = approved_qty * max(float(last_price), 0.0)
    approved_pct = approved_notional / max(float(equity_proxy), 1e-9)
    state["used_turnover_pct"] = used_turnover_pct + approved_pct

    return approved_qty, {
        "applied": True,
        "blocked": approved_qty <= 0.0,
        "used_turnover_pct": float(state.get("used_turnover_pct", 0.0) or 0.0),
        "max_turnover_pct": max_turnover_pct,
        "headroom_pct": headroom_pct,
        "proposed_pct": proposed_pct,
        "approved_pct": approved_pct,
        "scale": scale,
    }


def _event_blackout_windows() -> List[tuple[int, int]]:
    raw = os.getenv(
        "EVENT_LOCK_WINDOWS_ET",
        os.getenv("EVENT_BLACKOUT_WINDOWS_ET", "08:29-08:36,09:59-10:06,13:58-14:05,15:57-16:05"),
    ).strip()
    windows: List[tuple[int, int]] = []
    if not raw:
        return windows
    for part in raw.split(","):
        seg = part.strip()
        if "-" not in seg:
            continue
        start_s, end_s = seg.split("-", 1)
        try:
            sh, sm = [int(x) for x in start_s.split(":", 1)]
            eh, em = [int(x) for x in end_s.split(":", 1)]
            windows.append((sh * 60 + sm, eh * 60 + em))
        except Exception:
            continue
    return windows


def _in_event_blackout(now_et: datetime, windows: List[tuple[int, int]]) -> bool:
    if not windows:
        return False
    now_min = now_et.hour * 60 + now_et.minute
    for start_min, end_min in windows:
        if start_min <= end_min:
            if start_min <= now_min <= end_min:
                return True
        else:
            if now_min >= start_min or now_min <= end_min:
                return True
    return False


def _now_eastern() -> datetime:
    # Normalize to New York market clock without external deps.
    now_et = datetime.now(timezone.utc).astimezone().astimezone(datetime.now().astimezone().tzinfo)
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    except Exception:
        pass
    return now_et


def _feature_freshness_guard(
    features: Dict[str, float],
    *,
    max_age_seconds: float,
    required_keys: List[str],
) -> tuple[bool, str, float]:
    ts = float(features.get("snapshot_ts_utc", 0.0) or 0.0)
    if ts <= 0.0:
        return False, "missing_snapshot_ts", 1e9

    age = max(time.time() - ts, 0.0)
    if age > max(max_age_seconds, 0.1):
        return False, f"stale_snapshot age={age:.2f}s>{max_age_seconds:.2f}s", age

    for key in required_keys:
        val = float(features.get(key, 0.0) or 0.0)
        if key in {"last_price", "prev_close"}:
            if val <= 0.0:
                return False, f"missing_or_nonpositive_feature:{key}", age
        elif not math.isfinite(val):
            return False, f"non_finite_feature:{key}", age

    return True, f"fresh age={age:.2f}s", age


def _run_preopen_replay_sanity_check(
    *,
    project_root: str,
    timeout_seconds: int,
) -> tuple[bool, str]:
    script = Path(project_root) / "scripts" / "replay_preopen_sanity_check.py"
    if not script.exists():
        return False, f"missing_script:{script}"

    cmd = [sys.executable, str(script), "--hours", "24", "--json"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(timeout_seconds, 5),
        )
    except subprocess.TimeoutExpired:
        return False, "replay_sanity_timeout"

    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        return False, f"replay_sanity_failed rc={proc.returncode} out={out[:300]}"

    try:
        payload = json.loads(out) if out else {}
    except Exception:
        payload = {}
    if not bool(payload.get("ok", False)):
        return False, f"replay_sanity_not_ok failed={','.join(payload.get('failed_checks', []) or [])}"
    return True, "ok"


def _snapshot_debug_path(project_root: str, broker: Optional[str] = None) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(project_root, "governance", _shadow_profile_subdir(broker=broker), f"snapshot_debug_{day}.jsonl")


def _pnl_attribution_path(project_root: str, broker: Optional[str] = None) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(project_root, "governance", _shadow_profile_subdir(broker=broker), f"shadow_pnl_attribution_{day}.jsonl")


def _derive_flash_aux_features(sub_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    flash_rows = [r for r in sub_rows if "flash_crash" in str(r.get("bot_id", ""))]
    if not flash_rows:
        return {
            "aux_flash_score": 0.5,
            "aux_flash_direction": 0.0,
            "aux_flash_active": 0.0,
        }

    lead = max(flash_rows, key=lambda x: float(x.get("weight", 0.0)))
    action = str(lead.get("action", "HOLD"))
    direction = 1.0 if action == "BUY" else (-1.0 if action == "SELL" else 0.0)
    return {
        "aux_flash_score": float(lead.get("score", 0.5)),
        "aux_flash_direction": direction,
        "aux_flash_active": 1.0,
    }


def _weighted_master_vote(decisions: List[Dict[str, Any]]) -> tuple[str, float, float, List[str], Dict[str, float]]:
    if not decisions:
        return "HOLD", 0.5, 0.55, ["no_active_sub_bots"], {"vote": 0.0}

    net = 0.0
    total_w = 0.0
    for d in decisions:
        w = float(d["weight"])
        a = d["action"]
        vote = 1.0 if a == "BUY" else (-1.0 if a == "SELL" else 0.0)
        net += w * vote
        total_w += w

    net = net / max(total_w, 1e-8)
    net = _calibrate_vote(net, scale=0.9)
    score = _vote_to_score(net)
    threshold = _shift_threshold(0.60)

    if net > 0.22:
        action = "BUY"
    elif net < -0.22:
        action = "SELL"
    else:
        action = "HOLD"

    top = sorted(decisions, key=lambda x: abs(float(x["weight"]) * x["direction"]), reverse=True)[:3]
    top_text = ",".join(f"{x['bot_id']}:{x['action']}" for x in top) if top else "none"
    reasons = ["weighted_ensemble_vote", f"top_contributors={top_text}"]

    return action, score, threshold, reasons, {"vote": net}


def _master_vote_variant(
    name: str,
    decisions: List[Dict[str, Any]],
    features: Dict[str, float],
) -> tuple[str, float, float, List[str], Dict[str, float]]:
    base_action, base_score, base_threshold, base_reasons, base_vote = _weighted_master_vote(decisions)

    mom = float(features.get("mom_5m", 0.0))
    pct = float(features.get("pct_from_close", 0.0))
    vol = float(features.get("vol_30m", 0.0))
    range_pos = float(features.get("range_pos", 0.5))
    vix_pct = float(features.get("ctx_VIX_X_pct_from_close", 0.0))

    vote = float(base_vote.get("vote", 0.0))
    options_vote = float(features.get("options_specialist_vote", 0.0) or 0.0)
    futures_vote = float(features.get("futures_specialist_vote", 0.0) or 0.0)
    options_neg_bias = float(features.get("options_negative_bias_norm", 0.0) or 0.0)
    futures_neg_bias = float(features.get("futures_negative_bias_norm", 0.0) or 0.0)

    if _derivatives_super_trader_mode():
        master_vote_mult = _super_trader_env_float("DERIVATIVES_SUPER_TRADER_MASTER_VOTE_MULT", 1.55)
        risk_penalty_mult = _super_trader_env_float("DERIVATIVES_SUPER_TRADER_RISK_PENALTY_MULT", 0.90)
    else:
        master_vote_mult = 1.0
        risk_penalty_mult = 1.0

    options_vote *= master_vote_mult
    futures_vote *= master_vote_mult
    options_neg_bias = _clamp01(options_neg_bias * risk_penalty_mult)
    futures_neg_bias = _clamp01(futures_neg_bias * risk_penalty_mult)

    if name == "trend":
        vote += 0.40 * mom + 0.25 * pct + 0.10 * options_vote + 0.12 * futures_vote
        reasons = base_reasons + ["trend_master_bias"]
    elif name == "mean_revert":
        vote += -0.45 * (range_pos - 0.5) - 0.25 * mom + 0.10 * (options_neg_bias - 0.5)
        reasons = base_reasons + ["mean_revert_master_bias"]
    elif name == "shock":
        vote += -0.35 * vix_pct - 0.20 * vol + 0.30 * float(features.get("aux_flash_direction", 0.0))
        vote += 0.10 * futures_vote - 0.12 * max(options_neg_bias, futures_neg_bias)
        reasons = base_reasons + ["shock_master_bias"]
    else:
        reasons = base_reasons + ["default_master_bias"]

    if _derivatives_super_trader_mode():
        reasons = reasons + [
            f"derivatives_super_trader master_vote_mult={master_vote_mult:.2f}",
            f"derivatives_super_trader risk_penalty_mult={risk_penalty_mult:.2f}",
        ]

    vote = _calibrate_vote(vote, scale=0.85)
    score = _vote_to_score(vote)
    threshold = _shift_threshold(0.58)
    if vote > 0.20:
        action = "BUY"
    elif vote < -0.20:
        action = "SELL"
    else:
        action = "HOLD"

    return action, score, threshold, reasons, {"vote": vote}


def _grand_master_weights(features: Dict[str, float]) -> Dict[str, float]:
    trend_strength = abs(float(features.get("mom_5m", 0.0))) + abs(float(features.get("pct_from_close", 0.0)))
    chop_strength = abs(float(features.get("range_pos", 0.5)) - 0.5)
    shock_strength = abs(float(features.get("ctx_VIX_X_pct_from_close", 0.0))) + float(features.get("vol_30m", 0.0))
    shock_strength += 0.45 * float(features.get("options_vol_expectation_norm", 0.0) or 0.0)
    shock_strength += 0.35 * abs((float(features.get("futures_term_structure_norm", 0.5) or 0.5) - 0.5) * 2.0)

    prof = _shadow_profile_name()
    if prof in {"dividend", "long_term_dividend"}:
        # Dividend sleeve: favor persistent trend + valuation mean reversion, downweight shock chasing.
        w_trend = 1.30 + 2.2 * trend_strength
        w_mean = 1.25 + 2.0 * max(0.0, 0.5 - chop_strength)
        w_shock = 0.70 + 1.2 * shock_strength
    elif prof == "bond":
        # Bond sleeve: emphasize regime persistence + carry mean reversion, but keep shock filter alive.
        w_trend = 1.20 + 1.9 * trend_strength
        w_mean = 1.30 + 2.2 * max(0.0, 0.5 - chop_strength)
        w_shock = 0.90 + 1.4 * shock_strength
    elif prof == "long_term_core_etf":
        # Core ETF sleeve: patient trend participation with lower shock-chasing weight.
        w_trend = 1.35 + 2.5 * trend_strength
        w_mean = 1.20 + 2.1 * max(0.0, 0.5 - chop_strength)
        w_shock = 0.75 + 1.0 * shock_strength
    elif prof == "long_term_sector_rotation":
        # Sector sleeve: still rotation-sensitive, but more selective than fast tactical profiles.
        w_trend = 1.25 + 2.4 * trend_strength
        w_mean = 1.10 + 1.8 * max(0.0, 0.5 - chop_strength)
        w_shock = 0.85 + 1.3 * shock_strength
    else:
        w_trend = 1.0 + 3.0 * trend_strength
        w_mean = 1.0 + 2.0 * max(0.0, 0.5 - chop_strength)
        w_shock = 1.0 + 4.0 * shock_strength

    total = w_trend + w_mean + w_shock
    if total <= 0.0:
        return {"trend": 1 / 3, "mean_revert": 1 / 3, "shock": 1 / 3}
    return {
        "trend": w_trend / total,
        "mean_revert": w_mean / total,
        "shock": w_shock / total,
    }


def _grand_master_vote(
    master_outputs: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
) -> tuple[str, float, float, List[str], Dict[str, float]]:
    if not master_outputs:
        return "HOLD", 0.5, _shift_threshold(0.55), ["no_master_outputs"], {"vote": 0.0}

    vote = 0.0
    details: List[str] = []
    for name, out in master_outputs.items():
        w = float(weights.get(name, 0.0))
        v = float(out.get("vote", 0.0))
        vote += w * v
        details.append(f"{name}:{w:.2f}")

    vote = _calibrate_vote(vote, scale=0.80)
    score = _vote_to_score(vote)
    threshold = _shift_threshold(0.60)

    if vote > 0.24:
        action = "BUY"
    elif vote < -0.20:
        action = "SELL"
    else:
        action = "HOLD"

    reasons = ["grand_master_routing", "master_weights=" + ",".join(details)]
    return action, score, threshold, reasons, {"vote": vote}


def _options_master_signal(
    *,
    grand_action: str,
    grand_score: float,
    grand_vote: float,
    features: Dict[str, float],
) -> tuple[str, float, float, List[str], Dict[str, float]]:
    vol = max(float(features.get("vol_30m", 0.0)), 0.0)
    vix_pct = float(features.get("ctx_VIX_X_pct_from_close", 0.0))
    dollar_mom = float(features.get("ctx_UUP_mom_5m", 0.0))
    options_vote = float(features.get("options_specialist_vote", 0.0) or 0.0)
    futures_vote = float(features.get("futures_specialist_vote", 0.0) or 0.0)

    super_mode = _derivatives_super_trader_mode()

    # Convert spot-level confidence into options suitability with extra regime risk penalties.
    net = float(grand_vote)
    net += 0.20 * (grand_score - 0.5)
    net += 0.16 * options_vote + 0.10 * futures_vote
    net -= 0.25 * max(vix_pct, 0.0)
    net -= 0.20 * vol
    net += -0.10 * dollar_mom

    if super_mode:
        net_scale = _super_trader_env_float("OPTIONS_SUPER_TRADER_MASTER_NET_SCALE", 1.28)
        threshold_shift = _super_trader_env_float("OPTIONS_SUPER_TRADER_MASTER_THRESHOLD_SHIFT", -0.040)
        buy_trigger = _super_trader_env_float("OPTIONS_SUPER_TRADER_MASTER_BUY_TRIGGER", 0.20)
        sell_trigger = _super_trader_env_float("OPTIONS_SUPER_TRADER_MASTER_SELL_TRIGGER", -0.18)
        net = _calibrate_vote(net * net_scale, scale=0.82)
        threshold = min(max(_shift_threshold(0.62) + threshold_shift, 0.45), 0.75)
    else:
        net = _calibrate_vote(net, scale=0.75)
        threshold = _shift_threshold(0.62)
        buy_trigger = 0.25
        sell_trigger = -0.22

    score = _vote_to_score(net)

    if net > buy_trigger:
        action = "BUY"
    elif net < sell_trigger:
        action = "SELL"
    else:
        action = "HOLD"

    reasons = [
        "options_master_regime_filter",
        f"from_grand={grand_action}",
        f"vix_pct={vix_pct:.4f}",
        f"vol_30m={vol:.4f}",
        f"options_vote={options_vote:.3f}",
        f"futures_vote={futures_vote:.3f}",
    ]
    if super_mode:
        reasons = reasons + [
            "options_super_trader_master",
            f"buy_trigger={buy_trigger:.3f}",
            f"sell_trigger={sell_trigger:.3f}",
        ]
    return action, score, threshold, reasons, {"vote": net}

def _build_options_plan(
    *,
    symbol: str,
    mkt: Dict[str, float],
    master_action: str,
    master_score: float,
    master_vote: float,
    covered_call_shares: int,
) -> Dict[str, Any]:
    px = max(float(mkt.get("last_price", 0.0) or 0.0), 1.0)
    vol = max(float(mkt.get("vol_30m", 0.0) or 0.0), 0.0)
    range_pos = float(mkt.get("range_pos", 0.5) or 0.5)

    iv_atm_norm = float(mkt.get("options_iv_atm_norm", 0.0) or 0.0)
    iv_skew_norm = float(mkt.get("options_iv_skew_norm", 0.5) or 0.5)
    iv_term_norm = float(mkt.get("options_iv_term_structure_norm", 0.5) or 0.5)
    put_call_norm = float(mkt.get("options_put_call_oi_ratio_norm", 0.5) or 0.5)
    neg_bias = float(mkt.get("options_negative_bias_norm", 0.0) or 0.0)
    roll_yield = float(mkt.get("options_roll_yield_norm", 0.0) or 0.0)
    vwap_bias = float(mkt.get("options_vwap_bias_norm", 0.0) or 0.0)
    vol_expect = float(mkt.get("options_vol_expectation_norm", 0.0) or 0.0)
    spread_norm = float(mkt.get("options_spread_bps_norm", 0.0) or 0.0)
    event_prox = float(mkt.get("calendar_event_proximity_norm", 0.0) or 0.0)
    high_impact = float(mkt.get("calendar_high_impact_24h_norm", 0.0) or 0.0)
    expiry_week = float(mkt.get("calendar_options_expiry_week_norm", 0.0) or 0.0)

    near_exp_days = max(float(mkt.get("options_near_expiry_days", 0.0) or 0.0), 7.0)
    far_exp_days = max(float(mkt.get("options_far_expiry_days", 0.0) or 0.0), near_exp_days + 7.0)
    front_dte = max(int(round(near_exp_days if near_exp_days > 0.0 else 21.0)), 7)
    back_dte = max(int(round(far_exp_days if far_exp_days > 0.0 else 45.0)), front_dte + 7)

    # Conservative default: no options position when confidence is mixed.
    action = "HOLD"
    score = master_score
    threshold = _shift_threshold(0.58)
    reasons = ["options_filter_no_clear_edge"]
    plan = {
        "symbol": symbol,
        "options_style": "NONE",
        "strategy_family": "NONE",
        "underlying_price": px,
        "dte_days": front_dte,
        "contracts": 0,
        "legs": [],
        "strike": None,
    }

    can_cover = covered_call_shares >= 100
    high_vol_regime = (vol_expect >= 0.62) or (iv_atm_norm >= 0.60) or (vol >= 0.020)
    bearish_bias = (put_call_norm >= 0.58) or (neg_bias >= 0.58)
    low_event_risk = event_prox <= 0.55
    liquid_chain = spread_norm <= 0.80
    event_vol_edge = (event_prox >= 0.65) and (high_impact >= 0.40) and (vol_expect >= 0.50) and liquid_chain

    contracts = 1
    if can_cover:
        contracts = max(covered_call_shares // 100, 1)

    wheel_enabled = os.getenv("OPTIONS_WHEEL_ENABLED", "1").strip() == "1"
    wheel_cash_budget = max(float(os.getenv("WHEEL_CASH_SECURED_PUT_CASH", "0") or 0.0), 0.0)
    wheel_max_contracts = max(int(os.getenv("WHEEL_MAX_CONTRACTS", "5") or 5), 1)
    wheel_put_contracts = 1
    if wheel_cash_budget > 0.0:
        wheel_put_contracts = max(min(int(wheel_cash_budget // max(px * 100.0, 1.0)), wheel_max_contracts), 1)

    def _leg(side: str, option_type: str, strike_mult: float, expiry_days: int, qty: int = 1) -> Dict[str, Any]:
        return {
            "side": side,
            "type": option_type,
            "strike": round(px * strike_mult, 2),
            "expiry_days": int(expiry_days),
            "quantity": int(max(qty, 1)),
        }

    # 1) Protective collar for covered inventory during elevated downside/event risk.
    if can_cover and liquid_chain and ((event_prox >= 0.68) or bearish_bias):
        action = "BUY_TO_OPEN"
        score = max(0.60, master_score)
        reasons = [
            "protective_collar_overlay",
            "has_covered_shares",
            f"event_prox={event_prox:.3f}",
            f"neg_bias={neg_bias:.3f}",
        ]
        plan = {
            "symbol": symbol,
            "options_style": "PROTECTIVE_COLLAR",
            "strategy_family": "hedge",
            "underlying_price": px,
            "dte_days": front_dte,
            "contracts": contracts,
            "legs": [
                _leg("BUY_TO_OPEN", "PUT", 0.96, front_dte, contracts),
                _leg("SELL_TO_OPEN", "CALL", 1.04, front_dte, contracts),
            ],
            "strike": None,
        }

    # 2) Income-first covered overlays when we have shares.
    elif can_cover and master_action in {"HOLD", "SELL"} and range_pos > 0.60 and (not high_vol_regime):
        action = "SELL_TO_OPEN"
        score = max(0.60, master_score)
        reasons = [
            "covered_call_income_setup",
            "has_covered_shares",
            f"iv_term_norm={iv_term_norm:.3f}",
            f"neg_bias={neg_bias:.3f}",
        ]
        if wheel_enabled:
            reasons = reasons + ["wheel_phase_covered_call"]
        plan = {
            "symbol": symbol,
            "options_style": "WHEEL_COVERED_CALL" if wheel_enabled else "COVERED_CALL",
            "strategy_family": "wheel" if wheel_enabled else "income",
            "underlying_price": px,
            "dte_days": front_dte,
            "contracts": contracts,
            "legs": [_leg("SELL_TO_OPEN", "CALL", 1.03, front_dte, contracts)],
            "strike": round(px * 1.03, 2),
            "wheel_phase": "covered_call" if wheel_enabled else "income_call",
        }

    # 3) Wheel entry via cash-secured puts.
    elif wheel_enabled and (not can_cover) and master_action in {"BUY", "HOLD"} and master_score >= (threshold - 0.04) and low_event_risk and liquid_chain and ((iv_atm_norm >= 0.44) or (vol_expect >= 0.46)) and (not bearish_bias):
        action = "SELL_TO_OPEN"
        score = max(master_score, 0.57)
        reasons = [
            "wheel_cash_secured_put_setup",
            "no_covered_shares",
            f"iv_atm_norm={iv_atm_norm:.3f}",
            f"vol_expect={vol_expect:.3f}",
        ]
        put_strike_mult = 0.96 if high_vol_regime else 0.97
        plan = {
            "symbol": symbol,
            "options_style": "WHEEL_CASH_SECURED_PUT",
            "strategy_family": "wheel",
            "underlying_price": px,
            "dte_days": max(front_dte, 14),
            "contracts": wheel_put_contracts,
            "legs": [_leg("SELL_TO_OPEN", "PUT", put_strike_mult, max(front_dte, 14), wheel_put_contracts)],
            "strike": round(px * put_strike_mult, 2),
            "wheel_phase": "accumulate_shares",
        }

    # 4) Event-vol strategy: long straddle/strangle only when event edge is high.
    elif event_vol_edge and master_action in {"BUY", "SELL", "HOLD"}:
        action = "BUY_TO_OPEN"
        score = max(master_score, 0.58)
        if abs(iv_skew_norm - 0.5) <= 0.08:
            reasons = [
                "event_vol_long_straddle",
                f"event_prox={event_prox:.3f}",
                f"vol_expect={vol_expect:.3f}",
            ]
            plan = {
                "symbol": symbol,
                "options_style": "EVENT_VOL_STRADDLE",
                "strategy_family": "event_vol",
                "underlying_price": px,
                "dte_days": max(front_dte, 7),
                "contracts": 1,
                "legs": [
                    _leg("BUY_TO_OPEN", "CALL", 1.00, max(front_dte, 7), 1),
                    _leg("BUY_TO_OPEN", "PUT", 1.00, max(front_dte, 7), 1),
                ],
                "strike": round(px, 2),
            }
        else:
            reasons = [
                "event_vol_long_strangle",
                f"event_prox={event_prox:.3f}",
                f"iv_skew_norm={iv_skew_norm:.3f}",
            ]
            plan = {
                "symbol": symbol,
                "options_style": "EVENT_VOL_STRANGLE",
                "strategy_family": "event_vol",
                "underlying_price": px,
                "dte_days": max(front_dte, 7),
                "contracts": 1,
                "legs": [
                    _leg("BUY_TO_OPEN", "CALL", 1.03, max(front_dte, 7), 1),
                    _leg("BUY_TO_OPEN", "PUT", 0.97, max(front_dte, 7), 1),
                ],
                "strike": None,
            }

    elif master_action == "BUY" and master_score >= threshold:
        # 5) Poor man's covered call (LEAPS diagonal).
        if (not can_cover) and low_event_risk and liquid_chain and (master_score >= threshold + 0.05) and (iv_term_norm >= 0.58):
            action = "BUY_TO_OPEN"
            score = max(master_score, 0.59)
            leap_dte = max(back_dte, 120)
            reasons = [
                "poor_mans_covered_call",
                f"term_structure={iv_term_norm:.3f}",
                f"master_score={master_score:.3f}",
            ]
            plan = {
                "symbol": symbol,
                "options_style": "POOR_MANS_COVERED_CALL",
                "strategy_family": "diagonal_income",
                "underlying_price": px,
                "dte_days": leap_dte,
                "contracts": 1,
                "legs": [
                    _leg("BUY_TO_OPEN", "CALL", 0.90, leap_dte, 1),
                    _leg("SELL_TO_OPEN", "CALL", 1.03, max(front_dte, 14), 1),
                ],
                "strike": round(px * 0.90, 2),
            }

        # 6) Risk reversal bullish.
        elif low_event_risk and liquid_chain and (master_score >= threshold + 0.05) and (put_call_norm <= 0.55):
            action = "BUY_TO_OPEN"
            score = max(master_score, 0.58)
            reasons = [
                "risk_reversal_bullish",
                f"put_call_norm={put_call_norm:.3f}",
                f"neg_bias={neg_bias:.3f}",
            ]
            plan = {
                "symbol": symbol,
                "options_style": "RISK_REVERSAL_BULLISH",
                "strategy_family": "synthetic_directional",
                "underlying_price": px,
                "dte_days": max(front_dte, 21),
                "contracts": 1,
                "legs": [
                    _leg("BUY_TO_OPEN", "CALL", 1.02, max(front_dte, 21), 1),
                    _leg("SELL_TO_OPEN", "PUT", 0.96, max(front_dte, 21), 1),
                ],
                "strike": round(px * 1.02, 2),
            }

        elif high_vol_regime and low_event_risk and liquid_chain and (not bearish_bias):
            action = "SELL_TO_OPEN"
            score = max(master_score, 0.58)
            reasons = [
                "bull_put_credit_spread",
                f"vol_expect={vol_expect:.3f}",
                f"roll_yield={roll_yield:.3f}",
            ]
            plan = {
                "symbol": symbol,
                "options_style": "BULL_PUT_CREDIT_SPREAD",
                "strategy_family": "credit_spread",
                "underlying_price": px,
                "dte_days": front_dte,
                "contracts": contracts,
                "legs": [
                    _leg("SELL_TO_OPEN", "PUT", 0.98, front_dte, contracts),
                    _leg("BUY_TO_OPEN", "PUT", 0.94, front_dte, contracts),
                ],
                "strike": round(px * 0.98, 2),
            }
        elif iv_term_norm > 0.60 and low_event_risk and liquid_chain:
            action = "BUY_TO_OPEN"
            score = max(master_score, 0.57)
            reasons = [
                "bull_call_calendar",
                f"term_structure={iv_term_norm:.3f}",
                f"vwap_bias={vwap_bias:.3f}",
            ]
            plan = {
                "symbol": symbol,
                "options_style": "CALL_CALENDAR_SPREAD",
                "strategy_family": "calendar",
                "underlying_price": px,
                "dte_days": front_dte,
                "contracts": 1,
                "legs": [
                    _leg("SELL_TO_OPEN", "CALL", 1.01, front_dte, 1),
                    _leg("BUY_TO_OPEN", "CALL", 1.01, back_dte, 1),
                ],
                "strike": round(px * 1.01, 2),
            }
        else:
            action = "BUY_TO_OPEN"
            score = master_score
            reasons = [
                "bull_call_debit_spread",
                f"iv_skew={iv_skew_norm:.3f}",
                f"vol_expect={vol_expect:.3f}",
            ]
            plan = {
                "symbol": symbol,
                "options_style": "BULL_CALL_DEBIT_SPREAD",
                "strategy_family": "debit_spread",
                "underlying_price": px,
                "dte_days": max(front_dte, 14),
                "contracts": 1,
                "legs": [
                    _leg("BUY_TO_OPEN", "CALL", 1.00, max(front_dte, 14), 1),
                    _leg("SELL_TO_OPEN", "CALL", 1.05, max(front_dte, 14), 1),
                ],
                "strike": round(px * 1.00, 2),
            }

    elif master_action == "SELL" and (1.0 - master_score) >= (threshold - 0.03):
        # 7) Bearish calendar when the back month is rich and directional pressure is already negative.
        if (iv_term_norm > 0.60) and low_event_risk and liquid_chain and bearish_bias:
            action = "BUY_TO_OPEN"
            score = max(1.0 - master_score, 0.56)
            reasons = [
                "put_calendar_spread",
                f"term_structure={iv_term_norm:.3f}",
                f"put_call_norm={put_call_norm:.3f}",
            ]
            plan = {
                "symbol": symbol,
                "options_style": "PUT_CALENDAR_SPREAD",
                "strategy_family": "calendar",
                "underlying_price": px,
                "dte_days": front_dte,
                "contracts": 1,
                "legs": [
                    _leg("SELL_TO_OPEN", "PUT", 0.99, front_dte, 1),
                    _leg("BUY_TO_OPEN", "PUT", 0.99, back_dte, 1),
                ],
                "strike": round(px * 0.99, 2),
            }

        # 8) Risk reversal bearish.
        elif low_event_risk and liquid_chain and ((1.0 - master_score) >= (threshold - 0.05)) and bearish_bias:
            action = "BUY_TO_OPEN"
            score = max(1.0 - master_score, 0.58)
            reasons = [
                "risk_reversal_bearish",
                f"put_call_norm={put_call_norm:.3f}",
                f"neg_bias={neg_bias:.3f}",
            ]
            plan = {
                "symbol": symbol,
                "options_style": "RISK_REVERSAL_BEARISH",
                "strategy_family": "synthetic_directional",
                "underlying_price": px,
                "dte_days": max(front_dte, 21),
                "contracts": 1,
                "legs": [
                    _leg("BUY_TO_OPEN", "PUT", 0.98, max(front_dte, 21), 1),
                    _leg("SELL_TO_OPEN", "CALL", 1.04, max(front_dte, 21), 1),
                ],
                "strike": round(px * 0.98, 2),
            }

        elif high_vol_regime and low_event_risk and liquid_chain:
            action = "SELL_TO_OPEN"
            score = max(1.0 - master_score, 0.58)
            reasons = [
                "bear_call_credit_spread",
                f"neg_bias={neg_bias:.3f}",
                f"term_structure={iv_term_norm:.3f}",
            ]
            plan = {
                "symbol": symbol,
                "options_style": "BEAR_CALL_CREDIT_SPREAD",
                "strategy_family": "credit_spread",
                "underlying_price": px,
                "dte_days": front_dte,
                "contracts": 1,
                "legs": [
                    _leg("SELL_TO_OPEN", "CALL", 1.03, front_dte, 1),
                    _leg("BUY_TO_OPEN", "CALL", 1.07, front_dte, 1),
                ],
                "strike": round(px * 1.03, 2),
            }
        elif bearish_bias:
            action = "BUY_TO_OPEN"
            score = max(1.0 - master_score, 0.55)
            reasons = [
                "long_put_setup",
                f"put_call_norm={put_call_norm:.3f}",
                f"neg_bias={neg_bias:.3f}",
            ]
            plan = {
                "symbol": symbol,
                "options_style": "LONG_PUT",
                "strategy_family": "directional",
                "underlying_price": px,
                "dte_days": max(front_dte, 21),
                "contracts": 1,
                "legs": [_leg("BUY_TO_OPEN", "PUT", 0.98, max(front_dte, 21), 1)],
                "strike": round(px * 0.98, 2),
            }
        else:
            action = "BUY_TO_OPEN"
            score = max(1.0 - master_score, 0.53)
            reasons = [
                "bear_put_debit_spread",
                f"iv_skew={iv_skew_norm:.3f}",
                f"roll_yield={roll_yield:.3f}",
            ]
            plan = {
                "symbol": symbol,
                "options_style": "BEAR_PUT_DEBIT_SPREAD",
                "strategy_family": "debit_spread",
                "underlying_price": px,
                "dte_days": max(front_dte, 14),
                "contracts": 1,
                "legs": [
                    _leg("BUY_TO_OPEN", "PUT", 1.00, max(front_dte, 14), 1),
                    _leg("SELL_TO_OPEN", "PUT", 0.95, max(front_dte, 14), 1),
                ],
                "strike": round(px * 1.00, 2),
            }

    else:
        # 8) Broken wing butterfly for neutral mean-reverting high-vol conditions.
        if high_vol_regime and low_event_risk and liquid_chain and abs(vwap_bias - 0.5) >= 0.12:
            direction = "PUT" if bearish_bias else "CALL"
            action = "BUY_TO_OPEN"
            score = max(master_score, 0.56)
            reasons = [
                "broken_wing_butterfly",
                f"vwap_bias={vwap_bias:.3f}",
                f"direction={direction}",
            ]
            if direction == "CALL":
                legs = [
                    _leg("BUY_TO_OPEN", "CALL", 0.99, front_dte, 1),
                    _leg("SELL_TO_OPEN", "CALL", 1.03, front_dte, 2),
                    _leg("BUY_TO_OPEN", "CALL", 1.10, front_dte, 1),
                ]
                strike = round(px * 1.03, 2)
            else:
                legs = [
                    _leg("BUY_TO_OPEN", "PUT", 1.01, front_dte, 1),
                    _leg("SELL_TO_OPEN", "PUT", 0.97, front_dte, 2),
                    _leg("BUY_TO_OPEN", "PUT", 0.90, front_dte, 1),
                ]
                strike = round(px * 0.97, 2)
            plan = {
                "symbol": symbol,
                "options_style": "BROKEN_WING_BUTTERFLY",
                "strategy_family": "neutral_income",
                "underlying_price": px,
                "dte_days": front_dte,
                "contracts": 1,
                "legs": legs,
                "strike": strike,
            }

        elif high_vol_regime and low_event_risk and liquid_chain and expiry_week < 0.8 and abs(vwap_bias - 0.5) <= 0.08 and abs(iv_skew_norm - 0.5) <= 0.10:
            action = "SELL_TO_OPEN"
            score = max(master_score, 0.57)
            reasons = [
                "iron_butterfly_income",
                f"vol_expect={vol_expect:.3f}",
                f"iv_skew={iv_skew_norm:.3f}",
            ]
            plan = {
                "symbol": symbol,
                "options_style": "IRON_BUTTERFLY",
                "strategy_family": "neutral_income",
                "underlying_price": px,
                "dte_days": front_dte,
                "contracts": 1,
                "legs": [
                    _leg("SELL_TO_OPEN", "PUT", 1.00, front_dte, 1),
                    _leg("BUY_TO_OPEN", "PUT", 0.94, front_dte, 1),
                    _leg("SELL_TO_OPEN", "CALL", 1.00, front_dte, 1),
                    _leg("BUY_TO_OPEN", "CALL", 1.06, front_dte, 1),
                ],
                "strike": round(px, 2),
            }

        elif high_vol_regime and low_event_risk and liquid_chain and expiry_week < 0.8:
            action = "SELL_TO_OPEN"
            score = max(master_score, 0.56)
            reasons = [
                "iron_condor_income",
                f"vol_expect={vol_expect:.3f}",
                f"expiry_week={expiry_week:.3f}",
            ]
            plan = {
                "symbol": symbol,
                "options_style": "IRON_CONDOR",
                "strategy_family": "neutral_income",
                "underlying_price": px,
                "dte_days": front_dte,
                "contracts": 1,
                "legs": [
                    _leg("SELL_TO_OPEN", "PUT", 0.97, front_dte, 1),
                    _leg("BUY_TO_OPEN", "PUT", 0.93, front_dte, 1),
                    _leg("SELL_TO_OPEN", "CALL", 1.03, front_dte, 1),
                    _leg("BUY_TO_OPEN", "CALL", 1.07, front_dte, 1),
                ],
                "strike": None,
            }
        elif (abs(iv_term_norm - 0.5) >= 0.12) and low_event_risk and liquid_chain:
            direction = "PUT" if bearish_bias else "CALL"
            action = "BUY_TO_OPEN"
            score = max(master_score, 0.54)
            reasons = [
                "diagonal_calendar",
                f"term_structure={iv_term_norm:.3f}",
                f"negative_bias={neg_bias:.3f}",
            ]
            plan = {
                "symbol": symbol,
                "options_style": "DIAGONAL_CALENDAR_SPREAD",
                "strategy_family": "calendar",
                "underlying_price": px,
                "dte_days": front_dte,
                "contracts": 1,
                "legs": [
                    _leg("SELL_TO_OPEN", direction, 1.00 if direction == "CALL" else 0.99, front_dte, 1),
                    _leg("BUY_TO_OPEN", direction, 1.02 if direction == "CALL" else 0.97, back_dte, 1),
                ],
                "strike": round(px * (1.00 if direction == "CALL" else 0.99), 2),
            }

    plan["master_vote"] = master_vote
    plan["spread_width_pct"] = 0.04
    plan["wheel_enabled"] = wheel_enabled
    if "wheel_phase" not in plan and str(plan.get("strategy_family", "")).lower() == "wheel":
        plan["wheel_phase"] = "covered_call" if can_cover else "cash_secured_put"
    plan["context"] = {
        "options_iv_atm_norm": iv_atm_norm,
        "options_iv_skew_norm": iv_skew_norm,
        "options_iv_term_structure_norm": iv_term_norm,
        "options_put_call_oi_ratio_norm": put_call_norm,
        "options_negative_bias_norm": neg_bias,
        "options_roll_yield_norm": roll_yield,
        "options_vwap_bias_norm": vwap_bias,
        "options_vol_expectation_norm": vol_expect,
        "calendar_event_proximity_norm": event_prox,
        "calendar_options_expiry_week_norm": expiry_week,
    }

    return {
        "action": action,
        "score": min(max(score, 0.01), 0.99),
        "threshold": threshold,
        "reasons": reasons,
        "plan": plan,
    }


def _options_roll_manager(
    *,
    decision: Dict[str, Any],
    features: Dict[str, float],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    out = {
        "action": str(decision.get("action", "HOLD")),
        "score": float(decision.get("score", 0.5) or 0.5),
        "threshold": float(decision.get("threshold", 0.58) or 0.58),
        "reasons": list(decision.get("reasons", []) or []),
        "plan": dict(decision.get("plan", {}) or {}),
    }

    plan = out["plan"]
    style = str(plan.get("options_style", "NONE") or "NONE").upper()
    if style == "NONE":
        return out, {"active": False, "directive": "none"}

    dte = int(float(plan.get("dte_days", 0) or 0))
    roll_dte = max(int(os.getenv("OPTIONS_ROLL_DTE_DAYS", "7") or 7), 1)
    assign_band = max(float(os.getenv("OPTIONS_ASSIGNMENT_RISK_BAND", "0.01") or 0.01), 0.001)
    iv_crush_floor = _clamp01(float(os.getenv("OPTIONS_IV_CRUSH_VOL_EXPECT_MAX", "0.30") or 0.30))

    px = float(plan.get("underlying_price", features.get("last_price", 0.0)) or 0.0)
    event_prox = _clamp01(float(features.get("calendar_event_proximity_norm", 0.0) or 0.0))
    vol_expect = _clamp01(float(features.get("options_vol_expectation_norm", 0.0) or 0.0))

    assignment_risk = False
    for leg in (plan.get("legs", []) or []):
        if not isinstance(leg, dict):
            continue
        side = str(leg.get("side", "") or "").upper()
        opt_type = str(leg.get("option_type", "") or "").upper()
        strike = float(leg.get("strike", 0.0) or 0.0)
        if "SELL" not in side or strike <= 0.0 or px <= 0.0:
            continue
        if opt_type == "CALL" and strike <= px * (1.0 + assign_band):
            assignment_risk = True
            break
        if opt_type == "PUT" and strike >= px * (1.0 - assign_band):
            assignment_risk = True
            break

    directive = "none"
    if style in {"EVENT_VOL_STRADDLE", "EVENT_VOL_STRANGLE"} and event_prox <= 0.30 and vol_expect <= iv_crush_floor:
        directive = "close_post_event_iv_crush"
        out["action"] = "CLOSE"
    elif dte <= roll_dte or assignment_risk:
        directive = "roll_forward"
        out["action"] = "ROLL"

    if directive != "none":
        plan["lifecycle_directive"] = directive
        plan["roll_target_dte_days"] = max(dte + max(int(os.getenv("OPTIONS_ROLL_FORWARD_DAYS", "21") or 21), 7), 14)
        out["reasons"] = out["reasons"] + [
            f"options_roll_manager directive={directive}",
            f"dte={dte}",
            f"assignment_risk={int(assignment_risk)}",
            f"vol_expect={vol_expect:.3f}",
        ]

    return out, {
        "active": True,
        "directive": directive,
        "style": style,
        "dte_days": dte,
        "assignment_risk": bool(assignment_risk),
        "event_proximity_norm": event_prox,
        "vol_expectation_norm": vol_expect,
    }


def _futures_master_signal(
    *,
    grand_action: str,
    grand_score: float,
    grand_vote: float,
    features: Dict[str, float],
) -> tuple[str, float, float, List[str], Dict[str, float]]:
    vol = max(float(features.get("vol_30m", 0.0) or 0.0), 0.0)
    event_prox = float(features.get("calendar_event_proximity_norm", 0.0) or 0.0)
    high_impact = float(features.get("calendar_high_impact_24h_norm", 0.0) or 0.0)

    futures_vote = float(features.get("futures_specialist_vote", 0.0) or 0.0)
    options_vote = float(features.get("options_specialist_vote", 0.0) or 0.0)

    imbalance = float(features.get("futures_order_book_imbalance_norm", 0.5) or 0.5) - 0.5
    basis = float(features.get("futures_basis_bps_norm", 0.5) or 0.5) - 0.5
    term = float(features.get("futures_term_structure_norm", 0.5) or 0.5) - 0.5
    roll = float(features.get("futures_roll_yield_norm", 0.5) or 0.5) - 0.5
    neg_bias = float(features.get("futures_negative_bias_norm", 0.5) or 0.5) - 0.5

    super_mode = _derivatives_super_trader_mode()

    net = float(grand_vote)
    net += 0.22 * (grand_score - 0.5)
    net += 0.20 * futures_vote + 0.08 * options_vote
    net += 0.24 * imbalance + 0.16 * basis + 0.14 * term + 0.10 * roll
    net -= 0.15 * neg_bias
    net -= 0.20 * event_prox + 0.18 * high_impact + 0.16 * vol

    if super_mode:
        net_scale = _super_trader_env_float("FUTURES_SUPER_TRADER_MASTER_NET_SCALE", 1.30)
        threshold_shift = _super_trader_env_float("FUTURES_SUPER_TRADER_MASTER_THRESHOLD_SHIFT", -0.035)
        buy_trigger = _super_trader_env_float("FUTURES_SUPER_TRADER_MASTER_BUY_TRIGGER", 0.21)
        sell_trigger = _super_trader_env_float("FUTURES_SUPER_TRADER_MASTER_SELL_TRIGGER", -0.19)
        net = _calibrate_vote(net * net_scale, scale=0.84)
        threshold = min(max(_shift_threshold(0.61) + threshold_shift, 0.45), 0.75)
    else:
        net = _calibrate_vote(net, scale=0.80)
        threshold = _shift_threshold(0.61)
        buy_trigger = 0.24
        sell_trigger = -0.22

    score = _vote_to_score(net)

    if net > buy_trigger:
        action = "BUY"
    elif net < sell_trigger:
        action = "SELL"
    else:
        action = "HOLD"

    reasons = [
        "futures_master_regime_filter",
        f"from_grand={grand_action}",
        f"event_prox={event_prox:.3f}",
        f"high_impact={high_impact:.3f}",
        f"futures_vote={futures_vote:.3f}",
        f"imbalance={imbalance:+.3f}",
        f"basis={basis:+.3f}",
    ]
    if super_mode:
        reasons = reasons + [
            "futures_super_trader_master",
            f"buy_trigger={buy_trigger:.3f}",
            f"sell_trigger={sell_trigger:.3f}",
        ]

    return action, score, threshold, reasons, {"vote": net}


def _build_futures_plan(
    *,
    symbol: str,
    mkt: Dict[str, float],
    master_action: str,
    master_score: float,
    master_vote: float,
    symbol_is_futures: bool,
) -> Dict[str, Any]:
    threshold = _shift_threshold(0.58)

    plan = {
        "symbol": symbol,
        "futures_style": "NONE",
        "strategy_family": "NONE",
        "contracts": 0,
        "legs": [],
        "front_month": "M1",
    }

    action = "HOLD"
    score = float(master_score)
    reasons = ["futures_filter_no_clear_edge"]

    if not symbol_is_futures:
        reasons = ["non_futures_symbol"]
        plan["master_vote"] = float(master_vote)
        return {
            "action": action,
            "score": min(max(score, 0.01), 0.99),
            "threshold": threshold,
            "reasons": reasons,
            "plan": plan,
        }

    vol = max(float(mkt.get("vol_30m", 0.0) or 0.0), 0.0)
    range_pos = float(mkt.get("range_pos", 0.5) or 0.5)
    mom_5m = float(mkt.get("mom_5m", 0.0) or 0.0)

    spread_norm = float(mkt.get("futures_spread_bps_norm", 0.0) or 0.0)
    depth_norm = float(mkt.get("futures_depth_ratio_norm", 0.0) or 0.0)

    imbalance = float(mkt.get("futures_order_book_imbalance_norm", 0.5) or 0.5) - 0.5
    basis = float(mkt.get("futures_basis_bps_norm", 0.5) or 0.5) - 0.5
    term = float(mkt.get("futures_term_structure_norm", 0.5) or 0.5) - 0.5
    roll = float(mkt.get("futures_roll_yield_norm", 0.5) or 0.5) - 0.5
    vwap = float(mkt.get("futures_vwap_bias_norm", 0.5) or 0.5) - 0.5
    neg_bias = float(mkt.get("futures_negative_bias_norm", 0.5) or 0.5) - 0.5
    funding = float(mkt.get("futures_funding_rate_norm", 0.5) or 0.5) - 0.5

    event_prox = float(mkt.get("calendar_event_proximity_norm", 0.0) or 0.0)
    high_impact = float(mkt.get("calendar_high_impact_24h_norm", 0.0) or 0.0)

    contracts = max(int(os.getenv("FUTURES_DEFAULT_CONTRACTS", "1") or 1), 1)
    max_contracts = max(int(os.getenv("FUTURES_MAX_CONTRACTS", "4") or 4), contracts)
    if abs(master_vote) >= 0.35 and abs(imbalance) >= 0.18:
        contracts = min(max_contracts, contracts + 1)

    low_event_risk = event_prox <= 0.55 and high_impact <= 0.55
    liquid = spread_norm <= 0.85 and depth_norm >= 0.10
    high_conviction = abs(master_vote) >= 0.22 or abs(master_score - 0.5) >= 0.12

    def _fleg(side: str, month: str, qty: int, offset: int) -> Dict[str, Any]:
        return {
            "side": side,
            "contract": month,
            "quantity": int(max(qty, 1)),
            "month_offset": int(offset),
        }

    # 1) Event-risk flatten and conditional re-entry.
    if (event_prox >= 0.70) or (high_impact >= 0.70):
        action = "HOLD"
        score = 0.5 + 0.25 * (master_score - 0.5)
        reasons = [
            "futures_event_risk_flatten_reenter",
            f"event_prox={event_prox:.3f}",
            f"high_impact={high_impact:.3f}",
        ]
        plan = {
            "symbol": symbol,
            "futures_style": "FUTURES_EVENT_RISK_FLATTEN_REENTER",
            "strategy_family": "event_risk",
            "contracts": 0,
            "legs": [],
            "front_month": "M1",
            "reentry_preference": "trend" if abs(master_vote) >= 0.20 else "mean_revert",
        }

    # 2) Calendar/basis carry spread.
    elif low_event_risk and liquid and (abs(basis) >= 0.18 or abs(term) >= 0.18) and abs(roll) >= 0.08:
        carry_long = ((basis >= 0.0) and (term >= -0.05)) or (roll > 0.12)
        action = "BUY" if carry_long else "SELL"
        score = max(master_score, 0.58)
        reasons = [
            "futures_basis_carry_calendar",
            f"basis={basis:+.3f}",
            f"term={term:+.3f}",
            f"roll={roll:+.3f}",
        ]
        if carry_long:
            legs = [_fleg("BUY", "M1", min(contracts, 2), 0), _fleg("SELL", "M2", min(contracts, 2), 1)]
        else:
            legs = [_fleg("SELL", "M1", min(contracts, 2), 0), _fleg("BUY", "M2", min(contracts, 2), 1)]
        plan = {
            "symbol": symbol,
            "futures_style": "FUTURES_BASIS_CARRY_CALENDAR",
            "strategy_family": "calendar_spread",
            "contracts": min(contracts, 2),
            "legs": legs,
            "front_month": "M1",
        }

    # 3) Funding dislocation mean reversion.
    elif low_event_risk and liquid and abs(funding) >= 0.15 and abs(imbalance) >= 0.10 and abs(basis) <= 0.18:
        action = "SELL" if funding > 0.0 else "BUY"
        score = max(master_score, 0.57)
        reasons = [
            "futures_funding_mean_revert",
            f"funding={funding:+.3f}",
            f"imbalance={imbalance:+.3f}",
            f"basis={basis:+.3f}",
        ]
        plan = {
            "symbol": symbol,
            "futures_style": "FUTURES_FUNDING_MEAN_REVERT",
            "strategy_family": "mean_revert",
            "contracts": contracts,
            "legs": [_fleg("BUY" if action == "BUY" else "SELL", "M1", contracts, 0)],
            "front_month": "M1",
        }

    # 4) Order-book imbalance breakout.
    elif low_event_risk and liquid and (master_action in {"BUY", "SELL"}) and high_conviction and abs(imbalance) >= 0.22 and ((master_action == "BUY" and mom_5m >= 0.0 and range_pos >= 0.52 and vwap >= -0.08) or (master_action == "SELL" and mom_5m <= 0.0 and range_pos <= 0.48 and vwap <= 0.08)):
        action = master_action
        score = max(master_score, 0.58)
        reasons = [
            "futures_orderbook_imbalance_breakout",
            f"imbalance={imbalance:+.3f}",
            f"mom_5m={mom_5m:+.4f}",
            f"range_pos={range_pos:.3f}",
        ]
        plan = {
            "symbol": symbol,
            "futures_style": "FUTURES_ORDERBOOK_IMBALANCE_BREAKOUT",
            "strategy_family": "directional",
            "contracts": contracts,
            "legs": [_fleg("BUY" if action == "BUY" else "SELL", "M1", contracts, 0)],
            "front_month": "M1",
        }

    # 5) Trend breakout / pullback continuation.
    elif low_event_risk and liquid and (master_action in {"BUY", "SELL"}) and high_conviction and ((master_action == "BUY" and mom_5m >= 0.0 and range_pos >= 0.56) or (master_action == "SELL" and mom_5m <= 0.0 and range_pos <= 0.44)):
        action = master_action
        score = max(master_score, 0.57)
        reasons = [
            "futures_trend_breakout_continuation",
            f"mom_5m={mom_5m:+.4f}",
            f"range_pos={range_pos:.3f}",
            f"vote={master_vote:+.3f}",
        ]
        plan = {
            "symbol": symbol,
            "futures_style": "FUTURES_TREND_BREAKOUT_CONTINUATION",
            "strategy_family": "directional",
            "contracts": contracts,
            "legs": [_fleg("BUY" if action == "BUY" else "SELL", "M1", contracts, 0)],
            "front_month": "M1",
        }

    # 6) VWAP/imbalance mean reversion.
    elif low_event_risk and liquid and (abs(vwap) >= 0.12) and (abs(imbalance) >= 0.10):
        action = "SELL" if vwap > 0 else "BUY"
        score = max(master_score, 0.55)
        reasons = [
            "futures_vwap_imbalance_mean_revert",
            f"vwap={vwap:+.3f}",
            f"imbalance={imbalance:+.3f}",
        ]
        plan = {
            "symbol": symbol,
            "futures_style": "FUTURES_VWAP_IMBALANCE_MEAN_REVERT",
            "strategy_family": "mean_revert",
            "contracts": contracts,
            "legs": [_fleg("BUY" if action == "BUY" else "SELL", "M1", contracts, 0)],
            "front_month": "M1",
        }

    # 7) Term-structure roll rotation.
    elif low_event_risk and liquid and (abs(term) >= 0.10) and (abs(roll) >= 0.12):
        direction_score = term + roll - 0.35 * neg_bias - 0.15 * funding
        action = "BUY" if direction_score >= 0.0 else "SELL"
        score = max(master_score, 0.56)
        reasons = [
            "futures_term_structure_roll_rotation",
            f"term={term:+.3f}",
            f"roll={roll:+.3f}",
            f"neg_bias={neg_bias:+.3f}",
        ]
        if action == "BUY":
            legs = [_fleg("BUY", "M2", min(contracts, 2), 1), _fleg("SELL", "M1", min(contracts, 2), 0)]
        else:
            legs = [_fleg("SELL", "M2", min(contracts, 2), 1), _fleg("BUY", "M1", min(contracts, 2), 0)]
        plan = {
            "symbol": symbol,
            "futures_style": "FUTURES_TERM_STRUCTURE_ROLL_ROTATION",
            "strategy_family": "roll_rotation",
            "contracts": min(contracts, 2),
            "legs": legs,
            "front_month": "M1",
        }

    plan["master_vote"] = float(master_vote)
    plan["context"] = {
        "futures_spread_bps_norm": spread_norm,
        "futures_depth_ratio_norm": depth_norm,
        "futures_order_book_imbalance_norm": imbalance + 0.5,
        "futures_basis_bps_norm": basis + 0.5,
        "futures_term_structure_norm": term + 0.5,
        "futures_roll_yield_norm": roll + 0.5,
        "futures_vwap_bias_norm": vwap + 0.5,
        "futures_negative_bias_norm": neg_bias + 0.5,
        "calendar_event_proximity_norm": event_prox,
        "calendar_high_impact_24h_norm": high_impact,
        "vol_30m": vol,
    }

    return {
        "action": action,
        "score": min(max(float(score), 0.01), 0.99),
        "threshold": threshold,
        "reasons": reasons,
        "plan": plan,
    }



def _hash01(text: str) -> float:
    if not text:
        return 0.0
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return (h / 0xFFFFFFFF)


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _latest_trade_behavior_model(project_root: str) -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(project_root, "models", "trade_behavior_policy_*.npz")))
    if not paths:
        return None
    return paths[-1]


def _latest_trade_behavior_model_quantized(project_root: str) -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(project_root, "models", "trade_behavior_policy_*_quantized.npz")))
    if not paths:
        return None
    return paths[-1]


def _load_trade_behavior_model(project_root: str) -> Optional[Dict[str, Any]]:
    if np is None:
        return None
    prefer_quantized = os.getenv("TRADE_BEHAVIOR_PREFER_QUANTIZED", "1").strip() == "1"
    path = _latest_trade_behavior_model_quantized(project_root) if prefer_quantized else None
    if not path:
        path = _latest_trade_behavior_model(project_root)
    if not path:
        return None

    min_neutral_f1 = float(os.getenv("TRADE_BEHAVIOR_MIN_NEUTRAL_F1", "0.25"))
    strict_gate = os.getenv("TRADE_BEHAVIOR_STRICT_NEUTRAL_GATE", "1").strip() == "1"

    try:
        arr = np.load(path, allow_pickle=False)
        dtype = np.float16 if os.getenv("TRADE_BEHAVIOR_USE_FP16", "1").strip() == "1" else float
        temperature = 1.0
        if "temperature" in arr.files:
            try:
                temperature = float(np.asarray(arr["temperature"]).reshape(-1)[0])
            except Exception:
                temperature = 1.0
        if (not math.isfinite(temperature)) or temperature <= 0.0:
            temperature = 1.0

        class_logit_bias = np.zeros((3,), dtype=dtype)
        if "class_logit_bias" in arr.files:
            try:
                raw_bias = np.asarray(arr["class_logit_bias"]).reshape(-1)
                if raw_bias.shape[0] == 3:
                    class_logit_bias = raw_bias.astype(dtype)
            except Exception:
                class_logit_bias = np.zeros((3,), dtype=dtype)

        model = {
            "path": path,
            "W": arr["W"].astype(dtype),
            "b": arr["b"].astype(dtype),
            "mu": arr["mu"].astype(dtype),
            "sigma": arr["sigma"].astype(dtype),
            "temperature": float(temperature),
            "class_logit_bias": class_logit_bias,
        }
    except Exception:
        return None

    # Gate behavior bias on neutral-class quality to avoid overconfident directional skew.
    try:
        stamp = os.path.basename(path).replace("trade_behavior_policy_", "").replace(".npz", "")
        log_path = os.path.join(project_root, "logs", f"trade_behavior_policy_{stamp}.json")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            te = obj.get("test_metrics", {}) if isinstance(obj, dict) else {}
            neutral_f1 = float(te.get("neutral_f1", 0.0) or 0.0)
            model["neutral_f1"] = neutral_f1
            if strict_gate and neutral_f1 < min_neutral_f1:
                print(
                    f"Trade behavior model gated off: neutral_f1={neutral_f1:.3f} "
                    f"< min_neutral_f1={min_neutral_f1:.3f}"
                )
                return None
    except Exception:
        pass

    return model


def _futures_roll_manager(
    *,
    decision: Dict[str, Any],
    features: Dict[str, float],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    out = {
        "action": str(decision.get("action", "HOLD")),
        "score": float(decision.get("score", 0.5) or 0.5),
        "threshold": float(decision.get("threshold", 0.58) or 0.58),
        "reasons": list(decision.get("reasons", []) or []),
        "plan": dict(decision.get("plan", {}) or {}),
    }

    plan = out["plan"]
    style = str(plan.get("futures_style", "NONE") or "NONE").upper()
    if style == "NONE":
        return out, {"active": False, "directive": "none"}

    front_month = str(plan.get("front_month", "M1") or "M1").upper()
    expiry_week = _clamp01(float(features.get("calendar_options_expiry_week_norm", 0.0) or 0.0))
    term = float(features.get("futures_term_structure_norm", 0.5) or 0.5) - 0.5
    roll = float(features.get("futures_roll_yield_norm", 0.5) or 0.5) - 0.5
    event_prox = _clamp01(float(features.get("calendar_event_proximity_norm", 0.0) or 0.0))

    roll_window_min = _clamp01(float(os.getenv("FUTURES_ROLL_WINDOW_MIN", "0.72") or 0.72))
    term_roll_min = max(float(os.getenv("FUTURES_TERM_ROLL_MIN", "0.14") or 0.14), 0.02)

    directive = "none"
    if (front_month == "M1") and (expiry_week >= roll_window_min):
        directive = "roll_near_expiry"
    elif (front_month == "M1") and (abs(term) >= term_roll_min or abs(roll) >= term_roll_min):
        directive = "roll_term_structure"

    if directive != "none":
        out["action"] = "ROLL"
        plan["lifecycle_directive"] = directive
        plan["roll_from_month"] = front_month
        plan["front_month"] = "M2"
        contracts = max(int(plan.get("contracts", 1) or 1), 1)
        plan["roll_legs"] = [
            {"side": "SELL", "contract": front_month, "quantity": contracts},
            {"side": "BUY", "contract": "M2", "quantity": contracts},
        ]
        if event_prox >= 0.75:
            out["action"] = "HOLD"
            plan["lifecycle_directive"] = "defer_roll_event_risk"
            directive = "defer_roll_event_risk"
        out["reasons"] = out["reasons"] + [
            f"futures_roll_manager directive={directive}",
            f"expiry_week={expiry_week:.3f}",
            f"term={term:+.3f}",
            f"roll={roll:+.3f}",
        ]

    return out, {
        "active": True,
        "directive": directive,
        "style": style,
        "front_month": front_month,
        "expiry_week_norm": expiry_week,
        "term_signal": term,
        "roll_signal": roll,
        "event_proximity_norm": event_prox,
    }


_BEHAVIOR_SHOCK_SYMBOLS = {"UVXY", "VIXY", "SOXL", "SOXS", "MSTR", "SMCI", "COIN", "TSLA"}
_BEHAVIOR_MEAN_REVERT_SYMBOLS = {"TLT", "IEF", "SHY", "BND", "AGG", "GLD", "XLU", "XLP"}
_BEHAVIOR_SNAPSHOT_FEATURE_NAMES_V1 = [
    "snapshot_cov_ok",
    "snapshot_cov_log_ratio",
    "snapshot_cov_fill_ratio",
    "snapshot_replay_ok",
    "snapshot_replay_stale_ratio",
    "snapshot_replay_drift_ratio",
    "snapshot_divergence_ratio",
    "snapshot_triprate_ratio",
    "snapshot_queue_pressure_ratio",
]
_BEHAVIOR_SNAPSHOT_FEATURE_NAMES_V2 = [
    "snapshot_cov_ok",
    "snapshot_cov_log_ratio",
    "snapshot_replay_stale_ratio",
    "snapshot_replay_drift_ratio",
    "snapshot_divergence_ratio",
    "snapshot_triprate_ratio",
    "snapshot_queue_pressure_ratio",
    "snapshot_cov_fill_ratio",
    "snapshot_replay_ok",
    "canary_weight_cap_norm",
]
_BEHAVIOR_FEATURE_NAMES_V2 = [
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
    "dividend_strategy_mode_capture",
    "dividend_strategy_mode_compound",
    "dividend_strategy_mode_hybrid",
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
]

_BEHAVIOR_LANE_FEATURE_NAMES = [
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
    "swing_post_earnings_drift_norm",
    "swing_gap_continuation_norm",
    "swing_gap_fade_norm",
    "swing_vol_compression_breakout_norm",
    "swing_sector_relative_strength_norm",
    "swing_weekly_trend_confirm_norm",
    "swing_regime_trend_norm",
    "swing_regime_chop_norm",
    "swing_regime_alignment_norm",
    "bond_duration_regime_norm",
    "bond_curve_steepener_norm",
    "bond_curve_flattener_norm",
    "bond_carry_roll_norm",
    "bond_credit_risk_on_norm",
    "bond_credit_risk_off_norm",
    "bond_inflation_breakeven_norm",
]

_BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES = list(_CAPITAL_FLOW_FEATURE_KEYS)

_BEHAVIOR_FEATURE_NAMES_V2.extend(_BEHAVIOR_LANE_FEATURE_NAMES)
_BEHAVIOR_FEATURE_NAMES_V2.extend(_BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES)
_BEHAVIOR_SNAPSHOT_CONTEXT_CACHE: Dict[str, Any] = {"loaded_at_ts": 0.0, "values": {}}


def _behavior_clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _behavior_signed_scale(value: float, gain: float) -> float:
    return math.tanh(float(value) * float(gain))


def _behavior_load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _behavior_snapshot_context(project_root: str) -> Dict[str, float]:
    now_ts = time.time()
    ttl_seconds = max(float(os.getenv('BEHAVIOR_SNAPSHOT_CONTEXT_TTL_SECONDS', '30')), 2.0)

    loaded_at_ts = float(_BEHAVIOR_SNAPSHOT_CONTEXT_CACHE.get('loaded_at_ts', 0.0) or 0.0)
    cached_values = _BEHAVIOR_SNAPSHOT_CONTEXT_CACHE.get('values')
    if (now_ts - loaded_at_ts) <= ttl_seconds and isinstance(cached_values, dict) and cached_values:
        return dict(cached_values)

    try:
        try:
            from scripts.build_behavior_dataset_from_decisions import _external_feeds_context, _snapshot_health_context
        except Exception:
            from build_behavior_dataset_from_decisions import _external_feeds_context, _snapshot_health_context

        snapshot_context, _ = _snapshot_health_context(Path(project_root))
        external_context, _ = _external_feeds_context(Path(project_root), datetime.now(timezone.utc))

        if isinstance(snapshot_context, dict) and snapshot_context:
            values: Dict[str, float] = {}
            for key in _BEHAVIOR_FEATURE_NAMES_V2:
                if key.startswith('snapshot_'):
                    raw = snapshot_context.get(key, 0.0)
                    try:
                        values[key] = _behavior_clamp01(float(raw or 0.0))
                    except Exception:
                        values[key] = 0.0
                elif key.startswith('external_'):
                    raw = external_context.get(key, 0.0) if isinstance(external_context, dict) else 0.0
                    try:
                        values[key] = _behavior_clamp01(float(raw or 0.0))
                    except Exception:
                        values[key] = 0.0

            values.setdefault('snapshot_cov_ok', 1.0 if bool(snapshot_context.get('snapshot_cov_ok', 0.0)) else 0.0)
            values.setdefault('snapshot_cov_log_ratio', _behavior_clamp01(float(snapshot_context.get('snapshot_cov_log_ratio', 0.0) or 0.0)))
            values.setdefault('snapshot_cov_fill_ratio', _behavior_clamp01(float(snapshot_context.get('snapshot_cov_fill_ratio', 0.0) or 0.0)))
            values.setdefault('snapshot_replay_ok', _behavior_clamp01(float(snapshot_context.get('snapshot_replay_ok', 0.0) or 0.0)))
            values.setdefault('snapshot_replay_stale_ratio', _behavior_clamp01(float(snapshot_context.get('snapshot_replay_stale_ratio', 0.0) or 0.0)))
            values.setdefault('snapshot_replay_drift_ratio', _behavior_clamp01(float(snapshot_context.get('snapshot_replay_drift_ratio', 0.0) or 0.0)))
            values.setdefault('snapshot_divergence_ratio', _behavior_clamp01(float(snapshot_context.get('snapshot_divergence_ratio', 0.0) or 0.0)))
            values.setdefault('snapshot_triprate_ratio', _behavior_clamp01(float(snapshot_context.get('snapshot_triprate_ratio', 0.0) or 0.0)))
            values.setdefault('snapshot_queue_pressure_ratio', _behavior_clamp01(float(snapshot_context.get('snapshot_queue_pressure_ratio', 0.0) or 0.0)))
            values.setdefault('canary_weight_cap_norm', _behavior_clamp01(float(snapshot_context.get('canary_weight_cap_norm', _behavior_clamp01(float(os.getenv('CANARY_MAX_WEIGHT', '0.08')) / 0.20)) or 0.0)))

            _BEHAVIOR_SNAPSHOT_CONTEXT_CACHE['loaded_at_ts'] = now_ts
            _BEHAVIOR_SNAPSHOT_CONTEXT_CACHE['values'] = dict(values)
            return values
    except Exception:
        pass

    health = os.path.join(project_root, 'governance', 'health')
    coverage = _behavior_load_json(os.path.join(health, 'snapshot_coverage_latest.json'))
    replay = _behavior_load_json(os.path.join(health, 'replay_preopen_sanity_latest.json'))
    drift = _behavior_load_json(os.path.join(health, 'preopen_replay_drift_latest.json'))
    divergence = _behavior_load_json(os.path.join(health, 'data_source_divergence_latest.json'))
    triprate = _behavior_load_json(os.path.join(health, 'guardrail_triprate_latest.json'))
    queue_stress = _behavior_load_json(os.path.join(health, 'execution_queue_stress_latest.json'))

    coverage_ratio = float(coverage.get('coverage_ratio', 0.0) or 0.0)
    coverage_log_ratio = _behavior_clamp01(math.log1p(max(coverage_ratio, 0.0)) / 6.0)
    rows_scanned = max(float(coverage.get('rows_scanned', 0.0) or 0.0), 1.0)
    rows_with_sid = float(coverage.get('rows_with_snapshot_id', 0.0) or 0.0)
    coverage_fill_ratio = _behavior_clamp01(rows_with_sid / rows_scanned)

    replay_decision_stale = float((replay.get('decision') or {}).get('stale_windows', 0.0) or 0.0)
    replay_governance_stale = float((replay.get('governance') or {}).get('stale_windows', 0.0) or 0.0)
    replay_max_decision_stale = max(float((replay.get('thresholds') or {}).get('max_decision_stale_windows', 12.0) or 12.0), 1.0)
    replay_max_governance_stale = max(float((replay.get('thresholds') or {}).get('max_governance_stale_windows', 12.0) or 12.0), 1.0)
    replay_stale_ratio = _behavior_clamp01((replay_decision_stale + replay_governance_stale) / (replay_max_decision_stale + replay_max_governance_stale))

    drift_obj = drift.get('drift') or {}
    thresholds_obj = drift.get('thresholds') or {}
    row_drift = max(abs(float(drift_obj.get('decision_rows', 0.0) or 0.0)), abs(float(drift_obj.get('governance_rows', 0.0) or 0.0)))
    stale_drift = max(abs(float(drift_obj.get('decision_stale', 0.0) or 0.0)), abs(float(drift_obj.get('governance_stale', 0.0) or 0.0)))
    max_row_drift = max(float(thresholds_obj.get('max_row_drift', 1.2) or 1.2), 1e-6)
    max_stale_drift = max(float(thresholds_obj.get('max_stale_drift', 1.0) or 1.0), 1e-6)
    replay_drift_ratio = _behavior_clamp01((0.6 * (row_drift / max_row_drift)) + (0.4 * (stale_drift / max_stale_drift)))

    worst_spread = float(divergence.get('worst_relative_spread', 0.0) or 0.0)
    max_spread = max(float(divergence.get('max_relative_spread', 0.03) or 0.03), 1e-6)
    divergence_ratio = _behavior_clamp01(worst_spread / max_spread)

    trip_rate = float(triprate.get('trip_rate', 0.0) or 0.0)
    max_trip_rate = max(float(triprate.get('max_trip_rate', 0.4) or 0.4), 1e-6)
    triprate_ratio = _behavior_clamp01(trip_rate / max_trip_rate)

    depth_seen = float(queue_stress.get('max_queue_depth_seen', 0.0) or 0.0)
    depth_max = max(float(queue_stress.get('max_queue_depth', 2000.0) or 2000.0), 1.0)
    depth_ratio = _behavior_clamp01(depth_seen / depth_max)
    breach_rate = float(queue_stress.get('queue_breach_rate', 0.0) or 0.0)
    breach_rate_max = max(float(queue_stress.get('max_queue_breach_rate', 0.25) or 0.25), 1e-6)
    breach_ratio = _behavior_clamp01(breach_rate / breach_rate_max)
    queue_pressure_ratio = _behavior_clamp01(max(depth_ratio, breach_ratio))

    canary_weight_cap_norm = _behavior_clamp01(float(os.getenv('CANARY_MAX_WEIGHT', '0.08')) / 0.20)

    values: Dict[str, float] = {
        'snapshot_cov_ok': 1.0 if bool(coverage.get('ok', False)) else 0.0,
        'snapshot_cov_log_ratio': coverage_log_ratio,
        'snapshot_cov_fill_ratio': coverage_fill_ratio,
        'snapshot_replay_ok': 1.0 if bool(replay.get('ok', False)) else 0.0,
        'snapshot_replay_stale_ratio': replay_stale_ratio,
        'snapshot_replay_drift_ratio': replay_drift_ratio,
        'snapshot_divergence_ratio': divergence_ratio,
        'snapshot_triprate_ratio': triprate_ratio,
        'snapshot_queue_pressure_ratio': queue_pressure_ratio,
        'canary_weight_cap_norm': canary_weight_cap_norm,
    }

    _BEHAVIOR_SNAPSHOT_CONTEXT_CACHE['loaded_at_ts'] = now_ts
    _BEHAVIOR_SNAPSHOT_CONTEXT_CACHE['values'] = dict(values)
    return values


def _behavior_regime_index(symbol: str, features: Dict[str, float]) -> float:
    s = (symbol or '').upper()
    pct = float(features.get('pct_from_close', 0.0) or 0.0)
    mom = float(features.get('mom_5m', 0.0) or 0.0)
    vol = float(features.get('vol_30m', 0.0) or 0.0)

    if s in _BEHAVIOR_SHOCK_SYMBOLS or vol >= 0.03 or abs(pct) >= 0.04:
        return 2.0 / 3.0
    if s in _BEHAVIOR_MEAN_REVERT_SYMBOLS:
        return 1.0 / 3.0
    if abs(mom) >= 0.001 or abs(pct) >= 0.0015:
        return 0.0
    return 1.0


def _behavior_label_confidence_proxy(features: Dict[str, float]) -> float:
    pct = abs(float(features.get('pct_from_close', 0.0) or 0.0))
    mom = abs(float(features.get('mom_5m', 0.0) or 0.0))
    vol = abs(float(features.get('vol_30m', 0.0) or 0.0))
    edge = pct + (0.5 * mom) + (0.25 * vol)
    return _behavior_clamp01(edge * 25.0)


def _behavior_role_index_from_env() -> float:
    role_raw = os.getenv('BEHAVIOR_ACCOUNT_ROLE', 'INDIVIDUAL_TRADING').strip().upper()
    if role_raw == 'ROTH':
        return 0.0
    if role_raw == 'INDIVIDUAL_TRADING':
        return 1.0 / 3.0
    if role_raw == 'INDIVIDUAL_SWING':
        return 2.0 / 3.0
    return 1.0


def _behavior_session_event_context() -> Dict[str, float]:
    now_et = _now_eastern()
    now_min = now_et.hour * 60 + now_et.minute
    open_min = 9 * 60 + 30
    close_min = 16 * 60

    if now_min < open_min:
        session_bucket_norm = 0.0
    elif now_min <= close_min:
        session_bucket_norm = 0.5
    else:
        session_bucket_norm = 1.0

    mins_from_open_norm = _behavior_clamp01((now_min - open_min) / 390.0)
    mins_to_close_norm = _behavior_clamp01((close_min - now_min) / 390.0)

    event_window_proximity = 0.0
    for start_min, end_min in _event_blackout_windows():
        if start_min <= end_min:
            if start_min <= now_min <= end_min:
                event_window_proximity = 1.0
                break
            dist = min(abs(now_min - start_min), abs(now_min - end_min))
        else:
            in_window = now_min >= start_min or now_min <= end_min
            if in_window:
                event_window_proximity = 1.0
                break
            dist = min(abs(now_min - start_min), abs(now_min - end_min))
        event_window_proximity = max(event_window_proximity, _behavior_clamp01(1.0 - (dist / 30.0)))

    return {
        'session_bucket_norm': session_bucket_norm,
        'mins_from_open_norm': mins_from_open_norm,
        'mins_to_close_norm': mins_to_close_norm,
        'event_window_proximity': event_window_proximity,
    }


def _behavior_legacy_feature_vector(symbol: str, action_hint: str, features: Dict[str, float], snapshot_ctx: Dict[str, float]) -> Optional[Any]:
    if np is None:
        return None

    pct = float(features.get('pct_from_close', 0.0) or 0.0)
    mom = float(features.get('mom_5m', 0.0) or 0.0)
    vol = float(features.get('vol_30m', 0.0) or 0.0)

    vec = [
        math.tanh((pct + 0.5 * mom - 0.25 * vol) * 100.0),
        math.log1p(1.0),
        _behavior_role_index_from_env(),
        _hash01(symbol.upper()),
        _hash01(action_hint.upper()),
        datetime.now(timezone.utc).weekday() / 6.0,
        datetime.now(timezone.utc).hour / 23.0,
        _behavior_regime_index(symbol, features),
        _behavior_label_confidence_proxy(features),
    ]
    for key in _BEHAVIOR_SNAPSHOT_FEATURE_NAMES_V1:
        vec.append(float(snapshot_ctx.get(key, 0.0) or 0.0))

    return np.asarray([vec], dtype=float)


def _behavior_feature_vector_v2(symbol: str, action_hint: str, features: Dict[str, float], snapshot_ctx: Dict[str, float]) -> Optional[Any]:
    if np is None:
        return None

    pct = float(features.get('pct_from_close', 0.0) or 0.0)
    mom = float(features.get('mom_5m', 0.0) or 0.0)
    vol = float(features.get('vol_30m', 0.0) or 0.0)
    range_pos = _behavior_clamp01(float(features.get('range_pos', 0.5) or 0.5))
    spread_bps = abs(float(features.get('spread_bps', 0.0) or 0.0))

    ctx_vix_pct = float(
        features.get('ctx_VIX_X_pct_from_close', features.get('ctx_VIX_pct_from_close', 0.0))
        or 0.0
    )
    ctx_uup_pct = float(features.get('ctx_UUP_pct_from_close', 0.0) or 0.0)

    lag_slippage_bps = float(
        features.get('lag_slippage_bps', features.get('slippage_bps', features.get('execution_slippage_bps', 0.0)))
        or 0.0
    )
    lag_latency_ms = float(
        features.get('lag_latency_ms', features.get('latency_ms', features.get('execution_latency_ms', 0.0)))
        or 0.0
    )
    lag_impact_bps = float(
        features.get('lag_impact_bps', features.get('impact_bps', features.get('execution_impact_bps', 0.0)))
        or 0.0
    )

    active_sub_bots = float(features.get('active_sub_bots', 0.0) or 0.0)
    queue_depth = float(features.get('queue_depth', features.get('execution_queue_depth', 0.0)) or 0.0)
    dispatch_qty = abs(float(features.get('dispatch_qty', features.get('sized_qty', 0.0)) or 0.0))

    feature_freshness_enabled = os.getenv('FEATURE_FRESHNESS_GUARD_ENABLED', '1').strip() == '1'
    feature_freshness_max_age_seconds = float(os.getenv('FEATURE_FRESHNESS_MAX_AGE_SECONDS', '12'))
    feature_freshness_required = [
        s.strip()
        for s in os.getenv(
            'FEATURE_FRESHNESS_REQUIRED_KEYS',
            'last_price,prev_close,pct_from_close,vol_30m,mom_5m',
        ).split(',')
        if s.strip()
    ]
    freshness_ok, _, freshness_age_s = (
        _feature_freshness_guard(
            features,
            max_age_seconds=feature_freshness_max_age_seconds,
            required_keys=feature_freshness_required,
        )
        if feature_freshness_enabled
        else (True, 'disabled', 0.0)
    )

    master_latency_timeout_ms = max(float(os.getenv('MASTER_LATENCY_SLO_TIMEOUT_MS', '800')), 1.0)
    master_latency_ms = float(features.get('master_latency_ms', features.get('elapsed_ms', 0.0)) or 0.0)
    master_latency_ratio = _behavior_clamp01(master_latency_ms / master_latency_timeout_ms)
    if 'master_latency_slo_ok' in features:
        master_latency_slo_ok = 1.0 if bool(features.get('master_latency_slo_ok')) else 0.0
    else:
        master_latency_slo_ok = 1.0 if master_latency_ratio <= 1.0 else 0.0

    risk_pause_active = 1.0 if (
        _global_trading_halt_enabled()
        or bool(features.get('risk_pause_active', False))
        or bool(features.get('kill_switch_active', False))
    ) else 0.0

    session_ctx = _behavior_session_event_context()

    vec_map: Dict[str, float] = {
        'pnl_proxy': _behavior_signed_scale((pct + (0.5 * mom) - (0.25 * vol)) * 100.0, 1.0),
        'qty_log': math.log1p(1.0),
        'role_idx': _behavior_role_index_from_env(),
        'symbol_hash': _hash01(symbol.upper()),
        'action_hash': _hash01(action_hint.upper()),
        'dow': datetime.now(timezone.utc).weekday() / 6.0,
        'hour': datetime.now(timezone.utc).hour / 23.0,
        'regime_idx': _behavior_regime_index(symbol, features),
        'label_confidence_proxy': _behavior_label_confidence_proxy(features),
        'pct_from_close_scaled': _behavior_signed_scale(pct, 40.0),
        'mom_5m_scaled': _behavior_signed_scale(mom, 120.0),
        'vol_30m_scaled': _behavior_signed_scale(vol, 60.0),
        'range_pos': range_pos,
        'spread_bps_norm': _behavior_clamp01(spread_bps / 25.0),
        'ctx_vix_pct_scaled': _behavior_signed_scale(ctx_vix_pct, 60.0),
        'ctx_uup_pct_scaled': _behavior_signed_scale(ctx_uup_pct, 60.0),
        'lag_slippage_bps_norm': _behavior_clamp01(abs(lag_slippage_bps) / 10.0),
        'lag_latency_ms_norm': _behavior_clamp01(lag_latency_ms / 350.0),
        'lag_impact_bps_norm': _behavior_clamp01(abs(lag_impact_bps) / 10.0),
        'active_sub_bots_norm': _behavior_clamp01(active_sub_bots / 60.0),
        'queue_depth_norm': _behavior_clamp01(queue_depth / 1000.0),
        'dispatch_qty_norm': _behavior_clamp01(dispatch_qty / 20.0),
        'session_bucket_norm': session_ctx['session_bucket_norm'],
        'mins_from_open_norm': session_ctx['mins_from_open_norm'],
        'mins_to_close_norm': session_ctx['mins_to_close_norm'],
        'event_window_proximity': session_ctx['event_window_proximity'],
        'feature_freshness_ok': 1.0 if freshness_ok else 0.0,
        'feature_freshness_age_ratio': _behavior_clamp01(freshness_age_s / max(feature_freshness_max_age_seconds, 1.0)),
        'master_latency_slo_ok': master_latency_slo_ok,
        'master_latency_ratio': master_latency_ratio,
        'risk_pause_active': risk_pause_active,
        'options_chain_available': _behavior_clamp01(float(features.get('options_chain_available', 0.0) or 0.0)),
        'options_iv_atm_norm': _behavior_clamp01(float(features.get('options_iv_atm_norm', 0.0) or 0.0)),
        'options_iv_skew_norm': _behavior_clamp01(float(features.get('options_iv_skew_norm', 0.0) or 0.0)),
        'options_iv_term_structure_norm': _behavior_clamp01(float(features.get('options_iv_term_structure_norm', 0.0) or 0.0)),
        'options_put_call_oi_ratio_norm': _behavior_clamp01(float(features.get('options_put_call_oi_ratio_norm', 0.0) or 0.0)),
        'options_negative_bias_norm': _behavior_clamp01(float(features.get('options_negative_bias_norm', 0.0) or 0.0)),
        'options_roll_yield_norm': _behavior_clamp01(float(features.get('options_roll_yield_norm', 0.0) or 0.0)),
        'options_vwap_bias_norm': _behavior_clamp01(float(features.get('options_vwap_bias_norm', 0.0) or 0.0)),
        'options_vol_expectation_norm': _behavior_clamp01(float(features.get('options_vol_expectation_norm', 0.0) or 0.0)),
        'calendar_event_proximity_norm': _behavior_clamp01(float(features.get('calendar_event_proximity_norm', 0.0) or 0.0)),
        'calendar_high_impact_24h_norm': _behavior_clamp01(float(features.get('calendar_high_impact_24h_norm', 0.0) or 0.0)),
        'calendar_options_expiry_week_norm': _behavior_clamp01(float(features.get('calendar_options_expiry_week_norm', 0.0) or 0.0)),
        'calendar_dividend_events_30d_norm': _behavior_clamp01(float(features.get('calendar_dividend_events_30d_norm', 0.0) or 0.0)),
        'calendar_dividend_exdate_proximity_norm': _behavior_clamp01(float(features.get('calendar_dividend_exdate_proximity_norm', 0.0) or 0.0)),
        'calendar_dividend_payout_proximity_norm': _behavior_clamp01(float(features.get('calendar_dividend_payout_proximity_norm', 0.0) or 0.0)),
        'calendar_dividend_recent_exdate_norm': _behavior_clamp01(float(features.get('calendar_dividend_recent_exdate_norm', 0.0) or 0.0)),
        'calendar_dividend_quality_signal_norm': _behavior_clamp01(float(features.get('calendar_dividend_quality_signal_norm', 0.0) or 0.0)),
        'dividend_yield_norm': _behavior_clamp01(float(features.get('dividend_yield_norm', 0.0) or 0.0)),
        'dividend_payout_ratio_norm': _behavior_clamp01(float(features.get('dividend_payout_ratio_norm', 0.0) or 0.0)),
        'dividend_ex_date_proximity_norm': _behavior_clamp01(float(features.get('dividend_ex_date_proximity_norm', 0.0) or 0.0)),
        'dividend_pay_date_proximity_norm': _behavior_clamp01(float(features.get('dividend_pay_date_proximity_norm', 0.0) or 0.0)),
        'dividend_quality_score_norm': _behavior_clamp01(float(features.get('dividend_quality_score_norm', 0.0) or 0.0)),
        'dividend_capture_entry_signal_norm': _behavior_clamp01(float(features.get('dividend_capture_entry_signal_norm', 0.0) or 0.0)),
        'dividend_capture_exit_signal_norm': _behavior_clamp01(float(features.get('dividend_capture_exit_signal_norm', 0.0) or 0.0)),
        'dividend_compound_bias_norm': _behavior_clamp01(float(features.get('dividend_compound_bias_norm', 0.0) or 0.0)),
        'dividend_compound_growth_norm': _behavior_clamp01(float(features.get('dividend_compound_growth_norm', 0.0) or 0.0)),
        'dividend_compound_drawdown_norm': _behavior_clamp01(float(features.get('dividend_compound_drawdown_norm', 0.0) or 0.0)),
        'dividend_compound_steps_norm': _behavior_clamp01(float(features.get('dividend_compound_steps_norm', 0.0) or 0.0)),
        'dividend_strategy_mode_capture': _behavior_clamp01(float(features.get('dividend_strategy_mode_capture', 0.0) or 0.0)),
        'dividend_strategy_mode_compound': _behavior_clamp01(float(features.get('dividend_strategy_mode_compound', 0.0) or 0.0)),
        'dividend_strategy_mode_hybrid': _behavior_clamp01(float(features.get('dividend_strategy_mode_hybrid', 0.0) or 0.0)),
        'futures_order_book_imbalance_norm': _behavior_clamp01(float(features.get('futures_order_book_imbalance_norm', 0.0) or 0.0)),
        'futures_funding_rate_norm': _behavior_clamp01(float(features.get('futures_funding_rate_norm', 0.0) or 0.0)),
        'futures_basis_bps_norm': _behavior_clamp01(float(features.get('futures_basis_bps_norm', 0.0) or 0.0)),
        'futures_term_structure_norm': _behavior_clamp01(float(features.get('futures_term_structure_norm', 0.0) or 0.0)),
        'futures_negative_bias_norm': _behavior_clamp01(float(features.get('futures_negative_bias_norm', 0.0) or 0.0)),
        'futures_roll_yield_norm': _behavior_clamp01(float(features.get('futures_roll_yield_norm', 0.0) or 0.0)),
        'futures_vwap_bias_norm': _behavior_clamp01(float(features.get('futures_vwap_bias_norm', 0.0) or 0.0)),
        'options_specialist_active': _behavior_clamp01(float(features.get('options_specialist_active', 0.0) or 0.0)),
        'futures_specialist_active': _behavior_clamp01(float(features.get('futures_specialist_active', 0.0) or 0.0)),
        'options_specialist_vote': _behavior_signed_scale(float(features.get('options_specialist_vote', 0.0) or 0.0), 1.0),
        'futures_specialist_vote': _behavior_signed_scale(float(features.get('futures_specialist_vote', 0.0) or 0.0), 1.0),
        'active_options_sub_bots_norm': _behavior_clamp01(float(features.get('active_options_sub_bots', 0.0) or 0.0) / 20.0),
        'active_futures_sub_bots_norm': _behavior_clamp01(float(features.get('active_futures_sub_bots', 0.0) or 0.0) / 20.0),
        'snapshot_cov_ok': _behavior_clamp01(float(snapshot_ctx.get('snapshot_cov_ok', 1.0) or 1.0)),
        'snapshot_cov_log_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_cov_log_ratio', 0.0) or 0.0)),
        'snapshot_replay_stale_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_replay_stale_ratio', 0.0) or 0.0)),
        'snapshot_replay_drift_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_replay_drift_ratio', 0.0) or 0.0)),
        'snapshot_divergence_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_divergence_ratio', 0.0) or 0.0)),
        'snapshot_triprate_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_triprate_ratio', 0.0) or 0.0)),
        'snapshot_queue_pressure_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_queue_pressure_ratio', 0.0) or 0.0)),
        'snapshot_drill_ok': _behavior_clamp01(float(snapshot_ctx.get('snapshot_drill_ok', 0.0) or 0.0)),
        'snapshot_drill_restore_fail_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_drill_restore_fail_ratio', 0.0) or 0.0)),
        'snapshot_drill_missing_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_drill_missing_ratio', 0.0) or 0.0)),
        'snapshot_drill_recency_norm': _behavior_clamp01(float(snapshot_ctx.get('snapshot_drill_recency_norm', 0.0) or 0.0)),
        'canary_weight_cap_norm': _behavior_clamp01(float(snapshot_ctx.get('canary_weight_cap_norm', 0.0) or 0.0)),
        'snapshot_raw_sql_ingest_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_raw_sql_ingest_ratio', 0.0) or 0.0)),
        'snapshot_raw_count_norm': _behavior_clamp01(float(snapshot_ctx.get('snapshot_raw_count_norm', 0.0) or 0.0)),
        'snapshot_raw_file_count_norm': _behavior_clamp01(float(snapshot_ctx.get('snapshot_raw_file_count_norm', 0.0) or 0.0)),
        'snapshot_raw_bytes_norm': _behavior_clamp01(float(snapshot_ctx.get('snapshot_raw_bytes_norm', 0.0) or 0.0)),
        'snapshot_raw_json_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_raw_json_ratio', 0.0) or 0.0)),
        'snapshot_raw_event_file_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_raw_event_file_ratio', 0.0) or 0.0)),
        'snapshot_raw_lock_file_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_raw_lock_file_ratio', 0.0) or 0.0)),
        'snapshot_raw_recency_norm': _behavior_clamp01(float(snapshot_ctx.get('snapshot_raw_recency_norm', 0.0) or 0.0)),
        'snapshot_cov_fill_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_cov_fill_ratio', 0.0) or 0.0)),
        'snapshot_replay_ok': _behavior_clamp01(float(snapshot_ctx.get('snapshot_replay_ok', 0.0) or 0.0)),
        'snapshot_e2e_replay_ok': _behavior_clamp01(float(snapshot_ctx.get('snapshot_e2e_replay_ok', 0.0) or 0.0)),
        'snapshot_e2e_hash_match': _behavior_clamp01(float(snapshot_ctx.get('snapshot_e2e_hash_match', 0.0) or 0.0)),
        'snapshot_paper_replay_ok': _behavior_clamp01(float(snapshot_ctx.get('snapshot_paper_replay_ok', 0.0) or 0.0)),
        'snapshot_paper_replay_hash_match': _behavior_clamp01(float(snapshot_ctx.get('snapshot_paper_replay_hash_match', 0.0) or 0.0)),
        'external_feeds_ok': _behavior_clamp01(float(snapshot_ctx.get('external_feeds_ok', features.get('external_feeds_ok', 0.0)) or 0.0)),
        'external_feeds_recency_norm': _behavior_clamp01(float(snapshot_ctx.get('external_feeds_recency_norm', features.get('external_feeds_recency_norm', 0.0)) or 0.0)),
        'external_fred_unrate_norm': _behavior_clamp01(float(snapshot_ctx.get('external_fred_unrate_norm', features.get('external_fred_unrate_norm', 0.0)) or 0.0)),
        'external_fred_cpi_mom_norm': _behavior_clamp01(float(snapshot_ctx.get('external_fred_cpi_mom_norm', features.get('external_fred_cpi_mom_norm', 0.0)) or 0.0)),
        'external_fred_gdp_qoq_norm': _behavior_clamp01(float(snapshot_ctx.get('external_fred_gdp_qoq_norm', features.get('external_fred_gdp_qoq_norm', 0.0)) or 0.0)),
        'external_bls_unrate_norm': _behavior_clamp01(float(snapshot_ctx.get('external_bls_unrate_norm', features.get('external_bls_unrate_norm', 0.0)) or 0.0)),
        'external_bls_cpi_mom_norm': _behavior_clamp01(float(snapshot_ctx.get('external_bls_cpi_mom_norm', features.get('external_bls_cpi_mom_norm', 0.0)) or 0.0)),
        'external_census_population_log_norm': _behavior_clamp01(float(snapshot_ctx.get('external_census_population_log_norm', features.get('external_census_population_log_norm', 0.0)) or 0.0)),
        'external_bea_dataset_count_norm': _behavior_clamp01(float(snapshot_ctx.get('external_bea_dataset_count_norm', features.get('external_bea_dataset_count_norm', 0.0)) or 0.0)),
    }

    for key in _BEHAVIOR_LANE_FEATURE_NAMES:
        vec_map[key] = _behavior_clamp01(float(features.get(key, 0.0) or 0.0))
    for key in _BEHAVIOR_CAPITAL_FLOW_FEATURE_NAMES:
        raw = float(features.get(key, 0.0) or 0.0)
        if key == "capital_flow_signed_scaled":
            vec_map[key] = max(-1.0, min(raw, 1.0))
        else:
            vec_map[key] = _behavior_clamp01(raw)

    vec = [float(vec_map.get(name, 0.0) or 0.0) for name in _BEHAVIOR_FEATURE_NAMES_V2]
    return np.asarray([vec], dtype=float)


def _behavior_feature_vector(
    symbol: str,
    action_hint: str,
    features: Dict[str, float],
    *,
    feature_dim_hint: Optional[int] = None,
) -> Optional[Any]:
    snapshot_ctx = _behavior_snapshot_context(PROJECT_ROOT)

    # Preserve compatibility with older 18-feature models until retrain rolls out everywhere.
    legacy_dim = 9 + len(_BEHAVIOR_SNAPSHOT_FEATURE_NAMES_V1)
    if feature_dim_hint is not None and int(feature_dim_hint) <= legacy_dim:
        return _behavior_legacy_feature_vector(symbol, action_hint, features, snapshot_ctx)

    return _behavior_feature_vector_v2(symbol, action_hint, features, snapshot_ctx)



def _behavior_prior_from_model(
    model: Optional[Dict[str, Any]],
    *,
    symbol: str,
    action_hint: str,
    features: Dict[str, float],
) -> tuple[float, Dict[str, float]]:
    if model is None or np is None:
        return 0.0, {}

    try:
        mu = model["mu"]
        sigma = model["sigma"]
        W = model["W"]
        b = model["b"]

        expected_dim = int(np.asarray(mu).shape[0])
        if expected_dim <= 0:
            return 0.0, {}

        x = _behavior_feature_vector(symbol, action_hint, features, feature_dim_hint=expected_dim)
        if x is None:
            return 0.0, {}
        if x.shape[1] < expected_dim:
            pad = np.zeros((x.shape[0], expected_dim - x.shape[1]), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=1)
        elif x.shape[1] > expected_dim:
            x = x[:, :expected_dim]

        xz = (x - mu) / np.where(np.abs(sigma) < 1e-8, 1.0, sigma)
        temperature = float(model.get("temperature", 1.0) or 1.0)
        if (not math.isfinite(temperature)) or temperature <= 0.0:
            temperature = 1.0
        logits = (xz @ W + b) / temperature
        class_logit_bias = np.asarray(model.get("class_logit_bias", np.zeros((3,), dtype=float))).reshape(1, -1)
        if class_logit_bias.shape[1] == logits.shape[1]:
            logits = logits + class_logit_bias
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.clip(np.sum(probs, axis=1, keepdims=True), 1e-8, None)

        neg = float(probs[0, 0])
        neu = float(probs[0, 1])
        pos = float(probs[0, 2])
        prior = pos - neg

        return prior, {
            "behavior_prob_negative": neg,
            "behavior_prob_neutral": neu,
            "behavior_prob_positive": pos,
            "behavior_prior": prior,
            "behavior_temperature": temperature,
            "behavior_logit_bias_negative": float(class_logit_bias[0, 0]) if class_logit_bias.shape[1] >= 1 else 0.0,
            "behavior_logit_bias_neutral": float(class_logit_bias[0, 1]) if class_logit_bias.shape[1] >= 2 else 0.0,
            "behavior_logit_bias_positive": float(class_logit_bias[0, 2]) if class_logit_bias.shape[1] >= 3 else 0.0,
        }
    except Exception:
        return 0.0, {}



def _governance_recommendations(bots: List[SubBot]) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for b in bots:
        if b.active:
            continue
        acc = b.test_accuracy
        if acc is None:
            decision = "RETRAIN"
            why = "missing_classification_accuracy"
        elif acc < 0.50:
            decision = "REMOVE_OR_REBUILD"
            why = "persistent_underperformance_below_0.50"
        else:
            decision = "RETRAIN"
            why = "below_quality_floor"

        recs.append(
            {
                "bot_id": b.bot_id,
                "test_accuracy": acc,
                "current_reason": b.reason,
                "recommended_action": decision,
                "why": why,
            }
        )
    return recs


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    payload = enrich_log_row(dict(row or {}))
    if ("parent_message_id" not in payload) and str(payload.get("parent_decision_id") or "").strip():
        payload["parent_message_id"] = str(payload.get("parent_decision_id") or "").strip()
    JsonlWriteBuffer.shared().append(path, payload)


class JsonlWriteBuffer:
    _instance: Optional["JsonlWriteBuffer"] = None
    _recent_message_ids: Dict[str, Dict[str, float]] = {}
    _recent_message_ids_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "JsonlWriteBuffer":
        if cls._instance is None:
            cls._instance = cls()
            atexit.register(cls._instance.flush_all)
        return cls._instance

    def __init__(self) -> None:
        self.enabled = os.getenv("JSONL_BUFFER_ENABLED", "1").strip() == "1"
        self.max_items = max(int(os.getenv("JSONL_BUFFER_MAX_ITEMS", "80")), 1)
        self.max_age_seconds = max(float(os.getenv("JSONL_BUFFER_MAX_AGE_SECONDS", "2.5")), 0.2)
        self._buf: Dict[str, List[Dict[str, Any]]] = {}
        self._ts: Dict[str, float] = {}

    def append(self, path: str, row: Dict[str, Any]) -> None:
        if not self.enabled:
            self._flush_batch(path, [row])
            return
        now = time.time()
        bucket = self._buf.setdefault(path, [])
        if path not in self._ts:
            self._ts[path] = now
        bucket.append(dict(row or {}))
        if len(bucket) >= self.max_items or (now - self._ts.get(path, now)) >= self.max_age_seconds:
            self.flush_path(path)

    def flush_path(self, path: str) -> None:
        rows = self._buf.get(path, [])
        if not rows:
            return
        self._flush_batch(path, rows)
        self._buf[path] = []
        self._ts[path] = time.time()

    def flush_all(self) -> None:
        for path in list(self._buf.keys()):
            self.flush_path(path)

    @classmethod
    def _dedupe_window_seconds(cls) -> float:
        return max(float(os.getenv("JSONL_MESSAGE_ID_DEDUP_WINDOW_SECONDS", "900") or 900.0), 0.0)

    @classmethod
    def _dedupe_rows(cls, path: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return []

        window_seconds = cls._dedupe_window_seconds()
        if window_seconds <= 0.0:
            return list(rows)

        now = time.time()
        cutoff = now - window_seconds
        path_key = os.path.abspath(path)
        deduped: List[Dict[str, Any]] = []
        batch_seen: set[str] = set()

        with cls._recent_message_ids_lock:
            bucket = cls._recent_message_ids.setdefault(path_key, {})
            for key in [msg_id for msg_id, ts in bucket.items() if ts < cutoff]:
                bucket.pop(key, None)

            for row in rows:
                msg_id = str((row or {}).get("message_id") or "").strip()
                if not msg_id:
                    deduped.append(row)
                    continue
                if msg_id in batch_seen or msg_id in bucket:
                    continue
                batch_seen.add(msg_id)
                deduped.append(row)

        return deduped

    @classmethod
    def _remember_written_rows(cls, path: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        window_seconds = cls._dedupe_window_seconds()
        if window_seconds <= 0.0:
            return

        now = time.time()
        cutoff = now - window_seconds
        path_key = os.path.abspath(path)
        with cls._recent_message_ids_lock:
            bucket = cls._recent_message_ids.setdefault(path_key, {})
            for key in [msg_id for msg_id, ts in bucket.items() if ts < cutoff]:
                bucket.pop(key, None)
            for row in rows:
                msg_id = str((row or {}).get("message_id") or "").strip()
                if msg_id:
                    bucket[msg_id] = now

    @staticmethod
    def _flush_batch(path: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        rows = JsonlWriteBuffer._dedupe_rows(path, rows)
        if not rows:
            return

        channel = classify_channel_from_path(path)
        ctx = _path_context()
        mirror_paths = default_channel_mirror_paths(path, project_root=PROJECT_ROOT, ctx=ctx)

        def _write_once() -> int:
            if channel:
                return safe_append_channel_batch(
                    path,
                    rows,
                    project_root=PROJECT_ROOT,
                    source="run_shadow_training_loop.JsonlWriteBuffer",
                    channel=channel,
                    schema=channel,
                    mirror_paths=mirror_paths,
                )
            return safe_append_jsonl_batch(
                path,
                rows,
                project_root=PROJECT_ROOT,
                source="run_shadow_training_loop.JsonlWriteBuffer",
            )

        wrote = _write_once()
        if wrote >= len(rows):
            JsonlWriteBuffer._remember_written_rows(path, rows)
            return

        if not _route_storage_or_fail():
            print(f"[StorageRoute] write_failed path={path} wrote={wrote} expected={len(rows)}")
            JsonlWriteBuffer._emit_write_failure_event(path=path, error=RuntimeError("batch_write_failed"))
            return

        retry_wrote = _write_once()
        if retry_wrote >= len(rows):
            JsonlWriteBuffer._remember_written_rows(path, rows)
            return

        if retry_wrote < len(rows):
            print(f"[StorageRoute] write_retry_failed path={path} wrote={retry_wrote} expected={len(rows)}")
            JsonlWriteBuffer._emit_write_failure_event(path=path, error=RuntimeError("batch_write_retry_failed"))

    @staticmethod
    def _emit_write_failure_event(path: str, error: Exception) -> None:
        try:
            day = datetime.now(timezone.utc).strftime("%Y%m%d")
            fail_path = os.path.join(PROJECT_ROOT, "governance", "events", f"write_failures_{day}.jsonl")
            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "write_failure",
                "source": "run_shadow_training_loop.JsonlWriteBuffer",
                "target_path": path,
                "error": str(error),
                "error_type": type(error).__name__,
            }
            safe_append_jsonl_batch(
                fail_path,
                [row],
                project_root=PROJECT_ROOT,
                source="run_shadow_training_loop.JsonlWriteBuffer",
            )
        except Exception:
            return


def _split_bot_tiers(active_bots: List[SubBot]) -> Tuple[List[SubBot], List[SubBot]]:
    if not active_bots:
        return [], []
    slow_every = max(int(os.getenv("SLOW_BOT_EVERY_N_ITERS", "2")), 1)
    min_weight_fast = float(os.getenv("FAST_BOT_MIN_WEIGHT", "0.025"))
    min_acc_fast = float(os.getenv("FAST_BOT_MIN_ACC", "0.53"))
    if slow_every <= 1:
        return active_bots, []

    fast: List[SubBot] = []
    slow: List[SubBot] = []
    for b in active_bots:
        acc = float(b.test_accuracy or 0.5)
        if b.bot_role == "infrastructure_sub_bot" or _is_options_sub_bot(b) or _is_futures_sub_bot(b):
            slow.append(b)
            continue
        if b.weight >= min_weight_fast or acc >= min_acc_fast:
            fast.append(b)
        else:
            slow.append(b)
    if not fast:
        fast = active_bots[:]
        slow = []
    return fast, slow


def _top_paper_mirror_bots(
    active_bots: List[SubBot],
    *,
    top_n: int,
    min_accuracy: float,
) -> List[SubBot]:
    if top_n <= 0:
        return []
    ranked = sorted(
        [
            b for b in active_bots
            if b.active and b.bot_role != "infrastructure_sub_bot" and (not _is_options_sub_bot(b)) and (not _is_futures_sub_bot(b))
        ],
        key=lambda b: (float(b.test_accuracy or 0.0), float(b.weight or 0.0), b.bot_id),
        reverse=True,
    )
    out: List[SubBot] = []
    for b in ranked:
        if float(b.test_accuracy or 0.0) < min_accuracy:
            continue
        out.append(b)
        if len(out) >= top_n:
            break
    return out


def _sub_bot_signal_batch(bots: List[SubBot], mkt: Dict[str, float]) -> List[Tuple[SubBot, str, float, float, List[str]]]:
    out: List[Tuple[SubBot, str, float, float, List[str]]] = []
    for b in bots:
        action, score, threshold, reasons = _sub_bot_signal(b, mkt)
        out.append((b, action, score, threshold, reasons))
    return out


def _external_ingestion_extra_interval_seconds(project_root: str) -> int:
    path = os.path.join(project_root, "governance", "health", "ingestion_backpressure_latest.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if bool(payload.get("overload", False)):
            return max(int(payload.get("recommended_extra_interval_seconds", 0) or 0), 0)
    except Exception:
        return 0
    return 0




def _file_age_days(path: str, now_ts: float) -> float:
    try:
        return max((now_ts - os.path.getmtime(path)) / 86400.0, 0.0)
    except Exception:
        return 0.0


def _gzip_file(path: str) -> bool:
    if path.endswith('.gz'):
        return False
    gz_path = path + '.gz'
    if os.path.exists(gz_path):
        return False
    try:
        with open(path, 'rb') as src, gzip.open(gz_path, 'wb', compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        os.remove(path)
        return True
    except Exception:
        return False


def _run_log_maintenance(project_root: str, *, max_ops: int = 200) -> Dict[str, int]:
    now_ts = time.time()

    retention_decisions = int(os.getenv('LOG_RETENTION_DECISIONS_DAYS', '30'))
    retention_explanations = int(os.getenv('LOG_RETENTION_DECISION_EXPLANATIONS_DAYS', '30'))
    retention_governance = int(os.getenv('LOG_RETENTION_GOVERNANCE_DAYS', '45'))
    retention_exports = int(os.getenv('LOG_RETENTION_EXPORTS_DAYS', '30'))
    compress_after_days = float(os.getenv('LOG_COMPRESS_AFTER_DAYS', '1'))

    targets = [
        (os.path.join(project_root, 'decisions'), retention_decisions),
        (os.path.join(project_root, 'decision_explanations'), retention_explanations),
        (os.path.join(project_root, 'governance'), retention_governance),
        (os.path.join(project_root, 'exports'), retention_exports),
    ]

    ops = 0
    compressed = 0
    deleted = 0

    for base, retention_days in targets:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for name in files:
                if ops >= max_ops:
                    return {'compressed': compressed, 'deleted': deleted, 'ops': ops}

                path = os.path.join(root, name)
                age_days = _file_age_days(path, now_ts)

                # delete old artifacts
                if age_days >= max(float(retention_days), 1.0):
                    try:
                        os.remove(path)
                        deleted += 1
                        ops += 1
                    except Exception:
                        pass
                    continue

                # compress older verbose text/jsonl artifacts
                if age_days >= compress_after_days and (name.endswith('.jsonl') or name.endswith('.log')) and (not name.endswith('.gz')):
                    if _gzip_file(path):
                        compressed += 1
                        ops += 1

    return {'compressed': compressed, 'deleted': deleted, 'ops': ops}


def _shadow_profile_name() -> str:
    return os.getenv("SHADOW_PROFILE", "").strip().lower()


def _shadow_domain_name(broker: Optional[str] = None) -> str:
    raw = os.getenv("SHADOW_DOMAIN", "").strip().lower()
    if raw in {"equities", "crypto"}:
        return raw

    b = (broker or os.getenv("DATA_BROKER", "schwab")).strip().lower()
    return "crypto" if b == "coinbase" else "equities"


def _shadow_profile_subdir(broker: Optional[str] = None) -> str:
    prof = _shadow_profile_name()
    base = "shadow" if not prof else f"shadow_{prof}"
    domain = _shadow_domain_name(broker=broker)
    return f"{base}_{domain}" if domain else base


def _path_context(broker: Optional[str] = None):
    b = (broker or os.getenv("DATA_BROKER", "schwab")).strip().lower()
    return build_shadow_context(
        profile=_shadow_profile_name(),
        domain=_shadow_domain_name(broker=b),
        broker=b,
    )


def _threshold_shift() -> float:
    raw = os.getenv("SHADOW_THRESHOLD_SHIFT", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            return 0.0

    prof = _shadow_profile_name()
    if prof == "aggressive":
        return -0.03
    if prof in {"dividend", "long_term_dividend"}:
        # Dividend sleeve should be more selective and avoid noisy overnight churn.
        return +0.03
    if prof == "bond":
        # Bond sleeve: slightly stricter entry to reduce false positives in low-vol regimes.
        return +0.02
    if prof == "long_term_core_etf":
        return +0.035
    if prof == "long_term_sector_rotation":
        return +0.02
    return 0.0


def _shift_threshold(base: float) -> float:
    v = float(base) + _threshold_shift()
    return min(max(v, 0.45), 0.75)


def _is_dividend_profile() -> bool:
    return _shadow_profile_name() in {"dividend", "long_term_dividend"}


def _is_bond_profile() -> bool:
    return _shadow_profile_name() == "bond"


def _is_day_profile() -> bool:
    prof = _shadow_profile_name()
    if (not prof) or _is_dividend_profile() or _is_long_term_profile() or _is_bond_profile():
        return False
    if "swing" in prof:
        return False
    return any(tok in prof for tok in ("day", "intraday", "scalp", "aggressive"))


def _is_swing_profile() -> bool:
    prof = _shadow_profile_name()
    if (not prof) or _is_dividend_profile() or _is_long_term_profile() or _is_bond_profile():
        return False
    return any(tok in prof for tok in ("swing", "position"))


def _dividend_strategy_mode() -> str:
    mode = os.getenv("DIVIDEND_STRATEGY_MODE", "hybrid").strip().lower()
    if mode not in {"capture", "compound", "hybrid"}:
        return "hybrid"
    return mode


def _dividend_quality_symbols() -> set[str]:
    raw = os.getenv(
        "DIVIDEND_QUALITY_SYMBOLS",
        "SCHD,VIG,DGRO,HDV,NOBL,JNJ,PG,KO,PEP,MCD,XOM,CVX,ABT,MRK,O",
    ).strip()
    if not raw:
        return set()
    return {s.strip().upper() for s in raw.split(",") if s.strip()}


def _dividend_quality_score(symbol: str, features: Dict[str, float]) -> float:
    yield_norm = _clamp01(float(features.get("dividend_yield_norm", 0.0) or 0.0))
    payout_norm = _clamp01(float(features.get("dividend_payout_ratio_norm", 0.0) or 0.0))
    vol_norm = _clamp01(float(features.get("vol_30m", 0.0) or 0.0) / 0.03)
    calendar_quality = _clamp01(float(features.get("calendar_dividend_quality_signal_norm", 0.0) or 0.0))

    sustainable_yield = _clamp01(1.0 - (abs(yield_norm - 0.35) / 0.35))
    payout_safety = _clamp01(1.0 - (max(payout_norm - 0.78, 0.0) / 0.22))
    low_vol = _clamp01(1.0 - vol_norm)
    universe_bonus = 0.18 if symbol.upper() in _dividend_quality_symbols() else 0.0

    score = (
        (0.34 * sustainable_yield)
        + (0.32 * payout_safety)
        + (0.20 * low_vol)
        + (0.14 * calendar_quality)
        + universe_bonus
    )
    return _clamp01(score)


def _force_action_score(action: str, score: float, threshold: float) -> tuple[str, float]:
    if action == "BUY":
        return "BUY", max(float(score), min(float(threshold) + 0.03, 0.92))
    if action == "SELL":
        return "SELL", min(float(score), max((1.0 - float(threshold)) - 0.03, 0.08))
    hold_score = 0.5 + 0.35 * (float(score) - 0.5)
    return "HOLD", min(max(hold_score, 0.01), 0.99)


def _apply_dividend_strategy_overlay(
    *,
    symbol: str,
    action: str,
    score: float,
    threshold: float,
    reasons: List[str],
    features: Dict[str, float],
) -> tuple[str, float, List[str], Dict[str, float]]:
    if not _is_dividend_profile():
        return action, score, reasons, {}

    mode = _dividend_strategy_mode()
    quality_score = _dividend_quality_score(symbol, features)
    payout_ratio_norm = _clamp01(float(features.get("dividend_payout_ratio_norm", 0.0) or 0.0))

    cal_ex = _clamp01(float(features.get("calendar_dividend_exdate_proximity_norm", 0.0) or 0.0))
    cal_pay = _clamp01(float(features.get("calendar_dividend_payout_proximity_norm", 0.0) or 0.0))
    cal_recent_ex = _clamp01(float(features.get("calendar_dividend_recent_exdate_norm", 0.0) or 0.0))
    cal_events = _clamp01(float(features.get("calendar_dividend_events_30d_norm", 0.0) or 0.0))
    mkt_ex = _clamp01(float(features.get("dividend_ex_date_proximity_norm", 0.0) or 0.0))
    mkt_pay = _clamp01(float(features.get("dividend_pay_date_proximity_norm", 0.0) or 0.0))
    signal_available = _clamp01(float(features.get("dividend_signal_available", 0.0) or 0.0))

    ex_signal = max(cal_ex, mkt_ex)
    payout_signal = max(cal_pay, mkt_pay)
    capture_entry = _clamp01(((0.75 * ex_signal) + (0.25 * payout_signal)) * quality_score)
    capture_exit = _clamp01(max(cal_recent_ex, payout_signal * 0.75))
    compound_bias = _clamp01((0.65 * quality_score) + (0.25 * cal_events) + (0.10 * signal_available))

    out_features = {
        "dividend_strategy_mode_capture": 1.0 if mode == "capture" else 0.0,
        "dividend_strategy_mode_compound": 1.0 if mode == "compound" else 0.0,
        "dividend_strategy_mode_hybrid": 1.0 if mode == "hybrid" else 0.0,
        "dividend_quality_score_norm": quality_score,
        "dividend_capture_entry_signal_norm": capture_entry,
        "dividend_capture_exit_signal_norm": capture_exit,
        "dividend_compound_bias_norm": compound_bias,
    }

    new_action = str(action)
    new_score = float(score)
    new_reasons = list(reasons)

    if mode in {"capture", "hybrid"}:
        entry_floor = 0.23 if mode == "hybrid" else 0.18
        exit_floor = 0.18 if mode == "hybrid" else 0.12
        if capture_exit >= exit_floor:
            new_action, new_score = _force_action_score("SELL", new_score, threshold)
            new_reasons = new_reasons + [
                f"dividend_capture_exit_signal={capture_exit:.3f}",
                f"dividend_quality={quality_score:.3f}",
            ]
        elif capture_entry >= entry_floor:
            new_action, new_score = _force_action_score("BUY", new_score, threshold)
            new_reasons = new_reasons + [
                f"dividend_capture_entry_signal={capture_entry:.3f}",
                f"dividend_quality={quality_score:.3f}",
            ]

    if mode in {"compound", "hybrid"}:
        if mode == "compound":
            if new_action == "SELL" and quality_score >= 0.30 and payout_ratio_norm <= 0.95:
                new_action, new_score = _force_action_score("HOLD", new_score, threshold)
                new_reasons = new_reasons + ["dividend_compound_hold_override"]
            if new_action in {"HOLD", "SELL"} and compound_bias >= 0.22:
                new_action, new_score = _force_action_score("BUY", new_score, threshold)
                new_reasons = new_reasons + [
                    f"dividend_compound_buy_bias={compound_bias:.3f}",
                    f"dividend_quality={quality_score:.3f}",
                ]
        elif new_action == "SELL" and compound_bias >= 0.45 and capture_exit < 0.22:
            new_action, new_score = _force_action_score("HOLD", new_score, threshold)
            new_reasons = new_reasons + ["dividend_hybrid_hold_bias"]

    return new_action, new_score, new_reasons, out_features


def _apply_dividend_safety_tax_overlay(
    *,
    symbol: str,
    action: str,
    score: float,
    threshold: float,
    reasons: List[str],
    features: Dict[str, float],
    iter_count: int,
    state: Dict[str, Dict[str, float]],
) -> tuple[str, float, List[str], Dict[str, float]]:
    if not _is_dividend_profile():
        return action, score, reasons, {}

    mode = _dividend_strategy_mode()
    quality = _clamp01(float(features.get("dividend_quality_score_norm", 0.0) or 0.0))
    payout_ratio = _clamp01(float(features.get("dividend_payout_ratio_norm", 0.0) or 0.0))
    payout_safety = _clamp01(1.0 - (max(payout_ratio - 0.78, 0.0) / 0.22))
    vol_norm = _clamp01(float(features.get("vol_30m", 0.0) or 0.0) / 0.03)
    cal_quality = _clamp01(float(features.get("calendar_dividend_quality_signal_norm", 0.0) or 0.0))
    yield_norm = _clamp01(float(features.get("dividend_yield_norm", 0.0) or 0.0))

    yield_trap_risk = _clamp01(max(yield_norm - 0.70, 0.0) * 2.4 + max(payout_ratio - 0.85, 0.0) * 2.2)
    safety_composite = _clamp01(
        (0.36 * quality)
        + (0.24 * payout_safety)
        + (0.18 * (1.0 - vol_norm))
        + (0.14 * cal_quality)
        + (0.08 * (1.0 - yield_trap_risk))
    )

    ex_signal = max(
        _clamp01(float(features.get("calendar_dividend_exdate_proximity_norm", 0.0) or 0.0)),
        _clamp01(float(features.get("dividend_ex_date_proximity_norm", 0.0) or 0.0)),
    )
    spread_bps = max(float(features.get("spread_bps", 0.0) or 0.0), 0.0)
    ex_slippage_risk = _clamp01((0.45 * _clamp01(spread_bps / 30.0)) + (0.30 * vol_norm) + (0.25 * ex_signal))

    news_sent = float(features.get("news_sentiment", 0.0) or 0.0)
    mom = float(features.get("mom_5m", 0.0) or 0.0)
    compound_growth = _clamp01(float(features.get("dividend_compound_growth_norm", 0.0) or 0.0))
    growth_momentum = _clamp01((0.40 * _clamp01(max(news_sent, 0.0))) + (0.30 * _clamp01(max(mom, 0.0) / 0.01)) + (0.30 * compound_growth))

    row = state.setdefault(symbol.upper(), {"position_open": 0.0, "last_buy_iter": 0.0, "last_rebalance_iter": 0.0, "hold_iters": 0.0})
    position_open = float(row.get("position_open", 0.0) or 0.0)
    if str(action).upper() == "BUY":
        if position_open <= 0.0:
            row["last_buy_iter"] = float(iter_count)
            row["hold_iters"] = 0.0
        row["position_open"] = 1.0
    elif str(action).upper() == "SELL":
        row["position_open"] = 0.0
        row["hold_iters"] = 0.0
        row["last_rebalance_iter"] = float(iter_count)
    elif position_open > 0.0:
        row["hold_iters"] = float(row.get("hold_iters", 0.0) or 0.0) + 1.0

    hold_iters = float(row.get("hold_iters", 0.0) or 0.0)
    min_hold_iters = max(int(os.getenv("DIVIDEND_TAX_MIN_HOLD_ITERS", "45") or 45), 1)
    tax_qualified_hold_norm = _clamp01(hold_iters / float(min_hold_iters))

    rebalance_interval = max(int(os.getenv("DIVIDEND_REBALANCE_INTERVAL_ITERS", "14") or 14), 1)
    last_rebalance_iter = int(row.get("last_rebalance_iter", 0.0) or 0)
    iters_since_rebalance = (int(iter_count) - last_rebalance_iter) if last_rebalance_iter > 0 else 10**9
    rebalance_due_norm = _clamp01(iters_since_rebalance / float(rebalance_interval))

    new_action = str(action).upper()
    new_score = float(score)
    new_reasons = list(reasons)

    if new_action == "BUY" and safety_composite < 0.30:
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [f"dividend_safety_filter score={safety_composite:.3f}"]

    if new_action == "BUY" and ex_slippage_risk >= 0.72 and mode in {"capture", "hybrid"}:
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [f"dividend_ex_slippage_risk={ex_slippage_risk:.3f}"]

    if new_action == "SELL" and mode in {"compound", "hybrid"} and tax_qualified_hold_norm < 0.95 and safety_composite >= 0.32:
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [
            f"dividend_tax_hold_block qualified={tax_qualified_hold_norm:.3f} min_hold_iters={min_hold_iters}",
        ]

    cadence_ok = iters_since_rebalance >= rebalance_interval
    if (new_action in {"BUY", "SELL"}) and (not cadence_ok) and growth_momentum < 0.70:
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [
            f"dividend_rebalance_cadence_wait iters_since_rebalance={iters_since_rebalance} min={rebalance_interval}",
        ]

    if new_action == "HOLD" and mode in {"compound", "hybrid"} and growth_momentum >= 0.62 and safety_composite >= 0.45:
        new_action, new_score = _force_action_score("BUY", new_score, threshold)
        new_reasons = new_reasons + [
            f"dividend_growth_accumulate growth_momentum={growth_momentum:.3f}",
            f"dividend_safety={safety_composite:.3f}",
        ]

    out_features = {
        "dividend_safety_composite_norm": safety_composite,
        "dividend_ex_slippage_risk_norm": ex_slippage_risk,
        "dividend_tax_qualified_hold_norm": tax_qualified_hold_norm,
        "dividend_growth_momentum_norm": growth_momentum,
        "dividend_rebalance_due_norm": rebalance_due_norm,
    }
    return new_action, new_score, new_reasons, out_features


def _update_dividend_compound_metrics(
    *,
    symbol: str,
    symbol_return_1m: float,
    features: Dict[str, float],
    state: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    if not _is_dividend_profile():
        return {}

    mode = _dividend_strategy_mode()
    row = state.setdefault(symbol, {"equity": 1.0, "peak": 1.0, "steps": 0.0})

    quality = _clamp01(float(features.get("dividend_quality_score_norm", 0.0) or 0.0))
    enabled = mode in {"compound", "hybrid"} and quality >= 0.20

    if enabled:
        periods_per_year = max(float(os.getenv("DIVIDEND_COMPOUND_PERIODS_PER_YEAR", "98280")), 390.0)
        annual_yield = _clamp01(float(features.get("dividend_yield_norm", 0.0) or 0.0)) * 0.12
        dividend_carry = annual_yield / periods_per_year
        step_return = max(min(float(symbol_return_1m) + dividend_carry, 0.20), -0.20)

        equity = max(float(row.get("equity", 1.0) or 1.0) * (1.0 + step_return), 0.05)
        row["equity"] = equity
        row["steps"] = float(row.get("steps", 0.0) or 0.0) + 1.0
        row["peak"] = max(float(row.get("peak", 1.0) or 1.0), equity)

    equity = max(float(row.get("equity", 1.0) or 1.0), 0.05)
    peak = max(float(row.get("peak", equity) or equity), equity)
    growth = equity - 1.0
    drawdown_norm = _clamp01(1.0 - (equity / max(peak, 1e-8)))
    growth_norm = _clamp01((math.tanh(growth * 2.0) + 1.0) * 0.5)
    steps_norm = _clamp01(float(row.get("steps", 0.0) or 0.0) / 5000.0)

    return {
        "dividend_compound_equity": float(equity),
        "dividend_compound_growth": float(growth),
        "dividend_compound_drawdown_norm": drawdown_norm,
        "dividend_compound_growth_norm": growth_norm,
        "dividend_compound_steps_norm": steps_norm,
    }


def _is_long_term_profile() -> bool:
    return _shadow_profile_name() in {"long_term_dividend", "long_term_core_etf", "long_term_sector_rotation"}


def _long_term_horizon_years() -> int:
    raw = os.getenv("LONG_TERM_HORIZON_YEARS", "10").strip()
    try:
        years = int(float(raw))
    except ValueError:
        years = 10
    return max(years, 10)


def _parse_symbol_set(raw: str) -> set[str]:
    out: set[str] = set()
    for piece in (raw or "").split(","):
        sym = piece.strip().upper()
        if not sym:
            continue
        if sym == ".X":
            sym = "$SPX.X"
        elif sym.startswith(".") and len(sym) > 2:
            sym = "$" + sym
        out.add(sym)
    return out


def _bond_quote_symbol_universe() -> set[str]:
    out = _parse_symbol_set(os.getenv("BOND_SYMBOLS", "TLT,IEF,SHY,TIP,LQD,HYG"))
    for symbols in _bond_symbol_sets().values():
        out.update(symbols)
    return out


def _apply_bond_quote_quarantine(*, symbol: str, last_price: float, prev_close: float, closes: List[float]) -> tuple[float, float]:
    sym = str(symbol or "").strip().upper()
    if sym not in _bond_quote_symbol_universe():
        return float(last_price), float(prev_close)

    max_intraday_move = max(min(float(os.getenv("BOND_MARKET_SNAPSHOT_MAX_INTRADAY_MOVE", "0.12")), 0.50), 0.01)
    max_abs_price = max(float(os.getenv("BOND_MARKET_SNAPSHOT_MAX_ABS_PRICE", "300")), 1.0)
    hist_last = float(closes[-1]) if closes else 0.0

    fallback_last = 0.0
    if 0.0 < hist_last <= max_abs_price:
        fallback_last = hist_last
    elif 0.0 < prev_close <= max_abs_price:
        fallback_last = prev_close

    if last_price > max_abs_price:
        if fallback_last > 0.0:
            last_price = fallback_last
        else:
            raise RuntimeError(f"bond_quote_out_of_bounds symbol={sym} last_price={last_price:.4f}")

    if prev_close > max_abs_price:
        if fallback_last > 0.0:
            prev_close = fallback_last
        elif 0.0 < last_price <= max_abs_price:
            prev_close = last_price
        else:
            raise RuntimeError(f"bond_prev_close_out_of_bounds symbol={sym} prev_close={prev_close:.4f}")

    if prev_close > 0.0 and last_price > 0.0:
        rel_move = abs(last_price - prev_close) / max(prev_close, 1e-8)
        if rel_move > max_intraday_move:
            if fallback_last > 0.0:
                last_price = fallback_last
            else:
                raise RuntimeError(f"bond_quote_intraday_move_too_large symbol={sym} rel_move={rel_move:.4f}")

    return float(last_price), float(prev_close)


def _long_term_quality_universe(profile: str) -> set[str]:
    prof = (profile or "").strip().lower()
    if prof == "long_term_core_etf":
        return _parse_symbol_set(os.getenv("LONG_TERM_CORE_QUALITY_SYMBOLS", _LONG_TERM_CORE_QUALITY_DEFAULT))
    if prof == "long_term_sector_rotation":
        return _parse_symbol_set(os.getenv("LONG_TERM_SECTOR_QUALITY_SYMBOLS", _LONG_TERM_SECTOR_QUALITY_DEFAULT))
    if prof == "long_term_dividend":
        out = _parse_symbol_set(os.getenv("LONG_TERM_DIVIDEND_QUALITY_SYMBOLS", ""))
        out.update(_dividend_quality_symbols())
        return out
    return set()


def _long_term_buy_floors(profile: str) -> tuple[float, float]:
    prof = (profile or "").strip().lower()
    if prof == "long_term_core_etf":
        default_buy_floor = 0.56
    elif prof == "long_term_sector_rotation":
        default_buy_floor = 0.60
    elif prof == "long_term_dividend":
        default_buy_floor = 0.58
    else:
        default_buy_floor = 0.57

    raw_buy_floor = os.getenv("LONG_TERM_10Y_BUY_SCORE_MIN", "").strip()
    if raw_buy_floor:
        try:
            default_buy_floor = float(raw_buy_floor)
        except ValueError:
            pass

    buy_floor = _clamp01(default_buy_floor)

    strong_default = min(max(buy_floor + 0.14, 0.60), 0.90)
    raw_strong_floor = os.getenv("LONG_TERM_10Y_STRONG_SCORE_MIN", "").strip()
    if raw_strong_floor:
        try:
            strong_default = float(raw_strong_floor)
        except ValueError:
            pass

    strong_floor = _clamp01(max(strong_default, buy_floor))
    return buy_floor, strong_floor


def _long_term_10y_score(symbol: str, features: Dict[str, float], profile: str) -> tuple[float, Dict[str, float]]:
    universe = _long_term_quality_universe(profile)
    universe_hit = 1.0 if symbol.upper() in universe else 0.0

    vol_norm = _clamp01(abs(float(features.get("vol_30m", 0.0) or 0.0)) / 0.03)
    stability = _clamp01(1.0 - vol_norm)

    mom = float(features.get("mom_5m", 0.0) or 0.0)
    trend = _clamp01(0.5 + (mom * 12.0))

    pullback = abs(float(features.get("pct_from_close", 0.0) or 0.0))
    pullback_resilience = _clamp01(1.0 - (pullback / 0.12))

    range_pos = _clamp01(float(features.get("range_pos", 0.5) or 0.5))
    range_balance = _clamp01(1.0 - abs(range_pos - 0.5) * 1.8)

    dividend_quality = _clamp01(float(features.get("dividend_quality_score_norm", 0.0) or 0.0))
    if dividend_quality <= 0.0:
        dividend_quality = _dividend_quality_score(symbol, features)

    payout_ratio_norm = _clamp01(float(features.get("dividend_payout_ratio_norm", 0.0) or 0.0))
    payout_safety = _clamp01(1.0 - (max(payout_ratio_norm - 0.78, 0.0) / 0.22))

    compound_growth = _clamp01(float(features.get("dividend_compound_growth_norm", 0.0) or 0.0))
    compound_drawdown = _clamp01(float(features.get("dividend_compound_drawdown_norm", 0.0) or 0.0))
    compound_safety = _clamp01(1.0 - compound_drawdown)

    macro_stress = _clamp01(abs(float(features.get("ctx_VIX_X_pct_from_close", 0.0) or 0.0)) / 0.03)

    prof = (profile or "").strip().lower()
    if prof == "long_term_core_etf":
        score = (
            0.33 * stability
            + 0.23 * trend
            + 0.17 * pullback_resilience
            + 0.15 * range_balance
            + 0.12 * universe_hit
        )
    elif prof == "long_term_sector_rotation":
        score = (
            0.30 * trend
            + 0.22 * stability
            + 0.16 * pullback_resilience
            + 0.12 * range_balance
            + 0.12 * universe_hit
            + 0.08 * (1.0 - macro_stress)
        )
    elif prof == "long_term_dividend":
        score = (
            0.26 * dividend_quality
            + 0.20 * payout_safety
            + 0.16 * stability
            + 0.14 * compound_growth
            + 0.14 * compound_safety
            + 0.10 * universe_hit
        )
    else:
        score = (
            0.35 * stability
            + 0.25 * trend
            + 0.20 * pullback_resilience
            + 0.20 * universe_hit
        )

    components = {
        "long_term_quality_universe_hit": universe_hit,
        "long_term_quality_stability_norm": stability,
        "long_term_quality_trend_norm": trend,
        "long_term_quality_pullback_resilience_norm": pullback_resilience,
        "long_term_quality_range_balance_norm": range_balance,
        "long_term_quality_dividend_norm": dividend_quality,
        "long_term_quality_payout_safety_norm": payout_safety,
        "long_term_quality_compound_growth_norm": compound_growth,
        "long_term_quality_compound_safety_norm": compound_safety,
    }
    return _clamp01(score), components


def _apply_long_term_buy_hold_overlay(
    *,
    symbol: str,
    action: str,
    score: float,
    threshold: float,
    reasons: List[str],
    features: Dict[str, float],
) -> tuple[str, float, List[str], Dict[str, float]]:
    if not _is_long_term_profile():
        return action, score, reasons, {}

    profile = _shadow_profile_name()
    strict_buy_hold = _env_flag("LONG_TERM_STRICT_BUY_HOLD", "1")
    horizon_years = _long_term_horizon_years()
    ten_year_score, components = _long_term_10y_score(symbol, features, profile)
    buy_floor, strong_floor = _long_term_buy_floors(profile)

    original_action = str(action or "HOLD").upper()
    new_action = original_action
    new_score = float(score)
    new_reasons = list(reasons)

    if strict_buy_hold and new_action == "SELL":
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [
            f"long_term_buy_hold_sell_block horizon_years={horizon_years}",
            f"long_term_10y_score={ten_year_score:.3f}",
        ]

    if new_action == "BUY" and ten_year_score < buy_floor:
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [
            f"long_term_10y_buy_floor_block score={ten_year_score:.3f} min={buy_floor:.3f}",
            f"long_term_10y_horizon_years={horizon_years}",
        ]
    elif new_action == "HOLD" and ten_year_score >= strong_floor and new_score >= (float(threshold) - 0.03):
        new_action, new_score = _force_action_score("BUY", new_score, threshold)
        new_reasons = new_reasons + [
            f"long_term_10y_accumulation_signal score={ten_year_score:.3f} min={strong_floor:.3f}",
            f"long_term_10y_horizon_years={horizon_years}",
        ]

    if new_action == "BUY":
        new_reasons = new_reasons + [
            f"long_term_10y_candidate score={ten_year_score:.3f} buy_floor={buy_floor:.3f}",
            "long_term_10y_components "
            f"stability={components['long_term_quality_stability_norm']:.3f} "
            f"trend={components['long_term_quality_trend_norm']:.3f} "
            f"dividend={components['long_term_quality_dividend_norm']:.3f}",
        ]

    if new_action != original_action:
        new_reasons = new_reasons + [f"long_term_overlay_action_change {original_action}->{new_action}"]

    out_features = {
        "long_term_profile_active": 1.0,
        "long_term_strict_buy_hold": 1.0 if strict_buy_hold else 0.0,
        "long_term_horizon_years": float(horizon_years),
        "long_term_horizon_10y_plus": 1.0 if horizon_years >= 10 else 0.0,
        "long_term_10y_score_norm": ten_year_score,
        "long_term_10y_buy_floor_norm": buy_floor,
        "long_term_10y_strong_buy_floor_norm": strong_floor,
        **components,
    }

    return new_action, new_score, new_reasons, out_features


def _apply_long_term_allocation_policy(
    *,
    symbol: str,
    action: str,
    score: float,
    threshold: float,
    reasons: List[str],
    features: Dict[str, float],
    iter_count: int,
    state: Dict[str, Any],
) -> tuple[str, float, List[str], Dict[str, float]]:
    if not _is_long_term_profile():
        return action, score, reasons, {}

    dca_interval = max(int(os.getenv("LONG_TERM_DCA_INTERVAL_ITERS", "6") or 6), 1)
    strong_buffer = max(float(os.getenv("LONG_TERM_DCA_STRONG_SCORE_BUFFER", "0.08") or 0.08), 0.0)
    rebalance_band = max(float(os.getenv("LONG_TERM_REBALANCE_BAND_PCT", "0.035") or 0.035), 0.005)

    new_action = str(action).upper()
    new_score = float(score)
    new_reasons = list(reasons)

    last_buy_iter_by_symbol = state.setdefault("last_buy_iter_by_symbol", {})
    last_buy_iter = int(last_buy_iter_by_symbol.get(symbol.upper(), 0) or 0)
    iters_since_buy = int(iter_count - last_buy_iter) if last_buy_iter > 0 else 10**9

    pct_from_close = abs(float(features.get("pct_from_close", 0.0) or 0.0))
    strong_score = float(threshold) + strong_buffer

    if new_action == "BUY":
        if (dca_interval > 1) and (iters_since_buy < dca_interval) and (new_score < strong_score):
            new_action, new_score = _force_action_score("HOLD", new_score, threshold)
            new_reasons = new_reasons + [
                f"long_term_dca_cadence_wait iters_since_buy={iters_since_buy} min={dca_interval}",
            ]
        elif pct_from_close >= rebalance_band and new_score < strong_score:
            new_action, new_score = _force_action_score("HOLD", new_score, threshold)
            new_reasons = new_reasons + [
                f"long_term_rebalance_band_block pct_from_close={pct_from_close:.4f} band={rebalance_band:.4f}",
            ]
        elif pct_from_close <= (rebalance_band * 0.55) and new_score < (float(threshold) + 0.04):
            new_action, new_score = _force_action_score("HOLD", new_score, threshold)
            new_reasons = new_reasons + [
                f"long_term_rebalance_inside_band pct_from_close={pct_from_close:.4f} band={rebalance_band:.4f}",
            ]

    if new_action == "BUY":
        last_buy_iter_by_symbol[symbol.upper()] = int(iter_count)

    out_features = {
        "long_term_dca_interval_iters": float(dca_interval),
        "long_term_dca_iters_since_buy": float(max(min(iters_since_buy, 1_000_000), 0)),
        "long_term_rebalance_band_norm": _clamp01(rebalance_band / 0.10),
    }
    return new_action, new_score, new_reasons, out_features


def _ctx_pct_feature(features: Dict[str, float], symbol: str) -> float:
    key = _feature_symbol_key(symbol)
    return float(features.get(f"ctx_{key}_pct_from_close", 0.0) or 0.0)


def _session_phase_norms() -> tuple[float, float, float]:
    now_et = _now_eastern()
    now_min = (now_et.hour * 60) + now_et.minute
    open_min = (9 * 60) + 30
    close_min = 16 * 60
    if (now_min < open_min) or (now_min > close_min):
        return 0.0, 0.0, 0.0

    mins_from_open = max(now_min - open_min, 0)
    open_norm = _clamp01(1.0 - (mins_from_open / 45.0))
    midday_center = (12 * 60) + 30
    midday_norm = _clamp01(1.0 - (abs(now_min - midday_center) / 120.0))
    power_norm = _clamp01((now_min - (15 * 60)) / 60.0)
    return open_norm, midday_norm, power_norm


def _directional_action_from_value(value: float) -> str:
    if value > 0.0:
        return "BUY"
    if value < 0.0:
        return "SELL"
    return "HOLD"


def _trend_chop_regime_metrics(features: Dict[str, float]) -> tuple[float, float, float]:
    pct = float(features.get("pct_from_close", 0.0) or 0.0)
    mom = float(features.get("mom_5m", 0.0) or 0.0)
    vol = max(float(features.get("vol_30m", 0.0) or 0.0), 0.0)
    range_pos = _clamp01(float(features.get("range_pos", 0.5) or 0.5))
    spread_bps = max(float(features.get("spread_bps", 0.0) or 0.0), 0.0)

    ctx_moves = [
        _ctx_pct_feature(features, "SPY"),
        _ctx_pct_feature(features, "QQQ"),
        _ctx_pct_feature(features, "IWM"),
    ]
    ctx_used = [x for x in ctx_moves if abs(x) > 1e-8]
    benchmark_move = (_safe_mean(ctx_used) if ctx_used else 0.0)
    rel_move = pct - benchmark_move

    directional_anchor = pct if abs(pct) >= 0.0006 else mom
    aligned = 0
    compared = 0
    for probe in (mom, benchmark_move):
        if abs(probe) < 0.0003 or abs(directional_anchor) < 0.0003:
            continue
        compared += 1
        if probe * directional_anchor > 0.0:
            aligned += 1
    alignment_norm = 0.5 if compared <= 0 else _clamp01(aligned / compared)

    momentum_norm = _clamp01(abs(mom) / 0.0045)
    move_norm = _clamp01(abs(pct) / 0.012)
    bench_norm = _clamp01(abs(benchmark_move) / 0.008)
    rel_norm = _clamp01(abs(rel_move) / 0.010)
    range_extension = _clamp01(abs(range_pos - 0.5) * 2.0)
    spread_penalty = _clamp01(spread_bps / 24.0)
    vol_penalty = _clamp01(vol / 0.05)
    leadership_norm = _clamp01(0.5 + (rel_move * 35.0))

    trend_norm = _clamp01(
        (0.25 * momentum_norm)
        + (0.20 * move_norm)
        + (0.15 * range_extension)
        + (0.15 * alignment_norm)
        + (0.10 * leadership_norm)
        + (0.08 * rel_norm)
        + (0.07 * bench_norm)
        + (0.05 * (1.0 - spread_penalty))
        + (0.05 * (1.0 - vol_penalty))
    )

    mid_range = _clamp01(1.0 - (abs(range_pos - 0.5) * 2.0))
    disagreement = _clamp01(1.0 - alignment_norm)
    low_momentum = _clamp01(1.0 - momentum_norm)
    low_move = _clamp01(1.0 - move_norm)
    low_benchmark = _clamp01(1.0 - bench_norm)
    chop_norm = _clamp01(
        (0.24 * mid_range)
        + (0.22 * low_momentum)
        + (0.18 * low_move)
        + (0.14 * disagreement)
        + (0.12 * low_benchmark)
        + (0.10 * _clamp01(0.5 + spread_penalty - (0.5 * range_extension)))
    )

    if trend_norm >= 0.60:
        chop_norm = _clamp01(chop_norm * (1.0 - (0.45 * trend_norm)))
    if chop_norm >= 0.60:
        trend_norm = _clamp01(trend_norm * (1.0 - (0.35 * chop_norm)))

    return trend_norm, chop_norm, alignment_norm


def _apply_day_strategy_overlay(
    *,
    symbol: str,
    action: str,
    score: float,
    threshold: float,
    reasons: List[str],
    features: Dict[str, float],
    state: Dict[str, Any],
) -> tuple[str, float, List[str], Dict[str, float]]:
    if not _is_day_profile():
        return action, score, reasons, {}

    open_norm, midday_norm, power_norm = _session_phase_norms()
    mom = float(features.get("mom_5m", 0.0) or 0.0)
    pct = float(features.get("pct_from_close", 0.0) or 0.0)
    vol = max(float(features.get("vol_30m", 0.0) or 0.0), 0.0)
    spread_bps = max(float(features.get("spread_bps", 0.0) or 0.0), 0.0)
    bid = max(float(features.get("bid_size", 0.0) or 0.0), 0.0)
    ask = max(float(features.get("ask_size", 0.0) or 0.0), 0.0)
    regime_trend, regime_chop, regime_alignment = _trend_chop_regime_metrics(features)

    imbalance = (bid - ask) / max((bid + ask), 1e-8)
    max_spread = max(float(os.getenv("DAY_EXEC_COST_MAX_SPREAD_BPS", "20") or 20.0), 1.0)
    depth_ref = max(float(os.getenv("DAY_LIQUIDITY_DEPTH_REF", "1800") or 1800.0), 100.0)
    depth_norm = _clamp01((bid + ask) / depth_ref)

    auction_signal = _clamp01(open_norm * ((0.40 * _clamp01(abs(mom) / 0.01)) + (0.35 * _clamp01(abs(imbalance))) + (0.25 * _clamp01(abs(pct) / 0.01))))
    execution_cost_risk = _clamp01((0.70 * _clamp01(spread_bps / max_spread)) + (0.30 * _clamp01(vol / 0.05)))
    liquidity_vacuum_risk = _clamp01(max(_clamp01((spread_bps - 18.0) / 24.0), 1.0 - depth_norm))
    halt_resume_risk = _clamp01(max(execution_cost_risk, _clamp01(abs(pct) / 0.03), _clamp01(vol / 0.05)))

    halt_cooldown_s = max(int(os.getenv("DAY_HALT_RESUME_COOLDOWN_SECONDS", "180") or 180), 30)
    halt_trigger = _clamp01(float(os.getenv("DAY_HALT_TRIGGER_NORM", "0.90") or 0.90))
    halt_until_by_symbol = state.setdefault("halt_until_ts_by_symbol", {}) if isinstance(state, dict) else {}

    now_ts = time.time()
    current_halt_until = float(halt_until_by_symbol.get(symbol.upper(), 0.0) or 0.0)
    if halt_resume_risk >= halt_trigger:
        current_halt_until = max(current_halt_until, now_ts + halt_cooldown_s)
        halt_until_by_symbol[symbol.upper()] = current_halt_until

    new_action = str(action).upper()
    new_score = float(score)
    new_reasons = list(reasons)

    if new_action in {"BUY", "SELL"} and now_ts < current_halt_until:
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [f"day_halt_resume_cooldown active_s={int(current_halt_until - now_ts)}"]
    elif new_action in {"BUY", "SELL"} and execution_cost_risk >= 0.82:
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [f"day_execution_cost_guard risk={execution_cost_risk:.3f}"]
    elif new_action in {"BUY", "SELL"} and liquidity_vacuum_risk >= 0.78:
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [f"day_liquidity_vacuum_guard risk={liquidity_vacuum_risk:.3f}"]
    elif new_action in {"BUY", "SELL"} and regime_chop >= 0.68 and regime_trend <= 0.58:
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [f"day_regime_chop_guard chop={regime_chop:.3f} trend={regime_trend:.3f}"]
    elif new_action == "HOLD" and open_norm >= 0.25 and auction_signal >= 0.62:
        direct = _directional_action_from_value((0.65 * mom) + (0.35 * imbalance))
        if direct in {"BUY", "SELL"}:
            new_action, new_score = _force_action_score(direct, new_score, threshold)
            new_reasons = new_reasons + [
                f"day_opening_auction_imbalance signal={auction_signal:.3f}",
                f"day_auction_imbalance={imbalance:+.3f}",
            ]
    elif new_action in {"BUY", "SELL"} and midday_norm >= 0.72 and max(abs(mom) < 0.003, regime_chop >= 0.64):
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [f"day_midday_chop_guard mom={mom:+.4f} chop={regime_chop:.3f}"]
    elif new_action == "HOLD" and regime_trend >= 0.72 and regime_alignment >= 0.50 and execution_cost_risk < 0.62 and liquidity_vacuum_risk < 0.65:
        direct = _directional_action_from_value((0.55 * mom) + (0.30 * pct) + (0.15 * imbalance))
        if direct in {"BUY", "SELL"}:
            new_action, new_score = _force_action_score(direct, new_score, threshold)
            new_reasons = new_reasons + [f"day_regime_trend_bias trend={regime_trend:.3f} align={regime_alignment:.3f}"]
    elif new_action == "HOLD" and power_norm >= 0.55 and abs(mom) >= 0.003 and execution_cost_risk < 0.65 and regime_trend >= 0.58:
        direct = _directional_action_from_value(mom if regime_alignment >= 0.45 else pct)
        if direct in {"BUY", "SELL"}:
            new_action, new_score = _force_action_score(direct, new_score, threshold)
            new_reasons = new_reasons + [f"day_power_hour_trend mom={mom:+.4f} trend={regime_trend:.3f}"]

    out_features = {
        "day_opening_auction_signal_norm": auction_signal,
        "day_halt_resume_risk_norm": halt_resume_risk,
        "day_liquidity_vacuum_risk_norm": liquidity_vacuum_risk,
        "day_execution_cost_risk_norm": execution_cost_risk,
        "day_session_open_norm": open_norm,
        "day_session_midday_norm": midday_norm,
        "day_session_power_hour_norm": power_norm,
        "day_regime_trend_norm": regime_trend,
        "day_regime_chop_norm": regime_chop,
        "day_regime_alignment_norm": regime_alignment,
    }
    return new_action, new_score, new_reasons, out_features


def _apply_swing_strategy_overlay(
    *,
    symbol: str,
    action: str,
    score: float,
    threshold: float,
    reasons: List[str],
    features: Dict[str, float],
    state: Dict[str, Any],
) -> tuple[str, float, List[str], Dict[str, float]]:
    if not _is_swing_profile():
        return action, score, reasons, {}

    pct = float(features.get("pct_from_close", 0.0) or 0.0)
    mom = float(features.get("mom_5m", 0.0) or 0.0)
    vol = max(float(features.get("vol_30m", 0.0) or 0.0), 0.0)
    news_sent = float(features.get("news_sentiment", 0.0) or 0.0)
    news_shock = _clamp01(float(features.get("news_shock_rate", 0.0) or 0.0))
    event_prox = _clamp01(float(features.get("calendar_event_proximity_norm", 0.0) or 0.0))
    high_impact = _clamp01(float(features.get("calendar_high_impact_24h_norm", 0.0) or 0.0))
    regime_trend, regime_chop, regime_alignment = _trend_chop_regime_metrics(features)

    drift_strength = max(abs(news_sent), abs(mom), abs(pct))
    post_earnings_drift = _clamp01(((0.55 * max(news_shock, event_prox)) + (0.45 * high_impact)) * _clamp01(drift_strength / 0.02))

    gap_strength = _clamp01(abs(pct) / 0.02)
    continuation_norm = _clamp01(gap_strength if (pct * mom) > 0 else 0.0)
    gap_fade_norm = _clamp01(gap_strength if (pct * mom) < 0 else 0.0)

    squeeze_max_vol = max(float(os.getenv("SWING_SQUEEZE_VOL_MAX", "0.012") or 0.012), 0.003)
    squeeze_tightness = _clamp01((squeeze_max_vol - vol) / squeeze_max_vol)
    squeeze_breakout = _clamp01(squeeze_tightness * _clamp01(abs(mom) / 0.007))

    bench = _ctx_pct_feature(features, "SPY")
    if abs(bench) <= 1e-8:
        bench = _ctx_pct_feature(features, "VOO")
    rel = pct - bench
    sector_rs = _clamp01(0.5 + (rel * 25.0))

    ema_map = state.setdefault("weekly_trend_ema_by_symbol", {}) if isinstance(state, dict) else {}
    prev_ema = float(ema_map.get(symbol.upper(), 0.0) or 0.0)
    alpha = _clamp01(float(os.getenv("SWING_WEEKLY_TREND_ALPHA", "0.18") or 0.18))
    trend_input = (0.60 * mom) + (0.40 * pct)
    ema = ((1.0 - alpha) * prev_ema) + (alpha * trend_input)
    ema_map[symbol.upper()] = ema
    weekly_confirm = _clamp01(0.5 + (ema * 22.0))

    new_action = str(action).upper()
    new_score = float(score)
    new_reasons = list(reasons)

    if new_action == "HOLD" and regime_trend >= 0.72 and regime_alignment >= 0.52 and weekly_confirm >= 0.48 and sector_rs >= 0.48:
        direct = _directional_action_from_value((0.45 * pct) + (0.35 * mom) + (0.20 * rel))
        if direct in {"BUY", "SELL"}:
            new_action, new_score = _force_action_score(direct, new_score, threshold)
            new_reasons = new_reasons + [f"swing_regime_trend_bias trend={regime_trend:.3f} align={regime_alignment:.3f}"]
    elif new_action == "HOLD" and post_earnings_drift >= 0.60 and abs(news_sent) >= 0.15 and weekly_confirm >= 0.45 and regime_alignment >= 0.45:
        direct = _directional_action_from_value(news_sent)
        if direct in {"BUY", "SELL"}:
            new_action, new_score = _force_action_score(direct, new_score, threshold)
            new_reasons = new_reasons + [f"swing_post_earnings_drift={post_earnings_drift:.3f}"]
    elif new_action == "HOLD" and continuation_norm >= 0.68 and weekly_confirm >= 0.50 and regime_trend >= 0.58:
        direct = _directional_action_from_value(pct)
        if direct in {"BUY", "SELL"}:
            new_action, new_score = _force_action_score(direct, new_score, threshold)
            new_reasons = new_reasons + [f"swing_gap_continuation={continuation_norm:.3f}"]
    elif new_action == "HOLD" and squeeze_breakout >= 0.66 and weekly_confirm >= 0.52 and regime_chop <= 0.62:
        direct = _directional_action_from_value(mom if abs(mom) >= 0.001 else pct)
        if direct in {"BUY", "SELL"}:
            new_action, new_score = _force_action_score(direct, new_score, threshold)
            new_reasons = new_reasons + [f"swing_squeeze_breakout={squeeze_breakout:.3f}"]

    if new_action in {"BUY", "SELL"} and regime_chop >= 0.70 and continuation_norm < 0.65 and squeeze_breakout < 0.60:
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [f"swing_regime_chop_guard chop={regime_chop:.3f} trend={regime_trend:.3f}"]
    if new_action == "BUY" and (gap_fade_norm >= 0.74) and (weekly_confirm <= 0.45):
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [f"swing_gap_fade_risk={gap_fade_norm:.3f}"]
    if new_action == "BUY" and sector_rs < 0.40:
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [f"swing_sector_rs_weak={sector_rs:.3f}"]
    if new_action == "SELL" and sector_rs > 0.62:
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [f"swing_sector_rs_headwind={sector_rs:.3f}"]

    out_features = {
        "swing_post_earnings_drift_norm": post_earnings_drift,
        "swing_gap_continuation_norm": continuation_norm,
        "swing_gap_fade_norm": gap_fade_norm,
        "swing_vol_compression_breakout_norm": squeeze_breakout,
        "swing_sector_relative_strength_norm": sector_rs,
        "swing_weekly_trend_confirm_norm": weekly_confirm,
        "swing_regime_trend_norm": regime_trend,
        "swing_regime_chop_norm": regime_chop,
        "swing_regime_alignment_norm": regime_alignment,
    }
    return new_action, new_score, new_reasons, out_features


def _bond_symbol_sets() -> Dict[str, set[str]]:
    return {
        "long_duration": _parse_symbol_set(os.getenv("BOND_LONG_DURATION_SYMBOLS", "TLT,TLH,VGLT,EDV,ZROZ,IEF")),
        "short_duration": _parse_symbol_set(os.getenv("BOND_SHORT_DURATION_SYMBOLS", "SHY,FLOT,VGSH,SCHO")),
        "ig_credit": _parse_symbol_set(os.getenv("BOND_IG_CREDIT_SYMBOLS", "LQD,VCIT,IGIB")),
        "hy_credit": _parse_symbol_set(os.getenv("BOND_HY_CREDIT_SYMBOLS", "HYG,JNK,USHY")),
        "inflation": _parse_symbol_set(os.getenv("BOND_INFLATION_SYMBOLS", "TIP,VTIP,SCHP")),
    }


def _apply_bond_strategy_overlay(
    *,
    symbol: str,
    action: str,
    score: float,
    threshold: float,
    reasons: List[str],
    features: Dict[str, float],
) -> tuple[str, float, List[str], Dict[str, float]]:
    if not _is_bond_profile():
        return action, score, reasons, {}

    sym = symbol.upper()
    sets = _bond_symbol_sets()

    curve_proxy = _ctx_pct_feature(features, "TLT") - _ctx_pct_feature(features, "SHY")
    steepener = _clamp01(0.5 + (curve_proxy * 35.0))
    flattener = _clamp01(0.5 - (curve_proxy * 35.0))

    credit_proxy = _ctx_pct_feature(features, "HYG") - _ctx_pct_feature(features, "LQD")
    credit_risk_on = _clamp01(0.5 + (credit_proxy * 40.0))
    credit_risk_off = _clamp01(1.0 - credit_risk_on)

    inflation_proxy = _ctx_pct_feature(features, "TIP") - _ctx_pct_feature(features, "IEF")
    inflation_breakeven = _clamp01(0.5 + (inflation_proxy * 40.0))

    yield_norm = _clamp01(float(features.get("dividend_yield_norm", 0.0) or 0.0))
    vol_norm = _clamp01(float(features.get("vol_30m", 0.0) or 0.0) / 0.03)
    range_balance = _clamp01(1.0 - abs(float(features.get("range_pos", 0.5) or 0.5) - 0.5) * 1.8)
    carry_roll = _clamp01((0.50 * yield_norm) + (0.30 * (1.0 - vol_norm)) + (0.20 * range_balance))

    if sym in sets["long_duration"]:
        duration_regime = flattener
    elif sym in sets["short_duration"]:
        duration_regime = steepener
    else:
        duration_regime = 0.5

    new_action = str(action).upper()
    new_score = float(score)
    new_reasons = list(reasons)

    if new_action == "BUY" and carry_roll < 0.20:
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [f"bond_carry_roll_filter={carry_roll:.3f}"]

    if sym in sets["long_duration"] and steepener >= 0.66:
        if new_action == "BUY":
            new_action, new_score = _force_action_score("HOLD", new_score, threshold)
            new_reasons = new_reasons + [f"bond_duration_steepener_headwind={steepener:.3f}"]
        elif new_action == "HOLD":
            new_action, new_score = _force_action_score("SELL", new_score, threshold)
            new_reasons = new_reasons + [f"bond_duration_trim_signal={steepener:.3f}"]
    elif sym in sets["long_duration"] and flattener >= 0.66 and new_action == "HOLD":
        new_action, new_score = _force_action_score("BUY", new_score, threshold)
        new_reasons = new_reasons + [f"bond_duration_accumulate_signal={flattener:.3f}"]

    if sym in sets["short_duration"] and flattener >= 0.66 and new_action == "BUY":
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [f"bond_short_duration_headwind={flattener:.3f}"]

    if sym in sets["hy_credit"] and credit_risk_off >= 0.62 and new_action == "BUY":
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [f"bond_credit_risk_off={credit_risk_off:.3f}"]

    if sym in sets["ig_credit"] and credit_risk_on >= 0.62 and new_action == "SELL":
        new_action, new_score = _force_action_score("HOLD", new_score, threshold)
        new_reasons = new_reasons + [f"bond_ig_credit_support={credit_risk_on:.3f}"]

    if sym in sets["inflation"]:
        if inflation_breakeven >= 0.62 and new_action == "HOLD":
            new_action, new_score = _force_action_score("BUY", new_score, threshold)
            new_reasons = new_reasons + [f"bond_inflation_breakeven_buy={inflation_breakeven:.3f}"]
        elif inflation_breakeven <= 0.35 and new_action == "BUY":
            new_action, new_score = _force_action_score("HOLD", new_score, threshold)
            new_reasons = new_reasons + [f"bond_inflation_breakeven_guard={inflation_breakeven:.3f}"]

    out_features = {
        "bond_duration_regime_norm": duration_regime,
        "bond_curve_steepener_norm": steepener,
        "bond_curve_flattener_norm": flattener,
        "bond_carry_roll_norm": carry_roll,
        "bond_credit_risk_on_norm": credit_risk_on,
        "bond_credit_risk_off_norm": credit_risk_off,
        "bond_inflation_breakeven_norm": inflation_breakeven,
    }
    return new_action, new_score, new_reasons, out_features


def _behavior_lane_feature_preview(
    *,
    symbol: str,
    features: Dict[str, float],
    day_state: Optional[Dict[str, Any]] = None,
    swing_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    if _is_day_profile():
        _, _, _, out = _apply_day_strategy_overlay(
            symbol=symbol,
            action="HOLD",
            score=0.5,
            threshold=0.55,
            reasons=[],
            features=features,
            state={},
        )
        return {key: float(out.get(key, 0.0) or 0.0) for key in _BEHAVIOR_LANE_FEATURE_NAMES}

    if _is_swing_profile():
        preview_state = copy.deepcopy(swing_state) if isinstance(swing_state, dict) else {}
        _, _, _, out = _apply_swing_strategy_overlay(
            symbol=symbol,
            action="HOLD",
            score=0.5,
            threshold=0.55,
            reasons=[],
            features=features,
            state=preview_state,
        )
        return {key: float(out.get(key, 0.0) or 0.0) for key in _BEHAVIOR_LANE_FEATURE_NAMES}

    if _is_bond_profile():
        _, _, _, out = _apply_bond_strategy_overlay(
            symbol=symbol,
            action="HOLD",
            score=0.5,
            threshold=0.55,
            reasons=[],
            features=features,
        )
        return {key: float(out.get(key, 0.0) or 0.0) for key in _BEHAVIOR_LANE_FEATURE_NAMES}

    return {}


def _governance_path(project_root: str, broker: Optional[str] = None) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return governance_master_control_path(project_root, _path_context(broker=broker), day=day)


def _event_bus_path(project_root: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return runtime_event_legacy_path(project_root, day=day)


def _critical_alert_events_path(project_root: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(project_root, "governance", "alerts", f"critical_events_{day}.jsonl")


def _critical_alert_latest_path(project_root: str, broker: str) -> str:
    ctx = _path_context(broker=broker)
    return os.path.join(project_root, "governance", "alerts", f"critical_latest_{ctx.key}.json")


_CRITICAL_ALERT_LAST_TS: Dict[str, float] = {}


def _emit_critical_alert(
    *,
    project_root: str,
    broker: str,
    event: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    severity: str = "critical",
) -> Dict[str, Any]:
    sev = str(severity or "critical").strip().lower()
    if sev not in {"info", "warn", "critical"}:
        sev = "critical"

    suppress_seconds = max(int(os.getenv("CRITICAL_ALERT_SUPPRESS_SECONDS", "300") or 300), 0)
    key = f"{sev}:{event}:{message}"[:320]
    now_ts = time.time()
    prev_ts = float(_CRITICAL_ALERT_LAST_TS.get(key, 0.0) or 0.0)
    suppressed = (suppress_seconds > 0) and ((now_ts - prev_ts) < suppress_seconds)
    if suppressed:
        return {"sent": False, "suppressed": True, "key": key}

    _CRITICAL_ALERT_LAST_TS[key] = now_ts
    row: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "severity": sev,
        "event": str(event),
        "message": str(message),
        "broker": broker,
        "profile": _shadow_profile_name() or "default",
        "domain": _shadow_domain_name(broker=broker),
    }
    if details:
        row["details"] = dict(details)

    _append_jsonl(_critical_alert_events_path(project_root), row)
    safe_write_json_atomic(
        _critical_alert_latest_path(project_root, broker),
        row,
        project_root=project_root,
        source="run_shadow_training_loop.critical_alert",
    )
    _append_jsonl(
        _event_bus_path(project_root),
        {
            "timestamp_utc": row["timestamp_utc"],
            "event": "critical_alert",
            "severity": sev,
            "broker": broker,
            "profile": row["profile"],
            "domain": row["domain"],
            "alert_event": str(event),
            "message": str(message),
            "details": dict(details or {}),
        },
    )

    if _env_flag("CRITICAL_ALERT_ROUTE_PAGER", "0"):
        router = Path(project_root) / "scripts" / "pager_alert_router.py"
        if router.exists():
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(router),
                        "--severity",
                        sev,
                        "--event",
                        str(event),
                        "--message",
                        str(message),
                        "--suppress-seconds",
                        str(max(suppress_seconds, 30)),
                    ],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5,
                )
            except Exception:
                pass

    return {"sent": True, "suppressed": False, "key": key}


def _broker_truth_latest_path(project_root: str, broker: str) -> str:
    ctx = _path_context(broker=broker)
    return os.path.join(project_root, "governance", "health", f"broker_truth_{ctx.key}_latest.json")


def _order_idempotency_registry_path(project_root: str, broker: str) -> str:
    ctx = _path_context(broker=broker)
    return os.path.join(project_root, "governance", "health", f"order_idempotency_{ctx.key}_latest.json")


def _collect_numeric_key_values(payload: Any, out: Dict[str, List[float]]) -> None:
    if isinstance(payload, dict):
        for k, v in payload.items():
            key = str(k or "").strip().lower()
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out.setdefault(key, []).append(float(v))
            _collect_numeric_key_values(v, out)
    elif isinstance(payload, list):
        for row in payload:
            _collect_numeric_key_values(row, out)


def _pick_metric_candidate(numeric: Dict[str, List[float]], tokens: List[str]) -> float:
    vals: List[float] = []
    for key, rows in numeric.items():
        if any(tok in key for tok in tokens):
            vals.extend([float(v) for v in rows if math.isfinite(float(v))])
    positives = [v for v in vals if v > 0.0]
    if positives:
        return float(max(positives))
    if vals:
        return float(max(vals, key=lambda x: abs(x)))
    return 0.0


def _extract_account_metrics(payload: Any) -> Dict[str, float]:
    numeric: Dict[str, List[float]] = {}
    _collect_numeric_key_values(payload, numeric)

    equity = _pick_metric_candidate(numeric, ["liquidationvalue", "netliquidation", "equity", "accountvalue"])
    cash_balance = _pick_metric_candidate(numeric, ["cashbalance", "cashavailable", "moneymarketfund"])
    buying_power = _pick_metric_candidate(numeric, ["buyingpower", "daytradingbuyingpower"])
    available_funds = _pick_metric_candidate(numeric, ["availablefunds", "cashavailablefortrading", "availabletrading"])
    initial_margin = _pick_metric_candidate(numeric, ["initialmargin", "initialrequirement"])
    maintenance_margin = _pick_metric_candidate(numeric, ["maintenancemargin", "maintenancerequirement"])

    return {
        "equity": float(max(equity, 0.0)),
        "cash_balance": float(max(cash_balance, 0.0)),
        "buying_power": float(max(buying_power, 0.0)),
        "available_funds": float(max(available_funds, 0.0)),
        "initial_margin_requirement": float(max(initial_margin, 0.0)),
        "maintenance_margin_requirement": float(max(maintenance_margin, 0.0)),
    }


def _manual_position_map(manual_payload: Dict[str, Any]) -> Dict[str, float]:
    symbols = manual_payload.get("symbols") if isinstance(manual_payload, dict) else {}
    if not isinstance(symbols, dict):
        return {}

    out: Dict[str, float] = {}
    for sym, row in symbols.items():
        if not isinstance(row, dict):
            continue
        sym_key = str(sym or "").strip().upper()
        if not sym_key:
            continue
        qty = _to_float(row.get("position_qty", row.get("shares", 0.0)), 0.0)
        if not math.isfinite(qty):
            continue
        out[sym_key] = float(qty)
    return out


def _broker_margin_available_proxy(broker_truth: Dict[str, Any], lane: str) -> float:
    metrics = broker_truth.get("account_metrics") if isinstance(broker_truth, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}

    available = max(
        _to_float(metrics.get("available_funds"), 0.0),
        _to_float(metrics.get("buying_power"), 0.0),
        _to_float(metrics.get("cash_balance"), 0.0),
    )
    equity = _to_float(metrics.get("equity"), 0.0)
    if available <= 0.0 and equity > 0.0:
        lane_key = (lane or "").strip().lower()
        fallback_pct = 0.35 if lane_key == "options" else 0.30
        available = equity * fallback_pct

    caps = _parse_lane_float_caps(
        os.getenv(
            "DERIVATIVES_MARGIN_HEADROOM_PCT",
            "options:0.70,futures:0.65,default:0.70",
        )
    )
    lane_key = (lane or "").strip().lower()
    headroom_pct = float(caps.get(lane_key, caps.get("default", 0.70)))
    headroom_pct = min(max(headroom_pct, 0.05), 1.0)
    return float(max(available, 0.0) * headroom_pct)


def _default_capital_flow_state() -> Dict[str, Any]:
    return {
        "detected": False,
        "estimated_amount": 0.0,
        "ratio_to_equity": 0.0,
        "cash_balance_delta": 0.0,
        "equity_delta": 0.0,
        "agreement_norm": 0.0,
        **_default_capital_flow_features(),
    }


def _estimate_capital_flow_state(
    account_metrics: Dict[str, float],
    previous_metrics: Dict[str, float],
) -> Dict[str, Any]:
    out = _default_capital_flow_state()
    if not isinstance(account_metrics, dict) or not isinstance(previous_metrics, dict):
        return out

    cash_now = _to_float(account_metrics.get("cash_balance"), 0.0)
    cash_prev = _to_float(previous_metrics.get("cash_balance"), 0.0)
    equity_now = max(_to_float(account_metrics.get("equity"), 0.0), cash_now)
    equity_prev = max(_to_float(previous_metrics.get("equity"), 0.0), cash_prev)

    if cash_now <= 0.0 or cash_prev <= 0.0 or equity_now <= 0.0 or equity_prev <= 0.0:
        return out

    cash_delta = float(cash_now - cash_prev)
    equity_delta = float(equity_now - equity_prev)
    out["cash_balance_delta"] = cash_delta
    out["equity_delta"] = equity_delta

    if abs(cash_delta) <= 1e-9 or abs(equity_delta) <= 1e-9 or (cash_delta * equity_delta) < 0.0:
        return out

    dominant_delta = max(abs(cash_delta), abs(equity_delta), 1.0)
    agreement_norm = max(
        0.0,
        min(1.0, 1.0 - (abs(abs(cash_delta) - abs(equity_delta)) / dominant_delta)),
    )
    out["agreement_norm"] = float(agreement_norm)
    if agreement_norm <= 0.35:
        return out

    estimated_amount = math.copysign(min(abs(cash_delta), abs(equity_delta)) * agreement_norm, cash_delta)
    baseline_equity = max(equity_now, equity_prev, 1.0)

    min_abs = max(float(os.getenv("CAPITAL_FLOW_EVENT_MIN_ABS", "500") or 500.0), 0.0)
    min_ratio = max(float(os.getenv("CAPITAL_FLOW_EVENT_MIN_RATIO", "0.0025") or 0.0025), 0.0)
    if abs(estimated_amount) < max(min_abs, baseline_equity * min_ratio):
        return out

    scale_ratio = max(float(os.getenv("CAPITAL_FLOW_FEATURE_SCALE_RATIO", "0.05") or 0.05), 1e-4)
    ratio_to_equity = abs(estimated_amount) / baseline_equity

    out.update(
        {
            "detected": True,
            "estimated_amount": float(estimated_amount),
            "ratio_to_equity": float(ratio_to_equity),
            "capital_flow_signed_scaled": float(math.tanh((estimated_amount / baseline_equity) / scale_ratio)),
            "capital_flow_inflow_norm": float(min(max(max(estimated_amount, 0.0) / baseline_equity / scale_ratio, 0.0), 1.0)),
            "capital_flow_outflow_norm": float(min(max(max(-estimated_amount, 0.0) / baseline_equity / scale_ratio, 0.0), 1.0)),
        }
    )
    return out


def _effective_account_equity_proxy(
    broker_truth: Dict[str, Any],
    fallback_equity_proxy: float,
) -> Tuple[float, Dict[str, Any]]:
    metrics = broker_truth.get("account_metrics") if isinstance(broker_truth, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}

    age_iters = max(int(_to_float(broker_truth.get("age_iters"), 0.0) or 0), 0)
    max_age_iters = max(int(float(os.getenv("BROKER_TRUTH_EQUITY_MAX_AGE_ITERS", "18") or 18)), 0)
    status = str(broker_truth.get("status", "") or "").strip().lower()
    live_equity = max(
        _to_float(metrics.get("equity"), 0.0),
        _to_float(metrics.get("cash_balance"), 0.0),
    )

    use_live = (
        live_equity > 0.0
        and age_iters <= max_age_iters
        and status not in {"error", "disabled", "pending"}
    )
    effective_equity = float(live_equity if use_live else max(float(fallback_equity_proxy), 0.0))
    return effective_equity, {
        "source": "broker_truth_account_metrics" if use_live else "account_equity_proxy",
        "broker_equity": float(live_equity),
        "fallback_equity_proxy": float(max(float(fallback_equity_proxy), 0.0)),
        "age_iters": int(age_iters),
        "max_age_iters": int(max_age_iters),
        "status": status,
    }


def _estimate_options_margin_proxy(decision: Dict[str, Any]) -> float:
    plan = decision.get("plan") if isinstance(decision, dict) else {}
    if not isinstance(plan, dict):
        return 0.0

    style = str(plan.get("options_style", "NONE") or "NONE").upper()
    if style == "NONE":
        return 0.0

    px = max(_to_float(plan.get("underlying_price"), 0.0), 0.0)
    contracts = max(int(_to_float(plan.get("contracts"), 0.0) or 0), 0)
    if contracts <= 0:
        return 0.0

    notional = px * 100.0 * float(contracts)
    family = str(plan.get("strategy_family", "") or "").strip().lower()

    family_mult = {
        "hedge": 0.08,
        "income": 0.14,
        "wheel": 0.55,
        "credit_spread": 0.22,
        "debit_spread": 0.12,
        "event_vol": 0.24,
        "synthetic_directional": 0.32,
        "neutral_income": 0.28,
        "directional": 0.20,
        "calendar": 0.16,
        "diagonal_income": 0.18,
    }
    margin = notional * float(family_mult.get(family, 0.20))

    strike = _to_float(plan.get("strike"), 0.0)
    if style == "WHEEL_CASH_SECURED_PUT" and strike > 0.0:
        margin = strike * 100.0 * float(contracts)
    elif style in {"COVERED_CALL", "WHEEL_COVERED_CALL", "PROTECTIVE_COLLAR"}:
        margin = max(notional * 0.05, 0.0)

    legs = plan.get("legs") if isinstance(plan.get("legs"), list) else []
    strikes: List[float] = []
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        s = _to_float(leg.get("strike"), 0.0)
        if s > 0.0:
            strikes.append(s)
    if len(strikes) >= 2:
        width = max(max(strikes) - min(strikes), 0.0)
        defined_risk = width * 100.0 * float(contracts)
        if family in {"credit_spread", "debit_spread", "neutral_income", "calendar", "diagonal_income"}:
            margin = min(max(margin, 0.0), max(defined_risk, 0.0))

    return float(max(margin, 0.0))


def _estimate_futures_margin_proxy(decision: Dict[str, Any], features: Dict[str, float]) -> float:
    plan = decision.get("plan") if isinstance(decision, dict) else {}
    if not isinstance(plan, dict):
        return 0.0

    style = str(plan.get("futures_style", "NONE") or "NONE").upper()
    if style == "NONE":
        return 0.0

    contracts = max(int(_to_float(plan.get("contracts"), 0.0) or 0), 0)
    if contracts <= 0:
        return 0.0

    px = max(_to_float(features.get("last_price"), _to_float(plan.get("underlying_price"), 0.0)), 0.0)
    multiplier = max(_to_float(os.getenv("FUTURES_CONTRACT_MULTIPLIER_DEFAULT", "50"), 50.0), 1.0)
    margin_rate = max(_to_float(os.getenv("FUTURES_INITIAL_MARGIN_RATE", "0.12"), 0.12), 0.02)

    margin = float(contracts) * px * multiplier * margin_rate
    if style in {"FUTURES_BASIS_CARRY_CALENDAR", "FUTURES_TERM_STRUCTURE_ROLL_ROTATION"}:
        margin *= 0.55

    action = str(decision.get("action", "HOLD") or "HOLD").upper()
    if action == "ROLL":
        margin *= 0.65

    return float(max(margin, 0.0))


def _apply_derivatives_margin_guard(
    *,
    decision: Dict[str, Any],
    lane: str,
    broker_truth: Dict[str, Any],
    features: Dict[str, float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out = {
        "action": str(decision.get("action", "HOLD") or "HOLD").upper(),
        "score": float(decision.get("score", 0.5) or 0.5),
        "threshold": float(decision.get("threshold", 0.58) or 0.58),
        "reasons": list(decision.get("reasons", []) or []),
        "plan": dict(decision.get("plan", {}) or {}),
    }

    lane_key = (lane or "").strip().lower()
    trade_actions = {
        "options": {"BUY_TO_OPEN", "SELL_TO_OPEN", "ROLL"},
        "futures": {"BUY", "SELL", "ROLL"},
    }

    action = str(out.get("action", "HOLD") or "HOLD").upper()
    if action not in trade_actions.get(lane_key, set()):
        return out, {
            "active": True,
            "lane": lane_key,
            "ok": True,
            "reason": "not_trade_action",
            "required_margin_proxy": 0.0,
            "available_margin_proxy": _broker_margin_available_proxy(broker_truth, lane_key),
            "action": action,
        }

    if lane_key == "options":
        required = _estimate_options_margin_proxy(out)
    elif lane_key == "futures":
        required = _estimate_futures_margin_proxy(out, features)
    else:
        required = 0.0

    available = _broker_margin_available_proxy(broker_truth, lane_key)
    cushion = max(_to_float(os.getenv("DERIVATIVES_MARGIN_GUARD_MIN_CUSHION", "1.05"), 1.05), 1.0)
    ok = (required <= 0.0) or ((required * cushion) <= max(available, 0.0))

    meta = {
        "active": True,
        "lane": lane_key,
        "ok": bool(ok),
        "required_margin_proxy": float(required),
        "available_margin_proxy": float(available),
        "margin_cushion": float(cushion),
        "action": action,
        "reason": "ok" if ok else "margin_headroom_insufficient",
    }

    if ok:
        return out, meta

    score = float(out.get("score", 0.5) or 0.5)
    hold_score = min(max(0.5 + 0.35 * (score - 0.5), 0.01), 0.99)
    out["action"] = "HOLD"
    out["score"] = hold_score
    out["reasons"] = list(out.get("reasons", []) or []) + [
        (
            f"{lane_key}_margin_guard_block required={required:.2f} "
            f"available={available:.2f} cushion={cushion:.2f}"
        )
    ]
    plan = dict(out.get("plan", {}) or {})
    plan["margin_guard"] = {
        "blocked": True,
        "required_margin_proxy": float(required),
        "available_margin_proxy": float(available),
        "margin_cushion": float(cushion),
    }
    out["plan"] = plan
    return out, meta


def _load_order_idempotency_registry(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"entries": {}, "updated_utc": ""}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        if isinstance(obj, dict):
            entries = obj.get("entries")
            if not isinstance(entries, dict):
                obj["entries"] = {}
            return obj
    except Exception:
        pass
    return {"entries": {}, "updated_utc": ""}


def _save_order_idempotency_registry(path: str, payload: Dict[str, Any], project_root: str) -> None:
    out = dict(payload or {})
    out["updated_utc"] = datetime.now(timezone.utc).isoformat()
    safe_write_json_atomic(
        path,
        out,
        project_root=project_root,
        source="run_shadow_training_loop.order_idempotency",
    )


def _prune_order_idempotency_registry(
    state: Dict[str, Any],
    *,
    now_ts: float,
    ttl_seconds: float,
    max_entries: int,
) -> int:
    entries = state.get("entries")
    if not isinstance(entries, dict):
        state["entries"] = {}
        return 0

    removed = 0
    cutoff = max(now_ts - max(ttl_seconds, 1.0), 0.0)
    for key in list(entries.keys()):
        row = entries.get(key)
        ts = _to_float((row or {}).get("ts"), 0.0) if isinstance(row, dict) else 0.0
        if ts <= 0.0 or ts < cutoff:
            entries.pop(key, None)
            removed += 1

    if len(entries) > max_entries:
        ordered = sorted(
            entries.items(),
            key=lambda kv: _to_float((kv[1] or {}).get("ts"), 0.0) if isinstance(kv[1], dict) else 0.0,
            reverse=True,
        )
        keep = ordered[:max_entries]
        state["entries"] = {k: v for k, v in keep}
        removed += max(len(ordered) - len(keep), 0)

    return int(removed)


def _order_intent_key(
    *,
    symbol: str,
    action: str,
    lane: str,
    qty: float,
    price: float,
    strategy: str,
) -> str:
    raw = "|".join(
        [
            str(symbol or "").strip().upper(),
            str(action or "").strip().upper(),
            str(lane or "").strip().lower(),
            f"{float(qty):.6f}",
            f"{float(price):.6f}",
            str(strategy or "").strip().lower(),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def _reserve_order_intent(
    *,
    state: Dict[str, Any],
    key: str,
    now_ts: float,
    window_seconds: float,
    payload: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    entries = state.get("entries")
    if not isinstance(entries, dict):
        entries = {}
        state["entries"] = entries

    prior = entries.get(key)
    if isinstance(prior, dict):
        age = max(now_ts - _to_float(prior.get("ts"), 0.0), 0.0)
        if age <= max(window_seconds, 1.0):
            return False, {
                "ok": False,
                "duplicate": True,
                "reserved": False,
                "key": key,
                "age_seconds": float(age),
                "window_seconds": float(window_seconds),
                "prior": dict(prior),
            }

    row = dict(payload or {})
    row["ts"] = float(now_ts)
    row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    entries[key] = row
    return True, {
        "ok": True,
        "duplicate": False,
        "reserved": True,
        "key": key,
        "age_seconds": 0.0,
        "window_seconds": float(window_seconds),
    }


def _release_order_intent(state: Dict[str, Any], key: str) -> bool:
    entries = state.get("entries")
    if not isinstance(entries, dict):
        return False
    if key in entries:
        entries.pop(key, None)
        return True
    return False


def _fetch_broker_truth_snapshot(
    *,
    trader: BaseTrader,
    broker: str,
    simulate: bool,
    iter_count: int,
    manual_payload: Dict[str, Any],
    manual_tolerance: float,
    previous_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    out: Dict[str, Any] = {
        "timestamp_utc": now_iso,
        "iter": int(iter_count),
        "active": True,
        "broker": str(broker),
        "ok": True,
        "status": "ok",
        "source": "none",
        "position_count": 0,
        "open_orders_total": 0,
        "positions": {},
        "manual_positions": {},
        "mismatch_count": 0,
        "mismatch_examples": [],
        "account_metrics": {},
        "capital_flow": _default_capital_flow_state(),
    }

    manual_positions = _manual_position_map(manual_payload)
    out["manual_positions"] = dict(sorted(manual_positions.items())[:40])

    if simulate or broker != "schwab" or (trader.client is None):
        out["source"] = "manual_overrides_only"
        out["positions"] = dict(sorted(manual_positions.items())[:40])
        out["position_count"] = len(manual_positions)
        out["status"] = "manual_only"
        return out

    try:
        fetched = trader._live_fetch_accounts_payload()
    except Exception as exc:
        fetched = {"ok": False, "error": f"{type(exc).__name__}:{exc}"}

    if not bool(fetched.get("ok", False)):
        out["ok"] = False
        out["status"] = "error"
        out["source"] = "schwab_accounts_snapshot"
        out["error"] = str(fetched.get("error", "accounts_fetch_failed"))
        return out

    payload = fetched.get("payload")
    positions_rows = trader._extract_all_positions_from_payload(payload)
    open_order_ids = trader._extract_open_order_ids_from_payload(payload)
    account_metrics = _extract_account_metrics(payload)
    previous_metrics = (
        previous_state.get("account_metrics")
        if isinstance(previous_state, dict) and isinstance(previous_state.get("account_metrics"), dict)
        else {}
    )
    capital_flow = _estimate_capital_flow_state(account_metrics, previous_metrics)

    positions_map: Dict[str, float] = {}
    for row in positions_rows:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol", "")).strip().upper()
        if not sym:
            continue
        qty = _to_float(row.get("quantity"), 0.0)
        if not math.isfinite(qty) or abs(qty) <= 0.0:
            continue
        positions_map[sym] = float(qty)

    mismatch_examples: List[Dict[str, Any]] = []
    tolerance = max(float(manual_tolerance), 0.0)
    for sym, manual_qty in manual_positions.items():
        broker_qty = float(positions_map.get(sym, 0.0) or 0.0)
        delta = broker_qty - float(manual_qty)
        if abs(delta) > tolerance:
            mismatch_examples.append(
                {
                    "symbol": sym,
                    "broker_qty": float(broker_qty),
                    "manual_qty": float(manual_qty),
                    "delta_qty": float(delta),
                    "tolerance": float(tolerance),
                }
            )

    sorted_positions = sorted(positions_map.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
    out.update(
        {
            "source": "schwab_accounts_snapshot",
            "position_count": int(len(positions_map)),
            "open_orders_total": int(len(open_order_ids)),
            "positions": {k: float(v) for k, v in sorted_positions[:60]},
            "account_metrics": dict(account_metrics),
            "capital_flow": dict(capital_flow),
            "mismatch_count": int(len(mismatch_examples)),
            "mismatch_examples": mismatch_examples[:20],
            "ok": len(mismatch_examples) == 0,
            "status": "ok" if len(mismatch_examples) == 0 else "mismatch",
        }
    )
    return out


def _api_call_log_path(project_root: str, broker: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    ctx = _path_context(broker=broker)
    return os.path.join(project_root, "governance", "events", f"api_calls_{ctx.key}_{day}.jsonl")


def _loop_state_log_path(project_root: str, broker: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    ctx = _path_context(broker=broker)
    return os.path.join(project_root, "governance", "events", f"loop_state_{ctx.key}_{day}.jsonl")


def _log_api_call(
    *,
    project_root: str,
    broker: str,
    symbol: str,
    endpoint: str,
    status: str,
    latency_ms: float,
    error: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    row: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "broker": broker,
        "profile": _shadow_profile_name() or "default",
        "domain": _shadow_domain_name(broker=broker),
        "symbol": symbol,
        "endpoint": endpoint,
        "status": status,
        "latency_ms": round(float(latency_ms), 3),
    }
    if error:
        row["error"] = error
    if extra:
        row.update(extra)

    _append_jsonl(_api_call_log_path(project_root, broker), row)
    if status not in {"ok", "cache_hit"}:
        _append_jsonl(
            _event_bus_path(project_root),
            {
                "timestamp_utc": row["timestamp_utc"],
                "event": "api_call_issue",
                "broker": broker,
                "profile": row["profile"],
                "domain": row["domain"],
                "symbol": symbol,
                "endpoint": endpoint,
                "status": status,
                "latency_ms": row["latency_ms"],
                "error": error,
            },
        )


def _emit_loop_state(
    *,
    project_root: str,
    broker: str,
    prev_state: str,
    new_state: str,
    iter_count: int,
    reason: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    row: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "broker": broker,
        "profile": _shadow_profile_name() or "default",
        "domain": _shadow_domain_name(broker=broker),
        "iter": int(iter_count),
        "prev_state": prev_state,
        "state": new_state,
        "reason": reason,
    }
    if extra:
        row["details"] = extra
    _append_jsonl(_loop_state_log_path(project_root, broker), row)
    _append_jsonl(
        _event_bus_path(project_root),
        {
            "timestamp_utc": row["timestamp_utc"],
            "event": "loop_state_transition",
            "broker": broker,
            "profile": row["profile"],
            "domain": row["domain"],
            "iter": int(iter_count),
            "prev_state": prev_state,
            "state": new_state,
            "reason": reason,
            "details": extra or {},
        },
    )



def _gate_log_path(project_root: str, broker: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    ctx = _path_context(broker=broker)
    return os.path.join(project_root, "governance", "events", f"gate_logs_{ctx.key}_{day}.jsonl")


def _ingress_log_path(project_root: str, broker: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    ctx = _path_context(broker=broker)
    return os.path.join(project_root, "governance", "events", f"data_ingress_{ctx.key}_{day}.jsonl")


def _ingress_state_path(project_root: str, broker: str) -> str:
    return ingress_state_path(project_root, _path_context(broker=broker))


def _log_gate_event(
    *,
    project_root: str,
    broker: str,
    iter_count: int,
    symbol: str,
    gate: str,
    passed: bool,
    reason: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    row: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "broker": broker,
        "profile": _shadow_profile_name() or "default",
        "domain": _shadow_domain_name(broker=broker),
        "iter": int(iter_count),
        "symbol": symbol,
        "gate": gate,
        "passed": bool(passed),
        "reason": reason,
    }
    if details:
        row["details"] = details

    _append_jsonl(_gate_log_path(project_root, broker), row)
    if not bool(passed):
        _append_jsonl(
            _event_bus_path(project_root),
            {
                "timestamp_utc": row["timestamp_utc"],
                "event": "gate_blocked",
                "broker": broker,
                "profile": row["profile"],
                "domain": row["domain"],
                "iter": int(iter_count),
                "symbol": symbol,
                "gate": gate,
                "reason": reason,
                "details": details or {},
            },
        )


def _log_ingress_event(
    *,
    project_root: str,
    broker: str,
    iter_count: int,
    symbol: str,
    source: str,
    status: str,
    endpoint: str,
    latency_ms: float,
    reason: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    row: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "broker": broker,
        "profile": _shadow_profile_name() or "default",
        "domain": _shadow_domain_name(broker=broker),
        "iter": int(iter_count),
        "symbol": symbol,
        "source": source,
        "status": status,
        "endpoint": endpoint,
        "latency_ms": round(float(latency_ms), 3),
        "reason": reason,
    }
    if details:
        row["details"] = details

    _append_jsonl(_ingress_log_path(project_root, broker), row)
    if status != "ok":
        _append_jsonl(
            _event_bus_path(project_root),
            {
                "timestamp_utc": row["timestamp_utc"],
                "event": "data_ingress_issue",
                "broker": broker,
                "profile": row["profile"],
                "domain": row["domain"],
                "iter": int(iter_count),
                "symbol": symbol,
                "source": source,
                "status": status,
                "endpoint": endpoint,
                "reason": reason,
                "details": details or {},
            },
        )


def _write_ingress_state(*, project_root: str, broker: str, payload: Dict[str, Any]) -> None:
    path = _ingress_state_path(project_root, broker)
    ok = safe_write_json_atomic(
        path,
        payload,
        project_root=project_root,
        source="run_shadow_training_loop.ingress_state",
        indent=2,
        marker=True,
    )
    if ok:
        return

    if not _route_storage_or_fail():
        print(f"[StorageRoute] ingress_state_write_failed path={path}")
        return

    retry_ok = safe_write_json_atomic(
        path,
        payload,
        project_root=project_root,
        source="run_shadow_training_loop.ingress_state",
        indent=2,
        marker=True,
    )
    if not retry_ok:
        print(f"[StorageRoute] ingress_state_write_retry_failed path={path}")


def _parse_symbols(value: str) -> List[str]:
    raw_symbols = [s.strip().upper() for s in value.split(",") if s.strip()]

    symbols: List[str] = []
    for s in raw_symbols:
        # Shells can expand "$SPX.X" into ".X" if unquoted.
        if s == ".X":
            s = "$SPX.X"
        elif s.startswith(".") and len(s) > 2:
            s = "$" + s
        symbols.append(s)

    deduped: List[str] = []
    seen = set()
    for s in symbols:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)
    if not deduped:
        raise ValueError("No symbols provided")
    return deduped


def _parse_symbols_optional(value: str) -> List[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    try:
        return _parse_symbols(raw)
    except ValueError:
        return []


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _normalize_runtime_symbol(broker: str, symbol: str) -> str:
    raw = str(symbol or "").strip()
    if not raw:
        return ""
    if str(broker or "").strip().lower() == "coinbase":
        return CoinbaseMarketDataClient.normalize_symbol(raw)
    return raw.upper()


def _coinbase_runtime_prefix(profile: str) -> str:
    return "COINBASE_FUTURES" if str(profile or "").strip().lower() == "crypto_futures" else "COINBASE"


def _schwab_runtime_prefix(profile: str) -> str:
    return "SCHWAB_FUTURES" if str(profile or "").strip().lower() == "schwab_futures" else "SCHWAB"


def _symbol_policy_for_runtime(broker: str, profile: str, symbols: List[str]) -> Dict[str, Any]:
    broker_name = str(broker or "").strip().lower()
    if broker_name == "coinbase":
        prefix = _coinbase_runtime_prefix(profile)
    else:
        prefix = _schwab_runtime_prefix(profile)

    allowed = {
        _normalize_runtime_symbol(broker_name, symbol)
        for symbol in symbols
        if str(symbol or "").strip()
    }
    fast_symbols = [
        _normalize_runtime_symbol(broker_name, symbol)
        for symbol in _parse_symbols_optional(os.getenv(f"{prefix}_WATCH_SYMBOLS_FAST", ""))
        if _normalize_runtime_symbol(broker_name, symbol) in allowed
    ]
    slow_symbols = [
        _normalize_runtime_symbol(broker_name, symbol)
        for symbol in _parse_symbols_optional(os.getenv(f"{prefix}_WATCH_SYMBOLS_SLOW", ""))
        if _normalize_runtime_symbol(broker_name, symbol) in allowed
    ]
    fast_set = set(fast_symbols)
    slow_symbols = [symbol for symbol in slow_symbols if symbol not in fast_set]

    websocket_symbols: List[str] = []
    if broker_name == "coinbase":
        websocket_symbols = _parse_symbols_optional(os.getenv(f"{prefix}_WEBSOCKET_SYMBOLS", ""))
        websocket_symbols = [
            _normalize_runtime_symbol(broker_name, symbol)
            for symbol in websocket_symbols
            if _normalize_runtime_symbol(broker_name, symbol) in allowed
        ]
        if not websocket_symbols:
            websocket_symbols = list(fast_symbols)

    return {
        "fast_symbols": fast_symbols,
        "slow_symbols": slow_symbols,
        "websocket_symbols": websocket_symbols,
        "core_every_n": max(_env_int(f"{prefix}_CORE_EVERY_N_ITERS", _env_int("CORE_SYMBOL_EVERY_N_ITERS", 1)), 1),
        "fast_every_n": max(_env_int(f"{prefix}_FAST_EVERY_N_ITERS", 1), 1),
        "slow_every_n": max(_env_int(f"{prefix}_SLOW_EVERY_N_ITERS", 2), 1),
    }


def _coinbase_symbol_policy(profile: str, symbols: List[str]) -> Dict[str, Any]:
    return _symbol_policy_for_runtime("coinbase", profile, symbols)


def _schwab_symbol_policy(profile: str, symbols: List[str]) -> Dict[str, Any]:
    return _symbol_policy_for_runtime("schwab", profile, symbols)


def _feature_symbol_key(symbol: str) -> str:
    key = symbol.upper().replace("$", "").replace(".", "_")
    return key


def _parse_symbol_budgets(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not raw:
        return out
    for part in raw.split(','):
        seg = part.strip()
        if not seg or ':' not in seg:
            continue
        k, v = seg.split(':', 1)
        try:
            out[k.strip().upper()] = max(float(v.strip()), 0.0)
        except Exception:
            continue
    return out


def _queue_priority(action: str, score: float) -> int:
    a = (action or 'HOLD').upper()
    base = 0 if a == 'HOLD' else (2 if a == 'BUY' else 1)
    return int(base * 100 + max(float(score), 0.0) * 100)


def _merge_symbol_groups(*groups: str) -> List[str]:
    merged: List[str] = []
    seen = set()
    for group in groups:
        if not group:
            continue
        for sym in _parse_symbols(group):
            if sym in seen:
                continue
            seen.add(sym)
            merged.append(sym)

    if not merged:
        raise ValueError("No symbols provided across groups")
    return merged


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    shift = (weekday - d.weekday()) % 7
    d = d + timedelta(days=shift + 7 * (n - 1))
    return d


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def _easter_sunday(year: int) -> date:
    # Anonymous Gregorian algorithm.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    d = date(year, month, day)
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nyse_holidays(year: int) -> set[date]:
    holidays = {
        _observed_fixed_holiday(year, 1, 1),
        _nth_weekday_of_month(year, 1, 0, 3),   # MLK Day
        _nth_weekday_of_month(year, 2, 0, 3),   # Presidents Day
        _easter_sunday(year) - timedelta(days=2),  # Good Friday
        _last_weekday_of_month(year, 5, 0),      # Memorial Day
        _observed_fixed_holiday(year, 6, 19),    # Juneteenth
        _observed_fixed_holiday(year, 7, 4),     # Independence Day
        _nth_weekday_of_month(year, 9, 0, 1),    # Labor Day
        _nth_weekday_of_month(year, 11, 3, 4),   # Thanksgiving
        _observed_fixed_holiday(year, 12, 25),   # Christmas
    }
    return holidays


def _in_market_window(now_et: datetime, start_hour_et: int, end_hour_et: int) -> tuple[bool, str]:
    if now_et.weekday() >= 5:
        return False, "weekend"

    if now_et.date() in _nyse_holidays(now_et.year):
        return False, "holiday"

    hhmm = now_et.hour + (now_et.minute / 60.0)
    if hhmm < float(start_hour_et):
        return False, "pre_window"
    if hhmm >= float(end_hour_et):
        return False, "post_window"
    return True, "open"


def _auto_retrain_log_path(project_root: str, broker: Optional[str] = None) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(project_root, "governance", _shadow_profile_subdir(broker=broker), f"auto_retrain_events_{day}.jsonl")


def _heartbeat_path(project_root: str, broker: str) -> str:
    profile = _shadow_profile_name() or "default"
    domain = _shadow_domain_name(broker=broker)
    pid = os.getpid()
    name = f"shadow_loop_{profile}_{domain}_{broker}_{pid}.json"
    return os.path.join(project_root, "governance", "health", name)


def _write_heartbeat(
    *,
    project_root: str,
    broker: str,
    iter_count: int,
    symbols_total: int,
    context_total: int,
    state: str,
) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "broker": broker,
        "profile": _shadow_profile_name() or "default",
        "domain": _shadow_domain_name(broker=broker),
        "run_id": str(os.getenv("CORRELATION_RUN_ID", "") or "").strip(),
        "iter_id": str(os.getenv("CORRELATION_ITER_ID", "") or "").strip(),
        "iter": int(iter_count),
        "symbols_total": int(symbols_total),
        "context_total": int(context_total),
        "state": state,
        "log_schema_version": max(int(os.getenv("LOG_SCHEMA_VERSION", "2")), 1),
    }
    path = _heartbeat_path(project_root, broker)
    safe_write_json_atomic(
        path,
        payload,
        project_root=project_root,
        source="run_shadow_training_loop.heartbeat",
        indent=2,
        marker=True,
    )


def _count_underperformers(recommendations: List[Dict[str, Any]]) -> int:
    return sum(1 for r in recommendations if r.get("recommended_action") in {"RETRAIN", "REMOVE_OR_REBUILD"})


def _parse_size_to_gb(value: str) -> float:
    try:
        s = value.strip().upper()
        if s.endswith("G"):
            return float(s[:-1])
        if s.endswith("M"):
            return float(s[:-1]) / 1024.0
        if s.endswith("K"):
            return float(s[:-1]) / (1024.0 * 1024.0)
        return float(s)
    except Exception:
        return 0.0


def _memory_guard_snapshot() -> Dict[str, float]:
    snapshot: Dict[str, float] = {}

    try:
        proc = subprocess.run(["/usr/bin/memory_pressure", "-Q"], capture_output=True, text=True, check=False)
        out = proc.stdout or ""
        for raw in out.splitlines():
            line = raw.strip()
            lower = line.lower()
            if "free percentage" in lower:
                # Example: System-wide memory free percentage: 22%
                rhs = line.split(":", 1)[-1].strip().replace("%", "")
                snapshot["free_pct"] = float(rhs)
            elif "available percentage" in lower:
                rhs = line.split(":", 1)[-1].strip().replace("%", "")
                snapshot["available_pct"] = float(rhs)
    except Exception:
        pass

    try:
        proc = subprocess.run(["/usr/sbin/sysctl", "vm.swapusage"], capture_output=True, text=True, check=False)
        out = (proc.stdout or "").strip()
        # Example: vm.swapusage: total = 2048.00M  used = 58.50M  free = 1989.50M  (encrypted)
        if "used =" in out:
            used_part = out.split("used =", 1)[1].strip().split()[0]
            snapshot["swap_used_gb"] = _parse_size_to_gb(used_part)
    except Exception:
        pass

    return snapshot


def _mlx_retrain_lock_busy(project_root: str) -> tuple[bool, str]:
    lock_path = os.getenv("MLX_RETRAIN_LOCK_PATH", os.path.join(project_root, "governance", "mlx_retrain.lock"))
    try:
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                return False, lock_path
            except BlockingIOError:
                return True, lock_path
    except Exception:
        return False, lock_path


def _auto_retrain_memory_ok(min_free_pct: float, max_swap_gb: float) -> tuple[bool, Dict[str, float], str]:
    snap = _memory_guard_snapshot()

    free_pct = snap.get("free_pct")
    if free_pct is not None and free_pct < min_free_pct:
        return False, snap, f"free_pct_below_threshold free_pct={free_pct:.1f} min_free_pct={min_free_pct:.1f}"

    swap_used_gb = snap.get("swap_used_gb")
    if swap_used_gb is not None and swap_used_gb > max_swap_gb:
        return False, snap, f"swap_above_threshold swap_used_gb={swap_used_gb:.2f} max_swap_gb={max_swap_gb:.2f}"

    return True, snap, "ok"


def _spawn_auto_retrain(
    *,
    project_root: str,
    underperformers: int,
    sample_recommendations: List[Dict[str, Any]],
    broker: Optional[str] = None,
) -> subprocess.Popen:
    venv_py = os.path.join(project_root, ".venv312", "bin", "python")
    retrain_script = os.path.join(project_root, "scripts", "weekly_retrain.py")
    cmd = [venv_py, retrain_script, "--continue-on-error"]

    proc = subprocess.Popen(cmd, cwd=project_root)
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": "retrain_started",
        "pid": proc.pid,
        "command": cmd,
        "underperformer_count": underperformers,
        "sample_recommendations": sample_recommendations[:5],
    }
    _append_jsonl(_auto_retrain_log_path(project_root, broker=broker), event)
    print(f"[AutoRetrain] started pid={proc.pid} underperformers={underperformers}")
    return proc


def run_loop(
    symbols: List[str],
    context_symbols: List[str],
    broker: str,
    interval_seconds: int,
    max_iterations: int,
    simulate: bool,
    auto_retrain: bool,
    retrain_cooldown_minutes: int,
    retrain_min_underperformers: int,
    retrain_on_simulate: bool,
    session_start_hour: int,
    session_end_hour: int,
    bad_symbol_fail_limit: int,
    bad_symbol_retry_minutes: int,
    volatile_symbols: List[str],
    defensive_symbols: List[str],
) -> None:
    _enforce_data_only_lock()

    if _global_trading_halt_enabled():
        raise RuntimeError("GLOBAL_TRADING_HALT=1 set; refusing to start shadow loop")

    broker = (broker or "schwab").strip().lower()
    if broker not in {"schwab", "coinbase"}:
        raise RuntimeError(f"Unsupported broker: {broker}")

    os.environ["SHADOW_DOMAIN"] = _shadow_domain_name(broker=broker)

    run_seed = (
        f"{broker}|{_shadow_profile_name() or 'default'}|{_shadow_domain_name(broker=broker)}|"
        f"{os.getpid()}|{int(time.time() * 1000)}"
    )
    run_id = hashlib.sha1(run_seed.encode("utf-8")).hexdigest()[:16]
    os.environ["CORRELATION_RUN_ID"] = run_id
    os.environ["CORRELATION_ITER_ID"] = ""

    api_key = os.getenv("SCHWAB_API_KEY", "YOUR_KEY_HERE")
    secret = os.getenv("SCHWAB_SECRET", "YOUR_SECRET_HERE")
    redirect = os.getenv("SCHWAB_REDIRECT", "https://127.0.0.1:8080")
    covered_call_shares = int(os.getenv("COVERED_CALL_SHARES", "0"))

    if broker == "schwab" and (not simulate) and (api_key == "YOUR_KEY_HERE" or secret == "YOUR_SECRET_HERE"):
        raise RuntimeError("Real credentials are required unless --simulate is used")

    registry_path = os.path.join(PROJECT_ROOT, "master_bot_registry.json")
    registry, bots = _fresh_registry(registry_path)
    print(f"Loaded registry with {len(bots)} sub-bots")
    print(
        f"Shadow profile={_shadow_profile_name() or 'default'} domain={_shadow_domain_name(broker=broker)} "
        f"threshold_shift={_threshold_shift():+.3f}"
    )

    behavior_bias_enabled = os.getenv("ENABLE_TRADE_BEHAVIOR_BIAS", "1").strip() == "1"
    behavior_bias_strength = float(os.getenv("TRADE_BEHAVIOR_BIAS_STRENGTH", "0.08"))
    behavior_hold_neutral_min = float(os.getenv("TRADE_BEHAVIOR_HOLD_NEUTRAL_MIN", "0.62"))
    behavior_hold_margin_min = float(os.getenv("TRADE_BEHAVIOR_HOLD_MARGIN_MIN", "0.08"))
    behavior_model = _load_trade_behavior_model(PROJECT_ROOT) if behavior_bias_enabled else None
    if behavior_bias_enabled:
        if behavior_model is not None:
            print(
                f"Trade behavior bias enabled model={behavior_model.get('path')} "
                f"strength={behavior_bias_strength:.3f} "
                f"hold_neutral_min={behavior_hold_neutral_min:.3f} "
                f"hold_margin_min={behavior_hold_margin_min:.3f}"
            )
        else:
            print("Trade behavior bias enabled but no model found; running without behavior prior.")

    trader = BaseTrader(api_key, secret, redirect, mode="shadow")
    trader.token_path = os.path.join(PROJECT_ROOT, "token.json")

    client = None
    if not simulate:
        if broker == "schwab":
            auth_started = time.perf_counter()
            try:
                client = trader.authenticate()
                latency_ms = (time.perf_counter() - auth_started) * 1000.0
                _log_api_call(
                    project_root=PROJECT_ROOT,
                    broker=broker,
                    symbol="*",
                    endpoint="schwab.authenticate",
                    status="ok",
                    latency_ms=latency_ms,
                )
                _append_jsonl(
                    _event_bus_path(PROJECT_ROOT),
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "event": "api_auth_success",
                        "broker": broker,
                        "latency_ms": round(latency_ms, 3),
                    },
                )
                print("Shadow loop connected to Schwab market data.")
            except Exception as exc:
                latency_ms = (time.perf_counter() - auth_started) * 1000.0
                err = f"{type(exc).__name__}:{exc}"
                _log_api_call(
                    project_root=PROJECT_ROOT,
                    broker=broker,
                    symbol="*",
                    endpoint="schwab.authenticate",
                    status="error",
                    latency_ms=latency_ms,
                    error=err,
                )
                _append_jsonl(
                    _event_bus_path(PROJECT_ROOT),
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "event": "api_auth_error",
                        "broker": broker,
                        "latency_ms": round(latency_ms, 3),
                        "error": err,
                    },
                )
                raise RuntimeError(f"Schwab authentication failed: {err}") from exc
        else:
            client = CoinbaseMarketDataClient(timeout_seconds=float(os.getenv("COINBASE_TIMEOUT_SECONDS", "8")))
            client.set_live_symbols(list(dict.fromkeys(symbols + context_symbols)))
            _append_jsonl(
                _event_bus_path(PROJECT_ROOT),
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "event": "api_client_initialized",
                    "broker": broker,
                    "timeout_seconds": float(os.getenv("COINBASE_TIMEOUT_SECONDS", "8")),
                },
            )
            print("Shadow loop connected to Coinbase public market data.")
    else:
        print("Running in simulate mode (no external API calls).")

    effective_interval_seconds = max(int(interval_seconds), 5)
    if effective_interval_seconds != interval_seconds:
        print(f"Adjusted interval to safe minimum: {effective_interval_seconds}s")

    adaptive_interval_enabled = os.getenv('ADAPTIVE_INTERVAL_ENABLED', '1').strip() == '1'
    adaptive_interval_min = max(int(os.getenv('ADAPTIVE_INTERVAL_MIN_SECONDS', str(effective_interval_seconds))), 5)
    adaptive_interval_max = max(int(os.getenv('ADAPTIVE_INTERVAL_MAX_SECONDS', '30')), adaptive_interval_min)
    adaptive_interval_step_up = max(int(os.getenv('ADAPTIVE_INTERVAL_STEP_UP_SECONDS', '2')), 1)
    adaptive_interval_step_down = max(int(os.getenv('ADAPTIVE_INTERVAL_STEP_DOWN_SECONDS', '1')), 1)
    adaptive_interval_recover_streak = max(int(os.getenv('ADAPTIVE_INTERVAL_RECOVER_STREAK', '3')), 1)
    memory_auto_throttle_enabled = os.getenv('MEMORY_AUTO_THROTTLE_ENABLED', '1').strip() == '1'
    memory_throttle_min_free_pct = float(os.getenv('MEMORY_THROTTLE_MIN_FREE_PCT', '12'))
    memory_throttle_max_swap_gb = float(os.getenv('MEMORY_THROTTLE_MAX_SWAP_GB', '2.8'))
    memory_throttle_check_every_iters = max(int(os.getenv('MEMORY_THROTTLE_CHECK_EVERY_ITERS', '2')), 1)
    memory_throttle_step_up_seconds = max(int(os.getenv('MEMORY_THROTTLE_STEP_UP_SECONDS', '5')), 1)
    memory_throttle_recover_streak_needed = max(int(os.getenv('MEMORY_THROTTLE_RECOVER_STREAK', '3')), 1)
    memory_throttle_recover_streak = 0
    memory_throttle_active = False
    latest_memory_snapshot: Dict[str, float] = {}

    current_interval_seconds = effective_interval_seconds
    overload_streak = 0
    healthy_streak = 0

    iter_count = 0
    all_symbols = []
    seen_symbols = set()
    for s in symbols + context_symbols:
        if s in seen_symbols:
            continue
        seen_symbols.add(s)
        all_symbols.append(s)
    sim_prices = {s: 500.0 + (10.0 * i) for i, s in enumerate(all_symbols)}
    retrain_proc: Optional[subprocess.Popen] = None
    last_retrain_started_at = 0.0
    last_closed_reason: Optional[str] = None
    symbol_fail_counts: Dict[str, int] = {}
    symbol_quarantine_until: Dict[str, float] = {}
    symbol_last_price: Dict[str, float] = {}
    symbol_stale_counts: Dict[str, int] = {}
    symbol_regime_marker: Dict[str, tuple[int, int, int]] = {}
    symbol_regime_cooldown_until_iter: Dict[str, int] = {}
    execution_lag_by_symbol: Dict[str, Dict[str, float]] = {}
    dividend_compound_state: Dict[str, Dict[str, float]] = {}
    dividend_policy_state: Dict[str, Dict[str, float]] = {}
    day_strategy_state: Dict[str, Any] = {"halt_until_ts_by_symbol": {}}
    swing_strategy_state: Dict[str, Any] = {"weekly_trend_ema_by_symbol": {}}
    anomaly_fail_streak = 0
    anomaly_kill_switch_until_ts = 0.0
    blackout_windows = _event_blackout_windows()
    regime_cooldown_iters = max(int(os.getenv("REGIME_COOLDOWN_ITERS", "3")), 0)
    exposure_cap_long = max(int(os.getenv("CROSS_SYMBOL_MAX_LONG", "8")), 1)
    exposure_cap_short = max(int(os.getenv("CROSS_SYMBOL_MAX_SHORT", "8")), 1)
    anomaly_kill_threshold = max(int(os.getenv("ANOMALY_KILL_THRESHOLD", "10")), 1)
    anomaly_kill_cooldown_seconds = max(int(os.getenv("ANOMALY_KILL_COOLDOWN_SECONDS", "300")), 30)

    # Runtime layers: cache, circuit breaker, telemetry, checkpoint, backpressure, canary rollout.
    state_cache = StateCache(default_ttl_seconds=float(os.getenv("RUNTIME_CACHE_TTL_SECONDS", "2.0")))
    circuit_breaker = CircuitBreaker(
        fail_limit=int(os.getenv("RUNTIME_CIRCUIT_FAIL_LIMIT", "5")),
        cooldown_seconds=int(os.getenv("RUNTIME_CIRCUIT_COOLDOWN_SECONDS", "120")),
    )
    backpressure = BackpressureController(overload_ratio=float(os.getenv("RUNTIME_OVERLOAD_RATIO", "1.5")))
    telemetry = TelemetryEmitter(os.path.join(PROJECT_ROOT, "governance", _shadow_profile_subdir(broker=broker), "runtime_telemetry.jsonl"))
    checkpoint = CheckpointStore(os.path.join(PROJECT_ROOT, "governance", _shadow_profile_subdir(broker=broker), "runtime_checkpoint.json"))
    canary_rollout = CanaryRollout(max_weight=float(os.getenv("CANARY_MAX_WEIGHT", "0.08")))

    checkpoint_every = max(int(os.getenv("CHECKPOINT_EVERY_ITERS", "5")), 1)
    telemetry_every = max(int(os.getenv("TELEMETRY_EVERY_ITERS", "5")), 1)
    resume_from_checkpoint = os.getenv("RESUME_FROM_CHECKPOINT", "1").strip() == "1"
    skip_options_on_backpressure = os.getenv("SKIP_OPTIONS_ON_BACKPRESSURE", "1").strip() == "1"
    skip_futures_on_backpressure = os.getenv("SKIP_FUTURES_ON_BACKPRESSURE", "1").strip() == "1"
    enable_async_pipeline = os.getenv("ENABLE_ASYNC_PIPELINE", "1").strip() == "1" and (not simulate)
    async_workers = max(int(os.getenv("ASYNC_PIPELINE_WORKERS", "6")), 2)

    feature_cache_enabled = os.getenv("FEATURE_WINDOW_CACHE_ENABLED", "1").strip() == "1"
    feature_cache: Dict[str, Tuple[int, Dict[str, float]]] = {}
    slow_bot_every_n_iters = max(int(os.getenv("SLOW_BOT_EVERY_N_ITERS", "2")), 1)
    slow_bot_rows_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    news_context_enabled = (
        broker == "schwab"
        and (not simulate)
        and os.getenv("SCHWAB_NEWS_CONTEXT_ENABLED", "1").strip() == "1"
    )
    news_cache_ttl_seconds = max(float(os.getenv("SCHWAB_NEWS_CACHE_TTL_SECONDS", "180")), 15.0)
    news_lookback_hours = max(float(os.getenv("SCHWAB_NEWS_LOOKBACK_HOURS", "24")), 1.0)
    news_max_items = max(int(os.getenv("SCHWAB_NEWS_MAX_ITEMS", "20")), 3)
    news_cache: Dict[str, Tuple[float, Dict[str, float]]] = {}
    news_method_state: Dict[str, Any] = {"disabled": False, "warned": False, "method": ""}

    options_chain_enabled = (
        broker == "schwab"
        and (not simulate)
        and os.getenv("SCHWAB_OPTIONS_CHAIN_ENABLED", "1").strip() == "1"
    )
    options_chain_cache_ttl_seconds = max(float(os.getenv("SCHWAB_OPTIONS_CHAIN_CACHE_TTL_SECONDS", "150")), 20.0)
    options_chain_strike_count = max(int(os.getenv("SCHWAB_OPTIONS_CHAIN_STRIKE_COUNT", "30")), 8)
    options_chain_cache: Dict[str, Tuple[float, Dict[str, float]]] = {}
    options_chain_method_state: Dict[str, Any] = {"disabled": False, "warned": False, "method": ""}

    calendar_context_enabled = (
        broker == "schwab"
        and (not simulate)
        and os.getenv("SCHWAB_CALENDAR_CONTEXT_ENABLED", "1").strip() == "1"
    )
    calendar_cache_ttl_seconds = max(float(os.getenv("SCHWAB_CALENDAR_CACHE_TTL_SECONDS", "300")), 30.0)
    calendar_days_ahead = max(int(os.getenv("SCHWAB_CALENDAR_DAYS_AHEAD", "7")), 1)
    calendar_cache: Dict[str, Any] = {"ts": 0.0, "features": default_calendar_features()}
    calendar_method_state: Dict[str, Any] = {"disabled": False, "warned": False, "method": "", "source": ""}
    external_context_cache_ttl_seconds = max(float(os.getenv("EXTERNAL_CONTEXT_CACHE_TTL_SECONDS", "90")), 15.0)
    external_context_cache: Dict[str, Dict[str, Any]] = {
        "market_breadth": {"ts": 0.0, "payload": {}},
        "bond_reference": {"ts": 0.0, "payload": {}},
    }

    te_calendar_enabled = (
        calendar_context_enabled
        and os.getenv("TRADINGECONOMICS_CALENDAR_ENABLED", "1").strip() == "1"
    )
    te_calendar_country = os.getenv("TRADINGECONOMICS_CALENDAR_COUNTRY", "United States").strip()
    te_calendar_importance = os.getenv("TRADINGECONOMICS_CALENDAR_IMPORTANCE", "").strip()
    te_calendar_timeout_seconds = max(float(os.getenv("TRADINGECONOMICS_CALENDAR_TIMEOUT_SECONDS", "8")), 1.0)
    te_calendar_max_items = max(
        int(os.getenv("TRADINGECONOMICS_CALENDAR_MAX_ITEMS", os.getenv("SCHWAB_CALENDAR_MAX_ITEMS", "600"))),
        60,
    )
    te_api_key = os.getenv("TRADINGECONOMICS_API_KEY", "").strip()
    te_api_secret = os.getenv("TRADINGECONOMICS_API_SECRET", "").strip()
    te_auth_token = os.getenv("TRADINGECONOMICS_AUTH", "").strip()
    if not te_auth_token:
        if te_api_key and te_api_secret:
            te_auth_token = f"{te_api_key}:{te_api_secret}"
        elif te_api_key:
            te_auth_token = te_api_key
        else:
            te_auth_token = "guest:guest"

    # Keep retrain-critical logs while reducing high-volume layer noise by default.
    log_sub_bot_decisions = os.getenv("LOG_SUB_BOT_DECISIONS", "0").strip() == "1"
    log_master_variant_decisions = os.getenv("LOG_MASTER_VARIANT_DECISIONS", "0").strip() == "1"
    log_grand_master_decisions = os.getenv("LOG_GRAND_MASTER_DECISIONS", "1").strip() == "1"
    log_options_master_decisions = os.getenv("LOG_OPTIONS_MASTER_DECISIONS", "1").strip() == "1"
    log_futures_master_decisions = os.getenv("LOG_FUTURES_MASTER_DECISIONS", "1").strip() == "1"

    bot_cooldown_enabled = os.getenv("BOT_COOLDOWN_ENABLED", "1").strip() == "1"
    bot_cooldown_acc_floor = float(os.getenv("BOT_COOLDOWN_ACC_FLOOR", "0.53"))
    bot_cooldown_min_iters = max(int(os.getenv("BOT_COOLDOWN_MIN_ITERS", "2")), 1)
    bot_cooldown_max_iters = max(int(os.getenv("BOT_COOLDOWN_MAX_ITERS", "4")), bot_cooldown_min_iters)
    bot_next_eval_iter: Dict[str, int] = {}


    derivatives_specialists_enabled = os.getenv("DERIVATIVES_SPECIALISTS_ENABLED", "1").strip() == "1"
    derivatives_super_mode = _derivatives_super_trader_mode()
    options_specialists_raw = os.getenv(
        "OPTIONS_SPECIALIST_BOTS",
        "options_specialist_iv_surface,options_specialist_put_call_flow,options_specialist_skew_term_structure",
    ).strip()
    futures_specialists_raw = os.getenv(
        "FUTURES_SPECIALIST_BOTS",
        "futures_specialist_funding_basis,futures_specialist_orderbook_imbalance,futures_specialist_open_interest",
    ).strip()
    options_specialist_ids = [x.strip() for x in options_specialists_raw.split(",") if x.strip()]
    futures_specialist_ids = [x.strip() for x in futures_specialists_raw.split(",") if x.strip()]
    default_options_specialist_weight = "0.032" if derivatives_super_mode else "0.018"
    default_futures_specialist_weight = "0.034" if derivatives_super_mode else "0.018"
    options_specialist_weight = max(float(os.getenv("OPTIONS_SPECIALIST_WEIGHT", default_options_specialist_weight)), 0.001)
    futures_specialist_weight = max(float(os.getenv("FUTURES_SPECIALIST_WEIGHT", default_futures_specialist_weight)), 0.001)
    specialist_min_acc = float(os.getenv("DERIVATIVES_SPECIALIST_MIN_ACC", "0.58" if derivatives_super_mode else "0.54"))

    def _build_virtual_specialists(ids: List[str], role: str, weight: float) -> List[SubBot]:
        out: List[SubBot] = []
        for bot_id in ids:
            out.append(
                SubBot(
                    bot_id=str(bot_id),
                    weight=float(weight),
                    active=True,
                    reason="virtual_derivatives_specialist",
                    test_accuracy=specialist_min_acc,
                    promoted=False,
                    bot_role=role,
                )
            )
        return out


    paper_mirror_enabled = os.getenv("TOP_BOT_PAPER_TRADING_ENABLED", "0").strip() == "1"
    paper_mirror_top_n = max(int(os.getenv("TOP_BOT_PAPER_TRADING_TOP_N", "2")), 0)
    paper_mirror_min_accuracy = float(os.getenv("TOP_BOT_PAPER_TRADING_MIN_ACC", "0.0"))
    paper_mirror_profiles_raw = os.getenv("TOP_BOT_PAPER_TRADING_PROFILES", "").strip()
    paper_mirror_profiles = {
        x.strip().lower() for x in paper_mirror_profiles_raw.split(",") if x.strip()
    }
    current_profile = (_shadow_profile_name() or "default").strip().lower()
    paper_profile_ok = (not paper_mirror_profiles) or (current_profile in paper_mirror_profiles)
    paper_selected_ids: set[str] = set()

    if paper_mirror_enabled and (not paper_profile_ok):
        print(
            f"[PaperMirror] disabled profile={current_profile or 'default'} "
            f"allowed={sorted(paper_mirror_profiles)}"
        )

    paper_trader: Optional[BaseTrader] = None
    if paper_mirror_enabled and paper_profile_ok and paper_mirror_top_n > 0:
        paper_trader = BaseTrader(api_key, secret, redirect, mode="paper")
        paper_trader.execution_enabled = True
        paper_trader.market_data_only = False
        print(
            f"[PaperMirror] enabled top_n={paper_mirror_top_n} "
            f"min_acc={paper_mirror_min_accuracy:.3f} "
            f"profile={current_profile or 'default'}"
        )

    if resume_from_checkpoint:
        cp = checkpoint.load()
        cp_iter = int(cp.get("iter_count", 0) or 0)
        if cp_iter > 0:
            iter_count = cp_iter
            print(f"[Checkpoint] resumed iter_count={iter_count}")

    # Keep equities market-hours gated by default, but run crypto 24/7 by default.
    default_session_gate = '0' if broker == 'coinbase' else '1'
    default_blackout_gate = '0' if broker == 'coinbase' else '1'
    session_gate_enabled = os.getenv('SESSION_GATE_ENABLED', default_session_gate).strip() == '1'
    event_blackout_enabled = os.getenv('EVENT_LOCK_ENABLED', os.getenv('EVENT_BLACKOUT_ENABLED', default_blackout_gate)).strip() == '1'
    feature_freshness_enabled = os.getenv('FEATURE_FRESHNESS_GUARD_ENABLED', '1').strip() == '1'
    feature_freshness_max_age_seconds = float(os.getenv('FEATURE_FRESHNESS_MAX_AGE_SECONDS', '12'))
    feature_freshness_required = [
        x.strip() for x in os.getenv(
            'FEATURE_FRESHNESS_REQUIRED_KEYS',
            'last_price,prev_close,pct_from_close,vol_30m,mom_5m',
        ).split(',') if x.strip()
    ]
    master_latency_slo_enabled = os.getenv('MASTER_LATENCY_SLO_GUARD_ENABLED', '1').strip() == '1'
    master_latency_timeout_default = (
        os.getenv("MASTER_LATENCY_SLO_TIMEOUT_MS_CRYPTO", "1800")
        if broker == "coinbase"
        else os.getenv("MASTER_LATENCY_SLO_TIMEOUT_MS_EQUITIES", "800")
    )
    master_latency_slo_timeout_ms = float(os.getenv('MASTER_LATENCY_SLO_TIMEOUT_MS', master_latency_timeout_default))
    preopen_replay_sanity_enabled = os.getenv(
        'PREOPEN_REPLAY_SANITY_ENABLED',
        '0' if broker == 'coinbase' else '1',
    ).strip() == '1'
    preopen_replay_sanity_timeout_seconds = max(
        int(os.getenv('PREOPEN_REPLAY_SANITY_TIMEOUT_SECONDS', '30')),
        5,
    )
    snapshot_debug_mode = os.getenv('SNAPSHOT_DEBUG_MODE', '0').strip() == '1'
    gate_logging_enabled = os.getenv('LOG_GATE_EVALUATIONS', '1').strip() == '1'
    gate_log_passes = os.getenv('LOG_GATE_PASSES', '1').strip() == '1'
    data_ingress_logging_enabled = os.getenv('LOG_DATA_INGRESS', '1').strip() == '1'

    # Predeclare guards used in startup config payload; full state is initialized later.
    broker_truth_reconcile_enabled = _env_flag(
        "BROKER_TRUTH_RECONCILE_ENABLED",
        "1" if (broker == "schwab" and (not simulate)) else "0",
    )
    order_idempotency_enabled = _env_flag("ORDER_IDEMPOTENCY_ENABLED", "1")

    config_payload = {
        "symbols": symbols,
        "context_symbols": context_symbols,
        "interval_seconds": effective_interval_seconds,
        "session_start_hour": session_start_hour,
        "session_end_hour": session_end_hour,
        "auto_retrain": auto_retrain,
        "canary_max_weight": float(os.getenv("CANARY_MAX_WEIGHT", "0.08")),
        "async_pipeline": enable_async_pipeline,
        "session_gate_enabled": session_gate_enabled,
        "event_blackout_enabled": event_blackout_enabled,
        "news_context_enabled": news_context_enabled,
        "options_chain_enabled": options_chain_enabled,
        "calendar_context_enabled": calendar_context_enabled,
        "tradingeconomics_calendar_enabled": te_calendar_enabled,
        "log_sub_bot_decisions": log_sub_bot_decisions,
        "log_master_variant_decisions": log_master_variant_decisions,
        "log_grand_master_decisions": log_grand_master_decisions,
        "log_options_master_decisions": log_options_master_decisions,
        "gate_logging_enabled": gate_logging_enabled,
        "gate_log_passes": gate_log_passes,
        "data_ingress_logging_enabled": data_ingress_logging_enabled,
        "derivatives_specialists_enabled": derivatives_specialists_enabled,
        "derivatives_super_trader_mode": derivatives_super_mode,
        "options_specialist_weight": options_specialist_weight,
        "futures_specialist_weight": futures_specialist_weight,
        "broker_truth_reconcile_enabled": broker_truth_reconcile_enabled,
        "order_idempotency_enabled": order_idempotency_enabled,
        "lane_kill_switch_enabled": True,
        "dividend_strategy_mode": _dividend_strategy_mode() if _is_dividend_profile() else "n/a",
    }
    run_config_hash = config_hash(config_payload)
    _write_heartbeat(
        project_root=PROJECT_ROOT,
        broker=broker,
        iter_count=iter_count,
        symbols_total=len(symbols),
        context_total=len(context_symbols),
        state="starting",
    )
    print(f"[Config] hash={run_config_hash} async={enable_async_pipeline} canary_max_weight={float(os.getenv('CANARY_MAX_WEIGHT','0.08')):.3f}")
    print(
        "[Derivatives] "
        f"specialists_enabled={int(derivatives_specialists_enabled)} "
        f"super_trader={int(derivatives_super_mode)} "
        f"opt_weight={options_specialist_weight:.3f} "
        f"fut_weight={futures_specialist_weight:.3f}"
    )
    if _is_dividend_profile():
        print(f"[DividendStrategy] mode={_dividend_strategy_mode()} quality_symbols={len(_dividend_quality_symbols())}")
    print(f"[Correlation] run_id={run_id}")
    print(
        "[DecisionLogs] "
        f"sub_bot={int(log_sub_bot_decisions)} "
        f"master_variant={int(log_master_variant_decisions)} "
        f"grand_master={int(log_grand_master_decisions)} "
        f"options_master={int(log_options_master_decisions)}"
    )

    if preopen_replay_sanity_enabled and (not simulate):
        now_et = _now_eastern()
        open_now, closed_reason = _in_market_window(now_et, session_start_hour, session_end_hour)
        should_run = (not open_now) and (closed_reason == 'pre_window')
        if gate_logging_enabled:
            _log_gate_event(
                project_root=PROJECT_ROOT,
                broker=broker,
                iter_count=iter_count,
                symbol='*',
                gate='preopen_replay_sanity',
                passed=(not should_run),
                reason=('not_required' if not should_run else 'running'),
                details={
                    'now_et': now_et.isoformat(),
                    'open_now': bool(open_now),
                    'closed_reason': closed_reason,
                },
            )
        if should_run:
            ok, reason = _run_preopen_replay_sanity_check(
                project_root=PROJECT_ROOT,
                timeout_seconds=preopen_replay_sanity_timeout_seconds,
            )
            if gate_logging_enabled:
                _log_gate_event(
                    project_root=PROJECT_ROOT,
                    broker=broker,
                    iter_count=iter_count,
                    symbol='*',
                    gate='preopen_replay_sanity',
                    passed=bool(ok),
                    reason=('passed' if ok else str(reason)),
                    details={'timeout_seconds': preopen_replay_sanity_timeout_seconds},
                )
            if not ok:
                raise RuntimeError(f"Pre-open replay sanity check failed: {reason}")
            print('[ReplaySanity] pre-open replay check passed (24h)')
    else:
        if gate_logging_enabled:
            _log_gate_event(
                project_root=PROJECT_ROOT,
                broker=broker,
                iter_count=iter_count,
                symbol='*',
                gate='preopen_replay_sanity',
                passed=True,
                reason=('disabled' if not preopen_replay_sanity_enabled else 'simulate_mode'),
                details={
                    'enabled': bool(preopen_replay_sanity_enabled),
                    'simulate': bool(simulate),
                },
            )

    memory_guard_enabled = os.getenv("AUTO_RETRAIN_MEMORY_GUARD", "1").strip() == "1"
    auto_retrain_min_free_pct = float(os.getenv("AUTO_RETRAIN_MIN_FREE_PCT", "20"))
    auto_retrain_max_swap_gb = float(os.getenv("AUTO_RETRAIN_MAX_SWAP_GB", "2.2"))
    auto_retrain_healthy_streak_needed = max(int(os.getenv("AUTO_RETRAIN_HEALTHY_STREAK", "6")), 1)
    auto_retrain_healthy_streak = 0
    overload_mode = False

    consecutive_loss_streak = 0
    intraday_pnl_proxy = 0.0
    peak_intraday_pnl_proxy = 0.0
    kill_switch_until_ts = 0.0
    vol_shock_pause_until_ts = 0.0
    liquidity_pause_until_ts = 0.0
    max_consecutive_losses = max(int(os.getenv("RISK_MAX_CONSECUTIVE_LOSSES", "6")), 1)
    kill_switch_cooldown_seconds = max(int(os.getenv("RISK_KILL_SWITCH_COOLDOWN_SECONDS", "300")), 30)
    vol_shock_threshold = float(os.getenv("RISK_VOL_SHOCK_THRESHOLD", "0.05"))
    vol_shock_pause_seconds = max(int(os.getenv("RISK_VOL_SHOCK_PAUSE_SECONDS", "120")), 30)
    liquidity_spread_bps_threshold = float(os.getenv("RISK_LIQUIDITY_SPREAD_BPS_THRESHOLD", "35"))
    liquidity_pause_seconds = max(int(os.getenv("RISK_LIQUIDITY_PAUSE_SECONDS", "120")), 30)
    max_daily_loss_proxy = float(os.getenv("RISK_MAX_DAILY_LOSS_PROXY", "0.05"))
    max_intraday_drawdown_proxy = float(os.getenv("RISK_MAX_INTRADAY_DRAWDOWN_PROXY", "0.04"))
    symbol_circuit_hit_limit = max(int(os.getenv("RISK_SYMBOL_CIRCUIT_HIT_LIMIT", "4") or 4), 1)
    symbol_circuit_cooldown_seconds = max(int(os.getenv("RISK_SYMBOL_CIRCUIT_COOLDOWN_SECONDS", "180") or 180), 30)
    symbol_circuit_hits: Dict[str, int] = {}
    symbol_circuit_until: Dict[str, float] = {}

    lane_loss_streak: Dict[str, int] = {}
    lane_kill_until_ts: Dict[str, float] = {}
    lane_pause_counts: Dict[str, int] = {}
    lane_max_consecutive_losses = _parse_lane_int_caps(
        os.getenv(
            "RISK_LANE_MAX_CONSECUTIVE_LOSSES",
            "equities:6,day:4,swing:5,options:4,futures:4,long_term:8,default:6",
        )
    )
    lane_kill_cooldown_seconds = _parse_lane_int_caps(
        os.getenv(
            "RISK_LANE_KILL_SWITCH_COOLDOWN_SECONDS",
            "equities:300,day:240,swing:300,options:220,futures:220,long_term:360,default:300",
        )
    )
    lane_vol_shock_thresholds = _parse_lane_float_caps(
        os.getenv(
            "RISK_LANE_VOL_SHOCK_THRESHOLDS",
            "equities:0.050,day:0.040,swing:0.045,options:0.030,futures:0.028,long_term:0.060,default:0.050",
        )
    )
    lane_liquidity_spread_bps_thresholds = _parse_lane_float_caps(
        os.getenv(
            "RISK_LANE_LIQUIDITY_SPREAD_BPS_THRESHOLDS",
            "equities:35,day:28,swing:32,options:24,futures:24,long_term:40,default:35",
        )
    )

    broker_truth_refresh_iters = max(int(os.getenv("BROKER_TRUTH_REFRESH_ITERS", "6") or 6), 1)
    broker_truth_manual_tolerance = max(float(os.getenv("BROKER_TRUTH_MANUAL_QTY_TOLERANCE", "1.0") or 1.0), 0.0)
    broker_truth_state: Dict[str, Any] = {
        "active": bool(broker_truth_reconcile_enabled),
        "ok": True,
        "status": ("pending" if broker_truth_reconcile_enabled else "disabled"),
        "iter": 0,
        "age_iters": 0,
        "position_count": 0,
        "open_orders_total": 0,
        "positions": {},
        "manual_positions": {},
        "mismatch_count": 0,
        "mismatch_examples": [],
        "account_metrics": {},
        "capital_flow": _default_capital_flow_state(),
    }
    broker_truth_last_refresh_iter = 0

    order_idempotency_enabled = _env_flag("ORDER_IDEMPOTENCY_ENABLED", "1")
    order_idempotency_window_seconds = max(float(os.getenv("ORDER_IDEMPOTENCY_WINDOW_SECONDS", "180") or 180.0), 10.0)
    order_idempotency_ttl_seconds = max(
        float(os.getenv("ORDER_IDEMPOTENCY_TTL_SECONDS", "21600") or 21600.0),
        order_idempotency_window_seconds,
    )
    order_idempotency_max_entries = max(int(os.getenv("ORDER_IDEMPOTENCY_MAX_ENTRIES", "4000") or 4000), 100)
    order_idempotency_registry_path = _order_idempotency_registry_path(PROJECT_ROOT, broker)
    order_idempotency_state = (
        _load_order_idempotency_registry(order_idempotency_registry_path)
        if order_idempotency_enabled
        else {"entries": {}, "updated_utc": ""}
    )
    order_idempotency_dirty = False

    manual_trade_reconcile_enabled = os.getenv("MANUAL_TRADE_RECONCILE_ENABLED", "1").strip() == "1"
    manual_trade_reconcile_refresh_iters = max(int(os.getenv("MANUAL_TRADE_RECONCILE_REFRESH_ITERS", "8") or 8), 1)
    manual_trade_overrides: Dict[str, Any] = {"symbols": {}}

    portfolio_sector_map = _parse_symbol_label_map(os.getenv("PORTFOLIO_SYMBOL_SECTORS", ""))
    portfolio_risk_state: Dict[str, Any] = {
        "gross_notional": 0.0,
        "symbol_notional": {},
        "lane_notional": {},
        "sector_notional": {},
        "broker_notional": {},
    }
    long_term_policy_state: Dict[str, Any] = {"last_buy_iter_by_symbol": {}, "day_key": "", "used_turnover_pct": 0.0}

    log_maintenance_enabled = os.getenv('LOG_MAINTENANCE_ENABLED', '1').strip() == '1'
    log_maintenance_every_iters = max(int(os.getenv('LOG_MAINTENANCE_EVERY_ITERS', '20')), 1)
    log_maintenance_max_ops = max(int(os.getenv('LOG_MAINTENANCE_MAX_OPS', '300')), 10)

    exec_queue = ExecutionQueue(max_depth=max(int(os.getenv("EXEC_QUEUE_MAX_DEPTH", "4000")), 100))
    configured_equity_proxy = float(os.getenv("ACCOUNT_EQUITY_PROXY", "100000"))
    max_notional_pct = float(os.getenv("SIZING_MAX_NOTIONAL_PCT", "0.06"))
    symbol_budgets = _parse_symbol_budgets(os.getenv("PORTFOLIO_SYMBOL_BUDGETS", ""))
    portfolio_base_budget = float(os.getenv("PORTFOLIO_BASE_BUDGET", "1.0"))

    volatile_set = {x.upper() for x in volatile_symbols}
    defensive_set = {x.upper() for x in defensive_symbols}
    core_set = {x.upper() for x in symbols if x.upper() not in volatile_set and x.upper() not in defensive_set}
    core_every_n = max(int(os.getenv('CORE_SYMBOL_EVERY_N_ITERS', '1')), 1)
    volatile_every_n = max(int(os.getenv('VOLATILE_SYMBOL_EVERY_N_ITERS', '1')), 1)
    defensive_every_n = max(int(os.getenv('DEFENSIVE_SYMBOL_EVERY_N_ITERS', '2')), 1)

    def _record_snapshot_debug(symbol: str, reason: str, **extra: Any) -> None:
        if not snapshot_debug_mode:
            return
        row: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "iter": int(iter_count),
            "symbol": symbol,
            "reason": reason,
            "broker": broker,
            "profile": _shadow_profile_name() or "default",
            "domain": _shadow_domain_name(broker=broker),
        }
        if extra:
            row.update(extra)
        _append_jsonl(_snapshot_debug_path(PROJECT_ROOT, broker=broker), row)

    loop_state = "initialized"
    ingress_totals: Dict[str, int] = {
        "cache_ok": 0,
        "simulate_ok": 0,
        "api_ok": 0,
        "api_error": 0,
    }
    iter_ingress: Dict[str, int] = {
        "cache_ok": 0,
        "simulate_ok": 0,
        "api_ok": 0,
        "api_error": 0,
    }

    def _set_loop_state(new_state: str, reason: str = "", **extra: Any) -> None:
        nonlocal loop_state
        if (new_state == loop_state) and (not reason) and (not extra):
            return
        prev_state = loop_state
        loop_state = new_state
        _emit_loop_state(
            project_root=PROJECT_ROOT,
            broker=broker,
            prev_state=prev_state,
            new_state=new_state,
            iter_count=iter_count,
            reason=reason,
            extra=extra or None,
        )

    def _log_gate(symbol: str, gate: str, passed: bool, reason: str = "", **details: Any) -> None:
        if not gate_logging_enabled:
            return
        if bool(passed) and (not gate_log_passes):
            return
        _log_gate_event(
            project_root=PROJECT_ROOT,
            broker=broker,
            iter_count=iter_count,
            symbol=symbol,
            gate=gate,
            passed=bool(passed),
            reason=reason,
            details=details or None,
        )

    def _log_ingress(
        symbol: str,
        source: str,
        status: str,
        endpoint: str,
        latency_ms: float,
        reason: str = "",
        **details: Any,
    ) -> None:
        key = f"{source}_{status}" if f"{source}_{status}" in iter_ingress else "api_error"
        iter_ingress[key] = iter_ingress.get(key, 0) + 1
        ingress_totals[key] = ingress_totals.get(key, 0) + 1
        if not data_ingress_logging_enabled:
            return
        _log_ingress_event(
            project_root=PROJECT_ROOT,
            broker=broker,
            iter_count=iter_count,
            symbol=symbol,
            source=source,
            status=status,
            endpoint=endpoint,
            latency_ms=latency_ms,
            reason=reason,
            details=details or None,
        )

    def _publish_ingress_state(*, pause_gate: str = "", pause_reason: str = "") -> None:
        ingress_total_requests = sum(int(v or 0) for v in iter_ingress.values())
        ingress_error_count = int(iter_ingress.get("api_error", 0) or 0)
        ingress_error_rate = (float(ingress_error_count) / float(ingress_total_requests)) if ingress_total_requests > 0 else 0.0

        ingress_state = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": str(os.getenv("CORRELATION_RUN_ID", "") or "").strip(),
            "iter_id": str(os.getenv("CORRELATION_ITER_ID", "") or "").strip(),
            "iter": int(iter_count),
            "broker": broker,
            "profile": _shadow_profile_name() or "default",
            "domain": _shadow_domain_name(broker=broker),
            "loop_state": loop_state,
            "iter_counts": dict(iter_ingress),
            "total_counts": dict(ingress_totals),
            "iter_total_requests": int(ingress_total_requests),
            "iter_error_count": int(ingress_error_count),
            "iter_error_rate": round(float(ingress_error_rate), 6),
            "symbols_total": int(len(symbols)),
            "context_total": int(len(context_symbols)),
            "log_schema_version": max(int(os.getenv("LOG_SCHEMA_VERSION", "2")), 1),
        }
        if pause_gate:
            ingress_state["pause_gate"] = pause_gate
        if pause_reason:
            ingress_state["pause_reason"] = pause_reason

        _write_ingress_state(project_root=PROJECT_ROOT, broker=broker, payload=ingress_state)

        event_row: Dict[str, Any] = {
            "timestamp_utc": ingress_state["timestamp_utc"],
            "event": "data_ingress_summary",
            "broker": broker,
            "profile": ingress_state["profile"],
            "domain": ingress_state["domain"],
            "iter": int(iter_count),
            "iter_counts": dict(iter_ingress),
            "total_counts": dict(ingress_totals),
            "iter_total_requests": int(ingress_total_requests),
            "iter_error_count": int(ingress_error_count),
            "iter_error_rate": round(float(ingress_error_rate), 6),
        }
        if pause_gate:
            event_row["pause_gate"] = pause_gate
        if pause_reason:
            event_row["pause_reason"] = pause_reason

        _append_jsonl(_event_bus_path(PROJECT_ROOT), event_row)

    _set_loop_state("starting", reason="loop_bootstrap")

    while True:
        halt_active = _global_trading_halt_enabled()
        _log_gate("*", "global_trading_halt", not halt_active, reason=("halt_flag_set" if halt_active else "ok"))
        if halt_active:
            _set_loop_state("halted", reason="global_trading_halt")
            _write_heartbeat(
                project_root=PROJECT_ROOT,
                broker=broker,
                iter_count=iter_count,
                symbols_total=len(symbols),
                context_total=len(context_symbols),
                state="halted",
            )
            print("GLOBAL_TRADING_HALT=1 detected; stopping shadow loop.")
            return

        loop_started_at = time.time()
        iter_count += 1
        current_iter_id = f"{run_id}:{iter_count}"
        os.environ["CORRELATION_ITER_ID"] = current_iter_id
        iter_ingress["cache_ok"] = 0
        iter_ingress["simulate_ok"] = 0
        iter_ingress["api_ok"] = 0
        iter_ingress["api_error"] = 0

        if manual_trade_reconcile_enabled and (iter_count == 1 or (iter_count % manual_trade_reconcile_refresh_iters) == 0):
            manual_trade_overrides = _load_manual_trade_overrides(PROJECT_ROOT, broker)

        if broker_truth_reconcile_enabled and (
            iter_count == 1 or (iter_count - broker_truth_last_refresh_iter) >= broker_truth_refresh_iters
        ):
            broker_truth_state = _fetch_broker_truth_snapshot(
                trader=trader,
                broker=broker,
                simulate=simulate,
                iter_count=iter_count,
                manual_payload=manual_trade_overrides,
                manual_tolerance=broker_truth_manual_tolerance,
                previous_state=broker_truth_state,
            )
            broker_truth_last_refresh_iter = int(iter_count)
            broker_truth_state["age_iters"] = 0
            safe_write_json_atomic(
                _broker_truth_latest_path(PROJECT_ROOT, broker),
                broker_truth_state,
                project_root=PROJECT_ROOT,
                source="run_shadow_training_loop.broker_truth",
            )

            if not bool(broker_truth_state.get("ok", False)):
                _emit_critical_alert(
                    project_root=PROJECT_ROOT,
                    broker=broker,
                    event="broker_truth_reconcile",
                    message=str(broker_truth_state.get("status", "broker_truth_failed")),
                    details={
                        "iter": int(iter_count),
                        "mismatch_count": int(broker_truth_state.get("mismatch_count", 0) or 0),
                        "source": str(broker_truth_state.get("source", "")),
                    },
                    severity="critical",
                )
                if bool(broker_truth_state.get("mismatch_count", 0)):
                    print(
                        f"[BrokerTruth] mismatch count={int(broker_truth_state.get('mismatch_count', 0) or 0)} "
                        f"source={broker_truth_state.get('source', '')}"
                    )
        elif broker_truth_reconcile_enabled:
            broker_truth_state["age_iters"] = max(
                int(iter_count - int(broker_truth_state.get("iter", 0) or 0)),
                0,
            )
        else:
            broker_truth_state["status"] = "disabled"
            broker_truth_state["age_iters"] = 0
            broker_truth_state["capital_flow"] = _default_capital_flow_state()

        effective_equity_proxy, effective_equity_meta = _effective_account_equity_proxy(
            broker_truth=broker_truth_state,
            fallback_equity_proxy=configured_equity_proxy,
        )
        capital_flow_meta = broker_truth_state.get("capital_flow") if isinstance(broker_truth_state, dict) else {}
        if not isinstance(capital_flow_meta, dict):
            capital_flow_meta = _default_capital_flow_state()
        capital_flow_features = {
            key: float(capital_flow_meta.get(key, 0.0) or 0.0)
            for key in _CAPITAL_FLOW_FEATURE_KEYS
        }

        _write_heartbeat(
            project_root=PROJECT_ROOT,
            broker=broker,
            iter_count=iter_count,
            symbols_total=len(symbols),
            context_total=len(context_symbols),
            state="running",
        )
        _set_loop_state("running", reason="iteration_start")

        if retrain_proc is not None and retrain_proc.poll() is not None:
            rc = retrain_proc.returncode
            event = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "retrain_finished",
                "pid": retrain_proc.pid,
                "exit_code": rc,
            }
            _append_jsonl(_auto_retrain_log_path(PROJECT_ROOT, broker=broker), event)
            print(f"[AutoRetrain] finished pid={retrain_proc.pid} exit={rc}")
            retrain_proc = None
        now_et = _now_eastern()

        if session_gate_enabled:
            open_now, closed_reason = _in_market_window(now_et, session_start_hour, session_end_hour)
            _log_gate("*", "session_gate", bool(open_now), reason=(closed_reason if not open_now else "open"), now_et=now_et.isoformat())
            if not open_now:
                if closed_reason != last_closed_reason:
                    print(f"[SessionGate] paused reason={closed_reason} now_et={now_et.isoformat()}")
                    last_closed_reason = closed_reason
                _record_snapshot_debug('*', 'session_gate_paused', closed_reason=closed_reason, now_et=now_et.isoformat())
                _set_loop_state("paused_session_gate", reason=closed_reason)
                _publish_ingress_state(pause_gate="session_gate", pause_reason=closed_reason)
                time.sleep(max(effective_interval_seconds, 30))
                continue
            if last_closed_reason is not None:
                print(f"[SessionGate] resumed now_et={now_et.isoformat()}")
                last_closed_reason = None
        else:
            _log_gate("*", "session_gate", True, reason="disabled", now_et=now_et.isoformat())
            if iter_count == 1:
                print(f"[SessionGate] disabled broker={broker} mode=24x7")

        event_gate_blocked = bool(event_blackout_enabled and _in_event_blackout(now_et, blackout_windows))
        _log_gate(
            "*",
            "event_blackout",
            not event_gate_blocked,
            reason=("event_lock_window" if event_gate_blocked else ("disabled" if not event_blackout_enabled else "clear")),
            now_et=now_et.isoformat(),
        )
        if event_gate_blocked:
            print(f"[EventGate] paused reason=event_lock_window now_et={now_et.isoformat()}")
            _record_snapshot_debug('*', 'event_lock_paused', now_et=now_et.isoformat())
            _set_loop_state("paused_event_gate", reason="event_lock_window")
            _publish_ingress_state(pause_gate="event_blackout", pause_reason="event_lock_window")
            time.sleep(max(effective_interval_seconds, 10))
            continue

        anomaly_paused = bool(time.time() < anomaly_kill_switch_until_ts)
        _log_gate("*", "anomaly_killswitch", not anomaly_paused, reason=("data_anomaly" if anomaly_paused else "clear"))
        if anomaly_paused:
            rem = int(anomaly_kill_switch_until_ts - time.time())
            print(f"[KillSwitch] paused reason=data_anomaly remaining_s={max(rem, 0)}")
            _record_snapshot_debug('*', 'anomaly_killswitch_paused', remaining_seconds=max(rem, 0))
            _set_loop_state("paused_anomaly_killswitch", reason="data_anomaly", remaining_seconds=max(rem, 0))
            _publish_ingress_state(pause_gate="anomaly_killswitch", pause_reason="data_anomaly")
            time.sleep(max(min(rem, effective_interval_seconds), 5))
            continue

        if iter_count == 1 or iter_count % 10 == 0:
            registry, bots = _fresh_registry(registry_path)
            bots = _apply_canary_rollout_to_bots(bots, canary_rollout.max_weight)

        registry_active_bots = [b for b in bots if b.active]
        if not registry_active_bots:
            registry_active_bots = [
                SubBot(
                    bot_id="fallback_master_seed",
                    weight=1.0,
                    active=True,
                    reason="fallback",
                    test_accuracy=0.50,
                    bot_role="signal_sub_bot",
                )
            ]

        options_virtual_bots_template: List[SubBot] = []
        futures_virtual_bots_template: List[SubBot] = []
        if derivatives_specialists_enabled and (not _is_long_term_profile()):
            if broker == "schwab":
                options_virtual_bots_template = _build_virtual_specialists(
                    options_specialist_ids,
                    "options_sub_bot",
                    options_specialist_weight,
                )
                futures_virtual_bots_template = _build_virtual_specialists(
                    futures_specialist_ids,
                    "futures_sub_bot",
                    futures_specialist_weight,
                )
            elif broker == "coinbase":
                futures_virtual_bots_template = _build_virtual_specialists(
                    futures_specialist_ids,
                    "futures_sub_bot",
                    futures_specialist_weight,
                )

        active_bots = list(registry_active_bots)
        if _is_long_term_profile():
            pre_filter_count = len(active_bots)
            active_bots = _filter_long_term_registry_bots(active_bots)
            if not active_bots:
                active_bots = [
                    b for b in registry_active_bots
                    if not (_is_options_sub_bot(b) or _is_futures_sub_bot(b))
                ]
            if not active_bots:
                active_bots = list(registry_active_bots)
            removed = max(pre_filter_count - len(active_bots), 0)
            if removed > 0:
                print(
                    f"[LongTermIsolation] filtered_registry_bots removed={removed} "
                    f"kept={len(active_bots)}"
                )

        if paper_trader is not None:
            top_paper_bots = _top_paper_mirror_bots(
                active_bots,
                top_n=paper_mirror_top_n,
                min_accuracy=paper_mirror_min_accuracy,
            )
            new_selected_ids = {b.bot_id for b in top_paper_bots}
            if new_selected_ids != paper_selected_ids:
                paper_selected_ids = new_selected_ids
                selected_text = ",".join(
                    f"{b.bot_id}:{float(b.test_accuracy or 0.0):.3f}"
                    for b in top_paper_bots
                ) or "none"
                print(f"[PaperMirror] selected={selected_text}")


        fast_bots, slow_bots = _split_bot_tiers(active_bots)
        evaluate_slow_this_iter = (iter_count % slow_bot_every_n_iters == 0)

        latest_recs: List[Dict[str, Any]] = []
        exposure_state: Dict[str, int] = {"BUY": 0, "SELL": 0}

        context_market: Dict[str, Dict[str, float]] = {}

        def _fetch_symbol_snapshot(sym: str) -> Dict[str, float]:
            cache_key = f"mkt:{sym}"
            cached = state_cache.get(cache_key)
            if cached is not None:
                _log_api_call(
                    project_root=PROJECT_ROOT,
                    broker=broker,
                    symbol=sym,
                    endpoint="market_snapshot_cache",
                    status="cache_hit",
                    latency_ms=0.0,
                )
                _log_gate(sym, "market_data_fetch", True, reason="cache_hit", source="cache")
                _log_ingress(
                    sym,
                    source="cache",
                    status="ok",
                    endpoint="market_snapshot_cache",
                    latency_ms=0.0,
                    reason="cache_hit",
                )
                return cached

            cb_key = f"md:{sym}"
            if not circuit_breaker.allow(cb_key):
                _log_api_call(
                    project_root=PROJECT_ROOT,
                    broker=broker,
                    symbol=sym,
                    endpoint="circuit_breaker",
                    status="blocked",
                    latency_ms=0.0,
                    error="circuit_open",
                )
                _log_gate(sym, "market_data_circuit", False, reason="circuit_open", source="circuit_breaker")
                _log_ingress(
                    sym,
                    source="api",
                    status="error",
                    endpoint="circuit_breaker",
                    latency_ms=0.0,
                    reason="circuit_open",
                )
                raise RuntimeError("circuit_open")

            endpoint = "simulated_market_snapshot"
            ingress_source = "simulate"
            if not simulate:
                endpoint = "schwab.market_snapshot" if broker == "schwab" else "coinbase.market_snapshot"
                ingress_source = "api"

            started = time.perf_counter()
            try:
                if simulate:
                    snap = _market_snapshot_simulated(sim_prices[sym])
                    sim_prices[sym] = snap["last_price"]
                else:
                    if broker == "schwab":
                        snap = _market_snapshot_from_schwab(client, sym)
                    else:
                        snap = _market_snapshot_from_coinbase(client, sym)
                latency_ms = (time.perf_counter() - started) * 1000.0
                snap = dict(snap or {})
                snap["market_data_latency_ms"] = round(float(latency_ms), 3)
                if float(snap.get("queue_depth", 0.0) or 0.0) <= 0.0:
                    bid_size = max(float(snap.get("bid_size", 0.0) or 0.0), 0.0)
                    ask_size = max(float(snap.get("ask_size", 0.0) or 0.0), 0.0)
                    snap["queue_depth"] = bid_size + ask_size
                if float(snap.get("queue_depth_norm", 0.0) or 0.0) <= 0.0 and float(snap.get("futures_depth_ratio_norm", 0.0) or 0.0) > 0.0:
                    snap["queue_depth_norm"] = float(snap.get("futures_depth_ratio_norm", 0.0) or 0.0)

                state_cache.set(cache_key, snap)
                circuit_breaker.record_success(cb_key)
                _log_api_call(
                    project_root=PROJECT_ROOT,
                    broker=broker,
                    symbol=sym,
                    endpoint=endpoint,
                    status="ok",
                    latency_ms=latency_ms,
                )
                _log_gate(sym, "market_data_fetch", True, reason="ok", endpoint=endpoint)
                _log_ingress(
                    sym,
                    source=ingress_source,
                    status="ok",
                    endpoint=endpoint,
                    latency_ms=latency_ms,
                    reason="ok",
                )
                return snap
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000.0
                _log_api_call(
                    project_root=PROJECT_ROOT,
                    broker=broker,
                    symbol=sym,
                    endpoint=endpoint,
                    status="error",
                    latency_ms=latency_ms,
                    error=str(exc),
                    extra={"error_type": type(exc).__name__},
                )
                _log_gate(
                    sym,
                    "market_data_fetch",
                    False,
                    reason=f"{type(exc).__name__}:{exc}",
                    endpoint=endpoint,
                )
                _log_ingress(
                    sym,
                    source=ingress_source,
                    status="error",
                    endpoint=endpoint,
                    latency_ms=latency_ms,
                    reason=str(exc),
                    error_type=type(exc).__name__,
                )
                raise

        def _fetch_symbol_news_features(sym: str) -> Dict[str, float]:
            if not news_context_enabled:
                return _default_news_features()
            if news_method_state.get("disabled"):
                return _default_news_features()

            now_s = time.time()
            cached = news_cache.get(sym)
            if cached is not None and (now_s - cached[0]) <= news_cache_ttl_seconds:
                return dict(cached[1])

            payload = None
            for method_name in (
                "get_news",
                "get_news_headlines",
                "get_news_for_symbol",
                "get_news_headlines_for_symbol",
                "search_news",
            ):
                payload = _try_schwab_news_method(client, method_name, sym, news_max_items)
                if payload is not None:
                    news_method_state["method"] = method_name
                    break

            if payload is None:
                if not news_method_state.get("warned"):
                    print("[NewsFeed] schwab client has no accessible news endpoint; continuing with price-only features")
                    news_method_state["warned"] = True
                news_method_state["disabled"] = True
                feats = _default_news_features()
                news_cache[sym] = (now_s, feats)
                return feats

            items = _extract_news_items(payload, sym)
            feats = _summarize_news_items(
                items,
                symbol=sym,
                now_ts=now_s,
                lookback_seconds=news_lookback_hours * 3600.0,
                max_items=news_max_items,
            )
            news_cache[sym] = (now_s, feats)
            return dict(feats)

        def _fetch_symbol_options_features(sym: str, mkt: Dict[str, float]) -> Dict[str, float]:
            if (not options_chain_enabled) or _symbol_is_probably_futures(sym):
                return default_options_features()
            if options_chain_method_state.get("disabled"):
                return default_options_features()

            now_s = time.time()
            cached = options_chain_cache.get(sym)
            if cached is not None and (now_s - cached[0]) <= options_chain_cache_ttl_seconds:
                return dict(cached[1])

            payload = None
            for method_name in (
                "get_option_chain",
                "get_options_chain",
                "option_chain",
                "options_chain",
                "get_option_chain_for_symbol",
            ):
                payload = _try_schwab_option_chain_method(client, method_name, sym, options_chain_strike_count)
                if payload is not None:
                    options_chain_method_state["method"] = method_name
                    break

            if payload is None:
                if not options_chain_method_state.get("warned"):
                    print("[OptionsChain] no accessible Schwab options chain endpoint; using price-only context")
                    options_chain_method_state["warned"] = True
                options_chain_method_state["disabled"] = True
                feats = default_options_features()
                options_chain_cache[sym] = (now_s, feats)
                return feats

            feats = summarize_option_chain(
                payload,
                symbol=sym,
                underlying_price=float(mkt.get("last_price", 0.0) or 0.0),
                now_ts=now_s,
                max_contracts=max(int(os.getenv("SCHWAB_OPTIONS_CHAIN_MAX_CONTRACTS", "900")), 100),
            )

            # Some Schwab payloads expose expiries only in keyed maps like YYYY-MM-DD:NN.
            if isinstance(payload, dict) and (
                float(feats.get("options_near_expiry_days", 0.0) or 0.0) <= 0.0
                or float(feats.get("options_far_expiry_days", 0.0) or 0.0) <= 0.0
            ):
                expiry_days: List[float] = []
                for map_key in (
                    "callExpDateMap",
                    "putExpDateMap",
                    "expDateMap",
                    "expirationMap",
                    "expiryMap",
                ):
                    node = payload.get(map_key)
                    if isinstance(node, dict):
                        for exp_key in node.keys():
                            d = parse_expiry_key_days(str(exp_key), now_ts=now_s)
                            if d is not None:
                                expiry_days.append(max(float(d), 0.0))
                if expiry_days:
                    near_exp = min(expiry_days)
                    far_exp = max(expiry_days)
                    feats["options_near_expiry_days"] = near_exp
                    feats["options_far_expiry_days"] = far_exp
                    feats["options_near_expiry_norm"] = max(0.0, min(near_exp / 45.0, 1.0))
                    feats["options_far_expiry_norm"] = max(0.0, min(far_exp / 180.0, 1.0))

            options_chain_cache[sym] = (now_s, feats)
            return dict(feats)

        def _fetch_calendar_features(sym: str) -> Dict[str, float]:
            if not calendar_context_enabled:
                return default_calendar_features()
            if calendar_method_state.get("disabled"):
                return default_calendar_features()

            now_s = time.time()
            cached_ts = float(calendar_cache.get("ts", 0.0) or 0.0)
            cached_feats = calendar_cache.get("features")
            if (now_s - cached_ts) <= calendar_cache_ttl_seconds and isinstance(cached_feats, dict):
                return dict(cached_feats)

            payload = None
            calendar_source = ""
            calendar_endpoint = ""

            for method_name in (
                "get_market_calendar",
                "get_calendar",
                "get_events",
                "get_market_events",
                "get_economic_calendar",
            ):
                payload = _try_schwab_calendar_method(client, method_name, sym, calendar_days_ahead)
                if payload is not None:
                    calendar_method_state["method"] = method_name
                    calendar_method_state["source"] = "schwab"
                    calendar_source = "schwab"
                    calendar_endpoint = f"schwab.{method_name}"
                    break

            if payload is None and te_calendar_enabled:
                te_started = time.perf_counter()
                te_payload, te_endpoint, te_error = _try_tradingeconomics_calendar(
                    days_ahead=calendar_days_ahead,
                    country=te_calendar_country,
                    importance_csv=te_calendar_importance,
                    timeout_seconds=te_calendar_timeout_seconds,
                    auth_token=te_auth_token,
                    max_items=te_calendar_max_items,
                )
                te_latency_ms = (time.perf_counter() - te_started) * 1000.0
                if te_payload is not None:
                    payload = te_payload
                    calendar_method_state["method"] = te_endpoint
                    calendar_method_state["source"] = "tradingeconomics"
                    calendar_source = "tradingeconomics"
                    calendar_endpoint = te_endpoint
                    _log_api_call(
                        project_root=PROJECT_ROOT,
                        broker=broker,
                        symbol="*",
                        endpoint=te_endpoint,
                        status="ok",
                        latency_ms=te_latency_ms,
                        extra={"calendar_source": "tradingeconomics", "country": te_calendar_country},
                    )
                    if not calendar_method_state.get("te_announced"):
                        print("[CalendarFeed] using TradingEconomics calendar fallback")
                        calendar_method_state["te_announced"] = True
                else:
                    _log_api_call(
                        project_root=PROJECT_ROOT,
                        broker=broker,
                        symbol="*",
                        endpoint=(te_endpoint or "tradingeconomics.calendar"),
                        status="error",
                        latency_ms=te_latency_ms,
                        error=te_error,
                        extra={"calendar_source": "tradingeconomics", "country": te_calendar_country},
                    )

            if payload is None:
                if not calendar_method_state.get("warned"):
                    if te_calendar_enabled:
                        print("[CalendarFeed] no accessible Schwab or TradingEconomics calendar endpoint; using zero calendar features")
                    else:
                        print("[CalendarFeed] no accessible Schwab calendar endpoint; using zero calendar features")
                    calendar_method_state["warned"] = True
                if not te_calendar_enabled:
                    calendar_method_state["disabled"] = True
                feats = _external_macro_calendar_proxy_features(PROJECT_ROOT)
                calendar_cache["ts"] = now_s
                calendar_cache["features"] = feats
                return feats

            feats = summarize_calendar_payload(
                payload,
                now_ts=now_s,
                max_items=max(int(os.getenv("SCHWAB_CALENDAR_MAX_ITEMS", "600")), 60),
            )
            if calendar_source == "tradingeconomics":
                feats["calendar_feed_available"] = max(float(feats.get("calendar_feed_available", 0.0) or 0.0), 1.0)
            if calendar_endpoint:
                calendar_cache["source"] = calendar_source
                calendar_cache["endpoint"] = calendar_endpoint
            calendar_cache["ts"] = now_s
            calendar_cache["features"] = feats
            return dict(feats)

        def _load_external_context_category(name: str) -> Dict[str, Any]:
            token = str(name or "").strip().lower()
            if not token:
                return {}
            now_s = time.time()
            cache_row = external_context_cache.setdefault(token, {"ts": 0.0, "payload": {}})
            if (now_s - float(cache_row.get("ts", 0.0) or 0.0)) <= external_context_cache_ttl_seconds:
                payload = cache_row.get("payload")
                return dict(payload) if isinstance(payload, dict) else {}
            payload = load_latest_external_context(PROJECT_ROOT, token)
            cache_row["ts"] = now_s
            cache_row["payload"] = dict(payload) if isinstance(payload, dict) else {}
            return dict(cache_row["payload"])

        if context_symbols:
            if enable_async_pipeline:
                with ThreadPoolExecutor(max_workers=min(async_workers, max(len(context_symbols), 1))) as ex:
                    fut = {ex.submit(_fetch_symbol_snapshot, s): s for s in context_symbols}
                    for f in as_completed(fut):
                        ctx_symbol = fut[f]
                        try:
                            context_market[ctx_symbol] = f.result()
                        except Exception as exc:
                            opened = circuit_breaker.record_failure(f"md:{ctx_symbol}")
                            if opened:
                                print(f"[CircuitBreaker] opened symbol={ctx_symbol} layer=context")
                            print(f"[Context] symbol={ctx_symbol} market_data_error={exc}")
            else:
                for ctx_symbol in context_symbols:
                    try:
                        context_market[ctx_symbol] = _fetch_symbol_snapshot(ctx_symbol)
                    except Exception as exc:
                        opened = circuit_breaker.record_failure(f"md:{ctx_symbol}")
                        if opened:
                            print(f"[CircuitBreaker] opened symbol={ctx_symbol} layer=context")
                        print(f"[Context] symbol={ctx_symbol} market_data_error={exc}")

        now_ts = time.time()
        for symbol in symbols:
            u = symbol.upper()
            if u in volatile_set:
                cadence = volatile_every_n
            elif u in defensive_set:
                cadence = defensive_every_n
            else:
                cadence = core_every_n
            cadence_ok = not (cadence > 1 and (iter_count % cadence != 0))
            _log_gate(
                symbol,
                "symbol_cadence",
                cadence_ok,
                reason=("ok" if cadence_ok else "cadence_skip"),
                cadence=cadence,
            )
            if not cadence_ok:
                _record_snapshot_debug(symbol, 'cadence_skip', cadence=cadence)
                continue

            quarantine_until = symbol_quarantine_until.get(symbol, 0.0)
            quarantine_ok = now_ts >= quarantine_until
            _log_gate(
                symbol,
                "symbol_quarantine",
                quarantine_ok,
                reason=("ok" if quarantine_ok else "quarantined"),
                quarantine_seconds=round(max(quarantine_until - now_ts, 0.0), 3),
            )
            if not quarantine_ok:
                _record_snapshot_debug(symbol, 'quarantine_skip', quarantine_seconds=round(max(quarantine_until - now_ts, 0.0), 3))
                continue

            symbol_circuit_ok = circuit_breaker.allow(f"md:{symbol}")
            _log_gate(
                symbol,
                "symbol_market_circuit",
                symbol_circuit_ok,
                reason=("ok" if symbol_circuit_ok else "circuit_open"),
            )
            if not symbol_circuit_ok:
                _record_snapshot_debug(symbol, 'circuit_open_skip')
                continue

            try:
                mkt = _fetch_symbol_snapshot(symbol)
            except Exception as exc:
                symbol_fail_counts[symbol] = symbol_fail_counts.get(symbol, 0) + 1
                anomaly_fail_streak += 1
                if anomaly_fail_streak >= anomaly_kill_threshold:
                    anomaly_kill_switch_until_ts = time.time() + anomaly_kill_cooldown_seconds
                    print(f"[KillSwitch] tripped reason=market_data_errors cooldown_s={anomaly_kill_cooldown_seconds}")
                    anomaly_fail_streak = 0
                opened = circuit_breaker.record_failure(f"md:{symbol}")
                if opened:
                    print(f"[CircuitBreaker] opened symbol={symbol} layer=market_data")
                print(f"[ShadowLoop] iter={iter_count} symbol={symbol} market_data_error={exc}")
                _record_snapshot_debug(symbol, 'market_data_error', error=str(exc))
                _set_loop_state(
                    "degraded_market_data",
                    reason="market_data_error",
                    symbol=symbol,
                    error=str(exc),
                )
                if symbol_fail_counts[symbol] >= max(bad_symbol_fail_limit, 1):
                    symbol_quarantine_until[symbol] = now_ts + max(bad_symbol_retry_minutes, 1) * 60
                    print(
                        f"[SymbolGuard] quarantined symbol={symbol} "
                        f"for {max(bad_symbol_retry_minutes, 1)}m after {symbol_fail_counts[symbol]} failures"
                    )
                    symbol_fail_counts[symbol] = 0
                continue

            snapshot_usable = _is_usable_market_snapshot(mkt)
            _log_gate(
                symbol,
                "snapshot_usable",
                snapshot_usable,
                reason=("ok" if snapshot_usable else "invalid_snapshot"),
            )
            if not snapshot_usable:
                symbol_fail_counts[symbol] = symbol_fail_counts.get(symbol, 0) + 1
                anomaly_fail_streak += 1
                if anomaly_fail_streak >= anomaly_kill_threshold:
                    anomaly_kill_switch_until_ts = time.time() + anomaly_kill_cooldown_seconds
                    print(f"[KillSwitch] tripped reason=invalid_snapshot cooldown_s={anomaly_kill_cooldown_seconds}")
                    anomaly_fail_streak = 0
                print(
                    f"[SymbolGuard] invalid_snapshot symbol={symbol} "
                    f"count={symbol_fail_counts[symbol]}"
                )
                _record_snapshot_debug(symbol, 'invalid_snapshot', fail_count=symbol_fail_counts[symbol])
                if symbol_fail_counts[symbol] >= max(bad_symbol_fail_limit, 1):
                    symbol_quarantine_until[symbol] = now_ts + max(bad_symbol_retry_minutes, 1) * 60
                    print(
                        f"[SymbolGuard] quarantined symbol={symbol} "
                        f"for {max(bad_symbol_retry_minutes, 1)}m due to repeated invalid snapshot"
                    )
                    symbol_fail_counts[symbol] = 0
                continue

            px = float(mkt.get("last_price", 0.0))
            prev_px = float(symbol_last_price.get(symbol, 0.0))
            symbol_return_1m = ((px - prev_px) / prev_px) if (px > 0.0 and prev_px > 0.0) else 0.0

            max_abs_return_1m = max(float(os.getenv("MAX_SYMBOL_RETURN_1M_ABS", "0.35")), 0.01)
            outlier_guard_ok = not (px > 0.0 and prev_px > 0.0 and abs(symbol_return_1m) > max_abs_return_1m)
            _log_gate(
                symbol,
                "return_outlier_guard",
                outlier_guard_ok,
                reason=("ok" if outlier_guard_ok else "outlier_return"),
                return_1m=round(symbol_return_1m, 6),
                max_abs_return_1m=max_abs_return_1m,
            )
            if not outlier_guard_ok:
                symbol_fail_counts[symbol] = symbol_fail_counts.get(symbol, 0) + 1
                anomaly_fail_streak += 1
                print(
                    f"[SymbolGuard] outlier_return symbol={symbol} ret_1m={symbol_return_1m:.3f} "
                    f"prev={prev_px:.4f} px={px:.4f} limit={max_abs_return_1m:.3f} "
                    f"count={symbol_fail_counts[symbol]}"
                )
                _record_snapshot_debug(
                    symbol,
                    'outlier_return',
                    ret_1m=round(symbol_return_1m, 6),
                    prev_price=round(prev_px, 6),
                    price=round(px, 6),
                    limit=max_abs_return_1m,
                    fail_count=symbol_fail_counts[symbol],
                )
                if symbol_fail_counts[symbol] >= max(bad_symbol_fail_limit, 1):
                    symbol_quarantine_until[symbol] = now_ts + max(bad_symbol_retry_minutes, 1) * 60
                    print(
                        f"[SymbolGuard] quarantined symbol={symbol} "
                        f"for {max(bad_symbol_retry_minutes, 1)}m due to outlier returns"
                    )
                    symbol_fail_counts[symbol] = 0
                continue

            if px > 0.0 and prev_px > 0.0 and abs(px - prev_px) < 1e-10:
                symbol_stale_counts[symbol] = symbol_stale_counts.get(symbol, 0) + 1
            else:
                symbol_stale_counts[symbol] = 0
            symbol_last_price[symbol] = px

            stale_limit = max(int(os.getenv("STALE_PRICE_FAIL_LIMIT", "8")), 1)
            stale_guard_ok = symbol_stale_counts.get(symbol, 0) < stale_limit
            _log_gate(
                symbol,
                "stale_price_guard",
                stale_guard_ok,
                reason=("ok" if stale_guard_ok else "stale_price_limit"),
                stale_count=int(symbol_stale_counts.get(symbol, 0)),
                stale_limit=stale_limit,
            )
            if not stale_guard_ok:
                anomaly_fail_streak += 1
                if anomaly_fail_streak >= anomaly_kill_threshold:
                    anomaly_kill_switch_until_ts = time.time() + anomaly_kill_cooldown_seconds
                    print(f"[KillSwitch] tripped reason=stale_prices cooldown_s={anomaly_kill_cooldown_seconds}")
                    anomaly_fail_streak = 0
                symbol_quarantine_until[symbol] = now_ts + max(bad_symbol_retry_minutes, 1) * 60
                print(
                    f"[SymbolGuard] stale_price symbol={symbol} count={symbol_stale_counts[symbol]} "
                    f"quarantine={max(bad_symbol_retry_minutes, 1)}m"
                )
                _record_snapshot_debug(symbol, 'stale_price_quarantine', stale_count=symbol_stale_counts[symbol])
                symbol_stale_counts[symbol] = 0
                continue

            if symbol_fail_counts.get(symbol, 0) > 0:
                symbol_fail_counts[symbol] = 0
            anomaly_fail_streak = max(0, anomaly_fail_streak - 1)
            circuit_breaker.record_success(f"md:{symbol}")

            # One shared snapshot per symbol/iteration, reused by every sub-bot and master decision.
            snapshot_id = f"{symbol}-{iter_count}-{int(time.time() * 1000)}"
            _record_snapshot_debug(symbol, 'snapshot_created', snapshot_id=snapshot_id)

            # Attach macro regime context (e.g., VIX and dollar proxy) to every model decision.
            context_features: Dict[str, float] = {}
            for ctx_symbol, ctx_mkt in context_market.items():
                k = _feature_symbol_key(ctx_symbol)
                context_features[f"ctx_{k}_last_price"] = ctx_mkt.get("last_price", 0.0)
                context_features[f"ctx_{k}_pct_from_close"] = ctx_mkt.get("pct_from_close", 0.0)
                context_features[f"ctx_{k}_vol_30m"] = ctx_mkt.get("vol_30m", 0.0)
                context_features[f"ctx_{k}_mom_5m"] = ctx_mkt.get("mom_5m", 0.0)

            live_macro_snapshot = _load_external_context_category("live_macro")

            news_features = _default_news_features()
            if news_context_enabled:
                try:
                    news_features = _fetch_symbol_news_features(symbol)
                except Exception as exc:
                    news_features = _default_news_features()
                    _record_snapshot_debug(symbol, 'news_feed_error', error=str(exc))

            options_features = default_options_features()
            if options_chain_enabled and (not _symbol_is_probably_futures(symbol)):
                try:
                    options_features = _fetch_symbol_options_features(symbol, mkt)
                except Exception as exc:
                    options_features = default_options_features()
                    _record_snapshot_debug(symbol, 'options_chain_error', error=str(exc))

            calendar_features = default_calendar_features()
            if calendar_context_enabled:
                try:
                    calendar_features = _fetch_calendar_features(symbol)
                except Exception as exc:
                    calendar_features = default_calendar_features()
                    _record_snapshot_debug(symbol, 'calendar_feed_error', error=str(exc))

            if live_macro_snapshot:
                news_features = _merge_live_macro_news_features(
                    news_features,
                    symbol=symbol,
                    now_ts=now_ts,
                    snapshot=live_macro_snapshot,
                )
                calendar_features = _merge_live_macro_calendar_features(
                    calendar_features,
                    symbol=symbol,
                    now_ts=now_ts,
                    snapshot=live_macro_snapshot,
                )

            news_features = _augment_news_features_with_event_proxy(
                news_features,
                market_snapshot=mkt,
                calendar_features=calendar_features,
            )

            feature_cache_key = f"f:{symbol}"
            if feature_cache_enabled and feature_cache_key in feature_cache and feature_cache[feature_cache_key][0] == iter_count:
                shared_features = dict(feature_cache[feature_cache_key][1])
            else:
                shared_features = {
                    **mkt,
                    **context_features,
                    **news_features,
                    **options_features,
                    **calendar_features,
                }
                if feature_cache_enabled:
                    feature_cache[feature_cache_key] = (iter_count, dict(shared_features))

            breadth_features = summarize_breadth_context(
                symbol=symbol,
                market_snapshot=mkt,
                context_market=context_market,
                external_snapshot=_load_external_context_category("market_breadth"),
            )
            bond_reference_features = summarize_bond_reference_context(
                symbol=symbol,
                market_snapshot=shared_features,
                context_market=context_market,
                calendar_features=calendar_features,
                external_snapshot=_load_external_context_category("bond_reference"),
            )
            credit_context_features = summarize_credit_context(
                symbol=symbol,
                market_snapshot=shared_features,
                context_market=context_market,
                external_snapshot=_load_external_context_category("bond_reference"),
            )
            execution_lag_features = dict(default_execution_lag_features())
            execution_lag_features.update(execution_lag_by_symbol.get(symbol, {}))
            shared_features = {
                **shared_features,
                **default_breadth_features(),
                **default_bond_reference_features(),
                **default_credit_context_features(),
                **execution_lag_features,
                **breadth_features,
                **bond_reference_features,
                **credit_context_features,
            }

            freshness_ok, freshness_reason, freshness_age_s = _feature_freshness_guard(
                shared_features,
                max_age_seconds=feature_freshness_max_age_seconds,
                required_keys=feature_freshness_required,
            ) if feature_freshness_enabled else (True, 'disabled', 0.0)
            missing_feature_count = 0
            required_feature_count = len(feature_freshness_required)
            if required_feature_count > 0:
                for req_key in feature_freshness_required:
                    key = str(req_key or "").strip()
                    if not key:
                        continue
                    if shared_features.get(key) is None:
                        missing_feature_count += 1
            data_quality_features = summarize_data_quality_context(
                market_snapshot=shared_features,
                freshness_ok=freshness_ok,
                freshness_age_seconds=float(freshness_age_s),
                symbol_fail_count=int(symbol_fail_counts.get(symbol, 0) or 0),
                symbol_stale_count=int(symbol_stale_counts.get(symbol, 0) or 0),
                symbol_circuit_hits=int(symbol_circuit_hits.get(symbol, 0) or 0),
                quarantine_seconds=max(float(symbol_quarantine_until.get(symbol, 0.0) or 0.0) - now_ts, 0.0),
                missing_feature_count=missing_feature_count,
                required_feature_count=required_feature_count,
            )
            shared_features = {
                **shared_features,
                **default_data_quality_features(),
                **data_quality_features,
            }
            _log_gate(
                symbol,
                "feature_freshness",
                freshness_ok,
                reason=freshness_reason,
                age_seconds=round(float(freshness_age_s), 4),
                max_age_seconds=feature_freshness_max_age_seconds,
            )

            symbol_is_futures = _symbol_is_probably_futures(symbol) or (broker == "coinbase")
            options_specialist_bots = (
                list(options_virtual_bots_template)
                if derivatives_specialists_enabled and (not symbol_is_futures)
                else []
            )
            futures_specialist_bots = (
                list(futures_virtual_bots_template)
                if derivatives_specialists_enabled and symbol_is_futures
                else []
            )
            active_options_sub_bots = len(options_specialist_bots)
            active_futures_sub_bots = len(futures_specialist_bots)
            active_sub_bots_total = len(active_bots) + active_options_sub_bots + active_futures_sub_bots

            shared_features = {
                **shared_features,
                "active_sub_bots": float(active_sub_bots_total),
                "active_options_sub_bots": float(active_options_sub_bots),
                "active_futures_sub_bots": float(active_futures_sub_bots),
                **_default_lane_strategy_features(),
                **_default_capital_flow_features(),
                **capital_flow_features,
            }

            if _is_dividend_profile():
                div_mode = _dividend_strategy_mode()
                shared_features = {
                    **shared_features,
                    "dividend_strategy_mode_capture": 1.0 if div_mode == "capture" else 0.0,
                    "dividend_strategy_mode_compound": 1.0 if div_mode == "compound" else 0.0,
                    "dividend_strategy_mode_hybrid": 1.0 if div_mode == "hybrid" else 0.0,
                }
                shared_features["dividend_quality_score_norm"] = _dividend_quality_score(symbol, shared_features)
                shared_features = {
                    **shared_features,
                    **_update_dividend_compound_metrics(
                        symbol=symbol,
                        symbol_return_1m=symbol_return_1m,
                        features=shared_features,
                        state=dividend_compound_state,
                    ),
                }

            lane_preview_features = _behavior_lane_feature_preview(
                symbol=symbol,
                features=shared_features,
                day_state=day_strategy_state,
                swing_state=swing_strategy_state,
            )
            if lane_preview_features:
                shared_features = {
                    **shared_features,
                    **lane_preview_features,
                }

            mom = float(shared_features.get("mom_5m", 0.0))
            pct = float(shared_features.get("pct_from_close", 0.0))
            vix = float(shared_features.get("ctx_VIX_X_pct_from_close", 0.0))
            marker = (1 if mom > 0.001 else (-1 if mom < -0.001 else 0), 1 if pct > 0.001 else (-1 if pct < -0.001 else 0), 1 if vix > 0.002 else (-1 if vix < -0.002 else 0))
            last_marker = symbol_regime_marker.get(symbol)
            if regime_cooldown_iters > 0 and last_marker is not None and marker != last_marker:
                symbol_regime_cooldown_until_iter[symbol] = iter_count + regime_cooldown_iters
                print(f"[RegimeCooldown] symbol={symbol} cooldown_iters={regime_cooldown_iters} prev={last_marker} now={marker}")
            symbol_regime_marker[symbol] = marker

            sub_rows: List[Dict[str, Any]] = []
            candidate_bots = list(fast_bots)
            if evaluate_slow_this_iter:
                candidate_bots.extend(slow_bots)

            eval_bots: List[SubBot] = []
            reused_rows: List[Dict[str, Any]] = []
            for b in candidate_bots:
                key = f"{symbol}:{b.bot_id}"
                if bot_cooldown_enabled and b.bot_role != "infrastructure_sub_bot":
                    acc = float(b.test_accuracy or 0.5)
                    if acc < bot_cooldown_acc_floor:
                        weakness = min(max((bot_cooldown_acc_floor - acc) / max(bot_cooldown_acc_floor, 1e-6), 0.0), 1.0)
                        cooldown_span = int(round(bot_cooldown_min_iters + weakness * (bot_cooldown_max_iters - bot_cooldown_min_iters)))
                    else:
                        cooldown_span = 1
                else:
                    cooldown_span = 1

                next_iter = int(bot_next_eval_iter.get(key, 1))
                if iter_count >= next_iter:
                    eval_bots.append(b)
                    bot_next_eval_iter[key] = iter_count + max(cooldown_span, 1)
                else:
                    cached = slow_bot_rows_cache.get(symbol, {}).get(b.bot_id)
                    if cached is not None:
                        row = dict(cached)
                        row["reused"] = 1
                        reused_rows.append(row)

            batched_rows = _sub_bot_signal_batch(eval_bots, mkt)

            gates = {
                "market_data_ok": mkt["last_price"] > 0,
                "session_open_assumed": True,
                "risk_limit_ok": True,
            }

            for row in reused_rows:
                b_id = str(row.get("bot_id", ""))
                reasons = list(row.get("reasons", [])) + ["cooldown_reuse"]
                if log_sub_bot_decisions:
                    trader.execute_decision(
                        symbol=symbol,
                        action=str(row.get("action", "HOLD")),
                        quantity=1,
                        model_score=float(row.get("score", 0.5)),
                        threshold=float(row.get("threshold", 0.55)),
                        features=shared_features,
                        gates=gates,
                        reasons=reasons + [f"bot_id={b_id}", f"bot_role={row.get('bot_role', 'signal_sub_bot')}"],
                        strategy=b_id,
                        metadata={"layer": "sub_bot", "snapshot_id": snapshot_id, "bot_weight": row.get("weight", 0.0), "test_accuracy": row.get("test_accuracy"), "bot_role": row.get("bot_role", "signal_sub_bot"), "reused": True},
                    )
                sub_rows.append(row)

            for b, action, score, threshold, reasons in batched_rows:
                if log_sub_bot_decisions:
                    trader.execute_decision(
                        symbol=symbol,
                        action=action,
                        quantity=1,
                        model_score=score,
                        threshold=threshold,
                        features=shared_features,
                        gates=gates,
                        reasons=reasons + [f"bot_id={b.bot_id}", f"bot_role={b.bot_role}"],
                        strategy=b.bot_id,
                        metadata={"layer": "sub_bot", "snapshot_id": snapshot_id, "bot_weight": b.weight, "test_accuracy": b.test_accuracy, "bot_role": b.bot_role},
                    )

                row = {
                    "bot_id": b.bot_id,
                    "bot_role": b.bot_role,
                    "action": action,
                    "score": score,
                    "threshold": threshold,
                    "weight": b.weight if b.weight > 0 else 1.0 / max(len(active_bots), 1),
                    "direction": 1.0 if action == "BUY" else (-1.0 if action == "SELL" else 0.0),
                    "reasons": reasons,
                    "test_accuracy": b.test_accuracy,
                }
                sub_rows.append(row)
                slow_bot_rows_cache.setdefault(symbol, {})[b.bot_id] = row

            if (not evaluate_slow_this_iter) and slow_bots:
                for b in slow_bots:
                    row = slow_bot_rows_cache.get(symbol, {}).get(b.bot_id)
                    if row is not None and not any(r.get("bot_id") == b.bot_id for r in sub_rows):
                        reused = dict(row)
                        reused["reused"] = 1
                        sub_rows.append(reused)

            if paper_trader is not None and paper_selected_ids:
                for row in sub_rows:
                    bot_id = str(row.get("bot_id", ""))
                    action = str(row.get("action", "HOLD")).upper()
                    if bot_id not in paper_selected_ids or action not in {"BUY", "SELL"}:
                        continue
                    try:
                        paper_trader.execute_decision(
                            symbol=symbol,
                            action=action,
                            quantity=1,
                            model_score=float(row.get("score", 0.5)),
                            threshold=float(row.get("threshold", 0.55)),
                            features=shared_features,
                            gates=gates,
                            reasons=list(row.get("reasons", [])) + [f"paper_mirror_top_n={paper_mirror_top_n}", f"bot_id={bot_id}"],
                            strategy=f"paper_mirror::{bot_id}",
                            metadata={
                                "layer": "sub_bot_paper_mirror",
                                "snapshot_id": snapshot_id,
                                "source_profile": _shadow_profile_name() or "default",
                                "bot_weight": row.get("weight", 0.0),
                                "test_accuracy": row.get("test_accuracy"),
                            },
                        )
                    except Exception as exc:
                        print(f"[PaperMirror] order_failed symbol={symbol} bot_id={bot_id} err={exc}")

            options_specialist_rows: List[Dict[str, Any]] = []
            futures_specialist_rows: List[Dict[str, Any]] = []

            if options_specialist_bots:
                for b, action, score, threshold, reasons in _specialist_signal_batch(
                    options_specialist_bots,
                    shared_features,
                    segment="options",
                ):
                    row = {
                        "bot_id": b.bot_id,
                        "bot_role": b.bot_role,
                        "action": action,
                        "score": score,
                        "threshold": threshold,
                        "weight": b.weight if b.weight > 0 else 1.0 / max(len(options_specialist_bots), 1),
                        "direction": 1.0 if action == "BUY" else (-1.0 if action == "SELL" else 0.0),
                        "reasons": reasons,
                        "test_accuracy": b.test_accuracy,
                    }
                    options_specialist_rows.append(row)
                    if log_sub_bot_decisions:
                        trader.execute_decision(
                            symbol=symbol,
                            action=action,
                            quantity=1,
                            model_score=score,
                            threshold=threshold,
                            features=shared_features,
                            gates=gates,
                            reasons=reasons + [f"bot_id={b.bot_id}", "bot_role=options_sub_bot"],
                            strategy=b.bot_id,
                            metadata={
                                "layer": "options_sub_bot",
                                "snapshot_id": snapshot_id,
                                "bot_weight": row["weight"],
                                "test_accuracy": b.test_accuracy,
                                "bot_role": "options_sub_bot",
                            },
                        )

            if futures_specialist_bots:
                for b, action, score, threshold, reasons in _specialist_signal_batch(
                    futures_specialist_bots,
                    shared_features,
                    segment="futures",
                ):
                    row = {
                        "bot_id": b.bot_id,
                        "bot_role": b.bot_role,
                        "action": action,
                        "score": score,
                        "threshold": threshold,
                        "weight": b.weight if b.weight > 0 else 1.0 / max(len(futures_specialist_bots), 1),
                        "direction": 1.0 if action == "BUY" else (-1.0 if action == "SELL" else 0.0),
                        "reasons": reasons,
                        "test_accuracy": b.test_accuracy,
                    }
                    futures_specialist_rows.append(row)
                    if log_sub_bot_decisions:
                        trader.execute_decision(
                            symbol=symbol,
                            action=action,
                            quantity=1,
                            model_score=score,
                            threshold=threshold,
                            features=shared_features,
                            gates=gates,
                            reasons=reasons + [f"bot_id={b.bot_id}", "bot_role=futures_sub_bot"],
                            strategy=b.bot_id,
                            metadata={
                                "layer": "futures_sub_bot",
                                "snapshot_id": snapshot_id,
                                "bot_weight": row["weight"],
                                "test_accuracy": b.test_accuracy,
                                "bot_role": "futures_sub_bot",
                            },
                        )

            # Broadcast flash-crash + derivatives specialist context to all higher-level decisions.
            flash_aux = _derive_flash_aux_features(sub_rows)
            specialist_aux = _derive_specialist_aux_features(options_specialist_rows, futures_specialist_rows)
            all_sub_rows = sub_rows + options_specialist_rows + futures_specialist_rows
            shared_features = {
                **shared_features,
                **flash_aux,
                **specialist_aux,
            }
            master_decision_started_at = time.perf_counter()

            master_outputs: Dict[str, Dict[str, Any]] = {}
            for master_name in ("trend", "mean_revert", "shock"):
                m_action, m_score, m_threshold, m_reasons, m_vote = _master_vote_variant(
                    master_name,
                    sub_rows,
                    shared_features,
                )
                master_outputs[master_name] = {
                    "action": m_action,
                    "score": m_score,
                    "threshold": m_threshold,
                    "reasons": m_reasons,
                    "vote": m_vote["vote"],
                }
                if log_master_variant_decisions:
                    trader.execute_decision(
                        symbol=symbol,
                        action=m_action,
                        quantity=1,
                        model_score=m_score,
                        threshold=m_threshold,
                        features={**shared_features, "master_vote": m_vote["vote"], "active_sub_bots": float(active_sub_bots_total)},
                        gates={"ensemble_has_members": active_sub_bots_total > 0, "market_data_ok": mkt["last_price"] > 0},
                        reasons=m_reasons,
                        strategy=f"master_{master_name}_bot",
                        metadata={"layer": "master_bot", "master_name": master_name, "snapshot_id": snapshot_id},
                    )

            gm_weights = _grand_master_weights(shared_features)
            gm_action, gm_score, gm_threshold, gm_reasons, gm_vote = _grand_master_vote(master_outputs, gm_weights)
            gm_action, gm_score, gm_reasons = _apply_transaction_cost_penalty(
                action=gm_action,
                score=gm_score,
                threshold=gm_threshold,
                reasons=gm_reasons,
                features=shared_features,
            )

            behavior_prior, behavior_meta = _behavior_prior_from_model(
                behavior_model,
                symbol=symbol,
                action_hint=gm_action,
                features=shared_features,
            )
            if behavior_bias_enabled and behavior_meta:
                gm_action_pre_behavior = gm_action
                gm_score_pre_behavior = gm_score
                gm_vote_pre_behavior = float(gm_vote.get("vote", 0.0))
                gm_vote["vote"] = _calibrate_vote(float(gm_vote.get("vote", 0.0)) + (behavior_bias_strength * behavior_prior), scale=0.80)
                gm_score = _vote_to_score(gm_vote["vote"])
                if gm_vote["vote"] > 0.24:
                    gm_action = "BUY"
                elif gm_vote["vote"] < -0.20:
                    gm_action = "SELL"
                else:
                    gm_action = "HOLD"

                neutral_conf = float(behavior_meta.get("behavior_prob_neutral", 0.0) or 0.0)
                directional_conf = max(
                    float(behavior_meta.get("behavior_prob_positive", 0.0) or 0.0),
                    float(behavior_meta.get("behavior_prob_negative", 0.0) or 0.0),
                )
                neutral_margin = neutral_conf - directional_conf

                # Require strong neutral confidence before behavior bias is allowed to force HOLD.
                if (
                    gm_action == "HOLD"
                    and (
                        neutral_conf < behavior_hold_neutral_min
                        or neutral_margin < behavior_hold_margin_min
                    )
                ):
                    gm_action = gm_action_pre_behavior
                    gm_score = gm_score_pre_behavior
                    gm_vote["vote"] = gm_vote_pre_behavior
                    gm_reasons = gm_reasons + [
                        f"behavior_bias={behavior_prior:+.4f}",
                        f"neutral_gate_revert neutral={neutral_conf:.3f} margin={neutral_margin:.3f}",
                    ]
                else:
                    gm_reasons = gm_reasons + [
                        f"behavior_bias={behavior_prior:+.4f}",
                        f"neutral_conf={neutral_conf:.3f}",
                        f"neutral_margin={neutral_margin:.3f}",
                    ]
                shared_features = {**shared_features, **behavior_meta}

            if _is_dividend_profile():
                gm_action, gm_score, gm_reasons, div_overlay = _apply_dividend_strategy_overlay(
                    symbol=symbol,
                    action=gm_action,
                    score=gm_score,
                    threshold=gm_threshold,
                    reasons=gm_reasons,
                    features=shared_features,
                )
                if div_overlay:
                    shared_features = {**shared_features, **div_overlay}

                gm_action, gm_score, gm_reasons, div_policy_overlay = _apply_dividend_safety_tax_overlay(
                    symbol=symbol,
                    action=gm_action,
                    score=gm_score,
                    threshold=gm_threshold,
                    reasons=gm_reasons,
                    features=shared_features,
                    iter_count=iter_count,
                    state=dividend_policy_state,
                )
                if div_policy_overlay:
                    shared_features = {**shared_features, **div_policy_overlay}

            options_roll_meta: Dict[str, Any] = {"active": False, "directive": "none"}
            futures_roll_meta: Dict[str, Any] = {"active": False, "directive": "none"}
            options_margin_meta: Dict[str, Any] = {"active": True, "lane": "options", "ok": True, "reason": "not_evaluated", "required_margin_proxy": 0.0, "available_margin_proxy": 0.0}
            futures_margin_meta: Dict[str, Any] = {"active": True, "lane": "futures", "ok": True, "reason": "not_evaluated", "required_margin_proxy": 0.0, "available_margin_proxy": 0.0}
            manual_reconcile_meta: Dict[str, Any] = {"active": False}
            effective_covered_call_shares = int(
                manual_reconcile_meta.get("covered_call_shares_override", covered_call_shares)
                if isinstance(manual_reconcile_meta, dict)
                else covered_call_shares
            )

            if _is_long_term_profile():
                gm_action, gm_score, gm_reasons, long_term_overlay = _apply_long_term_buy_hold_overlay(
                    symbol=symbol,
                    action=gm_action,
                    score=gm_score,
                    threshold=gm_threshold,
                    reasons=gm_reasons,
                    features=shared_features,
                )
                if long_term_overlay:
                    shared_features = {**shared_features, **long_term_overlay}

                gm_action, gm_score, gm_reasons, long_term_alloc_overlay = _apply_long_term_allocation_policy(
                    symbol=symbol,
                    action=gm_action,
                    score=gm_score,
                    threshold=gm_threshold,
                    reasons=gm_reasons,
                    features=shared_features,
                    iter_count=iter_count,
                    state=long_term_policy_state,
                )
                if long_term_alloc_overlay:
                    shared_features = {**shared_features, **long_term_alloc_overlay}

            if _is_day_profile():
                gm_action, gm_score, gm_reasons, day_overlay = _apply_day_strategy_overlay(
                    symbol=symbol,
                    action=gm_action,
                    score=gm_score,
                    threshold=gm_threshold,
                    reasons=gm_reasons,
                    features=shared_features,
                    state=day_strategy_state,
                )
                if day_overlay:
                    shared_features = {**shared_features, **day_overlay}

            if _is_swing_profile():
                gm_action, gm_score, gm_reasons, swing_overlay = _apply_swing_strategy_overlay(
                    symbol=symbol,
                    action=gm_action,
                    score=gm_score,
                    threshold=gm_threshold,
                    reasons=gm_reasons,
                    features=shared_features,
                    state=swing_strategy_state,
                )
                if swing_overlay:
                    shared_features = {**shared_features, **swing_overlay}

            if _is_bond_profile():
                gm_action, gm_score, gm_reasons, bond_overlay = _apply_bond_strategy_overlay(
                    symbol=symbol,
                    action=gm_action,
                    score=gm_score,
                    threshold=gm_threshold,
                    reasons=gm_reasons,
                    features=shared_features,
                )
                if bond_overlay:
                    shared_features = {**shared_features, **bond_overlay}

            gm_action, gm_score, gm_reasons, manual_reconcile_meta = _apply_manual_trade_reconciler(
                symbol=symbol,
                action=gm_action,
                score=gm_score,
                threshold=gm_threshold,
                reasons=gm_reasons,
                manual_payload=manual_trade_overrides if manual_trade_reconcile_enabled else {"symbols": {}},
            )
            if bool(manual_reconcile_meta.get("active", False)):
                shared_features = {
                    **shared_features,
                    "manual_trade_reconcile_active": 1.0,
                    "manual_trade_position_qty_norm": _clamp01(abs(float(manual_reconcile_meta.get("position_qty", 0.0) or 0.0)) / 500.0),
                }

            if iter_count < symbol_regime_cooldown_until_iter.get(symbol, 0):
                gm_action = "HOLD"
                gm_score = 0.5 + 0.5 * (gm_score - 0.5)
                gm_reasons = gm_reasons + ["regime_change_cooldown"]

            gm_intent_action = gm_action
            gm_intent_score = gm_score

            gm_action, gm_score, gm_reasons = _enforce_cross_symbol_exposure_cap(
                action=gm_action,
                score=gm_score,
                reasons=gm_reasons,
                exposure_state=exposure_state,
                max_long=exposure_cap_long,
                max_short=exposure_cap_short,
            )

            pre_risk_action = gm_action
            risk_action, risk_reasons, risk_gates = apply_risk_limits(
                action=gm_action,
                symbol=symbol,
                exposure_state=exposure_state,
                features={
                    "volatility_1m": float(shared_features.get("volatility_1m", shared_features.get("vol", 0.0)) or 0.0),
                    "drawdown_proxy": abs(float(shared_features.get("pct_from_close", 0.0) or 0.0)),
                    "var_proxy": abs(float(shared_features.get("volatility_1m", shared_features.get("vol", 0.0)) or 0.0)) * 1.65,
                    "factor_exposure": abs(float(shared_features.get("mom_5m", 0.0) or 0.0)) + abs(float(shared_features.get("pct_from_close", 0.0) or 0.0)),
                    "daily_loss_proxy": abs(min(intraday_pnl_proxy, 0.0)),
                },
            )
            for risk_gate_name, risk_gate_ok in risk_gates.items():
                _log_gate(
                    symbol,
                    f"risk_{risk_gate_name}",
                    bool(risk_gate_ok),
                    reason=("ok" if bool(risk_gate_ok) else "blocked"),
                    action_before_risk=pre_risk_action,
                )
            risk_action_ok = risk_action == pre_risk_action
            _log_gate(
                symbol,
                "risk_action_block",
                risk_action_ok,
                reason=("ok" if risk_action_ok else f"risk_override:{pre_risk_action}->{risk_action}"),
            )
            if risk_action != pre_risk_action:
                gm_action = risk_action
                gm_score = 0.5 + 0.5 * (gm_score - 0.5)
            if risk_reasons:
                gm_reasons = gm_reasons + risk_reasons

            now_ts = time.time()
            vol_now = float(shared_features.get("volatility_1m", shared_features.get("vol", 0.0)) or 0.0)
            spread_now = float(shared_features.get("spread_bps", 0.0) or 0.0)

            if vol_now >= vol_shock_threshold:
                vol_shock_pause_until_ts = max(vol_shock_pause_until_ts, now_ts + vol_shock_pause_seconds)
            if spread_now >= liquidity_spread_bps_threshold:
                liquidity_pause_until_ts = max(liquidity_pause_until_ts, now_ts + liquidity_pause_seconds)
            if intraday_pnl_proxy <= -abs(max_daily_loss_proxy):
                kill_switch_until_ts = max(kill_switch_until_ts, now_ts + kill_switch_cooldown_seconds)

            intraday_drawdown_proxy = max(peak_intraday_pnl_proxy - intraday_pnl_proxy, 0.0)
            if intraday_drawdown_proxy >= abs(max_intraday_drawdown_proxy):
                kill_switch_until_ts = max(kill_switch_until_ts, now_ts + kill_switch_cooldown_seconds)

            pause_active = (now_ts < kill_switch_until_ts) or (now_ts < vol_shock_pause_until_ts) or (now_ts < liquidity_pause_until_ts)
            pause_gate_ok = (not pause_active) or (gm_action not in {"BUY", "SELL"})
            _log_gate(
                symbol,
                "circuit_pause_guard",
                pause_gate_ok,
                reason=("ok" if pause_gate_ok else "circuit_breaker_pause_active"),
                kill_switch_active=bool(now_ts < kill_switch_until_ts),
                vol_shock_pause_active=bool(now_ts < vol_shock_pause_until_ts),
                liquidity_pause_active=bool(now_ts < liquidity_pause_until_ts),
                intraday_drawdown_proxy=intraday_drawdown_proxy,
                max_intraday_drawdown_proxy=max_intraday_drawdown_proxy,
            )
            if pause_active and gm_action in {"BUY", "SELL"}:
                gm_action = "HOLD"
                gm_score = 0.5 + 0.5 * (gm_score - 0.5)
                gm_reasons = gm_reasons + ["circuit_breaker_pause_active"]

            symbol_pause_active = now_ts < float(symbol_circuit_until.get(symbol, 0.0) or 0.0)
            _log_gate(
                symbol,
                "symbol_circuit_guard",
                (not symbol_pause_active) or (gm_action not in {"BUY", "SELL"}),
                reason=("ok" if not symbol_pause_active else "symbol_circuit_cooldown"),
                symbol_circuit_hits=int(symbol_circuit_hits.get(symbol, 0) or 0),
                symbol_circuit_until=float(symbol_circuit_until.get(symbol, 0.0) or 0.0),
            )
            if symbol_pause_active and gm_action in {"BUY", "SELL"}:
                gm_action = "HOLD"
                gm_score = 0.5 + 0.5 * (gm_score - 0.5)
                gm_reasons = gm_reasons + ["symbol_circuit_cooldown"]

            consecutive_loss_guard_ok = not (consecutive_loss_streak >= max_consecutive_losses and gm_action in {"BUY", "SELL"})
            _log_gate(
                symbol,
                "consecutive_loss_guard",
                consecutive_loss_guard_ok,
                reason=("ok" if consecutive_loss_guard_ok else "consecutive_loss_pause"),
                consecutive_loss_streak=consecutive_loss_streak,
                max_consecutive_losses=max_consecutive_losses,
            )
            if not consecutive_loss_guard_ok:
                kill_switch_until_ts = max(kill_switch_until_ts, now_ts + kill_switch_cooldown_seconds)
                gm_action = "HOLD"
                gm_score = 0.5 + 0.5 * (gm_score - 0.5)
                gm_reasons = gm_reasons + [f"consecutive_loss_pause streak={consecutive_loss_streak}"]

            master_latency_ms = (time.perf_counter() - master_decision_started_at) * 1000.0
            master_latency_slo_ok = (not master_latency_slo_enabled) or (master_latency_ms <= master_latency_slo_timeout_ms)
            _log_gate(
                symbol,
                "master_latency_slo",
                master_latency_slo_ok,
                reason=("ok" if master_latency_slo_ok else "timeout"),
                elapsed_ms=round(float(master_latency_ms), 3),
                timeout_ms=master_latency_slo_timeout_ms,
            )
            if (not master_latency_slo_ok) and gm_action in {"BUY", "SELL"}:
                gm_action = "HOLD"
                gm_score = 0.5 + 0.5 * (gm_score - 0.5)
                gm_reasons = gm_reasons + [
                    f"master_latency_slo_timeout elapsed_ms={master_latency_ms:.1f} timeout_ms={master_latency_slo_timeout_ms:.1f}"
                ]

            freshness_trade_guard_ok = freshness_ok or (gm_action not in {"BUY", "SELL"})
            _log_gate(
                symbol,
                "feature_freshness_trade_guard",
                freshness_trade_guard_ok,
                reason=("ok" if freshness_trade_guard_ok else f"feature_freshness_guard:{freshness_reason}"),
                freshness_ok=bool(freshness_ok),
                freshness_reason=freshness_reason,
            )
            if (not freshness_ok) and gm_action in {"BUY", "SELL"}:
                gm_action = "HOLD"
                gm_score = 0.5 + 0.5 * (gm_score - 0.5)
                gm_reasons = gm_reasons + [f"feature_freshness_guard:{freshness_reason}"]

            gm_action, gm_score, gm_reasons, execution_guard_meta = _apply_execution_guard(
                action=gm_action,
                score=gm_score,
                threshold=gm_threshold,
                reasons=gm_reasons,
                features=shared_features,
                symbol_is_futures=symbol_is_futures,
                broker=broker,
            )
            _log_gate(
                symbol,
                "execution_guard",
                bool(execution_guard_meta.get("ok", True)),
                reason=("ok" if bool(execution_guard_meta.get("ok", True)) else "execution_guard_block"),
                market_kind=str(execution_guard_meta.get("market_kind", "") or ""),
                spread_bps=float(execution_guard_meta.get("spread_bps", 0.0) or 0.0),
                depth_norm=float(execution_guard_meta.get("depth_norm", 0.0) or 0.0),
                tx_cost_bps=float(execution_guard_meta.get("tx_cost_bps", 0.0) or 0.0),
                market_data_latency_ms=float(execution_guard_meta.get("market_data_latency_ms", 0.0) or 0.0),
                adverse_imbalance=float(execution_guard_meta.get("adverse_imbalance", 0.0) or 0.0),
            )

            runtime_lane = _runtime_lane_key(
                symbol=symbol,
                broker=broker,
                profile=_shadow_profile_name() or "default",
                symbol_is_futures=symbol_is_futures,
            )
            lane_budget_mult = _lane_budget_multiplier(runtime_lane)
            lane_base_budget = max(portfolio_base_budget * lane_budget_mult, 0.0)
            runtime_lane_code = {
                "equities": 0.0,
                "day": 1.0,
                "swing": 2.0,
                "options": 3.0,
                "futures": 4.0,
                "long_term": 5.0,
            }.get(runtime_lane, 0.0)

            lane_key = str(runtime_lane or "equities").strip().lower()
            lane_vol_threshold = float(lane_vol_shock_thresholds.get(lane_key, lane_vol_shock_thresholds.get("default", vol_shock_threshold)))
            lane_spread_threshold = float(lane_liquidity_spread_bps_thresholds.get(lane_key, lane_liquidity_spread_bps_thresholds.get("default", liquidity_spread_bps_threshold)))
            lane_cooldown = int(lane_kill_cooldown_seconds.get(lane_key, lane_kill_cooldown_seconds.get("default", kill_switch_cooldown_seconds)))
            lane_cooldown = max(lane_cooldown, 30)
            lane_trigger_reasons: List[str] = []
            if vol_now >= lane_vol_threshold:
                lane_trigger_reasons.append(f"vol_shock={vol_now:.4f}>={lane_vol_threshold:.4f}")
            if spread_now >= lane_spread_threshold:
                lane_trigger_reasons.append(f"spread_bps={spread_now:.2f}>={lane_spread_threshold:.2f}")

            lane_prev_until = float(lane_kill_until_ts.get(lane_key, 0.0) or 0.0)
            if lane_trigger_reasons:
                lane_new_until = max(lane_prev_until, now_ts + lane_cooldown)
                lane_kill_until_ts[lane_key] = lane_new_until
                if lane_new_until > lane_prev_until + 1e-6:
                    _emit_critical_alert(
                        project_root=PROJECT_ROOT,
                        broker=broker,
                        event="lane_kill_switch_engaged",
                        message=f"lane={lane_key} cooldown={lane_cooldown}s",
                        details={
                            "symbol": symbol,
                            "iter": int(iter_count),
                            "lane": lane_key,
                            "reasons": lane_trigger_reasons,
                            "cooldown_seconds": int(lane_cooldown),
                        },
                        severity="critical",
                    )

            lane_pause_until = float(lane_kill_until_ts.get(lane_key, 0.0) or 0.0)
            lane_pause_active = now_ts < lane_pause_until
            _log_gate(
                symbol,
                "lane_kill_switch_guard",
                (not lane_pause_active) or (gm_action not in {"BUY", "SELL"}),
                reason=("ok" if not lane_pause_active else f"lane_kill_switch_active:{lane_key}"),
                lane=lane_key,
                lane_pause_until=lane_pause_until,
                lane_loss_streak=int(lane_loss_streak.get(lane_key, 0) or 0),
            )
            if lane_pause_active and gm_action in {"BUY", "SELL"}:
                lane_pause_counts[lane_key] = int(lane_pause_counts.get(lane_key, 0) or 0) + 1
                gm_action = "HOLD"
                gm_score = 0.5 + 0.5 * (gm_score - 0.5)
                gm_reasons = gm_reasons + [f"lane_kill_switch_pause lane={lane_key}"]

            volatility_now = float(shared_features.get("volatility_1m", shared_features.get("vol", 0.0)) or 0.0)
            raw_qty = size_from_action(
                action=gm_action,
                score=gm_score,
                threshold=gm_threshold,
                volatility_1m=volatility_now,
                equity_proxy=effective_equity_proxy,
                max_notional_pct=max_notional_pct,
            )
            alloc_qty = allocate_quantity(
                raw_qty=raw_qty,
                symbol=symbol,
                score=gm_score,
                volatility_1m=volatility_now,
                base_budget=lane_base_budget,
                symbol_budgets=symbol_budgets,
            )

            alloc_qty, portfolio_risk_meta = _apply_portfolio_risk_engine_caps(
                symbol=symbol,
                broker=broker,
                lane=runtime_lane,
                action=gm_action,
                qty=alloc_qty,
                last_price=float(mkt.get("last_price", 0.0) or 0.0),
                equity_proxy=effective_equity_proxy,
                state=portfolio_risk_state,
                sector_map=portfolio_sector_map,
            )

            alloc_qty, long_term_turnover_meta = _apply_long_term_turnover_cap(
                qty=alloc_qty,
                last_price=float(mkt.get("last_price", 0.0) or 0.0),
                equity_proxy=effective_equity_proxy,
                state=long_term_policy_state,
            )

            if gm_action in {"BUY", "SELL"} and alloc_qty <= 0.0:
                gm_action = "HOLD"
                gm_score = 0.5 + 0.5 * (gm_score - 0.5)
                gm_reasons = gm_reasons + ["portfolio_risk_engine_qty_capped_to_zero"]

            idempotency_key = ""
            idempotency_meta: Dict[str, Any] = {
                "active": bool(order_idempotency_enabled),
                "ok": True,
                "duplicate": False,
                "reserved": False,
                "window_seconds": float(order_idempotency_window_seconds),
                "key": "",
            }
            if order_idempotency_enabled and gm_action in {"BUY", "SELL"} and alloc_qty > 0.0:
                idempotency_key = _order_intent_key(
                    symbol=symbol,
                    action=gm_action,
                    lane=lane_key,
                    qty=alloc_qty,
                    price=float(mkt.get("last_price", 0.0) or 0.0),
                    strategy="grand_master_bot",
                )
                idem_ok, idem_meta = _reserve_order_intent(
                    state=order_idempotency_state,
                    key=idempotency_key,
                    now_ts=now_ts,
                    window_seconds=order_idempotency_window_seconds,
                    payload={
                        "symbol": symbol,
                        "action": gm_action,
                        "lane": lane_key,
                        "qty": float(alloc_qty),
                        "price": float(mkt.get("last_price", 0.0) or 0.0),
                        "iter": int(iter_count),
                        "snapshot_id": snapshot_id,
                    },
                )
                idempotency_meta = {
                    "active": True,
                    **idem_meta,
                }
                if idem_ok:
                    order_idempotency_dirty = True
                if not idem_ok:
                    gm_action = "HOLD"
                    gm_score = 0.5 + 0.5 * (gm_score - 0.5)
                    alloc_qty = 0.0
                    gm_reasons = gm_reasons + [
                        (
                            "order_idempotency_guard duplicate_intent "
                            f"age_s={float(idem_meta.get('age_seconds', 0.0) or 0.0):.2f} "
                            f"window_s={float(idem_meta.get('window_seconds', 0.0) or 0.0):.2f}"
                        )
                    ]
                    _emit_critical_alert(
                        project_root=PROJECT_ROOT,
                        broker=broker,
                        event="order_idempotency_guard",
                        message="duplicate_trade_intent_blocked",
                        details={
                            "symbol": symbol,
                            "iter": int(iter_count),
                            "lane": lane_key,
                            "action": gm_action,
                            "key": str(idem_meta.get("key", "")),
                        },
                        severity="warn",
                    )
            _log_gate(
                symbol,
                "order_idempotency_guard",
                bool(idempotency_meta.get("ok", True)),
                reason=("ok" if bool(idempotency_meta.get("ok", True)) else "duplicate_intent"),
                idempotency_key=str(idempotency_meta.get("key", "")),
                window_seconds=float(idempotency_meta.get("window_seconds", 0.0) or 0.0),
                duplicate=bool(idempotency_meta.get("duplicate", False)),
            )

            req = OrderRequest(
                symbol=symbol,
                action=gm_action,
                quantity=alloc_qty,
                priority=_queue_priority(gm_action, gm_score),
                metadata={
                    "snapshot_id": snapshot_id,
                    "strategy": "grand_master_bot",
                    "runtime_lane": runtime_lane,
                    "lane_budget_mult": lane_budget_mult,
                },
            )
            enq_ok = exec_queue.enqueue(req)
            if (not enq_ok) and bool(idempotency_meta.get("reserved", False)) and idempotency_key:
                released = _release_order_intent(order_idempotency_state, idempotency_key)
                if released:
                    order_idempotency_dirty = True
            dispatch = exec_queue.pop() if enq_ok else None
            dispatch_qty = dispatch.quantity if dispatch is not None else 0.0
            _log_gate(
                symbol,
                "exec_queue",
                bool(enq_ok),
                reason=("ok" if bool(enq_ok) else "queue_overflow"),
                queue_depth=exec_queue.size(),
            )

            grand_features = {
                **shared_features,
                "grand_master_vote": gm_vote["vote"],
                "active_sub_bots": float(active_sub_bots_total),
                "active_options_sub_bots": float(active_options_sub_bots),
                "active_futures_sub_bots": float(active_futures_sub_bots),
                "sized_qty": dispatch_qty,
                "runtime_lane_code": runtime_lane_code,
                "lane_budget_mult": lane_budget_mult,
                "manual_trade_reconcile_active": 1.0 if bool(manual_reconcile_meta.get("active", False)) else 0.0,
            }
            grand_gates = {
                "ensemble_has_members": active_sub_bots_total > 0,
                "market_data_ok": mkt["last_price"] > 0,
                "risk_limit_ok": all(risk_gates.values()),
                "feature_freshness_ok": freshness_ok,
                "master_latency_slo_ok": master_latency_slo_ok,
                "execution_guard_ok": bool(execution_guard_meta.get("ok", True)),
                "portfolio_risk_engine_ok": not bool(portfolio_risk_meta.get("blocked", False)),
                "long_term_turnover_ok": not bool(long_term_turnover_meta.get("blocked", False)),
                "order_idempotency_ok": bool(idempotency_meta.get("ok", True)),
                "options_margin_ok": bool(options_margin_meta.get("ok", True)),
                "futures_margin_ok": bool(futures_margin_meta.get("ok", True)),
                "broker_truth_ok": bool(broker_truth_state.get("ok", True)),
                **risk_gates,
                "exec_queue_ok": enq_ok,
            }
            guard_blocked_intent = gm_intent_action in {"BUY", "SELL"} and gm_action != gm_intent_action
            if guard_blocked_intent:
                hits = int(symbol_circuit_hits.get(symbol, 0) or 0) + 1
                symbol_circuit_hits[symbol] = hits
                if hits >= symbol_circuit_hit_limit:
                    symbol_circuit_until[symbol] = max(
                        float(symbol_circuit_until.get(symbol, 0.0) or 0.0),
                        time.time() + symbol_circuit_cooldown_seconds,
                    )
            else:
                if gm_action in {"BUY", "SELL"}:
                    symbol_circuit_hits[symbol] = max(int(symbol_circuit_hits.get(symbol, 0) or 0) - 1, 0)

            if guard_blocked_intent and log_grand_master_decisions:
                trader.execute_decision(
                    symbol=symbol,
                    action=gm_intent_action,
                    quantity=0.0,
                    model_score=gm_intent_score,
                    threshold=gm_threshold,
                    features={
                        **grand_features,
                        "intent_only": 1,
                        "final_action": gm_action,
                    },
                    gates={
                        **grand_gates,
                        "intent_matches_final": False,
                    },
                    reasons=gm_reasons + [f"intent_logged final_action={gm_action}"],
                    strategy="grand_master_intent_bot",
                    metadata={
                        "layer": "grand_master",
                        "snapshot_id": snapshot_id,
                        "master_weights": gm_weights,
                        "queue_depth": exec_queue.size(),
                        "intent_action": gm_intent_action,
                        "intent_score": gm_intent_score,
                        "final_action": gm_action,
                        "intent_only": True,
                    },
                )

            if log_grand_master_decisions:
                trader.execute_decision(
                    symbol=symbol,
                    action=gm_action,
                    quantity=dispatch_qty,
                    model_score=gm_score,
                    threshold=gm_threshold,
                    features=grand_features,
                    gates=grand_gates,
                    reasons=gm_reasons,
                    strategy="grand_master_bot",
                    metadata={
                        "layer": "grand_master",
                        "snapshot_id": snapshot_id,
                        "master_weights": gm_weights,
                        "queue_depth": exec_queue.size(),
                        "intent_action": gm_intent_action,
                        "intent_score": gm_intent_score,
                        "guard_blocked_intent": guard_blocked_intent,
                    },
                )

            if _is_long_term_profile():
                optm_action = "HOLD"
                optm_score = 0.5 + 0.35 * (gm_score - 0.5)
                optm_threshold = _shift_threshold(0.60)
                optm_reasons = ["long_term_profile_options_master_disabled"]
                optm_vote = {"vote": 0.0}
                options_decision = {
                    "action": "HOLD",
                    "score": optm_score,
                    "threshold": optm_threshold,
                    "reasons": ["long_term_profile_options_plan_disabled"],
                    "plan": {
                        "symbol": symbol,
                        "options_style": "NONE",
                        "underlying_price": mkt.get("last_price", 0.0),
                        "dte_days": 0,
                        "contracts": 0,
                        "strike": None,
                        "master_vote": 0.0,
                    },
                }

                futm_action = "HOLD"
                futm_score = 0.5 + 0.30 * (gm_score - 0.5)
                futm_threshold = _shift_threshold(0.60)
                futm_reasons = ["long_term_profile_futures_master_disabled"]
                futm_vote = {"vote": 0.0}
                futures_decision = {
                    "action": "HOLD",
                    "score": futm_score,
                    "threshold": futm_threshold,
                    "reasons": ["long_term_profile_futures_plan_disabled"],
                    "plan": {
                        "symbol": symbol,
                        "futures_style": "NONE",
                        "contracts": 0,
                        "legs": [],
                        "master_vote": 0.0,
                    },
                }
            else:
                optm_action, optm_score, optm_threshold, optm_reasons, optm_vote = _options_master_signal(
                    grand_action=gm_action,
                    grand_score=gm_score,
                    grand_vote=gm_vote["vote"],
                    features=shared_features,
                )
                optm_action, optm_score, optm_reasons = _apply_transaction_cost_penalty(
                    action=optm_action,
                    score=optm_score,
                    threshold=optm_threshold,
                    reasons=optm_reasons,
                    features=shared_features,
                )
                if log_options_master_decisions:
                    trader.execute_decision(
                        symbol=symbol,
                        action=optm_action,
                        quantity=1,
                        model_score=optm_score,
                        threshold=optm_threshold,
                        features={**shared_features, "grand_master_vote": gm_vote["vote"], "grand_master_score": gm_score, "options_master_vote": optm_vote["vote"]},
                        gates={"market_data_ok": mkt["last_price"] > 0, "options_regime_ok": True},
                        reasons=optm_reasons,
                        strategy="options_master_bot",
                        metadata={"layer": "options_master", "snapshot_id": snapshot_id},
                    )

                if overload_mode and skip_options_on_backpressure:
                    options_decision = {
                        "action": "HOLD",
                        "score": optm_score,
                        "threshold": 0.58,
                        "reasons": ["backpressure_overload_skip_options"],
                        "plan": {
                            "symbol": symbol,
                            "options_style": "NONE",
                            "underlying_price": mkt.get("last_price", 0.0),
                            "dte_days": 0,
                            "contracts": 0,
                            "strike": None,
                            "master_vote": optm_vote["vote"],
                        },
                    }
                else:
                    options_decision = _build_options_plan(
                        symbol=symbol,
                        mkt=shared_features,
                        master_action=optm_action,
                        master_score=optm_score,
                        master_vote=optm_vote["vote"],
                        covered_call_shares=effective_covered_call_shares,
                    )
                options_decision, options_roll_meta = _options_roll_manager(
                    decision=options_decision,
                    features=shared_features,
                )
                options_decision, options_margin_meta = _apply_derivatives_margin_guard(
                    decision=options_decision,
                    lane="options",
                    broker_truth=broker_truth_state,
                    features=shared_features,
                )
                options_margin_ok = bool(options_margin_meta.get("ok", True))
                _log_gate(
                    symbol,
                    "options_margin_guard",
                    options_margin_ok,
                    reason=("ok" if options_margin_ok else str(options_margin_meta.get("reason", "margin_guard_block"))),
                    required_margin_proxy=float(options_margin_meta.get("required_margin_proxy", 0.0) or 0.0),
                    available_margin_proxy=float(options_margin_meta.get("available_margin_proxy", 0.0) or 0.0),
                )
                if not options_margin_ok:
                    _emit_critical_alert(
                        project_root=PROJECT_ROOT,
                        broker=broker,
                        event="options_margin_guard",
                        message="options_margin_headroom_insufficient",
                        details={
                            "symbol": symbol,
                            "iter": int(iter_count),
                            "required_margin_proxy": float(options_margin_meta.get("required_margin_proxy", 0.0) or 0.0),
                            "available_margin_proxy": float(options_margin_meta.get("available_margin_proxy", 0.0) or 0.0),
                        },
                        severity="warn",
                    )
                if log_options_master_decisions:
                    trader.execute_decision(
                        symbol=symbol,
                        action=options_decision["action"],
                        quantity=float(options_decision["plan"].get("contracts", 0) or 0),
                        model_score=options_decision["score"],
                        threshold=options_decision["threshold"],
                        features={**shared_features, "options_master_vote": optm_vote["vote"], "options_master_score": optm_score},
                        gates={"market_data_ok": mkt["last_price"] > 0, "options_plan_ready": True},
                        reasons=options_decision["reasons"],
                        strategy="master_options_bot",
                        metadata={"layer": "master_options", "snapshot_id": snapshot_id, "options_plan": options_decision["plan"]},
                    )

                futm_action, futm_score, futm_threshold, futm_reasons, futm_vote = _futures_master_signal(
                    grand_action=gm_action,
                    grand_score=gm_score,
                    grand_vote=gm_vote["vote"],
                    features=shared_features,
                )
                futm_action, futm_score, futm_reasons = _apply_transaction_cost_penalty(
                    action=futm_action,
                    score=futm_score,
                    threshold=futm_threshold,
                    reasons=futm_reasons,
                    features=shared_features,
                )

                if log_futures_master_decisions:
                    trader.execute_decision(
                        symbol=symbol,
                        action=futm_action,
                        quantity=1,
                        model_score=futm_score,
                        threshold=futm_threshold,
                        features={**shared_features, "grand_master_vote": gm_vote["vote"], "grand_master_score": gm_score, "futures_master_vote": futm_vote["vote"]},
                        gates={"market_data_ok": mkt["last_price"] > 0, "futures_regime_ok": symbol_is_futures},
                        reasons=futm_reasons,
                        strategy="futures_master_bot",
                        metadata={"layer": "futures_master", "snapshot_id": snapshot_id},
                    )

                if overload_mode and skip_futures_on_backpressure:
                    futures_decision = {
                        "action": "HOLD",
                        "score": futm_score,
                        "threshold": futm_threshold,
                        "reasons": ["backpressure_overload_skip_futures"],
                        "plan": {
                            "symbol": symbol,
                            "futures_style": "NONE",
                            "contracts": 0,
                            "legs": [],
                            "master_vote": futm_vote["vote"],
                        },
                    }
                else:
                    futures_decision = _build_futures_plan(
                        symbol=symbol,
                        mkt=shared_features,
                        master_action=futm_action,
                        master_score=futm_score,
                        master_vote=futm_vote["vote"],
                        symbol_is_futures=symbol_is_futures,
                    )
                futures_decision, futures_roll_meta = _futures_roll_manager(
                    decision=futures_decision,
                    features=shared_features,
                )
                futures_decision, futures_margin_meta = _apply_derivatives_margin_guard(
                    decision=futures_decision,
                    lane="futures",
                    broker_truth=broker_truth_state,
                    features=shared_features,
                )
                futures_margin_ok = bool(futures_margin_meta.get("ok", True))
                _log_gate(
                    symbol,
                    "futures_margin_guard",
                    futures_margin_ok,
                    reason=("ok" if futures_margin_ok else str(futures_margin_meta.get("reason", "margin_guard_block"))),
                    required_margin_proxy=float(futures_margin_meta.get("required_margin_proxy", 0.0) or 0.0),
                    available_margin_proxy=float(futures_margin_meta.get("available_margin_proxy", 0.0) or 0.0),
                )
                if not futures_margin_ok:
                    _emit_critical_alert(
                        project_root=PROJECT_ROOT,
                        broker=broker,
                        event="futures_margin_guard",
                        message="futures_margin_headroom_insufficient",
                        details={
                            "symbol": symbol,
                            "iter": int(iter_count),
                            "required_margin_proxy": float(futures_margin_meta.get("required_margin_proxy", 0.0) or 0.0),
                            "available_margin_proxy": float(futures_margin_meta.get("available_margin_proxy", 0.0) or 0.0),
                        },
                        severity="warn",
                    )

                if log_futures_master_decisions:
                    trader.execute_decision(
                        symbol=symbol,
                        action=futures_decision["action"],
                        quantity=float(futures_decision["plan"].get("contracts", 0) or 0),
                        model_score=futures_decision["score"],
                        threshold=futures_decision["threshold"],
                        features={**shared_features, "futures_master_vote": futm_vote["vote"], "futures_master_score": futm_score},
                        gates={"market_data_ok": mkt["last_price"] > 0, "futures_plan_ready": symbol_is_futures},
                        reasons=futures_decision["reasons"],
                        strategy="master_futures_bot",
                        metadata={"layer": "master_futures", "snapshot_id": snapshot_id, "futures_plan": futures_decision["plan"]},
                    )

            ret_1m = float(symbol_return_1m)
            exec_sim = simulate_execution(
                action=gm_action,
                last_price=float(mkt.get("last_price", 0.0) or 0.0),
                return_1m=ret_1m,
                spread_bps=float(shared_features.get("spread_bps", 8.0) or 8.0),
                volatility_1m=float(shared_features.get("volatility_1m", shared_features.get("vol", 0.0)) or 0.0),
                latency_ms=float(shared_features.get("market_data_latency_ms", 0.0) or os.getenv("EXEC_SIM_LATENCY_MS", "120")),
                bid_size=float(shared_features.get("bid_size", 1000.0) or 1000.0),
                ask_size=float(shared_features.get("ask_size", 1000.0) or 1000.0),
                order_size=dispatch_qty if dispatch_qty > 0 else 1.0,
                broker=broker,
                market_kind=("crypto" if broker == "coinbase" else "equities"),
                symbol=symbol,
            )
            expected_fill_delta_bps = 0.0
            live_last_price = float(mkt.get("last_price", 0.0) or 0.0)
            if live_last_price > 0.0 and float(exec_sim.expected_fill_price or 0.0) > 0.0:
                expected_fill_delta_bps = (
                    (float(exec_sim.expected_fill_price) - live_last_price) / live_last_price
                ) * 10000.0
            execution_lag_by_symbol[symbol] = {
                "lag_slippage_bps": float(exec_sim.slippage_bps),
                "lag_latency_ms": float(exec_sim.latency_ms),
                "lag_impact_bps": float(exec_sim.impact_bps),
                "lag_fee_bps": float(exec_sim.fee_bps),
                "lag_expected_fill_delta_bps": float(expected_fill_delta_bps),
                "lag_adjusted_return_1m": float(exec_sim.adjusted_return_1m),
                "lag_trade_action_norm": (
                    1.0 if gm_action == "BUY" else (-1.0 if gm_action == "SELL" else 0.0)
                ),
            }
            for row in all_sub_rows:
                role = str(row.get("bot_role", "signal_sub_bot")).strip().lower()
                layer = "sub_bot"
                if role == "options_sub_bot":
                    layer = "options_sub_bot"
                elif role == "futures_sub_bot":
                    layer = "futures_sub_bot"
                _append_jsonl(
                    _pnl_attribution_path(PROJECT_ROOT, broker=broker),
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "snapshot_id": snapshot_id,
                        "symbol": symbol,
                        "bot_id": row.get("bot_id"),
                        "layer": layer,
                        "action": row.get("action"),
                        "direction": row.get("direction"),
                        "weight": row.get("weight"),
                        "return_1m": ret_1m,
                        "pnl_proxy": float(row.get("direction", 0.0)) * ret_1m * float(row.get("weight", 0.0)),
                    },
                )
            _append_jsonl(
                _pnl_attribution_path(PROJECT_ROOT, broker=broker),
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "snapshot_id": snapshot_id,
                    "symbol": symbol,
                    "bot_id": "grand_master_bot",
                    "layer": "grand_master",
                    "action": gm_action,
                    "quantity": dispatch_qty,
                    "return_1m": ret_1m,
                    "pnl_proxy": exec_sim.adjusted_return_1m,
                    "slippage_bps": exec_sim.slippage_bps,
                    "latency_ms": exec_sim.latency_ms,
                    "expected_fill_price": exec_sim.expected_fill_price,
                    "fee_bps": exec_sim.fee_bps,
                },
            )

            intraday_pnl_proxy += float(exec_sim.adjusted_return_1m)
            peak_intraday_pnl_proxy = max(peak_intraday_pnl_proxy, intraday_pnl_proxy)
            if exec_sim.adjusted_return_1m < 0:
                consecutive_loss_streak = (consecutive_loss_streak + 1) if gm_action in {"BUY", "SELL"} else consecutive_loss_streak
            else:
                consecutive_loss_streak = 0

            lane_for_result = str(runtime_lane or "equities").strip().lower()
            lane_max_losses = int(lane_max_consecutive_losses.get(lane_for_result, lane_max_consecutive_losses.get("default", max_consecutive_losses)))
            lane_max_losses = max(lane_max_losses, 1)
            if gm_action in {"BUY", "SELL"}:
                if exec_sim.adjusted_return_1m < 0:
                    lane_loss_streak[lane_for_result] = int(lane_loss_streak.get(lane_for_result, 0) or 0) + 1
                else:
                    lane_loss_streak[lane_for_result] = max(int(lane_loss_streak.get(lane_for_result, 0) or 0) - 1, 0)

                if int(lane_loss_streak.get(lane_for_result, 0) or 0) >= lane_max_losses:
                    lane_cd = int(lane_kill_cooldown_seconds.get(lane_for_result, lane_kill_cooldown_seconds.get("default", kill_switch_cooldown_seconds)))
                    lane_cd = max(lane_cd, 30)
                    prev_until = float(lane_kill_until_ts.get(lane_for_result, 0.0) or 0.0)
                    new_until = max(prev_until, time.time() + lane_cd)
                    lane_kill_until_ts[lane_for_result] = new_until
                    if new_until > prev_until + 1e-6:
                        _emit_critical_alert(
                            project_root=PROJECT_ROOT,
                            broker=broker,
                            event="lane_consecutive_loss_pause",
                            message=f"lane={lane_for_result} streak={int(lane_loss_streak.get(lane_for_result, 0) or 0)}",
                            details={
                                "symbol": symbol,
                                "iter": int(iter_count),
                                "lane": lane_for_result,
                                "streak": int(lane_loss_streak.get(lane_for_result, 0) or 0),
                                "max_streak": int(lane_max_losses),
                                "cooldown_seconds": int(lane_cd),
                            },
                            severity="critical",
                        )

            broker_truth_snapshot = (
                json.loads(json.dumps(broker_truth_state, ensure_ascii=True))
                if isinstance(broker_truth_state, dict)
                else {}
            )

            recs = _governance_recommendations(bots)
            latest_recs = recs
            gov_row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "snapshot_id": snapshot_id,
                "market": mkt,
                "context_market": context_market,
                "active_sub_bots": int(active_sub_bots_total),
                "active_sub_bots_core": len([b for b in bots if b.active]),
                "active_options_sub_bots": int(active_options_sub_bots),
                "active_futures_sub_bots": int(active_futures_sub_bots),
                "inactive_sub_bots": len([b for b in bots if not b.active]),
                "master_action": gm_action,
                "master_score": gm_score,
                "master_vote": gm_vote["vote"],
                "master_intent_action": gm_intent_action,
                "master_intent_score": gm_intent_score,
                "master_guard_blocked_intent": guard_blocked_intent,
                "master_outputs": master_outputs,
                "grand_master_weights": gm_weights,
                "options_master": {"action": optm_action, "score": optm_score, "vote": optm_vote["vote"]},
                "futures_master": {"action": futm_action, "score": futm_score, "vote": futm_vote["vote"]},
                "flash_aux": flash_aux,
                "specialist_votes": {"options": float(shared_features.get("options_specialist_vote", 0.0) or 0.0), "futures": float(shared_features.get("futures_specialist_vote", 0.0) or 0.0)},
                "specialist_rows": {"options": options_specialist_rows, "futures": futures_specialist_rows},
                "news_features": {k: float(shared_features.get(k, 0.0) or 0.0) for k in _NEWS_FEATURE_KEYS},
                "options_chain_features": {k: float(shared_features.get(k, 0.0) or 0.0) for k in OPTIONS_FEATURE_KEYS},
                "futures_features": {k: float(shared_features.get(k, 0.0) or 0.0) for k in FUTURES_FEATURE_KEYS},
                "calendar_features": {k: float(shared_features.get(k, 0.0) or 0.0) for k in CALENDAR_FEATURE_KEYS},
                "dividend_features": {k: float(shared_features.get(k, 0.0) or 0.0) for k in _DIVIDEND_FEATURE_KEYS},
                "long_term_features": {k: float(shared_features.get(k, 0.0) or 0.0) for k in _LONG_TERM_FEATURE_KEYS},
                "lane_strategy_features": {k: float(shared_features.get(k, 0.0) or 0.0) for k in _LANE_STRATEGY_FEATURE_KEYS},
                "breadth_features": {k: float(shared_features.get(k, 0.0) or 0.0) for k in BREADTH_FEATURE_KEYS},
                "bond_reference_features": {k: float(shared_features.get(k, 0.0) or 0.0) for k in BOND_REFERENCE_FEATURE_KEYS},
                "credit_context_features": {k: float(shared_features.get(k, 0.0) or 0.0) for k in CREDIT_CONTEXT_FEATURE_KEYS},
                "data_quality_features": {k: float(shared_features.get(k, 0.0) or 0.0) for k in DATA_QUALITY_FEATURE_KEYS},
                "execution_lag_features": {k: float(shared_features.get(k, 0.0) or 0.0) for k in EXECUTION_LAG_FEATURE_KEYS},
                "options_plan": options_decision["plan"],
                "futures_plan": futures_decision["plan"],
                "options_roll_manager": options_roll_meta,
                "futures_roll_manager": futures_roll_meta,
                "options_margin_guard": options_margin_meta,
                "futures_margin_guard": futures_margin_meta,
                "manual_trade_reconcile": manual_reconcile_meta,
                "broker_truth_reconcile": broker_truth_snapshot,
                "capital_flow": {
                    "detected": bool(capital_flow_meta.get("detected", False)),
                    "estimated_amount": float(capital_flow_meta.get("estimated_amount", 0.0) or 0.0),
                    "ratio_to_equity": float(capital_flow_meta.get("ratio_to_equity", 0.0) or 0.0),
                    "cash_balance_delta": float(capital_flow_meta.get("cash_balance_delta", 0.0) or 0.0),
                    "equity_delta": float(capital_flow_meta.get("equity_delta", 0.0) or 0.0),
                    "agreement_norm": float(capital_flow_meta.get("agreement_norm", 0.0) or 0.0),
                    **{k: float(shared_features.get(k, 0.0) or 0.0) for k in _CAPITAL_FLOW_FEATURE_KEYS},
                },
                "order_idempotency": idempotency_meta,
                "execution_guard": execution_guard_meta,
                "portfolio_risk_engine": portfolio_risk_meta,
                "long_term_turnover_policy": long_term_turnover_meta,
                "lane_allocator": {"lane": runtime_lane, "lane_budget_mult": lane_budget_mult, "lane_base_budget": lane_base_budget},
                "execution_sim": {"slippage_bps": exec_sim.slippage_bps, "latency_ms": exec_sim.latency_ms, "expected_fill_price": exec_sim.expected_fill_price, "impact_bps": exec_sim.impact_bps, "fee_bps": exec_sim.fee_bps},
                "feature_freshness": {
                    "enabled": feature_freshness_enabled,
                    "ok": freshness_ok,
                    "reason": freshness_reason,
                    "age_seconds": round(float(freshness_age_s), 4),
                    "max_age_seconds": feature_freshness_max_age_seconds,
                },
                "master_latency_slo": {
                    "enabled": master_latency_slo_enabled,
                    "ok": master_latency_slo_ok,
                    "elapsed_ms": round(float(master_latency_ms), 3),
                    "timeout_ms": master_latency_slo_timeout_ms,
                },
                "portfolio": {
                    "equity_proxy": effective_equity_proxy,
                    "configured_equity_proxy": configured_equity_proxy,
                    "effective_equity_source": str(effective_equity_meta.get("source", "")),
                    "broker_equity": float(effective_equity_meta.get("broker_equity", 0.0) or 0.0),
                    "broker_truth_age_iters": int(effective_equity_meta.get("age_iters", 0) or 0),
                    "raw_qty": raw_qty,
                    "alloc_qty": alloc_qty,
                    "dispatch_qty": dispatch_qty,
                    "queue_depth": exec_queue.size(),
                    "runtime_lane": runtime_lane,
                    "lane_budget_mult": lane_budget_mult,
                },
                "circuit_breakers": {
                    "consecutive_loss_streak": consecutive_loss_streak,
                    "kill_switch_active": time.time() < kill_switch_until_ts,
                    "vol_shock_pause_active": time.time() < vol_shock_pause_until_ts,
                    "liquidity_pause_active": time.time() < liquidity_pause_until_ts,
                    "symbol_circuit_hits": int(symbol_circuit_hits.get(symbol, 0) or 0),
                    "symbol_circuit_active": time.time() < float(symbol_circuit_until.get(symbol, 0.0) or 0.0),
                    "lane": lane_key,
                    "lane_loss_streak": int(lane_loss_streak.get(lane_key, 0) or 0),
                    "lane_kill_switch_active": time.time() < float(lane_kill_until_ts.get(lane_key, 0.0) or 0.0),
                    "lane_kill_switch_until": float(lane_kill_until_ts.get(lane_key, 0.0) or 0.0),
                    "lane_pause_count": int(lane_pause_counts.get(lane_key, 0) or 0),
                    "intraday_drawdown_proxy": max(peak_intraday_pnl_proxy - intraday_pnl_proxy, 0.0),
                },
                "recommendations": recs,
                "top_active": registry.get("summary", {}).get("top_active", []),
            }
            _append_jsonl(_governance_path(PROJECT_ROOT, broker=broker), gov_row)
            _append_jsonl(_event_bus_path(PROJECT_ROOT), {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "decision_made",
                "symbol": symbol,
                "snapshot_id": snapshot_id,
                "broker": broker,
                "shadow_profile": _shadow_profile_name() or "default",
                "action": gm_action,
                "intent_action": gm_intent_action,
                "intent_blocked": guard_blocked_intent,
                "score": gm_score,
                "risk_limit_ok": all(risk_gates.values()),
                "execution_guard_ok": bool(execution_guard_meta.get("ok", True)),
                "portfolio_risk_engine_ok": not bool(portfolio_risk_meta.get("blocked", False)),
                "options_margin_ok": bool(options_margin_meta.get("ok", True)),
                "futures_margin_ok": bool(futures_margin_meta.get("ok", True)),
                "order_idempotency_ok": bool(idempotency_meta.get("ok", True)),
                "broker_truth_ok": bool(broker_truth_state.get("ok", True)),
                "runtime_lane": lane_key,
                "lane_kill_switch_active": time.time() < float(lane_kill_until_ts.get(lane_key, 0.0) or 0.0),
                "slippage_bps": exec_sim.slippage_bps,
            })

            intent_suffix = f" intent_action={gm_intent_action}" if guard_blocked_intent else ""
            print(
                f"[ShadowLoop] iter={iter_count} symbol={symbol} price={mkt['last_price']:.2f} "
                f"grand_action={gm_action}{intent_suffix} options_master={optm_action} options_action={options_decision['action']} futures_master={futm_action} futures_action={futures_decision['action']} "
                f"active_bots={active_sub_bots_total} core={len(active_bots)} "
                f"opts={active_options_sub_bots} fut={active_futures_sub_bots} "
                f"recs={len(recs)} snapshot_id={snapshot_id}"
            )

        _publish_ingress_state()

        if order_idempotency_enabled:
            pruned = _prune_order_idempotency_registry(
                order_idempotency_state,
                now_ts=time.time(),
                ttl_seconds=order_idempotency_ttl_seconds,
                max_entries=order_idempotency_max_entries,
            )
            if pruned > 0:
                order_idempotency_dirty = True
            if order_idempotency_dirty:
                _save_order_idempotency_registry(
                    order_idempotency_registry_path,
                    order_idempotency_state,
                    PROJECT_ROOT,
                )
                order_idempotency_dirty = False

        if auto_retrain and latest_recs:
            underperformers = _count_underperformers(latest_recs)
            cooldown_seconds = max(retrain_cooldown_minutes, 1) * 60
            cooldown_ok = (time.time() - last_retrain_started_at) >= cooldown_seconds
            simulate_ok = (not simulate) or retrain_on_simulate

            if retrain_proc is None and simulate_ok and cooldown_ok and underperformers >= retrain_min_underperformers:
                memory_ok = True
                memory_snapshot: Dict[str, float] = {}
                memory_reason = "guard_disabled"

                if memory_guard_enabled:
                    memory_ok, memory_snapshot, memory_reason = _auto_retrain_memory_ok(
                        min_free_pct=auto_retrain_min_free_pct,
                        max_swap_gb=auto_retrain_max_swap_gb,
                    )

                if memory_ok:
                    if memory_guard_enabled:
                        auto_retrain_healthy_streak += 1
                    else:
                        auto_retrain_healthy_streak = auto_retrain_healthy_streak_needed

                    if auto_retrain_healthy_streak >= auto_retrain_healthy_streak_needed:
                        lock_busy, lock_path = _mlx_retrain_lock_busy(PROJECT_ROOT)
                        if lock_busy:
                            event = {
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "event": "retrain_skipped_lock_busy",
                                "underperformer_count": underperformers,
                                "lock_path": lock_path,
                            }
                            _append_jsonl(_auto_retrain_log_path(PROJECT_ROOT, broker=broker), event)
                            print(f"[AutoRetrain] skipped reason=mlx_lock_busy lock_path={lock_path}")
                        else:
                            retrain_proc = _spawn_auto_retrain(
                                project_root=PROJECT_ROOT,
                                underperformers=underperformers,
                                sample_recommendations=latest_recs,
                                broker=broker,
                            )
                            last_retrain_started_at = time.time()
                            auto_retrain_healthy_streak = 0
                    else:
                        event = {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "event": "retrain_waiting_healthy_streak",
                            "underperformer_count": underperformers,
                            "healthy_streak": auto_retrain_healthy_streak,
                            "healthy_streak_needed": auto_retrain_healthy_streak_needed,
                            "snapshot": memory_snapshot,
                        }
                        _append_jsonl(_auto_retrain_log_path(PROJECT_ROOT, broker=broker), event)
                        print(
                            "[AutoRetrain] waiting reason=healthy_streak "
                            f"streak={auto_retrain_healthy_streak}/{auto_retrain_healthy_streak_needed}"
                        )
                else:
                    auto_retrain_healthy_streak = 0
                    event = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "event": "retrain_skipped_memory_guard",
                        "underperformer_count": underperformers,
                        "reason": memory_reason,
                        "snapshot": memory_snapshot,
                    }
                    _append_jsonl(_auto_retrain_log_path(PROJECT_ROOT, broker=broker), event)
                    print(f"[AutoRetrain] skipped reason=memory_guard details={memory_reason}")

        loop_seconds = time.time() - loop_started_at
        bp = backpressure.evaluate(loop_seconds=loop_seconds, interval_seconds=current_interval_seconds)
        overload_mode = bp.overloaded
        if overload_mode:
            print(f"[Backpressure] overloaded ratio={bp.ratio_vs_interval:.2f} loop_s={bp.loop_seconds:.2f}")

        if memory_auto_throttle_enabled and (iter_count % memory_throttle_check_every_iters == 0):
            mem_ok, mem_snapshot, mem_reason = _auto_retrain_memory_ok(
                min_free_pct=memory_throttle_min_free_pct,
                max_swap_gb=memory_throttle_max_swap_gb,
            )
            latest_memory_snapshot = mem_snapshot
            if not mem_ok:
                memory_throttle_active = True
                memory_throttle_recover_streak = 0
                overload_mode = True
                healthy_streak = 0
                prev_interval = current_interval_seconds
                current_interval_seconds = min(current_interval_seconds + memory_throttle_step_up_seconds, adaptive_interval_max)
                print(
                    f"[MemoryThrottle] active reason={mem_reason} "
                    f"interval={prev_interval}s->{current_interval_seconds}s"
                )
            elif memory_throttle_active:
                memory_throttle_recover_streak += 1
                if memory_throttle_recover_streak >= memory_throttle_recover_streak_needed:
                    memory_throttle_active = False
                    memory_throttle_recover_streak = 0
                    print("[MemoryThrottle] cleared")

        if adaptive_interval_enabled:
            prev_interval = current_interval_seconds
            if overload_mode:
                overload_streak += 1
                healthy_streak = 0
                current_interval_seconds = min(current_interval_seconds + adaptive_interval_step_up, adaptive_interval_max)
            else:
                healthy_streak += 1
                overload_streak = 0
                if healthy_streak >= adaptive_interval_recover_streak:
                    current_interval_seconds = max(current_interval_seconds - adaptive_interval_step_down, adaptive_interval_min)
                    healthy_streak = 0
            if current_interval_seconds != prev_interval:
                print(f"[AdaptiveInterval] changed from={prev_interval}s to={current_interval_seconds}s")

        external_floor = effective_interval_seconds + _external_ingestion_extra_interval_seconds(PROJECT_ROOT)
        if current_interval_seconds < external_floor:
            print(f"[IngestionBackpressure] applying external interval floor={external_floor}s")
            current_interval_seconds = external_floor

        if log_maintenance_enabled and (iter_count % log_maintenance_every_iters == 0):
            stats = _run_log_maintenance(PROJECT_ROOT, max_ops=log_maintenance_max_ops)
            if (stats.get('compressed', 0) + stats.get('deleted', 0)) > 0:
                print(
                    f"[LogMaintenance] compressed={stats.get('compressed', 0)} "
                    f"deleted={stats.get('deleted', 0)} ops={stats.get('ops', 0)}"
                )

        if iter_count % telemetry_every == 0:
            telemetry.emit({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "iter": iter_count,
                "config_hash": run_config_hash,
                "loop_seconds": round(bp.loop_seconds, 4),
                "ratio_vs_interval": round(bp.ratio_vs_interval, 4),
                "overloaded": bp.overloaded,
                "active_bots": len([b for b in bots if b.active]),
                "cache_size": len(state_cache._store),
                "current_interval_seconds": current_interval_seconds,
                "memory_throttle_active": memory_throttle_active,
                "memory_free_pct": latest_memory_snapshot.get("free_pct"),
                "memory_swap_used_gb": latest_memory_snapshot.get("swap_used_gb"),
            })

        if iter_count % checkpoint_every == 0:
            checkpoint.save({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "iter_count": iter_count,
                "config_hash": run_config_hash,
                "overloaded": overload_mode,
            })

        if max_iterations > 0 and iter_count >= max_iterations:
            _set_loop_state("completed", reason="max_iterations_reached", max_iterations=max_iterations)
            _write_heartbeat(
                project_root=PROJECT_ROOT,
                broker=broker,
                iter_count=iter_count,
                symbols_total=len(symbols),
                context_total=len(context_symbols),
                state="completed",
            )
            print("Reached max iterations, exiting.")
            return

        JsonlWriteBuffer.shared().flush_all()
        sleep_s = max(current_interval_seconds - loop_seconds, 0.0)
        time.sleep(sleep_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous shadow loop for sub-bot + master decision logging.")

    default_core = os.getenv(
        "WATCH_SYMBOLS_CORE",
        os.getenv("WATCH_SYMBOLS", os.getenv("WATCH_SYMBOL", "SPY,QQQ,DIA,IWM,MDY,VOO,VTI,RSP,AAPL,MSFT,NVDA,AMZN,GOOGL,META,AVGO,ORCL,CRM,ADBE,NFLX,DIS,WBD,GS,JPM,BKNG,ABNB,MAR,HLT")),
    )
    default_volatile = os.getenv(
        "WATCH_SYMBOLS_VOLATILE",
        "SOXL,SOXS,TQQQ,SQQQ,MSTR,SMCI,COIN,TSLA,PLTR,AMD,MRVL,ARM,IBIT,ETHA,MARA,RIOT,UVXY,VIXY,AAL,UAL,DAL,LUV,ALK,JBLU,CCL,RCL,NCLH,EXPE,JETS,XOP,OIH,OXY,SLB,HAL",
    )
    default_defensive = os.getenv(
        "WATCH_SYMBOLS_DEFENSIVE",
        "TLT,GLD,XLV,XLU,XLP,MO,HYG,LQD,UUP,XLE,XLF,XLI,XLK,XLY,IEF,SHY,TIP,TLH,JNK,AGG,BND,MUB,IGIB,USHY,FLOT,VGIT,SCHD,VIG,DGRO,HDV,NOBL,VYM,DIVO,JEPI,JEPQ,SPLV,VTV,JNJ,PG,KO,PEP,MCD,ABBV,ABT,MRK,PFE,T,VZ,O,VICI,MAIN,XOM,CVX,COP,EOG,MPC,PSX,VLO,KMI,ITA,LMT,NOC,RTX,GD,LHX,LDOS,DBC,UNG,CORN,SLV,USO,FXE,FXY,EFA,EEM,EWJ,FXI,VEA,VWO,IEFA,VGK,INDA,SMH,SOXX,VGT,IGV,XOP,OIH,JETS,VNQ,IYR",
    )

    parser.add_argument(
        "--symbols",
        default=None,
        help="Override full symbol list. If set, group args are ignored.",
    )
    parser.add_argument("--symbols-core", default=default_core)
    parser.add_argument("--symbols-volatile", default=default_volatile)
    parser.add_argument("--symbols-defensive", default=default_defensive)
    parser.add_argument("--core-every-n-iters", type=int, default=int(os.getenv("CORE_SYMBOL_EVERY_N_ITERS", "1")))
    parser.add_argument("--volatile-every-n-iters", type=int, default=int(os.getenv("VOLATILE_SYMBOL_EVERY_N_ITERS", "1")))
    parser.add_argument("--defensive-every-n-iters", type=int, default=int(os.getenv("DEFENSIVE_SYMBOL_EVERY_N_ITERS", "2")))
    parser.add_argument("--context-symbols", default=os.getenv("WATCH_CONTEXT_SYMBOLS", "$VIX.X,UUP"))
    parser.add_argument(
        "--profile",
        default=os.getenv("SHADOW_PROFILE", "").strip().lower(),
        help="Optional shadow profile tag (e.g. aggressive, crypto_futures).",
    )
    parser.add_argument(
        "--domain",
        default=os.getenv("SHADOW_DOMAIN", "").strip().lower(),
        help="Optional shadow domain override (equities|crypto).",
    )
    parser.add_argument("--broker", default=os.getenv("DATA_BROKER", "schwab"), choices=["schwab", "coinbase"])
    parser.add_argument("--interval-seconds", type=int, default=int(os.getenv("SHADOW_LOOP_INTERVAL", "12")))
    parser.add_argument("--max-iterations", type=int, default=int(os.getenv("SHADOW_LOOP_MAX_ITERS", "0")))
    parser.add_argument("--simulate", action="store_true", help="Run without Schwab API calls.")
    parser.add_argument(
        "--auto-retrain",
        action="store_true",
        default=os.getenv("AUTO_RETRAIN_ON_GOVERNANCE", "1").strip() == "1",
        help="Automatically trigger weekly retrain when underperformers exceed threshold.",
    )
    parser.add_argument(
        "--retrain-cooldown-minutes",
        type=int,
        default=int(os.getenv("AUTO_RETRAIN_COOLDOWN_MINUTES", "360")),
    )
    parser.add_argument(
        "--retrain-min-underperformers",
        type=int,
        default=int(os.getenv("AUTO_RETRAIN_MIN_UNDERPERFORMERS", "8")),
    )
    parser.add_argument(
        "--retrain-on-simulate",
        action="store_true",
        default=os.getenv("AUTO_RETRAIN_ON_SIMULATE", "0").strip() == "1",
        help="Allow auto retrain while --simulate is enabled.",
    )
    parser.add_argument(
        "--session-start-hour",
        type=int,
        default=int(os.getenv("MARKET_SESSION_START_HOUR", "8")),
        help="Session start hour in America/New_York (24h clock).",
    )
    parser.add_argument(
        "--session-end-hour",
        type=int,
        default=int(os.getenv("MARKET_SESSION_END_HOUR", "20")),
        help="Session end hour in America/New_York (24h clock).",
    )
    parser.add_argument(
        "--bad-symbol-fail-limit",
        type=int,
        default=int(os.getenv("BAD_SYMBOL_FAIL_LIMIT", "2")),
        help="Consecutive failures before a symbol is temporarily quarantined.",
    )
    parser.add_argument(
        "--bad-symbol-retry-minutes",
        type=int,
        default=int(os.getenv("BAD_SYMBOL_RETRY_MINUTES", "30")),
        help="Minutes to wait before retrying a quarantined symbol.",
    )
    args = parser.parse_args()

    profile_override = (args.profile or "").strip().lower()
    if profile_override:
        os.environ["SHADOW_PROFILE"] = profile_override

    domain_override = (args.domain or "").strip().lower()
    if domain_override in {"equities", "crypto"}:
        os.environ["SHADOW_DOMAIN"] = domain_override

    if not _route_storage_or_fail():
        return

    if _global_trading_halt_enabled():
        print("GLOBAL_TRADING_HALT=1 set; refusing to start shadow loop.")
        return

    if not _acquire_shadow_singleton_lock(PROJECT_ROOT, args.broker):
        return

    symbols = _parse_symbols(args.symbols) if args.symbols else _merge_symbol_groups(
        args.symbols_core,
        args.symbols_volatile,
        args.symbols_defensive,
    )
    volatile_symbols = _parse_symbols(args.symbols_volatile)
    defensive_symbols = _parse_symbols(args.symbols_defensive)
    core_every_n = max(args.core_every_n_iters, 1)
    volatile_every_n = max(args.volatile_every_n_iters, 1)
    defensive_every_n = max(args.defensive_every_n_iters, 1)
    current_profile = (args.profile or os.getenv("SHADOW_PROFILE", "")).strip().lower()

    context_symbols = _parse_symbols(args.context_symbols)
    if args.broker == "coinbase":
        symbols = [CoinbaseMarketDataClient.normalize_symbol(s) for s in symbols]
        context_symbols = [CoinbaseMarketDataClient.normalize_symbol(s) for s in context_symbols]
        # For crypto mode, if context stayed at equity defaults (before or after normalization),
        # swap to crypto context symbols to avoid repeated 404 fetches.
        equity_context_defaults = {"$VIX.X", "UUP", "$VIX.X-USD", "UUP-USD"}
        if context_symbols and all(s in equity_context_defaults for s in context_symbols):
            context_symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

        policy = _coinbase_symbol_policy(current_profile, symbols)
        volatile_symbols = list(policy["fast_symbols"])
        defensive_symbols = list(policy["slow_symbols"])
        core_every_n = int(policy["core_every_n"])
        volatile_every_n = int(policy["fast_every_n"])
        defensive_every_n = int(policy["slow_every_n"])
        if policy["websocket_symbols"]:
            os.environ["COINBASE_WEBSOCKET_SYMBOLS"] = ",".join(policy["websocket_symbols"])
        print(
            "[CoinbaseTiering] "
            f"profile={current_profile or 'default'} fast={len(volatile_symbols)} slow={len(defensive_symbols)} "
            f"websocket={len(policy['websocket_symbols'])} core_every_n={core_every_n} "
            f"fast_every_n={volatile_every_n} slow_every_n={defensive_every_n}"
        )
    elif args.broker == "schwab":
        symbols = [_normalize_runtime_symbol("schwab", s) for s in symbols]
        context_symbols = [_normalize_runtime_symbol("schwab", s) for s in context_symbols]
        policy = _schwab_symbol_policy(current_profile, symbols)
        volatile_symbols = list(policy["fast_symbols"])
        defensive_symbols = list(policy["slow_symbols"])
        core_every_n = int(policy["core_every_n"])
        volatile_every_n = int(policy["fast_every_n"])
        defensive_every_n = int(policy["slow_every_n"])
        print(
            "[SchwabTiering] "
            f"profile={current_profile or 'default'} fast={len(volatile_symbols)} slow={len(defensive_symbols)} "
            f"core_every_n={core_every_n} fast_every_n={volatile_every_n} slow_every_n={defensive_every_n}"
        )

    symbol_set = set(symbols)
    volatile_symbols = [symbol for symbol in volatile_symbols if symbol in symbol_set]
    volatile_set = set(volatile_symbols)
    defensive_symbols = [symbol for symbol in defensive_symbols if symbol in symbol_set and symbol not in volatile_set]
    core_count = len([symbol for symbol in symbols if symbol not in volatile_set and symbol not in set(defensive_symbols)])

    print(
        f"Symbol groups: core={core_count} "
        f"volatile={len(volatile_symbols)} "
        f"defensive={len(defensive_symbols)} total={len(symbols)} broker={args.broker}"
    )
    os.environ["CORE_SYMBOL_EVERY_N_ITERS"] = str(core_every_n)
    os.environ["VOLATILE_SYMBOL_EVERY_N_ITERS"] = str(volatile_every_n)
    os.environ["DEFENSIVE_SYMBOL_EVERY_N_ITERS"] = str(defensive_every_n)

    os.chdir(PROJECT_ROOT)
    run_loop(
        symbols=symbols,
        context_symbols=context_symbols,
        broker=args.broker,
        interval_seconds=args.interval_seconds,
        max_iterations=args.max_iterations,
        simulate=args.simulate,
        auto_retrain=args.auto_retrain,
        retrain_cooldown_minutes=args.retrain_cooldown_minutes,
        retrain_min_underperformers=args.retrain_min_underperformers,
        retrain_on_simulate=args.retrain_on_simulate,
        session_start_hour=args.session_start_hour,
        session_end_hour=args.session_end_hour,
        bad_symbol_fail_limit=args.bad_symbol_fail_limit,
        bad_symbol_retry_minutes=args.bad_symbol_retry_minutes,
        volatile_symbols=volatile_symbols,
        defensive_symbols=defensive_symbols,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user, stopping shadow loop.")
    except Exception as exc:
        try:
            broker = os.getenv("DATA_BROKER", "schwab").strip().lower()
            _append_jsonl(
                _event_bus_path(PROJECT_ROOT),
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "event": "shadow_loop_fatal",
                    "broker": broker,
                    "profile": _shadow_profile_name() or "default",
                    "domain": _shadow_domain_name(broker=broker),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        except Exception:
            pass
        print(f"Shadow loop fatal error: {type(exc).__name__}: {exc}")
        raise
