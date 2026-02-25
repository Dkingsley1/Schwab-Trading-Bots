import argparse
import atexit
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
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

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
from core.coinbase_market_data import CoinbaseMarketDataClient
from core.runtime_layers import (
    BackpressureController,
    CanaryRollout,
    CheckpointStore,
    CircuitBreaker,
    StateCache,
    TelemetryEmitter,
    config_hash,
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


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _global_trading_halt_enabled() -> bool:
    return _env_flag("GLOBAL_TRADING_HALT", "0") or HALT_FLAG_PATH.exists()


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
        bots.append(
            SubBot(
                bot_id=str(row.get("bot_id")),
                weight=float(row.get("weight", 0.0) or 0.0),
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


def _apply_canary_rollout_to_bots(bots: List[SubBot], canary_max_weight: float) -> List[SubBot]:
    if canary_max_weight <= 0.0:
        return bots

    active = [b for b in bots if b.active]
    if not active:
        return bots

    sandbox_enabled = os.getenv("CANARY_WEIGHT_SANDBOX_ENABLED", "1").strip() == "1"
    sandbox_per_bot_cap = min(
        max(float(os.getenv("CANARY_WEIGHT_SANDBOX_MAX", str(canary_max_weight))), 0.0),
        max(canary_max_weight, 0.0),
    ) if sandbox_enabled else canary_max_weight
    sandbox_total_cap = min(
        max(float(os.getenv("CANARY_WEIGHT_SANDBOX_TOTAL_MAX", "0.18")), 0.0),
        1.0,
    ) if sandbox_enabled else 1.0

    for b in active:
        if b.promoted:
            b.weight = min(max(b.weight, 0.0), sandbox_per_bot_cap)

    promoted = [b for b in active if b.promoted]
    non_promoted = [b for b in active if not b.promoted]
    promoted_total = sum(max(b.weight, 0.0) for b in promoted)
    if promoted and promoted_total > sandbox_total_cap:
        scale = sandbox_total_cap / max(promoted_total, 1e-8)
        for b in promoted:
            b.weight = max(b.weight, 0.0) * scale
        remainder = max(1.0 - sandbox_total_cap, 0.0)
        non_promoted_total = sum(max(b.weight, 0.0) for b in non_promoted)
        if non_promoted and non_promoted_total > 0.0:
            for b in non_promoted:
                b.weight = max(b.weight, 0.0) / non_promoted_total * remainder

    total = sum(max(b.weight, 0.0) for b in active)
    if total > 0.0:
        for b in active:
            b.weight = max(b.weight, 0.0) / total

    return bots


def _extract_quote_payload(raw: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    sym = symbol.upper()
    if sym in raw and isinstance(raw[sym], dict):
        return raw[sym]
    if raw and isinstance(next(iter(raw.values())), dict):
        return next(iter(raw.values()))
    return {}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_NEWS_FEATURE_KEYS = [
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

_NEWS_POSITIVE_TOKENS = {
    "beat", "beats", "upgrade", "upgrades", "outperform", "buy", "surge", "record", "growth", "raises", "strong", "bullish", "profit", "profits", "gain", "gains"
}

_NEWS_NEGATIVE_TOKENS = {
    "miss", "misses", "downgrade", "downgrades", "underperform", "sell", "drop", "plunge", "cut", "cuts", "weak", "bearish", "loss", "losses", "probe", "lawsuit", "bankruptcy"
}

_NEWS_SHOCK_TOKENS = {
    "guidance", "earnings", "fda", "recall", "investigation", "sec", "merger", "acquisition", "layoff", "default"
}


def _default_news_features() -> Dict[str, float]:
    return {k: 0.0 for k in _NEWS_FEATURE_KEYS}


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


def _is_usable_market_snapshot(mkt: Dict[str, float]) -> bool:
    # Premarket/after-hours feeds can temporarily miss last_price; allow sane fallbacks.
    if _to_float(mkt.get("last_price"), 0.0) > 0.0:
        return True
    if _to_float(mkt.get("prev_close"), 0.0) > 0.0:
        return True
    return False


def _market_snapshot_from_schwab(client: Any, symbol: str) -> Dict[str, float]:
    quote_resp = client.get_quote(symbol)
    quote_obj = quote_resp.json() if hasattr(quote_resp, "json") else {}
    quote = _extract_quote_payload(quote_obj if isinstance(quote_obj, dict) else {}, symbol)

    now_utc = datetime.now(timezone.utc)
    history_resp = client.get_price_history_every_minute(
        symbol,
        start_datetime=now_utc - timedelta(minutes=60),
        end_datetime=now_utc,
        need_extended_hours_data=False,
        need_previous_close=True,
    )
    hist_obj = history_resp.json() if hasattr(history_resp, "json") else {}
    candles = hist_obj.get("candles", []) if isinstance(hist_obj, dict) else []

    closes = [_to_float(c.get("close")) for c in candles if _to_float(c.get("close")) > 0]
    highs = [_to_float(c.get("high")) for c in candles if _to_float(c.get("high")) > 0]
    lows = [_to_float(c.get("low")) for c in candles if _to_float(c.get("low")) > 0]

    last_price = _to_float(quote.get("lastPrice"), 0.0)
    if last_price <= 0:
        last_price = _to_float(quote.get("mark"), 0.0)
    if last_price <= 0 and closes:
        last_price = closes[-1]

    prev_close = _to_float(quote.get("closePrice"), 0.0)
    if prev_close <= 0 and len(closes) > 1:
        prev_close = closes[0]
    if prev_close <= 0:
        prev_close = max(last_price, 1.0)

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

    return {
        "last_price": last_price,
        "prev_close": prev_close,
        "pct_from_close": pct_from_close,
        "vol_30m": vol_30m,
        "mom_5m": mom_5m,
        "range_pos": range_pos,
        "snapshot_ts_utc": now_utc.timestamp(),
    }


def _market_snapshot_from_coinbase(client: CoinbaseMarketDataClient, symbol: str) -> Dict[str, float]:
    snap = client.market_snapshot(symbol)
    out = dict(snap) if isinstance(snap, dict) else {}
    if float(out.get("snapshot_ts_utc", 0.0) or 0.0) <= 0.0:
        out["snapshot_ts_utc"] = time.time()
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
    return {
        "last_price": new_price,
        "prev_close": prev_close,
        "pct_from_close": pct_from_close,
        "vol_30m": vol_30m,
        "mom_5m": mom_5m,
        "range_pos": range_pos,
        "snapshot_ts_utc": time.time(),
    }


def _hash_unit(text: str) -> float:
    b = hashlib.sha256(text.encode("utf-8")).digest()[0]
    return (b / 255.0) * 2.0 - 1.0


def _calibrate_vote(raw_vote: float, scale: float = 0.9) -> float:
    return math.tanh(scale * raw_vote)


def _vote_to_score(vote: float) -> float:
    # Keep confidence bounded away from extreme saturation.
    score = 0.5 + 0.45 * vote
    return min(max(score, 0.01), 0.99)


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


def _event_blackout_windows() -> List[tuple[int, int]]:
    raw = os.getenv(
        "EVENT_LOCK_WINDOWS_ET",
        os.getenv("EVENT_BLACKOUT_WINDOWS_ET", "08:29-08:36,09:59-10:06,13:58-14:05"),
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
    if name == "trend":
        vote += 0.40 * mom + 0.25 * pct
        reasons = base_reasons + ["trend_master_bias"]
    elif name == "mean_revert":
        vote += -0.45 * (range_pos - 0.5) - 0.25 * mom
        reasons = base_reasons + ["mean_revert_master_bias"]
    elif name == "shock":
        vote += -0.35 * vix_pct - 0.20 * vol + 0.30 * float(features.get("aux_flash_direction", 0.0))
        reasons = base_reasons + ["shock_master_bias"]
    else:
        reasons = base_reasons + ["default_master_bias"]

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

    prof = _shadow_profile_name()
    if prof == "dividend":
        # Dividend sleeve: favor persistent trend + valuation mean reversion, downweight shock chasing.
        w_trend = 1.30 + 2.2 * trend_strength
        w_mean = 1.25 + 2.0 * max(0.0, 0.5 - chop_strength)
        w_shock = 0.70 + 1.2 * shock_strength
    elif prof == "bond":
        # Bond sleeve: emphasize regime persistence + carry mean reversion, but keep shock filter alive.
        w_trend = 1.20 + 1.9 * trend_strength
        w_mean = 1.30 + 2.2 * max(0.0, 0.5 - chop_strength)
        w_shock = 0.90 + 1.4 * shock_strength
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

    # Convert spot-level confidence into options suitability with extra regime risk penalties.
    net = float(grand_vote)
    net += 0.20 * (grand_score - 0.5)
    net -= 0.25 * max(vix_pct, 0.0)
    net -= 0.20 * vol
    net += -0.10 * dollar_mom
    net = _calibrate_vote(net, scale=0.75)

    score = _vote_to_score(net)
    threshold = _shift_threshold(0.62)

    if net > 0.25:
        action = "BUY"
    elif net < -0.22:
        action = "SELL"
    else:
        action = "HOLD"

    reasons = [
        "options_master_regime_filter",
        f"from_grand={grand_action}",
        f"vix_pct={vix_pct:.4f}",
        f"vol_30m={vol:.4f}",
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
    px = max(mkt.get("last_price", 0.0), 1.0)
    vol = max(mkt.get("vol_30m", 0.0), 0.0)
    range_pos = mkt.get("range_pos", 0.5)

    # Conservative default: no options position when confidence is mixed.
    action = "HOLD"
    score = master_score
    threshold = _shift_threshold(0.58)
    reasons = ["options_filter_no_clear_edge"]
    plan = {
        "symbol": symbol,
        "options_style": "NONE",
        "underlying_price": px,
        "dte_days": 21,
        "contracts": 0,
        "strike": None,
    }

    can_cover = covered_call_shares >= 100

    if can_cover and master_action in {"HOLD", "SELL"} and range_pos > 0.62 and vol < 0.02:
        contracts = max(covered_call_shares // 100, 1)
        action = "SELL_TO_OPEN"
        score = max(0.60, master_score)
        reasons = ["covered_call_income_setup", "has_covered_shares", "price_near_range_high"]
        plan = {
            "symbol": symbol,
            "options_style": "COVERED_CALL",
            "underlying_price": px,
            "dte_days": 21,
            "contracts": contracts,
            "strike": round(px * 1.03, 2),
        }
    elif master_action == "BUY" and master_score >= threshold:
        action = "BUY_TO_OPEN"
        score = master_score
        reasons = ["bullish_master_signal", "long_call_setup"]
        plan = {
            "symbol": symbol,
            "options_style": "LONG_CALL",
            "underlying_price": px,
            "dte_days": 30,
            "contracts": 1,
            "strike": round(px * 1.02, 2),
        }
    elif master_action == "SELL" and (1.0 - master_score) >= (threshold - 0.03):
        action = "BUY_TO_OPEN"
        score = max(1.0 - master_score, 0.01)
        reasons = ["bearish_master_signal", "long_put_setup"]
        plan = {
            "symbol": symbol,
            "options_style": "LONG_PUT",
            "underlying_price": px,
            "dte_days": 30,
            "contracts": 1,
            "strike": round(px * 0.98, 2),
        }

    plan["master_vote"] = master_vote
    return {
        "action": action,
        "score": min(max(score, 0.01), 0.99),
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

        model = {
            "path": path,
            "W": arr["W"].astype(dtype),
            "b": arr["b"].astype(dtype),
            "mu": arr["mu"].astype(dtype),
            "sigma": arr["sigma"].astype(dtype),
            "temperature": float(temperature),
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
    "snapshot_cov_ok",
    "snapshot_cov_log_ratio",
    "snapshot_replay_stale_ratio",
    "snapshot_replay_drift_ratio",
    "snapshot_divergence_ratio",
    "snapshot_triprate_ratio",
    "snapshot_queue_pressure_ratio",
    "canary_weight_cap_norm",
]
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
    feature_freshness_max_age_seconds = float(os.getenv('FEATURE_FRESHNESS_MAX_AGE_SECONDS', '20'))
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

    master_latency_timeout_ms = max(float(os.getenv('MASTER_LATENCY_SLO_TIMEOUT_MS', '900')), 1.0)
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
        'snapshot_cov_ok': _behavior_clamp01(float(snapshot_ctx.get('snapshot_cov_ok', 1.0) or 1.0)),
        'snapshot_cov_log_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_cov_log_ratio', 0.0) or 0.0)),
        'snapshot_replay_stale_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_replay_stale_ratio', 0.0) or 0.0)),
        'snapshot_replay_drift_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_replay_drift_ratio', 0.0) or 0.0)),
        'snapshot_divergence_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_divergence_ratio', 0.0) or 0.0)),
        'snapshot_triprate_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_triprate_ratio', 0.0) or 0.0)),
        'snapshot_queue_pressure_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_queue_pressure_ratio', 0.0) or 0.0)),
        'canary_weight_cap_norm': _behavior_clamp01(float(snapshot_ctx.get('canary_weight_cap_norm', 0.0) or 0.0)),
    }

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
    JsonlWriteBuffer.shared().append(path, row)


class JsonlWriteBuffer:
    _instance: Optional["JsonlWriteBuffer"] = None

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
        self._buf: Dict[str, List[str]] = {}
        self._ts: Dict[str, float] = {}

    def append(self, path: str, row: Dict[str, Any]) -> None:
        if not self.enabled:
            self._flush_direct(path, json.dumps(row, ensure_ascii=True) + "\n")
            return
        payload = json.dumps(row, ensure_ascii=True) + "\n"
        now = time.time()
        bucket = self._buf.setdefault(path, [])
        if path not in self._ts:
            self._ts[path] = now
        bucket.append(payload)
        if len(bucket) >= self.max_items or (now - self._ts.get(path, now)) >= self.max_age_seconds:
            self.flush_path(path)

    def flush_path(self, path: str) -> None:
        rows = self._buf.get(path, [])
        if not rows:
            return
        self._flush_direct(path, "".join(rows))
        self._buf[path] = []
        self._ts[path] = time.time()

    def flush_all(self) -> None:
        for path in list(self._buf.keys()):
            self.flush_path(path)

    @staticmethod
    def _flush_direct(path: str, payload: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(payload)


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
        if b.bot_role == "infrastructure_sub_bot":
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

    retention_decisions = int(os.getenv('LOG_RETENTION_DECISIONS_DAYS', '7'))
    retention_explanations = int(os.getenv('LOG_RETENTION_DECISION_EXPLANATIONS_DAYS', '7'))
    retention_governance = int(os.getenv('LOG_RETENTION_GOVERNANCE_DAYS', '10'))
    retention_exports = int(os.getenv('LOG_RETENTION_EXPORTS_DAYS', '7'))
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
    if prof == "dividend":
        # Dividend sleeve should be more selective and avoid noisy overnight churn.
        return +0.03
    if prof == "bond":
        # Bond sleeve: slightly stricter entry to reduce false positives in low-vol regimes.
        return +0.02
    return 0.0


def _shift_threshold(base: float) -> float:
    v = float(base) + _threshold_shift()
    return min(max(v, 0.45), 0.75)


def _governance_path(project_root: str, broker: Optional[str] = None) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(project_root, "governance", _shadow_profile_subdir(broker=broker), f"master_control_{day}.jsonl")


def _event_bus_path(project_root: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(project_root, "governance", "events", f"runtime_events_{day}.jsonl")


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
        "iter": int(iter_count),
        "symbols_total": int(symbols_total),
        "context_total": int(context_total),
        "state": state,
    }
    path = _heartbeat_path(project_root, broker)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)
    os.replace(tmp, path)


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
            client = trader.authenticate()
            print("Shadow loop connected to Schwab market data.")
        else:
            client = CoinbaseMarketDataClient(timeout_seconds=float(os.getenv("COINBASE_TIMEOUT_SECONDS", "8")))
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

    # Keep retrain-critical logs while reducing high-volume layer noise by default.
    log_sub_bot_decisions = os.getenv("LOG_SUB_BOT_DECISIONS", "0").strip() == "1"
    log_master_variant_decisions = os.getenv("LOG_MASTER_VARIANT_DECISIONS", "0").strip() == "1"
    log_grand_master_decisions = os.getenv("LOG_GRAND_MASTER_DECISIONS", "1").strip() == "1"
    log_options_master_decisions = os.getenv("LOG_OPTIONS_MASTER_DECISIONS", "1").strip() == "1"

    bot_cooldown_enabled = os.getenv("BOT_COOLDOWN_ENABLED", "1").strip() == "1"
    bot_cooldown_acc_floor = float(os.getenv("BOT_COOLDOWN_ACC_FLOOR", "0.53"))
    bot_cooldown_min_iters = max(int(os.getenv("BOT_COOLDOWN_MIN_ITERS", "2")), 1)
    bot_cooldown_max_iters = max(int(os.getenv("BOT_COOLDOWN_MAX_ITERS", "4")), bot_cooldown_min_iters)
    bot_next_eval_iter: Dict[str, int] = {}

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
    feature_freshness_max_age_seconds = float(os.getenv('FEATURE_FRESHNESS_MAX_AGE_SECONDS', '20'))
    feature_freshness_required = [
        x.strip() for x in os.getenv(
            'FEATURE_FRESHNESS_REQUIRED_KEYS',
            'last_price,prev_close,pct_from_close,vol_30m,mom_5m',
        ).split(',') if x.strip()
    ]
    master_latency_slo_enabled = os.getenv('MASTER_LATENCY_SLO_GUARD_ENABLED', '1').strip() == '1'
    master_latency_slo_timeout_ms = float(os.getenv('MASTER_LATENCY_SLO_TIMEOUT_MS', '900'))
    preopen_replay_sanity_enabled = os.getenv(
        'PREOPEN_REPLAY_SANITY_ENABLED',
        '0' if broker == 'coinbase' else '1',
    ).strip() == '1'
    preopen_replay_sanity_timeout_seconds = max(
        int(os.getenv('PREOPEN_REPLAY_SANITY_TIMEOUT_SECONDS', '30')),
        5,
    )
    snapshot_debug_mode = os.getenv('SNAPSHOT_DEBUG_MODE', '0').strip() == '1'

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
        "log_sub_bot_decisions": log_sub_bot_decisions,
        "log_master_variant_decisions": log_master_variant_decisions,
        "log_grand_master_decisions": log_grand_master_decisions,
        "log_options_master_decisions": log_options_master_decisions,
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
        if should_run:
            ok, reason = _run_preopen_replay_sanity_check(
                project_root=PROJECT_ROOT,
                timeout_seconds=preopen_replay_sanity_timeout_seconds,
            )
            if not ok:
                raise RuntimeError(f"Pre-open replay sanity check failed: {reason}")
            print('[ReplaySanity] pre-open replay check passed (24h)')

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

    log_maintenance_enabled = os.getenv('LOG_MAINTENANCE_ENABLED', '1').strip() == '1'
    log_maintenance_every_iters = max(int(os.getenv('LOG_MAINTENANCE_EVERY_ITERS', '20')), 1)
    log_maintenance_max_ops = max(int(os.getenv('LOG_MAINTENANCE_MAX_OPS', '300')), 10)

    exec_queue = ExecutionQueue(max_depth=max(int(os.getenv("EXEC_QUEUE_MAX_DEPTH", "4000")), 100))
    equity_proxy = float(os.getenv("ACCOUNT_EQUITY_PROXY", "100000"))
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


    while True:
        if _global_trading_halt_enabled():
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

        _write_heartbeat(
            project_root=PROJECT_ROOT,
            broker=broker,
            iter_count=iter_count,
            symbols_total=len(symbols),
            context_total=len(context_symbols),
            state="running",
        )

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
            if not open_now:
                if closed_reason != last_closed_reason:
                    print(f"[SessionGate] paused reason={closed_reason} now_et={now_et.isoformat()}")
                    last_closed_reason = closed_reason
                _record_snapshot_debug('*', 'session_gate_paused', closed_reason=closed_reason, now_et=now_et.isoformat())
                time.sleep(max(effective_interval_seconds, 30))
                continue
            if last_closed_reason is not None:
                print(f"[SessionGate] resumed now_et={now_et.isoformat()}")
                last_closed_reason = None
        else:
            if iter_count == 1:
                print(f"[SessionGate] disabled broker={broker} mode=24x7")

        if event_blackout_enabled and _in_event_blackout(now_et, blackout_windows):
            print(f"[EventGate] paused reason=event_lock_window now_et={now_et.isoformat()}")
            _record_snapshot_debug('*', 'event_lock_paused', now_et=now_et.isoformat())
            time.sleep(max(effective_interval_seconds, 10))
            continue

        if time.time() < anomaly_kill_switch_until_ts:
            rem = int(anomaly_kill_switch_until_ts - time.time())
            print(f"[KillSwitch] paused reason=data_anomaly remaining_s={max(rem, 0)}")
            _record_snapshot_debug('*', 'anomaly_killswitch_paused', remaining_seconds=max(rem, 0))
            time.sleep(max(min(rem, effective_interval_seconds), 5))
            continue

        if iter_count == 1 or iter_count % 10 == 0:
            registry, bots = _fresh_registry(registry_path)
            bots = _apply_canary_rollout_to_bots(bots, canary_rollout.max_weight)

        active_bots = [b for b in bots if b.active]
        if not active_bots:
            active_bots = [SubBot(bot_id="fallback_master_seed", weight=1.0, active=True, reason="fallback", test_accuracy=0.50, bot_role="signal_sub_bot")]

        fast_bots, slow_bots = _split_bot_tiers(active_bots)
        evaluate_slow_this_iter = (iter_count % slow_bot_every_n_iters == 0)

        latest_recs: List[Dict[str, Any]] = []
        exposure_state: Dict[str, int] = {"BUY": 0, "SELL": 0}

        context_market: Dict[str, Dict[str, float]] = {}

        def _fetch_symbol_snapshot(sym: str) -> Dict[str, float]:
            cache_key = f"mkt:{sym}"
            cached = state_cache.get(cache_key)
            if cached is not None:
                return cached

            cb_key = f"md:{sym}"
            if not circuit_breaker.allow(cb_key):
                raise RuntimeError("circuit_open")

            if simulate:
                snap = _market_snapshot_simulated(sim_prices[sym])
                sim_prices[sym] = snap["last_price"]
            else:
                if broker == "schwab":
                    snap = _market_snapshot_from_schwab(client, sym)
                else:
                    snap = _market_snapshot_from_coinbase(client, sym)

            state_cache.set(cache_key, snap)
            circuit_breaker.record_success(cb_key)
            return snap

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
                now_ts=now_s,
                lookback_seconds=news_lookback_hours * 3600.0,
                max_items=news_max_items,
            )
            news_cache[sym] = (now_s, feats)
            return dict(feats)

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
            if cadence > 1 and (iter_count % cadence != 0):
                _record_snapshot_debug(symbol, 'cadence_skip', cadence=cadence)
                continue

            quarantine_until = symbol_quarantine_until.get(symbol, 0.0)
            if now_ts < quarantine_until:
                _record_snapshot_debug(symbol, 'quarantine_skip', quarantine_seconds=round(max(quarantine_until - now_ts, 0.0), 3))
                continue

            if not circuit_breaker.allow(f"md:{symbol}"):
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
                if symbol_fail_counts[symbol] >= max(bad_symbol_fail_limit, 1):
                    symbol_quarantine_until[symbol] = now_ts + max(bad_symbol_retry_minutes, 1) * 60
                    print(
                        f"[SymbolGuard] quarantined symbol={symbol} "
                        f"for {max(bad_symbol_retry_minutes, 1)}m after {symbol_fail_counts[symbol]} failures"
                    )
                    symbol_fail_counts[symbol] = 0
                continue

            if not _is_usable_market_snapshot(mkt):
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
            if px > 0.0 and prev_px > 0.0 and abs(px - prev_px) < 1e-10:
                symbol_stale_counts[symbol] = symbol_stale_counts.get(symbol, 0) + 1
            else:
                symbol_stale_counts[symbol] = 0
            symbol_last_price[symbol] = px

            stale_limit = max(int(os.getenv("STALE_PRICE_FAIL_LIMIT", "8")), 1)
            if symbol_stale_counts.get(symbol, 0) >= stale_limit:
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

            news_features = _default_news_features()
            if news_context_enabled:
                try:
                    news_features = _fetch_symbol_news_features(symbol)
                except Exception as exc:
                    news_features = _default_news_features()
                    _record_snapshot_debug(symbol, 'news_feed_error', error=str(exc))

            feature_cache_key = f"f:{symbol}"
            if feature_cache_enabled and feature_cache_key in feature_cache and feature_cache[feature_cache_key][0] == iter_count:
                shared_features = dict(feature_cache[feature_cache_key][1])
            else:
                shared_features = {**mkt, **context_features, **news_features}
                if feature_cache_enabled:
                    feature_cache[feature_cache_key] = (iter_count, dict(shared_features))

            freshness_ok, freshness_reason, freshness_age_s = _feature_freshness_guard(
                shared_features,
                max_age_seconds=feature_freshness_max_age_seconds,
                required_keys=feature_freshness_required,
            ) if feature_freshness_enabled else (True, 'disabled', 0.0)

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

            # Broadcast flash-crash specialist signal as shared context to all higher-level decisions.
            flash_aux = _derive_flash_aux_features(sub_rows)
            shared_features = {**shared_features, **flash_aux}
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
                        features={**shared_features, "master_vote": m_vote["vote"], "active_sub_bots": len(active_bots)},
                        gates={"ensemble_has_members": len(active_bots) > 0, "market_data_ok": mkt["last_price"] > 0},
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

            if iter_count < symbol_regime_cooldown_until_iter.get(symbol, 0):
                gm_action = "HOLD"
                gm_score = 0.5 + 0.5 * (gm_score - 0.5)
                gm_reasons = gm_reasons + ["regime_change_cooldown"]

            gm_action, gm_score, gm_reasons = _enforce_cross_symbol_exposure_cap(
                action=gm_action,
                score=gm_score,
                reasons=gm_reasons,
                exposure_state=exposure_state,
                max_long=exposure_cap_long,
                max_short=exposure_cap_short,
            )

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
            if risk_action != gm_action:
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

            pause_active = (now_ts < kill_switch_until_ts) or (now_ts < vol_shock_pause_until_ts) or (now_ts < liquidity_pause_until_ts)
            if pause_active and gm_action in {"BUY", "SELL"}:
                gm_action = "HOLD"
                gm_score = 0.5 + 0.5 * (gm_score - 0.5)
                gm_reasons = gm_reasons + ["circuit_breaker_pause_active"]

            if consecutive_loss_streak >= max_consecutive_losses and gm_action in {"BUY", "SELL"}:
                kill_switch_until_ts = max(kill_switch_until_ts, now_ts + kill_switch_cooldown_seconds)
                gm_action = "HOLD"
                gm_score = 0.5 + 0.5 * (gm_score - 0.5)
                gm_reasons = gm_reasons + [f"consecutive_loss_pause streak={consecutive_loss_streak}"]

            master_latency_ms = (time.perf_counter() - master_decision_started_at) * 1000.0
            master_latency_slo_ok = (not master_latency_slo_enabled) or (master_latency_ms <= master_latency_slo_timeout_ms)
            if (not master_latency_slo_ok) and gm_action in {"BUY", "SELL"}:
                gm_action = "HOLD"
                gm_score = 0.5 + 0.5 * (gm_score - 0.5)
                gm_reasons = gm_reasons + [
                    f"master_latency_slo_timeout elapsed_ms={master_latency_ms:.1f} timeout_ms={master_latency_slo_timeout_ms:.1f}"
                ]

            if (not freshness_ok) and gm_action in {"BUY", "SELL"}:
                gm_action = "HOLD"
                gm_score = 0.5 + 0.5 * (gm_score - 0.5)
                gm_reasons = gm_reasons + [f"feature_freshness_guard:{freshness_reason}"]

            volatility_now = float(shared_features.get("volatility_1m", shared_features.get("vol", 0.0)) or 0.0)
            raw_qty = size_from_action(
                action=gm_action,
                score=gm_score,
                threshold=gm_threshold,
                volatility_1m=volatility_now,
                equity_proxy=equity_proxy,
                max_notional_pct=max_notional_pct,
            )
            alloc_qty = allocate_quantity(
                raw_qty=raw_qty,
                symbol=symbol,
                score=gm_score,
                volatility_1m=volatility_now,
                base_budget=portfolio_base_budget,
                symbol_budgets=symbol_budgets,
            )
            req = OrderRequest(
                symbol=symbol,
                action=gm_action,
                quantity=alloc_qty,
                priority=_queue_priority(gm_action, gm_score),
                metadata={"snapshot_id": snapshot_id, "strategy": "grand_master_bot"},
            )
            enq_ok = exec_queue.enqueue(req)
            dispatch = exec_queue.pop() if enq_ok else None
            dispatch_qty = dispatch.quantity if dispatch is not None else 0.0

            if log_grand_master_decisions:
                trader.execute_decision(
                    symbol=symbol,
                    action=gm_action,
                    quantity=dispatch_qty,
                    model_score=gm_score,
                    threshold=gm_threshold,
                    features={**shared_features, "grand_master_vote": gm_vote["vote"], "active_sub_bots": len(active_bots), "sized_qty": dispatch_qty},
                    gates={
                        "ensemble_has_members": len(active_bots) > 0,
                        "market_data_ok": mkt["last_price"] > 0,
                        "risk_limit_ok": all(risk_gates.values()),
                        "feature_freshness_ok": freshness_ok,
                        "master_latency_slo_ok": master_latency_slo_ok,
                        **risk_gates,
                        "exec_queue_ok": enq_ok,
                    },
                    reasons=gm_reasons,
                    strategy="grand_master_bot",
                    metadata={"layer": "grand_master", "snapshot_id": snapshot_id, "master_weights": gm_weights, "queue_depth": exec_queue.size()},
                )

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
                    mkt=mkt,
                    master_action=optm_action,
                    master_score=optm_score,
                    master_vote=optm_vote["vote"],
                    covered_call_shares=covered_call_shares,
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

            ret_1m = float(symbol_return_1m)
            exec_sim = simulate_execution(
                action=gm_action,
                last_price=float(mkt.get("last_price", 0.0) or 0.0),
                return_1m=ret_1m,
                spread_bps=float(shared_features.get("spread_bps", 8.0) or 8.0),
                volatility_1m=float(shared_features.get("volatility_1m", shared_features.get("vol", 0.0)) or 0.0),
                latency_ms=float(os.getenv("EXEC_SIM_LATENCY_MS", "120")),
                bid_size=float(shared_features.get("bid_size", 1000.0) or 1000.0),
                ask_size=float(shared_features.get("ask_size", 1000.0) or 1000.0),
                order_size=dispatch_qty if dispatch_qty > 0 else 1.0,
            )
            for row in sub_rows:
                _append_jsonl(
                    _pnl_attribution_path(PROJECT_ROOT, broker=broker),
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "snapshot_id": snapshot_id,
                        "symbol": symbol,
                        "bot_id": row.get("bot_id"),
                        "layer": "sub_bot",
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
                },
            )

            intraday_pnl_proxy += float(exec_sim.adjusted_return_1m)
            peak_intraday_pnl_proxy = max(peak_intraday_pnl_proxy, intraday_pnl_proxy)
            if exec_sim.adjusted_return_1m < 0:
                consecutive_loss_streak = (consecutive_loss_streak + 1) if gm_action in {"BUY", "SELL"} else consecutive_loss_streak
            else:
                consecutive_loss_streak = 0

            recs = _governance_recommendations(bots)
            latest_recs = recs
            gov_row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "snapshot_id": snapshot_id,
                "market": mkt,
                "context_market": context_market,
                "active_sub_bots": len([b for b in bots if b.active]),
                "inactive_sub_bots": len([b for b in bots if not b.active]),
                "master_action": gm_action,
                "master_score": gm_score,
                "master_vote": gm_vote["vote"],
                "master_outputs": master_outputs,
                "grand_master_weights": gm_weights,
                "options_master": {"action": optm_action, "score": optm_score, "vote": optm_vote["vote"]},
                "flash_aux": flash_aux,
                "options_plan": options_decision["plan"],
                "execution_sim": {"slippage_bps": exec_sim.slippage_bps, "latency_ms": exec_sim.latency_ms, "expected_fill_price": exec_sim.expected_fill_price, "impact_bps": exec_sim.impact_bps},
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
                "portfolio": {"equity_proxy": equity_proxy, "raw_qty": raw_qty, "alloc_qty": alloc_qty, "dispatch_qty": dispatch_qty, "queue_depth": exec_queue.size()},
                "circuit_breakers": {"consecutive_loss_streak": consecutive_loss_streak, "kill_switch_active": time.time() < kill_switch_until_ts, "vol_shock_pause_active": time.time() < vol_shock_pause_until_ts, "liquidity_pause_active": time.time() < liquidity_pause_until_ts},
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
                "score": gm_score,
                "risk_limit_ok": all(risk_gates.values()),
                "slippage_bps": exec_sim.slippage_bps,
            })

            print(
                f"[ShadowLoop] iter={iter_count} symbol={symbol} price={mkt['last_price']:.2f} "
                f"grand_action={gm_action} options_master={optm_action} options_action={options_decision['action']} "
                f"active_bots={len(active_bots)} recs={len(recs)} snapshot_id={snapshot_id}"
            )

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
        os.getenv("WATCH_SYMBOLS", os.getenv("WATCH_SYMBOL", "SPY,QQQ,AAPL,MSFT,NVDA")),
    )
    default_volatile = os.getenv(
        "WATCH_SYMBOLS_VOLATILE",
        "TSLA,AMD,META,PLTR,SMCI,COIN,MSTR,SOXL,IWM",
    )
    default_defensive = os.getenv(
        "WATCH_SYMBOLS_DEFENSIVE",
        "XLU,XLP,XLV,TLT,GLD,SLV,USO,UNG,DBA,PDBC,VNQ,IYR,O,PLD,SPG",
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

    context_symbols = _parse_symbols(args.context_symbols)
    if args.broker == "coinbase":
        symbols = [CoinbaseMarketDataClient.normalize_symbol(s) for s in symbols]
        context_symbols = [CoinbaseMarketDataClient.normalize_symbol(s) for s in context_symbols]
        # For crypto mode, if context stayed at equity defaults, swap to crypto context.
        if context_symbols == ["$VIX.X", "UUP"]:
            context_symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

    print(
        f"Symbol groups: core={len(_parse_symbols(args.symbols_core))} "
        f"volatile={len(_parse_symbols(args.symbols_volatile))} "
        f"defensive={len(_parse_symbols(args.symbols_defensive))} total={len(symbols)} broker={args.broker}"
    )
    os.environ["CORE_SYMBOL_EVERY_N_ITERS"] = str(max(args.core_every_n_iters, 1))
    os.environ["VOLATILE_SYMBOL_EVERY_N_ITERS"] = str(max(args.volatile_every_n_iters, 1))
    os.environ["DEFENSIVE_SYMBOL_EVERY_N_ITERS"] = str(max(args.defensive_every_n_iters, 1))

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
        volatile_symbols=_parse_symbols(args.symbols_volatile),
        defensive_symbols=_parse_symbols(args.symbols_defensive),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user, stopping shadow loop.")
