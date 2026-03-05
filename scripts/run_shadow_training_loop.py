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
        max(float(os.getenv("CANARY_WEIGHT_SANDBOX_TOTAL_MAX", "0.12")), 0.0),
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
]


def _default_news_features() -> Dict[str, float]:
    return {k: 0.0 for k in _NEWS_FEATURE_KEYS}


def _default_dividend_features() -> Dict[str, float]:
    return {k: 0.0 for k in _DIVIDEND_FEATURE_KEYS}


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

    if closes and last_price > 0.0:
        hist_last = closes[-1]
        if hist_last > 0.0:
            max_quote_dev = max(float(os.getenv("MARKET_SNAPSHOT_MAX_QUOTE_DEVIATION", "0.35")), 0.05)
            rel_dev = abs(last_price - hist_last) / max(hist_last, 1e-8)
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
    }
    out.update(futures_quote)
    out.update(_default_dividend_features())
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


def _specialist_bot_signal(bot: SubBot, features: Dict[str, float], *, segment: str) -> tuple[str, float, float, List[str]]:
    acc = bot.test_accuracy if bot.test_accuracy is not None else 0.53
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
        edge = _calibrate_vote(edge, scale=0.95)
        score = _vote_to_score(edge)
        threshold = _shift_threshold(0.57)
        reasons = [
            "options_specialist_signal",
            f"iv={iv:.3f}",
            f"skew={skew:.3f}",
            f"pcr_norm={pcr + 0.5:.3f}",
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
        edge = _calibrate_vote(edge, scale=0.90)
        score = _vote_to_score(edge)
        threshold = _shift_threshold(0.56)
        reasons = [
            "futures_specialist_signal",
            f"imbalance={imbalance:.3f}",
            f"funding_norm={funding + 0.5:.3f}",
            f"basis_norm={basis + 0.5:.3f}",
        ]

    score = 0.5 + (score - 0.5) * min(max((0.70 + 2.0 * (acc - 0.5)), 0.55), 1.20)
    score = min(max(score, 0.01), 0.99)

    if score >= threshold:
        action = "BUY"
    elif score <= (1.0 - threshold):
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

    contracts = 1
    if can_cover:
        contracts = max(covered_call_shares // 100, 1)

    def _leg(side: str, option_type: str, strike_mult: float, expiry_days: int, qty: int = 1) -> Dict[str, Any]:
        return {
            "side": side,
            "type": option_type,
            "strike": round(px * strike_mult, 2),
            "expiry_days": int(expiry_days),
            "quantity": int(max(qty, 1)),
        }

    # Income-first covered overlays when we have shares.
    if can_cover and master_action in {"HOLD", "SELL"} and range_pos > 0.60 and (not high_vol_regime):
        action = "SELL_TO_OPEN"
        score = max(0.60, master_score)
        reasons = [
            "covered_call_income_setup",
            "has_covered_shares",
            f"iv_term_norm={iv_term_norm:.3f}",
            f"neg_bias={neg_bias:.3f}",
        ]
        plan = {
            "symbol": symbol,
            "options_style": "COVERED_CALL",
            "strategy_family": "income",
            "underlying_price": px,
            "dte_days": front_dte,
            "contracts": contracts,
            "legs": [_leg("SELL_TO_OPEN", "CALL", 1.03, front_dte, contracts)],
            "strike": round(px * 1.03, 2),
        }

    elif master_action == "BUY" and master_score >= threshold:
        if high_vol_regime and low_event_risk and liquid_chain and (not bearish_bias):
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
        if high_vol_regime and low_event_risk and liquid_chain:
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
        if high_vol_regime and low_event_risk and liquid_chain and expiry_week < 0.8:
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
    "canary_weight_cap_norm",
    "snapshot_cov_fill_ratio",
    "snapshot_replay_ok",
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
        'canary_weight_cap_norm': _behavior_clamp01(float(snapshot_ctx.get('canary_weight_cap_norm', 0.0) or 0.0)),
        'snapshot_cov_fill_ratio': _behavior_clamp01(float(snapshot_ctx.get('snapshot_cov_fill_ratio', 0.0) or 0.0)),
        'snapshot_replay_ok': _behavior_clamp01(float(snapshot_ctx.get('snapshot_replay_ok', 0.0) or 0.0)),
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

    @staticmethod
    def _flush_batch(path: str, rows: List[Dict[str, Any]]) -> None:
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
            return

        if not _route_storage_or_fail():
            print(f"[StorageRoute] write_failed path={path} wrote={wrote} expected={len(rows)}")
            JsonlWriteBuffer._emit_write_failure_event(path=path, error=RuntimeError("batch_write_failed"))
            return

        retry_wrote = _write_once()
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
    if _shadow_profile_name() != "dividend":
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


def _update_dividend_compound_metrics(
    *,
    symbol: str,
    symbol_return_1m: float,
    features: Dict[str, float],
    state: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    if _shadow_profile_name() != "dividend":
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


def _governance_path(project_root: str, broker: Optional[str] = None) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return governance_master_control_path(project_root, _path_context(broker=broker), day=day)


def _event_bus_path(project_root: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return runtime_event_legacy_path(project_root, day=day)


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
    dividend_compound_state: Dict[str, Dict[str, float]] = {}
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

    bot_cooldown_enabled = os.getenv("BOT_COOLDOWN_ENABLED", "1").strip() == "1"
    bot_cooldown_acc_floor = float(os.getenv("BOT_COOLDOWN_ACC_FLOOR", "0.53"))
    bot_cooldown_min_iters = max(int(os.getenv("BOT_COOLDOWN_MIN_ITERS", "2")), 1)
    bot_cooldown_max_iters = max(int(os.getenv("BOT_COOLDOWN_MAX_ITERS", "4")), bot_cooldown_min_iters)
    bot_next_eval_iter: Dict[str, int] = {}


    derivatives_specialists_enabled = os.getenv("DERIVATIVES_SPECIALISTS_ENABLED", "1").strip() == "1"
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
    options_specialist_weight = max(float(os.getenv("OPTIONS_SPECIALIST_WEIGHT", "0.018")), 0.001)
    futures_specialist_weight = max(float(os.getenv("FUTURES_SPECIALIST_WEIGHT", "0.018")), 0.001)
    specialist_min_acc = float(os.getenv("DERIVATIVES_SPECIALIST_MIN_ACC", "0.54"))

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
    master_latency_slo_timeout_ms = float(os.getenv('MASTER_LATENCY_SLO_TIMEOUT_MS', '800'))
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
        "dividend_strategy_mode": _dividend_strategy_mode() if (_shadow_profile_name() == "dividend") else "n/a",
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
    if _shadow_profile_name() == "dividend":
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
            time.sleep(max(effective_interval_seconds, 10))
            continue

        anomaly_paused = bool(time.time() < anomaly_kill_switch_until_ts)
        _log_gate("*", "anomaly_killswitch", not anomaly_paused, reason=("data_anomaly" if anomaly_paused else "clear"))
        if anomaly_paused:
            rem = int(anomaly_kill_switch_until_ts - time.time())
            print(f"[KillSwitch] paused reason=data_anomaly remaining_s={max(rem, 0)}")
            _record_snapshot_debug('*', 'anomaly_killswitch_paused', remaining_seconds=max(rem, 0))
            _set_loop_state("paused_anomaly_killswitch", reason="data_anomaly", remaining_seconds=max(rem, 0))
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
        if derivatives_specialists_enabled:
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

                state_cache.set(cache_key, snap)
                circuit_breaker.record_success(cb_key)
                latency_ms = (time.perf_counter() - started) * 1000.0
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
                feats = default_calendar_features()
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

            freshness_ok, freshness_reason, freshness_age_s = _feature_freshness_guard(
                shared_features,
                max_age_seconds=feature_freshness_max_age_seconds,
                required_keys=feature_freshness_required,
            ) if feature_freshness_enabled else (True, 'disabled', 0.0)
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
            }

            if _shadow_profile_name() == "dividend":
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

            if _shadow_profile_name() == "dividend":
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
            )
            if pause_active and gm_action in {"BUY", "SELL"}:
                gm_action = "HOLD"
                gm_score = 0.5 + 0.5 * (gm_score - 0.5)
                gm_reasons = gm_reasons + ["circuit_breaker_pause_active"]

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
            }
            grand_gates = {
                "ensemble_has_members": active_sub_bots_total > 0,
                "market_data_ok": mkt["last_price"] > 0,
                "risk_limit_ok": all(risk_gates.values()),
                "feature_freshness_ok": freshness_ok,
                "master_latency_slo_ok": master_latency_slo_ok,
                **risk_gates,
                "exec_queue_ok": enq_ok,
            }
            guard_blocked_intent = gm_intent_action in {"BUY", "SELL"} and gm_action != gm_intent_action

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
                "flash_aux": flash_aux,
                "specialist_votes": {"options": float(shared_features.get("options_specialist_vote", 0.0) or 0.0), "futures": float(shared_features.get("futures_specialist_vote", 0.0) or 0.0)},
                "specialist_rows": {"options": options_specialist_rows, "futures": futures_specialist_rows},
                "options_chain_features": {k: float(shared_features.get(k, 0.0) or 0.0) for k in OPTIONS_FEATURE_KEYS},
                "futures_features": {k: float(shared_features.get(k, 0.0) or 0.0) for k in FUTURES_FEATURE_KEYS},
                "calendar_features": {k: float(shared_features.get(k, 0.0) or 0.0) for k in CALENDAR_FEATURE_KEYS},
                "dividend_features": {k: float(shared_features.get(k, 0.0) or 0.0) for k in _DIVIDEND_FEATURE_KEYS},
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
                "intent_action": gm_intent_action,
                "intent_blocked": guard_blocked_intent,
                "score": gm_score,
                "risk_limit_ok": all(risk_gates.values()),
                "slippage_bps": exec_sim.slippage_bps,
            })

            intent_suffix = f" intent_action={gm_intent_action}" if guard_blocked_intent else ""
            print(
                f"[ShadowLoop] iter={iter_count} symbol={symbol} price={mkt['last_price']:.2f} "
                f"grand_action={gm_action}{intent_suffix} options_master={optm_action} options_action={options_decision['action']} "
                f"active_bots={active_sub_bots_total} core={len(active_bots)} "
                f"opts={active_options_sub_bots} fut={active_futures_sub_bots} "
                f"recs={len(recs)} snapshot_id={snapshot_id}"
            )

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
        _write_ingress_state(project_root=PROJECT_ROOT, broker=broker, payload=ingress_state)
        _append_jsonl(
            _event_bus_path(PROJECT_ROOT),
            {
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
            },
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
        os.getenv("WATCH_SYMBOLS", os.getenv("WATCH_SYMBOL", "SPY,QQQ,AAPL,MSFT,NVDA")),
    )
    default_volatile = os.getenv(
        "WATCH_SYMBOLS_VOLATILE",
        "TSLA,AMD,META,PLTR,SMCI,COIN,MSTR,SOXL,IWM",
    )
    default_defensive = os.getenv(
        "WATCH_SYMBOLS_DEFENSIVE",
        "XLU,XLP,XLV,TLT,GLD,SLV,USO,UNG,DBA,PDBC,VNQ,IYR,O,PLD,SPG,ITA,LMT,NOC,RTX,GD,LHX,LDOS",
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

    context_symbols = _parse_symbols(args.context_symbols)
    if args.broker == "coinbase":
        symbols = [CoinbaseMarketDataClient.normalize_symbol(s) for s in symbols]
        context_symbols = [CoinbaseMarketDataClient.normalize_symbol(s) for s in context_symbols]
        # For crypto mode, if context stayed at equity defaults (before or after normalization),
        # swap to crypto context symbols to avoid repeated 404 fetches.
        equity_context_defaults = {"$VIX.X", "UUP", "$VIX.X-USD", "UUP-USD"}
        if context_symbols and all(s in equity_context_defaults for s in context_symbols):
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
