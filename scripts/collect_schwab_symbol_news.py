#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.base_trader import BaseTrader
from core.market_context_features import summarize_structured_news_items
from scripts.ops.sleeve_ticker_universe_expansion import UNIVERSES, build_payload as build_universe_payload


HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "schwab_symbol_news_latest.json"
EXTERNAL_CONTEXT_PATH = PROJECT_ROOT / "exports" / "external_context" / "schwab_symbol_news_latest.json"
EVENT_DIR = PROJECT_ROOT / "governance" / "events"

NEWS_ROW_KEYS = {
    "headline",
    "title",
    "summary",
    "description",
    "content",
    "url",
    "link",
    "published",
    "publishedDate",
    "dateTime",
    "datetime",
    "timestamp",
    "displayDate",
}
PAYLOAD_LIST_KEYS = ("items", "stories", "articles", "results", "headlines", "data", "news")
SOURCE_KEYS = ("source", "publisher", "provider", "sourceName", "channel", "vendor")
TITLE_KEYS = ("headline", "title", "storyHeadline", "headlineText", "name")
SUMMARY_KEYS = ("summary", "description", "abstract", "snippet", "content", "body")
URL_KEYS = ("url", "link", "storyUrl", "webUrl", "canonicalUrl")
TS_KEYS = ("publishedDate", "published", "dateTime", "datetime", "timestamp", "time", "displayDate", "created")
POSITIVE_TOKENS = {
    "beat",
    "beats",
    "bullish",
    "breakout",
    "growth",
    "gain",
    "gains",
    "surge",
    "surges",
    "rally",
    "record",
    "upgrade",
    "upgrades",
    "raises",
    "strong",
    "strength",
    "upside",
}
NEGATIVE_TOKENS = {
    "miss",
    "misses",
    "bearish",
    "drop",
    "drops",
    "selloff",
    "sell-off",
    "slump",
    "warning",
    "cuts",
    "cut",
    "downgrade",
    "downgrades",
    "weak",
    "weakness",
    "lawsuit",
    "probe",
}
CATALYST_TOKENS: dict[str, tuple[str, ...]] = {
    "earnings": ("earnings", "eps", "revenue", "results", "quarter"),
    "guidance": ("guidance", "outlook", "forecast", "raises", "cuts"),
    "analyst": ("upgrade", "downgrade", "price target", "initiates", "rating"),
    "regulatory": ("sec", "fda", "lawsuit", "investigation", "probe", "antitrust"),
    "m_and_a": ("merger", "acquisition", "buyout", "takeover", "deal"),
    "crypto": ("bitcoin", "crypto", "coinbase", "mining", "miner", "hashrate"),
    "rates_macro": ("fed", "rates", "yield", "inflation", "jobs", "treasury"),
    "commodity": ("oil", "gold", "crude", "natural gas", "copper"),
    "momentum": ("rally", "surge", "breakout", "record high", "volume"),
}
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
NON_SYMBOL_ENV_SUFFIXES = (
    "_COUNT",
    "_MAX_SYMBOLS",
    "_MAX_RUNTIME_SECONDS",
    "_TIMEOUT_SECONDS",
    "_SLEEP_SECONDS",
    "_LIMIT_PER_SYMBOL",
    "_MAX_ARCHIVE_FETCHES",
    "_PROFILE",
    "_POLICY",
    "_ENABLED",
)
BROAD_MARKET_FALLBACK_SYMBOLS = (
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "TLT",
    "IEF",
    "GLD",
    "UUP",
    "HYG",
    "LQD",
    "XLK",
    "XLF",
    "XLV",
    "XLE",
    "XLU",
    "XLI",
    "XLY",
)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in str(raw).replace("\n", ",").split(","):
        sym = part.strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _is_universe_symbol(raw: str) -> bool:
    sym = str(raw or "").strip().upper()
    if not sym:
        return False
    if sym.endswith("-USD"):
        return bool(re.match(r"^[A-Z0-9]+-USD$", sym))
    return bool(SYMBOL_RE.match(sym))


def _is_symbol_env_group_key(raw: str) -> bool:
    key = str(raw or "").strip().upper()
    if not key:
        return False
    if key.startswith("SLEEVE_TICKER_UNIVERSE_"):
        return False
    if any(key.endswith(suffix) for suffix in NON_SYMBOL_ENV_SUFFIXES):
        return False
    return key.endswith("_SYMBOLS")


def _truthy(raw: Any) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _ordered_unique(items: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = str(raw or "").strip().upper()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _runtime_value(env_overrides: Mapping[str, Any], key: str, default: str = "") -> str:
    raw = os.getenv(key)
    if raw is not None and str(raw).strip():
        return str(raw)
    raw = env_overrides.get(key)
    if raw is not None and str(raw).strip():
        return str(raw)
    return default


def _storage_pressure_snapshot(project_root: Path) -> dict[str, Any]:
    payload = _load_json(project_root / "governance" / "health" / "ingestion_storage_control_latest.json")
    if not payload:
        return {"active": False, "reason": "storage_control_missing"}

    status = str(payload.get("overall_status") or "").strip().lower()
    severity = str(payload.get("severity") or "").strip().lower()
    backpressure = payload.get("backpressure") if isinstance(payload.get("backpressure"), Mapping) else {}
    effective = backpressure.get("effective_raw_live") if isinstance(backpressure.get("effective_raw_live"), Mapping) else {}
    steady_state = payload.get("steady_state") if isinstance(payload.get("steady_state"), Mapping) else {}
    targets = steady_state.get("targets") if isinstance(steady_state.get("targets"), Mapping) else {}
    target_status = steady_state.get("target_status") if isinstance(steady_state.get("target_status"), Mapping) else {}

    total_pending = _safe_int(effective.get("total_pending_lines"), _safe_int(backpressure.get("total_pending_lines"), 0))
    core_pending = _safe_int(effective.get("core_pending_lines"), _safe_int(backpressure.get("core_pending_lines"), total_pending))
    stale_pending = _safe_int(effective.get("stale_stage_pending_lines"), _safe_int(backpressure.get("stale_stage_pending_lines"), 0))
    oldest_age = _safe_float(effective.get("oldest_pending_age_seconds"), _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0))
    total_target = _safe_int(targets.get("total_pending_lines"), _safe_int(backpressure.get("total_pending_lines_threshold"), 15000)) or 15000
    core_target = _safe_int(targets.get("core_pending_lines"), _safe_int(backpressure.get("pending_lines_threshold"), 5000)) or 5000
    oldest_target = _safe_float(targets.get("oldest_pending_age_seconds"), _safe_float(backpressure.get("oldest_age_threshold_seconds"), 600.0)) or 600.0
    steady_ready = bool(target_status.get("steady_state_ready", True))
    target_breaches = _safe_int(target_status.get("target_breach_count"), 0)

    active = bool(
        status not in {"", "ready"}
        or severity not in {"", "stable"}
        or total_pending > total_target
        or core_pending > core_target
        or stale_pending > 0
        or oldest_age > oldest_target
        or not steady_ready
        or target_breaches > 0
    )
    return {
        "active": active,
        "reason": "storage_pressure_active" if active else "storage_stable",
        "overall_status": status or "unknown",
        "severity": severity or "unknown",
        "core_pending_lines": core_pending,
        "total_pending_lines": total_pending,
        "stale_stage_pending_lines": stale_pending,
        "oldest_pending_age_seconds": round(oldest_age, 3),
        "core_pending_target": core_target,
        "total_pending_target": total_target,
        "oldest_pending_age_target": round(oldest_target, 3),
        "steady_state_ready": steady_ready,
        "target_breach_count": target_breaches,
    }


def _apply_slow_tier_storage_policy(
    project_root: Path,
    symbols: list[str],
    env_overrides: Mapping[str, Any],
    *,
    manual_override: bool,
) -> tuple[list[str], dict[str, Any]]:
    snapshot = _storage_pressure_snapshot(project_root)
    enabled = _truthy(_runtime_value(env_overrides, "TICKER_UNIVERSE_SLOW_TIER_DEFER_ON_STORAGE_PRESSURE", "0"))
    policy: dict[str, Any] = {
        "storage_profile": _runtime_value(env_overrides, "TICKER_UNIVERSE_STORAGE_PROFILE", ""),
        "slow_tier_defer_on_storage_pressure": enabled,
        "storage_pressure_active": bool(snapshot.get("active", False)),
        "storage_pressure": snapshot,
        "manual_override": bool(manual_override),
        "pre_policy_symbol_count": len(symbols),
        "active_symbol_count": len(symbols),
        "deferred_symbol_count": 0,
        "mode": "all_symbols",
    }
    if manual_override or not enabled or not snapshot.get("active", False):
        return symbols, policy

    slow_symbols = set(_parse_csv(_runtime_value(env_overrides, "TICKER_UNIVERSE_SLOW_SYMBOLS", "")))
    standard_symbols = _parse_csv(_runtime_value(env_overrides, "TICKER_UNIVERSE_STANDARD_SYMBOLS", ""))
    if slow_symbols:
        active_symbols = [symbol for symbol in symbols if symbol not in slow_symbols]
    elif standard_symbols:
        standard_set = set(standard_symbols)
        active_symbols = [symbol for symbol in symbols if symbol in standard_set]
    else:
        active_symbols = symbols[:500]

    if not active_symbols:
        active_symbols = symbols[:500]
    active_set = set(active_symbols)
    deferred_symbols = [symbol for symbol in symbols if symbol not in active_set]
    policy.update(
        {
            "mode": "slow_tier_deferred_for_storage_pressure",
            "active_symbol_count": len(active_symbols),
            "deferred_symbol_count": len(deferred_symbols),
            "deferred_symbol_examples": deferred_symbols[:20],
        }
    )
    return active_symbols, policy


def load_ticker_universe_with_policy(
    project_root: Path = PROJECT_ROOT,
    *,
    override_symbols: str | None = None,
) -> tuple[list[str], dict[str, list[str]], str, dict[str, Any]]:
    explicit = _parse_csv(override_symbols)
    if explicit:
        policy = {
            "storage_profile": "",
            "slow_tier_defer_on_storage_pressure": False,
            "storage_pressure_active": False,
            "manual_override": True,
            "pre_policy_symbol_count": len(explicit),
            "active_symbol_count": len(explicit),
            "deferred_symbol_count": 0,
            "mode": "manual_override",
        }
        return explicit, {symbol: ["manual_override"] for symbol in explicit}, "manual_override", policy

    payload = _load_json(project_root / "governance" / "health" / "sleeve_ticker_universe_latest.json")
    source = "sleeve_ticker_universe_latest"
    env_overrides = payload.get("env_overrides") if isinstance(payload.get("env_overrides"), dict) else {}
    if not env_overrides:
        payload = build_universe_payload(project_root)
        env_overrides = payload.get("env_overrides") if isinstance(payload.get("env_overrides"), dict) else {}
        source = "sleeve_ticker_universe_static"

    symbols: list[str] = []
    groups: dict[str, list[str]] = {}
    for key, raw in env_overrides.items():
        key_text = str(key or "").strip()
        if not _is_symbol_env_group_key(key_text):
            continue
        if not isinstance(raw, str):
            continue
        for symbol in _parse_csv(raw):
            if not _is_universe_symbol(symbol):
                continue
            symbols.append(symbol)
            groups.setdefault(symbol, []).append(key_text)

    if not symbols:
        source = "sleeve_ticker_universe_static_fallback"
        for key, values in UNIVERSES.items():
            for symbol in values:
                sym = str(symbol or "").strip().upper()
                if not sym:
                    continue
                symbols.append(sym)
                groups.setdefault(sym, []).append(key)

    unique = _ordered_unique(symbols)
    groups = {symbol: groups.get(symbol, []) for symbol in unique}
    unique, policy = _apply_slow_tier_storage_policy(project_root, unique, env_overrides, manual_override=False)
    groups = {symbol: groups.get(symbol, []) for symbol in unique}
    if policy.get("deferred_symbol_count", 0):
        source = f"{source}+slow_tier_deferred"
    return unique, groups, source, policy


def load_ticker_universe(project_root: Path = PROJECT_ROOT, *, override_symbols: str | None = None) -> tuple[list[str], dict[str, list[str]], str]:
    symbols, groups, source, _policy = load_ticker_universe_with_policy(project_root, override_symbols=override_symbols)
    return symbols, groups, source


def is_probably_schwab_symbol(symbol: str) -> bool:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return False
    if sym.startswith("/") or ":" in sym:
        return False
    if sym.endswith("-USD"):
        return False
    return bool(SYMBOL_RE.match(sym))


def _row_get_ci(row: Mapping[str, Any], *keys: str) -> Any:
    lookup = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key.lower() in lookup:
            return lookup[key.lower()]
    return None


def _parse_ts(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value > 1e12:
            value /= 1000.0
        return value if value > 1e9 else None
    text = str(raw or "").strip()
    if not text:
        return None
    if text.isdigit():
        value = float(text)
        if value > 1e12:
            value /= 1000.0
        return value if value > 1e9 else None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp()


def _iso_from_ts(raw: Any) -> str:
    ts = _parse_ts(raw)
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _looks_like_news_row(row: Mapping[str, Any]) -> bool:
    if not isinstance(row, Mapping):
        return False
    keys = {str(key) for key in row.keys()}
    if keys.intersection(NEWS_ROW_KEYS):
        return True
    if _row_get_ci(row, *TITLE_KEYS):
        return True
    return False


def extract_news_items(payload: Any, symbol: str = "") -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    sym = str(symbol or "").strip().upper()

    def consume(value: Any, depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(value, list):
            for row in value:
                consume(row, depth + 1)
            return
        if not isinstance(value, dict):
            return
        if _looks_like_news_row(value):
            items.append(dict(value))
            return
        keys = list(PAYLOAD_LIST_KEYS)
        if sym:
            keys.extend([sym, sym.lower()])
        consumed = False
        for key in keys:
            if key in value:
                consume(value.get(key), depth + 1)
                consumed = True
        if not consumed:
            for sub in value.values():
                if isinstance(sub, (dict, list)):
                    consume(sub, depth + 1)

    consume(payload)
    return items


def _extract_symbols(row: Mapping[str, Any], fallback_symbol: str) -> list[str]:
    values: list[str] = []
    for key in ("symbol", "ticker"):
        raw = _row_get_ci(row, key)
        if isinstance(raw, str):
            values.extend(_parse_csv(raw))
    for key in ("symbols", "tickers", "relatedSymbols", "relatedTickers", "securities"):
        raw = _row_get_ci(row, key)
        if isinstance(raw, str):
            values.extend(_parse_csv(raw))
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    values.extend(_parse_csv(item))
                elif isinstance(item, Mapping):
                    for sub_key in ("symbol", "ticker"):
                        sub = _row_get_ci(item, sub_key)
                        if isinstance(sub, str):
                            values.extend(_parse_csv(sub))
    if fallback_symbol:
        values.append(fallback_symbol)
    return _ordered_unique(values)


def normalize_news_item(row: Mapping[str, Any], *, symbol: str, source_method: str) -> dict[str, Any]:
    headline = str(_row_get_ci(row, *TITLE_KEYS) or "").strip()
    summary = str(_row_get_ci(row, *SUMMARY_KEYS) or "").strip()
    publisher = str(_row_get_ci(row, *SOURCE_KEYS) or "Schwab").strip()
    url = str(_row_get_ci(row, *URL_KEYS) or "").strip()
    raw_ts = None
    for key in TS_KEYS:
        raw_ts = _row_get_ci(row, key)
        if raw_ts:
            break
    published_at = _iso_from_ts(raw_ts)
    return {
        "symbol": str(symbol or "").strip().upper(),
        "headline": headline,
        "summary": summary[:1200],
        "publisher": publisher,
        "source": publisher,
        "url": url,
        "published_at": published_at,
        "timestamp": published_at,
        "symbols": _extract_symbols(row, str(symbol or "").strip().upper()),
        "source_method": str(source_method or ""),
    }


def dedupe_items(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in items:
        headline = str(row.get("headline") or row.get("title") or "").strip().lower()
        url = str(row.get("url") or "").strip().lower()
        ts = str(row.get("published_at") or row.get("timestamp") or "").strip()
        if not headline and not url:
            continue
        key = (headline[:200], url[:240], ts[:32])
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def _item_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        str(_row_get_ci(row, key) or "").strip()
        for key in (*TITLE_KEYS, *SUMMARY_KEYS)
        if str(_row_get_ci(row, key) or "").strip()
    )


def _load_public_schwab_items(project_root: Path) -> list[dict[str, Any]]:
    for path in (
        project_root / "exports" / "external_context" / "schwab_education_context_latest.json",
        project_root / "data" / "external_context" / "schwab_education_context_latest.json",
    ):
        payload = _load_json(path)
        items = payload.get("items") if isinstance(payload.get("items"), list) else []
        if items:
            return [dict(row) for row in items if isinstance(row, Mapping)]
    return []


def public_schwab_fallback_by_symbol(
    project_root: Path,
    symbols: Iterable[str],
    *,
    limit_per_symbol: int,
) -> dict[str, list[dict[str, Any]]]:
    universe = set(_ordered_unique(symbols))
    broad_symbols = [symbol for symbol in BROAD_MARKET_FALLBACK_SYMBOLS if symbol in universe]
    out: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in universe}
    for row in _load_public_schwab_items(project_root):
        text = _item_text(row)
        text_upper = f" {text.upper()} "
        row_symbols = set(_extract_symbols(row, ""))
        candidates = {symbol for symbol in row_symbols if symbol in universe}
        for symbol in universe:
            if len(symbol) < 3:
                continue
            if re.search(rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])", text_upper):
                candidates.add(symbol)
        if bool(row.get("broad_market", False) or row.get("macro_event", False)):
            candidates.update(broad_symbols)
        max_symbols_per_item = max(int(os.getenv("SCHWAB_SYMBOL_NEWS_PUBLIC_FALLBACK_MAX_SYMBOLS_PER_ITEM", "32") or 32), 1)
        for symbol in sorted(candidates)[:max_symbols_per_item]:
            normalized = normalize_news_item(row, symbol=symbol, source_method="schwab_public_context_fallback")
            if not normalized.get("published_at"):
                fetched = str(row.get("fetched_utc") or row.get("timestamp_utc") or "").strip()
                normalized["published_at"] = fetched
                normalized["timestamp"] = fetched
            normalized["fallback_source"] = "schwab_education_context"
            out.setdefault(symbol, []).append(normalized)
    return {symbol: dedupe_items(items)[:limit_per_symbol] for symbol, items in out.items() if items}


def _sentiment(text: str) -> float:
    tokens = [token for token in re.split(r"[^a-z]+", str(text or "").lower()) if token]
    if not tokens:
        return 0.0
    pos = sum(1 for token in tokens if token in POSITIVE_TOKENS)
    neg = sum(1 for token in tokens if token in NEGATIVE_TOKENS)
    if pos + neg <= 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / max(pos + neg, 1)))


def _catalyst_counts(items: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in items:
        text = f"{row.get('headline', '')} {row.get('summary', '')}".lower()
        for catalyst, tokens in CATALYST_TOKENS.items():
            if any(token in text for token in tokens):
                counts[catalyst] += 1
    return dict(counts)


def _symbol_features(symbol: str, items: list[dict[str, Any]], *, now_ts: float, max_items: int) -> dict[str, float]:
    features = summarize_structured_news_items(items, symbol=symbol, now_ts=now_ts, max_items=max_items)
    if not items:
        features.update(
            {
                "schwab_symbol_news_available": 0.0,
                "schwab_symbol_news_item_count_norm": 0.0,
                "schwab_symbol_news_freshness_norm": 0.0,
                "news_sentiment": 0.0,
            }
        )
        return features

    sentiments = [_sentiment(f"{row.get('headline', '')} {row.get('summary', '')}") for row in items]
    dated = [_parse_ts(row.get("published_at") or row.get("timestamp")) for row in items]
    dated = [ts for ts in dated if ts is not None]
    min_age = min((now_ts - ts for ts in dated), default=48.0 * 3600.0)
    features.update(
        {
            "schwab_symbol_news_available": 1.0,
            "schwab_symbol_news_item_count_norm": min(len(items) / max(float(max_items), 1.0), 1.0),
            "schwab_symbol_news_freshness_norm": max(0.0, min(1.0, 1.0 - (min_age / (48.0 * 3600.0)))),
            "news_sentiment": sum(sentiments) / max(len(sentiments), 1),
        }
    )
    catalysts = _catalyst_counts(items)
    total = max(len(items), 1)
    for catalyst, count in catalysts.items():
        features[f"schwab_news_catalyst_{catalyst}_norm"] = min(count / total, 1.0)
    return {key: float(value) for key, value in features.items() if isinstance(value, (int, float)) and math.isfinite(float(value))}


def try_broker_candidate_payload(client: Any, candidates: Iterable[tuple[str, tuple[Any, ...], dict[str, Any]]]) -> tuple[Any | None, str, str]:
    callable_seen = False
    last_error = ""
    for method_name, args, kwargs in candidates:
        method = getattr(client, method_name, None)
        if not callable(method):
            continue
        callable_seen = True
        try:
            resp = method(*args, **kwargs)
        except TypeError as exc:
            last_error = f"TypeError:{exc}"
            continue
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{exc}"
            continue
        if resp is None:
            continue
        if hasattr(resp, "json"):
            try:
                return resp.json(), method_name, ""
            except Exception as exc:
                last_error = f"json_error:{type(exc).__name__}:{exc}"
                continue
        return resp, method_name, ""
    if not callable_seen:
        return None, "", "no_callable_news_method"
    return None, "", last_error or "no_payload"


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _append_event(payload: Mapping[str, Any]) -> None:
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    event_path = EVENT_DIR / f"schwab_symbol_news_{day}.jsonl"
    summary = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "overall_status": payload.get("overall_status"),
        "requested_symbol_count": payload.get("requested_symbol_count"),
        "attempted_symbol_count": payload.get("attempted_symbol_count"),
        "symbols_with_news": payload.get("symbols_with_news"),
        "total_news_items": payload.get("total_news_items"),
        "coverage_ratio": payload.get("coverage_ratio"),
    }
    with event_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=True, separators=(",", ":")) + "\n")


def build_payload(
    *,
    project_root: Path = PROJECT_ROOT,
    symbols_arg: str | None = None,
    max_symbols: int = 0,
    limit_per_symbol: int = 50,
    max_runtime_seconds: float = 240.0,
    sleep_seconds: float = 0.02,
    authenticate: bool = True,
    quiet_auth: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()
    symbols, symbol_groups, universe_source, universe_policy = load_ticker_universe_with_policy(project_root, override_symbols=symbols_arg)
    if max_symbols > 0:
        universe_policy = {
            **universe_policy,
            "max_symbols_applied": int(max_symbols),
            "pre_max_symbol_count": len(symbols),
        }
        symbols = symbols[:max_symbols]
        universe_policy["active_symbol_count"] = len(symbols)
    limit_per_symbol = max(int(limit_per_symbol), 1)

    base_payload: dict[str, Any] = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "collector": "schwab_symbol_news",
        "universe_source": universe_source,
        "universe_policy": universe_policy,
        "requested_symbol_count": len(symbols),
        "limit_per_symbol": limit_per_symbol,
        "safety_contract": {
            "market_data_only": True,
            "live_execution_allowed": False,
            "writes_orders": False,
            "protected_volumes": ["/Volumes/VIDEO"],
            "scope": "broker_native_symbol_news_context",
        },
    }

    if not authenticate:
        payload = {
            **base_payload,
            "ok": False,
            "overall_status": "preview_only",
            "attempted_symbol_count": 0,
            "symbols_with_news": 0,
            "total_news_items": 0,
            "coverage_ratio": 0.0,
            "symbols": {},
            "derived": {"news_features": {}, "news_symbol_features": {}, "symbol_features": {}, "global_features": {}},
            "recommended_actions": ["run without --no-auth to collect Schwab symbol news"],
        }
        return payload

    try:
        trader = BaseTrader.from_env(mode="shadow", broker="schwab")
        if quiet_auth:
            with contextlib.redirect_stdout(sys.stderr):
                client = trader.authenticate()
        else:
            client = trader.authenticate()
    except Exception as exc:
        symbol_rows = {
            symbol: {
                "status": "auth_blocked",
                "ok": False,
                "groups": symbol_groups.get(symbol, []),
                "item_count": 0,
                "items": [],
                "error": f"{type(exc).__name__}:{exc}",
            }
            for symbol in symbols
        }
        return {
            **base_payload,
            "ok": False,
            "overall_status": "blocked_auth",
            "auth_ok": False,
            "auth_error": f"{type(exc).__name__}:{exc}",
            "attempted_symbol_count": 0,
            "symbols_with_news": 0,
            "total_news_items": 0,
            "coverage_ratio": 0.0,
            "symbols": symbol_rows,
            "derived": {"news_features": {}, "news_symbol_features": {}, "symbol_features": {}, "global_features": {}},
            "recommended_actions": [
                "./scripts/ops/opsctl.sh token-refresh --json",
                "./scripts/ops/opsctl.sh schwab-symbol-news-sync --json",
            ],
        }

    symbol_rows: dict[str, dict[str, Any]] = {}
    symbol_features: dict[str, dict[str, float]] = {}
    all_items: list[dict[str, Any]] = []
    attempted = 0
    stopped_for_runtime = False
    no_callable_count = 0
    method_counts: Counter[str] = Counter()

    for symbol in symbols:
        if max_runtime_seconds > 0 and (time.monotonic() - started) >= max_runtime_seconds:
            stopped_for_runtime = True
            break
        attempted += 1
        candidates = trader.broker_adapter.news_candidates(symbol=symbol, limit=limit_per_symbol)
        payload, method_name, error = try_broker_candidate_payload(client, candidates)
        if error == "no_callable_news_method":
            no_callable_count += 1
        raw_items = extract_news_items(payload, symbol) if payload is not None else []
        items = dedupe_items(
            normalize_news_item(row, symbol=symbol, source_method=method_name)
            for row in raw_items
            if isinstance(row, Mapping)
        )[:limit_per_symbol]
        status = "ok" if items else ("no_endpoint" if error == "no_callable_news_method" else "empty")
        if not is_probably_schwab_symbol(symbol) and not items:
            status = "unsupported_or_empty"
        method_counts.update([method_name or "none"])
        features = _symbol_features(symbol, items, now_ts=now_ts, max_items=limit_per_symbol)
        symbol_features[symbol] = features
        all_items.extend(items)
        symbol_rows[symbol] = {
            "status": status,
            "ok": bool(items),
            "groups": symbol_groups.get(symbol, []),
            "schwab_symbol_supported_hint": is_probably_schwab_symbol(symbol),
            "source_method": method_name,
            "item_count": len(items),
            "feature_summary": features,
            "catalyst_counts": _catalyst_counts(items),
            "items": items,
            "error": error,
        }
        if sleep_seconds > 0:
            time.sleep(max(float(sleep_seconds), 0.0))

    missing_symbols = symbols[attempted:]
    for symbol in missing_symbols:
        symbol_rows[symbol] = {
            "status": "deferred_runtime_budget",
            "ok": False,
            "groups": symbol_groups.get(symbol, []),
            "schwab_symbol_supported_hint": is_probably_schwab_symbol(symbol),
            "item_count": 0,
            "items": [],
        }
        symbol_features[symbol] = _symbol_features(symbol, [], now_ts=now_ts, max_items=limit_per_symbol)

    fallback_active = False
    fallback_symbol_count = 0
    if attempted > 0 and no_callable_count == attempted:
        fallback_items = public_schwab_fallback_by_symbol(project_root, symbols, limit_per_symbol=limit_per_symbol)
        fallback_symbol_count = len(fallback_items)
        fallback_active = fallback_symbol_count > 0
        for symbol, items in fallback_items.items():
            features = _symbol_features(symbol, items, now_ts=now_ts, max_items=limit_per_symbol)
            symbol_features[symbol] = features
            row = symbol_rows.setdefault(
                symbol,
                {
                    "groups": symbol_groups.get(symbol, []),
                    "schwab_symbol_supported_hint": is_probably_schwab_symbol(symbol),
                },
            )
            row.update(
                {
                    "status": "public_schwab_fallback",
                    "ok": True,
                    "source_method": "schwab_public_context_fallback",
                    "item_count": len(items),
                    "feature_summary": features,
                    "catalyst_counts": _catalyst_counts(items),
                    "items": items,
                    "error": "broker_native_news_endpoint_unavailable_public_schwab_fallback_used",
                }
            )

    all_items = dedupe_items(all_items)
    if fallback_active:
        for row in symbol_rows.values():
            items = row.get("items") if isinstance(row.get("items"), list) else []
            all_items.extend(dict(item) for item in items if isinstance(item, Mapping))
        all_items = dedupe_items(all_items)
    symbols_with_news = sum(1 for row in symbol_rows.values() if int(row.get("item_count", 0) or 0) > 0)
    coverage_ratio = symbols_with_news / max(len(symbols), 1)
    auth_ok = True
    if attempted > 0 and no_callable_count == attempted and fallback_active:
        overall_status = "ready_public_schwab_fallback"
    elif attempted > 0 and no_callable_count == attempted:
        overall_status = "degraded_no_broker_news_endpoint"
    elif stopped_for_runtime:
        overall_status = "partial_runtime_budget"
    elif attempted == len(symbols):
        overall_status = "ready"
    else:
        overall_status = "partial"

    source_counts = Counter(str(item.get("publisher") or item.get("source") or "unknown") for item in all_items)
    global_news_features: dict[str, float] = {}
    for features in symbol_features.values():
        for key, raw in features.items():
            try:
                value = float(raw)
            except Exception:
                continue
            if key == "news_sentiment":
                current = float(global_news_features.get(key, 0.0) or 0.0)
                if abs(value) >= abs(current):
                    global_news_features[key] = value
            else:
                global_news_features[key] = max(float(global_news_features.get(key, 0.0) or 0.0), value)
    global_news_features.update(
        {
            "schwab_symbol_news_coverage_norm": max(0.0, min(coverage_ratio, 1.0)),
            "schwab_symbol_news_total_items_norm": min(len(all_items) / max(len(symbols) * limit_per_symbol, 1), 1.0),
        }
    )
    payload = {
        **base_payload,
        "ok": bool(auth_ok and attempted > 0 and overall_status != "degraded_no_broker_news_endpoint"),
        "overall_status": overall_status,
        "auth_ok": auth_ok,
        "attempted_symbol_count": attempted,
        "deferred_symbol_count": max(len(symbols) - attempted, 0),
        "symbols_with_news": symbols_with_news,
        "total_news_items": len(all_items),
        "coverage_ratio": round(coverage_ratio, 6),
        "broker_native_news_endpoint_available": not (attempted > 0 and no_callable_count == attempted),
        "fallback_active": fallback_active,
        "fallback_mode": "schwab_public_context" if fallback_active else "",
        "fallback_symbol_count": fallback_symbol_count,
        "method_counts": dict(method_counts),
        "source_counts": dict(source_counts.most_common(20)),
        "symbols": symbol_rows,
        "items_by_symbol": {symbol: row.get("items", []) for symbol, row in symbol_rows.items()},
        "derived": {
            "news_features": global_news_features,
            "news_symbol_features": symbol_features,
            "symbol_features": {
                symbol: {
                    "external_context_source_available": 1.0 if bool(row.get("ok", False)) else 0.0,
                    "external_context_symbol_coverage_norm": float(symbol_features.get(symbol, {}).get("schwab_symbol_news_item_count_norm", 0.0) or 0.0),
                }
                for symbol, row in symbol_rows.items()
            },
            "global_features": {
                "external_context_schwab_symbol_news_available": 1.0 if symbols_with_news > 0 else 0.0,
                "external_context_schwab_symbol_news_coverage_norm": max(0.0, min(coverage_ratio, 1.0)),
            },
        },
        "recommended_actions": [
            "refresh Schwab auth token before rerunning symbol news sync"
            if not auth_ok
            else "Schwab client has no callable broker-native news endpoint; keep Schwab education and external context feeds active"
            if overall_status == "degraded_no_broker_news_endpoint"
            else "rerun with a higher --max-runtime-seconds to finish deferred symbols"
            if stopped_for_runtime
            else "Schwab symbol news context is current; decision loops can consume schwab_symbol_news external context",
        ],
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect broker-native Schwab symbol news for the active ticker universe.")
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbol override.")
    parser.add_argument("--max-symbols", type=int, default=int(os.getenv("SCHWAB_SYMBOL_NEWS_MAX_SYMBOLS", "0") or 0))
    parser.add_argument("--limit-per-symbol", type=int, default=int(os.getenv("SCHWAB_SYMBOL_NEWS_LIMIT_PER_SYMBOL", "50") or 50))
    parser.add_argument("--max-runtime-seconds", type=float, default=float(os.getenv("SCHWAB_SYMBOL_NEWS_MAX_RUNTIME_SECONDS", "240") or 240))
    parser.add_argument("--sleep-seconds", type=float, default=float(os.getenv("SCHWAB_SYMBOL_NEWS_SLEEP_SECONDS", "0.02") or 0.02))
    parser.add_argument("--no-auth", action="store_true", help="Preview the universe without authenticating or fetching.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(
        project_root=PROJECT_ROOT,
        symbols_arg=args.symbols,
        max_symbols=int(args.max_symbols),
        limit_per_symbol=int(args.limit_per_symbol),
        max_runtime_seconds=float(args.max_runtime_seconds),
        sleep_seconds=float(args.sleep_seconds),
        authenticate=not bool(args.no_auth),
        quiet_auth=bool(args.json),
    )
    _write_payload(HEALTH_PATH, payload)
    _write_payload(EXTERNAL_CONTEXT_PATH, payload)
    _append_event(payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "schwab_symbol_news "
            f"status={payload.get('overall_status')} "
            f"symbols={payload.get('attempted_symbol_count')}/{payload.get('requested_symbol_count')} "
            f"with_news={payload.get('symbols_with_news')} "
            f"items={payload.get('total_news_items')}"
        )
    return 0 if payload.get("overall_status") not in {"blocked_auth"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
