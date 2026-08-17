#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlencode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.collector_transport import fetch_text
from core.market_context_features import summarize_structured_news_items
from scripts.collect_schwab_symbol_news import (
    CATALYST_TOKENS,
    NEGATIVE_TOKENS,
    POSITIVE_TOKENS,
    dedupe_items,
    is_probably_schwab_symbol,
    load_ticker_universe_with_policy,
    normalize_news_item,
)


HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "ticker_news_context_latest.json"
EXTERNAL_CONTEXT_PATH = PROJECT_ROOT / "exports" / "external_context" / "ticker_news_context_latest.json"
EVENT_DIR = PROJECT_ROOT / "governance" / "events"
USER_AGENT_DEFAULT = os.getenv("TICKER_NEWS_USER_AGENT", "schwab-trading-bot/1.0")

CRYPTO_NEWS_FEEDS: dict[str, dict[str, Any]] = {
    "coindesk": {
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "publisher": "CoinDesk",
        "source_confidence_norm": 0.90,
    },
    "cointelegraph": {
        "url": "https://cointelegraph.com/rss",
        "publisher": "Cointelegraph",
        "source_confidence_norm": 0.80,
    },
    "decrypt": {
        "url": "https://decrypt.co/feed",
        "publisher": "Decrypt",
        "source_confidence_norm": 0.79,
    },
    "the_block": {
        "url": "https://www.theblock.co/rss.xml",
        "publisher": "The Block",
        "source_confidence_norm": 0.88,
    },
    "bitcoin_magazine": {
        "url": "https://bitcoinmagazine.com/.rss/full/",
        "publisher": "Bitcoin Magazine",
        "source_confidence_norm": 0.74,
    },
}

OPTIONAL_GLOBAL_NEWS_FEEDS: dict[str, dict[str, Any]] = {
    "globenewswire": {
        "url": "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire%20-%20News%20about%20Public%20Companies",
        "publisher": "GlobeNewswire",
        "source_confidence_norm": 0.66,
    },
    "pr_newswire": {
        "url": "https://www.prnewswire.com/rss/news-releases-list.rss",
        "publisher": "PR Newswire",
        "source_confidence_norm": 0.62,
    },
    "business_wire": {
        "url": "https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeEFVQXw==",
        "publisher": "Business Wire",
        "source_confidence_norm": 0.69,
    },
}

CRYPTO_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC-USD": ("bitcoin", "btc", "xbt"),
    "ETH-USD": ("ethereum", "ether", "eth"),
    "SOL-USD": ("solana", "sol"),
    "AVAX-USD": ("avalanche", "avax"),
    "LTC-USD": ("litecoin", "ltc"),
    "LINK-USD": ("chainlink", "link"),
    "DOGE-USD": ("dogecoin", "doge"),
    "XRP-USD": ("xrp", "ripple"),
    "ADA-USD": ("cardano", "ada"),
    "DOT-USD": ("polkadot", "dot"),
    "BCH-USD": ("bitcoin cash", "bch"),
    "UNI-USD": ("uniswap", "uni"),
    "AAVE-USD": ("aave",),
    "ATOM-USD": ("cosmos", "atom"),
    "NEAR-USD": ("near protocol", "near"),
    "OP-USD": ("optimism", "op"),
    "ARB-USD": ("arbitrum", "arb"),
    "ETC-USD": ("ethereum classic", "etc"),
    "XLM-USD": ("stellar", "xlm"),
    "HBAR-USD": ("hedera", "hbar"),
    "SUI-USD": ("sui",),
    "INJ-USD": ("injective", "inj"),
    "SEI-USD": ("sei",),
    "TIA-USD": ("celestia", "tia"),
    "PEPE-USD": ("pepe",),
    "SHIB-USD": ("shiba", "shib"),
    "BONK-USD": ("bonk",),
    "WIF-USD": ("dogwifhat", "wif"),
    "ONDO-USD": ("ondo",),
    "RENDER-USD": ("render",),
}

EQUITY_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "AAPL": ("apple",),
    "MSFT": ("microsoft",),
    "NVDA": ("nvidia",),
    "AMD": ("advanced micro devices",),
    "AMZN": ("amazon",),
    "GOOG": ("google", "alphabet"),
    "GOOGL": ("google", "alphabet"),
    "META": ("meta", "facebook"),
    "TSLA": ("tesla",),
    "COIN": ("coinbase",),
    "MSTR": ("microstrategy", "strategy"),
    "HUT": ("hut 8", "hut8"),
    "MARA": ("marathon digital",),
    "RIOT": ("riot platforms",),
    "CLSK": ("cleanspark",),
    "SCHW": ("charles schwab", "schwab"),
}


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        try:
            dt = parsedate_to_datetime(text)
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp()


def _iso_from_any(raw: Any) -> str:
    ts = _parse_ts(raw)
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _text(row: Mapping[str, Any]) -> str:
    return " ".join(str(row.get(key) or "").strip() for key in ("headline", "title", "summary", "description", "content") if str(row.get(key) or "").strip())


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
        text = _text(row).lower()
        for catalyst, tokens in CATALYST_TOKENS.items():
            if any(token in text for token in tokens):
                counts[catalyst] += 1
    return dict(counts)


def _symbol_features(symbol: str, items: list[dict[str, Any]], *, now_ts: float, max_items: int) -> dict[str, float]:
    features = summarize_structured_news_items(items, symbol=symbol, now_ts=now_ts, max_items=max_items)
    if not items:
        features.update(
            {
                "ticker_news_available": 0.0,
                "ticker_news_item_count_norm": 0.0,
                "ticker_news_freshness_norm": 0.0,
                "ticker_news_source_count_norm": 0.0,
                "news_sentiment": 0.0,
            }
        )
        return features
    sentiments = [_sentiment(_text(row)) for row in items]
    dated = [_parse_ts(row.get("published_at") or row.get("timestamp")) for row in items]
    dated = [ts for ts in dated if ts is not None]
    min_age = min((now_ts - ts for ts in dated), default=48.0 * 3600.0)
    sources = {str(row.get("source") or row.get("publisher") or "").strip().lower() for row in items if str(row.get("source") or row.get("publisher") or "").strip()}
    features.update(
        {
            "ticker_news_available": 1.0,
            "ticker_news_item_count_norm": min(len(items) / max(float(max_items), 1.0), 1.0),
            "ticker_news_freshness_norm": max(0.0, min(1.0, 1.0 - (min_age / (48.0 * 3600.0)))),
            "ticker_news_source_count_norm": min(len(sources) / 4.0, 1.0),
            "news_sentiment": sum(sentiments) / max(len(sentiments), 1),
        }
    )
    catalysts = _catalyst_counts(items)
    total = max(len(items), 1)
    for catalyst, count in catalysts.items():
        features[f"ticker_news_catalyst_{catalyst}_norm"] = min(count / total, 1.0)
    return {key: float(value) for key, value in features.items() if isinstance(value, (int, float)) and math.isfinite(float(value))}


def _rss_items(text: str, *, publisher: str, source_id: str, source_confidence_norm: float) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(str(text or "").encode("utf-8"))
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for item in root.findall(".//item") + root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        def first_text(*names: str) -> str:
            for name in names:
                try:
                    found = item.find(name)
                except SyntaxError:
                    found = None
                if found is not None and found.text and found.text.strip():
                    return found.text.strip()
                try:
                    found = item.find(f"{{http://www.w3.org/2005/Atom}}{name}")
                except SyntaxError:
                    found = None
                if found is not None and found.text and found.text.strip():
                    return found.text.strip()
            return ""

        link = first_text("link")
        atom_link = item.find("{http://www.w3.org/2005/Atom}link")
        if not link and atom_link is not None:
            link = str(atom_link.attrib.get("href") or "").strip()
        published = first_text("pubDate", "published", "updated", "dc:date")
        rows.append(
            {
                "headline": first_text("title"),
                "summary": first_text("description", "summary"),
                "url": link,
                "published_at": _iso_from_any(published),
                "timestamp": _iso_from_any(published),
                "publisher": publisher,
                "source": publisher,
                "source_id": source_id,
                "source_confidence_norm": float(source_confidence_norm),
                "schema_confidence_norm": 0.86,
            }
        )
    return [row for row in rows if row.get("headline")]


def _matches_symbol(row: Mapping[str, Any], symbol: str) -> bool:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return False
    text = f" {_text(row).upper()} "
    if re.search(rf"(?<![A-Z0-9]){re.escape(sym)}(?![A-Z0-9])", text):
        return True
    aliases = list(EQUITY_ALIAS_MAP.get(sym, ())) + list(CRYPTO_ALIASES.get(sym, ()))
    lowered = text.lower()
    return any(alias and alias.lower() in lowered for alias in aliases)


def _collect_rss_source(
    source_id: str,
    spec: Mapping[str, Any],
    *,
    timeout: float,
    user_agent: str,
    max_items: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = fetch_text(
        str(spec.get("url") or ""),
        user_agent=user_agent,
        timeout=timeout,
        accept="application/rss+xml, application/xml, text/xml, */*",
        collector_key="ticker_news_context",
        source_name=source_id,
        entity_key=str(spec.get("url") or ""),
        project_root=PROJECT_ROOT,
        source_confidence_norm=float(spec.get("source_confidence_norm", 0.7) or 0.7),
        schema_confidence_norm=0.86,
    )
    if not bool(result.get("ok", False)):
        return [], {
            "ok": False,
            "url": str(spec.get("url") or ""),
            "publisher": str(spec.get("publisher") or source_id),
            "error": str(result.get("error") or ""),
            "rows": 0,
        }
    rows = _rss_items(
        str(result.get("text") or ""),
        publisher=str(spec.get("publisher") or source_id),
        source_id=source_id,
        source_confidence_norm=float(spec.get("source_confidence_norm", 0.7) or 0.7),
    )[: max(int(max_items), 1)]
    return rows, {
        "ok": True,
        "url": str(spec.get("url") or ""),
        "publisher": str(spec.get("publisher") or source_id),
        "rows": len(rows),
        "status_code": result.get("status_code"),
        "duration_ms": result.get("duration_ms"),
    }


def _yahoo_symbol_feed(symbol: str) -> str:
    query = urlencode({"s": symbol, "region": "US", "lang": "en-US"})
    return f"https://feeds.finance.yahoo.com/rss/2.0/headline?{query}"


def _collect_yahoo_symbol_news(
    symbols: list[str],
    *,
    timeout: float,
    user_agent: str,
    limit_per_symbol: int,
    max_runtime_seconds: float,
    started_monotonic: float,
    sleep_seconds: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
    ok = 0
    attempted = 0
    errors: list[str] = []
    for symbol in symbols:
        if max_runtime_seconds > 0 and (time.monotonic() - started_monotonic) >= max_runtime_seconds:
            break
        if not is_probably_schwab_symbol(symbol):
            continue
        attempted += 1
        spec = {
            "url": _yahoo_symbol_feed(symbol),
            "publisher": "Yahoo Finance",
            "source_confidence_norm": 0.70,
        }
        rows, status = _collect_rss_source(f"yahoo_finance_{symbol}", spec, timeout=timeout, user_agent=user_agent, max_items=limit_per_symbol)
        if status.get("ok"):
            ok += 1
        elif status.get("error"):
            errors.append(f"{symbol}:{status.get('error')}")
        normalized = []
        for row in rows:
            item = dict(row)
            item["symbol"] = symbol
            item["symbols"] = [symbol]
            item["source_method"] = "yahoo_finance_symbol_rss"
            normalized.append(item)
        rows_by_symbol[symbol] = dedupe_items(normalized)[:limit_per_symbol]
        if sleep_seconds > 0:
            time.sleep(max(float(sleep_seconds), 0.0))
    return rows_by_symbol, {
        "ok": ok > 0,
        "attempted_symbols": attempted,
        "ok_symbols": ok,
        "error_count": max(attempted - ok, 0),
        "errors": errors[:10],
    }


def _load_existing_external_items(path: Path, *, source_method: str, publisher_fallback: str) -> list[dict[str, Any]]:
    payload = {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    rows: list[dict[str, Any]] = []
    items_by_symbol = payload.get("items_by_symbol") if isinstance(payload.get("items_by_symbol"), dict) else {}
    for symbol, raw_items in items_by_symbol.items():
        if not isinstance(raw_items, list):
            continue
        for row in raw_items:
            if not isinstance(row, Mapping):
                continue
            item = normalize_news_item(row, symbol=str(symbol), source_method=source_method)
            if not item.get("publisher"):
                item["publisher"] = publisher_fallback
                item["source"] = publisher_fallback
            rows.append(item)
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    for row in items:
        if not isinstance(row, Mapping):
            continue
        row_symbols = row.get("symbols") if isinstance(row.get("symbols"), list) else []
        for symbol in row_symbols:
            item = normalize_news_item(row, symbol=str(symbol), source_method=source_method)
            if not item.get("publisher"):
                item["publisher"] = publisher_fallback
                item["source"] = publisher_fallback
            rows.append(item)
    return dedupe_items(rows)


def build_payload(
    *,
    project_root: Path = PROJECT_ROOT,
    symbols_arg: str = "",
    max_symbols: int = 0,
    limit_per_symbol: int = 12,
    max_runtime_seconds: float = 240.0,
    timeout_seconds: float = 5.0,
    sleep_seconds: float = 0.02,
    include_optional_global_feeds: bool = False,
    include_existing_schwab: bool = True,
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
    user_agent = USER_AGENT_DEFAULT

    items_by_symbol: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in symbols}
    sources: dict[str, Any] = {}

    if include_existing_schwab:
        schwab_items = _load_existing_external_items(
            project_root / "exports" / "external_context" / "schwab_symbol_news_latest.json",
            source_method="schwab_symbol_news_context",
            publisher_fallback="Schwab",
        )
        for item in schwab_items:
            symbol = str(item.get("symbol") or "").strip().upper()
            if symbol in items_by_symbol:
                items_by_symbol[symbol].append(item)
        sources["schwab_symbol_news_context"] = {"ok": bool(schwab_items), "rows": len(schwab_items), "publisher": "Schwab/Schwab Network"}

    global_feed_specs = dict(CRYPTO_NEWS_FEEDS)
    if include_optional_global_feeds:
        global_feed_specs.update(OPTIONAL_GLOBAL_NEWS_FEEDS)
    global_rows: list[dict[str, Any]] = []
    for source_id, spec in global_feed_specs.items():
        if max_runtime_seconds > 0 and (time.monotonic() - started) >= max_runtime_seconds:
            sources[source_id] = {"ok": False, "rows": 0, "error": "runtime_budget_exhausted"}
            continue
        rows, status = _collect_rss_source(
            source_id,
            spec,
            timeout=timeout_seconds,
            user_agent=user_agent,
            max_items=max(limit_per_symbol * 8, 50),
        )
        sources[source_id] = status
        global_rows.extend(rows)

    for row in global_rows:
        for symbol in symbols:
            if _matches_symbol(row, symbol):
                item = dict(row)
                item["symbol"] = symbol
                item["symbols"] = [symbol]
                item["source_method"] = f"{item.get('source_id', 'global')}_rss_symbol_match"
                items_by_symbol[symbol].append(item)

    yahoo_symbols = [symbol for symbol in symbols if is_probably_schwab_symbol(symbol)]
    yahoo_by_symbol, yahoo_status = _collect_yahoo_symbol_news(
        yahoo_symbols,
        timeout=timeout_seconds,
        user_agent=user_agent,
        limit_per_symbol=limit_per_symbol,
        max_runtime_seconds=max_runtime_seconds,
        started_monotonic=started,
        sleep_seconds=sleep_seconds,
    )
    sources["yahoo_finance_symbol_rss"] = yahoo_status
    for symbol, rows in yahoo_by_symbol.items():
        if symbol in items_by_symbol:
            items_by_symbol[symbol].extend(rows)

    symbol_features: dict[str, dict[str, float]] = {}
    symbol_rows: dict[str, dict[str, Any]] = {}
    all_items: list[dict[str, Any]] = []
    for symbol in symbols:
        rows = dedupe_items(items_by_symbol.get(symbol, []))[:limit_per_symbol]
        items_by_symbol[symbol] = rows
        all_items.extend(rows)
        features = _symbol_features(symbol, rows, now_ts=now_ts, max_items=limit_per_symbol)
        symbol_features[symbol] = features
        symbol_rows[symbol] = {
            "status": "ok" if rows else "no_news",
            "ok": bool(rows),
            "groups": symbol_groups.get(symbol, []),
            "item_count": len(rows),
            "source_count": len({str(row.get("source") or row.get("publisher") or "").strip() for row in rows if str(row.get("source") or row.get("publisher") or "").strip()}),
            "feature_summary": features,
            "catalyst_counts": _catalyst_counts(rows),
            "items": rows,
        }
    all_items = dedupe_items(all_items)
    symbols_with_news = sum(1 for row in symbol_rows.values() if int(row.get("item_count", 0) or 0) > 0)
    ok_sources = sum(1 for row in sources.values() if isinstance(row, Mapping) and bool(row.get("ok", False)))
    source_counts = Counter(str(item.get("source") or item.get("publisher") or "unknown") for item in all_items)
    global_features: dict[str, float] = {}
    for features in symbol_features.values():
        for key, raw in features.items():
            try:
                value = float(raw)
            except Exception:
                continue
            if key == "news_sentiment":
                current = float(global_features.get(key, 0.0) or 0.0)
                if abs(value) >= abs(current):
                    global_features[key] = value
            else:
                global_features[key] = max(float(global_features.get(key, 0.0) or 0.0), value)
    coverage_ratio = symbols_with_news / max(len(symbols), 1)
    global_features.update(
        {
            "ticker_news_coverage_norm": max(0.0, min(coverage_ratio, 1.0)),
            "ticker_news_source_health_norm": ok_sources / max(len(sources), 1),
            "ticker_news_total_items_norm": min(len(all_items) / max(len(symbols) * limit_per_symbol, 1), 1.0),
        }
    )
    status = "ready" if symbols_with_news > 0 and ok_sources > 0 else "degraded"
    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "collector": "ticker_news_context",
        "ok": status == "ready",
        "overall_status": status,
        "universe_source": universe_source,
        "universe_policy": universe_policy,
        "requested_symbol_count": len(symbols),
        "symbols_with_news": symbols_with_news,
        "coverage_ratio": round(coverage_ratio, 6),
        "total_news_items": len(all_items),
        "source_count": len(sources),
        "ok_source_count": ok_sources,
        "source_counts": dict(source_counts.most_common(20)),
        "sources": sources,
        "symbols": symbol_rows,
        "items_by_symbol": items_by_symbol,
        "derived": {
            "news_features": global_features,
            "news_symbol_features": symbol_features,
            "symbol_features": {
                symbol: {
                    "external_context_source_available": 1.0 if bool(row.get("ok", False)) else 0.0,
                    "external_context_symbol_coverage_norm": float(symbol_features.get(symbol, {}).get("ticker_news_item_count_norm", 0.0) or 0.0),
                    "external_context_news_source_count_norm": float(symbol_features.get(symbol, {}).get("ticker_news_source_count_norm", 0.0) or 0.0),
                }
                for symbol, row in symbol_rows.items()
            },
            "global_features": {
                "external_context_ticker_news_available": 1.0 if symbols_with_news > 0 else 0.0,
                "external_context_ticker_news_coverage_norm": max(0.0, min(coverage_ratio, 1.0)),
                "external_context_ticker_news_source_health_norm": ok_sources / max(len(sources), 1),
            },
        },
        "safety_contract": {
            "market_data_only": True,
            "live_execution_allowed": False,
            "writes_orders": False,
            "protected_volumes": ["/Volumes/VIDEO"],
            "source_policy": "bounded_rss_and_existing_context_only",
        },
        "recommended_actions": [
            "increase --max-runtime-seconds or reduce --max-symbols if source coverage is partial",
            "enable --include-optional-global-feeds for press-release wire sources when host/network pressure is cool",
        ],
    }
    return payload


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _append_event(payload: Mapping[str, Any]) -> None:
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    event_path = EVENT_DIR / f"ticker_news_context_{day}.jsonl"
    summary = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "overall_status": payload.get("overall_status"),
        "requested_symbol_count": payload.get("requested_symbol_count"),
        "symbols_with_news": payload.get("symbols_with_news"),
        "total_news_items": payload.get("total_news_items"),
        "ok_source_count": payload.get("ok_source_count"),
        "source_count": payload.get("source_count"),
    }
    with event_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=True, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect multi-source ticker news context for the active ticker universe.")
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbol override.")
    parser.add_argument("--max-symbols", type=int, default=int(os.getenv("TICKER_NEWS_MAX_SYMBOLS", "0") or 0))
    parser.add_argument("--limit-per-symbol", type=int, default=int(os.getenv("TICKER_NEWS_LIMIT_PER_SYMBOL", "12") or 12))
    parser.add_argument("--max-runtime-seconds", type=float, default=float(os.getenv("TICKER_NEWS_MAX_RUNTIME_SECONDS", "240") or 240))
    parser.add_argument("--timeout-seconds", type=float, default=float(os.getenv("TICKER_NEWS_TIMEOUT_SECONDS", "5") or 5))
    parser.add_argument("--sleep-seconds", type=float, default=float(os.getenv("TICKER_NEWS_SLEEP_SECONDS", "0.02") or 0.02))
    parser.add_argument("--include-optional-global-feeds", action="store_true", help="Include press-release wire feeds in addition to Yahoo/crypto feeds.")
    parser.add_argument("--skip-existing-schwab", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(
        project_root=PROJECT_ROOT,
        symbols_arg=str(args.symbols or ""),
        max_symbols=int(args.max_symbols),
        limit_per_symbol=int(args.limit_per_symbol),
        max_runtime_seconds=float(args.max_runtime_seconds),
        timeout_seconds=float(args.timeout_seconds),
        sleep_seconds=float(args.sleep_seconds),
        include_optional_global_feeds=bool(args.include_optional_global_feeds),
        include_existing_schwab=not bool(args.skip_existing_schwab),
    )
    _write_payload(HEALTH_PATH, payload)
    _write_payload(EXTERNAL_CONTEXT_PATH, payload)
    _append_event(payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "ticker_news_context "
            f"status={payload.get('overall_status')} "
            f"symbols_with_news={payload.get('symbols_with_news')}/{payload.get('requested_symbol_count')} "
            f"items={payload.get('total_news_items')} "
            f"sources={payload.get('ok_source_count')}/{payload.get('source_count')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
