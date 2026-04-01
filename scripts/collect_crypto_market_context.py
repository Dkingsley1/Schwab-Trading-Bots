#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.market_context_features import default_structured_news_features, summarize_structured_news_items


USER_AGENT_DEFAULT = "Daniel Kingsley dan_kingsley@aol.com"
DERIBIT_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
KRAKEN_URL = "https://api.kraken.com/0/public/Ticker"
HYPERLIQUID_URL = "https://api.hyperliquid.xyz/info"
COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
DEFILLAMA_STABLECOINS_URL = "https://stablecoins.llama.fi/stablecoins?includePrices=true"
DEFILLAMA_DEXS_URL = "https://api.llama.fi/overview/dexs?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true"
ETHERSCAN_GAS_URL = "https://api.etherscan.io/v2/api"
COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"

COINBASE_STATUS_RSS_URL = "https://status.coinbase.com/history.rss"
COINBASE_MARKET_NOTICES_URL = "https://www.coinbase.com/derivatives/market-notices"

NEWS_FEED_SOURCES: dict[str, dict[str, str]] = {
    "coindesk": {
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "publisher": "CoinDesk",
    },
    "cointelegraph": {
        "url": "https://cointelegraph.com/rss",
        "publisher": "Cointelegraph",
    },
    "decrypt": {
        "url": "https://decrypt.co/feed",
        "publisher": "Decrypt",
    },
    "the_block": {
        "url": "https://www.theblock.co/rss.xml",
        "publisher": "The Block",
    },
    "bitcoin_magazine": {
        "url": "https://bitcoinmagazine.com/.rss/full/",
        "publisher": "Bitcoin Magazine",
    },
}

FEATURE_KEYS = [
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
]

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

_DERIBIT_SUPPORTED = {"BTC", "ETH", "SOL"}
_COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "AVAX": "avalanche-2",
    "DOGE": "dogecoin",
    "LINK": "chainlink",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
}
_KRAKEN_PAIRS = {
    "BTC": "XBTUSD",
    "ETH": "ETHUSD",
    "SOL": "SOLUSD",
    "AVAX": "AVAXUSD",
    "DOGE": "DOGEUSD",
    "LINK": "LINKUSD",
    "LTC": "LTCUSD",
    "BCH": "BCHUSD",
}

_ASSET_ALIASES = {
    "BTC": ("bitcoin", "btc", "xbt"),
    "ETH": ("ethereum", "ether", "eth"),
    "SOL": ("solana", "sol"),
    "AVAX": ("avalanche", "avax"),
    "DOGE": ("dogecoin", "doge"),
    "LINK": ("chainlink", "link"),
    "LTC": ("litecoin", "ltc"),
    "BCH": ("bitcoin cash", "bch"),
}

_POSITIVE_NEWS_TOKENS = {
    "approval",
    "approves",
    "approved",
    "rally",
    "surge",
    "surges",
    "breakout",
    "rebound",
    "recovery",
    "launch",
    "launches",
    "listed",
    "listing",
    "inflow",
    "inflows",
    "bullish",
    "gains",
    "strong",
}
_NEGATIVE_NEWS_TOKENS = {
    "delay",
    "delayed",
    "degraded",
    "outage",
    "incident",
    "investigating",
    "disabled",
    "disable",
    "halted",
    "exploit",
    "hack",
    "hacked",
    "breach",
    "depeg",
    "liquidation",
    "liquidations",
    "outflow",
    "outflows",
    "delist",
    "delisted",
    "probe",
    "lawsuit",
    "bearish",
    "drop",
    "selloff",
}
_SHOCK_NEWS_TOKENS = {
    "approval",
    "etf",
    "fork",
    "airdrop",
    "hack",
    "exploit",
    "outage",
    "incident",
    "depeg",
    "delayed",
    "disabled",
    "halted",
    "maintenance",
    "listing",
    "delist",
    "lawsuit",
    "probe",
    "investigating",
}
_BROAD_MARKET_TOKENS = {
    "bitcoin",
    "crypto",
    "market",
    "markets",
    "exchange",
    "trading",
    "etf",
    "stablecoin",
    "defi",
    "derivatives",
    "futures",
}
_CRYPTO_TOPIC_TOKENS = _BROAD_MARKET_TOKENS | {
    "blockchain",
    "web3",
    "token",
    "tokens",
    "coinbase",
    "binance",
    "ethereum",
    "solana",
    "litecoin",
    "dogecoin",
    "chainlink",
    "stablecoin",
    "wallet",
    "mining",
    "validator",
    "defi",
    "nft",
    "layer 2",
}

_HTML_TAG_RE = re.compile(r"(?is)<[^>]+>")
_HTML_LINK_RE = re.compile(r"""(?is)<a\b[^>]*href=["']([^"']+)["'][^>]*>(.*?)</a>""")
_CLOUDFLARE_TOKENS = (
    "Just a moment...",
    "__cf_chl_opt",
    "Enable JavaScript and cookies to continue",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


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


def _clamp11(value: float) -> float:
    return max(-1.0, min(float(value), 1.0))


def _signed_centered_norm(value: float, scale: float) -> float:
    return _clamp01(0.5 + (float(value) / max(float(scale), 1e-8)))


def _safe_log_norm(value: float, *, denom: float) -> float:
    if value <= 0.0:
        return 0.0
    return _clamp01(math.log1p(float(value)) / max(float(denom), 1e-8))


def _normalize_symbol(raw: str) -> str:
    return str(raw or "").strip().upper().replace("/", "-").replace("_", "-")


def _asset_from_symbol(raw: str) -> str:
    token = _normalize_symbol(raw)
    if not token:
        return ""
    if "-" in token:
        return token.split("-", 1)[0]
    return token


def _parse_symbols(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in str(raw or "").replace("\n", ",").split(","):
        symbol = _normalize_symbol(token)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _default_symbols() -> list[str]:
    raw = os.getenv("CRYPTO_MARKET_CONTEXT_SYMBOLS", "").strip()
    if not raw:
        raw = ",".join(
            filter(
                None,
                [
                    os.getenv("COINBASE_WATCH_SYMBOLS", ""),
                    os.getenv("COINBASE_FUTURES_WATCH_SYMBOLS", ""),
                ],
            )
        )
    symbols = _parse_symbols(raw)
    if not symbols:
        symbols = _parse_symbols("BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LTC-USD,LINK-USD,DOGE-USD")
    return symbols[:20]


def _http_json(
    url: str,
    *,
    user_agent: str,
    timeout: float,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> Any:
    req_headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }
    if headers:
        req_headers.update(headers)
    req = Request(url=url, method=method.upper(), headers=req_headers, data=body)
    with urlopen(req, timeout=max(float(timeout), 1.0)) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _http_text(
    url: str,
    *,
    user_agent: str,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> str:
    req_headers = {
        "User-Agent": user_agent,
        "Accept": "*/*",
    }
    if headers:
        req_headers.update(headers)
    req = Request(url=url, method="GET", headers=req_headers)
    with urlopen(req, timeout=max(float(timeout), 1.0)) as resp:
        return resp.read().decode("utf-8", "replace")


def _safe_http_json(**kwargs: Any) -> tuple[Any | None, str | None]:
    try:
        return _http_json(**kwargs), None
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        return None, str(exc)


def _safe_http_text(**kwargs: Any) -> tuple[str | None, str | None]:
    try:
        return _http_text(**kwargs), None
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        return None, str(exc)


def _zero_feature_map() -> dict[str, float]:
    return {key: 0.0 for key in FEATURE_KEYS}


def _default_news_feature_map() -> dict[str, float]:
    out = {key: 0.0 for key in _BASE_NEWS_FEATURE_KEYS}
    out.update(default_structured_news_features())
    return out


def _derive_deribit_asset_features(futures_payload: Any, options_payload: Any) -> tuple[dict[str, float], float]:
    futures_rows = futures_payload.get("result") if isinstance(futures_payload, dict) else None
    options_rows = options_payload.get("result") if isinstance(options_payload, dict) else None
    futures_rows = futures_rows if isinstance(futures_rows, list) else []
    options_rows = options_rows if isinstance(options_rows, list) else []

    futures_oi = sum(max(_to_float(row.get("open_interest"), 0.0), 0.0) for row in futures_rows if isinstance(row, dict))
    options_oi = sum(max(_to_float(row.get("open_interest"), 0.0), 0.0) for row in options_rows if isinstance(row, dict))

    iv_num = 0.0
    iv_den = 0.0
    for row in options_rows:
        if not isinstance(row, dict):
            continue
        oi = max(_to_float(row.get("open_interest"), 0.0), 0.0)
        iv = max(_to_float(row.get("mark_iv"), 0.0), 0.0)
        if oi > 0.0 and iv > 0.0:
            iv_num += oi * iv
            iv_den += oi
    weighted_iv = (iv_num / iv_den) if iv_den > 0.0 else 0.0

    basis_rows = []
    provider_price = 0.0
    for row in futures_rows:
        if not isinstance(row, dict):
            continue
        mark = _to_float(row.get("mark_price"), 0.0)
        underlying = max(_to_float(row.get("estimated_delivery_price"), 0.0), _to_float(row.get("last"), 0.0), 0.0)
        if provider_price <= 0.0:
            provider_price = max(_to_float(row.get("last"), 0.0), mark, _to_float(row.get("mid_price"), 0.0), 0.0)
        if mark > 0.0 and underlying > 0.0:
            basis_rows.append((mark - underlying) / underlying)

    basis = (sum(basis_rows) / max(len(basis_rows), 1)) if basis_rows else 0.0
    return (
        {
            "crypto_deribit_futures_oi_norm": _safe_log_norm(futures_oi, denom=20.0),
            "crypto_deribit_options_oi_norm": _safe_log_norm(options_oi, denom=20.0),
            "crypto_deribit_mark_iv_norm": _clamp01(weighted_iv / 120.0),
            "crypto_deribit_basis_norm": _signed_centered_norm(basis, 0.05),
        },
        provider_price,
    )


def _derive_kraken_asset_features(row: Mapping[str, Any]) -> tuple[dict[str, float], float]:
    close = _to_float((row.get("c") or [0.0])[0] if isinstance(row.get("c"), list) else 0.0, 0.0)
    volume = _to_float((row.get("v") or [0.0, 0.0])[1] if isinstance(row.get("v"), list) else 0.0, 0.0)
    low = _to_float((row.get("l") or [0.0, 0.0])[1] if isinstance(row.get("l"), list) else 0.0, 0.0)
    high = _to_float((row.get("h") or [0.0, 0.0])[1] if isinstance(row.get("h"), list) else 0.0, 0.0)
    range_ratio = ((high - low) / close) if close > 0.0 and high >= low else 0.0
    return (
        {
            "crypto_kraken_volume_norm": _safe_log_norm(volume, denom=16.0),
            "crypto_kraken_range_norm": _clamp01(range_ratio / 0.20),
        },
        close,
    )


def _derive_hyperliquid_asset_features(ctx: Mapping[str, Any]) -> tuple[dict[str, float], float]:
    funding = _to_float(ctx.get("funding"), 0.0)
    open_interest = max(_to_float(ctx.get("openInterest"), 0.0), 0.0)
    oracle_px = _to_float(ctx.get("oraclePx"), 0.0)
    mark_px = max(_to_float(ctx.get("markPx"), 0.0), _to_float(ctx.get("midPx"), 0.0), 0.0)
    basis = ((mark_px - oracle_px) / oracle_px) if mark_px > 0.0 and oracle_px > 0.0 else 0.0
    return (
        {
            "crypto_hyperliquid_funding_norm": _signed_centered_norm(funding, 0.0010),
            "crypto_hyperliquid_open_interest_norm": _safe_log_norm(open_interest, denom=14.0),
            "crypto_hyperliquid_basis_norm": _signed_centered_norm(basis, 0.03),
        },
        max(mark_px, oracle_px, 0.0),
    )


def _derive_coinmetrics_asset_features(row: Mapping[str, Any]) -> tuple[dict[str, float], float]:
    price = _to_float(row.get("PriceUSD"), 0.0)
    tx = max(_to_float(row.get("TxCnt"), 0.0), 0.0)
    active = max(_to_float(row.get("AdrActCnt"), 0.0), 0.0)
    return (
        {
            "crypto_coinmetrics_tx_count_norm": _safe_log_norm(tx, denom=15.0),
            "crypto_coinmetrics_active_addr_norm": _safe_log_norm(active, denom=16.0),
        },
        price,
    )


def _derive_coingecko_asset_features(row: Mapping[str, Any]) -> tuple[dict[str, float], float]:
    price = _to_float(row.get("current_price"), 0.0)
    volume = max(_to_float(row.get("total_volume"), 0.0), 0.0)
    chg_24h = _to_float(row.get("price_change_percentage_24h_in_currency"), _to_float(row.get("price_change_percentage_24h"), 0.0))
    chg_7d = _to_float(row.get("price_change_percentage_7d_in_currency"), 0.0)
    momentum = (chg_24h + (0.5 * chg_7d)) / 100.0
    return (
        {
            "crypto_coingecko_volume_norm": _safe_log_norm(volume, denom=27.0),
            "crypto_coingecko_momentum_norm": _signed_centered_norm(momentum, 0.15),
        },
        price,
    )


def _price_agreement_norm(prices: list[float], *, max_relative_spread: float) -> float:
    clean = [float(price) for price in prices if float(price) > 0.0]
    if len(clean) < 2:
        return 0.0
    spread = (max(clean) - min(clean)) / max(min(clean), 1e-8)
    return _clamp01(1.0 - (spread / max(float(max_relative_spread), 1e-6)))


def _clean_html_text(raw: str) -> str:
    text = html.unescape(str(raw or ""))
    text = _HTML_TAG_RE.sub(" ", text)
    return " ".join(text.split()).strip()


def _parse_feed_timestamp(raw: Any) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = parsedate_to_datetime(text)
    except Exception:
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _news_ts_seconds(raw: Any) -> float | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp()


def _text_has_alias(text: str, alias: str) -> bool:
    return re.search(r"\b" + re.escape(alias.lower()) + r"\b", text.lower()) is not None


def _related_symbols_from_text(text: str, asset_to_symbols: Mapping[str, list[str]]) -> list[str]:
    lowered = text.lower()
    out: list[str] = []
    seen: set[str] = set()
    for asset, aliases in _ASSET_ALIASES.items():
        if asset not in asset_to_symbols:
            continue
        if not any(_text_has_alias(lowered, alias) for alias in aliases):
            continue
        for token in [asset, *asset_to_symbols.get(asset, [])]:
            norm = _normalize_symbol(token)
            if norm and norm not in seen:
                seen.add(norm)
                out.append(norm)
    return out


def _headline_sentiment_and_shock(text: str) -> tuple[float, float]:
    lowered = text.lower()
    pos_hits = sum(1 for token in _POSITIVE_NEWS_TOKENS if token in lowered)
    neg_hits = sum(1 for token in _NEGATIVE_NEWS_TOKENS if token in lowered)
    shock_hits = sum(1 for token in _SHOCK_NEWS_TOKENS if token in lowered)
    sentiment = 0.0
    if pos_hits or neg_hits:
        sentiment = (pos_hits - neg_hits) / max(pos_hits + neg_hits, 1)
    shock = 0.0
    if shock_hits or neg_hits:
        shock = _clamp01((shock_hits * 0.28) + (abs(sentiment) * 0.35) + (0.18 if neg_hits else 0.0))
    return _clamp11(sentiment), shock


def _build_news_row(
    *,
    headline: str,
    summary: str,
    publisher: str,
    source_name: str,
    published: str | None,
    url: str,
    asset_to_symbols: Mapping[str, list[str]],
) -> dict[str, Any] | None:
    clean_headline = " ".join(str(headline or "").split()).strip()
    clean_summary = _clean_html_text(summary)
    if not clean_headline:
        return None
    text = " ".join(part for part in (clean_headline, clean_summary) if part).strip()
    related = _related_symbols_from_text(text, asset_to_symbols)
    lowered = text.lower()
    if not related and not any(token in lowered for token in _CRYPTO_TOPIC_TOKENS):
        return None
    sentiment_hint, shock_hint = _headline_sentiment_and_shock(text)
    broad_market = (not related) or any(token in lowered for token in _BROAD_MARKET_TOKENS)
    row: dict[str, Any] = {
        "headline": clean_headline,
        "summary": clean_summary,
        "source": publisher,
        "publisher": publisher,
        "channel": source_name,
        "published": published,
        "url": url,
        "broad_market": broad_market,
        "macro_event": False,
        "source_quality_norm": 0.8,
    }
    if related:
        row["symbols"] = related
        row["relatedSymbols"] = related
    if abs(sentiment_hint) > 0.0:
        row["sentiment_hint"] = sentiment_hint
    if shock_hint > 0.0:
        row["shock_hint"] = shock_hint
    return row


def _parse_rss_news_items(
    xml_text: str,
    *,
    source_name: str,
    publisher: str,
    asset_to_symbols: Mapping[str, list[str]],
    max_items: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return rows
    items = root.findall(".//item")
    if not items:
        items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
    for item in items[: max(max_items, 1)]:
        title = (
            item.findtext("title")
            or item.findtext("{http://www.w3.org/2005/Atom}title")
            or ""
        ).strip()
        summary = (
            item.findtext("description")
            or item.findtext("summary")
            or item.findtext("{http://www.w3.org/2005/Atom}summary")
            or item.findtext("{http://purl.org/rss/1.0/modules/content/}encoded")
            or ""
        ).strip()
        link = item.findtext("link") or ""
        if not link:
            atom_link = item.find("{http://www.w3.org/2005/Atom}link")
            if atom_link is not None:
                link = atom_link.attrib.get("href", "")
        published = _parse_feed_timestamp(
            item.findtext("pubDate")
            or item.findtext("updated")
            or item.findtext("{http://www.w3.org/2005/Atom}updated")
            or item.findtext("{http://www.w3.org/2005/Atom}published")
            or ""
        )
        row = _build_news_row(
            headline=title,
            summary=summary,
            publisher=publisher,
            source_name=source_name,
            published=published,
            url=str(link or "").strip(),
            asset_to_symbols=asset_to_symbols,
        )
        if row is not None:
            rows.append(row)
    return rows


def _collect_feed_news_source(
    *,
    source_name: str,
    feed_url: str,
    publisher: str,
    asset_to_symbols: Mapping[str, list[str]],
    user_agent: str,
    timeout: float,
    max_items: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    xml_text, err = _safe_http_text(url=feed_url, user_agent=user_agent, timeout=timeout)
    if not xml_text:
        return [], {"ok": False, "rows": 0, "url": feed_url, "error": err}
    rows = _parse_rss_news_items(
        xml_text,
        source_name=source_name,
        publisher=publisher,
        asset_to_symbols=asset_to_symbols,
        max_items=max_items,
    )
    return rows, {
        "ok": bool(rows),
        "rows": len(rows),
        "url": feed_url,
        "error": err,
    }


def _collect_coinbase_status_news(
    *,
    asset_to_symbols: Mapping[str, list[str]],
    user_agent: str,
    timeout: float,
    max_items: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, status = _collect_feed_news_source(
        source_name="coinbase_status",
        feed_url=COINBASE_STATUS_RSS_URL,
        publisher="Coinbase Status",
        asset_to_symbols=asset_to_symbols,
        user_agent=user_agent,
        timeout=timeout,
        max_items=max_items,
    )
    for row in rows:
        headline_text = " ".join(
            part for part in (str(row.get("headline") or ""), str(row.get("summary") or "")) if part
        )
        _, shock_hint = _headline_sentiment_and_shock(headline_text)
        if shock_hint > 0.0:
            row["shock_hint"] = max(_to_float(row.get("shock_hint"), 0.0), max(shock_hint, 0.55))
        row["broad_market"] = bool(row.get("broad_market", True))
    return rows, status


def _collect_coinbase_market_notices(
    *,
    asset_to_symbols: Mapping[str, list[str]],
    user_agent: str,
    timeout: float,
    max_items: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def _walk_nodes(node: Any) -> Iterable[Mapping[str, Any]]:
        if isinstance(node, Mapping):
            yield node
            for value in node.values():
                yield from _walk_nodes(value)
        elif isinstance(node, list):
            for value in node:
                yield from _walk_nodes(value)

    def _rich_text_plain(node: Any) -> str:
        parts: list[str] = []

        def _walk_text(cur: Any) -> None:
            if isinstance(cur, Mapping):
                if cur.get("nodeType") == "text":
                    value = str(cur.get("value") or "").strip()
                    if value:
                        parts.append(value)
                for value in cur.values():
                    _walk_text(value)
            elif isinstance(cur, list):
                for value in cur:
                    _walk_text(value)

        _walk_text(node)
        return " ".join(parts).strip()

    def _rich_text_link(node: Any) -> str:
        found = ""

        def _walk_link(cur: Any) -> None:
            nonlocal found
            if found:
                return
            if isinstance(cur, Mapping):
                if cur.get("nodeType") == "hyperlink":
                    uri = str(((cur.get("data") or {}).get("uri")) or "").strip()
                    if uri:
                        found = uri
                        return
                for value in cur.values():
                    _walk_link(value)
            elif isinstance(cur, list):
                for value in cur:
                    _walk_link(value)

        _walk_link(node)
        return found

    def _decode_suspense_payload(page_text: str) -> Any | None:
        marker = 'suspenseBridgeData":"'
        start = page_text.find(marker)
        if start < 0:
            return None
        start += len(marker)
        end = page_text.find('","disabledServicesFromQueryParams"', start)
        if end < 0:
            end = page_text.find('","clientConfig"', start)
        if end < 0:
            return None
        raw = page_text[start:end]
        try:
            decoded = json.loads('"' + raw + '"')
            return json.loads(decoded)
        except Exception:
            return None

    page_text, err = _safe_http_text(url=COINBASE_MARKET_NOTICES_URL, user_agent=user_agent, timeout=timeout)
    if not page_text:
        return [], {"ok": False, "rows": 0, "url": COINBASE_MARKET_NOTICES_URL, "error": err}
    if any(token in page_text for token in _CLOUDFLARE_TOKENS):
        return [], {
            "ok": False,
            "rows": 0,
            "url": COINBASE_MARKET_NOTICES_URL,
            "error": "cloudflare_challenge",
            "blocked_by_cloudflare": True,
        }

    rows: list[dict[str, Any]] = []
    suspense_payload = _decode_suspense_payload(page_text)
    if suspense_payload is not None:
        for node in _walk_nodes(suspense_payload):
            title = str(node.get("title") or "").strip().lower()
            table = node.get("table")
            if title != "notices" or not isinstance(table, Mapping):
                continue
            table_doc = table.get("table") if isinstance(table.get("table"), Mapping) else table
            content_rows = table_doc.get("content") if isinstance(table_doc, Mapping) else None
            content_rows = content_rows if isinstance(content_rows, list) else []
            for maybe_table in content_rows:
                table_rows = maybe_table.get("content") if isinstance(maybe_table, Mapping) else None
                table_rows = table_rows if isinstance(table_rows, list) else []
                for row_node in table_rows:
                    if not isinstance(row_node, Mapping) or row_node.get("nodeType") != "table-row":
                        continue
                    cells = row_node.get("content") if isinstance(row_node.get("content"), list) else []
                    values = [_rich_text_plain(cell) for cell in cells]
                    links = [_rich_text_link(cell) for cell in cells]
                    if len(values) < 5 or str(values[0]).strip().upper() == "ID":
                        continue
                    try:
                        published = datetime.strptime(str(values[2]).strip(), "%m/%d/%Y").replace(tzinfo=timezone.utc).isoformat()
                    except Exception:
                        published = None
                    headline = str(values[3] or values[0] or "").strip()
                    summary = " | ".join(part for part in (str(values[1]).strip(), str(values[4]).strip(), str(values[0]).strip()) if part)
                    url = str(links[3] or COINBASE_MARKET_NOTICES_URL).strip()
                    if published is None or not headline:
                        continue
                    text = " ".join(part for part in (headline, summary) if part).strip()
                    related = _related_symbols_from_text(text, asset_to_symbols)
                    sentiment_hint, shock_hint = _headline_sentiment_and_shock(text)
                    row: dict[str, Any] = {
                        "headline": headline,
                        "summary": summary,
                        "source": "Coinbase Market Notices",
                        "publisher": "Coinbase Market Notices",
                        "channel": "coinbase_market_notices",
                        "published": published,
                        "url": url,
                        "broad_market": not bool(related),
                        "macro_event": False,
                        "source_quality_norm": 0.82,
                    }
                    if related:
                        row["symbols"] = related
                        row["relatedSymbols"] = related
                    if abs(sentiment_hint) > 0.0:
                        row["sentiment_hint"] = sentiment_hint
                    if shock_hint > 0.0:
                        row["shock_hint"] = shock_hint
                    rows.append(row)
            break
    rows = _dedupe_news_rows(rows, max_items=max_items)
    return rows, {
        "ok": bool(rows),
        "rows": len(rows),
        "url": COINBASE_MARKET_NOTICES_URL,
        "error": err,
    }


def _dedupe_news_rows(rows: Iterable[Mapping[str, Any]], *, max_items: int) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        headline = " ".join(str(raw.get("headline") or "").split()).strip()
        published = str(raw.get("published") or "").strip()
        url = str(raw.get("url") or "").strip()
        if not headline:
            continue
        key = (headline.lower(), published, url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(raw))
    deduped.sort(key=lambda row: _news_ts_seconds(row.get("published")) or 0.0, reverse=True)
    return deduped[: max(max_items, 1)]


def _summarize_crypto_news_items(
    items: Iterable[Mapping[str, Any]],
    *,
    symbol: str,
    now_ts: float,
    max_items: int,
) -> dict[str, float]:
    out = _default_news_feature_map()
    rows: list[tuple[float, Mapping[str, Any]]] = []
    for raw in items:
        if not isinstance(raw, Mapping):
            continue
        ts = _news_ts_seconds(raw.get("published") or raw.get("publishedDate") or raw.get("timestamp"))
        if ts is None or ts > now_ts:
            continue
        age = now_ts - ts
        if age > 48.0 * 3600.0:
            continue
        headline = str(raw.get("headline") or raw.get("title") or "").strip()
        if not headline:
            continue
        rows.append((age, raw))

    if not rows:
        return out

    rows.sort(key=lambda item: item[0])
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
        if age <= 30.0 * 60.0:
            c30 += 1
        if age <= 2.0 * 60.0 * 60.0:
            c2h += 1
        if age <= 24.0 * 60.0 * 60.0:
            c24h += 1

        headline = " ".join(
            part for part in (str(row.get("headline") or ""), str(row.get("summary") or "")) if part
        )
        sentiment_hint = _to_float(row.get("sentiment_hint"), float("nan"))
        shock_hint = _to_float(row.get("shock_hint"), float("nan"))
        if not math.isfinite(sentiment_hint) or not math.isfinite(shock_hint):
            inferred_sentiment, inferred_shock = _headline_sentiment_and_shock(headline)
            if not math.isfinite(sentiment_hint):
                sentiment_hint = inferred_sentiment
            if not math.isfinite(shock_hint):
                shock_hint = inferred_shock
        if sentiment_hint > 0.0:
            pos_n += 1
        elif sentiment_hint < 0.0:
            neg_n += 1
        if shock_hint > 0.0:
            shock_n += 1

        weight = math.exp(-age / 3600.0)
        weight_sum += weight
        sent_sum += weight * sentiment_hint
        impact_sum += weight * max(abs(sentiment_hint), shock_hint)

    denom = float(max(max_items, 1))
    n = len(rows)
    out.update(
        {
            "news_available": 1.0,
            "news_items_30m": min(c30 / denom, 1.0),
            "news_items_2h": min(c2h / denom, 1.0),
            "news_items_24h": min(c24h / denom, 1.0),
            "news_sentiment": _clamp11((sent_sum / weight_sum) if weight_sum > 0.0 else 0.0),
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


def _collect_crypto_news(
    *,
    asset_to_symbols: Mapping[str, list[str]],
    user_agent: str,
    timeout: float,
    max_items_per_source: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    statuses: dict[str, dict[str, Any]] = {}

    coinbase_status_rows, coinbase_status_status = _collect_coinbase_status_news(
        asset_to_symbols=asset_to_symbols,
        user_agent=user_agent,
        timeout=timeout,
        max_items=max_items_per_source,
    )
    rows.extend(coinbase_status_rows)
    statuses["coinbase_status"] = coinbase_status_status

    market_notice_rows, market_notice_status = _collect_coinbase_market_notices(
        asset_to_symbols=asset_to_symbols,
        user_agent=user_agent,
        timeout=timeout,
        max_items=max_items_per_source,
    )
    rows.extend(market_notice_rows)
    statuses["coinbase_market_notices"] = market_notice_status

    for source_name, cfg in NEWS_FEED_SOURCES.items():
        source_rows, source_status = _collect_feed_news_source(
            source_name=source_name,
            feed_url=cfg["url"],
            publisher=cfg["publisher"],
            asset_to_symbols=asset_to_symbols,
            user_agent=user_agent,
            timeout=timeout,
            max_items=max_items_per_source,
        )
        rows.extend(source_rows)
        statuses[source_name] = source_status

    return _dedupe_news_rows(rows, max_items=max_items_per_source * max(len(statuses), 1)), statuses


def _collect_deribit(assets: list[str], *, user_agent: str, timeout: float) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, Any]]:
    symbol_features: dict[str, dict[str, float]] = {}
    provider_prices: dict[str, float] = {}
    status: dict[str, Any] = {"ok": False, "assets": [], "errors": []}
    for asset in assets:
        if asset not in _DERIBIT_SUPPORTED:
            continue
        futures_payload, futures_err = _safe_http_json(
            url=f"{DERIBIT_URL}?currency={asset}&kind=future",
            user_agent=user_agent,
            timeout=timeout,
        )
        options_payload, options_err = _safe_http_json(
            url=f"{DERIBIT_URL}?currency={asset}&kind=option",
            user_agent=user_agent,
            timeout=timeout,
        )
        if futures_err and options_err:
            status["errors"].append(f"{asset}:{futures_err}")
            continue
        features, price = _derive_deribit_asset_features(futures_payload, options_payload)
        if any(value > 0.0 for value in features.values()):
            symbol_features[asset] = features
            status["assets"].append(asset)
        if price > 0.0:
            provider_prices[asset] = price
    status["ok"] = bool(symbol_features)
    status["asset_count"] = len(status["assets"])
    status["errors"] = status["errors"][:10]
    return symbol_features, provider_prices, status


def _collect_kraken(assets: list[str], *, user_agent: str, timeout: float) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, Any]]:
    pair_to_asset = {pair: asset for asset, pair in _KRAKEN_PAIRS.items() if asset in assets}
    if not pair_to_asset:
        return {}, {}, {"ok": False, "pairs": 0, "errors": ["no_supported_assets"]}
    payload, err = _safe_http_json(
        url=f"{KRAKEN_URL}?pair={','.join(pair_to_asset.keys())}",
        user_agent=user_agent,
        timeout=timeout,
    )
    result = payload.get("result") if isinstance(payload, dict) else {}
    symbol_features: dict[str, dict[str, float]] = {}
    provider_prices: dict[str, float] = {}
    for pair, asset in pair_to_asset.items():
        row = result.get(pair) if isinstance(result, dict) else None
        if not isinstance(row, Mapping):
            continue
        features, price = _derive_kraken_asset_features(row)
        if any(value > 0.0 for value in features.values()):
            symbol_features[asset] = features
        if price > 0.0:
            provider_prices[asset] = price
    return symbol_features, provider_prices, {
        "ok": bool(symbol_features),
        "pairs": len(pair_to_asset),
        "resolved_assets": len(symbol_features),
        "error": err,
    }


def _collect_hyperliquid(assets: list[str], *, user_agent: str, timeout: float) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, Any]]:
    payload, err = _safe_http_json(
        url=HYPERLIQUID_URL,
        method="POST",
        headers={"content-type": "application/json"},
        body=json.dumps({"type": "metaAndAssetCtxs"}).encode("utf-8"),
        user_agent=user_agent,
        timeout=timeout,
    )
    if not isinstance(payload, list) or len(payload) < 2:
        return {}, {}, {"ok": False, "error": err or "invalid_payload"}
    meta = payload[0] if isinstance(payload[0], Mapping) else {}
    ctxs = payload[1] if isinstance(payload[1], list) else []
    universe = meta.get("universe") if isinstance(meta, Mapping) else None
    universe = universe if isinstance(universe, list) else []
    symbol_features: dict[str, dict[str, float]] = {}
    provider_prices: dict[str, float] = {}
    matched = 0
    for idx, item in enumerate(universe):
        if not isinstance(item, Mapping):
            continue
        asset = str(item.get("name") or "").strip().upper()
        if asset not in assets:
            continue
        ctx = ctxs[idx] if idx < len(ctxs) and isinstance(ctxs[idx], Mapping) else {}
        features, price = _derive_hyperliquid_asset_features(ctx)
        if any(value > 0.0 for value in features.values()):
            symbol_features[asset] = features
            matched += 1
        if price > 0.0:
            provider_prices[asset] = price
    return symbol_features, provider_prices, {"ok": bool(symbol_features), "matched_assets": matched, "error": err}


def _collect_coinmetrics(
    assets: list[str],
    *,
    user_agent: str,
    timeout: float,
    api_key: str,
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, Any]]:
    symbol_features: dict[str, dict[str, float]] = {}
    provider_prices: dict[str, float] = {}
    errors: list[str] = []
    headers = {"api_key": api_key} if api_key else None
    for asset in assets:
        url = f"{COINMETRICS_URL}?assets={asset.lower()}&metrics=PriceUSD,TxCnt,AdrActCnt&limit_per_asset=1"
        payload, err = _safe_http_json(url=url, user_agent=user_agent, timeout=timeout, headers=headers)
        rows = payload.get("data") if isinstance(payload, dict) else None
        rows = rows if isinstance(rows, list) else []
        if err:
            errors.append(f"{asset}:{err}")
            continue
        row = rows[0] if rows and isinstance(rows[0], Mapping) else None
        if not isinstance(row, Mapping):
            continue
        features, price = _derive_coinmetrics_asset_features(row)
        if any(value > 0.0 for value in features.values()):
            symbol_features[asset] = features
        if price > 0.0:
            provider_prices[asset] = price
    resolved_assets = len(symbol_features)
    return symbol_features, provider_prices, {
        "ok": bool(symbol_features),
        "resolved_assets": resolved_assets,
        "error": errors[0] if errors and resolved_assets == 0 else None,
        "errors": errors[:10],
        "partial_error_count": len(errors) if resolved_assets > 0 else 0,
    }


def _collect_defillama(*, user_agent: str, timeout: float) -> tuple[dict[str, float], dict[str, Any]]:
    stablecoins_payload, stablecoins_err = _safe_http_json(url=DEFILLAMA_STABLECOINS_URL, user_agent=user_agent, timeout=timeout)
    dexs_payload, dexs_err = _safe_http_json(url=DEFILLAMA_DEXS_URL, user_agent=user_agent, timeout=timeout)
    pegged_assets = stablecoins_payload.get("peggedAssets") if isinstance(stablecoins_payload, dict) else None
    pegged_assets = pegged_assets if isinstance(pegged_assets, list) else []
    current_total = 0.0
    prev_week_total = 0.0
    for row in pegged_assets:
        if not isinstance(row, Mapping):
            continue
        current_total += _to_float(((row.get("circulating") or {}).get("peggedUSD")), 0.0)
        prev_week_total += _to_float(((row.get("circulatingPrevWeek") or {}).get("peggedUSD")), 0.0)
    stablecoin_growth = ((current_total - prev_week_total) / prev_week_total) if prev_week_total > 0.0 else 0.0
    total24h = _to_float(dexs_payload.get("total24h"), 0.0) if isinstance(dexs_payload, Mapping) else 0.0
    total48 = _to_float(dexs_payload.get("total48hto24h"), 0.0) if isinstance(dexs_payload, Mapping) else 0.0
    dex_growth = ((total24h - total48) / total48) if total48 > 0.0 else 0.0
    features = {
        "crypto_defillama_stablecoin_growth_norm": _signed_centered_norm(stablecoin_growth, 0.20),
        "crypto_defillama_dex_volume_growth_norm": _signed_centered_norm(dex_growth, 0.30),
    }
    return features, {
        "ok": any(value > 0.0 for value in features.values()),
        "stablecoins_error": stablecoins_err,
        "dexs_error": dexs_err,
        "stablecoin_assets": len(pegged_assets),
        "total24h": total24h,
        "total48hto24h": total48,
    }


def _collect_etherscan(*, user_agent: str, timeout: float, api_key: str) -> tuple[dict[str, float], dict[str, Any]]:
    params = {"chainid": "1", "module": "gastracker", "action": "gasoracle"}
    if api_key:
        params["apikey"] = api_key
    payload, err = _safe_http_json(url=f"{ETHERSCAN_GAS_URL}?{urlencode(params)}", user_agent=user_agent, timeout=timeout)
    result = payload.get("result") if isinstance(payload, dict) else {}
    base_fee = _to_float(result.get("suggestBaseFee"), 0.0) if isinstance(result, Mapping) else 0.0
    fast_fee = _to_float(result.get("FastGasPrice"), 0.0) if isinstance(result, Mapping) else 0.0
    features = {"crypto_etherscan_gas_norm": _clamp01(max(base_fee, fast_fee, 0.0) / 100.0)}
    return features, {
        "ok": bool(payload.get("result")) if isinstance(payload, Mapping) else False,
        "error": err,
        "message": payload.get("message") if isinstance(payload, Mapping) else None,
        "keyless_rate_limited": isinstance(payload, Mapping) and "Missing/Invalid API Key" in str(payload.get("message") or ""),
    }


def _collect_coingecko(assets: list[str], *, user_agent: str, timeout: float) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, Any]]:
    id_to_asset = {coin_id: asset for asset, coin_id in _COINGECKO_IDS.items() if asset in assets}
    if not id_to_asset:
        return {}, {}, {"ok": False, "error": "no_supported_assets"}
    params = urlencode({"vs_currency": "usd", "ids": ",".join(id_to_asset.keys()), "price_change_percentage": "24h,7d"})
    payload, err = _safe_http_json(url=f"{COINGECKO_MARKETS_URL}?{params}", user_agent=user_agent, timeout=timeout)
    rows = payload if isinstance(payload, list) else []
    symbol_features: dict[str, dict[str, float]] = {}
    provider_prices: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        asset = id_to_asset.get(str(row.get("id") or "").strip().lower())
        if not asset:
            continue
        features, price = _derive_coingecko_asset_features(row)
        if any(value > 0.0 for value in features.values()):
            symbol_features[asset] = features
        if price > 0.0:
            provider_prices[asset] = price
    return symbol_features, provider_prices, {"ok": bool(symbol_features), "resolved_assets": len(symbol_features), "error": err}


def collect_crypto_market_context(
    *,
    symbols: list[str],
    user_agent: str,
    timeout: float = 12.0,
    max_relative_spread: float = 0.05,
    coinmetrics_api_key: str = "",
    etherscan_api_key: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()
    tracked_symbols = [_normalize_symbol(symbol) for symbol in symbols if _normalize_symbol(symbol)]
    asset_order: list[str] = []
    asset_to_symbols: dict[str, list[str]] = {}
    for symbol in tracked_symbols:
        asset = _asset_from_symbol(symbol)
        if not asset:
            continue
        asset_to_symbols.setdefault(asset, []).append(symbol)
        if asset not in asset_order:
            asset_order.append(asset)

    symbol_features: dict[str, dict[str, float]] = {symbol: _zero_feature_map() for symbol in tracked_symbols}
    asset_feature_accum: dict[str, dict[str, float]] = {}
    asset_provider_prices: dict[str, dict[str, float]] = {}

    deribit_rows, deribit_prices, deribit_status = _collect_deribit(asset_order, user_agent=user_agent, timeout=timeout)
    kraken_rows, kraken_prices, kraken_status = _collect_kraken(asset_order, user_agent=user_agent, timeout=timeout)
    hyper_rows, hyper_prices, hyper_status = _collect_hyperliquid(asset_order, user_agent=user_agent, timeout=timeout)
    coinmetrics_rows, coinmetrics_prices, coinmetrics_status = _collect_coinmetrics(
        asset_order,
        user_agent=user_agent,
        timeout=timeout,
        api_key=coinmetrics_api_key,
    )
    defillama_global, defillama_status = _collect_defillama(user_agent=user_agent, timeout=timeout)
    etherscan_global, etherscan_status = _collect_etherscan(user_agent=user_agent, timeout=timeout, api_key=etherscan_api_key)
    coingecko_rows, coingecko_prices, coingecko_status = _collect_coingecko(asset_order, user_agent=user_agent, timeout=timeout)

    news_rows, news_statuses = _collect_crypto_news(
        asset_to_symbols=asset_to_symbols,
        user_agent=user_agent,
        timeout=min(max(timeout, 4.0), 10.0),
        max_items_per_source=12,
    )

    for rows in (deribit_rows, kraken_rows, hyper_rows, coinmetrics_rows, coingecko_rows):
        for asset, feature_map in rows.items():
            asset_feature_accum.setdefault(asset, {}).update(feature_map)
    for source_name, price_map in (
        ("deribit", deribit_prices),
        ("kraken", kraken_prices),
        ("hyperliquid", hyper_prices),
        ("coinmetrics", coinmetrics_prices),
        ("coingecko", coingecko_prices),
    ):
        for asset, price in price_map.items():
            if price > 0.0:
                asset_provider_prices.setdefault(asset, {})[source_name] = price

    agreement_scores: list[float] = []
    momentum_scores: list[float] = []
    iv_scores: list[float] = []
    funding_scores: list[float] = []
    compared_assets = 0
    for asset in asset_order:
        feature_map = asset_feature_accum.setdefault(asset, {})
        agreement = _price_agreement_norm(list(asset_provider_prices.get(asset, {}).values()), max_relative_spread=max_relative_spread)
        if agreement > 0.0:
            agreement_scores.append(agreement)
            compared_assets += 1
        feature_map["crypto_cross_provider_price_agreement_norm"] = agreement
        if feature_map.get("crypto_coingecko_momentum_norm", 0.0) > 0.0:
            momentum_scores.append(float(feature_map["crypto_coingecko_momentum_norm"]))
        if feature_map.get("crypto_deribit_mark_iv_norm", 0.0) > 0.0:
            iv_scores.append(float(feature_map["crypto_deribit_mark_iv_norm"]))
        if feature_map.get("crypto_hyperliquid_funding_norm", 0.0) > 0.0:
            funding_scores.append(float(feature_map["crypto_hyperliquid_funding_norm"]))
        for symbol in asset_to_symbols.get(asset, []):
            symbol_features.setdefault(symbol, _zero_feature_map()).update(feature_map)

    global_features: dict[str, float] = {}
    if agreement_scores:
        global_features["crypto_cross_provider_price_agreement_norm"] = sum(agreement_scores) / len(agreement_scores)
    if momentum_scores:
        global_features["crypto_coingecko_momentum_norm"] = sum(momentum_scores) / len(momentum_scores)
    if iv_scores:
        global_features["crypto_deribit_mark_iv_norm"] = sum(iv_scores) / len(iv_scores)
    if funding_scores:
        global_features["crypto_hyperliquid_funding_norm"] = sum(funding_scores) / len(funding_scores)
    global_features.update(defillama_global)
    global_features.update(etherscan_global)

    market_statuses = {
        "deribit": deribit_status,
        "kraken": kraken_status,
        "hyperliquid": hyper_status,
        "coinmetrics": coinmetrics_status,
        "defillama": defillama_status,
        "etherscan": etherscan_status,
        "coingecko": coingecko_status,
    }
    market_ok_source_count = sum(1 for source in market_statuses.values() if bool(source.get("ok", False)))
    news_ok_source_count = sum(1 for source in news_statuses.values() if bool(source.get("ok", False)))

    all_errors: list[str] = []
    all_warnings: list[str] = []
    for source_name, source_status in market_statuses.items():
        err = source_status.get("error")
        if isinstance(err, str) and err.strip():
            qualified = f"{source_name}:{err}"
            if bool(source_status.get("ok", False)):
                all_warnings.append(qualified)
            else:
                all_errors.append(qualified)
    for source_name, source_status in news_statuses.items():
        err = source_status.get("error")
        if isinstance(err, str) and err.strip():
            all_warnings.append(f"{source_name}:{err}")
        elif not bool(source_status.get("ok", False)) and bool(source_status.get("blocked_by_cloudflare", False)):
            all_warnings.append(f"{source_name}:cloudflare_challenge")

    proxy_symbol = "BTC-USD" if "BTC-USD" in tracked_symbols else (tracked_symbols[0] if tracked_symbols else "BTC-USD")
    news_features = _summarize_crypto_news_items(news_rows, symbol=proxy_symbol, now_ts=now_ts, max_items=80)
    news_symbol_features = {
        symbol: _summarize_crypto_news_items(news_rows, symbol=symbol, now_ts=now_ts, max_items=60)
        for symbol in tracked_symbols
    }

    status = {
        "timestamp_utc": now.isoformat(),
        "ok": market_ok_source_count >= 5 and compared_assets >= 1,
        "tracked_symbols": len(tracked_symbols),
        "tracked_assets": len(asset_order),
        "ok_source_count": market_ok_source_count + news_ok_source_count,
        "source_count": len(market_statuses) + len(news_statuses),
        "market_ok_source_count": market_ok_source_count,
        "market_source_count": len(market_statuses),
        "news_ok_source_count": news_ok_source_count,
        "news_source_count": len(news_statuses),
        "news_row_count": len(news_rows),
        "news_symbol_count": sum(1 for feats in news_symbol_features.values() if float(feats.get("news_available", 0.0) or 0.0) > 0.0),
        "compared_assets": compared_assets,
        "error_count": len(all_errors),
        "warning_count": len(all_warnings),
        "errors": all_errors[:14],
        "warnings": all_warnings[:14],
        "sources": {
            **market_statuses,
            **news_statuses,
        },
    }
    payload = {
        "timestamp_utc": now.isoformat(),
        "provider": "crypto_market_context",
        "tracked_symbols": tracked_symbols,
        "tracked_assets": asset_order,
        "status": status,
        "sources": {
            "provider_prices": asset_provider_prices,
        },
        "derived": {
            "calendar_features": {},
            "news_features": news_features,
            "news_rows": news_rows[:120],
            "news_symbol_features": news_symbol_features,
            "news_source_counts": {
                name: int(source_status.get("rows", 0) or 0)
                for name, source_status in news_statuses.items()
            },
            "global_features": global_features,
            "symbol_features": symbol_features,
        },
    }
    return payload, status


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect free/public crypto market context for the crypto bots.")
    parser.add_argument("--symbols", default=",".join(_default_symbols()))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("CRYPTO_MARKET_CONTEXT_TIMEOUT_SECONDS", "12")))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload, status = collect_crypto_market_context(
        symbols=_parse_symbols(args.symbols),
        user_agent=str(os.getenv("CRYPTO_MARKET_CONTEXT_USER_AGENT") or USER_AGENT_DEFAULT).strip() or USER_AGENT_DEFAULT,
        timeout=float(args.timeout),
        max_relative_spread=max(float(os.getenv("CRYPTO_MARKET_CONTEXT_MAX_REL_SPREAD", "0.05")), 0.01),
        coinmetrics_api_key=str(os.getenv("COINMETRICS_API_KEY", "")).strip(),
        etherscan_api_key=str(os.getenv("ETHERSCAN_API_KEY", "")).strip(),
    )

    _write_json(PROJECT_ROOT / "exports" / "external_context" / "crypto_market_context_latest.json", payload)
    _write_json(PROJECT_ROOT / "governance" / "health" / "crypto_market_context_sync_latest.json", status)

    if args.json:
        print(json.dumps(status, ensure_ascii=True))
    else:
        print(
            "crypto_market_context ok={ok} tracked_symbols={tracked} ok_sources={ok_sources}/{total_sources} compared_assets={compared} news_rows={news_rows}".format(
                ok=status["ok"],
                tracked=status["tracked_symbols"],
                ok_sources=status["ok_source_count"],
                total_sources=status["source_count"],
                compared=status["compared_assets"],
                news_rows=status["news_row_count"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
