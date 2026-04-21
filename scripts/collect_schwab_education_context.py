#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urljoin, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.collector_transport import attach_collection_confidence, fetch_text
from core.market_context_features import summarize_structured_news_items


USER_AGENT_DEFAULT = "schwab-trading-bot/1.0"
SOURCE_CONTRACTS = {
    "schwab_live_webcasts": {"source_confidence_norm": 0.98, "schema_confidence_norm": 0.94},
    "schwab_ondemand_webcasts": {"source_confidence_norm": 0.98, "schema_confidence_norm": 0.94},
    "schwab_learn_trading": {"source_confidence_norm": 0.97, "schema_confidence_norm": 0.92},
    "schwab_network_page": {"source_confidence_norm": 0.97, "schema_confidence_norm": 0.91},
    "charles_schwab_youtube": {"source_confidence_norm": 0.95, "schema_confidence_norm": 0.9},
    "schwab_network_youtube": {"source_confidence_norm": 0.95, "schema_confidence_norm": 0.9},
}


def _resolve_yt_dlp_bin() -> str | None:
    discovered = shutil.which("yt-dlp")
    if discovered:
        return discovered
    for candidate in (
        "/opt/homebrew/bin/yt-dlp",
        "/usr/local/bin/yt-dlp",
    ):
        if Path(candidate).exists():
            return candidate
    return None


YT_DLP_BIN = _resolve_yt_dlp_bin()
YT_PLAYLIST_TIMEOUT_MIN_SECONDS = 10
YT_PLAYLIST_TIMEOUT_MAX_SECONDS = 30
_GENERIC_BROAD_SYMBOLS = ("SPY", "QQQ", "DIA", "IWM", "TLT")
PAGE_SPECS = [
    {
        "id": "schwab_live_webcasts",
        "title": "Schwab Coaching Live Webcasts",
        "url": "https://www.schwab.com/coaching/webcasts",
        "publisher": "Charles Schwab",
        "kind": "coaching",
    },
    {
        "id": "schwab_ondemand_webcasts",
        "title": "Schwab Coaching On-Demand Webcasts",
        "url": "https://www.schwab.com/coaching/ondemand-webcasts",
        "publisher": "Charles Schwab",
        "kind": "coaching",
    },
    {
        "id": "schwab_learn_trading",
        "title": "Schwab Learn Trading",
        "url": "https://www.schwab.com/learn/trading",
        "publisher": "Charles Schwab",
        "kind": "learn",
    },
    {
        "id": "schwab_network_page",
        "title": "Schwab Network",
        "url": "https://www.schwab.com/schwab-network",
        "publisher": "Schwab Network",
        "kind": "network",
    },
]
CHANNEL_SPECS = [
    {
        "id": "charles_schwab_youtube",
        "title": "Charles Schwab YouTube",
        "channel_name": "Charles Schwab",
        "url": "https://www.youtube.com/@CharlesSchwab",
        "publisher": "Charles Schwab",
    },
    {
        "id": "schwab_network_youtube",
        "title": "Schwab Network YouTube",
        "channel_name": "Schwab Network",
        "url": "https://www.youtube.com/@SchwabNetwork",
        "publisher": "Schwab Network",
    },
]
_DATE_RE = re.compile(r"\b([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})\b")
_BAD_TEXT_TOKENS = {
    "find a branch",
    "contact us",
    "chat",
    "open an account",
    "log in",
    "privacy",
    "terms",
    "site map",
    "account protection",
    "sipc",
    "fdic insurance",
    "broker check",
    "important notices",
    "accessibility help",
}
_BAD_URL_TOKENS = (
    "privacy",
    "terms",
    "facebook.com",
    "linkedin.com",
    "twitter.com",
    "instagram.com",
    "brokercheck.finra.org",
    "sipc.org",
    "occ.com",
)
_ALLOWED_HOSTS = {
    "schwab.com",
    "www.schwab.com",
    "schwabnetwork.com",
    "www.schwabnetwork.com",
    "youtube.com",
    "www.youtube.com",
}
_EXPLICIT_SYMBOL_RE = re.compile(r"\$([A-Z]{1,6}(?:-[A-Z]{2,6})?)\b")
_PAREN_SYMBOL_RE = re.compile(r"\(([A-Z]{1,5})\)")
_UPPERCASE_SYMBOL_RE = re.compile(r"\b([A-Z]{1,5})\b")
_SYMBOL_STOPWORDS = {
    "A",
    "AI",
    "AM",
    "APR",
    "AUG",
    "CEO",
    "CFO",
    "CPI",
    "EPS",
    "ETF",
    "ETFS",
    "FED",
    "FOMC",
    "GDP",
    "IPO",
    "LIVE",
    "MAY",
    "NAV",
    "NEWS",
    "PM",
    "PPI",
    "SEC",
    "TV",
    "USA",
    "USD",
}
_SYMBOL_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "apple": ("AAPL",),
    "microsoft": ("MSFT",),
    "nvidia": ("NVDA",),
    "amazon": ("AMZN",),
    "tesla": ("TSLA",),
    "meta": ("META",),
    "facebook": ("META",),
    "google": ("GOOGL",),
    "alphabet": ("GOOGL",),
    "netflix": ("NFLX",),
    "amd": ("AMD",),
    "intel": ("INTC",),
    "broadcom": ("AVGO",),
    "palantir": ("PLTR",),
    "coinbase": ("COIN",),
    "berkshire": ("BRK-B",),
    "schwab": ("SCHW",),
    "charles schwab": ("SCHW",),
    "s&p 500": ("SPY",),
    "sp 500": ("SPY",),
    "nasdaq 100": ("QQQ",),
    "nasdaq": ("QQQ",),
    "dow jones": ("DIA",),
    "small caps": ("IWM",),
    "russell 2000": ("IWM",),
    "treasury": ("TLT",),
    "treasuries": ("TLT",),
    "bond market": ("TLT",),
    "rates": ("TLT",),
    "gold": ("GLD",),
    "oil": ("USO", "XLE"),
    "energy": ("XLE",),
    "financials": ("XLF",),
    "banks": ("XLF",),
    "technology": ("XLK",),
    "tech": ("XLK",),
    "industrials": ("XLI",),
    "semiconductors": ("SMH",),
    "dividend": ("SCHD",),
    "bitcoin": ("BTC-USD",),
    "ethereum": ("ETH-USD",),
}
_KNOWN_TEXT_SYMBOLS = {symbol for symbols in _SYMBOL_ALIAS_MAP.values() for symbol in symbols}
_BROAD_MARKET_TOKENS = (
    "market",
    "economy",
    "economic",
    "fed",
    "federal reserve",
    "rates",
    "inflation",
    "jobs",
    "treasury",
    "yield",
    "volatility",
    "s&p",
    "nasdaq",
    "dow",
    "russell",
)
_POSITIVE_NEWS_TOKENS = (
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
    "record high",
    "upside",
    "strength",
    "strong",
    "upgrade",
    "upgrades",
    "raises",
)
_NEGATIVE_NEWS_TOKENS = (
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
    "weakness",
    "weak",
    "risk",
)
_SHOCK_NEWS_TOKENS = (
    "breaking",
    "urgent",
    "shock",
    "surprise",
    "crash",
    "spike",
    "spikes",
    "plunge",
    "plunges",
    "probe",
    "lawsuit",
    "tariff",
    "emergency",
    "halt",
    "recall",
)
_SIGNAL_TYPE_TOKENS = {
    "earnings": ("earnings", "eps", "revenue", "results", "beat", "miss"),
    "guidance": ("guidance", "outlook", "forecast", "raises", "cuts"),
    "regulatory": ("sec", "fda", "lawsuit", "probe", "investigation", "antitrust"),
    "macro": ("fed", "inflation", "jobs", "treasury", "yield", "rates"),
    "options": ("option", "options", "volatility", "iv", "gamma"),
    "dividend": ("dividend", "yield", "income", "reinvest"),
}


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        self._href = ""
        for key, value in attrs:
            if key.lower() == "href" and value:
                self._href = value
                break
        self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href is None:
            return
        text = " ".join(str(data or "").split()).strip()
        if text:
            self._parts.append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._href is None:
            return
        text = " ".join(self._parts).strip()
        self.links.append((self._href, text))
        self._href = None
        self._parts = []


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _source_contract(source_name: str) -> dict[str, float]:
    row = SOURCE_CONTRACTS.get(str(source_name or ""), {})
    return {
        "source_confidence_norm": float(row.get("source_confidence_norm", 0.9) or 0.9),
        "schema_confidence_norm": float(row.get("schema_confidence_norm", 0.9) or 0.9),
    }


def _fetch_text_result(url: str, *, source_name: str, user_agent: str, timeout: float) -> dict[str, Any]:
    contract = _source_contract(source_name)
    return fetch_text(
        url=url,
        user_agent=user_agent,
        timeout=timeout,
        collector_key="schwab_education_context",
        source_name=source_name,
        entity_key=url,
        project_root=PROJECT_ROOT,
        source_confidence_norm=contract["source_confidence_norm"],
        schema_confidence_norm=contract["schema_confidence_norm"],
        accept="text/html,application/json,text/plain,*/*",
    )


def _safe_http_text(url: str, *, user_agent: str, timeout: float) -> tuple[str | None, str | None]:
    result = _fetch_text_result(url, source_name="schwab_live_webcasts", user_agent=user_agent, timeout=timeout)
    if bool(result.get("ok", False)):
        return str(result.get("text") or ""), None
    return None, str(result.get("error") or "fetch_failed")


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").replace("\xa0", " ").split()).strip()


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _clamp11(value: float) -> float:
    return max(-1.0, min(float(value), 1.0))


def _link_date(text: str) -> str:
    match = _DATE_RE.search(str(text or ""))
    if not match:
        return ""
    try:
        return datetime.strptime(match.group(1), "%b %d, %Y").replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        return ""


def _normalize_url(base_url: str, href: str) -> str:
    url = urljoin(base_url, str(href or "").strip())
    parsed = urlparse(url)
    clean = parsed._replace(fragment="", query=parsed.query)
    return clean.geturl()


def _looks_relevant(url: str, text: str) -> bool:
    cleaned = _clean_text(text).lower()
    if len(cleaned) < 8 or cleaned in _BAD_TEXT_TOKENS:
        return False
    if any(token in cleaned for token in ("member sipc", "open an account", "satisfaction guarantee")):
        return False
    lower_url = str(url or "").lower()
    if any(token in lower_url for token in _BAD_URL_TOKENS):
        return False
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc.lower() not in _ALLOWED_HOSTS:
        return False
    path = parsed.path.lower()
    if "youtube.com" in parsed.netloc.lower():
        return True
    return any(
        token in path
        for token in (
            "/learn/",
            "/coaching/",
            "/schwab-network",
            "/resource/",
            "/video",
            "/shows",
        )
    )


def _content_type(url: str, text: str) -> str:
    merged = f"{url} {text}".lower()
    if "youtube.com" in merged:
        return "youtube_video"
    if "webcast" in merged or "workshop" in merged:
        return "webcast"
    if "video" in merged or "/video" in merged:
        return "video"
    if "article" in merged:
        return "article"
    if "podcast" in merged:
        return "podcast"
    return "page"


def _published_ts(raw: Any) -> float | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp()


def _headline_sentiment_and_shock(text: str) -> tuple[float, float]:
    lowered = str(text or "").lower()
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


def _signal_types(text: str) -> list[str]:
    lowered = str(text or "").lower()
    out: list[str] = []
    for label, tokens in _SIGNAL_TYPE_TOKENS.items():
        if any(token in lowered for token in tokens):
            out.append(label)
    return out


def _related_symbols_from_text(text: str) -> list[str]:
    raw_text = str(text or "")
    lowered = raw_text.lower()
    out: list[str] = []
    seen: set[str] = set()

    def add(symbol: str) -> None:
        token = str(symbol or "").strip().upper()
        if not token or token in seen:
            return
        seen.add(token)
        out.append(token)

    for match in _EXPLICIT_SYMBOL_RE.finditer(raw_text):
        add(match.group(1))
    for match in _PAREN_SYMBOL_RE.finditer(raw_text):
        token = match.group(1).strip().upper()
        if token not in _SYMBOL_STOPWORDS:
            add(token)
    for alias, symbols in _SYMBOL_ALIAS_MAP.items():
        if re.search(r"\b" + re.escape(alias.lower()) + r"\b", lowered):
            for symbol in symbols:
                add(symbol)
    for match in _UPPERCASE_SYMBOL_RE.finditer(raw_text):
        token = match.group(1).strip().upper()
        if token in _SYMBOL_STOPWORDS:
            continue
        if token in _KNOWN_TEXT_SYMBOLS:
            add(token)
    return out


def _source_quality_norm(row: Mapping[str, Any]) -> float:
    publisher = str(row.get("publisher") or row.get("source") or "").strip().lower()
    if "charles schwab" in publisher:
        return 0.97
    if "schwab network" in publisher:
        return 0.95
    return 0.9


def _enrich_training_row(row: Mapping[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    text = " ".join(
        part
        for part in (
            str(row.get("headline") or ""),
            str(row.get("title") or ""),
            str(row.get("summary") or ""),
            str(row.get("channel") or ""),
        )
        if str(part or "").strip()
    ).strip()
    related = _related_symbols_from_text(text)
    lowered = text.lower()
    broad_market = (not related) or any(token in lowered for token in _BROAD_MARKET_TOKENS)
    if broad_market and not related:
        related = list(_GENERIC_BROAD_SYMBOLS)
    if related:
        enriched["symbols"] = related
        enriched["relatedSymbols"] = related
    enriched["broad_market"] = bool(broad_market)
    enriched["macro_event"] = False
    enriched["source_quality_norm"] = round(_source_quality_norm(enriched), 4)
    signal_types = _signal_types(text)
    if signal_types:
        enriched["signal_types"] = signal_types
    sentiment_hint, shock_hint = _headline_sentiment_and_shock(text)
    if abs(sentiment_hint) > 0.0:
        enriched["sentiment_hint"] = round(sentiment_hint, 4)
    if shock_hint > 0.0:
        enriched["shock_hint"] = round(shock_hint, 4)
    return enriched


def _collect_news_feature_bundle(
    rows: list[dict[str, Any]],
    *,
    now_ts: float,
    max_items: int,
    symbol: str,
) -> dict[str, float]:
    filtered: list[tuple[float, dict[str, Any], str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        headline = _clean_text(
            " ".join(
                part
                for part in (
                    str(row.get("headline") or ""),
                    str(row.get("title") or ""),
                    str(row.get("summary") or ""),
                )
                if str(part or "").strip()
            )
        )
        if not headline:
            continue
        ts = _published_ts(row.get("publishedDate") or row.get("published"))
        if ts is None or ts > now_ts:
            continue
        age = now_ts - ts
        if age > 48.0 * 3600.0:
            continue
        filtered.append((age, dict(row), headline))
    if not filtered:
        return {}

    filtered.sort(key=lambda item: item[0])
    filtered = filtered[: max(max_items, 1)]
    c30 = c2h = c24h = 0
    pos_n = neg_n = shock_n = 0
    sent_sum = 0.0
    impact_sum = 0.0
    weight_sum = 0.0
    for age, row, headline in filtered:
        if age <= 30.0 * 60.0:
            c30 += 1
        if age <= 2.0 * 3600.0:
            c2h += 1
        if age <= 24.0 * 3600.0:
            c24h += 1
        sentiment_hint = float(row.get("sentiment_hint", 0.0) or 0.0)
        shock_hint = float(row.get("shock_hint", 0.0) or 0.0)
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

    n = len(filtered)
    denom = float(max(max_items, 1))
    features = {
        "news_available": min(n / denom, 1.0),
        "news_items_30m": min(c30 / denom, 1.0),
        "news_items_2h": min(c2h / denom, 1.0),
        "news_items_24h": min(c24h / denom, 1.0),
        "news_sentiment": _clamp11((sent_sum / weight_sum) if weight_sum > 0.0 else 0.0),
        "news_negative_share": neg_n / max(n, 1),
        "news_positive_share": pos_n / max(n, 1),
        "news_shock_rate": shock_n / max(n, 1),
        "news_recent_impact": min(impact_sum / max(weight_sum, 1e-8), 1.0),
    }
    features.update(
        summarize_structured_news_items(
            [row for _, row, _ in filtered],
            symbol=symbol,
            now_ts=now_ts,
            max_items=max_items,
        )
    )
    return {key: round(float(value), 6) for key, value in features.items() if math.isfinite(float(value))}


def _rows_for_symbol(rows: list[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target = str(symbol or "").strip().upper()
    for row in rows:
        related = {str(item or "").strip().upper() for item in list(row.get("symbols") or []) if str(item or "").strip()}
        if target in related:
            out.append(row)
    return out


def _schwab_global_features(
    *,
    items: list[dict[str, Any]],
    news_rows: list[dict[str, Any]],
    symbol_features: Mapping[str, Mapping[str, float]],
    now: datetime,
) -> dict[str, float]:
    total_items = float(max(len(items), 1))
    video_items = sum(1 for row in items if str(row.get("content_type") or "").lower() in {"youtube_video", "youtube_stream", "video", "webcast"})
    stream_items = sum(1 for row in items if str(row.get("content_type") or "").lower() == "youtube_stream")
    network_items = sum(1 for row in items if "schwab network" in str(row.get("publisher") or "").lower())
    return {
        "schwab_education_item_density_norm": round(min(len(news_rows) / 40.0, 1.0), 6),
        "schwab_education_recent_activity_norm": round(min(_recent_item_count(news_rows, now=now, lookback_hours=24.0) / 24.0, 1.0), 6),
        "schwab_education_symbol_coverage_norm": round(min(len(symbol_features) / 12.0, 1.0), 6),
        "schwab_education_video_share_norm": round(video_items / total_items, 6),
        "schwab_education_stream_share_norm": round(stream_items / total_items, 6),
        "schwab_education_network_share_norm": round(network_items / total_items, 6),
    }


def _extract_page_items(spec: Mapping[str, str], html: str, *, max_items: int) -> list[dict[str, Any]]:
    parser = _LinkParser()
    parser.feed(html)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for href, text in parser.links:
        url = _normalize_url(str(spec["url"]), href)
        title = _clean_text(text)
        if not _looks_relevant(url, title):
            continue
        if url in seen:
            continue
        seen.add(url)
        published = _link_date(title)
        row = {
            "headline": title,
            "title": title,
            "url": url,
            "publisher": str(spec["publisher"]),
            "source": str(spec["publisher"]),
            "source_id": str(spec["id"]),
            "channel": str(spec["title"]),
            "page_id": str(spec["id"]),
            "page_url": str(spec["url"]),
            "content_type": _content_type(url, title),
        }
        if published:
            row["publishedDate"] = published
        rows.append(row)
        if len(rows) >= max_items:
            break
    if not rows:
        rows.append(
            {
                "headline": str(spec["title"]),
                "title": str(spec["title"]),
                "url": str(spec["url"]),
                "publisher": str(spec["publisher"]),
                "source": str(spec["publisher"]),
                "source_id": str(spec["id"]),
                "channel": str(spec["title"]),
                "page_id": str(spec["id"]),
                "page_url": str(spec["url"]),
                "content_type": "landing_page",
            }
        )
    return rows


def _parse_playlist_payload(raw: str) -> dict[str, Any]:
    payload = json.loads(raw or "{}")
    return payload if isinstance(payload, dict) else {}


def _yt_playlist(
    channel_url: str,
    suffix: str,
    *,
    timeout: float,
    playlist_limit: int | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    if not YT_DLP_BIN:
        return None, "yt_dlp_not_found"
    target_url = f"{channel_url.rstrip('/')}/{suffix}"
    cmd = [YT_DLP_BIN, "--flat-playlist", "--dump-single-json", "--no-warnings"]
    if playlist_limit is not None and int(playlist_limit) > 0:
        cmd.extend(["--playlist-end", str(int(playlist_limit))])
    cmd.append(target_url)
    timeout_seconds = max(
        YT_PLAYLIST_TIMEOUT_MIN_SECONDS,
        min(int(math.ceil(float(timeout))), YT_PLAYLIST_TIMEOUT_MAX_SECONDS),
    )
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_seconds)
    except Exception as exc:
        return None, str(exc)
    if proc.returncode != 0:
        return None, (proc.stderr or proc.stdout or "yt_dlp_playlist_failed").strip()[-800:]
    try:
        return _parse_playlist_payload(proc.stdout), None
    except Exception as exc:
        return None, f"yt_dlp_json_parse_failed:{exc}"


def _entry_timestamp(entry: Mapping[str, Any]) -> str:
    raw = entry.get("timestamp") or entry.get("release_timestamp")
    try:
        if raw is not None:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc).isoformat()
    except Exception:
        pass
    upload_date = str(entry.get("upload_date") or "").strip()
    if len(upload_date) == 8 and upload_date.isdigit():
        try:
            return datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            return ""
    return ""


def _channel_items(spec: Mapping[str, str], *, max_items: int, timeout: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items: list[dict[str, Any]] = []
    status: dict[str, Any] = {
        "ok": False,
        "videos_ok": False,
        "streams_ok": False,
        "item_count": 0,
        "errors": [],
    }
    for suffix, content_type in (("videos", "youtube_video"), ("streams", "youtube_stream")):
        payload, error = _yt_playlist(str(spec["url"]), suffix, timeout=timeout, playlist_limit=max_items)
        if error or not payload:
            status["errors"].append(f"{suffix}:{error or 'empty_playlist'}")
            continue
        status[f"{suffix}_ok"] = True
        entries = payload.get("entries") if isinstance(payload.get("entries"), list) else []
        for entry in entries[:max_items]:
            if not isinstance(entry, Mapping):
                continue
            title = _clean_text(str(entry.get("title") or ""))
            if not title:
                continue
            video_url = str(entry.get("url") or entry.get("webpage_url") or "").strip()
            if video_url and not video_url.startswith("http"):
                if "watch?v=" not in video_url:
                    video_url = f"https://www.youtube.com/watch?v={video_url}"
            if not video_url:
                video_id = str(entry.get("id") or "").strip()
                if not video_id:
                    continue
                video_url = f"https://www.youtube.com/watch?v={video_id}"
            row = {
                "headline": title,
                "title": title,
                "url": video_url,
                "publisher": str(spec["publisher"]),
                "source": str(spec["publisher"]),
                "source_id": str(spec["id"]),
                "channel": str(spec["channel_name"]),
                "channel_url": str(spec["url"]),
                "content_type": content_type,
                "live_status": str(entry.get("live_status") or ""),
            }
            published = _entry_timestamp(entry)
            if published:
                row["publishedDate"] = published
            items.append(row)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in items:
        url = str(row.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(row)
    status["item_count"] = len(deduped)
    status["ok"] = bool(status["videos_ok"] or status["streams_ok"]) and bool(deduped)
    return deduped[: (max_items * 2)], status


def _recent_item_count(items: list[dict[str, Any]], *, now: datetime, lookback_hours: float) -> int:
    count = 0
    threshold = now.timestamp() - (max(float(lookback_hours), 0.0) * 3600.0)
    for row in items:
        raw = str(row.get("publishedDate") or "").strip()
        if not raw:
            continue
        try:
            ts = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
        except Exception:
            continue
        if ts >= threshold:
            count += 1
    return count


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in items:
        key = (str(row.get("url") or "").strip(), str(row.get("headline") or "").strip())
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduped.append(dict(row))
    return deduped


def _attach_item_contract(
    row: Mapping[str, Any],
    *,
    source_status: Mapping[str, Any],
    fallback_fetched_utc: str,
) -> dict[str, Any]:
    return attach_collection_confidence(
        dict(row),
        source_confidence_norm=float(source_status.get("source_confidence_norm", 0.0) or 0.0),
        schema_confidence_norm=float(source_status.get("schema_confidence_norm", 0.0) or 0.0),
        freshness_norm=float(source_status.get("freshness_norm", 0.0) or 0.0),
        fetched_utc=str(source_status.get("fetched_utc") or fallback_fetched_utc),
    )


def build_payload(*, timeout_seconds: float, user_agent: str, max_page_items: int, max_channel_items: int) -> tuple[dict[str, Any], dict[str, Any]]:
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    external_context_root = PROJECT_ROOT / "exports" / "external_context"
    health_root = PROJECT_ROOT / "governance" / "health"

    status: dict[str, Any] = {
        "timestamp_utc": now_iso,
        "provider": "schwab_education_context",
        "sources": {},
        "source_contracts": {},
    }

    page_items: list[dict[str, Any]] = []
    for spec in PAGE_SPECS:
        fetch_result = _fetch_text_result(
            str(spec["url"]),
            source_name=str(spec["id"]),
            user_agent=user_agent,
            timeout=timeout_seconds,
        )
        html = str(fetch_result.get("text") or "") if bool(fetch_result.get("ok", False)) else ""
        if not html:
            status["sources"][str(spec["id"])] = {
                "ok": False,
                "kind": "page",
                "source_url": str(spec["url"]),
                "item_count": 0,
                "required": True,
                "contract_participates": True,
                "source_confidence_norm": float(fetch_result.get("source_confidence_norm") or _source_contract(str(spec["id"]))["source_confidence_norm"]),
                "schema_confidence_norm": float(fetch_result.get("schema_confidence_norm") or _source_contract(str(spec["id"]))["schema_confidence_norm"]),
                "freshness_norm": float(fetch_result.get("freshness_norm") or 0.0),
                "fetched_utc": str(fetch_result.get("fetched_utc") or ""),
                "error": str(fetch_result.get("error") or "fetch_failed"),
            }
            continue
        rows = _extract_page_items(spec, html, max_items=max_page_items)
        page_items.extend(rows)
        status["sources"][str(spec["id"])] = {
            "ok": bool(rows),
            "kind": "page",
            "source_url": str(spec["url"]),
            "item_count": len(rows),
            "required": True,
            "contract_participates": True,
            "source_confidence_norm": float(fetch_result.get("source_confidence_norm") or _source_contract(str(spec["id"]))["source_confidence_norm"]),
            "schema_confidence_norm": float(fetch_result.get("schema_confidence_norm") or _source_contract(str(spec["id"]))["schema_confidence_norm"]),
            "freshness_norm": float(fetch_result.get("freshness_norm") or 0.0),
            "fetched_utc": str(fetch_result.get("fetched_utc") or ""),
            "error": "",
        }

    channel_items: list[dict[str, Any]] = []
    for spec in CHANNEL_SPECS:
        rows, channel_status = _channel_items(spec, max_items=max_channel_items, timeout=timeout_seconds)
        channel_items.extend(rows)
        status["sources"][str(spec["id"])] = {
            "ok": bool(channel_status.get("ok", False)),
            "kind": "youtube_channel",
            "source_url": str(spec["url"]),
            "item_count": int(channel_status.get("item_count", 0) or 0),
            "videos_ok": bool(channel_status.get("videos_ok", False)),
            "streams_ok": bool(channel_status.get("streams_ok", False)),
            "required": True,
            "contract_participates": True,
            "source_confidence_norm": float(channel_status.get("source_confidence_norm") or _source_contract(str(spec["id"]))["source_confidence_norm"]),
            "schema_confidence_norm": float(channel_status.get("schema_confidence_norm") or _source_contract(str(spec["id"]))["schema_confidence_norm"]),
            "freshness_norm": float(channel_status.get("freshness_norm") or (1.0 if channel_status.get("ok", False) else 0.0)),
            "fetched_utc": str(channel_status.get("fetched_utc") or now_iso),
            "error": "; ".join(channel_status.get("errors") or []),
        }

    items = [
        _attach_item_contract(
            _enrich_training_row(row),
            source_status=status["sources"].get(str(row.get("source_id") or ""), {}),
            fallback_fetched_utc=now_iso,
        )
        for row in _dedupe_items(page_items + channel_items)[:240]
    ]
    news_rows = items[:120]
    news_features = _collect_news_feature_bundle(news_rows, now_ts=now.timestamp(), max_items=120, symbol="SPY") if news_rows else {}
    symbol_counter = Counter(
        str(symbol or "").strip().upper()
        for row in news_rows
        for symbol in list(row.get("symbols") or [])
        if str(symbol or "").strip()
    )
    symbol_features: dict[str, dict[str, float]] = {}
    for symbol, _count in symbol_counter.most_common(24):
        rows_for_symbol = _rows_for_symbol(news_rows, symbol)
        if not rows_for_symbol:
            continue
        feature_map = _collect_news_feature_bundle(rows_for_symbol, now_ts=now.timestamp(), max_items=40, symbol=symbol)
        if not feature_map:
            continue
        recent_ts = max((_published_ts(row.get("publishedDate") or row.get("published")) or 0.0) for row in rows_for_symbol)
        stream_share = sum(1 for row in rows_for_symbol if str(row.get("content_type") or "").lower() == "youtube_stream") / max(len(rows_for_symbol), 1)
        feature_map["schwab_education_symbol_frequency_norm"] = round(min(len(rows_for_symbol) / 8.0, 1.0), 6)
        feature_map["schwab_education_symbol_recency_norm"] = round(math.exp(-max(now.timestamp() - recent_ts, 0.0) / (6.0 * 3600.0)) if recent_ts > 0.0 else 0.0, 6)
        feature_map["schwab_education_symbol_stream_share_norm"] = round(stream_share, 6)
        symbol_features[symbol] = feature_map
    global_features = _schwab_global_features(items=items, news_rows=news_rows, symbol_features=symbol_features, now=now)

    content_type_counts = dict(Counter(str(row.get("content_type") or "unknown") for row in items))
    publisher_counts = dict(Counter(str(row.get("publisher") or "unknown") for row in items))
    ok_count = sum(1 for row in status["sources"].values() if isinstance(row, Mapping) and bool(row.get("ok", False)))
    total_count = len(status["sources"])
    min_ok_sources_required = max(1, int((max(total_count, 1) * 2 + 2) // 3))
    status.update(
        {
            "ok": ok_count >= min_ok_sources_required and total_count > 0 and bool(items),
            "ok_source_count": ok_count,
            "source_count": total_count,
            "min_ok_sources_required": min_ok_sources_required,
            "item_count": len(items),
            "page_item_count": len(page_items),
            "channel_item_count": len(channel_items),
        }
    )

    payload = {
        "timestamp_utc": now_iso,
        "provider": "schwab_education_context",
        "collection_contract": {
            "source_contracts": {
                name: {
                    "source_confidence_norm": float((row or {}).get("source_confidence_norm", 0.0) or 0.0),
                    "schema_confidence_norm": float((row or {}).get("schema_confidence_norm", 0.0) or 0.0),
                    "freshness_norm": float((row or {}).get("freshness_norm", 0.0) or 0.0),
                }
                for name, row in status["sources"].items()
                if isinstance(row, Mapping)
            }
        },
        "status": status,
        "items": items,
        "derived": {
            "news_rows": news_rows,
            "news_features": news_features,
            "symbol_features": symbol_features,
            "global_features": global_features,
            "content_type_counts": content_type_counts,
            "publisher_counts": publisher_counts,
            "page_rows": page_items[:120],
            "channel_rows": channel_items[:120],
        },
        "paths": {
            "payload_path": str(external_context_root / "schwab_education_context_latest.json"),
            "status_path": str(health_root / "schwab_education_context_sync_latest.json"),
        },
    }
    payload["collection_contract"]["provider_confidence_norm"] = round(
        sum(
            float((row or {}).get("source_confidence_norm", 0.0) or 0.0)
            for row in status["sources"].values()
            if isinstance(row, Mapping) and bool(row.get("ok"))
        ) / max(sum(1 for row in status["sources"].values() if isinstance(row, Mapping) and bool(row.get("ok"))), 1),
        6,
    )
    status["source_contracts"] = payload["collection_contract"]["source_contracts"]
    return payload, status


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect official Schwab educational pages and channel archives into structured context.")
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--user-agent", default=USER_AGENT_DEFAULT)
    parser.add_argument("--max-page-items", type=int, default=40)
    parser.add_argument("--max-channel-items", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    payload, status = build_payload(
        timeout_seconds=float(args.timeout_seconds),
        user_agent=str(args.user_agent),
        max_page_items=max(int(args.max_page_items), 1),
        max_channel_items=max(int(args.max_channel_items), 1),
    )

    if not args.test_only:
        _write_json(PROJECT_ROOT / "exports" / "external_context" / "schwab_education_context_latest.json", payload)
        _write_json(PROJECT_ROOT / "governance" / "health" / "schwab_education_context_sync_latest.json", status)

    if args.json:
        print(json.dumps(status, ensure_ascii=True))
    else:
        print(
            "schwab_education_context ok={ok} ok_sources={ok_sources}/{total_sources} items={items}".format(
                ok=str(bool(status.get("ok", False))).lower(),
                ok_sources=int(status.get("ok_source_count", 0) or 0),
                total_sources=int(status.get("source_count", 0) or 0),
                items=int(status.get("item_count", 0) or 0),
            )
        )
        if not args.test_only:
            print(f"schwab_education_context_latest={PROJECT_ROOT / 'exports' / 'external_context' / 'schwab_education_context_latest.json'}")
            print(f"status_file={PROJECT_ROOT / 'governance' / 'health' / 'schwab_education_context_sync_latest.json'}")
    return 0 if bool(status.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
