#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.derivatives_features import summarize_calendar_payload
from core.market_context_features import load_latest_external_context, summarize_structured_news_items

USER_AGENT = "schwab-trading-bot/1.0"
RSS_DISCOVERY_RE = re.compile(r"""href=["']([^"']*(?:feed|rss|xml)[^"']*)["']""", re.IGNORECASE)
DATE_RE = re.compile(r"([A-Z][a-z]{2,9}\.? \d{1,2}, \d{4})")
TIME_RE = re.compile(r"^\d{1,2}:\d{2}\s*(?:a\.m\.|p\.m\.)$", re.IGNORECASE)
FED_EVENT_RE = re.compile(r"^(Speech|Discussion|Testimony|Remarks?)\s*-\s*(.+)$", re.IGNORECASE)
NUMERIC_FIELD_RE = re.compile(r"(?i)\b(actual|forecast|previous|prior|revised|revision)\b[^0-9\-]{0,8}(-?\d+(?:\.\d+)?)")
SPEAKER_RE = re.compile(r"(?i)\b(chair|vice chair|governor|president)\s+([A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){0,2})")

BLS_ICS_URL = "https://www.bls.gov/schedule/news_release/bls.ics"
FEED_DISCOVERY_PAGES = {
    "federal_reserve": "https://www.federalreserve.gov/feeds/feeds.htm",
    "treasury": "https://home.treasury.gov/news/press-releases",
    "bls": "https://www.bls.gov/bls/newsrels.htm",
    "bea": "https://www.bea.gov/news",
}
SOURCE_QUALITY = {
    "Federal Reserve": 0.99,
    "U.S. Treasury": 0.97,
    "Bureau of Labor Statistics": 0.96,
    "Bureau of Economic Analysis": 0.95,
}
HTML_LINK_RE = re.compile(r"""(?is)<a\b[^>]*href=["']([^"']+)["'][^>]*>(.*?)</a>""")
_ET_ZONE = ZoneInfo("America/New_York") if ZoneInfo is not None else None


def _federal_reserve_calendar_url(year: int, month: int) -> str:
    slug = datetime(year, month, 1, tzinfo=timezone.utc).strftime("%Y-%B").lower()
    return f"https://www.federalreserve.gov/newsevents/{slug}.htm"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _http_text(url: str, *, timeout: float = 25.0) -> str:
    req = Request(url=url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urlopen(req, timeout=max(float(timeout), 1.0)) as resp:
        return resp.read().decode("utf-8", "replace")


def _safe_http_text(url: str, *, timeout: float = 25.0) -> tuple[str | None, str | None]:
    try:
        return _http_text(url, timeout=timeout), None
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return None, str(exc)


def _parse_ts(raw: str) -> str | None:
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


def _source_publisher(source_name: str) -> str:
    return {
        "federal_reserve": "Federal Reserve",
        "treasury": "U.S. Treasury",
        "bls": "Bureau of Labor Statistics",
        "bea": "Bureau of Economic Analysis",
    }.get(source_name, source_name)


def _source_timeout_seconds(source_name: str, default_timeout: float) -> float:
    base = max(float(default_timeout), 1.0)
    if source_name == "treasury":
        return max(base, 15.0)
    if source_name == "bls":
        return max(base, 12.0)
    return base


def _market_session_label(raw_ts: str | None) -> str:
    if not raw_ts:
        return "unknown"
    try:
        dt = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
    except Exception:
        return "unknown"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_local = dt.astimezone(_ET_ZONE) if _ET_ZONE is not None else dt.astimezone(timezone.utc)
    minute = dt_local.hour * 60 + dt_local.minute
    if minute < 570:
        return "premarket"
    if minute <= 960:
        return "intraday"
    return "after_hours"


def _macro_event_type(text: str, source_name: str) -> str:
    lowered = str(text or "").lower()
    if any(tok in lowered for tok in ("fomc", "rate decision", "fed funds")):
        return "fomc"
    if source_name == "federal_reserve" and any(tok in lowered for tok in ("powell", "speech", "remarks", "testimony", "discussion")):
        return "fed_speech"
    if any(tok in lowered for tok in ("consumer price index", "cpi", "inflation")):
        return "inflation"
    if any(tok in lowered for tok in ("pce", "personal consumption expenditures")):
        return "pce"
    if any(tok in lowered for tok in ("employment situation", "nonfarm payroll", "payroll", "jobless", "unemployment", "labor")):
        return "labor"
    if any(tok in lowered for tok in ("gdp", "gross domestic product")):
        return "gdp"
    if any(tok in lowered for tok in ("pmi", "ism")):
        return "survey"
    if any(tok in lowered for tok in ("treasury auction", "note auction", "bond auction", "bill auction")):
        return "treasury_auction"
    if source_name == "bea":
        return "bea_release"
    if source_name == "bls":
        return "bls_release"
    return "macro_event"


def _macro_importance(event_type: str, text: str) -> tuple[str, float]:
    lowered = str(text or "").lower()
    high_types = {"fomc", "fed_speech", "inflation", "pce", "labor", "gdp", "treasury_auction"}
    medium_types = {"survey", "bea_release", "bls_release"}
    if event_type in high_types:
        return "High", 3.0
    if event_type in medium_types:
        return "Medium", 2.0
    if any(tok in lowered for tok in ("high impact", "major", "critical")):
        return "High", 3.0
    return "Medium", 2.0


def _extract_numeric_macro_fields(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for field, value_text in NUMERIC_FIELD_RE.findall(str(text or "")):
        try:
            value = float(value_text)
        except Exception:
            continue
        key = field.lower()
        if key == "prior":
            key = "previous"
        if key == "revision":
            key = "revised"
        out[key] = value
    return out


def _enrich_macro_row(row: dict[str, Any], *, source_name: str) -> dict[str, Any]:
    out = dict(row)
    title = str(out.get("title") or out.get("headline") or out.get("event") or "").strip()
    summary = str(out.get("summary") or "").strip()
    text = " ".join(part for part in (title, summary) if part).strip()
    event_type = _macro_event_type(text, source_name)
    importance, importance_score = _macro_importance(event_type, text)
    numeric_fields = _extract_numeric_macro_fields(text)
    speaker_match = SPEAKER_RE.search(text)
    speaker = speaker_match.group(2).strip() if speaker_match else ""
    published = str(out.get("datetime") or out.get("date") or out.get("published") or "").strip() or None
    out.setdefault("source", _source_publisher(source_name))
    out.setdefault("publisher", _source_publisher(source_name))
    out["macro_event_type"] = event_type
    out["importance"] = str(out.get("importance") or importance)
    out["impact"] = str(out.get("impact") or importance)
    out["importance_score"] = float(out.get("importance_score") or importance_score)
    out["market_session"] = str(out.get("market_session") or _market_session_label(published))
    out["source_quality_norm"] = float(out.get("source_quality_norm") or SOURCE_QUALITY.get(_source_publisher(source_name), 0.9))
    out["broad_market"] = bool(out.get("broad_market", True))
    out["macro_event"] = bool(out.get("macro_event", True))
    if speaker:
        out["speaker"] = speaker
    for key in ("actual", "forecast", "previous", "revised"):
        if key in numeric_fields and key not in out:
            out[key] = numeric_fields[key]
    return out


def _extract_feed_urls(page_url: str, page_html: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in RSS_DISCOVERY_RE.finditer(page_html):
        raw_candidate = match.group(1).strip()
        if not raw_candidate or " " in raw_candidate:
            continue
        if raw_candidate.startswith("#") or raw_candidate.lower().startswith("javascript:"):
            continue
        candidate = urljoin(page_url, raw_candidate)
        if not candidate.lower().startswith(("http://", "https://")):
            continue
        if "feedback" in candidate.lower():
            continue
        if "share?" in candidate.lower():
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        urls.append(candidate)
    return urls


def _discover_feed_urls(source_name: str, timeout: float, *, page_html: str | None = None) -> tuple[list[str], str | None]:
    page_url = FEED_DISCOVERY_PAGES[source_name]
    page_text = page_html
    error = None
    if page_text is None:
        page_text, error = _safe_http_text(page_url, timeout=timeout)
    if not page_text:
        return [], error
    urls = _extract_feed_urls(page_url, page_text)
    return urls[:3], None


def _parse_feed_items(xml_text: str, source_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return rows
    items = root.findall(".//item")
    if not items:
        items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
    publisher = _source_publisher(source_name)
    for item in items[:40]:
        title = (
            item.findtext("title")
            or item.findtext("{http://www.w3.org/2005/Atom}title")
            or ""
        ).strip()
        summary = (
            item.findtext("description")
            or item.findtext("summary")
            or item.findtext("{http://www.w3.org/2005/Atom}summary")
            or ""
        ).strip()
        link = item.findtext("link") or ""
        if not link:
            atom_link = item.find("{http://www.w3.org/2005/Atom}link")
            if atom_link is not None:
                link = atom_link.attrib.get("href", "")
        published = _parse_ts(
            item.findtext("pubDate")
            or item.findtext("updated")
            or item.findtext("{http://www.w3.org/2005/Atom}updated")
            or item.findtext("{http://www.w3.org/2005/Atom}published")
            or ""
        )
        if not title:
            continue
        rows.append(
            _enrich_macro_row(
                {
                    "headline": title,
                    "summary": summary,
                    "source": publisher,
                    "publisher": publisher,
                    "published": published,
                    "url": link,
                    "broad_market": True,
                    "macro_event": True,
                    "source_quality_norm": SOURCE_QUALITY.get(publisher, 0.9),
                },
                source_name=source_name,
            )
        )
    return rows


def _parse_news_links_from_html(page_text: str, source_name: str, page_url: str) -> list[dict[str, Any]]:
    publisher = _source_publisher(source_name)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    current_ts = datetime.now(timezone.utc).isoformat()
    for href, title_html in HTML_LINK_RE.findall(page_text or ""):
        candidate = urljoin(page_url, href.strip())
        if not candidate.lower().startswith(("http://", "https://")):
            continue
        if candidate in seen:
            continue
        title = re.sub(r"(?s)<[^>]+>", " ", title_html or "")
        title = " ".join(html.unescape(title).split()).strip()
        if len(title) < 12:
            continue
        lowered = title.lower()
        if lowered in {"news", "all news", "home", "subscribe", "press releases"}:
            continue
        if lowered.startswith(("read more", "learn more", "watch live", "view all")):
            continue
        if source_name == "treasury" and "/news/press-releases/" not in candidate.lower():
            continue
        if source_name == "bls" and not (
            "/news.release/" in candidate.lower()
            or any(
                token in lowered
                for token in (
                    "consumer price index",
                    "employment situation",
                    "producer price index",
                    "job openings",
                    "productivity",
                    "employment",
                    "payroll",
                )
            )
        ):
            continue
        if source_name == "bea" and not ("/news/" in candidate.lower() or "release" in lowered or "estimate" in lowered):
            continue
        if source_name == "federal_reserve" and not ("/newsevents/" in candidate.lower() or any(token in lowered for token in ("speech", "remarks", "statement", "minutes", "press release"))):
            continue
        seen.add(candidate)
        rows.append(
            _enrich_macro_row(
                {
                    "headline": title,
                    "summary": "",
                    "source": publisher,
                    "publisher": publisher,
                    "published": current_ts,
                    "url": candidate,
                    "broad_market": True,
                    "macro_event": True,
                    "source_quality_norm": SOURCE_QUALITY.get(publisher, 0.9),
                },
                source_name=source_name,
            )
        )
    return rows[:40]


def _parse_bls_ics(ics_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    event: dict[str, str] = {}
    in_event = False
    for raw in ics_text.splitlines():
        line = raw.strip()
        if line == "BEGIN:VEVENT":
            in_event = True
            event = {}
            continue
        if line == "END:VEVENT":
            if event:
                dt_raw = event.get("DTSTART") or event.get("DTSTART;VALUE=DATE")
                if not dt_raw:
                    for key, value in event.items():
                        if key.startswith("DTSTART;"):
                            dt_raw = value
                            break
                published = None
                if dt_raw:
                    try:
                        if "T" in dt_raw:
                            published = datetime.strptime(dt_raw.replace("Z", ""), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).isoformat()
                        else:
                            published = datetime.strptime(dt_raw, "%Y%m%d").replace(tzinfo=timezone.utc).isoformat()
                    except Exception:
                        published = None
                summary = event.get("SUMMARY", "").strip()
                if summary:
                    rows.append(
                        _enrich_macro_row(
                            {
                                "date": published,
                                "datetime": published,
                                "title": summary,
                                "event": summary,
                                "country": "United States",
                                "source": "Bureau of Labor Statistics",
                                "category": "macro_calendar",
                            },
                            source_name="bls",
                        )
                    )
            in_event = False
            event = {}
            continue
        if not in_event or ":" not in line:
            continue
        key, value = line.split(":", 1)
        event[key] = value
    return rows


def _strip_html_lines(html_text: str) -> list[str]:
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html_text)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>|</li>|</div>|</tr>|</td>|</h[1-6]>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    lines: list[str] = []
    for raw in text.splitlines():
        line = " ".join(raw.split()).strip()
        if line:
            lines.append(line)
    return lines


def _combine_month_day_time(year: int, month: int, day: int, time_text: str) -> str | None:
    normalized = time_text.lower().replace(".", "")
    try:
        dt = datetime.strptime(f"{year:04d}-{month:02d}-{day:02d} {normalized}", "%Y-%m-%d %I:%M %p")
    except Exception:
        return None
    return dt.replace(tzinfo=timezone.utc).isoformat()


def _parse_federal_reserve_calendar_text(page_text: str, *, year: int, month: int) -> list[dict[str, Any]]:
    lines = _strip_html_lines(page_text)
    rows: list[dict[str, Any]] = []
    idx = 0
    while idx < len(lines):
        time_line = lines[idx]
        if not TIME_RE.match(time_line):
            idx += 1
            continue
        if idx + 1 >= len(lines):
            break
        title_line = lines[idx + 1]
        if not FED_EVENT_RE.match(title_line):
            idx += 1
            continue
        description = ""
        day: int | None = None
        j = idx + 2
        while j < len(lines):
            line = lines[j]
            if TIME_RE.match(line):
                break
            if re.fullmatch(r"\d{1,2}", line):
                day = int(line)
                break
            lowered = line.lower()
            if lowered.startswith(("watch live", "read speech", "read remarks", "read testimony")):
                j += 1
                continue
            if not description and not line.startswith("At "):
                description = line
            j += 1
        if day is not None:
            published = _combine_month_day_time(year, month, day, time_line)
            if published:
                rows.append(
                    _enrich_macro_row(
                        {
                            "date": published,
                            "datetime": published,
                            "title": title_line,
                            "event": description or title_line,
                            "summary": description or title_line,
                            "country": "United States",
                            "source": "Federal Reserve",
                            "category": "macro_calendar",
                        },
                        source_name="federal_reserve",
                    )
                )
        idx = max(j, idx + 1)
    return rows


def _calendar_rows_from_news(news_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in news_rows:
        title = str(row.get("headline") or row.get("title") or "").strip()
        if not title:
            continue
        lowered = title.lower()
        if not any(tok in lowered for tok in ("cpi", "pce", "employment", "payroll", "gdp", "fomc", "powell", "auction", "treasury", "ism", "pmi", "remarks", "speech")):
            continue
        published = row.get("published")
        date_match = DATE_RE.search(title)
        if date_match:
            try:
                published = datetime.strptime(date_match.group(1).replace(".", ""), "%b %d, %Y").replace(tzinfo=timezone.utc).isoformat()
            except Exception:
                try:
                    published = datetime.strptime(date_match.group(1), "%B %d, %Y").replace(tzinfo=timezone.utc).isoformat()
                except Exception:
                    pass
        rows.append(
            _enrich_macro_row(
                {
                    "date": published,
                    "datetime": published,
                    "title": title,
                    "event": title,
                    "summary": str(row.get("summary") or "").strip(),
                    "country": "United States",
                    "source": row.get("source") or row.get("publisher") or "official_macro_news",
                    "category": "macro_calendar",
                },
                source_name="official_macro_news",
            )
        )
    return rows


def _load_existing_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect(args: argparse.Namespace) -> int:
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    external_context_root = PROJECT_ROOT / "exports" / "external_context"
    health_root = PROJECT_ROOT / "governance" / "health"

    status: dict[str, Any] = {
        "timestamp_utc": now_iso,
        "ok": True,
        "sources": {},
    }

    calendar_rows: list[dict[str, Any]] = []
    news_rows: list[dict[str, Any]] = []

    bls_ics, bls_error = _safe_http_text(BLS_ICS_URL, timeout=args.timeout_seconds)
    status["sources"]["bls_calendar"] = {"url": BLS_ICS_URL, "ok": bls_ics is not None, "error": bls_error}
    if bls_ics:
        calendar_rows.extend(_parse_bls_ics(bls_ics))

    fed_calendar_rows: list[dict[str, Any]] = []
    fed_calendar_errors: list[str] = []
    month_refs = [
        (now.year, now.month),
        ((now + timedelta(days=31)).year, (now + timedelta(days=31)).month),
    ]
    seen_months: set[tuple[int, int]] = set()
    for year, month in month_refs:
        if (year, month) in seen_months:
            continue
        seen_months.add((year, month))
        page_url = _federal_reserve_calendar_url(year, month)
        page_text, page_error = _safe_http_text(page_url, timeout=args.timeout_seconds)
        if not page_text:
            if page_error:
                future_month = (year, month) != (now.year, now.month)
                if not (future_month and "HTTP Error 404" in str(page_error)):
                    fed_calendar_errors.append(f"{year}-{month:02d}:{page_error}")
            continue
        fed_calendar_rows.extend(_parse_federal_reserve_calendar_text(page_text, year=year, month=month))
    status["sources"]["federal_reserve_calendar"] = {
        "ok": bool(fed_calendar_rows),
        "rows": len(fed_calendar_rows),
        "error": "; ".join(fed_calendar_errors[-3:]) if fed_calendar_errors else None,
    }
    calendar_rows.extend(fed_calendar_rows)

    for source_name in ("federal_reserve", "treasury", "bls", "bea"):
        page_url = FEED_DISCOVERY_PAGES[source_name]
        source_timeout = _source_timeout_seconds(source_name, args.timeout_seconds)
        page_text, page_error = _safe_http_text(page_url, timeout=source_timeout)
        feed_urls, discovery_error = _discover_feed_urls(source_name, source_timeout, page_html=page_text)
        source_status = {"ok": False, "feeds": feed_urls, "error": discovery_error or page_error, "rows": 0}
        source_news: list[dict[str, Any]] = []
        feed_errors: list[str] = []
        if feed_urls:
            for feed_url in feed_urls:
                xml_text, err = _safe_http_text(feed_url, timeout=source_timeout)
                if not xml_text:
                    if err:
                        feed_errors.append(f"{feed_url}:{err}")
                    continue
                parsed = _parse_feed_items(xml_text, source_name)
                if parsed:
                    source_status["ok"] = True
                    source_status["rows"] += len(parsed)
                    source_news.extend(parsed)
        if not source_news and page_text:
            fallback_rows = _parse_news_links_from_html(page_text, source_name, page_url)
            if fallback_rows:
                source_status["ok"] = True
                source_status["rows"] += len(fallback_rows)
                source_status["fallback"] = "html_page_parse"
                source_news.extend(fallback_rows)
        if feed_errors:
            source_status["error"] = "; ".join(feed_errors[-3:])
        if source_news:
            news_rows.extend(source_news)
        status["sources"][source_name] = source_status

    calendar_rows.extend(_calendar_rows_from_news(news_rows))
    seen_news: set[tuple[str, str]] = set()
    deduped_news: list[dict[str, Any]] = []
    for row in news_rows:
        key = (str(row.get("headline") or "").strip(), str(row.get("published") or "").strip())
        if key in seen_news:
            continue
        seen_news.add(key)
        deduped_news.append(row)
    news_rows = deduped_news[:120]

    calendar_features = summarize_calendar_payload(calendar_rows, now_ts=now.timestamp(), max_items=600) if calendar_rows else {}
    news_features = summarize_structured_news_items(news_rows, symbol="SPY", now_ts=now.timestamp(), max_items=120) if news_rows else {}
    event_type_counts: dict[str, int] = {}
    high_impact_7d = 0
    fed_speaker_7d = 0
    for row in calendar_rows:
        event_type = str(row.get("macro_event_type") or "macro_event").strip().lower()
        event_type_counts[event_type] = int(event_type_counts.get(event_type, 0)) + 1
        raw_dt = row.get("datetime") or row.get("date")
        try:
            dt = datetime.fromisoformat(str(raw_dt).replace("Z", "+00:00")) if raw_dt else None
        except Exception:
            dt = None
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta_s = (dt.astimezone(timezone.utc) - now).total_seconds()
            if 0 <= delta_s <= 7 * 24 * 3600:
                importance_score = float(row.get("importance_score") or 0.0)
                if importance_score >= 3.0:
                    high_impact_7d += 1
                if event_type == "fed_speech":
                    fed_speaker_7d += 1

    fred_context = _load_existing_json(external_context_root / "macro_cross_asset_latest.json")
    existing_bond_reference = load_latest_external_context(PROJECT_ROOT, "bond_reference")
    bond_overlay = fred_context.get("bond_reference_overlay") if isinstance(fred_context.get("bond_reference_overlay"), dict) else {}
    bond_reference = dict(existing_bond_reference) if isinstance(existing_bond_reference, dict) else {}
    if isinstance(bond_overlay, dict):
        for key, value in bond_overlay.items():
            if isinstance(value, dict) and isinstance(bond_reference.get(key), dict):
                merged = dict(bond_reference[key])
                merged.update(value)
                bond_reference[key] = merged
            else:
                bond_reference[key] = value
    bond_reference["calendar_treasury_auction_norm"] = max(
        float(calendar_features.get("calendar_treasury_auction_norm", 0.0) or 0.0),
        float(bond_reference.get("calendar_treasury_auction_norm", 0.0) or 0.0),
    )

    payload = {
        "timestamp_utc": now_iso,
        "provider": "official_macro_context",
        "status": status,
        "derived": {
            "calendar_features": calendar_features,
            "news_features": news_features,
            "calendar_rows": calendar_rows[:200],
            "news_rows": news_rows[:120],
            "event_type_counts": event_type_counts,
            "high_impact_event_count_7d": high_impact_7d,
            "fed_speaker_event_count_7d": fed_speaker_7d,
            "bond_reference_overlay": bond_overlay,
        },
    }

    status["ok"] = bool(calendar_rows or news_rows)

    if not args.test_only:
        _write_json(external_context_root / "official_macro_context_latest.json", payload)
        _write_json(external_context_root / "bond_reference_latest.json", bond_reference)
        _write_json(health_root / "official_macro_context_sync_latest.json", status)

    if args.json:
        print(json.dumps(status, ensure_ascii=True))
    else:
        print(
            f"official_macro_context ok={status['ok']} "
            f"calendar_rows={len(calendar_rows)} news_rows={len(news_rows)}"
        )
        if not args.test_only:
            print(f"official_macro_context_latest={external_context_root / 'official_macro_context_latest.json'}")
            print(f"bond_reference_latest={external_context_root / 'bond_reference_latest.json'}")
            print(f"status_file={health_root / 'official_macro_context_sync_latest.json'}")
    return 0 if status["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect free official macro context from public government calendars and feeds.")
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
