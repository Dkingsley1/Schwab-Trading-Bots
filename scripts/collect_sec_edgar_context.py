#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.collector_transport import attach_collection_confidence, fetch_json, fetch_text


EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_num}/{accession_number}/{primary_document}"
USER_AGENT_DEFAULT = "Daniel Kingsley dan_kingsley@aol.com"
EASTERN_TZ = ZoneInfo("America/New_York") if ZoneInfo is not None else timezone.utc
SOURCE_CONTRACTS = {
    "sec_edgar_tickers": {"source_confidence_norm": 0.99, "schema_confidence_norm": 0.97},
    "sec_edgar_submissions": {"source_confidence_norm": 0.99, "schema_confidence_norm": 0.95},
    "sec_edgar_archive": {"source_confidence_norm": 0.97, "schema_confidence_norm": 0.84},
}
DEFAULT_MAX_RUNTIME_SECONDS = 75.0
MIN_FETCH_TIMEOUT_SECONDS = 1.0
DEFAULT_MAX_ARCHIVE_FETCHES = 1

HIGH_IMPACT_FORMS = {
    "8-K",
    "10-K",
    "10-K/A",
    "10-Q",
    "10-Q/A",
    "20-F",
    "20-F/A",
    "6-K",
    "DEF 14A",
    "SC 13D",
    "SC 13D/A",
    "SC 13G",
    "SC 13G/A",
    "4",
}
EARNINGS_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "6-K"}
OWNERSHIP_FORMS = {"SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"}
INSIDER_FORMS = {"3", "4", "5"}
GUIDANCE_RE = re.compile(r"(?i)\b(guidance|outlook|forecast|raises|cuts)\b")
EARNINGS_RE = re.compile(r"(?i)\b(earnings|results|revenue|quarter|annual report|financial statements)\b")
REGULATORY_RE = re.compile(r"(?i)\b(investigation|lawsuit|compliance|restatement|sec|legal proceeding|material definitive)\b")
GUIDANCE_RAISE_RE = re.compile(r"(?i)\b(raise(?:s|d)? guidance|raising guidance|guidance increased|outlook improved|forecast increased)\b")
GUIDANCE_CUT_RE = re.compile(r"(?i)\b(cut(?:s|ting)? guidance|lower(?:s|ed)? outlook|forecast reduced|withdraw(?:s|n)? guidance)\b")
OFFERING_RE = re.compile(r"(?i)\b(offering|registered direct|at-the-market|ATM program|shelf registration|private placement)\b")
DILUTION_RE = re.compile(r"(?i)\b(dilution|dilutive|issue(?:d|s)? shares|common stock issuance|convertible note)\b")
MNA_RE = re.compile(r"(?i)\b(merger|acquisition|acquire|buyout|takeover|definitive agreement|strategic alternative)\b")
RESTATEMENT_RE = re.compile(r"(?i)\b(restatement|non-reliance|material weakness|revised financial statements)\b")
FINANCING_STRESS_RE = re.compile(r"(?i)\b(going concern|liquidity constraints?|covenant breach|default|waiver agreement|restructuring support)\b")
INSIDER_BUY_RE = re.compile(r"(?i)\b(purchase(?:d)?|acquir(?:e|ed)|buy(?:ing|ought)?)\b")
INSIDER_SELL_RE = re.compile(r"(?i)\b(sale|sold|sell(?:ing)?|dispos(?:e|ed|al))\b")
ESTIMATE_RAISE_RE = re.compile(r"(?i)\b(estimate(?:s)? (?:raised|increase|up)|target(?:s)? raised|analyst(?:s)? raise|consensus(?:.*)up)\b")
ESTIMATE_CUT_RE = re.compile(r"(?i)\b(estimate(?:s)? (?:cut|lowered|reduced)|target(?:s)? lowered|analyst(?:s)? cut|consensus(?:.*)down)\b")
WHISPER_BEAT_RE = re.compile(r"(?i)\b(beat(?:ing)? (?:whisper|consensus|estimate)|above whisper|stronger than expected)\b")
WHISPER_MISS_RE = re.compile(r"(?i)\b(miss(?:ed|ing)? (?:whisper|consensus|estimate)|below whisper|weaker than expected)\b")
SPLIT_RE = re.compile(r"(?i)\b(stock split|reverse split|share consolidation|split-adjusted)\b")
SPECIAL_DIVIDEND_RE = re.compile(r"(?i)\b(special dividend|one-time dividend|extra dividend)\b")
OFFERING_PRICED_RE = re.compile(r"(?i)\b(priced (?:the |a )?(?:offering|public offering|secondary offering)|offering priced)\b")
LOCKUP_RE = re.compile(r"(?i)\b(lock-up|lockup|lock up expiration|share unlock)\b")
SECONDARY_RE = re.compile(r"(?i)\b(secondary offering|follow-on offering|underwritten offering)\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _strongest_centered(values: Iterable[float], *, neutral: float = 0.5) -> float:
    values = list(values)
    if not values:
        return neutral
    return max(values, key=lambda value: abs(float(value) - neutral))


def _source_contract_name(url: str) -> str:
    text = str(url or "")
    if "company_tickers" in text:
        return "sec_edgar_tickers"
    if "/submissions/" in text:
        return "sec_edgar_submissions"
    if "/Archives/edgar/data/" in text:
        return "sec_edgar_archive"
    return "sec_edgar_submissions"


def _source_contract(source_name: str) -> dict[str, float]:
    row = SOURCE_CONTRACTS.get(str(source_name or ""), {})
    return {
        "source_confidence_norm": float(row.get("source_confidence_norm", 0.95) or 0.95),
        "schema_confidence_norm": float(row.get("schema_confidence_norm", 0.9) or 0.9),
    }


def _bounded_timeout(timeout: float, deadline: float | None) -> float:
    base = max(float(timeout), MIN_FETCH_TIMEOUT_SECONDS)
    if deadline is None:
        return base
    remaining = max(float(deadline) - time.monotonic(), 0.0)
    return max(min(base, remaining), MIN_FETCH_TIMEOUT_SECONDS)


def _has_fetch_budget(deadline: float | None) -> bool:
    return deadline is None or (float(deadline) - time.monotonic()) >= MIN_FETCH_TIMEOUT_SECONDS


def _http_json_result(url: str, *, user_agent: str, timeout: float = 20.0, retries: int = 0) -> dict[str, Any]:
    source_name = _source_contract_name(url)
    contract = _source_contract(source_name)
    return fetch_json(
        url=url,
        user_agent=user_agent,
        timeout=timeout,
        collector_key="sec_edgar_context",
        source_name=source_name,
        entity_key=url,
        project_root=PROJECT_ROOT,
        source_confidence_norm=contract["source_confidence_norm"],
        schema_confidence_norm=contract["schema_confidence_norm"],
        retries=max(int(retries), 0),
    )


def _safe_http_json(url: str, *, user_agent: str, timeout: float = 20.0) -> tuple[Any | None, str | None]:
    try:
        result = _http_json_result(url, user_agent=user_agent, timeout=timeout)
        if not bool(result.get("ok", False)):
            raise RuntimeError(str(result.get("error") or "http_json_failed"))
        return result.get("json"), None
    except (HTTPError, URLError, RuntimeError, TimeoutError, ValueError, OSError) as exc:
        return None, str(exc)


def _http_text_result(url: str, *, user_agent: str, timeout: float = 20.0, retries: int = 0) -> dict[str, Any]:
    source_name = _source_contract_name(url)
    contract = _source_contract(source_name)
    return fetch_text(
        url=url,
        user_agent=user_agent,
        timeout=timeout,
        collector_key="sec_edgar_context",
        source_name=source_name,
        entity_key=url,
        project_root=PROJECT_ROOT,
        source_confidence_norm=contract["source_confidence_norm"],
        schema_confidence_norm=contract["schema_confidence_norm"],
        retries=max(int(retries), 0),
    )


def _safe_http_text(url: str, *, user_agent: str, timeout: float = 20.0) -> tuple[str | None, str | None]:
    try:
        result = _http_text_result(url, user_agent=user_agent, timeout=timeout)
        if not bool(result.get("ok", False)):
            raise RuntimeError(str(result.get("error") or "http_text_failed"))
        return str(result.get("text") or ""), None
    except (HTTPError, URLError, RuntimeError, TimeoutError, ValueError, OSError) as exc:
        return None, str(exc)


def _normalize_symbol(raw: str) -> str:
    return str(raw or "").strip().upper().replace(".", "-")


def _parse_symbols(raw: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in str(raw or "").replace("\n", ",").split(","):
        symbol = _normalize_symbol(token)
        if not symbol:
            continue
        if any(ch in symbol for ch in ("/", "$")):
            continue
        if symbol.endswith("-USD"):
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _default_symbols() -> list[str]:
    raw = os.getenv("SEC_EDGAR_SYMBOLS", "").strip()
    if not raw:
        raw = ",".join(
            filter(
                None,
                [
                    os.getenv("SHADOW_SYMBOLS_CORE", ""),
                    os.getenv("SHADOW_SYMBOLS_VOLATILE", ""),
                    os.getenv("DIVIDEND_QUALITY_SYMBOLS", ""),
                ],
            )
        )
    symbols = _parse_symbols(raw)
    if not symbols:
        symbols = _parse_symbols("SPY,QQQ,AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,COIN,MSTR,PLTR,AMD,JPM,GS,XOM,CVX,JNJ,PG,ABBV")
    return symbols[:30]


def _ticker_map(payload: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(payload, dict):
        return out
    for row in payload.values():
        if not isinstance(row, dict):
            continue
        ticker = _normalize_symbol(row.get("ticker", ""))
        cik = str(row.get("cik_str") or "").strip()
        if ticker and cik.isdigit():
            out[ticker] = cik.zfill(10)
    return out


def _parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for candidate in (
        text.replace("Z", "+00:00"),
        text,
    ):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    try:
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _market_session(dt: datetime | None) -> str:
    if dt is None:
        return "unknown"
    local = dt.astimezone(EASTERN_TZ)
    minute = local.hour * 60 + local.minute
    if minute < 570:
        return "premarket"
    if minute <= 960:
        return "intraday"
    return "after_hours"


def _strip_filing_text(raw: str) -> str:
    text = HTML_TAG_RE.sub(" ", str(raw or ""))
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


def _filing_text_signals(text: str) -> dict[str, float]:
    cleaned = _strip_filing_text(text)
    if not cleaned:
        return {
            "guidance_raise": 0.0,
            "guidance_cut": 0.0,
            "offering": 0.0,
            "dilution": 0.0,
            "mna": 0.0,
            "restatement": 0.0,
            "financing_stress": 0.0,
            "insider_buy": 0.0,
            "insider_sell": 0.0,
            "estimate_raise": 0.0,
            "estimate_cut": 0.0,
            "whisper_beat": 0.0,
            "whisper_miss": 0.0,
            "split_hazard": 0.0,
            "special_dividend": 0.0,
            "offering_priced": 0.0,
            "lockup_secondary": 0.0,
        }
    return {
        "guidance_raise": 1.0 if GUIDANCE_RAISE_RE.search(cleaned) else 0.0,
        "guidance_cut": 1.0 if GUIDANCE_CUT_RE.search(cleaned) else 0.0,
        "offering": 1.0 if OFFERING_RE.search(cleaned) else 0.0,
        "dilution": 1.0 if DILUTION_RE.search(cleaned) else 0.0,
        "mna": 1.0 if MNA_RE.search(cleaned) else 0.0,
        "restatement": 1.0 if RESTATEMENT_RE.search(cleaned) else 0.0,
        "financing_stress": 1.0 if FINANCING_STRESS_RE.search(cleaned) else 0.0,
        "insider_buy": 1.0 if INSIDER_BUY_RE.search(cleaned) else 0.0,
        "insider_sell": 1.0 if INSIDER_SELL_RE.search(cleaned) else 0.0,
        "estimate_raise": 1.0 if ESTIMATE_RAISE_RE.search(cleaned) else 0.0,
        "estimate_cut": 1.0 if ESTIMATE_CUT_RE.search(cleaned) else 0.0,
        "whisper_beat": 1.0 if WHISPER_BEAT_RE.search(cleaned) else 0.0,
        "whisper_miss": 1.0 if WHISPER_MISS_RE.search(cleaned) else 0.0,
        "split_hazard": 1.0 if SPLIT_RE.search(cleaned) else 0.0,
        "special_dividend": 1.0 if SPECIAL_DIVIDEND_RE.search(cleaned) else 0.0,
        "offering_priced": 1.0 if OFFERING_PRICED_RE.search(cleaned) else 0.0,
        "lockup_secondary": 1.0 if (LOCKUP_RE.search(cleaned) or SECONDARY_RE.search(cleaned)) else 0.0,
    }


def _filing_archive_url(cik: str, accession_number: str, primary_document: str) -> str:
    cik_num = str(int(str(cik or "0")))
    accession = str(accession_number or "").replace("-", "").strip()
    return EDGAR_ARCHIVE_URL.format(
        cik_num=cik_num,
        accession_number=accession,
        primary_document=str(primary_document or "").strip(),
    )


def _attach_recent_filing_text_signals(
    *,
    cik: str,
    rows: list[dict[str, Any]],
    user_agent: str,
    timeout: float,
    max_fetch: int = DEFAULT_MAX_ARCHIVE_FETCHES,
    deadline: float | None = None,
) -> list[str]:
    errors: list[str] = []
    fetched = 0
    for row in rows:
        if not _has_fetch_budget(deadline):
            break
        if fetched >= max(int(max_fetch), 1):
            break
        primary_document = str(row.get("primary_document") or "").strip()
        accession_number = str(row.get("accession_number") or "").strip()
        if not primary_document or not accession_number:
            continue
        dt = _parse_dt(row.get("accepted_at") or row.get("filing_date"))
        if dt is None or dt < (datetime.now(timezone.utc) - timedelta(days=7)):
            continue
        url = _filing_archive_url(cik, accession_number, primary_document)
        fetch_result = _http_text_result(
            url,
            user_agent=user_agent,
            timeout=_bounded_timeout(timeout, deadline),
            retries=0,
        )
        if not bool(fetch_result.get("ok", False)):
            err = str(fetch_result.get("error") or "http_text_failed")
            errors.append(f"{accession_number}:{err}")
            continue
        text = str(fetch_result.get("text") or "")
        row["text_signals"] = _filing_text_signals(text)
        row["filing_url"] = url
        row["filing_fetch"] = {
            "fetched_utc": str(fetch_result.get("fetched_utc") or ""),
            "source_confidence_norm": float(fetch_result.get("source_confidence_norm") or _source_contract("sec_edgar_archive")["source_confidence_norm"]),
            "schema_confidence_norm": float(fetch_result.get("schema_confidence_norm") or _source_contract("sec_edgar_archive")["schema_confidence_norm"]),
            "freshness_norm": float(fetch_result.get("freshness_norm") or 0.0),
        }
        fetched += 1
    return errors


def _iter_recent_filings(submissions: dict[str, Any]) -> Iterable[dict[str, Any]]:
    recent = (((submissions.get("filings") or {}).get("recent")) or {}) if isinstance(submissions, dict) else {}
    if not isinstance(recent, dict):
        return []
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    acceptance = recent.get("acceptanceDateTime") or []
    primary_docs = recent.get("primaryDocDescription") or []
    primary_files = recent.get("primaryDocument") or []
    accession = recent.get("accessionNumber") or []
    n = max(len(forms), len(filing_dates), len(acceptance), len(primary_docs), len(primary_files), len(accession))
    rows: list[dict[str, Any]] = []
    for idx in range(n):
        form = str(forms[idx] if idx < len(forms) else "").strip().upper()
        filing_date = filing_dates[idx] if idx < len(filing_dates) else ""
        accepted = acceptance[idx] if idx < len(acceptance) else ""
        desc = str(primary_docs[idx] if idx < len(primary_docs) else "").strip()
        primary_file = str(primary_files[idx] if idx < len(primary_files) else "").strip()
        acc = str(accession[idx] if idx < len(accession) else "").strip()
        dt = _parse_dt(accepted) or _parse_dt(filing_date)
        rows.append(
            {
                "form": form,
                "filing_date": str(filing_date or ""),
                "accepted_at": dt.isoformat() if dt is not None else "",
                "description": desc,
                "primary_document": primary_file,
                "accession_number": acc,
                "market_session": _market_session(dt),
            }
        )
    return rows


def _derive_symbol_summary(symbol: str, cik: str, rows: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    cutoff_1d = now - timedelta(days=1)
    cutoff_7d = now - timedelta(days=7)
    cutoff_30d = now - timedelta(days=30)

    filings_1d = 0
    filings_7d = 0
    high_impact_1d = 0
    high_impact_7d = 0
    earnings_7d = 0
    guidance_7d = 0
    regulatory_7d = 0
    offering_7d = 0
    dilution_7d = 0
    mna_7d = 0
    restatement_7d = 0
    financing_stress_7d = 0
    ownership_30d = 0
    insider_30d = 0
    insider_buy_30d = 0
    insider_sell_30d = 0
    estimate_raise_30d = 0
    estimate_cut_30d = 0
    whisper_beat_30d = 0
    whisper_miss_30d = 0
    split_hazard_30d = 0
    special_dividend_30d = 0
    offering_priced_30d = 0
    lockup_secondary_30d = 0
    latest_ts: datetime | None = None
    session_counts = {"premarket": 0, "intraday": 0, "after_hours": 0}
    recent_items: list[dict[str, Any]] = []

    for row in rows:
        dt = _parse_dt(row.get("accepted_at") or row.get("filing_date"))
        form = str(row.get("form") or "").upper()
        desc = str(row.get("description") or "")
        text = f"{form} {desc}"
        text_signals = _filing_text_signals(text)
        if isinstance(row.get("text_signals"), dict):
            for key, value in row["text_signals"].items():
                text_signals[key] = max(float(text_signals.get(key, 0.0) or 0.0), _safe_float(value, 0.0))
        if dt is not None and (latest_ts is None or dt > latest_ts):
            latest_ts = dt
        session = str(row.get("market_session") or "")
        if session in session_counts:
            session_counts[session] += 1
        if dt is not None and dt >= cutoff_30d:
            if form in OWNERSHIP_FORMS:
                ownership_30d += 1
            if form in INSIDER_FORMS:
                insider_30d += 1
            if float(text_signals.get("insider_buy", 0.0) or 0.0) > 0.0:
                insider_buy_30d += 1
            if float(text_signals.get("insider_sell", 0.0) or 0.0) > 0.0:
                insider_sell_30d += 1
            if float(text_signals.get("estimate_raise", 0.0) or 0.0) > 0.0:
                estimate_raise_30d += 1
            if float(text_signals.get("estimate_cut", 0.0) or 0.0) > 0.0:
                estimate_cut_30d += 1
            if float(text_signals.get("whisper_beat", 0.0) or 0.0) > 0.0:
                whisper_beat_30d += 1
            if float(text_signals.get("whisper_miss", 0.0) or 0.0) > 0.0:
                whisper_miss_30d += 1
            if float(text_signals.get("split_hazard", 0.0) or 0.0) > 0.0:
                split_hazard_30d += 1
            if float(text_signals.get("special_dividend", 0.0) or 0.0) > 0.0:
                special_dividend_30d += 1
            if float(text_signals.get("offering_priced", 0.0) or 0.0) > 0.0:
                offering_priced_30d += 1
            if float(text_signals.get("lockup_secondary", 0.0) or 0.0) > 0.0:
                lockup_secondary_30d += 1
        if dt is None or dt < cutoff_7d:
            continue
        filings_7d += 1
        if form in HIGH_IMPACT_FORMS:
            high_impact_7d += 1
        if form in EARNINGS_FORMS or EARNINGS_RE.search(text):
            earnings_7d += 1
        if GUIDANCE_RE.search(text):
            guidance_7d += 1
        if form == "8-K" or REGULATORY_RE.search(text):
            regulatory_7d += 1
        if float(text_signals.get("offering", 0.0) or 0.0) > 0.0:
            offering_7d += 1
        if float(text_signals.get("dilution", 0.0) or 0.0) > 0.0:
            dilution_7d += 1
        if float(text_signals.get("mna", 0.0) or 0.0) > 0.0:
            mna_7d += 1
        if float(text_signals.get("restatement", 0.0) or 0.0) > 0.0:
            restatement_7d += 1
        if float(text_signals.get("financing_stress", 0.0) or 0.0) > 0.0:
            financing_stress_7d += 1
        if dt >= cutoff_1d:
            filings_1d += 1
            if form in HIGH_IMPACT_FORMS:
                high_impact_1d += 1
        if len(recent_items) < 8:
            recent_items.append(row)

    hours_since_latest = None
    if latest_ts is not None:
        hours_since_latest = max((now - latest_ts).total_seconds() / 3600.0, 0.0)
    estimate_drift_balance = (estimate_raise_30d - estimate_cut_30d) / max(estimate_raise_30d + estimate_cut_30d, 1)
    whisper_surprise_balance = (whisper_beat_30d - whisper_miss_30d) / max(whisper_beat_30d + whisper_miss_30d, 1)

    return {
        "symbol": symbol,
        "cik": cik,
        "filings_1d": filings_1d,
        "filings_7d": filings_7d,
        "high_impact_1d": high_impact_1d,
        "high_impact_7d": high_impact_7d,
        "earnings_7d": earnings_7d,
        "guidance_7d": guidance_7d,
        "regulatory_7d": regulatory_7d,
        "offering_7d": offering_7d,
        "dilution_7d": dilution_7d,
        "mna_7d": mna_7d,
        "restatement_7d": restatement_7d,
        "financing_stress_7d": financing_stress_7d,
        "ownership_30d": ownership_30d,
        "insider_30d": insider_30d,
        "insider_buy_30d": insider_buy_30d,
        "insider_sell_30d": insider_sell_30d,
        "estimate_raise_30d": estimate_raise_30d,
        "estimate_cut_30d": estimate_cut_30d,
        "whisper_beat_30d": whisper_beat_30d,
        "whisper_miss_30d": whisper_miss_30d,
        "split_hazard_30d": split_hazard_30d,
        "special_dividend_30d": special_dividend_30d,
        "offering_priced_30d": offering_priced_30d,
        "lockup_secondary_30d": lockup_secondary_30d,
        "hours_since_latest": round(hours_since_latest, 4) if hours_since_latest is not None else None,
        "latest_accepted_at": latest_ts.isoformat() if latest_ts is not None else None,
        "market_sessions": session_counts,
        "recent_filings": recent_items,
        "features": {
            "sec_filing_count_7d_norm": _clamp01(filings_7d / 4.0),
            "sec_high_impact_7d_norm": _clamp01(high_impact_7d / 3.0),
            "sec_earnings_7d_norm": _clamp01(earnings_7d / 2.0),
            "sec_guidance_7d_norm": _clamp01(guidance_7d / 2.0),
            "sec_regulatory_7d_norm": _clamp01(regulatory_7d / 3.0),
            "sec_offering_7d_norm": _clamp01(offering_7d / 2.0),
            "sec_dilution_7d_norm": _clamp01(dilution_7d / 2.0),
            "sec_mna_7d_norm": _clamp01(mna_7d / 1.5),
            "sec_restatement_7d_norm": _clamp01(restatement_7d / 1.5),
            "sec_financing_stress_7d_norm": _clamp01(financing_stress_7d / 1.5),
            "sec_ownership_30d_norm": _clamp01(ownership_30d / 2.0),
            "sec_insider_30d_norm": _clamp01(insider_30d / 4.0),
            "sec_insider_buy_30d_norm": _clamp01(insider_buy_30d / 3.0),
            "sec_insider_sell_30d_norm": _clamp01(insider_sell_30d / 3.0),
            "sec_estimate_revision_drift_norm": _clamp01(0.5 + (0.5 * estimate_drift_balance)),
            "sec_earnings_whisper_surprise_norm": _clamp01(0.5 + (0.5 * whisper_surprise_balance)),
            "sec_split_hazard_30d_norm": _clamp01(split_hazard_30d / 1.5),
            "sec_special_dividend_30d_norm": _clamp01(special_dividend_30d / 1.5),
            "sec_offering_priced_30d_norm": _clamp01(offering_priced_30d / 2.0),
            "sec_lockup_secondary_30d_norm": _clamp01(lockup_secondary_30d / 2.0),
            "sec_recent_proximity_norm": _clamp01(1.0 - ((hours_since_latest or 999.0) / 72.0)),
            "news_premarket_norm": _clamp01(session_counts["premarket"] / 3.0),
            "news_intraday_norm": _clamp01(session_counts["intraday"] / 3.0),
            "news_after_hours_norm": _clamp01(session_counts["after_hours"] / 3.0),
        },
    }


def _aggregate_features(symbol_rows: list[dict[str, Any]], request_count: int) -> dict[str, Any]:
    if not symbol_rows:
        return {
            "news_features": {},
            "calendar_features": {},
            "global_features": {},
            "symbol_features": {},
        }

    recent_symbols = sum(1 for row in symbol_rows if int(row.get("filings_7d") or 0) > 0)
    high_impact_1d = sum(int(row.get("high_impact_1d") or 0) for row in symbol_rows)
    filings_1d = sum(int(row.get("filings_1d") or 0) for row in symbol_rows)
    earnings_7d = max(_safe_float(((row.get("features") or {}).get("sec_earnings_7d_norm")), 0.0) for row in symbol_rows)
    guidance_7d = max(_safe_float(((row.get("features") or {}).get("sec_guidance_7d_norm")), 0.0) for row in symbol_rows)
    regulatory_7d = max(_safe_float(((row.get("features") or {}).get("sec_regulatory_7d_norm")), 0.0) for row in symbol_rows)
    offering_7d = max(_safe_float(((row.get("features") or {}).get("sec_offering_7d_norm")), 0.0) for row in symbol_rows)
    dilution_7d = max(_safe_float(((row.get("features") or {}).get("sec_dilution_7d_norm")), 0.0) for row in symbol_rows)
    mna_7d = max(_safe_float(((row.get("features") or {}).get("sec_mna_7d_norm")), 0.0) for row in symbol_rows)
    restatement_7d = max(_safe_float(((row.get("features") or {}).get("sec_restatement_7d_norm")), 0.0) for row in symbol_rows)
    financing_stress_7d = max(_safe_float(((row.get("features") or {}).get("sec_financing_stress_7d_norm")), 0.0) for row in symbol_rows)
    ownership_30d = max(_safe_float(((row.get("features") or {}).get("sec_ownership_30d_norm")), 0.0) for row in symbol_rows)
    insider_30d = max(_safe_float(((row.get("features") or {}).get("sec_insider_30d_norm")), 0.0) for row in symbol_rows)
    insider_buy_30d = max(_safe_float(((row.get("features") or {}).get("sec_insider_buy_30d_norm")), 0.0) for row in symbol_rows)
    insider_sell_30d = max(_safe_float(((row.get("features") or {}).get("sec_insider_sell_30d_norm")), 0.0) for row in symbol_rows)
    estimate_drift = _strongest_centered(
        _safe_float(((row.get("features") or {}).get("sec_estimate_revision_drift_norm")), 0.5) for row in symbol_rows
    )
    whisper_surprise = _strongest_centered(
        _safe_float(((row.get("features") or {}).get("sec_earnings_whisper_surprise_norm")), 0.5) for row in symbol_rows
    )
    split_hazard = max(_safe_float(((row.get("features") or {}).get("sec_split_hazard_30d_norm")), 0.0) for row in symbol_rows)
    special_dividend = max(_safe_float(((row.get("features") or {}).get("sec_special_dividend_30d_norm")), 0.0) for row in symbol_rows)
    offering_priced = max(_safe_float(((row.get("features") or {}).get("sec_offering_priced_30d_norm")), 0.0) for row in symbol_rows)
    lockup_secondary = max(_safe_float(((row.get("features") or {}).get("sec_lockup_secondary_30d_norm")), 0.0) for row in symbol_rows)
    proximity = max(_safe_float(((row.get("features") or {}).get("sec_recent_proximity_norm")), 0.0) for row in symbol_rows)
    premarket = max(_safe_float(((row.get("features") or {}).get("news_premarket_norm")), 0.0) for row in symbol_rows)
    intraday = max(_safe_float(((row.get("features") or {}).get("news_intraday_norm")), 0.0) for row in symbol_rows)
    after_hours = max(_safe_float(((row.get("features") or {}).get("news_after_hours_norm")), 0.0) for row in symbol_rows)

    coverage = _clamp01(recent_symbols / max(min(request_count, 10), 1))
    source_quality = 0.96 if recent_symbols > 0 else 0.0

    news_features = {
        "news_source_quality_norm": source_quality,
        "news_entity_relevance_norm": _clamp01(0.25 + (0.70 * coverage)),
        "news_topic_earnings_norm": max(earnings_7d, abs(whisper_surprise - 0.5) * 2.0),
        "news_topic_guidance_norm": max(guidance_7d, offering_7d, dilution_7d, abs(estimate_drift - 0.5) * 2.0),
        "news_topic_mna_norm": mna_7d,
        "news_topic_regulatory_norm": max(regulatory_7d, restatement_7d, financing_stress_7d, ownership_30d, insider_30d, split_hazard, lockup_secondary),
        "news_novelty_norm": _clamp01(coverage + 0.15),
        "news_duplicate_cluster_norm": 0.0,
        "news_premarket_norm": premarket,
        "news_intraday_norm": intraday,
        "news_after_hours_norm": after_hours,
        "news_recent_impact": _clamp01(max(regulatory_7d, earnings_7d, guidance_7d, offering_7d, dilution_7d, mna_7d, financing_stress_7d, proximity, split_hazard, special_dividend, offering_priced, lockup_secondary)),
    }
    calendar_features = {
        "calendar_feed_available": 1.0,
        "calendar_events_24h_norm": _clamp01(filings_1d / 8.0),
        "calendar_high_impact_24h_norm": _clamp01(high_impact_1d / 5.0),
        "calendar_event_proximity_norm": proximity,
        "calendar_next_event_norm": proximity,
    }
    global_features = {
        "sec_recent_symbols_norm": coverage,
        "sec_recent_filings_1d_norm": _clamp01(filings_1d / 8.0),
        "sec_recent_high_impact_1d_norm": _clamp01(high_impact_1d / 5.0),
        "sec_offering_7d_norm": offering_7d,
        "sec_dilution_7d_norm": dilution_7d,
        "sec_mna_7d_norm": mna_7d,
        "sec_restatement_7d_norm": restatement_7d,
        "sec_financing_stress_7d_norm": financing_stress_7d,
        "sec_ownership_30d_norm": ownership_30d,
        "sec_insider_30d_norm": insider_30d,
        "sec_insider_buy_30d_norm": insider_buy_30d,
        "sec_insider_sell_30d_norm": insider_sell_30d,
        "sec_estimate_revision_drift_norm": estimate_drift,
        "sec_earnings_whisper_surprise_norm": whisper_surprise,
        "sec_split_hazard_30d_norm": split_hazard,
        "sec_special_dividend_30d_norm": special_dividend,
        "sec_offering_priced_30d_norm": offering_priced,
        "sec_lockup_secondary_30d_norm": lockup_secondary,
    }
    symbol_features = {str(row.get("symbol")): dict((row.get("features") or {})) for row in symbol_rows}
    return {
        "news_features": news_features,
        "calendar_features": calendar_features,
        "global_features": global_features,
        "symbol_features": symbol_features,
    }


def collect_sec_edgar_context(
    *,
    symbols: list[str],
    user_agent: str,
    timeout: float = 20.0,
    pause_seconds: float = 0.18,
    max_runtime_seconds: float = DEFAULT_MAX_RUNTIME_SECONDS,
    max_archive_fetches: int = DEFAULT_MAX_ARCHIVE_FETCHES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    runtime_budget = max(float(max_runtime_seconds), 0.0)
    deadline = started_monotonic + runtime_budget if runtime_budget > 0.0 else None
    ticker_result = _http_json_result(
        EDGAR_TICKERS_URL,
        user_agent=user_agent,
        timeout=_bounded_timeout(timeout, deadline),
        retries=0,
    )
    ticker_payload = ticker_result.get("json") if bool(ticker_result.get("ok", False)) else None
    ticker_error = None if bool(ticker_result.get("ok", False)) else str(ticker_result.get("error") or "http_json_failed")
    ticker_by_symbol = _ticker_map(ticker_payload)
    symbol_rows: list[dict[str, Any]] = []
    fatal_errors: list[str] = []
    warnings: list[str] = []
    requested = 0
    resolved = 0
    deferred_symbols = 0
    deadline_exceeded = False
    filing_text_fetches = 0
    submissions_ok = 0
    archive_fetch_ok = 0

    for symbol in symbols:
        if not _has_fetch_budget(deadline):
            deadline_exceeded = True
            deferred_symbols = len(symbols) - requested
            break
        requested += 1
        cik = ticker_by_symbol.get(symbol)
        if not cik:
            continue
        resolved += 1
        submissions_result = _http_json_result(
            EDGAR_SUBMISSIONS_URL.format(cik=cik),
            user_agent=user_agent,
            timeout=_bounded_timeout(timeout, deadline),
            retries=0,
        )
        if not bool(submissions_result.get("ok", False)):
            err = str(submissions_result.get("error") or "http_json_failed")
            warnings.append(f"{symbol}:{err}")
            if _has_fetch_budget(deadline):
                time.sleep(min(max(float(pause_seconds), 0.0), max(float(deadline or 0.0) - time.monotonic(), 0.0) if deadline is not None else max(float(pause_seconds), 0.0)))
            continue
        submissions = submissions_result.get("json")
        submissions_ok += 1
        rows = list(_iter_recent_filings(submissions if isinstance(submissions, dict) else {}))
        text_errors = _attach_recent_filing_text_signals(
            cik=cik,
            rows=rows,
            user_agent=user_agent,
            timeout=timeout,
            max_fetch=max_archive_fetches,
            deadline=deadline,
        )
        filing_text_fetches += sum(1 for row in rows if isinstance(row.get("text_signals"), dict))
        archive_fetch_ok += sum(1 for row in rows if isinstance(row.get("filing_fetch"), dict))
        warnings.extend(f"{symbol}:{item}" for item in text_errors[:4])
        summary = _derive_symbol_summary(symbol, cik, rows, now)
        symbol_rows.append(
            attach_collection_confidence(
                summary,
                source_confidence_norm=_source_contract("sec_edgar_submissions")["source_confidence_norm"],
                schema_confidence_norm=_source_contract("sec_edgar_submissions")["schema_confidence_norm"],
                freshness_norm=float(((summary.get("features") or {}).get("sec_recent_proximity_norm", 0.0) or 0.0)),
                fetched_utc=str(submissions_result.get("fetched_utc") or ""),
            )
        )
        if _has_fetch_budget(deadline):
            time.sleep(min(max(float(pause_seconds), 0.0), max(float(deadline or 0.0) - time.monotonic(), 0.0) if deadline is not None else max(float(pause_seconds), 0.0)))

    derived = _aggregate_features(symbol_rows, request_count=requested)
    source_contracts = {
        "sec_edgar_tickers": {
            "ok": bool(ticker_by_symbol),
            "source_confidence_norm": float(ticker_result.get("source_confidence_norm") or _source_contract("sec_edgar_tickers")["source_confidence_norm"]),
            "schema_confidence_norm": float(ticker_result.get("schema_confidence_norm") or _source_contract("sec_edgar_tickers")["schema_confidence_norm"]),
            "freshness_norm": float(ticker_result.get("freshness_norm") or 0.0),
        },
        "sec_edgar_submissions": {
            "ok": submissions_ok > 0,
            "successful_fetches": submissions_ok,
            "source_confidence_norm": _source_contract("sec_edgar_submissions")["source_confidence_norm"],
            "schema_confidence_norm": _source_contract("sec_edgar_submissions")["schema_confidence_norm"],
            "freshness_norm": max((float(row.get("freshness_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0),
        },
        "sec_edgar_archive": {
            "ok": archive_fetch_ok > 0,
            "successful_fetches": archive_fetch_ok,
            "source_confidence_norm": _source_contract("sec_edgar_archive")["source_confidence_norm"],
            "schema_confidence_norm": _source_contract("sec_edgar_archive")["schema_confidence_norm"],
            "freshness_norm": max(
                (
                    float(((filing.get("filing_fetch") or {}).get("freshness_norm", 0.0) or 0.0))
                    for row in symbol_rows
                    for filing in (row.get("recent_filings") or [])
                    if isinstance(filing, dict)
                ),
                default=0.0,
            ),
        },
    }
    status = {
        "timestamp_utc": now.isoformat(),
        "ok": bool(ticker_by_symbol) and (len(symbol_rows) > 0),
        "requested_symbols": requested,
        "configured_symbols": len(symbols),
        "resolved_symbols": resolved,
        "tracked_symbols": len(symbol_rows),
        "deferred_symbols": max(deferred_symbols, 0),
        "deadline_exceeded": bool(deadline_exceeded),
        "max_runtime_seconds": round(runtime_budget, 3),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
        "filing_text_fetches": filing_text_fetches,
        "ticker_map_ok": bool(ticker_by_symbol),
        "ticker_map_error": ticker_error,
        "error_count": len(fatal_errors),
        "warning_count": len(warnings),
        "errors": fatal_errors[:20],
        "warnings": warnings[:20],
        "source_contracts": source_contracts,
    }
    payload = {
        "timestamp_utc": now.isoformat(),
        "provider": "sec_edgar_context",
        "contact_user_agent": user_agent,
        "tracked_symbols": symbols,
        "status": status,
        "collection_contract": {
            "source_contracts": source_contracts,
            "provider_confidence_norm": round(
                sum(float((row or {}).get("source_confidence_norm", 0.0) or 0.0) for row in source_contracts.values())
                / max(len(source_contracts), 1),
                6,
            ),
        },
        "symbol_rows": symbol_rows,
        "derived": derived,
    }
    return payload, status


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect SEC EDGAR filing context for tracked equities.")
    parser.add_argument("--symbols", default=",".join(_default_symbols()))
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--pause-seconds", type=float, default=0.18)
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=float(os.getenv("SEC_EDGAR_MAX_RUNTIME_SECONDS", str(DEFAULT_MAX_RUNTIME_SECONDS)) or DEFAULT_MAX_RUNTIME_SECONDS),
    )
    parser.add_argument(
        "--max-archive-fetches",
        type=int,
        default=int(os.getenv("SEC_EDGAR_MAX_ARCHIVE_FETCHES", str(DEFAULT_MAX_ARCHIVE_FETCHES)) or DEFAULT_MAX_ARCHIVE_FETCHES),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    user_agent = str(os.getenv("SEC_EDGAR_USER_AGENT") or USER_AGENT_DEFAULT).strip() or USER_AGENT_DEFAULT
    payload, status = collect_sec_edgar_context(
        symbols=symbols,
        user_agent=user_agent,
        timeout=args.timeout,
        pause_seconds=args.pause_seconds,
        max_runtime_seconds=args.max_runtime_seconds,
        max_archive_fetches=args.max_archive_fetches,
    )

    external_context_root = PROJECT_ROOT / "exports" / "external_context"
    health_root = PROJECT_ROOT / "governance" / "health"
    _write_json(external_context_root / "sec_edgar_latest.json", payload)
    _write_json(health_root / "sec_edgar_sync_latest.json", status)

    if args.json:
        print(json.dumps(status, ensure_ascii=True))
    else:
        print(
            "sec_edgar_context ok={ok} requested={req} resolved={res} tracked={tracked}".format(
                ok=status["ok"],
                req=status["requested_symbols"],
                res=status["resolved_symbols"],
                tracked=status["tracked_symbols"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
