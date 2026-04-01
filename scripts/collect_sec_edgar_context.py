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
from urllib.request import Request, urlopen

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
USER_AGENT_DEFAULT = "Daniel Kingsley dan_kingsley@aol.com"
EASTERN_TZ = ZoneInfo("America/New_York") if ZoneInfo is not None else timezone.utc

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


def _http_json(url: str, *, user_agent: str, timeout: float = 20.0) -> Any:
    req = Request(url=url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urlopen(req, timeout=max(float(timeout), 1.0)) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _safe_http_json(url: str, *, user_agent: str, timeout: float = 20.0) -> tuple[Any | None, str | None]:
    try:
        return _http_json(url, user_agent=user_agent, timeout=timeout), None
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
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


def _iter_recent_filings(submissions: dict[str, Any]) -> Iterable[dict[str, Any]]:
    recent = (((submissions.get("filings") or {}).get("recent")) or {}) if isinstance(submissions, dict) else {}
    if not isinstance(recent, dict):
        return []
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    acceptance = recent.get("acceptanceDateTime") or []
    primary_docs = recent.get("primaryDocDescription") or []
    accession = recent.get("accessionNumber") or []
    n = max(len(forms), len(filing_dates), len(acceptance), len(primary_docs), len(accession))
    rows: list[dict[str, Any]] = []
    for idx in range(n):
        form = str(forms[idx] if idx < len(forms) else "").strip().upper()
        filing_date = filing_dates[idx] if idx < len(filing_dates) else ""
        accepted = acceptance[idx] if idx < len(acceptance) else ""
        desc = str(primary_docs[idx] if idx < len(primary_docs) else "").strip()
        acc = str(accession[idx] if idx < len(accession) else "").strip()
        dt = _parse_dt(accepted) or _parse_dt(filing_date)
        rows.append(
            {
                "form": form,
                "filing_date": str(filing_date or ""),
                "accepted_at": dt.isoformat() if dt is not None else "",
                "description": desc,
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
    ownership_30d = 0
    insider_30d = 0
    latest_ts: datetime | None = None
    session_counts = {"premarket": 0, "intraday": 0, "after_hours": 0}
    recent_items: list[dict[str, Any]] = []

    for row in rows:
        dt = _parse_dt(row.get("accepted_at") or row.get("filing_date"))
        form = str(row.get("form") or "").upper()
        desc = str(row.get("description") or "")
        text = f"{form} {desc}"
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
        if dt >= cutoff_1d:
            filings_1d += 1
            if form in HIGH_IMPACT_FORMS:
                high_impact_1d += 1
        if len(recent_items) < 8:
            recent_items.append(row)

    hours_since_latest = None
    if latest_ts is not None:
        hours_since_latest = max((now - latest_ts).total_seconds() / 3600.0, 0.0)

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
        "ownership_30d": ownership_30d,
        "insider_30d": insider_30d,
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
            "sec_ownership_30d_norm": _clamp01(ownership_30d / 2.0),
            "sec_insider_30d_norm": _clamp01(insider_30d / 4.0),
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
    ownership_30d = max(_safe_float(((row.get("features") or {}).get("sec_ownership_30d_norm")), 0.0) for row in symbol_rows)
    insider_30d = max(_safe_float(((row.get("features") or {}).get("sec_insider_30d_norm")), 0.0) for row in symbol_rows)
    proximity = max(_safe_float(((row.get("features") or {}).get("sec_recent_proximity_norm")), 0.0) for row in symbol_rows)
    premarket = max(_safe_float(((row.get("features") or {}).get("news_premarket_norm")), 0.0) for row in symbol_rows)
    intraday = max(_safe_float(((row.get("features") or {}).get("news_intraday_norm")), 0.0) for row in symbol_rows)
    after_hours = max(_safe_float(((row.get("features") or {}).get("news_after_hours_norm")), 0.0) for row in symbol_rows)

    coverage = _clamp01(recent_symbols / max(min(request_count, 10), 1))
    source_quality = 0.96 if recent_symbols > 0 else 0.0

    news_features = {
        "news_source_quality_norm": source_quality,
        "news_entity_relevance_norm": _clamp01(0.25 + (0.70 * coverage)),
        "news_topic_earnings_norm": earnings_7d,
        "news_topic_guidance_norm": guidance_7d,
        "news_topic_regulatory_norm": max(regulatory_7d, ownership_30d, insider_30d),
        "news_novelty_norm": _clamp01(coverage + 0.15),
        "news_duplicate_cluster_norm": 0.0,
        "news_premarket_norm": premarket,
        "news_intraday_norm": intraday,
        "news_after_hours_norm": after_hours,
        "news_recent_impact": _clamp01(max(regulatory_7d, earnings_7d, guidance_7d, proximity)),
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
        "sec_ownership_30d_norm": ownership_30d,
        "sec_insider_30d_norm": insider_30d,
    }
    symbol_features = {str(row.get("symbol")): dict((row.get("features") or {})) for row in symbol_rows}
    return {
        "news_features": news_features,
        "calendar_features": calendar_features,
        "global_features": global_features,
        "symbol_features": symbol_features,
    }


def collect_sec_edgar_context(*, symbols: list[str], user_agent: str, timeout: float = 20.0, pause_seconds: float = 0.18) -> tuple[dict[str, Any], dict[str, Any]]:
    now = datetime.now(timezone.utc)
    ticker_payload, ticker_error = _safe_http_json(EDGAR_TICKERS_URL, user_agent=user_agent, timeout=timeout)
    ticker_by_symbol = _ticker_map(ticker_payload)
    symbol_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    requested = 0
    resolved = 0

    for symbol in symbols:
        requested += 1
        cik = ticker_by_symbol.get(symbol)
        if not cik:
            continue
        resolved += 1
        submissions, err = _safe_http_json(EDGAR_SUBMISSIONS_URL.format(cik=cik), user_agent=user_agent, timeout=timeout)
        if err:
            errors.append(f"{symbol}:{err}")
            time.sleep(max(float(pause_seconds), 0.0))
            continue
        rows = list(_iter_recent_filings(submissions if isinstance(submissions, dict) else {}))
        symbol_rows.append(_derive_symbol_summary(symbol, cik, rows, now))
        time.sleep(max(float(pause_seconds), 0.0))

    derived = _aggregate_features(symbol_rows, request_count=requested)
    status = {
        "timestamp_utc": now.isoformat(),
        "ok": bool(ticker_by_symbol) and (resolved > 0),
        "requested_symbols": requested,
        "resolved_symbols": resolved,
        "tracked_symbols": len(symbol_rows),
        "ticker_map_ok": bool(ticker_by_symbol),
        "ticker_map_error": ticker_error,
        "error_count": len(errors),
        "errors": errors[:20],
    }
    payload = {
        "timestamp_utc": now.isoformat(),
        "provider": "sec_edgar_context",
        "contact_user_agent": user_agent,
        "tracked_symbols": symbols,
        "status": status,
        "symbol_rows": symbol_rows,
        "derived": derived,
    }
    return payload, status


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect SEC EDGAR filing context for tracked equities.")
    parser.add_argument("--symbols", default=",".join(_default_symbols()))
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--pause-seconds", type=float, default=0.18)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    user_agent = str(os.getenv("SEC_EDGAR_USER_AGENT") or USER_AGENT_DEFAULT).strip() or USER_AGENT_DEFAULT
    payload, status = collect_sec_edgar_context(
        symbols=symbols,
        user_agent=user_agent,
        timeout=args.timeout,
        pause_seconds=args.pause_seconds,
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
