#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


USER_AGENT_DEFAULT = "Daniel Kingsley dan_kingsley@aol.com"
CFTC_INDEX_URLS = [
    "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
    "https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalViewable/index.htm",
]
NYFED_SOFR_RATE_URLS = [
    "https://markets.newyorkfed.org/api/rates/secured/sofr/last/1.json",
    "https://markets.newyorkfed.org/api/rates/secured/all/latest.json",
]
NYFED_SOFR_AVERAGES_URLS = [
    "https://markets.newyorkfed.org/api/rates/secured/sofr/averages/last/1.json",
    "https://markets.newyorkfed.org/api/rates/secured/all/latest.json",
]
CBOE_MARKET_STATS_URL = "https://www.cboe.com/us/options/market_statistics/market/"
CBOE_VIX_URL = "https://www.cboe.com/tradable-products/vix"
NASDAQ_THRESHOLD_URL = "https://www.nasdaqtrader.com/trader.aspx?id=RegSHOThreshold"
SEC_FTD_URL = "https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data"

_CFTC_HEADER_RE = re.compile(r"^(.+?)\s+\(CONTRACTS OF .+\)$")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?|[.]")
_COT_DATE_RE = re.compile(r"cot(\d{6})", re.IGNORECASE)
_CBOE_RATIO_ROW_RE = re.compile(r"^\d{1,2}:\d{2}\s+[AP]M\s+[\d,]+\s+[\d,]+\s+[\d,]+\s+([0-9]+(?:\.[0-9]+)?)$")


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
        if self._href is not None:
            text = str(data or "").strip()
            if text:
                self._parts.append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._href is None:
            return
        self.links.append((self._href, " ".join(self._parts).strip()))
        self._href = None
        self._parts = []


class _SectionTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sections: dict[str, list[list[str]]] = defaultdict(list)
        self._heading: str = ""
        self._heading_buf: list[str] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_name = tag.lower()
        if tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_buf = []
        elif tag_name == "tr":
            self._row = []
        elif tag_name in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        text = str(data or "")
        if self._heading_buf is not None:
            self._heading_buf.append(text)
        elif self._cell is not None:
            self._cell.append(text)

    def handle_endtag(self, tag: str) -> None:
        tag_name = tag.lower()
        if tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            heading = " ".join(part.strip() for part in (self._heading_buf or []) if str(part or "").strip()).strip()
            if heading:
                self._heading = heading
            self._heading_buf = None
        elif tag_name in {"td", "th"} and self._cell is not None and self._row is not None:
            cell = " ".join(part.strip() for part in self._cell if str(part or "").strip()).strip()
            self._row.append(cell)
            self._cell = None
        elif tag_name == "tr" and self._row is not None:
            row = [cell for cell in self._row if cell]
            if row:
                self.sections[self._heading].append(row)
            self._row = None


class _TextLineParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def handle_data(self, data: str) -> None:
        text = str(data or "").strip()
        if text:
            self.lines.append(text)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


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


def _normalize_symbol(raw: str) -> str:
    return str(raw or "").strip().upper().replace(".", "-")


def _parse_symbols(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in str(raw or "").replace("\n", ",").split(","):
        symbol = _normalize_symbol(token)
        if not symbol or symbol in seen or "/" in symbol or symbol.endswith("-USD"):
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _default_symbols() -> list[str]:
    raw = os.getenv("EXTENDED_QUANT_SYMBOLS", "").strip()
    if not raw:
        raw = ",".join(
            filter(
                None,
                [
                    os.getenv("SHADOW_SYMBOLS_CORE", ""),
                    os.getenv("SHADOW_SYMBOLS_VOLATILE", ""),
                    os.getenv("SHADOW_SYMBOLS_DEFENSIVE", ""),
                    os.getenv("BOND_SYMBOLS", ""),
                    os.getenv("BOND_CONTEXT_SYMBOLS", ""),
                ],
            )
        )
    symbols = _parse_symbols(raw)
    if not symbols:
        symbols = _parse_symbols(
            "SPY,QQQ,DIA,IWM,RSP,AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,PLTR,COIN,MSTR,JPM,GS,TLT,IEF,SHY,TIP,LQD,HYG"
        )
    return symbols[:80]


def _http_text(url: str, *, user_agent: str, timeout: float = 20.0) -> str:
    req = Request(
        url=url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/json,text/plain,*/*",
        },
    )
    with urlopen(req, timeout=max(float(timeout), 1.0)) as resp:
        return resp.read().decode("utf-8", "replace")


def _safe_http_text(url: str, *, user_agent: str, timeout: float = 20.0) -> tuple[str | None, str | None]:
    try:
        return _http_text(url, user_agent=user_agent, timeout=timeout), None
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        return None, str(exc)


def _safe_http_json(url: str, *, user_agent: str, timeout: float = 20.0) -> tuple[Any | None, str | None]:
    text, err = _safe_http_text(url, user_agent=user_agent, timeout=timeout)
    if err or text is None:
        return None, err
    try:
        return json.loads(text), None
    except Exception as exc:
        return None, str(exc)


def _safe_http_bytes(url: str, *, user_agent: str, timeout: float = 20.0) -> tuple[bytes | None, str | None]:
    try:
        req = Request(
            url=url,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/zip,application/octet-stream,*/*",
            },
        )
        with urlopen(req, timeout=max(float(timeout), 1.0)) as resp:
            return resp.read(), None
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        return None, str(exc)


def _parse_mmddyy_token(raw: str) -> datetime | None:
    try:
        return datetime.strptime(str(raw or "").strip(), "%m%d%y").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _discover_cftc_financial_report_url(user_agent: str, timeout: float) -> tuple[str | None, dict[str, Any]]:
    status: dict[str, Any] = {"ok": False, "error": None}
    latest_viewable_url: str | None = None
    latest_viewable_dt: datetime | None = None

    for index_url in CFTC_INDEX_URLS:
        html, err = _safe_http_text(index_url, user_agent=user_agent, timeout=timeout)
        if err or not html:
            status["error"] = err
            continue
        parser = _LinkParser()
        parser.feed(html)
        for href, _text in parser.links:
            match = _COT_DATE_RE.search(href)
            if not match:
                continue
            dt = _parse_mmddyy_token(match.group(1))
            if dt is None:
                continue
            if latest_viewable_dt is None or dt > latest_viewable_dt:
                latest_viewable_dt = dt
                latest_viewable_url = urljoin(index_url, href)
        if latest_viewable_url:
            break

    if not latest_viewable_url:
        status["error"] = status.get("error") or "no_cftc_viewable_link_found"
        return None, status

    viewable_html, err = _safe_http_text(latest_viewable_url, user_agent=user_agent, timeout=timeout)
    if err or not viewable_html:
        status["error"] = err or "cftc_viewable_fetch_failed"
        return None, status

    parser = _LinkParser()
    parser.feed(viewable_html)
    report_url = None
    report_dt: datetime | None = None
    for href, _text in parser.links:
        lower_href = href.lower()
        if "financial_lf" not in lower_href:
            continue
        match = re.search(r"financial_lf(\d{6})\.htm", lower_href)
        if not match:
            continue
        dt = _parse_mmddyy_token(match.group(1))
        if dt is None:
            continue
        if report_dt is None or dt > report_dt:
            report_dt = dt
            report_url = urljoin(latest_viewable_url, href)

    if not report_url:
        status["error"] = "no_cftc_financial_report_link_found"
        return None, status

    status.update(
        {
            "ok": True,
            "viewable_url": latest_viewable_url,
            "report_url": report_url,
            "report_date_utc": report_dt.isoformat() if report_dt is not None else None,
        }
    )
    return report_url, status


def _parse_number_line(line: str) -> list[float]:
    values: list[float] = []
    for token in _NUMBER_RE.findall(str(line or "")):
        if token == ".":
            values.append(0.0)
            continue
        values.append(_safe_float(token.replace(",", ""), 0.0))
    return values


def _parse_cftc_financial_long_report(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = [line.rstrip() for line in str(text or "").splitlines()]
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        match = _CFTC_HEADER_RE.match(line)
        if not match or line.upper().startswith("TRADERS IN FINANCIAL FUTURES"):
            continue
        name = match.group(1).strip()
        open_interest = 0.0
        positions: list[float] = []
        percents: list[float] = []
        for inner in range(idx + 1, min(idx + 14, len(lines))):
            current = lines[inner].strip()
            if "Open Interest is" in current:
                open_interest = _safe_float(current.split("Open Interest is", 1)[-1].replace(",", "").strip(), 0.0)
            elif current == "Positions" and inner + 1 < len(lines):
                positions = _parse_number_line(lines[inner + 1].strip())
            elif current.startswith("Percent of Open Interest Represented by Each Category of Trader") and inner + 1 < len(lines):
                percents = _parse_number_line(lines[inner + 1].strip())
                break
        if len(positions) < 14 or len(percents) < 14:
            continue
        rows.append(
            {
                "name": name,
                "open_interest": open_interest,
                "positions": {
                    "dealer_long": positions[0],
                    "dealer_short": positions[1],
                    "asset_manager_long": positions[3],
                    "asset_manager_short": positions[4],
                    "leveraged_long": positions[6],
                    "leveraged_short": positions[7],
                    "other_long": positions[9],
                    "other_short": positions[10],
                    "nonreportable_long": positions[12],
                    "nonreportable_short": positions[13],
                },
                "percent_of_oi": {
                    "dealer_long": percents[0],
                    "dealer_short": percents[1],
                    "asset_manager_long": percents[3],
                    "asset_manager_short": percents[4],
                    "leveraged_long": percents[6],
                    "leveraged_short": percents[7],
                    "other_long": percents[9],
                    "other_short": percents[10],
                    "nonreportable_long": percents[12],
                    "nonreportable_short": percents[13],
                },
            }
        )
    return rows


def _mean(values: Iterable[float]) -> float:
    items = [float(v) for v in values if math.isfinite(float(v))]
    if not items:
        return 0.0
    return sum(items) / len(items)


def _cot_net_pct(row: Mapping[str, Any], prefix: str) -> float:
    pct = row.get("percent_of_oi") if isinstance(row.get("percent_of_oi"), Mapping) else {}
    long_key = f"{prefix}_long"
    short_key = f"{prefix}_short"
    return _safe_float(pct.get(long_key), 0.0) - _safe_float(pct.get(short_key), 0.0)


def _derive_cftc_features(contract_rows: list[dict[str, Any]]) -> dict[str, Any]:
    equity_rows = [
        row
        for row in contract_rows
        if any(token in str(row.get("name") or "").upper() for token in ("E-MINI S&P 500", "NASDAQ-100 STOCK INDEX"))
    ]
    bond_rows = [
        row
        for row in contract_rows
        if "TREASURY" in str(row.get("name") or "").upper()
        and any(token in str(row.get("name") or "").upper() for token in ("2-YEAR", "5-YEAR", "10-YEAR", "ULTRA 10-YEAR", "TREASURY BONDS"))
    ]
    usd_rows = [
        row
        for row in contract_rows
        if "U.S. DOLLAR INDEX" in str(row.get("name") or "").upper()
    ]

    equity_asset_net = _mean(_cot_net_pct(row, "asset_manager") for row in equity_rows)
    equity_lev_net = _mean(_cot_net_pct(row, "leveraged") for row in equity_rows)
    bond_asset_net = _mean(_cot_net_pct(row, "asset_manager") for row in bond_rows)
    bond_lev_net = _mean(_cot_net_pct(row, "leveraged") for row in bond_rows)
    usd_lev_net = _mean(_cot_net_pct(row, "leveraged") for row in usd_rows)

    equity_risk_on = _clamp01(0.5 + (((0.65 * equity_asset_net) + (0.35 * equity_lev_net)) / 40.0))
    equity_crowding = _clamp01(abs(equity_lev_net) / 35.0)
    bond_risk_off = _clamp01(0.5 + (((0.60 * bond_asset_net) + (0.40 * bond_lev_net)) / 40.0))
    usd_bullish = _clamp01(0.5 + (usd_lev_net / 30.0))
    positioning_stress = _clamp01(max(equity_crowding, abs(bond_lev_net) / 35.0, abs(usd_lev_net) / 30.0))
    risk_on = _clamp01((0.50 * equity_risk_on) + (0.25 * (1.0 - bond_risk_off)) + (0.25 * (1.0 - usd_bullish)))

    return {
        "cot_equity_risk_on_norm": equity_risk_on,
        "cot_equity_crowding_norm": equity_crowding,
        "cot_bond_risk_off_norm": bond_risk_off,
        "cot_usd_bullish_norm": usd_bullish,
        "cot_macro_positioning_stress_norm": positioning_stress,
        "cot_risk_on_norm": risk_on,
    }


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for nested in value.values():
            yield from _walk_mappings(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_mappings(item)


def _lookup_first(mapping: Mapping[str, Any], aliases: Iterable[str]) -> float | None:
    lowered = {str(key).lower(): val for key, val in mapping.items()}
    for alias in aliases:
        if alias.lower() in lowered:
            value = _safe_float(lowered[alias.lower()], float("nan"))
            if math.isfinite(value):
                return value
    return None


def _extract_sofr_snapshot(rate_payload: Any, averages_payload: Any) -> dict[str, float]:
    candidates = list(_walk_mappings(rate_payload)) + list(_walk_mappings(averages_payload))
    rate = None
    avg30 = None
    avg90 = None
    avg180 = None
    index_value = None
    for row in candidates:
        name = str(row.get("type") or row.get("name") or row.get("series") or "").lower()
        if name and "sofr" not in name and "secured overnight financing rate" not in name:
            continue
        rate = rate if rate is not None else _lookup_first(row, ("percentRate", "rate", "value", "percent_rate"))
        avg30 = avg30 if avg30 is not None else _lookup_first(row, ("average30Day", "avg30Day", "sofr30DayAverage", "thirtyDayAverage"))
        avg90 = avg90 if avg90 is not None else _lookup_first(row, ("average90Day", "avg90Day", "sofr90DayAverage", "ninetyDayAverage"))
        avg180 = avg180 if avg180 is not None else _lookup_first(row, ("average180Day", "avg180Day", "sofr180DayAverage", "oneHundredEightyDayAverage"))
        index_value = index_value if index_value is not None else _lookup_first(row, ("index", "sofrIndex", "indexValue"))
    return {
        "rate": rate if rate is not None else 0.0,
        "avg30": avg30 if avg30 is not None else 0.0,
        "avg90": avg90 if avg90 is not None else 0.0,
        "avg180": avg180 if avg180 is not None else 0.0,
        "index": index_value if index_value is not None else 0.0,
    }


def _derive_sofr_features(snapshot: Mapping[str, float]) -> tuple[dict[str, float], dict[str, Any]]:
    rate = _safe_float(snapshot.get("rate"), 0.0)
    avg30 = _safe_float(snapshot.get("avg30"), 0.0)
    avg90 = _safe_float(snapshot.get("avg90"), 0.0)
    avg180 = _safe_float(snapshot.get("avg180"), 0.0)
    index_value = _safe_float(snapshot.get("index"), 0.0)
    term_gap = max(rate - max(avg90, 1e-8), 0.0) if avg90 > 0.0 else 0.0
    funding_stress = _clamp01(max(rate - avg30, rate - avg90, 0.0) / 0.75) if rate > 0.0 else 0.0
    global_features = {
        "sofr_level_norm": _clamp01((rate - 2.0) / 4.0) if rate > 0.0 else 0.0,
        "sofr_30d_avg_norm": _clamp01((avg30 - 2.0) / 4.0) if avg30 > 0.0 else 0.0,
        "sofr_90d_avg_norm": _clamp01((avg90 - 2.0) / 4.0) if avg90 > 0.0 else 0.0,
        "sofr_180d_avg_norm": _clamp01((avg180 - 2.0) / 4.0) if avg180 > 0.0 else 0.0,
        "sofr_term_pressure_norm": _clamp01(0.5 + term_gap),
        "sofr_funding_stress_norm": funding_stress,
        "sofr_index_norm": _clamp01(index_value / 2.0) if index_value > 0.0 else 0.0,
    }
    bond_overlay = {
        "reference_sofr": round(rate, 6) if rate > 0.0 else 0.0,
        "funding_stress_norm": funding_stress,
    }
    return global_features, bond_overlay


def _parse_cboe_market_stats_html(html: str) -> dict[str, float]:
    parser = _SectionTableParser()
    parser.feed(str(html or ""))
    out = {"total_ratio": 0.0, "index_ratio": 0.0, "equity_ratio": 0.0}
    for heading, rows in parser.sections.items():
        key = None
        heading_l = str(heading or "").strip().lower()
        if heading_l.startswith("total"):
            key = "total_ratio"
        elif "index options" in heading_l:
            key = "index_ratio"
        elif "equity options" in heading_l:
            key = "equity_ratio"
        if key is None:
            continue
        for row in rows:
            joined = " ".join(str(part or "").strip() for part in row if str(part or "").strip())
            match = _CBOE_RATIO_ROW_RE.match(joined)
            if match:
                out[key] = _safe_float(match.group(1), out[key])
    return out


def _parse_cboe_vix_spot(html: str) -> float:
    parser = _TextLineParser()
    parser.feed(str(html or ""))
    lines = parser.lines
    for idx, line in enumerate(lines):
        if str(line).strip().upper() != "VIX SPOT PRICE":
            continue
        for offset in (1, 2, -1, -2):
            pos = idx + offset
            if pos < 0 or pos >= len(lines):
                continue
            candidate = str(lines[pos]).strip().replace("$", "")
            value = _safe_float(candidate, float("nan"))
            if math.isfinite(value):
                return value
    return 0.0


def _parse_nasdaq_threshold_rows(html: str) -> list[dict[str, Any]]:
    parser = _SectionTableParser()
    parser.feed(str(html or ""))
    flat_rows = [row for rows in parser.sections.values() for row in rows]
    out: list[dict[str, Any]] = []
    capture = False
    for row in flat_rows:
        if len(row) >= 5 and row[0].strip().upper() == "SYMBOL" and "REG SHO" in " ".join(row).upper():
            capture = True
            continue
        if not capture or len(row) < 5:
            continue
        symbol = _normalize_symbol(row[0])
        if not symbol:
            continue
        out.append(
            {
                "symbol": symbol,
                "security_name": row[1].strip(),
                "market_category": row[2].strip(),
                "reg_sho_flag": row[3].strip().upper(),
                "rule3210": row[4].strip().upper(),
            }
        )
    return out


def _discover_first_zip_link(page_url: str, html: str) -> str | None:
    parser = _LinkParser()
    parser.feed(str(html or ""))
    for href, _text in parser.links:
        if str(href or "").lower().endswith(".zip"):
            return urljoin(page_url, href)
    return None


def _parse_sec_ftd_rows(raw_bytes: bytes, tracked_symbols: set[str]) -> list[dict[str, Any]]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            members = [name for name in zf.namelist() if not str(name or "").endswith("/")]
            preferred = [name for name in members if str(name or "").lower().endswith((".txt", ".csv"))]
            target = preferred[0] if preferred else (members[0] if members else None)
            if not target:
                return []
            text = zf.read(target).decode("utf-8", "replace")
    except Exception:
        return []

    latest_by_symbol: dict[str, dict[str, Any]] = {}
    for row in csv.reader(io.StringIO(text), delimiter="|"):
        if len(row) < 6:
            continue
        settlement = str(row[0] or "").strip()
        if settlement.upper() == "SETTLEMENT DATE" or not settlement.isdigit():
            continue
        symbol = _normalize_symbol(row[2])
        if tracked_symbols and symbol not in tracked_symbols:
            continue
        quantity = _safe_float(row[3], 0.0)
        price = _safe_float(row[5], 0.0)
        existing = latest_by_symbol.get(symbol)
        if existing is not None and str(existing.get("settlement_date") or "") >= settlement:
            continue
        latest_by_symbol[symbol] = {
            "symbol": symbol,
            "settlement_date": settlement,
            "quantity": quantity,
            "price": price,
            "description": str(row[4] or "").strip(),
        }
    return sorted(latest_by_symbol.values(), key=lambda item: (str(item.get("settlement_date") or ""), str(item.get("symbol") or "")), reverse=True)


def collect_extended_quant_context(
    *,
    symbols: list[str],
    user_agent: str,
    timeout: float = 20.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = datetime.now(timezone.utc)
    tracked_set = {symbol.upper() for symbol in symbols}
    errors: list[str] = []

    derived_global: dict[str, float] = {}
    derived_symbol: dict[str, dict[str, float]] = {}
    bond_overlay: dict[str, Any] = {}
    source_status: dict[str, Any] = {}

    cftc_report_url, cftc_status = _discover_cftc_financial_report_url(user_agent, timeout)
    cftc_rows: list[dict[str, Any]] = []
    if cftc_report_url:
        cftc_text, err = _safe_http_text(cftc_report_url, user_agent=user_agent, timeout=timeout)
        if err or not cftc_text:
            cftc_status["ok"] = False
            cftc_status["error"] = err
            errors.append(f"cftc:{err}")
        else:
            cftc_rows = _parse_cftc_financial_long_report(cftc_text)
            cftc_status["ok"] = bool(cftc_rows)
            cftc_status["rows"] = len(cftc_rows)
            if cftc_rows:
                derived_global.update(_derive_cftc_features(cftc_rows))
    source_status["cftc_cot"] = cftc_status

    sofr_rate_payload = None
    sofr_avg_payload = None
    sofr_rate_error = None
    sofr_avg_error = None
    for url in NYFED_SOFR_RATE_URLS:
        payload, err = _safe_http_json(url, user_agent=user_agent, timeout=timeout)
        if err or payload is None:
            sofr_rate_error = err
            continue
        sofr_rate_payload = payload
        sofr_rate_error = None
        break
    for url in NYFED_SOFR_AVERAGES_URLS:
        payload, err = _safe_http_json(url, user_agent=user_agent, timeout=timeout)
        if err or payload is None:
            sofr_avg_error = err
            continue
        sofr_avg_payload = payload
        sofr_avg_error = None
        break
    sofr_snapshot = _extract_sofr_snapshot(sofr_rate_payload, sofr_avg_payload)
    sofr_features, sofr_bond_overlay = _derive_sofr_features(sofr_snapshot)
    if any(value > 0.0 for value in sofr_snapshot.values()):
        derived_global.update(sofr_features)
        bond_overlay.update(sofr_bond_overlay)
    else:
        if sofr_rate_error:
            errors.append(f"sofr_rate:{sofr_rate_error}")
        if sofr_avg_error:
            errors.append(f"sofr_avg:{sofr_avg_error}")
    source_status["nyfed_sofr"] = {
        "ok": any(value > 0.0 for value in sofr_snapshot.values()),
        "rate_error": sofr_rate_error,
        "averages_error": sofr_avg_error,
        "snapshot": sofr_snapshot,
    }

    cboe_market_html, cboe_market_err = _safe_http_text(CBOE_MARKET_STATS_URL, user_agent=user_agent, timeout=timeout)
    cboe_ratios = _parse_cboe_market_stats_html(cboe_market_html or "") if cboe_market_html else {"total_ratio": 0.0, "index_ratio": 0.0, "equity_ratio": 0.0}
    cboe_vix_html, cboe_vix_err = _safe_http_text(CBOE_VIX_URL, user_agent=user_agent, timeout=timeout)
    cboe_vix_spot = _parse_cboe_vix_spot(cboe_vix_html or "") if cboe_vix_html else 0.0
    if any(value > 0.0 for value in cboe_ratios.values()) or cboe_vix_spot > 0.0:
        derived_global.update(
            {
                "cboe_total_put_call_norm": _clamp01(cboe_ratios["total_ratio"] / 1.5) if cboe_ratios["total_ratio"] > 0.0 else 0.0,
                "cboe_index_put_call_norm": _clamp01(cboe_ratios["index_ratio"] / 1.8) if cboe_ratios["index_ratio"] > 0.0 else 0.0,
                "cboe_equity_put_call_norm": _clamp01(cboe_ratios["equity_ratio"] / 1.5) if cboe_ratios["equity_ratio"] > 0.0 else 0.0,
                "cboe_put_call_stress_norm": _clamp01(max(cboe_ratios.values()) / 1.5),
                "cboe_vix_spot_norm": _clamp01(cboe_vix_spot / 45.0) if cboe_vix_spot > 0.0 else 0.0,
            }
        )
    else:
        if cboe_market_err:
            errors.append(f"cboe_market:{cboe_market_err}")
        if cboe_vix_err:
            errors.append(f"cboe_vix:{cboe_vix_err}")
    source_status["cboe"] = {
        "ok": any(value > 0.0 for value in cboe_ratios.values()) or cboe_vix_spot > 0.0,
        "market_error": cboe_market_err,
        "vix_error": cboe_vix_err,
        "ratios": cboe_ratios,
        "vix_spot": cboe_vix_spot,
    }

    threshold_html, threshold_err = _safe_http_text(NASDAQ_THRESHOLD_URL, user_agent=user_agent, timeout=timeout)
    threshold_rows = _parse_nasdaq_threshold_rows(threshold_html or "") if threshold_html else []
    threshold_symbol_features: dict[str, dict[str, float]] = {}
    if threshold_rows:
        for row in threshold_rows:
            symbol = str(row.get("symbol") or "").upper()
            if tracked_set and symbol not in tracked_set:
                continue
            threshold_symbol_features[symbol] = {
                "short_threshold_listed_norm": 1.0,
                "short_threshold_rule3210_norm": 1.0 if str(row.get("rule3210") or "").upper() == "Y" else 0.0,
            }
        derived_symbol.update(threshold_symbol_features)
        relevant_count = len(threshold_symbol_features) if tracked_set else len(threshold_rows)
        derived_global.update(
            {
                "short_threshold_symbol_share_norm": _clamp01(relevant_count / max(len(tracked_set), 1)),
                "short_threshold_total_listed_norm": _clamp01(len(threshold_rows) / 120.0),
                "short_threshold_recency_norm": 1.0,
            }
        )
    elif threshold_err:
        errors.append(f"threshold:{threshold_err}")
    source_status["nasdaq_threshold"] = {
        "ok": bool(threshold_rows),
        "error": threshold_err,
        "rows": len(threshold_rows),
        "tracked_hits": len(threshold_symbol_features),
    }

    sec_ftd_html, sec_ftd_err = _safe_http_text(SEC_FTD_URL, user_agent=user_agent, timeout=timeout)
    sec_ftd_zip_url = _discover_first_zip_link(SEC_FTD_URL, sec_ftd_html or "") if sec_ftd_html else None
    sec_ftd_rows: list[dict[str, Any]] = []
    sec_ftd_zip_err = None
    if sec_ftd_zip_url:
        sec_ftd_bytes, sec_ftd_zip_err = _safe_http_bytes(sec_ftd_zip_url, user_agent=user_agent, timeout=timeout)
        if sec_ftd_bytes:
            sec_ftd_rows = _parse_sec_ftd_rows(sec_ftd_bytes, tracked_set)
    if sec_ftd_rows:
        for row in sec_ftd_rows:
            symbol = str(row.get("symbol") or "").upper()
            features = derived_symbol.setdefault(symbol, {})
            features.update(
                {
                    "short_ftd_presence_norm": 1.0,
                    "short_ftd_quantity_norm": _clamp01(math.log10(max(_safe_float(row.get("quantity"), 0.0), 1.0)) / 7.0),
                }
            )
        derived_global.update(
            {
                "short_ftd_symbol_share_norm": _clamp01(len(sec_ftd_rows) / max(len(tracked_set), 1)),
                "short_ftd_total_hits_norm": _clamp01(len(sec_ftd_rows) / 60.0),
            }
        )
    else:
        if sec_ftd_err:
            errors.append(f"sec_ftd_page:{sec_ftd_err}")
        if sec_ftd_zip_err:
            errors.append(f"sec_ftd_zip:{sec_ftd_zip_err}")
    source_status["sec_ftd"] = {
        "ok": bool(sec_ftd_rows),
        "page_error": sec_ftd_err,
        "zip_error": sec_ftd_zip_err,
        "zip_url": sec_ftd_zip_url,
        "rows": len(sec_ftd_rows),
    }

    status = {
        "timestamp_utc": now.isoformat(),
        "ok": any(bool(section.get("ok")) for section in source_status.values()),
        "tracked_symbols": len(tracked_set),
        "error_count": len(errors),
        "errors": errors[:20],
        "sources": source_status,
    }
    payload = {
        "timestamp_utc": now.isoformat(),
        "provider": "extended_quant_context",
        "tracked_symbols": sorted(tracked_set),
        "status": status,
        "sources": {
            "cftc_rows": cftc_rows[:48],
            "nyfed_sofr_snapshot": sofr_snapshot,
            "cboe_ratios": cboe_ratios,
            "cboe_vix_spot": cboe_vix_spot,
            "nasdaq_threshold_rows": threshold_rows[:240],
            "sec_ftd_rows": sec_ftd_rows[:160],
        },
        "derived": {
            "calendar_features": {},
            "news_features": {},
            "global_features": derived_global,
            "symbol_features": derived_symbol,
            "bond_reference_overlay": bond_overlay,
        },
    }
    return payload, status


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect official extended quant context from public sources.")
    parser.add_argument("--symbols", default=",".join(_default_symbols()))
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    user_agent = str(os.getenv("SEC_EDGAR_USER_AGENT") or USER_AGENT_DEFAULT).strip() or USER_AGENT_DEFAULT
    payload, status = collect_extended_quant_context(
        symbols=symbols,
        user_agent=user_agent,
        timeout=args.timeout,
    )

    _write_json(PROJECT_ROOT / "exports" / "external_context" / "extended_quant_context_latest.json", payload)
    _write_json(PROJECT_ROOT / "governance" / "health" / "extended_quant_context_sync_latest.json", status)

    if args.json:
        print(json.dumps(status, ensure_ascii=True))
    else:
        print(
            "extended_quant_context ok={ok} tracked={tracked} errors={errors}".format(
                ok=status["ok"],
                tracked=status["tracked_symbols"],
                errors=status["error_count"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
