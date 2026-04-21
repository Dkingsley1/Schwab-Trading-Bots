#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.collector_transport import fetch_json, fetch_text

USER_AGENT = "schwab-trading-bot/1.0"
TREASURY_AUCTIONS_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
FINRA_REGSHO_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{stamp}.txt"
NASDAQ_TRADE_HALTS_URL = "https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts"
SOURCE_CONTRACTS = {
    "treasury_auctions": {"source_confidence_norm": 0.97, "schema_confidence_norm": 0.94},
    "finra_short_volume": {"source_confidence_norm": 0.94, "schema_confidence_norm": 0.92},
    "nasdaq_trade_halts": {"source_confidence_norm": 0.95, "schema_confidence_norm": 0.91},
    "local_micro": {"source_confidence_norm": 0.9, "schema_confidence_norm": 0.93},
}
DEFAULT_SYMBOLS = [
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "MDY",
    "VOO",
    "VTI",
    "RSP",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "AVGO",
    "ORCL",
    "CRM",
    "ADBE",
    "NFLX",
    "DIS",
    "GS",
    "JPM",
    "TLT",
    "IEF",
    "TLH",
    "VGIT",
    "VGLT",
    "EDV",
    "ZROZ",
    "SHY",
    "FLOT",
    "VGSH",
    "SCHO",
    "TIP",
    "VTIP",
    "SCHP",
    "LQD",
    "IGIB",
    "HYG",
    "JNK",
    "USHY",
    "AGG",
    "BND",
    "MUB",
    "XLU",
    "XLF",
    "GLD",
    "UUP",
    "COIN",
    "MSTR",
    "TSLA",
    "NVDA",
    "PLTR",
    "AMD",
    "SMCI",
    "SOXL",
    "SOXS",
    "TQQQ",
    "SQQQ",
    "UVXY",
    "VIXY",
    "XOP",
    "OIH",
    "SLB",
    "HAL",
    "AAL",
    "UAL",
    "DAL",
    "LUV",
    "JETS",
]

KNOWN_ETF_SYMBOLS = {
    "SPY", "QQQ", "DIA", "IWM", "MDY", "VOO", "VTI", "RSP", "TLT", "IEF", "TLH", "VGIT", "VGLT",
    "EDV", "ZROZ", "SHY", "FLOT", "VGSH", "TIP", "VTIP", "SCHP", "LQD", "IGIB", "HYG", "JNK",
    "USHY", "AGG", "BND", "MUB", "XLU", "XLF", "GLD", "UUP", "JETS", "SOXL", "SOXS", "TQQQ",
    "SQQQ", "UVXY", "VIXY", "XOP", "OIH",
}
ETF_FAMILY_BY_SYMBOL = {
    "SPY": "spdr",
    "XLU": "spdr",
    "XLF": "spdr",
    "XOP": "spdr",
    "OIH": "vaneck",
    "QQQ": "invesco",
    "DIA": "state_street",
    "IWM": "ishares",
    "MDY": "spdr",
    "VOO": "vanguard",
    "VTI": "vanguard",
    "VGIT": "vanguard",
    "VGLT": "vanguard",
    "VGSH": "vanguard",
    "VTIP": "vanguard",
    "RSP": "invesco",
    "TLT": "ishares",
    "IEF": "ishares",
    "TLH": "ishares",
    "SHY": "ishares",
    "TIP": "ishares",
    "LQD": "ishares",
    "IGIB": "ishares",
    "HYG": "ishares",
    "AGG": "ishares",
    "MUB": "ishares",
    "SCHP": "schwab",
    "FLOT": "ishares",
    "BND": "vanguard",
    "USHY": "ishares",
    "JNK": "state_street",
    "GLD": "state_street",
    "UUP": "invesco",
    "JETS": "us_global",
    "SOXL": "direxion",
    "SOXS": "direxion",
    "TQQQ": "proshares",
    "SQQQ": "proshares",
    "UVXY": "proshares",
    "VIXY": "proshares",
}
_RSS_ITEM_RE = re.compile(r"<item\b.*?>.*?<title>(.*?)</title>.*?<description>(.*?)</description>.*?</item>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _source_contract(source_name: str) -> dict[str, float]:
    row = SOURCE_CONTRACTS.get(str(source_name or ""), {})
    return {
        "source_confidence_norm": float(row.get("source_confidence_norm", 0.9) or 0.9),
        "schema_confidence_norm": float(row.get("schema_confidence_norm", 0.9) or 0.9),
    }


def _fetch_json_result(url: str, *, source_name: str, timeout: float = 12.0) -> dict[str, Any]:
    contract = _source_contract(source_name)
    return fetch_json(
        url=url,
        user_agent=USER_AGENT,
        timeout=timeout,
        collector_key="market_micro_context",
        source_name=source_name,
        entity_key=url,
        project_root=PROJECT_ROOT,
        source_confidence_norm=contract["source_confidence_norm"],
        schema_confidence_norm=contract["schema_confidence_norm"],
    )


def _fetch_text_result(url: str, *, source_name: str, timeout: float = 12.0) -> dict[str, Any]:
    contract = _source_contract(source_name)
    return fetch_text(
        url=url,
        user_agent=USER_AGENT,
        timeout=timeout,
        collector_key="market_micro_context",
        source_name=source_name,
        entity_key=url,
        project_root=PROJECT_ROOT,
        source_confidence_norm=contract["source_confidence_norm"],
        schema_confidence_norm=contract["schema_confidence_norm"],
    )


def _http_json(url: str, *, timeout: float = 12.0) -> Any:
    result = _fetch_json_result(url, source_name="market_micro_http", timeout=timeout)
    if not bool(result.get("ok", False)):
        raise RuntimeError(str(result.get("error") or "http_json_failed"))
    return result.get("json")


def _http_text(url: str, *, timeout: float = 12.0) -> str:
    result = _fetch_text_result(url, source_name="market_micro_http", timeout=timeout)
    if not bool(result.get("ok", False)):
        raise RuntimeError(str(result.get("error") or "http_text_failed"))
    return str(result.get("text") or "")


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


def _signed_centered_norm(value: float, scale: float) -> float:
    return _clamp01(0.5 + (float(value) / max(float(scale), 1e-8)))


def _parse_ts(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    text = str(raw).strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _strip_html(raw: str) -> str:
    text = _HTML_TAG_RE.sub(" ", str(raw or ""))
    return re.sub(r"\s+", " ", text).strip()


def _parse_nasdaq_trade_halt_rows(text: str) -> List[Dict[str, Any]]:
    raw = str(text or "")
    if not raw.strip():
        return []

    rows: List[Dict[str, Any]] = []
    if "|" in raw:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if len(lines) >= 2:
            header = [part.strip().lower() for part in lines[0].split("|")]
            idx = {name: pos for pos, name in enumerate(header)}
            symbol_idx = idx.get("symbol")
            reason_idx = idx.get("reason code") if "reason code" in idx else idx.get("reason")
            halt_time_idx = idx.get("halt time")
            resume_time_idx = idx.get("resume time")
            for line in lines[1:]:
                parts = [part.strip() for part in line.split("|")]
                if symbol_idx is None or symbol_idx >= len(parts):
                    continue
                symbol = str(parts[symbol_idx] or "").strip().upper()
                if not symbol:
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "reason": parts[reason_idx] if reason_idx is not None and reason_idx < len(parts) else "",
                        "halt_time": parts[halt_time_idx] if halt_time_idx is not None and halt_time_idx < len(parts) else "",
                        "resume_time": parts[resume_time_idx] if resume_time_idx is not None and resume_time_idx < len(parts) else "",
                    }
                )
            if rows:
                return rows

    for title, description in _RSS_ITEM_RE.findall(raw):
        text_row = _strip_html(f"{title} {description}")
        match = re.search(r"\b([A-Z]{1,5})\b", text_row)
        if not match:
            continue
        symbol = match.group(1)
        reason_match = re.search(r"(?:reason|code)\s*[:=-]?\s*([A-Z0-9 ]{2,16})", text_row, re.IGNORECASE)
        halt_match = re.search(r"(?:halt(?:ed)?(?: at| time)?|time)\s*[:=-]?\s*([0-9: ]+[APMapm\.]{0,4})", text_row)
        resume_match = re.search(r"(?:resume(?: time| at)?|resumption)\s*[:=-]?\s*([0-9: ]+[APMapm\.]{0,4})", text_row)
        rows.append(
            {
                "symbol": symbol,
                "reason": reason_match.group(1).strip() if reason_match else text_row,
                "halt_time": halt_match.group(1).strip() if halt_match else "",
                "resume_time": resume_match.group(1).strip() if resume_match else "",
            }
        )
    return rows


def _etf_fund_family(symbol: str) -> str:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return ""
    if sym in ETF_FAMILY_BY_SYMBOL:
        return ETF_FAMILY_BY_SYMBOL[sym]
    if sym.startswith("SCH"):
        return "schwab"
    if sym.startswith("VG") or sym.startswith("VT") or sym in {"VOO", "VTI", "BND"}:
        return "vanguard"
    if sym.startswith("I") or sym in {"TLT", "IEF", "SHY", "TIP", "LQD", "HYG", "AGG"}:
        return "ishares"
    if sym in KNOWN_ETF_SYMBOLS:
        return "other"
    return ""


def _parse_intraday_clock(raw: Any, now_utc: datetime) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M"):
        try:
            clock = datetime.strptime(text.replace(".", ""), fmt)
            return now_utc.replace(hour=clock.hour, minute=clock.minute, second=0, microsecond=0)
        except Exception:
            continue
    return _parse_ts(text)


def _fetch_nasdaq_trade_halts(*, symbols: set[str], timeout_seconds: float) -> Dict[str, Any]:
    fetch_result = _fetch_text_result(NASDAQ_TRADE_HALTS_URL, source_name="nasdaq_trade_halts", timeout=timeout_seconds)
    if not bool(fetch_result.get("ok", False)):
        contract = _source_contract("nasdaq_trade_halts")
        return {
            "ok": False,
            "rows": [],
            "all_rows_count": 0,
            "error": str(fetch_result.get("error") or "http_text_failed"),
            "url": NASDAQ_TRADE_HALTS_URL,
            "source_confidence_norm": float(fetch_result.get("source_confidence_norm") or contract["source_confidence_norm"]),
            "schema_confidence_norm": float(fetch_result.get("schema_confidence_norm") or contract["schema_confidence_norm"]),
            "freshness_norm": float(fetch_result.get("freshness_norm") or 0.0),
            "provenance": dict(fetch_result.get("provenance") or {}),
        }
    text = str(fetch_result.get("text") or "")
    all_rows = _parse_nasdaq_trade_halt_rows(text)
    rows = list(all_rows)
    if symbols:
        rows = [row for row in rows if str(row.get("symbol") or "").upper() in symbols]
    return {
        "ok": bool(str(text or "").strip()),
        "rows": rows,
        "all_rows_count": len(all_rows),
        "url": NASDAQ_TRADE_HALTS_URL,
        "source_confidence_norm": float(fetch_result.get("source_confidence_norm") or _source_contract("nasdaq_trade_halts")["source_confidence_norm"]),
        "schema_confidence_norm": float(fetch_result.get("schema_confidence_norm") or _source_contract("nasdaq_trade_halts")["schema_confidence_norm"]),
        "freshness_norm": float(fetch_result.get("freshness_norm") or 0.0),
        "provenance": dict(fetch_result.get("provenance") or {}),
    }


def _load_env_file(path: Path, *, override: bool = False) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value


def _bootstrap_env() -> None:
    for path, override in [
        (PROJECT_ROOT / ".env", False),
        (PROJECT_ROOT / ".env.live", True),
        (PROJECT_ROOT / "config" / ".env", True),
        (PROJECT_ROOT / "config" / ".env.live", True),
        (PROJECT_ROOT / ".env.secrets.local", True),
        (PROJECT_ROOT / ".env.live.secrets.local", True),
        (PROJECT_ROOT / "config" / ".env.secrets.local", True),
        (PROJECT_ROOT / "config" / ".env.live.secrets.local", True),
    ]:
        _load_env_file(path, override=override)


def _parse_symbol_csv(raw: str) -> List[str]:
    out: List[str] = []
    for token in str(raw or "").split(","):
        symbol = token.strip().upper()
        if not symbol:
            continue
        out.append(symbol)
    return out


def _default_symbol_list() -> List[str]:
    from_env = _parse_symbol_csv(os.getenv("MARKET_MICRO_SYMBOLS", ""))
    if from_env:
        return from_env

    merged: List[str] = []
    seen: set[str] = set()
    for raw in (
        os.getenv("SHADOW_SYMBOLS_CORE", ""),
        os.getenv("SHADOW_SYMBOLS_VOLATILE", ""),
        os.getenv("SHADOW_SYMBOLS_DEFENSIVE", ""),
        os.getenv("SHADOW_SYMBOLS_COMMOD_FX_INTL", ""),
        os.getenv("BOND_SYMBOLS", ""),
        os.getenv("BOND_CONTEXT_SYMBOLS", ""),
        ",".join(DEFAULT_SYMBOLS),
    ):
        for symbol in _parse_symbol_csv(raw):
            if symbol in seen:
                continue
            seen.add(symbol)
            merged.append(symbol)
    return merged


def _path_day_utc(path: Path) -> Optional[datetime]:
    parts = path.stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    stamp = parts[-1]
    if len(stamp) != 8 or not stamp.isdigit():
        return None
    try:
        return datetime.strptime(stamp, "%Y%m%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _recent_decision_paths(project_root: Path, *, lookback_days: int) -> List[Path]:
    since_utc = datetime.now(timezone.utc) - timedelta(days=max(int(lookback_days), 1))
    cutoff_day = (since_utc - timedelta(days=1)).date()
    out: List[Path] = []
    for path in (project_root / "decision_explanations").glob("shadow*/decision_explanations_*.jsonl"):
        day_utc = _path_day_utc(path)
        if day_utc is not None and day_utc.date() >= cutoff_day:
            out.append(path)
            continue
        try:
            mtime_utc = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            continue
        if mtime_utc >= since_utc - timedelta(days=1):
            out.append(path)
    return sorted({p.resolve() for p in out})


def _iter_recent_rows(project_root: Path, *, lookback_days: int, symbols: set[str]) -> Iterable[Dict[str, Any]]:
    allowed_strategies = {
        "grand_master_bot",
        "grand_master_intent_bot",
        "options_master_bot",
        "futures_master_bot",
    }
    allowed_layers = {
        "grand_master",
        "grand_master_intent",
        "options_master",
        "futures_master",
        "options_sub_bot",
        "futures_sub_bot",
    }
    for path in _recent_decision_paths(project_root, lookback_days=lookback_days):
        try:
            with path.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    strategy = str(row.get("strategy") or "").strip().lower()
                    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                    layer = str(
                        row.get("layer")
                        or metadata.get("layer")
                        or ""
                    ).strip().lower()
                    if strategy not in allowed_strategies and layer not in allowed_layers:
                        continue
                    symbol = str(row.get("symbol") or "").strip().upper()
                    if symbols and symbol not in symbols:
                        continue
                    features = row.get("features")
                    if not isinstance(features, dict):
                        continue
                    ts = _parse_ts(row.get("ts_utc") or row.get("timestamp_utc"))
                    if ts is None:
                        continue
                    row["_parsed_ts_utc"] = ts
                    yield row
        except Exception:
            continue


def _aggregate_local_micro_context(project_root: Path, *, lookback_days: int, symbols: set[str]) -> Dict[str, Dict[str, float]]:
    et_zone = ZoneInfo("America/New_York") if ZoneInfo is not None else timezone.utc
    per_symbol: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    family_acc: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    family_counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    last_quote_state: Dict[str, Dict[str, Any]] = {}

    for row in _iter_recent_rows(project_root, lookback_days=lookback_days, symbols=symbols):
        symbol = str(row.get("symbol") or "").strip().upper()
        ts_utc = row.get("_parsed_ts_utc")
        if not isinstance(ts_utc, datetime):
            continue
        ts_et = ts_utc.astimezone(et_zone) if et_zone is not None else ts_utc
        minute = ts_et.hour * 60 + ts_et.minute
        feats = row.get("features") if isinstance(row.get("features"), dict) else {}
        action = str(row.get("action") or "").strip().upper()
        quantity = abs(_safe_float(row.get("quantity"), 0.0))
        vol_30m = abs(_safe_float(feats.get("vol_30m"), 0.0))
        pct = abs(_safe_float(feats.get("pct_from_close"), 0.0))
        signed_pct = _safe_float(feats.get("pct_from_close"), 0.0)
        mom_5m = _safe_float(feats.get("mom_5m"), 0.0)
        spread_bps = abs(_safe_float(feats.get("spread_bps"), 0.0))
        bid_size = max(_safe_float(feats.get("bid_size"), 0.0), 0.0)
        ask_size = max(_safe_float(feats.get("ask_size"), 0.0), 0.0)
        queue_depth = max(_safe_float(feats.get("queue_depth"), 0.0), bid_size + ask_size)
        gamma = abs(_safe_float(feats.get("options_gamma_exposure_norm"), 0.0))
        unusual = abs(_safe_float(feats.get("options_unusual_flow_norm"), 0.0))
        put_call = abs(_safe_float(feats.get("options_put_call_oi_ratio_norm"), 0.5) - 0.5) * 2.0
        hy_ig = abs(_safe_float(feats.get("bond_hy_ig_flow_norm"), 0.5) - 0.5) * 2.0
        nav_stress = abs(_safe_float(feats.get("bond_nav_stress_norm"), 0.0))
        off_exchange_share = max(
            abs(_safe_float(feats.get("off_exchange_share_norm"), 0.0)),
            abs(_safe_float(feats.get("trf_share_norm"), 0.0)),
            abs(_safe_float(feats.get("ats_volume_share_norm"), 0.0)),
        )
        dark_pool = max(
            abs(_safe_float(feats.get("dark_pool_print_norm"), 0.0)),
            abs(_safe_float(feats.get("dark_pool_imbalance_norm"), 0.0)),
            abs(_safe_float(feats.get("dark_pool_block_trade_norm"), 0.0)),
            off_exchange_share,
        )
        etf_like = symbol in KNOWN_ETF_SYMBOLS or any(
            key in feats
            for key in (
                "etf_nav_premium_discount_norm",
                "etf_nav_discount_norm",
                "etf_creation_redemption_stress_norm",
                "etf_primary_secondary_liquidity_norm",
                "etf_underlying_basket_stress_norm",
            )
        )
        etf_nav = max(
            abs(_safe_float(feats.get("etf_nav_premium_discount_norm"), 0.0)),
            abs(_safe_float(feats.get("etf_nav_discount_norm"), 0.0)),
            nav_stress,
        ) if etf_like else 0.0
        etf_creation = max(
            abs(_safe_float(feats.get("etf_creation_redemption_stress_norm"), 0.0)),
            etf_nav,
            _clamp01(spread_bps / 35.0),
        ) if etf_like else 0.0
        etf_liquidity = _clamp01(
            max(
                _safe_float(feats.get("etf_primary_secondary_liquidity_norm"), 0.0),
                (0.55 * _clamp01(vol_30m / 0.02)) + (0.45 * (1.0 - _clamp01(spread_bps / 35.0))),
            )
        ) if etf_like else 0.0
        action_bias = 0.0
        if action == "BUY":
            action_bias = 1.0
        elif action == "SELL":
            action_bias = -1.0
        signed_move_alignment = signed_pct * mom_5m
        trend_persistence = _clamp01((0.55 * _clamp01(abs(signed_pct) / 0.02)) + (0.45 * _clamp01(abs(mom_5m) / 0.01))) if signed_move_alignment > 0.0 else 0.0
        reversal_risk = _clamp01((0.60 * _clamp01(abs(signed_pct) / 0.02)) + (0.40 * _clamp01(abs(mom_5m) / 0.01))) if signed_move_alignment < 0.0 else 0.0
        range_expansion = _clamp01(
            (0.50 * _clamp01(abs(signed_pct) / 0.02))
            + (0.30 * _clamp01(vol_30m / 0.02))
            + (0.20 * _clamp01(spread_bps / 25.0))
        )
        etf_basket_stress = max(
            abs(_safe_float(feats.get("etf_underlying_basket_stress_norm"), 0.0)),
            etf_nav,
            range_expansion,
        ) if etf_like else 0.0
        activity_weight = max(quantity, 1.0) * max(vol_30m, 0.005)
        prev_state = last_quote_state.get(symbol)
        spread_widening = 0.0
        depth_decay = 0.0
        quote_fade = 0.0
        if isinstance(prev_state, dict):
            prev_ts = prev_state.get("ts_utc")
            if isinstance(prev_ts, datetime):
                elapsed_seconds = max((ts_utc - prev_ts).total_seconds(), 0.0)
                if elapsed_seconds <= 5400.0:
                    prev_spread = max(_safe_float(prev_state.get("spread_bps"), 0.0), 0.0)
                    prev_depth = max(_safe_float(prev_state.get("queue_depth"), 0.0), 0.0)
                    prev_bid = max(_safe_float(prev_state.get("bid_size"), 0.0), 0.0)
                    prev_ask = max(_safe_float(prev_state.get("ask_size"), 0.0), 0.0)
                    spread_widening = _clamp01(max(spread_bps - prev_spread, 0.0) / 18.0)
                    if prev_depth > 0.0:
                        depth_decay = _clamp01(max(prev_depth - queue_depth, 0.0) / max(prev_depth, 1.0))
                    bid_fade = _clamp01(max(prev_bid - bid_size, 0.0) / max(prev_bid, 1.0))
                    ask_fade = _clamp01(max(prev_ask - ask_size, 0.0) / max(prev_ask, 1.0))
                    quote_fade = _clamp01(max(bid_fade, ask_fade, (0.60 * depth_decay) + (0.40 * spread_widening)))
        spread_regime = _clamp01((0.65 * _clamp01(spread_bps / 25.0)) + (0.35 * spread_widening))
        drive_alignment = 1.0 if (action_bias != 0.0 and (action_bias * signed_pct) > 0.0) else 0.0
        opening_drive = _clamp01(
            (0.40 * trend_persistence)
            + (0.25 * range_expansion)
            + (0.20 * drive_alignment)
            + (0.15 * _clamp01(vol_30m / 0.02))
        )
        closing_cross = _clamp01(
            (0.35 * _clamp01(vol_30m / 0.02))
            + (0.25 * _clamp01(abs(signed_pct) / 0.02))
            + (0.20 * abs(action_bias))
            + (0.20 * max(spread_regime, quote_fade))
        )
        auction_print = _clamp01(
            max(
                _clamp01(abs(signed_pct) / 0.02),
                _clamp01(vol_30m / 0.02),
                spread_regime,
                quote_fade,
                abs(action_bias) * 0.85,
            )
        )

        if 240 <= minute < 570:
            per_symbol[symbol]["premarket_pressure"] += max(
                _clamp01(abs(signed_pct) / 0.02),
                _clamp01(vol_30m / 0.015),
                _clamp01(abs(action_bias) * 0.8),
            )
            counts[symbol]["premarket_pressure"] += 1.0
        if 570 <= minute <= 600:
            per_symbol[symbol]["opening_auction"] += max(min(pct / 0.02, 1.0), min(vol_30m / 0.02, 1.0))
            counts[symbol]["opening_auction"] += 1.0
            if action_bias > 0.0:
                per_symbol[symbol]["opening_auction_buy_flow"] += activity_weight
            elif action_bias < 0.0:
                per_symbol[symbol]["opening_auction_sell_flow"] += activity_weight
            per_symbol[symbol]["auction_print_pressure"] += auction_print
            counts[symbol]["auction_print_pressure"] += 1.0
        if 570 <= minute <= 645:
            per_symbol[symbol]["opening_drive_pressure"] += opening_drive
            counts[symbol]["opening_drive_pressure"] += 1.0
        if 900 <= minute <= 960:
            per_symbol[symbol]["power_hour_pressure"] += max(
                _clamp01(abs(signed_pct) / 0.02),
                _clamp01(abs(mom_5m) / 0.01),
                _clamp01(vol_30m / 0.02),
            )
            counts[symbol]["power_hour_pressure"] += 1.0
        if 930 <= minute <= 960:
            per_symbol[symbol]["closing_auction"] += max(min(pct / 0.02, 1.0), min(vol_30m / 0.02, 1.0))
            counts[symbol]["closing_auction"] += 1.0
            if action_bias > 0.0:
                per_symbol[symbol]["closing_auction_buy_flow"] += activity_weight
            elif action_bias < 0.0:
                per_symbol[symbol]["closing_auction_sell_flow"] += activity_weight
            per_symbol[symbol]["closing_cross_pressure"] += closing_cross
            counts[symbol]["closing_cross_pressure"] += 1.0
            per_symbol[symbol]["auction_print_pressure"] += auction_print
            counts[symbol]["auction_print_pressure"] += 1.0

        bucket = "midday" if 660 <= minute <= 840 else "other"
        per_symbol[symbol][f"vol_{bucket}"] += vol_30m
        counts[symbol][f"vol_{bucket}"] += 1.0

        if action == "BUY":
            per_symbol[symbol]["buy_qty"] += max(quantity, 1.0)
        elif action == "SELL":
            per_symbol[symbol]["sell_qty"] += max(quantity, 1.0)

        per_symbol[symbol]["options_flow"] += max(gamma, unusual, put_call)
        counts[symbol]["options_flow"] += 1.0
        per_symbol[symbol]["credit_flow"] += max(hy_ig, nav_stress)
        counts[symbol]["credit_flow"] += 1.0
        if 570 <= minute <= 780:
            per_symbol[symbol]["gap_continuation"] += trend_persistence
            counts[symbol]["gap_continuation"] += 1.0
        per_symbol[symbol]["reversal_risk"] += reversal_risk
        counts[symbol]["reversal_risk"] += 1.0
        per_symbol[symbol]["trend_persistence"] += trend_persistence
        counts[symbol]["trend_persistence"] += 1.0
        per_symbol[symbol]["range_expansion"] += range_expansion
        counts[symbol]["range_expansion"] += 1.0
        if quantity >= 5.0:
            per_symbol[symbol]["block_trade"] += min(quantity / 25.0, 1.0)
            counts[symbol]["block_trade"] += 1.0
        if signed_pct <= -0.10:
            per_symbol[symbol]["ssr_active"] = 1.0
        per_symbol[symbol]["dark_pool_pressure"] += _clamp01(dark_pool)
        counts[symbol]["dark_pool_pressure"] += 1.0
        per_symbol[symbol]["off_exchange_share"] += _clamp01(off_exchange_share)
        counts[symbol]["off_exchange_share"] += 1.0
        per_symbol[symbol]["spread_regime"] += spread_regime
        counts[symbol]["spread_regime"] += 1.0
        per_symbol[symbol]["spread_widening"] += spread_widening
        counts[symbol]["spread_widening"] += 1.0
        per_symbol[symbol]["queue_depth_decay"] += depth_decay
        counts[symbol]["queue_depth_decay"] += 1.0
        per_symbol[symbol]["depth_collapse"] += depth_decay
        counts[symbol]["depth_collapse"] += 1.0
        per_symbol[symbol]["quote_fade_rate"] += quote_fade
        counts[symbol]["quote_fade_rate"] += 1.0
        if etf_like:
            per_symbol[symbol]["etf_nav_premium_discount"] += _clamp01(etf_nav)
            counts[symbol]["etf_nav_premium_discount"] += 1.0
            per_symbol[symbol]["etf_creation_redemption_stress"] += _clamp01(etf_creation)
            counts[symbol]["etf_creation_redemption_stress"] += 1.0
            per_symbol[symbol]["etf_primary_secondary_liquidity"] += _clamp01(etf_liquidity)
            counts[symbol]["etf_primary_secondary_liquidity"] += 1.0
            per_symbol[symbol]["etf_underlying_basket_stress"] += _clamp01(etf_basket_stress)
            counts[symbol]["etf_underlying_basket_stress"] += 1.0
            family = _etf_fund_family(symbol)
            if family:
                family_acc[family]["signed_flow"] += action_bias * activity_weight
                family_acc[family]["abs_flow"] += activity_weight
                family_acc[family]["creation_pressure"] += _clamp01(max(etf_creation, etf_nav, etf_basket_stress))
                family_counts[family]["creation_pressure"] += 1.0

        last_quote_state[symbol] = {
            "ts_utc": ts_utc,
            "spread_bps": spread_bps,
            "queue_depth": queue_depth,
            "bid_size": bid_size,
            "ask_size": ask_size,
        }

    out: Dict[str, Dict[str, float]] = {}
    family_features: Dict[str, Dict[str, float]] = {}
    for family, acc in family_acc.items():
        signed_flow = _safe_float(acc.get("signed_flow"), 0.0) / max(_safe_float(acc.get("abs_flow"), 0.0), 1.0)
        creation_pressure = _safe_float(acc.get("creation_pressure"), 0.0) / max(_safe_float(family_counts.get(family, {}).get("creation_pressure"), 1.0), 1.0)
        family_features[family] = {
            "etf_fund_family_flow_norm": _signed_centered_norm(signed_flow, 1.0),
            "etf_fund_family_creation_pressure_norm": _clamp01(creation_pressure),
        }
    for symbol, acc in per_symbol.items():
        c = counts.get(symbol, {})
        premarket = acc.get("premarket_pressure", 0.0) / max(c.get("premarket_pressure", 1.0), 1.0)
        opening = acc.get("opening_auction", 0.0) / max(c.get("opening_auction", 1.0), 1.0)
        power_hour = acc.get("power_hour_pressure", 0.0) / max(c.get("power_hour_pressure", 1.0), 1.0)
        closing = acc.get("closing_auction", 0.0) / max(c.get("closing_auction", 1.0), 1.0)
        opening_buy = acc.get("opening_auction_buy_flow", 0.0)
        opening_sell = acc.get("opening_auction_sell_flow", 0.0)
        closing_buy = acc.get("closing_auction_buy_flow", 0.0)
        closing_sell = acc.get("closing_auction_sell_flow", 0.0)
        opening_imbalance = (opening_buy - opening_sell) / max(opening_buy + opening_sell, 1.0)
        closing_imbalance = (closing_buy - closing_sell) / max(closing_buy + closing_sell, 1.0)
        opening_drive_pressure = acc.get("opening_drive_pressure", 0.0) / max(c.get("opening_drive_pressure", 1.0), 1.0)
        closing_cross_pressure = acc.get("closing_cross_pressure", 0.0) / max(c.get("closing_cross_pressure", 1.0), 1.0)
        auction_print_pressure = acc.get("auction_print_pressure", 0.0) / max(c.get("auction_print_pressure", 1.0), 1.0)
        midday_vol = acc.get("vol_midday", 0.0) / max(c.get("vol_midday", 1.0), 1.0)
        other_vol = acc.get("vol_other", 0.0) / max(c.get("vol_other", 1.0), 1.0)
        buy_qty = acc.get("buy_qty", 0.0)
        sell_qty = acc.get("sell_qty", 0.0)
        order_flow = (buy_qty - sell_qty) / max(buy_qty + sell_qty, 1.0)
        options_flow = acc.get("options_flow", 0.0) / max(c.get("options_flow", 1.0), 1.0)
        credit_flow = acc.get("credit_flow", 0.0) / max(c.get("credit_flow", 1.0), 1.0)
        gap_continuation = acc.get("gap_continuation", 0.0) / max(c.get("gap_continuation", 1.0), 1.0)
        reversal_risk = acc.get("reversal_risk", 0.0) / max(c.get("reversal_risk", 1.0), 1.0)
        trend_persistence = acc.get("trend_persistence", 0.0) / max(c.get("trend_persistence", 1.0), 1.0)
        range_expansion = acc.get("range_expansion", 0.0) / max(c.get("range_expansion", 1.0), 1.0)
        block_trade = acc.get("block_trade", 0.0) / max(c.get("block_trade", 1.0), 1.0)
        dark_pool_pressure = acc.get("dark_pool_pressure", 0.0) / max(c.get("dark_pool_pressure", 1.0), 1.0)
        off_exchange = acc.get("off_exchange_share", 0.0) / max(c.get("off_exchange_share", 1.0), 1.0)
        spread_regime = acc.get("spread_regime", 0.0) / max(c.get("spread_regime", 1.0), 1.0)
        spread_widening = acc.get("spread_widening", 0.0) / max(c.get("spread_widening", 1.0), 1.0)
        queue_depth_decay = acc.get("queue_depth_decay", 0.0) / max(c.get("queue_depth_decay", 1.0), 1.0)
        depth_collapse = acc.get("depth_collapse", 0.0) / max(c.get("depth_collapse", 1.0), 1.0)
        quote_fade_rate = acc.get("quote_fade_rate", 0.0) / max(c.get("quote_fade_rate", 1.0), 1.0)
        etf_nav = acc.get("etf_nav_premium_discount", 0.0) / max(c.get("etf_nav_premium_discount", 1.0), 1.0)
        etf_creation = acc.get("etf_creation_redemption_stress", 0.0) / max(c.get("etf_creation_redemption_stress", 1.0), 1.0)
        etf_liquidity = acc.get("etf_primary_secondary_liquidity", 0.0) / max(c.get("etf_primary_secondary_liquidity", 1.0), 1.0)
        etf_basket_stress = acc.get("etf_underlying_basket_stress", 0.0) / max(c.get("etf_underlying_basket_stress", 1.0), 1.0)
        family_payload = family_features.get(_etf_fund_family(symbol), {})
        relative_volume = 0.0
        if other_vol > 0.0:
            relative_volume = min(midday_vol / max(other_vol, 1e-8), 2.0) / 2.0
        session_open = _clamp01(max(opening, opening_drive_pressure))
        session_midday = _clamp01(midday_vol / max(midday_vol + other_vol, 1e-8))
        session_power_hour = _clamp01(power_hour)
        overnight_gap = _clamp01(
            (0.42 * premarket)
            + (0.23 * abs(opening_imbalance))
            + (0.20 * opening)
            + (0.15 * gap_continuation)
        )
        post_event_drift = _clamp01(
            (0.32 * premarket)
            + (0.28 * opening_drive_pressure)
            + (0.24 * gap_continuation)
            + (0.16 * trend_persistence)
        )
        lunch_chop = _clamp01(
            (0.38 * session_midday)
            + (0.24 * reversal_risk)
            + (0.18 * max(1.0 - trend_persistence, 0.0))
            + (0.12 * spread_regime)
            + (0.08 * quote_fade_rate)
        )
        open_close_imbalance_regime = _clamp01(
            (0.40 * abs(opening_imbalance))
            + (0.35 * abs(closing_imbalance))
            + (0.15 * opening_drive_pressure)
            + (0.10 * closing_cross_pressure)
        )
        symbol_cooldown_pressure = _clamp01(
            (0.32 * spread_widening)
            + (0.28 * quote_fade_rate)
            + (0.20 * depth_collapse)
            + (0.12 * reversal_risk)
            + (0.08 * range_expansion)
        )
        gap_fade_risk = _clamp01(
            (0.36 * reversal_risk)
            + (0.24 * max(1.0 - gap_continuation, 0.0))
            + (0.18 * abs(opening_imbalance))
            + (0.12 * spread_widening)
            + (0.10 * quote_fade_rate)
        )
        overnight_event_hazard = _clamp01(
            (0.34 * overnight_gap)
            + (0.24 * post_event_drift)
            + (0.18 * spread_regime)
            + (0.12 * dark_pool_pressure)
            + (0.12 * off_exchange)
        )
        tradeability_score = _clamp01(
            1.0
            - (
                (0.22 * spread_regime)
                + (0.17 * spread_widening)
                + (0.16 * depth_collapse)
                + (0.15 * quote_fade_rate)
                + (0.10 * dark_pool_pressure)
                + (0.08 * off_exchange)
                + (0.07 * _clamp01(acc.get("ssr_active", 0.0)))
                + (0.05 * etf_creation)
            )
            + (0.12 * etf_liquidity)
        )
        out[symbol] = {
            "market_micro_premarket_pressure_norm": _clamp01(premarket),
            "market_micro_opening_auction_norm": _clamp01(opening),
            "market_micro_opening_auction_imbalance_norm": _signed_centered_norm(opening_imbalance, 1.0),
            "market_micro_opening_drive_pressure_norm": _clamp01(opening_drive_pressure),
            "market_micro_power_hour_pressure_norm": _clamp01(power_hour),
            "market_micro_closing_auction_norm": _clamp01(closing),
            "market_micro_closing_auction_imbalance_norm": _signed_centered_norm(closing_imbalance, 1.0),
            "market_micro_closing_cross_pressure_norm": _clamp01(closing_cross_pressure),
            "market_micro_auction_print_pressure_norm": _clamp01(auction_print_pressure),
            "market_micro_relative_volume_norm": _clamp01(relative_volume),
            "market_micro_order_flow_imbalance_norm": _signed_centered_norm(order_flow, 1.0),
            "market_micro_options_flow_norm": _clamp01(options_flow),
            "market_micro_credit_flow_norm": _clamp01(credit_flow),
            "market_micro_gap_continuation_norm": _clamp01(gap_continuation),
            "market_micro_reversal_risk_norm": _clamp01(reversal_risk),
            "market_micro_trend_persistence_norm": _clamp01(trend_persistence),
            "market_micro_range_expansion_norm": _clamp01(range_expansion),
            "market_micro_block_trade_norm": _clamp01(block_trade),
            "market_micro_ssr_active_norm": _clamp01(acc.get("ssr_active", 0.0)),
            "market_micro_dark_pool_pressure_norm": _clamp01(dark_pool_pressure),
            "market_micro_off_exchange_share_norm": _clamp01(off_exchange),
            "market_micro_spread_regime_norm": _clamp01(spread_regime),
            "market_micro_spread_widening_norm": _clamp01(spread_widening),
            "market_micro_queue_depth_decay_norm": _clamp01(queue_depth_decay),
            "market_micro_depth_collapse_norm": _clamp01(depth_collapse),
            "market_micro_quote_fade_rate_norm": _clamp01(quote_fade_rate),
            "market_micro_tradeability_score_norm": _clamp01(tradeability_score),
            "market_micro_session_open_norm": _clamp01(session_open),
            "market_micro_session_midday_norm": _clamp01(session_midday),
            "market_micro_session_power_hour_norm": _clamp01(session_power_hour),
            "market_micro_overnight_gap_norm": _clamp01(overnight_gap),
            "market_micro_post_event_drift_norm": _clamp01(post_event_drift),
            "market_micro_lunch_chop_norm": _clamp01(lunch_chop),
            "market_micro_open_close_imbalance_regime_norm": _clamp01(open_close_imbalance_regime),
            "market_micro_symbol_cooldown_pressure_norm": _clamp01(symbol_cooldown_pressure),
            "market_micro_gap_fade_risk_norm": _clamp01(gap_fade_risk),
            "market_micro_overnight_event_hazard_norm": _clamp01(overnight_event_hazard),
            "etf_nav_premium_discount_norm": _clamp01(etf_nav),
            "etf_creation_redemption_stress_norm": _clamp01(etf_creation),
            "etf_primary_secondary_liquidity_norm": _clamp01(etf_liquidity),
            "etf_underlying_basket_stress_norm": _clamp01(etf_basket_stress),
            "etf_fund_family_flow_norm": float(family_payload.get("etf_fund_family_flow_norm", 0.0) or 0.0),
            "etf_fund_family_creation_pressure_norm": float(family_payload.get("etf_fund_family_creation_pressure_norm", 0.0) or 0.0),
        }
    return out


def _fetch_treasury_auction_context(*, timeout_seconds: float) -> Dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    query = urlencode(
        {
            "sort": "-auction_date",
            "page[size]": 25,
            "filter": f"auction_date:gte:{(now_utc - timedelta(days=45)).date().isoformat()}",
        }
    )
    url = f"{TREASURY_AUCTIONS_URL}?{query}"
    fetch_result = _fetch_json_result(url, source_name="treasury_auctions", timeout=timeout_seconds)
    if not bool(fetch_result.get("ok", False)):
        contract = _source_contract("treasury_auctions")
        return {
            "ok": False,
            "rows": [],
            "error": str(fetch_result.get("error") or "http_json_failed"),
            "url": url,
            "source_confidence_norm": float(fetch_result.get("source_confidence_norm") or contract["source_confidence_norm"]),
            "schema_confidence_norm": float(fetch_result.get("schema_confidence_norm") or contract["schema_confidence_norm"]),
            "freshness_norm": float(fetch_result.get("freshness_norm") or 0.0),
            "provenance": dict(fetch_result.get("provenance") or {}),
        }
    payload = fetch_result.get("json")
    rows = payload.get("data") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return {
            "ok": False,
            "rows": [],
            "error": "unexpected_response_shape",
            "url": url,
            "source_confidence_norm": float(fetch_result.get("source_confidence_norm") or _source_contract("treasury_auctions")["source_confidence_norm"]),
            "schema_confidence_norm": float(fetch_result.get("schema_confidence_norm") or _source_contract("treasury_auctions")["schema_confidence_norm"]),
            "freshness_norm": float(fetch_result.get("freshness_norm") or 0.0),
            "provenance": dict(fetch_result.get("provenance") or {}),
        }

    auction_tail_bps = 0.0
    auction_window = 0.0
    latest_rows: List[Dict[str, Any]] = []
    for row in rows[:12]:
        if not isinstance(row, dict):
            continue
        auction_date = _parse_ts(row.get("auction_date"))
        if auction_date is not None:
            days_since = max((now_utc - auction_date).total_seconds() / 86400.0, 0.0)
            auction_window = max(auction_window, 1.0 - _clamp01(days_since / 7.0))
        tail = max(
            abs(_safe_float(row.get("tail"), 0.0)),
            abs(_safe_float(row.get("auction_tail"), 0.0)),
            abs(_safe_float(row.get("tail_bps"), 0.0)),
        )
        auction_tail_bps = max(auction_tail_bps, tail)
        latest_rows.append(
            {
                "security_type": row.get("security_type"),
                "security_term": row.get("security_term"),
                "auction_date": row.get("auction_date"),
                "tail_bps": tail,
                "high_yield": _safe_float(row.get("high_yield"), 0.0),
                "bid_to_cover": _safe_float(row.get("bid_to_cover_ratio"), 0.0),
            }
        )
    return {
        "ok": True,
        "rows": latest_rows,
        "auction_tail_bps": float(auction_tail_bps),
        "auction_window_norm": float(_clamp01(auction_window)),
        "url": url,
        "source_confidence_norm": float(fetch_result.get("source_confidence_norm") or _source_contract("treasury_auctions")["source_confidence_norm"]),
        "schema_confidence_norm": float(fetch_result.get("schema_confidence_norm") or _source_contract("treasury_auctions")["schema_confidence_norm"]),
        "freshness_norm": float(fetch_result.get("freshness_norm") or 0.0),
        "provenance": dict(fetch_result.get("provenance") or {}),
    }


def _fetch_finra_short_volume(*, symbols: set[str], timeout_seconds: float, max_days: int) -> Dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    per_symbol: Dict[str, Dict[str, float]] = defaultdict(lambda: {"short_volume": 0.0, "total_volume": 0.0, "days": 0.0})
    errors: List[str] = []
    fetched = 0
    last_fetch_result: Dict[str, Any] = {}
    for offset in range(max(int(max_days), 1)):
        day = now_utc - timedelta(days=offset)
        stamp = day.strftime("%Y%m%d")
        url = FINRA_REGSHO_URL.format(stamp=stamp)
        try:
            fetch_result = _fetch_text_result(url, source_name="finra_short_volume", timeout=timeout_seconds)
            if not bool(fetch_result.get("ok", False)):
                raise RuntimeError(str(fetch_result.get("error") or "http_text_failed"))
            last_fetch_result = dict(fetch_result)
            text = str(fetch_result.get("text") or "")
        except (HTTPError, URLError, RuntimeError, TimeoutError, ValueError) as exc:
            errors.append(f"{stamp}:{exc}")
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        header = [col.strip() for col in lines[0].split("|")]
        idx = {name: pos for pos, name in enumerate(header)}
        sym_idx = idx.get("Symbol")
        short_idx = idx.get("ShortVolume")
        short_ex_idx = idx.get("ShortExemptVolume")
        total_idx = idx.get("TotalVolume")
        if sym_idx is None or short_idx is None or total_idx is None:
            continue
        fetched += 1
        for line in lines[1:]:
            parts = line.split("|")
            if sym_idx >= len(parts) or short_idx >= len(parts) or total_idx >= len(parts):
                continue
            symbol = str(parts[sym_idx]).strip().upper()
            if symbols and symbol not in symbols:
                continue
            short_volume = _safe_float(parts[short_idx], 0.0) + (_safe_float(parts[short_ex_idx], 0.0) if short_ex_idx is not None and short_ex_idx < len(parts) else 0.0)
            total_volume = _safe_float(parts[total_idx], 0.0)
            if total_volume <= 0.0:
                continue
            per_symbol[symbol]["short_volume"] += short_volume
            per_symbol[symbol]["total_volume"] += total_volume
            per_symbol[symbol]["days"] += 1.0
    rows: Dict[str, Dict[str, float]] = {}
    for symbol, acc in per_symbol.items():
        rows[symbol] = {
            "short_volume_ratio": float(acc["short_volume"] / max(acc["total_volume"], 1.0)),
            "days": float(acc["days"]),
        }
    contract = _source_contract("finra_short_volume")
    return {
        "ok": fetched > 0,
        "rows": rows,
        "fetched_days": fetched,
        "errors": errors[-5:],
        "source_confidence_norm": float(last_fetch_result.get("source_confidence_norm") or contract["source_confidence_norm"]),
        "schema_confidence_norm": float(last_fetch_result.get("schema_confidence_norm") or contract["schema_confidence_norm"]),
        "freshness_norm": float(last_fetch_result.get("freshness_norm") or (1.0 if fetched > 0 else 0.0)),
        "provenance": dict(last_fetch_result.get("provenance") or {}),
    }


def _aggregate_global_features(*, local_micro: Mapping[str, Mapping[str, float]], short_volume: Mapping[str, Any], treasury: Mapping[str, Any]) -> Dict[str, float]:
    symbol_rows = list(local_micro.values())
    premarket = max((float(row.get("market_micro_premarket_pressure_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    short_rows = short_volume.get("rows") if isinstance(short_volume.get("rows"), Mapping) else {}
    opening = max((float(row.get("market_micro_opening_auction_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    opening_imbalance = max((abs(float(row.get("market_micro_opening_auction_imbalance_norm", 0.5) or 0.5) - 0.5) * 2.0 for row in symbol_rows), default=0.0)
    opening_drive = max((float(row.get("market_micro_opening_drive_pressure_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    power_hour = max((float(row.get("market_micro_power_hour_pressure_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    closing = max((float(row.get("market_micro_closing_auction_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    closing_imbalance = max((abs(float(row.get("market_micro_closing_auction_imbalance_norm", 0.5) or 0.5) - 0.5) * 2.0 for row in symbol_rows), default=0.0)
    closing_cross = max((float(row.get("market_micro_closing_cross_pressure_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    auction_print = max((float(row.get("market_micro_auction_print_pressure_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    relative_vol = max((float(row.get("market_micro_relative_volume_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    options_flow = max((float(row.get("market_micro_options_flow_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    credit_flow = max((float(row.get("market_micro_credit_flow_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    gap_continuation = max((float(row.get("market_micro_gap_continuation_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    reversal_risk = max((float(row.get("market_micro_reversal_risk_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    trend_persistence = max((float(row.get("market_micro_trend_persistence_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    range_expansion = max((float(row.get("market_micro_range_expansion_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    block_trade = max((float(row.get("market_micro_block_trade_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    ssr_active = max((float(row.get("market_micro_ssr_active_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    dark_pool = max((float(row.get("market_micro_dark_pool_pressure_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    off_exchange = max((float(row.get("market_micro_off_exchange_share_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    spread_regime = max((float(row.get("market_micro_spread_regime_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    spread_widening = max((float(row.get("market_micro_spread_widening_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    queue_depth_decay = max((float(row.get("market_micro_queue_depth_decay_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    depth_collapse = max((float(row.get("market_micro_depth_collapse_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    quote_fade_rate = max((float(row.get("market_micro_quote_fade_rate_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    tradeability = min((float(row.get("market_micro_tradeability_score_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    session_open = max((float(row.get("market_micro_session_open_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    session_midday = max((float(row.get("market_micro_session_midday_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    session_power = max((float(row.get("market_micro_session_power_hour_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    overnight_gap = max((float(row.get("market_micro_overnight_gap_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    post_event_drift = max((float(row.get("market_micro_post_event_drift_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    etf_nav = max((float(row.get("etf_nav_premium_discount_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    etf_creation = max((float(row.get("etf_creation_redemption_stress_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    etf_liquidity = max((float(row.get("etf_primary_secondary_liquidity_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    etf_basket = max((float(row.get("etf_underlying_basket_stress_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    etf_family_flow = max((abs(float(row.get("etf_fund_family_flow_norm", 0.5) or 0.5) - 0.5) * 2.0 for row in symbol_rows), default=0.0)
    etf_family_creation = max((float(row.get("etf_fund_family_creation_pressure_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    order_flow = max((abs(float(row.get("market_micro_order_flow_imbalance_norm", 0.5) or 0.5) - 0.5) * 2.0 for row in symbol_rows), default=0.0)
    short_pressure = max((_clamp01((float((value or {}).get("short_volume_ratio", 0.0) or 0.0) - 0.45) / 0.20) for value in short_rows.values() if isinstance(value, Mapping)), default=0.0)
    auction_stress = max(
        _clamp01(float(treasury.get("auction_window_norm", 0.0) or 0.0)),
        _clamp01(float(treasury.get("auction_tail_bps", 0.0) or 0.0) / 6.0),
    )
    return {
        "market_micro_premarket_pressure_norm": _clamp01(premarket),
        "market_micro_opening_auction_norm": _clamp01(opening),
        "market_micro_opening_auction_imbalance_norm": _signed_centered_norm(opening_imbalance, 1.0),
        "market_micro_opening_drive_pressure_norm": _clamp01(opening_drive),
        "market_micro_power_hour_pressure_norm": _clamp01(power_hour),
        "market_micro_closing_auction_norm": _clamp01(closing),
        "market_micro_closing_auction_imbalance_norm": _signed_centered_norm(closing_imbalance, 1.0),
        "market_micro_closing_cross_pressure_norm": _clamp01(closing_cross),
        "market_micro_auction_print_pressure_norm": _clamp01(auction_print),
        "market_micro_relative_volume_norm": _clamp01(relative_vol),
        "market_micro_order_flow_imbalance_norm": _signed_centered_norm(order_flow, 1.0),
        "market_micro_options_flow_norm": _clamp01(options_flow),
        "market_micro_short_pressure_norm": _clamp01(short_pressure),
        "market_micro_credit_flow_norm": _clamp01(max(credit_flow, auction_stress)),
        "market_micro_gap_continuation_norm": _clamp01(gap_continuation),
        "market_micro_reversal_risk_norm": _clamp01(reversal_risk),
        "market_micro_trend_persistence_norm": _clamp01(trend_persistence),
        "market_micro_range_expansion_norm": _clamp01(range_expansion),
        "market_micro_block_trade_norm": _clamp01(block_trade),
        "market_micro_ssr_active_norm": _clamp01(ssr_active),
        "market_micro_dark_pool_pressure_norm": _clamp01(dark_pool),
        "market_micro_off_exchange_share_norm": _clamp01(off_exchange),
        "market_micro_spread_regime_norm": _clamp01(spread_regime),
        "market_micro_spread_widening_norm": _clamp01(spread_widening),
        "market_micro_queue_depth_decay_norm": _clamp01(queue_depth_decay),
        "market_micro_depth_collapse_norm": _clamp01(depth_collapse),
        "market_micro_quote_fade_rate_norm": _clamp01(quote_fade_rate),
        "market_micro_tradeability_score_norm": _clamp01(tradeability),
        "market_micro_session_open_norm": _clamp01(session_open),
        "market_micro_session_midday_norm": _clamp01(session_midday),
        "market_micro_session_power_hour_norm": _clamp01(session_power),
        "market_micro_overnight_gap_norm": _clamp01(overnight_gap),
        "market_micro_post_event_drift_norm": _clamp01(post_event_drift),
        "etf_nav_premium_discount_norm": _clamp01(etf_nav),
        "etf_creation_redemption_stress_norm": _clamp01(etf_creation),
        "etf_primary_secondary_liquidity_norm": _clamp01(etf_liquidity),
        "etf_underlying_basket_stress_norm": _clamp01(etf_basket),
        "etf_fund_family_flow_norm": _signed_centered_norm(etf_family_flow, 1.0),
        "etf_fund_family_creation_pressure_norm": _clamp01(etf_family_creation),
    }


def _apply_trade_halt_overlay(
    *,
    now_utc: datetime,
    symbol_features: Dict[str, Dict[str, float]],
    global_features: Dict[str, float],
    halt_snapshot: Mapping[str, Any],
) -> None:
    rows = halt_snapshot.get("rows") if isinstance(halt_snapshot.get("rows"), list) else []
    halt_norm = 0.0
    luld_norm = 0.0
    resume_norm = 0.0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        reason = str(row.get("reason") or "").upper()
        halt_dt = _parse_intraday_clock(row.get("halt_time"), now_utc)
        resume_dt = _parse_intraday_clock(row.get("resume_time"), now_utc)
        active = 1.0
        if isinstance(resume_dt, datetime):
            if resume_dt <= now_utc:
                active = _clamp01(1.0 - ((now_utc - resume_dt).total_seconds() / 1800.0))
            resume_window = _clamp01(1.0 - (abs((resume_dt - now_utc).total_seconds()) / 1800.0))
        else:
            resume_window = 0.0
        recent_halt = active
        if isinstance(halt_dt, datetime):
            recent_halt = max(recent_halt, _clamp01(1.0 - ((now_utc - halt_dt).total_seconds() / 7200.0)))
        per_symbol = symbol_features.setdefault(symbol, {})
        per_symbol["market_micro_trade_halt_norm"] = max(float(per_symbol.get("market_micro_trade_halt_norm", 0.0) or 0.0), recent_halt)
        if any(token in reason for token in ("LUDP", "LUDS", "VOLATILITY")):
            per_symbol["market_micro_luld_pause_norm"] = max(float(per_symbol.get("market_micro_luld_pause_norm", 0.0) or 0.0), recent_halt)
            luld_norm = max(luld_norm, recent_halt)
        per_symbol["market_micro_resume_window_norm"] = max(float(per_symbol.get("market_micro_resume_window_norm", 0.0) or 0.0), resume_window)
        base_tradeability = float(per_symbol.get("market_micro_tradeability_score_norm", 0.0) or 0.0)
        per_symbol["market_micro_tradeability_score_norm"] = _clamp01(base_tradeability * (1.0 - min((0.60 * recent_halt) + (0.40 * float(per_symbol.get("market_micro_luld_pause_norm", 0.0) or 0.0)), 0.95)))
        halt_norm = max(halt_norm, recent_halt)
        resume_norm = max(resume_norm, resume_window)

    global_features["market_micro_trade_halt_norm"] = max(float(global_features.get("market_micro_trade_halt_norm", 0.0) or 0.0), halt_norm)
    global_features["market_micro_luld_pause_norm"] = max(float(global_features.get("market_micro_luld_pause_norm", 0.0) or 0.0), luld_norm)
    global_features["market_micro_resume_window_norm"] = max(float(global_features.get("market_micro_resume_window_norm", 0.0) or 0.0), resume_norm)
    global_tradeability = float(global_features.get("market_micro_tradeability_score_norm", 0.0) or 0.0)
    global_features["market_micro_tradeability_score_norm"] = _clamp01(global_tradeability * (1.0 - min((0.60 * halt_norm) + (0.40 * luld_norm), 0.95)))


def collect(args: argparse.Namespace) -> int:
    _bootstrap_env()
    now_utc = datetime.now(timezone.utc)
    default_symbols = _default_symbol_list()
    raw_symbols = args.symbols or ",".join(default_symbols)
    symbols = {token.strip().upper() for token in raw_symbols.split(",") if token.strip()}
    external_root = PROJECT_ROOT / "exports" / "external_context"
    health_root = PROJECT_ROOT / "governance" / "health"

    status: Dict[str, Any] = {"timestamp_utc": now_utc.isoformat(), "ok": True, "sources": {}, "source_contracts": {}}
    local_micro = _aggregate_local_micro_context(PROJECT_ROOT, lookback_days=args.lookback_days, symbols=symbols)
    status["sources"]["local_micro"] = {
        "ok": bool(local_micro),
        "symbol_count": len(local_micro),
        "required": True,
        "contract_participates": True,
        **_source_contract("local_micro"),
        "freshness_norm": 1.0 if local_micro else 0.0,
    }

    try:
        treasury = _fetch_treasury_auction_context(timeout_seconds=args.timeout_seconds)
    except Exception as exc:
        treasury = {"ok": False, "rows": [], "error": str(exc)}
    status["sources"]["treasury_auctions"] = {
        "ok": bool(treasury.get("ok", False)),
        "rows": len(treasury.get("rows") or []),
        "error": treasury.get("error"),
        "required": True,
        "contract_participates": True,
        "source_confidence_norm": float(treasury.get("source_confidence_norm") or _source_contract("treasury_auctions")["source_confidence_norm"]),
        "schema_confidence_norm": float(treasury.get("schema_confidence_norm") or _source_contract("treasury_auctions")["schema_confidence_norm"]),
        "freshness_norm": float(treasury.get("freshness_norm") or 0.0),
    }

    try:
        short_volume = _fetch_finra_short_volume(symbols=symbols, timeout_seconds=args.timeout_seconds, max_days=args.finra_lookback_days)
    except Exception as exc:
        short_volume = {"ok": False, "rows": {}, "error": str(exc)}
    short_rows = short_volume.get("rows") if isinstance(short_volume.get("rows"), Mapping) else {}
    status["sources"]["finra_short_volume"] = {
        "ok": bool(short_volume.get("ok", False)),
        "symbol_count": len(short_rows),
        "error": short_volume.get("error"),
        "required": True,
        "contract_participates": True,
        "source_confidence_norm": float(short_volume.get("source_confidence_norm") or _source_contract("finra_short_volume")["source_confidence_norm"]),
        "schema_confidence_norm": float(short_volume.get("schema_confidence_norm") or _source_contract("finra_short_volume")["schema_confidence_norm"]),
        "freshness_norm": float(short_volume.get("freshness_norm") or 0.0),
    }

    try:
        trade_halts = _fetch_nasdaq_trade_halts(symbols=symbols, timeout_seconds=args.timeout_seconds)
    except Exception as exc:
        trade_halts = {"ok": False, "rows": [], "error": str(exc), "url": NASDAQ_TRADE_HALTS_URL}
    halt_rows = trade_halts.get("rows") if isinstance(trade_halts.get("rows"), list) else []
    status["sources"]["nasdaq_trade_halts"] = {
        "ok": bool(trade_halts.get("ok", False)),
        "rows": len(halt_rows),
        "all_rows_count": int(trade_halts.get("all_rows_count", 0) or 0),
        "error": trade_halts.get("error"),
        "required": False,
        "contract_participates": False,
        "source_confidence_norm": float(trade_halts.get("source_confidence_norm") or _source_contract("nasdaq_trade_halts")["source_confidence_norm"]),
        "schema_confidence_norm": float(trade_halts.get("schema_confidence_norm") or _source_contract("nasdaq_trade_halts")["schema_confidence_norm"]),
        "freshness_norm": float(trade_halts.get("freshness_norm") or 0.0),
    }

    global_features = _aggregate_global_features(local_micro=local_micro, short_volume=short_volume, treasury=treasury)
    _apply_trade_halt_overlay(
        now_utc=now_utc,
        symbol_features=local_micro if isinstance(local_micro, dict) else {},
        global_features=global_features,
        halt_snapshot=trade_halts,
    )
    symbol_features: Dict[str, Dict[str, float]] = {}
    for symbol in sorted(symbols):
        out = dict(local_micro.get(symbol, {}))
        short_meta = short_rows.get(symbol) if isinstance(short_rows, Mapping) else None
        short_ratio = _safe_float(short_meta.get("short_volume_ratio"), 0.0) if isinstance(short_meta, Mapping) else 0.0
        out["market_micro_short_pressure_norm"] = _clamp01(max((short_ratio - 0.45) / 0.20, 0.0))
        out["market_micro_credit_flow_norm"] = max(
            float(out.get("market_micro_credit_flow_norm", 0.0) or 0.0),
            _clamp01(float(treasury.get("auction_window_norm", 0.0) or 0.0) * 0.65),
        )
        if not out:
            out = dict(global_features)
        out.setdefault("market_micro_trade_halt_norm", 0.0)
        out.setdefault("market_micro_luld_pause_norm", 0.0)
        out.setdefault("market_micro_ssr_active_norm", 0.0)
        out.setdefault("market_micro_resume_window_norm", 0.0)
        out.setdefault("market_micro_dark_pool_pressure_norm", 0.0)
        out.setdefault("market_micro_off_exchange_share_norm", 0.0)
        out.setdefault("market_micro_opening_auction_imbalance_norm", 0.5)
        out.setdefault("market_micro_opening_drive_pressure_norm", 0.0)
        out.setdefault("market_micro_closing_auction_imbalance_norm", 0.5)
        out.setdefault("market_micro_closing_cross_pressure_norm", 0.0)
        out.setdefault("market_micro_auction_print_pressure_norm", 0.0)
        out.setdefault("market_micro_spread_regime_norm", 0.0)
        out.setdefault("market_micro_spread_widening_norm", 0.0)
        out.setdefault("market_micro_queue_depth_decay_norm", 0.0)
        out.setdefault("market_micro_depth_collapse_norm", 0.0)
        out.setdefault("market_micro_quote_fade_rate_norm", 0.0)
        tradeability = float(out.get("market_micro_tradeability_score_norm", 0.0) or 0.0)
        halt_penalty = (0.60 * float(out.get("market_micro_trade_halt_norm", 0.0) or 0.0)) + (0.40 * float(out.get("market_micro_luld_pause_norm", 0.0) or 0.0))
        out["market_micro_tradeability_score_norm"] = _clamp01(max(tradeability, 0.0) * (1.0 - min(halt_penalty, 0.95)))
        out.setdefault("market_micro_session_open_norm", 0.0)
        out.setdefault("market_micro_session_midday_norm", 0.0)
        out.setdefault("market_micro_session_power_hour_norm", 0.0)
        out.setdefault("market_micro_overnight_gap_norm", 0.0)
        out.setdefault("market_micro_post_event_drift_norm", 0.0)
        out.setdefault("etf_nav_premium_discount_norm", 0.0)
        out.setdefault("etf_creation_redemption_stress_norm", 0.0)
        out.setdefault("etf_primary_secondary_liquidity_norm", 0.0)
        out.setdefault("etf_underlying_basket_stress_norm", 0.0)
        out.setdefault("etf_fund_family_flow_norm", 0.0)
        out.setdefault("etf_fund_family_creation_pressure_norm", 0.0)
        symbol_features[symbol] = out

    payload = {
        "timestamp_utc": now_utc.isoformat(),
        "provider": "market_micro_context",
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
        "derived": {
            "global_features": global_features,
            "symbol_features": symbol_features,
            "treasury_auctions": treasury,
            "finra_short_volume": short_volume,
            "nasdaq_trade_halts": trade_halts,
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
    status["ok"] = any(bool(src.get("ok")) for src in status["sources"].values())

    if not args.test_only:
        _write_json(external_root / "market_micro_latest.json", payload)
        _write_json(health_root / "market_micro_sync_latest.json", status)

    if args.json:
        print(json.dumps(status, ensure_ascii=True))
    else:
        print(
            "market_micro_context "
            f"ok={status['ok']} "
            f"local_symbols={len(local_micro)} "
            f"short_symbols={len(short_rows)} "
            f"auction_rows={len(treasury.get('rows') or [])}"
        )
    return 0 if status["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect free market microstructure and trading context.")
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    parser.add_argument("--lookback-days", type=int, default=21)
    parser.add_argument("--finra-lookback-days", type=int, default=15)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
