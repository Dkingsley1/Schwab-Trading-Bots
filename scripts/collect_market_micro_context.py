#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "schwab-trading-bot/1.0"
TREASURY_AUCTIONS_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
FINRA_REGSHO_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{stamp}.txt"
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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _http_json(url: str, *, timeout: float = 12.0) -> Any:
    req = Request(url=url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(req, timeout=max(float(timeout), 1.0)) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _http_text(url: str, *, timeout: float = 12.0) -> str:
    req = Request(url=url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urlopen(req, timeout=max(float(timeout), 1.0)) as resp:
        return resp.read().decode("utf-8", "replace")


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


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _bootstrap_env() -> None:
    _load_env_file(PROJECT_ROOT / ".env")
    _load_env_file(PROJECT_ROOT / "config" / ".env")
    _load_env_file(PROJECT_ROOT / "config" / ".env.live")


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
        gamma = abs(_safe_float(feats.get("options_gamma_exposure_norm"), 0.0))
        unusual = abs(_safe_float(feats.get("options_unusual_flow_norm"), 0.0))
        put_call = abs(_safe_float(feats.get("options_put_call_oi_ratio_norm"), 0.5) - 0.5) * 2.0
        hy_ig = abs(_safe_float(feats.get("bond_hy_ig_flow_norm"), 0.5) - 0.5) * 2.0
        nav_stress = abs(_safe_float(feats.get("bond_nav_stress_norm"), 0.0))
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

    out: Dict[str, Dict[str, float]] = {}
    for symbol, acc in per_symbol.items():
        c = counts.get(symbol, {})
        premarket = acc.get("premarket_pressure", 0.0) / max(c.get("premarket_pressure", 1.0), 1.0)
        opening = acc.get("opening_auction", 0.0) / max(c.get("opening_auction", 1.0), 1.0)
        power_hour = acc.get("power_hour_pressure", 0.0) / max(c.get("power_hour_pressure", 1.0), 1.0)
        closing = acc.get("closing_auction", 0.0) / max(c.get("closing_auction", 1.0), 1.0)
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
        relative_volume = 0.0
        if other_vol > 0.0:
            relative_volume = min(midday_vol / max(other_vol, 1e-8), 2.0) / 2.0
        out[symbol] = {
            "market_micro_premarket_pressure_norm": _clamp01(premarket),
            "market_micro_opening_auction_norm": _clamp01(opening),
            "market_micro_power_hour_pressure_norm": _clamp01(power_hour),
            "market_micro_closing_auction_norm": _clamp01(closing),
            "market_micro_relative_volume_norm": _clamp01(relative_volume),
            "market_micro_order_flow_imbalance_norm": _signed_centered_norm(order_flow, 1.0),
            "market_micro_options_flow_norm": _clamp01(options_flow),
            "market_micro_credit_flow_norm": _clamp01(credit_flow),
            "market_micro_gap_continuation_norm": _clamp01(gap_continuation),
            "market_micro_reversal_risk_norm": _clamp01(reversal_risk),
            "market_micro_trend_persistence_norm": _clamp01(trend_persistence),
            "market_micro_range_expansion_norm": _clamp01(range_expansion),
            "market_micro_block_trade_norm": _clamp01(block_trade),
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
    payload = _http_json(url, timeout=timeout_seconds)
    rows = payload.get("data") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return {"ok": False, "rows": [], "error": "unexpected_response_shape", "url": url}

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
    }


def _fetch_finra_short_volume(*, symbols: set[str], timeout_seconds: float, max_days: int) -> Dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    per_symbol: Dict[str, Dict[str, float]] = defaultdict(lambda: {"short_volume": 0.0, "total_volume": 0.0, "days": 0.0})
    errors: List[str] = []
    fetched = 0
    for offset in range(max(int(max_days), 1)):
        day = now_utc - timedelta(days=offset)
        stamp = day.strftime("%Y%m%d")
        url = FINRA_REGSHO_URL.format(stamp=stamp)
        try:
            text = _http_text(url, timeout=timeout_seconds)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
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
    return {"ok": fetched > 0, "rows": rows, "fetched_days": fetched, "errors": errors[-5:]}


def _aggregate_global_features(*, local_micro: Mapping[str, Mapping[str, float]], short_volume: Mapping[str, Any], treasury: Mapping[str, Any]) -> Dict[str, float]:
    symbol_rows = list(local_micro.values())
    premarket = max((float(row.get("market_micro_premarket_pressure_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    short_rows = short_volume.get("rows") if isinstance(short_volume.get("rows"), Mapping) else {}
    opening = max((float(row.get("market_micro_opening_auction_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    power_hour = max((float(row.get("market_micro_power_hour_pressure_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    closing = max((float(row.get("market_micro_closing_auction_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    relative_vol = max((float(row.get("market_micro_relative_volume_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    options_flow = max((float(row.get("market_micro_options_flow_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    credit_flow = max((float(row.get("market_micro_credit_flow_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    gap_continuation = max((float(row.get("market_micro_gap_continuation_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    reversal_risk = max((float(row.get("market_micro_reversal_risk_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    trend_persistence = max((float(row.get("market_micro_trend_persistence_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    range_expansion = max((float(row.get("market_micro_range_expansion_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    block_trade = max((float(row.get("market_micro_block_trade_norm", 0.0) or 0.0) for row in symbol_rows), default=0.0)
    order_flow = max((abs(float(row.get("market_micro_order_flow_imbalance_norm", 0.5) or 0.5) - 0.5) * 2.0 for row in symbol_rows), default=0.0)
    short_pressure = max((_clamp01((float((value or {}).get("short_volume_ratio", 0.0) or 0.0) - 0.45) / 0.20) for value in short_rows.values() if isinstance(value, Mapping)), default=0.0)
    auction_stress = max(
        _clamp01(float(treasury.get("auction_window_norm", 0.0) or 0.0)),
        _clamp01(float(treasury.get("auction_tail_bps", 0.0) or 0.0) / 6.0),
    )
    return {
        "market_micro_premarket_pressure_norm": _clamp01(premarket),
        "market_micro_opening_auction_norm": _clamp01(opening),
        "market_micro_power_hour_pressure_norm": _clamp01(power_hour),
        "market_micro_closing_auction_norm": _clamp01(closing),
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
    }


def collect(args: argparse.Namespace) -> int:
    _bootstrap_env()
    now_utc = datetime.now(timezone.utc)
    default_symbols = _default_symbol_list()
    raw_symbols = args.symbols or ",".join(default_symbols)
    symbols = {token.strip().upper() for token in raw_symbols.split(",") if token.strip()}
    external_root = PROJECT_ROOT / "exports" / "external_context"
    health_root = PROJECT_ROOT / "governance" / "health"

    status: Dict[str, Any] = {"timestamp_utc": now_utc.isoformat(), "ok": True, "sources": {}}
    local_micro = _aggregate_local_micro_context(PROJECT_ROOT, lookback_days=args.lookback_days, symbols=symbols)
    status["sources"]["local_micro"] = {"ok": bool(local_micro), "symbol_count": len(local_micro)}

    try:
        treasury = _fetch_treasury_auction_context(timeout_seconds=args.timeout_seconds)
    except Exception as exc:
        treasury = {"ok": False, "rows": [], "error": str(exc)}
    status["sources"]["treasury_auctions"] = {"ok": bool(treasury.get("ok", False)), "rows": len(treasury.get("rows") or []), "error": treasury.get("error")}

    try:
        short_volume = _fetch_finra_short_volume(symbols=symbols, timeout_seconds=args.timeout_seconds, max_days=args.finra_lookback_days)
    except Exception as exc:
        short_volume = {"ok": False, "rows": {}, "error": str(exc)}
    short_rows = short_volume.get("rows") if isinstance(short_volume.get("rows"), Mapping) else {}
    status["sources"]["finra_short_volume"] = {"ok": bool(short_volume.get("ok", False)), "symbol_count": len(short_rows), "error": short_volume.get("error")}

    global_features = _aggregate_global_features(local_micro=local_micro, short_volume=short_volume, treasury=treasury)
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
        symbol_features[symbol] = out

    payload = {
        "timestamp_utc": now_utc.isoformat(),
        "provider": "market_micro_context",
        "derived": {
            "global_features": global_features,
            "symbol_features": symbol_features,
            "treasury_auctions": treasury,
            "finra_short_volume": short_volume,
        },
    }
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
