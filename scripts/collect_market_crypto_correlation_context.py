#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_MIRROR_DEFAULT = Path("/Volumes/BOT_LOGS/schwab_trading_bot")
LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "market_crypto_correlation.lock"

FEATURE_KEYS = [
    "market_crypto_risk_corr_norm",
    "market_crypto_spy_corr_norm",
    "market_crypto_qqq_corr_norm",
    "market_crypto_tlt_corr_norm",
    "market_crypto_uup_inverse_corr_norm",
    "market_crypto_gold_corr_norm",
    "market_crypto_current_alignment_norm",
    "market_crypto_divergence_norm",
    "market_crypto_corr_confidence_norm",
]

_SERIES_METRICS = ("pct_from_close", "mom_5m", "return_1m")
_STOCK_PROXY_SYMBOLS = ("SPY", "QQQ", "IWM", "HYG", "XLK", "XLF", "XLY")
_RISK_OFF_PROXY_SYMBOLS = ("TLT", "IEF", "SHY", "GLD", "UUP")
_CRYPTO_SYMBOLS = ("BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "DOGE-USD", "LINK-USD", "LTC-USD")
_US_EASTERN = ZoneInfo("America/New_York")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _safe_load_json(path: Path, default: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default or {})
    return payload if isinstance(payload, dict) else dict(default or {})


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


def _parse_ts(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value > 1e12:
            value /= 1000.0
        return value if value > 1e9 else None
    text = str(raw).strip()
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


def _norm_symbol(raw: Any) -> str:
    return str(raw or "").strip().upper()


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _file_signature(path: Path) -> dict[str, int]:
    try:
        stat = path.stat()
    except Exception:
        return {"size": -1, "mtime_ns": -1}
    return {"size": int(stat.st_size), "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))}


def _json_clone(value: Any, default: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except Exception:
        return default


def _path_signature_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        signature = _file_signature(path)
        rows.append(
            {
                "path": _path_key(path),
                "size": int(signature.get("size", -1)),
                "mtime_ns": int(signature.get("mtime_ns", -1)),
            }
        )
    return rows


def _serialize_series(series: Mapping[str, Mapping[str, Mapping[int, float]]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for symbol, by_metric in series.items():
        if not isinstance(by_metric, Mapping):
            continue
        out_symbol: dict[str, dict[str, float]] = {}
        for metric, by_bucket in by_metric.items():
            if not isinstance(by_bucket, Mapping):
                continue
            out_symbol[str(metric)] = {
                str(int(bucket)): float(value)
                for bucket, value in by_bucket.items()
                if math.isfinite(_safe_float(value, math.nan))
            }
        if out_symbol:
            out[str(symbol)] = out_symbol
    return out


def _deserialize_series(payload: Any) -> dict[str, dict[str, dict[int, float]]]:
    out: dict[str, dict[str, dict[int, float]]] = {}
    if not isinstance(payload, Mapping):
        return out
    for symbol, by_metric in payload.items():
        if not isinstance(by_metric, Mapping):
            continue
        out_symbol: dict[str, dict[int, float]] = {}
        for metric, by_bucket in by_metric.items():
            if not isinstance(by_bucket, Mapping):
                continue
            out_metric: dict[int, float] = {}
            for bucket, value in by_bucket.items():
                try:
                    bucket_int = int(bucket)
                except Exception:
                    continue
                coerced = _safe_float(value, math.nan)
                if math.isfinite(coerced):
                    out_metric[bucket_int] = coerced
            if out_metric:
                out_symbol[str(metric)] = out_metric
        if out_symbol:
            out[str(symbol)] = out_symbol
    return out


def _normalize_latest(payload: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not isinstance(payload, Mapping):
        return out
    for symbol, row in payload.items():
        if not isinstance(row, Mapping):
            continue
        out[str(symbol)] = {
            "ts": _safe_float(row.get("ts"), 0.0),
            "last_price": _safe_float(row.get("last_price"), 0.0),
            "pct_from_close": _safe_float(row.get("pct_from_close"), 0.0),
            "mom_5m": _safe_float(row.get("mom_5m"), 0.0),
            "return_1m": _safe_float(row.get("return_1m"), 0.0),
        }
    return out


def _is_crypto_symbol(symbol: str) -> bool:
    upper = _norm_symbol(symbol)
    return upper.endswith("-USD") or upper in {"BTC", "ETH", "SOL", "AVAX", "DOGE", "LINK", "LTC"}


def _corr_to_norm(corr: float | None) -> float:
    if corr is None or not math.isfinite(corr):
        return 0.5
    return _clamp01(0.5 + (0.5 * corr))


def _inverse_corr_to_norm(corr: float | None) -> float:
    if corr is None or not math.isfinite(corr):
        return 0.5
    return _clamp01(0.5 - (0.5 * corr))


def _series_snapshot(payload: Any) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    out: dict[str, float] = {}
    for key in ("last_price", "pct_from_close", "mom_5m", "return_1m"):
        value = _safe_float(payload.get(key), math.nan)
        if math.isfinite(value):
            out[key] = value
    return out


def _iter_repo_roots(project_root: Path, extra_roots: Iterable[Path] = ()) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in [project_root, *extra_roots]:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            resolved = candidate.expanduser()
        token = str(resolved)
        if token in seen:
            continue
        seen.add(token)
        if resolved.exists():
            roots.append(resolved)
    return roots


def _is_crypto_shadow_dir(name: str) -> bool:
    lowered = str(name or "").strip().lower()
    return lowered.startswith("shadow_crypto") or "_crypto" in lowered


def _select_master_control_paths(
    project_root: Path,
    *,
    lookback_days: int,
    extra_roots: Iterable[Path],
    include_crypto: bool = False,
    crypto_only: bool = False,
) -> list[Path]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max(int(lookback_days), 1))).strftime("%Y%m%d")
    selected: dict[str, Path] = {}
    for root in _iter_repo_roots(project_root, extra_roots):
        governance_root = root / "governance"
        if not governance_root.exists():
            continue
        for path in governance_root.glob("shadow*/master_control_*.jsonl"):
            is_crypto_path = _is_crypto_shadow_dir(path.parent.name)
            if crypto_only and not is_crypto_path:
                continue
            if (not include_crypto) and is_crypto_path:
                continue
            stamp = path.stem.rsplit("_", 1)[-1]
            if stamp < cutoff:
                continue
            try:
                rel = str(path.relative_to(root))
            except Exception:
                rel = str(path)
            current = selected.get(rel)
            if current is None:
                selected[rel] = path
                continue
            try:
                current_stat = current.stat()
                new_stat = path.stat()
            except Exception:
                selected[rel] = path
                continue
            if (new_stat.st_size, new_stat.st_mtime) >= (current_stat.st_size, current_stat.st_mtime):
                selected[rel] = path
    return sorted(selected.values())


def _select_master_control_csv_paths(project_root: Path, *, lookback_days: int, extra_roots: Iterable[Path]) -> list[Path]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max(int(lookback_days), 1))).strftime("%Y%m%d")
    selected: dict[str, Path] = {}
    for root in _iter_repo_roots(project_root, extra_roots):
        csv_root = root / "exports" / "csv"
        if not csv_root.exists():
            continue
        for path in csv_root.glob("master_control_*.csv"):
            stamp = path.stem.rsplit("_", 1)[-1]
            if stamp < cutoff:
                continue
            try:
                rel = str(path.relative_to(root))
            except Exception:
                rel = str(path)
            current = selected.get(rel)
            if current is None:
                selected[rel] = path
                continue
            try:
                current_stat = current.stat()
                new_stat = path.stat()
            except Exception:
                selected[rel] = path
                continue
            if (new_stat.st_size, new_stat.st_mtime) >= (current_stat.st_size, current_stat.st_mtime):
                selected[rel] = path
    return sorted(selected.values())


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _record_series_point(
    bucketed: dict[str, dict[str, dict[int, list[float]]]],
    latest: dict[str, dict[str, float]],
    *,
    ts: float,
    bucket_seconds: int,
    symbol: str,
    snapshot: Mapping[str, Any],
) -> None:
    sym = _norm_symbol(symbol)
    if not sym:
        return
    snap = _series_snapshot(snapshot)
    if not snap:
        return
    bucket = int(ts // bucket_seconds) * bucket_seconds
    for key in _SERIES_METRICS:
        if key in snap:
            bucketed[sym][key].setdefault(bucket, []).append(float(snap[key]))
    current_latest = latest.get(sym)
    if current_latest is None or ts >= _safe_float(current_latest.get("ts"), 0.0):
        latest[sym] = {
            "ts": float(ts),
            "last_price": float(snap.get("last_price", 0.0) or 0.0),
            "pct_from_close": float(snap.get("pct_from_close", 0.0) or 0.0),
            "mom_5m": float(snap.get("mom_5m", 0.0) or 0.0),
            "return_1m": float(snap.get("return_1m", 0.0) or 0.0),
        }


def _collect_series(
    *,
    paths: Iterable[Path],
    bucket_seconds: int,
) -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, dict[str, float]], dict[str, Any]]:
    bucketed: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(dict))
    latest: dict[str, dict[str, float]] = {}
    rows_scanned = 0
    symbols_seen: set[str] = set()
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    rows_scanned += 1
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    ts = _parse_ts(row.get("timestamp_utc"))
                    if ts is None:
                        continue
                    symbol = _norm_symbol(row.get("symbol"))
                    if symbol:
                        symbols_seen.add(symbol)
                    market = row.get("market")
                    if isinstance(market, Mapping):
                        _record_series_point(bucketed, latest, ts=ts, bucket_seconds=bucket_seconds, symbol=symbol, snapshot=market)
                    context_market = row.get("context_market")
                    if isinstance(context_market, Mapping):
                        for ctx_symbol, ctx_snapshot in context_market.items():
                            symbols_seen.add(_norm_symbol(ctx_symbol))
                            if isinstance(ctx_snapshot, Mapping):
                                _record_series_point(
                                    bucketed,
                                    latest,
                                    ts=ts,
                                    bucket_seconds=bucket_seconds,
                                    symbol=ctx_symbol,
                                    snapshot=ctx_snapshot,
                                )
        except Exception:
            continue

    finalized: dict[str, dict[str, dict[int, float]]] = {}
    for symbol, by_metric in bucketed.items():
        finalized[symbol] = {}
        for metric, by_bucket in by_metric.items():
            finalized[symbol][metric] = {bucket: _median(values) for bucket, values in by_bucket.items() if values}
    return finalized, latest, {
        "rows_scanned": rows_scanned,
        "symbols_seen": sorted(symbols_seen),
    }


def _collect_series_jsonl_path(
    *,
    path: Path,
    bucket_seconds: int,
) -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, dict[str, float]], dict[str, Any]]:
    return _collect_series(paths=[path], bucket_seconds=bucket_seconds)


def _collect_series_from_csv(
    *,
    paths: Iterable[Path],
    bucket_seconds: int,
) -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, dict[str, float]], dict[str, Any]]:
    bucketed: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(dict))
    latest: dict[str, dict[str, float]] = {}
    rows_scanned = 0
    symbols_seen: set[str] = set()
    for path in paths:
        try:
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    rows_scanned += 1
                    ts = _parse_ts(row.get("timestamp_utc"))
                    if ts is None:
                        continue
                    symbol = _norm_symbol(row.get("symbol"))
                    if symbol:
                        symbols_seen.add(symbol)
                    try:
                        market = json.loads(row.get("market") or "{}")
                    except Exception:
                        market = {}
                    if isinstance(market, Mapping):
                        _record_series_point(bucketed, latest, ts=ts, bucket_seconds=bucket_seconds, symbol=symbol, snapshot=market)
                    try:
                        context_market = json.loads(row.get("context_market") or "{}")
                    except Exception:
                        context_market = {}
                    if isinstance(context_market, Mapping):
                        for ctx_symbol, ctx_snapshot in context_market.items():
                            symbols_seen.add(_norm_symbol(ctx_symbol))
                            if isinstance(ctx_snapshot, Mapping):
                                _record_series_point(
                                    bucketed,
                                    latest,
                                    ts=ts,
                                    bucket_seconds=bucket_seconds,
                                    symbol=ctx_symbol,
                                    snapshot=ctx_snapshot,
                                )
        except Exception:
            continue

    finalized: dict[str, dict[str, dict[int, float]]] = {}
    for symbol, by_metric in bucketed.items():
        finalized[symbol] = {}
        for metric, by_bucket in by_metric.items():
            finalized[symbol][metric] = {bucket: _median(values) for bucket, values in by_bucket.items() if values}
    return finalized, latest, {
        "rows_scanned": rows_scanned,
        "symbols_seen": sorted(symbols_seen),
    }


def _collect_series_csv_path(
    *,
    path: Path,
    bucket_seconds: int,
) -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, dict[str, float]], dict[str, Any]]:
    return _collect_series_from_csv(paths=[path], bucket_seconds=bucket_seconds)


def _collect_series_cached(
    *,
    paths: Iterable[Path],
    bucket_seconds: int,
    cache_entries: Mapping[str, Any],
    loader: Any,
) -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, dict[str, float]], dict[str, Any], dict[str, Any]]:
    merged_series: dict[str, dict[str, dict[int, float]]] = {}
    merged_latest: dict[str, dict[str, float]] = {}
    rows_scanned = 0
    symbols_seen: set[str] = set()
    cache_hits = 0
    cache_misses = 0
    rebuilt_entries: dict[str, Any] = {}

    for path in paths:
        key = _path_key(path)
        signature = _file_signature(path)
        cached = cache_entries.get(key) if isinstance(cache_entries, Mapping) else None
        use_cache = (
            isinstance(cached, Mapping)
            and int(_safe_float(cached.get("bucket_seconds"), -1)) == int(bucket_seconds)
            and isinstance(cached.get("signature"), Mapping)
            and int(_safe_float(cached.get("signature", {}).get("size"), -1)) == signature["size"]
            and int(_safe_float(cached.get("signature", {}).get("mtime_ns"), -1)) == signature["mtime_ns"]
        )
        if use_cache:
            series = _deserialize_series(cached.get("series"))
            latest = _normalize_latest(cached.get("latest"))
            meta = {
                "rows_scanned": int(_safe_float(cached.get("rows_scanned"), 0.0)),
                "symbols_seen": list(cached.get("symbols_seen") or []),
            }
            cache_hits += 1
        else:
            series, latest, meta = loader(path=path, bucket_seconds=bucket_seconds)
            cache_misses += 1

        rebuilt_entries[key] = {
            "signature": signature,
            "bucket_seconds": int(bucket_seconds),
            "series": _serialize_series(series),
            "latest": latest,
            "rows_scanned": int(meta.get("rows_scanned", 0) or 0),
            "symbols_seen": sorted(str(symbol) for symbol in meta.get("symbols_seen", []) if str(symbol).strip()),
        }
        merged_series = _merge_series_maps(merged_series, series)
        merged_latest = _merge_latest_maps(merged_latest, latest)
        rows_scanned += int(meta.get("rows_scanned", 0) or 0)
        symbols_seen.update(str(symbol) for symbol in meta.get("symbols_seen", []) if str(symbol).strip())

    return merged_series, merged_latest, {
        "rows_scanned": rows_scanned,
        "symbols_seen": sorted(symbols_seen),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
    }, rebuilt_entries


def _merge_series_maps(
    left: dict[str, dict[str, dict[int, float]]],
    right: dict[str, dict[str, dict[int, float]]],
) -> dict[str, dict[str, dict[int, float]]]:
    merged = {symbol: {metric: dict(values) for metric, values in by_metric.items()} for symbol, by_metric in left.items()}
    for symbol, by_metric in right.items():
        current_symbol = merged.setdefault(symbol, {})
        for metric, values in by_metric.items():
            current_metric = current_symbol.setdefault(metric, {})
            for bucket, value in values.items():
                if bucket in current_metric:
                    current_metric[bucket] = (float(current_metric[bucket]) + float(value)) / 2.0
                else:
                    current_metric[bucket] = float(value)
    return merged


def _merge_latest_maps(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    merged = {symbol: dict(values) for symbol, values in left.items()}
    for symbol, row in right.items():
        current = merged.get(symbol)
        if current is None or _safe_float(row.get("ts"), 0.0) >= _safe_float(current.get("ts"), 0.0):
            merged[symbol] = dict(row)
    return merged


def _aggregate_series(
    series: dict[str, dict[str, dict[int, float]]],
    symbols: Iterable[str],
    metric: str,
) -> dict[int, float]:
    out: dict[int, list[float]] = defaultdict(list)
    for symbol in symbols:
        for bucket, value in series.get(_norm_symbol(symbol), {}).get(metric, {}).items():
            if math.isfinite(value):
                out[bucket].append(float(value))
    return {bucket: sum(values) / len(values) for bucket, values in out.items() if values}


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 1e-12 or var_y <= 1e-12:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def _series_correlation(
    left: dict[int, float],
    right: dict[int, float],
    *,
    min_points: int,
) -> tuple[float | None, int]:
    keys = sorted(set(left) & set(right))
    if len(keys) < 2:
        return None, len(keys)
    xs = [float(left[key]) for key in keys]
    ys = [float(right[key]) for key in keys]
    corr = _pearson(xs, ys)
    if corr is None or len(keys) < min_points:
        return corr, len(keys)
    return corr, len(keys)


def _series_correlation_asof(
    left: dict[int, float],
    right: dict[int, float],
    *,
    min_points: int,
    max_lag_seconds: int,
) -> tuple[float | None, int]:
    if not left or not right:
        return None, 0
    left_keys = sorted(left)
    right_keys = sorted(right)
    xs: list[float] = []
    ys: list[float] = []
    i = 0
    for right_key in right_keys:
        while i + 1 < len(left_keys) and left_keys[i + 1] <= right_key:
            i += 1
        candidate_keys = [left_keys[i]]
        if i + 1 < len(left_keys):
            candidate_keys.append(left_keys[i + 1])
        if i > 0:
            candidate_keys.append(left_keys[i - 1])
        best_key = None
        best_gap = None
        for candidate in candidate_keys:
            gap = abs(int(candidate) - int(right_key))
            if gap > max_lag_seconds:
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_key = candidate
        if best_key is None:
            continue
        xs.append(float(left[best_key]))
        ys.append(float(right[right_key]))
    if len(xs) < 2:
        return None, len(xs)
    corr = _pearson(xs, ys)
    if corr is None or len(xs) < min_points:
        return corr, len(xs)
    return corr, len(xs)


def _best_pair_correlation(
    series: dict[str, dict[str, dict[int, float]]],
    left_symbol: str,
    right_symbol: str,
    *,
    min_points: int,
    max_lag_seconds: int = 72 * 3600,
) -> dict[str, Any]:
    left = series.get(_norm_symbol(left_symbol), {})
    right = series.get(_norm_symbol(right_symbol), {})
    best = {"corr": None, "points": 0, "metric": "", "usable": False, "mode": "exact"}
    for metric in ("return_1m", "mom_5m", "pct_from_close"):
        corr, points = _series_correlation(left.get(metric, {}), right.get(metric, {}), min_points=min_points)
        if points > int(best["points"]):
            best = {
                "corr": corr,
                "points": points,
                "metric": metric,
                "usable": corr is not None and points >= min_points,
                "mode": "exact",
            }
        elif points == int(best["points"]) and corr is not None and best.get("corr") is None:
            best = {
                "corr": corr,
                "points": points,
                "metric": metric,
                "usable": corr is not None and points >= min_points,
                "mode": "exact",
            }
    if best.get("usable"):
        return best
    for metric in ("return_1m", "mom_5m", "pct_from_close"):
        corr, points = _series_correlation_asof(
            left.get(metric, {}),
            right.get(metric, {}),
            min_points=min_points,
            max_lag_seconds=max_lag_seconds,
        )
        if points > int(best["points"]):
            best = {
                "corr": corr,
                "points": points,
                "metric": metric,
                "usable": corr is not None and points >= min_points,
                "mode": "asof",
            }
        elif points == int(best["points"]) and corr is not None and best.get("corr") is None:
            best = {
                "corr": corr,
                "points": points,
                "metric": metric,
                "usable": corr is not None and points >= min_points,
                "mode": "asof",
            }
    return best


def _corr_confidence_norm(points: int) -> float:
    if points <= 0:
        return 0.0
    if points <= 2:
        return 0.05
    return _clamp01((float(points) - 2.0) / 18.0)


def _carry_forward_confidence_scale(ts_raw: Any, now: datetime, *, max_age_hours: float = 120.0) -> float:
    ts = _parse_ts(ts_raw)
    if ts is None:
        return 0.2
    age_hours = max((now.timestamp() - ts) / 3600.0, 0.0)
    return max(0.2, _clamp01(1.0 - (age_hours / max(max_age_hours, 1.0))))


def _pair_confidence(row: Mapping[str, Any]) -> float:
    base = _corr_confidence_norm(int(_safe_float(row.get("points"), 0.0)))
    if str(row.get("mode") or "exact").strip().lower() == "asof":
        return min(base, 0.55)
    return base


def _current_alignment_norm(left_snapshot: Mapping[str, Any], right_snapshot: Mapping[str, Any]) -> float:
    left_move = _safe_float(left_snapshot.get("pct_from_close"), 0.0)
    right_move = _safe_float(right_snapshot.get("pct_from_close"), 0.0)
    if abs(left_move) < 1e-9 and abs(right_move) < 1e-9:
        left_move = _safe_float(left_snapshot.get("mom_5m"), 0.0)
        right_move = _safe_float(right_snapshot.get("mom_5m"), 0.0)
    if abs(left_move) < 1e-9 and abs(right_move) < 1e-9:
        return 0.5
    same_direction = 1.0 if (left_move * right_move) > 0.0 else (-1.0 if (left_move * right_move) < 0.0 else 0.0)
    magnitude_gap = min(abs(left_move - right_move) / 0.05, 1.0)
    return _clamp01(0.5 + (0.38 * same_direction) + (0.12 * (1.0 - magnitude_gap)))


def _pick_latest(latest: Mapping[str, dict[str, float]], symbols: Iterable[str]) -> dict[str, float]:
    chosen: dict[str, float] = {}
    best_ts = -1.0
    for symbol in symbols:
        row = latest.get(_norm_symbol(symbol))
        if not isinstance(row, Mapping):
            continue
        ts = _safe_float(row.get("ts"), 0.0)
        if ts >= best_ts:
            best_ts = ts
            chosen = dict(row)
    return chosen


def _render_markdown_report(
    *,
    status: Mapping[str, Any],
    pair_rows: list[dict[str, Any]],
    latest: Mapping[str, dict[str, float]],
) -> str:
    lines = [
        "# Market/Crypto Correlation",
        "",
        f"- ok: {bool(status.get('ok', False))}",
        f"- files_scanned: {int(status.get('files_scanned', 0) or 0)}",
        f"- rows_scanned: {int(status.get('rows_scanned', 0) or 0)}",
        f"- stock_symbols_observed: {int(status.get('stock_symbols_observed', 0) or 0)}",
        f"- crypto_symbols_observed: {int(status.get('crypto_symbols_observed', 0) or 0)}",
        f"- aligned_pairs: {int(status.get('aligned_pairs', 0) or 0)}",
        f"- mode: {str(status.get('mode') or 'exact')}",
        "",
        "| Pair | Corr | Points | Metric | Confidence |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for row in pair_rows[:10]:
        corr = row.get("corr")
        lines.append(
            "| {left} vs {right} | {corr} | {points} | {metric} | {confidence:.3f} |".format(
                left=row.get("left"),
                right=row.get("right"),
                corr=("n/a" if corr is None else f"{float(corr):.4f}"),
                points=int(row.get("points", 0) or 0),
                metric=str(row.get("metric") or "n/a"),
                confidence=float(row.get("confidence_norm", 0.0) or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Latest Snapshots",
            "",
            "| Symbol | Pct From Close | Mom 5m | Return 1m |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for symbol in sorted(latest)[:12]:
        row = latest[symbol]
        lines.append(
            "| {symbol} | {pct:.4f} | {mom:.4f} | {ret:.4f} |".format(
                symbol=symbol,
                pct=_safe_float(row.get("pct_from_close"), 0.0),
                mom=_safe_float(row.get("mom_5m"), 0.0),
                ret=_safe_float(row.get("return_1m"), 0.0),
            )
        )
    warnings = status.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


def _is_us_equity_market_hours(now: datetime) -> bool:
    local = now.astimezone(_US_EASTERN)
    if local.weekday() >= 5:
        return False
    minute_of_day = (local.hour * 60) + local.minute
    return (9 * 60 + 30) <= minute_of_day <= (16 * 60)


def collect_market_crypto_correlation_context(
    *,
    project_root: Path,
    lookback_days: int = 5,
    bucket_seconds: int = 300,
    min_points: int = 6,
    extra_roots: Iterable[Path] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = datetime.now(timezone.utc)
    cache_path = project_root / "exports" / "external_context" / "market_crypto_correlation_cache_latest.json"
    cache_payload = _safe_load_json(cache_path, default={})
    cached_jsonl_entries = cache_payload.get("jsonl_file_cache") if isinstance(cache_payload.get("jsonl_file_cache"), Mapping) else {}
    cached_csv_entries = cache_payload.get("csv_file_cache") if isinstance(cache_payload.get("csv_file_cache"), Mapping) else {}
    last_usable = cache_payload.get("last_usable") if isinstance(cache_payload.get("last_usable"), Mapping) else {}
    stock_jsonl_paths = _select_master_control_paths(
        project_root,
        lookback_days=lookback_days,
        extra_roots=extra_roots,
        include_crypto=False,
    )
    crypto_jsonl_paths = _select_master_control_paths(
        project_root,
        lookback_days=lookback_days,
        extra_roots=extra_roots,
        include_crypto=True,
        crypto_only=True,
    )
    jsonl_paths = sorted(
        {str(path): path for path in [*stock_jsonl_paths, *crypto_jsonl_paths]}.values(),
        key=str,
    )
    csv_paths = _select_master_control_csv_paths(project_root, lookback_days=lookback_days, extra_roots=extra_roots)
    effective_bucket_seconds = max(int(bucket_seconds), 60)
    market_hours_bias = _is_us_equity_market_hours(now)
    asof_max_lag_seconds = 72 * 3600
    if market_hours_bias:
        asof_max_lag_seconds = max(effective_bucket_seconds * 2, 15 * 60)
    result_cache_state = {
        "lookback_days": max(int(lookback_days), 1),
        "bucket_seconds": int(effective_bucket_seconds),
        "min_points": max(int(min_points), 3),
        "market_hours_bias": bool(market_hours_bias),
        "asof_max_lag_seconds": int(asof_max_lag_seconds),
        "jsonl_paths": _path_signature_rows(jsonl_paths),
        "csv_paths": _path_signature_rows(csv_paths),
    }
    cached_result_state = cache_payload.get("last_result_state") if isinstance(cache_payload.get("last_result_state"), Mapping) else {}
    cached_result = cache_payload.get("last_result") if isinstance(cache_payload.get("last_result"), Mapping) else {}
    if cached_result_state == result_cache_state and cached_result:
        cached_status = _json_clone(cached_result.get("status"), {})
        cached_payload = _json_clone(cached_result.get("payload"), {})
        if isinstance(cached_status, dict) and isinstance(cached_payload, dict) and str(cached_status.get("mode") or "") != "carry_forward_last_usable":
            cached_status["timestamp_utc"] = now.isoformat()
            cached_status["cache_result_reused"] = True
            cached_payload["timestamp_utc"] = now.isoformat()
            cached_payload["status"] = cached_status
            return cached_payload, cached_status
    jsonl_series, jsonl_latest, jsonl_meta, rebuilt_jsonl_entries = _collect_series_cached(
        paths=jsonl_paths,
        bucket_seconds=effective_bucket_seconds,
        cache_entries=cached_jsonl_entries,
        loader=_collect_series_jsonl_path,
    )
    csv_series, csv_latest, csv_meta, rebuilt_csv_entries = _collect_series_cached(
        paths=csv_paths,
        bucket_seconds=effective_bucket_seconds,
        cache_entries=cached_csv_entries,
        loader=_collect_series_csv_path,
    )
    series = _merge_series_maps(jsonl_series, csv_series)
    latest = _merge_latest_maps(jsonl_latest, csv_latest)
    scan_meta = {
        "rows_scanned": int(jsonl_meta.get("rows_scanned", 0) or 0) + int(csv_meta.get("rows_scanned", 0) or 0),
        "symbols_seen": sorted(set(jsonl_meta.get("symbols_seen", [])) | set(csv_meta.get("symbols_seen", []))),
        "cache_hits": int(jsonl_meta.get("cache_hits", 0) or 0) + int(csv_meta.get("cache_hits", 0) or 0),
        "cache_misses": int(jsonl_meta.get("cache_misses", 0) or 0) + int(csv_meta.get("cache_misses", 0) or 0),
    }

    observed_symbols = set(series)
    stock_symbols = sorted(symbol for symbol in observed_symbols if not _is_crypto_symbol(symbol))
    crypto_symbols = sorted(symbol for symbol in observed_symbols if _is_crypto_symbol(symbol))

    stock_risk_symbols = [symbol for symbol in _STOCK_PROXY_SYMBOLS if symbol in observed_symbols]
    risk_off_symbols = [symbol for symbol in _RISK_OFF_PROXY_SYMBOLS if symbol in observed_symbols]
    core_crypto_symbols = [symbol for symbol in _CRYPTO_SYMBOLS if symbol in observed_symbols]

    stock_risk_series = _aggregate_series(series, stock_risk_symbols, "pct_from_close")
    crypto_risk_series = _aggregate_series(series, core_crypto_symbols, "pct_from_close")
    crypto_btc_series = series.get("BTC-USD", {}).get("pct_from_close", {})
    tlt_series = series.get("TLT", {}).get("pct_from_close", {})
    uup_series = series.get("UUP", {}).get("pct_from_close", {})
    gld_series = series.get("GLD", {}).get("pct_from_close", {})

    risk_corr, risk_points = _series_correlation(stock_risk_series, crypto_risk_series, min_points=min_points)
    risk_mode = "exact"
    if risk_points < min_points or risk_corr is None:
        risk_corr, risk_points = _series_correlation_asof(
            stock_risk_series,
            crypto_risk_series,
            min_points=min_points,
            max_lag_seconds=asof_max_lag_seconds,
        )
        if risk_points > 0:
            risk_mode = "asof"
    spy_row = _best_pair_correlation(series, "SPY", "BTC-USD", min_points=min_points, max_lag_seconds=asof_max_lag_seconds)
    qqq_row = _best_pair_correlation(series, "QQQ", "BTC-USD", min_points=min_points, max_lag_seconds=asof_max_lag_seconds)
    tlt_corr, tlt_points = _series_correlation(tlt_series, crypto_btc_series, min_points=min_points)
    tlt_mode = "exact"
    if tlt_points < min_points or tlt_corr is None:
        tlt_corr, tlt_points = _series_correlation_asof(
            tlt_series,
            crypto_btc_series,
            min_points=min_points,
            max_lag_seconds=asof_max_lag_seconds,
        )
        if tlt_points > 0:
            tlt_mode = "asof"
    uup_corr, uup_points = _series_correlation(uup_series, crypto_btc_series, min_points=min_points)
    uup_mode = "exact"
    if uup_points < min_points or uup_corr is None:
        uup_corr, uup_points = _series_correlation_asof(
            uup_series,
            crypto_btc_series,
            min_points=min_points,
            max_lag_seconds=asof_max_lag_seconds,
        )
        if uup_points > 0:
            uup_mode = "asof"
    gld_corr, gld_points = _series_correlation(gld_series, crypto_btc_series, min_points=min_points)
    gld_mode = "exact"
    if gld_points < min_points or gld_corr is None:
        gld_corr, gld_points = _series_correlation_asof(
            gld_series,
            crypto_btc_series,
            min_points=min_points,
            max_lag_seconds=asof_max_lag_seconds,
        )
        if gld_points > 0:
            gld_mode = "asof"

    latest_stock = _pick_latest(latest, stock_risk_symbols or risk_off_symbols)
    latest_crypto = _pick_latest(latest, core_crypto_symbols)
    current_alignment = _current_alignment_norm(latest_stock, latest_crypto)
    risk_corr_norm = _corr_to_norm(risk_corr)

    exact_pair_rows = [
        {
            "left": "stock_risk_basket",
            "right": "crypto_basket",
            "corr": risk_corr,
            "points": risk_points,
            "metric": "pct_from_close",
            "mode": risk_mode,
            "confidence_norm": _pair_confidence({"points": risk_points, "mode": risk_mode}),
        },
        {
            "left": "SPY",
            "right": "BTC-USD",
            "corr": spy_row.get("corr"),
            "points": spy_row.get("points"),
            "metric": spy_row.get("metric"),
            "mode": spy_row.get("mode", "exact"),
            "confidence_norm": _pair_confidence(spy_row),
        },
        {
            "left": "QQQ",
            "right": "BTC-USD",
            "corr": qqq_row.get("corr"),
            "points": qqq_row.get("points"),
            "metric": qqq_row.get("metric"),
            "mode": qqq_row.get("mode", "exact"),
            "confidence_norm": _pair_confidence(qqq_row),
        },
        {
            "left": "TLT",
            "right": "BTC-USD",
            "corr": tlt_corr,
            "points": tlt_points,
            "metric": "pct_from_close",
            "mode": tlt_mode,
            "confidence_norm": _pair_confidence({"points": tlt_points, "mode": tlt_mode}),
        },
        {
            "left": "UUP",
            "right": "BTC-USD",
            "corr": uup_corr,
            "points": uup_points,
            "metric": "pct_from_close",
            "mode": uup_mode,
            "confidence_norm": _pair_confidence({"points": uup_points, "mode": uup_mode}),
        },
        {
            "left": "GLD",
            "right": "BTC-USD",
            "corr": gld_corr,
            "points": gld_points,
            "metric": "pct_from_close",
            "mode": gld_mode,
            "confidence_norm": _pair_confidence({"points": gld_points, "mode": gld_mode}),
        },
    ]
    pair_rows = list(exact_pair_rows)
    global_features = {
        "market_crypto_risk_corr_norm": risk_corr_norm,
        "market_crypto_spy_corr_norm": _corr_to_norm(spy_row.get("corr")),
        "market_crypto_qqq_corr_norm": _corr_to_norm(qqq_row.get("corr")),
        "market_crypto_tlt_corr_norm": _corr_to_norm(tlt_corr),
        "market_crypto_uup_inverse_corr_norm": _inverse_corr_to_norm(uup_corr),
        "market_crypto_gold_corr_norm": _corr_to_norm(gld_corr),
        "market_crypto_current_alignment_norm": current_alignment,
        "market_crypto_divergence_norm": _clamp01(abs(current_alignment - risk_corr_norm) * 2.0),
        "market_crypto_corr_confidence_norm": _clamp01(
            statistics.mean(
                [
                    _corr_confidence_norm(risk_points),
                    _corr_confidence_norm(int(spy_row.get("points", 0) or 0)),
                    _corr_confidence_norm(int(qqq_row.get("points", 0) or 0)),
                    _corr_confidence_norm(tlt_points),
                    _corr_confidence_norm(uup_points),
                ]
            )
        ),
    }

    symbol_features: dict[str, dict[str, float]] = {}
    for symbol in core_crypto_symbols:
        spy_pair = _best_pair_correlation(series, "SPY", symbol, min_points=min_points, max_lag_seconds=asof_max_lag_seconds)
        qqq_pair = _best_pair_correlation(series, "QQQ", symbol, min_points=min_points, max_lag_seconds=asof_max_lag_seconds)
        tlt_pair = _best_pair_correlation(series, "TLT", symbol, min_points=min_points, max_lag_seconds=asof_max_lag_seconds)
        uup_pair = _best_pair_correlation(series, "UUP", symbol, min_points=min_points, max_lag_seconds=asof_max_lag_seconds)
        latest_symbol = latest.get(symbol, {})
        latest_proxy = _pick_latest(latest, stock_risk_symbols or risk_off_symbols)
        symbol_alignment = _current_alignment_norm(latest_proxy, latest_symbol)
        symbol_features[symbol] = {
            "market_crypto_risk_corr_norm": risk_corr_norm,
            "market_crypto_spy_corr_norm": _corr_to_norm(spy_pair.get("corr")),
            "market_crypto_qqq_corr_norm": _corr_to_norm(qqq_pair.get("corr")),
            "market_crypto_tlt_corr_norm": _corr_to_norm(tlt_pair.get("corr")),
            "market_crypto_uup_inverse_corr_norm": _inverse_corr_to_norm(uup_pair.get("corr")),
            "market_crypto_gold_corr_norm": _corr_to_norm(gld_corr),
            "market_crypto_current_alignment_norm": symbol_alignment,
            "market_crypto_divergence_norm": _clamp01(abs(symbol_alignment - risk_corr_norm) * 2.0),
            "market_crypto_corr_confidence_norm": _clamp01(
                statistics.mean(
                    [
                        _pair_confidence(spy_pair),
                        _pair_confidence(qqq_pair),
                        _pair_confidence(tlt_pair),
                        _pair_confidence(uup_pair),
                    ]
                )
            ),
        }

    for symbol in ("SPY", "QQQ", "TLT", "UUP", "GLD"):
        if symbol not in observed_symbols:
            continue
        btc_pair = _best_pair_correlation(series, symbol, "BTC-USD", min_points=min_points, max_lag_seconds=asof_max_lag_seconds)
        symbol_features[symbol] = {
            "market_crypto_risk_corr_norm": risk_corr_norm,
            "market_crypto_spy_corr_norm": _corr_to_norm(spy_row.get("corr")),
            "market_crypto_qqq_corr_norm": _corr_to_norm(qqq_row.get("corr")),
            "market_crypto_tlt_corr_norm": _corr_to_norm(tlt_corr),
            "market_crypto_uup_inverse_corr_norm": _inverse_corr_to_norm(uup_corr),
            "market_crypto_gold_corr_norm": _corr_to_norm(gld_corr),
            "market_crypto_current_alignment_norm": _current_alignment_norm(latest.get(symbol, {}), latest_crypto),
            "market_crypto_divergence_norm": _clamp01(abs(_corr_to_norm(btc_pair.get("corr")) - risk_corr_norm) * 2.0),
            "market_crypto_corr_confidence_norm": _pair_confidence(btc_pair),
        }

    exact_aligned_pairs = sum(
        1
        for row in exact_pair_rows
        if int(row.get("points", 0) or 0) >= min_points
        and row.get("corr") is not None
        and str(row.get("mode") or "exact") == "exact"
    )
    mode = "exact"
    approximate_aligned_pairs = sum(
        1
        for row in exact_pair_rows
        if int(row.get("points", 0) or 0) >= min_points
        and row.get("corr") is not None
        and str(row.get("mode") or "exact") == "asof"
    )
    if exact_aligned_pairs <= 0 and approximate_aligned_pairs > 0:
        mode = "approximate_overlap"
    if exact_aligned_pairs <= 0 and approximate_aligned_pairs <= 0 and last_usable:
        cached_global = last_usable.get("global_features") if isinstance(last_usable.get("global_features"), Mapping) else {}
        cached_symbol_features = last_usable.get("symbol_features") if isinstance(last_usable.get("symbol_features"), Mapping) else {}
        cached_pair_rows = last_usable.get("pair_metrics") if isinstance(last_usable.get("pair_metrics"), list) else []
        carry_scale = _carry_forward_confidence_scale(last_usable.get("timestamp_utc"), now)
        carry_alignment = current_alignment
        carry_risk_corr_norm = _safe_float(cached_global.get("market_crypto_risk_corr_norm"), risk_corr_norm)
        if cached_pair_rows and cached_global:
            mode = "carry_forward_last_usable"
            pair_rows = []
            for row in cached_pair_rows:
                if not isinstance(row, Mapping):
                    continue
                pair_rows.append(
                    {
                        **dict(row),
                        "mode": mode,
                        "historical_timestamp_utc": last_usable.get("timestamp_utc"),
                        "confidence_norm": min(_safe_float(row.get("confidence_norm"), 0.0), carry_scale),
                    }
                )
            global_features = {
                key: _safe_float(value, 0.0)
                for key, value in cached_global.items()
            }
            global_features["market_crypto_current_alignment_norm"] = carry_alignment
            global_features["market_crypto_divergence_norm"] = _clamp01(abs(carry_alignment - carry_risk_corr_norm) * 2.0)
            global_features["market_crypto_corr_confidence_norm"] = min(
                _safe_float(cached_global.get("market_crypto_corr_confidence_norm"), 0.0),
                carry_scale,
            )
            merged_symbol_features: dict[str, dict[str, float]] = {}
            for symbol, row in cached_symbol_features.items():
                if not isinstance(row, Mapping):
                    continue
                updated = {key: _safe_float(value, 0.0) for key, value in row.items()}
                if _norm_symbol(symbol) in latest:
                    updated_alignment = _current_alignment_norm(latest_stock, latest.get(_norm_symbol(symbol), {}))
                    updated["market_crypto_current_alignment_norm"] = updated_alignment
                    updated["market_crypto_divergence_norm"] = _clamp01(abs(updated_alignment - carry_risk_corr_norm) * 2.0)
                updated["market_crypto_corr_confidence_norm"] = min(
                    _safe_float(updated.get("market_crypto_corr_confidence_norm"), 0.0),
                    carry_scale,
                )
                merged_symbol_features[str(symbol)] = updated
            symbol_features = merged_symbol_features

    warnings: list[str] = []
    if not stock_risk_symbols:
        warnings.append("no_stock_risk_proxy_symbols_observed")
    if not core_crypto_symbols:
        warnings.append("no_core_crypto_symbols_observed")
    if risk_points < min_points:
        warnings.append(f"sparse_stock_crypto_overlap:{risk_points}<{min_points}")
    if approximate_aligned_pairs > 0:
        warnings.append(f"approximate_overlap_pairs:{approximate_aligned_pairs}")
    if market_hours_bias and exact_aligned_pairs <= 0 and approximate_aligned_pairs > 0:
        warnings.append("market_hours_exact_overlap_missing")
    if mode == "carry_forward_last_usable":
        warnings.append(f"{mode}:using_cached_overlap")

    aligned_pairs = sum(1 for row in pair_rows if int(row.get("points", 0) or 0) >= min_points and row.get("corr") is not None)
    status = {
        "timestamp_utc": now.isoformat(),
        "ok": bool(core_crypto_symbols) and (bool(stock_risk_symbols or risk_off_symbols) or mode != "exact"),
        "files_scanned": len(jsonl_paths) + len(csv_paths),
        "rows_scanned": int(scan_meta.get("rows_scanned", 0) or 0),
        "stock_symbols_observed": len(stock_symbols),
        "crypto_symbols_observed": len(crypto_symbols),
        "aligned_pairs": aligned_pairs,
        "exact_aligned_pairs": exact_aligned_pairs,
        "mode": mode,
        "market_hours_bias": market_hours_bias,
        "asof_max_lag_seconds": int(asof_max_lag_seconds),
        "cache_hits": int(scan_meta.get("cache_hits", 0) or 0),
        "cache_misses": int(scan_meta.get("cache_misses", 0) or 0),
        "cache_result_reused": False,
        "warning_count": len(warnings),
        "warnings": warnings,
        "roots": [str(path) for path in _iter_repo_roots(project_root, extra_roots)],
    }
    payload = {
        "timestamp_utc": now.isoformat(),
        "provider": "market_crypto_correlation",
        "status": status,
        "sources": {
            "master_control_jsonl_paths": [str(path) for path in jsonl_paths],
            "master_control_csv_paths": [str(path) for path in csv_paths],
        },
        "derived": {
            "calendar_features": {},
            "news_features": {},
            "global_features": global_features,
            "symbol_features": symbol_features,
            "pair_metrics": pair_rows,
            "latest_snapshots": latest,
        },
    }
    last_usable_payload = dict(last_usable) if last_usable else {}
    if exact_aligned_pairs > 0 or approximate_aligned_pairs > 0:
        last_usable_payload = {
            "timestamp_utc": now.isoformat(),
            "aligned_pairs": aligned_pairs,
            "pair_metrics": exact_pair_rows,
            "global_features": global_features,
            "symbol_features": symbol_features,
        }
    _write_json(
        cache_path,
        {
            "timestamp_utc": now.isoformat(),
            "schema_version": 1,
            "jsonl_file_cache": rebuilt_jsonl_entries,
            "csv_file_cache": rebuilt_csv_entries,
            "last_usable": last_usable_payload,
            "last_result_state": result_cache_state,
            "last_result": {
                "status": status,
                "payload": payload,
            },
        },
    )
    return payload, status


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect stock/crypto cross-market correlation context from live master-control snapshots.")
    parser.add_argument("--lookback-days", type=int, default=5)
    parser.add_argument("--bucket-seconds", type=int, default=300)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = open(LOCK_PATH, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_fh.seek(0)
        owner = lock_fh.read().strip()
        msg = {
            "ok": True,
            "busy": True,
            "reason": "market_crypto_correlation_lock_busy",
            "lock_path": str(LOCK_PATH),
            "owner": owner,
        }
        print(json.dumps(msg, ensure_ascii=True) if args.json else f"market_crypto_correlation busy owner={owner or 'unknown'}")
        return 0

    lock_fh.seek(0)
    lock_fh.truncate(0)
    lock_fh.write(json.dumps({"pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat()}, ensure_ascii=True))
    lock_fh.flush()

    extra_roots: list[Path] = []
    configured_external_raw = str(os.environ.get("BOT_LOGS_EXTERNAL_ROOT", str(EXTERNAL_MIRROR_DEFAULT))).strip()
    if configured_external_raw:
        configured_external = Path(configured_external_raw)
        extra_roots.append(configured_external)

    payload, status = collect_market_crypto_correlation_context(
        project_root=PROJECT_ROOT,
        lookback_days=max(int(args.lookback_days), 1),
        bucket_seconds=max(int(args.bucket_seconds), 60),
        min_points=max(int(args.min_points), 3),
        extra_roots=extra_roots,
    )

    _write_json(PROJECT_ROOT / "exports" / "external_context" / "market_crypto_correlation_latest.json", payload)
    _write_json(PROJECT_ROOT / "governance" / "health" / "market_crypto_correlation_sync_latest.json", status)
    report = _render_markdown_report(
        status=status,
        pair_rows=list(payload.get("derived", {}).get("pair_metrics", [])),
        latest=payload.get("derived", {}).get("latest_snapshots", {}),
    )
    report_path = PROJECT_ROOT / "exports" / "reports" / "market_crypto_correlation_latest.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    if args.json:
        print(json.dumps(status, ensure_ascii=True))
    else:
        print(
            "market_crypto_correlation ok={ok} files={files} rows={rows} aligned_pairs={pairs}".format(
                ok=status["ok"],
                files=status["files_scanned"],
                rows=status["rows_scanned"],
                pairs=status["aligned_pairs"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
