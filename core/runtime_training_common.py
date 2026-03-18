from __future__ import annotations

import glob
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from market_context_features import (
    BOND_REFERENCE_FEATURE_KEYS,
    BREADTH_FEATURE_KEYS,
    CREDIT_CONTEXT_FEATURE_KEYS,
    NEWS_STRUCTURED_FEATURE_KEYS,
    load_latest_external_context,
    summarize_bond_reference_context,
    summarize_breadth_context,
    summarize_credit_context,
)

try:
    from derivatives_features import summarize_calendar_payload
except Exception:
    summarize_calendar_payload = None


RuntimeObservation = Dict[str, Any]
RuntimeSequenceMap = Dict[Tuple[str, str], List[RuntimeObservation]]
RuntimeFeatureBuilder = Callable[[Sequence[RuntimeObservation], int], np.ndarray]
RuntimeLabelBuilder = Callable[[Sequence[RuntimeObservation], int, int], Optional[float]]
RuntimeSampleFilter = Callable[[Sequence[RuntimeObservation], int, int], bool]
RuntimeConfidenceBuilder = Callable[[Sequence[RuntimeObservation], int, int], float]

_ROOT_STRATEGY_PRIORITY = {
    "grand_master_bot": 0,
    "grand_master_intent_bot": 1,
}

_RUNTIME_NEWS_EVENT_KEYS = {
    "news_available",
    "news_items_30m",
    "news_items_2h",
    "news_items_24h",
    "news_sentiment",
    "news_negative_share",
    "news_positive_share",
    "news_shock_rate",
    "news_recent_impact",
    "news_novelty_norm",
}

_RUNTIME_CALENDAR_EVENT_KEYS = {
    "calendar_feed_available",
    "calendar_event_proximity_norm",
    "calendar_next_event_norm",
    "calendar_events_24h_norm",
    "calendar_high_impact_24h_norm",
    "calendar_macro_event_norm",
    "calendar_macro_surprise_norm",
    "calendar_macro_abs_surprise_norm",
    "calendar_macro_revision_norm",
    "calendar_fomc_event_norm",
    "calendar_cpi_event_norm",
    "calendar_labor_event_norm",
    "calendar_treasury_auction_norm",
}

_RUNTIME_GAP_FILL_KEYS = set(BREADTH_FEATURE_KEYS) | set(BOND_REFERENCE_FEATURE_KEYS) | set(CREDIT_CONTEXT_FEATURE_KEYS) | set(NEWS_STRUCTURED_FEATURE_KEYS) | _RUNTIME_NEWS_EVENT_KEYS | _RUNTIME_CALENDAR_EVENT_KEYS


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


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


def _path_day_utc(path: Path) -> Optional[datetime]:
    parts = path.stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    stamp = parts[-1]
    if len(stamp) != 8 or (not stamp.isdigit()):
        return None
    try:
        return datetime.strptime(stamp, "%Y%m%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _is_missing_feature(features: Mapping[str, Any], name: str) -> bool:
    if name not in features:
        return True
    try:
        value = float(features.get(name))
    except Exception:
        return True
    return not math.isfinite(value)


def _set_missing_feature(features: Dict[str, Any], name: str, value: Any) -> None:
    try:
        coerced = float(value)
    except Exception:
        return
    if not math.isfinite(coerced):
        return
    if _is_missing_feature(features, name):
        features[name] = coerced


def _feature_subset(payload: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in keys:
        try:
            value = float(payload.get(key))  # type: ignore[arg-type]
        except Exception:
            continue
        if math.isfinite(value):
            out[str(key)] = value
    return out


def _live_macro_gap_fill_features(payload: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not isinstance(payload, Mapping):
        return {}, {}

    active = bool(payload.get("active"))
    shock_hint = max(0.0, min(_safe_float(payload.get("shock_hint"), 0.0), 1.0))
    sentiment_hint = max(-1.0, min(_safe_float(payload.get("sentiment_hint"), 0.0), 1.0))
    stance = str(payload.get("stance") or "").strip().lower()
    template = str(payload.get("template") or "").strip().lower()
    source = str(payload.get("source") or "").strip().lower()
    strength = max(shock_hint, 1.0 if active else 0.0)
    if strength <= 0.0:
        return {}, {}

    macro_event = 1.0 if (template in {"powell", "fed", "fomc"} or "federal reserve" in source or "powell" in source) else min(0.7, strength)
    calendar_features = {
        "calendar_feed_available": strength,
        "calendar_event_proximity_norm": strength,
        "calendar_next_event_norm": strength,
        "calendar_events_24h_norm": strength,
        "calendar_high_impact_24h_norm": strength,
        "calendar_macro_event_norm": macro_event,
        "calendar_macro_surprise_norm": max(0.0, min(1.0, 0.5 + (sentiment_hint * 0.5))),
        "calendar_macro_abs_surprise_norm": abs(sentiment_hint),
        "calendar_fomc_event_norm": macro_event,
    }
    if "auction" in stance or "auction" in template:
        calendar_features["calendar_treasury_auction_norm"] = strength

    news_features = {
        "news_available": max(0.35, strength),
        "news_items_30m": min(strength, 1.0) * 0.6,
        "news_items_2h": min(strength, 1.0) * 0.75,
        "news_items_24h": min(strength, 1.0),
        "news_sentiment": sentiment_hint,
        "news_negative_share": max(-sentiment_hint, 0.0),
        "news_positive_share": max(sentiment_hint, 0.0),
        "news_shock_rate": strength,
        "news_recent_impact": strength,
        "news_source_quality_norm": 0.9,
        "news_entity_relevance_norm": 0.9 if bool(payload.get("broad_market")) else 0.65,
        "news_novelty_norm": min(0.45 + (strength * 0.5), 1.0),
    }
    return calendar_features, news_features


def _load_runtime_gap_fill_context(project_root: Path) -> Dict[str, Any]:
    tradingeconomics = load_latest_external_context(project_root, "tradingeconomics")
    market_breadth = load_latest_external_context(project_root, "market_breadth")
    bond_reference = load_latest_external_context(project_root, "bond_reference")
    live_macro = load_latest_external_context(project_root, "live_macro")

    te_derived = tradingeconomics.get("derived") if isinstance(tradingeconomics.get("derived"), Mapping) else {}
    te_calendar = te_derived.get("calendar_features") if isinstance(te_derived.get("calendar_features"), Mapping) else {}
    te_news = te_derived.get("news_features") if isinstance(te_derived.get("news_features"), Mapping) else {}
    te_calendar_rows = te_derived.get("calendar_rows") if isinstance(te_derived.get("calendar_rows"), list) else []

    calendar_features = _feature_subset(te_calendar, _RUNTIME_CALENDAR_EVENT_KEYS)
    if te_calendar_rows and callable(summarize_calendar_payload):
        try:
            summarized = summarize_calendar_payload(te_calendar_rows, now_ts=datetime.now(timezone.utc).timestamp(), max_items=600)
        except Exception:
            summarized = {}
        for key, value in _feature_subset(summarized, _RUNTIME_CALENDAR_EVENT_KEYS).items():
            if key not in calendar_features:
                calendar_features[key] = value

    news_features = _feature_subset(te_news, set(NEWS_STRUCTURED_FEATURE_KEYS) | _RUNTIME_NEWS_EVENT_KEYS)
    if te_news:
        news_features.setdefault("news_available", 0.35)
        news_features.setdefault("news_items_24h", 0.4)

    live_macro_calendar, live_macro_news = _live_macro_gap_fill_features(live_macro if isinstance(live_macro, Mapping) else {})
    breadth_features = summarize_breadth_context(
        symbol="SPY",
        market_snapshot={},
        context_market={},
        external_snapshot=market_breadth if isinstance(market_breadth, Mapping) else {},
    )

    return {
        "calendar_features": calendar_features,
        "news_features": news_features,
        "live_macro_calendar": live_macro_calendar,
        "live_macro_news": live_macro_news,
        "breadth_features": breadth_features,
        "bond_reference": bond_reference if isinstance(bond_reference, Mapping) else {},
    }


def _enrich_runtime_observation(
    obs: RuntimeObservation,
    *,
    carry_forward_features: Mapping[str, float],
    gap_fill_context: Mapping[str, Any],
) -> RuntimeObservation:
    enriched = dict(obs)
    features = dict(obs.get("features") if isinstance(obs.get("features"), Mapping) else {})

    for key, value in carry_forward_features.items():
        _set_missing_feature(features, key, value)

    calendar_features = gap_fill_context.get("calendar_features") if isinstance(gap_fill_context.get("calendar_features"), Mapping) else {}
    news_features = gap_fill_context.get("news_features") if isinstance(gap_fill_context.get("news_features"), Mapping) else {}
    live_macro_calendar = gap_fill_context.get("live_macro_calendar") if isinstance(gap_fill_context.get("live_macro_calendar"), Mapping) else {}
    live_macro_news = gap_fill_context.get("live_macro_news") if isinstance(gap_fill_context.get("live_macro_news"), Mapping) else {}
    breadth_features = gap_fill_context.get("breadth_features") if isinstance(gap_fill_context.get("breadth_features"), Mapping) else {}
    bond_reference = gap_fill_context.get("bond_reference") if isinstance(gap_fill_context.get("bond_reference"), Mapping) else {}

    for key, value in calendar_features.items():
        _set_missing_feature(features, str(key), value)
    for key, value in news_features.items():
        _set_missing_feature(features, str(key), value)
    for key, value in live_macro_calendar.items():
        _set_missing_feature(features, str(key), value)
    for key, value in live_macro_news.items():
        _set_missing_feature(features, str(key), value)
    for key, value in breadth_features.items():
        _set_missing_feature(features, str(key), value)

    symbol = str(obs.get("symbol") or "").strip().upper()
    bond_features = summarize_bond_reference_context(
        symbol=symbol,
        market_snapshot=features,
        context_market={},
        calendar_features=features,
        external_snapshot=bond_reference,
    )
    credit_features = summarize_credit_context(
        symbol=symbol,
        market_snapshot=features,
        context_market={},
        external_snapshot=bond_reference,
    )
    for key, value in bond_features.items():
        _set_missing_feature(features, str(key), value)
    for key, value in credit_features.items():
        _set_missing_feature(features, str(key), value)

    enriched["features"] = features
    return enriched


def _recent_decision_paths(project_root: Path, *, lookback_days: int) -> List[Path]:
    root = Path(project_root).expanduser().resolve()
    since_utc = datetime.now(timezone.utc) - timedelta(days=max(int(lookback_days), 1))
    cutoff_day = (since_utc - timedelta(days=1)).date()
    out: List[Path] = []
    for raw in glob.glob(str(root / "decision_explanations" / "shadow*" / "decision_explanations_*.jsonl")):
        path = Path(raw)
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


def observation_feature(obs: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    token = str(name or "").strip()
    if not token:
        return default

    if token == "last_price":
        return _safe_float(obs.get("price"), default)
    if token == "symbol_hash":
        return _stable_hash01(str(obs.get("symbol") or ""))
    if token == "mode_hash":
        return _stable_hash01(str(obs.get("mode") or ""))

    features = obs.get("features") if isinstance(obs.get("features"), dict) else {}
    return _safe_float(features.get(token), default)


def _stable_hash01(text: str) -> float:
    if not text:
        return 0.0
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h / 0xFFFFFFFF


def price_change(sequence: Sequence[RuntimeObservation], idx: int, lookback: int = 1) -> float:
    back = max(int(lookback), 1)
    if idx <= 0:
        return 0.0
    j = max(0, idx - back)
    prev_price = observation_feature(sequence[j], "last_price", 0.0)
    curr_price = observation_feature(sequence[idx], "last_price", 0.0)
    if prev_price <= 0.0 or curr_price <= 0.0:
        return 0.0
    return (curr_price / max(prev_price, 1e-8)) - 1.0


def feature_std(sequence: Sequence[RuntimeObservation], idx: int, name: str, window: int = 6) -> float:
    w = max(int(window), 1)
    start = max(0, idx - w + 1)
    vals = [observation_feature(sequence[j], name, 0.0) for j in range(start, idx + 1)]
    if not vals:
        return 0.0
    return float(np.std(np.asarray(vals, dtype=np.float64)))


def feature_mean(sequence: Sequence[RuntimeObservation], idx: int, name: str, window: int = 6) -> float:
    w = max(int(window), 1)
    start = max(0, idx - w + 1)
    vals = [observation_feature(sequence[j], name, 0.0) for j in range(start, idx + 1)]
    if not vals:
        return 0.0
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def feature_ema(sequence: Sequence[RuntimeObservation], idx: int, name: str, span: int = 6) -> float:
    use_span = max(int(span), 1)
    start = max(0, idx - (use_span * 6))
    alpha = 2.0 / (use_span + 1.0)
    out = 0.0
    initialized = False
    for j in range(start, idx + 1):
        val = observation_feature(sequence[j], name, 0.0)
        if not initialized:
            out = val
            initialized = True
        else:
            out = alpha * val + (1.0 - alpha) * out
    return float(out)


def rolling_drawdown(sequence: Sequence[RuntimeObservation], idx: int, window: int = 20) -> float:
    w = max(int(window), 1)
    start = max(0, idx - w + 1)
    prices = [observation_feature(sequence[j], "last_price", 0.0) for j in range(start, idx + 1)]
    prices = [p for p in prices if p > 0.0]
    if not prices:
        return 0.0
    peak = max(prices)
    curr = prices[-1]
    return (curr / max(peak, 1e-8)) - 1.0


def future_return(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> float:
    h = max(int(horizon), 1)
    curr_price = observation_feature(sequence[idx], "last_price", 0.0)
    fut_price = observation_feature(sequence[idx + h], "last_price", 0.0)
    if curr_price <= 0.0 or fut_price <= 0.0:
        return 0.0
    return (fut_price / max(curr_price, 1e-8)) - 1.0


def future_max_drawdown(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> float:
    h = max(int(horizon), 1)
    curr_price = observation_feature(sequence[idx], "last_price", 0.0)
    if curr_price <= 0.0:
        return 0.0
    worst = 0.0
    for j in range(idx + 1, idx + h + 1):
        price = observation_feature(sequence[j], "last_price", 0.0)
        if price <= 0.0:
            continue
        ret = (price / max(curr_price, 1e-8)) - 1.0
        worst = min(worst, ret)
    return worst


def future_realized_vol(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> float:
    h = max(int(horizon), 1)
    prices = [observation_feature(sequence[j], "last_price", 0.0) for j in range(idx, idx + h + 1)]
    prices = [p for p in prices if p > 0.0]
    if len(prices) < 3:
        return 0.0
    arr = np.asarray(prices, dtype=np.float64)
    rets = np.diff(np.log(np.maximum(arr, 1e-8)))
    if rets.size == 0:
        return 0.0
    return float(np.std(rets))


def symbol_role_features(symbol: str, role_map: Mapping[str, Sequence[str]]) -> Dict[str, float]:
    sym = str(symbol or "").strip().upper()
    out: Dict[str, float] = {}
    for role, symbols in role_map.items():
        token = str(role or "").strip().lower()
        key = f"role_{token}"
        out[key] = 1.0 if sym in {str(s).strip().upper() for s in symbols} else 0.0
    return out


def direction_label_builder(*, min_return: float = 0.0) -> RuntimeLabelBuilder:
    threshold = float(min_return)

    def _label(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> Optional[float]:
        ret = future_return(sequence, idx, horizon)
        return 1.0 if ret > threshold else 0.0

    return _label


def selective_direction_label_builder(*, min_abs_return: float = 0.0) -> RuntimeLabelBuilder:
    threshold = abs(float(min_abs_return))

    def _label(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> Optional[float]:
        ret = future_return(sequence, idx, horizon)
        if abs(ret) <= threshold:
            return None
        return 1.0 if ret > 0.0 else 0.0

    return _label


def multi_horizon_direction_label_builder(
    *,
    horizons: Sequence[int],
    min_return: float = 0.0,
) -> RuntimeLabelBuilder:
    threshold = abs(float(min_return))
    eval_horizons = sorted({max(int(h), 1) for h in horizons if int(h) > 0})

    def _label(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> Optional[float]:
        used_horizons = eval_horizons or [max(int(horizon), 1)]
        votes: List[float] = []
        for step_h in used_horizons:
            if (idx + step_h) >= len(sequence):
                return None
            ret = future_return(sequence, idx, step_h)
            if abs(ret) <= threshold:
                return None
            votes.append(1.0 if ret > 0.0 else 0.0)
        if not votes or len(set(votes)) != 1:
            return None
        return votes[0]

    return _label


def risk_support_label_builder(
    *,
    min_return: float = -0.001,
    max_drawdown: float = 0.015,
    max_realized_vol: float = 0.02,
    vol_multiplier: float = 3.0,
) -> RuntimeLabelBuilder:
    min_ret = float(min_return)
    max_dd = abs(float(max_drawdown))
    max_vol = abs(float(max_realized_vol))
    mult = max(float(vol_multiplier), 1.0)

    def _label(sequence: Sequence[RuntimeObservation], idx: int, horizon: int) -> Optional[float]:
        fwd_ret = future_return(sequence, idx, horizon)
        dd = abs(future_max_drawdown(sequence, idx, horizon))
        realized = future_realized_vol(sequence, idx, horizon)
        curr_vol = abs(observation_feature(sequence[idx], "vol_30m", 0.0))
        allowed_vol = max(max_vol, curr_vol * mult)
        return 1.0 if (fwd_ret >= min_ret and dd <= max_dd and realized <= allowed_vol) else 0.0

    return _label


def load_runtime_observation_sequences(
    project_root: Path,
    *,
    lookback_days: int = 14,
    mode_allowlist: Optional[Sequence[str]] = None,
    symbol_allowlist: Optional[Sequence[str]] = None,
) -> RuntimeSequenceMap:
    root = Path(project_root).expanduser().resolve()
    since_utc = datetime.now(timezone.utc) - timedelta(days=max(int(lookback_days), 1))
    mode_allow = {str(x).strip().lower() for x in (mode_allowlist or []) if str(x).strip()}
    symbol_allow = {str(x).strip().upper() for x in (symbol_allowlist or []) if str(x).strip()}
    gap_fill_context = _load_runtime_gap_fill_context(root)

    best_by_snapshot: Dict[Tuple[str, str, str], RuntimeObservation] = {}
    for path in _recent_decision_paths(root, lookback_days=max(int(lookback_days), 1)):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                    if str(metadata.get("layer") or "").strip().lower() != "grand_master":
                        continue
                    strategy = str(row.get("strategy") or "").strip().lower()
                    if strategy not in _ROOT_STRATEGY_PRIORITY:
                        continue

                    ts = _parse_ts(row.get("timestamp_utc"))
                    if ts is None or ts < since_utc:
                        continue
                    mode = str(row.get("mode") or "").strip().lower()
                    symbol = str(row.get("symbol") or "").strip().upper()
                    if not mode or not symbol:
                        continue
                    if mode_allow and mode not in mode_allow:
                        continue
                    if symbol_allow and symbol not in symbol_allow:
                        continue

                    gates = row.get("gates") if isinstance(row.get("gates"), dict) else {}
                    if ("market_data_ok" in gates) and (not bool(gates.get("market_data_ok"))):
                        continue

                    features = row.get("features") if isinstance(row.get("features"), dict) else {}
                    price = _safe_float(features.get("last_price"), 0.0)
                    if price <= 0.0:
                        continue

                    snapshot_id = str(metadata.get("snapshot_id") or row.get("snapshot_id") or row.get("parent_decision_id") or "").strip()
                    if not snapshot_id:
                        snapshot_id = f"{symbol}:{ts.isoformat()}"

                    obs = {
                        "mode": mode,
                        "symbol": symbol,
                        "strategy": strategy,
                        "strategy_priority": _ROOT_STRATEGY_PRIORITY[strategy],
                        "snapshot_id": snapshot_id,
                        "ts_epoch": float(ts.timestamp()),
                        "timestamp_utc": ts.isoformat(),
                        "price": price,
                        "features": dict(features),
                    }
                    key = (mode, symbol, snapshot_id)
                    prev = best_by_snapshot.get(key)
                    if prev is None or int(obs["strategy_priority"]) < int(prev.get("strategy_priority", 99)):
                        best_by_snapshot[key] = obs
        except Exception:
            continue

    grouped: RuntimeSequenceMap = defaultdict(list)
    for obs in best_by_snapshot.values():
        grouped[(str(obs["mode"]), str(obs["symbol"]))].append(obs)

    out: RuntimeSequenceMap = {}
    for key, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda x: (float(x.get("ts_epoch", 0.0)), int(x.get("strategy_priority", 99)), str(x.get("snapshot_id") or "")))
        deduped: List[RuntimeObservation] = []
        seen_snapshot_ids: set[str] = set()
        carry_forward_features: Dict[str, float] = {}
        for row in rows_sorted:
            sid = str(row.get("snapshot_id") or "")
            if sid in seen_snapshot_ids:
                continue
            seen_snapshot_ids.add(sid)
            enriched = _enrich_runtime_observation(
                row,
                carry_forward_features=carry_forward_features,
                gap_fill_context=gap_fill_context,
            )
            deduped.append(enriched)
            next_carry: Dict[str, float] = {}
            feature_map = enriched.get("features") if isinstance(enriched.get("features"), Mapping) else {}
            for name in _RUNTIME_GAP_FILL_KEYS:
                if name not in feature_map:
                    continue
                try:
                    value = float(feature_map.get(name))
                except Exception:
                    continue
                if math.isfinite(value):
                    next_carry[name] = value
            carry_forward_features.update(next_carry)
        if deduped:
            out[key] = deduped
    return out


def make_runtime_windowed_dataset(
    *,
    sequences: RuntimeSequenceMap,
    feature_builder: RuntimeFeatureBuilder,
    label_builder: RuntimeLabelBuilder,
    sample_filter: Optional[RuntimeSampleFilter] = None,
    confidence_builder: Optional[RuntimeConfidenceBuilder] = None,
    min_confidence: float = 0.0,
    window: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    w = max(int(window), 1)
    h = max(int(horizon), 1)
    min_conf = max(0.0, min(float(min_confidence), 1.0))

    samples: List[np.ndarray] = []
    labels: List[float] = []
    anchor_ts: List[float] = []
    sample_confidence: List[float] = []
    eligible_sequences = 0
    skipped_labels = 0
    skipped_filtered = 0
    skipped_low_confidence = 0
    feature_dim = 0

    for rows in sequences.values():
        if len(rows) < (w + h):
            continue
        eligible_sequences += 1
        for idx in range(w - 1, len(rows) - h):
            if sample_filter is not None:
                try:
                    include_sample = bool(sample_filter(rows, idx, h))
                except Exception:
                    include_sample = False
                if not include_sample:
                    skipped_filtered += 1
                    continue

            confidence = 1.0
            if confidence_builder is not None:
                try:
                    confidence = _safe_float(confidence_builder(rows, idx, h), 0.0)
                except Exception:
                    confidence = 0.0
                confidence = min(max(confidence, 0.0), 1.0)
                if confidence < min_conf:
                    skipped_low_confidence += 1
                    continue

            per_step: List[np.ndarray] = []
            for step_idx in range(idx - w + 1, idx + 1):
                vec = np.asarray(feature_builder(rows, step_idx), dtype=np.float32).reshape(-1)
                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                if vec.size == 0:
                    per_step = []
                    break
                if feature_dim == 0:
                    feature_dim = int(vec.size)
                per_step.append(vec)
            if not per_step:
                continue

            label = label_builder(rows, idx, h)
            if label is None or (not math.isfinite(float(label))):
                skipped_labels += 1
                continue

            sample = np.concatenate(per_step, axis=0)
            samples.append(sample.astype(np.float32))
            labels.append(float(label))
            anchor_ts.append(float(rows[idx].get("ts_epoch", 0.0)))
            sample_confidence.append(float(confidence))

    if not samples:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 1), dtype=np.float32), {
            "sequence_count": len(sequences),
            "eligible_sequences": eligible_sequences,
            "sample_count": 0,
            "feature_dim": 0,
            "window": w,
            "horizon": h,
            "positive_rate": 0.0,
            "skipped_labels": skipped_labels,
            "skipped_filtered": skipped_filtered,
            "skipped_low_confidence": skipped_low_confidence,
            "confidence_mean": 0.0,
            "confidence_min": 0.0,
            "confidence_max": 0.0,
            "min_confidence": float(min_conf),
            "_sample_confidence": np.zeros((0,), dtype=np.float32),
        }

    order = np.argsort(np.asarray(anchor_ts, dtype=np.float64))
    X = np.asarray([samples[i] for i in order], dtype=np.float32)
    y = np.asarray([[labels[i]] for i in order], dtype=np.float32)
    conf = np.asarray([sample_confidence[i] for i in order], dtype=np.float32)
    positive_rate = float(np.mean(y[:, 0])) if y.size else 0.0
    return X, y, {
        "sequence_count": len(sequences),
        "eligible_sequences": eligible_sequences,
        "sample_count": int(X.shape[0]),
        "feature_dim": int(feature_dim),
        "window": w,
        "horizon": h,
        "positive_rate": positive_rate,
        "skipped_labels": skipped_labels,
        "skipped_filtered": skipped_filtered,
        "skipped_low_confidence": skipped_low_confidence,
        "confidence_mean": float(np.mean(conf)) if conf.size else 0.0,
        "confidence_min": float(np.min(conf)) if conf.size else 0.0,
        "confidence_max": float(np.max(conf)) if conf.size else 0.0,
        "min_confidence": float(min_conf),
        "_sample_confidence": conf,
    }
