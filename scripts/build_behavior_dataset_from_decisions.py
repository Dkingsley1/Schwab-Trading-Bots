import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHOCK_SYMBOLS = {"UVXY", "VIXY", "SOXL", "SOXS", "MSTR", "SMCI", "COIN", "TSLA"}
MEAN_REVERT_SYMBOLS = {"TLT", "IEF", "SHY", "BND", "AGG", "GLD", "XLU", "XLP"}

FEATURE_NAMES = [
    "pnl_proxy",
    "qty_log",
    "role_idx",
    "symbol_hash",
    "action_hash",
    "dow",
    "hour",
    "regime_idx",
    "label_confidence_proxy",
    "pct_from_close_scaled",
    "mom_5m_scaled",
    "vol_30m_scaled",
    "range_pos",
    "spread_bps_norm",
    "ctx_vix_pct_scaled",
    "ctx_uup_pct_scaled",
    "lag_slippage_bps_norm",
    "lag_latency_ms_norm",
    "lag_impact_bps_norm",
    "active_sub_bots_norm",
    "queue_depth_norm",
    "dispatch_qty_norm",
    "session_bucket_norm",
    "mins_from_open_norm",
    "mins_to_close_norm",
    "event_window_proximity",
    "feature_freshness_ok",
    "feature_freshness_age_ratio",
    "master_latency_slo_ok",
    "master_latency_ratio",
    "risk_pause_active",
    "snapshot_cov_ok",
    "snapshot_cov_log_ratio",
    "snapshot_replay_stale_ratio",
    "snapshot_replay_drift_ratio",
    "snapshot_divergence_ratio",
    "snapshot_triprate_ratio",
    "snapshot_queue_pressure_ratio",
    "canary_weight_cap_norm",
]


def _safe_load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else default
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(float(value), hi))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _signed_scale(value: float, gain: float) -> float:
    return math.tanh(float(value) * float(gain))


def _hash01(text: str) -> float:
    if not text:
        return 0.0
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h / 0xFFFFFFFF


def _parse_ts(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    s = str(raw).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _iter_jsonl(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except Exception:
            continue



def _resolve_glob_paths(pattern: str, *, root: Path) -> List[Path]:
    pat = str(pattern or "").strip()
    if not pat:
        return []

    matches: List[Path] = []
    for raw in glob.glob(pat, recursive=True):
        p = Path(raw)
        if p.is_file():
            matches.append(p.resolve())

    if not os.path.isabs(pat):
        for raw in glob.glob(str(root / pat), recursive=True):
            p = Path(raw)
            if p.is_file():
                matches.append(p.resolve())

    uniq = {str(p): p for p in matches}
    return [uniq[k] for k in sorted(uniq.keys())]


def _role_index(mode_label: str) -> float:
    m = (mode_label or "").lower()
    if "swing" in m:
        return 2.0 / 3.0
    if "dividend" in m or "bond" in m:
        return 0.0
    return 1.0 / 3.0


def _regime_index(symbol: str, features: Dict[str, Any]) -> Tuple[float, str]:
    s = (symbol or "").upper()
    pct = abs(_to_float(features.get("pct_from_close"), 0.0))
    mom = abs(_to_float(features.get("mom_5m"), 0.0))
    vol = abs(_to_float(features.get("vol_30m"), 0.0))

    if s in SHOCK_SYMBOLS or vol >= 0.03 or pct >= 0.04:
        return 2.0 / 3.0, "shock"
    if s in MEAN_REVERT_SYMBOLS:
        return 1.0 / 3.0, "mean_revert"
    if mom >= 0.001 or pct >= 0.0015:
        return 0.0, "trend"
    return 1.0, "other"


def _event_windows_from_env() -> List[Tuple[int, int]]:
    raw = os.getenv("EVENT_LOCK_WINDOWS_ET", os.getenv("EVENT_BLACKOUT_WINDOWS_ET", "08:29-08:36,09:59-10:06,13:58-14:05")).strip()
    windows: List[Tuple[int, int]] = []
    if not raw:
        return windows
    for seg in raw.split(","):
        seg = seg.strip()
        if "-" not in seg:
            continue
        a, b = seg.split("-", 1)
        try:
            ah, am = [int(x) for x in a.split(":", 1)]
            bh, bm = [int(x) for x in b.split(":", 1)]
            windows.append((ah * 60 + am, bh * 60 + bm))
        except Exception:
            continue
    return windows


def _session_event_context(ts_utc: datetime, windows: List[Tuple[int, int]]) -> Dict[str, float]:
    if ZoneInfo is not None:
        ts_et = ts_utc.astimezone(ZoneInfo("America/New_York"))
    else:
        ts_et = ts_utc

    now_min = ts_et.hour * 60 + ts_et.minute
    open_min = 9 * 60 + 30
    close_min = 16 * 60

    if now_min < open_min:
        bucket = 0.0
    elif now_min <= close_min:
        bucket = 0.5
    else:
        bucket = 1.0

    mins_from_open = _clamp01((now_min - open_min) / 390.0)
    mins_to_close = _clamp01((close_min - now_min) / 390.0)

    proximity = 0.0
    for start_min, end_min in windows:
        if start_min <= end_min:
            if start_min <= now_min <= end_min:
                proximity = 1.0
                break
            dist = min(abs(now_min - start_min), abs(now_min - end_min))
        else:
            in_window = now_min >= start_min or now_min <= end_min
            if in_window:
                proximity = 1.0
                break
            dist = min(abs(now_min - start_min), abs(now_min - end_min))
        proximity = max(proximity, _clamp01(1.0 - (dist / 30.0)))

    return {
        "session_bucket_norm": bucket,
        "mins_from_open_norm": mins_from_open,
        "mins_to_close_norm": mins_to_close,
        "event_window_proximity": proximity,
    }


def _snapshot_health_context(project_root: Path) -> Tuple[Dict[str, float], Dict[str, Any]]:
    health = project_root / "governance" / "health"

    coverage = _safe_load_json(health / "snapshot_coverage_latest.json", default={})
    replay = _safe_load_json(health / "replay_preopen_sanity_latest.json", default={})
    drift = _safe_load_json(health / "preopen_replay_drift_latest.json", default={})
    divergence = _safe_load_json(health / "data_source_divergence_latest.json", default={})
    triprate = _safe_load_json(health / "guardrail_triprate_latest.json", default={})
    queue_stress = _safe_load_json(health / "execution_queue_stress_latest.json", default={})

    coverage_ratio = _to_float(coverage.get("coverage_ratio"), 0.0)
    coverage_log_ratio = _clamp01(math.log1p(max(coverage_ratio, 0.0)) / 6.0)

    replay_decision_stale = _to_float((replay.get("decision") or {}).get("stale_windows"), 0.0)
    replay_governance_stale = _to_float((replay.get("governance") or {}).get("stale_windows"), 0.0)
    replay_max_decision_stale = max(_to_float((replay.get("thresholds") or {}).get("max_decision_stale_windows"), 12.0), 1.0)
    replay_max_governance_stale = max(_to_float((replay.get("thresholds") or {}).get("max_governance_stale_windows"), 12.0), 1.0)
    replay_stale_ratio = _clamp01((replay_decision_stale + replay_governance_stale) / (replay_max_decision_stale + replay_max_governance_stale))

    drift_obj = drift.get("drift") or {}
    thresholds_obj = drift.get("thresholds") or {}
    row_drift = max(abs(_to_float(drift_obj.get("decision_rows"), 0.0)), abs(_to_float(drift_obj.get("governance_rows"), 0.0)))
    stale_drift = max(abs(_to_float(drift_obj.get("decision_stale"), 0.0)), abs(_to_float(drift_obj.get("governance_stale"), 0.0)))
    max_row_drift = max(_to_float(thresholds_obj.get("max_row_drift"), 1.2), 1e-6)
    max_stale_drift = max(_to_float(thresholds_obj.get("max_stale_drift"), 1.0), 1e-6)
    replay_drift_ratio = _clamp01((0.6 * (row_drift / max_row_drift)) + (0.4 * (stale_drift / max_stale_drift)))

    worst_spread = _to_float(divergence.get("worst_relative_spread"), 0.0)
    max_spread = max(_to_float(divergence.get("max_relative_spread"), 0.03), 1e-6)
    divergence_ratio = _clamp01(worst_spread / max_spread)

    trip_rate = _to_float(triprate.get("trip_rate"), 0.0)
    max_trip_rate = max(_to_float(triprate.get("max_trip_rate"), 0.4), 1e-6)
    triprate_ratio = _clamp01(trip_rate / max_trip_rate)

    depth_seen = _to_float(queue_stress.get("max_queue_depth_seen"), 0.0)
    depth_max = max(_to_float(queue_stress.get("max_queue_depth"), 2000.0), 1.0)
    depth_ratio = _clamp01(depth_seen / depth_max)
    breach_rate = _to_float(queue_stress.get("queue_breach_rate"), 0.0)
    breach_rate_max = max(_to_float(queue_stress.get("max_queue_breach_rate"), 0.25), 1e-6)
    breach_ratio = _clamp01(breach_rate / breach_rate_max)
    queue_pressure_ratio = _clamp01(max(depth_ratio, breach_ratio))

    canary_weight_cap_norm = _clamp01(_to_float(os.getenv("CANARY_MAX_WEIGHT", "0.08"), 0.08) / 0.20)

    context = {
        "snapshot_cov_ok": 1.0 if bool(coverage.get("ok", False)) else 0.0,
        "snapshot_cov_log_ratio": coverage_log_ratio,
        "snapshot_replay_stale_ratio": replay_stale_ratio,
        "snapshot_replay_drift_ratio": replay_drift_ratio,
        "snapshot_divergence_ratio": divergence_ratio,
        "snapshot_triprate_ratio": triprate_ratio,
        "snapshot_queue_pressure_ratio": queue_pressure_ratio,
        "canary_weight_cap_norm": canary_weight_cap_norm,
    }
    meta = {
        "coverage_ts": coverage.get("timestamp_utc"),
        "replay_ts": replay.get("timestamp_utc"),
        "drift_ts": drift.get("timestamp_utc"),
        "divergence_ts": divergence.get("timestamp_utc"),
        "triprate_ts": triprate.get("timestamp_utc"),
        "queue_stress_ts": queue_stress.get("timestamp_utc"),
    }
    return context, meta


def _load_governance_index(paths: List[Path], since_utc: datetime) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for row in _iter_jsonl(paths):
        ts = _parse_ts(row.get("timestamp_utc"))
        if ts is None or ts < since_utc:
            continue
        sid = str(row.get("snapshot_id") or "").strip()
        if not sid:
            continue

        freshness = row.get("feature_freshness") or {}
        latency = row.get("master_latency_slo") or {}
        portfolio = row.get("portfolio") or {}
        cb = row.get("circuit_breakers") or {}
        exec_sim = row.get("execution_sim") or {}

        out[sid] = {
            "active_sub_bots": _to_float(row.get("active_sub_bots"), 0.0),
            "queue_depth": _to_float(portfolio.get("queue_depth"), 0.0),
            "dispatch_qty": _to_float(portfolio.get("dispatch_qty"), 0.0),
            "feature_freshness_ok": 1.0 if bool(freshness.get("ok", True)) else 0.0,
            "feature_freshness_age_ratio": _clamp01(
                _to_float(freshness.get("age_seconds"), 0.0) / max(_to_float(freshness.get("max_age_seconds"), 20.0), 1e-6)
            ),
            "master_latency_slo_ok": 1.0 if bool(latency.get("ok", True)) else 0.0,
            "master_latency_ratio": _clamp01(
                _to_float(latency.get("elapsed_ms"), 0.0) / max(_to_float(latency.get("timeout_ms"), 900.0), 1e-6)
            ),
            "risk_pause_active": 1.0 if any(bool(cb.get(k, False)) for k in ("kill_switch_active", "vol_shock_pause_active", "liquidity_pause_active")) else 0.0,
            "exec_slippage_bps": _to_float(exec_sim.get("slippage_bps"), 0.0),
            "exec_latency_ms": _to_float(exec_sim.get("latency_ms"), 0.0),
            "exec_impact_bps": _to_float(exec_sim.get("impact_bps"), 0.0),
        }
    return out


def _load_exec_history(paths: List[Path], since_utc: datetime) -> Dict[str, List[Tuple[float, float, float, float]]]:
    by_symbol: Dict[str, List[Tuple[float, float, float, float]]] = defaultdict(list)
    for row in _iter_jsonl(paths):
        if str(row.get("layer") or "") != "grand_master":
            continue
        ts = _parse_ts(row.get("timestamp_utc"))
        if ts is None or ts < since_utc:
            continue
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        by_symbol[symbol].append(
            (
                ts.timestamp(),
                _to_float(row.get("slippage_bps"), 0.0),
                _to_float(row.get("latency_ms"), 0.0),
                _to_float(row.get("impact_bps"), 0.0),
            )
        )

    for sym in by_symbol:
        by_symbol[sym].sort(key=lambda x: x[0])
    return by_symbol


def _find_last_exec_metrics(history: List[Tuple[float, float, float, float]], ts_epoch: float) -> Tuple[float, float, float]:
    lo = 0
    hi = len(history)
    while lo < hi:
        mid = (lo + hi) // 2
        if history[mid][0] < ts_epoch:
            lo = mid + 1
        else:
            hi = mid
    idx = lo - 1
    if idx < 0:
        return 0.0, 0.0, 0.0

    # Small rolling average over prior observations; available at decision time.
    start = max(0, idx - 24)
    window = history[start : idx + 1]
    n = max(len(window), 1)
    slip = sum(r[1] for r in window) / n
    lat = sum(r[2] for r in window) / n
    impact = sum(r[3] for r in window) / n
    return slip, lat, impact


def _normalize_action(action: str) -> str:
    a = (action or "").upper().strip()
    if a in {"BUY", "SELL", "HOLD"}:
        return a
    return "HOLD"


def _direction_for_action(action: str) -> float:
    if action == "BUY":
        return 1.0
    if action == "SELL":
        return -1.0
    return 0.0


def _label_from_forward(
    *,
    action: str,
    forward_return: float,
    positive_thr: float,
    negative_thr: float,
    hold_pos_thr: float,
    hold_neg_thr: float,
) -> Tuple[str, float]:
    if action in {"BUY", "SELL"}:
        edge = _direction_for_action(action) * forward_return
        if edge >= positive_thr:
            conf = _clamp01(edge / max(positive_thr, 1e-6))
            return "positive", conf
        if edge <= -negative_thr:
            conf = _clamp01(abs(edge) / max(negative_thr, 1e-6))
            return "negative", conf
        conf = _clamp01(abs(edge) / max(max(positive_thr, negative_thr), 1e-6))
        return "neutral", conf

    abs_ret = abs(forward_return)
    if abs_ret <= hold_pos_thr:
        conf = _clamp01((hold_pos_thr - abs_ret) / max(hold_pos_thr, 1e-6))
        return "positive", conf
    if abs_ret >= hold_neg_thr:
        conf = _clamp01((abs_ret - hold_neg_thr) / max(hold_neg_thr, 1e-6))
        return "negative", conf
    conf = _clamp01(abs_ret / max(hold_neg_thr, 1e-6))
    return "neutral", conf


def _decision_feature_vector(
    *,
    row: Dict[str, Any],
    gov: Dict[str, float],
    lag_exec: Tuple[float, float, float],
    snapshot_context: Dict[str, float],
    event_windows: List[Tuple[int, int]],
) -> Tuple[List[float], str, float]:
    features = row.get("features") or {}
    ts = row["ts_utc"]

    symbol = row["symbol"]
    action = row["action"]
    role_idx = row["role_idx"]

    pct = _to_float(features.get("pct_from_close"), 0.0)
    mom = _to_float(features.get("mom_5m"), 0.0)
    vol = _to_float(features.get("vol_30m"), 0.0)
    range_pos = _clamp01(_to_float(features.get("range_pos"), 0.5))
    spread_bps = abs(_to_float(features.get("spread_bps"), 0.0))

    ctx_vix_pct = _to_float(features.get("ctx_VIX_X_pct_from_close"), _to_float(features.get("ctx_VIX_pct_from_close"), 0.0))
    ctx_uup_pct = _to_float(features.get("ctx_UUP_pct_from_close"), 0.0)

    pnl_proxy = _signed_scale((pct + (0.5 * mom) - (0.25 * vol)) * 100.0, 1.0)
    qty_log = math.log1p(abs(_to_float(row.get("quantity"), 0.0)))

    regime_idx, regime = _regime_index(symbol, features)
    label_confidence_proxy = _clamp01((abs(pct) + (0.5 * abs(mom)) + (0.25 * abs(vol))) * 25.0)

    slip, lat_ms, impact = lag_exec

    session = _session_event_context(ts, event_windows)

    vec = [
        pnl_proxy,
        qty_log,
        role_idx,
        _hash01(symbol),
        _hash01(action),
        ts.weekday() / 6.0,
        ts.hour / 23.0,
        regime_idx,
        label_confidence_proxy,
        _signed_scale(pct, 40.0),
        _signed_scale(mom, 120.0),
        _signed_scale(vol, 60.0),
        range_pos,
        _clamp01(spread_bps / 25.0),
        _signed_scale(ctx_vix_pct, 60.0),
        _signed_scale(ctx_uup_pct, 60.0),
        _clamp01(abs(slip) / 10.0),
        _clamp01(lat_ms / 350.0),
        _clamp01(abs(impact) / 10.0),
        _clamp01(_to_float(gov.get("active_sub_bots"), 0.0) / 60.0),
        _clamp01(_to_float(gov.get("queue_depth"), 0.0) / 1000.0),
        _clamp01(abs(_to_float(gov.get("dispatch_qty"), 0.0)) / 20.0),
        session["session_bucket_norm"],
        session["mins_from_open_norm"],
        session["mins_to_close_norm"],
        session["event_window_proximity"],
        _clamp01(_to_float(gov.get("feature_freshness_ok"), 1.0)),
        _clamp01(_to_float(gov.get("feature_freshness_age_ratio"), 0.0)),
        _clamp01(_to_float(gov.get("master_latency_slo_ok"), 1.0)),
        _clamp01(_to_float(gov.get("master_latency_ratio"), 0.0)),
        _clamp01(_to_float(gov.get("risk_pause_active"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_cov_ok"), 1.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_cov_log_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_replay_stale_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_replay_drift_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_divergence_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_triprate_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("snapshot_queue_pressure_ratio"), 0.0)),
        _clamp01(_to_float(snapshot_context.get("canary_weight_cap_norm"), 0.0)),
    ]

    return vec, regime, label_confidence_proxy


def main() -> int:
    parser = argparse.ArgumentParser(description="Build leak-free behavior dataset from shadow decisions (forward-return labels + rich context).")
    parser.add_argument("--decision-glob", default=str(PROJECT_ROOT / "decision_explanations" / "shadow*" / "decision_explanations_*.jsonl"))
    parser.add_argument("--governance-glob", default=str(PROJECT_ROOT / "governance" / "shadow*" / "master_control_*.jsonl"))
    parser.add_argument("--pnl-attribution-glob", default=str(PROJECT_ROOT / "governance" / "shadow*" / "shadow_pnl_attribution_*.jsonl"))
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "data" / "trade_history" / "trade_learning_dataset.json"))
    parser.add_argument("--policy", default=str(PROJECT_ROOT / "config" / "trade_learning_policy.json"))
    parser.add_argument("--lookback-hours", type=int, default=int(os.getenv("BEHAVIOR_DATASET_LOOKBACK_HOURS", "96")))
    parser.add_argument("--horizon-seconds", type=int, default=int(os.getenv("BEHAVIOR_DATASET_FORWARD_HORIZON_SECONDS", "300")))
    parser.add_argument("--max-examples", type=int, default=int(os.getenv("BEHAVIOR_DATASET_MAX_EXAMPLES", "120000")))
    parser.add_argument("--min-per-symbol", type=int, default=int(os.getenv("BEHAVIOR_DATASET_MIN_PER_SYMBOL", "8")))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    since_utc = now_utc - timedelta(hours=max(args.lookback_hours, 1))

    policy = _safe_load_json(Path(args.policy), default={})
    fw_cfg = policy.get("behavior_forward_labels", {})
    outcome_cfg = policy.get("outcome_learning", {})

    positive_bps = float(fw_cfg.get("positive_bps", 6.0))
    negative_bps = float(fw_cfg.get("negative_bps", 6.0))
    hold_positive_bps = float(fw_cfg.get("hold_positive_max_bps", 4.0))
    hold_negative_bps = float(fw_cfg.get("hold_negative_min_bps", 14.0))

    positive_thr = max(positive_bps, 0.1) / 10000.0
    negative_thr = max(negative_bps, 0.1) / 10000.0
    hold_pos_thr = max(hold_positive_bps, 0.1) / 10000.0
    hold_neg_thr = max(hold_negative_bps, 0.1) / 10000.0

    class_weights = outcome_cfg.get("class_weights", {"positive": 1.35, "neutral": 1.10, "negative": 0.95})
    regime_weights = outcome_cfg.get("regime_sample_weights", {"trend": 1.0, "mean_revert": 1.10, "shock": 1.25, "other": 1.0})

    decision_paths = _resolve_glob_paths(args.decision_glob, root=PROJECT_ROOT)
    if not decision_paths:
        decision_paths = sorted(Path(p) for p in PROJECT_ROOT.glob("decision_explanations/shadow*/decision_explanations_*.jsonl"))

    governance_paths = _resolve_glob_paths(args.governance_glob, root=PROJECT_ROOT)
    if not governance_paths:
        governance_paths = sorted(Path(p) for p in PROJECT_ROOT.glob("governance/shadow*/master_control_*.jsonl"))

    pnl_paths = _resolve_glob_paths(args.pnl_attribution_glob, root=PROJECT_ROOT)
    if not pnl_paths:
        pnl_paths = sorted(Path(p) for p in PROJECT_ROOT.glob("governance/shadow*/shadow_pnl_attribution_*.jsonl"))

    snapshot_context, snapshot_meta = _snapshot_health_context(PROJECT_ROOT)
    event_windows = _event_windows_from_env()

    gov_by_snapshot = _load_governance_index(governance_paths, since_utc=since_utc)
    exec_history = _load_exec_history(pnl_paths, since_utc=since_utc)

    by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    raw_rows = 0
    for row in _iter_jsonl(decision_paths):
        raw_rows += 1
        ts = _parse_ts(row.get("timestamp_utc"))
        if ts is None or ts < since_utc:
            continue
        if str(row.get("strategy") or "") != "grand_master_bot":
            continue

        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol:
            continue

        features = row.get("features") or {}
        last_price = _to_float(features.get("last_price"), 0.0)
        if last_price <= 0.0:
            continue

        action = _normalize_action(str(row.get("action") or "HOLD"))
        mode_label = str(row.get("mode") or "")
        snapshot_id = str((row.get("metadata") or {}).get("snapshot_id") or "").strip()

        by_symbol[symbol].append(
            {
                "timestamp_utc": ts.isoformat(),
                "ts_utc": ts,
                "ts_epoch": ts.timestamp(),
                "symbol": symbol,
                "action": action,
                "quantity": _to_float(row.get("quantity"), 0.0),
                "mode": mode_label,
                "role_idx": _role_index(mode_label),
                "snapshot_id": snapshot_id,
                "features": features,
                "last_price": last_price,
                "gates": row.get("gates") or {},
            }
        )

    examples: List[Dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    regime_counts: Dict[str, Counter] = defaultdict(Counter)
    skipped_no_horizon = 0
    skipped_low_symbol_rows = 0

    for symbol, rows in by_symbol.items():
        rows.sort(key=lambda r: r["ts_epoch"])
        if len(rows) < max(args.min_per_symbol, 2):
            skipped_low_symbol_rows += len(rows)
            continue

        j = 1
        n = len(rows)
        for i in range(n):
            base = rows[i]
            target_ts = base["ts_epoch"] + max(args.horizon_seconds, 30)

            if j <= i:
                j = i + 1
            while j < n and rows[j]["ts_epoch"] < target_ts:
                j += 1

            if j >= n:
                skipped_no_horizon += 1
                continue

            fut = rows[j]
            forward_return = (fut["last_price"] - base["last_price"]) / max(base["last_price"], 1e-6)

            label, label_conf = _label_from_forward(
                action=base["action"],
                forward_return=forward_return,
                positive_thr=positive_thr,
                negative_thr=negative_thr,
                hold_pos_thr=hold_pos_thr,
                hold_neg_thr=hold_neg_thr,
            )

            sid = base.get("snapshot_id") or ""
            gov = gov_by_snapshot.get(sid, {})
            lag_exec = _find_last_exec_metrics(exec_history.get(symbol, []), base["ts_epoch"])

            feats, regime, label_conf_proxy = _decision_feature_vector(
                row=base,
                gov=gov,
                lag_exec=lag_exec,
                snapshot_context=snapshot_context,
                event_windows=event_windows,
            )

            weight = (
                _to_float(class_weights.get(label), 1.0)
                * _to_float(regime_weights.get(regime), 1.0)
                * (0.5 + (0.5 * _clamp01(label_conf)))
            )

            examples.append(
                {
                    "id": len(examples),
                    "timestamp_utc": base["timestamp_utc"],
                    "symbol": symbol,
                    "action": base["action"],
                    "regime": regime,
                    "label": label,
                    "label_confidence": round(label_conf, 6),
                    "label_confidence_proxy": round(label_conf_proxy, 6),
                    "forward_return": round(forward_return, 8),
                    "horizon_seconds": int(args.horizon_seconds),
                    "sample_weight": round(max(weight, 0.05), 6),
                    "features": feats,
                }
            )
            label_counts[label] += 1
            regime_counts[regime][label] += 1

            if args.max_examples > 0 and len(examples) >= args.max_examples:
                break

        if args.max_examples > 0 and len(examples) >= args.max_examples:
            break

    payload = {
        "timestamp_utc": now_utc.isoformat(),
        "schema": "behavior_dataset_v2_forward_labels",
        "lookback_hours": int(args.lookback_hours),
        "horizon_seconds": int(args.horizon_seconds),
        "source": {
            "decision_files": len(decision_paths),
            "governance_files": len(governance_paths),
            "pnl_attribution_files": len(pnl_paths),
            "since_utc": since_utc.isoformat(),
            "raw_decision_rows_scanned": int(raw_rows),
        },
        "thresholds": {
            "positive_bps": positive_bps,
            "negative_bps": negative_bps,
            "hold_positive_max_bps": hold_positive_bps,
            "hold_negative_min_bps": hold_negative_bps,
        },
        "feature_dim": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "snapshot_context": {
            "features": {k: round(_to_float(v), 6) for k, v in snapshot_context.items()},
            "meta": snapshot_meta,
        },
        "rows": len(examples),
        "label_space": ["negative", "neutral", "positive"],
        "label_counts": dict(label_counts),
        "regime_label_counts": {k: dict(v) for k, v in regime_counts.items()},
        "skipped": {
            "no_horizon": int(skipped_no_horizon),
            "low_symbol_rows": int(skipped_low_symbol_rows),
        },
        "data": examples,
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    summary = {
        "timestamp_utc": payload["timestamp_utc"],
        "rows": payload["rows"],
        "feature_dim": payload["feature_dim"],
        "label_counts": payload["label_counts"],
        "regime_label_counts": payload["regime_label_counts"],
        "source": payload["source"],
        "out_file": str(out_path),
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=True))
    else:
        print(json.dumps(summary, ensure_ascii=True, indent=2))

    return 0 if len(examples) >= 50 else 2


if __name__ == "__main__":
    raise SystemExit(main())
