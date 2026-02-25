import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHOCK_SYMBOLS = {"UVXY", "VIXY", "SOXL", "SOXS", "MSTR", "SMCI", "COIN", "TSLA"}
MEAN_REVERT_SYMBOLS = {"TLT", "IEF", "SHY", "BND", "AGG", "GLD", "XLU", "XLP"}


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


def _to_epoch(ts: Any) -> float:
    if not ts:
        return 0.0
    s = str(ts)
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).timestamp()
    except Exception:
        return 0.0


def _hash01(text: str) -> float:
    if not text:
        return 0.0
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h / 0xFFFFFFFF


def _role_index(role: str) -> int:
    r = (role or "").upper()
    if r == "ROTH":
        return 0
    if r == "INDIVIDUAL_TRADING":
        return 1
    if r == "INDIVIDUAL_SWING":
        return 2
    return 3


def _regime_bucket(symbol: str, tx_type: str, pnl: float) -> str:
    s = (symbol or "").upper()
    t = (tx_type or "").lower()
    if s in SHOCK_SYMBOLS or abs(float(pnl)) >= 300.0:
        return "shock"
    if s in MEAN_REVERT_SYMBOLS or any(k in t for k in ("bond", "dividend", "rebalance", "income")):
        return "mean_revert"
    if any(k in t for k in ("trend", "breakout", "momentum", "swing")):
        return "trend"
    return "other"


def _regime_index(regime: str) -> int:
    r = (regime or "other").lower()
    if r == "trend":
        return 0
    if r == "mean_revert":
        return 1
    if r == "shock":
        return 2
    return 3


def _label_from_pnl(pnl: float, positive_threshold: float, negative_threshold: float) -> str:
    if pnl >= positive_threshold:
        return "positive"
    if pnl <= -negative_threshold:
        return "negative"
    return "neutral"


def _sample_weight(label: str, weights: Dict[str, float], regime: str, regime_weights: Dict[str, float]) -> float:
    w = float(weights.get(label, 1.0)) * float(regime_weights.get(regime, 1.0))
    return max(w, 0.05)


def _build_features(row: Dict[str, Any], pnl_scale: float, regime: str, label_confidence: float) -> List[float]:
    pnl = _to_float(row.get("pnl"), 0.0)
    qty = abs(_to_float(row.get("quantity"), 0.0))

    symbol = str(row.get("symbol") or "")
    tx_type = str(row.get("transaction_type") or "")
    role = str(row.get("account_role") or "UNKNOWN")

    pnl_scaled = math.tanh(pnl / max(pnl_scale, 1.0))
    qty_log = math.log1p(qty)
    role_idx = _role_index(role) / 3.0
    symbol_hash = _hash01(symbol)
    tx_hash = _hash01(tx_type)
    ts_epoch = _to_epoch(row.get("timestamp_utc"))
    dow = ((int(ts_epoch) // 86400) + 4) % 7 if ts_epoch > 0 else 0
    hour = (int(ts_epoch) // 3600) % 24 if ts_epoch > 0 else 0
    regime_idx = _regime_index(regime) / 3.0

    return [
        pnl_scaled,
        qty_log,
        role_idx,
        symbol_hash,
        tx_hash,
        dow / 6.0,
        hour / 23.0,
        regime_idx,
        max(0.0, min(label_confidence, 1.0)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build behavior-learning dataset from trade history (regime-aware labels).")
    parser.add_argument("--in-file", default=str(PROJECT_ROOT / "data" / "trade_history" / "trades_normalized.jsonl"))
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "data" / "trade_history" / "trade_learning_dataset.json"))
    parser.add_argument("--policy", default=str(PROJECT_ROOT / "config" / "trade_learning_policy.json"))
    args = parser.parse_args()

    in_path = Path(args.in_file)
    out_path = Path(args.out_file)
    policy = _safe_load_json(Path(args.policy), default={})

    if not in_path.exists():
        fallback = PROJECT_ROOT / "data" / "trade_history" / "trades_training_filtered.jsonl"
        if fallback.exists():
            in_path = fallback
            print(f"Input not found, using fallback: {in_path}")
        else:
            print(
                f"Missing input trade file: {in_path}\n"
                "Run import_trade_history.py first, then re-run this dataset build."
            )
            return 2

    outcome_cfg = policy.get("outcome_learning", {})
    weak_cfg = policy.get("weak_regime_label_quality", {})

    positive_threshold = float(outcome_cfg.get("positive_pnl_threshold", 0.0))
    negative_threshold = float(outcome_cfg.get("negative_pnl_threshold", 0.0))
    pnl_scale = float(outcome_cfg.get("pnl_scale", 250.0))
    weights = outcome_cfg.get("class_weights", {"positive": 1.4, "neutral": 1.0, "negative": 0.9})

    regime_threshold_multipliers = outcome_cfg.get(
        "regime_threshold_multipliers",
        {"trend": 1.0, "mean_revert": 1.15, "shock": 1.35, "other": 1.0},
    )
    regime_weights = outcome_cfg.get(
        "regime_sample_weights",
        {"trend": 1.0, "mean_revert": 1.10, "shock": 1.25, "other": 1.0},
    )

    weak_regimes = set(weak_cfg.get("regimes", ["mean_revert", "shock"]))
    min_confidence = float(weak_cfg.get("min_confidence", 0.35))
    drop_ambiguous = bool(weak_cfg.get("drop_ambiguous", True))

    rows: List[Dict[str, Any]] = []
    skipped_lines = 0
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                skipped_lines += 1

    examples: List[Dict[str, Any]] = []
    by_role_labels = defaultdict(lambda: defaultdict(int))
    by_role_pnl = defaultdict(list)
    by_regime_labels = defaultdict(lambda: defaultdict(int))
    skipped_ambiguous = 0

    for idx, row in enumerate(rows):
        pnl = _to_float(row.get("pnl"), 0.0)
        role = str(row.get("account_role") or "UNKNOWN")
        symbol = str(row.get("symbol") or "")
        tx_type = str(row.get("transaction_type") or "")

        regime = _regime_bucket(symbol=symbol, tx_type=tx_type, pnl=pnl)
        regime_mult = float(regime_threshold_multipliers.get(regime, 1.0))

        pos_thr = positive_threshold * max(regime_mult, 0.1)
        neg_thr = negative_threshold * max(regime_mult, 0.1)
        label = _label_from_pnl(pnl, positive_threshold=pos_thr, negative_threshold=neg_thr)

        denom = max(pos_thr if pnl >= 0 else neg_thr, 1.0)
        label_confidence = min(abs(pnl) / denom, 1.0)

        if drop_ambiguous and regime in weak_regimes and label == "neutral" and label_confidence < min_confidence:
            skipped_ambiguous += 1
            continue

        features = _build_features(row, pnl_scale=pnl_scale, regime=regime, label_confidence=label_confidence)
        weight = _sample_weight(label, weights, regime=regime, regime_weights=regime_weights)

        examples.append(
            {
                "id": idx,
                "timestamp_utc": row.get("timestamp_utc"),
                "account_role": role,
                "symbol": symbol,
                "transaction_type": tx_type,
                "regime": regime,
                "label_confidence": round(label_confidence, 6),
                "pnl": pnl,
                "label": label,
                "sample_weight": weight,
                "features": features,
            }
        )

        by_role_labels[role][label] += 1
        by_role_pnl[role].append(pnl)
        by_regime_labels[regime][label] += 1

    role_profiles = {}
    for role, pnl_vals in by_role_pnl.items():
        wins = sum(1 for x in pnl_vals if x > 0)
        losses = sum(1 for x in pnl_vals if x < 0)
        role_profiles[role] = {
            "count": len(pnl_vals),
            "net_pnl": round(sum(pnl_vals), 2),
            "win_rate": round(wins / max(wins + losses, 1), 4),
            "avg_pnl": round(sum(pnl_vals) / max(len(pnl_vals), 1), 4),
            "label_counts": dict(by_role_labels[role]),
        }

    global_counts = defaultdict(int)
    for ex in examples:
        global_counts[str(ex["label"])] += 1

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "examples": len(examples),
        "skipped_lines": skipped_lines,
        "skipped_ambiguous": skipped_ambiguous,
        "source": str(in_path),
        "feature_dim": 9,
        "label_space": ["negative", "neutral", "positive"],
        "label_counts": dict(global_counts),
        "regime_label_counts": {k: dict(v) for k, v in by_regime_labels.items()},
        "outcome_learning": {
            "positive_pnl_threshold": positive_threshold,
            "negative_pnl_threshold": negative_threshold,
            "pnl_scale": pnl_scale,
            "class_weights": {
                "positive": float(weights.get("positive", 1.4)),
                "neutral": float(weights.get("neutral", 1.0)),
                "negative": float(weights.get("negative", 0.9)),
            },
            "regime_threshold_multipliers": {
                "trend": float(regime_threshold_multipliers.get("trend", 1.0)),
                "mean_revert": float(regime_threshold_multipliers.get("mean_revert", 1.15)),
                "shock": float(regime_threshold_multipliers.get("shock", 1.35)),
                "other": float(regime_threshold_multipliers.get("other", 1.0)),
            },
            "regime_sample_weights": {
                "trend": float(regime_weights.get("trend", 1.0)),
                "mean_revert": float(regime_weights.get("mean_revert", 1.10)),
                "shock": float(regime_weights.get("shock", 1.25)),
                "other": float(regime_weights.get("other", 1.0)),
            },
        },
        "weak_regime_label_quality": {
            "regimes": sorted(list(weak_regimes)),
            "min_confidence": float(min_confidence),
            "drop_ambiguous": bool(drop_ambiguous),
        },
        "profiles": role_profiles,
        "data": examples,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("timestamp_utc", "examples", "label_counts", "regime_label_counts", "source")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
