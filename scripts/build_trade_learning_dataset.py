import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    return (h / 0xFFFFFFFF)


def _role_index(role: str) -> int:
    r = (role or "").upper()
    if r == "ROTH":
        return 0
    if r == "INDIVIDUAL_TRADING":
        return 1
    if r == "INDIVIDUAL_SWING":
        return 2
    return 3


def _label_from_pnl(pnl: float, positive_threshold: float, negative_threshold: float) -> str:
    if pnl >= positive_threshold:
        return "positive"
    if pnl <= -negative_threshold:
        return "negative"
    return "neutral"


def _sample_weight(label: str, weights: Dict[str, float]) -> float:
    w = float(weights.get(label, 1.0))
    return max(w, 0.05)


def _build_features(row: Dict[str, Any], pnl_scale: float) -> List[float]:
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

    return [
        pnl_scaled,
        qty_log,
        role_idx,
        symbol_hash,
        tx_hash,
        dow / 6.0,
        hour / 23.0,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build behavior-learning dataset from trade history (positive + negative + neutral).")
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
    positive_threshold = float(outcome_cfg.get("positive_pnl_threshold", 0.0))
    negative_threshold = float(outcome_cfg.get("negative_pnl_threshold", 0.0))
    pnl_scale = float(outcome_cfg.get("pnl_scale", 250.0))
    weights = outcome_cfg.get(
        "class_weights",
        {"positive": 1.4, "neutral": 1.0, "negative": 0.9},
    )

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

    for idx, row in enumerate(rows):
        pnl = _to_float(row.get("pnl"), 0.0)
        label = _label_from_pnl(pnl, positive_threshold=positive_threshold, negative_threshold=negative_threshold)
        role = str(row.get("account_role") or "UNKNOWN")

        features = _build_features(row, pnl_scale=pnl_scale)
        weight = _sample_weight(label, weights)

        examples.append(
            {
                "id": idx,
                "timestamp_utc": row.get("timestamp_utc"),
                "account_role": role,
                "symbol": row.get("symbol"),
                "transaction_type": row.get("transaction_type"),
                "pnl": pnl,
                "label": label,
                "sample_weight": weight,
                "features": features,
            }
        )

        by_role_labels[role][label] += 1
        by_role_pnl[role].append(pnl)

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
        "source": str(in_path),
        "feature_dim": 7,
        "label_space": ["negative", "neutral", "positive"],
        "label_counts": dict(global_counts),
        "outcome_learning": {
            "positive_pnl_threshold": positive_threshold,
            "negative_pnl_threshold": negative_threshold,
            "pnl_scale": pnl_scale,
            "class_weights": {
                "positive": float(weights.get("positive", 1.4)),
                "neutral": float(weights.get("neutral", 1.0)),
                "negative": float(weights.get("negative", 0.9)),
            },
        },
        "profiles": role_profiles,
        "data": examples,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("timestamp_utc", "examples", "label_counts", "source")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
