import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TS_RE = re.compile(r"_(\d{8})_(\d{6})$")


def _clamp(x: float) -> float:
    return max(0.0, min(float(x), 1.0))


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _trading_quality_score(forward_mean: float, delta: float, forward_std: float, overfit_gap: float) -> float:
    edge_component = _clamp((forward_mean - 0.48) / 0.10)
    generalization_component = _clamp((0.04 + delta) / 0.04)
    stability_component = _clamp((0.08 - forward_std) / 0.08)
    overfit_penalty = _clamp(overfit_gap / 0.15)

    score = (
        0.55 * edge_component
        + 0.25 * generalization_component
        + 0.20 * stability_component
        - 0.25 * overfit_penalty
        + 0.05
    )
    return _clamp(score)


def bot_id_from_log_name(name: str) -> str:
    base = name[:-5] if name.endswith(".json") else name
    m = TS_RE.search(base)
    if not m:
        return base
    return base[: m.start()]


def timestamp_from_log_name(name: str) -> datetime:
    base = name[:-5] if name.endswith(".json") else name
    m = TS_RE.search(base)
    if not m:
        return datetime.min.replace(tzinfo=timezone.utc)
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return dt.replace(tzinfo=timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward style validation over historical bot training logs.")
    parser.add_argument("--min-runs", type=int, default=4)
    parser.add_argument("--pass-forward-threshold", type=float, default=float(os.getenv("WALK_FORWARD_PASS_FORWARD_THRESHOLD", "0.52")))
    parser.add_argument("--pass-delta-threshold", type=float, default=float(os.getenv("WALK_FORWARD_PASS_DELTA_THRESHOLD", "-0.02")))
    parser.add_argument("--min-trading-quality-score", type=float, default=float(os.getenv("WALK_FORWARD_MIN_TRADING_QUALITY_SCORE", "0.48")))
    parser.add_argument("--max-overfit-gap", type=float, default=float(os.getenv("WALK_FORWARD_MAX_OVERFIT_GAP", "0.10")))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"))
    args = parser.parse_args()

    logs_dir = PROJECT_ROOT / "logs"
    groups = defaultdict(list)

    for p in logs_dir.glob("brain_refinery_*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        bot_id = bot_id_from_log_name(p.name)
        ts = timestamp_from_log_name(p.name)
        acc = (obj.get("metrics") or {}).get("test_accuracy")
        if acc is None:
            continue
        groups[bot_id].append((ts, float(acc)))

    report = {}
    for bot_id, vals in groups.items():
        vals.sort(key=lambda x: x[0])
        if len(vals) < args.min_runs:
            report[bot_id] = {"runs": len(vals), "status": "insufficient_runs"}
            continue

        split = max(int(len(vals) * 0.7), 1)
        train_part = [x[1] for x in vals[:split]]
        fwd_part = [x[1] for x in vals[split:]]
        if not fwd_part:
            fwd_part = [train_part[-1]]

        train_mean = mean(train_part)
        fwd_mean = mean(fwd_part)
        delta = fwd_mean - train_mean
        forward_std = pstdev(fwd_part) if len(fwd_part) > 1 else 0.0
        overfit_gap = train_mean - fwd_mean
        trading_quality_score = _trading_quality_score(
            forward_mean=float(fwd_mean),
            delta=float(delta),
            forward_std=float(forward_std),
            overfit_gap=float(overfit_gap),
        )

        status = "pass"
        if fwd_mean < float(args.pass_forward_threshold):
            status = "fail"
        if delta < float(args.pass_delta_threshold):
            status = "fail"
        if trading_quality_score < float(args.min_trading_quality_score):
            status = "fail"
        if overfit_gap > float(args.max_overfit_gap):
            status = "fail"

        report[bot_id] = {
            "runs": len(vals),
            "train_mean": round(train_mean, 6),
            "forward_mean": round(fwd_mean, 6),
            "delta": round(delta, 6),
            "forward_std": round(forward_std, 6),
            "overfit_gap": round(overfit_gap, 6),
            "trading_quality_score": round(trading_quality_score, 6),
            "status": status,
        }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "min_runs": args.min_runs,
        "thresholds": {
            "pass_forward_threshold": float(args.pass_forward_threshold),
            "pass_delta_threshold": float(args.pass_delta_threshold),
            "min_trading_quality_score": float(args.min_trading_quality_score),
            "max_overfit_gap": float(args.max_overfit_gap),
        },
        "bots": report,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
