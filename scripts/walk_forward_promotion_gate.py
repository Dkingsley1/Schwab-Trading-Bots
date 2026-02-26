import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _f(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _i(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def main() -> int:
    parser = argparse.ArgumentParser(description="Promotion gate based on walk-forward + trading quality metrics.")
    parser.add_argument("--in-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"))
    parser.add_argument("--min-forward-mean", type=float, default=float(os.getenv("PROMOTION_GATE_MIN_FORWARD_MEAN", "0.53")))
    parser.add_argument("--min-delta", type=float, default=float(os.getenv("PROMOTION_GATE_MIN_DELTA", "-0.01")))
    parser.add_argument("--min-trading-quality-score", type=float, default=float(os.getenv("PROMOTION_GATE_MIN_TRADING_QUALITY_SCORE", "0.52")))
    parser.add_argument("--max-overfit-gap", type=float, default=float(os.getenv("PROMOTION_GATE_MAX_OVERFIT_GAP", "0.10")))
    parser.add_argument("--max-fail-share", type=float, default=float(os.getenv("PROMOTION_GATE_MAX_FAIL_SHARE", "0.25")))
    parser.add_argument("--max-severe-overfit-share", type=float, default=float(os.getenv("PROMOTION_GATE_MAX_SEVERE_OVERFIT_SHARE", "0.10")))
    parser.add_argument("--min-runs-per-bot", type=int, default=int(os.getenv("PROMOTION_GATE_MIN_RUNS", "12")))
    parser.add_argument("--min-considered-bots", type=int, default=int(os.getenv("PROMOTION_GATE_MIN_CONSIDERED", "12")))
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_gate_latest.json"))
    args = parser.parse_args()

    in_path = Path(args.in_file)
    if not in_path.exists():
        raise SystemExit(f"walk-forward file missing: {in_path}")

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    bots = payload.get("bots", {}) if isinstance(payload, dict) else {}

    considered = 0
    fails = 0
    severe_overfit = 0
    tq_sum = 0.0
    fail_reasons = []

    for bot_id, row in bots.items():
        if not isinstance(row, dict):
            continue
        runs = _i(row.get("runs"), 0)
        if runs < int(args.min_runs_per_bot):
            continue
        if row.get("status") == "insufficient_runs":
            continue

        considered += 1
        fwd = _f(row.get("forward_mean"), 0.0)
        delta = _f(row.get("delta"), 0.0)
        tq = _f(row.get("trading_quality_score"), 0.0)
        overfit_gap = _f(row.get("overfit_gap"), _f(row.get("train_mean"), 0.0) - fwd)
        tq_sum += tq

        gate_fwd = fwd >= float(args.min_forward_mean)
        gate_delta = delta >= float(args.min_delta)
        gate_tq = tq >= float(args.min_trading_quality_score)
        gate_overfit = overfit_gap <= float(args.max_overfit_gap)

        ok = gate_fwd and gate_delta and gate_tq and gate_overfit
        if not ok:
            fails += 1
            fail_reasons.append(
                {
                    "bot_id": bot_id,
                    "runs": runs,
                    "forward_mean": round(fwd, 6),
                    "delta": round(delta, 6),
                    "trading_quality_score": round(tq, 6),
                    "overfit_gap": round(overfit_gap, 6),
                    "failed_gates": {
                        "forward_mean": not gate_fwd,
                        "delta": not gate_delta,
                        "trading_quality_score": not gate_tq,
                        "overfit_gap": not gate_overfit,
                    },
                }
            )

        if overfit_gap > float(args.max_overfit_gap) * 1.5:
            severe_overfit += 1

    fail_share = fails / max(considered, 1)
    severe_overfit_share = severe_overfit / max(considered, 1)
    coverage_ok = considered >= int(args.min_considered_bots)
    mean_trading_quality_score = tq_sum / max(considered, 1)

    promote_ok = (
        coverage_ok
        and (fail_share <= float(args.max_fail_share))
        and (severe_overfit_share <= float(args.max_severe_overfit_share))
    )

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "considered_bots": considered,
        "failed_bots": fails,
        "fail_share": round(fail_share, 6),
        "severe_overfit_bots": severe_overfit,
        "severe_overfit_share": round(severe_overfit_share, 6),
        "mean_trading_quality_score": round(mean_trading_quality_score, 6),
        "coverage_ok": coverage_ok,
        "promote_ok": promote_ok,
        "thresholds": {
            "min_forward_mean": float(args.min_forward_mean),
            "min_delta": float(args.min_delta),
            "min_trading_quality_score": float(args.min_trading_quality_score),
            "max_overfit_gap": float(args.max_overfit_gap),
            "max_fail_share": float(args.max_fail_share),
            "max_severe_overfit_share": float(args.max_severe_overfit_share),
            "min_runs_per_bot": int(args.min_runs_per_bot),
            "min_considered_bots": int(args.min_considered_bots),
        },
        "fail_examples": fail_reasons[:30],
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=True))
    return 0 if promote_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
