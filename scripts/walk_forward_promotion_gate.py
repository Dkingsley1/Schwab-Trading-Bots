import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Promotion gate based on walk-forward metrics.")
    parser.add_argument("--in-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"))
    parser.add_argument("--min-forward-mean", type=float, default=0.53)
    parser.add_argument("--min-delta", type=float, default=-0.01)
    parser.add_argument("--max-fail-share", type=float, default=0.25)
    parser.add_argument("--min-runs-per-bot", type=int, default=6)
    parser.add_argument("--min-considered-bots", type=int, default=8)
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_gate_latest.json"))
    args = parser.parse_args()

    in_path = Path(args.in_file)
    if not in_path.exists():
        raise SystemExit(f"walk-forward file missing: {in_path}")

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    bots = payload.get("bots", {}) if isinstance(payload, dict) else {}

    considered = 0
    fails = 0
    fail_reasons = []
    for bot_id, row in bots.items():
        if not isinstance(row, dict):
            continue
        runs = int(row.get("runs", 0) or 0)
        if runs < args.min_runs_per_bot:
            continue
        if row.get("status") == "insufficient_runs":
            continue

        considered += 1
        fwd = float(row.get("forward_mean", 0.0) or 0.0)
        delta = float(row.get("delta", 0.0) or 0.0)
        ok = (fwd >= args.min_forward_mean) and (delta >= args.min_delta)
        if not ok:
            fails += 1
            fail_reasons.append({"bot_id": bot_id, "runs": runs, "forward_mean": fwd, "delta": delta})

    fail_share = fails / max(considered, 1)
    coverage_ok = considered >= args.min_considered_bots
    promote_ok = coverage_ok and (fail_share <= args.max_fail_share)

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "considered_bots": considered,
        "failed_bots": fails,
        "fail_share": round(fail_share, 6),
        "coverage_ok": coverage_ok,
        "promote_ok": promote_ok,
        "thresholds": {
            "min_forward_mean": args.min_forward_mean,
            "min_delta": args.min_delta,
            "max_fail_share": args.max_fail_share,
            "min_runs_per_bot": args.min_runs_per_bot,
            "min_considered_bots": args.min_considered_bots,
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
