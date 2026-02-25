import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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
    parser = argparse.ArgumentParser(description="Leak/overfit guard using walk-forward train/forward gaps.")
    parser.add_argument("--walk-forward-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"))
    parser.add_argument("--min-runs", type=int, default=8)
    parser.add_argument("--max-overfit-gap", type=float, default=float(__import__("os").getenv("LEAK_GUARD_MAX_OVERFIT_GAP", "0.08")))
    parser.add_argument("--max-severe-overfit-gap", type=float, default=float(__import__("os").getenv("LEAK_GUARD_MAX_SEVERE_OVERFIT_GAP", "0.14")))
    parser.add_argument("--high-train-threshold", type=float, default=float(__import__("os").getenv("LEAK_GUARD_HIGH_TRAIN", "0.90")))
    parser.add_argument("--low-forward-threshold", type=float, default=float(__import__("os").getenv("LEAK_GUARD_LOW_FORWARD", "0.55")))
    parser.add_argument("--max-overfit-offenders", type=int, default=int(__import__("os").getenv("LEAK_GUARD_MAX_OVERFIT_OFFENDERS", "12")))
    parser.add_argument("--max-leak-offenders", type=int, default=int(__import__("os").getenv("LEAK_GUARD_MAX_LEAK_OFFENDERS", "2")))
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "leak_overfit_guard_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    wf = _load(Path(args.walk_forward_file))
    bots = wf.get("bots") if isinstance(wf.get("bots"), dict) else {}

    overfit = []
    severe_overfit = []
    leak_like = []

    for bot_id, row in bots.items():
        if not isinstance(row, dict):
            continue
        runs = _i(row.get("runs"), 0)
        if runs < int(args.min_runs):
            continue

        train_mean = _f(row.get("train_mean"), 0.0)
        forward_mean = _f(row.get("forward_mean"), 0.0)
        delta = _f(row.get("delta"), 0.0)
        gap = train_mean - forward_mean

        rec = {
            "bot_id": str(bot_id),
            "runs": runs,
            "train_mean": round(train_mean, 6),
            "forward_mean": round(forward_mean, 6),
            "delta": round(delta, 6),
            "train_forward_gap": round(gap, 6),
        }

        if gap > float(args.max_overfit_gap):
            overfit.append(rec)
        if gap > float(args.max_severe_overfit_gap):
            severe_overfit.append(rec)
        if train_mean >= float(args.high_train_threshold) and forward_mean <= float(args.low_forward_threshold):
            leak_like.append(rec)

    overfit.sort(key=lambda r: (-float(r["train_forward_gap"]), r["bot_id"]))
    severe_overfit.sort(key=lambda r: (-float(r["train_forward_gap"]), r["bot_id"]))
    leak_like.sort(key=lambda r: (-float(r["train_mean"]), float(r["forward_mean"]), r["bot_id"]))

    ok = (len(overfit) <= int(args.max_overfit_offenders)) and (len(leak_like) <= int(args.max_leak_offenders))

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": bool(ok),
        "thresholds": {
            "min_runs": int(args.min_runs),
            "max_overfit_gap": float(args.max_overfit_gap),
            "max_severe_overfit_gap": float(args.max_severe_overfit_gap),
            "high_train_threshold": float(args.high_train_threshold),
            "low_forward_threshold": float(args.low_forward_threshold),
            "max_overfit_offenders": int(args.max_overfit_offenders),
            "max_leak_offenders": int(args.max_leak_offenders),
        },
        "counts": {
            "overfit": len(overfit),
            "severe_overfit": len(severe_overfit),
            "leak_like": len(leak_like),
        },
        "overfit_examples": overfit[:40],
        "severe_overfit_examples": severe_overfit[:40],
        "leak_like_examples": leak_like[:40],
    }

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "leak_overfit_guard "
            f"ok={str(payload['ok']).lower()} "
            f"overfit={len(overfit)} severe={len(severe_overfit)} leak_like={len(leak_like)}"
        )

    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
