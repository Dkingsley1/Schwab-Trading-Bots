import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


SEGMENTS = {
    "trend": ["trend", "breakout", "donchian"],
    "mean_revert": ["mean_revert", "vwap", "bollinger", "keltner"],
    "shock": ["flash", "shock", "event", "crash", "anomaly"],
    "liquidity": ["liquidity", "spread", "order_flow", "microstructure"],
}


def _segment(bot_id: str) -> str:
    b = (bot_id or "").lower()
    for seg, keys in SEGMENTS.items():
        if any(k in b for k in keys):
            return seg
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description="Regime-segmented validation summary.")
    parser.add_argument("--in-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"))
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "regime_segmented_latest.json"))
    args = parser.parse_args()

    src = Path(args.in_file)
    if not src.exists():
        raise SystemExit(f"missing walk-forward file: {src}")

    payload = json.loads(src.read_text(encoding="utf-8"))
    bots = payload.get("bots", {}) if isinstance(payload, dict) else {}

    acc = {}
    for bot_id, row in bots.items():
        if not isinstance(row, dict) or row.get("status") == "insufficient_runs":
            continue
        seg = _segment(str(bot_id))
        acc.setdefault(seg, {"count": 0, "forward_mean_sum": 0.0, "delta_sum": 0.0, "pass": 0, "fail": 0})
        fwd = float(row.get("forward_mean", 0.0) or 0.0)
        delta = float(row.get("delta", 0.0) or 0.0)
        ok = str(row.get("status", "")).lower() == "pass"
        acc[seg]["count"] += 1
        acc[seg]["forward_mean_sum"] += fwd
        acc[seg]["delta_sum"] += delta
        acc[seg]["pass"] += 1 if ok else 0
        acc[seg]["fail"] += 0 if ok else 1

    out = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "segments": {}}
    for seg, v in acc.items():
        c = max(v["count"], 1)
        out["segments"][seg] = {
            "count": v["count"],
            "forward_mean": round(v["forward_mean_sum"] / c, 6),
            "delta_mean": round(v["delta_sum"] / c, 6),
            "pass_rate": round(v["pass"] / c, 6),
            "fail_count": v["fail"],
        }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
