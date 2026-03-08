import argparse
import glob
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_ts(raw: Any) -> datetime | None:
    s = str(raw or "").strip().replace("Z", "+00:00")
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _bps(fill: float, expected: float, action: str) -> float:
    if fill <= 0 or expected <= 0:
        return 0.0
    a = str(action or "").upper().strip()
    if a.startswith("BUY"):
        return max(((fill - expected) / expected) * 10000.0, 0.0)
    if a.startswith("SELL"):
        return max(((expected - fill) / expected) * 10000.0, 0.0)
    return abs((fill - expected) / expected) * 10000.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper execution calibration drift report.")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--max-mae-bps", type=float, default=35.0)
    ap.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "paper_execution_calibration_latest.json"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    since = datetime.now(timezone.utc) - timedelta(hours=max(int(args.hours), 1))
    vals: list[float] = []
    files_scanned = 0
    for raw in sorted(glob.glob(str(PROJECT_ROOT / "exports" / "trade_logs" / "**" / "paper_trades_*.jsonl"), recursive=True)):
        files_scanned += 1
        p = Path(raw)
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        row = json.loads(s)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    ts = _parse_ts(row.get("timestamp_utc"))
                    if ts is None or ts < since:
                        continue
                    fill = float(row.get("fill_price", 0.0) or 0.0)
                    exp = float(row.get("expected_fill_price", 0.0) or 0.0)
                    if fill <= 0.0 or exp <= 0.0:
                        continue
                    vals.append(_bps(fill, exp, row.get("action")))
        except Exception:
            continue

    vals.sort()
    n = len(vals)
    mae = (sum(vals) / n) if n > 0 else 0.0
    p95 = vals[min(max(int(0.95 * n) - 1, 0), n - 1)] if n > 0 else 0.0

    failed = []
    if n > 0 and mae > float(args.max_mae_bps):
        failed.append("mae_bps")

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": len(failed) == 0,
        "failed_checks": failed,
        "lookback_hours": int(args.hours),
        "files_scanned": int(files_scanned),
        "samples": int(n),
        "metrics": {"mae_bps": round(float(mae), 6), "p95_bps": round(float(p95), 6)},
        "thresholds": {"max_mae_bps": float(args.max_mae_bps)},
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(f"paper_execution_calibration_ok={int(out['ok'])} mae_bps={out['metrics']['mae_bps']:.4f}/{float(args.max_mae_bps):.4f}")
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
