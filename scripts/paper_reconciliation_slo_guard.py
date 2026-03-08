import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_iso_utc(value: str) -> datetime | None:
    raw = str(value or "").strip().replace("Z", "+00:00")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
    except Exception:
        return []
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper order lifecycle reconciliation SLO guard.")
    parser.add_argument("--lookback-minutes", type=int, default=60)
    parser.add_argument("--max-mismatch-rate", type=float, default=0.02)
    parser.add_argument("--max-error-rate", type=float, default=0.03)
    parser.add_argument("--max-staleness-minutes", type=float, default=10.0)
    parser.add_argument("--min-events", type=int, default=0)
    parser.add_argument("--in-file", default="")
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "paper_reconciliation_slo_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    day = now.strftime("%Y%m%d")
    src = Path(args.in_file) if args.in_file else (PROJECT_ROOT / "governance" / "events" / f"paper_execution_guard_{day}.jsonl")

    rows = _read_jsonl(src)
    start = now - timedelta(minutes=max(int(args.lookback_minutes), 1))
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("event", "")).strip() != "paper_order_lifecycle_reconcile":
            continue
        ts = _parse_iso_utc(str(row.get("timestamp_utc", "")))
        if ts is None or ts < start:
            continue
        filtered.append(row)

    total = len(filtered)
    mismatch_count = 0
    error_count = 0
    last_ts: datetime | None = None
    for row in filtered:
        ts = _parse_iso_utc(str(row.get("timestamp_utc", "")))
        if ts and (last_ts is None or ts > last_ts):
            last_ts = ts
        status = str(row.get("status", "")).strip().lower()
        if status == "mismatch":
            mismatch_count += 1
        elif status == "error":
            error_count += 1

    mismatch_rate = (mismatch_count / total) if total > 0 else 0.0
    error_rate = (error_count / total) if total > 0 else 0.0
    staleness_minutes = ((now - last_ts).total_seconds() / 60.0) if last_ts else 1e9

    failed: list[str] = []
    if total < max(int(args.min_events), 0):
        failed.append("events_low")
    if mismatch_rate > float(args.max_mismatch_rate):
        failed.append("mismatch_rate")
    if error_rate > float(args.max_error_rate):
        failed.append("error_rate")
    if total > 0 and staleness_minutes > float(args.max_staleness_minutes):
        failed.append("staleness")

    out = {
        "timestamp_utc": now.isoformat(),
        "ok": len(failed) == 0,
        "source_file": str(src),
        "lookback_minutes": int(args.lookback_minutes),
        "metrics": {
            "reconcile_events": int(total),
            "mismatch_count": int(mismatch_count),
            "error_count": int(error_count),
            "mismatch_rate": round(float(mismatch_rate), 6),
            "error_rate": round(float(error_rate), 6),
            "last_reconcile_timestamp_utc": last_ts.isoformat() if last_ts else "",
            "staleness_minutes": round(float(staleness_minutes), 4),
        },
        "thresholds": {
            "min_events": int(args.min_events),
            "max_mismatch_rate": float(args.max_mismatch_rate),
            "max_error_rate": float(args.max_error_rate),
            "max_staleness_minutes": float(args.max_staleness_minutes),
        },
        "failed_checks": failed,
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(
            "paper_reconciliation_slo "
            f"ok={int(bool(out['ok']))} events={total} "
            f"mismatch_rate={mismatch_rate:.6f}/{float(args.max_mismatch_rate):.6f} "
            f"error_rate={error_rate:.6f}/{float(args.max_error_rate):.6f}"
        )

    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
