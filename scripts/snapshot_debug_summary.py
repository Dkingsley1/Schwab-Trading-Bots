import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_ts(raw: Any) -> datetime:
    if not raw:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    s = str(raw).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _iter_debug_files(profile_substr: str) -> Iterable[Path]:
    gov = PROJECT_ROOT / "governance"
    if not gov.exists():
        return []
    files = sorted(gov.glob("shadow*/snapshot_debug_*.jsonl"))
    if profile_substr:
        token = profile_substr.lower()
        files = [p for p in files if token in p.as_posix().lower()]
    return files


def _load_rows(paths: Iterable[Path], since: datetime) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in paths:
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    ts = _parse_ts(row.get("timestamp_utc"))
                    if ts < since:
                        continue
                    row["_ts"] = ts
                    out.append(row)
        except Exception:
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize snapshot debug events.")
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--profile", default="", help="Filter by profile substring, e.g. conservative/aggressive/crypto")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=max(args.hours, 1))

    files = list(_iter_debug_files(args.profile))
    rows = _load_rows(files, since)

    reason_counts: Counter[str] = Counter()
    symbol_reason_counts: Counter[str] = Counter()
    hourly_reason: Dict[str, Counter[str]] = defaultdict(Counter)

    for r in rows:
        reason = str(r.get("reason", "unknown"))
        symbol = str(r.get("symbol", ""))
        ts = r.get("_ts")
        hour = ts.strftime("%Y-%m-%dT%H:00Z") if isinstance(ts, datetime) else "unknown"

        reason_counts[reason] += 1
        if symbol:
            symbol_reason_counts[f"{symbol}|{reason}"] += 1
        hourly_reason[hour][reason] += 1

    payload = {
        "timestamp_utc": now.isoformat(),
        "hours": int(args.hours),
        "since_utc": since.isoformat(),
        "file_count": len(files),
        "row_count": len(rows),
        "reason_counts": dict(reason_counts.most_common()),
        "top_symbol_reason": [
            {"symbol_reason": k, "count": v}
            for k, v in symbol_reason_counts.most_common(max(args.top, 1))
        ],
        "hourly_reason": {
            h: dict(c.most_common())
            for h, c in sorted(hourly_reason.items())
        },
    }

    out = PROJECT_ROOT / "governance" / "health" / "snapshot_debug_summary_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"snapshot_debug_rows={payload['row_count']} files={payload['file_count']} "
            f"window_hours={payload['hours']}"
        )
        print("Top reasons:")
        for reason, count in reason_counts.most_common(max(args.top, 1)):
            print(f" - {reason}: {count}")
        print("Top symbol|reason:")
        for row in payload["top_symbol_reason"]:
            print(f" - {row['symbol_reason']}: {row['count']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
