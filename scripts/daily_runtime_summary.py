import argparse
import glob
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _parse_ts(row: Dict[str, Any]) -> Optional[datetime]:
    ts = row.get("timestamp_utc")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _iter_files(pattern: str) -> Iterable[Path]:
    for p in sorted(glob.glob(pattern)):
        yield Path(p)


def _stale_windows(rows: List[Dict[str, Any]], stale_seconds: int) -> int:
    stamps = [s for s in (_parse_ts(r) for r in rows) if s is not None]
    if len(stamps) < 2:
        return 0
    stamps = sorted(stamps)
    gaps = 0
    for i in range(1, len(stamps)):
        if (stamps[i] - stamps[i - 1]).total_seconds() > stale_seconds:
            gaps += 1
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily runtime ops summary (watchdog + decisions + governance).")
    parser.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y%m%d"), help="UTC day in YYYYMMDD")
    parser.add_argument("--stale-seconds", type=int, default=180)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    day = args.day

    watchdog_files = list(_iter_files(str(PROJECT_ROOT / "governance" / "watchdog" / f"watchdog_events_{day}.jsonl")))
    decision_files = list(_iter_files(str(PROJECT_ROOT / "decision_explanations" / "shadow*" / f"decision_explanations_{day}.jsonl")))
    governance_files = list(_iter_files(str(PROJECT_ROOT / "governance" / "shadow*" / f"master_control_{day}.jsonl")))
    retrain_files = list(_iter_files(str(PROJECT_ROOT / "governance" / "shadow*" / f"auto_retrain_events_{day}.jsonl")))

    watchdog_rows: List[Dict[str, Any]] = []
    for f in watchdog_files:
        watchdog_rows.extend(_load_jsonl(f))

    decision_rows: List[Dict[str, Any]] = []
    for f in decision_files:
        decision_rows.extend(_load_jsonl(f))

    governance_rows: List[Dict[str, Any]] = []
    for f in governance_files:
        governance_rows.extend(_load_jsonl(f))

    retrain_rows: List[Dict[str, Any]] = []
    for f in retrain_files:
        retrain_rows.extend(_load_jsonl(f))

    restarts = 0
    throttled = 0
    restart_errors = 0
    for row in watchdog_rows:
        for t in row.get("targets", []) or []:
            action = str((t or {}).get("action", "none"))
            if action == "restart":
                restarts += 1
            elif action == "throttled":
                throttled += 1
            elif action == "error":
                restart_errors += 1

    skipped_statuses = {"BLOCKED", "DATA_ONLY_BLOCKED"}
    skipped_decisions = sum(1 for r in decision_rows if str(r.get("status", "")) in skipped_statuses)
    status_counts = Counter(str(r.get("status", "UNKNOWN")) for r in decision_rows)

    retrain_events = Counter(str(r.get("event", "none")) for r in retrain_rows)

    payload = {
        "day": day,
        "watchdog": {
            "events": len(watchdog_rows),
            "restarts": restarts,
            "throttled": throttled,
            "restart_errors": restart_errors,
            "files": [str(x) for x in watchdog_files],
        },
        "decision": {
            "rows": len(decision_rows),
            "skipped_decisions": skipped_decisions,
            "status_counts": dict(status_counts),
            "stale_windows": _stale_windows(decision_rows, args.stale_seconds),
            "files": [str(x) for x in decision_files],
        },
        "governance": {
            "rows": len(governance_rows),
            "stale_windows": _stale_windows(governance_rows, args.stale_seconds),
            "files": [str(x) for x in governance_files],
        },
        "retrain": {
            "rows": len(retrain_rows),
            "event_counts": dict(retrain_events),
            "files": [str(x) for x in retrain_files],
        },
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    print(
        "SUMMARY {day} | restarts={restarts} throttled={throttled} restart_errors={errors} "
        "| decision_rows={drows} skipped={skipped} decision_stale_windows={dstale} "
        "| governance_rows={grows} governance_stale_windows={gstale} "
        "| retrain_rows={rrows}".format(
            day=payload["day"],
            restarts=payload["watchdog"]["restarts"],
            throttled=payload["watchdog"]["throttled"],
            errors=payload["watchdog"]["restart_errors"],
            drows=payload["decision"]["rows"],
            skipped=payload["decision"]["skipped_decisions"],
            dstale=payload["decision"]["stale_windows"],
            grows=payload["governance"]["rows"],
            gstale=payload["governance"]["stale_windows"],
            rrows=payload["retrain"]["rows"],
        )
    )
    if retrain_events:
        print("Retrain events: " + ", ".join(f"{k}={v}" for k, v in sorted(retrain_events.items())))


if __name__ == "__main__":
    main()
