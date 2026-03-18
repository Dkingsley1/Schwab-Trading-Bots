import argparse
import glob
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


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


def _stale_windows_from_stamps(stamps: List[datetime], stale_seconds: int) -> int:
    if len(stamps) < 2:
        return 0
    stamps = sorted(stamps)
    gaps = 0
    for i in range(1, len(stamps)):
        if (stamps[i] - stamps[i - 1]).total_seconds() > stale_seconds:
            gaps += 1
    return gaps


def _summarize_watchdog(rows: Iterable[Dict[str, Any]]) -> dict[str, int]:
    events = 0
    restarts = 0
    throttled = 0
    restart_errors = 0
    for row in rows:
        events += 1
        for t in row.get("targets", []) or []:
            action = str((t or {}).get("action", "none"))
            if action == "restart":
                restarts += 1
            elif action == "throttled":
                throttled += 1
            elif action == "error":
                restart_errors += 1
    return {
        "events": events,
        "restarts": restarts,
        "throttled": throttled,
        "restart_errors": restart_errors,
    }


def _summarize_status_rows(rows: Iterable[Dict[str, Any]], *, stale_seconds: int, skipped_statuses: set[str]) -> dict[str, Any]:
    total_rows = 0
    skipped = 0
    status_counts: Counter[str] = Counter()
    stamps: List[datetime] = []

    for row in rows:
        total_rows += 1
        status = str(row.get("status", "UNKNOWN"))
        status_counts[status] += 1
        if status in skipped_statuses:
            skipped += 1
        ts = _parse_ts(row)
        if ts is not None:
            stamps.append(ts)

    return {
        "rows": total_rows,
        "skipped_decisions": skipped,
        "status_counts": dict(status_counts),
        "stale_windows": _stale_windows_from_stamps(stamps, stale_seconds),
    }


def _summarize_rows(rows: Iterable[Dict[str, Any]], *, stale_seconds: int) -> dict[str, Any]:
    total_rows = 0
    stamps: List[datetime] = []
    for row in rows:
        total_rows += 1
        ts = _parse_ts(row)
        if ts is not None:
            stamps.append(ts)
    return {
        "rows": total_rows,
        "stale_windows": _stale_windows_from_stamps(stamps, stale_seconds),
    }


def _summarize_events(rows: Iterable[Dict[str, Any]]) -> dict[str, Any]:
    total_rows = 0
    event_counts: Counter[str] = Counter()
    for row in rows:
        total_rows += 1
        event_counts[str(row.get("event", "none"))] += 1
    return {"rows": total_rows, "event_counts": dict(event_counts)}


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

    skipped_statuses = {"BLOCKED", "DATA_ONLY_BLOCKED"}
    watchdog_summary = _summarize_watchdog(row for f in watchdog_files for row in _iter_jsonl(f))
    decision_summary = _summarize_status_rows(
        (row for f in decision_files for row in _iter_jsonl(f)),
        stale_seconds=args.stale_seconds,
        skipped_statuses=skipped_statuses,
    )
    governance_summary = _summarize_rows(
        (row for f in governance_files for row in _iter_jsonl(f)),
        stale_seconds=args.stale_seconds,
    )
    retrain_summary = _summarize_events(row for f in retrain_files for row in _iter_jsonl(f))

    payload = {
        "day": day,
        "watchdog": {
            "events": watchdog_summary["events"],
            "restarts": watchdog_summary["restarts"],
            "throttled": watchdog_summary["throttled"],
            "restart_errors": watchdog_summary["restart_errors"],
            "files": [str(x) for x in watchdog_files],
        },
        "decision": {
            "rows": decision_summary["rows"],
            "skipped_decisions": decision_summary["skipped_decisions"],
            "status_counts": decision_summary["status_counts"],
            "stale_windows": decision_summary["stale_windows"],
            "files": [str(x) for x in decision_files],
        },
        "governance": {
            "rows": governance_summary["rows"],
            "stale_windows": governance_summary["stale_windows"],
            "files": [str(x) for x in governance_files],
        },
        "retrain": {
            "rows": retrain_summary["rows"],
            "event_counts": retrain_summary["event_counts"],
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
    if retrain_summary["event_counts"]:
        print("Retrain events: " + ", ".join(f"{k}={v}" for k, v in sorted(retrain_summary["event_counts"].items())))


if __name__ == "__main__":
    main()
