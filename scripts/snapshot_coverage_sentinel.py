import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_CONTROL_DAY_RE = re.compile(r"master_control_(\d{8})\.jsonl$")
REVERSE_SCAN_BLOCK_BYTES = 256 * 1024


def _parse_ts(raw: Any):
    if not raw:
        return None
    s = str(raw).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


def _latest_heartbeat_symbols_total(project_root: Path = PROJECT_ROOT) -> int:
    hb_dir = project_root / "governance" / "health"
    best_ts = None
    best_total = 0
    for p in hb_dir.glob("shadow_loop_*.json"):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
            ts = _parse_ts(row.get("timestamp_utc"))
            if ts is None:
                continue
            if best_ts is None or ts > best_ts:
                best_ts = ts
                best_total = int(row.get("symbols_total", 0) or 0)
        except Exception:
            continue
    return max(best_total, 1)


def _master_control_day(path: Path):
    match = MASTER_CONTROL_DAY_RE.search(path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _file_overlaps_window(path: Path, since: datetime) -> bool:
    day_start = _master_control_day(path)
    if day_start is not None:
        return (day_start + timedelta(days=1)) > since
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return True
    return mtime >= (since - timedelta(hours=1))


def _candidate_master_control_files(project_root: Path, since: datetime) -> list[Path]:
    files = [
        path
        for path in (project_root / "governance").glob("shadow*/master_control_*.jsonl")
        if _file_overlaps_window(path, since)
    ]
    files.sort(key=lambda path: (path.parent.name, path.name))
    return files


def _iter_recent_jsonl_rows(path: Path, since: datetime, *, block_bytes: int = REVERSE_SCAN_BLOCK_BYTES):
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            if position <= 0:
                return
            pending = b""
            seen_recent = False
            while position > 0:
                size = min(max(int(block_bytes), 1024), position)
                position -= size
                handle.seek(position)
                block = handle.read(size)
                if not block:
                    break
                pending = block + pending
                lines = pending.splitlines()
                if position > 0:
                    pending = lines[0] if lines else pending
                    complete_lines = lines[1:]
                else:
                    pending = b""
                    complete_lines = lines
                for raw_line in reversed(complete_lines):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue
                    ts = _parse_ts(row.get("timestamp_utc"))
                    if ts is None:
                        continue
                    if ts < since:
                        if seen_recent:
                            return
                        continue
                    seen_recent = True
                    yield row
    except FileNotFoundError:
        return


def build_payload(
    *,
    hours: int,
    min_coverage_ratio: float,
    project_root: Path = PROJECT_ROOT,
    now: datetime | None = None,
) -> dict[str, Any]:
    current_time = now if now is not None else datetime.now(timezone.utc)
    since = current_time - timedelta(hours=max(int(hours), 1))
    expected_symbols = _latest_heartbeat_symbols_total(project_root)

    total_rows = 0
    snapshot_rows = 0
    unique_snapshot_ids: set[str] = set()
    candidate_files = _candidate_master_control_files(project_root, since)

    for path in candidate_files:
        for row in _iter_recent_jsonl_rows(path, since):
            total_rows += 1
            snapshot_id = row.get("snapshot_id")
            if snapshot_id:
                snapshot_rows += 1
                unique_snapshot_ids.add(str(snapshot_id))

    unique_count = len(unique_snapshot_ids)
    expected_floor = max(expected_symbols, 1)
    coverage_ratio = unique_count / float(expected_floor)
    ok = (total_rows > 0) and (coverage_ratio >= float(min_coverage_ratio))
    return {
        "timestamp_utc": current_time.isoformat(),
        "ok": bool(ok),
        "window_hours": int(hours),
        "expected_symbols_floor": expected_floor,
        "files_considered": len(candidate_files),
        "rows_scanned": total_rows,
        "rows_with_snapshot_id": snapshot_rows,
        "unique_snapshot_ids": unique_count,
        "coverage_ratio": round(coverage_ratio, 6),
        "min_coverage_ratio": float(min_coverage_ratio),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Snapshot coverage sentinel.")
    parser.add_argument("--hours", type=int, default=2)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.75)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        hours=int(args.hours),
        min_coverage_ratio=float(args.min_coverage_ratio),
    )

    out = PROJECT_ROOT / "governance" / "health" / "snapshot_coverage_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print("snapshot_coverage_ok=" + str(payload["ok"]).lower() + f" ratio={payload['coverage_ratio']}")

    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
