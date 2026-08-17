import argparse
import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_REASONS = (
    'feature_freshness_guard',
    'master_latency_slo_timeout',
    'event_lock_paused',
    'circuit_open_skip',
)


def _parse_ts(raw):
    if not raw:
        return None
    s = str(raw).replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return int(default)


def _recent_files(root: Path, pattern: str, *, since: datetime, max_files: int) -> tuple[list[Path], int]:
    files: list[tuple[float, Path]] = []
    skipped_by_mtime = 0
    since_ts = since.timestamp()
    for p in root.glob(pattern):
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except Exception:
            continue
        if st.st_mtime < since_ts:
            skipped_by_mtime += 1
            continue
        files.append((float(st.st_mtime), p))
    files.sort(key=lambda row: (row[0], str(row[1])), reverse=True)
    if max_files > 0:
        files = files[:max_files]
    return [p for _, p in files], skipped_by_mtime


def _iter_tail_lines(path: Path, *, tail_bytes: int, max_lines: int):
    try:
        size = path.stat().st_size
    except Exception:
        return
    start = max(int(size) - max(int(tail_bytes), 0), 0) if tail_bytes > 0 else 0
    try:
        with path.open('rb') as f:
            if start > 0:
                f.seek(start)
                f.readline()
            lines_seen = 0
            for raw in f:
                if max_lines > 0 and lines_seen >= max_lines:
                    break
                lines_seen += 1
                yield raw.decode('utf-8', errors='ignore')
    except Exception:
        return


def _json_rows_from_recent_files(
    root: Path,
    pattern: str,
    *,
    since: datetime,
    max_files: int,
    tail_bytes: int,
    max_lines_per_file: int,
):
    files, skipped_by_mtime = _recent_files(root, pattern, since=since, max_files=max_files)
    stats = {
        'files_considered': len(files) + skipped_by_mtime,
        'files_scanned': 0,
        'files_skipped_mtime': skipped_by_mtime,
        'bytes_scanned_estimate': 0,
    }
    for p in files:
        stats['files_scanned'] += 1
        try:
            stats['bytes_scanned_estimate'] += min(int(p.stat().st_size), max(int(tail_bytes), 0) if tail_bytes > 0 else int(p.stat().st_size))
        except Exception:
            pass
        for line in _iter_tail_lines(p, tail_bytes=tail_bytes, max_lines=max_lines_per_file):
            try:
                row = json.loads(line)
            except Exception:
                continue
            yield row, stats


def main() -> int:
    parser = argparse.ArgumentParser(description='Guardrail trip-rate sentinel.')
    parser.add_argument('--hours', type=int, default=6)
    parser.add_argument('--max-trip-rate', type=float, default=0.40)
    parser.add_argument('--max-files', type=int, default=_env_int('GUARDRAIL_TRIPRATE_MAX_FILES', 120))
    parser.add_argument('--tail-bytes', type=int, default=_env_int('GUARDRAIL_TRIPRATE_TAIL_BYTES', 2_000_000))
    parser.add_argument('--max-lines-per-file', type=int, default=_env_int('GUARDRAIL_TRIPRATE_MAX_LINES_PER_FILE', 50000))
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    since = datetime.now(timezone.utc) - timedelta(hours=max(args.hours, 1))
    counters = Counter()
    total = 0
    scan_stats = {
        'files_considered': 0,
        'files_scanned': 0,
        'files_skipped_mtime': 0,
        'bytes_scanned_estimate': 0,
    }

    for row, stats in _json_rows_from_recent_files(
        PROJECT_ROOT / 'governance',
        'shadow*/master_control_*.jsonl',
        since=since,
        max_files=max(args.max_files, 0),
        tail_bytes=max(args.tail_bytes, 0),
        max_lines_per_file=max(args.max_lines_per_file, 0),
    ):
        scan_stats = stats
        ts = _parse_ts(row.get('timestamp_utc'))
        if ts is None or ts < since:
            continue
        total += 1
        # Main source: decision reasons captured in governance row via string fields is limited,
        # so inspect stringified row for guardrail reason tags.
        blob = json.dumps(row, ensure_ascii=True, separators=(',', ':'))
        for r in TARGET_REASONS:
            if r in blob:
                counters[r] += 1

    snapshot_stats = {
        'files_considered': 0,
        'files_scanned': 0,
        'files_skipped_mtime': 0,
        'bytes_scanned_estimate': 0,
    }
    for row, stats in _json_rows_from_recent_files(
        PROJECT_ROOT / 'governance',
        'shadow*/snapshot_debug_*.jsonl',
        since=since,
        max_files=max(args.max_files, 0),
        tail_bytes=max(args.tail_bytes, 0),
        max_lines_per_file=max(args.max_lines_per_file, 0),
    ):
        snapshot_stats = stats
        ts = _parse_ts(row.get('timestamp_utc'))
        if ts is None or ts < since:
            continue
        reason = str(row.get('reason', ''))
        if reason in {'event_lock_paused', 'circuit_open_skip'}:
            counters[reason] += 1

    scan_totals = {
        key: int(scan_stats.get(key, 0) or 0) + int(snapshot_stats.get(key, 0) or 0)
        for key in scan_stats
    }

    total_trips = sum(counters.values())
    trip_rate = (total_trips / float(max(total, 1)))
    ok = trip_rate <= float(args.max_trip_rate)

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'ok': bool(ok),
        'window_hours': int(args.hours),
        'samples': int(total),
        'trip_count': int(total_trips),
        'trip_rate': round(trip_rate, 6),
        'max_trip_rate': float(args.max_trip_rate),
        'trip_breakdown': dict(counters),
        'bounded_scan': True,
        'scan_limits': {
            'max_files_per_pattern': int(args.max_files),
            'tail_bytes_per_file': int(args.tail_bytes),
            'max_lines_per_file': int(args.max_lines_per_file),
        },
        'scan_stats': scan_totals,
    }

    out = PROJECT_ROOT / 'governance' / 'health' / 'guardrail_triprate_latest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print('guardrail_triprate_ok=' + str(payload['ok']).lower() + f" rate={payload['trip_rate']}")

    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
