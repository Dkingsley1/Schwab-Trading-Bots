import argparse
import json
import re
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_CONTROL_DAY_RE = re.compile(r"master_control_(\d{8})\.jsonl$")
REVERSE_SCAN_BLOCK_BYTES = 256 * 1024
RUNTIME_TRAINING_SNAPSHOT_REL = Path("exports/training/runtime_training_snapshot_latest.jsonl")
STALE_TAIL_ROW_LIMIT = 50


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
        if ".__external_symlink_backup_" not in path.parent.name
        if _file_overlaps_window(path, since)
    ]
    files.sort(key=lambda path: (path.parent.name, path.name))
    return files


def _runtime_training_snapshot_file(project_root: Path) -> Path:
    health_snapshot = project_root / "governance" / "health" / "runtime_training_snapshot_latest.json"
    try:
        payload = json.loads(health_snapshot.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    rows_path = str(payload.get("rows_path") or "").strip() if isinstance(payload, dict) else ""
    if rows_path:
        candidate = Path(rows_path).expanduser()
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        return candidate
    return project_root / RUNTIME_TRAINING_SNAPSHOT_REL


def _iter_recent_jsonl_rows(path: Path, since: datetime, *, block_bytes: int = REVERSE_SCAN_BLOCK_BYTES):
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            if position <= 0:
                return
            pending = b""
            seen_recent = False
            stale_tail_rows = 0
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
                        stale_tail_rows += 1
                        if stale_tail_rows >= STALE_TAIL_ROW_LIMIT:
                            return
                        continue
                    stale_tail_rows = 0
                    seen_recent = True
                    yield row
    except FileNotFoundError:
        return


def _iter_jsonl_tail_rows(path: Path, *, max_rows: int = 10000):
    """Yield a bounded tail of JSONL rows without applying timestamp freshness."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = deque(handle, maxlen=max(int(max_rows), 1))
    except FileNotFoundError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            yield row


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
    fallback_sources: list[str] = []
    primary_sources: list[str] = []
    required_unique_snapshots = max(1, int((expected_symbols * float(min_coverage_ratio)) + 0.999999))
    stopped_after_reaching_floor = False

    def _coverage_floor_reached() -> bool:
        return total_rows > 0 and len(unique_snapshot_ids) >= required_unique_snapshots

    runtime_snapshot = _runtime_training_snapshot_file(project_root)
    scan_paths: list[Path] = []
    if runtime_snapshot.exists() and _file_overlaps_window(runtime_snapshot, since):
        scan_paths = [runtime_snapshot]
        primary_sources.append(str(runtime_snapshot))
    else:
        scan_paths = list(candidate_files)

    for path in scan_paths:
        for row in _iter_recent_jsonl_rows(path, since):
            total_rows += 1
            snapshot_id = row.get("snapshot_id")
            if snapshot_id:
                snapshot_rows += 1
                unique_snapshot_ids.add(str(snapshot_id))
            if _coverage_floor_reached():
                stopped_after_reaching_floor = True
                break
        if stopped_after_reaching_floor:
            break

    if total_rows <= 0:
        if runtime_snapshot.exists():
            fallback_sources.append(str(runtime_snapshot))
            for row in _iter_jsonl_tail_rows(runtime_snapshot):
                total_rows += 1
                snapshot_id = row.get("snapshot_id")
                if snapshot_id:
                    snapshot_rows += 1
                    unique_snapshot_ids.add(str(snapshot_id))
                if _coverage_floor_reached():
                    stopped_after_reaching_floor = True
                    break

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
        "primary_sources": primary_sources,
        "primary_source_count": len(primary_sources),
        "fallback_sources": fallback_sources,
        "fallback_source_count": len(fallback_sources),
        "rows_scanned": total_rows,
        "rows_with_snapshot_id": snapshot_rows,
        "unique_snapshot_ids": unique_count,
        "required_unique_snapshot_floor": required_unique_snapshots,
        "stopped_after_reaching_floor": bool(stopped_after_reaching_floor),
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
