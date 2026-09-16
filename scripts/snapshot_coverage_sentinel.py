import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "snapshot_coverage_latest.json"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.storage_router import inspect_storage_path
from scripts.ops.long_runtime_common import eastern_off_hours_window, write_payload

MASTER_CONTROL_DAY_RE = re.compile(r"master_control_(\d{8})\.jsonl$")
REVERSE_SCAN_BLOCK_BYTES = 256 * 1024
RUNTIME_TRAINING_SNAPSHOT_REL = Path(
    "exports/training/runtime_training_snapshot_latest.jsonl"
)
COLLECTION_STARTUP_GRACE_SECONDS = 15 * 60
MAX_LINE_BYTES = 2 * 1024 * 1024


@dataclass
class ScanBudget:
    max_bytes: int = 64 * 1024 * 1024
    max_file_bytes: int = 32 * 1024 * 1024
    max_seconds: float = 30.0
    bytes_read: int = 0
    reasons: set[str] = field(default_factory=set)
    started: float = field(default_factory=time.monotonic)

    def admitted(self) -> bool:
        if time.monotonic() - self.started >= self.max_seconds:
            self.reasons.add("scan_deadline")
            return False
        if self.bytes_read >= self.max_bytes:
            self.reasons.add("scan_byte_limit")
            return False
        return True


def _readable(path: Path) -> bool:
    return inspect_storage_path(path).get("status") == "present"


def _parse_ts(raw: Any):
    if not raw:
        return None
    s = str(raw).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(s)
        return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
    except Exception:
        return None


def _latest_heartbeat_symbols_total(project_root: Path = PROJECT_ROOT) -> int:
    hb_dir = project_root / "governance" / "health"
    rows: list[tuple[datetime, int]] = []
    for p in hb_dir.glob("shadow_loop_*.json"):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
            ts = _parse_ts(row.get("timestamp_utc"))
            if ts is None:
                continue
            rows.append((ts, int(row.get("symbols_total", 0) or 0)))
        except Exception:
            continue
    if not rows:
        return 1
    best_ts = max(ts for ts, _total in rows)
    freshness_floor = best_ts - timedelta(minutes=5)
    return max([total for ts, total in rows if ts >= freshness_floor] or [1])


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
    governance = project_root / "governance"
    if not _readable(governance):
        return []
    files = [
        path
        for directory in governance.glob("shadow*")
        if ".__external_symlink_backup_" not in directory.name
        if _readable(directory) and directory.is_dir()
        for path in directory.glob("master_control_*.jsonl")
        if _readable(path)
        if _file_overlaps_window(path, since)
    ]
    files.sort(key=lambda path: (path.parent.name, path.name))
    return files


def _runtime_training_snapshot_file(project_root: Path) -> Path:
    health_snapshot = (
        project_root / "governance" / "health" / "runtime_training_snapshot_latest.json"
    )
    try:
        payload = json.loads(health_snapshot.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    rows_path = (
        str(payload.get("rows_path") or "").strip() if isinstance(payload, dict) else ""
    )
    if rows_path:
        candidate = Path(rows_path).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return candidate
    return project_root / RUNTIME_TRAINING_SNAPSHOT_REL


def _runtime_snapshot_window(
    project_root: Path, *, hours: int, now: datetime
) -> dict[str, Any]:
    health_path = (
        project_root / "governance" / "health" / "runtime_training_snapshot_latest.json"
    )
    try:
        payload = json.loads(health_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    coverage = (
        payload.get("coverage") if isinstance(payload.get("coverage"), dict) else {}
    )
    windows = (
        coverage.get("recent_windows")
        if isinstance(coverage.get("recent_windows"), dict)
        else {}
    )
    window = windows.get(str(max(int(hours), 1)))
    if not isinstance(window, dict):
        return {}
    ended = _parse_ts(window.get("window_ended_utc"))
    if ended is None:
        return {}
    age_minutes = (now - ended).total_seconds() / 60.0
    max_age_minutes = max(min(int(hours) * 15, 30), 10)
    produced = _parse_ts(payload.get("timestamp_utc"))
    counts = [
        window.get(key)
        for key in ("row_count", "rows_with_snapshot_id", "unique_snapshot_ids")
    ]
    if (
        not 0 <= age_minutes <= max_age_minutes
        or produced is None
        or not 0 <= (now - produced).total_seconds() <= max_age_minutes * 60
        or any(
            type(count) is not int or not 0 <= count <= 100_000_000 for count in counts
        )
        or not counts[2] <= counts[1] <= counts[0]
        or window.get("window_hours") != max(int(hours), 1)
    ):
        return {}
    return {
        **window,
        "window_age_minutes": round(age_minutes, 4),
        "max_window_age_minutes": max_age_minutes,
        "health_path": str(health_path),
        "rows_path": str(payload.get("rows_path") or ""),
    }


def _runtime_snapshot_schema_version(project_root: Path) -> int:
    path = (
        project_root / "governance" / "health" / "runtime_training_snapshot_latest.json"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return int(payload.get("schema_version", 1) or 1)
    except Exception:
        return 1


def _healthy_collection_startup_grace(
    project_root: Path,
    *,
    now: datetime,
    max_elapsed_seconds: int = COLLECTION_STARTUP_GRACE_SECONDS,
) -> dict[str, Any]:
    path = project_root / "governance" / "health" / "process_watchdog_latest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "active": False,
            "reason": "watchdog_evidence_missing",
            "path": str(path),
        }
    timestamp = _parse_ts(payload.get("timestamp_utc"))
    age_seconds = (
        (now - timestamp).total_seconds() if timestamp is not None else float("inf")
    )
    statuses = payload.get("status") if isinstance(payload.get("status"), list) else []
    all_sleeves = next(
        (
            row
            for row in statuses
            if isinstance(row, dict) and str(row.get("name") or "") == "all_sleeves"
        ),
        {},
    )
    try:
        elapsed = float(all_sleeves.get("process_elapsed_seconds", float("inf")))
    except (TypeError, ValueError):
        elapsed = float("inf")
    child_fanout = (
        all_sleeves.get("child_fanout")
        if isinstance(all_sleeves.get("child_fanout"), dict)
        else {}
    )
    restart_storms = (
        payload.get("restart_storms")
        if isinstance(payload.get("restart_storms"), list)
        else []
    )
    watchdog_ready = str(payload.get("overall_status") or "").strip().lower() == "ready"
    process_ready = bool(
        all_sleeves.get("process_live", False)
        and all_sleeves.get("heartbeat_ok", False)
        and (all_sleeves.get("child_fanout_ok", False) or child_fanout.get("ok", False))
    )
    active = bool(
        0 <= age_seconds <= 300.0
        and watchdog_ready
        and process_ready
        and not restart_storms
        and 0 <= elapsed <= max(float(max_elapsed_seconds), 0.0)
    )
    reason = (
        "healthy_collection_fanout_warming"
        if active
        else "startup_grace_requirements_not_met"
    )
    return {
        "active": active,
        "reason": reason,
        "path": str(path),
        "watchdog_age_seconds": (
            round(age_seconds, 3) if age_seconds != float("inf") else None
        ),
        "watchdog_ready": watchdog_ready,
        "process_ready": process_ready,
        "process_elapsed_seconds": (
            round(elapsed, 3) if elapsed != float("inf") else None
        ),
        "max_elapsed_seconds": int(max_elapsed_seconds),
        "restart_storm_count": len(restart_storms),
        "policy": "bounded startup grace requires fresh healthy fanout and never counts as strict coverage evidence",
    }


def _iter_reverse_rows(
    path: Path, *, budget: ScanBudget, block_bytes: int = REVERSE_SCAN_BLOCK_BYTES
):
    if not _readable(path):
        budget.reasons.add("source_unavailable_or_protected")
        return
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            if position <= 0:
                return
            pending = b""
            file_bytes = 0
            while position > 0 and budget.admitted():
                size = min(
                    max(int(block_bytes), 1024),
                    position,
                    budget.max_bytes - budget.bytes_read,
                    budget.max_file_bytes - file_bytes,
                )
                if size <= 0:
                    budget.reasons.add("per_file_byte_limit")
                    return
                position -= size
                handle.seek(position)
                block = handle.read(size)
                if not block:
                    break
                budget.bytes_read += len(block)
                file_bytes += len(block)
                pending = block + pending
                lines = pending.splitlines()
                if position > 0:
                    pending = lines[0] if lines else pending
                    complete_lines = lines[1:]
                else:
                    pending = b""
                    complete_lines = lines
                if len(pending) > MAX_LINE_BYTES:
                    budget.reasons.add("oversized_line")
                    return
                for raw_line in reversed(complete_lines):
                    if time.monotonic() - budget.started >= budget.max_seconds:
                        budget.reasons.add("scan_deadline")
                        return
                    if len(raw_line) > MAX_LINE_BYTES:
                        budget.reasons.add("oversized_line")
                        continue
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        yield row
    except OSError:
        budget.reasons.add("source_read_error")
        return


def _iter_recent_jsonl_rows(
    path: Path,
    since: datetime,
    *,
    block_bytes: int = REVERSE_SCAN_BLOCK_BYTES,
    budget: ScanBudget | None = None,
    now: datetime | None = None,
):
    scan = budget if budget is not None else ScanBudget()
    until = now if now is not None else datetime.now(timezone.utc)
    # Rows may be grouped by sequence, so old timestamps do not imply EOF.
    for row in _iter_reverse_rows(path, budget=scan, block_bytes=block_bytes):
        ts = _parse_ts(row.get("timestamp_utc"))
        if ts is None or ts < since or ts > until:
            continue
        yield row


def _iter_jsonl_tail_rows(
    path: Path, *, max_rows: int = 10000, budget: ScanBudget | None = None
):
    """Read only a byte/time-bounded reverse tail; historical rows carry no freshness credit."""
    scan = budget if budget is not None else ScanBudget()
    for count, row in enumerate(_iter_reverse_rows(path, budget=scan), 1):
        yield row
        if count >= max(int(max_rows), 1):
            return


def build_payload(
    *,
    hours: int,
    min_coverage_ratio: float,
    project_root: Path = PROJECT_ROOT,
    now: datetime | None = None,
    scan_budget: ScanBudget | None = None,
) -> dict[str, Any]:
    current_time = now if now is not None else datetime.now(timezone.utc)
    since = current_time - timedelta(hours=max(int(hours), 1))
    expected_symbols = _latest_heartbeat_symbols_total(project_root)

    total_rows = 0
    snapshot_rows = 0
    unique_snapshot_ids: set[str] = set()
    indexed_unique_count = 0
    budget = scan_budget if scan_budget is not None else ScanBudget()
    candidate_files = _candidate_master_control_files(project_root, since)
    fallback_sources: list[str] = []
    primary_sources: list[str] = []
    required_unique_snapshots = max(
        1, int((expected_symbols * float(min_coverage_ratio)) + 0.999999)
    )
    stopped_after_reaching_floor = False

    def _coverage_floor_reached() -> bool:
        return (
            total_rows > 0
            and indexed_unique_count + len(unique_snapshot_ids)
            >= required_unique_snapshots
        )

    runtime_snapshot = _runtime_training_snapshot_file(project_root)
    scan_paths: list[Path] = []
    indexed_window = _runtime_snapshot_window(
        project_root, hours=hours, now=current_time
    )
    if indexed_window:
        total_rows = int(indexed_window.get("row_count", 0) or 0)
        snapshot_rows = int(indexed_window.get("rows_with_snapshot_id", 0) or 0)
        indexed_unique_count = indexed_window["unique_snapshot_ids"]
        primary_sources.extend(
            [
                str(indexed_window.get("health_path") or ""),
                str(indexed_window.get("rows_path") or ""),
            ]
        )
        primary_sources = [source for source in primary_sources if source]
    elif (
        _runtime_snapshot_schema_version(project_root) < 2
        and _readable(runtime_snapshot)
        and _file_overlaps_window(runtime_snapshot, since)
    ):
        scan_paths = [runtime_snapshot]
        primary_sources.append(str(runtime_snapshot))
    else:
        scan_paths = list(candidate_files)

    for path in scan_paths:
        if not budget.admitted():
            break
        for row in _iter_recent_jsonl_rows(
            path, since, budget=budget, now=current_time
        ):
            total_rows += 1
            snapshot_id = row.get("snapshot_id")
            if isinstance(snapshot_id, str) and 0 < len(snapshot_id) <= 512:
                snapshot_rows += 1
                unique_snapshot_ids.add(str(snapshot_id))
            if _coverage_floor_reached():
                stopped_after_reaching_floor = True
                break
        if stopped_after_reaching_floor:
            break

    historical_rows = 0
    if total_rows <= 0:
        if _readable(runtime_snapshot) and budget.admitted():
            fallback_sources.append(str(runtime_snapshot))
            for row in _iter_jsonl_tail_rows(runtime_snapshot, budget=budget):
                historical_rows += 1

    unique_count = indexed_unique_count + len(unique_snapshot_ids)
    expected_floor = max(expected_symbols, 1)
    coverage_ratio = unique_count / float(expected_floor)
    evidence_ready = (total_rows > 0) and (coverage_ratio >= float(min_coverage_ratio))
    market_window = eastern_off_hours_window(now=current_time)
    startup_grace = _healthy_collection_startup_grace(project_root, now=current_time)
    startup_warming = bool(startup_grace.get("active", False) and total_rows > 0)
    operational_ok = bool(
        evidence_ready
        or (market_window.get("active", False) and total_rows > 0)
        or startup_warming
    )
    return {
        "timestamp_utc": current_time.isoformat(),
        "ok": bool(evidence_ready),
        "operational_ok": operational_ok,
        "overall_status": (
            "ready"
            if evidence_ready
            else (
                "warming_after_restart"
                if startup_warming
                else "collecting_off_hours" if operational_ok else "degraded"
            )
        ),
        "evidence_ready": bool(evidence_ready),
        "market_window": market_window,
        "startup_grace": startup_grace,
        "window_hours": int(hours),
        "expected_symbols_floor": expected_floor,
        "files_considered": len(candidate_files),
        "primary_sources": primary_sources,
        "primary_source_count": len(primary_sources),
        "indexed_snapshot_window": indexed_window,
        "fallback_sources": fallback_sources,
        "fallback_source_count": len(fallback_sources),
        "historical_tail_rows_diagnostic_only": historical_rows,
        "scan_budget": {
            "bytes_read": budget.bytes_read,
            "max_bytes": budget.max_bytes,
            "max_seconds": budget.max_seconds,
            "incomplete": bool(budget.reasons),
            "reasons": sorted(budget.reasons),
        },
        "counts_are_lower_bounds": not bool(indexed_window),
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
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        hours=int(args.hours),
        min_coverage_ratio=float(args.min_coverage_ratio),
    )

    out = Path(args.out_file).expanduser()
    write_payload(out, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "snapshot_coverage_ok="
            + str(payload["ok"]).lower()
            + f" ratio={payload['coverage_ratio']}"
        )

    return 0 if payload["operational_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
