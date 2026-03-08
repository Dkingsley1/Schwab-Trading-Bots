import argparse
import json
import math
import os
import re
import signal
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from snapshot_health_sql import debug_snapshot_ingest_coverage, sync_raw_debug_snapshots_to_sqlite


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMELINE_STAMP_RE = re.compile(r"^project_timeline(?:_print)?_(\d{8}_\d{6})\.(?:md|pdf|html)$")


def _collect_old_files(base: Path, older_than_days: int) -> list[Path]:
    if not base.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(int(older_than_days), 0))
    out: list[Path] = []
    for root, _, files in os.walk(base):
        for name in files:
            p = Path(root) / name
            try:
                mt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
            if mt < cutoff:
                out.append(p)
    return out


def _parse_timeline_stamp(stamp: str) -> datetime | None:
    try:
        dt = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _collect_old_timeline_files(base: Path, older_than_days: int, keep_latest_runs: int) -> tuple[list[Path], int, int]:
    if not base.exists():
        return [], 0, 0

    rows: list[tuple[Path, str, datetime]] = []
    total_files = 0
    for p in base.iterdir():
        if not p.is_file():
            continue
        total_files += 1
        match = TIMELINE_STAMP_RE.match(p.name)
        if not match:
            continue
        stamp = match.group(1)
        dt = _parse_timeline_stamp(stamp)
        if dt is None:
            continue
        rows.append((p, stamp, dt))

    keep_latest_runs = max(int(keep_latest_runs), 0)
    unique_stamps = sorted({stamp: dt for _, stamp, dt in rows}.items(), key=lambda item: item[1])
    keep_stamps = {stamp for stamp, _ in unique_stamps[-keep_latest_runs:]} if keep_latest_runs > 0 else set()

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(int(older_than_days), 0))
    out: list[Path] = []
    for p, stamp, dt in rows:
        if stamp in keep_stamps:
            continue
        if dt < cutoff:
            out.append(p)

    for p in base.glob("*.local_fallback"):
        try:
            mt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mt < cutoff:
            out.append(p)

    return out, total_files, len(unique_stamps)


def _parse_snapshot_dir_ts(path: Path) -> datetime:
    name = str(path.name)
    try:
        dt = datetime.strptime(name, "%Y%m%d_%H%M%S")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return datetime.now(timezone.utc)


def _snapshot_file_count(path: Path) -> int:
    n = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            n += 1
    return n


def _collect_old_snapshot_dirs(base: Path, older_than_days: int, keep_latest: int) -> tuple[list[Path], int]:
    if not base.exists():
        return [], 0

    dirs = [
        p for p in base.iterdir()
        if p.is_dir() and p.name != "latest"
    ]
    dirs.sort(key=_parse_snapshot_dir_ts)

    total_dirs = len(dirs)
    keep_latest = max(int(keep_latest), 0)
    keep_names = {d.name for d in dirs[-keep_latest:]} if keep_latest > 0 else set()

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(int(older_than_days), 0))
    out: list[Path] = []
    for d in dirs:
        if d.name in keep_names:
            continue
        if _parse_snapshot_dir_ts(d) < cutoff:
            out.append(d)

    return out, total_dirs


def _sqlite_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 ** 3)


def _vacuum_sqlite(path: Path) -> bool:
    conn = None
    try:
        conn = sqlite3.connect(str(path))
        conn.execute("VACUUM")
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _invoke_with_timeout(timeout_seconds: float, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    timeout_seconds = float(timeout_seconds or 0.0)
    if timeout_seconds <= 0:
        try:
            return {"ok": True, "timed_out": False, "timeout_seconds": 0.0, "result": fn(*args, **kwargs)}
        except Exception as exc:
            return {"ok": False, "timed_out": False, "timeout_seconds": 0.0, "error": str(exc)}

    if not hasattr(signal, "SIGALRM"):
        try:
            return {
                "ok": True,
                "timed_out": False,
                "timeout_seconds": timeout_seconds,
                "timeout_supported": False,
                "result": fn(*args, **kwargs),
            }
        except Exception as exc:
            return {
                "ok": False,
                "timed_out": False,
                "timeout_seconds": timeout_seconds,
                "timeout_supported": False,
                "error": str(exc),
            }

    alarm_seconds = max(int(math.ceil(timeout_seconds)), 1)

    class _OperationTimeout(Exception):
        pass

    def _alarm_handler(_signum, _frame):
        raise _OperationTimeout(f"timed_out_after_{alarm_seconds}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(alarm_seconds)
    try:
        result = fn(*args, **kwargs)
        return {"ok": True, "timed_out": False, "timeout_seconds": float(alarm_seconds), "result": result}
    except _OperationTimeout as exc:
        return {"ok": False, "timed_out": True, "timeout_seconds": float(alarm_seconds), "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "timed_out": False, "timeout_seconds": float(alarm_seconds), "error": str(exc)}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune old data artifacts by retention policy.")
    parser.add_argument("--decisions-days", type=int, default=int(os.getenv("RETENTION_DECISIONS_DAYS", "30")))
    parser.add_argument("--decision-explanations-days", type=int, default=int(os.getenv("RETENTION_DECISION_EXPLANATIONS_DAYS", "30")))
    parser.add_argument("--governance-days", type=int, default=int(os.getenv("RETENTION_GOVERNANCE_DAYS", "45")))
    parser.add_argument("--exports-days", type=int, default=int(os.getenv("RETENTION_EXPORTS_DAYS", "30")))
    parser.add_argument("--backup-drills-days", type=int, default=int(os.getenv("RETENTION_BACKUP_DRILLS_DAYS", "14")))
    parser.add_argument("--csv-days", type=int, default=int(os.getenv("RETENTION_CSV_DAYS", "10")))
    parser.add_argument("--logs-days", type=int, default=int(os.getenv("RETENTION_LOGS_DAYS", "14")))
    parser.add_argument("--watchdog-events-days", type=int, default=int(os.getenv("RETENTION_WATCHDOG_EVENTS_DAYS", "30")))
    parser.add_argument("--governance-channels-days", type=int, default=int(os.getenv("RETENTION_GOVERNANCE_CHANNELS_DAYS", "7")))
    parser.add_argument("--governance-shadow-days", type=int, default=int(os.getenv("RETENTION_GOVERNANCE_SHADOW_DAYS", "7")))
    parser.add_argument("--governance-health-days", type=int, default=int(os.getenv("RETENTION_GOVERNANCE_HEALTH_DAYS", "14")))
    parser.add_argument("--project-timeline-days", type=int, default=int(os.getenv("RETENTION_PROJECT_TIMELINE_DAYS", "30")))
    parser.add_argument("--project-timeline-keep-runs", type=int, default=int(os.getenv("RETENTION_PROJECT_TIMELINE_KEEP_RUNS", "40")))
    parser.add_argument("--debug-snapshots-days", type=int, default=int(os.getenv("RETENTION_DEBUG_SNAPSHOTS_DAYS", "3")))
    parser.add_argument("--debug-snapshots-keep", type=int, default=int(os.getenv("RETENTION_DEBUG_SNAPSHOTS_KEEP", "24")))
    parser.add_argument("--prune-debug-snapshots", action=argparse.BooleanOptionalAction, default=os.getenv("RETENTION_PRUNE_DEBUG_SNAPSHOTS", "1").strip() == "1")
    parser.add_argument("--require-debug-snapshot-sync", action=argparse.BooleanOptionalAction, default=os.getenv("RETENTION_REQUIRE_DEBUG_SNAPSHOT_SYNC", "1").strip() == "1")
    parser.add_argument("--debug-snapshot-sync-timeout-seconds", type=float, default=float(os.getenv("RETENTION_DEBUG_SNAPSHOT_SYNC_TIMEOUT_SECONDS", "600")))
    parser.add_argument("--sqlite-path", default=os.getenv("SNAPSHOT_CONTEXT_SQLITE_PATH", str(PROJECT_ROOT / "data" / "jsonl_link.sqlite3")))
    parser.add_argument("--sqlite-vacuum-over-gb", type=float, default=6.0)
    parser.add_argument("--sqlite-vacuum-timeout-seconds", type=float, default=float(os.getenv("RETENTION_SQLITE_VACUUM_TIMEOUT_SECONDS", "900")))
    parser.add_argument("--skip-sqlite-vacuum", action=argparse.BooleanOptionalAction, default=os.getenv("RETENTION_SKIP_SQLITE_VACUUM", "0").strip() == "1")
    parser.add_argument("--json", action="store_true", help="Accepted for orchestrator compatibility; JSON is always printed.")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    targets = {
        "decisions": (PROJECT_ROOT / "decisions", args.decisions_days),
        "decision_explanations": (PROJECT_ROOT / "decision_explanations", args.decision_explanations_days),
        "governance_watchdog": (PROJECT_ROOT / "governance" / "watchdog", args.governance_days),
        "governance_events": (PROJECT_ROOT / "governance" / "events", args.watchdog_events_days),
        "governance_channels": (PROJECT_ROOT / "governance" / "channels", args.governance_channels_days),
        "governance_health": (PROJECT_ROOT / "governance" / "health", args.governance_health_days),
        "exports_sql_reports": (PROJECT_ROOT / "exports" / "sql_reports", args.exports_days),
        "exports_csv": (PROJECT_ROOT / "exports" / "csv", args.csv_days),
        "backup_drills": (PROJECT_ROOT / "exports" / "backup_drills", args.backup_drills_days),
        "logs": (PROJECT_ROOT / "logs", args.logs_days),
    }

    governance_root = PROJECT_ROOT / "governance"
    if governance_root.exists():
        for child in sorted(governance_root.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith("shadow"):
                continue
            targets[f"governance_{name}"] = (child, args.governance_shadow_days)

    to_delete: list[Path] = []
    summary: dict[str, dict[str, int]] = {}
    for label, (base, days) in targets.items():
        rows = _collect_old_files(base, days)
        to_delete.extend(rows)
        summary[label] = {"candidates": len(rows), "older_than_days": int(days)}

    timeline_dir = PROJECT_ROOT / "exports" / "reports" / "project_timeline"
    timeline_rows, timeline_total_files, timeline_total_runs = _collect_old_timeline_files(
        timeline_dir,
        older_than_days=args.project_timeline_days,
        keep_latest_runs=args.project_timeline_keep_runs,
    )
    to_delete.extend(timeline_rows)
    summary["exports_project_timeline"] = {
        "candidates": len(timeline_rows),
        "older_than_days": int(args.project_timeline_days),
        "keep_latest_runs": int(args.project_timeline_keep_runs),
        "total_files": int(timeline_total_files),
        "total_runs": int(timeline_total_runs),
    }

    file_deleted = 0
    if args.apply:
        for p in to_delete:
            try:
                p.unlink()
                file_deleted += 1
            except OSError:
                pass

    sqlite_path = Path(args.sqlite_path)

    debug_root = PROJECT_ROOT / "exports" / "debug_snapshots"
    debug_candidates, debug_total_dirs = _collect_old_snapshot_dirs(
        debug_root,
        older_than_days=args.debug_snapshots_days,
        keep_latest=args.debug_snapshots_keep,
    )

    debug_sync_meta: dict[str, object] = {}
    debug_cov_meta: dict[str, object] = {}
    debug_dirs_deleted = 0
    debug_files_deleted = 0
    debug_dirs_skipped_not_ready = 0

    if args.apply and args.prune_debug_snapshots and debug_candidates:
        if args.require_debug_snapshot_sync:
            sync_result = _invoke_with_timeout(
                args.debug_snapshot_sync_timeout_seconds,
                sync_raw_debug_snapshots_to_sqlite,
                project_root=PROJECT_ROOT,
                sqlite_path=sqlite_path,
                snapshot_dirs=debug_candidates,
            )
            if bool(sync_result.get("ok", False)):
                value = sync_result.get("result")
                if isinstance(value, dict):
                    debug_sync_meta = value
                else:
                    debug_sync_meta = {"result": value}
            else:
                debug_sync_meta = {
                    "error": str(sync_result.get("error", "unknown_sync_error")),
                    "timed_out": bool(sync_result.get("timed_out", False)),
                    "timeout_seconds": float(sync_result.get("timeout_seconds", 0.0)),
                }
                if "timeout_supported" in sync_result:
                    debug_sync_meta["timeout_supported"] = bool(sync_result.get("timeout_supported"))

        ready_ids: set[str] = set()
        try:
            cov = debug_snapshot_ingest_coverage(
                project_root=PROJECT_ROOT,
                sqlite_path=sqlite_path,
                snapshot_dirs=debug_candidates,
            )
            rows = cov.get("rows") if isinstance(cov.get("rows"), list) else []
            for row in rows:
                if isinstance(row, dict) and bool(row.get("ready", False)):
                    ready_ids.add(str(row.get("snapshot_id", "")))
            debug_cov_meta = {
                "snapshot_total": int(cov.get("snapshot_total", 0) or 0),
                "ready_total": int(cov.get("ready_total", 0) or 0),
                "coverage_ratio": float(cov.get("coverage_ratio", 0.0) or 0.0),
            }
        except Exception as exc:
            debug_cov_meta = {"error": str(exc)}

        for d in debug_candidates:
            if args.require_debug_snapshot_sync and d.name not in ready_ids:
                debug_dirs_skipped_not_ready += 1
                continue
            try:
                file_count = _snapshot_file_count(d)
                shutil.rmtree(d)
                debug_dirs_deleted += 1
                debug_files_deleted += int(file_count)
            except OSError:
                continue

    size_before = _sqlite_size_gb(sqlite_path)
    vacuum_ran = False
    vacuum_ok = None
    vacuum_skipped = bool(args.skip_sqlite_vacuum)
    vacuum_meta: dict[str, object] = {}
    if args.apply and (not args.skip_sqlite_vacuum) and sqlite_path.exists() and size_before >= args.sqlite_vacuum_over_gb:
        vacuum_ran = True
        vacuum_result = _invoke_with_timeout(
            args.sqlite_vacuum_timeout_seconds,
            _vacuum_sqlite,
            sqlite_path,
        )
        if bool(vacuum_result.get("ok", False)):
            vacuum_ok = bool(vacuum_result.get("result", False))
        else:
            vacuum_ok = False
        vacuum_meta = {
            "timed_out": bool(vacuum_result.get("timed_out", False)),
            "timeout_seconds": float(vacuum_result.get("timeout_seconds", 0.0)),
        }
        if "timeout_supported" in vacuum_result:
            vacuum_meta["timeout_supported"] = bool(vacuum_result.get("timeout_supported"))
        if vacuum_result.get("error") is not None:
            vacuum_meta["error"] = str(vacuum_result.get("error"))
    size_after = _sqlite_size_gb(sqlite_path)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "apply": bool(args.apply),
        "targets": summary,
        "total_candidates": int(len(to_delete) + len(debug_candidates)),
        "deleted": int(file_deleted + debug_files_deleted),
        "deleted_files": int(file_deleted),
        "debug_snapshots": {
            "enabled": bool(args.prune_debug_snapshots),
            "require_sync": bool(args.require_debug_snapshot_sync),
            "retention_days": int(args.debug_snapshots_days),
            "keep_latest": int(args.debug_snapshots_keep),
            "sync_timeout_seconds": float(args.debug_snapshot_sync_timeout_seconds),
            "total_dirs": int(debug_total_dirs),
            "candidate_dirs": int(len(debug_candidates)),
            "deleted_dirs": int(debug_dirs_deleted),
            "deleted_files": int(debug_files_deleted),
            "skipped_not_ready": int(debug_dirs_skipped_not_ready),
            "sql_sync": debug_sync_meta,
            "ingest_coverage": debug_cov_meta,
        },
        "sqlite": {
            "path": str(sqlite_path),
            "size_gb_before": round(size_before, 3),
            "size_gb_after": round(size_after, 3),
            "vacuum_threshold_gb": args.sqlite_vacuum_over_gb,
            "vacuum_timeout_seconds": float(args.sqlite_vacuum_timeout_seconds),
            "vacuum_skipped": bool(vacuum_skipped),
            "vacuum_ran": vacuum_ran,
            "vacuum_ok": vacuum_ok,
            "vacuum_meta": vacuum_meta,
        },
    }

    out = PROJECT_ROOT / "governance" / "health" / "data_retention_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
