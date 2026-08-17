#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts import ops_data_plane
    from scripts.ops.long_runtime_common import write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from scripts import ops_data_plane
    from .long_runtime_common import write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "ops_data_plane_compaction_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "ops_data_plane_compaction.lock"
PROTECTED_VOLUME_ROOTS = (Path("/Volumes/VIDEO"),)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_under(path: Path, root: Path) -> bool:
    raw = str(path.expanduser())
    base = str(root.expanduser())
    return bool(raw == base or raw.startswith(f"{base}/"))


def _protected_storage_path(path: Path) -> bool:
    return any(_path_under(path, root) for root in PROTECTED_VOLUME_ROOTS)


def _default_archive_root(project_root: Path) -> Path:
    configured = str(os.getenv("BOT_OPS_DATA_PLANE_ARCHIVE_ROOT", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    active_root = str(
        os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "")
        or os.getenv("BOT_LOGS_ACTIVE_ROOT", "")
    ).strip()
    if active_root:
        return Path(active_root).expanduser() / "cold_archive"
    return project_root / "governance" / "archive"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (str(table),),
    ).fetchone()
    return row is not None


def _database_metrics(conn: sqlite3.Connection, db_path: Path) -> dict[str, Any]:
    page_size = int(conn.execute("PRAGMA page_size").fetchone()[0] or 0)
    page_count = int(conn.execute("PRAGMA page_count").fetchone()[0] or 0)
    freelist_count = int(conn.execute("PRAGMA freelist_count").fetchone()[0] or 0)
    return {
        "path": str(db_path),
        "size_bytes": int(db_path.stat().st_size) if db_path.exists() else 0,
        "page_size": page_size,
        "page_count": page_count,
        "freelist_count": freelist_count,
        "reclaimable_bytes": int(page_size * freelist_count),
    }


def _legacy_snapshot(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _table_exists(conn, "schema_drift_events"):
        return {"row_count": 0, "max_id": 0, "first_recorded_utc": "", "last_recorded_utc": ""}
    row = conn.execute(
        """
        SELECT COUNT(*), COALESCE(MAX(id), 0), COALESCE(MIN(recorded_utc), ''),
               COALESCE(MAX(recorded_utc), '')
        FROM schema_drift_events
        """
    ).fetchone()
    return {
        "row_count": int(row[0] or 0),
        "max_id": int(row[1] or 0),
        "first_recorded_utc": str(row[2] or ""),
        "last_recorded_utc": str(row[3] or ""),
    }


def _legacy_rollups(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rollups: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    cursor = conn.execute(
        """
        SELECT lane, source_rel, line_no, observed_schema_version,
               expected_schema_version, drift_kind, payload_sha256,
               run_id, iter_id, recorded_utc, metadata_json
        FROM schema_drift_events
        ORDER BY id
        """
    )
    for row in cursor:
        lane = str(row[0] or "")
        source_rel = str(row[1] or "")
        observed = int(row[3] or 0)
        expected = int(row[4] or 0)
        drift_kind = str(row[5] or "")
        key = (lane, source_rel, observed, expected, drift_kind)
        existing = rollups.get(key)
        if existing is None:
            existing = {
                "lane": lane,
                "source_rel": source_rel,
                "observed_schema_version": observed,
                "expected_schema_version": expected,
                "drift_kind": drift_kind,
                "occurrence_count": 0,
                "first_line_no": int(row[2] or 0),
                "last_line_no": int(row[2] or 0),
                "first_recorded_utc": str(row[9] or ""),
                "last_recorded_utc": str(row[9] or ""),
                "first_payload_sha256": str(row[6] or ""),
                "last_payload_sha256": str(row[6] or ""),
                "latest_run_id": str(row[7] or ""),
                "latest_iter_id": str(row[8] or ""),
                "metadata_json": str(row[10] or "{}"),
            }
            rollups[key] = existing
        existing["occurrence_count"] = int(existing["occurrence_count"]) + 1
        existing["last_line_no"] = int(row[2] or 0)
        existing["last_recorded_utc"] = str(row[9] or "")
        existing["last_payload_sha256"] = str(row[6] or "")
        existing["latest_run_id"] = str(row[7] or "")
        existing["latest_iter_id"] = str(row[8] or "")
        existing["metadata_json"] = str(row[10] or "{}")
    return [rollups[key] for key in sorted(rollups)]


def _archive_rollups(
    archive_root: Path,
    *,
    db_path: Path,
    snapshot: dict[str, Any],
    rollups: list[dict[str, Any]],
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_dir = archive_root / "ops_data_plane" / "schema_drift_rollups"
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / f"schema_drift_rollups_{timestamp}.jsonl.gz"
    temp_path = archive_path.with_name(f".{archive_path.name}.tmp.{os.getpid()}")
    with gzip.open(temp_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rollups:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temp_path, archive_path)
    archive_sha256 = _sha256_file(archive_path)
    with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
        verified_rollup_count = sum(1 for line in handle if line.strip())
    if verified_rollup_count != len(rollups):
        raise RuntimeError(
            f"schema drift archive verification failed: expected {len(rollups)} rows, got {verified_rollup_count}"
        )
    manifest_path = archive_path.with_suffix(".manifest.json")
    manifest = {
        "timestamp_utc": _now_utc(),
        "schema_version": 1,
        "archive_format": "jsonl_gzip",
        "source_database": str(db_path),
        "source_table": "schema_drift_events",
        "source_event_count": int(snapshot.get("row_count", 0) or 0),
        "source_max_id": int(snapshot.get("max_id", 0) or 0),
        "first_recorded_utc": str(snapshot.get("first_recorded_utc") or ""),
        "last_recorded_utc": str(snapshot.get("last_recorded_utc") or ""),
        "rollup_count": len(rollups),
        "archive_path": str(archive_path),
        "archive_size_bytes": int(archive_path.stat().st_size),
        "archive_sha256": archive_sha256,
        "verified_rollup_count": verified_rollup_count,
        "verified": True,
        "detail_authority": "source_rel JSONL; the hot SQLite rows are derivative ingestion audit evidence",
        "payload_policy": "full payloads remain in source JSONL; rollups preserve counts, coverage, and payload hashes",
    }
    write_payload(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def _merge_rollups(conn: sqlite3.Connection, rollups: list[dict[str, Any]]) -> None:
    conn.executemany(
        """
        INSERT INTO schema_drift_rollups(
            lane, source_rel, observed_schema_version, expected_schema_version, drift_kind,
            occurrence_count, first_line_no, last_line_no, first_recorded_utc,
            last_recorded_utc, first_payload_sha256, last_payload_sha256,
            sample_payload_json, latest_run_id, latest_iter_id, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?)
        ON CONFLICT(
            lane, source_rel, observed_schema_version, expected_schema_version, drift_kind
        ) DO UPDATE SET
            occurrence_count = schema_drift_rollups.occurrence_count + excluded.occurrence_count,
            first_line_no = MIN(schema_drift_rollups.first_line_no, excluded.first_line_no),
            last_line_no = excluded.last_line_no,
            first_recorded_utc = MIN(schema_drift_rollups.first_recorded_utc, excluded.first_recorded_utc),
            last_recorded_utc = MAX(schema_drift_rollups.last_recorded_utc, excluded.last_recorded_utc),
            last_payload_sha256 = excluded.last_payload_sha256,
            latest_run_id = excluded.latest_run_id,
            latest_iter_id = excluded.latest_iter_id,
            metadata_json = excluded.metadata_json
        """,
        [
            (
                row["lane"],
                row["source_rel"],
                row["observed_schema_version"],
                row["expected_schema_version"],
                row["drift_kind"],
                row["occurrence_count"],
                row["first_line_no"],
                row["last_line_no"],
                row["first_recorded_utc"],
                row["last_recorded_utc"],
                row["first_payload_sha256"],
                row["last_payload_sha256"],
                row["latest_run_id"],
                row["latest_iter_id"],
                ops_data_plane._bounded_utf8_text(row["metadata_json"], 2048),
            )
            for row in rollups
        ],
    )


def compact_ops_data_plane(
    project_root: Path = PROJECT_ROOT,
    *,
    db_path: Path | None = None,
    archive_root: Path | None = None,
    apply: bool = False,
    require_stack_stopped: bool = True,
    vacuum: bool = True,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    db_path = (db_path or ops_data_plane.resolve_db_path(project_root)).expanduser()
    archive_root = (archive_root or _default_archive_root(project_root)).expanduser()
    stopped_flag = project_root / "governance" / "health" / "STACK_STOPPED.flag"
    blockers: list[str] = []
    if apply and require_stack_stopped and not stopped_flag.exists():
        blockers.append("runtime_must_be_quiesced_before_ops_data_plane_compaction")
    if not db_path.exists():
        blockers.append("ops_data_plane_database_missing")
    if _protected_storage_path(archive_root):
        blockers.append("protected_archive_volume_rejected")
    if blockers:
        return {
            "timestamp_utc": _now_utc(),
            "schema_version": 1,
            "ok": False,
            "overall_status": "blocked",
            "grade": "F",
            "apply": bool(apply),
            "blockers": blockers,
            "warnings": [],
            "database_path": str(db_path),
            "archive_root": str(archive_root),
        }

    with sqlite3.connect(str(db_path), timeout=60.0) as conn:
        conn.execute("PRAGMA busy_timeout=60000")
        ops_data_plane.ensure_schema(conn)
        quick_check_before = str(conn.execute("PRAGMA quick_check(1)").fetchone()[0] or "")
        before = _database_metrics(conn, db_path)
        snapshot = _legacy_snapshot(conn)
        if quick_check_before.lower() != "ok":
            return {
                "timestamp_utc": _now_utc(),
                "schema_version": 1,
                "ok": False,
                "overall_status": "blocked",
                "grade": "F",
                "apply": bool(apply),
                "blockers": ["ops_data_plane_quick_check_failed"],
                "warnings": [],
                "database_before": before,
                "legacy_snapshot": snapshot,
                "archive_root": str(archive_root),
                "integrity": {"before": quick_check_before},
            }
        rollups = _legacy_rollups(conn) if apply and int(snapshot["row_count"]) > 0 else []

        if not apply or int(snapshot["row_count"]) == 0:
            return {
                "timestamp_utc": _now_utc(),
                "schema_version": 1,
                "ok": quick_check_before.lower() == "ok",
                "overall_status": "planned" if not apply and int(snapshot["row_count"]) > 0 else "ready",
                "grade": "A+" if quick_check_before.lower() == "ok" else "F",
                "apply": bool(apply),
                "blockers": [] if quick_check_before.lower() == "ok" else ["ops_data_plane_quick_check_failed"],
                "warnings": [],
                "database_before": before,
                "legacy_snapshot": snapshot,
                "planned_rollup_count": len(rollups) if apply else None,
                "archive_root": str(archive_root),
                "vacuum_requested": bool(vacuum),
            }

        archive = _archive_rollups(archive_root, db_path=db_path, snapshot=snapshot, rollups=rollups)
        conn.execute("BEGIN IMMEDIATE")
        current = _legacy_snapshot(conn)
        if current["row_count"] != snapshot["row_count"] or current["max_id"] != snapshot["max_id"]:
            conn.rollback()
            return {
                "timestamp_utc": _now_utc(),
                "schema_version": 1,
                "ok": False,
                "overall_status": "blocked",
                "grade": "F",
                "apply": True,
                "blockers": ["schema_drift_events_changed_during_compaction"],
                "warnings": [],
                "legacy_snapshot": snapshot,
                "legacy_recheck": current,
                "archive": archive,
            }
        _merge_rollups(conn, rollups)
        conn.execute("DROP TABLE schema_drift_events")
        conn.commit()
        ops_data_plane.ensure_schema(conn)
        vacuum_error = ""
        if vacuum:
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
                conn.execute("VACUUM")
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
            except sqlite3.Error as exc:
                vacuum_error = str(exc)
        quick_check_after = str(conn.execute("PRAGMA quick_check(1)").fetchone()[0] or "")
        after = _database_metrics(conn, db_path)
        legacy_after = _legacy_snapshot(conn)
        rollup_totals = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(occurrence_count), 0) FROM schema_drift_rollups"
        ).fetchone()

    blockers = []
    warnings = []
    if quick_check_after.lower() != "ok":
        blockers.append("ops_data_plane_quick_check_failed_after_compaction")
    if int(legacy_after["row_count"]) != 0:
        blockers.append("legacy_schema_drift_events_not_empty")
    if vacuum_error:
        warnings.append("ops_data_plane_vacuum_failed")
    reclaimed_bytes = max(int(before["size_bytes"]) - int(after["size_bytes"]), 0)
    return {
        "timestamp_utc": _now_utc(),
        "schema_version": 1,
        "ok": not blockers,
        "overall_status": "blocked" if blockers else ("watch" if warnings else "ready"),
        "grade": "F" if blockers else ("A" if warnings else "A+"),
        "apply": True,
        "blockers": blockers,
        "warnings": warnings,
        "database_before": before,
        "database_after": after,
        "legacy_snapshot": snapshot,
        "legacy_after": legacy_after,
        "rollup_rows": int(rollup_totals[0] or 0),
        "rollup_occurrences": int(rollup_totals[1] or 0),
        "archive": archive,
        "vacuum_requested": bool(vacuum),
        "vacuum_error": vacuum_error,
        "reclaimed_bytes": reclaimed_bytes,
        "reclaimed_gb": round(reclaimed_bytes / (1024**3), 6),
        "integrity": {"before": quick_check_before, "after": quick_check_after},
        "regression_guard": {
            "future_writes": "bounded_schema_drift_rollups",
            "full_payload_authority": "source_jsonl",
            "hot_duplicate_payload_writes": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Archive and compact legacy ops-data-plane schema drift evidence.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--out-file", type=Path)
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--skip-vacuum", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    project_root = args.project_root.expanduser().resolve()
    out_path = args.out_file or project_root / "governance" / "health" / DEFAULT_OUT_PATH.name
    lock_path = args.lock_file or project_root / "governance" / "locks" / DEFAULT_LOCK_PATH.name
    archive_root = args.archive_root
    if archive_root is None:
        archive_root = _default_archive_root(project_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload = {
                "timestamp_utc": _now_utc(),
                "schema_version": 1,
                "ok": True,
                "overall_status": "already_running",
                "grade": "A+",
                "apply": bool(args.apply),
                "blockers": [],
                "warnings": [],
            }
        else:
            try:
                payload = compact_ops_data_plane(
                    project_root,
                    db_path=args.db_path,
                    archive_root=archive_root,
                    apply=bool(args.apply),
                    require_stack_stopped=not bool(args.allow_live),
                    vacuum=not bool(args.skip_vacuum),
                )
            except Exception as exc:
                payload = {
                    "timestamp_utc": _now_utc(),
                    "schema_version": 1,
                    "ok": False,
                    "overall_status": "blocked",
                    "grade": "F",
                    "apply": bool(args.apply),
                    "blockers": ["ops_data_plane_compaction_exception"],
                    "warnings": [],
                    "error_class": type(exc).__name__,
                    "error": str(exc)[-2000:],
                    "database_path": str(args.db_path or ops_data_plane.resolve_db_path(project_root)),
                    "archive_root": str(archive_root),
                }
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        print(
            f"ops_data_plane_compaction={payload.get('overall_status')} "
            f"grade={payload.get('grade')} reclaimed_gb={payload.get('reclaimed_gb', 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
