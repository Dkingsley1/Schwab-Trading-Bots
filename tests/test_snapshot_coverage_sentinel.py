from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import snapshot_coverage_sentinel


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_candidate_files_skip_days_outside_window(tmp_path: Path) -> None:
    project_root = tmp_path
    today = project_root / "governance" / "shadow_default" / "master_control_20260416.jsonl"
    older = project_root / "governance" / "shadow_default" / "master_control_20260414.jsonl"
    _write_jsonl(today, [])
    _write_jsonl(older, [])

    since = datetime(2026, 4, 16, 10, 0, tzinfo=timezone.utc)
    files = snapshot_coverage_sentinel._candidate_master_control_files(project_root, since)

    assert files == [today]


def test_expected_symbol_floor_uses_maximum_fresh_parallel_heartbeat(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "shadow_loop_a.json", {"timestamp_utc": "2026-08-10T15:00:00+00:00", "symbols_total": 418})
    _write_json(health / "shadow_loop_b.json", {"timestamp_utc": "2026-08-10T15:00:02+00:00", "symbols_total": 32})
    _write_json(health / "shadow_loop_stale.json", {"timestamp_utc": "2026-08-10T14:00:00+00:00", "symbols_total": 900})

    assert snapshot_coverage_sentinel._latest_heartbeat_symbols_total(tmp_path) == 418


def test_build_payload_counts_recent_tail_rows_only(tmp_path: Path) -> None:
    project_root = tmp_path
    _write_json(
        project_root / "governance" / "health" / "shadow_loop_latest.json",
        {"timestamp_utc": "2026-04-16T12:00:00+00:00", "symbols_total": 2},
    )
    rows = [
        {"timestamp_utc": "2026-04-16T09:50:00+00:00", "snapshot_id": "old"},
        {"timestamp_utc": "2026-04-16T10:50:00+00:00", "snapshot_id": "recent-1"},
        {"timestamp_utc": "2026-04-16T11:10:00+00:00", "snapshot_id": "recent-2"},
        {"timestamp_utc": "2026-04-16T11:20:00+00:00"},
    ]
    _write_jsonl(
        project_root / "governance" / "shadow_default" / "master_control_20260416.jsonl",
        rows,
    )

    payload = snapshot_coverage_sentinel.build_payload(
        hours=2,
        min_coverage_ratio=0.75,
        project_root=project_root,
        now=datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc),
    )

    assert payload["ok"] is True
    assert payload["files_considered"] == 1
    assert payload["rows_scanned"] == 3
    assert payload["rows_with_snapshot_id"] == 2
    assert payload["unique_snapshot_ids"] == 2
    assert payload["coverage_ratio"] == 1.0


def test_build_payload_falls_back_to_runtime_training_snapshot(tmp_path: Path) -> None:
    project_root = tmp_path
    rows_path = project_root / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    _write_json(
        project_root / "governance" / "health" / "shadow_loop_latest.json",
        {"timestamp_utc": "2026-05-25T12:00:00+00:00", "symbols_total": 3},
    )
    _write_json(
        project_root / "governance" / "health" / "runtime_training_snapshot_latest.json",
        {"rows_path": str(rows_path)},
    )
    _write_jsonl(
        rows_path,
        [
            {"timestamp_utc": "2026-05-25T10:30:00+00:00", "snapshot_id": "btc"},
            {"timestamp_utc": "2026-05-25T10:40:00+00:00", "snapshot_id": "eth"},
            {"timestamp_utc": "2026-05-25T10:50:00+00:00", "snapshot_id": "sol"},
        ],
    )

    payload = snapshot_coverage_sentinel.build_payload(
        hours=2,
        min_coverage_ratio=0.75,
        project_root=project_root,
        now=datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc),
    )

    assert payload["ok"] is True
    assert payload["files_considered"] == 0
    assert payload["primary_source_count"] == 1
    assert payload["fallback_source_count"] == 0
    assert payload["rows_scanned"] == 3
    assert payload["rows_with_snapshot_id"] == 3
    assert payload["coverage_ratio"] == 1.0


def test_build_payload_uses_runtime_snapshot_tail_when_rows_are_historical(tmp_path: Path) -> None:
    project_root = tmp_path
    rows_path = project_root / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    _write_json(
        project_root / "governance" / "health" / "shadow_loop_latest.json",
        {"timestamp_utc": "2026-05-25T12:00:00+00:00", "symbols_total": 2},
    )
    _write_json(
        project_root / "governance" / "health" / "runtime_training_snapshot_latest.json",
        {"rows_path": str(rows_path)},
    )
    _write_jsonl(
        rows_path,
        [
            {"timestamp_utc": "2026-05-24T09:30:00+00:00", "snapshot_id": "spy-old"},
            {"timestamp_utc": "2026-05-24T09:35:00+00:00", "snapshot_id": "qqq-old"},
        ],
    )

    payload = snapshot_coverage_sentinel.build_payload(
        hours=2,
        min_coverage_ratio=0.75,
        project_root=project_root,
        now=datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc),
    )

    assert payload["ok"] is True
    assert payload["fallback_source_count"] == 1
    assert payload["rows_scanned"] == 2
    assert payload["rows_with_snapshot_id"] == 2
    assert payload["coverage_ratio"] == 1.0


def test_off_hours_shortfall_is_operationally_healthy_but_not_evidence_ready(tmp_path: Path) -> None:
    project_root = tmp_path
    rows_path = project_root / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    _write_json(
        project_root / "governance" / "health" / "shadow_loop_latest.json",
        {"timestamp_utc": "2026-08-09T12:00:00+00:00", "symbols_total": 10},
    )
    _write_json(
        project_root / "governance" / "health" / "runtime_training_snapshot_latest.json",
        {"rows_path": str(rows_path)},
    )
    _write_jsonl(rows_path, [{"timestamp_utc": "2026-08-09T11:30:00+00:00", "snapshot_id": "btc"}])

    payload = snapshot_coverage_sentinel.build_payload(
        hours=2,
        min_coverage_ratio=0.75,
        project_root=project_root,
        now=datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc),
    )

    assert payload["market_window"]["is_weekend"] is True
    assert payload["ok"] is False
    assert payload["evidence_ready"] is False
    assert payload["operational_ok"] is True
    assert payload["overall_status"] == "collecting_off_hours"


def test_market_hours_shortfall_remains_operational_failure(tmp_path: Path) -> None:
    project_root = tmp_path
    _write_json(
        project_root / "governance" / "health" / "shadow_loop_latest.json",
        {"timestamp_utc": "2026-08-10T15:00:00+00:00", "symbols_total": 10},
    )
    _write_jsonl(
        project_root / "governance" / "shadow_default" / "master_control_20260810.jsonl",
        [{"timestamp_utc": "2026-08-10T14:30:00+00:00", "snapshot_id": "spy"}],
    )

    payload = snapshot_coverage_sentinel.build_payload(
        hours=2,
        min_coverage_ratio=0.75,
        project_root=project_root,
        now=datetime(2026, 8, 10, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["market_window"]["active"] is False
    assert payload["evidence_ready"] is False
    assert payload["operational_ok"] is False
    assert payload["overall_status"] == "degraded"


def test_healthy_post_restart_fanout_is_warming_without_claiming_coverage(tmp_path: Path) -> None:
    now = datetime(2026, 8, 10, 15, 0, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "shadow_loop_latest.json",
        {"timestamp_utc": now.isoformat(), "symbols_total": 10},
    )
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "ready",
            "restart_storms": [],
            "status": [
                {
                    "name": "all_sleeves",
                    "process_live": True,
                    "heartbeat_ok": True,
                    "child_fanout_ok": True,
                    "process_elapsed_seconds": 240,
                }
            ],
        },
    )
    _write_jsonl(
        tmp_path / "governance" / "shadow_default" / "master_control_20260810.jsonl",
        [{"timestamp_utc": "2026-08-10T14:30:00+00:00", "snapshot_id": "spy"}],
    )

    payload = snapshot_coverage_sentinel.build_payload(
        hours=2,
        min_coverage_ratio=0.75,
        project_root=tmp_path,
        now=now,
    )

    assert payload["ok"] is False
    assert payload["evidence_ready"] is False
    assert payload["operational_ok"] is True
    assert payload["overall_status"] == "warming_after_restart"
    assert payload["startup_grace"]["active"] is True


def test_restart_storm_disables_collection_startup_grace(tmp_path: Path) -> None:
    now = datetime(2026, 8, 10, 15, 0, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(health / "shadow_loop_latest.json", {"timestamp_utc": now.isoformat(), "symbols_total": 10})
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "degraded",
            "restart_storms": [{"name": "all_sleeves"}],
            "status": [
                {
                    "name": "all_sleeves",
                    "process_live": True,
                    "heartbeat_ok": True,
                    "child_fanout_ok": True,
                    "process_elapsed_seconds": 120,
                }
            ],
        },
    )
    _write_jsonl(
        tmp_path / "governance" / "shadow_default" / "master_control_20260810.jsonl",
        [{"timestamp_utc": "2026-08-10T14:30:00+00:00", "snapshot_id": "spy"}],
    )

    payload = snapshot_coverage_sentinel.build_payload(
        hours=2,
        min_coverage_ratio=0.75,
        project_root=tmp_path,
        now=now,
    )

    assert payload["operational_ok"] is False
    assert payload["overall_status"] == "degraded"
    assert payload["startup_grace"]["active"] is False


def test_snapshot_coverage_uses_compact_recent_window_index(tmp_path: Path) -> None:
    now = datetime(2026, 8, 10, 15, 0, tzinfo=timezone.utc)
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("", encoding="utf-8")
    _write_json(
        tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "rows_path": str(rows_path),
            "coverage": {
                "recent_windows": {
                    "2": {
                        "window_hours": 2,
                        "window_ended_utc": now.isoformat(),
                        "row_count": 8,
                        "rows_with_snapshot_id": 8,
                        "unique_snapshot_ids": 8,
                        "unique_symbols": 4,
                    }
                }
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "shadow_loop_latest.json",
        {"timestamp_utc": now.isoformat(), "symbols_total": 10},
    )

    payload = snapshot_coverage_sentinel.build_payload(
        hours=2,
        min_coverage_ratio=0.75,
        project_root=tmp_path,
        now=now,
    )

    assert payload["ok"] is True
    assert payload["rows_scanned"] == 8
    assert payload["unique_snapshot_ids"] == 8
    assert payload["indexed_snapshot_window"]["unique_symbols"] == 4
