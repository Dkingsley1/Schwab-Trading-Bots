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
