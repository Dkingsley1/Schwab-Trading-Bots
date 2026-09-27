from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

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
    today = (
        project_root / "governance" / "shadow_default" / "master_control_20260416.jsonl"
    )
    older = (
        project_root / "governance" / "shadow_default" / "master_control_20260414.jsonl"
    )
    _write_jsonl(today, [])
    _write_jsonl(older, [])

    since = datetime(2026, 4, 16, 10, 0, tzinfo=timezone.utc)
    files = snapshot_coverage_sentinel._candidate_master_control_files(
        project_root, since
    )

    assert files == [today]


def test_expected_symbol_floor_uses_maximum_fresh_parallel_heartbeat(
    tmp_path: Path,
) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "shadow_loop_a.json",
        {"timestamp_utc": "2026-08-10T15:00:00+00:00", "symbols_total": 418},
    )
    _write_json(
        health / "shadow_loop_b.json",
        {"timestamp_utc": "2026-08-10T15:00:02+00:00", "symbols_total": 32},
    )
    _write_json(
        health / "shadow_loop_stale.json",
        {"timestamp_utc": "2026-08-10T14:00:00+00:00", "symbols_total": 900},
    )

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
        project_root
        / "governance"
        / "shadow_default"
        / "master_control_20260416.jsonl",
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
    rows_path = (
        project_root / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    )
    _write_json(
        project_root / "governance" / "health" / "shadow_loop_latest.json",
        {"timestamp_utc": "2026-05-25T12:00:00+00:00", "symbols_total": 3},
    )
    _write_json(
        project_root
        / "governance"
        / "health"
        / "runtime_training_snapshot_latest.json",
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


def test_historical_runtime_snapshot_tail_never_earns_fresh_coverage(
    tmp_path: Path,
) -> None:
    project_root = tmp_path
    rows_path = (
        project_root / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    )
    _write_json(
        project_root / "governance" / "health" / "shadow_loop_latest.json",
        {"timestamp_utc": "2026-05-25T12:00:00+00:00", "symbols_total": 2},
    )
    _write_json(
        project_root
        / "governance"
        / "health"
        / "runtime_training_snapshot_latest.json",
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

    assert payload["ok"] is False
    assert payload["fallback_source_count"] == 1
    assert payload["rows_scanned"] == 0
    assert payload["historical_tail_rows_diagnostic_only"] == 2
    assert payload["rows_with_snapshot_id"] == 0
    assert payload["coverage_ratio"] == 0.0


def test_off_hours_shortfall_is_operationally_healthy_but_not_evidence_ready(
    tmp_path: Path,
) -> None:
    project_root = tmp_path
    rows_path = (
        project_root / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    )
    _write_json(
        project_root / "governance" / "health" / "shadow_loop_latest.json",
        {"timestamp_utc": "2026-08-09T12:00:00+00:00", "symbols_total": 10},
    )
    _write_json(
        project_root
        / "governance"
        / "health"
        / "runtime_training_snapshot_latest.json",
        {"rows_path": str(rows_path)},
    )
    _write_jsonl(
        rows_path,
        [{"timestamp_utc": "2026-08-09T11:30:00+00:00", "snapshot_id": "btc"}],
    )

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
        project_root
        / "governance"
        / "shadow_default"
        / "master_control_20260810.jsonl",
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


def test_healthy_post_restart_fanout_is_warming_without_claiming_coverage(
    tmp_path: Path,
) -> None:
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
    _write_json(
        health / "shadow_loop_latest.json",
        {"timestamp_utc": now.isoformat(), "symbols_total": 10},
    )
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


def test_reverse_tail_does_not_read_entire_large_file(tmp_path: Path) -> None:
    path = tmp_path / "large.jsonl"
    with path.open("wb") as handle:
        handle.seek(100 * 1024 * 1024)
        handle.write(b'\n{"snapshot_id":"last"}\n')
    budget = snapshot_coverage_sentinel.ScanBudget(max_bytes=1024 * 1024)
    rows = list(
        snapshot_coverage_sentinel._iter_jsonl_tail_rows(
            path, max_rows=1, budget=budget
        )
    )
    assert rows == [{"snapshot_id": "last"}]
    assert budget.bytes_read <= snapshot_coverage_sentinel.REVERSE_SCAN_BLOCK_BYTES


def test_oversized_line_and_missing_ids_stay_bounded(tmp_path: Path) -> None:
    path = tmp_path / "large.jsonl"
    path.write_bytes(b"x" * (4 * 1024 * 1024))
    budget = snapshot_coverage_sentinel.ScanBudget()
    assert (
        list(snapshot_coverage_sentinel._iter_jsonl_tail_rows(path, budget=budget))
        == []
    )
    assert "oversized_line" in budget.reasons
    assert (
        budget.bytes_read
        <= snapshot_coverage_sentinel.MAX_LINE_BYTES
        + snapshot_coverage_sentinel.REVERSE_SCAN_BLOCK_BYTES
    )
    _write_jsonl(path, [{"timestamp_utc": "2026-08-10T15:00:00Z"}] * 1000)
    budget = snapshot_coverage_sentinel.ScanBudget(max_bytes=1024)
    list(
        snapshot_coverage_sentinel._iter_recent_jsonl_rows(
            path, datetime(2026, 8, 10, tzinfo=timezone.utc), budget=budget
        )
    )
    assert budget.bytes_read == 1024
    assert "scan_byte_limit" in budget.reasons


def test_future_and_non_object_rows_do_not_count(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '[]\n{"timestamp_utc":"2099-01-01T00:00:00Z","snapshot_id":"future"}\n'
    )
    assert (
        list(
            snapshot_coverage_sentinel._iter_recent_jsonl_rows(
                path, datetime(2026, 8, 10, tzinfo=timezone.utc)
            )
        )
        == []
    )


def test_unordered_rows_beyond_long_stale_tail_keep_only_current_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rows.jsonl"
    recent = {"timestamp_utc": "2026-08-10T14:00:00Z", "snapshot_id": "recent"}
    stale = {"timestamp_utc": "2026-08-09T14:00:00Z", "snapshot_id": "old"}
    future = {"timestamp_utc": "2026-08-10T16:00:00Z", "snapshot_id": "future"}
    _write_jsonl(path, [recent, *([stale] * 60), future])

    rows = list(
        snapshot_coverage_sentinel._iter_recent_jsonl_rows(
            path,
            datetime(2026, 8, 10, 13, tzinfo=timezone.utc),
            now=datetime(2026, 8, 10, 15, tzinfo=timezone.utc),
        )
    )

    assert rows == [recent]


@pytest.mark.parametrize(
    ("limits", "reason"),
    [
        ({"max_bytes": 8192}, "scan_byte_limit"),
        ({"max_file_bytes": 4096}, "per_file_byte_limit"),
        ({"max_seconds": 0.75}, "scan_deadline"),
    ],
)
def test_unordered_stale_tail_retains_scan_budgets(
    tmp_path: Path, monkeypatch, limits: dict, reason: str
) -> None:
    path = tmp_path / "rows.jsonl"
    _write_jsonl(
        path,
        [
            {"timestamp_utc": "2026-08-10T14:00:00Z", "snapshot_id": "recent"},
            *(
                [{"timestamp_utc": "2026-08-09T14:00:00Z", "snapshot_id": "old"}]
                * 1000
            ),
        ],
    )
    clock = [0.0]
    monkeypatch.setattr(snapshot_coverage_sentinel.time, "monotonic", lambda: clock[0])
    parse_ts = snapshot_coverage_sentinel._parse_ts

    def timed_parse(raw):
        clock[0] += 0.01
        return parse_ts(raw)

    monkeypatch.setattr(snapshot_coverage_sentinel, "_parse_ts", timed_parse)
    budget = snapshot_coverage_sentinel.ScanBudget(started=clock[0], **limits)
    rows = list(
        snapshot_coverage_sentinel._iter_recent_jsonl_rows(
            path,
            datetime(2026, 8, 10, 13, tzinfo=timezone.utc),
            block_bytes=1024,
            budget=budget,
            now=datetime(2026, 8, 10, 15, tzinfo=timezone.utc),
        )
    )

    assert rows == []
    assert reason in budget.reasons
    assert budget.bytes_read <= min(budget.max_bytes, budget.max_file_bytes)
    assert clock[0] <= budget.max_seconds + 0.01


def test_unordered_coverage_still_stops_at_existing_floor(
    tmp_path: Path, monkeypatch
) -> None:
    now = datetime(2026, 8, 10, 15, tzinfo=timezone.utc)
    _write_json(
        tmp_path / "governance/health/shadow_loop_latest.json",
        {"timestamp_utc": now.isoformat(), "symbols_total": 4},
    )
    path = tmp_path / "governance/shadow_default/master_control_20260810.jsonl"
    _write_jsonl(
        path,
        [
            *[
                {"timestamp_utc": "2026-08-10T14:00:00Z", "snapshot_id": snapshot_id}
                for snapshot_id in ("unneeded", "one", "two", "three")
            ],
            *(
                [{"timestamp_utc": "2026-08-09T14:00:00Z", "snapshot_id": "old"}]
                * 60
            ),
            {"timestamp_utc": "2026-08-10T16:00:00Z", "snapshot_id": "future"},
        ],
    )
    reverse_rows = snapshot_coverage_sentinel._iter_reverse_rows

    def bounded_rows(*args, **kwargs):
        for row in reverse_rows(*args, **kwargs):
            assert row["snapshot_id"] != "unneeded", "scan continued past coverage floor"
            yield row

    monkeypatch.setattr(snapshot_coverage_sentinel, "_iter_reverse_rows", bounded_rows)
    payload = snapshot_coverage_sentinel.build_payload(
        hours=2, min_coverage_ratio=0.75, project_root=tmp_path, now=now
    )

    assert payload["ok"] is True
    assert payload["unique_snapshot_ids"] == 3
    assert payload["rows_scanned"] == 3
    assert payload["required_unique_snapshot_floor"] == 3
    assert payload["coverage_ratio"] == 0.75
    assert payload["stopped_after_reaching_floor"] is True


def test_invalid_index_counts_and_future_timestamps_fail_closed(tmp_path: Path) -> None:
    now = datetime(2026, 8, 10, 15, tzinfo=timezone.utc)
    base = {
        "window_hours": 2,
        "window_ended_utc": now.isoformat(),
        "row_count": 8,
        "rows_with_snapshot_id": 8,
        "unique_snapshot_ids": 8,
    }
    for change in (
        {"unique_snapshot_ids": 10**15},
        {"unique_snapshot_ids": True},
        {"unique_snapshot_ids": "8"},
        {"rows_with_snapshot_id": -1},
        {"window_ended_utc": "2099-01-01T00:00:00Z"},
    ):
        _write_json(
            tmp_path / "governance/health/runtime_training_snapshot_latest.json",
            {
                "timestamp_utc": now.isoformat(),
                "coverage": {"recent_windows": {"2": {**base, **change}}},
            },
        )
        assert (
            snapshot_coverage_sentinel._runtime_snapshot_window(
                tmp_path, hours=2, now=now
            )
            == {}
        )


def test_protected_alias_is_rejected_before_open(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "alias.jsonl"
    path.symlink_to("/Volumes/VIDEO/private.jsonl")

    def forbidden_open(*args, **kwargs):
        raise AssertionError("must reject route before opening")

    monkeypatch.setattr(Path, "open", forbidden_open)
    budget = snapshot_coverage_sentinel.ScanBudget()
    assert (
        list(snapshot_coverage_sentinel._iter_jsonl_tail_rows(path, budget=budget))
        == []
    )
    assert "source_unavailable_or_protected" in budget.reasons


def test_expired_scan_budget_returns_without_io(tmp_path: Path) -> None:
    budget = snapshot_coverage_sentinel.ScanBudget(max_seconds=0)
    payload = snapshot_coverage_sentinel.build_payload(
        hours=2, min_coverage_ratio=0.75, project_root=tmp_path, scan_budget=budget
    )
    assert payload["ok"] is False
    assert payload["scan_budget"]["bytes_read"] == 0
