from __future__ import annotations

import fcntl
from pathlib import Path

from scripts import build_runtime_training_snapshot as src
import hashlib
import json
import sys
from datetime import datetime, timezone

import pytest

def test_single_flight_lock_reports_already_running_when_snapshot_builder_is_active(tmp_path: Path) -> None:
    lock_path = tmp_path / "governance" / "locks" / "runtime_training_snapshot.lock"
    rows_path = tmp_path / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    health_path = tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json"
    lock_path.parent.mkdir(parents=True)
    held = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        held.write("pid=123\n")
        held.flush()

        handle, payload = src._acquire_single_flight_lock(
            lock_path,
            project_root=tmp_path,
            health_path=health_path,
            rows_path=rows_path,
        )

        assert handle is None
        assert payload["ok"] is True
        assert payload["overall_status"] == "already_running"
        assert payload["already_running"] is True
        assert payload["single_flight_contract"]["prevents_duplicate_snapshot_builders"] is True
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()


def test_full_refresh_falls_back_to_jsonl_when_sqlite_returns_no_sequences(monkeypatch, tmp_path: Path) -> None:
    calls: list[bool] = []
    jsonl_sequences = {
        ("shadow_aggressive_equities", "SPY"): [
            {
                "timestamp_utc": "2026-07-12T12:00:00+00:00",
                "strategy": "test_strategy",
                "strategy_priority": 1,
                "snapshot_id": "SPY:1",
                "ts_epoch": 1.0,
                "price": 500.0,
                "features": {"x": 1.0},
                "mode": "shadow_aggressive_equities",
                "symbol": "SPY",
            }
        ]
    }

    def fake_load_runtime_observation_sequences(*args, **kwargs):
        prefer_sqlite = bool(kwargs.get("prefer_sqlite"))
        calls.append(prefer_sqlite)
        return {} if prefer_sqlite else jsonl_sequences

    monkeypatch.setattr(src.rtc, "load_runtime_observation_sequences", fake_load_runtime_observation_sequences)

    sequences, meta = src._full_refresh_sequences(
        tmp_path,
        lookback_days=14,
        mode_allowlist=[],
        symbol_allowlist=[],
        prefer_sqlite=True,
        max_observation_rows=80000,
    )

    assert calls == [True, False]
    assert sequences == jsonl_sequences
    assert meta["build_mode"] == "full_refresh_jsonl_fallback"
    assert meta["sqlite_empty_fallback"] is True


def test_full_refresh_keeps_sqlite_result_when_available(monkeypatch, tmp_path: Path) -> None:
    calls: list[bool] = []
    sqlite_sequences = {
        ("shadow_crypto", "BTC-USD"): [
            {
                "timestamp_utc": "2026-07-12T12:00:00+00:00",
                "strategy": "test_strategy",
                "strategy_priority": 1,
                "snapshot_id": "BTC:1",
                "ts_epoch": 1.0,
                "price": 60000.0,
                "features": {"x": 1.0},
                "mode": "shadow_crypto",
                "symbol": "BTC-USD",
            }
        ]
    }

    def fake_load_runtime_observation_sequences(*args, **kwargs):
        calls.append(bool(kwargs.get("prefer_sqlite")))
        return sqlite_sequences

    monkeypatch.setattr(src.rtc, "load_runtime_observation_sequences", fake_load_runtime_observation_sequences)

    sequences, meta = src._full_refresh_sequences(
        tmp_path,
        lookback_days=14,
        mode_allowlist=[],
        symbol_allowlist=[],
        prefer_sqlite=True,
        max_observation_rows=80000,
    )

    assert calls == [True]
    assert sequences == sqlite_sequences
    assert meta == {"build_mode": "full_refresh"}


def test_interrupted_rows_publication_preserves_previous_generation(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b"previous-generation\n")
    sequences = {("paper", "SPY"): [{"snapshot_id": "valid"}, {"invalid": object()}]}
    with pytest.raises(TypeError):
        src._publish_snapshot_rows(rows, sequences)
    assert rows.read_bytes() == b"previous-generation\n"
    assert not (tmp_path / ".rows.jsonl.building").exists()


def test_atomic_publication_returns_digest_of_exact_bytes(tmp_path):
    path = tmp_path / "rows.jsonl"
    row_count, sequence_count, digest = src._publish_snapshot_rows(
        path, {("paper", "SPY"): [{"snapshot_id": "one"}]}
    )
    assert (row_count, sequence_count) == (1, 1)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.stat().st_mode & 0o777 == 0o600


def test_reader_rejects_torn_rows_and_health_generations(tmp_path):
    rows = tmp_path / "rows.jsonl"
    health = tmp_path / "health.json"
    sequences = {
        ("paper", "SPY"): [
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "snapshot_id": "one",
            }
        ]
    }
    _, _, digest = src._publish_snapshot_rows(rows, sequences)
    health.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "lookback_days": 14,
                "rows_path": str(rows),
                "rows_sha256": digest,
            }
        )
    )
    arguments = dict(
        lookback_days=1, mode_allowlist=[], symbol_allowlist=[], snapshot_file=health
    )
    assert src.rtc._load_runtime_snapshot_rows(tmp_path, **arguments)
    sequences[("paper", "SPY")][0]["snapshot_id"] = "two"
    src._publish_snapshot_rows(rows, sequences)
    assert src.rtc._load_runtime_snapshot_rows(tmp_path, **arguments) == {}


def test_v2_snapshot_requires_hash_for_reader_and_reuse(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("{}\n")
    summary = {"schema_version": 2, "lookback_days": 14, "rows_path": str(rows)}
    health = tmp_path / "health.json"
    health.write_text(json.dumps(summary))
    assert src._snapshot_rows_match(summary) is False
    assert (
        src.rtc._load_runtime_snapshot_rows(
            tmp_path,
            lookback_days=1,
            mode_allowlist=[],
            symbol_allowlist=[],
            snapshot_file=health,
        )
        == {}
    )


def test_full_worker_timeout_reports_phase_and_reaps_child(monkeypatch, capsys):
    real_runner = src.run_bounded_process_group

    def hanging_worker(command, **kwargs):
        assert command[-1] == "--bounded-worker"
        return real_runner(
            [
                sys.executable,
                "-c",
                'import sys,time; print(\'{"snapshot_phase":"discovery"}\', file=sys.stderr, flush=True); time.sleep(10)',
            ],
            **kwargs,
        )

    monkeypatch.setattr(src, "run_bounded_process_group", hanging_worker)
    assert src._run_bounded_snapshot(["--json"], timeout_seconds=1) == 124
    report = json.loads(capsys.readouterr().out)
    assert report["last_phase"] == "discovery"
    assert report["timeout_cleanup"]["reaped"] is True
    assert report["publication_verified"] is False


def test_publisher_refuses_linked_temporary_file(tmp_path):
    target = tmp_path / "unrelated"
    target.write_text("preserved")
    (tmp_path / ".rows.jsonl.building").symlink_to(target)
    with pytest.raises(OSError):
        src._publish_snapshot_rows(tmp_path / "rows.jsonl", {})
    assert target.read_text() == "preserved"


def test_snapshot_reader_rejects_protected_row_route_without_target_metadata(tmp_path, monkeypatch):
    rows = tmp_path / "rows.jsonl"
    rows.symlink_to("/Volumes/VIDEO/rows.jsonl")
    summary = {"schema_version": 2, "lookback_days": 14, "rows_path": str(rows), "rows_sha256": "a" * 64}
    health = tmp_path / "health.json"
    health.write_text(json.dumps(summary))
    original = Path.lstat

    def guarded(path, *args, **kwargs):
        assert not str(path).casefold().startswith("/volumes/video")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", guarded)
    assert src._snapshot_rows_match(summary) is False
    assert src.rtc._load_runtime_snapshot_rows(tmp_path, lookback_days=1, mode_allowlist=[], symbol_allowlist=[], snapshot_file=health) == {}
