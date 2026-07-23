from __future__ import annotations

import fcntl
from pathlib import Path

from scripts import build_runtime_training_snapshot as src


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
