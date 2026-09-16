from __future__ import annotations

import fcntl
from pathlib import Path

from scripts import build_runtime_training_snapshot as src
import hashlib
import json
import sys
import subprocess
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest


def test_real_snapshot_worker_publishes_matching_manifest_and_rows(tmp_path):
    now = datetime.now(timezone.utc)
    source = tmp_path / "decisions" / "shadow" / f"trade_decisions_{now:%Y%m%d}.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "timestamp_utc": now.isoformat(),
                "symbol": "SPY",
                "strategy": "grand_master_bot",
                "features": {"last_price": 100},
                "metadata": {
                    "layer": "grand_master",
                    "mode": "shadow",
                    "snapshot_id": "fixture-one",
                },
            }
        )
        + "\n"
    )
    rows = tmp_path / "rows.jsonl"
    health = tmp_path / "health.json"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(src.__file__)),
            "--project-root",
            str(tmp_path),
            "--rows-path",
            str(rows),
            "--health-path",
            str(health),
            "--lock-path",
            str(tmp_path / "snapshot.lock"),
            "--seed-health-path",
            str(tmp_path / "absent-seed.json"),
            "--no-prefer-sqlite",
            "--max-runtime-seconds",
            "30",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["row_count"] == 1
    assert report["rows_sha256"] == hashlib.sha256(rows.read_bytes()).hexdigest()
    assert json.loads(health.read_text()) == report
    assert '"snapshot_phase": "completed"' in result.stderr


def test_incremental_sidecar_shares_scan_deadline_and_reports_partial(
    tmp_path, monkeypatch
):
    path = tmp_path / "rows.jsonl"
    path.write_text("{}\n")
    seen = {}
    monkeypatch.setenv("RUNTIME_TRAIN_PRICE_SIDECAR_ENABLED", "1")

    def bounded_sidecar(paths, **kwargs):
        seen.update(kwargs)
        kwargs["stats"]["byte_limit_hit"] = True
        return iter([])

    monkeypatch.setattr(src.rtc, "_iter_runtime_price_sidecar_rows", bounded_sidecar)
    monkeypatch.setattr(src.rtc, "_load_runtime_gap_fill_context", lambda *a: {})
    before = src.time.monotonic()
    count, stats = src._merge_candidate_rows_into_sequences(
        {},
        candidate_paths=[path],
        project_root=tmp_path,
        since_utc=datetime.now(timezone.utc),
        mode_allowlist=[],
        symbol_allowlist=[],
        max_runtime_seconds=30,
    )
    assert before + 30 <= seen["deadline_monotonic"] <= src.time.monotonic() + 30
    assert count == 0
    assert stats["price_sidecar_scan"]["byte_limit_hit"]
    assert stats["candidate_scan_partial"]


def test_expired_worker_scan_budget_preserves_base_without_reopening_sources(
    tmp_path, monkeypatch
):
    path = tmp_path / "rows.jsonl"
    path.write_text("{}\n")
    base = {("shadow", "SPY"): [{"snapshot_id": "existing"}]}
    monkeypatch.setenv("RUNTIME_TRAIN_PRICE_SIDECAR_ENABLED", "0")
    monkeypatch.setattr(src.rtc, "_load_runtime_gap_fill_context", lambda *a: {})
    monkeypatch.setattr(
        src,
        "_iter_recent_json_rows_newest_first",
        lambda *a, **kw: pytest.fail("scan resumed after publication reserve"),
    )
    count, stats = src._merge_candidate_rows_into_sequences(
        base,
        candidate_paths=[path],
        project_root=tmp_path,
        since_utc=datetime.now(timezone.utc),
        mode_allowlist=[],
        symbol_allowlist=[],
        max_runtime_seconds=180,
        deadline_monotonic=src.time.monotonic() - 1,
    )
    assert count == 0
    assert base[("shadow", "SPY")][0]["snapshot_id"] == "existing"
    assert stats["candidate_scan_timed_out"]
    assert stats["candidate_scan_partial"]


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


def test_light_refresh_existing_takes_precedence_over_reuse_and_publishes_health(
    tmp_path: Path, monkeypatch
) -> None:
    rows_path = tmp_path / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    health_path = tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json"
    rows_path.parent.mkdir(parents=True)
    health_path.parent.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    rows_path.write_text(
        json.dumps({"snapshot_id": "one", "symbol": "SPY", "timestamp_utc": now.isoformat()}) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    original_ts = now.replace(microsecond=0).isoformat()
    health_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "timestamp_utc": original_ts,
                "project_root": str(tmp_path),
                "lookback_days": 14,
                "mode_allowlist": [],
                "symbol_allowlist": [],
                "prefer_sqlite": True,
                "rows_path": str(rows_path),
                "rows_sha256": digest,
                "sequence_count": 1,
                "row_count": 1,
                "latest_row_timestamp_utc": now.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        src,
        "_full_refresh_sequences",
        lambda *args, **kwargs: pytest.fail("full refresh should not run"),
    )
    monkeypatch.setattr(
        src.rtc, "_load_runtime_snapshot_rows",
        lambda *a, **kw: pytest.fail("light refresh must not materialize sequences"),
    )
    monkeypatch.setattr(
        src, "_snapshot_rows_match",
        lambda *a, **kw: pytest.fail("light refresh must hash inside its bounded pass"),
    )

    rc = src._build_locked_snapshot(
        SimpleNamespace(
            lookback_days=14,
            prefer_sqlite=True,
            reuse_if_fresh_minutes=10,
            light_refresh_existing=True,
            json=True,
            incremental_max_runtime_seconds=0,
            incremental_max_candidate_rows=0,
            max_observation_rows=0,
            max_sequences=0,
            max_rows_per_sequence=0,
        ),
        tmp_path,
        rows_path,
        health_path,
        tmp_path / "snapshot.lock",
        None,
        [],
        [],
    )

    assert rc == 0
    refreshed = json.loads(health_path.read_text(encoding="utf-8"))
    assert refreshed["timestamp_utc"] != original_ts
    assert refreshed["build_mode"] == "light_metadata_refresh"
    assert refreshed["reuse_reason"] == "light_refresh_existing_snapshot"
    assert refreshed["age_minutes"] == 0.0
    assert refreshed["latest_row_timestamp_utc"] == now.isoformat()
    assert refreshed["coverage"]["recent_windows"]["1"]["row_count"] == 1
    assert refreshed["coverage"]["recent_windows"]["1"]["window_ended_utc"] == refreshed["timestamp_utc"]
    assert refreshed["coverage"]["recent_windows_scope"] == "stored_snapshot_rows"
    assert refreshed["coverage"]["current_ingestion_verified"] is False
    verification = refreshed["coverage"]["recent_windows_verification"]
    assert verification["status"] == "complete"
    assert verification["rows_sha256"] == digest
    assert verification["row_count"] == 1
    assert verification["bytes_read"] == verification["byte_limit"] == rows_path.stat().st_size


def _stored_coverage_fixture(tmp_path, rows=None):
    now = datetime.now(timezone.utc)
    rows = rows if rows is not None else [
        {"timestamp_utc": now.isoformat(), "snapshot_id": "one", "symbol": "SPY"}
    ]
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b"".join((json.dumps(row) + "\n").encode() for row in rows))
    return {
        "schema_version": 2, "project_root": str(tmp_path), "lookback_days": 14,
        "mode_allowlist": [], "symbol_allowlist": [], "prefer_sqlite": True,
        "rows_path": str(path), "rows_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sequence_count": 1, "row_count": len(rows),
        "latest_row_timestamp_utc": now.isoformat(),
        "timestamp_utc": (now - timedelta(hours=4)).isoformat(),
        "coverage": {"recent_windows": {"1": {"row_count": 999, "window_ended_utc": "old"}}},
    }


def _light_payload(summary, tmp_path, **kwargs):
    return src._light_refresh_existing_snapshot_payload(
        summary, project_root=tmp_path, health_path=tmp_path / "health.json",
        lookback_days=14, mode_allowlist=[], symbol_allowlist=[], prefer_sqlite=True,
        **kwargs,
    )


def test_stored_coverage_stream_is_exact_for_unordered_future_and_duplicate_rows(tmp_path):
    now = datetime.now(timezone.utc)
    rows = [
        {"timestamp_utc": (now - timedelta(hours=age)).isoformat(),
         "snapshot_id": snapshot_id, "symbol": symbol}
        for age, snapshot_id, symbol in (
            (-1, "future", "FUTURE"), (0, "recent", "SPY"), (24, "day", "SPY"),
            (6, "six", "QQQ"), (2, "two", "QQQ"), (1, "one", "SPY"),
            (0, "recent", "QQQ"), (25, "old", "OLD"),
        )
    ]
    summary = _stored_coverage_fixture(tmp_path, rows)
    streamed = src._verified_stored_coverage_windows(summary, now=now)
    full = src._coverage_summary({("shadow", "SPY"): rows}, now=now)
    assert streamed["recent_windows"] == full["recent_windows"]
    for hours, count in ((1, 3), (2, 4), (6, 5), (24, 6)):
        window = streamed["recent_windows"][str(hours)]
        assert window["row_count"] == window["rows_with_snapshot_id"] == count
        assert window["unique_snapshot_ids"] == count - 1
        assert window["unique_symbols"] == 2
        assert window["window_ended_utc"] == now.isoformat()
    assert streamed["recent_windows_verification"]["row_count"] == len(rows)


@pytest.mark.parametrize("failure", [
    "hash", "count_short", "count_long", "missing_hash", "legacy", "invalid_timestamp",
    "missing_symbol", "malformed", "truncated", "nested_sequence",
])
def test_light_coverage_integrity_failure_preserves_previous_summary(tmp_path, failure):
    summary = _stored_coverage_fixture(tmp_path)
    path = Path(summary["rows_path"])
    if failure == "hash":
        summary["rows_sha256"] = "0" * 64
    elif failure == "count_short":
        path.write_bytes(path.read_bytes() * 2)
        summary["rows_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif failure == "count_long":
        summary["row_count"] += 1
    elif failure == "missing_hash":
        summary.pop("rows_sha256")
    elif failure == "legacy":
        summary["schema_version"] = 1
    else:
        raw = path.read_bytes()
        if failure == "malformed":
            raw = b"{broken}\n"
        elif failure == "truncated":
            raw = raw[:-1]
        else:
            row = json.loads(raw)
            if failure == "invalid_timestamp":
                row["timestamp_utc"] = "invalid"
            elif failure == "missing_symbol":
                row.pop("symbol")
            else:
                row = {"rows": [row]}
            raw = (json.dumps(row) + "\n").encode()
        path.write_bytes(raw)
        summary["rows_sha256"] = hashlib.sha256(raw).hexdigest()
    before = json.dumps(summary, sort_keys=True)
    assert _light_payload(summary, tmp_path) == {}
    assert json.dumps(summary, sort_keys=True) == before


@pytest.mark.parametrize("budget", ["bytes", "line", "index", "time", "outer_deadline"])
def test_stored_coverage_budget_failure_never_returns_partial_windows(tmp_path, monkeypatch, budget):
    summary = _stored_coverage_fixture(tmp_path)
    kwargs = {}
    clock = [10.0]
    monkeypatch.setattr(src.time, "monotonic", lambda: clock[0])
    if budget == "bytes":
        kwargs["max_bytes"] = Path(summary["rows_path"]).stat().st_size - 1
    elif budget == "line":
        monkeypatch.setattr(src, "_LIGHT_COVERAGE_MAX_LINE_BYTES", 16)
    elif budget == "index":
        monkeypatch.setattr(src, "_LIGHT_COVERAGE_MAX_INDEX_BYTES", 1)
    else:
        if budget == "outer_deadline":
            kwargs["deadline_monotonic"] = 12.0
        record = src._record_recent_row

        def expire_after_final_row(*a, **kw):
            result = record(*a, **kw)
            clock[0] = 12.0 if budget == "outer_deadline" else 25.0
            return result

        monkeypatch.setattr(src, "_record_recent_row", expire_after_final_row)
    original = json.dumps(summary, sort_keys=True)
    assert src._verified_stored_coverage_windows(summary, now=datetime.now(timezone.utc), **kwargs) == {}
    assert json.dumps(summary, sort_keys=True) == original


@pytest.mark.parametrize("change", ["replace", "append"])
def test_stored_coverage_rejects_source_changes_even_with_matching_initial_bytes(tmp_path, monkeypatch, change):
    summary = _stored_coverage_fixture(tmp_path)
    path = Path(summary["rows_path"])
    record = src._record_recent_row

    def change_source(*a, **kw):
        result = record(*a, **kw)
        if change == "replace":
            replacement = tmp_path / "replacement.jsonl"
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        else:
            with path.open("ab") as handle:
                handle.write(b"{}\n")
        return result

    monkeypatch.setattr(src, "_record_recent_row", change_source)
    assert _light_payload(summary, tmp_path) == {}


def test_failed_light_coverage_falls_back_without_publishing_or_changing_budgets(tmp_path, monkeypatch):
    summary = _stored_coverage_fixture(tmp_path)
    summary["rows_sha256"] = "0" * 64
    health = tmp_path / "health.json"
    health.write_text(json.dumps(summary))
    before = health.read_bytes(), health.stat().st_mtime_ns
    deadline = src.time.monotonic() + 90
    args = SimpleNamespace(
        light_refresh_existing=True, reuse_if_fresh_minutes=0, lookback_days=14,
        prefer_sqlite=True, json=True, incremental_max_runtime_seconds=15,
        incremental_max_candidate_rows=25000, scan_deadline_monotonic=deadline,
    )

    def incremental(existing, **kwargs):
        assert existing == summary
        assert kwargs["max_runtime_seconds"] == 15
        assert kwargs["max_candidate_rows"] == 25000
        assert kwargs["deadline_monotonic"] == deadline
        assert (health.read_bytes(), health.stat().st_mtime_ns) == before
        raise RuntimeError("incremental owner reached")

    monkeypatch.setattr(src, "_incremental_snapshot_sequences", incremental)
    monkeypatch.setattr(src, "write_payload", lambda *a: pytest.fail("failed light scan published"))
    with pytest.raises(RuntimeError, match="incremental owner reached"):
        src._build_locked_snapshot(args, tmp_path, Path(summary["rows_path"]), health,
                                   tmp_path / "lock", None, [], [])


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


def test_manifest_interruption_keeps_committed_rows_and_alias(tmp_path, monkeypatch):
    rows = tmp_path / "rows.jsonl"
    health = tmp_path / "health.json"
    args = SimpleNamespace(
        light_refresh_existing=False, reuse_if_fresh_minutes=0,
        lookback_days=14, prefer_sqlite=True, json=True,
        incremental_max_runtime_seconds=30, incremental_max_candidate_rows=100,
        scan_deadline_monotonic=src.time.monotonic() + 120,
        max_observation_rows=100, max_sequences=10, max_rows_per_sequence=10,
    )
    sequences = {("shadow", "SPY"): [{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": "one", "ts_epoch": src.time.time(), "price": 100,
    }]}
    monkeypatch.setattr(src, "_incremental_snapshot_sequences", lambda *a, **kw: None)
    monkeypatch.setattr(src, "_full_refresh_sequences", lambda *a, **kw: (sequences, {}))
    monkeypatch.setattr(src, "_reusable_snapshot_payload", lambda *a, **kw: None)
    build = lambda: src._build_locked_snapshot(
        args, tmp_path, rows, health, tmp_path / "lock", None, [], []
    )
    assert build() == 0
    original = json.loads(health.read_text())
    original_rows = Path(original["rows_path"])
    assert original_rows != rows
    assert src._snapshot_rows_match(original)
    original_bytes = rows.read_bytes()
    write_payload = src.write_payload

    def interrupted(*a, **kw):
        raise OSError("interrupted before manifest commit")

    monkeypatch.setattr(src, "write_payload", interrupted)
    sequences[("shadow", "SPY")][0]["snapshot_id"] = "two"
    with pytest.raises(OSError, match="interrupted"):
        build()
    assert json.loads(health.read_text()) == original
    assert original_rows.read_bytes() == rows.read_bytes() == original_bytes
    assert src._snapshot_rows_match(original)

    monkeypatch.setattr(src, "write_payload", write_payload)

    def collect_after_commit():
        committed = json.loads(health.read_text())
        assert committed["rows_path"] != original["rows_path"]
        assert src._snapshot_rows_match(committed)

    monkeypatch.setattr(src.gc, "collect", collect_after_commit)
    assert build() == 0
    current = json.loads(health.read_text())
    assert rows.read_bytes() == Path(current["rows_path"]).read_bytes()
    assert original_rows.read_bytes() == original_bytes


@pytest.mark.parametrize("seed", [False, True])
@pytest.mark.parametrize("matches", [False, True])
def test_base_digest_checked_before_parse_and_reader_receives_deadline(
    tmp_path, monkeypatch, seed, matches
):
    monkeypatch.setattr(src, "_summary_config_compatible", lambda *a, **kw: True)
    monkeypatch.setattr(src, "_summary_can_seed_target", lambda *a, **kw: True)
    monkeypatch.setattr(src, "_snapshot_rows_match", lambda *a: matches)
    calls = []

    def read(*a, **kw):
        calls.append(kw)
        return {}

    monkeypatch.setattr(src.rtc, "_load_runtime_snapshot_rows", read)
    kwargs = dict(project_root=tmp_path, lookback_days=14,
                  mode_allowlist=[], symbol_allowlist=[], deadline_monotonic=123.0)
    summary = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "lookback_days": 14}
    if seed:
        result = src._seeded_snapshot_sequences(summary, seed_health_path=tmp_path / "health", **kwargs)
    else:
        result = src._incremental_snapshot_sequences(summary, health_path=tmp_path / "health", prefer_sqlite=True, **kwargs)
    assert result is None
    assert len(calls) == int(matches)
    if matches:
        assert calls[0]["deadline_monotonic"] == 123.0


def test_generation_retention_preserves_previous_recent_and_foreign_files(tmp_path):
    rows = tmp_path / "rows.jsonl"
    current = src._generation_rows_path(rows)
    previous = current.with_name("a" * 32 + ".jsonl")
    expired = current.with_name("b" * 32 + ".jsonl")
    recent = current.with_name("c" * 32 + ".jsonl")
    interrupted = current.with_name("." + "d" * 32 + ".jsonl.building")
    foreign = current.with_name("unrelated.jsonl")
    for path in (current, previous, expired, recent, foreign, interrupted):
        path.write_text("{}\n")
    for path in (previous, expired, foreign, interrupted):
        src.os.utime(path, (1, 1))
    src._finish_generation_publication(rows, current, {"rows_path": str(previous)})
    assert not expired.exists()
    assert interrupted.exists()  # Scratch has its own idle-verified recovery path.
    assert all(path.exists() for path in (current, previous, recent, foreign))


def test_generation_directory_cannot_be_a_symlink(tmp_path):
    target = tmp_path / "unrelated"
    target.mkdir()
    (tmp_path / ".rows.generations").symlink_to(target)
    with pytest.raises(ValueError, match="route"):
        src._generation_rows_path(tmp_path / "rows.jsonl")
    assert not list(target.iterdir())


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
