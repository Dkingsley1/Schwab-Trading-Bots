import gzip
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.build_runtime_training_snapshot as snapshot_builder
import scripts.run_cached_collector as cached_collector


def test_snapshot_builder_hashes_files_without_read_bytes(tmp_path, monkeypatch) -> None:
    target = tmp_path / "snapshot.jsonl"
    payload = (b"alpha\nbeta\n" * 1024) + b"omega\n"
    target.write_bytes(payload)

    def _fail_read_bytes(self: Path) -> bytes:
        raise AssertionError("streaming hash should not use Path.read_bytes")

    monkeypatch.setattr(Path, "read_bytes", _fail_read_bytes)

    assert snapshot_builder._sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_cached_collector_hashes_files_without_read_bytes(tmp_path, monkeypatch) -> None:
    target = tmp_path / "collector.json"
    payload = (b'{"ok":true}\n' * 1024) + b'{"ok":false}\n'
    target.write_bytes(payload)

    def _fail_read_bytes(self: Path) -> bytes:
        raise AssertionError("streaming hash should not use Path.read_bytes")

    monkeypatch.setattr(Path, "read_bytes", _fail_read_bytes)

    assert cached_collector._sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_snapshot_builder_reuses_fresh_compatible_snapshot(tmp_path) -> None:
    rows_path = tmp_path / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text('{"mode":"shadow_aggressive_equities","symbol":"SPY"}\n', encoding="utf-8")
    health_path = tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "project_root": str(tmp_path),
                "lookback_days": 14,
                "mode_allowlist": [],
                "symbol_allowlist": [],
                "prefer_sqlite": True,
                "rows_path": str(rows_path),
                "sequence_count": 1,
                "row_count": 1,
            }
        ),
        encoding="utf-8",
    )

    payload = snapshot_builder._reusable_snapshot_payload(
        snapshot_builder._load_json(health_path),
        project_root=tmp_path,
        lookback_days=14,
        mode_allowlist=[],
        symbol_allowlist=[],
        prefer_sqlite=True,
        max_age_minutes=60,
    )

    assert payload["reused"] is True
    assert payload["reuse_reason"] == "fresh_compatible_snapshot"
    assert payload["row_count"] == 1


def test_incremental_candidate_paths_prefer_current_runtime_variant(tmp_path, monkeypatch) -> None:
    runtime_dir = tmp_path / "decision_explanations" / "shadow_crypto"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    plain = runtime_dir / "decision_explanations_20260413.jsonl"
    gz_path = runtime_dir / "decision_explanations_20260413.jsonl.gz"
    plain.write_text("{}", encoding="utf-8")
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write("{}\n")

    monkeypatch.setattr(snapshot_builder.rtc, "_recent_decision_paths", lambda *_args, **_kwargs: [gz_path, plain])

    paths = snapshot_builder._incremental_candidate_paths(
        tmp_path,
        lookback_days=14,
        since_utc=datetime(2026, 4, 13, 0, 0, tzinfo=timezone.utc),
    )

    assert paths == [plain]


def test_incremental_snapshot_sequences_merge_new_runtime_rows(tmp_path, monkeypatch) -> None:
    base_ts = datetime.now(timezone.utc) - timedelta(minutes=30)
    rows_path = tmp_path / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(
        json.dumps(
            {
                "mode": "shadow_crypto",
                "symbol": "BTC-USD",
                "strategy": "grand_master_bot",
                "strategy_priority": 0,
                "snapshot_id": "snap-base",
                "ts_epoch": float(base_ts.timestamp()),
                "timestamp_utc": base_ts.isoformat(),
                "price": 100.0,
                "features": {"last_price": 100.0, "pct_from_close": 0.01},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    health_path = tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp_utc": base_ts.isoformat(),
        "project_root": str(tmp_path),
        "lookback_days": 14,
        "mode_allowlist": [],
        "symbol_allowlist": [],
        "prefer_sqlite": True,
        "rows_path": str(rows_path),
        "sequence_count": 1,
        "row_count": 1,
    }
    health_path.write_text(json.dumps(summary), encoding="utf-8")

    new_path = tmp_path / "decision_explanations" / "shadow_crypto" / "decision_explanations_20260413.jsonl"
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_ts = datetime.now(timezone.utc)
    new_path.write_text(
        json.dumps(
            {
                "timestamp_utc": new_ts.isoformat(),
                "mode": "shadow_crypto",
                "symbol": "BTC-USD",
                "strategy": "grand_master_bot",
                "features": {"last_price": 101.0, "pct_from_close": 0.02},
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-new"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(snapshot_builder.rtc, "_recent_decision_paths", lambda *_args, **_kwargs: [new_path])
    monkeypatch.setattr(snapshot_builder.rtc, "_load_runtime_gap_fill_context", lambda *_args, **_kwargs: {})

    incremental = snapshot_builder._incremental_snapshot_sequences(
        summary,
        project_root=tmp_path,
        health_path=health_path,
        lookback_days=2,
        mode_allowlist=[],
        symbol_allowlist=[],
        prefer_sqlite=True,
    )

    assert incremental is not None
    sequences, meta = incremental
    rows = sequences[("shadow_crypto", "BTC-USD")]
    assert [row["snapshot_id"] for row in rows] == ["snap-base", "snap-new"]
    assert meta["build_mode"] == "incremental_refresh"
    assert meta["incremental_row_count"] == 1


def test_incremental_snapshot_sequences_marks_partial_when_candidate_row_budget_hits(tmp_path, monkeypatch) -> None:
    base_ts = datetime.now(timezone.utc) - timedelta(minutes=30)
    rows_path = tmp_path / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(
        json.dumps(
            {
                "mode": "shadow_crypto",
                "symbol": "BTC-USD",
                "strategy": "grand_master_bot",
                "strategy_priority": 0,
                "snapshot_id": "snap-base",
                "ts_epoch": float(base_ts.timestamp()),
                "timestamp_utc": base_ts.isoformat(),
                "price": 100.0,
                "features": {"last_price": 100.0, "pct_from_close": 0.01},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    health_path = tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp_utc": base_ts.isoformat(),
        "project_root": str(tmp_path),
        "lookback_days": 14,
        "mode_allowlist": [],
        "symbol_allowlist": [],
        "prefer_sqlite": True,
        "rows_path": str(rows_path),
        "sequence_count": 1,
        "row_count": 1,
    }
    health_path.write_text(json.dumps(summary), encoding="utf-8")

    new_path = tmp_path / "decision_explanations" / "shadow_crypto" / "decision_explanations_20260413.jsonl"
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_ts = datetime.now(timezone.utc)
    new_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp_utc": new_ts.isoformat(),
                        "mode": "shadow_crypto",
                        "symbol": "BTC-USD",
                        "strategy": "grand_master_bot",
                        "features": {"last_price": 101.0, "pct_from_close": 0.02},
                        "metadata": {"layer": "grand_master", "snapshot_id": "snap-new-1"},
                    }
                ),
                json.dumps(
                    {
                        "timestamp_utc": (new_ts + timedelta(seconds=1)).isoformat(),
                        "mode": "shadow_crypto",
                        "symbol": "BTC-USD",
                        "strategy": "grand_master_bot",
                        "features": {"last_price": 102.0, "pct_from_close": 0.03},
                        "metadata": {"layer": "grand_master", "snapshot_id": "snap-new-2"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(snapshot_builder.rtc, "_recent_decision_paths", lambda *_args, **_kwargs: [new_path])
    monkeypatch.setattr(snapshot_builder.rtc, "_load_runtime_gap_fill_context", lambda *_args, **_kwargs: {})

    incremental = snapshot_builder._incremental_snapshot_sequences(
        summary,
        project_root=tmp_path,
        health_path=health_path,
        lookback_days=2,
        mode_allowlist=[],
        symbol_allowlist=[],
        prefer_sqlite=True,
        max_candidate_rows=1,
    )

    assert incremental is not None
    sequences, meta = incremental
    rows = sequences[("shadow_crypto", "BTC-USD")]
    assert [row["snapshot_id"] for row in rows] == ["snap-base", "snap-new-1"]
    assert meta["incremental_partial"] is True
    assert meta["incremental_scan"]["candidate_scan_row_limit_hit"] is True
    assert meta["incremental_scan"]["candidate_json_row_count"] == 1


def test_incremental_snapshot_sequences_recovers_split_channel_price(tmp_path, monkeypatch) -> None:
    base_ts = datetime.now(timezone.utc) - timedelta(minutes=30)
    rows_path = tmp_path / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(
        json.dumps(
            {
                "mode": "shadow_crypto",
                "symbol": "BTC-USD",
                "strategy": "grand_master_bot",
                "strategy_priority": 0,
                "snapshot_id": "snap-base",
                "ts_epoch": float(base_ts.timestamp()),
                "timestamp_utc": base_ts.isoformat(),
                "price": 100.0,
                "features": {"last_price": 100.0, "pct_from_close": 0.01},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    health_path = tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp_utc": base_ts.isoformat(),
        "project_root": str(tmp_path),
        "lookback_days": 14,
        "mode_allowlist": [],
        "symbol_allowlist": [],
        "prefer_sqlite": True,
        "rows_path": str(rows_path),
        "sequence_count": 1,
        "row_count": 1,
    }
    health_path.write_text(json.dumps(summary), encoding="utf-8")

    new_ts = datetime.now(timezone.utc)
    day = new_ts.strftime("%Y%m%d")
    market_path = (
        tmp_path
        / "governance"
        / "channels"
        / "decision"
        / "crypto_shadow"
        / f"decision_{day}.jsonl"
    )
    market_path.parent.mkdir(parents=True, exist_ok=True)
    market_path.write_text(
        json.dumps(
            {
                "timestamp_utc": (new_ts - timedelta(seconds=10)).isoformat(),
                "symbol": "BTC-USD",
                "snapshot_id": "snap-new",
                "market": {"last_price": 101.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    new_path = tmp_path / "decision_explanations" / "shadow_crypto" / f"decision_explanations_{day}.jsonl"
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_text(
        json.dumps(
            {
                "timestamp_utc": new_ts.isoformat(),
                "mode": "shadow_crypto",
                "symbol": "BTC-USD",
                "strategy": "grand_master_bot",
                "features": {"pct_from_close": 0.02},
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-new"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(snapshot_builder.rtc, "_recent_decision_paths", lambda *_args, **_kwargs: [new_path, market_path])
    monkeypatch.setattr(snapshot_builder.rtc, "_load_runtime_gap_fill_context", lambda *_args, **_kwargs: {})

    incremental = snapshot_builder._incremental_snapshot_sequences(
        summary,
        project_root=tmp_path,
        health_path=health_path,
        lookback_days=2,
        mode_allowlist=[],
        symbol_allowlist=[],
        prefer_sqlite=True,
    )

    assert incremental is not None
    sequences, meta = incremental
    rows = sequences[("shadow_crypto", "BTC-USD")]
    assert [row["snapshot_id"] for row in rows] == ["snap-base", "snap-new"]
    assert rows[-1]["price"] == 101.0
    assert rows[-1]["features"]["last_price"] == 101.0
    assert rows[-1]["features"]["price_recovered_from_sidecar"] == 1.0
    assert meta["incremental_row_count"] == 1


def test_seeded_snapshot_sequences_backfills_older_rows_from_global_snapshot(tmp_path, monkeypatch) -> None:
    base_ts = datetime.now(timezone.utc) - timedelta(days=2)
    old_ts = datetime.now(timezone.utc) - timedelta(days=30)
    rows_path = tmp_path / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(
        json.dumps(
            {
                "mode": "shadow_crypto",
                "symbol": "BTC-USD",
                "strategy": "grand_master_bot",
                "strategy_priority": 0,
                "snapshot_id": "snap-base",
                "ts_epoch": float(base_ts.timestamp()),
                "timestamp_utc": base_ts.isoformat(),
                "price": 100.0,
                "features": {"last_price": 100.0, "pct_from_close": 0.01},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seed_health_path = tmp_path / "governance" / "health" / "runtime_training_snapshot_latest.json"
    seed_health_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(tmp_path),
        "lookback_days": 14,
        "mode_allowlist": [],
        "symbol_allowlist": [],
        "prefer_sqlite": True,
        "rows_path": str(rows_path),
        "sequence_count": 1,
        "row_count": 1,
    }
    seed_health_path.write_text(json.dumps(summary), encoding="utf-8")

    old_path = tmp_path / "decision_explanations" / "shadow_crypto" / f"decision_explanations_{old_ts.strftime('%Y%m%d')}.jsonl"
    old_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.write_text(
        json.dumps(
            {
                "timestamp_utc": old_ts.isoformat(),
                "mode": "shadow_crypto",
                "symbol": "BTC-USD",
                "strategy": "grand_master_bot",
                "features": {"last_price": 95.0, "pct_from_close": -0.03},
                "metadata": {"layer": "grand_master", "snapshot_id": "snap-old"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(snapshot_builder.rtc, "_recent_decision_paths", lambda *_args, **_kwargs: [old_path])
    monkeypatch.setattr(snapshot_builder.rtc, "_load_runtime_gap_fill_context", lambda *_args, **_kwargs: {})

    seeded = snapshot_builder._seeded_snapshot_sequences(
        summary,
        seed_health_path=seed_health_path,
        project_root=tmp_path,
        lookback_days=60,
        mode_allowlist=[],
        symbol_allowlist=[],
    )

    assert seeded is not None
    sequences, meta = seeded
    rows = sequences[("shadow_crypto", "BTC-USD")]
    assert [row["snapshot_id"] for row in rows] == ["snap-old", "snap-base"]
    assert meta["build_mode"] == "seed_backfill_refresh"
    assert meta["seed_base_lookback_days"] == 14
    assert meta["seed_backfill_row_count"] == 1
