import json
import os
from datetime import datetime, timezone
from pathlib import Path

from core import runtime_training_common as runtime_common
from scripts.ops import hdf5_training_cache as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_snapshot(project_root: Path) -> tuple[Path, Path]:
    rows_path = project_root / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp_utc": "2026-06-19T10:00:00+00:00",
            "mode": "shadow_equities",
            "symbol": "AAPL",
            "price": 100.0,
            "features": {"last_price": 100.0, "pct_from_close": 0.01, "embedding_0": 0.2},
        },
        {
            "timestamp_utc": "2026-06-19T10:01:00+00:00",
            "mode": "shadow_equities",
            "symbol": "AAPL",
            "price": 101.0,
            "features": {"last_price": 101.0, "pct_from_close": 0.02, "embedding_0": 0.3},
        },
        {
            "timestamp_utc": "2026-06-19T10:02:00+00:00",
            "mode": "shadow_equities",
            "symbol": "AAPL",
            "price": 102.0,
            "features": {"last_price": 102.0, "pct_from_close": 0.03, "embedding_0": 0.4},
        },
    ]
    rows_path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")
    health_path = project_root / "governance" / "health" / "runtime_training_snapshot_latest.json"
    _write_json(
        health_path,
        {
            "timestamp_utc": "2026-06-19T10:05:00+00:00",
            "rows_path": str(rows_path),
            "row_count": len(rows),
            "sequence_count": 1,
            "rows_sha256": src._sha256_file(rows_path),
        },
    )
    return health_path, rows_path


def test_hdf5_training_cache_blocks_cleanly_when_h5py_missing(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health_path, _rows_path = _write_snapshot(project_root)
    out_path = project_root / "data" / "hdf5" / "runtime_training_snapshot_latest.h5"
    monkeypatch.setattr(src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(src, "_import_h5py", lambda: (None, "ModuleNotFoundError:No module named h5py"))

    payload = src.evaluate(
        apply=True,
        snapshot_health_path=health_path,
        rows_path_arg=None,
        out_path=out_path,
        health_path=project_root / "governance" / "health" / "hdf5_training_cache_latest.json",
        max_rows=100,
    )

    assert payload["overall_status"] == "blocked"
    assert payload["blockers"] == ["h5py_missing"]
    assert payload["hdf5_dependency"]["available"] is False
    assert out_path.exists() is False
    assert set(payload["lanes"]) == set(src.LANES)
    assert payload["local_only_contract"]["hdf5_role"] == "derived_training_cache"


def test_hdf5_training_cache_plans_all_six_lanes_with_source_hashes(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health_path, rows_path = _write_snapshot(project_root)
    out_path = project_root / "data" / "hdf5" / "runtime_training_snapshot_latest.h5"
    monkeypatch.setattr(src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(src, "_import_h5py", lambda: (object(), ""))

    payload = src.evaluate(
        apply=False,
        snapshot_health_path=health_path,
        rows_path_arg=None,
        out_path=out_path,
        health_path=project_root / "governance" / "health" / "hdf5_training_cache_latest.json",
        max_rows=100,
    )

    assert payload["overall_status"] == "planned"
    assert payload["cache"]["row_count"] == 3
    assert payload["cache"]["feature_count"] == 3
    assert payload["lanes"]["training_feature_matrices"]["status"] == "ready"
    assert payload["lanes"]["model_input_tensors"]["status"] == "ready"
    assert payload["lanes"]["embeddings"]["status"] == "ready"
    assert payload["lanes"]["walk_forward_datasets"]["status"] == "ready"
    assert payload["lanes"]["immutable_research_snapshots"]["status"] == "ready"
    assert payload["lanes"]["compressed_local_archives"]["status"] == "ready"
    assert payload["source_snapshot"]["rows_sha256"] == src._sha256_file(rows_path)
    assert payload["local_only_contract"]["icloud_safe"] is True


def test_hdf5_training_cache_apply_writes_ready_manifest_and_sidecar(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health_path, _rows_path = _write_snapshot(project_root)
    out_path = project_root / "data" / "hdf5" / "runtime_training_snapshot_latest.h5"
    monkeypatch.setattr(src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(src, "_import_h5py", lambda: (object(), ""))

    def fake_write_hdf5(*, h5py, out_path, rows, rows_summary, manifest):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake-hdf5")

    monkeypatch.setattr(src, "_write_hdf5", fake_write_hdf5)

    payload = src.evaluate(
        apply=True,
        snapshot_health_path=health_path,
        rows_path_arg=None,
        out_path=out_path,
        health_path=project_root / "governance" / "health" / "hdf5_training_cache_latest.json",
        max_rows=100,
    )

    sidecar = out_path.with_suffix(out_path.suffix + ".manifest.json")
    assert payload["overall_status"] == "ready"
    assert out_path.exists()
    assert sidecar.exists()
    assert payload["cache"]["h5_sha256"] == src._sha256_file(out_path)
    assert payload["cache"]["sidecar_manifest_sha256"] == src._sha256_file(sidecar)

    status_payload = src.evaluate(
        apply=False,
        snapshot_health_path=health_path,
        rows_path_arg=None,
        out_path=out_path,
        health_path=project_root / "governance" / "health" / "hdf5_training_cache_latest.json",
        max_rows=100,
    )
    assert status_payload["overall_status"] == "ready"
    assert status_payload["cache"]["existing_cache_reused"] is True


def test_hdf5_training_cache_blocks_when_writer_lock_is_owned(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health_path, _rows_path = _write_snapshot(project_root)
    out_path = project_root / "data" / "hdf5" / "runtime_training_snapshot_latest.h5"
    lock_path = project_root / "governance" / "locks" / "hdf5_training_cache.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"timestamp_utc": src._now_iso(), "pid": os.getpid()}, ensure_ascii=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(src, "_import_h5py", lambda: (object(), ""))

    def fail_write(**_kwargs):
        raise AssertionError("busy writer lock should prevent writes")

    monkeypatch.setattr(src, "_write_hdf5", fail_write)

    payload = src.evaluate(
        apply=True,
        snapshot_health_path=health_path,
        rows_path_arg=None,
        out_path=out_path,
        health_path=project_root / "governance" / "health" / "hdf5_training_cache_latest.json",
        max_rows=100,
        lock_path=lock_path,
        lock_stale_seconds=3600,
    )

    assert payload["overall_status"] == "blocked"
    assert payload["blockers"] == ["hdf5_cache_writer_busy"]
    assert payload["single_writer_guard"]["status"] == "busy"
    assert out_path.exists() is False


def test_hdf5_retention_deletes_only_generated_unpinned_caches(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setattr(src, "PROJECT_ROOT", project_root)
    root = project_root / "data" / "hdf5"
    root.mkdir(parents=True)
    current = root / "runtime_training_snapshot_latest.h5"
    current.write_bytes(b"current")
    user_file = root / "manual_research_notes.h5"
    user_file.write_bytes(b"user")
    pinned = root / "research_snapshot_keep.h5"
    pinned.write_bytes(b"pinned")
    pinned.with_suffix(pinned.suffix + ".manifest.json").write_text(
        json.dumps({"local_only_contract": {"hdf5_role": "derived_training_cache"}}),
        encoding="utf-8",
    )
    generated_paths = []
    for idx in range(3):
        path = root / f"runtime_training_snapshot_old_{idx}.h5"
        path.write_bytes(f"old-{idx}".encode("utf-8"))
        path.with_suffix(path.suffix + ".manifest.json").write_text(
            json.dumps({"local_only_contract": {"hdf5_role": "derived_training_cache"}}),
            encoding="utf-8",
        )
        os.utime(path, (100 + idx, 100 + idx))
        generated_paths.append(path)

    payload = src._retention_plan(out_path=current, keep_generated=1, apply=True)

    assert payload["status"] == "applied"
    assert generated_paths[2].exists()
    assert generated_paths[0].exists() is False
    assert generated_paths[1].exists() is False
    assert pinned.exists()
    assert user_file.exists()


def test_runtime_loader_prefers_fresh_hdf5_snapshot_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_TRAIN_USE_SNAPSHOT", "1")
    monkeypatch.setenv("RUNTIME_TRAIN_USE_HDF5_CACHE", "1")
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ts_epoch": 1.0,
        "snapshot_id": "h5-row-1",
        "mode": "shadow_equities",
        "symbol": "AAPL",
        "price": 100.0,
        "features": {"last_price": 100.0},
    }

    monkeypatch.setattr(
        runtime_common,
        "_load_hdf5_snapshot_rows",
        lambda *_args, **_kwargs: {("shadow_equities", "AAPL"): [row]},
    )

    def fail_jsonl(*_args, **_kwargs):
        raise AssertionError("fresh hdf5 cache should be preferred before JSONL snapshot fallback")

    monkeypatch.setattr(runtime_common, "_load_runtime_snapshot_rows", fail_jsonl)

    payload = runtime_common.load_runtime_observation_sequences(
        tmp_path,
        lookback_days=1,
        mode_allowlist=["shadow_equities"],
        symbol_allowlist=["AAPL"],
    )

    assert ("shadow_equities", "AAPL") in payload
    assert payload[("shadow_equities", "AAPL")][0]["snapshot_id"] == "h5-row-1"
