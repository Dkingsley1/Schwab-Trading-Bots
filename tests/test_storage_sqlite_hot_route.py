from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.ops import storage_sqlite_hot_route as src


def test_legacy_final_manifest_paths_are_checksum_verified_and_repaired(tmp_path: Path) -> None:
    dataset = tmp_path / "archive.parquet"
    dataset.mkdir()
    part = dataset / "part_000000000001_000000000010.parquet"
    part.write_bytes(b"verified-parquet-part")
    manifest = dataset / "_manifest.json"
    legacy_path = tmp_path / ".archive.parquet.partial_dataset" / part.name
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "table": "jsonl_records",
                "parts": [
                    {
                        "start_id": 1,
                        "end_id": 11,
                        "rows_exported": 10,
                        "output_path": str(legacy_path),
                        "size_bytes": part.stat().st_size,
                        "sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    preview = src.repair_final_manifest_paths(manifest, apply=False)
    assert preview["ok"] is True
    assert preview["changed_path_count"] == 1
    assert json.loads(manifest.read_text(encoding="utf-8"))["parts"][0]["output_path"] == str(legacy_path)

    applied = src.repair_final_manifest_paths(manifest, apply=True)
    repaired = json.loads(manifest.read_text(encoding="utf-8"))
    assert applied["overall_status"] == "repaired"
    assert repaired["schema_version"] == 1
    assert repaired["status"] == "complete"
    assert repaired["dataset_root"] == str(dataset)
    assert repaired["parts"][0]["output_path"] == str(part)
    assert repaired["path_repair"]["validation"] == "canonical_path_size_and_sha256"


def test_legacy_final_manifest_repair_rejects_tampered_part(tmp_path: Path) -> None:
    dataset = tmp_path / "archive.parquet"
    dataset.mkdir()
    part = dataset / "part_000000000001_000000000010.parquet"
    part.write_bytes(b"tampered")
    manifest = dataset / "_manifest.json"
    original = {
        "schema_version": 1,
        "parts": [
            {
                "start_id": 1,
                "end_id": 11,
                "rows_exported": 10,
                "output_path": str(tmp_path / ".archive.parquet.partial_dataset" / part.name),
                "size_bytes": part.stat().st_size,
                "sha256": "0" * 64,
            }
        ],
    }
    manifest.write_text(json.dumps(original), encoding="utf-8")

    result = src.repair_final_manifest_paths(manifest, apply=True)

    assert result["ok"] is False
    assert any("sha256_mismatch" in error for error in result["errors"])
    assert json.loads(manifest.read_text(encoding="utf-8")) == original


def _seed_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE jsonl_records (id INTEGER PRIMARY KEY, ingested_at TEXT, payload_json TEXT, source_rel TEXT)"
        )
        conn.execute("CREATE TABLE merge_state (name TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO jsonl_records VALUES (1, '2020-01-01T00:00:00+00:00', '{}', 'old.jsonl')")
        conn.execute("INSERT INTO jsonl_records VALUES (2, '2999-01-01T00:00:00+00:00', '{}', 'new.jsonl')")
        conn.execute("INSERT INTO merge_state VALUES ('cursor', '2')")
        conn.commit()


def _fake_export(src_conn, *, table, cutoff, out_path, **kwargs):
    if table != "jsonl_records":
        return {"table": table, "rows_exported": 0, "output_path": "", "size_bytes": 0}
    rows = int(src_conn.execute("SELECT COUNT(*) FROM jsonl_records WHERE ingested_at < ?", (cutoff,)).fetchone()[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"parquet-receipt")
    return {"table": table, "rows_exported": rows, "output_path": str(out_path), "size_bytes": out_path.stat().st_size}


def test_local_cache_rebuild_exports_cold_rows_and_atomically_prunes_old_cache(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "local_fallback_storage" / "data" / "jsonl_link.sqlite3"
    _seed_db(source)
    repo_db = tmp_path / "data" / "jsonl_link.sqlite3"
    repo_db.parent.mkdir(parents=True)
    repo_db.symlink_to(source)
    external = tmp_path / "external"
    external.mkdir()
    monkeypatch.setattr(src.writer_state, "writer_state_snapshot", lambda project_root: {"active": False, "running": False})
    monkeypatch.setenv("BOT_LOGS_SQLITE_HOT_ROUTE_EXPORT_ENGINE", "pyarrow")
    monkeypatch.setattr(src, "_export_cold_table_to_parquet", _fake_export)

    payload = src.build_local_cache_payload(
        tmp_path,
        relative_path="data/jsonl_link.sqlite3",
        hot_hours=18,
        apply=True,
        prune_old_cache=True,
        min_local_free_after_gb=0,
        min_external_free_after_gb=0,
        cold_export_root=external / "cold",
        batch_size=1000,
        compression="zstd",
        require_writer_idle=True,
        timeout_seconds=10,
        external_root=external,
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "rebuilt_pruned"
    assert payload["coverage_check"]["ok"] is True
    assert repo_db.resolve() == source.resolve()
    with sqlite3.connect(source) as conn:
        assert conn.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()[0] == 1
        assert conn.execute("SELECT value FROM merge_state WHERE name='cursor'").fetchone()[0] == "2"
    assert not Path(payload["old_cache_db"]).exists()


def test_local_cache_rebuild_refuses_active_writer_without_touching_source(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "local_fallback_storage" / "data" / "jsonl_link.sqlite3"
    _seed_db(source)
    original = source.read_bytes()
    external = tmp_path / "external"
    external.mkdir()
    monkeypatch.setattr(src.writer_state, "writer_state_snapshot", lambda project_root: {"active": True, "running": True})

    payload = src.build_local_cache_payload(
        tmp_path,
        relative_path="data/jsonl_link.sqlite3",
        hot_hours=18,
        apply=True,
        prune_old_cache=True,
        min_local_free_after_gb=0,
        min_external_free_after_gb=0,
        cold_export_root=external / "cold",
        batch_size=1000,
        compression="zstd",
        require_writer_idle=True,
        timeout_seconds=10,
        external_root=external,
    )

    assert payload["ok"] is False
    assert payload["overall_status"] == "blocked"
    assert "writer_not_idle" in payload["blockers"]
    assert source.read_bytes() == original


def test_local_cache_rebuild_accepts_orphaned_running_progress(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "local_fallback_storage" / "data" / "jsonl_link.sqlite3"
    _seed_db(source)
    external = tmp_path / "external"
    external.mkdir()
    monkeypatch.setattr(
        src.writer_state,
        "writer_state_snapshot",
        lambda project_root: {
            "active": False,
            "running": True,
            "progress_orphaned": True,
            "writer_lock_held": False,
            "child_writer_active": False,
        },
    )

    payload = src.build_local_cache_payload(
        tmp_path,
        relative_path="data/jsonl_link.sqlite3",
        hot_hours=18,
        apply=False,
        prune_old_cache=True,
        min_local_free_after_gb=0,
        min_external_free_after_gb=0,
        cold_export_root=external / "cold",
        batch_size=1000,
        compression="zstd",
        require_writer_idle=True,
        timeout_seconds=10,
        external_root=external,
    )

    assert payload["writer_idle"] is True
    assert "writer_not_idle" not in payload["blockers"]
    assert payload["ok"] is True


def test_cold_export_caps_batches_and_releases_arrow_memory(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "source.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE jsonl_records (id INTEGER, ingested_at TEXT, payload_json TEXT)")
        conn.executemany(
            "INSERT INTO jsonl_records VALUES (?, ?, ?)",
            [(idx, "2020-01-01T00:00:00+00:00", "x" * 32) for idx in range(205)],
        )
        conn.commit()
    monkeypatch.setenv("BOT_LOGS_SQLITE_HOT_ROUTE_MAX_EXPORT_BATCH_ROWS", "2")

    with src._connect(db_path, readonly=True) as conn:
        payload = src._export_cold_table_to_parquet(
            conn,
            table="jsonl_records",
            cutoff="2021-01-01T00:00:00+00:00",
            out_path=tmp_path / "cold.parquet",
            batch_size=50000,
            compression="zstd",
        )

    assert payload["rows_exported"] == 205
    assert payload["effective_batch_size"] == 100
    assert payload["batches_written"] == 3
    assert (tmp_path / "cold.parquet").is_file()


def test_duckdb_cold_export_is_row_exact_and_bounded(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "source.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE jsonl_records (id INTEGER, ingested_at TEXT, payload_json TEXT)")
        conn.executemany(
            "INSERT INTO jsonl_records VALUES (?, ?, ?)",
            [(1, "2020-01-01T00:00:00+00:00", "old"), (2, "2999-01-01T00:00:00+00:00", "new")],
        )
        conn.commit()
    monkeypatch.setenv("BOT_LOGS_SQLITE_DUCKDB_EXPORT_MEMORY_LIMIT", "128MB")
    monkeypatch.setenv("BOT_LOGS_SQLITE_DUCKDB_EXPORT_THREADS", "1")
    monkeypatch.setenv("BOT_LOGS_SQLITE_DUCKDB_EXPORT_ROW_GROUP_SIZE", "2048")

    payload = src._export_cold_table_to_parquet_duckdb(
        db_path,
        table="jsonl_records",
        cutoff="2021-01-01T00:00:00+00:00",
        out_path=tmp_path / "cold.parquet",
        compression="zstd",
        free_guard_root=tmp_path,
        min_free_after_bytes=0,
    )

    assert payload["rows_exported"] == 1
    assert payload["engine"] == "duckdb_partitioned"
    assert payload["memory_limit"] == "128MB"
    assert payload["threads"] == 1
    assert payload["row_group_size"] == 2048
    assert payload["part_count"] == 1
    assert (tmp_path / "cold.parquet").is_dir()


def test_isolated_arrow_export_releases_workers_and_writes_manifest(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "source.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE jsonl_records (id INTEGER PRIMARY KEY, ingested_at TEXT, payload_json TEXT)")
        conn.executemany(
            "INSERT INTO jsonl_records VALUES (?, ?, ?)",
            [(1, "2020-01-01T00:00:00+00:00", "old"), (2, "2999-01-01T00:00:00+00:00", "new")],
        )
        conn.commit()
    monkeypatch.setenv("BOT_LOGS_SQLITE_ARROW_WORKER_ID_SPAN", "1000")
    monkeypatch.setenv("BOT_LOGS_SQLITE_ARROW_WORKER_MIN_ID_SPAN", "100")
    monkeypatch.setenv("BOT_LOGS_SQLITE_ARROW_WORKER_BATCH_ROWS", "100")

    payload = src._export_cold_table_to_parquet_isolated(
        db_path,
        table="jsonl_records",
        cutoff="2021-01-01T00:00:00+00:00",
        out_path=tmp_path / "cold.parquet",
        compression="zstd",
        free_guard_root=tmp_path,
        min_free_after_bytes=0,
    )

    assert payload["rows_exported"] == 1
    assert payload["engine"] == "isolated_pyarrow"
    assert payload["part_count"] == 1
    assert payload["max_worker_rss_bytes"] > 0
    manifest_path = tmp_path / "cold.parquet" / "_manifest.json"
    assert manifest_path.is_file()
    manifest = src._load_json(manifest_path)
    assert manifest["status"] == "complete"
    assert manifest["schema_version"] == 2
    assert ".partial_dataset" not in manifest["parts"][0]["output_path"]


def test_quick_check_closes_helper_connection(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "source.sqlite3"
    db_path.touch()

    class _Connection:
        closed = False

        def execute(self, _sql: str):
            return self

        def fetchone(self):
            return ("ok",)

        def close(self):
            self.closed = True

    connection = _Connection()
    monkeypatch.setattr(src, "_connect", lambda *args, **kwargs: connection)

    assert src._quick_check(db_path, timeout_seconds=1) == {"ok": True, "result": "ok"}
    assert connection.closed is True


def test_isolated_arrow_export_resumes_verified_checkpoint(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "source.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE jsonl_records (id INTEGER PRIMARY KEY, ingested_at TEXT, payload_json TEXT)")
        conn.executemany(
            "INSERT INTO jsonl_records VALUES (?, ?, ?)",
            [
                (1, "2020-01-01T00:00:00+00:00", "first"),
                (2001, "2020-01-01T00:00:00+00:00", "last"),
            ],
        )
        conn.commit()
    monkeypatch.setenv("BOT_LOGS_SQLITE_ARROW_WORKER_ID_SPAN", "1000")
    monkeypatch.setenv("BOT_LOGS_SQLITE_ARROW_WORKER_MIN_ID_SPAN", "100")
    monkeypatch.setenv("BOT_LOGS_SQLITE_ARROW_WORKER_BATCH_ROWS", "100")
    real_run = src.subprocess.run
    calls = {"count": 0}

    def _interrupt_second_worker(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise KeyboardInterrupt()
        return real_run(*args, **kwargs)

    monkeypatch.setattr(src.subprocess, "run", _interrupt_second_worker)
    out_path = tmp_path / "cold.parquet"
    with pytest.raises(KeyboardInterrupt):
        src._export_cold_table_to_parquet_isolated(
            db_path,
            table="jsonl_records",
            cutoff="2021-01-01T00:00:00+00:00",
            out_path=out_path,
            compression="zstd",
            free_guard_root=tmp_path,
            min_free_after_bytes=0,
        )

    partial_root = tmp_path / ".cold.parquet.partial_dataset"
    checkpoint = src._load_json(partial_root / "_checkpoint.json")
    assert checkpoint["status"] == "interrupted"
    assert checkpoint["range_count"] == 1

    monkeypatch.setattr(src.subprocess, "run", real_run)
    payload = src._export_cold_table_to_parquet_isolated(
        db_path,
        table="jsonl_records",
        cutoff="2021-01-01T00:00:00+00:00",
        out_path=out_path,
        compression="zstd",
        free_guard_root=tmp_path,
        min_free_after_bytes=0,
    )

    assert payload["rows_exported"] == 2
    assert payload["resumed_range_count"] == 1
    assert payload["range_count"] == 3
    assert not partial_root.exists()
    final_manifest = src._load_json(out_path / "_manifest.json")
    assert final_manifest["status"] == "complete"
    assert all(".partial_dataset" not in row["output_path"] for row in final_manifest["parts"])
