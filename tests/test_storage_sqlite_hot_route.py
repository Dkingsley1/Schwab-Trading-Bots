from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from scripts.ops import storage_sqlite_hot_route as src


@pytest.fixture(autouse=True)
def sufficient_test_capacity(monkeypatch):
    monkeypatch.setattr(src, "_disk_free_bytes", lambda _: 512 * 1024**3)


@pytest.fixture(params=["msgpack", "json_fallback"])
def restore_encoder(request, monkeypatch):
    if request.param == "json_fallback":
        monkeypatch.setitem(sys.modules, "msgpack", None)
    else:
        msgpack = pytest.importorskip("msgpack")
        if msgpack.Packer.__module__ != "msgpack._cmsgpack":
            pytest.skip("optional native encoder is unavailable")
    encode, encoding = src._restore_row_encoder()
    assert encoding == (
        "msgpack_sqlite_scalars_v1"
        if request.param == "msgpack"
        else "json_type_pairs_ascii_v1"
    )
    return encode


@pytest.mark.parametrize(
    "left,right",
    [
        ([1], [1.0]),
        ([1], [True]),
        ([0], [False]),
        (["1"], [1]),
        ([b"text"], ["text"]),
        ([None], [""]),
        ([-0.0], [0.0]),
        (["ab", "c"], ["a", "bc"]),
        ([1, 2], [12]),
        (["a\x00b"], ["ab"]),
    ],
)
def test_restore_encoding_preserves_types_and_boundaries(restore_encoder, left, right):
    assert restore_encoder(left) != restore_encoder(right)


def test_restore_encoding_is_repeatable_and_identity_independent(restore_encoder):
    text = "same value " * 100
    values = [
        None,
        -(2**63),
        2**63 - 1,
        1.25,
        -0.0,
        b"\x00\xff",
        "\u2603\U0001f600\x00",
        text,
        text,
    ]
    copied = list(values)
    copied[-1] = text.encode().decode()
    assert restore_encoder(values) == restore_encoder(tuple(copied))
    assert restore_encoder(values) == restore_encoder(values)


def test_restore_json_fallback_keeps_existing_digest_encoding(monkeypatch):
    monkeypatch.setitem(sys.modules, "msgpack", None)
    encode, encoding = src._restore_row_encoder()
    assert encoding == "json_type_pairs_ascii_v1"
    assert encode([None, 1, b"a"]) == b'[["NoneType",null],["int",1],["bytes","61"]]\n'


def test_restore_without_native_extension_keeps_json_fallback(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "msgpack", SimpleNamespace(Packer=object))
    assert src._restore_row_encoder()[1] == "json_type_pairs_ascii_v1"


@pytest.mark.parametrize(
    "change", [None, "binary_to_text", "float_to_int", "signed_zero"]
)
def test_full_sqlite_parquet_restore_preserves_scalar_types(
    tmp_path, restore_encoder, change
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "source.sqlite3"
    with sqlite3.connect(source) as conn:
        conn.execute(
            "CREATE TABLE jsonl_records(id INTEGER PRIMARY KEY, ingested_at TEXT, payload_json TEXT, binary_value BLOB, real_value REAL, zero_value REAL, precise_value REAL, nullable TEXT)"
        )
        conn.execute(
            "INSERT INTO jsonl_records VALUES(?,?,?,?,?,?,?,?)",
            (
                1,
                "2020-01-01",
                "\u2603\U0001f600\x00",
                b"a",
                1.0,
                0.0,
                1.0000000000000002,
                None,
            ),
        )
        cursor = conn.execute("SELECT * FROM jsonl_records")
        row = dict(
            zip(
                (description[0] for description in cursor.description),
                cursor.fetchone(),
            )
        )
    if change == "binary_to_text":
        row["binary_value"] = "a"
    elif change == "float_to_int":
        row["real_value"] = 1
    elif change == "signed_zero":
        row["zero_value"] = -0.0
    path = tmp_path / "export.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    export = {"table": "jsonl_records", "rows_exported": 1, "output_path": str(path)}
    if change:
        with pytest.raises(RuntimeError, match="cold_restore_content_mismatch"):
            src._verify_cold_export(source, export, "2021-01-01", 10)
    else:
        proof = src._verify_cold_export(source, export, "2021-01-01", 10)
        assert proof["ok"] and proof["rows_verified"] == 1
        assert proof["source_sha256"] == proof["restored_sha256"]


def test_source_inspection_uses_full_timestamp_index_and_bounded_extrema():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE jsonl_records(id INTEGER PRIMARY KEY, ingested_at TEXT, source_rel TEXT, payload_json TEXT)")
    conn.execute("CREATE INDEX wrong_prefix ON jsonl_records(source_rel, ingested_at)")
    conn.execute("CREATE INDEX partial_time ON jsonl_records(ingested_at) WHERE source_rel='one'")
    conn.execute("CREATE INDEX full_time ON jsonl_records(ingested_at)")
    conn.executemany("INSERT INTO jsonl_records VALUES(?,?,?,?)",
        ((i, '2020-01-01', 'one', '{}') for i in range(1, 5001)))
    conn.execute("INSERT INTO jsonl_records VALUES(5001,'2026-09-14','two','{}')")
    statements = []
    conn.set_trace_callback(statements.append)
    steps = []
    conn.set_progress_handler(lambda: (steps.append(1) or int(len(steps) > 5)), 1000)
    counts = src._source_counts(conn, '2026-01-01')
    estimate = src._estimated_hot_db_bytes(conn, '2026-01-01', 1024**3, counts)
    assert counts['jsonl_records'] == {'total_rows': 5001, 'hot_rows': 1, 'cold_rows': 5000,
        'min_ingested_at': '2020-01-01', 'max_ingested_at': '2026-09-14'}
    assert estimate == 64 * 1024**2
    selects = [q for q in statements if 'FROM "jsonl_records"' in q]
    assert all('INDEXED BY "full_time"' in q for q in selects)
    assert sum('COUNT(*)' in q for q in selects) == 2
    conn.close()


def test_hot_size_counts_encoded_bytes_without_loading_text_overflow():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE jsonl_records(ingested_at TEXT, payload_json TEXT)")
    conn.execute("CREATE INDEX full_time ON jsonl_records(ingested_at)")
    text = chr(0x1F600) * (8 * 1024**2)
    conn.execute("INSERT INTO jsonl_records VALUES('2026-09-14',?)", (text,))
    counts = {'jsonl_records': {'total_rows': 1, 'hot_rows': 1, 'cold_rows': 0}}
    assert src._estimated_hot_db_bytes(conn, '2026-01-01', 0, counts) == int(len(text.encode()) * 2.25)
    conn.close()


def test_source_inspection_handles_missing_index_empty_and_null_dates():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE jsonl_records(ingested_at TEXT, payload_json TEXT)")
    assert src._timestamp_indexed_table(conn, 'jsonl_records') == '"jsonl_records"'
    assert src._source_counts(conn, '2026-01-01')['jsonl_records']['total_rows'] == 0
    conn.execute("INSERT INTO jsonl_records VALUES(NULL,'{}')")
    assert src._source_counts(conn, '2026-01-01')['jsonl_records']['total_rows'] == 1
    conn.close()


def test_legacy_final_manifest_paths_are_checksum_verified_and_repaired(
    tmp_path: Path,
) -> None:
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
    assert json.loads(manifest.read_text(encoding="utf-8"))["parts"][0][
        "output_path"
    ] == str(legacy_path)

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
                "output_path": str(
                    tmp_path / ".archive.parquet.partial_dataset" / part.name
                ),
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
        conn.execute(
            "INSERT INTO jsonl_records VALUES (1, '2020-01-01T00:00:00+00:00', '{}', 'old.jsonl')"
        )
        conn.execute(
            "INSERT INTO jsonl_records VALUES (2, '2999-01-01T00:00:00+00:00', '{}', 'new.jsonl')"
        )
        conn.execute("INSERT INTO merge_state VALUES ('cursor', '2')")
        conn.commit()


def _fake_export(src_conn, *, table, cutoff, out_path, **kwargs):
    if table != "jsonl_records":
        return {"table": table, "rows_exported": 0, "output_path": "", "size_bytes": 0}
    rows = int(
        src_conn.execute(
            "SELECT COUNT(*) FROM jsonl_records WHERE ingested_at < ?", (cutoff,)
        ).fetchone()[0]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"parquet-receipt")
    return {
        "table": table,
        "rows_exported": rows,
        "output_path": str(out_path),
        "size_bytes": out_path.stat().st_size,
    }


def _rebuild_fixture(tmp_path, monkeypatch):
    source = tmp_path / "local_fallback_storage/data/jsonl_link.sqlite3"
    _seed_db(source)
    route = tmp_path / "data/jsonl_link.sqlite3"
    route.parent.mkdir(parents=True)
    route.symlink_to(source)
    external = tmp_path / "external"
    external.mkdir()
    monkeypatch.setattr(
        src.writer_state, "writer_state_snapshot", lambda _: {"active": False}
    )
    monkeypatch.setenv("BOT_LOGS_SQLITE_HOT_ROUTE_EXPORT_ENGINE", "pyarrow")
    return source, dict(
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


def test_cold_rebuild_rejects_count_only_receipt(tmp_path, monkeypatch):
    source, options = _rebuild_fixture(tmp_path, monkeypatch)
    original = source.read_bytes()
    monkeypatch.setattr(src, "_export_cold_table_to_parquet", _fake_export)
    result = src.build_local_cache_payload(tmp_path, **options)
    assert result["ok"] is False
    assert source.read_bytes() == original
    assert result["atomic_switch"] == {}


@pytest.mark.parametrize("corruption", ["payload", "duplicate", "missing", "schema"])
def test_cold_restore_checks_every_typed_row(tmp_path, monkeypatch, corruption, restore_encoder):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source, _ = _rebuild_fixture(tmp_path, monkeypatch)
    path = tmp_path / "export.parquet"
    row = {
        "id": 1,
        "ingested_at": "2020-01-01T00:00:00+00:00",
        "payload_json": "{}",
        "source_rel": "old.jsonl",
    }
    if corruption == "payload":
        row["payload_json"] = "[]"
    if corruption == "schema":
        row["unexpected"] = "value"
    rows = [row, row] if corruption == "duplicate" else [row]
    pq.write_table(pa.Table.from_pylist(rows), path)
    export = {"table": "jsonl_records", "rows_exported": 1, "output_path": str(path)}
    if corruption == "missing":
        path.unlink()
    with pytest.raises((RuntimeError, OSError)):
        src._verify_cold_export(source, export, "2021-01-01", 10)


def test_rebuild_preserves_source_changed_during_export(tmp_path, monkeypatch):
    source, options = _rebuild_fixture(tmp_path, monkeypatch)
    export = src._export_cold_table_to_parquet

    def mutate(*args, **kwargs):
        result = export(*args, **kwargs)
        with sqlite3.connect(source) as conn:
            conn.execute("UPDATE merge_state SET value='new-writer-cursor'")
        return result

    monkeypatch.setattr(src, "_export_cold_table_to_parquet", mutate)
    result = src.build_local_cache_payload(tmp_path, **options)
    assert result["ok"] is False
    assert "source_changed_during_cold_export" in str(result["blockers"])
    with sqlite3.connect(source) as conn:
        assert conn.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()[0] == 2
        assert (
            conn.execute("SELECT value FROM merge_state").fetchone()[0]
            == "new-writer-cursor"
        )


@pytest.mark.parametrize("lock_kind", ["storage", "writer"])
def test_rebuild_respects_kernel_owned_lock(tmp_path, monkeypatch, lock_kind):
    source, options = _rebuild_fixture(tmp_path, monkeypatch)
    original = source.read_bytes()
    path = (
        tmp_path / "governance/locks/storage_maintenance.lock"
        if lock_kind == "storage"
        else src.configured_sql_writer_lock_path(tmp_path)
    )
    with src._exclusive_lock(path):
        result = src.build_local_cache_payload(tmp_path, **options)
    assert result["ok"] is False
    assert "maintenance_lock_busy" in str(result["blockers"])
    assert source.read_bytes() == original
    assert path.exists()


def test_preview_reserves_external_hot_staging_allocation(tmp_path, monkeypatch):
    source, options = _rebuild_fixture(tmp_path, monkeypatch)
    options.update(apply=False, min_external_free_after_gb=64)
    monkeypatch.setattr(src, "_disk_free_bytes", lambda _: 65 * 1024**3)
    monkeypatch.setattr(src, "_inspect_source", lambda *args: (
        {}, 2 * 1024**3, {"hot_payload_bytes": 2 * 1024**3}))
    result = src.build_local_cache_payload(tmp_path, **options)
    assert "staging_capacity_below_payload" in result["blockers"]


def test_hot_build_does_not_analyze_or_modify_source(tmp_path):
    source = tmp_path / "source.sqlite3"
    _seed_db(source)
    original = source.read_bytes()
    result = src._build_hot_db(
        source_db=source,
        dest_tmp=tmp_path / "new.sqlite3",
        cutoff="2021-01-01",
        timeout_seconds=10,
    )
    assert result["quick_check"]["ok"] is True
    assert source.read_bytes() == original
    with sqlite3.connect(source) as conn:
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE name='sqlite_stat1'"
            ).fetchone()
            is None
        )


def test_hot_build_preserves_sequence_views_and_triggers(tmp_path):
    source = tmp_path / "source.sqlite3"
    with sqlite3.connect(source) as conn:
        conn.execute(
            "CREATE TABLE jsonl_records (id INTEGER PRIMARY KEY AUTOINCREMENT, ingested_at TEXT, payload_json TEXT)"
        )
        conn.execute("INSERT INTO jsonl_records VALUES (1, '2999-01-01', '{}')")
        conn.execute("INSERT INTO jsonl_records VALUES (100, '2020-01-01', '{}')")
        conn.execute("CREATE TABLE events (id INTEGER)")
        conn.execute("CREATE VIEW records_view AS SELECT * FROM jsonl_records")
        conn.execute(
            "CREATE TRIGGER records_insert AFTER INSERT ON jsonl_records BEGIN INSERT INTO events VALUES (NEW.id); END"
        )
    target = tmp_path / "hot.sqlite3"
    result = src._build_hot_db(
        source_db=source, dest_tmp=target, cutoff="2021-01-01", timeout_seconds=10
    )
    assert result["quick_check"]["ok"] is True
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT COUNT(*) FROM records_view").fetchone()[0] == 1
        conn.execute(
            "INSERT INTO jsonl_records (ingested_at, payload_json) VALUES ('2999-01-01', '{}')"
        )
        assert conn.execute("SELECT id FROM events").fetchone()[0] == 101


@pytest.mark.parametrize("payload_size", [16, 65536])
@pytest.mark.parametrize("hot_rows", [0, 3])
def test_hot_build_maintains_exact_indexes_during_copy(
    tmp_path, monkeypatch, payload_size, hot_rows
):
    source = tmp_path / "source.sqlite3"
    target = tmp_path / "hot.sqlite3"
    definitions = [
        "CREATE INDEX ordinary_time ON jsonl_records(ingested_at)",
        "CREATE INDEX \"wide expression\" ON jsonl_records(json_extract(payload_json, '$.symbol') COLLATE NOCASE DESC)",
        "CREATE INDEX tail_key ON jsonl_records(tail DESC, ingested_at)",
        "CREATE UNIQUE INDEX partial_unique ON jsonl_records(tail COLLATE NOCASE) WHERE id > 0 AND tail IS NOT NULL",
    ]
    with sqlite3.connect(source) as conn:
        conn.execute(
            "CREATE TABLE jsonl_records(id INTEGER PRIMARY KEY AUTOINCREMENT, ingested_at TEXT, payload_json TEXT, tail TEXT)"
        )
        for sql in definitions:
            conn.execute(sql)
        conn.executemany(
            "INSERT INTO jsonl_records VALUES(?,?,?,?)",
            (
                (
                    i,
                    "2999-01-01" if i <= hot_rows else "2020-01-01",
                    json.dumps({"symbol": f"sym{i}", "pad": "x" * payload_size}),
                    f"tail{i}",
                )
                for i in range(1, 5)
            ),
        )
        expected_schema = conn.execute(
            "SELECT name, sql FROM sqlite_schema WHERE type='index' ORDER BY name"
        ).fetchall()
    original = source.read_bytes()
    statements = []
    connect = src._connect

    def traced_connect(path, **options):
        conn = connect(path, **options)
        if path == target:
            conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(src, "_connect", traced_connect)
    result = src._build_hot_db(
        source_db=source, dest_tmp=target, cutoff="2021-01-01", timeout_seconds=10
    )
    assert result["index_build_strategy"] == "maintained_during_row_copy"
    assert set(result["indexes_created"]) == {name for name, _ in expected_schema}
    first_copy = next(
        i for i, sql in enumerate(statements) if sql.startswith("INSERT INTO main.")
    )
    assert all(statements.index(sql) < first_copy for sql in definitions)
    assert source.read_bytes() == original
    with sqlite3.connect(target) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert (
            conn.execute(
                "SELECT name, sql FROM sqlite_schema WHERE type='index' ORDER BY name"
            ).fetchall()
            == expected_schema
        )
        assert conn.execute(
            'SELECT id FROM jsonl_records INDEXED BY "wide expression" '
            "ORDER BY json_extract(payload_json, '$.symbol') COLLATE NOCASE DESC"
        ).fetchall() == [(i,) for i in range(hot_rows, 0, -1)]
        assert conn.execute(
            "SELECT id FROM jsonl_records INDEXED BY tail_key ORDER BY tail DESC"
        ).fetchall() == [(i,) for i in range(hot_rows, 0, -1)]
        if hot_rows:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO jsonl_records VALUES(9, '2999-01-01', '{}', 'TAIL1')"
                )


def test_rebuild_total_deadline_is_cooperative_and_restored(tmp_path, monkeypatch):
    source, options = _rebuild_fixture(tmp_path, monkeypatch)
    original = source.read_bytes()
    monkeypatch.setattr(
        src,
        "_inspect_source",
        lambda *args: ({}, 64 * 1024**2, {"hot_payload_bytes": 0}),
    )
    ticks = iter([0.0])
    monkeypatch.setattr(src.time, "monotonic", lambda: next(ticks, 2000.0))
    result = src.build_local_cache_payload(tmp_path, operation_seconds=1, **options)
    assert result["ok"] is False
    assert source.read_bytes() == original
    assert src._REBUILD_DEADLINE is None


def test_rebuild_accepts_stable_wal_database(tmp_path, monkeypatch):
    source, options = _rebuild_fixture(tmp_path, monkeypatch)
    with sqlite3.connect(source) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
    result = src.build_local_cache_payload(tmp_path, **options)
    assert result["ok"] is True, result.get("blockers")


def test_source_signature_detects_committed_wal_changes(tmp_path):
    source = tmp_path / "source.sqlite3"
    _seed_db(source)
    with sqlite3.connect(source) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        before = src._source_family_signature(source)
        conn.execute("UPDATE merge_state SET value='changed'")
        conn.commit()
        after = src._source_family_signature(source)
        assert after["database"] == before["database"]
        assert after["wal"] != before["wal"]


def test_staged_copy_with_valid_sqlite_but_changed_content_is_rejected(
    tmp_path, monkeypatch
):
    source, options = _rebuild_fixture(tmp_path, monkeypatch)
    original = source.read_bytes()
    copy = src._copy_staged_cache

    def corrupt(src_path, destination, **kwargs):
        copy(src_path, destination, **kwargs)
        with sqlite3.connect(destination) as conn:
            conn.execute("UPDATE jsonl_records SET payload_json='[]'")

    monkeypatch.setattr(src, "_copy_staged_cache", corrupt)
    result = src.build_local_cache_payload(tmp_path, **options)
    assert result["ok"] is False
    assert "local_staged_copy_sha256_mismatch" in str(result["blockers"])
    assert source.read_bytes() == original


def test_staging_budget_can_cap_pessimistic_estimate_without_reducing_reserves():
    gib = 1024**3
    result = src._hot_staging_budget(74 * gib, 33 * gib, 69 * gib,
                                    120 * gib, 32 * gib, 64 * gib)
    assert result["ready"]
    assert result["estimate_exceeds_capacity"]
    assert result["max_database_bytes"] < 37 * gib
    assert result["reserved_staging_bytes"] <= 37 * gib
    assert result["reserved_staging_bytes"] <= (120 - 64) * gib


@pytest.mark.parametrize("field,value", [(0, None), (1, None), (2, None), (3, None),
                                        (0, True), (1, -1), (2, 0), (3, 0)])
def test_staging_budget_rejects_unknown_or_insufficient_capacity(field, value):
    gib = 1024**3
    args = [74 * gib, 33 * gib, 69 * gib, 120 * gib, 32 * gib, 64 * gib]
    args[field] = value
    assert not src._hot_staging_budget(*args)["ready"]


def test_sqlite_page_ceiling_stops_growth_and_preserves_source(tmp_path):
    source = tmp_path / "source.sqlite3"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE jsonl_records(id INTEGER PRIMARY KEY, ingested_at TEXT, payload_json BLOB)")
        conn.execute("INSERT INTO jsonl_records VALUES (1, '2999-01-01', zeroblob(8388608))")
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / "staged.sqlite3"
    with pytest.raises(sqlite3.DatabaseError, match="full"):
        src._build_hot_db(source_db=source, dest_tmp=target, cutoff="2021-01-01",
                          timeout_seconds=10, max_database_bytes=2 * 1024**2)
    assert target.stat().st_size <= 2 * 1024**2
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original


def test_rebuild_with_size_cap_smaller_than_estimate_remains_verified(tmp_path, monkeypatch):
    source, options = _rebuild_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(src, "_disk_free_bytes", lambda _: 65 * 1024**3)
    real_inspect = src._inspect_source

    def inspect(*args):
        counts, estimate, observation = real_inspect(*args)
        return counts, 2 * 1024**3, observation

    monkeypatch.setattr(src, "_inspect_source", inspect)
    result = src.build_local_cache_payload(tmp_path, **options)
    assert result["ok"], result["blockers"]
    assert result["staging_budget"]["estimate_exceeds_capacity"]
    assert result["hot_db"]["size_bytes"] <= result["staging_budget"]["max_database_bytes"]
    assert result["cold_exports"][0]["restore_verification"]["ok"]
    assert result["coverage_check"]["ok"]
    proof = json.loads(Path(result["restore_proof_path"]).read_text())
    verification = proof["staged_copy_verification"]
    assert verification["ok"]
    assert verification["local_sha256"] == verification["external_sha256"]
    assert len(verification["local_sha256"]) == 64
    assert proof["staging_budget"] == result["staging_budget"]


def test_staged_copy_stops_when_free_space_changes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"x" * (17 * 1024**2))
    target = tmp_path / "staged"
    values = iter([32 * 1024**2, 0])
    monkeypatch.setattr(src, "_disk_free_bytes", lambda _: next(values))
    with pytest.raises(RuntimeError, match="local_free_during_copy"):
        src._copy_staged_cache(source, target, reserve_bytes=1)
    assert target.stat().st_size == 16 * 1024**2
    assert source.stat().st_size == 17 * 1024**2


def test_source_inspection_interrupts_sql_vm_work(tmp_path, monkeypatch):
    source = tmp_path / "source.sqlite3"
    _seed_db(source)

    def expensive(conn, cutoff):
        return conn.execute(
            "WITH RECURSIVE n(x) AS (VALUES(0) UNION ALL SELECT x+1 FROM n WHERE x<100000000) SELECT sum(x) FROM n"
        ).fetchone()

    monkeypatch.setattr(src, "_source_counts", expensive)
    ticks = iter([0.0, 1000.0])
    monkeypatch.setattr(src.time, "monotonic", lambda: next(ticks, 1000.0))
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        src._inspect_source(source, "2021-01-01", source.stat().st_size, 1)


def test_local_cache_rebuild_exports_cold_rows_and_atomically_prunes_old_cache(
    monkeypatch, tmp_path: Path, restore_encoder
) -> None:
    source = tmp_path / "local_fallback_storage" / "data" / "jsonl_link.sqlite3"
    _seed_db(source)
    repo_db = tmp_path / "data" / "jsonl_link.sqlite3"
    repo_db.parent.mkdir(parents=True)
    repo_db.symlink_to(source)
    external = tmp_path / "external"
    external.mkdir()
    monkeypatch.setattr(
        src.writer_state,
        "writer_state_snapshot",
        lambda project_root: {"active": False, "running": False},
    )
    monkeypatch.setenv("BOT_LOGS_SQLITE_HOT_ROUTE_EXPORT_ENGINE", "pyarrow")

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

    assert payload["ok"] is True, payload
    assert payload["overall_status"] == "rebuilt_pruned"
    assert payload["coverage_check"]["ok"] is True
    proof = payload["cold_exports"][0]["restore_verification"]
    assert proof["validation"] == "full_typed_row_restore_sha256"
    assert proof["row_encoding"] in {"msgpack_sqlite_scalars_v1", "json_type_pairs_ascii_v1"}
    assert proof["source_sha256"] == proof["restored_sha256"]
    assert Path(payload["restore_proof_path"]).is_file()
    assert repo_db.resolve() == source.resolve()
    with sqlite3.connect(source) as conn:
        assert conn.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()[0] == 1
        assert (
            conn.execute(
                "SELECT value FROM merge_state WHERE name='cursor'"
            ).fetchone()[0]
            == "2"
        )
    assert not Path(payload["old_cache_db"]).exists()


def test_local_cache_rebuild_refuses_active_writer_without_touching_source(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "local_fallback_storage" / "data" / "jsonl_link.sqlite3"
    _seed_db(source)
    original = source.read_bytes()
    external = tmp_path / "external"
    external.mkdir()
    monkeypatch.setattr(
        src.writer_state,
        "writer_state_snapshot",
        lambda project_root: {"active": True, "running": True},
    )

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


def test_cold_capacity_checks_export_volume_not_staging(tmp_path, monkeypatch):
    from types import SimpleNamespace

    cold, staging = tmp_path / "cold", tmp_path / "staging"
    monkeypatch.setattr(src, "_capacity_probe", lambda path: path)
    monkeypatch.setattr(
        Path,
        "stat",
        lambda path, **kw: SimpleNamespace(st_dev=1 if path == cold else 2),
    )
    monkeypatch.setattr(
        src,
        "_disk_free_bytes",
        lambda path: 63 * 1024**3 if path == cold else 200 * 1024**3,
    )
    result = src._cold_capacity_budget(cold, staging, 4 * 1024**3, 64 * 1024**3)
    assert not result["ready"]
    assert result["probe_path"] == str(cold)
    assert result["minimum_free_after_bytes"] == 64 * 1024**3
    assert not result["same_filesystem_as_staging"]


def test_cold_capacity_reserves_staging_only_on_shared_filesystem(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    monkeypatch.setattr(src, "_capacity_probe", lambda path: path)
    monkeypatch.setattr(Path, "stat", lambda path, **kw: SimpleNamespace(st_dev=1))
    result = src._cold_capacity_budget(
        tmp_path / "cold", tmp_path / "staging", 4 * 1024**3, 64 * 1024**3
    )
    assert result["ready"] and result["same_filesystem_as_staging"]
    assert result["minimum_free_after_bytes"] == 68 * 1024**3


def test_cache_capacity_refuses_symlink_destination(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="unsafe_cache_storage_route"):
        src._capacity_probe(link / "cold")


def test_cache_cold_destination_capacity_failure_preserves_source(
    tmp_path, monkeypatch
):
    source = tmp_path / "local_fallback_storage/data/jsonl_link.sqlite3"
    _seed_db(source)
    before = source.read_bytes()
    external, cold = tmp_path / "external", tmp_path / "cold"
    external.mkdir()
    cold.mkdir()
    monkeypatch.setattr(
        src.writer_state, "writer_state_snapshot", lambda root: {"active": False}
    )
    monkeypatch.setattr(
        src, "_disk_free_bytes", lambda path: (20 if path == cold else 512) * 1024**3
    )
    result = src.build_local_cache_payload(
        tmp_path,
        relative_path="data/jsonl_link.sqlite3",
        hot_hours=18,
        apply=True,
        prune_old_cache=True,
        min_local_free_after_gb=32,
        min_external_free_after_gb=64,
        cold_export_root=cold,
        batch_size=1000,
        compression="zstd",
        require_writer_idle=True,
        timeout_seconds=10,
        external_root=external,
    )
    assert result["blockers"] == ["cold_export_destination_free_below_guard"]
    assert source.read_bytes() == before
    assert not list(cold.iterdir())


def test_local_cache_rebuild_accepts_orphaned_running_progress(
    monkeypatch, tmp_path: Path
) -> None:
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


def test_cold_export_caps_batches_and_releases_arrow_memory(
    monkeypatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "source.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE jsonl_records (id INTEGER, ingested_at TEXT, payload_json TEXT)"
        )
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


def test_duckdb_cold_export_is_row_exact_and_bounded(
    monkeypatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "source.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE jsonl_records (id INTEGER, ingested_at TEXT, payload_json TEXT)"
        )
        conn.executemany(
            "INSERT INTO jsonl_records VALUES (?, ?, ?)",
            [
                (1, "2020-01-01T00:00:00+00:00", "old"),
                (2, "2999-01-01T00:00:00+00:00", "new"),
            ],
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


def test_isolated_arrow_export_releases_workers_and_writes_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "source.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE jsonl_records (id INTEGER PRIMARY KEY, ingested_at TEXT, payload_json TEXT)"
        )
        conn.executemany(
            "INSERT INTO jsonl_records VALUES (?, ?, ?)",
            [
                (1, "2020-01-01T00:00:00+00:00", "old"),
                (2, "2999-01-01T00:00:00+00:00", "new"),
            ],
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

        def set_progress_handler(self, callback, steps):
            assert callable(callback)
            assert steps == 1000

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


def test_isolated_arrow_export_resumes_verified_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "source.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE jsonl_records (id INTEGER PRIMARY KEY, ingested_at TEXT, payload_json TEXT)"
        )
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
    assert all(
        ".partial_dataset" not in row["output_path"] for row in final_manifest["parts"]
    )


@pytest.mark.parametrize(
    ("valid", "supplied_token", "allowed"),
    [
        (True, "", False),
        (True, "wrong", False),
        (False, "owned", False),
        (True, "owned", True),
    ],
)
def test_cli_requires_explicit_existing_hold_authority(
    tmp_path, monkeypatch, valid, supplied_token, allowed
):
    from types import SimpleNamespace
    from core.runtime_maintenance import MAINTENANCE_HOLD_TOKEN_ENV

    out = tmp_path / "result.json"
    monkeypatch.setattr(
        src.sys,
        "argv",
        [
            "storage_sqlite_hot_route.py",
            "--project-root",
            str(tmp_path),
            "--rebuild-local-cache",
            "--apply",
            "--out-file",
            str(out),
            "--json",
        ],
    )
    monkeypatch.setattr(
        src, "resolve_external_storage", lambda: SimpleNamespace(external_root=tmp_path)
    )
    monkeypatch.setenv(MAINTENANCE_HOLD_TOKEN_ENV, supplied_token)
    monkeypatch.setattr(
        src,
        "maintenance_hold_snapshot",
        lambda _: {
            "active": True,
            "valid": valid,
            "token": "owned",
            "owner": "another_owner",
        },
    )

    def forbidden(*args, **kwargs):
        pytest.fail("existing hold must not be engaged or released")

    monkeypatch.setattr(src, "engage_maintenance_hold", forbidden)
    monkeypatch.setattr(src, "release_maintenance_hold", forbidden)
    monkeypatch.setattr(
        src,
        "_wait_for_writer_idle",
        (lambda *args, **kwargs: {"ok": True}) if allowed else forbidden,
    )
    monkeypatch.setattr(
        src,
        "build_local_cache_payload",
        (lambda *args, **kwargs: {"ok": True}) if allowed else forbidden,
    )
    assert src.main() == (0 if allowed else 1)
    result = json.loads(out.read_text())
    assert result["maintenance_coordination"]["owned_hold"] is False
    if not allowed:
        assert result["blockers"] == [
            (
                "runtime_maintenance_hold_not_authorized"
                if valid
                else "runtime_maintenance_hold_invalid"
            )
        ]


def test_rebuild_receipt_dates_completion_not_start(tmp_path, monkeypatch):
    from scripts.ops import storage_sqlite_hot_route as owner

    monkeypatch.setattr(
        owner,
        "_build_local_cache_payload",
        lambda *a, **kw: {
            "timestamp_utc": "2026-09-23T12:00:00+00:00",
            "ok": True,
        },
    )
    monkeypatch.setattr(owner, "_iso", lambda: "2026-09-23T12:20:00+00:00")
    result = owner.build_local_cache_payload(tmp_path, apply=True)
    assert result["started_at_utc"] == "2026-09-23T12:00:00+00:00"
    assert (
        result["timestamp_utc"]
        == result["assessment_completed_at_utc"]
        == "2026-09-23T12:20:00+00:00"
    )
