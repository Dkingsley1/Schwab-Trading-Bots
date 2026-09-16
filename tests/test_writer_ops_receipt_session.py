import json
import sqlite3

import pytest

from scripts import link_jsonl_to_sql as writer


def sync(root, conn, session, name, *, dry_run=False):
    path = root / "decisions" / f"{name}.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"value": name, "log_schema_version": 2}) + "\n")
    return writer._sync_file_to_sqlite(
        conn,
        "jsonl_records",
        root,
        path,
        start_line=0,
        start_offset_bytes=0,
        dry_run=dry_run,
        lock_retries=0,
        lock_retry_delay_seconds=0.01,
        latency_all=None,
        latency_stream=None,
        invalid_log_path=None,
        invalid_sample_limit=0,
        run_id="",
        iter_id="",
        ops_session=session,
    )


def primary():
    conn = sqlite3.connect(":memory:")
    writer._ensure_sqlite_schema(conn, "jsonl_records")
    return conn


def test_two_sources_share_one_fully_checked_connection_and_commit_receipts(
    tmp_path, monkeypatch
):
    checked = []
    original = writer.ops_data_plane._assert_sqlite_quick_check_ok

    def check(conn):
        checked.append(conn)
        original(conn)

    monkeypatch.setattr(writer.ops_data_plane, "_assert_sqlite_quick_check_ok", check)
    session = writer._OpsReceiptSession(tmp_path)
    conn = primary()
    try:
        for name in ("first", "second"):
            result = sync(tmp_path, conn, session, name)
            assert result["inserted"] == 1
            assert result["ops_write_failures"] == 0
            with sqlite3.connect(
                writer.ops_data_plane.resolve_db_path(tmp_path)
            ) as observer:
                assert (
                    observer.execute(
                        "SELECT watermark_value FROM source_watermarks WHERE entity_key=?",
                        (f"decisions/{name}.jsonl",),
                    )
                    .fetchone()[0]
                    .startswith("1:")
                )
        assert len(checked) == 1
        saved = session.get()
    finally:
        session.close()
        conn.close()
    with pytest.raises(sqlite3.ProgrammingError):
        saved.execute("SELECT 1")
    with pytest.raises(RuntimeError):
        session.get()
    assert len(checked) == 1


@pytest.mark.parametrize("failure", ["connect", "watermark"])
def test_receipt_failure_is_visible_without_reconnect_storm_or_source_loss(
    tmp_path, monkeypatch, failure
):
    attempts = []
    original = writer.ops_data_plane.connect

    def connect(root):
        attempts.append(root)
        if failure == "connect":
            raise sqlite3.OperationalError("database is locked")
        return original(root)

    def watermark(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(writer.ops_data_plane, "connect", connect)
    if failure == "watermark":
        monkeypatch.setattr(writer.ops_data_plane, "record_watermark", watermark)
    session = writer._OpsReceiptSession(tmp_path)
    conn = primary()
    try:
        for name in ("first", "second"):
            result = sync(tmp_path, conn, session, name)
            assert result["ops_write_failures"] >= 1
            assert result["inserted"] == 1
        assert len(attempts) == 1
        assert session._conn is None
        assert conn.execute("SELECT count(*) FROM jsonl_records").fetchone()[0] == 2
    finally:
        session.close()
        conn.close()


def test_dry_run_does_not_open_receipt_session(tmp_path, monkeypatch):
    monkeypatch.setattr(
        writer.ops_data_plane, "connect", lambda root: pytest.fail("receipt opened")
    )
    session = writer._OpsReceiptSession(tmp_path)
    conn = primary()
    try:
        sync(tmp_path, conn, session, "dry", dry_run=True)
        assert not session.attempted
    finally:
        session.close()
        conn.close()


def test_failed_commit_invalidates_session_without_double_close_error(
    tmp_path, monkeypatch
):
    closed = []

    class BrokenCommit:
        def execute(self, *args, **kwargs):
            return None

        def commit(self):
            raise sqlite3.OperationalError("database is locked")

        def rollback(self):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr(writer.ops_data_plane, "connect", lambda root: BrokenCommit())
    session = writer._OpsReceiptSession(tmp_path)
    conn = primary()
    try:
        assert sync(tmp_path, conn, session, "first")["ops_write_failures"] >= 1
        assert session._conn is None
    finally:
        session.close()
        conn.close()
    assert closed == [True]
