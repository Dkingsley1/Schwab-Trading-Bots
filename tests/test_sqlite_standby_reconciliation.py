import sqlite3

import pytest

from core.sqlite_standby_reconciliation import verify_records


@pytest.fixture
def pair(tmp_path):
    paths = []
    for directory in ("source", "target"):
        folder = tmp_path / directory
        folder.mkdir()
        path = folder / "bot_channel_queue.sqlite3"
        connection = sqlite3.connect(path)
        connection.executescript("""
            CREATE TABLE channel_messages(id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT);
            CREATE TABLE channel_consumer_state(consumer TEXT, channel TEXT, last_id INTEGER,
                last_message_id TEXT, updated_at TEXT, PRIMARY KEY(consumer,channel));
            CREATE TABLE channel_processing_claims(id INTEGER PRIMARY KEY, state TEXT);
            INSERT INTO channel_messages(payload) VALUES('retained payload');
            INSERT INTO channel_consumer_state VALUES('reader','test',1,'message','old');
            INSERT INTO channel_processing_claims VALUES(1,'finalized');
        """)
        connection.close()
        paths.append(path)
    return paths


def mutate(path, sql):
    with sqlite3.connect(path) as connection:
        connection.executescript(sql)


def test_exact_rows_and_forward_cursors(pair):
    source, target = pair
    mutate(
        target,
        "UPDATE channel_consumer_state SET last_id=2,last_message_id='new',updated_at='new';",
    )
    assert verify_records(source, target)["all_operational_rows_preserved"] is True


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM channel_messages;",
        "UPDATE channel_messages SET payload='changed';",
        "UPDATE channel_consumer_state SET last_id=0;",
        "UPDATE channel_consumer_state SET last_message_id='wrong';",
        "UPDATE channel_processing_claims SET state='claimed';",
        "UPDATE sqlite_sequence SET seq=0;",
        "CREATE TABLE unreviewed(id INTEGER PRIMARY KEY);",
        "ALTER TABLE channel_messages ADD COLUMN unexpected TEXT;",
    ],
)
def test_missing_or_changed_evidence_blocks(pair, sql):
    source, target = pair
    mutate(target, sql)
    with pytest.raises(ValueError):
        verify_records(source, target)
    assert source.exists()


def test_unknown_name_rejected_before_open(tmp_path):
    with pytest.raises(ValueError, match="not_allowed"):
        verify_records(tmp_path / "unknown.sqlite3", tmp_path / "unknown.sqlite3")


def test_guard_failure_is_not_swallowed(pair):
    def guard():
        raise RuntimeError("hold lost")

    with pytest.raises(RuntimeError, match="hold lost"):
        verify_records(*pair, guard=guard)


def test_report_remap_and_merge_cursor(tmp_path):
    paths = []
    for label in ("source", "target"):
        folder = tmp_path / label
        folder.mkdir()
        path = folder / "jsonl_link.sqlite3"
        with sqlite3.connect(path) as connection:
            for name in (
                "json_file_records",
                "jsonl_records",
                "db_maintenance_events",
                "one_numbers_snapshots",
            ):
                connection.execute(
                    f"CREATE TABLE {name}(id INTEGER PRIMARY KEY AUTOINCREMENT,payload TEXT)"
                )
                connection.execute(f"INSERT INTO {name}(payload) VALUES('original')")
            connection.execute(
                "CREATE TABLE shard_merge_state(shard_name TEXT PRIMARY KEY,last_jsonl_id INTEGER,last_json_file_id INTEGER,updated_at TEXT)"
            )
            connection.execute(
                "INSERT INTO shard_merge_state VALUES('shard',1,1,'old')"
            )
        paths.append(path)
    source, target = paths
    mutate(
        target,
        "UPDATE one_numbers_snapshots SET payload='other'; INSERT INTO one_numbers_snapshots(payload) VALUES('original'); UPDATE shard_merge_state SET last_jsonl_id=2;",
    )
    with pytest.raises(ValueError, match="report_missing"):
        verify_records(source, target)
    assert verify_records(
        source, target, report_mapping=[{"standby_id": 1, "primary_id": 2}]
    )["all_operational_rows_preserved"]
    mutate(target, "UPDATE shard_merge_state SET last_json_file_id=0;")
    with pytest.raises(ValueError, match="shard_merge_state"):
        verify_records(
            source, target, report_mapping=[{"standby_id": 1, "primary_id": 2}]
        )
