import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.ops import ingestion_verification as src
from scripts.ops import ingestion_storage_control as control

START = "2020-01-01T00:00:01Z"
END = "2020-01-01T00:00:03Z"


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", raising=False)
    return tmp_path


def database(root, *, rel="data/jsonl_link.sqlite3", table="jsonl_records", index=True):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = src.TABLES[table]
    with sqlite3.connect(path) as conn:
        conn.execute(
            f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, ingested_at TEXT, payload_json TEXT, payload_sha1 TEXT, {stream} TEXT)"
        )
        if index:
            conn.execute(
                f"CREATE INDEX idx_{table}_ingested_at ON {table}(ingested_at)"
            )
    return path


def insert(path, stamp, *, payload='{"value":1}', digest=None, table="jsonl_records"):
    with sqlite3.connect(path) as conn:
        conn.execute(
            f"INSERT INTO {table} (ingested_at, payload_json, payload_sha1, {src.TABLES[table]}) VALUES (?,?,?,?)",
            (
                stamp,
                payload,
                (
                    digest
                    if digest is not None
                    else hashlib.sha1(payload.encode()).hexdigest()
                ),
                "decision",
            ),
        )


def verify(root, **kwargs):
    return src.build_verification(root, since=START, until=END, **kwargs)


def test_window_hashes_and_source_completeness_are_separate(root):
    path = database(root)
    for stamp in (
        "2020-01-01T00:00:00.999999+00:00",
        "2020-01-01T00:00:01+00:00",
        "2020-01-01T00:00:02.999999+00:00",
        "2020-01-01T00:00:03+00:00",
    ):
        insert(path, stamp)
    before = path.read_bytes()
    result = verify(root)
    assert result["stored_row_verification_complete"]
    table = result["databases"][0]["tables"][0]
    assert table["committed_rows"] == table["checked_rows"] == 2
    assert table["checked_rows_by_stream"] == {"decision": 2}
    assert not result["all_new_source_data_ingested"]
    assert not any(result["authority"].values())
    assert path.read_bytes() == before


@pytest.mark.parametrize("payload,digest", [("{}", "bad"), ("not json", None)])
def test_bad_hash_or_json_does_not_certify_verification(root, payload, digest):
    path = database(root)
    insert(path, START, payload=payload, digest=digest)
    result = verify(root)
    assert not result["stored_row_verification_complete"]
    assert result["databases"][0]["status"] == "failed"


def test_json_snapshot_table_is_verified(root):
    path = database(root, table="json_file_records")
    insert(path, START, table="json_file_records", payload="[1,2,3]")
    assert verify(root)["stored_row_verification_complete"]


def test_aliases_deduplicate_but_distinct_copies_remain_separate(root):
    path = database(root, rel="local_fallback_storage/data/jsonl_link.sqlite3")
    insert(path, START)
    (root / "data").mkdir()
    (root / "data/jsonl_link.sqlite3").symlink_to(path)
    other = database(root, rel="data/sql_link_shards/jsonl_link_runtime.sqlite3")
    insert(other, START)
    result = verify(root)
    assert len(result["databases"]) == 2
    assert sorted(len(row["aliases"]) for row in result["databases"]) == [1, 2]
    assert "not unique global events" in result["scope"]["counting"]


@pytest.mark.parametrize("budget", [{"max_rows": 1}, {"max_payload_mib": 1}])
def test_limits_preserve_exact_count_but_report_incomplete_hash_coverage(root, budget):
    path = database(root)
    for _ in range(2):
        insert(path, START, payload=json.dumps({"value": "x" * (600 * 1024)}))
    result = verify(root, **budget)
    table = result["databases"][0]["tables"][0]
    assert table["committed_rows"] == 2
    assert table["checked_rows"] == 1
    assert not result["stored_row_verification_complete"]
    assert table["reason"] == "payload_or_row_budget"


def test_no_database_does_not_create_one_or_pass(root):
    assert not verify(root)["stored_row_verification_complete"]
    assert not (root / "data").exists()


def test_missing_index_is_incomplete_without_full_table_scan(root):
    database(root, index=False)
    row = verify(root)["databases"][0]["tables"][0]
    assert row["reason"] == "required_time_index_missing"
    assert row["status"] == "incomplete"


def test_corrupt_database_is_explicit_and_preserved(root):
    path = database(root)
    path.write_bytes(b"not sqlite")
    result = verify(root)
    assert result["databases"][0]["status"] == "incomplete"
    assert path.read_bytes() == b"not sqlite"


def test_committed_wal_rows_are_visible_without_immutable_mode(root):
    path = database(root)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute(
            "INSERT INTO jsonl_records VALUES (1,?,?,?,?)",
            (START, "{}", hashlib.sha1(b"{}").hexdigest(), "decision"),
        )
        conn.commit()
        assert verify(root)["databases"][0]["tables"][0]["committed_rows"] == 1
    finally:
        conn.close()


def test_protected_directory_and_receipt_aliases_are_not_opened(root, monkeypatch):
    (root / "data").mkdir()
    (root / "data/sql_link_shards").symlink_to("/Volumes/VIDEO/private")
    (root / "governance").symlink_to("/Volumes/VIDEO/private")
    real_lstat = Path.lstat

    def check(path, *args, **kwargs):
        assert not path.is_relative_to("/Volumes/VIDEO")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", check)
    result = verify(root)
    assert result["route_findings"][0]["status"] == "protected_path"
    assert result["source_census"]["status"] == "unavailable"


def test_database_replacement_is_rejected_before_open(root):
    path = database(root)
    rows, _ = src._databases(root)
    path.rename(path.with_suffix(".old"))
    database(root)
    result = src._verify_database(
        rows[0],
        START[:19],
        END[:19],
        {"deadline": src.time.monotonic() + 10, "bytes": 100, "rows": 10},
    )
    assert result["reason"] == "ValueError:database_identity_changed"


def test_deadline_does_not_open_database(root, monkeypatch):
    database(root)
    rows, _ = src._databases(root)
    monkeypatch.setattr(
        src.sqlite3, "connect", lambda *a, **kw: pytest.fail("opened after deadline")
    )
    assert (
        src._verify_database(rows[0], START, END, {"deadline": 0})["reason"]
        == "deadline"
    )


@pytest.mark.parametrize(
    "since,until",
    [
        ("2020-01-01", END),
        (END, START),
        ("2020-01-01T00:00:01.000001Z", END),
        (START, "2999-01-01T00:00:00Z"),
    ],
)
def test_bad_windows_rejected(root, since, until):
    with pytest.raises(ValueError):
        src.build_verification(root, since=since, until=until)


@pytest.mark.parametrize(
    "budget", [{"max_seconds": 0}, {"max_payload_mib": 1025}, {"max_rows": -1}]
)
def test_invalid_budgets_rejected(root, budget):
    with pytest.raises(ValueError):
        verify(root, **budget)


def test_census_freshness_is_not_refreshed_by_audit(root):
    folder = root / "governance/health"
    folder.mkdir(parents=True)
    (folder / "ingestion_backpressure_latest.json").write_text(
        json.dumps({"timestamp_utc": START, "pending_lines_total": 20})
    )
    result = verify(root)
    assert result["source_census"]["status"] == "stale"
    assert result["source_census"]["pending_lines_total"] == 20


def test_cli_keeps_verification_separate_from_full_health(root, monkeypatch, capsys):
    database(root)
    monkeypatch.setattr(
        "sys.argv",
        [
            "control",
            "--project-root",
            str(root),
            "--verify-new-ingestion",
            "--since",
            START,
            "--until",
            END,
            "--json",
        ],
    )
    monkeypatch.setattr(
        control, "build_payload", lambda *_: pytest.fail("full health invoked")
    )
    assert control.main() == 0
    assert json.loads(capsys.readouterr().out)["stored_row_verification_complete"]
    assert (root / "governance/health/ingestion_verification_latest.json").exists()
    assert not (
        root / "governance/health/ingestion_storage_control_latest.json"
    ).exists()


def test_definitions_include_data_classes_and_separate_verification_contract(root):
    from scripts.ops.ingestion_data_contract import build_data_plane_definition

    result = build_data_plane_definition(root)
    assert {row["class"] for row in result["data_classes"]} == {
        "source_payload",
        "append_only_evidence",
        "versioned_json_snapshot",
        "durable_queue",
        "analytical_mirror",
        "sealed_history",
    }
    assert not result["verification_contract"]["global_unique_event_count"]


@pytest.mark.parametrize(
    "args",
    [
        ["--verify-new-ingestion"],
        ["--since", START],
        ["--definitions-only", "--verify-new-ingestion", "--since", START],
    ],
)
def test_cli_rejects_ambiguous_or_incomplete_modes(root, monkeypatch, args):
    monkeypatch.setattr("sys.argv", ["control", "--project-root", str(root), *args])
    with pytest.raises(SystemExit) as exc:
        control.main()
    assert exc.value.code == 2


def test_cli_cannot_publish_through_protected_alias(root, monkeypatch):
    output = root / "protected-output"
    output.symlink_to("/Volumes/VIDEO/private")
    monkeypatch.setattr(
        "sys.argv",
        [
            "control",
            "--project-root",
            str(root),
            "--verify-new-ingestion",
            "--since",
            START,
            "--until",
            END,
            "--out-file",
            str(output / "report.json"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        control.main()
    assert exc.value.code == 2
