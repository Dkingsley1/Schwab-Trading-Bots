"""Read-only custody checks for the three explicitly named central standbys."""

import sqlite3
import time

TABLES = {
    "bot_channel_queue.sqlite3": {
        "channel_messages",
        "channel_consumer_state",
        "channel_processing_claims",
    },
    "snapshot_context.sqlite3": {
        "snapshot_health_records",
        "debug_snapshot_file_blobs",
        "debug_snapshot_raw_records",
    },
    "jsonl_link.sqlite3": {
        "json_file_records",
        "jsonl_records",
        "db_maintenance_events",
        "one_numbers_snapshots",
        "shard_merge_state",
    },
}


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def verify_records(
    source, target, *, report_mapping=(), guard=lambda: None, timeout=900
):
    """Caller must own a maintenance hold and verify both files are quiescent."""
    expected = TABLES.get(source.name)
    if expected is None or target.name != source.name:
        raise ValueError("central_retirement_database_not_allowed")
    guard()
    deadline = time.monotonic() + timeout
    connection = sqlite3.connect(target.as_uri() + "?mode=ro&immutable=1", uri=True)
    result = []
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute(
            "ATTACH DATABASE ? AS standby", (source.as_uri() + "?mode=ro&immutable=1",)
        )
        connection.set_progress_handler(
            lambda: int(time.monotonic() >= deadline), 10000
        )
        for schema in ("main", "standby"):
            names = {
                r[0]
                for r in connection.execute(
                    f"SELECT name FROM {schema}.sqlite_master WHERE type='table' AND name NOT GLOB 'sqlite_*'"
                )
            }
            if names != expected:
                raise ValueError("central_retirement_unreviewed_schema:" + schema)
        mapping = {r["standby_id"]: r["primary_id"] for r in report_mapping}
        if len(mapping) != len(report_mapping):
            raise ValueError("central_retirement_duplicate_report_mapping")
        for table in sorted(expected):
            guard()
            quoted = _quote(table)
            columns = connection.execute(
                f"PRAGMA standby.table_info({quoted})"
            ).fetchall()
            if (
                not columns
                or columns
                != connection.execute(f"PRAGMA main.table_info({quoted})").fetchall()
            ):
                raise ValueError("central_retirement_schema_mismatch:" + table)
            keys = [r[1] for r in sorted(columns, key=lambda r: r[5]) if r[5]]
            if not keys:
                raise ValueError("central_retirement_primary_key_required:" + table)
            join = " AND ".join(f"s.{_quote(k)}=t.{_quote(k)}" for k in keys)
            predicate = " AND ".join(
                f"s.{_quote(r[1])} IS t.{_quote(r[1])}" for r in columns
            )
            method = "exact_rows"
            if table == "one_numbers_snapshots":
                checked = 0
                for row in connection.execute(f"SELECT * FROM standby.{quoted}"):
                    checked += 1
                    if checked > 10000 or time.monotonic() >= deadline:
                        raise ValueError("central_retirement_report_budget")
                    actual = connection.execute(
                        f"SELECT * FROM main.{quoted} WHERE id=?",
                        (mapping.get(row[0], row[0]),),
                    ).fetchone()
                    if actual is None or row[1:] != actual[1:]:
                        raise ValueError("central_retirement_report_missing_or_changed")
                result.append(
                    {
                        "table": table,
                        "verified_rows": checked,
                        "method": "receipt_mapped_report_payloads",
                    }
                )
                continue
            if table == "shard_merge_state":
                predicate = "t.last_jsonl_id >= s.last_jsonl_id AND t.last_json_file_id >= s.last_json_file_id"
                method = "nonregressed_merge_cursors"
            elif table == "channel_consumer_state":
                predicate = "t.last_id >= s.last_id AND (t.last_id > s.last_id OR t.last_message_id IS s.last_message_id)"
                method = "nonregressed_consumer_cursors"
            mismatch = connection.execute(
                f"SELECT 1 FROM standby.{quoted} s WHERE NOT EXISTS "
                f"(SELECT 1 FROM main.{quoted} t WHERE {join} AND {predicate}) LIMIT 1"
            ).fetchone()
            if mismatch:
                raise ValueError("central_retirement_missing_or_changed_rows:" + table)
            count = connection.execute(
                f"SELECT COUNT(*) FROM standby.{quoted}"
            ).fetchone()[0]
            result.append({"table": table, "verified_rows": count, "method": method})
        sequence_gap = connection.execute(
            "SELECT 1 FROM standby.sqlite_sequence s WHERE NOT EXISTS "
            "(SELECT 1 FROM main.sqlite_sequence t WHERE t.name=s.name AND t.seq>=s.seq) LIMIT 1"
        ).fetchone()
        if sequence_gap:
            raise ValueError("central_retirement_sequence_regression")
        guard()
        return {
            "all_operational_rows_preserved": True,
            "tables": result,
            "sqlite_statistics": "optimization_metadata_preserved_in_independent_backup",
        }
    finally:
        connection.close()
