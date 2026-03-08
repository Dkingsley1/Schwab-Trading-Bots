import importlib.util
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "link_jsonl_to_sql.py"
SPEC = importlib.util.spec_from_file_location("link_jsonl_to_sql_module", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LinkJsonlToSqlTests(unittest.TestCase):
    def test_extract_correlation_fields_prefers_top_level(self) -> None:
        row = {
            "run_id": "run-top",
            "iter_id": "iter-top",
            "decision_id": "decision-top",
            "parent_decision_id": "parent-top",
            "log_schema_version": "9",
            "metadata": {
                "run_id": "run-meta",
                "iter_id": "iter-meta",
                "decision_id": "decision-meta",
                "parent_decision_id": "parent-meta",
                "log_schema_version": "5",
            },
        }

        run_id, iter_id, decision_id, parent_decision_id, schema_version = MODULE._extract_correlation_fields(row)

        self.assertEqual(run_id, "run-top")
        self.assertEqual(iter_id, "iter-top")
        self.assertEqual(decision_id, "decision-top")
        self.assertEqual(parent_decision_id, "parent-top")
        self.assertEqual(schema_version, 9)

    def test_extract_correlation_fields_falls_back_to_metadata(self) -> None:
        row = {
            "metadata": {
                "run_id": "run-meta",
                "iter_id": "iter-meta",
                "decision_id": "decision-meta",
                "parent_decision_id": "parent-meta",
                "log_schema_version": "4",
            }
        }

        run_id, iter_id, decision_id, parent_decision_id, schema_version = MODULE._extract_correlation_fields(row)

        self.assertEqual(run_id, "run-meta")
        self.assertEqual(iter_id, "iter-meta")
        self.assertEqual(decision_id, "decision-meta")
        self.assertEqual(parent_decision_id, "parent-meta")
        self.assertEqual(schema_version, 4)

    def test_ensure_sqlite_schema_migrates_existing_table(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "ingest.sqlite3"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    """
                    CREATE TABLE jsonl_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_file TEXT NOT NULL,
                        source_rel TEXT NOT NULL,
                        line_no INTEGER NOT NULL,
                        ingested_at TEXT NOT NULL,
                        payload_sha1 TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        UNIQUE(source_file, line_no)
                    )
                    """
                )
                MODULE._ensure_sqlite_schema(conn, "jsonl_records")
                cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(jsonl_records)")}
            finally:
                conn.close()

        self.assertIn("run_id", cols)
        self.assertIn("iter_id", cols)
        self.assertIn("decision_id", cols)
        self.assertIn("parent_decision_id", cols)
        self.assertIn("log_schema_version", cols)

    def test_sync_file_to_sqlite_writes_correlation_columns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path = root / "ingest.sqlite3"
            jsonl_path = root / "decisions" / "test.jsonl"
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "run_id": "run-1",
                                "iter_id": "run-1:1",
                                "decision_id": "d-1",
                                "parent_decision_id": "p-1",
                                "log_schema_version": 7,
                                "value": 123,
                            }
                        ),
                        "{not-json}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            conn = sqlite3.connect(str(db_path))
            try:
                MODULE._ensure_sqlite_schema(conn, "jsonl_records")
                result = MODULE._sync_file_to_sqlite(
                    conn,
                    "jsonl_records",
                    root,
                    jsonl_path,
                    start_line=0,
                    start_offset_bytes=0,
                    dry_run=False,
                    lock_retries=0,
                    lock_retry_delay_seconds=0.01,
                    latency_all=None,
                    latency_stream=None,
                    invalid_log_path=None,
                    invalid_sample_limit=0,
                    run_id="",
                    iter_id="",
                )
                conn.commit()
                row = conn.execute(
                    "SELECT run_id, iter_id, decision_id, parent_decision_id, log_schema_version FROM jsonl_records"
                ).fetchone()
            finally:
                conn.close()

        self.assertEqual(result["inserted"], 1)
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["last_line"], 2)
        self.assertGreater(result["last_offset_bytes"], 0)
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "run-1")
        self.assertEqual(row[1], "run-1:1")
        self.assertEqual(row[2], "d-1")
        self.assertEqual(row[3], "p-1")
        self.assertEqual(int(row[4]), 7)

    def test_discover_jsonl_files_prioritizes_decision_streams(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data").mkdir(parents=True, exist_ok=True)
            (root / "decisions").mkdir(parents=True, exist_ok=True)

            data_file = root / "data" / "misc.jsonl"
            decision_file = root / "decisions" / "trade_decisions_20260101.jsonl"
            data_file.write_text("{}\n", encoding="utf-8")
            time.sleep(0.01)
            decision_file.write_text("{}\n", encoding="utf-8")

            files = MODULE.discover_jsonl_files(root)
            rels = [str(p.relative_to(root)) for p in files]

        self.assertIn("data/misc.jsonl", rels)
        self.assertIn("decisions/trade_decisions_20260101.jsonl", rels)
        self.assertLess(rels.index("decisions/trade_decisions_20260101.jsonl"), rels.index("data/misc.jsonl"))

    def test_derive_start_cursor_resets_on_inode_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "file.jsonl"
            p.write_text("{}\n", encoding="utf-8")
            st = p.stat()

            progress = {
                "last_line": 10,
                "last_offset_bytes": 100,
                "mtime": st.st_mtime,
                "file_inode": int(st.st_ino) + 1,
                "file_size_bytes": st.st_size,
            }
            line, offset, reason = MODULE._derive_start_cursor(progress, st)

        self.assertEqual(line, 0)
        self.assertEqual(offset, 0)
        self.assertEqual(reason, "inode_changed")


if __name__ == "__main__":
    unittest.main()
