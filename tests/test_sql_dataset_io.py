import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sql_dataset_io.py"
SPEC = importlib.util.spec_from_file_location("sql_dataset_io_module", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SqlDatasetIoTests(unittest.TestCase):
    def test_split_paths_uses_sqlite_when_source_rel_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            decision_dir = root / "decision_explanations" / "shadow"
            decision_dir.mkdir(parents=True)
            present_path = decision_dir / "decision_explanations_present.jsonl"
            missing_path = decision_dir / "decision_explanations_missing.jsonl"
            present_path.write_text("{}\n", encoding="utf-8")
            missing_path.write_text("{}\n", encoding="utf-8")

            sqlite_path = root / "jsonl_link.sqlite3"
            conn = sqlite3.connect(str(sqlite_path))
            try:
                conn.execute(
                    "CREATE TABLE jsonl_records (source_rel TEXT, line_no INTEGER, payload_json TEXT)"
                )
                conn.execute(
                    "INSERT INTO jsonl_records VALUES (?, ?, ?)",
                    (
                        "decision_explanations/shadow/decision_explanations_present.jsonl",
                        1,
                        json.dumps({"ok": True}),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            sql_rels, fallbacks = MODULE.split_paths_by_sqlite_coverage(
                project_root=root,
                paths=[present_path, missing_path],
                sqlite_path=sqlite_path,
            )

        self.assertEqual(sql_rels, ["decision_explanations/shadow/decision_explanations_present.jsonl"])
        self.assertEqual(fallbacks, [missing_path])

    def test_split_paths_falls_back_when_external_sqlite_query_fails(self) -> None:
        class FailingConn:
            def execute(self, *_args, **_kwargs):
                raise sqlite3.OperationalError("unable to open database file")

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "governance" / "shadow" / "master_control_latest.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("{}\n", encoding="utf-8")
            sqlite_path = root / "jsonl_link.sqlite3"
            sqlite_path.write_text("", encoding="utf-8")

            with patch.object(MODULE, "_connect_readonly", return_value=FailingConn()):
                sql_rels, fallbacks = MODULE.split_paths_by_sqlite_coverage(
                    project_root=root,
                    paths=[path],
                    sqlite_path=sqlite_path,
                )

        self.assertEqual(sql_rels, [])
        self.assertEqual(fallbacks, [path])

    def test_iter_like_patterns_uses_literal_prefix_and_reads_matching_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sqlite_path = root / "jsonl_link.sqlite3"
            conn = sqlite3.connect(str(sqlite_path))
            try:
                conn.execute(
                    "CREATE TABLE jsonl_records (source_rel TEXT, line_no INTEGER, payload_json TEXT)"
                )
                conn.execute(
                    "CREATE INDEX idx_jsonl_records_source_rel_line ON jsonl_records(source_rel, line_no)"
                )
                conn.execute(
                    "INSERT INTO jsonl_records VALUES (?, ?, ?)",
                    (
                        "decisions/paper/trade_decisions_20260531.jsonl.gz",
                        1,
                        json.dumps({"symbol": "SOXX"}),
                    ),
                )
                conn.execute(
                    "INSERT INTO jsonl_records VALUES (?, ?, ?)",
                    (
                        "governance/events/unrelated.jsonl",
                        1,
                        json.dumps({"symbol": "IGNORE"}),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            rows = list(
                MODULE.iter_sqlite_jsonl_rows_by_like_patterns(
                    sqlite_path=sqlite_path,
                    like_patterns=["decisions/%/trade_decisions_20260531.jsonl%"],
                )
            )

        self.assertEqual(rows, [{"symbol": "SOXX"}])


if __name__ == "__main__":
    unittest.main()
