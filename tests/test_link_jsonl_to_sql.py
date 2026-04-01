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

    def test_sync_file_to_sqlite_checkpoints_within_large_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path = root / "ingest.sqlite3"
            jsonl_path = root / "decisions" / "large.jsonl"
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                json.dumps({"run_id": "run-1", "iter_id": "iter-1", "decision_id": f"d-{idx}", "value": idx})
                for idx in range(2001)
            ]
            jsonl_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            checkpoints = []
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
                    checkpoint_every_lines=1000,
                    checkpoint_cb=lambda payload: checkpoints.append(dict(payload)),
                )
                row_count = conn.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()[0]
            finally:
                conn.close()

        self.assertEqual(result["inserted"], 2001)
        self.assertEqual(result["invalid"], 0)
        self.assertEqual(result["last_line"], 2001)
        self.assertEqual(int(row_count), 2001)
        self.assertGreaterEqual(len(checkpoints), 2)
        self.assertEqual(int(checkpoints[0]["last_line"]), 1000)
        self.assertEqual(int(checkpoints[-1]["last_line"]), 2001)

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

    def test_prioritize_jsonl_files_by_pending_bytes_prefers_largest_backlog(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gate_file = root / "governance" / "channels" / "gate" / "default_crypto_schwab" / "gate_20260329.jsonl"
            api_file = root / "governance" / "events" / "api_calls_default_crypto_coinbase_20260329.jsonl"
            gate_file.parent.mkdir(parents=True, exist_ok=True)
            api_file.parent.mkdir(parents=True, exist_ok=True)
            gate_file.write_text(("{}\n" * 1000), encoding="utf-8")
            api_file.write_text(("{}\n" * 100), encoding="utf-8")

            prioritized = MODULE._prioritize_jsonl_files_by_pending_bytes(
                [api_file, gate_file],
                project_root=root,
                sqlite_state={
                    str(api_file.relative_to(root)): {
                        "last_offset_bytes": api_file.stat().st_size,
                    },
                    str(gate_file.relative_to(root)): {
                        "last_offset_bytes": 0,
                    },
                },
            )

        rels = [str(p.relative_to(root)) for p in prioritized]
        self.assertEqual(rels[0], "governance/channels/gate/default_crypto_schwab/gate_20260329.jsonl")

    def test_prioritize_jsonl_files_by_pending_bytes_prefers_hot_governance_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gate_logs = root / "governance" / "events" / "gate_logs_default_crypto_coinbase_20260329.jsonl"
            runtime_file = root / "governance" / "channels" / "runtime" / "default_crypto_schwab" / "runtime_20260329.jsonl"
            gate_logs.parent.mkdir(parents=True, exist_ok=True)
            runtime_file.parent.mkdir(parents=True, exist_ok=True)
            gate_logs.write_text(("{}\n" * 400), encoding="utf-8")
            runtime_file.write_text(("{}\n" * 1000), encoding="utf-8")

            prioritized = MODULE._prioritize_jsonl_files_by_pending_bytes(
                [runtime_file, gate_logs],
                project_root=root,
                sqlite_state={
                    str(runtime_file.relative_to(root)): {"last_offset_bytes": 0},
                    str(gate_logs.relative_to(root)): {"last_offset_bytes": 0},
                },
            )

        rels = [str(p.relative_to(root)) for p in prioritized]
        self.assertEqual(rels[0], "governance/events/gate_logs_default_crypto_coinbase_20260329.jsonl")

    def test_prioritize_jsonl_files_by_pending_bytes_deprioritizes_deferred_analytics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            decision_file = root / "decisions" / "shadow_crypto" / "trade_decisions_20260329.jsonl"
            pnl_file = root / "governance" / "shadow_crypto" / "shadow_pnl_attribution_20260329.jsonl"
            decision_file.parent.mkdir(parents=True, exist_ok=True)
            pnl_file.parent.mkdir(parents=True, exist_ok=True)
            decision_file.write_text(("{}\n" * 100), encoding="utf-8")
            pnl_file.write_text(("{}\n" * 1000), encoding="utf-8")

            prioritized = MODULE._prioritize_jsonl_files_by_pending_bytes(
                [pnl_file, decision_file],
                project_root=root,
                sqlite_state={
                    str(decision_file.relative_to(root)): {"last_offset_bytes": 0},
                    str(pnl_file.relative_to(root)): {"last_offset_bytes": 0},
                },
            )

        rels = [str(p.relative_to(root)) for p in prioritized]
        self.assertEqual(rels[0], "decisions/shadow_crypto/trade_decisions_20260329.jsonl")

    def test_limit_prioritized_jsonl_files_reserves_budget_for_core_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            core_a = root / "decisions" / "shadow_crypto" / "trade_decisions_20260329.jsonl"
            core_b = root / "decision_explanations" / "shadow_crypto" / "decision_explanations_20260329.jsonl"
            deferred = root / "governance" / "shadow_crypto" / "shadow_pnl_attribution_20260329.jsonl"
            for path in [core_a, core_b, deferred]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(("{}\n" * 10), encoding="utf-8")

            kept = MODULE._limit_prioritized_jsonl_files(
                [core_a, core_b, deferred],
                project_root=root,
                max_files=2,
                max_deferred_files=1,
            )

        rels = [str(p.relative_to(root)) for p in kept]
        self.assertEqual(
            rels,
            [
                "decisions/shadow_crypto/trade_decisions_20260329.jsonl",
                "decision_explanations/shadow_crypto/decision_explanations_20260329.jsonl",
            ],
        )

    def test_limit_prioritized_jsonl_files_allows_deferred_when_no_core_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            deferred_a = root / "governance" / "shadow_crypto" / "shadow_pnl_attribution_20260329.jsonl"
            deferred_b = root / "governance" / "events" / "api_calls_default_crypto_coinbase_20260329.jsonl"
            for path in [deferred_a, deferred_b]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(("{}\n" * 10), encoding="utf-8")

            kept = MODULE._limit_prioritized_jsonl_files(
                [deferred_a, deferred_b],
                project_root=root,
                max_files=1,
                max_deferred_files=0,
            )

        rels = [str(p.relative_to(root)) for p in kept]
        self.assertEqual(rels, ["governance/shadow_crypto/shadow_pnl_attribution_20260329.jsonl"])

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
