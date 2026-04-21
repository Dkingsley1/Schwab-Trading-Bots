import importlib.util
import json
import os
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
        self.assertIn("source_day_utc", cols)
        self.assertIn("source_stream", cols)
        self.assertIn("source_partition_key", cols)

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
                    "SELECT run_id, iter_id, decision_id, parent_decision_id, log_schema_version, source_day_utc, source_stream, source_partition_key FROM jsonl_records"
                ).fetchone()
            finally:
                conn.close()
            with sqlite3.connect(str(root / "governance" / "ops_data_plane.sqlite3")) as ops_conn:
                dead_letter_count = int(ops_conn.execute("SELECT COUNT(*) FROM ingest_dead_letters").fetchone()[0] or 0)
                schema_drift_count = int(ops_conn.execute("SELECT COUNT(*) FROM schema_drift_events").fetchone()[0] or 0)
                watermark = ops_conn.execute(
                    "SELECT watermark_value FROM source_watermarks WHERE collector_key='jsonl_sql' AND entity_key=?",
                    ("decisions/test.jsonl",),
                ).fetchone()

        self.assertEqual(result["inserted"], 1)
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["last_line"], 2)
        self.assertGreater(result["last_offset_bytes"], 0)
        self.assertEqual(result["ops_write_failures"], 0)
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "run-1")
        self.assertEqual(row[1], "run-1:1")
        self.assertEqual(row[2], "d-1")
        self.assertEqual(row[3], "p-1")
        self.assertEqual(int(row[4]), 7)
        self.assertEqual(row[6], "decisions")
        self.assertTrue(str(row[7]).startswith(str(row[5])))
        self.assertEqual(dead_letter_count, 1)
        self.assertEqual(schema_drift_count, 1)
        self.assertIsNotNone(watermark)

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

    def test_sync_file_to_sqlite_respects_max_lines_per_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path = root / "ingest.sqlite3"
            jsonl_path = root / "governance" / "channels" / "runtime" / "default_crypto_schwab" / "runtime_20260101.jsonl"
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_path.write_text("\n".join(json.dumps({"value": idx}) for idx in range(5)) + "\n", encoding="utf-8")

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
                    max_lines_per_file=2,
                )
                row_count = conn.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()[0]
            finally:
                conn.close()

        self.assertEqual(result["inserted"], 2)
        self.assertEqual(result["last_line"], 2)
        self.assertGreater(result["last_offset_bytes"], 0)
        self.assertEqual(int(row_count), 2)

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

    def test_discover_jsonl_files_writes_discovery_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "decisions" / "trade_decisions_20260101.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

            files = MODULE.discover_jsonl_files(root)
            manifest_path = root / "governance" / "health" / "jsonl_discovery_manifest_latest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(files, [path])
        self.assertEqual(payload["file_count"], 1)
        self.assertEqual(payload["files"][0]["source_rel"], "decisions/trade_decisions_20260101.jsonl")
        self.assertEqual(payload["files"][0]["temperature"], "hot")

    def test_discover_jsonl_files_skips_redundant_legacy_hot_channel_logs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            legacy_runtime = root / "governance" / "events" / "runtime_events_20260417.jsonl"
            legacy_api = root / "governance" / "events" / "api_calls_default_crypto_coinbase_20260417.jsonl"
            channel_runtime = root / "governance" / "channels" / "runtime" / "default_equities_schwab" / "runtime_20260417.jsonl"
            channel_api = root / "governance" / "channels" / "api" / "default_crypto_schwab" / "api_20260417.jsonl"
            for path in (legacy_runtime, legacy_api, channel_runtime, channel_api):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")

            files = MODULE.discover_jsonl_files(root)
            rels = {str(p.relative_to(root)) for p in files}

        self.assertIn("governance/channels/runtime/default_equities_schwab/runtime_20260417.jsonl", rels)
        self.assertIn("governance/channels/api/default_crypto_schwab/api_20260417.jsonl", rels)
        self.assertNotIn("governance/events/runtime_events_20260417.jsonl", rels)
        self.assertNotIn("governance/events/api_calls_default_crypto_coinbase_20260417.jsonl", rels)

    def test_classify_stream_treats_decision_channels_and_schema_violations_separately(self) -> None:
        self.assertEqual(
            MODULE._classify_stream("governance/channels/decision/intraday_aggressive_equities_schwab/decision_20260417.jsonl"),
            "decisions",
        )
        self.assertEqual(
            MODULE._classify_stream("governance/events/channel_schema_violations_20260417.jsonl"),
            "schema_violations",
        )

    def test_json_file_stream_classifies_external_context_and_event_store(self) -> None:
        self.assertEqual(
            MODULE._json_file_stream("exports/external_context/sec_edgar_latest.json"),
            "external_context",
        )
        self.assertEqual(
            MODULE._json_file_stream("exports/external_feeds/tradingeconomics/latest.json"),
            "external_feeds",
        )
        self.assertEqual(
            MODULE._json_file_stream("governance/feature_store/latest.json"),
            "feature_store",
        )
        self.assertEqual(
            MODULE._json_file_stream("governance/health/point_in_time_event_store_latest.json"),
            "event_store",
        )

    def test_json_file_filters_respect_include_streams(self) -> None:
        matched = MODULE._matches_rel_filters(
            source_rel="exports/external_context/sec_edgar_latest.json",
            stream=MODULE._json_file_stream("exports/external_context/sec_edgar_latest.json"),
            include_streams=["external_context"],
            exclude_streams=[],
            path_contains=[],
            path_not_contains=[],
        )
        filtered = MODULE._matches_rel_filters(
            source_rel="exports/external_context/sec_edgar_latest.json",
            stream=MODULE._json_file_stream("exports/external_context/sec_edgar_latest.json"),
            include_streams=["feature_store"],
            exclude_streams=[],
            path_contains=[],
            path_not_contains=[],
        )

        self.assertTrue(matched)
        self.assertFalse(filtered)

    def test_discover_json_files_includes_external_context_and_feature_store(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            external_context = root / "exports" / "external_context" / "sec_edgar_latest.json"
            feature_store = root / "governance" / "feature_store" / "latest.json"
            external_context.parent.mkdir(parents=True, exist_ok=True)
            feature_store.parent.mkdir(parents=True, exist_ok=True)
            external_context.write_text("{}", encoding="utf-8")
            feature_store.write_text("{}", encoding="utf-8")

            files = MODULE.discover_json_files(root)
            rels = {str(path.relative_to(root)) for path in files}

        self.assertIn("exports/external_context/sec_edgar_latest.json", rels)
        self.assertIn("governance/feature_store/latest.json", rels)

    def test_prioritize_jsonl_files_prefers_recent_hot_files_over_stale_hot_backlog(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fresh = root / "decisions" / "shadow_aggressive_equities" / "trade_decisions_20260417.jsonl"
            stale = root / "decisions" / "shadow_conservative_equities" / "trade_decisions_20260414.jsonl"
            fresh.parent.mkdir(parents=True, exist_ok=True)
            stale.parent.mkdir(parents=True, exist_ok=True)
            fresh.write_text("{}\n", encoding="utf-8")
            stale.write_text("{}\n", encoding="utf-8")

            now = time.time()
            old = now - (3 * 24 * 60 * 60)
            os.utime(fresh, (now, now))
            os.utime(stale, (old, old))

            prioritized = MODULE._prioritize_jsonl_files_by_pending_bytes(
                [stale, fresh],
                project_root=root,
                sqlite_state={},
            )
            rels = [str(path.relative_to(root)) for path in prioritized]

        self.assertEqual(rels[0], "decisions/shadow_aggressive_equities/trade_decisions_20260417.jsonl")

    def test_prioritize_jsonl_files_prefers_current_day_filename_over_newer_mtime_on_stale_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            current_day = root / "governance" / "shadow_conservative_equities" / "master_control_20260417.jsonl"
            stale_name = root / "governance" / "shadow_conservative_equities" / "master_control_20260415.jsonl"
            current_day.parent.mkdir(parents=True, exist_ok=True)
            current_day.write_text("{}\n", encoding="utf-8")
            stale_name.write_text("{}\n", encoding="utf-8")

            now = time.time()
            os.utime(stale_name, (now, now))
            os.utime(current_day, (now - 60, now - 60))

            prioritized = MODULE._prioritize_jsonl_files_by_pending_bytes(
                [stale_name, current_day],
                project_root=root,
                sqlite_state={},
            )
            rels = [str(path.relative_to(root)) for path in prioritized]

        self.assertEqual(rels[0], "governance/shadow_conservative_equities/master_control_20260417.jsonl")

    def test_discover_jsonl_files_refreshes_when_new_file_appears(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "decisions" / "trade_decisions_20260101.jsonl"
            first.parent.mkdir(parents=True, exist_ok=True)
            first.write_text("{}\n", encoding="utf-8")

            original = os.environ.get("JSONL_DISCOVERY_MANIFEST_MAX_AGE_SECONDS")
            os.environ["JSONL_DISCOVERY_MANIFEST_MAX_AGE_SECONDS"] = "3600"
            try:
                first_pass = MODULE.discover_jsonl_files(root)
                second = root / "decisions" / "trade_decisions_20260102.jsonl"
                second.write_text("{}\n", encoding="utf-8")
                second_pass = MODULE.discover_jsonl_files(root)
            finally:
                if original is None:
                    os.environ.pop("JSONL_DISCOVERY_MANIFEST_MAX_AGE_SECONDS", None)
                else:
                    os.environ["JSONL_DISCOVERY_MANIFEST_MAX_AGE_SECONDS"] = original

        self.assertEqual(first_pass, [first])
        self.assertEqual(second_pass[0], second)
        self.assertEqual(second_pass[1], first)

    def test_prioritize_jsonl_files_keeps_hot_lane_ahead_of_deferred_and_cold(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            decision_file = root / "decisions" / "trade_decisions_20260101.jsonl"
            deferred_file = root / "governance" / "channels" / "runtime" / "runtime_events_20260101.jsonl"
            cold_file = root / "governance" / "shadow_intraday_aggressive_equities" / "shadow_pnl_attribution_20260101.jsonl"
            for path in (decision_file, deferred_file, cold_file):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")

            ordered = MODULE._prioritize_jsonl_files_by_pending_bytes(
                [cold_file, deferred_file, decision_file],
                project_root=root,
                sqlite_state={},
            )
            rels = [str(path.relative_to(root)) for path in ordered]

        self.assertEqual(rels[0], "decisions/trade_decisions_20260101.jsonl")
        self.assertEqual(rels[-1], "governance/shadow_intraday_aggressive_equities/shadow_pnl_attribution_20260101.jsonl")

    def test_record_top_pending_includes_storage_and_stale_labels(self) -> None:
        rows = []

        MODULE._record_top_pending(
            rows,
            source_rel="governance/channels/runtime/runtime_events_20260101.jsonl",
            pending_lines=42,
            oldest_age_seconds=90000.0,
            total_lines=120,
            last_line=78,
            top_n=5,
        )

        self.assertEqual(rows[0]["storage_temperature"], "warm")
        self.assertEqual(rows[0]["storage_tier"], "primary_warm")
        self.assertEqual(rows[0]["ingestion_lane"], "deferred_lane")
        self.assertEqual(rows[0]["stale_age_bucket"], "stale_lt_7d")

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

    def test_limit_prioritized_jsonl_files_keeps_cold_lane_out_when_core_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            core = root / "decisions" / "shadow_intraday_aggressive_equities" / "trade_decisions_20260329.jsonl"
            cold = root / "governance" / "shadow_intraday_aggressive_equities" / "shadow_pnl_attribution_20260329.jsonl"
            for path in [core, cold]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(("{}\n" * 10), encoding="utf-8")

            kept = MODULE._limit_prioritized_jsonl_files(
                [core, cold],
                project_root=root,
                max_files=2,
                max_deferred_files=1,
            )

        rels = [str(p.relative_to(root)) for p in kept]
        self.assertEqual(rels, ["decisions/shadow_intraday_aggressive_equities/trade_decisions_20260329.jsonl"])

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

    def test_main_marks_mysql_disabled_when_running_sqlite_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sqlite_db = root / "data" / "jsonl_link.sqlite3"
            health_file = root / "governance" / "health" / "jsonl_sql_ingestion_health_latest.json"

            original_argv = list(os.sys.argv)
            try:
                os.sys.argv = [
                    "link_jsonl_to_sql.py",
                    "--project-root",
                    str(root),
                    "--mode",
                    "sqlite",
                    "--sqlite-db",
                    str(sqlite_db),
                    "--health-file",
                    str(health_file),
                ]
                rc = MODULE.main()
            finally:
                os.sys.argv = original_argv

            payload = json.loads(health_file.read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertTrue(payload["sqlite"]["enabled"])
        self.assertEqual(payload["sqlite"]["status"], "active")
        self.assertFalse(payload["mysql"]["enabled"])
        self.assertEqual(payload["mysql"]["status"], "disabled_by_link_mode")
        self.assertFalse(payload["sinks"]["mysql"]["enabled"])
        self.assertEqual(payload["sinks"]["mysql"]["status"], "disabled_by_link_mode")


if __name__ == "__main__":
    unittest.main()
