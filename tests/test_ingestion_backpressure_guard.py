import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ingestion_backpressure_guard.py"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module():
    spec = importlib.util.spec_from_file_location("ingestion_backpressure_guard", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load ingestion_backpressure_guard module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IngestionBackpressureGuardTests(unittest.TestCase):
    def test_should_ignore_internal_ingest_journals(self) -> None:
        module = _load_module()

        self.assertTrue(
            module._should_ignore_backpressure_file(
                "governance/health/jsonl_ingest_batch_journal_governance_latest.jsonl"
            )
        )
        self.assertTrue(
            module._should_ignore_backpressure_file(
                "governance/events/jsonl_ingest_batches_governance_20260327.jsonl"
            )
        )
        self.assertTrue(
            module._should_ignore_backpressure_file(
                "governance/training/raw_training_source_queue_latest.jsonl"
            )
        )
        self.assertTrue(
            module._should_ignore_backpressure_file(
                "governance/training/raw_training_eligible_source_queue_latest.jsonl"
            )
        )
        self.assertTrue(
            module._should_ignore_backpressure_file(
                "governance/shadow_aggressive_equities/runtime_telemetry.jsonl"
            )
        )
        self.assertFalse(
            module._should_ignore_backpressure_file(
                "governance/events/auth_events_20260327.jsonl"
            )
        )

    def test_deferred_backpressure_tracks_analytics_streams_separately(self) -> None:
        module = _load_module()

        self.assertTrue(
            module._is_support_backpressure_file(
                "governance/watchdog/failover_events.jsonl"
            )
        )
        self.assertTrue(
            module._is_deferred_backpressure_file(
                "governance/watchdog/pager_alerts.jsonl"
            )
        )
        self.assertFalse(
            module._is_support_backpressure_file(
                "governance/events/gate_logs_default_20260329.jsonl"
            )
        )
        self.assertTrue(
            module._is_deferred_backpressure_file(
                "governance/shadow_crypto/shadow_pnl_attribution_20260329.jsonl"
            )
        )
        self.assertTrue(
            module._is_cold_backpressure_file(
                "governance/shadow_crypto/shadow_pnl_attribution_20260329.jsonl"
            )
        )
        self.assertTrue(
            module._is_cold_backpressure_file(
                "governance/health/platform_control_plane_20260406.jsonl"
            )
        )
        self.assertTrue(
            module._is_deferred_backpressure_file(
                "governance/events/api_calls_default_crypto_coinbase_20260329.jsonl"
            )
        )
        self.assertTrue(
            module._is_deferred_backpressure_file(
                "governance/channels/ingress/default_crypto_coinbase/ingress_20260329.jsonl"
            )
        )
        self.assertTrue(
            module._is_deferred_backpressure_file(
                "governance/channels/loop_state/default_crypto_schwab/loop_state_20260329.jsonl"
            )
        )
        self.assertTrue(
            module._is_deferred_backpressure_file(
                "decision_explanations/shadow_intraday_aggressive_equities/decision_explanations_20260329.jsonl"
            )
        )
        self.assertTrue(
            module._is_deferred_backpressure_file(
                "data/stale_stage/decision_explanations/project/decision_explanations/shadow_crypto_futures_crypto/decision_explanations_20260413.jsonl"
            )
        )
        self.assertTrue(
            module._is_cold_backpressure_file(
                "data/stale_stage/decision_explanations/project/decision_explanations/shadow_crypto_futures_crypto/decision_explanations_20260413.jsonl"
            )
        )
        self.assertTrue(
            module._is_stale_stage_backpressure_file(
                "data/stale_stage/decision_explanations/project/decision_explanations/shadow_crypto_futures_crypto/decision_explanations_20260413.jsonl"
            )
        )
        self.assertTrue(
            module._is_deferred_backpressure_file(
                "governance/channels/risk/intraday_aggressive_equities_schwab/risk_20260329.jsonl"
            )
        )
        self.assertTrue(
            module._is_deferred_backpressure_file(
                "governance/events/loop_state_default_crypto_coinbase_20260329.jsonl"
            )
        )
        self.assertFalse(
            module._is_deferred_backpressure_file(
                "governance/channels/decision/default_crypto_schwab/decision_20260329.jsonl"
            )
        )

    def test_age_pressure_ignores_tiny_tail(self) -> None:
        module = _load_module()

        self.assertFalse(
            module._age_pressure_triggered(
                oldest_pending_age_seconds=900.0,
                pending_lines=10,
                threshold_seconds=240.0,
                min_pending_lines=100,
            )
        )
        self.assertTrue(
            module._age_pressure_triggered(
                oldest_pending_age_seconds=900.0,
                pending_lines=200,
                threshold_seconds=240.0,
                min_pending_lines=100,
            )
        )

    def test_resolve_sqlite_state_prefers_shard_progress(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            governance_root = project_root / "governance"
            shard_root = governance_root / "sql_link_shards"
            shard_root.mkdir(parents=True)

            rel = "decision_explanations/paper/decision_explanations_20260326.jsonl"
            (governance_root / "jsonl_sql_link_state.json").write_text(
                json.dumps(
                    {
                        "sqlite": {
                            rel: {
                                "last_line": 50,
                                "file_size_bytes": 500,
                                "mtime": 100.0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (shard_root / "jsonl_sql_link_state_trading.json").write_text(
                json.dumps(
                    {
                        "sqlite": {
                            rel: {
                                "last_line": 120,
                                "file_size_bytes": 1200,
                                "mtime": 200.0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            sqlite_state, state_files, state_mode = module._resolve_sqlite_state(project_root, None)

            self.assertEqual(state_mode, "sharded_merged")
            self.assertTrue(any(path.endswith("jsonl_sql_link_state_trading.json") for path in state_files))
            self.assertEqual(sqlite_state[rel]["last_line"], 120)

    def test_large_file_uses_progress_density_estimate(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "large.jsonl"
            path.write_bytes(b"x" * 200)
            st = path.stat()

            total = module._estimated_total_lines(
                path,
                st,
                {"last_line": 50, "file_size_bytes": 100},
                max_exact_bytes=16,
                sample_bytes=32,
            )

            self.assertEqual(total, 100)

    def test_large_file_sampling_estimate_without_progress(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sampled.jsonl"
            path.write_text(("row\n" * 50), encoding="utf-8")
            st = path.stat()

            total = module._estimated_total_lines(
                path,
                st,
                {},
                max_exact_bytes=16,
                sample_bytes=64,
            )

            self.assertGreater(total, 0)

    def test_large_sparse_line_file_does_not_fall_back_to_tiny_rows(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sparse.jsonl"
            path.write_bytes(b"x" * 16384)
            st = path.stat()

            total = module._estimated_total_lines(
                path,
                st,
                {},
                max_exact_bytes=16,
                sample_bytes=1024,
            )

            self.assertEqual(total, 1)

            detail = module._estimated_total_lines_detail(
                path,
                st,
                {},
                max_exact_bytes=16,
                sample_bytes=1024,
            )
            self.assertTrue(detail["sparse_large_line"])
            self.assertEqual(detail["line_estimate_method"], "sparse_no_newline_sample")

    def test_load_journal_progress_tracks_highest_checkpoint(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            health_root = project_root / "governance" / "health"
            health_root.mkdir(parents=True)
            rel = "governance/execution_lanes/execution_results_20260422.jsonl"
            journal = health_root / "jsonl_ingest_batch_journal_governance_latest.jsonl"
            journal.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event": "file_checkpoint",
                                "source_rel": rel,
                                "last_line": 1200,
                                "last_offset_bytes": 120000,
                                "timestamp_utc": "2026-04-22T23:58:00+00:00",
                            }
                        ),
                        json.dumps(
                            {
                                "event": "file_complete",
                                "source_rel": rel,
                                "last_line": 1600,
                                "last_offset_bytes": 200000,
                                "timestamp_utc": "2026-04-22T23:59:00+00:00",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            progress, sources = module._load_journal_progress(project_root)

            self.assertEqual(progress[rel]["last_line"], 1600)
            self.assertEqual(progress[rel]["last_offset_bytes"], 200000)
            self.assertEqual(len(sources), 1)

    def test_journal_reconciliation_recovers_missing_state_progress(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "execution_results_20260422.jsonl"
            path.write_text(("row\n" * 5000), encoding="utf-8")
            st = path.stat()

            reconciled_last_line, used = module._journal_reconciled_last_line(
                stat=st,
                state_last_line=0,
                journal_progress={
                    "last_line": 3200,
                    "last_offset_bytes": 6400,
                    "journal_timestamp_epoch": float(st.st_mtime) - 30.0,
                },
            )

            self.assertTrue(used)
            self.assertEqual(reconciled_last_line, 3200)

    def test_last_line_for_state_tolerates_subsecond_mtime_rounding(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trade_decisions_20260519.jsonl"
            path.write_text("row\n" * 10, encoding="utf-8")
            st = path.stat()

            last_line = module._last_line_for_state(
                "decisions/shadow_swing_aggressive_equities/trade_decisions_20260519.jsonl",
                st,
                {
                    "last_line": 10,
                    "last_offset_bytes": st.st_size,
                    "mtime": float(st.st_mtime) + 0.25,
                    "file_inode": st.st_ino,
                    "file_size_bytes": st.st_size,
                },
            )

            self.assertEqual(last_line, 10)

    def test_resolve_sqlite_state_prefers_newer_inode_progress_over_stale_higher_line_count(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            shard_root = project_root / "governance" / "sql_link_shards"
            shard_root.mkdir(parents=True)

            rel = "governance/channels/decision/default_crypto_schwab/decision_20260412.jsonl"
            (shard_root / "jsonl_sql_link_state_shadow.json").write_text(
                json.dumps(
                    {
                        "sqlite": {
                            rel: {
                                "last_line": 3453,
                                "file_size_bytes": 329446759,
                                "mtime": 1776007673.0,
                                "file_inode": 263998551,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (shard_root / "jsonl_sql_link_state_governance.json").write_text(
                json.dumps(
                    {
                        "sqlite": {
                            rel: {
                                "last_line": 1590,
                                "file_size_bytes": 151914337,
                                "mtime": 1776038014.0,
                                "file_inode": 265434204,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            sqlite_state, _, _ = module._resolve_sqlite_state(project_root, None)

            self.assertEqual(sqlite_state[rel]["last_line"], 1590)
            self.assertEqual(sqlite_state[rel]["file_inode"], 265434204)

    def test_resolve_sqlite_state_prefers_current_file_inode_over_stale_newer_inode(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            shard_root = project_root / "governance" / "sql_link_shards"
            shard_root.mkdir(parents=True)

            rel = "decisions/shadow_swing_aggressive_equities/trade_decisions_20260519.jsonl"
            source = project_root / rel
            source.parent.mkdir(parents=True)
            source.write_text("row\n" * 10, encoding="utf-8")
            st = source.stat()
            (shard_root / "jsonl_sql_link_state_aggressive_trading.json").write_text(
                json.dumps(
                    {
                        "sqlite": {
                            rel: {
                                "last_line": 10,
                                "last_offset_bytes": st.st_size,
                                "mtime": st.st_mtime,
                                "file_inode": st.st_ino,
                                "file_size_bytes": st.st_size,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (shard_root / "jsonl_sql_link_state_crypto_trading.json").write_text(
                json.dumps(
                    {
                        "sqlite": {
                            rel: {
                                "last_line": 11310,
                                "last_offset_bytes": 583807461,
                                "mtime": st.st_mtime + 1000.0,
                                "file_inode": st.st_ino + 99,
                                "file_size_bytes": 583807461,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            sqlite_state, _, _ = module._resolve_sqlite_state(project_root, None)

            self.assertEqual(sqlite_state[rel]["last_line"], 10)
            self.assertEqual(sqlite_state[rel]["file_inode"], st.st_ino)


if __name__ == "__main__":
    unittest.main()
