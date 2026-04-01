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
            module._is_deferred_backpressure_file(
                "governance/shadow_crypto/shadow_pnl_attribution_20260329.jsonl"
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


if __name__ == "__main__":
    unittest.main()
