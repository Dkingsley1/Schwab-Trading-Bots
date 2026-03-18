import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class SnapshotSyncPipelineTests(unittest.TestCase):
    def test_sync_snapshot_health_requires_full_debug_sync(self) -> None:
        module = _load_module("sync_snapshot_health_to_sql_test", ROOT / "scripts" / "sync_snapshot_health_to_sql.py")
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            health_root = project_root / "governance" / "health"
            _write_json(health_root / "snapshot_coverage_latest.json", {"timestamp_utc": "2026-03-11T00:00:00+00:00", "ok": True, "coverage_ratio": 1.0, "rows_scanned": 1, "rows_with_snapshot_id": 1})
            _write_json(health_root / "replay_preopen_sanity_latest.json", {"timestamp_utc": "2026-03-11T00:00:00+00:00", "ok": True, "decision": {"stale_windows": 0}, "governance": {"stale_windows": 0}, "thresholds": {"max_decision_stale_windows": 1, "max_governance_stale_windows": 1}})
            _write_json(health_root / "preopen_replay_drift_latest.json", {"timestamp_utc": "2026-03-11T00:00:00+00:00", "drift": {}, "thresholds": {}})
            _write_json(health_root / "data_source_divergence_latest.json", {"timestamp_utc": "2026-03-11T00:00:00+00:00"})
            _write_json(health_root / "guardrail_triprate_latest.json", {"timestamp_utc": "2026-03-11T00:00:00+00:00"})
            _write_json(health_root / "execution_queue_stress_latest.json", {"timestamp_utc": "2026-03-11T00:00:00+00:00"})
            _write_json(project_root / "exports" / "state_snapshot_drills" / "latest.json", {"timestamp_utc": "2026-03-11T00:00:00+00:00", "ok": True})

            snap_dir = project_root / "exports" / "debug_snapshots" / "20260311_000000"
            snap_dir.mkdir(parents=True, exist_ok=True)
            (snap_dir / "sample.json").write_text("{\"ok\": true}", encoding="utf-8")

            sqlite_path = project_root / "data" / "jsonl_link.sqlite3"
            out_path = project_root / "governance" / "health" / "snapshot_sql_sync_latest.json"

            old_argv = sys.argv[:]
            try:
                sys.argv = [
                    str(ROOT / "scripts" / "sync_snapshot_health_to_sql.py"),
                    "--project-root", str(project_root),
                    "--sqlite-path", str(sqlite_path),
                    "--no-persist",
                    "--require-full-debug-sync",
                    "--out-file", str(out_path),
                ]
                rc = module.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 2)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertFalse(bool(payload.get("debug_sync_ok", False)))
            coverage = payload.get("debug_snapshot_ingest_coverage") if isinstance(payload.get("debug_snapshot_ingest_coverage"), dict) else {}
            self.assertEqual(int(coverage.get("snapshot_total", 0)), 1)
            self.assertEqual(float(coverage.get("coverage_ratio", 1.0)), 0.0)

    def test_data_retention_policy_blocks_debug_snapshot_purge_without_training_success(self) -> None:
        module = _load_module("data_retention_policy_test", ROOT / "scripts" / "data_retention_policy.py")
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            module.PROJECT_ROOT = project_root

            snap_dir = project_root / "exports" / "debug_snapshots" / "20260310_000000"
            snap_dir.mkdir(parents=True, exist_ok=True)
            (snap_dir / "sample.json").write_text("{\"ok\": true}", encoding="utf-8")

            old_argv = sys.argv[:]
            try:
                sys.argv = [
                    str(ROOT / "scripts" / "data_retention_policy.py"),
                    "--apply",
                    "--debug-snapshots-days", "0",
                    "--debug-snapshots-keep", "0",
                    "--require-training-success",
                    "--json",
                ]
                rc = module.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 0)
            self.assertTrue(snap_dir.exists())
            payload = json.loads((project_root / "governance" / "health" / "data_retention_latest.json").read_text(encoding="utf-8"))
            debug = payload.get("debug_snapshots") if isinstance(payload.get("debug_snapshots"), dict) else {}
            self.assertEqual(int(debug.get("skipped_training_guard", 0)), 1)
            guard = debug.get("training_success_guard") if isinstance(debug.get("training_success_guard"), dict) else {}
            self.assertFalse(bool(guard.get("ok", True)))

    def test_data_retention_policy_allows_snapshot_guard_purge(self) -> None:
        module = _load_module("data_retention_policy_snapshot_test", ROOT / "scripts" / "data_retention_policy.py")
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            module.PROJECT_ROOT = project_root

            snap_dir = project_root / "exports" / "debug_snapshots" / "20260310_000000"
            snap_dir.mkdir(parents=True, exist_ok=True)
            (snap_dir / "sample.json").write_text("{\"ok\": true}", encoding="utf-8")

            _write_json(
                project_root / "governance" / "health" / "snapshot_training_coverage_latest.json",
                {
                    "timestamp_utc": "2026-03-11T12:00:00+00:00",
                    "all_snapshot_data_incorporated": True,
                    "snapshot_raw_sql_ingest_ratio": 1.0,
                    "snapshot_cov_fill_ratio": 1.0,
                    "snapshot_feature_coverage_ratio": 1.0,
                    "reason": "ok",
                },
            )

            old_argv = sys.argv[:]
            try:
                sys.argv = [
                    str(ROOT / "scripts" / "data_retention_policy.py"),
                    "--apply",
                    "--debug-snapshots-days", "0",
                    "--debug-snapshots-keep", "0",
                    "--require-snapshot-training-coverage",
                    "--json",
                ]
                rc = module.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 0)
            self.assertFalse(snap_dir.exists())


if __name__ == "__main__":
    unittest.main()
