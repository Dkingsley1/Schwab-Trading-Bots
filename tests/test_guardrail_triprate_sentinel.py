import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class GuardrailTriprateSentinelTests(unittest.TestCase):
    def test_bounded_scan_uses_recent_tail_and_skips_stale_files(self) -> None:
        module = _load_module("guardrail_triprate_sentinel_test", ROOT / "scripts" / "guardrail_triprate_sentinel.py")
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            module.PROJECT_ROOT = project_root
            shadow = project_root / "governance" / "shadow_alpha"
            shadow.mkdir(parents=True)

            recent_ts = datetime.now(timezone.utc).isoformat()
            old_ts = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
            stale_file = shadow / "master_control_20260429.jsonl"
            stale_file.write_text(
                json.dumps({"timestamp_utc": old_ts, "reason": "feature_freshness_guard"}) + "\n",
                encoding="utf-8",
            )
            stale_mtime = time.time() - (2 * 24 * 60 * 60)
            os.utime(stale_file, (stale_mtime, stale_mtime))

            recent_file = shadow / "master_control_20260501.jsonl"
            recent_file.write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp_utc": old_ts, "note": "old"}),
                        json.dumps({"timestamp_utc": recent_ts, "recommendations": ["feature_freshness_guard"]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot_file = shadow / "snapshot_debug_20260501.jsonl"
            snapshot_file.write_text(
                json.dumps({"timestamp_utc": recent_ts, "reason": "circuit_open_skip"}) + "\n",
                encoding="utf-8",
            )

            old_argv = sys.argv[:]
            try:
                sys.argv = [
                    str(ROOT / "scripts" / "guardrail_triprate_sentinel.py"),
                    "--hours",
                    "6",
                    "--max-files",
                    "10",
                    "--tail-bytes",
                    "20000",
                    "--json",
                ]
                rc = module.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 2)
            payload = json.loads((project_root / "governance" / "health" / "guardrail_triprate_latest.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["bounded_scan"])
            self.assertEqual(payload["samples"], 1)
            self.assertEqual(payload["trip_breakdown"]["feature_freshness_guard"], 1)
            self.assertEqual(payload["trip_breakdown"]["circuit_open_skip"], 1)
            self.assertGreaterEqual(payload["scan_stats"]["files_skipped_mtime"], 1)


if __name__ == "__main__":
    unittest.main()
