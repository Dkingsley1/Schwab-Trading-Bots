import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "daily_state_snapshot_drill.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("daily_state_snapshot_drill", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load daily_state_snapshot_drill module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DailyStateSnapshotDrillTests(unittest.TestCase):
    def test_large_file_hash_only_mode(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            module.PROJECT_ROOT = root
            (root / "governance" / "watchdog").mkdir(parents=True, exist_ok=True)

            large_file = root / "large.bin"
            large_file.write_bytes(b"A" * 128)

            out_root = root / "exports" / "state_snapshot_drills"
            argv = [
                "daily_state_snapshot_drill.py",
                "--out-root",
                str(out_root),
                "--targets",
                str(large_file),
                "--max-copy-bytes",
                "16",
                "--json",
            ]
            with mock.patch.object(sys, "argv", argv):
                rc = module.main()

            self.assertEqual(rc, 0)
            latest = json.loads((out_root / "latest.json").read_text(encoding="utf-8"))
            row = latest["rows"][0]
            self.assertEqual(row["copy_mode"], "metadata_only_large_file")
            self.assertEqual(row["snapshot"], "")
            self.assertEqual(row["restored"], "")
            self.assertTrue(row["restore_ok"])

    def test_small_file_copy_restore_mode(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            module.PROJECT_ROOT = root
            (root / "governance" / "watchdog").mkdir(parents=True, exist_ok=True)

            small_file = root / "small.txt"
            small_file.write_text("hello", encoding="utf-8")

            out_root = root / "exports" / "state_snapshot_drills"
            argv = [
                "daily_state_snapshot_drill.py",
                "--out-root",
                str(out_root),
                "--targets",
                str(small_file),
                "--max-copy-bytes",
                "1024",
                "--json",
            ]
            with mock.patch.object(sys, "argv", argv):
                rc = module.main()

            self.assertEqual(rc, 0)
            latest = json.loads((out_root / "latest.json").read_text(encoding="utf-8"))
            row = latest["rows"][0]
            self.assertEqual(row["copy_mode"], "full_copy_restore")
            self.assertNotEqual(row["snapshot"], "")
            self.assertNotEqual(row["restored"], "")
            self.assertTrue(row["restore_ok"])


if __name__ == "__main__":
    unittest.main()
