import importlib.util
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
