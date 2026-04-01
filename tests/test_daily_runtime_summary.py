import importlib.util
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "daily_runtime_summary.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("daily_runtime_summary", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load daily_runtime_summary module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DailyRuntimeSummaryTests(unittest.TestCase):
    def test_summarize_status_rows_counts_and_stale_windows(self) -> None:
        module = _load_module()
        base = datetime(2026, 3, 13, 12, 0, tzinfo=timezone.utc)
        rows = [
            {"timestamp_utc": base.isoformat(), "status": "SHADOW_ONLY"},
            {"timestamp_utc": (base + timedelta(seconds=60)).isoformat(), "status": "BLOCKED"},
            {
                "timestamp_utc": (base + timedelta(seconds=400)).isoformat(),
                "status": "DATA_ONLY_BLOCKED",
                "safety": {"market_data_only": True, "execution_enabled": False},
            },
        ]

        summary = module._summarize_status_rows(rows, stale_seconds=180, skipped_statuses={"BLOCKED", "DATA_ONLY_BLOCKED"})

        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["skipped_decisions"], 1)
        self.assertEqual(summary["observe_only_data_blocked"], 1)
        self.assertEqual(summary["status_counts"]["SHADOW_ONLY"], 1)
        self.assertEqual(summary["stale_windows"], 1)


if __name__ == "__main__":
    unittest.main()
