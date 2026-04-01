import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "sql_runtime_report.py"
PYTHON_BIN = PROJECT_ROOT / ".venv312" / "bin" / "python"


def _init_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE jsonl_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_rel TEXT NOT NULL,
            ingested_at TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    rows = [
        (
            "decision_explanations/shadow_crypto/decision_explanations_20260327.jsonl",
            "2026-03-27T03:40:00+00:00",
            json.dumps({"timestamp_utc": "2026-03-27T03:40:00+00:00", "status": "SHADOW_ONLY", "action": "HOLD", "symbol": "BTC"}),
        ),
        (
            "governance/shadow_crypto/master_control_20260327.jsonl",
            "2026-03-27T03:41:00+00:00",
            json.dumps({"timestamp_utc": "2026-03-27T03:41:00+00:00", "master_action": "allow"}),
        ),
        (
            "governance/watchdog/watchdog_events_20260327.jsonl",
            "2026-03-27T03:42:00+00:00",
            json.dumps(
                {
                    "timestamp_utc": "2026-03-27T03:42:00+00:00",
                    "targets": [
                        {"action": "hold", "note": "schwab"},
                        {"action": "hold", "note": "coinbase"},
                    ],
                }
            ),
        ),
    ]
    conn.executemany("INSERT INTO jsonl_records (source_rel, ingested_at, payload_json) VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()


class SqlRuntimeReportTests(unittest.TestCase):
    def test_fast_mode_skips_global_totals(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path = root / "jsonl_link.sqlite3"
            out_dir = root / "report"
            _init_db(db_path)

            proc = subprocess.run(
                [
                    str(PYTHON_BIN),
                    str(SCRIPT_PATH),
                    "--db",
                    str(db_path),
                    "--day",
                    "20260327",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = (out_dir / "runtime_sql_report.md").read_text(encoding="utf-8")
            self.assertIn("Global DB totals skipped in fast mode.", report)
            self.assertNotIn("Total rows in `jsonl_records`", report)


if __name__ == "__main__":
    unittest.main()
