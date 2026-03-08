import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.incident_auto_halt import evaluate_incident
from scripts.promotion_quality_gate import evaluate_quality
from scripts.replay_end_to_end_deterministic import run_replay


class OpsQualityIncidentTests(unittest.TestCase):
    def test_replay_hash_is_deterministic(self) -> None:
        payload = {
            "equity_proxy": 100000.0,
            "rows": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "score": 0.62,
                    "threshold": 0.55,
                    "features": {"volatility_1m": 0.01, "pct_from_close": 0.002, "mom_5m": 0.001},
                },
                {
                    "symbol": "MSFT",
                    "action": "SELL",
                    "score": 0.59,
                    "threshold": 0.55,
                    "features": {"volatility_1m": 0.012, "pct_from_close": -0.001, "mom_5m": -0.001},
                },
            ],
        }
        out1 = run_replay(payload)
        out2 = run_replay(payload)
        self.assertTrue(out1["ok"])
        self.assertEqual(out1["replay_hash"], out2["replay_hash"])
        self.assertEqual(out1["row_count"], 2)

    def test_reconciliation_slo_guard_breaches_on_high_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            in_file = td_path / "events.jsonl"
            now = datetime.now(timezone.utc).isoformat()
            rows = [
                {"timestamp_utc": now, "event": "position_reconcile", "status": "mismatch"},
                {"timestamp_utc": now, "event": "position_reconcile", "status": "ok"},
            ]
            in_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

            script = PROJECT_ROOT / "scripts" / "live_reconciliation_slo_guard.py"
            out_file = td_path / "out.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--in-file",
                    str(in_file),
                    "--out-file",
                    str(out_file),
                    "--lookback-minutes",
                    "60",
                    "--max-mismatch-rate",
                    "0.10",
                    "--alert-file",
                    str(td_path / "alerts.jsonl"),
                    "--latest-alert-file",
                    str(td_path / "latest_alert.json"),
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2)
            payload = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertFalse(payload["ok"])
            self.assertIn("mismatch_rate", payload["failed_checks"])

    def test_promotion_quality_evaluate_requires_all_gates(self) -> None:
        ok, failed, _ = evaluate_quality(
            {"promote_ok": True, "considered_bots": 40, "fail_share": 0.10},
            {"ok": True},
            {"ok": True},
            {"ok": False},
            {"ok": True},
            {"ok": True},
            max_fail_share=0.25,
            min_considered_bots=20,
            require_replay=True,
            require_reconciliation_slo=True,
        )
        self.assertFalse(ok)
        self.assertIn("leak_overfit_not_ok", failed)

    def test_incident_evaluate_fails_on_bad_quality_gate(self) -> None:
        ok, failed, detail = evaluate_incident(
            {"ok": True},
            {"ok": False},
            {"ok": True},
            require_daily_verify=True,
            require_quality_gate=True,
            require_reconciliation_slo=True,
        )
        self.assertFalse(ok)
        self.assertIn("promotion_quality_gate_not_ok", failed)
        self.assertFalse(detail["promotion_quality_gate_ok"])


    def test_replay_expected_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            script = PROJECT_ROOT / "scripts" / "replay_end_to_end_deterministic.py"
            out_file = td_path / "replay.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--expected-hash",
                    "deadbeef",
                    "--out-file",
                    str(out_file),
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2)
            payload = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertFalse(payload["ok"])
            self.assertFalse(payload["hash_match"])

    def test_incident_auto_halt_trips_and_clears(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            script = PROJECT_ROOT / "scripts" / "incident_auto_halt.py"

            daily_file = td_path / "daily.json"
            quality_file = td_path / "quality.json"
            recon_file = td_path / "recon.json"
            mode_file = td_path / "mode.json"
            state_file = td_path / "state.json"
            event_log = td_path / "events.jsonl"
            latest_alert = td_path / "latest.json"
            halt_flag = td_path / "GLOBAL_TRADING_HALT.flag"

            quality_file.write_text(json.dumps({"ok": True}), encoding="utf-8")
            recon_file.write_text(json.dumps({"ok": True}), encoding="utf-8")
            mode_file.write_text(json.dumps({"market_data_only": False, "allow_order_execution": True}), encoding="utf-8")

            daily_file.write_text(json.dumps({"ok": False}), encoding="utf-8")

            base_cmd = [
                sys.executable,
                str(script),
                "--daily-verify-file",
                str(daily_file),
                "--quality-gate-file",
                str(quality_file),
                "--reconciliation-file",
                str(recon_file),
                "--mode-file",
                str(mode_file),
                "--state-file",
                str(state_file),
                "--event-log",
                str(event_log),
                "--latest-alert-file",
                str(latest_alert),
                "--halt-flag",
                str(halt_flag),
                "--trip-streak",
                "2",
                "--clear-streak",
                "2",
                "--auto-clear",
                "--json",
            ]

            first = subprocess.run(base_cmd, check=False, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0)
            self.assertFalse(halt_flag.exists())

            second = subprocess.run(base_cmd, check=False, capture_output=True, text=True)
            self.assertEqual(second.returncode, 2)
            self.assertTrue(halt_flag.exists())

            daily_file.write_text(json.dumps({"ok": True}), encoding="utf-8")
            third = subprocess.run(base_cmd, check=False, capture_output=True, text=True)
            self.assertEqual(third.returncode, 0)
            self.assertTrue(halt_flag.exists())

            fourth = subprocess.run(base_cmd, check=False, capture_output=True, text=True)
            self.assertEqual(fourth.returncode, 0)
            self.assertFalse(halt_flag.exists())


    def test_paper_reconciliation_slo_guard_breaches_on_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            in_file = td_path / "paper_events.jsonl"
            now = datetime.now(timezone.utc).isoformat()
            rows = [
                {"timestamp_utc": now, "event": "paper_order_lifecycle_reconcile", "status": "mismatch"},
                {"timestamp_utc": now, "event": "paper_order_lifecycle_reconcile", "status": "ok"},
            ]
            in_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

            script = PROJECT_ROOT / "scripts" / "paper_reconciliation_slo_guard.py"
            out_file = td_path / "paper_recon.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--in-file",
                    str(in_file),
                    "--out-file",
                    str(out_file),
                    "--lookback-minutes",
                    "60",
                    "--max-mismatch-rate",
                    "0.10",
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2)
            payload = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertFalse(payload["ok"])
            self.assertIn("mismatch_rate", payload["failed_checks"])

    def test_paper_replay_drill_expected_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            in_file = td_path / "paper_trades_test.jsonl"
            now = datetime.now(timezone.utc).isoformat()
            rows = [
                {
                    "timestamp_utc": now,
                    "symbol": "AAPL",
                    "action": "BUY",
                    "quantity": 1.0,
                    "model_score": 0.61,
                    "threshold": 0.55,
                    "strategy": "paper_test",
                    "fill_price": 100.0,
                    "expected_fill_price": 100.01,
                }
            ]
            in_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

            script = PROJECT_ROOT / "scripts" / "paper_replay_drill.py"
            out_file = td_path / "paper_replay.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--in-file",
                    str(in_file),
                    "--min-rows",
                    "1",
                    "--expected-hash",
                    "deadbeef",
                    "--out-file",
                    str(out_file),
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2)
            payload = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertFalse(payload["ok"])
            self.assertFalse(payload["hash_match"])
            self.assertIn("expected_hash_mismatch", payload["failed_checks"])


if __name__ == "__main__":
    unittest.main()
