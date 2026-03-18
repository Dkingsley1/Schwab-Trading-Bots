import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.paper_execution_calibration_report as report


def test_paper_execution_calibration_report_emits_grouped_recommendations(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "trade_logs" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = project_root / "governance" / "health" / "paper_execution_calibration_latest.json"
    row = {
        "timestamp_utc": "2026-03-12T15:00:00+00:00",
        "symbol": "BTC-USD",
        "action": "BUY",
        "reference_price": 100.0,
        "fill_price": 100.0,
        "expected_fill_price": 100.8,
        "expected_slippage_bps": 80.0,
        "metadata": {"source_profile": "default"},
    }
    (log_dir / "paper_trades_paper.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper_execution_calibration_report.py",
            "--hours",
            "24",
            "--max-mae-bps",
            "100",
            "--out-file",
            str(out_file),
        ],
    )

    rc = report.main()
    payload = json.loads(out_file.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["samples"] == 1
    assert payload["by_market_kind"]["crypto"]["recommended_slippage_scale"] == 0.25
    assert payload["by_profile"]["default"]["samples"] == 1
    assert payload["top_symbols"][0]["symbol"] == "BTC-USD"
