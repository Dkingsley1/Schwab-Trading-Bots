import json
from datetime import datetime, timedelta, timezone
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
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": "BTC-USD",
        "action": "BUY",
        "reference_price": 100.0,
        "fill_price": 100.0,
        "expected_fill_price": 100.8,
        "expected_slippage_bps": 80.0,
        "paper_fill_source": "broker_paper_fill",
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
            "--min-independent-samples",
            "1",
            "--out-file",
            str(out_file),
        ],
    )

    rc = report.main()
    payload = json.loads(out_file.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["samples"] == 1
    assert payload["overall_status"] == "ready"
    assert payload["by_market_kind"]["crypto"]["recommended_slippage_scale"] == 0.25
    assert payload["by_profile"]["default"]["samples"] == 1
    assert payload["top_symbols"][0]["symbol"] == "BTC-USD"
    assert payload["drift_series"][0]["bucket_start_utc"].endswith("+00:00")
    assert payload["line_graph"]["series"][0]["key"] == "mean_observed_slippage_bps"


def test_expected_fill_model_is_not_counted_as_independent_calibration(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "trade_logs" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = project_root / "governance" / "health" / "paper_execution_calibration_latest.json"
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": "SPY",
        "action": "BUY",
        "reference_price": 100.0,
        "fill_price": 100.1,
        "expected_fill_price": 100.1,
        "expected_slippage_bps": 10.0,
        "paper_fill_source": "expected_fill_model",
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
            "--out-file",
            str(out_file),
        ],
    )

    assert report.main() == 0
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["samples"] == 0
    assert payload["model_derived_samples"] == 1
    assert payload["independent_evidence_ready"] is False
    assert payload["overall_status"] == "evidence_pending"
    assert payload["model_derived_diagnostics"]["promotion_evidence_eligible"] is False


def test_paper_execution_calibration_report_respects_reset_cutoff(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "trade_logs" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = project_root / "governance" / "health" / "paper_execution_calibration_latest.json"
    row = {
        "timestamp_utc": "2026-06-25T14:00:00+00:00",
        "symbol": "AAPL",
        "action": "BUY",
        "reference_price": 100.0,
        "fill_price": 100.0,
        "expected_fill_price": 102.0,
        "expected_slippage_bps": 200.0,
        "metadata": {"source_profile": "default"},
    }
    (log_dir / "paper_trades_paper.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    monkeypatch.setenv("PAPER_EXECUTION_CALIBRATION_MIN_TIMESTAMP_UTC", "2026-06-25T14:30:00+00:00")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper_execution_calibration_report.py",
            "--hours",
            "24",
            "--out-file",
            str(out_file),
        ],
    )

    rc = report.main()
    payload = json.loads(out_file.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["samples"] == 0
    assert payload["ok"] is True
    assert payload["calibration_window"]["reset_active"] is True
    assert payload["calibration_window"]["skipped_before_cutoff"] == 1


def test_candidate_identity_is_required_for_independent_promotion_calibration(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "trade_logs" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = project_root / "governance" / "health" / "paper_execution_calibration_latest.json"
    now = datetime.now(timezone.utc)
    candidate_id = "pc-test-g4"
    candidate_path = project_root / "governance" / "runtime" / "production_candidate_state.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "generation": 4,
                "scope_windows_started_utc": {
                    "execution": (now - timedelta(hours=4)).isoformat(),
                    "data": (now - timedelta(hours=4)).isoformat(),
                    "dependencies": (now - timedelta(hours=4)).isoformat(),
                },
            }
        ),
        encoding="utf-8",
    )
    base = {
        "timestamp_utc": (now - timedelta(hours=1)).isoformat(),
        "symbol": "SPY",
        "action": "BUY",
        "reference_price": 100.0,
        "fill_price": 100.2,
        "expected_fill_price": 100.1,
        "expected_slippage_bps": 10.0,
        "paper_fill_source": "broker_paper_fill",
        "metadata": {"source_profile": "default"},
    }
    current = {**base, "candidate_id": candidate_id}
    mismatched = {**base, "candidate_id": "pc-old-g3", "symbol": "QQQ"}
    missing = {**base, "symbol": "IWM"}
    (log_dir / "paper_trades_candidate.jsonl").write_text(
        "\n".join(json.dumps(row) for row in (current, mismatched, missing)) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper_execution_calibration_report.py",
            "--hours",
            "24",
            "--min-independent-samples",
            "1",
            "--out-file",
            str(out_file),
        ],
    )

    assert report.main() == 0
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["independent_samples"] == 1
    assert payload["candidate_binding"]["candidate_id"] == candidate_id
    assert payload["candidate_binding"]["bound"] is True
    assert payload["candidate_binding"]["eligible_rows_identity_clean"] is False
    assert payload["candidate_binding"]["identity_mismatch_rows_excluded"] == 1
    assert payload["candidate_binding"]["identity_missing_rows_excluded"] == 1
