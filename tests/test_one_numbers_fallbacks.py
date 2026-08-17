from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import canary_auto_tuner
from scripts.ops import daily_ops_report


def test_canary_auto_tuner_reads_latest_metrics_csv_when_summary_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(canary_auto_tuner, "PROJECT_ROOT", tmp_path)
    one_numbers_dir = tmp_path / "exports" / "one_numbers"
    one_numbers_dir.mkdir(parents=True, exist_ok=True)
    (one_numbers_dir / "latest_metrics.csv").write_text(
        "section,label,value,metric\nCurrent Day,Data Quality Score,88.50,data_quality_score\n",
        encoding="utf-8",
    )

    assert canary_auto_tuner._one_numbers_data_quality() == 88.5


def test_daily_ops_report_reads_legacy_latest_csv_when_metrics_alias_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(daily_ops_report, "PROJECT_ROOT", tmp_path)
    one_numbers_dir = tmp_path / "exports" / "one_numbers"
    one_numbers_dir.mkdir(parents=True, exist_ok=True)
    (one_numbers_dir / "latest.csv").write_text(
        "metric,value\ndata_quality_score,81.25\npressure_index,0.125\n",
        encoding="utf-8",
    )

    payload = daily_ops_report._one_numbers()

    assert payload["data_quality_score"] == "81.25"
    assert payload["pressure_index"] == "0.125"
