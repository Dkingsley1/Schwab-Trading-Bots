from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import one_numbers_regression_guard as guard


def test_guard_detects_timeframe_collapse_and_missing_history(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    one_numbers_dir = tmp_path / "exports" / "one_numbers"
    one_numbers_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "governance" / "health").mkdir(parents=True, exist_ok=True)

    (one_numbers_dir / "one_numbers_summary.json").write_text(
        json.dumps(
            {
                "requested_day": "20260423",
                "resolved_day": "20260423",
                "report_mode": "lightweight_cached",
                "month_to_date_days_covered": "2",
                "all_time_days_covered": "2",
                "combined_decision_total_rows": "84766",
                "month_to_date_decision_total_rows": "84766",
                "all_time_decision_total_rows": "84766",
                "combined_blocked_total": "10",
                "month_to_date_blocked_total": "10",
                "all_time_blocked_total": "10",
                "data_blocked_total": "4",
                "month_to_date_data_blocked_total": "4",
                "all_time_data_blocked_total": "4",
                "risk_blocked_total": "6",
                "month_to_date_risk_blocked_total": "6",
                "all_time_risk_blocked_total": "6",
            }
        ),
        encoding="utf-8",
    )
    (one_numbers_dir / "latest.csv").write_text("label,value\n", encoding="utf-8")
    (one_numbers_dir / "latest_metrics.csv").write_text("section,label,value,metric\n", encoding="utf-8")

    payload = guard.build_payload(tmp_path)
    weakness_names = {item["name"] for item in payload["weaknesses"]}

    assert payload["overall_status"] == "degraded"
    assert payload["timeframe_collapse_detected"] is True
    assert "one_numbers_original_start_unpinned" in weakness_names
    assert "durable_rollup_history_missing" in weakness_names
    assert "lightweight_report_mode" in weakness_names
    assert "timeframe_collapse_detected" in weakness_names


def test_guard_ready_with_durable_history_and_distinct_rollups(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260421")
    one_numbers_dir = tmp_path / "exports" / "one_numbers"
    one_numbers_dir.mkdir(parents=True, exist_ok=True)
    health_dir = tmp_path / "governance" / "health"
    health_dir.mkdir(parents=True, exist_ok=True)

    (one_numbers_dir / "one_numbers_summary.json").write_text(
        json.dumps(
            {
                "requested_day": "20260423",
                "resolved_day": "20260423",
                "report_mode": "full",
                "month_to_date_days_covered": "2",
                "all_time_days_covered": "3",
                "combined_decision_total_rows": "100",
                "month_to_date_decision_total_rows": "180",
                "all_time_decision_total_rows": "260",
                "combined_blocked_total": "4",
                "month_to_date_blocked_total": "7",
                "all_time_blocked_total": "11",
                "data_blocked_total": "1",
                "month_to_date_data_blocked_total": "2",
                "all_time_data_blocked_total": "3",
                "risk_blocked_total": "3",
                "month_to_date_risk_blocked_total": "5",
                "all_time_risk_blocked_total": "8",
            }
        ),
        encoding="utf-8",
    )
    (health_dir / "one_numbers_rollup_history.json").write_text(
        json.dumps(
            {
                "history_by_day": {
                    "20260421": {"day_utc": "20260421", "metrics": {"combined_decision_total_rows": "80"}},
                    "20260422": {"day_utc": "20260422", "metrics": {"combined_decision_total_rows": "80"}},
                    "20260423": {"day_utc": "20260423", "metrics": {"combined_decision_total_rows": "100"}},
                }
            }
        ),
        encoding="utf-8",
    )
    (one_numbers_dir / "latest.csv").write_text("label,value\n", encoding="utf-8")
    (one_numbers_dir / "latest_metrics.csv").write_text("section,label,value,metric\n", encoding="utf-8")

    payload = guard.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["weaknesses"] == []
    assert payload["timeframe_collapse_detected"] is False


def test_guard_uses_pinned_config_start_day(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("ONE_NUMBERS_ORIGINAL_START_DAY", raising=False)
    monkeypatch.delenv("ONE_NUMBERS_EXPECTED_START_DAY", raising=False)
    monkeypatch.delenv("INFRA_SUPERVISOR_ONE_NUMBERS_START_DAY", raising=False)
    one_numbers_dir = tmp_path / "exports" / "one_numbers"
    one_numbers_dir.mkdir(parents=True, exist_ok=True)
    health_dir = tmp_path / "governance" / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "one_numbers_start_day.txt").write_text(
        "# pinned recovered start\n20260422\n",
        encoding="utf-8",
    )

    (one_numbers_dir / "one_numbers_summary.json").write_text(
        json.dumps(
            {
                "requested_day": "20260423",
                "resolved_day": "20260423",
                "report_mode": "full",
                "month_to_date_days_covered": "2",
                "all_time_days_covered": "2",
                "combined_decision_total_rows": "100",
                "month_to_date_decision_total_rows": "180",
                "all_time_decision_total_rows": "180",
                "combined_blocked_total": "4",
                "month_to_date_blocked_total": "7",
                "all_time_blocked_total": "7",
                "data_blocked_total": "0",
                "month_to_date_data_blocked_total": "0",
                "all_time_data_blocked_total": "0",
                "risk_blocked_total": "3",
                "month_to_date_risk_blocked_total": "5",
                "all_time_risk_blocked_total": "5",
            }
        ),
        encoding="utf-8",
    )
    (health_dir / "one_numbers_rollup_history.json").write_text(
        json.dumps(
            {
                "history_by_day": {
                    "20260422": {"day_utc": "20260422"},
                    "20260423": {"day_utc": "20260423"},
                }
            }
        ),
        encoding="utf-8",
    )
    (one_numbers_dir / "latest.csv").write_text("label,value\n", encoding="utf-8")
    (one_numbers_dir / "latest_metrics.csv").write_text("section,label,value,metric\n", encoding="utf-8")

    payload = guard.build_payload(tmp_path)
    weakness_names = {item["name"] for item in payload["weaknesses"]}

    assert "one_numbers_original_start_unpinned" not in weakness_names
    assert payload["original_coverage_contract"]["expected_start_day"] == "20260422"
    assert payload["original_coverage_contract"]["expected_start_source"].endswith("config/one_numbers_start_day.txt")


def test_guard_keeps_full_repair_when_runtime_throttle_blocks_full_refresh(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260422")
    one_numbers_dir = tmp_path / "exports" / "one_numbers"
    one_numbers_dir.mkdir(parents=True, exist_ok=True)
    health_dir = tmp_path / "governance" / "health"
    health_dir.mkdir(parents=True, exist_ok=True)

    (one_numbers_dir / "one_numbers_summary.json").write_text(
        json.dumps(
            {
                "requested_day": "20260423",
                "resolved_day": "20260423",
                "report_mode": "full",
                "month_to_date_days_covered": "2",
                "all_time_days_covered": "2",
                "combined_decision_total_rows": "0",
                "month_to_date_decision_total_rows": "100",
                "all_time_decision_total_rows": "150",
                "combined_blocked_total": "0",
                "month_to_date_blocked_total": "5",
                "all_time_blocked_total": "8",
                "data_blocked_total": "0",
                "month_to_date_data_blocked_total": "1",
                "all_time_data_blocked_total": "1",
                "risk_blocked_total": "0",
                "month_to_date_risk_blocked_total": "4",
                "all_time_risk_blocked_total": "7",
            }
        ),
        encoding="utf-8",
    )
    (one_numbers_dir / "latest.csv").write_text("label,value\n", encoding="utf-8")
    (one_numbers_dir / "latest_metrics.csv").write_text("section,label,value,metric\n", encoding="utf-8")
    (health_dir / "one_numbers_rollup_history.json").write_text(
        json.dumps(
            {
                "history_by_day": {
                    "20260422": {"day_utc": "20260422", "metrics": {"combined_decision_total_rows": "100"}},
                    "20260423": {"day_utc": "20260423", "metrics": {"combined_decision_total_rows": "0"}},
                }
            }
        ),
        encoding="utf-8",
    )
    (health_dir / "runtime_throttle_control_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "blocked",
                "throttle_profile": "protect_live",
                "host_saturation_score": 99.0,
            }
        ),
        encoding="utf-8",
    )

    payload = guard.build_payload(tmp_path)
    advisory_names = {item["name"] for item in payload["advisories"]}

    assert payload["overall_status"] == "ready"
    assert payload["repair_plan"]["preferred_mode"] == "full"
    assert payload["repair_plan"]["full_refresh_blocked_by_throttle"] is True
    assert payload["runtime_throttle"]["throttle_profile"] == "protect_live"
    assert "full_refresh_throttle_guarded" in advisory_names
    assert "--lightweight" not in payload["repair_plan"]["recommended_command"]
    assert payload["assigned_infrastructure_drift_bot"]["bot"] == "system_drift_autopilot"


def test_guard_flags_source_days_missing_from_rollup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260421")
    one_numbers_dir = tmp_path / "exports" / "one_numbers"
    one_numbers_dir.mkdir(parents=True, exist_ok=True)
    health_dir = tmp_path / "governance" / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    source_dir = tmp_path / "decision_explanations" / "paper"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "decision_explanations_20260421.jsonl.local_fallback.1").write_text("{}\n", encoding="utf-8")
    (source_dir / "decision_explanations_20260422.jsonl").write_text("{}\n", encoding="utf-8")

    (one_numbers_dir / "one_numbers_summary.json").write_text(
        json.dumps(
            {
                "requested_day": "20260422",
                "resolved_day": "20260422",
                "report_mode": "full",
                "month_to_date_days_covered": "1",
                "all_time_days_covered": "1",
                "combined_decision_total_rows": "100",
                "month_to_date_decision_total_rows": "100",
                "all_time_decision_total_rows": "100",
                "combined_blocked_total": "4",
                "month_to_date_blocked_total": "4",
                "all_time_blocked_total": "4",
                "data_blocked_total": "1",
                "month_to_date_data_blocked_total": "1",
                "all_time_data_blocked_total": "1",
                "risk_blocked_total": "3",
                "month_to_date_risk_blocked_total": "3",
                "all_time_risk_blocked_total": "3",
            }
        ),
        encoding="utf-8",
    )
    (health_dir / "one_numbers_rollup_history.json").write_text(
        json.dumps({"history_by_day": {"20260422": {"day_utc": "20260422"}}}),
        encoding="utf-8",
    )
    (one_numbers_dir / "latest.csv").write_text("label,value\n", encoding="utf-8")
    (one_numbers_dir / "latest_metrics.csv").write_text("section,label,value,metric\n", encoding="utf-8")

    payload = guard.build_payload(tmp_path)
    weakness_names = {item["name"] for item in payload["weaknesses"]}

    assert payload["overall_status"] == "degraded"
    assert "one_numbers_history_starts_after_expected" in weakness_names
    assert "one_numbers_source_days_missing_from_rollup" in weakness_names
    assert payload["original_coverage_contract"]["source_days_missing_from_history_sample"] == ["20260421"]
    assert payload["repair_plan"]["backfill_commands"][0][-2:] == ["--day", "20260421"]


def test_guard_flags_decision_rows_missing_when_governance_exists(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260422")
    one_numbers_dir = tmp_path / "exports" / "one_numbers"
    one_numbers_dir.mkdir(parents=True, exist_ok=True)
    health_dir = tmp_path / "governance" / "health"
    health_dir.mkdir(parents=True, exist_ok=True)

    (one_numbers_dir / "one_numbers_summary.json").write_text(
        json.dumps(
            {
                "requested_day": "20260423",
                "resolved_day": "20260423",
                "report_mode": "full",
                "month_to_date_days_covered": "2",
                "all_time_days_covered": "2",
                "combined_decision_total_rows": "0",
                "combined_governance_total_rows": "86056",
                "month_to_date_decision_total_rows": "84766",
                "all_time_decision_total_rows": "84766",
                "combined_blocked_total": "0",
                "month_to_date_blocked_total": "4313",
                "all_time_blocked_total": "4313",
                "data_blocked_total": "0",
                "month_to_date_data_blocked_total": "0",
                "all_time_data_blocked_total": "0",
                "risk_blocked_total": "0",
                "month_to_date_risk_blocked_total": "4313",
                "all_time_risk_blocked_total": "4313",
            }
        ),
        encoding="utf-8",
    )
    (health_dir / "one_numbers_rollup_history.json").write_text(
        json.dumps({"history_by_day": {"20260422": {"day_utc": "20260422"}, "20260423": {"day_utc": "20260423"}}}),
        encoding="utf-8",
    )
    (one_numbers_dir / "latest.csv").write_text("label,value\n", encoding="utf-8")
    (one_numbers_dir / "latest_metrics.csv").write_text("section,label,value,metric\n", encoding="utf-8")

    payload = guard.build_payload(tmp_path)
    weakness_names = {item["name"] for item in payload["weaknesses"]}

    assert payload["overall_status"] == "degraded"
    assert "decision_rows_missing_with_governance_activity" in weakness_names
