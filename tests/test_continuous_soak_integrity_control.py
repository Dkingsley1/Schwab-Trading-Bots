import json
from datetime import datetime, timedelta, timezone

from scripts.ops import continuous_soak_integrity_control as control


def test_soak_hardening_a_plus_does_not_fake_elapsed_completion(tmp_path) -> None:
    files = {
        "scripts/ops/production_excellence_control.py": (
            "verify_candidate_event_chain candidate_chain_recovery_anchor all_evidence_windows_reset "
            "scope_windows_started_utc changed_scopes required_hours thirty_day_window"
        ),
        "scripts/ops/source_verification_autorefresh.py": (
            "source_verification_retry_state.json starvation_override"
        ),
        "scripts/ops/unattended_soak_readiness.py": "",
        "scripts/ops/storage_backpressure_autopilot.py": "",
        "scripts/ops/memory_pressure_intelligence.py": "",
        "scripts/ops/grade_regression_guard.py": "",
        "scripts/ops/incident_timeline.py": "",
    }
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    payload = control.build_payload(tmp_path)

    assert payload["control_grade"] == "A+"
    assert payload["operational_capacity_ready"] is False
    assert payload["clean_720_hours_complete"] is False
    assert payload["elapsed_evidence_grade"] != "A+"
    assert payload["runtime_checks"]["paper_runtime_regression_clear"] is False
    assert payload["runtime_checks"]["paper_truth_reconciled_A_plus"] is False


def test_candidate_drift_preserves_observed_age_but_receives_zero_clean_credit(tmp_path) -> None:
    now = datetime(2026, 8, 16, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True)
    (health / "production_excellence_control_latest.json").write_text(
        json.dumps(
            {
                "candidate": {
                    "candidate_ready": False,
                    "candidate_drift": True,
                    "event_chain": {"ok": True, "event_count": 2},
                    "scope_windows_started_utc": {
                        "operations": (now - timedelta(hours=800)).isoformat()
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    payload = control.build_payload(tmp_path, now=now)

    assert payload["observed_window_elapsed_hours"] == 800.0
    assert payload["clean_window_elapsed_hours"] == 0.0
    assert payload["candidate_drift_invalidates_elapsed_credit"] is True
    assert payload["clean_720_hours_complete"] is False


def test_pre_reset_time_is_preserved_as_segmented_history_without_clean_credit(tmp_path) -> None:
    now = datetime(2026, 8, 18, 18, 0, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True)
    event_path = tmp_path / "governance" / "evidence" / "production_candidate_events.jsonl"
    event_path.parent.mkdir(parents=True)
    events = [
        {
            "timestamp_utc": (now - timedelta(hours=200)).isoformat(),
            "event_type": "candidate_change_accepted",
            "candidate_id": "candidate-1",
            "generation": 1,
            "change_reason": "first reviewed change",
            "changed_scopes": ["data"],
        },
        {
            "timestamp_utc": (now - timedelta(hours=50)).isoformat(),
            "event_type": "candidate_change_accepted",
            "candidate_id": "candidate-2",
            "generation": 2,
            "change_reason": "latest reviewed change",
            "changed_scopes": ["operations"],
        },
    ]
    event_path.write_text("\n".join(json.dumps(row) for row in events) + "\n", encoding="utf-8")
    (tmp_path / "governance" / "runtime").mkdir(parents=True)
    (tmp_path / "governance" / "runtime" / "production_candidate_state.json").write_text(
        json.dumps({"initialized_at_utc": (now - timedelta(hours=300)).isoformat()}),
        encoding="utf-8",
    )
    (health / "production_excellence_control_latest.json").write_text(
        json.dumps(
            {
                "candidate": {
                    "candidate_ready": True,
                    "candidate_drift": False,
                    "event_chain": {"ok": True, "event_count": 2, "path": str(event_path)},
                    "scope_windows_started_utc": {
                        "data": (now - timedelta(hours=200)).isoformat(),
                        "operations": (now - timedelta(hours=50)).isoformat(),
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    payload = control.build_payload(tmp_path, now=now)
    history = payload["historical_soak_evidence"]

    assert payload["main_soak_elapsed_hours"] == 300.0
    assert payload["main_soak_elapsed_days"] == 12.5
    assert payload["main_soak_progress_percent"] == 41.667
    assert payload["main_soak_includes_pre_reset_time"] is True
    assert payload["main_soak_count_is_promotion_credit"] is False
    assert payload["clean_window_elapsed_hours"] == 50.0
    assert history["historical_segmented_wall_clock_hours"] == 300.0
    assert history["wall_clock_hours_before_latest_full_system_window"] == 250.0
    assert history["candidate_event_count"] == 2
    assert history["segment_count"] == 3
    assert history["scope_window_elapsed_hours"] == {"data": 200.0, "operations": 50.0}
    assert history["historical_time_preserved"] is True
    assert history["counts_toward_current_clean_720_hours"] is False
    assert payload["grading_contract"]["pre_reset_time_is_preserved_as_segmented_history"] is True
    assert payload["grading_contract"]["pre_reset_time_is_included_in_main_soak_count"] is True


def test_planned_host_maintenance_preserves_credit_without_earning_offline_hours(tmp_path) -> None:
    now = datetime(2026, 8, 20, 16, 0, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True)
    event_path = tmp_path / "governance" / "evidence" / "production_candidate_events.jsonl"
    event_path.parent.mkdir(parents=True)
    event_path.write_text(
        json.dumps(
            {
                "timestamp_utc": (now - timedelta(hours=100)).isoformat(),
                "event_type": "candidate_change_accepted",
                "candidate_id": "candidate-1",
                "generation": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = tmp_path / "governance" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "production_candidate_state.json").write_text(
        json.dumps({"initialized_at_utc": (now - timedelta(hours=100)).isoformat()}),
        encoding="utf-8",
    )
    (health / "production_excellence_control_latest.json").write_text(
        json.dumps(
            {
                "candidate": {
                    "candidate_ready": True,
                    "candidate_drift": False,
                    "event_chain": {"ok": True, "event_count": 1, "path": str(event_path)},
                    "scope_windows_started_utc": {
                        "operations": (now - timedelta(hours=50)).isoformat(),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    maintenance = tmp_path / "governance" / "maintenance_events"
    maintenance.mkdir(parents=True)
    (maintenance / "20260820_test_update.json").write_text(
        json.dumps(
            {
                "event_id": "20260820_test_update",
                "status": "completed",
                "classification": "planned_host_software_maintenance",
                "title": "Test operating-system update",
                "actual_offline_window": {
                    "offline_start_utc": (now - timedelta(hours=40)).isoformat(),
                    "offline_end_utc": (now - timedelta(hours=36)).isoformat(),
                },
                "soak_accounting": {
                    "counts_as_system_degradation": False,
                    "counts_as_trading_system_failure": False,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = control.build_payload(tmp_path, now=now)

    assert payload["observed_window_elapsed_hours"] == 50.0
    assert payload["clean_window_planned_maintenance_excluded_hours"] == 4.0
    assert payload["clean_window_elapsed_hours"] == 46.0
    assert payload["main_soak_elapsed_hours"] == 100.0
    assert payload["main_soak_active_runtime_evidence_hours"] == 96.0
    assert payload["planned_maintenance"]["event_count"] == 1
    assert payload["planned_maintenance"]["current_candidate_reset_count"] == 0
    assert payload["planned_maintenance"]["pre_event_soak_credit_preserved"] is True
    assert payload["planned_maintenance"]["offline_time_earns_active_runtime_credit"] is False
    assert payload["grading_contract"]["planned_host_maintenance_is_an_explicit_restart_exception"] is True
