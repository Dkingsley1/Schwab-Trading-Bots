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
