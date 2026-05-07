from __future__ import annotations

from scripts.ops import drainer_intelligence_layer as src


def _fleet(writer_lock_held: bool = False) -> dict:
    return {
        "overall_status": "ready",
        "ready_drainer_count": 2,
        "writer_lock_held": writer_lock_held,
        "active_drainer": {
            "name": "core_decision_drainer",
            "status": "ready",
            "pending_lines": 12000,
            "priority_score": 72000,
            "assigned_pressure_lane": "core_decision_backpressure",
            "live_window_safe": True,
        },
        "candidate_drainers": [
            {
                "name": "core_decision_drainer",
                "status": "ready",
                "pending_lines": 12000,
                "priority_score": 72000,
                "assigned_pressure_lane": "core_decision_backpressure",
                "live_window_safe": True,
            },
            {
                "name": "settlement_reconciliation_drainer",
                "status": "ready",
                "pending_lines": 1200,
                "priority_score": 54000,
                "assigned_pressure_lane": "settlement_reconciliation_backpressure",
                "live_window_safe": True,
            },
        ],
        "metrics": {"total_pending_lines": 13200},
    }


def test_drainer_intelligence_waits_when_writer_is_active() -> None:
    payload = src.build_intelligence_from_payloads(
        fleet=_fleet(writer_lock_held=True),
        super_drainer={"overall_status": "waiting_for_writer", "active_drainer": "core_decision_drainer"},
        memory={"history_count": 4, "recent_progress_rate": 0.0, "recent_target_met_rate": 0.0},
        storage={"overall_status": "blocked", "severity": "critical", "backpressure": {"total_pending_lines": 13200}},
        runtime={"overall_status": "ready"},
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={"active": True},
        target_pending_lines=10000,
    )

    decision = payload["decision_packet"]
    assert payload["overall_status"] == "ready"
    assert decision["action"] == "wait_for_writer_then_re_score"
    assert decision["selected_drainer"] == "core_decision_drainer"
    assert "writer_active" in decision["risk_flags"]
    assert "recent_progress_rate_low" in decision["risk_flags"]
    assert decision["writer_health"]["state"] == "active_progressing"
    assert payload["lane_intelligence"][0]["recommended_mode"] == "wait_then_re_score"
    assert payload["lane_family_summary"][0]["family"] == "core_decision"
    assert payload["control_contract"]["starts_parallel_sql_writers"] is False


def test_drainer_intelligence_micro_drains_after_pressure_relief() -> None:
    payload = src.build_intelligence_from_payloads(
        fleet=_fleet(writer_lock_held=False),
        super_drainer={"overall_status": "ready", "active_drainer": "core_decision_drainer"},
        memory={"history_count": 1, "recent_progress_rate": 1.0, "recent_target_met_rate": 0.0},
        storage={"overall_status": "blocked", "severity": "critical", "backpressure": {"total_pending_lines": 13200}},
        runtime={"overall_status": "blocked", "memory_pressure_level": "high", "host_saturation_score": 88},
        memory_efficiency={"overall_status": "blocked", "memory_snapshot": {"memory_pressure_state": "yellow", "memory_pressure_kind": "swap"}},
        writer={"active": False},
        target_pending_lines=10000,
    )

    decision = payload["decision_packet"]
    assert decision["action"] == "run_micro_drain_after_pressure_relief"
    assert decision["adaptive_target_pending_lines"] == 5000
    assert decision["recommended_max_waves"] == 1
    assert "memory_pressure_high" in decision["risk_flags"]
    assert "runtime_pressure_high" in decision["risk_flags"]


def test_drainer_intelligence_runs_bounded_wave_when_writer_idle() -> None:
    payload = src.build_intelligence_from_payloads(
        fleet=_fleet(writer_lock_held=False),
        super_drainer={"overall_status": "ready", "active_drainer": "core_decision_drainer"},
        memory={"history_count": 2, "recent_progress_rate": 0.5, "recent_target_met_rate": 0.0},
        storage={"overall_status": "blocked", "severity": "critical", "backpressure": {"total_pending_lines": 13200}},
        runtime={"overall_status": "ready"},
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={"active": False},
        target_pending_lines=10000,
    )

    decision = payload["decision_packet"]
    assert decision["action"] == "run_bounded_wave"
    assert decision["next_ready_drainer"] == "settlement_reconciliation_drainer"
    assert decision["confidence"] >= 0.7
    assert payload["drain_playbook"][0]["step"] == "run_selected_lane"
    assert payload["safety_envelope"]["max_apply_waves_now"] == 2
    assert payload["lane_intelligence"][0]["name"] == "core_decision_drainer"


def test_drainer_intelligence_detects_stale_writer_progress() -> None:
    payload = src.build_intelligence_from_payloads(
        fleet=_fleet(writer_lock_held=True),
        super_drainer={"overall_status": "waiting_for_writer", "active_drainer": "core_decision_drainer"},
        memory={"history_count": 5, "recent_progress_rate": 0.0, "recent_target_met_rate": 0.0},
        storage={"overall_status": "blocked", "severity": "critical", "backpressure": {"total_pending_lines": 13200}},
        runtime={"overall_status": "ready"},
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={"active": True, "progress_age_minutes": 61.0, "cycle_age_minutes": 70.0, "merged_rows_this_cycle": 3200},
        target_pending_lines=10000,
    )

    decision = payload["decision_packet"]
    assert decision["action"] == "verify_writer_progress_then_re_score"
    assert decision["writer_health"]["state"] == "stale_progress"
    assert "writer_progress_stale" in decision["risk_flags"]
    assert payload["safety_envelope"]["writer_recovery_required"] is True
    assert payload["drain_playbook"][0]["step"] == "inspect_writer"
