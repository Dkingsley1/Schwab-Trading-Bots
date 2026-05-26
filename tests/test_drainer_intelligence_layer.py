from __future__ import annotations

import json

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
    micro_command = payload["drain_playbook"][1]["command"]
    assert "--sql-manager-timeout-cap-seconds" in micro_command
    assert micro_command[micro_command.index("--sql-manager-timeout-cap-seconds") + 1] == "420"
    assert "--command-timeout-seconds" in micro_command
    assert micro_command[micro_command.index("--command-timeout-seconds") + 1] == "540"


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
    assert "--sql-manager-timeout-cap-seconds" in payload["drain_playbook"][0]["command"]
    assert payload["safety_envelope"]["max_apply_waves_now"] == 2
    assert payload["lane_intelligence"][0]["name"] == "core_decision_drainer"


def test_drainer_intelligence_tightens_intake_after_refill_wave() -> None:
    payload = src.build_intelligence_from_payloads(
        fleet=_fleet(writer_lock_held=False),
        super_drainer={"overall_status": "applied_with_followups", "active_drainer": "core_decision_drainer"},
        memory={
            "history_count": 4,
            "recent_progress_rate": 0.25,
            "recent_target_met_rate": 0.0,
            "recent_refill_rate": 0.5,
            "latest_event": {
                "initial_pending_lines": 34504,
                "final_pending_lines": 36333,
                "pending_lines_net_change": 1829,
                "waves_run": 1,
                "refill_detected": True,
            },
        },
        storage={"overall_status": "blocked", "severity": "critical", "backpressure": {"total_pending_lines": 36333}},
        runtime={"overall_status": "ready"},
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={"active": False},
        target_pending_lines=10000,
    )

    decision = payload["decision_packet"]
    assert decision["action"] == "tighten_intake_then_re_score"
    assert "recent_refill_after_drain" in decision["risk_flags"]
    assert payload["drain_playbook"][0]["step"] == "pressure_relief"
    assert payload["drain_playbook"][1]["step"] == "runtime_throttle"
    assert payload["learning_summary"]["latest_refill_detected"] is True
    assert payload["learning_summary"]["latest_pending_lines_net_change"] == 1829


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


def test_drainer_intelligence_uses_sql_overlay_for_lane_ranking() -> None:
    fleet = _fleet(writer_lock_held=True)
    fleet["candidate_drainers"].append(
        {
            "name": "governance_execution_drainer",
            "status": "idle",
            "pending_lines": 0,
            "priority_score": 80000,
            "assigned_pressure_lane": "governance_execution_backpressure",
            "live_window_safe": True,
        }
    )

    payload = src.build_intelligence_from_payloads(
        fleet=fleet,
        super_drainer={"overall_status": "waiting_for_writer", "active_drainer": "core_decision_drainer"},
        memory={"history_count": 4, "recent_progress_rate": 0.4, "recent_target_met_rate": 0.0},
        storage={
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {"total_pending_lines": 759207},
            "sql_ingestion_pending_overlay": {
                "active": True,
                "used_for_pressure": True,
                "total_pending_lines": 758206,
                "core_pending_lines": 311901,
                "support_pending_lines": 446305,
                "top_pending_files": [
                    {
                        "source_rel": "governance/health/jsonl_ingest_batch_journal_health_fast_latest.jsonl",
                        "shard": "governance",
                        "stream": "governance",
                        "pressure_lane": "support",
                        "pending_lines": 125777,
                    },
                    {
                        "source_rel": "decisions/paper/trade_decisions_20260519.jsonl",
                        "shard": "trading",
                        "stream": "decisions",
                        "pressure_lane": "core",
                        "pending_lines": 92520,
                    },
                ],
            },
        },
        runtime={"overall_status": "ready"},
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={"active": True},
        target_pending_lines=10000,
    )

    lanes = {row["name"]: row for row in payload["lane_intelligence"]}
    decision = payload["decision_packet"]
    assert decision["storage_overlay_used"] is True
    assert decision["next_ready_drainer"] == "governance_execution_drainer"
    assert lanes["governance_execution_drainer"]["status"] == "ready"
    assert lanes["governance_execution_drainer"]["pending_lines"] == 446305
    assert lanes["governance_execution_drainer"]["storage_overlay_pending_lines"] == 446305
    assert lanes["core_decision_drainer"]["pending_lines"] == 311901
    assert payload["lane_family_summary"][0]["family"] == "governance_telemetry"


def test_drainer_intelligence_uses_current_total_not_super_drainer_high_watermark() -> None:
    payload = src.build_intelligence_from_payloads(
        fleet=_fleet(writer_lock_held=True),
        super_drainer={
            "overall_status": "applied_with_followups",
            "active_drainer": "core_decision_drainer",
            "summary": {
                "initial_pending_lines": 759207,
                "final_pending_lines": 628010,
                "pending_lines_delta": 131197,
            },
        },
        memory={"history_count": 4, "recent_progress_rate": 0.4, "recent_target_met_rate": 0.0},
        storage={
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {"total_pending_lines": 628010},
            "sql_ingestion_pending_overlay": {
                "active": True,
                "used_for_pressure": True,
                "total_pending_lines": 627009,
                "core_pending_lines": 180704,
                "support_pending_lines": 446305,
            },
        },
        runtime={"overall_status": "ready"},
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={"active": True},
        target_pending_lines=10000,
    )

    decision = payload["decision_packet"]
    assert decision["total_pending_lines"] == 628010
    assert decision["pressure_forecast"]["remaining_pending_lines"] == 618010


def test_drainer_intelligence_prefers_fresh_storage_total_over_stale_super_final() -> None:
    payload = src.build_intelligence_from_payloads(
        fleet=_fleet(writer_lock_held=True),
        super_drainer={
            "overall_status": "applied_with_followups",
            "active_drainer": "core_decision_drainer",
            "summary": {
                "initial_pending_lines": 759207,
                "final_pending_lines": 628010,
                "pending_lines_delta": 131197,
            },
        },
        memory={"history_count": 4, "recent_progress_rate": 0.4, "recent_target_met_rate": 0.0},
        storage={
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {"total_pending_lines": 40742},
            "sql_ingestion_pending_overlay": {
                "active": True,
                "used_for_pressure": True,
                "total_pending_lines": 39726,
                "core_pending_lines": 39726,
            },
        },
        runtime={"overall_status": "ready"},
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={"active": True},
        target_pending_lines=10000,
    )

    decision = payload["decision_packet"]
    assert decision["total_pending_lines"] == 40742
    assert decision["pressure_forecast"]["remaining_pending_lines"] == 30742
    assert payload["storage_overlay_context"]["total_pending_lines"] == 39726


def test_backlog_section_scorecard_grades_sparse_core_pressure() -> None:
    fleet = _fleet(writer_lock_held=False)
    fleet["active_drainer"]["pending_lines"] = 24461
    fleet["active_drainer"]["sparse_large_line_pressure"] = {
        "active": True,
        "file_count": 2,
        "pending_lines": 3251,
        "file_size_bytes": 9682107364,
        "top_files": [
            {
                "source_rel": "governance/channels/decision/crypto_futures_crypto_schwab/decision_20260520.jsonl",
                "pending_lines": 3141,
                "file_size_bytes": 3808586253,
            }
        ],
    }
    fleet["metrics"] = {
        "core_pending_lines": 33578,
        "deferred_pending_lines": 6709,
        "support_pending_lines": 55,
        "cold_pending_lines": 0,
        "total_pending_lines": 40287,
    }

    payload = src.build_intelligence_from_payloads(
        fleet=fleet,
        super_drainer={"overall_status": "ready", "active_drainer": "core_decision_drainer"},
        memory={"history_count": 4, "recent_progress_rate": 0.4, "recent_target_met_rate": 0.0},
        storage={
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {
                "core_pending_lines": 33989,
                "deferred_pending_lines": 6975,
                "cold_pending_lines": 0,
                "support_pending_lines": 144,
                "stale_stage_pending_lines": 0,
                "total_pending_lines": 40964,
                "oldest_pending_age_seconds": 420472.844,
                "raw_live": {
                    "line_estimation": {
                        "sparse_large_line_files": 3,
                        "sparse_large_line_pending_lines": 3416,
                        "sparse_large_line_bytes": 9964193671,
                        "sparse_large_line_active": True,
                    }
                },
            },
        },
        runtime={"overall_status": "ready"},
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={
            "writer_state_before": {
                "active": False,
                "active_source": "orphaned_progress",
                "progress_orphaned": True,
                "progress_age_minutes": 6.968,
                "completed_merge_count": 2,
                "timed_out_shard_count": 1,
            }
        },
        target_pending_lines=10000,
    )

    scorecard = payload["backlog_section_scorecard"]
    sections = {row["section_id"]: row for row in scorecard["sections"]}
    assert scorecard["overall_grade"] == "F"
    assert payload["decision_packet"]["backlog_grade"] == "F"
    assert sections["core_decision"]["grade"] == "F"
    assert sections["crypto_sparse_decision"]["grade"] == "F"
    assert sections["writer_merge_health"]["grade"] == "D"
    assert sections["support_watchdog"]["grade"] == "A++"
    assert scorecard["operator_next_focus"][0]["section_id"] in {"core_decision", "crypto_sparse_decision"}
    assert "writer_progress_orphaned" in payload["decision_packet"]["risk_flags"]


def test_backlog_section_scorecard_prefers_live_storage_counts_over_stale_fleet_metrics() -> None:
    fleet = _fleet(writer_lock_held=False)
    fleet["metrics"] = {
        "core_pending_lines": 36129,
        "deferred_pending_lines": 8137,
        "support_pending_lines": 199,
        "cold_pending_lines": 0,
        "total_pending_lines": 44266,
    }

    payload = src.build_intelligence_from_payloads(
        fleet=fleet,
        super_drainer={"overall_status": "ready", "active_drainer": "core_decision_drainer"},
        memory={"history_count": 4, "recent_progress_rate": 0.4, "recent_target_met_rate": 0.0},
        storage={
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {
                "core_pending_lines": 32773,
                "deferred_pending_lines": 915,
                "cold_pending_lines": 0,
                "support_pending_lines": 68,
                "stale_stage_pending_lines": 0,
                "total_pending_lines": 33688,
                "oldest_pending_age_seconds": 424259.228,
            },
        },
        runtime={"overall_status": "ready"},
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={"active": False},
        target_pending_lines=10000,
    )

    sections = {row["section_id"]: row for row in payload["backlog_section_scorecard"]["sections"]}
    assert sections["core_decision"]["pending_lines"] == 32773
    assert sections["deferred_data_quality"]["pending_lines"] == 915
    assert sections["support_watchdog"]["pending_lines"] == 68
    assert payload["decision_packet"]["total_pending_lines"] == 33688


def test_sparse_scorecard_prefers_live_line_estimation_over_stale_fleet_pressure() -> None:
    fleet = _fleet(writer_lock_held=False)
    fleet["active_drainer"]["sparse_large_line_pressure"] = {
        "active": True,
        "file_count": 9,
        "pending_lines": 2077,
        "file_size_bytes": 12_758_707_596,
        "estimated_pending_bytes": 483_308_414,
        "top_files": [
            {
                "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260520.jsonl",
                "estimated_pending_bytes": 483_308_414,
            }
        ],
    }

    payload = src.build_intelligence_from_payloads(
        fleet=fleet,
        super_drainer={"overall_status": "ready", "active_drainer": "core_decision_drainer"},
        memory={"history_count": 4, "recent_progress_rate": 0.4, "recent_target_met_rate": 0.0},
        storage={
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {
                "core_pending_lines": 33077,
                "deferred_pending_lines": 1427,
                "cold_pending_lines": 0,
                "support_pending_lines": 6,
                "stale_stage_pending_lines": 0,
                "total_pending_lines": 34504,
                "oldest_pending_age_seconds": 424596.539,
                "raw_live": {
                    "line_estimation": {
                        "sparse_large_line_files": 9,
                        "sparse_large_line_pending_lines": 753,
                        "sparse_large_line_bytes": 12_758_707_596,
                        "sparse_large_line_pending_bytes": 483_308_414,
                        "sparse_large_line_active": True,
                    }
                },
            },
        },
        runtime={"overall_status": "ready"},
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={"active": False},
        target_pending_lines=10000,
    )

    sections = {row["section_id"]: row for row in payload["backlog_section_scorecard"]["sections"]}
    assert sections["crypto_sparse_decision"]["pending_lines"] == 753
    assert sections["crypto_sparse_decision"]["score"] >= 60.0
    assert sections["crypto_sparse_decision"]["evidence"][0] == "sparse_pending_lines=753"
    assert "sparse_estimated_pending_bytes=483308414" in sections["crypto_sparse_decision"]["evidence"]


def test_runtime_capacity_scores_protect_live_soft_block_as_strained() -> None:
    payload = src.build_intelligence_from_payloads(
        fleet=_fleet(writer_lock_held=False),
        super_drainer={"overall_status": "ready", "active_drainer": "core_decision_drainer"},
        memory={"history_count": 2, "recent_progress_rate": 0.5, "recent_target_met_rate": 0.0},
        storage={"overall_status": "blocked", "severity": "critical", "backpressure": {"total_pending_lines": 30344}},
        runtime={
            "overall_status": "blocked",
            "throttle_profile": "protect_live",
            "host_saturation_score": 68.6,
            "compute_pressure_level": "high",
            "memory_pressure_level": "normal",
        },
        memory_efficiency={
            "overall_status": "blocked",
            "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"},
        },
        writer={"active": False},
        target_pending_lines=10000,
    )

    sections = {row["section_id"]: row for row in payload["backlog_section_scorecard"]["sections"]}
    assert sections["runtime_capacity"]["grade"] == "C"
    assert "memory_pressure_high" not in payload["decision_packet"]["risk_flags"]


def test_backlog_section_scorecard_grades_clean_backlog_green() -> None:
    fleet = _fleet(writer_lock_held=False)
    fleet["active_drainer"]["pending_lines"] = 200
    fleet["metrics"] = {
        "core_pending_lines": 200,
        "deferred_pending_lines": 100,
        "support_pending_lines": 25,
        "cold_pending_lines": 0,
        "total_pending_lines": 325,
    }

    payload = src.build_intelligence_from_payloads(
        fleet=fleet,
        super_drainer={"overall_status": "ready", "active_drainer": "core_decision_drainer"},
        memory={"history_count": 2, "recent_progress_rate": 1.0, "recent_target_met_rate": 1.0},
        storage={
            "overall_status": "ready",
            "backpressure": {
                "core_pending_lines": 200,
                "deferred_pending_lines": 100,
                "support_pending_lines": 25,
                "cold_pending_lines": 0,
                "stale_stage_pending_lines": 0,
                "total_pending_lines": 325,
                "oldest_pending_age_seconds": 120,
            },
        },
        runtime={"overall_status": "ready", "host_saturation_score": 35},
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={"active": False},
        target_pending_lines=10000,
    )

    scorecard = payload["backlog_section_scorecard"]
    sections = {row["section_id"]: row for row in scorecard["sections"]}
    assert scorecard["overall_grade"] == "A++"
    assert scorecard["overall_score"] >= 95
    assert sections["writer_merge_health"]["grade"] == "A+"
    assert sections["runtime_capacity"]["grade"] == "A++"
    assert {row["grade"] for row in scorecard["sections"]} <= {"A++", "A+", "A", "B"}
    assert payload["backlog_needs_packet"]["overall_status"] == "clear"
    assert payload["backlog_needs_packet"]["needs"] == []


def test_runtime_capacity_scores_cool_soft_degraded_host_as_backlog_safe() -> None:
    payload = src.build_intelligence_from_payloads(
        fleet=_fleet(writer_lock_held=False),
        super_drainer={"overall_status": "ready", "active_drainer": "core_decision_drainer"},
        memory={"history_count": 2, "recent_progress_rate": 1.0, "recent_target_met_rate": 1.0},
        storage={
            "overall_status": "ready",
            "backpressure": {
                "core_pending_lines": 285,
                "deferred_pending_lines": 64,
                "support_pending_lines": 54,
                "cold_pending_lines": 0,
                "stale_stage_pending_lines": 0,
                "total_pending_lines": 349,
                "oldest_pending_age_seconds": 39.351,
            },
        },
        runtime={
            "overall_status": "degraded",
            "host_saturation_score": 41.63,
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "normal",
        },
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={"active": False},
        target_pending_lines=10000,
    )

    sections = {row["section_id"]: row for row in payload["backlog_section_scorecard"]["sections"]}
    assert sections["runtime_capacity"]["grade"] == "A++"
    assert payload["backlog_section_scorecard"]["overall_grade"] == "A++"
    assert "runtime_pressure_high" not in payload["decision_packet"]["risk_flags"]
    assert payload["backlog_needs_packet"]["overall_status"] == "clear"


def test_runtime_capacity_scores_bounded_soft_degraded_host_as_stable() -> None:
    payload = src.build_intelligence_from_payloads(
        fleet=_fleet(writer_lock_held=False),
        super_drainer={"overall_status": "ready", "active_drainer": "core_decision_drainer"},
        memory={"history_count": 2, "recent_progress_rate": 1.0, "recent_target_met_rate": 1.0},
        storage={
            "overall_status": "ready",
            "backpressure": {
                "core_pending_lines": 17,
                "deferred_pending_lines": 0,
                "support_pending_lines": 0,
                "cold_pending_lines": 0,
                "stale_stage_pending_lines": 0,
                "total_pending_lines": 17,
                "oldest_pending_age_seconds": 0.0,
            },
        },
        runtime={
            "overall_status": "degraded",
            "host_saturation_score": 65.47,
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "normal",
        },
        memory_efficiency={"overall_status": "needs_work", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={"active": False},
        target_pending_lines=10000,
    )

    sections = {row["section_id"]: row for row in payload["backlog_section_scorecard"]["sections"]}
    assert sections["runtime_capacity"]["grade"] == "B"
    assert sections["runtime_capacity"]["score"] == 82.0
    assert payload["backlog_needs_packet"]["overall_status"] == "needs_attention"
    assert payload["backlog_needs_packet"]["top_need_section"] == "runtime_capacity"


def test_fresh_active_writer_progress_scores_as_a_not_b() -> None:
    payload = src.build_intelligence_from_payloads(
        fleet=_fleet(writer_lock_held=True),
        super_drainer={"overall_status": "waiting_for_writer", "active_drainer": "core_decision_drainer"},
        memory={"history_count": 2, "recent_progress_rate": 1.0, "recent_target_met_rate": 1.0},
        storage={
            "overall_status": "ready",
            "backpressure": {
                "core_pending_lines": 17,
                "deferred_pending_lines": 0,
                "support_pending_lines": 0,
                "cold_pending_lines": 0,
                "stale_stage_pending_lines": 0,
                "total_pending_lines": 17,
                "oldest_pending_age_seconds": 0.0,
            },
        },
        runtime={"overall_status": "ready", "host_saturation_score": 35, "compute_pressure_level": "normal", "memory_pressure_level": "normal"},
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={
            "writer_state_before": {
                "active": True,
                "current_step": "merge_primary",
                "progress_age_minutes": 1.6,
                "completed_merge_count": 3,
                "timed_out_shard_count": 0,
            }
        },
        target_pending_lines=10000,
    )

    sections = {row["section_id"]: row for row in payload["backlog_section_scorecard"]["sections"]}
    assert sections["writer_merge_health"]["grade"] == "A"
    assert sections["writer_merge_health"]["score"] == 92.0


def test_backlog_needs_packet_names_exact_core_requirements() -> None:
    payload = src.build_intelligence_from_payloads(
        fleet=_fleet(writer_lock_held=False),
        super_drainer={"overall_status": "ready", "active_drainer": "core_decision_drainer"},
        memory={
            "history_count": 5,
            "recent_progress_rate": 0.3,
            "recent_target_met_rate": 0.0,
            "latest_event": {"initial_pending_lines": 27000, "final_pending_lines": 29644, "waves_run": 1},
        },
        storage={
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {
                "core_pending_lines": 29644,
                "deferred_pending_lines": 700,
                "support_pending_lines": 31,
                "cold_pending_lines": 0,
                "total_pending_lines": 30344,
                "oldest_pending_age_seconds": 425751.199,
            },
        },
        runtime={
            "overall_status": "blocked",
            "throttle_profile": "protect_live",
            "host_saturation_score": 68.6,
            "compute_pressure_level": "high",
            "memory_pressure_level": "normal",
        },
        memory_efficiency={
            "overall_status": "blocked",
            "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"},
        },
        writer={"active": False},
        target_pending_lines=10000,
    )

    packet = payload["backlog_needs_packet"]
    core_need = next(row for row in packet["needs"] if row["section_id"] == "core_decision")

    assert packet["overall_status"] == "needs_attention"
    assert packet["top_need_section"] == "core_decision"
    assert packet["accelerator_contract"]["log_fix_frames"] is True
    assert packet["fix_reference_frame"]["primary_blocker"] == "core_decision"
    assert "core_pending_lines <= 15000 for a C grade" in core_need["exit_criteria"]
    assert "core_pending_lines <= 5000" in core_need["a_grade_exit_criteria"]
    assert payload["backlog_needs_packet"]["a_grade_lift_contract"]["target_grade"] == "A"
    assert "core_decision" in payload["backlog_needs_packet"]["a_grade_lift_contract"]["blocking_sections"]
    assert "oldest_pending_age_seconds" in core_need["measurements_to_check"]
    assert core_need["accelerator"] == "core_decision_drainer"
    assert "backlog_drain_needs" in payload["control_contract"]["feeds"]


def test_drainer_intelligence_verifies_measurement_when_wave_progress_is_invisible() -> None:
    payload = src.build_intelligence_from_payloads(
        fleet=_fleet(writer_lock_held=False),
        super_drainer={"overall_status": "applied_with_followups", "active_drainer": "core_decision_drainer"},
        memory={
            "history_count": 6,
            "recent_progress_rate": 0.3,
            "recent_target_met_rate": 0.0,
            "latest_event": {
                "initial_pending_lines": 30344,
                "final_pending_lines": 30344,
                "pending_lines_net_change": 0,
                "waves_run": 1,
                "progress_waves": 1,
                "target_met": False,
                "refill_detected": False,
            },
        },
        storage={
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {
                "core_pending_lines": 29644,
                "deferred_pending_lines": 700,
                "support_pending_lines": 31,
                "total_pending_lines": 30344,
                "oldest_pending_age_seconds": 425751.199,
            },
        },
        runtime={"overall_status": "ready"},
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={"active": False},
        target_pending_lines=10000,
    )

    assert payload["decision_packet"]["action"] == "verify_drain_measurement_then_re_score"
    assert "visible_pending_progress_missing" in payload["decision_packet"]["risk_flags"]
    assert payload["drain_playbook"][1]["step"] == "refresh_drainer_fleet"
    core_need = next(row for row in payload["backlog_needs_packet"]["needs"] if row["section_id"] == "core_decision")
    assert "pending_lines_delta" in core_need["measurements_to_check"]
    assert payload["learning_summary"]["latest_no_visible_pending_progress"] is True


def test_fix_reference_ledger_deduplicates_same_need_snapshot(tmp_path) -> None:
    payload = src.build_intelligence_from_payloads(
        fleet=_fleet(writer_lock_held=False),
        super_drainer={"overall_status": "ready", "active_drainer": "core_decision_drainer"},
        memory={"history_count": 2, "recent_progress_rate": 0.5, "recent_target_met_rate": 0.0},
        storage={
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {
                "core_pending_lines": 29644,
                "deferred_pending_lines": 700,
                "support_pending_lines": 31,
                "total_pending_lines": 30344,
                "oldest_pending_age_seconds": 425751.199,
            },
        },
        runtime={"overall_status": "ready"},
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={"active": False},
        target_pending_lines=10000,
    )
    ledger = tmp_path / "backlog_drain_fix_ledger.jsonl"

    src._append_fix_reference_if_changed(ledger, payload)
    src._append_fix_reference_if_changed(ledger, payload)

    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["event_type"] == "backlog_drain_fix_reference_frame"
    assert rows[0]["top_need_section"] == "core_decision"


def test_current_writer_orphaned_state_overrides_stale_super_writer_active() -> None:
    payload = src.build_intelligence_from_payloads(
        fleet=_fleet(writer_lock_held=False),
        super_drainer={
            "overall_status": "waiting_for_writer",
            "active_drainer": "core_decision_drainer",
            "writer_state_before": {
                "active": True,
                "current_step": "shard_linking",
                "progress_age_minutes": 4.0,
            },
        },
        memory={"history_count": 4, "recent_progress_rate": 0.5, "recent_target_met_rate": 0.0},
        storage={"overall_status": "blocked", "severity": "critical", "backpressure": {"total_pending_lines": 13200}},
        runtime={"overall_status": "ready"},
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={
            "writer_state_before": {
                "active": False,
                "active_source": "orphaned_progress",
                "progress_orphaned": True,
                "progress_age_minutes": 8.0,
            }
        },
        target_pending_lines=10000,
    )

    decision = payload["decision_packet"]
    assert "writer_progress_orphaned" in decision["risk_flags"]
    assert "writer_active" not in decision["risk_flags"]
    assert decision["action"] == "run_bounded_wave"
    assert decision["writer_health"]["state"] == "orphaned_progress"


def test_current_writer_after_wait_overrides_active_before_state() -> None:
    payload = src.build_intelligence_from_payloads(
        fleet=_fleet(writer_lock_held=False),
        super_drainer={"overall_status": "ready", "active_drainer": "core_decision_drainer"},
        memory={"history_count": 4, "recent_progress_rate": 0.5, "recent_target_met_rate": 0.0},
        storage={"overall_status": "blocked", "severity": "critical", "backpressure": {"total_pending_lines": 13200}},
        runtime={"overall_status": "ready"},
        memory_efficiency={"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}},
        writer={
            "writer_state_before": {
                "active": True,
                "active_source": "writer_lock",
                "current_step": "merge_primary",
                "progress_age_minutes": 4.0,
            },
            "writer_state_after_wait": {
                "active": False,
                "active_source": "idle",
                "current_step": "complete",
                "progress_age_minutes": 0.1,
                "completed_merge_count": 5,
            },
        },
        target_pending_lines=10000,
    )

    decision = payload["decision_packet"]
    assert "writer_active" not in decision["risk_flags"]
    assert decision["writer_health"]["state"] == "idle"
    assert decision["action"] == "run_bounded_wave"
