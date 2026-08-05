from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import distributed_cell_architecture as cells


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_distributed_cell_architecture_builds_seven_cells(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "whole_system_intelligence_latest.json", {"timestamp_utc": cells.iso_now(), "overall_status": "ready"})
    _write_json(health / "ingestion_storage_control_latest.json", {"timestamp_utc": cells.iso_now(), "overall_status": "ready"})
    _write_json(health / "training_runtime_control_latest.json", {"timestamp_utc": cells.iso_now(), "overall_status": "ready"})
    _write_json(health / "macro_event_intelligence_latest.json", {"timestamp_utc": cells.iso_now(), "overall_status": "ready"})
    _write_json(health / "paper_profitability_control_latest.json", {"timestamp_utc": cells.iso_now(), "overall_status": "ready"})

    payload = cells.build_payload(project_root=tmp_path, apply=False, cell_root=tmp_path / "governance" / "cells")

    assert payload["cell_count"] == 7
    assert {row["cell_id"] for row in payload["cells"]} == {
        "control_plane",
        "sleeve_cells",
        "storage_writer_cell",
        "training_cell",
        "market_data_cell",
        "execution_paper_cell",
        "infra_cell",
    }
    assert payload["federation_contract"]["single_writer_authority"] == "storage_writer_cell"
    assert payload["intercell_bus"]["single_writer_authority"] == "storage_writer_cell"
    assert payload["distributed_runtime_arbitration"]["parallel_sqlite_commit_writers_allowed"] is False
    assert "storage_writer_cell" in payload["cell_dependency_graph"]["training_cell"]["depends_on"]
    market_contract = payload["cell_resource_contracts"]["market_data_cell"]
    assert market_contract["primary_budget"] == "required_context_first_optional_news_bounded"
    market_state = next(row for row in cells.CELL_DEFINITIONS if row["cell_id"] == "market_data_cell")
    assert "ticker_news_context" in {row["name"] for row in market_state["surfaces"]}
    infra_state = next(row for row in cells.CELL_DEFINITIONS if row["cell_id"] == "infra_cell")
    infra_paths = {row["path"] for row in infra_state["surfaces"]}
    assert "governance/health/infrastructure_autofix_bot_latest.json" in infra_paths
    assert "governance/health/infrabot_library_self_awareness_control_latest.json" in infra_paths
    assert payload["protected_volumes"]["VIDEO"] == "never_touched"
    assert "/Volumes/VIDEO" in payload["integration_contract"]["never_touch_protected_volumes"]


def test_distributed_cell_architecture_separates_guarded_soak_from_raw_backlog(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    now = cells.iso_now()
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "overall_grade": "A+",
            "safe_to_leave_unattended": True,
            "blockers": [],
        },
    )
    _write_json(
        health / "runtime_paper_regression_guard_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "paper_armed": True,
            "paper_blocked": False,
            "failed_guard_count": 0,
        },
    )
    _write_json(health / "health_fast_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True, "strict_all_clear": True})
    _write_json(health / "runtime_gate_dashboard_latest.json", {"timestamp_utc": now, "overall": {"status": "ok", "ok": True, "attention": []}})
    _write_json(
        health / "system_drift_guard_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "ok": True,
            "metrics": {"blocked_surface_count": 0, "degraded_surface_count": 0, "stale_surface_count": 0},
        },
    )
    _write_json(health / "training_quality_control_latest.json", {"timestamp_utc": now, "overall_status": "needs_attention"})

    payload = cells.build_payload(project_root=tmp_path, apply=False, cell_root=tmp_path / "governance" / "cells")

    assert payload["operational_health"]["status"] == "ready"
    assert payload["operational_health"]["grade"] == "A+"
    assert payload["operational_health"]["managed_raw_need_count"] > 0
    assert payload["raw_operational_health"]["status"] == "blocked"
    assert payload["integration_contract"]["separates_guarded_soak_health_from_raw_production_backlog"] is True


def test_distributed_cell_architecture_markdown_includes_sleeve_guard_posture(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    now = cells.iso_now()
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "protective_tightening",
            "ok": True,
            "controlled_profitability_grade": "A+",
            "raw_profitability_grade": "D",
            "financial_profitability_grade": "D",
            "weak_sleeve_recurrence_guard_contract": {
                "control_ready": True,
                "control_posture_grade": "A+",
                "paper_only": True,
                "live_execution_allowed": False,
                "profile_count": 2,
                "guarded_profile_count": 2,
                "top_recurrent_loss_causes": [
                    {"cause": "source_quality:low", "profile_count": 2},
                    {"cause": "session:intraday", "profile_count": 2},
                ],
            },
            "weak_sleeve_systemic_weak_point_contract": {
                "active": True,
                "control_ready": True,
                "control_posture_grade": "A+",
                "paper_only": True,
                "live_execution_allowed": False,
                "systemic_weak_point_count": 1,
                "top_systemic_causes": [
                    {"cause": "source_quality:low", "family": "source_quality", "profile_count": 2},
                ],
            },
        },
    )

    payload = cells.build_payload(project_root=tmp_path, apply=False, cell_root=tmp_path / "governance" / "cells")
    markdown = cells._markdown(payload)

    assert payload["sleeve_guard_posture"]["posture"] == "paper_repair_guarded_with_systemic_weak_point_locks"
    assert payload["integration_contract"]["includes_sleeve_weak_point_recurrence_and_systemic_guard_posture"] is True
    assert "Sleeve guard posture:" in markdown
    assert "Recurrence guarded: 2/2" in markdown
    assert "Systemic weak points: 1" in markdown
    assert "controlled A+" in markdown
    assert "raw D" in markdown
    assert "source_quality:low" in markdown


def test_distributed_cell_architecture_markdown_has_stable_empty_needs_footer() -> None:
    markdown = cells._markdown({"top_needs": []})

    assert "## Next Needs\n\n- None.\n" in markdown
    assert markdown.endswith("- None.\n")


def test_distributed_cell_runtime_markdown_stays_out_of_tracked_docs() -> None:
    relative_path = cells.DEFAULT_MARKDOWN_PATH.relative_to(cells.PROJECT_ROOT)

    assert relative_path.as_posix() == "governance/reports/distributed_cell_architecture_latest.md"


def test_distributed_cell_architecture_normalizes_controlled_production_states(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    now = cells.iso_now()
    _write_json(
        health / "whole_system_intelligence_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "degraded",
            "system_signal_bus": {
                "summary": {
                    "blocked_signal_count": 0,
                    "severe_signal_count": 0,
                    "storage_critical": False,
                    "memory_pressure_high": False,
                    "runtime_pressure_high": False,
                    "writer_recovery_required": False,
                    "global_halt_active": False,
                }
            },
            "system_process_contracts": {"blocked_contract_count": 0},
        },
    )
    for name in ("whole_system_governor", "autonomic_resource_governor", "runtime_throttle_control", "memory_pressure_intelligence"):
        _write_json(health / f"{name}_latest.json", {"timestamp_utc": now, "overall_status": "ready"})
    _write_json(health / "sleeve_ticker_universe_latest.json", {"timestamp_utc": now, "overall_status": "ready"})
    _write_json(health / "backlog_pump_infrabots_latest.json", {"timestamp_utc": now, "overall_status": "ready"})
    _write_json(health / "paper_profitability_control_latest.json", {"timestamp_utc": now, "overall_status": "protective_tightening", "ok": True})
    _write_json(
        health / "data_collection_observation_rollup_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "degraded",
            "zero_observation_repair_lane": {"active": True},
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"timestamp_utc": now, "overall_status": "ready"})
    _write_json(
        health / "storage_quota_guard_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "degraded",
            "quota_summary": {
                "hard_breaches": 0,
                "soft_breaches": 1,
                "blocked_families": [],
                "degraded_families": ["sql_link_shards"],
            },
        },
    )
    _write_json(health / "storage_backpressure_autopilot_latest.json", {"timestamp_utc": now, "overall_status": "running", "ok": True})
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "waiting_for_writer",
            "ok": True,
            "summary": {"writer_active_after_wait": True},
        },
    )
    _write_json(health / "writer_process_intelligence_latest.json", {"timestamp_utc": now, "overall_status": "ready"})
    _write_json(health / "backlog_pcore_accelerator_latest.json", {"timestamp_utc": now, "overall_status": "ready"})
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "constrained",
            "launch_blockers": ["autonomic_training_budget_closed"],
            "prep_allowed": True,
            "resource_guard": {"training_ok": True},
            "storage_quota_training_gate": {"hard_breaches": 0},
        },
    )
    _write_json(
        health / "training_quality_control_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "needs_attention",
            "training_quality_score": 100.0,
            "failure_taxonomy": {"failure_buckets": ["coverage_shortfall", "training_not_confirmed"]},
            "rollout": {"exact_replay_ready": True},
        },
    )
    _write_json(health / "training_data_intake_expansion_latest.json", {"timestamp_utc": now, "overall_status": "ready"})
    _write_json(health / "training_labeling_intelligence_latest.json", {"timestamp_utc": now, "overall_status": "ready"})
    _write_json(health / "macro_event_intelligence_latest.json", {"timestamp_utc": now, "overall_status": "ready"})
    _write_json(health / "schwab_symbol_news_latest.json", {"timestamp_utc": now, "overall_status": "ready_public_schwab_fallback"})
    _write_json(
        health / "master_infrastructure_supervisor_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "degraded",
            "metrics": {"blocked_check_count": 0, "hard_failed_attempt_count": 0, "degraded_attempt_count": 0},
        },
    )

    cell_root = tmp_path / "governance" / "cells"
    payload = cells.build_payload(project_root=tmp_path, apply=True, cell_root=cell_root)
    surfaces = {
        row["name"]: row
        for state_path in cell_root.glob("*/state.json")
        for row in json.loads(state_path.read_text(encoding="utf-8"))["surfaces"]
    }

    assert payload["raw_operational_health"]["status"] == "ready"
    assert payload["raw_operational_health"]["grade"] == "A+"
    assert payload["top_needs"] == []
    assert surfaces["paper_profitability_control"]["raw_status"] == "protective_tightening"
    assert surfaces["paper_profitability_control"]["controlled_state_reason"] == "profitability_protective_tightening_is_controlled_risk_posture"
    assert surfaces["writer_cycle_coordinator"]["raw_status"] == "waiting_for_writer"
    assert surfaces["storage_quota_guard"]["controlled_state_reason"] == "stateful_sql_soft_quota_compaction_debt_managed_by_guarded_soak"
    assert surfaces["training_runtime"]["controlled_state_reason"] == "training_budget_closed_is_managed_during_guarded_paper_soak"
    assert surfaces["data_collection_observation_rollup"]["controlled_state_reason"] == "zero_observation_targets_are_routed_to_targeted_repair_lane"
    assert surfaces["master_infrastructure_supervisor"]["controlled_state_reason"] == "master_infrastructure_degraded_only_by_advisory_refreshable_checks"
    assert cells._controlled_surface_state("writer_cycle_coordinator", "handoff_released", {"ok": True})["status"] == "complete"


def test_guarded_soak_accepts_architecture_self_reference_drift_debt(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    now = cells.iso_now()
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "overall_grade": "A+",
            "safe_to_leave_unattended": True,
            "blockers": [],
        },
    )
    _write_json(
        health / "runtime_paper_regression_guard_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "paper_armed": True,
            "paper_blocked": False,
            "failed_guard_count": 0,
        },
    )
    _write_json(health / "health_fast_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True, "strict_all_clear": True})
    _write_json(health / "runtime_gate_dashboard_latest.json", {"timestamp_utc": now, "overall": {"status": "ok", "ok": True, "attention": []}})
    _write_json(
        health / "system_drift_guard_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "degraded",
            "ok": False,
            "metrics": {"blocked_surface_count": 0, "degraded_surface_count": 1, "stale_surface_count": 0},
            "surfaces": [{"name": "system_architecture_autopilot", "status": "degraded"}],
        },
    )

    guarded = cells._guarded_paper_soak_health(tmp_path)

    assert guarded["ready"] is True
    assert guarded["system_drift_ready"] is True
    assert guarded["system_drift_context"]["managed"] is True
