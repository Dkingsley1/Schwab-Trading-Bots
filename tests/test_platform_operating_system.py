from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import platform_operating_system as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_project(project_root: Path) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v10",
                    "active": True,
                    "master_candidate": True,
                    "sleeve_profile": "crypto_futures",
                    "training_candidate_after_threshold": True,
                    "quality_score": 0.91,
                },
                {
                    "bot_id": "brain_refinery_v35",
                    "active": True,
                    "sleeve_profile": "default",
                    "needs_runtime_input_repair": True,
                    "exclude_from_training": True,
                    "quality_score": 0.99,
                },
            ]
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "timestamp_utc": "2099-01-01T00:00:00+00:00",
            "overall_status": "blocked",
            "pressure_index": 4.2,
            "backpressure": {
                "total_pending_lines": 22000,
                "core_pending_lines": 18000,
                "deferred_pending_lines": 4000,
                "support_pending_lines": 0,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 900,
                "oldest_age_threshold_seconds": 240,
            },
        },
    )
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "timestamp_utc": "2099-01-01T00:00:00+00:00",
            "overall_status": "ready",
            "writer_state_before": {"current_step": "complete", "completed_shard_count": 25, "planned_shard_count": 25},
        },
    )
    _write_json(health / "writer_process_intelligence_latest.json", {"timestamp_utc": "2099-01-01T00:00:00+00:00", "overall_status": "ready"})
    _write_json(health / "drainer_intelligence_layer_latest.json", {"timestamp_utc": "2099-01-01T00:00:00+00:00", "overall_status": "ready"})
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"timestamp_utc": "2099-01-01T00:00:00+00:00", "overall_status": "ready", "host_saturation_score": 41.0},
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "timestamp_utc": "2099-01-01T00:00:00+00:00",
            "overall_status": "ready",
            "memory_snapshot": {"swap_used_gb": 1.1},
        },
    )
    _write_json(
        health / "computer_task_intelligence_latest.json",
        {"timestamp_utc": "2099-01-01T00:00:00+00:00", "overall_status": "ready", "foreground_apps": ["Logic Pro"]},
    )
    _write_json(health / "autonomic_resource_governor_latest.json", {"timestamp_utc": "2099-01-01T00:00:00+00:00", "overall_status": "ready"})
    _write_json(health / "all_sleeves_launcher_latest.json", {"timestamp_utc": "2099-01-01T00:00:00+00:00", "overall_status": "ready"})
    _write_json(health / "watchdog_intelligence_latest.json", {"timestamp_utc": "2099-01-01T00:00:00+00:00", "overall_status": "ready"})
    _write_json(health / "process_watchdog_latest.json", {"timestamp_utc": "2099-01-01T00:00:00+00:00", "overall_status": "ready"})
    _write_json(
        health / "source_verification_latest.json",
        {"timestamp_utc": "2099-01-01T00:00:00+00:00", "overall_status": "ready", "required_ready_count": 4, "required_total_count": 4},
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "timestamp_utc": "2099-01-01T00:00:00+00:00",
            "overall_status": "ready",
            "reentry_gate": {"memory_batch10_safe": True, "memory_batch20_safe": False},
        },
    )
    _write_json(health / "training_quality_control_latest.json", {"timestamp_utc": "2099-01-01T00:00:00+00:00", "overall_status": "ready"})
    _write_json(health / "bot_quality_autopilot_latest.json", {"timestamp_utc": "2099-01-01T00:00:00+00:00", "overall_status": "ready"})
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "timestamp_utc": "2099-01-01T00:00:00+00:00",
            "overall_status": "protective_tightening",
            "profitability_grade": "A",
            "financial_profitability_grade": "A",
            "operational_control_grade": "A+",
            "operational_outcome_grade": "A+",
            "raw_operational_outcome_grade": "B",
            "realized_pnl_total": 25.0,
            "unrealized_pnl_total": 75.0,
            "profit_harvest_report_card": {
                "headline_grade": "A+",
                "control_grade": "A+",
                "raw_outcome_grade": "D",
                "control_score_norm": 0.999,
            },
            "active_profile_controls": {
                "crypto_futures": {
                    "daily_harvest_goal": {
                        "active": True,
                        "target_pnl": 120.0,
                        "block_new_adds": True,
                        "expand_collection_after_target_met": True,
                    }
                }
            },
            "paper_harvest_execution_intent_count": 9,
        },
    )
    _write_json(
        health / "paper_runtime_profitability_controls_latest.json",
        {
            "timestamp_utc": "2099-01-01T00:00:00+00:00",
            "overall_status": "ready",
            "profile_controls": {"crypto_futures": {"daily_harvest_goal": {"active": True, "target_pnl": 120.0}}},
            "strategy_controls": {"v10": {"mode": "paper_only"}},
        },
    )
    _write_json(health / "auth_lease_manager_latest.json", {"timestamp_utc": "2099-01-01T00:00:00+00:00", "overall_status": "ready"})
    _write_json(health / "global_killswitch_latest.json", {"timestamp_utc": "2099-01-01T00:00:00+00:00", "overall_status": "ready"})
    _write_json(health / "system_needs_intelligence_latest.json", {"timestamp_utc": "2099-01-01T00:00:00+00:00", "overall_status": "ready"})


def test_platform_operating_system_builds_all_eight_sections(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path, ledger_path=tmp_path / "governance" / "platform_os" / "system_event_ledger.jsonl")

    assert payload["section_count"] == 8
    assert payload["all_eight_sections_active"] is True
    assert payload["raw_platform_grade"] in {"A", "A+", "A+", "B", "C", "D"}
    assert payload["platform_grade"] in {"A", "A+", "A+"}
    assert payload["control_credit"]["points"] > 0
    assert set(payload["sections"]) == set(src.SECTION_ARTIFACTS)
    assert payload["invariants"]["live_execution_authority_added"] is False
    assert payload["invariants"]["do_not_touch_video_volume"] is True
    assert "/Volumes/VIDEO" in payload["invariants"]["protected_volume_denylist"]
    assert payload["sections"]["slo_control"]["breach_count"] >= 2
    assert payload["sections"]["slo_control"]["status"] == "guarded"
    assert payload["sections"]["slo_control"]["section_grade"] in {"A+", "A+"}
    assert payload["sections"]["slo_control"]["all_sections_a_plus"] is True
    assert payload["sections"]["slo_control"]["low_section_cards"] == []
    assert payload["sections"]["slo_control"]["section_report_card"]
    assert all(
        row["section_grade"] in {"A+", "A+"}
        for row in payload["sections"]["slo_control"]["section_report_card"]
    )
    assert payload["sections"]["slo_control"]["outcome_grade"] in {"C", "D", "F"}
    assert payload["sections"]["release_train"]["status"] == "gated"
    assert payload["sections"]["paper_execution_truth_layer"]["truth_invariants"]["paper_only"] is True
    assert payload["sections"]["human_coexistence_layer"]["status"] == "user_app_priority"
    assert payload["sections"]["bot_lifecycle_state_machine"]["repair_first_sample"][0]["bot_id"] == "brain_refinery_v35"


def test_platform_operating_system_writes_artifacts_and_dedupes_events(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    ledger = tmp_path / "governance" / "platform_os" / "system_event_ledger.jsonl"
    out = tmp_path / "governance" / "health" / "platform_operating_system_latest.json"
    config = tmp_path / "config" / "platform_operating_system.json"

    payload = src.build_payload(tmp_path, ledger_path=ledger)
    apply_result = src.write_outputs(tmp_path, payload, out_path=out, ledger_path=ledger, config_path=config, apply=True)

    assert out.exists()
    assert config.exists()
    assert apply_result["events_appended"] > 0
    assert all((tmp_path / "governance" / "platform_os" / filename).exists() for filename in src.SECTION_ARTIFACTS.values())

    payload = src.build_payload(tmp_path, ledger_path=ledger)
    apply_result = src.write_outputs(tmp_path, payload, out_path=out, ledger_path=ledger, config_path=config, apply=True)

    assert apply_result["events_appended"] == 0
