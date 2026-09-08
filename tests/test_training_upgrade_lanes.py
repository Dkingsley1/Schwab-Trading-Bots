import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import calibration_abstention_control as calibration_src
from scripts.ops import coverage_gap_closer as gap_closer_src
from scripts.ops import training_requalification_lane as requal_src
from scripts.ops import walk_forward_coverage_seed as coverage_src
import scripts.retrain_lane_scheduler as lane_scheduler_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_training_requalification_lane_surfaces_ready_candidate(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v1",
                    "active": False,
                    "lifecycle_state": "inactive_backlog",
                    "quality_score": 0.61,
                    "bot_role": "signal_sub_bot",
                }
            ]
        },
    )
    (project_root / "models").mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)
    (project_root / "models" / "brain_refinery_v1_20260401.npz").write_text("model", encoding="utf-8")
    (project_root / "logs" / "brain_refinery_v1_20260401.json").write_text("{}", encoding="utf-8")
    _write_json(project_root / "governance" / "training_diagnostics" / "brain_refinery_v1_latest.json", {"status": "ok", "sample_count": 40})

    payload = requal_src.build_payload(project_root)

    assert payload["candidate_count"] == 1
    assert payload["reactivation_ready_count"] == 1
    assert payload["top_candidates"][0]["actions"] == ["seed_walk_forward_coverage"]


def test_training_requalification_lane_does_not_label_missing_diagnostic_as_runtime_input_gap(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v35_dmi_state_machine",
                    "active": False,
                    "lifecycle_state": "inactive_backlog",
                    "quality_score": 0.99,
                    "test_accuracy": 0.85,
                    "bot_role": "signal_sub_bot",
                }
            ]
        },
    )
    (project_root / "models").mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)
    (project_root / "models" / "brain_refinery_v35_dmi_state_machine_20260401.npz").write_text("model", encoding="utf-8")
    (project_root / "logs" / "brain_refinery_v35_dmi_state_machine_20260401.json").write_text("{}", encoding="utf-8")

    payload = requal_src.build_payload(project_root)

    actions = payload["top_candidates"][0]["actions"]
    assert "refresh_training_diagnostics" in actions
    assert "repair_runtime_inputs" not in actions


def test_training_requalification_lane_keeps_high_quality_probation_candidates_in_coverage_pool(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v35_dmi_state_machine",
                    "active": False,
                    "lifecycle_state": "probation",
                    "quality_score": 0.99,
                    "test_accuracy": 0.85,
                    "bot_role": "signal_sub_bot",
                }
            ]
        },
    )
    (project_root / "models").mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)
    (project_root / "models" / "brain_refinery_v35_dmi_state_machine_20260404.npz").write_text("model", encoding="utf-8")
    (project_root / "logs" / "brain_refinery_v35_dmi_state_machine_20260404.json").write_text("{}", encoding="utf-8")
    _write_json(
        project_root / "governance" / "training_diagnostics" / "brain_refinery_v35_dmi_state_machine_latest.json",
        {"status": "deferred_sample_starved", "sample_count": 0},
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "walk_forward_latest.json",
        {"bots": {"brain_refinery_v35_dmi_state_machine": {"runs": 1, "status": "insufficient_runs"}}},
    )

    payload = requal_src.build_payload(project_root)

    assert payload["candidate_count"] == 1
    assert payload["top_candidates"][0]["walk_forward_runs"] == 1
    assert payload["top_candidates"][0]["actions"] == ["repair_runtime_inputs", "seed_walk_forward_coverage"]


def test_training_requalification_lane_promotes_regime_fit_bootstrap_candidates(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v101_guard_heavy_regime_memory",
                    "active": False,
                    "lifecycle_state": "inactive",
                    "quality_score": 0.0,
                    "bot_role": "signal_sub_bot",
                    "reason": "new_runtime_candidate",
                    "promotion_reason": "new_runtime_candidate",
                }
            ]
        },
    )
    _write_json(
        project_root / "governance" / "health" / "roster_resilience_planner_latest.json",
        {
            "current_regime": {
                "live_regime": "risk_off_shock",
                "regime_fit_replacements": [
                    {
                        "bot_id": "brain_refinery_v101_guard_heavy_regime_memory",
                        "regime_fit_score": 3,
                    }
                ],
            }
        },
    )

    payload = requal_src.build_payload(project_root)

    assert payload["current_regime"]["live_regime"] == "risk_off_shock"
    assert payload["current_regime"]["priority_candidate_count"] == 1
    assert payload["top_candidates"][0]["bot_id"] == "brain_refinery_v101_guard_heavy_regime_memory"
    assert payload["top_candidates"][0]["bootstrap_candidate"] is True
    assert payload["top_candidates"][0]["current_regime_priority"] is True
    assert payload["top_candidates"][0]["actions"] == ["rebuild_model_artifact", "repair_runtime_inputs", "seed_walk_forward_coverage"]


def test_training_requalification_lane_can_surface_deleted_sample_starved_collection_candidate(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v23_atr_adx_regime",
                    "active": False,
                    "deleted_from_rotation": True,
                    "lifecycle_state": "deleted",
                    "quality_score": 0.25,
                    "bot_role": "signal_sub_bot",
                }
            ]
        },
    )
    _write_json(
        project_root / "governance" / "training_diagnostics" / "brain_refinery_v23_atr_adx_regime_latest.json",
        {
            "status": "deferred_sample_starved",
            "sample_count": 0,
            "eligible_sequences": 0,
            "observation_count": 6,
            "sequence_count": 3,
        },
    )

    default_payload = requal_src.build_payload(project_root)
    payload = requal_src.build_payload(project_root, include_sample_starved_deleted=True)

    assert default_payload["candidate_count"] == 0
    assert payload["sample_starved_collection_candidate_count"] == 1
    assert payload["top_candidates"][0]["bot_id"] == "brain_refinery_v23_atr_adx_regime"
    assert payload["top_candidates"][0]["sample_starved_requalification_candidate"] is True
    assert "reactivate_collection_only" in payload["top_candidates"][0]["actions"]
    assert "repair_runtime_inputs" in payload["top_candidates"][0]["actions"]


def test_training_requalification_apply_repair_reactivates_deleted_sample_starved_as_collect_only(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v37_order_flow_proxy",
                    "active": False,
                    "deleted_from_rotation": True,
                    "lifecycle_state": "deleted",
                    "quality_score": 0.49,
                    "bot_role": "signal_sub_bot",
                    "weight": 0.3,
                }
            ]
        },
    )
    _write_json(
        project_root / "governance" / "training_diagnostics" / "brain_refinery_v37_order_flow_proxy_latest.json",
        {
            "status": "deferred_sample_starved",
            "sample_count": 0,
            "eligible_sequences": 0,
            "observation_count": 220,
            "sequence_count": 47,
        },
    )

    repair = requal_src.apply_repairs(
        project_root,
        include_bot_ids=["brain_refinery_v37_order_flow_proxy"],
        include_sample_starved_deleted=True,
    )
    registry = json.loads((project_root / "master_bot_registry.json").read_text(encoding="utf-8"))
    row = registry["sub_bots"][0]

    assert repair["collection_requalification_count"] == 1
    assert repair["unresolved_count"] == 0
    assert row["active"] is True
    assert row["deleted_from_rotation"] is False
    assert row["lifecycle_state"] == "data_collection_only"
    assert row["data_collection_active"] is True
    assert row["training_excluded"] is True
    assert row["exclude_from_training"] is True
    assert row["trading_enabled"] is False
    assert row["execution_enabled"] is False
    assert row["paper_execution_allowed"] is False
    assert row["weight"] == 0.0
    assert row["minimum_training_observations"] >= 200


def test_training_requalification_uses_fresh_bot_needs_to_repair_missing_diagnostic(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v11_stoch_vol"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": bot_id,
                    "active": False,
                    "deleted_from_rotation": True,
                    "lifecycle_state": "deleted",
                    "bot_role": "signal_sub_bot",
                    "weight": 0.4,
                    "allocation_weight": 0.4,
                    "paper_trading_enabled": True,
                }
            ]
        },
    )
    _write_json(
        project_root / "governance" / "health" / "bot_needs_intelligence_latest.json",
        {
            "timestamp_utc": "2026-07-31T23:46:55+00:00",
            "ok": True,
            "bot_needs": [
                {
                    "bot_id": bot_id,
                    "primary_need": "reactivate_sample_starved_collection",
                    "evidence": {"sample_count": 0, "eligible_sequences": 0, "observation_count": 7},
                }
            ],
        },
    )

    payload = requal_src.build_payload(project_root, include_sample_starved_deleted=True)

    assert payload["sample_starved_collection_candidate_count"] == 1
    assert payload["bot_needs_reactivation_authority"]["status"] == "ready"
    assert payload["top_candidates"][0]["bot_needs_authorized_missing_diagnostic"] is True
    assert "refresh_training_diagnostics" not in payload["top_candidates"][0]["actions"]

    repair = requal_src.apply_repairs(
        project_root,
        include_bot_ids=[bot_id],
        include_sample_starved_deleted=True,
    )
    registry = json.loads((project_root / "master_bot_registry.json").read_text(encoding="utf-8"))
    row = registry["sub_bots"][0]
    diagnostic_path = project_root / "governance" / "training_diagnostics" / f"{bot_id}_latest.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))

    assert repair["collection_requalification_count"] == 1
    assert repair["synthesized_diagnostic_count"] == 1
    assert repair["unresolved_count"] == 0
    assert Path(repair["registry_backup_path"]).exists()
    assert diagnostic["status"] == "deferred_sample_starved"
    assert diagnostic["reason"] == "bot_needs_missing_diagnostic_collection_requalification"
    assert diagnostic["training_performed"] is False
    assert diagnostic["master_update_applied"] is False
    assert diagnostic["execution_allowed"] is False
    assert diagnostic["paper_execution_allowed"] is False
    assert diagnostic["live_execution_allowed"] is False
    assert diagnostic["runtime_meta"]["synthetic_missing_diagnostic_repair"] is True
    assert row["active"] is True
    assert row["lifecycle_state"] == "data_collection_only"
    assert row["data_collection_active"] is True
    assert row["training_excluded"] is True
    assert row["paper_trading_enabled"] is False
    assert row["paper_execution_allowed"] is False
    assert row["execution_enabled"] is False
    assert row["live_trading_enabled"] is False
    assert row["weight"] == 0.0
    assert row["allocation_weight"] == 0.0


def test_training_requalification_rejects_stale_bot_needs_missing_diagnostic_authority(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v12_stale_authority"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": bot_id,
                    "active": False,
                    "deleted_from_rotation": True,
                    "lifecycle_state": "deleted",
                }
            ]
        },
    )
    bot_needs_path = project_root / "governance" / "health" / "bot_needs_intelligence_latest.json"
    _write_json(
        bot_needs_path,
        {
            "ok": True,
            "bot_needs": [
                {
                    "bot_id": bot_id,
                    "primary_need": "reactivate_sample_starved_collection",
                    "evidence": {"sample_count": 0, "eligible_sequences": 0},
                }
            ],
        },
    )
    stale_timestamp = time.time() - (requal_src.BOT_NEEDS_REACTIVATION_MAX_AGE_HOURS + 1.0) * 3600.0
    os.utime(bot_needs_path, (stale_timestamp, stale_timestamp))

    payload = requal_src.build_payload(project_root, include_sample_starved_deleted=True)

    assert payload["sample_starved_collection_candidate_count"] == 0
    assert payload["bot_needs_reactivation_authority"]["status"] == "stale"


def test_training_requalification_preserves_true_collect_only_bootstrap(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v1614_collect_only"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": bot_id,
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                    "data_collection_active": True,
                    "training_excluded": True,
                    "bot_role": "infrastructure_sub_bot",
                }
            ]
        },
    )
    _write_json(project_root / "logs" / f"{bot_id}_latest.json", {"status": "failed", "sample_count": 1})
    diagnostic_path = project_root / "governance" / "training_diagnostics" / f"{bot_id}_latest.json"
    _write_json(
        diagnostic_path,
        {
            "status": "collect_only_label_contract_ready",
            "sample_count": 0,
            "runtime_meta": {"diagnostic_kind": "collect_only_label_contract_bootstrap"},
        },
    )

    repair = requal_src.apply_repairs(project_root, include_bot_ids=[bot_id])
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))

    assert repair["repaired_rows"][0]["diagnostic_rebuilt"] is False
    assert diagnostic["status"] == "collect_only_label_contract_ready"


def test_training_requalification_replaces_bootstrap_for_paper_live_bot(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v17_mixed_regime"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": bot_id,
                    "active": True,
                    "lifecycle_state": "paper_live_data",
                    "data_collection_active": True,
                    "training_excluded": True,
                    "bot_role": "signal_sub_bot",
                }
            ]
        },
    )
    _write_json(
        project_root / "logs" / f"{bot_id}_latest.json",
        {
            "metrics": {
                "acted_count": 40,
                "acted_accuracy": 0.7,
                "accuracy_lift_over_majority": 0.12,
                "positive_rate": 0.5,
            }
        },
    )
    diagnostic_path = project_root / "governance" / "training_diagnostics" / f"{bot_id}_latest.json"
    _write_json(
        diagnostic_path,
        {
            "status": "collect_only_label_contract_ready",
            "sample_count": 0,
            "runtime_meta": {"diagnostic_kind": "collect_only_label_contract_bootstrap"},
        },
    )

    repair = requal_src.apply_repairs(project_root, include_bot_ids=[bot_id])
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))

    assert repair["repaired_rows"][0]["diagnostic_rebuilt"] is True
    assert diagnostic["status"] == "passed"
    assert diagnostic["repaired_from_log"] is True


def test_training_requalification_apply_repair_restores_registry_and_diagnostic(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v10_seasonal",
                    "active": False,
                    "lifecycle_state": "inactive",
                    "quality_score": 0.99,
                    "bot_role": "signal_sub_bot",
                    "model_path": "",
                    "log_file": "",
                }
            ]
        },
    )
    (project_root / "models").mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)
    (project_root / "models" / "brain_refinery_v10_seasonal_20260401.npz").write_text("model", encoding="utf-8")
    _write_json(
        project_root / "logs" / "brain_refinery_v10_seasonal_20260401.json",
        {
            "timestamp": "2026-04-01T00:00:00+00:00",
            "metrics": {
                "acted_count": 12,
                "acted_accuracy": 0.61,
                "accuracy_lift_over_majority": 0.05,
                "positive_rate": 0.52,
            },
        },
    )
    _write_json(
        project_root / "governance" / "health" / "runtime_training_snapshot_latest.json",
        {"row_count": 100, "sequence_count": 10},
    )

    repair = requal_src.apply_repairs(project_root, include_bot_ids=["brain_refinery_v10_seasonal"])
    payload = requal_src.build_payload(project_root)
    registry = json.loads((project_root / "master_bot_registry.json").read_text(encoding="utf-8"))
    diagnostic = json.loads(
        (project_root / "governance" / "training_diagnostics" / "brain_refinery_v10_seasonal_latest.json").read_text(
            encoding="utf-8"
        )
    )

    assert repair["repaired_count"] == 1
    assert repair["unresolved_count"] == 0
    assert registry["sub_bots"][0]["model_path"].endswith("brain_refinery_v10_seasonal_20260401.npz")
    assert registry["sub_bots"][0]["log_file"].endswith("brain_refinery_v10_seasonal_20260401.json")
    assert diagnostic["repaired_from_log"] is True
    assert diagnostic["status"] == "passed"
    assert payload["reactivation_ready_count"] == 1
    assert payload["top_candidates"][0]["actions"] == ["seed_walk_forward_coverage"]


def test_training_requalification_apply_repair_uses_training_diagnostic_artifact_as_source(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v35_dmi_state_machine",
                    "active": False,
                    "lifecycle_state": "inactive",
                    "quality_score": 0.99,
                    "test_accuracy": 0.85,
                    "bot_role": "signal_sub_bot",
                    "model_path": "",
                    "log_file": "",
                }
            ]
        },
    )
    (project_root / "models").mkdir(parents=True, exist_ok=True)
    (project_root / "governance" / "training_diagnostics").mkdir(parents=True, exist_ok=True)
    (project_root / "models" / "brain_refinery_v35_dmi_state_machine_20260404.npz").write_text("model", encoding="utf-8")
    _write_json(
        project_root / "governance" / "training_diagnostics" / "brain_refinery_v35_dmi_state_machine_20260404.json",
        {
            "status": "passed",
            "sample_count": 42,
            "eligible_sequences": 14,
            "sequence_count": 14,
            "positive_rate": 0.51,
        },
    )
    _write_json(
        project_root / "governance" / "health" / "runtime_training_snapshot_latest.json",
        {"row_count": 100, "sequence_count": 10},
    )

    repair = requal_src.apply_repairs(project_root, include_bot_ids=["brain_refinery_v35_dmi_state_machine"])
    payload = requal_src.build_payload(project_root)
    registry = json.loads((project_root / "master_bot_registry.json").read_text(encoding="utf-8"))
    diagnostic = json.loads(
        (project_root / "governance" / "training_diagnostics" / "brain_refinery_v35_dmi_state_machine_latest.json").read_text(
            encoding="utf-8"
        )
    )

    assert repair["unresolved_count"] == 0
    assert registry["sub_bots"][0]["log_file"].endswith("brain_refinery_v35_dmi_state_machine_20260404.json")
    assert diagnostic["repair_source_path"].endswith("brain_refinery_v35_dmi_state_machine_20260404.json")
    assert diagnostic["status"] == "passed"
    assert payload["top_candidates"][0]["actions"] == ["seed_walk_forward_coverage"]


def test_walk_forward_coverage_seed_uses_requalification_lane(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json", {"considered_bots": 0, "coverage_shortfall_bots": 4, "thresholds": {"min_considered_bots": 4, "min_runs_per_bot": 12}})
    _write_json(
        project_root / "governance" / "health" / "training_requalification_latest.json",
        {
            "top_reactivation_ready": [{"bot_id": "brain_refinery_v1", "priority": 88.0, "walk_forward_runs": 4}],
            "top_candidates": [{"bot_id": "brain_refinery_v2", "priority": 80.0, "walk_forward_runs": 1}],
        },
    )

    payload = coverage_src.build_payload(project_root, limit=4)

    assert payload["coverage_shortfall_bots"] == 4
    assert payload["seed_queue"][0]["bot_id"] == "brain_refinery_v1"
    assert payload["seed_queue"][0]["recommended_runs"] == 10
    assert payload["seed_queue"][1]["bot_id"] == "brain_refinery_v2"
    assert payload["seed_queue"][1]["recommended_runs"] == 13


def test_coverage_gap_closer_stages_four_best_non_infrastructure_candidates(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": "brain_refinery_v10_seasonal", "active": False, "deleted_from_rotation": False, "bot_role": "signal_sub_bot"},
                {"bot_id": "brain_refinery_v99_defensive_dividend_concentration", "active": False, "deleted_from_rotation": False, "bot_role": "options_sub_bot"},
                {"bot_id": "brain_refinery_v35_dmi_state_machine", "active": False, "deleted_from_rotation": False, "bot_role": "signal_sub_bot"},
                {"bot_id": "brain_refinery_v13_choppy", "active": False, "deleted_from_rotation": False, "bot_role": "signal_sub_bot"},
                {"bot_id": "brain_refinery_v68_risk_budget_layer", "active": False, "deleted_from_rotation": False, "bot_role": "infrastructure_sub_bot"},
            ]
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "seed_queue": [
                {"bot_id": "brain_refinery_v10_seasonal", "bot_role": "signal_sub_bot", "priority": 100.0, "current_runs": 4, "runs_remaining": 8},
                {"bot_id": "brain_refinery_v99_defensive_dividend_concentration", "bot_role": "options_sub_bot", "priority": 99.0, "current_runs": 2, "runs_remaining": 10},
                {"bot_id": "brain_refinery_v35_dmi_state_machine", "bot_role": "signal_sub_bot", "priority": 98.0, "current_runs": 1, "runs_remaining": 11},
                {"bot_id": "brain_refinery_v13_choppy", "bot_role": "signal_sub_bot", "priority": 97.0, "current_runs": 1, "runs_remaining": 11},
                {"bot_id": "brain_refinery_v68_risk_budget_layer", "bot_role": "infrastructure_sub_bot", "priority": 120.0, "current_runs": 6, "runs_remaining": 6},
            ]
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"coverage_shortfall_bots": 4, "considered_bots": 0, "thresholds": {"min_considered_bots": 4}},
    )

    payload = gap_closer_src.run_gap_closer(
        project_root,
        candidate_limit=8,
        stage_count=4,
        max_cycles=1,
        retrain_timeout_sec=1,
        refresh_timeout_sec=1,
        wait_for_idle_timeout_sec=1,
        poll_sec=1,
        stall_limit=1,
        retrain_profile="coverage_canary",
        apply_stage=True,
        launch=False,
        auto_launch_off_hours=False,
        clear_other_candidates=True,
        out_path=project_root / "governance" / "walk_forward" / "coverage_gap_closer_latest.json",
        queue_out_path=project_root / "governance" / "walk_forward" / "coverage_gap_closer_queue.jsonl",
    )
    registry = json.loads((project_root / "master_bot_registry.json").read_text(encoding="utf-8"))
    staged = {row["bot_id"] for row in registry["sub_bots"] if row.get("coverage_candidate_active")}

    assert payload["active_stage_candidates"][0]["bot_id"] == "brain_refinery_v10_seasonal"
    assert payload["active_stage_candidates"][1]["bot_id"] == "brain_refinery_v35_dmi_state_machine"
    assert payload["active_stage_candidates"][-1]["bot_id"] == "brain_refinery_v99_defensive_dividend_concentration"
    assert payload["recommended_command"][1] == "coverage-gap-closer"
    assert payload["recommended_retrain_command"] == []
    assert "brain_refinery_v68_risk_budget_layer" not in staged
    assert staged == {
        "brain_refinery_v10_seasonal",
        "brain_refinery_v99_defensive_dividend_concentration",
        "brain_refinery_v35_dmi_state_machine",
        "brain_refinery_v13_choppy",
    }
    assert payload["autopilot_contract"]["can_apply_stage"] is True
    assert payload["autopilot_contract"]["stage_candidate_count"] == 4


def test_coverage_gap_closer_keeps_non_coverage_ready_candidate_out_of_active_stage(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "seed_queue": [
                {
                    "bot_id": "brain_refinery_v35_dmi_state_machine",
                    "bot_role": "signal_sub_bot",
                    "priority": 100.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "actions": ["repair_runtime_inputs", "seed_walk_forward_coverage"],
                },
                {
                    "bot_id": "brain_refinery_v4_simple",
                    "bot_role": "signal_sub_bot",
                    "priority": 99.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "actions": ["repair_runtime_inputs", "seed_walk_forward_coverage"],
                },
                {
                    "bot_id": "brain_refinery_v13_choppy",
                    "bot_role": "signal_sub_bot",
                    "priority": 98.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "actions": ["repair_runtime_inputs", "seed_walk_forward_coverage"],
                },
                {
                    "bot_id": "brain_refinery_v100_stock_crypto_overlap_context",
                    "bot_role": "signal_sub_bot",
                    "priority": 97.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "actions": ["rebuild_model_artifact", "repair_runtime_inputs"],
                },
            ]
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"coverage_shortfall_bots": 3, "considered_bots": 1, "thresholds": {"min_considered_bots": 4}},
    )

    pool = gap_closer_src._candidate_pool(project_root, candidate_limit=8, stage_count=4)

    assert [row["bot_id"] for row in pool["active_stage"]] == [
        "brain_refinery_v35_dmi_state_machine",
        "brain_refinery_v4_simple",
        "brain_refinery_v13_choppy",
        "brain_refinery_v100_stock_crypto_overlap_context",
    ]
    assert pool["backup_candidates"] == []


def test_coverage_gap_closer_stages_repair_needed_backups_without_launching(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    repair_actions = [
        "rebuild_model_artifact",
        "recover_training_log",
        "refresh_training_diagnostics",
        "repair_runtime_inputs",
        "generate_walk_forward_runs",
    ]
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "seed_queue": [
                {
                    "bot_id": "brain_refinery_v35_dmi_state_machine",
                    "bot_role": "signal_sub_bot",
                    "priority": 100.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "needs_runtime_input_repair": True,
                    "actions": repair_actions,
                },
                {
                    "bot_id": "brain_refinery_v4_simple",
                    "bot_role": "signal_sub_bot",
                    "priority": 99.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "needs_runtime_input_repair": True,
                    "actions": repair_actions,
                },
                {
                    "bot_id": "brain_refinery_v13_choppy",
                    "bot_role": "signal_sub_bot",
                    "priority": 98.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "needs_runtime_input_repair": True,
                    "actions": repair_actions,
                },
            ]
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"coverage_shortfall_bots": 3, "considered_bots": 1, "thresholds": {"min_considered_bots": 4}},
    )
    _write_json(
        project_root / "governance" / "health" / "training_runtime_control_latest.json",
        {"overall_status": "ready", "snapshot_ready": True, "coverage_repair_ready": True},
    )
    _write_json(project_root / "governance" / "health" / "resource_guard_latest.json", {"swap_used_gb": 0.1})
    _write_json(
        project_root / "governance" / "health" / "live_runtime_separation_control_latest.json",
        {
            "overall_status": "degraded",
            "release_contract": {"live_lane_should_be_read_only": True},
            "clearance_plan": {"cold_lane_refresh": {"overall_status": "ready"}},
        },
    )

    payload = gap_closer_src._build_payload(
        project_root,
        candidate_limit=8,
        stage_count=3,
        retrain_profile="coverage_canary",
    )

    assert payload["staged_candidate_count"] == 3
    assert {row["coverage_stage_kind"] for row in payload["active_stage_candidates"]} == {
        "runtime_input_repair_required"
    }
    assert payload["autopilot_contract"]["can_apply_stage"] is True
    assert payload["autopilot_contract"]["can_launch_now"] is False
    assert payload["autopilot_contract"]["repair_required_count"] == 3
    assert payload["autopilot_contract"]["launch_state"] == "runtime_input_repair_required"
    assert "runtime_input_repair_required" in payload["autopilot_contract"]["blocking_reasons"]


def test_coverage_gap_closer_blocks_launch_for_preflight_repairs_even_without_runtime_input_gap(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    preflight_actions = [
        "rebuild_model_artifact",
        "recover_training_log",
        "refresh_training_diagnostics",
        "generate_walk_forward_runs",
    ]
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "seed_queue": [
                {
                    "bot_id": "brain_refinery_v35_dmi_state_machine",
                    "bot_role": "signal_sub_bot",
                    "priority": 100.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "needs_runtime_input_repair": False,
                    "actions": preflight_actions,
                }
            ]
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"coverage_shortfall_bots": 1, "considered_bots": 3, "thresholds": {"min_considered_bots": 4}},
    )
    _write_json(
        project_root / "governance" / "health" / "training_runtime_control_latest.json",
        {"overall_status": "ready", "snapshot_ready": True, "coverage_repair_ready": True},
    )
    _write_json(project_root / "governance" / "health" / "resource_guard_latest.json", {"swap_used_gb": 0.1})
    _write_json(
        project_root / "governance" / "health" / "live_runtime_separation_control_latest.json",
        {"overall_status": "ready", "release_contract": {"live_lane_should_be_read_only": False}},
    )

    payload = gap_closer_src._build_payload(
        project_root,
        candidate_limit=4,
        stage_count=1,
        retrain_profile="coverage_canary",
    )

    assert payload["active_stage_candidates"][0]["coverage_stage_kind"] == "coverage_preflight_repair_required"
    assert payload["autopilot_contract"]["repair_required_count"] == 0
    assert payload["autopilot_contract"]["preflight_repair_required_count"] == 1
    assert payload["autopilot_contract"]["can_launch_now"] is False
    assert payload["autopilot_contract"]["launch_state"] == "coverage_preflight_repair_required"
    assert "coverage_preflight_repair_required" in payload["autopilot_contract"]["blocking_reasons"]


def test_coverage_gap_closer_treats_rebuild_and_targeted_retrain_as_launch_work(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "seed_queue": [
                {
                    "bot_id": "brain_refinery_v35_dmi_state_machine",
                    "bot_role": "signal_sub_bot",
                    "priority": 100.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "needs_runtime_input_repair": False,
                    "actions": [
                        "rebuild_model_artifact",
                        "targeted_retrain",
                        "generate_walk_forward_runs",
                    ],
                }
            ]
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"coverage_shortfall_bots": 1, "considered_bots": 3, "thresholds": {"min_considered_bots": 4}},
    )
    _write_json(
        project_root / "governance" / "health" / "training_runtime_control_latest.json",
        {"overall_status": "ready", "snapshot_ready": True, "coverage_repair_ready": True},
    )
    _write_json(project_root / "governance" / "health" / "resource_guard_latest.json", {"swap_used_gb": 0.1})
    _write_json(
        project_root / "governance" / "health" / "live_runtime_separation_control_latest.json",
        {"overall_status": "ready", "release_contract": {"live_lane_should_be_read_only": False}},
    )

    payload = gap_closer_src._build_payload(
        project_root,
        candidate_limit=4,
        stage_count=1,
        retrain_profile="coverage_canary",
    )

    assert payload["autopilot_contract"]["preflight_repair_required_count"] == 0
    assert payload["autopilot_contract"]["can_launch_now"] is True
    assert payload["autopilot_contract"]["launch_state"] == "ready_to_launch"


def test_coverage_gap_closer_prefers_sample_viable_candidates_over_recent_failed_bot(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "seed_queue": [
                {
                    "bot_id": "brain_refinery_v35_dmi_state_machine",
                    "bot_role": "signal_sub_bot",
                    "priority": 100.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "actions": ["repair_runtime_inputs", "seed_walk_forward_coverage"],
                },
                {
                    "bot_id": "brain_refinery_v4_simple",
                    "bot_role": "signal_sub_bot",
                    "priority": 99.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "actions": ["repair_runtime_inputs", "seed_walk_forward_coverage"],
                },
                {
                    "bot_id": "brain_refinery_v13_choppy",
                    "bot_role": "signal_sub_bot",
                    "priority": 98.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "actions": ["repair_runtime_inputs", "seed_walk_forward_coverage"],
                },
                {
                    "bot_id": "brain_refinery_v100_stock_crypto_overlap_context",
                    "bot_role": "signal_sub_bot",
                    "priority": 97.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "actions": ["repair_runtime_inputs", "seed_walk_forward_coverage"],
                },
            ]
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"coverage_shortfall_bots": 3, "considered_bots": 1, "thresholds": {"min_considered_bots": 4}},
    )
    _write_json(
        project_root / "governance" / "training_diagnostics" / "brain_refinery_v35_dmi_state_machine_latest.json",
        {"status": "failed", "sample_count": 0},
    )
    _write_json(
        project_root / "governance" / "training_diagnostics" / "brain_refinery_v4_simple_latest.json",
        {"status": "deferred_sample_starved", "sample_count": 127},
    )
    _write_json(
        project_root / "governance" / "training_diagnostics" / "brain_refinery_v13_choppy_latest.json",
        {"status": "deferred_sample_starved", "sample_count": 37},
    )
    _write_json(
        project_root / "governance" / "training_diagnostics" / "brain_refinery_v100_stock_crypto_overlap_context_latest.json",
        {"status": "deferred_sample_starved", "sample_count": 55},
    )

    pool = gap_closer_src._candidate_pool(project_root, candidate_limit=8, stage_count=3)

    assert [row["bot_id"] for row in pool["active_stage"]] == [
        "brain_refinery_v4_simple",
        "brain_refinery_v13_choppy",
        "brain_refinery_v100_stock_crypto_overlap_context",
    ]
    assert pool["backup_candidates"][0]["bot_id"] == "brain_refinery_v35_dmi_state_machine"


def test_coverage_gap_closer_allows_rebuild_ready_candidate_into_active_stage(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "seed_queue": [
                {
                    "bot_id": "brain_refinery_v4_simple",
                    "bot_role": "signal_sub_bot",
                    "priority": 99.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "actions": ["repair_runtime_inputs", "seed_walk_forward_coverage"],
                },
                {
                    "bot_id": "brain_refinery_v13_choppy",
                    "bot_role": "signal_sub_bot",
                    "priority": 98.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "actions": ["repair_runtime_inputs", "seed_walk_forward_coverage"],
                },
                {
                    "bot_id": "brain_refinery_v100_stock_crypto_overlap_context",
                    "bot_role": "signal_sub_bot",
                    "priority": 97.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "actions": ["rebuild_model_artifact", "repair_runtime_inputs", "generate_walk_forward_runs"],
                },
                {
                    "bot_id": "brain_refinery_v1",
                    "bot_role": "signal_sub_bot",
                    "priority": 96.0,
                    "current_runs": 0,
                    "runs_remaining": 12,
                    "actions": ["rebuild_model_artifact", "recover_training_log", "repair_runtime_inputs"],
                },
            ]
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"coverage_shortfall_bots": 3, "considered_bots": 1, "thresholds": {"min_considered_bots": 4}},
    )

    pool = gap_closer_src._candidate_pool(project_root, candidate_limit=8, stage_count=3)

    assert [row["bot_id"] for row in pool["active_stage"]] == [
        "brain_refinery_v4_simple",
        "brain_refinery_v13_choppy",
        "brain_refinery_v100_stock_crypto_overlap_context",
    ]
    assert pool["backup_candidates"][0]["bot_id"] == "brain_refinery_v1"


def test_coverage_gap_closer_surfaces_launch_ready_autopilot_when_runtime_is_clear(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "seed_queue": [
                {
                    "bot_id": "brain_refinery_v10_seasonal",
                    "bot_role": "signal_sub_bot",
                    "priority": 100.0,
                    "current_runs": 4,
                    "runs_remaining": 8,
                    "needs_runtime_input_repair": False,
                }
            ]
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"coverage_shortfall_bots": 1, "considered_bots": 3, "thresholds": {"min_considered_bots": 4}},
    )
    _write_json(
        project_root / "governance" / "health" / "training_runtime_control_latest.json",
        {"overall_status": "ready", "snapshot_ready": True},
    )
    _write_json(project_root / "governance" / "health" / "resource_guard_latest.json", {"swap_used_gb": 1.0})
    _write_json(
        project_root / "governance" / "health" / "live_runtime_separation_control_latest.json",
        {"overall_status": "ready", "release_contract": {"live_lane_should_be_read_only": False}},
    )
    monkeypatch.setattr(gap_closer_src, "_refresh_artifacts", lambda *args, **kwargs: [])

    payload = gap_closer_src.run_gap_closer(
        project_root,
        candidate_limit=4,
        stage_count=1,
        max_cycles=1,
        retrain_timeout_sec=1,
        refresh_timeout_sec=1,
        wait_for_idle_timeout_sec=1,
        poll_sec=1,
        stall_limit=1,
        retrain_profile="coverage_canary",
        apply_stage=False,
        launch=False,
        auto_launch_off_hours=False,
        clear_other_candidates=True,
        out_path=project_root / "governance" / "walk_forward" / "coverage_gap_closer_latest.json",
        queue_out_path=project_root / "governance" / "walk_forward" / "coverage_gap_closer_queue.jsonl",
    )

    assert payload["autopilot_contract"]["overall_status"] == "ready"
    assert payload["autopilot_contract"]["launch_state"] == "ready_to_launch"
    assert payload["autopilot_contract"]["can_launch_now"] is True
    assert payload["recommended_command"][1] == "retrain-force-targeted"


def test_coverage_gap_closer_respects_training_launch_contract_budget_block(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "seed_queue": [
                {
                    "bot_id": "brain_refinery_v10_seasonal",
                    "bot_role": "signal_sub_bot",
                    "priority": 100.0,
                    "current_runs": 4,
                    "runs_remaining": 8,
                    "needs_runtime_input_repair": False,
                }
            ]
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"coverage_shortfall_bots": 1, "considered_bots": 3, "thresholds": {"min_considered_bots": 4}},
    )
    _write_json(
        project_root / "governance" / "health" / "training_runtime_control_latest.json",
        {
            "overall_status": "constrained",
            "snapshot_ready": True,
            "coverage_repair_ready": True,
            "training_quality": {"overall_status": "needs_attention", "training_quality_score": 98.0},
            "training_launch_contract": {
                "launch_allowed": False,
                "prep_allowed": True,
                "launch_blockers": ["autonomic_training_budget_closed"],
                "recommended_batch_size": 0,
            },
        },
    )
    _write_json(project_root / "governance" / "health" / "resource_guard_latest.json", {"swap_used_gb": 1.0})
    _write_json(
        project_root / "governance" / "health" / "live_runtime_separation_control_latest.json",
        {"overall_status": "ready", "release_contract": {"live_lane_should_be_read_only": False}},
    )
    monkeypatch.setattr(gap_closer_src, "_refresh_artifacts", lambda *args, **kwargs: [])

    def _fail_run_json(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("coverage gap closer must not launch when training budget is closed")

    monkeypatch.setattr(gap_closer_src, "_run_json", _fail_run_json)

    payload = gap_closer_src.run_gap_closer(
        project_root,
        candidate_limit=4,
        stage_count=1,
        max_cycles=1,
        retrain_timeout_sec=1,
        refresh_timeout_sec=1,
        wait_for_idle_timeout_sec=1,
        poll_sec=1,
        stall_limit=1,
        retrain_profile="coverage_canary",
        apply_stage=False,
        launch=True,
        auto_launch_off_hours=False,
        clear_other_candidates=True,
        out_path=project_root / "governance" / "walk_forward" / "coverage_gap_closer_latest.json",
        queue_out_path=project_root / "governance" / "walk_forward" / "coverage_gap_closer_queue.jsonl",
    )

    assert payload["autopilot_contract"]["launch_state"] == "stage_only_training_blocked"
    assert payload["autopilot_contract"]["can_launch_now"] is False
    assert "training_launch_contract_blocked" in payload["autopilot_contract"]["blocking_reasons"]
    assert "autonomic_training_budget_closed" in payload["autopilot_contract"]["blocking_reasons"]
    assert payload["recommended_command"][1] == "coverage-gap-closer"
    assert payload["recommended_retrain_command"] == []
    assert payload["cycle_records"] == []


def test_coverage_gap_closer_opens_off_hours_auto_launch_window_when_live_lane_is_read_only(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "seed_queue": [
                {
                    "bot_id": "brain_refinery_v10_seasonal",
                    "bot_role": "signal_sub_bot",
                    "priority": 100.0,
                    "current_runs": 4,
                    "runs_remaining": 8,
                    "needs_runtime_input_repair": False,
                }
            ]
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"coverage_shortfall_bots": 1, "considered_bots": 3, "thresholds": {"min_considered_bots": 4}},
    )
    _write_json(
        project_root / "governance" / "health" / "training_runtime_control_latest.json",
        {"overall_status": "ready", "snapshot_ready": True},
    )
    _write_json(project_root / "governance" / "health" / "resource_guard_latest.json", {"swap_used_gb": 1.0})
    _write_json(
        project_root / "governance" / "health" / "live_runtime_separation_control_latest.json",
        {
            "overall_status": "blocked",
            "release_contract": {"live_lane_should_be_read_only": True},
            "clearance_plan": {"cold_lane_refresh": {"overall_status": "ready"}},
        },
    )
    monkeypatch.setattr(gap_closer_src, "_refresh_artifacts", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        gap_closer_src,
        "_off_hours_window",
        lambda *args, **kwargs: {"active": True, "label": "off_hours", "timezone": "America/New_York"},
    )

    payload = gap_closer_src.run_gap_closer(
        project_root,
        candidate_limit=4,
        stage_count=1,
        max_cycles=1,
        retrain_timeout_sec=1,
        refresh_timeout_sec=1,
        wait_for_idle_timeout_sec=1,
        poll_sec=1,
        stall_limit=1,
        retrain_profile="coverage_canary",
        apply_stage=False,
        launch=False,
        auto_launch_off_hours=False,
        clear_other_candidates=True,
        out_path=project_root / "governance" / "walk_forward" / "coverage_gap_closer_latest.json",
        queue_out_path=project_root / "governance" / "walk_forward" / "coverage_gap_closer_queue.jsonl",
    )

    assert payload["autopilot_contract"]["launch_state"] == "auto_launch_off_hours_ready"
    assert payload["autopilot_contract"]["can_auto_launch_off_hours"] is True
    assert payload["autopilot_contract"]["can_launch_now"] is True


def test_coverage_gap_closer_arms_next_off_hours_launch_when_market_is_still_open(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "seed_queue": [
                {
                    "bot_id": "brain_refinery_v10_seasonal",
                    "bot_role": "signal_sub_bot",
                    "priority": 100.0,
                    "current_runs": 4,
                    "runs_remaining": 8,
                    "needs_runtime_input_repair": False,
                }
            ]
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"coverage_shortfall_bots": 1, "considered_bots": 3, "thresholds": {"min_considered_bots": 4}},
    )
    _write_json(
        project_root / "governance" / "health" / "training_runtime_control_latest.json",
        {"overall_status": "ready", "snapshot_ready": True},
    )
    _write_json(project_root / "governance" / "health" / "resource_guard_latest.json", {"swap_used_gb": 1.0})
    _write_json(
        project_root / "governance" / "health" / "live_runtime_separation_control_latest.json",
        {
            "overall_status": "blocked",
            "release_contract": {"live_lane_should_be_read_only": True},
            "clearance_plan": {"cold_lane_refresh": {"overall_status": "ready"}},
        },
    )
    monkeypatch.setattr(gap_closer_src, "_refresh_artifacts", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        gap_closer_src,
        "_off_hours_window",
        lambda *args, **kwargs: {"active": False, "label": "market_hours", "timezone": "America/New_York", "window_start_local": "16:15", "window_end_local": "09:20"},
    )

    payload = gap_closer_src.run_gap_closer(
        project_root,
        candidate_limit=4,
        stage_count=1,
        max_cycles=1,
        retrain_timeout_sec=1,
        refresh_timeout_sec=1,
        wait_for_idle_timeout_sec=1,
        poll_sec=1,
        stall_limit=1,
        retrain_profile="coverage_canary",
        apply_stage=False,
        launch=False,
        auto_launch_off_hours=False,
        clear_other_candidates=True,
        out_path=project_root / "governance" / "walk_forward" / "coverage_gap_closer_latest.json",
        queue_out_path=project_root / "governance" / "walk_forward" / "coverage_gap_closer_queue.jsonl",
    )

    assert payload["autopilot_contract"]["launch_state"] == "armed_for_off_hours_auto_launch"
    assert payload["autopilot_contract"]["can_auto_launch_off_hours"] is False
    assert payload["autopilot_contract"]["auto_launch_pending"] is True
    assert payload["autopilot_contract"]["launch_contract"]["window_active"] is False


def test_coverage_gap_closer_blocks_requested_launch_when_training_quality_is_blocked(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "seed_queue": [
                {
                    "bot_id": "brain_refinery_v10_seasonal",
                    "bot_role": "signal_sub_bot",
                    "priority": 100.0,
                    "current_runs": 4,
                    "runs_remaining": 8,
                    "needs_runtime_input_repair": False,
                }
            ]
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"coverage_shortfall_bots": 1, "considered_bots": 3, "thresholds": {"min_considered_bots": 4}},
    )
    _write_json(
        project_root / "governance" / "health" / "training_runtime_control_latest.json",
        {
            "overall_status": "blocked",
            "snapshot_ready": True,
            "coverage_repair_ready": True,
            "training_quality": {"overall_status": "blocked"},
        },
    )
    _write_json(project_root / "governance" / "health" / "resource_guard_latest.json", {"swap_used_gb": 1.0})
    _write_json(
        project_root / "governance" / "health" / "live_runtime_separation_control_latest.json",
        {"overall_status": "ready", "release_contract": {"live_lane_should_be_read_only": False}},
    )
    monkeypatch.setattr(gap_closer_src, "_refresh_artifacts", lambda *args, **kwargs: [])

    def _fail_run_json(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("retrain command should not launch while training quality is blocked")

    monkeypatch.setattr(gap_closer_src, "_run_json", _fail_run_json)

    payload = gap_closer_src.run_gap_closer(
        project_root,
        candidate_limit=4,
        stage_count=1,
        max_cycles=1,
        retrain_timeout_sec=1,
        refresh_timeout_sec=1,
        wait_for_idle_timeout_sec=1,
        poll_sec=1,
        stall_limit=1,
        retrain_profile="coverage_canary",
        apply_stage=False,
        launch=True,
        auto_launch_off_hours=False,
        clear_other_candidates=True,
        out_path=project_root / "governance" / "walk_forward" / "coverage_gap_closer_latest.json",
        queue_out_path=project_root / "governance" / "walk_forward" / "coverage_gap_closer_queue.jsonl",
    )

    assert payload["autopilot_contract"]["can_launch_now"] is False
    assert payload["autopilot_contract"]["launch_state"] == "stage_only_training_blocked"
    assert "training_runtime_blocked" in payload["launch_decision"]["launch_blocked_reasons"]
    assert "training_quality_blocked" in payload["launch_decision"]["launch_blocked_reasons"]
    assert payload["cycle_records"] == []
    assert payload["recommended_command"][1] == "coverage-gap-closer"
    assert payload["recommended_retrain_command"] == []


def test_coverage_gap_closer_rotates_timeout_prone_candidate_immediately(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"coverage_shortfall_bots": 3, "considered_bots": 1, "thresholds": {"min_considered_bots": 4, "min_runs_per_bot": 12}},
    )
    _write_json(
        project_root / "governance" / "health" / "training_runtime_control_latest.json",
        {"overall_status": "ready", "snapshot_ready": True, "coverage_repair_ready": True},
    )
    _write_json(project_root / "governance" / "health" / "resource_guard_latest.json", {"swap_used_gb": 0.1})
    _write_json(
        project_root / "governance" / "health" / "live_runtime_separation_control_latest.json",
        {"overall_status": "ready", "release_contract": {"live_lane_should_be_read_only": False}},
    )
    staged_rows = [
        {
            "bot_id": "brain_refinery_v99_defensive_dividend_concentration",
            "bot_role": "options_sub_bot",
            "queue_bucket": "options",
            "priority": 100.0,
            "current_runs": 0,
            "runs_remaining": 12,
        }
    ]
    backup_rows = [
        {
            "bot_id": "brain_refinery_v35_dmi_state_machine",
            "bot_role": "signal_sub_bot",
            "queue_bucket": "signal",
            "priority": 99.0,
            "current_runs": 0,
            "runs_remaining": 12,
        }
    ]

    monkeypatch.setattr(
        gap_closer_src,
        "_candidate_pool",
        lambda *args, **kwargs: {
            "coverage_seed": {},
            "promotion_readiness": {"coverage_shortfall_bots": 3, "considered_bots": 1, "thresholds": {"min_considered_bots": 4, "min_runs_per_bot": 12}},
            "min_considered_bots": 4,
            "active_stage": [dict(row) for row in staged_rows],
            "backup_candidates": [dict(row) for row in backup_rows],
        },
    )
    monkeypatch.setattr(gap_closer_src, "_refresh_artifacts", lambda *args, **kwargs: [])
    monkeypatch.setattr(gap_closer_src, "_wait_for_retrain_idle", lambda *args, **kwargs: (True, []))
    monkeypatch.setattr(gap_closer_src, "_load_walk_forward_runs", lambda *_args, **_kwargs: {})

    run_calls: list[dict[str, object]] = []

    def _fake_run_json(cmd: list[str], *, cwd: Path, timeout_sec: int, env: dict[str, str] | None = None) -> dict:
        run_calls.append({"cmd": list(cmd), "env": dict(env or {})})
        return {
            "cmd": list(cmd),
            "rc": 1,
            "timed_out": False,
            "stdout_tail": "FAIL: /tmp/core/brain_refinery_v99_defensive_dividend_concentration.py (exit=124)",
            "stderr_tail": "[Timeout] command exceeded 600s",
            "payload": {
                "failure_details": [
                    {
                        "bot_id": "brain_refinery_v99_defensive_dividend_concentration",
                        "rc": 124,
                        "reason": "[Timeout] command exceeded 600s",
                    }
                ]
            },
        }

    monkeypatch.setattr(gap_closer_src, "_run_json", _fake_run_json)

    payload = gap_closer_src.run_gap_closer(
        project_root,
        candidate_limit=4,
        stage_count=1,
        max_cycles=1,
        retrain_timeout_sec=30,
        refresh_timeout_sec=30,
        wait_for_idle_timeout_sec=30,
        poll_sec=1,
        stall_limit=2,
        retrain_profile="coverage_canary",
        apply_stage=False,
        launch=True,
        auto_launch_off_hours=False,
        clear_other_candidates=True,
        out_path=project_root / "governance" / "walk_forward" / "coverage_gap_closer_latest.json",
        queue_out_path=project_root / "governance" / "walk_forward" / "coverage_gap_closer_queue.jsonl",
    )

    assert payload["cycle_records"][0]["timeout_like_bot_ids"] == ["brain_refinery_v99_defensive_dividend_concentration"]
    assert payload["cycle_records"][0]["swapped_out_bot_ids"] == ["brain_refinery_v99_defensive_dividend_concentration"]
    assert payload["active_stage_candidates"][0]["bot_id"] == "brain_refinery_v35_dmi_state_machine"
    assert run_calls[0]["env"]["RETRAIN_TRIGGER_SOURCE"] == "coverage_gap_closer"
    assert "--retrain-profile" in run_calls[0]["cmd"]
    assert "coverage_canary" in run_calls[0]["cmd"]


def test_retrain_lane_scheduler_isolates_infrastructure_bots_in_their_own_lane(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": "brain_refinery_v4_simple", "active": True, "bot_role": "signal_sub_bot"},
                {"bot_id": "brain_refinery_v13_choppy", "active": True, "lifecycle_state": "probation", "bot_role": "signal_sub_bot"},
                {"bot_id": "brain_refinery_v99_new_lane", "active": True, "bot_role": "signal_sub_bot"},
                {"bot_id": "brain_refinery_v56_meta_ranker", "active": True, "lifecycle_state": "probation", "bot_role": "infrastructure_sub_bot"},
            ]
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "walk_forward_latest.json",
        {
            "bots": {
                "brain_refinery_v4_simple": {"runs": 40},
                "brain_refinery_v13_choppy": {"runs": 12},
                "brain_refinery_v99_new_lane": {"runs": 3},
                "brain_refinery_v56_meta_ranker": {"runs": 8},
            }
        },
    )
    _write_json(
        project_root / "governance" / "health" / "new_bot_admission_guard_latest.json",
        {"candidates": [{"bot_id": "brain_refinery_v99_new_lane"}]},
    )
    _write_json(
        project_root / "governance" / "health" / "champion_challenger_probation_latest.json",
        {"monitored_candidates": [{"bot_id": "brain_refinery_v13_choppy"}, {"bot_id": "brain_refinery_v56_meta_ranker"}]},
    )

    payload = lane_scheduler_src.build_payload(
        registry=json.loads((project_root / "master_bot_registry.json").read_text(encoding="utf-8")),
        walk_forward=json.loads((project_root / "governance" / "walk_forward" / "walk_forward_latest.json").read_text(encoding="utf-8")),
        new_bot_admission_guard=json.loads((project_root / "governance" / "health" / "new_bot_admission_guard_latest.json").read_text(encoding="utf-8")),
        probation_guard=json.loads((project_root / "governance" / "health" / "champion_challenger_probation_latest.json").read_text(encoding="utf-8")),
        target_bot_ids=[],
        max_targets=4,
        new_bot_max_runs=24,
    )

    assert payload["lanes"]["infrastructure"]["candidate_count"] == 1
    assert payload["lanes"]["infrastructure"]["bot_ids"] == ["brain_refinery_v56_meta_ranker"]
    assert "brain_refinery_v56_meta_ranker" in payload["selected_bot_ids"]


def test_calibration_abstention_control_generates_tightening_recommendation(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "health" / "training_label_audit_latest.json",
        {
            "active_overacting": [
                {
                    "bot_id": "brain_refinery_v43_intraday_ultrafast_proxy",
                    "accuracy_lift_over_majority": -0.03,
                    "acceptance_rate": 0.18,
                    "acted_accuracy": 0.51,
                }
            ]
        },
    )

    payload = calibration_src.build_payload(project_root)

    assert payload["overall_status"] == "needs_tuning"
    assert payload["calibration_confidence_score"] < 100.0
    assert payload["a_plus_contract"]["override_ready"] is False
    assert payload["recommendations"][0]["mode"] == "tighten"
    assert payload["recommendations"][0]["target_acceptance_rate"] < 0.18


def test_calibration_abstention_control_apply_writes_bot_and_family_overrides(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "health" / "training_label_audit_latest.json",
        {
            "active_overacting": [
                {
                    "bot_id": "brain_refinery_v43_intraday_ultrafast_proxy",
                    "accuracy_lift_over_majority": -0.03,
                    "acceptance_rate": 0.18,
                    "acted_accuracy": 0.51,
                }
            ]
        },
    )
    _write_json(
        project_root / "governance" / "health" / "training_quality_control_latest.json",
        {
            "targeted_actions": {
                "weak_sleeves": [
                    {
                        "profile": "dividend",
                        "ending_net_pnl_total": -138.87,
                        "win_rate": 0.0,
                    }
                ]
            }
        },
    )

    payload = calibration_src.build_payload(project_root)
    override_payload = calibration_src.build_override_payload(payload)

    assert payload["family_recommendations"][0]["family"] == "dividend"
    assert payload["override_state"]["bot_override_count"] == 0
    assert "brain_refinery_v43_intraday_ultrafast_proxy" in override_payload["bot_overrides"]
    assert "dividend" in override_payload["family_overrides"]
    assert override_payload["bot_overrides"]["brain_refinery_v43_intraday_ultrafast_proxy"]["acted_prob_threshold_uplift"] > 0.0
    assert override_payload["family_overrides"]["dividend"]["acted_prob_threshold_uplift"] > 0.0


def test_calibration_abstention_control_preserves_existing_precision_overrides(tmp_path: Path) -> None:
    payload = {
        "recommendations": [
            {
                "bot_id": "brain_refinery_v42_tick_to_swing_alignment",
                "family": "swing",
                "mode": "tighten",
                "target_acceptance_rate": 0.18,
                "confidence_threshold_uplift": 0.09,
                "recommended_abstention_budget": 0.82,
            }
        ],
        "family_recommendations": [],
    }
    existing = {
        "bot_overrides": {
            "brain_refinery_v47_swing_1w_3w": {
                "mode": "tighten",
                "family": "swing",
                "acted_prob_threshold_uplift": 0.09,
                "target_acceptance_rate": 0.18,
                "recommended_abstention_budget": 0.82,
            }
        },
        "family_overrides": {},
    }

    override_payload = calibration_src.build_override_payload(payload, existing)

    assert "brain_refinery_v42_tick_to_swing_alignment" in override_payload["bot_overrides"]
    assert "brain_refinery_v47_swing_1w_3w" in override_payload["bot_overrides"]


def test_calibration_abstention_control_retires_unsafe_loosen_override(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v999_sparse_underactor"
    _write_json(
        project_root / "governance" / "health" / "training_label_audit_latest.json",
        {
            "active_underacting": [
                {
                    "bot_id": bot_id,
                    "acted_coverage": 0.0,
                    "acceptance_rate": 0.0,
                    "abstention_evidence_sufficient": False,
                }
            ]
        },
    )
    existing = {
        "bot_overrides": {
            bot_id: {
                "mode": "loosen",
                "acted_prob_threshold_uplift": -0.05,
            }
        },
        "family_overrides": {},
    }

    payload = calibration_src.build_payload(project_root)
    overrides = calibration_src.build_override_payload(payload, existing)

    assert payload["recommendations"][0]["mode"] == "collect_evidence"
    assert payload["recommendations"][0]["direct_loosen_allowed"] is False
    assert bot_id not in overrides["bot_overrides"]
    assert overrides["retired_overrides"][0]["reason"] == "unsafe_direct_loosen_retired"


def test_calibration_abstention_control_sanitizes_nested_regime_overrides() -> None:
    payload = {
        "candidate_binding": {"candidate_id": "candidate-1", "generation": 1},
        "recommendations": [],
        "family_recommendations": [],
    }
    existing = {
        "bot_overrides": {},
        "family_overrides": {},
        "regime_overrides": {
            "dividend": {
                "risk_on": {
                    "mode": "tighten",
                    "acted_prob_threshold_uplift": 0.03,
                },
                "risk_off": {
                    "mode": "loosen",
                    "acted_prob_threshold_uplift": -0.04,
                },
            }
        },
    }

    overrides = calibration_src.build_override_payload(payload, existing)

    assert set(overrides["regime_overrides"]["dividend"]) == {"risk_on"}
    retained = overrides["regime_overrides"]["dividend"]["risk_on"]
    assert retained["mode"] == "tighten"
    assert retained["valid_candidate_id"] == "candidate-1"
    assert {
        "scope": "regime",
        "key": "dividend:risk_off",
        "reason": "unsafe_direct_loosen_retired",
    } in overrides["retired_overrides"]
