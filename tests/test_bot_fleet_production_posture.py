import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops import bot_fleet_production_posture as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _iso_minutes_ago(minutes: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


def _registry_rows(*, live_authority: bool = False, contract_gap: bool = False) -> list[dict]:
    rows = [
        {
            "bot_id": "bot_a",
            "bot_role": "signal_sub_bot",
            "lifecycle_state": "paper_live_data",
            "active": True,
            "data_collection_active": True,
            "paper_live_data_enabled": True,
            "label_contract": {"label": "profitability"},
            "target_functions": ["collect_observations", "paper_trade_candidate"],
            "direct_execution_allowed": live_authority,
        },
        {
            "bot_id": "bot_b",
            "bot_role": "infrastructure_sub_bot",
            "lifecycle_state": "data_collection_only",
            "active": True,
            "data_collection_active": True,
            "paper_live_data_enabled": False,
            "label_contract": {"label": "runtime_quality"},
            "target_functions": ["collect_observations", "diagnostics"],
        },
        {
            "bot_id": "bot_old",
            "bot_role": "legacy",
            "lifecycle_state": "deleted",
            "active": False,
            "data_collection_active": False,
            "deleted_from_rotation": True,
        },
    ]
    if contract_gap:
        rows[1].pop("label_contract")
        rows[1]["target_functions"] = []
    return rows


def _seed_project(
    project_root: Path,
    *,
    live_authority: bool = False,
    contract_gap: bool = False,
    include_overfit: bool = True,
    guarded_teacher_blocks: bool = False,
) -> None:
    health = project_root / "governance" / "health"
    _write_json(project_root / "master_bot_registry.json", {"sub_bots": _registry_rows(live_authority=live_authority, contract_gap=contract_gap)})
    _write_json(
        health / "paper_live_data_standard_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "counts_after": {
                "non_deleted_bots": 2,
                "data_collection_active_bots": 2,
                "paper_live_data_enabled_bots": 1,
                "collection_until_standard_bots": 1,
                "direct_execution_allowed_bots": 1 if live_authority else 0,
                "live_trading_enabled_bots": 1 if live_authority else 0,
            },
            "safety_contract": {
                "paper_trade_lock": "1",
                "market_data_only": "0" if live_authority else "1",
                "allow_order_execution": "1" if live_authority else "0",
                "live_execution_allowed": bool(live_authority),
                "paper_mirror_all_active_sub_bots": "1",
            },
        },
    )
    _write_json(
        health / "data_collection_observation_rollup_latest.json",
        {
            "overall_status": "ready",
            "collector_count": 2,
            "effective_bots_with_observations": 2,
            "bots_with_observations": 2,
            "unmanaged_zero_observation_count": 0,
            "zero_observation_count": 0,
            "total_observations": 1250,
            "training_ready_count": 2,
            "collection_coverage_score": 100.0,
            "data_quality_score": 100.0,
        },
    )
    _write_json(
        health / "health_fast_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "operational_readiness": {
                "guarded_paper": {"status": "ready", "ok": True, "blockers": [], "paper_ramp_stage": "armed"}
            },
            "process_watchdog": {
                "all_sleeves_effective_runtime": {
                    "status": "ready",
                    "ok": True,
                    "child_process_count": 4,
                    "child_fanout_ok": True,
                    "heartbeat_ok": True,
                }
            },
        },
    )
    _write_json(
        health / "sleeve_ingestion_production_control_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "production_grade_contract": {"grade": "A+", "score": 100.0, "missing": []},
            "ingestion_mode_contract": {
                "mode": "production_owned_manifest_first",
                "paper_soak_allowed": True,
                "live_money_blocked": True,
            },
        },
    )
    _write_json(
        health / "bot_quality_autopilot_latest.json",
        {
            "overall_status": "needs_work",
            "quality_blockers": {
                "planned_queue_count": 1,
                "repair_runtime_input_bot_ids": [],
                "students_without_teachers": 0,
                "coverage_shortfall_bots": 0,
                "refresh_diagnostics_bot_ids": ["bot_b"],
            },
            "attempts": [{"cmd": ["./scripts/ops/opsctl.sh", "training-quality"]}],
        },
    )
    _write_json(
        health / "bot_intelligence_mesh_latest.json",
        {
            "overall_status": "ready",
            "communication_readiness_score": 100.0,
            "quality_readiness_score": 85.3,
            "quality_target_status": "needs_work",
            "bot_count": 2,
            "active_bot_count": 2,
            "missing_tiers": [],
            "hierarchy_edge_summary": {
                "edge_count_total": 4,
                "active_sub_or_infra_route_ratio": 1.0,
                "active_master_route_ratio": 1.0,
            },
            "a_plus_target_contract": {"blocker_count": 1, "summary": {"training_ready_gap": 1}},
        },
    )
    _write_json(
        health / "training_quality_control_latest.json",
        {
            "overall_status": "ready",
            "training_quality_score": 100.0,
            "targeted_actions": {
                "selected_targeted_retrain_bot_ids": ["bot_b"],
                "precompute_target_bot_ids": ["bot_b"],
                "weak_sleeves": [{"sleeve": "test", "reason": "thin"}],
                "top_label_actions": ["increase bot_b labels"],
            },
        },
    )
    _write_json(health / "supportability_control_latest.json", {"overall_status": "ready", "students_without_teachers": 0})
    _write_json(
        project_root / "governance" / "distillation" / "teacher_quality_latest.json",
        {
            "overall_status": "ready",
            "summary": {
                "qualified_teacher_count": 2,
                "elite_teacher_count": 1,
                "strong_teacher_count": 1,
                "uncovered_student_role_count": 0,
            },
            "student_role_coverage": {"uncovered_roles": []},
        },
    )
    if include_overfit:
        _write_json(
            health / "overfitting_awareness_latest.json",
            {
                "overall_status": "guarded" if guarded_teacher_blocks else "ready",
                "risk_bot_count": 0,
                "hard_risk_bot_count": 0,
                "guarded_bot_count": 3 if guarded_teacher_blocks else 0,
                "high_accuracy_guarded_bot_count": 3 if guarded_teacher_blocks else 0,
                "blocked_teacher_bot_count": 3 if guarded_teacher_blocks else 0,
                "blocked_teacher_bot_ids": ["bot_a", "bot_b", "bot_c"] if guarded_teacher_blocks else [],
                "teacher_ineligible_bot_count": 3 if guarded_teacher_blocks else 0,
            },
        )


def test_healthy_fleet_gets_a_plus_and_writes_runtime_override(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    override_path = tmp_path / "config" / ".env.bot_fleet_production_posture_override"

    payload = src.build_payload(tmp_path, apply=True, override_path=override_path)

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["production_posture_contract"]["grade"] == "A+"
    assert payload["production_posture_contract"]["missing"] == []
    assert payload["registry_contract"]["active_bots"] == 2
    assert payload["registry_contract"]["data_collection_active_bots"] == 2
    assert payload["paper_standard_contract"]["paper_live_data_enabled_bots"] == 1
    assert payload["quality_lane_contract"]["quality_debt_mode"] == "owned_repair_lanes"
    assert payload["bot_lanes"]["quality_repair"]["bot_count"] == 1
    assert payload["bot_lanes"]["overfit_containment"]["live_execution_authority"] is False

    text = override_path.read_text(encoding="utf-8")
    assert "BOT_FLEET_PRODUCTION_POSTURE_ENABLED=1" in text
    assert "BOT_FLEET_PRODUCTION_GRADE=A+" in text
    assert "BOT_FLEET_ACTIVE_BOT_COUNT=2" in text
    assert "BOT_FLEET_WEAK_BOT_ROUTING=owned_repair_lanes" in text
    assert "MARKET_DATA_ONLY=1" in text
    assert "ALLOW_ORDER_EXECUTION=0" in text


def test_live_execution_drift_blocks_whole_fleet_posture(tmp_path: Path) -> None:
    _seed_project(tmp_path, live_authority=True)

    payload = src.build_payload(tmp_path)

    missing = set(payload["production_posture_contract"]["missing"])
    assert payload["overall_status"] == "blocked"
    assert "registry_live_authority_absent" in missing
    assert "paper_live_execution_locked" in missing
    assert payload["registry_contract"]["live_authority_count"] == 1
    assert payload["paper_standard_contract"]["live_execution_locked"] is False
    assert payload["production_posture_contract"]["grade"] == "D"


def test_guarded_teacher_lockout_is_controlled_overfit_containment_not_fleet_failure(tmp_path: Path) -> None:
    _seed_project(tmp_path, guarded_teacher_blocks=True)

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["production_posture_contract"]["grade"] == "A+"
    assert payload["overfitting_contract"]["generalization_guard_ready"] is True
    assert payload["overfitting_contract"]["teacher_lockout_enforced"] is True
    assert payload["bot_lanes"]["overfit_containment"]["guarded_bot_count"] == 3


def test_recent_health_fast_inside_runtime_signal_window_does_not_false_downgrade(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    health_path = tmp_path / "governance" / "health" / "health_fast_latest.json"
    health_payload = json.loads(health_path.read_text(encoding="utf-8"))
    health_payload["timestamp_utc"] = _iso_minutes_ago(60)
    _write_json(health_path, health_payload)

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["production_posture_contract"]["grade"] == "A+"
    assert "source_artifacts_fresh" not in payload["production_posture_contract"]["missing"]
    assert payload["source_freshness_contract"]["sources"]["health_fast"]["max_age_minutes"] == 90.0


def test_stale_health_fast_outside_runtime_signal_window_blocks_fleet_posture(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    health_path = tmp_path / "governance" / "health" / "health_fast_latest.json"
    health_payload = json.loads(health_path.read_text(encoding="utf-8"))
    health_payload["timestamp_utc"] = _iso_minutes_ago(120)
    _write_json(health_path, health_payload)

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert "source_artifacts_fresh" in payload["production_posture_contract"]["missing"]
    assert "health_fast" in payload["source_freshness_contract"]["stale_or_missing"]


def test_missing_bot_contracts_block_whole_fleet_posture(tmp_path: Path) -> None:
    _seed_project(tmp_path, contract_gap=True)

    payload = src.build_payload(tmp_path)

    missing = set(payload["production_posture_contract"]["missing"])
    assert payload["overall_status"] == "blocked"
    assert "registry_label_contracts_complete" in missing
    assert "registry_target_functions_complete" in missing
    assert payload["registry_contract"]["missing_label_contract_count"] == 1
    assert payload["registry_contract"]["missing_target_functions_count"] == 1


def test_missing_overfitting_awareness_blocks_teacher_and_freshness_contracts(tmp_path: Path) -> None:
    _seed_project(tmp_path, include_overfit=False)

    payload = src.build_payload(tmp_path)

    missing = set(payload["production_posture_contract"]["missing"])
    assert payload["overall_status"] == "blocked"
    assert "overfitting_guard_ready" in missing
    assert "source_artifacts_fresh" in missing
    assert payload["overfitting_contract"]["generalization_guard_ready"] is False
    assert "overfitting_awareness" in payload["source_freshness_contract"]["stale_or_missing"]
