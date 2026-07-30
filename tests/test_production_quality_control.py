from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import production_quality_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _write_readiness(project_root: Path, payload: dict) -> None:
    _write_json(project_root / "governance" / "health" / "live_canary_readiness_contract_latest.json", payload)


def test_production_quality_control_routes_live_canary_blockers(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_readiness(
        project_root,
        {
            "overall_status": "blocked",
            "live_canary_money_ready": False,
            "ready_gate_count": 2,
            "gate_count": 7,
            "blockers": [
                "raw_profitability_posture_blocked",
                "storage_pressure_clean_blocked",
                "promotion_paper_gate_freshness_blocked",
            ],
            "gates": [
                {"gate_id": "raw_profitability_posture", "ready": False, "blockers": ["raw_profitability_grade_below_A"]},
                {"gate_id": "storage_pressure_clean", "ready": False, "blockers": ["storage_pressure_index_too_high"]},
                {"gate_id": "promotion_paper_gate_freshness", "ready": False, "blockers": ["paper_performance_not_ready_or_stale"]},
            ],
        },
    )

    payload = src.build_payload(project_root)

    lane_ids = {lane["lane_id"] for lane in payload["active_lanes"]}
    assert payload["overall_status"] == "blocked"
    assert payload["live_execution_authority"] is False
    assert payload["safe_apply_only"] is True
    assert payload["live_orders_must_remain_disabled"] is True
    assert lane_ids == {"raw_profitability_recovery", "storage_pressure_clean", "promotion_paper_freshness"}
    assert ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"] in payload["ordered_repair_commands"]
    assert ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--quick-bounded", "--json"] in payload["ordered_repair_commands"]
    assert ["./scripts/ops/opsctl.sh", "daily-verify-remediation", "--apply", "--json"] in payload["ordered_repair_commands"]
    assert payload["quality_checks"]["all_active_lanes_have_stop_conditions"] is True


def test_production_quality_control_ready_when_live_canary_ready(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_readiness(
        project_root,
        {
            "overall_status": "ready",
            "live_canary_money_ready": True,
            "ready_gate_count": 7,
            "gate_count": 7,
            "blockers": [],
            "gates": [],
        },
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["live_orders_must_remain_disabled"] is False
    assert payload["active_lane_count"] == 0


def test_production_quality_control_manages_live_money_locks_when_soak_is_ready(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_readiness(
        project_root,
        {
            "overall_status": "blocked",
            "live_canary_money_ready": False,
            "ready_gate_count": 3,
            "gate_count": 7,
            "blockers": [
                "raw_profitability_posture_blocked",
                "sleeve_paper_trading_continuity_blocked",
                "storage_pressure_clean_blocked",
                "promotion_paper_gate_freshness_blocked",
            ],
            "gates": [
                {
                    "gate_id": "raw_profitability_posture",
                    "ready": False,
                    "blockers": ["raw_profitability_grade_below_A", "raw_profitability_hard_block_below_C"],
                    "evidence": {"raw_profitability_grade": "D"},
                },
                {
                    "gate_id": "sleeve_paper_trading_continuity",
                    "ready": False,
                    "blockers": ["paper_ramp_blockers_present"],
                    "evidence": {
                        "paper_truth_status": "ready",
                        "runtime_paper_status": "ready",
                        "paper_ramp_blockers": ["memory_pressure_above_paper_400_gate"],
                    },
                },
                {
                    "gate_id": "storage_pressure_clean",
                    "ready": False,
                    "blockers": ["storage_pressure_index_too_high"],
                    "evidence": {
                        "overall_status": "ready",
                        "severity": "stable",
                        "pressure_index": 0.447,
                        "total_pending_lines": 1348,
                        "max_total_pending_lines": 15000,
                    },
                },
                {
                    "gate_id": "promotion_paper_gate_freshness",
                    "ready": False,
                    "blockers": ["promotion_quality_gate_not_ready_or_stale"],
                    "evidence": {},
                },
            ],
        },
    )
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "overall_grade": "A+",
            "safe_to_leave_unattended": True,
            "blockers": [],
            "sections": {
                "runtime_loops": {"ready": True, "grade": "A+"},
                "storage": {"ready": True, "grade": "A+"},
            },
        },
    )
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "ok": True,
            "overall_status": "protective_tightening",
            "profitability_grade_basis": "controlled_recovery_posture",
            "raw_profitability_grade": "D",
            "controlled_profitability_grade": "A+",
            "profitability_display_grade": "A+ controlled / D raw",
        },
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["live_orders_must_remain_disabled"] is True
    assert payload["raw_active_lane_count"] == 4
    assert payload["active_lane_count"] == 0
    assert set(payload["managed_live_money_locks"]) == {
        "raw_profitability_below_live_canary_floor_controlled_recovery_active",
        "paper_400_expansion_paused_existing_paper_soak_ready",
        "storage_pressure_above_live_canary_floor_bounded_for_soak",
        "promotion_pipeline_live_money_locked_current_soak_ready",
    }
    assert payload["quality_checks"]["managed_live_money_locks_have_no_live_execution_authority"] is True


def test_production_quality_control_manages_bounded_storage_pressure_live_money_lock(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_readiness(
        project_root,
        {
            "overall_status": "blocked",
            "live_canary_money_ready": False,
            "blockers": ["storage_pressure_clean_blocked"],
            "gates": [
                {
                    "gate_id": "storage_pressure_clean",
                    "ready": False,
                    "blockers": ["storage_pressure_index_too_high", "storage_total_pending_lines_too_high"],
                    "evidence": {
                        "overall_status": "ready",
                        "severity": "stable",
                        "pressure_index": 0.624,
                        "max_pressure_index": 0.2,
                        "total_pending_lines": 187146,
                        "max_total_pending_lines": 15000,
                    },
                }
            ],
        },
    )
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "overall_grade": "A+",
            "safe_to_leave_unattended": True,
            "blockers": [],
            "sections": {
                "storage": {"ready": True, "grade": "A+", "ingestion_soak_ready": True},
            },
        },
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["active_lane_count"] == 0
    assert payload["managed_live_money_locks"] == ["storage_pressure_above_live_canary_floor_bounded_for_soak"]
    lock = payload["managed_live_money_lock_lanes"][0]["managed_lock"]
    assert lock["managed_pressure_ceiling"] == 0.8
    assert lock["managed_total_pending_ceiling"] == 300000
    assert lock["live_execution_authority"] is False


def test_production_quality_control_does_not_manage_unexplained_paper_dropout(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_readiness(
        project_root,
        {
            "overall_status": "blocked",
            "live_canary_money_ready": False,
            "blockers": ["sleeve_paper_trading_continuity_blocked"],
            "gates": [
                {
                    "gate_id": "sleeve_paper_trading_continuity",
                    "ready": False,
                    "blockers": ["runtime_paper_failed_checks_present"],
                    "evidence": {
                        "paper_truth_status": "ready",
                        "runtime_paper_status": "blocked",
                        "runtime_paper_failed_checks": ["paper_trading_inactive_for_eligible_sleeve"],
                    },
                }
            ],
        },
    )
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {"ok": True, "overall_status": "ready", "overall_grade": "A+", "safe_to_leave_unattended": True, "blockers": []},
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert payload["active_lane_count"] == 1
    assert payload["managed_live_money_locks"] == []


def test_production_quality_control_manages_auth_probe_drift_when_paper_soak_operable(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_readiness(
        project_root,
        {
            "overall_status": "blocked",
            "live_canary_money_ready": False,
            "blockers": ["auth_token_continuity_blocked"],
            "gates": [
                {
                    "gate_id": "auth_token_continuity",
                    "ready": False,
                    "blockers": ["broker_auth_not_ok", "auth_status_not_ready", "auth_expires_below_1800s"],
                    "evidence": {
                        "broker_ready": True,
                        "broker_auth_ok": False,
                        "broker_network_ok": True,
                        "schwab_auth_status": "degraded",
                        "auth_lease_status": "degraded",
                        "lease_state": "warning",
                        "expires_in_seconds": 1324,
                    },
                }
            ],
        },
    )
    _write_json(
        health / "unattended_soak_readiness_latest.json",
        {"ok": True, "overall_status": "ready", "overall_grade": "A+", "safe_to_leave_unattended": True, "blockers": []},
    )
    _write_json(
        health / "schwab_auth_supervisor_latest.json",
        {
            "overall_status": "degraded",
            "paper_soak_auth_operable": True,
            "token": {"ready": True, "expires_in_seconds": 1324},
        },
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["active_lane_count"] == 0
    assert payload["live_orders_must_remain_disabled"] is True
    assert payload["managed_live_money_locks"] == ["auth_probe_not_live_money_clean_paper_soak_operable"]
    assert payload["managed_live_money_lock_lanes"][0]["managed_lock"]["paper_soak_auth_operable"] is True


def test_production_quality_control_delegates_safe_execution_to_governor(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    calls: dict[str, object] = {}
    _write_readiness(
        project_root,
        {
            "overall_status": "blocked",
            "live_canary_money_ready": False,
            "blockers": ["auth_token_continuity_blocked"],
            "gates": [{"gate_id": "auth_token_continuity", "ready": False, "blockers": ["auth_expires_below_1800s"]}],
        },
    )

    def fake_governor_build_payload(
        root: Path,
        *,
        apply: bool,
        max_actions: int,
        execute_safe_repairs: bool,
        max_execute_actions: int,
        command_timeout_seconds: int,
    ) -> dict:
        calls.update(
            {
                "root": root,
                "apply": apply,
                "max_actions": max_actions,
                "execute_safe_repairs": execute_safe_repairs,
                "max_execute_actions": max_execute_actions,
                "command_timeout_seconds": command_timeout_seconds,
            }
        )
        return {"apply_result": {"safe_repair_execution": {"executed_count": 1, "live_execution_authority": False}}}

    monkeypatch.setattr(src.infrabot_adaptive_governor, "build_payload", fake_governor_build_payload)

    payload = src.build_payload(
        project_root,
        apply=True,
        execute_safe_repairs=True,
        max_actions=5,
        max_execute_actions=2,
        command_timeout_seconds=45,
    )

    assert calls["root"] == project_root
    assert calls["apply"] is True
    assert calls["execute_safe_repairs"] is True
    assert calls["max_actions"] == 5
    assert calls["max_execute_actions"] == 2
    assert payload["execution_result"]["executed_count"] == 1
    assert (project_root / "governance" / "health" / "production_quality_control_latest.json").exists()
