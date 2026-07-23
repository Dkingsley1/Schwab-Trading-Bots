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
