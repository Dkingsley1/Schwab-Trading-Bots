from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "ops" / "capital_rotation_control.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("capital_rotation_control", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load capital_rotation_control")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_base_sources(root: Path, *, pressure_guarded: bool = False) -> None:
    health = root / "governance" / "health"
    _write_json(
        health / "capital_growth_intelligence_latest.json",
        {
            "ok": True,
            "overall_status": "capital_growth_controls_ready",
            "live_money_scaling": {
                "allowed": False,
                "blockers": ["live_execution_still_requires_separate_operator_approval"],
            },
            "sleeve_growth_plan": [
                {
                    "profile": "quality_growth",
                    "growth_score": 92.0,
                    "growth_grade": "A+",
                    "capital_action": "candidate_for_growth",
                    "budget_reason": "scale_candidate_after_repeatability",
                    "executions": 44,
                    "win_rate": 0.64,
                    "realized_pnl": 80.0,
                    "unrealized_pnl": 5.0,
                    "net_pnl": 85.0,
                    "paper_sim_budget_pct": 0.06,
                    "paper_sim_budget_usd": 600.0,
                },
                {
                    "profile": "intraday_aggressive",
                    "growth_score": 30.0,
                    "growth_grade": "F",
                    "capital_action": "cap_or_quarantine",
                    "budget_reason": "quarantined_or_weak_profile",
                    "executions": 80,
                    "win_rate": 0.22,
                    "realized_pnl": -12.0,
                    "unrealized_pnl": -110.0,
                    "net_pnl": -122.0,
                    "paper_sim_budget_pct": 0.005,
                    "paper_sim_budget_usd": 50.0,
                },
            ],
        },
    )
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "ok": True,
            "overall_status": "protective_tightening",
            "active_profile_controls": {
                "intraday_aggressive": {
                    "action": "quarantine_new_entries",
                    "block_new_entries": True,
                    "position_size_multiplier": 0.05,
                }
            },
        },
    )
    _write_json(
        health / "whole_system_governor_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "sleeve_budgets": [
                {
                    "group": "quality_growth",
                    "value_score": 0.82,
                    "cost_score": 0.22,
                    "risk_score": 0.08,
                    "capture_tier": "normal_digest",
                    "governor_action": "normal_guarded_run",
                },
                {
                    "group": "intraday_aggressive",
                    "value_score": 0.34,
                    "cost_score": 0.58,
                    "risk_score": 0.52,
                    "capture_tier": "thin_digest",
                    "governor_action": "quarantine_until_policy_review",
                },
            ],
        },
    )
    _write_json(
        health / "paper_400_ramp_latest.json",
        {"ok": not pressure_guarded, "overall_status": "ready", "paper_ramp_stage": "blocked" if pressure_guarded else "armed", "blockers": ["storage_pressure"] if pressure_guarded else []},
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "ok": not pressure_guarded,
            "overall_status": "degraded" if pressure_guarded else "ready",
            "compute_pressure_level": "elevated" if pressure_guarded else "normal",
            "memory_pressure_level": "elevated" if pressure_guarded else "normal",
        },
    )
    _write_json(
        health / "health_fast_latest.json",
        {
            "ok": not pressure_guarded,
            "overall_status": "degraded" if pressure_guarded else "ready",
            "memory": {"overall_status": "needs_work" if pressure_guarded else "ready"},
            "storage": {"pressure_index": 0.82 if pressure_guarded else 0.20, "backpressure": {"total_pending_lines": 18000 if pressure_guarded else 100}},
            "operational_readiness": {
                "live_execution": {
                    "status": "blocked_read_only",
                    "blockers": ["live_execution_requires_explicit_operator_control"],
                },
                "guarded_paper": {
                    "status": "blocked" if pressure_guarded else "ready",
                    "paper_ramp_stage": "blocked" if pressure_guarded else "armed",
                    "blockers": ["runtime_status=degraded"] if pressure_guarded else [],
                    "paper_ramp_blockers": ["ingestion_or_backpressure_above_paper_400_gate"] if pressure_guarded else [],
                },
            },
        },
    )


def test_capital_rotation_builds_paper_wave_without_live_money(tmp_path: Path) -> None:
    module = _load_module()
    _write_base_sources(tmp_path)

    payload = module.build_payload(tmp_path)

    assert payload["overall_status"] == "capital_rotation_ready"
    assert payload["authority_boundary"] == "advisory_and_paper_rotation_only_no_live_money_movement"
    assert payload["runtime_contract"]["paper_rotation_allowed"] is True
    assert payload["runtime_contract"]["live_money_rotation_allowed"] is False
    assert payload["live_money_promotion_gate"]["allowed"] is False
    rows = {row["profile"]: row for row in payload["sleeve_rotation_plan"]}
    assert rows["quality_growth"]["recommended_action"] == "paper_expand_candidate"
    assert rows["quality_growth"]["paper_rotation_delta_norm"] > 0.0
    assert rows["quality_growth"]["live_rotation_delta_norm"] == 0.0
    assert rows["intraday_aggressive"]["recommended_action"] == "quarantine_or_reduce_only"
    assert rows["intraday_aggressive"]["paper_rotation_delta_norm"] < 0.0
    assert payload["portfolio_rotation"]["inflow_candidate_count"] == 1
    assert payload["portfolio_rotation"]["outflow_or_quarantine_count"] == 1


def test_capital_rotation_holds_inflow_when_pressure_guarded(tmp_path: Path) -> None:
    module = _load_module()
    _write_base_sources(tmp_path, pressure_guarded=True)

    payload = module.build_payload(tmp_path)

    assert payload["overall_status"] == "capital_rotation_advisory_only"
    assert payload["paper_rotation_gate"]["allowed"] is False
    assert payload["runtime_contract"]["paper_rotation_action_mode"] == "advisory_only_pressure_or_ramp_guarded"
    rows = {row["profile"]: row for row in payload["sleeve_rotation_plan"]}
    assert rows["quality_growth"]["recommended_action"] == "hold_inflow_until_pressure_clears"
    assert rows["quality_growth"]["paper_rotation_delta_norm"] == 0.0
    assert "runtime_pressure_guarded" in rows["quality_growth"]["why"]
    assert payload["runtime_contract"]["live_execution_allowed"] is False


def test_capital_rotation_prefers_newer_direct_ramp_over_stale_fast_health(tmp_path: Path) -> None:
    module = _load_module()
    _write_base_sources(tmp_path)
    health = tmp_path / "governance" / "health"
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    new = datetime.now(timezone.utc).isoformat()
    _write_json(
        health / "health_fast_latest.json",
        {
            "timestamp_utc": old,
            "ok": False,
            "overall_status": "degraded",
            "memory": {"overall_status": "needs_work"},
            "storage": {"pressure_index": 0.20, "backpressure": {"total_pending_lines": 100}},
            "operational_readiness": {
                "live_execution": {
                    "status": "blocked_read_only",
                    "blockers": ["live_execution_requires_explicit_operator_control"],
                },
                "guarded_paper": {
                    "status": "blocked",
                    "paper_ramp_stage": "blocked",
                    "blockers": ["paper_ramp_not_armed"],
                    "paper_ramp_blockers": ["runtime_capacity_not_ready_for_400_paper"],
                },
            },
        },
    )
    _write_json(
        health / "paper_400_ramp_latest.json",
        {"timestamp_utc": new, "ok": True, "stage": "armed", "blockers": []},
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": new,
            "ok": True,
            "overall_status": "ready",
            "compute_pressure_level": "normal",
            "memory_pressure_level": "normal",
        },
    )

    payload = module.build_payload(tmp_path)

    assert payload["overall_status"] == "capital_rotation_ready"
    assert payload["paper_rotation_gate"]["allowed"] is True
    assert payload["paper_rotation_gate"]["stale_health_fast_paper_gate_ignored"] is True
    assert payload["runtime_pressure"]["health_fast_ignored_as_stale"] is True
    assert "paper_ramp_not_armed" not in payload["paper_rotation_gate"]["blockers"]


def test_capital_rotation_allows_paper_tilt_under_advisory_elevated_compute(tmp_path: Path) -> None:
    module = _load_module()
    _write_base_sources(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "health_fast_latest.json",
        {
            "ok": True,
            "overall_status": "guarded_ready",
            "runtime_pressure": {
                "overall_status": "advisory",
                "host_saturation_score": 58.0,
                "compute_pressure_level": "elevated",
                "memory_pressure_level": "normal",
            },
            "memory": {"overall_status": "ready"},
            "storage": {"pressure_index": 0.20, "backpressure": {"total_pending_lines": 100}},
            "operational_readiness": {
                "live_execution": {
                    "status": "blocked_read_only",
                    "blockers": ["live_execution_requires_explicit_operator_control"],
                },
                "guarded_paper": {"status": "ready", "paper_ramp_stage": "armed", "blockers": [], "paper_ramp_blockers": []},
            },
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "ok": True,
            "overall_status": "advisory",
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "normal",
        },
    )

    payload = module.build_payload(tmp_path)

    assert payload["overall_status"] == "capital_rotation_ready"
    assert payload["runtime_pressure"]["guarded"] is False
    assert payload["paper_rotation_gate"]["allowed"] is True


def test_capital_rotation_apply_writes_live_locked_override(tmp_path: Path) -> None:
    module = _load_module()
    _write_base_sources(tmp_path)
    override = tmp_path / "config" / ".env.capital_rotation_control_override"

    payload = module.build_payload(tmp_path, apply=True, override_path=override)

    assert payload["write_result"]["applied"] is True
    text = override.read_text(encoding="utf-8")
    assert "CAPITAL_ROTATION_PAPER_TILT_ALLOWED=1" in text
    assert "CAPITAL_ROTATION_LIVE_MONEY_ALLOWED=0" in text
    assert "CAPITAL_ROTATION_LIVE_EXECUTION_ALLOWED=0" in text
