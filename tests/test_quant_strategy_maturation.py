from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "quant_strategy_maturation_v1.json"
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "quant_strategy_maturation.py"
spec = importlib.util.spec_from_file_location("quant_strategy_maturation", MODULE_PATH)
quant_maturation = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(quant_maturation)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_promotion_green(tmp_path: Path) -> None:
    _write_json(tmp_path / "governance" / "health" / "promotion_quality_gate_latest.json", {"ok": True, "failed_checks": []})
    _write_json(
        tmp_path / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"promote_ok": True, "blocking_reasons": []},
    )
    _write_json(
        tmp_path / "governance" / "champion_challenger" / "promotion_packet_latest.json",
        {
            "ok": True,
            "ready_for_committee": True,
            "packet_complete": True,
            "signature": {"verified": True},
        },
    )


def test_quant_strategy_maturation_pack_has_seven_collection_only_lanes() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    pack = payload["pack"]
    lanes = pack["lanes"]

    assert payload["quant_strategy_maturation_version"] == "quant_strategy_maturation_v1"
    assert pack["lane_count"] == 7
    assert len(lanes) == 7
    assert pack["paper_trading_enabled"] is False
    assert pack["live_trading_enabled"] is False
    assert pack["execution_enabled"] is False
    assert pack["allocation_enabled"] is False
    assert pack["activation_policy"]["registry_promotion_side_effects_allowed"] is False
    assert {lane["slug"] for lane in lanes} == {
        "unified_equity_factor_sleeve",
        "residual_stat_arb_ou_pairs",
        "vol_risk_premium_dispersion_dealer",
        "event_corporate_action_arb",
        "international_adr_fx_rates_rv",
        "microstructure_execution_intelligence",
        "capacity_aware_meta_allocator",
    }
    for lane in lanes:
        assert lane["current_state"] == "collection_only"
        assert lane["paper_trading_enabled"] is False
        assert lane["live_trading_enabled"] is False
        assert lane["execution_enabled"] is False
        assert lane["allocation_enabled"] is False
        assert "queue_backpressure_clear" in lane["paper_canary_when"]
        assert len(lane["candidate_bot_ids"]) >= 6


def test_quant_strategy_maturation_blocks_when_pressure_gates_are_red(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _write_json(tmp_path / "config" / "quant_strategy_maturation_v1.json", config)
    _write_json(
        tmp_path / "governance" / "health" / "health_fast_latest.json",
        {
            "global_halt": {"halt": False, "clear_blockers": ["queue_backpressure_active"]},
            "runtime_pressure": {"overall_status": "blocked", "compute_pressure_level": "high"},
            "storage": {
                "severity": "critical",
                "backpressure": {
                    "core_pending_lines": 59000,
                    "total_pending_lines": 59600,
                    "oldest_pending_age_seconds": 260,
                    "pending_lines_threshold": 15000,
                    "oldest_age_threshold_seconds": 240,
                },
            },
        },
    )
    _write_json(tmp_path / "governance" / "health" / "runtime_throttle_latest.json", {"overall_status": "blocked"})
    _write_json(
        tmp_path / "governance" / "health" / "writer_cycle_coordinator_latest.json",
        {"overall_status": "waiting_for_writer", "writer_state_before": {"active": True}},
    )
    _write_json(
        tmp_path / "governance" / "health" / "paper_400_ramp_latest.json",
        {"overall_status": "blocked", "blockers": ["ingestion_or_backpressure_above_paper_400_gate"]},
    )

    payload = quant_maturation.build_payload(tmp_path)

    assert payload["overall_status"] == "collection_runtime_active_paper_canary_blocked"
    assert "queue_backpressure_clear" in payload["failed_gates"]
    assert "runtime_capacity_ready" in payload["failed_gates"]
    assert "writer_idle_or_coordinated" in payload["failed_gates"]
    assert payload["collection_runtime_active"] is True
    assert payload["paper_trading_enabled"] is False
    assert payload["live_trading_enabled"] is False


def test_quant_strategy_maturation_still_requires_quality_gate_when_capacity_is_green(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _write_json(tmp_path / "config" / "quant_strategy_maturation_v1.json", config)
    _write_json(
        tmp_path / "governance" / "health" / "health_fast_latest.json",
        {
            "global_halt": {"halt": False, "clear_blockers": []},
            "runtime_pressure": {"overall_status": "ready", "compute_pressure_level": "normal"},
            "storage": {
                "severity": "ready",
                "backpressure": {
                    "core_pending_lines": 100,
                    "total_pending_lines": 150,
                    "oldest_pending_age_seconds": 10,
                    "pending_lines_threshold": 15000,
                    "oldest_age_threshold_seconds": 240,
                },
            },
        },
    )
    _write_json(tmp_path / "governance" / "health" / "runtime_throttle_latest.json", {"overall_status": "ready"})
    _write_json(
        tmp_path / "governance" / "health" / "writer_cycle_coordinator_latest.json",
        {"overall_status": "ready", "writer_state_before": {"active": False}},
    )
    _write_json(tmp_path / "governance" / "health" / "paper_400_ramp_latest.json", {"overall_status": "ready"})

    payload = quant_maturation.build_payload(tmp_path)

    assert payload["overall_status"] == "collection_runtime_active_paper_canary_blocked"
    assert payload["failed_gates"] == ["promotion_quality_gates_ready"]
    assert payload["collection_runtime_active"] is True
    assert payload["activation_policy"]["paper_canary_allowed_now"] is False


def test_quant_strategy_maturation_blocks_degraded_elevated_runtime(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _write_json(tmp_path / "config" / "quant_strategy_maturation_v1.json", config)
    _write_json(
        tmp_path / "governance" / "health" / "health_fast_latest.json",
        {
            "global_halt": {"halt": False, "clear_blockers": []},
            "runtime_pressure": {"overall_status": "degraded", "compute_pressure_level": "elevated"},
            "storage": {
                "severity": "ready",
                "backpressure": {
                    "core_pending_lines": 100,
                    "total_pending_lines": 150,
                    "oldest_pending_age_seconds": 10,
                    "pending_lines_threshold": 15000,
                    "oldest_age_threshold_seconds": 240,
                },
            },
        },
    )
    _write_json(tmp_path / "governance" / "health" / "runtime_throttle_latest.json", {"overall_status": "ready"})
    _write_json(
        tmp_path / "governance" / "health" / "writer_cycle_coordinator_latest.json",
        {"overall_status": "ready", "writer_state_before": {"active": False}},
    )
    _write_json(tmp_path / "governance" / "health" / "paper_400_ramp_latest.json", {"overall_status": "ready", "ok": True})
    _write_promotion_green(tmp_path)

    payload = quant_maturation.build_payload(tmp_path)

    assert payload["overall_status"] == "collection_runtime_active_paper_canary_blocked"
    assert payload["failed_gates"] == ["runtime_capacity_ready"]
    assert payload["collection_runtime_active"] is True


def test_quant_strategy_maturation_reports_paper_canary_ready_when_all_gates_are_green(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _write_json(tmp_path / "config" / "quant_strategy_maturation_v1.json", config)
    _write_json(
        tmp_path / "governance" / "health" / "health_fast_latest.json",
        {
            "global_halt": {"halt": False, "clear_blockers": []},
            "runtime_pressure": {"overall_status": "ready", "compute_pressure_level": "normal"},
            "storage": {
                "severity": "ready",
                "backpressure": {
                    "core_pending_lines": 100,
                    "total_pending_lines": 150,
                    "oldest_pending_age_seconds": 10,
                    "pending_lines_threshold": 15000,
                    "oldest_age_threshold_seconds": 240,
                },
            },
        },
    )
    _write_json(tmp_path / "governance" / "health" / "runtime_throttle_latest.json", {"overall_status": "ready"})
    _write_json(
        tmp_path / "governance" / "health" / "writer_cycle_coordinator_latest.json",
        {"overall_status": "ready", "writer_state_before": {"active": False}},
    )
    _write_json(tmp_path / "governance" / "health" / "paper_400_ramp_latest.json", {"overall_status": "ready", "ok": True})
    _write_promotion_green(tmp_path)

    payload = quant_maturation.build_payload(tmp_path)

    assert payload["overall_status"] == "paper_canary_queue_ready"
    assert payload["failed_gates"] == []
    assert payload["collection_runtime_active"] is True
