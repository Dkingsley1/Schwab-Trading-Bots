from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "quant_strategy_lane_upgrades_v1.json"
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "quant_strategy_lane_upgrades.py"
spec = importlib.util.spec_from_file_location("quant_strategy_lane_upgrades", MODULE_PATH)
lane_upgrades = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(lane_upgrades)


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


def test_quant_strategy_lane_upgrades_install_fifty_six_safe_modules() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    pack = payload["pack"]
    lanes = pack["lanes"]
    modules = [module for lane in lanes for module in lane["upgrade_modules"]]

    assert payload["quant_strategy_lane_upgrades_version"] == "quant_strategy_lane_upgrades_v1"
    assert pack["lane_count"] == 7
    assert len(lanes) == 7
    assert len(modules) == 56
    assert pack["total_upgrade_modules"] == 56
    assert pack["paper_trading_enabled"] is False
    assert pack["live_trading_enabled"] is False
    assert pack["execution_enabled"] is False
    assert pack["allocation_enabled"] is False
    assert pack["heavy_training_enabled"] is False
    assert pack["new_high_volume_collectors_enabled"] is False
    assert pack["registry_promotion_side_effects_allowed"] is False
    for lane in lanes:
        assert lane["current_state"] == "collection_only_upgraded"
        assert len(lane["upgrade_modules"]) == 8
        assert lane["paper_trading_enabled"] is False
        assert lane["live_trading_enabled"] is False
        assert lane["execution_enabled"] is False
        assert lane["allocation_enabled"] is False
        for module in lane["upgrade_modules"]:
            assert module["safe_now"] is True
            assert module["outputs"]


def test_quant_strategy_lane_upgrades_report_collection_runtime_active_with_paper_blocked(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _write_json(tmp_path / "config" / "quant_strategy_lane_upgrades_v1.json", config)
    _write_json(
        tmp_path / "governance" / "health" / "health_fast_latest.json",
        {
            "global_halt": {"halt": False, "clear_blockers": []},
            "runtime_pressure": {"overall_status": "degraded", "compute_pressure_level": "elevated"},
            "storage": {
                "severity": "stable",
                "backpressure": {
                    "core_pending_lines": 7404,
                    "total_pending_lines": 7410,
                    "oldest_pending_age_seconds": 81.159,
                    "pending_lines_threshold": 15000,
                    "oldest_age_threshold_seconds": 240,
                },
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "paper_400_ramp_latest.json",
        {"ok": False, "stage": "blocked", "blockers": ["runtime_capacity_not_ready_for_400_paper"]},
    )

    payload = lane_upgrades.build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["overall_status"] == "collection_runtime_active_paper_activation_blocked"
    assert payload["lane_count"] == 7
    assert payload["total_upgrade_modules"] == 56
    assert payload["safe_now_upgrade_modules"] == 56
    assert payload["collection_runtime_active"] is True
    assert payload["paper_activation_ready"] is False
    assert payload["runtime_activation_ready"] is False
    assert payload["gate_state"]["storage_green"] is True
    assert payload["gate_state"]["runtime_green"] is False
    assert payload["forbidden_enabled"] == []


def test_quant_strategy_lane_upgrades_still_block_activation_without_promotion_quality(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _write_json(tmp_path / "config" / "quant_strategy_lane_upgrades_v1.json", config)
    _write_json(
        tmp_path / "governance" / "health" / "health_fast_latest.json",
        {
            "global_halt": {"halt": False, "clear_blockers": []},
            "runtime_pressure": {"overall_status": "ready", "compute_pressure_level": "normal"},
            "storage": {
                "severity": "stable",
                "backpressure": {
                    "core_pending_lines": 100,
                    "total_pending_lines": 120,
                    "oldest_pending_age_seconds": 5,
                    "pending_lines_threshold": 15000,
                    "oldest_age_threshold_seconds": 240,
                },
            },
        },
    )
    _write_json(tmp_path / "governance" / "health" / "paper_400_ramp_latest.json", {"ok": True, "stage": "ready"})

    payload = lane_upgrades.build_payload(tmp_path)

    assert payload["overall_status"] == "collection_runtime_active_paper_activation_blocked"
    assert payload["collection_runtime_active"] is True
    assert payload["paper_activation_ready"] is False
    assert payload["gate_state"]["runtime_green"] is True
    assert payload["gate_state"]["paper_400_ready"] is True
    assert payload["gate_state"]["promotion_quality_ready"] is False


def test_quant_strategy_lane_upgrades_allow_paper_activation_when_all_gates_are_green(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _write_json(tmp_path / "config" / "quant_strategy_lane_upgrades_v1.json", config)
    _write_json(
        tmp_path / "governance" / "health" / "health_fast_latest.json",
        {
            "global_halt": {"halt": False, "clear_blockers": []},
            "runtime_pressure": {"overall_status": "ready", "compute_pressure_level": "normal"},
            "storage": {
                "severity": "ready",
                "backpressure": {
                    "core_pending_lines": 100,
                    "total_pending_lines": 120,
                    "oldest_pending_age_seconds": 5,
                    "pending_lines_threshold": 15000,
                    "oldest_age_threshold_seconds": 240,
                },
            },
        },
    )
    _write_json(tmp_path / "governance" / "health" / "paper_400_ramp_latest.json", {"ok": True, "stage": "ready"})
    _write_promotion_green(tmp_path)

    payload = lane_upgrades.build_payload(tmp_path)

    assert payload["overall_status"] == "paper_activation_ready"
    assert payload["collection_runtime_active"] is True
    assert payload["paper_activation_ready"] is True
    assert payload["runtime_activation_ready"] is True
    assert payload["gate_state"]["promotion_quality_ready"] is True
