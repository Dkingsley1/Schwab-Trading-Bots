import json
from pathlib import Path

from scripts.ops.deep_quant_layer_upgrade import build_payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_deep_quant_layer_upgrade_is_collection_only(tmp_path):
    health_root = tmp_path / "governance" / "health"
    _write_json(
        health_root / "strategy_inventory_latest.json",
        {
            "bot_count": 1651,
            "sleeve_count": 110,
            "strategy_count": 766,
            "advanced_collection_sleeves": [
                "statistical_arbitrage",
                "signal_governance_integrity",
                "uncertainty_robust_control",
                "transaction_cost_slippage_intelligence",
                "portfolio_construction",
                "causal_regime_discovery",
                "dealer_positioning_gamma_inventory",
                "event_intelligence",
                "alpha_research_os",
            ],
            "sleeves": [],
        },
    )
    _write_json(
        health_root / "quant_strategy_lane_upgrades_latest.json",
        {
            "overall_status": "collection_runtime_active_paper_activation_blocked",
            "paper_activation_ready": False,
            "gate_state": {
                "global_halt_clear": True,
                "storage_green": True,
                "runtime_green": True,
                "paper_400_ready": True,
                "promotion_quality_ready": False,
                "quality_gate_ok": False,
                "promotion_readiness_ok": False,
                "promotion_packet_ok": False,
                "promotion_quality_failed_checks": ["promotion_gate_blocked"],
                "promotion_readiness_blockers": ["insufficient_walk_forward_coverage"],
            },
        },
    )
    _write_json(health_root / "retrain_launch_latest.json", {"state": "running", "pid": 123})

    payload = build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["layer_count"] == 10
    assert payload["covered_layer_count"] == 10
    assert payload["paper_enabled_layer_count"] == 0
    assert payload["live_enabled_layer_count"] == 0
    assert payload["execution_enabled_layer_count"] == 0
    assert payload["training_intake_enabled_layer_count"] == 0
    assert "large_training_batch_running_control_plane_only" in payload["activation_blockers"]
    assert {layer["activation_state"] for layer in payload["layers"]} == {"collection_only_advisory"}
