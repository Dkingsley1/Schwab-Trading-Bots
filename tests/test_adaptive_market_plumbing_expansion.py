from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import roster_expansion_slots as slots_src


ADAPTIVE_MARKET_PLUMBING_BOT_IDS = {
    "brain_refinery_v694_collateral_quality_liquidity_ladder_bot",
    "brain_refinery_v695_margin_call_cascade_preemption_bot",
    "brain_refinery_v696_repo_haircut_shock_bridge_bot",
    "brain_refinery_v697_forced_liquidation_waterfall_bot",
    "brain_refinery_v698_collateral_margin_regression_guard_bot",
    "brain_refinery_v699_dealer_gamma_inventory_pressure_bot",
    "brain_refinery_v700_gamma_flip_crowding_detector_bot",
    "brain_refinery_v701_vanna_charm_flow_decay_bot",
    "brain_refinery_v702_vol_control_rebalance_pressure_bot",
    "brain_refinery_v703_dealer_gamma_inventory_regression_guard_bot",
    "brain_refinery_v704_etf_creation_redemption_flow_bot",
    "brain_refinery_v705_authorized_participant_liquidity_gap_bot",
    "brain_refinery_v706_etf_nav_premium_discount_dislocation_bot",
    "brain_refinery_v707_basket_liquidity_stress_router_bot",
    "brain_refinery_v708_etf_flow_regression_guard_bot",
    "brain_refinery_v709_ensemble_disagreement_resolver_bot",
    "brain_refinery_v710_signal_half_life_decay_governor_bot",
    "brain_refinery_v711_alpha_crowding_duplicate_detector_bot",
    "brain_refinery_v712_experiment_kill_criteria_arbiter_bot",
    "brain_refinery_v713_signal_governance_regression_guard_bot",
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_adaptive_market_plumbing_slots_are_planned() -> None:
    specs = {
        str(row.get("bot_id") or ""): row
        for row in slots_src.DEFAULT_SLOT_SPECS
        if str(row.get("bot_id") or "") in ADAPTIVE_MARKET_PLUMBING_BOT_IDS
    }

    assert set(specs) == ADAPTIVE_MARKET_PLUMBING_BOT_IDS
    assert {row["sleeve_profile"] for row in specs.values()} == {
        "collateral_margin_liquidity",
        "dealer_positioning_gamma_inventory",
        "etf_flow_creation_redemption",
        "signal_governance_integrity",
    }
    assert all(row["sleeve_family"] == "quant_models" for row in specs.values())
    assert specs["brain_refinery_v698_collateral_margin_regression_guard_bot"]["bot_role"] == "infrastructure_sub_bot"
    assert specs["brain_refinery_v703_dealer_gamma_inventory_regression_guard_bot"]["bot_role"] == "infrastructure_sub_bot"
    assert specs["brain_refinery_v708_etf_flow_regression_guard_bot"]["bot_role"] == "infrastructure_sub_bot"
    assert specs["brain_refinery_v713_signal_governance_regression_guard_bot"]["bot_role"] == "infrastructure_sub_bot"


def test_adaptive_market_plumbing_apply_keeps_collection_only_contract(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    registry_path = project_root / "master_bot_registry.json"
    _write_json(registry_path, {"summary": {}, "sub_bots": []})

    apply_result = slots_src.apply_registry(project_root, registry_path=registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = {
        str(row.get("bot_id") or ""): row
        for row in registry.get("sub_bots", [])
        if str(row.get("bot_id") or "") in ADAPTIVE_MARKET_PLUMBING_BOT_IDS
    }

    assert set(rows) == ADAPTIVE_MARKET_PLUMBING_BOT_IDS
    assert apply_result["added_slots"] >= len(ADAPTIVE_MARKET_PLUMBING_BOT_IDS)
    for row in rows.values():
        assert row["active"] is True
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["data_collection_active"] is True
        assert row["training_excluded"] is True
        assert row["exclude_from_training"] is True
        assert row["training_candidate_after_threshold"] is True
        assert row["allocation_enabled"] is False
        assert row["paper_trading_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["minimum_training_observations"] >= 3000
        assert "research_only" in row["labeling_tags"]
        assert row["direct_execution_allowed"] is False


def test_sleeve_strategy_manifest_includes_adaptive_market_plumbing() -> None:
    manifest = json.loads((PROJECT_ROOT / "config" / "sleeve_strategy_expansion.json").read_text(encoding="utf-8"))
    sleeves = {str(row.get("name") or ""): row for row in manifest["sleeves"]}

    assert set(sleeves) >= {
        "collateral_margin_liquidity",
        "dealer_positioning_gamma_inventory",
        "etf_flow_creation_redemption",
        "signal_governance_integrity",
    }
    assert len(sleeves["collateral_margin_liquidity"]["strategies"]) == 5
    assert len(sleeves["dealer_positioning_gamma_inventory"]["strategies"]) == 5
    assert len(sleeves["etf_flow_creation_redemption"]["strategies"]) == 5
    assert len(sleeves["signal_governance_integrity"]["strategies"]) == 5
