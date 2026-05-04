from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import roster_expansion_slots as slots_src


ADVANCED_QUANT_RESEARCH_BOT_IDS = {
    "brain_refinery_v734_fourier_neural_operator_surface_bot",
    "brain_refinery_v735_deeponet_payoff_operator_bot",
    "brain_refinery_v736_neural_cde_path_encoder_bot",
    "brain_refinery_v737_koopman_dynamic_mode_regime_bot",
    "brain_refinery_v738_neural_operator_surrogate_regression_guard_bot",
    "brain_refinery_v739_bayesian_neural_uncertainty_bot",
    "brain_refinery_v740_conformal_prediction_interval_guard_bot",
    "brain_refinery_v741_distributionally_robust_optimizer_bot",
    "brain_refinery_v742_online_convex_regret_minimizer_bot",
    "brain_refinery_v743_uncertainty_robust_control_regression_guard_bot",
    "brain_refinery_v744_causal_discovery_dag_bot",
    "brain_refinery_v745_invariant_risk_minimization_bot",
    "brain_refinery_v746_bayesian_online_changepoint_bot",
    "brain_refinery_v747_hidden_semi_markov_duration_bot",
    "brain_refinery_v748_causal_regime_discovery_regression_guard_bot",
    "brain_refinery_v749_martingale_optimal_transport_pricing_bot",
    "brain_refinery_v750_normalizing_flow_density_surface_bot",
    "brain_refinery_v751_least_squares_monte_carlo_optimal_stopping_bot",
    "brain_refinery_v752_rough_bergomi_forward_variance_bot",
    "brain_refinery_v753_martingale_flow_pricing_regression_guard_bot",
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_advanced_quant_research_slots_are_planned() -> None:
    specs = {
        str(row.get("bot_id") or ""): row
        for row in slots_src.DEFAULT_SLOT_SPECS
        if str(row.get("bot_id") or "") in ADVANCED_QUANT_RESEARCH_BOT_IDS
    }

    assert set(specs) == ADVANCED_QUANT_RESEARCH_BOT_IDS
    assert {row["sleeve_profile"] for row in specs.values()} == {
        "neural_operator_surrogates",
        "uncertainty_robust_control",
        "causal_regime_discovery",
        "martingale_flow_pricing",
    }
    assert all(row["sleeve_family"] == "quant_models" for row in specs.values())
    assert specs["brain_refinery_v738_neural_operator_surrogate_regression_guard_bot"]["bot_role"] == "infrastructure_sub_bot"
    assert specs["brain_refinery_v743_uncertainty_robust_control_regression_guard_bot"]["bot_role"] == "infrastructure_sub_bot"
    assert specs["brain_refinery_v748_causal_regime_discovery_regression_guard_bot"]["bot_role"] == "infrastructure_sub_bot"
    assert specs["brain_refinery_v753_martingale_flow_pricing_regression_guard_bot"]["bot_role"] == "infrastructure_sub_bot"
    assert "fourier_neural_operator_surface_trace" in specs["brain_refinery_v734_fourier_neural_operator_surface_bot"]["data_intake_collections"]
    assert "conformal_prediction_interval_trace" in specs["brain_refinery_v740_conformal_prediction_interval_guard_bot"]["data_intake_collections"]
    assert "causal_discovery_dag_trace" in specs["brain_refinery_v744_causal_discovery_dag_bot"]["data_intake_collections"]
    assert "martingale_optimal_transport_pricing_trace" in specs["brain_refinery_v749_martingale_optimal_transport_pricing_bot"]["data_intake_collections"]


def test_advanced_quant_research_apply_keeps_collection_only_contract(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    registry_path = project_root / "master_bot_registry.json"
    _write_json(registry_path, {"summary": {}, "sub_bots": []})

    apply_result = slots_src.apply_registry(project_root, registry_path=registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = {
        str(row.get("bot_id") or ""): row
        for row in registry.get("sub_bots", [])
        if str(row.get("bot_id") or "") in ADVANCED_QUANT_RESEARCH_BOT_IDS
    }

    assert set(rows) == ADVANCED_QUANT_RESEARCH_BOT_IDS
    assert apply_result["added_slots"] >= len(ADVANCED_QUANT_RESEARCH_BOT_IDS)
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


def test_advanced_quant_research_provider_and_storage_contracts(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    registry_path = project_root / "master_bot_registry.json"
    _write_json(registry_path, {"summary": {}, "sub_bots": []})

    slots_src.apply_registry(project_root, registry_path=registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = {
        str(row.get("sleeve_profile") or ""): row
        for row in registry.get("sub_bots", [])
        if str(row.get("bot_id") or "") in {
            "brain_refinery_v734_fourier_neural_operator_surface_bot",
            "brain_refinery_v739_bayesian_neural_uncertainty_bot",
            "brain_refinery_v744_causal_discovery_dag_bot",
            "brain_refinery_v749_martingale_optimal_transport_pricing_bot",
        }
    }

    assert rows["neural_operator_surrogates"]["provider_capability_profile"] == "research_only_neural_operator_surrogate_proxy"
    assert rows["uncertainty_robust_control"]["provider_capability_profile"] == "research_only_uncertainty_robust_control_proxy"
    assert rows["causal_regime_discovery"]["provider_capability_profile"] == "research_only_causal_regime_discovery_proxy"
    assert rows["martingale_flow_pricing"]["provider_capability_profile"] == "research_only_martingale_flow_pricing_proxy"
    assert "governance/quant_models/neural_operator_surrogates" in rows["neural_operator_surrogates"]["storage_targets"]
    assert "governance/quant_models/uncertainty_robust_control" in rows["uncertainty_robust_control"]["storage_targets"]
    assert "governance/quant_models/causal_regime_discovery" in rows["causal_regime_discovery"]["storage_targets"]
    assert "governance/quant_models/martingale_flow_pricing" in rows["martingale_flow_pricing"]["storage_targets"]


def test_sleeve_strategy_manifest_includes_advanced_quant_research_wave() -> None:
    manifest = json.loads((PROJECT_ROOT / "config" / "sleeve_strategy_expansion.json").read_text(encoding="utf-8"))
    sleeves = {str(row.get("name") or ""): row for row in manifest["sleeves"]}

    assert set(sleeves) >= {
        "neural_operator_surrogates",
        "uncertainty_robust_control",
        "causal_regime_discovery",
        "martingale_flow_pricing",
    }
    assert len(sleeves["neural_operator_surrogates"]["strategies"]) == 5
    assert len(sleeves["uncertainty_robust_control"]["strategies"]) == 5
    assert len(sleeves["causal_regime_discovery"]["strategies"]) == 5
    assert len(sleeves["martingale_flow_pricing"]["strategies"]) == 5
