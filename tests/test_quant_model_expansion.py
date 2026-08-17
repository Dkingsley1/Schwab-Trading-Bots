import json
from pathlib import Path

from scripts.ops import roster_expansion_slots as slots
import scripts.run_all_sleeves as run_all_sleeves
import scripts.run_specialized_sleeve_shadow as specialized


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_new_quant_sleeves_are_registered_for_collection() -> None:
    config = json.loads((PROJECT_ROOT / "config" / "sleeve_strategy_expansion.json").read_text(encoding="utf-8"))
    sleeve_names = {row["name"] for row in config["sleeves"]}

    for sleeve in {
        "signature_hawkes_generators",
        "crowd_physics_games",
        "lit_order_book_transformers",
        "critic_hmm_pinsde",
        "causal_omni_symbolic",
        "rlbf_dms_equivariant",
        "arbitrage_execution_safety",
        "geometry_spillover_durability",
        "institutional_data_plumbing",
        "lobdif_crisis_microstructure",
        "macro_crisis_scenario_lab",
        "xva_counterparty_margin",
        "credit_derivatives_cdx_cds",
        "securitized_products_mbs_abs_clo",
        "repo_securities_lending",
        "market_data_tape_normalization",
        "provider_adapter_verification",
        "proof_quantum_formal_backends",
        "model_risk_validation",
        "transaction_cost_slippage_intelligence",
        "portfolio_construction",
        "event_intelligence",
        "feature_quality_data_confidence",
        "liquidity_regime",
        "system_governor_expansion",
    }:
        assert sleeve in sleeve_names
        assert sleeve in config["ticker_universes"]
        assert sleeve in specialized.SLEEVE_DEFAULTS
        assert sleeve in run_all_sleeves.SPECIALIZED_SLEEVE_PROFILES
        assert specialized.SLEEVE_DEFAULTS[sleeve]["domain"] == "quant_models"

    assert specialized.SLEEVE_DEFAULTS["institutional_data_plumbing"]["source_gated"] == "1"
    assert int(specialized.SLEEVE_DEFAULTS["institutional_data_plumbing"]["min_interval"]) >= 600
    assert int(specialized.SLEEVE_DEFAULTS["lobdif_crisis_microstructure"]["min_interval"]) >= 300
    assert int(specialized.SLEEVE_DEFAULTS["macro_crisis_scenario_lab"]["min_interval"]) >= 600


def test_quant_roster_slots_are_research_only_and_labeled() -> None:
    by_id = {row["bot_id"]: slots._slot_registry_row(row) for row in slots.DEFAULT_SLOT_SPECS}

    for bot_id in {
        "brain_refinery_v488_signature_hawkes_games_regression_guard_bot",
        "brain_refinery_v496_order_book_transformer_resource_guard_bot",
        "brain_refinery_v497_agentic_quant_memory_guard_bot",
        "brain_refinery_v504_causal_omni_symbolic_regression_guard_bot",
        "brain_refinery_v505_rlbf_dms_equivariant_resource_guard_bot",
        "brain_refinery_v510_formal_verification_smart_agent_safety_guard_bot",
        "brain_refinery_v516_arbitrage_execution_safety_regression_guard_bot",
        "brain_refinery_v517_geometry_spillover_durability_resource_guard_bot",
        "brain_refinery_v522_event_driven_flink_mlx_pipeline_guard_bot",
        "brain_refinery_v523_feature_store_symmetry_feast_tecton_guard_bot",
        "brain_refinery_v528_toxic_liquidity_vpin_stress_injector_bot",
        "brain_refinery_v529_flash_freeze_slippage_model_bot",
        "brain_refinery_v531_replication_crisis_shield_bot",
        "brain_refinery_v535_fed_2026_adverse_scenario_dataset_guard_bot",
        "brain_refinery_v540_xva_counterparty_margin_regression_guard_bot",
        "brain_refinery_v545_credit_derivatives_cdx_cds_regression_guard_bot",
        "brain_refinery_v550_securitized_products_cashflow_regression_guard_bot",
        "brain_refinery_v555_repo_securities_lending_regression_guard_bot",
        "brain_refinery_v560_tape_normalization_regression_guard_bot",
        "brain_refinery_v565_provider_adapter_verification_regression_guard_bot",
        "brain_refinery_v570_proof_quantum_formal_regression_guard_bot",
        "brain_refinery_v575_model_risk_validation_regression_guard_bot",
        "brain_refinery_v580_transaction_cost_slippage_regression_guard_bot",
        "brain_refinery_v585_portfolio_construction_regression_guard_bot",
        "brain_refinery_v590_event_intelligence_geopolitical_shock_score_guard_bot",
        "brain_refinery_v595_feature_quality_data_confidence_regression_guard_bot",
        "brain_refinery_v600_liquidity_regime_regression_guard_bot",
        "brain_refinery_v605_system_governor_expansion_regression_guard_bot",
    }:
        row = by_id[bot_id]
        assert row["active"] is True
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["direct_execution_allowed"] is False
        assert row["execution_policy_label"] == "research_only_no_execution"
        assert row["eligible_for_master_vote"] is False
        assert row["label_contract"]["contract_version"] == "quant_research_labels_v1"


def test_institutional_data_plumbing_is_credential_gated() -> None:
    by_id = {row["bot_id"]: slots._slot_registry_row(row) for row in slots.DEFAULT_SLOT_SPECS}
    row = by_id["brain_refinery_v520_sentiment_echo_bloomberg_lunarcrush_collector_bot"]

    assert row["sleeve_profile"] == "institutional_data_plumbing"
    assert row["source_credential_gated"] is True
    assert row["provider_capability_profile"] == "research_only_source_gated_institutional_plumbing"
    assert "vendor_rate_limit_state" in row["data_intake_collections"]
    assert "arxiv_qfin_recent_research_intake" in row["data_intake_collections"]
    assert "ssrn_market_infrastructure_reference" in row["data_intake_collections"]
    assert "governance/quant_models/institutional_data_plumbing" in row["storage_targets"]
    assert row["direct_execution_allowed"] is False
    assert "https://arxiv.org/list/q-fin/recent" in row["public_source_urls"]
    assert "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3122051" in row["public_source_urls"]


def test_advanced_gap_expansion_is_source_gated_and_collection_only() -> None:
    by_id = {row["bot_id"]: slots._slot_registry_row(row) for row in slots.DEFAULT_SLOT_SPECS}
    expected = {
        "brain_refinery_v536_xva_cva_dva_exposure_ladder_bot": "xva_counterparty_margin",
        "brain_refinery_v541_credit_derivatives_cdx_itraxx_basis_bot": "credit_derivatives_cdx_cds",
        "brain_refinery_v546_securitized_products_mbs_prepayment_oas_bot": "securitized_products_mbs_abs_clo",
        "brain_refinery_v551_repo_sofr_funding_pressure_bot": "repo_securities_lending",
        "brain_refinery_v556_tape_opra_nbbo_alignment_bot": "market_data_tape_normalization",
        "brain_refinery_v561_provider_credential_readiness_guard_bot": "provider_adapter_verification",
        "brain_refinery_v566_proof_zkp_privacy_backend_guard_bot": "proof_quantum_formal_backends",
    }

    for bot_id, sleeve in expected.items():
        row = by_id[bot_id]
        assert row["sleeve_profile"] == sleeve
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["direct_execution_allowed"] is False
        assert row["source_credential_gated"] is True
        assert row["provider_capability_profile"].startswith("research_only_")
        assert row["label_contract"]["contract_version"] == "quant_research_labels_v1"


def test_stability_intelligence_expansion_is_collection_only() -> None:
    by_id = {row["bot_id"]: slots._slot_registry_row(row) for row in slots.DEFAULT_SLOT_SPECS}
    expected = {
        "brain_refinery_v571_model_risk_calibration_decay_sentinel_bot": "model_risk_validation",
        "brain_refinery_v576_transaction_cost_spread_decay_fill_realism_bot": "transaction_cost_slippage_intelligence",
        "brain_refinery_v581_portfolio_construction_exposure_netting_bot": "portfolio_construction",
        "brain_refinery_v586_event_intelligence_fed_speaker_surprise_bot": "event_intelligence",
        "brain_refinery_v591_feature_quality_missing_data_penalty_bot": "feature_quality_data_confidence",
        "brain_refinery_v596_liquidity_regime_open_close_auction_imbalance_bot": "liquidity_regime",
        "brain_refinery_v601_system_governor_collector_priority_ranker_bot": "system_governor_expansion",
    }

    for bot_id, sleeve in expected.items():
        row = by_id[bot_id]
        assert row["sleeve_profile"] == sleeve
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["direct_execution_allowed"] is False
        assert row["provider_capability_profile"].startswith("research_only_")
        assert row["label_contract"]["contract_version"] == "quant_research_labels_v1"


def test_fed_2026_scenario_dataset_is_registered() -> None:
    scenario_path = PROJECT_ROOT / "config" / "stress_scenarios" / "fed_2026_supervisory_severely_adverse.json"
    modules_path = PROJECT_ROOT / "config" / "stress_scenarios" / "fed_2026_stress_modules.json"
    plumbing_path = PROJECT_ROOT / "config" / "stress_scenarios" / "fed_2026_source_plumbing.json"
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    modules = json.loads(modules_path.read_text(encoding="utf-8"))
    plumbing = json.loads(plumbing_path.read_text(encoding="utf-8"))
    by_id = {row["bot_id"]: slots._slot_registry_row(row) for row in slots.DEFAULT_SLOT_SPECS}
    row = by_id["brain_refinery_v535_fed_2026_adverse_scenario_dataset_guard_bot"]
    module_ids = {item["module_id"] for item in modules["stress_modules"]}
    module_bot_ids = {
        "brain_refinery_v654_fed_2026_equity_crash_vol_spike_bot",
        "brain_refinery_v655_fed_2026_credit_spread_blowout_bot",
        "brain_refinery_v656_fed_2026_housing_price_shock_bot",
        "brain_refinery_v657_fed_2026_cre_price_shock_bot",
        "brain_refinery_v658_fed_2026_unemployment_recession_bot",
        "brain_refinery_v659_fed_2026_global_recession_deflation_bot",
        "brain_refinery_v660_fed_2026_commodity_inflation_shock_bot",
        "brain_refinery_v661_fed_2026_treasury_yield_shock_bot",
        "brain_refinery_v662_fed_2026_usd_stress_bot",
        "brain_refinery_v663_fed_2026_counterparty_default_contagion_bot",
    }

    assert payload["scenario_id"] == "fed_2026_supervisory_severely_adverse"
    assert payload["scope"]["start_quarter"] == "2026-Q1"
    assert payload["scope"]["end_quarter"] == "2029-Q1"
    assert payload["key_stress_anchors"]["unemployment_peak_pct"] == 10.0
    assert payload["domestic_variables"]["rows"][1][-1] == 72.0
    assert modules["module_count"] == 10
    assert "fed_2026_equity_crash_volatility_spike" in module_ids
    assert "fed_2026_counterparty_default_contagion_shock" in module_ids
    assert len(plumbing["stress_module_map"]) == 10
    assert "fed_2026_stress_module_map" in plumbing["internal_feature_keys"]
    assert row["provider_capability_profile"] == "research_only_public_macro_stress_dataset"
    assert row["direct_execution_allowed"] is False
    assert "https://www.federalreserve.gov/publications/2026-stress-test-scenarios.htm" in row["public_source_urls"]
    for bot_id in module_bot_ids:
        module_row = by_id[bot_id]
        assert module_row["sleeve_profile"] == "macro_crisis_scenario_lab"
        assert module_row["lifecycle_state"] == "data_collection_only"
        assert module_row["training_excluded"] is True
        assert module_row["direct_execution_allowed"] is False
        assert "fed_2026_stress_module_map" in module_row["data_intake_collections"]


def test_stochastic_and_deep_quant_expansion_is_collection_only() -> None:
    by_id = {row["bot_id"]: slots._slot_registry_row(row) for row in slots.DEFAULT_SLOT_SPECS}
    expected = {
        "brain_refinery_v606_mckean_vlasov_master_equation_control_bot": "crowd_physics_games",
        "brain_refinery_v607_quantum_tensor_network_mps_bot": "geometry_spillover_durability",
        "brain_refinery_v608_multifidelity_stochastic_programming_bot": "geometry_spillover_durability",
        "brain_refinery_v609_differentiable_tatonnement_price_discovery_bot": "arbitrage_execution_safety",
        "brain_refinery_v610_signature_lead_lag_detector_bot": "signature_hawkes_generators",
        "brain_refinery_v611_probabilistic_chaos_propagation_bot": "crowd_physics_games",
        "brain_refinery_v612_mckean_vlasov_sde_sensitivity_bot": "crowd_physics_games",
        "brain_refinery_v613_mlmc_sequential_estimation_bot": "geometry_spillover_durability",
        "brain_refinery_v614_signature_volterra_kernel_calibration_bot": "signature_hawkes_generators",
        "brain_refinery_v615_dual_tatonnement_price_discovery_bot": "arbitrage_execution_safety",
        "brain_refinery_v616_probabilistic_propagation_of_chaos_bot": "crowd_physics_games",
        "brain_refinery_v617_experience_accumulation_memory_design_bot": "critic_hmm_pinsde",
        "brain_refinery_v624_stochastic_differential_games_control_bot": "crowd_physics_games",
        "brain_refinery_v625_nonlocal_fractional_laplacian_bot": "transport_topology_research",
        "brain_refinery_v626_infinite_dimensional_heston_bot": "quant_pricing_models",
        "brain_refinery_v627_lie_group_rough_path_signature_bot": "signature_hawkes_generators",
        "brain_refinery_v628_mean_field_games_controls_bot": "crowd_physics_games",
        "brain_refinery_v629_wasserstein_gradient_flow_bot": "transport_topology_research",
        "brain_refinery_v630_malliavin_wiener_greeks_bot": "quant_pricing_models",
        "brain_refinery_v631_tqft_braid_group_topology_bot": "transport_topology_research",
        "brain_refinery_v632_mfgc_congestion_bot": "crowd_physics_games",
        "brain_refinery_v633_spde_manifold_lob_fluid_bot": "lit_order_book_transformers",
        "brain_refinery_v664_sabr_vol_surface_bot": "quant_pricing_models",
        "brain_refinery_v665_svi_ssvi_vol_surface_bot": "quant_pricing_models",
        "brain_refinery_v666_dupire_local_vol_surface_bot": "quant_pricing_models",
        "brain_refinery_v667_bates_heston_jump_diffusion_bot": "quant_pricing_models",
        "brain_refinery_v668_hull_white_rates_model_bot": "state_space_models",
        "brain_refinery_v669_cir_intensity_credit_rates_bot": "state_space_models",
        "brain_refinery_v670_hjm_forward_rate_model_bot": "state_space_models",
        "brain_refinery_v671_sofr_market_model_bot": "state_space_models",
        "brain_refinery_v672_dcc_garch_correlation_bot": "tail_dependency_risk",
        "brain_refinery_v673_evt_peaks_over_threshold_tail_bot": "tail_dependency_risk",
    }

    for bot_id, sleeve in expected.items():
        row = by_id[bot_id]
        assert row["sleeve_profile"] == sleeve
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["direct_execution_allowed"] is False
        assert row["label_contract"]["contract_version"] == "quant_research_labels_v1"
        assert "quant_model_feature_surface" in row["data_intake_collections"]
        assert "quantlib_pricing_benchmark" in row["data_intake_collections"]


def test_deep_exotic_expansion_is_collection_only() -> None:
    by_id = {row["bot_id"]: slots._slot_registry_row(row) for row in slots.DEFAULT_SLOT_SPECS}
    expected = {
        "brain_refinery_v618_rough_vvix_exotics_bot": "variance_volatility_swaps",
        "brain_refinery_v619_quantum_barrier_path_amplitude_options_bot": "barrier_lookback_options",
        "brain_refinery_v620_cross_asset_correlation_heat_swaps_bot": "dispersion_trading",
        "brain_refinery_v621_cliquet_global_floor_local_cap_bot": "structured_products",
        "brain_refinery_v622_signature_trend_follower_options_bot": "rainbow_options",
        "brain_refinery_v623_esg_linked_ccds_bot": "synthetic_cdo",
    }

    for bot_id, sleeve in expected.items():
        row = by_id[bot_id]
        assert row["sleeve_profile"] == sleeve
        assert row["sleeve_family"] == "exotic_derivatives"
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["direct_execution_allowed"] is False
        assert "cross_sleeve_correlation_matrix" in row["data_intake_collections"]
