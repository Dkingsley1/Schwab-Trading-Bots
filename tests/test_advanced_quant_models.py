from core import advanced_quant_models as src


def test_advanced_quant_model_features_include_expansion_keys() -> None:
    features = src.summarize_quant_model_features(
        {
            "last_price": 100.0,
            "atm_strike": 100.0,
            "expiry_days": 30,
            "implied_volatility": 0.22,
            "mom_1m": 0.001,
            "mom_5m": -0.002,
            "ctx_SPY_mom_5m": 0.001,
            "order_book_imbalance_norm": 0.25,
            "source_confidence_norm": 0.70,
        },
        external_snapshots={"macro": {"ok": True}, "news": {"ok": True}, "market_micro": {"ok": True}},
    )

    required = {
        "quant_signature_path_dna_norm",
        "quant_hawkes_self_exciting_norm",
        "quant_limit_order_book_transformer_norm",
        "quant_graph_laplacian_tda_diffusion_norm",
        "quant_agentic_self_correction_critic_norm",
        "quant_nonhomogeneous_hmm_confidence_norm",
        "quant_physics_informed_neural_sde_norm",
        "quant_double_machine_learning_causal_norm",
        "quant_neuro_symbolic_agent_norm",
        "quant_cross_modal_embedding_omni_sensor_norm",
        "quant_rlbf_backtracking_feedback_norm",
        "quant_differentiable_market_simulator_norm",
        "quant_equivariant_neural_network_norm",
        "quant_dainn_arbitrage_invariant_norm",
        "quant_markovian_execution_control_norm",
        "quant_end_to_end_diff_backtest_norm",
        "quant_portfolio_durability_norm",
        "quant_information_geometry_manifold_norm",
        "quant_graph_attention_spillover_norm",
        "quant_agentic_wallet_intent_norm",
        "quant_rough_path_signature_kernel_norm",
        "quant_quantum_classical_hybrid_optimization_norm",
        "quant_formal_verification_safety_norm",
        "quant_lobdif_order_book_diffusion_norm",
        "quant_fractional_hurst_rough_vol_norm",
        "quant_differentiable_market_impact_norm",
        "quant_persistent_homology_flash_crash_norm",
        "quant_toxic_liquidity_injection_norm",
        "quant_flash_freeze_slippage_norm",
        "quant_photonic_quantum_optimization_norm",
        "quant_replication_crisis_shield_norm",
        "quant_synthetic_crisis_market_gan_norm",
        "quant_correlation_convergence_norm",
        "quant_macro_stress_2026_driver_norm",
        "quant_fed_2026_scenario_integrity_norm",
        "quant_fed_2026_equity_crash_vol_spike_norm",
        "quant_fed_2026_credit_spread_blowout_norm",
        "quant_fed_2026_housing_price_shock_norm",
        "quant_fed_2026_cre_price_shock_norm",
        "quant_fed_2026_unemployment_recession_norm",
        "quant_fed_2026_global_recession_deflation_norm",
        "quant_fed_2026_commodity_inflation_shock_norm",
        "quant_fed_2026_treasury_yield_shock_norm",
        "quant_fed_2026_usd_stress_norm",
        "quant_fed_2026_counterparty_default_contagion_norm",
        "quant_sabr_vol_surface_norm",
        "quant_svi_ssvi_vol_surface_norm",
        "quant_dupire_local_vol_surface_norm",
        "quant_bates_jump_vol_norm",
        "quant_hull_white_rates_norm",
        "quant_cir_intensity_norm",
        "quant_hjm_forward_rate_norm",
        "quant_sofr_market_model_norm",
        "quant_dcc_garch_correlation_norm",
        "quant_evt_pot_tail_norm",
        "quant_covid_2020_pandemic_replay_norm",
        "quant_mckean_vlasov_control_norm",
        "quant_tensor_network_mps_norm",
        "quant_multifidelity_stochastic_programming_norm",
        "quant_differentiable_tatonnement_norm",
        "quant_signature_lead_lag_detector_norm",
        "quant_chaos_propagation_norm",
        "quant_mckean_vlasov_sde_sensitivity_norm",
        "quant_mlmc_sequential_estimation_norm",
        "quant_signature_volterra_kernel_calibration_norm",
        "quant_dual_tatonnement_price_discovery_norm",
        "quant_probabilistic_propagation_of_chaos_norm",
        "quant_experience_accumulation_memory_norm",
        "quant_rough_vvix_exotics_norm",
        "quant_quantum_barrier_path_amplitude_norm",
        "quant_cross_asset_correlation_heat_swap_norm",
        "quant_cliquet_global_floor_local_cap_norm",
        "quant_signature_trend_follower_options_norm",
        "quant_esg_contingent_cds_norm",
        "quant_sdg_control_norm",
        "quant_nonlocal_fractional_laplacian_norm",
        "quant_infinite_dimensional_heston_norm",
        "quant_lie_group_rough_path_signature_norm",
        "quant_mean_field_games_controls_norm",
        "quant_wasserstein_gradient_flow_norm",
        "quant_malliavin_wiener_greeks_norm",
        "quant_tqft_braid_group_norm",
        "quant_mfgc_congestion_norm",
        "quant_spde_manifold_lob_fluid_norm",
        "quant_mlx_nn_available_norm",
        "quant_mlx_optimizers_available_norm",
        "quant_mlx_lm_available_norm",
        "quant_mlx_graphs_available_norm",
        "quant_mlx_snn_available_norm",
        "quant_mlx_vision_available_norm",
        "quant_esig_signature_available_norm",
        "quant_quantlib_available_norm",
        "quant_quantlib_pricing_benchmark_norm",
        "quant_strategy_carry_edge_norm",
        "quant_strategy_mean_reversion_edge_norm",
        "quant_strategy_volatility_rv_edge_norm",
        "quant_strategy_microstructure_edge_norm",
        "quant_strategy_tail_hedge_edge_norm",
        "quant_strategy_crypto_basis_edge_norm",
        "quant_strategy_kelly_sizing_readiness_norm",
        "quant_strategy_portfolio_fit_norm",
        "quant_strategy_selection_confidence_norm",
        "quant_strategy_execution_alignment_norm",
        "quant_strategy_risk_adjusted_conviction_norm",
        "quant_strategy_allocation_bias_norm",
    }

    assert required.issubset(features)
    assert set(src.QUANT_MODEL_FEATURE_KEYS).issubset(features)
    assert all(0.0 <= float(features[key]) <= 1.0 for key in required)


def test_quant_strategy_scorecard_turns_quant_context_into_actionable_edges() -> None:
    features = src.summarize_quant_model_features(
        {
            "last_price": 42500.0,
            "atm_strike": 42500.0,
            "expiry_days": 14,
            "implied_volatility": 0.45,
            "mom_1m": -0.002,
            "mom_5m": 0.001,
            "flow_direction_signed": 0.35,
            "edge_norm": 0.72,
            "source_confidence_norm": 0.88,
            "market_micro_tradeability_score_norm": 0.90,
            "execution_fitness_norm": 0.84,
            "futures_basis_bps_norm": 0.70,
            "futures_roll_yield_norm": 0.62,
            "crypto_hyperliquid_funding_norm": 0.82,
            "crypto_basis_norm": 0.76,
            "crypto_open_interest_change_norm": 0.68,
            "crypto_cross_provider_agreement_norm": 0.81,
            "options_iv_realized_spread_norm": 0.79,
            "trend_persistence_norm": 0.28,
            "strategy_overlap_pressure_norm": 0.12,
            "cross_sleeve_correlation_pressure_norm": 0.18,
            "walk_forward_parameter_stability_norm": 0.74,
        },
        external_snapshots={
            "macro": {"ok": True},
            "news": {"ok": True},
            "market_micro": {"ok": True},
            "crypto_correlation": {"ok": True},
        },
    )

    assert features["quant_strategy_carry_edge_norm"] > 0.35
    assert features["quant_strategy_crypto_basis_edge_norm"] > 0.40
    assert features["quant_strategy_kelly_sizing_readiness_norm"] > 0.25
    assert features["quant_strategy_portfolio_fit_norm"] > 0.50
    assert features["quant_strategy_selection_confidence_norm"] > 0.35
    assert features["quant_strategy_execution_alignment_norm"] > 0.45
    assert features["quant_strategy_risk_adjusted_conviction_norm"] > 0.25
    assert features["quant_strategy_allocation_bias_norm"] > 0.35


def test_quant_model_inventory_tracks_mlx_and_new_models() -> None:
    inventory = src.quant_model_inventory()
    models = set(inventory["implemented_models"])

    assert "mlx_compile_fair_value_path" in models
    assert "quantlib_black_scholes_benchmark_proxy" in models
    assert "sabr_stochastic_alpha_beta_rho_vol_surface_proxy" in models
    assert "svi_ssvi_arbitrage_free_vol_surface_proxy" in models
    assert "dupire_local_volatility_surface_proxy" in models
    assert "bates_heston_jump_diffusion_proxy" in models
    assert "hull_white_one_factor_rates_proxy" in models
    assert "cir_short_rate_credit_intensity_proxy" in models
    assert "hjm_forward_rate_model_proxy" in models
    assert "sofr_libor_market_model_proxy" in models
    assert "dynamic_conditional_correlation_garch_proxy" in models
    assert "extreme_value_theory_peaks_over_threshold_proxy" in models
    assert "limit_order_book_transformer_lit_proxy" in models
    assert "double_machine_learning_causal_inference_proxy" in models
    assert "unified_cross_modal_embeddings_omni_sensor_proxy" in models
    assert "differentiable_market_simulator_proxy" in models
    assert "differentiable_arbitrage_invariant_neural_network_proxy" in models
    assert "formal_verification_smart_agent_safety_proxy" in models
    assert "lobdif_order_book_diffusion_proxy" in models
    assert "fed_2026_supervisory_scenario_dataset_proxy" in models
    assert "fed_2026_equity_crash_volatility_spike_proxy" in models
    assert "fed_2026_counterparty_default_contagion_proxy" in models
    assert "covid_2020_pandemic_crash_replay_proxy" in models
    assert "mckean_vlasov_master_equation_control_proxy" in models
    assert "quantum_tensor_network_matrix_product_state_proxy" in models
    assert "multifidelity_stochastic_programming_proxy" in models
    assert "differentiable_tatonnement_price_discovery_proxy" in models
    assert "signature_lead_lag_detector_proxy" in models
    assert "probabilistic_chaos_propagation_proxy" in models
    assert "mckean_vlasov_sde_sensitivity_proxy" in models
    assert "multi_level_monte_carlo_sequential_estimation_proxy" in models
    assert "signature_volterra_kernel_calibration_proxy" in models
    assert "dual_tatonnement_price_discovery_proxy" in models
    assert "probabilistic_propagation_of_chaos_proxy" in models
    assert "experience_accumulation_memory_design_proxy" in models
    assert "rough_volatility_vvix_exotics_proxy" in models
    assert "quantum_barrier_path_amplitude_option_proxy" in models
    assert "cross_asset_correlation_heat_swap_proxy" in models
    assert "cliquet_global_floor_local_cap_proxy" in models
    assert "signature_trend_follower_option_proxy" in models
    assert "esg_linked_contingent_credit_default_swap_proxy" in models
    assert "stochastic_differential_games_control_proxy" in models
    assert "nonlocal_fractional_laplacian_proxy" in models
    assert "infinite_dimensional_heston_model_proxy" in models
    assert "lie_group_rough_path_signature_proxy" in models
    assert "mean_field_games_of_controls_proxy" in models
    assert "wasserstein_gradient_flow_measure_optimization_proxy" in models
    assert "malliavin_wiener_space_infinite_dimensional_greeks_proxy" in models
    assert "topological_quantum_field_theory_braid_group_proxy" in models
    assert "mfgc_congestion_control_proxy" in models
    assert "spde_manifold_limit_order_book_fluid_proxy" in models
    assert "quant_strategy_scorecard_layer" in models
    assert "quant_strategy_risk_adjusted_conviction_router" in models
    assert {
        "mlx_core_random",
        "mx_grad",
        "mlx_compile",
        "mlx_nn",
        "mlx_optimizers",
        "mlx_lm",
        "mlx_graphs",
        "mlx_snn",
        "mlx_vision",
        "esig",
        "roughpy",
        "quantlib",
        "fair_value_gradient",
    }.issubset(inventory["mlx_hooks"])
    assert "transformer_sequence" in inventory["resource_profile"]
    assert "dml_crossfit_folds" in inventory["resource_profile"]
    assert "dainn_layers" in inventory["resource_profile"]
    assert "formal_checks" in inventory["resource_profile"]
