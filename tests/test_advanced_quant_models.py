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
    }

    assert required.issubset(features)
    assert set(src.QUANT_MODEL_FEATURE_KEYS).issubset(features)
    assert all(0.0 <= float(features[key]) <= 1.0 for key in required)


def test_quant_model_inventory_tracks_mlx_and_new_models() -> None:
    inventory = src.quant_model_inventory()
    models = set(inventory["implemented_models"])

    assert "mlx_compile_fair_value_path" in models
    assert "limit_order_book_transformer_lit_proxy" in models
    assert "double_machine_learning_causal_inference_proxy" in models
    assert "unified_cross_modal_embeddings_omni_sensor_proxy" in models
    assert "differentiable_market_simulator_proxy" in models
    assert {"mlx_core_random", "mx_grad", "mlx_compile", "fair_value_gradient"}.issubset(inventory["mlx_hooks"])
    assert "transformer_sequence" in inventory["resource_profile"]
    assert "dml_crossfit_folds" in inventory["resource_profile"]
