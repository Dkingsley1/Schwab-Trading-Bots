from core.stability_intelligence_plumbing import (
    adaptive_learning_priority_score,
    catastrophic_forgetting_risk_score,
    causal_representation_stability_score,
    event_risk_score,
    feature_confidence_score,
    governor_pressure_score,
    liquidity_regime_stress_score,
    model_calibration_decay_score,
    portfolio_conflict_score,
    simulation_to_reality_gap_score,
    transaction_cost_pressure_score,
)


def test_stability_intelligence_scores_are_bounded() -> None:
    scores = [
        model_calibration_decay_score(
            calibration_error=0.11,
            challenger_gap=0.06,
            stress_replay_pass_rate=0.72,
            leakage_risk=0.08,
        ),
        transaction_cost_pressure_score(
            spread_bps=18,
            slippage_bps=22,
            fill_rate=0.81,
            queue_adverse_selection=0.35,
        ),
        portfolio_conflict_score(
            gross_exposure=2.4,
            net_exposure=0.9,
            hedge_ratio_error=0.18,
            sleeve_correlation=0.62,
        ),
        event_risk_score(
            surprise_magnitude=0.7,
            time_to_event_minutes=90,
            source_confidence=0.82,
            historical_impact=0.55,
        ),
        liquidity_regime_stress_score(
            spread_bps=32,
            quote_fade_rate=0.4,
            auction_imbalance=0.5,
            halt_reopen_flag=True,
        ),
        governor_pressure_score(
            cpu_pressure=0.65,
            memory_pressure=0.58,
            backlog_ratio=0.42,
            halt_pressure=0.31,
            storage_pressure=0.47,
        ),
    ]

    assert all(0.0 <= score <= 1.0 for score in scores)
    assert feature_confidence_score(
        missing_rate=0.05,
        stale_rate=0.04,
        source_disagreement=0.10,
        label_confidence=0.92,
    ) > feature_confidence_score(
        missing_rate=0.40,
        stale_rate=0.35,
        source_disagreement=0.50,
        label_confidence=0.92,
    )


def test_governor_pressure_rises_with_halt_and_backlog_pressure() -> None:
    calm = governor_pressure_score(
        cpu_pressure=0.25,
        memory_pressure=0.25,
        backlog_ratio=0.10,
        halt_pressure=0.0,
        storage_pressure=0.20,
    )
    pressured = governor_pressure_score(
        cpu_pressure=0.75,
        memory_pressure=0.70,
        backlog_ratio=0.85,
        halt_pressure=0.80,
        storage_pressure=0.70,
    )

    assert pressured > calm


def test_adaptive_kernel_scores_gate_learning_under_pressure() -> None:
    low_pressure_priority = adaptive_learning_priority_score(
        uncertainty=0.75,
        drift=0.62,
        observation_value=0.80,
        runtime_pressure=0.10,
    )
    high_pressure_priority = adaptive_learning_priority_score(
        uncertainty=0.75,
        drift=0.62,
        observation_value=0.80,
        runtime_pressure=0.80,
    )
    forgetting = catastrophic_forgetting_risk_score(
        legacy_replay_drop=0.13,
        new_slice_gain=0.18,
        rehearsal_coverage=0.35,
        regime_distance=0.70,
    )
    sim_gap = simulation_to_reality_gap_score(
        paper_live_slippage_gap=38,
        replay_fill_error=0.34,
        synthetic_stress_error=0.45,
        live_context_coverage=0.55,
    )
    stable_causal = causal_representation_stability_score(
        intervention_consistency=0.82,
        feature_overlap_leakage=0.05,
        source_disagreement=0.08,
        regime_transfer_success=0.72,
    )
    leaky_causal = causal_representation_stability_score(
        intervention_consistency=0.82,
        feature_overlap_leakage=0.55,
        source_disagreement=0.40,
        regime_transfer_success=0.72,
    )

    assert low_pressure_priority > high_pressure_priority
    assert all(0.0 <= score <= 1.0 for score in [forgetting, sim_gap, stable_causal, leaky_causal])
    assert stable_causal > leaky_causal
