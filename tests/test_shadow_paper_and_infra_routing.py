import scripts.run_shadow_training_loop as loop


def _bot(bot_id: str, *, role: str = "signal_sub_bot", acc: float = 0.60) -> loop.SubBot:
    return loop.SubBot(
        bot_id=bot_id,
        weight=0.10,
        active=True,
        reason="test",
        test_accuracy=acc,
        promoted=False,
        bot_role=role,
    )


def test_top_paper_mirror_bots_all_active_includes_all_non_infra_registry_bots() -> None:
    bots = [
        _bot("signal_a", role="signal_sub_bot", acc=0.52),
        _bot("options_a", role="options_sub_bot", acc=0.40),
        _bot("futures_a", role="futures_sub_bot", acc=0.41),
        _bot("infra_a", role="infrastructure_sub_bot", acc=0.99),
    ]

    selected = loop._top_paper_mirror_bots(
        bots,
        top_n=1,
        min_accuracy=0.95,
        segment="all_active",
        mirror_all_active=True,
    )

    assert {b.bot_id for b in selected} == {"signal_a", "options_a", "futures_a"}


def test_top_paper_mirror_bots_all_active_derivatives_ignore_caps() -> None:
    bots = [
        _bot("options_a", role="options_sub_bot", acc=0.30),
        _bot("options_b", role="options_sub_bot", acc=0.31),
        _bot("options_c", role="options_sub_bot", acc=0.32),
    ]

    selected = loop._top_paper_mirror_bots(
        bots,
        top_n=1,
        min_accuracy=0.99,
        segment="options",
        mirror_all_active=True,
    )

    assert {b.bot_id for b in selected} == {"options_a", "options_b", "options_c"}


def test_infrastructure_observer_kind_maps_live_registry_infra_ids() -> None:
    assert loop._infrastructure_observer_kind("brain_refinery_v59_risk_sentinel") == "risk_sentinel"
    assert loop._infrastructure_observer_kind("brain_refinery_v67_correlation_penalty_layer") == "cross_venue_divergence"
    assert loop._infrastructure_observer_kind("brain_refinery_v68_risk_budget_layer") == "risk_budget"
    assert loop._infrastructure_observer_kind("brain_refinery_v69_cost_aware_execution_filter") == "execution_feasibility"
    assert loop._infrastructure_observer_kind("brain_refinery_v80_execution_feasibility_sentinel") == "execution_feasibility"
    assert loop._infrastructure_observer_kind("brain_refinery_v86_risk_budget_allocator_v2") == "risk_budget"


def test_infrastructure_observer_signal_uses_named_risk_budget_duty() -> None:
    bot = _bot("brain_refinery_v86_risk_budget_allocator_v2", role="infrastructure_sub_bot", acc=0.65)

    action, score, threshold, reasons, meta = loop._infrastructure_observer_signal(
        bot,
        features={"mom_5m": 0.02, "pct_from_close": 0.01},
        decisions=[],
    )

    assert action in {"BUY", "SELL", "HOLD"}
    assert score >= 0.0
    assert threshold > 0.0
    assert reasons[0] == "infra_risk_budget_allocator"
    assert float(meta["confidence_scale"]) >= 0.0


def test_derive_infrastructure_aux_features_respects_risk_sentinel_veto() -> None:
    out = loop._derive_infrastructure_aux_features(
        [
            {
                "bot_id": "brain_refinery_v59_risk_sentinel",
                "observer_meta": {"vote": -0.70, "risk": 0.93},
                "direction": -1.0,
                "weight": 0.10,
            }
        ]
    )

    assert out["infra_risk_throttle_norm"] >= 0.93
    assert out["infra_veto_active"] == 1.0
