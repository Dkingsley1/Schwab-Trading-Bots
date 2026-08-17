from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import scripts.run_shadow_training_loop as loop


def _bot(
    bot_id: str,
    *,
    role: str = "signal_sub_bot",
    acc: float = 0.60,
    paper_live_data_enabled: bool = False,
    paper_execution_authority: bool = False,
    paper_probation_authority: bool = False,
    quality_score: float = 0.60,
) -> loop.SubBot:
    return loop.SubBot(
        bot_id=bot_id,
        weight=0.10,
        active=True,
        reason="test",
        test_accuracy=acc,
        promoted=False,
        bot_role=role,
        paper_live_data_enabled=paper_live_data_enabled,
        paper_execution_authority=paper_execution_authority,
        paper_probation_authority=paper_probation_authority,
        quality_score=quality_score,
    )


def test_top_paper_mirror_bots_ignores_legacy_all_active_bypass() -> None:
    bots = [
        _bot("signal_a", role="signal_sub_bot", acc=0.62, paper_execution_authority=True),
        _bot("options_a", role="options_sub_bot", acc=0.64, paper_execution_authority=True),
        _bot("futures_a", role="futures_sub_bot", acc=0.63, paper_execution_authority=True),
        _bot("infra_a", role="infrastructure_sub_bot", acc=0.99, paper_execution_authority=True),
        _bot("signal_without_authority", role="signal_sub_bot", acc=0.99),
    ]

    selected = loop._top_paper_mirror_bots(
        bots,
        top_n=1,
        min_accuracy=0.56,
        segment="all_active",
        mirror_all_active=True,
    )

    assert [b.bot_id for b in selected] == ["signal_a"]


def test_top_paper_mirror_bots_derivatives_respect_hard_cap(monkeypatch) -> None:
    monkeypatch.setenv("PAPER_EXECUTION_COHORT_MAX_PER_SEGMENT", "2")
    bots = [
        _bot("options_a", role="options_sub_bot", acc=0.60, paper_execution_authority=True),
        _bot("options_b", role="options_sub_bot", acc=0.61, paper_execution_authority=True),
        _bot("options_c", role="options_sub_bot", acc=0.62, paper_execution_authority=True),
    ]

    selected = loop._top_paper_mirror_bots(
        bots,
        top_n=3,
        min_accuracy=0.56,
        segment="options",
        mirror_all_active=True,
    )

    assert [b.bot_id for b in selected] == ["options_c", "options_b"]


def test_top_paper_mirror_bots_standard_filters_to_explicit_paper_cohort(monkeypatch) -> None:
    monkeypatch.setenv("PAPER_LIVE_DATA_STANDARD_ENABLED", "1")
    bots = [
        _bot("collector_high_score", acc=0.99, paper_live_data_enabled=False),
        _bot(
            "paper_allowed_lower_score",
            acc=0.61,
            paper_live_data_enabled=True,
            paper_execution_authority=True,
        ),
        _bot("legacy_flag_without_authority", acc=0.99, paper_live_data_enabled=True),
    ]

    selected = loop._top_paper_mirror_bots(
        bots,
        top_n=2,
        min_accuracy=0.0,
        segment="core",
        mirror_all_active=False,
    )

    assert [b.bot_id for b in selected] == ["paper_allowed_lower_score"]


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


def test_consensus_entry_economics_is_conservative_and_not_promotion_evidence() -> None:
    result = loop._derive_consensus_entry_economics(
        {
            "score": 0.82,
            "consensus_ratio": 0.90,
            "net_vote_ratio": 0.80,
            "distinct_correlation_clusters": 3,
        },
        {
            "execution_fitness_norm": 0.90,
            "spread_bps": 2.0,
            "expected_slippage_bps": 1.0,
        },
    )

    assert result["predicted_edge_lower_confidence_bound_bps"] > 0.0
    assert result["expected_round_trip_cost_bps"] == 6.0
    assert result["predicted_edge_is_promotion_evidence"] is False


def test_runtime_market_session_blocks_unvalidated_extended_hours() -> None:
    schwab = loop._runtime_market_session_features(
        "schwab",
        now_utc=datetime(2026, 8, 17, 1, 0, tzinfo=timezone.utc),
    )
    coinbase = loop._runtime_market_session_features("coinbase")

    assert schwab["market_session"] == "overnight"
    assert schwab["extended_session_validated"] is False
    assert coinbase["market_session"] == "continuous_24x7"
    assert coinbase["extended_session_validated"] is True


def test_runtime_registry_overlay_requires_hash_bound_identical_membership(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "master_bot_registry.json"
    candidate_path = tmp_path / "governance" / "health" / "paper_live_data_standard_registry_candidate_latest.json"
    guard_path = tmp_path / "governance" / "health" / "paper_live_data_standard_source_write_guard_latest.json"
    health_path = tmp_path / "governance" / "health" / "paper_live_data_standard_latest.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    source = {"summary": {}, "sub_bots": [{"bot_id": "signal_a", "paper_execution_authority": False}]}
    candidate = {
        "summary": {"paper_live_data_standard_version": "paper_live_data_standard_v2"},
        "sub_bots": [{"bot_id": "signal_a", "paper_execution_authority": True}],
    }
    source_path.write_text(json.dumps(source), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    guard_path.write_text(
        json.dumps(
            {
                "source_write_blocked": True,
                "candidate_path": str(candidate_path),
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    health_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    monkeypatch.setenv("PAPER_LIVE_DATA_STANDARD_ENABLED", "1")
    monkeypatch.setattr(loop, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(loop, "PROJECT_ROOT_PATH", Path(tmp_path))

    loaded = loop._load_registry(str(source_path))

    assert loaded["sub_bots"][0]["paper_execution_authority"] is True
