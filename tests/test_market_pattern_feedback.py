from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import market_pattern_feedback as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_pattern_project(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config" / "market_pattern_observation_v1.json",
        {
            "schema_version": 1,
            "policy_id": "market_pattern_observation_v1",
            "source_artifacts": {
                "regime_control": "governance/health/regime_control_plane_latest.json",
                "market_cycle": "governance/health/market_cycle_state_latest.json",
                "market_move": "governance/health/market_move_explainer_latest.json",
                "paper_profitability": "governance/health/paper_profitability_control_latest.json",
                "independent_fills": "governance/health/independent_fill_evidence_acquisition_latest.json",
                "bot_profitability": "governance/health/bot_profitability_scalability_latest.json",
            },
            "observable_dimensions": [
                {
                    "dimension": "risk_appetite",
                    "feature_keys": ["risk_on_norm", "risk_off_norm", "hold_ratio"],
                    "source_ids": ["market_cycle", "regime_control"],
                    "sleeve_uses": ["equity_core", "conservative"],
                },
                {
                    "dimension": "post_cost_evidence_feedback",
                    "feature_keys": ["candidate_post_cost_sample_count"],
                    "source_ids": ["paper_profitability", "independent_fills"],
                    "sleeve_uses": ["all_trading_sleeves"],
                },
            ],
            "pattern_response": {
                "defensive_high_vol_chop": {
                    "boost": ["volatility", "conservative"],
                    "downshift": ["intraday_aggressive", "swing_aggressive"],
                    "collection_focus": [
                        "volatility_regime_fit",
                        "spread_and_slippage",
                    ],
                },
                "mixed_transition": {
                    "boost": ["conservative", "volatility"],
                    "downshift": ["intraday_aggressive"],
                    "collection_focus": ["regime_transition_label"],
                },
                "system_hold_consensus": {
                    "boost": ["market_making_liquidity", "pairs_correlation"],
                    "downshift": ["swing_aggressive"],
                    "collection_focus": ["why_hold", "threshold_distance"],
                },
                "source_context_available": {
                    "context": ["all_trading_sleeves"],
                    "collection_focus": ["source_receipt", "feature_backfill"],
                },
                "symbol_specific_evidence_gap": {
                    "context": ["all_trading_sleeves"],
                    "collection_focus": ["symbol_level_attribution_rows"],
                },
            },
        },
    )
    _write_json(
        tmp_path / "config" / "paper_evidence_collection_controls_v1.json",
        {
            "schema_version": 1,
            "enabled": True,
            "paper_only": True,
            "live_execution_allowed": False,
            "defaults": {
                "minimum_model_score_edge_over_threshold": 0.02,
                "max_entries_per_symbol_day": 8,
                "new_entry_cooldown_seconds": 180,
            },
            "profiles": {
                "volatility": {
                    "max_entries_per_symbol_day": 8,
                    "new_entry_cooldown_seconds": 180,
                }
            },
        },
    )
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "regime_control_plane_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "regime_state": "mixed_transition",
            "stance_label": "neutral",
            "stance_score": 0.10,
        },
    )
    _write_json(
        health / "market_cycle_state_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "cycle_phase": "defensive_high_vol_chop",
            "market_regime": "high_vol_chop",
            "confidence": 0.70,
            "aggregate_signals": {
                "risk_on_norm": 0.01,
                "risk_off_norm": 0.21,
                "defensive_rotation_norm": 0.20,
                "trend_confirmation_norm": 0.12,
                "stress_norm": 0.89,
                "hold_ratio": 1.0,
                "buy_ratio": 0.0,
                "sell_ratio": 0.0,
            },
        },
    )
    _write_json(
        health / "market_move_explainer_latest.json",
        {
            "overall_status": "thin",
            "symbol": "BTC",
            "primary_readout": "insufficient symbol-specific evidence",
            "primary_confidence": 0.64,
            "ranked_drivers": [],
            "unknowns": ["no_recent_btc_shadow_attribution_rows"],
        },
    )
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "overall_status": "ready",
            "paper_debt_recovery_contract": {
                "state": "fresh_forward_collecting",
                "remaining_debt_amount": 0.0,
                "candidate_proof": {
                    "sample_count": 0,
                    "minimum_samples": 30,
                    "observed_days": 0,
                    "minimum_observed_days": 3,
                    "positive_post_cost_lower_confidence_bound_95": False,
                },
                "promotion_blockers": ["candidate_post_cost_samples_below_floor"],
            },
        },
    )
    _write_json(
        health / "independent_fill_evidence_acquisition_latest.json",
        {
            "ok": True,
            "overall_status": "waiting_for_source",
            "candidate_eligible_ledger_records": 0,
            "trade_log_materialization": {"ok": True, "status": "ready"},
        },
    )
    _write_json(
        health / "bot_profitability_scalability_latest.json",
        {
            "overall_status": "waiting_for_evidence",
            "profitability_diagnosis": {
                "paper_balance_recovery_plan": {
                    "sample_count": 0,
                    "min_post_cost_samples": 30,
                    "observed_days": 0,
                    "min_observed_days": 3,
                }
            },
        },
    )


def test_market_pattern_feedback_reports_patterns_and_sleeve_context(
    tmp_path: Path,
) -> None:
    _seed_pattern_project(tmp_path)

    payload = src.build_payload(
        tmp_path,
        config_path=tmp_path / "config" / "market_pattern_observation_v1.json",
        paper_collection_path=tmp_path
        / "config"
        / "paper_evidence_collection_controls_v1.json",
    )

    pattern_ids = {row["pattern_id"] for row in payload["patterns"]}
    assert payload["overall_status"] == "ready"
    assert "defensive_high_vol_chop" in pattern_ids
    assert "mixed_transition" in pattern_ids
    assert "system_hold_consensus" in pattern_ids
    assert "source_context_available" in pattern_ids
    assert "symbol_specific_evidence_gap" in pattern_ids
    assert payload["observable_dimension_count"] == 2
    assert payload["platform_feedback_contract"]["can_change_live_execution"] is False

    sleeves = {row["sleeve"]: row for row in payload["sleeve_feedback"]}
    assert (
        sleeves["volatility"]["paper_sampling_posture"]
        == "prioritize_bounded_paper_sampling"
    )
    assert (
        sleeves["intraday_aggressive"]["paper_sampling_posture"]
        == "downshift_or_context_first"
    )
    assert (
        "symbol_specific_evidence_gap" in sleeves["volatility"]["context_pattern_ids"]
    )
    assert sleeves["volatility"]["paper_collection_profile"]["paper_only"] is True
    assert (
        payload["profitability_evidence_gaps"]["candidate_post_cost_sample_count"] == 0
    )
    assert (
        "record symbol-level driver features with each paper intent"
        in payload["recommended_actions"]
    )


def test_market_pattern_feedback_writes_health_platform_and_history(
    tmp_path: Path,
) -> None:
    _seed_pattern_project(tmp_path)
    out = tmp_path / "governance" / "health" / "market_pattern_feedback_latest.json"
    platform = (
        tmp_path
        / "governance"
        / "platform_intelligence"
        / "market_pattern_feedback_latest.json"
    )
    history = tmp_path / "governance" / "market_patterns" / "feedback_history.jsonl"

    rc = src.main(
        [
            "--project-root",
            str(tmp_path),
            "--config-file",
            str(tmp_path / "config" / "market_pattern_observation_v1.json"),
            "--paper-collection-file",
            str(tmp_path / "config" / "paper_evidence_collection_controls_v1.json"),
            "--out-file",
            str(out),
            "--platform-out-file",
            str(platform),
            "--history-file",
            str(history),
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["paper_only"] is True
    assert (
        json.loads(platform.read_text(encoding="utf-8"))["pattern_count"]
        == payload["pattern_count"]
    )
    rows = [
        json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[-1]["pattern_count"] == payload["pattern_count"]
