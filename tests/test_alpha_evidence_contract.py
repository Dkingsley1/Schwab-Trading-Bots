from __future__ import annotations

import json
from pathlib import Path

from core import accountability
from core.alpha_evidence_contract import (
    bind_candidate_identity,
    build_cross_sleeve_alpha_map,
    build_net_edge_contract,
    clear_runtime_cache,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate_root(tmp_path: Path) -> Path:
    _write_json(
        tmp_path / "config" / "alpha_generation_control_v1.json",
        {
            "primary_objective": "candidate_bound_post_cost_residual_alpha",
            "candidate_binding": {
                "state_path": "governance/runtime/production_candidate_state.json",
                "scope_names": ["strategy", "execution"],
                "cache_seconds": 0,
            },
            "net_edge": {
                "required_cost_components": [
                    "half_spread",
                    "fees",
                    "slippage",
                    "market_impact",
                ],
                "uncertainty_buffer_bps": 1.0,
                "spread_charge_fraction": 0.5,
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "runtime" / "production_candidate_state.json",
        {
            "candidate_id": "pc-test-g9",
            "generation": 9,
            "accepted_at_utc": "2026-08-20T12:00:00+00:00",
            "overall_sha256": "receipt-9",
            "scope_windows_started_utc": {
                "strategy": "2026-08-20T12:00:00+00:00",
                "execution": "2026-08-20T13:00:00+00:00",
            },
            "live_execution_authority": False,
        },
    )
    clear_runtime_cache()
    return tmp_path


def test_candidate_binding_attaches_only_after_cutoff_and_preserves_mismatch(
    tmp_path: Path,
) -> None:
    root = _candidate_root(tmp_path)
    current = bind_candidate_identity(
        {
            "timestamp_utc": "2026-08-20T13:00:01+00:00",
            "symbol": "SPY",
            "metadata": {},
        },
        project_root=root,
    )
    historical = bind_candidate_identity(
        {"timestamp_utc": "2026-08-20T12:59:59+00:00", "symbol": "SPY"},
        project_root=root,
    )
    mismatch = bind_candidate_identity(
        {
            "timestamp_utc": "2026-08-20T14:00:00+00:00",
            "production_candidate_id": "pc-other",
            "metadata": {"production_candidate_id": "pc-other"},
        },
        project_root=root,
    )

    assert current["candidate_binding"]["status"] == "candidate_bound"
    assert current["metadata"]["production_candidate_id"] == "pc-test-g9"
    assert historical["candidate_binding"]["status"] == "pre_candidate_history"
    assert "production_candidate_id" not in historical
    assert mismatch["candidate_binding"]["status"] == "candidate_identity_mismatch"
    assert mismatch["production_candidate_id"] == "pc-other"


def test_net_edge_requires_observed_costs_and_never_converts_model_score() -> None:
    policy = {
        "net_edge": {
            "required_cost_components": [
                "half_spread",
                "fees",
                "slippage",
                "market_impact",
            ],
            "uncertainty_buffer_bps": 1.0,
            "spread_charge_fraction": 0.5,
        }
    }
    complete = build_net_edge_contract(
        {
            "model_score": 0.99,
            "features": {
                "expected_gross_edge_bps": 10.0,
                "spread_bps": 2.0,
                "fee_bps": 1.0,
                "slippage_bps": 1.0,
                "market_impact_bps": 1.0,
            },
        },
        policy=policy,
    )
    incomplete = build_net_edge_contract(
        {"model_score": 0.99, "features": {"expected_gross_edge_bps": 10.0}},
        policy=policy,
    )

    assert complete["estimable"] is True
    assert complete["known_cost_bps"] == 4.0
    assert complete["conservative_net_edge_bps"] == 5.0
    assert complete["model_score_converted_to_edge"] is False
    assert incomplete["estimable"] is False
    assert incomplete["conservative_net_edge_bps"] is None
    assert incomplete["unknown_cost_defaults_used"] is False


def test_accountability_writer_adds_candidate_and_cross_sleeve_contract(
    tmp_path: Path,
) -> None:
    root = _candidate_root(tmp_path)
    path = root / "governance" / "channels" / "decision" / "dividend" / "decision.jsonl"

    wrote = accountability.safe_append_channel_batch(
        str(path),
        [
            {
                "timestamp_utc": "2026-08-20T13:10:00+00:00",
                "symbol": "SCHD",
                "profile": "dividend",
                "action": "HOLD",
                "features": {
                    "expected_gross_edge_bps": 8.0,
                    "spread_bps": 2.0,
                    "fee_bps": 0.0,
                    "slippage_bps": 1.0,
                    "market_impact_bps": 1.0,
                },
            }
        ],
        project_root=str(root),
        source="unit_test",
        channel="decision",
        schema="decision",
    )

    row = json.loads(path.read_text(encoding="utf-8"))
    assert wrote == 1
    assert row["metadata"]["production_candidate_id"] == "pc-test-g9"
    assert (
        row["alpha_evidence_contract"]["candidate_binding_status"] == "candidate_bound"
    )
    assert row["cross_sleeve_alpha_contract"]["shared_trade_logic_allowed"] is False
    assert row["alpha_evidence_contract"]["live_execution_authority"] is False


def test_cross_sleeve_map_selects_independent_residual_owners_and_caps_weights() -> (
    None
):
    sleeves = ["a", "b", "c", "d", "e"]
    series: dict[str, list[dict]] = {sleeve: [] for sleeve in sleeves}
    for day_index in range(30):
        leader = sleeves[day_index % len(sleeves)]
        for sleeve in sleeves:
            series[sleeve].append(
                {
                    "day_utc": f"202609{day_index + 1:02d}",
                    "mean_post_cost_return_bps": 20.0 if sleeve == leader else 5.0,
                }
            )

    result = build_cross_sleeve_alpha_map(
        series,
        statistically_qualified_sleeves=sleeves,
        minimum_profitable_sleeves=4,
        minimum_common_days=30,
        maximum_pairwise_correlation=0.5,
        maximum_single_sleeve_weight=0.25,
    )

    assert result["evidence_ready"] is True
    assert result["selected_sleeve_count"] == 5
    assert sum(result["research_weights"].values()) == 1.0
    assert max(result["research_weights"].values()) <= 0.25
    assert result["cash_weight"] == 0.0
    assert result["automatic_allocation_allowed"] is False
