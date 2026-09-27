from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops.alpha_concept_input_materializer import build_payload

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _row(index: int, *, candidate_id: str = "pc-test-g1") -> dict:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=index)
    score = 0.2 + (index % 7) * 0.1
    realized = (score - 0.5) * 0.002 + ((index % 3) - 1) * 0.00001
    action = "BUY" if index % 9 == 0 else "HOLD"
    cost_bps = 4.0 if action == "BUY" else 0.0
    gross = realized if action == "BUY" else None
    post_cost = gross - cost_bps / 10_000.0 if gross is not None else realized
    strategy = f"sleeve::test::strategy_{index % 2}::v1"
    return {
        "id": index,
        "timestamp_utc": timestamp.isoformat(),
        "symbol": ["SPY", "QQQ", "IWM"][index % 3],
        "action": action,
        "quantity": 1.0 if action == "BUY" else 0.0,
        "production_candidate_id": candidate_id,
        "production_candidate_generation": 1,
        "candidate_binding": {
            "candidate_bound": True,
            "observed_candidate_id": candidate_id,
        },
        "decision": "BLOCK" if action == "HOLD" else "ALLOW",
        "decision_id": f"decision-{index}",
        "profile": "test_sleeve",
        "sleeve_id": "test_sleeve",
        "selected_strategy_id": strategy,
        "source_strategy": "grand_master_bot",
        "model_score": score,
        "signed_forecast_score": 2.0 * score - 1.0,
        "log_schema_version": 2,
        "schema_valid": True,
        "regime": "trend" if index % 2 else "mean_revert",
        "forward_return": realized,
        "forward_return_primary": realized,
        "forward_return_aux": realized * 1.2,
        "gross_directional_forward_return": gross,
        "post_cost_forward_return": post_cost,
        "round_trip_cost_bps": cost_bps,
        "post_cost_label": action == "BUY",
        "horizon_seconds": 300,
        "aux_horizon_seconds": 900,
        "counterfactual_action_outcomes": {
            "5m": {
                "raw_market_return": realized,
                "buy_post_cost_return": realized - 0.0004,
                "sell_post_cost_return": -realized - 0.0004,
                "hold_return": 0.0,
            }
        },
        "measurement_context": {
            "last_price": 100.0 + index * 0.01,
            "pct_from_close": realized,
            "mom_5m": realized * 0.5,
            "vol_30m": 0.001 + (index % 5) * 0.0001,
            "spread_bps": 2.0 + index % 2,
            "market_micro_relative_volume_norm": 0.5,
            "market_micro_order_flow_imbalance_norm": score,
            "ctx_SPY_pct_from_close": realized * 0.8,
            "ctx_QQQ_pct_from_close": realized * 0.9,
            "ctx_TLT_pct_from_close": -realized * 0.2,
            "ctx_UUP_pct_from_close": -realized * 0.1,
        },
        "features": [],
    }


def _project(tmp_path: Path) -> Path:
    policy = json.loads(
        (
            PROJECT_ROOT / "config" / "alpha_measurement_materialization_v1.json"
        ).read_text(encoding="utf-8")
    )
    _write(tmp_path / "config" / "alpha_measurement_materialization_v1.json", policy)
    _write(
        tmp_path / "governance" / "runtime" / "production_candidate_state.json",
        {
            "candidate_id": "pc-test-g1",
            "generation": 1,
            "accepted_at_utc": "2026-01-01T00:00:00+00:00",
        },
    )
    _write(
        tmp_path / "governance" / "health" / "paper_performance_latest.json",
        {
            "profitability_evidence_window": {
                "candidate_id": "pc-test-g1",
                "candidate_generation": 1,
                "candidate_cutoff_utc": "2026-01-01T00:00:00+00:00",
                "evidence_through_utc": "2026-01-07T00:00:00+00:00",
                "candidate_filter_active": True,
                "candidate_binding_required": True,
                "candidate_binding_mismatch_rows_excluded": 0,
            }
        },
    )
    rows = [_row(index) for index in range(144)]
    rows.append(_row(145, candidate_id="pc-prior"))
    _write(
        tmp_path / "data" / "trade_history" / "trade_learning_dataset.json",
        {
            "schema": "behavior_dataset_v8_candidate_bound_multi_horizon_counterfactuals",
            "feature_names": [],
            "data": rows,
        },
    )
    return tmp_path


def test_materializer_keeps_only_exact_post_cutoff_schema_v2_candidate_rows(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)

    payload = build_payload(
        root,
        config_path=root / "config" / "alpha_measurement_materialization_v1.json",
        generated_at_utc="2026-01-07T01:00:00+00:00",
    )
    diagnostics = payload["diagnostics"]
    schema = diagnostics["schema_v2_candidate_validation"]
    trades = diagnostics["post_cost_trade_delta_validation"]

    assert payload["candidate_id"] == "pc-test-g1"
    assert schema["candidate_row_count"] == 144
    assert schema["exclusions"]["candidate_identity_mismatch"] == 1
    assert schema["historical_rows_relabelled"] == 0
    assert schema["cross_candidate_rows_pooled"] == 0
    assert trades["valid_schema_v2_post_cost_trade_delta_count"] == 16
    assert trades["hold_rows_count_as_realized_trade_pnl"] is False


def test_counterfactual_forecasts_do_not_become_economic_profit_evidence(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    payload = build_payload(
        root,
        config_path=root / "config" / "alpha_measurement_materialization_v1.json",
    )

    ic = payload["measurements"]["information_coefficient_term_structure"]
    economic = payload["measurements"]["factor_neutral_residualization"]
    capture = payload["diagnostics"]["counterfactual_capture"]
    observations = ic["inputs"]["observations"]

    assert len(observations) > 144
    assert len(observations) == len(
        {(row["decision_id"], row["horizon"]) for row in observations}
    )
    assert ic["economic_grade_eligible"] is False
    assert economic["economic_grade_eligible"] is True
    assert len(economic["inputs"]["target_returns"]) == 16
    assert capture["hold_decision_count"] == 128
    assert capture["counts_as_realized_trade_profitability"] is False


def test_walk_forward_is_purged_embargoed_and_allocation_has_no_authority(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    payload = build_payload(
        root,
        config_path=root / "config" / "alpha_measurement_materialization_v1.json",
    )
    walk = payload["diagnostics"]["purged_embargoed_walk_forward"]
    allocation = payload["diagnostics"]["cross_sleeve_allocation_gate"]

    assert walk["available"] is True
    assert walk["purged"] is True
    assert walk["embargoed"] is True
    assert walk["fold_count"] > 0
    assert all(row["purge_gap_seconds"] >= 86400 for row in walk["folds"])
    fold_backends = {row["backend"] for row in walk["folds"]}
    assert walk["resolved_backend"] == "+".join(sorted(fold_backends))
    assert allocation["automatic_allocation_allowed"] is False
    assert allocation["live_execution_authority"] is False
    assert not any(payload["authority_contract"].values())


def test_capacity_refuses_normalized_volume_and_invalid_trade_identity(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    dataset_path = root / "data" / "trade_history" / "trade_learning_dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    broken = deepcopy(dataset["data"][0])
    broken["id"] = 999
    broken["decision_id"] = "broken-post-cost"
    broken["post_cost_forward_return"] = 1.0
    dataset["data"].append(broken)
    _write(dataset_path, dataset)

    payload = build_payload(
        root,
        config_path=root / "config" / "alpha_measurement_materialization_v1.json",
    )
    capacity = payload["measurements"]["capacity_impact_surface"]["inputs"]
    capacity_truth = payload["diagnostics"]["capacity_truth"]
    post_cost = payload["diagnostics"]["post_cost_trade_delta_validation"]

    assert capacity["daily_dollar_volume"] is None
    assert capacity_truth["normalized_relative_volume_used_as_adv"] is False
    assert capacity_truth["capacity_remains_blocked_without_direct_adv"] is True
    assert post_cost["invalid_reasons"]["post_cost_additive_identity_failed"] == 1
