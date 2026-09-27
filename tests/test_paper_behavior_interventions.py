from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.paper_behavior_interventions import (
    build_runtime_overlay_proposal,
    evaluate_paper_behavior_interventions,
    file_sha256,
    load_policy,
)
from scripts.ops.paper_behavior_intervention_drill import build_payload as build_drill
from scripts.ops.paper_profitability_control import (
    _paper_behavior_intervention_admission_contract,
    build_runtime_control_payload,
)
import scripts.run_shadow_training_loop as shadow

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _policy() -> dict:
    return load_policy(project_root=PROJECT_ROOT)


def _candidate_binding() -> dict:
    return {
        "candidate_id": "candidate-g1",
        "candidate_generation": 1,
        "accepted_at_utc": "2026-08-26T12:00:00+00:00",
        "candidate_state_sha256": "a" * 64,
        "live_execution_authority": False,
        "valid": True,
    }


def _contract() -> dict:
    return build_runtime_overlay_proposal(
        policy=_policy(),
        candidate_binding=_candidate_binding(),
        policy_sha256="b" * 64,
        generated_at_utc=datetime.now(timezone.utc),
    )


def _healthy_features() -> dict:
    return dict(_policy()["healthy_feature_baseline"])


def test_clean_buy_is_preserved_without_size_increase() -> None:
    result = evaluate_paper_behavior_interventions(
        action="BUY",
        features=_healthy_features(),
        runtime_contract=_contract(),
    )
    assert result["action"] == "BUY"
    assert result["entry_size_multiplier_norm"] == 1.0
    assert result["triggered_interventions"] == []


def test_stale_buy_is_blocked() -> None:
    features = _healthy_features()
    features["quote_age_ms"] = 5000.0
    result = evaluate_paper_behavior_interventions(
        action="BUY", features=features, runtime_contract=_contract()
    )
    assert result["action"] == "HOLD"
    assert result["entry_size_multiplier_norm"] == 0.0
    assert "stale_data_abstention" in result["triggered_interventions"]


def test_thin_liquidity_reduces_but_does_not_reverse_buy() -> None:
    features = _healthy_features()
    features.update(
        {
            "market_micro_tradeability_score_norm": 0.5,
            "execution_fitness_norm": 0.52,
            "liquidity_quality_norm": 0.5,
        }
    )
    result = evaluate_paper_behavior_interventions(
        action="BUY", features=features, runtime_contract=_contract()
    )
    assert result["action"] == "BUY"
    assert 0.0 < result["entry_size_multiplier_norm"] < 1.0
    assert "liquidity_size_throttle" in result["triggered_interventions"]


def test_integer_loss_streak_throttles_without_collapsing_to_zero() -> None:
    features = _healthy_features()
    features["lane_loss_streak"] = 4
    result = evaluate_paper_behavior_interventions(
        action="BUY", features=features, runtime_contract=_contract()
    )
    assert result["action"] == "BUY"
    assert 0.25 <= result["entry_size_multiplier_norm"] <= 0.65
    assert "drawdown_loss_streak_throttle" in result["triggered_interventions"]


@pytest.mark.parametrize("action", ["SELL", "HOLD"])
def test_exit_and_hold_are_preserved_even_under_bad_conditions(action: str) -> None:
    features = _healthy_features()
    features.update({"quote_age_ms": 9000.0, "portfolio_drawdown_pressure_norm": 1.0})
    result = evaluate_paper_behavior_interventions(
        action=action, features=features, runtime_contract=_contract()
    )
    assert result["action"] == action
    assert result["entry_size_multiplier_norm"] == 1.0


def test_invalid_contract_is_noop() -> None:
    contract = _contract()
    contract["live_execution_allowed"] = True
    result = evaluate_paper_behavior_interventions(
        action="BUY",
        features={"quote_age_ms": 9000.0},
        runtime_contract=contract,
    )
    assert result["action"] == "BUY"
    assert result["entry_size_multiplier_norm"] == 1.0
    assert result["active"] is False


def _materialize_drill_root(tmp_path: Path) -> tuple[Path, dict]:
    config = tmp_path / "config"
    runtime = tmp_path / "governance" / "runtime"
    research = tmp_path / "governance" / "research"
    config.mkdir(parents=True)
    runtime.mkdir(parents=True)
    research.mkdir(parents=True)
    policy_path = config / "paper_behavior_intervention_drill_v1.json"
    policy_path.write_text(
        (
            PROJECT_ROOT / "config" / "paper_behavior_intervention_drill_v1.json"
        ).read_text(),
        encoding="utf-8",
    )
    candidate_path = runtime / "production_candidate_state.json"
    candidate_path.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-g7",
                "generation": 7,
                "accepted_at_utc": "2026-08-26T12:00:00+00:00",
                "accepted_git_head": "abc123",
                "live_execution_authority": False,
            }
        ),
        encoding="utf-8",
    )
    payload = build_drill(project_root=tmp_path)
    (research / "paper_behavior_intervention_drill_latest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (config / "trading_behavior_drill_program_v1.json").write_text(
        (
            PROJECT_ROOT / "config" / "trading_behavior_drill_program_v1.json"
        ).read_text(),
        encoding="utf-8",
    )
    program_payload = {
        "timestamp_utc": payload["timestamp_utc"],
        "status": "ready",
        "ok": True,
        "control_grade": "A+",
        "control_score": 100.0,
        "admission_eligible": True,
        "run_id": "tdp-candidate-g7-test",
        "candidate_binding": payload["candidate_binding"],
        "suite_results": [
            {
                "suite_id": suite_id,
                "ok": True,
                "control_grade": "A+",
            }
            for suite_id in (
                "profitability_crisis",
                "profitability_adversarial",
                "paper_behavior_intervention",
            )
        ],
        "comparison": {"non_regressed": True, "status": "baseline_established"},
        "behavior_change_contract": {
            "eligible_for_single_writer_admission": True,
            "runtime_overlay_proposal_sha256": payload[
                "runtime_overlay_proposal_sha256"
            ],
        },
        "authority_contract": {
            "can_write_runtime_control": False,
            "can_submit_paper_orders": False,
            "can_submit_live_orders": False,
            "can_access_broker": False,
            "can_access_network": False,
            "can_change_live_execution": False,
        },
        "resource_contract": {
            "runtime_control_writes": 0,
            "paper_orders_submitted": 0,
            "live_orders_submitted": 0,
            "broker_requests": 0,
            "network_requests": 0,
        },
        "receipts": {"input_signature_sha256": "c" * 64},
    }
    (research / "trading_behavior_drill_program_latest.json").write_text(
        json.dumps(program_payload), encoding="utf-8"
    )
    return candidate_path, payload


def test_complete_drill_pack_is_a_plus_and_admission_eligible(tmp_path: Path) -> None:
    _, payload = _materialize_drill_root(tmp_path)
    assert payload["ok"] is True
    assert payload["control_grade"] == "A+"
    assert payload["admission_eligible"] is True
    assert payload["scenario_summary"]["passed_scenario_count"] == 14
    assert payload["scenario_summary"]["case_count"] == 17


def test_profitability_controller_admits_fresh_candidate_bound_proposal(
    tmp_path: Path,
) -> None:
    _, _ = _materialize_drill_root(tmp_path)
    paper = {
        "accounting_views": {
            "candidate_forward_flow": {
                "candidate_id": "candidate-g7",
                "candidate_generation": 7,
                "sample_count": 0,
                "observed_days": 0,
                "post_cost_pnl_delta_total": 0.0,
            }
        }
    }
    admission = _paper_behavior_intervention_admission_contract(
        project_root=tmp_path,
        paper_performance=paper,
    )
    assert admission["active"] is True
    assert admission["status"] == "paper_probation"
    assert admission["failed_checks"] == []
    assert admission["runtime_overlay_contract"]["live_execution_allowed"] is False


def test_profitability_controller_rejects_missing_drill_program(
    tmp_path: Path,
) -> None:
    _, _ = _materialize_drill_root(tmp_path)
    (
        tmp_path
        / "governance"
        / "research"
        / "trading_behavior_drill_program_latest.json"
    ).unlink()
    admission = _paper_behavior_intervention_admission_contract(
        project_root=tmp_path,
        paper_performance={},
    )

    assert admission["active"] is False
    assert "drill_program_artifact_loaded" in admission["failed_checks"]


def test_candidate_change_invalidates_proposal(tmp_path: Path) -> None:
    candidate_path, _ = _materialize_drill_root(tmp_path)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["generation"] = 8
    candidate["candidate_id"] = "candidate-g8"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    admission = _paper_behavior_intervention_admission_contract(
        project_root=tmp_path,
        paper_performance={},
    )
    assert admission["active"] is False
    assert admission["status"] == "not_admitted"
    assert "candidate_id_matches" in admission["failed_checks"]
    assert admission["runtime_overlay_contract"]["active"] is False


def test_runtime_payload_keeps_single_admitted_overlay() -> None:
    overlay = _contract()
    payload = {
        "paper_behavior_intervention_admission_contract": {"active": True},
        "paper_behavior_intervention_overlay_contract": overlay,
    }
    runtime = build_runtime_control_payload(payload)
    assert runtime["paper_behavior_intervention_overlay_contract"] == overlay
    assert (
        runtime["global_runtime_policy"][
            "apply_candidate_bound_paper_behavior_interventions"
        ]
        is True
    )
    assert (
        runtime["global_runtime_policy"][
            "paper_behavior_interventions_live_execution_allowed"
        ]
        is False
    )


def test_shadow_runtime_applies_multiplier_to_existing_paper_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    monkeypatch.setattr(
        shadow, "_paper_behavior_intervention_contract", lambda: contract
    )
    monkeypatch.setattr(
        shadow,
        "_production_candidate_context",
        lambda _root: {
            "production_candidate_id": "candidate-g1",
            "production_candidate_generation": 1,
            "production_candidate_state_file_sha256": "a" * 64,
        },
    )
    features = _healthy_features()
    features["candidate_bound_post_cost_samples"] = 5
    features["paper_profitability_strategy_size_multiplier_norm"] = 0.8
    action, _, reasons, out = shadow._apply_admitted_paper_behavior_interventions(
        profile="dividend",
        strategy="paper_mirror::candidate",
        action="BUY",
        score=0.7,
        threshold=0.55,
        reasons=[],
        features=features,
    )
    assert action == "BUY"
    assert out["paper_profitability_strategy_size_multiplier_norm"] == 0.25
    assert out["paper_profitability_size_multiplier_norm"] == 0.25
    assert "paper_behavior_intervention_size=0.250" in reasons


def test_shadow_runtime_candidate_mismatch_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shadow, "_paper_behavior_intervention_contract", _contract)
    monkeypatch.setattr(
        shadow,
        "_production_candidate_context",
        lambda _root: {
            "production_candidate_id": "candidate-g2",
            "production_candidate_generation": 2,
            "production_candidate_state_file_sha256": "b" * 64,
        },
    )
    action, score, reasons, out = shadow._apply_admitted_paper_behavior_interventions(
        profile="dividend",
        strategy="paper_mirror::candidate",
        action="BUY",
        score=0.7,
        threshold=0.55,
        reasons=["original"],
        features={"quote_age_ms": 9000.0},
    )
    assert action == "BUY"
    assert score == 0.7
    assert reasons == ["original"]
    assert out["paper_behavior_intervention_candidate_binding_ready_norm"] == 0.0


def test_shadow_runtime_candidate_file_hash_mismatch_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shadow, "_paper_behavior_intervention_contract", _contract)
    monkeypatch.setattr(
        shadow,
        "_production_candidate_context",
        lambda _root: {
            "production_candidate_id": "candidate-g1",
            "production_candidate_generation": 1,
            "production_candidate_state_file_sha256": "c" * 64,
        },
    )
    action, _, _, out = shadow._apply_admitted_paper_behavior_interventions(
        profile="dividend",
        strategy="paper_mirror::candidate",
        action="BUY",
        score=0.7,
        threshold=0.55,
        reasons=[],
        features={"quote_age_ms": 9000.0},
    )
    assert action == "BUY"
    assert out["paper_behavior_intervention_candidate_binding_ready_norm"] == 0.0


def test_materialized_candidate_receipt_is_real_file_hash(tmp_path: Path) -> None:
    candidate_path, payload = _materialize_drill_root(tmp_path)
    assert payload["candidate_binding"]["candidate_state_sha256"] == file_sha256(
        candidate_path
    )
