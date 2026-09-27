from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.institutional_research_extensions import (
    build_dataset_version,
    classify_material_change,
    decision_extension_metadata,
    evaluate_candidate_risk_schedule,
    evaluate_execution_frontier,
    evaluate_factor_exposure,
    evaluate_incident_ownership,
    load_policy,
    plan_research_dag,
    reconcile_cross_engine_valuation,
    select_dataset_version,
    structural_probe,
    verify_dataset_versions,
)
from scripts.ops.institutional_research_extensions_control import build_payload

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_policy_has_eight_owned_controls_and_six_firm_families() -> None:
    report = structural_probe(load_policy(), project_root=PROJECT_ROOT)

    assert report["ok"] is True
    assert report["ready_control_count"] == report["control_count"] == 8
    assert report["firm_reference_count"] == 10
    assert report["firm_organization_count"] >= 6
    assert all(value > 0 for value in report["control_adoption"].values())
    assert report["live_execution_authority"] is False


def test_factor_benchmark_reports_exposure_without_calling_intercept_alpha() -> None:
    policy = load_policy()
    observations = [
        {
            "strategy_return": 0.001
            + 0.6 * (index / 10000.0)
            - 0.2 * ((index % 7) / 10000.0),
            "equity_market": index / 10000.0,
            "value": (index % 7) / 10000.0,
        }
        for index in range(1, 61)
    ]

    report = evaluate_factor_exposure(
        observations,
        candidate_id="pc-test",
        policy=policy,
        factor_ids=["equity_market", "value"],
    )

    assert report["ok"] is True
    assert report["evidence_eligible"] is True
    assert report["factor_loadings"]["equity_market"] == pytest.approx(0.6, abs=0.02)
    assert report["factor_loadings"]["value"] == pytest.approx(-0.2, abs=0.03)
    assert report["intercept_is_proven_alpha"] is False
    assert report["strategy_admission_authority"] is False


def test_factor_benchmark_abstains_below_observation_floor() -> None:
    report = evaluate_factor_exposure(
        [{"strategy_return": 0.01, "value": 0.01}],
        candidate_id="pc-test",
        policy=load_policy(),
        factor_ids=["value"],
    )

    assert report["status"] == "collecting"
    assert report["evidence_eligible"] is False


def test_incident_ownership_requires_slo_and_closeout_receipts() -> None:
    now = datetime(2026, 8, 23, 12, 30, tzinfo=timezone.utc)
    ready = evaluate_incident_ownership(
        [
            {
                "incident_id": "inc-1",
                "pipeline_id": "quotes",
                "owner": "data-reliability",
                "severity": "high",
                "status": "resolved",
                "detected_at_utc": "2026-08-23T12:00:00Z",
                "acknowledged_at_utc": "2026-08-23T12:05:00Z",
                "resolved_at_utc": "2026-08-23T12:20:00Z",
                "root_cause_receipt": "root-1",
                "remediation_receipt": "repair-1",
            }
        ],
        policy=load_policy(),
        now=now,
    )
    incomplete = evaluate_incident_ownership(
        [
            {
                "incident_id": "inc-2",
                "pipeline_id": "macro",
                "owner": "",
                "severity": "critical",
                "status": "resolved",
                "detected_at_utc": "2026-08-23T11:00:00Z",
            }
        ],
        policy=load_policy(),
        now=now,
    )

    assert ready["ok"] is True
    assert ready["incidents"][0]["acknowledgement_seconds"] == 300
    assert incomplete["ok"] is False
    assert "owner" in incomplete["incidents"][0]["missing_fields"]
    assert incomplete["execution_authority"] is False


def test_material_change_classification_preserves_history_but_requires_forward_scope() -> (
    None
):
    report = classify_material_change(
        ["core/risk_engine.py", "docs/architecture/RISK.md"],
        candidate_id="pc-test",
        policy=load_policy(),
    )

    assert report["change_class"] == "risk_execution_critical"
    assert report["forward_evidence_hours"] == 720
    assert report["cumulative_soak_history_preserved"] is True
    assert report["affected_scope_forward_window_required"] is True
    assert report["automatic_acceptance_authority"] is False


def _risk_snapshot(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_id": "pc-test",
        "broker_truth_age_seconds": 30,
        "gross_exposure_ratio": 0.5,
        "net_exposure_ratio": 0.3,
        "drawdown_ratio": 0.02,
        "daily_loss_ratio": 0.005,
        "buying_power_utilization_ratio": 0.4,
        "symbol_weights": {"SPY": 0.2, "SCHD": 0.15},
        "sleeve_weights": {"dividend": 0.35, "baseline_parallel": 0.3},
        "asset_types": ["EQUITY", "ETF"],
        "instruction_types": ["BUY", "SELL_TO_CLOSE"],
    }
    payload.update(overrides)
    return payload


def test_candidate_risk_schedule_is_candidate_bound_and_advisory() -> None:
    ready = evaluate_candidate_risk_schedule(
        _risk_snapshot(), candidate_id="pc-test", policy=load_policy()
    )
    stale = evaluate_candidate_risk_schedule(
        _risk_snapshot(broker_truth_age_seconds=300),
        candidate_id="pc-test",
        policy=load_policy(),
    )

    assert ready["ok"] is True
    assert stale["ok"] is False
    assert "broker_truth_freshness" in stale["failed_checks"]
    assert ready["changes_risk_limits"] is False
    assert ready["execution_authority"] is False


def test_execution_frontier_selects_only_mature_positive_post_cost_choice() -> None:
    report = evaluate_execution_frontier(
        [
            {
                "alternative_id": "fast",
                "candidate_id": "pc-test",
                "expected_alpha_bps": 8.0,
                "spread_bps": 1.0,
                "impact_bps": 3.0,
                "adverse_selection_bps": 1.0,
                "fees_bps": 0.2,
                "alpha_decay_bps": 0.2,
                "fill_probability": 0.9,
                "participation_rate": 0.1,
                "independent_fill_count": 45,
            },
            {
                "alternative_id": "thin",
                "candidate_id": "pc-test",
                "expected_alpha_bps": 20.0,
                "spread_bps": 1.0,
                "impact_bps": 1.0,
                "adverse_selection_bps": 1.0,
                "fees_bps": 0.2,
                "alpha_decay_bps": 0.2,
                "fill_probability": 0.9,
                "participation_rate": 0.1,
                "independent_fill_count": 2,
            },
        ],
        candidate_id="pc-test",
        policy=load_policy(),
    )

    assert report["selected_alternative_id"] == "fast"
    assert report["alternatives"][1]["evidence_ready"] is False
    assert report["order_authority"] is False


def test_research_dag_reuses_exact_receipts_and_invalidates_descendants() -> None:
    policy = load_policy()
    first = plan_research_dag(
        candidate_id="pc-test",
        code_receipt="code-1",
        dataset_receipts={"quotes": "data-1"},
        checkpoint_receipts={},
        policy=policy,
    )
    checkpoints = {
        row["stage_id"]: row["expected_receipt_sha256"] for row in first["stages"]
    }
    reused = plan_research_dag(
        candidate_id="pc-test",
        code_receipt="code-1",
        dataset_receipts={"quotes": "data-1"},
        checkpoint_receipts=checkpoints,
        policy=policy,
    )
    checkpoints["source_snapshot"] = "stale"
    invalidated = plan_research_dag(
        candidate_id="pc-test",
        code_receipt="code-1",
        dataset_receipts={"quotes": "data-1"},
        checkpoint_receipts=checkpoints,
        policy=policy,
    )

    assert first["run_count"] == 7
    assert reused["reuse_count"] == 7
    assert invalidated["run_count"] == 7
    assert invalidated["launch_authority"] is False


def test_versioned_dataset_manifest_supports_verified_time_travel() -> None:
    first = build_dataset_version(
        "quotes",
        [
            {
                "path": "quotes-1.jsonl",
                "sha256": "a" * 64,
                "size_bytes": 10,
                "row_count": 1,
            }
        ],
        committed_at_utc="2026-08-23T12:00:00Z",
    )
    second = build_dataset_version(
        "quotes",
        [
            {
                "path": "quotes-2.jsonl",
                "sha256": "b" * 64,
                "size_bytes": 20,
                "row_count": 2,
            }
        ],
        committed_at_utc="2026-08-23T13:00:00Z",
        parent_version_id=first["version_id"],
    )

    verification = verify_dataset_versions([second, first])
    selected = select_dataset_version([first, second], as_of_utc="2026-08-23T12:30:00Z")

    assert verification["ok"] is True
    assert selected is not None and selected["version_id"] == first["version_id"]
    assert verification["source_retirement_authority"] is False


def test_cross_engine_valuation_reuses_independent_oracle_and_keeps_probe_ineligible() -> (
    None
):
    report = reconcile_cross_engine_valuation(
        candidate_id="pc-test",
        product_id="option-SPY",
        valuation_time_utc="2026-08-23T12:00:00Z",
        measures=["present_value", "delta"],
        primary_engine={
            "provider_id": "local-primary",
            "model_id": "black-scholes-local",
            "values": {"present_value": 1.0, "delta": 0.5},
        },
        oracle_engine={
            "provider_id": "independent-oracle",
            "model_id": "finite-difference-oracle",
            "signed_attestation_id": "synthetic-signature",
            "values": {"present_value": 1.005, "delta": 0.50001},
        },
        policy=load_policy(),
        synthetic_probe=True,
    )

    assert report["ok"] is True
    assert report["structurally_independent"] is True
    assert report["evidence_eligible"] is False
    assert report["order_authority"] is False


def test_decision_metadata_is_receipted_and_never_changes_route_authority() -> None:
    metadata = decision_extension_metadata(
        decision_family_id="dividend_income", policy=load_policy()
    )

    assert len(metadata["control_ids"]) == 8
    assert metadata["receipt_sha256"]
    assert metadata["metadata_only"] is True
    assert metadata["existing_route_authority_unchanged"] is True
    assert metadata["execution_authority"] is False


def test_runtime_control_separates_structural_readiness_from_earned_evidence() -> None:
    payload = build_payload(PROJECT_ROOT)

    assert payload["ok"] is True
    assert payload["implementation_grade"] == "A+"
    assert (
        payload["implementation_ready_count"]
        == payload["implementation_control_count"]
        == 8
    )
    assert payload["evidence_ready_count"] < payload["evidence_control_count"] == 8
    assert payload["overall_status"] == "ready_with_evidence_debt"
    assert payload["paper_soak_ready"] is True
    assert payload["cumulative_soak_history_preserved"] is True
    assert payload["reset_soak_clock"] is False
    assert payload["live_execution_authority"] is False
