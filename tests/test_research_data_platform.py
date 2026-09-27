from __future__ import annotations

from pathlib import Path

import pytest

from core.research_data_platform import (
    REQUIRED_CAPABILITIES,
    ResearchDataCatalog,
    build_reproducibility_receipt,
    evaluate_feed_slo,
    evaluate_source_value,
    load_policy,
    new_alpha_record,
    normalize_simulation_event,
    portfolio_alpha_advisory,
    select_bitemporal_rows,
    structural_probe,
    transition_alpha,
    validate_policy,
    verify_alpha_history,
)
from scripts.ops.research_data_platform_control import build_payload
from scripts.ops.live_feed_status_contract import _research_data_platform_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_policy_has_ten_owned_capabilities_and_complete_family_coverage() -> None:
    policy = load_policy()
    report = validate_policy(policy, project_root=PROJECT_ROOT)

    assert report["ok"] is True, report["errors"]
    assert report["capability_count"] == len(REQUIRED_CAPABILITIES) == 10
    assert report["data_product_count"] == 10
    assert report["decision_family_count"] == 15
    assert min(report["family_product_counts"].values()) >= 2
    assert report["authority_safe"] is True


def test_catalog_filters_products_and_fails_closed_on_entitlements() -> None:
    policy = load_policy()
    catalog = ResearchDataCatalog(policy)

    macro = catalog.query_catalog(decision_family_id="macro_rates_fx")
    assert any(row["dataset_id"] == "official_us_macro_v1" for row in macro)
    assert all("macro_rates_fx" in row["consumer_decision_families"] for row in macro)
    assert catalog.authorize_use("official_us_macro_v1", "research")["authorized"]

    denied = catalog.authorize_use("broker_market_observations_v1", "research")
    assert denied["authorized"] is False
    assert "required_entitlement_missing" in denied["reasons"]
    assert "external_terms_review_pending" in denied["reasons"]
    assert denied["credentials_stored"] is False
    assert denied["execution_authority"] is False

    permitted = catalog.authorize_use(
        "broker_market_observations_v1",
        "research",
        entitlement_states={
            "broker_market_data_access": "active",
            "license_terms_review:broker_account_restricted": "approved",
        },
    )
    assert permitted["authorized"] is True


def test_query_plan_is_point_in_time_receipted_and_deterministic() -> None:
    catalog = ResearchDataCatalog(load_policy())
    kwargs = {
        "dataset_ids": ["official_us_macro_v1", "point_in_time_feature_store_v1"],
        "consumer_id": "test-consumer",
        "purpose": "research",
        "as_of_utc": "2026-08-20T12:00:00Z",
        "valid_at_utc": "2026-08-19T12:00:00Z",
    }
    first = catalog.build_query_plan(**kwargs)
    second = catalog.build_query_plan(**kwargs)

    assert first["query_receipt_sha256"] == second["query_receipt_sha256"]
    assert first["future_knowledge_allowed"] is False
    assert first["execution_authority"] is False
    with pytest.raises(PermissionError, match="dataset_use_not_authorized"):
        catalog.build_query_plan(
            **{
                **kwargs,
                "dataset_ids": ["broker_market_observations_v1"],
            }
        )


def test_bitemporal_selection_prevents_revision_lookahead() -> None:
    contract = load_policy()["bitemporal_contract"]
    rows = [
        {
            "series_id": "CPI",
            "period": "2026-07",
            "effective_at_utc": "2026-08-01T00:00:00Z",
            "known_at_utc": "2026-08-02T00:00:00Z",
            "superseded_at_utc": "2026-08-10T00:00:00Z",
            "revision_id": "initial",
            "value": 100.0,
        },
        {
            "series_id": "CPI",
            "period": "2026-07",
            "effective_at_utc": "2026-08-01T00:00:00Z",
            "known_at_utc": "2026-08-10T00:00:00Z",
            "revision_id": "revised",
            "value": 101.0,
        },
    ]

    before_revision = select_bitemporal_rows(
        rows,
        as_of_utc="2026-08-05T00:00:00Z",
        valid_at_utc="2026-08-01T12:00:00Z",
        natural_key_columns=["series_id", "period"],
        contract=contract,
    )
    after_revision = select_bitemporal_rows(
        rows,
        as_of_utc="2026-08-12T00:00:00Z",
        valid_at_utc="2026-08-01T12:00:00Z",
        natural_key_columns=["series_id", "period"],
        contract=contract,
    )

    assert [row["revision_id"] for row in before_revision] == ["initial"]
    assert [row["revision_id"] for row in after_revision] == ["revised"]


def test_alpha_lifecycle_rejects_skips_and_requires_evidence() -> None:
    lifecycle = load_policy()["alpha_lifecycle"]
    record = new_alpha_record(
        "alpha-test", observed_at_utc="2026-08-20T12:00:00Z"
    )

    with pytest.raises(ValueError, match="alpha_transition_not_allowed"):
        transition_alpha(
            record,
            "validated_alpha",
            evidence={},
            observed_at_utc="2026-08-20T12:01:00Z",
            lifecycle=lifecycle,
        )
    with pytest.raises(ValueError, match="alpha_transition_evidence_missing"):
        transition_alpha(
            record,
            "candidate_alpha",
            evidence={"hypothesis_id": "h1"},
            observed_at_utc="2026-08-20T12:01:00Z",
            lifecycle=lifecycle,
        )

    transitioned = transition_alpha(
        record,
        "candidate_alpha",
        evidence={
            "dataset_receipts": ["dataset-receipt"],
            "hypothesis_id": "h1",
            "candidate_id": "candidate-1",
        },
        observed_at_utc="2026-08-20T12:01:00Z",
        lifecycle=lifecycle,
    )
    assert transitioned["state"] == "candidate_alpha"
    assert transitioned["live_execution_authority"] is False
    assert verify_alpha_history(transitioned)["ok"] is True


def test_source_value_does_not_confuse_source_presence_with_alpha() -> None:
    policy = load_policy()
    collecting = evaluate_source_value(
        "macro-source",
        {"quality": 1.0, "freshness": 1.0, "availability": 1.0},
        policy,
    )
    assert collecting["status"] == "collecting"
    assert collecting["score"] is None
    assert collecting["qualified"] is False

    evaluated = evaluate_source_value(
        "macro-source",
        {
            "candidate_id": "candidate-1",
            "candidate_bound_samples": 30,
            "quality": 0.95,
            "freshness": 1.0,
            "availability": 1.0,
            "incremental_information": 0.8,
            "net_post_cost_contribution": 0.7,
            "nonredundancy": 0.9,
        },
        policy,
    )
    assert evaluated["evidence_ready"] is True
    assert evaluated["qualified"] is True
    assert evaluated["purchase_authority"] is False
    assert evaluated["retirement_authority"] is False


def test_portfolio_combination_remains_candidate_bound_advisory() -> None:
    policy = load_policy()
    sleeve_ids = ("dividend", "bond", "fx", "volatility")
    report = portfolio_alpha_advisory(
        candidate_id="candidate-1",
        sleeves=[
            {
                "sleeve_id": sleeve_id,
                "candidate_id": "candidate-1",
                "qualified": True,
                "independent_fills": 35,
                "expected_return_bps": 8.0 - index,
                "cost_bps": 1.0,
            }
            for index, sleeve_id in enumerate(sleeve_ids)
        ],
        covariance={
            left: {right: 1.0 if left == right else 0.1 for right in sleeve_ids}
            for left in sleeve_ids
        },
        current_weights={},
        policy=policy,
    )

    assert report["ok"] is True
    assert report["advisory_only"] is True
    assert report["execution_authority"] is False

    abstain = portfolio_alpha_advisory(
        candidate_id="candidate-2",
        sleeves=[],
        covariance={},
        current_weights={},
        policy=policy,
    )
    assert abstain["status"] == "abstain"
    assert abstain["execution_authority"] is False


def test_simulation_modes_share_payload_identity_without_submission_authority() -> None:
    policy = load_policy()
    event = {
        "event_id": "event-1",
        "trace_id": "trace-1",
        "candidate_id": "candidate-1",
        "symbol": "SPY",
        "event_time_utc": "2026-08-20T12:00:00Z",
        "known_at_utc": "2026-08-20T12:00:01Z",
        "event_type": "market_observation",
        "payload": {"price": 100.0, "size": 5},
    }
    paper = normalize_simulation_event(event, mode="paper", policy=policy)
    live = normalize_simulation_event(event, mode="live", policy=policy)

    assert paper["payload_sha256"] == live["payload_sha256"]
    assert paper["submission_authority"] is False
    assert live["submission_authority"] is False
    assert live["representation_only"] is True


def test_feed_slo_and_reproducibility_fail_closed() -> None:
    policy = load_policy()
    product = ResearchDataCatalog(policy).dataset("official_us_macro_v1")
    stale = evaluate_feed_slo(
        product,
        {
            "age_seconds": product["freshness_slo_seconds"] + 1,
            "completeness_ratio": 1.0,
            "validity_ratio": 1.0,
            "availability_ratio": 1.0,
            "correction_ratio": 0.0,
        },
        policy,
    )
    assert stale["ok"] is False
    assert "freshness" in stale["failed_checks"]
    assert stale["execution_authority"] is False

    with pytest.raises(ValueError, match="reproducibility_materials_missing"):
        build_reproducibility_receipt({"candidate_id": "candidate-1"}, policy)
    materials = {
        key: f"receipt-{key}"
        for key in policy["reproducibility_contract"]["required_materials"]
    }
    first = build_reproducibility_receipt(materials, policy)
    second = build_reproducibility_receipt(materials, policy)
    assert first["receipt_sha256"] == second["receipt_sha256"]
    assert first["external_attestation"] is False
    assert first["execution_authority"] is False


def test_structural_probe_and_runtime_control_keep_evidence_separate() -> None:
    policy = load_policy()
    probe = structural_probe(policy)
    payload = build_payload(PROJECT_ROOT)

    assert probe["ok"] is True
    assert probe["ready_count"] == probe["control_count"] == 10
    assert probe["candidate_bound_economic_evidence"] is False
    assert payload["implementation_grade"] == "A+"
    assert payload["implementation_ready_count"] == 10
    assert payload["implementation_control_count"] == 10
    assert payload["evidence_ready_count"] <= payload["evidence_control_count"] == 10
    assert payload["paper_soak_ready"] is True
    assert payload["paper_impact"] == "none"
    assert payload["live_promotion_ready"] is False
    assert payload["live_execution_authority"] is False
    assert payload["soak_acceptance"]["reset_soak_clock"] is False
    assert payload["soak_acceptance"]["changes_signal_or_order_semantics"] is False


def test_livefeed_reports_evidence_debt_without_false_paper_degradation() -> None:
    row = _research_data_platform_row(
        {
            "research_data_platform": {
                "present": True,
                "fresh": True,
                "age_seconds": 3.0,
                "payload": {
                    "overall_status": "ready_with_evidence_debt",
                    "implementation_grade": "A+",
                    "implementation_ready_count": 10,
                    "implementation_control_count": 10,
                    "evidence_ready_count": 2,
                    "evidence_control_count": 10,
                    "catalog": {
                        "ready_product_count": 8,
                        "data_product_count": 10,
                        "decision_family_count": 15,
                    },
                    "source_value": {"qualified_count": 0, "source_count": 20},
                    "candidate_binding": {"candidate_id": "pc-test", "bound": True},
                    "paper_soak_ready": True,
                    "paper_impact": "none",
                    "live_promotion_ready": False,
                    "live_execution_authority": False,
                },
            }
        }
    )

    assert row["status"] == "ready_with_evidence_debt"
    assert row["implementation_grade"] == "A+"
    assert row["paper_soak_ready"] is True
    assert row["paper_impact"] == "none"
    assert row["action"] == "none"
    assert row["live_promotion_ready"] is False
    assert row["live_authority"] is False
