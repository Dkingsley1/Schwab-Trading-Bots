from copy import deepcopy

from core.economic_source_registry import (
    build_economic_source_inventory,
    load_economic_source_registry,
    validate_economic_source_registry,
)


def test_economic_source_registry_is_valid_and_expands_grouped_sources() -> None:
    registry = load_economic_source_registry()
    validation = validate_economic_source_registry(registry)
    inventory = build_economic_source_inventory(registry)

    assert validation["ok"] is True
    assert validation["direct_source_count"] == 33
    assert validation["expanded_group_member_count"] == 32
    assert inventory["summary"]["total_routed_source_count"] == 65
    assert inventory["summary"]["new_direct_source_ids"] == [
        "fdic_bank_financials",
        "nyfed_primary_dealer_statistics",
    ]
    assert inventory["authority"]["paper_execution_authority"] is False
    assert inventory["authority"]["live_execution_authority"] is False


def test_new_sources_have_explicit_producer_capability_plane_and_family_routes() -> None:
    inventory = build_economic_source_inventory(load_economic_source_registry())
    sources = {row["source_id"]: row for row in inventory["sources"]}

    nyfed = sources["nyfed_primary_dealer_statistics"]
    assert nyfed["producer_id"] == "public_financial_context"
    assert set(nyfed["capability_ids"]) == {
        "dealer_balance_sheet",
        "repo_conditions",
        "collateral_settlement_stress",
    }
    assert {"funding_stress", "positioning_crowding", "securities_lending", "credit_curve"} <= set(
        nyfed["decision_plane_ids"]
    )

    fdic = sources["fdic_bank_financials"]
    assert fdic["producer_id"] == "public_financial_context"
    assert fdic["capability_ids"] == ["bank_credit_conditions"]
    assert {"funding_stress", "credit_curve"} <= set(fdic["decision_plane_ids"])

    assert all(row["required_for_collection_or_paper"] is False for row in inventory["sources"])
    assert all(row.get("decision_family_ids") for row in inventory["sources"])


def test_registry_rejects_insecure_sources_and_execution_authority() -> None:
    registry = deepcopy(load_economic_source_registry())
    registry["sources"][0]["official_url"] = "http://example.invalid/source"
    registry["sources"][0]["decision_plane_ids"] = []
    registry["contract"]["live_execution_authority"] = True

    validation = validate_economic_source_registry(registry)
    assert validation["ok"] is False
    assert any("official_url must use https" in error for error in validation["errors"])
    assert any("decision_plane_ids must not be empty" in error for error in validation["errors"])
    assert "contract.live_execution_authority must be false" in validation["errors"]
