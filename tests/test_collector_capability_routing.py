import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from core.collector_capability_routing import build_capability_routing, validate_catalog
from scripts.ops.collector_capability_control import build_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "config" / "collector_capability_catalog_v1.json"


def _catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _collector_contracts(catalog: dict) -> dict:
    rows = []
    for producer in catalog["producers"]:
        if producer["producer_kind"] != "collector":
            continue
        rows.append(
            {
                "name": producer["collector_name"],
                "required": producer["collector_name"] in {"market_micro_context", "crypto_market_context"},
                "fresh": True,
                "ok": True,
                "contract_ok": True,
                "age_seconds": 30,
            }
        )
    return {"timestamp_utc": "2026-08-13T16:00:00+00:00", "rows": rows, "required_failures": []}


def _hierarchy() -> dict:
    assignments = [
        {
            "bot_id": bot_id,
            "cell_id": "equity/core/daily/signal",
            "sleeve_id": "equity",
            "sub_sleeve_id": "core",
            "horizon_id": "daily",
            "regime_scope": "market_signal",
            "role_id": "signal",
            "regime_profile": {"axes": {}},
            "regime_metadata_access": {"runtime_context_required_axis_ids": []},
        }
        for bot_id in ("alpha_a", "alpha_b")
    ]
    return {
        "timestamp_utc": "2026-08-13T16:00:00+00:00",
        "assignment_count": len(assignments),
        "assignment_receipt_sha256": "fixture",
        "assignments": assignments,
    }


def _set_path(payload: dict, dotted_path: str, value: object) -> None:
    target = payload
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def _write_artifact_fixtures(
    root: Path,
    catalog: dict,
    *,
    omit_materialized_receipt: bool = False,
    omit_materialized_rows: bool = False,
) -> None:
    for producer in catalog["producers"]:
        if producer["producer_kind"] != "artifact":
            continue
        payload: dict = {
            "timestamp_utc": "2026-08-13T16:00:00+00:00",
            "ok": True,
            "overall_status": "ready",
        }
        contract = producer.get("capability_evidence_contract", {})
        if contract.get("mode") == "capability_rows":
            payload["live_promotion_ready"] = True
            payload[contract.get("path", "capabilities")] = [] if omit_materialized_rows else [
                {
                    contract.get("id_field", "capability_id"): capability_id,
                    contract.get("usable_field", "usable"): True,
                    contract.get("receipt_field", "proof_receipt_sha256"): ""
                    if omit_materialized_receipt
                    else f"proof-{capability_id}",
                }
                for capability_id in producer["capabilities"]
            ]
        for proof in producer.get("capability_proofs", {}).values():
            for path in proof.get("paths", []):
                _set_path(payload, path, {})
            for path, value in proof.get("equals", {}).items():
                _set_path(payload, path, value)
        path = root / producer["artifact_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_repository_catalog_is_complete_and_execution_free() -> None:
    catalog = _catalog()
    capability_count = sum(len(plane["capabilities"]) for plane in catalog["planes"])

    assert validate_catalog(catalog) == []
    assert len(catalog["planes"]) == 25
    assert capability_count >= 250
    assert set(catalog["safety_contract"].values()) == {False}


def test_router_shares_profiles_and_reports_unsupported_coverage(tmp_path: Path) -> None:
    catalog = _catalog()
    health, routing = build_capability_routing(
        tmp_path,
        catalog,
        _collector_contracts(catalog),
        _hierarchy(),
        now=datetime(2026, 8, 13, 16, 0, tzinfo=timezone.utc),
    )

    assert health["ok"] is True
    assert health["paper_soak_ready"] is True
    assert health["live_promotion_ready"] is False
    assert health["summary"]["bot_binding_count"] == 2
    assert health["summary"]["subscription_profile_count"] == 1
    assert health["coverage_debt"]["gap_count"] > 0
    assert health["coverage_debt"]["blocks_guarded_paper_soak"] is False
    assert routing["bot_bindings"][0]["profile_id"] == routing["bot_bindings"][1]["profile_id"]
    assert set(routing["authority_contract"].values()) == {False}
    assert routing["cache_contract"]["router_launches_physical_collectors"] is False


def test_required_collector_failure_blocks_guarded_paper_soak(tmp_path: Path) -> None:
    catalog = _catalog()
    contracts = deepcopy(_collector_contracts(catalog))
    failed = next(row for row in contracts["rows"] if row["name"] == "market_micro_context")
    failed.update({"fresh": False, "contract_ok": False})
    contracts["required_failures"] = ["market_micro_context"]

    health, _ = build_capability_routing(
        tmp_path,
        catalog,
        contracts,
        _hierarchy(),
        now=datetime(2026, 8, 13, 16, 0, tzinfo=timezone.utc),
    )

    assert health["ok"] is True
    assert health["paper_soak_ready"] is False
    assert health["paper_soak_blockers"] == ["required_collector_failure:market_micro_context"]


def test_repository_runtime_maps_every_collector_and_binds_every_bot() -> None:
    health, routing = build_payload(PROJECT_ROOT)

    assert health["structural_blockers"] == []
    assert health["current_collector_mapping"]["complete"] is True
    assert health["summary"]["mapped_current_collector_count"] == health["summary"]["current_collector_count"]
    assert health["summary"]["bot_binding_count"] == health["summary"]["assignment_count"]
    assert len(routing["bot_bindings"]) == health["summary"]["assignment_count"]
    assert routing["operating_mode"] == "metadata_subscription_shadow_only"
    assert set(routing["authority_contract"].values()) == {False}
    account_snapshot = next(
        row
        for row in health["producer_health"]
        if row["producer_id"] == "broker_account_snapshot_control"
    )
    assert {
        "broker_accounts",
        "cash_balance",
        "buying_power",
        "account_positions",
        "position_cost_basis",
        "account_restrictions",
        "margin_state",
        "account_reconciliation",
    }.issubset(set(account_snapshot["usable_capabilities"]))
    account_positions = next(
        row
        for row in routing["capability_resolutions"]
        if row["capability_id"] == "account_positions"
    )
    assert account_positions["selected_producer_id"] == "broker_account_snapshot_control"
    assert account_positions["selected_proof"]["mode"] == "field_level_payload_proof"
    central_bank = next(
        row
        for row in health["producer_health"]
        if row["producer_id"] == "central_bank_liquidity_context"
    )
    assert {
        "central_bank_balance_sheets",
        "liquidity_facilities",
        "funding_stress",
        "repo_conditions",
        "global_liquidity_regime",
    }.issubset(set(central_bank["capabilities"]))
    assert health["coverage_debt"]["next_admission_candidates"]


def test_live_capability_readiness_is_candidate_specific(tmp_path: Path) -> None:
    catalog = _catalog()
    _write_artifact_fixtures(tmp_path, catalog)

    health, routing = build_capability_routing(
        tmp_path,
        catalog,
        _collector_contracts(catalog),
        _hierarchy(),
        now=datetime(2026, 8, 13, 16, 0, tzinfo=timezone.utc),
    )

    assert health["live_promotion_ready"] is True
    assert health["summary"]["required_capability_usable_ratio"] == 1.0
    assert health["summary"]["full_catalog_coverage_ready"] is False
    assert health["coverage_debt"]["candidate_blocking_gap_count"] == 0
    assert health["coverage_debt"]["optional_gap_count"] > 0
    assert health["coverage_debt"]["plane_rollups"]
    assert health["coverage_debt"]["next_admission_candidates"] == []
    assert health["coverage_debt"]["admission_contract"]["minimum_subscribed_bot_count"] == 3
    assert health["coverage_debt"]["admission_contract"]["human_approval_required"] is True
    assert health["coverage_debt"]["admission_contract"]["automatic_producer_creation"] is False
    assert health["coverage_debt"]["gap_receipt_sha256"]
    assert health["provider_resilience"]["provider_selection_published"] is True
    assert health["provider_resilience"]["no_source_required_capability_ids"] == []
    assert routing["capability_resolutions"]
    assert all(row["selected_producer_id"] for row in routing["capability_resolutions"] if row["required"])


def test_materialized_capability_requires_field_level_receipt(tmp_path: Path) -> None:
    catalog = _catalog()
    _write_artifact_fixtures(tmp_path, catalog, omit_materialized_receipt=True)

    health, _ = build_capability_routing(
        tmp_path,
        catalog,
        _collector_contracts(catalog),
        _hierarchy(),
        now=datetime(2026, 8, 13, 16, 0, tzinfo=timezone.utc),
    )

    assert health["live_promotion_ready"] is False
    assert health["summary"]["unavailable_required_capability_count"] > 0
    materializer = next(
        row
        for row in health["producer_health"]
        if row["producer_id"] == "capability_materialization_control"
    )
    assert materializer["usable"] is True
    assert materializer["usable_capabilities"] == []


def test_materialized_capability_rows_cannot_fall_back_to_producer_health(tmp_path: Path) -> None:
    catalog = _catalog()
    _write_artifact_fixtures(tmp_path, catalog, omit_materialized_rows=True)

    health, _ = build_capability_routing(
        tmp_path,
        catalog,
        _collector_contracts(catalog),
        _hierarchy(),
        now=datetime(2026, 8, 13, 16, 0, tzinfo=timezone.utc),
    )

    assert health["live_promotion_ready"] is False
    materializer = next(
        row
        for row in health["producer_health"]
        if row["producer_id"] == "capability_materialization_control"
    )
    assert materializer["usable"] is True
    assert materializer["usable_capabilities"] == []
    assert all(
        proof["mode"] == "materialized_capability_receipt" and proof["passed"] is False
        for proof in materializer["capability_proofs"].values()
    )
