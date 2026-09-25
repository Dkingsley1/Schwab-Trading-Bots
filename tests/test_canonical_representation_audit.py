from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import canonical_representation_audit as audit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _contract(strategy_id: str, sleeve_id: str) -> dict:
    return {
        "strategy_id": strategy_id,
        "sleeve_id": sleeve_id,
        "authority": {
            "can_create_intent": False,
            "can_reverse_intent": False,
            "can_increase_quantity": False,
            "can_allocate_capital": False,
            "can_change_labels": False,
            "can_grant_promotion": False,
            "can_submit_live_order": False,
        },
        "measurement_parameters": {
            "automatic_live_promotion_allowed": False,
            "may_allocate_capital": False,
        },
    }


def _populate_project(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config" / "system_role_contracts_v1.json",
        {
            "policy_id": "system_responsibility_and_authority_v1",
            "roles": [{"role_id": "data_collector"}, {"role_id": "strategy_bot"}],
            "components": [{"component_id": "collector"}],
            "state_domains": [{"domain_id": "paper_state"}],
            "control_surface_bindings": [{"component_id": "collector"}],
            "exclusive_action_owners": [{"action": "live_submit"}],
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "system_role_contract_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "policy_id": "system_responsibility_and_authority_v1",
            "summary": {
                "role_count": 2,
                "component_count": 1,
                "state_domain_count": 1,
                "control_surface_binding_count": 1,
                "exclusive_action_count": 1,
                "registry_role_coverage_ratio": 1.0,
                "authority_conflict_count": 0,
            },
            "blockers": [],
            "warnings": [],
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "summary": {
                "total_bots": 2,
                "active_bots": 2,
                "data_collection_active_bots": 2,
                "paper_live_data_enabled_bots": 1,
                "legacy_bootstrap_paper_bots": 0,
                "collection_until_standard_bots": 1,
                "standard_promoted_paper_bots": 0,
                "deletion_guard_ok": True,
            }
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "bot_organization_latest.json",
        {
            "ok": True,
            "overall_status": "ready_with_review_debt",
            "registry_bot_count": 2,
            "organized_bot_count": 2,
            "organization_coverage_ratio": 1.0,
            "explicit_sleeve_ratio": 0.5,
            "review_queue_count": 1,
            "structural_grade": "A+",
            "grade": "D",
            "invalid_assignment_bot_ids": [],
            "duplicate_bot_ids": [],
            "counts": {"sleeves": {"equity_core": 1, "ops_health": 1}},
            "bot_setup_summary": {"hardening": {"overall_status": "ready"}},
            "tripwire_summary": {
                "overall_status": "active_advisory",
                "active_tripwire_count": 2,
                "blocking_tripwire_count": 0,
                "severity_counts": {"advisory": 2},
                "category_counts": {"classification_quality": 2},
                "hardening": {"overall_status": "ready"},
                "active_tripwires": [
                    {
                        "tripwire_id": "review_debt_present",
                        "severity": "advisory",
                        "category": "classification_quality",
                        "owner": "operator_review",
                        "action": "prioritize_low_confidence_assignments_without_granting_execution",
                        "evidence_required": ["review_queue"],
                    },
                    {
                        "tripwire_id": "explicit_sleeve_coverage_low",
                        "severity": "advisory",
                        "category": "classification_quality",
                        "owner": "operator_review",
                        "action": "increase_explicit_sleeve_metadata_coverage_before_runtime_routing_changes",
                        "evidence_required": ["explicit_sleeve_ratio"],
                    },
                ],
                "blocking_tripwires": [],
            },
        },
    )
    _write_json(
        tmp_path / "config" / "sleeve_strategy_contracts_v1.json",
        {
            "policy_id": "sleeve_strategy_specialization_v1",
            "source_manifest": "config/sleeve_strategy_expansion.json",
            "sleeves": {"equity_core": {}, "day_trading": {}},
            "included_sleeves": ["options_on_futures_aggressive"],
            "derived_sleeve_policy": {"execution_eligible": False},
        },
    )
    _write_json(
        tmp_path / "config" / "sleeve_strategy_expansion.json",
        {
            "schema_version": 1,
            "sleeves": [
                {"name": "equity_core"},
                {"name": "day_trading"},
                {"name": "options_on_futures_aggressive"},
            ],
        },
    )
    contracts = {
        "sleeve::equity_core::trend::v1": _contract(
            "sleeve::equity_core::trend::v1", "equity_core"
        ),
        "sleeve::day_trading::vwap::v1": _contract(
            "sleeve::day_trading::vwap::v1", "day_trading"
        ),
        "sleeve::options_on_futures_aggressive::carry::v1": _contract(
            "sleeve::options_on_futures_aggressive::carry::v1",
            "options_on_futures_aggressive",
        ),
    }
    _write_json(
        tmp_path / "governance" / "research" / "sleeve_strategy_contracts_latest.json",
        {
            "policy_id": "sleeve_strategy_specialization_v1",
            "contract_count": len(contracts),
            "contracts": contracts,
            "authority_contract": {
                "can_create_intent": False,
                "can_reverse_intent": False,
                "can_increase_quantity": False,
                "can_allocate_capital": False,
                "can_change_labels": False,
                "can_grant_promotion": False,
                "can_submit_live_order": False,
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "paper_live_data_standard_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "counts_after": {
                "total_bots": 2,
                "active_bots": 2,
                "data_collection_active_bots": 2,
                "paper_live_data_enabled_bots": 1,
                "legacy_bootstrap_paper_bots": 0,
                "collection_until_standard_bots": 1,
                "standard_promoted_paper_bots": 0,
                "paper_execution_authority_bots": 0,
                "paper_probation_authority_bots": 1,
                "direct_execution_allowed_bots": 0,
                "live_trading_enabled_bots": 0,
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "runtime_gate_dashboard_latest.json",
        {
            "ok": False,
            "overall_status": "warn",
            "overall": {
                "status": "warn",
                "ok": False,
                "attention": [
                    "paper_execution_safety_guard_active",
                    "promotion_not_ready",
                ],
                "attention_tiers": {
                    "critical": [],
                    "degraded": [],
                    "watch": ["paper_execution_safety_guard_active"],
                    "advisory": ["promotion_not_ready"],
                },
            },
            "artifacts": {
                "all_sleeves_launcher": {"summary": {"paper_execution_ready": False}},
                "ingestion_storage_control": {
                    "summary": {
                        "overall_status": "ready",
                        "pressure_index": 0.1,
                        "estimated_total_drain_minutes": 0.1,
                    }
                },
                "external_backlog_drain": {
                    "summary": {
                        "overall_status": "ready",
                        "writer_busy": False,
                        "aged_candidate_files": 0,
                    }
                },
            },
        },
    )
    _write_json(
        tmp_path
        / "governance"
        / "health"
        / "bot_profitability_scalability_latest.json",
        {
            "ok": True,
            "overall_status": "ready_with_evidence_debt",
            "control_grade": "A+",
            "economic_and_scale_evidence_grade": "F",
            "evidence_debt": ["p01"],
            "candidate_binding": {"bound": False},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "paper_profitability_control_latest.json",
        {
            "ok": False,
            "overall_status": "blocked_missing_evidence",
            "paper_summary": {"ending_net_pnl_total": 0.0},
        },
    )
    _write_json(
        tmp_path
        / "governance"
        / "health"
        / "profitability_evidence_firewall_latest.json",
        {"ok": False, "overall_status": "blocked"},
    )
    _write_json(
        tmp_path / "governance" / "health" / "sleeve_scalability_selector_latest.json",
        {
            "ok": True,
            "overall_status": "ready_with_evidence_debt",
            "summary": {
                "application_eligible_sleeve_count": 0,
                "earned_scalability_goal_count": 0,
            },
            "evidence_debt": ["candidate_bound"],
        },
    )
    _write_json(
        tmp_path
        / "governance"
        / "health"
        / "schwab_account_snapshot_refresh_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "broker_truth_ok": True,
            "broker_truth_status": "ok",
            "broker_truth_v2_grade": "A",
            "account_count": 1,
            "failed_account_count": 0,
            "published_as_canonical": True,
            "broker_truth_mismatch_count": 0,
        },
    )
    _write_json(
        tmp_path
        / "governance"
        / "health"
        / "broker_truth_shared_snapshot_schwab_latest.json",
        {"broker": "schwab", "timestamp_utc": "2026-09-05T12:00:00+00:00"},
    )
    _write_json(
        tmp_path
        / "governance"
        / "health"
        / "schwab_broker_boundary_control_latest.json",
        {"ok": True, "overall_status": "ready", "broker": "schwab"},
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "pressure_index": 0.1,
            "bounded_recovery_contract": {"estimated_total_drain_minutes": 0.1},
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "external_backlog_drain_latest.json",
        {"ok": True, "overall_status": "ready", "writer_busy": False},
    )


def test_canonical_audit_keeps_evidence_debt_separate_from_drift(
    tmp_path: Path,
) -> None:
    _populate_project(tmp_path)

    payload = audit.build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["summary"]["degraded_count"] == 0
    assert payload["summary"]["critical_count"] == 0
    finding_ids = {finding["finding_id"] for finding in payload["findings"]}
    assert "bot_organization_active_tripwires_present" in finding_ids
    assert "bot_organization_review_debt_present" not in finding_ids
    assert "bot_organization_contains_non_strategy_sleeves" not in finding_ids
    assert "paper_standard_ready_execution_guarded" not in finding_ids
    assert "dashboard_warn_is_noncritical_attention" not in finding_ids
    assert "profitability_control_grade_separated_from_economic_evidence" in finding_ids
    assert "paper_profitability_missing_evidence" in finding_ids
    assert "no_application_eligible_sleeves" in finding_ids


def test_canonical_audit_fails_on_true_count_and_contract_drift(tmp_path: Path) -> None:
    _populate_project(tmp_path)
    role_report = json.loads(
        (
            tmp_path / "governance" / "health" / "system_role_contract_latest.json"
        ).read_text(encoding="utf-8")
    )
    role_report["summary"]["role_count"] = 99
    _write_json(
        tmp_path / "governance" / "health" / "system_role_contract_latest.json",
        role_report,
    )
    contract_report = json.loads(
        (
            tmp_path
            / "governance"
            / "research"
            / "sleeve_strategy_contracts_latest.json"
        ).read_text(encoding="utf-8")
    )
    contract_report["contracts"].pop("sleeve::day_trading::vwap::v1")
    contract_report["contract_count"] = len(contract_report["contracts"])
    _write_json(
        tmp_path / "governance" / "research" / "sleeve_strategy_contracts_latest.json",
        contract_report,
    )

    payload = audit.build_payload(tmp_path)

    assert payload["ok"] is False
    assert payload["summary"]["degraded_count"] >= 2
    finding_ids = {finding["finding_id"] for finding in payload["findings"]}
    assert "system_role_role_count_mismatch" in finding_ids
    assert "core_sleeves_missing_from_strategy_contracts" in finding_ids


def test_canonical_audit_flags_strategy_authority_leak_as_critical(
    tmp_path: Path,
) -> None:
    _populate_project(tmp_path)
    contract_report = json.loads(
        (
            tmp_path
            / "governance"
            / "research"
            / "sleeve_strategy_contracts_latest.json"
        ).read_text(encoding="utf-8")
    )
    contract_report["contracts"]["sleeve::equity_core::trend::v1"]["authority"][
        "can_submit_live_order"
    ] = True
    _write_json(
        tmp_path / "governance" / "research" / "sleeve_strategy_contracts_latest.json",
        contract_report,
    )

    payload = audit.build_payload(tmp_path)

    assert payload["ok"] is False
    assert payload["summary"]["critical_count"] == 1
    assert any(
        finding["finding_id"] == "strategy_contract_authority_leak"
        for finding in payload["findings"]
    )


def test_account_capability_builder_without_report_is_advisory(tmp_path: Path) -> None:
    _populate_project(tmp_path)
    script_path = tmp_path / "scripts" / "ops" / "schwab_account_capability_truth.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# builder exists\n", encoding="utf-8")

    payload = audit.build_payload(tmp_path)

    assert payload["ok"] is True
    assert any(
        finding["finding_id"] == "schwab_account_capability_truth_not_published"
        and finding["severity"] == "advisory"
        for finding in payload["findings"]
    )


def test_embedded_account_capability_satisfies_account_truth(tmp_path: Path) -> None:
    _populate_project(tmp_path)
    script_path = tmp_path / "scripts" / "ops" / "schwab_account_capability_truth.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# builder exists\n", encoding="utf-8")
    _write_json(
        tmp_path / "governance" / "health" / "account_position_study_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "accounts": [
                {
                    "account_policy_key": "schwab_roth_ira_primary",
                    "account_capability_truth": {
                        "operator_classification": {"live_execution_authority": False},
                        "canary_preflight": {"live_execution_authority": False},
                    },
                }
            ],
        },
    )

    payload = audit.build_payload(tmp_path)

    assert payload["ok"] is True
    finding_ids = {finding["finding_id"] for finding in payload["findings"]}
    assert "schwab_account_capability_truth_not_published" not in finding_ids
    metrics = payload["checks"]["broker_account_truth_representation"]["metrics"]
    assert metrics["embedded_account_capability_count"] == 1
    assert metrics["embedded_account_capability_covers_accounts"] is True
