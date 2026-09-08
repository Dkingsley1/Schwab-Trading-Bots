import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops import institutional_capability_control as control


NOW = datetime(2026, 8, 21, 14, 0, tzinfo=timezone.utc)
CANDIDATE_ID = "pc-test-g9"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _artifact(project_root: Path, relative: str, payload: dict, *, timestamp: datetime = NOW) -> None:
    _write(
        project_root / relative,
        {"timestamp_utc": timestamp.isoformat(), **payload},
    )


def _binding() -> dict:
    return {"candidate_id": CANDIDATE_ID, "bound": True}


def _build_fixture(project_root: Path) -> None:
    _write(
        project_root / "config" / "institutional_capability_control_v1.json",
        {
            "policy_id": "institutional_capability_control_v1",
            "provider_policy": {
                "authoritative_provider_family_target_min": 15,
                "authoritative_provider_family_target_max": 30,
            },
            "pillars": [{"pillar_id": pillar, "title": pillar} for pillar in (
                "scientific_research_platform",
                "market_visibility_and_data_lineage",
                "independent_execution_evidence",
                "selection_bias_and_overfit_control",
                "resource_routing_and_role_separation",
                "market_access_risk_controls",
            )],
            "conditional_external_entitlements": [
                {
                    "entitlement_id": "depth",
                    "provider": "Direct depth",
                    "status": "optional_unsubscribed",
                    "required_only_for_live_families": ["market_making"],
                }
            ],
        },
    )
    _write(
        project_root / "governance" / "runtime" / "production_candidate_state.json",
        {
            "candidate_id": CANDIDATE_ID,
            "generation": 9,
            "scope_windows_started_utc": {
                "execution": "2026-08-21T12:00:00+00:00",
                "data": "2026-08-21T12:00:00+00:00",
                "dependencies": "2026-08-21T12:00:00+00:00",
                "strategy": "2026-08-21T12:00:00+00:00",
            },
        },
    )
    for relative in (
        "core/execution_simulator.py",
        "scripts/ops/independent_fill_evidence_acquisition.py",
        "scripts/paper_execution_calibration_report.py",
        "core/live_execution_controls.py",
        "scripts/global_risk_killswitch.py",
    ):
        path = project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# owner\n", encoding="utf-8")
    _artifact(
        project_root,
        "governance/research/sleeve_strategy_specialization_latest.json",
        {
            "ok": True,
            "candidate_binding": _binding(),
            "contract_coverage": {"strategy_count": 10, "complete_contract_count": 10},
            "strategy_library": {"strategy_count": 100, "complete_contract_count": 100},
            "quality_summary": {"validated_good_count": 0},
        },
    )
    _artifact(
        project_root,
        "governance/research/quantitative_challenger_latest.json",
        {"ok": True, "candidate_binding": _binding()},
    )
    _artifact(project_root, "governance/health/point_in_time_event_store_latest.json", {"ok": True})
    _artifact(
        project_root,
        "governance/experiments/immutable_experiment_ledger_latest.json",
        {
            "append_only_ready": True,
            "ledger_row_count": 5,
            "signed_row_count": 5,
            "latest_exact_replay_ready": False,
            "latest_attestation_ready": False,
        },
    )
    _artifact(
        project_root,
        "governance/research/multiple_testing_guard_latest.json",
        {
            "contract_present": True,
            "family_size": 100,
            "correction_method": "benjamini_hochberg_fdr",
            "statistical_evidence_ready": True,
            "statistical_evidence_blockers": [],
            "candidate_binding": _binding(),
        },
    )
    _artifact(
        project_root,
        "governance/health/source_verification_latest.json",
        {"ok": True, "sources": [{"ok": True, "fresh": True} for _ in range(15)]},
    )
    _artifact(
        project_root,
        "governance/health/collector_capability_control_latest.json",
        {
            "ok": True,
            "summary": {
                "plane_count": 10,
                "capability_count": 40,
                "assignment_count": 50,
                "producer_count": 15,
                "required_capability_usable_ratio": 0.9,
                "runtime_live_ready_route_count": 0,
            },
            "coverage_debt": {
                "blocks_guarded_paper_soak": False,
                "candidate_blocking_gap_count": 2,
            },
        },
    )
    _artifact(
        project_root,
        "governance/health/independent_fill_evidence_acquisition_latest.json",
        {
            "candidate_binding": _binding(),
            "candidate_eligible_ledger_records": 30,
            "accepted_ledger_records": 40,
            "conflict_count": 0,
            "control_contract": {"exact_candidate_identity_required": True},
        },
    )
    _artifact(
        project_root,
        "governance/health/paper_execution_calibration_latest.json",
        {
            "candidate_binding": _binding(),
            "minimum_independent_samples": 30,
            "independent_evidence_ready": True,
            "metrics": {"mae_bps": 2.0},
        },
    )
    _artifact(
        project_root,
        "governance/health/autonomic_resource_governor_latest.json",
        {
            "ok": True,
            "overall_status": "advisory",
            "budgets": {
                "runtime_pressure_source": {
                    "runtime_hot": False,
                    "memory_pressure_level": "normal",
                }
            },
        },
    )
    _artifact(
        project_root,
        "governance/health/system_role_contract_latest.json",
        {"ok": True, "summary": {"role_count": 15, "authority_conflict_count": 0}},
    )
    _artifact(project_root, "governance/health/control_surface_ownership_latest.json", {"ok": True})
    _artifact(
        project_root,
        "governance/health/live_order_ledger_control_latest.json",
        {"ok": True, "live_execution_authority": False},
    )
    _artifact(
        project_root,
        "governance/risk/risk_service_boundary_latest.json",
        {
            "independent_service_boundary": {"service_count": 5, "service_isolation_ready": True},
            "services": {"pre_trade_service": {"evaluated_orders": 0}},
            "input_health": {"sources_ready": False},
        },
    )
    _artifact(
        project_root,
        "governance/health/live_reconciliation_slo_latest.json",
        {"ok": True, "metrics": {"reconcile_events": 0}},
    )
    _artifact(project_root, "governance/health/live_canary_control_latest.json", {"live_execution_allowed": False})


def test_control_separates_paper_readiness_from_evidence_and_entitlements(tmp_path: Path) -> None:
    _build_fixture(tmp_path)

    payload = control.build_payload(tmp_path, now=NOW)

    assert payload["overall_status"] == "ready_with_evidence_debt"
    assert payload["ok"] is True
    assert payload["summary"]["implementation_ready_count"] == 6
    assert payload["summary"]["paper_soak_ready_count"] == 6
    assert payload["summary"]["candidate_evidence_ready_count"] < 6
    assert payload["live_promotion_ready"] is False
    assert payload["provider_policy"]["ten_thousand_sources_required"] is False
    assert payload["conditional_external_entitlements"][0]["blocks_guarded_paper_soak"] is False


def test_candidate_mismatch_fails_only_the_affected_paper_pillar(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    fill_path = tmp_path / "governance" / "health" / "independent_fill_evidence_acquisition_latest.json"
    fill = json.loads(fill_path.read_text(encoding="utf-8"))
    fill["candidate_binding"] = {"candidate_id": "pc-old", "bound": True}
    fill_path.write_text(json.dumps(fill), encoding="utf-8")

    payload = control.build_payload(tmp_path, now=NOW)

    assert payload["overall_status"] == "paper_attention"
    assert payload["paper_soak_ready"] is False
    assert "independent_execution_evidence" in payload["paper_blockers"]
    market_data = next(row for row in payload["pillars"] if row["pillar_id"] == "market_visibility_and_data_lineage")
    assert market_data["paper_soak_ready"] is True


def test_stale_local_artifact_gets_bounded_refresh_route(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    source_path = tmp_path / "governance" / "health" / "source_verification_latest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["timestamp_utc"] = (NOW - timedelta(hours=2)).isoformat()
    source_path.write_text(json.dumps(source), encoding="utf-8")

    payload = control.build_payload(tmp_path, now=NOW)

    action = next(row for row in payload["bounded_local_refresh_actions"] if row["artifact_id"] == "source_verification")
    assert action["state"] == "stale"
    assert action["automatic_live_authority"] is False
    assert payload["control_contract"]["subscriptions_fills_attestations_and_operator_release_are_not_self_healable"] is True


def test_external_runtime_heat_is_advisory_when_guarded_paper_lanes_remain_available(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    resource_path = tmp_path / "governance" / "health" / "autonomic_resource_governor_latest.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    resource["budgets"] = {
        "runtime_pressure_source": {
            "runtime_hot": True,
            "memory_pressure_level": "normal",
            "attribution": {"paper_execution_pressure_dominant": False},
        },
        "collectors": {"mode": "normal"},
        "live_loops": {"mode": "protected_read_only"},
    }
    resource_path.write_text(json.dumps(resource), encoding="utf-8")

    payload = control.build_payload(tmp_path, now=NOW)

    pillar = next(row for row in payload["pillars"] if row["pillar_id"] == "resource_routing_and_role_separation")
    assert pillar["paper_soak_ready"] is True
    assert pillar["metrics"]["runtime_hot_advisory_only"] is True
    assert pillar["metrics"]["paper_pressure_dominant"] is False


def test_paper_dominant_runtime_pressure_blocks_resource_paper_pillar(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    resource_path = tmp_path / "governance" / "health" / "autonomic_resource_governor_latest.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    resource["budgets"] = {
        "runtime_pressure_source": {
            "runtime_hot": True,
            "memory_pressure_level": "normal",
            "attribution": {"paper_execution_pressure_dominant": True},
        },
        "collectors": {"mode": "normal"},
        "live_loops": {"mode": "protected_read_only"},
    }
    resource_path.write_text(json.dumps(resource), encoding="utf-8")

    payload = control.build_payload(tmp_path, now=NOW)

    pillar = next(row for row in payload["pillars"] if row["pillar_id"] == "resource_routing_and_role_separation")
    assert pillar["paper_soak_ready"] is False
    assert "paper_runtime_pressure_hot" in pillar["paper_blockers"]
