from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.master_grandmaster_evidence import (
    synthesize_master_grandmaster_evidence,
    validate_policy,
)
from core.regime_taxonomy import classify_regime_profile
from scripts.ops import (
    artifact_freshness_slo,
    master_grandmaster_evidence_control,
    runtime_artifact_refresh,
    runtime_gate_dashboard,
    source_mutation_guard,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 11, 16, 0, tzinfo=timezone.utc)
TIMESTAMP = NOW.isoformat()


def _policy() -> dict:
    return json.loads(
        (PROJECT_ROOT / "config" / "master_grandmaster_evidence_v2.json").read_text(
            encoding="utf-8"
        )
    )


def _organization_policy() -> dict:
    return json.loads(
        (PROJECT_ROOT / "config" / "bot_organization_v1.json").read_text(encoding="utf-8")
    )


def _profile(bot_id: str, direction: str = "bull_trend") -> dict:
    model = _organization_policy()["regime_model"]
    return classify_regime_profile(
        row={
            "bot_id": bot_id,
            "bot_role": "signal_sub_bot",
            "preferred_regimes": [],
            "regime_axes": {
                "market_direction": [direction],
                "volatility_state": ["normal"],
                "liquidity_state": ["deep"],
                "macro_state": ["growth_expansion"],
                "rates_credit_state": ["neutral"],
                "correlation_state": ["stable"],
                "event_phase": ["continuous"],
                "market_session": ["intraday"],
            },
        },
        module_spec={},
        classification_text="fixture signal bot",
        raw_role="signal_sub_bot",
        role_id="signal",
        sub_sleeve_id="trend_and_momentum",
        horizon_id="daily_to_multiday",
        model=model,
    )


def _assignment(bot_id: str, sleeve_id: str, direction: str = "bull_trend") -> dict:
    profile = _profile(bot_id, direction)
    return {
        "bot_id": bot_id,
        "sleeve_id": sleeve_id,
        "sub_sleeve_id": "trend_and_momentum",
        "cohort_id": "daily_to_multiday_bull_normal",
        "role_id": "signal",
        "active": True,
        "regime_scope": "market_signal",
        "shadow_vote_eligible": True,
        "correlation_cluster_id": f"{sleeve_id}/trend/daily",
        "classification_confidence": 1.0,
        "needs_review": False,
        "review_reasons": [],
        "regime_profile_id": profile["profile_id"],
        "regime_profile": profile,
    }


def _inputs() -> dict:
    assignments = [
        _assignment("alpha_equity", "equity_core"),
        _assignment("alpha_options", "options_flow"),
    ]
    receipt = "fixture-assignment-receipt"
    return {
        "bot_organization_health": {
            "timestamp_utc": TIMESTAMP,
            "ok": True,
            "overall_status": "ready",
            "registry_bot_count": len(assignments),
            "organized_bot_count": len(assignments),
            "organization_coverage_ratio": 1.0,
            "regime_axis_coverage_ratio": 1.0,
            "regime_axis_specificity_ratio": 1.0,
            "assignment_receipt_sha256": receipt,
        },
        "bot_hierarchy": {
            "timestamp_utc": TIMESTAMP,
            "assignment_count": len(assignments),
            "assignment_receipt_sha256": receipt,
            "regime_model_id": "multi_axis_regime_taxonomy_v1",
            "assignments": assignments,
        },
        "regime_payload": {
            "timestamp_utc": TIMESTAMP,
            "overall_status": "ready",
            "regime_state": "risk_on_trend",
            "regime_axes": {
                "market_direction": ["bull_trend"],
                "volatility_state": ["normal"],
                "liquidity_state": ["deep"],
                "macro_state": ["growth_expansion"],
                "rates_credit_state": ["neutral"],
                "correlation_state": ["stable"],
                "event_phase": ["continuous"],
                "market_session": ["intraday"],
            },
        },
        "paper_truth": {
            "timestamp_utc": TIMESTAMP,
            "ok": True,
            "overall_status": "ready",
            "score": 100.0,
            "sleeve_scorecards": [],
        },
        "profitability_evidence": {
            "timestamp_utc": TIMESTAMP,
            "ok": True,
            "overall_status": "ready_with_evidence_debt",
            "economic_evidence_score": 80.0,
            "live_promotion_ready": False,
        },
        "source_verification": {
            "timestamp_utc": TIMESTAMP,
            "ok": True,
            "overall_status": "ready",
            "source_evidence_score": 100.0,
        },
        "runtime_throttle": {
            "timestamp_utc": TIMESTAMP,
            "ok": True,
            "overall_status": "advisory",
        },
        "account_positions": {
            "timestamp_utc": TIMESTAMP,
            "ok": True,
            "position_count": 2,
        },
        "execution_calibration": {
            "timestamp_utc": TIMESTAMP,
            "ok": True,
            "overall_status": "evidence_pending",
            "independent_samples": 0,
            "independent_evidence_ready": False,
        },
        "sleeve_profitability": {
            "timestamp_utc": TIMESTAMP,
            "ok": True,
            "overall_status": "ready",
            "top_sleeves": [],
            "bottom_sleeves": [],
        },
    }


def _synthesize(inputs: dict | None = None) -> dict:
    values = inputs or _inputs()
    return synthesize_master_grandmaster_evidence(
        policy=_policy(),
        regime_model=_organization_policy()["regime_model"],
        now=NOW,
        **values,
    )


def test_policy_is_shadow_only_and_fails_closed_on_authority() -> None:
    policy = _policy()
    assert validate_policy(policy) == []

    unsafe = copy.deepcopy(policy)
    unsafe["safety_contract"]["direct_live_order_authority"] = True
    assert (
        "master_grandmaster_safety_direct_live_order_authority_must_be_false"
        in validate_policy(unsafe)
    )


def test_synthesis_separates_paper_coordination_from_live_evidence() -> None:
    result = _synthesize()

    assert result["ok"] is True
    assert result["structural_grade"] == "A+"
    assert result["paper_coordination_ready"] is True
    assert result["human_live_review_evidence_ready"] is False
    assert result["integrity_blockers"] == []
    assert result["operational_holds"] == []
    assert "profitability_evidence_ready" in result["promotion_blockers"]
    assert "independent_execution_evidence_ready" in result["promotion_blockers"]
    assert result["sleeve_master_count"] == 2
    assert all(row["status"] == "ready_shadow" for row in result["sleeve_masters"])


def test_runtime_pressure_holds_coordination_without_corrupting_structural_grade() -> None:
    inputs = _inputs()
    inputs["runtime_throttle"]["ok"] = False
    inputs["runtime_throttle"]["overall_status"] = "degraded"

    result = _synthesize(inputs)

    assert result["ok"] is True
    assert result["structural_grade"] == "A+"
    assert result["overall_status"] == "operational_hold"
    assert result["paper_coordination_ready"] is False
    assert result["operational_holds"] == [
        "master_grandmaster_runtime_capacity_not_ready"
    ]


def test_stale_required_hierarchy_fails_closed() -> None:
    inputs = _inputs()
    inputs["bot_hierarchy"]["timestamp_utc"] = (
        NOW - timedelta(days=2)
    ).isoformat()

    result = _synthesize(inputs)

    assert result["ok"] is False
    assert result["structural_grade"] == "F"
    assert result["overall_status"] == "blocked_integrity"
    assert (
        "master_grandmaster_required_source_not_fresh:bot_hierarchy"
        in result["integrity_blockers"]
    )


def test_future_dated_required_source_fails_closed() -> None:
    inputs = _inputs()
    inputs["paper_truth"]["timestamp_utc"] = (NOW + timedelta(minutes=6)).isoformat()

    result = _synthesize(inputs)

    check = result["source_checks"]["paper_truth"]
    assert check["timestamp_valid"] is False
    assert check["future_skew_minutes"] == 6.0
    assert (
        "master_grandmaster_required_source_not_fresh:paper_truth"
        in result["integrity_blockers"]
    )


def test_critical_regime_mismatch_is_guarded_and_auditable() -> None:
    inputs = _inputs()
    inputs["regime_payload"]["regime_axes"].update(
        {
            "market_direction": ["bear_trend"],
            "volatility_state": ["crisis"],
            "liquidity_state": ["dislocated"],
        }
    )

    result = _synthesize(inputs)

    assert all(row["status"] == "regime_guarded" for row in result["sleeve_masters"])
    assert all(
        row["regime_compatibility"]["incompatible_bot_count"] == 1
        for row in result["sleeve_masters"]
    )
    assert all(
        row["regime_compatibility"]["hard_mismatch_examples"]
        for row in result["sleeve_masters"]
    )


def test_output_is_deterministic_and_has_no_execution_authority() -> None:
    first = _synthesize()
    second = _synthesize()

    assert first["evidence_epoch"] == second["evidence_epoch"]
    assert first["sleeve_masters"] == second["sleeve_masters"]
    assert first["authority"]["paper_order_authority"] is False
    assert first["authority"]["live_order_authority"] is False
    assert first["authority"]["order_payload_created"] is False
    assert first["authority"]["automatic_promotion_authority"] is False
    assert first["grand_master"]["automatic_live_promotion_allowed"] is False
    assert all(row["authority"]["live_order_authority"] is False for row in first["sleeve_masters"])


def test_control_build_is_path_isolated_and_does_not_write(tmp_path: Path) -> None:
    values = _inputs()
    paths: dict[str, Path] = {}
    for name, payload in values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    policy_path = tmp_path / "master_policy.json"
    organization_policy_path = tmp_path / "organization_policy.json"
    policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
    organization_policy_path.write_text(json.dumps(_organization_policy()), encoding="utf-8")
    packet_out = tmp_path / "packets.json"

    health, catalog = master_grandmaster_evidence_control.build_payload(
        tmp_path,
        policy_path=policy_path,
        organization_policy_path=organization_policy_path,
        bot_organization_health_path=paths["bot_organization_health"],
        bot_hierarchy_path=paths["bot_hierarchy"],
        regime_context_path=paths["regime_payload"],
        paper_truth_path=paths["paper_truth"],
        profitability_evidence_path=paths["profitability_evidence"],
        source_verification_path=paths["source_verification"],
        runtime_throttle_path=paths["runtime_throttle"],
        account_positions_path=paths["account_positions"],
        execution_calibration_path=paths["execution_calibration"],
        sleeve_profitability_path=paths["sleeve_profitability"],
        packet_out_path=packet_out,
        now=NOW,
    )

    assert health["ok"] is True
    assert health["sleeve_master_summary"]["catalog_path"] == str(packet_out)
    assert "sleeve_masters" not in health
    assert catalog["sleeve_master_count"] == 2
    assert len(catalog["sleeve_masters"]) == 2
    assert health["publication_receipt"]["receipt_sha256"] == catalog[
        "publication_receipt_sha256"
    ]
    assert not packet_out.exists()


def test_repository_build_covers_every_organized_bot() -> None:
    health, catalog = master_grandmaster_evidence_control.build_payload(PROJECT_ROOT)

    assert health["ok"] is True
    assert health["structural_grade"] == "A+"
    assert health["organized_bot_count"] >= 1000
    assert catalog["organized_bot_count"] == health["organized_bot_count"]
    assert catalog["sleeve_master_count"] == len(catalog["sleeve_masters"])
    assert catalog["authority"]["live_order_authority"] is False


def test_repository_wiring_requires_fresh_owned_evidence() -> None:
    refresh_steps = {row["name"]: row for row in runtime_artifact_refresh._step_specs(PROJECT_ROOT)}
    freshness = artifact_freshness_slo._artifact_contract(PROJECT_ROOT)
    dashboard = runtime_gate_dashboard._artifact_config(PROJECT_ROOT)
    ownership = json.loads(
        (PROJECT_ROOT / "config" / "control_surface_ownership_v1.json").read_text(
            encoding="utf-8"
        )
    )
    owned_resources = {str(row.get("resource_path") or "") for row in ownership["controls"]}
    master_step = refresh_steps["master_grandmaster_evidence_v2"]
    freshness_step = refresh_steps["artifact_freshness_slo_post_master"]

    assert "profitability_evidence_firewall" in master_step["depends_on"]
    assert "runtime_throttle_control_post_settlement_verified" in master_step["depends_on"]
    assert freshness_step["depends_on"] == [
        "master_grandmaster_evidence_v2",
        "control_surface_ownership",
        "system_role_contract",
    ]
    step_names = [row["name"] for row in runtime_artifact_refresh._step_specs(PROJECT_ROOT)]
    assert step_names.index("master_grandmaster_evidence_v2") < step_names.index(
        "artifact_freshness_slo_post_master"
    )
    profitability_scope = {
        row["name"]
        for row in runtime_artifact_refresh._select_scope_specs(
            runtime_artifact_refresh._step_specs(PROJECT_ROOT), "profitability"
        )
    }
    assert "master_grandmaster_evidence_v2" in profitability_scope
    assert "artifact_freshness_slo_post_master" in profitability_scope
    assert freshness["master_grandmaster_evidence_v2"]["required"] is True
    assert dashboard["master_grandmaster_evidence_v2"]["required"] is True
    assert "core/master_grandmaster_evidence.py" in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    assert (
        "scripts/ops/master_grandmaster_evidence_control.py"
        in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    )
    assert "governance/health/master_grandmaster_evidence_v2_latest.json" in owned_resources
    assert "governance/master_grandmaster/evidence_packets_v2_latest.json" in owned_resources


def test_dashboard_summary_preserves_authority_and_evidence_distinctions() -> None:
    payload = _synthesize()
    payload["sleeve_master_summary"] = {
        "status_counts": {"ready_shadow": 2},
        "grade_counts": {"A+": 2},
    }

    summary = runtime_gate_dashboard._artifact_summary(
        "master_grandmaster_evidence_v2", payload
    )

    assert summary["structural_grade"] == "A+"
    assert summary["paper_coordination_ready"] is True
    assert summary["human_live_review_evidence_ready"] is False
    assert summary["automatic_live_promotion_allowed"] is False
    assert summary["master_status_counts"] == {"ready_shadow": 2}


def test_refresh_status_normalizes_evidence_debt_but_not_integrity_failure() -> None:
    advisory = runtime_artifact_refresh._step_status(
        {
            "rc": 0,
            "payload": {"ok": True, "overall_status": "ready_with_evidence_debt"},
        },
        name="master_grandmaster_evidence_v2",
    )
    integrity_failure = runtime_artifact_refresh._step_status(
        {
            "rc": 2,
            "payload": {"ok": False, "overall_status": "blocked_integrity"},
        },
        name="master_grandmaster_evidence_v2",
    )

    assert advisory == "ready_advisory"
    assert integrity_failure == "blocked_integrity"
    assert (
        "master_grandmaster_evidence_v2_not_ok"
        in runtime_gate_dashboard._DEGRADED_ATTENTION
    )
