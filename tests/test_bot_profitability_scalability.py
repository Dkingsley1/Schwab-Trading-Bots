from __future__ import annotations

import copy
import json
from pathlib import Path

from core import bot_profitability_scalability as control
from scripts.ops import artifact_freshness_slo
from scripts.ops import bot_profitability_scalability_control
from scripts.ops import runtime_artifact_refresh
from scripts.ops import runtime_gate_dashboard
from scripts.ops import source_mutation_guard


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _policy() -> dict:
    return json.loads(
        (PROJECT_ROOT / "config" / "bot_profitability_scalability_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _assignment(bot_id: str, *, cell: str, cluster: str) -> dict:
    return {
        "bot_id": bot_id,
        "sleeve_id": "equity",
        "sub_sleeve_id": "trend",
        "cell_id": cell,
        "correlation_cluster_id": cluster,
        "shadow_vote_eligible": True,
        "regime_profile_id": "test_profile",
        "regime_profile": {
            "schema_version": 1,
            "profile_id": "test_profile",
            "scope": "market_signal",
            "axes": {
                "market_direction": {"values": ["bull_trend"], "not_applicable": False},
                "volatility_state": {"values": ["normal"], "not_applicable": False},
                "liquidity_state": {"values": ["normal"], "not_applicable": False},
            },
        },
    }


def _observations(bot_id: str, *, offset: float = 0.0) -> list[dict]:
    rows = []
    for index in range(36):
        day = index % 6 + 1
        regime = "bull_normal" if index % 2 == 0 else "range_normal"
        outcome = 1.0 + offset + (index % 5) * 0.05
        rows.append(
            {
                "bot_id": bot_id,
                "decision_id": f"{bot_id}-{index}",
                "timestamp_utc": f"2026-08-{day:02d}T15:{index % 60:02d}:00+00:00",
                "day_utc": f"2026-08-{day:02d}",
                "candidate_bound": True,
                "profile": "equity",
                "strategy": "trend",
                "regime": regime,
                "weight_share": 1.0,
                "post_cost_pnl": outcome,
                "post_cost_return_bps": 8.0 + outcome,
                "notional": 100.0,
                "confidence": 0.8,
                "slippage_bps": 2.0,
                "spread_bps": 1.0,
                "latency_ms": 20.0,
                "partial_fill_ratio": 1.0,
                "tradeability": 0.9,
                "source": "independent_broker_paper",
            }
        )
    return rows


def _ready_artifacts() -> dict[str, dict]:
    return {
        "bot_organization_policy": {
            "regime_model": _policy_from_path("config/bot_organization_v1.json")["regime_model"]
        },
        "regime_context": {
            "overall_status": "ready",
            "regime_state": "bull normal liquid",
            "regime_axes": {
                "market_direction": ["bull_trend"],
                "volatility_state": ["normal"],
                "liquidity_state": ["normal"],
            },
        },
        "profitability_firewall": {
            "live_promotion_ready": True,
            "baseline_controls": [
                {"control_id": "08_multiple_testing_firewall", "evidence_ready": True},
                {"control_id": "09_oos_regime_lcb", "evidence_ready": True},
            ],
            "controls": [
                {"control_id": "h01_independent_fill_truth", "evidence_ready": True},
                {"control_id": "h04_locked_holdout_vault", "evidence_ready": True},
                {"control_id": "h05_adversarial_execution_replay", "evidence_ready": True},
            ],
        },
        "paper_execution_calibration": {
            "independent_samples": 100,
            "independent_evidence_ready": True,
        },
        "feature_store": {
            "ok": True,
            "strict_status": "ready",
            "contract_hashes": {"dataset_manifest_sha256": "abc"},
            "point_in_time_contract": {"seed_ready": True},
        },
        "runtime_throttle": {"ok": True, "throttle_profile": "sustain"},
        "resource_governor": {"ok": True},
        "cold_archive": {
            "ok": True,
            "archive_root": "/archive",
            "manifest_path": "/archive/manifest.jsonl",
            "reader_commands": ["archive-read"],
        },
        "sleeve_ingestion": {
            "data_tier_contract": {"core_priority": "1", "cold_budget": "quota_limited"}
        },
    }


def _policy_from_path(path: str) -> dict:
    return json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def _runtime_evidence() -> dict:
    return {
        "runtime_loop_process_count": 1,
        "runtime_checkpoint_count": 2,
        "order_idempotency_registry_count": 2,
        "decision_identity_coverage_ratio": 1.0,
        "duplicate_source_row_count": 0,
    }


def test_policy_is_complete_and_execution_free() -> None:
    policy = _policy()

    assert control.validate_policy(policy) == []

    unsafe = copy.deepcopy(policy)
    unsafe["safety_contract"]["live_execution_authority"] = True
    assert "safety_live_execution_authority_must_be_false" in control.validate_policy(unsafe)


def test_candidate_bound_profiles_cover_all_profitability_and_scale_controls() -> None:
    assignments = [
        _assignment("alpha", cell="equity/trend/a", cluster="equity/trend/a"),
        _assignment("beta", cell="equity/trend/b", cluster="equity/trend/b"),
    ]
    observations = _observations("alpha") + _observations("beta", offset=0.2)

    health, manifest = control.build_control_payload(
        _policy(), assignments, observations, _ready_artifacts(), _runtime_evidence()
    )

    assert health["ok"] is True
    assert health["control_grade"] == "A+"
    assert health["implemented_control_count"] == 16
    assert health["evidence_ready_control_count"] == 16
    assert health["economic_and_scale_evidence_grade"] == "A+"
    assert health["planned_active_bot_count"] == 2
    assert health["live_allocation_ready"] is True
    assert health["automatic_allocation_allowed"] is False
    assert manifest["activation_plan"]["application_allowed"] is False
    assert manifest["activation_plan"]["regime_compatible_bot_count"] == 2
    assert all(row["learned_preferred_regimes"] for row in manifest["profiles"])
    assert all(row["capacity_curve"]["maximum_supported_notional"] > 0 for row in manifest["profiles"])


def test_missing_economic_evidence_stays_debt_without_blocking_collection() -> None:
    assignments = [_assignment("alpha", cell="equity/trend/a", cluster="equity/trend/a")]
    artifacts = _ready_artifacts()
    artifacts["profitability_firewall"] = {"live_promotion_ready": False}
    artifacts["paper_execution_calibration"] = {
        "independent_samples": 0,
        "independent_evidence_ready": False,
    }

    health, manifest = control.build_control_payload(
        _policy(), assignments, [], artifacts, _runtime_evidence()
    )

    assert health["ok"] is True
    assert health["overall_status"] == "ready_with_evidence_debt"
    assert health["control_grade"] == "A+"
    assert health["economic_and_scale_evidence_grade"] != "A+"
    assert health["paper_collection_ready"] is True
    assert health["live_allocation_ready"] is False
    assert health["planned_active_bot_count"] == 0
    assert manifest["activation_plan"]["selected"] == []


def test_activation_plan_fails_closed_on_current_regime_mismatch() -> None:
    assignments = [_assignment("alpha", cell="equity/trend/a", cluster="equity/trend/a")]
    artifacts = _ready_artifacts()
    _, matching_manifest = control.build_control_payload(
        _policy(), assignments, _observations("alpha"), artifacts, _runtime_evidence()
    )
    artifacts["regime_context"]["regime_state"] = "bear high volatility illiquid"
    artifacts["regime_context"]["regime_axes"] = {
        "market_direction": ["bear_trend"],
        "volatility_state": ["high"],
        "liquidity_state": ["thin"],
    }

    health, manifest = control.build_control_payload(
        _policy(), assignments, _observations("alpha"), artifacts, _runtime_evidence()
    )

    profile = manifest["profiles"][0]
    assert health["planned_active_bot_count"] == 0
    assert profile["current_regime_compatibility"]["compatible"] is False
    assert profile["current_regime_compatibility"]["reason"] == "critical_regime_axis_mismatch"
    assert manifest["receipt_sha256"] != matching_manifest["receipt_sha256"]


def test_observation_extraction_is_candidate_bound_and_idempotent() -> None:
    row = {
        "timestamp_utc": "2026-08-02T14:00:00+00:00",
        "decision_id": "decision-1",
        "paper_profile": "default",
        "post_cost_pnl_delta": 4.0,
        "post_cost_return_bps": 8.0,
        "execution_notional": 100.0,
        "metadata": {
            "constituent_attribution": [
                {"bot_id": "alpha", "weight_share": 0.75, "confidence": 0.8},
                {"bot_id": "beta", "weight_share": 0.25, "confidence": 0.7},
            ]
        },
    }

    payload = control.extract_bot_observations(
        [row, row],
        known_bot_ids={"alpha", "beta"},
        candidate_cutoff_utc="2026-08-01T00:00:00+00:00",
    )

    assert payload["scan"]["duplicate_row_count"] == 1
    assert payload["scan"]["unique_decision_count"] == 1
    assert len(payload["observations"]) == 2
    assert sum(item["post_cost_pnl"] for item in payload["observations"]) == 4.0
    assert all(item["candidate_bound"] for item in payload["observations"])


def test_lazy_model_cache_loads_on_demand_and_evicts_by_limits_and_pressure() -> None:
    cache = control.LazyModelCache(
        maximum_models=2,
        maximum_bytes=20,
        inactive_ttl_seconds=5,
    )
    loads: list[str] = []

    def load(key: str) -> str:
        loads.append(key)
        return key

    assert cache.get("a", lambda: load("a"), estimated_bytes=8, now_monotonic=0) == "a"
    assert cache.get("a", lambda: load("a-again"), estimated_bytes=8, now_monotonic=1) == "a"
    assert loads == ["a"]
    cache.get("b", lambda: load("b"), estimated_bytes=8, now_monotonic=2)
    cache.get("c", lambda: load("c"), estimated_bytes=8, now_monotonic=3)
    assert cache.snapshot()["keys"] == ["b", "c"]
    assert cache.evict_inactive(now_monotonic=10) == ["b", "c"]
    cache.get("d", lambda: load("d"), estimated_bytes=8, memory_pressure="critical")
    assert cache.snapshot()["loaded_model_count"] == 0


def test_control_build_is_path_isolated_and_does_not_write(tmp_path: Path) -> None:
    policy = _policy()
    config_path = tmp_path / "config" / "bot_profitability_scalability_v1.json"
    hierarchy_path = tmp_path / "governance" / "bot_organization" / "bot_hierarchy_latest.json"
    config_path.parent.mkdir(parents=True)
    hierarchy_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(policy), encoding="utf-8")
    hierarchy_path.write_text(
        json.dumps({"assignments": [_assignment("alpha", cell="a", cluster="a")]}),
        encoding="utf-8",
    )

    health, manifest = bot_profitability_scalability_control.build_payload(
        tmp_path,
        config_path=config_path,
        process_inventory={"runtime_loop_process_count": 0, "processes": []},
    )

    assert health["ok"] is True
    assert health["control_grade"] == "A+"
    assert manifest["catalog_bot_count"] == 1
    assert not (tmp_path / "governance" / "health" / "bot_profitability_scalability_latest.json").exists()


def test_repository_wiring_requires_and_protects_the_integrated_control() -> None:
    refresh_steps = {row["name"]: row for row in runtime_artifact_refresh._step_specs(PROJECT_ROOT)}
    freshness = artifact_freshness_slo._artifact_contract(PROJECT_ROOT)
    dashboard = runtime_gate_dashboard._artifact_config(PROJECT_ROOT)
    ownership = json.loads(
        (PROJECT_ROOT / "config" / "control_surface_ownership_v1.json").read_text(encoding="utf-8")
    )
    owned = {str(row.get("resource_path") or "") for row in ownership.get("controls", [])}

    assert "bot_profitability_scalability_control" in refresh_steps
    assert freshness["bot_profitability_scalability_control"]["required"] is True
    assert dashboard["bot_profitability_scalability_control"]["required"] is True
    assert "core/bot_profitability_scalability.py" in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    assert "governance/health/bot_profitability_scalability_latest.json" in owned
    assert "governance/bot_organization/bot_profitability_scalability_latest.json" in owned


def test_dashboard_summary_keeps_control_and_economic_grades_separate() -> None:
    summary = runtime_gate_dashboard._artifact_summary(
        "bot_profitability_scalability_control",
        {
            "overall_status": "ready_with_evidence_debt",
            "control_grade": "A+",
            "economic_and_scale_evidence_grade": "C",
            "implemented_control_count": 16,
            "evidence_ready_control_count": 11,
            "catalog_bot_count": 1781,
            "ranked_bot_count": 20,
            "planned_active_bot_count": 4,
            "live_allocation_ready": False,
            "evidence_debt": ["p04", "p05"],
        },
    )

    assert summary["control_grade"] == "A+"
    assert summary["evidence_grade"] == "C"
    assert summary["live_allocation_ready"] is False
    assert summary["evidence_debt_count"] == 2
