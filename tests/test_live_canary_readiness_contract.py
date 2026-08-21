from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops import live_canary_readiness_contract as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_ready_artifacts(project_root: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    health = project_root / "governance" / "health"
    debt_recovery = {
        "state": "cleared_and_proven",
        "debt_cleared": True,
        "live_promotion_ready": True,
        "baseline_debt_amount": 100.0,
        "remaining_debt_amount": 0.0,
        "recovery_progress_norm": 1.0,
        "candidate_attribution": {
            "candidate_id": "candidate-ready",
            "sample_count": 40,
            "observed_days": 5,
            "total_candidate_attributed_pnl": 110.0,
        },
        "candidate_proof": {"ready": True},
        "risk_budget": {"new_entries_paused": False},
        "promotion_blockers": [],
    }
    scaling_contract = {
        "active": True,
        "mode": "candidate_bound_sleeve_strategy_scaling_v1",
        "paper_only": True,
        "live_execution_allowed": False,
        "source_ready": True,
        "entry_only": True,
        "keep_sells_and_reduce_only_paths_open": True,
        "fail_closed_on_missing_or_mismatched_candidate_evidence": True,
        "global_entry_size_cap_norm": 1.0,
        "maximum_above_baseline_entry_size_multiplier_norm": 1.10,
        "candidate_binding": {
            "candidate_id": "candidate-ready",
            "candidate_binding_valid": True,
        },
        "profile_controls": {
            "default": {
                "tier": "validated_baseline",
                "block_new_entries": False,
            }
        },
        "above_baseline_ready_count": 0,
        "scale_up_ready": False,
        "hard_limits": {
            "never_scale_from_loss_recovery_pressure": True,
            "never_use_martingale": True,
            "never_average_down_for_recovery": True,
            "never_scale_above_1_10x_from_profitability_evidence": True,
            "portfolio_and_execution_risk_caps_remain_authoritative": True,
        },
    }
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "raw_profitability_grade": "A",
            "paper_debt_recovery_contract": debt_recovery,
            "sleeve_strategy_profitability_scaling_contract": scaling_contract,
        },
    )
    _write_json(
        health / "paper_runtime_profitability_controls_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "raw_profitability_grade": "A",
            "paper_debt_recovery_contract": debt_recovery,
            "sleeve_strategy_profitability_scaling_contract": scaling_contract,
        },
    )
    _write_json(
        health / "paper_execution_truth_layer_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "ok": True,
            "failed_checks": [],
            "gates": {
                "paper_broker_truth_reconciliation": {
                    "ok": True,
                    "score": 100.0,
                    "mismatch_count": 0,
                    "source_verification_ok": True,
                }
            },
        },
    )
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True, "failed_checks": []})
    _write_json(health / "paper_400_ramp_latest.json", {"timestamp_utc": now, "overall_status": "ready", "stage": "armed", "blockers": []})
    _write_json(health / "broker_readiness_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ready_for_open": True, "auth_ok": True, "network_ok": True, "token_expires_in_seconds": 2400})
    _write_json(health / "schwab_auth_supervisor_latest.json", {"timestamp_utc": now, "overall_status": "ready", "token": {"expires_in_seconds": 2400}})
    _write_json(health / "auth_lease_manager_latest.json", {"timestamp_utc": now, "overall_status": "ready", "lease_state": "healthy", "expires_in_seconds": 2400})
    _write_json(health / "ingestion_storage_control_latest.json", {"timestamp_utc": now, "overall_status": "ready", "pressure_index": 0.01, "backpressure": {"total_pending_lines": 0}})
    _write_json(health / "health_gates_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(health / "promotion_quality_gate_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(health / "paper_performance_latest.json", {"timestamp_utc": now, "overall_status": "ready", "ok": True})
    _write_json(
        health / "health_fast_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"status": "ready", "ok": True},
                "collector_repair": {"status": "ready", "ok": True},
                "platform_repair": {"status": "ready", "ok": True},
            },
            "process_watchdog": {
                "all_sleeves_effective_runtime": {
                    "ok": True,
                    "status": "ready",
                    "child_process_count": 16,
                }
            },
            "platform_intelligence": {"overall_status": "ready"},
            "platform_brain_v4": {"overall_status": "ready"},
            "platform_brain_v5": {"overall_status": "ready"},
            "platform_stabilization_quality": {"overall_status": "ready"},
            "system_architecture_hardening": {"overall_status": "ready"},
        },
    )
    _write_json(
        health / "live_money_readiness_contract_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "blocked",
            "faithful_live_money_ready": False,
            "sections": [
                {
                    "section_id": "risk_controls",
                    "ready": True,
                    "grade": "A+",
                    "grade_floor_met": True,
                    "blockers": [],
                }
            ],
        },
    )
    _write_json(
        health / "live_canary_control_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "blocked",
            "recommended_mode": "validate_only",
            "target_canary_weight": 0.01,
            "applied_canary_weight": 0.01,
            "canary_weight_ok": True,
        },
    )
    _write_json(
        health / "production_readiness_control_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "guarded",
            "ok": True,
            "domain_count": 9,
            "ready_domain_count": 8,
            "blocked_domain_count": 0,
            "live_runtime_promotion_allowed": False,
            "live_money_production_bar_ready": True,
            "live_money_canary_consideration_ready": True,
            "blockers": [],
        },
    )
    _write_json(
        health / "use_mode_compliance_guard_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "use_mode": "personal",
            "personal_use": {"grade": "A+", "perfect_personal_use_ready": True, "personal_live_money_ready": False},
            "commercial_use": {
                "commercial_use_intent_detected": False,
                "commercial_clearance_status": "not_requested_personal_mode",
                "blockers": [],
            },
            "authority_boundaries": {
                "does_not_enable_live_execution": True,
                "live_execution_authority": False,
                "customer_funds_allowed": False,
                "customer_order_execution_allowed": False,
                "raw_profitability_is_not_live_money_proof": True,
            },
        },
    )
    _write_json(
        health / "commercial_readiness_control_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "ready",
            "commercial_product_mode": "personal_only",
            "commercial_intent": False,
            "commercial_release_ready": False,
            "commercial_release_blocked": False,
            "grade": "A+",
            "blockers": [],
            "authority_boundaries": {
                "live_execution_authority": False,
                "customer_funds_allowed": False,
                "customer_order_execution_allowed": False,
            },
        },
    )


def test_live_canary_continuity_separates_operational_truth_from_replay_evidence(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _seed_ready_artifacts(project_root)
    health = project_root / "governance" / "health"
    now = datetime.now(timezone.utc).isoformat()
    _write_json(
        health / "paper_execution_truth_layer_latest.json",
        {
            "timestamp_utc": now,
            "overall_status": "blocked",
            "ok": False,
            "failed_checks": ["decision_replay_harness"],
            "input_freshness": {"operational_inputs_fresh": True},
            "gates": {
                **{gate_id: {"ok": True, "status": "ready"} for gate_id in src.PAPER_OPERATIONAL_GATE_IDS},
                "decision_replay_harness": {"ok": False, "status": "blocked"},
            },
        },
    )
    monkeypatch.setattr(src.source_mutation_guard, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "dirty_count": 0, "dirty_entries": []})
    monkeypatch.setattr(src.production_flow_smoke, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "failed_checks": []})

    payload = src.build_payload(project_root)

    paper_gate = next(gate for gate in payload["gates"] if gate["gate_id"] == "sleeve_paper_trading_continuity")
    paper_milestone = next(row for row in payload["live_money_canary_milestones"] if row["milestone_id"] == "m02_live_like_paper_execution")
    assert paper_gate["ready"] is True
    assert paper_gate["evidence"]["paper_truth_evidence_pending"] is True
    assert paper_milestone["ready"] is False
    assert "paper_execution_evidence_not_ready" in paper_milestone["blockers"]


def test_paper_truth_operational_state_accepts_explicit_non_blocking_freshness_advisory() -> None:
    gates = {
        gate_id: {"ok": True, "status": "ready"}
        for gate_id in src.PAPER_OPERATIONAL_GATE_IDS
    }
    gates["artifact_freshness_guard"] = {
        "ok": False,
        "status": "warn",
        "grade_blocking": False,
        "advisory_only": True,
    }

    ready, blockers, gate_states = src._paper_truth_operational_state(
        {
            "input_freshness": {"operational_inputs_fresh": True},
            "gates": gates,
        }
    )

    assert ready is True
    assert blockers == []
    assert gate_states["artifact_freshness_guard"] is True


def test_paper_truth_operational_state_fails_closed_on_blocking_freshness_warning() -> None:
    gates = {
        gate_id: {"ok": True, "status": "ready"}
        for gate_id in src.PAPER_OPERATIONAL_GATE_IDS
    }
    gates["artifact_freshness_guard"] = {
        "ok": False,
        "status": "warn",
        "grade_blocking": True,
    }

    ready, blockers, gate_states = src._paper_truth_operational_state(
        {
            "input_freshness": {"operational_inputs_fresh": True},
            "gates": gates,
        }
    )

    assert ready is False
    assert blockers == ["paper_truth_operational_gate_not_ready:artifact_freshness_guard"]
    assert gate_states["artifact_freshness_guard"] is False


def test_live_canary_readiness_contract_blocks_raw_d_grade(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _seed_ready_artifacts(project_root)
    _write_json(project_root / "governance" / "health" / "paper_profitability_control_latest.json", {"overall_status": "ready", "raw_profitability_grade": "D"})
    monkeypatch.setattr(src.source_mutation_guard, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "dirty_count": 0, "dirty_entries": []})
    monkeypatch.setattr(src.production_flow_smoke, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "failed_checks": []})

    payload = src.build_payload(project_root, out_path=project_root / "governance" / "health" / "live_canary_readiness_contract_latest.json")

    raw_gate = next(gate for gate in payload["gates"] if gate["gate_id"] == "raw_profitability_posture")
    assert payload["overall_status"] == "blocked"
    assert payload["live_money_canary_blocked"] is True
    assert raw_gate["ready"] is False
    assert "raw_profitability_hard_block_below_C" in raw_gate["blockers"]
    assert "no raw D-grade posture" in payload["infrastructure_message"]


def test_live_canary_readiness_contract_uses_configured_auth_floor(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _seed_ready_artifacts(project_root)
    health = project_root / "governance" / "health"
    config = project_root / "config" / "live_canary_readiness_contract.json"
    _write_json(config, {"auth_min_expires_in_seconds": 1200, "sustained_window_hours": 168})
    for artifact in ("broker_readiness_latest.json", "schwab_auth_supervisor_latest.json", "auth_lease_manager_latest.json"):
        payload = json.loads((health / artifact).read_text(encoding="utf-8"))
        payload["token_expires_in_seconds"] = 1500
        payload["expires_in_seconds"] = 1500
        payload["token"] = {"expires_in_seconds": 1500}
        payload["lease_state"] = "healthy"
        _write_json(health / artifact, payload)
    monkeypatch.setattr(src.source_mutation_guard, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "dirty_count": 0, "dirty_entries": []})
    monkeypatch.setattr(src.production_flow_smoke, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "failed_checks": []})

    payload = src.build_payload(project_root, config_path=config)

    auth_gate = next(gate for gate in payload["gates"] if gate["gate_id"] == "auth_token_continuity")
    assert auth_gate["ready"] is True
    assert auth_gate["blockers"] == []


def test_live_canary_readiness_contract_can_clear_after_sustained_window(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    out_path = project_root / "governance" / "health" / "live_canary_readiness_contract_latest.json"
    _seed_ready_artifacts(project_root)
    monkeypatch.setattr(src.source_mutation_guard, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "dirty_count": 0, "dirty_entries": []})
    monkeypatch.setattr(src.production_flow_smoke, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "failed_checks": []})
    _write_json(
        out_path,
        {
            "overall_status": "blocked",
            "continuous_all_gates_ready_since_utc": (datetime.now(timezone.utc) - timedelta(hours=730)).isoformat(),
        },
    )

    payload = src.build_payload(project_root, out_path=out_path)

    assert payload["overall_status"] == "ready"
    assert payload["live_canary_money_ready"] is True
    assert payload["sustained_window"]["sustained_window_met"] is True
    assert payload["ready_gate_count"] == payload["gate_count"]
    assert payload["required_live_money_canary_milestones_ready"] is True
    assert payload["ready_required_milestone_count"] == payload["required_milestone_count"]


def test_live_canary_readiness_contract_blocks_uncleared_paper_recovery_balance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    _seed_ready_artifacts(project_root)
    health = project_root / "governance" / "health"
    paper_profit = json.loads(
        (health / "paper_profitability_control_latest.json").read_text(encoding="utf-8")
    )
    paper_profit["paper_debt_recovery_contract"] = {
        "state": "recovering",
        "debt_cleared": False,
        "live_promotion_ready": False,
        "baseline_debt_amount": 20_000.0,
        "remaining_debt_amount": 12_000.0,
        "recovery_progress_norm": 0.4,
        "promotion_blockers": ["paper_recovery_balance_not_cleared"],
    }
    _write_json(health / "paper_profitability_control_latest.json", paper_profit)
    monkeypatch.setattr(
        src.source_mutation_guard,
        "build_payload",
        lambda _root: {"ok": True, "overall_status": "ready", "dirty_count": 0, "dirty_entries": []},
    )
    monkeypatch.setattr(
        src.production_flow_smoke,
        "build_payload",
        lambda _root: {"ok": True, "overall_status": "ready", "failed_checks": []},
    )

    payload = src.build_payload(project_root)
    gate = next(row for row in payload["gates"] if row["gate_id"] == "paper_debt_recovery_proof")

    assert gate["ready"] is False
    assert gate["evidence"]["remaining_debt_amount"] == 12_000.0
    assert "paper_recovery_balance_not_cleared" in gate["blockers"]
    assert payload["live_canary_money_ready"] is False


def test_live_canary_readiness_contract_rejects_scaling_contract_with_live_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    _seed_ready_artifacts(project_root)
    health = project_root / "governance" / "health"
    paper_profit = json.loads(
        (health / "paper_profitability_control_latest.json").read_text(encoding="utf-8")
    )
    scaling = paper_profit["sleeve_strategy_profitability_scaling_contract"]
    scaling["paper_only"] = False
    scaling["live_execution_allowed"] = True
    scaling["fail_closed_on_missing_or_mismatched_candidate_evidence"] = False
    _write_json(health / "paper_profitability_control_latest.json", paper_profit)
    monkeypatch.setattr(
        src.source_mutation_guard,
        "build_payload",
        lambda _root: {"ok": True, "overall_status": "ready", "dirty_count": 0, "dirty_entries": []},
    )
    monkeypatch.setattr(
        src.production_flow_smoke,
        "build_payload",
        lambda _root: {"ok": True, "overall_status": "ready", "failed_checks": []},
    )

    payload = src.build_payload(project_root)
    gate = next(
        row
        for row in payload["gates"]
        if row["gate_id"] == "candidate_bound_sleeve_strategy_scaling"
    )

    assert gate["ready"] is False
    assert "sleeve_strategy_scaling_not_paper_only" in gate["blockers"]
    assert "sleeve_strategy_scaling_claims_live_authority" in gate["blockers"]
    assert "sleeve_strategy_scaling_not_fail_closed" in gate["blockers"]
    assert payload["live_canary_money_ready"] is False


def test_live_canary_readiness_contract_blocks_until_live_money_milestones_clear(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    out_path = project_root / "governance" / "health" / "live_canary_readiness_contract_latest.json"
    _seed_ready_artifacts(project_root)
    monkeypatch.setattr(src.source_mutation_guard, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "dirty_count": 0, "dirty_entries": []})
    monkeypatch.setattr(src.production_flow_smoke, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "failed_checks": []})
    _write_json(
        out_path,
        {
            "overall_status": "blocked",
            "continuous_all_gates_ready_since_utc": (datetime.now(timezone.utc) - timedelta(hours=170)).isoformat(),
        },
    )

    payload = src.build_payload(project_root, out_path=out_path)

    assert payload["overall_status"] == "blocked"
    assert payload["live_canary_money_ready"] is False
    assert "live_money_canary_milestones_not_ready" in payload["blockers"]
    assert "m01_continuous_soak_no_hard_blockers" in payload["blocked_milestones"]
    milestone = next(row for row in payload["live_money_canary_milestones"] if row["milestone_id"] == "m01_continuous_soak_no_hard_blockers")
    assert milestone["ready"] is False
    assert "continuous_soak_below_720h" in milestone["blockers"]


def test_live_canary_readiness_contract_blocks_oversized_initial_canary_weight(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    out_path = project_root / "governance" / "health" / "live_canary_readiness_contract_latest.json"
    _seed_ready_artifacts(project_root)
    _write_json(
        project_root / "governance" / "health" / "live_canary_control_latest.json",
        {
            "overall_status": "blocked",
            "recommended_mode": "validate_only",
            "target_canary_weight": 0.08,
            "applied_canary_weight": 0.08,
        },
    )
    _write_json(
        out_path,
        {
            "overall_status": "blocked",
            "continuous_all_gates_ready_since_utc": (datetime.now(timezone.utc) - timedelta(hours=730)).isoformat(),
        },
    )
    monkeypatch.setattr(src.source_mutation_guard, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "dirty_count": 0, "dirty_entries": []})
    monkeypatch.setattr(src.production_flow_smoke, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "failed_checks": []})

    payload = src.build_payload(project_root, out_path=out_path)

    assert payload["overall_status"] == "blocked"
    assert "m08_microscopic_canary_plan" in payload["blocked_milestones"]
    milestone = next(row for row in payload["live_money_canary_milestones"] if row["milestone_id"] == "m08_microscopic_canary_plan")
    assert milestone["ready"] is False
    assert "initial_canary_weight_above_0.0100" in milestone["blockers"]


def test_live_canary_readiness_contract_blocks_when_production_bar_not_ready(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    out_path = project_root / "governance" / "health" / "live_canary_readiness_contract_latest.json"
    _seed_ready_artifacts(project_root)
    _write_json(
        project_root / "governance" / "health" / "production_readiness_control_latest.json",
        {
            "overall_status": "blocked",
            "ok": False,
            "domain_count": 9,
            "ready_domain_count": 7,
            "blocked_domain_count": 1,
            "live_money_production_bar_ready": False,
            "live_money_canary_consideration_ready": False,
            "blockers": ["live_money_production_bar:immutable_evidence_store_missing"],
        },
    )
    _write_json(
        out_path,
        {
            "overall_status": "blocked",
            "continuous_all_gates_ready_since_utc": (datetime.now(timezone.utc) - timedelta(hours=730)).isoformat(),
        },
    )
    monkeypatch.setattr(src.source_mutation_guard, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "dirty_count": 0, "dirty_entries": []})
    monkeypatch.setattr(src.production_flow_smoke, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "failed_checks": []})

    payload = src.build_payload(project_root, out_path=out_path)

    assert payload["overall_status"] == "blocked"
    assert payload["live_canary_money_ready"] is False
    assert "m10_live_money_production_bar" in payload["blocked_milestones"]
    assert "m09_explainable_trade_permission" in payload["blocked_milestones"]
    milestone = next(row for row in payload["live_money_canary_milestones"] if row["milestone_id"] == "m10_live_money_production_bar")
    assert milestone["ready"] is False
    assert "live_money_production_bar_not_ready" in milestone["blockers"]


def test_live_canary_readiness_contract_blocks_commercial_customer_execution_boundary(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    out_path = project_root / "governance" / "health" / "live_canary_readiness_contract_latest.json"
    _seed_ready_artifacts(project_root)
    _write_json(
        out_path,
        {
            "overall_status": "blocked",
            "continuous_all_gates_ready_since_utc": (datetime.now(timezone.utc) - timedelta(hours=730)).isoformat(),
        },
    )
    _write_json(
        project_root / "governance" / "health" / "use_mode_compliance_guard_latest.json",
        {
            "overall_status": "blocked",
            "use_mode": "commercial_software",
            "personal_use": {"grade": "A+", "perfect_personal_use_ready": True, "personal_live_money_ready": False},
            "commercial_use": {
                "commercial_use_intent_detected": True,
                "commercial_clearance_status": "blocked_requires_compliance_review",
                "blockers": ["broker_dealer_review_not_approved", "broker_dealer_customer_execution_review_required"],
            },
            "authority_boundaries": {
                "does_not_enable_live_execution": True,
                "live_execution_authority": False,
                "customer_order_execution_allowed": False,
            },
        },
    )
    monkeypatch.setattr(src.source_mutation_guard, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "dirty_count": 0, "dirty_entries": []})
    monkeypatch.setattr(src.production_flow_smoke, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "failed_checks": []})

    payload = src.build_payload(project_root, out_path=out_path)

    assert payload["overall_status"] == "blocked"
    assert payload["live_canary_money_ready"] is False
    assert "m11_use_mode_and_commercial_boundary" in payload["blocked_milestones"]
    milestone = next(row for row in payload["live_money_canary_milestones"] if row["milestone_id"] == "m11_use_mode_and_commercial_boundary")
    assert milestone["ready"] is False
    assert "commercial_boundary_blockers_present" in milestone["blockers"]
    assert "broker_dealer_review_not_approved" in milestone["evidence"]["commercial_blockers"]


def test_live_canary_readiness_contract_blocks_commercial_readiness_framework(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    out_path = project_root / "governance" / "health" / "live_canary_readiness_contract_latest.json"
    _seed_ready_artifacts(project_root)
    _write_json(
        out_path,
        {
            "overall_status": "blocked",
            "continuous_all_gates_ready_since_utc": (datetime.now(timezone.utc) - timedelta(hours=730)).isoformat(),
        },
    )
    _write_json(
        project_root / "governance" / "health" / "commercial_readiness_control_latest.json",
        {
            "overall_status": "blocked",
            "commercial_product_mode": "paid_signals_newsletter",
            "commercial_intent": True,
            "commercial_release_ready": False,
            "commercial_release_blocked": True,
            "blockers": ["marketing_claim_control:marketing_review_not_approved"],
            "authority_boundaries": {"live_execution_authority": False},
        },
    )
    monkeypatch.setattr(src.source_mutation_guard, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "dirty_count": 0, "dirty_entries": []})
    monkeypatch.setattr(src.production_flow_smoke, "build_payload", lambda _root: {"ok": True, "overall_status": "ready", "failed_checks": []})

    payload = src.build_payload(project_root, out_path=out_path)

    assert payload["overall_status"] == "blocked"
    assert "m11_use_mode_and_commercial_boundary" in payload["blocked_milestones"]
    milestone = next(row for row in payload["live_money_canary_milestones"] if row["milestone_id"] == "m11_use_mode_and_commercial_boundary")
    assert "commercial_readiness_status=blocked" in milestone["blockers"]
    assert "commercial_release_blocked" in milestone["blockers"]
    assert "marketing_claim_control:marketing_review_not_approved" in milestone["evidence"]["commercial_readiness_blockers"]
