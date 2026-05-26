import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.promotion_quality_gate as promotion_quality_gate


def test_promotion_quality_gate_resolves_stale_daily_verify_failures_from_fresher_artifacts() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {"promote_ok": True, "considered_bots": 5, "fail_share": 0.2},
        {"ok": False, "failed_checks": ["new_bot_graduation_gate", "replay_hash_registry_guard", "promotion_quality_gate"]},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=True,
    )

    assert ok is True
    assert failed_checks == []
    assert details["daily_verify_ok"] is True
    assert details["daily_verify_unresolved_failed_checks"] == []
    assert sorted(details["daily_verify_resolved_failed_checks"]) == [
        "new_bot_graduation_gate",
        "promotion_quality_gate",
        "replay_hash_registry_guard",
    ]


def test_promotion_quality_gate_ignores_recovered_incomplete_daily_verify_run() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {"promote_ok": True, "considered_bots": 5, "fail_share": 0.0},
        {"ok": False, "failed_checks": ["incomplete_run_recovered"]},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=True,
    )

    assert ok is True
    assert failed_checks == []
    assert details["daily_verify_ok"] is True
    assert details["daily_verify_unresolved_failed_checks"] == []
    assert details["daily_verify_resolved_failed_checks"] == ["incomplete_run_recovered"]


def test_promotion_quality_gate_requires_feature_manifest_packet_and_probation_guards() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {"promote_ok": True, "considered_bots": 5, "fail_share": 0.0},
        {"ok": True, "failed_checks": []},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        feature_store_manifest={
            "ok": True,
            "strict_ok": True,
            "point_in_time_contract": {"complete": True},
            "contract_hashes": {"dataset_manifest_sha256": "a" * 64},
        },
        new_bot_admission_guard={"ok": True},
        champion_challenger_probation_guard={"ok": True},
        promotion_packet={"ok": True},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=True,
    )

    assert ok is True
    assert failed_checks == []
    assert details["feature_store_manifest_ready"] is True
    assert details["promotion_packet_ok"] is True


def test_promotion_quality_gate_accepts_seed_ready_feature_store_contract() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {"promote_ok": True, "considered_bots": 5, "fail_share": 0.0},
        {"ok": True, "failed_checks": []},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        feature_store_manifest={
            "ok": True,
            "strict_ok": False,
            "strict_seed_ready": True,
            "point_in_time_contract": {"complete": False, "seed_ready": True},
            "contract_hashes": {"dataset_manifest_sha256": "a" * 64},
        },
        new_bot_admission_guard={"ok": True},
        champion_challenger_probation_guard={"ok": True},
        promotion_packet={"ok": True},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=False,
    )

    assert ok is True
    assert failed_checks == []
    assert details["feature_store_manifest_ready"] is True


def test_promotion_quality_gate_treats_owner_replay_and_reconciliation_as_advisory_when_scope_is_idle() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        promotion_gate={"promote_ok": False, "considered_bots": 0, "fail_share": 0.0},
        daily_verify={"ok": False, "failed_checks": ["promotion_quality_gate"]},
        graduation_gate={"ok": True, "graduation_scope_active_count": 0},
        leak_overfit={"ok": True},
        replay_gate={"ok": False},
        replay_hash_registry_gate={"ok": True},
        reconciliation_slo={"ok": False},
        feature_store_manifest={
            "ok": True,
            "strict_ok": False,
            "strict_seed_ready": True,
            "point_in_time_contract": {"complete": False, "seed_ready": True},
            "contract_hashes": {"dataset_manifest_sha256": "a" * 64},
        },
        bot_support_owner_guard={"ok": False},
        golden_replay_regression_guard={"ok": False},
        new_bot_admission_guard={"ok": True},
        champion_challenger_probation_guard={"ok": False},
        promotion_packet={"ok": False},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=True,
    )

    assert ok is True
    assert failed_checks == []
    assert details["promotion"]["promotion_scope_active"] is False


def test_promotion_quality_gate_resolves_new_daily_verify_failures_when_artifacts_recover() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {"promote_ok": True, "considered_bots": 5, "fail_share": 0.0},
        {
            "ok": False,
            "failed_checks": [
                "new_bot_admission_guard",
                "champion_challenger_probation_guard",
                "promotion_packet_builder",
            ],
        },
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        feature_store_manifest={
            "ok": True,
            "strict_ok": True,
            "point_in_time_contract": {"complete": True},
            "contract_hashes": {"dataset_manifest_sha256": "a" * 64},
        },
        new_bot_admission_guard={"ok": True},
        champion_challenger_probation_guard={"ok": True},
        promotion_packet={"ok": True},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=True,
    )

    assert ok is True
    assert failed_checks == []
    assert details["daily_verify_unresolved_failed_checks"] == []
    assert sorted(details["daily_verify_resolved_failed_checks"]) == [
        "champion_challenger_probation_guard",
        "new_bot_admission_guard",
        "promotion_packet_builder",
    ]


def test_promotion_quality_gate_treats_idle_promotion_scope_as_non_blocking() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {"promote_ok": False, "considered_bots": 0, "fail_share": 0.0},
        {"ok": False, "failed_checks": ["promotion_quality_gate"]},
        {"ok": True, "graduation_scope_active_count": 0},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        feature_store_manifest={
            "ok": True,
            "strict_ok": True,
            "point_in_time_contract": {"complete": True},
            "contract_hashes": {"dataset_manifest_sha256": "a" * 64},
        },
        new_bot_admission_guard={"ok": True},
        champion_challenger_probation_guard={"ok": True},
        promotion_packet={"ok": False},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=True,
    )

    assert ok is True
    assert failed_checks == []
    assert details["promotion"]["promotion_scope_active"] is False
    assert details["daily_verify_unresolved_failed_checks"] == []


def test_promotion_quality_gate_can_ignore_recursive_daily_verify_failures() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {"promote_ok": False, "considered_bots": 0, "fail_share": 0.0},
        {"ok": False, "failed_checks": ["promotion_quality_gate", "unhandled_exception"]},
        {"ok": True, "graduation_scope_active_count": 0},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        feature_store_manifest={
            "ok": True,
            "strict_ok": True,
            "point_in_time_contract": {"complete": True},
            "contract_hashes": {"dataset_manifest_sha256": "a" * 64},
        },
        new_bot_admission_guard={"ok": True},
        champion_challenger_probation_guard={"ok": True},
        promotion_packet={"ok": False},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=True,
        ignore_daily_verify_failed_checks={"promotion_quality_gate", "unhandled_exception"},
    )

    assert ok is True
    assert failed_checks == []
    assert details["daily_verify_unresolved_failed_checks"] == []
    assert sorted(details["daily_verify_resolved_failed_checks"]) == [
        "promotion_quality_gate",
        "unhandled_exception",
    ]


def test_promotion_quality_gate_breaks_recursive_daily_verify_loop_in_active_scope() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {"promote_ok": False, "considered_bots": 3, "fail_share": 0.0},
        {"ok": False, "failed_checks": ["promotion_quality_gate"]},
        {"ok": True, "graduation_scope_active_count": 0},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        feature_store_manifest={
            "ok": True,
            "strict_ok": True,
            "point_in_time_contract": {"complete": True},
            "contract_hashes": {"dataset_manifest_sha256": "a" * 64},
        },
        new_bot_admission_guard={"ok": True},
        champion_challenger_probation_guard={"ok": True},
        promotion_packet={"ok": False},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=True,
    )

    assert ok is False
    assert "daily_verify_not_ok" not in failed_checks
    assert "promotion_gate_blocked" in failed_checks
    assert "insufficient_considered_bots" in failed_checks
    assert details["daily_verify_unresolved_failed_checks"] == []
    assert details["daily_verify_resolved_failed_checks"] == ["promotion_quality_gate"]


def test_promotion_quality_gate_resolves_idle_promotion_packet_failure() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {"promote_ok": False, "considered_bots": 0, "fail_share": 0.0},
        {"ok": False, "failed_checks": ["promotion_packet_builder"]},
        {"ok": True, "graduation_scope_active_count": 0},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        feature_store_manifest={
            "ok": True,
            "strict_ok": True,
            "point_in_time_contract": {"complete": True},
            "contract_hashes": {"dataset_manifest_sha256": "a" * 64},
        },
        new_bot_admission_guard={"ok": True},
        champion_challenger_probation_guard={"ok": True},
        promotion_packet={"ok": False},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=True,
    )

    assert ok is True
    assert failed_checks == []
    assert details["daily_verify_unresolved_failed_checks"] == []
    assert details["daily_verify_resolved_failed_checks"] == ["promotion_packet_builder"]


def test_promotion_quality_gate_resolves_fresh_snapshot_and_freshness_failures() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {"promote_ok": True, "considered_bots": 5, "fail_share": 0.0},
        {"ok": False, "failed_checks": ["snapshot_coverage_sentinel", "data_source_divergence_bot", "artifact_freshness"]},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        snapshot_coverage_guard={"ok": True},
        data_source_divergence_guard={"ok": True},
        artifact_freshness_guard={"ok": True},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=True,
    )

    assert ok is True
    assert failed_checks == []
    assert details["daily_verify_unresolved_failed_checks"] == []
    assert sorted(details["daily_verify_resolved_failed_checks"]) == [
        "artifact_freshness",
        "data_source_divergence_bot",
        "snapshot_coverage_sentinel",
    ]


def test_promotion_quality_gate_resolves_remediated_daily_verify_in_idle_scope() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {"promote_ok": False, "considered_bots": 0, "fail_share": 0.0},
        {
            "ok": False,
            "failed_checks": [
                "feature_store_manifest",
                "nightly_resilience_check",
                "state_snapshot_drill",
                "db_integrity",
            ],
        },
        {"ok": True, "graduation_scope_active_count": 0},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        feature_store_manifest={
            "ok": True,
            "strict_seed_ready": True,
            "point_in_time_contract": {"seed_ready": True},
            "contract_hashes": {"dataset_manifest_sha256": "a" * 64},
        },
        new_bot_admission_guard={"ok": True},
        nightly_resilience_guard={"ok": False},
        state_snapshot_drill={"ok": True},
        db_integrity_guard={"ok": True},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=True,
    )

    assert ok is True
    assert failed_checks == []
    assert details["daily_verify_unresolved_failed_checks"] == []
    assert sorted(details["daily_verify_resolved_failed_checks"]) == [
        "db_integrity",
        "feature_store_manifest",
        "nightly_resilience_check",
        "state_snapshot_drill",
    ]
