import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.promotion_quality_gate as promotion_quality_gate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_promotion_quality_gate_resolves_stale_daily_verify_failures_from_fresher_artifacts() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {"promote_ok": True, "considered_bots": 5, "fail_share": 0.2},
        {
            "ok": False,
            "failed_checks": [
                "new_bot_graduation_gate",
                "paper_reconciliation_slo_guard",
                "replay_hash_registry_guard",
                "promotion_quality_gate",
            ],
        },
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        paper_reconciliation_slo_guard={"ok": True},
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
        "paper_reconciliation_slo_guard",
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


def test_promotion_quality_gate_main_writes_operating_contract(
    tmp_path: Path, monkeypatch
) -> None:
    files = {
        "promotion": tmp_path / "promotion.json",
        "daily": tmp_path / "daily.json",
        "graduation": tmp_path / "graduation.json",
        "owner": tmp_path / "owner.json",
        "admission": tmp_path / "admission.json",
        "leak": tmp_path / "leak.json",
        "replay": tmp_path / "replay.json",
        "replay_hash": tmp_path / "replay_hash.json",
        "feature_store": tmp_path / "feature_store.json",
        "schema": tmp_path / "schema.json",
        "golden": tmp_path / "golden.json",
        "cohort": tmp_path / "cohort.json",
        "probation": tmp_path / "probation.json",
        "reconciliation": tmp_path / "reconciliation.json",
        "paper_reconciliation": tmp_path / "paper_reconciliation.json",
        "paper_truth": tmp_path / "paper_truth.json",
        "paper_calibration": tmp_path / "paper_calibration.json",
        "promotion_packet": tmp_path / "promotion_packet.json",
        "snapshot": tmp_path / "snapshot.json",
        "divergence": tmp_path / "divergence.json",
        "freshness": tmp_path / "freshness.json",
        "nightly": tmp_path / "nightly.json",
        "state": tmp_path / "state.json",
        "db": tmp_path / "db.json",
        "queue": tmp_path / "queue.json",
        "resource": tmp_path / "resource.json",
        "out": tmp_path / "promotion_quality_gate_latest.json",
    }
    _write_json(
        files["promotion"],
        {"promote_ok": False, "considered_bots": 1, "fail_share": 0.0},
    )
    _write_json(files["daily"], {"ok": True, "failed_checks": []})
    for key in (
        "graduation",
        "owner",
        "admission",
        "leak",
        "replay",
        "replay_hash",
        "schema",
        "golden",
        "cohort",
        "probation",
        "reconciliation",
        "paper_reconciliation",
        "snapshot",
        "divergence",
        "freshness",
        "nightly",
        "state",
        "db",
        "queue",
        "resource",
    ):
        _write_json(files[key], {"ok": True})
    _write_json(
        files["feature_store"],
        {
            "ok": True,
            "strict_ok": True,
            "point_in_time_contract": {"complete": True},
            "contract_hashes": {"dataset_manifest_sha256": "a" * 64},
        },
    )
    _write_json(files["paper_truth"], {"ok": True, "promotion_ready": True})
    _write_json(
        files["paper_calibration"],
        {"ok": True, "independent_evidence_ready": True},
    )
    _write_json(files["promotion_packet"], {"ok": True})

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promotion_quality_gate.py",
            "--promotion-gate-file",
            str(files["promotion"]),
            "--daily-verify-file",
            str(files["daily"]),
            "--graduation-file",
            str(files["graduation"]),
            "--bot-support-owner-file",
            str(files["owner"]),
            "--new-bot-admission-file",
            str(files["admission"]),
            "--leak-overfit-file",
            str(files["leak"]),
            "--replay-file",
            str(files["replay"]),
            "--replay-hash-registry-file",
            str(files["replay_hash"]),
            "--feature-store-manifest",
            str(files["feature_store"]),
            "--schema-compatibility-file",
            str(files["schema"]),
            "--golden-replay-file",
            str(files["golden"]),
            "--cohort-drift-file",
            str(files["cohort"]),
            "--probation-guard-file",
            str(files["probation"]),
            "--reconciliation-file",
            str(files["reconciliation"]),
            "--paper-reconciliation-file",
            str(files["paper_reconciliation"]),
            "--paper-execution-truth-layer-file",
            str(files["paper_truth"]),
            "--paper-execution-calibration-file",
            str(files["paper_calibration"]),
            "--promotion-packet-file",
            str(files["promotion_packet"]),
            "--snapshot-coverage-file",
            str(files["snapshot"]),
            "--data-source-divergence-file",
            str(files["divergence"]),
            "--artifact-freshness-file",
            str(files["freshness"]),
            "--nightly-resilience-file",
            str(files["nightly"]),
            "--state-snapshot-drill-file",
            str(files["state"]),
            "--db-integrity-file",
            str(files["db"]),
            "--execution-queue-stress-file",
            str(files["queue"]),
            "--resource-guard-file",
            str(files["resource"]),
            "--out-file",
            str(files["out"]),
            "--json",
        ],
    )

    assert promotion_quality_gate.main() == 2
    payload = json.loads(files["out"].read_text(encoding="utf-8"))

    assert payload["overall_status"] == "blocked"
    assert payload["promotion_ready"] is False
    assert payload["recommended_actions"]
    assert payload["operating_contract"]["complete"] is True
    assert "automatic_live_promotion" in payload["operating_contract"][
        "blocked_authority"
    ]


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


def test_promotion_quality_gate_uses_effective_considered_floor() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {
            "promote_ok": True,
            "considered_bots": 2,
            "fail_share": 0.0,
            "effective_thresholds": {"min_considered_bots": 2},
        },
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
    assert details["promotion"]["min_considered_bots"] == 2
    assert details["promotion"]["configured_min_considered_bots"] == 4


def test_promotion_quality_gate_scopes_new_bot_admission_to_promotion_candidates() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {
            "promote_ok": True,
            "considered_bots": 1,
            "fail_share": 0.0,
            "effective_thresholds": {"min_considered_bots": 1},
            "considered_bot_ids": ["brain_refinery_v10_seasonal"],
            "pass_examples": [{"bot_id": "brain_refinery_v10_seasonal"}],
        },
        {"ok": False, "failed_checks": ["new_bot_admission_guard", "execution_queue_stress_bot"]},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        new_bot_admission_guard={
            "ok": False,
            "blocking_candidates": [{"bot_id": "brain_refinery_v13_choppy"}],
        },
        execution_queue_stress_guard={"ok": True},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=False,
    )

    assert ok is True
    assert failed_checks == []
    assert details["daily_verify_unresolved_failed_checks"] == []
    assert details["new_bot_admission_relevant_blocking_ids"] == []


def test_promotion_quality_gate_allows_targeted_pass_when_global_graduation_floor_is_behind() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {
            "promote_ok": True,
            "considered_bots": 2,
            "fail_share": 0.0,
            "effective_thresholds": {"min_considered_bots": 2},
            "considered_bot_ids": ["brain_refinery_v10_seasonal", "brain_refinery_v17_mixed_regime"],
            "pass_examples": [{"bot_id": "brain_refinery_v10_seasonal"}],
        },
        {"ok": False, "failed_checks": ["new_bot_graduation_gate"]},
        {
            "ok": False,
            "maturity": {"mature_bots": 8, "mature_pass_bots": 1, "mature_pass_rate": 0.125},
        },
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        new_bot_admission_guard={"ok": True},
        promotion_packet={"ok": True},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=False,
    )

    assert ok is True
    assert failed_checks == []
    assert details["graduation_ok"] is False
    assert details["graduation_effective_ok"] is True
    assert details["daily_verify_resolved_failed_checks"] == ["new_bot_graduation_gate"]


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


def test_promotion_quality_gate_resolves_resource_guard_when_current_guard_is_clean() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {"promote_ok": True, "considered_bots": 5, "fail_share": 0.2},
        {"ok": False, "failed_checks": ["resource_guard"]},
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
        resource_guard={"resource_guard_ok": True},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=True,
    )

    assert ok is True
    assert failed_checks == []
    assert details["daily_verify_unresolved_failed_checks"] == []
    assert details["daily_verify_resolved_failed_checks"] == ["resource_guard"]


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
