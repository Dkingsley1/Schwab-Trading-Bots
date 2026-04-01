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
