import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.daily_auto_verify as daily_auto_verify


def test_nightly_resilience_check_uses_slow_timeout_budget() -> None:
    assert daily_auto_verify._timeout_for_check("nightly_resilience_check", 300) == 300
    assert daily_auto_verify._timeout_for_check("replay_preopen_sanity", 300) == 45
    assert daily_auto_verify._timeout_for_check("session_ready", 300) == daily_auto_verify.DEFAULT_CMD_TIMEOUT_SEC


def test_daily_auto_verify_resource_guard_uses_collection_profile() -> None:
    cmd = daily_auto_verify._resource_guard_check_cmd()

    assert "--profile" in cmd
    assert cmd[cmd.index("--profile") + 1] == "collection"


def test_promotion_quality_gate_check_ignores_recursive_daily_verify_failures() -> None:
    source = (PROJECT_ROOT / "scripts" / "daily_auto_verify.py").read_text(encoding="utf-8")

    assert '--ignore-daily-verify-check' in source
    assert '"promotion_packet_builder"' in source
    assert '"nightly_resilience_check"' in source
    assert '"promotion_quality_gate"' in source
    assert '"unhandled_exception"' in source


def test_daily_auto_verify_accepts_unsigned_seed_ready_promotion_packet_as_non_operational() -> None:
    stdout = json.dumps(
        {
            "committee_packet_seed_ready": True,
            "signing_material_ready": False,
            "signature": {"status": "missing_signing_key"},
            "gate_results": {
                "training_success_confirmed": True,
                "feature_store_manifest_strict_ok": True,
                "new_bot_admission_ok": True,
            },
        }
    )

    assert daily_auto_verify._promotion_packet_builder_ok(2, stdout, "") is True


def test_daily_auto_verify_accepts_signed_seed_ready_promotion_packet_as_non_operational() -> None:
    stdout = json.dumps(
        {
            "committee_packet_seed_ready": True,
            "signing_material_ready": True,
            "trained_models_complete": True,
            "signature": {"status": "verified", "verified": True},
            "replayability_contract": {"hash_bundle_complete": True, "exact_replay_ready": True},
            "gate_results": {
                "training_success_confirmed": False,
                "feature_store_manifest_strict_ok": True,
                "new_bot_admission_ok": True,
            },
        }
    )

    assert daily_auto_verify._promotion_packet_builder_ok(2, stdout, "") is True


def test_daily_auto_verify_keeps_bad_promotion_packet_blocking() -> None:
    stdout = json.dumps(
        {
            "committee_packet_seed_ready": True,
            "signing_material_ready": False,
            "signature": {"status": "missing_signing_key"},
            "gate_results": {
                "training_success_confirmed": True,
                "feature_store_manifest_strict_ok": False,
            },
        }
    )

    assert daily_auto_verify._promotion_packet_builder_ok(2, stdout, "") is False
