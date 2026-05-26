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


def test_promotion_quality_gate_check_ignores_recursive_daily_verify_failures() -> None:
    source = (PROJECT_ROOT / "scripts" / "daily_auto_verify.py").read_text(encoding="utf-8")

    assert '--ignore-daily-verify-check' in source
    assert '"promotion_packet_builder"' in source
    assert '"nightly_resilience_check"' in source
    assert '"promotion_quality_gate"' in source
    assert '"unhandled_exception"' in source
