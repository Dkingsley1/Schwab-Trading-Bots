import time

from scripts.shadow_watchdog import (
    Target,
    _can_restart,
    _evaluate_halt_auto_clear,
    _find_matching_rows,
    _parse_reason_set,
)


def test_watchdog_restart_rate_limit() -> None:
    t = Target(name="x", match="x", start_cmd="echo hi")
    now = time.time()

    assert _can_restart(t, now, max_restarts=2, window_seconds=60)
    t.restart_times.append(now)
    assert _can_restart(t, now, max_restarts=2, window_seconds=60)
    t.restart_times.append(now)
    assert not _can_restart(t, now, max_restarts=2, window_seconds=60)


def test_parse_reason_set_normalizes_and_deduplicates() -> None:
    reasons = _parse_reason_set(" incident_auto_halt , GLOBAL_RISK_KILLSWITCH,incident_auto_halt ,, ")
    assert reasons == {"incident_auto_halt", "global_risk_killswitch"}


def test_halt_auto_clear_requires_paper_only_guard() -> None:
    should_clear, reason = _evaluate_halt_auto_clear(
        halt_active=True,
        halt_reason="incident_auto_halt",
        halt_age_seconds=600.0,
        operator_stop_active=False,
        auto_clear_enabled=True,
        min_age_seconds=300,
        allowed_reasons={"incident_auto_halt"},
        require_paper_only=True,
        market_data_only=False,
        allow_order_execution=True,
    )

    assert not should_clear
    assert reason == "paper_only_guard_failed"


def test_halt_auto_clear_rejects_unapproved_reason() -> None:
    should_clear, reason = _evaluate_halt_auto_clear(
        halt_active=True,
        halt_reason="operator_manual_override",
        halt_age_seconds=900.0,
        operator_stop_active=False,
        auto_clear_enabled=True,
        min_age_seconds=300,
        allowed_reasons={"incident_auto_halt", "global_risk_killswitch"},
        require_paper_only=True,
        market_data_only=True,
        allow_order_execution=False,
    )

    assert not should_clear
    assert reason.startswith("reason_not_allowed")


def test_halt_auto_clear_allows_eligible_case() -> None:
    should_clear, reason = _evaluate_halt_auto_clear(
        halt_active=True,
        halt_reason="incident_auto_halt",
        halt_age_seconds=301.0,
        operator_stop_active=False,
        auto_clear_enabled=True,
        min_age_seconds=300,
        allowed_reasons={"incident_auto_halt"},
        require_paper_only=True,
        market_data_only=True,
        allow_order_execution=False,
    )

    assert should_clear
    assert reason == "eligible"


def test_halt_auto_clear_allows_softguard_api_circuit_in_paper_mode() -> None:
    should_clear, reason = _evaluate_halt_auto_clear(
        halt_active=True,
        halt_reason="softguard_api_circuit_opened",
        halt_age_seconds=901.0,
        operator_stop_active=False,
        auto_clear_enabled=True,
        min_age_seconds=300,
        allowed_reasons={"incident_auto_halt", "softguard_api_circuit_opened"},
        require_paper_only=True,
        market_data_only=True,
        allow_order_execution=False,
    )

    assert should_clear
    assert reason == "eligible"


def test_find_matching_rows_excludes_watchdog_command() -> None:
    rows = [
        (100, "python scripts/shadow_watchdog.py --schwab-start-cmd '/tmp/scripts/run_parallel_shadows.py'"),
        (200, "python scripts/run_parallel_shadows.py --broker schwab"),
    ]

    matches = _find_matching_rows(rows, "scripts/run_parallel_shadows.py")
    assert matches == [(200, "python scripts/run_parallel_shadows.py --broker schwab")]
