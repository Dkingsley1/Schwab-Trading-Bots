import time
import json
from datetime import datetime, timezone

from scripts.shadow_watchdog import (
    Target,
    _can_restart,
    _decode_start_cmd,
    _evaluate_halt_auto_clear,
    _find_matching_rows,
    _heartbeat_health,
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
        halt_payload_valid=True,
        halt_payload_error="",
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
        halt_payload_valid=True,
        halt_payload_error="",
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
        halt_payload_valid=True,
        halt_payload_error="",
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
        halt_payload_valid=True,
        halt_payload_error="",
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


def test_halt_auto_clear_allows_malformed_payload_in_paper_mode() -> None:
    should_clear, reason = _evaluate_halt_auto_clear(
        halt_active=True,
        halt_reason="",
        halt_payload_valid=False,
        halt_payload_error="empty_payload",
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
    assert reason.startswith("malformed_payload_eligible:")


def test_find_matching_rows_excludes_watchdog_command() -> None:
    rows = [
        (100, "python scripts/shadow_watchdog.py --schwab-start-cmd '/tmp/scripts/run_parallel_shadows.py'"),
        (200, "python scripts/run_parallel_shadows.py --broker schwab"),
    ]

    matches = _find_matching_rows(rows, "scripts/run_parallel_shadows.py")
    assert matches == [(200, "python scripts/run_parallel_shadows.py --broker schwab")]


def test_decode_start_cmd_accepts_json_argv() -> None:
    raw = '["/tmp/New project/.venv312/bin/python","/tmp/New project/scripts/run_parallel_shadows.py"]'
    assert _decode_start_cmd(raw) == [
        "/tmp/New project/.venv312/bin/python",
        "/tmp/New project/scripts/run_parallel_shadows.py",
    ]


def test_decode_start_cmd_recovers_legacy_unquoted_space_path() -> None:
    raw = (
        "/tmp/New project/.venv312/bin/python "
        "/tmp/New project/scripts/run_dividend_shadow.py --interval-seconds 60"
    )
    assert _decode_start_cmd(raw) == [
        "/tmp/New project/.venv312/bin/python",
        "/tmp/New project/scripts/run_dividend_shadow.py",
        "--interval-seconds",
        "60",
    ]


def test_decode_start_cmd_recovers_legacy_opsctl_space_path() -> None:
    raw = "/tmp/New project/scripts/ops/opsctl.sh coinbase-futures-start --paper --live-data"
    assert _decode_start_cmd(raw) == [
        "/tmp/New project/scripts/ops/opsctl.sh",
        "coinbase-futures-start",
        "--paper",
        "--live-data",
    ]


def test_heartbeat_health_ignores_simulated_rows_and_non_target_profiles(tmp_path) -> None:
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    conservative = tmp_path / "shadow_loop_conservative_equities_schwab_111.json"
    conservative.write_text(
        json.dumps(
            {
                "timestamp_utc": now_iso,
                "pid": 111,
                "profile": "conservative",
            }
        ),
        encoding="utf-8",
    )
    dividend = tmp_path / "shadow_loop_dividend_equities_schwab_222.json"
    dividend.write_text(
        json.dumps(
            {
                "timestamp_utc": now_iso,
                "pid": 222,
                "profile": "dividend",
            }
        ),
        encoding="utf-8",
    )
    aggressive = tmp_path / "shadow_loop_aggressive_equities_schwab_333.json"
    aggressive.write_text(
        json.dumps(
            {
                "timestamp_utc": now_iso,
                "pid": 333,
                "profile": "aggressive",
            }
        ),
        encoding="utf-8",
    )

    target = Target(
        name="schwab_parallel",
        match="scripts/run_parallel_shadows.py",
        start_cmd="echo hi",
        heartbeat_glob=str(tmp_path / "shadow_loop_*_equities_schwab_*.json"),
        heartbeat_stale_seconds=600,
        min_healthy_heartbeats=1,
        heartbeat_profiles=("conservative", "aggressive"),
        exclude_matches=("--simulate",),
    )

    ok, count, age, live_count = _heartbeat_health(
        target,
        rows_by_pid={
            111: "python scripts/run_shadow_training_loop.py --broker schwab --simulate",
            222: "python scripts/run_shadow_training_loop.py --broker schwab",
            333: "python scripts/run_shadow_training_loop.py --broker schwab",
        },
    )

    assert ok is True
    assert count == 1
    assert age is not None
    assert live_count == 1


def test_heartbeat_health_requires_live_pid_for_processless_mode(tmp_path) -> None:
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    conservative = tmp_path / "shadow_loop_conservative_equities_schwab_111.json"
    conservative.write_text(
        json.dumps(
            {
                "timestamp_utc": now_iso,
                "pid": 111,
                "profile": "conservative",
            }
        ),
        encoding="utf-8",
    )

    target = Target(
        name="schwab_parallel",
        match="scripts/run_parallel_shadows.py",
        start_cmd="echo hi",
        heartbeat_glob=str(tmp_path / "shadow_loop_*_equities_schwab_*.json"),
        heartbeat_stale_seconds=600,
        min_healthy_heartbeats=1,
        heartbeat_profiles=("conservative",),
    )

    ok, count, age, live_count = _heartbeat_health(target, rows_by_pid={})

    assert ok is True
    assert count == 1
    assert age is not None
    assert live_count == 0
