import time
import json
from datetime import datetime, timezone

from scripts.shadow_watchdog import (
    Target,
    _build_default_schwab_cmd,
    _can_restart,
    _creative_pause_guard_active,
    _decode_start_cmd,
    _evaluate_halt_auto_clear,
    _find_matching_rows,
    _heartbeat_health,
    _heartbeat_startup_grace_active,
    _parse_ps_etime_seconds,
    _parse_reason_set,
    _restart_guard_active_for_target,
    _schwab_live_heartbeat_exclude_matches,
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


def test_build_default_schwab_cmd_uses_all_sleeves_parent() -> None:
    cmd = _build_default_schwab_cmd(simulate=False)

    assert "run_all_sleeves.py" in cmd
    assert "--with-aggressive-modes" in cmd
    assert "run_parallel_shadows.py" not in cmd


def test_parse_ps_etime_seconds_handles_macos_formats() -> None:
    assert _parse_ps_etime_seconds("00:01") == 1.0
    assert _parse_ps_etime_seconds("01:02:03") == 3723.0
    assert _parse_ps_etime_seconds("2-03:04:05") == 183845.0


def test_schwab_parent_heartbeat_startup_grace_prevents_restart_storm() -> None:
    target = Target(
        name="schwab_parallel",
        match="scripts/run_all_sleeves.py",
        start_cmd="echo hi",
        heartbeat_glob="/tmp/missing_*.json",
        heartbeat_stale_seconds=180,
        heartbeat_startup_grace_seconds=420,
    )

    assert _heartbeat_startup_grace_active(
        target,
        proc_live=True,
        hb_required=True,
        hb_ok=False,
        process_age_seconds=120.0,
    ) is True
    assert _heartbeat_startup_grace_active(
        target,
        proc_live=True,
        hb_required=True,
        hb_ok=False,
        process_age_seconds=421.0,
    ) is False


def test_creative_pause_guard_suppresses_shadow_restart_for_music(tmp_path, monkeypatch) -> None:
    from scripts import shadow_watchdog

    pause_path = tmp_path / "creative_heavy_research_pause_latest.json"
    pause_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "active": True,
                "creative_session_level": "hot",
                "creative_session_kind": "music_playback_hot",
                "env_contract": {
                    "TRAINING_RUNTIME_PAUSED_FOR_CREATIVE": "1",
                    "SHADOW_RESEARCH_PAUSED_FOR_CREATIVE": "1",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(shadow_watchdog, "CREATIVE_PAUSE_LATEST", pause_path)

    target = Target(
        name="coinbase_shadow",
        match="scripts/run_shadow_training_loop.py --broker coinbase",
        start_cmd=["./scripts/ops/opsctl.sh", "coinbase-start", "--paper"],
    )

    active, reason = _restart_guard_active_for_target(target)
    assert _creative_pause_guard_active() is True
    assert active is True
    assert reason == "creative_audio_pause_guard_active"


def test_runtime_pressure_guard_suppresses_coinbase_restart(tmp_path, monkeypatch) -> None:
    from scripts import shadow_watchdog

    override_path = tmp_path / ".env.runtime_resource_guard_override"
    override_path.write_text("PAPER_CRYPTO_FEED_RUNTIME_PAUSED_FOR_PRESSURE=1\n", encoding="utf-8")
    monkeypatch.setattr(shadow_watchdog, "RUNTIME_RESOURCE_OVERRIDE", override_path)

    target = Target(
        name="coinbase",
        match="scripts/run_shadow_training_loop.py --broker coinbase",
        start_cmd=["./scripts/ops/opsctl.sh", "coinbase-start", "--paper", "--live-data"],
    )

    active, reason = _restart_guard_active_for_target(target)

    assert active is True
    assert reason == "paper_crypto_feed_pressure_guard_active"


def test_live_schwab_watchdog_excludes_simulated_heartbeat_coverage_by_default() -> None:
    assert _schwab_live_heartbeat_exclude_matches(simulate_schwab=False) == ("--simulate",)
    assert _schwab_live_heartbeat_exclude_matches(simulate_schwab=True) == ()
    assert _schwab_live_heartbeat_exclude_matches(
        simulate_schwab=False,
        allow_simulated_heartbeats=True,
    ) == ()


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


def test_heartbeat_health_can_count_standby_simulated_rows_when_allowed(tmp_path) -> None:
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
        match="scripts/run_all_sleeves.py",
        start_cmd="echo hi",
        heartbeat_glob=str(tmp_path / "shadow_loop_*_equities_schwab_*.json"),
        heartbeat_stale_seconds=600,
        min_healthy_heartbeats=1,
        heartbeat_profiles=("conservative",),
        exclude_matches=("--simulate",),
        heartbeat_exclude_matches=(),
    )

    ok, count, age, live_count = _heartbeat_health(
        target,
        rows_by_pid={111: "python scripts/run_shadow_training_loop.py --broker schwab --simulate"},
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


def test_aggressive_modes_target_is_not_allowed_to_be_processless() -> None:
    target = Target(
        name="aggressive_modes_parallel",
        match="scripts/run_parallel_aggressive_modes.py",
        start_cmd="echo hi",
        heartbeat_glob="/tmp/shadow_loop_*aggressive*_equities_schwab_*.json",
        heartbeat_stale_seconds=180,
        min_healthy_heartbeats=2,
        heartbeat_profiles=("intraday_aggressive", "swing_aggressive"),
        allow_processless_heartbeat_live=False,
    )

    assert target.allow_processless_heartbeat_live is False


def test_tripwire_suppresses_parent_live_heartbeat_loss(tmp_path, monkeypatch) -> None:
    from scripts import shadow_watchdog

    events = tmp_path / "tripwire_events.jsonl"
    latest = tmp_path / "tripwire_latest.json"
    monkeypatch.setattr(shadow_watchdog, "TRIPWIRE_EVENTS", events)
    monkeypatch.setattr(shadow_watchdog, "TRIPWIRE_LATEST", latest)

    target = Target(
        name="schwab_parallel",
        match="scripts/run_all_sleeves.py",
        start_cmd="echo hi",
        heartbeat_glob=str(tmp_path / "missing_*.json"),
        heartbeat_stale_seconds=180,
        suppress_tripwire_when_parent_live=True,
    )

    payload = shadow_watchdog._tripwire_payload(
        [target],
        [
            {
                "name": "schwab_parallel",
                "process_live": True,
                "heartbeat_lost": True,
                "match_count": 1,
                "action": "none",
                "note": "process_live,heartbeat_ok=False",
            }
        ],
        enabled=True,
        streak_threshold=1,
    )

    assert payload["active"] is False
    assert target.tripwire_open is False
