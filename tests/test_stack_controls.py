import argparse
from pathlib import Path

import scripts.run_all_sleeves as run_all_sleeves


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMANDS_PATH = PROJECT_ROOT / "COMMANDS.md"
OPSCTL_PATH = PROJECT_ROOT / "scripts" / "ops" / "opsctl.sh"


def test_opsctl_top_level_command_aliases_are_unique() -> None:
    text = _read(OPSCTL_PATH)
    main_case = text.rsplit('case "$cmd" in', 1)[1]
    aliases: list[str] = []
    for raw in main_case.splitlines():
        if not raw.startswith("  ") or raw.startswith("    ") or not raw.endswith(")"):
            continue
        pattern = raw.strip()[:-1]
        if not pattern or pattern == "*":
            continue
        aliases.extend(pattern.split("|"))

    duplicates = sorted(alias for alias in set(aliases) if aliases.count(alias) > 1)
    assert duplicates == []


LOAD_RUNTIME_ENV_PATH = PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh"
LIVE_FEED_TAIL_PATH = PROJECT_ROOT / "scripts" / "ops" / "live_feed_tail.sh"
LIVE_FEED_LAUNCHD_PATH = PROJECT_ROOT / "scripts" / "ops" / "run_livefeed_local_launchd.sh"
LIVE_FEED_HEAVY_GUARDED_PATH = PROJECT_ROOT / "scripts" / "ops" / "live_feed_heavy_guarded.sh"
WATCHDOG_INSTALL_PATH = PROJECT_ROOT / "scripts" / "install_shadow_watchdog_launchd.sh"
INFRA_INSTALL_PATH = PROJECT_ROOT / "scripts" / "install_infra_stack_launchd.sh"
OPS_AUTOMATION_INSTALL_PATH = PROJECT_ROOT / "scripts" / "ops" / "install_ops_automation_launchd.sh"
STARTUP_PROMPT_INSTALL_PATH = PROJECT_ROOT / "scripts" / "install_startup_start_prompt_launchd.sh"
STARTUP_PROMPT_RUN_PATH = PROJECT_ROOT / "scripts" / "ops" / "run_startup_start_prompt_launchd.sh"
STARTUP_PROMPT_NOTIFIER_PATH = PROJECT_ROOT / "scripts" / "ops" / "startup_prompt_notifier.swift"
PRODUCTION_HARDENING_WATCH_INSTALL_PATH = PROJECT_ROOT / "scripts" / "install_production_hardening_watch_launchd.sh"
SOAK_SELF_HEAL_INSTALL_PATH = PROJECT_ROOT / "scripts" / "install_soak_self_healing_launchd.sh"
STORAGE_BACKPRESSURE_AUTOPILOT_RUN_PATH = PROJECT_ROOT / "scripts" / "ops" / "run_storage_backpressure_autopilot_launchd.sh"
RUNTIME_SMOOTH_MODE_RUN_PATH = PROJECT_ROOT / "scripts" / "ops" / "run_runtime_smooth_mode_launchd.sh"
PRODUCTION_HARDENING_WATCH_RUN_PATH = PROJECT_ROOT / "scripts" / "ops" / "run_production_hardening_watch_launchd.sh"
RETRAIN_DAILY_PATH = PROJECT_ROOT / "scripts" / "retrain_daily_small_batch.sh"
RETRAIN_WEEKLY_PATH = PROJECT_ROOT / "scripts" / "retrain_weekly_full_sweep.sh"
DAILY_AUTO_VERIFY_RUN_PATH = PROJECT_ROOT / "scripts" / "ops" / "run_daily_auto_verify_launchd.sh"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_storage_resilience_opsctl_uses_bounded_fast_integrity_checks() -> None:
    text = _read(OPSCTL_PATH)

    assert 'storage_resilience_control.py" --fast "$@"' in text


def test_infra_stack_installer_includes_ops_and_daily_verify() -> None:
    text = _read(INFRA_INSTALL_PATH)

    assert "install_daily_auto_verify_launchd.sh" in text
    assert "scripts/ops/install_ops_automation_launchd.sh" in text
    assert "install_observability_exporter_launchd.sh" in text
    assert "install_production_resilience_control_launchd.sh" in text


def test_daily_auto_verify_runs_annual_tax_policy_rollover_check() -> None:
    text = _read(DAILY_AUTO_VERIFY_RUN_PATH)

    assert "tax_regulation_update.py" in text
    assert "--auto" in text
    assert "daily_tax_regulation_update_end" in text


def test_ops_automation_installer_includes_context_jobs() -> None:
    text = _read(OPS_AUTOMATION_INSTALL_PATH)

    assert "run_market_crypto_correlation_launchd.sh" in text
    assert "run_fx_market_context_launchd.sh" in text
    assert "run_options_flow_context_launchd.sh" in text
    assert "run_official_macro_context_launchd.sh" in text
    assert "run_schwab_education_context_launchd.sh" in text
    assert "run_storage_backpressure_autopilot_launchd.sh" in text
    assert "run_infrastructure_autofix_launchd.sh" in text
    assert "run_master_infrastructure_supervisor_launchd.sh" in text
    assert "run_command_validity_launchd.sh" in text
    assert "run_system_drift_guard_launchd.sh" in text
    assert "run_system_drift_autopilot_launchd.sh" in text
    assert "run_storage_standby_prune_launchd.sh" in text
    assert "run_grade_regression_autopilot_launchd.sh" in text
    assert "run_section_grade_autopilot_launchd.sh" in text
    assert "run_creative_cotenant_guard_launchd.sh" in text
    assert "run_runtime_smooth_mode_launchd.sh" in text
    assert "run_production_hardening_watch_launchd.sh" in text
    assert "production_resilience_control.py" in text
    assert "com.dankingsley.ops.runtime_smooth_mode" in text
    assert "com.dankingsley.ops.production_resilience_control" in text
    assert "com.dankingsley.ops.production_hardening_watch" in text
    assert "RUNTIME_SMOOTH_MODE_INTERVAL_SECONDS" in text
    assert "PRODUCTION_RESILIENCE_CONTROL_INTERVAL_SECONDS" in text
    assert "PRODUCTION_HARDENING_WATCH_INTERVAL_SECONDS" in text
    assert "ops_runtime_smooth_mode.out.log" in text
    assert "ops_production_hardening_watch.out.log" in text
    assert "com.dankingsley.ops.master_infrastructure_supervisor" in text


def test_production_resilience_control_launchd_is_periodic_and_live_locked() -> None:
    text = _read(PROJECT_ROOT / "scripts" / "install_production_resilience_control_launchd.sh")

    assert "resolve_runtime_python" in text
    assert "production_resilience_control.py" in text
    assert "PRODUCTION_RESILIENCE_CONTROL_INTERVAL_SECONDS:-300" in text
    assert "MARKET_DATA_ONLY</key><string>1" in text
    assert "ALLOW_ORDER_EXECUTION</key><string>0" in text
    assert "BOT_LIVE_MONEY_LOCKED_DURING_SOAK</key><string>1" in text
    assert "ProcessType</key><string>Background" in text


def test_production_hardening_watch_launchd_keeps_live_locked_and_bounded() -> None:
    text = _read(PRODUCTION_HARDENING_WATCH_RUN_PATH)

    assert "production_hardening_watch_launchd.lock" in text
    assert "MARKET_DATA_ONLY=1" in text
    assert "ALLOW_ORDER_EXECUTION=0" in text
    assert "TOP_BOT_ENABLE_LIVE_EXECUTION=0" in text
    assert "PRODUCTION_HARDENING_WATCH_EXECUTE_SAFE_REPAIRS" in text
    assert "PRODUCTION_HARDENING_WATCH_EXECUTE_ON_WATCH" in text
    assert "production-hardening-watch" in text
    assert "--max-execute-actions" in text
    assert "PRODUCTION_PILLAR_REFRESH_ENABLED" in text
    assert "--profile production" in text
    assert "PRODUCTION_PILLAR_REFRESH_COOLDOWN_MINUTES:-45" in text
    assert "/usr/bin/nice" in text


def test_production_hardening_watch_installer_is_standalone() -> None:
    text = _read(PRODUCTION_HARDENING_WATCH_INSTALL_PATH)

    assert "com.dankingsley.ops.production_hardening_watch" in text
    assert "run_production_hardening_watch_launchd.sh" in text
    assert "PRODUCTION_HARDENING_WATCH_INTERVAL_SECONDS" in text
    assert "PRODUCTION_HARDENING_WATCH_EXECUTE_SAFE_REPAIRS" in text
    assert "MARKET_DATA_ONLY" in text
    assert "ALLOW_ORDER_EXECUTION" in text
    assert "ops_production_hardening_watch.out.log" in text


def test_runtime_smooth_mode_launchd_applies_memory_and_runtime_controls() -> None:
    text = _read(RUNTIME_SMOOTH_MODE_RUN_PATH)

    assert "runtime_smooth_mode_launchd.lock" in text
    assert "RUNTIME_SMOOTH_MODE_LOCK_ROOT" in text
    assert "RUNTIME_SMOOTH_MODE_LOCK_STALE_SECONDS" in text
    assert "RUNTIME_SMOOTH_MODE_AUTOMATIC" in text
    assert "memory-pressure-intelligence --apply --json" in text
    assert "runtime-throttle --apply --json" in text
    assert "exec /usr/bin/nice" not in text


def test_storage_backpressure_autopilot_launchd_runs_multi_cycle_clearance() -> None:
    text = _read(STORAGE_BACKPRESSURE_AUTOPILOT_RUN_PATH)

    assert "backlog_drain_uniform_process.py" in text
    assert "load_runtime_env.sh" in text
    assert "--max-cycles" in text
    assert "STORAGE_BACKPRESSURE_AUTOPILOT_MAX_CYCLES" in text
    assert "--target-pending-lines" in text
    assert "STORAGE_BACKPRESSURE_AUTOPILOT_TARGET_PENDING_LINES" in text
    assert "--target-retention-debt-gb" in text
    assert "STORAGE_BACKPRESSURE_AUTOPILOT_TARGET_RETENTION_DEBT_GB" in text


def test_uniform_backlog_drain_process_is_late_loaded_and_exposed() -> None:
    opsctl = _read(OPSCTL_PATH)
    loader = _read(LOAD_RUNTIME_ENV_PATH)

    assert "backlog-drain-uniform-process|uniform-drain-process" in opsctl
    assert "backlog_drain_uniform_process.py" in opsctl
    assert ".env.backlog_drain_uniform_override" in loader
    assert loader.index(".env.backlog_pcore_accelerator_override") < loader.index(".env.backlog_drain_uniform_override")
    assert loader.index(".env.load_shape_smooth_override") < loader.index(".env.backlog_drain_uniform_override")


def test_run_all_sleeves_uses_signal_handlers() -> None:
    handlers = []

    def fake_signal(sig, handler):
        handlers.append((sig, handler))
        return None

    original_signal = run_all_sleeves.signal.signal
    try:
        run_all_sleeves.signal.signal = fake_signal
        run_all_sleeves._install_signal_handlers()
    finally:
        run_all_sleeves.signal.signal = original_signal

    assert handlers == [
        (run_all_sleeves.signal.SIGINT, run_all_sleeves._handle_shutdown_signal),
        (run_all_sleeves.signal.SIGTERM, run_all_sleeves._handle_shutdown_signal),
    ]


def test_run_all_sleeves_child_nice_targets_are_parent_relative() -> None:
    assert run_all_sleeves._relative_nice_increment_for_target(8, parent_nice=5) == 3
    assert run_all_sleeves._nice_prefix_for_target(8, parent_nice=5) == ["nice", "-n", "3"]


def test_run_all_sleeves_child_nice_does_not_try_to_lift_above_parent() -> None:
    assert run_all_sleeves._relative_nice_increment_for_target(4, parent_nice=5) == 0
    assert run_all_sleeves._nice_prefix_for_target(4, parent_nice=5) == ["nice", "-n", "0"]


def test_run_all_sleeves_paper_executor_uses_runtime_nice(monkeypatch) -> None:
    monkeypatch.setenv("PAPER_EXECUTION_RUNTIME_NICE", "18")

    assert run_all_sleeves._paper_executor_target_nice(6) == 18
    assert run_all_sleeves._nice_prefix_for_target(
        run_all_sleeves._paper_executor_target_nice(6),
        parent_nice=5,
    ) == ["nice", "-n", "13"]


def test_run_all_sleeves_paper_executor_nice_falls_back_to_baseline(monkeypatch) -> None:
    monkeypatch.setenv("PAPER_EXECUTION_RUNTIME_NICE", "not-an-int")
    monkeypatch.delenv("PAPER_SHADOW_RUNTIME_NICE", raising=False)

    assert run_all_sleeves._paper_executor_target_nice(6) == 6


def test_paper_trade_lock_disables_live_executor(monkeypatch) -> None:
    monkeypatch.setenv("PAPER_TRADE_LOCK", "1")
    monkeypatch.setattr(run_all_sleeves, "_emit_incident_snapshot", lambda *_args, **_kwargs: None)
    args = argparse.Namespace(with_live_executor=True)

    assert run_all_sleeves._apply_paper_trade_lock(args) is True
    assert args.with_live_executor is False


def test_run_all_sleeves_breaker_respects_data_quality_warmup() -> None:
    args = argparse.Namespace(
        broker="schwab",
        breaker_min_data_quality=75.0,
        breaker_max_blocked_rate=0.35,
        breaker_min_pnl_proxy=-0.02,
        breaker_data_quality_grace_seconds=900,
    )
    metrics = {
        "data_quality_score": "25.00",
        "combined_blocked_rate": "0.000000",
        "stocks_pnl_proxy": "0.000000",
    }

    reasons, domain = run_all_sleeves._breaker_reasons(metrics, args, runtime_seconds=120.0)

    assert domain == "stocks"
    assert reasons == []


def test_run_all_sleeves_breaker_enforces_data_quality_after_warmup() -> None:
    args = argparse.Namespace(
        broker="schwab",
        breaker_min_data_quality=75.0,
        breaker_max_blocked_rate=0.35,
        breaker_min_pnl_proxy=-0.02,
        breaker_data_quality_grace_seconds=900,
    )
    metrics = {
        "data_quality_score": "25.00",
        "combined_blocked_rate": "0.000000",
        "stocks_pnl_proxy": "0.000000",
    }

    reasons, domain = run_all_sleeves._breaker_reasons(metrics, args, runtime_seconds=901.0)

    assert domain == "stocks"
    assert reasons == ["data_quality_low:25.00"]


def test_run_all_sleeves_breaker_ignores_stale_metrics() -> None:
    args = argparse.Namespace(
        broker="schwab",
        breaker_min_data_quality=75.0,
        breaker_max_blocked_rate=0.35,
        breaker_min_pnl_proxy=-0.02,
        breaker_data_quality_grace_seconds=900,
        breaker_max_metric_age_seconds=300,
    )
    metrics = {
        "_breaker_source_present": True,
        "_breaker_source_age_seconds": 301.0,
        "data_quality_score": "25.00",
        "combined_blocked_rate": "0.000000",
        "stocks_pnl_proxy": "0.000000",
    }

    actionable, reason = run_all_sleeves._breaker_metrics_actionable(metrics, args)
    reasons, domain = run_all_sleeves._breaker_reasons(metrics, args, runtime_seconds=901.0)

    assert actionable is False
    assert reason == "source_stale"
    assert domain == "stocks"
    assert reasons == []


def test_run_all_sleeves_breaker_ignores_closed_schwab_session() -> None:
    args = argparse.Namespace(
        broker="schwab",
        breaker_min_data_quality=75.0,
        breaker_max_blocked_rate=0.35,
        breaker_min_pnl_proxy=-0.02,
        breaker_data_quality_grace_seconds=900,
        breaker_max_metric_age_seconds=300,
    )
    metrics = {
        "data_quality_session_aware": "true",
        "data_quality_session_open": "false",
        "data_quality_score": "25.00",
        "combined_blocked_rate": "0.000000",
        "stocks_pnl_proxy": "0.000000",
    }

    actionable, reason = run_all_sleeves._breaker_metrics_actionable(metrics, args)
    reasons, _domain = run_all_sleeves._breaker_reasons(metrics, args, runtime_seconds=901.0)

    assert actionable is False
    assert reason == "market_session_closed"
    assert reasons == []


def test_run_all_sleeves_breaker_parks_execution_without_parking_collectors() -> None:
    specs = {
        "baseline_parallel": run_all_sleeves.JobSpec(
            "baseline_parallel", [], {}, breaker_group=run_all_sleeves.COLLECTION_BREAKER_GROUP
        ),
        "bond": run_all_sleeves.JobSpec(
            "bond", [], {}, breaker_group=run_all_sleeves.COLLECTION_BREAKER_GROUP
        ),
        "paper_executor": run_all_sleeves.JobSpec(
            "paper_executor", [], {}, breaker_group=run_all_sleeves.EXECUTION_BREAKER_GROUP
        ),
    }

    parked = run_all_sleeves._breaker_policy_parked_jobs(
        specs,
        {
            run_all_sleeves.COLLECTION_BREAKER_GROUP: 0.0,
            run_all_sleeves.EXECUTION_BREAKER_GROUP: 200.0,
        },
        now=100.0,
    )

    assert parked == {"paper_executor"}


def test_run_all_sleeves_heartbeat_watch_respects_startup_grace(tmp_path) -> None:
    heartbeat = tmp_path / "execution_lane_paper_latest.json"
    spec = run_all_sleeves.JobSpec(
        "paper_executor",
        [],
        {},
        breaker_group="core",
        heartbeat_path=heartbeat,
        heartbeat_stale_seconds=240,
        heartbeat_startup_grace_seconds=240,
    )

    stale, reason = run_all_sleeves._job_heartbeat_stale(
        spec,
        started_at=100.0,
        now_ts=200.0,
    )

    assert stale is False
    assert reason == "startup_grace"


def test_run_all_sleeves_heartbeat_watch_detects_stale_payload(tmp_path) -> None:
    heartbeat = tmp_path / "execution_lane_paper_latest.json"
    heartbeat.write_text('{"stale": true}', encoding="utf-8")
    spec = run_all_sleeves.JobSpec(
        "paper_executor",
        [],
        {},
        breaker_group="core",
        heartbeat_path=heartbeat,
        heartbeat_stale_seconds=240,
        heartbeat_startup_grace_seconds=30,
    )

    stale, reason = run_all_sleeves._job_heartbeat_stale(
        spec,
        started_at=0.0,
        now_ts=120.0,
    )

    assert stale is True
    assert reason == "payload_stale"


def test_run_all_sleeves_keeps_paper_lane_alive_for_paused_heartbeat(monkeypatch) -> None:
    monkeypatch.setenv("PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED", "0")
    monkeypatch.setenv("PAPER_EXECUTION_RUNTIME_PAUSED_FOR_PRESSURE", "1")
    monkeypatch.delenv("PAPER_RECONCILIATION_HEARTBEAT_WHEN_PAUSED", raising=False)

    assert run_all_sleeves._paper_execution_consumer_enabled() is True

    monkeypatch.setenv("PAPER_RECONCILIATION_HEARTBEAT_WHEN_PAUSED", "0")

    assert run_all_sleeves._paper_execution_consumer_enabled() is False


def test_run_all_sleeves_recycles_execution_lane_on_code_change(tmp_path) -> None:
    watched = tmp_path / "base_trader.py"
    watched.write_text("# v1\n", encoding="utf-8")
    spec = run_all_sleeves.JobSpec(
        "paper_executor",
        [],
        {},
        breaker_group="core",
        max_runtime_seconds=0,
        code_watch_paths=(watched,),
    )

    recycle, reason = run_all_sleeves._job_recycle_due(spec, started_at=100.0, now_ts=120.0)

    assert recycle is True
    assert reason == "code_changed:base_trader.py"


def test_run_all_sleeves_recycles_execution_lane_on_max_runtime() -> None:
    spec = run_all_sleeves.JobSpec(
        "paper_executor",
        [],
        {},
        breaker_group="core",
        max_runtime_seconds=60,
    )

    recycle, reason = run_all_sleeves._job_recycle_due(spec, started_at=100.0, now_ts=161.0)

    assert recycle is True
    assert reason == "max_runtime_seconds=60"


def test_run_all_sleeves_launcher_health_marks_degraded_children() -> None:
    class FakeProc:
        def __init__(self, pid: int, exit_code):
            self.pid = pid
            self._exit_code = exit_code

        def poll(self):
            return self._exit_code

    specs = {
        "baseline_parallel": run_all_sleeves.JobSpec("baseline_parallel", [], {}, breaker_group="core"),
        "bond": run_all_sleeves.JobSpec("bond", [], {}, breaker_group="core"),
        "fx": run_all_sleeves.JobSpec("fx", [], {}, breaker_group="core"),
    }
    procs = {
        "baseline_parallel": FakeProc(101, None),
        "bond": FakeProc(102, 1),
    }

    payload = run_all_sleeves._launcher_health_payload(
        specs=specs,
        procs=procs,
        proc_started_at={"baseline_parallel": 50.0, "bond": 60.0},
        restart_history={"bond": [80.0]},
        quarantined_jobs={},
        launcher_started_at=40.0,
        phase="running",
        note="test",
    )

    assert payload["overall_status"] == "degraded"
    assert payload["expected_job_count"] == 3
    assert payload["running_job_count"] == 1
    assert payload["exited_job_count"] == 1
    assert payload["missing_job_count"] == 1
    assert payload["repair_packet"]["status"] == "needs_repair"
    assert payload["repair_packet"]["problem_job_count"] == 2
    assert "sleeve_launcher_parent_watchdog" in [row["name"] for row in payload["repair_infrabots"]]
    states = {row["name"]: row["state"] for row in payload["jobs"]}
    assert states == {"baseline_parallel": "running", "bond": "exited", "fx": "missing"}


def test_run_all_sleeves_launcher_health_does_not_repair_not_yet_spawned_starting_jobs() -> None:
    class FakeProc:
        def __init__(self, pid: int, exit_code):
            self.pid = pid
            self._exit_code = exit_code

        def poll(self):
            return self._exit_code

    specs = {
        "baseline_parallel": run_all_sleeves.JobSpec("baseline_parallel", [], {}, breaker_group="core"),
        "dividend": run_all_sleeves.JobSpec("dividend", [], {}, breaker_group="core"),
        "bond": run_all_sleeves.JobSpec("bond", [], {}, breaker_group="core"),
    }
    procs = {
        "baseline_parallel": FakeProc(101, None),
        "dividend": FakeProc(102, 0),
    }

    payload = run_all_sleeves._launcher_health_payload(
        specs=specs,
        procs=procs,
        proc_started_at={"baseline_parallel": 50.0, "dividend": 60.0},
        restart_history={},
        quarantined_jobs={},
        launcher_started_at=40.0,
        phase="starting",
        note="test",
    )

    assert payload["repair_packet"]["status"] == "clear"
    assert payload["repair_packet"]["problem_job_count"] == 0


def test_run_all_sleeves_launcher_health_treats_fanout_parked_jobs_as_policy() -> None:
    class FakeProc:
        def __init__(self, pid: int, exit_code):
            self.pid = pid
            self._exit_code = exit_code

        def poll(self):
            return self._exit_code

    specs = {
        "baseline_parallel": run_all_sleeves.JobSpec("baseline_parallel", [], {}, breaker_group="core"),
        "volatility": run_all_sleeves.JobSpec("volatility", [], {}, breaker_group="core"),
        "aggressive_modes": run_all_sleeves.JobSpec("aggressive_modes", [], {}, breaker_group="core"),
    }
    procs = {
        "baseline_parallel": FakeProc(101, None),
        "volatility": FakeProc(102, -15),
        "aggressive_modes": FakeProc(103, -15),
    }

    payload = run_all_sleeves._launcher_health_payload(
        specs=specs,
        procs=procs,
        proc_started_at={"baseline_parallel": 50.0, "volatility": 60.0, "aggressive_modes": 70.0},
        restart_history={},
        quarantined_jobs={},
        launcher_started_at=40.0,
        phase="running",
        note="test",
        policy_parked_jobs={"volatility", "aggressive_modes"},
    )

    assert payload["overall_status"] == "ready"
    assert payload["policy_parked_job_count"] == 2
    assert payload["repair_packet"]["status"] == "clear"
    assert payload["repair_packet"]["problem_job_count"] == 0
    assert payload["launcher_readiness_contract"]["readiness_status"] == "stable_with_parked_lanes"
    assert payload["launcher_readiness_contract"]["max_new_collect_only_sleeves"] == 3
    parked = {row["name"]: row["policy_parked"] for row in payload["jobs"]}
    assert parked["volatility"] is True
    assert parked["aggressive_modes"] is True


def test_run_all_sleeves_launcher_health_treats_clean_exits_as_session_parked() -> None:
    class FakeProc:
        def __init__(self, pid: int, exit_code):
            self.pid = pid
            self._exit_code = exit_code

        def poll(self):
            return self._exit_code

    specs = {
        "baseline_parallel": run_all_sleeves.JobSpec("baseline_parallel", [], {}, breaker_group="core"),
        "volatility": run_all_sleeves.JobSpec("volatility", [], {}, breaker_group="core"),
    }
    procs = {
        "baseline_parallel": FakeProc(101, None),
        "volatility": FakeProc(102, 0),
    }

    payload = run_all_sleeves._launcher_health_payload(
        specs=specs,
        procs=procs,
        proc_started_at={"baseline_parallel": 50.0, "volatility": 60.0},
        restart_history={},
        quarantined_jobs={},
        launcher_started_at=40.0,
        phase="running",
        note="test",
        clean_exited_jobs={"volatility"},
    )

    assert payload["overall_status"] == "ready"
    assert payload["clean_exited_job_count"] == 1
    assert payload["repair_packet"]["status"] == "clear"
    assert payload["repair_packet"]["problem_job_count"] == 0
    assert payload["launcher_readiness_contract"]["readiness_status"] == "stable_with_parked_lanes"
    clean = {row["name"]: row["clean_exited"] for row in payload["jobs"]}
    assert clean["volatility"] is True


def test_run_all_sleeves_process_fanout_policy_parks_optional_sleeves(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RUN_ALL_SLEEVES_SPECIALIZED_PROFILE_ALLOWLIST", raising=False)
    monkeypatch.delenv("PAPER_SOAK_SPECIALIZED_ALLOWLIST_BYPASS_FANOUT", raising=False)
    override = tmp_path / "override.env"
    override.write_text(
        "\n".join(
            [
                "PROCESS_FANOUT_GUARD_ACTIVE=1",
                "RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES=0",
                "OPS_WATCHDOG_ALL_SLEEVES_WITH_AGGRESSIVE=0",
                "RUN_ALL_SLEEVES_WITH_DIVIDEND_CAPTURE=0",
            ]
        ),
        encoding="utf-8",
    )

    policy = run_all_sleeves._process_fanout_policy(override)

    assert run_all_sleeves._job_parked_by_fanout_policy("volatility", policy) is True
    assert run_all_sleeves._job_parked_by_fanout_policy("aggressive_modes", policy) is True
    assert run_all_sleeves._job_parked_by_fanout_policy("dividend_capture", policy) is True
    assert run_all_sleeves._job_parked_by_fanout_policy("baseline_parallel", policy) is False


def test_run_all_sleeves_cpu_pressure_guard_narrows_specialized_to_paper_allowlist(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUN_ALL_SLEEVES_SPECIALIZED_PROFILE_ALLOWLIST", "volatility,pairs_correlation,unknown_profile")
    monkeypatch.setenv("PAPER_SOAK_SPECIALIZED_ALLOWLIST_BYPASS_FANOUT", "1")
    override = tmp_path / "override.env"
    override.write_text(
        "\n".join(
            [
                "PROCESS_FANOUT_GUARD_ACTIVE=1",
                "PROCESS_FANOUT_GUARD_REASON=runtime_cpu_pressure",
                "RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES=0",
                "OPS_WATCHDOG_ALL_SLEEVES_WITH_AGGRESSIVE=0",
                "RUN_ALL_SLEEVES_WITH_DIVIDEND_CAPTURE=0",
            ]
        ),
        encoding="utf-8",
    )

    class Args:
        with_specialized_sleeves = True
        with_aggressive_modes = True
        with_dividend_capture = True

    args = Args()
    policy = run_all_sleeves._process_fanout_policy(override)
    changes = run_all_sleeves._apply_process_fanout_policy_to_args(args, policy)

    assert policy["specialized_allowlist_profiles"] == ["volatility", "pairs_correlation"]
    assert run_all_sleeves._specialized_profiles_for_launch() == ("volatility", "pairs_correlation")
    assert run_all_sleeves._job_parked_by_fanout_policy("volatility", policy) is False
    assert run_all_sleeves._job_parked_by_fanout_policy("earnings_event", policy) is True
    assert changes == ["specialized_sleeves_allowlist_only", "aggressive_modes", "dividend_capture"]
    assert args.with_specialized_sleeves is True
    assert args.with_aggressive_modes is False
    assert args.with_dividend_capture is False


def test_run_all_sleeves_memory_pressure_guard_overrides_paper_allowlist(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUN_ALL_SLEEVES_SPECIALIZED_PROFILE_ALLOWLIST", "volatility,pairs_correlation")
    monkeypatch.setenv("PAPER_SOAK_SPECIALIZED_ALLOWLIST_BYPASS_FANOUT", "1")
    override = tmp_path / "override.env"
    override.write_text(
        "\n".join(
            [
                "PROCESS_FANOUT_GUARD_ACTIVE=1",
                "PROCESS_FANOUT_GUARD_REASON=memory_pressure",
                "RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES=0",
            ]
        ),
        encoding="utf-8",
    )

    policy = run_all_sleeves._process_fanout_policy(override)

    assert policy["specialized_allowlist_bypass_enabled"] is False
    assert run_all_sleeves._job_parked_by_fanout_policy("volatility", policy) is True


def test_run_all_sleeves_applies_process_fanout_policy_before_building_specs(monkeypatch) -> None:
    monkeypatch.delenv("RUN_ALL_SLEEVES_SPECIALIZED_PROFILE_ALLOWLIST", raising=False)
    monkeypatch.delenv("PAPER_SOAK_SPECIALIZED_ALLOWLIST_BYPASS_FANOUT", raising=False)
    class Args:
        with_specialized_sleeves = True
        with_aggressive_modes = True
        with_dividend_capture = True

    args = Args()

    changes = run_all_sleeves._apply_process_fanout_policy_to_args(
        args,
        {
            "active": True,
            "specialized_enabled": False,
            "aggressive_enabled": False,
            "dividend_capture_enabled": False,
        },
    )

    assert changes == ["specialized_sleeves", "aggressive_modes", "dividend_capture"]
    assert args.with_specialized_sleeves is False
    assert args.with_aggressive_modes is False
    assert args.with_dividend_capture is False


def test_run_all_sleeves_disk_gate_uses_external_runtime_storage(monkeypatch, tmp_path) -> None:
    external_root = tmp_path / "external"
    external_root.mkdir()
    free_by_path = {
        str(run_all_sleeves.PROJECT_ROOT): 4.5,
        str(external_root): 705.0,
    }

    def fake_disk_free_gb(path: Path) -> float:
        return free_by_path[str(path)]

    monkeypatch.setattr(run_all_sleeves, "_disk_free_gb", fake_disk_free_gb)

    status = run_all_sleeves._disk_gate_status(
        {"mode": "external", "active_root": str(external_root)},
        local_min_free_gb=2.0,
        storage_min_free_gb=15.0,
    )

    assert status["ok"] is True
    assert status["storage_probe"] == str(external_root)


def test_run_all_sleeves_disk_gate_keeps_local_headroom(monkeypatch, tmp_path) -> None:
    external_root = tmp_path / "external"
    external_root.mkdir()
    free_by_path = {
        str(run_all_sleeves.PROJECT_ROOT): 1.0,
        str(external_root): 705.0,
    }

    def fake_disk_free_gb(path: Path) -> float:
        return free_by_path[str(path)]

    monkeypatch.setattr(run_all_sleeves, "_disk_free_gb", fake_disk_free_gb)

    status = run_all_sleeves._disk_gate_status(
        {"mode": "external", "active_root": str(external_root)},
        local_min_free_gb=2.0,
        storage_min_free_gb=15.0,
    )

    assert status["ok"] is False
    assert status["blocked_reasons"] == ["local_project_disk"]


def test_commands_start_stop_section_uses_stack_entrypoint() -> None:
    text = _read(COMMANDS_PATH)

    assert "### Start the full live stack" in text
    assert "./scripts/ops/opsctl.sh start" in text
    assert "### Refresh the livefeed mirror without restarting sleeves" in text
    assert "./scripts/ops/opsctl.sh livefeed-refresh" in text
    assert "### Stop the stack" in text
    assert "### Validate documented commands" in text
    assert "./scripts/ops/opsctl.sh command-validity --json" in text
    assert "### Review the cross-system drift mesh" in text
    assert "./scripts/ops/opsctl.sh system-drift-guard --json" in text
    assert "### Repair safe cross-system drift surfaces" in text
    assert "./scripts/ops/opsctl.sh system-drift-autopilot --apply --json" in text
    assert "### Heavy operator livefeed view" in text
    assert "./scripts/ops/opsctl.sh feed --source main --heavy --no-heavy-ttl --color --red-actions" in text
    assert "### Heavy live feed with file diagnostics" in text
    assert "./scripts/ops/opsctl.sh feed --source main --heavy --show-files --no-heavy-ttl --color --red-actions" in text
    assert "### Active bot stack PDF" in text
    assert "./scripts/ops/open_report_artifact.sh botstack" in text


def test_startup_start_prompt_launchd_is_guarded_and_discoverable() -> None:
    runner = _read(STARTUP_PROMPT_RUN_PATH)
    installer = _read(STARTUP_PROMPT_INSTALL_PATH)
    notifier = _read(STARTUP_PROMPT_NOTIFIER_PATH)
    opsctl = _read(OPSCTL_PATH)

    assert "display notification" in runner
    assert 'buttons {"No", "Yes"}' in runner
    assert "giving up after $TIMEOUT_SECONDS" in runner
    assert "startup_start_prompt_latest.json" in runner
    assert "paper-lock --apply --json" in runner
    assert "start_args=(start)" in runner
    assert "STARTUP_START_PROMPT_FORCE_RESTART" in runner
    assert "STARTUP_START_PROMPT_NO_BROWSER" in runner
    assert "SCHWAB_AUTH_BROWSER_DISABLED=1" in runner
    assert "SCHWAB_AUTH_ALLOW_BROWSER_OPEN=0" in runner
    assert "PREMARKET_TOKEN_BROWSER_AUTH_DISABLED=1" in runner
    assert "CHROME_HEADLESS_QUIET_MODE=1" in runner
    assert "REPORT_HEADLESS_BROWSER_RENDER_ENABLED=0" in runner
    assert "PROJECT_TIMELINE_AUTO_RENDER_PDF=0" in runner
    assert "BROWSER=/usr/bin/false" in runner
    assert "--dry-run" in runner
    assert "without showing UI or starting the stack" in runner
    assert 'print -r -- "actionable_notification|$result"' in runner
    assert "/usr/bin/open -W -n -a" in runner
    assert 'prompt_transport' in runner
    assert 'fail_closed_no_response' in runner

    assert 'UNNotificationAction' in notifier
    assert 'title: "Start"' in notifier
    assert 'title: "Not Now"' in notifier
    assert 'UNNotificationDismissActionIdentifier' in notifier
    assert 'finish("timeout")' in notifier
    assert 'CommandLine.arguments.contains("--self-test")' in notifier
    assert 'self_test_ready' in notifier
    assert 'validate_actionable_notifier' in runner
    assert 'actionable_notification_ready' in runner

    assert "com.dankingsley.startup_start_prompt" in installer
    assert "run_startup_start_prompt_launchd.sh" in installer
    assert "<key>RunAtLoad</key>" in installer
    assert "<key>KeepAlive</key>" in installer
    assert "<false/>" in installer
    assert "STARTUP_START_PROMPT_TIMEOUT_SECONDS" in installer
    assert "STARTUP_START_PROMPT_NO_BROWSER" in installer
    assert "SCHWAB_AUTH_BROWSER_DISABLED" in installer
    assert "SCHWAB_AUTH_ALLOW_BROWSER_OPEN" in installer
    assert "PREMARKET_TOKEN_BROWSER_AUTH_DISABLED" in installer
    assert "CHROME_HEADLESS_QUIET_MODE" in installer
    assert "REPORT_HEADLESS_BROWSER_RENDER_ENABLED" in installer
    assert "PROJECT_TIMELINE_AUTO_RENDER_PDF" in installer
    assert "--no-kickstart|--next-login-only" in installer
    assert '/usr/bin/swiftc -O "$NOTIFIER_SOURCE"' in installer
    assert "com.dankingsley.SchwabStartupPrompt" in installer
    assert "STARTUP_START_PROMPT_APP" in installer

    assert "startup-start-prompt|startup-prompt|login-start-prompt" in opsctl
    assert "startup-start-prompt-test|startup-prompt-test|login-start-prompt-test" in opsctl
    assert "install_startup_start_prompt_launchd.sh" in opsctl
    assert "run_startup_start_prompt_launchd.sh" in opsctl
    assert "startup-start-prompt [--install|--uninstall]" in opsctl
    assert "[--no-browser|--allow-browser]" in opsctl


def test_browser_quiet_override_keeps_soak_reports_from_spawning_headless_chrome() -> None:
    override = _read(PROJECT_ROOT / "config" / ".env.browser_quiet_override")

    assert "CHROME_HEADLESS_QUIET_MODE=1" in override
    assert "REPORT_HEADLESS_BROWSER_RENDER_ENABLED=0" in override
    assert "PROJECT_TIMELINE_AUTO_RENDER_PDF=0" in override


def test_start_stack_blocks_cleanly_on_operator_stop_or_global_halt() -> None:
    text = _read(PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh")

    assert "OPERATOR_STOP.flag" in text
    assert "GLOBAL_TRADING_HALT.flag" in text
    assert "stack_start_status=blocked_by_safety_flags" in text


def test_start_stack_waits_for_launchd_watchdog_readiness() -> None:
    text = _read(PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh")

    assert "wait_for_process_match" in text
    assert "SHADOW_WATCHDOG_START_TIMEOUT_SECONDS" in text
    assert 'wait_for_process_match "$WD_MATCH"' in text
    assert "global-halt-status --json" in text
    assert "global-halt-refresh --json" in text
    assert "operator-release --json" in text
    assert "global-halt-auto-clear --json" in text
    assert "clear-all-halts --json" in text
    assert "--dry-run" in text
    assert "stack_start_dry_run=1" in text


def test_stop_start_lifecycle_restores_unattended_supervisors() -> None:
    opsctl = _read(OPSCTL_PATH)
    start_stack = _read(PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh")

    assert "STACK_STOPPED.flag" in opsctl
    assert 'stop_launchd_service "com.dankingsley.reboot_resilience_guard"' in opsctl
    assert 'stop_launchd_service "com.dankingsley.failover_hot_standby"' in opsctl
    assert "restore_unattended_support_services" in start_stack
    assert 'recover_launchd_label "com.dankingsley.ops.watchdog" 1' in start_stack
    assert 'recover_launchd_label "com.dankingsley.ops.sql_link_writer" 1' in start_stack
    assert 'recover_launchd_label "com.dankingsley.reboot_resilience_guard" 1' in start_stack
    assert 'rm -f "$STACK_STOPPED_FLAG"' in start_stack


def test_start_stack_compacts_legacy_ops_evidence_before_releasing_stop_flag() -> None:
    start_stack = _read(PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh")
    compactor_call = 'ops_data_plane_compactor.py" --apply --json'
    release_stop_flag = 'rm -f "$STACK_STOPPED_FLAG"'

    assert "BOT_OPS_DATA_PLANE_STARTUP_COMPACTION" in start_stack
    assert compactor_call in start_stack
    assert start_stack.index(compactor_call) < start_stack.index(release_stop_flag)


def test_soak_self_healing_launchd_has_event_trigger_and_polling_fallback() -> None:
    standalone = _read(SOAK_SELF_HEAL_INSTALL_PATH)
    aggregate = _read(OPS_AUTOMATION_INSTALL_PATH)

    for installer in (standalone, aggregate):
        assert "<key>WatchPaths</key>" in installer
        assert "governance/runtime/soak_self_healing.trigger" in installer
        assert "<key>StartInterval</key>" in installer
        assert "<key>ThrottleInterval</key>" in installer


def test_production_hardening_watch_uses_bounded_background_evidence_refresh() -> None:
    standalone = _read(PRODUCTION_HARDENING_WATCH_INSTALL_PATH)
    aggregate = _read(OPS_AUTOMATION_INSTALL_PATH)
    runner = _read(PRODUCTION_HARDENING_WATCH_RUN_PATH)

    for installer in (standalone, aggregate):
        assert "READINESS_EVIDENCE_REFRESH_PROFILE" in installer
        assert "<string>accrual</string>" in installer
        assert "PRODUCTION_PILLAR_REFRESH_ENABLED" in installer
        assert "PRODUCTION_PILLAR_REFRESH_COOLDOWN_MINUTES" in installer
        assert "PRODUCTION_PILLAR_REFRESH_STEP_TIMEOUT_SECONDS" in installer
        assert "<key>ProcessType</key><string>Background</string>" in installer
        assert "<key>LowPriorityIO</key><true/>" in installer
    assert '--profile "${READINESS_EVIDENCE_REFRESH_PROFILE:-accrual}"' in runner
    assert "--profile production" in runner
    assert '${PRODUCTION_PILLAR_REFRESH_COOLDOWN_MINUTES:-45}' in runner


def test_archive_automation_has_no_protected_volume_escape_hatch_or_default() -> None:
    archive_owners = (
        PROJECT_ROOT / "scripts" / "ops" / "cold_archive_compactor.py",
        PROJECT_ROOT / "scripts" / "ops" / "deep_cold_storage_layer.py",
        PROJECT_ROOT / "scripts" / "ops" / "manifest_backed_offload_worker.py",
        PROJECT_ROOT / "scripts" / "ops" / "retention_intelligence_v2.py",
        PROJECT_ROOT / "scripts" / "ops" / "storage_retention_unison.py",
        PROJECT_ROOT / "scripts" / "ops" / "run_data_retention_launchd.sh",
    )

    for path in archive_owners:
        text = _read(path)
        assert "BOT_VIDEO_COLD_ARCHIVE_ROOT" not in text
    assert "BOT_ALLOW_VIDEO_COLD_ARCHIVE=0" in _read(PROJECT_ROOT / "config" / ".env.example")


def test_start_stack_certifies_all_sleeves_restart_handoff() -> None:
    text = _read(PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh")

    assert "wait_for_process_absent" in text
    assert "ALL_SLEEVES_STOP_TIMEOUT_SECONDS" in text
    assert "wait_for_process_stable" in text
    assert "ALL_SLEEVES_START_TIMEOUT_SECONDS" in text
    assert "ALL_SLEEVES_START_STABLE_SECONDS" in text
    assert "all_sleeves=failed_to_stop_before_restart" in text
    assert "all_sleeves=failed_to_start" in text
    assert "all_sleeves=started pid=" in text


def test_start_stack_pauses_watchdog_before_force_restart_drain() -> None:
    text = _read(PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh")

    pause_call = "if ! pause_shadow_watchdog_for_restart; then"
    sleeve_kill = 'pkill -f "scripts/run_all_sleeves.py"'
    assert 'launchctl bootout "$domain" "$plist"' in text
    assert "shadow_watchdog=paused_for_restart" in text
    assert pause_call in text
    assert text.index(pause_call) < text.index(sleeve_kill)
    force_restart = 'if [[ "$FORCE_RESTART" == "1" ]]; then'
    orchestrator_branch = 'if [[ "$ORCHESTRATOR_MODE" == "watchdog" ]]; then'
    assert text.index(force_restart) < text.index(pause_call) < text.index(orchestrator_branch)
    assert "SHADOW_WATCHDOG_PAUSED_FOR_RESTART=1" in text
    assert "resume_shadow_watchdog_after_restart" in text
    assert "restart_exit_cleanup" in text


def test_start_stack_resumes_shadow_watchdog_after_all_sleeves_are_stable() -> None:
    text = _read(PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh")

    stable_marker = 'coinbase_futures_log=logs/watchdog_coinbase_futures_loop.log'
    resume_call = "if ! resume_shadow_watchdog_after_restart; then"
    support_restore = "if ! restore_unattended_support_services; then"
    assert text.index(stable_marker) < text.rindex(resume_call) < text.rindex(support_restore)


def test_start_stack_uses_process_watchdog_as_single_worker_owner() -> None:
    text = _read(PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh")

    assert "run_process_watchdog_handoff" in text
    assert 'OPS_WATCHDOG_REQUIRE_ALL_SLEEVES=1' in text
    assert 'OPS_WATCHDOG_REQUIRE_COINBASE="$WITH_COINBASE"' in text
    assert 'OPS_WATCHDOG_REQUIRE_COINBASE_FUTURES="$WITH_COINBASE"' in text
    assert 'stack_start_owner=process_watchdog' in text
    assert 'all_sleeves_log=logs/watchdog_all_sleeves.log' in text
    assert 'coinbase_log=logs/watchdog_coinbase_loop.log' in text
    assert 'coinbase_futures_log=logs/watchdog_coinbase_futures_loop.log' in text
    assert 'nohup "${CMD[@]}"' not in text
    assert 'nohup "${CB_CMD[@]}"' not in text


def test_start_stack_preflight_is_idempotent_for_running_managed_stack() -> None:
    text = _read(PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh")

    allow_running = 'PREFLIGHT_ARGS+=(--allow-running)'
    kill_duplicates = 'PREFLIGHT_ARGS+=(--apply-kill-duplicates)'
    assert 'grep -F "scripts/run_all_sleeves.py"' in text
    assert allow_running in text
    assert kill_duplicates in text
    assert text.index(allow_running) < text.index(kill_duplicates)


def test_paper_trade_lock_is_present_on_stack_entrypoints() -> None:
    start_stack = _read(PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh")
    all_sleeves_launchd = _read(PROJECT_ROOT / "scripts" / "ops" / "run_all_sleeves_launchd.sh")
    shadow_watchdog_launchd = _read(PROJECT_ROOT / "scripts" / "ops" / "run_shadow_watchdog_launchd.sh")

    for text in (start_stack, all_sleeves_launchd, shadow_watchdog_launchd):
        assert "PAPER_TRADE_LOCK" in text
        assert "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR" in text
        assert "live_data_paper_trade_only" in text


def test_opsctl_stop_supports_dry_run() -> None:
    text = _read(OPSCTL_PATH)

    assert "stack_stop_dry_run=1" in text
    assert "stack_stop_status=ready_to_stop" in text


def test_live_refresh_cleanup_covers_all_schwab_owned_sleeves() -> None:
    opsctl = _read(OPSCTL_PATH)
    start_stack = _read(PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh")

    for text in (opsctl, start_stack):
        assert 'pkill -f "scripts/run_dividend_shadow.py"' in text
        assert 'pkill -f "scripts/run_dividend_capture_shadow.py"' in text
        assert 'pkill -f "scripts/run_bond_shadow.py"' in text
        assert 'pkill -f "scripts/run_fx_shadow.py"' in text


def test_livefeed_refresh_starts_fx_when_all_source_requested() -> None:
    text = _read(OPSCTL_PATH)

    assert 'if [[ "$SOURCE" == "fx" || "$SOURCE" == "all" ]]; then' in text
    assert '"$PROJECT_ROOT/scripts/ops/opsctl.sh" fx-start --paper --live-data' in text
    assert '"$PROJECT_ROOT/scripts/ops/opsctl.sh" fx-start --paper --force-restart --live-data' in text
    assert 'if [[ "$SOURCE" == "fx" || "$SOURCE" == "schwab" || "$SOURCE" == "all" ]]; then' not in text
    assert "livefeed-refresh|live-feed-refresh [paper default] [--dry-run] [--force-restart]" in text
    assert "livefeed_refresh_completed source=$SOURCE" in text
    assert "livefeed-refresh-guard" in text
    assert "livefeed_refresh_guard.py" in text


def test_feed_refresh_is_supervised_ensure_by_default() -> None:
    text = _read(OPSCTL_PATH)

    assert "LIVEFEED_FORCE_RESTART=0" in text
    assert "--force-restart) LIVEFEED_FORCE_RESTART=1" in text
    assert "FEED_REFRESH_LOCK_DIR" in text
    assert "feed_refresh_already_running" in text
    assert "all_sleeves_running" in text
    assert "action=kept_running" in text
    assert 'grep -F "scripts/run_all_sleeves.py"' in text
    assert 'grep -v "scripts/shadow_watchdog.py"' in text
    assert "kill_livefeed_local_mirror()" in text
    assert 'kill -9 "$pid"' in text
    assert "local_mirror_alive = any" in text
    assert 'if [[ "$LIVEFEED_FORCE_RESTART" == "1" ]]; then\n        kill_schwab_live_loops' in text
    assert '"$PROJECT_ROOT/scripts/ops/opsctl.sh" schwab-futures-start --paper --live-data' in text
    assert '"$PROJECT_ROOT/scripts/ops/opsctl.sh" coinbase-start --paper --live-data' in text
    assert "force_restart=$LIVEFEED_FORCE_RESTART" in text


def test_opsctl_exposes_commands_hygiene() -> None:
    text = _read(OPSCTL_PATH)

    assert "commands-hygiene" in text
    assert "scripts/ops/commands_hygiene_bot.py" in text
    assert "command-validity" in text
    assert "commands-verify" in text
    assert "command-audit" in text
    assert "scripts/ops/command_validity_bot.py" in text
    assert "system-drift-guard" in text
    assert "drift-autopilot" in text
    assert "scripts/ops/system_drift_guard.py" in text
    assert "scripts/ops/system_drift_autopilot.py" in text
    assert "system-explainers" in text
    assert "system-explainer-docs" in text
    assert "scripts/ops/system_explainer_docs.py" in text
    assert "options-flow-export-hygiene" in text
    assert "scripts/ops/options_flow_export_hygiene_bot.py" in text
    assert "options-flow-efficiency" in text
    assert "scripts/ops/options_flow_efficiency_bot.py" in text
    assert "bot-stack-report" in text
    assert "scripts/bot_stack_status_report.py" in text
    assert "global-halt-status" in text
    assert "global-halt-refresh" in text
    assert "global-halt-auto-clear" in text
    assert "clear-all-halts" in text
    assert "operator-release" in text
    assert "scripts/global_risk_killswitch.py" in text
    assert "scripts/operator_control.py" in text
    assert "collector-contracts" in text
    assert "scripts/collector_contracts.py" in text
    assert "runtime-throttle" in text
    assert "guard-intelligence|guard-brain" in text
    assert "guard_intelligence_layer.py" in text
    assert "super-intelligence|system-super-intelligence" in text
    assert "system_intelligence_coordinator.py" in text
    assert "pressure-relief|pressure-relief-control|pressure-governor" in text
    assert "pressure_relief_control.py" in text
    assert "health-fast|fast-health" in text
    assert "health_fast.py" in text
    assert "post-restart-settle" in text
    assert "post_restart_settlement.py" in text
    assert "alpha-intelligence-evolution|alpha-advancement" in text
    assert "alpha_intelligence_evolution_expansion.py" in text
    assert "intelligence-layer-advancement|intelligence-layer-v2" in text
    assert "intelligence_layer_advancement_expansion.py" in text
    assert "apex-self-awareness-intelligence|thousand-bot-apex" in text
    assert "apex_self_awareness_intelligence_expansion.py" in text
    assert "deep-recursive-awareness|recursive-awareness|platform-brain-v3" in text
    assert "deep_recursive_awareness_expansion.py" in text
    assert "paper-400-ramp|paper-ramp-400|paper-cap-400" in text
    assert "paper_400_ramp_control.py" in text
    assert "account-position-study [--json] [--day YYYYMMDD] [--profiles CSV]" in text
    assert "account-buildout-plan [--study-file PATH]" in text
    assert "portfolio-risk-ledger [--allocator PATH]" in text
    assert "sleeve-allocator [--one-numbers PATH]" in text
    assert "covered-call-roll-watch [--json] [--today YYYY-MM-DD]" in text
    assert "schwab-account-snapshot-refresh [--json] [--skip-derived]" in text
    assert "notify-test [--enable-imessage]" in text
    assert "spacex-ipo-watch [--json] [--loop] [--symbol SPCX]" in text
    assert ".env.paper_400_ramp_override" in _read(PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh")
    assert "platform-intelligence|platform-intelligence-expansion|bot-admission|bot-lifecycle-manager" in text
    assert "platform_intelligence_expansion.py" in text
    assert ".env.platform_intelligence_override" in _read(PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh")
    assert "platform-brain-v4|grande-brain" in text
    assert "platform_brain_v4.py" in text
    assert ".env.platform_brain_v4_override" in _read(PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh")
    assert "platform-brain-v5|reflex-brain" in text
    assert "platform_brain_v5.py" in text
    assert ".env.platform_brain_v5_override" in _read(PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh")
    assert "platform-stabilization|quality-stabilizer" in text
    assert "platform_stabilization_quality.py" in text
    assert ".env.platform_stabilization_quality_override" in _read(PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh")
    assert "platform-settlement-stabilization|settlement-stabilization" in text
    assert "platform_settlement_stabilization.py" in text
    assert ".env.platform_settlement_stabilization_override" in _read(PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh")


def test_opsctl_exposes_backlog_pcore_accelerator() -> None:
    text = _read(OPSCTL_PATH)

    assert "backlog-pcore-accelerator|pcore-backlog-accelerator|backlog-accelerator" in text
    assert "backlog_pcore_accelerator.py" in text
    assert "backlog-pcore-accelerator [--apply] [--json]" in text


def test_opsctl_exposes_sleeve_ingestion_production_control() -> None:
    text = _read(OPSCTL_PATH)
    env_text = _read(PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh")
    intelligence_text = _read(PROJECT_ROOT / "scripts" / "ops" / "system_intelligence_coordinator.py")

    assert "sleeve-ingestion-production-control|sleeve-ingestion-control|sleeve-ingest-production" in text
    assert "sleeve_ingestion_production_control.py" in text
    assert "sleeve-ingestion-production-control [--apply] [--json]" in text
    assert ".env.sleeve_ingestion_production_override" in env_text
    assert "sleeve_ingestion_production_control_latest.json" in intelligence_text
    assert "sleeve_ingestion_production_control" in intelligence_text


def test_opsctl_exposes_bot_fleet_production_posture() -> None:
    text = _read(OPSCTL_PATH)
    env_text = _read(PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh")
    intelligence_text = _read(PROJECT_ROOT / "scripts" / "ops" / "system_intelligence_coordinator.py")

    assert "bot-fleet-production-posture|bot-fleet-posture|all-bot-production-posture|all-bots-production" in text
    assert "bot_fleet_production_posture.py" in text
    assert "bot-fleet-production-posture [--apply] [--json]" in text
    assert ".env.bot_fleet_production_posture_override" in env_text
    assert env_text.index(".env.sleeve_ingestion_production_override") < env_text.index(".env.bot_fleet_production_posture_override")
    assert "bot_fleet_production_posture_latest.json" in intelligence_text
    assert "_bot_fleet_production_metrics" in intelligence_text


def test_opsctl_exposes_income_operating_platform() -> None:
    text = _read(OPSCTL_PATH)

    assert "income-operating-platform|income-platform-control|income-reliability|income-control" in text
    assert "income_operating_platform.py" in text
    assert "account-policy-context|account-rules|account-context" in text
    assert "account_policy_context.py" in text
    assert "income-operating-platform|income-platform-control [--apply] [--json]" in text


def test_macro_context_sync_does_not_pass_json_to_bls_helper() -> None:
    text = _read(OPSCTL_PATH)

    assert "bls_args=()" in text
    assert "collect_bls_census_data.py\" \"${bls_args[@]}\"" in text
    assert "collect_global_central_bank_context.py\" \"${global_args[@]}\"" in text
    assert "collect_fx_market_context.py\" --json" in text
    assert "collect_public_policy_context.py\" --json" in text
    assert "collect_official_macro_context.py\" \"$@\"" in text
    assert "synchronize_global_central_bank_context.py\" \"${sync_args[@]}\"" in text
    assert 'exit "$macro_rc"' in text
    assert "throttle-control" in text
    assert "scripts/ops/runtime_throttle_control.py" in text
    assert "creative-cotenant-guard" in text
    assert "scripts/ops/creative_cotenant_guard.py" in text
    start_stack_text = _read(PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh")
    assert "--paper-lane-only" in start_stack_text
    assert "paper_execution_lane=singleton_verified_after_restart" in start_stack_text
    assert "livefeed-refresh" in text
    assert "live-feed-refresh" in text
    assert "dashboard-refresh" in text
    assert "runtime-artifact-refresh" in text
    assert "runtime-contract-refresh" in text
    assert "runtime_artifact_refresh.py" in text
    assert "--skip-refresh" in text
    assert "blocked_by_safety_flags" in text
    assert "global-halt-auto-clear" in text
    assert "grade-regression-guard" in text
    assert "grade_regression_guard.py" in text
    assert "grade-regression-autopilot" in text
    assert "grade_regression_autopilot.py" in text
    assert "section-grade-guard" in text
    assert "section_grade_guard.py" in text
    assert "section-grade-autopilot" in text
    assert "section_grade_autopilot.py" in text


def test_official_macro_launchd_refreshes_underlying_fred_context() -> None:
    text = _read(PROJECT_ROOT / "scripts" / "ops" / "run_official_macro_context_launchd.sh")
    installer = _read(PROJECT_ROOT / "scripts" / "ops" / "install_ops_automation_launchd.sh")

    assert 'opsctl.sh" macro-context-sync --json' in text
    assert "OFFICIAL_MACRO_CONTEXT_REFRESH_INTERVAL_SECONDS:-21600" in installer
    assert "MARKET_DATA_ONLY" in installer
    assert "ALLOW_ORDER_EXECUTION" in installer


def test_refresh_finder_logs_publishes_bot_logs_shortcut() -> None:
    text = _read(PROJECT_ROOT / "scripts" / "refresh_finder_logs.sh")

    assert 'BOT_LOGS_FINDER_ALIAS_PATH="${BOT_LOGS_FINDER_ALIAS_PATH:-$HOME/bot_logs}"' in text
    assert 'BOT_LOGS_FINDER_DESKTOP_PATH="${BOT_LOGS_FINDER_DESKTOP_PATH:-$HOME/Desktop/Bot Logs}"' in text
    assert 'BOT_LOGS_FINDER_DESKTOP_SHORTCUTS="${BOT_LOGS_FINDER_DESKTOP_SHORTCUTS:-1}"' in text
    assert 'mount_candidates="${BOT_LOGS_EXTERNAL_MOUNT_CANDIDATES:-$mount_root}"' in text
    assert 'if [[ -d "$candidate/$project_dir" ]]; then' in text
    assert 'if ln -sfn "$BOT_LOGS_ALIAS_TARGET" "$BOT_LOGS_FINDER_ALIAS_PATH"; then' in text
    assert 'if [[ "$BOT_LOGS_FINDER_DESKTOP_SHORTCUTS" == "1" ]]; then' in text
    assert '"bot_logs_alias_path":"$BOT_LOGS_FINDER_ALIAS_PATH"' in text
    assert '"bot_logs_alias_ok":$([[ "$BOT_LOGS_ALIAS_OK" == "1" ]] && echo true || echo false)' in text
    assert '"publish_bot_logs_desktop_shortcut":$([[ "$BOT_LOGS_FINDER_DESKTOP_SHORTCUTS" == "1" ]] && echo true || echo false)' in text


def test_storage_failback_sync_republishes_finder_shortcuts() -> None:
    text = _read(PROJECT_ROOT / "scripts" / "ops" / "storage_failback_sync.py")

    assert 'def _sync_bot_logs_finder_shortcuts' in text
    assert 'bot_logs_finder_sync.py' in text
    assert "'finder_sync': _sync_bot_logs_finder_shortcuts(PROJECT_ROOT)" in text
    assert '"target_volume_device_identifier": str(target_volume.device_identifier) if target_volume else ""' in text
    assert '"candidate_mount_roots": [str(path) for path in resolution.candidate_mount_roots]' in text
    assert '"match_reason": str(resolution.match_reason)' in text


def test_runtime_env_and_storage_guard_support_mount_candidates() -> None:
    env_text = _read(PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh")
    guard_text = _read(PROJECT_ROOT / "scripts" / "ops" / "storage_eject_guard.swift")
    guard_runner = _read(PROJECT_ROOT / "scripts" / "ops" / "run_storage_eject_guard_launchd.sh")
    guard_installer = _read(PROJECT_ROOT / "scripts" / "install_storage_eject_guard_launchd.sh")

    assert '"$PROJECT_ROOT/config/.env.browser_quiet_override"' in env_text
    assert '"$PROJECT_ROOT/config/.env.storage_target_override"' in env_text
    assert 'export BOT_LOGS_EXTERNAL_MOUNT_CANDIDATES="${BOT_LOGS_EXTERNAL_MOUNT_CANDIDATES:-$BOT_LOGS_EXTERNAL_MOUNT}"' in env_text
    assert 'export BOT_LOGS_EXTERNAL_VOLUME_NAME="${BOT_LOGS_EXTERNAL_VOLUME_NAME:-BOT_LOGS}"' in env_text
    assert 'ProcessInfo.processInfo.environment["BOT_LOGS_EXTERNAL_MOUNT_CANDIDATES"]' in guard_text
    assert 'ProcessInfo.processInfo.environment["BOT_LOGS_EXTERNAL_VOLUME_NAME"]' in guard_text
    assert 'ProcessInfo.processInfo.environment["BOT_LOGS_EXTERNAL_VOLUME_UUID"]' in guard_text
    assert 'diskutil list -plist external' in guard_text
    assert 'diskutil mount reason=' in guard_text
    assert 'handleObservedDiskAppeared' in guard_text
    assert 'startMountPollTimer' in guard_text
    assert 'confirmDisappearAndRestartLocal' in guard_text
    assert 'external_still_available_after_disappear' in guard_text
    assert 'storage-switch-external --no-refresh' in guard_text
    assert 'storage-transition-coordinator --transition-mode external --apply --json' in guard_text
    assert 'storage-switch-local --no-refresh local-after-eject' in guard_text
    assert 'split-brain-reconcile --force-failback-if-hashes-match --json' in guard_text
    assert 'external-backlog-drain --apply --follow-through' in guard_text
    assert 'storage-pressure-clearance --apply --max-cycles' in guard_text
    assert 'global-halt-refresh --json' in guard_text
    assert 'global-halt-auto-clear --json' in guard_text
    assert 'storage-reconnect-regression-guard --json' in guard_text
    assert 'currentStorageMode()' in guard_text
    assert 'storage_eject_guard_latest.json' in guard_text
    assert 'external_disconnected_standby' in guard_text
    assert 'externalWriteProbeReady' in guard_text
    assert 'writeTransitionState' in guard_text
    assert 'if mode.hasPrefix("external")' in guard_text
    assert 'stack restart suppressed' in guard_text
    assert 'STORAGE_EJECT_GUARD_BINARY' in guard_runner
    assert 'SWIFT_MODULE_CACHE_PATH="$SWIFT_CACHE_DIR"' in guard_runner
    assert '/usr/bin/swiftc -typecheck "$GUARD_SOURCE"' in guard_runner
    assert 'SWIFT_MODULE_CACHE_PATH="$SWIFT_CACHE_DIR"' in guard_installer
    assert '/usr/bin/swiftc -O "$GUARD_SOURCE"' in guard_installer
    assert 'ThrottleInterval' in guard_installer


def test_storage_reconnect_guard_commands_are_wired() -> None:
    opsctl = _read(OPSCTL_PATH)
    install_ops = _read(PROJECT_ROOT / "scripts" / "ops" / "install_ops_automation_launchd.sh")

    assert "storage-reconnect-regression-guard|storage-reconnect-guard" in opsctl
    assert "storage_reconnect_regression_guard.py" in opsctl
    assert "storage-reconnect-infrabot|storage-recovery-infrabot|storage-auto-recovery-bot" in opsctl
    assert "storage_reconnect_infrabot.py" in opsctl
    assert "storage-reconnect-regression-guard [--skip-launchd] [--skip-swift-parse] [--json]" in opsctl
    assert "storage-reconnect-infrabot [--apply] [--timeout-sec N] [--json]" in opsctl
    assert "run_storage_reconnect_infrabot_launchd.sh" in install_ops
    assert "com.dankingsley.ops.storage_reconnect_infrabot" in install_ops
    assert "install_job \"com.dankingsley.ops.storage_reconnect_infrabot\"" in install_ops


def test_core_bot_catalog_command_is_wired() -> None:
    opsctl = _read(OPSCTL_PATH)

    assert "core-bot-catalog|bot-catalog" in opsctl
    assert "build_core_bot_catalog.py" in opsctl
    assert "core-bot-catalog|bot-catalog [--md-out PATH] [--json-out PATH] [--json]" in opsctl


def test_storage_disaster_recovery_command_and_launchd_are_wired() -> None:
    opsctl = _read(OPSCTL_PATH)
    infra = _read(PROJECT_ROOT / "scripts" / "install_infra_stack_launchd.sh")

    assert "storage-disaster-recovery|storage-recovery-bot" in opsctl
    assert 'storage-disaster-recovery --apply --json' in _read(PROJECT_ROOT / "COMMANDS.md")
    assert 'install_storage_disaster_recovery_launchd.sh' in infra


def test_storage_standby_prune_command_is_wired() -> None:
    opsctl = _read(OPSCTL_PATH)
    commands = _read(PROJECT_ROOT / "COMMANDS.md")

    assert "storage-prune-standby|storage-standby-prune" in opsctl
    assert "storage_standby_prune.py" in opsctl
    assert "./scripts/ops/opsctl.sh storage-prune-standby --json" in commands


def test_options_paper_profile_defaults_are_narrowed() -> None:
    watchdog = _read(WATCHDOG_INSTALL_PATH)
    run_watchdog = _read(PROJECT_ROOT / "scripts" / "ops" / "run_shadow_watchdog_launchd.sh")
    run_all_sleeves = _read(PROJECT_ROOT / "scripts" / "ops" / "run_all_sleeves_launchd.sh")
    start_stack = _read(PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh")
    opsctl = _read(OPSCTL_PATH)

    expected = "default,aggressive,intraday_aggressive,swing_aggressive,options_on_futures,options_on_futures_aggressive"
    assert expected in watchdog
    assert expected in run_watchdog
    assert expected in run_all_sleeves
    assert expected in start_stack
    assert expected in opsctl


def test_paper_mirror_all_active_defaults_to_bounded_authority_cohorts() -> None:
    opsctl = _read(OPSCTL_PATH)
    runtime_env = _read(PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh")
    start_stack = _read(PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh")
    shadow_loop = _read(PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py")
    process_watchdog = _read(PROJECT_ROOT / "scripts" / "ops" / "process_watchdog.py")

    for text in (opsctl, runtime_env, start_stack):
        assert 'PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS:-0' in text
        assert 'PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS:-1' not in text
    assert 'os.getenv("PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS", "0")' in shadow_loop
    assert "env.setdefault('PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS', '0')" in process_watchdog
    assert "--require-coinbase-futures" in process_watchdog
    assert "OPS_WATCHDOG_REQUIRE_COINBASE_FUTURES', '1'" in process_watchdog
    assert "shadow_loop_default_crypto_coinbase_*.json" in process_watchdog
    assert "heartbeat_fresh = " in process_watchdog
    assert "'process_live': bool(process_live)" in process_watchdog
    assert "--require-coinbase-futures" in opsctl


def test_snapshot_debug_reason_argument_is_restart_storm_safe() -> None:
    shadow_loop = _read(PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py")

    assert "def _record_snapshot_debug(symbol: str, event_reason: str, **extra: Any) -> None:" in shadow_loop
    assert '"reason": event_reason,' in shadow_loop
    assert 'row["detail_reason"] = detail_reason' in shadow_loop


def test_shadow_decision_flow_persists_one_livefeed_operator_summary() -> None:
    shadow_loop = _read(PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py")

    assert "institutional_decision_flow_sleeve_playbooks_v4" in shadow_loop
    assert 'institutional_decision_control.get("operator_summary")' in shadow_loop
    assert '"operator_summary": institutional_decision_operator_summary' in shadow_loop
    assert '"institutional_decision_flow_decision_state"' in shadow_loop
    assert '"institutional_decision_flow_stage_progress"' in shadow_loop
    assert '"institutional_decision_flow_playbook_sha256"' in shadow_loop
    assert '"institutional_decision_flow_summary_sha256"' in shadow_loop
    assert '"institutional_decision_flow_ingestion_route_quality_norm"' in shadow_loop
    assert '"institutional_decision_flow_ingestion_route_receipt_valid"' in shadow_loop


def test_runtime_env_has_keychain_handoff_and_calm_support_defaults() -> None:
    runtime_env = _read(PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh")

    assert "SCHWAB_KEYCHAIN_FALLBACK_ENABLED" in runtime_env
    assert "security find-generic-password" in runtime_env
    assert "schwab_trading_bot/SCHWAB_API_KEY" in runtime_env
    assert "schwab_trading_bot/SCHWAB_SECRET" in runtime_env
    assert "schwab_trading_bot/SCHWAB_REDIRECT" in runtime_env
    assert 'ASYNC_PIPELINE_WORKERS="${ASYNC_PIPELINE_WORKERS:-4}"' in runtime_env
    assert 'OPS_SUPPORT_JOBS_BACKGROUND_POLICY="${OPS_SUPPORT_JOBS_BACKGROUND_POLICY:-1}"' in runtime_env
    assert 'SUPPORT_MAINTENANCE_CONCURRENCY="${SUPPORT_MAINTENANCE_CONCURRENCY:-2}"' in runtime_env
    assert runtime_env.index(".env.guard_intelligence_override") < runtime_env.index(".env.process_fanout_guard_override")
    assert ".env.load_shape_smooth_override" in runtime_env
    assert runtime_env.index(".env.accelerator_always_on_override") < runtime_env.index(".env.load_shape_smooth_override")


def test_load_shape_smooth_override_caps_backlog_pressure_without_disabling_drain() -> None:
    override = _read(PROJECT_ROOT / "config" / ".env.load_shape_smooth_override")

    assert "RUNTIME_SMOOTH_MODE_AUTOMATIC=1" in override
    assert "MAINTENANCE_SLOT_SMOOTH_GATE_ENABLED=1" in override
    assert "SQL_LINK_SERVICE_SINGLE_WRITER_ONLY=1" in override
    assert "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES=2" in override
    assert "SQL_LINK_WRITER_NICE=4" in override
    assert "MAINTENANCE_SLOT_RUNTIME_ROOT=$PROJECT_ROOT/runtime/maintenance_slots" in override
    assert "storage_backpressure_autopilot" in override


def test_shadow_watchdog_defaults_cover_fx_and_dividend_capture() -> None:
    watchdog = _read(WATCHDOG_INSTALL_PATH)
    run_watchdog = _read(PROJECT_ROOT / "scripts" / "ops" / "run_shadow_watchdog_launchd.sh")
    run_all_sleeves = _read(PROJECT_ROOT / "scripts" / "ops" / "run_all_sleeves_launchd.sh")

    assert "run_shadow_watchdog_launchd.sh" in watchdog
    assert "default,conservative,aggressive,intraday_aggressive,swing_aggressive,dividend,bond,fx" in watchdog
    assert "default,conservative,aggressive,intraday_aggressive,swing_aggressive,dividend,bond,fx" in run_watchdog
    assert "run_all_sleeves.py" in run_watchdog
    assert "--with-aggressive-modes" in run_watchdog
    assert "RUN_ALL_SLEEVES_WITH_FX" in run_all_sleeves
    assert "RUN_ALL_SLEEVES_WITH_DIVIDEND_CAPTURE" in run_all_sleeves
    assert "SHADOW_WATCHDOG_DIRECT_CHILD_SLEEVES" in run_watchdog
    assert 'SHADOW_WATCHDOG_DIRECT_CHILD_SLEEVES:-0' in run_watchdog
    assert "--watch-dividend-capture" in run_watchdog


def test_livefeed_refresh_market_correlation_is_async_by_default() -> None:
    opsctl = _read(OPSCTL_PATH)

    assert "LIVEFEED_REFRESH_MARKET_CORRELATION_SYNC" in opsctl
    assert "LIVEFEED_REFRESH_MARKET_CORRELATION_ASYNC:-1" in opsctl
    assert "market_correlation_running" in opsctl
    assert "LIVEFEED_REFRESH_MARKET_CORRELATION_TIMEOUT_SECONDS:-90" in opsctl
    assert "market_correlation_sync_started_async" in opsctl
    assert 'nohup "$PROJECT_ROOT/scripts/ops/opsctl.sh" market-correlation-sync \\' in opsctl
    assert 'MARKET_CRYPTO_CORRELATION_TIMEOUT_SECONDS:-90' in opsctl
    assert 'MARKET_CRYPTO_CORRELATION_LOOKBACK_DAYS:-1' in opsctl
    assert 'bounded_market_crypto_correlation_sync.py' in opsctl
    assert 'exec "$PROJECT_ROOT/scripts/ops/opsctl.sh" market-correlation-sync \\' in _read(
        PROJECT_ROOT / "scripts" / "ops" / "run_market_crypto_correlation_launchd.sh"
    )
    assert "bounded_market_crypto_correlation_sync.py" in _read(PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py")
    assert '"--timeout-seconds",' in _read(PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py")
    assert "market-correlation-sync [--lookback-days N] [--bucket-seconds N] [--min-points N] [--timeout-seconds N] [--json]" in opsctl


def test_live_feed_tail_has_memory_aware_heavy_defaults() -> None:
    text = _read(LIVE_FEED_TAIL_PATH)

    assert "BOT_MEMORY_EFFICIENCY_PROFILE" in text
    assert "LIVE_FEED_HEAVY_DEFAULT_LINES" in text
    assert "LIVE_FEED_HEAVY_PRESSURE_LINES" in text
    assert "LIVE_FEED_DECISION_FILE_MODE_PRESSURE" in text
    assert "LIVE_FEED_INCLUDE_WATCHDOG_LOG_DEFAULT" in text
    assert "LIVE_FEED_INCLUDE_COINBASE_WATCHDOG_LOG" in text
    assert "LIVE_FEED_STATUS_SNAPSHOT_DEFAULT" in text
    assert "LIVE_FEED_SHOW_FILE_LIST_DEFAULT" in text
    assert "LIVE_FEED_SUPPRESS_FUTURES_SPECIALIST_INTENTS_DEFAULT" in text
    assert "LIVE_FEED_SUPPRESS_JSON_FRAGMENTS_DEFAULT" in text
    assert "LIVE_FEED_SUPPRESS_TAIL_HEADERS_DEFAULT" in text
    assert "LIVE_FEED_DEDUP_REPEATED_LINES_DEFAULT" in text
    assert "LIVE_FEED_SHOW_KEEPALIVE_DEFAULT" in text
    assert "LIVE_FEED_VISIBLE_KEEPALIVE_ALLOWED" in text
    assert "LIVE_FEED_IMPORTANT_ONLY_DEFAULT" in text
    assert "LIVE_FEED_COLOR" in text
    assert "LIVE_FEED_COLOR_PALETTE" in text
    assert "--color|--highlight" in text
    assert "--no-color|--no-highlight" in text
    assert "--red-only|--red" in text
    assert "--semantic-color|--semantic-colors" in text
    assert "--status-snapshot" in text
    assert "--no-status-snapshot" in text
    assert "--show-files" in text
    assert "--hide-files" in text
    assert "--show-futures-specialist-intents" in text
    assert "--hide-futures-specialist-intents" in text
    assert "--show-json-fragments" in text
    assert "--hide-json-fragments" in text
    assert "--show-tail-headers" in text
    assert "--hide-tail-headers" in text
    assert "--dedupe-repeats" in text
    assert "--no-dedupe-repeats" in text
    assert "--show-keepalive" in text
    assert "--hide-keepalive" in text
    assert "--important-only" in text
    assert "--all-feed-events" in text
    assert "--heavy-ttl" in text
    assert "--no-heavy-ttl" in text
    assert "--heavy-ttl-seconds" in text
    assert "COLOR_ENABLED" in text
    assert "COLOR_PALETTE" in text
    assert "highlight_enabled" in text
    assert "highlight_palette" in text
    assert "LIVE_FEED_HEAVY_INCLUDE_ALL_DECISION_DIRS" in text
    assert "LIVE_FEED_HEAVY_MAX_FOLLOW_FILES" in text
    assert "LIVE_FEED_HEAVY_TAIL_BYTES" in text
    assert "LIVE_FEED_HEAVY_BOOTSTRAP_MAX_LINES" in text
    assert "LIVE_FEED_HEAVY_SNAPSHOT_MAX_LINES" in text
    assert "LIVE_FEED_HEAVY_KEEPALIVE_SECONDS_DEFAULT" in text
    assert "LIVE_FEED_KEEPALIVE_DECISION_SNAPSHOT" in text
    assert "LIVE_FEED_KEEPALIVE_DECISION_EVERY" in text
    assert "LIVE_FEED_DECISION_SNAPSHOT_MAX_LINES" in text
    assert "LIVE_FEED_DECISION_SNAPSHOT_TAIL_BYTES" in text
    assert "LIVE_FEED_MAX_LINE_CHARS" in text
    assert "LIVE_FEED_DECISION_MAX_AGE_HOURS" in text
    assert "LIVEFEED_HEALTH_FILE" in text
    assert "LIVEFEED_HEALTH_WRITER" in text
    assert "livefeed_local_latest.json" in text
    assert "status_snapshot" in text
    assert "show_file_list" in text
    assert "tail_probe_ok" in text
    assert "tail -n 0" in text
    assert "skipped_file_count" in text
    assert "live_feed_files_skipped" in text
    assert "suppress_futures_specialist_intents" in text
    assert "suppress_json_fragments" in text
    assert "suppress_tail_headers" in text
    assert "dedup_repeated_lines" in text
    assert "show_keepalive" in text
    assert "visible_keepalive_allowed" in text
    assert "emit_live_feed_keepalive \"0\"" in text
    assert "emit_livefeed_decision_paper_snapshot" in text
    assert 'if [[ "$HEAVY_REQUESTED" == "1" && "$INCLUDE_DECISIONS" == "1" ]]' in text
    assert "truncate_live_lines 0 0" in text
    assert 'important_override="${2:-$IMPORTANT_ONLY}"' in text
    assert "[decision-latest]" in text
    assert "normalize_decision_record" in text
    assert "lane_source=" in text
    assert "age_source=" in text
    assert "file_age=" in text
    assert "schema={contract_state}" in text
    assert "decision_disposition" in text
    assert "decision_blocking_stage" in text
    assert "disposition={disposition}" in text
    assert "blocking_stage={blocking_stage}" in text
    assert "flow={flow_disposition}" in text
    assert "flow_class={flow_classification}" in text
    assert "flow_stage={flow_stage}" in text
    assert "flow_state={flow_decision_state}" in text
    assert "flow_current={flow_current_stage}" in text
    assert "flow_progress={flow_progress_text}" in text
    assert "flow_blocker={flow_blocking_reason}" in text
    assert "flow_regime={flow_regime_state}" in text
    assert "flow_edge_state={flow_edge_state}" in text
    assert "flow_transition={flow_transition}" in text
    assert "flow_paper_gate={flow_paper_gate}" in text
    assert "flow_live_gate={flow_live_gate}" in text
    assert "flow_playbook={flow_playbook}" in text
    assert "flow_receipt={flow_summary_receipt}" in text
    assert "flow_data_status={flow_data_status}" in text
    assert "flow_data_state={flow_data_state}" in text
    assert "flow_data_quality={flow_data_quality}" in text
    assert "flow_data_paper={flow_data_paper_coverage}" in text
    assert "flow_data_live={flow_data_live_coverage}" in text
    assert "flow_data_receipt={flow_data_receipt}" in text
    assert "*_equities_schwab/decision_*.jsonl" in text
    assert "*_crypto_coinbase/decision_*.jsonl" in text
    assert 'source in {"schwab_futures", "futures"}' in text
    assert 'source in {"coinbase_futures", "futures"}' in text
    assert "flow_summary_receipt and flow_playbook" in text
    assert "flow_contract_priority" in text
    assert "flow_utility={flow_utility}" in text
    assert "flow_qty_cap={flow_quantity_multiplier}" in text
    assert "flow_evidence={flow_evaluation_id}" in text
    assert "flow_family={flow_family}" in text
    assert "flow_policy={flow_policy}" in text
    assert "flow_execution_eligible=" in text
    assert "[decision-route] level=watch status=degraded" in text
    assert 'explicit_level = token_value(lower, "level")' in text
    assert 'failed=[^[:space:]][^[:space:]]*' in text
    assert "[paper]" in text
    assert "[paper-data]" in text
    assert "[paper-profit]" in text
    assert 'raw_state = "recovery_debt" if control_ready and raw_evidence_based else "needs_attention"' in text
    assert "raw_blocking_soak={as_bool(raw_blocking_soak)}" in text
    assert "operational_control_grade" in text
    assert 'f"economic={controlled_grade} "' in text
    assert "raw_gap_to_a={as_num(raw_gap_to_a)}" in text
    assert "weak_zero_entry={as_bool(runtime_enforcement.get('block_new_entries_on_weak_profiles'))}" in text
    assert "reduce_only_open={as_bool(runtime_enforcement.get('keep_sells_and_reduce_only_paths_open'))}" in text
    assert "[paper-truth]" in text
    assert "prioritize_heavy_livefeed_files" in text
    assert "execution_lane_paper_latest.json" in text
    assert "paper_live_data_standard_latest.json" in text
    assert "paper_execution_truth_layer_latest.json" in text
    assert "next_keepalive_seconds" in text
    assert "keepalive_count=0" in text
    assert "important_only" in text
    assert "important_operator_line" in text
    assert "(status-contract|system|collection|fx-provider|auth|schwab-auth|storage|throttle|soak|dashboard|paper|paper-data|paper-profit|profit-hardening|paper-truth|decision-latest|decision-route)" in text
    assert "important_pat" in text
    assert "live_feed_files_hidden" in text
    assert "emit_livefeed_status_snapshot" in text
    assert "env_broker_config" in text
    assert "[broker]" in text
    assert "live_feed_status_contract" in text
    assert "[status-contract]" in text
    assert "build_status_snapshot" in text
    assert "format_status_lines" in text
    assert 'promotion_state = "evidence_pending"' in text
    assert "drop_stale_bootstrap_state_lines" in text
    assert "run_filtered_state_safe_snapshot" in text
    assert "spacex_ipo_downside_watch_latest.json" in text
    assert "macro_event_intelligence_latest.json" in text
    assert "live_macro_latest.json" in text
    assert "mac_notification_watch_state.json" in text
    assert "remote_alert_control_latest.json" in text
    assert "runtime_throttle_control_latest.json" in text
    assert "write_livefeed_health \"running\"" in text
    assert '"writer_mode":"local_mirror"' in text
    assert "include_coinbase_watchdog_log" in text
    assert "HEAVY_TTL_ENABLED_OVERRIDE" in text
    assert "HEAVY_TTL_SECONDS_OVERRIDE" in text
    assert "terminate_livefeed_descendants" in text
    guarded_text = _read(LIVE_FEED_HEAVY_GUARDED_PATH)
    assert "terminate_process_tree" in guarded_text
    assert "stop_heavy_tree" in guarded_text
    assert 'tail -c "$HEAVY_TAIL_BYTES"' in text
    assert "truncate_live_lines" in text
    assert "colorize_line" in text
    assert "[ALERT]" in text
    assert "[WATCH]" in text
    assert "[OK]" in text
    assert "Do not trap EXIT here" in text
    assert "trap 'cleanup_live_feed; exit 130' INT" in text
    assert "trap 'cleanup_live_feed' EXIT" not in text
    assert "append_decision_channel_dir" in text
    assert "governance/channels/decision/$dir/decision_*.jsonl" in text
    assert "append_trade_decision_dir \"paper\"" in text
    assert "local_fallback_storage/decisions/$dir/trade_decisions_*.jsonl" in text
    assert "[FLOW]" in text
    assert "append_decision_file" in text
    assert 'out = "[decision]"' in text
    assert 'append_token(out, "ts", ts)' in text
    assert "master_intent_action" in text
    assert "master_score" in text
    assert "shadow_profile" in text
    assert "driver" in text
    assert "human_length" in text
    assert "looks_like_json_fragment" in text
    assert "suppressible_futures_specialist_intent" in text
    assert "compact_infra_noise_line" in text
    assert "[StorageRoute]" in text
    assert "autosync_skipped_external_low_space" in text
    assert "free_gb" in text
    assert "min_gb" in text
    assert "[ShadowLock] busy" in text
    assert "owner_pid" in text
    assert "symbols" in text
    assert "tail_file_header" in text
    assert "normalized_repeat_key" in text
    assert "[json-fragment skipped" in text
    assert "strategy=? score=?" not in text
    assert "--heavy" in text
    assert '"$SOURCE" == "infra"' in text
    assert "append_heavy_health_files" in text
    assert "--include-watchdog-log" in text
    assert '--no-watchdog-log' in text
    assert 'if [[ "$SOURCE" == "all" && "$INCLUDE_DECISIONS" == "1" ]]' in text
    assert 'HEAVY_INCLUDE_ALL_DECISION_DIRS" == "1"' in text
    assert "capped_files" in text
    assert 'if [[ "$DECISION_FILE_MODE" == "latest_only" ]]' in text
    assert 'if [[ "$INCLUDE_WATCHDOG_LOG" == "1" ]]; then' in text


def test_livefeed_launchd_wrapper_targets_existing_tail_script() -> None:
    text = _read(LIVE_FEED_LAUNCHD_PATH)

    assert "load_runtime_env.sh" in text
    assert "LIVE_FEED_INCLUDE_COINBASE_WATCHDOG_LOG" in text
    assert 'LIVEFEED_LOCAL_SOURCE:-main' in text
    assert 'LIVEFEED_LOCAL_LINES:-80' in text
    assert 'LIVEFEED_LOCAL_HEAVY:-0' in text
    assert 'live_feed_tail.sh" "${args[@]}"' in text


def test_guarded_heavy_livefeed_does_not_disable_visible_keepalive() -> None:
    text = _read(LIVE_FEED_HEAVY_GUARDED_PATH)

    assert "LIVE_FEED_KEEPALIVE_SECONDS=20" in text
    assert "LIVE_FEED_SHOW_KEEPALIVE_DEFAULT=0" not in text


def test_notification_override_pages_tripwire_and_restart_storms() -> None:
    env_text = _read(PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh")
    override = _read(PROJECT_ROOT / "config" / ".env.notification_override")

    assert ".env.notification_override" in env_text
    assert "MAC_NOTIFICATION_WATCH_IMESSAGE_MIN_SEVERITY=critical" in override
    assert "tripwire" in override
    assert "restart_storm" in override


def test_retrain_entrypoints_stamp_trigger_source_and_logs() -> None:
    opsctl = _read(OPSCTL_PATH)
    daily = _read(RETRAIN_DAILY_PATH)
    weekly = _read(RETRAIN_WEEKLY_PATH)

    assert 'RETRAIN_TRIGGER_SOURCE="opsctl_retrain"' in opsctl
    assert 'RETRAIN_TRIGGER_SOURCE="opsctl_retrain_force_full"' in opsctl
    assert 'RETRAIN_TRIGGER_SOURCE="opsctl_retrain_force_targeted"' in opsctl
    assert 'RETRAIN_TRIGGER_SOURCE="launchd_daily_small"' in daily
    assert 'RETRAIN_TRIGGER_SOURCE="launchd_weekly_full"' in weekly
    assert 'RETRAIN_LAUNCH_LOG_PATH="$RUN_LOG"' in daily
    assert 'RETRAIN_LAUNCH_LOG_PATH="$RUN_LOG"' in weekly
    assert 'exec > >(tee -a "$RUN_LOG") 2>&1' in daily
    assert 'exec > >(tee -a "$RUN_LOG") 2>&1' in weekly
    assert "overnight-training-window" in weekly
    assert 'OVERNIGHT_TRAINING_WINDOW_TARGET="${OVERNIGHT_TRAINING_WINDOW_TARGET:-100}"' in weekly
    assert "--window-target" in weekly
