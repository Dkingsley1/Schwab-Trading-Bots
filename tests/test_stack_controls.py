import argparse
from pathlib import Path

import scripts.run_all_sleeves as run_all_sleeves


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMANDS_PATH = PROJECT_ROOT / "COMMANDS.md"
OPSCTL_PATH = PROJECT_ROOT / "scripts" / "ops" / "opsctl.sh"
LIVE_FEED_TAIL_PATH = PROJECT_ROOT / "scripts" / "ops" / "live_feed_tail.sh"
WATCHDOG_INSTALL_PATH = PROJECT_ROOT / "scripts" / "install_shadow_watchdog_launchd.sh"
INFRA_INSTALL_PATH = PROJECT_ROOT / "scripts" / "install_infra_stack_launchd.sh"
OPS_AUTOMATION_INSTALL_PATH = PROJECT_ROOT / "scripts" / "ops" / "install_ops_automation_launchd.sh"
STORAGE_BACKPRESSURE_AUTOPILOT_RUN_PATH = PROJECT_ROOT / "scripts" / "ops" / "run_storage_backpressure_autopilot_launchd.sh"
RETRAIN_DAILY_PATH = PROJECT_ROOT / "scripts" / "retrain_daily_small_batch.sh"
RETRAIN_WEEKLY_PATH = PROJECT_ROOT / "scripts" / "retrain_weekly_full_sweep.sh"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_infra_stack_installer_includes_ops_and_daily_verify() -> None:
    text = _read(INFRA_INSTALL_PATH)

    assert "install_daily_auto_verify_launchd.sh" in text
    assert "scripts/ops/install_ops_automation_launchd.sh" in text


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
    assert "com.dankingsley.ops.master_infrastructure_supervisor" in text


def test_storage_backpressure_autopilot_launchd_runs_multi_cycle_clearance() -> None:
    text = _read(STORAGE_BACKPRESSURE_AUTOPILOT_RUN_PATH)

    assert "--max-cycles" in text
    assert "STORAGE_BACKPRESSURE_AUTOPILOT_MAX_CYCLES" in text
    assert "--target-pending-lines" in text
    assert "STORAGE_BACKPRESSURE_AUTOPILOT_TARGET_PENDING_LINES" in text
    assert "--target-retention-debt-gb" in text
    assert "STORAGE_BACKPRESSURE_AUTOPILOT_TARGET_RETENTION_DEBT_GB" in text


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
    assert "### Refresh the live loops without reinstalling the stack watchdog" in text
    assert "./scripts/ops/opsctl.sh livefeed-refresh" in text
    assert "### Stop the stack" in text
    assert "### Validate documented commands" in text
    assert "./scripts/ops/opsctl.sh command-validity --json" in text
    assert "### Review the cross-system drift mesh" in text
    assert "./scripts/ops/opsctl.sh system-drift-guard --json" in text
    assert "### Repair safe cross-system drift surfaces" in text
    assert "./scripts/ops/opsctl.sh system-drift-autopilot --apply --json" in text
    assert "### Heavy live feed view across all sections" in text
    assert "./scripts/ops/opsctl.sh feed --source all --heavy" in text
    assert "### Heavy infrastructure live feed view" in text
    assert "./scripts/ops/opsctl.sh feed --source infra --heavy --lines 160" in text
    assert "### Light live feed tail for all feeds" in text
    assert "### Active bot stack PDF" in text
    assert "./scripts/ops/open_report_artifact.sh botstack" in text


def test_start_stack_blocks_cleanly_on_operator_stop_or_global_halt() -> None:
    text = _read(PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh")

    assert "OPERATOR_STOP.flag" in text
    assert "GLOBAL_TRADING_HALT.flag" in text
    assert "stack_start_status=blocked_by_safety_flags" in text
    assert "global-halt-status --json" in text
    assert "global-halt-refresh --json" in text
    assert "operator-release --json" in text
    assert "global-halt-auto-clear --json" in text
    assert "clear-all-halts --json" in text
    assert "--dry-run" in text
    assert "stack_start_dry_run=1" in text


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
    assert '"$PROJECT_ROOT/scripts/ops/opsctl.sh" fx-start --paper --force-restart --live-data' in text
    assert 'if [[ "$SOURCE" == "fx" || "$SOURCE" == "schwab" || "$SOURCE" == "all" ]]; then' not in text
    assert "livefeed-refresh|live-feed-refresh [paper default] [--dry-run]" in text
    assert "livefeed_refresh_completed source=$SOURCE" in text


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
    assert "post-restart-settle" in text
    assert "post_restart_settlement.py" in text
    assert "alpha-intelligence-evolution|alpha-advancement" in text
    assert "alpha_intelligence_evolution_expansion.py" in text
    assert "intelligence-layer-advancement|intelligence-layer-v2" in text
    assert "intelligence_layer_advancement_expansion.py" in text
    assert "apex-self-awareness-intelligence|thousand-bot-apex" in text
    assert "apex_self_awareness_intelligence_expansion.py" in text


def test_macro_context_sync_does_not_pass_json_to_bls_helper() -> None:
    text = _read(OPSCTL_PATH)

    assert "bls_args=()" in text
    assert "collect_bls_census_data.py\" \"${bls_args[@]}\"" in text
    assert "throttle-control" in text
    assert "scripts/ops/runtime_throttle_control.py" in text
    assert "creative-cotenant-guard" in text
    assert "scripts/ops/creative_cotenant_guard.py" in text
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


def test_paper_mirror_all_active_defaults_to_calm_mode() -> None:
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


def test_shadow_watchdog_defaults_cover_fx_and_dividend_capture() -> None:
    watchdog = _read(WATCHDOG_INSTALL_PATH)
    run_watchdog = _read(PROJECT_ROOT / "scripts" / "ops" / "run_shadow_watchdog_launchd.sh")
    run_all_sleeves = _read(PROJECT_ROOT / "scripts" / "ops" / "run_all_sleeves_launchd.sh")

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
    assert "LIVE_FEED_COLOR" in text
    assert "LIVE_FEED_COLOR_PALETTE" in text
    assert "--color|--highlight" in text
    assert "--no-color|--no-highlight" in text
    assert "--red-only|--red" in text
    assert "--semantic-color|--semantic-colors" in text
    assert "COLOR_ENABLED" in text
    assert "COLOR_PALETTE" in text
    assert "highlight_enabled" in text
    assert "highlight_palette" in text
    assert "LIVE_FEED_HEAVY_INCLUDE_ALL_DECISION_DIRS" in text
    assert "LIVE_FEED_HEAVY_MAX_FOLLOW_FILES" in text
    assert "LIVE_FEED_HEAVY_TAIL_BYTES" in text
    assert "LIVE_FEED_HEAVY_BOOTSTRAP_MAX_LINES" in text
    assert "LIVE_FEED_HEAVY_SNAPSHOT_MAX_LINES" in text
    assert "LIVE_FEED_MAX_LINE_CHARS" in text
    assert "LIVE_FEED_DECISION_MAX_AGE_HOURS" in text
    assert 'tail -c "$HEAVY_TAIL_BYTES"' in text
    assert "truncate_live_lines" in text
    assert "colorize_line" in text
    assert "[ALERT]" in text
    assert "[WATCH]" in text
    assert "[OK]" in text
    assert "[FLOW]" in text
    assert "append_decision_file" in text
    assert "[decision] ts=" in text
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
