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


def test_commands_start_stop_section_uses_stack_entrypoint() -> None:
    text = _read(COMMANDS_PATH)

    assert "### Start the full live stack" in text
    assert "./scripts/ops/opsctl.sh start" in text
    assert "### Refresh the live loops without reinstalling the stack watchdog" in text
    assert "### Stop the stack" in text
    assert "### Validate documented commands" in text
    assert "./scripts/ops/opsctl.sh command-validity --json" in text
    assert "### Heavy live feed view across all feeds and decisions" in text
    assert "./scripts/ops/opsctl.sh feed --source all --include-decisions" in text
    assert "### Light live feed tail for all feeds" in text
    assert "### Active bot stack PDF" in text
    assert "./scripts/ops/opsctl.sh bot-stack-report --top 25 --render-pdf --allow-gui-pdf-renderer" in text


def test_opsctl_exposes_commands_hygiene() -> None:
    text = _read(OPSCTL_PATH)

    assert "commands-hygiene" in text
    assert "scripts/ops/commands_hygiene_bot.py" in text
    assert "command-validity" in text
    assert "commands-verify" in text
    assert "command-audit" in text
    assert "scripts/ops/command_validity_bot.py" in text
    assert "options-flow-export-hygiene" in text
    assert "scripts/ops/options_flow_export_hygiene_bot.py" in text
    assert "options-flow-efficiency" in text
    assert "scripts/ops/options_flow_efficiency_bot.py" in text
    assert "bot-stack-report" in text
    assert "scripts/bot_stack_status_report.py" in text


def test_options_paper_profile_defaults_are_narrowed() -> None:
    watchdog = _read(WATCHDOG_INSTALL_PATH)
    run_watchdog = _read(PROJECT_ROOT / "scripts" / "ops" / "run_shadow_watchdog_launchd.sh")
    run_all_sleeves = _read(PROJECT_ROOT / "scripts" / "ops" / "run_all_sleeves_launchd.sh")
    start_stack = _read(PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh")
    opsctl = _read(OPSCTL_PATH)

    assert "default,aggressive,intraday_aggressive,swing_aggressive" in watchdog
    assert "default,aggressive,intraday_aggressive,swing_aggressive" in run_watchdog
    assert "default,aggressive,intraday_aggressive,swing_aggressive" in run_all_sleeves
    assert "default,aggressive,intraday_aggressive,swing_aggressive" in start_stack
    assert "default,aggressive,intraday_aggressive,swing_aggressive" in opsctl


def test_shadow_watchdog_defaults_cover_fx_and_dividend_capture() -> None:
    watchdog = _read(WATCHDOG_INSTALL_PATH)
    run_watchdog = _read(PROJECT_ROOT / "scripts" / "ops" / "run_shadow_watchdog_launchd.sh")
    run_all_sleeves = _read(PROJECT_ROOT / "scripts" / "ops" / "run_all_sleeves_launchd.sh")

    assert "default,conservative,aggressive,intraday_aggressive,swing_aggressive,dividend,bond,fx" in watchdog
    assert "default,conservative,aggressive,intraday_aggressive,swing_aggressive,dividend,bond,fx" in run_watchdog
    assert "RUN_ALL_SLEEVES_WITH_FX" in run_all_sleeves
    assert "RUN_ALL_SLEEVES_WITH_DIVIDEND_CAPTURE" in run_all_sleeves
    assert "--watch-dividend-capture" in run_watchdog


def test_live_feed_tail_has_memory_aware_heavy_defaults() -> None:
    text = _read(LIVE_FEED_TAIL_PATH)

    assert "BOT_MEMORY_EFFICIENCY_PROFILE" in text
    assert "LIVE_FEED_HEAVY_DEFAULT_LINES" in text
    assert "LIVE_FEED_HEAVY_PRESSURE_LINES" in text
    assert "LIVE_FEED_DECISION_FILE_MODE_PRESSURE" in text
    assert 'if [[ "$SOURCE" == "all" && "$INCLUDE_DECISIONS" == "1" ]]' in text
    assert 'if [[ "$DECISION_FILE_MODE" == "latest_only" ]]' in text


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
