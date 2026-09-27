from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPSCTL = PROJECT_ROOT / "scripts" / "ops" / "opsctl.sh"


def test_explicit_cpu_policy_owns_detached_job_priority() -> None:
    text = OPSCTL.read_text(encoding="utf-8")

    assert "unsetopt BG_NICE" in text


def test_forced_observer_restarts_use_exclusive_fences() -> None:
    text = OPSCTL.read_text(encoding="utf-8")

    assert 'engage_component_restart_fence "schwab_futures"' in text
    assert 'engage_component_restart_fence "coinbase_spot"' in text
    assert 'engage_component_restart_fence "coinbase_futures"' in text
    assert "trap release_component_restart_fence EXIT HUP INT TERM" in text


def test_schwab_futures_restart_is_token_aware_and_stable() -> None:
    text = OPSCTL.read_text(encoding="utf-8")
    pattern = "scripts/run_shadow_training_loop.py --broker schwab --profile $FUTURES_PROFILE"

    assert f'terminate_runtime_processes "{pattern}"' in text
    assert f'wait_runtime_process_stable "{pattern}" 20 5' in text
    assert f'pkill -f "{pattern}"' not in text


def test_forced_observer_restarts_clear_dead_owner_locks() -> None:
    text = OPSCTL.read_text(encoding="utf-8")

    assert text.count('lock_watchdog.py" --apply --json') >= 3
    assert "wait_process_predicate_stable coinbase_spot_running 20 5" in text
    assert "wait_process_predicate_stable coinbase_futures_running 20 5" in text
