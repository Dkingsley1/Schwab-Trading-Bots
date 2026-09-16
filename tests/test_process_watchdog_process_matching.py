import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import process_watchdog as watchdog

COINBASE_COMMAND = (
    "/Users/dankingsley/PycharmProjects/schwab_trading_bot/.venv314/bin/python "
    "/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/run_shadow_training_loop.py "
    "--runtime-cpu-class market_decision --broker coinbase --symbols BTC-USD,ETH-USD "
    "--interval-seconds 20 --max-iterations 0"
)


def test_command_match_allows_runtime_flags_between_script_and_broker() -> None:
    assert watchdog._command_matches_pattern(
        COINBASE_COMMAND,
        "scripts/run_shadow_training_loop.py --broker coinbase",
    )


def test_coinbase_process_count_detects_runtime_routed_collector(monkeypatch) -> None:
    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        assert args[0] == ["ps", "-axo", "stat=,command="]
        return SimpleNamespace(stdout=f"S+ {COINBASE_COMMAND}\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        watchdog._proc_running(
            "scripts/run_shadow_training_loop.py --broker coinbase",
            exclude_patterns=["--profile crypto_futures"],
        )
        == 1
    )


def test_coinbase_spot_match_excludes_futures_profile(monkeypatch) -> None:
    futures_command = f"{COINBASE_COMMAND} --profile crypto_futures"

    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(stdout=f"S+ {futures_command}\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        watchdog._proc_running(
            "scripts/run_shadow_training_loop.py --broker coinbase",
            exclude_patterns=["--profile crypto_futures"],
        )
        == 0
    )


def test_process_count_falls_back_to_pgrep_when_ps_is_denied(monkeypatch) -> None:
    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        if args[0] == ["ps", "-axo", "stat=,command="]:
            raise PermissionError("ps denied")
        assert args[0] == [
            "pgrep",
            "-fl",
            "scripts/run_shadow_training_loop.py",
        ]
        return SimpleNamespace(
            stdout=(
                f"101 {COINBASE_COMMAND}\n"
                f"102 {COINBASE_COMMAND} --profile crypto_futures\n"
            )
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        watchdog._proc_running(
            "scripts/run_shadow_training_loop.py --broker coinbase",
            exclude_patterns=["--profile crypto_futures"],
        )
        == 1
    )


def test_matching_pids_falls_back_to_pgrep_when_ps_is_denied(monkeypatch) -> None:
    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        if args[0] == ["ps", "-axo", "pid,command"]:
            raise PermissionError("ps denied")
        assert args[0] == [
            "pgrep",
            "-fl",
            "scripts/run_shadow_training_loop.py",
        ]
        return SimpleNamespace(
            stdout=(
                f"101 {COINBASE_COMMAND}\n"
                f"102 {COINBASE_COMMAND} --profile crypto_futures\n"
            )
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert watchdog._matching_pids(
        "scripts/run_shadow_training_loop.py --broker coinbase",
        exclude_patterns=["--profile crypto_futures"],
    ) == [101]


def test_process_count_returns_unknown_when_all_process_scans_are_unavailable(
    monkeypatch,
) -> None:
    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        if args[0] == ["ps", "-axo", "stat=,command="]:
            raise PermissionError("ps denied")
        return SimpleNamespace(
            returncode=3,
            stdout="",
            stderr="pgrep: Cannot get process list",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        watchdog._proc_running("scripts/run_shadow_training_loop.py --broker coinbase")
        == watchdog.PROCESS_SCAN_UNKNOWN
    )


def test_supervised_restart_uses_runtime_flag_tolerant_coinbase_match() -> None:
    start_stack = (PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh").read_text(
        encoding="utf-8"
    )
    opsctl = (PROJECT_ROOT / "scripts" / "ops" / "opsctl.sh").read_text(
        encoding="utf-8"
    )
    process_watchdog = (
        PROJECT_ROOT / "scripts" / "ops" / "process_watchdog.py"
    ).read_text(encoding="utf-8")

    assert 'grep -F "scripts/run_shadow_training_loop.py"' in start_stack
    assert 'grep -F -- "--broker coinbase"' in start_stack
    assert "wait_for_coinbase_spot_stable" in start_stack
    assert "wait_for_coinbase_futures_stable" in start_stack
    assert "pause_process_watchdog_for_restart" in start_stack
    assert "resume_process_watchdog_after_restart" in start_stack
    assert (
        '"scripts/run_shadow_training_loop.py --broker coinbase --symbols"'
        not in start_stack
    )
    assert 'index($0, "scripts/run_shadow_training_loop.py") > 0' in opsctl
    assert 'index($0, "--broker coinbase") > 0' in opsctl
    assert (
        process_watchdog.count("'--runtime-cpu-class'")
        + process_watchdog.count('"--runtime-cpu-class"')
        >= 2
    )


def test_restart_fence_blocks_unauthorized_watchdog_repairs(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        watchdog, "maintenance_hold_snapshot", lambda _root: {"active": False}
    )
    monkeypatch.setattr(
        watchdog,
        "stack_restart_fence_snapshot",
        lambda _root: {
            "active": True,
            "token": "owner-token",
            "payload": {"token": "owner-token"},
        },
    )
    monkeypatch.setattr(watchdog, "OPERATOR_STOP_FLAG", tmp_path / "OPERATOR_STOP.flag")
    monkeypatch.setattr(
        watchdog, "GLOBAL_HALT_FLAG", tmp_path / "GLOBAL_TRADING_HALT.flag"
    )
    monkeypatch.delenv("STACK_RESTART_FENCE_TOKEN", raising=False)

    blocked = watchdog._safety_pause_state()
    assert blocked["active"] is True
    assert blocked["reason"] == "stack_restart_in_progress"
    assert "token" not in blocked["stack_restart_fence"]

    monkeypatch.setenv("STACK_RESTART_FENCE_TOKEN", "owner-token")
    authorized = watchdog._safety_pause_state()
    assert authorized["active"] is False
    assert authorized["stack_restart_fence"]["authorized_caller"] is True


def test_watchdog_refuses_to_spawn_critical_child_from_low_priority_parent(
    monkeypatch,
) -> None:
    monkeypatch.setattr(watchdog.os, "getpriority", lambda _which, _who: 5)
    state = watchdog._launcher_priority_state({"required_launcher_max_nice": 0})
    assert state["required"] is True
    assert state["compliant"] is False
    assert state["reason"] == "launcher_priority_noncompliant"
