import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ZSH = "/bin/zsh" if Path("/bin/zsh").exists() else shutil.which("zsh")
pytestmark = pytest.mark.skipif(ZSH is None, reason="production wrapper requires zsh")


def _fixture(tmp_path):
    ops = tmp_path / "scripts" / "ops"
    ops.mkdir(parents=True)
    wrapper = ops / "run_production_hardening_watch_launchd.sh"
    wrapper.write_bytes((ROOT / "scripts/ops" / wrapper.name).read_bytes())
    command = ops / "opsctl.sh"
    command.write_text("""#!/bin/sh
printf '%s|live=%s|lane=%s\\n' "$*" "$ALLOW_ORDER_EXECUTION" "$EXECUTION_LANE_LIVE_ENABLED" >> "$TRACE"
case "$*" in
  *"--profile accrual"*)
    if [ "${WAIT_FOR_RELEASE:-0}" = 1 ]; then
      while [ ! -f "$RELEASE" ]; do sleep 0.05; done
    fi
    exit "${ACCRUAL_RC:-0}" ;;
  *"--profile production"*) exit "${PRODUCTION_RC:-0}" ;;
  production-hardening-watch*) exit "${WATCH_RC:-0}" ;;
esac
""")
    command.chmod(0o700)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "TRACE": str(tmp_path / "trace"),
        "RELEASE": str(tmp_path / "release"),
        "PRODUCTION_HARDENING_WATCH_LOCK_ROOT": str(tmp_path / "locks"),
        "PRODUCTION_HARDENING_WATCH_EXECUTE_SAFE_REPAIRS": "1",
        "PRODUCTION_HARDENING_WATCH_EXECUTE_ON_WATCH": "1",
    }
    return wrapper, env


@pytest.mark.parametrize(
    "accrual,production,watcher,expected",
    [(2, 0, 0, 2), (0, 7, 0, 7), (0, 0, 2, 2), (0, 0, 0, 0)],
)
def test_failed_refresh_does_not_starve_independent_profiles_or_watcher(
    tmp_path, accrual, production, watcher, expected
):
    wrapper, env = _fixture(tmp_path)
    env.update(
        ACCRUAL_RC=str(accrual), PRODUCTION_RC=str(production), WATCH_RC=str(watcher)
    )
    result = subprocess.run(
        [ZSH, str(wrapper)], env=env, text=True, capture_output=True, timeout=10
    )
    assert result.returncode == expected, result.stderr
    calls = Path(env["TRACE"]).read_text().splitlines()
    assert len(calls) == 3
    assert "--profile accrual" in calls[0] and "--profile production" in calls[1]
    assert calls[2].startswith("production-hardening-watch")
    assert all("live=0|lane=0" in row for row in calls)
    assert ("--execute-safe-repairs" in calls[2]) == (accrual == production == 0)
    assert ("--execute-on-watch" in calls[2]) == (accrual == production == 0)


def test_old_legacy_lock_is_not_stolen_by_age(tmp_path):
    wrapper, env = _fixture(tmp_path)
    legacy = (
        Path(env["PRODUCTION_HARDENING_WATCH_LOCK_ROOT"])
        / "production_hardening_watch_launchd.lock"
    )
    legacy.mkdir(parents=True)
    os.utime(legacy, (1, 1))
    result = subprocess.run(
        [ZSH, str(wrapper)], env=env, text=True, capture_output=True, timeout=10
    )
    assert result.returncode == 75
    assert "legacy_lock_present" in result.stderr
    assert legacy.exists() and not Path(env["TRACE"]).exists()


def test_os_lock_excludes_concurrent_wrappers_and_releases_after_completion(tmp_path):
    wrapper, env = _fixture(tmp_path)
    first = subprocess.Popen(
        [ZSH, str(wrapper)],
        env={**env, "WAIT_FOR_RELEASE": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not Path(env["TRACE"]).exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert Path(env["TRACE"]).exists()
        lock = (
            Path(env["PRODUCTION_HARDENING_WATCH_LOCK_ROOT"])
            / "production_hardening_watch_launchd.lockfile"
        )
        os.utime(lock, (1, 1))
        second = subprocess.run(
            [ZSH, str(wrapper)], env=env, text=True, capture_output=True, timeout=5
        )
        assert second.returncode == 0
        assert "wrapper_lock_unavailable" in second.stderr
        assert len(Path(env["TRACE"]).read_text().splitlines()) == 1
    finally:
        Path(env["RELEASE"]).touch()
        first.communicate(timeout=5)
    assert first.returncode == 0
    third = subprocess.run(
        [ZSH, str(wrapper)], env=env, text=True, capture_output=True, timeout=5
    )
    assert third.returncode == 0, third.stderr
    assert len(Path(env["TRACE"]).read_text().splitlines()) == 6
