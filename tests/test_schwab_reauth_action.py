import fcntl
import os
import json
import shlex
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from scripts.ops import schwab_reauth_action as action


def test_terminal_command_quotes_root_without_accepting_commands(tmp_path):
    root = tmp_path / "project ' ; echo unsafe"
    assert shlex.split(action.terminal_command(root)) == [
        str(root / ".venv314/bin/python"),
        str(root / "scripts/ops/schwab_reauth_action.py"), "--run"]


def test_click_opens_fixed_command_in_terminal(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(action.subprocess, "run", lambda cmd, **kw:
                        calls.append((cmd, kw)) or SimpleNamespace(returncode=0))
    assert action.open_terminal(tmp_path) == 0
    assert calls[0][0][-1] == action.terminal_command(tmp_path)
    assert calls[0][0][:2] == ["/usr/bin/osascript", "-e"]
    assert calls[0][1]["timeout"] == 15


def test_reauth_runs_only_fixed_flow_with_live_locked(tmp_path, monkeypatch):
    monkeypatch.setattr(action, "callback_busy", lambda: False)
    calls = []
    monkeypatch.setattr(action.subprocess, "run", lambda cmd, **kw:
                        calls.append((cmd, kw)) or SimpleNamespace(returncode=0))
    assert action.run_auth(tmp_path) == 0
    cmd, kw = calls[0]
    assert cmd[1:] == ["token-refresh-interactive", "--force",
                       "--callback-timeout-seconds", "600", "--json"]
    assert kw["env"]["ALLOW_ORDER_EXECUTION"] == "0"
    assert kw["env"]["BOT_LIVE_MONEY_LOCKED_DURING_SOAK"] == "1"


def test_double_click_does_not_start_second_flow(tmp_path, monkeypatch):
    lock = tmp_path / "governance/locks/schwab_reauth_action.lock"
    lock.parent.mkdir(parents=True)
    monkeypatch.setattr(action.subprocess, "run", lambda *a, **kw: pytest.fail("duplicate auth"))
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert action.run_auth(tmp_path) == 0


def test_existing_callback_is_not_killed_or_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(action, "callback_busy", lambda: True)
    monkeypatch.setattr(action.subprocess, "run", lambda *a, **kw: pytest.fail("occupied callback"))
    assert action.run_auth(tmp_path) == 75


def test_fifo_lock_is_rejected(tmp_path):
    lock = tmp_path / "governance/locks/schwab_reauth_action.lock"
    lock.parent.mkdir(parents=True)
    os.mkfifo(lock)
    with pytest.raises(ValueError, match="regular_file"):
        action.run_auth(tmp_path)


def test_old_notification_click_rechecks_successful_auto_refresh(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(action.notifications, "PROJECT_ROOT", tmp_path)
    now = datetime.now(timezone.utc)
    guard = {
        "timestamp_utc": now.isoformat(), "ok": True, "token_ready_after": True,
        "network": {"ok": True}, "ready_min_expires_seconds": 900,
        "auth": {"attempted": False, "ok": True, "reason": "not_needed"},
        "token_after": {"exists": True, "token_path": str(tmp_path / "token.json"),
                        "expires_at": now.timestamp() + 1800, "expires_in_seconds": 1800},
    }
    path = tmp_path / "governance/health/premarket_token_guard_latest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(guard))
    monkeypatch.setattr(action.subprocess, "run", lambda *a, **kw: pytest.fail("unneeded sign-in"))
    monkeypatch.setattr(action, "callback_busy", lambda: pytest.fail("unneeded callback probe"))
    assert action.run_auth(tmp_path) == 0
    assert "already renewed automatically" in capsys.readouterr().out


def test_recovery_check_refuses_symlink_report(tmp_path):
    path = tmp_path / "governance/health/auth_lease_manager_latest.json"
    path.parent.mkdir(parents=True)
    target = tmp_path / "elsewhere.json"
    target.write_text("{}")
    path.symlink_to(target)
    assert not action._already_recovered(tmp_path)
