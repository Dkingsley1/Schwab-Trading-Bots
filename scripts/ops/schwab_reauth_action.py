"""Click-only launcher for one operator-supervised Schwab authorization flow."""

import argparse
import fcntl
import os
from pathlib import Path
import shlex
import socket
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.storage_router import inspect_storage_path


def terminal_command(root=ROOT):
    return shlex.join([str(root / ".venv314/bin/python"),
                       str(root / "scripts/ops/schwab_reauth_action.py"), "--run"])


def open_terminal(root=ROOT):
    script = '''on run argv
tell application "Terminal"
activate
do script (item 1 of argv)
end tell
end run'''
    return subprocess.run(["/usr/bin/osascript", "-e", script, terminal_command(root)],
                          check=False, timeout=15).returncode


def callback_busy():
    try:
        with socket.create_connection(("127.0.0.1", 8182), timeout=0.2):
            return True
    except OSError:
        return False


def run_auth(root=ROOT):
    lock = root / "governance/locks/schwab_reauth_action.lock"
    route = inspect_storage_path(lock, boundary_root=root, allow_external=False)
    if route["status"] not in {"present", "missing"} or route.get("symlinks"):
        raise ValueError("unsafe_auth_action_lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("auth_action_lock_requires_regular_file")
    with os.fdopen(fd, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Schwab sign-in is already in progress. Use the existing browser window.")
            return 0
        if callback_busy():
            print("The Schwab callback port is already in use. Finish the existing login; no new flow was started.")
            return 75
        env = {**os.environ, "MARKET_DATA_ONLY": "1", "ALLOW_ORDER_EXECUTION": "0",
               "TOP_BOT_ENABLE_LIVE_EXECUTION": "0", "EXECUTION_LANE_LIVE_ENABLED": "0",
               "BOT_LIVE_MONEY_LOCKED_DURING_SOAK": "1",
               "SCHWAB_AUTH_ALLOW_BROWSER_OPEN": "1", "SCHWAB_AUTH_BROWSER_DISABLED": "0"}
        return subprocess.run([
            str(root / "scripts/ops/opsctl.sh"), "token-refresh-interactive", "--force",
            "--callback-timeout-seconds", "600", "--json",
        ], cwd=root, env=env, check=False).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    return run_auth() if args.run else open_terminal()


if __name__ == "__main__":
    raise SystemExit(main())
