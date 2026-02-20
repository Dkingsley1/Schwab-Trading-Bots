import os
import signal
import subprocess
import sys
from typing import List


VALID_MODES = {"shadow", "paper", "live"}


def parse_modes() -> List[str]:
    raw = os.getenv("SWITCHBOARD_MODES", "shadow,paper")
    modes = [m.strip().lower() for m in raw.split(",") if m.strip()]
    for m in modes:
        if m not in VALID_MODES:
            raise ValueError(f"Invalid mode '{m}'. Allowed: {sorted(VALID_MODES)}")
    if not modes:
        raise ValueError("No modes provided in SWITCHBOARD_MODES")
    return modes


def launch_modes(project_root: str, modes: List[str]) -> List[subprocess.Popen]:
    py = os.path.join(project_root, ".venv312", "bin", "python")
    main_py = os.path.join(project_root, "main.py")

    children: List[subprocess.Popen] = []
    for mode in modes:
        env = os.environ.copy()
        env["BOT_MODE"] = mode
        proc = subprocess.Popen([py, main_py], env=env)
        children.append(proc)
        print(f"Started mode={mode} pid={proc.pid}")

    return children


def shutdown(children: List[subprocess.Popen]) -> None:
    for p in children:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)


def main() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    modes = parse_modes()
    children = launch_modes(project_root, modes)

    try:
        exit_codes = [p.wait() for p in children]
        print(f"Exit codes: {exit_codes}")
        if any(code != 0 for code in exit_codes):
            raise SystemExit(1)
    except KeyboardInterrupt:
        print("Interrupted. Shutting down child processes...")
        shutdown(children)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
