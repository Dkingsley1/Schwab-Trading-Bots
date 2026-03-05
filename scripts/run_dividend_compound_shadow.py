import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT_ROOT / ".venv312" / "bin" / "python"
RUNNER = PROJECT_ROOT / "scripts" / "run_dividend_shadow.py"


def main() -> int:
    if not VENV_PY.exists():
        print(f"ERROR: missing venv python: {VENV_PY}")
        return 2
    if not RUNNER.exists():
        print(f"ERROR: missing runner script: {RUNNER}")
        return 2

    env = os.environ.copy()
    env.setdefault("DIVIDEND_STRATEGY_MODE", "compound")

    cmd = [str(VENV_PY), str(RUNNER), *sys.argv[1:]]
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
