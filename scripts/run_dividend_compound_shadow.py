import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python

os.environ.setdefault("BOT_RUNTIME_LANE", os.getenv("BOT_SHADOW_RUNTIME_LANE", "shadow"))

VENV_PY = resolve_runtime_python(PROJECT_ROOT)
RUNNER = PROJECT_ROOT / "scripts" / "run_dividend_shadow.py"


def main() -> int:
    if not VENV_PY.exists():
        print(f"ERROR: missing venv python: {VENV_PY}")
        return 2
    if not RUNNER.exists():
        print(f"ERROR: missing runner script: {RUNNER}")
        return 2

    env = os.environ.copy()
    env.setdefault("MARKET_DATA_ONLY", "1")
    env.setdefault("ALLOW_ORDER_EXECUTION", "0")
    env.setdefault("SHADOW_DOMAIN", "equities")
    env.setdefault("SHADOW_PROFILE", "dividend_compound")
    env.setdefault("DIVIDEND_STRATEGY_MODE", "compound")

    cmd = [str(VENV_PY), str(RUNNER), *sys.argv[1:]]
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
