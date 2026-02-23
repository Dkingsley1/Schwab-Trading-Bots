import argparse
import os
import shlex
import subprocess
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT_ROOT / ".venv312" / "bin" / "python"
WEEKLY_RETRAIN = PROJECT_ROOT / "scripts" / "weekly_retrain.py"
PARALLEL_SHADOWS = PROJECT_ROOT / "scripts" / "run_parallel_shadows.py"


def _scan_processes() -> list[tuple[int, str]]:
    proc = subprocess.run(
        ["ps", "-ax", "-o", "pid=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout or ""

    rows: list[tuple[int, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1]
        if "manual_retrain_with_pause.py" in cmd:
            continue
        rows.append((pid, cmd))
    return rows


def _find_shadow_targets() -> tuple[list[int], str | None]:
    rows = _scan_processes()

    parallel = [(pid, cmd) for pid, cmd in rows if "scripts/run_parallel_shadows.py" in cmd]
    if parallel:
        pids = sorted({pid for pid, _ in parallel})
        restart_cmd = parallel[0][1]
        return pids, restart_cmd

    shadow = [(pid, cmd) for pid, cmd in rows if "scripts/run_shadow_training_loop.py" in cmd]
    if shadow:
        pids = sorted({pid for pid, _ in shadow})
        return pids, None

    return [], None


def _terminate_pids(pids: list[int], timeout_seconds: int) -> None:
    if not pids:
        return

    for pid in pids:
        subprocess.run(["kill", "-TERM", str(pid)], check=False)

    start = time.time()
    while time.time() - start < timeout_seconds:
        alive = []
        for pid in pids:
            try:
                subprocess.run(["kill", "-0", str(pid)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                alive.append(pid)
            except Exception:
                pass
        if not alive:
            return
        time.sleep(0.5)

    for pid in pids:
        subprocess.run(["kill", "-KILL", str(pid)], check=False)


def _run_retrain(dry_run: bool, continue_on_error: bool) -> int:
    cmd = [str(VENV_PY), str(WEEKLY_RETRAIN)]
    if continue_on_error:
        cmd.append("--continue-on-error")

    print("$ " + " ".join(cmd))
    if dry_run:
        return 0

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return int(proc.returncode)


def _default_symbol_env(base_env: dict[str, str]) -> dict[str, str]:
    env = dict(base_env)
    env.setdefault("MARKET_DATA_ONLY", "1")
    env.setdefault("ALLOW_ORDER_EXECUTION", "0")
    env.setdefault("SHADOW_SYMBOLS_CORE", "SPY,QQQ,AAPL,MSFT,NVDA,DIA,IWM,MDY")
    env.setdefault("SHADOW_SYMBOLS_VOLATILE", "SOXL,SOXS,MSTR,SMCI,COIN,TSLA,UVXY,VIXY")
    env.setdefault(
        "SHADOW_SYMBOLS_DEFENSIVE",
        "TLT,GLD,XLV,XLU,XLP,HYG,LQD,UUP,XLE,XLF,XLI,XLK,XLY,IEF,SHY,TIP,TLH,JNK",
    )
    env.setdefault("SHADOW_SYMBOLS_COMMOD_FX_INTL", "DBC,UNG,CORN,SLV,USO,FXE,FXY,EFA,EEM,EWJ,FXI")
    return env


def _restart_parallel(restart_cmd: str | None, dry_run: bool) -> None:
    env = _default_symbol_env(os.environ)

    # Always restart with explicit symbols so relaunch never depends on missing shell exports.
    cmd = [
        str(VENV_PY),
        str(PARALLEL_SHADOWS),
        "--simulate",
        "--symbols-core",
        env["SHADOW_SYMBOLS_CORE"],
        "--symbols-volatile",
        env["SHADOW_SYMBOLS_VOLATILE"],
        "--symbols-defensive",
        f"{env['SHADOW_SYMBOLS_DEFENSIVE']},{env['SHADOW_SYMBOLS_COMMOD_FX_INTL']}",
    ]

    print("Restarting shadows with: " + " ".join(cmd))
    if dry_run:
        return

    subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pause running shadows, run manual retrain, then restart shadows."
    )
    parser.add_argument("--no-restart", action="store_true", help="Do not restart shadows after retrain.")
    parser.add_argument("--timeout-seconds", type=int, default=20, help="Graceful shutdown wait before SIGKILL.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-continue-on-error", dest="continue_on_error", action="store_false")
    parser.set_defaults(continue_on_error=True)
    args = parser.parse_args()

    pids, _restart_cmd = _find_shadow_targets()
    if pids:
        print(f"Stopping shadow processes: {pids}")
        if not args.dry_run:
            _terminate_pids(pids, max(args.timeout_seconds, 1))
    else:
        print("No running shadow processes found.")

    rc = _run_retrain(dry_run=args.dry_run, continue_on_error=args.continue_on_error)

    if not args.no_restart:
        _restart_parallel(_restart_cmd, dry_run=args.dry_run)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
