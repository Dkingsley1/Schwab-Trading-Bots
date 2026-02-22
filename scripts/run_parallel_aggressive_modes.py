import argparse
import fcntl
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT_ROOT / ".venv312" / "bin" / "python"
SHADOW_LOOP = PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py"
TOKEN_PATH = PROJECT_ROOT / "token.json"


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _global_trading_halt_enabled() -> bool:
    return _env_flag("GLOBAL_TRADING_HALT", "0")


def _domain_for_broker(broker: str) -> str:
    return "crypto" if (broker or "").strip().lower() == "coinbase" else "equities"


def _acquire_singleton_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            fh.seek(0)
            owner = fh.read().strip()
        except Exception:
            owner = "unknown"
        fh.close()
        print(f"[AggressiveModesLock] busy lock_path={lock_path} owner={owner or 'unknown'}")
        return None

    fh.seek(0)
    fh.truncate(0)
    fh.write(f"pid={os.getpid()} started={time.time():.0f} cmd={' '.join(sys.argv)}")
    fh.flush()
    print(f"[AggressiveModesLock] acquired lock_path={lock_path} pid={os.getpid()}")
    return fh


def _stream(name: str, pipe) -> None:
    for line in iter(pipe.readline, ""):
        sys.stdout.write(f"[{name}] {line}")
    pipe.close()


def _spawn_profile(
    *,
    profile_name: str,
    threshold_shift: float,
    broker: str,
    simulate: bool,
    symbols_core: str | None,
    symbols_volatile: str | None,
    symbols_defensive: str | None,
    context_symbols: str | None,
    interval_seconds: int,
    max_iterations: int,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["MARKET_DATA_ONLY"] = "1"
    env["ALLOW_ORDER_EXECUTION"] = "0"
    env["SHADOW_PROFILE"] = profile_name
    env["SHADOW_THRESHOLD_SHIFT"] = f"{threshold_shift:.3f}"
    env["SHADOW_DOMAIN"] = _domain_for_broker(broker)

    cmd = [
        str(VENV_PY),
        str(SHADOW_LOOP),
        "--broker",
        broker,
        "--interval-seconds",
        str(interval_seconds),
        "--max-iterations",
        str(max_iterations),
    ]
    if simulate:
        cmd.append("--simulate")
    if symbols_core:
        cmd.extend(["--symbols-core", symbols_core])
    if symbols_volatile:
        cmd.extend(["--symbols-volatile", symbols_volatile])
    if symbols_defensive:
        cmd.extend(["--symbols-defensive", symbols_defensive])
    if context_symbols:
        cmd.extend(["--context-symbols", context_symbols])

    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _stop_processes(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        if p.poll() is None:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()


def _wait_for_token_or_exit(proc: subprocess.Popen, timeout_seconds: int) -> bool:
    start = time.time()
    while True:
        if proc.poll() is not None:
            return False
        if TOKEN_PATH.exists() and TOKEN_PATH.stat().st_size > 0:
            return True
        if (time.time() - start) > timeout_seconds:
            return False
        time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run aggressive intraday and aggressive swing shadow profiles in parallel.")
    parser.add_argument("--simulate", action="store_true", help="Run without Schwab API auth.")
    parser.add_argument("--broker", default=os.getenv("DATA_BROKER", "schwab"), choices=["schwab", "coinbase"])
    parser.add_argument("--max-iterations", type=int, default=int(os.getenv("SHADOW_LOOP_MAX_ITERS", "0")))
    parser.add_argument(
        "--auth-bootstrap-timeout-seconds",
        type=int,
        default=int(os.getenv("PARALLEL_SHADOW_AUTH_BOOTSTRAP_TIMEOUT", "600")),
    )

    parser.add_argument("--intraday-threshold-shift", type=float, default=-0.08)
    parser.add_argument("--swing-threshold-shift", type=float, default=-0.04)
    parser.add_argument("--intraday-interval-seconds", type=int, default=8)
    parser.add_argument("--swing-interval-seconds", type=int, default=75)

    parser.add_argument("--intraday-symbols-core", default=os.getenv("SHADOW_SYMBOLS_CORE", "SPY,QQQ,AAPL,MSFT,NVDA,DIA,IWM,MDY"))
    parser.add_argument("--intraday-symbols-volatile", default=os.getenv("SHADOW_SYMBOLS_VOLATILE", "SOXL,SOXS,TSLA,COIN,MSTR,SMCI,UVXY,VIXY"))
    parser.add_argument("--intraday-symbols-defensive", default=os.getenv("SHADOW_SYMBOLS_DEFENSIVE", "TLT,GLD,IEF,SHY"))
    parser.add_argument("--intraday-context-symbols", default=os.getenv("WATCH_CONTEXT_SYMBOLS", "$VIX.X,UUP"))

    parser.add_argument("--swing-symbols-core", default=os.getenv("SHADOW_SYMBOLS_CORE", "SPY,QQQ,AAPL,MSFT,NVDA,DIA,IWM,MDY"))
    parser.add_argument("--swing-symbols-volatile", default=os.getenv("SHADOW_SYMBOLS_VOLATILE", "SOXL,SOXS,TSLA,COIN,MSTR,SMCI,UVXY,VIXY"))
    parser.add_argument(
        "--swing-symbols-defensive",
        default=(os.getenv("SHADOW_SYMBOLS_DEFENSIVE", "TLT,GLD,XLV,XLU,XLP,HYG,LQD,UUP,XLE,XLF,XLI,XLK,XLY,IEF,SHY,TIP,TLH,JNK")
                 + "," + os.getenv("SHADOW_SYMBOLS_COMMOD_FX_INTL", "DBC,UNG,CORN,SLV,USO,FXE,FXY,EFA,EEM,EWJ,FXI")).strip(","),
    )
    parser.add_argument("--swing-context-symbols", default=os.getenv("WATCH_CONTEXT_SYMBOLS", "$VIX.X,UUP"))
    args = parser.parse_args()

    if _global_trading_halt_enabled():
        print("GLOBAL_TRADING_HALT=1 set; refusing to start aggressive modes.")
        return 3

    lock_path = Path(os.getenv("PARALLEL_AGGRESSIVE_LOCK_PATH", str(PROJECT_ROOT / "governance" / "parallel_aggressive_modes.lock")))
    lock_handle = _acquire_singleton_lock(lock_path)
    if lock_handle is None:
        return 1

    if not VENV_PY.exists():
        print(f"ERROR: missing venv python: {VENV_PY}")
        return 2
    if not SHADOW_LOOP.exists():
        print(f"ERROR: missing shadow loop script: {SHADOW_LOOP}")
        return 2

    intraday = _spawn_profile(
        profile_name="intraday_aggressive",
        threshold_shift=args.intraday_threshold_shift,
        broker=args.broker,
        simulate=args.simulate,
        symbols_core=args.intraday_symbols_core,
        symbols_volatile=args.intraday_symbols_volatile,
        symbols_defensive=args.intraday_symbols_defensive,
        context_symbols=args.intraday_context_symbols,
        interval_seconds=max(args.intraday_interval_seconds, 5),
        max_iterations=args.max_iterations,
    )
    print(f"Started intraday_aggressive pid={intraday.pid}")
    t1 = threading.Thread(target=_stream, args=("intraday_aggressive", intraday.stdout), daemon=True)
    t1.start()

    if not args.simulate and not (TOKEN_PATH.exists() and TOKEN_PATH.stat().st_size > 0):
        print("token.json missing: waiting for intraday OAuth bootstrap before starting swing...")
        ok = _wait_for_token_or_exit(intraday, args.auth_bootstrap_timeout_seconds)
        if not ok:
            _stop_processes([intraday])
            print("Stopped: intraday exited or OAuth bootstrap timeout before token.json.")
            return 1
        time.sleep(2.0)

    swing = _spawn_profile(
        profile_name="swing_aggressive",
        threshold_shift=args.swing_threshold_shift,
        broker=args.broker,
        simulate=args.simulate,
        symbols_core=args.swing_symbols_core,
        symbols_volatile=args.swing_symbols_volatile,
        symbols_defensive=args.swing_symbols_defensive,
        context_symbols=args.swing_context_symbols,
        interval_seconds=max(args.swing_interval_seconds, 5),
        max_iterations=args.max_iterations,
    )
    print(f"Started swing_aggressive pid={swing.pid}")

    domain = _domain_for_broker(args.broker)
    print(
        "Logs: decision_explanations/shadow_intraday_aggressive_{domain} and "
        "decision_explanations/shadow_swing_aggressive_{domain}".format(domain=domain)
    )

    t2 = threading.Thread(target=_stream, args=("swing_aggressive", swing.stdout), daemon=True)
    t2.start()

    procs = [intraday, swing]
    try:
        while True:
            if _global_trading_halt_enabled():
                print("GLOBAL_TRADING_HALT=1 detected; stopping aggressive modes.")
                _stop_processes(procs)
                return 0

            exits = [p.poll() for p in procs]
            if any(code is not None for code in exits):
                _stop_processes(procs)
                print(f"Stopped because one aggressive mode exited: {exits}")
                return 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping aggressive modes...")
        _stop_processes(procs)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
