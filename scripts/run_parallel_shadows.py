import argparse
import fcntl
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HALT_FLAG_PATH = PROJECT_ROOT / "governance" / "health" / "GLOBAL_TRADING_HALT.flag"
VENV_PY = PROJECT_ROOT / ".venv312" / "bin" / "python"
SHADOW_LOOP = PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py"
TOKEN_PATH = PROJECT_ROOT / "token.json"
RESOURCE_GUARD_SCRIPT = PROJECT_ROOT / "scripts" / "resource_guard.py"


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _global_trading_halt_enabled() -> bool:
    return _env_flag("GLOBAL_TRADING_HALT", "0") or HALT_FLAG_PATH.exists()


def _route_storage_or_fail() -> bool:
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from core.storage_router import describe_storage_routing, route_runtime_storage

        routing = route_runtime_storage(PROJECT_ROOT)
        print(describe_storage_routing(routing))
        return True
    except Exception as exc:
        print(f"[StorageRoute] startup blocked err={exc}")
        return False


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
        print(f"[ParallelLock] busy lock_path={lock_path} owner={owner or 'unknown'}")
        return None

    fh.seek(0)
    fh.truncate(0)
    fh.write(f"pid={os.getpid()} started={time.time():.0f} cmd={' '.join(sys.argv)}")
    fh.flush()
    print(f"[ParallelLock] acquired lock_path={lock_path} pid={os.getpid()}")
    return fh


def _stream(name: str, pipe) -> None:
    for line in iter(pipe.readline, ""):
        sys.stdout.write(f"[{name}] {line}")
    pipe.close()


def _spawn_profile(
    name: str,
    threshold_shift: float,
    broker: str,
    auto_retrain: bool,
    simulate: bool,
    symbols: str | None,
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
    env["SHADOW_PROFILE"] = name
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
    if auto_retrain:
        cmd.append("--auto-retrain")
    if simulate:
        cmd.append("--simulate")

    if symbols:
        cmd.extend(["--symbols", symbols])
    else:
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


def _resource_guard_ok() -> bool:
    if os.getenv("ENABLE_RESOURCE_GUARD", "1").strip() != "1":
        return True
    if not RESOURCE_GUARD_SCRIPT.exists():
        return True
    proc = subprocess.run([str(VENV_PY), str(RESOURCE_GUARD_SCRIPT)], capture_output=True, text=True, check=False)
    out = (proc.stdout or "").strip()
    if out:
        print(f"[ResourceGuard] {out}")
    if proc.returncode != 0:
        print("[ResourceGuard] startup blocked due to system pressure.")
        return False
    return True


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
    parser = argparse.ArgumentParser(description="Run conservative and aggressive shadow profiles in parallel.")
    parser.add_argument("--simulate", action="store_true", help="Run both profiles without Schwab API auth.")
    parser.add_argument("--conservative-auto-retrain", action="store_true", default=False)
    parser.add_argument("--no-conservative-auto-retrain", dest="conservative_auto_retrain", action="store_false")
    parser.add_argument("--aggressive-auto-retrain", action="store_true", default=False)
    parser.add_argument("--conservative-threshold-shift", type=float, default=0.00)
    parser.add_argument("--aggressive-threshold-shift", type=float, default=-0.03)
    parser.add_argument("--symbols", default=None, help="Override full symbol list for both profiles.")
    parser.add_argument("--symbols-core", default=None)
    parser.add_argument("--symbols-volatile", default=None)
    parser.add_argument("--symbols-defensive", default=None)
    parser.add_argument("--context-symbols", default=None)
    parser.add_argument("--broker", default=os.getenv("DATA_BROKER", "schwab"), choices=["schwab", "coinbase"])
    parser.add_argument("--interval-seconds", type=int, default=int(os.getenv("SHADOW_LOOP_INTERVAL", "15")))
    parser.add_argument("--max-iterations", type=int, default=int(os.getenv("SHADOW_LOOP_MAX_ITERS", "0")))
    parser.add_argument(
        "--auth-bootstrap-timeout-seconds",
        type=int,
        default=int(os.getenv("PARALLEL_SHADOW_AUTH_BOOTSTRAP_TIMEOUT", "600")),
        help="If token.json is missing, wait this long for first profile to complete OAuth before starting second.",
    )
    args = parser.parse_args()

    if not _route_storage_or_fail():
        return 5

    if _global_trading_halt_enabled():
        print("GLOBAL_TRADING_HALT=1 set; refusing to start parallel shadows.")
        return 3

    if not _resource_guard_ok():
        return 4

    lock_path = Path(os.getenv("PARALLEL_SHADOW_LOCK_PATH", str(PROJECT_ROOT / "governance" / "parallel_shadow.lock")))
    lock_handle = _acquire_singleton_lock(lock_path)
    if lock_handle is None:
        return 1

    if not VENV_PY.exists():
        print(f"ERROR: missing venv python: {VENV_PY}")
        return 2
    if not SHADOW_LOOP.exists():
        print(f"ERROR: missing shadow loop script: {SHADOW_LOOP}")
        return 2

    conservative = _spawn_profile(
        "conservative",
        threshold_shift=args.conservative_threshold_shift,
        broker=args.broker,
        auto_retrain=args.conservative_auto_retrain,
        simulate=args.simulate,
        symbols=args.symbols,
        symbols_core=args.symbols_core,
        symbols_volatile=args.symbols_volatile,
        symbols_defensive=args.symbols_defensive,
        context_symbols=args.context_symbols,
        interval_seconds=args.interval_seconds,
        max_iterations=args.max_iterations,
    )
    print(f"Started conservative pid={conservative.pid}")

    t1 = threading.Thread(target=_stream, args=("conservative", conservative.stdout), daemon=True)
    t1.start()

    if not args.simulate and not (TOKEN_PATH.exists() and TOKEN_PATH.stat().st_size > 0):
        print("token.json missing: waiting for conservative OAuth bootstrap before starting aggressive...")
        ok = _wait_for_token_or_exit(conservative, args.auth_bootstrap_timeout_seconds)
        if not ok:
            _stop_processes([conservative])
            print("Stopped: conservative exited or OAuth bootstrap timeout reached before token.json was created.")
            return 1
        time.sleep(2.0)

    aggressive = _spawn_profile(
        "aggressive",
        threshold_shift=args.aggressive_threshold_shift,
        broker=args.broker,
        auto_retrain=args.aggressive_auto_retrain,
        simulate=args.simulate,
        symbols=args.symbols,
        symbols_core=args.symbols_core,
        symbols_volatile=args.symbols_volatile,
        symbols_defensive=args.symbols_defensive,
        context_symbols=args.context_symbols,
        interval_seconds=args.interval_seconds,
        max_iterations=args.max_iterations,
    )
    print(f"Started aggressive pid={aggressive.pid}")
    domain = _domain_for_broker(args.broker)
    print(
        "Logs: decision_explanations/shadow_conservative_{domain} and "
        "decision_explanations/shadow_aggressive_{domain}".format(domain=domain)
    )

    t2 = threading.Thread(target=_stream, args=("aggressive", aggressive.stdout), daemon=True)
    t2.start()

    procs = [conservative, aggressive]
    try:
        while True:
            if _global_trading_halt_enabled():
                print("GLOBAL_TRADING_HALT=1 detected; stopping both profiles.")
                _stop_processes(procs)
                return 0

            exits = [p.poll() for p in procs]
            if any(code is not None for code in exits):
                _stop_processes(procs)
                print(f"Stopped because one profile exited: {exits}")
                return 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping both profiles...")
        _stop_processes(procs)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
