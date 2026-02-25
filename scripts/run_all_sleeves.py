import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT_ROOT / ".venv312" / "bin" / "python"
PARALLEL_SHADOWS = PROJECT_ROOT / "scripts" / "run_parallel_shadows.py"
DIVIDEND_SHADOW = PROJECT_ROOT / "scripts" / "run_dividend_shadow.py"
BOND_SHADOW = PROJECT_ROOT / "scripts" / "run_bond_shadow.py"
AGGRESSIVE_MODES = PROJECT_ROOT / "scripts" / "run_parallel_aggressive_modes.py"
HALT_FLAG_PATH = PROJECT_ROOT / "governance" / "health" / "GLOBAL_TRADING_HALT.flag"
PREFLIGHT_SCRIPT = PROJECT_ROOT / "scripts" / "shadow_preflight.py"
DEBUG_SNAPSHOT_SCRIPT = PROJECT_ROOT / "scripts" / "collect_debug_snapshot.sh"
CAPTURE_CONFIG_SCRIPT = PROJECT_ROOT / "scripts" / "capture_run_config.py"


@dataclass
class JobSpec:
    name: str
    cmd: list[str]
    env: dict[str, str]
    breaker_group: str


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _global_trading_halt_enabled() -> bool:
    return _env_flag("GLOBAL_TRADING_HALT", "0") or HALT_FLAG_PATH.exists()




def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)

def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


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
        print(f"[AllSleevesLock] busy lock_path={lock_path} owner={owner or 'unknown'}")
        return None

    fh.seek(0)
    fh.truncate(0)
    fh.write(f"pid={os.getpid()} started={time.time():.0f} cmd={' '.join(sys.argv)}")
    fh.flush()
    print(f"[AllSleevesLock] acquired lock_path={lock_path} pid={os.getpid()}")
    return fh


def _stream(name: str, pipe) -> None:
    for line in iter(pipe.readline, ""):
        sys.stdout.write(f"[{name}] {line}")
    pipe.close()


def _spawn(spec: JobSpec) -> subprocess.Popen:
    proc = subprocess.Popen(
        spec.cmd,
        cwd=str(PROJECT_ROOT),
        env=spec.env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    print(f"Started {spec.name} pid={proc.pid}")
    t = threading.Thread(target=_stream, args=(spec.name, proc.stdout), daemon=True)
    t.start()
    return proc


def _stop_processes(procs: dict[str, subprocess.Popen]) -> None:
    for proc in procs.values():
        if proc.poll() is None:
            proc.terminate()
    for proc in procs.values():
        if proc.poll() is None:
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()


def _within_restart_budget(restarts: list[float], max_restarts_per_hour: int) -> bool:
    now = time.time()
    one_hour_ago = now - 3600
    while restarts and restarts[0] < one_hour_ago:
        restarts.pop(0)
    return len(restarts) < max_restarts_per_hour


def _read_one_numbers(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _breaker_reasons(metrics: dict, args) -> tuple[list[str], str]:
    reasons: list[str] = []
    dq = _safe_float(metrics.get("data_quality_score"), 0.0)
    blocked = _safe_float(metrics.get("combined_blocked_rate"), 0.0)

    if dq < args.breaker_min_data_quality:
        reasons.append(f"data_quality_low:{dq:.2f}")
    if blocked > args.breaker_max_blocked_rate:
        reasons.append(f"blocked_rate_high:{blocked:.4f}")

    broker_domain = "stocks" if args.broker == "schwab" else "crypto"
    pnl_key = "stocks_pnl_proxy" if broker_domain == "stocks" else "crypto_pnl_proxy"
    pnl_val = _safe_float(metrics.get(pnl_key), 0.0)
    if pnl_val < args.breaker_min_pnl_proxy:
        reasons.append(f"{pnl_key}_low:{pnl_val:.6f}")

    return reasons, broker_domain


def _emit_incident_snapshot(reason: str, detail: str = "") -> None:
    if not DEBUG_SNAPSHOT_SCRIPT.exists():
        return
    try:
        proc = subprocess.run([str(DEBUG_SNAPSHOT_SCRIPT)], cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
        msg = (proc.stdout or "").strip().splitlines()[-1:] or [""]
        print(f"[IncidentSnapshot] reason={reason} detail={detail} rc={proc.returncode} note={msg[0] if msg else ''}")
    except Exception as exc:
        print(f"[IncidentSnapshot] failed reason={reason} err={exc}")


def _capture_full_run_config(args: argparse.Namespace) -> None:
    try:
        if CAPTURE_CONFIG_SCRIPT.exists() and VENV_PY.exists():
            subprocess.run([str(VENV_PY), str(CAPTURE_CONFIG_SCRIPT)], cwd=str(PROJECT_ROOT), check=False)

        keys = [
            "MARKET_DATA_ONLY",
            "ALLOW_ORDER_EXECUTION",
            "DATA_BROKER",
            "SHADOW_SYMBOLS_CORE",
            "SHADOW_SYMBOLS_VOLATILE",
            "SHADOW_SYMBOLS_DEFENSIVE",
            "SHADOW_SYMBOLS_COMMOD_FX_INTL",
            "ASYNC_PIPELINE_WORKERS",
            "SHADOW_LOOP_INTERVAL",
            "ADAPTIVE_INTERVAL_MAX_SECONDS",
            "CANARY_MAX_WEIGHT",
            "GLOBAL_TRADING_HALT",
        ]
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "launcher": "run_all_sleeves.py",
            "argv": sys.argv,
            "args": vars(args),
            "env": {k: os.getenv(k, "") for k in keys},
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        payload["config_hash"] = hashlib.sha256(encoded).hexdigest()[:16]

        out_dir = PROJECT_ROOT / "governance" / "session_configs"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"all_sleeves_config_{stamp}.json"
        out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        (out_dir / "all_sleeves_latest.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"[ConfigFreeze] hash={payload['config_hash']} file={out}")
    except Exception as exc:
        print(f"[ConfigFreeze] warning failed: {exc}")


def _run_preflight(args: argparse.Namespace) -> bool:
    if not PREFLIGHT_SCRIPT.exists():
        return True
    cmd = [
        str(VENV_PY),
        str(PREFLIGHT_SCRIPT),
        "--broker",
        args.broker,
        "--symbols-core",
        args.symbols_core,
        "--symbols-volatile",
        args.symbols_volatile,
        "--symbols-defensive",
        args.symbols_defensive,
    ]
    if args.simulate:
        cmd.append("--simulate")
    if not getattr(args, "strict_preflight_duplicates", False):
        cmd.append("--allow-running")

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    out = (proc.stdout or "").strip()
    if out:
        print(out)
    if proc.returncode != 0:
        _emit_incident_snapshot("preflight_failed", f"rc={proc.returncode}")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Schwab sleeves together: baseline + dividend + bond (+ optional aggressive modes).")
    parser.add_argument("--simulate", action="store_true", help="Run all sleeves in simulation mode.")
    parser.add_argument("--with-aggressive-modes", action="store_true", help="Also run intraday+swing aggressive modes.")
    parser.add_argument("--parallel-interval-seconds", type=int, default=int(os.getenv("SHADOW_LOOP_INTERVAL", "15")))
    parser.add_argument("--dividend-interval-seconds", type=int, default=int(os.getenv("DIVIDEND_SHADOW_INTERVAL", "60")))
    parser.add_argument("--bond-interval-seconds", type=int, default=int(os.getenv("BOND_SHADOW_INTERVAL", "90")))
    parser.add_argument("--broker", default=os.getenv("DATA_BROKER", "schwab"), choices=["schwab", "coinbase"])
    parser.add_argument("--max-iterations", type=int, default=int(os.getenv("SHADOW_LOOP_MAX_ITERS", "0")))
    parser.add_argument("--symbols-core", default=os.getenv("SHADOW_SYMBOLS_CORE", ""))
    parser.add_argument("--symbols-volatile", default=os.getenv("SHADOW_SYMBOLS_VOLATILE", ""))
    parser.add_argument("--symbols-defensive", default=(os.getenv("SHADOW_SYMBOLS_DEFENSIVE", "") + "," + os.getenv("SHADOW_SYMBOLS_COMMOD_FX_INTL", "")).strip(","))
    parser.add_argument("--dividend-symbols", default=os.getenv("DIVIDEND_SYMBOLS", ""))
    parser.add_argument("--bond-symbols", default=os.getenv("BOND_SYMBOLS", ""))
    parser.add_argument("--restart-delay-seconds", type=int, default=int(os.getenv("ALL_SLEEVES_RESTART_DELAY", "3")))
    parser.add_argument("--max-restarts-per-hour", type=int, default=int(os.getenv("ALL_SLEEVES_MAX_RESTARTS_PER_HOUR", "40")))
    parser.add_argument("--no-restart-on-exit", dest="restart_on_exit", action="store_false", default=True)
    parser.add_argument(
        "--strict-preflight-duplicates",
        action="store_true",
        default=os.getenv("RUN_ALL_SLEEVES_STRICT_PREFLIGHT_DUPLICATES", "0").strip() == "1",
        help="Fail preflight when a parallel launcher is already running.",
    )

    parser.add_argument("--nice-baseline", type=int, default=int(os.getenv("SLEEVE_NICE_BASELINE", "6")))
    parser.add_argument("--nice-dividend", type=int, default=int(os.getenv("SLEEVE_NICE_DIVIDEND", "10")))
    parser.add_argument("--nice-bond", type=int, default=int(os.getenv("SLEEVE_NICE_BOND", "10")))
    parser.add_argument("--nice-aggressive", type=int, default=int(os.getenv("SLEEVE_NICE_AGGRESSIVE", "5")))
    parser.add_argument("--workers-baseline", type=int, default=int(os.getenv("SLEEVE_WORKERS_BASELINE", os.getenv("ASYNC_PIPELINE_WORKERS", "4"))))
    parser.add_argument("--workers-dividend", type=int, default=int(os.getenv("SLEEVE_WORKERS_DIVIDEND", "2")))
    parser.add_argument("--workers-bond", type=int, default=int(os.getenv("SLEEVE_WORKERS_BOND", "2")))
    parser.add_argument("--workers-aggressive", type=int, default=int(os.getenv("SLEEVE_WORKERS_AGGRESSIVE", "3")))

    parser.add_argument("--disable-circuit-breakers", action="store_true")
    parser.add_argument("--breaker-one-numbers-path", default=str(PROJECT_ROOT / "exports" / "one_numbers" / "one_numbers_summary.json"))
    parser.add_argument("--breaker-check-interval-seconds", type=int, default=int(os.getenv("ALL_SLEEVES_BREAKER_CHECK_SECONDS", "60")))
    parser.add_argument("--breaker-consecutive-breaches", type=int, default=int(os.getenv("ALL_SLEEVES_BREAKER_STREAK", "2")))
    parser.add_argument("--breaker-cooldown-seconds", type=int, default=int(os.getenv("ALL_SLEEVES_BREAKER_COOLDOWN", "300")))
    parser.add_argument("--breaker-min-data-quality", type=float, default=float(os.getenv("ALL_SLEEVES_BREAKER_MIN_DQ", "75")))
    parser.add_argument("--breaker-max-blocked-rate", type=float, default=float(os.getenv("ALL_SLEEVES_BREAKER_MAX_BLOCKED", "0.35")))
    parser.add_argument("--breaker-min-pnl-proxy", type=float, default=float(os.getenv("ALL_SLEEVES_BREAKER_MIN_PNL", "-0.020")))
    parser.add_argument(
        "--hard-min-free-gb",
        type=float,
        default=float(os.getenv("ALL_SLEEVES_HARD_MIN_FREE_GB", "15")),
        help="Hard startup block if free disk is below this GB threshold.",
    )

    args = parser.parse_args()

    if _global_trading_halt_enabled():
        print("GLOBAL_TRADING_HALT=1 set; refusing to start all sleeves.")
        _emit_incident_snapshot("global_halt_refusal", "startup")
        return 3

    if not VENV_PY.exists():
        print(f"ERROR: missing venv python: {VENV_PY}")
        return 2

    free_gb = _disk_free_gb(PROJECT_ROOT)
    if free_gb < max(float(args.hard_min_free_gb), 0.1):
        print(
            f"[HardDiskGate] blocked free_gb={free_gb:.2f} "
            f"min_required_gb={float(args.hard_min_free_gb):.2f}"
        )
        _emit_incident_snapshot("hard_disk_gate_blocked", f"free_gb={free_gb:.2f}")
        return 5

    lock_path = Path(os.getenv("ALL_SLEEVES_LOCK_PATH", str(PROJECT_ROOT / "governance" / "all_sleeves.lock")))
    lock_handle = _acquire_singleton_lock(lock_path)
    if lock_handle is None:
        _emit_incident_snapshot("all_sleeves_lock_busy", str(lock_path))
        return 1

    if not _run_preflight(args):
        print("[Preflight] startup blocked.")
        return 4

    _capture_full_run_config(args)

    base_env = os.environ.copy()
    base_env["MARKET_DATA_ONLY"] = "1"
    base_env["ALLOW_ORDER_EXECUTION"] = "0"

    specs: dict[str, JobSpec] = {}

    parallel_cmd = [
        "nice", "-n", str(args.nice_baseline),
        str(VENV_PY), str(PARALLEL_SHADOWS),
        "--broker", args.broker,
        "--interval-seconds", str(max(args.parallel_interval_seconds, 5)),
        "--max-iterations", str(args.max_iterations),
    ]
    if args.simulate:
        parallel_cmd.append("--simulate")
    if args.symbols_core:
        parallel_cmd.extend(["--symbols-core", args.symbols_core])
    if args.symbols_volatile:
        parallel_cmd.extend(["--symbols-volatile", args.symbols_volatile])
    if args.symbols_defensive:
        parallel_cmd.extend(["--symbols-defensive", args.symbols_defensive])
    env = dict(base_env)
    env["ASYNC_PIPELINE_WORKERS"] = str(max(args.workers_baseline, 1))
    specs["baseline_parallel"] = JobSpec("baseline_parallel", parallel_cmd, env, breaker_group="core")

    dividend_cmd = [
        "nice", "-n", str(args.nice_dividend),
        str(VENV_PY), str(DIVIDEND_SHADOW),
        "--broker", args.broker,
        "--interval-seconds", str(max(args.dividend_interval_seconds, 15)),
        "--max-iterations", str(args.max_iterations),
    ]
    if args.simulate:
        dividend_cmd.append("--simulate")
    if args.dividend_symbols:
        dividend_cmd.extend(["--symbols", args.dividend_symbols])
    env = dict(base_env)
    env["ASYNC_PIPELINE_WORKERS"] = str(max(args.workers_dividend, 1))
    specs["dividend"] = JobSpec("dividend", dividend_cmd, env, breaker_group="core")

    bond_cmd = [
        "nice", "-n", str(args.nice_bond),
        str(VENV_PY), str(BOND_SHADOW),
        "--broker", args.broker,
        "--interval-seconds", str(max(args.bond_interval_seconds, 15)),
        "--max-iterations", str(args.max_iterations),
    ]
    if args.simulate:
        bond_cmd.append("--simulate")
    if args.bond_symbols:
        bond_cmd.extend(["--symbols", args.bond_symbols])
    env = dict(base_env)
    env["ASYNC_PIPELINE_WORKERS"] = str(max(args.workers_bond, 1))
    specs["bond"] = JobSpec("bond", bond_cmd, env, breaker_group="core")

    if args.with_aggressive_modes:
        aggressive_cmd = [
            "nice", "-n", str(args.nice_aggressive),
            str(VENV_PY), str(AGGRESSIVE_MODES),
            "--broker", args.broker,
            "--max-iterations", str(args.max_iterations),
        ]
        if args.simulate:
            aggressive_cmd.append("--simulate")
        env = dict(base_env)
        env["ASYNC_PIPELINE_WORKERS"] = str(max(args.workers_aggressive, 1))
        specs["aggressive_modes"] = JobSpec("aggressive_modes", aggressive_cmd, env, breaker_group="core")

    procs: dict[str, subprocess.Popen] = {}
    restart_history: dict[str, list[float]] = {name: [] for name in specs}
    breaker_streaks: dict[str, int] = {"core": 0}
    group_disabled_until: dict[str, float] = {"core": 0.0}
    last_breaker_check_ts = 0.0
    breaker_path = Path(args.breaker_one_numbers_path)

    try:
        for name, spec in specs.items():
            procs[name] = _spawn(spec)
            time.sleep(0.8)

        print("All sleeves live:", ", ".join(specs.keys()))
        while True:
            if _global_trading_halt_enabled():
                print("GLOBAL_TRADING_HALT=1 detected; stopping all sleeves.")
                _stop_processes(procs)
                _emit_incident_snapshot("global_halt_detected", "runtime")
                return 0

            now = time.time()
            if not args.disable_circuit_breakers and (now - last_breaker_check_ts) >= max(args.breaker_check_interval_seconds, 15):
                last_breaker_check_ts = now
                metrics = _read_one_numbers(breaker_path)
                reasons, _domain = _breaker_reasons(metrics, args)
                if reasons:
                    breaker_streaks["core"] = breaker_streaks.get("core", 0) + 1
                    print(
                        f"[CircuitBreaker] breach_streak={breaker_streaks['core']}/{max(args.breaker_consecutive_breaches,1)} "
                        f"reasons={'|'.join(reasons)}"
                    )
                else:
                    breaker_streaks["core"] = 0

                if breaker_streaks["core"] >= max(args.breaker_consecutive_breaches, 1):
                    group_disabled_until["core"] = now + max(args.breaker_cooldown_seconds, 30)
                    breaker_streaks["core"] = 0
                    print(
                        f"[CircuitBreaker] TRIPPED group=core cooldown_s={max(args.breaker_cooldown_seconds,30)} "
                        f"reasons={'|'.join(reasons)}"
                    )
                    _emit_incident_snapshot("circuit_breaker_tripped", "|".join(reasons))
                    for name, proc in list(procs.items()):
                        if specs[name].breaker_group != "core":
                            continue
                        if proc.poll() is None:
                            proc.terminate()

            for name, proc in list(procs.items()):
                code = proc.poll()
                if code is None:
                    continue

                print(f"[{name}] exited code={code}")
                if not args.restart_on_exit:
                    _stop_processes(procs)
                    _emit_incident_snapshot("sleeve_exit_no_restart", f"{name}:{code}")
                    print("Stopped because one sleeve exited and restart mode is disabled.")
                    return 1

                grp = specs[name].breaker_group
                cooldown_until = group_disabled_until.get(grp, 0.0)
                if cooldown_until > time.time():
                    remaining = int(max(cooldown_until - time.time(), 0))
                    print(f"[{name}] restart_paused circuit_breaker_cooldown_remaining_s={remaining}")
                    continue

                if not _within_restart_budget(restart_history[name], args.max_restarts_per_hour):
                    _stop_processes(procs)
                    _emit_incident_snapshot("restart_budget_exceeded", f"{name}:{args.max_restarts_per_hour}")
                    print(
                        f"Stopped: {name} exceeded restart budget "
                        f"({args.max_restarts_per_hour}/hour)."
                    )
                    return 1

                time.sleep(max(args.restart_delay_seconds, 1))
                restart_history[name].append(time.time())
                procs[name] = _spawn(specs[name])
                print(f"[{name}] restart_count_last_hour={len(restart_history[name])}")

            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping all sleeves...")
        _stop_processes(procs)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
