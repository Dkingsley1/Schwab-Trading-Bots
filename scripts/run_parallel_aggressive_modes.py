import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HALT_FLAG_PATH = PROJECT_ROOT / "governance" / "health" / "GLOBAL_TRADING_HALT.flag"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python

VENV_PY = resolve_runtime_python(PROJECT_ROOT)
SHADOW_LOOP = PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py"
TOKEN_PATH = PROJECT_ROOT / "token.json"
RESOURCE_GUARD_SCRIPT = PROJECT_ROOT / "scripts" / "resource_guard.py"


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    profile_name: str
    ingress_instance: str
    threshold_shift: float
    symbols_core: str
    symbols_volatile: str
    symbols_defensive: str
    context_symbols: str
    interval_seconds: int


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


def _safe_token(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _parse_symbols(raw: str | None) -> list[str]:
    seen: set[str] = set()
    symbols: list[str] = []
    for part in str(raw or "").split(","):
        sym = part.strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        symbols.append(sym)
    return symbols


def _join_symbols(symbols: list[str]) -> str:
    return ",".join(symbols)


def _ingress_summary_path(*, profile_name: str, broker: str, ingress_instance: str = "") -> Path:
    domain = _domain_for_broker(broker)
    base = PROJECT_ROOT / "governance" / "health" / f"data_ingress_latest_{profile_name}_{domain}_{broker}.json"
    token = _safe_token(ingress_instance)
    if not token:
        return base
    return base.with_name(f"{base.stem}_{token}{base.suffix}")


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _merge_counter_maps(rows: list[dict], key: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for row in rows:
        payload = row.get(key, {}) if isinstance(row.get(key, {}), dict) else {}
        for name, value in payload.items():
            merged[str(name)] = merged.get(str(name), 0) + int(value or 0)
    return merged


def _aggregate_ingress_payload(
    profile_name: str,
    broker: str,
    child_payloads: list[dict],
    *,
    expected_instances: list[str] | None = None,
) -> dict:
    valid = [row for row in child_payloads if isinstance(row, dict) and row]
    if not valid:
        return {}
    expected = sorted({str(item or "").strip() for item in (expected_instances or []) if str(item or "").strip()})
    newest = max(valid, key=lambda row: str(row.get("timestamp_utc") or ""))
    iter_counts = _merge_counter_maps(valid, "iter_counts")
    total_counts = _merge_counter_maps(valid, "total_counts")
    iter_total_requests = sum(int(row.get("iter_total_requests", 0) or 0) for row in valid)
    iter_error_count = sum(int(row.get("iter_error_count", 0) or 0) for row in valid)
    present_instances = sorted(
        {
            str(row.get("ingress_instance") or "")
            for row in valid
            if str(row.get("ingress_instance") or "").strip()
        }
    )
    missing_instances = [name for name in expected if name not in present_instances]
    if expected and len(expected) > 1:
        loop_state = "split_running" if len(present_instances) == len(expected) else "split_partial"
    else:
        loop_state = str(newest.get("loop_state") or "")
    rows_by_instance = {
        str(row.get("ingress_instance") or "").strip(): row
        for row in valid
        if str(row.get("ingress_instance") or "").strip()
    }
    split_profiles = []
    for ingress_instance in (expected or present_instances):
        row = rows_by_instance.get(ingress_instance, {})
        split_profiles.append(
            {
                "ingress_instance": ingress_instance,
                "loop_state": str(row.get("loop_state") or ("missing" if ingress_instance in missing_instances else "")),
                "iter_total_requests": int(row.get("iter_total_requests", 0) or 0),
                "iter_error_count": int(row.get("iter_error_count", 0) or 0),
                "symbols_total": int(row.get("symbols_total", 0) or 0),
            }
        )
    return {
        "timestamp_utc": str(newest.get("timestamp_utc") or ""),
        "run_id": str(newest.get("run_id") or ""),
        "iter_id": str(newest.get("iter_id") or ""),
        "iter": max(int(row.get("iter", 0) or 0) for row in valid),
        "broker": broker,
        "profile": profile_name,
        "domain": _domain_for_broker(broker),
        "loop_state": loop_state,
        "iter_counts": iter_counts,
        "total_counts": total_counts,
        "iter_total_requests": int(iter_total_requests),
        "iter_error_count": int(iter_error_count),
        "iter_error_rate": round((float(iter_error_count) / float(iter_total_requests)), 6) if iter_total_requests > 0 else 0.0,
        "symbols_total": sum(int(row.get("symbols_total", 0) or 0) for row in valid),
        "context_total": max(int(row.get("context_total", 0) or 0) for row in valid),
        "log_schema_version": max(int(row.get("log_schema_version", 1) or 1) for row in valid),
        "ingress_instances": present_instances,
        "ingress_segment_count": len(valid),
        "expected_ingress_instances": expected,
        "missing_ingress_instances": missing_instances,
        "split_profiles": split_profiles,
    }


def _bootstrap_ingress_state(worker: WorkerSpec, broker: str) -> None:
    if not worker.ingress_instance:
        return
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": "",
        "iter_id": "",
        "iter": 0,
        "broker": broker,
        "profile": worker.profile_name,
        "domain": _domain_for_broker(broker),
        "loop_state": "booting",
        "iter_counts": {},
        "total_counts": {},
        "iter_total_requests": 0,
        "iter_error_count": 0,
        "iter_error_rate": 0.0,
        "symbols_total": len(
            {
                *(_parse_symbols(worker.symbols_core)),
                *(_parse_symbols(worker.symbols_volatile)),
                *(_parse_symbols(worker.symbols_defensive)),
            }
        ),
        "context_total": len(_parse_symbols(worker.context_symbols)),
        "log_schema_version": max(int(os.getenv("LOG_SCHEMA_VERSION", "2")), 1),
        "ingress_instance": worker.ingress_instance,
    }
    _write_json(
        _ingress_summary_path(
            profile_name=worker.profile_name,
            broker=broker,
            ingress_instance=worker.ingress_instance,
        ),
        payload,
    )


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
    worker: WorkerSpec,
    broker: str,
    simulate: bool,
    max_iterations: int,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["MARKET_DATA_ONLY"] = "1"
    env["ALLOW_ORDER_EXECUTION"] = "0"
    env["SHADOW_PROFILE"] = worker.profile_name
    env["SHADOW_THRESHOLD_SHIFT"] = f"{worker.threshold_shift:.3f}"
    env["SHADOW_DOMAIN"] = _domain_for_broker(broker)
    if worker.ingress_instance:
        env["SHADOW_INGRESS_INSTANCE"] = worker.ingress_instance
    else:
        env.pop("SHADOW_INGRESS_INSTANCE", None)

    cmd = [
        str(VENV_PY),
        str(SHADOW_LOOP),
        "--broker",
        broker,
        "--interval-seconds",
        str(worker.interval_seconds),
        "--max-iterations",
        str(max_iterations),
    ]
    if simulate:
        cmd.append("--simulate")
    if worker.symbols_core:
        cmd.extend(["--symbols-core", worker.symbols_core])
    if worker.symbols_volatile:
        cmd.extend(["--symbols-volatile", worker.symbols_volatile])
    if worker.symbols_defensive:
        cmd.extend(["--symbols-defensive", worker.symbols_defensive])
    if worker.context_symbols:
        cmd.extend(["--context-symbols", worker.context_symbols])

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


def _build_worker_specs(args: argparse.Namespace) -> list[WorkerSpec]:
    workers = [
        WorkerSpec(
            name="intraday_aggressive",
            profile_name="intraday_aggressive",
            ingress_instance="",
            threshold_shift=args.intraday_threshold_shift,
            symbols_core=args.intraday_symbols_core,
            symbols_volatile=args.intraday_symbols_volatile,
            symbols_defensive=args.intraday_symbols_defensive,
            context_symbols=args.intraday_context_symbols,
            interval_seconds=max(args.intraday_interval_seconds, 5),
        ),
        WorkerSpec(
            name="swing_aggressive",
            profile_name="swing_aggressive",
            ingress_instance="",
            threshold_shift=args.swing_threshold_shift,
            symbols_core=args.swing_symbols_core,
            symbols_volatile=args.swing_symbols_volatile,
            symbols_defensive=args.swing_symbols_defensive,
            context_symbols=args.swing_context_symbols,
            interval_seconds=max(args.swing_interval_seconds, 5),
        ),
    ]
    if not bool(args.split_ingress_loops) or str(args.broker).strip().lower() != "schwab":
        return workers

    split_workers: list[WorkerSpec] = []
    for worker in workers:
        main_core = _parse_symbols(worker.symbols_core)
        main_volatile = _parse_symbols(worker.symbols_volatile)
        defensive = _parse_symbols(worker.symbols_defensive)
        if main_core or main_volatile:
            split_workers.append(
                WorkerSpec(
                    name=f"{worker.profile_name}_core_volatile",
                    profile_name=worker.profile_name,
                    ingress_instance="core_volatile",
                    threshold_shift=worker.threshold_shift,
                    symbols_core=_join_symbols(main_core),
                    symbols_volatile=_join_symbols(main_volatile),
                    symbols_defensive="",
                    context_symbols=worker.context_symbols,
                    interval_seconds=worker.interval_seconds,
                )
            )
        if defensive:
            defensive_interval = (
                max(int(args.intraday_defensive_interval_seconds), worker.interval_seconds)
                if worker.profile_name == "intraday_aggressive"
                else max(int(args.swing_defensive_interval_seconds), worker.interval_seconds)
            )
            split_workers.append(
                WorkerSpec(
                    name=f"{worker.profile_name}_defensive",
                    profile_name=worker.profile_name,
                    ingress_instance="defensive",
                    threshold_shift=worker.threshold_shift,
                    symbols_core="",
                    symbols_volatile="",
                    symbols_defensive=_join_symbols(defensive),
                    context_symbols=worker.context_symbols,
                    interval_seconds=max(defensive_interval, 5),
                )
            )
    return split_workers or workers


def _write_aggregated_ingress_states(workers: list[WorkerSpec], broker: str) -> None:
    by_profile: dict[str, list[WorkerSpec]] = {}
    for worker in workers:
        by_profile.setdefault(worker.profile_name, []).append(worker)
    for profile_name, rows in by_profile.items():
        expected_instances = [worker.ingress_instance for worker in rows if worker.ingress_instance]
        child_payloads = []
        for worker in rows:
            if not worker.ingress_instance:
                continue
            child_payloads.append(
                _load_json(
                    _ingress_summary_path(
                        profile_name=profile_name,
                        broker=broker,
                        ingress_instance=worker.ingress_instance,
                    )
                )
            )
        if not child_payloads:
            continue
        payload = _aggregate_ingress_payload(
            profile_name,
            broker,
            child_payloads,
            expected_instances=expected_instances,
        )
        if payload:
            _write_json(_ingress_summary_path(profile_name=profile_name, broker=broker), payload)


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
    parser.add_argument(
        "--split-ingress-loops",
        action="store_true",
        default=os.getenv("AGGRESSIVE_SPLIT_INGRESS_LOOPS", "1").strip().lower() in {"1", "true", "yes", "on"},
    )
    parser.add_argument(
        "--intraday-defensive-interval-seconds",
        type=int,
        default=int(os.getenv("INTRADAY_AGGRESSIVE_DEFENSIVE_INTERVAL_SECONDS", "20")),
    )
    parser.add_argument(
        "--swing-defensive-interval-seconds",
        type=int,
        default=int(os.getenv("SWING_AGGRESSIVE_DEFENSIVE_INTERVAL_SECONDS", "150")),
    )
    parser.add_argument(
        "--split-stagger-seconds",
        type=float,
        default=float(os.getenv("AGGRESSIVE_SPLIT_STAGGER_SECONDS", "2")),
    )

    parser.add_argument("--intraday-symbols-core", default=os.getenv("SHADOW_SYMBOLS_CORE", "SPY,QQQ,DIA,IWM,MDY,VOO,VTI,RSP,AAPL,MSFT,NVDA,AMZN,GOOGL,META,AVGO,ORCL,CRM,ADBE,NFLX,DIS,WBD,GS,JPM,BKNG,ABNB,MAR,HLT"))
    parser.add_argument("--intraday-symbols-volatile", default=os.getenv("SHADOW_SYMBOLS_VOLATILE", "SOXL,SOXS,TQQQ,SQQQ,MSTR,SMCI,COIN,TSLA,PLTR,AMD,MRVL,ARM,IBIT,ETHA,MARA,RIOT,UVXY,VIXY,AAL,UAL,DAL,LUV,ALK,JBLU,CCL,RCL,NCLH,EXPE,JETS,XOP,OIH,OXY,SLB,HAL"))
    parser.add_argument("--intraday-symbols-defensive", default=os.getenv("SHADOW_SYMBOLS_DEFENSIVE", "TLT,GLD,XLV,XLU,XLP,MO,HYG,LQD,UUP,XLE,XLF,XLI,XLK,XLY,IEF,SHY,TIP,TLH,JNK,AGG,BND,MUB,IGIB,USHY,FLOT,VGIT,SCHD,VIG,DGRO,HDV,NOBL,VYM,DIVO,JEPI,JEPQ,SPLV,VTV,JNJ,PG,KO,PEP,MCD,ABBV,ABT,MRK,PFE,T,VZ,O,VICI,MAIN,XOM,CVX,COP,EOG,MPC,PSX,VLO,KMI,ITA,LMT,NOC,RTX,GD,LHX,LDOS"))
    parser.add_argument("--intraday-context-symbols", default=os.getenv("WATCH_CONTEXT_SYMBOLS", "$VIX.X,UUP"))

    parser.add_argument("--swing-symbols-core", default=os.getenv("SHADOW_SYMBOLS_CORE", "SPY,QQQ,DIA,IWM,MDY,VOO,VTI,RSP,AAPL,MSFT,NVDA,AMZN,GOOGL,META,AVGO,ORCL,CRM,ADBE,NFLX,DIS,WBD,GS,JPM,BKNG,ABNB,MAR,HLT"))
    parser.add_argument("--swing-symbols-volatile", default=os.getenv("SHADOW_SYMBOLS_VOLATILE", "SOXL,SOXS,TQQQ,SQQQ,MSTR,SMCI,COIN,TSLA,PLTR,AMD,MRVL,ARM,IBIT,ETHA,MARA,RIOT,UVXY,VIXY,AAL,UAL,DAL,LUV,ALK,JBLU,CCL,RCL,NCLH,EXPE,JETS,XOP,OIH,OXY,SLB,HAL"))
    parser.add_argument(
        "--swing-symbols-defensive",
        default=(os.getenv("SHADOW_SYMBOLS_DEFENSIVE", "TLT,GLD,XLV,XLU,XLP,MO,HYG,LQD,UUP,XLE,XLF,XLI,XLK,XLY,IEF,SHY,TIP,TLH,JNK,AGG,BND,MUB,IGIB,USHY,FLOT,VGIT,SCHD,VIG,DGRO,HDV,NOBL,VYM,DIVO,JEPI,JEPQ,SPLV,VTV,JNJ,PG,KO,PEP,MCD,ABBV,ABT,MRK,PFE,T,VZ,O,VICI,MAIN,XOM,CVX,COP,EOG,MPC,PSX,VLO,KMI,ITA,LMT,NOC,RTX,GD,LHX,LDOS")
                 + "," + os.getenv("SHADOW_SYMBOLS_COMMOD_FX_INTL", "DBC,UNG,CORN,SLV,USO,FXE,FXY,EFA,EEM,EWJ,FXI,VEA,VWO,IEFA,VGK,INDA,SMH,SOXX,VGT,IGV,XOP,OIH,JETS,VNQ,IYR")).strip(","),
    )
    parser.add_argument("--swing-context-symbols", default=os.getenv("WATCH_CONTEXT_SYMBOLS", "$VIX.X,UUP"))
    args = parser.parse_args()

    if not _route_storage_or_fail():
        return 5

    if _global_trading_halt_enabled():
        print("GLOBAL_TRADING_HALT=1 set; refusing to start aggressive modes.")
        return 3

    if not _resource_guard_ok():
        return 4

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

    workers = _build_worker_specs(args)
    first_worker = workers[0]
    procs: dict[str, subprocess.Popen] = {}

    first_proc = _spawn_profile(
        worker=first_worker,
        broker=args.broker,
        simulate=args.simulate,
        max_iterations=args.max_iterations,
    )
    _bootstrap_ingress_state(first_worker, args.broker)
    procs[first_worker.name] = first_proc
    print(f"Started {first_worker.name} pid={first_proc.pid}")
    t1 = threading.Thread(target=_stream, args=(first_worker.name, first_proc.stdout), daemon=True)
    t1.start()

    if not args.simulate and not (TOKEN_PATH.exists() and TOKEN_PATH.stat().st_size > 0):
        print("token.json missing: waiting for aggressive OAuth bootstrap before starting remaining workers...")
        ok = _wait_for_token_or_exit(first_proc, args.auth_bootstrap_timeout_seconds)
        if not ok:
            _stop_processes(list(procs.values()))
            print("Stopped: aggressive bootstrap exited or OAuth bootstrap timeout before token.json.")
            return 1
        time.sleep(2.0)

    for worker in workers[1:]:
        if args.split_stagger_seconds > 0:
            time.sleep(float(args.split_stagger_seconds))
        proc = _spawn_profile(
            worker=worker,
            broker=args.broker,
            simulate=args.simulate,
            max_iterations=args.max_iterations,
        )
        _bootstrap_ingress_state(worker, args.broker)
        procs[worker.name] = proc
        print(f"Started {worker.name} pid={proc.pid}")
        t = threading.Thread(target=_stream, args=(worker.name, proc.stdout), daemon=True)
        t.start()

    domain = _domain_for_broker(args.broker)
    print(
        "Logs: decision_explanations/shadow_intraday_aggressive_{domain} and "
        "decision_explanations/shadow_swing_aggressive_{domain}".format(domain=domain)
    )
    if any(worker.ingress_instance for worker in workers):
        print(
            "Ingress split enabled: aggregated summaries written to "
            f"{_ingress_summary_path(profile_name='intraday_aggressive', broker=args.broker)} and "
            f"{_ingress_summary_path(profile_name='swing_aggressive', broker=args.broker)}"
        )

    try:
        while True:
            _write_aggregated_ingress_states(workers, args.broker)
            if _global_trading_halt_enabled():
                print("GLOBAL_TRADING_HALT=1 detected; stopping aggressive modes.")
                _stop_processes(list(procs.values()))
                return 0

            exits = [p.poll() for p in procs.values()]
            if any(code is not None for code in exits):
                _stop_processes(list(procs.values()))
                _write_aggregated_ingress_states(workers, args.broker)
                print(f"Stopped because one aggressive mode exited: {exits}")
                return 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping aggressive modes...")
        _stop_processes(list(procs.values()))
        _write_aggregated_ingress_states(workers, args.broker)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
