import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, time as dt_time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python

VENV_PY = resolve_runtime_python(PROJECT_ROOT)
SHADOW_LOOP = PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py"
FX_CONTEXT_SYNC = PROJECT_ROOT / "scripts" / "collect_fx_market_context.py"
LOAD_RUNTIME_ENV = PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh"
HEALTH_ROOT = PROJECT_ROOT / "governance" / "health"

DEFAULT_FX_SYMBOLS = "UUP,FXE,FXY,FXB,FXC,FXA,CYB,EUO,YCS,UDN"
DEFAULT_FX_CONTEXT_SYMBOLS = "SPY,QQQ,TLT,GLD,UUP,FXE,FXY,FXB,FXC,FXA"
DEFAULT_FX_REALTIME_SYMBOLS = "EURUSD,USDJPY,GBPUSD,USDCHF,USDCAD,AUDUSD"
DEFAULT_FX_REALTIME_CONTEXT_SYMBOLS = "EURUSD,USDJPY"


def _runtime_profile(simulate: bool) -> str:
    profile = os.getenv("BOT_RUNTIME_PROFILE", "sim" if simulate else "live").strip().lower()
    return profile if profile in {"sim", "live"} else ("sim" if simulate else "live")


def _bootstrap_runtime_env(base_env: dict[str, str], profile: str) -> dict[str, str]:
    if not LOAD_RUNTIME_ENV.exists():
        return base_env
    source_cmd = (
        f"source {shlex.quote(str(LOAD_RUNTIME_ENV))} {shlex.quote(profile)} --quiet >/dev/null 2>&1 && env -0"
    )
    result = subprocess.run(
        ["/bin/zsh", "-lc", source_cmd],
        cwd=str(PROJECT_ROOT),
        env=base_env,
        capture_output=True,
        text=False,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        return base_env
    merged = base_env.copy()
    for chunk in result.stdout.split(b"\0"):
        if not chunk or b"=" not in chunk:
            continue
        key, value = chunk.split(b"=", 1)
        merged[key.decode("utf-8", "ignore")] = value.decode("utf-8", "ignore")
    return merged


def _env_flag(env: dict[str, str], name: str, default: bool) -> bool:
    raw = str(env.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _parse_time(raw: str, fallback: dt_time) -> dt_time:
    text = str(raw or "").strip()
    if not text:
        return fallback
    try:
        hour_text, minute_text = text.split(":", 1)
        hour = max(0, min(int(hour_text), 23))
        minute = max(0, min(int(minute_text), 59))
        return dt_time(hour=hour, minute=minute)
    except Exception:
        return fallback


def _parse_iso_weekdays(raw: str) -> set[int]:
    out: set[int] = set()
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        if 1 <= value <= 7:
            out.add(value)
    return out or {1, 2, 3, 4, 5}


def _count_csv(raw: str) -> int:
    return len([part for part in str(raw or "").split(",") if part.strip()])


def _realtime_interval_seconds(env: dict[str, str], requested_interval: int, realtime_symbols: str) -> int:
    base_interval = max(int(requested_interval), 15)
    symbol_count = max(_count_csv(realtime_symbols), 1)
    max_credits_per_minute = max(int(str(env.get("FX_TWELVE_DATA_MAX_CREDITS_PER_MINUTE", "8") or "8")), 1)
    reserved_credits = max(int(str(env.get("FX_TWELVE_DATA_CREDIT_RESERVE", "2") or "2")), 0)
    usable_credits = max(max_credits_per_minute - reserved_credits, 1)
    throttled_interval = int(math.ceil((60.0 * float(symbol_count)) / float(usable_credits)))
    return max(base_interval, throttled_interval)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".ok").write_text("", encoding="utf-8")


def _terminate_child(child: subprocess.Popen | None) -> None:
    if child is None or child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    child.kill()
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _session_state(env: dict[str, str]) -> tuple[bool, dict[str, object]]:
    tz_name = str(env.get("FX_PROXY_SESSION_TIMEZONE", "America/New_York") or "America/New_York").strip()
    try:
        session_tz = ZoneInfo(tz_name)
    except Exception:
        session_tz = ZoneInfo("America/New_York")
        tz_name = "America/New_York"
    now_utc = datetime.now(timezone.utc)
    local_now = now_utc.astimezone(session_tz)
    session_days = _parse_iso_weekdays(env.get("FX_PROXY_SESSION_ISO_WEEKDAYS", "1,2,3,4,5"))
    start_time = _parse_time(env.get("FX_PROXY_SESSION_START", "09:30"), dt_time(hour=9, minute=30))
    end_time = _parse_time(env.get("FX_PROXY_SESSION_END", "16:00"), dt_time(hour=16, minute=0))
    local_clock = local_now.timetz().replace(tzinfo=None)
    same_day_window = start_time <= end_time
    if same_day_window:
        time_open = start_time <= local_clock < end_time
    else:
        time_open = local_clock >= start_time or local_clock < end_time
    open_now = local_now.isoweekday() in session_days and time_open
    payload = {
        "timestamp_utc": now_utc.isoformat(),
        "timezone": tz_name,
        "local_timestamp": local_now.isoformat(),
        "session_start_local": start_time.strftime("%H:%M"),
        "session_end_local": end_time.strftime("%H:%M"),
        "session_iso_weekdays": sorted(session_days),
        "weekday_open": local_now.isoweekday() in session_days,
        "time_open": bool(time_open),
        "open_now": bool(open_now),
    }
    return open_now, payload


def _forex_session_state(env: dict[str, str]) -> tuple[bool, dict[str, object]]:
    tz_name = str(env.get("FX_FOREX_SESSION_TIMEZONE", "America/New_York") or "America/New_York").strip()
    try:
        session_tz = ZoneInfo(tz_name)
    except Exception:
        session_tz = ZoneInfo("America/New_York")
        tz_name = "America/New_York"
    now_utc = datetime.now(timezone.utc)
    local_now = now_utc.astimezone(session_tz)
    week_start = _parse_time(env.get("FX_FOREX_SESSION_WEEK_START", "17:00"), dt_time(hour=17, minute=0))
    week_end = _parse_time(env.get("FX_FOREX_SESSION_WEEK_END", "17:00"), dt_time(hour=17, minute=0))
    local_clock = local_now.timetz().replace(tzinfo=None)
    weekday = local_now.isoweekday()
    open_now = False
    if weekday == 7:
        open_now = local_clock >= week_start
    elif weekday in {1, 2, 3, 4}:
        open_now = True
    elif weekday == 5:
        open_now = local_clock < week_end
    payload = {
        "timestamp_utc": now_utc.isoformat(),
        "timezone": tz_name,
        "local_timestamp": local_now.isoformat(),
        "week_start_local": week_start.strftime("%H:%M"),
        "week_end_local": week_end.strftime("%H:%M"),
        "weekday_iso": weekday,
        "open_now": bool(open_now),
    }
    return open_now, payload


def _run_fx_context_sync(env: dict[str, str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        [str(VENV_PY), str(FX_CONTEXT_SYNC), "--json"],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return int(proc.returncode), (proc.stdout or "").strip(), (proc.stderr or "").strip()


def _write_session_status(
    *,
    broker: str,
    symbols: str,
    context_symbols: str,
    session: dict[str, object],
    mode: str,
    reason: str,
) -> None:
    forex_session = session.get("forex_session") if isinstance(session.get("forex_session"), dict) else {}
    proxy_session = session.get("proxy_session") if isinstance(session.get("proxy_session"), dict) else {}
    ts = str(
        forex_session.get("timestamp_utc")
        or proxy_session.get("timestamp_utc")
        or datetime.now(timezone.utc).isoformat()
    )
    session_payload = {
        "timestamp_utc": ts,
        "broker": broker,
        "profile": "fx",
        "mode": mode,
        "reason": reason,
        "session": session,
        "symbols_total": _count_csv(symbols),
        "context_total": _count_csv(context_symbols),
        "log_schema_version": 1,
    }
    _write_json(HEALTH_ROOT / "fx_shadow_session_latest.json", session_payload)


def _write_off_hours_health(
    *,
    wrapper_pid: int,
    broker: str,
    symbols: str,
    context_symbols: str,
    iter_count: int,
    session: dict[str, object],
    mode: str,
    reason: str,
) -> None:
    forex_session = session.get("forex_session") if isinstance(session.get("forex_session"), dict) else {}
    proxy_session = session.get("proxy_session") if isinstance(session.get("proxy_session"), dict) else {}
    ts = str(
        forex_session.get("timestamp_utc")
        or proxy_session.get("timestamp_utc")
        or datetime.now(timezone.utc).isoformat()
    )
    run_id = f"fx-wrapper-{wrapper_pid}"
    iter_id = f"{run_id}:{iter_count}"
    symbol_count = _count_csv(symbols)
    context_count = _count_csv(context_symbols)
    loop_payload = {
        "timestamp_utc": ts,
        "pid": int(wrapper_pid),
        "broker": broker,
        "profile": "fx",
        "domain": "equities",
        "run_id": run_id,
        "iter_id": iter_id,
        "iter": int(iter_count),
        "symbols_total": symbol_count,
        "context_total": context_count,
        "state": mode,
        "off_hours_reason": reason,
        "session": session,
        "log_schema_version": 2,
    }
    ingress_payload = {
        "timestamp_utc": ts,
        "run_id": run_id,
        "iter_id": iter_id,
        "iter": int(iter_count),
        "broker": broker,
        "profile": "fx",
        "domain": "equities",
        "loop_state": mode,
        "iter_counts": {
            "cache_ok": 0,
            "simulate_ok": 0,
            "api_ok": 0,
            "api_error": 0,
        },
        "total_counts": {
            "cache_ok": 0,
            "simulate_ok": 0,
            "api_ok": 0,
            "api_error": 0,
        },
        "iter_total_requests": 0,
        "iter_error_count": 0,
        "iter_error_rate": 0.0,
        "symbols_total": symbol_count,
        "context_total": context_count,
        "off_hours_reason": reason,
        "session": session,
        "log_schema_version": 2,
    }
    _write_json(HEALTH_ROOT / f"shadow_loop_fx_equities_{broker}_{wrapper_pid}.json", loop_payload)
    _write_json(HEALTH_ROOT / f"data_ingress_latest_fx_equities_{broker}.json", ingress_payload)
    _write_session_status(
        broker=broker,
        symbols=symbols,
        context_symbols=context_symbols,
        session=session,
        mode=mode,
        reason=reason,
    )


def _build_loop_cmd(
    *,
    args: argparse.Namespace,
    symbols: str,
    context_symbols: str,
    interval_seconds: int | None = None,
) -> list[str]:
    cmd = [
        str(VENV_PY),
        str(SHADOW_LOOP),
        "--broker",
        args.broker,
        "--profile",
        "fx",
        "--domain",
        "equities",
        "--symbols",
        symbols,
        "--context-symbols",
        context_symbols,
        "--interval-seconds",
        str(int(interval_seconds if interval_seconds is not None else args.interval_seconds)),
        "--max-iterations",
        str(args.max_iterations),
    ]
    if args.simulate:
        cmd.append("--simulate")
    if args.auto_retrain:
        cmd.append("--auto-retrain")
    return cmd


def _run_supervised_session(args: argparse.Namespace, env: dict[str, str]) -> int:
    wrapper_pid = os.getpid()
    child: subprocess.Popen | None = None
    iter_count = 0
    off_hours_only = _env_flag(env, "FX_OFF_HOURS_CONTEXT_ONLY", True)
    realtime_quotes_enabled = _env_flag(env, "FX_REALTIME_QUOTES_ENABLED", True) and bool(str(env.get("TWELVE_DATA_API_KEY", "")).strip())
    realtime_symbols = str(env.get("FX_REALTIME_SYMBOLS", DEFAULT_FX_REALTIME_SYMBOLS) or DEFAULT_FX_REALTIME_SYMBOLS)
    realtime_context_symbols = str(env.get("FX_REALTIME_CONTEXT_SYMBOLS", DEFAULT_FX_REALTIME_CONTEXT_SYMBOLS) or DEFAULT_FX_REALTIME_CONTEXT_SYMBOLS)
    realtime_interval_seconds = _realtime_interval_seconds(env, int(args.interval_seconds), realtime_symbols)
    sync_interval = max(int(str(env.get("FX_OFF_HOURS_CONTEXT_SYNC_SECONDS", "300")) or "300"), 60)
    poll_interval = max(min(int(args.interval_seconds), 60), 15)
    last_sync_ts = 0.0
    last_mode: str | None = None
    active_run_mode: str | None = None

    try:
        while True:
            iter_count += 1
            proxy_session_open, proxy_session = _session_state(env)
            forex_session_open, forex_session = _forex_session_state(env)
            session = {
                "forex_session": forex_session,
                "proxy_session": proxy_session,
            }
            desired_mode = ""
            desired_reason = ""
            desired_symbols = args.symbols
            desired_context_symbols = args.context_symbols
            if proxy_session_open or (not off_hours_only):
                desired_mode = "live_proxy_market_hours"
                desired_reason = "proxy_market_open"
            elif forex_session_open and realtime_quotes_enabled:
                desired_mode = "live_forex_quotes"
                desired_reason = "forex_session_open_twelve_data"
                desired_symbols = realtime_symbols
                desired_context_symbols = realtime_context_symbols

            if desired_mode:
                _write_session_status(
                    broker=args.broker,
                    symbols=desired_symbols,
                    context_symbols=desired_context_symbols,
                    session=session,
                    mode=desired_mode,
                    reason=desired_reason,
                )
                if child is None or child.poll() is not None or active_run_mode != desired_mode:
                    if child is not None and child.poll() is None and active_run_mode != desired_mode:
                        _terminate_child(child)
                    if last_mode != desired_mode:
                        print(
                            f"[FXSession] state={desired_mode} "
                            f"forex_open={int(forex_session_open)} proxy_open={int(proxy_session_open)} "
                            f"local={proxy_session.get('local_timestamp')} tz={proxy_session.get('timezone')} "
                            f"interval={realtime_interval_seconds if desired_mode == 'live_forex_quotes' else int(args.interval_seconds)}"
                        )
                    child = subprocess.Popen(
                        _build_loop_cmd(
                            args=args,
                            symbols=desired_symbols,
                            context_symbols=desired_context_symbols,
                            interval_seconds=(realtime_interval_seconds if desired_mode == "live_forex_quotes" else int(args.interval_seconds)),
                        ),
                        cwd=str(PROJECT_ROOT),
                        env=env,
                    )
                    active_run_mode = desired_mode
                last_mode = desired_mode
            else:
                if child is not None and child.poll() is None:
                    print(
                        "[FXSession] state=context_only_off_hours stopping_live_proxy_loop "
                        f"local={proxy_session.get('local_timestamp')} tz={proxy_session.get('timezone')}"
                    )
                    _terminate_child(child)
                child = None
                active_run_mode = None
                now_ts = time.time()
                if now_ts - last_sync_ts >= float(sync_interval):
                    rc, out, err = _run_fx_context_sync(env)
                    sync_summary = out.splitlines()[-1] if out else ""
                    err_summary = err.splitlines()[-1] if err else ""
                    print(
                        "[FXSession] context_sync "
                        f"rc={rc} summary={sync_summary or 'none'}"
                        + (f" err={err_summary}" if err_summary else "")
                    )
                    last_sync_ts = now_ts
                off_hours_mode = "forex_session_context_only" if forex_session_open else "forex_weekend_closed"
                off_hours_reason = "forex_session_open_proxy_market_closed" if forex_session_open else "forex_weekend_closed"
                _write_off_hours_health(
                    wrapper_pid=wrapper_pid,
                    broker=args.broker,
                    symbols=args.symbols,
                    context_symbols=args.context_symbols,
                    iter_count=iter_count,
                    session=session,
                    mode=off_hours_mode,
                    reason=off_hours_reason,
                )
                if last_mode != "off_hours":
                    print(
                        f"[FXSession] state={off_hours_mode} "
                        f"forex_open={int(forex_session_open)} "
                        f"proxy_open={int(proxy_session_open)} "
                        f"local={proxy_session.get('local_timestamp')} tz={proxy_session.get('timezone')}"
                    )
                last_mode = "off_hours"
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        return 130
    finally:
        _terminate_child(child)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dedicated FX shadow profile (paper-only proxy lane).")
    parser.add_argument("--broker", default=os.getenv("DATA_BROKER", "schwab"), choices=["schwab", "coinbase"])
    parser.add_argument("--simulate", action="store_true", help="Use simulated market feed.")
    parser.add_argument("--symbols", default=os.getenv("FX_SYMBOLS", DEFAULT_FX_SYMBOLS))
    parser.add_argument("--context-symbols", default=os.getenv("FX_CONTEXT_SYMBOLS", DEFAULT_FX_CONTEXT_SYMBOLS))
    parser.add_argument("--interval-seconds", type=int, default=int(os.getenv("FX_SHADOW_INTERVAL", "45")))
    parser.add_argument("--max-iterations", type=int, default=int(os.getenv("FX_SHADOW_MAX_ITERS", "0")))
    parser.add_argument("--auto-retrain", action="store_true", default=False)
    args = parser.parse_args()

    if not VENV_PY.exists():
        print(f"ERROR: missing venv python: {VENV_PY}")
        return 2
    if not SHADOW_LOOP.exists():
        print(f"ERROR: missing shadow loop script: {SHADOW_LOOP}")
        return 2

    env = _bootstrap_runtime_env(os.environ.copy(), _runtime_profile(args.simulate))
    env["MARKET_DATA_ONLY"] = "1"
    env["ALLOW_ORDER_EXECUTION"] = "0"
    env["SHADOW_PROFILE"] = "fx"
    env["SHADOW_DOMAIN"] = "equities"
    env.setdefault("SHADOW_THRESHOLD_SHIFT", "+0.01")

    print("Starting FX shadow profile...")
    print("Symbols:", args.symbols)
    print("Context symbols:", args.context_symbols)
    print("Command:", " ".join(_build_loop_cmd(args=args, symbols=args.symbols, context_symbols=args.context_symbols)))

    if args.simulate or int(args.max_iterations) > 0:
        proc = subprocess.Popen(
            _build_loop_cmd(args=args, symbols=args.symbols, context_symbols=args.context_symbols),
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        return proc.wait()
    return _run_supervised_session(args, env)


if __name__ == "__main__":
    raise SystemExit(main())
