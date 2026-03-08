import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = PROJECT_ROOT / "token.json"


def _is_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    return os.access(path, os.W_OK)


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _parse_symbols(value: str) -> List[str]:
    return [s.strip().upper() for s in value.split(",") if s.strip()]


def _proc_running(match: str, exclude: List[str] | None = None) -> int:
    proc = subprocess.run(["ps", "-ax", "-o", "command="], capture_output=True, text=True, check=False)
    out = proc.stdout or ""
    excludes = [x for x in (exclude or []) if x]
    count = 0
    for line in out.splitlines():
        if match not in line:
            continue
        if any(marker in line for marker in excludes):
            continue
        count += 1
    return count


def _check_disk(min_free_gb: float) -> Dict[str, object]:
    usage = shutil.disk_usage(PROJECT_ROOT)
    free_gb = usage.free / (1024 ** 3)
    ok = free_gb >= min_free_gb
    return {
        "name": "disk_free",
        "ok": ok,
        "details": f"free_gb={free_gb:.2f} min_required_gb={min_free_gb:.2f}",
    }


def _check_storage_route() -> Dict[str, object]:
    external_mount = os.getenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS").strip() or "/Volumes/BOT_LOGS"
    external_dir = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", "schwab_trading_bot").strip() or "schwab_trading_bot"
    external_root_cfg = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "").strip()
    if external_root_cfg:
        external_root = Path(external_root_cfg).expanduser()
    else:
        external_root = Path(external_mount).expanduser() / external_dir

    local_root = Path(
        os.getenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(PROJECT_ROOT / "local_fallback_storage"))
    ).expanduser()

    prefer_external = os.getenv("BOT_LOGS_PREFER_EXTERNAL", "1").strip().lower() not in {"0", "false", "no", "off"}
    external_ok = _is_writable_directory(external_root)
    local_ok = _is_writable_directory(local_root)
    mode = "external" if (prefer_external and external_ok) else "local_fallback"

    return {
        "name": "storage_route_writable",
        "ok": external_ok or local_ok,
        "details": (
            f"mode={mode} external_ok={int(external_ok)} local_ok={int(local_ok)} "
            f"external_root={external_root} local_root={local_root}"
        ),
    }


def _check_safety_env() -> List[Dict[str, object]]:
    mdo = os.getenv("MARKET_DATA_ONLY", "1").strip()
    aoe = os.getenv("ALLOW_ORDER_EXECUTION", "0").strip()
    return [
        {
            "name": "market_data_only_lock",
            "ok": mdo == "1",
            "details": f"MARKET_DATA_ONLY={mdo}",
        },
        {
            "name": "allow_order_execution_lock",
            "ok": aoe == "0",
            "details": f"ALLOW_ORDER_EXECUTION={aoe}",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checks before starting shadow runtimes.")
    parser.add_argument("--broker", choices=["schwab", "coinbase"], default=os.getenv("DATA_BROKER", "schwab"))
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--symbols-core", default=os.getenv("SHADOW_SYMBOLS_CORE", ""))
    parser.add_argument("--symbols-volatile", default=os.getenv("SHADOW_SYMBOLS_VOLATILE", ""))
    parser.add_argument("--symbols-defensive", default=os.getenv("SHADOW_SYMBOLS_DEFENSIVE", ""))
    parser.add_argument("--extra-symbols", default=os.getenv("SHADOW_SYMBOLS_COMMOD_FX_INTL", ""))
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--allow-running", action="store_true", help="Do not fail if matching runtime process is already active.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checks: List[Dict[str, object]] = []

    checks.extend(_check_safety_env())

    checks.append(
        {
            "name": "global_halt_not_set",
            "ok": not _flag("GLOBAL_TRADING_HALT", "0"),
            "details": f"GLOBAL_TRADING_HALT={os.getenv('GLOBAL_TRADING_HALT', '0').strip()}",
        }
    )

    checks.append(_check_storage_route())

    core = _parse_symbols(args.symbols_core)
    vol = _parse_symbols(args.symbols_volatile)
    defensive = _parse_symbols(args.symbols_defensive + ("," + args.extra_symbols if args.extra_symbols else ""))
    total = len(dict.fromkeys(core + vol + defensive))
    checks.append(
        {
            "name": "symbol_groups_non_empty",
            "ok": total > 0,
            "details": f"core={len(core)} volatile={len(vol)} defensive={len(defensive)} total={total}",
        }
    )

    checks.append(_check_disk(max(args.min_free_gb, 0.1)))

    if args.broker == "schwab" and not args.simulate:
        key = os.getenv("SCHWAB_API_KEY", "").strip()
        secret = os.getenv("SCHWAB_SECRET", "").strip()
        invalid = {
            "",
            "YOUR_KEY_HERE",
            "YOUR_SECRET_HERE",
            "YOUR_REAL_KEY",
            "YOUR_REAL_SECRET",
            "<real_key>",
            "<real_secret>",
        }
        checks.append(
            {
                "name": "schwab_credentials_present",
                "ok": key not in invalid and secret not in invalid,
                "details": "SCHWAB_API_KEY/SCHWAB_SECRET placeholders not allowed for non-simulate mode",
            }
        )
        checks.append(
            {
                "name": "token_present",
                "ok": TOKEN_PATH.exists() and TOKEN_PATH.stat().st_size > 0,
                "details": f"token_path={TOKEN_PATH}",
            }
        )

    if args.broker == "schwab":
        running = _proc_running(
            "scripts/run_parallel_shadows.py",
            exclude=[
                "scripts/shadow_watchdog.py",
                "scripts/ops/preflight_autofix.py",
                "scripts/shadow_preflight.py",
            ],
        )
        checks.append(
            {
                "name": "no_duplicate_parallel_launcher",
                "ok": args.allow_running or running == 0,
                "details": f"running={running}",
            }
        )
    else:
        running = _proc_running(
            "scripts/run_shadow_training_loop.py --broker coinbase",
            exclude=[
                "scripts/ops/preflight_autofix.py",
                "scripts/shadow_preflight.py",
            ],
        )
        checks.append(
            {
                "name": "no_duplicate_coinbase_loop",
                "ok": args.allow_running or running == 0,
                "details": f"running={running}",
            }
        )

    ok = all(bool(c.get("ok")) for c in checks)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "broker": args.broker,
        "simulate": bool(args.simulate),
        "checks": checks,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        status = "PASS" if ok else "FAIL"
        print(f"PREFLIGHT {status} broker={args.broker} simulate={int(bool(args.simulate))}")
        for c in checks:
            tag = "PASS" if c.get("ok") else "FAIL"
            print(f" - {tag} {c.get('name')}: {c.get('details')}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
