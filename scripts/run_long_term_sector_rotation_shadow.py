import argparse
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT_ROOT / ".venv312" / "bin" / "python"
SHADOW_LOOP = PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py"

DEFAULT_SECTOR_ROTATION_SYMBOLS = "XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY,SMH,SOXX,ITB,KRE,IBB,ITA,JETS,XOP,OIH"
DEFAULT_SECTOR_CONTEXT_SYMBOLS = "SPY,QQQ,IWM,TLT,GLD,UUP"
DEFAULT_SECTOR_QUALITY_SYMBOLS = "XLK,XLV,XLP,XLU,XLI,XLF,XLE,SMH,SOXX"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run long-term sector rotation shadow profile.")
    parser.add_argument("--broker", default=os.getenv("DATA_BROKER", "schwab"), choices=["schwab", "coinbase"])
    parser.add_argument("--simulate", action="store_true", help="Use simulated market feed.")
    parser.add_argument("--symbols", default=os.getenv("LONG_TERM_SECTOR_SYMBOLS", DEFAULT_SECTOR_ROTATION_SYMBOLS))
    parser.add_argument("--context-symbols", default=os.getenv("LONG_TERM_SECTOR_CONTEXT_SYMBOLS", DEFAULT_SECTOR_CONTEXT_SYMBOLS))
    parser.add_argument("--quality-symbols", default=os.getenv("LONG_TERM_SECTOR_QUALITY_SYMBOLS", DEFAULT_SECTOR_QUALITY_SYMBOLS))
    parser.add_argument("--interval-seconds", type=int, default=int(os.getenv("LONG_TERM_SECTOR_INTERVAL", "150")))
    parser.add_argument("--max-iterations", type=int, default=int(os.getenv("LONG_TERM_SECTOR_MAX_ITERS", "0")))
    parser.add_argument("--auto-retrain", action="store_true", default=False)
    args = parser.parse_args()

    if not VENV_PY.exists():
        print(f"ERROR: missing venv python: {VENV_PY}")
        return 2
    if not SHADOW_LOOP.exists():
        print(f"ERROR: missing shadow loop script: {SHADOW_LOOP}")
        return 2

    env = os.environ.copy()
    env["MARKET_DATA_ONLY"] = "1"
    env["ALLOW_ORDER_EXECUTION"] = "0"
    env["SHADOW_PROFILE"] = "long_term_sector_rotation"
    env["SHADOW_DOMAIN"] = "equities"
    env["LONG_TERM_STRICT_BUY_HOLD"] = "1"
    env["LONG_TERM_SECTOR_QUALITY_SYMBOLS"] = args.quality_symbols
    env.setdefault("LONG_TERM_HORIZON_YEARS", "10")
    env.setdefault("LONG_TERM_10Y_BUY_SCORE_MIN", "0.60")
    env.setdefault("LONG_TERM_10Y_STRONG_SCORE_MIN", "0.74")
    env.setdefault("SHADOW_THRESHOLD_SHIFT", "+0.02")

    cmd = [
        str(VENV_PY),
        str(SHADOW_LOOP),
        "--broker",
        args.broker,
        "--profile",
        "long_term_sector_rotation",
        "--domain",
        "equities",
        "--symbols",
        args.symbols,
        "--context-symbols",
        args.context_symbols,
        "--interval-seconds",
        str(args.interval_seconds),
        "--max-iterations",
        str(args.max_iterations),
    ]
    if args.simulate:
        cmd.append("--simulate")
    if args.auto_retrain:
        cmd.append("--auto-retrain")

    print("Starting long-term sector rotation profile...")
    print("Symbols:", args.symbols)
    print("Context symbols:", args.context_symbols)
    print("Quality symbols:", args.quality_symbols)
    print("Command:", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
