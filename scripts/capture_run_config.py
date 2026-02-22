import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "governance" / "session_configs"

KEYS = [
    "MARKET_DATA_ONLY",
    "ALLOW_ORDER_EXECUTION",
    "DATA_BROKER",
    "SHADOW_SYMBOLS_CORE",
    "SHADOW_SYMBOLS_VOLATILE",
    "SHADOW_SYMBOLS_DEFENSIVE",
    "SHADOW_SYMBOLS_COMMOD_FX_INTL",
    "FEATURE_TIMEFRAMES",
    "FEATURE_WINDOWS",
    "SHADOW_LOOP_INTERVAL",
    "GLOBAL_TRADING_HALT",
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(PROJECT_ROOT),
        "env": {k: os.getenv(k, "") for k in KEYS},
    }
    out = OUT_DIR / f"session_config_{ts}.json"
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    latest = OUT_DIR / "latest.json"
    latest.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
