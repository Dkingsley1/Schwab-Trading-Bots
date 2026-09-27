"""Read-only, explicit-symbol capture for shared decision reports. Never orders."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.decision_candle_store import identity, publish
from core.schd_capture_store import checked
from scripts.ops.schd_candle_report import READ_ONLY_ENV, fetch_child


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--fetch-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        identity("schwab", args.symbol)
        if "-USD" in args.symbol:
            raise ValueError("use_coinbase_observer_for_spot_crypto")
        if checked(ROOT, ROOT / "governance/health/SYSTEM_POWER_OFF.flag").exists():
            raise ValueError("system_power_off")
        if args.fetch_child:
            captured = fetch_child(ROOT, symbol=args.symbol)
            captured["source"]["observed_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            print(json.dumps(captured, allow_nan=False))
            return 0
        child = subprocess.run(
            [sys.executable, __file__, "--symbol", args.symbol, "--fetch-child"],
            cwd=ROOT,
            env=dict(os.environ, **READ_ONLY_ENV),
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if child.returncode or len(child.stdout) > 1024 * 1024:
            raise ValueError("bounded_schwab_capture_failed_check_provider_health")
        receipt = publish(ROOT, json.loads(child.stdout))
        print(
            json.dumps(
                {"ok": True, "candle_context": receipt, "order_authority": False}
            )
        )
        return 0
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": (
                        str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                    ),
                    "order_authority": False,
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
