import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.halt_flags import write_halt_flag_atomic

OPERATOR_FLAG = PROJECT_ROOT / "governance" / "health" / "OPERATOR_STOP.flag"
GLOBAL_HALT_FLAG = PROJECT_ROOT / "governance" / "health" / "GLOBAL_TRADING_HALT.flag"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_flag(path: Path, payload: dict[str, Any]) -> None:
    ok = write_halt_flag_atomic(
        path,
        payload,
        project_root=str(PROJECT_ROOT),
        source="operator_control",
    )
    if not ok:
        raise RuntimeError(f"flag_write_failed:{path}")


def _remove_flag(path: Path) -> bool:
    if not path.exists():
        return False
    path.unlink()
    return True


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Operator red-button control for live trading safety flags.")
    parser.add_argument("--engage", action="store_true", help="Set OPERATOR_STOP flag (big red button).")
    parser.add_argument("--release", action="store_true", help="Clear OPERATOR_STOP flag.")
    parser.add_argument("--set-global-halt", action="store_true", help="Also set GLOBAL_TRADING_HALT flag.")
    parser.add_argument("--clear-global-halt", action="store_true", help="Also clear GLOBAL_TRADING_HALT flag.")
    parser.add_argument("--reason", default="operator_manual_override")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    operator_action = "none"
    global_action = "none"

    if args.engage and args.release:
        raise SystemExit("cannot_use_both_engage_and_release")

    if args.engage:
        _write_flag(
            OPERATOR_FLAG,
            {
                "timestamp_utc": _now(),
                "reason": str(args.reason or "operator_manual_override"),
                "operator": os.getenv("USER", "unknown"),
                "source": "scripts/operator_control.py",
            },
        )
        operator_action = "engaged"
    elif args.release:
        removed = _remove_flag(OPERATOR_FLAG)
        operator_action = "released" if removed else "already_clear"

    if args.set_global_halt and args.clear_global_halt:
        raise SystemExit("cannot_use_both_set_and_clear_global_halt")

    if args.set_global_halt:
        _write_flag(
            GLOBAL_HALT_FLAG,
            {
                "timestamp_utc": _now(),
                "reason": str(args.reason or "operator_manual_override"),
                "operator": os.getenv("USER", "unknown"),
                "source": "scripts/operator_control.py",
            },
        )
        global_action = "halt_set"
    elif args.clear_global_halt:
        removed = _remove_flag(GLOBAL_HALT_FLAG)
        global_action = "halt_cleared" if removed else "already_clear"

    payload = {
        "timestamp_utc": _now(),
        "operator_action": operator_action,
        "global_action": global_action,
        "operator_stop": OPERATOR_FLAG.exists(),
        "global_halt": GLOBAL_HALT_FLAG.exists(),
        "operator_flag": str(OPERATOR_FLAG),
        "global_halt_flag": str(GLOBAL_HALT_FLAG),
        "operator_payload": _load(OPERATOR_FLAG) if OPERATOR_FLAG.exists() else {},
        "global_payload": _load(GLOBAL_HALT_FLAG) if GLOBAL_HALT_FLAG.exists() else {},
    }

    latest = PROJECT_ROOT / "governance" / "health" / "operator_control_latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "operator_control "
            f"operator_stop={int(payload['operator_stop'])} global_halt={int(payload['global_halt'])} "
            f"operator_action={operator_action} global_action={global_action}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
