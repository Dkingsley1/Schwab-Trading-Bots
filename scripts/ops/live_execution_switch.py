#!/usr/bin/env python3
"""Explicit local operator permission for live execution; no order submission."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.live_execution_switch import (
    PURPOSES,
    WARNING,
    switch_off,
    switch_on,
    switch_status,
)


def notify_transition(command, result):
    from scripts.ops.mac_notification_watch import _notify_mac

    verified = result.get("ok") is True and result.get("switch") in {"ON", "OFF"}
    if verified:
        title = "Live Execution " + result["switch"]
        if result["switch"] == "ON":
            body = (
                f"Permission: {result.get('symbol')} / {result.get('purpose')} / "
                f"{result.get('session')}, until {result.get('expires_at_utc')}. "
                "Existing checks still apply. No order submitted."
            )
        else:
            body = "New orders and replacements blocked in updated processes. Pending orders and holdings unchanged. Platform power unchanged."
    else:
        title = f"Live Execution {command.upper()} not confirmed"
        reasons = result.get("activation_blockers") or [
            result.get("error") or "state verification failed"
        ]
        body = (
            "; ".join(str(item) for item in reasons[:3])
            + ". Check status and the broker; do not assume the requested change succeeded."
        )
    try:
        return _notify_mac(
            title,
            body,
            subtitle="Real-money permission - not platform power",
            group_key="schwab_live_execution_switch",
            open_target=(ROOT / "COMMANDS.md").as_uri(),
        )
    except Exception as exc:
        # Notification failure cannot undo or disguise a verified OFF write.
        return {"returncode": 1, "error": type(exc).__name__}


def main(argv=None):
    parser = argparse.ArgumentParser(description=WARNING)
    parser.add_argument(
        "command", choices=("status", "off", "on"), nargs="?", default="status"
    )
    parser.add_argument("--purpose", choices=PURPOSES)
    parser.add_argument("--symbol")
    parser.add_argument("--session", choices=("NORMAL", "AM", "PM"), default="NORMAL")
    parser.add_argument(
        "--minutes", type=int, choices=range(1, 61), default=30, metavar="1..60"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "on":
            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                raise ValueError(
                    "interactive_operator_terminal_required_no_unattended_activation"
                )
            print(WARNING)
            purpose = (
                args.purpose
                or input("Scope (supervised_broker_test / production_canary): ").strip()
            )
            symbol = (args.symbol or input("Symbol: ")).strip().upper()
            if purpose not in PURPOSES:
                raise ValueError("invalid_scope")
            phrase = f"ENABLE LIVE PERMISSION {purpose} {symbol} {args.session} {args.minutes} MINUTES"
            print(
                "Existing order checks and exact per-order confirmation remain mandatory for supervised tests."
            )
            if input(f"Type exactly: {phrase}\n") != phrase:
                raise ValueError("exact_live_switch_confirmation_required")
            result = switch_on(
                ROOT,
                purpose=purpose,
                symbol=symbol,
                session=args.session,
                minutes=args.minutes,
            )
        elif args.command == "off":
            result = switch_off(ROOT)
        else:
            result = switch_status(ROOT)
    except (OSError, ValueError, RuntimeError, EOFError) as exc:
        result = {
            "ok": False,
            "transition_verified": False,
            "error": str(exc),
            "warning": "Do not assume a failed OFF write stopped orders. Check status and the broker directly.",
        }
    if args.command in {"on", "off"}:
        result["notification"] = notify_transition(args.command, result)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
