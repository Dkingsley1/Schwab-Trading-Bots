import argparse
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.base_trader import BaseTrader
from core.channel_queue import ChannelQueue
from core.execution_lane_pipeline import (
    EXECUTION_INTENT_CHANNEL,
    EXECUTION_PROMOTED_CHANNEL,
    configure_trader_for_lane,
    process_execution_intent,
    queue_db_path,
    update_lane_health,
)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _live_execution_enabled() -> bool:
    return _env_flag("TOP_BOT_ENABLE_LIVE_EXECUTION", "0") or _env_flag("EXECUTION_LANE_LIVE_ENABLED", "0")


def _build_trader(mode: str) -> tuple[BaseTrader, bool, str]:
    trader = BaseTrader(
        os.getenv("SCHWAB_API_KEY", "YOUR_KEY_HERE"),
        os.getenv("SCHWAB_SECRET", "YOUR_SECRET_HERE"),
        os.getenv("SCHWAB_REDIRECT", "https://127.0.0.1:8182"),
        mode=mode,
    )
    configure_trader_for_lane(trader, mode)
    if mode != "live":
        return trader, True, ""

    try:
        trader.client = trader.authenticate()
        return trader, True, ""
    except Exception as exc:
        return trader, False, str(exc)


def _channel_for_mode(mode: str) -> str:
    return EXECUTION_INTENT_CHANNEL if mode == "paper" else EXECUTION_PROMOTED_CHANNEL


def main() -> int:
    parser = argparse.ArgumentParser(description="Run standalone paper/live execution lane consumer.")
    parser.add_argument("--mode", choices=("paper", "live"), required=True)
    parser.add_argument("--once", action="store_true", help="Process one batch and exit.")
    parser.add_argument("--limit", type=int, default=max(int(os.getenv("EXECUTION_LANE_BATCH_LIMIT", "200")), 1))
    parser.add_argument("--poll-seconds", type=float, default=max(float(os.getenv("EXECUTION_LANE_POLL_SECONDS", "2.0")), 0.2))
    parser.add_argument("--queue-db", default=os.getenv("BOT_CHANNEL_QUEUE_DB", ""))
    args = parser.parse_args()

    queue_path = queue_db_path(str(PROJECT_ROOT), args.queue_db)
    queue = ChannelQueue(queue_path)
    channel = _channel_for_mode(args.mode)
    consumer = f"execution_lane_{args.mode}"

    if args.mode == "live" and not _live_execution_enabled():
        auth_error = "live_execution_disabled_by_env"
        print(f"[ExecutionLane] {auth_error}")
        update_lane_health(
            project_root=str(PROJECT_ROOT),
            mode=args.mode,
            processed_count=0,
            queue_channel=channel,
            queue_db_override=args.queue_db,
            auth_ok=False,
            auth_error=auth_error,
        )
        return 3

    trader, auth_ok, auth_error = _build_trader(args.mode)
    if args.mode == "live" and not auth_ok:
        print(f"[ExecutionLane] live auth unavailable err={auth_error}")
        update_lane_health(
            project_root=str(PROJECT_ROOT),
            mode=args.mode,
            processed_count=0,
            queue_channel=channel,
            queue_db_override=args.queue_db,
            auth_ok=False,
            auth_error=auth_error,
        )
        return 2

    processed_total = 0
    while True:
        messages = queue.read_from_cursor(consumer=consumer, channel=channel, limit=max(int(args.limit), 1))
        if not messages:
            update_lane_health(
                project_root=str(PROJECT_ROOT),
                mode=args.mode,
                processed_count=processed_total,
                queue_channel=channel,
                queue_db_override=args.queue_db,
                auth_ok=auth_ok,
                auth_error=auth_error,
            )
            if args.once:
                return 0
            time.sleep(max(float(args.poll_seconds), 0.2))
            continue

        for message in messages:
            process_execution_intent(
                project_root=str(PROJECT_ROOT),
                trader=trader,
                mode=args.mode,
                message=message,
                queue_db_override=args.queue_db,
            )
            processed_total += 1

        queue.ack_messages(consumer=consumer, channel=channel, messages=messages)
        update_lane_health(
            project_root=str(PROJECT_ROOT),
            mode=args.mode,
            processed_count=processed_total,
            queue_channel=channel,
            queue_db_override=args.queue_db,
            auth_ok=auth_ok,
            auth_error=auth_error,
        )

        if args.once:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
