import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PAPER_TRADE_LOCK_PATH = PROJECT_ROOT / "governance" / "health" / "PAPER_TRADE_LOCK.flag"
CONTROL_ENV_FILES = (
    PROJECT_ROOT / "config" / ".env.runtime_resource_guard_override",
    PROJECT_ROOT / "config" / ".env.paper_400_ramp_override",
    PROJECT_ROOT / "config" / ".env.local_storage_reserve_override",
)
CONTROL_ENV_KEYS = {
    "EXECUTION_LANE_BATCH_SLEEP_SECONDS",
    "EXECUTION_LANE_BATCH_LIMIT",
    "EXECUTION_LANE_BACKLOG_SLEEP_SECONDS",
    "EXECUTION_LANE_HEALTH_UPDATE_SECONDS",
    "EXECUTION_LANE_HOST_LOAD_SLEEP_SECONDS",
    "EXECUTION_LANE_HOST_LOAD_SOFT_CAP",
    "EXECUTION_LANE_LIVE_MAX_INTENT_AGE_SECONDS",
    "EXECUTION_LANE_MESSAGE_SLEEP_SECONDS",
    "EXECUTION_LANE_PAPER_MAX_INTENT_AGE_SECONDS",
    "EXECUTION_LANE_POLL_SECONDS",
    "EXECUTION_LANE_STALE_FAST_DRAIN_ENABLED",
    "EXECUTION_LANE_STALE_FAST_DRAIN_LIMIT",
    "EXECUTION_LANE_STALE_FAST_DRAIN_PASSES",
    "PAPER_400_RAMP_BLOCKED_RUNTIME_PAUSE",
    "PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED",
    "PAPER_EXECUTION_RUNTIME_NICE",
    "PAPER_EXECUTION_RUNTIME_PAUSED_FOR_PRESSURE",
    "PAPER_EXECUTION_RUNTIME_PAUSED_FOR_LOCAL_STORAGE",
    "PAPER_RECONCILIATION_HEARTBEAT_WHEN_PAUSED",
    "PAPER_RECONCILIATION_HEARTBEAT_SECONDS",
    "PAPER_SHADOW_RUNTIME_NICE",
}
_CONTROL_ENV_VALUES: dict[str, str] = {}

from core.base_trader import BaseTrader
from core.brokers import BrokerRuntimeConfig, available_broker_names, normalize_broker_name
from core.channel_queue import ChannelQueue
from core.execution_lane_pipeline import (
    EXECUTION_INTENT_CHANNEL,
    EXECUTION_PROMOTED_CHANNEL,
    configure_trader_for_lane,
    emit_paper_reconciliation_heartbeat,
    process_execution_intent,
    publish_execution_result,
    queue_db_path,
    update_lane_health,
)


def _env_flag(name: str, default: str = "0") -> bool:
    return _control_env_value(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _clean_env_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_control_env() -> None:
    values: dict[str, str] = {}
    for path in CONTROL_ENV_FILES:
        if not path.exists() or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in CONTROL_ENV_KEYS:
                values[key] = _clean_env_value(value)
    _CONTROL_ENV_VALUES.clear()
    _CONTROL_ENV_VALUES.update(values)


def _control_env_value(name: str, default: str = "") -> str:
    if name in _CONTROL_ENV_VALUES:
        return _CONTROL_ENV_VALUES[name]
    return os.getenv(name, default)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(int(_control_env_value(name, str(default))), minimum)
    except ValueError:
        return max(int(default), minimum)


def _env_float(name: str, default: float, *, minimum: float = 0.2) -> float:
    try:
        return max(float(_control_env_value(name, str(default))), minimum)
    except ValueError:
        return max(float(default), minimum)


def _lane_health_update_due(
    last_update_monotonic: float,
    interval_seconds: float,
    *,
    now_monotonic: float | None = None,
) -> bool:
    now = time.monotonic() if now_monotonic is None else float(now_monotonic)
    return bool(
        last_update_monotonic <= 0.0
        or now - last_update_monotonic >= max(interval_seconds, 0.0)
    )


def _parse_ts(raw: object) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _message_created_at(message: object) -> datetime | None:
    payload = getattr(message, "payload", {}) if hasattr(message, "payload") else {}
    if not isinstance(payload, dict):
        payload = {}
    for raw in (
        getattr(message, "created_at", ""),
        payload.get("timestamp_utc"),
        payload.get("created_at"),
    ):
        parsed = _parse_ts(raw)
        if parsed is not None:
            return parsed
    return None


def _intent_max_age_seconds(mode: str) -> float:
    if str(mode or "").strip().lower() == "paper":
        return _env_float("EXECUTION_LANE_PAPER_MAX_INTENT_AGE_SECONDS", 900.0, minimum=0.0)
    return _env_float("EXECUTION_LANE_LIVE_MAX_INTENT_AGE_SECONDS", 60.0, minimum=0.0)


def _stale_intent_detail(mode: str, message: object) -> tuple[bool, float | None, float]:
    max_age_seconds = _intent_max_age_seconds(mode)
    if max_age_seconds <= 0.0:
        return False, None, max_age_seconds
    created_at = _message_created_at(message)
    if created_at is None:
        return False, None, max_age_seconds
    age_seconds = max((datetime.now(timezone.utc) - created_at).total_seconds(), 0.0)
    return age_seconds > max_age_seconds, round(age_seconds, 3), max_age_seconds


def _cooldown_sleep_seconds(*, batch_sleep_seconds: float, messages_read: int, batch_limit: int) -> float:
    sleep_seconds = max(float(batch_sleep_seconds), 0.0)
    load_cap = _env_float("EXECUTION_LANE_HOST_LOAD_SOFT_CAP", 0.0, minimum=0.0)
    if load_cap > 0.0:
        try:
            load_1m = float(os.getloadavg()[0])
        except Exception:
            load_1m = 0.0
        if load_1m >= load_cap:
            sleep_seconds = max(
                sleep_seconds,
                _env_float("EXECUTION_LANE_HOST_LOAD_SLEEP_SECONDS", 3.0, minimum=0.0),
            )
    if messages_read >= max(int(batch_limit), 1):
        sleep_seconds = max(
            sleep_seconds,
            _env_float("EXECUTION_LANE_BACKLOG_SLEEP_SECONDS", 0.0, minimum=0.0),
        )
    return sleep_seconds


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def _emit_stale_skip_batch(
    *,
    mode: str,
    messages: list,
    max_age_seconds: float,
    queue_db_override: str,
) -> None:
    if not messages:
        return
    created_values = [str(getattr(message, "created_at", "") or "") for message in messages]
    row = _stale_skip_audit_row(
        mode=mode,
        channel=str(getattr(messages[0], "channel", "") or ""),
        queue_db_override=queue_db_override,
        count=len(messages),
        first_id=int(getattr(messages[0], "id", 0) or 0),
        last_id=int(getattr(messages[-1], "id", 0) or 0),
        first_message_id=str(getattr(messages[0], "message_id", "") or ""),
        last_message_id=str(getattr(messages[-1], "message_id", "") or ""),
        oldest_created_at=min((value for value in created_values if value), default=""),
        newest_created_at=max((value for value in created_values if value), default=""),
        max_age_seconds=max_age_seconds,
        drain_mode="batch",
    )
    _publish_stale_skip_audit(row, queue_db_override=queue_db_override)


def _stale_skip_audit_row(
    *,
    mode: str,
    channel: str,
    queue_db_override: str,
    count: int,
    first_id: int,
    last_id: int,
    first_message_id: str,
    last_message_id: str,
    oldest_created_at: str,
    newest_created_at: str,
    max_age_seconds: float,
    drain_mode: str,
) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": str(mode),
        "channel": str(channel or ""),
        "queue_db_override": str(queue_db_override or ""),
        "action": "ack_without_execute",
        "reason": "stale_execution_intent",
        "count": int(count),
        "first_id": int(first_id),
        "last_id": int(last_id),
        "first_message_id": str(first_message_id or ""),
        "last_message_id": str(last_message_id or ""),
        "oldest_created_at": str(oldest_created_at or ""),
        "newest_created_at": str(newest_created_at or ""),
        "max_age_seconds": float(max_age_seconds),
        "drain_mode": str(drain_mode or "batch"),
        "trading_accuracy_policy": "stale paper intents are not executed as current fills",
    }


def _publish_stale_skip_audit(row: dict, *, queue_db_override: str) -> None:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    events_path = PROJECT_ROOT / "governance" / "events" / f"execution_lane_stale_skips_{day}.jsonl"
    latest_path = PROJECT_ROOT / "governance" / "health" / "execution_lane_stale_skip_latest.json"
    _append_jsonl(events_path, row)
    _write_json(latest_path, row)

    # Keep a channel-level audit without enqueueing one result per stale row.
    # If the queue is locked, still let the caller acknowledge stale intents.
    try:
        publish_execution_result(
            project_root=str(PROJECT_ROOT),
            payload={
                "timestamp_utc": row["timestamp_utc"],
                "mode": str(row.get("mode") or "paper"),
                "consumer": f"execution_lane_{str(row.get('mode') or 'paper')}",
                "intent_channel": row["channel"],
                "intent_message_id": row["last_message_id"],
                "intent_created_at": row["newest_created_at"],
                "result_status": "STALE_INTENT_SKIPPED",
                "result": row,
            },
            queue_db_override=queue_db_override,
        )
    except Exception as exc:
        row["result_publish_error"] = str(exc)
        row["result_publish_error_type"] = type(exc).__name__
        _append_jsonl(events_path, {**row, "event": "stale_skip_result_publish_error"})
        _write_json(latest_path, row)


def _stale_fast_drain_enabled() -> bool:
    return _env_flag("EXECUTION_LANE_STALE_FAST_DRAIN_ENABLED", "1")


def _stale_fast_drain_limit(default_limit: int) -> int:
    configured = _env_int("EXECUTION_LANE_STALE_FAST_DRAIN_LIMIT", max(int(default_limit), 5000))
    return max(configured, max(int(default_limit), 1))


def _stale_fast_drain_passes(default_passes: int = 1) -> int:
    return _env_int("EXECUTION_LANE_STALE_FAST_DRAIN_PASSES", default_passes)


def _drain_stale_prefix(
    *,
    queue: ChannelQueue,
    consumer: str,
    channel: str,
    mode: str,
    queue_db_override: str,
    batch_limit: int,
) -> int:
    if str(mode or "").strip().lower() != "paper" or not _stale_fast_drain_enabled():
        return 0
    max_age_seconds = _intent_max_age_seconds(mode)
    if max_age_seconds <= 0.0:
        return 0
    stale_before = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
    prefix = queue.stale_prefix(
        consumer=consumer,
        channel=channel,
        stale_before=stale_before,
        limit=_stale_fast_drain_limit(batch_limit),
    )
    count = int(prefix.get("count") or 0)
    if count <= 0:
        return 0

    row = _stale_skip_audit_row(
        mode=mode,
        channel=channel,
        queue_db_override=queue_db_override,
        count=count,
        first_id=int(prefix.get("first_id") or 0),
        last_id=int(prefix.get("last_id") or 0),
        first_message_id=str(prefix.get("first_message_id") or ""),
        last_message_id=str(prefix.get("last_message_id") or ""),
        oldest_created_at=str(prefix.get("oldest_created_at") or ""),
        newest_created_at=str(prefix.get("newest_created_at") or ""),
        max_age_seconds=max_age_seconds,
        drain_mode="stale_prefix_fast_drain",
    )
    _publish_stale_skip_audit(row, queue_db_override=queue_db_override)
    queue.ack_through(
        consumer=consumer,
        channel=channel,
        last_id=int(prefix.get("last_id") or 0),
        last_message_id=str(prefix.get("last_message_id") or ""),
    )
    return count


def _paper_execution_target_nice() -> int | None:
    raw = (
        _control_env_value("PAPER_EXECUTION_RUNTIME_NICE", "").strip()
        or _control_env_value("PAPER_SHADOW_RUNTIME_NICE", "").strip()
    )
    if not raw:
        return None
    try:
        return max(min(int(raw), 20), 0)
    except ValueError:
        return None


def _apply_paper_execution_nice() -> None:
    target = _paper_execution_target_nice()
    if target is None:
        return
    try:
        current = int(os.nice(0))
        if target > current:
            os.nice(min(target - current, 20))
    except Exception:
        return


def _paper_trade_lock_enabled() -> bool:
    lock_override = os.getenv("PAPER_TRADE_LOCK_PATH", "").strip()
    lock_path = Path(lock_override) if lock_override else PAPER_TRADE_LOCK_PATH
    return _env_flag("PAPER_TRADE_LOCK", "0") or lock_path.exists()


def _live_execution_enabled() -> bool:
    return _env_flag("TOP_BOT_ENABLE_LIVE_EXECUTION", "0") or _env_flag("EXECUTION_LANE_LIVE_ENABLED", "0")


def _paper_execution_paused_for_runtime() -> bool:
    _load_control_env()
    consumer_enabled = _control_env_value("PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED", "1").strip().lower()
    return (
        _env_flag("PAPER_EXECUTION_RUNTIME_PAUSED_FOR_PRESSURE", "0")
        or _env_flag("PAPER_EXECUTION_RUNTIME_PAUSED_FOR_LOCAL_STORAGE", "0")
        or _env_flag("PAPER_400_RAMP_BLOCKED_RUNTIME_PAUSE", "0")
        or consumer_enabled in {"0", "false", "no", "off"}
    )


def _build_trader(mode: str, broker: str) -> tuple[BaseTrader, bool, str]:
    trader = BaseTrader.from_env(mode=mode, broker=broker)
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
    _load_control_env()
    broker_runtime = BrokerRuntimeConfig.from_env()
    parser = argparse.ArgumentParser(description="Run standalone paper/live execution lane consumer.")
    parser.add_argument("--mode", choices=("paper", "live"), required=True)
    parser.add_argument("--broker", default="", choices=list(available_broker_names()))
    parser.add_argument("--once", action="store_true", help="Process one batch and exit.")
    parser.add_argument("--drain-stale-only", action="store_true", help="Only bulk-ack stale paper intents at the queue head, then exit.")
    parser.add_argument("--stale-drain-passes", type=int, default=_stale_fast_drain_passes(1))
    parser.add_argument("--limit", type=int, default=_env_int("EXECUTION_LANE_BATCH_LIMIT", 200))
    parser.add_argument("--poll-seconds", type=float, default=_env_float("EXECUTION_LANE_POLL_SECONDS", 2.0))
    parser.add_argument("--batch-sleep-seconds", type=float, default=_env_float("EXECUTION_LANE_BATCH_SLEEP_SECONDS", 0.0, minimum=0.0))
    parser.add_argument("--queue-db", default=os.getenv("BOT_CHANNEL_QUEUE_DB", ""))
    args = parser.parse_args()
    if args.mode == "paper":
        _apply_paper_execution_nice()
    broker = normalize_broker_name(
        args.broker
        or (
            broker_runtime.broker_for_role("paper")
            if args.mode == "paper"
            else broker_runtime.broker_for_role("execution")
        )
    )

    queue_path = queue_db_path(str(PROJECT_ROOT), args.queue_db)
    queue = ChannelQueue(queue_path)
    channel = _channel_for_mode(args.mode)
    consumer = f"execution_lane_{args.mode}"

    if args.drain_stale_only:
        drained = 0
        for _ in range(max(int(args.stale_drain_passes), 1)):
            batch_drained = _drain_stale_prefix(
                queue=queue,
                consumer=consumer,
                channel=channel,
                mode=args.mode,
                queue_db_override=args.queue_db,
                batch_limit=max(int(args.limit), 1),
            )
            drained += int(batch_drained)
            if batch_drained < max(int(args.limit), 1):
                break
        update_lane_health(
            project_root=str(PROJECT_ROOT),
            mode=args.mode,
            processed_count=drained,
            queue_channel=channel,
            queue_db_override=args.queue_db,
            auth_ok=True,
            auth_error="",
        )
        print(f"[ExecutionLane] stale_drain_only mode={args.mode} drained={drained}")
        return 0

    if args.mode == "live" and _paper_trade_lock_enabled():
        auth_error = "paper_trade_lock_active"
        print(f"[ExecutionLane] live blocked: {auth_error}")
        update_lane_health(
            project_root=str(PROJECT_ROOT),
            mode=args.mode,
            processed_count=0,
            queue_channel=channel,
            queue_db_override=args.queue_db,
            auth_ok=False,
            auth_error=auth_error,
        )
        return 4

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

    processed_total = 0
    skipped_stale_total = 0
    last_paper_reconcile_heartbeat = 0.0
    heartbeat_interval = max(float(_control_env_value("PAPER_RECONCILIATION_HEARTBEAT_SECONDS", "180") or 180.0), 30.0)
    last_lane_health_update = 0.0
    lane_health_interval = max(float(_control_env_value("EXECUTION_LANE_HEALTH_UPDATE_SECONDS", "60") or 60.0), 10.0)
    trader: BaseTrader | None = None
    auth_ok = True
    auth_error = ""

    if args.mode == "paper" and _paper_execution_paused_for_runtime():
        pause_reason = "paper_execution_paused_for_runtime_pressure"
        print(f"[ExecutionLane] paper paused: {pause_reason}")
        trader, auth_ok, auth_error = _build_trader(args.mode, broker)
        while _paper_execution_paused_for_runtime():
            if trader is not None and _env_flag("PAPER_RECONCILIATION_HEARTBEAT_WHEN_PAUSED", "1"):
                last_paper_reconcile_heartbeat = emit_paper_reconciliation_heartbeat(
                    project_root=str(PROJECT_ROOT),
                    trader=trader,
                    last_emit_monotonic=last_paper_reconcile_heartbeat,
                    min_interval_seconds=heartbeat_interval,
                    reason="execution_lane_paused",
                )
            if _lane_health_update_due(last_lane_health_update, lane_health_interval):
                update_lane_health(
                    project_root=str(PROJECT_ROOT),
                    mode=args.mode,
                    processed_count=processed_total,
                    queue_channel=channel,
                    queue_db_override=args.queue_db,
                    auth_ok=bool(auth_ok),
                    auth_error=pause_reason if auth_ok else (auth_error or pause_reason),
                )
                last_lane_health_update = time.monotonic()
            if args.once:
                return 5
            time.sleep(max(float(args.poll_seconds), 5.0))

    if trader is None:
        trader, auth_ok, auth_error = _build_trader(args.mode, broker)
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

    update_lane_health(
        project_root=str(PROJECT_ROOT),
        mode=args.mode,
        processed_count=processed_total,
        queue_channel=channel,
        queue_db_override=args.queue_db,
        auth_ok=auth_ok,
        auth_error=auth_error,
    )
    last_lane_health_update = time.monotonic()
    while True:
        _load_control_env()
        if args.mode == "paper" and _paper_execution_paused_for_runtime():
            if _lane_health_update_due(last_lane_health_update, lane_health_interval):
                update_lane_health(
                    project_root=str(PROJECT_ROOT),
                    mode=args.mode,
                    processed_count=processed_total,
                    queue_channel=channel,
                    queue_db_override=args.queue_db,
                    auth_ok=False,
                    auth_error="paper_execution_paused_for_runtime_pressure",
                )
                last_lane_health_update = time.monotonic()
            if args.once:
                return 5
            time.sleep(max(_env_float("EXECUTION_LANE_POLL_SECONDS", args.poll_seconds), 5.0))
            continue

        batch_limit = _env_int("EXECUTION_LANE_BATCH_LIMIT", args.limit)
        poll_seconds = _env_float("EXECUTION_LANE_POLL_SECONDS", args.poll_seconds)
        batch_sleep_seconds = _env_float("EXECUTION_LANE_BATCH_SLEEP_SECONDS", args.batch_sleep_seconds, minimum=0.0)
        message_sleep_seconds = _env_float("EXECUTION_LANE_MESSAGE_SLEEP_SECONDS", 0.0, minimum=0.0)
        fast_drained = _drain_stale_prefix(
            queue=queue,
            consumer=consumer,
            channel=channel,
            mode=args.mode,
            queue_db_override=args.queue_db,
            batch_limit=batch_limit,
        )
        if fast_drained > 0:
            skipped_stale_total += int(fast_drained)
            if args.once or _lane_health_update_due(last_lane_health_update, lane_health_interval):
                update_lane_health(
                    project_root=str(PROJECT_ROOT),
                    mode=args.mode,
                    processed_count=processed_total + skipped_stale_total,
                    queue_channel=channel,
                    queue_db_override=args.queue_db,
                    auth_ok=auth_ok,
                    auth_error=auth_error,
                )
                last_lane_health_update = time.monotonic()
            if args.once:
                return 0

        messages = queue.read_from_cursor(consumer=consumer, channel=channel, limit=batch_limit)
        if not messages:
            if args.mode == "paper":
                last_paper_reconcile_heartbeat = emit_paper_reconciliation_heartbeat(
                    project_root=str(PROJECT_ROOT),
                    trader=trader,
                    last_emit_monotonic=last_paper_reconcile_heartbeat,
                    min_interval_seconds=heartbeat_interval,
                    reason="execution_lane_idle",
                )
            if args.once or _lane_health_update_due(last_lane_health_update, lane_health_interval):
                update_lane_health(
                    project_root=str(PROJECT_ROOT),
                    mode=args.mode,
                    processed_count=processed_total + skipped_stale_total,
                    queue_channel=channel,
                    queue_db_override=args.queue_db,
                    auth_ok=auth_ok,
                    auth_error=auth_error,
                )
                last_lane_health_update = time.monotonic()
            if args.once:
                return 0
            time.sleep(poll_seconds)
            continue

        stale_messages = []
        stale_max_age_seconds = _intent_max_age_seconds(args.mode)
        for message in messages:
            stale, _age_seconds, max_age_seconds = _stale_intent_detail(args.mode, message)
            if stale:
                stale_messages.append(message)
                stale_max_age_seconds = max_age_seconds
                continue
            now_mono = time.monotonic()
            if _lane_health_update_due(last_lane_health_update, lane_health_interval, now_monotonic=now_mono):
                update_lane_health(
                    project_root=str(PROJECT_ROOT),
                    mode=args.mode,
                    processed_count=processed_total,
                    queue_channel=channel,
                    queue_db_override=args.queue_db,
                    auth_ok=auth_ok,
                    auth_error=auth_error,
                )
                last_lane_health_update = now_mono
            process_execution_intent(
                project_root=str(PROJECT_ROOT),
                trader=trader,
                mode=args.mode,
                message=message,
                queue_db_override=args.queue_db,
            )
            processed_total += 1
            if message_sleep_seconds > 0.0:
                time.sleep(message_sleep_seconds)
            now_mono = time.monotonic()
            if _lane_health_update_due(last_lane_health_update, lane_health_interval, now_monotonic=now_mono):
                update_lane_health(
                    project_root=str(PROJECT_ROOT),
                    mode=args.mode,
                    processed_count=processed_total,
                    queue_channel=channel,
                    queue_db_override=args.queue_db,
                    auth_ok=auth_ok,
                    auth_error=auth_error,
                )
                last_lane_health_update = now_mono

        if stale_messages:
            _emit_stale_skip_batch(
                mode=args.mode,
                messages=stale_messages,
                max_age_seconds=stale_max_age_seconds,
                queue_db_override=args.queue_db,
            )
            skipped_stale_total += len(stale_messages)

        queue.ack_messages(consumer=consumer, channel=channel, messages=messages)
        if args.mode == "paper":
            last_paper_reconcile_heartbeat = emit_paper_reconciliation_heartbeat(
                project_root=str(PROJECT_ROOT),
                trader=trader,
                last_emit_monotonic=last_paper_reconcile_heartbeat,
                min_interval_seconds=heartbeat_interval,
                reason="execution_lane_batch",
            )
        if args.once or _lane_health_update_due(last_lane_health_update, lane_health_interval):
            update_lane_health(
                project_root=str(PROJECT_ROOT),
                mode=args.mode,
                processed_count=processed_total + skipped_stale_total,
                queue_channel=channel,
                queue_db_override=args.queue_db,
                auth_ok=auth_ok,
                auth_error=auth_error,
            )
            last_lane_health_update = time.monotonic()

        if args.once:
            return 0
        cooldown_seconds = _cooldown_sleep_seconds(
            batch_sleep_seconds=batch_sleep_seconds,
            messages_read=len(messages),
            batch_limit=batch_limit,
        )
        if cooldown_seconds > 0:
            time.sleep(cooldown_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
