import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PAPER_TRADE_LOCK_PATH = PROJECT_ROOT / "governance" / "health" / "PAPER_TRADE_LOCK.flag"
EXECUTION_LANE_LOCK_ROOT = PROJECT_ROOT / "governance" / "locks"
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
from core.cpu_workload_policy import (
    load_cpu_workload_policy,
    nice_target_for_class,
    taskpolicy_executable,
)
from core.brokers import (
    BrokerRuntimeConfig,
    available_broker_names,
    normalize_broker_name,
)
from core.channel_queue import ChannelQueue
from core.system_role_contracts import evaluate_component_action
from core.execution_lane_pipeline import (
    EXECUTION_INTENT_CHANNEL,
    EXECUTION_PROMOTED_CHANNEL,
    configure_trader_for_lane,
    emit_paper_reconciliation_heartbeat,
    process_execution_intent,
    publish_execution_consumer_failure,
    publish_execution_replay_suppressed,
    publish_execution_result,
    queue_db_path,
    update_lane_health,
)

CPU_WORKLOAD_POLICY = load_cpu_workload_policy()


def _env_flag(name: str, default: str = "0") -> bool:
    return _control_env_value(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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


def _acquire_execution_lane_lock(mode: str):
    lock_root = Path(
        os.getenv("EXECUTION_LANE_LOCK_ROOT", str(EXECUTION_LANE_LOCK_ROOT))
    ).expanduser()
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"execution_lane_{mode}.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    waiting_reported = False
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not waiting_reported:
                print(
                    f"[ExecutionLaneLock] standby mode={mode} "
                    f"lock_path={lock_path} reason=active_consumer_exists"
                )
                waiting_reported = True
            time.sleep(2.0)
    handle.seek(0)
    handle.truncate()
    handle.write(
        f"pid={os.getpid()} mode={mode} "
        f"started_utc={datetime.now(timezone.utc).isoformat()}\n"
    )
    handle.flush()
    print(
        f"[ExecutionLaneLock] acquired mode={mode} lock_path={lock_path} pid={os.getpid()}"
    )
    return handle


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
        return _env_float(
            "EXECUTION_LANE_PAPER_MAX_INTENT_AGE_SECONDS", 900.0, minimum=0.0
        )
    return _env_float("EXECUTION_LANE_LIVE_MAX_INTENT_AGE_SECONDS", 60.0, minimum=0.0)


def _stale_intent_detail(
    mode: str, message: object
) -> tuple[bool, float | None, float, str]:
    normalized_mode = str(mode or "").strip().lower()
    max_age_seconds = _intent_max_age_seconds(mode)
    if max_age_seconds <= 0.0:
        return False, None, max_age_seconds, "freshness_check_disabled"
    created_at = _message_created_at(message)
    if created_at is None:
        return (
            normalized_mode == "live",
            None,
            max_age_seconds,
            (
                "live_intent_created_at_missing"
                if normalized_mode == "live"
                else "created_at_missing_paper_compatible"
            ),
        )
    signed_age_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
    max_future_skew = _env_float(
        "EXECUTION_LANE_LIVE_MAX_FUTURE_SKEW_SECONDS",
        2.0,
        minimum=0.0,
    )
    if normalized_mode == "live" and signed_age_seconds < -max_future_skew:
        return (
            True,
            round(signed_age_seconds, 3),
            max_age_seconds,
            "live_intent_created_at_in_future",
        )
    age_seconds = max(signed_age_seconds, 0.0)
    stale = age_seconds > max_age_seconds
    return (
        stale,
        round(age_seconds, 3),
        max_age_seconds,
        "live_intent_expired" if stale else "fresh",
    )


def _cooldown_sleep_seconds(
    *, batch_sleep_seconds: float, messages_read: int, batch_limit: int
) -> float:
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
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    freshness_failures: list[str] | None = None,
) -> None:
    if not messages:
        return
    created_values = [
        str(getattr(message, "created_at", "") or "") for message in messages
    ]
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
        freshness_failures=freshness_failures,
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
    freshness_failures: list[str] | None = None,
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
        "freshness_failures": sorted(
            {str(item) for item in (freshness_failures or []) if str(item)}
        ),
        "trading_accuracy_policy": "stale paper intents are not executed as current fills",
    }


def _publish_stale_skip_audit(row: dict, *, queue_db_override: str) -> None:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    events_path = (
        PROJECT_ROOT
        / "governance"
        / "events"
        / f"execution_lane_stale_skips_{day}.jsonl"
    )
    latest_path = (
        PROJECT_ROOT / "governance" / "health" / "execution_lane_stale_skip_latest.json"
    )
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


def _process_execution_message_with_outcome(
    *,
    trader,
    mode: str,
    message,
    queue_db_override: str,
) -> dict[str, object]:
    """Contain one poison message so the long-running lane stays available."""

    try:
        published = process_execution_intent(
            project_root=str(PROJECT_ROOT),
            trader=trader,
            mode=mode,
            message=message,
            queue_db_override=queue_db_override,
        )
        result_payload = (
            published.get("result")
            if isinstance(published, dict)
            and isinstance(published.get("result"), dict)
            else {}
        )
        return {
            "handled": True,
            "durable_outcome": True,
            "result_status": str(result_payload.get("result_status") or ""),
            "result_message_id": str(result_payload.get("message_id") or ""),
            "recovery_action": "ack_after_processing_claim_finalized",
        }
    except Exception as exc:
        recovery_action = (
            "dead_letter_ack_reconcile_before_next_intent"
            if str(mode).strip().lower() == "live"
            else "dead_letter_ack_and_continue"
        )
        failure = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": "execution_lane_consumer_exception",
            "mode": str(mode),
            "message_id": str(getattr(message, "message_id", "") or ""),
            "channel": str(getattr(message, "channel", "") or ""),
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "recovery_action": recovery_action,
        }
        dead_letter_published = False
        dead_letter_message_id = ""
        try:
            dead_letter = publish_execution_consumer_failure(
                project_root=str(PROJECT_ROOT),
                mode=mode,
                message=message,
                error=exc,
                queue_db_override=queue_db_override,
            )
            dead_letter_message_id = str(dead_letter.get("message_id") or "")
            dead_letter_published = True
            failure["dead_letter_published"] = True
        except Exception as publish_exc:
            failure["dead_letter_published"] = False
            failure["dead_letter_publish_error_type"] = type(publish_exc).__name__
            failure["dead_letter_publish_error"] = str(publish_exc)[:1000]
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        audit_published = False
        try:
            _append_jsonl(
                PROJECT_ROOT
                / "governance"
                / "events"
                / f"execution_lane_consumer_failures_{day}.jsonl",
                failure,
            )
            audit_published = True
        except Exception as audit_exc:
            failure["local_audit_published"] = False
            failure["local_audit_error_type"] = type(audit_exc).__name__
            failure["local_audit_error"] = str(audit_exc)[:1000]
        print(
            f"[ExecutionLaneError] mode={mode} "
            f"message_id={failure['message_id'] or 'unknown'} "
            f"error_type={failure['error_type']} action={recovery_action}"
        )
        return {
            "handled": False,
            "durable_outcome": bool(dead_letter_published or audit_published),
            "result_status": f"{str(mode).strip().upper()}_CONSUMER_ERROR_BLOCKED",
            "result_message_id": dead_letter_message_id,
            "recovery_action": recovery_action,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }


def _process_execution_message_safely(
    *,
    trader,
    mode: str,
    message,
    queue_db_override: str,
) -> bool:
    """Compatibility wrapper for callers that only need handled/not-handled."""

    outcome = _process_execution_message_with_outcome(
        trader=trader,
        mode=mode,
        message=message,
        queue_db_override=queue_db_override,
    )
    return bool(outcome.get("handled", False))


def _stale_fast_drain_enabled() -> bool:
    return _env_flag("EXECUTION_LANE_STALE_FAST_DRAIN_ENABLED", "1")


def _stale_fast_drain_limit(default_limit: int) -> int:
    configured = _env_int(
        "EXECUTION_LANE_STALE_FAST_DRAIN_LIMIT", max(int(default_limit), 5000)
    )
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
        requested = max(min(int(raw), 20), 0)
    except ValueError:
        return None
    if _control_env_value("BOT_CPU_WORKLOAD_POLICY_LOCKED", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return nice_target_for_class(CPU_WORKLOAD_POLICY, "paper_execution", requested)
    return requested


def _apply_paper_execution_nice() -> dict[str, object]:
    target = _paper_execution_target_nice()
    if target is None:
        return {"applied": False, "reason": "no_target"}
    try:
        current = int(os.nice(0))
        if target > current:
            os.nice(min(target - current, 20))
        observed = int(os.nice(0))
    except Exception as exc:
        return {
            "applied": False,
            "reason": f"nice_failed:{exc.__class__.__name__}",
            "target_nice": target,
        }
    locked = _control_env_value(
        "BOT_CPU_WORKLOAD_POLICY_LOCKED", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    taskpolicy_ok: bool | None = None
    taskpolicy_reason = "not_requested"
    if (
        locked
        and sys.platform == "darwin"
        and _control_env_value("BOT_CPU_TASKPOLICY_SELF_HEAL", "1").strip().lower()
        in {"1", "true", "yes", "on"}
    ):
        taskpolicy_path = taskpolicy_executable()
        if not taskpolicy_path:
            taskpolicy_reason = "taskpolicy_unavailable"
        else:
            try:
                proc = subprocess.run(
                    [taskpolicy_path, "-B", "-p", str(os.getpid())],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError as exc:
                taskpolicy_ok = False
                taskpolicy_reason = f"taskpolicy_failed:{exc.__class__.__name__}"
            else:
                taskpolicy_ok = proc.returncode == 0
                taskpolicy_reason = (
                    "darwin_background_removed"
                    if taskpolicy_ok
                    else "taskpolicy_nonzero"
                )
    return {
        "applied": observed != current,
        "current_nice": current,
        "target_nice": target,
        "observed_nice": observed,
        "managed_restart_required": bool(locked and observed > target),
        "hard_affinity_claimed": False,
        "taskpolicy_ok": taskpolicy_ok,
        "taskpolicy_reason": taskpolicy_reason,
    }


def _paper_trade_lock_enabled() -> bool:
    lock_override = os.getenv("PAPER_TRADE_LOCK_PATH", "").strip()
    lock_path = Path(lock_override) if lock_override else PAPER_TRADE_LOCK_PATH
    return _env_flag("PAPER_TRADE_LOCK", "0") or lock_path.exists()


def _live_execution_enabled() -> bool:
    return _env_flag("TOP_BOT_ENABLE_LIVE_EXECUTION", "0") or _env_flag(
        "EXECUTION_LANE_LIVE_ENABLED", "0"
    )


def _paper_execution_paused_for_runtime() -> bool:
    _load_control_env()
    consumer_enabled = (
        _control_env_value("PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED", "1")
        .strip()
        .lower()
    )
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
    parser = argparse.ArgumentParser(
        description="Run standalone paper/live execution lane consumer."
    )
    parser.add_argument("--mode", choices=("paper", "live"), required=True)
    parser.add_argument("--broker", default="", choices=list(available_broker_names()))
    parser.add_argument(
        "--once", action="store_true", help="Process one batch and exit."
    )
    parser.add_argument(
        "--drain-stale-only",
        action="store_true",
        help="Only bulk-ack stale paper intents at the queue head, then exit.",
    )
    parser.add_argument(
        "--stale-drain-passes", type=int, default=_stale_fast_drain_passes(1)
    )
    parser.add_argument(
        "--limit", type=int, default=_env_int("EXECUTION_LANE_BATCH_LIMIT", 200)
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=_env_float("EXECUTION_LANE_POLL_SECONDS", 2.0),
    )
    parser.add_argument(
        "--batch-sleep-seconds",
        type=float,
        default=_env_float("EXECUTION_LANE_BATCH_SLEEP_SECONDS", 0.0, minimum=0.0),
    )
    parser.add_argument("--queue-db", default=os.getenv("BOT_CHANNEL_QUEUE_DB", ""))
    args = parser.parse_args()
    _execution_lane_lock = _acquire_execution_lane_lock(args.mode)
    if args.mode == "paper":
        cpu_result = _apply_paper_execution_nice()
        print(
            "[RuntimeCPU] class=paper_execution "
            f"current={cpu_result.get('current_nice', '')} "
            f"target={cpu_result.get('target_nice', '')} "
            f"restart_required={int(bool(cpu_result.get('managed_restart_required', False)))} "
            "hard_affinity=0"
        )
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

    role_contract_path = PROJECT_ROOT / "config" / "system_role_contracts_v1.json"
    paper_runtime_pause_active = bool(
        args.mode == "paper" and _paper_execution_paused_for_runtime()
    )
    if role_contract_path.is_file() and not paper_runtime_pause_active:
        component_id = (
            "paper_execution_gateway"
            if args.mode == "paper"
            else "live_execution_gateway"
        )
        action = "paper_submit" if args.mode == "paper" else "live_submit"
        state_domain = (
            "paper_order_submission"
            if args.mode == "paper"
            else "live_order_submission"
        )
        authority = evaluate_component_action(
            PROJECT_ROOT,
            component_id=component_id,
            action=action,
            state_domain=state_domain,
        )
        if not bool(authority.get("ok", False)):
            auth_error = "system_role_authority_denied:" + ",".join(
                str(item) for item in authority.get("blockers", []) if str(item)
            )
            print(f"[ExecutionLane] {args.mode} blocked: {auth_error}")
            update_lane_health(
                project_root=str(PROJECT_ROOT),
                mode=args.mode,
                processed_count=0,
                queue_channel=channel,
                queue_db_override=args.queue_db,
                auth_ok=False,
                auth_error=auth_error,
            )
            return 6

    processed_total = 0
    skipped_stale_total = 0
    last_paper_reconcile_heartbeat = 0.0
    last_live_order_reconcile_heartbeat = 0.0
    heartbeat_interval = max(
        float(
            _control_env_value("PAPER_RECONCILIATION_HEARTBEAT_SECONDS", "180") or 180.0
        ),
        30.0,
    )
    live_order_reconcile_interval = max(
        float(
            _control_env_value("LIVE_ORDER_RECONCILIATION_HEARTBEAT_SECONDS", "5")
            or 5.0
        ),
        1.0,
    )
    last_lane_health_update = 0.0
    lane_health_interval = max(
        float(_control_env_value("EXECUTION_LANE_HEALTH_UPDATE_SECONDS", "60") or 60.0),
        10.0,
    )
    trader: BaseTrader | None = None
    auth_ok = True
    auth_error = ""
    paper_runtime_hold_reported = False

    if args.mode == "paper" and _paper_execution_paused_for_runtime():
        pause_reason = "paper_execution_paused_for_runtime_pressure"
        print(f"[ExecutionLane] paper paused: {pause_reason}")
        trader, auth_ok, auth_error = _build_trader(args.mode, broker)
        while _paper_execution_paused_for_runtime():
            if trader is not None and _env_flag(
                "PAPER_RECONCILIATION_HEARTBEAT_WHEN_PAUSED", "1"
            ):
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
                    auth_error=(auth_error if not auth_ok else ""),
                    hold_reason=pause_reason,
                )
                paper_runtime_hold_reported = True
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

    if args.mode == "live":
        while True:
            live_reconciliation = trader.reconcile_durable_live_orders(
                interrupted_stale_seconds=max(
                    _env_float(
                        "LIVE_ORDER_INTERRUPTED_STALE_SECONDS", 5.0, minimum=0.0
                    ),
                    0.0,
                ),
                full_account_scan=True,
            )
            last_live_order_reconcile_heartbeat = time.monotonic()
            if bool(live_reconciliation.get("ok", False)):
                auth_error = ""
                break
            reconciliation_error = "live_order_reconciliation_blocked:" + ",".join(
                str(item)
                for item in live_reconciliation.get("blockers", [])[:5]
                if str(item)
            )
            update_lane_health(
                project_root=str(PROJECT_ROOT),
                mode=args.mode,
                processed_count=processed_total,
                queue_channel=channel,
                queue_db_override=args.queue_db,
                auth_ok=auth_ok,
                auth_error=reconciliation_error,
            )
            print(f"[ExecutionLane] live reconcile blocked: {reconciliation_error}")
            if args.once:
                return 7
            time.sleep(max(float(args.poll_seconds), live_order_reconcile_interval))

    update_lane_health(
        project_root=str(PROJECT_ROOT),
        mode=args.mode,
        processed_count=processed_total,
        queue_channel=channel,
        queue_db_override=args.queue_db,
        auth_ok=auth_ok,
        auth_error=auth_error,
    )
    paper_runtime_hold_reported = False
    last_lane_health_update = time.monotonic()
    while True:
        _load_control_env()
        if (
            args.mode == "live"
            and (time.monotonic() - last_live_order_reconcile_heartbeat)
            >= live_order_reconcile_interval
        ):
            live_reconciliation = trader.reconcile_durable_live_orders(
                interrupted_stale_seconds=max(
                    _env_float(
                        "LIVE_ORDER_INTERRUPTED_STALE_SECONDS", 5.0, minimum=0.0
                    ),
                    0.0,
                ),
                full_account_scan=True,
            )
            last_live_order_reconcile_heartbeat = time.monotonic()
            if not bool(live_reconciliation.get("ok", False)):
                reconciliation_error = "live_order_reconciliation_blocked:" + ",".join(
                    str(item)
                    for item in live_reconciliation.get("blockers", [])[:5]
                    if str(item)
                )
                update_lane_health(
                    project_root=str(PROJECT_ROOT),
                    mode=args.mode,
                    processed_count=processed_total,
                    queue_channel=channel,
                    queue_db_override=args.queue_db,
                    auth_ok=auth_ok,
                    auth_error=reconciliation_error,
                )
                if args.once:
                    return 7
                time.sleep(
                    max(
                        _env_float("EXECUTION_LANE_POLL_SECONDS", args.poll_seconds),
                        1.0,
                    )
                )
                continue
        if args.mode == "paper" and _paper_execution_paused_for_runtime():
            paper_runtime_hold_reported = True
            if _lane_health_update_due(last_lane_health_update, lane_health_interval):
                update_lane_health(
                    project_root=str(PROJECT_ROOT),
                    mode=args.mode,
                    processed_count=processed_total,
                    queue_channel=channel,
                    queue_db_override=args.queue_db,
                    auth_ok=bool(auth_ok),
                    auth_error=(auth_error if not auth_ok else ""),
                    hold_reason="paper_execution_paused_for_runtime_pressure",
                )
                last_lane_health_update = time.monotonic()
            if args.once:
                return 5
            time.sleep(
                max(_env_float("EXECUTION_LANE_POLL_SECONDS", args.poll_seconds), 5.0)
            )
            continue

        if args.mode == "paper" and paper_runtime_hold_reported:
            update_lane_health(
                project_root=str(PROJECT_ROOT),
                mode=args.mode,
                processed_count=processed_total + skipped_stale_total,
                queue_channel=channel,
                queue_db_override=args.queue_db,
                auth_ok=bool(auth_ok),
                auth_error=(auth_error if not auth_ok else ""),
            )
            paper_runtime_hold_reported = False
            last_lane_health_update = time.monotonic()

        batch_limit = _env_int("EXECUTION_LANE_BATCH_LIMIT", args.limit)
        poll_seconds = _env_float("EXECUTION_LANE_POLL_SECONDS", args.poll_seconds)
        batch_sleep_seconds = _env_float(
            "EXECUTION_LANE_BATCH_SLEEP_SECONDS", args.batch_sleep_seconds, minimum=0.0
        )
        message_sleep_seconds = _env_float(
            "EXECUTION_LANE_MESSAGE_SLEEP_SECONDS", 0.0, minimum=0.0
        )
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
            if args.once or _lane_health_update_due(
                last_lane_health_update, lane_health_interval
            ):
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

        messages = queue.read_from_cursor(
            consumer=consumer, channel=channel, limit=batch_limit
        )
        if not messages:
            if args.mode == "paper":
                last_paper_reconcile_heartbeat = emit_paper_reconciliation_heartbeat(
                    project_root=str(PROJECT_ROOT),
                    trader=trader,
                    last_emit_monotonic=last_paper_reconcile_heartbeat,
                    min_interval_seconds=heartbeat_interval,
                    reason="execution_lane_idle",
                )
            if args.once or _lane_health_update_due(
                last_lane_health_update, lane_health_interval
            ):
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
        acknowledged_messages = []
        live_reconcile_blocked = False
        processing_claim_blocked = False
        stale_freshness_failures: set[str] = set()
        stale_max_age_seconds = _intent_max_age_seconds(args.mode)
        for message in messages:
            stale, _age_seconds, max_age_seconds, freshness_failure = (
                _stale_intent_detail(args.mode, message)
            )
            if stale:
                stale_messages.append(message)
                acknowledged_messages.append(message)
                stale_freshness_failures.add(freshness_failure)
                stale_max_age_seconds = max_age_seconds
                continue
            now_mono = time.monotonic()
            if _lane_health_update_due(
                last_lane_health_update, lane_health_interval, now_monotonic=now_mono
            ):
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
            try:
                processing_claim = queue.claim_message_processing(
                    consumer=consumer,
                    channel=channel,
                    message=message,
                )
            except Exception as claim_exc:
                auth_error = (
                    "execution_processing_claim_failed:"
                    f"{type(claim_exc).__name__}:{str(claim_exc)[:500]}"
                )
                processing_claim_blocked = True
                break

            if not bool(processing_claim.get("claimed", False)):
                prior_state = str(processing_claim.get("state") or "").lower()
                if args.mode == "live" and prior_state in {
                    "processing",
                    "outcome_ambiguous",
                }:
                    live_reconciliation = trader.reconcile_durable_live_orders(
                        interrupted_stale_seconds=0.0,
                        full_account_scan=True,
                    )
                    last_live_order_reconcile_heartbeat = time.monotonic()
                    if not bool(live_reconciliation.get("ok", False)):
                        reconciliation_blockers = [
                            str(item)
                            for item in live_reconciliation.get("blockers", [])[:5]
                            if str(item)
                        ]
                        auth_error = (
                            "live_order_reconciliation_blocked_before_replay_ack"
                        )
                        if reconciliation_blockers:
                            auth_error += ":" + ",".join(reconciliation_blockers)
                        live_reconcile_blocked = True
                        break
                    auth_error = ""
                try:
                    replay_audit = publish_execution_replay_suppressed(
                        project_root=str(PROJECT_ROOT),
                        mode=args.mode,
                        message=message,
                        prior_claim=processing_claim,
                        queue_db_override=args.queue_db,
                    )
                    if prior_state in {"processing", "outcome_ambiguous"}:
                        queue.finalize_message_processing(
                            consumer=consumer,
                            channel=channel,
                            message=message,
                            state="replay_suppressed",
                            outcome_status=str(
                                replay_audit.get("result_status")
                                or f"{args.mode.upper()}_REPLAY_SUPPRESSED"
                            ),
                            outcome_message_id=str(
                                replay_audit.get("message_id") or ""
                            ),
                            details={
                                "prior_state": prior_state,
                                "policy": "never_reexecute_an_already_claimed_intent",
                            },
                        )
                except Exception as replay_exc:
                    auth_error = (
                        "execution_replay_audit_failed:"
                        f"{type(replay_exc).__name__}:{str(replay_exc)[:500]}"
                    )
                    processing_claim_blocked = True
                    break
                acknowledged_messages.append(message)
                processed_total += 1
                continue

            outcome = _process_execution_message_with_outcome(
                trader=trader,
                mode=args.mode,
                message=message,
                queue_db_override=args.queue_db,
            )
            processed_total += 1
            durable_outcome = bool(outcome.get("durable_outcome", False))
            handled = bool(outcome.get("handled", False))
            if not durable_outcome:
                try:
                    queue.finalize_message_processing(
                        consumer=consumer,
                        channel=channel,
                        message=message,
                        state="outcome_ambiguous",
                        outcome_status=str(outcome.get("result_status") or ""),
                        outcome_message_id=str(
                            outcome.get("result_message_id") or ""
                        ),
                        details={
                            "reason": "execution_outcome_has_no_durable_audit",
                            "handled": handled,
                        },
                    )
                except Exception:
                    pass
                auth_error = "execution_outcome_audit_unavailable"
                processing_claim_blocked = True
                break
            if not handled and args.mode == "live":
                live_reconciliation = trader.reconcile_durable_live_orders(
                    interrupted_stale_seconds=0.0,
                    full_account_scan=True,
                )
                last_live_order_reconcile_heartbeat = time.monotonic()
                if not bool(live_reconciliation.get("ok", False)):
                    reconciliation_blockers = [
                        str(item)
                        for item in live_reconciliation.get("blockers", [])[:5]
                        if str(item)
                    ]
                    auth_error = (
                        "live_order_reconciliation_blocked_after_consumer_error"
                    )
                    if reconciliation_blockers:
                        auth_error += ":" + ",".join(reconciliation_blockers)
                    try:
                        queue.finalize_message_processing(
                            consumer=consumer,
                            channel=channel,
                            message=message,
                            state="outcome_ambiguous",
                            outcome_status=str(outcome.get("result_status") or ""),
                            outcome_message_id=str(
                                outcome.get("result_message_id") or ""
                            ),
                            details={
                                "reason": "live_reconciliation_blocked_after_consumer_error",
                                "blockers": reconciliation_blockers,
                            },
                        )
                    except Exception:
                        pass
                    live_reconcile_blocked = True
                    break
                auth_error = ""
            try:
                queue.finalize_message_processing(
                    consumer=consumer,
                    channel=channel,
                    message=message,
                    state=("completed" if handled else "dead_lettered"),
                    outcome_status=str(outcome.get("result_status") or ""),
                    outcome_message_id=str(outcome.get("result_message_id") or ""),
                    details={
                        "handled": handled,
                        "recovery_action": str(
                            outcome.get("recovery_action") or ""
                        ),
                    },
                )
            except Exception as finalize_exc:
                auth_error = (
                    "execution_processing_finalize_failed:"
                    f"{type(finalize_exc).__name__}:{str(finalize_exc)[:500]}"
                )
                processing_claim_blocked = True
                break
            acknowledged_messages.append(message)
            if message_sleep_seconds > 0.0:
                time.sleep(message_sleep_seconds)
            now_mono = time.monotonic()
            if _lane_health_update_due(
                last_lane_health_update, lane_health_interval, now_monotonic=now_mono
            ):
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
                freshness_failures=sorted(stale_freshness_failures),
            )
            skipped_stale_total += len(stale_messages)

        queue.ack_messages(
            consumer=consumer,
            channel=channel,
            messages=acknowledged_messages,
        )
        if args.mode == "paper":
            last_paper_reconcile_heartbeat = emit_paper_reconciliation_heartbeat(
                project_root=str(PROJECT_ROOT),
                trader=trader,
                last_emit_monotonic=last_paper_reconcile_heartbeat,
                min_interval_seconds=heartbeat_interval,
                reason="execution_lane_batch",
            )
        if args.once or _lane_health_update_due(
            last_lane_health_update, lane_health_interval
        ):
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

        if live_reconcile_blocked:
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
                return 7
            time.sleep(max(poll_seconds, live_order_reconcile_interval))
            continue

        if processing_claim_blocked:
            update_lane_health(
                project_root=str(PROJECT_ROOT),
                mode=args.mode,
                processed_count=processed_total + skipped_stale_total,
                queue_channel=channel,
                queue_db_override=args.queue_db,
                auth_ok=False,
                auth_error=auth_error,
            )
            last_lane_health_update = time.monotonic()
            if args.once:
                return 8
            time.sleep(max(poll_seconds, 2.0))
            continue

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
