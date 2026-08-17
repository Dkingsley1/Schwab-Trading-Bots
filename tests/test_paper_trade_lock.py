import json
import os
import sys
from pathlib import Path

import pytest

import scripts.run_execution_lane as run_execution_lane
from core.channel_queue import ChannelQueue
from core.execution_lane_pipeline import EXECUTION_INTENT_CHANNEL
from scripts.ops import lock_watchdog


@pytest.fixture(autouse=True)
def _use_local_execution_lane_root(monkeypatch):
    monkeypatch.setenv("BOT_LOGS_PREFER_EXTERNAL", "0")
    monkeypatch.delenv("EXECUTION_LANE_ROOT", raising=False)


def test_live_execution_lane_blocks_when_paper_trade_lock_active(tmp_path: Path, monkeypatch) -> None:
    queue_db = tmp_path / "data" / "queue.sqlite3"
    lock_path = tmp_path / "governance" / "health" / "PAPER_TRADE_LOCK.flag"

    monkeypatch.setattr(run_execution_lane, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_execution_lane, "PAPER_TRADE_LOCK_PATH", lock_path)
    monkeypatch.setenv("PAPER_TRADE_LOCK", "1")
    monkeypatch.setenv("BOT_CHANNEL_QUEUE_DB", str(queue_db))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_execution_lane.py", "--mode", "live", "--once", "--queue-db", str(queue_db)],
    )

    assert run_execution_lane.main() == 4

    health_path = tmp_path / "governance" / "health" / "execution_lane_live_latest.json"
    payload = json.loads(health_path.read_text(encoding="utf-8"))
    assert payload["auth_ok"] is False
    assert payload["auth_error"] == "paper_trade_lock_active"


def test_paper_execution_lane_pauses_when_runtime_guard_blocks_consumer(tmp_path: Path, monkeypatch) -> None:
    queue_db = tmp_path / "data" / "queue.sqlite3"
    heartbeat_calls: list[str] = []

    monkeypatch.setattr(run_execution_lane, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_execution_lane, "CONTROL_ENV_FILES", ())
    monkeypatch.setattr(run_execution_lane, "_build_trader", lambda mode, broker: (object(), True, ""))
    monkeypatch.setattr(
        run_execution_lane,
        "emit_paper_reconciliation_heartbeat",
        lambda **kwargs: heartbeat_calls.append(str(kwargs.get("reason") or "")) or 1.0,
    )
    monkeypatch.setenv("BOT_CHANNEL_QUEUE_DB", str(queue_db))
    monkeypatch.setenv("PAPER_EXECUTION_RUNTIME_PAUSED_FOR_PRESSURE", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_execution_lane.py", "--mode", "paper", "--once", "--queue-db", str(queue_db)],
    )

    assert run_execution_lane.main() == 5

    health_path = tmp_path / "governance" / "health" / "execution_lane_paper_latest.json"
    payload = json.loads(health_path.read_text(encoding="utf-8"))
    assert payload["auth_ok"] is True
    assert payload["auth_error"] == "paper_execution_paused_for_runtime_pressure"
    assert heartbeat_calls == ["execution_lane_paused"]


def test_execution_lane_loads_runtime_control_env(tmp_path: Path, monkeypatch) -> None:
    control_env = tmp_path / "runtime.env"
    control_env.write_text(
        "\n".join(
            [
                "EXECUTION_LANE_BATCH_LIMIT=50",
                "EXECUTION_LANE_BATCH_SLEEP_SECONDS=2.0",
                "EXECUTION_LANE_BACKLOG_SLEEP_SECONDS=5.0",
                "EXECUTION_LANE_HOST_LOAD_SOFT_CAP=6.0",
                "EXECUTION_LANE_HOST_LOAD_SLEEP_SECONDS=5.0",
                "EXECUTION_LANE_LIVE_MAX_INTENT_AGE_SECONDS=60",
                "EXECUTION_LANE_MESSAGE_SLEEP_SECONDS=0.04",
                "EXECUTION_LANE_PAPER_MAX_INTENT_AGE_SECONDS=900",
                "EXECUTION_LANE_POLL_SECONDS='4.0'",
                "PAPER_EXECUTION_RUNTIME_NICE=20",
                "IGNORED_KEY=1",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_execution_lane, "CONTROL_ENV_FILES", (control_env,))
    monkeypatch.delenv("EXECUTION_LANE_BATCH_LIMIT", raising=False)
    monkeypatch.delenv("EXECUTION_LANE_POLL_SECONDS", raising=False)
    monkeypatch.delenv("PAPER_EXECUTION_RUNTIME_NICE", raising=False)
    monkeypatch.delenv("IGNORED_KEY", raising=False)

    run_execution_lane._load_control_env()

    assert run_execution_lane._env_int("EXECUTION_LANE_BATCH_LIMIT", 200) == 50
    assert run_execution_lane._env_float("EXECUTION_LANE_BATCH_SLEEP_SECONDS", 0.0, minimum=0.0) == 2.0
    assert run_execution_lane._env_float("EXECUTION_LANE_BACKLOG_SLEEP_SECONDS", 0.0, minimum=0.0) == 5.0
    assert run_execution_lane._env_float("EXECUTION_LANE_HOST_LOAD_SOFT_CAP", 0.0, minimum=0.0) == 6.0
    assert run_execution_lane._env_float("EXECUTION_LANE_HOST_LOAD_SLEEP_SECONDS", 0.0, minimum=0.0) == 5.0
    assert run_execution_lane._env_float("EXECUTION_LANE_LIVE_MAX_INTENT_AGE_SECONDS", 0.0, minimum=0.0) == 60.0
    assert run_execution_lane._env_float("EXECUTION_LANE_MESSAGE_SLEEP_SECONDS", 0.0, minimum=0.0) == 0.04
    assert run_execution_lane._env_float("EXECUTION_LANE_PAPER_MAX_INTENT_AGE_SECONDS", 0.0, minimum=0.0) == 900.0
    assert run_execution_lane._env_float("EXECUTION_LANE_POLL_SECONDS", 2.0) == 4.0
    assert run_execution_lane._paper_execution_target_nice() == 20
    assert "IGNORED_KEY" not in os.environ


def test_paper_execution_lane_skips_stale_intents_without_trader_execution(tmp_path: Path, monkeypatch) -> None:
    queue_db = tmp_path / "data" / "queue.sqlite3"
    queue = ChannelQueue(queue_db)
    queue.enqueue(
        channel=EXECUTION_INTENT_CHANNEL,
        payload={
            "message_id": "stale-paper-intent",
            "timestamp_utc": "2000-01-01T00:00:00+00:00",
            "symbol": "SPY",
            "action": "BUY",
            "quantity": 1,
        },
        message_id="stale-paper-intent",
    )

    class _Trader:
        def execute_decision(self, **_kwargs):
            raise AssertionError("stale paper intent should not execute")

    monkeypatch.setattr(run_execution_lane, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_execution_lane, "CONTROL_ENV_FILES", ())
    monkeypatch.setattr(run_execution_lane, "_build_trader", lambda mode, broker: (_Trader(), True, ""))
    monkeypatch.setattr(run_execution_lane, "emit_paper_reconciliation_heartbeat", lambda **kwargs: 1.0)
    monkeypatch.setenv("BOT_CHANNEL_QUEUE_DB", str(queue_db))
    monkeypatch.setenv("EXECUTION_LANE_PAPER_MAX_INTENT_AGE_SECONDS", "60")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_execution_lane.py", "--mode", "paper", "--once", "--queue-db", str(queue_db)],
    )

    assert run_execution_lane.main() == 0

    state = queue.consumer_state(consumer="execution_lane_paper", channel=EXECUTION_INTENT_CHANNEL)
    latest = json.loads((tmp_path / "governance" / "health" / "execution_lane_stale_skip_latest.json").read_text(encoding="utf-8"))
    assert state["last_id"] == 1
    assert latest["action"] == "ack_without_execute"
    assert latest["reason"] == "stale_execution_intent"
    assert latest["count"] == 1
    assert latest["trading_accuracy_policy"] == "stale paper intents are not executed as current fills"


def test_paper_execution_lane_drain_stale_only_bulk_acks_prefix_without_trader(tmp_path: Path, monkeypatch) -> None:
    queue_db = tmp_path / "data" / "queue.sqlite3"
    queue = ChannelQueue(queue_db)
    for idx in range(2):
        message_id = f"stale-paper-intent-{idx}"
        queue.enqueue(
            channel=EXECUTION_INTENT_CHANNEL,
            payload={
                "message_id": message_id,
                "timestamp_utc": "2000-01-01T00:00:00+00:00",
                "symbol": "SPY",
                "action": "BUY",
                "quantity": 1,
            },
            message_id=message_id,
        )
    queue.enqueue(
        channel=EXECUTION_INTENT_CHANNEL,
        payload={
            "message_id": "fresh-paper-intent",
            "timestamp_utc": "2999-01-01T00:00:00+00:00",
            "symbol": "QQQ",
            "action": "BUY",
            "quantity": 1,
        },
        message_id="fresh-paper-intent",
    )

    def _unexpected_build_trader(*_args, **_kwargs):
        raise AssertionError("drain-stale-only should not build a trader")

    monkeypatch.setattr(run_execution_lane, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_execution_lane, "CONTROL_ENV_FILES", ())
    monkeypatch.setattr(run_execution_lane, "_build_trader", _unexpected_build_trader)
    monkeypatch.setenv("BOT_CHANNEL_QUEUE_DB", str(queue_db))
    monkeypatch.setenv("BOT_LOGS_PREFER_EXTERNAL", "0")
    monkeypatch.setenv("EXECUTION_LANE_PAPER_MAX_INTENT_AGE_SECONDS", "60")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_execution_lane.py",
            "--mode",
            "paper",
            "--drain-stale-only",
            "--limit",
            "10",
            "--queue-db",
            str(queue_db),
        ],
    )

    assert run_execution_lane.main() == 0

    state = queue.consumer_state(consumer="execution_lane_paper", channel=EXECUTION_INTENT_CHANNEL)
    latest = json.loads((tmp_path / "governance" / "health" / "execution_lane_stale_skip_latest.json").read_text(encoding="utf-8"))
    assert state["last_id"] == 2
    assert latest["count"] == 2
    assert latest["drain_mode"] == "stale_prefix_fast_drain"
    assert latest["last_message_id"] == "stale-paper-intent-1"


def test_lock_watchdog_recognizes_paper_policy_locks(tmp_path: Path) -> None:
    lock_path = tmp_path / "governance" / "locks" / "paper_trade.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("policy=live_data_paper_trade_only\n", encoding="utf-8")

    assert lock_watchdog._is_policy_lock(lock_path, lock_path.read_text(encoding="utf-8")) is True
