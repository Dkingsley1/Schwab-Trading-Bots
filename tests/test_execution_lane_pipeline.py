import json
from pathlib import Path

from core.base_trader import BaseTrader
from core.channel_queue import ChannelMessage, ChannelQueue, default_queue_db_path
from core.execution_lane_pipeline import (
    EXECUTION_INTENT_CHANNEL,
    EXECUTION_PROMOTED_CHANNEL,
    EXECUTION_PROMOTION_CHANNEL,
    EXECUTION_RESULT_CHANNEL,
    configure_trader_for_lane,
    evaluate_live_promotion,
    process_execution_intent,
    publish_execution_intent,
    update_lane_health,
)


def test_default_queue_db_path_prefers_local_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("BOT_CHANNEL_QUEUE_DB", raising=False)
    monkeypatch.setenv("BOT_CHANNEL_QUEUE_PREFER_LOCAL", "1")

    assert default_queue_db_path(tmp_path) == str(
        tmp_path / "local_fallback_storage" / "data" / "bot_channel_queue.sqlite3"
    )


def test_default_queue_db_path_respects_explicit_override(tmp_path: Path, monkeypatch) -> None:
    override = tmp_path / "custom" / "queue.sqlite3"
    monkeypatch.setenv("BOT_CHANNEL_QUEUE_DB", str(override))

    assert default_queue_db_path(tmp_path) == str(override)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_gates(project_root: Path, *, promote_ok: bool, quality_ok: bool) -> None:
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_gate_latest.json",
        {
            "promote_ok": bool(promote_ok),
            "coverage_ok": bool(promote_ok),
            "considered_bots": 5,
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "lane_promotion_gate_latest.json",
        {
            "promote_ok": True,
            "coverage_ok": True,
            "lanes": {
                "default": {
                    "promote_ok": True,
                    "coverage_ok": True,
                }
            },
        },
    )
    _write_json(
        project_root / "governance" / "health" / "promotion_quality_gate_latest.json",
        {
            "ok": bool(quality_ok),
            "failed_checks": ([] if quality_ok else ["promotion_gate_blocked"]),
        },
    )


def test_publish_execution_intent_enqueues_channel_message(tmp_path: Path) -> None:
    row = publish_execution_intent(
        project_root=str(tmp_path),
        payload={
            "symbol": "SPY",
            "action": "BUY",
            "quantity": 1.0,
            "model_score": 0.66,
            "threshold": 0.55,
            "strategy": "grand_master_bot",
            "metadata": {"snapshot_id": "snap-1"},
        },
    )

    queue = ChannelQueue(default_queue_db_path(tmp_path))
    messages = queue.read_from_cursor(consumer="pytest", channel=EXECUTION_INTENT_CHANNEL, limit=10)

    assert row["message_id"]
    assert len(messages) == 1
    assert messages[0].payload["symbol"] == "SPY"
    assert messages[0].payload["strategy"] == "grand_master_bot"


def test_evaluate_live_promotion_respects_existing_gate_truth(tmp_path: Path) -> None:
    _seed_gates(tmp_path, promote_ok=False, quality_ok=False)

    out = evaluate_live_promotion(
        project_root=str(tmp_path),
        intent={
            "intent_kind": "master",
            "symbol": "SPY",
            "action": "BUY",
            "metadata": {
                "allow_live_promotion": True,
                "runtime_lane": "default",
            },
        },
    )

    assert out["promote_ok"] is False
    assert "promotion_gate_blocked" in out["reasons"]
    assert "promotion_quality_gate_blocked" in out["reasons"]


def test_process_execution_intent_paper_emits_result_and_promoted_message(tmp_path: Path) -> None:
    _seed_gates(tmp_path, promote_ok=True, quality_ok=True)
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [],
        },
    )

    trader = BaseTrader("dummy_key", "dummy_secret", "https://127.0.0.1:8182", mode="paper")
    trader.project_root = str(tmp_path)
    trader.set_mode("paper")
    configure_trader_for_lane(trader, "paper")

    message = ChannelMessage(
        id=1,
        channel=EXECUTION_INTENT_CHANNEL,
        message_id="intent-1",
        parent_message_id="",
        run_id="run-1",
        iter_id="iter-1",
        source_path="",
        payload={
            "message_id": "intent-1",
            "intent_kind": "master",
            "symbol": "SPY",
            "action": "BUY",
            "quantity": 1.0,
            "model_score": 0.64,
            "threshold": 0.55,
            "features": {"last_price": 100.0, "spread_bps": 1.0},
            "gates": {"market_data_ok": True, "risk_limit_ok": True},
            "reasons": ["score_above_threshold"],
            "strategy": "grand_master_bot",
            "metadata": {
                "snapshot_id": "snap-1",
                "allow_live_promotion": True,
                "runtime_lane": "default",
            },
        },
        created_at="2026-03-31T20:00:00+00:00",
    )

    out = process_execution_intent(
        project_root=str(tmp_path),
        trader=trader,
        mode="paper",
        message=message,
    )

    queue = ChannelQueue(default_queue_db_path(tmp_path))
    result_rows = queue.read_from_cursor(consumer="pytest_results", channel=EXECUTION_RESULT_CHANNEL, limit=10)
    promotion_rows = queue.read_from_cursor(consumer="pytest_promotions", channel=EXECUTION_PROMOTION_CHANNEL, limit=10)
    promoted_rows = queue.read_from_cursor(consumer="pytest_live", channel=EXECUTION_PROMOTED_CHANNEL, limit=10)

    assert out["result"]["result_status"] == "PAPER_EXECUTED"
    assert len(result_rows) == 1
    assert len(promotion_rows) == 1
    assert promotion_rows[0].payload["promotion"]["promote_ok"] is True
    assert len(promoted_rows) == 1
    assert promoted_rows[0].payload["target_mode"] == "live"


def test_update_lane_health_marks_stale_consumer_with_backlog(tmp_path: Path, monkeypatch) -> None:
    queue = ChannelQueue(default_queue_db_path(tmp_path))
    queue.enqueue(
        channel=EXECUTION_INTENT_CHANNEL,
        payload={"message_id": "intent-1", "timestamp_utc": "2026-03-31T20:00:00+00:00"},
        message_id="intent-1",
    )
    with queue._connect() as conn:
        conn.execute(
            """
            INSERT INTO channel_consumer_state (consumer, channel, last_id, last_message_id, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("execution_lane_paper", EXECUTION_INTENT_CHANNEL, 0, "", "2026-03-31T20:00:00+00:00"),
        )
        conn.commit()

    monkeypatch.setenv("EXECUTION_LANE_STALE_AFTER_SECONDS", "60")
    update_lane_health(
        project_root=str(tmp_path),
        mode="paper",
        processed_count=0,
        queue_channel=EXECUTION_INTENT_CHANNEL,
    )

    payload = json.loads((tmp_path / "governance" / "health" / "execution_lane_paper_latest.json").read_text(encoding="utf-8"))
    assert payload["stale"] is True
    assert payload["pending_rows"] == 1
    assert payload["queue_oldest_age_seconds"] is not None
    assert payload["consumer_idle_seconds"] is not None


def test_update_lane_health_does_not_mark_stale_when_consumer_is_caught_up(tmp_path: Path, monkeypatch) -> None:
    queue = ChannelQueue(default_queue_db_path(tmp_path))
    queue.enqueue(
        channel=EXECUTION_INTENT_CHANNEL,
        payload={"message_id": "intent-1", "timestamp_utc": "2026-03-31T20:00:00+00:00"},
        message_id="intent-1",
    )
    queue.ack_through(
        consumer="execution_lane_paper",
        channel=EXECUTION_INTENT_CHANNEL,
        last_id=1,
        last_message_id="intent-1",
    )
    with queue._connect() as conn:
        conn.execute(
            """
            UPDATE channel_consumer_state
            SET updated_at=?
            WHERE consumer=? AND channel=?
            """,
            ("2026-03-31T20:00:00+00:00", "execution_lane_paper", EXECUTION_INTENT_CHANNEL),
        )
        conn.commit()

    monkeypatch.setenv("EXECUTION_LANE_STALE_AFTER_SECONDS", "60")
    update_lane_health(
        project_root=str(tmp_path),
        mode="paper",
        processed_count=1,
        queue_channel=EXECUTION_INTENT_CHANNEL,
    )

    payload = json.loads((tmp_path / "governance" / "health" / "execution_lane_paper_latest.json").read_text(encoding="utf-8"))
    assert payload["pending_rows"] == 0
    assert payload["stale"] is False
