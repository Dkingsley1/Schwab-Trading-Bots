import json
import sys
from pathlib import Path

import scripts.run_execution_lane as run_execution_lane
from scripts.ops import lock_watchdog


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


def test_lock_watchdog_recognizes_paper_policy_locks(tmp_path: Path) -> None:
    lock_path = tmp_path / "governance" / "locks" / "paper_trade.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("policy=live_data_paper_trade_only\n", encoding="utf-8")

    assert lock_watchdog._is_policy_lock(lock_path, lock_path.read_text(encoding="utf-8")) is True
