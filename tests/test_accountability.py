from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import accountability


def test_thin_low_signal_payloads_dedupes_observe_only_shadow_rows(monkeypatch) -> None:
    monkeypatch.setenv("LOW_SIGNAL_LOG_THINNING_ENABLED", "1")
    monkeypatch.setenv("LOW_SIGNAL_DECISION_WINDOW_SECONDS", "60")
    monkeypatch.setattr(accountability.time, "time", lambda: 1_000.0)
    accountability._LOW_SIGNAL_RECENT.clear()

    path = "/tmp/decision_explanations/shadow_aggressive_equities/decision_explanations_20260326.jsonl"
    rows = [
        {
            "status": "DATA_ONLY_BLOCKED",
            "symbol": "SPY",
            "action": "BUY",
            "strategy": "brain_refinery_v10_seasonal",
            "reasons": ["score_above_threshold"],
            "safety": {"market_data_only": True, "execution_enabled": False},
        },
        {
            "status": "DATA_ONLY_BLOCKED",
            "symbol": "SPY",
            "action": "BUY",
            "strategy": "brain_refinery_v10_seasonal",
            "reasons": ["score_above_threshold"],
            "safety": {"market_data_only": True, "execution_enabled": False},
        },
    ]

    kept = accountability._thin_low_signal_payloads(path, rows)

    assert len(kept) == 1


def test_thin_low_signal_payloads_keeps_execution_relevant_data_only_rows(monkeypatch) -> None:
    monkeypatch.setenv("LOW_SIGNAL_LOG_THINNING_ENABLED", "1")
    monkeypatch.setattr(accountability.time, "time", lambda: 1_000.0)
    accountability._LOW_SIGNAL_RECENT.clear()

    path = "/tmp/decision_explanations/shadow_aggressive_equities/decision_explanations_20260326.jsonl"
    rows = [
        {
            "status": "DATA_ONLY_BLOCKED",
            "symbol": "SPY",
            "action": "BUY",
            "strategy": "brain_refinery_v10_seasonal",
            "safety": {"market_data_only": False, "execution_enabled": True},
        },
        {
            "status": "DATA_ONLY_BLOCKED",
            "symbol": "SPY",
            "action": "BUY",
            "strategy": "brain_refinery_v10_seasonal",
            "safety": {"market_data_only": False, "execution_enabled": True},
        },
    ]

    kept = accountability._thin_low_signal_payloads(path, rows)

    assert len(kept) == 2


def test_thin_low_signal_payloads_dedupes_repetitive_paper_guard_blocks(monkeypatch) -> None:
    monkeypatch.setenv("LOW_SIGNAL_LOG_THINNING_ENABLED", "1")
    monkeypatch.setenv("LOW_SIGNAL_EXECUTION_GUARD_WINDOW_SECONDS", "60")
    monkeypatch.setattr(accountability.time, "time", lambda: 1_000.0)
    accountability._LOW_SIGNAL_RECENT.clear()

    path = "/tmp/governance/events/paper_execution_guard_20260326.jsonl"
    rows = [
        {
            "event": "pre_trade_check",
            "status": "blocked",
            "reason": "order_notional_limit",
            "mode": "paper",
            "details": {
                "symbol": "BTC-USD",
                "action": "BUY",
                "gate": "order_notional_limit",
            },
        },
        {
            "event": "pre_trade_check",
            "status": "blocked",
            "reason": "order_notional_limit",
            "mode": "paper",
            "details": {
                "symbol": "BTC-USD",
                "action": "BUY",
                "gate": "order_notional_limit",
            },
        },
    ]

    kept = accountability._thin_low_signal_payloads(path, rows)

    assert len(kept) == 1


def test_schema_violation_log_dedupes_and_summarizes_payload(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CHANNEL_SCHEMA_VIOLATION_WINDOW_SECONDS", "60")
    monkeypatch.setattr(accountability.time, "time", lambda: 1_000.0)
    accountability._SCHEMA_VIOLATION_RECENT.clear()

    payload = {
        "timestamp_utc": "2026-04-17T12:00:00+00:00",
        "symbol": "SPY",
        "event": "runtime_tick",
        "status": "bad_schema",
        "message_id": "msg-1",
        "details": {"huge": True},
    }

    accountability._schema_violation_log(
        project_root=str(tmp_path),
        source="unit_test",
        channel="runtime",
        target_path="governance/channels/runtime/test.jsonl",
        payload=payload,
        errors=["missing:event"],
    )
    accountability._schema_violation_log(
        project_root=str(tmp_path),
        source="unit_test",
        channel="runtime",
        target_path="governance/channels/runtime/test.jsonl",
        payload=payload,
        errors=["missing:event"],
    )

    day = accountability.datetime.now(accountability.timezone.utc).strftime("%Y%m%d")
    out_path = tmp_path / "governance" / "events" / f"channel_schema_violations_{day}.jsonl"
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(rows) == 1
    assert rows[0]["signature"]
    assert rows[0]["payload"]["symbol"] == "SPY"
    assert rows[0]["payload"]["payload_key_count"] >= 5
    assert "details" not in rows[0]["payload"]
