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


def test_thin_low_signal_risk_attribution_preserves_signal_and_samples_zero_outcomes(monkeypatch) -> None:
    monkeypatch.setenv("LOW_SIGNAL_LOG_THINNING_ENABLED", "1")
    monkeypatch.setenv("RISK_ATTRIBUTION_LOW_SIGNAL_WINDOW_SECONDS", "300")
    now = {"value": 1_000.0}
    monkeypatch.setattr(accountability.time, "time", lambda: now["value"])
    accountability._LOW_SIGNAL_RECENT.clear()
    accountability._LOW_SIGNAL_SUPPRESSED.clear()

    path = "/tmp/governance/shadow_crypto/shadow_pnl_attribution_20260802.jsonl"
    low_signal = {
        "bot_id": "brain_refinery_v10_seasonal",
        "layer": "sub_bot",
        "symbol": "BTC-USD",
        "action": "HOLD",
        "direction": 0.0,
        "return_1m": 0.0,
        "pnl_proxy": 0.0,
    }

    first = accountability._thin_low_signal_payloads(
        path,
        [low_signal, {**low_signal, "symbol": "ETH-USD", "return_1m": 0.002}],
    )
    assert len(first) == 1
    assert first[0]["sampling_contract"]["window_seconds"] == 300.0
    assert first[0]["sampling_contract"]["sample_scope"] == "bot_profile_action_direction"

    now["value"] = 1_100.0
    assert accountability._thin_low_signal_payloads(path, [low_signal]) == []

    now["value"] = 1_301.0
    next_window = accountability._thin_low_signal_payloads(path, [low_signal])
    assert len(next_window) == 1
    assert next_window[0]["sampling_contract"]["suppressed_since_previous"] == 2

    retained = accountability._thin_low_signal_payloads(
        path,
        [
            {**low_signal, "bot_id": "grand_master_bot", "layer": "grand_master"},
            {**low_signal, "bot_id": "nonzero_pnl_bot", "return_1m": 0.001, "pnl_proxy": 0.0001},
            {**low_signal, "bot_id": "position_bot", "action": "BUY", "quantity": 0.25},
            {**low_signal, "action": "BUY"},
        ],
    )
    assert [row["bot_id"] for row in retained] == [
        "grand_master_bot",
        "nonzero_pnl_bot",
        "position_bot",
        "brain_refinery_v10_seasonal",
    ]

    now["value"] = 1_302.0
    moving_but_flat = {**low_signal, "return_1m": 0.01}
    assert accountability._thin_low_signal_payloads(path, [moving_but_flat]) == []


def test_bulk_risk_channel_queue_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("BOT_CHANNEL_QUEUE_RISK_ENABLED", raising=False)
    assert accountability._queue_channel_enabled("risk") is False
    assert accountability._queue_channel_enabled("decision") is True

    monkeypatch.setenv("BOT_CHANNEL_QUEUE_RISK_ENABLED", "1")
    assert accountability._queue_channel_enabled("risk") is True


def test_channel_append_acknowledges_rows_intentionally_removed_by_sampling(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOW_SIGNAL_LOG_THINNING_ENABLED", "1")
    monkeypatch.setenv("RISK_ATTRIBUTION_LOW_SIGNAL_WINDOW_SECONDS", "300")
    monkeypatch.setattr(accountability.time, "time", lambda: 1_000.0)
    accountability._LOW_SIGNAL_RECENT.clear()
    accountability._LOW_SIGNAL_SUPPRESSED.clear()
    path = tmp_path / "governance" / "shadow_crypto" / "shadow_pnl_attribution_20260802.jsonl"
    base = {
        "bot_id": "flat_bot",
        "layer": "sub_bot",
        "action": "HOLD",
        "direction": 0.0,
        "return_1m": 0.002,
        "pnl_proxy": 0.0,
    }

    acknowledged = accountability.safe_append_channel_batch(
        str(path),
        [{**base, "symbol": "BTC-USD"}, {**base, "symbol": "ETH-USD"}],
        project_root=str(tmp_path),
        source="unit_test",
        channel="risk",
        schema="risk",
    )

    stored = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert acknowledged == 2
    assert len(stored) == 1
    assert stored[0]["sampling_contract"]["representative_symbol"] == "BTC-USD"


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


def test_signal_generation_event_logs_good_and_bad_decisions(tmp_path: Path) -> None:
    path = tmp_path / "decision_explanations" / "shadow_aggressive_equities" / "decision_explanations_20260428.jsonl"

    wrote = accountability.safe_append_jsonl_batch(
        str(path),
        [
            {"timestamp_utc": "2026-04-28T14:30:00+00:00", "status": "SHADOW_ONLY", "symbol": "SPY", "action": "BUY", "score": 0.7, "threshold": 0.5},
            {"timestamp_utc": "2026-04-28T14:31:00+00:00", "status": "PAPER_GUARD_BLOCKED", "symbol": "QQQ", "action": "BUY", "score": 0.8, "threshold": 0.5},
        ],
        project_root=str(tmp_path),
        source="unit_test",
    )

    day = accountability.datetime.now(accountability.timezone.utc).strftime("%Y%m%d")
    signal_path = tmp_path / "governance" / "events" / f"signal_generation_{day}.jsonl"
    rows = [json.loads(line) for line in signal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert wrote == 2
    assert [row["signal_quality"] for row in rows] == ["good_signal", "bad_signal"]
    assert rows[0]["reason"] == "trade_intent_generated"
    assert rows[1]["reason"] == "trade_intent_blocked"


def test_signal_generation_thins_repeated_low_value_bad_signals(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIGNAL_GENERATION_BAD_SIGNAL_THINNING_ENABLED", "1")
    monkeypatch.setenv("SIGNAL_GENERATION_BAD_SIGNAL_WINDOW_SECONDS", "120")
    monkeypatch.setattr(accountability.time, "time", lambda: 2_000.0)
    accountability._LOW_SIGNAL_RECENT.clear()

    path = tmp_path / "decisions" / "shadow_crypto" / "trade_decisions_20260428.jsonl"
    wrote = accountability.safe_append_jsonl_batch(
        str(path),
        [
            {"timestamp_utc": "2026-04-28T14:30:00+00:00", "symbol": "BTC-USD", "action": "HOLD", "strategy": "master_trend_bot"},
            {"timestamp_utc": "2026-04-28T14:30:01+00:00", "symbol": "BTC-USD", "action": "HOLD", "strategy": "master_trend_bot"},
            {"timestamp_utc": "2026-04-28T14:30:02+00:00", "symbol": "BTC-USD", "action": "HOLD", "strategy": "master_trend_bot"},
            {"timestamp_utc": "2026-04-28T14:31:00+00:00", "status": "SHADOW_ONLY", "symbol": "ETH-USD", "action": "BUY", "strategy": "master_trend_bot"},
            {"timestamp_utc": "2026-04-28T14:32:00+00:00", "status": "PAPER_GUARD_BLOCKED", "symbol": "SOL-USD", "action": "BUY", "strategy": "master_trend_bot"},
        ],
        project_root=str(tmp_path),
        source="unit_test",
    )

    day = accountability.datetime.now(accountability.timezone.utc).strftime("%Y%m%d")
    signal_path = tmp_path / "governance" / "events" / f"signal_generation_{day}.jsonl"
    rows = [json.loads(line) for line in signal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert wrote == 5
    assert [(row["signal_quality"], row["reason"], row["symbol"]) for row in rows] == [
        ("bad_signal", "hold_or_no_trade_signal", "BTC-USD"),
        ("good_signal", "trade_intent_generated", "ETH-USD"),
        ("bad_signal", "trade_intent_blocked", "SOL-USD"),
    ]


def test_signal_generation_caps_bad_signal_batch_without_capping_good_signals(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIGNAL_GENERATION_BAD_SIGNAL_THINNING_ENABLED", "1")
    monkeypatch.setenv("SIGNAL_GENERATION_BAD_SIGNAL_BATCH_CAP", "2")
    monkeypatch.setattr(accountability.time, "time", lambda: 2_500.0)
    accountability._LOW_SIGNAL_RECENT.clear()

    path = tmp_path / "decisions" / "shadow_aggressive_equities" / "trade_decisions_20260428.jsonl"
    wrote = accountability.safe_append_jsonl_batch(
        str(path),
        [
            {"timestamp_utc": "2026-04-28T14:30:00+00:00", "symbol": "SPY", "action": "HOLD", "strategy": "hold_bot_a"},
            {"timestamp_utc": "2026-04-28T14:30:01+00:00", "symbol": "QQQ", "action": "HOLD", "strategy": "hold_bot_b"},
            {"timestamp_utc": "2026-04-28T14:30:02+00:00", "symbol": "IWM", "action": "HOLD", "strategy": "hold_bot_c"},
            {"timestamp_utc": "2026-04-28T14:31:00+00:00", "status": "SHADOW_ONLY", "symbol": "AAPL", "action": "BUY", "strategy": "intent_bot"},
        ],
        project_root=str(tmp_path),
        source="unit_test",
    )

    day = accountability.datetime.now(accountability.timezone.utc).strftime("%Y%m%d")
    signal_path = tmp_path / "governance" / "events" / f"signal_generation_{day}.jsonl"
    rows = [json.loads(line) for line in signal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert wrote == 4
    assert [row["signal_quality"] for row in rows] == ["bad_signal", "bad_signal", "good_signal"]
    assert [row["symbol"] for row in rows] == ["SPY", "QQQ", "AAPL"]


def test_enrich_log_row_labels_schwab_equity_route() -> None:
    row = accountability.enrich_log_row(
        {
            "timestamp_utc": "2026-06-24T14:30:00+00:00",
            "broker": "schwab",
            "provider": "schwab_quote",
            "domain": "equities",
            "profile": "aggressive",
            "symbol": "SPY",
            "channel": "ingress",
        }
    )

    assert row["source_broker"] == "schwab"
    assert row["source_venue"] == "schwab"
    assert row["asset_class"] == "equities"
    assert row["routing_lane"] == "schwab_equities"
    assert row["source_quality_label"] == "broker_native"
    assert "broker_schwab" in row["data_labels"]
    assert row["data_route"]["route_key"] == "schwab_equities:ingress:schwab_quote"


def test_enrich_log_row_labels_coinbase_crypto_route() -> None:
    row = accountability.enrich_log_row(
        {
            "timestamp_utc": "2026-06-24T14:30:00+00:00",
            "broker": "coinbase",
            "source": "coinbase_ticker",
            "symbol": "BTC-USD",
            "channel": "api",
        }
    )

    assert row["source_broker"] == "coinbase"
    assert row["source_venue"] == "coinbase"
    assert row["asset_class"] == "crypto"
    assert row["routing_lane"] == "coinbase_crypto"
    assert row["source_quality_label"] == "exchange_native"
    assert "asset_crypto" in row["data_labels"]


def test_enrich_log_row_labels_broker_confirmed_account_snapshot_quality() -> None:
    row = accountability.enrich_log_row(
        {
            "timestamp_utc": "2026-08-04T20:10:47+00:00",
            "event": "get_accounts_snapshot",
            "status": "ok",
            "mode": "shadow",
            "broker": "schwab",
            "channel": "execution_guard",
            "details": {
                "status_code": 200,
                "account_snapshot_mode": "connected_account_aggregate",
            },
        }
    )

    assert row["source_broker"] == "schwab"
    assert row["routing_lane"] == "execution_guard"
    assert row["source_quality_label"] == "broker_native"
    assert row["source_quality_score"] == 0.95
    assert "quality_unclassified" not in row["data_labels"]


def test_enrich_log_row_labels_sparse_signal_generation_source_path(tmp_path: Path) -> None:
    row = accountability.enrich_log_row(
        {
            "timestamp_utc": "2026-06-24T20:31:49+00:00",
            "event": "signal_generation",
            "source_path": str(
                tmp_path / "decisions" / "shadow_intraday_aggressive_equities" / "trade_decisions_20260624.jsonl"
            ),
            "symbol": "VGIT",
            "strategy": "master_futures_bot",
        }
    )

    assert row["source_broker"] == "schwab"
    assert row["source_provider"] == "schwab"
    assert row["source_venue"] == "schwab"
    assert row["asset_class"] == "equities"
    assert row["routing_lane"] == "schwab_equities"
    assert row["source_quality_label"] == "broker_native"


def test_channel_append_enriches_sparse_schwab_crypto_path(tmp_path: Path) -> None:
    path = tmp_path / "governance" / "channels" / "decision" / "default_crypto_schwab" / "decision_20260624.jsonl"

    wrote = accountability.safe_append_channel_batch(
        str(path),
        [{"timestamp_utc": "2026-06-24T14:30:00+00:00", "symbol": "ETH-USD", "action": "HOLD"}],
        project_root=str(tmp_path),
        source="unit_test",
        channel="decision",
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert wrote == 1
    assert rows[0]["source_broker"] == "schwab"
    assert rows[0]["asset_class"] == "crypto"
    assert rows[0]["routing_lane"] == "schwab_crypto"
    assert rows[0]["data_route"]["channel"] == "decision"


def test_channel_append_enriches_sparse_schwab_equity_path(tmp_path: Path) -> None:
    path = tmp_path / "governance" / "channels" / "ingress" / "default_equities_schwab" / "ingress_20260624.jsonl"

    wrote = accountability.safe_append_channel_batch(
        str(path),
        [{"timestamp_utc": "2026-06-24T14:30:00+00:00", "symbol": "SPY", "status": "ok", "endpoint": "quote"}],
        project_root=str(tmp_path),
        source="unit_test",
        channel="ingress",
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert wrote == 1
    assert rows[0]["source_broker"] == "schwab"
    assert rows[0]["asset_class"] == "equities"
    assert rows[0]["routing_lane"] == "schwab_equities"
    assert rows[0]["source_quality_label"] == "broker_native"


def test_channel_append_repairs_placeholder_route_and_canonicalizes_master_action(tmp_path: Path) -> None:
    path = tmp_path / "governance" / "channels" / "decision" / "dividend_equities_schwab" / "decision_20260804.jsonl"

    wrote = accountability.safe_append_channel_batch(
        str(path),
        [
            {
                "timestamp_utc": "2026-08-04T17:56:17+00:00",
                "symbol": "SPYD",
                "broker": "schwab",
                "shadow_profile": "dividend",
                "master_action": "HOLD",
                "asset_class": "unknown",
                "routing_lane": "unclassified",
                "source_quality_label": "unclassified",
                "source_quality_score": 0.5,
                "data_labels": ["asset_unknown", "lane_unclassified", "quality_unclassified"],
                "data_route": {
                    "source_provider": "unknown",
                    "asset_class": "unknown",
                    "routing_lane": "unclassified",
                    "source_quality_label": "unclassified",
                    "source_quality_score": 0.5,
                    "route_key": "unclassified:schwab",
                    "labels": ["asset_unknown", "lane_unclassified", "quality_unclassified"],
                },
            }
        ],
        project_root=str(tmp_path),
        source="unit_test",
        channel="decision",
        schema="decision",
    )

    row = json.loads(path.read_text(encoding="utf-8"))
    assert wrote == 1
    assert row["action"] == "HOLD"
    assert row["action_source"] == "master_action"
    assert row["schema_valid"] is True
    assert row["asset_class"] == "equities"
    assert row["routing_lane"] == "schwab_equities"
    assert row["source_quality_label"] == "broker_native"
    assert row["source_quality_score"] == 0.95
    assert row["data_route"]["route_key"] == "schwab_equities:decision:schwab"
    assert row["data_route"]["channel"] == "decision"
    assert row["data_route"]["source_provider"] == "schwab"
    assert "asset_unknown" not in row["data_labels"]
    assert "lane_unclassified" not in row["data_route"]["labels"]
