import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import run_execution_lane as execution_lane_runner
from core.base_trader import BaseTrader
from core.channel_queue import ChannelMessage, ChannelQueue, default_queue_db_path
from core.execution_lane_pipeline import (
    EXECUTION_TRANSPORT_FEATURE_KEYS,
    EXECUTION_INTENT_CHANNEL,
    EXECUTION_PROMOTED_CHANNEL,
    EXECUTION_PROMOTION_CHANNEL,
    EXECUTION_RESULT_CHANNEL,
    configure_trader_for_lane,
    evaluate_paper_standard_gateway,
    evaluate_live_promotion,
    emit_paper_reconciliation_heartbeat,
    process_execution_intent,
    publish_execution_intent,
    update_lane_health,
)
from core.institutional_decision_flow import (
    apply_paper_decision_flow_control,
    evaluate_decision,
    load_policy,
)
from core.sleeve_strategy_specialization import attach_strategy_specialization


@pytest.fixture(autouse=True)
def _use_local_execution_lane_root(monkeypatch):
    monkeypatch.setenv("BOT_LOGS_PREFER_EXTERNAL", "0")
    monkeypatch.delenv("EXECUTION_LANE_ROOT", raising=False)


def test_default_queue_db_path_prefers_local_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("BOT_CHANNEL_QUEUE_DB", raising=False)
    monkeypatch.setenv("BOT_CHANNEL_QUEUE_PREFER_LOCAL", "1")

    assert default_queue_db_path(tmp_path) == str(
        tmp_path / "local_fallback_storage" / "data" / "bot_channel_queue.sqlite3"
    )


def test_execution_lane_health_update_cadence_is_bounded() -> None:
    assert execution_lane_runner._lane_health_update_due(0.0, 60.0, now_monotonic=100.0) is True
    assert execution_lane_runner._lane_health_update_due(100.0, 60.0, now_monotonic=159.9) is False
    assert execution_lane_runner._lane_health_update_due(100.0, 60.0, now_monotonic=160.0) is True


def test_default_queue_db_path_prefers_routed_storage_when_external_is_preferred(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("BOT_CHANNEL_QUEUE_DB", raising=False)
    monkeypatch.delenv("BOT_CHANNEL_QUEUE_PREFER_LOCAL", raising=False)
    monkeypatch.setenv("BOT_LOGS_PREFER_EXTERNAL", "1")

    assert default_queue_db_path(tmp_path) == str(
        tmp_path / "data" / "bot_channel_queue.sqlite3"
    )


def test_default_queue_db_path_respects_explicit_override(tmp_path: Path, monkeypatch) -> None:
    override = tmp_path / "custom" / "queue.sqlite3"
    monkeypatch.setenv("BOT_CHANNEL_QUEUE_DB", str(override))

    assert default_queue_db_path(tmp_path) == str(override)


def test_channel_queue_schema_check_skips_locked_existing_db(tmp_path: Path, monkeypatch) -> None:
    queue_path = tmp_path / "data" / "bot_channel_queue.sqlite3"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text("placeholder", encoding="utf-8")

    class _LockedSchemaConnection:
        def execute(self, _sql, *_args):
            raise sqlite3.OperationalError("database is locked")

        def close(self) -> None:
            return None

    monkeypatch.setattr(sqlite3, "connect", lambda *_args, **_kwargs: _LockedSchemaConnection())

    queue = ChannelQueue(queue_path)

    assert queue.db_path == queue_path


def test_channel_queue_connect_tolerates_locked_wal_pragma(tmp_path: Path, monkeypatch) -> None:
    queue_path = tmp_path / "data" / "bot_channel_queue.sqlite3"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("BOT_CHANNEL_QUEUE_WAL_RETRY_COUNT", "1")
    monkeypatch.setattr(ChannelQueue, "_schema_ready", lambda self: True)

    class _Connection:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def execute(self, sql, *_args):
            text = str(sql)
            self.commands.append(text)
            if text == "PRAGMA journal_mode=WAL":
                raise sqlite3.OperationalError("database is locked")
            return self

        def close(self) -> None:
            return None

    holder: dict[str, _Connection] = {}

    def _connect(*_args, **_kwargs):
        conn = _Connection()
        holder["conn"] = conn
        return conn

    monkeypatch.setattr(sqlite3, "connect", _connect)

    queue = ChannelQueue(queue_path)
    conn = queue._connect()

    assert conn is holder["conn"]
    assert "PRAGMA busy_timeout=30000" in holder["conn"].commands
    assert holder["conn"].commands.count("PRAGMA journal_mode=WAL") == 1
    assert "PRAGMA synchronous=NORMAL" not in holder["conn"].commands


def test_channel_queue_stale_prefix_stops_before_fresh_intent(tmp_path: Path) -> None:
    queue = ChannelQueue(default_queue_db_path(tmp_path))
    future_ts = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    queue.enqueue(
        channel=EXECUTION_INTENT_CHANNEL,
        payload={"message_id": "stale-1", "timestamp_utc": "2026-03-31T20:00:00+00:00"},
        message_id="stale-1",
    )
    queue.enqueue(
        channel=EXECUTION_INTENT_CHANNEL,
        payload={"message_id": "fresh-1", "timestamp_utc": future_ts},
        message_id="fresh-1",
    )
    queue.enqueue(
        channel=EXECUTION_INTENT_CHANNEL,
        payload={"message_id": "stale-2", "timestamp_utc": "2026-03-31T20:01:00+00:00"},
        message_id="stale-2",
    )

    prefix = queue.stale_prefix(
        consumer="execution_lane_paper",
        channel=EXECUTION_INTENT_CHANNEL,
        stale_before=datetime.now(timezone.utc),
        limit=10,
    )

    assert prefix["count"] == 1
    assert prefix["last_message_id"] == "stale-1"
    assert prefix["stopped_at_fresh"] is True


def test_channel_queue_quarantines_corrupt_db_and_recreates_schema(tmp_path: Path) -> None:
    queue_path = tmp_path / "data" / "bot_channel_queue.sqlite3"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_bytes(b"not a sqlite database")
    Path(f"{queue_path}-wal").write_text("stale wal", encoding="utf-8")

    queue = ChannelQueue(queue_path)
    message_id = queue.enqueue(
        channel=EXECUTION_INTENT_CHANNEL,
        payload={"message_id": "intent-corrupt-repair", "symbol": "BTC-USD"},
        message_id="intent-corrupt-repair",
    )
    messages = queue.read_from_cursor(consumer="pytest_corrupt_repair", channel=EXECUTION_INTENT_CHANNEL, limit=10)

    assert message_id == "intent-corrupt-repair"
    assert len(messages) == 1
    assert messages[0].payload["symbol"] == "BTC-USD"
    assert queue.last_repair["active"] is True
    assert any(row["original_path"] == str(queue_path) for row in queue.last_repair["moved"])
    assert list(queue_path.parent.glob("bot_channel_queue.sqlite3.corrupt-*"))
    assert list(queue_path.parent.glob("bot_channel_queue.sqlite3-wal.corrupt-*")) or not Path(f"{queue_path}-wal").exists()


def test_channel_queue_repairs_symlinked_external_target_without_replacing_link(tmp_path: Path) -> None:
    repo_data = tmp_path / "repo" / "data"
    external_data = tmp_path / "external" / "data"
    repo_data.mkdir(parents=True, exist_ok=True)
    external_data.mkdir(parents=True, exist_ok=True)
    target = external_data / "bot_channel_queue.sqlite3"
    target.write_bytes(b"not a sqlite database")
    link = repo_data / "bot_channel_queue.sqlite3"
    link.symlink_to(target)

    queue = ChannelQueue(link)
    queue.enqueue(
        channel=EXECUTION_INTENT_CHANNEL,
        payload={"message_id": "intent-symlink-repair", "symbol": "ETH-USD"},
        message_id="intent-symlink-repair",
    )

    assert link.is_symlink()
    assert link.resolve(strict=False) == target.resolve(strict=False)
    assert list(external_data.glob("bot_channel_queue.sqlite3.corrupt-*"))
    assert any(row["via_symlink"] is True for row in queue.last_repair["moved"])
    assert queue.pending_count(consumer="pytest_symlink_repair", channel=EXECUTION_INTENT_CHANNEL) == 1


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_gates(project_root: Path, *, promote_ok: bool, quality_ok: bool) -> None:
    role_contract_source = Path(__file__).resolve().parents[1] / "config" / "system_role_contracts_v1.json"
    role_contract_target = project_root / "config" / "system_role_contracts_v1.json"
    role_contract_target.parent.mkdir(parents=True, exist_ok=True)
    role_contract_target.write_text(role_contract_source.read_text(encoding="utf-8"), encoding="utf-8")
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


def _paper_consensus_metadata(bot_ids: list[str], *, segment: str = "core") -> dict:
    ids = sorted(bot_ids)
    manifest = {
        "policy": "paper_execution_authority_v2",
        "profile": "baseline",
        "segment": segment,
        "constituent_bot_ids": ids,
    }
    return {
        "layer": "paper_portfolio_consensus",
        "source_profile": "baseline",
        "signal_segment": segment,
        "paper_execution_authority_version": "paper_execution_authority_v2",
        "paper_execution_diversity_ready": True,
        "paper_execution_distinct_correlation_clusters": 2,
        "constituent_count": len(ids),
        "constituent_bot_ids": ids,
        "constituent_bot_ids_truncated": False,
        "constituent_bot_ids_sha256": hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "paper_execution_cohort_manifest": manifest,
        "paper_execution_cohort_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _qualified_decision_flow_metadata(
    *,
    symbol: str = "SPY",
    action: str = "BUY",
    quantity: float = 1.0,
    profile: str = "default",
) -> dict:
    policy = load_policy()
    timestamp = datetime.now(timezone.utc).isoformat()
    evaluation = evaluate_decision(
        {
            "timestamp_utc": timestamp,
            "message_id": f"flow-{symbol}-{action}",
            "run_id": "run-flow",
            "snapshot_id": "snapshot-flow",
            "broker": "schwab",
            "shadow_profile": profile,
            "shadow_domain": "equities",
            "routing_lane": "default",
            "symbol": symbol,
            "action": action,
            "master_action": action,
            "master_intent_action": action,
            "master_intent_score": 0.82 if action == "BUY" else 0.18,
            "source_quality_score": 1.0,
            "feature_freshness": {"ok": True},
            "data_quality_features": {
                "data_quality_quote_agreement_norm": 1.0,
                "data_quality_missing_feature_ratio_norm": 0.0,
            },
            "market": {
                "last_price": 100.0,
                "spread_bps": 1.0,
                "market_data_latency_ms": 20.0,
                "market_impact_curve": {"1000": 0.4},
            },
            "market_micro_features": {
                "market_micro_tradeability_score_norm": 0.95,
                "market_micro_trend_persistence_norm": 0.80,
                "market_micro_post_event_drift_norm": 0.75,
                "market_micro_reversal_risk_norm": 0.10,
            },
            "grand_master_meta": {
                "specialist_consensus": 0.80 if action == "BUY" else -0.80,
                "sleeve_master_consensus": 0.75 if action == "BUY" else -0.75,
                "directional_alignment": 0.70 if action == "BUY" else -0.70,
                "master_disagreement": 0.10,
                "quant_strategy_fit": 0.85,
                "quant_data_confidence": 1.0,
            },
            "allocation_confidence": {
                "allocation_confidence_norm": 0.90,
                "allocation_conflict_norm": 0.10,
                "portfolio_overlap_pressure_norm": 0.10,
            },
            "execution_guard": {"ok": True},
            "execution_sim": {
                "slippage_bps": 0.5,
                "impact_bps": 0.2,
                "fee_bps": 0.1,
            },
            "portfolio": {"lane_budget_mult": 1.0},
            "portfolio_risk_engine": {"blocked": False},
            "long_term_turnover_policy": {"blocked": False},
            "circuit_breakers": {},
            "broker_truth_reconcile": {"ok": True},
            "position_context": {
                "truth_available": True,
                "current_quantity": 0.0,
                "short_permission_confirmed": True,
                "linked_leg_truth_ready": True,
                "defined_risk_structure_ready": True,
            },
            "quantitative_evidence": {
                "selection_bias_control": 0.90,
                "independent_samples": 0.90,
                "uncertainty_calibration": 0.90,
                "signal_decay_fit": 0.90,
                "payoff_asymmetry": 0.90,
                "capacity_headroom": 0.90,
                "crowding_residual": 0.90,
                "tail_survival": 0.90,
                "regime_stability": 0.90,
            },
            "predicted_edge_lower_confidence_bound_bps": 40.0,
            "post_cost_samples": 100,
            "post_cost_lower_confidence_bound": 0.01,
        },
        policy,
    )
    output_action, output_quantity, control = apply_paper_decision_flow_control(
        target_mode="paper",
        current_action=action,
        quantity=quantity,
        evaluation=evaluation,
        policy=policy,
    )
    assert evaluation["qualified_shadow_candidate"] is True
    assert (output_action, output_quantity) == (action, quantity)
    metadata = {
        "layer": "grand_master",
        "source_profile": profile,
        "shadow_domain": "equities",
        "lifecycle_state": "",
        "institutional_decision_flow": {
            "policy_receipt": evaluation["policy_receipt"],
            "evaluation": evaluation,
            "control": control,
        },
    }
    return attach_strategy_specialization(
        {
            **metadata,
            "production_candidate_id": "pc-test-g1",
        },
        profile=profile,
        raw_strategy="grand_master_bot",
        features={
            "market_regime_snapshot": {
                "regime_state": "mixed_transition",
            }
        },
        action=action,
        quantity=quantity,
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
            "features": {
                "last_price": 500.0,
                "spread_bps": 2.0,
                "training_only_feature": 123.0,
            },
            "metadata": {"snapshot_id": "snap-1"},
        },
    )

    queue = ChannelQueue(default_queue_db_path(tmp_path))
    messages = queue.read_from_cursor(consumer="pytest", channel=EXECUTION_INTENT_CHANNEL, limit=10)

    assert row["message_id"]
    assert len(messages) == 1
    assert messages[0].payload["symbol"] == "SPY"
    assert messages[0].payload["strategy"] == "grand_master_bot"
    assert messages[0].payload["features"] == {"last_price": 500.0, "spread_bps": 2.0}
    assert messages[0].payload["execution_transport"]["compacted"] is True
    assert set(messages[0].payload["features"]) <= EXECUTION_TRANSPORT_FEATURE_KEYS
    intent_path = tmp_path / "governance" / "execution_lanes" / f"execution_intents_{datetime.now(timezone.utc):%Y%m%d}.jsonl"
    persisted = json.loads(intent_path.read_text(encoding="utf-8").splitlines()[-1])
    assert "training_only_feature" not in persisted["features"]


def test_publish_execution_intent_retries_locked_queue(monkeypatch, tmp_path: Path) -> None:
    attempts = {"count": 0}
    original_enqueue = ChannelQueue.enqueue

    def flaky_enqueue(self, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return original_enqueue(self, **kwargs)

    monkeypatch.setattr(ChannelQueue, "enqueue", flaky_enqueue)
    row = publish_execution_intent(
        project_root=str(tmp_path),
        payload={
            "symbol": "SPY",
            "action": "BUY",
            "quantity": 1.0,
            "model_score": 0.66,
            "threshold": 0.55,
            "strategy": "grand_master_bot",
        },
    )

    queue = ChannelQueue(default_queue_db_path(tmp_path))
    messages = queue.read_from_cursor(consumer="pytest_retry", channel=EXECUTION_INTENT_CHANNEL, limit=10)

    assert attempts["count"] == 3
    assert row["message_id"]
    assert len(messages) == 1


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


def test_direct_live_execution_fails_closed_without_sleeve_policy_receipt(
    tmp_path: Path,
) -> None:
    trader = BaseTrader(
        "dummy_key",
        "dummy_secret",
        "https://127.0.0.1:8182",
        mode="live",
    )
    trader.project_root = str(tmp_path)
    configure_trader_for_lane(trader, "live")
    message = ChannelMessage(
        id=1,
        channel=EXECUTION_PROMOTED_CHANNEL,
        message_id="live-missing-flow",
        parent_message_id="paper-parent",
        run_id="run-live",
        iter_id="iter-live",
        source_path="",
        payload={
            "message_id": "live-missing-flow",
            "intent_kind": "master",
            "symbol": "SPY",
            "action": "BUY",
            "quantity": 1.0,
            "strategy": "grand_master_bot",
            "metadata": {"source_profile": "default"},
        },
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    out = process_execution_intent(
        project_root=str(tmp_path),
        trader=trader,
        mode="live",
        message=message,
    )

    assert out["result"]["result_status"] == "LIVE_DECISION_FLOW_BLOCKED"
    assert out["result"]["decision_flow_guard"]["allow_execute"] is False
    assert "decision_flow_metadata_missing" in out["result"][
        "decision_flow_guard"
    ]["reasons"]


def test_process_execution_intent_paper_emits_result_and_promoted_message(tmp_path: Path) -> None:
    _seed_gates(tmp_path, promote_ok=True, quality_ok=True)
    _write_json(
        tmp_path / "governance" / "allocator" / "portfolio_allocator_service_latest.json",
        {"ok": True, "approved_intents": [{"symbol": "SPY", "side": "BUY", "approved_qty": 1.0}]},
    )
    _write_json(
        tmp_path / "governance" / "risk" / "risk_service_boundary_latest.json",
        {"ok": True, "pre_trade_decisions": [{"symbol": "SPY", "requested_action": "BUY", "approved_action": "BUY", "risk_limit_ok": True}]},
    )
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
                **_qualified_decision_flow_metadata(),
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


def test_process_execution_intent_blocks_promotion_on_stale_realism_fill(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PAPER_REALISM_MIN_PROMOTION_SCORE", "65")
    _seed_gates(tmp_path, promote_ok=True, quality_ok=True)
    _write_json(
        tmp_path / "governance" / "allocator" / "portfolio_allocator_service_latest.json",
        {"ok": True, "approved_intents": [{"symbol": "NVDA", "side": "SELL_TO_OPEN", "approved_qty": 10.0}]},
    )
    _write_json(
        tmp_path / "governance" / "risk" / "risk_service_boundary_latest.json",
        {"ok": True, "pre_trade_decisions": [{"symbol": "NVDA", "requested_action": "SELL_TO_OPEN", "approved_action": "SELL_TO_OPEN", "risk_limit_ok": True}]},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"sub_bots": []})

    trader = BaseTrader("dummy_key", "dummy_secret", "https://127.0.0.1:8182", mode="paper")
    trader.project_root = str(tmp_path)
    trader.set_mode("paper")
    configure_trader_for_lane(trader, "paper")

    message = ChannelMessage(
        id=1,
        channel=EXECUTION_INTENT_CHANNEL,
        message_id="intent-stale-option",
        parent_message_id="",
        run_id="run-1",
        iter_id="iter-1",
        source_path="",
        payload={
            "message_id": "intent-stale-option",
            "intent_kind": "master",
            "symbol": "NVDA",
            "action": "SELL_TO_OPEN",
            "quantity": 10.0,
            "model_score": 0.78,
            "threshold": 0.55,
            "features": {
                "last_price": 4.0,
                "spread_bps": 60.0,
                "volatility_1m": 0.02,
                "latency_ms": 500.0,
                "bid_size": 5.0,
                "ask_size": 5.0,
                "quote_age_ms": 6000.0,
                "open_interest": 0.0,
            },
            "gates": {"market_data_ok": True, "risk_limit_ok": True},
            "reasons": ["score_above_threshold"],
            "strategy": "covered_call_roll_watch",
            "metadata": {
                "asset_class": "options",
                "sleeve": "covered_call",
                "allow_live_promotion": True,
                "runtime_lane": "default",
                "order_type": "limit",
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
    promoted_rows = queue.read_from_cursor(consumer="pytest_live_stale", channel=EXECUTION_PROMOTED_CHANNEL, limit=10)

    result = out["result"]["result"]
    assert out["result"]["result_status"] == "PAPER_PROFITABILITY_GUARD_BLOCKED"
    assert "paper_order" not in result
    assert result["live_guard_decision"]["gate"] == "paper_profitability_entry_policy"
    assert result["live_guard_decision"]["reason"] == "paper_profitability_entry_policy_block"
    assert out["promotion"]["promotion"]["promote_ok"] is False
    assert len(promoted_rows) == 0


def test_paper_standard_gateway_blocks_collection_only_intent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PAPER_LIVE_DATA_STANDARD_ENABLED", "1")
    _seed_gates(tmp_path, promote_ok=True, quality_ok=True)
    _write_json(
        tmp_path / "governance" / "allocator" / "portfolio_allocator_service_latest.json",
        {"ok": True, "approved_intents": [{"symbol": "SPY", "side": "BUY", "approved_qty": 1.0}]},
    )
    _write_json(
        tmp_path / "governance" / "risk" / "risk_service_boundary_latest.json",
        {"ok": True, "pre_trade_decisions": [{"symbol": "SPY", "requested_action": "BUY", "approved_action": "BUY", "risk_limit_ok": True}]},
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v167_intraday_opening_range_momentum_burst",
                    "active": True,
                    "data_collection_active": True,
                    "paper_standard_cohort": "collection_until_standard_met",
                    "paper_live_data_enabled": False,
                    "direct_execution_allowed": False,
                    "live_trading_enabled": False,
                }
            ],
        },
    )

    trader = BaseTrader("dummy_key", "dummy_secret", "https://127.0.0.1:8182", mode="paper")
    trader.project_root = str(tmp_path)
    trader.set_mode("paper")
    configure_trader_for_lane(trader, "paper")

    message = ChannelMessage(
        id=1,
        channel=EXECUTION_INTENT_CHANNEL,
        message_id="intent-standard-block",
        parent_message_id="",
        run_id="run-1",
        iter_id="iter-1",
        source_path="",
        payload={
            "message_id": "intent-standard-block",
            "intent_kind": "paper_mirror",
            "symbol": "SPY",
            "action": "BUY",
            "quantity": 1.0,
            "model_score": 0.64,
            "threshold": 0.55,
            "strategy": "paper_mirror::brain_refinery_v167_intraday_opening_range_momentum_burst",
            "metadata": {"allow_live_promotion": False},
        },
        created_at="2026-03-31T20:00:00+00:00",
    )

    out = process_execution_intent(
        project_root=str(tmp_path),
        trader=trader,
        mode="paper",
        message=message,
    )

    gateway = evaluate_paper_standard_gateway(project_root=str(tmp_path), intent=message.payload)
    assert gateway["allow_execute"] is False
    assert out["result"]["result_status"] == "PAPER_STANDARD_BLOCKED"
    assert out["result"]["paper_standard_gateway"]["reasons"] == ["paper_standard_bot_not_in_explicit_paper_cohort"]


def test_paper_standard_gateway_allows_explicit_paper_bot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PAPER_LIVE_DATA_STANDARD_ENABLED", "1")
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v26_restored_probation",
                    "active": True,
                    "bot_role": "signal_sub_bot",
                    "lifecycle_state": "paper_live_data",
                    "test_accuracy": 0.61,
                    "quality_score": 0.72,
                    "paper_standard_cohort": "legacy_bootstrap",
                    "paper_live_data_enabled": True,
                    "paper_execution_allowed": True,
                    "paper_execution_authority": False,
                    "paper_probation_authority": True,
                    "paper_probation_requalification_allowed": True,
                    "paper_execution_authority_version": "paper_execution_authority_v2",
                    "direct_execution_allowed": False,
                    "live_trading_enabled": False,
                }
            ],
        },
    )

    gateway = evaluate_paper_standard_gateway(
        project_root=str(tmp_path),
        intent={"strategy": "paper_mirror::brain_refinery_v26_restored_probation"},
    )

    assert gateway["allow_execute"] is True
    assert gateway["paper_standard_cohort"] == "legacy_bootstrap"


def test_paper_standard_gateway_uses_only_hash_bound_candidate_overlay(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PAPER_LIVE_DATA_STANDARD_ENABLED", "1")
    source_path = tmp_path / "master_bot_registry.json"
    candidate_path = tmp_path / "governance" / "health" / "paper_live_data_standard_registry_candidate_latest.json"
    guard_path = tmp_path / "governance" / "health" / "paper_live_data_standard_source_write_guard_latest.json"
    health_path = tmp_path / "governance" / "health" / "paper_live_data_standard_latest.json"
    source = {
        "summary": {},
        "sub_bots": [
            {
                "bot_id": "signal_a",
                "active": True,
                "bot_role": "signal_sub_bot",
                "lifecycle_state": "paper_live_data",
                "test_accuracy": 0.64,
                "quality_score": 0.75,
                "paper_execution_authority": False,
                "paper_execution_authority_version": "paper_execution_authority_v2",
            }
        ],
    }
    candidate = {
        "summary": {"paper_live_data_standard_version": "paper_live_data_standard_v2"},
        "sub_bots": [
            {
                **source["sub_bots"][0],
                "paper_execution_authority": True,
                "direct_execution_allowed": False,
                "live_trading_enabled": False,
            }
        ],
    }
    _write_json(source_path, source)
    _write_json(candidate_path, candidate)
    _write_json(
        guard_path,
        {
            "source_write_blocked": True,
            "source_path": str(source_path),
            "candidate_path": str(candidate_path),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        },
    )
    _write_json(health_path, {"ok": True})
    intent = {"strategy": "paper_mirror::signal_a"}

    allowed = evaluate_paper_standard_gateway(project_root=str(tmp_path), intent=intent)
    assert allowed["allow_execute"] is True
    assert allowed["registry_provenance"]["source"] == "hash_bound_candidate_overlay"

    candidate_path.write_text(candidate_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    blocked = evaluate_paper_standard_gateway(project_root=str(tmp_path), intent=intent)
    assert blocked["allow_execute"] is False
    assert blocked["registry_provenance"]["candidate_overlay_valid"] is False
    assert "candidate_registry_hash_mismatch" in blocked["registry_provenance"]["reasons"]


def test_paper_standard_gateway_does_not_authorize_virtual_name_patterns(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PAPER_LIVE_DATA_STANDARD_ENABLED", "1")
    _write_json(tmp_path / "master_bot_registry.json", {"sub_bots": []})

    gateway = evaluate_paper_standard_gateway(
        project_root=str(tmp_path),
        intent={"strategy": "paper_mirror::options_specialist_unregistered"},
    )

    assert gateway["allow_execute"] is False
    assert gateway["virtual_allowed"] is False
    assert gateway["reasons"] == ["paper_standard_bot_missing_from_registry"]


def test_paper_standard_gateway_validates_consensus_constituents(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PAPER_LIVE_DATA_STANDARD_ENABLED", "1")
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "eligible_bot",
                    "active": True,
                    "bot_role": "signal_sub_bot",
                    "lifecycle_state": "paper_live_data",
                    "test_accuracy": 0.64,
                    "quality_score": 0.75,
                    "paper_live_data_enabled": True,
                    "paper_execution_allowed": True,
                    "paper_execution_authority": True,
                    "paper_execution_authority_version": "paper_execution_authority_v2",
                    "direct_execution_allowed": False,
                    "live_trading_enabled": False,
                },
                {
                    "bot_id": "collection_only_bot",
                    "active": True,
                    "bot_role": "signal_sub_bot",
                    "lifecycle_state": "data_collection_only",
                    "test_accuracy": 0.90,
                    "quality_score": 0.90,
                    "paper_live_data_enabled": False,
                    "paper_execution_allowed": False,
                    "paper_execution_authority": False,
                    "paper_execution_authority_version": "paper_execution_authority_v2",
                    "direct_execution_allowed": False,
                    "live_trading_enabled": False,
                },
                {
                    "bot_id": "eligible_bot_b",
                    "active": True,
                    "bot_role": "signal_sub_bot",
                    "lifecycle_state": "paper_live_data",
                    "test_accuracy": 0.62,
                    "quality_score": 0.73,
                    "paper_live_data_enabled": True,
                    "paper_execution_allowed": True,
                    "paper_execution_authority": True,
                    "paper_execution_authority_version": "paper_execution_authority_v2",
                    "direct_execution_allowed": False,
                    "live_trading_enabled": False,
                },
            ]
        },
    )
    base_intent = {
        "strategy": "paper_portfolio_consensus::baseline::core",
        "metadata": _paper_consensus_metadata(["eligible_bot", "eligible_bot_b"]),
    }

    allowed = evaluate_paper_standard_gateway(project_root=str(tmp_path), intent=base_intent)
    blocked = evaluate_paper_standard_gateway(
        project_root=str(tmp_path),
        intent={
            **base_intent,
            "metadata": _paper_consensus_metadata(["eligible_bot", "collection_only_bot"]),
        },
    )

    assert allowed["allow_execute"] is True
    assert allowed["consensus_constituent_count"] == 2
    assert blocked["allow_execute"] is False
    assert blocked["consensus_invalid_bot_ids"] == ["collection_only_bot"]
    assert blocked["reasons"] == ["paper_standard_consensus_contains_ineligible_bot"]


def test_paper_standard_gateway_fails_closed_on_incomplete_consensus_identity(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PAPER_LIVE_DATA_STANDARD_ENABLED", "1")

    gateway = evaluate_paper_standard_gateway(
        project_root=str(tmp_path),
        intent={
            "strategy": "paper_portfolio_consensus::baseline::core",
            "metadata": {
                "layer": "paper_portfolio_consensus",
                "constituent_bot_ids": [],
                "constituent_bot_ids_truncated": True,
            },
        },
    )

    assert gateway["allow_execute"] is False
    assert "paper_standard_consensus_missing_constituents" in gateway["reasons"]
    assert "paper_standard_consensus_constituents_truncated" in gateway["reasons"]
    assert "paper_standard_consensus_authority_version_mismatch" in gateway["reasons"]


def test_paper_standard_gateway_binds_consensus_to_current_candidate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PAPER_LIVE_DATA_STANDARD_ENABLED", "1")
    rows = []
    for bot_id in ("eligible_a", "eligible_b"):
        rows.append(
            {
                "bot_id": bot_id,
                "active": True,
                "bot_role": "signal_sub_bot",
                "lifecycle_state": "paper_live_data",
                "test_accuracy": 0.64,
                "quality_score": 0.75,
                "paper_execution_authority": True,
                "paper_execution_authority_version": "paper_execution_authority_v2",
                "direct_execution_allowed": False,
                "live_trading_enabled": False,
            }
        )
    _write_json(tmp_path / "master_bot_registry.json", {"sub_bots": rows})
    _write_json(
        tmp_path / "governance" / "runtime" / "production_candidate_state.json",
        {"candidate_id": "candidate-current"},
    )
    metadata = _paper_consensus_metadata(["eligible_a", "eligible_b"])

    missing = evaluate_paper_standard_gateway(
        project_root=str(tmp_path),
        intent={"strategy": "paper_portfolio_consensus::baseline::core", "metadata": metadata},
    )
    matched = evaluate_paper_standard_gateway(
        project_root=str(tmp_path),
        intent={
            "strategy": "paper_portfolio_consensus::baseline::core",
            "metadata": {**metadata, "production_candidate_id": "candidate-current"},
        },
    )

    assert "paper_standard_production_candidate_id_missing" in missing["reasons"]
    assert missing["allow_execute"] is False
    assert matched["allow_execute"] is True


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
    monkeypatch.setenv("EXECUTION_LANE_HEALTH_QUEUE_STATS_ENABLED", "1")
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
    monkeypatch.setenv("EXECUTION_LANE_HEALTH_QUEUE_STATS_ENABLED", "1")
    update_lane_health(
        project_root=str(tmp_path),
        mode="paper",
        processed_count=1,
        queue_channel=EXECUTION_INTENT_CHANNEL,
    )

    payload = json.loads((tmp_path / "governance" / "health" / "execution_lane_paper_latest.json").read_text(encoding="utf-8"))
    assert payload["pending_rows"] == 0
    assert payload["stale"] is False


def test_update_lane_health_allows_active_backlog_grace_before_marking_stale(tmp_path: Path, monkeypatch) -> None:
    queue = ChannelQueue(default_queue_db_path(tmp_path))
    queue.enqueue(
        channel=EXECUTION_INTENT_CHANNEL,
        payload={"message_id": "intent-1", "timestamp_utc": "2099-03-31T20:00:00+00:00"},
        message_id="intent-1",
    )
    with queue._connect() as conn:
        conn.execute(
            """
            INSERT INTO channel_consumer_state (consumer, channel, last_id, last_message_id, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("execution_lane_paper", EXECUTION_INTENT_CHANNEL, 0, "", "2099-03-31T19:59:00+00:00"),
        )
        conn.commit()

    monkeypatch.setenv("EXECUTION_LANE_STALE_AFTER_SECONDS", "60")
    monkeypatch.setenv("EXECUTION_LANE_HEALTH_QUEUE_STATS_ENABLED", "1")
    update_lane_health(
        project_root=str(tmp_path),
        mode="paper",
        processed_count=0,
        queue_channel=EXECUTION_INTENT_CHANNEL,
    )

    payload = json.loads((tmp_path / "governance" / "health" / "execution_lane_paper_latest.json").read_text(encoding="utf-8"))
    assert payload["pending_rows"] == 1
    assert payload["stale"] is False
    assert payload["stale_grace_seconds"] >= 60


def test_update_lane_health_writes_heartbeat_when_queue_stats_fail(tmp_path: Path, monkeypatch) -> None:
    class _BrokenQueue:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("queue unavailable")

    monkeypatch.setattr("core.execution_lane_pipeline.ChannelQueue", _BrokenQueue)
    monkeypatch.setenv("EXECUTION_LANE_HEALTH_QUEUE_STATS_ENABLED", "1")

    update_lane_health(
        project_root=str(tmp_path),
        mode="paper",
        processed_count=7,
        queue_channel=EXECUTION_INTENT_CHANNEL,
        auth_ok=False,
        auth_error="paper_execution_paused_for_runtime_pressure",
    )

    payload = json.loads((tmp_path / "governance" / "health" / "execution_lane_paper_latest.json").read_text(encoding="utf-8"))
    assert payload["queue_stats_available"] is False
    assert payload["queue_stats_status"] == "error"
    assert payload["queue_stats_error_type"] == "RuntimeError"
    assert payload["pending_rows_unknown"] is True
    assert payload["stale"] is False
    assert payload["auth_error"] == "paper_execution_paused_for_runtime_pressure"


def test_update_lane_health_skips_queue_stats_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("EXECUTION_LANE_HEALTH_QUEUE_STATS_ENABLED", raising=False)

    update_lane_health(
        project_root=str(tmp_path),
        mode="paper",
        processed_count=3,
        queue_channel=EXECUTION_INTENT_CHANNEL,
    )

    payload = json.loads((tmp_path / "governance" / "health" / "execution_lane_paper_latest.json").read_text(encoding="utf-8"))
    assert payload["queue_stats_available"] is False
    assert payload["queue_stats_status"] == "skipped"
    assert payload["queue_stats_skip_reason"] == "disabled_for_nonblocking_execution_lane_heartbeat"
    assert payload["pending_rows_unknown"] is True
    assert payload["stale"] is False


def test_update_lane_health_reports_stale_skip_only_result_activity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BOT_LOGS_PREFER_EXTERNAL", "0")
    monkeypatch.setenv("EXECUTION_LANE_HEALTH_RESULT_FRESH_SECONDS", "900")
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    result_path = tmp_path / "governance" / "execution_lanes" / f"execution_results_{day}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "paper",
                "result_status": "STALE_INTENT_SKIPPED",
                "result": {"reason": "stale_execution_intent"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    update_lane_health(
        project_root=str(tmp_path),
        mode="paper",
        processed_count=1,
        queue_channel=EXECUTION_INTENT_CHANNEL,
    )

    payload = json.loads((tmp_path / "governance" / "health" / "execution_lane_paper_latest.json").read_text(encoding="utf-8"))
    assert payload["result_activity_status"] == "stale_skip_only"
    assert payload["stale_skip_only_result_activity"] is True
    assert payload["fresh_paper_executed"] is False
    assert payload["execution_result_evidence"]["stale_skip_rows"] == 1


def test_update_lane_health_keeps_old_stale_skip_audit_non_active(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BOT_LOGS_PREFER_EXTERNAL", "0")
    monkeypatch.setenv("EXECUTION_LANE_HEALTH_RESULT_FRESH_SECONDS", "900")
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    result_path = tmp_path / "governance" / "execution_lanes" / f"execution_results_{day}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "timestamp_utc": (datetime.now(timezone.utc) - timedelta(seconds=1200)).isoformat(),
                "mode": "paper",
                "result_status": "STALE_INTENT_SKIPPED",
                "result": {"reason": "stale_execution_intent"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    update_lane_health(
        project_root=str(tmp_path),
        mode="paper",
        processed_count=1,
        queue_channel=EXECUTION_INTENT_CHANNEL,
    )

    payload = json.loads((tmp_path / "governance" / "health" / "execution_lane_paper_latest.json").read_text(encoding="utf-8"))
    assert payload["result_activity_status"] == "old_stale_skip_audit_only"
    assert payload["stale_skip_only_result_activity"] is False
    assert payload["execution_result_evidence"]["historical_stale_skip_only"] is True
    assert payload["execution_result_evidence"]["stale_skip_only"] is False


def test_emit_paper_reconciliation_heartbeat_writes_guard_event(tmp_path: Path) -> None:
    class _Guard:
        def reconcile_order_lifecycle(self, *, broker_open_orders):
            return {
                "ok": True,
                "missing_on_broker": [],
                "missing_local": [],
                "position_checks": [],
                "open_orders_local_total": 0,
                "open_orders_broker_total": len(broker_open_orders),
            }

    class _Trader:
        mode = "paper"
        mode_label = "paper"
        live_account_hash = ""
        live_guard = _Guard()

    last_emit = emit_paper_reconciliation_heartbeat(
        project_root=str(tmp_path),
        trader=_Trader(),
        min_interval_seconds=180.0,
    )

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = tmp_path / "governance" / "events" / f"paper_execution_guard_{day}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert last_emit > 0.0
    assert len(rows) == 1
    assert rows[0]["event"] == "paper_order_lifecycle_reconcile"
    assert rows[0]["status"] == "ok"
    assert rows[0]["details"]["heartbeat"] is True
    assert rows[0]["details"]["order_lifecycle_reconcile"]["ok"] is True


def test_emit_paper_reconciliation_heartbeat_throttles(tmp_path: Path) -> None:
    class _Guard:
        def reconcile_order_lifecycle(self, *, broker_open_orders):
            return {"ok": True}

    class _Trader:
        mode_label = "paper"
        live_account_hash = ""
        live_guard = _Guard()

    last_emit = emit_paper_reconciliation_heartbeat(
        project_root=str(tmp_path),
        trader=_Trader(),
        min_interval_seconds=180.0,
    )
    second_emit = emit_paper_reconciliation_heartbeat(
        project_root=str(tmp_path),
        trader=_Trader(),
        last_emit_monotonic=last_emit,
        min_interval_seconds=180.0,
    )

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = tmp_path / "governance" / "events" / f"paper_execution_guard_{day}.jsonl"
    assert second_emit == last_emit
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
