import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.live_execution_controls import LiveExecutionGuard, LiveRiskConfig, production_order_firewall_check


def _cfg(**overrides):
    base = LiveRiskConfig(
        max_position_qty_per_symbol=10.0,
        max_order_notional=5000.0,
        max_open_orders_total=5,
        max_open_orders_per_symbol=2,
        daily_loss_cap=100.0,
        api_fail_limit=2,
        api_cooldown_seconds=60,
        trade_min_interval_seconds=8.0,
        trade_min_interval_global_seconds=1.0,
        max_slippage_bps=35.0,
        max_fill_deviation_bps=45.0,
        min_execution_realism_score=25.0,
        min_effective_fill_ratio=0.50,
        max_reject_probability=0.80,
        max_cancel_probability=0.85,
        max_stale_quote_probability=0.80,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _write_firewall_fixture(project_root: Path, *, excellence_ready: bool, symbols: list[str]) -> dict:
    now = datetime.now(timezone.utc)
    config = {
        "live_execution_risk_firewall": {
            "allow_order_execution_env": "ALLOW_ORDER_EXECUTION",
            "market_data_only_env": "MARKET_DATA_ONLY",
            "market_data_only_default": True,
            "halt_flags": [],
            "required_safety_flags": [],
            "max_order_quantity": 5,
            "max_single_order_notional": 100,
            "allowed_asset_types": ["EQUITY"],
            "allowed_instructions": ["BUY", "SELL"],
            "canary_allowlist_path": "governance/runtime/live_canary_allowlist.json",
            "canary_plan_path": "config/live_canary_micro_policy_v1.json",
            "production_candidate_state_path": "governance/runtime/production_candidate_state.json",
            "symbol_lifecycle_path": "config/symbol_lifecycle_v1.json",
            "require_pinned_account_reference": False,
            "production_excellence_artifact": "governance/health/production_excellence_control_latest.json",
            "require_production_excellence_for_live_submit": True,
        }
    }
    config_path = project_root / "config" / "production_readiness_control_v1.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    candidate_id = "pc-test-candidate"
    candidate = project_root / "governance" / "runtime" / "production_candidate_state.json"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(
        json.dumps({"candidate_id": candidate_id, "accepted_at_utc": (now - timedelta(minutes=2)).isoformat()}),
        encoding="utf-8",
    )
    plan = project_root / "config" / "live_canary_micro_policy_v1.json"
    plan.write_text(
        json.dumps(
            {
                "hard_limits": {
                    "max_order_notional_usd": 100,
                    "max_order_quantity": 1,
                    "max_daily_loss_usd": 2,
                    "max_cumulative_loss_usd": 10,
                    "max_concurrent_positions": 1,
                },
                "stages": [{"stage": 1, "symbols": symbols}],
                "activation_contract": {"max_allowlist_duration_hours": 4},
            }
        ),
        encoding="utf-8",
    )
    lifecycle = project_root / "config" / "symbol_lifecycle_v1.json"
    lifecycle.write_text(json.dumps({"renamed_symbols": {"SPLG": "SPYM"}}), encoding="utf-8")
    allowlist = project_root / "governance" / "runtime" / "live_canary_allowlist.json"
    allowlist.parent.mkdir(parents=True, exist_ok=True)
    allowlist.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "enabled": True,
                "candidate_id": candidate_id,
                "stage": 1,
                "symbols": symbols,
                "issued_at_utc": (now - timedelta(minutes=1)).isoformat(),
                "expires_at_utc": (now + timedelta(hours=1)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    excellence = project_root / "governance" / "health" / "production_excellence_control_latest.json"
    excellence.parent.mkdir(parents=True, exist_ok=True)
    excellence.write_text(
        json.dumps(
            {
                "ten_out_of_ten_ready": excellence_ready,
                "live_money_consideration_ready": excellence_ready,
            }
        ),
        encoding="utf-8",
    )
    return {
        "orderType": "LIMIT",
        "price": 10.0,
        "orderLegCollection": [
            {
                "instruction": "BUY",
                "instrument": {"symbol": "AAPL", "assetType": "EQUITY"},
            }
        ],
    }


def test_production_firewall_requires_ten_pillar_evidence(tmp_path: Path) -> None:
    order_spec = _write_firewall_fixture(tmp_path, excellence_ready=False, symbols=["AAPL"])

    decision = production_order_firewall_check(
        project_root=tmp_path,
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        order_spec=order_spec,
        env={"ALLOW_ORDER_EXECUTION": "1", "MARKET_DATA_ONLY": "0"},
    )

    assert decision.ok is False
    assert decision.reason == "production_excellence_not_ready"
    assert "production_excellence_not_ready" in decision.details["blockers"]


def test_production_firewall_fails_closed_when_required_role_contract_is_missing(tmp_path: Path) -> None:
    order_spec = _write_firewall_fixture(tmp_path, excellence_ready=True, symbols=["AAPL"])
    config_path = tmp_path / "config" / "production_readiness_control_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["live_execution_risk_firewall"]["require_system_role_contract_for_live_submit"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")

    decision = production_order_firewall_check(
        project_root=tmp_path,
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        order_spec=order_spec,
        env={"ALLOW_ORDER_EXECUTION": "1", "MARKET_DATA_ONLY": "0"},
    )

    assert decision.ok is False
    assert "system_role_contract_live_submit_denied" in decision.details["blockers"]
    assert decision.details["system_role_contract_decision"]["ok"] is False


def test_production_firewall_allows_only_qualified_canary_entries(tmp_path: Path) -> None:
    order_spec = _write_firewall_fixture(tmp_path, excellence_ready=True, symbols=["AAPL"])

    allowed = production_order_firewall_check(
        project_root=tmp_path,
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        order_spec=order_spec,
        env={"ALLOW_ORDER_EXECUTION": "1", "MARKET_DATA_ONLY": "0"},
    )
    blocked_spec = json.loads(json.dumps(order_spec))
    blocked_spec["orderLegCollection"][0]["instrument"]["symbol"] = "MSFT"
    blocked = production_order_firewall_check(
        project_root=tmp_path,
        symbol="MSFT",
        action="BUY",
        quantity=1.0,
        order_spec=blocked_spec,
        env={"ALLOW_ORDER_EXECUTION": "1", "MARKET_DATA_ONLY": "0"},
    )

    assert allowed.ok is True
    assert blocked.ok is False
    assert blocked.reason == "symbol_not_in_live_canary_allowlist"


def test_production_firewall_rejects_order_leg_symbol_mismatch(tmp_path: Path) -> None:
    order_spec = _write_firewall_fixture(tmp_path, excellence_ready=True, symbols=["AAPL"])
    order_spec["orderLegCollection"][0]["instrument"]["symbol"] = "MSFT"

    decision = production_order_firewall_check(
        project_root=tmp_path,
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        order_spec=order_spec,
        env={"ALLOW_ORDER_EXECUTION": "1", "MARKET_DATA_ONLY": "0"},
    )

    assert decision.ok is False
    assert "order_symbol_mismatch" in decision.details["blockers"]


def test_production_firewall_requires_reference_price_for_market_entry(tmp_path: Path) -> None:
    order_spec = _write_firewall_fixture(tmp_path, excellence_ready=True, symbols=["AAPL"])
    order_spec.pop("price")

    decision = production_order_firewall_check(
        project_root=tmp_path,
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        order_spec=order_spec,
        env={"ALLOW_ORDER_EXECUTION": "1", "MARKET_DATA_ONLY": "0"},
    )

    assert decision.ok is False
    assert "reference_price_required_for_notional_cap" in decision.details["blockers"]


def test_production_firewall_requires_transition_integrity_when_enabled(tmp_path: Path) -> None:
    order_spec = _write_firewall_fixture(tmp_path, excellence_ready=True, symbols=["AAPL"])
    config_path = tmp_path / "config" / "production_readiness_control_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    policy = config["live_execution_risk_firewall"]
    policy["require_live_transition_integrity_for_live_submit"] = True
    policy["live_transition_integrity_artifact"] = "governance/health/live_transition.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    transition_path = tmp_path / "governance" / "health" / "live_transition.json"
    transition_path.write_text(
        json.dumps({"control_grade": "A+", "ready_for_live_transition": False}),
        encoding="utf-8",
    )

    blocked = production_order_firewall_check(
        project_root=tmp_path,
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        order_spec=order_spec,
        env={"ALLOW_ORDER_EXECUTION": "1", "MARKET_DATA_ONLY": "0"},
    )
    transition_path.write_text(
        json.dumps({"control_grade": "A+", "ready_for_live_transition": True}),
        encoding="utf-8",
    )
    allowed = production_order_firewall_check(
        project_root=tmp_path,
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        order_spec=order_spec,
        env={"ALLOW_ORDER_EXECUTION": "1", "MARKET_DATA_ONLY": "0"},
    )

    assert blocked.ok is False
    assert "live_transition_integrity_not_ready" in blocked.details["blockers"]
    assert allowed.ok is True


def test_production_firewall_rejects_stale_candidate_allowlist(tmp_path: Path) -> None:
    order_spec = _write_firewall_fixture(tmp_path, excellence_ready=True, symbols=["AAPL"])
    allowlist_path = tmp_path / "governance" / "runtime" / "live_canary_allowlist.json"
    allowlist = json.loads(allowlist_path.read_text(encoding="utf-8"))
    allowlist["candidate_id"] = "pc-old-candidate"
    allowlist_path.write_text(json.dumps(allowlist), encoding="utf-8")

    decision = production_order_firewall_check(
        project_root=tmp_path,
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        order_spec=order_spec,
        env={"ALLOW_ORDER_EXECUTION": "1", "MARKET_DATA_ONLY": "0"},
    )

    assert decision.ok is False
    assert "canary_allowlist_candidate_mismatch" in decision.details["blockers"]


def test_production_firewall_requires_pinned_account_reference(tmp_path: Path) -> None:
    order_spec = _write_firewall_fixture(tmp_path, excellence_ready=True, symbols=["AAPL"])
    config_path = tmp_path / "config" / "production_readiness_control_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["live_execution_risk_firewall"]["require_pinned_account_reference"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")

    blocked = production_order_firewall_check(
        project_root=tmp_path,
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        order_spec=order_spec,
        env={"ALLOW_ORDER_EXECUTION": "1", "MARKET_DATA_ONLY": "0", "SCHWAB_ACCOUNT_HASH_AUTO_DISCOVER": "1"},
    )
    allowed = production_order_firewall_check(
        project_root=tmp_path,
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        order_spec=order_spec,
        env={
            "ALLOW_ORDER_EXECUTION": "1",
            "MARKET_DATA_ONLY": "0",
            "SCHWAB_ACCOUNT_HASH": "redacted-test-hash",
            "SCHWAB_ACCOUNT_HASH_AUTO_DISCOVER": "0",
        },
    )

    assert blocked.ok is False
    assert "live_account_reference_not_pinned" in blocked.details["blockers"]
    assert allowed.ok is True


def test_live_risk_config_cannot_exceed_micro_canary_policy(tmp_path: Path, monkeypatch) -> None:
    _write_firewall_fixture(tmp_path, excellence_ready=True, symbols=["AAPL"])
    monkeypatch.setenv("LIVE_MAX_POSITION_QTY_PER_SYMBOL", "250")
    monkeypatch.setenv("LIVE_MAX_ORDER_NOTIONAL", "25000")
    monkeypatch.setenv("LIVE_MAX_OPEN_ORDERS_TOTAL", "30")
    monkeypatch.setenv("LIVE_MAX_OPEN_ORDERS_PER_SYMBOL", "3")
    monkeypatch.setenv("LIVE_MAX_DAILY_LOSS", "1000")
    monkeypatch.setenv("LIVE_MAX_CUMULATIVE_LOSS", "1000")

    config = LiveRiskConfig.from_env(tmp_path)

    assert config.max_position_qty_per_symbol == 1
    assert config.max_order_notional == 100
    assert config.max_open_orders_total == 1
    assert config.max_open_orders_per_symbol == 1
    assert config.daily_loss_cap == 2
    assert config.cumulative_loss_cap == 10
    assert LiveRiskConfig.from_env().cumulative_loss_cap == 0


def test_cumulative_loss_cap_survives_guard_restart(tmp_path: Path) -> None:
    state_path = tmp_path / "live_risk_budget_state.json"
    config = _cfg(
        cumulative_loss_cap=10.0,
        risk_state_path=str(state_path),
        risk_state_candidate_id="pc-test-candidate",
        trade_min_interval_seconds=0.0,
        trade_min_interval_global_seconds=0.0,
    )
    first_guard = LiveExecutionGuard(config)
    first_guard.record_realized_pnl(-10.0, now_ts=1_700_000_000.0)

    restarted_guard = LiveExecutionGuard(config)
    decision = restarted_guard.pre_trade_check(
        symbol="SCHD",
        action="BUY",
        quantity=1.0,
        reference_price=70.0,
        now_ts=1_700_000_001.0,
    )

    assert decision.ok is False
    assert decision.gate == "cumulative_loss_cap"
    assert restarted_guard.snapshot()["realized_pnl_cumulative"] == -10.0


def test_corrupt_persistent_risk_state_fails_closed(tmp_path: Path) -> None:
    state_path = tmp_path / "live_risk_budget_state.json"
    state_path.write_text("not-json", encoding="utf-8")
    guard = LiveExecutionGuard(
        _cfg(
            cumulative_loss_cap=10.0,
            risk_state_path=str(state_path),
            risk_state_candidate_id="pc-test-candidate",
        )
    )

    decision = guard.pre_trade_check(
        symbol="SCHD",
        action="BUY",
        quantity=1.0,
        reference_price=70.0,
    )

    assert decision.ok is False
    assert decision.gate == "persistent_risk_state"
    assert decision.reason.startswith("risk_state_invalid:")


def test_position_limit_blocks_projected_qty():
    guard = LiveExecutionGuard(_cfg(max_position_qty_per_symbol=5.0))
    guard.record_fill(symbol="AAPL", action="BUY", quantity=5.0, fill_price=100.0, now_ts=1000.0)

    decision = guard.pre_trade_check(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        reference_price=100.0,
        now_ts=1010.0,
    )

    assert decision.ok is False
    assert decision.gate == "position_limit"


def test_daily_loss_cap_blocks_new_trade():
    guard = LiveExecutionGuard(_cfg(daily_loss_cap=50.0))
    guard.record_realized_pnl(-60.0, now_ts=1000.0)

    decision = guard.pre_trade_check(
        symbol="MSFT",
        action="BUY",
        quantity=1.0,
        reference_price=200.0,
        now_ts=1010.0,
    )

    assert decision.ok is False
    assert decision.gate == "daily_loss_cap"


def test_trade_throttle_symbol_blocks_fast_reentry():
    guard = LiveExecutionGuard(_cfg(trade_min_interval_seconds=10.0, trade_min_interval_global_seconds=0.0))

    first = guard.pre_trade_check(
        symbol="NVDA",
        action="BUY",
        quantity=1.0,
        reference_price=100.0,
        now_ts=1000.0,
    )
    assert first.ok is True
    guard.mark_trade_submitted(symbol="NVDA", now_ts=1000.0)

    second = guard.pre_trade_check(
        symbol="NVDA",
        action="BUY",
        quantity=1.0,
        reference_price=100.0,
        now_ts=1005.0,
    )
    assert second.ok is False
    assert second.gate == "trade_throttle_symbol"


def test_api_failure_guard_trips_circuit_breaker():
    guard = LiveExecutionGuard(_cfg(api_fail_limit=2, api_cooldown_seconds=30))

    assert guard.allow_api_call("broker_api") is True
    tripped_1 = guard.record_api_failure("broker_api")
    tripped_2 = guard.record_api_failure("broker_api")

    assert tripped_1 is False
    assert tripped_2 is True
    assert guard.allow_api_call("broker_api") is False


def test_open_order_limits_enforced():
    guard = LiveExecutionGuard(_cfg(max_open_orders_total=2, max_open_orders_per_symbol=1))

    guard.register_open_order(order_id="1", symbol="AAPL", action="BUY", quantity=1.0)

    by_symbol = guard.pre_trade_check(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        reference_price=100.0,
        now_ts=1000.0,
    )
    assert by_symbol.ok is False
    assert by_symbol.gate == "open_order_limit_symbol"

    guard.register_open_order(order_id="2", symbol="MSFT", action="BUY", quantity=1.0)
    total = guard.pre_trade_check(
        symbol="NVDA",
        action="BUY",
        quantity=1.0,
        reference_price=100.0,
        now_ts=1001.0,
    )
    assert total.ok is False
    assert total.gate == "open_order_limit_total"


def test_set_local_position_is_used_by_pre_trade_check():
    guard = LiveExecutionGuard(_cfg(max_position_qty_per_symbol=10.0))
    guard.set_local_position(symbol="AAPL", quantity=9.0, avg_price=100.0)

    decision = guard.pre_trade_check(
        symbol="AAPL",
        action="BUY",
        quantity=2.0,
        reference_price=100.0,
        now_ts=1010.0,
    )

    assert decision.ok is False
    assert decision.gate == "position_limit"


def test_slippage_limit_blocks_adverse_buy_price():
    guard = LiveExecutionGuard(_cfg(max_slippage_bps=20.0, trade_min_interval_seconds=0.0, trade_min_interval_global_seconds=0.0))

    decision = guard.pre_trade_check(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        reference_price=100.0,
        intended_price=100.30,
        now_ts=1010.0,
    )

    assert decision.ok is False
    assert decision.gate == "slippage_limit"
    assert float(decision.details.get("adverse_slippage_bps", 0.0)) > 20.0


def test_slippage_limit_allows_favorable_sell_price():
    guard = LiveExecutionGuard(_cfg(max_slippage_bps=20.0, trade_min_interval_seconds=0.0, trade_min_interval_global_seconds=0.0))
    guard.set_local_position(symbol="AAPL", quantity=1.0, avg_price=99.0)

    decision = guard.pre_trade_check(
        symbol="AAPL",
        action="SELL",
        quantity=1.0,
        reference_price=100.0,
        intended_price=100.20,
        now_ts=1010.0,
    )

    assert decision.ok is True


def test_record_fill_includes_modeling_metrics():
    guard = LiveExecutionGuard(_cfg())
    result = guard.record_fill(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        fill_price=100.08,
        reference_price=100.00,
        now_ts=1010.0,
    )

    assert float(result["expected_fill_price"]) > 0.0
    assert float(result["realized_slippage_bps"]) > 0.0
    assert "fill_quality" in result
    assert "fill_deviation_bps" in result["fill_quality"]

    snap = guard.snapshot()
    assert snap["fill_modeling"]["fill_count"] == 1


def test_fill_quality_can_fail_deviation_threshold():
    guard = LiveExecutionGuard(_cfg(max_fill_deviation_bps=5.0))
    quality = guard.evaluate_fill_quality(
        action="BUY",
        actual_fill_price=100.20,
        expected_fill_price=100.00,
    )
    assert quality["ok"] is False
    assert quality["reason"] == "fill_deviation_limit"
    assert float(quality["fill_deviation_bps"]) > 5.0


def test_live_pre_trade_realism_guard_blocks_stale_option_quote():
    guard = LiveExecutionGuard(
        _cfg(
            trade_min_interval_seconds=0.0,
            trade_min_interval_global_seconds=0.0,
            allow_new_short_positions=True,
        )
    )

    decision = guard.pre_trade_check(
        symbol="NVDA_covered_call",
        action="SELL_TO_OPEN",
        quantity=10.0,
        reference_price=4.0,
        now_ts=1010.0,
        enforce_execution_realism=True,
        spread_bps=60.0,
        volatility_1m=0.02,
        latency_ms=500.0,
        bid_size=5.0,
        ask_size=5.0,
        broker="schwab",
        market_kind="options",
        session="regular",
        order_type="limit",
        asset_class="options",
        sleeve="covered_call",
        quote_age_ms=6000.0,
        open_interest=0.0,
    )

    assert decision.ok is False
    assert decision.gate == "execution_realism_guard"
    assert "simulated_stale_quote_rejected" in decision.details["reasons"]


def test_reconcile_order_lifecycle_detects_mismatch_and_position_break():
    guard = LiveExecutionGuard(_cfg())
    guard.register_open_order(order_id="o1", symbol="AAPL", action="BUY", quantity=1.0)
    guard.set_local_position(symbol="AAPL", quantity=2.0, avg_price=100.0)
    guard.reconcile_broker_position(symbol="AAPL", broker_qty=1.0)

    out = guard.reconcile_order_lifecycle(
        broker_open_orders=[{"order_id": "o2", "symbol": "AAPL"}],
        position_tolerance=0.0001,
    )

    assert out["ok"] is False
    assert out["missing_on_broker"] == ["o1"]
    assert out["missing_local"] == ["o2"]
    assert out["position_checks"]
    assert out["position_checks"][0]["ok"] is False



def test_reconcile_broker_position_marks_manual_adjustment_window():
    guard = LiveExecutionGuard(_cfg())
    guard.set_local_position(symbol="AAPL", quantity=0.0, avg_price=0.0)

    out = guard.reconcile_broker_position(
        symbol="AAPL",
        broker_qty=1.0,
        tolerance=0.0001,
        manual_adjustment_tolerance=2.0,
    )

    assert out["ok"] is False
    assert out["manual_adjustment_detected"] is True
    assert out["status"] == "manual_adjustment_detected"
