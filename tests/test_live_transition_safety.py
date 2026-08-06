from core.live_transition_safety import canary_stage_contract, evaluate_release_interlock, reconcile_broker_truth
from scripts.ops import live_transition_chaos_harness
from scripts.ops import live_transition_integrity_control


def _ready_signals() -> dict:
    return {
        "restart_reconciled": True,
        "auth_ready": True,
        "auth_generation_stable": True,
        "quote_fresh": True,
        "sources_ready": True,
        "reconciliation_clean": True,
        "drawdown_within_limit": True,
        "storage_ready": True,
        "production_ready": True,
        "operator_release_present": True,
        "exit_route_ready": True,
        "broker_reachable": True,
    }


def test_each_entry_fault_relocks_without_disabling_valid_exit() -> None:
    signal_by_fault = {
        "restart_state_unreconciled": "restart_reconciled",
        "auth_generation_changed": "auth_generation_stable",
        "quote_stale": "quote_fresh",
        "decision_source_degraded": "sources_ready",
        "broker_reconciliation_mismatch": "reconciliation_clean",
        "drawdown_limit_breached": "drawdown_within_limit",
        "durable_storage_unavailable": "storage_ready",
        "production_evidence_not_ready": "production_ready",
        "operator_release_missing": "operator_release_present",
    }
    for expected_fault, signal in signal_by_fault.items():
        signals = _ready_signals()
        signals[signal] = False
        result = evaluate_release_interlock(signals)
        assert result["entry_allowed"] is False
        assert expected_fault in result["entry_lock_reasons"]
        assert result["risk_reducing_exit_allowed"] is True


def test_auth_failure_relocks_entries_and_exits() -> None:
    signals = _ready_signals()
    signals["auth_ready"] = False

    result = evaluate_release_interlock(signals, risk_reducing_exit=True)

    assert result["entry_allowed"] is False
    assert result["risk_reducing_exit_allowed"] is False
    assert "auth_not_ready_for_exit" in result["exit_blockers"]


def test_reconciliation_reports_all_five_surfaces() -> None:
    local = {
        "orders": [{"order_id": "one", "status": "open"}],
        "fills": [{"order_id": "one", "filled_quantity": 0.0}],
        "positions": [{"symbol": "SPY", "quantity": 1.0}],
        "buying_power": 100.0,
        "cancels": [{"order_id": "two", "status": "canceled"}],
    }
    broker = {
        "orders": [{"order_id": "one", "status": "filled"}],
        "fills": [{"order_id": "one", "filled_quantity": 1.0}],
        "positions": [{"symbol": "SPY", "quantity": 2.0}],
        "buying_power": 50.0,
        "cancels": [{"order_id": "two", "status": "open"}],
    }

    result = reconcile_broker_truth(local, broker)

    assert result["ok"] is False
    assert set(result["mismatch_count_by_surface"]) == {"orders", "fills", "positions", "buying_power", "cancels"}


def test_canary_is_microscopic_and_never_auto_scales() -> None:
    ready = canary_stage_contract(
        requested_weight=0.0025,
        clean_evidence_windows=1,
        sleeve_count=1,
        open_position_count=0,
    )
    oversized = canary_stage_contract(
        requested_weight=0.0101,
        clean_evidence_windows=100,
        sleeve_count=1,
        open_position_count=0,
    )

    assert ready["ok"] is True
    assert ready["automatic_scaling_allowed"] is False
    assert oversized["ok"] is False


def test_transition_chaos_harness_covers_all_required_faults(tmp_path) -> None:
    payload = live_transition_chaos_harness.build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["grade"] == "A+"
    assert payload["scenario_count"] == 7


def test_transition_auth_signals_accept_current_nested_supervisor_schema() -> None:
    signals = live_transition_integrity_control._auth_runtime_signals(
        {
            "ok": True,
            "overall_status": "ready",
            "token": {"ready": True, "refresh_needed": False},
            "broker_readiness": {"auth_ok": True, "network_ok": True},
        }
    )

    assert signals == {
        "auth_ready": True,
        "auth_generation_stable": True,
        "broker_reachable": True,
    }


def test_transition_auth_signals_fail_closed_on_nested_refresh_need() -> None:
    signals = live_transition_integrity_control._auth_runtime_signals(
        {
            "ok": True,
            "overall_status": "ready",
            "token": {"ready": True, "refresh_needed": True},
            "broker_readiness": {"auth_ok": True, "network_ok": True},
        }
    )

    assert signals["auth_ready"] is True
    assert signals["auth_generation_stable"] is False
