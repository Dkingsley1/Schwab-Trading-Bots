from core.brokers.capability_contract import (
    all_adapter_conformance,
    evaluate_order_request,
    load_capability_contracts,
)


def _request(asset_type: str = "EQUITY", quantity: float = 1.0) -> dict:
    return {
        "symbol": "SPY",
        "action": "BUY",
        "quantity": quantity,
        "asset_type": asset_type,
        "order_spec": {"orderType": "MARKET", "duration": "DAY"},
    }


def test_all_broker_adapters_conform_to_detailed_contract() -> None:
    assert all_adapter_conformance()["ok"] is True


def test_schwab_paper_futures_are_simulated_but_not_live_authorized() -> None:
    contracts = load_capability_contracts()
    paper = evaluate_order_request(
        "schwab", _request("FUTURE"), mode="paper", contracts=contracts
    )
    live = evaluate_order_request(
        "schwab",
        _request("FUTURE"),
        mode="live",
        contracts=contracts,
        require_production_eligible=True,
    )

    assert paper["ok"] is True
    assert paper["implementation"] == "local_realism_simulator"
    assert live["ok"] is False
    assert "asset_class_not_supported_for_live:FUTURE" in live["reasons"]


def test_market_data_only_and_test_double_cannot_become_production_execution() -> None:
    coinbase = evaluate_order_request(
        "coinbase", _request("CRYPTO"), mode="live", require_production_eligible=True
    )
    mock = evaluate_order_request(
        "mock", _request(), mode="live", require_production_eligible=True
    )

    assert coinbase["ok"] is False
    assert "live_execution_not_supported" in coinbase["reasons"]
    assert mock["ok"] is False
    assert "broker_mode_not_production_eligible" in mock["reasons"]


def test_schwab_complex_options_and_roll_action_are_explicitly_supported() -> None:
    request = {
        "symbol": "SPY",
        "action": "ROLL",
        "quantity": 1.0,
        "asset_type": "OPTION",
        "order_spec": {
            "orderType": "NET_DEBIT",
            "duration": "DAY",
            "price": 1.25,
            "orderLegCollection": [
                {
                    "instruction": "BUY_TO_OPEN",
                    "quantity": 1,
                    "instrument": {"symbol": "SPY_OPTION", "assetType": "OPTION"},
                }
            ],
        },
    }

    assert (
        evaluate_order_request(
            "schwab", request, mode="live", require_production_eligible=True
        )["ok"]
        is True
    )
