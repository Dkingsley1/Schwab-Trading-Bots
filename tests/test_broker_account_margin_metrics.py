from scripts import run_shadow_training_loop as loop


def test_extract_account_metrics_prefers_current_balances_over_initial_day_trading_power() -> None:
    payload = {
        "securitiesAccount": {
            "initialBalances": {
                "availableFundsNonMarginableTrade": 214.06,
                "buyingPower": 214.06,
                "cashBalance": 214.06,
                "dayTradingBuyingPower": 119676.0,
                "equity": 39443.95,
                "liquidationValue": 39443.95,
                "maintenanceRequirement": 39607.0,
            },
            "currentBalances": {
                "cashBalance": 270.51,
                "liquidationValue": 39289.06,
                "availableFunds": 214.06,
                "buyingPower": 214.06,
                "dayTradingBuyingPower": 214.06,
                "equity": 40165.06,
                "maintenanceRequirement": 39894.55,
            },
            "projectedBalances": {
                "availableFunds": 214.06,
                "buyingPower": 214.06,
                "dayTradingBuyingPower": 214.06,
                "stockBuyingPower": 214.06,
            },
        }
    }

    metrics = loop._extract_account_metrics(payload)

    assert metrics["buying_power"] == 214.06
    assert metrics["available_funds"] == 214.06
    assert metrics["cash_balance"] == 270.51
    assert metrics["maintenance_margin_requirement"] == 39894.55


def test_broker_margin_available_proxy_uses_available_funds_before_cash_or_large_buying_power() -> None:
    broker_truth = {
        "account_metrics": {
            "available_funds": 214.06,
            "buying_power": 119676.0,
            "cash_balance": 270.51,
            "equity": 40165.06,
        }
    }

    available = loop._broker_margin_available_proxy(broker_truth, "options")

    assert round(available, 3) == round(214.06 * 0.70, 3)


def test_options_margin_proxy_uses_crypto_contract_multiplier(monkeypatch) -> None:
    monkeypatch.delenv("CRYPTO_OPTIONS_CONTRACT_MULTIPLIER", raising=False)
    decision = {
        "action": "BUY_TO_OPEN",
        "plan": {
            "symbol": "BTC-USD",
            "options_style": "BEAR_PUT_DEBIT_SPREAD",
            "strategy_family": "debit_spread",
            "underlying_price": 76000.0,
            "contracts": 1,
            "legs": [
                {"side": "BUY_TO_OPEN", "type": "PUT", "strike": 76000.0, "quantity": 1},
                {"side": "SELL_TO_OPEN", "type": "PUT", "strike": 72200.0, "quantity": 1},
            ],
        },
    }

    required = loop._estimate_options_margin_proxy(decision)

    assert required == 3800.0


def test_options_margin_guard_is_advisory_for_shadow_without_broker_truth(monkeypatch) -> None:
    monkeypatch.setenv("MARKET_DATA_ONLY", "1")
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "0")
    decision = {
        "action": "BUY_TO_OPEN",
        "score": 0.62,
        "threshold": 0.58,
        "reasons": ["bear_put_debit_spread"],
        "plan": {
            "symbol": "BTC-USD",
            "options_style": "BEAR_PUT_DEBIT_SPREAD",
            "strategy_family": "debit_spread",
            "underlying_price": 76000.0,
            "contracts": 1,
            "legs": [
                {"side": "BUY_TO_OPEN", "type": "PUT", "strike": 76000.0, "quantity": 1},
                {"side": "SELL_TO_OPEN", "type": "PUT", "strike": 72200.0, "quantity": 1},
            ],
        },
    }

    out, meta = loop._apply_derivatives_margin_guard(
        decision=decision,
        lane="options",
        broker_truth={"status": "disabled", "account_metrics": {}},
        features={},
    )

    assert out["action"] == "BUY_TO_OPEN"
    assert meta["ok"] is True
    assert meta["advisory"] is True
    assert meta["execution_enabled"] is False
    assert meta["reason"] == "shadow_no_broker_margin_truth"


def test_options_margin_guard_blocks_execution_without_broker_truth(monkeypatch) -> None:
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    decision = {
        "action": "BUY_TO_OPEN",
        "score": 0.62,
        "threshold": 0.58,
        "reasons": ["bear_put_debit_spread"],
        "plan": {
            "symbol": "BTC-USD",
            "options_style": "BEAR_PUT_DEBIT_SPREAD",
            "strategy_family": "debit_spread",
            "underlying_price": 76000.0,
            "contracts": 1,
            "legs": [
                {"side": "BUY_TO_OPEN", "type": "PUT", "strike": 76000.0, "quantity": 1},
                {"side": "SELL_TO_OPEN", "type": "PUT", "strike": 72200.0, "quantity": 1},
            ],
        },
    }

    out, meta = loop._apply_derivatives_margin_guard(
        decision=decision,
        lane="options",
        broker_truth={"status": "disabled", "account_metrics": {}},
        features={},
    )

    assert out["action"] == "HOLD"
    assert meta["ok"] is False
    assert meta["execution_enabled"] is True
    assert meta["reason"] == "broker_margin_truth_unavailable"
