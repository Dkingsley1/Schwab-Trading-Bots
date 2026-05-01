# --- THE CONTROL ROOM ---

# --- THE BRAIN (Apple Silicon Optimized) ---
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# --- THE PLUMBING (Schwab & Data) ---
from core.base_trader import BaseTrader
from core.brokers import BrokerRuntimeConfig
from core.runtime_config import RuntimeConfig
import pandas as pd
import numpy as np

# --- THE OPTIONS MATH ---
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.numerical import delta, theta, gamma
import pandas_ta as ta

# --- SYSTEM TOOLS ---
import sqlite3  # To store your "Failure Logs" locally
import datetime
import os

def _enforce_market_data_only_lock() -> tuple[str, str]:
    market_data_only = os.getenv("MARKET_DATA_ONLY", "1").strip()
    allow_order_execution = os.getenv("ALLOW_ORDER_EXECUTION", "0").strip()

    if market_data_only != "1" or allow_order_execution != "0":
        raise RuntimeError(
            "Refusing to start: requires MARKET_DATA_ONLY=1 and ALLOW_ORDER_EXECUTION=0"
        )

    return market_data_only, allow_order_execution


def _run_decision_probe(trader: BaseTrader):
    """Runs one synthetic decision through the routing layer for validation."""
    result = trader.execute_decision(
        symbol="SPY",
        action="BUY",
        quantity=1,
        model_score=0.62,
        threshold=0.55,
        features={"rsi": 48.2, "drawdown": -0.01, "vol10": 0.012},
        gates={"risk_limit_ok": True, "spread_ok": True, "session_open": True},
        reasons=["score_above_threshold", "risk_checks_passed"],
        strategy="bootstrap_probe",
        metadata={"source": "main_startup_probe"},
    )
    print(f"Decision probe status: {result['status']} (mode={result['mode']})")


def start_factory():
    market_data_only, allow_order_execution = _enforce_market_data_only_lock()

    cfg = RuntimeConfig.from_env()
    broker_runtime = BrokerRuntimeConfig.from_env()
    print(f"Startup mode: {cfg.mode}")
    print(
        "Safety config: "
        f"MARKET_DATA_ONLY={market_data_only} "
        f"ALLOW_ORDER_EXECUTION={allow_order_execution}"
    )
    print(
        "Broker roles: "
        f"auth={broker_runtime.auth_broker_name} "
        f"market_data={broker_runtime.market_data_provider_name} "
        f"paper_execution={broker_runtime.paper_execution_broker_name} "
        f"live_execution={broker_runtime.execution_broker_name}"
    )

    # BaseTrader handles auth/data plumbing and decision routing.
    mecca = BaseTrader.from_env(mode=cfg.mode, role="auth", runtime_config=broker_runtime)

    if mecca.credentials_are_placeholder():
        print(f"Credentials are placeholders. Skipping {mecca.broker_display_name} handshake.")
        _run_decision_probe(mecca)
        return

    mecca.authenticate()

    try:
        account_rows = mecca.fetch_connected_account_rows()
        if account_rows:
            account_number = str(account_rows[0].get("account_number") or "").strip() or "unknown"
            print(f"Bot Factory Online. Connected to {mecca.broker_display_name} account: {account_number}")
        else:
            print(f"Bot Factory Online. Connected to {mecca.broker_display_name}.")
    except Exception as e:
        print(f"Connection established, but could not fetch account info: {e}")

    if os.getenv("RUN_DECISION_PROBE", "1") == "1":
        _run_decision_probe(mecca)


if __name__ == "__main__":
    start_factory()
