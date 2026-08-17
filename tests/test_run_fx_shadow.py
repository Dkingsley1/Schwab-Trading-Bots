from scripts import run_fx_shadow as src


def test_realtime_interval_seconds_counts_unique_realtime_and_context_symbols() -> None:
    env = {
        "FX_TWELVE_DATA_MAX_CREDITS_PER_MINUTE": "8",
        "FX_TWELVE_DATA_CREDIT_RESERVE": "3",
    }

    duplicate_context_interval = src._realtime_interval_seconds(
        env,
        requested_interval=15,
        realtime_symbols="EURUSD,USDJPY,GBPUSD",
        realtime_context_symbols="EURUSD,USDJPY",
    )
    expanded_context_interval = src._realtime_interval_seconds(
        env,
        requested_interval=15,
        realtime_symbols="EURUSD,USDJPY,GBPUSD",
        realtime_context_symbols="EURUSD,USDJPY,AUDUSD",
    )

    assert duplicate_context_interval == 36
    assert expanded_context_interval == 48


def test_budgeted_realtime_symbols_prefers_context_pairs_and_defers_excess() -> None:
    symbols, budget = src._budgeted_realtime_symbols(
        {
            "FX_TWELVE_DATA_MAX_CREDITS_PER_MINUTE": "4",
            "FX_TWELVE_DATA_CREDIT_RESERVE": "2",
            "FX_TWELVE_DATA_MAX_PAIRS_PER_RUN": "6",
        },
        realtime_symbols="EURUSD,USDJPY,GBPUSD,USDCHF",
        realtime_context_symbols="USDJPY,EURUSD",
    )

    assert symbols == "USDJPY,EURUSD"
    assert budget["selected_symbols"] == ["USDJPY", "EURUSD"]
    assert budget["deferred_symbols"] == ["GBPUSD", "USDCHF"]


def test_fx_supervisor_mode_falls_back_to_context_only_during_provider_cooldown() -> None:
    mode = src._fx_supervisor_mode(
        proxy_session_open=False,
        forex_session_open=True,
        off_hours_only=True,
        provider_status={
            "enabled": True,
            "available": False,
            "reason": "provider_cooldown:daily_quota",
            "cooldown": {"kind": "daily_quota", "remaining_seconds": 321.0},
        },
        default_symbols="UUP,FXE",
        default_context_symbols="SPY,QQQ",
        realtime_symbols="EURUSD,USDJPY",
        realtime_context_symbols="EURUSD,USDJPY",
    )

    assert mode["live"] is False
    assert mode["mode"] == "forex_session_context_only"
    assert str(mode["reason"]).startswith("twelve_data_daily_quota_cooldown")
    assert mode["symbols"] == "UUP,FXE"


def test_fx_supervisor_mode_uses_live_quotes_when_provider_available() -> None:
    mode = src._fx_supervisor_mode(
        proxy_session_open=False,
        forex_session_open=True,
        off_hours_only=True,
        provider_status={"enabled": True, "available": True, "reason": "available", "cooldown": {}},
        default_symbols="UUP,FXE",
        default_context_symbols="SPY,QQQ",
        realtime_symbols="EURUSD,USDJPY",
        realtime_context_symbols="EURUSD",
    )

    assert mode["live"] is True
    assert mode["mode"] == "live_forex_quotes"
    assert mode["symbols"] == "EURUSD,USDJPY"
