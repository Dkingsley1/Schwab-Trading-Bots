import importlib.util
from pathlib import Path

from core import runtime_python as src


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path):
    module_name = f"test_loader_{path.stem}_{path.stat().st_mtime_ns}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shadow_launchers_default_to_python314_after_runtime_flip(monkeypatch) -> None:
    monkeypatch.delenv("BOT_RUNTIME_LANE", raising=False)
    monkeypatch.delenv("BOT_SHADOW_RUNTIME_LANE", raising=False)
    monkeypatch.delenv("BOT_PYTHON_VERSION", raising=False)
    monkeypatch.setattr(src, "_python_supports_module", lambda path_text, module_name: ".venv312/bin/python" in path_text and module_name == "mlx")

    launcher_paths = [
        PROJECT_ROOT / "scripts" / "run_dividend_shadow.py",
        PROJECT_ROOT / "scripts" / "run_dividend_capture_shadow.py",
        PROJECT_ROOT / "scripts" / "run_dividend_compound_shadow.py",
        PROJECT_ROOT / "scripts" / "run_long_term_core_etf_shadow.py",
        PROJECT_ROOT / "scripts" / "run_long_term_dividend_compound_shadow.py",
        PROJECT_ROOT / "scripts" / "run_long_term_sector_rotation_shadow.py",
        PROJECT_ROOT / "scripts" / "run_specialized_sleeve_shadow.py",
        PROJECT_ROOT / "scripts" / "run_volatility_shadow.py",
        PROJECT_ROOT / "scripts" / "run_pairs_correlation_shadow.py",
        PROJECT_ROOT / "scripts" / "run_stat_arb_market_neutral_shadow.py",
        PROJECT_ROOT / "scripts" / "run_earnings_event_shadow.py",
        PROJECT_ROOT / "scripts" / "run_commodity_inflation_shadow.py",
        PROJECT_ROOT / "scripts" / "run_international_macro_shadow.py",
        PROJECT_ROOT / "scripts" / "run_market_making_liquidity_shadow.py",
        PROJECT_ROOT / "scripts" / "run_short_bias_hedge_shadow.py",
        PROJECT_ROOT / "scripts" / "run_single_name_options_event_shadow.py",
        PROJECT_ROOT / "scripts" / "run_rates_credit_macro_shadow.py",
        PROJECT_ROOT / "scripts" / "run_cash_rotation_tactical_shadow.py",
        PROJECT_ROOT / "scripts" / "run_futures_index_intraday_shadow.py",
        PROJECT_ROOT / "scripts" / "run_futures_rates_curve_shadow.py",
        PROJECT_ROOT / "scripts" / "run_futures_commodity_macro_shadow.py",
        PROJECT_ROOT / "scripts" / "run_crypto_futures_basis_shadow.py",
        PROJECT_ROOT / "scripts" / "run_futures_event_reaction_shadow.py",
        PROJECT_ROOT / "scripts" / "run_options_on_futures_shadow.py",
        PROJECT_ROOT / "scripts" / "run_options_on_futures_aggressive_shadow.py",
    ]

    for path in launcher_paths:
        module = _load_module(path)
        assert ".venv314/bin/python" in str(module.VENV_PY)

    failover_module = _load_module(PROJECT_ROOT / "scripts" / "failover_hot_standby.py")
    assert ".venv314/bin/python" in str(failover_module.RUNTIME_PY)


def test_shadow_launchers_honor_explicit_portable_override(monkeypatch) -> None:
    monkeypatch.setenv("BOT_RUNTIME_LANE", "shadow314")
    monkeypatch.delenv("BOT_SHADOW_RUNTIME_LANE", raising=False)
    monkeypatch.setattr(src, "_python_supports_module", lambda path_text, module_name: ".venv312/bin/python" in path_text and module_name == "mlx")

    module = _load_module(PROJECT_ROOT / "scripts" / "run_dividend_shadow.py")

    assert ".venv314/bin/python" in str(module.VENV_PY)


def test_failover_hot_standby_defaults_to_live_data_standby() -> None:
    module = _load_module(PROJECT_ROOT / "scripts" / "failover_hot_standby.py")

    cmd = module._default_standby_cmd()

    assert "opsctl.sh feed-refresh --source schwab --paper" in cmd
    assert "--simulate" not in cmd
    assert module._simulate_disallowed("python scripts/run_parallel_shadows.py --simulate", False) is True
    assert module._simulate_disallowed("python scripts/run_parallel_shadows.py --simulate", True) is False


def test_failover_hot_standby_respects_swap_research_pause(monkeypatch, tmp_path: Path) -> None:
    module = _load_module(PROJECT_ROOT / "scripts" / "failover_hot_standby.py")
    swap_override = tmp_path / ".env.swap_pressure_override"
    swap_override.write_text(
        "SWAP_PRESSURE_TIER=pause_research\n"
        "SWAP_PRESSURE_SWAP_USED_GB=19.059\n"
        "SWAP_PRESSURE_HEAVY_RESEARCH_PAUSED=1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "SWAP_OVERRIDE_PATH", swap_override)
    monkeypatch.setattr(module, "MEMORY_OVERRIDE_PATH", tmp_path / "missing.env")
    monkeypatch.delenv("SWAP_PRESSURE_HEAVY_RESEARCH_PAUSED", raising=False)
    monkeypatch.delenv("TRAINING_RUNTIME_PAUSED_FOR_SWAP", raising=False)

    state = module._swap_research_pause_state()

    assert state["active"] is True
    assert state["tier"] == "pause_research"
    assert state["swap_used_gb"] == "19.059"


def test_failover_hot_standby_suppresses_standby_when_live_parent_is_active() -> None:
    module = _load_module(PROJECT_ROOT / "scripts" / "failover_hot_standby.py")
    start_attempts: list[str] = []

    event = module._build_failover_event(
        primary_alive=False,
        live_parent_alive=True,
        heartbeat_age_sec=999.0,
        max_heartbeat_age_sec=150.0,
        swap_pause={"active": False},
        standby_cmd="scripts/ops/opsctl.sh feed-refresh --source schwab --paper",
        allow_simulate=False,
        start_cmd=start_attempts.append,
    )

    assert event["action"] == "live_parent_active_primary_stale"
    assert event["standby_skip_reason"] == "live_parent_alive"
    assert event["standby_ok"] is False
    assert start_attempts == []
