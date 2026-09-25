from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.ops import library_research_extensions as src


def prices(values):
    return pd.Series(
        values,
        index=pd.date_range("2024-01-01", periods=len(values), freq="h"),
        dtype=float,
    )


@pytest.mark.parametrize(
    "equity,total,drawdown",
    [
        ([10000, 11000, 9900, 10500], 0.05, -0.1),
        ([9000, 8000, 8500], -0.15, -0.2),
        ([10000, 10000, 10000], 0.0, 0.0),
        ([10000, 40000, 20000], 1.0, -0.5),
    ],
)
def test_performance_reconciles_cash_and_is_not_annualized(equity, total, drawdown):
    result = src.performance_diagnostics(prices(equity), starting_cash=10000)
    assert result["equity_reconstruction_parity"]
    assert result["total_return_fraction"] == pytest.approx(total)
    assert result["max_drawdown_fraction"] == pytest.approx(drawdown)
    assert result["annualized"] is False
    assert result["trade_win_rate"] is None
    assert result["return_period_count"] == len(equity)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("values", [[10000, 0], [10000, -1], [10000, float("nan")]])
def test_invalid_equity_rejected(values):
    with pytest.raises(ValueError, match="positive_finite"):
        src.performance_diagnostics(prices(values), starting_cash=10000)


@pytest.mark.parametrize(
    "values,status",
    [
        (np.arange(40) + 100, "insufficient_history"),
        (np.full(180, 50), "insufficient_variation"),
    ],
)
def test_unavailable_volatility_does_not_fit(values, status, monkeypatch):
    import arch

    def forbidden(*args, **kwargs):
        raise AssertionError("must not fit unavailable inputs")

    monkeypatch.setattr(arch, "arch_model", forbidden)
    result = src.volatility_diagnostics(prices(values))
    assert result["status"] == status
    assert result["converged"] is None
    assert result["next_bar_volatility"] is None


def test_volatility_caps_history_iterations_and_rejects_nonconvergence(monkeypatch):
    import arch

    seen = {}

    class Model:
        def fit(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(convergence_flag=9)

    def model(returns, **kwargs):
        seen["rows"] = len(returns)
        seen["spec"] = kwargs
        return Model()

    monkeypatch.setattr(arch, "arch_model", model)
    values = 100 + np.sin(np.arange(1500) / 10)
    result = src.volatility_diagnostics(prices(values))
    assert seen["rows"] == 1000
    assert seen["options"]["maxiter"] == 100
    assert seen["spec"]["p"] == seen["spec"]["q"] == 1
    assert result["status"] == "not_converged"
    assert result["converged"] is False
    assert result["next_bar_volatility"] is None


@pytest.mark.parametrize(
    "variance,status",
    [
        (-1.0, "invalid_forecast"),
        (float("nan"), "invalid_forecast"),
        (4.0, "estimated"),
    ],
)
def test_volatility_forecast_units_and_rejection(monkeypatch, variance, status):
    import arch

    forecast = SimpleNamespace(variance=pd.DataFrame([[variance]]))
    fitted = SimpleNamespace(
        convergence_flag=0,
        forecast=lambda **_: forecast,
        params={"alpha[1]": 0.1, "beta[1]": 0.8},
    )
    monkeypatch.setattr(
        arch, "arch_model", lambda *_, **__: SimpleNamespace(fit=lambda **_: fitted)
    )
    result = src.volatility_diagnostics(prices(100 + np.sin(np.arange(180))))
    assert result["status"] == status
    assert result["next_bar_volatility"] == (
        pytest.approx(0.02) if status == "estimated" else None
    )
    assert result["out_of_sample_validated"] is False


def test_garch_fit_on_seeded_variable_returns():
    rng = np.random.default_rng(14)
    values = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 300)))
    result = src.volatility_diagnostics(prices(values))
    assert result["status"] == "estimated", result
    assert result["next_bar_volatility"] > 0


def test_native_indicator_parity_optional():
    pytest.importorskip("pandas_ta_classic")
    import talib
    from scripts.ops.library_research import fixture

    frame = pd.DataFrame(fixture()["candles"])
    close = frame["close"].to_numpy()
    result = src.indicator_diagnostics(
        frame, talib.SMA(close, 10), talib.SMA(close, 30)
    )
    assert result["sma_parity"] and result["sma_prefix_invariance"]
    mismatch = src.indicator_diagnostics(
        frame, talib.SMA(close, 10) + 1, talib.SMA(close, 30)
    )
    assert mismatch["sma_parity"] is False
