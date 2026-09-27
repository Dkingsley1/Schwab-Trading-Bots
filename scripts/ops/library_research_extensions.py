"""Offline diagnostics only; imported inside the bounded research worker."""

from __future__ import annotations

import math
import warnings


def indicator_diagnostics(data, fast, slow):
    import numpy as np
    import pandas_ta_classic as ta

    close = data["close"]
    native_fast = ta.sma(close, length=10, talib=False)
    native_slow = ta.sma(close, length=30, talib=False)
    prefix_size = len(close) - 5
    prefix = ta.sma(close.iloc[:prefix_size], length=30, talib=False)
    parity = bool(
        np.allclose(native_fast, fast, rtol=1e-10, atol=1e-8, equal_nan=True)
        and np.allclose(native_slow, slow, rtol=1e-10, atol=1e-8, equal_nan=True)
    )
    causal = bool(np.allclose(prefix, native_slow.iloc[:prefix_size], equal_nan=True))
    values = {
        "sma10": float(native_fast.iloc[-1]),
        "sma30": float(native_slow.iloc[-1]),
        "rsi14": float(ta.rsi(close, length=14, talib=False).iloc[-1]),
        "atr14": float(
            ta.atr(data["high"], data["low"], close, length=14, talib=False).iloc[-1]
        ),
    }
    return {
        "backend": "pandas_ta_classic_native_talib_false",
        "sma_parity": parity,
        "sma_prefix_invariance": causal,
        **{
            name: value if math.isfinite(value) else None
            for name, value in values.items()
        },
        "unavailable_indicators": [
            name for name, value in values.items() if not math.isfinite(value)
        ],
        "scope": "fixed_causal_indicators_not_all_indicator_search",
    }


def timing_control(data, wanted, *, cash, bar_seconds):
    import numpy as np
    import vectorbt as vbt
    from backtesting import Backtest, Strategy

    instances = []

    class TimingStrategy(Strategy):
        def init(self):
            instances.append(self)

        def next(self):
            target = bool(wanted[len(self.data) - 1])
            if target and not self.position:
                self.buy(size=1)
            elif not target and self.position:
                self.position.close()

    with warnings.catch_warnings(record=True) as notices:
        result = Backtest(
            data.rename(columns=str.title),
            TimingStrategy,
            cash=cash,
            spread=0.0,
            commission=0.0,
            margin=1.0,
            trade_on_close=False,
            hedging=False,
            exclusive_orders=True,
            finalize_trades=False,
        ).run()
    reference = vbt.Portfolio.from_signals(
        data["close"],
        np.r_[False, wanted[:-1]],
        np.r_[False, ~wanted[:-1]],
        price=data["open"],
        size=1,
        init_cash=cash,
        fees=0.0,
        slippage=0.0,
        accumulate=False,
        direction="longonly",
        freq=f"{bar_seconds}s",
    )
    strategy = instances[0]
    fills = []
    for trade in (*strategy.closed_trades, *strategy.trades):
        fills.append((trade.entry_bar, 1, float(trade.entry_price)))
        if trade.exit_bar is not None:
            fills.append((trade.exit_bar, -1, float(trade.exit_price)))
    fills.sort()
    expected = [
        (int(row.idx), 1 if int(row.side) == 0 else -1, float(row.price))
        for row in reference.orders.records.itertuples()
    ]
    equity = float(result["Equity Final [$]"])
    reference_equity = float(reference.final_value())
    parity = (
        len(fills) == len(expected)
        and all(
            a[:2] == b[:2] and math.isclose(a[2], b[2], rel_tol=1e-10, abs_tol=1e-8)
            for a, b in zip(fills, expected)
        )
        and math.isclose(equity, reference_equity, rel_tol=1e-10, abs_tol=1e-7)
    )
    return {
        "scope": "zero_cost_next_open_timing_control_only",
        "costs_included": False,
        "backtesting_final_equity": equity,
        "vectorbt_zero_cost_final_equity": reference_equity,
        "fill_count": len(fills),
        "open_positions": len(strategy.trades),
        "fill_timing_price_and_equity_parity": bool(parity),
        "warning_categories": sorted({type(w.message).__name__ for w in notices}),
    }


def performance_diagnostics(equity, *, starting_cash):
    import numpy as np
    import pandas as pd
    import quantstats as qs

    values = np.asarray(equity, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any() or starting_cash <= 0:
        raise ValueError("positive_finite_equity_required")
    returns = equity.pct_change(fill_method=None)
    returns.iloc[0] = values[0] / starting_cash - 1
    if not np.isfinite(returns).all():
        raise ValueError("finite_period_returns_required")
    total = float(qs.stats.comp(returns))
    # Explicit price baseline avoids QuantStats' return/price scale heuristics.
    normalized = equity / starting_cash * 100_000
    baseline = pd.Series(
        [100_000.0], index=[equity.index[0] - pd.Timedelta(nanoseconds=1)]
    )
    drawdown = float(qs.stats.max_drawdown(pd.concat([baseline, normalized])))
    volatility = float(
        qs.stats.volatility(returns, annualize=False, prepare_returns=False)
    )
    expected_total = values[-1] / starting_cash - 1
    peaks = np.maximum.accumulate(np.r_[starting_cash, values])[1:]
    expected_drawdown = float(np.min(values / peaks - 1))
    parity = math.isclose(total, expected_total, abs_tol=1e-10) and math.isclose(
        drawdown, expected_drawdown, abs_tol=1e-10
    )
    return {
        "backend": "quantstats",
        "source": "cost_inclusive_vectorbt_equity",
        "return_period_count": len(returns),
        "total_return_fraction": total,
        "max_drawdown_fraction": drawdown,
        "volatility_per_observed_bar": volatility,
        "positive_bar_fraction": float((returns > 0).mean()),
        "equity_reconstruction_parity": bool(parity),
        "annualized": False,
        "trade_win_rate": None,
    }


def volatility_diagnostics(close):
    import numpy as np
    from arch import arch_model

    returns = close.pct_change(fill_method=None).dropna().iloc[-1000:] * 100
    base = {
        "backend": "arch",
        "model": "zero_mean_GARCH_1_1_normal",
        "observed_return_count": len(returns),
        "max_observations": 1000,
        "max_optimizer_iterations": 100,
        "out_of_sample_validated": False,
        "forecast_units": "fraction_per_next_observed_bar_not_annualized",
    }
    if len(returns) < 100:
        return {
            **base,
            "status": "insufficient_history",
            "converged": None,
            "next_bar_volatility": None,
        }
    if not np.isfinite(returns).all():
        raise ValueError("nonfinite_volatility_returns")
    if float(returns.std()) < 1e-8:
        return {
            **base,
            "status": "insufficient_variation",
            "converged": None,
            "next_bar_volatility": None,
        }
    with warnings.catch_warnings(record=True) as notices:
        fitted = arch_model(
            returns, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False
        ).fit(disp="off", update_freq=0, show_warning=False, options={"maxiter": 100})
    converged = int(fitted.convergence_flag) == 0
    if not converged:
        return {
            **base,
            "status": "not_converged",
            "converged": False,
            "next_bar_volatility": None,
        }
    variance = float(fitted.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
    if not math.isfinite(variance) or variance < 0:
        return {
            **base,
            "status": "invalid_forecast",
            "converged": False,
            "next_bar_volatility": None,
        }
    persistence = float(fitted.params["alpha[1]"] + fitted.params["beta[1]"])
    return {
        **base,
        "status": "estimated",
        "converged": True,
        "next_bar_volatility": math.sqrt(variance) / 100,
        "persistence": persistence,
        "stationary_fit": persistence < 1,
        "warning_categories": sorted({type(w.message).__name__ for w in notices}),
    }
