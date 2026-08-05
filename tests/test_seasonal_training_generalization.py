from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from brain_refinery_v10_seasonal import simulate_seasonal  # noqa: E402


def test_seasonal_simulator_is_bounded_balanced_and_regime_varying() -> None:
    np.random.seed(42)
    prices = simulate_seasonal(5000)
    returns = np.diff(np.log(prices))

    assert prices.shape == (5000,)
    assert np.isfinite(prices).all()
    assert prices.min() > 0.0
    assert np.max(np.abs(returns)) <= 0.040001
    assert 0.35 <= float(np.mean(returns > 0.0)) <= 0.65

    block_volatility = [float(np.std(block)) for block in np.array_split(returns, 8)]
    assert max(block_volatility) > min(block_volatility) * 1.15


def test_seasonal_simulator_rejects_empty_lengths_cleanly() -> None:
    assert simulate_seasonal(0).shape == (0,)
