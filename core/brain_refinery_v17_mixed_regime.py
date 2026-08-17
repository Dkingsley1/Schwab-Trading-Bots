import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_price_indicator_bot


FEATURE_SOURCE = "prices"


def rsi(prices, period=14):
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period) + 1e-8
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rolling_min(x, window):
    out = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = np.min(x[start : i + 1])
    return out


def rolling_max(x, window):
    out = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = np.max(x[start : i + 1])
    return out


def simulate_mixed_regime(n=8000):
    prices = np.zeros(n, dtype=np.float64)
    prices[0] = 100.0
    fair_value = 100.0

    for i in range(1, n):
        cycle = i // 650
        block = cycle % 4
        if i % 650 == 0:
            fair_value = prices[i - 1] * (1.0 + 0.04 * np.random.randn())

        if block in (0, 2):
            trend_dir = 1.0 if block == 0 else -1.0
            drift = trend_dir * 0.00135
            ret = drift + 0.0039 * np.random.randn()
            prices[i] = max(0.1, prices[i - 1] * np.exp(ret))
        elif block == 1:
            reversion = 0.14 * np.log(max(fair_value, 0.1) / max(prices[i - 1], 0.1))
            ret = np.clip(reversion, -0.018, 0.018) + 0.0055 * np.random.randn()
            prices[i] = max(0.1, prices[i - 1] * np.exp(ret))
        else:
            shock = np.random.choice([-1.0, 1.0]) * np.random.uniform(0.0, 0.012)
            ret = shock + 0.0075 * np.random.randn()
            prices[i] = max(0.1, prices[i - 1] * np.exp(ret))

    return prices


def build_features(prices):
    prices = np.asarray(prices, dtype=np.float64)
    log_ret = np.diff(np.log(np.maximum(prices, 1e-8)), prepend=np.log(max(prices[0], 1e-8)))
    ret_3 = rolling_mean(log_ret, 3)
    ret_8 = rolling_mean(log_ret, 8)
    ret_21 = rolling_mean(log_ret, 21)
    ret_55 = rolling_mean(log_ret, 55)

    ema_fast = ema(prices, 12)
    ema_mid = ema(prices, 34)
    ema_slow = ema(prices, 89)
    trend_fast = (ema_fast - ema_mid) / (prices + 1e-8)
    trend_slow = (ema_mid - ema_slow) / (prices + 1e-8)
    price_gap = (prices - ema_mid) / (prices + 1e-8)

    vol_fast = rolling_std(log_ret, 12)
    vol_slow = rolling_std(log_ret, 55) + 1e-8
    vol_ratio = vol_fast / vol_slow
    trend_pressure = ret_21 / (vol_slow * np.sqrt(21.0) + 1e-8)
    reversion_pressure = -price_gap * vol_ratio

    lo = rolling_min(prices, 80)
    hi = rolling_max(prices, 80)
    range_position = ((prices - lo) / (hi - lo + 1e-8)) - 0.5
    range_escape = np.sign(range_position) * np.maximum(np.abs(range_position) - 0.35, 0.0)
    rsi14 = (rsi(prices, 14) - 50.0) / 50.0

    return np.stack(
        [
            log_ret,
            ret_3,
            ret_8,
            ret_21,
            ret_55,
            trend_fast,
            trend_slow,
            price_gap,
            vol_fast,
            vol_slow,
            vol_ratio,
            trend_pressure,
            reversion_pressure,
            range_position,
            range_escape,
            rsi14,
        ],
        axis=1,
    )


def train_brain():
    return train_price_indicator_bot(
        run_tag="brain_refinery_v17_mixed_regime",
        feature_names=[
            "log_ret",
            "ret_3",
            "ret_8",
            "ret_21",
            "ret_55",
            "trend_fast",
            "trend_slow",
            "price_gap",
            "vol_fast",
            "vol_slow",
            "vol_ratio",
            "trend_pressure",
            "reversion_pressure",
            "range_position",
            "range_escape",
            "rsi14",
        ],
        feature_builder=build_features,
        price_simulator=simulate_mixed_regime,
        num_points=8000,
        window=64,
        horizon=12,
        learning_rate=0.0007,
        epochs=240,
        batch_size=128,
        patience=24,
    )


if __name__ == "__main__":
    train_brain()
