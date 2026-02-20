import numpy as np

from indicator_bot_common import rolling_std, train_indicator_bot


def profile_proxy(close, volume, window=60, bins=24):
    poc = np.zeros_like(close)
    va_width = np.zeros_like(close)

    for i in range(len(close)):
        start = max(0, i - window + 1)
        p = close[start : i + 1]
        v = volume[start : i + 1]
        pmin, pmax = np.min(p), np.max(p)
        if pmax <= pmin:
            poc[i] = close[i]
            va_width[i] = 0.0
            continue
        edges = np.linspace(pmin, pmax, bins + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        vol_by_bin = np.zeros(bins)
        for j, b in enumerate(idx):
            vol_by_bin[b] += v[j]
        k = int(np.argmax(vol_by_bin))
        poc[i] = 0.5 * (edges[k] + edges[k + 1])
        va_width[i] = (edges[min(k + 1, bins)] - edges[max(k - 1, 0)]) / max(close[i], 1e-8)

    return poc, va_width


def build_features(panel):
    c = panel["close"]
    v = panel["volume"]
    r = panel["ret"]

    poc, va_width = profile_proxy(c, v, window=60, bins=24)
    dist_poc = (c - poc) / (poc + 1e-8)
    pull = np.abs(dist_poc) / (rolling_std(r, 20) + 1e-8)
    vol_z = (v - np.mean(v)) / (np.std(v) + 1e-8)

    return np.stack([r, dist_poc, va_width, pull, vol_z], axis=1)


if __name__ == "__main__":
    train_indicator_bot(
        run_tag="brain_refinery_v36_volume_profile_proxy",
        feature_names=["ret", "dist_poc", "value_area_width", "poc_pull", "volume_z"],
        feature_builder=build_features,
    )
