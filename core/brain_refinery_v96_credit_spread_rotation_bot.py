import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    direction_label_builder,
    feature_ema,
    feature_std,
    observation_feature,
    price_change,
    rolling_drawdown as runtime_rolling_drawdown,
    symbol_role_features,
)

_CREDIT_ROLE_MAP = {
    "hy_credit": ["HYG", "JNK", "USHY"],
    "ig_credit": ["LQD", "IGIB"],
    "treasury": ["TLT", "IEF", "SHY", "TIP"],
}


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def rolling_drawdown(close, window=220):
    out = np.zeros_like(close)
    for i in range(len(close)):
        start = max(0, i - window + 1)
        peak = np.max(close[start : i + 1])
        out[i] = (close[i] - peak) / (peak + 1e-8)
    return out


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]
    b = panel["bench_ret"]

    q_ret = hold_sample(r, 1170)
    q_bench = hold_sample(b, 1170)
    rel = q_ret - q_bench

    rel_fast = ema(rel, 6)
    rel_slow = ema(rel, 16)
    spread_mom = rel_fast - rel_slow

    dd = rolling_drawdown(c, window=260)
    dd_fast = ema(dd, 8)
    dd_slow = ema(dd, 21)

    vol = rolling_std(r, 120)
    tail_risk = np.maximum(-dd_fast, 0.0) + np.maximum(vol - ema(vol, 20), 0.0)

    # Credit sleeve proxy: prefer widening relative strength, avoid tail-risk spikes.
    credit_rotation = spread_mom - 0.45 * tail_risk

    return np.stack([r, rel, rel_fast, rel_slow, spread_mom, dd, dd_fast, dd_slow, vol, tail_risk, credit_rotation], axis=1)


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _CREDIT_ROLE_MAP)
    return np.asarray(
        [
            observation_feature(obs, "pct_from_close"),
            observation_feature(obs, "mom_5m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "ctx_UUP_pct_from_close"),
            observation_feature(obs, "bond_credit_risk_on_norm"),
            observation_feature(obs, "bond_credit_risk_off_norm"),
            observation_feature(obs, "bond_carry_roll_norm"),
            observation_feature(obs, "bond_credit_spread_level_norm"),
            observation_feature(obs, "bond_credit_spread_change_norm"),
            observation_feature(obs, "bond_hy_ig_flow_norm"),
            observation_feature(obs, "bond_nav_stress_norm"),
            observation_feature(obs, "bond_nav_discount_norm"),
            observation_feature(obs, "calendar_macro_surprise_norm"),
            observation_feature(obs, "options_negative_bias_norm"),
            observation_feature(obs, "options_roll_yield_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "options_put_call_oi_ratio_norm"),
            observation_feature(obs, "options_gamma_exposure_norm"),
            observation_feature(obs, "options_unusual_flow_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_ema(sequence, idx, "bond_credit_risk_on_norm", 5),
            feature_ema(sequence, idx, "bond_credit_risk_off_norm", 5),
            feature_std(sequence, idx, "pct_from_close", 8),
            runtime_rolling_drawdown(sequence, idx, 12),
            roles["role_hy_credit"],
            roles["role_ig_credit"],
            roles["role_treasury"],
        ],
        dtype=np.float32,
    )


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v96_credit_spread_rotation_bot",
        feature_names=[
            "ret",
            "rel",
            "rel_fast",
            "rel_slow",
            "spread_mom",
            "drawdown",
            "dd_fast",
            "dd_slow",
            "vol",
            "tail_risk",
            "credit_rotation",
        ],
        feature_builder=build_features,
        window=84,
        horizon=24,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v96_credit_spread_rotation_bot",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "ctx_UUP_pct_from_close",
            "bond_credit_risk_on_norm",
            "bond_credit_risk_off_norm",
            "bond_carry_roll_norm",
            "bond_credit_spread_level_norm",
            "bond_credit_spread_change_norm",
            "bond_hy_ig_flow_norm",
            "bond_nav_stress_norm",
            "bond_nav_discount_norm",
            "calendar_macro_surprise_norm",
            "options_negative_bias_norm",
            "options_roll_yield_norm",
            "options_vol_expectation_norm",
            "options_put_call_oi_ratio_norm",
            "options_gamma_exposure_norm",
            "options_unusual_flow_norm",
            "breadth_risk_off_norm",
            "ret_3",
            "ret_6",
            "bond_credit_risk_on_ema_5",
            "bond_credit_risk_off_ema_5",
            "pct_from_close_std_8",
            "drawdown_12",
            "role_hy_credit",
            "role_ig_credit",
            "role_treasury",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=direction_label_builder(min_return=0.0),
        lookback_days=30,
        mode_allowlist=["shadow_bond_equities"],
        window=28,
        horizon=18,
        min_samples=192,
        min_sequences=2,
        fallback_trainer=_train_synthetic,
    )
