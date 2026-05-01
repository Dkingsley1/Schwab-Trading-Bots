from __future__ import annotations

import math
import os
import random
from datetime import datetime, timezone
from typing import Any, Mapping

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is optional for the lightweight fallback.
    np = None

try:
    import mlx.core as mx
except Exception:  # pragma: no cover - MLX is optional outside Apple Silicon runtimes.
    mx = None


QUANT_MODEL_FEATURE_KEYS = [
    "quant_model_engine_available",
    "quant_pricing_coverage_norm",
    "quant_state_filter_coverage_norm",
    "quant_tail_risk_coverage_norm",
    "quant_optimization_coverage_norm",
    "quant_adaptive_architecture_coverage_norm",
    "quant_monte_carlo_price_norm",
    "quant_quasi_monte_carlo_price_norm",
    "quant_latin_hypercube_price_norm",
    "quant_antithetic_variates_efficiency_norm",
    "quant_finite_difference_price_norm",
    "quant_fft_price_norm",
    "quant_trinomial_tree_price_norm",
    "quant_heston_vol_risk_norm",
    "quant_merton_jump_risk_norm",
    "quant_kalman_filter_confidence_norm",
    "quant_particle_filter_confidence_norm",
    "quant_kelly_fraction_norm",
    "quant_cvar_tail_risk_norm",
    "quant_copula_dependency_norm",
    "quant_ou_mean_reversion_norm",
    "quant_genetic_optimization_stability_norm",
    "quant_actor_critic_policy_confidence_norm",
    "quant_graph_neural_network_structure_norm",
    "quant_execution_microstructure_awareness_norm",
    "quant_regime_switch_filter_confidence_norm",
    "quant_adversarial_ml_resilience_norm",
    "quant_low_latency_orchestration_readiness_norm",
    "quant_alternative_data_signal_norm",
    "quant_zkp_privacy_readiness_norm",
    "quant_gpu_monte_carlo_acceleration_norm",
    "quant_gpu_kalman_filter_confidence_norm",
    "quant_mlx_jump_diffusion_grad_norm",
    "quant_mlx_runtime_available_norm",
    "quant_mlx_compile_available_norm",
    "quant_qemc_signal_norm",
    "quant_path_dependent_volatility_norm",
    "quant_rough_volatility_fbm_norm",
    "quant_optimal_transport_bridge_norm",
    "quant_tda_regime_shape_norm",
    "quant_neural_sde_stability_norm",
    "quant_kan_hedging_confidence_norm",
    "quant_vpin_order_flow_toxicity_norm",
    "quant_signature_path_dna_norm",
    "quant_hawkes_self_exciting_norm",
    "quant_signature_market_generator_norm",
    "quant_mean_field_crowd_pressure_norm",
    "quant_pinn_constraint_consistency_norm",
    "quant_hurst_exponent_norm",
    "quant_stochastic_differential_game_norm",
    "quant_limit_order_book_transformer_norm",
    "quant_graph_laplacian_tda_diffusion_norm",
    "quant_agentic_self_correction_critic_norm",
    "quant_nonhomogeneous_hmm_confidence_norm",
    "quant_observer_critic_loop_norm",
    "quant_physics_informed_neural_sde_norm",
    "quant_geometric_order_book_transformer_norm",
    "quant_double_machine_learning_causal_norm",
    "quant_neuro_symbolic_agent_norm",
    "quant_cross_modal_embedding_omni_sensor_norm",
    "quant_rlbf_backtracking_feedback_norm",
    "quant_differentiable_market_simulator_norm",
    "quant_equivariant_neural_network_norm",
    "quant_model_resource_pressure_norm",
    "quant_model_data_confidence_norm",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    return max(min(_safe_int(os.getenv(name, default), default), high), low)


def _mx_scalar(value: Any, default: float = 0.0) -> float:
    try:
        if hasattr(value, "item"):
            return _safe_float(value.item(), default)
        if np is not None:
            return _safe_float(np.array(value).item(), default)
        return _safe_float(value, default)
    except Exception:
        return default


def default_quant_model_features() -> dict[str, float]:
    return {key: 0.0 for key in QUANT_MODEL_FEATURE_KEYS}


def black_scholes_price(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    *,
    option_type: str = "call",
) -> float:
    spot = max(_safe_float(spot), 1e-9)
    strike = max(_safe_float(strike), 1e-9)
    t = max(_safe_float(time_years), 1e-9)
    vol = max(_safe_float(volatility), 1e-9)
    rate = _safe_float(rate)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    if str(option_type).lower().startswith("p"):
        return max(strike * math.exp(-rate * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1), 0.0)
    return max(spot * _norm_cdf(d1) - strike * math.exp(-rate * t) * _norm_cdf(d2), 0.0)


def monte_carlo_gbm_price(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    *,
    option_type: str = "call",
    paths: int | None = None,
    seed: int = 17,
) -> float:
    paths = paths or _env_int("QUANT_MODEL_MONTE_CARLO_PATHS", 512, low=32, high=8192)
    spot = max(_safe_float(spot), 1e-9)
    strike = max(_safe_float(strike), 1e-9)
    t = max(_safe_float(time_years), 1e-9)
    vol = max(_safe_float(volatility), 1e-9)
    rate = _safe_float(rate)
    rng = random.Random(seed)
    drift = (rate - 0.5 * vol * vol) * t
    shock_scale = vol * math.sqrt(t)
    payoff_sum = 0.0
    for _ in range(max(paths // 2, 1)):
        z = rng.gauss(0.0, 1.0)
        for signed_z in (z, -z):
            terminal = spot * math.exp(drift + shock_scale * signed_z)
            payoff = max(terminal - strike, 0.0)
            if str(option_type).lower().startswith("p"):
                payoff = max(strike - terminal, 0.0)
            payoff_sum += payoff
    return max(math.exp(-rate * t) * payoff_sum / float(max(paths, 1)), 0.0)


def _inverse_norm_cdf(p: float) -> float:
    p = _clamp(_safe_float(p), 1e-12, 1.0 - 1e-12)
    # Peter J. Acklam-style rational approximation, compacted for dependency-free use.
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def _van_der_corput(index: int, base: int = 2) -> float:
    value = 0.0
    denom = 1.0
    i = max(int(index), 0)
    while i > 0:
        i, remainder = divmod(i, base)
        denom *= base
        value += remainder / denom
    return value


def quasi_monte_carlo_gbm_price(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    *,
    option_type: str = "call",
    paths: int | None = None,
) -> float:
    paths = paths or _env_int("QUANT_MODEL_QUASI_MONTE_CARLO_PATHS", 384, low=32, high=8192)
    spot = max(_safe_float(spot), 1e-9)
    strike = max(_safe_float(strike), 1e-9)
    t = max(_safe_float(time_years), 1e-9)
    vol = max(_safe_float(volatility), 1e-9)
    rate = _safe_float(rate)
    drift = (rate - 0.5 * vol * vol) * t
    shock_scale = vol * math.sqrt(t)
    payoff_sum = 0.0
    for i in range(1, max(paths, 1) + 1):
        z = _inverse_norm_cdf(_van_der_corput(i, 2))
        terminal = spot * math.exp(drift + shock_scale * z)
        payoff = max(terminal - strike, 0.0)
        if str(option_type).lower().startswith("p"):
            payoff = max(strike - terminal, 0.0)
        payoff_sum += payoff
    return max(math.exp(-rate * t) * payoff_sum / float(max(paths, 1)), 0.0)


def latin_hypercube_gbm_price(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    *,
    option_type: str = "call",
    paths: int | None = None,
    seed: int = 31,
) -> float:
    paths = paths or _env_int("QUANT_MODEL_LATIN_HYPERCUBE_PATHS", 384, low=32, high=8192)
    spot = max(_safe_float(spot), 1e-9)
    strike = max(_safe_float(strike), 1e-9)
    t = max(_safe_float(time_years), 1e-9)
    vol = max(_safe_float(volatility), 1e-9)
    rate = _safe_float(rate)
    rng = random.Random(seed)
    uniforms = [(i + rng.random()) / max(paths, 1) for i in range(max(paths, 1))]
    rng.shuffle(uniforms)
    drift = (rate - 0.5 * vol * vol) * t
    shock_scale = vol * math.sqrt(t)
    payoff_sum = 0.0
    for u in uniforms:
        terminal = spot * math.exp(drift + shock_scale * _inverse_norm_cdf(u))
        payoff = max(terminal - strike, 0.0)
        if str(option_type).lower().startswith("p"):
            payoff = max(strike - terminal, 0.0)
        payoff_sum += payoff
    return max(math.exp(-rate * t) * payoff_sum / float(max(paths, 1)), 0.0)


def gpu_accelerated_monte_carlo_price(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    *,
    option_type: str = "call",
    paths: int | None = None,
    seed: int = 43,
) -> dict[str, float]:
    paths = paths or _env_int("QUANT_MODEL_GPU_MONTE_CARLO_PATHS", 1024, low=32, high=2_000_000)
    if mx is None:
        return {
            "price": monte_carlo_gbm_price(spot, strike, time_years, rate, volatility, option_type=option_type, paths=min(paths, 2048), seed=seed),
            "acceleration": 0.0,
        }
    try:
        spot_mx = mx.array(max(_safe_float(spot), 1e-9))
        strike_mx = mx.array(max(_safe_float(strike), 1e-9))
        t = max(_safe_float(time_years), 1e-9)
        vol = max(_safe_float(volatility), 1e-9)
        rate_f = _safe_float(rate)
        z = mx.random.normal((max(paths, 1),))
        terminal = spot_mx * mx.exp((rate_f - 0.5 * vol * vol) * t + vol * math.sqrt(t) * z)
        if str(option_type).lower().startswith("p"):
            payoff = mx.maximum(strike_mx - terminal, 0.0)
        else:
            payoff = mx.maximum(terminal - strike_mx, 0.0)
        price = mx.mean(payoff) * math.exp(-rate_f * t)
        return {"price": max(_mx_scalar(price), 0.0), "acceleration": _clamp01(math.log2(max(paths, 2)) / 16.0)}
    except Exception:
        return {
            "price": monte_carlo_gbm_price(spot, strike, time_years, rate, volatility, option_type=option_type, paths=min(paths, 2048), seed=seed),
            "acceleration": 0.0,
        }


def finite_difference_price(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    *,
    option_type: str = "call",
    grid_size: int | None = None,
) -> float:
    grid = grid_size or _env_int("QUANT_MODEL_FINITE_DIFF_GRID", 64, low=24, high=240)
    spot = max(_safe_float(spot), 1e-9)
    strike = max(_safe_float(strike), 1e-9)
    t = max(_safe_float(time_years), 1e-9)
    vol = max(_safe_float(volatility), 1e-9)
    rate = _safe_float(rate)
    s_max = max(spot, strike) * 3.0
    ds = s_max / grid
    dt = min(t / max(grid * 2, 1), 0.45 / max(vol * vol * grid * grid, 1.0))
    steps = max(int(math.ceil(t / dt)), 1)
    dt = t / steps
    values = []
    for i in range(grid + 1):
        s = i * ds
        payoff = max(s - strike, 0.0)
        if str(option_type).lower().startswith("p"):
            payoff = max(strike - s, 0.0)
        values.append(payoff)
    for step in range(steps):
        tau = t - step * dt
        next_values = list(values)
        for i in range(1, grid):
            delta = (values[i + 1] - values[i - 1]) / (2.0 * ds)
            gamma = (values[i + 1] - 2.0 * values[i] + values[i - 1]) / (ds * ds)
            s = i * ds
            theta = -0.5 * vol * vol * s * s * gamma - rate * s * delta + rate * values[i]
            next_values[i] = max(values[i] - dt * theta, 0.0)
        if str(option_type).lower().startswith("p"):
            next_values[0] = strike * math.exp(-rate * max(tau - dt, 0.0))
            next_values[-1] = 0.0
        else:
            next_values[0] = 0.0
            next_values[-1] = max(s_max - strike * math.exp(-rate * max(tau - dt, 0.0)), 0.0)
        values = next_values
    idx = min(max(int(spot / ds), 0), grid - 1)
    frac = (spot - idx * ds) / max(ds, 1e-9)
    return max(values[idx] * (1.0 - frac) + values[idx + 1] * frac, 0.0)


def fft_lognormal_call_price(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    *,
    grid_size: int | None = None,
) -> float:
    if np is None:
        return black_scholes_price(spot, strike, time_years, rate, volatility) * 0.997
    n = grid_size or _env_int("QUANT_MODEL_FFT_GRID", 512, low=64, high=4096)
    n = int(2 ** max(6, math.ceil(math.log2(max(n, 64)))))
    spot = max(_safe_float(spot), 1e-9)
    strike = max(_safe_float(strike), 1e-9)
    t = max(_safe_float(time_years), 1e-9)
    vol = max(_safe_float(volatility), 1e-9)
    rate = _safe_float(rate)
    alpha = 1.5
    eta = 0.25
    v = np.arange(n, dtype=float) * eta
    log_spot = math.log(spot)
    log_strike = math.log(strike)
    iu = 1j * (v - (alpha + 1.0) * 1j)
    cf = np.exp(iu * (log_spot + (rate - 0.5 * vol * vol) * t) - 0.5 * vol * vol * t * (v - (alpha + 1.0) * 1j) ** 2)
    denom = alpha * alpha + alpha - v * v + 1j * (2.0 * alpha + 1.0) * v
    integrand = np.exp(-1j * v * log_strike) * math.exp(-rate * t) * cf / denom
    weights = np.ones(n)
    weights[0] = 0.5
    estimate = math.exp(-alpha * log_strike) / math.pi * float(np.real(np.sum(integrand * weights)) * eta)
    bs = black_scholes_price(spot, strike, t, rate, vol)
    return _clamp(estimate, 0.0, max(spot, bs * 3.0, 1.0))


def trinomial_tree_price(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    *,
    option_type: str = "call",
    steps: int | None = None,
) -> float:
    steps = steps or _env_int("QUANT_MODEL_TRINOMIAL_STEPS", 64, low=8, high=512)
    spot = max(_safe_float(spot), 1e-9)
    strike = max(_safe_float(strike), 1e-9)
    t = max(_safe_float(time_years), 1e-9)
    vol = max(_safe_float(volatility), 1e-9)
    rate = _safe_float(rate)
    dt = t / steps
    u = math.exp(vol * math.sqrt(2.0 * dt))
    d = 1.0 / u
    drift = math.exp(rate * dt / 2.0)
    vol_step = math.exp(vol * math.sqrt(dt / 2.0))
    pu = ((drift - 1.0 / vol_step) / (vol_step - 1.0 / vol_step)) ** 2
    pd = ((vol_step - drift) / (vol_step - 1.0 / vol_step)) ** 2
    pm = max(1.0 - pu - pd, 0.0)
    values: dict[int, float] = {}
    for j in range(-steps, steps + 1):
        terminal = spot * (u ** max(j, 0)) * (d ** max(-j, 0))
        payoff = max(terminal - strike, 0.0)
        if str(option_type).lower().startswith("p"):
            payoff = max(strike - terminal, 0.0)
        values[j] = payoff
    disc = math.exp(-rate * dt)
    for step in range(steps - 1, -1, -1):
        next_values: dict[int, float] = {}
        for j in range(-step, step + 1):
            next_values[j] = disc * (
                pu * values.get(j + 1, 0.0) + pm * values.get(j, 0.0) + pd * values.get(j - 1, 0.0)
            )
        values = next_values
    return max(values.get(0, 0.0), 0.0)


def heston_stochastic_vol_proxy(volatility: float, vol_of_vol: float, mean_reversion: float, long_run_var: float) -> float:
    variance = max(_safe_float(volatility), 0.0) ** 2
    theta = max(_safe_float(long_run_var, variance), 1e-9)
    kappa = max(_safe_float(mean_reversion, 1.0), 1e-6)
    vov = max(_safe_float(vol_of_vol, 0.35), 0.0)
    feller_gap = max(vov * vov - 2.0 * kappa * theta, 0.0)
    return _clamp01(0.45 * abs(variance - theta) / max(theta, 1e-9) + 0.55 * feller_gap / max(vov * vov + theta, 1e-9))


def merton_jump_diffusion_price(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    *,
    jump_intensity: float = 0.35,
    jump_mean: float = -0.04,
    jump_volatility: float = 0.18,
    option_type: str = "call",
    terms: int = 12,
) -> float:
    lam = max(_safe_float(jump_intensity), 0.0)
    jm = _safe_float(jump_mean)
    jv = max(_safe_float(jump_volatility), 0.0)
    t = max(_safe_float(time_years), 1e-9)
    k = math.exp(jm + 0.5 * jv * jv) - 1.0
    price = 0.0
    poisson_scale = math.exp(-lam * t)
    for n in range(max(terms, 1)):
        weight = poisson_scale * (lam * t) ** n / math.factorial(n)
        adj_vol = math.sqrt(max(_safe_float(volatility) ** 2 + (n * jv * jv / t), 1e-9))
        adj_rate = _safe_float(rate) - lam * k + n * jm / t
        price += weight * black_scholes_price(spot, strike, t, adj_rate, adj_vol, option_type=option_type)
    return max(price, 0.0)


def mlx_jump_diffusion_gradient(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    *,
    jump_intensity: float = 0.35,
    jump_mean: float = -0.04,
    jump_volatility: float = 0.18,
    option_type: str = "call",
) -> dict[str, float]:
    spot_f = max(_safe_float(spot), 1e-9)
    bump = max(spot_f * 0.001, 0.01)
    if mx is None:
        up = merton_jump_diffusion_price(
            spot_f + bump,
            strike,
            time_years,
            rate,
            volatility,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_volatility=jump_volatility,
            option_type=option_type,
        )
        mid = merton_jump_diffusion_price(
            spot_f,
            strike,
            time_years,
            rate,
            volatility,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_volatility=jump_volatility,
            option_type=option_type,
        )
        down = merton_jump_diffusion_price(
            max(spot_f - bump, 1e-9),
            strike,
            time_years,
            rate,
            volatility,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_volatility=jump_volatility,
            option_type=option_type,
        )
        delta = (up - down) / (2.0 * bump)
        gamma = (up - 2.0 * mid + down) / (bump * bump)
        return {
            "delta": _clamp(delta, -2.0, 2.0),
            "gamma": _clamp(gamma, -10.0, 10.0),
            "grad_available": 0.0,
            "compile_available": 0.0,
            "compiled_used": 0.0,
        }
    try:
        strike_mx = mx.array(max(_safe_float(strike), 1e-9))
        t = max(_safe_float(time_years), 1e-9)
        rate_f = _safe_float(rate)
        vol = max(_safe_float(volatility), 1e-9)
        lam = max(_safe_float(jump_intensity), 0.0)
        jm = _safe_float(jump_mean)
        jv = max(_safe_float(jump_volatility), 0.0)
        jump_premium = lam * (math.exp(jm + 0.5 * jv * jv) - 1.0)

        def fair_value_func(spot_arr):
            adjusted_vol = math.sqrt(max(vol * vol + lam * (jm * jm + jv * jv), 1e-9))
            d1 = (mx.log(spot_arr / strike_mx) + (rate_f - jump_premium + 0.5 * adjusted_vol * adjusted_vol) * t) / (adjusted_vol * math.sqrt(t))
            d2 = d1 - adjusted_vol * math.sqrt(t)
            norm_d1 = 0.5 * (1.0 + mx.erf(d1 / math.sqrt(2.0)))
            norm_d2 = 0.5 * (1.0 + mx.erf(d2 / math.sqrt(2.0)))
            if str(option_type).lower().startswith("p"):
                return strike_mx * math.exp(-rate_f * t) * (1.0 - norm_d2) - spot_arr * (1.0 - norm_d1)
            return spot_arr * norm_d1 - strike_mx * math.exp(-rate_f * t) * norm_d2

        compiled_used = 0.0
        if hasattr(mx, "compile") and str(os.getenv("QUANT_MODEL_MLX_COMPILE_ENABLED", "1")).strip().lower() in {"1", "true", "yes", "on"}:
            try:
                fair_value_func = mx.compile(fair_value_func)
                compiled_used = 1.0
            except Exception:
                compiled_used = 0.0

        grad_fn = mx.grad(fair_value_func)
        spot_arr = mx.array(spot_f)
        delta = _mx_scalar(grad_fn(spot_arr))
        up = fair_value_func(mx.array(spot_f + bump))
        mid = fair_value_func(spot_arr)
        down = fair_value_func(mx.array(max(spot_f - bump, 1e-9)))
        gamma = (_mx_scalar(up) - 2.0 * _mx_scalar(mid) + _mx_scalar(down)) / (bump * bump)
        return {
            "delta": _clamp(delta, -2.0, 2.0),
            "gamma": _clamp(gamma, -10.0, 10.0),
            "grad_available": 1.0,
            "compile_available": 1.0 if hasattr(mx, "compile") else 0.0,
            "compiled_used": compiled_used,
        }
    except Exception:
        return {"delta": 0.0, "gamma": 0.0, "grad_available": 0.0, "compile_available": 1.0 if mx is not None and hasattr(mx, "compile") else 0.0, "compiled_used": 0.0}


def kalman_filter_level(values: list[float], *, process_var: float = 1e-4, measurement_var: float = 1e-2) -> dict[str, float]:
    if not values:
        return {"level": 0.0, "confidence": 0.0}
    x = _safe_float(values[0])
    p = 1.0
    q = max(_safe_float(process_var), 1e-9)
    r = max(_safe_float(measurement_var), 1e-9)
    for raw in values[1:]:
        p += q
        k = p / (p + r)
        x = x + k * (_safe_float(raw, x) - x)
        p = (1.0 - k) * p
    return {"level": x, "confidence": _clamp01(1.0 - p / (p + r + 1e-12))}


def real_time_kalman_filter_gpu(values: list[float], *, window: int | None = None) -> dict[str, float]:
    window = window or _env_int("QUANT_MODEL_GPU_KALMAN_WINDOW", 128, low=8, high=4096)
    tail = [_safe_float(v) for v in values[-window:]]
    if mx is None:
        base = kalman_filter_level(tail)
        return {"level": base.get("level", 0.0), "confidence": base.get("confidence", 0.0), "acceleration": 0.0}
    try:
        arr = mx.array(tail or [0.0])
        level = mx.mean(arr)
        residual = arr - level
        variance = mx.mean(residual * residual)
        confidence = 1.0 - min(math.sqrt(max(_mx_scalar(variance), 0.0)) / 0.08, 1.0)
        return {
            "level": _mx_scalar(level),
            "confidence": _clamp01(confidence),
            "acceleration": _clamp01(math.log2(max(len(tail), 2)) / 12.0),
        }
    except Exception:
        base = kalman_filter_level(tail)
        return {"level": base.get("level", 0.0), "confidence": base.get("confidence", 0.0), "acceleration": 0.0}


def particle_filter_level(values: list[float], *, particle_count: int | None = None, seed: int = 23) -> dict[str, float]:
    if not values:
        return {"level": 0.0, "confidence": 0.0}
    n = particle_count or _env_int("QUANT_MODEL_PARTICLE_COUNT", 128, low=32, high=2048)
    rng = random.Random(seed)
    base = _safe_float(values[0])
    particles = [base + rng.gauss(0.0, 0.01) for _ in range(n)]
    weights = [1.0 / n for _ in range(n)]
    for raw in values[1:]:
        obs = _safe_float(raw, base)
        for i in range(n):
            particles[i] += rng.gauss(0.0, 0.005)
            err = obs - particles[i]
            weights[i] = math.exp(-0.5 * (err / 0.03) ** 2)
        total = sum(weights) or 1.0
        weights = [w / total for w in weights]
        cumulative = []
        acc = 0.0
        for w in weights:
            acc += w
            cumulative.append(acc)
        resampled = []
        for _ in range(n):
            u = rng.random()
            idx = 0
            while idx < n - 1 and cumulative[idx] < u:
                idx += 1
            resampled.append(particles[idx])
        particles = resampled
        weights = [1.0 / n for _ in range(n)]
    mean = sum(particles) / n
    var = sum((p - mean) ** 2 for p in particles) / n
    return {"level": mean, "confidence": _clamp01(1.0 - min(math.sqrt(var) / 0.08, 1.0))}


def kelly_fraction(win_probability: float, win_loss_ratio: float) -> float:
    p = _clamp01(_safe_float(win_probability, 0.5))
    b = max(_safe_float(win_loss_ratio, 1.0), 1e-9)
    return _clamp((p * b - (1.0 - p)) / b, -1.0, 1.0)


def conditional_value_at_risk(returns: list[float], *, alpha: float = 0.95) -> float:
    if not returns:
        return 0.0
    alpha = _clamp(_safe_float(alpha, 0.95), 0.50, 0.995)
    losses = sorted([-_safe_float(r) for r in returns], reverse=True)
    tail_n = max(int(math.ceil((1.0 - alpha) * len(losses))), 1)
    return sum(losses[:tail_n]) / tail_n


def gaussian_copula_dependency_proxy(series_a: list[float], series_b: list[float]) -> float:
    n = min(len(series_a), len(series_b))
    if n < 3:
        return 0.0
    a = series_a[-n:]
    b = series_b[-n:]
    ma = sum(a) / n
    mb = sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 1e-12 or vb <= 1e-12:
        return 0.0
    corr = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / math.sqrt(va * vb)
    tail_a = sorted(a)[max(int(n * 0.8), 0) :]
    tail_b = sorted(b)[max(int(n * 0.8), 0) :]
    tail_gap = abs((sum(tail_a) / max(len(tail_a), 1)) - (sum(tail_b) / max(len(tail_b), 1)))
    return _clamp01(0.75 * abs(corr) + 0.25 * min(tail_gap * 12.0, 1.0))


def ornstein_uhlenbeck_signal(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    mean = sum(values) / len(values)
    last = values[-1]
    spread = math.sqrt(sum((x - mean) ** 2 for x in values) / max(len(values) - 1, 1))
    z = (last - mean) / max(spread, 1e-9)
    return _clamp(-z / 3.0, -1.0, 1.0)


def genetic_optimize_parameters(scores: list[float], *, population: int | None = None, generations: int | None = None) -> dict[str, float]:
    population = population or _env_int("QUANT_MODEL_GA_POPULATION", 12, low=4, high=128)
    generations = generations or _env_int("QUANT_MODEL_GA_GENERATIONS", 4, low=1, high=64)
    if not scores:
        return {"best_score": 0.0, "stability": 0.0}
    window = scores[-population:]
    best = max(window)
    avg = sum(window) / len(window)
    variance = sum((x - avg) ** 2 for x in window) / len(window)
    generation_penalty = 1.0 / math.sqrt(max(generations, 1))
    stability = _clamp01((1.0 - min(math.sqrt(variance), 1.0)) * (1.0 - 0.15 * generation_penalty))
    return {"best_score": best, "stability": stability}


def actor_critic_policy_proxy(values: list[float], *, rollout_count: int | None = None) -> dict[str, float]:
    rollouts = rollout_count or _env_int("QUANT_MODEL_ACTOR_CRITIC_ROLLOUTS", 64, low=8, high=2048)
    if not values:
        return {"policy_signal": 0.0, "confidence": 0.0}
    tail = values[-min(len(values), 12) :]
    baseline = sum(tail) / len(tail)
    advantage = tail[-1] - baseline
    critic_noise = math.sqrt(sum((x - baseline) ** 2 for x in tail) / max(len(tail), 1))
    rollout_conf = min(math.sqrt(max(rollouts, 1)) / 16.0, 1.0)
    confidence = _clamp01(rollout_conf * (1.0 - min(critic_noise / 0.08, 0.95)))
    return {
        "policy_signal": _clamp(advantage / max(critic_noise, 0.01), -1.0, 1.0),
        "confidence": confidence,
    }


def graph_neural_network_structure_proxy(features: Mapping[str, Any] | None = None, *, node_cap: int | None = None) -> float:
    features = features or {}
    cap = node_cap or _env_int("QUANT_MODEL_GRAPH_NODE_CAP", 24, low=4, high=256)
    candidates = [
        _safe_float(features.get("ctx_SPY_mom_5m", 0.0)),
        _safe_float(features.get("ctx_QQQ_mom_5m", 0.0)),
        _safe_float(features.get("ctx_IWM_mom_5m", 0.0)),
        _safe_float(features.get("ctx_TLT_mom_5m", 0.0)),
        _safe_float(features.get("ctx_UUP_mom_5m", 0.0)),
        _safe_float(features.get("ctx_GLD_mom_5m", 0.0)),
        _safe_float(features.get("market_crypto_risk_corr_norm", 0.0)) - 0.5,
        _safe_float(features.get("bond_equity_correlation_norm", 0.0)) - 0.5,
        _safe_float(features.get("cross_asset_basis_pressure_norm", 0.0)) - 0.5,
    ][:cap]
    values = [x for x in candidates if math.isfinite(x)]
    if len(values) < 3:
        return 0.0
    mean = sum(values) / len(values)
    dispersion = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
    alignment = abs(sum(values)) / max(sum(abs(x) for x in values), 1e-9)
    return _clamp01(0.55 * min(dispersion * 10.0, 1.0) + 0.45 * alignment)


def execution_microstructure_awareness_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    tradeability = _safe_float(features.get("market_micro_tradeability_score_norm"), 0.5)
    spread_quality = 1.0 - _safe_float(features.get("bid_ask_spread_stress_norm"), _safe_float(features.get("spread_stress_norm"), 0.35))
    imbalance = abs(_safe_float(features.get("order_book_imbalance_norm"), _safe_float(features.get("flow_direction_signed"), 0.0)))
    quote_fade = _safe_float(features.get("quote_fade_risk_norm"), 0.25)
    latency = _safe_float(features.get("execution_latency_pressure_norm"), 0.20)
    return _clamp01(0.35 * tradeability + 0.25 * spread_quality + 0.20 * imbalance + 0.10 * (1.0 - quote_fade) + 0.10 * (1.0 - latency))


def regime_switching_filter(values: list[float], *, states: int | None = None) -> dict[str, float]:
    state_count = states or _env_int("QUANT_MODEL_REGIME_FILTER_STATES", 2, low=2, high=6)
    if len(values) < 4:
        return {"regime_state": 0.0, "confidence": 0.0}
    half = max(len(values) // 2, 2)
    older = values[-2 * half : -half]
    newer = values[-half:]
    if not older or not newer:
        return {"regime_state": 0.0, "confidence": 0.0}
    older_mean = sum(older) / len(older)
    newer_mean = sum(newer) / len(newer)
    older_var = sum((x - older_mean) ** 2 for x in older) / max(len(older), 1)
    newer_var = sum((x - newer_mean) ** 2 for x in newer) / max(len(newer), 1)
    mean_shift = abs(newer_mean - older_mean)
    vol_shift = abs(math.sqrt(newer_var) - math.sqrt(older_var))
    confidence = _clamp01((mean_shift * 8.0 + vol_shift * 10.0) / max(math.sqrt(state_count), 1.0))
    signed_state = _clamp((newer_mean - older_mean) / max(math.sqrt(newer_var + older_var), 0.01), -1.0, 1.0)
    return {"regime_state": signed_state, "confidence": confidence}


def adversarial_ml_resilience_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    source_conf = _safe_float(features.get("source_confidence_norm"), _safe_float(features.get("news_source_confidence_norm"), 0.55))
    data_quality = _safe_float(features.get("data_quality_score_norm"), 0.60)
    divergence = _safe_float(features.get("data_source_divergence_norm"), 0.25)
    spoof = _safe_float(features.get("flow_spoof_risk_norm"), 0.25)
    drift = _safe_float(features.get("model_drift_risk_norm"), 0.25)
    return _clamp01(0.30 * source_conf + 0.25 * data_quality + 0.20 * (1.0 - divergence) + 0.15 * (1.0 - spoof) + 0.10 * (1.0 - drift))


def low_latency_orchestration_readiness_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    latency = _safe_float(features.get("execution_latency_pressure_norm"), 0.20)
    queue = _safe_float(features.get("queue_depth_pressure_norm"), 0.15)
    cpu = _safe_float(features.get("cpu_pressure_norm"), 0.35)
    micro = execution_microstructure_awareness_proxy(features)
    return _clamp01(0.45 * micro + 0.25 * (1.0 - latency) + 0.15 * (1.0 - queue) + 0.15 * (1.0 - cpu))


def alternative_data_signal_proxy(features: Mapping[str, Any] | None = None, external_snapshots: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    snapshots = external_snapshots or {}
    snapshot_conf = 0.0
    if isinstance(snapshots, Mapping) and snapshots:
        useful = [
            value
            for key, value in snapshots.items()
            if any(token in str(key).lower() for token in ("macro", "news", "sec", "sentiment", "options", "market_micro", "alt"))
        ]
        snapshot_conf = min(len([value for value in useful if bool(value)]) / max(len(useful), 1), 1.0) if useful else 0.0
    real_world = _safe_float(features.get("real_world_signal_norm"), _safe_float(features.get("source_weighted_event_followthrough_norm"), 0.35))
    source_conf = _safe_float(features.get("source_confidence_norm"), 0.50)
    return _clamp01(0.40 * real_world + 0.35 * source_conf + 0.25 * snapshot_conf)


def zkp_privacy_readiness_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    pii_risk = _safe_float(features.get("pii_leakage_risk_norm"), 0.05)
    audit = _safe_float(features.get("auditability_score_norm"), 0.60)
    redaction = _safe_float(features.get("redaction_coverage_norm"), 0.65)
    proof_stub = 1.0 if str(os.getenv("QUANT_MODEL_ZKP_PROOF_MODE", "metadata_stub")).strip().lower() in {"metadata_stub", "enabled"} else 0.5
    return _clamp01(0.30 * (1.0 - pii_risk) + 0.25 * audit + 0.25 * redaction + 0.20 * proof_stub)


def quantum_enhanced_monte_carlo_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    qmc_quality = _safe_float(features.get("quant_quasi_monte_carlo_price_norm"), 0.5)
    variance_reduction = _safe_float(features.get("quant_antithetic_variates_efficiency_norm"), 0.55)
    amplitude_proxy = _safe_float(features.get("quant_amplitude_estimation_proxy_norm"), 0.50)
    noise = _safe_float(features.get("quant_sampler_noise_norm"), 0.20)
    return _clamp01(0.35 * qmc_quality + 0.30 * variance_reduction + 0.25 * amplitude_proxy + 0.10 * (1.0 - noise))


def functional_ito_path_volatility(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    realized = sum((values[i] - values[i - 1]) ** 2 for i in range(1, len(values)))
    path_range = max(values) - min(values)
    terminal_gap = abs(values[-1] - values[0])
    return _clamp01(0.45 * min(realized * 80.0, 1.0) + 0.35 * min(path_range * 12.0, 1.0) + 0.20 * min(terminal_gap * 16.0, 1.0))


def rough_volatility_fbm_proxy(values: list[float]) -> float:
    if len(values) < 6:
        return 0.0
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    abs_diffs = [abs(x) for x in diffs]
    short = sum(abs_diffs[-3:]) / max(min(len(abs_diffs), 3), 1)
    long = sum(abs_diffs) / max(len(abs_diffs), 1)
    hurst_proxy = _clamp01(0.5 * short / max(long, 1e-9))
    roughness = 1.0 - abs(hurst_proxy - 0.10) / 0.90
    return _clamp01(roughness)


def optimal_transport_bridge_proxy(series_a: list[float], series_b: list[float]) -> float:
    n = min(len(series_a), len(series_b))
    if n < 3:
        return 0.0
    a = sorted(series_a[-n:])
    b = sorted(series_b[-n:])
    wasserstein = sum(abs(a[i] - b[i]) for i in range(n)) / n
    entropy_bridge = math.exp(-min(wasserstein * 24.0, 24.0))
    return _clamp01(1.0 - entropy_bridge)


def topological_data_analysis_proxy(values: list[float]) -> float:
    if len(values) < 6:
        return 0.0
    signs = []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        signs.append(1 if diff > 0 else (-1 if diff < 0 else 0))
    turns = sum(1 for i in range(1, len(signs)) if signs[i] != 0 and signs[i - 1] != 0 and signs[i] != signs[i - 1])
    range_norm = min((max(values) - min(values)) * 10.0, 1.0)
    loopiness = turns / max(len(signs) - 1, 1)
    return _clamp01(0.60 * loopiness + 0.40 * range_norm)


def neural_sde_stability_proxy(values: list[float], *, steps: int | None = None) -> float:
    steps = steps or _env_int("QUANT_MODEL_NEURAL_SDE_STEPS", 32, low=4, high=512)
    if len(values) < 4:
        return 0.0
    drift = sum(values[-min(len(values), 6) :]) / max(min(len(values), 6), 1)
    vol = math.sqrt(sum((x - drift) ** 2 for x in values[-min(len(values), 12) :]) / max(min(len(values), 12), 1))
    step_penalty = min(math.sqrt(steps) / 32.0, 1.0)
    return _clamp01((1.0 - min(vol / 0.08, 1.0)) * (0.75 + 0.25 * step_penalty))


def kan_hedging_confidence_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    greek_conf = _safe_float(features.get("higher_order_greeks_confidence_norm"), _safe_float(features.get("quant_mlx_jump_diffusion_grad_norm"), 0.45))
    hedge_cost = _safe_float(features.get("hedge_cost_pressure_norm"), 0.30)
    surface_stability = _safe_float(features.get("options_surface_stability_norm"), 0.55)
    return _clamp01(0.40 * greek_conf + 0.35 * surface_stability + 0.25 * (1.0 - hedge_cost))


def vpin_order_flow_toxicity_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    imbalance = abs(_safe_float(features.get("order_book_imbalance_norm"), _safe_float(features.get("flow_direction_signed"), 0.0)))
    spread = _safe_float(features.get("bid_ask_spread_stress_norm"), _safe_float(features.get("spread_stress_norm"), 0.25))
    quote_fade = _safe_float(features.get("quote_fade_risk_norm"), 0.25)
    sweep = _safe_float(features.get("sweep_block_flow_toxicity_norm"), 0.35)
    return _clamp01(0.35 * imbalance + 0.25 * spread + 0.20 * quote_fade + 0.20 * sweep)


def signature_transform_path_dna(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    increments = [values[i] - values[i - 1] for i in range(1, len(values))]
    level_1 = abs(sum(increments))
    level_2 = abs(sum(increments[i] * increments[j] for i in range(len(increments)) for j in range(i + 1, len(increments))))
    energy = sum(abs(x) for x in increments)
    return _clamp01(0.45 * min(level_1 * 14.0, 1.0) + 0.35 * min(level_2 * 120.0, 1.0) + 0.20 * min(energy * 10.0, 1.0))


def multivariate_hawkes_self_exciting_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    event_rate = _safe_float(features.get("event_burst_rate_norm"), _safe_float(features.get("macro_event_density_norm"), 0.25))
    flow_cluster = _safe_float(features.get("order_flow_cluster_norm"), _safe_float(features.get("sweep_block_flow_toxicity_norm"), 0.35))
    vol_cluster = _safe_float(features.get("volatility_cluster_norm"), _safe_float(features.get("options_vol_of_vol_change_norm"), 0.35))
    decay = _safe_float(features.get("event_decay_norm"), 0.45)
    return _clamp01(0.30 * event_rate + 0.30 * flow_cluster + 0.25 * vol_cluster + 0.15 * (1.0 - decay))


def signature_market_generator_proxy(values: list[float]) -> float:
    if len(values) < 6:
        return 0.0
    signature = signature_transform_path_dna(values)
    tda = topological_data_analysis_proxy(values)
    rough = rough_volatility_fbm_proxy(values)
    return _clamp01(0.45 * signature + 0.30 * tda + 0.25 * rough)


def mean_field_game_crowd_pressure_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    breadth = 1.0 - _safe_float(features.get("market_breadth_health_norm"), 0.55)
    crowding = _safe_float(features.get("crowding_pressure_norm"), _safe_float(features.get("narrative_crowding_norm"), 0.35))
    correlation = _safe_float(features.get("cross_sleeve_correlation_pressure_norm"), _safe_float(features.get("quant_copula_dependency_norm"), 0.35))
    return _clamp01(0.35 * breadth + 0.35 * crowding + 0.30 * correlation)


def physics_informed_nn_constraint_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    no_arbitrage = _safe_float(features.get("no_arbitrage_consistency_norm"), 0.60)
    monotonicity = _safe_float(features.get("surface_monotonicity_norm"), 0.55)
    calendar = _safe_float(features.get("calendar_spread_consistency_norm"), 0.55)
    residual = _safe_float(features.get("pricing_pde_residual_norm"), 0.35)
    return _clamp01(0.25 * no_arbitrage + 0.25 * monotonicity + 0.25 * calendar + 0.25 * (1.0 - residual))


def hurst_exponent_proxy(values: list[float]) -> float:
    if len(values) < 8:
        return 0.5
    mean = sum(values) / len(values)
    demeaned = [x - mean for x in values]
    cumulative = []
    acc = 0.0
    for value in demeaned:
        acc += value
        cumulative.append(acc)
    r = max(cumulative) - min(cumulative)
    s = math.sqrt(sum((x - mean) ** 2 for x in values) / max(len(values), 1))
    rs = r / max(s, 1e-9)
    hurst = math.log(max(rs, 1.000001)) / math.log(max(len(values), 2))
    return _clamp(hurst, 0.0, 1.0)


def stochastic_differential_game_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    crowd = mean_field_game_crowd_pressure_proxy(features)
    hedge = kan_hedging_confidence_proxy(features)
    execution = execution_microstructure_awareness_proxy(features)
    toxicity = vpin_order_flow_toxicity_proxy(features)
    return _clamp01(0.30 * crowd + 0.25 * hedge + 0.25 * execution + 0.20 * (1.0 - toxicity))


def limit_order_book_transformer_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    seq_len = _env_int("QUANT_MODEL_TRANSFORMER_SEQUENCE", 64, low=8, high=1024)
    imbalance = abs(_safe_float(features.get("order_book_imbalance_norm"), _safe_float(features.get("flow_direction_signed"), 0.0)))
    depth = _safe_float(features.get("book_depth_health_norm"), _safe_float(features.get("market_micro_tradeability_score_norm"), 0.55))
    spread = _safe_float(features.get("bid_ask_spread_stress_norm"), _safe_float(features.get("spread_stress_norm"), 0.25))
    fade = _safe_float(features.get("quote_fade_risk_norm"), 0.25)
    coverage = min(math.log2(max(seq_len, 8)) / 10.0, 1.0)
    return _clamp01(0.30 * imbalance + 0.25 * depth + 0.20 * (1.0 - spread) + 0.15 * (1.0 - fade) + 0.10 * coverage)


def graph_laplacian_tda_diffusion_proxy(values: list[float], features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    node_cap = _env_int("QUANT_MODEL_LAPLACIAN_NODE_CAP", 24, low=4, high=512)
    tda_shape = topological_data_analysis_proxy(values)
    graph_structure = graph_neural_network_structure_proxy(features)
    corr_pressure = _safe_float(features.get("cross_sleeve_correlation_pressure_norm"), _safe_float(features.get("quant_copula_dependency_norm"), 0.35))
    smoothness = 1.0 - min(abs(tda_shape - graph_structure), 1.0)
    cap_factor = min(math.log2(max(node_cap, 4)) / 9.0, 1.0)
    return _clamp01(0.35 * tda_shape + 0.25 * graph_structure + 0.20 * smoothness + 0.10 * corr_pressure + 0.10 * cap_factor)


def agentic_self_correction_critic_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    drift = _safe_float(features.get("model_drift_risk_norm"), 0.25)
    blocked = _safe_float(features.get("blocked_rate_norm"), _safe_float(features.get("combined_blocked_rate"), 0.20))
    replay = _safe_float(features.get("golden_replay_pass_rate_norm"), 0.65)
    correction = _safe_float(features.get("critic_correction_success_norm"), 0.50)
    disagreement = _safe_float(features.get("observer_critic_disagreement_norm"), 0.25)
    return _clamp01(0.25 * (1.0 - drift) + 0.20 * (1.0 - blocked) + 0.20 * replay + 0.20 * correction + 0.15 * (1.0 - disagreement))


def nonhomogeneous_hmm_confidence_proxy(values: list[float], features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    states = _env_int("QUANT_MODEL_NHHMM_STATES", 3, low=2, high=8)
    regime = regime_switching_filter(values, states=states)
    covariate = max(
        _safe_float(features.get("macro_event_density_norm"), 0.0),
        _safe_float(features.get("options_surface_change_norm"), 0.0),
        _safe_float(features.get("volatility_cluster_norm"), 0.0),
    )
    data = min(len(values) / 7.0, 1.0)
    return _clamp01(0.50 * _safe_float(regime.get("confidence"), 0.0) + 0.30 * covariate + 0.20 * data)


def observer_critic_loop_proxy(features: Mapping[str, Any] | None = None, external_snapshots: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    observer = alternative_data_signal_proxy(features, external_snapshots)
    critic = agentic_self_correction_critic_proxy(features)
    latency = low_latency_orchestration_readiness_proxy(features)
    provenance = _safe_float(features.get("decision_provenance_coverage_norm"), _safe_float(features.get("source_confidence_norm"), 0.55))
    return _clamp01(0.30 * observer + 0.30 * critic + 0.20 * latency + 0.20 * provenance)


def physics_informed_neural_sde_proxy(values: list[float], features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    steps = _env_int("QUANT_MODEL_PIN_SDE_STEPS", 24, low=4, high=512)
    neural = neural_sde_stability_proxy(values, steps=steps)
    physics = physics_informed_nn_constraint_proxy(features)
    rough = rough_volatility_fbm_proxy(values)
    residual = _safe_float(features.get("neural_sde_constraint_residual_norm"), _safe_float(features.get("pricing_pde_residual_norm"), 0.30))
    return _clamp01(0.35 * neural + 0.35 * physics + 0.15 * rough + 0.15 * (1.0 - residual))


def geometric_order_book_transformer_proxy(values: list[float], features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    lit = limit_order_book_transformer_proxy(features)
    laplacian = graph_laplacian_tda_diffusion_proxy(values, features)
    micro = execution_microstructure_awareness_proxy(features)
    toxicity = vpin_order_flow_toxicity_proxy(features)
    return _clamp01(0.30 * lit + 0.25 * laplacian + 0.25 * micro + 0.20 * (1.0 - toxicity))


def double_machine_learning_causal_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    folds = _env_int("QUANT_MODEL_DML_CROSSFIT_FOLDS", 3, low=2, high=10)
    treatment_conf = _safe_float(features.get("causal_treatment_signal_norm"), _safe_float(features.get("source_weighted_event_followthrough_norm"), 0.40))
    confounder = _safe_float(features.get("confounder_balance_norm"), _safe_float(features.get("cross_sleeve_correlation_pressure_norm"), 0.35))
    residual = _safe_float(features.get("causal_residual_stability_norm"), 0.55)
    data = _safe_float(features.get("quant_model_data_confidence_norm"), 0.50)
    fold_quality = min(math.log2(max(folds, 2)) / 4.0, 1.0)
    return _clamp01(0.30 * treatment_conf + 0.25 * (1.0 - confounder) + 0.20 * residual + 0.15 * data + 0.10 * fold_quality)


def neuro_symbolic_agent_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    rule = _safe_float(features.get("rule_consistency_norm"), _safe_float(features.get("no_arbitrage_consistency_norm"), 0.55))
    neural = _safe_float(features.get("adaptive_policy_confidence_norm"), _safe_float(features.get("quant_actor_critic_policy_confidence_norm"), 0.45))
    provenance = _safe_float(features.get("decision_provenance_coverage_norm"), _safe_float(features.get("source_confidence_norm"), 0.55))
    contradiction = _safe_float(features.get("symbolic_contradiction_risk_norm"), 0.20)
    return _clamp01(0.30 * rule + 0.25 * neural + 0.25 * provenance + 0.20 * (1.0 - contradiction))


def cross_modal_embedding_omni_sensor_proxy(features: Mapping[str, Any] | None = None, external_snapshots: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    snapshots = external_snapshots or {}
    modalities = [
        "macro",
        "news",
        "sec",
        "market_micro",
        "options",
        "crypto",
        "fx",
        "audio",
        "transcript",
    ]
    available = 0
    if isinstance(snapshots, Mapping):
        for key, value in snapshots.items():
            if bool(value) and any(token in str(key).lower() for token in modalities):
                available += 1
    coverage = min(available / 5.0, 1.0)
    embedding_dim = _env_int("QUANT_MODEL_CROSS_MODAL_EMBED_DIM", 128, low=16, high=2048)
    dim_quality = min(math.log2(max(embedding_dim, 16)) / 11.0, 1.0)
    source_conf = _safe_float(features.get("source_confidence_norm"), 0.55)
    alt = alternative_data_signal_proxy(features, snapshots)
    return _clamp01(0.30 * coverage + 0.25 * dim_quality + 0.25 * source_conf + 0.20 * alt)


def rlbf_backtracking_feedback_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    backtracks = _env_int("QUANT_MODEL_RLBF_BACKTRACK_CAP", 8, low=1, high=128)
    critic = agentic_self_correction_critic_proxy(features)
    reward = _safe_float(features.get("reward_stability_norm"), _safe_float(features.get("execution_fitness_norm"), 0.50))
    rollback = _safe_float(features.get("rollback_success_norm"), _safe_float(features.get("golden_replay_pass_rate_norm"), 0.60))
    penalty = _safe_float(features.get("backtracking_penalty_norm"), 0.20)
    cap_quality = min(math.log2(max(backtracks, 1) + 1.0) / 7.0, 1.0)
    return _clamp01(0.30 * critic + 0.25 * reward + 0.20 * rollback + 0.15 * (1.0 - penalty) + 0.10 * cap_quality)


def differentiable_market_simulator_proxy(values: list[float], features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    steps = _env_int("QUANT_MODEL_DMS_STEPS", 32, low=4, high=512)
    path = functional_ito_path_volatility(values)
    micro = execution_microstructure_awareness_proxy(features)
    gradient = _safe_float(features.get("simulator_gradient_stability_norm"), _safe_float(features.get("quant_mlx_jump_diffusion_grad_norm"), 0.45))
    slippage = _safe_float(features.get("slippage_pressure_norm"), _safe_float(features.get("hedge_cost_pressure_norm"), 0.25))
    step_quality = min(math.log2(max(steps, 4)) / 9.0, 1.0)
    return _clamp01(0.25 * path + 0.25 * micro + 0.25 * gradient + 0.15 * (1.0 - slippage) + 0.10 * step_quality)


def equivariant_neural_network_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    channels = _env_int("QUANT_MODEL_EQUIVARIANT_CHANNELS", 16, low=4, high=256)
    graph = graph_neural_network_structure_proxy(features)
    symmetry = _safe_float(features.get("symmetry_consistency_norm"), _safe_float(features.get("sector_rotation_symmetry_norm"), 0.50))
    cross_asset = _safe_float(features.get("cross_asset_invariance_norm"), 1.0 - _safe_float(features.get("cross_sleeve_correlation_pressure_norm"), 0.35))
    overfit = _safe_float(features.get("equivariant_overfit_risk_norm"), _safe_float(features.get("model_drift_risk_norm"), 0.25))
    channel_quality = min(math.log2(max(channels, 4)) / 8.0, 1.0)
    return _clamp01(0.25 * graph + 0.25 * symmetry + 0.20 * cross_asset + 0.20 * (1.0 - overfit) + 0.10 * channel_quality)


def quant_model_resource_profile() -> dict[str, float]:
    workers = _env_int("QUANT_MODEL_MAX_WORKERS", 1, low=1, high=16)
    mc_paths = _env_int("QUANT_MODEL_MONTE_CARLO_PATHS", 512, low=32, high=8192)
    qmc_paths = _env_int("QUANT_MODEL_QUASI_MONTE_CARLO_PATHS", 384, low=32, high=8192)
    lhs_paths = _env_int("QUANT_MODEL_LATIN_HYPERCUBE_PATHS", 384, low=32, high=8192)
    fd_grid = _env_int("QUANT_MODEL_FINITE_DIFF_GRID", 64, low=24, high=240)
    fft_grid = _env_int("QUANT_MODEL_FFT_GRID", 512, low=64, high=4096)
    tri_steps = _env_int("QUANT_MODEL_TRINOMIAL_STEPS", 64, low=8, high=512)
    particles = _env_int("QUANT_MODEL_PARTICLE_COUNT", 128, low=32, high=2048)
    ga_pop = _env_int("QUANT_MODEL_GA_POPULATION", 12, low=4, high=128)
    ga_gen = _env_int("QUANT_MODEL_GA_GENERATIONS", 4, low=1, high=64)
    actor_rollouts = _env_int("QUANT_MODEL_ACTOR_CRITIC_ROLLOUTS", 64, low=8, high=2048)
    graph_nodes = _env_int("QUANT_MODEL_GRAPH_NODE_CAP", 24, low=4, high=256)
    micro_lookback = _env_int("QUANT_MODEL_MICROSTRUCTURE_LOOKBACK", 240, low=30, high=3600)
    regime_states = _env_int("QUANT_MODEL_REGIME_FILTER_STATES", 2, low=2, high=6)
    gpu_mc_paths = _env_int("QUANT_MODEL_GPU_MONTE_CARLO_PATHS", 1024, low=32, high=2_000_000)
    gpu_kalman_window = _env_int("QUANT_MODEL_GPU_KALMAN_WINDOW", 128, low=8, high=4096)
    neural_sde_steps = _env_int("QUANT_MODEL_NEURAL_SDE_STEPS", 32, low=4, high=512)
    signature_depth = _env_int("QUANT_MODEL_SIGNATURE_DEPTH", 2, low=1, high=6)
    hawkes_windows = _env_int("QUANT_MODEL_HAWKES_WINDOWS", 32, low=4, high=512)
    transformer_seq = _env_int("QUANT_MODEL_TRANSFORMER_SEQUENCE", 64, low=8, high=1024)
    laplacian_nodes = _env_int("QUANT_MODEL_LAPLACIAN_NODE_CAP", 24, low=4, high=512)
    critic_replay = _env_int("QUANT_MODEL_CRITIC_REPLAY_CAP", 128, low=8, high=4096)
    nhhmm_states = _env_int("QUANT_MODEL_NHHMM_STATES", 3, low=2, high=8)
    pin_sde_steps = _env_int("QUANT_MODEL_PIN_SDE_STEPS", 24, low=4, high=512)
    dml_folds = _env_int("QUANT_MODEL_DML_CROSSFIT_FOLDS", 3, low=2, high=10)
    cross_modal_dim = _env_int("QUANT_MODEL_CROSS_MODAL_EMBED_DIM", 128, low=16, high=2048)
    rlbf_backtracks = _env_int("QUANT_MODEL_RLBF_BACKTRACK_CAP", 8, low=1, high=128)
    dms_steps = _env_int("QUANT_MODEL_DMS_STEPS", 32, low=4, high=512)
    equivariant_channels = _env_int("QUANT_MODEL_EQUIVARIANT_CHANNELS", 16, low=4, high=256)
    pressure = (
        0.15 * workers / 8.0
        + 0.10 * mc_paths / 4096.0
        + 0.05 * qmc_paths / 4096.0
        + 0.05 * lhs_paths / 4096.0
        + 0.10 * fd_grid / 160.0
        + 0.10 * fft_grid / 2048.0
        + 0.08 * tri_steps / 256.0
        + 0.12 * particles / 1024.0
        + 0.07 * ga_pop / 64.0
        + 0.07 * ga_gen / 32.0
        + 0.09 * actor_rollouts / 1024.0
        + 0.06 * graph_nodes / 128.0
        + 0.04 * micro_lookback / 1800.0
        + 0.02 * regime_states / 6.0
        + 0.07 * gpu_mc_paths / 500000.0
        + 0.03 * gpu_kalman_window / 2048.0
        + 0.03 * neural_sde_steps / 256.0
        + 0.02 * signature_depth / 6.0
        + 0.02 * hawkes_windows / 256.0
        + 0.03 * transformer_seq / 512.0
        + 0.02 * laplacian_nodes / 256.0
        + 0.02 * critic_replay / 2048.0
        + 0.01 * nhhmm_states / 8.0
        + 0.02 * pin_sde_steps / 256.0
        + 0.01 * dml_folds / 10.0
        + 0.02 * cross_modal_dim / 1024.0
        + 0.01 * rlbf_backtracks / 64.0
        + 0.02 * dms_steps / 256.0
        + 0.01 * equivariant_channels / 128.0
    )
    return {
        "workers": float(workers),
        "monte_carlo_paths": float(mc_paths),
        "quasi_monte_carlo_paths": float(qmc_paths),
        "latin_hypercube_paths": float(lhs_paths),
        "finite_diff_grid": float(fd_grid),
        "fft_grid": float(fft_grid),
        "trinomial_steps": float(tri_steps),
        "particle_count": float(particles),
        "ga_population": float(ga_pop),
        "ga_generations": float(ga_gen),
        "actor_critic_rollouts": float(actor_rollouts),
        "graph_node_cap": float(graph_nodes),
        "microstructure_lookback": float(micro_lookback),
        "regime_filter_states": float(regime_states),
        "gpu_monte_carlo_paths": float(gpu_mc_paths),
        "gpu_kalman_window": float(gpu_kalman_window),
        "neural_sde_steps": float(neural_sde_steps),
        "signature_depth": float(signature_depth),
        "hawkes_windows": float(hawkes_windows),
        "transformer_sequence": float(transformer_seq),
        "laplacian_node_cap": float(laplacian_nodes),
        "critic_replay_cap": float(critic_replay),
        "nhhmm_states": float(nhhmm_states),
        "pin_sde_steps": float(pin_sde_steps),
        "dml_crossfit_folds": float(dml_folds),
        "cross_modal_embed_dim": float(cross_modal_dim),
        "rlbf_backtrack_cap": float(rlbf_backtracks),
        "dms_steps": float(dms_steps),
        "equivariant_channels": float(equivariant_channels),
        "mlx_runtime_available": 1.0 if mx is not None else 0.0,
        "mlx_compile_available": 1.0 if mx is not None and hasattr(mx, "compile") else 0.0,
        "resource_pressure_norm": round(_clamp01(pressure), 4),
    }


def _returns_from_features(features: Mapping[str, Any]) -> list[float]:
    candidates = [
        _safe_float(features.get("mom_1m", 0.0)),
        _safe_float(features.get("mom_5m", 0.0)),
        _safe_float(features.get("mom_15m", 0.0)),
        _safe_float(features.get("pct_from_close", 0.0)),
        _safe_float(features.get("ctx_SPY_mom_5m", 0.0)),
        _safe_float(features.get("ctx_QQQ_mom_5m", 0.0)),
        -_safe_float(features.get("ctx_VIX_X_pct_from_close", 0.0)),
    ]
    out = [x for x in candidates if math.isfinite(x)]
    return out or [0.0]


def summarize_quant_model_features(
    features: Mapping[str, Any] | None = None,
    *,
    external_snapshots: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    features = features or {}
    spot = max(
        _safe_float(features.get("last_price"), 0.0)
        or _safe_float(features.get("close"), 0.0)
        or _safe_float(features.get("ctx_SPY_last_price"), 0.0)
        or 100.0,
        1e-6,
    )
    strike = max(_safe_float(features.get("atm_strike"), spot), 1e-6)
    expiry_days = max(_safe_float(features.get("expiry_days"), 30.0), 1.0)
    time_years = expiry_days / 365.0
    rate = _safe_float(features.get("risk_free_rate"), 0.045)
    vol = max(
        _safe_float(features.get("implied_volatility"), 0.0)
        or _safe_float(features.get("options_iv_percentile_norm"), 0.25) * 0.55
        or _safe_float(features.get("vol_30m"), 0.18),
        0.05,
    )
    vol = min(vol, 1.50)
    returns = _returns_from_features(features)
    context_returns = [
        _safe_float(features.get("ctx_TLT_mom_5m", 0.0)),
        _safe_float(features.get("ctx_UUP_mom_5m", 0.0)),
        _safe_float(features.get("market_crypto_risk_corr_norm", 0.0)) - 0.5,
        _safe_float(features.get("flow_direction_signed", 0.0)),
    ]
    mc = monte_carlo_gbm_price(spot, strike, time_years, rate, vol)
    qmc = quasi_monte_carlo_gbm_price(spot, strike, time_years, rate, vol)
    lhs = latin_hypercube_gbm_price(spot, strike, time_years, rate, vol)
    gpu_mc = gpu_accelerated_monte_carlo_price(spot, strike, time_years, rate, vol)
    fd = finite_difference_price(spot, strike, time_years, rate, vol)
    fft = fft_lognormal_call_price(spot, strike, time_years, rate, vol)
    tri = trinomial_tree_price(spot, strike, time_years, rate, vol)
    merton = merton_jump_diffusion_price(
        spot,
        strike,
        time_years,
        rate,
        vol,
        jump_intensity=max(_safe_float(features.get("market_micro_gap_fade_risk_norm"), 0.25), 0.05),
    )
    heston = heston_stochastic_vol_proxy(
        vol,
        _safe_float(features.get("options_vol_of_vol_change_norm"), 0.35),
        _safe_float(features.get("quant_heston_mean_reversion"), 1.5),
        max(vol * vol * 0.85, 1e-9),
    )
    resource = quant_model_resource_profile()
    price_scale = max(spot, 1.0)
    bs_price = black_scholes_price(spot, strike, time_years, rate, vol)
    plain_error = abs(mc - bs_price)
    variance_reduced_error = min(abs(qmc - bs_price), abs(lhs - bs_price))
    antithetic_efficiency = _clamp01(1.0 - variance_reduced_error / max(plain_error + 1e-9, bs_price * 0.03, 1e-9))
    kalman = kalman_filter_level(returns)
    gpu_kalman = real_time_kalman_filter_gpu(returns)
    particle = particle_filter_level(returns)
    kelly = kelly_fraction(
        0.5 + 0.25 * _clamp(_safe_float(features.get("flow_direction_signed"), 0.0), -1.0, 1.0),
        1.0 + abs(_safe_float(features.get("edge_norm"), _safe_float(features.get("execution_fitness_norm"), 0.5))),
    )
    cvar = conditional_value_at_risk(returns, alpha=0.95)
    copula = gaussian_copula_dependency_proxy(returns, context_returns)
    ou = ornstein_uhlenbeck_signal(returns)
    ga = genetic_optimize_parameters([0.5 + min(max(x, -0.49), 0.49) for x in returns])
    actor_critic = actor_critic_policy_proxy(returns)
    graph_structure = graph_neural_network_structure_proxy(features)
    microstructure = execution_microstructure_awareness_proxy(features)
    regime_switch = regime_switching_filter(returns)
    mlx_grad = mlx_jump_diffusion_gradient(
        spot,
        strike,
        time_years,
        rate,
        vol,
        jump_intensity=max(_safe_float(features.get("market_micro_gap_fade_risk_norm"), 0.25), 0.05),
    )
    adversarial = adversarial_ml_resilience_proxy(features)
    latency = low_latency_orchestration_readiness_proxy(features)
    alt_data = alternative_data_signal_proxy(features, external_snapshots)
    zkp = zkp_privacy_readiness_proxy(features)
    qemc = quantum_enhanced_monte_carlo_proxy(
        {
            **dict(features),
            "quant_quasi_monte_carlo_price_norm": _clamp01(qmc / price_scale),
            "quant_antithetic_variates_efficiency_norm": antithetic_efficiency,
        }
    )
    path_vol = functional_ito_path_volatility(returns)
    rough_vol = rough_volatility_fbm_proxy(returns)
    ot_bridge = optimal_transport_bridge_proxy(returns, context_returns)
    tda = topological_data_analysis_proxy(returns)
    neural_sde = neural_sde_stability_proxy(returns)
    kan = kan_hedging_confidence_proxy({**dict(features), "quant_mlx_jump_diffusion_grad_norm": _clamp01((abs(mlx_grad.get("delta", 0.0)) + min(abs(mlx_grad.get("gamma", 0.0)), 1.0)) / 2.0)})
    vpin = vpin_order_flow_toxicity_proxy(features)
    signature = signature_transform_path_dna(returns)
    hawkes = multivariate_hawkes_self_exciting_proxy(features)
    signature_gen = signature_market_generator_proxy(returns)
    mean_field = mean_field_game_crowd_pressure_proxy(features)
    pinn = physics_informed_nn_constraint_proxy(features)
    hurst = hurst_exponent_proxy(returns)
    sdg = stochastic_differential_game_proxy(features)
    lit = limit_order_book_transformer_proxy(features)
    laplacian = graph_laplacian_tda_diffusion_proxy(returns, features)
    critic = agentic_self_correction_critic_proxy(features)
    nhhmm = nonhomogeneous_hmm_confidence_proxy(returns, features)
    observer_critic = observer_critic_loop_proxy(features, external_snapshots)
    pin_sde = physics_informed_neural_sde_proxy(returns, features)
    geometric_lit = geometric_order_book_transformer_proxy(returns, features)
    dml = double_machine_learning_causal_proxy(features)
    neuro_symbolic = neuro_symbolic_agent_proxy(features)
    omni_sensor = cross_modal_embedding_omni_sensor_proxy(features, external_snapshots)
    rlbf = rlbf_backtracking_feedback_proxy(features)
    dms = differentiable_market_simulator_proxy(returns, features)
    equivariant = equivariant_neural_network_proxy(features)
    snapshots = external_snapshots or {}
    external_conf = 0.0
    if isinstance(snapshots, Mapping) and snapshots:
        external_conf = min(sum(1 for value in snapshots.values() if bool(value)) / max(len(snapshots), 1), 1.0)
    data_conf = _clamp01(
        0.30
        + 0.20 * min(len(returns) / 7.0, 1.0)
        + 0.20 * _safe_float(features.get("market_micro_tradeability_score_norm"), 0.5)
        + 0.15 * _safe_float(features.get("options_surface_change_norm"), 0.0)
        + 0.15 * external_conf
    )
    out = default_quant_model_features()
    out.update(
        {
            "quant_model_engine_available": 1.0,
            "quant_pricing_coverage_norm": 1.0,
            "quant_state_filter_coverage_norm": 1.0,
            "quant_tail_risk_coverage_norm": 1.0,
            "quant_optimization_coverage_norm": 1.0,
            "quant_adaptive_architecture_coverage_norm": 1.0,
            "quant_monte_carlo_price_norm": _clamp01(mc / price_scale),
            "quant_quasi_monte_carlo_price_norm": _clamp01(qmc / price_scale),
            "quant_latin_hypercube_price_norm": _clamp01(lhs / price_scale),
            "quant_antithetic_variates_efficiency_norm": antithetic_efficiency,
            "quant_finite_difference_price_norm": _clamp01(fd / price_scale),
            "quant_fft_price_norm": _clamp01(fft / price_scale),
            "quant_trinomial_tree_price_norm": _clamp01(tri / price_scale),
            "quant_heston_vol_risk_norm": heston,
            "quant_merton_jump_risk_norm": _clamp01(abs(merton - bs_price) / price_scale * 4.0),
            "quant_kalman_filter_confidence_norm": _clamp01(kalman.get("confidence", 0.0)),
            "quant_particle_filter_confidence_norm": _clamp01(particle.get("confidence", 0.0)),
            "quant_kelly_fraction_norm": _clamp01((kelly + 1.0) / 2.0),
            "quant_cvar_tail_risk_norm": _clamp01(abs(cvar) * 16.0),
            "quant_copula_dependency_norm": copula,
            "quant_ou_mean_reversion_norm": _clamp01((ou + 1.0) / 2.0),
            "quant_genetic_optimization_stability_norm": _clamp01(ga.get("stability", 0.0)),
            "quant_actor_critic_policy_confidence_norm": _clamp01(actor_critic.get("confidence", 0.0)),
            "quant_graph_neural_network_structure_norm": graph_structure,
            "quant_execution_microstructure_awareness_norm": microstructure,
            "quant_regime_switch_filter_confidence_norm": _clamp01(regime_switch.get("confidence", 0.0)),
            "quant_adversarial_ml_resilience_norm": adversarial,
            "quant_low_latency_orchestration_readiness_norm": latency,
            "quant_alternative_data_signal_norm": alt_data,
            "quant_zkp_privacy_readiness_norm": zkp,
            "quant_gpu_monte_carlo_acceleration_norm": _clamp01(gpu_mc.get("acceleration", 0.0)),
            "quant_gpu_kalman_filter_confidence_norm": _clamp01(gpu_kalman.get("confidence", 0.0)),
            "quant_mlx_jump_diffusion_grad_norm": _clamp01((abs(mlx_grad.get("delta", 0.0)) + min(abs(mlx_grad.get("gamma", 0.0)), 1.0)) / 2.0),
            "quant_mlx_runtime_available_norm": 1.0 if mx is not None else 0.0,
            "quant_mlx_compile_available_norm": 1.0 if mx is not None and hasattr(mx, "compile") else 0.0,
            "quant_qemc_signal_norm": qemc,
            "quant_path_dependent_volatility_norm": path_vol,
            "quant_rough_volatility_fbm_norm": rough_vol,
            "quant_optimal_transport_bridge_norm": ot_bridge,
            "quant_tda_regime_shape_norm": tda,
            "quant_neural_sde_stability_norm": neural_sde,
            "quant_kan_hedging_confidence_norm": kan,
            "quant_vpin_order_flow_toxicity_norm": vpin,
            "quant_signature_path_dna_norm": signature,
            "quant_hawkes_self_exciting_norm": hawkes,
            "quant_signature_market_generator_norm": signature_gen,
            "quant_mean_field_crowd_pressure_norm": mean_field,
            "quant_pinn_constraint_consistency_norm": pinn,
            "quant_hurst_exponent_norm": hurst,
            "quant_stochastic_differential_game_norm": sdg,
            "quant_limit_order_book_transformer_norm": lit,
            "quant_graph_laplacian_tda_diffusion_norm": laplacian,
            "quant_agentic_self_correction_critic_norm": critic,
            "quant_nonhomogeneous_hmm_confidence_norm": nhhmm,
            "quant_observer_critic_loop_norm": observer_critic,
            "quant_physics_informed_neural_sde_norm": pin_sde,
            "quant_geometric_order_book_transformer_norm": geometric_lit,
            "quant_double_machine_learning_causal_norm": dml,
            "quant_neuro_symbolic_agent_norm": neuro_symbolic,
            "quant_cross_modal_embedding_omni_sensor_norm": omni_sensor,
            "quant_rlbf_backtracking_feedback_norm": rlbf,
            "quant_differentiable_market_simulator_norm": dms,
            "quant_equivariant_neural_network_norm": equivariant,
            "quant_model_resource_pressure_norm": _clamp01(resource["resource_pressure_norm"]),
            "quant_model_data_confidence_norm": data_conf,
        }
    )
    return out


def quant_model_inventory() -> dict[str, Any]:
    resource = quant_model_resource_profile()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "implemented_models": [
            "monte_carlo_simulation",
            "quasi_monte_carlo_simulation",
            "latin_hypercube_sampling",
            "antithetic_variates",
            "finite_difference_pricing",
            "fast_fourier_transform_pricing",
            "trinomial_tree_pricing",
            "heston_stochastic_volatility_proxy",
            "merton_jump_diffusion",
            "kalman_filter",
            "particle_filter",
            "kelly_criterion",
            "conditional_value_at_risk",
            "gaussian_copula_dependency_proxy",
            "ornstein_uhlenbeck_process_signal",
            "order_book_imbalance_feature_hooks",
            "genetic_algorithm_backtest_optimization",
            "sentiment_analysis_agent_hooks",
            "actor_critic_policy_proxy",
            "graph_neural_network_structure_proxy",
            "execution_microstructure_awareness_proxy",
            "regime_switching_filter",
            "adversarial_machine_learning_resilience_proxy",
            "low_latency_agent_orchestration_proxy",
            "alternative_data_ingestion_signal_proxy",
            "zero_knowledge_proof_privacy_readiness_proxy",
            "gpu_accelerated_monte_carlo",
            "real_time_kalman_filter_gpu",
            "mlx_grad_jump_diffusion_greeks",
            "mlx_compile_fair_value_path",
            "quantum_enhanced_monte_carlo_proxy",
            "functional_ito_path_dependent_volatility",
            "rough_volatility_fractional_brownian_motion_proxy",
            "optimal_transport_schrodinger_bridge_proxy",
            "topological_data_analysis_regime_shape_proxy",
            "neural_sde_stability_proxy",
            "kolmogorov_arnold_network_hedging_proxy",
            "vpin_order_flow_toxicity_proxy",
            "signature_transforms_path_dna",
            "multivariate_hawkes_self_exciting_proxy",
            "signature_based_market_generator_proxy",
            "mean_field_game_crowd_engine_proxy",
            "physics_informed_neural_network_constraint_proxy",
            "fractional_brownian_motion_hurst_exponent_proxy",
            "stochastic_differential_game_proxy",
            "limit_order_book_transformer_lit_proxy",
            "graph_laplacian_diffusion_for_tda_proxy",
            "agentic_self_correction_critic_loop_proxy",
            "nonhomogeneous_hidden_markov_model_proxy",
            "observer_critic_loop_proxy",
            "physics_informed_neural_sde_proxy",
            "geometric_order_book_transformer_glit_proxy",
            "double_machine_learning_causal_inference_proxy",
            "neuro_symbolic_agent_integration_proxy",
            "unified_cross_modal_embeddings_omni_sensor_proxy",
            "reinforcement_learning_with_backtracking_feedback_proxy",
            "differentiable_market_simulator_proxy",
            "equivariant_neural_network_symmetry_proxy",
        ],
        "feature_keys": list(QUANT_MODEL_FEATURE_KEYS),
        "resource_profile": resource,
        "mlx_hooks": {
            "mlx_core_random": mx is not None and hasattr(mx, "random"),
            "mx_grad": mx is not None and hasattr(mx, "grad"),
            "mlx_compile": mx is not None and hasattr(mx, "compile"),
            "fair_value_gradient": mx is not None and hasattr(mx, "grad"),
        },
        "execution_policy": {
            "direct_execution_allowed": False,
            "paper_trading_allowed": False,
            "purpose": "research_feature_collection_and_risk_context",
        },
    }
