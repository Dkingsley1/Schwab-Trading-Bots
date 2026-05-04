from __future__ import annotations

import math
import os
import random
from importlib import util as importlib_util
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

try:
    import QuantLib as ql
except Exception:  # pragma: no cover - QuantLib is optional in lightweight test/runtime shells.
    ql = None


def _module_available(name: str) -> bool:
    try:
        return importlib_util.find_spec(name) is not None
    except Exception:
        return False


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
    "quant_sabr_vol_surface_norm",
    "quant_svi_ssvi_vol_surface_norm",
    "quant_dupire_local_vol_surface_norm",
    "quant_bates_jump_vol_norm",
    "quant_hull_white_rates_norm",
    "quant_cir_intensity_norm",
    "quant_hjm_forward_rate_norm",
    "quant_sofr_market_model_norm",
    "quant_dcc_garch_correlation_norm",
    "quant_evt_pot_tail_norm",
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
    "quant_mlx_nn_available_norm",
    "quant_mlx_optimizers_available_norm",
    "quant_mlx_lm_available_norm",
    "quant_mlx_graphs_available_norm",
    "quant_mlx_snn_available_norm",
    "quant_mlx_vision_available_norm",
    "quant_esig_signature_available_norm",
    "quant_quantlib_available_norm",
    "quant_quantlib_pricing_benchmark_norm",
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
    "quant_dainn_arbitrage_invariant_norm",
    "quant_markovian_execution_control_norm",
    "quant_end_to_end_diff_backtest_norm",
    "quant_portfolio_durability_norm",
    "quant_information_geometry_manifold_norm",
    "quant_graph_attention_spillover_norm",
    "quant_agentic_wallet_intent_norm",
    "quant_rough_path_signature_kernel_norm",
    "quant_quantum_classical_hybrid_optimization_norm",
    "quant_formal_verification_safety_norm",
    "quant_lobdif_order_book_diffusion_norm",
    "quant_fractional_hurst_rough_vol_norm",
    "quant_differentiable_market_impact_norm",
    "quant_persistent_homology_flash_crash_norm",
    "quant_toxic_liquidity_injection_norm",
    "quant_flash_freeze_slippage_norm",
    "quant_photonic_quantum_optimization_norm",
    "quant_replication_crisis_shield_norm",
    "quant_synthetic_crisis_market_gan_norm",
    "quant_correlation_convergence_norm",
    "quant_macro_stress_2026_driver_norm",
    "quant_fed_2026_scenario_integrity_norm",
    "quant_fed_2026_equity_crash_vol_spike_norm",
    "quant_fed_2026_credit_spread_blowout_norm",
    "quant_fed_2026_housing_price_shock_norm",
    "quant_fed_2026_cre_price_shock_norm",
    "quant_fed_2026_unemployment_recession_norm",
    "quant_fed_2026_global_recession_deflation_norm",
    "quant_fed_2026_commodity_inflation_shock_norm",
    "quant_fed_2026_treasury_yield_shock_norm",
    "quant_fed_2026_usd_stress_norm",
    "quant_fed_2026_counterparty_default_contagion_norm",
    "quant_covid_2020_pandemic_replay_norm",
    "quant_mckean_vlasov_control_norm",
    "quant_tensor_network_mps_norm",
    "quant_multifidelity_stochastic_programming_norm",
    "quant_differentiable_tatonnement_norm",
    "quant_signature_lead_lag_detector_norm",
    "quant_chaos_propagation_norm",
    "quant_mckean_vlasov_sde_sensitivity_norm",
    "quant_mlmc_sequential_estimation_norm",
    "quant_signature_volterra_kernel_calibration_norm",
    "quant_dual_tatonnement_price_discovery_norm",
    "quant_probabilistic_propagation_of_chaos_norm",
    "quant_experience_accumulation_memory_norm",
    "quant_rough_vvix_exotics_norm",
    "quant_quantum_barrier_path_amplitude_norm",
    "quant_cross_asset_correlation_heat_swap_norm",
    "quant_cliquet_global_floor_local_cap_norm",
    "quant_signature_trend_follower_options_norm",
    "quant_esg_contingent_cds_norm",
    "quant_sdg_control_norm",
    "quant_nonlocal_fractional_laplacian_norm",
    "quant_infinite_dimensional_heston_norm",
    "quant_lie_group_rough_path_signature_norm",
    "quant_mean_field_games_controls_norm",
    "quant_wasserstein_gradient_flow_norm",
    "quant_malliavin_wiener_greeks_norm",
    "quant_tqft_braid_group_norm",
    "quant_mfgc_congestion_norm",
    "quant_spde_manifold_lob_fluid_norm",
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


def quantlib_black_scholes_price(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    *,
    option_type: str = "call",
) -> float | None:
    if ql is None:
        return None
    try:
        spot = max(_safe_float(spot), 1e-9)
        strike = max(_safe_float(strike), 1e-9)
        t = max(_safe_float(time_years), 1.0 / 365.0)
        vol = max(_safe_float(volatility), 1e-9)
        rate = _safe_float(rate)
        calculation_date = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = calculation_date
        maturity_date = calculation_date + max(int(round(t * 365.0)), 1)
        day_count = ql.Actual365Fixed()
        calendar = ql.NullCalendar()
        payoff_type = ql.Option.Put if str(option_type).lower().startswith("p") else ql.Option.Call
        option = ql.VanillaOption(ql.PlainVanillaPayoff(payoff_type, strike), ql.EuropeanExercise(maturity_date))
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(spot)),
            ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, 0.0, day_count)),
            ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, rate, day_count)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, calendar, vol, day_count)),
        )
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        return max(float(option.NPV()), 0.0)
    except Exception:
        return None


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


def sabr_vol_surface_proxy(
    forward: float,
    strike: float,
    time_years: float,
    volatility: float,
    *,
    beta: float = 0.5,
    rho: float = -0.35,
    vol_of_vol: float = 0.45,
) -> float:
    fwd = max(_safe_float(forward), 1e-9)
    k = max(_safe_float(strike), 1e-9)
    t = max(_safe_float(time_years), 1e-9)
    alpha = max(_safe_float(volatility), 1e-9)
    beta_f = _clamp(_safe_float(beta, 0.5), 0.0, 1.0)
    rho_f = _clamp(_safe_float(rho, -0.35), -0.999, 0.999)
    nu = max(_safe_float(vol_of_vol, 0.45), 0.0)
    log_moneyness = abs(math.log(fwd / k))
    skew_pressure = abs(rho_f) * nu * math.sqrt(t)
    smile_pressure = min(log_moneyness * nu / max(alpha, 1e-9), 2.0)
    beta_shape = abs(1.0 - beta_f)
    return _clamp01(0.30 * min(alpha / 1.50, 1.0) + 0.30 * skew_pressure + 0.25 * smile_pressure + 0.15 * beta_shape)


def svi_ssvi_vol_surface_proxy(features: Mapping[str, Any]) -> float:
    skew = _safe_float(features.get("options_skew_norm"), _safe_float(features.get("iv_skew_norm"), 0.35))
    smoothness = _safe_float(features.get("vol_surface_smoothness_norm"), 0.65)
    butterfly_risk = _safe_float(features.get("butterfly_arbitrage_risk_norm"), 0.15)
    calendar_risk = _safe_float(features.get("calendar_arbitrage_risk_norm"), 0.15)
    surface_change = _safe_float(features.get("options_surface_change_norm"), 0.25)
    return _clamp01(
        0.25 * skew
        + 0.25 * smoothness
        + 0.20 * (1.0 - butterfly_risk)
        + 0.20 * (1.0 - calendar_risk)
        + 0.10 * surface_change
    )


def dupire_local_vol_surface_proxy(features: Mapping[str, Any], path_volatility: float, svi_quality: float) -> float:
    local_curvature = _safe_float(features.get("local_vol_curvature_norm"), _safe_float(features.get("smile_curvature_norm"), 0.30))
    calendar_risk = _safe_float(features.get("calendar_arbitrage_risk_norm"), 0.15)
    realized_alignment = _safe_float(features.get("realized_implied_vol_alignment_norm"), 0.55)
    return _clamp01(0.30 * svi_quality + 0.25 * path_volatility + 0.20 * local_curvature + 0.15 * realized_alignment + 0.10 * (1.0 - calendar_risk))


def bates_jump_vol_proxy(heston_risk: float, merton_jump_risk: float, features: Mapping[str, Any]) -> float:
    vol_of_vol = _safe_float(features.get("options_vol_of_vol_change_norm"), _safe_float(features.get("vol_of_vol_stress_norm"), 0.35))
    jump_gap = _safe_float(features.get("market_micro_gap_fade_risk_norm"), _safe_float(features.get("gap_risk_norm"), 0.25))
    return _clamp01(0.35 * heston_risk + 0.35 * merton_jump_risk + 0.15 * vol_of_vol + 0.15 * jump_gap)


def hull_white_rates_proxy(features: Mapping[str, Any]) -> float:
    rate_vol = _safe_float(features.get("rate_volatility_norm"), _safe_float(features.get("rates_volatility_norm"), 0.25))
    curve_shift = _safe_float(features.get("yield_curve_slope_change_norm"), _safe_float(features.get("duration_stress_norm"), 0.25))
    mean_reversion = _safe_float(features.get("rates_mean_reversion_confidence_norm"), 0.55)
    convexity = _safe_float(features.get("convexity_stress_norm"), _safe_float(features.get("duration_convexity_norm"), 0.25))
    return _clamp01(0.35 * rate_vol + 0.30 * curve_shift + 0.20 * mean_reversion + 0.15 * convexity)


def cir_intensity_proxy(features: Mapping[str, Any]) -> float:
    credit_spread = _safe_float(features.get("credit_spread_stress_norm"), _safe_float(features.get("bbb_spread_stress_norm"), 0.30))
    default_probability = _safe_float(features.get("default_probability_stress_norm"), _safe_float(features.get("hazard_rate_stress_norm"), 0.25))
    funding = _safe_float(features.get("funding_stress_norm"), _safe_float(features.get("repo_funding_stress_norm"), 0.25))
    positivity_quality = 1.0 - _safe_float(features.get("negative_intensity_risk_norm"), 0.10)
    return _clamp01(0.35 * credit_spread + 0.30 * default_probability + 0.20 * funding + 0.15 * positivity_quality)


def hjm_forward_rate_proxy(features: Mapping[str, Any]) -> float:
    curve_shape = _safe_float(features.get("yield_curve_shape_change_norm"), _safe_float(features.get("yield_curve_slope_change_norm"), 0.25))
    forward_dispersion = _safe_float(features.get("forward_rate_dispersion_norm"), _safe_float(features.get("swap_curve_dispersion_norm"), 0.25))
    rate_vol = _safe_float(features.get("rate_volatility_norm"), 0.25)
    factor_coverage = _safe_float(features.get("curve_pca_coverage_norm"), _safe_float(features.get("factor_model_confidence_norm"), 0.55))
    return _clamp01(0.30 * curve_shape + 0.25 * forward_dispersion + 0.25 * rate_vol + 0.20 * factor_coverage)


def sofr_market_model_proxy(features: Mapping[str, Any]) -> float:
    sofr_shift = _safe_float(features.get("sofr_rate_change_norm"), _safe_float(features.get("front_end_rate_shock_norm"), 0.25))
    swaption_vol = _safe_float(features.get("swaption_vol_surface_norm"), _safe_float(features.get("rates_option_vol_norm"), 0.30))
    tenor_basis = _safe_float(features.get("tenor_basis_stress_norm"), _safe_float(features.get("basis_spread_stress_norm"), 0.20))
    rate_corr = _safe_float(features.get("rates_correlation_norm"), _safe_float(features.get("cross_sleeve_correlation_pressure_norm"), 0.35))
    return _clamp01(0.30 * sofr_shift + 0.25 * swaption_vol + 0.25 * tenor_basis + 0.20 * rate_corr)


def dcc_garch_correlation_proxy(returns: list[float], context_returns: list[float], features: Mapping[str, Any]) -> float:
    realized_cluster = min(sum(abs(x) for x in returns) / max(len(returns), 1) * 20.0, 1.0)
    context_cluster = min(sum(abs(x) for x in context_returns) / max(len(context_returns), 1) * 4.0, 1.0)
    corr_pressure = _safe_float(features.get("cross_sleeve_correlation_pressure_norm"), _safe_float(features.get("correlation_convergence_pressure_norm"), 0.35))
    regime_shift = _safe_float(features.get("regime_transition_pressure_norm"), _safe_float(features.get("model_drift_risk_norm"), 0.25))
    return _clamp01(0.30 * realized_cluster + 0.25 * context_cluster + 0.25 * corr_pressure + 0.20 * regime_shift)


def evt_pot_tail_proxy(returns: list[float], cvar_value: float, features: Mapping[str, Any]) -> float:
    losses = sorted([-float(x) for x in returns if float(x) < 0.0], reverse=True)
    max_loss = losses[0] if losses else 0.0
    threshold = losses[max(min(len(losses) // 4, len(losses) - 1), 0)] if losses else 0.0
    exceedance_ratio = sum(1 for loss in losses if loss >= threshold and threshold > 0.0) / max(len(returns), 1)
    tail_surface = _safe_float(features.get("tail_risk_surface_norm"), _safe_float(features.get("tail_risk_pressure_norm"), 0.30))
    return _clamp01(max(abs(cvar_value) * 16.0, max_loss * 25.0, exceedance_ratio * 2.0, tail_surface))


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


def differentiable_arbitrage_invariant_nn_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    layers = _env_int("QUANT_MODEL_DAINN_LAYERS", 3, low=1, high=24)
    gradient = _safe_float(features.get("simulator_gradient_stability_norm"), _safe_float(features.get("quant_mlx_jump_diffusion_grad_norm"), 0.45))
    no_arb = 1.0 - _safe_float(features.get("arbitrage_violation_rate_norm"), _safe_float(features.get("pricing_model_dispersion_norm"), 0.20))
    parity = 1.0 - _safe_float(features.get("put_call_parity_gap_norm"), _safe_float(features.get("basis_dislocation_norm"), 0.20))
    convexity = _safe_float(features.get("convexity_constraint_pass_norm"), _safe_float(features.get("pinn_constraint_consistency_norm"), 0.55))
    layer_quality = min(math.log2(max(layers, 1) + 1.0) / 5.0, 1.0)
    return _clamp01(0.25 * gradient + 0.25 * no_arb + 0.20 * parity + 0.20 * convexity + 0.10 * layer_quality)


def high_dimensional_markovian_execution_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    states = _env_int("QUANT_MODEL_MARKOV_EXEC_STATES", 5, low=2, high=64)
    micro = execution_microstructure_awareness_proxy(features)
    queue = 1.0 - _safe_float(features.get("queue_jitter_norm"), _safe_float(features.get("latency_jitter_norm"), 0.25))
    spread = 1.0 - _safe_float(features.get("bid_ask_spread_stress_norm"), 0.25)
    fill = _safe_float(features.get("fill_quality_norm"), _safe_float(features.get("paper_execution_calibration_norm"), 0.55))
    state_quality = min(math.log2(max(states, 2)) / 6.0, 1.0)
    return _clamp01(0.30 * micro + 0.20 * queue + 0.20 * spread + 0.20 * fill + 0.10 * state_quality)


def end_to_end_differentiable_backtest_proxy(values: list[float], features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    steps = _env_int("QUANT_MODEL_DIFF_BACKTEST_STEPS", 48, low=4, high=1024)
    simulator = differentiable_market_simulator_proxy(values, features)
    replay = _safe_float(features.get("golden_replay_pass_rate_norm"), 0.60)
    cost = 1.0 - _safe_float(features.get("slippage_pressure_norm"), _safe_float(features.get("hedge_cost_pressure_norm"), 0.25))
    gradient = _safe_float(features.get("backtest_gradient_stability_norm"), _safe_float(features.get("simulator_gradient_stability_norm"), 0.45))
    step_quality = min(math.log2(max(steps, 4)) / 10.0, 1.0)
    return _clamp01(0.25 * simulator + 0.25 * replay + 0.20 * cost + 0.20 * gradient + 0.10 * step_quality)


def portfolio_durability_resilient_alternatives_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    scenarios = _env_int("QUANT_MODEL_DURABILITY_SCENARIOS", 12, low=2, high=256)
    drawdown = 1.0 - _safe_float(features.get("drawdown_pressure_norm"), _safe_float(features.get("cvar_breach_norm"), 0.25))
    alternatives = _safe_float(features.get("resilient_alternatives_coverage_norm"), _safe_float(features.get("alternative_data_signal_norm"), 0.45))
    correlation = 1.0 - _safe_float(features.get("cross_sleeve_correlation_pressure_norm"), 0.35)
    tail_budget = 1.0 - _safe_float(features.get("tail_budget_pressure_norm"), _safe_float(features.get("tail_risk_pressure_norm"), 0.30))
    scenario_quality = min(math.log2(max(scenarios, 2)) / 8.0, 1.0)
    return _clamp01(0.25 * drawdown + 0.20 * alternatives + 0.20 * correlation + 0.25 * tail_budget + 0.10 * scenario_quality)


def information_geometry_manifold_proxy(values: list[float], features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    dims = _env_int("QUANT_MODEL_INFO_GEOMETRY_DIM", 8, low=2, high=256)
    if not values:
        values = [0.0]
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / max(len(values), 1)
    curvature = _clamp01(math.sqrt(max(variance, 0.0)) * 32.0)
    fisher = _safe_float(features.get("fisher_information_stability_norm"), 1.0 - _safe_float(features.get("model_drift_risk_norm"), 0.25))
    manifold = _safe_float(features.get("statistical_manifold_alignment_norm"), _safe_float(features.get("cross_asset_invariance_norm"), 0.50))
    dim_quality = min(math.log2(max(dims, 2)) / 8.0, 1.0)
    return _clamp01(0.25 * (1.0 - curvature) + 0.30 * fisher + 0.25 * manifold + 0.20 * dim_quality)


def graph_attention_spillover_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    heads = _env_int("QUANT_MODEL_GAT_HEADS", 4, low=1, high=32)
    graph = graph_neural_network_structure_proxy(features)
    spillover = _safe_float(features.get("cross_asset_spillover_signal_norm"), _safe_float(features.get("market_crypto_risk_corr_norm"), 0.45))
    attention = _safe_float(features.get("attention_stability_norm"), 1.0 - _safe_float(features.get("cross_sleeve_correlation_pressure_norm"), 0.35))
    contagion = _safe_float(features.get("spillover_contagion_pressure_norm"), _safe_float(features.get("tail_dependency_pressure_norm"), 0.30))
    head_quality = min(math.log2(max(heads, 1) + 1.0) / 5.0, 1.0)
    return _clamp01(0.25 * graph + 0.25 * spillover + 0.20 * attention + 0.15 * (1.0 - contagion) + 0.15 * head_quality)


def agentic_wallet_intent_execution_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    intents = _env_int("QUANT_MODEL_WALLET_INTENT_CAP", 8, low=1, high=128)
    provenance = _safe_float(features.get("decision_provenance_coverage_norm"), 0.50)
    auth = _safe_float(features.get("auth_lease_health_norm"), 0.60)
    intent_match = _safe_float(features.get("intent_execution_match_norm"), _safe_float(features.get("rule_consistency_norm"), 0.55))
    safety = _safe_float(features.get("formal_safety_pass_norm"), _safe_float(features.get("golden_replay_pass_rate_norm"), 0.60))
    cap_quality = min(math.log2(max(intents, 1) + 1.0) / 7.0, 1.0)
    return _clamp01(0.25 * provenance + 0.20 * auth + 0.25 * intent_match + 0.20 * safety + 0.10 * cap_quality)


def rough_path_signature_kernel_proxy(values: list[float], features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    depth = _env_int("QUANT_MODEL_SIGNATURE_KERNEL_DEPTH", 3, low=1, high=8)
    signature = signature_transform_path_dna(values)
    rough = rough_volatility_fbm_proxy(values)
    kernel = _safe_float(features.get("signature_kernel_alignment_norm"), _safe_float(features.get("path_similarity_norm"), 0.50))
    truncation = 1.0 - _safe_float(features.get("signature_truncation_error_norm"), 0.20)
    depth_quality = min(math.log2(max(depth, 1) + 1.0) / 4.0, 1.0)
    return _clamp01(0.25 * signature + 0.20 * rough + 0.25 * kernel + 0.20 * truncation + 0.10 * depth_quality)


def quantum_classical_hybrid_optimization_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    iterations = _env_int("QUANT_MODEL_HYBRID_OPT_ITERATIONS", 16, low=2, high=512)
    qemc = quantum_enhanced_monte_carlo_proxy(features)
    kelly = _safe_float(features.get("quant_kelly_fraction_norm"), 0.50)
    genetic = _safe_float(features.get("quant_genetic_optimization_stability_norm"), _safe_float(features.get("walk_forward_parameter_stability_norm"), 0.50))
    allocation = _safe_float(features.get("allocation_stability_context_norm"), _safe_float(features.get("allocation_stability_norm"), 0.50))
    iter_quality = min(math.log2(max(iterations, 2)) / 9.0, 1.0)
    return _clamp01(0.25 * qemc + 0.20 * kelly + 0.25 * genetic + 0.20 * allocation + 0.10 * iter_quality)


def formal_verification_smart_agent_safety_proxy(features: Mapping[str, Any] | None = None) -> float:
    features = features or {}
    checks = _env_int("QUANT_MODEL_FORMAL_CHECKS", 12, low=1, high=256)
    invariant = _safe_float(features.get("formal_invariant_pass_norm"), _safe_float(features.get("rule_consistency_norm"), 0.60))
    replay = _safe_float(features.get("golden_replay_pass_rate_norm"), 0.60)
    policy = 1.0 - _safe_float(features.get("execution_policy_violation_norm"), _safe_float(features.get("rotation_violation_norm"), 0.05))
    provenance = _safe_float(features.get("decision_provenance_coverage_norm"), 0.50)
    check_quality = min(math.log2(max(checks, 1) + 1.0) / 8.0, 1.0)
    return _clamp01(0.30 * invariant + 0.20 * replay + 0.25 * policy + 0.15 * provenance + 0.10 * check_quality)


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
    dainn_layers = _env_int("QUANT_MODEL_DAINN_LAYERS", 3, low=1, high=24)
    markov_exec_states = _env_int("QUANT_MODEL_MARKOV_EXEC_STATES", 5, low=2, high=64)
    diff_backtest_steps = _env_int("QUANT_MODEL_DIFF_BACKTEST_STEPS", 48, low=4, high=1024)
    durability_scenarios = _env_int("QUANT_MODEL_DURABILITY_SCENARIOS", 12, low=2, high=256)
    info_geometry_dim = _env_int("QUANT_MODEL_INFO_GEOMETRY_DIM", 8, low=2, high=256)
    gat_heads = _env_int("QUANT_MODEL_GAT_HEADS", 4, low=1, high=32)
    wallet_intent_cap = _env_int("QUANT_MODEL_WALLET_INTENT_CAP", 8, low=1, high=128)
    signature_kernel_depth = _env_int("QUANT_MODEL_SIGNATURE_KERNEL_DEPTH", 3, low=1, high=8)
    hybrid_opt_iterations = _env_int("QUANT_MODEL_HYBRID_OPT_ITERATIONS", 16, low=2, high=512)
    formal_checks = _env_int("QUANT_MODEL_FORMAL_CHECKS", 12, low=1, high=256)
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
        + 0.01 * dainn_layers / 12.0
        + 0.01 * markov_exec_states / 32.0
        + 0.02 * diff_backtest_steps / 512.0
        + 0.01 * durability_scenarios / 128.0
        + 0.01 * info_geometry_dim / 128.0
        + 0.01 * gat_heads / 16.0
        + 0.01 * wallet_intent_cap / 64.0
        + 0.01 * signature_kernel_depth / 8.0
        + 0.01 * hybrid_opt_iterations / 256.0
        + 0.01 * formal_checks / 128.0
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
        "dainn_layers": float(dainn_layers),
        "markov_exec_states": float(markov_exec_states),
        "diff_backtest_steps": float(diff_backtest_steps),
        "durability_scenarios": float(durability_scenarios),
        "info_geometry_dim": float(info_geometry_dim),
        "gat_heads": float(gat_heads),
        "wallet_intent_cap": float(wallet_intent_cap),
        "signature_kernel_depth": float(signature_kernel_depth),
        "hybrid_opt_iterations": float(hybrid_opt_iterations),
        "formal_checks": float(formal_checks),
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
    quantlib_price = quantlib_black_scholes_price(spot, strike, time_years, rate, vol)
    quantlib_benchmark = (
        _clamp01(1.0 - abs(quantlib_price - bs_price) / max(bs_price * 0.03, price_scale * 0.003, 1e-9))
        if quantlib_price is not None
        else 0.0
    )
    merton_jump_risk = _clamp01(abs(merton - bs_price) / price_scale * 4.0)
    plain_error = abs(mc - bs_price)
    variance_reduced_error = min(abs(qmc - bs_price), abs(lhs - bs_price))
    antithetic_efficiency = _clamp01(1.0 - variance_reduced_error / max(plain_error + 1e-9, bs_price * 0.03, 1e-9))
    sabr = sabr_vol_surface_proxy(
        spot * math.exp(rate * time_years),
        strike,
        time_years,
        vol,
        beta=_safe_float(features.get("sabr_beta"), 0.5),
        rho=_safe_float(features.get("sabr_rho"), _safe_float(features.get("skew_rho_proxy"), -0.35)),
        vol_of_vol=_safe_float(features.get("sabr_vol_of_vol"), _safe_float(features.get("options_vol_of_vol_change_norm"), 0.45)),
    )
    svi_ssvi = svi_ssvi_vol_surface_proxy(features)
    dupire = dupire_local_vol_surface_proxy(features, functional_ito_path_volatility(returns), svi_ssvi)
    bates = bates_jump_vol_proxy(heston, merton_jump_risk, features)
    hull_white = hull_white_rates_proxy(features)
    cir_intensity = cir_intensity_proxy(features)
    hjm = hjm_forward_rate_proxy(features)
    sofr_lmm = sofr_market_model_proxy(features)
    kalman = kalman_filter_level(returns)
    gpu_kalman = real_time_kalman_filter_gpu(returns)
    particle = particle_filter_level(returns)
    kelly = kelly_fraction(
        0.5 + 0.25 * _clamp(_safe_float(features.get("flow_direction_signed"), 0.0), -1.0, 1.0),
        1.0 + abs(_safe_float(features.get("edge_norm"), _safe_float(features.get("execution_fitness_norm"), 0.5))),
    )
    cvar = conditional_value_at_risk(returns, alpha=0.95)
    copula = gaussian_copula_dependency_proxy(returns, context_returns)
    dcc_garch = dcc_garch_correlation_proxy(returns, context_returns, features)
    evt_pot = evt_pot_tail_proxy(returns, cvar, features)
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
    dainn = differentiable_arbitrage_invariant_nn_proxy(
        {
            **dict(features),
            "simulator_gradient_stability_norm": _safe_float(features.get("simulator_gradient_stability_norm"), dms),
            "pinn_constraint_consistency_norm": pinn,
        }
    )
    markov_exec = high_dimensional_markovian_execution_proxy(features)
    diff_backtest = end_to_end_differentiable_backtest_proxy(returns, {**dict(features), "simulator_gradient_stability_norm": dms})
    durability = portfolio_durability_resilient_alternatives_proxy(
        {
            **dict(features),
            "tail_risk_pressure_norm": max(_clamp01(abs(cvar) * 16.0), copula),
            "alternative_data_signal_norm": alt_data,
        }
    )
    info_geometry = information_geometry_manifold_proxy(returns, features)
    gat_spillover = graph_attention_spillover_proxy(
        {
            **dict(features),
            "market_crypto_risk_corr_norm": _safe_float(features.get("market_crypto_risk_corr_norm"), copula),
        }
    )
    wallet_intent = agentic_wallet_intent_execution_proxy(features)
    rough_path_kernel = rough_path_signature_kernel_proxy(returns, features)
    hybrid_opt = quantum_classical_hybrid_optimization_proxy(
        {
            **dict(features),
            "quant_quasi_monte_carlo_price_norm": _clamp01(qmc / price_scale),
            "quant_antithetic_variates_efficiency_norm": antithetic_efficiency,
            "quant_kelly_fraction_norm": _clamp01((kelly + 1.0) / 2.0),
            "quant_genetic_optimization_stability_norm": _clamp01(ga.get("stability", 0.0)),
        }
    )
    formal_safety = formal_verification_smart_agent_safety_proxy(features)
    lobdif = _clamp01(0.45 * lit + 0.35 * geometric_lit + 0.20 * microstructure)
    fractional_hurst = _clamp01(0.55 * hurst + 0.45 * rough_vol)
    market_impact = _clamp01(
        0.45 * dms
        + 0.35 * markov_exec
        + 0.20 * (1.0 - _safe_float(features.get("slippage_pressure_norm"), _safe_float(features.get("hedge_cost_pressure_norm"), 0.25)))
    )
    persistent_homology = _clamp01(0.60 * tda + 0.40 * laplacian)
    toxic_liquidity = _clamp01(0.70 * vpin + 0.30 * _safe_float(features.get("stress_injection_replay_norm"), 0.45))
    flash_freeze = _clamp01(
        0.35 * _safe_float(features.get("quote_fade_risk_norm"), 0.25)
        + 0.35 * _safe_float(features.get("bid_ask_spread_stress_norm"), _safe_float(features.get("spread_stress_norm"), 0.25))
        + 0.30 * toxic_liquidity
    )
    photonic_quantum = _clamp01(0.60 * hybrid_opt + 0.40 * qemc)
    replication_shield = _clamp01(
        0.45 * _safe_float(features.get("golden_replay_pass_rate_norm"), 0.60)
        + 0.35 * _safe_float(features.get("walk_forward_parameter_stability_norm"), _clamp01(ga.get("stability", 0.0)))
        + 0.20 * (1.0 - _safe_float(features.get("model_drift_risk_norm"), 0.25))
    )
    synthetic_crisis = _clamp01(0.45 * signature_gen + 0.30 * hawkes + 0.25 * max(_clamp01(abs(cvar) * 16.0), copula))
    correlation_convergence = _clamp01(
        max(
            copula,
            _safe_float(features.get("cross_sleeve_correlation_pressure_norm"), 0.35),
            _safe_float(features.get("correlation_convergence_pressure_norm"), 0.35),
        )
    )
    macro_stress_2026 = _clamp01(
        0.30 * _safe_float(features.get("vix_stress_norm"), _safe_float(features.get("options_iv_percentile_norm"), 0.25))
        + 0.25 * _safe_float(features.get("credit_spread_stress_norm"), _safe_float(features.get("bbb_spread_stress_norm"), 0.30))
        + 0.20 * _safe_float(features.get("unemployment_stress_norm"), 0.35)
        + 0.15 * correlation_convergence
        + 0.10 * _safe_float(features.get("real_estate_stress_norm"), 0.30)
    )
    fed_2026_integrity = _clamp01(_safe_float(features.get("fed_2026_scenario_integrity_norm"), 0.85))
    fed_2026_equity_crash_vol = _clamp01(
        max(
            _safe_float(features.get("fed_2026_equity_crash_vol_spike_norm"), 0.0),
            0.45 * _safe_float(features.get("vix_stress_norm"), _safe_float(features.get("options_iv_percentile_norm"), 0.25))
            + 0.35 * _safe_float(features.get("equity_drawdown_norm"), _safe_float(features.get("market_drawdown_norm"), 0.30))
            + 0.20 * correlation_convergence,
        )
    )
    fed_2026_credit_blowout = _clamp01(
        max(
            _safe_float(features.get("fed_2026_credit_spread_blowout_norm"), 0.0),
            0.50 * _safe_float(features.get("credit_spread_stress_norm"), _safe_float(features.get("bbb_spread_stress_norm"), 0.30))
            + 0.25 * _safe_float(features.get("default_probability_stress_norm"), _safe_float(features.get("hazard_rate_stress_norm"), 0.25))
            + 0.25 * _safe_float(features.get("funding_stress_norm"), 0.25),
        )
    )
    fed_2026_housing_shock = _clamp01(
        max(
            _safe_float(features.get("fed_2026_housing_price_shock_norm"), 0.0),
            0.40 * _safe_float(features.get("house_price_stress_norm"), _safe_float(features.get("housing_price_stress_norm"), 0.30))
            + 0.30 * _safe_float(features.get("mortgage_rate_stress_norm"), _safe_float(features.get("rate_volatility_context_norm"), 0.25))
            + 0.30 * _safe_float(features.get("unemployment_stress_norm"), 0.35),
        )
    )
    fed_2026_cre_shock = _clamp01(
        max(
            _safe_float(features.get("fed_2026_cre_price_shock_norm"), 0.0),
            0.45 * _safe_float(features.get("cre_price_stress_norm"), _safe_float(features.get("commercial_real_estate_stress_norm"), 0.35))
            + 0.30 * _safe_float(features.get("regional_bank_stress_norm"), _safe_float(features.get("bank_stress_norm"), 0.25))
            + 0.25 * fed_2026_credit_blowout,
        )
    )
    fed_2026_unemployment_recession = _clamp01(
        max(
            _safe_float(features.get("fed_2026_unemployment_recession_norm"), 0.0),
            0.50 * _safe_float(features.get("unemployment_stress_norm"), 0.35)
            + 0.30 * _safe_float(features.get("real_gdp_contraction_norm"), _safe_float(features.get("macro_recession_norm"), 0.30))
            + 0.20 * _safe_float(features.get("income_stress_norm"), 0.25),
        )
    )
    fed_2026_global_deflation = _clamp01(
        max(
            _safe_float(features.get("fed_2026_global_recession_deflation_norm"), 0.0),
            0.35 * _safe_float(features.get("global_growth_stress_norm"), _safe_float(features.get("euro_area_growth_stress_norm"), 0.35))
            + 0.25 * _safe_float(features.get("deflation_stress_norm"), _safe_float(features.get("inflation_downside_stress_norm"), 0.25))
            + 0.20 * _safe_float(features.get("fx_pressure_norm"), _safe_float(features.get("usd_stress_norm"), 0.30))
            + 0.20 * correlation_convergence,
        )
    )
    fed_2026_commodity_inflation = _clamp01(
        max(
            _safe_float(features.get("fed_2026_commodity_inflation_shock_norm"), 0.0),
            0.40 * _safe_float(features.get("commodity_shock_norm"), _safe_float(features.get("oil_price_shock_norm"), 0.25))
            + 0.30 * _safe_float(features.get("inflation_expectation_stress_norm"), _safe_float(features.get("inflation_shock_norm"), 0.25))
            + 0.30 * _safe_float(features.get("energy_futures_stress_norm"), _safe_float(features.get("futures_curve_stress_norm"), 0.25)),
        )
    )
    fed_2026_treasury_yield_shock = _clamp01(
        max(
            _safe_float(features.get("fed_2026_treasury_yield_shock_norm"), 0.0),
            0.40 * _safe_float(features.get("front_end_rate_shock_norm"), _safe_float(features.get("rate_shock_norm"), 0.25))
            + 0.35 * _safe_float(features.get("long_rate_shock_norm"), _safe_float(features.get("duration_stress_norm"), 0.25))
            + 0.25 * _safe_float(features.get("inflation_expectation_stress_norm"), _safe_float(features.get("inflation_shock_norm"), 0.25)),
        )
    )
    fed_2026_usd_stress = _clamp01(
        max(
            _safe_float(features.get("fed_2026_usd_stress_norm"), 0.0),
            0.40 * _safe_float(features.get("usd_stress_norm"), _safe_float(features.get("uup_momentum_norm"), 0.25))
            + 0.30 * _safe_float(features.get("fx_pressure_norm"), _safe_float(features.get("em_fx_pressure_norm"), 0.25))
            + 0.30 * _safe_float(features.get("safe_haven_flow_norm"), 0.25),
        )
    )
    fed_2026_counterparty_contagion = _clamp01(
        max(
            _safe_float(features.get("fed_2026_counterparty_default_contagion_norm"), 0.0),
            0.35 * fed_2026_credit_blowout
            + 0.25 * _safe_float(features.get("counterparty_default_stress_norm"), _safe_float(features.get("xva_exposure_stress_norm"), 0.25))
            + 0.20 * _safe_float(features.get("repo_funding_stress_norm"), _safe_float(features.get("funding_stress_norm"), 0.25))
            + 0.20 * fed_2026_equity_crash_vol,
        )
    )
    covid_2020_pandemic = _clamp01(
        max(
            _safe_float(features.get("covid_2020_pandemic_replay_norm"), 0.0),
            0.20 * _safe_float(features.get("vix_stress_norm"), _safe_float(features.get("options_iv_percentile_norm"), 0.30))
            + 0.18 * _safe_float(features.get("unemployment_stress_norm"), 0.45)
            + 0.16 * _safe_float(features.get("liquidity_facility_stress_norm"), _safe_float(features.get("funding_stress_norm"), 0.35))
            + 0.14 * _safe_float(features.get("pandemic_shutdown_stress_norm"), _safe_float(features.get("event_burst_rate_norm"), 0.35))
            + 0.12 * _safe_float(features.get("oil_price_shock_norm"), _safe_float(features.get("commodity_shock_norm"), 0.25))
            + 0.10 * correlation_convergence
            + 0.10 * _safe_float(features.get("policy_rate_zero_bound_norm"), _safe_float(features.get("rate_cut_shock_norm"), 0.30)),
        )
    )
    mckean_vlasov = _clamp01(
        max(
            _safe_float(features.get("mckean_vlasov_control_norm"), 0.0),
            0.35 * mean_field
            + 0.25 * sdg
            + 0.20 * rlbf
            + 0.20 * (1.0 - correlation_convergence),
        )
    )
    tensor_mps = _clamp01(
        max(
            _safe_float(features.get("tensor_network_mps_norm"), 0.0),
            0.30 * graph_structure
            + 0.25 * signature
            + 0.20 * info_geometry
            + 0.15 * hybrid_opt
            + 0.10 * _safe_float(features.get("source_confidence_norm"), 0.55),
        )
    )
    multifidelity = _clamp01(
        max(
            _safe_float(features.get("multifidelity_stochastic_programming_norm"), 0.0),
            0.25 * _clamp01(qmc / price_scale)
            + 0.20 * _clamp01(lhs / price_scale)
            + 0.20 * antithetic_efficiency
            + 0.20 * durability
            + 0.15 * replication_shield,
        )
    )
    tatonnement = _clamp01(
        max(
            _safe_float(features.get("differentiable_tatonnement_norm"), 0.0),
            0.30 * market_impact
            + 0.25 * markov_exec
            + 0.20 * (1.0 - _safe_float(features.get("spread_stress_norm"), _safe_float(features.get("bid_ask_spread_stress_norm"), 0.25)))
            + 0.15 * wallet_intent
            + 0.10 * formal_safety,
        )
    )
    lead_lag_detector = _clamp01(
        max(
            _safe_float(features.get("signature_lead_lag_detector_norm"), 0.0),
            0.35 * signature
            + 0.25 * hawkes
            + 0.20 * _safe_float(features.get("cross_sleeve_correlation_pressure_norm"), correlation_convergence)
            + 0.20 * microstructure,
        )
    )
    chaos_propagation = _clamp01(
        max(
            _safe_float(features.get("chaos_propagation_norm"), 0.0),
            0.30 * mean_field
            + 0.25 * correlation_convergence
            + 0.20 * hawkes
            + 0.15 * tda
            + 0.10 * covid_2020_pandemic,
        )
    )
    mckean_vlasov_sensitivity = _clamp01(
        max(
            _safe_float(features.get("mckean_vlasov_sde_sensitivity_norm"), 0.0),
            0.35 * mckean_vlasov
            + 0.25 * rough_vol
            + 0.20 * pin_sde
            + 0.20 * sdg,
        )
    )
    mlmc_sequential = _clamp01(
        max(
            _safe_float(features.get("mlmc_sequential_estimation_norm"), 0.0),
            0.30 * _clamp01(mc / price_scale)
            + 0.25 * _clamp01(qmc / price_scale)
            + 0.20 * antithetic_efficiency
            + 0.15 * _clamp01(kalman.get("confidence", 0.0))
            + 0.10 * (1.0 - _clamp01(resource["resource_pressure_norm"])),
        )
    )
    volterra_kernel = _clamp01(
        max(
            _safe_float(features.get("signature_volterra_kernel_calibration_norm"), 0.0),
            0.35 * signature
            + 0.25 * rough_path_kernel
            + 0.20 * path_vol
            + 0.20 * fractional_hurst,
        )
    )
    dual_tatonnement = _clamp01(
        max(
            _safe_float(features.get("dual_tatonnement_price_discovery_norm"), 0.0),
            0.40 * tatonnement
            + 0.25 * market_impact
            + 0.20 * wallet_intent
            + 0.15 * dainn,
        )
    )
    probabilistic_chaos = _clamp01(
        max(
            _safe_float(features.get("probabilistic_propagation_of_chaos_norm"), 0.0),
            0.45 * chaos_propagation
            + 0.25 * mckean_vlasov
            + 0.15 * hawkes
            + 0.15 * correlation_convergence,
        )
    )
    rough_vvix = _clamp01(
        max(
            _safe_float(features.get("rough_vvix_exotics_norm"), _safe_float(features.get("vix_on_vix_exotics_norm"), 0.0)),
            0.35 * fractional_hurst
            + 0.25 * _safe_float(features.get("vvix_stress_norm"), _safe_float(features.get("vol_of_vol_stress_norm"), 0.35))
            + 0.20 * _safe_float(features.get("vix_stress_norm"), _safe_float(features.get("options_iv_percentile_norm"), 0.30))
            + 0.20 * heston,
        )
    )
    quantum_barrier = _clamp01(
        max(
            _safe_float(features.get("quantum_barrier_path_amplitude_norm"), 0.0),
            0.35 * qemc
            + 0.25 * _safe_float(features.get("barrier_touch_risk_norm"), _safe_float(features.get("options_barrier_touch_risk_norm"), 0.30))
            + 0.20 * path_vol
            + 0.20 * hybrid_opt,
        )
    )
    correlation_heat_swap = _clamp01(
        max(
            _safe_float(features.get("cross_asset_correlation_heat_swap_norm"), 0.0),
            0.45 * correlation_convergence
            + 0.25 * copula
            + 0.20 * gat_spillover
            + 0.10 * macro_stress_2026,
        )
    )
    cliquet_floor_cap = _clamp01(
        max(
            _safe_float(features.get("cliquet_global_floor_local_cap_norm"), 0.0),
            0.30 * path_vol
            + 0.25 * _safe_float(features.get("structured_payoff_stress_norm"), _safe_float(features.get("coupon_barrier_stress_norm"), 0.35))
            + 0.20 * _safe_float(features.get("realized_vol_stress_norm"), _safe_float(features.get("options_iv_percentile_norm"), 0.30))
            + 0.15 * _safe_float(features.get("floor_cap_distance_norm"), 0.45)
            + 0.10 * durability,
        )
    )
    signature_trend_option = _clamp01(
        max(
            _safe_float(features.get("signature_trend_follower_options_norm"), 0.0),
            0.30 * lead_lag_detector
            + 0.25 * signature
            + 0.20 * rough_path_kernel
            + 0.15 * _safe_float(features.get("trend_persistence_norm"), _safe_float(features.get("lead_lag_confidence_norm"), 0.40))
            + 0.10 * _safe_float(features.get("options_surface_change_norm"), _safe_float(features.get("options_iv_percentile_norm"), 0.30)),
        )
    )
    esg_ccds = _clamp01(
        max(
            _safe_float(features.get("esg_contingent_cds_norm"), _safe_float(features.get("ccds_stress_norm"), 0.0)),
            0.35 * _safe_float(features.get("credit_spread_stress_norm"), _safe_float(features.get("bbb_spread_stress_norm"), 0.30))
            + 0.25 * _safe_float(features.get("esg_controversy_stress_norm"), _safe_float(features.get("issuer_esg_event_norm"), 0.25))
            + 0.20 * _safe_float(features.get("hazard_rate_stress_norm"), _safe_float(features.get("default_probability_stress_norm"), 0.30))
            + 0.20 * macro_stress_2026,
        )
    )
    sdg_control = _clamp01(
        max(
            _safe_float(features.get("sdg_control_norm"), 0.0),
            0.45 * sdg
            + 0.25 * formal_safety
            + 0.15 * mckean_vlasov
            + 0.15 * market_impact,
        )
    )
    nonlocal_fractional_laplacian = _clamp01(
        max(
            _safe_float(features.get("nonlocal_fractional_laplacian_norm"), 0.0),
            0.35 * laplacian
            + 0.25 * fractional_hurst
            + 0.20 * tda
            + 0.20 * rough_path_kernel,
        )
    )
    infinite_heston = _clamp01(
        max(
            _safe_float(features.get("infinite_dimensional_heston_norm"), 0.0),
            0.40 * heston
            + 0.25 * rough_vol
            + 0.20 * _safe_float(features.get("options_vol_of_vol_change_norm"), _safe_float(features.get("vol_of_vol_stress_norm"), 0.35))
            + 0.15 * pin_sde,
        )
    )
    lie_group_signature = _clamp01(
        max(
            _safe_float(features.get("lie_group_rough_path_signature_norm"), 0.0),
            0.40 * signature
            + 0.25 * rough_path_kernel
            + 0.20 * graph_structure
            + 0.15 * info_geometry,
        )
    )
    mfg_controls = _clamp01(
        max(
            _safe_float(features.get("mean_field_games_controls_norm"), _safe_float(features.get("mfg_controls_norm"), 0.0)),
            0.40 * mean_field
            + 0.25 * mckean_vlasov
            + 0.20 * sdg
            + 0.15 * rlbf,
        )
    )
    wasserstein_flow = _clamp01(
        max(
            _safe_float(features.get("wasserstein_gradient_flow_norm"), 0.0),
            0.35 * ot_bridge
            + 0.25 * info_geometry
            + 0.20 * tda
            + 0.20 * hybrid_opt,
        )
    )
    malliavin_greeks = _clamp01(
        max(
            _safe_float(features.get("malliavin_wiener_greeks_norm"), 0.0),
            0.30 * infinite_heston
            + 0.25 * _clamp01((abs(mlx_grad.get("delta", 0.0)) + min(abs(mlx_grad.get("gamma", 0.0)), 1.0)) / 2.0)
            + 0.20 * heston
            + 0.15 * pin_sde
            + 0.10 * path_vol,
        )
    )
    tqft_braid = _clamp01(
        max(
            _safe_float(features.get("tqft_braid_group_norm"), 0.0),
            0.35 * tda
            + 0.25 * lie_group_signature
            + 0.20 * graph_structure
            + 0.20 * info_geometry,
        )
    )
    mfgc_congestion = _clamp01(
        max(
            _safe_float(features.get("mfgc_congestion_norm"), 0.0),
            0.40 * mfg_controls
            + 0.25 * mean_field
            + 0.20 * market_impact
            + 0.15 * toxic_liquidity,
        )
    )
    spde_lob_fluid = _clamp01(
        max(
            _safe_float(features.get("spde_manifold_lob_fluid_norm"), 0.0),
            0.30 * geometric_lit
            + 0.25 * lit
            + 0.20 * laplacian
            + 0.15 * pin_sde
            + 0.10 * microstructure,
        )
    )
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
    experience_memory = _clamp01(
        max(
            _safe_float(features.get("experience_accumulation_memory_norm"), 0.0),
            0.25 * critic
            + 0.20 * observer_critic
            + 0.20 * rlbf
            + 0.20 * _safe_float(features.get("replay_memory_hit_rate_norm"), _safe_float(features.get("experience_replay_coverage_norm"), 0.50))
            + 0.15 * data_conf,
        )
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
            "quant_merton_jump_risk_norm": merton_jump_risk,
            "quant_sabr_vol_surface_norm": sabr,
            "quant_svi_ssvi_vol_surface_norm": svi_ssvi,
            "quant_dupire_local_vol_surface_norm": dupire,
            "quant_bates_jump_vol_norm": bates,
            "quant_hull_white_rates_norm": hull_white,
            "quant_cir_intensity_norm": cir_intensity,
            "quant_hjm_forward_rate_norm": hjm,
            "quant_sofr_market_model_norm": sofr_lmm,
            "quant_dcc_garch_correlation_norm": dcc_garch,
            "quant_evt_pot_tail_norm": evt_pot,
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
            "quant_mlx_nn_available_norm": 1.0 if _module_available("mlx.nn") else 0.0,
            "quant_mlx_optimizers_available_norm": 1.0 if _module_available("mlx.optimizers") else 0.0,
            "quant_mlx_lm_available_norm": 1.0 if _module_available("mlx_lm") else 0.0,
            "quant_mlx_graphs_available_norm": 1.0 if _module_available("mlx_graphs") else 0.0,
            "quant_mlx_snn_available_norm": 1.0 if _module_available("mlxsnn") else 0.0,
            "quant_mlx_vision_available_norm": 1.0 if _module_available("mlx_vision") else 0.0,
            "quant_esig_signature_available_norm": 1.0 if (_module_available("esig") or _module_available("roughpy")) else 0.0,
            "quant_quantlib_available_norm": 1.0 if ql is not None else 0.0,
            "quant_quantlib_pricing_benchmark_norm": quantlib_benchmark,
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
            "quant_dainn_arbitrage_invariant_norm": dainn,
            "quant_markovian_execution_control_norm": markov_exec,
            "quant_end_to_end_diff_backtest_norm": diff_backtest,
            "quant_portfolio_durability_norm": durability,
            "quant_information_geometry_manifold_norm": info_geometry,
            "quant_graph_attention_spillover_norm": gat_spillover,
            "quant_agentic_wallet_intent_norm": wallet_intent,
            "quant_rough_path_signature_kernel_norm": rough_path_kernel,
            "quant_quantum_classical_hybrid_optimization_norm": hybrid_opt,
            "quant_formal_verification_safety_norm": formal_safety,
            "quant_lobdif_order_book_diffusion_norm": lobdif,
            "quant_fractional_hurst_rough_vol_norm": fractional_hurst,
            "quant_differentiable_market_impact_norm": market_impact,
            "quant_persistent_homology_flash_crash_norm": persistent_homology,
            "quant_toxic_liquidity_injection_norm": toxic_liquidity,
            "quant_flash_freeze_slippage_norm": flash_freeze,
            "quant_photonic_quantum_optimization_norm": photonic_quantum,
            "quant_replication_crisis_shield_norm": replication_shield,
            "quant_synthetic_crisis_market_gan_norm": synthetic_crisis,
            "quant_correlation_convergence_norm": correlation_convergence,
            "quant_macro_stress_2026_driver_norm": macro_stress_2026,
            "quant_fed_2026_scenario_integrity_norm": fed_2026_integrity,
            "quant_fed_2026_equity_crash_vol_spike_norm": fed_2026_equity_crash_vol,
            "quant_fed_2026_credit_spread_blowout_norm": fed_2026_credit_blowout,
            "quant_fed_2026_housing_price_shock_norm": fed_2026_housing_shock,
            "quant_fed_2026_cre_price_shock_norm": fed_2026_cre_shock,
            "quant_fed_2026_unemployment_recession_norm": fed_2026_unemployment_recession,
            "quant_fed_2026_global_recession_deflation_norm": fed_2026_global_deflation,
            "quant_fed_2026_commodity_inflation_shock_norm": fed_2026_commodity_inflation,
            "quant_fed_2026_treasury_yield_shock_norm": fed_2026_treasury_yield_shock,
            "quant_fed_2026_usd_stress_norm": fed_2026_usd_stress,
            "quant_fed_2026_counterparty_default_contagion_norm": fed_2026_counterparty_contagion,
            "quant_covid_2020_pandemic_replay_norm": covid_2020_pandemic,
            "quant_mckean_vlasov_control_norm": mckean_vlasov,
            "quant_tensor_network_mps_norm": tensor_mps,
            "quant_multifidelity_stochastic_programming_norm": multifidelity,
            "quant_differentiable_tatonnement_norm": tatonnement,
            "quant_signature_lead_lag_detector_norm": lead_lag_detector,
            "quant_chaos_propagation_norm": chaos_propagation,
            "quant_mckean_vlasov_sde_sensitivity_norm": mckean_vlasov_sensitivity,
            "quant_mlmc_sequential_estimation_norm": mlmc_sequential,
            "quant_signature_volterra_kernel_calibration_norm": volterra_kernel,
            "quant_dual_tatonnement_price_discovery_norm": dual_tatonnement,
            "quant_probabilistic_propagation_of_chaos_norm": probabilistic_chaos,
            "quant_experience_accumulation_memory_norm": experience_memory,
            "quant_rough_vvix_exotics_norm": rough_vvix,
            "quant_quantum_barrier_path_amplitude_norm": quantum_barrier,
            "quant_cross_asset_correlation_heat_swap_norm": correlation_heat_swap,
            "quant_cliquet_global_floor_local_cap_norm": cliquet_floor_cap,
            "quant_signature_trend_follower_options_norm": signature_trend_option,
            "quant_esg_contingent_cds_norm": esg_ccds,
            "quant_sdg_control_norm": sdg_control,
            "quant_nonlocal_fractional_laplacian_norm": nonlocal_fractional_laplacian,
            "quant_infinite_dimensional_heston_norm": infinite_heston,
            "quant_lie_group_rough_path_signature_norm": lie_group_signature,
            "quant_mean_field_games_controls_norm": mfg_controls,
            "quant_wasserstein_gradient_flow_norm": wasserstein_flow,
            "quant_malliavin_wiener_greeks_norm": malliavin_greeks,
            "quant_tqft_braid_group_norm": tqft_braid,
            "quant_mfgc_congestion_norm": mfgc_congestion,
            "quant_spde_manifold_lob_fluid_norm": spde_lob_fluid,
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
            "sabr_stochastic_alpha_beta_rho_vol_surface_proxy",
            "svi_ssvi_arbitrage_free_vol_surface_proxy",
            "dupire_local_volatility_surface_proxy",
            "bates_heston_jump_diffusion_proxy",
            "hull_white_one_factor_rates_proxy",
            "cir_short_rate_credit_intensity_proxy",
            "hjm_forward_rate_model_proxy",
            "sofr_libor_market_model_proxy",
            "dynamic_conditional_correlation_garch_proxy",
            "extreme_value_theory_peaks_over_threshold_proxy",
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
            "quantlib_black_scholes_benchmark_proxy",
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
            "differentiable_arbitrage_invariant_neural_network_proxy",
            "high_dimensional_markovian_order_execution_proxy",
            "end_to_end_differentiable_backtesting_proxy",
            "portfolio_durability_resilient_alternatives_proxy",
            "information_geometry_statistical_manifold_proxy",
            "graph_attention_network_cross_asset_spillover_proxy",
            "agentic_wallet_intent_based_execution_proxy",
            "rough_path_signature_kernel_proxy",
            "quantum_classical_hybrid_optimization_proxy",
            "formal_verification_smart_agent_safety_proxy",
            "lobdif_order_book_diffusion_proxy",
            "fractional_hurst_rough_volatility_proxy",
            "differentiable_market_impact_proxy",
            "persistent_homology_flash_crash_proxy",
            "toxic_liquidity_vpin_stress_injector_proxy",
            "flash_freeze_slippage_model_proxy",
            "photonic_quantum_optimization_proxy",
            "replication_crisis_shield_proxy",
            "synthetic_crisis_market_gan_proxy",
            "correlation_convergence_simulation_proxy",
            "macro_stress_2026_driver_proxy",
            "fed_2026_supervisory_scenario_dataset_proxy",
            "fed_2026_equity_crash_volatility_spike_proxy",
            "fed_2026_corporate_credit_spread_blowout_proxy",
            "fed_2026_housing_price_shock_proxy",
            "fed_2026_commercial_real_estate_shock_proxy",
            "fed_2026_unemployment_recession_shock_proxy",
            "fed_2026_global_recession_deflation_shock_proxy",
            "fed_2026_commodity_inflation_shock_proxy",
            "fed_2026_treasury_yield_shock_proxy",
            "fed_2026_us_dollar_stress_proxy",
            "fed_2026_counterparty_default_contagion_proxy",
            "covid_2020_pandemic_crash_replay_proxy",
            "mckean_vlasov_master_equation_control_proxy",
            "quantum_tensor_network_matrix_product_state_proxy",
            "multifidelity_stochastic_programming_proxy",
            "differentiable_tatonnement_price_discovery_proxy",
            "signature_lead_lag_detector_proxy",
            "probabilistic_chaos_propagation_proxy",
            "mckean_vlasov_sde_sensitivity_proxy",
            "multi_level_monte_carlo_sequential_estimation_proxy",
            "signature_volterra_kernel_calibration_proxy",
            "dual_tatonnement_price_discovery_proxy",
            "probabilistic_propagation_of_chaos_proxy",
            "experience_accumulation_memory_design_proxy",
            "rough_volatility_vvix_exotics_proxy",
            "quantum_barrier_path_amplitude_option_proxy",
            "cross_asset_correlation_heat_swap_proxy",
            "cliquet_global_floor_local_cap_proxy",
            "signature_trend_follower_option_proxy",
            "esg_linked_contingent_credit_default_swap_proxy",
            "stochastic_differential_games_control_proxy",
            "nonlocal_fractional_laplacian_proxy",
            "infinite_dimensional_heston_model_proxy",
            "lie_group_rough_path_signature_proxy",
            "mean_field_games_of_controls_proxy",
            "wasserstein_gradient_flow_measure_optimization_proxy",
            "malliavin_wiener_space_infinite_dimensional_greeks_proxy",
            "topological_quantum_field_theory_braid_group_proxy",
            "mfgc_congestion_control_proxy",
            "spde_manifold_limit_order_book_fluid_proxy",
        ],
        "feature_keys": list(QUANT_MODEL_FEATURE_KEYS),
        "resource_profile": resource,
        "mlx_hooks": {
            "mlx_core_random": mx is not None and hasattr(mx, "random"),
            "mx_grad": mx is not None and hasattr(mx, "grad"),
            "mlx_compile": mx is not None and hasattr(mx, "compile"),
            "mlx_nn": _module_available("mlx.nn"),
            "mlx_optimizers": _module_available("mlx.optimizers"),
            "mlx_lm": _module_available("mlx_lm"),
            "mlx_graphs": _module_available("mlx_graphs"),
            "mlx_snn": _module_available("mlxsnn"),
            "mlx_vision": _module_available("mlx_vision"),
            "esig": _module_available("esig"),
            "roughpy": _module_available("roughpy"),
            "quantlib": ql is not None,
            "fair_value_gradient": mx is not None and hasattr(mx, "grad"),
        },
        "execution_policy": {
            "direct_execution_allowed": False,
            "paper_trading_allowed": False,
            "purpose": "research_feature_collection_and_risk_context",
        },
    }
