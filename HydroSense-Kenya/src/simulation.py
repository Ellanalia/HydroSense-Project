"""
simulation.py – HydroSense-Kenya
ICS 2207 Scientific Computing | Level 5
"""

import numpy as np


def _dSdt(S, R, I, ET, field_capacity_pct, drainage_coefficient):
    """Rate of change of soil moisture (continuous analogue)."""
    drainage = max(0.0, drainage_coefficient * (S - field_capacity_pct))
    return 0.1 * (R + I) - ET - drainage


def euler_simulation(S0, rainfall_array, et_array, irrigation_array,
                     field_capacity_pct, drainage_coefficient, min_moisture_pct=0.0):
    """Simulate daily soil moisture using the Euler method."""
    n = len(rainfall_array)
    S = np.zeros(n + 1)
    S[0] = S0
    for t in range(n):
        dS = _dSdt(S[t], rainfall_array[t], irrigation_array[t],
                   et_array[t], field_capacity_pct, drainage_coefficient)
        S[t + 1] = max(min_moisture_pct, S[t] + dS)
    return S


def runge_kutta_simulation(S0, rainfall_array, et_array, irrigation_array,
                            field_capacity_pct, drainage_coefficient, min_moisture_pct=0.0):
    """Simulate daily soil moisture using the 4th-order Runge-Kutta method."""
    n = len(rainfall_array)
    S = np.zeros(n + 1)
    S[0] = S0
    for t in range(n):
        R, I, ET = rainfall_array[t], irrigation_array[t], et_array[t]
        k1 = _dSdt(S[t],           R, I, ET, field_capacity_pct, drainage_coefficient)
        k2 = _dSdt(S[t] + 0.5*k1, R, I, ET, field_capacity_pct, drainage_coefficient)
        k3 = _dSdt(S[t] + 0.5*k2, R, I, ET, field_capacity_pct, drainage_coefficient)
        k4 = _dSdt(S[t] + k3,     R, I, ET, field_capacity_pct, drainage_coefficient)
        S[t + 1] = max(min_moisture_pct, S[t] + (k1 + 2*k2 + 2*k3 + k4) / 6.0)
    return S


def monte_carlo_rainfall(mean_rainfall, std_rainfall, n_scenarios=1000, n_days=30, seed=42):
    """Generate rainfall uncertainty scenarios using Monte Carlo (log-normal distribution)."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(loc=mean_rainfall, scale=std_rainfall, size=(n_scenarios, n_days))
    return np.maximum(0.0, raw)   # rainfall cannot be negative


def summarise_monte_carlo(mc_simulations):
    """Return mean, 5th, and 95th percentile across Monte Carlo moisture trajectories."""
    return {
        'mean':  np.mean(mc_simulations, axis=0),
        'p5':    np.percentile(mc_simulations, 5, axis=0),
        'p95':   np.percentile(mc_simulations, 95, axis=0),
        'p25':   np.percentile(mc_simulations, 25, axis=0),
        'p75':   np.percentile(mc_simulations, 75, axis=0),
    }
