"""
optimization.py – HydroSense-Kenya
ICS 2207 Scientific Computing | Level 5
"""

import numpy as np
from simulation import euler_simulation


def greedy_irrigation_schedule(S0, rainfall_array, et_array,
                                field_capacity_pct, drainage_coefficient,
                                min_moisture_pct, target_moisture_pct):
    """
    Greedy schedule: irrigate only when soil moisture is predicted to fall
    below the stress threshold, bringing it to target.
    """
    n = len(rainfall_array)
    irrigation = np.zeros(n)
    S = S0

    from simulation import _dSdt
    for t in range(n):
        # Predict next S without irrigation
        dS_no_irr = _dSdt(S, rainfall_array[t], 0.0, et_array[t],
                           field_capacity_pct, drainage_coefficient)
        S_pred = S + dS_no_irr
        if S_pred < min_moisture_pct:
            # How much irrigation (mm) to reach target?
            shortfall = target_moisture_pct - S_pred
            irrigation[t] = max(0.0, shortfall / 0.1)   # convert % to mm
        # Step forward with chosen irrigation
        from simulation import _dSdt as _d
        dS = _d(S, rainfall_array[t], irrigation[t], et_array[t],
                field_capacity_pct, drainage_coefficient)
        S = max(min_moisture_pct, S + dS)

    return irrigation


def minimise_water_use(S0, rainfall_array, et_array,
                        field_capacity_pct, drainage_coefficient,
                        min_moisture_pct, target_moisture_pct,
                        n_iter=500, lr=0.1, seed=42):
    """
    Gradient-descent-style optimisation to minimise total irrigation while
    keeping soil moisture above min_moisture_pct at every time step.

    Returns
    -------
    dict: optimised_schedule, total_water_used_mm, stress_days, loss_history
    """
    rng = np.random.default_rng(seed)
    n = len(rainfall_array)

    # Initialise with small random irrigation amounts
    irrigation = np.maximum(0.0, rng.normal(1.0, 0.5, n))
    loss_history = []

    for _ in range(n_iter):
        # Forward pass: simulate moisture
        S_traj = euler_simulation(S0, rainfall_array, et_array, irrigation,
                                   field_capacity_pct, drainage_coefficient, min_moisture_pct)

        # Loss = total water used + penalty for stress days
        stress_penalty = 100.0 * np.sum(np.maximum(0.0, min_moisture_pct - S_traj[1:]))
        total_water = np.sum(irrigation)
        loss = total_water + stress_penalty
        loss_history.append(loss)

        # Numerical gradient w.r.t. each irrigation value
        grad = np.zeros(n)
        for t in range(n):
            irr_plus = irrigation.copy()
            irr_plus[t] += 1e-3
            S_plus = euler_simulation(S0, rainfall_array, et_array, irr_plus,
                                       field_capacity_pct, drainage_coefficient, min_moisture_pct)
            stress_plus = 100.0 * np.sum(np.maximum(0.0, min_moisture_pct - S_plus[1:]))
            grad[t] = ((np.sum(irr_plus) + stress_plus) - loss) / 1e-3

        # Gradient descent step + project to non-negative
        irrigation = np.maximum(0.0, irrigation - lr * grad)

    # Final simulation
    S_final = euler_simulation(S0, rainfall_array, et_array, irrigation,
                                field_capacity_pct, drainage_coefficient, min_moisture_pct)
    stress_days = int(np.sum(S_final[1:] < min_moisture_pct))

    return {
        'optimised_schedule': irrigation,
        'total_water_used_mm': float(np.sum(irrigation)),
        'stress_days': stress_days,
        'loss_history': loss_history,
        'S_trajectory': S_final
    }
