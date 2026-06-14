"""
numerical_methods.py – HydroSense-Kenya
ICS 2207 Scientific Computing | Level 3

Manually implemented numerical methods:
  - Root finding:     bisection, newton_raphson, secant
  - Differentiation:  forward_difference, backward_difference, central_difference
  - Integration:      trapezoidal_rule, simpsons_rule
  - Linear systems:   gaussian_elimination
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# ROOT FINDING
# ══════════════════════════════════════════════════════════════════════════════

def bisection(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) > 0:
        return {'root': None, 'iterations': 0, 'error': None, 'converged': False, 'errors': []}
    errors = []
    for i in range(1, max_iter + 1):
        mid = (a + b) / 2.0
        err = abs(b - a) / 2.0
        errors.append(err)
        if f(mid) == 0 or err < tol:
            return {'root': mid, 'iterations': i, 'error': err, 'converged': True, 'errors': errors}
        if f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid
    return {'root': (a + b) / 2.0, 'iterations': max_iter, 'error': abs(b - a) / 2.0, 'converged': False, 'errors': errors}


def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    x = x0
    errors = []
    for i in range(1, max_iter + 1):
        fx = f(x)
        err = abs(fx)
        errors.append(err)
        if err < tol:
            return {'root': x, 'iterations': i, 'error': err, 'converged': True, 'errors': errors}
        dfx = df(x)
        if dfx == 0:
            return {'root': x, 'iterations': i, 'error': err, 'converged': False, 'errors': errors}
        x = x - fx / dfx
    return {'root': x, 'iterations': max_iter, 'error': abs(f(x)), 'converged': False, 'errors': errors}


def secant(f, x0, x1, tol=1e-6, max_iter=100):
    errors = []
    for i in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)
        if f1 - f0 == 0:
            return {'root': x1, 'iterations': i, 'error': abs(f1), 'converged': False, 'errors': errors}
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        err = abs(x2 - x1)
        errors.append(err)
        if err < tol:
            return {'root': x2, 'iterations': i, 'error': err, 'converged': True, 'errors': errors}
        x0, x1 = x1, x2
    return {'root': x1, 'iterations': max_iter, 'error': abs(f(x1)), 'converged': False, 'errors': errors}


# ══════════════════════════════════════════════════════════════════════════════
# DIFFERENTIATION
# ══════════════════════════════════════════════════════════════════════════════

def forward_difference(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

def backward_difference(f, x, h=1e-5):
    return (f(x) - f(x - h)) / h

def central_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def trapezoidal_rule(y_values, dx):
    y = np.asarray(y_values, dtype=float)
    return dx * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

def simpsons_rule(y_values, dx):
    y = np.asarray(y_values, dtype=float)
    n = len(y) - 1
    if n % 2 != 0:
        raise ValueError("Simpson's rule requires an even number of intervals (odd number of points).")
    result = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return (dx / 3.0) * result


# ══════════════════════════════════════════════════════════════════════════════
# LINEAR SYSTEMS
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_elimination(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])
    for col in range(n):
        max_row = col + np.argmax(np.abs(Ab[col:, col]))
        Ab[[col, max_row]] = Ab[[max_row, col]]
        if Ab[col, col] == 0:
            raise ValueError(f"Singular matrix at column {col}.")
        for row in range(col + 1, n):
            factor = Ab[row, col] / Ab[col, col]
            Ab[row, col:] -= factor * Ab[col, col:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    return x
