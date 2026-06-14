"""test_root_finding.py – HydroSense-Kenya | Level 6 pytest tests"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pytest
from numerical_methods import bisection, newton_raphson, secant

f  = lambda x: x**2 - 4
df = lambda x: 2*x
f_irr  = lambda I: 27.9 + 0.1*I - 4.5 - 33.0
df_irr = lambda I: 0.1

class TestBisection:
    def test_known_root(self):
        r = bisection(f, 0, 3)
        assert abs(r['root'] - 2.0) < 1e-4

    def test_converged(self):
        assert bisection(f, 0, 3)['converged'] is True

    def test_irrigation_root(self):
        r = bisection(f_irr, 0, 200)
        assert r['converged'] and r['root'] >= 0

    def test_no_bracket_returns_not_converged(self):
        r = bisection(f, 3, 5)
        assert not r['converged']

class TestNewtonRaphson:
    def test_known_root(self):
        r = newton_raphson(f, df, x0=1.5)
        assert abs(r['root'] - 2.0) < 1e-5

    def test_converged(self):
        assert newton_raphson(f, df, x0=1.5)['converged'] is True

    def test_fewer_iters_than_bisection(self):
        r_nr = newton_raphson(f, df, x0=1.5)
        r_bi = bisection(f, 0, 3)
        assert r_nr['iterations'] < r_bi['iterations']

class TestSecant:
    def test_known_root(self):
        r = secant(f, x0=1.0, x1=3.0)
        assert abs(r['root'] - 2.0) < 1e-4

    def test_converged(self):
        assert secant(f, x0=1.0, x1=3.0)['converged'] is True
