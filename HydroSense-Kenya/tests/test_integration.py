"""test_integration.py – HydroSense-Kenya | Level 6 pytest tests"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pytest
import numpy as np
from numerical_methods import trapezoidal_rule, simpsons_rule

x = np.linspace(0, np.pi, 101)
y_sin = np.sin(x)
dx = x[1] - x[0]

class TestTrapezoidalRule:
    def test_sine_integral(self):
        assert abs(trapezoidal_rule(y_sin, dx) - 2.0) < 0.01

    def test_constant_function(self):
        y = np.full(101, 5.0)
        assert abs(trapezoidal_rule(y, 0.1) - 50.0) < 1e-6

    def test_positive_result(self):
        assert trapezoidal_rule(np.array([1.0,2.0,3.0]), 1.0) > 0

class TestSimpsonsRule:
    def test_sine_integral(self):
        assert abs(simpsons_rule(y_sin, dx) - 2.0) < 0.001

    def test_more_accurate_than_trap(self):
        y = np.sin(np.linspace(0, np.pi, 11))
        dx2 = np.pi / 10
        assert abs(simpsons_rule(y, dx2) - 2.0) <= abs(trapezoidal_rule(y, dx2) - 2.0)

    def test_odd_intervals_raises(self):
        with pytest.raises(ValueError):
            simpsons_rule(np.array([1.0,2.0,3.0,4.0]), 1.0)
