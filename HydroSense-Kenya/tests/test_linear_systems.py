"""test_linear_systems.py – HydroSense-Kenya | Level 6 pytest tests"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pytest
import numpy as np
from numerical_methods import gaussian_elimination

class TestGaussianElimination:
    def test_2x2(self):
        A = np.array([[2.,1.],[1.,3.]])
        b = np.array([5.,10.])
        x = gaussian_elimination(A, b)
        assert abs(x[0]-1.0) < 1e-6 and abs(x[1]-3.0) < 1e-6

    def test_identity(self):
        A = np.eye(3); b = np.array([4.,7.,2.])
        assert np.allclose(gaussian_elimination(A, b), b, atol=1e-8)

    def test_ax_equals_b(self):
        np.random.seed(0)
        A = np.random.rand(4,4) + np.eye(4)*5
        b = np.random.rand(4)
        x = gaussian_elimination(A, b)
        assert np.allclose(A @ x, b, atol=1e-6)

    def test_3zone_allocation(self):
        A = np.array([[1.,1.,1.],[1.,0.,-1.5],[0.,1.,-1.2]])
        b = np.array([60.,0.,0.])
        x = gaussian_elimination(A, b)
        assert np.allclose(A @ x, b, atol=1e-4)
