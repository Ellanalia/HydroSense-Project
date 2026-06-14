"""test_simulation.py – HydroSense-Kenya | Level 6 pytest tests"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pytest
import numpy as np
from simulation import euler_simulation, runge_kutta_simulation, monte_carlo_rainfall

N=30; S0=33.0; FC=41.0; DC=0.18
RAIN=np.full(N,3.); ET=np.full(N,4.5); IRR=np.zeros(N)

class TestEuler:
    def test_length(self):    assert len(euler_simulation(S0,RAIN,ET,IRR,FC,DC)) == N+1
    def test_starts_S0(self): assert euler_simulation(S0,RAIN,ET,IRR,FC,DC)[0] == S0
    def test_declines_without_rain(self):
        S = euler_simulation(S0, np.zeros(N), np.full(N,5.0), IRR, FC, DC)
        assert S[-1] < S0
    def test_non_negative(self):
        S = euler_simulation(S0, np.zeros(N), np.full(N,20.), IRR, FC, DC)
        assert all(s >= 0 for s in S)
    def test_does_not_exceed_field_capacity_without_irrigation(self):
        S = euler_simulation(S0, np.zeros(N), np.zeros(N), IRR, FC, DC)
        assert all(s <= FC for s in S)

class TestRK4:
    def test_length(self):    assert len(runge_kutta_simulation(S0,RAIN,ET,IRR,FC,DC)) == N+1
    def test_starts_S0(self): assert runge_kutta_simulation(S0,RAIN,ET,IRR,FC,DC)[0] == S0
    def test_non_negative(self):
        S = runge_kutta_simulation(S0, np.zeros(N), np.full(N,20.), IRR, FC, DC)
        assert all(s >= 0 for s in S)
    def test_same_final_order_of_magnitude_as_euler(self):
        e = euler_simulation(S0,RAIN,ET,IRR,FC,DC)[-1]
        r = runge_kutta_simulation(S0,RAIN,ET,IRR,FC,DC)[-1]
        assert abs(e - r) < 5.0   # same ballpark

class TestMonteCarlo:
    def test_shape(self):
        assert monte_carlo_rainfall(5.,2.,1000,30,seed=42).shape == (1000,30)
    def test_non_negative(self):
        assert (monte_carlo_rainfall(5.,2.,1000,30,seed=42) >= 0).all()
    def test_reproducible(self):
        r1 = monte_carlo_rainfall(5.,2.,500,30,seed=99)
        r2 = monte_carlo_rainfall(5.,2.,500,30,seed=99)
        assert np.allclose(r1,r2)
    def test_mean_close_to_input(self):
        mc = monte_carlo_rainfall(5.,2.,5000,30,seed=0)
        assert abs(mc.mean() - 5.0) < 0.5
