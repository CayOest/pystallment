import pytest
import numpy as np

import discrete_pricer as dp

@pytest.mark.parametrize("S, expected", [(95, 9.924985820040824),(100, 7.4345679279511785),(105, 5.455621099180533)])
def test_simple_2_bermuda_var_spot(S, expected):
    r = 0.02
    d = 0.01
    vola = 0.2
    t = [0.5, 1]
    K = [100, 100]

    pricer = dp.BermudaPutPricer(S, r, d, vola, t, K)
    val = pricer.price()
    print("val = ", val)
    assert val == pytest.approx(expected)
    bound = pricer.stop
    print("bound = ", bound)
    assert bound[0] == pytest.approx(82.80551518426644)
    assert bound[1] == pytest.approx(K[-1])

@pytest.mark.parametrize("K_, expected_val, expected_bound", [
    (95, 5.096320651503248, 78.6652394250529),
    (100, 7.4345679279511785, 82.80551518426644),
    (105, 10.283838603451343, 86.94579094347964)])
def test_simple_2_bermuda_var_strike(K_, expected_val, expected_bound):
    S = 100
    r = 0.02
    d = 0.01
    vola = 0.2
    t = [0.5, 1]
    K = [K_, K_]

    pricer = dp.BermudaPutPricer(S, r, d, vola, t, K)
    val = pricer.price()
    print("val = ", val)
    assert val == pytest.approx(expected_val)
    bound = pricer.stop
    print("bound = ", bound)
    assert bound[0] == pytest.approx(expected_bound)
    assert bound[1] == pytest.approx(K[-1])

@pytest.mark.parametrize("r, expected_val, expected_bound", [
    (0.02, 5.103832345193979, 85.42072392819739),
    (0.05, 4.125830538769154, 90.37999452221896),
    (0.1, 2.889762081301484, 94.17925382683958)])
def test_simple_2_bermuda_var_r_zero_d(r, expected_val, expected_bound):
    S = 105
    d = 0.00
    vola = 0.2
    t = [0.5, 1]
    K_ = 100
    K = [K_, K_]

    pricer = dp.BermudaPutPricer(S, r, d, vola, t, K)
    val = pricer.price()
    print("val = ", val)
    assert val == pytest.approx(expected_val)
    bound = pricer.stop
    print("bound = ", bound)
    assert bound[0] == pytest.approx(expected_bound)
    assert bound[1] == pytest.approx(K[-1])

@pytest.mark.parametrize("r, expected_val, expected_bound", [
    (0.02, 5.80893567123389, 78.39673533495166),
    (0.05, 4.756018088407906, 88.09369576069055),
    (0.1, 3.406415035950081, 93.13602352444543)])
def test_simple_2_bermuda_var_r_fixed_d0_02(r, expected_val, expected_bound):
    S = 105
    d = 0.02
    vola = 0.2
    t = [0.5, 1]
    K_ = 100
    K = [K_, K_]

    pricer = dp.BermudaPutPricer(S, r, d, vola, t, K)
    val = pricer.price()
    print("val = ", val)
    assert val == pytest.approx(expected_val)
    bound = pricer.stop
    print("bound = ", bound)
    assert bound[0] == pytest.approx(expected_bound)
    assert bound[1] == pytest.approx(K[-1])

def test_simple_4_bermuda():
    S = 105
    K_ = 100
    r = 0.02
    d = 0.01
    vola = 0.2
    t = [0.25, 0.5, 0.75, 1]
    K = np.ones(4)*K_

    pricer = dp.BermudaPutPricer(S, r, d, vola, t, K)
    val = pricer.price()
    print("val = ", val)
    assert val == pytest.approx(5.484359556609775)
    bound = pricer.stop
    assert bound[-1] == pytest.approx(K[-1])