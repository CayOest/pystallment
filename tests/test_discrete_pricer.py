import pytest
import numpy as np

import discrete_pricer as dp
import option as opt
import test_data as td

@pytest.mark.parametrize("S, expected", [(95, 9.924985820040824),(100, 7.4345679279511785),(105, 5.455621099180533)])
def test_simple_2_bermuda_var_spot(S, expected):
    r = 0.02
    d = 0.01
    vola = 0.2
    t = [0.5, 1]
    K = [100, 100]
    option = opt.make_bermuda_put(S, r, d, vola, t, K)

    pricer = dp.BermudaPricer(option)
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
    option = opt.make_bermuda_put(S, r, d, vola, t, K)

    pricer = dp.BermudaPricer(option)
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
    option = opt.make_bermuda_put(S, r, d, vola, t, K)

    pricer = dp.BermudaPricer(option)
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
    option = opt.make_bermuda_put(S, r, d, vola, t, K)

    pricer = dp.BermudaPricer(option)
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
    option = opt.make_bermuda_put(S, r, d, vola, t, K)

    pricer = dp.BermudaPricer(option)
    val = pricer.price()
    print("val = ", val)
    assert val == pytest.approx(5.484359556609775)
    bound = pricer.stop
    assert bound[-1] == pytest.approx(K[-1])
    

@pytest.mark.parametrize("S, expected", [(95, 2.848602143652627),(100, 4.772140468389111),(105, 7.30436820013986)])
def test_simple_2_installment_var_spot(S, expected):
    K = 100
    r = 0.02
    d = 0.01
    vola = 0.2
    t = [0.5, 1]
    q = [5]
    option = opt.make_installment_call(S, K, r, d, vola, t, q)
    print(repr(option))

    pricer = dp.InstallmentCallPricer(option)
    val = pricer.price()
    print("val = ", val)
    assert val == pytest.approx(expected)
    bound = pricer.stop
    print("bound = ", bound)
    assert bound[0] == pytest.approx(98.3605222246888)
    assert bound[1] == pytest.approx(K)

@pytest.mark.parametrize("K, expected_val, expected_bound", [
    (95, 6.916660537072353, 93.941801550),
    (100, 4.77214046838911, 98.360522),
    (105, 3.17237901234, 102.7643)])
def test_simple_2_installment_var_strike(K, expected_val, expected_bound):
    S = 100
    r = 0.02
    d = 0.01
    vola = 0.2
    t = [0.5, 1]
    q = [5]
    option = opt.make_installment_call(S, K, r, d, vola, t, q)
    print(repr(option))

    pricer = dp.InstallmentCallPricer(option)
    val = pricer.price()
    print("val = ", val)
    assert val == pytest.approx(expected_val)
    bound = pricer.stop
    print("bound = ", bound)
    assert bound[0] == pytest.approx(expected_bound)
    assert bound[1] == pytest.approx(K)

def test_simple_4_installment():
    S = 105
    K = 100
    r = 0.02
    d = 0.01
    vola = 0.2
    t = [0.25, 0.5, 0.75, 1]
    q_ = K*(1-np.exp(-r*0.25))
    q = (np.ones(3)*q_).tolist()
    option = opt.make_installment_call(S, K, r, d, vola, t, q)
    print(repr(option))

    pricer = dp.InstallmentCallPricer(option)
    val = pricer.price()
    assert val == pytest.approx(9.95180768851261)
    bound = pricer.stop
    assert bound[-1] == pytest.approx(K)

@pytest.mark.parametrize("S, r", [
    (95, 0.02),
    (95, 0.04),
    (95, 0.1),
    (100, 0.02),
    (100, 0.04),
    (100, 0.1),
    (105, 0.02),
    (105, 0.04),
    (105, 0.1),
    ])
def test_formula(S, r):
    K = 100
    d = 0
    vola = 0.2
    t = [0.25, 0.5, 0.75, 1]
    q_ = K*(1-np.exp(-r*0.25))
    q = (np.ones(3)*q_).tolist()

    call_option = opt.make_installment_call(S, K, r, d, vola, t, q)
    print(repr(call_option))
    call_pricer = dp.InstallmentCallPricer(call_option)
    call_val = call_pricer.price()

    put_option = opt.make_bermuda_put(S, r, d, vola, t, K*np.ones(4))
    print(repr(put_option))
    put_pricer = dp.BermudaPricer(put_option)
    put_val = put_pricer.price()

    npv = np.exp(-r*t[-1])*K
    for i in range(len(q)):
        npv += np.exp(-r*t[i])*q[i]

    check = put_val + S - call_val - npv
    assert check == pytest.approx(0.0, abs=2e-3)

    for i in range(len(call_pricer.stop)):
        assert call_pricer.stop[i] == pytest.approx(put_pricer.stop[i], rel=1e-3)

@pytest.mark.parametrize("q, S, gaver, krishni", td.anton_inst_call)
def test_extrapolation(q, S, gaver, krishni):
    K  = 100
    r = 0.03
    d = 0.05
    vola = 0.2
    T = 1
    methods = [('poly', 5), ('poly', 8), ('rich', 3), ('rich', 4), ('rich', 5)]

    print(f"gaver = {gaver:.3f}")
    print(f"krishni = {krishni:.3f}")

    for m, n in methods:
        option = opt.make_installment_call(S=S, K=K, r=r, d=d, vola=vola, t=T, q=q)
        pricer = dp.ExtrapolationPricer(option, n, interpol=m)
        val = pricer.calc()
        print(f"{m} = {val:.3f}")

class ExtrapolationTest:
    def setup_method(self):
        self.values = {}

    @pytest.mark.parametrize("vola, S, T, q, CNFD", td.ciurlia_inst_call)
    def test_extrapolation_ciurlia(self, vola, S, T, q, CNFD):
        K = 100
        r = 0.05
        d = 0.04

        print(f"CNFD = {CNFD:.3f}")

        methods = [('poly', 3), ('poly', 5), ('rich', 3), ('rich', 4), ('rich', 5)]

        values = {}
        for m, n in methods:
            option = opt.make_installment_call(S=S, K=K, r=r, d=d, vola=vola, t=T, q=q)
            pricer = dp.ExtrapolationPricer(option, n, interpol=m)
            val = pricer.calc()
            print(f"{m}, n: {n} = {val:.3f}")
            self.values[(m, n)].append(val)