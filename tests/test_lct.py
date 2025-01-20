import pytest

import test_data as td
from pystallment.algorithms import lct as lct
from pystallment import black_scholes as bs, option


@pytest.mark.skip()
def test_vanilla_call():
    S = 105
    K = 100
    r = 0.03
    d = 0.01
    vola = 0.2
    T = 1
    bs_value = bs.call(S, K, r, d, vola, T)
    print("BS Value = ", bs_value)

    for n in range(2, 11):
        opt = option.ContinuousInstallmentOption(S, K, r, d, vola, T, 0, +1)
        lct_pricer = lct.LCTPricer(opt, n)
        lct_value = lct_pricer.vanilla_value()
        print(f"LCT ({n}) = {lct_value}, diff={(lct_value-bs_value)*100/bs_value}%")

@pytest.mark.parametrize("vola, S, T, q, CNFD", td.ciurlia_inst_call)
def test_installment_call_ciurlia(vola, S, T, q, CNFD):
    K = 100
    r = 0.05
    d = 0.04

    print(f"CNFD = {CNFD:.3f}")

    n_ = [1000]
    for n in n_:
        opt = option.ContinuousInstallmentOption(S=S, K=K, r=r, d=d, vola=vola, T=T, q=q, phi=+1)
        pricer = lct.LCTPricer(opt, num_steps=7)
        val = pricer.value()
        print(f"val ({n}) = {val:.3f}, diff = {(val-CNFD)*100/max(val, CNFD):.3f} %")
        assert val == pytest.approx(CNFD, abs=1.5e-1)