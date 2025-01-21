import pytest

from pystallment.algorithms import fdm as fdm
from pystallment import option as opt
import test_data as td

@pytest.mark.skip
def test_single_american_put():
    S = 95
    r = 0.02
    d = 0.0
    K = 100
    vola = 0.2
    T = 1
    expected = 9.559604714303656
    option = opt.AmericanOption(S, K, r, d, vola, T, phi=-1)

    space_steps = int(1e4)
    pricer = fdm.FDMPricer(option)
    pricer.space_steps = space_steps

    price = pricer.price()
    assert price == pytest.approx(expected, abs=5e-2)

@pytest.mark.parametrize("S, r, d, expected", td.std_american_put)
def test_american_put(S, r, d, expected):
    K = 100
    vola = 0.2
    T = 1
    option = opt.AmericanOption(S, K, r, d, vola, T, phi=-1)

    pricer = fdm.FDMPricer(option)
    price = pricer.price()
    assert price == pytest.approx(expected, rel=1e-2)

@pytest.mark.skip
def test_single_installment_call_ciurlia():
    (0.2, 104, 1, 8, )
    S = 104
    K = 100
    r = 0.05
    d = 0.04
    vola = 0.2
    T = 1
    q = 8

    expected = 3.7678

    option = opt.ContinuousInstallmentOption(S=S, K=K, r=r, d=d, vola=vola, T=T, q=q, phi=+1)
    pricer = fdm.FDMPricer(option)
    pricer.space_steps *= 10
    val = pricer.price()
    print(f"val = {val:.3f}, diff = {(val - expected) * 100 / max(val,expected):.3f} %")

    assert val == pytest.approx(expected, rel=1e-2)

@pytest.mark.parametrize("vola, S, T, q, CNFD", td.ciurlia_inst_call_short)
def test_installment_call_ciurlia(vola, S, T, q, CNFD):
    K = 100
    r = 0.05
    d = 0.04

    print(f"CNFD = {CNFD:.3f}")

    option = opt.ContinuousInstallmentOption(S=S, K=K, r=r, d=d, vola=vola, T=T, q=q, phi=+1)
    pricer = fdm.FDMPricer(option)
    val = pricer.price()
    print(f"FDM = {val:.3f}, diff = {(val-CNFD)*100/max(val, CNFD):.3f} %")
    assert val == pytest.approx(CNFD, rel=1e-2)