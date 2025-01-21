import pytest

from pystallment.algorithms import binomial as bp
from pystallment import black_scholes as bs, option as opt
import test_data as td

@pytest.mark.parametrize("S, r, d, expected", td.std_american_put)
def test_american_put(S, r, d, expected):
    K = 100
    vola = 0.2
    T = 1

    option = opt.AmericanOption(S, K, r, d, vola, T, phi=-1)
    pricer = bp.BinomialPricer(option)
    price = pricer.price()
    assert price == pytest.approx(expected, rel=1e-3)

@pytest.mark.parametrize("S, r, d, expected", [
    (95, 0.02, 0.00, 6.273149615726745),
    (105, 0.02, 0.00, 12.049445099063213),
    (95, 0.05, 0.00, 7.512960484843028),
    (105, 0.05, 0.00, 13.859733677467393),
    (95, 0.1, 0.00, 9.865172136958078),
    (105, 0.1, 0.00, 17.094682147393506),
    # d = 0.04
    (95, 0.02, 0.04, 4.771117612387967),
    (105, 0.02, 0.04, 9.785606805514979),
    (95, 0.05, 0.04, 5.666670106328403),
    (105, 0.05, 0.04, 11.050302717875677),
    (95, 0.1, 0.04, 7.643585168018241),
    (105, 0.1, 0.04, 13.918641713394047),
    ])
def test_american_call(S, r, d, expected):
    K = 100
    vola = 0.2
    T = 1
    option = opt.AmericanOption(S, K, r, d, vola, T, phi=+1)
    pricer = bp.BinomialPricer(option)
    price = pricer.price()
    assert price == pytest.approx(expected, rel=1e-3)

@pytest.mark.parametrize("S, r, d", [
    (95, 0.02, 0.00),
    (105, 0.05, 0.00),
    (95, 0.1, 0.00),
    (95, 0.02, 0.04),
    (105, 0.05, 0.04),
    (95, 0.1, 0.04),
    ])
def test_euro_option(S, r, d):
    K = 100
    vola = 0.2
    T = 1
    types = [-1, +1]
    for phi in types:
        option = opt.Option(S, K, r, d, vola, T, phi=phi)
        pricer = bp.BinomialPricer(option)
        price = pricer.price()
        expected = bs.option_value(S, K, r, d, vola, T, phi)
        assert price == pytest.approx(expected, rel=1e-3)

@pytest.mark.parametrize("vola, S, T, q, CNFD", td.ciurlia_inst_call_short)
def test_installment_call_ciurlia(vola, S, T, q, CNFD):
    K = 100
    r = 0.05
    d = 0.04

    print(f"CNFD = {CNFD:.3f}")

    option = opt.ContinuousInstallmentOption(S=S, K=K, r=r, d=d, vola=vola, T=T, q=q, phi=+1)
    pricer = bp.BinomialPricer(option, num_steps=1000, factor_adjustment='r-d')
    val = pricer.price()
    print(f"val = {val:.3f}, diff = {(val-CNFD)*100/max(val, CNFD):.3f} %")
    assert val == pytest.approx(CNFD, abs=1e-2)