import pytest

import binomial_pricer as bp
import option as opt

@pytest.mark.parametrize("S, r, d, expected", [
    (95, 0.02, 0.00, 9.559604714303656),
    (105, 0.02, 0.00, 5.182835507452405),
    (95, 0.05, 0.00, 8.452646381525147),
    (105, 0.05, 0.00, 4.306098201373055),
    (95, 0.1, 0.00, 7.138640520662231),
    (105, 0.1, 0.00, 3.2045597568873134),
    # d = 0.04
    (95, 0.02, 0.04, 11.385533383653373),
    (105, 0.02, 0.04, 6.5735670941797775),
    (95, 0.05, 0.04, 9.755163939182946),
    (105, 0.05, 0.04, 5.362941132857127),
    (95, 0.1, 0.04, 8.011765981901965),
    (105, 0.1, 0.04, 3.9592945293385684),
    ])
def test_american_put(S, r, d, expected):
    K = 100
    vola = 0.2
    T = 1

    option = opt.AmericanOption(S, K, r, d, vola, T, phi=-1)
    pricer = bp.BinomialPricer(option)
    price = pricer.calc()
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
    price = pricer.calc()
    assert price == pytest.approx(expected, rel=1e-3)


