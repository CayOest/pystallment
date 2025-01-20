import pytest

from pystallment.algorithms import lsmc as lsmc
from pystallment import option as opt
from pystallment.algorithms.binomial import BinomialPricer

def test_american_put():
    S = 95
    K = 100
    r = 0.05
    d = 0.04
    vola = 0.2
    T = 1.0

    option = opt.AmericanOption(S, K, r, d, vola, T, -1)
    bprice = 9.754
    print(f"Binomial Price = {bprice:.3f}")
    mcp = lsmc.LSMCPricer(option)
    mcp.seed = 42
    mcp.num_paths = int(1e4)
    mcprice = mcp.price()
    print(f"MC Price = {mcprice:.3f}")
    assert mcprice == pytest.approx(bprice, 1e-1)

def test_installment_call():
    S = 96
    K = 100
    r = 0.05
    d = 0.04
    vola = 0.2
    T = 1.0
    q = 3

    option = opt.ContinuousInstallmentOption(S, K, r, d, vola, T, q, phi=+1)

    # Binomial Price
    bprice = 3.652
    print(f"Binomial Price = {bprice:.3f}")

    mcp = lsmc.LSMCPricer(option)
    mcp.num_paths = int(1e4)
    mcp.seed = 42
    mcprice = mcp.price()
    print(f"MC Price = {mcprice:.3f}")
    assert mcprice == pytest.approx(bprice, 1e-1)


def test_american_installment_call():
    S = 96
    K = 100
    r = 0.05
    d = 0.04
    vola = 0.2
    T = 1.0
    q = 3

    option = opt.AmericanContinuousInstallmentOption(S, K, r, d, vola, T, q, phi=+1)

    mcp = lsmc.LSMCPricer(option)
    mcp.num_paths = int(1e4)
    mcp.seed = 42
    mcprice = mcp.price()
    print(f"MC Price = {mcprice:.3f}")
    assert mcprice == pytest.approx(3.8362, abs=1e-1)
