import pytest

import lsmc_pricer as lsmc
import option as opt
import binomial as bin
from binomial import BinomialPricer


def test_american_put():
    S = 95
    K = 100
    r = 0.05
    d = 0.04
    vola = 0.2
    T = 1.0

    option = opt.AmericanOption(S, K, r, d, vola, T, -1)
    bp = BinomialPricer(option)
    bprice = bp.calc()
    print(f"Binomial Price = {bprice:.3f}")
    mcp = lsmc.LSMCPricer(option)
    mcp.num_paths = int(1e5)
    mcprice = mcp.calc()
    print(f"MC Price = {mcprice:.3f}")

def test_installment_call():
    S = 96
    K = 100
    r = 0.02
    d = 0.00
    vola = 0.2
    T = 1.0
    q = 0

    option = opt.ContinuousInstallmentOption(S, K, r, d, vola, T, q, phi=+1)

    # Binomial Price
    bp = BinomialPricer(option)
    bprice = bp.calc()
    print(f"Binomial Price = {bprice:.3f}")

    mcp = lsmc.LSMCPricer(option)
    mcp.num_paths = int(1e5)
    mcprice = mcp.calc()
    print(f"MC Price = {mcprice:.3f}")