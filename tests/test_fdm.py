import pytest

from pystallment.algorithms import fdm as fdm
from pystallment import option as opt
import test_data as td

@pytest.mark.parametrize("vola, S, T, q, CNFD", td.ciurlia_inst_call_short)
def test_installment_call_ciurlia(vola, S, T, q, CNFD):
    K = 100
    r = 0.05
    d = 0.04

    print(f"CNFD = {CNFD:.3f}")

    n_ = [1000]
    for n in n_:
        option = opt.ContinuousInstallmentOption(S=S, K=K, r=r, d=d, vola=vola, T=T, q=q, phi=+1)
        pricer = fdm.FDMPricer(option)
        val = pricer.price()
        print(f"val ({n}) = {val:.3f}, diff = {(val-CNFD)*100/max(val, CNFD):.3f} %")
        assert val == pytest.approx(CNFD, rel=1e-2)