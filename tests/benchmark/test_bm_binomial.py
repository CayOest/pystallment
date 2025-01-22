import pytest
import pytest_benchmark
import pystallment.option as opt
import pystallment.algorithms.binomial as bp

@pytest.mark.parametrize("n", [1000, 2000, 4000])
def test_american_put_performance(benchmark, n):
    S = 105
    r = 0.05
    d = 0.04
    K = 100
    vola = 0.2
    T = 1

    option = opt.AmericanOption(S, K, r, d, vola, T, phi=-1)
    pricer = bp.BinomialPricer(option)
    pricer.num_steps = n
    price = benchmark(pricer.price)