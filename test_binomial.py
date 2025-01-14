import test_binomial
from binomial_pricer import BinomialPricer

if __name__ == "__main__":
    S = 105
    K = 100
    r = 0.02
    d = 0.00
    vola = 0.2
    T = 1
    N = 1000

    pricer = BinomialPricer(S, K, r, d, vola, T, 0, -1)
    pricer.num_steps = N
    price = pricer.calc()
    print("Put = ", price)
    print("Ex Bound = ", pricer.ex_bound)