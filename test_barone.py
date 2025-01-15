from barone_pricer import BaronePricer

if __name__ == "__main__":
    S = 105
    K = 100
    r = 0.02
    d = 0.00
    vola = 0.2
    T = 1

    pricer = BaronePricer(S, K, r, d, vola, T, -1)
    price = pricer.calc()
    print("Put = ", price)