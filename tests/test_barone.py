from barone_pricer import BaronePricer

if __name__ == "__main__":
    S = 90
    K = 100
    r = 0.08
    d = 0.0
    vola = 0.2
    T = 3

    pricer = BaronePricer(S, K, r, d, vola, T, -1)
    price = pricer.calc()
    print("Put = ", price)