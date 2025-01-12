import numpy as np

from discrete_pricer import InstallmentCallPricer, BermudaPutPricer, call, put
from continuous_pricer import ContinuousInstallmentOptionPricer
from fdm_pricer import FDMPricer

def single_check_fixed_q(S, K, r, d, vola, T, q, n):
    print("n = ", n)
    t = np.arange(T/n, T + T/n, T/n)
    q_ = np.ones(len(t))*q
    q_[-1] = K

    pricer = InstallmentCallPricer(S, r, d, vola, t, q_)
    call = pricer.price()
    print("Installment Price = ", call)

    K_ = np.zeros(len(t))
    K_[-1] = K
    for i in range(n - 1):
        K_[i] = 0
        for j in range(i, n):
            K_[i] += q_[j] * np.exp(-r * (t[j] - t[i]))

    bpricer = BermudaPutPricer(S, r, d, vola, t, K_)
    put = bpricer.price()
    print("Put Price = ", put)

    print("Installment Stops = ", pricer.stop)
    print("Bermuda Put Stops = ", bpricer.stop)

    check = put + S * np.exp(-d * T) - call - np.sum(q_ * np.exp(-r * t))
    print(f"check = {check:.6e}")
    print(100*'*')

    return (call, put, check, pricer.stop, bpricer.stop)

def single_check_fixed_K(S, K, r, d, vola, T, n):
    q = K*(1-np.exp(-r*T/n))
    return single_check_fixed_q(S, K, r, d, vola, T, q, n)

import matplotlib.pyplot as plt

# Beispiel-Nutzung
if __name__ == "__main__":
    S = 1
    K = 1
    r = 0.02
    d = 0.00
    vola = 0.2
    T = 1
    q = 1

    p = FDMPricer(S, K, r, d, vola, T, q=q)
    price = p.calc()
    print("American Put = ", price)

    p = ContinuousInstallmentOptionPricer(S, K, r, d, vola, T, q=r*K, phi=+1)
    call = p.value()
    print("Installment Call = ", call)

    check = price + S - call - K
    print("check = ", check)

    p.phi = -1
    p.q = 2
    iput = p.value()
    print("Installment Put = ", iput)
    print("check = ", iput-price)

    # n = 2
    #single_check_fixed_q(S, K, r, d, vola, T, q, 2)
    #single_check_fixed_K(S, K, r, d, vola, T, 2)

    # Plot initialisieren
    plt.ion()  # Interaktive Plot-Anzeige einschalten
    fig, ax = plt.subplots(figsize=(8, 6))  # Figur und Achse erstellen

    p = ContinuousInstallmentOptionPricer(S, K, r, d, vola, T, 0, -1)
    print("vanilla value = ", p.vanilla_value())
    p2 = put(S, K, r, d, vola, T)
    print("BS value = ", p2)

    for q in [1, 3, 8]:
        print("q = ", q)
        lctpricer = ContinuousInstallmentOptionPricer(S, K, r, d, vola, T, q, -1)
        value = lctpricer.value()
        print("value = ", value)
        t = np.linspace(0.001, 0.999, 1000)
        sb = [lctpricer.stop_bound(T-t[i]) for i in range(len(t))]

        ax.plot(t, sb, label=f'q = {q}')  # Linie hinzufügen
        ax.legend()  # Legende aktualisieren
        plt.draw()  # Zeichne den aktuellen Plot
        plt.pause(0.1)

        values = {}
        for n in range(3, 3):
            (c, p, check, call_stop, put_stop) = single_check_fixed_K(S, K, r, d, vola, T, n)

            # Plotten
            t = np.arange(T/n, T + T/n, T/n)
            ax.plot(t, call_stop, label=f'q = {q}, n = {n}')  # Linie hinzufügen
            ax.legend()  # Legende aktualisieren
            plt.draw()  # Zeichne den aktuellen Plot
            plt.pause(0.1)

    # Interaktivität deaktivieren und finalen Plot anzeigen
    plt.ioff()
    plt.show()