import numpy as np

from discrete_pricer import InstallmentCallPricer, BermudaPutPricer, call, put
from lct_pricer import LCTPricer
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

def _check_formula(S, K, r, vola, T):
    q = r*K
    call_pricer = FDMPricer(S, K, r, 0, vola, T, q, phi=+1)
    call_price = call_pricer.calc()
    print(f"Call Price, q = {q} = {call_price:.3f}")
    put_pricer = FDMPricer(S, K, r, 0, vola, T, 0, phi=-1)
    put_pricer.is_american = True
    put_price = put_pricer.calc()
    print(f"Put Price, = {put_price:.3f}")
    check = put_price + S - call_price - K
    print(f"Check = {check:.5f}")
    print("Stop = ", call_pricer.stop)
    print("Ex = ", put_pricer.ex_bound)
    plt.ion()  # Interaktive Plot-Anzeige einschalten
    fig, ax = plt.subplots(figsize=(8, 6))  # Figur und Achse erstellen

    ax.plot( call_pricer.stop, label=f'stop')  # Linie hinzufügen
    ax.legend()  # Legende aktualisieren
    plt.draw()  # Zeichne den aktuellen Plot
    plt.pause(0.1)
    ax.plot(put_pricer.ex_bound, label=f'ex')  # Linie hinzufügen
    ax.legend()  # Legende aktualisieren
    plt.draw()  # Zeichne den aktuellen Plot
    plt.pause(0.1)

# Beispiel-Nutzung
if __name__ == "__main__":
    S = 96
    K = 100
    r = 0.1
    d = 0.04
    vola = 0.2
    T = 1
    phi = +1
    plot_boundaries = False
    plot_discrete_boundaries = False
    check_formula = True

    if check_formula:
        _check_formula(S, K, r, vola, T)

    if plot_boundaries:
        # Plot initialisieren
        plt.ion()  # Interaktive Plot-Anzeige einschalten
        fig, ax = plt.subplots(figsize=(8, 6))  # Figur und Achse erstellen

    print("BS price = ", call(S, K, r, d, vola, T))

    for q in [1, 3, 8]:
        print("q = ", q)
        p = FDMPricer(S, K, r, d, vola, T, q=q, phi=phi)
        price = p.calc()
        print("FDM price = ", price)

        lctpricer = LCTPricer(S, K, r, d, vola, T, q, phi)
        value = lctpricer.value()
        print("LCT price = ", value)

        if plot_boundaries:
            t = np.linspace(0.001, 0.999, 1000)
            sb = [lctpricer.stop_bound(T-t[i]) for i in range(len(t))]

            ax.plot(t, sb, label=f'q = {q}')  # Linie hinzufügen
            ax.legend()  # Legende aktualisieren
            plt.draw()  # Zeichne den aktuellen Plot
            plt.pause(0.1)

        if plot_discrete_boundaries:
            values = {}
            for n in range(3, 10):
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