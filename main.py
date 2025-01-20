import numpy as np

from discrete import InstallmentCallPricer, BermudaPutPricer, call, put
from lct_pricer import LCTPricer
from fdm import FDMPricer

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