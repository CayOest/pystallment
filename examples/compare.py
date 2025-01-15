import numpy as np
from fdm_pricer import FDMPricer
from discrete_pricer import InstallmentCallPricer, BermudaPutPricer, RichardsonPricer

import matplotlib.pyplot as plt

def check_formula(S, K, r, vola, T, space_steps, time_steps, plot_boundaries=True):
    print(f"Continuous Check, M = {space_steps}, N = {time_steps}")
    # price installment call with rate r*K
    call_pricer = FDMPricer(S, K, r, 0, vola, T, q=r*K, phi=+1)
    call_pricer.space_steps = space_steps
    call_pricer.time_steps = time_steps
    call_price = call_pricer.calc()
    print(f"Call Price = {call_price:.3f}")

    # price American put
    put_pricer = FDMPricer(S, K, r, 0, vola, T, 0, phi=-1)
    put_pricer.is_american = True
    put_pricer.space_steps = space_steps
    put_pricer.time_steps = time_steps
    put_price = put_pricer.calc()
    print(f"Put Price = {put_price:.3f}")
    
    check = put_price + S - call_price - K
    print(f"Check = {check:.5f}")
    print(100 * '*')

    if plot_boundaries:
        t = np.linspace(0, T, time_steps+1)
        ax.plot( t, call_pricer.stop, label=f'stop, M={time_steps}, N={space_steps}', lw=5, color='red')  # Linie hinzufügen
        ax.plot(t, put_pricer.ex_bound, label=f'ex, M={time_steps}, N={space_steps}', lw=1, color='blue')  # Linie hinzufügen
        ax.legend()  # Legende aktualisieren
        plt.draw()  # Zeichne den aktuellen Plot
        plt.pause(0.1)

def check_discrete(S, K, r, vola, T, plot_boundaries=True, n=8):
    print("Discrete Check, n = ", n)
    q = K*(1-np.exp(-r*T/n))
    t = np.arange(T / n, T + T / n, T / n)
    q_ = np.ones(len(t)) * q
    q_[-1] = K

    call_pricer = InstallmentCallPricer(S, r, 0, vola, t, q_)
    call = call_pricer.price()
    print("Installment Price = ", call)

    K_ = np.zeros(len(t))
    K_[-1] = K
    for i in range(n - 1):
        K_[i] = 0
        for j in range(i, n):
            K_[i] += q_[j] * np.exp(-r * (t[j] - t[i]))

    put_pricer = BermudaPutPricer(S, r, 0, vola, t, K_)
    put = put_pricer.price()
    print("Put Price = ", put)

    check = put + S - call - np.sum(q_ * np.exp(-r * t))
    print(f"check = {check:.6e}")
    print(100 * '*')

    if plot_boundaries:
        ax.plot(t, call_pricer.stop, label=f'call, n = {n}', lw=5, color='pink')  # Linie hinzufügen
        ax.plot(t, put_pricer.stop, label=f'put, n = {n}', lw=1, color='cyan')  # Linie hinzufügen
        ax.legend()  # Legende aktualisieren
        plt.draw()  # Zeichne den aktuellen Plot
        plt.pause(0.1)

if __name__ == "__main__":
    #option params
    S = 105
    K = 100
    r = 0.02
    vola = 0.2
    T = 1

    # fdm params
    N = [3, 4]
    fdm_space_steps = [10**i for i in N]
    fdm_time_steps = 2000*T

    # discrete params
    n = [3, 5, 8]

    # what to test
    plot_boundaries = False
    test_discrete = False
    test_continuous = False
    test_richardson = True

    if test_richardson:
        pricer = RichardsonPricer(S, K, r, d=0.0, vola=vola, T=T, phi=+1, q=r*K)
        pricer.n =3
        value = pricer.calc()
        print("value = ", value)

    if plot_boundaries:
        plt.ion()  # Interaktive Plot-Anzeige einschalten
        fig, ax = plt.subplots(figsize=(8, 6))  # Figur und Achse erstellen

    if test_continuous:
        for ss in fdm_space_steps:
            check_formula(S, K, r, vola, T, ss, fdm_time_steps, plot_boundaries)

    if test_discrete:
        for n_ in n:
            check_discrete(S, K, r, vola, T, plot_boundaries, n_)

    if plot_boundaries:
        plt.ioff()
        plt.show()