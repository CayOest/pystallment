import numpy as np
import matplotlib.pyplot as plt

from pystallment.algorithms.binomial import BinomialPricer
from pystallment.algorithms.fdm import FDMPricer
from pystallment.algorithms.discrete import InstallmentCallPricer, BermudaPricer
import pystallment.option as opt


def check_formula(S, K, r, vola, T, space_steps, time_steps, plot_boundaries=True):
    print(f"Continuous Check, M = {space_steps}, N = {time_steps}")
    # price installment call with rate r*K
    option = opt.ContinuousInstallmentOption(S, K, r, 0, vola, T, r*K, phi=+1)
    call_pricer = FDMPricer(option)
    call_pricer.space_steps = space_steps
    call_pricer.time_steps = time_steps
    call_price = call_pricer.price()
    print(f"Call Price = {call_price:.3f}")

    # price American put
    option = opt.AmericanOption(S, K, r, 0, vola, T, phi=-1)
    put_pricer = FDMPricer(option)
    put_pricer.space_steps = space_steps
    put_pricer.time_steps = time_steps
    put_price = put_pricer.price()
    print(f"Put Price = {put_price:.3f}")
    
    check = put_price + S - call_price - K
    print(f"Check = {check:.5f}")
    print(100 * '*')

    if plot_boundaries:
        t = np.linspace(0, T, time_steps+1)
        ax.plot( t, call_pricer.stop, label=f'stop, M={time_steps}, N={space_steps}', lw=5, color='red')  # Linie hinzufügen
        ax.plot(t, put_pricer.ex, label=f'ex, M={time_steps}, N={space_steps}', lw=1, color='blue')  # Linie hinzufügen
        ax.legend()  # Legende aktualisieren
        plt.draw()  # Zeichne den aktuellen Plot
        plt.pause(0.1)

def check_discrete(S, K, r, vola, T, plot_boundaries=True, n=8):
    print("Discrete Check, n = ", n)
    q = K*(1-np.exp(-r*T/n))
    t = np.arange(T / n, T + T / n, T / n)
    q_ = np.ones(len(t)) * q
    q_[-1] = K

    option = opt.DiscreteInstallmentOption(S, r, 0, vola, t, q_, +1 )
    call_pricer = InstallmentCallPricer(option)
    call = call_pricer.price()
    print("Installment Price = ", call)

    K_ = np.zeros(len(t))
    K_[-1] = K
    for i in range(n - 1):
        K_[i] = 0
        for j in range(i, n):
            K_[i] += q_[j] * np.exp(-r * (t[j] - t[i]))

    option = opt.BermudaOption(S, r, 0, vola, t, K_, -1)
    put_pricer = BermudaPricer(option)
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
    plot_boundaries = True
    test_discrete = True
    test_continuous = True

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