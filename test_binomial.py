from binomial_pricer import BinomialPricer
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    S = 105
    K = 100
    r = 0.02
    d = 0.00
    q = r*K
    vola = 0.2
    T = 1
    N = 10000
    plot_boundaries = True

    pricer1 = BinomialPricer(S, K, r, d, vola, T, q, +1)
    pricer1.is_american = False
    pricer1.num_steps = N
    price1 = pricer1.calc()
    print("Value = ", price1)
    print("Stop Bound = ", pricer1.stop_bound)

    pricer2 = BinomialPricer(S, K, r, d, vola, T, 0, -1)
    pricer2.is_american = True
    pricer2.num_steps = N
    price2 = pricer2.calc()
    print("Value = ", price2)
    print("Ex Bound = ", pricer2.ex_bound)
    check = price2 + S - price1 - K
    print("check = ", check)

    if plot_boundaries:
        plt.ion()  # Interaktive Plot-Anzeige einschalten
        fig, ax = plt.subplots(figsize=(8, 6))  # Figur und Achse erstellen
        t = np.linspace(0, T, N + 1)
        ax.plot(t, pricer1.stop_bound, label=f'stop, N={N}', lw=5,
                color='red')  # Linie hinzufügen
        ax.plot(t, pricer2.ex_bound, label=f'ex, N={N}', lw=1,
                color='blue')  # Linie hinzufügen

        ax.legend()  # Legende aktualisieren
        plt.draw()  # Zeichne den aktuellen Plot
        plt.ioff()
        plt.show()