import numpy as np

# Thomas-Algorithmus für tridiagonale Matrizen
def thomas_algorithm(a, b, c, d):
    """
    Thomas-Algorithmus zur Lösung von Ax = d,
    wobei A eine tridiagonale Matrix ist.
    a: Unterdiagonale (n-1 Elemente)
    b: Hauptdiagonale (n Elemente)
    c: Oberdiagonale (n-1 Elemente)
    d: Rechte Seite (n Elemente)
    """
    n = len(d)
    c_ = np.zeros(n - 1)
    d_ = np.zeros(n)

    # Vorwärtselimination
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * c_[i - 1]
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / denom
    d_[n - 1] = (d[n - 1] - a[n - 2] * d_[n - 2]) / (b[n - 1] - a[n - 2] * c_[n - 2])

    # Rückwärtssubstitution
    x = np.zeros(n)
    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x

class FDMPricer:
    def __init__(self, S, K, r, d, vola, T, q = 0, phi=1):
        self.spot = S
        self.K = K
        self.r = r
        self.d = d
        self.vola = vola
        self.T = T
        self.space_steps = 10000
        self.time_steps = 1000
        self.q = q
        self.phi = phi
        self.is_american = False

    def _calc(self):
        # Parameter
        self.stop = np.zeros(self.time_steps + 1)
        self.ex_bound = np.zeros(self.time_steps + 1)
        S_max = max(3 * self.spot, 3 * self.K)  # Maximale Preisgrenze dynamisch anpassen
        # Diskretisierung
        delta_S = S_max / self.space_steps  # Schrittweite im Aktienkurs
        delta_t = self.T / self.time_steps  # Zeitschrittweite
        S = np.linspace(0, S_max, self.space_steps + 1)  # Preisgitter
        self.stop[self.time_steps] = self.K
        self.ex_bound[self.time_steps] = self.K

        # Payoff der amerikanischen Put-Option bei Fälligkeit
        payoff = np.maximum(self.phi*(S-self.K), 0)

        # Matrix A-Koeffizienten gemäß der vollständigen Diskretisierung
        a = np.zeros(self.space_steps - 1)
        b = np.zeros(self.space_steps - 1)
        c = np.zeros(self.space_steps - 1)
        for j in range(1, self.space_steps):
            S_j = S[j]
            a[j - 1] = (-0.5 * self.vola ** 2 * S_j ** 2 / delta_S ** 2 + (self.r-self.d) * S_j / (2 * delta_S)) * delta_t if j > 1 else 0
            b[j - 1] = 1 + self.vola ** 2 * S_j ** 2 / delta_S ** 2 * delta_t + self.r * delta_t
            c[j - 1] = (-0.5 * self.vola ** 2 * S_j ** 2 / delta_S ** 2 - (self.r-self.d) * S_j / (2 * delta_S)) * delta_t if j < self.space_steps - 1 else 0

        # Rückwärtsinduktion
        V = payoff.copy()
        for n in range(self.time_steps - 1, -1, -1):  # Zeitrückwärts iterieren
            V_ = V[1:self.space_steps] - self.q * delta_t
            V_inner = thomas_algorithm(a[1:], b, c[:-1], V_)  # Lösen mit Thomas-Algorithmus
            # Randbedingungen setzen
            if self.phi == +1:
                V[0] = 0  # Linke Randbedingung
                V[-1] = S_max  # Rechte Randbedingung
            else:
                V[0] = self.K
                V[-1] = 0

            # Frühzeitige Ausübungsbedingung berücksichtigen
            if self.is_american:
                for j in range(0, self.space_steps-1):
                    if V_inner[j] < self.phi*(S[j] - self.K):
                        self.ex_bound[n] = S[j]
                        V_inner[j] = self.phi*( S[j] - self.K)

            V[1:self.space_steps] = V_inner
            for j in range(self.space_steps+1):
                if V[j] < 0:
                    if self.phi == 1:
                        self.stop[n] = max(self.stop[n], S[j])
                    else:
                        if self.stop[n] == 0:
                            self.stop[n] = S[j]
                    V[j] = 0

        # Interpolation anpassen, um den exakten Spotpreis zu treffen
        option_price = np.interp(self.spot, S, V)
        return option_price

    def calc(self):
        return self._calc()
