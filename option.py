import numpy as np

class Option:
    def __init__(self, S, K, r, d, vola, t, phi):
        self.S = S
        self.K = K
        self.r = r
        self.d = d
        self.vola = vola
        self.t = t
        self.phi = phi

    def __repr__(self):
        return (f"Option(S={self.S}, K={self.K}, r={self.r}, d={self.d}, "
                f"vola={self.vola}, t={self.t}, phi={self.phi})")

    def __str__(self):
        option_type = "Call" if self.phi == 1 else "Put"
        return (f"Option Details:\n"
                f"  Spot Price (S): {self.S}\n"
                f"  Strike Price (K): {self.K}\n"
                f"  Risk-free Rate (r): {self.r}\n"
                f"  Dividend Yield (d): {self.d}\n"
                f"  Volatility (vola): {self.vola}\n"
                f"  Time to Maturity (t): {self.t}\n"
                f"  Option Type (phi): {option_type}")

    def payoff(self, x):
        return np.maximum(self.phi*(x - self.K), 0)

class InstallmentOption(Option):
    def __init__(self, S, K_, r, d, vola, T, q, phi):
        if isinstance(q, list):
            K = q + [K_]
        else:
            K = [q, K_]
        super().__init__(S, K, r, d, vola, T, phi)

def make_installment_call(S, K, r, d, vola, t, q):
    return InstallmentOption(S, K, r, d, vola, t, q, +1)

class BermudaOption(Option):
    def __init__(self, S, r, d, vola, t, K, phi):
        super().__init__(S, K, r, d, vola, t, phi)

def make_bermuda_put(S, r, d, vola, t, K):
    return BermudaOption(S, r, d, vola, t, K, -1)

class AmericanOption(Option):
    def __init__(self, S, K, r, d, vola, T, phi):
        super().__init__(S, K, r, d, vola, T, phi)