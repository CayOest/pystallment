import numbers

import numpy as np

class Option:
    def __init__(self, S, K, r, d, vola, T, phi):
        self.S = S
        self.K = K
        self.r = r
        self.d = d
        self.vola = vola
        self.T = T
        self.phi = phi

    def __repr__(self):
        return (f"Option(S={self.S}, K={self.K}, r={self.r}, d={self.d}, "
                f"vola={self.vola}, T={self.T}, phi={self.phi})")

    def __str__(self):
        option_type = "Call" if self.phi == 1 else "Put"
        return (f"Option Details:\n"
                f"  Spot Price (S): {self.S}\n"
                f"  Strike Price (K): {self.K}\n"
                f"  Risk-free Rate (r): {self.r}\n"
                f"  Dividend Yield (d): {self.d}\n"
                f"  Volatility (vola): {self.vola}\n"
                f"  Time to Maturity (T): {self.T}\n"
                f"  Option Type (phi): {option_type}")

    def payoff(self, x):
        if hasattr(self.K, "__get_item__"):
            X = self.K[-1]
        else:
            X = self.K
        return np.maximum(self.phi*(x - X), 0)

class ContinuousInstallmentOption(Option):
    def __init__(self, S, K, r, d, vola, T, q, phi):
        super().__init__(S, K, r, d, vola, T, phi)
        self.q = q

class DiscreteInstallmentOption(Option):
    def __init__(self, S, r, d, vola, t, q, phi):
        super().__init__(S, q, r, d, vola, t[-1], phi)
        self.q = q
        self.t = t
        self.K = self.q

def make_discrete_installment_call(S, r, d, vola, t, q):
    if hasattr(t, "__get_item__"):
        if not hasattr(q, "__get_item__"):
            raise TypeError("t is a list, but q is not.")
        else:
            return DiscreteInstallmentOption(S, r, d, vola, t, q, +1)

def continuous_to_discrete(option, n):
    if not isinstance(option, ContinuousInstallmentOption):
        raise TypeError("option must be of type ContinuousInstallmentOption")

    dt = option.T/n
    t = np.linspace(dt, option.T, n)
    q = np.ones(n)*option.q/option.r*(1-np.exp(-option.r*dt))
    q[-1] = option.K
    return DiscreteInstallmentOption(option.S, option.r, option.d, option.vola, t, q, option.phi)

class BermudaOption(Option):
    def __init__(self, S, r, d, vola, t, K_, phi):
        if isinstance(K_, numbers.Number):
            K = np.ones(len(t))*K_
        else:
            K = K_
        super().__init__(S, K[-1], r, d, vola, t[-1], phi)
        self.t = t
        self.K = K

class AmericanOption(Option):
    def __init__(self, S, K, r, d, vola, T, phi):
        super().__init__(S, K, r, d, vola, T, phi)

class AmericanContinuousInstallmentOption(AmericanOption):
    def __init__(self, S, K, r, d, vola, T, q, phi):
        super().__init__(S, K, r, d, vola, T, phi)
        self.q = q