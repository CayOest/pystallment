import numpy as np
import discrete_pricer
from scipy.stats import norm
from scipy.optimize import root_scalar

class BaronePricer:
    def __init__(self, S, K, r, d, vola, T, phi):
        self.S = S
        self.K = K
        self.r = r
        self.d = d
        self.vola = vola
        self.T = T
        self.phi = phi

    def _find_stop(self, f):
        bracket = [0.1 * self.K, 2.0 * self.K]
        res = root_scalar(
            f,
            method='brentq',
            bracket=bracket,
        )
        return res.root

    def _calc(self):
        M = 2*self.r/self.vola**2
        N = 2*(self.r-self.d)/self.vola**2
        K_ = 1-np.exp(-self.r*self.T)
        q1 = 0.5*( -(N-1) - np.sqrt((N-1)**2 + 4*M/K_))
        def f(x):
            eu_val = discrete_pricer.put(x, self.K, self.r, self.d, self.vola, self.T)
            val = self.K - x - eu_val
            d1 = discrete_pricer.d1d2(x, self.K, self.r, self.d, self.vola, self.T)[0]
            val += (1-np.exp(-self.r*self.T))*norm.cdf(-d1)*x/q1
            return val

        S_ = self._find_stop(f)
        if self.S <= S_:
            return self.K - self.S
        else:
            val = discrete_pricer.put(self.S, self.K, self.r, self.d, self.vola, self.T)
            d1 = discrete_pricer.d1d2(S_, self.K, self.r, self.d, self.vola, self.T)[0]
            A1 = -S_/q1*(1-np.exp(-self.d*self.T)*norm.cdf(-d1))
            val += A1*(self.S/S_)**q1
            return val

    def calc(self):
        return self._calc()