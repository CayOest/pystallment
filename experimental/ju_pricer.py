import numpy as np
from pystallment.algorithms.discrete import option_value
from scipy.stats import norm

class JUPricer:
    def __init__(self, S, K, r, d, vola, T, q, phi):
        self.S = S
        self.K = K
        self.r = r
        self.d = d
        self.vola = vola
        self.T = T
        self.phi = phi

    def _calc_bound(self):

        h = 1-np.exp(-self.r*self.T)
        alfa = 2*self.r/self.vola**2
        beta = 2*(self.r-self.d)/self.vola**2
        lh = 0.5*(-(beta-1)+self.phi*(np.sqrt((beta-1)**2 + 4*alfa/h)))


        def f(x):
            d1 = discrete_pricer.d1d2(x, self.K, self.r, self.d, self.vola, self.T)[0]
            val = self.phi * np.exp(-self.d*self.T)*norm.cdf(self.phi*d1(x))
            eu_val = option_value(x, self.K, self.r, self.d, self.vola, self.T, self.phi)
            val += (self.phi * (x - self.K)*lh-eu_val)/x
            val -= self.phi
            return val

        # S_ = solve(f)
        S_ = self.K
        if self.phi*(S_-self.S) <= 0:
            return self.phi*(self.S-self.K)
        else:
            eu_val = option_value(self.S, self.K, self.r, self.d, self.vola, self.T, self.phi)
            eu_val_ = option_value(S_, self.K, self.r, self.d, self.vola, self.T, self.phi)
            hAh = self.phi*(S_-self.K) - eu_val_
            b = (1-h)*alfa*l_h/(2*(2*lh+beta-1))
            c = (1-h)*alfa/(2*lh+beta-1)*(1/hAh*)
            val = eu_val + (hAh*(self.S/S_)**lh)/(1-X)

    def _calc(self):
        S_ = self._calc_bound()
        if self.phi*(S_-self.S) <= 0:
            return self.phi(self.S - self.K)


        val = option_value(self.S, self.K, self.r, self.d, self.vola, self.T, sleepelf.phi)


    def calc(self):
        return self._calc()