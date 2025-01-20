import numpy as np
import math

"""
This module calculates the continuous installment call with the method introduced by Kimura, "Valuing continuous installment
options". 
"""

def gaver_lct(f, t, num_steps = 10):
    alpha = np.log(2) / t
    n = num_steps
    N = 2 * n
    A = [f(alpha * k) for k in range(1, N + 1)]
    B = np.zeros(N)
    G = np.zeros(n)
    for j in range(1, n + 1):
        for m in range(j, N - j + 1):
            B[m - 1] = (1 + m / j) * A[m - 1] - m / j * A[m]
        for i in range(j, N - j + 1):
            A[i - 1] = B[i - 1]
        G[j - 1] = B[j - 1]

    wt = np.zeros(n)
    for k in range(1, n + 1):
        wt[k - 1] = (-1) ** (n - k) * k ** n / math.factorial(k) / math.factorial(n - k)
    val = np.dot(wt, G)
    return val

class LCTPricer:
    def __init__(self, option, num_steps = 4):
        self.option = option
        self.num_steps = num_steps

    def _theta(self, l):
        poly = [0.5 * self.option.vola ** 2, self.option.r - self.option.d - 0.5 * self.option.vola ** 2, -(l + self.option.r)]
        theta = np.roots(poly)
        if theta[0] < 0:
            theta = np.flip(theta)

        return theta

    def _lct_stop(self, l):
        theta = self._theta(l)
        a = 2*(l + self.option.d)*self.option.q
        if self.option.phi==1:
            b = l*(1-theta[1])*self.option.K*self.option.vola**2
            return (a/b)**(1/theta[0])*self.option.K
        else:
            b = l*(theta[0]-1)*self.option.K*self.option.vola**2
            return (a / b) ** (1 / theta[1]) * self.option.K

    def _lct_value_van(self, l):
        theta = self._theta(l)
        def xi(i, l):
            val = self.option.K * l / (theta[0]-theta[1])/(l+self.option.d)
            val *= (1 - (self.option.r-self.option.d)/(l+self.option.r)*theta[1-i])
            val *= (self.option.S/self.option.K)**theta[i]
            return val

        if self.option.phi == 1:
            if self.option.S < self.option.K:
                return xi(0, l)
            else:
                return xi(1, l) + self.option.phi*(l*self.option.S/(l+self.option.d) - l*self.option.K/(l+self.option.r))
        else:
            if self.option.S < self.option.K:
                return xi(0, l) + self.option.phi*(l*self.option.S/(l+self.option.d) - l*self.option.K/(l+self.option.r))
            else:
                return xi(1, l)

    def _lct_value(self, l):
        stop = self._lct_stop(l)
        theta = self._theta(l)
        if self.option.phi*self.option.S > self.option.phi*stop:
            val = self._lct_value_van(l)
            if self.option.q > 0:
                val -= self.option.q / (l + self.option.r)
                if self.option.phi == 1:
                    val += self.option.q*theta[0] * (self.option.S / stop)**theta[1] / (l + self.option.r) / (theta[0]-theta[1])
                else:
                    val -= self.option.q*theta[1] * (self.option.S / stop)**theta[0] / (l + self.option.r) / (theta[0]-theta[1])
        else:
            val = 0

        return val

    def stop_bound(self, t):
        return gaver_lct(self._lct_stop, t, self.num_steps)

    def price(self):
        return gaver_lct(self._lct_value, self.option.T, self.num_steps)

    def vanilla_value(self):
        return gaver_lct(self._lct_value_van, self.option.T, self.num_steps)

