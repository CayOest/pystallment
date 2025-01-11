import numpy as np
import math

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

class ContinuousInstallmentOptionPricer:
    def __init__(self, S, K, r, d, vola, T, q, phi = 1):
        self.S = S
        self.K = K
        self.r = r
        self.d = d
        self.vola = vola
        self.T = T
        self.q = q
        self.num_steps = 10
        self.phi = phi

    def _theta(self, l):
        poly = [0.5 * self.vola ** 2, self.r - self.d - 0.5 * self.vola ** 2, -(l + self.r)]
        theta = np.roots(poly)
        if theta[0] < 0:
            theta = np.flip(theta)

        return theta

    def _lct_stop(self, l):
        theta = self._theta(l)
        a = 2*(l + self.d)*self.q*self.phi
        b = l*(1-theta[1])*self.K*self.vola**2
        return (a/b)**(1/theta[0])*self.K

    def _lct_value_van(self, l):
        theta = self._theta(l)
        def xi(i, l):
            val = self.K * l / (theta[0]-theta[1])/(l+self.d)
            val *= (1 - (self.r-self.d)/(l+self.r)*theta[1-i])
            val *= (self.S/self.K)**theta[i]
            return val

        if (self.phi*self.S < self.phi*self.K):
            return xi(0, l)
        else:
            return xi(1, l) + self.phi* l*self.S/(l+self.d) - l*self.K/(l+self.r)

    def _lct_value(self, l):
        stop = self._lct_stop(l)
        theta = self._theta(self, l)
        if (self.S > stop):
            val = self._lct_value_van(l)
            if self.q > 0:
                val += self.q*theta[0] * (self.S / stop)**theta[1] / (l + self.r) / (theta[0]-theta[1])
                val -= self.q/(l + self.r)
        else:
            val = 0

        return val

    def stop_bound(self, t):
        return gaver_lct(self._lct_stop, t, self.num_steps)

    def value(self):
        return gaver_lct(self._lct_value_van, self.T, self.num_steps)

