import math

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import root_scalar
from abc import ABC, abstractmethod

import black_scholes as bs

def _mvn_cdf(cov, y):
    if len(y) == 1:
        return norm.cdf(y[0], loc=0, scale=1)

    mean = np.zeros(len(y))
    mvn = multivariate_normal(mean=mean, cov=cov)
    return mvn.cdf(y)

def _gen_cov(t_):
    n = len(t_)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i <= j:
                cov[i, j] = np.sqrt(t_[i] / t_[j])
            else:
                cov[i, j] = np.sqrt(t_[j] / t_[i])
    return cov

def _find_stop(f, S_):
    xtol = 1e-24

    if (len(S_) == 1):
        bracket = [0.8 * S_[0], 1.2 * S_[0]]
    else:
        bracket = [0.9 * S_[0], 1.1 * S_[0]]
    res = root_scalar(
        f,
        method='brentq',
        bracket=bracket,
        xtol=xtol
    )
    return res.root

def _twist( R):
    R_ = R.copy()
    for j in range(len(R_)):
        R_[j, -1] *= -1
        R_[-1, j] *= -1
    return R_

class PricerBase(ABC):
    def __init__(self, S, r, d, vola, t, K):
        self.S = S
        self.r = r
        self.d = d
        self.vola = vola
        self.t = t
        self.K = K
        self.stop = None

    @abstractmethod
    def _calc(self, S, t_, K_, S_, t_k):
        """Abstrakte Methode für die spezifische Berechnung. Muss in abgeleiteten Klassen implementiert werden."""
        pass

    def _find_bracket(self, f, guess):
        print(f(guess))
        a = 0.1
        while( a < 1.0 and f( (1-a)*guess )*f( (1+a)*guess ) > 0):
            a += 0.1
        return [(1-a)*guess, (1+a)*guess]

    def _find_stop(self, f, S_):
        xtol = 1e-12
        bracket = self._find_bracket(f, S_[0])
        res = root_scalar(f, method='brentq', bracket=bracket, xtol=xtol)
        return res.root

    def _generate_stops(self):
        stops = [self.K[-1]]
        for k in range(len(self.t) - 2, -1, -1):
            t_k = self.t[k]
            K_next = self.K[k + 1:]
            t_next = self.t[k + 1:]
            stop = self._calc_stop(t_next, K_next, stops, t_k, self.K[k])
            stops.insert(0, stop)
        return stops

    @abstractmethod
    def _calc_stop(self, t_, K_, S_, t_k, K_k):
        pass

    def price(self):
        self.stop = self._generate_stops()
        return self._calc(self.S, self.t, self.K, self.stop, 0)


class InstallmentCallPricer(PricerBase):
    def _calc(self, S, t_, K_, S_, t_k):
        R_ = _gen_cov(t_)
        dplus = [bs.d1(S, S_[i], self.r, self.d, self.vola, t_[i] - t_k)[0] for i in range(len(t_))]
        dminus = [bs.d2(S, S_[i], self.r, self.d, self.vola, t_[i] - t_k)[1] for i in range(len(t_))]
        val = S * np.exp(-self.d * (t_[-1] - t_k)) * _mvn_cdf(R_, dplus)
        for i in range(len(t_)):
            val -= np.exp(-self.r * (t_[i] - t_k)) * K_[i] * _mvn_cdf(R_[:i+1, :i+1], dminus[:i+1])
        return val

    def _calc_stop(self, t_, K_, S_, t_k, K_k):
        def f(x):
            return self._calc(x, t_, K_, S_, t_k) - K_k
        return self._find_stop(f, S_)


class BermudaPutPricer(PricerBase):
    def _calc(self, S, t_, K_, S_, t_k):
        R_ = _gen_cov(t_)
        dplus = [bs.d1(S, S_[i], self.r, self.d, self.vola, t_[i] - t_k) for i in range(len(t_))]
        dminus = [bs.d2(S, S_[i], self.r, self.d, self.vola, t_[i] - t_k) for i in range(len(t_))]
        val = S * np.exp(-self.d * (t_[-1] - t_k)) * (_mvn_cdf(R_, dplus) - 1)
        for i in range(len(t_)):
            dminus_ = dminus[:i+1].copy()
            dminus_[-1] *= -1
            R__ = _twist(R_[:i+1, :i+1])
            val += np.exp(-self.r * (t_[i] - t_k)) * K_[i] * _mvn_cdf(R__, dminus_)
        return val

    def _calc_stop(self, t_, K_, S_, t_k, K_k):
        def f(x):
            return self._calc(x, t_, K_, S_, t_k) - K_k + x

        return self._find_stop(f, S_)

class RichardsonPricer():
    def __init__(self, S, K, r, d, vola, T, q, phi):
        self.S = S
        self.K = K
        self.r = r
        self.d = d
        self.vola = vola
        self.T = T
        self.q = q
        self.phi = phi
        self.n = 3

    def _weight(self, i):
        return (-1)**(self.n-i) * i**(self.n-1)/math.factorial(i-1)/math.factorial(self.n-i)

    def calc(self):
        value = self._weight(1) * bs.option_value(self.S, self.K, self.r, self.d, self.vola, self.T, self.phi)
        for i in range(2, self.n+1):
            dt = self.T/i
            t = np.linspace(dt, self.T, i)
            q_ = np.ones(len(t))*self.q/self.r*(1-np.exp(-self.r*dt))
            q_[-1] = self.K
            ip = InstallmentCallPricer(self.S, self.r, self.d, self.vola, t, q_)
            val = ip.price()
            value += self._weight(i) * val

        return value
