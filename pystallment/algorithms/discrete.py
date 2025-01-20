from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

from pystallment import black_scholes as bs, option as opt


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

class DiscretePricer(ABC):
    def __init__(self, option):
        self.option = option

    @abstractmethod
    def _calc(self, S, t_, K_, S_, t_k):
        """Abstrakte Methode für die spezifische Berechnung. Muss in abgeleiteten Klassen implementiert werden."""
        pass

    def _find_bracket(self, f, guess):
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
        stops = [self.option.K[-1]]
        for k in range(len(self.option.t) - 2, -1, -1):
            t_k = self.option.t[k]
            K_next = self.option.K[k + 1:]
            t_next = self.option.t[k + 1:]
            stop = self._calc_stop(t_next, K_next, stops, t_k, self.option.K[k])
            stops.insert(0, stop)
        return stops

    @abstractmethod
    def _calc_stop(self, t_, K_, S_, t_k, K_k):
        pass

    def price(self):
        self.stop = self._generate_stops()
        return self._calc(self.option.S, self.option.t, self.option.K, self.stop, 0)


class InstallmentCallPricer(DiscretePricer):
    def _calc(self, S, t_, K_, S_, t_k):
        R_ = _gen_cov(t_)
        dplus = [bs.d1(S, S_[i], self.option.r, self.option.d, self.option.vola, t_[i] - t_k) for i in range(len(t_))]
        dminus = [bs.d2_from_d1(dplus[i], self.option.vola, t_[i] - t_k) for i in range(len(t_)) ]
        val = S * np.exp(-self.option.d * (t_[-1] - t_k)) * _mvn_cdf(R_, dplus)
        for i in range(len(t_)):
            val -= np.exp(-self.option.r * (t_[i] - t_k)) * K_[i] * _mvn_cdf(R_[:i+1, :i+1], dminus[:i+1])
        return val

    def _calc_stop(self, t_, K_, S_, t_k, K_k):
        def f(x):
            return self._calc(x, t_, K_, S_, t_k) - K_k
        return self._find_stop(f, S_)


class BermudaPricer(DiscretePricer):
    def __init__(self, bermuda_option):
        super().__init__(bermuda_option)

    def _calc(self, S, t_, K_, S_, t_k):
        R_ = _gen_cov(t_)
        dplus = [bs.d1(S, S_[i], self.option.r, self.option.d, self.option.vola, t_[i] - t_k) for i in range(len(t_))]
        dminus = [bs.d2_from_d1(dplus[i], self.option.vola, t_[i] - t_k) for i in range(len(t_))]
        val = S * np.exp(-self.option.d * (t_[-1] - t_k)) * (_mvn_cdf(R_, dplus) - 1)
        for i in range(len(t_)):
            dminus_ = dminus[:i+1].copy()
            dminus_[-1] *= -1
            R__ = _twist(R_[:i+1, :i+1])
            val += np.exp(-self.option.r * (t_[i] - t_k)) * K_[i] * _mvn_cdf(R__, dminus_)
        return val

    def _calc_stop(self, t_, K_, S_, t_k, K_k):
        def f(x):
            return self._calc(x, t_, K_, S_, t_k) - K_k + x

        return self._find_stop(f, S_)

def option_value(option):
    return bs.option_value(option.S, option.K, option.r, option.d, option.vola, option.T, option.phi)

class ExtrapolationPricer:
    def __init__(self, option, n, style = 'inst', interpol = 'poly'):
        self.option = option
        self.n = n
        self.style = style
        self.interpol = interpol
        self.plot = False

    def _weight(self, i):
        val = (-1) ** (self.n - i)
        for j in range(1, i+1):
            val *= 1.0*i/j
        for j in range(1, self.n-i+1):
            val *= 1.0*i/j
        return val

    def calc(self):
        if self.style != 'inst':
            raise TypeError("Style != inst not supported.")

        x = []
        values = []
        x.append(1)
        bs_option = opt.Option(self.option.S, self.option.K, self.option.r, self.option.d, self.option.vola, self.option.T, self.option.phi )
        values.append(option_value(bs_option))
        for k in range(2, self.n+1):
            dopt = opt.continuous_to_discrete(self.option, k)
            p = InstallmentCallPricer(dopt)
            x.insert(0, 1/k)
            values.insert(0, p.price())

        if self.interpol == 'poly':
            coeffs = np.polyfit(x, values, deg=3)
            # Polynom aus Koeffizienten erstellen
            polynom = np.poly1d(coeffs)

            if self.plot:
                # Werte für das Polynom berechnen
                x_interp = np.linspace(0, 1, 100)  # Feinere x-Werte
                y_interp = polynom(x_interp)

                # Visualisierung
                plt.scatter(x, values, label="Datenpunkte", color="red")
                plt.plot(x_interp, y_interp, label=f"Polynom 2. Grades", linestyle="--")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend()
                plt.show()

            return polynom(0)

        elif self.interpol == 'rich':
            values = np.flip(values)
            val = 0
            for k in range(1, self.n+1):
                val += self._weight(k)*values[k-1]

            return val