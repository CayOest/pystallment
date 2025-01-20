import numpy as np
from scipy.stats import norm

def d1(S, K, r, d, vola, tau):
    """
    Compute the upper limit d1 for the cumulative normal distribution

    :param S: spot
    :param K: strike
    :param r: riskfree rate
    :param d: dividend yield
    :param vola: volatility
    :param tau: time to maturity
    :return: the parameter
    """
    d1 = np.log(S / K) + (r - d + vola ** 2 / 2) * tau
    d1 /= vola * np.sqrt(tau)
    return d1

def d2_from_d1(d1, vola, tau):
    return d1 - vola * np.sqrt(tau)

def d2(S, K, r, d, vola, tau):
    d2_from_d1( d1(S, K, r, d, vola, tau), vola, np.sqrt(tau))
    return d2

def option_value(S, K, r, d, vola, tau, phi):
    """
    Compute the analytical price of a European standard option in the Black-Scholes model

    :param S: spot
    :param K: strike
    :param r: riskfree rate
    :param d: dividiend yield
    :param vola: volatility
    :param tau: time to maturity
    :param phi: option type (+1 for call, -1 for put
    :return: the option price
    """
    d1_ = d1(S, K, r, d, vola, tau)
    d2_ = d2_from_d1(d1_, vola, tau)
    return phi * S * np.exp(-d * tau) * norm.cdf(phi * d1_) - phi * K * np.exp(-r * tau) * norm.cdf(phi * d2_)

def call(S, K, r, d, vola, tau):
    """
    Compute the analytical price of a European call option in the Black-Scholes model

    :param S: spot
    :param K: strike
    :param r: riskfree rate
    :param d: dividiend yield
    :param vola: volatility
    :param tau: time to maturity
    :return: the call price
    """
    return option_value(S, K, r, d, vola, tau, +1)

def put(S, K, r, d, vola, tau):
    """
       Compute the analytical price of a European put option in the Black-Scholes model

       :param S: spot
       :param K: strike
       :param r: riskfree rate
       :param d: dividiend yield
       :param vola: volatility
       :param tau: time to maturity
       :return: the put price
       """
    return option_value(S, K, r, d, vola, tau, -1)
