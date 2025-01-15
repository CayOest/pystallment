import numpy as np
from scipy.stats import norm

def d1(S, K, r, d, vola, tau):
    d1 = np.log(S / K) + (r - d + vola ** 2 / 2) * tau
    d1 /= vola * np.sqrt(tau)
    return d1

def d2(S, K, r, d, vola, tau):
    d2 = d1(S, K, r, d, vola, tau) - vola * np.sqrt(tau)
    return d2

def option_value(S, K, r, d, vola, tau, phi):
    d1_ = d1(S, K, r, d, vola, tau)
    d2_ = d2(S, K, r, d, vola, tau)
    return phi * S * np.exp(-d * tau) * norm.cdf(phi * d1_) - phi * K * np.exp(-r * tau) * norm.cdf(phi * d2_)

def call(S, K, r, d, vola, tau):
    return option_value(S, K, r, d, vola, tau, +1)

def put(S, K, r, d, vola, tau):
    return option_value(S, K, r, d, vola, tau, -1)
