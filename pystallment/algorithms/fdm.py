import numpy as np

def thomas_algorithm(a, b, c, d):
    """
    Thomas-Algorithm for solving Ax = d,
    a: lower diagonal (n-1 elements)
    b: main diagonal (n elements)
    c: upper diagonal (n-1 elements)
    d: right hand side (n Elements)
    """
    n = len(d)
    c_ = np.zeros(n - 1)
    d_ = np.zeros(n)

    # Vorwärtselimination
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * c_[i - 1]
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / denom
    d_[n - 1] = (d[n - 1] - a[n - 2] * d_[n - 2]) / (b[n - 1] - a[n - 2] * c_[n - 2])

    # Rückwärtssubstitution
    x = np.zeros(n)
    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x

class FDMPricer:
    def __init__(self, option):
        self.option = option
        self.space_steps = 1000
        self.time_steps = int(1600*self.option.T)

    def _calc(self):
        # Parameter
        self.is_american = False
        self.stop = np.zeros(self.time_steps + 1)
        self.ex_bound = np.zeros(self.time_steps + 1)
        S_max = max(3 * self.option.S, 3 * self.option.K)

        delta_S = S_max / self.space_steps  # step size in space
        delta_t = self.option.T / self.time_steps  # step size in time
        S = np.linspace(0, S_max, self.space_steps + 1)  # space grid
        self.stop[self.time_steps] = self.option.K
        self.ex_bound[self.time_steps] = self.option.K

        payoff = np.maximum(self.option.phi*(S-self.option.K), 0)

        a = np.zeros(self.space_steps -1 )
        b = np.zeros(self.space_steps - 1)
        c = np.zeros(self.space_steps - 1)
        for j in range(1, self.space_steps):
            S_j = S[j]
            a[j - 1] = (-0.5 * self.option.vola ** 2 * S_j ** 2 / delta_S ** 2 + (self.option.r-self.option.d) * S_j / (2 * delta_S)) * delta_t if j > 1 else 0
            b[j - 1] = 1 + self.option.vola ** 2 * S_j ** 2 / delta_S ** 2 * delta_t + self.option.r * delta_t
            c[j - 1] = (-0.5 * self.option.vola ** 2 * S_j ** 2 / delta_S ** 2 - (self.option.r-self.option.d) * S_j / (2 * delta_S)) * delta_t if j < self.space_steps - 1 else 0

        q = 0
        if hasattr(self.option, "q"):
            q = self.option.q

        V = payoff.copy()
        for n in range(self.time_steps - 1, -1, -1):
            V_ = V[1:self.space_steps] - q * delta_t
            V_inner = thomas_algorithm(a[1:], b, c[:self.space_steps-1], V_)

            # boundary conditions
            if self.option.phi == +1:
                V[0] = 0
                V[-1] = S_max
            else:
                V[0] = self.option.K
                V[-1] = 0

            if self.is_american:
                # todo: only works for phi = +1, i. e. call
                for j in range(0, self.space_steps-1):
                    if V_inner[j] < self.option.phi*(S[j] - self.option.K):
                        V_inner[j] = self.option.phi*( S[j] - self.option.K)
                        self.ex_bound[n] = S[j]

            V[1:self.space_steps] = V_inner
            for j in range(self.space_steps+1):
                if V[j] < 0:
                    if self.option.phi == 1:
                        self.stop[n] = max(self.stop[n], S[j])
                    else:
                        if self.stop[n] == 0:
                            self.stop[n] = S[j]
                    V[j] = 0

        option_price = np.interp(self.option.S, S, V)
        return option_price

    def price(self):
        return self._calc()
