import numpy as np

from pystallment.option import AmericanOption


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
    """
    Class FDMPricer calculates the price of a European/American vanilla/installment option.
    At the moment, only the implicit method is implemented.
    todo: Implement general Runge-Kutta
    """
    def __init__(self, option):
        self._option = option
        self.space_steps = 1000
        self.time_steps = int(1600*self._option.T)
        self._is_american = isinstance(self._option, AmericanOption)
        self._stop = None
        self._ex = None

    @property
    def stop(self):
        return self._stop

    @property
    def ex(self):
        return self._ex
        
    def _init_bounds(self):
        self._stop = np.zeros(self.time_steps + 1)
        self._ex = np.zeros(self.time_steps + 1)
        self._stop[self.time_steps] = self._option.K
        self._ex[self.time_steps] = self._option.K

    def _calc(self):
        # Parameter
        S_max = max(3 * self._option.S, 3 * self._option.K)

        delta_S = S_max / self.space_steps  # step size in space
        delta_t = self._option.T / self.time_steps  # step size in time
        S = np.linspace(0, S_max, self.space_steps + 1)  # space grid
        
        a = np.zeros(self.space_steps -1 )
        b = np.zeros(self.space_steps - 1)
        c = np.zeros(self.space_steps - 1)
        for j in range(1, self.space_steps):
            S_j = S[j]
            a[j - 1] = (-0.5 * self._option.vola ** 2 * S_j ** 2 / delta_S ** 2 + (self._option.r-self._option.d) * S_j / (2 * delta_S)) * delta_t if j > 1 else 0
            b[j - 1] = 1 + self._option.vola ** 2 * S_j ** 2 / delta_S ** 2 * delta_t + self._option.r * delta_t
            c[j - 1] = (-0.5 * self._option.vola ** 2 * S_j ** 2 / delta_S ** 2 - (self._option.r-self._option.d) * S_j / (2 * delta_S)) * delta_t if j < self.space_steps - 1 else 0

        q = 0
        if hasattr(self._option, "installment_rate"):
            q = self._option.installment_rate

        exercise_values = self._option.payoff(S)
        V = exercise_values

        for t in range(self.time_steps - 1, -1, -1):
            V_ = V[1:self.space_steps] - q * delta_t
            V_inner = thomas_algorithm(a[1:], b, c[:self.space_steps-1], V_)

            # boundary conditions
            if self._option.phi == +1:
                V[0] = 0
                V[-1] = S_max
            else:
                V[0] = self._option.K
                V[-1] = 0

            V[1:self.space_steps] = V_inner

            # adjust for exercise events
            if self._is_american:
                exercise = V < exercise_values
                V = np.where(exercise, exercise_values, V)
                idx = -1 if self._option.phi == -1 else 0
                ex_index = np.where(exercise)[0][idx] if np.any(exercise) else None
                self._ex[t] = S[ex_index] if ex_index is not None else self._ex[t+1]

            # adjust for stop events
            stop = V < 0
            V = np.where(stop, 0, V)

        return np.interp(self._option.S, S, V)

    def price(self):
        self._init_bounds()
        return self._calc()
