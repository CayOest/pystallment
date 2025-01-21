import numpy as np

from pystallment.option import AmericanOption
from scipy.linalg import solve_banded

def _solve_scipy(upper, main, lower, b):
    ab = np.array([upper, main, lower])
    x = solve_banded((1, 1), ab, b)
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
        self._delta_S = 0.0
        self._delta_t = 0.0

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
        
    def _get_matrix(self, S):
        a = np.zeros(self.space_steps - 1)
        b = np.zeros(self.space_steps - 1)
        c = np.zeros(self.space_steps - 1)
        alpha = -0.5*self._option.vola**2/self._delta_S**2
        beta = (self._option.r-self._option.d)/(2*self._delta_S)
        n = len(S)-1
        a[1:] = (alpha*(S[2:(len(S)-1)]**2) + beta*S[2:(len(S)-1)])*self._delta_t
        b = 1 + self._option.vola**2/self._delta_S**2 * self._delta_t * (S[1:n]**2) + self._option.r * self._delta_t
        c[:(len(c)-1)] = (alpha*(S[1:(n-1)]**2) -beta*(S[1:(n-1)]))*self._delta_t
        return (a, b, c)

    def _calc(self):
        # Parameter
        S_max = max(3 * self._option.S, 3 * self._option.K)

        self._delta_S = S_max / self.space_steps  # step size in space
        self._delta_t = self._option.T / self.time_steps  # step size in time
        S = np.linspace(0, S_max, self.space_steps + 1)  # space grid

        q = 0
        if hasattr(self._option, "installment_rate"):
            q = self._option.installment_rate

        exercise_values = self._option.payoff(S)
        V = exercise_values
        (a, b, c) = self._get_matrix(S)

        for t in range(self.time_steps - 1, -1, -1):
            V_ = V[1:self.space_steps] - q * self._delta_t
            a_ = np.append(a[1:], 0)
            c_ = np.append([0], c[:(len(c)-1)])
            V_inner = _solve_scipy(c_, b, a_, V_)

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
            idx = -1 if self._option.phi == +1 else 0
            stop_index = np.where(stop)[0][idx] if np.any(stop) else None
            self._stop[t] = S[stop_index] if stop_index is not None else self._stop[t + 1]

        return np.interp(self._option.S, S, V)

    def price(self):
        self._init_bounds()
        return self._calc()
