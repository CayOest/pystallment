import numpy as np
import numpy.random
import time

from pystallment.option import AmericanOption

class LSMCPricer:
    """
    LSMCPricer prices vanilla/installment options using antithetic paths.
    It extrapolates continuation value by Hermite polynomials 
    """
    def __init__(self, option, num_paths = 50000, fit = 'hermite'):
        self.option = option
        self.num_paths = int(num_paths)
        self.time_steps = int(self.option.T*320)
        self.fit = fit
        self.seed = None
        self._is_american = isinstance(self.option, AmericanOption)

    def _generate_paths(self):
        rng = numpy.random.default_rng(self.seed)

        self.paths = np.zeros((self.num_paths, self.time_steps + 1))
        self.dt = self.option.T / self.time_steps

        alpha = (self.option.r-self.option.d - 0.5*self.option.vola**2)*self.dt
        beta = self.option.vola*np.sqrt(self.dt)

        self.paths[:, 0] = self.option.S

        n = int(self.num_paths/2)
        E = rng.normal(size=(n, self.time_steps))
        E = np.cumsum(E, axis = 1)
        O = np.ones((n, self.time_steps))
        O = np.cumsum(O, axis = 1)
        self.paths[:n, 1:] = self.option.S * np.exp( alpha*O + beta*E )
        self.paths[n:, 1:] = self.option.S * np.exp(alpha * O - beta * E)

    def price(self):
        self._generate_paths()
        payoffs = self.option.payoff(self.paths)

        q = 0
        if hasattr(self.option, "installment_rate"):
            q = self.option.q

        V = payoffs[:, -1]
        stop_times = np.ones(self.num_paths) * self.time_steps
        for t in range(self.time_steps, 0, -1):
            itm = payoffs[:, t] > 0
            _df = np.exp(-self.option.r * self.dt * (stop_times - t))
            y = _df * V - q / self.option.r * (1 - _df)

            if self._is_american:
                S_itm = self.paths[itm, t]
                y_itm = y[itm]
                if len(S_itm) > 0:
                    if self.fit == 'hermite':
                        fitted = np.polynomial.Hermite.fit(S_itm, y_itm, 4)
                        continuation_value = fitted(S_itm)
                    elif self.fit == 'poly':
                        regression = np.polyfit(S_itm, y_itm, 2)
                        continuation_value = np.polyval(regression, S_itm)
                    else:
                        raise TypeError(f"fit method {self.fit} not supported.")

                    # Entscheidung: Ausüben oder Fortführen
                    exercise = payoffs[itm, t] > continuation_value
                    stop_times[itm] = np.where(exercise, t, stop_times[itm])
                    V[itm] = np.where(exercise, payoffs[itm, t], V[itm])

            if np.abs(q) > 1e-12:
                oom = ~itm
                S_oom = self.paths[oom, t]
                y_oom = y[oom]
                if len(S_oom) > 0:
                    if self.fit == 'hermite':
                        fitted = np.polynomial.Hermite.fit(S_oom, y_oom, 4)
                        continuation_value = fitted(S_oom)

                    if self.fit == 'poly':
                        regression = np.polyfit(S_oom, y_oom, 3)
                        continuation_value = np.polyval(regression, S_oom)

                    # Entscheidung: Fortführen oder Stoppen der Ratenzahlung
                    stop = continuation_value <= 0
                    stop_times[oom] = np.where(stop, t, stop_times[oom])
                    V[oom] = np.where(stop, 0, V[oom])

        # Diskontierter Erwartungswert am Anfang
        _df = np.exp(-self.option.r*self.dt*stop_times)
        y = _df*V - q/self.option.r*(1-_df)
        option_price = np.mean(y)
        return option_price
