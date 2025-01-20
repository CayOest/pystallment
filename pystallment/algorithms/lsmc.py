import numpy as np
import numpy.random

from pystallment.option import AmericanOption

class LSMCPricer:
    def __init__(self, option, num_paths = 10000, fit = 'hermite'):
        self.option = option
        self.num_paths = int(num_paths)
        self.time_steps = int(self.option.T*320)
        self.plot = False
        self.is_american = isinstance(self.option, AmericanOption)
        self.fit = fit
        self.seed = None

    def _generate_paths(self):
        rng = numpy.random.default_rng(self.seed)

        self.paths = np.zeros((self.num_paths, self.time_steps + 1))
        self.dt = self.option.T / self.time_steps

        for i in range(int(self.num_paths/2)):
            self.paths[i, 0] = self.option.S
            self.paths[self.num_paths-i-1, 0] = self.option.S
            for t in range(1, self.time_steps+1):
                e = rng.normal()
                self.paths[i, t] =  self.paths[i][t-1]*np.exp((self.option.r-self.option.d - 0.5*self.option.vola**2)*self.dt + self.option.vola*e*np.sqrt(self.dt))
                e = -e
                self.paths[self.num_paths-i-1, t] = self.paths[self.num_paths-i-1][t - 1] * np.exp((self.option.r - self.option.d - 0.5 * self.option.vola ** 2) * self.dt + self.option.vola * e * np.sqrt(
                    self.dt))

    def price(self):
        self._generate_paths()
        payoffs = self.option.payoff(self.paths)

        q = 0
        if hasattr(self.option, "q"):
            q = self.option.q

        V = payoffs[:, -1]
        stop_times = np.ones(self.num_paths) * self.time_steps
        for t in range(self.time_steps, 0, -1):
            itm = payoffs[:, t] > 0
            _df = np.exp(-self.option.r * self.dt * (stop_times - t))
            y = _df * V - q / self.option.r * (1 - _df)

            if self.is_american:
                S_itm = self.paths[itm, t]
                y_itm = y[itm]
                if len(S_itm) > 0:
                    if self.fit == 'hermite':
                        fitted = np.polynomial.Hermite.fit(S_itm, y_itm, 4)
                        continuation_value = fitted(S_itm)

                    if self.fit == 'poly':
                        regression = np.polyfit(S_itm, y_itm, 2)
                        continuation_value = np.polyval(regression, S_itm)

                    # Entscheidung: Ausüben oder Fortführen
                    exercise = payoffs[itm, t] > continuation_value
                    stop_times[itm] = np.where(exercise, t, stop_times[itm])
                    V[itm] = np.where(exercise, payoffs[itm, t], V[itm])

            if q != 0:
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
                    stop = continuation_value < 0
                    stop_times[oom] = np.where(stop, t, stop_times[oom])
                    V[oom] = np.where(stop, 0, V[oom])

        # Diskontierter Erwartungswert am Anfang
        _df = np.exp(-self.option.r*self.dt*stop_times)
        y = _df*V - q/self.option.r*(1-_df)
        option_price = np.mean(y)
        return option_price
