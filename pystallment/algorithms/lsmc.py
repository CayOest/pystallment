import numpy as np

from pystallment.option import AmericanOption


class LSMCPricer:
    def __init__(self, option, num_paths = 10000, fit = 'hermite'):
        self.option = option
        self.num_paths = int(num_paths)
        self.time_steps = int(self.option.T*320)
        self.plot = False
        self.is_american = isinstance(self.option, AmericanOption)
        self.fit = fit

    def _generate_paths(self):
        self.paths = np.zeros((self.num_paths, self.time_steps + 1))
        self.dt = self.option.T / self.time_steps

        for i in range(int(self.num_paths/2)):
            self.paths[i, 0] = self.option.S
            self.paths[self.num_paths-i-1, 0] = self.option.S
            for t in range(1, self.time_steps+1):
                e = np.random.normal()
                self.paths[i, t] =  self.paths[i][t-1]*np.exp((self.option.r-self.option.d - 0.5*self.option.vola**2)*self.dt + self.option.vola*e*np.sqrt(self.dt))
                e = -e
                self.paths[self.num_paths-i-1, t] = self.paths[self.num_paths-i-1][t - 1] * np.exp((self.option.r - self.option.d - 0.5 * self.option.vola ** 2) * self.dt + self.option.vola * e * np.sqrt(
                    self.dt))

    def calc(self):
        self._generate_paths()
        payoffs = self.option.payoff(self.paths)
        df = np.exp(-self.option.r*self.dt)

        qi = 0
        if hasattr(self.option, "q"):
            qi = self.option.q / self.option.r * (1 - df)

        # Rückwärtsinduktion mit Regression
        V = payoffs[:, -1]  # Endzeitwerte (Payoff bei Endfälligkeit)
        stop_times = np.ones(self.num_paths) * self.time_steps
        for t in range(self.time_steps, 0, -1):
            if self.is_american:
                # Identifiziere In-the-Money-Pfade
                in_the_money = payoffs[:, t] > 0
                S_itm = self.paths[in_the_money, t]
                V_itm = V[in_the_money] * df
                # Regression: Schätze den Fortführungswert
                if len(S_itm) > 0:
                    if self.fit == 'hermite':
                        fitted = np.polynomial.Hermite.fit(S_itm, V_itm, 3)
                        continuation_value = fitted(S_itm)

                    if self.fit == 'poly':
                        regression = np.polyfit(S_itm, V_itm, 2)
                        continuation_value = np.polyval(regression, S_itm)

                    # Entscheidung: Ausüben oder Fortführen
                    exercise = payoffs[in_the_money, t] > continuation_value
                    V[in_the_money] = np.where(exercise, payoffs[in_the_money, t], V_itm)
                V[~in_the_money] *= df
            if hasattr(self.option, "q"):
                _df = np.exp(-self.option.r*self.dt*(stop_times-t))
                y = _df*V - self.option.q/self.option.r*(1-_df)
                oom = payoffs[:,t] == 0
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
        option_price = np.mean(V) * df
        if hasattr(self.option, "q"):
            _df = np.exp(-self.option.r*self.dt*stop_times)
            y = _df*V - self.option.q/self.option.r*(1-_df)
            option_price = np.mean(y)
        return option_price
