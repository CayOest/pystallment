import option as opt
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermvander

from option import AmericanOption


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

        for i in range(self.num_paths):
            self.paths[i, 0] = self.option.S
            for t in range(1, self.time_steps+1):
                e = np.random.normal()
                self.paths[i, t] =  self.paths[i][t-1]*np.exp((self.option.r-self.option.d - 0.5*self.option.vola**2)*self.dt + self.option.vola*e*np.sqrt(self.dt))

    def calc(self):
        self._generate_paths()
        payoffs = self.option.payoff(self.paths)

        df = np.exp(-self.option.r*self.dt)
#        qi = self.option.q/self.option.r(1-df)
        # Rückwärtsinduktion mit Regression
        V = payoffs[:, -1]  # Endzeitwerte (Payoff bei Endfälligkeit)
        for t in range(self.time_steps, 0, -1):
            # Identifiziere In-the-Money-Pfade
            in_the_money = payoffs[:, t] > 0
            S_itm = self.paths[in_the_money, t]
           # V = df*V - qi
            V_itm = V[in_the_money] * df
            # Regression: Schätze den Fortführungswert
            if len(S_itm) > 0:
                if self.fit == 'hermite':
                    # Designmatrix für Hermite-Polynome
                    hermite_matrix = hermvander(S_itm, 4)
                    # Regression: Löse das lineare Gleichungssystem
                    coeffs = np.linalg.lstsq(hermite_matrix, V_itm, rcond=None)[0]
                    # Berechnung des Fortführungswerts
                    continuation_value = hermite_matrix @ coeffs

                if self.fit == 'poly':
                    regression = np.polyfit(S_itm, V_itm, 2)  # Quadratische Regression
                    continuation_value = np.polyval(regression, S_itm)

                # Entscheidung: Ausüben oder Fortführen
                exercise = payoffs[in_the_money, t] > continuation_value
                V[in_the_money] = np.where(exercise, payoffs[in_the_money, t], V_itm)
            V[~in_the_money] *= df

        # Diskontierter Erwartungswert am Anfang
        option_price = np.mean(V) * df
        return option_price
