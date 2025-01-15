import numbers

import numpy as np

from option import AmericanOption

class BinomialPricer:
    def __init__(self, option, num_steps = 1000, factor_adjustment='r-d'):
        # factor_adjustment can be 'r-d' or '-d' or no correction
        self.option = option
        self.num_steps = num_steps
        self.is_american = isinstance(option, AmericanOption)
        self.factor_adjustment = factor_adjustment

    def _init_bounds(self):
        self.stop_bound = np.zeros(self.num_steps + 1)
        self.ex_bound = np.zeros(self.num_steps + 1)
        self.stop_bound[-1] = self.option.K
        self.ex_bound[-1] = self.option.K

    def _get_up_down_p(self, dt):
        up = np.exp(self.option.vola * np.sqrt(dt))
        do = 1 / up

        if self.factor_adjustment == 'r-d':
            alfa = np.exp((self.option.r - self.option.d) * dt)
            up *= alfa
            do *= alfa
            p = (alfa - do) / (up - do)
        elif self.factor_adjustment == '-d':
            alfa = np.exp(- self.option.d * dt)
            up *= alfa
            do *= alfa
            p = (alfa * np.exp(self.option.r * dt) - do) / (up - do)
        else:
            pass

        return (up, do, p)

    def _generate_prices(self, k, up, do):
        prices = np.ones(k)
        for i in range(k):
            prices[i] = up ** (k - i - 1) * do ** i * self.option.S
        return prices
    
    def _check_stop_event(self, step, Vi, Si):
        if Vi <= 0:  # stop event
            Vi = 0
            if self.option.phi == -1:
                self.stop_bound[step] = Si
            else:
                if self.stop_bound[step] == 0:
                    self.stop_bound[step] = Si

        if self.is_american:
            exercise = self.option.payoff(Si)
            if Vi <= exercise:  # exercise event
                Vi = exercise
                if self.ex_bound[step] == 0:
                    self.ex_bound[step] = Si
        return Vi

    def _get_ttm(self):
        if hasattr(self.option, "T"):
            if isinstance(self.option.T, numbers.Number):
                return self.option.T
            elif hasattr(self.option.T, "__get_item__"):
                return self.option.T[-1]
        elif hasattr(self.option, "t"):
            print(self.option.t)
            if isinstance(self.option.t, numbers.Number):
                return self.option.t
            elif hasattr(self.option.t, "__get_item__"):
                return self.option.t[-1]

        raise TypeError("Could not determine time to maturity from option")

    def _get_installment_rate(self):
        q = 0
        if hasattr(self.option, "q"):
            if isinstance(self.option.q, float):
                q = self.option.q
        return q

    def _calc(self):
        # get parameters
        T = self._get_ttm()
        dt = T/self.num_steps
        up, do, p = self._get_up_down_p(dt)

        # build price tree
        prices = self._generate_prices(self.num_steps+1, up, do)
        V = self.option.payoff(prices)

        # get optional installment rate
        qdt = self._get_installment_rate()*dt

        # discount factor
        df = np.exp(-self.option.r*dt)

        for step in range(self.num_steps-1, -1, -1):
            prices = self._generate_prices(step+1, up, do)

            for i in range(step + 1):
                V[i] = df*(p*V[i] + (1-p)*V[i+1] - qdt)
                V[i] = self._check_stop_event(step, V[i], prices[i])

        return V[0]

    def calc(self):
        self._init_bounds()
        return self._calc()