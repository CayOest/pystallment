import numbers
import numpy as np

from pystallment.option import AmericanOption

class BinomialPricer:
    def __init__(self, option, num_steps = 1000, factor_adjustment='r-d'):
        """
        Construct a BinomialPricer for American option and installment options
        :param option: option of type AmericanOption or ContinuousInstallmentOption
        :param num_steps: depth of the binomial tree
        :param factor_adjustment: the up and down steps in the tree are adjusted ('r-d' or '-d')
        """
        self.num_steps = num_steps
        self.factor_adjustment = factor_adjustment
        self._option = option
        self._is_american = isinstance(option, AmericanOption)

    def _init_bounds(self):
        self.stop_bound = np.zeros(self.num_steps + 1)
        self.ex_bound = np.zeros(self.num_steps + 1)
        self.stop_bound[-1] = self._option.K
        self.ex_bound[-1] = self._option.K

    def _get_up_down_p(self, dt):
        up = np.exp(self._option.vola * np.sqrt(dt))
        do = 1 / up

        if self.factor_adjustment == 'r-d':
            alfa = np.exp((self._option.r - self._option.d) * dt)
            up *= alfa
            do *= alfa
            p = (alfa - do) / (up - do)
        elif self.factor_adjustment == '-d':
            alfa = np.exp(- self._option.d * dt)
            up *= alfa
            do *= alfa
            p = (alfa * np.exp(self._option.r * dt) - do) / (up - do)
        else:
            pass

        return (up, do, p)

    def _generate_all_prices(self, up, do):
        prices = np.zeros((self.num_steps+1, self.num_steps+1))
        prices[0, 0] = self._option.S
        alpha = do/up
        alphas = np.cumprod( np.ones(self.num_steps)*alpha )
        for step in range(1, self.num_steps+1):
            prices[step, 0] = prices[step-1, 0]*up
            prices[step, 1:(step+1)] = prices[step, 0]*alphas[:step]

        return prices

    def _check_stop_event(self, step, V, S):
        if self._is_american:
            exercise_value = self._option.payoff(S)
            exercise = V <= exercise_value
            V = np.where(exercise, exercise_value, V)
            idx = -1 if self._option.phi == -1 else 0
            ex_index = np.where(exercise)[0][idx] if np.any(exercise) else None
            self.ex_bound[step] = S[ex_index] if ex_index is not None else self.ex_bound[step + 1]

        stop = V < 0
        V = np.where(stop, 0, V)
        idx = -1 if self._option.phi == +1 else 0
        stop_index = np.where(stop)[0][idx] if np.any(stop) else None
        self.stop_bound[step] = S[stop_index] if stop_index is not None else self.stop_bound[step + 1]
        return V

    def _get_ttm(self):
        if hasattr(self._option, "T"):
            if isinstance(self._option.T, numbers.Number):
                return self._option.T
            elif hasattr(self._option.T, "__get_item__"):
                return self._option.T[-1]
        elif hasattr(self._option, "t"):
            if isinstance(self._option.t, numbers.Number):
                return self._option.t
            elif hasattr(self._option.t, "__get_item__"):
                return self._option.t[-1]

        raise TypeError("Could not determine time to maturity from option")

    def _get_installment_rate(self):
        q = 0
        if hasattr(self._option, "q"):
            if isinstance(self._option.q, numbers.Number):
                q = self._option.q
        elif hasattr(self._option, "K"):
            if isinstance(self._option.K, list):
                q = self._option.K[0]
        return q

    def _iterate_tree(self):
        # discount factor
        df = np.exp(-self._option.r*self._dt)

        # get optional installment rate
        qi = self._get_installment_rate()/self._option.r*(1-df)

        V = self._option.payoff(self._prices[self.num_steps, :])
        for step in range(self.num_steps-1, -1, -1):
            V_ = df*(self._p*V[:(step+1)] + (1-self._p)*V[1:]) - qi
            V_ = self._check_stop_event(step, V_, self._prices[step, :(step+1)])
            V = V_

        return V[0]

    def price(self):
        self._init_bounds()
        # get parameters
        T = self._get_ttm()
        self._dt = T / self.num_steps
        # build price tree
        up, do, self._p = self._get_up_down_p(self._dt)
        self._prices = self._generate_all_prices(up, do)

        return self._iterate_tree()