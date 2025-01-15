import numpy as np

class BinomialPricer:
    def __init__(self, S, K, r, d, vola, T, q, phi):
        self.S = S
        self.K = K
        self.r = r
        self.d = d
        self.vola = vola
        self.T = T
        self.q = q
        self.num_steps = 10000
        self.phi = phi
        self.is_american = True

    def _calc(self):
        dt = self.T/self.num_steps

        qi = self.q/self.r*(1-np.exp(-self.r*dt))
        self.stop_bound = np.zeros(self.num_steps+1)
        self.ex_bound = np.zeros(self.num_steps + 1)

        up = np.exp(self.vola*np.sqrt(dt))
        do = 1/up

        correction = 1
        if correction == 1:
            # correction 1
            alfa = np.exp((self.r - self.d) * dt)
            up *= alfa
            do *= alfa
            p = (alfa - do) / (up - do)
        else:
            # correction 2
            alfa = np.exp(- self.d * dt)
            up *= alfa
            do *= alfa
            p = (alfa*np.exp(self.r*dt) - do) / (up - do)

        S_ = np.ones(self.num_steps+1)
        n = self.num_steps
        for i in range(self.num_steps+1):
            S_[i] = up**(n-i) * do**i * self.S

        V = np.maximum(self.phi*(S_-self.K), 0)

        self.stop_bound[-1] = self.K
        self.ex_bound[-1] = self.K

        for step in range(self.num_steps-1, -1, -1):
            for i in range(step+1):
                S_[i] = up**(step-i) * do**i * self.S

            for i in range(step + 1):
                V[i] = np.exp(-self.r*dt)*(p*V[i] + (1-p)*V[i+1] - self.q * dt)
                if V[i] <= 0: # stop event
                    V[i] = 0
                    if self.phi == -1:
                        self.stop_bound[step] = S_[i]
                    else:
                        if self.stop_bound[step] == 0:
                            self.stop_bound[step] = S_[i]

                if self.is_american:
                    exercise = self.phi*(S_[i]-self.K)
                    if V[i] <= exercise: #exercise event
                        V[i] = exercise
                        if self.ex_bound[step] == 0:
                            self.ex_bound[step] = S_[i]

        return V[0]

    def calc(self):
        return self._calc()