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


    def _calc(self):
        def payoff(x):
            return np.maximum(self.phi*(x-self.K), 0)

        dt = self.T/self.num_steps
        u = np.exp(self.vola*np.sqrt(dt))
        uu = u*u
        d = 1/u
        p = (np.exp((self.r-self.d)*dt) - d)/(u-d)
        S_ = np.zeros(self.num_steps+1)
        S_[0] = self.S*d**self.num_steps
        for i in range(1, self.num_steps+1):
            S_[i] = S_[i-1]*uu

        V = np.zeros(self.num_steps+1)
        V = [payoff(S_[i]) for i in range(self.num_steps+1)]
        Rinv = np.exp(-self.r*dt)

        self.ex_bound = np.zeros(self.num_steps)
        self.ex_bound[-1] = self.K
        for step in range(self.num_steps-1, -1, -1):
            for i in range(step):
                V[i] = (p * V[i+1] + (1-p)*V[i])*Rinv
                S_[i] = d*S_[i+1]
                if self.phi*(S_[i]-self.K) > V[i]: # exercise option
                    V[i] = self.phi*(S_[i]-self.K)
                    if self.phi == 1:
                        if self.ex_bound[step] == 0:
                            self.ex_bound[step] = S_[i]
                    else:
                        self.ex_bound[step] = max(self.ex_bound[step], S_[i])

                # if V[i] < 0: # stop option
                #     V[i] = 0

        return V[0]

    def calc(self):
        return self._calc()