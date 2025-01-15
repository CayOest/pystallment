

class Option:
    def __init__(self, S, K, r, d, vola, T, phi):
        self.S = S
        self.K = K
        self.r = r
        self.d = d
        self.vola = vola
        self.T = T
        self.phi = phi


class Installment(Option):
    def __init__(self, S, K, r, d, vola, T, q, phi, style = 'discrete'):
        super().__init__(S, K, r, d, vola, T, phi)
        self.q = q
        self.style = style

class BermudaOption(Option):
    def __init__(self, S, r, d, vola, T, K, phi):
        super().__init__(S, K, r, d, vola, T, phi)
        self.K = K

