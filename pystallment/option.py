import numbers
import numpy as np

class Option:
    def __init__(self, S, K, r, d, vola, T, phi):
        """
        Initialize an Option object.

        :param S: Spot price of the underlying asset.
        :param K: Strike price of the option.
        :param r: Risk-free interest rate (annualized).
        :param d: Dividend yield (annualized).
        :param vola: Volatility of the underlying asset (annualized).
        :param T: Time to maturity in years.
        :param phi: Option type indicator (1 for Call, -1 for Put).
        """
        self.S = S
        self.K = K
        self.r = r
        self.d = d
        self.vola = vola
        self.T = T
        self.phi = phi

    @property
    def spot(self):
        """
        Returns the spot price of the option.

        :return: Spot price.
        """
        return self.S

    @property
    def strike(self):
        """
        Returns the strike price of the option.

        :return: Strike price.
        """
        return self.K

    @property
    def riskfree_rate(self):
        """
        Returns the risk-free interest rate of the option.

        :return: Risk-free rate.
        """
        return self.r

    @property
    def dividend_yield(self):
        """
        Returns the dividend yield of the option.

        :return: Dividend yield.
        """
        return self.d

    @property
    def volatility(self):
        """
        Returns the volatility of the underlying asset.

        :return: Volatility.
        """
        return self.vola

    @property
    def maturity(self):
        """
        Returns the time to maturity of the option in years.

        :return: Time to maturity.
        """
        return self.T

    def __repr__(self):
        """
        Returns a detailed string representation of the Option object for debugging purposes.

        :return: String representation of the Option.
        """
        return (f"Option(S={self.S}, K={self.K}, r={self.r}, d={self.d}, "
                f"vola={self.vola}, T={self.T}, phi={self.phi})")

    def __str__(self):
        """
        Returns a user-friendly description of the option details.

        Displays the spot price, strike price, risk-free rate, dividend yield,
        volatility, time to maturity, and option type (Call or Put).

        :return: String representation of the Option details.
        """
        option_type = "Call" if self.phi == 1 else "Put"
        return (f"Option Details:\n"
                f"  Spot Price (S): {self.S}\n"
                f"  Strike Price (K): {self.K}\n"
                f"  Risk-free Rate (r): {self.r}\n"
                f"  Dividend Yield (d): {self.d}\n"
                f"  Volatility (vola): {self.vola}\n"
                f"  Time to Maturity (T): {self.T}\n"
                f"  Option Type (phi): {option_type}")

    def payoff(self, x):
        """
        Computes the payoff of the option given a specific price.

        :param x: Price(s) at which the option payoff is evaluated.
        :return: Payoff value(s).
        """
        if hasattr(self.K, "__get_item__"):
            X = self.K[-1]
        else:
            X = self.K
        return np.maximum(self.phi * (x - X), 0)

class ContinuousInstallmentOption(Option):
    def __init__(self, S, K, r, d, vola, T, q, phi):
        """
        Initialize a ContinuousInstallmentOption object.

        :param S: Spot price of the underlying asset.
        :param K: Strike price of the option.
        :param r: Risk-free interest rate (annualized).
        :param d: Dividend yield (annualized).
        :param vola: Volatility of the underlying asset (annualized).
        :param T: Time to maturity in years.
        :param q: Installment rate (continuous).
        :param phi: Option type indicator (1 for Call, -1 for Put).
        """
        super().__init__(S, K, r, d, vola, T, phi)
        self.q = q

    @property
    def installment_rate(self):
        """
        Returns the continuous installment rate of the option.

        :return: Installment rate.
        """
        return self.q

class DiscreteInstallmentOption(Option):
    def __init__(self, S, r, d, vola, t, q, phi):
        """
        Initialize a DiscreteInstallmentOption object.

        :param S: Spot price of the underlying asset.
        :param r: Risk-free interest rate (annualized).
        :param d: Dividend yield (annualized).
        :param vola: Volatility of the underlying asset (annualized).
        :param t: Exercise dates of the option.
        :param q: Installment rates at each exercise date.
        :param phi: Option type indicator (1 for Call, -1 for Put).
        """
        super().__init__(S, q, r, d, vola, t[-1], phi)
        self.q = q
        self.t = t
        self.K = self.q

    @property
    def exercise_dates(self):
        """
        Returns the exercise dates of the option.

        :return: Exercise dates.
        """
        return self.t

    @property
    def strikes(self):
        """
        Returns the strike prices of the option.

        :return: Strike prices.
        """
        return self.K

def continuous_to_discrete(option, n):
    """
    Converts a ContinuousInstallmentOption into a DiscreteInstallmentOption.

    :param option: The option to convert.
    :param n: Number of discrete time intervals.
    :return: A DiscreteInstallmentOption instance with discrete installments.
    """
    if not isinstance(option, ContinuousInstallmentOption):
        raise TypeError("option must be of type ContinuousInstallmentOption")

    dt = option.maturity / n
    t = np.linspace(dt, option.maturity, n)
    q = np.ones(n) * option.installment_rate / option.riskfree_rate * (1 - np.exp(-option.riskfree_rate * dt))
    q[-1] = option.K
    return DiscreteInstallmentOption(option.S, option.r, option.d, option.vola, t, q, option.phi)

class BermudaOption(Option):
    def __init__(self, S, r, d, vola, t, K_, phi):
        """
        Initialize a BermudaOption object.

        :param S: Spot price of the underlying asset.
        :param r: Risk-free interest rate (annualized).
        :param d: Dividend yield (annualized).
        :param vola: Volatility of the underlying asset (annualized).
        :param t: Exercise dates of the option.
        :param K_: Strike price(s) of the option.
        :param phi: Option type indicator (1 for Call, -1 for Put).
        """
        if isinstance(K_, numbers.Number):
            K = np.ones(len(t)) * K_
        else:
            K = K_
        super().__init__(S, K[-1], r, d, vola, t[-1], phi)
        self.t = t
        self.K = K

class AmericanOption(Option):
    def __init__(self, S, K, r, d, vola, T, phi):
        """
        Initialize an AmericanOption object.

        :param S: Spot price of the underlying asset.
        :param K: Strike price of the option.
        :param r: Risk-free interest rate (annualized).
        :param d: Dividend yield (annualized).
        :param vola: Volatility of the underlying asset (annualized).
        :param T: Time to maturity in years.
        :param phi: Option type indicator (1 for Call, -1 for Put).
        """
        super().__init__(S, K, r, d, vola, T, phi)

class AmericanContinuousInstallmentOption(AmericanOption):
    def __init__(self, S, K, r, d, vola, T, q, phi):
        """
        Initialize an AmericanContinuousInstallmentOption object.

        :param S: Spot price of the underlying asset.
        :param K: Strike price of the option.
        :param r: Risk-free interest rate (annualized).
        :param d: Dividend yield (annualized).
        :param vola: Volatility of the underlying asset (annualized).
        :param T: Time to maturity in years.
        :param q: Continuous installment rate.
        :param phi: Option type indicator (1 for Call, -1 for Put).
        """
        super().__init__(S, K, r, d, vola, T, phi)
        self.q = q
