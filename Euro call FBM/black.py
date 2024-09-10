from common_imports import np, norm

class BlackScholesModel:
    def __init__(self, S, K, r, sigma, t, T):
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.t = t
        self.T = T
        self.DT = self._time_to_maturity()  # Cache time to maturity to avoid recalculating

    def _time_to_maturity(self):
        """Calculate time to maturity (T - t)."""
        return self.T - self.t

    def _d1(self):
        """Calculate d1 for the Black-Scholes model."""
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.DT) / (self.sigma * np.sqrt(self.DT))

    def _d2(self):
        """Calculate d2 for the Black-Scholes model."""
        return self._d1() - self.sigma * np.sqrt(self.DT)

    def price(self):
        """Calculate the option price using the Black-Scholes model."""
        d1 = self._d1()
        d2 = self._d2()
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.DT) * norm.cdf(d2)
