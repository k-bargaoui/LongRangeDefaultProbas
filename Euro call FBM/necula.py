from common_imports import np, norm

class NeculaModel:
    def __init__(self, S, K, r, sigma, t, T, H):
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.t = t
        self.T = T
        self.H = H
        self.T_H_diff = self._T_H_diff()  # Cache T_H_diff to avoid recalculating

    def _T_H_diff(self):
        """Calculate the time scaling factor T^(2H) - t^(2H)."""
        return self.T**(2 * self.H) - self.t**(2 * self.H)

    def _d1(self):
        """Calculate d1 for the Necula model."""
        return (np.log(self.S / self.K) + self.r * (self.T - self.t) + 0.5 * self.sigma ** 2 * self.T_H_diff) / (self.sigma * np.sqrt(self.T_H_diff))

    def _d2(self):
        """Calculate d2 for the Necula model."""
        return self._d1() - self.sigma * np.sqrt(self.T_H_diff)

    def price(self):
        """Calculate the option price using the Necula model."""
        d1 = self._d1()
        d2 = self._d2()
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * (self.T - self.t)) * norm.cdf(d2)

    def proba_default(self, VA, sigma_A, mu, company_debt):
        """Calculate the probability of default using the Necula model."""
        time_factor = np.sqrt(self.T_H_diff)
        distance_to_default = (np.log(VA / company_debt) + mu * (self.T - self.t) - 0.5 * sigma_A ** 2 * self.T_H_diff) / (sigma_A * time_factor)
        return (1 - norm.cdf(distance_to_default)) * 100
