from common_imports import np, norm, gamma, pi

class RostekModel:
    def __init__(self, S, K, r, sigma, t, T, H):
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.t = t
        self.T = T
        self.H = H
        self.roH = self._ro_h()  # Cache roH to avoid recalculating

    def _ro_h(self):
        """Calculate the roH factor used in Rostek model."""
        if self.H != 0.5:
            return (np.sin(pi * (self.H - 0.5)) / (pi * (self.H - 0.5))) * (gamma(1.5 - self.H) ** 2 / gamma(2 - 2 * self.H))
        return gamma(1.5 - self.H) ** 2 / gamma(2 - 2 * self.H)

    def _d1_rostek(self):
        """Calculate d1 for the Rostek model."""
        DT_H = (self.T - self.t) ** self.H
        return (np.log(self.S / self.K) + self.r * (self.T - self.t) + 0.5 * self.roH * self.sigma ** 2 * (self.T - self.t) ** (2 * self.H)) / (np.sqrt(self.roH) * self.sigma * DT_H)

    def _d2_rostek(self):
        """Calculate d2 for the Rostek model."""
        return self._d1_rostek() - np.sqrt(self.roH) * self.sigma * (self.T - self.t) ** self.H

    def price(self):
        """Calculate the option price using the Rostek model."""
        d1_term = self.S * norm.cdf(self._d1_rostek())
        d2_term = self.K * np.exp(-self.r * (self.T - self.t)) * norm.cdf(self._d2_rostek())
        return d1_term - d2_term

    def proba_default(self, VA, sigma_A, mu, company_debt):
        """Calculate the probability of default using the Rostek model."""
        DT = np.sqrt(self.T ** (2 * self.H) - self.t ** (2 * self.H))
        d1 = (np.log(VA / company_debt) + mu * (self.T - self.t) + 0.5 * sigma_A ** 2 * (self.T ** (2 * self.H) - self.t ** (2 * self.H))) / (sigma_A * DT)
        distance_to_default = d1 - np.sqrt(self.roH) * sigma_A * (self.T - self.t) ** self.H
        return (1 - norm.cdf(distance_to_default)) * 100
