from common_imports import np, norm, pi, LinearRegression, plt, optimize
from calibrationModels import CalibrationsModels

class Necula(CalibrationsModels):
    """
    Necula model class for calibrating default probability and asset volatility.
    
    Inherits from CalibrationsModels and uses the Necula model for pricing equity
    as a call option on the firm's assets. It calibrates the asset volatility and 
    computes the default probability based on distance to default.
    
    Attributes:
        ticker (str): The stock ticker symbol.
        market_cap (pd.DataFrame): The company's market capitalization time series.
        debt (pd.DataFrame): The company's debt data.
        T (int): Time to maturity in years.
        frequency (int): Frequency of the data (default: 252, for daily data).
        rf (float): Risk-free rate (default: 0).
        epsilon (float): Convergence threshold for calibration (default: 1e-5).
        H0: a defaulted value of the Hurst parameter
    """
    def __init__(self, ticker, market_cap, debt, T, frequency=None, rf=0, epsilon=1e-5):
        if frequency is None:
            frequency = 252 // np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        super().__init__(ticker, market_cap, debt, T, frequency, rf, epsilon)
        self.H0 = 0.5  # Initial value of H

    # d1 and d2 from Necula
    def d1(self, x, sigma_A, t, T, H, mu):
        term1 = np.log(x / self.company_debt) + mu * (T - t)
        term2 = 0.5 * sigma_A ** 2 * (T ** (2 * H) - t ** (2 * H))
        denominator = sigma_A * np.sqrt(T ** (2 * H) - t ** (2 * H))
        return (term1 + term2) / denominator

    def d2(self, x, sigma_A, t, T, H, mu):
        term1 = np.log(x / self.company_debt) + mu * (T - t)
        term2 = -0.5 * sigma_A ** 2 * (T ** (2 * H) - t ** (2 * H))
        denominator = sigma_A * np.sqrt(T ** (2 * H) - t ** (2 * H))
        return (term1 + term2) / denominator

    # Inverse the Black-Scholes formula with Necula's expressions for d1 and d2
    def inversed_formula(self, x, t, T, H, equity_value, sigma_A):
        d1_val = self.d1(x, sigma_A, t, T, H, self.rf)
        d2_val = self.d2(x, sigma_A, t, T, H, self.rf)
        term1 = x * norm.cdf(d1_val)
        term2 = self.company_debt * np.exp(-self.rf * (T - t)) * norm.cdf(d2_val)
        return term1 - term2 - equity_value

    # Update values using regression to estimate sigma_A and H
    def update_values_regression(self, Var, sigma_A, iteration, n, plot=False):
        delta_t = np.arange(1, len(Var) + 1) / self.frequency
        log_delta_t = np.log(delta_t)
        log_var_tau = np.log(Var)

        model = LinearRegression().fit(log_delta_t.reshape(-1, 1), log_var_tau)
        slope, intercept = model.coef_[0], model.intercept_

        # Calculate H and update sigma_A
        H = slope / 2
        sigma_A_former = sigma_A
        sigma_A = np.exp(intercept / 2) * 252 ** H

        if plot:
            plt.scatter(log_delta_t, log_var_tau, label='Data')
            plt.plot(log_delta_t, model.predict(log_delta_t.reshape(-1, 1)), color='red', label='Regression')
            plt.xlabel('log(Δt)')
            plt.ylabel('log(Var(τ(Δt)))')
            plt.title(f"Regression at iteration {iteration}")
            plt.legend()
            plt.show()

        return sigma_A, sigma_A_former, H

    # Helper function for sigma estimate
    def sigma_estimate(self, asset_values, H):
        step = np.arange(1, 11)  # Sampling steps
        var_tau = [
            np.var(np.diff(np.log(asset_values[::s]))) for s in step
        ]
        sigma_A = [
            np.sqrt(var) * (252 / s) ** H for s, var in zip(step, var_tau)
        ]
        return np.mean(sigma_A)

    def calibrate(self):
        sigma_E = np.std(np.diff(np.log(self.company_market_cap))) * np.sqrt(self.frequency[0])
        sigma_A = sigma_E
        sigma_A_former = 0
        H = self.H0
        H_former = 0

        n_iter = 1
        while np.abs(sigma_A - sigma_A_former) > self.epsilon or np.abs(H - H_former) > self.epsilon:
            asset_values = {}
            for f in self.frequency:
                resampled_assets = []
                n = len(self.company_market_cap)
                for day in range(0, n, n // f):
                    t = day / n
                    equity_value = self.company_market_cap.iloc[day]
                    asset_value = optimize.newton(
                        self.inversed_formula, self.company_debt, 
                        args=(t, t + self.T, H, equity_value, sigma_A), 
                        maxiter=100
                    )
                    resampled_assets.append(asset_value)
                asset_values[f] = resampled_assets

            # Update variance and mean based on frequency
            Var = [np.var(np.diff(np.log(asset_values[f]))) for f in self.frequency]

            n_iter += 1
            H_former = H
            sigma_A, sigma_A_former, H = self.update_values_regression(Var, sigma_A, n_iter, n)

        t = len(self.company_market_cap) / 252
        mu = (1 / t) * np.log(asset_values[self.frequency[0]][-1] / asset_values[self.frequency[0]][0]) + \
             (sigma_A ** 2) / 2 * (t ** (2 * H - 1))

        distance_to_default = self.d2(asset_values[self.frequency[0]][-1], sigma_A, t, t + self.T, H, mu)
        default_probability = (1 - norm.cdf(distance_to_default)) * 100

        return dict({"sigma": sigma_A,
                    "distance_to_default":distance_to_default,
                    "default_probability":default_probability,
                    "mu":mu, "H":H, "sigma_A_former":sigma_A_former})
    
    