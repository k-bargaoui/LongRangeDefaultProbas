from common_imports import gamma, np, norm, LinearRegression, plt, optimize, pi
from calibrationModels import CalibrationsModels

class Rostek(CalibrationsModels):
    """
    Rostek model class for calibrating default probability and asset volatility.
    
    Inherits from CalibrationsModels and uses the Rostek model for pricing equity
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
        self.H0 = 0.5  # Initial H value

    # Compute ro(H) function
    def ro_h(self, H):
        if H != 0.5:
            return (np.sin(pi * (H - 0.5)) / (pi * (H - 0.5))) * (gamma(1.5 - H) ** 2) / gamma(2 - 2 * H)
        return (gamma(1.5 - H) ** 2) / gamma(2 - 2 * H)

    # d1 and d2 from Rostek model
    def d1(self, x, sigma_A, t, T, H, mu):
        roH = self.ro_h(H)
        num = np.log(x / self.company_debt) + mu * (T - t) + 0.5 * roH * sigma_A ** 2 * (T - t) ** (2 * H)
        denom = np.sqrt(roH) * sigma_A * (T - t) ** H
        return num / denom

    def d2(self, x, sigma_A, t, T, H, mu):
        roH = self.ro_h(H)
        return self.d1(x, sigma_A, t, T, H, mu) - np.sqrt(roH) * sigma_A * (T - t) ** H

    # Inverse Black-Scholes formula using d1 and d2
    def inversed_formula(self, x, t, T, H, equity_value, sigma_A):
        d1_term = x * norm.cdf(self.d1(x, sigma_A, t, T, H, self.rf))
        d2_term = self.company_debt * np.exp(-self.rf * (T - t)) * norm.cdf(self.d2(x, sigma_A, t, T, H, self.rf))
        return d1_term - d2_term - equity_value

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

    # Function to calculate c_H(t, delta, H)
    def c_H(self, t, delta, H):
        return ((t + delta) ** (2 * H + 1) - t ** (2 * H + 1) - delta ** (2 * H + 1)) / ((2 * H + 1) * t * H)

    def calibrate(self):
        sigma_A_former = 0
        H = self.H0
        H_former = 0
        sigma_E = np.std(np.diff(np.log(self.company_market_cap))) * np.sqrt(self.frequency[0])
        sigma_A = sigma_E

        n_iter = 1
        while np.abs(sigma_A - sigma_A_former) > self.epsilon or np.abs(H - H_former) > self.epsilon:
            asset_values = {}
            n = len(self.company_market_cap)
            for f in self.frequency:
                fasset_values = []
                for day in range(0, n, n // f):
                    t = day / n
                    equity_value = self.company_market_cap.iloc[day]
                    asset_value = optimize.newton(
                        self.inversed_formula, self.company_debt, 
                        args=(t, t + self.T, H, equity_value, sigma_A), 
                        maxiter=100
                    )
                    fasset_values.append(asset_value)
                asset_values[f] = fasset_values
            # Update variance and mean based on frequency
            Var = [np.var(np.diff(np.log(asset_values[f]))) for f in self.frequency]
            Mean = [np.mean(np.diff(np.log(asset_values[f]))) for f in self.frequency]

            n_iter += 1
            H_former = H
            sigma_A, sigma_A_former, H = self.update_values_regression(Var, sigma_A, n_iter, n)

        t = int(n / 252)
        mu = (1 / t) * np.log(asset_values[self.frequency[0]][-1] / asset_values[self.frequency[0]][0]) + \
             (sigma_A ** 2) / 2 * (t ** (2 * H - 1))

        distance_to_default = self.d2(asset_values[self.frequency[0]][-1], sigma_A, t, t + self.T, H, mu)
        default_probability = (1 - norm.cdf(distance_to_default)) * 100

        return dict({"sigma": sigma_A,
                    "distance_to_default":distance_to_default,
                    "default_probability":default_probability,
                    "mu":mu, "H":H, "sigma_A_former":sigma_A_former})
