from common_imports import np, norm, pi, LinearRegression, plt, optimize
from calibrationModels import CalibrationModel


class Necula(CalibrationModel):
    """
    A class representing the Necula model for calibrating financial models.

    Inherits from CalibrationsModels and is used for estimating asset volatility,
    the Hurst exponent, and default probabilities.

    Attributes:
        Inherits attributes from the CalibrationsModels class, including ticker, 
        market cap, debt, maturity, etc.
    """

    def __init__(self, ticker, market_cap, debt, T, frequency=None, rf=0, epsilon=1e-5):
        """
        Initialize the Necula model with company data and parameters.

        Args:
            ticker (str): Ticker symbol of the company.
            market_cap (pandas.DataFrame): Market capitalization data.
            debt (pandas.DataFrame): Debt data of the company.
            T (float): Time to maturity (in years).
            frequency (numpy.ndarray, optional): Data frequency (default is 
                252 divided by [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).
            rf (float, optional): Risk-free interest rate, default is 0.
            epsilon (float, optional): Tolerance for convergence, default is 1e-5.
        """
        if frequency is None:
            frequency = 252 // np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        super().__init__(ticker, market_cap, debt, T, frequency, rf, epsilon)
        self.H0 = 0.5  # Initial value for the Hurst exponent

    def d1(self, x, sigma_A, t, T, H, mu):
        """
        Calculate the d1 parameter of the Necula model.

        Args:
            x (float): Asset value.
            sigma_A (float): Volatility of the asset.
            t (float): Current time.
            T (float): Time to maturity.
            H (float): Hurst exponent.
            mu (float): Drift of the asset.

        Returns:
            float: The d1 parameter.
        """
        return (np.log(x / self.company_debt) + mu * (T - t) +
                0.5 * sigma_A ** 2 * (T ** (2 * H) - t ** (2 * H))) / (
                    sigma_A * np.sqrt(T ** (2 * H) - t ** (2 * H))
                )

    def d2(self, x, sigma_A, t, T, H, mu):
        """
        Calculate the d2 parameter of the Necula model.

        Args:
            x (float): Asset value.
            sigma_A (float): Volatility of the asset.
            t (float): Current time.
            T (float): Time to maturity.
            H (float): Hurst exponent.
            mu (float): Drift of the asset.

        Returns:
            float: The d2 parameter.
        """
        return (self.d1(x, sigma_A, t, T, H, mu) -
                sigma_A * np.sqrt(T ** (2 * H) - t ** (2 * H)))

    def inversed_formula(self, x, t, T, H, equity_value, sigma_A):
        """
        Calculate the inverse formula for asset values.

        Args:
            x (float): Asset value.
            t (float): Current time.
            T (float): Time to maturity.
            H (float): Hurst exponent.
            equity_value (float): Market equity value.
            sigma_A (float): Volatility of the asset.

        Returns:
            float: The difference between the predicted and actual equity values.
        """
        d1_term = x * norm.cdf(self.d1(x, sigma_A, t, T, H, self.rf))
        d2_term = (self.company_debt * np.exp(-self.rf * (T - t)) *
                   norm.cdf(self.d2(x, sigma_A, t, T, H, self.rf)))
        return d1_term - d2_term - equity_value

    def update_values_regression(self, Var, sigma_A, iteration, n, plot=False):
        """
        Update regression values to estimate the Hurst exponent (H) and asset volatility (sigma_A).

        Args:
            Var (list): Variance values.
            sigma_A (float): Asset volatility.
            iteration (int): Iteration number.
            n (int): Number of data points.
            plot (bool, optional): If True, plots the regression (default is False).

        Returns:
            tuple: Updated values of sigma_A, sigma_A_former, and H.
        """
        var_tau = np.array(Var)
        delta_t = n // self.frequency

        log_delta_t = np.log(delta_t)
        log_var_tau = np.log(var_tau)

        X = log_delta_t.reshape(-1, 1)
        y = log_var_tau

        model = LinearRegression()
        model.fit(X, y)

        slope = model.coef_[0]
        intercept = model.intercept_

        H = slope / 2
        sigma_A_former = sigma_A
        sigma_A = np.exp(intercept / 2) * 252 ** H

        if plot:
            plt.scatter(log_delta_t, y, label='Data')
            plt.plot(log_delta_t, model.predict(X), color='red', label='Linear regression')
            plt.xlabel('log(Δt)')
            plt.ylabel('log(Var(τ(Δt)))')
            plt.title(f"Iteration {iteration}")
            plt.legend()
            plt.show()

        return sigma_A, sigma_A_former, H

    def c_H(self, t, delta, H):
        """
        Calculate the c_H parameter for the Necula model.

        Args:
            t (float): Time.
            delta (float): Delta parameter.
            H (float): Hurst exponent.

        Returns:
            float: The c_H parameter.
        """
        return ((t + delta) ** (2 * H + 1) - t ** (2 * H + 1) -
                delta ** (2 * H + 1)) / ((2 * H + 1) * t * H)

    def sigma_estimate(self, VA, H):
        """
        Estimate the sigma parameter based on asset values and Hurst exponent.

        Args:
            VA (numpy.ndarray): Asset values.
            H (float): Hurst exponent.

        Returns:
            float: Estimated asset volatility.
        """
        n = VA.shape[0]
        step = np.arange(1, 11)
        Var = []

        for s in step:
            asset_resampled = np.array([VA[i] for i in range(0, n, s)])
            log_ret = np.diff(np.log(asset_resampled))
            Var.append(np.var(log_ret))

        sigma_A = np.sqrt(np.mean(Var)) * ((252 / step) ** H).mean()

        return sigma_A

    def calibrate(self):
        """
        Calibrate the Necula model to estimate asset volatility, Hurst exponent, and default probability.

        Returns:
            dict: Contains calibrated parameters including sigma_A, distance_to_default, 
            default_probability, H, and mu.
        """
        sigma_A_former = 0
        H = self.H0
        H_former = 0
        sigma_E = np.std(np.diff(np.log(self.company_market_cap))) * np.sqrt(self.frequency[0])
        sigma_A = sigma_E

        iteration = 0
        while (np.abs(sigma_A - sigma_A_former) > self.epsilon or 
               np.abs(H - H_former) > self.epsilon):
            asset_values = {}

            for f in self.frequency:
                asset_values[f] = []
                n = self.company_market_cap.shape[0]
                for day in range(0, n, n // f):
                    t = day / n
                    equity_value = self.company_market_cap[day]
                    asset_value = optimize.newton(
                        self.inversed_formula, 
                        self.company_debt, 
                        args=(t, t + self.T, H, equity_value, sigma_A), 
                        maxiter=100
                    )
                    asset_values[f].append(asset_value)

            Var = [np.var(np.diff(np.log(asset_values[f]))) for f in self.frequency]
            iteration += 1
            H_former = H
            sigma_A, sigma_A_former, H = self.update_values_regression(Var, sigma_A, iteration, n)

        t = int(n / 252)
        mu = ((1 / t) * np.log(asset_values[self.frequency[0]][-1] / 
            asset_values[self.frequency[0]][0]) + 
            (sigma_A ** 2) / 2 * (t ** (2 * H - 1)))
        distance_to_default = self.d2(asset_values[self.frequency[0]][-1], sigma_A, t, t + self.T, H, mu)
        default_probability = (1 - norm.cdf(distance_to_default)) * 100

        return {
            "sigma": sigma_A,
            "default_probability": default_probability,
            "mu": mu,
            "H": H,
        }
