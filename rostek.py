from common_imports import gamma, np, norm, LinearRegression, plt, optimize, pi
from calibrationModels import CalibrationModel


class Rostek(CalibrationModel):
    """
    A class representing the Rostek model for calibrating financial models.

    Inherits from CalibrationsModels.

    Attributes:
        Inherits attributes from the CalibrationsModels class.

    Methods:
        ro_h(H): Calculate ro_h parameter of the Rostek model.
        d1(x, sigma_A, t, T, H, mu): Calculate d1 parameter of the Rostek model.
        d2(x, sigma_A, t, T, H, mu): Calculate d2 parameter of the Rostek model.
        inversed_formula(x, t, T, H, equity_value, sigma_A): Calculate the inverse formula of the Rostek model.
        update_values_regression(Var, sigma_A, iteration, n, plot=False): Update regression values for the Rostek model.
        c_H(t, delta, H): Calculate c_H parameter of the Rostek model.
        calibrate(): Calibrate the Rostek model to fit the data and return a dictionary of calibrated parameters.
    """

    def __init__(self, ticker, market_cap, debt, T,
                 frequency=252 // np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                 rf=0, epsilon=1e-5):
        """
        Initialize the Rostek model with the given parameters.

        Args:
            ticker (str): Ticker symbol of the company.
            market_cap (pandas.DataFrame): DataFrame containing market capitalization data.
            debt (pandas.DataFrame): DataFrame containing debt data.
            T (float): Time to maturity (in years).
            frequency (numpy.ndarray, optional): Array containing frequencies of data.
            rf (float, optional): Risk-free interest rate.
            epsilon (float, optional): Tolerance parameter for convergence (default is 10e-5).
        """
        super().__init__(ticker, market_cap, debt, T, frequency, rf, epsilon)
        self.H0 = 0.5

    def ro_h(self, H):
        """
        Calculate ro_h parameter based on Hurst exponent H.

        Args:
            H (float): Hurst exponent.

        Returns:
            float: The ro_h parameter.
        """
        if H != 0.5:
            return ((np.sin(pi * (H - 0.5)) / (pi * (H - 0.5))) *
                    ((gamma(1.5 - H) ** 2) / (gamma(2 - 2 * H))))
        return ((gamma(1.5 - H) ** 2) / (gamma(2 - 2 * H)))

    def d1(self, x, sigma_A, t, T, H, mu):
        """
        Calculate the d1 parameter for the Rostek model.

        Args:
            x (float): Input variable.
            sigma_A (float): Volatility parameter.
            t (float): Current time.
            T (float): Time to maturity.
            H (float): Hurst exponent.
            mu (float): Mean parameter.

        Returns:
            float: The d1 parameter.
        """
        roH = self.ro_h(H)
        return (((np.log(x / self.company_debt)) + mu * (T - t) +
                 0.5 * roH * (sigma_A ** 2) * ((T - t) ** (2 * H))) /
                (np.sqrt(roH) * sigma_A * ((T - t) ** H)))

    def d2(self, x, sigma_A, t, T, H, mu):
        """
        Calculate the d2 parameter for the Rostek model.

        Args:
            x (float): Input variable.
            sigma_A (float): Volatility parameter.
            t (float): Current time.
            T (float): Time to maturity.
            H (float): Hurst exponent.
            mu (float): Mean parameter.

        Returns:
            float: The d2 parameter.
        """
        roH = self.ro_h(H)
        return (self.d1(x, sigma_A, t, T, H, mu) -
                np.sqrt(roH) * sigma_A * ((T - t) ** H))

    def inversed_formula(self, x, t, T, H, equity_value, sigma_A):
        """
        Calculate the inverse formula of the Rostek model.

        Args:
            x (float): Input variable.
            t (float): Current time.
            T (float): Time to maturity.
            H (float): Hurst exponent.
            equity_value (float): Equity value.
            sigma_A (float): Volatility parameter.

        Returns:
            float: The inverse formula value.
        """
        d1_term = x * norm.cdf(self.d1(x, sigma_A, t, T, H, self.rf))
        d2_term = (self.company_debt * np.exp(-self.rf * (T - t)) *
                   norm.cdf(self.d2(x, sigma_A, t, T, H, self.rf)))
        return d1_term - d2_term - equity_value

    def update_values_regression(self, Var, sigma_A, iteration, n, plot=False):
        """
        Update regression values for the Rostek model.

        Args:
            Var (numpy.ndarray): Array containing variance values.
            sigma_A (float): Volatility parameter.
            iteration (int): Iteration number.
            n (int): Number of data points.
            plot (bool, optional): Whether to plot the results (default is False).

        Returns:
            tuple: Updated sigma_A, sigma_A_former, and H values.
        """
        log_delta_t = np.log(n // self.frequency)
        log_var_tau = np.log(Var)

        model = LinearRegression()
        model.fit(log_delta_t.reshape(-1, 1), log_var_tau)

        slope = model.coef_[0]
        intercept = model.intercept_

        H = slope / 2
        sigma_A_former = sigma_A
        sigma_A = np.exp(intercept / 2) * 252 ** H

        if plot:
            plt.scatter(log_delta_t, log_var_tau, label='Data')
            plt.plot(log_delta_t, model.predict(log_delta_t.reshape(-1, 1)),
                     color='red', label='Regression')
            plt.xlabel('log(Delta t)')
            plt.ylabel('log(Var(tau(Delta t)))')
            plt.title(f"Iteration {iteration} Regression")
            plt.legend()
            plt.show()

        return sigma_A, sigma_A_former, H

    def c_H(self, t, delta, H):
        """
        Calculate c_H parameter of the Rostek model.

        Args:
            t (float): Current time.
            delta (float): Delta value.
            H (float): Hurst exponent.

        Returns:
            float: The c_H parameter.
        """
        return ((t + delta)**(2 * H + 1) - t**(2 * H + 1) -
                 delta**(2 * H + 1)) / ((2 * H + 1) * t * H)

    def calibrate(self):
        """
        Calibrate the Rostek model.

        Returns:
            dict: A dictionary containing calibrated parameters.
                  Includes sigma, distance_to_default, default_probability, mu, H, and sigma_A_former.
        """
        sigma_A_former = 0
        H = self.H0
        H_former = 0
        sigma_E = (np.std(np.diff(np.log(self.company_market_cap), n=1)) *
                    np.sqrt(self.frequency[0]))
        sigma_A = sigma_E

        iteration = 1
        asset_values = {}

        # Calibration loop
        while (np.abs(sigma_A - sigma_A_former) > self.epsilon or
               np.abs(H - H_former) > self.epsilon):
            for f in self.frequency:
                fasset_values = []
                n = self.company_market_cap.shape[0]
                days = range(0, n, n // f)
                for day in days:
                    t = day / n
                    equity_value = self.company_market_cap[day]
                    fasset_values.append(optimize.newton(
                        self.inversed_formula, self.company_debt,
                        args=(t, t + self.T, H, equity_value, sigma_A),
                        maxiter=100))
                asset_values[f] = fasset_values

            Var = [np.var(np.diff(np.log(asset_values[f]), n=1)) for f in self.frequency]

            H_former = H
            sigma_A, sigma_A_former, H = self.update_values_regression(
                Var, sigma_A, iteration, len(self.company_market_cap))

            iteration += 1

        t = len(self.company_market_cap) / 252
        mu = ((1/t) * np.log(asset_values[self.frequency[0]][-1] /
              asset_values[self.frequency[0]][0]) +
              (sigma_A**2) / 2 * (t ** (2 * H - 1)))

        distance_to_default = self.d2(asset_values[self.frequency[0]][-1],
                                       sigma_A, t, t + self.T, H, mu)
        default_probability = (1 - norm.cdf(distance_to_default)) * 100

        return {
            "sigma": sigma_A,
            "default_probability": default_probability,
            "mu": mu,
            "H": H,
        }
