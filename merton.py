from common_imports import np, norm, optimize
from calibrationModels import CalibrationModel


class Merton(CalibrationModel):
    """
    A class representing the Merton model for calibrating financial models.
    
    Inherits from the CalibrationsModels class and is used to calibrate 
    financial models, estimate asset volatility, and calculate the probability 
    of default.

    Attributes:
        ticker (str): Ticker symbol of the company.
        market_cap (pandas.DataFrame): Market capitalization data.
        debt (pandas.DataFrame): Debt data of the company.
        T (float): Time to maturity (in years).
        frequency (int): Data frequency (default is 252).
        rf (float): Risk-free interest rate (default is 0).
        epsilon (float): Tolerance for convergence (default is 1e-5).
    """

    def __init__(self, ticker, market_cap, debt, T, frequency=252, rf=0, epsilon=1e-5):
        """
        Initialize the Merton model with company data and parameters.

        Args:
            ticker (str): Ticker symbol of the company.
            market_cap (pandas.DataFrame): Market capitalization data.
            debt (pandas.DataFrame): Debt data of the company.
            T (float): Time to maturity (in years).
            frequency (int, optional): Data frequency, default is 252 (daily).
            rf (float, optional): Risk-free interest rate, default is 0.
            epsilon (float, optional): Tolerance for convergence, default is 1e-5.
        """
        super().__init__(ticker, market_cap, debt, T, frequency, rf, epsilon)

    def d1(self, x, sigma_A, current_time, mu):
        """
        Calculate the d1 parameter in the Merton model.

        Args:
            x (float): Asset value at current time.
            sigma_A (float): Volatility of the asset.
            current_time (float): Time to maturity.
            mu (float): Drift of the asset.

        Returns:
            float: The d1 parameter value.
        """
        return (np.log(x / self.company_debt) + mu * current_time) / (
            sigma_A * np.sqrt(current_time)
        )

    def d2(self, x, sigma_A, current_time, mu):
        """
        Calculate the d2 parameter, which is d1 minus asset volatility term.

        Args:
            x (float): Asset value at current time.
            sigma_A (float): Volatility of the asset.
            current_time (float): Time to maturity.
            mu (float): Drift of the asset.

        Returns:
            float: The d2 parameter value.
        """
        return self.d1(x, sigma_A, current_time, mu) - sigma_A * np.sqrt(current_time)

    def inversed_formula(self, x, current_time, equity_value, sigma_A):
        """
        Calculate the inverse formula used to calibrate asset values.

        Args:
            x (float): Asset value at current time.
            current_time (float): Time to maturity.
            equity_value (float): Market equity value.
            sigma_A (float): Volatility of the asset.

        Returns:
            float: The result of the inverse formula.
        """
        mu = self.rf + (sigma_A ** 2) / 2
        term1 = x * norm.cdf(self.d1(x, sigma_A, current_time, mu))
        term2 = (
            self.company_debt * np.exp(-self.rf * current_time) * norm.cdf(
                self.d2(x, sigma_A, current_time, mu)
            )
        )
        return term1 - term2 - equity_value

    def calibrate(self):
        """
        Calibrate the Merton model to estimate asset volatility and probability of default.

        Returns:
            dict: Contains the calibrated parameters:
                - 'sigma': Asset volatility.
                - 'distance_to_default': Distance to default.
                - 'default_probability': Default probability (as percentage).
                - 'mu': Drift of the asset.
        """
        sigma_A_former = 0
        sigma_E = np.std(np.diff(np.log(self.company_market_cap))) * np.sqrt(self.frequency)
        sigma_A = sigma_E
        asset_values = []

        while np.abs(sigma_A - sigma_A_former) > self.epsilon:
            sigma_A_former = sigma_A
            asset_values.clear()

            for dt in range(self.company_market_cap.shape[0]):
                current_time = self.T + (self.company_market_cap.shape[0] - dt - 1) / self.frequency
                equity_value = self.company_market_cap[dt]

                # Solve for asset value at each time step
                asset_value = optimize.newton(
                    self.inversed_formula, 
                    self.company_debt, 
                    args=(current_time, equity_value, sigma_A), 
                    maxiter=50
                )
                asset_values.append(asset_value)

            # Recalculate asset volatility (sigma_A)
            sigma_A = np.std(np.diff(np.log(asset_values))) * np.sqrt(self.frequency)

        mu = np.mean(np.diff(np.log(asset_values))) * self.frequency + (sigma_A ** 2) / 2

        # Calculate distance to default and default probability
        distance_to_default = self.d2(asset_values[-1], sigma_A, current_time, mu)
        default_probability = (1 - norm.cdf(distance_to_default)) * 100

        return {
            "sigma": sigma_A,
            "default_probability": default_probability,
            "mu": mu,
        }
