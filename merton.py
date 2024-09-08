from common_imports import np, norm, optimize
from calibrationModels import CalibrationsModels

class Merton(CalibrationsModels):
    """
    Merton model class for calibrating default probability and asset volatility.
    
    Inherits from CalibrationsModels and uses the Merton model for pricing equity
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
    """
    def __init__(self, ticker, market_cap, debt, T, frequency=252, rf=0, epsilon=1e-5):
        super().__init__(ticker, market_cap, debt, T, frequency, rf, epsilon)
        # Assuming the parent class sets the necessary attributes

    def d1(self, x, sigma_A, current_time, mu):
        numerator = np.log(x / self.company_debt) + mu * current_time
        denominator = sigma_A * np.sqrt(current_time)
        return numerator / denominator

    def d2(self, x, sigma_A, current_time, mu):
        return self.d1(x, sigma_A, current_time, mu) - sigma_A * np.sqrt(current_time)

    def inversed_formula(self, x, current_time, equity_value, sigma_A):
        mu = self.rf + 0.5 * sigma_A ** 2
        d1 = self.d1(x, sigma_A, current_time, mu)
        d2 = self.d2(x, sigma_A, current_time, mu)
        term1 = x * norm.cdf(d1)
        term2 = self.company_debt * np.exp(-self.rf * current_time) * norm.cdf(d2)
        return term1 - term2 - equity_value

    def calibrate(self):
        # Calculate initial asset volatility using equity volatility
        sigma_E = np.std(np.diff(np.log(self.company_market_cap))) * np.sqrt(self.frequency)
        sigma_A = sigma_E
        sigma_A_prev = 0  # Initialize previous sigma_A for convergence check

        num_points = len(self.company_market_cap)
        time_indices = np.arange(num_points)
        current_times = self.T + (num_points - time_indices - 1) / self.frequency

        while np.abs(sigma_A - sigma_A_prev) > self.epsilon:
            asset_values = []

            for equity_value, current_time in zip(self.company_market_cap, current_times):
                # Find the asset value that solves the Merton model equation
                asset_value = optimize.newton(
                    func=self.inversed_formula,
                    x0=self.company_debt,
                    args=(current_time, equity_value, sigma_A),
                    maxiter=50
                )
                asset_values.append(asset_value)

            # Update sigma_A for convergence
            sigma_A_prev = sigma_A
            sigma_A = np.std(np.diff(np.log(asset_values))) * np.sqrt(self.frequency)

        # Calculate the drift term mu
        mu = np.mean(np.diff(np.log(asset_values))) * self.frequency + 0.5 * sigma_A ** 2

        # Compute distance to default and default probability
        latest_asset_value = asset_values[-1]
        current_time = current_times[-1]
        distance_to_default = self.d2(latest_asset_value, sigma_A, current_time, mu)
        default_probability = (1 - norm.cdf(distance_to_default)) * 100  # In percentage

        return dict({"sigma": sigma_A, "distance_to_default":distance_to_default,
                      "default_probability":default_probability,"mu":mu})