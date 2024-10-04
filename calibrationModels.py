class CalibrationModel:
    def __init__(self, ticker, market_cap, debt, T, frequency=252, rf=0, epsilon=1e-5):
        """
        Initialize the calibration model with key financial and market parameters.
        
        Parameters:
        - ticker: str
            Stock ticker of the company.
        - market_cap: DataFrame
            Historical market capitalization data (with ticker as column).
        - debt: DataFrame
            Company debt information (with ticker as column).
        - T: float
            Time to maturity (in years).
        - frequency: int, optional
            Trading days in a year (default is 252).
        - rf: float, optional
            Risk-free rate (default is 0).
        - epsilon: float, optional
            Tolerance level for calibration (default is 1e-5).
        """
        self.ticker = ticker
        self.market_cap = market_cap[ticker]
        self.debt = debt[ticker]
        self.T = T
        self.frequency = frequency
        self.rf = rf
        self.epsilon = epsilon
        
        # Extracting company debt and market cap
        self.company_debt = self.debt.iloc[0]
        self.company_market_cap = self.market_cap
