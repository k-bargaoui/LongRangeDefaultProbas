class CalibrationsModels:
    def __init__(self, ticker, market_cap, debt, T, frequency=252, rf=0, epsilon=1e-5):
        """
        Initialize the base calibration model with key financial and market parameters.
        
        Parameters:
        - ticker: str, stock ticker of the company
        - market_cap: DataFrame, historical market capitalization data (with ticker as column)
        - debt: DataFrame, company debt information (with ticker as column)
        - T: float, time to maturity (in years)
        - frequency: int, optional, default=252 (trading days in a year)
        - rf: float, optional, default=0 (risk-free rate)
        - epsilon: float, optional, default=1e-5 (tolerance level for calibration)
        """
        self.ticker = ticker
        self.market_cap = market_cap
        self.debt = debt
        self.T = T
        self.frequency = frequency
        self.rf = rf
        self.epsilon = epsilon
        
        # Extract the company's initial debt and market capitalization data
        self.company_debt = debt[ticker].iloc[0]  # First row of the ticker's debt column
        self.company_market_cap = market_cap[ticker]  # Full column for the market cap of the ticker