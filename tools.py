from common_imports import np, pd, plt, os
from merton import Merton
from necula import Necula
from rostek import Rostek
from visualisations import plot_output

class Tools:
    def __init__(self, market_cap, debt, ticker_list, maturity_set=[1, 5, 10]):
        self.market_cap = market_cap
        self.debt = debt
        self.ticker_list = ticker_list
        self.maturity_set = maturity_set
        self.hurst_coeffs_df = pd.DataFrame(columns=['Model', 'Maturity', 'H_Coefficient'])
        self.default_probas_df = pd.DataFrame(columns=['Model', 'Maturity', 'DP(%)'])
        self.mu_df = pd.DataFrame(columns=['Model', 'Maturity', 'Mu'])
        self.sigma_df = pd.DataFrame(columns=['Model', 'Maturity', 'Sigma'])
        self.results = None

    def calibrate_and_combine_results(self, ticker_list):
        new_default_probas = []
        new_mu_values = []
        new_sigma_values = []
        new_hurst_coeffs = []

        models = {
            "Merton": Merton,
            "Necula": Necula,
            "Rostek": Rostek
        }
        
        print("For all tickers, for all models, computing {default probability, mu, sigma, H} for [1Y, 5Y, 10Y]..")
        
        for ticker in ticker_list:
            if ticker == "CGG FP Equity":
                break
            
            for model_name, model_class in models.items():
                for maturity in self.maturity_set:
                    # Calibrate model             
                    result = model_class(ticker, self.market_cap, self.debt, T=maturity).calibrate()
                    
                    # Collect results
                    new_default_probas.append({
                        'Ticker': ticker,
                        'Model': model_name,
                        'Maturity': maturity,
                        'DP(%)': result["default_probability"]
                    })
                    
                    new_mu_values.append({
                        'Ticker': ticker,
                        'Model': model_name,
                        'Maturity': maturity,
                        'Mu': result["mu"]
                    })
                    
                    new_sigma_values.append({
                        'Ticker': ticker,
                        'Model': model_name,
                        'Maturity': maturity,
                        'Sigma': result["sigma"]
                    })
                    
                    # H_Coefficient
                    H_value = result.get("H", "N.A") if model_name != "Merton" else 0.5
                    new_hurst_coeffs.append({
                        'Ticker': ticker,
                        'Model': model_name,
                        'Maturity': maturity,
                        'H': H_value
                    })

        # Combine results into dataframes
        self.default_probas_df = pd.concat([self.default_probas_df, pd.DataFrame(new_default_probas)], ignore_index=True)
        self.mu_df = pd.concat([self.mu_df, pd.DataFrame(new_mu_values)], ignore_index=True)
        self.sigma_df = pd.concat([self.sigma_df, pd.DataFrame(new_sigma_values)], ignore_index=True)
        self.hurst_coeffs_df = pd.concat([self.hurst_coeffs_df, pd.DataFrame(new_hurst_coeffs)], ignore_index=True)

        # Merging dataframes on Ticker, Maturity, and Model
        merged_df = pd.merge(self.default_probas_df, self.sigma_df, on=['Ticker', 'Maturity', 'Model'])
        merged_df = pd.merge(merged_df, self.mu_df, on=['Ticker', 'Maturity', 'Model'])
        merged_df = pd.merge(merged_df, self.hurst_coeffs_df, on=['Ticker', 'Maturity', 'Model'])
        
        merged_df.set_index(['Ticker', 'Model', 'Maturity'], inplace=True)
        
        # Set display float format, and show all rows
        pd.set_option('display.float_format', '{:.4f}'.format)
        pd.set_option('display.max_rows', None)
        
        # Reorganize columns to create the desired layout
        self.default_probas_df = self.default_probas_df[['Ticker', 'Model', 'Maturity', 'DP(%)']]
        self.sigma_df = self.sigma_df[['Ticker', 'Model', 'Maturity', 'Sigma']]
        self.mu_df = self.mu_df[['Ticker', 'Model', 'Maturity', 'Mu']]
        self.hurst_coeffs_df = self.hurst_coeffs_df[['Ticker', 'Model', 'Maturity', 'H']]
        merged_df = merged_df[['DP(%)', 'Sigma', 'Mu', 'H']]
        
        self.results = merged_df

    def get_default_probabilities(self):
        return self.default_probas_df

    def get_hurst_coeffs(self):
        return self.hurst_coeffs_df

    def get_mu_values(self):
        return self.mu_df

    def get_sigma_values(self):
        return self.sigma_df

    def get_results(self):
        if self.results is None:
            print("No results found from previous run, recalibration needed.")
        else:    
            return self.results
        
    def dump_results(self, window, style):
        if not os.path.exists('Calibration_Results'):
            os.makedirs('Calibration_Results')
        
        self.results.to_csv(os.path.join('Calibration_Results', f'full_results_{window}_{style}.csv'), sep=";")
        self.default_probas_df.to_csv(os.path.join('Calibration_Results', f'default_proba_{window}_{style}.csv'), sep=";")
        self.sigma_df.to_csv(os.path.join('Calibration_Results', f'sigma_{window}_{style}.csv'), sep=";")
        self.mu_df.to_csv(os.path.join('Calibration_Results', f'mu_{window}_{style}.csv'), sep=";")
        self.hurst_coeffs_df.to_csv(os.path.join('Calibration_Results', f'H_{window}_{style}.csv'), sep=";")

    # Plotters         
    
    def plot_probabilities(self, specific_ticker=None):
        plot_output(self.get_default_probabilities(), 'DP(%)', specific_ticker)

    def plot_hurst_coeffs(self, specific_ticker=None):
        plot_output(self.get_hurst_coeffs(), 'H', specific_ticker) 

    def plot_sigma(self, specific_ticker=None):
        plot_output(self.get_sigma_values(), 'Sigma', specific_ticker)

    def plot_mu(self, specific_ticker=None):
        plot_output(self.get_mu_values(), 'Mu', specific_ticker)
