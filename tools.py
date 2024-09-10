from common_imports import np, pd, plt
from merton import Merton
from necula import Necula
from rostek import Rostek
from visualisations import plot_probability

class Tools:
    def __init__(self, ticker, market_cap, debt, maturitySet=[1, 2, 5, 10, 15]):
        self.ticker = ticker
        self.market_cap = market_cap
        self.debt = debt
        self.maturitySet = maturitySet
        self.hurst_coeffs = pd.DataFrame(columns=['Model', 'Maturity', 'H_Coefficient'])
        self.default_probas_df = pd.DataFrame(columns=['Model', 'Maturity', 'default proba'])
        self.mu_df = pd.DataFrame(columns=['Model', 'Maturity', 'Mu'])
        self.sigma_df = pd.DataFrame(columns=['Model', 'Maturity', 'Sigma'])
        self.results = None

    def compute_proba_default(self):
        models = {
            "Merton": Merton,
            "Necula": Necula,
            "Rostek": Rostek
        }

        new_default_probas = []
        new_mu_values = []
        new_sigma_values = []
        new_hurst_coeffs = []

        for m in self.maturitySet:
            for model_name, model_class in models.items():
                result = model_class(self.ticker, self.market_cap, self.debt, T=m).calibrate()

                # Collect results
                new_default_probas.append({'Model': model_name, 'Maturity': m, 'default proba': result["default_probability"]})
                new_mu_values.append({'Model': model_name, 'Maturity': m, 'Mu': result["mu"]})
                new_sigma_values.append({'Model': model_name, 'Maturity': m, 'Sigma': result["sigma"]})

                # H_Coefficient
                H_value = result.get("H", "N.A") if model_name != "Merton" else "N.A"
                new_hurst_coeffs.append({'Model': model_name, 'Maturity': m, 'H_Coefficient': H_value})

        # Update dataframes
        self.default_probas_df = pd.concat([self.default_probas_df, pd.DataFrame(new_default_probas)], ignore_index=True)
        self.mu_df = pd.concat([self.mu_df, pd.DataFrame(new_mu_values)], ignore_index=True)
        self.sigma_df = pd.concat([self.sigma_df, pd.DataFrame(new_sigma_values)], ignore_index=True)
        self.hurst_coeffs = pd.concat([self.hurst_coeffs, pd.DataFrame(new_hurst_coeffs)], ignore_index=True)

    def getDefaultProbabilities(self):
        return self.default_probas_df

    def getHurstCoeffs(self):
        return self.hurst_coeffs

    def getMuValues(self):
        return self.mu_df

    def getSigmaValues(self):
        return self.sigma_df

    def getLastRunResults(self):
        return self.results

    def calibrate_and_get_params_dict(self):
        self.compute_proba_default()
        self.results = {
            "probas": self.getDefaultProbabilities(),
            "coeffs": self.getHurstCoeffs(),
            "mus": self.getMuValues(),
            "sigmas": self.getSigmaValues()
        }
        return self.results

    def calibrate_and_get_combined_results(self):
        self.compute_proba_default()
        combined_df = pd.merge(pd.merge(pd.merge(self.default_probas_df, self.mu_df, on=['Maturity', 'Model']),
                                        self.hurst_coeffs, on=['Maturity', 'Model']),
                               self.sigma_df, on=['Maturity', 'Model'])
        return combined_df

    def plot_probabilities(self):
        plot_probability(self.getDefaultProbabilities(), self.ticker)





    
