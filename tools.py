from common_imports import np, pd, plt
from merton import Merton
from necula import Necula
from rostek import Rostek
from visualisations import plot_probability

class Tools:
    def __init__(self, ticker, market_cap, debt, maturitySet = [1, 2, 5, 10, 15]):
        self.ticker = ticker
        self.market_cap = market_cap
        self.debt = debt
        self.maturitySet = maturitySet
        self.hurst_coeffs = pd.DataFrame(columns=['Model', 'Maturity', 'H_Coefficient'])
        self.default_probas_df = pd.DataFrame(columns=['Model', 'Maturity', 'default proba'])
        self.mu_df=pd.DataFrame(columns=['Model', 'Maturity', 'Mu'])
        self.sigma_df=pd.DataFrame(columns=['Model', 'Maturity', 'Sigma'])
        self.proba_merton = None
        self.proba_necula = None
        self.proba_rostek = None

    def compute_proba_default(self):
        self.proba_merton = np.zeros(len(self.maturitySet))
        self.proba_necula = np.zeros(len(self.maturitySet))
        self.proba_rostek = np.zeros(len(self.maturitySet))

        for i, m in enumerate(self.maturitySet):
            #Calibrate all models
            merton_result = Merton(self.ticker, self.market_cap, self.debt, T=m).calibrate()
            necula_result = Necula(self.ticker, self.market_cap, self.debt, T=m).calibrate()
            rostek_result = Rostek(self.ticker, self.market_cap, self.debt, T=m).calibrate()

            #extract probabilities and store them in the proba DF
            self.proba_merton[i] = merton_result["default_probability"]
            self.proba_necula[i] = necula_result["default_probability"]
            self.proba_rostek[i] = rostek_result["default_probability"]

            self.default_probas_df.loc[len(self.default_probas_df)] = ["Merton", m, self.proba_merton[i]]
            self.default_probas_df.loc[len(self.default_probas_df)] = ["Necula", m, self.proba_necula[i]]
            self.default_probas_df.loc[len(self.default_probas_df)] = ["Rostek", m, self.proba_rostek[i]]

            #extract and store H_coeffs
            self.hurst_coeffs .loc[len(self.hurst_coeffs )] = ["Merton", m, "N.A"]            
            self.hurst_coeffs .loc[len(self.hurst_coeffs )] = ["Necula", m, necula_result["H"]]
            self.hurst_coeffs .loc[len(self.hurst_coeffs )] = ["Rostek", m, rostek_result["H"]]

            #extract mu
            self.mu_df.loc[len(self.mu_df )] = ["Merton", m, merton_result["mu"]]
            self.mu_df.loc[len(self.mu_df )] = ["Necula", m, necula_result["mu"]]
            self.mu_df.loc[len(self.mu_df )] = ["Rostek", m, rostek_result["mu"]]

            #extract sigma
            self.sigma_df.loc[len(self.sigma_df )] = ["Merton", m, merton_result["sigma"]]
            self.sigma_df.loc[len(self.sigma_df )] = ["Necula", m, necula_result["sigma"]]
            self.sigma_df.loc[len(self.sigma_df )] = ["Rostek", m, rostek_result["sigma"]]        
    

    def getDefaultProbabilities(self):
        return self.default_probas_df
    
    def getHurstCoeffs(self):
        return self.hurst_coeffs
    
    def getMuValues(self):
        return self.mu_df
    
    def getSigmaValues(self):
        return self.sigma_df
    
    def calibrateAndGeParamsDict(self):
        self.compute_proba_default()
        probas = self.getDefaultProbabilities()
        coeffs = self.getHurstCoeffs()
        mus = self.getMuValues()
        sigmas = self.getSigmaValues()
        return dict({"probas":probas, "coeffs":coeffs,"mus":mus,"sigmas":sigmas})
    
    def calibrateAndGetCombinedResults(self):
        self.compute_proba_default()
        p = self.getDefaultProbabilities()
        h = self.getHurstCoeffs()
        m = self.getMuValues()
        s = self.getSigmaValues()
        d=pd.merge(pd.merge(pd.merge(p, m, on=['Maturity', 'Model']), h, on=['Maturity', 'Model']), s, on=['Maturity', 'Model'])
        return d
    
    def plotProbabilities(self):
        plot_probability(self.maturitySet, self.proba_merton, self.proba_necula, self.proba_rostek, self.ticker)
        return
    




    
