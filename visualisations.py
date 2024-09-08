from common_imports import plt

def plot_probability(maturity, proba_merton, proba_necula, proba_rostek, ticker):
    plt.figure()
    plt.plot(maturity, proba_merton, label="Merton")
    plt.plot(maturity, proba_necula, label="Necula")
    plt.plot(maturity, proba_rostek, label="Rostek")
    plt.legend()
    plt.title(f"Default probability vs. maturity T for {ticker}")
    plt.xlabel("Maturity (T)")
    plt.ylabel("Probability")
    plt.show()
