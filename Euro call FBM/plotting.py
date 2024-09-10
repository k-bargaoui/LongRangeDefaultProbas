from common_imports import *

def plot_probability(mu_range, probas_necula, probas_rostek, xlabel, ylabel, title):
    plt.plot(mu_range, probas_necula, label="Necula")
    plt.plot(mu_range, probas_rostek, label="Rostek")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()