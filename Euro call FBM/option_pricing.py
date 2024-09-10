# option_pricing.py
from common_imports import *
from utils import ro_h, d1_rostek, d2_rostek

def black_scholes_call(S, K, r, sigma, t, T):
    DT = T - t
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * DT) / (sigma * np.sqrt(DT))
    d2 = d1 - sigma * np.sqrt(DT)
    return S * norm.cdf(d1) - K * np.exp(-r * DT) * norm.cdf(d2)

def necula_call(S, K, r, sigma, t, T, H):
    T_H_diff = T**(2 * H) - t**(2 * H)
    d1 = (np.log(S / K) + r * (T - t) + 0.5 * sigma ** 2 * T_H_diff) / (sigma * np.sqrt(T_H_diff))
    d2 = d1 - sigma * np.sqrt(T_H_diff)
    return S * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)

def rostek_call(S, K, r, sigma, t, T, H):
    roH = ro_h(H)
    d1_term = S * norm.cdf(d1_rostek(S, K, r, sigma, t, T, H, roH))
    d2_term = K * np.exp(-r * (T - t)) * norm.cdf(d2_rostek(S, K, r, sigma, t, T, H, roH))
    return d1_term - d2_term


#NEcula
# d1 calculation
def d1(x, sigma_A, t, T, H, mu, company_debt):
    time_factor = np.sqrt(T ** (2 * H) - t ** (2 * H))
    return (np.log(x / company_debt) + mu * (T - t) + 0.5 * sigma_A ** 2 * (T ** (2 * H) - t ** (2 * H))) / (sigma_A * time_factor)

# d2 calculation
def d2(x, sigma_A, t, T, H, mu, company_debt):
    time_factor = np.sqrt(T ** (2 * H) - t ** (2 * H))
    return (np.log(x / company_debt) + mu * (T - t) - 0.5 * sigma_A ** 2 * (T ** (2 * H) - t ** (2 * H))) / (sigma_A * time_factor)

# Probability of default using Necula's formula
def proba_necula(VA, sigma_A, t, T, H, mu, company_debt):
    distance_to_default = d2(VA, sigma_A, t, T, H, mu, company_debt)
    default_probability = (1 - norm.cdf(distance_to_default)) * 100
    return default_probability


#Rostek default

def ro_h(H):
    if H != 0.5:
        return ((np.sin(pi * (H - 0.5)) / (pi * (H - 0.5))) * ((gamma(1.5 - H) ** 2) / (gamma(2 - 2 * H))))
    return ((gamma(1.5 - H) ** 2) / (gamma(2 - 2 * H)))

def d1_r(x, sigma_A, t, T, H, mu, company_debt):
    roH = ro_h(H)
    return (((np.log(x / company_debt)) + mu * (T - t) + 0.5 * roH * (sigma_A ** 2) * (
                (T - t) ** (2 * H))) / (np.sqrt(roH) * sigma_A * ((T - t) ** H)))

def d2_r(x, sigma_A, t, T, H, mu, company_debt):
    roH = ro_h(H)
    return d1(x, sigma_A, t, T, H, mu, company_debt) - np.sqrt(roH) * sigma_A * ((T - t) ** H)

def proba_rostek(VA, sigma_A, t, T, H, mu, company_debt):
    
    distance_to_default = d2_r(VA, sigma_A, t, T, H, mu, company_debt)
    default_probability = (1 - norm.cdf(distance_to_default)) * 100
    return default_probability

