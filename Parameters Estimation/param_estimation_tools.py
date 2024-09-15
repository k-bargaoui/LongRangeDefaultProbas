from common_imports import FBM, np, LinearRegression

def evol_va(mu, sigma_a, fbm_sample, times, H):
    """Compute the evolved volatility adjustment."""
    return np.exp(mu * times - (sigma_a ** 2 / 2) * times ** (2 * H) + sigma_a * fbm_sample)

def compute_var_log_returns(VA, steps):
    """Compute variance of log returns for different resampling steps."""
    n = VA.shape[0]
    Var = []
    for s in steps:
        resampled_values = VA[::s]
        log_ret = np.diff(np.log(resampled_values))
        Var.append(np.var(log_ret))
    return np.array(Var)

def get_H_slope(VA):
    """Estimate Hurst exponent H using the slope of variance of log returns."""
    steps = np.arange(1, 11)
    Var = compute_var_log_returns(VA, steps)
    log_step = np.log(steps)
    log_var_tau = np.log(Var)
    model = LinearRegression(fit_intercept=False)
    model.fit(log_step.reshape(-1, 1), log_var_tau - np.log(Var[0]))
    return model.coef_[0] / 2

def get_sigma_slope(VA):
    """Estimate volatility sigma from the slope of variance of log returns."""
    steps = np.arange(1, 11)
    Var = compute_var_log_returns(VA, steps)
    log_step = np.log(steps)
    log_var_tau = np.log(Var)
    model = LinearRegression()
    model.fit(log_step.reshape(-1, 1), log_var_tau)
    return np.exp(model.intercept_ / 2) * len(VA) ** (model.coef_[0] / 2)

def get_H_from_drift(V_A, drift):
    """Estimate Hurst exponent H from drift and variance of log returns."""
    delta = 1 / len(V_A)
    X = np.diff(np.log(V_A)) - drift * delta
    N = len(X)
    if N % 2 == 1:
        raise ValueError("Time series length must be even for this estimation method.")
    even_odd_pairs = sum((X[2 * i + 1] + X[2 * i]) ** 2 for i in range(N // 2 - 1))
    odd_even_pairs = sum((X[2 * i + 2] + X[2 * i + 1]) ** 2 for i in range(N // 2 - 1))
    denominator = np.mean(X ** 2)
    return (1 / (2 * np.log(2))) * np.log((even_odd_pairs + odd_even_pairs) / (2 * N) / denominator)

def get_sigma_1(VA, H):
    """Estimate sigma from variance of log returns and Hurst exponent."""
    steps = np.arange(1, 11)
    Var = compute_var_log_returns(VA, steps)
    return np.mean(np.sqrt(Var) * (len(VA) / steps) ** H)

def get_sigma_2(H, drift, V_A):
    """Estimate sigma from drift, Hurst exponent, and log returns."""
    delta = 1 / len(V_A)
    X = np.diff(np.log(V_A)) - drift * delta
    return np.sum(X ** (1 / H)) ** H

def H_error(sigma, mu, H, n, n_it=10, T=1):
    """Calculate Hurst exponent error based on estimated and true values."""
    f = FBM(n, hurst=H, length=T, method='daviesharte')
    VA = evol_va(mu, sigma, f.fbm(), f.times(), H)
    return get_H_slope(VA)

def get_drift(VA):
    """Estimate drift from log returns of VA."""
    delta_t = 1 / 252
    Y = np.log(VA)
    Z = np.diff(Y)
    return np.mean(Z) / delta_t

def H_error_from_drift(sigma, mu, H, n, n_it=10, T=1):
    """Calculate Hurst exponent error based on drift estimation."""
    f = FBM(n, hurst=H, length=T, method='daviesharte')
    VA = evol_va(mu, sigma, f.fbm(), f.times(), H)
    drift = get_drift(VA)
    H1 = get_H_from_drift(VA, mu - sigma ** 2 / 2)
    Hest = get_H_from_drift(VA, drift)
    return H1, Hest

def sigma_error(true_sigma, mu, H, n, n_it=10, T=1):
    """Calculate sigma error based on true and estimated values."""
    f = FBM(n, hurst=H, length=T, method='daviesharte')
    VA = evol_va(mu, true_sigma, f.fbm(), f.times(), H)
    H_est = get_H_slope(VA)
    return get_sigma_1(VA, H), get_sigma_1(VA, H_est)

def sigma_error_from_drift(true_sigma, mu, H, n, n_it=10, T=1):
    """Calculate sigma error based on drift estimation and Hurst exponent."""
    f = FBM(n, hurst=H, length=T, method='daviesharte')
    VA = evol_va(mu, true_sigma, f.fbm(), f.times(), H)
    drift = get_drift(VA)
    sigma_1 = get_sigma_2(H, mu - true_sigma ** 2 / 2, VA)
    H_est = get_H_from_drift(VA, drift)
    return sigma_1, get_sigma_2(H_est, drift, VA)

def sigma_error_intercept(true_sigma, mu, H, n, n_it=10, T=1):
    """Calculate sigma based on intercept estimation."""
    f = FBM(n, hurst=H, length=T, method='daviesharte')
    VA = evol_va(mu, true_sigma, f.fbm(), f.times(), H)
    return get_sigma_slope(VA)

def get_mu(VA, sigma):
    """Estimate mu from drift and sigma."""
    drift = get_drift(VA)
    return drift + sigma ** 2 / 2

def get_mu2(VA, H, sigma_A):
    """Estimate mu using a different method based on variance and Hurst exponent."""
    delta = 1 / 252
    log_ret = np.diff(np.log(VA))
    mean_log_ret = np.mean(log_ret)
    t = 1
    return (mean_log_ret + sigma_A ** 2 / (2 * t) * ((t + delta) ** (2 * H + 1) - t ** (2 * H + 1) - delta ** (2 * H + 1)) / (2 * H + 1)) / delta

def get_mu3(VA, H, sigma_A):
    """Estimate mu using a third method based on variance and Hurst exponent."""
    delta = 1 / 252
    T = len(VA) * delta
    return (1 / T) * np.log(VA[-1] / VA[0]) + sigma_A ** 2 / 2 * T ** (2 * H - 1)

def mu_error(sigma, mu, H, n, n_it=10, T=1):
    """Calculate mu error based on different estimation methods."""
    f = FBM(n, hurst=H, length=T * n / 252, method='daviesharte')
    VA = evol_va(mu, sigma, f.fbm(), f.times(), H)
    return get_mu(VA, sigma), get_mu3(VA, H, sigma)

def get_mu_merton(VA, sigma):
    """Estimate mu using the Merton method."""
    delta = 1 / 252
    return np.mean(np.diff(np.log(VA))) / delta + sigma ** 2 / 2

def mu_merton_error(sigma, mu, n, n_it=10, T=1):
    """Calculate mu error using the Merton method."""
    f = FBM(n, hurst=0.5, length=T * n / 252, method='daviesharte')
    VA = evol_va(mu, sigma, f.fbm(), f.times(), 0.5)
    return get_mu_merton(VA, sigma)
