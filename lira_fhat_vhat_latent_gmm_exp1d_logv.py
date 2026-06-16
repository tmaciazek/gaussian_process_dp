import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import optimize, signal, linalg

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# ============================================================
# Kernel, labels, and posterior quantities
# ============================================================


def exponential_kernel(x: np.ndarray, y: np.ndarray, ell: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.exp(-np.abs(x[:, None] - y[None, :]) / ell)


def solve_spd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve A X = B for symmetric positive-definite A using Cholesky.

    Falls back to np.linalg.solve if the Cholesky factorization fails.
    """
    try:
        c, lower = linalg.cho_factor(A, lower=True, check_finite=False)
        return linalg.cho_solve((c, lower), B, check_finite=False)
    except linalg.LinAlgError:
        return np.linalg.solve(A, B)



def threshold_labels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x < 0.5, -1.0, 1.0)


def noisy_threshold_labels(x: np.ndarray, m_eps: float, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    base = threshold_labels(x)
    if not (0.0 <= m_eps <= 1.0):
        raise ValueError("m_eps must lie in [0, 1].")
    eps = rng.uniform(-m_eps, m_eps, size=x.shape)
    return base * (1.0 - m_eps) + eps



def scaled_threshold_labels(x: np.ndarray, m_eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not (0.0 <= m_eps <= 1.0):
        raise ValueError("m_eps must lie in [0, 1].")
    return threshold_labels(x) * (1.0 - m_eps)



def posterior_mean_and_normalized_covariance(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    ell: float,
    r: float,
    jitter: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)

    K_tt = exponential_kernel(x_eval, x_eval, ell=ell)
    if len(x_train) == 0:
        return np.zeros(len(x_eval), dtype=float), K_tt

    K_xx = exponential_kernel(x_train, x_train, ell=ell)
    A = K_xx + (r**2 + jitter) * np.eye(len(x_train))
    K_tx = exponential_kernel(x_eval, x_train, ell=ell)

    alpha = solve_spd(A, y_train)
    mu = K_tx @ alpha

    tmp = solve_spd(A, K_tx.T)
    cov = K_tt - K_tx @ tmp
    cov = 0.5 * (cov + cov.T)
    return mu, cov



def draw_posterior_sample_paths(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    ell: float,
    r: float,
    sigma: float,
    jitter: float,
    n_draws: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    mu, cov = posterior_mean_and_normalized_covariance(
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        ell=ell,
        r=r,
        jitter=jitter,
    )
    if n_draws < 1:
        raise ValueError("n_draws must be positive.")
    if sigma == 0.0:
        draws = np.tile(mu, (n_draws, 1))
        return mu, draws

    scaled_cov = (sigma ** 2) * cov
    scaled_cov = 0.5 * (scaled_cov + scaled_cov.T)
    eigvals, eigvecs = np.linalg.eigh(scaled_cov)
    eigvals = np.maximum(eigvals, 0.0)
    L = eigvecs @ np.diag(np.sqrt(eigvals))
    z = rng.normal(size=(n_draws, len(x_eval)))
    draws = mu[None, :] + z @ L.T
    return mu, draws



def posterior_mean_and_normalized_variance_at_x0(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x0: float,
    ell: float,
    r: float,
    jitter: float,
) -> Tuple[float, float]:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if len(x_train) == 0:
        return 0.0, 1.0

    K_xx = exponential_kernel(x_train, x_train, ell=ell)
    A = K_xx + (r**2 + jitter) * np.eye(len(x_train))
    k0 = exponential_kernel(np.array([x0]), x_train, ell=ell).reshape(-1)

    alpha = solve_spd(A, y_train)
    mu0 = float(k0 @ alpha)

    tmp = solve_spd(A, k0)
    v0 = 1.0 - float(k0 @ tmp)
    v0 = max(v0, 0.0)
    return mu0, v0



def released_scalar_statistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x0: float,
    ell: float,
    r: float,
    sigma: float,
    jitter: float,
    n_posterior_draws: int,
    rng: np.random.Generator,
) -> float:
    mu0, v0 = posterior_mean_and_normalized_variance_at_x0(
        x_train=x_train,
        y_train=y_train,
        x0=x0,
        ell=ell,
        r=r,
        jitter=jitter,
    )
    if n_posterior_draws < 1:
        raise ValueError("n_posterior_draws must be positive.")
    if sigma == 0.0 or v0 == 0.0:
        return mu0
        
    if np.isinf(sigma):
    	draws = np.sqrt(v0) * rng.normal(size=n_posterior_draws)
    else:
    	draws = mu0 + sigma * np.sqrt(v0) * rng.normal(size=n_posterior_draws)
    return float(np.mean(draws))


# ============================================================
# Dataset generation
# ============================================================


def draw_dataset_out(n: int, m_eps: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    #x = rng.uniform(0.0, 1.0, size=n)
    x1 = rng.uniform(0.0, 0.5, size=int(n/2))
    x2 = rng.uniform(0.5, 1.0, size=n-int(n/2))
    x = np.concatenate([x1,x2])
    y = noisy_threshold_labels(x, m_eps=m_eps, rng=rng)
    return x, y



def draw_dataset_in(n: int, x0: float, m_eps: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be positive.")
    #x_rest = rng.uniform(0.0, 1.0, size=n - 1)
    x_rest1 = rng.uniform(0.0, 0.5, size=int(n/2))
    x_rest2 = rng.uniform(0.5, 1.0, size=n-1-int(n/2))
    x = np.concatenate([[x0], x_rest1, x_rest2])
    #x = np.concatenate([[x0], x_rest])
    y = noisy_threshold_labels(x, m_eps=m_eps, rng=rng)
    return x, y


# ============================================================
# Transformations
# ============================================================


def transform_statistic(
    values: np.ndarray,
    sigma: float,
    transform_eps: float,
    asinh_scale: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if sigma == 0.0:
        clipped = np.clip(values, -1.0 + transform_eps, 1.0 - transform_eps)
        return np.log((1.0 + clipped) / (1.0 - clipped))
    if asinh_scale <= 0.0:
        raise ValueError("asinh_scale must be positive.")
    return np.arcsinh(values / asinh_scale)



def transform_description(density_model: str, sigma: float, transform_eps: float, asinh_scale: float) -> str:
    if density_model == "latent_gmm2":
        if sigma == 0.0:
            return (
                "phi = log((1 + clip(f, -1+eps, 1-eps)) / (1 - clip(f, -1+eps, 1-eps))) "
                f"with eps={transform_eps}; latent model f(x0) = tanh(phi/2), phi ~ GMM(2)"
            )
        return "raw released statistic f(x0); latent model f(x0) = tanh(phi/2) + psi with phi ~ GMM(2), psi Gaussian"
    if sigma == 0.0:
        return (
            "phi = log((1 + clip(f, -1+eps, 1-eps)) / (1 - clip(f, -1+eps, 1-eps))) "
            f"with eps={transform_eps}"
        )
    return f"phi = asinh(f / c) with c={asinh_scale}"



def transformed_axis_label(density_model: str, sigma: float, asinh_scale: float) -> str:
    if density_model == "latent_gmm2":
        return r"released statistic $f(x_0)$"
    if sigma == 0.0:
        return r"transformed statistic $\phi = \log\!\left(\frac{1+f(x_0)}{1-f(x_0)}\right)$"
    return rf"transformed statistic $\phi = \operatorname{{asinh}}(f(x_0)/{asinh_scale:g})$"


def latent_gmm2_sigma0_f_logpdf_1d(
    x: np.ndarray,
    fit: Dict[str, object],
    density_floor: float = 1e-300,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    transform_eps = float(fit["transform_eps"])
    clipped = np.clip(x, -1.0 + transform_eps, 1.0 - transform_eps)
    phi = np.log((1.0 + clipped) / (1.0 - clipped))
    phi_fit = {
        "model_type": "gmm2",
        "weights": np.asarray(fit["weights"], dtype=float),
        "means": np.asarray(fit["means"], dtype=float),
        "variances": np.asarray(fit["latent_variances"], dtype=float),
    }
    logpdf_phi = gmm2_logpdf_1d(phi, phi_fit, density_floor=density_floor)
    log_jac = np.log(2.0) - np.log(1.0 - clipped**2)
    logpdf = logpdf_phi + log_jac
    outside = (x <= -1.0) | (x >= 1.0)
    logpdf[outside] = np.log(density_floor)
    return np.maximum(logpdf, np.log(density_floor))


# ============================================================
# Density models
# ============================================================


def gaussian_log_kernel_matrix(eval_points: np.ndarray, centers: np.ndarray, bandwidth: float) -> np.ndarray:
    eval_points = np.asarray(eval_points, dtype=float).reshape(-1, 1)
    centers = np.asarray(centers, dtype=float).reshape(1, -1)
    z = (eval_points - centers) / bandwidth
    return -0.5 * np.log(2.0 * np.pi) - np.log(bandwidth) - 0.5 * z**2



def gaussian_component_logpdf_1d(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    means = np.asarray(means, dtype=float).reshape(1, -1)
    variances = np.asarray(variances, dtype=float).reshape(1, -1)
    return -0.5 * (np.log(2.0 * np.pi * variances) + (x - means) ** 2 / variances)



def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a_max = np.max(a, axis=axis, keepdims=True)
    stabilized = a - a_max
    summed = np.sum(np.exp(stabilized), axis=axis, keepdims=True)
    out = a_max + np.log(summed)
    return np.squeeze(out, axis=axis)



def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out



def _softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _inverse_softplus(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    tiny = np.finfo(float).tiny
    y = np.maximum(y, tiny)
    out = np.empty_like(y, dtype=float)
    large = y > 20.0
    out[large] = y[large]
    out[~large] = np.log(np.expm1(y[~large]))
    return out


def _safe_log_probs(p: np.ndarray, floor: float = 1e-300) -> np.ndarray:
    return np.log(np.maximum(np.asarray(p, dtype=float), floor))


def silverman_bandwidth(samples: np.ndarray, bandwidth_floor: float = 1e-3) -> float:
    samples = np.asarray(samples, dtype=float).reshape(-1)
    n = samples.size
    if n < 2:
        return bandwidth_floor

    std = float(np.std(samples, ddof=1))
    iqr = float(np.subtract(*np.percentile(samples, [75.0, 25.0])))
    sigma = min(std, iqr / 1.34) if iqr > 0.0 else std
    if sigma <= 0.0:
        sigma = std if std > 0.0 else 1.0

    bw = 0.9 * sigma * n ** (-1.0 / 5.0)
    return max(float(bw), bandwidth_floor)



def fit_kde_1d(
    samples: np.ndarray,
    bandwidth: Optional[float] = None,
    bandwidth_scale: float = 1.0,
    bandwidth_floor: float = 1e-3,
) -> Dict[str, object]:
    samples = np.asarray(samples, dtype=float).reshape(-1)
    if samples.size == 0:
        raise ValueError("Need at least one sample to fit KDE.")

    if bandwidth is None:
        bandwidth = silverman_bandwidth(samples, bandwidth_floor=bandwidth_floor)
    bandwidth = max(float(bandwidth) * float(bandwidth_scale), bandwidth_floor)

    return {
        "model_type": "kde",
        "samples": samples.copy(),
        "bandwidth": bandwidth,
    }



def kde_logpdf_1d(
    x: np.ndarray,
    fit: Dict[str, object],
    density_floor: float = 1e-300,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    samples = np.asarray(fit["samples"], dtype=float).reshape(-1)
    bandwidth = float(fit["bandwidth"])

    log_terms = gaussian_log_kernel_matrix(x, samples, bandwidth)
    log_density = _logsumexp(log_terms, axis=1) - np.log(samples.size)
    return np.maximum(log_density, np.log(density_floor))



def fit_gmm2_1d(
    samples: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
    var_floor: float = 1e-6,
) -> Dict[str, object]:
    samples = np.asarray(samples, dtype=float).reshape(-1)
    if samples.size == 0:
        raise ValueError("Need at least one sample to fit GMM.")

    n = samples.size
    overall_var = max(float(np.var(samples)), var_floor)
    if n == 1:
        means = np.array([samples[0], samples[0]], dtype=float)
        variances = np.array([overall_var, overall_var], dtype=float)
        weights = np.array([0.5, 0.5], dtype=float)
        return {
            "model_type": "gmm2",
            "weights": weights,
            "means": means,
            "variances": variances,
            "log_likelihood": float(gaussian_component_logpdf_1d(samples, means[:1], variances[:1])[0, 0]),
            "n_iter": 0,
        }

    q25, q75 = np.percentile(samples, [25.0, 75.0])
    if q25 == q75:
        q25 = float(np.min(samples))
        q75 = float(np.max(samples))
    means = np.array([q25, q75], dtype=float)
    variances = np.full(2, overall_var, dtype=float)
    weights = np.array([0.5, 0.5], dtype=float)

    prev_log_likelihood = -np.inf
    n_iter_done = 0

    for n_iter in range(1, max_iter + 1):
        comp_logpdf = gaussian_component_logpdf_1d(samples, means, variances)
        weighted_logpdf = comp_logpdf + np.log(weights).reshape(1, -1)
        log_norm = _logsumexp(weighted_logpdf, axis=1)
        responsibilities = np.exp(weighted_logpdf - log_norm[:, None])

        nk = responsibilities.sum(axis=0)
        nk = np.maximum(nk, 1e-12)
        weights = nk / n
        means = (responsibilities * samples[:, None]).sum(axis=0) / nk
        centered_sq = (samples[:, None] - means.reshape(1, -1)) ** 2
        variances = (responsibilities * centered_sq).sum(axis=0) / nk
        variances = np.maximum(variances, var_floor)
        weights = np.maximum(weights, 1e-12)
        weights = weights / weights.sum()

        log_likelihood = float(np.sum(log_norm))
        n_iter_done = n_iter
        if abs(log_likelihood - prev_log_likelihood) <= tol * (1.0 + abs(log_likelihood)):
            prev_log_likelihood = log_likelihood
            break
        prev_log_likelihood = log_likelihood

    order = np.argsort(means)
    means = means[order]
    variances = variances[order]
    weights = weights[order]

    return {
        "model_type": "gmm2",
        "weights": weights,
        "means": means,
        "variances": variances,
        "log_likelihood": prev_log_likelihood,
        "n_iter": n_iter_done,
    }



def gmm2_logpdf_1d(
    x: np.ndarray,
    fit: Dict[str, object],
    density_floor: float = 1e-300,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    weights = np.asarray(fit["weights"], dtype=float).reshape(1, -1)
    means = np.asarray(fit["means"], dtype=float)
    variances = np.asarray(fit["variances"], dtype=float)
    comp_logpdf = gaussian_component_logpdf_1d(x, means, variances)
    log_density = _logsumexp(comp_logpdf + _safe_log_probs(weights), axis=1)
    return np.maximum(log_density, np.log(density_floor))



def _latent_gmm2_params_from_theta(
    theta: np.ndarray,
    latent_std_floor: float,
    obs_std_floor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    theta = np.asarray(theta, dtype=float)
    weight1 = float(_sigmoid(np.array([theta[0]]))[0])
    weights = np.array([weight1, 1.0 - weight1], dtype=float)
    means = np.array([theta[1], theta[2]], dtype=float)
    latent_stds = latent_std_floor + _softplus(theta[3:5])
    latent_variances = latent_stds**2
    obs_std = obs_std_floor + float(_softplus(np.array([theta[5]]))[0])
    obs_variance = obs_std**2
    return weights, means, latent_variances, obs_variance




def positive_log_transform(values: np.ndarray, transform_eps: float) -> np.ndarray:
    """Log transform for nonnegative statistics, phi = log(max(v, eps))."""
    values = np.asarray(values, dtype=float)
    clipped = np.maximum(values, transform_eps)
    return np.log(clipped)


def positive_log_gmm_logpdf_1d(
    x: np.ndarray,
    fit: Dict[str, object],
    density_floor: float = 1e-300,
) -> np.ndarray:
    """Log-density under a GMM in phi=log(v) space.

    The returned value is a density with respect to the original positive
    v-variable, so it includes the Jacobian |d log(v) / dv| = 1/v.
    In in/out likelihood ratios the Jacobian cancels, but including it makes
    raw-v histogram overlays meaningful.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    transform_eps = float(fit.get("transform_eps", 1e-12))
    clipped = np.maximum(x, transform_eps)
    phi = np.log(clipped)
    phi_fit = {
        "model_type": "gmm2",
        "weights": np.asarray(fit["weights"], dtype=float),
        "means": np.asarray(fit["means"], dtype=float),
        "variances": np.asarray(fit["latent_variances"], dtype=float),
    }
    logpdf_phi = gmm2_logpdf_1d(phi, phi_fit, density_floor=density_floor)
    log_jac = -np.log(clipped)
    logpdf = logpdf_phi + log_jac
    return np.maximum(logpdf, np.log(density_floor))


def fit_positive_log_gmm2_1d(
    samples: np.ndarray,
    transform_eps: float = 1e-12,
    max_iter: int = 200,
    tol: float = 1e-6,
    var_floor: float = 1e-6,
) -> Dict[str, object]:
    """Fit a two-component GMM after phi=log(max(v, eps)) transformation."""
    samples = np.asarray(samples, dtype=float).reshape(-1)
    phi_samples = positive_log_transform(samples, transform_eps=transform_eps)
    phi_gmm = fit_gmm2_1d(phi_samples, max_iter=max_iter, tol=tol, var_floor=var_floor)
    return {
        "model_type": "positive_log_gmm2",
        "weights": np.asarray(phi_gmm["weights"], dtype=float),
        "means": np.asarray(phi_gmm["means"], dtype=float),
        "latent_variances": np.asarray(phi_gmm["variances"], dtype=float),
        "obs_variance": 0.0,
        "optimizer_success": True,
        "optimizer_message": "GMM(2) fit in phi=log(max(v, eps)) space",
        "optimizer_n_iter": int(phi_gmm["n_iter"]),
        "optimizer_n_restarts": 1,
        "optimizer_objective": float(-phi_gmm["log_likelihood"]),
        "quad_order": 0,
        "transform_eps": transform_eps,
        "var_floor": var_floor,
        "obs_var_floor": 0.0,
        "quad_nodes": np.empty(0, dtype=float),
        "quad_weights": np.empty(0, dtype=float),
        "phi_fit_model": "gmm2",
        "transform_kind": "positive_log",
    }


def _latent_phi_gmm_logpdf_on_mu_grid(
    mu_nodes: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    latent_variances: np.ndarray,
    transform_eps: float,
    density_floor: float = 1e-300,
) -> np.ndarray:
    mu_nodes = np.asarray(mu_nodes, dtype=float).reshape(-1)
    clipped = np.clip(mu_nodes, -1.0 + transform_eps, 1.0 - transform_eps)
    phi_nodes = np.log((1.0 + clipped) / (1.0 - clipped))
    comp_logpdf = gaussian_component_logpdf_1d(phi_nodes, means, latent_variances)
    logpdf_phi = _logsumexp(comp_logpdf + _safe_log_probs(weights).reshape(1, -1), axis=1)
    log_jac = np.log(2.0) - np.log(1.0 - clipped**2)
    logpdf = logpdf_phi + log_jac
    outside = (mu_nodes <= -1.0) | (mu_nodes >= 1.0)
    logpdf[outside] = np.log(density_floor)
    return np.maximum(logpdf, np.log(density_floor))


def latent_gmm2_logpdf_fft_convolution(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    latent_variances: np.ndarray,
    obs_variance: float,
    transform_eps: float = 1e-8,
    fft_grid_size: int = 8193,
    fft_margin_stds: float = 8.0,
    density_floor: float = 1e-300,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if obs_variance <= 0.0:
        raise ValueError('fft_convolution requires strictly positive observation variance')

    obs_std = float(np.sqrt(obs_variance))
    lower = -1.0 + transform_eps
    upper = 1.0 - transform_eps

    grid_size = max(int(fft_grid_size), 1025)
    if grid_size % 2 == 0:
        grid_size += 1

    max_abs_x = float(np.max(np.abs(x))) if x.size > 0 else 0.0
    margin = max(float(fft_margin_stds) * obs_std, 0.1, max_abs_x - upper + 4.0 * obs_std)
    xmin = min(lower - margin, float(np.min(x)) - 4.0 * obs_std if x.size > 0 else lower - margin)
    xmax = max(upper + margin, float(np.max(x)) + 4.0 * obs_std if x.size > 0 else upper + margin)
    grid = np.linspace(xmin, xmax, grid_size)
    dx = float(grid[1] - grid[0])

    mu_density = np.zeros_like(grid)
    inside = (grid > lower) & (grid < upper)
    mu_density[inside] = np.exp(
        _latent_phi_gmm_logpdf_on_mu_grid(
            grid[inside],
            weights=weights,
            means=means,
            latent_variances=latent_variances,
            transform_eps=transform_eps,
            density_floor=density_floor,
        )
    )
    mu_area = float(np.trapezoid(mu_density, grid))
    if not np.isfinite(mu_area) or mu_area <= 0.0:
        return np.full(x.shape, np.log(density_floor), dtype=float)
    mu_density /= mu_area

    center = grid_size // 2
    offsets = (np.arange(grid_size) - center) * dx
    kernel = np.exp(-0.5 * (offsets / obs_std) ** 2) / (np.sqrt(2.0 * np.pi) * obs_std)
    kernel /= float(np.sum(kernel) * dx)

    conv_density = signal.fftconvolve(mu_density, kernel, mode='same') * dx
    conv_density = np.maximum(conv_density, density_floor)
    conv_area = float(np.trapezoid(conv_density, grid))
    if np.isfinite(conv_area) and conv_area > 0.0:
        conv_density /= conv_area

    density = np.interp(x, grid, conv_density, left=density_floor, right=density_floor)
    return np.log(np.maximum(density, density_floor))



def latent_gmm2_logpdf_from_params(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    latent_variances: np.ndarray,
    obs_variance: float,
    quad_nodes: np.ndarray,
    quad_weights: np.ndarray,
    density_floor: float = 1e-300,
    integration_method: str = "gauss_hermite",
    transform_eps: float = 1e-8,
    conv_nodes: Optional[np.ndarray] = None,
    conv_weights: Optional[np.ndarray] = None,
    fft_grid_size: int = 8193,
    fft_margin_stds: float = 8.0,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    means = np.asarray(means, dtype=float).reshape(-1)
    latent_variances = np.asarray(latent_variances, dtype=float).reshape(-1)

    if integration_method == "fft_convolution":
        return latent_gmm2_logpdf_fft_convolution(
            x,
            weights=weights,
            means=means,
            latent_variances=latent_variances,
            obs_variance=obs_variance,
            transform_eps=transform_eps,
            fft_grid_size=fft_grid_size,
            fft_margin_stds=fft_margin_stds,
            density_floor=density_floor,
        )

    if integration_method == "bounded_convolution":
        if conv_nodes is None or conv_weights is None:
            raise ValueError("bounded_convolution requires conv_nodes and conv_weights")
        conv_nodes = np.asarray(conv_nodes, dtype=float).reshape(-1)
        conv_weights = np.asarray(conv_weights, dtype=float).reshape(-1)
        lower = -1.0 + transform_eps
        upper = 1.0 - transform_eps
        mu_nodes = 0.5 * (upper - lower) * conv_nodes + 0.5 * (upper + lower)
        mu_weights = 0.5 * (upper - lower) * conv_weights
        logpdf_mu = _latent_phi_gmm_logpdf_on_mu_grid(
            mu_nodes,
            weights=weights,
            means=means,
            latent_variances=latent_variances,
            transform_eps=transform_eps,
            density_floor=density_floor,
        )
        log_kernel = gaussian_component_logpdf_1d(x, mu_nodes, np.full(mu_nodes.shape, obs_variance))
        log_density = _logsumexp(
            log_kernel + (_safe_log_probs(mu_weights) + logpdf_mu).reshape(1, -1),
            axis=1,
        )
        return np.maximum(log_density, np.log(density_floor))

    quad_nodes = np.asarray(quad_nodes, dtype=float).reshape(-1)
    quad_weights = np.asarray(quad_weights, dtype=float).reshape(-1)

    comp_logdens = []
    for k in range(2):
        latent_std = np.sqrt(latent_variances[k])
        phi_nodes = means[k] + np.sqrt(2.0) * latent_std * quad_nodes
        mu_nodes = np.tanh(phi_nodes / 2.0)
        logpdf_nodes = gaussian_component_logpdf_1d(x, mu_nodes, np.full(mu_nodes.shape, obs_variance))
        comp_logdens.append(_logsumexp(logpdf_nodes + _safe_log_probs(quad_weights).reshape(1, -1), axis=1) - 0.5 * np.log(np.pi))

    comp_logdens = np.column_stack(comp_logdens)
    log_density = _logsumexp(comp_logdens + _safe_log_probs(weights).reshape(1, -1), axis=1)
    return np.maximum(log_density, np.log(density_floor))



def _latent_gmm2_initial_phi(samples: np.ndarray, transform_eps: float) -> np.ndarray:
    samples = np.asarray(samples, dtype=float).reshape(-1)
    scale = max(1.0, float(np.quantile(np.abs(samples), 0.995)))
    scaled = np.clip(samples / scale, -1.0 + transform_eps, 1.0 - transform_eps)
    return np.log((1.0 + scaled) / (1.0 - scaled))



def fit_latent_gmm2_1d(
    samples: np.ndarray,
    sigma: float,
    transform_eps: float = 1e-8,
    max_iter: int = 200,
    tol: float = 1e-6,
    var_floor: float = 1e-6,
    obs_var_floor: float = 1e-6,
    quad_order: int = 40,
    n_restarts: int = 5,
    small_sigma_threshold: float = 0.2,
    small_sigma_method: str = "fft_convolution",
    fft_grid_size: int = 8193,
    fft_margin_stds: float = 8.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    samples = np.asarray(samples, dtype=float).reshape(-1)
    if samples.size == 0:
        raise ValueError("Need at least one sample to fit latent GMM(2).")

    if sigma == 0.0:
        phi_samples = transform_statistic(samples, sigma=0.0, transform_eps=transform_eps, asinh_scale=1.0)
        phi_gmm = fit_gmm2_1d(phi_samples, max_iter=max_iter, tol=tol, var_floor=var_floor)
        return {
            "model_type": "latent_gmm2",
            "sigma_zero_collapse": True,
            "weights": np.asarray(phi_gmm["weights"], dtype=float),
            "means": np.asarray(phi_gmm["means"], dtype=float),
            "latent_variances": np.asarray(phi_gmm["variances"], dtype=float),
            "obs_variance": 0.0,
            "optimizer_success": True,
            "optimizer_message": "collapsed to GMM(2) fit in phi-space because sigma=0",
            "optimizer_n_iter": int(phi_gmm["n_iter"]),
            "optimizer_n_restarts": 1,
            "optimizer_objective": float(-phi_gmm["log_likelihood"]),
            "quad_order": 0,
            "transform_eps": transform_eps,
            "var_floor": var_floor,
            "obs_var_floor": 0.0,
            "quad_nodes": np.empty(0, dtype=float),
            "quad_weights": np.empty(0, dtype=float),
            "phi_fit_model": "gmm2",
        }

    if quad_order < 5:
        raise ValueError("quad_order must be at least 5.")
    if rng is None:
        rng = np.random.default_rng(0)

    integration_method = small_sigma_method if sigma <= small_sigma_threshold else "gauss_hermite"
    quad_nodes, quad_weights = np.polynomial.hermite.hermgauss(quad_order)
    conv_nodes, conv_weights = np.polynomial.legendre.leggauss(quad_order)
    latent_std_floor = np.sqrt(var_floor)
    obs_std_floor = np.sqrt(obs_var_floor)

    phi_init = _latent_gmm2_initial_phi(samples, transform_eps=transform_eps)
    phi_gmm = fit_gmm2_1d(phi_init, max_iter=max_iter, tol=tol, var_floor=var_floor)

    clipped_samples = np.clip(samples, -1.0, 1.0)
    residual_scale = float(np.std(samples - clipped_samples))
    sample_std = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
    obs_std_guess = max(residual_scale, 0.05 * max(sample_std, 1.0), 5.0 * obs_std_floor)

    base_theta = np.array(
        [
            np.log(phi_gmm["weights"][0] / phi_gmm["weights"][1]),
            phi_gmm["means"][0],
            phi_gmm["means"][1],
            float(_inverse_softplus(np.array([max(np.sqrt(phi_gmm["variances"][0]) - latent_std_floor, 1e-12)]))[0]),
            float(_inverse_softplus(np.array([max(np.sqrt(phi_gmm["variances"][1]) - latent_std_floor, 1e-12)]))[0]),
            float(_inverse_softplus(np.array([max(obs_std_guess - obs_std_floor, 1e-12)]))[0]),
        ],
        dtype=float,
    )

    q25_phi, q75_phi = np.percentile(phi_init, [25.0, 75.0])
    rough_std = max(float(np.std(phi_init, ddof=1)) if phi_init.size > 1 else 1.0, latent_std_floor)
    alt_theta = np.array(
        [
            0.0,
            q25_phi,
            q75_phi,
            float(_inverse_softplus(np.array([max(rough_std - latent_std_floor, 1e-12)]))[0]),
            float(_inverse_softplus(np.array([max(rough_std - latent_std_floor, 1e-12)]))[0]),
            float(_inverse_softplus(np.array([max(max(obs_std_guess * 2.0, 10.0 * obs_std_floor) - obs_std_floor, 1e-12)]))[0]),
        ],
        dtype=float,
    )

    starts = [base_theta, alt_theta]
    for _ in range(max(0, n_restarts - 2)):
        jitter = np.array(
            [
                rng.normal(scale=0.5),
                rng.normal(scale=0.5),
                rng.normal(scale=0.5),
                rng.normal(scale=0.25),
                rng.normal(scale=0.25),
                rng.normal(scale=0.5),
            ],
            dtype=float,
        )
        starts.append(base_theta + jitter)

    def objective(theta: np.ndarray) -> float:
        weights, means, latent_variances, obs_variance = _latent_gmm2_params_from_theta(
            theta, latent_std_floor=latent_std_floor, obs_std_floor=obs_std_floor
        )
        if (not np.all(np.isfinite(weights)) or not np.all(np.isfinite(means)) or
                not np.all(np.isfinite(latent_variances)) or not np.isfinite(obs_variance) or
                np.any(latent_variances <= 0.0) or obs_variance <= 0.0):
            return float(np.inf)
        logpdf = latent_gmm2_logpdf_from_params(
            samples,
            weights=weights,
            means=means,
            latent_variances=latent_variances,
            obs_variance=obs_variance,
            quad_nodes=quad_nodes,
            quad_weights=quad_weights,
            integration_method=integration_method,
            transform_eps=transform_eps,
            conv_nodes=conv_nodes,
            conv_weights=conv_weights,
            fft_grid_size=fft_grid_size,
            fft_margin_stds=fft_margin_stds,
        )
        if not np.all(np.isfinite(logpdf)):
            return float(np.inf)
        return float(-np.sum(logpdf))

    best_res = None
    best_fun = np.inf
    for theta0 in starts:
        res = optimize.minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": tol},
        )
        if np.isfinite(res.fun) and res.fun < best_fun:
            best_fun = float(res.fun)
            best_res = res

    if best_res is None:
        raise RuntimeError("Latent GMM(2) fit failed for all restarts.")

    weights, means, latent_variances, obs_variance = _latent_gmm2_params_from_theta(
        best_res.x,
        latent_std_floor=latent_std_floor,
        obs_std_floor=obs_std_floor,
    )
    order = np.argsort(means)
    weights = weights[order]
    means = means[order]
    latent_variances = latent_variances[order]

    return {
        "model_type": "latent_gmm2",
        "sigma_zero_collapse": False,
        "weights": weights,
        "means": means,
        "latent_variances": latent_variances,
        "obs_variance": obs_variance,
        "optimizer_success": bool(best_res.success),
        "optimizer_message": str(best_res.message),
        "optimizer_n_iter": int(best_res.nit),
        "optimizer_n_restarts": len(starts),
        "optimizer_objective": float(best_res.fun),
        "quad_order": quad_order,
        "integration_method": integration_method,
        "small_sigma_threshold": small_sigma_threshold,
        "transform_eps": transform_eps,
        "var_floor": var_floor,
        "obs_var_floor": obs_var_floor,
        "quad_nodes": quad_nodes,
        "quad_weights": quad_weights,
        "conv_nodes": conv_nodes,
        "conv_weights": conv_weights,
        "fft_grid_size": int(fft_grid_size),
        "fft_margin_stds": float(fft_margin_stds),
        "phi_fit_model": "gmm2",
    }


def latent_gmm2_logpdf_1d(
    x: np.ndarray,
    fit: Dict[str, object],
    density_floor: float = 1e-300,
) -> np.ndarray:
    if bool(fit.get("sigma_zero_collapse", False)):
        phi_fit = {
            "model_type": "gmm2",
            "weights": np.asarray(fit["weights"], dtype=float),
            "means": np.asarray(fit["means"], dtype=float),
            "variances": np.asarray(fit["latent_variances"], dtype=float),
        }
        return gmm2_logpdf_1d(x, phi_fit, density_floor=density_floor)

    return latent_gmm2_logpdf_from_params(
        x,
        weights=np.asarray(fit["weights"], dtype=float),
        means=np.asarray(fit["means"], dtype=float),
        latent_variances=np.asarray(fit["latent_variances"], dtype=float),
        obs_variance=float(fit["obs_variance"]),
        quad_nodes=np.asarray(fit["quad_nodes"], dtype=float),
        quad_weights=np.asarray(fit["quad_weights"], dtype=float),
        density_floor=density_floor,
        integration_method=str(fit.get("integration_method", "gauss_hermite")),
        transform_eps=float(fit.get("transform_eps", 1e-8)),
        conv_nodes=np.asarray(fit.get("conv_nodes", np.empty(0)), dtype=float),
        conv_weights=np.asarray(fit.get("conv_weights", np.empty(0)), dtype=float),
        fft_grid_size=int(fit.get("fft_grid_size", 8193)),
        fft_margin_stds=float(fit.get("fft_margin_stds", 8.0)),
    )


def fit_density_1d(
    samples: np.ndarray,
    density_model: str,
    sigma: float,
    bandwidth_scale: float,
    gmm_max_iter: int,
    gmm_tol: float,
    gmm_var_floor: float,
    transform_eps: float,
    latent_obs_var_floor: float,
    latent_quad_order: int,
    latent_n_restarts: int,
    latent_small_sigma_threshold: float,
    latent_small_sigma_method: str,
    latent_fft_grid_size: int,
    latent_fft_margin_stds: float,
    rng: Optional[np.random.Generator],
) -> Dict[str, object]:
    if density_model == "kde":
        return fit_kde_1d(samples, bandwidth_scale=bandwidth_scale)
    if density_model == "gmm2":
        return fit_gmm2_1d(samples, max_iter=gmm_max_iter, tol=gmm_tol, var_floor=gmm_var_floor)
    if density_model == "latent_gmm2":
        return fit_latent_gmm2_1d(
            samples,
            sigma=sigma,
            transform_eps=transform_eps,
            max_iter=gmm_max_iter,
            tol=gmm_tol,
            var_floor=gmm_var_floor,
            obs_var_floor=latent_obs_var_floor,
            quad_order=latent_quad_order,
            n_restarts=latent_n_restarts,
            small_sigma_threshold=latent_small_sigma_threshold,
            small_sigma_method=latent_small_sigma_method,
            fft_grid_size=latent_fft_grid_size,
            fft_margin_stds=latent_fft_margin_stds,
            rng=rng,
        )
    raise ValueError(f"Unknown density_model: {density_model}")



def density_logpdf_1d(x: np.ndarray, fit: Dict[str, object]) -> np.ndarray:
    model_type = fit.get("model_type")
    if model_type == "kde":
        return kde_logpdf_1d(x, fit)
    if model_type == "gmm2":
        return gmm2_logpdf_1d(x, fit)
    if model_type == "latent_gmm2":
        return latent_gmm2_logpdf_1d(x, fit)
    if model_type == "positive_log_gmm2":
        return positive_log_gmm_logpdf_1d(x, fit)
    raise ValueError(f"Unknown fit model_type: {model_type}")



def lira_scores_1d(
    stats: np.ndarray,
    fit_in: Dict[str, object],
    fit_out: Dict[str, object],
) -> np.ndarray:
    return density_logpdf_1d(stats, fit_in) - density_logpdf_1d(stats, fit_out)



def auc_from_scores(scores_in: np.ndarray, scores_out: np.ndarray) -> float:
    scores_in = np.asarray(scores_in, dtype=float)
    scores_out = np.asarray(scores_out, dtype=float)
    combined = np.concatenate([scores_in, scores_out])
    labels = np.concatenate([np.ones_like(scores_in, dtype=int), np.zeros_like(scores_out, dtype=int)])
    order = np.argsort(combined)
    ranks = np.empty_like(order, dtype=float)

    i = 0
    while i < len(combined):
        j = i + 1
        while j < len(combined) and combined[order[j]] == combined[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j

    n_in = len(scores_in)
    n_out = len(scores_out)
    rank_sum_in = float(np.sum(ranks[labels == 1]))
    u_stat = rank_sum_in - n_in * (n_in + 1) / 2.0
    return u_stat / (n_in * n_out)



def tpr_at_fpr(scores_in: np.ndarray, scores_out: np.ndarray, fpr: float) -> float:
    if not (0.0 < fpr < 1.0):
        raise ValueError("fpr must lie in (0,1)")
    scores_in = np.asarray(scores_in, dtype=float)
    scores_out = np.asarray(scores_out, dtype=float)
    threshold = float(np.quantile(scores_out, 1.0 - fpr))
    return float(np.mean(scores_in >= threshold))


# ============================================================
# Sample-path mean/variance latent-GMM LiRA experiment
# ============================================================


def released_sample_mean_and_variance_at_x0(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x0: float,
    ell: float,
    r: float,
    sigma: float,
    jitter: float,
    n_posterior_draws: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    """
    Draw L scalar posterior values at x0 and return

        f_hat = (1/L) sum_l f_D^{(l)}(x0),
        v_hat = (1/(L sigma^2)) sum_l (f_D^{(l)}(x0) - f_hat)^2.

    Also returns the analytic posterior mean mu0 and normalized variance v0 used
    to generate the draws.
    """
    if n_posterior_draws < 1:
        raise ValueError("n_posterior_draws must be positive.")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("This f_hat/v_hat experiment requires finite sigma > 0.")

    mu0, v0 = posterior_mean_and_normalized_variance_at_x0(
        x_train=x_train,
        y_train=y_train,
        x0=x0,
        ell=ell,
        r=r,
        jitter=jitter,
    )

    draws = mu0 + sigma * np.sqrt(max(v0, 0.0)) * rng.normal(size=n_posterior_draws)
    f_hat = float(np.mean(draws))

    # For L = 1 the empirical variance around the sample mean is not a usable
    # statistic for the attack. We mark it as NaN and skip all v_hat fitting,
    # scoring, plotting, and reporting downstream.
    if n_posterior_draws == 1:
        v_hat = np.nan
    else:
        v_hat = float(np.sum((draws - f_hat) ** 2) / (n_posterior_draws * sigma**2))
        v_hat = max(v_hat, 0.0)

    return f_hat, v_hat, float(mu0), float(v0)


def sample_shadow_fhat_and_vhat(
    n_shadow: int,
    n: int,
    x0: float,
    ell: float,
    r: float,
    sigma: float,
    jitter: float,
    m_eps: float,
    n_posterior_draws: int,
    rng: np.random.Generator,
    membership: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample f_hat, v_hat, analytic mu0, and analytic v0 under in/out."""
    f_hats = np.empty(n_shadow, dtype=float)
    v_hats = np.empty(n_shadow, dtype=float)
    mu0s = np.empty(n_shadow, dtype=float)
    v0s = np.empty(n_shadow, dtype=float)

    for i in range(n_shadow):
        if membership == "in":
            x_train, y_train = draw_dataset_in(n=n, x0=x0, m_eps=m_eps, rng=rng)
        elif membership == "out":
            x_train, y_train = draw_dataset_out(n=n, m_eps=m_eps, rng=rng)
        else:
            raise ValueError("membership must be 'in' or 'out'.")

        f_hat, v_hat, mu0, v0 = released_sample_mean_and_variance_at_x0(
            x_train=x_train,
            y_train=y_train,
            x0=x0,
            ell=ell,
            r=r,
            sigma=sigma,
            jitter=jitter,
            n_posterior_draws=n_posterior_draws,
            rng=rng,
        )
        f_hats[i] = f_hat
        v_hats[i] = v_hat
        mu0s[i] = mu0
        v0s[i] = v0

    return f_hats, v_hats, mu0s, v0s


def latent_lira_scores_1d(stats: np.ndarray, fit_in: Dict[str, object], fit_out: Dict[str, object]) -> np.ndarray:
    return density_logpdf_1d(stats, fit_in) - density_logpdf_1d(stats, fit_out)


@dataclass
class Config:
    n: int = 10
    x0: float = 0.5
    ell: float = 0.2
    r: float = 0.5
    sigma: float = 0.5
    m_eps: float = 0.0
    n_posterior_draws: int = 10
    n_shadow: int = 10000
    n_eval: int = 10000
    jitter: float = 1e-10
    transform_eps: float = 1e-8
    bins: int = 60
    gmm_max_iter: int = 200
    gmm_tol: float = 1e-6
    gmm_var_floor: float = 1e-6
    latent_obs_var_floor: float = 1e-6
    latent_quad_order: int = 40
    latent_n_restarts: int = 16
    latent_small_sigma_threshold: float = 0.5
    latent_small_sigma_method: str = "fft_convolution"
    latent_fft_grid_size: int = 8193
    latent_fft_margin_stds: float = 8.0
    seed: int = 0
    save_dir: str = "lira_fhat_vhat_logv_gmm_results"
    plot_name: str = "fhat_vhat_logv_gmm_hist.png"
    show_plot: bool = False
    no_save: bool = False


def fit_latent_density_for_stat(samples: np.ndarray, cfg: Config, rng: np.random.Generator) -> Dict[str, object]:
    """Fit the f_hat marginal with the original latent-GMM model.

    We deliberately avoid the FFT convolution evaluator for f_hat here.  At
    small r the FFT/interpolation path can produce small numerical ripples in
    the plotted density.  The bounded-convolution evaluator is slower but gives
    a smooth density curve while preserving the same latent model

        f_hat = tanh(phi/2) + psi,    phi ~ GMM(2), psi Gaussian.
    """
    return fit_latent_gmm2_1d(
        samples,
        sigma=cfg.sigma,
        transform_eps=cfg.transform_eps,
        max_iter=cfg.gmm_max_iter,
        tol=cfg.gmm_tol,
        var_floor=cfg.gmm_var_floor,
        obs_var_floor=cfg.latent_obs_var_floor,
        quad_order=cfg.latent_quad_order,
        n_restarts=cfg.latent_n_restarts,
        small_sigma_threshold=np.inf,
        small_sigma_method="bounded_convolution",
        fft_grid_size=cfg.latent_fft_grid_size,
        fft_margin_stds=cfg.latent_fft_margin_stds,
        rng=rng,
    )


def fit_latent_density_for_vhat(samples: np.ndarray, cfg: Config) -> Dict[str, object]:
    """Fit the v-hat marginal using phi=log(max(v, eps)).

    This is appropriate for the empirical sample-variance statistic, whose
    support is nonnegative rather than confined to [-1, 1] or [0, 1].
    """
    return fit_positive_log_gmm2_1d(
        samples,
        transform_eps=cfg.transform_eps,
        max_iter=cfg.gmm_max_iter,
        tol=cfg.gmm_tol,
        var_floor=cfg.gmm_var_floor,
    )


def _metrics(scores_in: np.ndarray, scores_out: np.ndarray) -> Dict[str, float]:
    return {
        "auc": auc_from_scores(scores_in, scores_out),
        "tpr_at_1pct_fpr": tpr_at_fpr(scores_in, scores_out, 0.01),
        "tpr_at_5pct_fpr": tpr_at_fpr(scores_in, scores_out, 0.05),
        "tpr_at_10pct_fpr": tpr_at_fpr(scores_in, scores_out, 0.10),
    }


def run_experiment(cfg: Config) -> Dict[str, object]:
    rng = np.random.default_rng(cfg.seed)

    if not np.isfinite(cfg.sigma) or cfg.sigma <= 0.0:
        raise ValueError("This experiment requires finite --sigma > 0 because v_hat divides by sigma^2.")

    use_vhat = cfg.n_posterior_draws > 1

    fhat_in, vhat_in, mu0_in, v0_in = sample_shadow_fhat_and_vhat(
        n_shadow=cfg.n_shadow,
        n=cfg.n,
        x0=cfg.x0,
        ell=cfg.ell,
        r=cfg.r,
        sigma=cfg.sigma,
        jitter=cfg.jitter,
        m_eps=cfg.m_eps,
        n_posterior_draws=cfg.n_posterior_draws,
        rng=rng,
        membership="in",
    )
    fhat_out, vhat_out, mu0_out, v0_out = sample_shadow_fhat_and_vhat(
        n_shadow=cfg.n_shadow,
        n=cfg.n,
        x0=cfg.x0,
        ell=cfg.ell,
        r=cfg.r,
        sigma=cfg.sigma,
        jitter=cfg.jitter,
        m_eps=cfg.m_eps,
        n_posterior_draws=cfg.n_posterior_draws,
        rng=rng,
        membership="out",
    )

    fit_fhat_in = fit_latent_density_for_stat(fhat_in, cfg, rng)
    fit_fhat_out = fit_latent_density_for_stat(fhat_out, cfg, rng)

    if use_vhat:
        fit_vhat_in = fit_latent_density_for_vhat(vhat_in, cfg)
        fit_vhat_out = fit_latent_density_for_vhat(vhat_out, cfg)
    else:
        fit_vhat_in = None
        fit_vhat_out = None

    eval_fhat_in, eval_vhat_in, eval_mu0_in, eval_v0_in = sample_shadow_fhat_and_vhat(
        n_shadow=cfg.n_eval,
        n=cfg.n,
        x0=cfg.x0,
        ell=cfg.ell,
        r=cfg.r,
        sigma=cfg.sigma,
        jitter=cfg.jitter,
        m_eps=cfg.m_eps,
        n_posterior_draws=cfg.n_posterior_draws,
        rng=rng,
        membership="in",
    )
    eval_fhat_out, eval_vhat_out, eval_mu0_out, eval_v0_out = sample_shadow_fhat_and_vhat(
        n_shadow=cfg.n_eval,
        n=cfg.n,
        x0=cfg.x0,
        ell=cfg.ell,
        r=cfg.r,
        sigma=cfg.sigma,
        jitter=cfg.jitter,
        m_eps=cfg.m_eps,
        n_posterior_draws=cfg.n_posterior_draws,
        rng=rng,
        membership="out",
    )

    scores_fhat_in = latent_lira_scores_1d(eval_fhat_in, fit_fhat_in, fit_fhat_out)
    scores_fhat_out = latent_lira_scores_1d(eval_fhat_out, fit_fhat_in, fit_fhat_out)

    metrics_fhat = _metrics(scores_fhat_in, scores_fhat_out)

    if use_vhat:
        scores_vhat_in = latent_lira_scores_1d(eval_vhat_in, fit_vhat_in, fit_vhat_out)
        scores_vhat_out = latent_lira_scores_1d(eval_vhat_out, fit_vhat_in, fit_vhat_out)
        scores_sum_in = scores_fhat_in + scores_vhat_in
        scores_sum_out = scores_fhat_out + scores_vhat_out
        metrics_vhat = _metrics(scores_vhat_in, scores_vhat_out)
        metrics_sum = _metrics(scores_sum_in, scores_sum_out)
    else:
        scores_vhat_in = np.full_like(scores_fhat_in, np.nan, dtype=float)
        scores_vhat_out = np.full_like(scores_fhat_out, np.nan, dtype=float)
        scores_sum_in = scores_fhat_in.copy()
        scores_sum_out = scores_fhat_out.copy()
        metrics_vhat = None
        metrics_sum = metrics_fhat.copy()

    return {
        "config": cfg,
        "use_vhat": use_vhat,
        "fhat_in": fhat_in,
        "fhat_out": fhat_out,
        "vhat_in": vhat_in,
        "vhat_out": vhat_out,
        "mu0_in": mu0_in,
        "mu0_out": mu0_out,
        "v0_in": v0_in,
        "v0_out": v0_out,
        "fit_fhat_in": fit_fhat_in,
        "fit_fhat_out": fit_fhat_out,
        "fit_vhat_in": fit_vhat_in,
        "fit_vhat_out": fit_vhat_out,
        "eval_fhat_in": eval_fhat_in,
        "eval_fhat_out": eval_fhat_out,
        "eval_vhat_in": eval_vhat_in,
        "eval_vhat_out": eval_vhat_out,
        "eval_mu0_in": eval_mu0_in,
        "eval_mu0_out": eval_mu0_out,
        "eval_v0_in": eval_v0_in,
        "eval_v0_out": eval_v0_out,
        "scores_fhat_in": scores_fhat_in,
        "scores_fhat_out": scores_fhat_out,
        "scores_vhat_in": scores_vhat_in,
        "scores_vhat_out": scores_vhat_out,
        "scores_sum_in": scores_sum_in,
        "scores_sum_out": scores_sum_out,
        "metrics_fhat": metrics_fhat,
        "metrics_vhat": metrics_vhat,
        "metrics_sum": metrics_sum,
    }

def summarize(name: str, values: np.ndarray) -> str:
    values = np.asarray(values, dtype=float)
    q05, q50, q95 = np.quantile(values, [0.05, 0.50, 0.95])
    return (
        f"{name}: mean={np.mean(values):.6g}, std={np.std(values):.6g}, "
        f"q05={q05:.6g}, median={q50:.6g}, q95={q95:.6g}"
    )


def format_latent_gmm(name: str, fit: Dict[str, object]) -> str:
    weights = np.asarray(fit["weights"], dtype=float)
    means = np.asarray(fit["means"], dtype=float)
    variances = np.asarray(fit["latent_variances"], dtype=float)
    obs_var = float(fit["obs_variance"])
    method = str(fit.get("integration_method", fit.get("transform_kind", "sigma_zero_collapse")))
    model_name = "log-GMM(2)" if fit.get("model_type") == "positive_log_gmm2" else "latent GMM(2)"
    return (
        f"{name} {model_name}: "
        f"weights=[{weights[0]:.6g}, {weights[1]:.6g}], "
        f"latent means=[{means[0]:.6g}, {means[1]:.6g}], "
        f"latent stds=[{np.sqrt(variances[0]):.6g}, {np.sqrt(variances[1]):.6g}], "
        f"obs std={np.sqrt(obs_var):.6g}, method={method}, "
        f"success={fit.get('optimizer_success', True)}"
    )


def print_metrics(title: str, metrics: Dict[str, float], final=False) -> None:
    print(title)
    print(f"  AUC           = {metrics['auc']:.6f}")
    print(f"  TPR @ 1% FPR  = {metrics['tpr_at_1pct_fpr']:.6f}")
    print(f"  TPR @ 5% FPR  = {metrics['tpr_at_5pct_fpr']:.6f}")
    print(f"  TPR @ 10% FPR = {metrics['tpr_at_10pct_fpr']:.6f}")
    
    if final:
    	save_dir = "lira_exp1D_results"
    	os.makedirs(save_dir, exist_ok=True)
    	filename = "lira_r"+f"{cfg.r:.3f}"+"_sigma"+f"{cfg.sigma}"+"_L"+f"{cfg.n_posterior_draws}_seed"+f"{cfg.seed}"
    	roc_results = [metrics['auc'],metrics['tpr_at_1pct_fpr'],metrics['tpr_at_5pct_fpr'],metrics['tpr_at_10pct_fpr']]
    	roc_results = np.array(roc_results)
    	np.save(save_dir+"/"+filename, roc_results)
    	print("Saved "+filename)


def print_result(res: Dict[str, object]) -> None:
    cfg = res["config"]
    use_vhat = bool(res.get("use_vhat", cfg.n_posterior_draws > 1))

    print("LiRA-like posterior-sample mean + sample-variance experiment")
    print(
        f"n={cfg.n}, x0={cfg.x0}, ell={cfg.ell}, r={cfg.r}, sigma={cfg.sigma}, "
        f"m_eps={cfg.m_eps}, L={cfg.n_posterior_draws}, "
        f"n_shadow={cfg.n_shadow}, n_eval={cfg.n_eval}, seed={cfg.seed}"
    )
    print("Statistics:")
    print("  f_hat(x0) = L^{-1} sum_l f_D^{(l)}(x0)")
    if use_vhat:
        print("  v_hat(x0) = (L sigma^2)^{-1} sum_l (f_D^{(l)}(x0) - f_hat(x0))^2")
        print("Combined score: LLR_f_hat + LLR_v_hat")
    else:
        print("  L = 1, so v_hat is undefined; using the f_hat-only attack.")
        print("Combined score: same as LLR_f_hat")
    print("Density model: latent_gmm2 for f_hat; unit-interval logit GMM for v_hat, fitted separately for each in/out marginal.")
    print()
    print(summarize("fhat_in", res["fhat_in"]))
    print(summarize("fhat_out", res["fhat_out"]))
    if use_vhat:
        print(summarize("vhat_in", res["vhat_in"]))
        print(summarize("vhat_out", res["vhat_out"]))
    print()
    print(format_latent_gmm("fhat in", res["fit_fhat_in"]))
    print(format_latent_gmm("fhat out", res["fit_fhat_out"]))
    if use_vhat:
        print(format_latent_gmm("vhat in", res["fit_vhat_in"]))
        print(format_latent_gmm("vhat out", res["fit_vhat_out"]))
    print()
    if use_vhat:
        print_metrics("f_hat-only attack metrics:", res["metrics_fhat"])
        print()
        print_metrics("v_hat-only attack metrics:", res["metrics_vhat"])
        print()
        print_metrics("Combined sum-of-LLRs attack metrics:", res["metrics_sum"], final=True)
    else:
        print_metrics("f_hat-only attack metrics:", res["metrics_fhat"], final=True)
        print()
        print("v_hat-only and combined f_hat+v_hat attacks skipped because L = 1.")
def plot_histograms(res: Dict[str, object]) -> Tuple[object, np.ndarray]:
    if plt is None:
        raise RuntimeError("matplotlib is not available.")

    cfg = res["config"]
    use_vhat = bool(res.get("use_vhat", cfg.n_posterior_draws > 1))

    if use_vhat:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        axes = np.asarray(axes).reshape(-1)
        panels = [
            (
                axes[0],
                np.asarray(res["fhat_in"], dtype=float),
                np.asarray(res["fhat_out"], dtype=float),
                res["fit_fhat_in"],
                res["fit_fhat_out"],
                r"$\hat f_D(x_0)$",
                "Sample-path average statistic",
            ),
            (
                axes[1],
                np.asarray(res["vhat_in"], dtype=float),
                np.asarray(res["vhat_out"], dtype=float),
                res["fit_vhat_in"],
                res["fit_vhat_out"],
                r"$\hat v_D(x_0)$",
                "Normalized sample-variance statistic",
            ),
        ]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.3))
        axes = np.asarray([ax])
        panels = [
            (
                ax,
                np.asarray(res["fhat_in"], dtype=float),
                np.asarray(res["fhat_out"], dtype=float),
                res["fit_fhat_in"],
                res["fit_fhat_out"],
                r"$\hat f_D(x_0)$",
                "Sample-path statistic; $L=1$",
            ),
        ]

    for ax, vals_in, vals_out, fit_in, fit_out, xlabel, title in panels:
        # For v_hat, the model is fitted in phi=log(v) space, but we plot the
        # histogram in raw v-space. density_logpdf_1d includes the 1/v Jacobian,
        # so the overlay is a proper raw-v density.
        plot_in = vals_in
        plot_out = vals_out
        xlabel_plot = xlabel
        title_plot = title

        xmin = min(float(np.min(plot_in)), float(np.min(plot_out)))
        xmax = max(float(np.max(plot_in)), float(np.max(plot_out)))

        if fit_in.get("model_type") == "positive_log_gmm2":
            # v_hat panel: raw v-space is nonnegative and the fitted log-GMM
            # density includes the 1/v Jacobian.
            lower = max(cfg.transform_eps, min(max(0.0, xmin), xmax))
            pad = 0.05 * max(xmax - lower, 1e-8)
            xs = np.linspace(lower, xmax + pad, 800)
            fit_label = "log-GMM(2)"
        else:
            # f_hat panel: raw f-space is signed.  Do not clip the grid at zero.
            pad = 0.05 * max(xmax - xmin, 1e-8)
            xs = np.linspace(xmin - pad, xmax + pad, 800)
            fit_label = "latent GMM(2)"

        dens_in = np.exp(density_logpdf_1d(xs, fit_in))
        dens_out = np.exp(density_logpdf_1d(xs, fit_out))

        ax.hist(plot_in, bins=cfg.bins, density=True, alpha=0.45, label="in")
        ax.hist(plot_out, bins=cfg.bins, density=True, alpha=0.45, label="out")
        ax.plot(xs, dens_in, linewidth=2.0, label=f"{fit_label} fit in")
        ax.plot(xs, dens_out, linewidth=2.0, label=f"{fit_label} fit out")
        ax.set_xlabel(xlabel_plot)
        ax.set_ylabel("density")
        ax.set_title(title_plot)
        ax.grid(True, alpha=0.25)

    axes[0].legend(frameon=False)
    fig.suptitle(
        rf"$x_0={cfg.x0}$, $n={cfg.n}$, $\ell={cfg.ell}$, $r={cfg.r}$, "
        rf"$\sigma={cfg.sigma}$, $L={cfg.n_posterior_draws}$"
    )
    fig.tight_layout()
    return fig, axes


def _fit_arrays_for_save(prefix: str, fit: Dict[str, object]) -> Dict[str, np.ndarray]:
    out = {
        f"{prefix}_weights": np.asarray(fit["weights"], dtype=float),
        f"{prefix}_means": np.asarray(fit["means"], dtype=float),
        f"{prefix}_latent_variances": np.asarray(fit["latent_variances"], dtype=float),
        f"{prefix}_obs_variance": np.array(float(fit["obs_variance"])),
    }
    return out


def save_outputs(res: Dict[str, object]) -> None:
    cfg = res["config"]
    use_vhat = bool(res.get("use_vhat", cfg.n_posterior_draws > 1))
    os.makedirs(cfg.save_dir, exist_ok=True)

    stem = (
        f"fhat_vhat_latent_n{cfg.n}_x0{cfg.x0:g}_ell{cfg.ell:g}_r{cfg.r:g}_"
        f"sigma{cfg.sigma:g}_L{cfg.n_posterior_draws}_meps{cfg.m_eps:g}_seed{cfg.seed}"
    )
    data_path = os.path.join(cfg.save_dir, stem + ".npz")

    fit_payload = {}
    fit_payload.update(_fit_arrays_for_save("fit_fhat_in", res["fit_fhat_in"]))
    fit_payload.update(_fit_arrays_for_save("fit_fhat_out", res["fit_fhat_out"]))
    if use_vhat:
        fit_payload.update(_fit_arrays_for_save("fit_vhat_in", res["fit_vhat_in"]))
        fit_payload.update(_fit_arrays_for_save("fit_vhat_out", res["fit_vhat_out"]))

    metrics_vhat = np.full(4, np.nan, dtype=float)
    if use_vhat:
        metrics_vhat = np.array([
            res["metrics_vhat"]["auc"],
            res["metrics_vhat"]["tpr_at_1pct_fpr"],
            res["metrics_vhat"]["tpr_at_5pct_fpr"],
            res["metrics_vhat"]["tpr_at_10pct_fpr"],
        ])

    np.savez_compressed(
        data_path,
        use_vhat=np.array(use_vhat),
        fhat_in=res["fhat_in"],
        fhat_out=res["fhat_out"],
        vhat_in=res["vhat_in"],
        vhat_out=res["vhat_out"],
        mu0_in=res["mu0_in"],
        mu0_out=res["mu0_out"],
        v0_in=res["v0_in"],
        v0_out=res["v0_out"],
        eval_fhat_in=res["eval_fhat_in"],
        eval_fhat_out=res["eval_fhat_out"],
        eval_vhat_in=res["eval_vhat_in"],
        eval_vhat_out=res["eval_vhat_out"],
        scores_fhat_in=res["scores_fhat_in"],
        scores_fhat_out=res["scores_fhat_out"],
        scores_vhat_in=res["scores_vhat_in"],
        scores_vhat_out=res["scores_vhat_out"],
        scores_sum_in=res["scores_sum_in"],
        scores_sum_out=res["scores_sum_out"],
        metrics_fhat=np.array([
            res["metrics_fhat"]["auc"],
            res["metrics_fhat"]["tpr_at_1pct_fpr"],
            res["metrics_fhat"]["tpr_at_5pct_fpr"],
            res["metrics_fhat"]["tpr_at_10pct_fpr"],
        ]),
        metrics_vhat=metrics_vhat,
        metrics_sum=np.array([
            res["metrics_sum"]["auc"],
            res["metrics_sum"]["tpr_at_1pct_fpr"],
            res["metrics_sum"]["tpr_at_5pct_fpr"],
            res["metrics_sum"]["tpr_at_10pct_fpr"],
        ]),
        **fit_payload,
    )
    print(f"Saved data to {data_path}")

    if plt is not None:
        fig, _ = plot_histograms(res)
        plot_path = os.path.join(cfg.save_dir, cfg.plot_name)
        fig.savefig(plot_path, bbox_inches="tight", dpi=200)
        print(f"Saved plot to {plot_path}")
        if not cfg.show_plot:
            plt.close(fig)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "LiRA-like attack using two statistics from L posterior draws at x0: "
            "f_hat and normalized sample variance v_hat. Fits latent_gmm2 separately "
            "to in/out histograms of each statistic and scores by the sum of marginal LLRs."
        )
    )
    parser.add_argument("--n", type=int, default=10, help="Training set size")
    parser.add_argument("--x0", type=float, default=0.5, help="Candidate point for membership inference")
    parser.add_argument("--ell", type=float, default=0.2, help="Exponential-kernel lengthscale")
    parser.add_argument("--r", type=float, default=0.5, help="Regularization parameter")
    parser.add_argument("--sigma", type=float, default=0.5, help="Finite posterior sample-path scale; must be > 0")
    parser.add_argument("--m-eps", type=float, default=0.0, help="Label-noise level")
    parser.add_argument("--n-posterior-draws", type=int, default=1, help="Number L of posterior draws")
    parser.add_argument("--n-shadow", type=int, default=10000, help="Number of shadow datasets per hypothesis")
    parser.add_argument("--n-eval", type=int, default=10000, help="Number of evaluation datasets per hypothesis")
    parser.add_argument("--jitter", type=float, default=1e-10, help="Numerical jitter for matrix solves")
    parser.add_argument("--transform-eps", type=float, default=1e-8, help="Clipping epsilon used by latent GMM initialisation")
    parser.add_argument("--bins", type=int, default=60, help="Number of histogram bins")
    parser.add_argument("--gmm-max-iter", type=int, default=200, help="Maximum optimizer iterations")
    parser.add_argument("--gmm-tol", type=float, default=1e-6, help="Relative tolerance for fitting")
    parser.add_argument("--gmm-var-floor", type=float, default=1e-6, help="Minimum latent/component variance")
    parser.add_argument("--latent-obs-var-floor", type=float, default=1e-6, help="Minimum observation-noise variance")
    parser.add_argument("--latent-quad-order", type=int, default=40, help="Gauss-Hermite quadrature order")
    parser.add_argument("--latent-n-restarts", type=int, default=16, help="Number of latent_gmm2 optimizer restarts")
    parser.add_argument(
        "--latent-small-sigma-threshold",
        type=float,
        default=0.5,
        help="Use the small-sigma evaluator when sigma is at or below this threshold",
    )
    parser.add_argument(
        "--latent-small-sigma-method",
        type=str,
        choices=["fft_convolution", "bounded_convolution"],
        default="fft_convolution",
        help="Evaluator used by latent_gmm2 in the small-sigma regime",
    )
    parser.add_argument("--latent-fft-grid-size", type=int, default=8193, help="FFT-style convolution grid size")
    parser.add_argument("--latent-fft-margin-stds", type=float, default=8.0, help="FFT convolution grid padding")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="lira_fhat_vhat_logv_gmm_results", help="Output directory")
    parser.add_argument("--plot-name", type=str, default="fhat_vhat_logv_gmm_hist.png", help="Saved plot filename")
    parser.add_argument("--show-plot", action="store_true", help="Show histogram plot interactively")
    parser.add_argument("--no-save", action="store_true", help="Do not save .npz or plot files")

    args = parser.parse_args()
    return Config(
        n=args.n,
        x0=args.x0,
        ell=args.ell,
        r=args.r,
        sigma=args.sigma,
        m_eps=args.m_eps,
        n_posterior_draws=args.n_posterior_draws,
        n_shadow=args.n_shadow,
        n_eval=args.n_eval,
        jitter=args.jitter,
        transform_eps=args.transform_eps,
        bins=args.bins,
        gmm_max_iter=args.gmm_max_iter,
        gmm_tol=args.gmm_tol,
        gmm_var_floor=args.gmm_var_floor,
        latent_obs_var_floor=args.latent_obs_var_floor,
        latent_quad_order=args.latent_quad_order,
        latent_n_restarts=args.latent_n_restarts,
        latent_small_sigma_threshold=args.latent_small_sigma_threshold,
        latent_small_sigma_method=args.latent_small_sigma_method,
        latent_fft_grid_size=args.latent_fft_grid_size,
        latent_fft_margin_stds=args.latent_fft_margin_stds,
        seed=args.seed,
        save_dir=args.save_dir,
        plot_name=args.plot_name,
        show_plot=args.show_plot,
        no_save=args.no_save,
    )


if __name__ == "__main__":
    cfg = parse_args()
    result = run_experiment(cfg)
    print_result(result)
    if not cfg.no_save:
        save_outputs(result)
    elif cfg.show_plot:
        fig, _ = plot_histograms(result)
        plt.savefig('lira_exp1D_results/lira_hist_L'+str(cfg.n_posterior_draws)+'.pdf')
