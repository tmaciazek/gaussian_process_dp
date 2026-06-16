#!/usr/bin/env python3
"""
Coarse-to-refined non-private searches for the exponential-kernel excursion-set
experiment on [0,1].

This script runs TWO separate utility searches for the same fixed latent target
f_* and the same repeated noisy datasets:

1. Posterior distribution search over theta=(ell_model, r, sigma).
   The posterior distribution is evaluated through the pointwise posterior
   excursion probability

       p_D(x; theta) = P(F(x) > u | D, theta)
                     = Phi((mu_D(x)-u)/(sigma sqrt(k_D(x,x)))).

   The objective is integrated binary cross-entropy (BCE) against the true
   excursion indicator s_*(x)=1{f_*(x)>u}. IoU, volume error, and effective
   dimension d_eff=tr[K(K+r^2I)^{-1}] are reported alongside the BCE-selected
   candidates, but are not used as posterior-distribution selection objectives.

2. Posterior-mean plug-in search over theta_mu=(ell_model, r) ONLY.
   The posterior mean induces the hard excursion set

       Omega_mu(D) = {x : mu_D(x) > u}.

   The objective is weighted IoU/Jaccard overlap with the true excursion set.
   No sigma grid is used for this search, since mu_D does not depend on sigma.

Analytic note: candidate evaluation uses posterior means and covariance
DIAGONALS for the posterior-distribution BCE search. It does not draw posterior
sample paths, and it does not compute finite-L Monte Carlo corrections.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None

try:
    from scipy.linalg import cho_factor as _cho_factor, cho_solve as _cho_solve
except Exception:  # pragma: no cover - fallback for minimal environments
    _cho_factor = None
    _cho_solve = None

try:
    from scipy.special import ndtr as _normal_cdf
except Exception:  # pragma: no cover - fallback for minimal environments
    from math import erf, sqrt

    def _normal_cdf(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        flat = z.ravel()
        vals = np.array([0.5 * (1.0 + erf(float(v) / sqrt(2.0))) for v in flat], dtype=float)
        return vals.reshape(z.shape)


try:
    from dp_utils import epsilon_for_delta_beta_exp_1d
except ImportError as exc:  # pragma: no cover - fail only if DP-aware refinement is used
    epsilon_for_delta_beta_exp_1d = None
    _DP_UTILS_IMPORT_ERROR = exc
else:
    _DP_UTILS_IMPORT_ERROR = None


# ----------------------------
# Kernel and linear algebra
# ----------------------------

def exp_kernel(x: np.ndarray, z: np.ndarray, ell: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    z = np.asarray(z, dtype=float).reshape(1, -1)
    return np.exp(-np.abs(x - z) / ell)


def symmetrize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def gaussian_factor(cov: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    """Return L such that L @ z has approximately covariance cov."""
    cov = symmetrize(cov)
    try:
        return np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(cov)
        w = np.maximum(w, 0.0)
        return v @ np.diag(np.sqrt(w))


def sample_gaussian(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    factor = gaussian_factor(cov)
    z = rng.standard_normal(mean.shape[0])
    return mean + factor @ z


def solve_spd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a x = b for SPD a using Cholesky when available."""
    if _cho_factor is not None and _cho_solve is not None:
        c_and_lower = _cho_factor(a, lower=True, check_finite=False, overwrite_a=False)
        return _cho_solve(c_and_lower, b, check_finite=False)

    # NumPy fallback.
    L = np.linalg.cholesky(a)
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


def effective_dimension_from_kernel(k_xx: np.ndarray, r: float) -> float:
    """Effective dimension d_eff = tr[K(K+r^2 I)^(-1)]."""
    if r <= 0:
        raise ValueError("r must be positive")
    eigvals = np.linalg.eigvalsh(symmetrize(k_xx))
    eigvals = np.maximum(eigvals, 0.0)
    return float(np.sum(eigvals / (eigvals + r ** 2)))


# ----------------------------
# True fixed latent target f_*
# ----------------------------

def sample_fixed_target_on_grid(
    x_grid: np.ndarray,
    ell_true: float,
    m_eps: float,
    rng: np.random.Generator,
    jitter: float = 1e-10,
) -> np.ndarray:
    """
    Sample one fixed target path on x_grid from a unit-scale exponential-kernel
    GP and rescale so that max |f_*| = 1 - m_eps.
    """
    if not (0.0 <= m_eps < 1.0):
        raise ValueError("m_eps must satisfy 0 <= m_eps < 1")
    if ell_true <= 0:
        raise ValueError("ell_true must be positive")

    k = exp_kernel(x_grid, x_grid, ell_true)
    k = symmetrize(k)
    k[np.diag_indices_from(k)] += jitter
    f = sample_gaussian(np.zeros(len(x_grid)), k, rng)
    max_abs = float(np.max(np.abs(f)))
    if max_abs <= 1e-14:
        return np.zeros_like(f)
    return ((1.0 - m_eps) / max_abs) * f



def sample_stationary_ou_values(
    points: np.ndarray,
    ell: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the stationary unit-variance OU GP at arbitrary 1D points.

    For the exponential kernel k(s,t)=exp(-|s-t|/ell), sorted finite-dimensional
    samples can be generated exactly by the Markov recursion

        f(z_i) = rho_i f(z_{i-1}) + sqrt(1-rho_i^2) eps_i,
        rho_i = exp(-(z_i-z_{i-1})/ell).

    The returned array is aligned with the input point order. Repeated points
    receive exactly the same sampled value because dx=0 gives rho=1.
    """
    if ell <= 0:
        raise ValueError("ell must be positive")
    points = np.asarray(points, dtype=float)
    if points.ndim != 1:
        raise ValueError("points must be one-dimensional")
    if points.size == 0:
        return np.empty(0, dtype=float)

    order = np.argsort(points, kind="mergesort")
    sorted_points = points[order]
    vals_sorted = np.empty(points.size, dtype=float)
    vals_sorted[0] = rng.standard_normal()
    for i in range(1, points.size):
        dx = max(float(sorted_points[i] - sorted_points[i - 1]), 0.0)
        rho = np.exp(-dx / ell)
        innov_sd = np.sqrt(max(1.0 - rho * rho, 0.0))
        vals_sorted[i] = rho * vals_sorted[i - 1] + innov_sd * rng.standard_normal()

    vals = np.empty_like(vals_sorted)
    vals[order] = vals_sorted
    return vals


def build_trials_from_clean_values(
    x_trains: list[np.ndarray],
    y_clean_trains: list[np.ndarray],
    m_eps: float,
    seed: int,
) -> list[dict]:
    """Add uniform observation noise to precomputed latent training values."""
    rng = np.random.default_rng(seed)
    trials: list[dict] = []
    for x_train, y_clean in zip(x_trains, y_clean_trains):
        noise = rng.uniform(-m_eps, m_eps, size=len(x_train))
        y_train = y_clean + noise
        trials.append({"x_train": x_train, "y_train": y_train, "y_clean": y_clean, "noise": noise})
    return trials


def sample_fixed_target_and_trials_joint_ou(
    x_target: np.ndarray,
    ell_true: float,
    m_eps: float,
    n_trials: int,
    n_train: int,
    seed: int,
    threshold: float,
    weights: np.ndarray,
    target_reject: bool = False,
    volume_min: float = 0.1,
    volume_max: float = 0.9,
    max_components: int | None = None,
    min_component_width: float = 0.0,
    min_mean_component_width: float = 0.0,
    max_attempts: int = 10000,
) -> tuple[np.ndarray, dict, list[dict]]:
    """Sample one fixed target and all repeated datasets from one OU path.

    Workflow:
      1. sample all training designs X_train^(j) once;
      2. sample the stationary OU path jointly on the target grid and all
         training points;
      3. compute acceptance diagnostics using only the target-grid path;
      4. if accepted, keep both the target-grid values and the corresponding
         exact latent training values, then add independent uniform noise.

    Thus y_clean is no longer obtained by interpolation from the grid; it is the
    exact finite-dimensional OU draw at the random training covariates, rescaled
    by the same factor as the target-grid path.
    """
    if not (0.0 <= m_eps < 1.0):
        raise ValueError("m_eps must satisfy 0 <= m_eps < 1")
    if ell_true <= 0:
        raise ValueError("ell_true must be positive")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    x_target = np.asarray(x_target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    domain_length = float(x_target[-1] - x_target[0])
    if volume_min < 0 or volume_max > domain_length or volume_min > volume_max:
        raise ValueError("invalid target rejection volume range")
    if max_components is not None and max_components < 0:
        raise ValueError("max_components must be non-negative or None")
    if min_component_width < 0:
        raise ValueError("min_component_width must be non-negative")
    if min_mean_component_width < 0:
        raise ValueError("min_mean_component_width must be non-negative")

    # Sample all random designs first, then hold them fixed across target
    # rejection attempts. This keeps the accepted target and all datasets tied
    # to a single fixed latent function while accepting/rejecting only on the
    # target-grid diagnostics.
    design_rng = np.random.default_rng(seed + 1)
    x_trains = [np.sort(design_rng.uniform(0.0, 1.0, size=n_train)) for _ in range(n_trials)]
    all_points = np.concatenate([x_target] + x_trains)
    n_target = len(x_target)

    path_rng = np.random.default_rng(seed)
    last_diag: dict | None = None
    accepted: tuple[np.ndarray, list[np.ndarray], dict] | None = None
    attempts_to_use = max_attempts if target_reject else 1

    for attempt in range(1, attempts_to_use + 1):
        f_all_raw = sample_stationary_ou_values(all_points, ell_true, path_rng)
        f_target_raw = f_all_raw[:n_target]
        max_abs = float(np.max(np.abs(f_target_raw)))
        scale = 0.0 if max_abs <= 1e-14 else (1.0 - m_eps) / max_abs
        f_target = scale * f_target_raw

        offset = n_target
        y_clean_trains: list[np.ndarray] = []
        for x_train in x_trains:
            m = len(x_train)
            y_clean_trains.append(scale * f_all_raw[offset:offset + m])
            offset += m

        diag = excursion_diagnostics(f_target, weights, threshold)
        last_diag = diag
        if target_reject:
            volume_ok = volume_min <= diag["volume"] <= volume_max
            comp_ok = max_components is None or diag["n_components"] <= max_components
            min_width_ok = diag["component_width_min"] >= min_component_width
            mean_width_ok = diag["component_width_mean"] >= min_mean_component_width
            accept = volume_ok and comp_ok and min_width_ok and mean_width_ok
        else:
            accept = True

        if accept:
            target_diag = {k: v for k, v in diag.items() if k != "indicator"}
            target_diag.update({
                "target_sampling_method": "joint_ou_target_grid_and_training_points",
                "target_training_values": "joint_ou_no_interpolation",
                "target_designs_sampled_before_rejection": True,
                "target_rejection_sampling": bool(target_reject),
                "target_rejection_attempts": int(attempt),
                "target_rejection_volume_min": float(volume_min) if target_reject else None,
                "target_rejection_volume_max": float(volume_max) if target_reject else None,
                "target_rejection_max_components": None if (not target_reject or max_components is None) else int(max_components),
                "target_rejection_min_component_width": float(min_component_width) if target_reject else None,
                "target_rejection_min_mean_component_width": float(min_mean_component_width) if target_reject else None,
                "target_rejection_max_attempts": int(max_attempts) if target_reject else None,
            })
            accepted = (f_target, y_clean_trains, target_diag)
            break

    if accepted is None:
        last = last_diag or {"volume": None, "n_components": None, "component_width_min": None, "component_width_mean": None}
        raise RuntimeError(
            f"Failed to sample an accepted target after {max_attempts} attempts. "
            f"Last diagnostics: volume={last['volume']}, "
            f"n_components={last['n_components']}, "
            f"min_width={last['component_width_min']}, "
            f"mean_width={last['component_width_mean']}."
        )

    f_target, y_clean_trains, target_diag = accepted
    trials = build_trials_from_clean_values(
        x_trains=x_trains,
        y_clean_trains=y_clean_trains,
        m_eps=m_eps,
        seed=seed + 2,
    )
    return f_target, target_diag, trials


# ----------------------------
# Posterior computation
# ----------------------------

def posterior_mean_from_kernels(
    k_xx: np.ndarray,
    k_tx: np.ndarray,
    y_train: np.ndarray,
    r: float,
    jitter: float = 1e-10,
) -> np.ndarray:
    """Fast posterior mean using precomputed kernels."""
    if r <= 0:
        raise ValueError("r must be positive")

    a = symmetrize(k_xx + (r ** 2 + jitter) * np.eye(k_xx.shape[0]))
    alpha = solve_spd(a, y_train)
    return k_tx @ alpha


def posterior_mean_and_cov_diag_from_kernels(
    k_xx: np.ndarray,
    k_tx: np.ndarray,
    y_train: np.ndarray,
    r: float,
    jitter: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast posterior mean and covariance diagonal core using precomputed kernels.

    This avoids recomputing exp kernels for every ridge value when several r's
    share the same ell_model. It also avoids forming the dense K_tt matrix.
    """
    if r <= 0:
        raise ValueError("r must be positive")

    a = symmetrize(k_xx + (r ** 2 + jitter) * np.eye(k_xx.shape[0]))
    alpha = solve_spd(a, y_train)
    mu = k_tx @ alpha

    solved = solve_spd(a, k_tx.T)
    # For the normalised exponential kernel, k(x,x)=1 on the diagonal.
    cov_diag = 1.0 - np.einsum("ij,ji->i", k_tx, solved, optimize=True)
    cov_diag = np.maximum(cov_diag, 0.0)
    return mu, cov_diag


def posterior_mean_and_cov_from_kernels(
    k_xx: np.ndarray,
    k_tx: np.ndarray,
    k_tt: np.ndarray,
    y_train: np.ndarray,
    r: float,
    jitter: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Posterior mean and full covariance core on the target grid.

    This is used only for post-hoc diagnostics of the actual one-path release
    F_D ~ GP(mu_D, sigma^2 k_D). The main grid search still uses only the
    covariance diagonal for speed.
    """
    if r <= 0:
        raise ValueError("r must be positive")

    a = symmetrize(k_xx + (r ** 2 + jitter) * np.eye(k_xx.shape[0]))
    alpha = solve_spd(a, y_train)
    mu = k_tx @ alpha

    solved = solve_spd(a, k_tx.T)
    cov = symmetrize(k_tt - k_tx @ solved)
    # Numerical roundoff can create tiny negative diagonal entries.
    diag = np.maximum(np.diag(cov), 0.0)
    cov[np.diag_indices_from(cov)] = diag
    return mu, cov


# ----------------------------
# Excursion-set / volume utilities
# ----------------------------

def trapezoid_weights(x_grid: np.ndarray) -> np.ndarray:
    """Weights w such that trapz(values, x_grid) == dot(w, values)."""
    x = np.asarray(x_grid, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("x_grid must be one-dimensional with at least two points")
    dx = np.diff(x)
    w = np.empty_like(x, dtype=float)
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    if len(x) > 2:
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def true_excursion_indicator(values: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(values) > threshold).astype(float)


def weighted_excursion_volume(values: np.ndarray, weights: np.ndarray, threshold: float) -> float:
    indicator = true_excursion_indicator(values, threshold)
    return float(np.dot(weights, indicator))


def count_excursion_components(indicator: np.ndarray) -> int:
    """Count connected positive components/runs in a 1D hard excursion indicator."""
    s = np.asarray(indicator, dtype=bool)
    if s.ndim != 1:
        raise ValueError("indicator must be one-dimensional")
    if s.size == 0:
        return 0
    starts = s & np.concatenate(([True], ~s[:-1]))
    return int(np.sum(starts))


def excursion_component_widths(indicator: np.ndarray, weights: np.ndarray) -> list[float]:
    """Approximate widths/measures of positive connected components on a 1D grid.

    The component widths are computed using the same quadrature weights as the
    excursion volume. Hence sum(widths) equals dot(weights, indicator).
    """
    s = np.asarray(indicator, dtype=bool)
    w = np.asarray(weights, dtype=float)
    if s.ndim != 1 or w.ndim != 1 or s.size != w.size:
        raise ValueError("indicator and weights must be one-dimensional arrays of the same length")
    widths: list[float] = []
    i = 0
    n = s.size
    while i < n:
        if not s[i]:
            i += 1
            continue
        j = i + 1
        while j < n and s[j]:
            j += 1
        widths.append(float(np.dot(w[i:j], s[i:j].astype(float))))
        i = j
    return widths


def excursion_diagnostics(values: np.ndarray, weights: np.ndarray, threshold: float) -> dict:
    """Grid-based diagnostics for the true excursion set {values > threshold}."""
    indicator = true_excursion_indicator(values, threshold)
    widths = excursion_component_widths(indicator, weights)
    volume = float(np.dot(weights, indicator))
    if widths:
        min_width = float(np.min(widths))
        mean_width = float(np.mean(widths))
        median_width = float(np.median(widths))
    else:
        min_width = 0.0
        mean_width = 0.0
        median_width = 0.0
    return {
        "indicator": indicator,
        "volume": volume,
        "n_components": int(len(widths)),
        "component_width_min": min_width,
        "component_width_mean": mean_width,
        "component_width_median": median_width,
    }


def sample_fixed_target_with_rejection_on_grid(
    x_grid: np.ndarray,
    ell_true: float,
    m_eps: float,
    rng: np.random.Generator,
    threshold: float,
    weights: np.ndarray,
    volume_min: float = 0.1,
    volume_max: float = 0.9,
    max_components: int | None = None,
    min_component_width: float = 0.0,
    min_mean_component_width: float = 0.0,
    max_attempts: int = 10000,
) -> tuple[np.ndarray, dict]:
    """Rejection-sample f_* from the same GP prior until the target is resolvable.

    The accepted target must satisfy
        volume_min <= T(f_*) <= volume_max,
        N_comp(f_*) <= max_components          if max_components is not None,
        min component width >= min_component_width,
        mean component width >= min_mean_component_width.

    Component widths and volume are measured on the target grid using the same
    quadrature weights as the utility calculations.
    """
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    domain_length = float(x_grid[-1] - x_grid[0])
    if volume_min < 0 or volume_max > domain_length or volume_min > volume_max:
        raise ValueError("invalid target rejection volume range")
    if max_components is not None and max_components < 0:
        raise ValueError("max_components must be non-negative or None")
    if min_component_width < 0:
        raise ValueError("min_component_width must be non-negative")
    if min_mean_component_width < 0:
        raise ValueError("min_mean_component_width must be non-negative")

    last_diag: dict | None = None
    for attempt in range(1, max_attempts + 1):
        f = sample_fixed_target_on_grid(x_grid, ell_true, m_eps, rng)
        diag = excursion_diagnostics(f, weights, threshold)
        last_diag = diag
        volume_ok = volume_min <= diag["volume"] <= volume_max
        comp_ok = max_components is None or diag["n_components"] <= max_components
        min_width_ok = diag["component_width_min"] >= min_component_width
        mean_width_ok = diag["component_width_mean"] >= min_mean_component_width
        if volume_ok and comp_ok and min_width_ok and mean_width_ok:
            diag = {k: v for k, v in diag.items() if k != "indicator"}
            diag.update({
                "target_rejection_sampling": True,
                "target_rejection_attempts": attempt,
                "target_rejection_volume_min": float(volume_min),
                "target_rejection_volume_max": float(volume_max),
                "target_rejection_max_components": None if max_components is None else int(max_components),
                "target_rejection_min_component_width": float(min_component_width),
                "target_rejection_min_mean_component_width": float(min_mean_component_width),
                "target_rejection_max_attempts": int(max_attempts),
            })
            return f, diag

    last = last_diag or {"volume": None, "n_components": None, "component_width_min": None, "component_width_mean": None}
    raise RuntimeError(
        f"Failed to sample an accepted target after {max_attempts} attempts. "
        f"Last diagnostics: volume={last['volume']}, "
        f"n_components={last['n_components']}, "
        f"min_width={last['component_width_min']}, "
        f"mean_width={last['component_width_mean']}."
    )


def prior_excursion_volume_variance_exp_kernel(
    ell: float,
    domain_length: float = 1.0,
    n_quad: int = 20000,
) -> float:
    """
    Var[int_0^L 1{f(x)>0} dx] for a centred GP with
    k(x,x') = exp(-|x-x'| / ell).
    """
    if ell <= 0:
        raise ValueError("ell must be positive")
    if domain_length <= 0:
        raise ValueError("domain_length must be positive")
    if n_quad < 2:
        raise ValueError("n_quad must be at least 2")

    h = np.linspace(0.0, domain_length, n_quad)
    rho = np.exp(-h / ell)
    integrand = (domain_length - h) * np.arcsin(np.clip(rho, 0.0, 1.0))
    return float(np.trapezoid(integrand, h) / np.pi)


def posterior_excursion_probability(
    mu: np.ndarray,
    cov_diag: np.ndarray,
    sigma: float,
    threshold: float,
) -> np.ndarray:
    """
    p_D(x) = P(F(x) > threshold | D).

    In the sigma=0 or zero-variance limit, this becomes the hard posterior-mean
    excursion indicator 1{mu(x)>threshold}.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    mu = np.asarray(mu, dtype=float)
    cov_diag = np.asarray(cov_diag, dtype=float)

    if sigma == 0.0:
        return (mu > threshold).astype(float)

    sd = sigma * np.sqrt(np.maximum(cov_diag, 0.0))
    probs = np.empty_like(mu, dtype=float)
    mask = sd > 1e-14
    probs[mask] = _normal_cdf((mu[mask] - threshold) / sd[mask])
    probs[~mask] = (mu[~mask] > threshold).astype(float)
    return probs


def integrated_bce_soft(
    p_excursion: np.ndarray,
    s_true: np.ndarray,
    weights: np.ndarray,
    clip: float = 1e-6,
) -> float:
    """Integrated binary cross-entropy for posterior excursion probabilities."""
    if not (0.0 < clip < 0.5):
        raise ValueError("clip must lie in (0, 0.5)")
    p = np.clip(np.asarray(p_excursion, dtype=float), clip, 1.0 - clip)
    s = np.asarray(s_true, dtype=float)
    loss = -(s * np.log(p) + (1.0 - s) * np.log(1.0 - p))
    return float(np.dot(weights, loss))



def hard_set_iou(
    s_pred: np.ndarray,
    s_true: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted IoU/Jaccard score for two hard excursion indicators."""
    pred = np.asarray(s_pred, dtype=bool)
    true = np.asarray(s_true, dtype=bool)
    inter = float(np.dot(weights, np.logical_and(pred, true).astype(float)))
    union = float(np.dot(weights, np.logical_or(pred, true).astype(float)))
    if union <= 1e-14:
        return 1.0
    return inter / union


def hard_set_error(
    s_pred: np.ndarray,
    s_true: np.ndarray,
    weights: np.ndarray,
) -> float:
    pred = np.asarray(s_pred, dtype=float)
    true = np.asarray(s_true, dtype=float)
    return float(np.dot(weights, np.abs(pred - true)))


def sample_exp_prior_target_and_train(
    x_target: np.ndarray,
    x_train: np.ndarray,
    ell: float,
    n_draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an exponential-kernel GP prior jointly at target and train points.

    The exponential kernel on a line is the stationary Ornstein--Uhlenbeck
    covariance, so sorted one-dimensional samples can be generated exactly by
    the Markov recursion

        f(x_i) = rho_i f(x_{i-1}) + sqrt(1-rho_i^2) z_i,
        rho_i = exp(-(x_i-x_{i-1})/ell).

    This avoids an O(n_target^3) Cholesky factorization when estimating the
    utility of actual posterior sample-path releases.
    """
    if ell <= 0:
        raise ValueError("ell must be positive")
    if n_draws <= 0:
        raise ValueError("n_draws must be positive")

    x_target = np.asarray(x_target, dtype=float)
    x_train = np.asarray(x_train, dtype=float)
    n_target = x_target.size
    points = np.concatenate([x_target, x_train])
    order = np.argsort(points, kind="mergesort")
    sorted_points = points[order]

    vals_sorted = np.empty((points.size, n_draws), dtype=float)
    vals_sorted[0, :] = rng.standard_normal(n_draws)
    for i in range(1, points.size):
        dx = max(float(sorted_points[i] - sorted_points[i - 1]), 0.0)
        rho = np.exp(-dx / ell)
        innov_sd = np.sqrt(max(1.0 - rho * rho, 0.0))
        vals_sorted[i, :] = rho * vals_sorted[i - 1, :] + innov_sd * rng.standard_normal(n_draws)

    vals = np.empty_like(vals_sorted)
    vals[order, :] = vals_sorted
    return vals[:n_target, :], vals[n_target:, :]


def solve_spd_many(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a X = B for SPD a and one or many right-hand sides."""
    if _cho_factor is not None and _cho_solve is not None:
        c_and_lower = _cho_factor(a, lower=True, check_finite=False, overwrite_a=False)
        return _cho_solve(c_and_lower, b, check_finite=False)
    L = np.linalg.cholesky(a)
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


def estimate_single_path_iou_for_distribution_rows(
    rows: pd.DataFrame,
    trials: list[dict],
    x_target: np.ndarray,
    f_target: np.ndarray,
    threshold: float,
    n_draws_per_trial: int,
    seed: int,
    jitter: float = 1e-10,
) -> pd.DataFrame:
    """Estimate IoU of the actual one posterior sample-path release.

    For each selected posterior-distribution row theta=(ell,r,sigma), this
    estimates

        E_{D,F_D}[ IoU({x: F_D(x)>u}, {x: f_*(x)>u}) ],

    and the randomized-release variation

        E_D[ Std_{F_D|D}( IoU({x: F_D(x)>u}, {x: f_*(x)>u}) ) ].

    The expectation is estimated over the same repeated noisy datasets D used
    in the grid search and over n_draws_per_trial independent posterior paths
    per dataset.

    Sampling uses Matheron's rule for the exponential-kernel GP.  This is exact
    for the target-grid posterior induced by the same kernel/ridge model, but it
    avoids factoring the dense n_target x n_target posterior covariance.
    """
    out = rows.copy()
    metric_cols = [
        "single_path_iou_mean",
        # By convention this is the randomized-release variation:
        # E_D[Std_{F_D|D}(IoU)], not the pooled Std_{D,F_D}(IoU).
        "single_path_iou_std",
        "single_path_iou_stderr",
        "single_path_iou_n",
        "single_path_iou_draws_per_dataset",
        "single_path_iou_total_std",
        "single_path_iou_total_stderr",
        "single_path_iou_conditional_std_mean",
        "single_path_iou_conditional_std_std",
        "single_path_iou_conditional_std_stderr",
        "single_path_iou_dataset_mean_std",
        "single_path_sign_error_mean",
        "single_path_volume_mean",
        "single_path_volume_abs_err_mean",
    ]
    if len(out) == 0:
        for col in metric_cols:
            out[col] = []
        return out
    if n_draws_per_trial <= 0:
        for col in metric_cols:
            out[col] = np.nan
        out["single_path_iou_n"] = 0
        out["single_path_iou_draws_per_dataset"] = 0
        return out

    weights = trapezoid_weights(x_target)
    s_true = true_excursion_indicator(f_target, threshold).astype(bool)
    true_volume = float(np.dot(weights, s_true.astype(float)))
    rng = np.random.default_rng(seed)

    # Deduplicate candidates across the unconstrained and epsilon-feasible top-3 lists.
    unique_rows = out.copy()
    for col in ["ell_model", "r", "sigma"]:
        unique_rows[f"_{col}_round"] = unique_rows[col].astype(float).round(12)
    unique_rows = unique_rows.drop_duplicates(subset=["_ell_model_round", "_r_round", "_sigma_round"])

    estimates: dict[tuple[float, float, float], dict] = {}
    iterator = unique_rows.itertuples(index=False)
    if tqdm is not None:
        iterator = tqdm(
            list(iterator),
            desc="single_path_iou_selected",
            unit="candidate",
            dynamic_ncols=True,
        )

    for row in iterator:
        ell = float(row.ell_model)
        r = float(row.r)
        sigma = float(row.sigma)
        key = (round(ell, 12), round(r, 12), round(sigma, 12))
        if key in estimates:
            continue

        iou_vals: list[float] = []
        sign_error_vals: list[float] = []
        volume_vals: list[float] = []
        volume_abs_err_vals: list[float] = []
        per_dataset_iou_means: list[float] = []
        per_dataset_iou_stds: list[float] = []

        for trial in trials:
            x_train = trial["x_train"]
            y_train = trial["y_train"]
            k_xx = exp_kernel(x_train, x_train, ell)
            k_tx = exp_kernel(x_target, x_train, ell)
            a = symmetrize(k_xx + (r ** 2 + jitter) * np.eye(k_xx.shape[0]))

            alpha = solve_spd_many(a, y_train)
            mu = k_tx @ alpha

            if sigma > 0.0:
                f0_target, f0_train = sample_exp_prior_target_and_train(
                    x_target=x_target,
                    x_train=x_train,
                    ell=ell,
                    n_draws=n_draws_per_trial,
                    rng=rng,
                )
                obs_noise0 = r * rng.standard_normal((len(x_train), n_draws_per_trial))
                residual = y_train.reshape(-1, 1) - (f0_train + obs_noise0)
                beta = solve_spd_many(a, residual)
                core_draws = f0_target + k_tx @ beta
                f_draws = mu.reshape(-1, 1) + sigma * (core_draws - mu.reshape(-1, 1))
            else:
                f_draws = np.repeat(mu.reshape(-1, 1), n_draws_per_trial, axis=1)

            s_draws = f_draws > threshold
            inter = weights @ np.logical_and(s_draws, s_true.reshape(-1, 1)).astype(float)
            union = weights @ np.logical_or(s_draws, s_true.reshape(-1, 1)).astype(float)
            ious = np.where(union > 1e-14, inter / union, 1.0)
            ious_arr = np.asarray(ious, dtype=float)
            per_dataset_iou_means.append(float(np.mean(ious_arr)))
            per_dataset_iou_stds.append(float(np.std(ious_arr)))
            sign_errors = weights @ np.logical_xor(s_draws, s_true.reshape(-1, 1)).astype(float)
            volumes = weights @ s_draws.astype(float)

            iou_vals.extend(ious_arr.tolist())
            sign_error_vals.extend(np.asarray(sign_errors, dtype=float).tolist())
            volume_vals.extend(np.asarray(volumes, dtype=float).tolist())
            volume_abs_err_vals.extend(np.abs(np.asarray(volumes, dtype=float) - true_volume).tolist())

        iou_arr = np.asarray(iou_vals, dtype=float)
        per_d_mean_arr = np.asarray(per_dataset_iou_means, dtype=float)
        per_d_std_arr = np.asarray(per_dataset_iou_stds, dtype=float)
        n = int(iou_arr.size)
        n_datasets = int(per_d_mean_arr.size)
        total_std = float(np.std(iou_arr)) if n else float("nan")
        cond_std_mean = float(np.mean(per_d_std_arr)) if n_datasets else float("nan")
        cond_std_std = float(np.std(per_d_std_arr)) if n_datasets else float("nan")
        dataset_mean_std = float(np.std(per_d_mean_arr)) if n_datasets else float("nan")
        estimates[key] = {
            "single_path_iou_mean": float(np.mean(iou_arr)) if n else float("nan"),
            # Main reported STD: randomized-release variation for fixed D,
            # averaged over datasets, E_D[Std_{F_D|D}(IoU)].
            "single_path_iou_std": cond_std_mean,
            "single_path_iou_stderr": float(np.std(per_d_mean_arr) / np.sqrt(n_datasets)) if n_datasets else float("nan"),
            "single_path_iou_n": n,
            "single_path_iou_draws_per_dataset": int(n_draws_per_trial),
            "single_path_iou_total_std": total_std,
            "single_path_iou_total_stderr": float(total_std / np.sqrt(n)) if n else float("nan"),
            "single_path_iou_conditional_std_mean": cond_std_mean,
            "single_path_iou_conditional_std_std": cond_std_std,
            "single_path_iou_conditional_std_stderr": float(cond_std_std / np.sqrt(n_datasets)) if n_datasets else float("nan"),
            "single_path_iou_dataset_mean_std": dataset_mean_std,
            "single_path_sign_error_mean": float(np.mean(sign_error_vals)) if n else float("nan"),
            "single_path_volume_mean": float(np.mean(volume_vals)) if n else float("nan"),
            "single_path_volume_abs_err_mean": float(np.mean(volume_abs_err_vals)) if n else float("nan"),
        }

    for col in metric_cols:
        out[col] = np.nan
    out["single_path_iou_n"] = 0
    out["single_path_iou_draws_per_dataset"] = 0
    for idx, row in out.iterrows():
        key = (round(float(row["ell_model"]), 12), round(float(row["r"]), 12), round(float(row["sigma"]), 12))
        est = estimates.get(key, {})
        for col, value in est.items():
            out.at[idx, col] = value
    return out

# ----------------------------
# Helpers
# ----------------------------

def parse_float_list(text_or_parts: str | list[str]) -> list[float]:
    """Parse comma-separated floats, accepting shell-split parts with spaces."""
    if isinstance(text_or_parts, list):
        text = ",".join(text_or_parts)
    else:
        text = text_or_parts
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty float list")
    return vals


def make_tagged_out_prefix(out_prefix: Path, seed: int, m_eps: float, create_dir: bool = True) -> Path:
    """Return output prefix in excursion_set_bce_iou_results with seed/M tag in the name."""
    outdir = Path("excursion_set_bce_iou_results")
    if create_dir:
        outdir.mkdir(parents=True, exist_ok=True)
    m_tag = f"{m_eps:g}"
    base_name = out_prefix.name
    tag = f"seed{seed}_M{m_tag}"
    if tag not in base_name:
        base_name = f"{base_name}_{tag}"
    return outdir / base_name


def unique_sorted(values: Iterable[float], decimals: int = 12) -> list[float]:
    return sorted({round(float(v), decimals) for v in values if float(v) >= 0})


def build_trials(
    n_trials: int,
    n_train: int,
    x_target: np.ndarray,
    f_target: np.ndarray,
    m_eps: float,
    seed: int,
) -> list[dict]:
    """Build repeated datasets from one fixed target f_*."""
    rng = np.random.default_rng(seed)
    trials: list[dict] = []
    for _ in range(n_trials):
        x_train = np.sort(rng.uniform(0.0, 1.0, size=n_train))
        y_clean = np.interp(x_train, x_target, f_target)
        noise = rng.uniform(-m_eps, m_eps, size=n_train)
        y_train = y_clean + noise
        trials.append({"x_train": x_train, "y_train": y_train, "y_clean": y_clean, "noise": noise})
    return trials


def make_distribution_candidates(
    ell_values: Iterable[float],
    r_values: Iterable[float],
    sigma_values: Iterable[float],
) -> list[tuple[float, float, float]]:
    return [(float(e), float(r), float(s)) for e in ell_values for r in r_values for s in sigma_values]


def make_mean_candidates(
    ell_values: Iterable[float],
    r_values: Iterable[float],
) -> list[tuple[float, float]]:
    return [(float(e), float(r)) for e in ell_values for r in r_values]


def local_refined_values(center: float, kind: str) -> list[float]:
    """Refined local grid around one coarse value."""
    if kind == "ell":
        factors = [0.7, 0.85, 1.0, 1.2, 1.5]
        return unique_sorted(center * f for f in factors if center * f > 0)
    if kind == "r":
        factors = [0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0]
        return unique_sorted(center * f for f in factors if center * f > 0)
    if kind == "sigma":
        if center == 0.0:
            return [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75]
        factors = [0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0]
        vals = [center * f for f in factors if center * f > 0]
        return unique_sorted(vals)
    raise ValueError(f"unknown kind: {kind}")


def make_refined_distribution_candidates(top_rows: pd.DataFrame) -> list[tuple[float, float, float]]:
    candidates: set[tuple[float, float, float]] = set()
    for _, row in top_rows.iterrows():
        ell0 = float(row["ell_model"])
        r0 = float(row["r"])
        sigma0 = float(row["sigma"])
        for ell in local_refined_values(ell0, "ell"):
            for r in local_refined_values(r0, "r"):
                for sigma in local_refined_values(sigma0, "sigma"):
                    candidates.add((round(ell, 12), round(r, 12), round(sigma, 12)))
    return sorted(candidates)


def make_refined_mean_candidates(top_rows: pd.DataFrame) -> list[tuple[float, float]]:
    candidates: set[tuple[float, float]] = set()
    for _, row in top_rows.iterrows():
        ell0 = float(row["ell_model"])
        r0 = float(row["r"])
        for ell in local_refined_values(ell0, "ell"):
            for r in local_refined_values(r0, "r"):
                candidates.add((round(ell, 12), round(r, 12)))
    return sorted(candidates)


def epsilon_for_distribution_candidate(
    ell: float,
    r: float,
    sigma: float,
    n: int,
    delta: float,
    L: int,
) -> float:
    """Compute the paper's 1D exponential-kernel DP bound for one candidate."""
    if sigma <= 0.0:
        return float("inf")
    if epsilon_for_delta_beta_exp_1d is None:
        raise ImportError(
            "Could not import epsilon_for_delta_beta_exp_1d from dp_utils. "
            "Run this script from an environment where dp_utils.py is importable."
        ) from _DP_UTILS_IMPORT_ERROR

    eps = epsilon_for_delta_beta_exp_1d(
        n=n,
        r=np.asarray([float(r)], dtype=float),
        kappa=np.exp(-1.0 / float(ell)),
        sigma=np.asarray([float(sigma)], dtype=float),
        eta=0.0,
        L=L,
        delta=delta,
    )["Epsilon"]
    arr = np.asarray(eps, dtype=float).reshape(-1)
    return float(arr[0]) if arr.size else float("nan")


def add_epsilon_column(
    df: pd.DataFrame,
    n: int,
    delta: float,
    L: int,
) -> pd.DataFrame:
    """Return a copy with an epsilon column for posterior-distribution rows."""
    out = df.copy()
    if len(out) == 0:
        out["epsilon"] = []
        return out
    out["epsilon"] = [
        epsilon_for_distribution_candidate(
            ell=float(row.ell_model),
            r=float(row.r),
            sigma=float(row.sigma),
            n=n,
            delta=delta,
            L=L,
        )
        for row in out.itertuples(index=False)
    ]
    return out


def top_bce_rows(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Top-k rows by posterior-distribution BCE."""
    if len(df) == 0:
        return df.copy()
    return df.sort_values(
        ["bce_post_mean", "volume_abs_err_mean", "ell_model", "r", "sigma"]
    ).head(top_k).copy()


def top_private_bce_rows(
    df: pd.DataFrame,
    top_k: int,
    epsilon_threshold: float,
) -> pd.DataFrame:
    """Top-k BCE rows among candidates satisfying epsilon < epsilon_threshold."""
    if "epsilon" not in df.columns:
        raise ValueError("top_private_bce_rows requires an epsilon column")
    feasible = df[np.isfinite(df["epsilon"]) & (df["epsilon"] < epsilon_threshold)].copy()
    return top_bce_rows(feasible, top_k)


def unique_distribution_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate rows by (ell_model, r, sigma), preserving first occurrence."""
    if len(rows) == 0:
        return rows.copy()
    out = rows.copy()
    for col in ["ell_model", "r", "sigma"]:
        out[f"_{col}_round"] = out[col].astype(float).round(12)
    out = out.drop_duplicates(subset=["_ell_model_round", "_r_round", "_sigma_round"])
    return out.drop(columns=["_ell_model_round", "_r_round", "_sigma_round"])


def distribution_frontier_summary(
    df: pd.DataFrame,
    epsilon_threshold: float,
) -> dict:
    """
    Summarise unconstrained versus epsilon-constrained posterior-distribution BCE.

    The posterior-distribution frontier is selected by BCE only. IoU, volume MAE,
    and uncertainty diagnostics are reported alongside for the BCE-selected rows.
    We intentionally do not optimise IoU inside this summary, because hard IoU is
    mostly a diagnostic for the BCE-selected posterior distribution rather than a
    separate posterior-distribution selection rule.
    """
    if len(df) == 0:
        return {}
    if "epsilon" not in df.columns:
        raise ValueError("distribution_frontier_summary requires an epsilon column")

    best_bce = df.sort_values(["bce_post_mean", "ell_model", "r", "sigma"]).iloc[0]
    feasible = df[np.isfinite(df["epsilon"]) & (df["epsilon"] < epsilon_threshold)]

    out = {
        "epsilon_threshold": float(epsilon_threshold),
        "n_candidates": int(len(df)),
        "n_epsilon_feasible_candidates": int(len(feasible)),
        "best_unconstrained_bce": float(best_bce["bce_post_mean"]),
        "best_unconstrained_bce_iou": float(best_bce["posterior_hard_iou_mean"]),
        "best_unconstrained_bce_volume_mae": float(best_bce["volume_abs_err_mean"]),
        "best_unconstrained_bce_epsilon": float(best_bce["epsilon"]),
        "best_unconstrained_bce_ell_model": float(best_bce["ell_model"]),
        "best_unconstrained_bce_r": float(best_bce["r"]),
        "best_unconstrained_bce_sigma": float(best_bce["sigma"]),
        "best_unconstrained_bce_effective_dim_mean": float(best_bce["effective_dim_mean"]),
        "best_unconstrained_bce_effective_dim_std": float(best_bce["effective_dim_std"]),
    }
    if len(feasible) == 0:
        out.update({
            "has_epsilon_feasible_candidate": False,
            "best_epsilon_feasible_bce": float("nan"),
            "best_epsilon_feasible_bce_iou": float("nan"),
            "best_epsilon_feasible_bce_volume_mae": float("nan"),
            "best_epsilon_feasible_bce_epsilon": float("nan"),
            "bce_gap_epsilon_feasible": float("nan"),
            "relative_bce_increase_epsilon_feasible": float("nan"),
            "absolute_iou_loss_for_bce_selected_epsilon_feasible": float("nan"),
            "relative_iou_loss_for_bce_selected_epsilon_feasible": float("nan"),
            "volume_mae_gap_for_bce_selected_epsilon_feasible": float("nan"),
        })
        return out

    best_priv_bce = feasible.sort_values(["bce_post_mean", "ell_model", "r", "sigma"]).iloc[0]
    iou_unconstrained = float(best_bce["posterior_hard_iou_mean"])
    iou_priv = float(best_priv_bce["posterior_hard_iou_mean"])
    out.update({
        "has_epsilon_feasible_candidate": True,
        "best_epsilon_feasible_bce": float(best_priv_bce["bce_post_mean"]),
        "best_epsilon_feasible_bce_iou": iou_priv,
        "best_epsilon_feasible_bce_volume_mae": float(best_priv_bce["volume_abs_err_mean"]),
        "best_epsilon_feasible_bce_epsilon": float(best_priv_bce["epsilon"]),
        "best_epsilon_feasible_bce_ell_model": float(best_priv_bce["ell_model"]),
        "best_epsilon_feasible_bce_r": float(best_priv_bce["r"]),
        "best_epsilon_feasible_bce_sigma": float(best_priv_bce["sigma"]),
        "best_epsilon_feasible_bce_effective_dim_mean": float(best_priv_bce["effective_dim_mean"]),
        "best_epsilon_feasible_bce_effective_dim_std": float(best_priv_bce["effective_dim_std"]),
        "bce_gap_epsilon_feasible": float(best_priv_bce["bce_post_mean"] - best_bce["bce_post_mean"]),
        "relative_bce_increase_epsilon_feasible": float(best_priv_bce["bce_post_mean"] / best_bce["bce_post_mean"] - 1.0) if float(best_bce["bce_post_mean"]) > 0 else float("nan"),
        "absolute_iou_loss_for_bce_selected_epsilon_feasible": float(iou_unconstrained - iou_priv),
        "relative_iou_loss_for_bce_selected_epsilon_feasible": float(1.0 - iou_priv / iou_unconstrained) if iou_unconstrained > 0 else float("nan"),
        "volume_mae_gap_for_bce_selected_epsilon_feasible": float(best_priv_bce["volume_abs_err_mean"] - best_bce["volume_abs_err_mean"]),
    })
    return out


# ----------------------------
# Candidate evaluation: posterior distribution BCE
# ----------------------------

def evaluate_distribution_candidates(
    candidates: list[tuple[float, float, float]],
    trials: list[dict],
    x_target: np.ndarray,
    f_target: np.ndarray,
    n_train: int,
    n_target_grid: int,
    ell_true: float,
    m_eps: float,
    threshold: float,
    stage: str,
    bce_clip: float = 1e-6,
    hard_set_cutoff: float = 0.5,
) -> pd.DataFrame:
    """Evaluate posterior-distribution candidates using integrated BCE."""
    if not (0.0 <= hard_set_cutoff <= 1.0):
        raise ValueError("hard_set_cutoff must lie in [0, 1]")

    sigmas_by_ell_r: dict[float, dict[float, list[float]]] = {}
    for ell_model, r, sigma in candidates:
        ell_key = float(ell_model)
        r_key = float(r)
        sigmas_by_ell_r.setdefault(ell_key, {}).setdefault(r_key, []).append(float(sigma))
    sigmas_by_ell_r = {
        ell: {r: unique_sorted(sigmas) for r, sigmas in r_map.items()}
        for ell, r_map in sigmas_by_ell_r.items()
    }

    weights = trapezoid_weights(x_target)
    s_true = true_excursion_indicator(f_target, threshold)
    true_volume = float(np.dot(weights, s_true))
    domain_length = float(x_target[-1] - x_target[0])
    prior_bce = float(domain_length * np.log(2.0))
    oracle_const_p = float(np.clip(true_volume / domain_length, bce_clip, 1.0 - bce_clip))
    oracle_const_bce = float(-domain_length * (true_volume / domain_length * np.log(oracle_const_p)
                                               + (1.0 - true_volume / domain_length) * np.log(1.0 - oracle_const_p)))

    rows: list[dict] = []
    total_groups = sum(len(r_map) for r_map in sigmas_by_ell_r.values())
    group_idx = 0
    progress = tqdm(total=total_groups, desc=stage, unit="(ell,r)", dynamic_ncols=True) if tqdm is not None else None

    for ell_model, r_to_sigmas in sorted(sigmas_by_ell_r.items()):
        kernel_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for trial in trials:
            k_xx = exp_kernel(trial["x_train"], trial["x_train"], ell_model)
            k_tx = exp_kernel(x_target, trial["x_train"], ell_model)
            kernel_cache.append((k_xx, k_tx, trial["y_train"]))

        for r, sigma_values in sorted(r_to_sigmas.items()):
            group_idx += 1
            bce_vals: dict[float, list[float]] = {s: [] for s in sigma_values}
            hard_iou_vals: dict[float, list[float]] = {s: [] for s in sigma_values}
            hard_error_vals: dict[float, list[float]] = {s: [] for s in sigma_values}
            hard_volume_vals: dict[float, list[float]] = {s: [] for s in sigma_values}
            est_volume_vals: dict[float, list[float]] = {s: [] for s in sigma_values}
            volume_abs_errs: dict[float, list[float]] = {s: [] for s in sigma_values}
            volume_sq_errs: dict[float, list[float]] = {s: [] for s in sigma_values}
            uncertainty_integral_vals: dict[float, list[float]] = {s: [] for s in sigma_values}
            effective_dim_vals: list[float] = []

            for k_xx, k_tx, y_train in kernel_cache:
                effective_dim_vals.append(effective_dimension_from_kernel(k_xx, r))
                mu, cov_diag = posterior_mean_and_cov_diag_from_kernels(
                    k_xx=k_xx,
                    k_tx=k_tx,
                    y_train=y_train,
                    r=r,
                )

                for sigma in sigma_values:
                    p_exc = posterior_excursion_probability(
                        mu=mu,
                        cov_diag=cov_diag,
                        sigma=sigma,
                        threshold=threshold,
                    )
                    bce = integrated_bce_soft(p_exc, s_true, weights, clip=bce_clip)
                    s_hard = (p_exc >= hard_set_cutoff).astype(float)
                    hard_iou = hard_set_iou(s_hard, s_true, weights)
                    hard_error = hard_set_error(s_hard, s_true, weights)
                    hard_volume = float(np.dot(weights, s_hard))
                    est_volume = float(np.dot(weights, p_exc))
                    uncertainty_integral = float(np.dot(weights, p_exc * (1.0 - p_exc)))

                    bce_vals[sigma].append(bce)
                    hard_iou_vals[sigma].append(hard_iou)
                    hard_error_vals[sigma].append(hard_error)
                    hard_volume_vals[sigma].append(hard_volume)
                    est_volume_vals[sigma].append(est_volume)
                    volume_abs_errs[sigma].append(abs(est_volume - true_volume))
                    volume_sq_errs[sigma].append((est_volume - true_volume) ** 2)
                    uncertainty_integral_vals[sigma].append(uncertainty_integral)

                del mu, cov_diag

            group_rows = []
            for sigma in sigma_values:
                row = {
                    "stage": stage,
                    "search_type": "posterior_distribution_bce",
                    "estimator": "analytic_excursion_probability_bce_fast",
                    "n_train": n_train,
                    "n_target_grid": n_target_grid,
                    "ell_true": ell_true,
                    "ell_model": ell_model,
                    "m_eps": m_eps,
                    "threshold": threshold,
                    "r": r,
                    "sigma": sigma,
                    "n_trials": len(trials),
                    "hard_set_cutoff": hard_set_cutoff,
                    "bce_clip": bce_clip,
                    "true_excursion_volume": true_volume,
                    "prior_probability_bce": prior_bce,
                    "oracle_constant_bce": oracle_const_bce,
                    "bce_post_mean": float(np.mean(bce_vals[sigma])),
                    "bce_post_std": float(np.std(bce_vals[sigma])),
                    "posterior_hard_iou_mean": float(np.mean(hard_iou_vals[sigma])),
                    "posterior_hard_iou_std": float(np.std(hard_iou_vals[sigma])),
                    "posterior_hard_error_mean": float(np.mean(hard_error_vals[sigma])),
                    "posterior_hard_error_std": float(np.std(hard_error_vals[sigma])),
                    "posterior_hard_volume_mean": float(np.mean(hard_volume_vals[sigma])),
                    "est_ET_mean": float(np.mean(est_volume_vals[sigma])),
                    "est_ET_std_across_trials": float(np.std(est_volume_vals[sigma])),
                    "volume_abs_err_mean": float(np.mean(volume_abs_errs[sigma])),
                    "volume_abs_err_std": float(np.std(volume_abs_errs[sigma])),
                    "volume_rmse": float(np.sqrt(np.mean(volume_sq_errs[sigma]))),
                    "uncertainty_integral_mean": float(np.mean(uncertainty_integral_vals[sigma])),
                    "effective_dim_mean": float(np.mean(effective_dim_vals)),
                    "effective_dim_std": float(np.std(effective_dim_vals)),
                    "effective_dim_min": float(np.min(effective_dim_vals)),
                    "effective_dim_max": float(np.max(effective_dim_vals)),
                }
                rows.append(row)
                group_rows.append(row)

            best_group = min(group_rows, key=lambda row: row["bce_post_mean"])
            if progress is not None:
                progress.set_postfix(
                    ell=f"{ell_model:g}",
                    r=f"{r:g}",
                    best_bce=f"{best_group['bce_post_mean']:.4f}",
                    refresh=False,
                )
                progress.update(1)
            elif group_idx == 1 or group_idx == total_groups or group_idx % max(1, total_groups // 20) == 0:
                print(
                    f"[{stage} {group_idx:>3}/{total_groups}] "
                    f"ell={ell_model:g}, r={r:g}, best_group_bce={best_group['bce_post_mean']:.4f}"
                )

        del kernel_cache

    if progress is not None:
        progress.close()

    return pd.DataFrame(rows)


# ----------------------------
# Candidate evaluation: posterior mean IoU
# ----------------------------

def evaluate_mean_candidates(
    candidates: list[tuple[float, float]],
    trials: list[dict],
    x_target: np.ndarray,
    f_target: np.ndarray,
    n_train: int,
    n_target_grid: int,
    ell_true: float,
    m_eps: float,
    threshold: float,
    stage: str,
) -> pd.DataFrame:
    """Evaluate posterior-mean plug-in candidates using IoU. No sigma grid."""
    r_by_ell: dict[float, list[float]] = {}
    for ell_model, r in candidates:
        r_by_ell.setdefault(float(ell_model), []).append(float(r))
    r_by_ell = {ell: unique_sorted(rs) for ell, rs in r_by_ell.items()}

    weights = trapezoid_weights(x_target)
    s_true = true_excursion_indicator(f_target, threshold)
    true_volume = float(np.dot(weights, s_true))
    rows: list[dict] = []
    total_groups = sum(len(rs) for rs in r_by_ell.values())
    group_idx = 0
    progress = tqdm(total=total_groups, desc=stage, unit="(ell,r)", dynamic_ncols=True) if tqdm is not None else None

    for ell_model, r_values in sorted(r_by_ell.items()):
        kernel_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for trial in trials:
            k_xx = exp_kernel(trial["x_train"], trial["x_train"], ell_model)
            k_tx = exp_kernel(x_target, trial["x_train"], ell_model)
            kernel_cache.append((k_xx, k_tx, trial["y_train"]))

        for r in sorted(r_values):
            group_idx += 1
            iou_vals: list[float] = []
            error_vals: list[float] = []
            volume_vals: list[float] = []
            volume_abs_errs: list[float] = []
            effective_dim_vals: list[float] = []

            for k_xx, k_tx, y_train in kernel_cache:
                effective_dim_vals.append(effective_dimension_from_kernel(k_xx, r))
                mu = posterior_mean_from_kernels(
                    k_xx=k_xx,
                    k_tx=k_tx,
                    y_train=y_train,
                    r=r,
                )
                s_mu = (mu > threshold).astype(float)
                iou_vals.append(hard_set_iou(s_mu, s_true, weights))
                error_vals.append(hard_set_error(s_mu, s_true, weights))
                volume = float(np.dot(weights, s_mu))
                volume_vals.append(volume)
                volume_abs_errs.append(abs(volume - true_volume))
                del mu

            row = {
                "stage": stage,
                "search_type": "posterior_mean_iou",
                "estimator": "posterior_mean_plugin_hard_set",
                "n_train": n_train,
                "n_target_grid": n_target_grid,
                "ell_true": ell_true,
                "ell_model": ell_model,
                "m_eps": m_eps,
                "threshold": threshold,
                "r": r,
                "n_trials": len(trials),
                "true_excursion_volume": true_volume,
                "meanfun_iou_mean": float(np.mean(iou_vals)),
                "meanfun_iou_std": float(np.std(iou_vals)),
                "meanfun_sign_error_mean": float(np.mean(error_vals)),
                "meanfun_sign_error_std": float(np.std(error_vals)),
                "meanfun_volume_mean": float(np.mean(volume_vals)),
                "meanfun_volume_abs_err_mean": float(np.mean(volume_abs_errs)),
                "effective_dim_mean": float(np.mean(effective_dim_vals)),
                "effective_dim_std": float(np.std(effective_dim_vals)),
                "effective_dim_min": float(np.min(effective_dim_vals)),
                "effective_dim_max": float(np.max(effective_dim_vals)),
            }
            rows.append(row)

            if progress is not None:
                progress.set_postfix(
                    ell=f"{ell_model:g}",
                    r=f"{r:g}",
                    iou=f"{row['meanfun_iou_mean']:.4f}",
                    refresh=False,
                )
                progress.update(1)
            elif group_idx == 1 or group_idx == total_groups or group_idx % max(1, total_groups // 20) == 0:
                print(
                    f"[{stage} {group_idx:>3}/{total_groups}] "
                    f"ell={ell_model:g}, r={r:g}, mean_iou={row['meanfun_iou_mean']:.4f}"
                )

        del kernel_cache

    if progress is not None:
        progress.close()

    return pd.DataFrame(rows)


# ----------------------------
# Main experiment
# ----------------------------

def run_coarse_refine_search(
    n_train: int,
    n_target_grid: int,
    ell_true: float,
    ell_values: list[float],
    r_values: list[float],
    sigma_values: list[float],
    n_trials: int,
    m_eps: float,
    threshold: float,
    seed: int,
    out_prefix: Path,
    top_k: int,
    bce_clip: float = 1e-6,
    hard_set_cutoff: float = 0.5,
    epsilon_threshold: float = 10.0,
    epsilon_delta: float = 0.005,
    epsilon_L: int = 1,
    private_refine_rounds: int = 3,
    target_sieve: bool = True,
    target_sieve_ell: float | None = None,
    target_sieve_sd_factor: float = 0.0,
    target_sieve_extreme_tol: float = 1e-12,
    target_reject: bool = False,
    target_reject_volume_min: float = 0.1,
    target_reject_volume_max: float = 0.9,
    target_reject_max_components: int | None = None,
    target_reject_min_component_width: float = 0.0,
    target_reject_min_mean_component_width: float = 0.0,
    target_reject_max_attempts: int = 10000,
    n_single_path_iou_draws: int = 1,
    single_path_iou_seed_offset: int = 100000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[dict]] | None:
    x_target = np.linspace(0.0, 1.0, n_target_grid)
    weights = trapezoid_weights(x_target)
    f_target, target_diag, trials = sample_fixed_target_and_trials_joint_ou(
        x_target=x_target,
        ell_true=ell_true,
        m_eps=m_eps,
        n_trials=n_trials,
        n_train=n_train,
        seed=seed,
        threshold=threshold,
        weights=weights,
        target_reject=target_reject,
        volume_min=target_reject_volume_min,
        volume_max=target_reject_volume_max,
        max_components=target_reject_max_components,
        min_component_width=target_reject_min_component_width,
        min_mean_component_width=target_reject_min_mean_component_width,
        max_attempts=target_reject_max_attempts,
    )
    s_true = true_excursion_indicator(f_target, threshold)
    t_true = float(target_diag["volume"])
    n_excursion_components = int(target_diag["n_components"])
    domain_length = float(x_target[-1] - x_target[0])
    prior_bce = float(domain_length * np.log(2.0))
    oracle_p = float(np.clip(t_true / domain_length, bce_clip, 1.0 - bce_clip))
    oracle_const_bce = float(-domain_length * ((t_true / domain_length) * np.log(oracle_p)
                                               + (1.0 - t_true / domain_length) * np.log(1.0 - oracle_p)))

    print(f"True excursion volume T_u(f_*): {t_true:.6f}")
    print(f"True excursion components N_comp(f_*): {n_excursion_components}")
    print(
        "True excursion component widths: "
        f"min={target_diag['component_width_min']:.6f}, "
        f"median={target_diag['component_width_median']:.6f}, "
        f"mean={target_diag['component_width_mean']:.6f}"
    )
    if target_diag["target_rejection_sampling"]:
        print(
            "Target rejection sampling accepted after "
            f"{target_diag['target_rejection_attempts']} attempt(s)."
        )
    print("Target/training latent values sampled jointly by OU recursion; no interpolation for y_clean.")
    print(f"Prior probability baseline BCE loss: {prior_bce:.6f}")
    print(f"Oracle constant-probability BCE loss: {oracle_const_bce:.6f}")

    if target_sieve:
        if threshold != 0.0:
            print(
                "WARNING: target sieve uses the prior-centred benchmark L/2, "
                "which is theoretically calibrated for threshold=0."
            )
        sieve_ell = ell_true if target_sieve_ell is None else float(target_sieve_ell)
        prior_var = prior_excursion_volume_variance_exp_kernel(
            ell=sieve_ell,
            domain_length=domain_length,
        )
        prior_sd = float(np.sqrt(prior_var))
        prior_center = 0.5 * domain_length
        gap_from_prior_center = abs(t_true - prior_center)
        near_prior_center = gap_from_prior_center < target_sieve_sd_factor * prior_sd
        extreme_volume = (
            t_true <= target_sieve_extreme_tol
            or t_true >= domain_length - target_sieve_extreme_tol
        )

        print(
            "Target sieve: "
            f"ell={sieve_ell:g}, SD(T)={prior_sd:.6f}, "
            f"|T-L/2|={gap_from_prior_center:.6f}, "
            f"threshold={target_sieve_sd_factor:.3g}*SD={target_sieve_sd_factor * prior_sd:.6f}, "
            f"extreme={extreme_volume}"
        )

        if near_prior_center or extreme_volume:
            reasons = []
            if near_prior_center:
                reasons.append("too close to the prior-volume baseline")
            if extreme_volume:
                reasons.append("excursion volume is numerically 0 or 1")
            print(
                "Skipping grid search and writing no output files because the target is "
                + " and ".join(reasons)
                + "."
            )
            return None

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Trials were already built jointly with the accepted target path above,
    # so y_clean=f_*(X_train) is exact under the OU finite-dimensional draw
    # rather than a linear interpolation from the target grid.

    # Posterior distribution search: ell, r, sigma; objective = BCE only.
    # The refinement has two branches:
    #   (i)  one unconstrained refinement around top-k coarse BCE candidates;
    #   (ii) iterative privacy-feasible refinement around top-k BCE candidates satisfying epsilon < threshold.
    # IoU is only reported alongside the BCE-selected posterior-distribution candidates.
    coarse_distribution_candidates = make_distribution_candidates(ell_values, r_values, sigma_values)
    print(f"\n=== Coarse posterior-distribution BCE search: {len(coarse_distribution_candidates)} candidates ===")
    coarse_distribution_df = evaluate_distribution_candidates(
        candidates=coarse_distribution_candidates,
        trials=trials,
        x_target=x_target,
        f_target=f_target,
        n_train=n_train,
        n_target_grid=n_target_grid,
        ell_true=ell_true,
        m_eps=m_eps,
        threshold=threshold,
        stage="coarse_distribution_bce",
        bce_clip=bce_clip,
        hard_set_cutoff=hard_set_cutoff,
    )
    coarse_distribution_df = add_epsilon_column(
        coarse_distribution_df,
        n=n_train,
        delta=epsilon_delta,
        L=epsilon_L,
    )

    top_coarse_distribution = top_bce_rows(coarse_distribution_df, top_k)
    top_coarse_distribution_private = top_private_bce_rows(
        coarse_distribution_df,
        top_k=top_k,
        epsilon_threshold=epsilon_threshold,
    )

    # Round 1: refine around both unconstrained and privacy-feasible coarse anchors.
    round1_anchors = unique_distribution_rows(
        pd.concat([top_coarse_distribution, top_coarse_distribution_private], ignore_index=True)
    )
    refined_distribution_dfs: list[pd.DataFrame] = []
    refined_distribution_df = pd.DataFrame()

    if len(round1_anchors) > 0:
        refined_distribution_candidates = make_refined_distribution_candidates(round1_anchors)
        print(
            f"\n=== Refined posterior-distribution BCE search round 1 "
            f"around {len(round1_anchors)} anchors: {len(refined_distribution_candidates)} candidates ==="
        )
        refined_round_df = evaluate_distribution_candidates(
            candidates=refined_distribution_candidates,
            trials=trials,
            x_target=x_target,
            f_target=f_target,
            n_train=n_train,
            n_target_grid=n_target_grid,
            ell_true=ell_true,
            m_eps=m_eps,
            threshold=threshold,
            stage="refined_distribution_bce_round1",
            bce_clip=bce_clip,
            hard_set_cutoff=hard_set_cutoff,
        )
        refined_round_df = add_epsilon_column(refined_round_df, n=n_train, delta=epsilon_delta, L=epsilon_L)
        refined_distribution_dfs.append(refined_round_df)
        refined_distribution_df = pd.concat(refined_distribution_dfs, ignore_index=True)

    # Rounds 2..private_refine_rounds: refine only around the best privacy-feasible BCE anchors.
    for refine_round in range(2, private_refine_rounds + 1):
        candidate_pool = refined_distribution_df if len(refined_distribution_df) else coarse_distribution_df
        private_anchors = top_private_bce_rows(
            candidate_pool,
            top_k=top_k,
            epsilon_threshold=epsilon_threshold,
        )
        if len(private_anchors) == 0:
            print(
                f"No epsilon<{epsilon_threshold:g} posterior-distribution candidates found "
                f"for private refinement round {refine_round}; stopping private refinement."
            )
            break
        private_refined_candidates = make_refined_distribution_candidates(private_anchors)
        print(
            f"\n=== Private refined posterior-distribution BCE search round {refine_round} "
            f"around top {len(private_anchors)} epsilon<{epsilon_threshold:g} anchors: "
            f"{len(private_refined_candidates)} candidates ==="
        )
        private_round_df = evaluate_distribution_candidates(
            candidates=private_refined_candidates,
            trials=trials,
            x_target=x_target,
            f_target=f_target,
            n_train=n_train,
            n_target_grid=n_target_grid,
            ell_true=ell_true,
            m_eps=m_eps,
            threshold=threshold,
            stage=f"private_refined_distribution_bce_round{refine_round}",
            bce_clip=bce_clip,
            hard_set_cutoff=hard_set_cutoff,
        )
        private_round_df = add_epsilon_column(private_round_df, n=n_train, delta=epsilon_delta, L=epsilon_L)
        refined_distribution_dfs.append(private_round_df)
        refined_distribution_df = pd.concat(refined_distribution_dfs, ignore_index=True)
        refined_distribution_df = unique_distribution_rows(refined_distribution_df)

    if len(refined_distribution_df) == 0:
        refined_distribution_df = coarse_distribution_df.copy()

    # Posterior mean search: ell, r only; objective = IoU.
    coarse_mean_candidates = make_mean_candidates(ell_values, r_values)
    print(f"\n=== Coarse posterior-mean IoU search: {len(coarse_mean_candidates)} candidates ===")
    coarse_mean_df = evaluate_mean_candidates(
        candidates=coarse_mean_candidates,
        trials=trials,
        x_target=x_target,
        f_target=f_target,
        n_train=n_train,
        n_target_grid=n_target_grid,
        ell_true=ell_true,
        m_eps=m_eps,
        threshold=threshold,
        stage="coarse_mean_iou",
    )
    top_coarse_mean = coarse_mean_df.sort_values(
        ["meanfun_iou_mean", "meanfun_sign_error_mean", "meanfun_volume_abs_err_mean", "ell_model", "r"],
        ascending=[False, True, True, True, True],
    ).head(top_k).copy()
    refined_mean_candidates = make_refined_mean_candidates(top_coarse_mean)
    print(
        f"\n=== Refined posterior-mean IoU search around top {top_k}: "
        f"{len(refined_mean_candidates)} candidates ==="
    )
    refined_mean_df = evaluate_mean_candidates(
        candidates=refined_mean_candidates,
        trials=trials,
        x_target=x_target,
        f_target=f_target,
        n_train=n_train,
        n_target_grid=n_target_grid,
        ell_true=ell_true,
        m_eps=m_eps,
        threshold=threshold,
        stage="refined_mean_iou",
    )

    top3_distribution_df = top_bce_rows(refined_distribution_df, 3)
    top3_distribution_private_df = top_private_bce_rows(
        refined_distribution_df,
        top_k=3,
        epsilon_threshold=epsilon_threshold,
    )
    frontier_summary = distribution_frontier_summary(refined_distribution_df, epsilon_threshold)
    top3_mean_df = refined_mean_df.sort_values(
        ["meanfun_iou_mean", "meanfun_sign_error_mean", "meanfun_volume_abs_err_mean", "ell_model", "r"],
        ascending=[False, True, True, True, True],
    ).head(3).copy()

    # Post-hoc operational utility for the actual one-path release.
    # This is intentionally evaluated only for the selected BCE rows to keep the
    # main grid search fast. The single released object is F_D, and the decision
    # is its thresholded excursion set {x: F_D(x)>u}.
    selected_for_single_path = unique_distribution_rows(
        pd.concat([top3_distribution_df, top3_distribution_private_df], ignore_index=True)
    )
    selected_for_single_path = estimate_single_path_iou_for_distribution_rows(
        rows=selected_for_single_path,
        trials=trials,
        x_target=x_target,
        f_target=f_target,
        threshold=threshold,
        n_draws_per_trial=n_single_path_iou_draws,
        seed=seed + single_path_iou_seed_offset,
    )
    top3_distribution_df = estimate_single_path_iou_for_distribution_rows(
        rows=top3_distribution_df,
        trials=trials,
        x_target=x_target,
        f_target=f_target,
        threshold=threshold,
        n_draws_per_trial=0,
        seed=seed + single_path_iou_seed_offset,
    )
    top3_distribution_private_df = estimate_single_path_iou_for_distribution_rows(
        rows=top3_distribution_private_df,
        trials=trials,
        x_target=x_target,
        f_target=f_target,
        threshold=threshold,
        n_draws_per_trial=0,
        seed=seed + single_path_iou_seed_offset,
    )
    for target_df_name in ["top3_distribution_df", "top3_distribution_private_df"]:
        target_df = locals()[target_df_name]
        for idx, row in target_df.iterrows():
            mask = (
                np.isclose(selected_for_single_path["ell_model"].astype(float), float(row["ell_model"]))
                & np.isclose(selected_for_single_path["r"].astype(float), float(row["r"]))
                & np.isclose(selected_for_single_path["sigma"].astype(float), float(row["sigma"]))
            )
            if not np.any(mask):
                continue
            source = selected_for_single_path.loc[mask].iloc[0]
            for col in [
                "single_path_iou_mean",
                "single_path_iou_std",
                "single_path_iou_stderr",
                "single_path_iou_n",
                "single_path_iou_draws_per_dataset",
                "single_path_iou_total_std",
                "single_path_iou_total_stderr",
                "single_path_iou_conditional_std_mean",
                "single_path_iou_conditional_std_std",
                "single_path_iou_conditional_std_stderr",
                "single_path_iou_dataset_mean_std",
                "single_path_sign_error_mean",
                "single_path_volume_mean",
                "single_path_volume_abs_err_mean",
            ]:
                target_df.at[idx, col] = source[col]
        if target_df_name == "top3_distribution_df":
            top3_distribution_df = target_df
        else:
            top3_distribution_private_df = target_df

    best_path = out_prefix.with_name(out_prefix.name + "_best.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "distribution_objective": "integrated_binary_cross_entropy",
                "posterior_mean_objective": "weighted_iou",
                "baselines": {
                    "true_excursion_volume": t_true,
                    "true_excursion_components": n_excursion_components,
                    "true_excursion_component_width_min": target_diag["component_width_min"],
                    "true_excursion_component_width_median": target_diag["component_width_median"],
                    "true_excursion_component_width_mean": target_diag["component_width_mean"],
                    "target_sampling_method": target_diag.get("target_sampling_method"),
                    "target_training_values": target_diag.get("target_training_values"),
                    "target_designs_sampled_before_rejection": target_diag.get("target_designs_sampled_before_rejection"),
                    "target_rejection_sampling": target_diag["target_rejection_sampling"],
                    "target_rejection_attempts": target_diag["target_rejection_attempts"],
                    "target_rejection_volume_min": target_diag["target_rejection_volume_min"],
                    "target_rejection_volume_max": target_diag["target_rejection_volume_max"],
                    "target_rejection_max_components": target_diag["target_rejection_max_components"],
                    "target_rejection_min_component_width": target_diag["target_rejection_min_component_width"],
                    "target_rejection_min_mean_component_width": target_diag["target_rejection_min_mean_component_width"],
                    "target_rejection_max_attempts": target_diag["target_rejection_max_attempts"],
                    "prior_probability_bce": prior_bce,
                    "oracle_constant_bce": oracle_const_bce,
                },
                "epsilon_threshold": epsilon_threshold,
                "epsilon_delta": epsilon_delta,
                "epsilon_L": epsilon_L,
                "private_refine_rounds": private_refine_rounds,
                "n_single_path_iou_draws": n_single_path_iou_draws,
                "single_path_iou_seed_offset": single_path_iou_seed_offset,
                "distribution_frontier_summary": frontier_summary,
                "top_coarse_distribution_by_bce": top_coarse_distribution.to_dict(orient="records"),
                "top_coarse_distribution_by_bce_eps_feasible": top_coarse_distribution_private.to_dict(orient="records"),
                "top3_refined_distribution_by_bce": top3_distribution_df.to_dict(orient="records"),
                "top3_refined_distribution_by_bce_eps_feasible": top3_distribution_private_df.to_dict(orient="records"),
                "top_coarse_posterior_mean_by_iou": top_coarse_mean.to_dict(orient="records"),
                "top3_refined_posterior_mean_by_iou": top3_mean_df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )
    print(f"Saved best settings to: {best_path}")

    top3_npy_path = out_prefix.with_name(out_prefix.name + "_top3_choices.npy")
    np.save(
        top3_npy_path,
        {
            "distribution_objective": "integrated_binary_cross_entropy",
            "posterior_mean_objective": "weighted_iou",
            "epsilon_threshold": epsilon_threshold,
            "epsilon_delta": epsilon_delta,
            "epsilon_L": epsilon_L,
            "private_refine_rounds": private_refine_rounds,
            "n_single_path_iou_draws": n_single_path_iou_draws,
            "single_path_iou_seed_offset": single_path_iou_seed_offset,
            "distribution_frontier_summary": frontier_summary,
            "top_coarse_distribution_by_bce": top_coarse_distribution.to_records(index=False),
            "top_coarse_distribution_by_bce_eps_feasible": top_coarse_distribution_private.to_records(index=False),
            "top3_refined_distribution_by_bce": top3_distribution_df.to_records(index=False),
            "top3_refined_distribution_by_bce_eps_feasible": top3_distribution_private_df.to_records(index=False),
            "top_coarse_posterior_mean_by_iou": top_coarse_mean.to_records(index=False),
            "top3_refined_posterior_mean_by_iou": top3_mean_df.to_records(index=False),
        },
        allow_pickle=True,
    )
    print(f"Saved top-3 choices to: {top3_npy_path}")

    # Save the full evaluated grids so the privacy-constrained frontier can be analysed post hoc.
    '''
    coarse_distribution_csv = out_prefix.with_name(out_prefix.name + "_distribution_coarse.csv")
    refined_distribution_csv = out_prefix.with_name(out_prefix.name + "_distribution_refined.csv")
    coarse_mean_csv = out_prefix.with_name(out_prefix.name + "_mean_coarse.csv")
    refined_mean_csv = out_prefix.with_name(out_prefix.name + "_mean_refined.csv")
    coarse_distribution_df.to_csv(coarse_distribution_csv, index=False)
    refined_distribution_df.to_csv(refined_distribution_csv, index=False)
    coarse_mean_df.to_csv(coarse_mean_csv, index=False)
    refined_mean_df.to_csv(refined_mean_csv, index=False)
    print(f"Saved full distribution coarse grid to: {coarse_distribution_csv}")
    print(f"Saved full distribution refined grid to: {refined_distribution_csv}")
    print(f"Saved full posterior-mean coarse grid to: {coarse_mean_csv}")
    print(f"Saved full posterior-mean refined grid to: {refined_mean_csv}")
    '''
    
    print(f"\nTrue excursion volume T_u(f_*): {t_true:.6f}")
    print(f"True excursion components N_comp(f_*): {n_excursion_components}")
    print(
        "True excursion component widths: "
        f"min={target_diag['component_width_min']:.6f}, "
        f"median={target_diag['component_width_median']:.6f}, "
        f"mean={target_diag['component_width_mean']:.6f}"
    )
    print("\nTop coarse posterior-distribution choices by BCE:")
    for i, row in enumerate(top_coarse_distribution.to_dict(orient="records"), start=1):
        print(
            f"  {i}. ell={row['ell_model']:g}, r={row['r']:g}, "
            f"d_eff={row['effective_dim_mean']:.2f}, sigma={row['sigma']:g}, "
            f"BCE={row['bce_post_mean']:.4f}, eps={row['epsilon']:.3g}, "
            f"hard_IoU={row['posterior_hard_iou_mean']:.4f}, volume_MAE={row['volume_abs_err_mean']:.4f}"
        )

    print("\nTop 3 refined posterior-distribution choices by BCE:")
    for i, row in enumerate(top3_distribution_df.to_dict(orient="records"), start=1):
        print(
            f"  {i}. ell={row['ell_model']:g}, r={row['r']:g}, "
            f"d_eff={row['effective_dim_mean']:.2f}, sigma={row['sigma']:g}, "
            f"BCE={row['bce_post_mean']:.4f}, eps={row['epsilon']:.3g}, "
            f"hard_IoU={row['posterior_hard_iou_mean']:.4f}, "
            f"E_DF[single_path_IoU]={row['single_path_iou_mean']:.4f}, "
            f"E_D[Std_F|D]={row['single_path_iou_conditional_std_mean']:.4f}, "
            f"volume_MAE={row['volume_abs_err_mean']:.4f}"
        )

    print(f"\nTop 3 refined posterior-distribution choices by BCE among epsilon<{epsilon_threshold:g}:")
    if len(top3_distribution_private_df) == 0:
        print("  none found")
    for i, row in enumerate(top3_distribution_private_df.to_dict(orient="records"), start=1):
        print(
            f"  {i}. ell={row['ell_model']:g}, r={row['r']:g}, "
            f"d_eff={row['effective_dim_mean']:.2f}, sigma={row['sigma']:g}, "
            f"BCE={row['bce_post_mean']:.4f}, eps={row['epsilon']:.3g}, "
            f"hard_IoU={row['posterior_hard_iou_mean']:.4f}, "
            f"E_DF[single_path_IoU]={row['single_path_iou_mean']:.4f}, "
            f"E_D[Std_F|D]={row['single_path_iou_conditional_std_mean']:.4f}, "
            f"volume_MAE={row['volume_abs_err_mean']:.4f}"
        )

    print("\nPrivacy-constrained BCE-selected frontier summary:")
    for key, value in frontier_summary.items():
        print(f"  {key}: {value}")

    print("\nTop coarse posterior-mean choices by IoU, searched over (ell,r) only:")
    for i, row in enumerate(top_coarse_mean.to_dict(orient="records"), start=1):
        print(
            f"  {i}. ell={row['ell_model']:g}, r={row['r']:g}, "
            f"d_eff={row['effective_dim_mean']:.2f}, "
            f"IoU={row['meanfun_iou_mean']:.4f}, sign_error={row['meanfun_sign_error_mean']:.4f}, "
            f"volume_MAE={row['meanfun_volume_abs_err_mean']:.4f}"
        )

    print("\nTop 3 refined posterior-mean choices by IoU, searched over (ell,r) only:")
    for i, row in enumerate(top3_mean_df.to_dict(orient="records"), start=1):
        print(
            f"  {i}. ell={row['ell_model']:g}, r={row['r']:g}, "
            f"d_eff={row['effective_dim_mean']:.2f}, "
            f"IoU={row['meanfun_iou_mean']:.4f}, sign_error={row['meanfun_sign_error_mean']:.4f}, "
            f"volume_MAE={row['meanfun_volume_abs_err_mean']:.4f}"
        )

    return coarse_distribution_df, refined_distribution_df, coarse_mean_df, refined_mean_df, x_target, f_target, trials


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coarse-to-refined searches: posterior-distribution BCE-only selection plus posterior-mean IoU baseline for excursion-set recovery."
    )
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-target-grid", type=int, default=800)
    parser.add_argument("--ell-true", type=float, default=0.5)
    parser.add_argument("--ell-model", type=str, nargs="+", default=["0.35,0.42,0.5,0.6,0.75,1"])
    parser.add_argument("--r-values", type=str, nargs="+", default=["0.05,0.1,0.2,0.5,1,2,5"])
    parser.add_argument("--sigma-values", type=str, nargs="+", default=["0,0.1,0.5,1,2,5"])
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument(
        "--n-posterior-draws",
        type=int,
        default=None,
        help=(
            "backwards-compatible alias for --n-single-path-iou-draws when the "
            "latter is not supplied; finite-L grid-search metrics are not computed"
        ),
    )
    parser.add_argument(
        "--n-single-path-iou-draws",
        type=int,
        default=None,
        help=(
            "posterior sample paths per noisy dataset for post-hoc IoU of the "
            "actual one-path release, evaluated only for selected top-3 BCE rows; "
            "set to 0 to disable. Defaults to --n-posterior-draws if supplied, else 1"
        ),
    )
    parser.add_argument(
        "--single-path-iou-seed-offset",
        type=int,
        default=100000,
        help="seed offset for post-hoc one-path IoU Monte Carlo draws",
    )
    parser.add_argument("--m-eps", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--top-k-coarse", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--out-prefix", type=str, default="exp_excursion_set_bce_iou_gridsearch")
    parser.add_argument(
        "--bce-clip",
        type=float,
        default=1e-6,
        help="clip posterior probabilities to [clip, 1-clip] inside BCE",
    )
    parser.add_argument(
        "--hard-set-cutoff",
        type=float,
        default=0.5,
        help="cutoff for hard set diagnostics from posterior probabilities",
    )
    parser.add_argument(
        "--epsilon-threshold",
        type=float,
        default=10.0,
        help="privacy threshold used for epsilon-feasible BCE refinement",
    )
    parser.add_argument(
        "--epsilon-delta",
        type=float,
        default=0.005,
        help="delta used in the DP epsilon calculation",
    )
    parser.add_argument(
        "--epsilon-L",
        type=int,
        default=1,
        help="number of released posterior paths used in the DP epsilon calculation",
    )
    parser.add_argument(
        "--private-refine-rounds",
        type=int,
        default=3,
        help="number of distribution-refinement rounds for the epsilon-feasible BCE branch",
    )
    parser.add_argument(
        "--no-target-sieve",
        action="store_true",
        help="disable the pre-grid-search target-volume sieve",
    )
    parser.add_argument(
        "--target-sieve-ell",
        type=float,
        default=None,
        help="lengthscale used for the prior SD in the target sieve; default is ell_true",
    )
    parser.add_argument(
        "--target-sieve-sd-factor",
        type=float,
        default=0.0,
        help="skip if |T(f_*)-1/2| is below this factor times SD(T)",
    )
    parser.add_argument(
        "--target-sieve-extreme-tol",
        type=float,
        default=1e-12,
        help="tolerance for treating T(f_*) as numerically 0 or 1",
    )
    parser.add_argument(
        "--target-reject",
        action="store_true",
        help="rejection-sample f_* until it satisfies resolvability constraints",
    )
    parser.add_argument(
        "--target-reject-volume-min",
        type=float,
        default=0.1,
        help="minimum accepted excursion volume when --target-reject is enabled",
    )
    parser.add_argument(
        "--target-reject-volume-max",
        type=float,
        default=0.9,
        help="maximum accepted excursion volume when --target-reject is enabled",
    )
    parser.add_argument(
        "--target-reject-max-components",
        type=int,
        default=None,
        help="maximum accepted number of excursion components when --target-reject is enabled",
    )
    parser.add_argument(
        "--target-reject-min-component-width",
        type=float,
        default=0.0,
        help="minimum accepted positive component width when --target-reject is enabled",
    )
    parser.add_argument(
        "--target-reject-min-mean-component-width",
        type=float,
        default=0.0,
        help="minimum accepted average positive component width when --target-reject is enabled",
    )
    parser.add_argument(
        "--target-reject-max-attempts",
        type=int,
        default=10000,
        help="maximum number of target rejection-sampling attempts",
    )
    args = parser.parse_args()

    if args.n_single_path_iou_draws is None:
        n_single_path_iou_draws = args.n_posterior_draws if args.n_posterior_draws is not None else 1
        if args.n_posterior_draws is not None:
            print(
                "Note: using --n-posterior-draws as the number of posterior paths "
                "per dataset for the selected one-path IoU diagnostic; no finite-L "
                "grid-search metric is computed."
            )
    else:
        n_single_path_iou_draws = args.n_single_path_iou_draws
        if args.n_posterior_draws is not None:
            print(
                "Note: --n-posterior-draws is ignored because "
                "--n-single-path-iou-draws was supplied explicitly."
            )

    ell_values = parse_float_list(args.ell_model)
    r_values = parse_float_list(args.r_values)
    sigma_values = parse_float_list(args.sigma_values)
    out_prefix = make_tagged_out_prefix(Path(args.out_prefix), args.seed, args.m_eps, create_dir=False)
    print(f"Outputs will use prefix if target passes sieve: {out_prefix}")

    if any(e <= 0 for e in ell_values):
        raise ValueError("all ell_model values must be positive")
    if any(r <= 0 for r in r_values):
        raise ValueError("all r values must be positive")
    if any(s < 0 for s in sigma_values):
        raise ValueError("all sigma values must be non-negative")
    if not (0.0 < args.bce_clip < 0.5):
        raise ValueError("bce_clip must lie in (0, 0.5)")
    if args.epsilon_threshold <= 0:
        raise ValueError("epsilon_threshold must be positive")
    if args.epsilon_delta <= 0 or args.epsilon_delta >= 1:
        raise ValueError("epsilon_delta must lie in (0, 1)")
    if args.epsilon_L <= 0:
        raise ValueError("epsilon_L must be positive")
    if args.private_refine_rounds < 1:
        raise ValueError("private_refine_rounds must be at least 1")
    if n_single_path_iou_draws < 0:
        raise ValueError("n_single_path_iou_draws must be non-negative")
    if args.target_reject_volume_min < 0 or args.target_reject_volume_max > 1:
        raise ValueError("target reject volume bounds must lie in [0, 1]")
    if args.target_reject_volume_min > args.target_reject_volume_max:
        raise ValueError("target reject volume min must be <= volume max")
    if args.target_reject_max_components is not None and args.target_reject_max_components < 0:
        raise ValueError("target reject max components must be non-negative")
    if args.target_reject_min_component_width < 0:
        raise ValueError("target reject min component width must be non-negative")
    if args.target_reject_min_mean_component_width < 0:
        raise ValueError("target reject min mean component width must be non-negative")
    if args.target_reject_max_attempts <= 0:
        raise ValueError("target reject max attempts must be positive")

    result = run_coarse_refine_search(
        n_train=args.n_train,
        n_target_grid=args.n_target_grid,
        ell_true=args.ell_true,
        ell_values=ell_values,
        r_values=r_values,
        sigma_values=sigma_values,
        n_trials=args.n_trials,
        m_eps=args.m_eps,
        threshold=args.threshold,
        seed=args.seed,
        out_prefix=out_prefix,
        top_k=args.top_k_coarse,
        bce_clip=args.bce_clip,
        hard_set_cutoff=args.hard_set_cutoff,
        epsilon_threshold=args.epsilon_threshold,
        epsilon_delta=args.epsilon_delta,
        epsilon_L=args.epsilon_L,
        private_refine_rounds=args.private_refine_rounds,
        target_sieve=not args.no_target_sieve,
        target_sieve_ell=args.target_sieve_ell,
        target_sieve_sd_factor=args.target_sieve_sd_factor,
        target_sieve_extreme_tol=args.target_sieve_extreme_tol,
        target_reject=args.target_reject,
        target_reject_volume_min=args.target_reject_volume_min,
        target_reject_volume_max=args.target_reject_volume_max,
        target_reject_max_components=args.target_reject_max_components,
        target_reject_min_component_width=args.target_reject_min_component_width,
        target_reject_min_mean_component_width=args.target_reject_min_mean_component_width,
        target_reject_max_attempts=args.target_reject_max_attempts,
        n_single_path_iou_draws=n_single_path_iou_draws,
        single_path_iou_seed_offset=args.single_path_iou_seed_offset,
    )
    if result is None:
        return


if __name__ == "__main__":
    main()
