#!/usr/bin/env python3
"""Recalculate one M_xi column of the paper's 1D excursion-set table.

The coarse-to-refined search evaluates one global hyperparameter setting over
independent (D,f_*) pairs. Candidates are ranked by average analytic BCE. The
reported references use the requested secondary selection:

* unconstrained: lowest epsilon among the three best average-BCE candidates;
* private: highest estimated one-path IoU among the three best average-BCE
  candidates satisfying epsilon < epsilon_0.

The cutoff C for the unconstrained analytic probability map is selected on a
separate validation sample. By default a one-path release is thresholded at
c=t; optionally, a path-value threshold c is selected on that separate
validation sample. For each main pair, the private one-path utility is estimated
from repeated independent draws from the posterior; these repetitions do not
compose privacy. The script reports medians and interquartile ranges of the
pair-level quantities and writes a copy-ready one-column LaTeX table.

Privacy is calculated through the tightened 1D-exponential accountant in
dp_utils.py, including the improved RDP bound and RDP-to-DP conversion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# NumPy 2.0 added ``trapezoid``; ``trapz`` is numerically equivalent and keeps
# the experiment runnable with older NumPy releases.
_trapezoid = getattr(np, "trapezoid", None)
if _trapezoid is None:  # pragma: no cover - exercised with NumPy < 2.0
    _trapezoid = np.trapz

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


_REQUIRED_DP_ACCOUNTANT_VERSION = "tight-rdp-2026-09"

try:
    import dp_utils as _dp_utils
except ImportError as exc:  # pragma: no cover - fail only if accounting is used
    _dp_utils = None
    _DP_UTILS_IMPORT_ERROR = exc
else:
    _loaded_accountant_version = getattr(_dp_utils, "ACCOUNTANT_VERSION", None)
    if _loaded_accountant_version != _REQUIRED_DP_ACCOUNTANT_VERSION:
        _DP_UTILS_IMPORT_ERROR = ImportError(
            "dp_utils.py does not provide the required tightened accountant "
            f"version {_REQUIRED_DP_ACCOUNTANT_VERSION!r}; found "
            f"{_loaded_accountant_version!r}."
        )
        _dp_utils = None
    else:
        _DP_UTILS_IMPORT_ERROR = None


def _require_tight_dp_utils():
    """Return the tightened accountant module or raise a targeted error."""
    if _dp_utils is None:
        raise ImportError(
            "Could not load the tightened accountant from dp_utils.py. Place "
            "the current dp_utils.py beside this script."
        ) from _DP_UTILS_IMPORT_ERROR
    return _dp_utils


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
    exact finite-dimensional OU draw at the random training covariates. The
    latent target itself is left on its original GP scale. Its sup-norm in the
    bounded-response model is approximated on the union of the target mesh and
    all sampled training locations. This guarantees that every simulated
    response obeys |y_i| <= 1 when 0 <= m_eps < 1.
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
        # Include the training locations in the mesh approximation of
        # ||f_*||_infty.  Using only x_target could let an inter-mesh training
        # value exceed the declared response bound after rescaling.
        max_abs = float(np.max(np.abs(f_all_raw)))
        scale = 0.0 if max_abs <= 1e-14 else (1.0 - m_eps) / max_abs
        # Keep f_* itself as the original GP draw. Only the response signal is
        # normalised, as in y=(1-M_xi)f_*(x)/||f_*||_infty+xi.
        f_target = f_target_raw

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
    return (np.asarray(values) >= threshold).astype(float)


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
    """Grid-based diagnostics for the true excursion set {values >= threshold}."""
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
    return float(_trapezoid(integrand, h) / np.pi)


def posterior_excursion_probability(
    mu: np.ndarray,
    cov_diag: np.ndarray,
    sigma: float,
    threshold: float,
) -> np.ndarray:
    """
    p_D(x) = P(F(x) >= threshold | D).

    In the sigma=0 or zero-variance limit, this becomes the hard posterior-mean
    excursion indicator 1{mu(x)>=threshold}.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    mu = np.asarray(mu, dtype=float)
    cov_diag = np.asarray(cov_diag, dtype=float)

    if sigma == 0.0:
        return (mu >= threshold).astype(float)

    sd = sigma * np.sqrt(np.maximum(cov_diag, 0.0))
    probs = np.empty_like(mu, dtype=float)
    mask = sd > 1e-14
    probs[mask] = _normal_cdf((mu[mask] - threshold) / sd[mask])
    probs[~mask] = (mu[~mask] >= threshold).astype(float)
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

            s_draws = f_draws >= threshold
            inter = weights @ np.logical_and(s_draws, s_true.reshape(-1, 1)).astype(float)
            union = weights @ np.logical_or(s_draws, s_true.reshape(-1, 1)).astype(float)
            ious = np.ones_like(union, dtype=float)
            np.divide(inter, union, out=ious, where=union > 1e-14)
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
        factors = [0.7, 0.85, 1.0, 1.15, 1.3]
        return unique_sorted(center * f for f in factors if center * f > 0)
    if kind == "r":
        factors = [0.5, 0.7, 0.7, 0.85, 0.92, 1.0, 1.15, 1.3, 1.5]
        return unique_sorted(center * f for f in factors if center * f > 0)
    if kind == "sigma":
        if center == 0.0:
            return [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75]
        factors = [0.55, 0.7, 0.85, 0.92, 1.0, 1.1, 1.2, 1.3, 1.5]
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
    M_Y: float = 1.0,
) -> float:
    """Compute the tightened 1D exponential-kernel DP bound for one candidate."""
    if sigma <= 0.0:
        return float("inf")
    dp = _require_tight_dp_utils()
    eps = dp.epsilon_for_delta(
        n=n,
        r=float(r),
        kappa=float(np.exp(-1.0 / float(ell))),
        sigma=float(sigma),
        eta=0.0,
        L=L,
        delta=delta,
        model="exp_1d",
        M_Y=M_Y,
        return_details=False,
    )
    return float(eps)


def add_epsilon_column(
    df: pd.DataFrame,
    n: int,
    delta: float,
    L: int,
    M_Y: float = 1.0,
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
            M_Y=M_Y,
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
                s_mu = (mu >= threshold).astype(float)
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
    if len(top3_distribution_private_df) == 0:
        raise RuntimeError(
            f"No refined candidate satisfies epsilon < {epsilon_threshold:g}; "
            "expand the r/sigma search range before reporting a table column."
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
                "release_L": int(epsilon_L),
                "release_repetitions_per_pair": int(n_single_path_iou_draws),
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
            "release_L": int(epsilon_L),
            "release_repetitions_per_pair": int(n_single_path_iou_draws),
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
# Prior-average grid search over independent (f_*, D) pairs
# ----------------------------

def _summary_stats_array(values: list[float] | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def summarize_pair_pool(pairs: list[dict]) -> dict:
    """Compact diagnostics for the sampled prior-average (f_*,D) pool."""
    return {
        "n_pairs": int(len(pairs)),
        "true_excursion_volume": _summary_stats_array([p["true_volume"] for p in pairs]),
        "true_excursion_components": _summary_stats_array([p["target_diag"]["n_components"] for p in pairs]),
        "true_excursion_component_width_min": _summary_stats_array([p["target_diag"]["component_width_min"] for p in pairs]),
        "true_excursion_component_width_median": _summary_stats_array([p["target_diag"]["component_width_median"] for p in pairs]),
        "true_excursion_component_width_mean": _summary_stats_array([p["target_diag"]["component_width_mean"] for p in pairs]),
        "target_rejection_attempts": _summary_stats_array([p["target_diag"].get("target_rejection_attempts", 1) for p in pairs]),
        "observed_response_max": _summary_stats_array([p["observed_response_max"] for p in pairs]),
    }


def sample_prior_average_pairs(
    n_pairs: int,
    n_train: int,
    x_target: np.ndarray,
    ell_true: float,
    m_eps: float,
    threshold: float,
    seed: int,
    M_Y: float = 1.0,
    target_reject: bool = False,
    target_reject_volume_min: float = 0.1,
    target_reject_volume_max: float = 0.9,
    target_reject_max_components: int | None = None,
    target_reject_min_component_width: float = 0.0,
    target_reject_min_mean_component_width: float = 0.0,
    target_reject_max_attempts: int = 10000,
) -> tuple[list[dict], dict]:
    """Sample and store independent accepted (f_*, D) pairs in RAM.

    Each pair contains its own accepted latent target f_* on the common target
    grid and one noisy training dataset D sampled from that target. The target
    rejection rule is the same resolvability rule used in the per-target script:
    acceptance is based only on the target-grid excursion diagnostics.
    """
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    weights = trapezoid_weights(x_target)
    pairs: list[dict] = []

    iterator = range(n_pairs)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="sample_pairs", unit="pair", dynamic_ncols=True)

    for pair_idx in iterator:
        # Use separated seeds so that each pair has an independent design, path,
        # and noise draw while remaining reproducible from the top-level seed.
        pair_seed = int(seed + 10007 * pair_idx)
        f_target, target_diag, trials = sample_fixed_target_and_trials_joint_ou(
            x_target=x_target,
            ell_true=ell_true,
            m_eps=m_eps,
            n_trials=1,
            n_train=n_train,
            seed=pair_seed,
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
        trial = trials[0]
        observed_max = float(np.max(np.abs(trial["y_train"])))
        if observed_max > M_Y + 1e-12 * max(1.0, M_Y):
            raise ValueError(
                "Generated responses violate the declared DP response bound: "
                f"pair={pair_idx}, max |y_i|={observed_max:.12g} > "
                f"M_Y={M_Y:.12g}."
            )
        s_true = true_excursion_indicator(f_target, threshold).astype(float)
        true_volume = float(np.dot(weights, s_true))
        pairs.append(
            {
                "pair_index": int(pair_idx),
                "seed": pair_seed,
                "f_target": f_target,
                "s_true": s_true,
                "true_volume": true_volume,
                "target_diag": target_diag,
                "x_train": trial["x_train"],
                "y_train": trial["y_train"],
                "y_clean": trial["y_clean"],
                "noise": trial["noise"],
                "observed_response_max": observed_max,
            }
        )

    return pairs, summarize_pair_pool(pairs)


def evaluate_distribution_candidates_prior_average(
    candidates: list[tuple[float, float, float]],
    pairs: list[dict],
    x_target: np.ndarray,
    n_train: int,
    n_target_grid: int,
    ell_true: float,
    m_eps: float,
    threshold: float,
    stage: str,
    bce_clip: float = 1e-6,
    hard_set_cutoff: float = 0.5,
) -> pd.DataFrame:
    """Evaluate candidates by averaging BCE/IoU over stored (f_*,D) pairs.

    This is the prior-average analogue of evaluate_distribution_candidates().
    Unlike the per-target oracle search, there is no fixed f_* shared across all
    trials: each stored pair supplies its own target-grid truth and one training
    dataset.
    """
    if not pairs:
        raise ValueError("pairs must be non-empty")
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
    domain_length = float(x_target[-1] - x_target[0])
    prior_bce = float(domain_length * np.log(2.0))
    true_volumes = np.asarray([p["true_volume"] for p in pairs], dtype=float)
    oracle_const_bces = []
    for tv in true_volumes:
        oracle_const_p = float(np.clip(tv / domain_length, bce_clip, 1.0 - bce_clip))
        oracle_const_bces.append(
            float(-domain_length * ((tv / domain_length) * np.log(oracle_const_p)
                                    + (1.0 - tv / domain_length) * np.log(1.0 - oracle_const_p)))
        )

    rows: list[dict] = []
    total_ell = len(sigmas_by_ell_r)
    progress = tqdm(total=total_ell, desc=stage, unit="ell", dynamic_ncols=True) if tqdm is not None else None

    for ell_idx, (ell_model, r_to_sigmas) in enumerate(sorted(sigmas_by_ell_r.items()), start=1):
        # Per-ell accumulators. We compute K for each pair once for this ell and
        # reuse it across all r and sigma values under this ell.
        acc: dict[tuple[float, float], dict[str, list[float]]] = {}
        deff_vals: dict[float, list[float]] = {}
        for r, sigma_values in sorted(r_to_sigmas.items()):
            deff_vals[float(r)] = []
            for sigma in sigma_values:
                acc[(float(r), float(sigma))] = {
                    "bce": [],
                    "hard_iou": [],
                    "hard_error": [],
                    "hard_volume": [],
                    "est_volume": [],
                    "volume_abs_err": [],
                    "volume_sq_err": [],
                    "uncertainty_integral": [],
                }

        for pair in pairs:
            x_train = pair["x_train"]
            y_train = pair["y_train"]
            s_true = pair["s_true"]
            true_volume = float(pair["true_volume"])

            k_xx = exp_kernel(x_train, x_train, ell_model)
            k_tx = exp_kernel(x_target, x_train, ell_model)

            for r, sigma_values in sorted(r_to_sigmas.items()):
                r = float(r)
                deff_vals[r].append(effective_dimension_from_kernel(k_xx, r))
                mu, cov_diag = posterior_mean_and_cov_diag_from_kernels(
                    k_xx=k_xx,
                    k_tx=k_tx,
                    y_train=y_train,
                    r=r,
                )
                for sigma in sigma_values:
                    sigma = float(sigma)
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
                    a = acc[(r, sigma)]
                    a["bce"].append(bce)
                    a["hard_iou"].append(hard_iou)
                    a["hard_error"].append(hard_error)
                    a["hard_volume"].append(hard_volume)
                    a["est_volume"].append(est_volume)
                    a["volume_abs_err"].append(abs(est_volume - true_volume))
                    a["volume_sq_err"].append((est_volume - true_volume) ** 2)
                    a["uncertainty_integral"].append(uncertainty_integral)

        best_bce_this_ell = float("inf")
        for r, sigma_values in sorted(r_to_sigmas.items()):
            r = float(r)
            deff_arr = np.asarray(deff_vals[r], dtype=float)
            for sigma in sigma_values:
                sigma = float(sigma)
                a = acc[(r, sigma)]
                bce_arr = np.asarray(a["bce"], dtype=float)
                hard_iou_arr = np.asarray(a["hard_iou"], dtype=float)
                hard_error_arr = np.asarray(a["hard_error"], dtype=float)
                hard_volume_arr = np.asarray(a["hard_volume"], dtype=float)
                est_volume_arr = np.asarray(a["est_volume"], dtype=float)
                volume_abs_err_arr = np.asarray(a["volume_abs_err"], dtype=float)
                volume_sq_err_arr = np.asarray(a["volume_sq_err"], dtype=float)
                uncertainty_arr = np.asarray(a["uncertainty_integral"], dtype=float)
                best_bce_this_ell = min(best_bce_this_ell, float(np.mean(bce_arr)))

                rows.append({
                    "stage": stage,
                    "search_type": "prior_average_posterior_distribution_bce",
                    "estimator": "analytic_excursion_probability_bce_fast_prior_average",
                    "n_train": n_train,
                    "n_target_grid": n_target_grid,
                    "ell_true": ell_true,
                    "ell_model": ell_model,
                    "m_eps": m_eps,
                    "threshold": threshold,
                    "r": r,
                    "sigma": sigma,
                    "n_pairs": len(pairs),
                    # Keep n_trials for compatibility with existing analysis scripts.
                    "n_trials": len(pairs),
                    "hard_set_cutoff": hard_set_cutoff,
                    "bce_clip": bce_clip,
                    "true_excursion_volume_mean": float(np.mean(true_volumes)),
                    "true_excursion_volume_std": float(np.std(true_volumes)),
                    "prior_probability_bce": prior_bce,
                    "oracle_constant_bce_mean": float(np.mean(oracle_const_bces)),
                    "oracle_constant_bce_std": float(np.std(oracle_const_bces)),
                    "bce_post_mean": float(np.mean(bce_arr)),
                    "bce_post_std": float(np.std(bce_arr)),
                    "posterior_hard_iou_mean": float(np.mean(hard_iou_arr)),
                    "posterior_hard_iou_std": float(np.std(hard_iou_arr)),
                    "posterior_hard_error_mean": float(np.mean(hard_error_arr)),
                    "posterior_hard_error_std": float(np.std(hard_error_arr)),
                    "posterior_hard_volume_mean": float(np.mean(hard_volume_arr)),
                    "posterior_hard_volume_std": float(np.std(hard_volume_arr)),
                    "est_ET_mean": float(np.mean(est_volume_arr)),
                    "est_ET_std_across_pairs": float(np.std(est_volume_arr)),
                    "volume_abs_err_mean": float(np.mean(volume_abs_err_arr)),
                    "volume_abs_err_std": float(np.std(volume_abs_err_arr)),
                    "volume_rmse": float(np.sqrt(np.mean(volume_sq_err_arr))),
                    "uncertainty_integral_mean": float(np.mean(uncertainty_arr)),
                    "uncertainty_integral_std": float(np.std(uncertainty_arr)),
                    "effective_dim_mean": float(np.mean(deff_arr)),
                    "effective_dim_std": float(np.std(deff_arr)),
                    "effective_dim_min": float(np.min(deff_arr)),
                    "effective_dim_max": float(np.max(deff_arr)),
                })

        if progress is not None:
            progress.set_postfix(ell=f"{ell_model:g}", best_bce=f"{best_bce_this_ell:.4f}", refresh=False)
            progress.update(1)
        elif ell_idx == 1 or ell_idx == total_ell:
            print(f"[{stage} {ell_idx}/{total_ell}] ell={ell_model:g}, best_bce={best_bce_this_ell:.4f}")

    if progress is not None:
        progress.close()

    return pd.DataFrame(rows)



def estimate_single_path_iou_for_distribution_rows_pairs(
    rows: pd.DataFrame,
    pairs: list[dict],
    x_target: np.ndarray,
    threshold: float,
    n_draws_per_pair: int,
    seed: int,
    n_release_paths: int = 1,
    release_fraction_cutoff: float = 0.5,
    jitter: float = 1e-10,
) -> pd.DataFrame:
    """Estimate IoU of an L-path posterior-sample release over stored (f_*,D) pairs.

    Here n_release_paths is the actual number L of posterior sample paths that
    are released. For L=1, each draw is thresholded directly at ``threshold``.
    For L>1, the released hard set is

        {x : L^{-1} sum_{l=1}^L 1{F_D^{(l)}(x) >= threshold}
             >= release_fraction_cutoff}.

    The argument n_draws_per_pair is the number of independent repetitions of
    this L-path release used to estimate expected IoU. The
    ``release_fraction_cutoff`` argument is ignored when L=1.
    """
    if n_release_paths <= 0:
        raise ValueError("n_release_paths must be positive")
    if not (0.0 <= release_fraction_cutoff <= 1.0):
        raise ValueError("release_fraction_cutoff must lie in [0,1]")

    out = rows.copy()
    metric_cols = [
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
        "release_iou_mean",
        "release_iou_std",
        "release_L",
        "release_fraction_cutoff",
        "release_repetitions_per_pair",
    ]
    if len(out) == 0:
        for col in metric_cols:
            out[col] = []
        return out
    if n_draws_per_pair <= 0:
        for col in metric_cols:
            out[col] = np.nan
        out["single_path_iou_n"] = 0
        out["single_path_iou_draws_per_dataset"] = 0
        out["release_L"] = int(n_release_paths)
        out["release_fraction_cutoff"] = (
            np.nan if n_release_paths == 1 else float(release_fraction_cutoff)
        )
        out["release_repetitions_per_pair"] = 0
        return out

    weights = trapezoid_weights(x_target)

    unique_rows = out.copy()
    for col in ["ell_model", "r", "sigma"]:
        unique_rows[f"_{col}_round"] = unique_rows[col].astype(float).round(12)
    unique_rows = unique_rows.drop_duplicates(subset=["_ell_model_round", "_r_round", "_sigma_round"])

    estimates: dict[tuple[float, float, float], dict] = {}
    iterator = unique_rows.itertuples(index=False)
    if tqdm is not None:
        iterator = tqdm(list(iterator), desc="L_path_release_iou_selected", unit="candidate", dynamic_ncols=True)

    n_total_draws = int(n_draws_per_pair) * int(n_release_paths)

    for row in iterator:
        ell = float(row.ell_model)
        r = float(row.r)
        sigma = float(row.sigma)
        key = (round(ell, 12), round(r, 12), round(sigma, 12))
        if key in estimates:
            continue

        # Restart the stream for each candidate. This supplies common random
        # numbers across the top-three comparison and makes results invariant
        # to candidate row order.
        rng = np.random.default_rng(seed)

        iou_vals: list[float] = []
        sign_error_vals: list[float] = []
        volume_vals: list[float] = []
        volume_abs_err_vals: list[float] = []
        per_pair_iou_means: list[float] = []
        per_pair_iou_stds: list[float] = []

        for pair in pairs:
            x_train = pair["x_train"]
            y_train = pair["y_train"]
            s_true = pair["s_true"].astype(bool)
            true_volume = float(pair["true_volume"])
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
                    n_draws=n_total_draws,
                    rng=rng,
                )
                obs_noise0 = r * rng.standard_normal((len(x_train), n_total_draws))
                residual = y_train.reshape(-1, 1) - (f0_train + obs_noise0)
                beta = solve_spd_many(a, residual)
                core_draws = f0_target + k_tx @ beta
                f_draws = mu.reshape(-1, 1) + sigma * (core_draws - mu.reshape(-1, 1))
            else:
                f_draws = np.repeat(mu.reshape(-1, 1), n_total_draws, axis=1)

            if n_release_paths == 1:
                # A one-path release is simply the excursion set of that path;
                # no second vote-fraction cutoff is needed.
                s_releases = f_draws.reshape(
                    len(x_target), int(n_draws_per_pair)
                ) >= threshold
            else:
                # General L-path compatibility path.
                f_draws = f_draws.reshape(
                    len(x_target), int(n_release_paths), int(n_draws_per_pair)
                )
                release_fraction = np.mean(f_draws >= threshold, axis=1)
                s_releases = release_fraction >= release_fraction_cutoff

            inter = weights @ np.logical_and(s_releases, s_true.reshape(-1, 1)).astype(float)
            union = weights @ np.logical_or(s_releases, s_true.reshape(-1, 1)).astype(float)
            ious_arr = np.ones_like(union, dtype=float)
            np.divide(inter, union, out=ious_arr, where=union > 1e-14)
            per_pair_iou_means.append(float(np.mean(ious_arr)))
            per_pair_iou_stds.append(float(np.std(ious_arr)))

            sign_errors = weights @ np.logical_xor(s_releases, s_true.reshape(-1, 1)).astype(float)
            volumes = weights @ s_releases.astype(float)

            iou_vals.extend(np.asarray(ious_arr, dtype=float).tolist())
            sign_error_vals.extend(np.asarray(sign_errors, dtype=float).tolist())
            volume_vals.extend(np.asarray(volumes, dtype=float).tolist())
            volume_abs_err_vals.extend(np.abs(np.asarray(volumes, dtype=float) - true_volume).tolist())

        iou_arr = np.asarray(iou_vals, dtype=float)
        per_pair_mean_arr = np.asarray(per_pair_iou_means, dtype=float)
        per_pair_std_arr = np.asarray(per_pair_iou_stds, dtype=float)
        n = int(iou_arr.size)
        n_pairs = int(per_pair_mean_arr.size)
        total_std = float(np.std(iou_arr)) if n else float("nan")
        cond_std_mean = float(np.mean(per_pair_std_arr)) if n_pairs else float("nan")
        cond_std_std = float(np.std(per_pair_std_arr)) if n_pairs else float("nan")
        pair_mean_std = float(np.std(per_pair_mean_arr)) if n_pairs else float("nan")
        estimates[key] = {
            # Backward-compatible column names; for L>1 these are L-path release metrics.
            "single_path_iou_mean": float(np.mean(iou_arr)) if n else float("nan"),
            "single_path_iou_std": cond_std_mean,
            "single_path_iou_stderr": float(np.std(per_pair_mean_arr) / np.sqrt(n_pairs)) if n_pairs else float("nan"),
            "single_path_iou_n": n,
            "single_path_iou_draws_per_dataset": int(n_draws_per_pair),
            "single_path_iou_total_std": total_std,
            "single_path_iou_total_stderr": float(total_std / np.sqrt(n)) if n else float("nan"),
            "single_path_iou_conditional_std_mean": cond_std_mean,
            "single_path_iou_conditional_std_std": cond_std_std,
            "single_path_iou_conditional_std_stderr": float(cond_std_std / np.sqrt(n_pairs)) if n_pairs else float("nan"),
            "single_path_iou_dataset_mean_std": pair_mean_std,
            "single_path_sign_error_mean": float(np.mean(sign_error_vals)) if n else float("nan"),
            "single_path_volume_mean": float(np.mean(volume_vals)) if n else float("nan"),
            "single_path_volume_abs_err_mean": float(np.mean(volume_abs_err_vals)) if n else float("nan"),
            "release_iou_mean": float(np.mean(iou_arr)) if n else float("nan"),
            "release_iou_std": cond_std_mean,
            "release_L": int(n_release_paths),
            "release_fraction_cutoff": (
                np.nan if n_release_paths == 1 else float(release_fraction_cutoff)
            ),
            "release_repetitions_per_pair": int(n_draws_per_pair),
        }

    for col in metric_cols:
        out[col] = np.nan
    out["single_path_iou_n"] = 0
    out["single_path_iou_draws_per_dataset"] = 0
    out["release_L"] = int(n_release_paths)
    out["release_fraction_cutoff"] = (
        np.nan if n_release_paths == 1 else float(release_fraction_cutoff)
    )
    out["release_repetitions_per_pair"] = int(n_draws_per_pair)
    for idx, row in out.iterrows():
        key = (round(float(row["ell_model"]), 12), round(float(row["r"]), 12), round(float(row["sigma"]), 12))
        est = estimates.get(key, {})
        for col, value in est.items():
            out.at[idx, col] = value
    return out


def _copy_single_path_estimates(target_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    """Copy selected one-path diagnostics from deduplicated source rows."""
    if len(target_df) == 0 or len(source_df) == 0:
        return target_df
    out = target_df.copy()
    cols = [
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
        "release_iou_mean",
        "release_iou_std",
        "release_L",
        "release_fraction_cutoff",
        "release_repetitions_per_pair",
    ]
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    for idx, row in out.iterrows():
        mask = (
            np.isclose(source_df["ell_model"].astype(float), float(row["ell_model"]))
            & np.isclose(source_df["r"].astype(float), float(row["r"]))
            & np.isclose(source_df["sigma"].astype(float), float(row["sigma"]))
        )
        if np.any(mask):
            src = source_df.loc[mask].iloc[0]
            for col in cols:
                if col in source_df.columns:
                    out.at[idx, col] = src[col]
    return out


def add_prior_average_single_path_summary(frontier: dict, top_uncon: pd.DataFrame, top_priv: pd.DataFrame) -> dict:
    """Attach one-path diagnostics for the selected top BCE rows to the summary."""
    out = dict(frontier)
    if len(top_uncon) > 0:
        u = top_uncon.iloc[0]
        out["best_unconstrained_bce_single_path_iou"] = float(u.get("single_path_iou_mean", np.nan))
        out["best_unconstrained_bce_single_path_iou_condstd"] = float(u.get("single_path_iou_std", np.nan))
    if len(top_priv) > 0:
        p = top_priv.iloc[0]
        out["best_epsilon_feasible_bce_single_path_iou"] = float(p.get("single_path_iou_mean", np.nan))
        out["best_epsilon_feasible_bce_single_path_iou_condstd"] = float(p.get("single_path_iou_std", np.nan))
        if out.get("best_unconstrained_bce_single_path_iou", float("nan")) > 0:
            out["single_path_iou_loss_unconstrained_minus_private"] = float(
                out["best_unconstrained_bce_single_path_iou"] - out["best_epsilon_feasible_bce_single_path_iou"]
            )
            out["relative_single_path_iou_loss_unconstrained_minus_private"] = float(
                1.0 - out["best_epsilon_feasible_bce_single_path_iou"] / out["best_unconstrained_bce_single_path_iou"]
            )
        if out.get("best_unconstrained_bce_iou", float("nan")) > 0:
            out["single_path_iou_loss_vs_unconstrained_hard_map"] = float(
                out["best_unconstrained_bce_iou"] - out["best_epsilon_feasible_bce_single_path_iou"]
            )
            out["relative_single_path_iou_loss_vs_unconstrained_hard_map"] = float(
                1.0 - out["best_epsilon_feasible_bce_single_path_iou"] / out["best_unconstrained_bce_iou"]
            )
    return out



def _finite_np(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def median_iqr_summary(values: list[float] | np.ndarray) -> dict:
    """Return n, median, q25, q75, min, max for finite values."""
    arr = _finite_np(values)
    if arr.size == 0:
        return {
            "n": 0,
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _fmt_float(x: float, digits: int = 3) -> str:
    if not np.isfinite(float(x)):
        return "NA"
    return f"{float(x):.{digits}f}"


def _fmt_pct(x: float, digits: int = 1) -> str:
    if not np.isfinite(float(x)):
        return "NA"
    value = 100.0 * float(x)
    if abs(value) < 0.5 * 10.0 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}\\%"


def _fmt_median_iqr(summary: dict, kind: str = "num") -> str:
    if kind == "pct":
        digits = 0 if abs(100.0 * float(summary["median"])) >= 100.0 else 1
        return (
            f"{_fmt_pct(summary['median'], digits)}\\,["
            f"{_fmt_pct(summary['q25'], digits)},{_fmt_pct(summary['q75'], digits)}]"
        )
    if kind == "deff":
        magnitude = abs(float(summary["median"]))
        digits = 1 if magnitude >= 10.0 else (2 if magnitude >= 1.0 else 3)
        return (
            f"{_fmt_float(summary['median'], digits)}\\,["
            f"{_fmt_float(summary['q25'], digits)},{_fmt_float(summary['q75'], digits)}]"
        )
    if kind == "epsilon":
        return f"{_fmt_float(summary['median'], 2)}\\,[{_fmt_float(summary['q25'], 2)},{_fmt_float(summary['q75'], 2)}]"
    if kind == "small":
        return f"{_fmt_float(summary['median'], 4)}\\,[{_fmt_float(summary['q25'], 4)},{_fmt_float(summary['q75'], 4)}]"
    return f"{_fmt_float(summary['median'], 3)}\\,[{_fmt_float(summary['q25'], 3)},{_fmt_float(summary['q75'], 3)}]"


def _fmt_scalar(x: float, kind: str = "num") -> str:
    if kind == "pct":
        return _fmt_pct(x)
    if kind == "epsilon":
        if not np.isfinite(float(x)):
            return "NA"
        value = float(x)
        if abs(value) >= 1.0e4:
            exponent = int(np.floor(np.log10(abs(value))))
            coefficient = value / (10.0 ** exponent)
            return f"{coefficient:.3g}\\cdotp 10^{{{exponent}}}"
        if abs(value) >= 100.0:
            return f"{value:.0f}"
        if abs(value) >= 10.0:
            return f"{value:.1f}"
        return f"{value:.2f}"
    if kind == "small":
        return _fmt_float(x, 4)
    return _fmt_float(x, 3)


def select_main_text_reference_rows(
    top3_unconstrained: pd.DataFrame,
    top3_private: pd.DataFrame,
    private_iou_column: str = "single_path_iou_mean",
) -> tuple[pd.Series | None, pd.Series | None]:
    """Select the two-stage reference rows requested for the table.

    1. Among the three lowest-average-BCE unconstrained rows, choose the one
       with the lowest epsilon.
    2. Among the three lowest-average-BCE rows satisfying epsilon < epsilon_0,
       choose the one with the highest estimated sampled-path IoU in
       ``private_iou_column``.
    """
    uncon_row = None
    if len(top3_unconstrained) > 0:
        uncon_row = top3_unconstrained.sort_values(
            ["epsilon", "bce_post_mean", "ell_model", "r", "sigma"],
            ascending=[True, True, True, True, True],
        ).iloc[0]

    priv_row = None
    if len(top3_private) > 0:
        if (
            private_iou_column in top3_private.columns
            and top3_private[private_iou_column].notna().any()
        ):
            priv_row = top3_private.sort_values(
                [private_iou_column, "bce_post_mean", "ell_model", "r", "sigma"],
                ascending=[False, True, True, True, True],
            ).iloc[0]
        else:
            priv_row = top3_private.sort_values(
                ["bce_post_mean", "ell_model", "r", "sigma"],
                ascending=[True, True, True, True],
            ).iloc[0]

    return uncon_row, priv_row


def _candidate_signature(row: pd.Series) -> tuple[float, float, float]:
    """Stable identity for a hyperparameter candidate."""
    return (
        round(float(row["ell_model"]), 12),
        round(float(row["r"]), 12),
        round(float(row["sigma"]), 12),
    )



def evaluate_candidate_pairwise_for_main_table(
    row: pd.Series,
    pairs: list[dict],
    x_target: np.ndarray,
    threshold: float,
    bce_clip: float,
    hard_set_cutoff: float,
    n_draws_per_pair: int,
    seed: int,
    hard_map_cutoff_override: float | None = None,
    path_threshold_override: float | None = None,
    release_fraction_cutoff_override: float | None = None,
    n_release_paths: int = 1,
    jitter: float = 1e-10,
) -> dict:
    """Compute pair-level diagnostics for one selected candidate.

    For L=1, the released set is the excursion set of the sampled path,
    ``{x : F_D^(1)(x) >= threshold}``. For L>1, it is

        {x : L^{-1} sum_l 1{F_D^{(l)}(x) >= threshold}
             >= release_fraction_cutoff}.

    n_draws_per_pair is the number of independent repetitions of this L-path
    release used to estimate expected utility. The release-fraction cutoff is
    ignored when L=1.
    """
    if n_release_paths <= 0:
        raise ValueError("n_release_paths must be positive")

    ell = float(row["ell_model"])
    r = float(row["r"])
    sigma = float(row["sigma"])
    epsilon = float(row.get("epsilon", np.nan))

    weights = trapezoid_weights(x_target)
    rng = np.random.default_rng(seed)
    hard_map_cutoff_to_use = (
        float(hard_set_cutoff)
        if hard_map_cutoff_override is None
        else float(hard_map_cutoff_override)
    )
    # Kept for backward compatibility; by default this is the scientific threshold.
    path_threshold_to_use = (
        float(threshold)
        if path_threshold_override is None
        else float(path_threshold_override)
    )
    release_fraction_cutoff_to_use = (
        0.5
        if release_fraction_cutoff_override is None
        else float(release_fraction_cutoff_override)
    )
    if not (0.0 <= release_fraction_cutoff_to_use <= 1.0):
        raise ValueError("release_fraction_cutoff_override must lie in [0,1]")

    raw: dict[str, list[float]] = {
        "bce": [],
        "hard_map_iou": [],
        "hard_map_error": [],
        "hard_map_volume": [],
        "posterior_expected_volume": [],
        "volume_abs_err": [],
        "effective_dim": [],
        # Backward-compatible names; for L>1 these are L-path vote release metrics.
        "one_path_iou_mean": [],
        "one_path_iou_std": [],
        "one_path_volume_abs_err_mean": [],
        "one_path_sign_error_mean": [],
        "release_iou_mean": [],
        "release_iou_std": [],
    }

    n_total_draws = int(n_draws_per_pair) * int(n_release_paths)

    for pair in pairs:
        x_train = pair["x_train"]
        y_train = pair["y_train"]
        s_true = pair["s_true"].astype(bool)
        s_true_float = s_true.astype(float)
        true_volume = float(pair["true_volume"])

        k_xx = exp_kernel(x_train, x_train, ell)
        k_tx = exp_kernel(x_target, x_train, ell)
        raw["effective_dim"].append(effective_dimension_from_kernel(k_xx, r))

        mu, cov_diag = posterior_mean_and_cov_diag_from_kernels(
            k_xx=k_xx,
            k_tx=k_tx,
            y_train=y_train,
            r=r,
        )
        p_exc = posterior_excursion_probability(
            mu=mu,
            cov_diag=cov_diag,
            sigma=sigma,
            threshold=threshold,
        )
        raw["bce"].append(integrated_bce_soft(p_exc, s_true_float, weights, clip=bce_clip))

        s_hard = (p_exc >= hard_map_cutoff_to_use).astype(float)
        raw["hard_map_iou"].append(hard_set_iou(s_hard, s_true_float, weights))
        raw["hard_map_error"].append(hard_set_error(s_hard, s_true_float, weights))
        hard_volume = float(np.dot(weights, s_hard))
        raw["hard_map_volume"].append(hard_volume)
        est_volume = float(np.dot(weights, p_exc))
        raw["posterior_expected_volume"].append(est_volume)
        raw["volume_abs_err"].append(abs(est_volume - true_volume))

        if n_draws_per_pair > 0:
            a = symmetrize(k_xx + (r ** 2 + jitter) * np.eye(k_xx.shape[0]))
            if sigma > 0.0:
                f0_target, f0_train = sample_exp_prior_target_and_train(
                    x_target=x_target,
                    x_train=x_train,
                    ell=ell,
                    n_draws=n_total_draws,
                    rng=rng,
                )
                obs_noise0 = r * rng.standard_normal((len(x_train), n_total_draws))
                residual = y_train.reshape(-1, 1) - (f0_train + obs_noise0)
                beta = solve_spd_many(a, residual)
                core_draws = f0_target + k_tx @ beta
                f_draws = mu.reshape(-1, 1) + sigma * (core_draws - mu.reshape(-1, 1))
            else:
                f_draws = np.repeat(mu.reshape(-1, 1), n_total_draws, axis=1)

            if n_release_paths == 1:
                # Directly threshold each posterior path at the scientific
                # excursion level.  For L=1 this is equivalent to every vote
                # cutoff c in (0,1].
                s_releases = f_draws.reshape(
                    len(x_target), int(n_draws_per_pair)
                ) >= path_threshold_to_use
            else:
                f_draws = f_draws.reshape(
                    len(x_target), int(n_release_paths), int(n_draws_per_pair)
                )
                release_fraction = np.mean(
                    f_draws >= path_threshold_to_use, axis=1
                )
                s_releases = release_fraction >= release_fraction_cutoff_to_use

            inter = weights @ np.logical_and(s_releases, s_true.reshape(-1, 1)).astype(float)
            union = weights @ np.logical_or(s_releases, s_true.reshape(-1, 1)).astype(float)
            ious = np.ones_like(union, dtype=float)
            np.divide(inter, union, out=ious, where=union > 1e-14)
            volumes = weights @ s_releases.astype(float)
            sign_errors = weights @ np.logical_xor(s_releases, s_true.reshape(-1, 1)).astype(float)

            raw["one_path_iou_mean"].append(float(np.mean(ious)))
            raw["one_path_iou_std"].append(float(np.std(ious)))
            raw["one_path_volume_abs_err_mean"].append(float(np.mean(np.abs(volumes - true_volume))))
            raw["one_path_sign_error_mean"].append(float(np.mean(sign_errors)))
            raw["release_iou_mean"].append(float(np.mean(ious)))
            raw["release_iou_std"].append(float(np.std(ious)))
        else:
            for key in [
                "one_path_iou_mean",
                "one_path_iou_std",
                "one_path_volume_abs_err_mean",
                "one_path_sign_error_mean",
                "release_iou_mean",
                "release_iou_std",
            ]:
                raw[key].append(float("nan"))

    summaries = {name: median_iqr_summary(vals) for name, vals in raw.items()}
    return {
        "hyperparameters": {
            "ell_model": ell,
            "r": r,
            "sigma": sigma,
            "epsilon": epsilon,
            "bce_post_mean": float(row.get("bce_post_mean", np.nan)),
            "posterior_hard_iou_mean": float(row.get("posterior_hard_iou_mean", np.nan)),
            "single_path_iou_mean": float(row.get("single_path_iou_mean", np.nan)),
            "release_iou_mean": float(row.get("release_iou_mean", row.get("single_path_iou_mean", np.nan))),
            "hard_map_cutoff": hard_map_cutoff_to_use,
            "path_threshold": path_threshold_to_use,
            "release_fraction_cutoff": (
                None if n_release_paths == 1 else release_fraction_cutoff_to_use
            ),
            "release_rule": (
                "direct_path_threshold" if n_release_paths == 1 else "path_vote_fraction"
            ),
            "release_L": int(n_release_paths),
            "release_repetitions_per_pair": int(n_draws_per_pair),
        },
        "raw": raw,
        "summary": summaries,
    }



def _iou_curve_for_threshold_grid(
    scores: np.ndarray,
    pred_weights: np.ndarray,
    inter_weights: np.ndarray,
    true_volume: float,
    threshold_grid: np.ndarray,
) -> np.ndarray:
    """Weighted IoU({score >= t}, true_set) for all t in threshold_grid.

    This uses sorting/cumulative sums, avoiding an O(n_grid * n_points)
    thresholding loop.  Ties match the script's final-table convention
    score >= threshold.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    pred_weights = np.asarray(pred_weights, dtype=float).reshape(-1)
    inter_weights = np.asarray(inter_weights, dtype=float).reshape(-1)
    threshold_grid = np.asarray(threshold_grid, dtype=float).reshape(-1)

    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    wp = pred_weights[order]
    wi = inter_weights[order]

    cwp = np.concatenate(([0.0], np.cumsum(wp)))
    cwi = np.concatenate(([0.0], np.cumsum(wi)))
    # side="left" gives score >= threshold.
    idx = np.searchsorted(s, threshold_grid, side="left")

    pred_vol = cwp[-1] - cwp[idx]
    inter = cwi[-1] - cwi[idx]
    union = true_volume + pred_vol - inter
    out = np.ones_like(union, dtype=float)
    np.divide(inter, union, out=out, where=union > 1e-14)
    return out


def _choose_best_threshold(
    threshold_grid: np.ndarray,
    mean_iou_curve: np.ndarray,
    default_value: float,
) -> tuple[float, float]:
    """Choose best threshold, breaking ties by closeness to default_value."""
    threshold_grid = np.asarray(threshold_grid, dtype=float)
    mean_iou_curve = np.asarray(mean_iou_curve, dtype=float)
    if threshold_grid.size == 0 or mean_iou_curve.size == 0:
        return float("nan"), float("nan")
    best = float(np.nanmax(mean_iou_curve))
    tol = 1e-12
    candidates = np.where(np.isfinite(mean_iou_curve) & (mean_iou_curve >= best - tol))[0]
    if candidates.size == 0:
        idx = int(np.nanargmax(mean_iou_curve))
    else:
        idx = int(candidates[np.argmin(np.abs(threshold_grid[candidates] - default_value))])
    return float(threshold_grid[idx]), float(mean_iou_curve[idx])


def tune_one_path_value_threshold_for_candidate(
    row: pd.Series,
    pairs: list[dict],
    x_target: np.ndarray,
    true_excursion_threshold: float,
    n_draws_per_pair: int,
    seed: int,
    path_value_threshold_grid: np.ndarray,
    jitter: float = 1e-10,
) -> dict:
    """Select a sampled-path value threshold c on validation pairs.

    For every validation pair and posterior draw, this evaluates

        IoU({x: F_D^(1)(x) >= c}, {x: f_*(x) >= t})

    on ``path_value_threshold_grid`` and maximises the mean IoU over both pairs
    and draws. Ties are broken in favour of the value closest to the scientific
    excursion threshold t. The validation draws are utility Monte Carlo only;
    they are not additional releases and therefore do not compose privacy.
    """
    if not pairs:
        raise ValueError("pairs must be non-empty")
    if n_draws_per_pair <= 0:
        raise ValueError("n_draws_per_pair must be positive")

    grid = np.asarray(path_value_threshold_grid, dtype=float).reshape(-1)
    if grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("path_value_threshold_grid must contain at least two finite values")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("path_value_threshold_grid must be strictly increasing")

    ell = float(row["ell_model"])
    r = float(row["r"])
    sigma = float(row["sigma"])
    weights = trapezoid_weights(x_target)
    rng = np.random.default_rng(seed)
    iou_sum = np.zeros(grid.size, dtype=float)
    n_iou = 0

    for pair in pairs:
        x_train = pair["x_train"]
        y_train = pair["y_train"]
        s_true = pair["s_true"].astype(bool)
        true_volume = float(pair["true_volume"])
        inter_weights = weights * s_true.astype(float)

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
                n_draws=n_draws_per_pair,
                rng=rng,
            )
            obs_noise0 = r * rng.standard_normal((len(x_train), n_draws_per_pair))
            residual = y_train.reshape(-1, 1) - (f0_train + obs_noise0)
            beta = solve_spd_many(a, residual)
            core_draws = f0_target + k_tx @ beta
            f_draws = mu.reshape(-1, 1) + sigma * (
                core_draws - mu.reshape(-1, 1)
            )
        else:
            f_draws = np.repeat(mu.reshape(-1, 1), n_draws_per_pair, axis=1)

        for draw_idx in range(n_draws_per_pair):
            iou_sum += _iou_curve_for_threshold_grid(
                scores=f_draws[:, draw_idx],
                pred_weights=weights,
                inter_weights=inter_weights,
                true_volume=true_volume,
                threshold_grid=grid,
            )
            n_iou += 1

    mean_iou_curve = iou_sum / float(n_iou)
    selected_c, selected_mean_iou = _choose_best_threshold(
        threshold_grid=grid,
        mean_iou_curve=mean_iou_curve,
        default_value=true_excursion_threshold,
    )
    default_idx = int(np.argmin(np.abs(grid - true_excursion_threshold)))
    selected_idx = int(np.argmin(np.abs(grid - selected_c)))
    return {
        "ell_model": ell,
        "r": r,
        "sigma": sigma,
        "path_value_threshold_c": selected_c,
        "path_value_mean_iou": selected_mean_iou,
        "scientific_threshold_t": float(true_excursion_threshold),
        "scientific_threshold_grid_mean_iou": float(mean_iou_curve[default_idx]),
        "path_value_threshold_grid": grid.tolist(),
        "path_value_iou_curve": mean_iou_curve.tolist(),
        "selected_at_grid_boundary": bool(
            selected_idx == 0 or selected_idx == grid.size - 1
        ),
        "n_validation_pairs": int(len(pairs)),
        "n_draws_per_validation_pair": int(n_draws_per_pair),
        "common_random_seed": int(seed),
    }



def tune_thresholds_for_selected_candidate(
    row: pd.Series,
    pairs: list[dict],
    x_target: np.ndarray,
    threshold: float,
    hard_set_cutoff: float,
    n_draws_per_pair: int,
    seed: int,
    map_threshold_grid: np.ndarray,
    path_threshold_grid: np.ndarray,
    n_release_paths: int = 1,
    jitter: float = 1e-10,
) -> dict:
    """Tune final hard-map cutoff and L-path vote-fraction cutoff.

    This is called only after the final candidates have been selected.  It does
    not affect grid search, BCE selection, epsilon computation, or candidate
    ranking.  The released set is

        {x : L^{-1} sum_l 1{F_D^{(l)}(x) >= threshold} >= q}.

    The argument path_threshold_grid is retained for backward compatibility but
    is now interpreted as the grid of vote-fraction cutoffs q in [0,1].
    """
    if n_release_paths <= 0:
        raise ValueError("n_release_paths must be positive")

    ell = float(row["ell_model"])
    r = float(row["r"])
    sigma = float(row["sigma"])

    weights = trapezoid_weights(x_target)
    rng = np.random.default_rng(seed)

    map_threshold_grid = np.asarray(map_threshold_grid, dtype=float)
    fraction_cutoff_grid = np.asarray(path_threshold_grid, dtype=float)

    map_iou_sum = np.zeros(map_threshold_grid.size, dtype=float)
    map_count = 0

    release_iou_sum = np.zeros(fraction_cutoff_grid.size, dtype=float)
    release_count = 0

    n_total_draws = int(n_draws_per_pair) * int(n_release_paths)

    for pair in pairs:
        x_train = pair["x_train"]
        y_train = pair["y_train"]
        s_true = pair["s_true"].astype(bool)
        true_volume = float(pair["true_volume"])
        pred_weights = weights
        inter_weights = weights * s_true.astype(float)

        k_xx = exp_kernel(x_train, x_train, ell)
        k_tx = exp_kernel(x_target, x_train, ell)
        mu, cov_diag = posterior_mean_and_cov_diag_from_kernels(
            k_xx=k_xx,
            k_tx=k_tx,
            y_train=y_train,
            r=r,
        )
        p_exc = posterior_excursion_probability(
            mu=mu,
            cov_diag=cov_diag,
            sigma=sigma,
            threshold=threshold,
        )
        map_iou_sum += _iou_curve_for_threshold_grid(
            scores=p_exc,
            pred_weights=pred_weights,
            inter_weights=inter_weights,
            true_volume=true_volume,
            threshold_grid=map_threshold_grid,
        )
        map_count += 1

        if n_draws_per_pair > 0 and fraction_cutoff_grid.size > 0:
            a = symmetrize(k_xx + (r ** 2 + jitter) * np.eye(k_xx.shape[0]))
            if sigma > 0.0:
                f0_target, f0_train = sample_exp_prior_target_and_train(
                    x_target=x_target,
                    x_train=x_train,
                    ell=ell,
                    n_draws=n_total_draws,
                    rng=rng,
                )
                obs_noise0 = r * rng.standard_normal((len(x_train), n_total_draws))
                residual = y_train.reshape(-1, 1) - (f0_train + obs_noise0)
                beta = solve_spd_many(a, residual)
                core_draws = f0_target + k_tx @ beta
                f_draws = mu.reshape(-1, 1) + sigma * (core_draws - mu.reshape(-1, 1))
            else:
                f_draws = np.repeat(mu.reshape(-1, 1), n_total_draws, axis=1)

            f_draws = f_draws.reshape(len(x_target), int(n_release_paths), int(n_draws_per_pair))
            release_fraction = np.mean(f_draws >= threshold, axis=1)

            pair_curve = np.zeros(fraction_cutoff_grid.size, dtype=float)
            for j in range(release_fraction.shape[1]):
                pair_curve += _iou_curve_for_threshold_grid(
                    scores=release_fraction[:, j],
                    pred_weights=pred_weights,
                    inter_weights=inter_weights,
                    true_volume=true_volume,
                    threshold_grid=fraction_cutoff_grid,
                )
            release_iou_sum += pair_curve / float(release_fraction.shape[1])
            release_count += 1

    map_curve = map_iou_sum / max(map_count, 1)
    hard_map_threshold, hard_map_mean_iou = _choose_best_threshold(
        map_threshold_grid,
        map_curve,
        default_value=hard_set_cutoff,
    )

    if release_count > 0:
        release_curve = release_iou_sum / float(release_count)
        release_fraction_cutoff, release_mean_iou = _choose_best_threshold(
            fraction_cutoff_grid,
            release_curve,
            default_value=0.5,
        )
    else:
        release_curve = np.full(fraction_cutoff_grid.shape, np.nan, dtype=float)
        release_fraction_cutoff, release_mean_iou = 0.5, float("nan")

    map_default_idx = int(np.argmin(np.abs(map_threshold_grid - hard_set_cutoff))) if map_threshold_grid.size else 0
    release_default_idx = int(np.argmin(np.abs(fraction_cutoff_grid - 0.5))) if fraction_cutoff_grid.size else 0

    return {
        "ell_model": ell,
        "r": r,
        "sigma": sigma,
        "hard_map_threshold": hard_map_threshold,
        "hard_map_mean_iou": hard_map_mean_iou,
        "hard_map_default_threshold": float(hard_set_cutoff),
        "hard_map_default_grid_mean_iou": float(map_curve[map_default_idx]) if map_threshold_grid.size else float("nan"),
        "hard_map_threshold_grid": map_threshold_grid.tolist(),
        "hard_map_iou_curve": map_curve.tolist(),
        # Backward-compatible names for JSON/printing.
        "path_threshold": float(threshold),
        "path_mean_iou": release_mean_iou,
        "path_default_threshold": float(threshold),
        "path_default_grid_mean_iou": float(release_curve[release_default_idx]) if fraction_cutoff_grid.size else float("nan"),
        "path_threshold_grid": fraction_cutoff_grid.tolist(),
        "path_iou_curve": release_curve.tolist(),
        # Explicit L-path release names.
        "release_L": int(n_release_paths),
        "release_fraction_cutoff": release_fraction_cutoff,
        "release_fraction_mean_iou": release_mean_iou,
        "release_fraction_default_cutoff": 0.5,
        "release_fraction_default_grid_mean_iou": float(release_curve[release_default_idx]) if fraction_cutoff_grid.size else float("nan"),
        "release_fraction_cutoff_grid": fraction_cutoff_grid.tolist(),
        "release_fraction_iou_curve": release_curve.tolist(),
    }


def build_main_text_median_iqr_table(
    uncon_row: pd.Series | None,
    priv_row: pd.Series | None,
    uncon_diag: dict | None,
    priv_diag: dict | None,
    m_xi: float,
    epsilon_threshold: float = 10.0,
    one_path_threshold_c: float | None = None,
    one_path_threshold_tuned: bool = False,
) -> tuple[pd.DataFrame, dict, str]:
    """Build exactly one M_xi column of the paper table."""
    if uncon_row is None or priv_row is None or uncon_diag is None or priv_diag is None:
        empty = pd.DataFrame(columns=["metric", "value"])
        return empty, {}, ""

    uncon_raw = {k: _finite_np(v) for k, v in uncon_diag["raw"].items()}
    priv_raw = {k: _finite_np(v) for k, v in priv_diag["raw"].items()}

    # Pairwise relative quantities use the common stored pair order.  If a
    # denominator is nonpositive/nonfinite, the corresponding value is discarded.
    bce_den = np.asarray(uncon_diag["raw"]["bce"], dtype=float)
    bce_num = np.asarray(priv_diag["raw"]["bce"], dtype=float)
    rel_bce = np.full_like(bce_den, np.nan, dtype=float)
    bce_mask = np.isfinite(bce_den) & np.isfinite(bce_num) & (bce_den > 0)
    rel_bce[bce_mask] = bce_num[bce_mask] / bce_den[bce_mask] - 1.0

    hard_den = np.asarray(uncon_diag["raw"]["hard_map_iou"], dtype=float)
    one_path_priv = np.asarray(priv_diag["raw"]["one_path_iou_mean"], dtype=float)
    rel_iou_vs_hard_map = np.full_like(hard_den, np.nan, dtype=float)
    hard_mask = np.isfinite(hard_den) & np.isfinite(one_path_priv) & (hard_den > 0)
    rel_iou_vs_hard_map[hard_mask] = 1.0 - one_path_priv[hard_mask] / hard_den[hard_mask]

    summaries = {
        "m_xi": {"scalar": float(m_xi)},
        "nsr": {"scalar": float(m_xi / (1.0 - m_xi))},
        "epsilon_unconstrained": {"scalar": float(uncon_row.get("epsilon", np.nan))},
        "deff_unconstrained": median_iqr_summary(uncon_raw["effective_dim"]),
        "deff_private": median_iqr_summary(priv_raw["effective_dim"]),
        "relative_bce_increase": median_iqr_summary(rel_bce),
        "analytic_iou_unconstrained": median_iqr_summary(uncon_raw["hard_map_iou"]),
        "one_path_iou_private": median_iqr_summary(priv_raw["one_path_iou_mean"]),
        "relative_iou_gap": median_iqr_summary(rel_iou_vs_hard_map),
        "one_path_threshold_c": {
            "scalar": (
                float(one_path_threshold_c)
                if one_path_threshold_c is not None
                else float(priv_diag["hyperparameters"]["path_threshold"])
            )
        },
        "one_path_threshold_tuned": bool(one_path_threshold_tuned),
    }

    epsilon_label = f"{epsilon_threshold:g}"
    rows = [
        (r"$M_\xi$", f"{m_xi:g}"),
        ("NSR", f"{m_xi / (1.0 - m_xi):.2f}"),
        (r"$\varepsilon$, unconstrained", _fmt_scalar(summaries["epsilon_unconstrained"]["scalar"], "epsilon")),
        (r"$d_{\rm eff}$, unconstrained", _fmt_median_iqr(summaries["deff_unconstrained"], "deff")),
        (rf"$d_{{\rm eff}}$, $\varepsilon<{epsilon_label}$", _fmt_median_iqr(summaries["deff_private"], "deff")),
        ("relative BCE increase", _fmt_median_iqr(summaries["relative_bce_increase"], "pct")),
        ("IoU, unconstrained", _fmt_median_iqr(summaries["analytic_iou_unconstrained"], "num")),
        (rf"$1$-path IoU mean, $\varepsilon<{epsilon_label}$", _fmt_median_iqr(summaries["one_path_iou_private"], "num")),
        ("relative IoU gap", _fmt_median_iqr(summaries["relative_iou_gap"], "pct")),
    ]
    table = pd.DataFrame(rows, columns=["metric", "value"])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        "$M_\\xi$ & $" + f"{m_xi:g}" + r"$ \\",
        r"\midrule",
    ]
    for metric, value in rows[1:]:
        lines.append(f"{metric} & \\({value}\\) \\\\")
    if one_path_threshold_tuned:
        path_clause = (
            "the private IoU is the per-pair mean over posterior draws after "
            f"thresholding each one-path release at the validation-selected "
            f"path-value cutoff \\(c={float(summaries['one_path_threshold_c']['scalar']):g}\\), "
            "while the true excursion set uses \\(t\\)."
        )
    else:
        path_clause = (
            "the private IoU is the per-pair mean over posterior draws after "
            "directly thresholding each one-path release at the excursion level \\(t\\)."
        )
    caption = (
        r"\caption{One column of the excursion-set utility table. Median "
        r"[interquartile range] is taken over the stored independent \((D,f_*)\)-pairs. "
        r"The unconstrained reference is the lowest-\(\varepsilon\) candidate among "
        r"the three lowest-average-BCE candidates. The private reference is the "
        r"highest one-path-IoU candidate among the three lowest-average-BCE candidates "
        r"satisfying the privacy constraint. The unconstrained IoU thresholds analytic "
        r"excursion probabilities at the cutoff selected on a separate validation sample; "
        + path_clause
        + "}"
    )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        caption,
        r"\label{tab:noise_transition_column}",
        r"\end{table}",
    ])
    latex = "\n".join(lines)

    return table, summaries, latex


def build_paper_pairwise_frame(
    pairs: list[dict],
    uncon_diag: dict,
    priv_diag: dict,
) -> pd.DataFrame:
    """Return the pair-level quantities from which the table column is formed."""
    bce_uncon = np.asarray(uncon_diag["raw"]["bce"], dtype=float)
    bce_priv = np.asarray(priv_diag["raw"]["bce"], dtype=float)
    iou_uncon = np.asarray(uncon_diag["raw"]["hard_map_iou"], dtype=float)
    iou_priv = np.asarray(priv_diag["raw"]["one_path_iou_mean"], dtype=float)
    n = len(pairs)
    for name, arr in [
        ("bce_unconstrained", bce_uncon),
        ("bce_private", bce_priv),
        ("analytic_iou_unconstrained", iou_uncon),
        ("one_path_iou_private", iou_priv),
    ]:
        if arr.size != n:
            raise ValueError(f"{name} has {arr.size} values for {n} pairs")

    relative_bce = np.full(n, np.nan, dtype=float)
    bce_mask = np.isfinite(bce_uncon) & np.isfinite(bce_priv) & (bce_uncon > 0)
    relative_bce[bce_mask] = bce_priv[bce_mask] / bce_uncon[bce_mask] - 1.0

    relative_iou = np.full(n, np.nan, dtype=float)
    iou_mask = np.isfinite(iou_uncon) & np.isfinite(iou_priv) & (iou_uncon > 0)
    relative_iou[iou_mask] = 1.0 - iou_priv[iou_mask] / iou_uncon[iou_mask]

    return pd.DataFrame({
        "pair_index": [int(p["pair_index"]) for p in pairs],
        "pair_seed": [int(p["seed"]) for p in pairs],
        "true_excursion_volume": [float(p["true_volume"]) for p in pairs],
        "bce_unconstrained": bce_uncon,
        "bce_private": bce_priv,
        "relative_bce_increase": relative_bce,
        "effective_dim_unconstrained": np.asarray(
            uncon_diag["raw"]["effective_dim"], dtype=float
        ),
        "effective_dim_private": np.asarray(
            priv_diag["raw"]["effective_dim"], dtype=float
        ),
        "analytic_iou_unconstrained": iou_uncon,
        "one_path_iou_private": iou_priv,
        "relative_iou_gap": relative_iou,
    })



def run_prior_average_grid_search(
    n_pairs: int,
    n_train: int,
    n_target_grid: int,
    ell_true: float,
    ell_values: list[float],
    r_values: list[float],
    sigma_values: list[float],
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
    M_Y: float = 1.0,
    private_refine_rounds: int = 3,
    target_reject: bool = False,
    target_reject_volume_min: float = 0.1,
    target_reject_volume_max: float = 0.9,
    target_reject_max_components: int | None = None,
    target_reject_min_component_width: float = 0.0,
    target_reject_min_mean_component_width: float = 0.0,
    target_reject_max_attempts: int = 10000,
    n_single_path_iou_draws: int = 1,
    single_path_iou_seed_offset: int = 100000,
    n_validation_pairs: int | None = None,
    validation_seed_offset: int = 10000000,
    tune_final_thresholds: bool = True,
    final_map_threshold_grid_size: int = 101,
    tune_one_path_threshold: bool = False,
    one_path_threshold_radius: float = 0.5,
    one_path_threshold_grid_size: int = 101,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[dict]]:
    x_target = np.linspace(0.0, 1.0, n_target_grid)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Sampling {n_pairs} independent (f_*, D) pairs "
        f"with n_train={n_train}, ell_true={ell_true:g}, M={m_eps:g}."
    )
    pairs, pair_summary = sample_prior_average_pairs(
        n_pairs=n_pairs,
        n_train=n_train,
        x_target=x_target,
        ell_true=ell_true,
        m_eps=m_eps,
        threshold=threshold,
        seed=seed,
        M_Y=M_Y,
        target_reject=target_reject,
        target_reject_volume_min=target_reject_volume_min,
        target_reject_volume_max=target_reject_volume_max,
        target_reject_max_components=target_reject_max_components,
        target_reject_min_component_width=target_reject_min_component_width,
        target_reject_min_mean_component_width=target_reject_min_mean_component_width,
        target_reject_max_attempts=target_reject_max_attempts,
    )
    print("Stored pair pool in RAM.")
    print(
        "Target pool summary: "
        f"volume median={pair_summary['true_excursion_volume']['median']:.4f}, "
        f"components median={pair_summary['true_excursion_components']['median']:.1f}, "
        f"mean width median={pair_summary['true_excursion_component_width_mean']['median']:.4f}."
    )

    coarse_distribution_candidates = make_distribution_candidates(ell_values, r_values, sigma_values)
    print(f"\n=== Coarse prior-average posterior-distribution BCE search: {len(coarse_distribution_candidates)} candidates ===")
    coarse_distribution_df = evaluate_distribution_candidates_prior_average(
        candidates=coarse_distribution_candidates,
        pairs=pairs,
        x_target=x_target,
        n_train=n_train,
        n_target_grid=n_target_grid,
        ell_true=ell_true,
        m_eps=m_eps,
        threshold=threshold,
        stage="coarse_prior_average_distribution_bce",
        bce_clip=bce_clip,
        hard_set_cutoff=hard_set_cutoff,
    )
    coarse_distribution_df = add_epsilon_column(
        coarse_distribution_df,
        n=n_train,
        delta=epsilon_delta,
        L=epsilon_L,
        M_Y=M_Y,
    )

    top_coarse_distribution = top_bce_rows(coarse_distribution_df, top_k)
    top_coarse_distribution_private = top_private_bce_rows(
        coarse_distribution_df,
        top_k=top_k,
        epsilon_threshold=epsilon_threshold,
    )

    round1_anchors = unique_distribution_rows(
        pd.concat([top_coarse_distribution, top_coarse_distribution_private], ignore_index=True)
    )
    refined_distribution_dfs: list[pd.DataFrame] = []
    refined_distribution_df = pd.DataFrame()

    if len(round1_anchors) > 0:
        refined_distribution_candidates = make_refined_distribution_candidates(round1_anchors)
        print(
            f"\n=== Refined prior-average BCE search round 1 "
            f"around {len(round1_anchors)} anchors: {len(refined_distribution_candidates)} candidates ==="
        )
        refined_round_df = evaluate_distribution_candidates_prior_average(
            candidates=refined_distribution_candidates,
            pairs=pairs,
            x_target=x_target,
            n_train=n_train,
            n_target_grid=n_target_grid,
            ell_true=ell_true,
            m_eps=m_eps,
            threshold=threshold,
            stage="refined_prior_average_distribution_bce_round1",
            bce_clip=bce_clip,
            hard_set_cutoff=hard_set_cutoff,
        )
        refined_round_df = add_epsilon_column(
            refined_round_df,
            n=n_train,
            delta=epsilon_delta,
            L=epsilon_L,
            M_Y=M_Y,
        )
        refined_distribution_dfs.append(refined_round_df)
        refined_distribution_df = pd.concat(refined_distribution_dfs, ignore_index=True)

    for refine_round in range(2, private_refine_rounds + 1):
        candidate_pool = refined_distribution_df if len(refined_distribution_df) else coarse_distribution_df
        private_anchors = top_private_bce_rows(candidate_pool, top_k=top_k, epsilon_threshold=epsilon_threshold)
        if len(private_anchors) == 0:
            print(
                f"No epsilon<{epsilon_threshold:g} candidates found for private refinement "
                f"round {refine_round}; stopping."
            )
            break
        private_refined_candidates = make_refined_distribution_candidates(private_anchors)
        print(
            f"\n=== Private refined prior-average BCE search round {refine_round} "
            f"around top {len(private_anchors)} epsilon<{epsilon_threshold:g} anchors: "
            f"{len(private_refined_candidates)} candidates ==="
        )
        private_round_df = evaluate_distribution_candidates_prior_average(
            candidates=private_refined_candidates,
            pairs=pairs,
            x_target=x_target,
            n_train=n_train,
            n_target_grid=n_target_grid,
            ell_true=ell_true,
            m_eps=m_eps,
            threshold=threshold,
            stage=f"private_refined_prior_average_distribution_bce_round{refine_round}",
            bce_clip=bce_clip,
            hard_set_cutoff=hard_set_cutoff,
        )
        private_round_df = add_epsilon_column(
            private_round_df,
            n=n_train,
            delta=epsilon_delta,
            L=epsilon_L,
            M_Y=M_Y,
        )
        refined_distribution_dfs.append(private_round_df)
        refined_distribution_df = pd.concat(refined_distribution_dfs, ignore_index=True)
        refined_distribution_df = unique_distribution_rows(refined_distribution_df)

    if len(refined_distribution_df) == 0:
        refined_distribution_df = coarse_distribution_df.copy()

    top3_distribution_df = top_bce_rows(refined_distribution_df, 3)
    top3_distribution_private_df = top_private_bce_rows(
        refined_distribution_df,
        top_k=3,
        epsilon_threshold=epsilon_threshold,
    )

    # Only the three epsilon-feasible BCE rows need sampled-path utility for
    # the requested private-row selection. The unconstrained reference is
    # selected by epsilon and reported through analytic probabilities.
    selected_for_single_path = unique_distribution_rows(
        top3_distribution_private_df.copy()
    )
    selected_for_single_path = estimate_single_path_iou_for_distribution_rows_pairs(
        rows=selected_for_single_path,
        pairs=pairs,
        x_target=x_target,
        threshold=threshold,
        n_draws_per_pair=(0 if tune_one_path_threshold else n_single_path_iou_draws),
        seed=seed + single_path_iou_seed_offset,
        n_release_paths=epsilon_L,
    )
    top3_distribution_df = _copy_single_path_estimates(top3_distribution_df, selected_for_single_path)
    top3_distribution_private_df = _copy_single_path_estimates(top3_distribution_private_df, selected_for_single_path)

    frontier_summary = distribution_frontier_summary(refined_distribution_df, epsilon_threshold)
    frontier_summary = add_prior_average_single_path_summary(
        frontier_summary,
        top3_distribution_df,
        top3_distribution_private_df,
    )

    main_text_uncon_row, main_text_priv_row = select_main_text_reference_rows(
        top3_distribution_df,
        top3_distribution_private_df,
    )
    main_text_uncon_diag = None
    main_text_priv_diag = None
    main_text_table_df = pd.DataFrame(columns=["metric", "value"])
    main_text_pairwise_df = pd.DataFrame()
    main_text_table_summaries: dict = {}
    main_text_latex_table = ""
    runtime_protocol_checks: dict[str, bool] = {}
    validation_pair_summary = None
    main_text_threshold_tuning: dict = {
        "enabled": bool(tune_final_thresholds or tune_one_path_threshold),
        "analytic_probability_cutoff_enabled": bool(tune_final_thresholds),
        "one_path_value_threshold_enabled": bool(tune_one_path_threshold),
        "release_L": int(epsilon_L),
        "validation_seed": None,
        "n_validation_pairs": 0,
        "unconstrained_reference": None,
        "private_candidates": [],
        "private_reference": None,
        "one_path_postprocessing": (
            "validation-selected path-value threshold c"
            if tune_one_path_threshold
            else "directly threshold each posterior path at t"
        ),
    }
    if main_text_uncon_row is not None and main_text_priv_row is not None:
        uncon_hard_map_cutoff = float(hard_set_cutoff)
        priv_hard_map_cutoff = float(hard_set_cutoff)
        priv_path_threshold = float(threshold)
        validation_pairs: list[dict] | None = None

        if tune_final_thresholds or tune_one_path_threshold:
            n_validation = n_pairs if n_validation_pairs is None else int(n_validation_pairs)
            if n_validation <= 0:
                raise ValueError("n_validation_pairs must be positive")
            validation_seed = int(seed + validation_seed_offset)
            print(
                f"\nSampling a separate validation set of {n_validation} "
                "independent (D,f_*) pairs for post-processing selection..."
            )
            validation_pairs, validation_pair_summary = sample_prior_average_pairs(
                n_pairs=n_validation,
                n_train=n_train,
                x_target=x_target,
                ell_true=ell_true,
                m_eps=m_eps,
                threshold=threshold,
                seed=validation_seed,
                M_Y=M_Y,
                target_reject=target_reject,
                target_reject_volume_min=target_reject_volume_min,
                target_reject_volume_max=target_reject_volume_max,
                target_reject_max_components=target_reject_max_components,
                target_reject_min_component_width=target_reject_min_component_width,
                target_reject_min_mean_component_width=target_reject_min_mean_component_width,
                target_reject_max_attempts=target_reject_max_attempts,
            )
            main_pair_seeds = {int(pair["seed"]) for pair in pairs}
            validation_pair_seeds = {
                int(pair["seed"]) for pair in validation_pairs
            }
            if main_pair_seeds & validation_pair_seeds:
                raise ValueError(
                    "validation_seed_offset creates overlap between main and "
                    "validation pair seeds; choose a different offset"
                )
            main_text_threshold_tuning["validation_seed"] = validation_seed
            main_text_threshold_tuning["n_validation_pairs"] = n_validation

        if tune_final_thresholds:
            if validation_pairs is None:
                raise RuntimeError("internal error: missing validation pairs")
            if final_map_threshold_grid_size < 2:
                raise ValueError("final_map_threshold_grid_size must be at least 2")
            map_threshold_grid = np.linspace(0.0, 1.0, final_map_threshold_grid_size)
            print("\nTuning the unconstrained analytic-probability cutoff C...")
            main_text_threshold_tuning["unconstrained_reference"] = tune_thresholds_for_selected_candidate(
                row=main_text_uncon_row,
                pairs=validation_pairs,
                x_target=x_target,
                threshold=threshold,
                hard_set_cutoff=hard_set_cutoff,
                n_draws_per_pair=0,
                seed=validation_seed + 300001,
                map_threshold_grid=map_threshold_grid,
                path_threshold_grid=np.empty(0, dtype=float),
                n_release_paths=1,
            )
            uncon_hard_map_cutoff = float(main_text_threshold_tuning["unconstrained_reference"]["hard_map_threshold"])
            print(f"  Selected C={uncon_hard_map_cutoff:.4g}.")

        if tune_one_path_threshold:
            if validation_pairs is None:
                raise RuntimeError("internal error: missing validation pairs")
            if one_path_threshold_grid_size < 2:
                raise ValueError("one_path_threshold_grid_size must be at least 2")
            if one_path_threshold_radius <= 0.0:
                raise ValueError("one_path_threshold_radius must be positive")
            path_value_grid = np.linspace(
                threshold - one_path_threshold_radius,
                threshold + one_path_threshold_radius,
                one_path_threshold_grid_size,
            )
            print(
                "\nTuning a sample-path value threshold c for each of the "
                "three epsilon-feasible BCE candidates..."
            )
            top3_distribution_private_df = top3_distribution_private_df.copy()
            top3_distribution_private_df["validation_one_path_threshold_c"] = np.nan
            top3_distribution_private_df["validation_one_path_iou_mean"] = np.nan
            top3_distribution_private_df["validation_path_threshold_at_boundary"] = False
            candidate_tunings: list[dict] = []
            main_pair_path_evaluations: list[pd.DataFrame] = []
            # Use common random numbers across candidates to reduce Monte Carlo
            # noise in the highest-IoU comparison.
            path_validation_seed = int(validation_seed + 500001)
            for idx, candidate_row in top3_distribution_private_df.iterrows():
                tuning = tune_one_path_value_threshold_for_candidate(
                    row=candidate_row,
                    pairs=validation_pairs,
                    x_target=x_target,
                    true_excursion_threshold=threshold,
                    n_draws_per_pair=n_single_path_iou_draws,
                    seed=path_validation_seed,
                    path_value_threshold_grid=path_value_grid,
                )
                candidate_tunings.append(tuning)
                top3_distribution_private_df.at[
                    idx, "validation_one_path_threshold_c"
                ] = float(tuning["path_value_threshold_c"])
                top3_distribution_private_df.at[
                    idx, "validation_one_path_iou_mean"
                ] = float(tuning["path_value_mean_iou"])
                top3_distribution_private_df.at[
                    idx, "validation_path_threshold_at_boundary"
                ] = bool(tuning["selected_at_grid_boundary"])
                main_eval = estimate_single_path_iou_for_distribution_rows_pairs(
                    rows=pd.DataFrame([candidate_row]),
                    pairs=pairs,
                    x_target=x_target,
                    threshold=float(tuning["path_value_threshold_c"]),
                    n_draws_per_pair=n_single_path_iou_draws,
                    seed=seed + single_path_iou_seed_offset,
                    n_release_paths=1,
                )
                main_pair_path_evaluations.append(main_eval)

            main_path_source = pd.concat(
                main_pair_path_evaluations, ignore_index=True
            )
            top3_distribution_private_df = _copy_single_path_estimates(
                top3_distribution_private_df, main_path_source
            )
            for tuning in candidate_tunings:
                mask = (
                    np.isclose(
                        top3_distribution_private_df["ell_model"].astype(float),
                        float(tuning["ell_model"]),
                    )
                    & np.isclose(
                        top3_distribution_private_df["r"].astype(float),
                        float(tuning["r"]),
                    )
                    & np.isclose(
                        top3_distribution_private_df["sigma"].astype(float),
                        float(tuning["sigma"]),
                    )
                )
                if np.any(mask):
                    tuning["search_pair_one_path_iou_mean"] = float(
                        top3_distribution_private_df.loc[
                            mask, "single_path_iou_mean"
                        ].iloc[0]
                    )

            main_text_threshold_tuning["private_candidates"] = candidate_tunings
            main_text_threshold_tuning["path_validation_seed"] = path_validation_seed
            main_text_uncon_row, main_text_priv_row = select_main_text_reference_rows(
                top3_distribution_df,
                top3_distribution_private_df,
                private_iou_column="single_path_iou_mean",
            )
            if main_text_priv_row is None:
                raise RuntimeError("no private reference remained after threshold tuning")
            priv_path_threshold = float(
                main_text_priv_row["validation_one_path_threshold_c"]
            )
            selected_private_tuning = next(
                result for result in candidate_tunings
                if np.isclose(result["ell_model"], float(main_text_priv_row["ell_model"]))
                and np.isclose(result["r"], float(main_text_priv_row["r"]))
                and np.isclose(result["sigma"], float(main_text_priv_row["sigma"]))
            )
            main_text_threshold_tuning["private_reference"] = selected_private_tuning
            print(
                "  Selected the private row by its sampled-path IoU on the main "
                f"search pairs, using its validation-tuned c={priv_path_threshold:.4g}; "
                f"main-pair mean IoU={float(main_text_priv_row['single_path_iou_mean']):.4f}, "
                f"validation mean IoU="
                f"{float(main_text_priv_row['validation_one_path_iou_mean']):.4f}."
            )
            if bool(selected_private_tuning["selected_at_grid_boundary"]):
                print(
                    "  WARNING: selected c lies on the search-grid boundary; "
                    "increase --one-path-threshold-radius and rerun."
                )
        else:
            main_text_threshold_tuning["private_reference"] = {
                "path_value_threshold_c": float(threshold),
                "scientific_threshold_t": float(threshold),
                "selection": "fixed_c_equals_t",
            }

        # Refresh these auxiliary frontier diagnostics after the optional
        # candidate-specific c evaluation on the main search pairs.
        frontier_summary = add_prior_average_single_path_summary(
            frontier_summary,
            top3_distribution_df,
            top3_distribution_private_df,
        )

        print("\nComputing pair-level median[IQR] diagnostics for the main-text table...")
        main_text_uncon_diag = evaluate_candidate_pairwise_for_main_table(
            row=main_text_uncon_row,
            pairs=pairs,
            x_target=x_target,
            threshold=threshold,
            bce_clip=bce_clip,
            hard_set_cutoff=hard_set_cutoff,
            n_draws_per_pair=0,
            seed=seed + single_path_iou_seed_offset + 700001,
            hard_map_cutoff_override=uncon_hard_map_cutoff,
            path_threshold_override=threshold,
            n_release_paths=epsilon_L,
        )
        main_text_priv_diag = evaluate_candidate_pairwise_for_main_table(
            row=main_text_priv_row,
            pairs=pairs,
            x_target=x_target,
            threshold=threshold,
            bce_clip=bce_clip,
            hard_set_cutoff=hard_set_cutoff,
            n_draws_per_pair=n_single_path_iou_draws,
            seed=seed + single_path_iou_seed_offset + 900001,
            hard_map_cutoff_override=priv_hard_map_cutoff,
            path_threshold_override=priv_path_threshold,
            n_release_paths=epsilon_L,
        )
        main_text_table_df, main_text_table_summaries, main_text_latex_table = build_main_text_median_iqr_table(
            main_text_uncon_row,
            main_text_priv_row,
            main_text_uncon_diag,
            main_text_priv_diag,
            m_xi=m_eps,
            epsilon_threshold=epsilon_threshold,
            one_path_threshold_c=priv_path_threshold,
            one_path_threshold_tuned=tune_one_path_threshold,
        )
        main_text_pairwise_df = build_paper_pairwise_frame(
            pairs=pairs,
            uncon_diag=main_text_uncon_diag,
            priv_diag=main_text_priv_diag,
        )

        expected_uncon = top3_distribution_df.sort_values(
            ["epsilon", "bce_post_mean", "ell_model", "r", "sigma"],
            ascending=[True, True, True, True, True],
        ).iloc[0]
        expected_private = top3_distribution_private_df.sort_values(
            ["single_path_iou_mean", "bce_post_mean", "ell_model", "r", "sigma"],
            ascending=[False, True, True, True, True],
        ).iloc[0]
        recomputed_feasible_top3 = top_private_bce_rows(
            refined_distribution_df,
            top_k=3,
            epsilon_threshold=epsilon_threshold,
        )
        bce_den_check = main_text_pairwise_df[
            "bce_unconstrained"
        ].to_numpy(dtype=float)
        bce_num_check = main_text_pairwise_df["bce_private"].to_numpy(dtype=float)
        finite_bce = (
            np.isfinite(bce_den_check)
            & np.isfinite(bce_num_check)
            & (bce_den_check > 0.0)
        )
        expected_relative_bce = np.full(len(main_text_pairwise_df), np.nan)
        expected_relative_bce[finite_bce] = (
            bce_num_check[finite_bce] / bce_den_check[finite_bce] - 1.0
        )
        iou_den_check = main_text_pairwise_df[
            "analytic_iou_unconstrained"
        ].to_numpy(dtype=float)
        iou_num_check = main_text_pairwise_df[
            "one_path_iou_private"
        ].to_numpy(dtype=float)
        finite_iou = (
            np.isfinite(iou_den_check)
            & np.isfinite(iou_num_check)
            & (iou_den_check > 0.0)
        )
        expected_relative_iou = np.full(len(main_text_pairwise_df), np.nan)
        expected_relative_iou[finite_iou] = (
            1.0 - iou_num_check[finite_iou] / iou_den_check[finite_iou]
        )
        runtime_protocol_checks = {
            "main_pairs_are_independent_by_seed": (
                len({int(pair["seed"]) for pair in pairs}) == len(pairs)
            ),
            "unconstrained_is_lowest_epsilon_among_top3_bce": (
                _candidate_signature(main_text_uncon_row)
                == _candidate_signature(expected_uncon)
            ),
            "private_pool_is_top3_bce_after_strict_epsilon_filter": bool(
                [
                    _candidate_signature(row)
                    for _, row in top3_distribution_private_df.iterrows()
                ]
                == [
                    _candidate_signature(row)
                    for _, row in recomputed_feasible_top3.iterrows()
                ]
                and np.all(
                    top3_distribution_private_df["epsilon"].to_numpy(dtype=float)
                    < epsilon_threshold
                )
            ),
            "private_is_highest_sampled_path_iou_in_feasible_top3": (
                _candidate_signature(main_text_priv_row)
                == _candidate_signature(expected_private)
            ),
            "reported_one_path_mean_uses_requested_draw_count": bool(
                int(main_text_priv_diag["hyperparameters"]["release_repetitions_per_pair"])
                == int(n_single_path_iou_draws)
            ),
            "relative_bce_is_pairwise_private_over_unconstrained_minus_one": bool(
                np.allclose(
                    main_text_pairwise_df.loc[
                        finite_bce, "relative_bce_increase"
                    ].to_numpy(dtype=float),
                    expected_relative_bce[finite_bce],
                )
            ),
            "relative_iou_is_pairwise_one_minus_private_path_over_unconstrained_analytic": bool(
                np.allclose(
                    main_text_pairwise_df.loc[
                        finite_iou, "relative_iou_gap"
                    ].to_numpy(dtype=float),
                    expected_relative_iou[finite_iou],
                )
            ),
            "analytic_C_uses_separate_validation_pairs": bool(
                (not tune_final_thresholds)
                or (
                    validation_pairs is not None
                    and not (
                        {int(pair["seed"]) for pair in pairs}
                        & {int(pair["seed"]) for pair in validation_pairs}
                    )
                )
            ),
            "one_path_c_rule_is_respected": bool(
                (
                    tune_one_path_threshold
                    and main_text_threshold_tuning["private_reference"] is not None
                    and np.isclose(
                        priv_path_threshold,
                        float(
                            main_text_threshold_tuning["private_reference"][
                                "path_value_threshold_c"
                            ]
                        ),
                    )
                )
                or (
                    not tune_one_path_threshold
                    and np.isclose(priv_path_threshold, threshold)
                )
            ),
        }
        failed_runtime_checks = [
            name for name, passed in runtime_protocol_checks.items() if not passed
        ]
        if failed_runtime_checks:
            raise RuntimeError(
                "paper-table protocol check failed: "
                + ", ".join(failed_runtime_checks)
            )
        print("\nRuntime paper-table checks: all passed")
        for name in runtime_protocol_checks:
            print(f"  OK: {name}")

        frontier_summary["main_text_release_L"] = int(epsilon_L)
        frontier_summary["main_text_unconstrained_reference_ell_model"] = float(main_text_uncon_row["ell_model"])
        frontier_summary["main_text_unconstrained_reference_r"] = float(main_text_uncon_row["r"])
        frontier_summary["main_text_unconstrained_reference_sigma"] = float(main_text_uncon_row["sigma"])
        frontier_summary["main_text_unconstrained_reference_epsilon"] = float(main_text_uncon_row["epsilon"])
        frontier_summary["main_text_one_path_postprocessing"] = (
            "validation_selected_path_value_threshold"
            if tune_one_path_threshold
            else "direct_path_threshold_at_t"
        )
        frontier_summary["main_text_path_threshold_c"] = float(priv_path_threshold)
        frontier_summary["main_text_true_excursion_threshold_t"] = float(threshold)
        frontier_summary["main_text_private_reference_ell_model"] = float(main_text_priv_row["ell_model"])
        frontier_summary["main_text_private_reference_r"] = float(main_text_priv_row["r"])
        frontier_summary["main_text_private_reference_sigma"] = float(main_text_priv_row["sigma"])
        frontier_summary["main_text_private_reference_epsilon"] = float(main_text_priv_row["epsilon"])

        table_csv_path = out_prefix.with_name(out_prefix.name + "_paper_table_column.csv")
        table_tex_path = out_prefix.with_name(out_prefix.name + "_paper_table_column.tex")
        pairwise_csv_path = out_prefix.with_name(out_prefix.name + "_paper_table_pairwise.csv")
        search_csv_path = out_prefix.with_name(out_prefix.name + "_search_candidates.csv")
        main_text_table_df.to_csv(table_csv_path, index=False)
        with open(table_tex_path, "w", encoding="utf-8") as f:
            f.write(main_text_latex_table + "\n")
        main_text_pairwise_df.to_csv(pairwise_csv_path, index=False)
        refined_distribution_df.to_csv(search_csv_path, index=False)
        print(f"Saved paper table column to: {table_csv_path}")
        print(f"Saved LaTeX table column to: {table_tex_path}")
        print(f"Saved pair-level table inputs to: {pairwise_csv_path}")
        print(f"Saved refined candidate summaries to: {search_csv_path}")

    best_path = out_prefix.with_name(out_prefix.name + "_best.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_type": "prior_average_over_independent_fstar_D_pairs",
                "distribution_objective": "expected_integrated_binary_cross_entropy_over_pairs",
                "selection_protocol": (
                    "unconstrained: lowest epsilon among the three lowest-average-BCE "
                    "rows; private: highest sampled-path IoU among the three lowest-"
                    "average-BCE rows satisfying epsilon below the budget"
                ),
                "pair_pool_summary": pair_summary,
                "validation_pair_pool_summary": validation_pair_summary,
                "target_sampling_method": "joint_ou_target_grid_and_training_points_per_pair",
                "target_training_values": "joint_ou_no_interpolation",
                "target_rejection_sampling": bool(target_reject),
                "target_rejection_volume_min": target_reject_volume_min if target_reject else None,
                "target_rejection_volume_max": target_reject_volume_max if target_reject else None,
                "target_rejection_max_components": target_reject_max_components if target_reject else None,
                "target_rejection_min_component_width": target_reject_min_component_width if target_reject else None,
                "target_rejection_min_mean_component_width": target_reject_min_mean_component_width if target_reject else None,
                "target_rejection_max_attempts": target_reject_max_attempts if target_reject else None,
                "epsilon_threshold": epsilon_threshold,
                "epsilon_delta": epsilon_delta,
                "epsilon_L": epsilon_L,
                "dp_model": "exp_1d",
                "dp_response_bound_M_Y": M_Y,
                "dp_accountant_version": _REQUIRED_DP_ACCOUNTANT_VERSION,
                "private_refine_rounds": private_refine_rounds,
                "n_single_path_iou_draws": n_single_path_iou_draws,
                "release_L": int(epsilon_L),
                "release_repetitions_per_pair": int(n_single_path_iou_draws),
                "single_path_iou_seed_offset": single_path_iou_seed_offset,
                "distribution_frontier_summary": frontier_summary,
                "main_text_selected_unconstrained_reference": (
                    None if main_text_uncon_row is None else main_text_uncon_row.to_dict()
                ),
                "main_text_selected_private_reference": (
                    None if main_text_priv_row is None else main_text_priv_row.to_dict()
                ),
                "main_text_median_iqr_table": main_text_table_df.to_dict(orient="records"),
                "main_text_median_iqr_summaries": main_text_table_summaries,
                "main_text_latex_table": main_text_latex_table,
                "main_text_threshold_tuning": main_text_threshold_tuning,
                "runtime_paper_table_checks": runtime_protocol_checks,
                "main_text_pairwise_diagnostics": {
                    "unconstrained_reference": None if main_text_uncon_diag is None else {
                        "hyperparameters": main_text_uncon_diag["hyperparameters"],
                        "summary": main_text_uncon_diag["summary"],
                    },
                    "private_reference": None if main_text_priv_diag is None else {
                        "hyperparameters": main_text_priv_diag["hyperparameters"],
                        "summary": main_text_priv_diag["summary"],
                    },
                },
                "top_coarse_distribution_by_bce": top_coarse_distribution.to_dict(orient="records"),
                "top_coarse_distribution_by_bce_eps_feasible": top_coarse_distribution_private.to_dict(orient="records"),
                "top3_refined_distribution_by_bce": top3_distribution_df.to_dict(orient="records"),
                "top3_refined_distribution_by_bce_eps_feasible": top3_distribution_private_df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )
    print(f"Saved prior-average best settings to: {best_path}")

    top3_npy_path = out_prefix.with_name(out_prefix.name + "_top3_choices.npy")
    np.save(
        top3_npy_path,
        {
            "experiment_type": "prior_average_over_independent_fstar_D_pairs",
            "epsilon_threshold": epsilon_threshold,
            "epsilon_delta": epsilon_delta,
            "epsilon_L": epsilon_L,
            "dp_model": "exp_1d",
            "dp_response_bound_M_Y": M_Y,
            "dp_accountant_version": _REQUIRED_DP_ACCOUNTANT_VERSION,
            "private_refine_rounds": private_refine_rounds,
            "n_single_path_iou_draws": n_single_path_iou_draws,
            "release_L": int(epsilon_L),
            "release_repetitions_per_pair": int(n_single_path_iou_draws),
            "single_path_iou_seed_offset": single_path_iou_seed_offset,
            "distribution_frontier_summary": frontier_summary,
            "main_text_median_iqr_table": main_text_table_df.to_records(index=False),
            "main_text_median_iqr_summaries": main_text_table_summaries,
            "main_text_latex_table": main_text_latex_table,
            "main_text_threshold_tuning": main_text_threshold_tuning,
            "runtime_paper_table_checks": runtime_protocol_checks,
            "top_coarse_distribution_by_bce": top_coarse_distribution.to_records(index=False),
            "top_coarse_distribution_by_bce_eps_feasible": top_coarse_distribution_private.to_records(index=False),
            "top3_refined_distribution_by_bce": top3_distribution_df.to_records(index=False),
            "top3_refined_distribution_by_bce_eps_feasible": top3_distribution_private_df.to_records(index=False),
        },
        allow_pickle=True,
    )
    print(f"Saved prior-average top-3 choices to: {top3_npy_path}")

    print("\nTop 3 refined prior-average posterior-distribution choices by BCE:")
    for i, row in enumerate(top3_distribution_df.to_dict(orient="records"), start=1):
        print(
            f"  {i}. ell={row['ell_model']:g}, r={row['r']:g}, "
            f"d_eff={row['effective_dim_mean']:.2f}, sigma={row['sigma']:g}, "
            f"BCE={row['bce_post_mean']:.4f}, eps={row['epsilon']:.3g}, "
            f"hard_IoU={row['posterior_hard_iou_mean']:.4f}, "
            f"release_IoU={row.get('release_iou_mean', row.get('single_path_iou_mean', float('nan'))):.4f}, "
            f"volume_MAE={row['volume_abs_err_mean']:.4f}"
        )

    print(f"\nTop 3 refined prior-average choices by BCE among epsilon<{epsilon_threshold:g}:")
    for i, row in enumerate(top3_distribution_private_df.to_dict(orient="records"), start=1):
        print(
            f"  {i}. ell={row['ell_model']:g}, r={row['r']:g}, "
            f"d_eff={row['effective_dim_mean']:.2f}, sigma={row['sigma']:g}, "
            f"BCE={row['bce_post_mean']:.4f}, eps={row['epsilon']:.3g}, "
            f"hard_IoU={row['posterior_hard_iou_mean']:.4f}, "
            f"release_IoU={row.get('release_iou_mean', row.get('single_path_iou_mean', float('nan'))):.4f}, "
            f"volume_MAE={row['volume_abs_err_mean']:.4f}"
        )

    print("\nPrior-average privacy-constrained frontier summary:")
    for key, value in frontier_summary.items():
        if isinstance(value, (int, float, np.floating)):
            print(f"  {key}: {value}")

    if main_text_uncon_row is not None and main_text_priv_row is not None:
        print("\nSelected rows for main-text median[IQR] table:")
        print(
            "  Unconstrained reference: "
            f"ell={float(main_text_uncon_row['ell_model']):g}, "
            f"r={float(main_text_uncon_row['r']):g}, "
            f"sigma={float(main_text_uncon_row['sigma']):g}, "
            f"eps={float(main_text_uncon_row['epsilon']):.4g}"
        )
        print(
            "  Private reference: "
            f"ell={float(main_text_priv_row['ell_model']):g}, "
            f"r={float(main_text_priv_row['r']):g}, "
            f"sigma={float(main_text_priv_row['sigma']):g}, "
            f"eps={float(main_text_priv_row['epsilon']):.4g}"
        )
        if main_text_threshold_tuning.get("enabled"):
            info = main_text_threshold_tuning.get("unconstrained_reference")
            print("\nValidation-selected post-processing cutoffs:")
            if info is not None:
                print(
                    f"  analytic unconstrained cutoff C={info['hard_map_threshold']:.6g} "
                    f"(C=0.5 validation IoU={info['hard_map_default_grid_mean_iou']:.4f}, "
                    f"selected-C validation IoU={info['hard_map_mean_iou']:.4f})"
                )
            private_threshold_info = main_text_threshold_tuning.get("private_reference")
            if main_text_threshold_tuning.get("one_path_value_threshold_enabled"):
                print(
                    f"  one-path path-value cutoff c="
                    f"{float(private_threshold_info['path_value_threshold_c']):.6g} "
                    f"(true-set threshold t={threshold:g}, validation mean IoU="
                    f"{float(private_threshold_info['path_value_mean_iou']):.4f})"
                )
            else:
                print(f"  one-path release: fixed path-value cutoff c=t={threshold:g}")

        print("\nMain-text median [IQR] table:")
        print(main_text_table_df.to_string(index=False))
        print("\nLaTeX table:")
        print(main_text_latex_table)

    return coarse_distribution_df, refined_distribution_df, x_target, pairs


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prior-average coarse-to-refined posterior-distribution BCE search "
            "over independent accepted (f_*,D) pairs. This script selects one "
            "global hyperparameter setting by average BCE, rather than running "
            "a separate per-target oracle grid search."
        )
    )
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-target-grid", type=int, default=800)
    parser.add_argument("--ell-true", type=float, default=1.0)
    parser.add_argument("--ell-model", type=str, nargs="+", default=["0.08,0.13,0.2,0.25,0.35,0.5,0.6,0.75,1"])
    parser.add_argument("--r-values", type=str, nargs="+", default=["0.05,0.1,0.2,0.5,1,2,5"])
    parser.add_argument("--sigma-values", type=str, nargs="+", default=["0,0.1,0.5,1,2,5"])
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=None,
        help="number of independent accepted (f_*,D) pairs to average over",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1000,
        help="backwards-compatible alias for --n-pairs when --n-pairs is not supplied",
    )
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
            "number of Monte Carlo repetitions per stored pair for estimating "
            "the final one-path release IoU. Defaults to --n-posterior-draws "
            "if supplied, else 50 as in the paper table"
        ),
    )
    parser.add_argument(
        "--single-path-iou-seed-offset",
        type=int,
        default=100000,
        help="seed offset for post-hoc one-path IoU Monte Carlo draws",
    )
    parser.add_argument("--m-eps", "--m-xi", dest="m_eps", type=float, default=0.5)
    parser.add_argument(
        "--M-Y",
        type=float,
        default=1.0,
        help="declared absolute response bound used by the tightened Exp-1D accountant",
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--top-k-coarse", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-prefix", type=str, default="exp_excursion_set_bce_iou_prioravg")
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
        help=(
            "number of released posterior paths; this table requires L=1. "
            "The 50 Monte Carlo repetitions are utility estimation only"
        ),
    )
    parser.add_argument(
        "--private-refine-rounds",
        type=int,
        default=3,
        help="number of distribution-refinement rounds for the epsilon-feasible BCE branch",
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
    parser.add_argument(
        "--tune-final-thresholds",
        dest="tune_final_thresholds",
        action="store_true",
        default=True,
        help=(
            "tune the analytic-probability cutoff C on a separate validation "
            "sample (enabled by default)"
        ),
    )
    parser.add_argument(
        "--no-tune-final-thresholds",
        dest="tune_final_thresholds",
        action="store_false",
        help="use --hard-set-cutoff directly instead of separate validation (not caption-exact)",
    )
    parser.add_argument(
        "--n-validation-pairs",
        type=int,
        default=None,
        help="size of the separate cutoff-validation sample; default is --n-pairs",
    )
    parser.add_argument(
        "--validation-seed-offset",
        type=int,
        default=10000000,
        help="seed offset for the independent cutoff-validation sample",
    )
    parser.add_argument(
        "--final-map-threshold-grid-size",
        type=int,
        default=101,
        help="number of grid points in [0,1] for tuning final hard-map probability cutoffs",
    )
    parser.add_argument(
        "--tune-one-path-threshold",
        action="store_true",
        help=(
            "on the separate validation sample, tune the path-value cutoff c "
            "in {x: f_D^(1)(x) >= c}; by default c equals --threshold"
        ),
    )
    parser.add_argument(
        "--one-path-threshold-radius",
        type=float,
        default=0.5,
        help=(
            "when --tune-one-path-threshold is used, search c over "
            "[t-radius,t+radius]"
        ),
    )
    parser.add_argument(
        "--one-path-threshold-grid-size",
        type=int,
        default=101,
        help="number of c values in the validation search grid",
    )
    args = parser.parse_args()

    n_pairs = args.n_pairs if args.n_pairs is not None else args.n_trials

    if args.n_single_path_iou_draws is None:
        n_single_path_iou_draws = args.n_posterior_draws if args.n_posterior_draws is not None else 50
        if args.n_posterior_draws is not None:
            print(
                "Note: using --n-posterior-draws as the number of Monte Carlo "
                "repetitions per stored pair for the selected L-path release IoU "
                "diagnostic; no finite-L grid-search metric is computed."
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
    print(f"Outputs will use prefix: {out_prefix}")

    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    if args.n_validation_pairs is not None and args.n_validation_pairs <= 0:
        raise ValueError("n_validation_pairs must be positive")
    if args.validation_seed_offset == 0:
        raise ValueError("validation_seed_offset must be nonzero so validation is separate")
    if not (0.0 < args.m_eps < 1.0):
        raise ValueError("m_xi/m_eps must lie in (0,1)")
    if args.M_Y <= 0:
        raise ValueError("M_Y must be positive")
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
    if args.epsilon_L != 1:
        raise ValueError(
            "This paper-table script reports a one-path release, so --epsilon-L "
            "must equal 1. The Monte Carlo value --n-posterior-draws does not "
            "compose privacy."
        )
    if args.private_refine_rounds < 1:
        raise ValueError("private_refine_rounds must be at least 1")
    if n_single_path_iou_draws <= 0:
        raise ValueError("n_single_path_iou_draws must be positive")
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
    if args.final_map_threshold_grid_size < 2:
        raise ValueError("final_map_threshold_grid_size must be at least 2")
    if args.one_path_threshold_grid_size < 2:
        raise ValueError("one_path_threshold_grid_size must be at least 2")
    if args.one_path_threshold_radius <= 0:
        raise ValueError("one_path_threshold_radius must be positive")
    _require_tight_dp_utils()

    caption_checks = {
        "1000 independent search/report pairs": n_pairs == 1000,
        "50 posterior draws per pair": n_single_path_iou_draws == 50,
        "training-set size 100": args.n_train == 100,
        "generator kernel exp(-|x-x'|)": np.isclose(args.ell_true, 1.0),
        "one released path": args.epsilon_L == 1,
        "epsilon budget 10": np.isclose(args.epsilon_threshold, 10.0),
        "delta 0.005": np.isclose(args.epsilon_delta, 0.005),
        "threshold t=0": np.isclose(args.threshold, 0.0),
        "separate analytic cutoff validation": bool(args.tune_final_thresholds),
        "unconditioned GP targets": not args.target_reject,
        "response bound M_Y=1": np.isclose(args.M_Y, 1.0),
    }
    print("\nPaper-protocol check:")
    for label, ok in caption_checks.items():
        print(f"  {'OK' if ok else 'DIFF'}: {label}")
    if not all(caption_checks.values()):
        print(
            "WARNING: rows marked DIFF depart from the supplied table protocol."
        )
    if args.target_reject:
        print(
            "WARNING: --target-reject conditions the law of f_* and must be "
            "disclosed in the paper; omit it for unconditioned GP draws."
        )
    print(
        "Selection convention: lowest-epsilon member of the three best average-"
        "BCE rows (unconstrained), and highest one-path-IoU member of the three "
        "best average-BCE rows with epsilon below the budget (private)."
    )
    if args.tune_one_path_threshold:
        print(
            "One-path post-processing: tune the sampled-function cutoff c on "
            "the separate validation pairs, then report on the untouched main pairs."
        )
    else:
        print("One-path post-processing: fixed c=t (default).")

    run_prior_average_grid_search(
        n_pairs=n_pairs,
        n_train=args.n_train,
        n_target_grid=args.n_target_grid,
        ell_true=args.ell_true,
        ell_values=ell_values,
        r_values=r_values,
        sigma_values=sigma_values,
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
        M_Y=args.M_Y,
        private_refine_rounds=args.private_refine_rounds,
        target_reject=args.target_reject,
        target_reject_volume_min=args.target_reject_volume_min,
        target_reject_volume_max=args.target_reject_volume_max,
        target_reject_max_components=args.target_reject_max_components,
        target_reject_min_component_width=args.target_reject_min_component_width,
        target_reject_min_mean_component_width=args.target_reject_min_mean_component_width,
        target_reject_max_attempts=args.target_reject_max_attempts,
        n_single_path_iou_draws=n_single_path_iou_draws,
        single_path_iou_seed_offset=args.single_path_iou_seed_offset,
        n_validation_pairs=args.n_validation_pairs,
        validation_seed_offset=args.validation_seed_offset,
        tune_final_thresholds=args.tune_final_thresholds,
        final_map_threshold_grid_size=args.final_map_threshold_grid_size,
        tune_one_path_threshold=args.tune_one_path_threshold,
        one_path_threshold_radius=args.one_path_threshold_radius,
        one_path_threshold_grid_size=args.one_path_threshold_grid_size,
    )


if __name__ == "__main__":
    main()
