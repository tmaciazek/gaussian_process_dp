import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr

from synthetic_pollution_utils import (
    sample_latent_pollution_fields,
    make_unit_square_grid,
)
from dp_utils import epsilon_for_delta_exp_kernel_unit_square


# ---------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------


def pairwise_distances(X, Z):
    """
    Euclidean pairwise distances between X and Z.
    """
    X = np.asarray(X)
    Z = np.asarray(Z)

    X_norm = np.sum(X**2, axis=1)[:, None]
    Z_norm = np.sum(Z**2, axis=1)[None, :]

    D2 = X_norm + Z_norm - 2.0 * X @ Z.T
    D2 = np.maximum(D2, 0.0)

    return np.sqrt(D2)


def exponential_kernel(X, Z, ell):
    """
    Exponential kernel on [0,1]^2:

        k(x,z) = exp(-||x-z|| / ell).
    """
    if ell <= 0:
        raise ValueError("ell must be positive.")

    return np.exp(-pairwise_distances(X, Z) / ell)


# ---------------------------------------------------------------------
# GP posterior and losses
# ---------------------------------------------------------------------


@dataclass
class GPGridSearchResult:
    ell: float
    r: float
    sigma: float
    mean_bce: float
    std_bce: float
    se_bce: float
    epsilon: Optional[float] = None


@dataclass
class WorldData:
    X_train: np.ndarray
    y_train: np.ndarray
    f_train: np.ndarray
    signal_train: np.ndarray
    noise: np.ndarray


def gp_posterior_mean_and_var_diag(
    X_train,
    y_train,
    X_eval,
    ell,
    r,
    jitter=1e-8,
):
    """
    Compute posterior mean mu_D and base posterior variance diagonal k_D(x,x).

    Convention:

        K_r = K(X,X) + r^2 I
        mu_D(x) = k_X(x)^T K_r^{-1} y
        k_D(x,x) = k(x,x) - k_X(x)^T K_r^{-1} k_X(x)

    The full posterior is GP(mu_D, sigma^2 k_D), so sigma is applied later.
    """
    n = X_train.shape[0]

    K_xx = exponential_kernel(X_train, X_train, ell)
    K_r = K_xx + (r**2 + jitter) * np.eye(n)

    try:
        c_factor = cho_factor(K_r, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return None, None

    alpha = cho_solve(c_factor, y_train, check_finite=False)

    K_eval_train = exponential_kernel(X_eval, X_train, ell)

    mu = K_eval_train @ alpha
    v = cho_solve(c_factor, K_eval_train.T, check_finite=False)

    # Exponential kernel has k(x,x)=1.
    var_diag = 1.0 - np.sum(K_eval_train * v.T, axis=1)
    var_diag = np.maximum(var_diag, 1e-12)

    return mu, var_diag


def gp_posterior_mean_and_cov(
    X_train,
    y_train,
    X_eval,
    ell,
    r,
    jitter=1e-8,
):
    """
    Compute posterior mean and full base posterior covariance on X_eval.

    This is used only for finite-dimensional one-path diagnostic sampling, so
    X_eval should be a relatively coarse grid.
    """
    n = X_train.shape[0]

    K_xx = exponential_kernel(X_train, X_train, ell)
    K_r = K_xx + (r**2 + jitter) * np.eye(n)

    c_factor = cho_factor(K_r, lower=True, check_finite=False)
    alpha = cho_solve(c_factor, y_train, check_finite=False)

    K_eval_train = exponential_kernel(X_eval, X_train, ell)
    K_eval_eval = exponential_kernel(X_eval, X_eval, ell)

    mu = K_eval_train @ alpha
    v = cho_solve(c_factor, K_eval_train.T, check_finite=False)
    cov = K_eval_eval - K_eval_train @ v

    cov = 0.5 * (cov + cov.T)
    cov[np.diag_indices_from(cov)] += jitter

    return mu, cov


def posterior_excursion_probability(mu, var_diag, sigma, threshold):
    """
    Compute p_D(x) = Phi((mu_D(x)-t)/(sigma sqrt(k_D(x,x)))).
    """
    std = sigma * np.sqrt(var_diag)
    z = (mu - threshold) / std
    return ndtr(z)


def binary_cross_entropy(p, s, eps=1e-8):
    """
    BCE between predicted probabilities p and binary labels s.

    p and s may be one-dimensional, or p may have shape (n_sigma, n_points).
    """
    p = np.clip(p, eps, 1.0 - eps)
    s = s.astype(float)

    return -np.mean(
        s * np.log(p) + (1.0 - s) * np.log(1.0 - p),
        axis=-1,
    )


def iou_score(pred, true):
    """
    Intersection-over-union for one binary mask.
    """
    pred = pred.astype(bool)
    true = true.astype(bool)

    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()

    if union == 0:
        return 1.0

    return intersection / union


def to_signed_latent(g):
    """Map a generated field g in [0,1] to a signed latent field in [-1,1]."""
    return 2.0 * np.asarray(g) - 1.0


def smooth_path_grids(path_grids, sigma_pixels, truncate=3.0):
    """
    Apply fixed Gaussian smoothing to released sample-path value grids.

    This is privacy-free post-processing. The smoothing is applied to the
    released path values before thresholding, not to the binary mask and not
    to the extracted contour curve.

    Parameters
    ----------
    path_grids : ndarray, shape (..., m, m)
        Sample-path values on a square grid.
    sigma_pixels : float
        Gaussian smoothing bandwidth in grid-pixel units. A value of zero
        leaves the paths unchanged.
    truncate : float
        Truncation radius in standard deviations for scipy.ndimage.gaussian_filter.
    """
    path_grids = np.asarray(path_grids)

    if sigma_pixels <= 0:
        return path_grids.copy()

    smoothed = np.empty_like(path_grids, dtype=float)

    flat_in = path_grids.reshape((-1,) + path_grids.shape[-2:])
    flat_out = smoothed.reshape((-1,) + smoothed.shape[-2:])

    for i, grid in enumerate(flat_in):
        flat_out[i] = gaussian_filter(
            grid,
            sigma=sigma_pixels,
            mode="nearest",
            truncate=truncate,
        )

    return smoothed




# ---------------------------------------------------------------------
# Population sampling
# ---------------------------------------------------------------------


def sample_one_dataset(
    fields,
    field_idx,
    n_data,
    M_xi,
    rng,
):
    """
    Sample random design locations and noisy attenuated responses:

        y_i = (1 - M_xi) f_*(x_i) + xi_i,

    where xi_i ~ Uniform[-M_xi, M_xi].
    """
    X_train = rng.uniform(0.0, 1.0, size=(n_data, 2))

    g_train = fields(X_train, field_idx=field_idx)
    f_train = to_signed_latent(g_train)
    signal_train = (1.0 - M_xi) * f_train

    noise = rng.uniform(-M_xi, M_xi, size=n_data)
    y_train = signal_train + noise

    return WorldData(
        X_train=X_train,
        y_train=y_train,
        f_train=f_train,
        signal_train=signal_train,
        noise=noise,
    )


def sample_population_worlds(
    n_reps,
    n_data,
    M_xi,
    latent_seed,
    data_seed,
    reference_grid_size,
    latent_K=2,
    min_width=0.20,
    max_width=0.35,
    anisotropy_max=1.4,
    background_strength=0.0,
    nonlinearity="sigmoid",
    sigmoid_gamma=5.0,
    sigmoid_center_quantile=0.65,
):
    """
    Sample n_reps independent latent fields and datasets.

    By default this uses the sigmoid-warped broad-plume latent model:

        h(x) = smooth plume score,
        f_*(x) = sigmoid(gamma * (h(x) - q)),

    followed by the usual range normalization in synthetic_pollution_utils.py.
    """
    K_arg = None if latent_K is None or int(latent_K) <= 0 else int(latent_K)

    fields = sample_latent_pollution_fields(
        n_fields=n_reps,
        seed=latent_seed,
        K=K_arg,
        min_width=min_width,
        max_width=max_width,
        anisotropy_max=anisotropy_max,
        background_strength=background_strength,
        nonlinearity=nonlinearity,
        sigmoid_gamma=sigmoid_gamma,
        sigmoid_center_quantile=sigmoid_center_quantile,
        normalize_range=True,
        reference_grid_size=reference_grid_size,
    )

    rng = np.random.default_rng(data_seed)

    worlds = [
        sample_one_dataset(
            fields=fields,
            field_idx=j,
            n_data=n_data,
            M_xi=M_xi,
            rng=rng,
        )
        for j in range(n_reps)
    ]

    return fields, worlds


# ---------------------------------------------------------------------
# DP helpers
# ---------------------------------------------------------------------


def dp_epsilon_for_params(n, ell, r, sigma, M_Y, L, delta, alpha_grid_size):
    """
    Epsilon for one parameter triple using the alpha-optimized bound in
    dp_utils.py for the 2D unit-square exponential kernel.
    """
    out = epsilon_for_delta_exp_kernel_unit_square(
        n=n,
        ell=ell,
        r=r,
        sigma=sigma,
        M_Y=M_Y,
        L=L,
        delta=delta,
        grid_size=alpha_grid_size,
    )
    return float(out["Epsilon"])


def make_candidate_grid(ell_grid, r_grid, sigma_grid):
    return [
        (float(ell), float(r), float(sigma))
        for ell in ell_grid
        for r in r_grid
        for sigma in sigma_grid
    ]


def make_refined_candidates(
    top_results,
    ell_bounds=(0.025, 1.0),
    r_bounds=(0.25, 10.00),
    sigma_bounds=(0.02, 1.20),
    factors=(0.75, 0.85, 0.925, 1.0, 1.075, 1.15, 1.25),
    decimals=8,
):
    """
    Build one local multiplicative refinement grid around each of the top
    coarse private candidates.
    """
    candidates = set()

    for res in top_results:
        ell_vals = np.clip(np.array(factors) * res.ell, *ell_bounds)
        r_vals = np.clip(np.array(factors) * res.r, *r_bounds)
        sigma_vals = np.clip(np.array(factors) * res.sigma, *sigma_bounds)

        ell_vals = np.unique(np.round(ell_vals, decimals))
        r_vals = np.unique(np.round(r_vals, decimals))
        sigma_vals = np.unique(np.round(sigma_vals, decimals))

        for ell in ell_vals:
            for r in r_vals:
                for sigma in sigma_vals:
                    candidates.add((float(ell), float(r), float(sigma)))

    return sorted(candidates)


# ---------------------------------------------------------------------
# Population objectives
# ---------------------------------------------------------------------


def evaluate_population_candidates(
    worlds,
    X_eval,
    s_true_all,
    threshold,
    candidates,
    n_data,
    M_Y,
    L,
    delta,
    epsilon0=None,
    alpha_grid_size=81,
    jitter=1e-8,
    verbose=True,
    label="grid",
):
    """
    Evaluate candidate triples by mean BCE over sampled worlds. If epsilon0
    is not None, candidates with epsilon > epsilon0 are skipped.
    """
    n_reps = len(worlds)

    # Group sigmas by (ell, r) so the posterior mean/variance is computed once.
    grouped = {}
    for ell, r, sigma in candidates:
        grouped.setdefault((float(ell), float(r)), []).append(float(sigma))

    best = None
    all_results = []

    n_total = len(grouped)

    for counter, ((ell, r), sigmas) in enumerate(grouped.items(), start=1):
        sigmas = np.array(sorted(set(sigmas)), dtype=float)
        n_sigma = len(sigmas)

        epsilons = np.array([
            dp_epsilon_for_params(
                n=n_data,
                ell=ell,
                r=r,
                sigma=sigma,
                M_Y=M_Y,
                L=L,
                delta=delta,
                alpha_grid_size=alpha_grid_size,
            )
            for sigma in sigmas
        ])

        feasible = np.ones(n_sigma, dtype=bool)
        if epsilon0 is not None:
            feasible = epsilons < epsilon0

        if not np.any(feasible):
            if verbose:
                print(
                    f"[{label} {counter:>3}/{n_total}] "
                    f"ell={ell:.4g}, r={r:.4g}: no epsilon-feasible sigmas"
                )
            continue

        bce_by_sigma_rep = np.full((n_sigma, n_reps), np.nan)

        for j, world in enumerate(worlds):
            mu, var_diag = gp_posterior_mean_and_var_diag(
                X_train=world.X_train,
                y_train=world.y_train,
                X_eval=X_eval,
                ell=ell,
                r=r,
                jitter=jitter,
            )

            if mu is None:
                continue

            denom = sigmas[:, None] * np.sqrt(var_diag)[None, :]
            z = (mu[None, :] - threshold) / denom
            p = ndtr(z)

            bce_by_sigma_rep[:, j] = binary_cross_entropy(
                p,
                s_true_all[j],
            )

        for a, sigma in enumerate(sigmas):
            if not feasible[a]:
                continue

            bces = bce_by_sigma_rep[a]
            bces = bces[np.isfinite(bces)]

            if len(bces) == 0:
                continue

            mean_bce = float(np.mean(bces))
            std_bce = float(np.std(bces, ddof=1)) if len(bces) > 1 else 0.0
            se_bce = float(std_bce / np.sqrt(len(bces)))

            result = GPGridSearchResult(
                ell=float(ell),
                r=float(r),
                sigma=float(sigma),
                mean_bce=mean_bce,
                std_bce=std_bce,
                se_bce=se_bce,
                epsilon=float(epsilons[a]),
            )

            all_results.append(result)

            if best is None or result.mean_bce < best.mean_bce:
                best = result

        if verbose:
            if best is None:
                best_msg = "no feasible candidate yet"
            else:
                best_msg = (
                    f"best=(ell={best.ell:.4g}, r={best.r:.4g}, "
                    f"sigma={best.sigma:.4g}, eps={best.epsilon:.4g}), "
                    f"mean BCE={best.mean_bce:.6f}"
                )
            print(
                f"[{label} {counter:>3}/{n_total}] "
                f"ell={ell:.4g}, r={r:.4g}, {best_msg}"
            )

    if best is None:
        raise RuntimeError(
            "No valid parameter setting found. For private search, try "
            "increasing epsilon0, increasing sigma/r grid, or changing M_Y."
        )

    all_results = sorted(all_results, key=lambda z: z.mean_bce)
    return best, all_results


def run_population_gridsearch(
    worlds,
    X_eval,
    s_true_all,
    threshold,
    ell_grid,
    r_grid,
    sigma_grid,
    n_data,
    M_Y,
    L,
    delta,
    alpha_grid_size=81,
    verbose=True,
):
    """
    Unconstrained population-BCE grid search over a rectangular grid.
    """
    candidates = make_candidate_grid(ell_grid, r_grid, sigma_grid)
    return evaluate_population_candidates(
        worlds=worlds,
        X_eval=X_eval,
        s_true_all=s_true_all,
        threshold=threshold,
        candidates=candidates,
        n_data=n_data,
        M_Y=M_Y,
        L=L,
        delta=delta,
        epsilon0=None,
        alpha_grid_size=alpha_grid_size,
        verbose=verbose,
        label="public",
    )


def run_private_coarse_to_refined_gridsearch(
    worlds,
    X_eval,
    s_true_all,
    threshold,
    n_data,
    M_Y,
    L,
    delta,
    epsilon0,
    alpha_grid_size=81,
    verbose=True,
):
    """
    Private grid search:

    1. coarse grid over the user-specified private grid;
    2. take top-3 feasible candidates by population BCE;
    3. one multiplicative local refinement round around those top candidates.
    """
    coarse_ell_grid = np.array([
        0.05, 0.10, 0.14, 0.18, 0.24, 0.35, 0.45
    ])

    coarse_r_grid = np.array([
        2.50, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0
    ])

    coarse_sigma_grid = np.array([
        0.01, 0.05, 0.08, 0.12, 0.18, 0.27, 0.40
    ])

    coarse_candidates = make_candidate_grid(
        coarse_ell_grid,
        coarse_r_grid,
        coarse_sigma_grid,
    )

    print("Running private coarse grid search...")
    coarse_best, coarse_results = evaluate_population_candidates(
        worlds=worlds,
        X_eval=X_eval,
        s_true_all=s_true_all,
        threshold=threshold,
        candidates=coarse_candidates,
        n_data=n_data,
        M_Y=M_Y,
        L=L,
        delta=delta,
        epsilon0=epsilon0,
        alpha_grid_size=alpha_grid_size,
        verbose=verbose,
        label="private coarse",
    )

    top3 = coarse_results[:3]
    refined_candidates = make_refined_candidates(top3)

    print("Running private refined grid search around top-3 coarse candidates...")
    refined_best, refined_results = evaluate_population_candidates(
        worlds=worlds,
        X_eval=X_eval,
        s_true_all=s_true_all,
        threshold=threshold,
        candidates=refined_candidates,
        n_data=n_data,
        M_Y=M_Y,
        L=L,
        delta=delta,
        epsilon0=epsilon0,
        alpha_grid_size=alpha_grid_size,
        verbose=verbose,
        label="private refined",
    )

    return {
        "coarse_best": coarse_best,
        "coarse_results": coarse_results,
        "refined_best": refined_best,
        "refined_results": refined_results,
        "top3_coarse": top3,
    }


def compute_population_probabilities(
    worlds,
    X_eval,
    ell,
    r,
    sigma,
    threshold,
    jitter=1e-8,
):
    """
    Compute p_{D_j}(x) for each sampled world j using fixed hyperparameters.
    """
    p_all = []
    mu_all = []
    var_all = []

    for world in worlds:
        mu, var_diag = gp_posterior_mean_and_var_diag(
            X_train=world.X_train,
            y_train=world.y_train,
            X_eval=X_eval,
            ell=ell,
            r=r,
            jitter=jitter,
        )

        if mu is None:
            raise RuntimeError("Cholesky failed while recomputing best model.")

        p = posterior_excursion_probability(
            mu=mu,
            var_diag=var_diag,
            sigma=sigma,
            threshold=threshold,
        )

        p_all.append(p)
        mu_all.append(mu)
        var_all.append(var_diag)

    return np.array(p_all), np.array(mu_all), np.array(var_all)


def choose_population_probability_threshold_by_iou(
    p_all,
    s_true_all,
    C_grid=None,
):
    """
    Choose one global C maximizing mean IoU over sampled worlds.
    """
    if C_grid is None:
        C_grid = np.linspace(0.001, 0.999, 999)

    mean_ious = []
    std_ious = []

    s_true_all = s_true_all.astype(bool)

    for C in C_grid:
        pred_all = p_all >= C

        intersections = np.logical_and(pred_all, s_true_all).sum(axis=1)
        unions = np.logical_or(pred_all, s_true_all).sum(axis=1)

        ious = np.ones_like(unions, dtype=float)
        nonempty = unions > 0
        ious[nonempty] = intersections[nonempty] / unions[nonempty]

        mean_ious.append(np.mean(ious))
        std_ious.append(np.std(ious, ddof=1) if len(ious) > 1 else 0.0)

    mean_ious = np.array(mean_ious)
    std_ious = np.array(std_ious)

    best_idx = int(np.argmax(mean_ious))

    return {
        "C": float(C_grid[best_idx]),
        "mean_iou": float(mean_ious[best_idx]),
        "std_iou": float(std_ious[best_idx]),
        "se_iou": float(std_ious[best_idx] / np.sqrt(p_all.shape[0])),
        "C_grid": C_grid,
        "mean_ious": mean_ious,
        "std_ious": std_ious,
    }


def choose_population_score_threshold_by_iou(
    score_all,
    s_true_all,
    threshold_grid=None,
):
    """
    Choose one global threshold c maximizing mean IoU over sampled worlds:

        mean_j IoU({score_j(x) >= c}, Omega_j).

    This is used for one-path releases, where the released object is a
    sampled function value rather than a probability map.
    """
    score_all = np.asarray(score_all)
    s_true_all = np.asarray(s_true_all).astype(bool)

    if threshold_grid is None:
        lo = float(np.quantile(score_all, 0.001))
        hi = float(np.quantile(score_all, 0.999))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(score_all))
            hi = float(np.max(score_all))
        if hi <= lo:
            threshold_grid = np.array([lo])
        else:
            threshold_grid = np.linspace(lo, hi, 999)
    else:
        threshold_grid = np.asarray(threshold_grid, dtype=float)

    mean_ious = []
    std_ious = []

    for c in threshold_grid:
        pred_all = score_all >= c

        intersections = np.logical_and(pred_all, s_true_all).sum(axis=1)
        unions = np.logical_or(pred_all, s_true_all).sum(axis=1)

        ious = np.ones_like(unions, dtype=float)
        nonempty = unions > 0
        ious[nonempty] = intersections[nonempty] / unions[nonempty]

        mean_ious.append(np.mean(ious))
        std_ious.append(np.std(ious, ddof=1) if len(ious) > 1 else 0.0)

    mean_ious = np.array(mean_ious)
    std_ious = np.array(std_ious)

    best_idx = int(np.argmax(mean_ious))

    return {
        "threshold": float(threshold_grid[best_idx]),
        "mean_iou": float(mean_ious[best_idx]),
        "std_iou": float(std_ious[best_idx]),
        "se_iou": float(std_ious[best_idx] / np.sqrt(score_all.shape[0])),
        "threshold_grid": threshold_grid,
        "mean_ious": mean_ious,
        "std_ious": std_ious,
    }


def compute_one_path_iou_summary(
    fields,
    worlds,
    M_xi,
    threshold,
    ell,
    r,
    sigma,
    path_grid_size,
    seed,
    path_threshold_grid_size=999,
    path_smoothing_sigma=2.0,
    path_smoothing_truncate=3.0,
    jitter=1e-8,
):
    """
    Report private one-sample-path IoU over (f_*,D) pairs.

    This draws one exact finite-dimensional posterior GP sample on a separate
    coarse path grid for each world. Each released path is then smoothed by a
    fixed Gaussian filter before thresholding. The one-path IoU reported here
    is therefore the IoU of the smoothed post-processed released path.

    The threshold is selected globally by maximizing mean IoU over worlds
    using the smoothed released path values. The natural threshold t is also
    reported for comparison.
    """
    rng = np.random.default_rng(seed)

    _, _, _, _, X_path_grid = make_unit_square_grid(path_grid_size)
    X_path = X_path_grid.reshape(-1, 2)

    G_path_all = fields(X_path_grid)
    F_path_all = to_signed_latent(G_path_all)
    signal_path_all = (1.0 - M_xi) * F_path_all
    s_true_all = (signal_path_all >= threshold).reshape(len(worlds), -1)

    release_values = []

    for world in worlds:
        mu, cov = gp_posterior_mean_and_cov(
            X_train=world.X_train,
            y_train=world.y_train,
            X_eval=X_path,
            ell=ell,
            r=r,
            jitter=jitter,
        )

        try:
            L_chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Slightly stronger numerical diagonal correction.
            cov = cov.copy()
            cov[np.diag_indices_from(cov)] += 1e-6
            L_chol = np.linalg.cholesky(cov)

        z = rng.standard_normal(X_path.shape[0])
        f_release = mu + sigma * (L_chol @ z)
        release_values.append(f_release)

    release_values_raw = np.array(release_values)
    release_grids_raw = release_values_raw.reshape(
        len(worlds),
        path_grid_size,
        path_grid_size,
    )
    release_grids = smooth_path_grids(
        release_grids_raw,
        sigma_pixels=path_smoothing_sigma,
        truncate=path_smoothing_truncate,
    )
    release_values = release_grids.reshape(len(worlds), -1)

    # Natural threshold t, for comparison, applied after smoothing.
    natural_summary = choose_population_score_threshold_by_iou(
        score_all=release_values,
        s_true_all=s_true_all,
        threshold_grid=np.array([threshold]),
    )

    # IoU-selected global path-value threshold. Include t explicitly in the
    # search grid, so the selected threshold is never worse than the natural
    # threshold on this Monte Carlo sample.
    lo = float(np.quantile(release_values, 0.001))
    hi = float(np.quantile(release_values, 0.999))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(release_values))
        hi = float(np.max(release_values))

    if hi <= lo:
        path_threshold_grid = np.array([threshold])
    else:
        path_threshold_grid = np.linspace(lo, hi, path_threshold_grid_size)
        path_threshold_grid = np.unique(
            np.concatenate([path_threshold_grid, np.array([threshold])])
        )

    selected_summary = choose_population_score_threshold_by_iou(
        score_all=release_values,
        s_true_all=s_true_all,
        threshold_grid=path_threshold_grid,
    )

    selected_threshold = selected_summary["threshold"]
    released_sets = release_values >= selected_threshold

    ious = []
    volumes = []
    for j in range(len(worlds)):
        ious.append(iou_score(released_sets[j], s_true_all[j]))
        volumes.append(float(np.mean(released_sets[j])))

    ious = np.array(ious)
    volumes = np.array(volumes)

    return {
        "path_grid_size": int(path_grid_size),
        "path_smoothing_sigma": float(path_smoothing_sigma),
        "path_smoothing_truncate": float(path_smoothing_truncate),
        "selected_path_threshold": float(selected_threshold),
        "natural_path_threshold": float(threshold),
        "mean_iou": float(np.mean(ious)),
        "std_iou": float(np.std(ious, ddof=1)) if len(ious) > 1 else 0.0,
        "se_iou": float(np.std(ious, ddof=1) / np.sqrt(len(ious))) if len(ious) > 1 else 0.0,
        "mean_released_volume": float(np.mean(volumes)),
        "natural_threshold_mean_iou": natural_summary["mean_iou"],
        "natural_threshold_std_iou": natural_summary["std_iou"],
        "natural_threshold_se_iou": natural_summary["se_iou"],
        "per_rep_iou": ious.tolist(),
        "per_rep_released_volume": volumes.tolist(),
    }


def sample_private_path_excursion_sets_for_world(
    fields,
    field_idx,
    world,
    M_xi,
    threshold,
    ell,
    r,
    sigma,
    path_threshold,
    path_grid_size,
    n_paths,
    seed,
    path_smoothing_sigma=2.0,
    path_smoothing_truncate=3.0,
    jitter=1e-8,
):
    """
    Draw several finite-dimensional private posterior sample paths for one
    world and return their thresholded excursion-set masks on a coarse grid.

    These are diagnostic visualizations of the actual one-path released object:

        f_D^{(l)} ~ GP(mu_D, sigma^2 k_D),
        released set = {x : f_D^{(l)}(x) >= path_threshold}.

    Before thresholding, each sampled path is smoothed by the same fixed
    Gaussian filter used in compute_one_path_iou_summary. The threshold is
    typically the global path threshold selected by mean IoU on smoothed paths;
    if this equals the natural threshold t, the plotted paths are simply
    {S_h f_D^{(l)} >= t}.
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    rng = np.random.default_rng(seed)

    _, _, X1_path, X2_path, X_path_grid = make_unit_square_grid(path_grid_size)
    X_path = X_path_grid.reshape(-1, 2)

    G_path = fields(X_path_grid, field_idx=field_idx)
    F_path = to_signed_latent(G_path)
    signal_path = (1.0 - M_xi) * F_path
    s_true_path_grid = signal_path >= threshold
    s_true_path = s_true_path_grid.reshape(-1)

    mu, cov = gp_posterior_mean_and_cov(
        X_train=world.X_train,
        y_train=world.y_train,
        X_eval=X_path,
        ell=ell,
        r=r,
        jitter=jitter,
    )

    try:
        L_chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov.copy()
        cov[np.diag_indices_from(cov)] += 1e-6
        L_chol = np.linalg.cholesky(cov)

    z = rng.standard_normal((X_path.shape[0], n_paths))
    release_values = mu[:, None] + sigma * (L_chol @ z)
    release_values = release_values.T

    release_grids_raw = release_values.reshape(n_paths, path_grid_size, path_grid_size)
    release_grids = smooth_path_grids(
        release_grids_raw,
        sigma_pixels=path_smoothing_sigma,
        truncate=path_smoothing_truncate,
    )
    release_values = release_grids.reshape(n_paths, -1)

    released_sets = release_values >= path_threshold
    released_grids = released_sets.reshape(n_paths, path_grid_size, path_grid_size)

    path_ious = np.array([
        iou_score(released_sets[j], s_true_path)
        for j in range(n_paths)
    ])

    return X1_path, X2_path, released_grids, s_true_path_grid, path_ious


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def _maybe_contour(ax, X1, X2, Z, level, **kwargs):
    """
    Draw a contour only if the level lies strictly inside the plotted range.
    """
    z_min = np.nanmin(Z)
    z_max = np.nanmax(Z)

    if z_min < level < z_max:
        ax.contour(X1, X2, Z, levels=[level], **kwargs)


def plot_population_diagnostic(
    X1,
    X2,
    F_star,
    signal_grid,
    s_true_grid,
    p_grid,
    decision_grid,
    X_train,
    y_train,
    threshold,
    latent_threshold,
    best,
    M_xi,
    prob_threshold,
    iou_value,
    rep_idx,
    title_prefix="Population-best",
    output_path=None,
    path_X1=None,
    path_X2=None,
    path_decision_grids=None,
    path_threshold=None,
    path_iou_values=None,
):
    """
    Plot one representative sampled world.

    If path_decision_grids is provided, the bottom-left panel shows the true
    excursion set with several smoothed private one-path released-set boundaries
    overlaid. Otherwise it falls back to the probability-set boundary.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.0))

    ax = axes[0, 0]
    im = ax.contourf(X1, X2, F_star, levels=40, vmin=-1.0, vmax=1.0)
    _maybe_contour(
        ax,
        X1,
        X2,
        F_star,
        level=latent_threshold,
        colors="black",
        linewidths=2.0,
    )
    fig.colorbar(im, ax=ax)
    ax.set_title(r"Signed latent field $f_*=2g_*-1$")
    ax.set_aspect("equal")

    ax = axes[0, 1]
    im = ax.contourf(X1, X2, signal_grid, levels=40)
    _maybe_contour(ax, X1, X2, signal_grid, level=threshold, colors="black", linewidths=2.0)
    fig.colorbar(im, ax=ax)
    ax.set_title(r"Noiseless attenuated signal $(1-M_\xi)f_*$")
    ax.set_aspect("equal")

    ax = axes[0, 2]
    sc = ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        s=35,
        edgecolors="black",
        linewidths=0.4,
    )
    fig.colorbar(sc, ax=ax)
    ax.set_title(r"Observed data $y_i$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")

    # ------------------------------------------------------------
    # True set with either smoothed private one-path boundaries or probability boundary
    # ------------------------------------------------------------

    ax = axes[1, 0]
    im = ax.contourf(X1, X2, s_true_grid.astype(float), levels=[-0.1, 0.5, 1.1])
    _maybe_contour(
        ax,
        X1,
        X2,
        s_true_grid.astype(float),
        level=0.5,
        colors="black",
        linewidths=2.0,
    )

    if path_decision_grids is not None:
        if path_X1 is None or path_X2 is None:
            raise ValueError("path_X1 and path_X2 must be provided with path_decision_grids.")

        path_colors = plt.cm.tab10.colors

        for j, path_grid in enumerate(path_decision_grids):
            _maybe_contour(
                ax,
                path_X1,
                path_X2,
                path_grid.astype(float),
                level=0.5,
                colors=[path_colors[j % len(path_colors)]],
                linewidths=1.4,
                linestyles="--",
                alpha=0.70,
            )

        legend_handles = [
            Line2D([0], [0], color="black", lw=2.0, label="True boundary"),
            Line2D(
                [0],
                [0],
                color="red",
                lw=1.5,
                linestyle="--",
                label="Smoothed private 1-path boundaries",
            ),
        ]
        ax.legend(handles=legend_handles, loc="upper right")

        if path_iou_values is None:
            path_iou_text = ""
        else:
            path_iou_text = f", mean path IoU={np.mean(path_iou_values):.3f}"

        if path_threshold is None:
            ax.set_title("True set with smoothed private 1-path boundaries" + path_iou_text)
        else:
            ax.set_title(
                f"True set with smoothed private 1-path boundaries, c={path_threshold:.3g}" +
                path_iou_text
            )
    else:
        _maybe_contour(
            ax,
            X1,
            X2,
            decision_grid.astype(float),
            level=0.5,
            colors="red",
            linewidths=2.0,
            linestyles="--",
        )
        legend_handles = [
            Line2D([0], [0], color="black", lw=2.0, label="True boundary"),
            Line2D(
                [0],
                [0],
                color="red",
                lw=2.0,
                linestyle="--",
                label="Probability-set boundary",
            ),
        ]
        ax.legend(handles=legend_handles, loc="upper right")
        ax.set_title("True set with probability-set boundary")

    fig.colorbar(im, ax=ax)
    ax.set_aspect("equal")

    ax = axes[1, 1]
    im = ax.contourf(X1, X2, p_grid, levels=40, vmin=0.0, vmax=1.0)
    _maybe_contour(
        ax,
        X1,
        X2,
        p_grid,
        level=prob_threshold,
        colors="red",
        linewidths=2.0,
        linestyles="--",
    )
    fig.colorbar(im, ax=ax)
    ax.set_title(r"Posterior excursion probability $p_D(x)$")
    ax.set_aspect("equal")

    ax = axes[1, 2]
    im = ax.contourf(X1, X2, decision_grid.astype(float), levels=[-0.1, 0.5, 1.1])
    _maybe_contour(
        ax,
        X1,
        X2,
        decision_grid.astype(float),
        level=0.5,
        colors="red",
        linewidths=2.0,
        linestyles="--",
    )
    _maybe_contour(
        ax,
        X1,
        X2,
        s_true_grid.astype(float),
        level=0.5,
        colors="black",
        linewidths=2.0,
    )
    fig.colorbar(im, ax=ax)
    legend_handles = [
        Line2D([0], [0], color="black", lw=2.0, label="True boundary"),
        Line2D([0], [0], color="red", lw=2.0, linestyle="--", label="Probability-set boundary"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_title(f"Probability set with true boundary, C={prob_threshold:.3f}")
    ax.set_aspect("equal")

    for ax in axes.ravel():
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

    eps_part = "" if best.epsilon is None else rf", $\epsilon={best.epsilon:.3g}$"
    fig.suptitle(
        rf"{title_prefix} diagnostic rep {rep_idx}, "
        rf"$M_\xi={M_xi:.3g}$, $q={latent_threshold:.3g}$, $t={threshold:.3g}$, "
        rf"$\ell={best.ell:.3g}$, $r={best.r:.3g}$, "
        rf"$\sigma={best.sigma:.3g}$, "
        rf"mean BCE={best.mean_bce:.4f}, "
        rf"$C={prob_threshold:.3f}$, prob-set IoU={iou_value:.3f}" + eps_part,
        fontsize=14,
    )

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    plt.show()

def plot_C_curve(C_grid, mean_ious, C_star, output_path=None, title="Population selection of probability threshold"):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(C_grid, mean_ious)
    ax.axvline(C_star, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Probability threshold C")
    ax.set_ylabel("Mean IoU over sampled worlds")
    ax.set_title(title)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved C-threshold curve to {output_path}")

    plt.show()


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------


def result_to_jsonable(result):
    return asdict(result) if result is not None else None


def summary_to_jsonable(C_summary):
    return {
        "C": C_summary["C"],
        "mean_iou": C_summary["mean_iou"],
        "std_iou": C_summary["std_iou"],
        "se_iou": C_summary["se_iou"],
    }


def save_results_json(
    path,
    args,
    public_best,
    public_results,
    public_C_summary,
    public_iou_at_C_half,
    private_payload,
):
    payload = {
        "args": vars(args),
        "public": {
            "best_hyperparameters": result_to_jsonable(public_best),
            "population_threshold": summary_to_jsonable(public_C_summary),
            "mean_iou_at_C_0_5": public_iou_at_C_half,
            "gridsearch_results": [asdict(r) for r in public_results],
        },
        "private": private_payload,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved results to {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-reps", type=int, default=30)
    parser.add_argument("--n-data", type=int, default=400)
    parser.add_argument("--M-xi", type=float, default=0.5)

    parser.add_argument("--latent-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=1)
    parser.add_argument("--path-seed", type=int, default=123)

    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--reference-grid-size", type=int, default=180)
    parser.add_argument("--path-grid-size", type=int, default=60)
    parser.add_argument("--path-threshold-grid-size", type=int, default=999)
    parser.add_argument(
        "--n-path-plot-lines",
        type=int,
        default=5,
        help="Number of private one-path released-set boundaries to overlay in private diagnostic plots.",
    )
    parser.add_argument(
        "--path-plot-seed",
        type=int,
        default=456,
        help="Random seed for the smoothed private one-path boundaries shown in diagnostic plots.",
    )

    parser.add_argument(
        "--path-smoothing-sigma",
        type=float,
        default=2.0,
        help=(
            "Gaussian smoothing sigma for private sample-path values, "
            "in path-grid pixels. Use 0 for no smoothing."
        ),
    )
    parser.add_argument(
        "--path-smoothing-truncate",
        type=float,
        default=3.0,
        help="Gaussian smoothing truncate parameter for scipy.ndimage.gaussian_filter.",
    )

    parser.add_argument("--threshold-fraction", type=float, default=0.5, help="Threshold q on the original [0,1] generated field g_*. The GP uses f_*=2g_*-1, so the signed latent threshold is 2q-1.")

    parser.add_argument("--latent-K", type=int, default=2)
    parser.add_argument("--min-width", type=float, default=0.20)
    parser.add_argument("--max-width", type=float, default=0.35)
    parser.add_argument("--anisotropy-max", type=float, default=1.4)
    parser.add_argument("--background-strength", type=float, default=0.0)
    parser.add_argument(
        "--nonlinearity",
        choices=["identity", "sigmoid"],
        default="sigmoid",
        help="Latent-field transform. The default uses sigmoid-warped broad plumes.",
    )
    parser.add_argument("--sigmoid-gamma", type=float, default=5.0)
    parser.add_argument("--sigmoid-center-quantile", type=float, default=0.65)

    parser.add_argument("--epsilon0", type=float, default=10.0)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--M-Y", type=float, default=1.0)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--alpha-grid-size", type=int, default=81)

    parser.add_argument(
        "--plot-rep",
        type=int,
        default=0,
        help="Index of the first representative population world to plot.",
    )

    parser.add_argument(
        "--n-plot-reps",
        type=int,
        default=10,
        help="Number of representative population worlds to plot.",
    )

    parser.add_argument("--output-dir", type=str, default="excursion_2d_population_results")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-public", action="store_true")

    args = parser.parse_args()

    if args.n_reps <= 0:
        raise ValueError("--n-reps must be positive.")
    if args.n_data <= 1:
        raise ValueError("--n-data must be at least 2 for the DP bound.")
    if not (0.0 <= args.M_xi <= 1.0):
        raise ValueError("--M-xi must satisfy 0 <= M_xi <= 1.")
    if args.M_xi == 1.0:
        print("Warning: M_xi=1 is degenerate because the signal vanishes and t=0.")
    if not (0.0 < args.threshold_fraction < 1.0):
        raise ValueError("--threshold-fraction should lie in (0,1).")
    if args.latent_K < 0:
        raise ValueError("--latent-K must be nonnegative; use 0 for random K.")
    if args.min_width <= 0 or args.max_width <= 0 or args.max_width < args.min_width:
        raise ValueError("Require 0 < --min-width <= --max-width.")
    if args.anisotropy_max < 1.0:
        raise ValueError("--anisotropy-max must be at least 1.")
    if args.sigmoid_gamma <= 0:
        raise ValueError("--sigmoid-gamma must be positive.")
    if not (0.0 < args.sigmoid_center_quantile < 1.0):
        raise ValueError("--sigmoid-center-quantile must lie in (0,1).")
    if args.path_threshold_grid_size < 2:
        raise ValueError("--path-threshold-grid-size must be at least 2.")
    if args.n_path_plot_lines <= 0:
        raise ValueError("--n-path-plot-lines must be positive.")
    if args.path_smoothing_sigma < 0:
        raise ValueError("--path-smoothing-sigma must be nonnegative.")
    if args.path_smoothing_truncate <= 0:
        raise ValueError("--path-smoothing-truncate must be positive.")
    if not (0 <= args.plot_rep < args.n_reps):
        raise ValueError("--plot-rep must satisfy 0 <= plot_rep < n_reps.")
    if args.n_plot_reps <= 0:
        raise ValueError("--n-plot-reps must be positive.")
    if args.plot_rep + args.n_plot_reps > args.n_reps:
        raise ValueError("--plot-rep + --n-plot-reps must be at most --n-reps.")
    if args.path_grid_size <= 1:
        raise ValueError("--path-grid-size must be at least 2.")
    if args.epsilon0 <= 0:
        raise ValueError("--epsilon0 must be positive.")
    if args.M_Y < 0:
        raise ValueError("--M-Y must be nonnegative.")
    if args.L <= 0:
        raise ValueError("--L must be positive.")

    delta = args.delta if args.delta is not None else args.n_data ** (-1.1)
    if not (0.0 < delta < 1.0):
        raise ValueError("--delta must satisfy 0 < delta < 1.")

    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Sample population of worlds.
    # ------------------------------------------------------------

    print("Sampling latent fields and datasets...")
    fields, worlds = sample_population_worlds(
        n_reps=args.n_reps,
        n_data=args.n_data,
        M_xi=args.M_xi,
        latent_seed=args.latent_seed,
        data_seed=args.data_seed,
        reference_grid_size=args.reference_grid_size,
        latent_K=args.latent_K,
        min_width=args.min_width,
        max_width=args.max_width,
        anisotropy_max=args.anisotropy_max,
        background_strength=args.background_strength,
        nonlinearity=args.nonlinearity,
        sigmoid_gamma=args.sigmoid_gamma,
        sigmoid_center_quantile=args.sigmoid_center_quantile,
    )

    # ------------------------------------------------------------
    # 2. Evaluation grid and true excursion sets.
    # ------------------------------------------------------------

    _, _, X1, X2, X_grid = make_unit_square_grid(args.grid_size)
    X_eval = X_grid.reshape(-1, 2)

    # The generator returns g_*(x) in [0,1].  For the GP experiment we
    # work with the signed anomaly field f_*(x)=2g_*(x)-1 in [-1,1].
    G_all = fields(X_grid)
    F_all = to_signed_latent(G_all)
    signal_all = (1.0 - args.M_xi) * F_all

    # args.threshold_fraction is the original [0,1] threshold q for g_*.
    # On the signed scale this is q_signed=2q-1, and the signal threshold is
    # t=(1-M_xi) q_signed.
    latent_threshold = 2.0 * args.threshold_fraction - 1.0
    threshold = (1.0 - args.M_xi) * latent_threshold

    s_true_all_grid = signal_all >= threshold
    s_true_all = s_true_all_grid.reshape(args.n_reps, -1)

    excursion_volumes = s_true_all.mean(axis=1)
    excursion_std = float(excursion_volumes.std(ddof=1)) if args.n_reps > 1 else 0.0

    print()
    print("Population setup:")
    print(f"  n_reps = {args.n_reps}")
    print(f"  n_data = {args.n_data}")
    print(f"  M_xi = {args.M_xi}")
    print(f"  original [0,1] threshold q = {args.threshold_fraction:.6f}")
    print(f"  signed latent threshold 2q-1 = {latent_threshold:.6f}")
    print(f"  signal threshold t = {threshold:.6f}")
    print(f"  mean excursion volume = {excursion_volumes.mean():.4f}")
    print(f"  std excursion volume = {excursion_std:.4f}")
    print()
    print("Latent-field setup:")
    print(f"  nonlinearity = {args.nonlinearity}")
    print(f"  K = {'random' if args.latent_K == 0 else args.latent_K}")
    print(f"  width range = [{args.min_width}, {args.max_width}]")
    print(f"  anisotropy_max = {args.anisotropy_max}")
    print(f"  background_strength = {args.background_strength}")
    if args.nonlinearity == "sigmoid":
        print(f"  sigmoid_gamma = {args.sigmoid_gamma}")
        print(f"  sigmoid_center_quantile = {args.sigmoid_center_quantile}")
    print()
    print("DP setup:")
    print(f"  epsilon0 = {args.epsilon0}")
    print(f"  delta = {delta:.6g}")
    print(f"  M_Y = {args.M_Y}")
    print(f"  L = {args.L}")
    print()

    # ------------------------------------------------------------
    # 3. Public/unconstrained population BCE grid search.
    # ------------------------------------------------------------

    public_best = None
    public_results = []
    public_C_summary = None
    public_C_half_summary = None
    p_public_all = None

    public_ell_grid = np.array([
        0.10, 0.14, 0.18, 0.24, 0.35, 0.45, 0.5
    ])

    public_r_grid = np.array([
       0.27, 0.40, 0.60, 0.90, 1.30, 1.50, 1.70, 2.0, 2.5
    ])

    public_sigma_grid = np.array([
        0.05, 0.08, 0.12, 0.18, 0.27, 0.40, 0.60
    ])

    if not args.skip_public:
        print("Running public/unconstrained population BCE grid search...")
        public_best, public_results = run_population_gridsearch(
            worlds=worlds,
            X_eval=X_eval,
            s_true_all=s_true_all,
            threshold=threshold,
            ell_grid=public_ell_grid,
            r_grid=public_r_grid,
            sigma_grid=public_sigma_grid,
            n_data=args.n_data,
            M_Y=args.M_Y,
            L=args.L,
            delta=delta,
            alpha_grid_size=args.alpha_grid_size,
            verbose=not args.no_verbose,
        )

        print()
        print("Public population-best hyperparameters:")
        print(f"  ell      = {public_best.ell}")
        print(f"  r        = {public_best.r}")
        print(f"  sigma    = {public_best.sigma}")
        print(f"  epsilon  = {public_best.epsilon:.6f}")
        print(f"  mean BCE = {public_best.mean_bce:.6f}")
        print(f"  std BCE  = {public_best.std_bce:.6f}")
        print(f"  SE BCE   = {public_best.se_bce:.6f}")
        print()

        p_public_all, _, _ = compute_population_probabilities(
            worlds=worlds,
            X_eval=X_eval,
            ell=public_best.ell,
            r=public_best.r,
            sigma=public_best.sigma,
            threshold=threshold,
        )

        public_C_summary = choose_population_probability_threshold_by_iou(
            p_all=p_public_all,
            s_true_all=s_true_all,
            C_grid=np.linspace(0.001, 0.999, 999),
        )
        public_C_half_summary = choose_population_probability_threshold_by_iou(
            p_all=p_public_all,
            s_true_all=s_true_all,
            C_grid=np.array([0.5]),
        )

        print("Public population-selected probability threshold:")
        print(f"  C        = {public_C_summary['C']:.6f}")
        print(f"  mean IoU = {public_C_summary['mean_iou']:.6f}")
        print(f"  mean IoU at C=0.5 = {public_C_half_summary['mean_iou']:.6f}")
        print()

    # ------------------------------------------------------------
    # 4. Private coarse-to-refined grid search.
    # ------------------------------------------------------------

    private_search = run_private_coarse_to_refined_gridsearch(
        worlds=worlds,
        X_eval=X_eval,
        s_true_all=s_true_all,
        threshold=threshold,
        n_data=args.n_data,
        M_Y=args.M_Y,
        L=args.L,
        delta=delta,
        epsilon0=args.epsilon0,
        alpha_grid_size=args.alpha_grid_size,
        verbose=not args.no_verbose,
    )

    private_best = private_search["refined_best"]

    print()
    print("Private refined population-best hyperparameters:")
    print(f"  ell      = {private_best.ell}")
    print(f"  r        = {private_best.r}")
    print(f"  sigma    = {private_best.sigma}")
    print(f"  epsilon  = {private_best.epsilon:.6f}")
    print(f"  mean BCE = {private_best.mean_bce:.6f}")
    print(f"  std BCE  = {private_best.std_bce:.6f}")
    print(f"  SE BCE   = {private_best.se_bce:.6f}")
    print()

    p_private_all, _, _ = compute_population_probabilities(
        worlds=worlds,
        X_eval=X_eval,
        ell=private_best.ell,
        r=private_best.r,
        sigma=private_best.sigma,
        threshold=threshold,
    )

    private_C_summary = choose_population_probability_threshold_by_iou(
        p_all=p_private_all,
        s_true_all=s_true_all,
        C_grid=np.linspace(0.001, 0.999, 999),
    )
    private_C_half_summary = choose_population_probability_threshold_by_iou(
        p_all=p_private_all,
        s_true_all=s_true_all,
        C_grid=np.array([0.5]),
    )

    print("Private population-selected probability threshold:")
    print(f"  C        = {private_C_summary['C']:.6f}")
    print(f"  mean IoU = {private_C_summary['mean_iou']:.6f}")
    print(f"  std IoU  = {private_C_summary['std_iou']:.6f}")
    print(f"  SE IoU   = {private_C_summary['se_iou']:.6f}")
    print(f"  mean IoU at C=0.5 = {private_C_half_summary['mean_iou']:.6f}")
    print()

    print("Sampling one private posterior path per world for released-set IoU...")
    private_one_path = compute_one_path_iou_summary(
        fields=fields,
        worlds=worlds,
        M_xi=args.M_xi,
        threshold=threshold,
        ell=private_best.ell,
        r=private_best.r,
        sigma=private_best.sigma,
        path_grid_size=args.path_grid_size,
        seed=args.path_seed,
        path_threshold_grid_size=args.path_threshold_grid_size,
        path_smoothing_sigma=args.path_smoothing_sigma,
        path_smoothing_truncate=args.path_smoothing_truncate,
    )

    print("Private smoothed 1-path released-set IoU:")
    print(f"  path grid size = {private_one_path['path_grid_size']}")
    print(f"  path smoothing sigma = {private_one_path['path_smoothing_sigma']:.6f} pixels")
    print(f"  path smoothing truncate = {private_one_path['path_smoothing_truncate']:.6f}")
    print(f"  selected path threshold = {private_one_path['selected_path_threshold']:.6f}")
    print(f"  natural path threshold = {private_one_path['natural_path_threshold']:.6f}")
    print(f"  mean IoU = {private_one_path['mean_iou']:.6f}")
    print(f"  std IoU  = {private_one_path['std_iou']:.6f}")
    print(f"  SE IoU   = {private_one_path['se_iou']:.6f}")
    print(f"  mean IoU at natural threshold = {private_one_path['natural_threshold_mean_iou']:.6f}")
    print(f"  mean released volume = {private_one_path['mean_released_volume']:.6f}")
    print()

    # ------------------------------------------------------------
    # 5. Save summary.
    # ------------------------------------------------------------

    if not args.no_save:
        stem = (
            f"pop_nreps{args.n_reps}_n{args.n_data}"
            f"_Mxi{args.M_xi:g}_eps{args.epsilon0:g}"
            f"_grid{args.grid_size}_lseed{args.latent_seed}_dseed{args.data_seed}"
        )

        json_path = os.path.join(args.output_dir, f"{stem}_summary.json")
        private_C_curve_path = os.path.join(args.output_dir, f"{stem}_private_C_curve.png")
        public_C_curve_path = os.path.join(args.output_dir, f"{stem}_public_C_curve.png")
    else:
        stem = None
        json_path = None
        private_C_curve_path = None
        public_C_curve_path = None

    private_payload = {
        "epsilon0": args.epsilon0,
        "delta": delta,
        "M_Y": args.M_Y,
        "L": args.L,
        "coarse_best": result_to_jsonable(private_search["coarse_best"]),
        "refined_best": result_to_jsonable(private_search["refined_best"]),
        "top3_coarse": [asdict(r) for r in private_search["top3_coarse"]],
        "coarse_results": [asdict(r) for r in private_search["coarse_results"]],
        "refined_results": [asdict(r) for r in private_search["refined_results"]],
        "population_threshold": summary_to_jsonable(private_C_summary),
        "mean_iou_at_C_0_5": private_C_half_summary["mean_iou"],
        "one_path_released_iou": private_one_path,
    }

    if json_path is not None:
        save_results_json(
            path=json_path,
            args=args,
            public_best=public_best,
            public_results=public_results,
            public_C_summary=public_C_summary if public_C_summary is not None else {
                "C": None, "mean_iou": None, "std_iou": None, "se_iou": None
            },
            public_iou_at_C_half=(
                None if public_C_half_summary is None else public_C_half_summary["mean_iou"]
            ),
            private_payload=private_payload,
        )

    # ------------------------------------------------------------
    # 6. Plots.
    # ------------------------------------------------------------

    if not args.no_plots:
        C_private = private_C_summary["C"]
        C_public = None if public_C_summary is None else public_C_summary["C"]

        for rep_idx in range(args.plot_rep, args.plot_rep + args.n_plot_reps):
            # Public/unconstrained probability-set diagnostic.
            if p_public_all is not None and C_public is not None and public_best is not None:
                p_public_grid = p_public_all[rep_idx].reshape(args.grid_size, args.grid_size)
                public_decision_grid = p_public_grid >= C_public

                public_rep_iou = iou_score(
                    public_decision_grid.reshape(-1),
                    s_true_all[rep_idx],
                )

                print(
                    f"Plotting public diagnostic rep {rep_idx}: "
                    f"IoU={public_rep_iou:.4f}, "
                    f"true volume={s_true_all[rep_idx].mean():.4f}, "
                    f"pred volume={public_decision_grid.mean():.4f}"
                )

                if not args.no_save:
                    public_fig_path_rep = os.path.join(
                        args.output_dir,
                        f"{stem}_public_diagnostic_rep{rep_idx}.png",
                    )
                else:
                    public_fig_path_rep = None

                plot_population_diagnostic(
                    X1=X1,
                    X2=X2,
                    F_star=F_all[rep_idx],
                    signal_grid=signal_all[rep_idx],
                    s_true_grid=s_true_all_grid[rep_idx],
                    p_grid=p_public_grid,
                    decision_grid=public_decision_grid,
                    X_train=worlds[rep_idx].X_train,
                    y_train=worlds[rep_idx].y_train,
                    threshold=threshold,
                    latent_threshold=latent_threshold,
                    best=public_best,
                    M_xi=args.M_xi,
                    prob_threshold=C_public,
                    iou_value=public_rep_iou,
                    rep_idx=rep_idx,
                    title_prefix="Public/unconstrained population-best",
                    output_path=public_fig_path_rep,
                )

            # Private probability-set diagnostic.
            p_private_grid = p_private_all[rep_idx].reshape(args.grid_size, args.grid_size)
            private_decision_grid = p_private_grid >= C_private

            private_rep_iou = iou_score(
                private_decision_grid.reshape(-1),
                s_true_all[rep_idx],
            )

            print(
                f"Plotting private diagnostic rep {rep_idx}: "
                f"IoU={private_rep_iou:.4f}, "
                f"true volume={s_true_all[rep_idx].mean():.4f}, "
                f"pred volume={private_decision_grid.mean():.4f}"
            )

            if not args.no_save:
                private_fig_path_rep = os.path.join(
                    args.output_dir,
                    f"{stem}_private_diagnostic_rep{rep_idx}.png",
                )
            else:
                private_fig_path_rep = None

            path_threshold_for_plot = private_one_path["selected_path_threshold"]
            (
                path_X1_plot,
                path_X2_plot,
                private_path_decision_grids,
                _,
                private_path_iou_values,
            ) = sample_private_path_excursion_sets_for_world(
                fields=fields,
                field_idx=rep_idx,
                world=worlds[rep_idx],
                M_xi=args.M_xi,
                threshold=threshold,
                ell=private_best.ell,
                r=private_best.r,
                sigma=private_best.sigma,
                path_threshold=path_threshold_for_plot,
                path_grid_size=args.path_grid_size,
                n_paths=args.n_path_plot_lines,
                seed=args.path_plot_seed + rep_idx,
                path_smoothing_sigma=args.path_smoothing_sigma,
                path_smoothing_truncate=args.path_smoothing_truncate,
            )

            print(
                f"  Smoothed private 1-path plotted boundaries: "
                f"n={args.n_path_plot_lines}, "
                f"path threshold={path_threshold_for_plot:.4f}, "
                f"mean IoU={private_path_iou_values.mean():.4f}, "
                f"std IoU={private_path_iou_values.std(ddof=1) if len(private_path_iou_values) > 1 else 0.0:.4f}"
            )

            plot_population_diagnostic(
                X1=X1,
                X2=X2,
                F_star=F_all[rep_idx],
                signal_grid=signal_all[rep_idx],
                s_true_grid=s_true_all_grid[rep_idx],
                p_grid=p_private_grid,
                decision_grid=private_decision_grid,
                X_train=worlds[rep_idx].X_train,
                y_train=worlds[rep_idx].y_train,
                threshold=threshold,
                latent_threshold=latent_threshold,
                best=private_best,
                M_xi=args.M_xi,
                prob_threshold=C_private,
                iou_value=private_rep_iou,
                rep_idx=rep_idx,
                title_prefix="Private population-best",
                output_path=private_fig_path_rep,
                path_X1=path_X1_plot,
                path_X2=path_X2_plot,
                path_decision_grids=private_path_decision_grids,
                path_threshold=path_threshold_for_plot,
                path_iou_values=private_path_iou_values,
            )

        plot_C_curve(
            C_grid=private_C_summary["C_grid"],
            mean_ious=private_C_summary["mean_ious"],
            C_star=private_C_summary["C"],
            output_path=private_C_curve_path,
            title="Private population selection of probability threshold",
        )

        if public_C_summary is not None:
            plot_C_curve(
                C_grid=public_C_summary["C_grid"],
                mean_ious=public_C_summary["mean_ious"],
                C_star=public_C_summary["C"],
                output_path=public_C_curve_path,
                title="Public population selection of probability threshold",
            )


if __name__ == "__main__":
    main()
