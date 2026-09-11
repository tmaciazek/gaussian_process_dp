#!/usr/bin/env python3
"""
Summarise fixed public/private GP hyperparameters over many synthetic 2D
excursion-set worlds.

This script does not run a grid search. It assumes that the best public and
private hyperparameters are already known, then draws many independent
(f_*, D)-pairs and prints a plain terminal table with median [IQR] statistics.

The private 1-path metrics are computed from smoothed posterior sample paths:
    posterior sample path -> Gaussian smoothing -> threshold -> IoU.

Place this file next to:
    run_2d_excursion_gp_private_sigmoid_smoothed.py
    synthetic_pollution_utils.py
    dp_utils.py
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import gaussian_filter

try:
    import dp_utils as DP
except ImportError as exc:
    raise ImportError(
        "Could not import dp_utils.py. Place the tightened dp_utils.py in the "
        "same directory as this script."
    ) from exc

import run_2d_excursion_gp_private_sigmoid_smoothed as base


_REQUIRED_DP_ACCOUNTANT_VERSION = "tight-rdp-2026-09"
if getattr(DP, "ACCOUNTANT_VERSION", None) != _REQUIRED_DP_ACCOUNTANT_VERSION:
    raise ImportError(
        "dp_utils.py does not provide the required tightened accountant "
        f"version {_REQUIRED_DP_ACCOUNTANT_VERSION!r}; found "
        f"{getattr(DP, 'ACCOUNTANT_VERSION', None)!r}."
    )


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------


def median_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    return (
        float(np.median(x)),
        float(np.quantile(x, 0.25)),
        float(np.quantile(x, 0.75)),
    )


def fmt_scalar(x, digits=3):
    if x is None or not np.isfinite(x):
        return "nan"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.{digits}f}"


def fmt_med_iqr(x, digits=3, suffix=""):
    med, q25, q75 = median_iqr(x)
    return (
        f"{fmt_scalar(med, digits)}{suffix} "
        f"[{fmt_scalar(q25, digits)}{suffix},{fmt_scalar(q75, digits)}{suffix}]"
    )


def print_bash_table(rows):
    col1 = max(len(r[0]) for r in rows + [("Metric", "")])
    col2 = max(len(r[1]) for r in rows + [("", "median [IQR]")])

    sep = "-" * (col1 + col2 + 5)
    print()
    print(sep)
    print(f"{'Metric':<{col1}} | {'median [IQR]':>{col2}}")
    print(sep)
    for name, value in rows:
        print(f"{name:<{col1}} | {value:>{col2}}")
    print(sep)
    print()


# ---------------------------------------------------------------------
# Tightened generic-response privacy accountant
# ---------------------------------------------------------------------


def tightened_generic_dp_details_2d(
    n,
    ell,
    r,
    sigma,
    M_Y,
    L,
    delta,
    grid_size,
):
    """Return tightened generic-response DP details on the unit square.

    For ``k(x,x') = exp(-||x-x'||/ell)`` on ``[0,1]^2``, the global kernel
    lower bound is ``kappa = exp(-sqrt(2)/ell)``.  The accountant uses the
    minimum of the coupled and operator sensitivity bounds, the signed
    rank-two covariance bound, and the improved RDP-to-DP conversion.
    """
    if ell <= 0.0:
        raise ValueError("ell must be positive for DP accounting.")

    kappa = float(np.exp(-np.sqrt(2.0) / ell))
    sensitivity = DP.generic_sensitivity_components(
        n=n,
        r=r,
        kappa=kappa,
        M_Y=M_Y,
    )
    source = (
        "coupled"
        if sensitivity["coupled"] <= sensitivity["operator"]
        else "operator"
    )
    details = dict(
        DP.epsilon_for_delta(
            n=n,
            r=r,
            kappa=kappa,
            sigma=sigma,
            eta=0.0,
            L=L,
            delta=delta,
            model="generic",
            M_Y=M_Y,
            sensitivity_override=sensitivity["minimum"],
            grid_size=grid_size,
            return_details=True,
        )
    )
    details.update(
        {
            "AccountantVersion": _REQUIRED_DP_ACCOUNTANT_VERSION,
            "Domain": "[0,1]^2",
            "Kappa": kappa,
            "SensitivitySource": source,
            "SensitivityCoupled": float(sensitivity["coupled"]),
            "SensitivityOperator": float(sensitivity["operator"]),
        }
    )
    return details


# ---------------------------------------------------------------------
# GP helper with effective dimension
# ---------------------------------------------------------------------


def gp_posterior_probability_bce_iou_deff(
    world,
    X_eval,
    s_true_eval,
    ell,
    r,
    sigma,
    threshold,
    probability_threshold=None,
    jitter=1e-8,
):
    """
    Compute posterior probability map, BCE, optional IoU of thresholded
    probability map, and effective dimension trace K(K+r^2 I)^(-1).
    """
    X_train = world.X_train
    y_train = world.y_train
    n = X_train.shape[0]

    K_xx = base.exponential_kernel(X_train, X_train, ell)
    K_r = K_xx + (r**2 + jitter) * np.eye(n)

    c_factor = cho_factor(K_r, lower=True, check_finite=False)

    alpha = cho_solve(c_factor, y_train, check_finite=False)

    # d_eff = tr(K(K+r^2I)^(-1)) = n - r^2 tr((K+r^2I)^(-1)).
    # This is exact but costs an n x n triangular solve.
    K_r_inv = cho_solve(c_factor, np.eye(n), check_finite=False)
    d_eff = float(n - (r**2) * np.trace(K_r_inv))

    K_eval_train = base.exponential_kernel(X_eval, X_train, ell)

    mu = K_eval_train @ alpha
    v = cho_solve(c_factor, K_eval_train.T, check_finite=False)

    var_diag = 1.0 - np.sum(K_eval_train * v.T, axis=1)
    var_diag = np.maximum(var_diag, 1e-12)

    p = base.posterior_excursion_probability(
        mu=mu,
        var_diag=var_diag,
        sigma=sigma,
        threshold=threshold,
    )

    bce = float(base.binary_cross_entropy(p, s_true_eval))

    if probability_threshold is None:
        iou = np.nan
    else:
        iou = float(base.iou_score(p >= probability_threshold, s_true_eval))

    return p, bce, iou, d_eff


# ---------------------------------------------------------------------
# Private posterior path metrics
# ---------------------------------------------------------------------


def cholesky_with_jitter(cov, base_jitter=1e-8, max_tries=6):
    cov = 0.5 * (cov + cov.T)
    eye = np.eye(cov.shape[0])

    for k in range(max_tries):
        jitter = base_jitter * (10**k)
        try:
            return np.linalg.cholesky(cov + jitter * eye)
        except np.linalg.LinAlgError:
            pass

    return np.linalg.cholesky(cov + 1e-4 * eye)


def smooth_path_grids(path_grids, sigma_pixels, truncate):
    path_grids = np.asarray(path_grids)

    if sigma_pixels <= 0:
        return path_grids.copy()

    out = np.empty_like(path_grids, dtype=float)
    for j, grid in enumerate(path_grids):
        out[j] = gaussian_filter(
            grid,
            sigma=sigma_pixels,
            mode="nearest",
            truncate=truncate,
        )
    return out


def private_smoothed_path_iou_stats(
    world,
    X_path,
    true_path_flat,
    path_grid_size,
    ell,
    r,
    sigma,
    threshold,
    n_paths,
    rng,
    smoothing_sigma,
    smoothing_truncate,
):
    """
    Draw n_paths private posterior sample paths for one world, smooth them,
    threshold them, and return mean/std IoU across posterior draws.
    """
    mu, cov = base.gp_posterior_mean_and_cov(
        X_train=world.X_train,
        y_train=world.y_train,
        X_eval=X_path,
        ell=ell,
        r=r,
    )

    L_chol = cholesky_with_jitter(cov)

    Z = rng.standard_normal(size=(X_path.shape[0], n_paths))
    paths = (mu[:, None] + sigma * (L_chol @ Z)).T
    path_grids_raw = paths.reshape(n_paths, path_grid_size, path_grid_size)

    path_grids = smooth_path_grids(
        path_grids_raw,
        sigma_pixels=smoothing_sigma,
        truncate=smoothing_truncate,
    )
    path_values = path_grids.reshape(n_paths, -1)

    ious = np.array([
        base.iou_score(path_values[k] >= threshold, true_path_flat)
        for k in range(n_paths)
    ])

    return {
        "mean": float(np.mean(ious)),
        "std": float(np.std(ious, ddof=1)) if len(ious) > 1 else 0.0,
        "all": ious,
    }



# ---------------------------------------------------------------------
# Loading fixed hyperparameters from a grid-search JSON summary
# ---------------------------------------------------------------------


def _coerce_existing_json_path(path):
    """
    Resolve a grid-search JSON path.

    The default path points to the usual output directory. If that does not
    exist, we also try the same basename in the current working directory.
    """
    if path is None:
        return None

    candidates = [path]

    if not path.endswith(".json"):
        candidates.append(path + ".json")

    basename = os.path.basename(path)
    if basename != path:
        candidates.append(basename)
        if not basename.endswith(".json"):
            candidates.append(basename + ".json")

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find hyperparameter JSON. Tried:\n  "
        + "\n  ".join(candidates)
    )


def _nested_get(payload, path, default=None):
    cur = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _argv_has_flag(flag):
    """
    Return True if a command-line flag was explicitly provided.

    This lets the JSON file set defaults while still allowing command-line
    overrides such as --M-xi or --public-ell.
    """
    prefix = flag + "="
    return any(arg == flag or arg.startswith(prefix) for arg in sys.argv[1:])


def _set_from_json_unless_cli(args, attr, value, flag):
    if value is None:
        return
    if _argv_has_flag(flag):
        return
    setattr(args, attr, value)


def apply_gridsearch_json_to_args(args):
    """
    Load public/private hyperparameters and experiment settings from a saved
    grid-search summary JSON.

    M_xi is read from the JSON contents, not from the filename. This avoids
    mistakes when a file name contains an old Mxi value but the JSON payload
    has the correct one.

    Command-line arguments override JSON values.
    """
    resolved_path = _resolve_hyperparams_json_template(args.hyperparams_json, args)
    json_path = _coerce_existing_json_path(resolved_path)
    if json_path is None:
        return args, None

    with open(json_path, "r") as f:
        payload = json.load(f)

    public_best = _nested_get(payload, ["public", "best_hyperparameters"], {})
    public_thr = _nested_get(payload, ["public", "population_threshold"], {})

    private_best = (
        _nested_get(payload, ["private", "refined_best"], None)
        or _nested_get(payload, ["private", "best_hyperparameters"], None)
        or _nested_get(payload, ["private", "coarse_best"], {})
    )

    saved_args = payload.get("args", {})

    _set_from_json_unless_cli(args, "public_ell", public_best.get("ell"), "--public-ell")
    _set_from_json_unless_cli(args, "public_r", public_best.get("r"), "--public-r")
    _set_from_json_unless_cli(args, "public_sigma", public_best.get("sigma"), "--public-sigma")
    _set_from_json_unless_cli(args, "public_C", public_thr.get("C"), "--public-C")

    _set_from_json_unless_cli(args, "private_ell", private_best.get("ell"), "--private-ell")
    _set_from_json_unless_cli(args, "private_r", private_best.get("r"), "--private-r")
    _set_from_json_unless_cli(args, "private_sigma", private_best.get("sigma"), "--private-sigma")

    # Use the JSON payload, not the filename, for M_xi and related settings.
    # Do not override n_worlds or n_paths; this script intentionally uses a
    # larger Monte Carlo population than the grid-search run.
    _set_from_json_unless_cli(args, "n_data", saved_args.get("n_data"), "--n-data")
    _set_from_json_unless_cli(args, "M_xi", saved_args.get("M_xi"), "--M-xi")
    _set_from_json_unless_cli(args, "threshold_fraction", saved_args.get("threshold_fraction"), "--threshold-fraction")

    _set_from_json_unless_cli(args, "latent_seed", saved_args.get("latent_seed"), "--latent-seed")
    _set_from_json_unless_cli(args, "data_seed", saved_args.get("data_seed"), "--data-seed")
    _set_from_json_unless_cli(args, "grid_size", saved_args.get("grid_size"), "--grid-size")
    _set_from_json_unless_cli(args, "reference_grid_size", saved_args.get("reference_grid_size"), "--reference-grid-size")

    _set_from_json_unless_cli(args, "latent_K", saved_args.get("latent_K"), "--latent-K")
    _set_from_json_unless_cli(args, "min_width", saved_args.get("min_width"), "--min-width")
    _set_from_json_unless_cli(args, "max_width", saved_args.get("max_width"), "--max-width")
    _set_from_json_unless_cli(args, "anisotropy_max", saved_args.get("anisotropy_max"), "--anisotropy-max")
    _set_from_json_unless_cli(args, "background_strength", saved_args.get("background_strength"), "--background-strength")
    _set_from_json_unless_cli(args, "nonlinearity", saved_args.get("nonlinearity"), "--nonlinearity")
    _set_from_json_unless_cli(args, "sigmoid_gamma", saved_args.get("sigmoid_gamma"), "--sigmoid-gamma")
    _set_from_json_unless_cli(args, "sigmoid_center_quantile", saved_args.get("sigmoid_center_quantile"), "--sigmoid-center-quantile")

    _set_from_json_unless_cli(args, "epsilon0", saved_args.get("epsilon0"), "--epsilon0")
    _set_from_json_unless_cli(args, "M_Y", saved_args.get("M_Y"), "--M-Y")
    _set_from_json_unless_cli(args, "L", saved_args.get("L"), "--L")
    _set_from_json_unless_cli(args, "alpha_grid_size", saved_args.get("alpha_grid_size"), "--alpha-grid-size")

    _set_from_json_unless_cli(args, "path_grid_size", saved_args.get("path_grid_size"), "--path-grid-size")
    _set_from_json_unless_cli(args, "path_smoothing_sigma", saved_args.get("path_smoothing_sigma"), "--path-smoothing-sigma")
    _set_from_json_unless_cli(args, "path_smoothing_truncate", saved_args.get("path_smoothing_truncate"), "--path-smoothing-truncate")

    _set_from_json_unless_cli(args, "delta", payload.get("delta"), "--delta")

    args.loaded_hyperparams_json = json_path
    return args, payload




# ---------------------------------------------------------------------
# Loading fixed hyperparameters from a grid-search JSON summary
# ---------------------------------------------------------------------


def _format_compact_number(x):
    """Format a numeric value compactly for filenames, e.g. 0.6 -> '0.6'."""
    try:
        x = float(x)
    except Exception:
        return str(x)
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return format(x, "g")


def _resolve_hyperparams_json_template(path, args):
    """
    Fill simple placeholders in the hyperparameter JSON path.

    Supported placeholders:
        {Mxi}, {n_data}, {epsilon0}, {grid_size}, {latent_seed}, {data_seed}

    This makes the default JSON filename track command-line overrides such as
    --M-xi 0.6.
    """
    if path is None:
        return None

    replacements = {
        "Mxi": _format_compact_number(getattr(args, "M_xi", None)),
        "n_data": str(getattr(args, "n_data", "")),
        "epsilon0": _format_compact_number(getattr(args, "epsilon0", None)),
        "grid_size": str(getattr(args, "grid_size", "")),
        "latent_seed": str(getattr(args, "latent_seed", "")),
        "data_seed": str(getattr(args, "data_seed", "")),
    }

    try:
        return path.format(**replacements)
    except Exception:
        return path


def _coerce_existing_json_path(path):
    """
    Resolve a grid-search JSON path. If the path does not exist, also try the
    same basename in the current working directory, with and without .json.
    """
    if path is None:
        return None

    candidates = [path]

    if not path.endswith(".json"):
        candidates.append(path + ".json")

    basename = os.path.basename(path)
    if basename != path:
        candidates.append(basename)
        if not basename.endswith(".json"):
            candidates.append(basename + ".json")

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find hyperparameter JSON. Tried:\n  "
        + "\n  ".join(candidates)
    )


def _nested_get(payload, path, default=None):
    cur = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _argv_has_flag(flag):
    """
    Return True if a command-line flag was explicitly provided.

    This lets the JSON file set defaults while still allowing command-line
    overrides such as --M-xi or --public-ell.
    """
    prefix = flag + "="
    return any(arg == flag or arg.startswith(prefix) for arg in sys.argv[1:])


def _set_from_json_unless_cli(args, attr, value, flag):
    if value is None:
        return
    if _argv_has_flag(flag):
        return
    setattr(args, attr, value)


def apply_gridsearch_json_to_args(args):
    """
    Load public/private hyperparameters and experiment settings from a saved
    grid-search summary JSON.

    M_xi is read from the JSON contents, not from the filename. However, the
    default filename itself is a template depending on the command-line M_xi;
    for example --M-xi 0.6 looks for an Mxi0.6 JSON by default.

    Command-line arguments override JSON values.
    """
    resolved_path = _resolve_hyperparams_json_template(args.hyperparams_json, args)
    json_path = _coerce_existing_json_path(resolved_path)
    if json_path is None:
        return args, None

    with open(json_path, "r") as f:
        payload = json.load(f)

    public_best = _nested_get(payload, ["public", "best_hyperparameters"], {})
    public_thr = _nested_get(payload, ["public", "population_threshold"], {})

    private_best = (
        _nested_get(payload, ["private", "refined_best"], None)
        or _nested_get(payload, ["private", "best_hyperparameters"], None)
        or _nested_get(payload, ["private", "coarse_best"], {})
    )

    saved_args = payload.get("args", {})

    _set_from_json_unless_cli(args, "public_ell", public_best.get("ell"), "--public-ell")
    _set_from_json_unless_cli(args, "public_r", public_best.get("r"), "--public-r")
    _set_from_json_unless_cli(args, "public_sigma", public_best.get("sigma"), "--public-sigma")
    _set_from_json_unless_cli(args, "public_C", public_thr.get("C"), "--public-C")

    _set_from_json_unless_cli(args, "private_ell", private_best.get("ell"), "--private-ell")
    _set_from_json_unless_cli(args, "private_r", private_best.get("r"), "--private-r")
    _set_from_json_unless_cli(args, "private_sigma", private_best.get("sigma"), "--private-sigma")

    # Use the JSON payload, not the filename, for M_xi and related settings
    # unless the user explicitly overrides them on the command line.
    _set_from_json_unless_cli(args, "n_data", saved_args.get("n_data"), "--n-data")
    _set_from_json_unless_cli(args, "M_xi", saved_args.get("M_xi"), "--M-xi")
    _set_from_json_unless_cli(args, "threshold_fraction", saved_args.get("threshold_fraction"), "--threshold-fraction")

    _set_from_json_unless_cli(args, "latent_seed", saved_args.get("latent_seed"), "--latent-seed")
    _set_from_json_unless_cli(args, "data_seed", saved_args.get("data_seed"), "--data-seed")
    _set_from_json_unless_cli(args, "grid_size", saved_args.get("grid_size"), "--grid-size")
    _set_from_json_unless_cli(args, "reference_grid_size", saved_args.get("reference_grid_size"), "--reference-grid-size")

    _set_from_json_unless_cli(args, "latent_K", saved_args.get("latent_K"), "--latent-K")
    _set_from_json_unless_cli(args, "min_width", saved_args.get("min_width"), "--min-width")
    _set_from_json_unless_cli(args, "max_width", saved_args.get("max_width"), "--max-width")
    _set_from_json_unless_cli(args, "anisotropy_max", saved_args.get("anisotropy_max"), "--anisotropy-max")
    _set_from_json_unless_cli(args, "background_strength", saved_args.get("background_strength"), "--background-strength")
    _set_from_json_unless_cli(args, "nonlinearity", saved_args.get("nonlinearity"), "--nonlinearity")
    _set_from_json_unless_cli(args, "sigmoid_gamma", saved_args.get("sigmoid_gamma"), "--sigmoid-gamma")
    _set_from_json_unless_cli(args, "sigmoid_center_quantile", saved_args.get("sigmoid_center_quantile"), "--sigmoid-center-quantile")

    _set_from_json_unless_cli(args, "epsilon0", saved_args.get("epsilon0"), "--epsilon0")
    _set_from_json_unless_cli(args, "M_Y", saved_args.get("M_Y"), "--M-Y")
    _set_from_json_unless_cli(args, "L", saved_args.get("L"), "--L")
    _set_from_json_unless_cli(args, "alpha_grid_size", saved_args.get("alpha_grid_size"), "--alpha-grid-size")

    _set_from_json_unless_cli(args, "path_grid_size", saved_args.get("path_grid_size"), "--path-grid-size")
    _set_from_json_unless_cli(args, "path_smoothing_sigma", saved_args.get("path_smoothing_sigma"), "--path-smoothing-sigma")
    _set_from_json_unless_cli(args, "path_smoothing_truncate", saved_args.get("path_smoothing_truncate"), "--path-smoothing-truncate")

    _set_from_json_unless_cli(args, "delta", payload.get("delta"), "--delta")

    args.loaded_hyperparams_json = json_path
    return args, payload



# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    # Monte Carlo setup.
    parser.add_argument("--n-worlds", type=int, default=1000)
    parser.add_argument("--n-data", type=int, default=400)
    parser.add_argument("--M-xi", type=float, default=0.5)
    parser.add_argument("--n-paths", type=int, default=50)

    parser.add_argument("--latent-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=1)
    parser.add_argument("--path-seed", type=int, default=123)

    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument(
        "--path-grid-size",
        type=int,
        default=40,
        help=(
            "Grid size for exact posterior path sampling. The path calculation "
            "is cubic in path_grid_size^2, so increasing this can be expensive."
        ),
    )
    parser.add_argument("--reference-grid-size", type=int, default=180)

    # Latent model.
    parser.add_argument("--latent-K", type=int, default=2, help="Use 0 for random K.")
    parser.add_argument("--min-width", type=float, default=0.20)
    parser.add_argument("--max-width", type=float, default=0.35)
    parser.add_argument("--anisotropy-max", type=float, default=1.4)
    parser.add_argument("--background-strength", type=float, default=0.0)
    parser.add_argument("--nonlinearity", choices=["identity", "sigmoid"], default="sigmoid")
    parser.add_argument("--sigmoid-gamma", type=float, default=5.0)
    parser.add_argument("--sigmoid-center-quantile", type=float, default=0.65)

    # Threshold q on original [0,1] generated field g_*.
    parser.add_argument("--threshold-fraction", type=float, default=0.5)

    # Fixed public hyperparameters and selected probability threshold.
    parser.add_argument("--public-ell", type=float, default=0.5)
    parser.add_argument("--public-r", type=float, default=1.3)
    parser.add_argument("--public-sigma", type=float, default=0.18)
    parser.add_argument("--public-C", type=float, default=0.417)

    # Fixed private hyperparameters.
    parser.add_argument("--private-ell", type=float, default=0.135)
    parser.add_argument("--private-r", type=float, default=8.0)
    parser.add_argument("--private-sigma", type=float, default=0.0375)

    # DP setup.
    parser.add_argument("--epsilon0", type=float, default=10.0)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--M-Y", type=float, default=1.0)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument(
        "--alpha-grid-size",
        type=int,
        default=81,
        help=(
            "Optimizer grid size used by the tightened dp_utils.py accountant. "
            "The --alpha-grid-size name is retained for compatibility."
        ),
    )

    # Smoothed private sample-path release.
    parser.add_argument(
        "--path-smoothing-sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma for private sample-path values, in path-grid pixels.",
    )
    parser.add_argument("--path-smoothing-truncate", type=float, default=3.0)

    # Output / progress.
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-npz", type=str, default=None)

    parser.add_argument(
        "--hyperparams-json",
        type=str,
        default=(
            "excursion_2d_population_results/"
            "pop_nreps30_n400_Mxi{Mxi}_eps9_grid80_lseed0_dseed1_summary.json"
        ),
        help=(
            "Saved grid-search summary JSON from which to read public/private "
            "hyperparameters and experiment settings. Command-line arguments "
            "override values read from this file. Use --hyperparams-json none "
            "to disable JSON loading."
        ),
    )

    args = parser.parse_args()

    if args.hyperparams_json is not None and args.hyperparams_json.lower() == "none":
        args.hyperparams_json = None

    args, loaded_summary_payload = apply_gridsearch_json_to_args(args)

    if args.n_worlds <= 0:
        raise ValueError("--n-worlds must be positive.")
    if args.n_data <= 1:
        raise ValueError("--n-data must be at least 2.")
    if args.n_paths <= 0:
        raise ValueError("--n-paths must be positive.")
    if not (0.0 <= args.M_xi <= 1.0):
        raise ValueError("--M-xi must satisfy 0 <= M_xi <= 1.")
    if not (0.0 < args.threshold_fraction < 1.0):
        raise ValueError("--threshold-fraction must lie in (0,1).")
    if args.path_grid_size <= 1:
        raise ValueError("--path-grid-size must be at least 2.")
    if args.path_smoothing_sigma < 0:
        raise ValueError("--path-smoothing-sigma must be nonnegative.")
    if args.M_Y < 0:
        raise ValueError("--M-Y must be nonnegative.")
    if args.L <= 0:
        raise ValueError("--L must be positive.")
    if args.alpha_grid_size < 3:
        raise ValueError("--alpha-grid-size must be at least 3.")

    delta = args.delta if args.delta is not None else args.n_data ** (-1.1)

    latent_threshold = 2.0 * args.threshold_fraction - 1.0
    threshold = (1.0 - args.M_xi) * latent_threshold

    print("Fixed-hyperparameter Monte Carlo summary")
    print("----------------------------------------")
    print(f"n_worlds = {args.n_worlds}")
    print(f"n_data = {args.n_data}")
    print(f"n_paths per world = {args.n_paths}")
    if getattr(args, "loaded_hyperparams_json", None) is not None:
        print(f"loaded hyperparams JSON = {args.loaded_hyperparams_json}")
    else:
        print("loaded hyperparams JSON = none")
    print(f"M_xi = {args.M_xi}")
    print(f"q on [0,1] scale = {args.threshold_fraction}")
    print(f"signed latent threshold = {latent_threshold:.6f}")
    print(f"signal/path threshold = {threshold:.6f}")
    print(f"grid_size = {args.grid_size}")
    print(f"path_grid_size = {args.path_grid_size}")
    print(f"path_smoothing_sigma = {args.path_smoothing_sigma}")
    print()
    print("Public hyperparameters:")
    print(f"  ell={args.public_ell}, r={args.public_r}, sigma={args.public_sigma}, C={args.public_C}")
    print("Private hyperparameters:")
    print(f"  ell={args.private_ell}, r={args.private_r}, sigma={args.private_sigma}")
    print()

    # DP epsilons are deterministic for fixed parameters. Compute them here
    # through the tightened accountant rather than inheriting a potentially
    # stale privacy helper from the simulation module.
    dp_public = tightened_generic_dp_details_2d(
        n=args.n_data,
        ell=args.public_ell,
        r=args.public_r,
        sigma=args.public_sigma,
        M_Y=args.M_Y,
        L=args.L,
        delta=delta,
        grid_size=args.alpha_grid_size,
    )
    dp_private = tightened_generic_dp_details_2d(
        n=args.n_data,
        ell=args.private_ell,
        r=args.private_r,
        sigma=args.private_sigma,
        M_Y=args.M_Y,
        L=args.L,
        delta=delta,
        grid_size=args.alpha_grid_size,
    )
    eps_public = float(dp_public["Epsilon"])
    eps_private = float(dp_private["Epsilon"])

    print(f"DP accountant = {_REQUIRED_DP_ACCOUNTANT_VERSION}")
    print(f"epsilon public/unconstrained = {eps_public:.6f}")
    print(
        "  sensitivity public = "
        f"{dp_public['DeltaN']:.6f} ({dp_public['SensitivitySource']})"
    )
    print(f"epsilon private = {eps_private:.6f}")
    print(
        "  sensitivity private = "
        f"{dp_private['DeltaN']:.6f} ({dp_private['SensitivitySource']})"
    )
    print()

    print("Sampling latent fields and datasets...")
    fields, worlds = base.sample_population_worlds(
        n_reps=args.n_worlds,
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

    # Evaluation grid for probability BCE and public IoU.
    _, _, _, _, X_grid = base.make_unit_square_grid(args.grid_size)
    X_eval = X_grid.reshape(-1, 2)

    G_all = fields(X_grid)
    F_all = base.to_signed_latent(G_all)
    signal_all = (1.0 - args.M_xi) * F_all
    s_true_all_grid = signal_all >= threshold
    s_true_all = s_true_all_grid.reshape(args.n_worlds, -1)

    # Separate path grid for private sample-path IoU.
    _, _, _, _, X_path_grid = base.make_unit_square_grid(args.path_grid_size)
    X_path = X_path_grid.reshape(-1, 2)

    G_path_all = fields(X_path_grid)
    F_path_all = base.to_signed_latent(G_path_all)
    signal_path_all = (1.0 - args.M_xi) * F_path_all
    s_true_path_all_grid = signal_path_all >= threshold
    s_true_path_all = s_true_path_all_grid.reshape(args.n_worlds, -1)

    rng_paths = np.random.default_rng(args.path_seed)

    public_bce = np.empty(args.n_worlds)
    private_bce = np.empty(args.n_worlds)
    public_iou = np.empty(args.n_worlds)
    d_eff_public = np.empty(args.n_worlds)
    d_eff_private = np.empty(args.n_worlds)
    path_iou_mean = np.empty(args.n_worlds)
    path_iou_std = np.empty(args.n_worlds)

    t0 = time.time()

    for j, world in enumerate(worlds):
        _, bce_pub, iou_pub, deff_pub = gp_posterior_probability_bce_iou_deff(
            world=world,
            X_eval=X_eval,
            s_true_eval=s_true_all[j],
            ell=args.public_ell,
            r=args.public_r,
            sigma=args.public_sigma,
            threshold=threshold,
            probability_threshold=args.public_C,
        )

        _, bce_priv, _, deff_priv = gp_posterior_probability_bce_iou_deff(
            world=world,
            X_eval=X_eval,
            s_true_eval=s_true_all[j],
            ell=args.private_ell,
            r=args.private_r,
            sigma=args.private_sigma,
            threshold=threshold,
            probability_threshold=None,
        )

        path_stats = private_smoothed_path_iou_stats(
            world=world,
            X_path=X_path,
            true_path_flat=s_true_path_all[j],
            path_grid_size=args.path_grid_size,
            ell=args.private_ell,
            r=args.private_r,
            sigma=args.private_sigma,
            threshold=threshold,
            n_paths=args.n_paths,
            rng=rng_paths,
            smoothing_sigma=args.path_smoothing_sigma,
            smoothing_truncate=args.path_smoothing_truncate,
        )

        public_bce[j] = bce_pub
        private_bce[j] = bce_priv
        public_iou[j] = iou_pub
        d_eff_public[j] = deff_pub
        d_eff_private[j] = deff_priv
        path_iou_mean[j] = path_stats["mean"]
        path_iou_std[j] = path_stats["std"]

        if args.progress_every > 0 and ((j + 1) % args.progress_every == 0 or j == 0):
            elapsed = time.time() - t0
            rate = (j + 1) / max(elapsed, 1e-12)
            remaining = (args.n_worlds - (j + 1)) / max(rate, 1e-12)
            print(
                f"[{j + 1:>5}/{args.n_worlds}] "
                f"elapsed={elapsed/60:.1f} min, "
                f"eta={remaining/60:.1f} min"
            )

    relative_bce_increase = 100.0 * (private_bce - public_bce) / np.maximum(public_bce, 1e-12)
    relative_iou_gap = 100.0 * (public_iou - path_iou_mean) / np.maximum(public_iou, 1e-12)

    rows = [
        ("epsilon, unconstrained", fmt_scalar(eps_public, 1)),
        ("d_eff, unconstrained", fmt_med_iqr(d_eff_public, digits=2)),
        ("d_eff, epsilon<10", fmt_med_iqr(d_eff_private, digits=2)),
        ("relative BCE increase", fmt_med_iqr(relative_bce_increase, digits=1, suffix="%")),
        ("IoU, unconstrained", fmt_med_iqr(public_iou, digits=3)),
        ("1-path IoU mean, epsilon<10", fmt_med_iqr(path_iou_mean, digits=3)),
        ("1-path IoU STD, epsilon<10", fmt_med_iqr(path_iou_std, digits=3)),
        ("relative IoU gap", fmt_med_iqr(relative_iou_gap, digits=1, suffix="%")),
    ]

    print_bash_table(rows)

    print("Notes:")
    print("  - epsilon, unconstrained is the DP bound for the public hyperparameters.")
    print("  - private epsilon for the fixed private hyperparameters is "
          f"{eps_private:.6f}.")
    print("  - 1-path metrics use smoothed private sample paths before thresholding.")
    print("  - The path threshold is the natural signal threshold "
          f"t={threshold:.6f}.")
    print()

    summary = {
        "args": vars(args),
        "loaded_hyperparams_json": getattr(args, "loaded_hyperparams_json", None),
        "dp_accountant_version": _REQUIRED_DP_ACCOUNTANT_VERSION,
        "dp_public": dp_public,
        "dp_private": dp_private,
        "delta": delta,
        "threshold": threshold,
        "latent_threshold": latent_threshold,
        "epsilon_public": eps_public,
        "epsilon_private": eps_private,
        "rows": rows,
        "metrics": {
            "public_bce": public_bce.tolist(),
            "private_bce": private_bce.tolist(),
            "d_eff_public": d_eff_public.tolist(),
            "d_eff_private": d_eff_private.tolist(),
            "public_iou": public_iou.tolist(),
            "path_iou_mean": path_iou_mean.tolist(),
            "path_iou_std": path_iou_std.tolist(),
            "relative_bce_increase": relative_bce_increase.tolist(),
            "relative_iou_gap": relative_iou_gap.tolist(),
        },
    }

    if args.output_json is not None:
        outdir = os.path.dirname(args.output_json)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON summary to {args.output_json}")

    if args.output_npz is not None:
        outdir = os.path.dirname(args.output_npz)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        np.savez(
            args.output_npz,
            public_bce=public_bce,
            private_bce=private_bce,
            d_eff_public=d_eff_public,
            d_eff_private=d_eff_private,
            public_iou=public_iou,
            path_iou_mean=path_iou_mean,
            path_iou_std=path_iou_std,
            relative_bce_increase=relative_bce_increase,
            relative_iou_gap=relative_iou_gap,
            epsilon_public=np.array(eps_public),
            epsilon_private=np.array(eps_private),
            dp_accountant_version=np.array(_REQUIRED_DP_ACCOUNTANT_VERSION),
            dp_delta_n_public=np.array(dp_public["DeltaN"]),
            dp_delta_n_private=np.array(dp_private["DeltaN"]),
            dp_optimal_alpha_public=np.array(dp_public["OptimalAlpha"]),
            dp_optimal_alpha_private=np.array(dp_private["OptimalAlpha"]),
            threshold=np.array(threshold),
            latent_threshold=np.array(latent_threshold),
        )
        print(f"Saved NPZ arrays to {args.output_npz}")


if __name__ == "__main__":
    main()
