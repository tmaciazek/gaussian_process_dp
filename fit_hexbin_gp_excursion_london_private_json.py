#!/usr/bin/env python3
"""
Fit a spatial GP excursion-set model to hexagon-level London transaction prices.

Pipeline:
  1. Read HM Land Registry Price Paid Data (PPD).
  2. Select Greater London existing leasehold flats/maisonettes by default:
       property_type == F, new_build == N, tenure == L, ppd_category == A.
  3. Join transaction postcodes to a postcode-coordinate lookup (ONSPD or Code-Point Open).
  4. Aggregate transactions to a fixed Matplotlib hexbin grid using median log sale price.
  5. Define binary labels s_h = 1{median_log_price_h > threshold}.
  6. Form raw centred responses y_raw,h = median_log_price_h - threshold,
     then symmetrically clip and rescale y_h = clip(y_raw,h, -B, B)/B so M_Y=1.
  7. Select GP hyperparameters by validation BCE.
  8. Select excursion probability cutoff C by validation IoU.
  9. Refit on all hexagons and release/plot the boundary of {p_D(x) >= C}.

Example:
  python fit_hexbin_gp_excursion_london.py \
      --ppd pp-2018.csv \
      --postcode-lookup ONSPD.csv \
      --threshold 13.5 \
      --gridsize 40 \
      --mincnt 1 \
      --out-figure london_hexbin_gp_excursion.png \
      --out-boundary london_hexbin_gp_boundary.csv \
      --out-summary london_hexbin_gp_summary.json

Notes:
  - Coordinates are normalised to [0,1]^2 for GP fitting, so lengthscales are in
    units of the normalised Greater London bounding box.
  - The exponential kernel is k(x,z)=exp(-||x-z||/ell).
  - The GP posterior follows the paper-style parametrisation:
        K_r = K + r^2 I,
        mu(x) = k_x^T K_r^{-1} y,
        k_D(x,x) = 1 - k_x^T K_r^{-1} k_x,
        p_D(x) = Phi(mu(x)/(sigma sqrt(k_D(x,x))))
    because the response is centred at the excursion threshold and then rescaled symmetrically, so the excursion event is still f(x)>0.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import ndtr
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
# No DP-accounting import is needed here: hyperparameters and epsilon are read from JSON.


PPD_COLUMNS = [
    "transaction_id",
    "price",
    "date",
    "postcode",
    "property_type",
    "new_build",
    "tenure",
    "paon",
    "saon",
    "street",
    "locality",
    "town_city",
    "district",
    "county",
    "ppd_category",
    "record_status",
]

SEGMENT_DESCRIPTIONS = {
    "existing-leasehold-flats": (
        "existing leasehold flats/maisonettes "
        "(property_type=F, new_build=N, tenure=L, ppd_category=A)"
    ),
    "all-standard-residential": (
        "all standard Greater London residential transactions, excluding property_type=O"
    ),
}


def normalise_postcode(s: pd.Series) -> pd.Series:
    """Upper-case postcodes and remove all whitespace for reliable joins."""
    return (
        s.astype("string")
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
        .str.strip()
    )


def read_ppd_london(path: str | Path, segment: str) -> pd.DataFrame:
    """Read HMLR PPD CSV and keep the requested Greater London segment."""
    usecols = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]
    names = [PPD_COLUMNS[i] for i in usecols]

    df = pd.read_csv(
        path,
        header=None,
        names=names,
        usecols=usecols,
        dtype="string",
        low_memory=False,
    )

    for col in ["property_type", "new_build", "tenure", "county", "ppd_category", "record_status"]:
        df[col] = df[col].astype("string").str.upper().str.strip()

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    london = df[
        (df["county"] == "GREATER LONDON")
        & (df["record_status"] == "A")
        & (df["ppd_category"] == "A")
        & df["postcode"].notna()
        & df["price"].notna()
        & (df["price"] > 0)
    ].copy()

    if segment == "existing-leasehold-flats":
        london = london[
            (london["property_type"] == "F")
            & (london["new_build"] == "N")
            & (london["tenure"] == "L")
        ].copy()
    elif segment == "all-standard-residential":
        london = london[london["property_type"] != "O"].copy()
    else:
        raise ValueError(f"Unknown segment: {segment}")

    london["postcode_key"] = normalise_postcode(london["postcode"])
    london["log_price"] = np.log(london["price"].astype(float))
    return london


def _find_col(cols: list[str], candidates: list[str], pattern: Optional[str] = None) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    if pattern is not None:
        rx = re.compile(pattern, flags=re.IGNORECASE)
        for c in cols:
            if rx.search(c):
                return c
    return None


def read_postcode_lookup(path: str | Path) -> Tuple[pd.DataFrame, str, str, str]:
    """
    Read postcode lookup and detect postcode/coordinate columns.

    Supports common ONSPD columns:
        pcd, pcd2, pcds, oseast1m, osnrth1m, lat, long
    and common Code-Point Open style columns:
        Postcode, Eastings, Northings
    """
    lookup = pd.read_csv(path, dtype="string", low_memory=False)
    cols = list(lookup.columns)

    pc_col = _find_col(cols, ["pcds", "pcd", "pcd2", "postcode", "Postcode"], r"^pcd|post")
    east_col = _find_col(cols, ["oseast1m", "easting", "eastings", "x"], r"east")
    north_col = _find_col(cols, ["osnrth1m", "northing", "northings", "y"], r"north")

    lat_col = _find_col(cols, ["lat", "latitude"], r"^lat")
    lon_col = _find_col(cols, ["long", "lon", "longitude"], r"lon|long")

    if pc_col is None:
        raise ValueError("Could not detect postcode column in lookup. Expected pcd/pcds/postcode.")

    if east_col is not None and north_col is not None:
        x_col, y_col = east_col, north_col
        coord_type = "bng"
    elif lon_col is not None and lat_col is not None:
        x_col, y_col = lon_col, lat_col
        coord_type = "lonlat"
    else:
        raise ValueError(
            "Could not detect coordinate columns. Expected either easting/northing "
            "or longitude/latitude columns."
        )

    keep = lookup[[pc_col, x_col, y_col]].copy()
    keep.columns = ["postcode", "x", "y"]
    keep["postcode_key"] = normalise_postcode(keep["postcode"])
    keep["x"] = pd.to_numeric(keep["x"], errors="coerce")
    keep["y"] = pd.to_numeric(keep["y"], errors="coerce")
    keep = keep.dropna(subset=["postcode_key", "x", "y"]).drop_duplicates("postcode_key")

    return keep[["postcode_key", "x", "y"]], "x", "y", coord_type


def join_coordinates(london: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    mapped = london.merge(lookup, on="postcode_key", how="inner")
    if mapped.empty:
        raise ValueError("No London transactions matched the postcode lookup.")
    return mapped


def make_hexbin_dataset(
    mapped: pd.DataFrame,
    gridsize: int,
    mincnt: int,
) -> pd.DataFrame:
    """Use Matplotlib's hexbin to compute non-empty hexagon centres and median log prices."""
    x = mapped["x"].to_numpy(dtype=float)
    y = mapped["y"].to_numpy(dtype=float)
    log_price = mapped["log_price"].to_numpy(dtype=float)

    fig, ax = plt.subplots()
    hb_med = ax.hexbin(
        x,
        y,
        C=log_price,
        reduce_C_function=np.median,
        gridsize=gridsize,
        mincnt=mincnt,
    )
    centers = np.asarray(hb_med.get_offsets(), dtype=float)
    med_log_price = np.asarray(hb_med.get_array(), dtype=float)
    plt.close(fig)

    # For counts, use the same hexbin grid. Matplotlib returns non-empty bins in
    # the same grid coordinate system; we merge by centre to avoid relying on order.
    fig, ax = plt.subplots()
    hb_count = ax.hexbin(x, y, gridsize=gridsize, mincnt=mincnt)
    count_centers = np.asarray(hb_count.get_offsets(), dtype=float)
    counts = np.asarray(hb_count.get_array(), dtype=float)
    plt.close(fig)

    med_df = pd.DataFrame({
        "x": centers[:, 0],
        "y": centers[:, 1],
        "median_log_price": med_log_price,
    })
    count_df = pd.DataFrame({
        "x": count_centers[:, 0],
        "y": count_centers[:, 1],
        "n_transactions": counts.astype(int),
    })

    # Floating-point centres are generated by the same routine, so exact merge is
    # usually fine; rounding makes the merge robust to tiny representation noise.
    for df in [med_df, count_df]:
        df["x_key"] = df["x"].round(8)
        df["y_key"] = df["y"].round(8)

    hex_df = med_df.merge(
        count_df[["x_key", "y_key", "n_transactions"]],
        on=["x_key", "y_key"],
        how="left",
    ).drop(columns=["x_key", "y_key"])
    hex_df["n_transactions"] = hex_df["n_transactions"].fillna(0).astype(int)
    return hex_df.sort_values(["x", "y"]).reset_index(drop=True)


def normalise_xy(X: np.ndarray, bounds: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
    if bounds is None:
        xmin, ymin = X.min(axis=0)
        xmax, ymax = X.max(axis=0)
        bounds = {"xmin": float(xmin), "xmax": float(xmax), "ymin": float(ymin), "ymax": float(ymax)}
    scale_x = bounds["xmax"] - bounds["xmin"]
    scale_y = bounds["ymax"] - bounds["ymin"]
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("Degenerate coordinate range; cannot normalise coordinates.")
    Xn = np.empty_like(X, dtype=float)
    Xn[:, 0] = (X[:, 0] - bounds["xmin"]) / scale_x
    Xn[:, 1] = (X[:, 1] - bounds["ymin"]) / scale_y
    return Xn, bounds


def exp_kernel(X: np.ndarray, Z: np.ndarray, ell: float) -> np.ndarray:
    X2 = np.sum(X * X, axis=1)[:, None]
    Z2 = np.sum(Z * Z, axis=1)[None, :]
    D2 = np.maximum(X2 + Z2 - 2.0 * X @ Z.T, 0.0)
    D = np.sqrt(D2)
    return np.exp(-D / ell)


def gp_fit(X: np.ndarray, y: np.ndarray, ell: float, r: float, jitter: float = 1e-8):
    K = exp_kernel(X, X, ell)
    K.flat[:: K.shape[0] + 1] += r * r + jitter
    cho = cho_factor(K, lower=True, check_finite=False)
    alpha = cho_solve(cho, y, check_finite=False)
    return cho, alpha


def gp_predict_prob(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    ell: float,
    r: float,
    sigma: float,
    chunk_size: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return p(f>0), posterior mean, and unscaled posterior variance at X_test."""
    cho, alpha = gp_fit(X_train, y_train, ell, r)
    L = cho[0]
    lower = cho[1]
    if not lower:
        # scipy may store an upper factor if lower=False; this script requests lower=True.
        L = L.T

    n_test = X_test.shape[0]
    p = np.empty(n_test, dtype=float)
    mu = np.empty(n_test, dtype=float)
    var = np.empty(n_test, dtype=float)

    for start in range(0, n_test, chunk_size):
        stop = min(start + chunk_size, n_test)
        Ks = exp_kernel(X_train, X_test[start:stop], ell)
        mu_chunk = Ks.T @ alpha
        V = solve_triangular(L, Ks, lower=True, check_finite=False)
        var_chunk = 1.0 - np.sum(V * V, axis=0)
        var_chunk = np.maximum(var_chunk, 1e-12)
        p_chunk = ndtr(mu_chunk / (sigma * np.sqrt(var_chunk)))

        mu[start:stop] = mu_chunk
        var[start:stop] = var_chunk
        p[start:stop] = p_chunk

    return p, mu, var




def gp_sample_paths(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    ell: float,
    r: float,
    sigma: float,
    n_samples: int,
    seed: int,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Sample posterior paths at X_test from GP(mu_D, sigma^2 k_D)."""
    cho, alpha = gp_fit(X_train, y_train, ell, r, jitter=jitter)
    L = cho[0]
    lower = cho[1]
    if not lower:
        L = L.T

    Ks = exp_kernel(X_train, X_test, ell)
    mu = Ks.T @ alpha
    V = solve_triangular(L, Ks, lower=True, check_finite=False)
    Kss = exp_kernel(X_test, X_test, ell)
    cov = Kss - V.T @ V
    cov = 0.5 * (cov + cov.T)

    n_test = X_test.shape[0]
    eye = np.eye(n_test)
    chol = None
    current_jitter = jitter
    for _ in range(8):
        try:
            chol = np.linalg.cholesky(cov + current_jitter * eye)
            break
        except np.linalg.LinAlgError:
            current_jitter *= 10.0
    if chol is None:
        raise np.linalg.LinAlgError('Failed to compute Cholesky factor for posterior covariance in gp_sample_paths.')

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_test, n_samples))
    samples = mu[:, None] + sigma * (chol @ z)
    return samples


def binary_cross_entropy(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def iou_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def parse_grid(s: str) -> list[float]:
    vals = [float(v.strip()) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("Grid string must contain at least one numeric value.")
    return vals



# ---------------------------------------------------------------------
# DP-constrained hyperparameter search helpers
# ---------------------------------------------------------------------


def dp_epsilon_for_params(
    n: int,
    ell: float,
    r: float,
    sigma: float,
    M_Y: float,
    L: int,
    delta: float,
    alpha_grid_size: int,
) -> float:
    """
    Epsilon for one parameter triple using exactly the same helper as the
    synthetic 2D experiment:

        epsilon_for_delta_exp_kernel_unit_square(
            n=n, ell=ell, r=r, sigma=sigma, M_Y=M_Y, L=L,
            delta=delta, grid_size=alpha_grid_size,
        )

    Coordinates are normalised to [0,1]^2 before GP fitting, so this applies
    the same unit-square 2D exponential-kernel bound as in
    run_2d_excursion_gp_private_sigmoid_smoothed.py.
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


def make_candidate_grid(ell_grid, r_grid, sigma_grid) -> list[tuple[float, float, float]]:
    return [
        (float(ell), float(r), float(sigma))
        for ell in ell_grid
        for r in r_grid
        for sigma in sigma_grid
    ]


def make_refined_candidates_from_rows(
    rows: list[dict],
    ell_bounds=(0.025, 1.0),
    r_bounds=(0.25, 10.0),
    sigma_bounds=(0.02, 1.20),
    factors=(0.75, 0.85, 0.925, 1.0, 1.075, 1.15, 1.25),
    decimals: int = 8,
) -> list[tuple[float, float, float]]:
    """Multiplicative local refinement around the top coarse private candidates."""
    candidates = set()
    for row in rows:
        ell_vals = np.clip(np.array(factors) * float(row["ell"]), *ell_bounds)
        r_vals = np.clip(np.array(factors) * float(row["r"]), *r_bounds)
        sigma_vals = np.clip(np.array(factors) * float(row["sigma"]), *sigma_bounds)

        ell_vals = np.unique(np.round(ell_vals, decimals))
        r_vals = np.unique(np.round(r_vals, decimals))
        sigma_vals = np.unique(np.round(sigma_vals, decimals))

        for ell in ell_vals:
            for r in r_vals:
                for sigma in sigma_vals:
                    candidates.add((float(ell), float(r), float(sigma)))
    return sorted(candidates)


def evaluate_candidates_bce_on_split(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    candidates: list[tuple[float, float, float]],
    chunk_size: int,
    n_for_epsilon: int,
    M_Y: float,
    L: int,
    delta: float,
    alpha_grid_size: int,
    epsilon0: Optional[float] = None,
    verbose: bool = True,
    label: str = "grid",
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate candidate triples by validation BCE. If epsilon0 is not None,
    discard triples with epsilon >= epsilon0. The epsilon calculation is the
    same unit-square 2D exponential-kernel formula used in the synthetic script.
    """
    Xtr, ytr = X[train_idx], y[train_idx]
    Xva = X[val_idx]
    lab_va = labels[val_idx]

    grouped: dict[tuple[float, float], list[float]] = {}
    for ell, r, sigma in candidates:
        grouped.setdefault((float(ell), float(r)), []).append(float(sigma))

    rows = []
    best = None
    n_groups = len(grouped)

    for counter, ((ell, r), sigmas) in enumerate(grouped.items(), start=1):
        sigmas = sorted(set(sigmas))

        eps_by_sigma = {}
        feasible_sigmas = []
        for sigma in sigmas:
            if epsilon0 is None:
                # Unconstrained/public search: no need to spend time evaluating epsilon.
                eps_val = np.nan
                feasible = True
            else:
                eps_val = dp_epsilon_for_params(
                    n=n_for_epsilon,
                    ell=ell,
                    r=r,
                    sigma=sigma,
                    M_Y=M_Y,
                    L=L,
                    delta=delta,
                    alpha_grid_size=alpha_grid_size,
                )
                feasible = eps_val < epsilon0
            eps_by_sigma[sigma] = eps_val
            if feasible:
                feasible_sigmas.append(sigma)

        if not feasible_sigmas:
            if verbose:
                print(f"[{label} {counter:>3}/{n_groups}] ell={ell:.4g}, r={r:.4g}: no epsilon-feasible sigmas")
            continue

        try:
            cho, alpha = gp_fit(Xtr, ytr, ell, r)
            L_chol = cho[0]
            if not cho[1]:
                L_chol = L_chol.T
            Ks = exp_kernel(Xtr, Xva, ell)
            mu_va = Ks.T @ alpha
            V = solve_triangular(L_chol, Ks, lower=True, check_finite=False)
            var_va = np.maximum(1.0 - np.sum(V * V, axis=0), 1e-12)
        except Exception as exc:
            print(f"Skipping ell={ell:g}, r={r:g}: {exc}")
            continue

        for sigma in feasible_sigmas:
            p_va = ndtr(mu_va / (sigma * np.sqrt(var_va)))
            bce = binary_cross_entropy(lab_va, p_va)
            pred_05 = p_va >= 0.5
            iou_05 = iou_score(lab_va, pred_05)
            row = {
                "ell": float(ell),
                "r": float(r),
                "sigma": float(sigma),
                "epsilon": float(eps_by_sigma[sigma]),
                "val_bce": float(bce),
                "val_iou_at_0p5": float(iou_05),
            }
            rows.append(row)
            if best is None or row["val_bce"] < best["val_bce"]:
                best = row.copy()

        if verbose:
            if best is None:
                best_msg = "no feasible candidate yet"
            else:
                best_msg = (
                    f"best=(ell={best['ell']:.4g}, r={best['r']:.4g}, "
                    f"sigma={best['sigma']:.4g}, eps={best['epsilon']:.4g}), "
                    f"val BCE={best['val_bce']:.6f}"
                )
            print(f"[{label} {counter:>3}/{n_groups}] ell={ell:.4g}, r={r:.4g}, {best_msg}")

    if best is None:
        raise RuntimeError(
            "No epsilon-feasible parameter setting found. Try increasing --epsilon0, "
            "changing --response-scale if clipping is too aggressive, or expanding the private grids."
        )

    results = pd.DataFrame(rows).sort_values("val_bce").reset_index(drop=True)
    return best, results


def private_coarse_to_refined_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_for_epsilon: int,
    M_Y: float,
    L: int,
    delta: float,
    epsilon0: float,
    alpha_grid_size: int,
    chunk_size: int,
    coarse_ell_grid: list[float],
    coarse_r_grid: list[float],
    coarse_sigma_grid: list[float],
    top_k: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Private hyperparameter search mirroring the synthetic script:
      1. coarse grid over private candidates satisfying epsilon < epsilon0;
      2. take top-k feasible candidates by validation BCE;
      3. one multiplicative refinement round around those candidates.
    """
    coarse_candidates = make_candidate_grid(coarse_ell_grid, coarse_r_grid, coarse_sigma_grid)

    print("Running private coarse grid search...")
    coarse_best, coarse_results = evaluate_candidates_bce_on_split(
        X=X,
        y=y,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        candidates=coarse_candidates,
        chunk_size=chunk_size,
        n_for_epsilon=n_for_epsilon,
        M_Y=M_Y,
        L=L,
        delta=delta,
        alpha_grid_size=alpha_grid_size,
        epsilon0=epsilon0,
        verbose=verbose,
        label="private coarse",
    )

    top_rows = coarse_results.head(top_k).to_dict(orient="records")
    refined_candidates = make_refined_candidates_from_rows(top_rows)

    print(f"Running private refined grid search around top-{top_k} coarse candidates...")
    refined_best, refined_results = evaluate_candidates_bce_on_split(
        X=X,
        y=y,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        candidates=refined_candidates,
        chunk_size=chunk_size,
        n_for_epsilon=n_for_epsilon,
        M_Y=M_Y,
        L=L,
        delta=delta,
        alpha_grid_size=alpha_grid_size,
        epsilon0=epsilon0,
        verbose=verbose,
        label="private refined",
    )

    return {
        "coarse_best": coarse_best,
        "coarse_results": coarse_results,
        "refined_best": refined_best,
        "refined_results": refined_results,
        "top_coarse": top_rows,
    }

def validation_split(n: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * n)))
    n_val = min(n_val, n - 1)
    val_idx = np.sort(idx[:n_val])
    train_idx = np.sort(idx[n_val:])
    return train_idx, val_idx


def grid_search_bce(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    ell_grid: list[float],
    r_grid: list[float],
    sigma_grid: list[float],
    val_frac: float,
    seed: int,
    chunk_size: int,
) -> Tuple[dict, pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
    train_idx, val_idx = validation_split(len(y), val_frac, seed)
    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]
    lab_va = labels[val_idx]

    rows = []
    best = None

    print(f"Validation split: {len(train_idx):,} train hexagons, {len(val_idx):,} validation hexagons")
    for ell in ell_grid:
        for r in r_grid:
            try:
                # Compute mean/variance once for fixed ell,r; sigma only rescales probabilities.
                cho, alpha = gp_fit(Xtr, ytr, ell, r)
                L = cho[0]
                if not cho[1]:
                    L = L.T
                Ks = exp_kernel(Xtr, Xva, ell)
                mu_va = Ks.T @ alpha
                V = solve_triangular(L, Ks, lower=True, check_finite=False)
                var_va = np.maximum(1.0 - np.sum(V * V, axis=0), 1e-12)
            except Exception as exc:
                print(f"Skipping ell={ell:g}, r={r:g}: {exc}")
                continue

            for sigma in sigma_grid:
                p_va = ndtr(mu_va / (sigma * np.sqrt(var_va)))
                bce = binary_cross_entropy(lab_va, p_va)
                pred_05 = p_va >= 0.5
                iou_05 = iou_score(lab_va, pred_05)
                row = {
                    "ell": ell,
                    "r": r,
                    "sigma": sigma,
                    "val_bce": bce,
                    "val_iou_at_0p5": iou_05,
                }
                rows.append(row)
                if best is None or bce < best["val_bce"]:
                    best = row.copy()

    if best is None:
        raise RuntimeError("Grid search failed for all hyperparameter combinations.")

    results = pd.DataFrame(rows).sort_values("val_bce").reset_index(drop=True)
    return best, results, (train_idx, val_idx)


def choose_C_by_validation_iou(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    ell: float,
    r: float,
    sigma: float,
    C_grid: np.ndarray,
) -> Tuple[float, float, pd.DataFrame]:
    p_va, _, _ = gp_predict_prob(
        X[train_idx], y[train_idx], X[val_idx], ell, r, sigma, chunk_size=max(1000, len(val_idx))
    )
    lab_va = labels[val_idx]
    rows = []
    for C in C_grid:
        pred = p_va >= C
        rows.append({
            "C": float(C),
            "val_iou": iou_score(lab_va, pred),
            "val_bce_fixed_hyperparams": binary_cross_entropy(lab_va, p_va),
            "pred_positive_fraction": float(pred.mean()),
        })
    df = pd.DataFrame(rows).sort_values(["val_iou", "C"], ascending=[False, True]).reset_index(drop=True)
    return float(df.loc[0, "C"]), float(df.loc[0, "val_iou"]), df


def make_prediction_grid(X_orig: np.ndarray, grid_size: int, margin_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, ymin = X_orig.min(axis=0)
    xmax, ymax = X_orig.max(axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= margin_frac * dx
    xmax += margin_frac * dx
    ymin -= margin_frac * dy
    ymax += margin_frac * dy

    gx = np.linspace(xmin, xmax, grid_size)
    gy = np.linspace(ymin, ymax, grid_size)
    XX, YY = np.meshgrid(gx, gy, indexing="xy")
    Xgrid = np.column_stack([XX.ravel(), YY.ravel()])
    return XX, YY, Xgrid


def estimate_hex_support_radius(
    X_hex_orig: np.ndarray,
    user_radius: Optional[float],
    radius_factor: float,
) -> float:
    """
    Radius used to mask prediction/path grids to the observed hexagon support.

    If user_radius is provided, it is interpreted in the original coordinate
    units (e.g. metres for British National Grid). Otherwise we estimate a
    typical hex-centre spacing by the median nearest-neighbour distance between
    non-empty hexagon centres, and multiply by radius_factor.
    """
    if user_radius is not None:
        if user_radius <= 0:
            raise ValueError("--support-radius must be positive when supplied.")
        return float(user_radius)

    if radius_factor <= 0:
        raise ValueError("--support-radius-factor must be positive.")

    if len(X_hex_orig) < 2:
        raise ValueError("At least two hexagon centres are needed to estimate the support radius.")

    tree = cKDTree(X_hex_orig)
    dists, _ = tree.query(X_hex_orig, k=2)
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if len(nn) == 0:
        raise ValueError("Could not estimate nonzero nearest-neighbour hexagon spacing.")
    return float(radius_factor * np.median(nn))


def mask_grid_to_observed_hex_support(
    X_grid_orig: np.ndarray,
    X_hex_orig: np.ndarray,
    support_radius: float,
) -> np.ndarray:
    """Return True for grid points close enough to at least one non-empty hexagon."""
    tree = cKDTree(X_hex_orig)
    dists, _ = tree.query(X_grid_orig, k=1)
    return dists <= support_radius


def apply_support_mask_2d(Z: np.ndarray, mask_flat: np.ndarray) -> np.ndarray:
    """Set values outside the observed-hexagon support to NaN for plotting/contours."""
    out = np.asarray(Z, dtype=float).copy()
    mask = mask_flat.reshape(out.shape)
    out[~mask] = np.nan
    return out


def apply_support_mask_3d(Z: np.ndarray, mask_flat: np.ndarray) -> np.ndarray:
    """Set values outside support to NaN for an array of path grids (..., n_paths)."""
    out = np.asarray(Z, dtype=float).copy()
    mask = mask_flat.reshape(out.shape[:2])
    out[~mask, :] = np.nan
    return out


def smooth_path_grids(path_grids: np.ndarray, sigma: float) -> np.ndarray:
    """
    Smooth posterior sample-path grids before extracting zero-contours.

    Parameters
    ----------
    path_grids:
        Array of shape (ny, nx, n_paths).
    sigma:
        Gaussian smoothing bandwidth in grid-cell units. Use sigma=0 to disable.

    Notes
    -----
    Smoothing is applied before the observed-support mask is imposed, so the
    Gaussian filter does not interact with NaNs. This is a visual post-processing
    step for the displayed path-boundary contours only; it does not change the
    fitted GP, the probability boundary, the hyperparameters, or the DP accounting.
    """
    if sigma <= 0:
        return np.asarray(path_grids, dtype=float).copy()
    out = np.asarray(path_grids, dtype=float).copy()
    for j in range(out.shape[-1]):
        out[:, :, j] = gaussian_filter(out[:, :, j], sigma=float(sigma), mode="nearest")
    return out


def extract_contours(XX: np.ndarray, YY: np.ndarray, Z: np.ndarray, level: float) -> list[pd.DataFrame]:
    fig, ax = plt.subplots()
    Z_plot = np.ma.masked_invalid(Z)
    if Z_plot.count() == 0 or not (np.nanmin(Z) < level < np.nanmax(Z)):
        plt.close(fig)
        return []
    cs = ax.contour(XX, YY, Z_plot, levels=[level])
    contour_dfs = []
    # Matplotlib exposes all contour segments through allsegs.
    for seg_id, seg in enumerate(cs.allsegs[0]):
        if len(seg) == 0:
            continue
        contour_dfs.append(pd.DataFrame({
            "segment": seg_id,
            "x": seg[:, 0],
            "y": seg[:, 1],
        }))
    plt.close(fig)
    return contour_dfs



def plot_minimal_comparison(
    hex_df: pd.DataFrame,
    XX: np.ndarray,
    YY: np.ndarray,
    P_public: np.ndarray,
    C_public: float,
    threshold: float,
    coord_type: str,
    out_figure: str,
    title: str,
    sample_XX: Optional[np.ndarray] = None,
    sample_YY: Optional[np.ndarray] = None,
    sample_paths: Optional[np.ndarray] = None,
    sample_level: float = 0.0,
    hex_plot_gridsize: int = 40,
) -> None:
    """
    Minimal paper-style plot:
      - greyscale observed hexagon median log prices;
      - the non-private posterior-probability excursion boundary p_D(x)=C;
      - optional single posterior sample-path boundaries f_D^{(j)}(x)=0.

    The posterior probability field itself is not plotted.
    """
    fig, ax = plt.subplots(figsize=(8.4, 8.0))

    hb = ax.hexbin(
        hex_df["x"].to_numpy(),
        hex_df["y"].to_numpy(),
        C=hex_df["median_log_price"].to_numpy(),
        reduce_C_function=np.mean,
        gridsize=hex_plot_gridsize,
        mincnt=1,
        cmap="Greys",
        alpha=0.90,
        linewidths=0.0,
    )
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("median log sale price")

    P_plot = np.ma.masked_invalid(P_public)
    if P_plot.count() > 0 and np.nanmin(P_public) < C_public < np.nanmax(P_public):
        ax.contour(
            XX,
            YY,
            P_plot,
            levels=[C_public],
            colors=["black"],
            linewidths=2.4,
        )

    if sample_paths is not None and sample_XX is not None and sample_YY is not None:
        path_colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        n_paths = sample_paths.shape[-1]
        for j in range(n_paths):
            path_grid = np.ma.masked_invalid(sample_paths[:, :, j])
            if path_grid.count() == 0:
                continue
            path_min = float(np.nanmin(sample_paths[:, :, j]))
            path_max = float(np.nanmax(sample_paths[:, :, j]))
            if path_min < sample_level < path_max:
                ax.contour(
                    sample_XX,
                    sample_YY,
                    path_grid,
                    levels=[sample_level],
                    colors=[path_colors[j % len(path_colors)]],
                    linewidths=1.7,
                    linestyles="--",
                )

    legend_handles = [
        Line2D([0], [0], color="black", lw=2.4, label=r"non-private boundary"),
    ]
    if sample_paths is not None and sample_XX is not None and sample_YY is not None:
        legend_handles.append(
            Line2D([0], [0], color="tab:orange", lw=1.7, ls="--", label="private 1-path boundaries")
        )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    if coord_type == "bng":
        ax.set_xlabel("British National Grid easting", fontsize=15)
        ax.set_ylabel("British National Grid northing", fontsize=15)
    else:
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    ax.legend(handles=legend_handles, loc="best", fontsize=12)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_figure, dpi=250, bbox_inches="tight")
    print(f"Saved minimal comparison figure to {out_figure}")
    plt.show()



def _get_nested(d: dict, path: list[str]):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _first_present(d: dict, paths: list[list[str]], name: str):
    for path in paths:
        val = _get_nested(d, path)
        if val is not None:
            return val
    raise KeyError(
        f"Could not find {name} in hyperparameter JSON. Tried paths: "
        + ", ".join(".".join(p) for p in paths)
    )


def load_hyperparameters_from_summary(path: str | Path) -> dict:
    """
    Read public/private GP hyperparameters and probability thresholds from a
    previous london_hexbin_gp_summary.json file.

    Expected keys are those produced by fit_hexbin_gp_excursion_london_private_*.py:
      public_best_hyperparameters_by_bce: {ell, r, sigma, ...}
      public_selected_probability_threshold: {C, ...}
      private_best_hyperparameters_by_bce: {ell, r, sigma, epsilon, ...}
      private_selected_probability_threshold: {C, ...}

    The loader also accepts the older nested format used in some synthetic runs.
    """
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    public_hp = _first_present(
        summary,
        [
            ["public_best_hyperparameters_by_bce"],
            ["public", "best_hyperparameters"],
            ["public", "best_hyperparameters_by_bce"],
        ],
        "public hyperparameters",
    )
    public_C = _first_present(
        summary,
        [
            ["public_selected_probability_threshold", "C"],
            ["public", "population_threshold", "C"],
            ["public", "selected_probability_threshold", "C"],
        ],
        "public probability threshold C",
    )

    private_hp = _first_present(
        summary,
        [
            ["private_best_hyperparameters_by_bce"],
            ["private", "best_hyperparameters"],
            ["private", "refined_best"],
        ],
        "private hyperparameters",
    )
    private_C = _first_present(
        summary,
        [
            ["private_selected_probability_threshold", "C"],
            ["private", "population_threshold", "C"],
            ["private", "selected_probability_threshold", "C"],
        ],
        "private probability threshold C",
    )

    def hp_to_float_dict(hp, label):
        out = {}
        for key in ["ell", "r", "sigma"]:
            if key not in hp:
                raise KeyError(f"Missing {key!r} in {label} hyperparameters.")
            out[key] = float(hp[key])
        if "epsilon" in hp and hp["epsilon"] is not None:
            out["epsilon"] = float(hp["epsilon"])
        return out

    loaded = {
        "summary_raw": summary,
        "public_hp": hp_to_float_dict(public_hp, "public"),
        "public_C": float(public_C),
        "private_hp": hp_to_float_dict(private_hp, "private"),
        "private_C": float(private_C),
    }

    # Optional experiment settings stored in the previous summary.
    for key in [
        "segment",
        "threshold",
        "gridsize",
        "mincnt",
        "response_scale_B",
        "response_bound_M_Y",
        "support_radius",
        "support_radius_factor",
    ]:
        if key in summary:
            loaded[key] = summary[key]

    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate the minimal London hexbin GP excursion plot using "
            "hyperparameters and probability thresholds read from a previous "
            "london_hexbin_gp_summary.json, with no grid search."
        )
    )
    parser.add_argument("--ppd", required=True, help="Path to pp-2018.csv Price Paid Data file.")
    parser.add_argument("--postcode-lookup", required=True, help="Path to ONSPD or Code-Point Open CSV with coordinates.")
    parser.add_argument(
        "--hyperparams-json",
        default="london_hexbin_gp_summary.json",
        help="JSON summary containing public/private hyperparameters and C thresholds.",
    )
    parser.add_argument(
        "--segment",
        choices=["existing-leasehold-flats", "all-standard-residential"],
        default=None,
        help="Override segment stored in the JSON. Default: use JSON value or existing-leasehold-flats.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Override log-price threshold stored in the JSON.")
    parser.add_argument("--gridsize", type=int, default=None, help="Override hexbin gridsize stored in the JSON.")
    parser.add_argument("--mincnt", type=int, default=None, help="Override minimum transactions per non-empty hexagon.")
    parser.add_argument(
        "--response-scale",
        type=float,
        default=None,
        help="Override symmetric clipping scale B stored in the JSON.",
    )
    parser.add_argument("--pred-grid-size", type=int, default=160, help="Resolution of prediction grid for the public boundary.")
    parser.add_argument("--path-grid-size", type=int, default=60, help="Resolution of grid for posterior sample-path boundaries.")
    parser.add_argument("--n-sample-paths", type=int, default=3, help="Number of private posterior sample-path boundaries to overlay.")
    parser.add_argument("--path-seed", type=int, default=123, help="Random seed for private posterior path sampling.")
    parser.add_argument(
        "--smooth-path-sigma",
        type=float,
        default=1.0,
        help=(
            "Gaussian smoothing bandwidth, in path-grid cells, applied to posterior "
            "sample paths before plotting zero-contour boundaries. Use 0 to disable."
        ),
    )
    parser.add_argument("--pred-margin-frac", type=float, default=0.02)
    parser.add_argument(
        "--support-radius",
        type=float,
        default=None,
        help=(
            "Radius in original coordinate units used to mask posterior fields/path "
            "contours to observed hexagon support. If omitted, uses JSON value if "
            "available, otherwise estimates it from hex-centre spacing."
        ),
    )
    parser.add_argument(
        "--support-radius-factor",
        type=float,
        default=None,
        help="Multiplier for automatically estimated support radius. Default: JSON value or 1.35.",
    )
    parser.add_argument(
        "--no-mask-outside-hex-support",
        action="store_true",
        help="Disable masking of probability fields/path contours outside observed hexagon support.",
    )
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--out-figure", default="london_hexbin_gp_minimal_from_json.png")
    parser.add_argument("--out-boundary", default="london_hexbin_gp_boundary_from_json.csv")
    parser.add_argument("--out-private-boundary", default="london_hexbin_gp_private_boundary_from_json.csv")
    parser.add_argument("--out-hexagons", default="london_hexbin_gp_hexagons_from_json.csv")
    parser.add_argument("--out-summary", default="london_hexbin_gp_from_json_summary.json")
    parser.add_argument("--no-show", action="store_true", help="Use a non-interactive backend and do not display the figure.")
    args = parser.parse_args()

    if args.no_show:
        plt.switch_backend("Agg")

    if args.smooth_path_sigma < 0:
        raise ValueError("--smooth-path-sigma must be nonnegative.")

    loaded = load_hyperparameters_from_summary(args.hyperparams_json)
    summary_raw = loaded["summary_raw"]

    segment = args.segment or loaded.get("segment") or "existing-leasehold-flats"
    threshold = float(args.threshold if args.threshold is not None else loaded.get("threshold", 13.5))
    gridsize = int(args.gridsize if args.gridsize is not None else loaded.get("gridsize", 40))
    mincnt = int(args.mincnt if args.mincnt is not None else loaded.get("mincnt", 1))
    response_scale = float(
        args.response_scale if args.response_scale is not None else loaded.get("response_scale_B", 1.0)
    )
    support_radius_factor = float(
        args.support_radius_factor
        if args.support_radius_factor is not None
        else loaded.get("support_radius_factor", 1.35)
    )

    # If --support-radius was not supplied, reuse a stored radius if present;
    # otherwise estimate it later from observed hexagon spacing.
    support_radius_arg = args.support_radius
    if support_radius_arg is None and loaded.get("support_radius") is not None:
        support_radius_arg = float(loaded["support_radius"])

    public_hp = loaded["public_hp"]
    private_hp = loaded["private_hp"]
    C_public = float(loaded["public_C"])
    C_private = float(loaded["private_C"])
    private_epsilon = private_hp.get("epsilon", None)

    print(f"Loaded hyperparameters from {args.hyperparams_json}")
    print(
        "  public:  "
        f"ell={public_hp['ell']:.6g}, r={public_hp['r']:.6g}, "
        f"sigma={public_hp['sigma']:.6g}, C={C_public:.6g}"
    )
    eps_msg = "" if private_epsilon is None else f", epsilon={private_epsilon:.6g}"
    print(
        "  private: "
        f"ell={private_hp['ell']:.6g}, r={private_hp['r']:.6g}, "
        f"sigma={private_hp['sigma']:.6g}, C={C_private:.6g}{eps_msg}"
    )
    print(
        f"Using segment={segment}, threshold={threshold:g}, gridsize={gridsize}, "
        f"mincnt={mincnt}, response_scale B={response_scale:g}"
    )

    london = read_ppd_london(args.ppd, segment=segment)
    segment_desc = SEGMENT_DESCRIPTIONS[segment]
    print(f"Selected segment: {segment_desc}")
    print(f"Selected transactions before geocoding: {len(london):,}")
    print(f"Distinct transaction postcodes before geocoding: {london['postcode_key'].nunique():,}")

    lookup, _, _, coord_type = read_postcode_lookup(args.postcode_lookup)
    mapped = join_coordinates(london, lookup)
    print(f"Matched transactions: {len(mapped):,}")
    print(f"Distinct matched postcodes: {mapped['postcode_key'].nunique():,}")

    hex_df = make_hexbin_dataset(mapped, gridsize=gridsize, mincnt=mincnt)
    hex_df["response"] = hex_df["median_log_price"] - threshold
    hex_df["label"] = (hex_df["response"] > 0.0).astype(int)
    pos_frac = float(hex_df["label"].mean())
    print(f"Non-empty training hexagons: {len(hex_df):,}")
    print(f"Positive hexagons above threshold {threshold:g}: {hex_df['label'].sum():,} ({pos_frac:.3f})")
    print(f"Median of hexagon median log prices: {hex_df['median_log_price'].median():.4f}")

    X_orig = hex_df[["x", "y"]].to_numpy(dtype=float)
    X, bounds = normalise_xy(X_orig)

    if response_scale <= 0:
        raise ValueError("--response-scale must be positive.")
    y_raw = hex_df["response"].to_numpy(dtype=float)
    y = np.clip(y_raw, -response_scale, response_scale) / response_scale
    labels = hex_df["label"].to_numpy(dtype=int)
    clipped_fraction = float(np.mean(np.abs(y_raw) > response_scale))
    max_abs_raw = float(np.max(np.abs(y_raw)))
    max_abs_used = float(np.max(np.abs(y)))
    print(
        f"Centred response max |y_raw|={max_abs_raw:.4g}; "
        f"using y=clip(y_raw,-B,B)/B with B={response_scale:g}, so |y|<=1. "
        f"Clipped fraction: {clipped_fraction:.3f}"
    )

    hex_df["response_raw"] = y_raw
    hex_df["response_scaled"] = y
    hex_df["response_used"] = y
    hex_df["response_scale_B"] = response_scale
    hex_df.to_csv(args.out_hexagons, index=False)
    print(f"Saved hexagon dataset to {args.out_hexagons}")

    # ------------------------------------------------------------------
    # Refit using the hyperparameters read from JSON; no grid search and no
    # validation C-selection are performed here.
    # ------------------------------------------------------------------
    XX, YY, Xgrid_orig = make_prediction_grid(X_orig, args.pred_grid_size, args.pred_margin_frac)
    Xgrid, _ = normalise_xy(Xgrid_orig, bounds=bounds)

    p_public_grid, _, _ = gp_predict_prob(
        X_train=X,
        y_train=y,
        X_test=Xgrid,
        ell=public_hp["ell"],
        r=public_hp["r"],
        sigma=public_hp["sigma"],
        chunk_size=args.chunk_size,
    )
    P_public = p_public_grid.reshape(XX.shape)

    p_private_grid, _, _ = gp_predict_prob(
        X_train=X,
        y_train=y,
        X_test=Xgrid,
        ell=private_hp["ell"],
        r=private_hp["r"],
        sigma=private_hp["sigma"],
        chunk_size=args.chunk_size,
    )
    P_private = p_private_grid.reshape(XX.shape)

    support_radius = None
    support_mask = np.ones(Xgrid_orig.shape[0], dtype=bool)
    if not args.no_mask_outside_hex_support:
        support_radius = estimate_hex_support_radius(
            X_hex_orig=X_orig,
            user_radius=support_radius_arg,
            radius_factor=support_radius_factor,
        )
        support_mask = mask_grid_to_observed_hex_support(
            X_grid_orig=Xgrid_orig,
            X_hex_orig=X_orig,
            support_radius=support_radius,
        )
        P_public = apply_support_mask_2d(P_public, support_mask)
        P_private = apply_support_mask_2d(P_private, support_mask)
        print(
            "Masked posterior probability fields outside observed hexagon support: "
            f"radius={support_radius:.6g}, kept {support_mask.mean():.3f} of prediction-grid points"
        )

    XX_path, YY_path, Xpath_orig = make_prediction_grid(X_orig, args.path_grid_size, args.pred_margin_frac)
    Xpath, _ = normalise_xy(Xpath_orig, bounds=bounds)

    private_path_samples = gp_sample_paths(
        X_train=X,
        y_train=y,
        X_test=Xpath,
        ell=private_hp["ell"],
        r=private_hp["r"],
        sigma=private_hp["sigma"],
        n_samples=args.n_sample_paths,
        seed=args.path_seed,
    ).reshape(XX_path.shape + (args.n_sample_paths,))

    if args.smooth_path_sigma > 0:
        private_path_samples = smooth_path_grids(
            private_path_samples,
            sigma=args.smooth_path_sigma,
        )
        print(
            "Smoothed posterior sample paths before contouring: "
            f"Gaussian sigma={args.smooth_path_sigma:g} path-grid cells"
        )
    else:
        print("Posterior sample paths are not smoothed (--smooth-path-sigma=0).")

    path_support_mask = np.ones(Xpath_orig.shape[0], dtype=bool)
    if not args.no_mask_outside_hex_support:
        assert support_radius is not None
        path_support_mask = mask_grid_to_observed_hex_support(
            X_grid_orig=Xpath_orig,
            X_hex_orig=X_orig,
            support_radius=support_radius,
        )
        private_path_samples = apply_support_mask_3d(private_path_samples, path_support_mask)
        print(
            "Masked posterior sample paths outside observed hexagon support: "
            f"kept {path_support_mask.mean():.3f} of path-grid points"
        )

    contour_dfs = extract_contours(XX, YY, P_public, level=C_public)
    if contour_dfs:
        boundary_df = pd.concat(contour_dfs, ignore_index=True)
    else:
        boundary_df = pd.DataFrame(columns=["segment", "x", "y"])
    boundary_df.to_csv(args.out_boundary, index=False)
    print(f"Saved public boundary coordinates to {args.out_boundary}")

    private_contour_dfs = extract_contours(XX, YY, P_private, level=C_private)
    if private_contour_dfs:
        private_boundary_df = pd.concat(private_contour_dfs, ignore_index=True)
    else:
        private_boundary_df = pd.DataFrame(columns=["segment", "x", "y"])
    private_boundary_df.to_csv(args.out_private_boundary, index=False)
    print(f"Saved private boundary coordinates to {args.out_private_boundary}")

    eps_title = "" if private_epsilon is None else f"; private epsilon={private_epsilon:.3g}"
    minimal_title = (
        "Greater London leasehold flats prices 2018: excursion boundaries\n"
    )
    plot_minimal_comparison(
        hex_df=hex_df,
        XX=XX,
        YY=YY,
        P_public=P_public,
        C_public=C_public,
        threshold=threshold,
        coord_type=coord_type,
        out_figure=args.out_figure,
        title=minimal_title,
        sample_XX=XX_path,
        sample_YY=YY_path,
        sample_paths=private_path_samples,
        sample_level=0.0,
        hex_plot_gridsize=gridsize,
    )

    out_summary = {
        "source_hyperparams_json": str(args.hyperparams_json),
        "segment": segment,
        "segment_description": segment_desc,
        "threshold": threshold,
        "gridsize": gridsize,
        "mincnt": mincnt,
        "n_transactions_matched": int(len(mapped)),
        "n_hexagons": int(len(hex_df)),
        "n_positive_hexagons": int(hex_df["label"].sum()),
        "positive_fraction": pos_frac,
        "coordinate_type": coord_type,
        "normalisation_bounds": bounds,
        "response_scale_B": float(response_scale),
        "max_abs_raw_response": max_abs_raw,
        "max_abs_used_response": max_abs_used,
        "clipped_fraction": clipped_fraction,
        "public_hyperparameters_from_json": public_hp,
        "public_C_from_json": C_public,
        "private_hyperparameters_from_json": private_hp,
        "private_C_from_json": C_private,
        "n_sample_paths": int(args.n_sample_paths),
        "path_grid_size": int(args.path_grid_size),
        "path_seed": int(args.path_seed),
        "smooth_path_sigma": float(args.smooth_path_sigma),
        "mask_outside_hex_support": not args.no_mask_outside_hex_support,
        "support_radius": None if support_radius is None else float(support_radius),
        "support_radius_factor": float(support_radius_factor),
        "prediction_grid_kept_fraction": float(support_mask.mean()),
        "path_grid_kept_fraction": float(path_support_mask.mean()),
        "outputs": {
            "minimal_comparison_figure": args.out_figure,
            "public_boundary": args.out_boundary,
            "private_boundary": args.out_private_boundary,
            "hexagons": args.out_hexagons,
        },
        "source_summary_args": summary_raw.get("args", None),
    }
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2)
    print(f"Saved from-json summary to {args.out_summary}")


if __name__ == "__main__":
    main()
