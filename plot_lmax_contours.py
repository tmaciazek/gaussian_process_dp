"""Plot capped release counts using the tightened GP-posterior accountant.

The three panels use, respectively, the improved generic bounded-response
sensitivity, the halved fixed-RKHS-response sensitivity, and the specialized
one-dimensional exponential-kernel sensitivity and mean-term refinement.
All RDP evaluation and RDP-to-DP conversion is delegated to ``dp_utils.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects, ticker

import dp_utils as DP


ModelType = Literal["generic", "rkhs", "exp_1d"]
MODELS: tuple[ModelType, ...] = ("generic", "rkhs", "exp_1d")


def sensitivity_curves(
    *,
    rs: np.ndarray,
    n: int,
    kappa: float,
    M_Y: float,
    rkhs_norm: float,
    tolerance: float,
) -> dict[ModelType, np.ndarray]:
    """Precompute the three tightened sensitivities, which depend only on r."""
    curves: dict[ModelType, np.ndarray] = {
        model: np.empty(len(rs), dtype=float) for model in MODELS
    }

    for i, r_value in enumerate(rs):
        r = float(r_value)
        curves["generic"][i] = DP.delta_n_generic(
            n=n,
            r=r,
            kappa=kappa,
            M_Y=M_Y,
            tolerance=tolerance,
        )
        curves["rkhs"][i] = DP.delta_n_rkhs(
            n=n,
            r=r,
            kappa=kappa,
            rkhs_norm=rkhs_norm,
        )
        curves["exp_1d"][i] = DP.delta_n_exp_1d(
            n=n,
            r=r,
            kappa=kappa,
            M_Y=M_Y,
        )

    return curves


def log10_epsilon(
    *,
    L: int,
    n: int,
    r: float,
    kappa: float,
    sigma: float,
    eta: float,
    delta_dp: float,
    model: ModelType,
    M_Y: float,
    rkhs_norm: float | None,
    sensitivity: float,
    beta_safety: float,
    alpha_grid_size: int,
    xatol: float,
) -> float:
    """Return log10(epsilon_L) for a positive integer number of paths L."""
    if L < 1:
        raise ValueError("L must be a positive integer.")

    value = DP.log10_epsilon_for_delta(
        n=n,
        r=float(r),
        kappa=kappa,
        sigma=float(sigma),
        eta=eta,
        L=int(L),
        delta=delta_dp,
        model=model,
        M_Y=M_Y,
        rkhs_norm=rkhs_norm,
        sensitivity_override=sensitivity,
        beta_safety=beta_safety,
        grid_size=alpha_grid_size,
        xatol=xatol,
    )

    value = float(value)
    if not np.isfinite(value):
        raise FloatingPointError(f"Non-finite log10(epsilon) returned for L={L}.")
    return value


def find_lmax(
    *,
    epsilon_budget: float,
    L_cap: int,
    n: int,
    r: float,
    kappa: float,
    sigma: float,
    eta: float,
    delta_dp: float,
    model: ModelType,
    M_Y: float,
    rkhs_norm: float | None,
    sensitivity: float,
    beta_safety: float,
    alpha_grid_size: int,
    xatol: float,
) -> tuple[int, bool]:
    """
    Find the largest integer L <= L_cap satisfying epsilon_L <= epsilon_budget.

    Returns
    -------
    lmax:
        0 if even L=1 is infeasible; otherwise the largest feasible L found.
    capped:
        True when L=L_cap is feasible, so the true L_max may exceed L_cap.

    Notes
    -----
    The search assumes epsilon_L is nondecreasing in L, as implied by additive
    RDP composition followed by optimal conversion to (epsilon, delta)-DP.
    Exceptions from the epsilon routine are treated conservatively as infeasible.
    """
    if epsilon_budget <= 0:
        raise ValueError("epsilon_budget must be positive.")
    if L_cap < 1:
        raise ValueError("L_cap must be at least 1.")

    target = float(np.log10(epsilon_budget))

    def feasible(L: int) -> bool:
        try:
            return log10_epsilon(
                L=L,
                n=n,
                r=r,
                kappa=kappa,
                sigma=sigma,
                eta=eta,
                delta_dp=delta_dp,
                model=model,
                M_Y=M_Y,
                rkhs_norm=rkhs_norm,
                sensitivity=sensitivity,
                beta_safety=beta_safety,
                alpha_grid_size=alpha_grid_size,
                xatol=xatol,
            ) <= target
        except Exception:
            return False

    if not feasible(1):
        return 0, False

    if feasible(L_cap):
        return L_cap, True

    # Invariant: low is feasible and high is infeasible.
    low, high = 1, L_cap
    while low + 1 < high:
        mid = (low + high) // 2
        if feasible(mid):
            low = mid
        else:
            high = mid

    return low, False


def compute_lmax_grid(
    *,
    sigmas: np.ndarray,
    rs: np.ndarray,
    epsilon_budget: float,
    L_cap: int,
    n: int,
    kappa: float,
    eta: float,
    delta_dp: float,
    model: ModelType,
    M_Y: float,
    rkhs_norm: float | None,
    sensitivities: np.ndarray,
    beta_safety: float,
    alpha_grid_size: int,
    xatol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute L_max on the (r, sigma) grid.

    Returns integer-valued Lmax and a Boolean mask showing values censored at
    L_cap. Array convention: result[i, j] corresponds to rs[i], sigmas[j].
    """
    lmax = np.zeros((len(rs), len(sigmas)), dtype=np.int64)
    capped = np.zeros_like(lmax, dtype=bool)

    total = len(rs) * len(sigmas)
    count = 0

    for i, r in enumerate(rs):
        for j, sigma in enumerate(sigmas):
            count += 1
            value, hit_cap = find_lmax(
                epsilon_budget=epsilon_budget,
                L_cap=L_cap,
                n=n,
                r=float(r),
                kappa=kappa,
                sigma=float(sigma),
                eta=eta,
                delta_dp=delta_dp,
                model=model,
                M_Y=M_Y,
                rkhs_norm=rkhs_norm,
                sensitivity=float(sensitivities[i]),
                beta_safety=beta_safety,
                alpha_grid_size=alpha_grid_size,
                xatol=xatol,
            )
            lmax[i, j] = value
            capped[i, j] = hit_cap

        if (i + 1) % max(1, len(rs) // 10) == 0 or i + 1 == len(rs):
            print(f"{model}: finished {count}/{total} grid points")

    return lmax, capped


def compute_log10_epsilon_grid(
    *,
    sigmas: np.ndarray,
    rs: np.ndarray,
    L: int,
    n: int,
    kappa: float,
    eta: float,
    delta_dp: float,
    model: ModelType,
    M_Y: float,
    rkhs_norm: float | None,
    sensitivities: np.ndarray,
    beta_safety: float,
    alpha_grid_size: int,
    xatol: float,
) -> np.ndarray:
    """
    Compute log10(epsilon_L) on the (r, sigma) grid for a fixed L.

    These continuous fixed-L grids are used for smooth contour boundaries.
    Array convention: result[i, j] corresponds to rs[i], sigmas[j].
    """
    Z = np.full((len(rs), len(sigmas)), np.nan, dtype=float)

    for i, r in enumerate(rs):
        for j, sigma in enumerate(sigmas):
            try:
                Z[i, j] = log10_epsilon(
                    L=L,
                    n=n,
                    r=float(r),
                    kappa=kappa,
                    sigma=float(sigma),
                    eta=eta,
                    delta_dp=delta_dp,
                    model=model,
                    M_Y=M_Y,
                    rkhs_norm=rkhs_norm,
                    sensitivity=float(sensitivities[i]),
                    beta_safety=beta_safety,
                    alpha_grid_size=alpha_grid_size,
                    xatol=xatol,
                )
            except Exception:
                Z[i, j] = np.nan

    return Z



def positive_log10_lmax(lmax: np.ndarray) -> np.ma.MaskedArray:
    """Return log10(L_max), masking points for which L_max=0."""
    values = np.full(lmax.shape, np.nan, dtype=float)
    positive = lmax > 0
    values[positive] = np.log10(lmax[positive])
    return np.ma.masked_invalid(values)



def choose_contour_label_position(
    contours,
    *,
    ax: plt.Axes,
    x_fraction: float = 0.55,
    edge_margin: float = 0.08,
) -> tuple[float, float] | None:
    """
    Choose an interior label position on a one-level contour set.

    Selection is performed in log10 coordinates so that placement is natural
    on the logarithmic axes. Points close to the axes are excluded.
    """
    if not getattr(contours, "allsegs", None):
        return None
    if not contours.allsegs or not contours.allsegs[0]:
        return None

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    lx_min, lx_max = np.log10([x_min, x_max])
    ly_min, ly_max = np.log10([y_min, y_max])

    lx_lo = lx_min + edge_margin * (lx_max - lx_min)
    lx_hi = lx_max - edge_margin * (lx_max - lx_min)
    ly_lo = ly_min + edge_margin * (ly_max - ly_min)
    ly_hi = ly_max - edge_margin * (ly_max - ly_min)

    target_lx = lx_min + x_fraction * (lx_max - lx_min)
    target_ly = 0.5 * (ly_min + ly_max)

    best_point = None
    best_score = np.inf

    for segment in contours.allsegs[0]:
        if len(segment) == 0:
            continue

        x = segment[:, 0]
        y = segment[:, 1]

        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x > 0)
            & (y > 0)
        )
        if not np.any(valid):
            continue

        x = x[valid]
        y = y[valid]
        lx = np.log10(x)
        ly = np.log10(y)

        interior = (
            (lx >= lx_lo)
            & (lx <= lx_hi)
            & (ly >= ly_lo)
            & (ly <= ly_hi)
        )
        if np.any(interior):
            x = x[interior]
            y = y[interior]
            lx = lx[interior]
            ly = ly[interior]

        # Prioritize the requested horizontal position, with a weak vertical
        # centering term to avoid selecting awkward end points.
        scores = np.abs(lx - target_lx) + 0.08 * np.abs(ly - target_ly)
        idx = int(np.argmin(scores))
        if scores[idx] < best_score:
            best_score = float(scores[idx])
            best_point = (float(x[idx]), float(y[idx]))

    return best_point


def add_lmax_panel(
    *,
    ax: plt.Axes,
    S: np.ndarray,
    R: np.ndarray,
    lmax: np.ndarray,
    title: str,
    L_cap: int,
    contour_thresholds: list[int],
    epsilon_grids: dict[int, np.ndarray],
    epsilon_budget: float,
    filled_levels: np.ndarray,
    cmap,
):
    """
    Draw one L_max panel.

    The filled field is based on the integer-valued L_max grid. Labelled
    boundaries are drawn from the smooth fixed-L functions
    epsilon_L(r, sigma) = epsilon_budget.
    """
    Z = positive_log10_lmax(lmax)
    target = float(np.log10(epsilon_budget))

    ax.set_xscale("log")
    ax.set_yscale("log")

    if np.all(Z.mask):
        ax.text(
            0.5,
            0.5,
            "No feasible one-path release",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
        )
        contourf = None
    else:
        contourf = ax.contourf(
            S,
            R,
            Z,
            levels=filled_levels,
            extend="max",
            cmap=cmap,
            zorder=1,
        )

    # Draw the infeasible hatching first so that the L_max=1 contour and its
    # label remain visible on top.
    Z_one = epsilon_grids.get(1)
    if Z_one is not None and np.any(np.isfinite(Z_one)):
        finite_values = Z_one[np.isfinite(Z_one)]
        z_max = float(np.max(finite_values))
        if z_max > target:
            upper = max(z_max, target) + 1e-12
            ax.contourf(
                S,
                R,
                Z_one,
                levels=[target, upper],
                colors="none",
                hatches=["////"],
                zorder=2,
            )

    # Smooth dotted boundary of the region censored at L_cap.
    Z_cap = epsilon_grids.get(L_cap)
    if Z_cap is not None and np.any(np.isfinite(Z_cap)):
        z_min = float(np.nanmin(Z_cap))
        z_max = float(np.nanmax(Z_cap))
        if z_min <= target <= z_max:
            ax.contour(
                S,
                R,
                Z_cap,
                levels=[target],
                linewidths=1.2,
                linestyles="dotted",
                colors="black",
                zorder=3,
            )

    usable_thresholds = sorted(
        {
            int(value)
            for value in contour_thresholds
            if 1 <= int(value) <= L_cap and int(value) in epsilon_grids
        }
    )

    # Give successive contours slightly different horizontal label positions
    # to prevent labels from forming a single crowded column.
    n_thresholds = max(1, len(usable_thresholds))
    label_fractions = np.linspace(0.47, 0.68, n_thresholds)

    for index, L0 in enumerate(usable_thresholds):
        Z_eps = epsilon_grids[L0]
        finite = np.isfinite(Z_eps)
        if not np.any(finite):
            continue

        z_min = float(np.nanmin(Z_eps))
        z_max = float(np.nanmax(Z_eps))
        if not (z_min <= target <= z_max):
            continue

        is_one_path = L0 == 1
        linewidth = 2.2 if is_one_path else 1.4
        fontsize = 13 if is_one_path else 12

        contours = ax.contour(
            S,
            R,
            Z_eps,
            levels=[target],
            linewidths=linewidth,
            colors="red",
            zorder=5 if is_one_path else 4,
        )

        manual_position = choose_contour_label_position(
            contours,
            ax=ax,
            x_fraction=float(label_fractions[index]),
            edge_margin=0.10 if is_one_path else 0.08,
        )

        if manual_position is None:
            continue

        labels = ax.clabel(
            contours,
            fmt={target: rf"$L_{{\max}}={L0:g}$"},
            manual=[manual_position],
            inline=False,
            fontsize=fontsize,
            zorder=6,
        )

        for label in labels:
            # Prevent labels from extending outside the axes.
            label.set_clip_on(True)
            label.set_clip_path(ax.patch)
            label.set_color("red")
            label.set_path_effects(
                [
                    patheffects.Stroke(linewidth=4.0, foreground="white"),
                    patheffects.Normal(),
                ]
            )

    ax.set_xlabel(r"$\sigma$", fontsize=20)
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis="both", labelsize=14)

    return contourf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the maximum number L_max of posterior paths that can be "
            "released under a fixed (epsilon, delta)-DP budget."
        )
    )

    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--kappa", type=float, default=np.sqrt(0.1))
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--delta-dp", type=float, default=1e-3)
    parser.add_argument("--epsilon-budget", type=float, default=10.0)
    parser.add_argument("--M-Y", type=float, default=1.0)
    parser.add_argument("--rkhs-norm", type=float, default=1.0)

    parser.add_argument("--sigma-min", type=float, default=0.1)
    parser.add_argument("--sigma-max", type=float, default=10.0)
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--r-max", type=float, default=10.0)
    parser.add_argument("--num-grid", type=int, default=80)

    parser.add_argument(
        "--L-cap",
        type=int,
        default=50000,
        help=(
            "Largest L tested. Values equal to this cap are lower bounds on "
            "the true L_max and are enclosed by a dotted boundary."
        ),
    )
    parser.add_argument(
        "--contour-thresholds",
        type=int,
        nargs="+",
        default=[1, 10, 100, 1000, 10_000],
        help="Integer L_max contour levels.",
    )

    parser.add_argument("--beta-safety", type=float, default=1e-8)
    parser.add_argument("--alpha-grid-size", type=int, default=72)
    parser.add_argument("--xatol", type=float, default=1e-7)
    parser.add_argument("--sensitivity-tolerance", type=float, default=1e-10)

    parser.add_argument("--output-prefix", type=str, default="lmax_contours")
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument(
        "--no-save-npz",
        action="store_true",
        help="Do not save the computed grids as an NPZ file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sigma_min <= 0 or args.sigma_max <= args.sigma_min:
        raise ValueError("Require 0 < sigma_min < sigma_max.")
    if args.r_min <= 0 or args.r_max <= args.r_min:
        raise ValueError("Require 0 < r_min < r_max.")
    if args.num_grid < 2:
        raise ValueError("num_grid must be at least 2.")
    if args.eta != 0.0:
        raise ValueError(
            "The three-panel plot requires eta=0 because the specialized "
            "Exp-1D pointwise refinement is implemented for eta=0."
        )
    if args.M_Y < 0:
        raise ValueError("M_Y must be nonnegative.")
    if args.rkhs_norm < 0:
        raise ValueError("rkhs_norm must be nonnegative.")
    if args.alpha_grid_size < 3:
        raise ValueError("alpha_grid_size must be at least 3.")

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    sigmas = np.logspace(
        np.log10(args.sigma_min),
        np.log10(args.sigma_max),
        args.num_grid,
    )
    rs = np.logspace(
        np.log10(args.r_min),
        np.log10(args.r_max),
        args.num_grid,
    )
    S, R = np.meshgrid(sigmas, rs)

    print("Computing tightened sensitivity curves...")
    sensitivities = sensitivity_curves(
        rs=rs,
        n=args.n,
        kappa=args.kappa,
        M_Y=args.M_Y,
        rkhs_norm=args.rkhs_norm,
        tolerance=args.sensitivity_tolerance,
    )

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    epsilon_grids: dict[str, dict[int, np.ndarray]] = {}

    smooth_L_values = sorted(
        {
            1,
            args.L_cap,
            *[
                int(value)
                for value in args.contour_thresholds
                if 1 <= int(value) <= args.L_cap
            ],
        }
    )

    for model in MODELS:
        rkhs_norm = args.rkhs_norm if model == "rkhs" else None

        print(f"Computing {model} L_max grid...")
        results[model] = compute_lmax_grid(
            sigmas=sigmas,
            rs=rs,
            epsilon_budget=args.epsilon_budget,
            L_cap=args.L_cap,
            n=args.n,
            kappa=args.kappa,
            eta=args.eta,
            delta_dp=args.delta_dp,
            model=model,
            M_Y=args.M_Y,
            rkhs_norm=rkhs_norm,
            sensitivities=sensitivities[model],
            beta_safety=args.beta_safety,
            alpha_grid_size=args.alpha_grid_size,
            xatol=args.xatol,
        )

        epsilon_grids[model] = {}
        for L0 in smooth_L_values:
            print(f"Computing {model} smooth boundary grid for L={L0}...")
            epsilon_grids[model][L0] = compute_log10_epsilon_grid(
                sigmas=sigmas,
                rs=rs,
                L=L0,
                n=args.n,
                kappa=args.kappa,
                eta=args.eta,
                delta_dp=args.delta_dp,
                model=model,
                M_Y=args.M_Y,
                rkhs_norm=rkhs_norm,
                sensitivities=sensitivities[model],
                beta_safety=args.beta_safety,
                alpha_grid_size=args.alpha_grid_size,
                xatol=args.xatol,
            )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19.5, 5.2),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    max_log10 = max(1.0, np.log10(args.L_cap))
    filled_levels = np.linspace(0.0, max_log10, 17)
    cmap = plt.get_cmap("Greys").copy()
    cmap.set_bad("white")

    titles = {
        "generic": "Generic responses",
        "rkhs": r"RKHS responses: $f_*\in\mathcal{H}_k$",
        "exp_1d": "1D exponential kernel",
    }

    last_contourf = None
    for ax, model in zip(axes_flat, MODELS):
        lmax, capped = results[model]
        contourf = add_lmax_panel(
            ax=ax,
            S=S,
            R=R,
            lmax=lmax,
            title=titles[model],
            L_cap=args.L_cap,
            contour_thresholds=args.contour_thresholds,
            epsilon_grids=epsilon_grids[model],
            epsilon_budget=args.epsilon_budget,
            filled_levels=filled_levels,
            cmap=cmap,
        )
        if contourf is not None:
            last_contourf = contourf

    axes_flat[0].set_ylabel(r"$r$", fontsize=20)

    xticks = [value for value in [0.1, 0.5, 1, 5, 10] if args.sigma_min <= value <= args.sigma_max]
    yticks = [value for value in [0.1, 0.5, 1, 5, 10] if args.r_min <= value <= args.r_max]

    for ax in axes_flat:
        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{value:g}" for value in xticks])
        if yticks:
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{value:g}" for value in yticks])
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    if last_contourf is not None:
        cbar = fig.colorbar(last_contourf, ax=axes_flat.tolist(), location="right")
        tick_values = [
            10**power
            for power in range(int(np.floor(max_log10)) + 1)
            if 10**power <= args.L_cap
        ]
        cbar.set_ticks([np.log10(value) for value in tick_values])
        cbar.set_ticklabels([f"{value:g}" for value in tick_values])
        cbar.set_label(r"$L_{\max}$", fontsize=18)
        cbar.ax.tick_params(labelsize=12)

    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    npz_path = output_prefix.with_suffix(".npz")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    if not args.no_save_npz:
        payload: dict[str, object] = {
            "sigmas": sigmas,
            "rs": rs,
            "epsilon_budget": args.epsilon_budget,
            "L_cap": args.L_cap,
            "n": args.n,
            "kappa": args.kappa,
            "eta": args.eta,
            "delta_dp": args.delta_dp,
            "M_Y": args.M_Y,
            "rkhs_norm": args.rkhs_norm,
            "accountant_version": getattr(DP, "ACCOUNTANT_VERSION", "unknown"),
        }
        for model, (lmax, capped) in results.items():
            payload[f"Lmax_{model}"] = lmax
            payload[f"capped_{model}"] = capped
            payload[f"sensitivity_{model}"] = sensitivities[model]
            for L0, Z_eps in epsilon_grids[model].items():
                payload[f"log10_epsilon_{model}_L{L0}"] = Z_eps
        np.savez(npz_path, **payload)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    if not args.no_save_npz:
        print(f"Saved {npz_path}")


if __name__ == "__main__":
    main()
