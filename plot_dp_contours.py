"""Create the three-panel contour figure using only tightened DP bounds.

Panels:
    1. generic bounded-response log10(epsilon);
    2. fixed-RKHS-response or 1D-exponential-kernel log10(epsilon);
    3. generic epsilon quotient for eta=sigma versus the baseline eta.

The default baseline is eta=0.  Both PDF and PNG figures and the underlying
NumPy arrays are saved.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, patheffects, ticker


def _load_dp_utils() -> ModuleType:
    """Load the tightened sibling utility despite attachment-name suffixes."""
    directory = Path(__file__).resolve().parent
    candidates = ("dp_utils(3).py", "dp_utils.py", "dp_rdp_utils.py")
    for filename in candidates:
        path = directory / filename
        if path.is_file():
            spec = importlib.util.spec_from_file_location("gp_dp_utils_tight", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if getattr(module, "ACCOUNTANT_VERSION", None) == "tight-rdp-2026-09":
                return module
    names = ", ".join(candidates)
    raise ImportError(f"Could not find a tightened DP utility module: {names}")


DP = _load_dp_utils()


def _sensitivity_curves(
    *,
    rs: np.ndarray,
    n: int,
    kappa: float,
    M_Y: float,
    rkhs_norm: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generic = np.empty(len(rs), dtype=float)
    coupled = np.empty(len(rs), dtype=float)
    operator = np.empty(len(rs), dtype=float)
    rkhs = np.empty(len(rs), dtype=float)
    exp_1d = np.empty(len(rs), dtype=float)

    for index, r in enumerate(rs):
        components = DP.generic_sensitivity_components(
            n=n,
            r=float(r),
            kappa=kappa,
            M_Y=M_Y,
            tolerance=tolerance,
        )
        coupled[index] = components["coupled"]
        operator[index] = components["operator"]
        generic[index] = components["minimum"]
        rkhs[index] = DP.delta_n_rkhs(
            n=n,
            r=float(r),
            kappa=kappa,
            rkhs_norm=rkhs_norm,
        )
        exp_1d[index] = DP.delta_n_exp_1d(
            n=n,
            r=float(r),
            kappa=kappa,
            M_Y=M_Y,
        )

    return generic, coupled, operator, rkhs, exp_1d


def compute_epsilon_grids(
    *,
    sigmas: np.ndarray,
    rs: np.ndarray,
    generic_sensitivity: np.ndarray,
    middle_sensitivity: np.ndarray,
    middle_model: str,
    n: int,
    kappa: float,
    eta_baseline: float,
    L: int,
    delta_dp: float,
    M_Y: float,
    rkhs_norm: float,
    beta_safety: float,
    alpha_grid_size: int,
    xatol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute generic, selected middle-panel, and enhanced generic grids."""
    shape = (len(rs), len(sigmas))
    generic_baseline = np.empty(shape, dtype=float)
    middle_baseline = np.empty(shape, dtype=float)
    generic_eta_sigma = np.empty(shape, dtype=float)

    total = len(rs) * len(sigmas)
    completed = 0
    report_every = max(1, len(rs) // 10)

    for i, r in enumerate(rs):
        for j, sigma in enumerate(sigmas):
            common = dict(
                n=n,
                r=float(r),
                kappa=kappa,
                sigma=float(sigma),
                L=L,
                delta=delta_dp,
                M_Y=M_Y,
                beta_safety=beta_safety,
                grid_size=alpha_grid_size,
                xatol=xatol,
                return_details=False,
            )

            generic_baseline[i, j] = DP.epsilon_for_delta(
                **common,
                model="generic",
                eta=eta_baseline,
                sensitivity_override=float(generic_sensitivity[i]),
            )
            middle_baseline[i, j] = DP.epsilon_for_delta(
                **common,
                model=middle_model,
                rkhs_norm=rkhs_norm,
                eta=eta_baseline,
                sensitivity_override=float(middle_sensitivity[i]),
            )
            generic_eta_sigma[i, j] = DP.epsilon_for_delta(
                **common,
                model="generic",
                eta=float(sigma),
                sensitivity_override=float(generic_sensitivity[i]),
            )
            completed += 1

        if (i + 1) % report_every == 0 or i + 1 == len(rs):
            print(f"Finished {completed}/{total} parameter pairs")

    return generic_baseline, middle_baseline, generic_eta_sigma


def _positive_log10(values: np.ndarray) -> np.ndarray:
    floor = np.finfo(float).tiny
    return np.log10(np.maximum(values, floor))


def _add_epsilon_panel(
    *,
    ax: plt.Axes,
    sigma_mesh: np.ndarray,
    r_mesh: np.ndarray,
    log10_epsilon: np.ndarray,
    levels: np.ndarray,
    threshold_levels: np.ndarray,
    title: str,
):
    filled = ax.contourf(
        sigma_mesh,
        r_mesh,
        log10_epsilon,
        levels=levels,
        extend="both",
        cmap=cm.Greys,
    )

    finite = log10_epsilon[np.isfinite(log10_epsilon)]
    if finite.size:
        active = threshold_levels[
            (threshold_levels >= np.min(finite))
            & (threshold_levels <= np.max(finite))
        ]
        if active.size:
            contours = ax.contour(
                sigma_mesh,
                r_mesh,
                log10_epsilon,
                levels=active,
                linewidths=1.5,
                colors="black",
            )
            labels = {
                0.0: r"$\varepsilon=1$",
                1.0: r"$\varepsilon=10$",
                2.0: r"$\varepsilon=100$",
            }
            label_artists = ax.clabel(
                contours,
                fmt=labels,
                # Do not delete a long section of the contour beneath each
                # label.  This is especially conspicuous on logarithmic axes.
                inline=False,
                fontsize=12,
            )
            for label in label_artists:
                # A narrow halo keeps the text legible while masking only the
                # line immediately adjacent to each glyph.
                label.set_path_effects(
                    [
                        patheffects.Stroke(linewidth=10, foreground="white"),
                        patheffects.Normal(),
                    ]
                )

    ax.set_title(title, fontsize=18)
    return filled


def _configure_axes(
    axes: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    r_min: float,
    r_max: float,
) -> None:
    default_ticks = np.asarray([0.1, 0.5, 1.0, 5.0, 10.0])
    sigma_ticks = default_ticks[
        (default_ticks >= sigma_min) & (default_ticks <= sigma_max)
    ]
    r_ticks = default_ticks[(default_ticks >= r_min) & (default_ticks <= r_max)]

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\sigma$", fontsize=18)
        ax.tick_params(axis="both", labelsize=12)
        if sigma_ticks.size:
            ax.xaxis.set_major_locator(ticker.FixedLocator(sigma_ticks))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        if r_ticks.size:
            ax.yaxis.set_major_locator(ticker.FixedLocator(r_ticks))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))

    axes[0].set_ylabel(r"$r$", fontsize=18)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot three GP-posterior DP panels with tightened bounds."
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--kappa", type=float, default=math.sqrt(0.1))
    parser.add_argument("--eta", type=float, default=0.0,
                        help="baseline eta; default 0")
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--delta-dp", type=float, default=1e-4)
    parser.add_argument("--M-Y", type=float, default=1.0)
    parser.add_argument("--rkhs-norm", type=float, default=1.0)
    parser.add_argument(
        "--middle-panel",
        choices=("rkhs", "exp-1d"),
        default="rkhs",
        help=(
            "bound shown in the middle panel: fixed-RKHS responses "
            "(default) or the specialized 1D exponential-kernel bound"
        ),
    )

    parser.add_argument("--sigma-min", type=float, default=0.1)
    parser.add_argument("--sigma-max", type=float, default=10.0)
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--r-max", type=float, default=10.0)
    parser.add_argument("--num-grid", type=int, default=80)

    parser.add_argument("--beta-safety", type=float, default=1e-8)
    parser.add_argument("--alpha-grid-size", type=int, default=100)
    parser.add_argument("--xatol", type=float, default=1e-7)
    parser.add_argument("--sensitivity-tolerance", type=float, default=1e-10)

    parser.add_argument("--output-prefix", type=str,
                        default="dp_contours_tighter_three_panel")
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()

    if args.num_grid < 2:
        parser.error("--num-grid must be at least 2")
    if args.sigma_min <= 0 or args.sigma_max <= args.sigma_min:
        parser.error("require 0 < --sigma-min < --sigma-max")
    if args.r_min <= 0 or args.r_max <= args.r_min:
        parser.error("require 0 < --r-min < --r-max")
    if args.middle_panel == "exp-1d" and args.eta != 0.0:
        parser.error(
            "--middle-panel exp-1d requires --eta 0 because its pointwise "
            "mean refinement is implemented for the ordinary mechanism"
        )

    sigmas = np.logspace(
        np.log10(args.sigma_min), np.log10(args.sigma_max), args.num_grid
    )
    rs = np.logspace(np.log10(args.r_min), np.log10(args.r_max), args.num_grid)
    sigma_mesh, r_mesh = np.meshgrid(sigmas, rs)

    print("Computing tightened sensitivity curves...")
    generic_delta, coupled_delta, operator_delta, rkhs_delta, exp_1d_delta = (
        _sensitivity_curves(
            rs=rs,
            n=args.n,
            kappa=args.kappa,
            M_Y=args.M_Y,
            rkhs_norm=args.rkhs_norm,
            tolerance=args.sensitivity_tolerance,
        )
    )

    if args.middle_panel == "rkhs":
        middle_model = "rkhs"
        middle_delta = rkhs_delta
        middle_title = r"RKHS responses: $f_*\in\mathcal{H}_k$"
    else:
        middle_model = "exp_1d"
        middle_delta = exp_1d_delta
        middle_title = "1D exponential kernel"

    print("Computing the three tightened epsilon grids...")
    epsilon_generic, epsilon_middle, epsilon_eta_sigma = compute_epsilon_grids(
        sigmas=sigmas,
        rs=rs,
        generic_sensitivity=generic_delta,
        middle_sensitivity=middle_delta,
        middle_model=middle_model,
        n=args.n,
        kappa=args.kappa,
        eta_baseline=args.eta,
        L=args.L,
        delta_dp=args.delta_dp,
        M_Y=args.M_Y,
        rkhs_norm=args.rkhs_norm,
        beta_safety=args.beta_safety,
        alpha_grid_size=args.alpha_grid_size,
        xatol=args.xatol,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        quotient = epsilon_eta_sigma / epsilon_generic
    quotient = np.where(np.isfinite(quotient) & (quotient > 0.0), quotient, np.nan)

    log_generic = _positive_log10(epsilon_generic)
    log_middle = _positive_log10(epsilon_middle)
    epsilon_levels = np.arange(0.0, 7.0, 1.0)
    threshold_levels = np.asarray([0.0, 1.0, 2.0])
    quotient_levels = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 5.2),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    epsilon_filled = _add_epsilon_panel(
        ax=axes[0],
        sigma_mesh=sigma_mesh,
        r_mesh=r_mesh,
        log10_epsilon=log_generic,
        levels=epsilon_levels,
        threshold_levels=threshold_levels,
        title="Generic responses",
    )
    _add_epsilon_panel(
        ax=axes[1],
        sigma_mesh=sigma_mesh,
        r_mesh=r_mesh,
        log10_epsilon=log_middle,
        levels=epsilon_levels,
        threshold_levels=threshold_levels,
        title=middle_title,
    )

    quotient_filled = axes[2].contourf(
        sigma_mesh,
        r_mesh,
        quotient,
        levels=quotient_levels,
        extend="both",
        cmap=cm.Greys_r,
    )
    baseline_label = "0" if args.eta == 0 else f"{args.eta:g}"
    axes[2].set_title(
        rf"Generic: $\varepsilon_{{\eta=\sigma}}/"
        rf"\varepsilon_{{\eta={baseline_label}}}$",
        fontsize=18,
    )

    _configure_axes(
        axes,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        r_min=args.r_min,
        r_max=args.r_max,
    )

    epsilon_bar = fig.colorbar(epsilon_filled, ax=axes[:2], location="right")
    epsilon_bar.set_label(r"$\log_{10}(\varepsilon)$", fontsize=16)
    epsilon_bar.ax.tick_params(labelsize=11)

    quotient_bar = fig.colorbar(quotient_filled, ax=axes[2], location="right")
    quotient_bar.set_label(
        rf"$\varepsilon_{{\eta=\sigma}}/"
        rf"\varepsilon_{{\eta={baseline_label}}}$",
        fontsize=16,
    )
    quotient_bar.ax.tick_params(labelsize=11)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    npz_path = output_prefix.with_suffix(".npz")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    saved_arrays = dict(
        sigmas=sigmas,
        rs=rs,
        epsilon_generic=epsilon_generic,
        epsilon_middle=epsilon_middle,
        epsilon_eta_sigma=epsilon_eta_sigma,
        epsilon_quotient=quotient,
        generic_sensitivity=generic_delta,
        coupled_sensitivity=coupled_delta,
        operator_sensitivity=operator_delta,
        rkhs_sensitivity=rkhs_delta,
        exp_1d_sensitivity=exp_1d_delta,
        epsilon_levels=epsilon_levels,
        quotient_levels=quotient_levels,
        middle_panel=args.middle_panel,
        n=args.n,
        kappa=args.kappa,
        eta_baseline=args.eta,
        L=args.L,
        delta_dp=args.delta_dp,
        M_Y=args.M_Y,
        rkhs_norm=args.rkhs_norm,
    )
    if args.middle_panel == "rkhs":
        saved_arrays["epsilon_rkhs"] = epsilon_middle
    else:
        saved_arrays["epsilon_exp_1d"] = epsilon_middle
    np.savez(npz_path, **saved_arrays)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    print(f"Saved {npz_path}")


if __name__ == "__main__":
    main()
