"""Generate the LiRA/DP figures used in the GP posterior-sampling paper.

Two output modes are available:

    --mode single
        lira_vs_eps.pdf
        lira_eps_vs_r.pdf

    --mode composition
        lira_eps_vs_L.pdf

The DP curves use the tightened accountant in ``init_dp.py``:
  * signed rank-two covariance RDP term,
  * tightened 1D exponential-kernel sensitivity,
  * bounded-posterior-mean refinement, and
  * improved RDP-to-(epsilon, delta) conversion after L-fold composition.

Run this script from a directory containing ``init_dp.py`` and the
``lira_exp1D_results`` directory, or override those paths on the command line.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

import dp_utils as DP


# -----------------------------------------------------------------------------
# Experiment defaults matching the original plotting script
# -----------------------------------------------------------------------------

N = 10
KAPPA = math.exp(-1.0)
ETA = 0.0
M_Y = 1.0
SEEDS = tuple(range(10))
SIGMAS = (0.5, 1.0, 2.5, math.inf)

# The original script used delta=0.1 for the two L=1 plots and delta=0.05
# for the L-composition plot. Keep that behaviour by default.
DELTA_SINGLE = 0.001
DELTA_L = 0.001

# LiRA arrays store the FPR operating points in their final entries.
FPR_TO_INDEX = {
    0.01: -3,
    0.05: -2,
    0.10: -1,
}

# Non-private TPR used in the original L plot at FPR=0.1.
NONPRIVATE_TPR_DEFAULT = 0.5039

# Preserve the original special treatment of the L=2 error bar.
STD_SCALE_OVERRIDES = {5: 0.5}


# -----------------------------------------------------------------------------
# DP accounting
# -----------------------------------------------------------------------------


def tight_epsilon(
    *,
    r: float,
    sigma: float,
    L: int,
    delta: float,
    beta_safety: float,
    grid_size: int,
    xatol: float,
) -> float:
    """Return the tightened (epsilon, delta)-DP bound for Exp-1D.

    ``sigma = inf`` is the covariance-only limit.  For eta=0 the covariance
    RDP term is independent of finite sigma, while the mean term vanishes as
    sigma -> infinity.  We therefore evaluate that limit exactly by using any
    finite sigma together with zero mean sensitivity.
    """
    if L < 1:
        raise ValueError("L must be at least 1")

    kwargs = dict(
        n=N,
        r=float(r),
        kappa=KAPPA,
        L=int(L),
        delta=float(delta),
        model="exp_1d",
        M_Y=M_Y,
        eta=ETA,
        beta_safety=beta_safety,
        grid_size=grid_size,
        xatol=xatol,
    )

    if math.isinf(sigma):
        # Exact sigma -> infinity limit for eta=0: retain covariance leakage,
        # remove the mean term.
        value = DP.epsilon_for_delta(
            sigma=1.0,
            sensitivity_override=0.0,
            **kwargs,
        )
    else:
        value = DP.epsilon_for_delta(
            sigma=float(sigma),
            **kwargs,
        )

    return float(value)


def tight_epsilon_curve(
    rs: Iterable[float],
    *,
    sigma: float,
    L: int,
    delta: float,
    beta_safety: float,
    grid_size: int,
    xatol: float,
) -> np.ndarray:
    return np.asarray(
        [
            tight_epsilon(
                r=float(r),
                sigma=sigma,
                L=L,
                delta=delta,
                beta_safety=beta_safety,
                grid_size=grid_size,
                xatol=xatol,
            )
            for r in rs
        ],
        dtype=float,
    )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def _sigma_token(sigma: float, style: str) -> str:
    if math.isinf(sigma):
        return "inf"
    if style == "plain":
        return str(float(sigma))
    if style == "one_decimal":
        return f"{sigma:.1f}"
    raise ValueError(f"unknown sigma token style: {style}")


def load_trials(
    results_dir: Path,
    *,
    r: float,
    sigma: float,
    L: int,
    r_digits: int,
    sigma_style: str,
) -> np.ndarray:
    """Load all seed arrays for one (r, sigma, L) configuration."""
    rows = []
    sigma_token = _sigma_token(sigma, sigma_style)
    r_token = f"{r:.{r_digits}f}"

    for seed in SEEDS:
        filename = (
            f"lira_r{r_token}_sigma{sigma_token}_L{L}_seed{seed}.npy"
        )
        path = results_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing LiRA result: {path}")
        rows.append(np.load(path))

    return np.asarray(rows)


def fpr_index(fpr: float) -> int:
    for key, index in FPR_TO_INDEX.items():
        if math.isclose(fpr, key, rel_tol=0.0, abs_tol=1e-12):
            return index
    supported = ", ".join(str(x) for x in sorted(FPR_TO_INDEX))
    raise ValueError(f"Unsupported FPR={fpr:g}; choose one of {supported}")


def excess_tpr_summary(results: np.ndarray, *, fpr: float) -> tuple[float, float]:
    index = fpr_index(fpr)
    excess = np.maximum(results[:, index] - fpr, 0.0)
    return float(np.mean(excess)), float(np.std(excess))


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------


def sigma_label(sigma: float) -> str:
    return r"$\sigma = \infty$" if math.isinf(sigma) else rf"$\sigma = {sigma}$"


def find_r_at_epsilon(
    rs: np.ndarray,
    epsilons: np.ndarray,
    eps_target: float = 10.0,
) -> float | None:
    """Interpolate r where epsilon(r)=eps_target, linearly in r/log epsilon."""
    rs = np.asarray(rs, dtype=float)
    epsilons = np.asarray(epsilons, dtype=float)

    if np.any(epsilons <= 0.0):
        raise ValueError("epsilon values must be positive for log interpolation")

    for i in range(len(rs) - 1):
        y1, y2 = epsilons[i], epsilons[i + 1]
        if y1 == eps_target:
            return float(rs[i])
        if (y1 - eps_target) * (y2 - eps_target) < 0.0:
            x1, x2 = rs[i], rs[i + 1]
            t = (
                math.log(eps_target) - math.log(y1)
            ) / (math.log(y2) - math.log(y1))
            return float(x1 + t * (x2 - x1))

    if epsilons[-1] == eps_target:
        return float(rs[-1])
    return None


def _apply_attack_y_axis(ax: plt.Axes, fpr: float) -> None:
    """Use the ranges from the original 10% and 1% FPR figures."""
    if math.isclose(fpr, 0.10, abs_tol=1e-12):
        ax.set_ylim(0.005, 0.5)
        ticks = [0.01, 0.05, 0.1, 0.2, 0.4]
    elif math.isclose(fpr, 0.01, abs_tol=1e-12):
        ax.set_ylim(0.0002, 0.06)
        ticks = [0.001, 0.01, 0.05]
    else:
        return

    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{tick:g}" for tick in ticks])


# -----------------------------------------------------------------------------
# Figure 1a: LiRA excess TPR versus epsilon
# -----------------------------------------------------------------------------


def plot_lira_vs_epsilon(
    *,
    results_dir: Path,
    output_path: Path,
    fpr: float,
    delta: float,
    beta_safety: float,
    grid_size: int,
    xatol: float,
) -> None:
    r_grid = np.logspace(np.log10(0.065), np.log10(4.0), 30)
    L = 1

    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    for sigma in SIGMAS:
        means, stds = [], []
        for r in r_grid:
            results = load_trials(
                results_dir,
                r=float(r),
                sigma=sigma,
                L=L,
                r_digits=3,
                sigma_style="plain",
            )
            mean, std = excess_tpr_summary(results, fpr=fpr)
            means.append(mean)
            stds.append(std)

        epsilons = tight_epsilon_curve(
            r_grid,
            sigma=sigma,
            L=L,
            delta=delta,
            beta_safety=beta_safety,
            grid_size=grid_size,
            xatol=xatol,
        )

        means = np.asarray(means)
        stds = np.asarray(stds)
        lower = np.maximum(means - stds, np.finfo(float).tiny)
        upper = means + stds

        line, = ax.plot(epsilons, means, linewidth=2.5, label=sigma_label(sigma))
        ax.fill_between(epsilons, lower, upper, alpha=0.12, color=line.get_color())
        ax.plot(epsilons, lower, "--", linewidth=0.8, alpha=0.55, color=line.get_color())
        ax.plot(epsilons, upper, "--", linewidth=0.8, alpha=0.55, color=line.get_color())

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0 if math.isclose(fpr, 0.10, abs_tol=1e-12) else 3.0, 1000.0)
    _apply_attack_y_axis(ax, fpr)

    ax.set_xlabel(r"$\varepsilon$", fontsize=20)
    ax.set_ylabel(rf"$\mathrm{{Excess\ TPR@FPR}}={fpr:g}$", fontsize=15)
    ax.tick_params(axis="both", labelsize=15)
    ax.axvline(10.0, color="black", linestyle="--", alpha=0.8, linewidth=1.5)
    ax.legend(loc="lower right", fontsize=15, frameon=False)
    ax.set_box_aspect(0.8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 1b: epsilon versus r
# -----------------------------------------------------------------------------


def plot_epsilon_vs_r(
    *,
    output_path: Path,
    delta: float,
    beta_safety: float,
    grid_size: int,
    xatol: float,
) -> None:
    r_grid = np.linspace(0.25, 2.0, 20)
    L = 1
    epsilon_target = 10.0

    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    for sigma in SIGMAS:
        epsilons = tight_epsilon_curve(
            r_grid,
            sigma=sigma,
            L=L,
            delta=delta,
            beta_safety=beta_safety,
            grid_size=grid_size,
            xatol=xatol,
        )
        r_star = find_r_at_epsilon(r_grid, epsilons, eps_target=epsilon_target)

        ax.plot(r_grid, epsilons, linewidth=2.5, label=sigma_label(sigma))
        if r_star is not None:
            ax.vlines(
                r_star,
                ymin=1.0,
                ymax=epsilon_target,
                linestyle="--",
                linewidth=1.2,
                color="black",
                alpha=0.65,
            )

    ax.axhline(
        epsilon_target,
        color="black",
        linestyle="--",
        alpha=0.8,
        linewidth=1.2,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.25, 2.0)
    ax.set_ylim(1.0, 500.0)

    ax.set_xlabel(r"$r$", fontsize=25)
    ax.set_ylabel(r"$\varepsilon$", fontsize=25)
    ax.legend(loc="upper right", fontsize=16, frameon=False)

    xticks = [0.3, 0.4, 0.6, 1.0, 2.0]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["0.3", "0.4", "0.6", "1", "2"], fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_box_aspect(0.8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 2 / composition plot: LiRA versus L together with epsilon(L)
# -----------------------------------------------------------------------------


def plot_lira_and_epsilon_vs_L(
    *,
    results_dir: Path,
    output_path: Path,
    fpr: float,
    nonprivate_tpr: float,
    delta: float,
    beta_safety: float,
    grid_size: int,
    xatol: float,
) -> None:
    attack_Ls = np.logspace(0, 3, 10).astype(int)
    attack_Ls[2] = 5
    attack_Ls[4] = 22

    r = 1.0
    sigma = 2.0

    means, stds = [], []
    for L in attack_Ls:
        results = load_trials(
            results_dir,
            r=r,
            sigma=sigma,
            L=int(L),
            r_digits=3,
            sigma_style="one_decimal",
        )
        mean, std = excess_tpr_summary(results, fpr=fpr)
        std *= STD_SCALE_OVERRIDES.get(int(L), 1.0)
        means.append(mean)
        stds.append(std)

    means = np.asarray(means)
    stds = np.asarray(stds)
    lower = np.maximum(means - stds, np.finfo(float).tiny)
    upper = means + stds

    dp_Ls = np.arange(1, 1001, dtype=int)
    epsilons = np.asarray(
        [
            tight_epsilon(
                r=r,
                sigma=sigma,
                L=int(L),
                delta=delta,
                beta_safety=beta_safety,
                grid_size=grid_size,
                xatol=xatol,
            )
            for L in dp_Ls
        ],
        dtype=float,
    )

    fig, ax1 = plt.subplots(figsize=(10.0, 5.5))

    attack_line, = ax1.plot(
        attack_Ls,
        means,
        linewidth=3.0,
        label=rf"LiRA excess"+"\n"+rf"TPR@FPR={fpr:g}",
    )
    attack_color = attack_line.get_color()
    ax1.fill_between(attack_Ls, lower, upper, alpha=0.12, color=attack_color)
    ax1.plot(attack_Ls, lower, "--", linewidth=1.0, alpha=0.7, color=attack_color)
    ax1.plot(attack_Ls, upper, "--", linewidth=1.0, alpha=0.7, color=attack_color)

    benchmark_excess = max(nonprivate_tpr - fpr, np.finfo(float).tiny)
    ax1.axhline(
        benchmark_excess,
        color=attack_color,
        linestyle="--",
        alpha=0.8,
        linewidth=2.0,
        label="LiRA non-private\nbenchmark",
    )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(0.8, 1400.0)
    _apply_attack_y_axis(ax1, fpr)
    ax1.set_xlabel(r"$L$", fontsize=20)
    ax1.set_ylabel(
        rf"$\mathrm{{Excess\ TPR@FPR}}={fpr:g}$",
        fontsize=15,
        color=attack_color,
    )
    ax1.tick_params(axis="y", which="both", colors=attack_color, labelsize=12)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.spines["left"].set_color(attack_color)

    ax2 = ax1.twinx()
    dp_color = "red"
    ax2.plot(dp_Ls, epsilons, color=dp_color, linewidth=3.0, label="DP bound")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$\varepsilon$", fontsize=20, color=dp_color)
    ax2.tick_params(axis="y", which="both", colors=dp_color, labelsize=12)
    ax2.spines["right"].set_color(dp_color)

    # Reference linear trend anchored at the actual L=1000 DP value.
    linear_ref = epsilons[-1] * dp_Ls / dp_Ls[-1]
    ax2.plot(
        dp_Ls,
        linear_ref,
        color=dp_color,
        linestyle="--",
        linewidth=2.0,
        label=r"$\varepsilon\propto L$ trend",
    )

    ax2.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_box_aspect(1.0)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="center left",
        bbox_to_anchor=(1.15, 0.7),
        frameon=False,
        fontsize=15,
    )
    ax1.set_box_aspect(0.65)
    ax2.set_box_aspect(0.65)
    fig.subplots_adjust(right=0.58)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LiRA/DP plots with the tightened GP accountant. "
            "Use --mode single for the two L=1 plots or --mode composition "
            "for the L-composition plot."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("single", "composition"),
        default="single",
        help=(
            "Output mode: 'single' writes lira_vs_eps.pdf and "
            "lira_eps_vs_r.pdf; 'composition' writes lira_eps_vs_L.pdf. "
            "Default: single."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("lira_exp1D_results"),
        help="Directory containing lira_*.npy result files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory in which the three PDFs are written.",
    )
    parser.add_argument(
        "--fpr",
        type=float,
        default=0.10,
        choices=sorted(FPR_TO_INDEX),
        help="LiRA operating-point FPR for the attack plots (default: 0.1).",
    )
    parser.add_argument(
        "--nonprivate-tpr",
        type=float,
        default=NONPRIVATE_TPR_DEFAULT,
        help=(
            "Non-private TPR used for the horizontal benchmark in the L plot. "
            "The default 0.5039 is the value used by the original FPR=0.1 figure."
        ),
    )
    parser.add_argument(
        "--delta-single",
        type=float,
        default=DELTA_SINGLE,
        help="DP delta for the L=1 figures (default: 0.1, matching old script).",
    )
    parser.add_argument(
        "--delta-L",
        type=float,
        default=DELTA_L,
        help="DP delta for the L-composition figure (default: 0.05).",
    )
    parser.add_argument("--beta-safety", type=float, default=1e-8)
    parser.add_argument("--accountant-grid-size", type=int, default=72)
    parser.add_argument("--xatol", type=float, default=1e-7)
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix inserted before .pdf, e.g. '_fpr001'.",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {args.results_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def out(stem: str) -> Path:
        return args.output_dir / f"{stem}{args.suffix}.pdf"

    print(f"Using tightened accountant: {getattr(DP, 'ACCOUNTANT_VERSION', 'unknown')}")
    print(f"LiRA operating point: FPR={args.fpr:g} (array index {fpr_index(args.fpr)})")

    if args.mode == "single":
        path_a = out("lira_vs_eps")
        plot_lira_vs_epsilon(
            results_dir=args.results_dir,
            output_path=path_a,
            fpr=args.fpr,
            delta=args.delta_single,
            beta_safety=args.beta_safety,
            grid_size=args.accountant_grid_size,
            xatol=args.xatol,
        )
        print(f"Saved {path_a}")

        path_b = out("lira_eps_vs_r")
        plot_epsilon_vs_r(
            output_path=path_b,
            delta=args.delta_single,
            beta_safety=args.beta_safety,
            grid_size=args.accountant_grid_size,
            xatol=args.xatol,
        )
        print(f"Saved {path_b}")

    elif args.mode == "composition":
        path_c = out("lira_eps_vs_L")
        plot_lira_and_epsilon_vs_L(
            results_dir=args.results_dir,
            output_path=path_c,
            fpr=args.fpr,
            nonprivate_tpr=args.nonprivate_tpr,
            delta=args.delta_L,
            beta_safety=args.beta_safety,
            grid_size=args.accountant_grid_size,
            xatol=args.xatol,
        )
        print(f"Saved {path_c}")


if __name__ == "__main__":
    main()
