"""Tight DP accounting utilities for GP posterior sampling.

Only the tightened bounds are implemented:

* the signed rank-two covariance term;
* the improved RDP-to-DP conversion;
* the minimum of the coupled and operator generic sensitivities;
* the factor-two-improved fixed-RKHS-response sensitivity; and
* the bounded-mean refinement for the 1D exponential kernel.

The optional ``eta`` parameter is the scale of an independent draw from the
GP prior added to the posterior draw.  For ``eta > 0`` the tightened enhanced
mechanism bound is used.
"""

from __future__ import annotations

import heapq
import math
from typing import Callable

import numpy as np


ACCOUNTANT_VERSION = "tight-rdp-2026-09"


def _validate_common(n: int, r: float, kappa: float) -> None:
    if n < 1:
        raise ValueError("n must be at least 1")
    if r <= 0.0:
        raise ValueError("r must be positive")
    if not (0.0 <= kappa <= 1.0):
        raise ValueError("kappa must lie in [0, 1]")


def v_n(n: int, r: float, kappa: float) -> float:
    """Return V_n(r) = 1 - kappa^2 (n-1)/(n-1+r^2)."""
    _validate_common(n, r, kappa)
    value = 1.0 - kappa * kappa * (n - 1.0) / (n - 1.0 + r * r)
    return min(1.0, max(0.0, value))


def phi_from_v(r: float, v: float) -> float:
    """Return sup_{0 <= u <= v} u/(r^2+u)^2."""
    if r <= 0.0:
        raise ValueError("r must be positive")
    if v < 0.0:
        raise ValueError("v must be nonnegative")
    u_star = min(v, r * r)
    return u_star / (r * r + u_star) ** 2


def phi_n(n: int, r: float, kappa: float) -> float:
    """Return Phi_n(r)."""
    return phi_from_v(r, v_n(n, r, kappa))


def tau_tilde(
    n: int,
    r: float,
    kappa: float,
    sigma: float,
    eta: float = 0.0,
) -> float:
    """Return the covariance parameter for the enhanced mechanism."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if eta < 0.0:
        raise ValueError("eta must be nonnegative")
    v = v_n(n, r, kappa)
    return sigma * sigma * v / (
        sigma * sigma * r * r + eta * eta * (v + r * r)
    )


def _coupled_supremum_upper(
    n: int,
    r: float,
    v: float,
    tolerance: float = 1e-10,
) -> float:
    """Certified numerical upper enclosure of the coupled scalar maximum."""
    if not (0.0 <= v <= 1.0 + 1e-12):
        raise ValueError("the coupled bound requires 0 <= V_n(r) <= 1")
    if v <= 0.0:
        return 0.0

    q = r * r
    scale = math.sqrt(max(0.0, n - 1.0)) / r
    peak_second = 4.0 * q / (
        math.sqrt(9.0 * q * q + 8.0 * q) + 3.0 * q
    )

    def first(u: float) -> float:
        return math.sqrt(max(0.0, u)) / (q + u)

    def second(u: float) -> float:
        return scale * u * math.sqrt(max(0.0, 1.0 - u)) / (q + u)

    def objective(u: float) -> float:
        return first(u) + second(u)

    def clip(value: float, lower: float, upper: float) -> float:
        return min(upper, max(lower, value))

    def interval_upper(lower: float, upper: float) -> float:
        return (
            first(clip(q, lower, upper))
            + second(clip(peak_second, lower, upper))
        )

    candidates = [0.0, v, 0.5 * v, min(q, v), min(peak_second, v)]
    best = max(objective(u) for u in candidates)
    heap: list[tuple[float, float, float]] = [
        (-interval_upper(0.0, v), 0.0, v)
    ]

    for _ in range(200_000):
        upper = max(best, -heap[0][0])
        if upper - best <= tolerance * max(1.0, best):
            return upper * (1.0 + 2e-14) + 2e-14

        _, lower, right = heapq.heappop(heap)
        midpoint = 0.5 * (lower + right)
        best = max(best, objective(midpoint))

        for sub_lower, sub_upper in ((lower, midpoint), (midpoint, right)):
            bound = interval_upper(sub_lower, sub_upper)
            if bound >= best:
                heapq.heappush(heap, (-bound, sub_lower, sub_upper))

        if not heap:
            return best * (1.0 + 2e-14) + 2e-14

    raise RuntimeError("generic sensitivity maximization did not converge")


def generic_sensitivity_components(
    n: int,
    r: float,
    kappa: float,
    M_Y: float = 1.0,
    tolerance: float = 1e-10,
) -> dict[str, float]:
    """Return the coupled, operator, and minimum generic sensitivities."""
    _validate_common(n, r, kappa)
    if M_Y < 0.0:
        raise ValueError("M_Y must be nonnegative")

    v = v_n(n, r, kappa)
    coupled = 2.0 * M_Y * _coupled_supremum_upper(
        n=n, r=r, v=v, tolerance=tolerance
    )
    operator = (
        M_Y
        * math.sqrt(max(0.0, n - 1.0))
        / r
        * v
        / (r * r + v)
        + 2.0 * M_Y * math.sqrt(phi_from_v(r, v))
    )
    return {
        "coupled": coupled,
        "operator": operator,
        "minimum": min(coupled, operator),
    }


def delta_n_generic(
    n: int,
    r: float,
    kappa: float,
    M_Y: float = 1.0,
    tolerance: float = 1e-10,
) -> float:
    """Return min{bar Delta_n(r), hat Delta_n(r)}."""
    return generic_sensitivity_components(
        n=n,
        r=r,
        kappa=kappa,
        M_Y=M_Y,
        tolerance=tolerance,
    )["minimum"]


def delta_n_rkhs(
    n: int,
    r: float,
    kappa: float,
    rkhs_norm: float,
) -> float:
    """Return ||f_*||_H V_n(r)/(r^2+V_n(r))."""
    if rkhs_norm < 0.0:
        raise ValueError("rkhs_norm must be nonnegative")
    v = v_n(n, r, kappa)
    return rkhs_norm * v / (r * r + v)


def delta_n_exp_1d(
    n: int,
    r: float,
    kappa: float,
    M_Y: float = 1.0,
) -> float:
    """Return the O(1) 1D-exponential sensitivity 4 M_Y sqrt(Phi_n)."""
    if M_Y < 0.0:
        raise ValueError("M_Y must be nonnegative")
    return 2.0 * M_Y * math.sqrt(phi_n(n, r, kappa))


def psi_alpha_tight(alpha: float, tau: float) -> float:
    """Signed rank-two covariance contribution for one posterior draw."""
    if alpha <= 1.0:
        raise ValueError("alpha must be greater than 1")
    if tau < 0.0:
        raise ValueError("tau must be nonnegative")
    if tau == 0.0:
        return 0.0

    a = alpha - 1.0
    if a * tau >= 1.0:
        return math.inf
    argument = alpha * a * tau * tau / (1.0 + tau)
    if argument >= 1.0:
        return math.inf
    return -math.log1p(-argument) / (2.0 * a)


def rdp_bound_tight(
    *,
    alpha: float,
    v: float,
    r: float,
    sigma: float,
    sensitivity: float,
    eta: float = 0.0,
    pointwise_sensitivity: float | None = None,
) -> float:
    """Return the tightened one-draw RDP bound.

    ``pointwise_sensitivity`` activates the bounded-mean refinement.  It is
    currently used only for the ordinary (eta=0) 1D exponential mechanism.
    """
    if r <= 0.0 or sigma <= 0.0:
        raise ValueError("r and sigma must be positive")
    if eta < 0.0 or sensitivity < 0.0:
        raise ValueError("eta and sensitivity must be nonnegative")
    if pointwise_sensitivity is not None and pointwise_sensitivity < 0.0:
        raise ValueError("pointwise_sensitivity must be nonnegative")

    q = r * r
    tau = sigma * sigma * v / (
        sigma * sigma * q + eta * eta * (v + q)
    )
    if tau == 0.0:
        return 0.0
    if not (1.0 < alpha < 1.0 + 1.0 / tau):
        return math.inf

    covariance = psi_alpha_tight(alpha, tau)
    a = alpha - 1.0
    d2 = sensitivity * sensitivity

    if pointwise_sensitivity is not None:
        if eta != 0.0:
            raise ValueError(
                "the pointwise refinement is implemented only for eta=0"
            )
        denominator = q - a * v
        if denominator <= 0.0:
            return math.inf
        directional2 = min(v * d2, pointwise_sensitivity**2)
        mean = alpha / (2.0 * sigma * sigma) * (
            d2 + alpha * directional2 / denominator
        )
    else:
        denominator = (
            sigma * sigma * (q - a * v)
            + eta * eta * (v + q)
        )
        if denominator <= 0.0:
            return math.inf
        mean = alpha / 2.0 * (v + q) / denominator * d2

    return covariance + mean


def _model_sensitivity(
    *,
    model: str,
    n: int,
    r: float,
    kappa: float,
    M_Y: float,
    rkhs_norm: float | None,
    sensitivity_tolerance: float,
) -> tuple[float, float | None]:
    if model == "generic":
        return (
            delta_n_generic(
                n=n,
                r=r,
                kappa=kappa,
                M_Y=M_Y,
                tolerance=sensitivity_tolerance,
            ),
            None,
        )
    if model == "rkhs":
        if rkhs_norm is None:
            raise ValueError("rkhs_norm is required for model='rkhs'")
        return delta_n_rkhs(n, r, kappa, rkhs_norm), None
    if model == "exp_1d":
        return delta_n_exp_1d(n, r, kappa, M_Y), 2.0 * M_Y
    raise ValueError("model must be 'generic', 'rkhs', or 'exp_1d'")


def _golden_section_minimize(
    objective: Callable[[float], float],
    left: float,
    right: float,
    tolerance: float,
    max_iterations: int = 200,
) -> tuple[float, float]:
    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - inverse_phi * (right - left)
    x2 = left + inverse_phi * (right - left)
    f1 = objective(x1)
    f2 = objective(x2)

    for _ in range(max_iterations):
        if right - left <= tolerance:
            break
        if f1 > f2:
            left = x1
            x1, f1 = x2, f2
            x2 = left + inverse_phi * (right - left)
            f2 = objective(x2)
        else:
            right = x2
            x2, f2 = x1, f1
            x1 = right - inverse_phi * (right - left)
            f1 = objective(x1)

    minimizer = 0.5 * (left + right)
    return objective(minimizer), minimizer


def _minimize_over_beta(
    objective: Callable[[float], float],
    beta_safety: float,
    grid_size: int,
    xatol: float,
) -> tuple[float, float]:
    if not (0.0 < beta_safety < 0.5):
        raise ValueError("beta_safety must lie in (0, 0.5)")
    if grid_size < 12:
        raise ValueError("grid_size must be at least 12")

    lower = beta_safety
    upper = 1.0 - beta_safety
    split = min(0.1, upper)
    geometric_count = max(6, grid_size // 2)
    linear_count = max(6, grid_size - geometric_count)
    grid = np.unique(
        np.concatenate(
            (
                np.geomspace(lower, split, geometric_count),
                np.linspace(split, upper, linear_count),
            )
        )
    )
    values = np.asarray([objective(float(beta)) for beta in grid])
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return math.inf, math.nan

    ordered = finite_indices[np.argsort(values[finite_indices])]
    candidates: list[tuple[float, float]] = [
        (float(values[index]), float(grid[index])) for index in ordered[:5]
    ]
    for index in ordered[:5]:
        lo = float(grid[max(0, index - 1)])
        hi = float(grid[min(len(grid) - 1, index + 1)])
        if hi > lo:
            candidates.append(
                _golden_section_minimize(
                    objective,
                    lo,
                    hi,
                    tolerance=xatol,
                )
            )

    return min(candidates, key=lambda item: item[0])


def improved_rdp_to_dp(
    *,
    alpha: float,
    rdp_epsilon: float,
    L: int,
    delta: float,
) -> float:
    """Compose L RDP releases and apply the improved conversion."""
    if alpha <= 1.0:
        raise ValueError("alpha must be greater than 1")
    if L < 1:
        raise ValueError("L must be at least 1")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1)")
    a = alpha - 1.0
    return (
        L * rdp_epsilon
        + math.log(a / alpha)
        - (math.log(delta) + math.log(alpha)) / a
    )


def epsilon_for_delta(
    *,
    n: int,
    r: float,
    kappa: float,
    sigma: float,
    L: int,
    delta: float,
    model: str,
    M_Y: float = 1.0,
    rkhs_norm: float | None = None,
    eta: float = 0.0,
    sensitivity_override: float | None = None,
    beta_safety: float = 1e-8,
    grid_size: int = 72,
    xatol: float = 1e-7,
    sensitivity_tolerance: float = 1e-10,
    return_details: bool = False,
) -> float | dict[str, float | str | tuple[float, float]]:
    """Optimize the tightened RDP guarantee and improved conversion."""
    _validate_common(n, r, kappa)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if eta < 0.0:
        raise ValueError("eta must be nonnegative")
    if M_Y < 0.0:
        raise ValueError("M_Y must be nonnegative")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1)")
    if L < 1:
        raise ValueError("L must be at least 1")
    if model not in {"generic", "rkhs", "exp_1d"}:
        raise ValueError("model must be 'generic', 'rkhs', or 'exp_1d'")
    if model == "exp_1d" and eta != 0.0:
        raise ValueError("the exp_1d pointwise refinement requires eta=0")

    v = v_n(n, r, kappa)
    if sensitivity_override is None:
        sensitivity, pointwise = _model_sensitivity(
            model=model,
            n=n,
            r=r,
            kappa=kappa,
            M_Y=M_Y,
            rkhs_norm=rkhs_norm,
            sensitivity_tolerance=sensitivity_tolerance,
        )
    else:
        if sensitivity_override < 0.0:
            raise ValueError("sensitivity_override must be nonnegative")
        sensitivity = sensitivity_override
        pointwise = 2.0 * M_Y if model == "exp_1d" else None
    tau = tau_tilde(n, r, kappa, sigma, eta)

    if tau == 0.0:
        details: dict[str, float | str | tuple[float, float]] = {
            "Model": model,
            "Epsilon": 0.0,
            "Delta": delta,
            "OptimalAlpha": math.inf,
            "OptimalBeta": 0.0,
            "RDPAtOptimalAlpha": 0.0,
            "Vn": v,
            "PhiN": phi_from_v(r, v),
            "DeltaN": sensitivity,
            "Tau": 0.0,
            "AlphaRange": (1.0, math.inf),
        }
        return details if return_details else 0.0

    def objective(beta: float) -> float:
        alpha = 1.0 + beta / tau
        rho = rdp_bound_tight(
            alpha=alpha,
            v=v,
            r=r,
            sigma=sigma,
            sensitivity=sensitivity,
            eta=eta,
            pointwise_sensitivity=pointwise,
        )
        return improved_rdp_to_dp(
            alpha=alpha,
            rdp_epsilon=rho,
            L=L,
            delta=delta,
        )

    epsilon, beta = _minimize_over_beta(
        objective,
        beta_safety=beta_safety,
        grid_size=grid_size,
        xatol=xatol,
    )
    epsilon = max(0.0, epsilon)
    alpha = 1.0 + beta / tau
    rho = rdp_bound_tight(
        alpha=alpha,
        v=v,
        r=r,
        sigma=sigma,
        sensitivity=sensitivity,
        eta=eta,
        pointwise_sensitivity=pointwise,
    )

    details = {
        "Model": model,
        "Epsilon": epsilon,
        "Delta": delta,
        "OptimalAlpha": alpha,
        "OptimalBeta": beta,
        "RDPAtOptimalAlpha": rho,
        "Vn": v,
        "PhiN": phi_from_v(r, v),
        "DeltaN": sensitivity,
        "Tau": tau,
        "AlphaRange": (1.0, 1.0 + 1.0 / tau),
    }
    return details if return_details else epsilon


def log10_epsilon_for_delta(**kwargs) -> float:
    """Return log10 epsilon for contour plotting."""
    kwargs["return_details"] = False
    epsilon = float(epsilon_for_delta(**kwargs))
    return math.log10(max(epsilon, np.finfo(float).tiny))


__all__ = [
    "ACCOUNTANT_VERSION",
    "delta_n_exp_1d",
    "delta_n_generic",
    "delta_n_rkhs",
    "epsilon_for_delta",
    "generic_sensitivity_components",
    "improved_rdp_to_dp",
    "log10_epsilon_for_delta",
    "phi_from_v",
    "phi_n",
    "psi_alpha_tight",
    "rdp_bound_tight",
    "tau_tilde",
    "v_n",
]
