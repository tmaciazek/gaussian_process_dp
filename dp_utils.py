# dp_utils.py

import math
import numpy as np


def v_n(n, r, kappa):
    """
    Equivalent of Mathematica:

        VnFun[n_, r_, kappa_] :=
            1 - kappa^2 (n - 1)/(n - 1 + r^2)
    """
    return 1.0 - kappa**2 * (n - 1.0) / (n - 1.0 + r**2)


def phi_n(n, r, kappa):
    """
    Equivalent of Mathematica:

        PhiNFun[n_, r_, kappa_] :=
            Module[{v}, v = VnFun[n, r, kappa];
                If[v >= r^2, 1/(4 r^2), v/(v + r^2)^2]]
    """
    v = v_n(n, r, kappa)

    if v >= r**2:
        return 1.0 / (4.0 * r**2)

    return v / (v + r**2)**2


def tau_tilde(n, r, kappa, sigma, eta):
    """
    Equivalent of Mathematica:

        TauTildeFun[n_, r_, kappa_, sigma_, eta_] :=
            Module[{v}, v = VnFun[n, r, kappa];
                sigma^2 v/(sigma^2 r^2 + eta^2 (v + r^2))]
    """
    v = v_n(n, r, kappa)

    # Handles sigma = np.inf or math.inf by taking the limiting value.
    if math.isinf(sigma):
        return v / r**2

    return sigma**2 * v / (sigma**2 * r**2 + eta**2 * (v + r**2))


# ---------------------------------------------------------------------
# Core Exp1D DP utility functions
# ---------------------------------------------------------------------

def delta_n_exp_1d(n, r, kappa):
    """
    Equivalent of Mathematica:

        DeltaNExp1DFun[n_, r_, kappa_] :=
            Module[{phi}, phi = PhiNFun[n, r, kappa];
            4 Sqrt[phi]]/Sqrt[2];
    """
    phi = phi_n(n, r, kappa)
    return 4.0 * math.sqrt(phi) / math.sqrt(2.0)


def psi_beta(beta, tau):
    """
    Equivalent of the PsiBetaFun[beta, tau] expression used inside
    EpsilonForDeltaBetaExp1D and DeltaForEpsilonBetaExp1D.
    """
    log1ptau = math.log1p(tau)

    term1 = (
        0.5 * log1ptau
        - tau / (2.0 * beta) * (math.log(1.0 + tau + beta) - log1ptau)
    )

    term2 = (
        -0.5 * log1ptau
        - tau / (2.0 * beta) * math.log1p(-beta)
    )

    return max(term1, term2)


def rdp_bound_beta_exp_1d(n, r, kappa, sigma, eta, L, beta):
    """
    Equivalent of Mathematica:

        RDPBoundBetaExp1DFun[n, r, kappa, sigma, eta, L, beta]
    """
    v = v_n(n, r, kappa)
    delta_n = delta_n_exp_1d(n, r, kappa)
    tau = tau_tilde(n, r, kappa, sigma, eta)

    alpha = 1.0 + beta / tau

    A = sigma**2 * r**2 + eta**2 * (v + r**2)
    denom = A * (1.0 - beta)

    return (
        2.0 * L * psi_beta(beta, tau)
        + L * (alpha / 2.0) * ((v + r**2) / denom) * delta_n**2
    )


# ---------------------------------------------------------------------
# Small replacement for FastBetaMinimize
# ---------------------------------------------------------------------

def fast_beta_minimize(obj, beta_min=1e-8, beta_max=1.0 - 1e-8, grid_size=31):
    """
    Lightweight replacement for FastBetaMinimize.

    First evaluates `obj` on a grid, then refines around the best point
    using golden-section search.

    Returns
    -------
    value_star : float
        Minimum objective value.

    beta_star : float
        Approximate minimizer.
    """
    if not (0.0 < beta_min < beta_max < 1.0):
        raise ValueError("Require 0 < beta_min < beta_max < 1.")

    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")

    grid = np.linspace(beta_min, beta_max, grid_size)
    values = np.array([obj(float(b)) for b in grid])

    best_idx = int(np.argmin(values))

    if best_idx == 0:
        left, right = grid[0], grid[1]
    elif best_idx == grid_size - 1:
        left, right = grid[-2], grid[-1]
    else:
        left, right = grid[best_idx - 1], grid[best_idx + 1]

    beta_star, value_star = golden_section_minimize(obj, left, right)

    return value_star, beta_star


def golden_section_minimize(obj, left, right, tol=1e-12, max_iter=200):
    """
    Simple bounded scalar minimizer.
    """
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0

    x1 = right - inv_phi * (right - left)
    x2 = left + inv_phi * (right - left)

    f1 = obj(x1)
    f2 = obj(x2)

    for _ in range(max_iter):
        if abs(right - left) < tol:
            break

        if f1 > f2:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + inv_phi * (right - left)
            f2 = obj(x2)
        else:
            right = x2
            x2 = x1
            f2 = f1
            x1 = right - inv_phi * (right - left)
            f1 = obj(x1)

    beta_star = 0.5 * (left + right)
    value_star = obj(beta_star)

    return beta_star, value_star


# ---------------------------------------------------------------------
# epsilon(delta), optimized over beta
# ---------------------------------------------------------------------

def epsilon_for_delta_beta_exp_1d(
    n,
    r,
    kappa,
    sigma,
    eta,
    L,
    delta,
    beta_safety=1e-8,
    grid_size=31,
):
    """
    Equivalent of Mathematica:

        EpsilonForDeltaBetaExp1D[
            n, r, kappa, sigma, eta, L, delta,
            betaSafety_: 10^-8,
            gridSize_: 31
        ]
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must satisfy 0 < delta < 1.")

    v = v_n(n, r, kappa)
    delta_n = delta_n_exp_1d(n, r, kappa)
    tau = tau_tilde(n, r, kappa, sigma, eta)

    A = sigma**2 * r**2 + eta**2 * (v + r**2)

    c_psi = 2.0 * L
    c_mean = (L / 2.0) * ((v + r**2) / A) * delta_n**2
    c_delta = tau * math.log(1.0 / delta)

    def obj(beta):
        alpha = 1.0 + beta / tau
        psi = psi_beta(beta, tau)
        rdp = c_psi * psi + c_mean * alpha / (1.0 - beta)
        return rdp + c_delta / beta

    eps_star, beta_star = fast_beta_minimize(
        obj,
        beta_min=beta_safety,
        beta_max=1.0 - beta_safety,
        grid_size=grid_size,
    )

    alpha_star = 1.0 + beta_star / tau
    rdp_star = rdp_bound_beta_exp_1d(
        n, r, kappa, sigma, eta, L, beta_star
    )

    return {
        "Case": "Exp1D",
        "Epsilon": eps_star,
        "Delta": delta,
        "OptimalBeta": beta_star,
        "OptimalAlpha": alpha_star,
        "RDPAtOptimalBeta": rdp_star,
        "Vn": v,
        "DeltaN": delta_n,
        "TauTilde": tau,
        "AlphaRange": (1.0, 1.0 + 1.0 / tau),
        "BetaRange": (0.0, 1.0),
    }


# ---------------------------------------------------------------------
# delta(epsilon), optimized over beta
# ---------------------------------------------------------------------

def delta_for_epsilon_beta_exp_1d(
    n,
    r,
    kappa,
    sigma,
    eta,
    L,
    epsilon,
    beta_safety=1e-8,
    grid_size=31,
):
    """
    Equivalent of Mathematica:

        DeltaForEpsilonBetaExp1D[
            n, r, kappa, sigma, eta, L, epsilon,
            betaSafety_: 10^-8,
            gridSize_: 31
        ]
    """
    v = v_n(n, r, kappa)
    delta_n = delta_n_exp_1d(n, r, kappa)
    tau = tau_tilde(n, r, kappa, sigma, eta)

    A = sigma**2 * r**2 + eta**2 * (v + r**2)

    c_psi = 2.0 * L
    c_mean = (L / 2.0) * ((v + r**2) / A) * delta_n**2

    def obj(beta):
        alpha = 1.0 + beta / tau
        psi = psi_beta(beta, tau)
        rdp = c_psi * psi + c_mean * alpha / (1.0 - beta)
        return (beta / tau) * (rdp - epsilon)

    log_delta_star, beta_star = fast_beta_minimize(
        obj,
        beta_min=beta_safety,
        beta_max=1.0 - beta_safety,
        grid_size=grid_size,
    )

    alpha_star = 1.0 + beta_star / tau
    rdp_star = rdp_bound_beta_exp_1d(
        n, r, kappa, sigma, eta, L, beta_star
    )

    return {
        "Case": "Exp1D",
        "Epsilon": epsilon,
        "Delta": math.exp(log_delta_star),
        "LogDelta": log_delta_star,
        "OptimalBeta": beta_star,
        "OptimalAlpha": alpha_star,
        "RDPAtOptimalBeta": rdp_star,
        "Vn": v,
        "DeltaN": delta_n,
        "TauTilde": tau,
        "AlphaRange": (1.0, 1.0 + 1.0 / tau),
        "BetaRange": (0.0, 1.0),
    }

# ---------------------------------------------------------------------
# Generic alpha-optimized bound for posterior sample-path release
# ---------------------------------------------------------------------

def kappa_exp_kernel_unit_square(ell, diameter=math.sqrt(2.0)):
    """
    Lower kernel value on a unit square for the exponential kernel

        k(x,x') = exp(-||x-x'|| / ell).

    Since diam([0,1]^2)=sqrt(2), kappa = exp(-sqrt(2)/ell).
    """
    if ell <= 0:
        raise ValueError("ell must be positive.")
    return math.exp(-diameter / ell)


def delta_n_general(n, r, kappa, M_Y=1.0):
    """
    Mean-sensitivity bound used in the alpha-form RDP bound:

        Delta_n(r) = 2 M_Y (1 + sqrt(n-1)/r) sqrt(Phi_n(r)).
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    if r <= 0:
        raise ValueError("r must be positive.")
    if M_Y < 0:
        raise ValueError("M_Y must be nonnegative.")

    phi = phi_n(n, r, kappa)
    return np.sqrt(2.) * M_Y * (1.0 + math.sqrt(n - 1.0) / r) * math.sqrt(phi)


def psi_alpha(alpha, tau):
    """
    psi_alpha(tau) from the RDP bound:

        max{ 1/2 log(1+tau)
             - 1/[2(alpha-1)] log((1+alpha tau)/(1+tau)),
             -1/2 log(1+tau)
             - 1/[2(alpha-1)] log(1 - tau(alpha-1)) }.

    Requires alpha > 1 and 0 <= tau(alpha-1) < 1.
    """
    if alpha <= 1.0:
        raise ValueError("alpha must be greater than 1.")
    if tau < 0.0:
        raise ValueError("tau must be nonnegative.")

    u = alpha - 1.0

    if tau == 0.0:
        return 0.0

    if tau * u >= 1.0:
        return math.inf

    log1ptau = math.log1p(tau)

    term1 = (
        0.5 * log1ptau
        - (1.0 / (2.0 * u))
        * (math.log1p(alpha * tau) - log1ptau)
    )

    term2 = (
        -0.5 * log1ptau
        - (1.0 / (2.0 * u)) * math.log1p(-tau * u)
    )

    return max(term1, term2)


def rdp_bound_alpha_general(n, r, kappa, sigma, M_Y, alpha):
    """
    Alpha-form one-sample-path RDP bound:

        2 psi_alpha(V_n(r)/r^2)
        + alpha/2 * (V_n(r)+r^2)/(r^2-(alpha-1)V_n(r))
          * (Delta_n(r)/sigma)^2.

    The valid range is

        1 < alpha < 1 + r^2 / V_n(r).
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    v = v_n(n, r, kappa)
    if v <= 0.0:
        return 0.0

    alpha_max = 1.0 + r**2 / v
    if not (1.0 < alpha < alpha_max):
        return math.inf

    tau = v / r**2
    delta_n = delta_n_general(n, r, kappa, M_Y=M_Y)

    denom = r**2 - (alpha - 1.0) * v
    if denom <= 0.0:
        return math.inf

    covariance_term = 2.0 * psi_alpha(alpha, tau)
    mean_term = (
        (alpha / 2.0)
        * ((v + r**2) / denom)
        * (delta_n / sigma) ** 2
    )

    return covariance_term + mean_term


def fast_alpha_minimize(obj, alpha_min, alpha_max, grid_size=81):
    """
    Minimize a scalar objective over alpha in (alpha_min, alpha_max).
    A coarse grid is followed by golden-section refinement.
    """
    if not (alpha_min < alpha_max):
        raise ValueError("Require alpha_min < alpha_max.")
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")

    grid = np.linspace(alpha_min, alpha_max, grid_size)
    values = np.array([obj(float(a)) for a in grid])

    finite = np.isfinite(values)
    if not np.any(finite):
        return math.inf, math.nan

    # Penalize non-finite values for choosing a local bracket.
    values_for_argmin = values.copy()
    values_for_argmin[~finite] = math.inf
    best_idx = int(np.argmin(values_for_argmin))

    if best_idx == 0:
        left, right = grid[0], grid[1]
    elif best_idx == grid_size - 1:
        left, right = grid[-2], grid[-1]
    else:
        left, right = grid[best_idx - 1], grid[best_idx + 1]

    alpha_star, value_star = golden_section_minimize(obj, left, right)
    return value_star, alpha_star


def epsilon_for_delta_alpha_general(
    n,
    r,
    kappa,
    sigma,
    M_Y,
    L,
    delta,
    alpha_safety=1e-8,
    grid_size=81,
):
    """
    Convert the alpha-form RDP bound to (epsilon, delta)-DP and optimize
    over alpha:

        epsilon(alpha) = L * RDP_alpha
                         + log(1/delta)/(alpha-1).

    Valid alpha range:

        1 < alpha < 1 + r^2 / V_n(r).
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must satisfy 0 < delta < 1.")
    if L <= 0:
        raise ValueError("L must be positive.")

    v = v_n(n, r, kappa)
    delta_n_val = delta_n_general(n, r, kappa, M_Y=M_Y)

    if v <= 0.0:
        return {
            "Case": "AlphaGeneral",
            "Epsilon": 0.0,
            "Delta": delta,
            "OptimalAlpha": math.inf,
            "RDPAtOptimalAlpha": 0.0,
            "Vn": v,
            "PhiN": phi_n(n, r, kappa),
            "DeltaN": delta_n_val,
            "Kappa": kappa,
            "AlphaRange": (1.0, math.inf),
        }

    alpha_upper = 1.0 + r**2 / v
    alpha_min = 1.0 + alpha_safety
    alpha_max = alpha_upper - alpha_safety

    if alpha_max <= alpha_min:
        alpha_min = 1.0 + 0.1 * (alpha_upper - 1.0)
        alpha_max = 1.0 + 0.9 * (alpha_upper - 1.0)

    log_delta = math.log(1.0 / delta)

    def obj(alpha):
        rdp = rdp_bound_alpha_general(
            n=n,
            r=r,
            kappa=kappa,
            sigma=sigma,
            M_Y=M_Y,
            alpha=alpha,
        )
        return L * rdp + log_delta / (alpha - 1.0)

    eps_star, alpha_star = fast_alpha_minimize(
        obj,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        grid_size=grid_size,
    )

    rdp_star = rdp_bound_alpha_general(
        n=n,
        r=r,
        kappa=kappa,
        sigma=sigma,
        M_Y=M_Y,
        alpha=alpha_star,
    )

    return {
        "Case": "AlphaGeneral",
        "Epsilon": eps_star,
        "Delta": delta,
        "OptimalAlpha": alpha_star,
        "RDPAtOptimalAlpha": rdp_star,
        "Vn": v,
        "PhiN": phi_n(n, r, kappa),
        "DeltaN": delta_n_val,
        "Kappa": kappa,
        "AlphaRange": (1.0, alpha_upper),
    }


def epsilon_for_delta_exp_kernel_unit_square(
    n,
    ell,
    r,
    sigma,
    M_Y=1.0,
    L=1,
    delta=None,
    alpha_safety=1e-8,
    grid_size=81,
):
    """
    Convenience wrapper for the 2D unit-square exponential kernel

        k(x,x') = exp(-||x-x'|| / ell),

    for which kappa = exp(-sqrt(2)/ell). If delta is None, uses
    delta = n^{-1.1}.
    """
    if delta is None:
        delta = n ** (-1.1)

    kappa = kappa_exp_kernel_unit_square(ell)

    out = epsilon_for_delta_alpha_general(
        n=n,
        r=r,
        kappa=kappa,
        sigma=sigma,
        M_Y=M_Y,
        L=L,
        delta=delta,
        alpha_safety=alpha_safety,
        grid_size=grid_size,
    )

    out["Case"] = "ExpKernelUnitSquare"
    out["ell"] = ell
    return out
