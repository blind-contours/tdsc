"""Output-space survival TMLE (Moore & van der Laan style), per-timepoint.

The classical comparator TDSC replaces: for each arm a and horizon t, fluctuate
the fitted discrete-time hazards through the one-dimensional logistic submodel

    logit h_eps(k | a, x) = logit h(k | a, x) + eps * H_t(k, x),   k <= t,
    H_t(k, x) = - S(t|a,x) / ( S(k|a,x) g_a(x) S_c(k-1|a,x) ),

solving the score equation (the martingale component of the EIF) by Newton
steps, iterating until P_n D_{a,t} ~ 0. Each of the 2K coordinates gets its own
fluctuated copy of the hazards -- the per-coordinate bookkeeping (and the lack
of coherence across t) that motivates TDSC's single universal update.

Uses the same initial nuisances as the one-step and TDSC, so the three-way
comparison isolates (i) one-step vs TMLE in output space and (ii) output-space
vs weight-space TMLE.
"""
import numpy as np

from .influence import survival_from_hazards
from .model import person_bin_arrays


def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def ostmle_curve(data, h1, h0, g, Sc_lag, max_iter=25, tol_scale=True):
    """Per-timepoint output-space TMLE for all 2K targets.

    Returns psi (2K,) and D (n, 2K) evaluated at each target's fluctuated fit.
    """
    A, K = data["A"], data["K"]
    n = h1.shape[0]
    at_risk, dN = person_bin_arrays(data["Ttil"], data["Delta"], K)
    psi = np.empty(2 * K)
    D = np.empty((n, 2 * K))
    for a, h_base, ga in ((1, h1, g), (0, h0, 1.0 - g)):
        ind = (A == a).astype(np.float64)
        w_g = ind / ga
        for t in range(1, K + 1):
            h = h_base.copy()
            for _ in range(max_iter):
                S = survival_from_hazards(h)
                # H (n, t): clever covariate for bins 1..t
                H = -(S[:, t - 1:t] / np.clip(S[:, :t], 1e-3, None)) \
                    / (np.clip(Sc_lag[:, :t], 1e-3, None) * ga[:, None])
                resid = at_risk[:, :t] * (dN[:, :t] - h[:, :t])
                score = float(np.sum(ind[:, None] * H * resid))
                sd_mart = np.std(np.sum(ind[:, None] * H * resid, axis=1), ddof=1)
                tol = sd_mart / (np.sqrt(n) * np.log(n)) if tol_scale else 1e-5
                if abs(score / n) <= max(tol, 1e-10):
                    break
                info = float(np.sum(ind[:, None] * at_risk[:, :t] * H ** 2
                                    * h[:, :t] * (1.0 - h[:, :t])))
                if info < 1e-12:
                    break
                eps = np.clip(score / info, -5.0, 5.0)
                h[:, :t] = _expit(_logit(h[:, :t]) + eps * H)
            S = survival_from_hazards(h)
            St = S[:, t - 1]
            mart = np.cumsum(
                (w_g[:, None] / (np.clip(Sc_lag[:, :t], 1e-3, None)
                                 * np.clip(S[:, :t], 1e-3, None)))
                * (dN[:, :t] - at_risk[:, :t] * h[:, :t]), axis=1)[:, -1]
            col = (0 if a == 1 else K) + (t - 1)
            psi[col] = St.mean()
            D[:, col] = -St * mart + St - St.mean()
    return psi, D
