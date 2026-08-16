"""Efficient influence functions for treatment-specific discrete-time survival.

Target vector: psi = (S_1(t_1..t_K), S_0(t_1..t_K)) with
S_a(t) = E_X[ prod_{k<=t} (1 - h(k|a,X)) ].

Discrete-time EIF (Moore & van der Laan 2009 form) for S_a(t):

  D_a,t(O) = - sum_{k<=t} 1{A=a} / (g_a(X) * S_c(k-1|a,X))
                * S(t|a,X)/S(k|a,X) * (dN(k) - Y(k) h(k|a,X))
             + S(t|a,X) - S_a(t)

Contrast IFs are linear: D_benefit,t = D_1,t - D_0,t and
D_rmst = sum_t D_benefit,t (bin width 1).
"""
import numpy as np


def survival_from_hazards(h):
    """h (n,K) -> S (n,K) with S[:, k-1] = prod_{j<=k} (1 - h_j)."""
    return np.cumprod(1.0 - h, axis=1)


def eif_matrix(A, at_risk, dN, h1, h0, g, Sc_lag):
    """Stacked EIF matrix D (n, 2K): cols 0..K-1 are S_1(t_k), K..2K-1 are S_0(t_k).

    The plug-in centering term uses the empirical mean, so each column of D has
    exact mean equal to P_n[D] of the *uncentered* IF -- i.e. column means are
    the bias-correction terms driven to ~0 by targeting.
    """
    n, K = h1.shape
    S1 = survival_from_hazards(h1)
    S0 = survival_from_hazards(h0)
    D = np.empty((n, 2 * K))
    for a, (h, S, ga) in ((1, (h1, S1, g)), (0, (h0, S0, 1.0 - g))):
        ind = (A == a).astype(np.float64)
        M = dN - at_risk * h                              # martingale residuals
        inv = 1.0 / (np.clip(Sc_lag, 1e-3, None) * np.clip(S, 1e-3, None))
        C = np.cumsum((ind / ga)[:, None] * inv * M, axis=1)
        block = -S * C + S - S.mean(axis=0, keepdims=True)
        D[:, (0 if a == 1 else K):(K if a == 1 else 2 * K)] = block
    return D


def plugin_estimates(h1, h0):
    S1 = survival_from_hazards(h1).mean(axis=0)
    S0 = survival_from_hazards(h0).mean(axis=0)
    return np.concatenate([S1, S0])  # psi-hat, same layout as eif_matrix cols


def contrast_ifs(D):
    """Benefit-curve IFs (n,K) and RMST-difference IF (n,) from stacked D."""
    K = D.shape[1] // 2
    D_ben = D[:, :K] - D[:, K:]
    return D_ben, D_ben.sum(axis=1)


def contrast_estimates(psi):
    K = psi.shape[0] // 2
    benefit = psi[:K] - psi[K:]
    return benefit, benefit.sum()
