"""Estimators compared in the simulation study.

All return the stacked target vector psi (2K,) in the influence.py layout
(S1 at t_1..t_K then S0), plus, where applicable, an IF matrix for inference.
"""
import numpy as np

from .influence import eif_matrix, plugin_estimates, survival_from_hazards
from .model import person_bin_arrays, nuisances
from .targeting import tda_target


def naive_plugin(h1, h0):
    return plugin_estimates(h1, h0)


def one_step(data, h1, h0, g, Sc_lag):
    """Per-target AIPCW one-step: plug-in + P_n[D] at the initial fit."""
    at_risk, dN = person_bin_arrays(data["Ttil"], data["Delta"], data["K"])
    D = eif_matrix(data["A"], at_risk, dN, h1, h0, g, Sc_lag)
    psi = plugin_estimates(h1, h0) + D.mean(axis=0)
    return psi, D


def km_by_arm(data, adjusted=False, g=None, Sc_lag=None):
    """Kaplan-Meier per arm; adjusted=True uses IPTW x IPCW weights.

    Unadjusted KM is the confounded/informatively-censored baseline; the
    weighted version is consistent when g and S_c are correct.
    """
    A, K = data["A"], data["K"]
    at_risk, dN = person_bin_arrays(data["Ttil"], data["Delta"], K)
    psi = np.empty(2 * K)
    for a in (1, 0):
        ind = (A == a).astype(np.float64)
        if adjusted:
            w = (ind / (g if a == 1 else 1.0 - g))[:, None] / Sc_lag
        else:
            w = ind[:, None] * np.ones_like(at_risk)
        num = (w * dN).sum(axis=0)
        den = np.clip((w * at_risk).sum(axis=0), 1e-8, None)
        S = np.cumprod(1.0 - num / den)
        psi[(0 if a == 1 else K):(K if a == 1 else 2 * K)] = S
    return psi


def _pava_nonincreasing(y):
    """Project a sequence onto non-increasing monotone sequences (PAVA)."""
    vals, wts = [], []
    for v in -np.asarray(y, dtype=float):
        vals.append(v)
        wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2 = vals.pop(), wts.pop()
            v1, w1 = vals.pop(), wts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
    return -np.concatenate([np.full(int(w), v) for v, w in zip(vals, wts)])


def isotonize(psi):
    """Project each arm's curve onto monotone non-increasing, clipped to [0,1].

    Baseline isolating how much of TDSC's advantage over the per-target
    one-step is explained by monotonicity alone (model-space structure that
    the one-step lacks and the targeted plug-in has by construction).
    """
    K = psi.shape[0] // 2
    out = np.empty_like(psi)
    out[:K] = np.clip(_pava_nonincreasing(psi[:K]), 0.0, 1.0)
    out[K:] = np.clip(_pava_nonincreasing(psi[K:]), 0.0, 1.0)
    return out


def tda(net, censnet, data, propnet=None, ridge=1e-2, max_iter=50,
        residual_correction=False, verbose=False):
    """Full TDSC estimator: universal targeting, then plug-in + final EIFs.

    residual_correction=True adds the leftover P_n[D_k] one-step-style to any
    coordinate whose EIF equation was not solved to tolerance, restoring exact
    double robustness of the point estimate when targeting stalls.
    """
    h1_init, h0_init, g, Sc_lag = nuisances(net, censnet, data, propnet=propnet)
    res = tda_target(net, data, g, Sc_lag, ridge=ridge, max_iter=max_iter,
                     verbose=verbose)
    psi = plugin_estimates(res["h1"], res["h0"])
    if residual_correction:
        unsolved = np.abs(res["final_pnd"]) > res["final_tol"]
        psi = psi + unsolved * res["final_pnd"]
    return dict(psi=psi, D=res["D"], D_proj=res["D_proj"], history=res["history"],
                converged=res["converged"], final_pnd=res["final_pnd"],
                final_tol=res["final_tol"], final_pnd_proj=res["final_pnd_proj"],
                init=dict(h1=h1_init, h0=h0_init, g=g, Sc_lag=Sc_lag))
