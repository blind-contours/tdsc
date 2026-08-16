"""Cross-fitted TDSC (CV-TMLE structure).

V-fold split. For fold v: nuisance networks are trained on the complement
I_{-v}; the targeting step then solves the EIF equations *on the held-out
fold* I_v (gradients, influence functions, and the plug-in all evaluated on
data independent of the initial fit). Fold plug-ins are averaged with weights
|I_v|/n; the pooled per-observation influence matrix (each row from its own
fold's targeted fit) supplies variance and bands.

This satisfies the sample-splitting premise of TDA condition (T2)/ADML (B1)
without Donsker conditions on the network class.
"""
import numpy as np

from .influence import plugin_estimates
from .model import nuisances, train_censnet, train_dragonsurv
from .targeting import tda_target


def _subset(data, idx):
    return dict(X=data["X"][idx], A=data["A"][idx], Ttil=data["Ttil"][idx],
                Delta=data["Delta"][idx], K=data["K"])


def tda_crossfit(data, V=5, seed=0, epochs=500, ridge=1e-2, max_iter=50):
    """Returns pooled psi (2K,), top-up variant, pooled IF matrix D (n, 2K),
    and per-fold diagnostics."""
    n = data["X"].shape[0]
    K = data["K"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, V)

    D_all = np.empty((n, 2 * K))
    psi = np.zeros(2 * K)
    psi_topup = np.zeros(2 * K)
    diags = []
    for v, va_idx in enumerate(folds):
        tr_idx = np.setdiff1d(perm, va_idx, assume_unique=True)
        data_tr, data_va = _subset(data, tr_idx), _subset(data, va_idx)
        net = train_dragonsurv(data_tr, max_epochs=epochs, seed=seed * 100 + v)
        censnet = train_censnet(data_tr, max_epochs=epochs, seed=seed * 100 + v + 50)
        _, _, g_va, Sc_va = nuisances(net, censnet, data_va)
        res = tda_target(net, data_va, g_va, Sc_va, ridge=ridge, max_iter=max_iter)
        psi_v = plugin_estimates(res["h1"], res["h0"])
        unsolved = np.abs(res["final_pnd"]) > res["final_tol"]
        w = len(va_idx) / n
        psi += w * psi_v
        psi_topup += w * (psi_v + unsolved * res["final_pnd"])
        D_all[va_idx] = res["D"]
        diags.append(dict(fold=v, iters=len(res["history"]),
                          converged=bool(res["converged"]),
                          unsolved=int(unsolved.sum())))
    return dict(psi=psi, psi_topup=psi_topup, D=D_all, diags=diags)
