"""Cross-fitted TDSC (DML-style: train AND target out-of-fold).

V-fold split. For fold v: nuisance networks are trained on the complement
I_{-v}, and the targeting step ALSO runs on I_{-v} (fitting and targeting are
both independent of the held-out fold). The held-out fold supplies evaluation:
plug-in curves, influence functions at the targeted fit, and an exact one-step
residual correction P_{n,v}[D] so the EIF equations hold exactly on held-out
data. By Proposition 2 of the paper this is a cross-fitted AIPCW one-step at
the targeted nuisances, satisfying the sample-splitting premise of TDA (T2) /
ADML (B1) without Donsker conditions.

Note: an earlier variant that targeted ON the held-out fold (CV-TMLE style,
solving 2K EIF equations over a p ~ 4000 gradient span with n/V ~ 200
observations) overfits the fold and fails badly -- see
results/mc_crossfit_foldtarget.npz. Do not target on small validation folds
with a high-dimensional targeting submodel.
"""
import numpy as np
import torch

from .influence import eif_matrix, plugin_estimates
from .model import (nuisances, person_bin_arrays, train_censnet,
                    train_dragonsurv, train_propnet)
from .targeting import _gradient_matrix, tda_target


def _subset(data, idx):
    return dict(X=data["X"][idx], A=data["A"][idx], Ttil=data["Ttil"][idx],
                Delta=data["Delta"][idx], K=data["K"])


def tda_crossfit(data, V=5, seed=0, epochs=500, ridge=1e-2, max_iter=50,
                 cols_out=None, cols_w=None):
    """Returns pooled psi (2K,) = plug-in + exact held-out residual correction,
    the plug-in-only variant, pooled IF matrix D (n, 2K), and diagnostics.

    cols_out / cols_w blind the outcome vs weight models to covariate subsets
    (nuisance-misspecification scenarios). Whenever a mask is set, the
    propensity comes from a standalone per-fold network on cols_w (the trunk
    head would inherit the outcome mask); with no masks the legacy trunk-head
    propensity is used, matching the well-specified stores."""
    n = data["X"].shape[0]
    K = data["K"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, V)

    D_all = np.empty((n, 2 * K))
    D_proj_all = np.empty((n, 2 * K))
    psi = np.zeros(2 * K)
    psi_plug = np.zeros(2 * K)
    diags = []
    for v, va_idx in enumerate(folds):
        tr_idx = np.setdiff1d(perm, va_idx, assume_unique=True)
        data_tr, data_va = _subset(data, tr_idx), _subset(data, va_idx)
        net = train_dragonsurv(data_tr, max_epochs=epochs, seed=seed * 100 + v,
                               cols=cols_out)
        censnet = train_censnet(data_tr, max_epochs=epochs, seed=seed * 100 + v + 50,
                                cols=cols_w)
        propnet = None
        if cols_out is not None or cols_w is not None:
            propnet = train_propnet(data_tr, max_epochs=epochs,
                                    seed=seed * 100 + v + 70, cols=cols_w)
        # target on the SAME out-of-fold data the nuisances were trained on
        _, _, g_tr, Sc_tr = nuisances(net, censnet, data_tr, propnet=propnet)
        res_tr = tda_target(net, data_tr, g_tr, Sc_tr, ridge=ridge, max_iter=max_iter)
        # evaluate on the held-out fold: plug-in, IFs, exact residual correction
        h1_va, h0_va, g_va, Sc_va = nuisances(net, censnet, data_va, propnet=propnet)
        at_risk, dN = person_bin_arrays(data_va["Ttil"], data_va["Delta"], K)
        D_va = eif_matrix(data_va["A"], at_risk, dN, h1_va, h0_va, g_va, Sc_va)
        # working-submodel (projected) EIFs on the held-out fold, using the
        # training-fold projection coefficients alpha
        with torch.no_grad():
            phi1_va, phi0_va = net.features(
                torch.as_tensor(data_va["X"], dtype=torch.float32))
        G_va = _gradient_matrix(data_va["A"], at_risk, dN, h1_va, h0_va,
                                phi1_va.numpy().astype(np.float64),
                                phi0_va.numpy().astype(np.float64))
        D_proj_va = G_va @ res_tr["alpha"]
        plug_v = plugin_estimates(h1_va, h0_va)
        resid_v = D_va.mean(axis=0)
        w = len(va_idx) / n
        psi_plug += w * plug_v
        psi += w * (plug_v + resid_v)          # one-step at targeted nuisances
        D_all[va_idx] = D_va
        D_proj_all[va_idx] = D_proj_va
        diags.append(dict(fold=v, iters=len(res_tr["history"]),
                          train_converged=bool(res_tr["converged"]),
                          heldout_max_resid=float(np.max(np.abs(resid_v)))))
    return dict(psi=psi, psi_plug=psi_plug, D=D_all, D_proj=D_proj_all, diags=diags)
